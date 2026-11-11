"""Cosmos3-Edge (reasoner): weight loading, parity with transformers, and video inputs.

The parity tests build a tiny random checkpoint in NVIDIA's original layout (raw tensor names, generator tensors
that must be dropped, subfolder shards) and compare against transformers' Cosmos3EdgeForConditionalGeneration in
fp32 on the MLX CPU device. They are skipped when torch or a transformers build with cosmos3_edge is missing.
"""

import importlib
import json
import os

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_vlm.models.cosmos3_edge import Model, ModelConfig

IMG, VID, VS, VE = 19, 18, 20, 21
TOL = 2e-4


def _raw_names(converted):
    """mlx-vlm names -> names in NVIDIA's original transformer/ shards."""
    out = []
    for key in converted:
        if key.startswith("language_model.model."):
            key = key[len("language_model.model.") :]
            for a, b in (
                (".self_attn.q_proj.", ".self_attn.to_q."),
                (".self_attn.k_proj.", ".self_attn.to_k."),
                (".self_attn.v_proj.", ".self_attn.to_v."),
                (".self_attn.o_proj.", ".self_attn.to_out."),
                (".mlp.fc1.", ".mlp.up_proj."),
                (".mlp.fc2.", ".mlp.down_proj."),
            ):
                key = key.replace(a, b)
        elif key.startswith("vision_tower."):
            key = "model.visual." + key[len("vision_tower.") :]
        elif key.startswith("multi_modal_projector."):
            key = "model.projector." + key[len("multi_modal_projector.") :]
        elif key.startswith("language_model.lm_head."):
            key = key[len("language_model.") :]
        out.append(key)
    return out


GENERATOR_KEYS = [
    "layers.0.mlp_moe_gen.up_proj.weight",
    "layers.0.input_layernorm_moe_gen.weight",
    "layers.1.self_attn.add_q_proj.weight",
    "layers.1.self_attn.to_add_out.weight",
    "layers.1.self_attn.k_norm_und_for_gen.weight",
    "time_embedder.linear_1.weight",
    "proj_in.weight",
    "norm_moe_gen.weight",
    "action_modality_embed",
]


@pytest.fixture(scope="module")
def tiny_model():
    cases = json.load(open(os.path.join(os.path.dirname(__file__), "model_cases.json")))
    cases = cases if isinstance(cases, list) else cases["cases"]
    config = next(c for c in cases if c["id"] == "TestModels.cosmos3_edge")["config"]
    return Model(ModelConfig.from_dict(config))


def test_sanitize_original_layout_and_idempotent(tiny_model):
    converted = dict(tree_flatten(tiny_model.parameters()))
    raw = dict(zip(_raw_names(converted), converted.values()))
    raw.update({k: mx.zeros((1,)) for k in GENERATOR_KEYS})

    once = tiny_model.sanitize(raw)
    assert set(once) == set(converted)
    assert set(tiny_model.sanitize(once)) == set(converted)
    assert set(tiny_model.sanitize(converted)) == set(converted)


# ---------------------------------------------------------------- parity with transformers


@pytest.fixture
def cpu(monkeypatch):
    """fp32 parity runs on the CPU: Metal fp32 matmuls are not bit-faithful enough for a 2e-4 tolerance."""
    ar = importlib.import_module("mlx_vlm.generate.ar")
    monkeypatch.setattr(ar, "generation_stream", mx.default_stream(mx.cpu))
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Cosmos3EdgeForConditionalGeneration"):
        pytest.skip("transformers without cosmos3_edge")

    cfg = transformers.Cosmos3EdgeConfig(
        text_config=dict(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=160,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 1e8,
                "mrope_section": [4, 2, 2],
            },
        ),
        vision_config=dict(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            patch_size=16,
            num_patches=256,
            spatial_merge_size=2,
        ),
        projector_hidden_size=96,
    )
    torch.manual_seed(0)
    ref = transformers.Cosmos3EdgeForConditionalGeneration(cfg).float().eval()
    with torch.no_grad():
        for name, p in ref.named_parameters():
            is_norm = "norm" in name
            p.copy_(
                torch.randn_like(p) * (0.5 if is_norm else 0.15)
                + (1.0 if is_norm and "weight" in name else 0.0)
            )

    # Write it in the original layout: language tensors with raw names next to generator tensors, vision apart.
    tmp = tmp_path_factory.mktemp("cosmos3_edge")
    lm, vis = {}, {}
    for key, value in ref.state_dict().items():
        if key.startswith("model.language_model."):
            key = key[len("model.language_model.") :]
            for a, b in (
                (".q_proj.", ".to_q."),
                (".k_proj.", ".to_k."),
                (".v_proj.", ".to_v."),
                (".o_proj.", ".to_out."),
                (".mlp.fc1.", ".mlp.up_proj."),
                (".mlp.fc2.", ".mlp.down_proj."),
            ):
                key = key.replace(a, b)
            lm[key] = value.contiguous()
        elif key.startswith("model."):
            vis[key] = value.contiguous()
        else:
            lm[key] = value.contiguous()
    gen = {
        k: torch.randn(4, 4)
        for k in GENERATOR_KEYS
        if "norm" not in k and "embed" not in k
    }
    gen.update(
        {
            f"layers.{i}.self_attn.k_norm_und_for_gen.weight": torch.randn(16)
            for i in range(3)
        }
    )
    (tmp / "transformer").mkdir()
    (tmp / "vision_encoder").mkdir()
    st.save_file(
        {**lm, **gen}, str(tmp / "transformer" / "diffusion_pytorch_model.safetensors")
    )
    st.save_file(vis, str(tmp / "vision_encoder" / "model.safetensors"))
    weight_map = {
        k: "transformer/diffusion_pytorch_model.safetensors" for k in {**lm, **gen}
    }
    weight_map.update({k: "vision_encoder/model.safetensors" for k in vis})
    json.dump(
        {"metadata": {}, "weight_map": weight_map},
        open(tmp / "model.safetensors.index.json", "w"),
    )
    config = cfg.to_dict()
    config["architectures"] = ["Cosmos3EdgeForConditionalGeneration"]
    json.dump(config, open(tmp / "config.json", "w"))

    from mlx_vlm.utils import load_model

    with mx.stream(mx.cpu):
        model = load_model(tmp)
    return torch, ref, model


def _case(name):
    rng = np.random.default_rng(0)

    def txt(n):
        return rng.integers(30, 256, n).tolist()

    def block(n, kind):
        return [VS] + [kind] * n + [VE]

    c = {}
    if name == "text":
        ids = txt(23)
    elif (
        name == "two_images"
    ):  # the second is smaller than the 16x16 position grid (antialiased resize path)
        ids = txt(4) + block(96, IMG) + txt(3) + block(24, IMG) + txt(6)
        c["image_grid_thw"] = [[1, 16, 24], [1, 8, 12]]
    elif name == "video":  # 3 frames with timestamp text between them
        ids = txt(4) + sum([txt(4) + block(20, VID) for _ in range(3)], []) + txt(5)
        c["video_grid_thw"] = [[3, 8, 10]]
    c["input_ids"] = np.array([ids], dtype=np.int64)
    for grid, pix in (
        ("image_grid_thw", "pixel_values"),
        ("video_grid_thw", "pixel_values_videos"),
    ):
        if grid in c:
            c[grid] = np.array(c[grid], dtype=np.int64)
            c[pix] = rng.standard_normal(
                (int(np.prod(c[grid], axis=1).sum()), 768)
            ).astype(np.float32)
    return c


def _torch_inputs(torch, c):
    types = np.zeros_like(c["input_ids"])
    types[c["input_ids"] == IMG] = 1
    types[c["input_ids"] == VID] = 2
    kw = {k: torch.tensor(v) for k, v in c.items()}
    kw["mm_token_type_ids"] = torch.tensor(types)
    return kw


def _mlx_inputs(c):
    return {
        k: mx.array(v) for k, v in c.items() if k not in ("input_ids", "pixel_values")
    }


def _reset(model):
    model.language_model._position_ids = model.language_model._rope_deltas = None


@pytest.mark.parametrize("name", ["text", "two_images", "video"])
def test_logits_and_greedy_match_transformers(reference, cpu, name):
    from mlx_vlm.generate.ar import generate_step

    torch, ref, model = reference
    c = _case(name)
    with torch.no_grad():
        ref.model.rope_deltas = None
        want_logits = ref(**_torch_inputs(torch, c)).logits[0].numpy()
        ref.model.rope_deltas = None
        want = ref.generate(
            **_torch_inputs(torch, c),
            max_new_tokens=8,
            do_sample=False,
            eos_token_id=None,
            pad_token_id=0,
        )[0, c["input_ids"].shape[1] :].tolist()

    ids = mx.array(c["input_ids"])
    pixels = mx.array(c["pixel_values"]) if "pixel_values" in c else None
    with mx.stream(mx.cpu):
        _reset(model)
        logits = np.array(
            model(ids, pixels, **_mlx_inputs(c)).logits.astype(mx.float32)
        )[0]
        assert np.abs(logits - want_logits).max() < TOL
        for step in (None, 17):  # whole prompt, and chunked prefill
            _reset(model)
            got = [
                int(t)
                for t, _ in generate_step(
                    ids,
                    model,
                    pixels,
                    None,
                    max_tokens=8,
                    temperature=0.0,
                    prefill_step_size=step,
                    **_mlx_inputs(c),
                )
            ]
            assert got == want, f"prefill_step_size={step}"


def test_generator_tensors_dropped(reference):
    _, ref, model = reference
    assert sum(p.numel() for p in ref.parameters()) == sum(
        v.size for _, v in tree_flatten(model.parameters())
    )


# ---------------------------------------------------------------- video inputs


def test_video_inputs_match_transformers(tmp_path):
    """mlx-vlm's video path must give the same frames (NVIDIA's fps rule), "<t seconds>" markers and pixels as
    transformers' processor run on the full clip."""
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    hf = pytest.importorskip("transformers.models.cosmos3_edge.processing_cosmos3_edge")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from transformers.models.cosmos3_edge.image_processing_cosmos3_edge import (
        Cosmos3EdgeImageProcessor,
    )
    from transformers.models.cosmos3_edge.video_processing_cosmos3_edge import (
        Cosmos3EdgeVideoProcessor,
    )
    from transformers.video_utils import VideoMetadata

    from mlx_vlm.models.cosmos3_edge.processing_cosmos3_edge import Cosmos3EdgeProcessor
    from mlx_vlm.utils import prepare_inputs

    clip = str(tmp_path / "clip.mp4")
    writer = cv2.VideoWriter(clip, cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for i in range(90):
        frame = np.zeros((240, 320, 3), np.uint8)
        frame[:, (i * 3) % 300 : (i * 3) % 300 + 20] = 255
        writer.write(frame)
    writer.release()

    special = ["<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>"]
    words = (
        ["[UNK]", "[PAD]"]
        + special
        + list("<>.0123456789")
        + ["seconds", "what", "happens"]
    )
    tk = Tokenizer(
        models.WordLevel({t: i for i, t in enumerate(words)}, unk_token="[UNK]")
    )
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        unk_token="[UNK]",
        pad_token="[PAD]",
        additional_special_tokens=special,
    )

    def make(cls):
        p = cls(
            image_processor=Cosmos3EdgeImageProcessor(),
            tokenizer=tok,
            video_processor=Cosmos3EdgeVideoProcessor(return_metadata=True),
        )
        if cls is Cosmos3EdgeProcessor:
            p.video_processor.sample_frames_in_loader = (
                True  # what from_pretrained sets
            )
        return p

    # Reference frames come from the same decoder mlx-vlm uses (OpenCV), so only the frame choice can differ.
    cap, frames = cv2.VideoCapture(clip), []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    frames = np.stack(frames).transpose(0, 3, 1, 2)

    prompt = "what happens <|vision_start|><|video_pad|><|vision_end|>"
    for fps in (4, 3, 1, None):
        kw = {} if fps is None else {"fps": fps}
        want = make(hf.Cosmos3EdgeProcessor)(
            text=[prompt],
            videos=[frames],
            video_metadata=[VideoMetadata(total_num_frames=90, fps=30.0)],
            return_tensors="np",
            return_metadata=True,
            **kw,
        )
        got = prepare_inputs(
            make(Cosmos3EdgeProcessor), videos=[clip], prompts=[prompt], **kw
        )
        assert (
            np.array(got["video_grid_thw"]).tolist() == want["video_grid_thw"].tolist()
        ), fps
        np.testing.assert_array_equal(np.array(got["input_ids"]), want["input_ids"])
        np.testing.assert_allclose(
            np.array(got["pixel_values_videos"]), want["pixel_values_videos"], atol=1e-5
        )
