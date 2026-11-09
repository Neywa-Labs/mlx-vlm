import json
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from mlx_vlm.models.ricursive import Model, ModelConfig
from mlx_vlm.models.ricursive.language import MLP, Attention
from mlx_vlm.quant_utils import quantize_model
from mlx_vlm.utils import load, load_model, save_weights


def small_config(**kwargs):
    cases = json.loads(Path(__file__).with_name("model_cases.json").read_text())
    fields = next(
        case["config"] for case in cases["cases"] if case["id"] == "ricursive"
    )
    fields.update(kwargs)
    return ModelConfig(**fields)


def test_value_residual_is_blended_before_projection_and_gate():
    config = small_config(
        hidden_size=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=2,
        num_hidden_layers=1,
        layer_types=["full_attention"],
    )
    attn = Attention(config, 0)
    attn.v_proj.weight = mx.array([[2.0, 1.0], [-1.0, 3.0]])
    attn.o_proj.weight = mx.eye(2)
    attn.g_proj.weight = mx.zeros((2, 2))
    attn.value_res_lambda = mx.array([np.log(3)], mx.float32)
    x = mx.array([[[1.0, -2.0]]])
    seed = mx.array([[[5.0, 6.0]]])
    # sigmoid(log(3)) = 0.75: blend = [2, 0], project = [4, -2], gate = 0.5.
    np.testing.assert_allclose(
        np.array(attn(x, value_residual=seed)), [[[2.0, -1.0]]], atol=1e-6
    )
    assert not hasattr(attn, "rope")


def test_ffn_clips_only_the_upper_gate_bound():
    mlp = MLP(small_config(hidden_size=2, intermediate_size=2, ffn_clamp_threshold=2.0))
    mlp.gate_proj.weight = mx.eye(2)
    mlp.up_proj.weight = mx.array([[2.0, 0.0], [0.0, 2.0]])
    mlp.down_proj.weight = mx.eye(2)
    x = mx.array([[[-4.0, 4.0]]])
    gate = np.array([-4.0, 2.0])
    expected = gate / (1 + np.exp(-gate)) * [-2.0, 2.0]
    np.testing.assert_allclose(np.array(mlp(x))[0, 0], expected, atol=1e-6)


@pytest.mark.parametrize("tie_embeddings", [False, True])
def test_checkpoint_load_quantization_and_sanitize_roundtrip(tmp_path, tie_embeddings):
    config = small_config(tie_word_embeddings=tie_embeddings)
    model = Model(config)
    upstream = {
        k.removeprefix("language_model."): v
        for k, v in tree_flatten(model.parameters())
    }
    converted = model.sanitize(upstream)
    repeated = model.sanitize(converted)
    assert converted.keys() == repeated.keys()
    for key in converted:
        assert converted[key] is repeated[key]
    model.load_weights(list(converted.items()), strict=True)

    (tmp_path / "config.json").write_text(json.dumps(asdict(config)))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), upstream)
    loaded = load_model(tmp_path)
    ids = mx.array([[1, 2, 3, 4, 5, 6]])
    expected = model(ids).logits
    np.testing.assert_allclose(
        np.array(loaded(ids).logits), np.array(expected), atol=1e-6
    )
    np.testing.assert_allclose(
        np.array(loaded.language_model(ids, logits_to_keep=1).logits),
        np.array(expected[:, -1:]),
        atol=1e-6,
    )
    with pytest.raises(ValueError, match="nonnegative"):
        loaded.language_model(ids, logits_to_keep=-1)

    quantized, quant_config = quantize_model(loaded, asdict(config), 32, 4)
    destination = tmp_path / "quantized"
    save_weights(destination, quantized)
    (destination / "config.json").write_text(json.dumps(quant_config))
    reloaded = load_model(destination)
    np.testing.assert_allclose(
        np.array(reloaded(ids).logits), np.array(quantized(ids).logits), atol=1e-6
    )


def test_tokenizer_loads_without_remote_code_and_preserves_chat_template(tmp_path):
    config = small_config()
    save_weights(tmp_path, Model(config))
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "auto_map": {"AutoConfig": "missing.Configuration"},
                "eos_token_id": [2, 3],
            }
        )
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"<unk>": 0, "hello": 1, "<end>": 2, "<eot>": 3})
        ),
        unk_token="<unk>",
        eos_token="<end>",
        pad_token="<end>",
        chat_template="{% for message in messages %}{{ message.content }}{% endfor %}",
    )
    tokenizer.save_pretrained(tmp_path)
    _, processor = load(tmp_path, trust_remote_code=False)
    assert (
        processor.apply_chat_template(
            [{"role": "user", "content": "hello"}], tokenize=False
        )
        == "hello"
    )
    assert processor.encode("hello") == [1]
    assert processor.stopping_criteria(2)
    assert processor.stopping_criteria(3)


@pytest.mark.parametrize(
    "fields",
    [
        {"layer_types": ["full_attention"]},
        {"layer_types": ["unknown"] * 5},
        {"num_attention_heads": 3},
        {"sliding_window": 0},
        {"hidden_act": "relu"},
    ],
)
def test_invalid_config(fields):
    with pytest.raises(ValueError):
        small_config(**fields)
