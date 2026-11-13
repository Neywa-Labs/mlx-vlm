import json
from dataclasses import asdict

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from mlx_vlm.models.gemma4.config import ModelConfig, TextConfig, VisionConfig
from mlx_vlm.models.gemma4.gemma4 import Model
from mlx_vlm.models.gemma4.language import LanguageModel
from mlx_vlm.quant_utils import quantize_model
from mlx_vlm.utils import load_model, skip_multimodal_module


def text_config(tied):
    return TextConfig(
        hidden_size=32,
        num_hidden_layers=1,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        global_head_dim=8,
        vocab_size=64,
        vocab_size_per_layer_input=64,
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=0,
        tie_word_embeddings=tied,
        final_logit_softcapping=None,
    )


@pytest.mark.parametrize("tied", [True, False])
def test_output_projection_uses_configured_weights(tied):
    model = LanguageModel(text_config(tied))
    hidden = mx.ones((1, 1, 32))
    model.model.embed_tokens.weight = mx.ones((64, 32))
    if not tied:
        model.lm_head.weight = mx.full((64, 32), 2.0)
    logits = model.logits_from_hidden(hidden)
    expected = 32.0 if tied else 64.0
    assert mx.allclose(logits, mx.full((1, 1, 64), expected)).item()


@pytest.mark.parametrize(
    "prefix",
    [
        "lm_head",
        "model.lm_head",
        "language_model.lm_head",
        "model.language_model.lm_head",
    ],
)
def test_sanitize_preserves_untied_head_and_quantization_fields(prefix):
    config = ModelConfig(
        text_config=text_config(False),
        vision_config=VisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            position_embedding_size=16,
        ),
        audio_config=None,
    )
    model = Model(config)
    weights = {
        f"{prefix}.{name}": mx.ones((2, 2)) for name in ("weight", "scales", "biases")
    }
    sanitized = model.sanitize(weights)
    assert set(sanitized) == {
        f"language_model.lm_head.{name}" for name in ("weight", "scales", "biases")
    }
    assert set(model.sanitize(sanitized)) == set(sanitized)


@pytest.mark.parametrize("quantized", [False, True])
def test_untied_checkpoint_strict_load(quantized):
    config = ModelConfig(
        text_config=text_config(False),
        vision_config=VisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            position_embedding_size=16,
        ),
        audio_config=None,
    )
    model = Model(config)
    if quantized:
        nn.quantize(
            model,
            group_size=32,
            bits=4,
            class_predicate=lambda path, _: path == "language_model.lm_head",
        )
    checkpoint = dict(tree_flatten(model.parameters()))
    suffixes = ("weight", "scales", "biases") if quantized else ("weight",)
    for suffix in suffixes:
        checkpoint[f"lm_head.{suffix}"] = checkpoint.pop(
            f"language_model.lm_head.{suffix}"
        )
    model.load_weights(list(model.sanitize(checkpoint).items()), strict=True)


@pytest.mark.parametrize(
    "tied,quantized", [(True, False), (False, False), (False, True)]
)
def test_checkpoint_load_model_and_forward(tmp_path, tied, quantized):
    mx.random.seed(791)
    config = ModelConfig(
        text_config=text_config(tied),
        tie_word_embeddings=tied,
        image_token_id=60,
        vision_config=VisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            position_embedding_size=16,
            patch_size=2,
            pooling_kernel_size=2,
            default_output_length=1,
        ),
        audio_config=None,
    )
    reference = Model(config)
    if quantized:
        nn.quantize(
            reference,
            group_size=32,
            bits=4,
            class_predicate=lambda path, _: path == "language_model.lm_head",
        )
    checkpoint = dict(tree_flatten(reference.parameters()))
    if not tied:
        suffixes = ("weight", "scales", "biases") if quantized else ("weight",)
        for suffix in suffixes:
            checkpoint[f"lm_head.{suffix}"] = checkpoint.pop(
                f"language_model.lm_head.{suffix}"
            )
    if not quantized:
        checkpoint = {
            (
                ("model." + key.replace("language_model.model.", "language_model."))
                if not key.startswith("lm_head.")
                else key
            ): value
            for key, value in checkpoint.items()
        }
    raw_config = asdict(config)
    if quantized:
        raw_config["quantization"] = {"group_size": 32, "bits": 4}
    (tmp_path / "config.json").write_text(json.dumps(raw_config))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), checkpoint)
    loaded = load_model(tmp_path, strict=True)
    tokens = mx.array([[1, 5, 9]])
    assert mx.allclose(
        loaded(tokens).logits, reference(tokens).logits, atol=1e-5, rtol=1e-5
    ).item()

    # Exercise the image encoder and fusion after loading, including the untied
    # projection. Two different synthetic images must affect the logits.
    image_tokens = mx.array([[1, 60, 9]])
    pixels = mx.arange(48).reshape((1, 3, 4, 4)).astype(mx.float32) / 47
    actual_image_logits = loaded(image_tokens, pixel_values=pixels).logits
    expected_image_logits = reference(image_tokens, pixel_values=pixels).logits
    assert mx.allclose(
        actual_image_logits, expected_image_logits, atol=1e-5, rtol=1e-5
    ).item()
    other_logits = loaded(image_tokens, pixel_values=1 - pixels).logits
    assert not mx.allclose(
        actual_image_logits, other_logits, atol=1e-5, rtol=1e-5
    ).item()


@pytest.mark.parametrize("bits", [4, 8])
def test_untied_output_head_quantization_round_trip(tmp_path, bits):
    mx.random.seed(987)
    config = ModelConfig(
        text_config=text_config(False),
        tie_word_embeddings=False,
        vision_config=VisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            position_embedding_size=16,
        ),
        audio_config=None,
    )
    model = Model(config)
    quant_predicate = model.quant_predicate
    model, raw_config = quantize_model(
        model,
        asdict(config),
        group_size=32,
        bits=bits,
        quant_predicate=lambda path, module: (
            not skip_multimodal_module(path) and quant_predicate(path, module)
        ),
    )
    weights = dict(tree_flatten(model.parameters()))
    assert all(
        f"language_model.lm_head.{suffix}" in weights
        for suffix in ("weight", "scales", "biases")
    )
    (tmp_path / "config.json").write_text(json.dumps(raw_config))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    reloaded = load_model(tmp_path, strict=True)
    tokens = mx.array([[1, 5, 9]])
    assert mx.allclose(
        reloaded(tokens).logits, model(tokens).logits, atol=1e-5, rtol=1e-5
    ).item()
