# Copyright © 2026 Neywa Labs Inc.

"""BF16 regression values from the upstream PyTorch implementation."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_vlm.models.ricursive.config import ModelConfig
from mlx_vlm.models.ricursive.language import (
    MLP,
    Attention,
    RicursiveModel,
    RMSNorm,
    RotaryEmbedding,
)


@pytest.fixture
def config():
    return ModelConfig(
        vocab_size=8,
        hidden_size=8,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        layer_types=["full_attention"],
        use_valres=True,
        embed_scale=71.55417527999327,
    )


@pytest.fixture
def values():
    return mx.array(
        [[[-3.25, -0.75, -0.125, 0.015625, 0.75, 1.25, 2.5, 5.5]]],
        dtype=mx.bfloat16,
    )


def test_embedding_scale_rounds_after_multiplication(config, values):
    model = RicursiveModel(config)
    expected = mx.array(
        [[[-233, -53.75, -8.9375, 1.1171875, 53.75, 89.5, 179, 394]]],
        dtype=mx.bfloat16,
    )
    assert mx.array_equal(model._embed(None, values), expected)


def test_norm_rounds_before_weight_multiplication(values):
    norm = RMSNorm(8, eps=1e-6)
    norm.weight = mx.array(
        [0.8125, 1.125, -0.625, 2.375, 0.9375, 0.5625, -1.25, 1.625],
        dtype=mx.bfloat16,
    )
    expected = mx.array(
        [
            -1.0625,
            -0.337890625,
            0.03125,
            0.01483154296875,
            0.28125,
            0.28125,
            -1.25,
            3.578125,
        ],
        dtype=mx.bfloat16,
    )
    assert mx.array_equal(norm(values).reshape(-1), expected)


@pytest.mark.parametrize("width", [128, 5120])
def test_norm_matches_tokenwise_execution(width):
    mx.random.seed(19)
    norm = RMSNorm(width, eps=1e-6)
    norm.weight = mx.random.normal((width,)).astype(mx.bfloat16)
    x = (mx.random.normal((2, 5, width)) * 0.005).astype(mx.bfloat16)
    serial = mx.concatenate(
        [norm(mx.contiguous(x[:, i : i + 1])) for i in range(x.shape[1])], axis=1
    )
    assert mx.array_equal(norm(x), serial)


@pytest.mark.parametrize(
    "logit,expected",
    [
        (
            0.671875,
            [
                -1.46875,
                -0.5390625,
                0.1298828125,
                -0.5,
                -0.439453125,
                0.99609375,
                2.921875,
                2.265625,
            ],
        ),
        (
            -0.53125,
            [
                0.0517578125,
                -0.357421875,
                0.34765625,
                -0.9375,
                -1.453125,
                0.77734375,
                3.28125,
                -0.474609375,
            ],
        ),
        (
            -2.578125,
            [
                1.625,
                -0.1689453125,
                0.5703125,
                -1.390625,
                -2.5,
                0.5546875,
                3.65625,
                -3.328125,
            ],
        ),
    ],
)
def test_value_residual_rounding(config, values, logit, expected):
    attention = Attention(config, 0)
    attention.v_proj = nn.Identity()
    attention.value_res_lambda = mx.array([logit], dtype=mx.bfloat16)
    seed = mx.array(
        [[[2, -0.125, 0.625, -1.5, -2.75, 0.5, 3.75, -4]]], dtype=mx.bfloat16
    )
    _, _, actual = attention._project_qkv(values, None, seed)
    assert mx.array_equal(actual.reshape(-1), mx.array(expected, dtype=mx.bfloat16))


def test_silu_rounds_before_up_product(config, values):
    mlp = MLP(config)
    mlp.gate_proj = mlp.up_proj = mlp.down_proj = nn.Identity()
    expected = mx.array(
        [
            0.39453125,
            0.1796875,
            0.00732421875,
            0.00012302398681640625,
            0.380859375,
            1.21875,
            5.78125,
            30.125,
        ],
        dtype=mx.bfloat16,
    )
    assert mx.array_equal(mlp(values).reshape(-1), expected)


@pytest.mark.parametrize("length", [1, 8, 513])
def test_compiled_mlp_preserves_updated_weights(config, length):
    mx.random.seed(31)
    mlp = MLP(config)
    mlp.eval()
    x = mx.random.normal((2, length, config.hidden_size))
    for _ in range(2):
        expected = mlp._forward(x)
        actual = mlp(x)
        assert actual.shape == expected.shape
        assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5)
        mlp.gate_proj.weight = mlp.gate_proj.weight * 0.5


def test_rope_rounds_trigonometry_before_rotation(values):
    rope = RotaryEmbedding(8, 10000)
    x = mx.broadcast_to(values[:, None], (2, 1, 1, 8))
    expected = mx.array(
        [
            [-3.25, -0.75, -0.125, 0.015625, 0.75, 1.25, 2.5, 5.5],
            [
                2.140625,
                1.4453125,
                0.439453125,
                -5.1875,
                -2.546875,
                -0.2236328125,
                2.453125,
                1.828125,
            ],
        ],
        dtype=mx.bfloat16,
    )
    assert mx.array_equal(rope(x, offset=mx.array([0, 1234])).reshape(2, 8), expected)
