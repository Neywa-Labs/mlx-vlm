"""Attention over a 6-bit cache: the fused decode kernels and the dequantize + SDPA prefill path,
against an fp32 reference, through the cache dispatch, and their limits."""

import mlx.core as mx
import pytest

import mlx_vlm.models.base as base
from mlx_vlm.kernels.quantized_decode_attention import (
    quantized_decode_attention,
    quantized_decode_attention_supported,
)
from mlx_vlm.models.cache import BatchQuantizedKVCache, QuantizedKVCache

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="the decode attention kernels are Metal only"
)

D = 256


def _packed(B, H, T, group_size, seed, slack=37):
    """Packed K and V as a cache holds them: slices of longer buffers."""
    mx.random.seed(seed)
    k = mx.random.normal((B, H, T + slack, D)).astype(mx.bfloat16)
    v = mx.random.normal((B, H, T + slack, D)).astype(mx.bfloat16)
    qk = tuple(x[..., :T, :] for x in mx.quantize(k, group_size=group_size, bits=6))
    qv = tuple(x[..., :T, :] for x in mx.quantize(v, group_size=group_size, bits=6))
    return qk, qv


def _reference(q, qk, qv, scale, mask, group_size):
    """fp32 attention for queries (B, n_q, L, D) that are the last L of T keys."""
    k = mx.dequantize(*qk, group_size=group_size, bits=6).astype(mx.float32)
    v = mx.dequantize(*qv, group_size=group_size, bits=6).astype(mx.float32)
    B, n_q, L = q.shape[:3]
    H, T = k.shape[1], k.shape[2]
    qf = q.astype(mx.float32).reshape(B, H, n_q // H, L, D) * scale
    s = qf @ mx.expand_dims(k, 2).swapaxes(-1, -2)
    if isinstance(mask, str):
        mask = mx.arange(T - L, T)[:, None] >= mx.arange(T)[None]
    if mask is not None:
        m = mask.reshape(mask.shape[0] if mask.ndim == 4 else 1, 1, 1, L, T)
        s = (
            mx.where(m, s, -float("inf"))
            if m.dtype == mx.bool_
            else s + m.astype(mx.float32)
        )
    p = mx.softmax(s, axis=-1, precise=True)
    return (p @ mx.expand_dims(v, 2)).reshape(B, n_q, L, D)


def _mask(kind, B, T):
    if kind == "bool":
        starts = mx.array([0, 300][:B])
        return (mx.arange(T)[None, :] >= starts[:, None]).reshape(B, 1, 1, T)
    if kind == "additive":
        starts = mx.array([0, 50][:B])
        keep = mx.arange(T)[None, :] >= starts[:, None]
        return mx.where(keep, 0.0, -1e9).astype(mx.bfloat16).reshape(B, 1, 1, T)
    return kind


@pytest.mark.parametrize(
    "B,n_q,H,T,group_size,mask_kind",
    [
        (1, 16, 2, 1, 64, None),
        (1, 16, 2, 65, 64, "causal"),
        (1, 16, 2, 4099, 64, None),
        (2, 16, 2, 2048, 64, "bool"),
        (2, 16, 2, 777, 64, "additive"),
        (1, 24, 4, 3000, 64, None),
        (1, 4, 4, 513, 64, None),
        (1, 16, 2, 999, 32, None),
        (1, 16, 2, 999, 128, None),
    ],
)
def test_matches_fp32_reference(B, n_q, H, T, group_size, mask_kind):
    """The kernels at any length (dispatch only uses them from _MIN_KEYS keys)."""
    qk, qv = _packed(B, H, T, group_size, seed=T + n_q)
    mx.random.seed(1)
    q = mx.random.normal((B, n_q, 1, D)).astype(mx.bfloat16)
    mask = _mask(mask_kind, B, T)
    scale = D**-0.5

    got = quantized_decode_attention(q, qk, qv, scale, mask, group_size=group_size)
    ref = _reference(q, qk, qv, scale, mask, group_size)
    shipped = base.quantized_scaled_dot_product_attention(
        q, qk, qv, scale=scale, mask=mask, group_size=group_size, bits=6
    )
    assert got.shape == (B, n_q, 1, D) and got.dtype == mx.bfloat16
    err = mx.abs(got.astype(mx.float32) - ref).max().item()
    shipped_err = mx.abs(shipped.astype(mx.float32) - ref).max().item()
    # One bf16 rounding of the output; never worse than the quantized_matmul path.
    assert err <= max(shipped_err, 2**-8 * mx.abs(ref).max().item())


@pytest.mark.parametrize("prompt_len,uses_kernel", [(1500, True), (700, False)])
@pytest.mark.parametrize(
    "make_cache",
    [lambda: QuantizedKVCache(64, 6), lambda: BatchQuantizedKVCache([0], 64, 6)],
)
def test_cache_decode_dispatch(monkeypatch, make_cache, prompt_len, uses_kernel):
    cache = make_cache()
    mx.random.seed(5)
    prompt_k = mx.random.normal((1, 2, prompt_len, D)).astype(mx.bfloat16)
    prompt_v = mx.random.normal((1, 2, prompt_len, D)).astype(mx.bfloat16)
    cache.update_and_fetch(prompt_k, prompt_v)
    step_k = mx.random.normal((1, 2, 1, D)).astype(mx.bfloat16)
    step_v = mx.random.normal((1, 2, 1, D)).astype(mx.bfloat16)
    keys, values = cache.update_and_fetch(step_k, step_v)
    q = mx.random.normal((1, 16, 1, D)).astype(mx.bfloat16)
    scale = D**-0.5
    # The shipped path scales `queries` in place (`queries *= scale`): give
    # every call its own array.
    monkeypatch.setattr(base, "QUANT_DECODE_KERNEL", True)
    dispatched = base.scaled_dot_product_attention(
        mx.array(q), keys, values, cache, scale, None
    )
    monkeypatch.setattr(base, "QUANT_DECODE_KERNEL", False)
    shipped = base.scaled_dot_product_attention(
        mx.array(q), keys, values, cache, scale, None
    )
    direct = quantized_decode_attention(q, keys, values, scale, None, group_size=64)

    assert keys[0].shape[2] == prompt_len + 1
    expected = direct if uses_kernel else shipped
    assert mx.array_equal(dispatched, expected).item()
    ref = _reference(q, keys, values, scale, None, 64)
    err = mx.abs(dispatched.astype(mx.float32) - ref).max().item()
    shipped_err = mx.abs(shipped.astype(mx.float32) - ref).max().item()
    assert err <= max(shipped_err, 2**-8 * mx.abs(ref).max().item())


def test_unsupported_calls_fall_back():
    qk, qv = _packed(1, 2, 1024, 64, seed=0)
    q = mx.zeros((1, 16, 1, D), dtype=mx.bfloat16)
    assert quantized_decode_attention_supported(q, qk, qv, None, 64, 6)
    short = tuple(x[..., :1023, :] for x in qk)
    assert not quantized_decode_attention_supported(q, short, qv, None, 64, 6)
    assert not quantized_decode_attention_supported(q, qk, qv, None, 64, 8)
    assert not quantized_decode_attention_supported(q[:, :, :0], qk, qv, None, 64, 6)
    two = mx.zeros((1, 16, 2, D), dtype=mx.bfloat16)
    assert not quantized_decode_attention_supported(two, qk, qv, None, 64, 6)
    small = mx.zeros((1, 16, 1, 128), dtype=mx.bfloat16)
    assert not quantized_decode_attention_supported(small, qk, qv, None, 64, 6)
    per_head = mx.zeros((1, 16, 1, 1024), dtype=mx.bool_)
    assert not quantized_decode_attention_supported(q, qk, qv, per_head, 64, 6)
    uneven = mx.zeros((1, 15, 1, D), dtype=mx.bfloat16)
    assert not quantized_decode_attention_supported(uneven, qk, qv, None, 64, 6)


@pytest.mark.parametrize("mask_kind", ["causal", "bool"])
def test_prefill_chunk_dequant_sdpa(monkeypatch, mask_kind):
    """A 64-token prompt chunk over a 6-bit batch cache, as the model's prefill runs it."""
    cache = BatchQuantizedKVCache([0], 64, 6)
    mx.random.seed(7)
    prompt_k = mx.random.normal((1, 2, 900, D)).astype(mx.bfloat16)
    prompt_v = mx.random.normal((1, 2, 900, D)).astype(mx.bfloat16)
    cache.update_and_fetch(prompt_k, prompt_v)
    chunk_k = mx.random.normal((1, 2, 64, D)).astype(mx.bfloat16)
    chunk_v = mx.random.normal((1, 2, 64, D)).astype(mx.bfloat16)
    keys, values = cache.update_and_fetch(chunk_k, chunk_v)
    T = keys[0].shape[2]
    q = mx.random.normal((1, 16, 64, D)).astype(mx.bfloat16)
    if mask_kind == "causal":
        mask = "causal"
    else:
        mask = (mx.arange(T - 64, T)[:, None] >= mx.arange(T)[None]).reshape(
            1, 1, 64, T
        )
    scale = D**-0.5

    monkeypatch.setattr(base, "QUANT_PREFILL_SDPA", True)
    fused = base.scaled_dot_product_attention(
        mx.array(q), keys, values, cache, scale, mask
    )
    monkeypatch.setattr(base, "QUANT_PREFILL_SDPA", False)
    shipped = base.scaled_dot_product_attention(
        mx.array(q), keys, values, cache, scale, mask
    )

    assert T == 964 and fused.shape == (1, 16, 64, D) and fused.dtype == mx.bfloat16
    direct = mx.fast.scaled_dot_product_attention(
        q,
        mx.dequantize(*keys, group_size=64, bits=6),
        mx.dequantize(*values, group_size=64, bits=6),
        scale=scale,
        mask=mask,
    )
    assert mx.array_equal(fused, direct).item()
    ref = _reference(q, keys, values, scale, mask, 64)
    err = mx.abs(fused.astype(mx.float32) - ref).max().item()
    shipped_err = mx.abs(shipped.astype(mx.float32) - ref).max().item()
    assert err <= max(shipped_err, 2**-8 * mx.abs(ref).max().item())


def _both_paths(monkeypatch, flag, q, keys, values, mask):
    cache = QuantizedKVCache(64, 6)
    scale = D**-0.5
    monkeypatch.setattr(base, flag, True)
    enabled = base.scaled_dot_product_attention(
        mx.array(q), keys, values, cache, scale, mask
    )
    monkeypatch.setattr(base, flag, False)
    shipped = base.scaled_dot_product_attention(
        mx.array(q), keys, values, cache, scale, mask
    )
    return enabled, shipped


def _close(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    assert mx.isfinite(a).all().item()
    assert mx.abs(a - b).max().item() <= 2**-6 * max(mx.abs(b).max().item(), 1)


def test_decode_one_dimensional_mask(monkeypatch):
    qk, qv = _packed(1, 2, 1024, 64, seed=1)
    q = mx.random.normal((1, 16, 1, D)).astype(mx.bfloat16)
    mask = mx.arange(1024) >= 100
    _close(*_both_paths(monkeypatch, "QUANT_DECODE_KERNEL", q, qk, qv, mask))


def test_decode_fully_masked_row(monkeypatch):
    qk, qv = _packed(1, 2, 1024, 64, seed=2)
    q = mx.random.normal((1, 16, 1, D)).astype(mx.bfloat16)
    mask = mx.zeros((1, 1, 1, 1024), dtype=mx.bool_)
    _close(*_both_paths(monkeypatch, "QUANT_DECODE_KERNEL", q, qk, qv, mask))


def test_decode_value_head_dim_falls_back(monkeypatch):
    qk, _ = _packed(1, 2, 1024, 64, seed=3)
    v = mx.random.normal((1, 2, 1024, 128)).astype(mx.bfloat16)
    qv = mx.quantize(v, group_size=64, bits=6)
    q = mx.random.normal((1, 2, 1, D)).astype(mx.bfloat16)
    enabled, shipped = _both_paths(monkeypatch, "QUANT_DECODE_KERNEL", q, qk, qv, None)
    assert shipped.shape == (1, 2, 1, 128)
    _close(enabled, shipped)
    assert not quantized_decode_attention_supported(q, qk, qv, None, 64, 6)


def test_prefill_float32_additive_mask(monkeypatch):
    qk, qv = _packed(1, 2, 200, 64, seed=4)
    q = mx.random.normal((1, 16, 8, D)).astype(mx.bfloat16)
    mask = mx.random.normal((1, 1, 8, 200)).astype(mx.float32)
    _close(*_both_paths(monkeypatch, "QUANT_PREFILL_SDPA", q, qk, qv, mask))


def test_decode_per_head_mask_falls_back(monkeypatch):
    """B == n_q: a per-head mask has as many values as a per-row one."""
    qk, qv = _packed(2, 2, 1024, 64, seed=5)
    q = mx.random.normal((2, 2, 1, D)).astype(mx.bfloat16)
    mask = mx.array([True, False])[None, :, None, None] | (mx.arange(1024) >= 512)
    _close(*_both_paths(monkeypatch, "QUANT_DECODE_KERNEL", q, qk, qv, mask))
    assert not quantized_decode_attention_supported(q, qk, qv, mask, 64, 6)
