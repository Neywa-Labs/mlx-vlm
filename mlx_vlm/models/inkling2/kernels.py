# Copyright © 2026 Neywa Labs Inc, Abdur Rahim

"""Inference kernels that retain Inkling 2's FP32 intermediate arithmetic."""

import math
from functools import lru_cache

import mlx.core as mx


@lru_cache(maxsize=None)
def _short_convolution(write_window, add_residual):
    source = r"""
        uint i = thread_position_in_grid.x;
        int B = x_shape[0], L = x_shape[1], C = x_shape[2];
        if (i >= B * L * C) return;
        int c = i % C, t = (i / C) % L, b = i / (C * L);
        float value = 0;
        for (int j = 0; j < W; ++j) {
            int p = t + j - W + 1;
            float xv = p < 0
                ? float(state[b * state_strides[0] + (p + W - 1) * state_strides[1]
                              + c * state_strides[2]])
                : ((HAS_MASK && !mask[b * mask_strides[0] + p * mask_strides[1]])
                    ? 0.0f
                    : float(x[b * x_strides[0] + p * x_strides[1] + c * x_strides[2]]));
            float w = weight[c * weight_strides[0] + j * weight_strides[1]];
            value = value + xv * w;
        }
        float input = float(x[b * x_strides[0] + t * x_strides[1] + c * x_strides[2]]);
        // Match the BF16 convolution output before the decoder's FP32 residual add.
        float result = float(T((value + input) / divisor));
    """
    inputs = ["state", "weight", "x", "mask", "divisor"]
    outputs = ["out"]
    if add_residual:
        inputs.append("residual")
        source += r"""
        result += float(residual[b * residual_strides[0] + t * residual_strides[1]
                                 + c * residual_strides[2]]);
        """
    source += "out[i] = OT(result);"
    if write_window:
        outputs.append("window")
        source += r"""
        int base = b * (W - 1 + L) * C + c;
        window[base + (W - 1 + t) * C] = input;
        if (t == 0) {
            for (int j = 0; j < W - 1; ++j) {
                window[base + j * C] = state[b * state_strides[0] + j * state_strides[1]
                                              + c * state_strides[2]];
            }
        }
        """
    return mx.fast.metal_kernel(
        name=f"inkling2_short_convolution_{int(write_window)}_{int(add_residual)}",
        input_names=inputs,
        output_names=outputs,
        ensure_row_contiguous=False,
        compile_options={"math_mode": "safe"},
        source=source,
    )


@lru_cache(maxsize=None)
def _divisor(width):
    return mx.array(math.sqrt(width + 1), mx.float32)


def short_convolution(
    state, weight, x, mask, width, residual=None, output_dtype=None, write_window=False
):
    """Return None off Metal, otherwise emit the output and optional FP32 window."""
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        return None
    inputs = [state, weight, x, x if mask is None else mask, _divisor(width)]
    output_dtype = x.dtype if output_dtype is None else output_dtype
    shapes, dtypes = [x.shape], [output_dtype]
    if residual is not None:
        inputs.append(residual)
    if write_window:
        shapes.append((x.shape[0], width - 1 + x.shape[1], x.shape[2]))
        dtypes.append(mx.float32)
    outputs = _short_convolution(write_window, residual is not None)(
        inputs=inputs,
        template=[
            ("T", x.dtype),
            ("OT", output_dtype),
            ("W", width),
            ("HAS_MASK", mask is not None),
        ],
        grid=(x.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=shapes,
        output_dtypes=dtypes,
    )
    return outputs if write_window else outputs[0]


@lru_cache(maxsize=1)
def _attention_head_kernel():
    return mx.fast.metal_kernel(
        name="inkling2_prepare_attention_heads",
        input_names=["q", "k", "v", "r", "q_scale", "r_scale", "params"],
        output_names=["qo", "ko", "vo", "ro"],
        compile_options={"math_mode": "safe"},
        source=r"""
        // One SIMD group per row and head. Inputs are row-contiguous BF16.
        const uint lane = thread_index_in_simdgroup;
        const uint heads = q_shape[2], kv_heads = k_shape[2];
        const uint slot = threadgroup_position_in_grid.y;
        const uint row = threadgroup_position_in_grid.z;
        if (slot >= heads + 2 * kv_heads) {
            if (lane >= 16) return;
            const uint head = slot - heads - 2 * kv_heads;
            const uint base = (row * heads + head) * 16;
            // MLX sums a 16-element relative-position head sequentially.
            float sum = 0, own = 0;
            for (uint i = 0; i < 16; ++i) {
                float x = float(r[base + i]) * float(r_scale[head * 16 + i]);
                const float square = x * x;
                sum += square;
                if (i == lane) own = x;
            }
            float factor = params[1] / (metal::precise::sqrt(sum) + 1e-6f);
            ro[base + lane] = bfloat16_t(own * (factor > 1.0f ? 1.0f : factor));
            return;
        }
        const bool query = slot < heads;
        const bool value = slot >= heads + kv_heads;
        const uint head = query ? slot : slot - heads - (value ? kv_heads : 0);
        const uint offset = head * 128 + lane * 4;
        const uint base = row * (query ? heads : kv_heads) * 128 + offset;
        auto input = query ? q : (value ? v : k);
        auto output = query ? qo : (value ? vo : ko);
        // Match MLX's 128-wide reduction: four consecutive squares per lane,
        // followed by a SIMD sum. Keep the square's FP32 rounding point.
        float xs[4], sum = 0;
        for (uint i = 0; i < 4; ++i) {
            xs[i] = float(input[base + i]);
            if (query) xs[i] *= float(q_scale[offset + i]);
            const float square = xs[i] * xs[i];
            sum += square;
        }
        sum = simd_sum(sum);
        float factor;
        if (value) {
            factor = metal::precise::rsqrt(sum / 128.0f + params[2]);
        } else {
            float ratio = params[0] / (metal::precise::sqrt(sum) + 1e-6f);
            // mx.minimum propagates a NaN norm to the entire head.
            factor = ratio > 1.0f ? 1.0f : ratio;
        }
        for (uint i = 0; i < 4; ++i) output[base + i] = bfloat16_t(xs[i] * factor);
        """,
    )


@lru_cache(maxsize=16)
def _head_parameters(clip, rel_clip, eps):
    return mx.array([clip, rel_clip, eps], dtype=mx.float32)


def prepare_attention_heads(q, k, v, r, q_scale, r_scale, clip, rel_clip, eps):
    """Return Q/K/R after clipping and V after RMSNorm, or None for the MLX path."""
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        return None
    batch, rows, heads, dim = q.shape
    kv_heads = k.shape[2]
    if (
        batch != 1
        or not 1 <= rows <= 8
        or dim != 128
        or k.shape != (batch, rows, kv_heads, dim)
        or v.shape != k.shape
        or r.shape != (batch, rows, heads, 16)
        or q_scale.shape != (heads, dim)
        or r_scale.shape != (heads, 16)
        or any(a.dtype != mx.bfloat16 for a in (q, k, v, r, q_scale, r_scale))
    ):
        return None
    return _attention_head_kernel()(
        inputs=[q, k, v, r, q_scale, r_scale, _head_parameters(clip, rel_clip, eps)],
        grid=(32, 2 * (heads + kv_heads), rows),
        threadgroup=(32, 1, 1),
        output_shapes=[q.shape, k.shape, v.shape, r.shape],
        output_dtypes=[mx.bfloat16] * 4,
    )


# Rows of the head per simdgroup, and simdgroups per threadgroup.
_HEAD_ROWS = 4
_HEAD_SIMDGROUPS = 8


@lru_cache(maxsize=1)
def _head_logits_kernel():
    return mx.fast.metal_kernel(
        name="inkling2_head_fp32_logits",
        input_names=["h", "w"],
        output_names=["out"],
        source=r"""
        // Each simdgroup produces R logits for all M rows; lanes split K into
        // 8-wide chunks. Products and sums are FP32.
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const int row0 = (int(threadgroup_position_in_grid.x) * SG + int(sg)) * R;
        if (row0 >= N) return;
        float acc[R][M];
        for (int r = 0; r < R; ++r)
            for (int m = 0; m < M; ++m) acc[r][m] = 0.0f;
        for (int c = lane; c < K / 8; c += 32) {
            const int k = c * 8;
            float wv[R][8];
            for (int r = 0; r < R; ++r) {
                const int row = min(row0 + r, N - 1);
                auto wp = (const device vec<T, 4>*)(w + size_t(row) * K + k);
                const vec<T, 4> a = wp[0], b = wp[1];
                for (int j = 0; j < 4; ++j) {
                    wv[r][j] = float(a[j]);
                    wv[r][j + 4] = float(b[j]);
                }
            }
            for (int m = 0; m < M; ++m) {
                auto hp = (const device float4*)(h + m * K + k);
                const float4 x0 = hp[0], x1 = hp[1];
                for (int r = 0; r < R; ++r) {
                    acc[r][m] += wv[r][0] * x0.x + wv[r][1] * x0.y + wv[r][2] * x0.z
                        + wv[r][3] * x0.w + wv[r][4] * x1.x + wv[r][5] * x1.y
                        + wv[r][6] * x1.z + wv[r][7] * x1.w;
                }
            }
        }
        for (int r = 0; r < R; ++r) {
            for (int m = 0; m < M; ++m) {
                const float total = simd_sum(acc[r][m]);
                if (lane == 0 && row0 + r < N) out[size_t(m) * N + row0 + r] = total;
            }
        }
        """,
    )


def head_logits(h, weight):
    """FP32 logits (M, N) of rows h (M, K) against a head (N, K), or None.

    Reads the head in its own dtype instead of an FP32 copy of it.
    """
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        return None
    m, k = h.shape
    n = weight.shape[0]
    if (
        weight.ndim != 2
        or weight.shape[1] != k
        or k % 8
        or weight.dtype not in (mx.bfloat16, mx.float16, mx.float32)
    ):
        return None
    rows = _HEAD_ROWS * _HEAD_SIMDGROUPS
    return _head_logits_kernel()(
        inputs=[mx.contiguous(h.astype(mx.float32)), weight],
        template=[
            ("T", weight.dtype),
            ("M", m),
            ("K", k),
            ("N", n),
            ("R", _HEAD_ROWS),
            ("SG", _HEAD_SIMDGROUPS),
        ],
        grid=((n + rows - 1) // rows * 32 * _HEAD_SIMDGROUPS, 1, 1),
        threadgroup=(32 * _HEAD_SIMDGROUPS, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[mx.float32],
    )[0]
