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


# Single-query decode attention over a long chronological K/V cache, split over
# key blocks (flash-decoding). One threadgroup reads one key block of one KV
# head once for all of that head's query heads, instead of once per query head.
_DECODE_BLOCK = 512
_DECODE_SIMDGROUPS = 8


@lru_cache(maxsize=1)
def _decode_partials():
    return mx.fast.metal_kernel(
        name="inkling2_decode_attention_partials",
        input_names=["q", "k", "v", "rel", "key_start", "sizes", "scale"],
        output_names=["part_max", "part_sum", "part_acc"],
        ensure_row_contiguous=False,
        source=r"""
        constexpr int G = HQ / HKV;
        constexpr int E = D / 32;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const int block = threadgroup_position_in_grid.x;
        const int g = threadgroup_position_in_grid.y;
        const int S = sizes[0];
        const int start = block * BS;
        const int end = min(start + BS, S);
        // Keys before key_start are a batch row's left padding.
        const int first = max(start, key_start[0]);
        const float sc = scale;

        float qv[G][E];
        for (int h = 0; h < G; ++h)
            for (int e = 0; e < E; ++e)
                qv[h][e] = float(q[(g * G + h) * D + lane * E + e]) * sc;
        // A finite floor instead of -inf: an empty block or simdgroup then
        // carries zero weight without relying on infinities.
        float m[G], l[G], acc[G][E];
        for (int h = 0; h < G; ++h) {
            m[h] = -1e30f;
            l[h] = 0.0f;
            for (int e = 0; e < E; ++e) acc[h][e] = 0.0f;
        }
        const device T* k_head = k + g * k_strides[1];
        const device T* v_head = v + g * v_strides[1];
        for (int j = first + int(sg); j < end; j += SG) {
            const device T* kp = k_head + size_t(j) * k_strides[2] + lane * E;
            const device T* vp = v_head + size_t(j) * v_strides[2] + lane * E;
            float kv[E], vv[E];
            for (int e = 0; e < E; ++e) {
                kv[e] = float(kp[e]);
                vv[e] = float(vp[e]);
            }
            // The relative bias covers the newest REL keys only.
            const int dist = S - 1 - j;
            for (int h = 0; h < G; ++h) {
                float s = 0.0f;
                for (int e = 0; e < E; ++e) s += qv[h][e] * kv[e];
                s = simd_sum(s);
                if (dist < REL) s += float(rel[(g * G + h) * REL + dist]);
                const float mn = max(m[h], s);
                const float f = metal::exp(m[h] - mn);
                const float p = metal::exp(s - mn);
                l[h] = l[h] * f + p;
                for (int e = 0; e < E; ++e) acc[h][e] = acc[h][e] * f + p * vv[e];
                m[h] = mn;
            }
        }

        threadgroup float tm[SG][G];
        threadgroup float tl[SG][G];
        threadgroup float ta[SG][G][D];
        for (int h = 0; h < G; ++h) {
            if (lane == 0) {
                tm[sg][h] = m[h];
                tl[sg][h] = l[h];
            }
            for (int e = 0; e < E; ++e) ta[sg][h][lane * E + e] = acc[h][e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Thread t combines head t / D, dimension t % D across the simdgroups.
        for (uint t = thread_position_in_threadgroup.x; t < uint(G * D); t += SG * 32) {
            const int h = t / D, d = t % D;
            float M = -1e30f;
            for (int s = 0; s < SG; ++s) M = max(M, tm[s][h]);
            float L = 0.0f, A = 0.0f;
            for (int s = 0; s < SG; ++s) {
                const float w = metal::exp(tm[s][h] - M);
                L += tl[s][h] * w;
                A += ta[s][h][d] * w;
            }
            const int row = block * HQ + g * G + h;
            part_acc[size_t(row) * D + d] = A;
            if (d == 0) {
                part_max[row] = M;
                part_sum[row] = L;
            }
        }
        """,
    )


@lru_cache(maxsize=1)
def _decode_combine():
    return mx.fast.metal_kernel(
        name="inkling2_decode_attention_combine",
        input_names=["part_max", "part_sum", "part_acc", "sinks", "sizes"],
        output_names=["out"],
        source=r"""
        const int h = threadgroup_position_in_grid.x;
        const int d = thread_position_in_threadgroup.x;
        const int nblocks = sizes[1];
        // The softmax starts from the sink logit, whose value row is zero.
        float M = float(sinks[h]);
        for (int b = 0; b < nblocks; ++b) M = max(M, part_max[b * HQ + h]);
        float L = metal::exp(float(sinks[h]) - M), A = 0.0f;
        for (int b = 0; b < nblocks; ++b) {
            const float w = metal::exp(part_max[b * HQ + h] - M);
            L += part_sum[b * HQ + h] * w;
            A += part_acc[size_t(b * HQ + h) * D + d] * w;
        }
        out[h * D + d] = T(A / L);
        """,
    )


def decode_attention(q, k, v, rel, sinks, scale, key_start=None):
    """One query of one sequence over a chronological cache, or None for SDPA.

    q: (1, HQ, 1, D); k, v: (1, HKV, S, D) with the query at the last key;
    rel: (HQ, R), the bias at backward distances below R; sinks: (HQ,);
    key_start: optional (1,) int array, the first valid key (left padding).
    Returns (1, HQ, 1, D) in q's dtype.
    """
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        return None
    _, hq, _, d = q.shape
    hkv, s = k.shape[1], k.shape[2]
    if (
        q.shape != (1, hq, 1, d)
        or k.shape != (1, hkv, s, d)
        or v.shape != k.shape
        or rel.shape[0] != hq
        or sinks.shape != (hq,)
        or any(a.dtype != mx.bfloat16 for a in (q, k, v))
        or d % 32
        or hq % hkv
        # Per-simdgroup partials for every query head of a KV head must fit
        # in 32 KB of threadgroup memory.
        or _DECODE_SIMDGROUPS * (hq // hkv) * (d + 2) * 4 > 32768
    ):
        return None
    nblocks = (s + _DECODE_BLOCK - 1) // _DECODE_BLOCK
    if key_start is None:
        key_start = mx.zeros((1,), mx.int32)
    common = [("T", q.dtype), ("D", d), ("HQ", hq), ("HKV", hkv)]
    part_max, part_sum, part_acc = _decode_partials()(
        inputs=[
            mx.contiguous(q.reshape(hq, d)),
            k,
            v,
            mx.contiguous(rel),
            key_start.astype(mx.int32),
            mx.array([s], dtype=mx.int32),
            mx.array(scale, dtype=mx.float32),
        ],
        template=[
            *common,
            ("REL", rel.shape[-1]),
            ("BS", _DECODE_BLOCK),
            ("SG", _DECODE_SIMDGROUPS),
        ],
        grid=(nblocks * 32 * _DECODE_SIMDGROUPS, hkv, 1),
        threadgroup=(32 * _DECODE_SIMDGROUPS, 1, 1),
        output_shapes=[(nblocks * hq,), (nblocks * hq,), (nblocks * hq, d)],
        output_dtypes=[mx.float32] * 3,
    )
    out = _decode_combine()(
        inputs=[
            part_max,
            part_sum,
            part_acc,
            sinks.astype(mx.float32),
            mx.array([s, nblocks], dtype=mx.int32),
        ],
        template=common,
        grid=(hq * d, 1, 1),
        threadgroup=(d, 1, 1),
        output_shapes=[(hq, d)],
        output_dtypes=[q.dtype],
    )[0]
    return out.reshape(1, hq, 1, d)
