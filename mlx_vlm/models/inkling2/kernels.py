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


# 8-bit global KV (MLX_VLM_INKLING_KV8): each key or value row (one token, one
# head) is stored as its D values in int8 followed by their FP32 absmax scale as
# 4 more int8, so a cache row is one ordinary int8 array of D + 4. Cache
# classes, batch padding, trimming and prefix-cache copies handle it as any
# other array, at D + 4 bytes a head-token instead of 2 * D. Keys are L2-clipped
# and values RMS-normalised, so one scale per row suffices. The flash and
# split-K kernels read the rows directly; the other paths unpack them.
@mx.compile
def kv8_pack(x):
    """(..., D) float -> (..., D + 4) int8: values by the row absmax, then its scale."""
    xf = x.astype(mx.float32)
    scale = mx.max(mx.abs(xf), axis=-1, keepdims=True) / 127.0
    values = mx.round(xf / mx.maximum(scale, 1e-30)).astype(mx.int8)
    return mx.concatenate([values, scale.view(mx.int8)], axis=-1)


def kv8_unpack(packed, dtype=mx.bfloat16):
    """The inverse of kv8_pack: (..., D + 4) int8 -> (..., D) in dtype."""
    d = packed.shape[-1] - 4
    scale = mx.contiguous(packed[..., d:]).view(mx.float32)
    return (packed[..., :d].astype(mx.float32) * scale).astype(dtype)


def _packed_kv(q, k, v):
    """False for BF16 K/V rows of D, True for kv8_pack rows, None otherwise."""
    d = q.shape[-1]
    if q.dtype != mx.bfloat16 or k.dtype != v.dtype or v.shape != k.shape:
        return None
    if k.dtype == mx.bfloat16 and k.shape[-1] == d:
        return False
    if k.dtype == mx.int8 and k.shape[-1] == d + 4:
        return True
    return None


# The relative-attention kernel below is by Jun Kim (@jundot), from
# ml-explore/mlx-lm pull request #2086 (mlx_lm/models/inkling2/kernels.py,
# whose file header reads Copyright 2026 Apple Inc., MIT License):
# causal flash attention with a relative position bias and a sink logit per
# query. Queries per threadgroup, 8 for each of its 4 simdgroups.
_BQ = 32
_TG = 128

# The online softmax of each row starts from its sink logit, so the zero-value
# sink needs no key.
_RELATIVE_ATTENTION_SOURCE = r"""
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr int BK = 16;
    constexpr int LDK = BK + 8;
    constexpr int LDV = D + 8;
    const int L = sizes[0];
    const int S = sizes[1];
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int batch = threadgroup_position_in_grid.z;
    const int head = threadgroup_position_in_grid.y;
    const int kv_head = head / (HQ / HKV);
    const int q0 = threadgroup_position_in_grid.x * BQ;
    const int start = key_start[batch];
    const float qscale = scale * LOG2E;

    // The lane holds row fm and columns fn, fn + 1 of each 8x8 fragment.
    const short fm = ((lane / 4) & 4) + (lane / 2) % 4;
    const short fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
    const int row = q0 + int(sg) * 8 + fm;
    const bool row_ok = row < L;
    const int diag = S - L + row;

    // q and k are clipped and v has unit RMS, so half holds them.
    threadgroup half Ks[D * LDK];
    threadgroup half Vs[BK * LDV];

    simdgroup_half8x8 q_frag[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        half2 qv = half2(0.0h);
        if (row_ok) {
            const device T* qp = q + batch * q_strides[0] + head * q_strides[1]
                + row * q_strides[2] + db * 8 + fn;
            qv = half2(float(qp[0]), float(qp[1]));
        }
        q_frag[db].thread_elements()[0] = qv.x;
        q_frag[db].thread_elements()[1] = qv.y;
    }
    simdgroup_float8x8 acc[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        acc[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float row_max = row_ok ? float(sinks[(batch * L + row) * HQ + head]) * LOG2E : 0.0f;
    float row_sum = 1.0f;
    const int bias_row = batch * L + (row_ok ? row : 0);
    const device T* bias = rel + (size_t(bias_row) * HQ + head) * R;

    const int first_diag = S - L + q0;
    const int last_diag = S - L + min(q0 + BQ, L) - 1;
    int first_key = WINDOW > 0 ? max(0, first_diag - WINDOW + 1) : 0;
    first_key = max(first_key, start);
    // PACKED: each key/value row is D int8 values, then their FP32 scale
    // (4 bytes, aligned as rows are D + 4 bytes and D is a multiple of 8).
    auto k_head = k + batch * k_strides[0] + kv_head * k_strides[1];
    auto v_head = v + batch * v_strides[0] + kv_head * v_strides[1];

    for (int k0 = first_key / BK * BK; k0 <= last_diag; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int c = tid; c < BK * D / 8; c += TG) {
            const int r = c / (D / 8);
            const int col = (c % (D / 8)) * 8;
            const int key = k0 + r;
            float kf[8], vf[8];
            for (int j = 0; j < 8; ++j) {
                kf[j] = 0.0f;
                vf[j] = 0.0f;
            }
            if (key < S) {
                auto krow = k_head + key * k_strides[2];
                auto vrow = v_head + key * v_strides[2];
                if (PACKED) {
                    const float ks = *((const device float*)(krow + D));
                    const float vs = *((const device float*)(vrow + D));
                    const char4 ka = *((const device char4*)(krow + col));
                    const char4 kb = *((const device char4*)(krow + col + 4));
                    const char4 va = *((const device char4*)(vrow + col));
                    const char4 vb = *((const device char4*)(vrow + col + 4));
                    for (int j = 0; j < 4; ++j) {
                        kf[j] = float(ka[j]) * ks;
                        kf[j + 4] = float(kb[j]) * ks;
                        vf[j] = float(va[j]) * vs;
                        vf[j + 4] = float(vb[j]) * vs;
                    }
                } else {
                    auto kp = (const device vec<T, 4>*)(krow + col);
                    auto vp = (const device vec<T, 4>*)(vrow + col);
                    for (int j = 0; j < 8; ++j) {
                        kf[j] = float(kp[j / 4][j % 4]);
                        vf[j] = float(vp[j / 4][j % 4]);
                    }
                }
            }
            for (int j = 0; j < 8; ++j) {
                Ks[(col + j) * LDK + r] = half(kf[j]);
                Vs[r * LDV + col + j] = half(vf[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (q0 + int(sg) * 8 >= L) {
            continue;
        }

        simdgroup_float8x8 scores[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            scores[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int db = 0; db < D / 8; ++db) {
                simdgroup_half8x8 k_frag;
                simdgroup_load(k_frag, Ks + db * 8 * LDK + kb * 8, LDK);
                simdgroup_multiply_accumulate(scores[kb], q_frag[db], k_frag, scores[kb]);
            }
        }

        float tile_max = -INFINITY;
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const int key = k0 + kb * 8 + fn + e;
                const int d = diag - key;
                const bool visible = row_ok && d >= 0 && key >= start
                    && (WINDOW == 0 || d < WINDOW);
                float s = -INFINITY;
                if (visible) {
                    s = scores[kb].thread_elements()[e] * qscale;
                    if (d < R) {
                        s += float(bias[d]) * LOG2E;
                    }
                }
                scores[kb].thread_elements()[e] = s;
                tile_max = max(tile_max, s);
            }
        }
        // The 4 lanes of a row are lane ^ 1 and lane ^ 8 apart.
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 1));
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 8));
        const float new_max = max(row_max, tile_max);
        const float factor = fast::exp2(row_max - new_max);
        float tile_sum = 0.0f;
        simdgroup_half8x8 probs[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const float p = fast::exp2(scores[kb].thread_elements()[e] - new_max);
                probs[kb].thread_elements()[e] = half(p);
                tile_sum += p;
            }
        }
        tile_sum += simd_shuffle_xor(tile_sum, 1);
        tile_sum += simd_shuffle_xor(tile_sum, 8);
        row_sum = row_sum * factor + tile_sum;
        row_max = new_max;
        for (int db = 0; db < D / 8; ++db) {
            acc[db].thread_elements()[0] *= factor;
            acc[db].thread_elements()[1] *= factor;
            for (int kb = 0; kb < BK / 8; ++kb) {
                simdgroup_half8x8 v_frag;
                simdgroup_load(v_frag, Vs + kb * 8 * LDV + db * 8, LDV);
                simdgroup_multiply_accumulate(acc[db], probs[kb], v_frag, acc[db]);
            }
        }
    }

    if (row_ok) {
        device T* op = out + (size_t(batch * L + row) * HQ + head) * D;
        for (int db = 0; db < D / 8; ++db) {
            op[db * 8 + fn] = T(acc[db].thread_elements()[0] / row_sum);
            op[db * 8 + fn + 1] = T(acc[db].thread_elements()[1] / row_sum);
        }
    }
"""


@lru_cache(maxsize=1)
def _relative_attention_kernel():
    return mx.fast.metal_kernel(
        name="inkling2_relative_attention",
        input_names=["q", "k", "v", "rel", "sinks", "key_start", "sizes", "scale"],
        output_names=["out"],
        source=_RELATIVE_ATTENTION_SOURCE,
        ensure_row_contiguous=False,
    )


def relative_attention(q, k, v, r, rel_proj, sinks, scale, window, key_start=None):
    """Causal attention with the relative bias and sink logit, or None for the MLX path.

    q: (B, Hq, L, D) BF16 at the last L of the keys; k, v: (B, Hkv, S, D) BF16
    without a sink row, or int8 rows of D + 4 from kv8_pack; r: (B, L, Hq,
    d_rel); rel_proj: (d_rel, R), the bias at distances below R; sinks: (B, L,
    Hq); key_start: (B,), each row's first valid key. Returns (B, L, Hq, D).
    """
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        return None
    B, Hq, L, D = q.shape
    Hkv, S = k.shape[1], k.shape[2]
    packed = _packed_kv(q, k, v)
    # Threadgroup memory holds a K and a V tile in half precision.
    if packed is None or D % 8 or D > 256 or Hq % Hkv or k.shape[0] != B:
        return None
    if key_start is None:
        key_start = mx.zeros((B,), mx.int32)
    return _relative_attention_kernel()(
        inputs=[
            q,
            k,
            v,
            mx.contiguous(r @ rel_proj.astype(r.dtype)),
            mx.contiguous(sinks.astype(mx.float32)),
            mx.contiguous(key_start.astype(mx.int32)),
            mx.array([L, S], dtype=mx.int32),
            mx.array(scale, dtype=mx.float32),
        ],
        template=[
            ("T", q.dtype),
            ("D", D),
            ("HQ", Hq),
            ("HKV", Hkv),
            ("R", rel_proj.shape[-1]),
            ("WINDOW", window),
            ("BQ", _BQ),
            ("TG", _TG),
            ("PACKED", packed),
        ],
        grid=((L + _BQ - 1) // _BQ * _TG, Hq, B),
        threadgroup=(_TG, 1, 1),
        output_shapes=[(B, L, Hq, D)],
        output_dtypes=[q.dtype],
    )[0]


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
        // PACKED: each key/value row is D int8 values, then their FP32 scale.
        auto k_head = k + g * k_strides[1];
        auto v_head = v + g * v_strides[1];
        for (int j = first + int(sg); j < end; j += SG) {
            auto krow = k_head + size_t(j) * k_strides[2];
            auto vrow = v_head + size_t(j) * v_strides[2];
            const float ks = PACKED ? *((const device float*)(krow + D)) : 1.0f;
            const float vs = PACKED ? *((const device float*)(vrow + D)) : 1.0f;
            float kv[E], vv[E];
            for (int e = 0; e < E; ++e) {
                kv[e] = float(krow[lane * E + e]) * ks;
                vv[e] = float(vrow[lane * E + e]) * vs;
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

    q: (1, HQ, 1, D) BF16; k, v: (1, HKV, S, D) BF16 with the query at the last
    key, or int8 rows of D + 4 from kv8_pack; rel: (HQ, R), the bias at backward
    distances below R; sinks: (HQ,); key_start: optional (1,) int array, the
    first valid key (left padding). Returns (1, HQ, 1, D) in q's dtype.
    """
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        return None
    _, hq, _, d = q.shape
    hkv, s = k.shape[1], k.shape[2]
    packed = _packed_kv(q, k, v)
    if (
        packed is None
        or q.shape != (1, hq, 1, d)
        or k.shape[:3] != (1, hkv, s)
        or rel.shape[0] != hq
        or sinks.shape != (hq,)
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
            ("PACKED", packed),
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
