# Copyright © 2026 Apple Inc.

"""Decode attention (one query position) over a 6-bit affine-quantized KV cache.

`quantized_matmul` with grouped-query attention reads every KV head once per
query head that shares it, so attention over a long 6-bit cache runs at a
fraction of the memory bandwidth. These kernels read each packed key and value
row once for all query heads of its KV head:

1. scores: one thread per key dequantizes its row and dots it with all R query
   heads (the queries sit in threadgroup memory); fp32 scores (B, H, R, T).
2. the mask and an fp32 softmax over T run as ordinary MLX ops.
3. values: one SIMD group per stretch of keys, each lane owning D / 32 output
   columns of all R heads; threadgroups write partial sums that MLX adds up.

Inputs are the cache's (packed, scales, biases) slices as stored, strides and
all: no copy of the cache. MLX packs 6-bit values as a little-endian bit stream,
four values per three bytes.
"""

import mlx.core as mx

from ._common import scalar

_SCORES_SOURCE = """
    uint t = thread_position_in_grid.x;
    uint h = thread_position_in_grid.y;
    uint b = thread_position_in_grid.z;
    uint tid = thread_position_in_threadgroup.x;
    uint T = kw_shape[2];

    // The R query heads of KV head h, scaled, in fp32: q is (B, n_q, 1, D), any strides.
    threadgroup float4 qs[R * D / 4];
    threadgroup float* qf = (threadgroup float*)qs;
    for (uint i = tid; i < R * D; i += TG) {
        uint r = i / D, d = i % D;
        qf[i] = static_cast<float>(q[b * q_strides[0] + (h * R + r) * q_strides[1] + d * q_strides[3]]) * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t >= T) {
        return;
    }

    const device uint32_t* kr = kw + b * kw_strides[0] + h * kw_strides[1] + t * kw_strides[2];
    const device auto* sr = ks + b * ks_strides[0] + h * ks_strides[1] + t * ks_strides[2];
    const device auto* br = kb + b * kb_strides[0] + h * kb_strides[1] + t * kb_strides[2];

    float acc[R];
    for (int r = 0; r < R; r++) {
        acc[r] = 0.0f;
    }
    // 16 values per step: three 32-bit words, 96 bits, inside one group.
    for (int c = 0; c < D / 16; c++) {
        int ws = kw_strides[3];
        uint w0 = kr[(3 * c) * ws], w1 = kr[(3 * c + 1) * ws], w2 = kr[(3 * c + 2) * ws];
        int g = (16 * c) / G;
        float scale = static_cast<float>(sr[g * ks_strides[3]]);
        float bias = static_cast<float>(br[g * kb_strides[3]]);
        float4 k[4];
        k[0] = float4(w0 & 63u, (w0 >> 6) & 63u, (w0 >> 12) & 63u, (w0 >> 18) & 63u);
        k[1] = float4((w0 >> 24) & 63u, (w0 >> 30) | ((w1 & 15u) << 2), (w1 >> 4) & 63u, (w1 >> 10) & 63u);
        k[2] = float4((w1 >> 16) & 63u, (w1 >> 22) & 63u, (w1 >> 28) | ((w2 & 3u) << 4), (w2 >> 2) & 63u);
        k[3] = float4((w2 >> 8) & 63u, (w2 >> 14) & 63u, (w2 >> 20) & 63u, (w2 >> 26) & 63u);
        for (int i = 0; i < 4; i++) {
            k[i] = k[i] * scale + bias;
        }
        for (int r = 0; r < R; r++) {
            const threadgroup float4* qc = qs + (r * D + 16 * c) / 4;
            acc[r] += dot(qc[0], k[0]) + dot(qc[1], k[1]) + dot(qc[2], k[2]) + dot(qc[3], k[3]);
        }
    }
    device float* o = scores + ((b * H + h) * R) * T + t;
    for (int r = 0; r < R; r++) {
        o[r * T] = acc[r];
    }
"""

_VALUES_SOURCE = """
    uint blk = threadgroup_position_in_grid.x;
    uint h = threadgroup_position_in_grid.y;
    uint b = threadgroup_position_in_grid.z;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint T = vw_shape[2];
    uint start = blk * bk;
    uint stop = min(start + bk, T);

    // Lane `lane` owns columns [8 lane, 8 lane + 8) of all R heads: the 48 bits
    // at bit 48 lane, which start at bit 0 or 16 of word 3 lane / 2.
    float4 o[R][2];
    for (int r = 0; r < R; r++) {
        o[r][0] = 0.0f;
        o[r][1] = 0.0f;
    }
    const device float* pr = probs + ((b * H + h) * R) * T;
    int g = (8 * lane) / G;
    int ws = vw_strides[3];
    bool odd = lane & 1;
    const device uint32_t* vh = vw + b * vw_strides[0] + h * vw_strides[1] + (3 * lane / 2) * ws;
    const device auto* sh = vs + b * vs_strides[0] + h * vs_strides[1] + g * vs_strides[3];
    const device auto* bh = vb + b * vb_strides[0] + h * vb_strides[1] + g * vb_strides[3];
    // Tiles of 32 keys per SIMD group: lane i loads the R probabilities of key
    // t0 + i (coalesced), and simd_shuffle hands each key's to every lane.
    for (uint t0 = start + 32 * sg; t0 < stop; t0 += 32 * NSG) {
        float pl[R];
        bool live = t0 + lane < stop;
        for (int r = 0; r < R; r++) {
            pl[r] = live ? pr[r * T + t0 + lane] : 0.0f;
        }
        uint n = min(32u, stop - t0);
        for (uint i = 0; i < n; i++) {
            uint t = t0 + i;
            const device uint32_t* vr = vh + t * vw_strides[2];
            uint wa = vr[0], wb = vr[ws];
            uint lo = odd ? ((wa >> 16) | (wb << 16)) : wa;
            uint hi = odd ? (wb >> 16) : wb;
            float scale = static_cast<float>(sh[t * vs_strides[2]]);
            float bias = static_cast<float>(bh[t * vb_strides[2]]);
            float4 v0 = float4(lo & 63u, (lo >> 6) & 63u, (lo >> 12) & 63u, (lo >> 18) & 63u) * scale + bias;
            float4 v1 = float4((lo >> 24) & 63u, (lo >> 30) | ((hi & 15u) << 2), (hi >> 4) & 63u, (hi >> 10) & 63u)
                * scale + bias;
            for (int r = 0; r < R; r++) {
                float4 p = float4(simd_shuffle(pl[r], ushort(i)));
                o[r][0] = fma(p, v0, o[r][0]);
                o[r][1] = fma(p, v1, o[r][1]);
            }
        }
    }

    // Add the SIMD groups' sums in threadgroup memory, then write this block's partial.
    threadgroup float4 red[R * D / 4];
    for (uint s = 0; s < NSG; s++) {
        if (sg == s) {
            for (int r = 0; r < R; r++) {
                for (int q = 0; q < 2; q++) {
                    uint idx = (r * D + 8 * lane) / 4 + q;
                    red[idx] = (s == 0 ? float4(0.0f) : red[idx]) + o[r][q];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint NB = threadgroups_per_grid.x;
    device float4* out = (device float4*)(partial + (((b * H + h) * NB + blk) * R) * D);
    for (uint i = sg * 32 + lane; i < R * D / 4; i += NSG * 32) {
        out[i] = red[i];
    }
"""

_scores_kernel = mx.fast.metal_kernel(
    name="quantized_decode_attention_scores",
    input_names=["q", "kw", "ks", "kb", "scale"],
    output_names=["scores"],
    source=_SCORES_SOURCE,
    ensure_row_contiguous=False,
)

_values_kernel = mx.fast.metal_kernel(
    name="quantized_decode_attention_values",
    input_names=["probs", "vw", "vs", "vb", "bk"],
    output_names=["partial"],
    source=_VALUES_SOURCE,
    ensure_row_contiguous=False,
)

_SCORES_TG = 256
_VALUES_NSG = 4
_VALUES_BLOCKS = 256
# Below this many keys, MLX's quantized_matmul path is faster: the kernels'
# extra launches cost more than the reads they save.
_MIN_KEYS = 1024


def quantized_decode_attention_supported(
    queries, q_keys, q_values, mask, group_size, bits
) -> bool:
    """True when the kernels handle this call: one query position, 6-bit,
    head dim 256 for queries, keys and values, groups of 32/64/128, a whole number of query heads per KV
    head, at least _MIN_KEYS keys, and a mask that is None, a string, or one
    value per (row, key)."""
    if bits != 6 or group_size not in (32, 64, 128):
        return False
    if queries.ndim != 4 or queries.shape[2] != 1 or queries.shape[3] != 256:
        return False
    packed = q_keys[0]
    if packed.ndim != 4 or packed.dtype != mx.uint32 or packed.shape[-1] != 48:
        return False
    v_packed = q_values[0]
    if v_packed.ndim != 4 or v_packed.dtype != mx.uint32 or v_packed.shape[-1] != 48:
        return False
    if packed.shape[2] < _MIN_KEYS:
        return False
    n_q, n_kv = queries.shape[1], packed.shape[1]
    if n_q % n_kv or n_q // n_kv > 16:
        return False
    if mask is None or isinstance(mask, str):
        return True
    # Laid out as the unfused path aligns it: (..., L, T), with a 3-D or 4-D
    # mask's leading axis the batch and a 4-D mask's second axis the heads.
    # Only one value per (row, key) is supported, so per-head masks fall back.
    shape = mask.shape
    if not 1 <= len(shape) <= 4 or shape[-1] != packed.shape[2]:
        return False
    if len(shape) >= 2 and shape[-2] != 1:
        return False
    if len(shape) == 4 and shape[1] != 1:
        return False
    return len(shape) < 3 or shape[0] in (1, queries.shape[0])


def _block_keys(T: int) -> int:
    # About _VALUES_BLOCKS blocks per KV head, each a whole number of 32-key
    # tiles for every SIMD group.
    tile = 32 * _VALUES_NSG
    return -(-T // (_VALUES_BLOCKS * tile)) * tile


def quantized_decode_attention(
    queries: mx.array,
    q_keys: tuple,
    q_values: tuple,
    scale: float,
    mask=None,
    group_size: int = 64,
) -> mx.array:
    """softmax(q K^T * scale + mask) V for queries (B, n_q, 1, 256) over a 6-bit
    cache whose keys and values are (packed, scales, biases), each
    (B, n_kv, T, ...). mask: None, "causal" (a no-op for one query), or an
    array broadcastable to (B, 1, 1, T), boolean or additive. Returns
    (B, n_q, 1, 256) in the queries' dtype."""
    B, n_q, _, D = queries.shape
    kw, ks, kb = q_keys
    vw, vs, vb = q_values
    H, T = kw.shape[1], kw.shape[2]
    R = n_q // H
    template = [("R", R), ("D", D), ("G", group_size), ("H", H)]
    scores = _scores_kernel(
        inputs=[queries, kw, ks, kb, scalar(scale)],
        template=template + [("TG", _SCORES_TG)],
        grid=(-(-T // _SCORES_TG) * _SCORES_TG, H, B),
        threadgroup=(_SCORES_TG, 1, 1),
        output_shapes=[(B, H, R, T)],
        output_dtypes=[mx.float32],
    )[0]
    if mask is not None and not isinstance(mask, str):
        mask = mask.reshape(mask.size // T, 1, 1, T)
        if mask.dtype == mx.bool_:
            # The finite minimum, as the unfused path uses: a fully masked row
            # gets uniform weights rather than NaN.
            scores = mx.where(mask, scores, mx.finfo(mx.float32).min)
        else:
            scores = scores + mask.astype(mx.float32)
    probs = mx.softmax(scores, axis=-1, precise=True)
    BK = _block_keys(T)
    NB = -(-T // BK)
    partial = _values_kernel(
        inputs=[probs, vw, vs, vb, scalar(BK, mx.uint32)],
        template=template + [("NSG", _VALUES_NSG)],
        grid=(NB * 32 * _VALUES_NSG, H, B),
        threadgroup=(32 * _VALUES_NSG, 1, 1),
        output_shapes=[(B, H, NB, R, D)],
        output_dtypes=[mx.float32],
    )[0]
    out = partial.sum(axis=2)
    return out.reshape(B, n_q, 1, D).astype(queries.dtype)
