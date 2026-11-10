"""Block projections with the reduction geometry of single-step decoding.

The batch size selects MLX's QMV/GEMV reduction. Tiling the batch and time
positions together reuses weights without changing that accumulation order.
"""

from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn

_POSITION = r"""
const int lane = thread_index_in_simdgroup;
const int kl = lane % L;
const int row = (threadgroup_position_in_grid.y * SG * (32 / L) +
    simdgroup_index_in_threadgroup * (32 / L) + lane / L) * R;
const int r = min(row, N - 1);
const int first = threadgroup_position_in_grid.x * V;
float result[R * V] = {0};
"""

_STORE = r"""
for (int rv = 0; rv < R * V; ++rv) {
    const int rr = rv / V;
    const int v = rv % V;
    if constexpr (SINGLE && QUANTIZED) {
        result[rv] = simd_sum(result[rv]);
    } else {
        #pragma unroll
        for (int step = L / 2; step >= 1; step /= 2)
            result[rv] += simd_shuffle_down(result[rv], step);
    }
    if (kl == 0 && row + rr < N && first + v < M)
        out[(first + v) * N + row + rr] = O(result[rv]);
}
"""


@lru_cache(maxsize=1)
def _nvfp4_kernel():
    return mx.fast.metal_kernel(
        name="nvfp4_verify_qmv",
        input_names=["x", "weight", "scales"],
        output_names=["out"],
        source=_POSITION + r"""
        for (int g = kl; g < K / 16; g += L) {
            float scale[R];
            float4 decoded[R][4];
            #pragma unroll
            for (int rr = 0; rr < R; ++rr) {
                int wr = min(row + rr, N - 1);
                const device ushort* w = (const device ushort*)weight + wr * K / 4;
                uchar bits = scales[wr * (K / 16) + g];
                int exponent = (bits >> 3) & 15;
                float s = exponent == 0 ? float(bits & 7) / 512.0f :
                    ldexp(float(8 + (bits & 7)), exponent - 10);
                if (bits & 128) s = -s;
                if ((bits & 127) == 127) s = NAN;
                scale[rr] = s;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    ushort packed = w[g * 4 + j];
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        int q = (packed >> (i * 4)) & 15;
                        // E2M1 becomes an exact half after an exponent shift.
                        ushort h = ((q & 8) << 12) | ((q & 7) << 9);
                        decoded[rr][j][i] = float(as_type<half>(h) * half(16384));
                    }
                }
            }
            #pragma unroll
            for (int v = 0; v < V; ++v) {
                const device float4* xv = (const device float4*)x +
                    min(first + v, M - 1) * (K / 4) + g * 4;
                float acc[R] = {0};
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float4 a = xv[j];
                    #pragma unroll
                    for (int rr = 0; rr < R; ++rr) {
                        float4 d = decoded[rr][j];
                        if constexpr (SINGLE)
                            acc[rr] += a[0] * d[0] + a[1] * d[1] +
                                a[2] * d[2] + a[3] * d[3];
                        else
                            acc[rr] += dot(d, a);
                    }
                }
                #pragma unroll
                for (int rr = 0; rr < R; ++rr) {
                    if constexpr (SINGLE) {
                        // QMV rounds the scaled group dot before adding it;
                        // a single expression would contract this into FMA.
                        float scaled = scale[rr] * acc[rr];
                        result[rr * V + v] += scaled;
                    } else {
                        result[rr * V + v] += scale[rr] * acc[rr];
                    }
                }
            }
        }
        """ + _STORE,
    )


@lru_cache(maxsize=1)
def _dense_kernel():
    return mx.fast.metal_kernel(
        name="dense_verify_gemv",
        input_names=["x", "weight"],
        output_names=["out"],
        source=_POSITION + r"""
        const device vec<T, 4>* w = (const device vec<T, 4>*)weight + r * K / 4;
        const device vec<T, 4>* x4 = (const device vec<T, 4>*)x;
        constexpr int U = SINGLE ? 1 : 8;
        for (int base = 0; base < K / 4; base += L * U) {
            float4 decoded[U];
            #pragma unroll
            for (int i = 0; i < U; ++i)
                decoded[i] = float4(w[base + i * L + kl]);
            #pragma unroll
            for (int v = 0; v < V; ++v) {
                float acc = 0;
                #pragma unroll
                for (int i = 0; i < U; ++i) {
                    float4 a = float4(x4[min(first + v, M - 1) * (K / 4) +
                        base + i * L + kl]);
                    if constexpr (SINGLE) {
                        #pragma unroll
                        for (int j = 0; j < 4; ++j)
                            result[v] += decoded[i][j] * a[j];
                    } else {
                        acc += dot(decoded[i], a);
                    }
                }
                if constexpr (!SINGLE) result[v] += acc;
            }
        }
        """ + _STORE,
    )


def native_block_linear(layer, x):
    """Share dense/NVFP4 weights across positions without switching to GEMM."""
    from ..quantization.nvfp4 import ScaledQuantizedLinear

    if (
        type(layer) not in (nn.Linear, ScaledQuantizedLinear)
        or layer.training
        or x.ndim != 3
        or not 1 <= x.shape[0] <= 8
        or not 2 <= x.shape[1] <= 8
        or x.dtype not in (mx.bfloat16, mx.float16, mx.float32)
        or not mx.metal.is_available()
        or mx.default_device() != mx.gpu
    ):
        return None
    b, t, k = x.shape
    n = layer.weight.shape[0]
    quantized = isinstance(layer, ScaledQuantizedLinear)
    if quantized:
        if (
            layer.mode != "nvfp4"
            or layer.bits != 4
            or layer.group_size != 16
            or layer._quantize_activations
            or k % 512
            or k != layer.scales.shape[-1] * 16
        ):
            return None
        projected = x.astype(mx.float32)
        if "pre_quant_scale" in layer:
            projected *= layer.pre_quant_scale.astype(mx.float32)
        inputs = [projected, layer.weight, layer.scales]
        lanes = 32 if b == 1 else 16
        kernel = _nvfp4_kernel()
        dtype = mx.float32
    else:
        # Match the native GEMV/GEMV-wide dispatch, including its eight-step
        # reduction for multiple batch rows.
        if (
            "bias" in layer
            or x.dtype not in (mx.bfloat16, mx.float16)
            or layer.weight.dtype != x.dtype
            or k != layer.weight.shape[-1]
            or k % (128 if b == 1 else 1024)
        ):
            return None
        if k >= 16 * n:
            return None
        architecture = mx.device_info().get("architecture", "").split("_g")[-1][:2]
        if b > 1 and (not architecture.isdigit() or int(architecture) < 15):
            return None
        inputs = [x, layer.weight]
        lanes = 32 if b <= 5 or n <= 64 else 16
        kernel = _dense_kernel()
        dtype = x.dtype
    vectors = min(b * t, 5)
    groups = 4
    row_tile = 2 if quantized and (b == 1 or k > n) else 1
    rows = groups * (32 // lanes) * row_tile
    output = kernel(
        inputs=inputs,
        template=[
            ("T", x.dtype),
            ("O", dtype),
            ("K", k),
            ("N", n),
            ("M", b * t),
            ("V", vectors),
            ("SG", groups),
            ("L", lanes),
            ("R", row_tile),
            ("SINGLE", b == 1),
            ("QUANTIZED", quantized),
        ],
        grid=(
            32 * ((b * t + vectors - 1) // vectors),
            groups * ((n + rows - 1) // rows),
            1,
        ),
        threadgroup=(32, groups, 1),
        output_shapes=[(b, t, n)],
        output_dtypes=[dtype],
    )[0]
    if quantized:
        output = layer.apply_scale(output, layer.global_scale).astype(x.dtype)
    return output + layer.bias if "bias" in layer else output
