"""SigLIP2-style vision tower of Cosmos3-Edge and its 2x2 patch merger.

Patches arrive packed (one row per 16x16 patch, all images and frames concatenated), with `grid_thw` giving each
image's or video's (t, h, w) in patches. The learned 16x16 position grid is resized to every (h, w) with torch's
antialiased bilinear filter (F.interpolate(..., antialias=True)), then put in 2x2 block order to match the patches.
Attention never crosses an image or video-frame boundary.
"""

import math
from functools import lru_cache
from typing import List

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import ModelConfig, VisionConfig


def _aa_bilinear_weights(in_size: int, out_size: int) -> np.ndarray:
    """(out_size, in_size) weights of torch's bilinear interpolation with antialias=True, align_corners=False."""
    scale = in_size / out_size
    support = scale if scale >= 1.0 else 1.0
    invscale = 1.0 / scale if scale >= 1.0 else 1.0
    w = np.zeros((out_size, in_size), dtype=np.float64)
    for i in range(out_size):
        center = scale * (i + 0.5)
        xmin = max(int(center - support + 0.5), 0)
        xmax = min(int(center + support + 0.5), in_size)
        taps = []
        for j in range(xmax - xmin):
            x = abs((j + xmin - center + 0.5) * invscale)
            taps.append(1.0 - x if x < 1.0 else 0.0)
        total = sum(taps)
        for j, t in enumerate(taps):
            w[i, xmin + j] = t / total if total != 0 else 0.0
    return w.astype(np.float32)


@lru_cache(maxsize=256)
def _resize_matrices(grid: int, h: int, w: int):
    return (
        mx.array(_aa_bilinear_weights(grid, h)),
        mx.array(_aa_bilinear_weights(grid, w)),
    )


def gelu_tanh(x):
    return (
        0.5
        * x
        * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * mx.power(x, 3))))
    )


class VisionEmbeddings(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.patch_embedding = nn.Linear(
            config.num_channels * config.patch_size * config.patch_size,
            config.hidden_size,
        )
        self.grid = int(config.num_patches**0.5)
        self.position_embedding = nn.Embedding(config.num_patches, config.hidden_size)

    def positions(self, grid_thw: List[List[int]]) -> mx.array:
        m = self.config.spatial_merge_size
        table = self.position_embedding.weight.reshape(self.grid, self.grid, -1)
        table = table.astype(mx.float32)
        chunks = []
        for t, h, w in grid_thw:
            wh, ww = _resize_matrices(self.grid, h, w)
            pe = mx.einsum("hi,ijc,wj->hwc", wh, table, ww)
            pe = pe.astype(self.position_embedding.weight.dtype)
            pe = pe.reshape(h // m, m, w // m, m, -1).transpose(0, 2, 1, 3, 4)
            pe = pe.reshape(h * w, -1)
            chunks.append(mx.tile(pe, (t, 1)) if t > 1 else pe)
        return mx.concatenate(chunks, axis=0)

    def __call__(self, pixel_values: mx.array, grid_thw: List[List[int]]) -> mx.array:
        x = self.patch_embedding(pixel_values.astype(self.patch_embedding.weight.dtype))
        return x + self.positions(grid_thw)


class VisionAttention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.scale = (dim // self.num_heads) ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        S, L, D = x.shape
        q = self.q_proj(x).reshape(S, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(S, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(S, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.out_proj(o.transpose(0, 2, 1, 3).reshape(S, L, D))


class VisionMLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x):
        return self.fc2(gelu_tanh(self.fc1(x)))


class VisionEncoderLayer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = VisionAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = VisionMLP(config)

    def __call__(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))


class VisionEncoder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layers = [
            VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.embeddings = VisionEmbeddings(config)
        self.encoder = VisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def __call__(
        self, pixel_values: mx.array, grid_thw, max_patches_per_batch: int = 16384
    ):
        grid = [list(map(int, g)) for g in np.asarray(grid_thw).tolist()]
        h = self.embeddings(pixel_values, grid)
        # One segment per image or video frame; segments of equal length run as one batch.
        segments, start = [], 0
        for t, gh, gw in grid:
            for _ in range(t):
                segments.append((start, gh * gw))
                start += gh * gw
        by_length = {}
        for i, (_, n) in enumerate(segments):
            by_length.setdefault(n, []).append(i)
        out = [None] * len(segments)
        for n, idxs in by_length.items():
            per = max(1, max_patches_per_batch // n)
            for c in range(0, len(idxs), per):
                part = idxs[c : c + per]
                x = mx.stack([h[segments[i][0] : segments[i][0] + n] for i in part])
                y = self.post_layernorm(self.encoder(x))
                mx.eval(y)
                for j, i in enumerate(part):
                    out[i] = y[j]
        return mx.concatenate(out, axis=0)


class PatchMerger(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        v = config.vision_config
        self.m2 = v.spatial_merge_size**2
        self.in_dim = v.hidden_size
        self.norm = nn.LayerNorm(v.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(
            v.hidden_size * self.m2, config.projector_hidden_size
        )
        self.linear_fc2 = nn.Linear(
            config.projector_hidden_size, config.text_config.hidden_size
        )

    def __call__(self, x):
        x = self.norm(x.reshape(-1, self.m2, self.in_dim)).reshape(
            -1, self.m2 * self.in_dim
        )
        return self.linear_fc2(nn.gelu(self.linear_fc1(x)))
