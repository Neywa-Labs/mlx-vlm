"""Cosmos3-Edge language model: Llama-style decoder with GQA, ReLU^2 MLP and interleaved 3D M-RoPE."""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import (
    LanguageModelOutput,
    cache_position_offset,
    create_attention_mask,
    kv_sequence_length,
    project_logits,
    scaled_dot_product_attention,
)
from .config import ModelConfig, TextConfig


class MRoPE:
    """Interleaved 3D rotary embedding: frequency i uses T, except i % 3 == 1 -> H and i % 3 == 2 -> W
    (for i below 3 * section)."""

    def __init__(self, config: TextConfig):
        dim = config.head_dim
        inv = 1.0 / (
            config.rope_theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim)
        )
        sel = np.zeros((3, dim // 2), dtype=np.float32)
        sel[0] = 1.0
        for axis in (1, 2):
            idx = np.arange(axis, config.mrope_section[axis] * 3, 3)
            sel[0, idx] = 0.0
            sel[axis, idx] = 1.0
        self.inv_sel = mx.array(sel * inv[None, :])

    def __call__(self, position_ids: mx.array, dtype):
        # position_ids: (3, B, L) -> cos, sin: (B, 1, L, head_dim)
        pos = position_ids.astype(mx.float32)
        freqs = (
            pos[0][..., None] * self.inv_sel[0]
            + pos[1][..., None] * self.inv_sel[1]
            + pos[2][..., None] * self.inv_sel[2]
        )
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb).astype(dtype)[:, None], mx.sin(emb).astype(dtype)[:, None]


def rotate_half(x):
    h = x.shape[-1] // 2
    return mx.concatenate([-x[..., h:], x[..., :h]], axis=-1)


class Attention(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        bias = config.attention_bias
        dim = config.hidden_size
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=bias)

    def __call__(self, x, cos, sin, mask=None, cache=None):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        if isinstance(mask, mx.array):
            mask = mask[..., : kv_sequence_length(k)]
        o = scaled_dot_product_attention(q, k, v, cache, scale=self.scale, mask=mask)
        return self.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, -1))


class MLP(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.fc1 = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.fc2 = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_bias
        )

    def __call__(self, x):
        return self.fc2(mx.square(nn.relu(self.fc1(x))))


class DecoderLayer(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = MLP(config)

    def __call__(self, x, cos, sin, mask=None, cache=None):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask, cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class Cosmos3EdgeTextModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rope = MRoPE(config)

    def __call__(
        self, inputs, inputs_embeds=None, mask=None, cache=None, position_ids=None
    ):
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        if cache is None:
            cache = [None] * len(self.layers)
        if mask is None:
            mask = create_attention_mask(h, cache[0])
        cos, sin = self.rope(position_ids, h.dtype)
        for layer, c in zip(self.layers, cache):
            h = layer(h, cos, sin, mask, c)
        return self.norm(h)


def get_rope_index(
    input_ids,
    image_grid_thw=None,
    video_grid_thw=None,
    attention_mask=None,
    image_token_id: int = 19,
    video_token_id: int = 18,
    spatial_merge_size: int = 2,
):
    """Port of Cosmos3EdgeModel.get_rope_index (transformers 5.22).

    Text tokens get sequential positions on all three axes. A vision block starting at position s gets T = s,
    H = s + row and W = s + col (in merged patches), and the next position is s + max(h, w) / merge. Videos are
    split into single frames, since the processor puts a timestamp text block between frames.
    Returns position_ids (3, B, L) and rope_deltas (B, 1) as numpy arrays.
    """
    ids = np.asarray(input_ids)
    B, L = ids.shape
    types = np.zeros_like(ids)
    types[ids == image_token_id] = 1
    types[ids == video_token_id] = 2
    frames = []
    if video_grid_thw is not None:
        for t, h, w in np.asarray(video_grid_thw).tolist():
            frames.extend([[1, h, w]] * int(t))
    grids = {
        1: iter(
            np.asarray(image_grid_thw).tolist() if image_grid_thw is not None else []
        ),
        2: iter(frames),
    }
    pos = np.zeros((3, B, L), dtype=np.int64)
    deltas = []
    for b in range(B):
        keep = (
            np.ones(L, bool)
            if attention_mask is None
            else np.asarray(attention_mask[b]).astype(bool)
        )
        row = types[b][keep].tolist()
        chunks, cur, i = [], 0, 0
        while i < len(row):
            j = i
            while j < len(row) and row[j] == row[i]:
                j += 1
            if row[i] == 0:
                n = j - i
                chunks.append(np.broadcast_to(np.arange(n) + cur, (3, n)))
                cur += n
            else:
                _, gh, gw = next(grids[row[i]])
                lh, lw = gh // spatial_merge_size, gw // spatial_merge_size
                hh, ww = np.meshgrid(np.arange(lh), np.arange(lw), indexing="ij")
                chunks.append(
                    np.stack(
                        [
                            np.full(lh * lw, cur),
                            hh.reshape(-1) + cur,
                            ww.reshape(-1) + cur,
                        ]
                    )
                )
                cur += max(gh, gw) // spatial_merge_size
            i = j
        llm = np.concatenate(chunks, axis=1) if chunks else np.zeros((3, 0), np.int64)
        pos[:, b, keep] = llm
        deltas.append(int(llm.max()) + 1 - int(keep.sum()) if llm.size else 0)
    return pos, np.array(deltas, dtype=np.int64).reshape(B, 1)


class LanguageModel(nn.Module):
    def __init__(self, args: TextConfig, config: ModelConfig = None):
        super().__init__()
        self.args = args
        self.config = config
        self.model_type = args.model_type
        self.model = Cosmos3EdgeTextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self._position_ids = None
        self._rope_deltas = None

    def get_rope_index(
        self, input_ids, image_grid_thw=None, video_grid_thw=None, attention_mask=None
    ):
        cfg = self.config
        pos, deltas = get_rope_index(
            input_ids,
            image_grid_thw,
            video_grid_thw,
            attention_mask,
            image_token_id=cfg.image_token_id if cfg else 19,
            video_token_id=cfg.video_token_id if cfg else 18,
            spatial_merge_size=cfg.vision_config.spatial_merge_size if cfg else 2,
        )
        return mx.array(pos), mx.array(deltas)

    def _positions(self, B, L, offset, position_ids, rope_deltas):
        if rope_deltas is not None:
            self._rope_deltas = rope_deltas
        if position_ids is not None:
            if position_ids.ndim == 2:
                position_ids = mx.broadcast_to(
                    position_ids[None], (3, *position_ids.shape)
                )
            self._position_ids = position_ids
        stored = self._position_ids
        if (
            stored is not None
            and stored.shape[1] == B
            and stored.shape[-1] >= offset + L
        ):
            return stored[..., offset : offset + L]
        delta = (
            self._rope_deltas
            if self._rope_deltas is not None
            else mx.zeros((B, 1), mx.int32)
        )
        if delta.shape[0] < B:
            delta = mx.broadcast_to(delta[:1], (B, 1))
        p = mx.arange(offset, offset + L)[None, :] + delta[:B].astype(mx.int32)
        return mx.broadcast_to(p[None], (3, B, L))

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        if (
            kwargs.get("pixel_values") is not None
            and kwargs.get("position_ids") is None
        ):
            self._position_ids = self._rope_deltas = None
        offset = (
            cache_position_offset(cache[0]) if cache and cache[0] is not None else 0
        )
        ref = inputs_embeds if inputs_embeds is not None else inputs
        B, L = ref.shape[0], ref.shape[1]
        position_ids = self._positions(
            B, L, offset, kwargs.get("position_ids"), kwargs.get("rope_deltas")
        )
        if isinstance(mask, mx.array) and mask.ndim == 2:
            mask = None  # padding masks are folded into position_ids; attention stays causal
        out = self.model(
            inputs,
            inputs_embeds=inputs_embeds,
            mask=mask,
            cache=cache,
            position_ids=position_ids,
        )
        projection = (
            self.model.embed_tokens.as_linear
            if self.args.tie_word_embeddings
            else self.lm_head
        )
        out = project_logits(out, projection, kwargs.get("logits_to_keep"))
        return LanguageModelOutput(logits=out)

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.head_dim

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads
