import mlx.core as mx
import mlx.nn as nn

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    project_logits,
    scaled_dot_product_attention,
)
from ..cache import KVCache, RotatingKVCache
from .config import ModelConfig


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.gate_full_precision = config.gate_full_precision
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"

        self.q_proj = nn.Linear(
            config.hidden_size, self.n_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        if config.use_valres:
            self.value_res_lambda = mx.zeros((1,))
        if config.use_gated_attention:
            self.g_proj = nn.Linear(
                config.hidden_size, self.n_heads * self.head_dim, bias=False
            )
        if self.is_sliding:
            self.rope = nn.RoPE(
                self.head_dim, traditional=False, base=config.rope_theta
            )

    def __call__(self, x, mask=None, cache=None, value_residual=None):
        batch, length, _ = x.shape
        value_states = x
        if value_residual is not None:
            blend = mx.sigmoid(self.value_res_lambda[0]).astype(x.dtype)
            seed = value_residual.astype(x.dtype)
            value_states = seed + blend * (x - seed)

        queries = self.q_norm(
            self.q_proj(x).reshape(batch, length, self.n_heads, self.head_dim)
        )
        keys = self.k_norm(
            self.k_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        )
        values = self.v_proj(value_states).reshape(
            batch, length, self.n_kv_heads, self.head_dim
        )
        queries, keys, values = (
            v.transpose(0, 2, 1, 3) for v in (queries, keys, values)
        )

        if self.is_sliding:
            offset = cache.offset if cache is not None else 0
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        out = (
            scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask
            )
            .transpose(0, 2, 1, 3)
            .reshape(batch, length, -1)
        )
        if hasattr(self, "g_proj"):
            gate = self.g_proj(x)
            if self.gate_full_precision:
                gate = gate.astype(mx.float32)
            out = out * mx.sigmoid(gate).astype(out.dtype)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.clamp_threshold = config.ffn_clamp_threshold
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, x):
        gate, up = self.gate_proj(x), self.up_proj(x)
        if self.clamp_threshold:
            gate = mx.minimum(gate, self.clamp_threshold)
            up = mx.clip(up, -self.clamp_threshold, self.clamp_threshold)
        return self.down_proj(nn.silu(gate) * up)


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(self, x, mask=None, cache=None, value_residual=None):
        attn = self.self_attn(self.input_layernorm(x), mask, cache, value_residual)
        x = x + self.post_attention_layernorm(attn)
        return x + self.post_feedforward_layernorm(
            self.mlp(self.pre_feedforward_layernorm(x))
        )


class RicursiveModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, inputs, cache=None, inputs_embeds=None):
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        if self.config.embed_scale is not None:
            h = h * self.config.embed_scale
        # The residual is the scaled input embedding, before the first pre-norm.
        value_residual = h if self.config.use_valres else None
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, c in zip(self.layers, cache):
            window = self.config.sliding_window if layer.self_attn.is_sliding else None
            mask = create_attention_mask(h, c, window_size=window)
            h = layer(h, mask, c, value_residual)
        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = RicursiveModel(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs=None,
        cache=None,
        input_embeddings=None,
        inputs_embeds=None,
        logits_to_keep: int = 0,
        **kwargs,
    ) -> LanguageModelOutput:
        if inputs is None:
            inputs = kwargs.get("input_ids")
        if inputs_embeds is None:
            inputs_embeds = input_embeddings
        out = self.model(inputs, cache, inputs_embeds)
        projection = (
            self.model.embed_tokens.as_linear
            if self.config.tie_word_embeddings
            else self.lm_head
        )
        return LanguageModelOutput(
            logits=project_logits(out, projection, logits_to_keep)
        )

    def make_cache(self):
        return [
            (
                RotatingKVCache(max_size=self.config.sliding_window, keep=0)
                if kind == "sliding_attention"
                and self.config.sliding_window is not None
                else KVCache()
            )
            for kind in self.config.layer_types
        ]

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.config.head_dim

    @property
    def n_kv_heads(self):
        return self.config.num_key_value_heads
