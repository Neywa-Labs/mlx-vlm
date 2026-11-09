import mlx.core as mx
import mlx.nn as nn

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    project_logits,
    scaled_dot_product_attention,
)
from ..linear import DECODE_BLOCK_SIZE, linear
from ..quantized_verifier import optimized_affine_linear
from .cache import KVCache, RotatingKVCache
from .config import ModelConfig


def _linear(layer, x):
    # The fused affine kernel matches singleton QMV. Native batches can use a
    # different reduction, so preserve their arithmetic with the shared helper.
    if (
        x.ndim == 3
        and x.shape[0] == 1
        and 1 < x.shape[1] <= DECODE_BLOCK_SIZE
        and not getattr(layer, "training", False)
    ):
        output = optimized_affine_linear(layer, x)
        if output is not None:
            return output
    return linear(layer, x)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.gate_full_precision = config.gate_full_precision
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None

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
        decode_block = (
            cache is not None and not self.training and 1 < length <= DECODE_BLOCK_SIZE
        )
        value_states = x
        if value_residual is not None:
            blend = mx.sigmoid(self.value_res_lambda[0]).astype(x.dtype)
            seed = value_residual.astype(x.dtype)
            value_states = seed + blend * (x - seed)

        queries = self.q_norm(
            _linear(self.q_proj, x).reshape(batch, length, self.n_heads, self.head_dim)
        )
        keys = self.k_norm(
            _linear(self.k_proj, x).reshape(
                batch, length, self.n_kv_heads, self.head_dim
            )
        )
        values = _linear(self.v_proj, value_states).reshape(
            batch, length, self.n_kv_heads, self.head_dim
        )
        queries, keys, values = (
            v.transpose(0, 2, 1, 3) for v in (queries, keys, values)
        )

        out = (
            self._attention_block(queries, keys, values, cache)
            if decode_block
            else None
        )
        if decode_block and out is None:
            # Keep one-token RoPE, attention, and physical cache order while
            # sharing projection work across the verification block.
            outputs = []
            for index in range(length):
                token = x[:, index : index + 1]
                token_mask = create_attention_mask(
                    token, cache, window_size=self.sliding_window
                )
                q, k, v = (
                    mx.contiguous(tensor[:, :, index : index + 1])
                    for tensor in (queries, keys, values)
                )
                if self.is_sliding:
                    q = self.rope(q, offset=cache.offset)
                    k = self.rope(k, offset=cache.offset)
                k, v = cache.update_and_fetch(k, v)
                outputs.append(
                    scaled_dot_product_attention(
                        q, k, v, cache=cache, scale=self.scale, mask=token_mask
                    )
                )
            out = mx.concatenate(outputs, axis=2)
        elif not decode_block:
            out = self._attention(queries, keys, values, mask, cache)
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        if hasattr(self, "g_proj"):
            gate = _linear(self.g_proj, x)
            if self.gate_full_precision:
                gate = gate.astype(mx.float32)
            out = out * mx.sigmoid(gate).astype(out.dtype)
        return _linear(self.o_proj, out)

    def _attention_block(self, queries, keys, values, cache):
        fetch = getattr(cache, "fetch_verification_block", None)
        if fetch is None:
            return None
        if self.is_sliding:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
        views = fetch(keys, values)
        if views is None:
            return None
        keys, values, mask = views
        batch, heads, length, dim = queries.shape
        queries = queries.transpose(0, 2, 1, 3).reshape(batch * length, heads, 1, dim)
        keys = keys.reshape(batch * length, *keys.shape[2:])
        values = values.reshape(batch * length, *values.shape[2:])
        mask = mask.reshape(batch * length, 1, 1, mask.shape[-1])
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        return output.reshape(batch, length, heads, dim).transpose(0, 2, 1, 3)

    def _attention(self, queries, keys, values, mask, cache):
        if self.is_sliding:
            offset = cache.offset if cache is not None else 0
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        return scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )


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
        gate, up = _linear(self.gate_proj, x), _linear(self.up_proj, x)
        if self.clamp_threshold:
            gate = mx.minimum(gate, self.clamp_threshold)
            up = mx.clip(up, -self.clamp_threshold, self.clamp_threshold)
        return _linear(self.down_proj, nn.silu(gate) * up)


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

    def __call__(
        self,
        inputs,
        cache=None,
        inputs_embeds=None,
        capture_layer_ids=None,
        hidden_sink=None,
    ):
        capture = set(capture_layer_ids or [])
        if len(capture) != len(capture_layer_ids or []) or any(
            type(index) is not int or not 0 <= index < len(self.layers)
            for index in capture
        ):
            raise ValueError("capture_layer_ids must be unique valid layer indices")
        captured = {}
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        if self.config.embed_scale is not None:
            h = h * self.config.embed_scale
        # The residual is the scaled input embedding, before the first pre-norm.
        value_residual = h if self.config.use_valres else None
        if cache is None:
            cache = [None] * len(self.layers)
        for index, (layer, c) in enumerate(zip(self.layers, cache)):
            window = self.config.sliding_window if layer.self_attn.is_sliding else None
            mask = create_attention_mask(h, c, window_size=window)
            h = layer(h, mask, c, value_residual)
            if index in capture:
                captured[index] = h
        if hidden_sink is not None:
            hidden_sink.extend(captured[index] for index in capture_layer_ids or [])
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
        capture_layer_ids=None,
        **kwargs,
    ) -> LanguageModelOutput:
        if inputs is None:
            inputs = kwargs.get("input_ids")
        if inputs_embeds is None:
            inputs_embeds = input_embeddings
        if kwargs.pop("speculative_verify", False):
            from ...speculative.common import verify_forward

            output, transaction = verify_forward(
                self, inputs, cache, capture_layer_ids=capture_layer_ids
            )
            output.gdn_states = transaction
            return output
        hidden_sink = [] if capture_layer_ids is not None else None
        out = self.model(inputs, cache, inputs_embeds, capture_layer_ids, hidden_sink)
        projection = (
            self.model.embed_tokens.as_linear
            if self.config.tie_word_embeddings
            else self.lm_head
        )
        return LanguageModelOutput(
            logits=project_logits(
                out,
                (
                    projection
                    if self.training
                    else lambda hidden: _linear(projection, hidden)
                ),
                logits_to_keep,
            ),
            hidden_states=hidden_sink,
        )

    def rollback_speculative_cache(self, caches, transaction, accepted, block_size):
        from ...speculative.cache_state import rollback_speculative_cache

        return rollback_speculative_cache(caches, transaction, accepted, block_size)

    def chunked_prefill_policy(
        self, *, draft_model=None, draft_kind=None, prefill_kwargs=None, **kwargs
    ):
        return draft_model is None or (
            draft_kind == "dflash"
            and (prefill_kwargs or {}).get("capture_layer_ids") is not None
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
