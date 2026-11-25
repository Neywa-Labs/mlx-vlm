from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ..bailing_hybrid.language import route_experts
from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from ..cache import ArraysCache, KVCache
from ..gated_delta import gated_delta_update
from ..mla import MultiLinear, latent_length, max_absorbed_queries
from ..mlp import SwiGLUMLP as MLP
from ..qwen3_next.language import Qwen3NextRMSNormGated as RMSNormGated
from ..switch_layers import SwitchGLU
from .config import ModelConfig

# Kept in float32 on conversion: the router scores in float32, and the
# decay parameters of the delta rule are exponentiated.
FLOAT32_PARAMS = ("mlp.gate.weight", "e_score_correction_bias", "A_log", "dt_bias")


def l2_normalize(x: mx.array, eps: float = 1e-12) -> mx.array:
    """x / sqrt(sum(x^2) + eps), with no head_dim**-0.5 readout scale."""
    d = x.shape[-1]
    return d**-0.5 * mx.fast.rms_norm(x, None, eps / d)


class OmoikaneGatedDeltaNet(nn.Module):
    """Gated delta rule with a short causal convolution on q, k and v."""

    def __init__(self, args: ModelConfig):
        super().__init__()
        cfg = args.gdn_config
        self.num_heads = cfg["num_heads"]
        self.head_k_dim = cfg["head_dim_k"]
        self.head_v_dim = cfg["head_dim_v"]
        self.key_dim = self.num_heads * self.head_k_dim
        self.value_dim = self.num_heads * self.head_v_dim
        self.conv_kernel_size = cfg["short_conv_kernel"]
        self.qk_l2norm = cfg["qk_l2norm"]
        dim = args.hidden_size

        self.q_proj = nn.Linear(dim, self.key_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.key_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.value_dim, bias=False)
        self.g_proj = nn.Linear(dim, self.value_dim, bias=False)
        self.a_proj = nn.Linear(dim, self.num_heads, bias=False)
        self.b_proj = nn.Linear(dim, self.num_heads, bias=False)
        self.o_proj = nn.Linear(self.value_dim, dim, bias=False)

        def conv(channels):
            return nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=self.conv_kernel_size,
                groups=channels,
                padding=0,
                bias=cfg["conv_bias"],
            )

        self.q_conv1d = conv(self.key_dim)
        self.k_conv1d = conv(self.key_dim)
        self.v_conv1d = conv(self.value_dim)

        self.A_log = mx.zeros(self.num_heads)
        self.dt_bias = mx.zeros(self.num_heads)
        self.o_norm = RMSNormGated(self.head_v_dim, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, S, _ = x.shape
        H, Dk, Dv = self.num_heads, self.head_k_dim, self.head_v_dim

        qkv = mx.concatenate([self.q_proj(x), self.k_proj(x), self.v_proj(x)], -1)
        if mask is not None:
            if mask.shape[0] != B:
                mask = None
            else:
                qkv = mx.where(mask[..., None], qkv, 0)

        n_keep = self.conv_kernel_size - 1
        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros((B, n_keep, qkv.shape[-1]), dtype=qkv.dtype)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            cache.update_window(0, conv_input, n_keep, lengths=cache.lengths)

        # The delta rule runs in float32, as in training.
        q, k, v = mx.split(conv_input, [self.key_dim, 2 * self.key_dim], axis=-1)
        q = nn.silu(self.q_conv1d(q)).reshape(B, S, H, Dk).astype(mx.float32)
        k = nn.silu(self.k_conv1d(k)).reshape(B, S, H, Dk).astype(mx.float32)
        v = nn.silu(self.v_conv1d(v)).reshape(B, S, H, Dv).astype(mx.float32)
        if self.qk_l2norm:
            q, k = l2_normalize(q), l2_normalize(k)

        out, _ = gated_delta_update(
            q,
            k,
            v,
            self.a_proj(x).astype(mx.float32),
            self.b_proj(x).astype(mx.float32),
            self.A_log,
            self.dt_bias,
            mask=mask,
            use_kernel=not self.training,
            cache=cache,
        )
        if cache is not None:
            cache.advance(S)

        out = self.o_norm(out, self.g_proj(x).reshape(B, S, H, Dv))
        return self.o_proj(out.reshape(B, S, -1).astype(x.dtype))


class OmoikaneAttention(nn.Module):
    """MLA with decoupled RoPE and a sigmoid output gate.

    The cache holds the normalized latent and the shared RoPE key. Unlike
    DeepSeek-V3, RoPE rotates halves (not interleaved pairs), and the output
    is multiplied by sigmoid(o_gate_proj(x)) before o_proj.
    """

    def __init__(self, args: ModelConfig):
        super().__init__()
        cfg = args.mla_config
        self.num_heads = cfg["num_heads"]
        self.q_lora_rank = cfg["q_lora_rank"]
        self.kv_lora_rank = cfg["kv_lora_rank"]
        self.qk_nope_head_dim = cfg["qk_nope_head_dim"]
        self.qk_rope_head_dim = cfg["qk_rope_head_dim"]
        self.v_head_dim = cfg["v_head_dim"]
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scale = self.q_head_dim**-0.5
        dim = args.hidden_size

        self.q_a_proj = nn.Linear(dim, self.q_lora_rank, bias=False)
        self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=args.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=args.rms_norm_eps)
        self.embed_q = MultiLinear(
            self.qk_nope_head_dim, self.kv_lora_rank, self.num_heads
        )
        self.unembed_out = MultiLinear(
            self.kv_lora_rank, self.v_head_dim, self.num_heads
        )
        self._absorbed_dims = (
            self.kv_lora_rank,
            self.qk_nope_head_dim,
            self.v_head_dim,
        )
        self.o_gate_proj = nn.Linear(dim, self.num_heads * self.v_head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, dim, bias=False)

        self.rope = nn.RoPE(
            self.qk_rope_head_dim, traditional=False, base=cfg["rope_theta"]
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)

        offset = cache.offset if cache is not None else 0
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)

        kv_latent = mx.expand_dims(kv_latent, axis=1)
        if cache is not None:
            kv_latent, k_pe = cache.update_and_fetch(kv_latent, k_pe)

        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask,
                pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
            )

        absorbed = L == 1 or L <= max_absorbed_queries(
            *self._absorbed_dims, latent_length(kv_latent)
        )
        if absorbed:
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        output = scaled_dot_product_attention(
            q_nope, k, v, cache=cache, scale=self.scale, mask=pe_scores
        )
        if absorbed:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output * mx.sigmoid(self.o_gate_proj(x)))


class OmoikaneMoEGate(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        cfg = args.moe_config
        self.top_k = cfg["num_experts_per_tok"]
        self.n_group = cfg["group_limited_routing"]["n_group"]
        self.topk_group = cfg["group_limited_routing"]["topk_group"]
        self.norm_topk_prob = cfg["norm_topk_prob"]
        self.routed_scaling_factor = cfg["routed_scaling_factor"]
        n_experts = cfg["n_routed_experts"]
        self.weight = mx.zeros((n_experts, args.hidden_size))
        self.e_score_correction_bias = mx.zeros((n_experts,))

    def __call__(self, x: mx.array):
        # Sigmoid scores in float32; groups outside the best topk_group are
        # excluded with -inf, and the weights are the uncorrected scores.
        return route_experts(
            x,
            self.weight,
            self.e_score_correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.norm_topk_prob,
            self.routed_scaling_factor,
        )


class OmoikaneMoE(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        cfg = args.moe_config
        self.gate = OmoikaneMoEGate(args)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, cfg["moe_intermediate_size"], cfg["n_routed_experts"]
        )
        self.n_shared_experts = cfg["n_shared_experts"]
        if self.n_shared_experts:
            self.shared_experts = MLP(
                args.hidden_size, cfg["moe_intermediate_size"] * self.n_shared_experts
            )

    def __call__(self, x: mx.array) -> mx.array:
        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        if self.n_shared_experts:
            y = y + self.shared_experts(x)
        return y


class OmoikaneDecoderLayer(nn.Module):
    def __init__(self, args: ModelConfig, layer_idx: int):
        super().__init__()
        self.is_linear = layer_idx not in args.mla_layer_indices
        if self.is_linear:
            self.self_attn = OmoikaneGatedDeltaNet(args)
        else:
            self.self_attn = OmoikaneAttention(args)

        if layer_idx in args.dense_layer_indices or not args.moe_config:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)
        else:
            self.mlp = OmoikaneMoE(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class OmoikaneModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            OmoikaneDecoderLayer(args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ssm_idx = next((i for i, l in enumerate(self.layers) if l.is_linear), None)
        self.fa_idx = next(
            (i for i, l in enumerate(self.layers) if not l.is_linear), None
        )

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask = None
        if self.fa_idx is not None:
            # MLA adds the RoPE scores through the mask, so it needs an array.
            fa_mask = create_attention_mask(h, cache[self.fa_idx], return_array=True)
        ssm_mask = None
        if self.ssm_idx is not None:
            ssm_mask = create_ssm_mask(h, cache[self.ssm_idx])

        for layer, c in zip(self.layers, cache):
            h = layer(h, ssm_mask if layer.is_linear else fa_mask, cache=c)

        return self.norm(h)


def _is_mtp_weight(key: str, num_hidden_layers: int) -> bool:
    """The MTP module is stored as the layer after the last one."""
    prefix = "model.layers."
    if not key.startswith(prefix):
        return False
    layer_index = key[len(prefix) :].split(".", 1)[0]
    return layer_index.isdigit() and int(layer_index) >= num_hidden_layers


class LanguageModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = OmoikaneModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        out = self.model(inputs, cache, inputs_embeds=inputs_embeds)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return LanguageModelOutput(logits=out)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [ArraysCache(size=2) if l.is_linear else KVCache() for l in self.layers]

    def sanitize(self, weights):
        args = self.args
        if args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        # Generation does not use the MTP module (it predicts token t+2).
        if any(_is_mtp_weight(k, args.num_hidden_layers) for k in weights):
            weights = {
                k: v
                for k, v in weights.items()
                if not _is_mtp_weight(k, args.num_hidden_layers)
            }

        mla = args.mla_config
        for l in range(args.num_hidden_layers):
            prefix = f"model.layers.{l}"

            # Stacked experts, with gate and up in one tensor.
            if f"{prefix}.mlp.experts.gate_up_proj" in weights:
                gate_up = weights.pop(f"{prefix}.mlp.experts.gate_up_proj")
                gate, up = mx.split(gate_up, 2, axis=1)
                weights[f"{prefix}.mlp.switch_mlp.gate_proj.weight"] = gate
                weights[f"{prefix}.mlp.switch_mlp.up_proj.weight"] = up
            if f"{prefix}.mlp.experts.down_proj" in weights:
                weights[f"{prefix}.mlp.switch_mlp.down_proj.weight"] = weights.pop(
                    f"{prefix}.mlp.experts.down_proj"
                )

            # Split kv_b_proj into per-head key and value up-projections.
            if f"{prefix}.self_attn.kv_b_proj.weight" in weights:
                kv_b = weights.pop(f"{prefix}.self_attn.kv_b_proj.weight")
                nope = mla["qk_nope_head_dim"]
                kv_b = kv_b.reshape(mla["num_heads"], nope + mla["v_head_dim"], -1)
                weights[f"{prefix}.self_attn.embed_q.weight"] = mx.contiguous(
                    kv_b[:, :nope, :].swapaxes(-1, -2)
                )
                weights[f"{prefix}.self_attn.unembed_out.weight"] = mx.contiguous(
                    kv_b[:, nope:, :]
                )

        # Depthwise convolutions: [channels, 1, kernel] -> [channels, kernel, 1].
        for k, v in weights.items():
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
        return weights

    @property
    def cast_predicate(self):
        def predicate(k):
            return not k.endswith(FLOAT32_PARAMS)

        return predicate
