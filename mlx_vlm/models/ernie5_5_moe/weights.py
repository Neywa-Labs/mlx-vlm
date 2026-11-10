# Copyright © 2026 Neywa Labs Inc.

"""Released block-FP8 and MLX checkpoint normalization."""

import re

import mlx.core as mx


def dequantize_fp8(weight, scales, block_size=(128, 128)):
    """Decode E4M3FN bytes with floating point or UE8M0 block scales."""
    from ...quantization.checkpoint import _E4M3_DECODE_LUT

    if weight.dtype == mx.uint8:
        weight = _E4M3_DECODE_LUT[weight]
    else:
        weight = weight.astype(mx.float32)
    if scales.dtype == mx.uint8:
        scales = mx.power(2.0, scales.astype(mx.float32) - 127)
    else:
        scales = scales.astype(mx.float32)
    rows, cols = weight.shape[-2:]
    br, bc = block_size
    if rows % br == 0 and cols % bc == 0:
        blocked = weight.reshape(*weight.shape[:-2], rows // br, br, cols // bc, bc)
        weight = (blocked * scales[..., :, None, :, None]).reshape(weight.shape)
    else:
        scale = mx.repeat(mx.repeat(scales, br, axis=-2), bc, axis=-1)
        weight = weight * scale[..., :rows, :cols]
    return weight.astype(mx.bfloat16)


def sanitize(weights, config):
    weights = {k.removeprefix("language_model."): v for k, v in weights.items()}
    out = {}
    for key, value in weights.items():
        match = re.search(r"model\.layers\.(\d+)\.", key)
        if match and int(match[1]) >= config.num_hidden_layers:
            continue  # ECHO is not part of the next-token backbone.
        if key.endswith("_scale_inv"):
            if key.removesuffix("_scale_inv") not in weights:
                raise ValueError(f"FP8 scale without its weight: {key}")
            continue
        scale_key = key + "_scale_inv"
        if scale_key in weights:
            value = dequantize_fp8(value, weights[scale_key])
        elif value.dtype == mx.uint8:
            raise ValueError(f"FP8 weight without its scale: {key}")

        if key == "model.mhc_contract.hc_head_fn":
            value = value.T
        key = key.replace("model.lm_head.", "multimax.")
        key = key.replace("model.mhc_contract.hc_head_", "model.hc_head.")
        for src, dst in (("self_attention", "attn"), ("mlp", "ffn")):
            prefix = f".{src}_hyper_connection."
            if prefix in key:
                key = key.replace(prefix, f".{dst}_hc.")
                if key.endswith("mapping_proj.weight"):
                    key = key.replace("mapping_proj.weight", "fn")
                    value = value.T
                key = key.replace(f".{dst}_hc.bias", f".{dst}_hc.base")
        remap = {
            ".input_layernorm.": ".attn_norm.",
            ".post_attention_layernorm.": ".ffn_norm.",
            ".block_sparse_moe.": ".ffn.",
            ".mlp.": ".ffn.",
            ".self_attn.": ".attn.",
            ".core_attention.": ".",
            ".linear_q_down_proj.": ".wq_a.",
            ".linear_q_up_proj.": ".wq_b.",
            ".linear_kv_proj.": ".wkv.",
            ".q_layernorm.": ".q_norm.",
            ".kv_layernorm.": ".kv_norm.",
            ".compressor.linear_wkv.": ".compressor.wkv.",
            ".compressor.linear_wgate.": ".compressor.wgate.",
            ".attn.gate_proj.": ".attn.o_gate.",
            ".attn.o_proj.": ".attn.wo_b.",
            ".attn.vha_postmix_U": ".attn.vha.U",
            ".attn.vha_postmix_V": ".attn.vha.V",
            ".attn.softmax_offset": ".attn.attn_sink",
        }
        for src, dst in remap.items():
            key = key.replace(src, dst)
        if key.endswith(".attn.linear_o_group_proj"):
            key = key.replace(".linear_o_group_proj", ".wo_a.weight")
            value = value.reshape(config.o_groups, config.o_lora_rank, -1)
        for src, dst in (("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")):
            key = key.replace(f".shared_experts.{src}.", f".shared_experts.{dst}.")
        out[key] = value

    for i in range(config.num_hidden_layers):
        prefix = f"model.layers.{i}"
        for kind in ("attn", "ffn"):
            keys = [
                f"{prefix}.{kind}_hc.alpha_{part}" for part in ("pre", "post", "res")
            ]
            if keys[0] in out:
                out[f"{prefix}.{kind}_hc.scale"] = mx.concatenate(
                    [out.pop(k).reshape(-1) for k in keys]
                )
        for src, dst in (("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")):
            for suffix in ("weight", "scales", "biases"):
                ep = f"{prefix}.ffn.experts"
                if f"{ep}.0.{src}.{suffix}" in out:
                    out[f"{prefix}.ffn.switch_mlp.{dst}.{suffix}"] = mx.stack(
                        [
                            out.pop(f"{ep}.{e}.{src}.{suffix}")
                            for e in range(config.n_routed_experts)
                        ]
                    )
    return out
