"""NVIDIA Cosmos3-Edge reasoner (text, image and video in, text out).

Reference: transformers `Cosmos3EdgeForConditionalGeneration`. The original nvidia/Cosmos3-Edge checkpoint also holds
the video generator; those tensors are dropped in `sanitize`, so only the reasoner is loaded.
"""

import re
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import InputEmbeddingsFeatures
from .config import ModelConfig
from .language import LanguageModel
from .vision import PatchMerger, VisionModel

# Generator-only tensors stored in the same shards as the reasoner.
_GENERATOR_KEYS = [
    re.compile(p)
    for p in [
        r"^(?:model\.language_model\.)?action_modality_embed$",
        r"^(?:model\.language_model\.)?action_proj_(?:in|out)\.",
        r"^(?:model\.language_model\.)?time_embedder\.",
        r"^(?:model\.language_model\.)?proj_(?:in|out)\.",
        r"^(?:model\.language_model\.)?norm_moe_gen\.",
        r"^(?:model\.language_model\.)?layers\.\d+\.(?:input|post_attention)_layernorm_moe_gen\.",
        r"^(?:model\.language_model\.)?layers\.\d+\.mlp_moe_gen\.",
        r"^(?:model\.language_model\.)?layers\.\d+\.self_attn\."
        r"(?:add_[qkv]_proj|to_add_out|norm_added_[qk]|k_norm_und_for_gen)\.",
    ]
]

# Original checkpoint names -> transformers names.
_RAW_RENAMES = [
    (re.compile(r"^embed_tokens\."), "model.language_model.embed_tokens."),
    (re.compile(r"^norm\."), "model.language_model.norm."),
    (re.compile(r"^layers\."), "model.language_model.layers."),
    (re.compile(r"\.self_attn\.to_q\."), ".self_attn.q_proj."),
    (re.compile(r"\.self_attn\.to_k\."), ".self_attn.k_proj."),
    (re.compile(r"\.self_attn\.to_v\."), ".self_attn.v_proj."),
    (re.compile(r"\.self_attn\.to_out\."), ".self_attn.o_proj."),
    (re.compile(r"\.mlp\.up_proj\."), ".mlp.fc1."),
    (re.compile(r"\.mlp\.down_proj\."), ".mlp.fc2."),
]

# transformers names -> mlx-vlm module paths.
_PREFIXES = [
    ("model.language_model.", "language_model.model."),
    ("model.visual.", "vision_tower."),
    ("model.projector.", "multi_modal_projector."),
    ("lm_head.", "language_model.lm_head."),
]


def _scatter(
    embeds: mx.array, ids: np.ndarray, token: int, features: mx.array
) -> mx.array:
    positions = np.flatnonzero(ids.reshape(-1) == token)
    if len(positions) != features.shape[0]:
        raise ValueError(
            f"{len(positions)} placeholder tokens but {features.shape[0]} vision features"
        )
    B, L, D = embeds.shape
    flat = embeds.reshape(B * L, D)
    flat[mx.array(positions.astype(np.uint32))] = features.astype(embeds.dtype)
    return flat.reshape(B, L, D)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.vision_tower = VisionModel(config.vision_config)
        self.multi_modal_projector = PatchMerger(config)
        self.language_model = LanguageModel(config.text_config, config)

    def vision_features(self, pixel_values, grid_thw):
        dtype = self.vision_tower.embeddings.patch_embedding.weight.dtype
        return self.multi_modal_projector(
            self.vision_tower(pixel_values.astype(dtype), grid_thw)
        )

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        image_grid_thw = kwargs.get("image_grid_thw")
        video_grid_thw = kwargs.get("video_grid_thw")
        pixel_values_videos = kwargs.get("pixel_values_videos")
        mask = kwargs.get("mask")

        embeds = self.language_model.model.embed_tokens(input_ids)
        ids = np.array(input_ids)
        if pixel_values is not None:
            embeds = _scatter(
                embeds,
                ids,
                self.config.image_token_id,
                self.vision_features(pixel_values, image_grid_thw),
            )
        if pixel_values_videos is not None:
            embeds = _scatter(
                embeds,
                ids,
                self.config.video_token_id,
                self.vision_features(pixel_values_videos, video_grid_thw),
            )
        position_ids, rope_deltas = self.language_model.get_rope_index(
            ids,
            image_grid_thw if pixel_values is not None else None,
            video_grid_thw if pixel_values_videos is not None else None,
            np.array(mask) if mask is not None else None,
        )
        if pixel_values is None and pixel_values_videos is None:
            # Text only: the three M-RoPE axes are equal, so hand back the (batch, length) form other mRoPE models use.
            position_ids = position_ids[0]
        return InputEmbeddingsFeatures(
            inputs_embeds=embeds, position_ids=position_ids, rope_deltas=rope_deltas
        )

    @property
    def layers(self):
        return self.language_model.model.layers

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        features = self.get_input_embeddings(
            input_ids, pixel_values, mask=mask, **kwargs
        )
        kwargs.update({"pixel_values": pixel_values, **features.to_dict()})
        kwargs.pop("inputs_embeds")
        return self.language_model(
            input_ids,
            inputs_embeds=features.inputs_embeds,
            mask=mask,
            cache=cache,
            **kwargs,
        )

    @staticmethod
    def sanitize(weights):
        """Accepts the original checkpoint, transformers-style names, or an already converted mlx-vlm model."""
        out = {}
        for key, value in weights.items():
            if any(p.search(key) for p in _GENERATOR_KEYS):
                continue
            for pattern, repl in _RAW_RENAMES:
                key = pattern.sub(repl, key)
            if any(p.search(key) for p in _GENERATOR_KEYS):
                continue
            for old, new in _PREFIXES:
                if key.startswith(old):
                    key = new + key[len(old) :]
                    break
            out[key] = value
        return out
