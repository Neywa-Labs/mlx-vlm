import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures
from ..qwen3_5 import Model as QwenDenseModel
from ..qwen3_5.language import LanguageModel as DenseLanguageModel
from ..qwen3_5.vision import VisionModel
from ..qwen3_5_moe import Model as QwenMoEModel
from ..qwen3_5_moe.language import LanguageModel as MoELanguageModel
from ..qwen3_vl.qwen3_vl import masked_scatter
from .audio import AudioEncoder, sanitize_audio_weights
from .config import ModelConfig


class Model(QwenDenseModel):
    def __init__(self, config: ModelConfig):
        nn.Module.__init__(self)
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.is_moe = getattr(config.text_config, "num_experts", 0) > 0
        language_cls = MoELanguageModel if self.is_moe else DenseLanguageModel
        self.language_model = language_cls(config.text_config, config)
        self.audio_tower = AudioEncoder(config.audio_config)

    def get_input_embeddings(self, input_ids, pixel_values=None, **kwargs):
        embeddings = self.language_model.model.embed_tokens(input_ids)

        def insert(features, token_id, modality):
            nonlocal embeddings
            mask = input_ids == token_id
            count = int(mask.sum().item())
            features = features.reshape(-1, embeddings.shape[-1])
            if count != features.shape[0]:
                raise ValueError(
                    f"{modality} features and tokens do not match: {features.shape[0]} features, {count} tokens."
                )
            embeddings = masked_scatter(
                embeddings,
                mx.broadcast_to(mask[..., None], embeddings.shape),
                features.astype(embeddings.dtype),
            )

        for values, grid, token, name in (
            (
                pixel_values,
                kwargs.get("image_grid_thw"),
                self.config.image_token_id,
                "Image",
            ),
            (
                kwargs.get("pixel_values_videos"),
                kwargs.get("video_grid_thw"),
                self.config.video_token_id,
                "Video",
            ),
        ):
            if values is not None:
                dtype = self.vision_tower.patch_embed.proj.weight.dtype
                features, _ = self.vision_tower(values.astype(dtype), grid)
                insert(features, token, name)

        audio = kwargs.get("audio_feature")
        if audio is not None:
            # Bound Whisper/Q-Former activation memory for long recordings.
            outputs = []
            for segment in audio:
                features = self.audio_tower(
                    segment[None].astype(self.audio_tower.conv1.weight.dtype)
                )
                mx.eval(features)
                outputs.append(features)
            insert(mx.concatenate(outputs), self.config.audio_token_id, "Audio")

        positions, deltas = self.language_model.get_rope_index(
            input_ids,
            kwargs.get("image_grid_thw"),
            kwargs.get("video_grid_thw"),
            kwargs.get("attention_mask", kwargs.get("mask")),
        )
        return InputEmbeddingsFeatures(
            inputs_embeds=embeddings, position_ids=positions, rope_deltas=deltas
        )

    def sanitize(self, weights):
        weights = sanitize_audio_weights(weights)
        base = QwenMoEModel if self.is_moe else QwenDenseModel
        return base.sanitize(self, weights)
