from dataclasses import dataclass, field
from typing import Optional

from ..base import BaseModelConfig
from ..qwen3_5.config import ModelConfig as QwenConfig
from ..qwen3_5.config import TextConfig as DenseTextConfig
from ..qwen3_5.config import VisionConfig
from ..qwen3_5_moe.config import TextConfig as MoETextConfig
from ..qwen3_vl.config import _config_kwargs


@dataclass
class QFormerConfig(BaseModelConfig):
    hidden_size: int = 4104
    encoder_width: int = 1280
    intermediate_size: int = 12288
    num_attention_heads: int = 12
    num_hidden_layers: int = 2
    query_length: int = 1
    out_hidden_size: int = 2048
    cross_attention_freq: int = 1
    layer_norm_eps: float = 1e-12


@dataclass
class AudioConfig(BaseModelConfig):
    model_type: str = "whisper"
    d_model: int = 1280
    num_mel_bins: int = 128
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    encoder_layers: int = 32
    max_source_positions: int = 1500
    second_per_window: float = 0.1
    second_stride: float = 0.1
    layer_norm_eps: float = 1e-5
    qformer_config: QFormerConfig = field(default_factory=QFormerConfig)

    def __post_init__(self):
        if isinstance(self.qformer_config, dict):
            self.qformer_config = QFormerConfig.from_dict(self.qformer_config)
        if self.window_size < 1 or self.window_stride < 1:
            raise ValueError(
                "Audio window and stride must span at least one encoder frame."
            )
        if self.window_size > self.max_source_positions:
            raise ValueError("Audio window exceeds the encoder sequence length.")

    @property
    def window_size(self):
        return round(50 * self.second_per_window)

    @property
    def window_stride(self):
        return round(50 * self.second_stride)

    @property
    def tokens_per_segment(self):
        windows = (
            self.max_source_positions - self.window_size
        ) // self.window_stride + 1
        return windows * self.qformer_config.query_length


class TextConfig:
    @classmethod
    def from_dict(cls, params):
        config_cls = MoETextConfig if params.get("num_experts", 0) else DenseTextConfig
        return config_cls.from_dict(params)


@dataclass
class ModelConfig(QwenConfig):
    audio_config: Optional[AudioConfig] = None
    audio_token_id: int = 248076
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054

    @classmethod
    def from_dict(cls, params):
        params = dict(params)
        for name, config_cls in (
            ("text", TextConfig),
            ("vision", VisionConfig),
            ("audio", AudioConfig),
        ):
            key = f"{name}_config"
            if isinstance(params.get(key), dict):
                params[key] = config_cls.from_dict(params[key])
        return cls(**_config_kwargs(cls, params))
