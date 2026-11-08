import inspect
from dataclasses import dataclass, field
from typing import List, Optional, Union

from ..base import BaseModelConfig


def _kwargs(cls, params):
    return {k: v for k, v in params.items() if k in inspect.signature(cls).parameters}


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "cosmos3_edge_text"
    vocab_size: int = 131072
    hidden_size: int = 2048
    intermediate_size: int = 9216
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: Optional[int] = 8
    head_dim: Optional[int] = 128
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1e8
    mrope_section: List[int] = field(default_factory=lambda: [24, 20, 20])
    hidden_act: str = "relu2"
    attention_bias: bool = False
    mlp_bias: bool = False
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.hidden_act != "relu2":
            raise ValueError(f"Unsupported hidden_act {self.hidden_act!r}")
        if (
            len(self.mrope_section) != 3
            or sum(self.mrope_section) != self.head_dim // 2
        ):
            raise ValueError(
                f"mrope_section {self.mrope_section} does not match head_dim {self.head_dim}"
            )

    @classmethod
    def from_dict(cls, params):
        params = dict(params or {})
        rope = params.pop("rope_parameters", None) or params.get("rope_scaling") or {}
        if "rope_theta" in rope:
            params.setdefault("rope_theta", rope["rope_theta"])
        if "mrope_section" in rope:
            params.setdefault("mrope_section", list(rope["mrope_section"]))
        return cls(**_kwargs(cls, params))


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "cosmos3_edge_vision"
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_channels: int = 3
    patch_size: int = 16
    layer_norm_eps: float = 1e-6
    num_patches: int = 256
    spatial_merge_size: int = 2
    hidden_act: str = "gelu_pytorch_tanh"

    def __post_init__(self):
        if self.hidden_act not in ("gelu_pytorch_tanh", "gelu_tanh"):
            raise ValueError(f"Unsupported vision hidden_act {self.hidden_act!r}")


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = field(default_factory=TextConfig)
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    model_type: str = "cosmos3_edge"
    projector_hidden_size: int = 11520
    image_token_id: int = 19
    video_token_id: int = 18
    vision_start_token_id: int = 20
    vision_end_token_id: int = 21
    image_token_index: Optional[int] = None
    video_token_index: Optional[int] = None
    eos_token_id: Optional[Union[int, List[int]]] = 11
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id
        if self.video_token_index is None:
            self.video_token_index = self.video_token_id

    @classmethod
    def from_dict(cls, params):
        params = dict(params)
        text = params.get("text_config") or {}
        if isinstance(text, dict):
            text = dict(text)
            text.setdefault(
                "tie_word_embeddings", params.get("tie_word_embeddings", False)
            )
            params["text_config"] = TextConfig.from_dict(text)
        vision = params.get("vision_config") or {}
        if isinstance(vision, dict):
            params["vision_config"] = VisionConfig.from_dict(vision)
        if "eos_token_id" not in params and isinstance(text, TextConfig):
            params["eos_token_id"] = 11
        return cls(**_kwargs(cls, params))
