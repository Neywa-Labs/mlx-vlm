from .config import ModelConfig, TextConfig, VisionConfig
from .language import LanguageModel
from .mistral_large4 import Model


def validate_sharding(config, world_size):
    """Expert parallelism permits uneven partitions, unlike tensor parallelism."""
    text = config.get("text_config") or config
    experts = text.get("n_routed_experts", TextConfig.n_routed_experts)
    if world_size < 1 or world_size > experts:
        raise ValueError(
            f"Select between 1 and {experts} workers for this expert-parallel model"
        )
