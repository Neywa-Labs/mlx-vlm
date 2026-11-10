# Copyright © 2026 Neywa Labs Inc.

"""Select the bundled ECHO auxiliaries; decoder parameters stay shared."""

from ....models.ernie5_5_moe.config import ModelConfig as TextConfig
from ....models.ernie5_5_moe.echo import ECHO_WEIGHT_NAMES, extract_echo_weights
from ..mtp_split import MTPSplitter


class Ernie55MTPSplitter(MTPSplitter):
    output_model_type = "ernie5_5_mtp"
    block_size_extra = 2

    def select_keys(self, key, text_config):
        key = key.removeprefix("language_model.")
        prefix = f"model.layers.{text_config.get('num_hidden_layers', 43)}."
        name = key.removeprefix(prefix)
        return key.startswith("echo.") or (
            key.startswith(prefix)
            and (
                name.removesuffix("_scale_inv") in ECHO_WEIGHT_NAMES
                or name
                in ("e_proj.scales", "e_proj.biases", "h_proj.scales", "h_proj.biases")
            )
        )

    def rename(self, tensors, text_config):
        return extract_echo_weights(tensors, TextConfig.from_dict(text_config))

    def quantization_from_source(self, tensors, source_config):
        if any(key.endswith(".scales") for key in tensors):
            return source_config.get("quantization")
        return None

    def should_quantize_key(self, key):
        # Preserve the source precision of the auxiliary projections.
        return False
