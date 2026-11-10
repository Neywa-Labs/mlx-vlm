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
        return key.startswith("echo.") or (
            key.startswith(prefix) and key.removeprefix(prefix) in ECHO_WEIGHT_NAMES
        )

    def rename(self, tensors, text_config):
        return extract_echo_weights(tensors, TextConfig.from_dict(text_config))

    def should_quantize_key(self, key):
        # Auxiliary projections stay dense, as in ordinary target conversion.
        return False
