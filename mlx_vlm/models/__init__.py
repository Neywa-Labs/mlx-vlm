import importlib

# Native formats without an architecture identifier opt in here. Recognition
# and conversion stay in each model's config module.
_NATIVE_CONFIG_ADAPTERS = ("mistral_large3", "mistral_large4")


def normalize_config(config: dict) -> dict:
    """Resolve recognized native schemas without changing declared architectures."""
    if any(
        config.get(key)
        for key in ("model_type", "speculators_model_type", "model_file")
    ):
        return config
    for name in _NATIVE_CONFIG_ADAPTERS:
        adapter = importlib.import_module(f"{__name__}.{name}.config")
        if adapter.matches_native_config(config):
            return adapter.config_from_params(config)
    return config
