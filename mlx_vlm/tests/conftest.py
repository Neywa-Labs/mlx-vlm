import os

import pytest

# float32 matmul runs at TF32 precision on hardware with matrix units, which is
# looser than the float32 references these tests compare against. Set rather than
# setdefault, so an inherited MLX_ENABLE_TF32=1 doesn't leak into the test run.
os.environ["MLX_ENABLE_TF32"] = "0"


def pytest_sessionfinish(session, exitstatus):
    """Release thread-local MLX resources before Python finalization."""
    del session, exitstatus

    import mlx.core as mx

    clear_streams = getattr(mx, "clear_streams", None)
    if clear_streams is not None:
        clear_streams()


@pytest.fixture
def ernie55_model():
    import mlx.core as mx

    from mlx_vlm.models.ernie5_5_moe import Model, ModelConfig
    from mlx_vlm.tests.test_models import model_case

    mx.random.seed(55)
    model = Model(ModelConfig.from_dict(model_case("ernie5_5_moe")["config"]))
    model.eval()
    return model


@pytest.fixture
def ernie55_learned_model(ernie55_model):
    import mlx.core as mx

    # Exercise learned mixers/routers instead of their zero initialization.
    for layer in ernie55_model.layers:
        for hc in (layer.attn_hc, layer.ffn_hc):
            hc.fn = mx.random.normal(hc.fn.shape) * 0.02
            hc.base = mx.random.normal(hc.base.shape) * 0.1
        layer.attn.vha.U = mx.random.normal(layer.attn.vha.U.shape) * 0.1
        layer.attn.vha.V = mx.random.normal(layer.attn.vha.V.shape) * 0.1
        if hasattr(layer.ffn, "gate"):
            for name in ("weight", "weight_1"):
                layer.ffn.gate[name] = (
                    mx.random.normal(layer.ffn.gate[name].shape) * 0.1
                )
    return ernie55_model


@pytest.fixture
def ernie55_echo(ernie55_model):
    import mlx.core as mx

    from mlx_vlm.speculative.drafters.ernie5_5_mtp import Model, ModelConfig

    draft = Model(ModelConfig(text_config=ernie55_model.config))
    draft.eval()
    mx.eval(ernie55_model.parameters(), draft.parameters())
    return ernie55_model, draft
