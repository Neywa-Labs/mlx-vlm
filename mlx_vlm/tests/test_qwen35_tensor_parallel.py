# SPDX-License-Identifier: Apache-2.0
"""Dense Qwen TP math on CPU; rendezvous only is simulated."""

import copy
from concurrent.futures import ThreadPoolExecutor
from threading import Condition

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx.utils import tree_flatten

from mlx_vlm.models.qwen3_5.config import ModelConfig, TextConfig, VisionConfig
from mlx_vlm.models.qwen3_5.language import LanguageModel


class _Group:
    def __init__(self, rank: int, size: int):
        self._rank, self._size = rank, size
        self.step = 0

    def rank(self):
        return self._rank

    def size(self):
        return self._size


class _Collectives:
    """Bounded rendezvous of real lazy arrays; no sockets or native groups."""

    def __init__(self, monkeypatch, size):
        self.groups = [_Group(rank, size) for rank in range(size)]
        self.condition = Condition()
        self.rounds = {}
        # Build graphs in workers, evaluate them sequentially on the caller.
        # MLX's ordinary streams cannot cross the creating thread.
        self.stream = mx.new_thread_unsafe_stream(mx.cpu)
        # Prefill may request eager scheduling on its creating thread.
        # This CPU harness evaluates the complete rank graphs on the caller.
        monkeypatch.setattr(mx, "async_eval", lambda *arrays: None)
        monkeypatch.setattr(mx.distributed, "all_sum", self.all_sum)
        monkeypatch.setattr(mx.distributed, "all_gather", self.all_gather)

    def exchange(self, x, group, kind):
        step = group.step
        group.step += 1
        with self.condition:
            round_ = self.rounds.setdefault(step, {"kind": kind, "inputs": {}})
            assert round_["kind"] == kind, "Ranks disagree on collective order"
            round_["inputs"][group.rank()] = x
            if len(round_["inputs"]) == group.size():
                parts = [round_["inputs"][rank] for rank in range(group.size())]
                assert all(part.shape == x.shape for part in parts)
                round_["result"] = (
                    sum(parts[1:], parts[0])
                    if kind == "sum"
                    else mx.concatenate(parts, axis=0)
                )
                self.condition.notify_all()
            assert self.condition.wait_for(
                lambda: "result" in round_, timeout=10
            ), "Missing TP peer/collective"
            return round_["result"]

    def all_sum(self, x, *, group, **_):
        return self.exchange(x, group, "sum")

    def all_gather(self, x, *, group, **_):
        return self.exchange(x, group, "gather")

    def run(self, fn):
        def on_cpu(rank):
            previous = mx.default_device()
            mx.set_default_device(mx.cpu)
            try:
                with mx.stream(self.stream):
                    return fn(rank)
            finally:
                mx.set_default_device(previous)

        with ThreadPoolExecutor(max_workers=len(self.groups)) as executor:
            return list(executor.map(on_cpu, range(len(self.groups))))


@pytest.fixture(autouse=True)
def _cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _model():
    args = TextConfig(
        model_type="qwen3_5_text",
        hidden_size=64,
        intermediate_size=128,
        linear_num_value_heads=6,
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        rms_norm_eps=1e-6,
        vocab_size=65,
        max_position_embeddings=128,
        full_attention_interval=2,
        rope_parameters={
            "type": "default",
            "rope_theta": 10000,
            "partial_rotary_factor": 0.25,
            "mrope_section": [1, 1, 2],
        },
    )
    model = LanguageModel(
        args,
        ModelConfig(
            text_config=args, vision_config=VisionConfig(), model_type="qwen3_5"
        ),
    )
    # The provider enables fused Metal MRoPE based on hardware presence,
    # independently of default_device. Exercise its ordinary CPU math here.
    model.layers[1].self_attn.rotary_emb.fused_apply = False
    return model


def _close(x, y):
    np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("quantized", [False, True])
def test_dense_tp_prefill_decode_and_local_cache(monkeypatch, quantized):
    mx.random.seed(5)
    reference = _model()
    if quantized:
        nn.quantize(reference, group_size=32, bits=4)
    reference.eval()
    world = _Collectives(monkeypatch, 2)
    shards = [copy.deepcopy(reference) for _ in world.groups]
    for model, group in zip(shards, world.groups):
        model.shard(group)
        assert model.args.num_key_value_heads == 2  # global planning metadata
        assert model.lm_head.weight.shape == reference.lm_head.weight.shape
    caches = [model.make_cache() for model in shards]
    reference_cache = reference.make_cache()
    for tokens in (mx.array([[1, 5, 3]]), mx.array([[2]]), mx.array([[4]])):
        expected = reference(tokens, cache=reference_cache).logits
        outputs = world.run(
            lambda rank, tokens=tokens: shards[rank](tokens, cache=caches[rank]).logits
        )
        for rank, output in enumerate(outputs):
            _close(output, expected)
            _close(caches[rank][0][1], mx.split(reference_cache[0][1], 2, axis=1)[rank])
            _close(
                caches[rank][1].keys, mx.split(reference_cache[1].keys, 2, axis=1)[rank]
            )
            key_dim = reference.layers[0].linear_attn.key_dim
            parts = mx.split(reference_cache[0][0], [key_dim, 2 * key_dim], axis=-1)
            _close(
                caches[rank][0][0],
                mx.concatenate(
                    [mx.split(part, 2, axis=-1)[rank] for part in parts], axis=-1
                ),
            )
    assert [r["kind"] for r in world.rounds.values()] == ["sum"] * 12
    with pytest.raises(ValueError, match="already sharded"):
        shards[0].shard(world.groups[0])


def test_invalid_shard_does_not_partially_mutate():
    model = _model()
    nn.quantize(model, group_size=64, bits=4)
    before = dict(tree_flatten(model.parameters()))
    # GDN out_proj's 192 input channels split at 96, inside a group of 64.
    with pytest.raises(ValueError, match="quantization groups"):
        model.shard(_Group(0, 2))
    assert all(
        dict(tree_flatten(model.parameters()))[k] is v for k, v in before.items()
    )
    with pytest.raises(ValueError, match="Invalid"):
        model.shard(_Group(2, 2))
