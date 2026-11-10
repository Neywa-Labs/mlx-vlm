"""Model-agnostic ASR plumbing over the standard STT training interface.

Models own everything model-specific: ``prepare_example(audio, text,
language)`` builds one example, ``forward_logits(**inputs)`` scores it, and
``training_objective`` names the loss. Nothing here knows an architecture.
"""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_vlm.trainer.common.utils import iterate_batch_indices


class ASRDataset:
    """Speech rows prepared lazily by the model, one example per row.

    Rows follow the shared speech format ``{"audio": path, "text": str}`` with
    an optional ``language``. Relative audio paths resolve against the folder
    the split was loaded from.
    """

    def __init__(self, rows: Sequence[Mapping], model, language: str = "en"):
        self.rows, self.model, self.language = rows, model, language
        self.base_path = getattr(rows, "base_path", None)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        audio = row["audio"]
        if self.base_path is not None and isinstance(audio, str):
            audio = str(Path(self.base_path) / audio)
        example = self.model.prepare_example(
            audio, row["text"], row.get("language") or self.language
        )
        # Cache framework-neutral values; half precision has no NumPy dtype.
        return {
            key: (
                np.asarray(value.astype(mx.float32))
                if isinstance(value, mx.array) and value.dtype == mx.bfloat16
                else np.asarray(value)
            )
            for key, value in example.items()
        }

    def __len__(self) -> int:
        return len(self.rows)


def iterate_examples(dataset, batch_size, *, length: Callable, train=False, seed=None):
    """Yield length-grouped lists of single examples, sharded across ranks."""
    world = mx.distributed.init()
    for indices in iterate_batch_indices(
        len(dataset),
        batch_size,
        length_key=lambda index: length(dataset[index]),
        train=train,
        rank=world.rank(),
        world_size=world.size(),
        seed=seed,
    ):
        yield [
            {key: mx.array(value) for key, value in dataset[index].items()}
            for index in indices
        ]


def per_example(loss: Callable) -> Callable:
    """Average a single-example loss over a batch without padding any audio.

    Each example runs through the model alone, so padding never changes an
    encoder's audio representation. Counts add; other metrics are weighted.
    """

    def batched(model, batch: Sequence[Any]):
        total, weight, metrics = 0.0, 0.0, {}
        for example in batch:
            value, values = loss(model, example)
            share = values["weight"].astype(mx.float32)
            total, weight = total + value * share, weight + share
            for key, item in values.items():
                scale = 1 if key == "weight" or key.startswith("num_") else share
                metrics[key] = metrics.get(key, 0) + item * scale
        for key in metrics:
            if key != "weight" and not key.startswith("num_"):
                metrics[key] = metrics[key] / mx.maximum(weight, 1)
        return total / mx.maximum(weight, 1), metrics

    return batched


def make_asr_task(model, args, dataset_config, loss: Callable, length: Callable):
    """Build a task whose data and forward both come from the model."""
    from functools import partial

    from mlx_vlm.trainer.common.task import TrainingTask
    from mlx_vlm.trainer.common.utils import (
        enable_model_gradient_checkpointing,
        save_trainable_weights,
    )
    from mlx_vlm.trainer.datasets.loading import CacheDataset

    language = getattr(dataset_config, "language", None) or "en"
    return TrainingTask(
        loss=per_example(loss),
        batches=partial(iterate_examples, batch_size=args.batch_size, length=length),
        prepare_dataset=lambda rows: CacheDataset(
            ASRDataset(rows, model, language), max_size=args.cache_size, process=False
        ),
        checkpoint=enable_model_gradient_checkpointing,
        save_weights=save_trainable_weights,
        unit="tokens",
    )
