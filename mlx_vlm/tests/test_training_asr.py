"""ASR next-token and CTC training over the shared engine with tiny models."""

import itertools
import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest

from mlx_vlm.train import parse_args
from mlx_vlm.trainer import SFTTrainConfig, Trainer
from mlx_vlm.trainer.datasets.loading import RawSplit
from mlx_vlm.trainer.losses.ctc import ctc_loss

VOCAB, MELS = 6, 4
SEEN = []  # (audio, language) pairs the model was asked to prepare


def features(path):
    seed = int(path.rsplit("/", 1)[-1].split(".")[0])
    return mx.array(np.random.default_rng(seed).normal(size=(1, 6 + seed, MELS)))


class TinyDecoder(nn.Module):
    """A model that defines the training interface itself (next-token)."""

    training_objective = "sft"

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, 8)
        self.audio = nn.Linear(MELS, 8)
        self.head = nn.Linear(8, VOCAB)

    def prepare_example(self, audio, text, language="en"):
        SEEN.append((audio, language))
        ids = [int(token) for token in text.split()]
        return {
            "input_ids": np.array([0, 5, *ids]),
            "completion_mask": np.array([0, 0] + [1] * len(ids)),
            "input_features": features(audio),
        }

    def forward_logits(self, input_ids, input_features):
        hidden = self.embedding(input_ids)
        return self.head(hidden + self.audio(input_features).mean(axis=1)[:, None])


class TinyEncoder(nn.Module):
    """A model that defines the training interface itself (CTC)."""

    training_objective = "ctc"
    blank_id = VOCAB - 1

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(MELS, VOCAB)

    def prepare_example(self, audio, text, language="en"):
        return {
            "input_features": features(audio),
            "labels": [int(token) for token in text.split()],
        }

    def forward_logits(self, input_features):
        return self.layer(input_features)


def rows(base_path=None):
    texts = ["1 2", "1 2 3", "3", "2 1"]
    return RawSplit(
        [{"audio": f"audio/{i}.wav", "text": text} for i, text in enumerate(texts)],
        base_path,
    )


def config(tmp_path, iters):
    return SFTTrainConfig(
        batch_size=2,
        iters=iters,
        steps_per_report=iters,
        steps_per_eval=iters,
        adapter_file=str(tmp_path / "adapters.safetensors"),
    )


def brute_force_ctc(log_probs, target, blank):
    """Sum every frame alignment that collapses to the target."""
    total = 0.0
    for path in itertools.product(range(log_probs.shape[1]), repeat=len(log_probs)):
        collapsed = [k for k, _ in itertools.groupby(path) if k != blank]
        if collapsed == target:
            total += math.exp(sum(log_probs[t, k] for t, k in enumerate(path)))
    return -math.log(total)


def test_ctc_loss_matches_alignment_enumeration_and_masks_padding():
    logits = mx.random.normal((2, 5, 3), key=mx.random.key(0))
    log_probs = np.array(nn.log_softmax(logits, axis=-1))
    expected = [
        brute_force_ctc(log_probs[0], [1, 1], blank=0),
        brute_force_ctc(log_probs[1, :3], [2], blank=0),
    ]
    loss, metrics = ctc_loss(
        logits, mx.array([[1, 1], [2, -100]]), mx.array([5, 3]), blank_id=0
    )
    assert loss.item() == pytest.approx(sum(expected) / 3, rel=1e-5)
    assert metrics["weight"].item() == 3 and metrics["num_frames"].item() == 8
    gradient = mx.grad(lambda x: ctc_loss(x, mx.array([[1, 1], [2, -100]]))[0])(logits)
    assert mx.isfinite(gradient).all().item()


def test_ctc_loss_skips_targets_longer_than_their_frames():
    logits = mx.zeros((2, 2, 3))
    loss, metrics = ctc_loss(logits, mx.array([[1, 1, 2], [1, -100, -100]]))
    assert metrics["weight"].item() == 1
    assert loss.item() == pytest.approx(
        brute_force_ctc(np.full((2, 3), -math.log(3)), [1], blank=0), rel=1e-5
    )


def test_asr_sft_trains_transcript_tokens_from_speech_rows(tmp_path):
    SEEN.clear()
    trainer = Trainer(
        TinyDecoder(),
        config(tmp_path, 30),
        task="asr",
        train_dataset=rows(tmp_path),
        eval_dataset=rows(tmp_path),
        dataset_config={"language": "de"},
        optimizer=optim.Adam(learning_rate=0.05),
    )
    before = trainer.evaluate()
    metrics = trainer.train()
    assert metrics["val_loss"] < before["val_loss"]
    # Only transcript tokens are supervised; audio paths resolve to the dataset root.
    assert before["val_num_tokens"] == 8
    assert SEEN[0] == (str(tmp_path / "audio/0.wav"), "de")


def test_asr_ctc_is_selected_by_the_model(tmp_path):
    trainer = Trainer(
        TinyEncoder(),
        config(tmp_path, 40),
        task="asr",
        train_dataset=rows(),
        eval_dataset=rows(),
        optimizer=optim.Adam(learning_rate=0.05),
    )
    before = trainer.evaluate()
    metrics = trainer.train()
    assert metrics["val_loss"] < before["val_loss"]
    assert before["val_num_frames"] == 6 + 7 + 8 + 9
    assert before["val_num_tokens"] == 8


def test_requested_algorithm_must_match_the_model():
    with pytest.raises(ValueError, match="trains with 'ctc'"):
        Trainer(TinyEncoder(), task="asr", algorithm="sft", train_dataset=rows())


def test_cli_leaves_the_asr_objective_to_the_model():
    args, settings = parse_args(["--model", "unused", "--task", "asr"])
    assert args.algorithm is None and isinstance(settings, SFTTrainConfig)
