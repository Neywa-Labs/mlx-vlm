"""Reusable Connectionist Temporal Classification loss helpers."""

import mlx.core as mx
import mlx.nn as nn

IGNORE_INDEX = -100
_NEG = -1e30  # Finite stand-in for log(0): keeps logsumexp gradients defined.


def ctc_loss(
    logits: mx.array,
    targets: mx.array,
    input_lengths: mx.array | None = None,
    *,
    blank_id: int = 0,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[mx.array, dict[str, mx.array]]:
    """Return mean per-target-token CTC loss and scalar metrics.

    ``logits`` [B, T, V] are frame scores; ``targets`` [B, S] are right-padded
    with ``ignore_index``; ``input_lengths`` [B] counts valid frames (all T
    when omitted). Examples whose targets cannot be aligned within their
    frames contribute neither loss nor weight.
    """
    if logits.ndim != 3 or targets.ndim != 2 or logits.shape[0] != targets.shape[0]:
        raise ValueError("Expected logits [batch, time, vocab] and targets [batch, S].")
    batch, time, _ = logits.shape
    valid = targets != ignore_index
    target_lengths = valid.sum(axis=1)

    # Interleave blanks: state 2i is a blank, state 2i + 1 is target i.
    states = 2 * targets.shape[1] + 1
    extended = mx.full((batch, states), blank_id, dtype=mx.int32)
    extended[:, 1::2] = mx.where(valid, targets, blank_id).astype(mx.int32)
    emissions = mx.take_along_axis(
        nn.log_softmax(logits.astype(mx.float32), axis=-1),
        mx.broadcast_to(extended[:, None, :], (batch, time, states)),
        axis=2,
    )
    # A state may also be entered from two back when that skips only a blank
    # between different labels.
    skip = mx.concatenate(
        [
            mx.zeros((batch, 2), dtype=mx.bool_),
            mx.logical_and(
                extended[:, 2:] != blank_id, extended[:, 2:] != extended[:, :-2]
            ),
        ],
        axis=1,
    )
    floor = mx.full((batch, 2), _NEG)
    alpha = mx.where(mx.arange(states)[None, :] < 2, emissions[:, 0], _NEG)
    for step in range(1, time):
        shifted = mx.concatenate([floor, alpha], axis=1)
        updated = (
            mx.logsumexp(
                mx.stack(
                    [
                        alpha,
                        shifted[:, 1:-1],
                        mx.where(skip, shifted[:, :-2], _NEG),
                    ]
                ),
                axis=0,
            )
            + emissions[:, step]
        )
        alpha = (
            updated
            if input_lengths is None
            else mx.where((step < input_lengths)[:, None], updated, alpha)
        )

    # Paths end on the last target or the blank after it.
    end = (2 * target_lengths)[:, None]
    last_blank = mx.take_along_axis(alpha, end, axis=1)
    last_label = mx.where(
        end > 0, mx.take_along_axis(alpha, mx.maximum(end - 1, 0), axis=1), _NEG
    )
    nll = -mx.logsumexp(mx.concatenate([last_blank, last_label], axis=1), axis=1)

    feasible = nll < -_NEG / 2
    num_tokens = mx.where(feasible, target_lengths, 0).sum()
    loss = mx.where(feasible, nll, 0).sum() / mx.maximum(num_tokens, 1)
    frames = (
        mx.array(batch * time)
        if input_lengths is None
        else mx.minimum(input_lengths, time).sum()
    )
    return loss, {
        "weight": num_tokens,
        "num_tokens": num_tokens,
        "num_frames": frames.astype(mx.int32),
        "num_sequences": mx.array(batch, dtype=mx.int32),
    }
