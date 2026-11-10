"""Token cross-entropy shared by language, vision-language, and speech models.

Callers own model forwarding and causal shifting: logits [B, T, V] must
already align with targets [B, T], e.g. input_ids[:, 1:] for next-token loss.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def make_loss_mask(targets: mx.array, lengths: mx.array) -> mx.array:
    """Mask inclusive [start, end] positions in the unshifted token sequence.

    Targets are shifted by one, so their first column has position 1. Each
    row in lengths has shape [2]. This matches the ASR transcript boundaries.
    """
    if targets.ndim != 2:
        raise ValueError("targets must have shape [batch, time].")
    if lengths.shape != (targets.shape[0], 2):
        raise ValueError("lengths must have shape [batch, 2].")
    steps = mx.arange(1, targets.shape[1] + 1)[None, :]
    return mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:2])


def token_cross_entropy(
    logits: mx.array,
    targets: mx.array,
    *,
    ignore_index: int | None = -100,
) -> mx.array:
    """Return unreduced token cross-entropy, zeroing ignored labels.

    Preference objectives need sequence log-probabilities, while SFT needs a
    mean over supervised tokens. Keeping the unreduced calculation here makes
    both paths share identical target validation and CE semantics.
    """
    if targets.ndim != 2 or logits.ndim != 3:
        raise ValueError(
            "Expected logits [batch, time, vocab] and targets [batch, time]."
        )
    if logits.shape[:-1] != targets.shape:
        raise ValueError("Logits and targets must have matching batch/time dimensions.")
    mask = mx.ones(targets.shape, dtype=mx.bool_)
    if ignore_index is not None:
        mask = targets != ignore_index
    safe_targets = mx.where(mask, targets, 0)
    token_loss = nn.losses.cross_entropy(logits, safe_targets, reduction="none")
    return mx.where(mask, token_loss, 0).astype(mx.float32)


def cross_entropy(
    logits: mx.array,
    targets: mx.array,
    mask: mx.array | None = None,
    *,
    ignore_index: int | None = -100,
    vision_mask: mx.array | None = None,
) -> tuple[mx.array, dict[str, mx.array]]:
    """Return mean supervised-token loss and scalar auxiliary metrics.

    An optional boolean mask selects target positions. Ignored labels are
    excluded even when selected by the mask; masked or ignored labels are
    replaced before cross-entropy so sentinels are never used as class IDs.
    ``weight`` and ``num_tokens`` count supervised targets; ``token_accuracy``
    is masked next-token accuracy and ``nll`` enables report-level perplexity.
    An optional ``vision_mask`` identifies valid image/video positions in the
    model's forward inputs, aligned with logits [B, T]. Its count is reported
    independently of the supervision mask and does not change the objective.
    With no supervised tokens, returns zero loss and zero count. All selected
    labels must be valid vocabulary indices. No shifting happens here.
    """
    if targets.ndim != 2 or logits.ndim != 3:
        raise ValueError(
            "Expected logits [batch, time, vocab] and targets [batch, time]."
        )
    if logits.shape[:-1] != targets.shape:
        raise ValueError("Logits and targets must have matching batch/time dimensions.")
    if vision_mask is not None and vision_mask.shape != targets.shape:
        raise ValueError("vision_mask must align with logits batch/time dimensions.")
    if mask is None:
        mask = mx.ones(targets.shape, dtype=mx.bool_)
    elif mask.shape != targets.shape:
        raise ValueError("mask must have the same shape as targets.")
    else:
        mask = mask.astype(mx.bool_)
    if ignore_index is not None:
        mask = mx.logical_and(mask, targets != ignore_index)

    safe_targets = mx.where(mask, targets, 0)
    token_loss = token_cross_entropy(logits, safe_targets, ignore_index=None)
    ntokens = mask.sum()
    denominator = mx.maximum(ntokens.astype(mx.float32), 1.0)
    loss = mx.where(mask, token_loss, 0).sum() / denominator
    correct = mx.logical_and(mx.argmax(logits, axis=-1) == safe_targets, mask).sum()
    return loss, {
        "weight": ntokens,
        "num_tokens": ntokens,
        "num_vision_tokens": (
            mx.array(0, dtype=mx.int32)
            if vision_mask is None
            else vision_mask.astype(mx.bool_).sum()
        ),
        "token_accuracy": correct.astype(mx.float32) / denominator,
        "nll": loss,
    }
