"""Next-token cross-entropy for autoregressive ASR models.

Examples come from the model's ``prepare_example``: unshifted ``input_ids``, a
``completion_mask`` over transcript tokens, and its ``forward_logits`` inputs.
The objective is the shared VLM one; only the model call differs.
"""

import mlx.core as mx

from mlx_vlm.trainer.asr.common import make_asr_task
from mlx_vlm.trainer.vlm.cross_entropy.runtime import _TEXT_FIELDS, vlm_sft_loss


def _forward(model, input_ids, attention_mask, batch, **kwargs):
    inputs = {key: value for key, value in batch.items() if key not in _TEXT_FIELDS}
    return model.forward_logits(input_ids=input_ids, **inputs, **kwargs)


def asr_sft_loss(model, example):
    """Score one example; prompts and audio placeholders are never targets."""
    input_ids = example["input_ids"][None]
    batch = {
        **example,
        "input_ids": input_ids,
        "attention_mask": mx.ones_like(input_ids),
        "completion_mask": example["completion_mask"][None],
    }
    return vlm_sft_loss(model, batch, train_on_completions=True, forward=_forward)


def make_task(model, args, processing_class=None, dataset_config=None):
    """Bind the model's training interface to supervised cross-entropy."""
    return make_asr_task(
        model,
        args,
        dataset_config,
        asr_sft_loss,
        length=lambda example: len(example["input_ids"]),
    )
