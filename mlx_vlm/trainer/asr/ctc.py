"""CTC training for frame-synchronous (streaming-capable) ASR models.

Examples come from the model's ``prepare_example``: ``labels`` (transcript
token ids) and its ``forward_logits`` inputs. The blank is ``model.blank_id``.
"""

import mlx.core as mx

from mlx_vlm.trainer.asr.common import make_asr_task
from mlx_vlm.trainer.losses.ctc import ctc_loss
from mlx_vlm.trainer.vlm.cross_entropy.runtime import _logits_from_output


def asr_ctc_loss(model, example):
    """Score one example's frame logits against its transcript."""
    inputs = {key: value for key, value in example.items() if key != "labels"}
    logits = _logits_from_output(model.forward_logits(**inputs))
    return ctc_loss(
        logits.astype(mx.float32), example["labels"][None], blank_id=model.blank_id
    )


def make_task(model, args, processing_class=None, dataset_config=None):
    """Bind the model's training interface to the CTC objective."""
    return make_asr_task(
        model,
        args,
        dataset_config,
        asr_ctc_loss,
        length=lambda example: len(example["labels"]),
    )
