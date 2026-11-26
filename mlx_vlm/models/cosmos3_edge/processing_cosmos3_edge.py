"""Processor for Cosmos3-Edge in mlx-vlm.

transformers' ``Cosmos3EdgeProcessor`` does the real work (image resizing, frame timestamps, prompt expansion). Two things
are adapted so mlx-vlm's video path gives the same inputs as the reference processor:

* Frame choice: mlx-vlm decodes and samples the clip itself. Cosmos3-Edge picks frames with its own rule
  (``num_frames = int(total / video_fps * fps)``, clamped to [min_frames, max_frames], no rounding to an even count), so
  the video processor is marked ``sample_frames_in_loader`` and mlx-vlm calls its ``sample_frames`` while decoding.
* Metadata: ``prepare_inputs`` only forwards keywords the processor declares, so ``video_metadata`` is a named
  parameter here. It is converted to transformers' ``VideoMetadata`` and the frames are passed with
  ``do_sample_frames=False``, so the timestamps in the prompt come from the frames actually used. Without this the
  reference processor saw already-sampled frames with no fps, assumed 24 fps and resampled them down to 4 frames.
"""

from ..base import install_auto_processor_patch

# Older transformers releases have no cosmos3_edge processor; the model itself still loads without it.
try:
    from transformers.models.cosmos3_edge.processing_cosmos3_edge import (
        Cosmos3EdgeProcessor as _HFCosmos3EdgeProcessor,
    )
except ImportError:
    _HFCosmos3EdgeProcessor = None

_METADATA_FIELDS = (
    "total_num_frames",
    "fps",
    "frames_indices",
    "width",
    "height",
    "duration",
)


def _to_hf_metadata(metadata):
    from transformers.video_utils import VideoMetadata as _HFVideoMetadata

    if metadata is None or isinstance(metadata, _HFVideoMetadata):
        return metadata
    if isinstance(metadata, dict):
        values = {k: metadata.get(k) for k in _METADATA_FIELDS}
    else:
        values = {k: getattr(metadata, k, None) for k in _METADATA_FIELDS}
    if values["frames_indices"] is not None:
        values["frames_indices"] = [int(i) for i in values["frames_indices"]]
    return _HFVideoMetadata(**{k: v for k, v in values.items() if v is not None})


__all__ = []

if _HFCosmos3EdgeProcessor is not None:

    class Cosmos3EdgeProcessor(_HFCosmos3EdgeProcessor):
        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
            processor = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
            if getattr(processor, "video_processor", None) is not None:
                processor.video_processor.sample_frames_in_loader = True
            return processor

        def __call__(
            self,
            images=None,
            text=None,
            videos=None,
            fps=None,
            video_metadata=None,
            **kwargs,
        ):
            # ``fps`` is accepted and dropped: mlx-vlm has already sampled the frames at that rate, and the timestamps
            # come from ``video_metadata``.
            if videos is not None and video_metadata is not None:
                kwargs["video_metadata"] = [_to_hf_metadata(m) for m in video_metadata]
                kwargs["do_sample_frames"] = False
            if videos is not None:
                # The reference processor reads the metadata back to build the "<t seconds>" markers, which needs
                # return_metadata (apply_chat_template sets it the same way). It is not a model input, so drop it after.
                kwargs["return_metadata"] = True
            out = super().__call__(images=images, text=text, videos=videos, **kwargs)
            out.pop("video_metadata", None)
            return out

    __all__.append("Cosmos3EdgeProcessor")
    install_auto_processor_patch("cosmos3_edge", Cosmos3EdgeProcessor)
