"""NumPy preprocessing for WeMM-Video, including aligned video soundtracks.

Timestamp placement follows Tencent/WeMM-Video's Apache-2.0 wemm_media.py.
"""

import copy
import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
from transformers import WhisperFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature

from ..base import to_mlx
from ..qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor, _load_qwen_vl_json
from .config import AudioConfig

PLACEHOLDERS = {
    "image": "<|vision_start|><|image_pad|><|vision_end|>",
    "video": "<|vision_start|><|video_pad|><|vision_end|>",
    "audio": "<|audio_start|><|audio_pad|><|audio_end|>",
}


def audio_timestamp_groups(count, window, duration, interval=1.0):
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("timestamp_interval_sec must be finite and positive.")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Audio duration must be finite and positive.")
    duration = min(duration, count * window)
    effective = max(1, min(count, math.ceil(duration / window - 1e-9)))
    groups = max(1, math.ceil(duration / interval - 1e-9))
    base, remainder = divmod(effective, groups)
    sizes = [base] * groups
    for i in range(1, remainder + 1):
        sizes[min(groups - 1, math.floor(i * groups / remainder) - 1)] += 1
    sizes[-1] += count - effective
    return sizes, [i * interval for i in range(groups)]


def audio_token_ids(tokenizer, count, window, duration, interval=1.0):
    sizes, timestamps = audio_timestamp_groups(count, window, duration, interval)
    ids = [tokenizer.convert_tokens_to_ids("<|audio_start|>")]
    for size, timestamp in zip(sizes, timestamps):
        ids.extend(
            tokenizer.encode(f"<{timestamp:.1f} seconds>", add_special_tokens=False)
        )
        ids.extend([tokenizer.convert_tokens_to_ids("<|audio_pad|>")] * size)
    return ids + [tokenizer.convert_tokens_to_ids("<|audio_end|>")]


def video_token_ids(
    tokenizer,
    grid,
    metadata,
    temporal_patch_size,
    merge_size,
    audio_count=0,
    audio_duration=None,
    window=0.1,
):
    time, height, width = map(int, grid)
    indices = list(metadata["frames_indices"])
    indices += indices[-1:] * (-len(indices) % temporal_patch_size)
    if len(indices) != time * temporal_patch_size:
        raise ValueError("Video frame metadata does not match the patch grid.")
    groups = [0] * time
    if audio_count:
        rate = len(metadata["frames_indices"]) / audio_duration
        boundaries = np.floor(
            np.arange(1, time, dtype=np.float32)
            * np.float32(temporal_patch_size / rate / window)
        ).astype(int)
        bounds = [0, *np.clip(boundaries, 0, audio_count).tolist(), audio_count]
        groups = [right - left for left, right in zip(bounds, bounds[1:])]
    start, end, video, audio = [
        tokenizer.convert_tokens_to_ids(token)
        for token in (
            "<|vision_start|>",
            "<|vision_end|>",
            "<|video_pad|>",
            "<|audio_pad|>",
        )
    ]
    ids = []
    for index in range(time):
        left, right = (
            indices[index * temporal_patch_size],
            indices[(index + 1) * temporal_patch_size - 1],
        )
        seconds = (
            (left + right) / (2 * metadata["fps"])
            if audio_count
            else (left / metadata["fps"] + right / metadata["fps"]) / 2
        )
        ids.extend(
            tokenizer.encode(f"<{seconds:.1f} seconds>", add_special_tokens=False)
        )
        ids.extend(
            [start]
            + [video] * (height * width // merge_size**2)
            + [audio] * groups[index]
            + [end]
        )
    return ids


class WeMMVideoProcessor(Qwen3VLProcessor):
    supports_multiple_audio = True
    # Soundtrack extraction needs the original path, before the shared loader
    # replaces it with sampled frames.
    handles_video_loading = True

    @classmethod
    def get_attributes(cls):
        return ["image_processor", "tokenizer", "video_processor"]

    def __init__(self, *args, audio_config, wemm_defaults=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.audio_config = AudioConfig.from_dict(audio_config)
        self.feature_extractor = WhisperFeatureExtractor(
            feature_size=self.audio_config.num_mel_bins,
            sampling_rate=16000,
            hop_length=160,
            chunk_length=30,
        )
        self.wemm_defaults = {
            "fps": 2.0,
            "min_frames": 4,
            "max_frames": 128,
            "total_pixels": 25165824,
            "min_pixels": 4096,
            "timestamp_interval_sec": 1.0,
            **(wemm_defaults or {}),
        }

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        if not Path(path).is_dir():
            from huggingface_hub import snapshot_download

            path = snapshot_download(
                str(path),
                allow_patterns=["*.json", "*.jinja"],
                **{
                    key: kwargs[key]
                    for key in (
                        "revision",
                        "cache_dir",
                        "token",
                        "local_files_only",
                        "force_download",
                    )
                    if key in kwargs
                },
            )
        # The MoE checkpoint advertises a custom config, but its tokenizer is
        # native. Avoid AutoTokenizer prompting to execute that config's code.
        kwargs.setdefault("trust_remote_code", False)
        native = Qwen3VLProcessor.from_pretrained(path, **kwargs)
        config = _load_qwen_vl_json(path, "config.json")
        proc_config = _load_qwen_vl_json(path, "processor_config.json") or {}
        return cls(
            image_processor=native.image_processor,
            tokenizer=native.tokenizer,
            video_processor=native.video_processor,
            chat_template=native.chat_template,
            audio_config=config["audio_config"],
            wemm_defaults=proc_config.get("wemm_defaults"),
        )

    def save_pretrained(self, save_directory, **kwargs):
        destination = Path(save_directory)
        destination.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(destination, **kwargs)
        self.image_processor.save_pretrained(destination)
        self.video_processor.save_pretrained(destination)
        (destination / "chat_template.jinja").write_text(self.chat_template)
        (destination / "processor_config.json").write_text(
            json.dumps(
                {
                    "processor_class": "WeMMVideoProcessor",
                    "wemm_defaults": self.wemm_defaults,
                    "audio_config": asdict(self.audio_config),
                },
                indent=2,
            )
            + "\n"
        )

    def apply_chat_template(
        self,
        conversation,
        *,
        tokenize=False,
        return_dict=False,
        return_tensors="mlx",
        enable_thinking=False,
        **kwargs,
    ):
        if tokenize:
            inputs = self(
                messages=conversation,
                return_tensors=return_tensors,
                enable_thinking=enable_thinking,
                **kwargs,
            )
            return inputs if return_dict else inputs["input_ids"]
        normalized = []
        for message in conversation:
            row = dict(message)
            if isinstance(row["content"], list):
                parts = []
                for item in row["content"]:
                    kind = item.get("type")
                    if kind == "text":
                        parts.append(item["text"])
                    elif kind in (*PLACEHOLDERS, "video_audio"):
                        parts.append(
                            PLACEHOLDERS["video" if kind == "video_audio" else kind]
                        )
                    else:
                        raise ValueError(f"Unsupported WeMM content type: {kind}")
                row["content"] = "".join(parts)
            normalized.append(row)
        return self.tokenizer.apply_chat_template(
            normalized, tokenize=False, enable_thinking=enable_thinking, **kwargs
        )

    @staticmethod
    def _items(value):
        return [] if value is None else value if isinstance(value, list) else [value]

    def _audio_features(self, source, *, pcm16=True):
        from ...utils import load_audio

        samples = np.asarray(load_audio(source, sr=16000), dtype=np.float32)
        if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all():
            raise ValueError("Audio must contain finite, nonempty mono samples.")
        duration = len(samples) / 16000
        if pcm16:
            # libsndfile's PCM16 encoding floors and saturates the waveform.
            samples = (
                np.clip(np.floor(samples * 32768), -32768, 32767).astype(np.float32)
                / 32768
            )
        samples = np.pad(samples, (0, max(0, 16000 - len(samples))))
        features = [
            self.feature_extractor(
                samples[i : i + 480000], sampling_rate=16000, return_tensors="np"
            )["input_features"]
            for i in range(0, len(samples), 480000)
        ]
        return np.concatenate(features), duration

    @staticmethod
    def _soundtrack(source):
        if not isinstance(source, (str, Path)):
            raise ValueError(
                "Video soundtracks need a video path/URL or an explicit audio waveform."
            )
        result = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-i",
                str(source),
                "-map",
                "0:a:0",
                "-f",
                "f32le",
                "-ac",
                "1",
                "-ar",
                "16000",
                "pipe:1",
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise ValueError(
                f"Cannot decode video audio: {result.stderr.decode(errors='replace').strip()}"
            )
        return np.frombuffer(result.stdout, dtype="<f4").copy()

    def _video(self, source, options, metadata=None):
        from ...utils import VideoSampling, load_video

        processor = copy.copy(self.video_processor)
        if isinstance(source, (str, Path)):

            def sample_frames(info, **_):
                total = info.total_num_frames
                desired = options.get("nframes") or total / info.fps * options["fps"]
                lower = math.ceil(options["min_frames"] / 2) * 2
                upper = min(options["max_frames"], total) // 2 * 2
                count = int(min(max(desired, lower), upper, total)) // 2 * 2
                return np.linspace(0, total - 1, count, dtype=int)

            frames, info = load_video(
                str(source),
                VideoSampling(
                    fps=options["fps"],
                    min_frames=options["min_frames"],
                    max_frames=options["max_frames"],
                ),
                frame_sampler=sample_frames,
            )
            metadata = asdict(info)
        else:
            frames = processor._prepare_video(source)
            metadata = metadata or dict(
                frames_indices=list(range(len(frames))), fps=options["fps"]
            )
            if not isinstance(metadata, dict):
                metadata = asdict(metadata)
        count = len(frames)
        maximum = int(
            max(
                min(768 * 32**2, options["total_pixels"] / count * 2),
                int(options["min_pixels"] * 1.05),
            )
        )
        if options.get("max_pixels") is not None:
            maximum = min(maximum, options["max_pixels"])
        processor.min_pixels = count * options["min_pixels"]
        processor.max_pixels = count * maximum
        return processor(videos=[frames], video_metadata=[metadata]), metadata

    def __call__(
        self,
        text=None,
        images=None,
        audio=None,
        videos=None,
        *,
        question=None,
        messages=None,
        video_audio=None,
        use_audio_in_video=False,
        video_metadata=None,
        fps=None,
        nframes=None,
        min_frames=None,
        max_frames=None,
        total_pixels=None,
        min_pixels=None,
        max_pixels=None,
        timestamp_interval_sec=None,
        return_tensors="mlx",
        padding=True,
        add_special_tokens=False,
        add_generation_prompt=True,
        enable_thinking=False,
        **kwargs,
    ):
        if question is not None and text is not None:
            raise ValueError("Use either text (a rendered prompt) or question.")
        if messages is not None:
            if any(
                value is not None
                for value in (text, question, images, audio, videos, video_audio)
            ):
                raise ValueError(
                    "Put media and text inside messages, or pass them separately."
                )
            images, audio, videos = [], [], []
            for message in messages:
                if isinstance(message["content"], list):
                    for item in message["content"]:
                        kind = item.get("type")
                        if kind == "image":
                            images.append(item.get("image", item.get("path")))
                        elif kind == "audio":
                            audio.append(item.get("audio", item.get("path")))
                        elif kind in ("video", "video_audio"):
                            videos.append(
                                dict(
                                    item,
                                    use_audio_in_video=(
                                        kind == "video_audio"
                                        or item.get(
                                            "use_audio_in_video", use_audio_in_video
                                        )
                                    ),
                                )
                            )
            text = self.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
            )
        images, audio, videos = map(self._items, (images, audio, videos))
        videos += [
            dict(video=v, use_audio_in_video=True) for v in self._items(video_audio)
        ]
        if video_metadata is not None and len(video_metadata) != len(videos):
            raise ValueError("Expected one video_metadata entry per video.")
        if question is not None:
            content = [
                {"type": kind}
                for kind, values in (
                    ("image", images),
                    ("video", videos),
                    ("audio", audio),
                )
                for _ in values
            ]
            content.append({"type": "text", "text": question})
            text = self.apply_chat_template(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": content},
                ],
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
            )
        if isinstance(text, list) and len(text) == 1:
            text = text[0]
        if not isinstance(text, str):
            raise ValueError("WeMM-Video expects one conversation at a time.")
        options = dict(self.wemm_defaults)
        overrides = dict(
            fps=fps,
            nframes=nframes,
            min_frames=min_frames,
            max_frames=max_frames,
            total_pixels=total_pixels,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        options.update({k: v for k, v in overrides.items() if v is not None})
        interval = (
            timestamp_interval_sec
            if timestamp_interval_sec is not None
            else options["timestamp_interval_sec"]
        )
        blocks = {kind: [] for kind in PLACEHOLDERS}
        pixel_values, image_grids, video_values, video_grids = [], [], [], []
        for source in images:
            from ...utils import process_image

            image = process_image(source, None, self.image_processor)
            result = self.image_processor(
                [image],
                **{
                    k: v
                    for k, v in {
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                    }.items()
                    if v is not None
                },
            )
            grid = result["image_grid_thw"][0]
            count = int(np.prod(grid)) // self.image_processor.merge_size**2
            ids = (
                [self.vision_start_token_id]
                + [self.image_token_id] * count
                + [self.vision_end_token_id]
            )
            blocks["image"].append((ids, None))
            pixel_values.append(result["pixel_values"])
            image_grids.append(result["image_grid_thw"])
        for source in audio:
            features, duration = self._audio_features(source)
            count = len(features) * self.audio_config.tokens_per_segment
            ids = audio_token_ids(
                self.tokenizer,
                count,
                self.audio_config.second_per_window,
                duration,
                interval,
            )
            blocks["audio"].append((ids, features))
        for index, source in enumerate(videos):
            item = source if isinstance(source, dict) else {"video": source}
            source = item.get("video", item.get("path"))
            video_options = {
                **options,
                **{k: item[k] for k in overrides if item.get(k) is not None},
            }
            result, metadata = self._video(
                source,
                video_options,
                video_metadata[index] if video_metadata is not None else None,
            )
            features, duration, count = None, None, 0
            if item.get("use_audio_in_video", use_audio_in_video):
                waveform = item.get("audio")
                if waveform is None:
                    waveform = self._soundtrack(source)
                features, duration = self._audio_features(waveform, pcm16=False)
                count = len(features) * self.audio_config.tokens_per_segment
            ids = video_token_ids(
                self.tokenizer,
                result["video_grid_thw"][0],
                metadata,
                self.video_processor.temporal_patch_size,
                self.video_processor.merge_size,
                count,
                duration,
                self.audio_config.second_per_window,
            )
            blocks["video"].append((ids, features))
            video_values.append(result["pixel_values_videos"])
            video_grids.append(result["video_grid_thw"])

        # Consume complete placeholder blocks in prompt order. This also keeps
        # standalone audio and video soundtrack features in the same order.
        source_ids = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        patterns = {
            kind: self.tokenizer.encode(value, add_special_tokens=False)
            for kind, value in PLACEHOLDERS.items()
        }
        offsets = dict.fromkeys(patterns, 0)
        ids, features, index = [], [], 0
        while index < len(source_ids):
            for kind, pattern in patterns.items():
                if source_ids[index : index + len(pattern)] == pattern:
                    if offsets[kind] >= len(blocks[kind]):
                        raise ValueError(f"Extra {kind} placeholder in the prompt.")
                    block, feature = blocks[kind][offsets[kind]]
                    ids.extend(block)
                    if feature is not None:
                        features.append(feature)
                    offsets[kind] += 1
                    index += len(pattern)
                    break
            else:
                ids.append(source_ids[index])
                index += 1
        if any(offsets[kind] != len(blocks[kind]) for kind in blocks):
            raise ValueError("Media items and prompt placeholders differ.")
        data = {
            "input_ids": np.asarray([ids], dtype=np.int64),
            "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
        }
        for values, grids, value_key, grid_key in (
            (pixel_values, image_grids, "pixel_values", "image_grid_thw"),
            (video_values, video_grids, "pixel_values_videos", "video_grid_thw"),
        ):
            if values:
                data[value_key], data[grid_key] = np.concatenate(
                    values
                ), np.concatenate(grids)
        if features:
            data["audio_feature"] = np.concatenate(features)
        if return_tensors not in ("mlx", "np", None):
            raise ValueError("WeMM-Video supports return_tensors='mlx' or 'np'.")
        return BatchFeature(data=to_mlx(data) if return_tensors == "mlx" else data)

    @property
    def model_input_names(self):
        return [
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
            "audio_feature",
        ]
