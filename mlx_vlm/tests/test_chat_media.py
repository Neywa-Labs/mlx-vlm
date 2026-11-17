"""Chat UI attachment routing: images, videos and audio go to their own inputs."""

import pytest

from mlx_vlm.chat_media import extract_media_from_message, media_kind
from mlx_vlm.prompt_utils import get_message_json


@pytest.mark.parametrize(
    "path, mime_type, expected",
    [
        ("/tmp/gradio/abc/launch.v4.mp4", None, "video"),
        ("/tmp/clip.MOV", None, "video"),
        ("/tmp/clip.webm", None, "video"),
        ("/tmp/speech.wav", None, "audio"),
        ("/tmp/speech.m4a", None, "audio"),
        ("/tmp/photo.jpg", None, "image"),
        ("/tmp/photo.png", None, "image"),
        ("/tmp/no_suffix", None, "image"),
        # Gradio's reported MIME type wins over the suffix.
        ("/tmp/upload.bin", "video/mp4", "video"),
        ("/tmp/upload.bin", "audio/mpeg", "audio"),
        ("/tmp/upload.bin", "application/octet-stream", "image"),
    ],
)
def test_media_kind(path, mime_type, expected):
    assert media_kind(path, mime_type) == expected


def test_extract_media_splits_files_by_kind_in_order():
    message = {
        "text": "describe these",
        "files": [
            {"path": "/tmp/a.png"},
            "/tmp/clip.mp4",
            {"path": "/tmp/voice.wav", "mime_type": "audio/wav"},
            {"path": "/tmp/b.jpg"},
        ],
    }

    media = extract_media_from_message(message)

    assert media.images == ["/tmp/a.png", "/tmp/b.jpg"]
    assert media.videos == ["/tmp/clip.mp4"]
    assert media.audio == ["/tmp/voice.wav"]


@pytest.mark.parametrize(
    "message",
    [
        "a plain text message is not a file path",
        {"text": "no attachments", "files": []},
        {"text": "missing path", "files": [{"path": None}]},
        None,
    ],
)
def test_extract_media_without_attachments(message):
    media = extract_media_from_message(message)
    assert media.images == media.videos == media.audio == []


def test_extract_media_accepts_single_file_key():
    media = extract_media_from_message({"file": {"path": "/tmp/clip.mov"}})
    assert media.videos == ["/tmp/clip.mov"]


def test_lfm3_omni_message_carries_video_and_audio_parts():
    message = get_message_json(
        "lfm3_omni",
        "What happens here?",
        num_images=0,
        num_audios=1,
        video=["/tmp/clip.mp4"],
    )

    kinds = [part["type"] for part in message["content"]]
    assert kinds.count("video") == 1
    assert kinds.count("audio") == 1
    assert kinds[-1] == "text"
