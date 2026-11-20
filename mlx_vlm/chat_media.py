"""Route files attached in the Gradio chat UI to the right input kind.

``chat_ui`` loads a model at import time, so this logic lives in its own
module to stay importable (and testable) without Gradio or a model.
"""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass, field
from pathlib import Path

VIDEO_SUFFIXES = frozenset(
    {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".mpeg", ".mpg"}
)
AUDIO_SUFFIXES = frozenset(
    {".wav", ".mp3", ".flac", ".ogg", ".oga", ".m4a", ".aac", ".opus"}
)


@dataclass
class MessageMedia:
    """Attachments of one chat message, split by input kind."""

    images: list[str] = field(default_factory=list)
    videos: list[str] = field(default_factory=list)
    audio: list[str] = field(default_factory=list)


def media_kind(path: str, mime_type: str | None = None) -> str:
    """Return ``"video"``, ``"audio"`` or ``"image"`` for an attached file.

    The MIME type Gradio reports wins; otherwise the file suffix decides.
    Anything unrecognised stays an image, which was the UI's only kind before.
    """
    mime = mime_type or mimetypes.guess_type(path)[0] or ""
    if mime.startswith("video/"):
        return "video"
    if mime.startswith("audio/"):
        return "audio"
    if mime.startswith("image/"):
        return "image"
    suffix = Path(path).suffix.lower()
    if suffix in VIDEO_SUFFIXES:
        return "video"
    if suffix in AUDIO_SUFFIXES:
        return "audio"
    return "image"


def _file_entries(message) -> list:
    if not isinstance(message, dict):
        return []
    files = message.get("files")
    if files:
        return list(files)
    file = message.get("file")
    return [file] if file else []


def extract_media_from_message(message) -> MessageMedia:
    """Collect every attachment of a multimodal chat message, in order."""
    media = MessageMedia()
    buckets = {"image": media.images, "video": media.videos, "audio": media.audio}
    for entry in _file_entries(message):
        if isinstance(entry, dict):
            path, mime_type = entry.get("path"), entry.get("mime_type")
        else:
            path, mime_type = entry, None
        if not path:
            continue
        path = str(path)
        buckets[media_kind(path, mime_type)].append(path)
    return media
