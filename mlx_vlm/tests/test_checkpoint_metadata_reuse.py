import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx_vlm.server import _checkpoint_worker as worker
from mlx_vlm.server import checkpoint_store as store


@pytest.fixture
def snapshots(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights.safetensors").write_bytes(b"weights" * 1024 + b"tail")
    (source / "tokenizer_config.json").write_text('{"template":"old"}')
    (source / "obsolete.json").write_text("{}")
    names = ["weights.safetensors", "tokenizer_config.json", "obsolete.json"]
    old, manifest = store.prepare_snapshot(
        source, names, tmp_path / "receiver", chunk=1024
    )
    return source, names, old, manifest


def updated(snapshots, root):
    source, names, old, manifest = snapshots
    names.remove("obsolete.json")
    (source / "tokenizer_config.json").write_text('{"template":"new"}')
    final, new = store.prepare_snapshot(source, names, root, chunk=1024)
    return final, new


def test_metadata_change_reuses_source_weights_without_stream_or_rescan(
    snapshots, monkeypatch
):
    source, names, old, manifest = snapshots
    names.remove("obsolete.json")
    (source / "tokenizer_config.json").write_text('{"template":"new"}')
    # APFS reuse must not read bytes; a normal copy still hashes its fallback.
    clone = store._clone
    cloned = []

    def cloning(src, dst):
        cloned.append(src)
        return clone(src, dst)

    monkeypatch.setattr(store, "_clone", cloning)
    stream = lambda *a: pytest.fail(
        "Metadata change must not enter full streaming preparation"
    )
    final, new = store.prepare_snapshot(
        source, names, old.parent, chunk=1024, stream=stream
    )
    assert old / "weights.safetensors" in cloned
    assert source / "weights.safetensors" not in cloned
    assert (
        new["files"]["weights.safetensors"] == manifest["files"]["weights.safetensors"]
    )
    assert final != old
    assert not (final / "obsolete.json").exists()
    assert (old / "tokenizer_config.json").read_text() == '{"template":"old"}'


def test_metadata_only_payload_and_independent_publication(
    snapshots, tmp_path, monkeypatch
):
    import mlx.core as mx
    import numpy as np

    _, _, old, previous = snapshots
    final, manifest = updated(snapshots, tmp_path / "sender")
    receiver = store.SnapshotStore(old.parent)
    missing, reused = receiver.missing(manifest)
    assert reused == previous["files"]["weights.safetensors"]["size"]
    assert missing["weights.safetensors"] == []
    partial = receiver.partial(manifest)
    assert (partial / "weights.safetensors").stat().st_ino != (
        old / "weights.safetensors"
    ).stat().st_ino
    payloads = [
        (final / name).read_bytes()[i * 1024 : (i + 1) * 1024]
        for name in manifest["files"]
        for i in missing[name]
    ]
    assert sum(map(len, payloads)) == len(
        (final / "tokenizer_config.json").read_bytes()
    )
    payloads = iter(payloads)
    monkeypatch.setattr(
        mx.distributed,
        "recv",
        lambda *a, **kw: mx.array(np.frombuffer(next(payloads), dtype=np.uint8)),
    )
    command = dict(manifest=manifest, missing=missing, path=str(partial))
    signatures = worker.copy_chunks(
        SimpleNamespace(rank=lambda: 1),
        command,
        lambda **kw: None,
        reused_signatures=receiver.reused_signatures,
    )
    published = receiver.publish(manifest, signatures)
    assert receiver.existing(manifest) == published
    assert not (published / "obsolete.json").exists()
    assert (published / "weights.safetensors").read_bytes() == (
        old / "weights.safetensors"
    ).read_bytes()


def test_cold_pipeline_seeds_verified_weights_by_expected_digest(snapshots, tmp_path):
    _, _, old, previous = snapshots
    final, manifest = updated(snapshots, tmp_path / "sender")
    receiver = store.SnapshotStore(old.parent, partial_id="a" * 64)
    layout = {
        "chunk_bytes": 1024,
        "files": {
            n: {"size": r["size"], "expected_sha256": r["sha256"]}
            for n, r in manifest["files"].items()
        },
    }
    receiver.seed_partial(layout["files"], 1024)
    hashes = worker.partial_hashes(receiver.partial(None), layout, lambda **kw: None)
    assert (
        hashes["weights.safetensors"]
        == previous["files"]["weights.safetensors"]["chunks"]
    )
    assert hashes["tokenizer_config.json"] == [None]


@pytest.mark.parametrize("mutation", ["bytes", "symlink", "receipt", "chunks"])
def test_invalid_donor_is_not_reused(snapshots, tmp_path, mutation):
    _, _, old, previous = snapshots
    _, manifest = updated(snapshots, tmp_path / "sender")
    path = old / "weights.safetensors"
    if mutation == "bytes":
        before = path.stat()
        path.chmod(0o600)
        path.write_bytes(b"x" * before.st_size)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        path.chmod(0o444)
    elif mutation == "symlink":
        old.chmod(0o700)
        path.unlink()
        path.symlink_to(tmp_path / "elsewhere")
    elif mutation == "receipt":
        (old.parent / ".verified" / (old.name + ".json")).write_text("{}")
    else:
        manifest["files"]["weights.safetensors"]["chunks"][0] = "0" * 64
    receiver = store.SnapshotStore(old.parent)
    missing, reused = receiver.missing(manifest)
    assert reused == 0
    assert missing["weights.safetensors"]


def test_changed_weights_cannot_reuse_by_name_or_size(snapshots, tmp_path):
    source, names, old, previous = snapshots
    p = source / "weights.safetensors"
    p.write_bytes(b"x" * p.stat().st_size)
    _, manifest = store.prepare_snapshot(source, names, tmp_path / "sender", chunk=1024)
    receiver = store.SnapshotStore(old.parent)
    missing, _ = receiver.missing(manifest)
    assert missing["weights.safetensors"] == list(
        range(len(manifest["files"]["weights.safetensors"]["chunks"]))
    )


def test_reused_file_modified_before_commit_is_rejected(snapshots, tmp_path):
    _, _, old, previous = snapshots
    _, manifest = updated(snapshots, tmp_path / "sender")
    receiver = store.SnapshotStore(old.parent)
    receiver.missing(manifest)
    signatures = dict(receiver.reused_signatures)
    p = receiver.partial(manifest) / "weights.safetensors"
    p.write_bytes(b"x" * p.stat().st_size)
    with pytest.raises(ValueError, match="changed after chunk verification"):
        receiver.publish(manifest, signatures)


def test_cancelled_reuse_preserves_original_and_retry(snapshots, tmp_path):
    _, _, old, previous = snapshots
    _, manifest = updated(snapshots, tmp_path / "sender")
    receiver = store.SnapshotStore(old.parent)
    stop = threading.Event()
    stop.set()
    with pytest.raises(InterruptedError):
        receiver.missing(manifest, stop=stop)
    assert receiver.existing(previous) == old
    missing, reused = receiver.missing(manifest)
    assert reused == previous["files"]["weights.safetensors"]["size"]
    assert missing["weights.safetensors"] == []


def test_interrupted_copy_scratch_does_not_block_retry(snapshots, tmp_path):
    _, _, old, previous = snapshots
    _, manifest = updated(snapshots, tmp_path / "sender")
    receiver = store.SnapshotStore(old.parent)
    scratch = receiver.partial(manifest) / (".reuse-" + "b" * 32)
    scratch.write_bytes(b"unverified interrupted copy")
    missing, reused = receiver.missing(manifest)
    assert not scratch.exists()
    assert missing["weights.safetensors"] == []


@pytest.mark.skipif(__import__("sys").platform != "darwin", reason="APFS copy-on-write")
def test_metadata_change_does_not_read_sealed_weights_on_apfs(snapshots, monkeypatch):
    source, names, old, manifest = snapshots
    names.remove("obsolete.json")
    (source / "tokenizer_config.json").write_text('{"template":"new"}')
    original = Path.open

    def guarded(path, *args, **kw):
        if path.name == "weights.safetensors":
            pytest.fail("Verified APFS files must not be read again")
        return original(path, *args, **kw)

    monkeypatch.setattr(Path, "open", guarded)
    final, manifest = store.prepare_snapshot(source, names, old.parent, chunk=1024)
    # Hide only the exact snapshot receipt so the old snapshot is the donor.
    (old.parent / ".verified" / (final.name + ".json")).unlink()
    receiver = store.SnapshotStore(old.parent)
    missing, reused = receiver.missing(manifest)
    assert missing["weights.safetensors"] == []
