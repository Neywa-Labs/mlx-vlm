# Copyright © 2026 Neywa Labs Inc.

import ctypes
import errno
import hashlib
import json
import os
import queue
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from mlx_vlm.server import checkpoint_store as store
from mlx_vlm.server import checkpoint_transfer as transfer
from mlx_vlm.server._checkpoint_worker import RDMAUnavailable, connection_call
from mlx_vlm.server.checkpoint_transfer import (
    Helper,
    Pair,
    TransferPool,
    TransferUnavailable,
    bootstrap,
)
from mlx_vlm.server.cluster_types import PrepareModel, RDMAPair
from mlx_vlm.server.distributed import ClusterConfig, ModelStager, Worker


@pytest.fixture
def pool():
    value = TransferPool()
    yield value
    value.close()


@pytest.fixture
def snapshot(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "weights.safetensors").write_bytes(bytes(range(256)) * 35 + b"tail")
    names = ["config.json", "weights.safetensors"]
    root = tmp_path / "cache"
    final, manifest = store.prepare_snapshot(source, names, root, chunk=1024)
    return source, names, root, final, manifest


def test_manifest_reuse_does_not_read_weight_bytes(snapshot, monkeypatch):
    source, names, root, final, manifest = snapshot
    original = Path.open

    def guarded(path, *args, **kwargs):
        if path.suffix == ".safetensors":
            raise AssertionError("Warm preparation must not rescan weights")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    assert store.prepare_snapshot(source, names, root, chunk=1024) == (final, manifest)
    assert store.SnapshotStore(root).existing(manifest) == final
    assert not final.stat().st_mode & 0o222


def test_snapshot_does_not_modify_or_hardlink_source(snapshot):
    source, names, root, final, manifest = snapshot
    for name in names:
        assert (source / name).stat().st_ino != (final / name).stat().st_ino
        assert (source / name).stat().st_mode & 0o200
        assert not (final / name).stat().st_mode & 0o222
    (source / names[1]).write_bytes(b"updated")
    new_final, new_manifest = store.prepare_snapshot(source, names, root, chunk=1024)
    assert new_final != final and new_manifest != manifest
    assert (final / names[1]).read_bytes().endswith(b"tail")


def test_same_size_mutation_with_restored_mtime_cannot_reuse_receipt(snapshot):
    _, _, root, final, manifest = snapshot
    path = final / "weights.safetensors"
    before = path.stat()
    path.chmod(0o644)
    path.write_bytes(b"X" * before.st_size)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    path.chmod(0o444)
    with pytest.raises(ValueError, match="checksum mismatch"):
        store.SnapshotStore(root).existing(manifest)


def test_resume_checks_partial_chunks_and_tail(snapshot, tmp_path):
    _, _, _, final, manifest = snapshot
    destination = store.SnapshotStore(tmp_path / "receiver")
    partial = destination.partial(manifest)
    data = (final / "weights.safetensors").read_bytes()
    (partial / "weights.safetensors").write_bytes(
        data[:1024] + b"X" * 1024 + data[2048:3072]
    )
    missing, reused = destination.missing(manifest)
    assert reused == 2048
    assert missing["weights.safetensors"] == [
        1,
        *range(3, len(manifest["files"]["weights.safetensors"]["chunks"])),
    ]
    assert missing["config.json"] == [0]


@pytest.mark.parametrize(
    "name", ["../escape", "/escape", "a/../escape", ".hidden", "a//b"]
)
def test_manifest_paths_cannot_escape(name, snapshot):
    manifest = json.loads(json.dumps(snapshot[-1]))
    manifest["files"][name] = manifest["files"].pop("config.json")
    with pytest.raises(ValueError):
        store.validate(manifest)


def test_symlinked_partial_is_not_followed(snapshot, tmp_path):
    manifest = snapshot[-1]
    destination = store.SnapshotStore(tmp_path / "receiver")
    partial = destination.partial(manifest)
    outside = tmp_path / "outside"
    outside.write_text("untouched")
    (partial / "config.json").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        destination.missing(manifest)
    assert outside.read_text() == "untouched"


def test_readonly_rsync_partial_can_be_resumed(snapshot, tmp_path):
    manifest = snapshot[-1]
    destination = store.SnapshotStore(tmp_path / "receiver")
    partial = destination.partial(manifest)
    file = partial / "config.json"
    file.write_bytes((snapshot[-2] / "config.json").read_bytes())
    file.chmod(0o444)
    partial.chmod(0o555)
    destination.writable_partial(manifest)
    assert partial.stat().st_mode & 0o200 and file.stat().st_mode & 0o200
    missing, reused = destination.missing(manifest)
    assert not missing["config.json"] and reused == file.stat().st_size


@pytest.mark.parametrize(
    "name,error",
    [("config.json", "changed after"), ("unexpected.py", "Unexpected file")],
)
def test_publish_rejects_unverified_changes(snapshot, tmp_path, name, error):
    manifest = snapshot[-1]
    destination = store.SnapshotStore(tmp_path / "receiver")
    partial = destination.partial(manifest)
    signatures = {}
    for filename in manifest["files"]:
        (partial / filename).write_bytes((snapshot[-2] / filename).read_bytes())
        signatures[filename] = store.signature(partial / filename)
    (partial / name).write_text("unverified content")
    with pytest.raises(ValueError, match=error):
        destination.publish(manifest, signatures)
    assert not (destination.root / store.snapshot_id(manifest)).exists()


def test_snapshot_root_symlink_is_rejected(snapshot, tmp_path):
    manifest = snapshot[-1]
    destination = store.SnapshotStore(tmp_path / "receiver")
    (destination.root / store.snapshot_id(manifest)).symlink_to(snapshot[-2])
    with pytest.raises(ValueError, match="symlink"):
        destination.existing(manifest)


def test_cancelled_preparation_does_not_publish(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights.safetensors").write_bytes(b"weights")
    stop = threading.Event()
    stop.set()
    with pytest.raises(InterruptedError):
        store.prepare_snapshot(
            source, ["weights.safetensors"], tmp_path / "cache", stop=stop
        )
    assert not list((tmp_path / "cache").glob(".building-*"))


def test_helper_identity_mismatch_closes_owned_process():
    with pytest.raises(RuntimeError, match="identity"):
        Helper(
            [sys.executable, "-u", "-c", bootstrap()],
            timeout=5,
            physical_id="wrong-device",
        )


def test_explicit_rdma_never_silently_falls_back(snapshot, tmp_path, pool):
    _, _, root, final, manifest = snapshot
    worker = Worker("http://local-test", model_dir=str(tmp_path / "receiver"))
    config = ClusterConfig((worker,), root, tmp_path / "lock", python=sys.executable)
    with pytest.raises(RuntimeError, match="No direct RDMA"):
        pool.deliver(
            config, worker, final, manifest, "jaccl_rdma", lambda *a, **k: None
        )
    assert not pool.pairs
    assert not (Path(worker.model_dir) / store.snapshot_id(manifest)).exists()


@pytest.mark.parametrize("transport", ["auto", "ssh_rsync"])
def test_preparation_transports_share_verified_snapshots_and_pool(
    tmp_path, transport, pool
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "weights.safetensors").write_bytes(b"actual checkpoint")
    worker = Worker("http://local-test", model_dir=str(tmp_path / "worker"))
    config = ClusterConfig(
        (worker,), tmp_path / "cache", tmp_path / "lock", python=sys.executable
    )
    stager = ModelStager(config, transfers=pool)
    updates = []
    report = lambda phase, **data: updates.append((phase, data))
    first = stager.prepare(
        str(source), validate_sharding=False, transport=transport, progress=report
    )
    assert stager.worker_transfers[worker.url]["transport"] == "ssh_rsync"
    assert any(p == "transport_fallback" for p, _ in updates) is (transport == "auto")
    assert any(p == "verifying_worker" for p, _ in updates)
    path = Path(stager.worker_paths[(first, worker.url)])
    assert path.parent == Path(worker.model_dir) and path != Path(first)
    assert (path / "weights.safetensors").read_bytes() == (
        source / "weights.safetensors"
    ).read_bytes()
    assert not list(Path(worker.model_dir).glob(".partial-*"))
    pair = next(iter(pool.pairs.values()))
    pid = pair.remote.pid
    second = ModelStager(config, transfers=pool)
    assert (
        second.prepare(str(source), validate_sharding=False, transport="jaccl_rdma")
        == first
    )
    assert second.worker_transfers[worker.url]["transport"] == "verified_cache"
    # Switching transport, including ordinary inference staging, uses the
    # same snapshot without a connection check or checkpoint retransmission.
    assert (
        second.prepare(str(source), validate_sharding=False, progress=report) == first
    )
    assert pair.remote.pid == pid
    assert updates[-1][1]["bytes_transferred"] == 0
    assert updates[-1][1]["bytes_reused"] == sum(
        p.stat().st_size for p in source.iterdir()
    )
    pool.close()
    assert pair.remote.process.poll() is not None


def test_transport_contract_and_topology_validation():
    assert PrepareModel(model="org/model").transport == "ssh_rsync"
    with pytest.raises(ValueError):
        PrepareModel(model="org/model", transport="http")
    with pytest.raises(ValueError):
        RDMAPair(
            local_address="not-an-ip", local_device="rdma_en4", remote_device="rdma_en5"
        )


def test_chunk_hashes_and_whole_file_identity(snapshot):
    manifest = snapshot[-1]
    data = (snapshot[-2] / "weights.safetensors").read_bytes()
    record = manifest["files"]["weights.safetensors"]
    assert record["sha256"] == hashlib.sha256(data).hexdigest()
    assert record["chunks"] == [
        hashlib.sha256(data[i : i + 1024]).hexdigest()
        for i in range(0, len(data), 1024)
    ]


@pytest.mark.parametrize("code", [errno.EXDEV, errno.ENOTSUP, errno.ENOSYS])
def test_unsupported_clone_uses_verified_copy(tmp_path, monkeypatch, code):
    clone = Mock(side_effect=lambda *args: (ctypes.set_errno(code), -1)[1])
    monkeypatch.setattr(store.sys, "platform", "darwin")
    monkeypatch.setattr(
        store.ctypes, "CDLL", lambda *a, **kw: SimpleNamespace(clonefile=clone)
    )
    source = tmp_path / "source"
    source.mkdir()
    data = b"checkpoint content"
    (source / "weights.safetensors").write_bytes(data)
    final, manifest = store.prepare_snapshot(
        source, ["weights.safetensors"], tmp_path / "cache"
    )
    assert (final / "weights.safetensors").read_bytes() == data
    assert (
        manifest["files"]["weights.safetensors"]["sha256"]
        == hashlib.sha256(data).hexdigest()
    )
    clone.assert_called_once()


@pytest.mark.parametrize(
    "code", [errno.EACCES, errno.EPERM, errno.ENOSPC, errno.EIO, errno.EINVAL]
)
def test_clone_errors_abort_instead_of_copying(tmp_path, monkeypatch, code):
    clone = Mock(side_effect=lambda *args: (ctypes.set_errno(code), -1)[1])
    monkeypatch.setattr(store.sys, "platform", "darwin")
    monkeypatch.setattr(
        store.ctypes, "CDLL", lambda *a, **kw: SimpleNamespace(clonefile=clone)
    )
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights.safetensors").write_bytes(b"checkpoint content")
    root = tmp_path / "cache"
    with pytest.raises(OSError) as caught:
        store.prepare_snapshot(source, ["weights.safetensors"], root)
    assert caught.value.errno == code
    assert not list(root.glob(".building-*"))
    assert not list(root.rglob("weights.safetensors"))


@pytest.mark.parametrize(
    "message",
    [
        "[jaccl] Recv failed with errno=2",
        "[jaccl] Send failed with errno=54",
        "[jaccl] Couldn't connect (error: 61)",
        "[jaccl] Accept failed (error: 60)",
        "[jaccl] Could not find device rdma_nonexistent (6 available)",
    ],
)
def test_native_connection_failures_are_classified(message):
    error = RuntimeError(message)
    with pytest.raises(RDMAUnavailable, match="jaccl") as caught:
        connection_call(Mock(side_effect=error))
    assert caught.value.__cause__ is error


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("[jaccl] Unsupported dtype for JACCL operation"),
        RuntimeError(
            "[jaccl] The configuration should define a valid mesh or a valid ring."
        ),
        RuntimeError("[jaccl] Register memory region failed"),
        RuntimeError("unrecognized future backend failure"),
        ValueError("RDMA connection challenge mismatch"),
        TypeError("unexpected backend keyword"),
        ImportError("missing dependency"),
        OSError(errno.ENOSPC, "disk full"),
    ],
)
def test_native_defects_are_not_classified_as_connection_failures(error):
    with pytest.raises(type(error)) as caught:
        connection_call(Mock(side_effect=error))
    assert caught.value is error


@pytest.mark.parametrize(
    "checking,code,expected",
    [
        (True, "rdma_unavailable", TransferUnavailable),
        (False, "rdma_unavailable", RuntimeError),
        (True, "helper_error", RuntimeError),
        (True, "unknown_error", RuntimeError),
    ],
)
def test_helper_fallback_requires_explicit_connection_error(checking, code, expected):
    helper = object.__new__(Helper)
    helper.timeout = 1
    helper.events = queue.Queue()
    helper.events.put(
        {"event": "error", "id": "test", "message": "failure", "code": code}
    )
    with pytest.raises(expected) as caught:
        helper.wait("test", connection_check=checking)
    assert type(caught.value) is expected


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("Incompatible checkpoint helper protocol"),
        TimeoutError("Helper bootstrap timed out"),
        TypeError("Unexpected implementation error"),
    ],
)
def test_local_helper_startup_errors_do_not_become_fallback(
    connecting_pair, error, monkeypatch
):
    pair, _ = connecting_pair
    monkeypatch.setattr(transfer, "Helper", Mock(side_effect=error))
    with pytest.raises(type(error)) as caught:
        pair.connect()
    assert caught.value is error


@pytest.fixture
def connecting_pair(monkeypatch):
    pair = object.__new__(Pair)
    pair.validated = False
    pair.config = SimpleNamespace(timeout=1)
    pair.worker = SimpleNamespace(
        ssh="peer",
        rdma={
            "local_address": "127.0.0.1",
            "local_device": "rdma_en4",
            "remote_device": "rdma_en5",
        },
    )
    pair.remote = Mock()
    pair.local = None
    local = Mock()
    monkeypatch.setattr(transfer, "Helper", Mock(return_value=local))
    return pair, local


@pytest.mark.parametrize(
    "error,expected",
    [
        (TimeoutError("Connection timed out"), TransferUnavailable),
        (TransferUnavailable("JACCL unavailable"), TransferUnavailable),
        (RuntimeError("Incompatible checkpoint helper protocol"), RuntimeError),
        (ValueError("RDMA connection challenge mismatch"), ValueError),
        (TypeError("unexpected backend keyword"), TypeError),
        (InterruptedError("Checkpoint preparation cancelled"), InterruptedError),
    ],
)
def test_connection_check_only_converts_expected_failures(
    connecting_pair, error, expected
):
    pair, local = connecting_pair

    def wait(*args, **kwargs):
        if kwargs.get("event") == "initializing":
            return {}
        raise error

    local.wait.side_effect = wait
    with pytest.raises(expected) as caught:
        pair.connect()
    assert type(caught.value) is expected
    assert not pair.validated
    local.close.assert_called_once()
    pair.remote.close.assert_called_once()


def test_initialization_ack_timeout_does_not_fallback(connecting_pair):
    pair, local = connecting_pair
    error = TimeoutError("Helper did not acknowledge initialization")
    local.wait.side_effect = error
    try:
        with pytest.raises(TimeoutError) as caught:
            pair.connect()
        assert caught.value is error
        assert not pair.validated
    finally:
        pair.close()


@pytest.mark.parametrize(
    "code,expected",
    [
        (errno.EADDRNOTAVAIL, TransferUnavailable),
        (errno.ENETDOWN, TransferUnavailable),
        (errno.EACCES, PermissionError),
        (errno.EMFILE, OSError),
    ],
)
def test_bind_only_falls_back_for_unavailable_network(
    connecting_pair, monkeypatch, code, expected
):
    pair, _ = connecting_pair
    socket = MagicMock()
    socket.__enter__.return_value.bind.side_effect = OSError(code, os.strerror(code))
    monkeypatch.setattr(transfer.socket, "socket", Mock(return_value=socket))
    try:
        with pytest.raises(expected) as caught:
            pair.connect()
        assert type(caught.value) is expected
        assert not pair.validated
    finally:
        pair.close()


@pytest.mark.parametrize(
    "error",
    [
        TransferUnavailable("Connection refused"),
        RuntimeError("Unexpected checkpoint helper command sequence"),
        ValueError("RDMA connection challenge mismatch"),
        TypeError("unexpected backend keyword"),
        TimeoutError("Helper bootstrap timed out"),
        InterruptedError("Checkpoint preparation cancelled"),
    ],
)
def test_auto_falls_back_only_on_classified_connection_errors(
    snapshot, tmp_path, monkeypatch, error, pool
):
    _, _, root, final, manifest = snapshot
    worker = Worker(
        "http://local-test",
        model_dir=str(tmp_path / "receiver"),
        rdma={
            "local_address": "127.0.0.1",
            "local_device": "rdma_en4",
            "remote_device": "rdma_en5",
        },
    )
    config = ClusterConfig((worker,), root, tmp_path / "lock", python=sys.executable)
    monkeypatch.setattr(Pair, "connect", Mock(side_effect=error))
    updates = []

    def deliver():
        return pool.deliver(
            config,
            worker,
            final,
            manifest,
            "auto",
            lambda phase, **data: updates.append((phase, data)),
        )

    if isinstance(error, TransferUnavailable):
        result = deliver()
        assert result["transport"] == "ssh_rsync" and result["verified"]
        assert result["fallback_reason"] == str(error)
        assert sum(phase == "transport_fallback" for phase, _ in updates) == 1
        assert (Path(result["path"]) / "weights.safetensors").read_bytes() == (
            final / "weights.safetensors"
        ).read_bytes()
    else:
        with pytest.raises(type(error)) as caught:
            deliver()
        assert caught.value is error
        assert not any(phase == "transport_fallback" for phase, _ in updates)
        assert not pool.pairs
        assert not (Path(worker.model_dir) / store.snapshot_id(manifest)).exists()
