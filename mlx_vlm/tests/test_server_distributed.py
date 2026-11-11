# Copyright © 2026 Neywa Labs Inc.
# SPDX-License-Identifier: MIT

"""Distributed request routing and staging without GPUs, downloads or SSH."""

import asyncio
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import mlx_vlm.server as server
import mlx_vlm.utils as utils
from mlx_vlm.server import cli, generation
from mlx_vlm.server.distributed import (
    CANCEL_ROUTE,
    MANIFEST,
    ClusterConfig,
    Coordinator,
    DistributedWorker,
    ModelStager,
    Worker,
    _manifest,
    _semantic,
    cluster_lock,
)
from mlx_vlm.server.distributed_schedule import RequestScope, request_scope, schedule
from mlx_vlm.server.rsync_progress import RsyncProgress, run_rsync
from mlx_vlm.server.runtime import runtime


@pytest.fixture
def cluster(tmp_path):
    return ClusterConfig(
        (Worker("http://rank0"), Worker("http://rank1", "mac-two")),
        tmp_path / "models",
        tmp_path / "cluster.lock",
        python=sys.executable,
    )


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text('{"model_type":"test"}')
    (source / "model.safetensors").write_bytes(b"fixture weights")
    (source / "tokenizer.json").write_text("{}")
    architecture = NS(Model=NS(shard=lambda: None))
    monkeypatch.setattr(utils, "get_model_and_args", lambda *_: (architecture, "test"))
    return source


def test_staging_reuses_local_content_and_resolves_hub_only_once(
    cluster, checkpoint, monkeypatch
):
    resolve = Mock(return_value=checkpoint)
    monkeypatch.setattr(utils, "get_model_path", resolve)
    stager = ModelStager(cluster)
    remote = Mock(return_value=0)  # Peer already has the verified snapshot.
    monkeypatch.setattr(stager, "_remote", remote)
    result = Path(stager.prepare("org/model", "revision-a"))
    assert result.parent == cluster.model_dir
    assert (result / "model.safetensors").read_bytes() == b"fixture weights"
    assert json.loads((result / MANIFEST).read_text()) == _manifest(checkpoint)
    assert stager.prepare("org/model", "revision-a") == str(result)
    resolve.assert_called_once_with("org/model", revision="revision-a")
    remote.assert_called_once()


def test_hub_symlinks_are_materialized(cluster, checkpoint, tmp_path, monkeypatch):
    weights = checkpoint / "model.safetensors"
    blob = tmp_path / "blob"
    weights.rename(blob)
    weights.symlink_to(blob)
    stager = ModelStager(cluster)
    monkeypatch.setattr(stager, "_remote", lambda *a, **k: 0)
    result = Path(stager.prepare(str(checkpoint)))
    assert not (result / weights.name).is_symlink()
    assert (result / weights.name).read_bytes() == blob.read_bytes()


def test_incomplete_or_unsupported_checkpoint_does_not_transfer(
    cluster, checkpoint, monkeypatch
):
    stager = ModelStager(cluster)
    remote = Mock()
    monkeypatch.setattr(stager, "_remote", remote)
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "missing.safetensors"}})
    )
    with pytest.raises(ValueError, match="Incomplete checkpoint"):
        stager.prepare(str(checkpoint))
    remote.assert_not_called()
    monkeypatch.setattr(
        utils, "get_model_and_args", lambda *_: (NS(Model=object), "test")
    )
    with pytest.raises(ValueError, match="no distributed shard"):
        stager.prepare(str(checkpoint))


def test_transfer_failure_never_marks_model_prepared(cluster, checkpoint, monkeypatch):
    stager = ModelStager(cluster)
    monkeypatch.setattr(
        stager, "_remote", lambda *a, **k: 3 if a[1][-1] == "verify" else 0
    )
    original = stager._run
    transfers = []

    def run(args, **kwargs):
        if args[0] == "rsync":
            transfers.append(args)
            return subprocess.CompletedProcess(args, 1, "", "transfer interrupted")
        return original(args, **kwargs)

    monkeypatch.setattr(stager, "_run", run)
    with pytest.raises(RuntimeError, match="transfer interrupted"):
        stager.prepare(str(checkpoint))
    assert not stager.prepared
    assert "--checksum" in transfers[0] and "-aL" in transfers[0]
    assert transfers[0][-1].startswith("mac-two:")


@pytest.mark.parametrize("chunk_size", [1, 7, 1024])
def test_rsync_progress_handles_file_boundaries_reuse_and_carriage_returns(chunk_size):
    updates = []
    tracker = RsyncProgress(3100, lambda **data: updates.append(data), interval=0)
    wire = (
        b"MLX_VLM_FILE:.d..t......:4096\n"
        b"MLX_VLM_FILE:.f         :100\n"
        b"MLX_VLM_FILE:>f+++++++++:2000\n"
        b"        1,000  50%  1.00MB/s 0:00:01\r"
        b"          900  45%  1.00MB/s 0:00:01\r"
        b"        2,000 100%  1.00MB/s 0:00:00\r"
        b"        2,000 100%  1.00MB/s 0:00:00 (xfr#1, to-chk=1/3)\n"
        b"MLX_VLM_FILE:>f+++++++:0\n"
        b"            0 100%  0.00KB/s 00:00:00\n"
        b"MLX_VLM_FILE:>f+++++++:1000\n"
        b"          500  50%  1.00MB/s 00:00:01\r"
        b"         1000 100%  1.00MB/s 00:00:00"
    )
    for offset in range(0, len(wire), chunk_size):
        tracker.feed(wire[offset : offset + chunk_size])
    tracker.finish()
    assert updates[-1] == dict(
        bytes_transferred=3000, bytes_reused=100, total_bytes=3100
    )
    assert [u["bytes_transferred"] for u in updates] == sorted(
        u["bytes_transferred"] for u in updates
    )
    assert all(u["bytes_transferred"] + u["bytes_reused"] <= 3100 for u in updates)


def test_rsync_progress_throttles_but_flushes_last_measurement(monkeypatch):
    from mlx_vlm.server import rsync_progress

    clock = [10.0]
    monkeypatch.setattr(rsync_progress.time, "monotonic", lambda: clock[0])
    updates = []
    tracker = RsyncProgress(100, lambda **data: updates.append(data))
    tracker.feed(b"MLX_VLM_FILE:>f+++++++++:100\n10 10% 1MB/s\r")
    tracker.feed(b"20 20% 1MB/s\r30 30% 1MB/s\r")
    assert len(updates) == 1
    clock[0] += 0.5
    tracker.feed(b"40 40% 1MB/s\r")
    tracker.feed(b"50 50% 1MB/s\r")
    tracker.finish()
    assert [u["bytes_transferred"] for u in updates] == [10, 40, 50]
    tracker.finish()
    assert len(updates) == 3


def test_rsync_runner_publishes_before_exit_and_drains_stderr(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    release = tmp_path / "release"
    script = tmp_path / "rsync-fixture"
    script.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, sys, time\n"
        "assert sys.stdin.read() == 'model.safetensors\\n'\n"
        "sys.stderr.write('diagnostic' * 10000); sys.stderr.flush()\n"
        "sys.stdout.write('MLX_VLM_FILE:>f+++++++++:100\\n25 25% 1MB/s\\r'); sys.stdout.flush()\n"
        f"while not pathlib.Path({str(release)!r}).exists(): time.sleep(0.01)\n"
        "sys.stdout.write('100 100% 1MB/s\\r'); sys.stdout.flush()\n"
    )
    script.chmod(0o700)
    received, updates = threading.Event(), []

    def progress(**data):
        updates.append(data)
        received.set()

    with ThreadPoolExecutor(1) as pool:
        job = pool.submit(
            run_rsync,
            [str(script)],
            input="model.safetensors\n",
            timeout=5,
            total_bytes=100,
            progress=progress,
        )
        try:
            assert received.wait(3)
            assert updates[0]["bytes_transferred"] == 25
            assert (
                not job.done()
            )  # Child cannot exit until the consumer observes progress.
        finally:
            release.touch()
        result = job.result(timeout=3)
    assert result.returncode == 0 and len(result.stderr) == 4096
    assert updates[-1]["bytes_transferred"] == 100


@pytest.mark.parametrize("failure", ["timeout", "callback", "exit"])
def test_rsync_runner_failures_do_not_invent_completion_or_leave_processes(
    tmp_path, failure
):
    pid_file = tmp_path / "pid"
    script = tmp_path / "rsync-fixture"
    script.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, sys, time\n"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
        "sys.stdout.write('MLX_VLM_FILE:>f+++++++++:100\\n25 25% 1MB/s\\r'); sys.stdout.flush()\n"
        + (
            "sys.stderr.write('copy interrupted'); sys.exit(23)\n"
            if failure == "exit"
            else "time.sleep(30)\n"
        )
    )
    script.chmod(0o700)
    updates = []

    def progress(**data):
        updates.append(data)
        if failure == "callback":
            raise ValueError("consumer failed")

    options = dict(input="", timeout=1, total_bytes=100, progress=progress)
    if failure == "exit":
        result = run_rsync([str(script)], **options)
        assert result.returncode == 23 and result.stderr == "copy interrupted"
    else:
        with pytest.raises(
            subprocess.TimeoutExpired if failure == "timeout" else ValueError
        ):
            run_rsync([str(script)], **options)
    assert updates and all(u["bytes_transferred"] == 25 for u in updates)
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


def test_real_rsync_progress_reuses_existing_files_and_counts_payload(tmp_path):
    import shutil

    if not shutil.which("rsync"):
        pytest.skip("rsync is not installed")
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "existing").write_bytes(b"same content")
    (destination / "existing").write_bytes(b"same content")
    (source / "new").write_bytes(b"new checkpoint" * 100)
    total = sum(p.stat().st_size for p in source.iterdir())
    updates = []
    result = run_rsync(
        [
            "rsync",
            "-aL",
            "--checksum",
            "--partial",
            "--files-from=-",
            str(source) + "/",
            str(destination) + "/",
        ],
        input="existing\nnew\n",
        timeout=10,
        total_bytes=total,
        progress=lambda **data: updates.append(data),
    )
    assert result.returncode == 0, result.stderr
    assert updates[-1] == dict(
        bytes_transferred=1400, bytes_reused=12, total_bytes=1412
    )
    assert (destination / "new").read_bytes() == (source / "new").read_bytes()


@pytest.mark.parametrize("bad_checksum", [False, True])
def test_staging_progress_is_live_per_worker_and_readiness_requires_verification(
    cluster, checkpoint, monkeypatch, bad_checksum
):
    stager = ModelStager(cluster)
    updates = []

    def remote(worker, args, **kwargs):
        if args[-1] == "verify":
            return 3
        if args[-1] == "commit":
            assert updates[-1][0] == "verifying_worker"
            return 3 if bad_checksum else 0
        return 0

    def copy(args, *, input, timeout, total_bytes, progress):
        assert "--checksum" in args and "--partial" in args
        assert args[-1].startswith("mac-two:")
        for count in (total_bytes // 2, total_bytes):
            progress(bytes_transferred=count, bytes_reused=0, total_bytes=total_bytes)
            assert updates[-1][1]["bytes_transferred"] == count
        assert not any(
            phase == "worker_ready" and data["node_id"] == "b"
            for phase, data in updates
        )
        return subprocess.CompletedProcess(args, 0, "", "")

    from dataclasses import replace

    monkeypatch.setattr(
        stager,
        "config",
        replace(
            cluster,
            workers=(
                replace(cluster.workers[0], node_id="a"),
                replace(cluster.workers[1], node_id="b"),
            ),
        ),
    )
    monkeypatch.setattr(stager, "_remote", remote)
    monkeypatch.setattr("mlx_vlm.server.distributed.run_rsync", copy)
    if bad_checksum:
        with pytest.raises(RuntimeError, match="checksum verification failed"):
            stager.prepare(
                str(checkpoint),
                progress=lambda phase, **data: updates.append((phase, data)),
            )
        assert not stager.prepared and updates[-1][0] == "verifying_worker"
    else:
        stager.prepare(
            str(checkpoint),
            progress=lambda phase, **data: updates.append((phase, data)),
        )
        assert updates[-1][0] == "worker_ready"
    copying = [data for phase, data in updates if phase == "transferring"]
    assert len(copying) == 3 and all(data["node_id"] == "b" for data in copying)
    assert copying[0]["bytes_transferred"] == 0
    assert copying[1]["bytes_transferred"] < copying[1]["total_bytes"]


def test_corrupt_existing_snapshot_is_not_used(cluster, checkpoint, monkeypatch):
    stager = ModelStager(cluster)
    monkeypatch.setattr(stager, "_remote", lambda *a, **k: 0)
    result = Path(stager.prepare(str(checkpoint)))
    (result / "model.safetensors").write_bytes(b"corrupted")
    fresh = ModelStager(cluster)
    with pytest.raises(RuntimeError, match="failed verification"):
        fresh.prepare(str(checkpoint))


def test_selecting_another_model_stages_new_snapshot(cluster, checkpoint, monkeypatch):
    second = checkpoint.parent / "second-model"
    second.mkdir()
    (second / "config.json").write_bytes((checkpoint / "config.json").read_bytes())
    (second / "model.safetensors").write_bytes(b"different weights")
    stager = ModelStager(cluster)
    monkeypatch.setattr(stager, "_remote", lambda *a, **k: 0)
    first_path = stager.prepare(str(checkpoint))
    second_path = stager.prepare(str(second))
    assert first_path != second_path
    assert (
        Path(second_path) / "model.safetensors"
    ).read_bytes() == b"different weights"
    assert stager.prepare(str(checkpoint)) == first_path


@pytest.mark.parametrize(
    "change",
    [
        {"workers": [{"url": "http://one"}]},
        {"workers": [{"url": "http://one"}, {"url": "http://one"}]},
        {"model_dir": "/tmp/a;touch-b"},
        {
            "workers": [
                {"url": "http://one"},
                {"url": "http://two", "ssh": "-oProxyCommand=bad"},
            ]
        },
        {"workers": [{"url": "http://user:secret@one"}, {"url": "http://two"}]},
        {"timeout": 0},
    ],
)
def test_invalid_cluster_configuration(tmp_path, change):
    path = tmp_path / "cluster.json"
    path.write_text(
        json.dumps(
            {
                "workers": [{"url": "http://one"}, {"url": "http://two"}],
                "model_dir": "/tmp/models",
                "shared_lock": "/tmp/model.lock",
                **change,
            }
        )
    )
    with pytest.raises(ValueError):
        ClusterConfig.read(path)


def test_cluster_lock_prevents_overlapping_clients(cluster):
    with cluster_lock(cluster.shared_lock):
        with pytest.raises(HTTPException) as error:
            with cluster_lock(cluster.shared_lock):
                pass
    assert error.value.status_code == 409
    with cluster_lock(cluster.shared_lock):
        pass


def _transport(monkeypatch, handler):
    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(**kwargs, transport=httpx.MockTransport(handler)),
    )


def _health(rank):
    return {
        "distributed_worker": {
            "rank": rank,
            "world_size": 2,
            "protocol_version": 2,
            "backend": "jaccl",
            "fingerprint": "matching-code-settings",
        }
    }


def test_coordinator_waits_for_staging_then_fans_out_identical_requests(
    cluster, monkeypatch
):
    coordinator = Coordinator(cluster)
    calls = []
    entered = set()

    def stage(*args):
        calls.append("staged")
        return str(cluster.model_dir / "snapshot")

    monkeypatch.setattr(coordinator.stager, "prepare", stage)
    monkeypatch.setenv(cluster.api_key_env, "worker-token")

    async def handler(request):
        rank = int(request.url.host[-1])
        assert request.headers["authorization"] == "Bearer worker-token"
        if request.method == "GET":
            return httpx.Response(200, json=_health(rank))
        assert calls[0] == "staged"
        entered.add(rank)
        for _ in range(100):
            if len(entered) == 2:
                break
            await asyncio.sleep(0)
        assert len(entered) == 2  # Sequential fan-out would deadlock a collective.
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(
            200,
            json={
                "id": f"rank-{rank}",
                "created": rank + 1,
                "model": body["model"],
                "output": [{"text": "answer"}],
                "usage": {"output_tokens": 1},
                "timings": {"elapsed": rank + 1},
            },
        )

    _transport(monkeypatch, handler)
    result, status = asyncio.run(
        coordinator.relay(
            "/v1/responses",
            {"model": "org/model", "distributed": True, "input": "hello"},
        )
    )
    assert status == 200 and result["model"] == "org/model"
    assert calls[1] == calls[2]
    assert calls[1]["seed"] is not None and calls[1]["store"] is False


def test_compaction_comparison_uses_decrypted_content(tmp_path):
    from cryptography.fernet import Fernet

    from mlx_vlm.server.compaction import CAPSULE_PREFIX

    key_file = tmp_path / "key"
    key_file.write_bytes(Fernet.generate_key())
    cipher = Fernet(key_file.read_bytes())

    def capsule(text):
        return {
            "type": "compaction",
            "encrypted_content": CAPSULE_PREFIX
            + cipher.encrypt(json.dumps({"items": [{"text": text}]}).encode()).decode(),
        }

    first, second = capsule("same"), capsule("same")
    assert first != second
    assert _semantic(first, key_file) == _semantic(second, key_file)
    assert _semantic(first, key_file) != _semantic(capsule("different"), key_file)
    assert _semantic({"usage": {"output_tokens": 1}}, None) != _semantic(
        {"usage": {"output_tokens": 2}}, None
    )


def test_staging_error_never_fans_out_or_poisons_group(cluster, monkeypatch):
    coordinator = Coordinator(cluster)
    monkeypatch.setenv(cluster.api_key_env, "key")
    stage = Mock(side_effect=httpx.ConnectError("Hub unavailable"))
    monkeypatch.setattr(coordinator.stager, "prepare", stage)

    def handler(request):
        assert request.method == "GET"
        return httpx.Response(200, json=_health(int(request.url.host[-1])))

    _transport(monkeypatch, handler)
    with pytest.raises(HTTPException, match="Model staging failed") as error:
        asyncio.run(coordinator.relay("/v1/responses", {"model": "org/model"}))
    assert error.value.status_code == 400
    assert not coordinator.failure.exists()
    with cluster_lock(cluster.shared_lock):
        pass


def test_cancelled_transfer_keeps_lock_until_thread_finishes(cluster, monkeypatch):
    coordinator = Coordinator(cluster)
    entered, release = threading.Event(), threading.Event()
    monkeypatch.setenv(cluster.api_key_env, "key")

    def stage(*args):
        entered.set()
        assert release.wait(5)
        return str(cluster.model_dir / "snapshot")

    def handler(request):
        assert request.method == "GET"
        return httpx.Response(200, json=_health(int(request.url.host[-1])))

    monkeypatch.setattr(coordinator.stager, "prepare", stage)
    _transport(monkeypatch, handler)

    async def cancel_request():
        task = asyncio.create_task(
            coordinator.relay("/v1/responses", {"model": "org/model"})
        )
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            with pytest.raises(HTTPException) as error:
                with cluster_lock(cluster.shared_lock):
                    pass
            assert error.value.status_code == 409
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_request())
    assert not coordinator.failure.exists()
    with cluster_lock(cluster.shared_lock):
        pass


def test_worker_fingerprint_includes_context_limit(cluster, monkeypatch):
    worker = DistributedWorker.__new__(DistributedWorker)
    worker.source_hash = "same-source"
    worker.instance_id = "test-worker-instance"
    worker.backend = "jaccl"
    worker.group = NS(rank=lambda: 0, size=lambda: 2)
    monkeypatch.delenv("MLX_VLM_COMPACTION_KEY_FILE", raising=False)
    monkeypatch.setattr(runtime.config, "max_kv_size", 4096)
    first = worker.info()["fingerprint"]
    monkeypatch.setattr(runtime.config, "max_kv_size", 8192)
    assert worker.info()["fingerprint"] != first


@pytest.mark.parametrize("failure", ["mismatch", "transport", "server"])
def test_failed_rank_poisons_group_and_prevents_next_request(
    cluster, monkeypatch, failure
):
    coordinator = Coordinator(cluster)
    cluster.model_dir.mkdir()
    monkeypatch.setenv(cluster.api_key_env, "key")
    stage = Mock(return_value=str(cluster.model_dir / "snapshot"))
    monkeypatch.setattr(coordinator.stager, "prepare", stage)

    async def handler(request):
        rank = int(request.url.host[-1])
        if request.method == "GET":
            return httpx.Response(200, json=_health(rank))
        if failure == "transport" and rank == 1:
            raise httpx.ReadTimeout("timeout")
        return httpx.Response(
            500 if failure == "server" else 200,
            json={"output": str(rank) if failure == "mismatch" else "a"},
        )

    _transport(monkeypatch, handler)
    with pytest.raises(HTTPException) as error:
        asyncio.run(coordinator.relay("/v1/responses", {"model": "org/model"}))
    assert error.value.status_code == 502
    assert coordinator.failure.exists()
    with pytest.raises(HTTPException) as error:
        asyncio.run(coordinator.relay("/v1/responses", {"model": "org/model"}))
    assert error.value.status_code == 503
    stage.assert_called_once()


def test_worker_rank_mismatch_fails_before_staging(cluster, monkeypatch):
    coordinator = Coordinator(cluster)
    stage = Mock()
    monkeypatch.setattr(coordinator.stager, "prepare", stage)
    monkeypatch.setenv(cluster.api_key_env, "key")
    _transport(monkeypatch, lambda request: httpx.Response(200, json=_health(0)))
    with pytest.raises(HTTPException) as error:
        asyncio.run(coordinator.relay("/v1/responses", {"model": "org/model"}))
    assert error.value.status_code == 503
    stage.assert_not_called()


def test_failed_rank_does_not_wait_for_peer_blocked_before_headers(
    cluster, monkeypatch
):
    async def check():
        coordinator = Coordinator(cluster)
        monkeypatch.setenv(cluster.api_key_env, "key")
        monkeypatch.setattr(
            coordinator.stager, "prepare", lambda *a: "/models/snapshot"
        )
        blocked = asyncio.Event()
        cancelled = set()

        async def handler(request):
            rank = int(request.url.host[-1])
            if request.method == "GET":
                return httpx.Response(200, json=_health(rank))
            if request.url.path == CANCEL_ROUTE:
                cancelled.add(rank)
                return httpx.Response(202, json={})
            if rank == 0:
                blocked.set()
                await asyncio.Event().wait()
            await blocked.wait()
            raise httpx.ConnectError("rank disconnected")

        _transport(monkeypatch, handler)
        with pytest.raises(HTTPException) as error:
            await asyncio.wait_for(
                coordinator.relay("/v1/responses", {"model": "a"}), 2
            )
        assert error.value.status_code == 502 and coordinator.failure.exists()
        assert coordinator.active == 0 and cancelled == {0, 1}
        assert len(asyncio.all_tasks()) == 1

    asyncio.run(check())


@pytest.mark.parametrize("body", ["not-json", "[]", '{"request_id":"invalid"}'])
def test_worker_cancel_rejects_invalid_bodies(monkeypatch, body):
    worker = NS(remember_cancellations=Mock())
    monkeypatch.setattr(runtime, "distributed_worker", worker)
    monkeypatch.setenv("MLX_VLM_SERVER_API_KEY", "worker-key")
    with TestClient(server.app) as client:
        result = client.post(
            CANCEL_ROUTE, content=body, headers={"Authorization": "Bearer worker-key"}
        )
    assert result.status_code == 422
    worker.remember_cancellations.assert_not_called()


@pytest.mark.parametrize(
    "options",
    [{"adapter_path": "adapter"}, {"previous_response_id": "resp"}],
)
def test_unsupported_distributed_options_fail_before_staging(cluster, options):
    coordinator = Coordinator(cluster)
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            coordinator.relay("/v1/responses", {"model": "org/model", **options})
        )
    assert error.value.status_code == 400
    assert not cluster.model_dir.exists()


def test_http_flag_and_authentication_preserve_local_routes(monkeypatch):
    relay = AsyncMock(return_value=({"answer": "distributed"}, 200))
    monkeypatch.setattr(
        runtime, "distributed_coordinator", NS(relay=relay, config=NS(launch=None))
    )
    monkeypatch.setattr(runtime, "distributed_worker", None)
    monkeypatch.setenv("MLX_VLM_SERVER_API_KEY", "coordinator-key")
    headers = {"Authorization": "Bearer coordinator-key"}
    body = {"model": "org/model", "input": "hello", "distributed": True}
    with TestClient(server.app) as client:
        assert client.post("/v1/responses", json=body).status_code == 401
        relay.assert_not_called()
        assert client.post("/v1/responses", headers=headers, json=body).json() == {
            "answer": "distributed"
        }
        relay.assert_awaited_once()
        # Invalid local input reaches the normal Pydantic route, without forwarding.
        for payload in ({}, {"distributed": False}):
            assert (
                client.post("/v1/responses", headers=headers, json=payload).status_code
                == 422
            )
        assert relay.await_count == 1
        assert (
            client.post(
                "/v1/responses", headers=headers, json={**body, "distributed": "true"}
            ).status_code
            == 422
        )


def test_unconfigured_distributed_flag_does_not_fall_back_to_local(monkeypatch):
    monkeypatch.setattr(runtime, "distributed_coordinator", None)
    monkeypatch.setattr(runtime, "distributed_worker", None)
    monkeypatch.delenv("MLX_VLM_SERVER_API_KEY", raising=False)
    with TestClient(server.app) as client:
        response = client.post(
            "/v1/responses",
            json={"model": "org/model", "input": "a", "distributed": True},
        )
    assert response.status_code == 503


def test_original_loader_remains_default(monkeypatch):
    model, processor = NS(config=object()), object()
    native = Mock(return_value=(model, processor))
    monkeypatch.setattr(generation, "load", native)
    monkeypatch.setattr(runtime, "distributed_worker", None)
    assert generation.load_model_resources("local/model", None) == (
        model,
        processor,
        model.config,
    )
    assert native.call_args.args == ("local/model", None)


def test_distributed_loader_is_selected_only_inside_worker(monkeypatch):
    model, processor = NS(config=object()), object()
    worker = NS(load=Mock(return_value=(model, processor)))
    monkeypatch.setattr(runtime, "distributed_worker", worker)
    native = Mock(side_effect=AssertionError("local loader selected"))
    monkeypatch.setattr(generation, "load", native)
    assert generation.load_model_resources("/models/snapshot", None) == (
        model,
        processor,
        model.config,
    )
    worker.load.assert_called_once()


def test_worker_rejects_unstaged_paths_before_loading(cluster, monkeypatch):
    worker = DistributedWorker.__new__(DistributedWorker)
    worker.model_dir = cluster.model_dir
    native = Mock()
    monkeypatch.setattr(utils, "load", native)
    with pytest.raises(ValueError, match="coordinator-staged"):
        worker.load("org/model", None)
    native.assert_not_called()


def test_worker_uses_native_lazy_loader_before_sharding(cluster, monkeypatch):
    import mlx.core as mx

    path = cluster.model_dir / "snapshot"
    path.mkdir(parents=True)
    (path / MANIFEST).write_text("{}")
    calls = []
    model = NS(
        language_model=NS(layers=[]),
        shard=lambda group: calls.append("shard"),
        parameters=lambda: "all-parameters-including-vision",
        eval=lambda: calls.append("eval-mode"),
    )
    processor = object()
    native = Mock(
        side_effect=lambda *a, **k: (calls.append("load") or (model, processor))
    )
    monkeypatch.setattr(utils, "load", native)
    monkeypatch.setattr(mx, "eval", lambda *a: calls.append(a[0]))
    monkeypatch.setattr(mx.distributed, "all_sum", lambda *a, **k: "barrier")
    worker = DistributedWorker.__new__(DistributedWorker)
    worker.model_dir = cluster.model_dir
    worker.group = object()
    assert worker.load(str(path), None) == (model, processor)
    native.assert_called_once_with(
        str(path), lazy=True, trust_remote_code=False, local_files_only=True
    )
    assert calls == [
        "load",
        "shard",
        "all-parameters-including-vision",
        "eval-mode",
        "barrier",
    ]


def test_worker_startup_disables_independent_caches_and_hub_access(
    cluster, monkeypatch
):
    from huggingface_hub import constants as hub_constants

    from mlx_vlm.server import distributed

    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("MLX_VLM_PRELOAD_") and key != "MLX_VLM_DRAFT_MODEL"
    }
    monkeypatch.setattr(os, "environ", env)
    monkeypatch.setenv("MLX_VLM_SERVER_API_KEY", "key")
    monkeypatch.setenv("APC_ENABLED", "1")
    monkeypatch.setenv("MLX_VLM_MAX_NUM_SEQS", "8")
    monkeypatch.setattr(runtime, "config", runtime.config)
    monkeypatch.setattr(runtime, "distributed_worker", None)
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", False)
    worker = object()
    factory = Mock(return_value=worker)
    monkeypatch.setattr(distributed, "DistributedWorker", factory)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlx_vlm.server",
            "--distributed-worker",
            "--distributed-model-dir",
            str(cluster.model_dir),
        ],
    )
    run = Mock()
    monkeypatch.setattr(cli.uvicorn, "run", run)
    cli.main()
    assert runtime.distributed_worker is worker
    assert runtime.config.apc_enabled is False
    assert runtime.config.vision_cache_size == 0
    assert os.environ["MLX_VLM_MAX_NUM_SEQS"] == "8"
    assert hub_constants.HF_HUB_OFFLINE is True
    run.assert_called_once()


@pytest.mark.parametrize(
    "args",
    [
        ["--distributed-config", "cluster.json"],
        ["--distributed-worker", "--api-key", "key"],
        [
            "--distributed-worker",
            "--api-key",
            "key",
            "--distributed-model-dir",
            "/tmp/models",
            "--model",
            "org/model",
        ],
    ],
)
def test_cli_rejects_incomplete_cluster_setup(monkeypatch, args):
    monkeypatch.delenv("MLX_VLM_SERVER_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["mlx_vlm.server", *args])
    run = Mock()
    monkeypatch.setattr(cli.uvicorn, "run", run)
    with pytest.raises(SystemExit):
        cli.main()
    run.assert_not_called()


class EventStream(httpx.AsyncByteStream):
    def __init__(self, source):
        self.source = source

    async def __aiter__(self):
        async for event in self.source():
            yield event.encode()


def _sse(rank, text, model="/models/snapshot"):
    return (
        "data: "
        + json.dumps(
            {
                "id": f"rank-{rank}",
                "created": rank + 1,
                "model": model,
                "choices": [{"delta": {"content": text}}],
            }
        )
        + "\n\n"
    )


@pytest.mark.parametrize("mismatch", [False, True])
def test_stream_forwards_tokens_before_completion_and_checks_every_rank(
    cluster, monkeypatch, mismatch
):
    async def check():
        release = asyncio.Event()
        coordinator = Coordinator(cluster)
        monkeypatch.setenv(cluster.api_key_env, "key")
        monkeypatch.setattr(
            coordinator.stager, "prepare", lambda *a: "/models/snapshot"
        )

        async def handler(request):
            rank = int(request.url.host[-1])
            if request.method == "GET":
                return httpx.Response(200, json=_health(rank))

            async def events():
                yield _sse(rank, "different" if mismatch and rank else "hello")
                await release.wait()
                yield "data: [DONE]\n\n"

            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=EventStream(events),
            )

        _transport(monkeypatch, handler)
        response = await coordinator.relay(
            "/v1/chat/completions", {"model": "org/model", "stream": True}
        )
        first = await asyncio.wait_for(anext(response.body_iterator), 2)
        assert "hello" in first and "org/model" in first and "[DONE]" not in first
        assert coordinator.active == 1 and not release.is_set()
        release.set()
        rest = "".join([event async for event in response.body_iterator])
        assert ("[DONE]" in rest) is not mismatch
        assert ("distributed_error" in rest) is mismatch
        assert coordinator.failure.exists() is mismatch
        assert coordinator.active == 0

    asyncio.run(check())


def test_stream_disconnect_cancels_all_ranks_and_drains_before_unlock(
    cluster, monkeypatch
):
    async def check():
        coordinator = Coordinator(cluster)
        monkeypatch.setenv(cluster.api_key_env, "key")
        monkeypatch.setattr(
            coordinator.stager, "prepare", lambda *a: "/models/snapshot"
        )
        release = asyncio.Event()
        cancelled = set()

        async def handler(request):
            rank = int(request.url.host[-1])
            if request.method == "GET":
                return httpx.Response(200, json=_health(rank))
            if request.url.path == CANCEL_ROUTE:
                assert coordinator.active == 1
                cancelled.add(rank)
                if len(cancelled) == 2:
                    release.set()
                return httpx.Response(202, json={})

            async def events():
                yield _sse(rank, "hello")
                await release.wait()

            return httpx.Response(200, stream=EventStream(events))

        _transport(monkeypatch, handler)
        response = await coordinator.relay(
            "/v1/chat/completions", {"model": "org/model", "stream": True}
        )
        await anext(response.body_iterator)
        await asyncio.wait_for(response.body_iterator.aclose(), 2)
        assert cancelled == {0, 1} and coordinator.active == 0
        assert not coordinator.failure.exists()
        assert len(asyncio.all_tasks()) == 1

    asyncio.run(check())


def test_same_model_requests_overlap_and_switch_waits_for_streams(cluster, monkeypatch):
    async def check():
        coordinator = Coordinator(cluster)
        monkeypatch.setenv(cluster.api_key_env, "key")
        stages, seeds = [], []

        def stage(model, revision):
            stages.append(model)
            return "/models/" + model

        monkeypatch.setattr(coordinator.stager, "prepare", stage)
        release = asyncio.Event()

        async def handler(request):
            rank = int(request.url.host[-1])
            if request.method == "GET":
                return httpx.Response(200, json=_health(rank))
            body = json.loads(request.content)
            seeds.append(body["seed"])

            async def events():
                yield _sse(rank, "hello", body["model"])
                await release.wait()
                yield "data: [DONE]\n\n"

            return httpx.Response(200, stream=EventStream(events))

        _transport(monkeypatch, handler)
        first, second = await asyncio.wait_for(
            asyncio.gather(
                *[
                    coordinator.relay(
                        "/v1/chat/completions", {"model": "a", "stream": True}
                    )
                    for _ in range(2)
                ]
            ),
            2,
        )
        assert coordinator.active == 2 and stages == ["a"]
        assert len(set(seeds)) == 1
        switching = asyncio.create_task(
            coordinator.relay("/v1/chat/completions", {"model": "b", "stream": True})
        )
        await asyncio.sleep(0.01)
        assert not switching.done() and stages == ["a"]
        release.set()
        for response in (first, second):
            assert "[DONE]" in "".join(
                [event async for event in response.body_iterator]
            )
        third = await asyncio.wait_for(switching, 2)
        assert stages == ["a", "b"]
        assert "[DONE]" in "".join([event async for event in third.body_iterator])
        assert coordinator.active == 0

    asyncio.run(check())


@pytest.mark.parametrize("spec", ["2.3", "2.4"])
@pytest.mark.parametrize("disconnect_on", ["http.response.start", "http.response.body"])
def test_stream_asgi_disconnect_releases_unstarted_or_suspended_iterator(
    cluster, monkeypatch, spec, disconnect_on
):
    from starlette.requests import ClientDisconnect

    async def check():
        coordinator = Coordinator(cluster)
        monkeypatch.setenv(cluster.api_key_env, "key")
        monkeypatch.setattr(
            coordinator.stager, "prepare", lambda *a: "/models/snapshot"
        )
        release, disconnect = asyncio.Event(), asyncio.Event()
        cancelled = set()

        async def handler(request):
            rank = int(request.url.host[-1])
            if request.method == "GET":
                return httpx.Response(200, json=_health(rank))
            if request.url.path == CANCEL_ROUTE:
                assert coordinator.active == 1
                cancelled.add(rank)
                if len(cancelled) == 2:
                    release.set()
                return httpx.Response(202, json={})

            async def events():
                yield _sse(rank, "hello")
                await release.wait()

            return httpx.Response(200, stream=EventStream(events))

        _transport(monkeypatch, handler)
        response = await coordinator.relay(
            "/v1/chat/completions", {"model": "org/model", "stream": True}
        )

        async def send(message):
            if message["type"] == disconnect_on:
                if spec == "2.4":
                    raise OSError("Client disconnected")
                disconnect.set()
                await asyncio.Event().wait()

        async def receive():
            await disconnect.wait()
            return {"type": "http.disconnect"}

        async def serve():
            try:
                await response(
                    {"type": "http", "asgi": {"spec_version": spec}}, receive, send
                )
            except ClientDisconnect:
                assert spec == "2.4"

        await asyncio.wait_for(serve(), 2)
        assert coordinator.active == 0 and cancelled == {0, 1}
        assert not coordinator.failure.exists()
        assert len(asyncio.all_tasks()) == 1

    asyncio.run(check())


@pytest.mark.parametrize("phase", ["headers", "body"])
def test_nonstream_cancellation_drains_workers_without_poisoning_group(
    cluster, monkeypatch, phase
):
    async def check():
        coordinator = Coordinator(cluster)
        monkeypatch.setenv(cluster.api_key_env, "key")
        monkeypatch.setattr(
            coordinator.stager, "prepare", lambda *a: "/models/snapshot"
        )
        started, release = asyncio.Event(), asyncio.Event()
        cancelled, reading = set(), set()

        async def handler(request):
            rank = int(request.url.host[-1])
            if request.method == "GET":
                return httpx.Response(200, json=_health(rank))
            if request.url.path == CANCEL_ROUTE:
                assert coordinator.active == 1
                cancelled.add(rank)
                if len(cancelled) == 2:
                    release.set()
                return httpx.Response(202, json={})

            async def wait_for_cancel():
                reading.add(rank)
                if len(reading) == 2:
                    started.set()
                await release.wait()

            if phase == "headers":
                await wait_for_cancel()
                return httpx.Response(500, json={"error": "Request cancelled"})

            async def body():
                await wait_for_cancel()
                yield '{"error": "Request cancelled"}'

            return httpx.Response(500, stream=EventStream(body))

        _transport(monkeypatch, handler)
        task = asyncio.create_task(
            coordinator.relay("/v1/chat/completions", {"model": "a"})
        )
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        assert cancelled == {0, 1} and coordinator.active == 0
        assert not coordinator.failure.exists()
        assert len(asyncio.all_tasks()) == 1

    asyncio.run(check())


def _state(ready=(), active=(), cancelled=(), request_cancels=()):
    return {
        "phase": "schedule",
        "ready": list(ready),
        "active": list(active),
        "cancelled": list(cancelled),
        "cancelled_requests": list(request_cancels),
        "error": None,
        "stop": False,
    }


def test_scheduler_late_arrivals_share_rank_zero_order_and_cancellation():
    a, b, c = ("a:1", "hash-a"), ("b:1", "hash-b"), ("c:1", "hash-c")
    chosen, cancelled, roots, stop = schedule(
        [
            _state([a, b, c], [1], cancelled=[1]),
            _state([c, a], [1], request_cancels=["c"]),
        ],
        8,
    )
    assert chosen == ["a:1"] and cancelled == {1} and roots == {"c"} and not stop
    assert schedule([_state([a, b, c]), _state([c, b, a])], 2)[0] == ["a:1", "b:1"]


@pytest.mark.parametrize(
    "other",
    [
        _state([("a:1", "different-input")]),
        _state([("a:1", "hash")], [99]),
        {**_state(), "phase": "tokens"},
    ],
)
def test_scheduler_rejects_divergence_before_model_step(other):
    with pytest.raises(RuntimeError):
        schedule([_state([("a:1", "hash")]), other], 4)


def test_generation_scope_survives_preprocessing_threads(monkeypatch):
    from mlx_vlm.tests.test_server import _capture_requests, _ready_generator

    gen = _ready_generator()
    queued = _capture_requests(gen)
    monkeypatch.setattr(runtime, "distributed_worker", object())

    async def check():
        token = request_scope.set(RequestScope("a" * 32))
        try:
            for _ in range(2):
                _, iterator = await asyncio.to_thread(gen.generate, "hello")
                iterator.close()
        finally:
            request_scope.reset(token)

    asyncio.run(check())
    assert [item.request_id for item in queued] == ["a" * 32 + ":1", "a" * 32 + ":2"]


@pytest.mark.parametrize(
    "filters", [{}, {"top_n_sigma": 2}, {"p_less": True}, {"typical_p": 0.8}]
)
def test_distributed_sampling_ignores_rank_global_rng(monkeypatch, filters):
    import mlx.core as mx

    monkeypatch.setattr(runtime, "distributed_worker", object())
    sampler = generation.ResponseGenerator.__new__(
        generation.ResponseGenerator
    )._make_sampler(generation.GenerationArguments(temperature=0.8, seed=37, **filters))
    logits = mx.array([[0.2, 0.3, 0.4, 0.5], [0.1, 0.4, 0.3, 0.2]])
    samples = []
    for rank in range(2):
        mx.random.seed(100 + rank)
        samples.append(
            [
                sampler.sample_target(
                    logits, row_ids=[3, 9], positions=[step, step + 1]
                ).tolist()
                for step in range(16)
            ]
        )
    assert samples[0] == samples[1]


def _native_batch_worker(limit):
    """Real ring collectives and native batching; deterministic synthetic model."""
    from contextlib import nullcontext
    from queue import Queue

    import mlx.core as mx
    import mlx.nn as nn

    from mlx_vlm.models.base import LanguageModelOutput
    from mlx_vlm.models.cache import KVCache
    from mlx_vlm.server.distributed_schedule import gather_json
    from mlx_vlm.tests.test_server import _generator
    from mlx_vlm.tokenizer_utils import NaiveStreamingDetokenizer
    from mlx_vlm.utils import StoppingCriteria

    mx.set_default_device(mx.cpu)
    # Metal residency tuning is unrelated to this CPU-only ring test.
    generation.wired_limit = lambda *a, **kw: nullcontext()
    worker = DistributedWorker("ring", "/tmp/mlx-vlm-synthetic-no-weights")
    runtime.distributed_worker = worker
    os.environ["MLX_VLM_MAX_NUM_SEQS"] = str(limit)
    group = worker.group
    mx.random.seed(100 + group.rank())  # Deliberately different process RNG state.
    sizes = []

    class TinyCollectiveLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = []

        def make_cache(self):
            return [KVCache()]

        def __call__(self, tokens, cache=None, **kwargs):
            batch, length = tokens.shape
            sizes.append(batch)
            values = mx.ones((batch, 1, length, 1))
            cache[0].update_and_fetch(values, values)
            logits = mx.broadcast_to(
                mx.arange(8, dtype=mx.float32) / 3, (batch, length, 8)
            )
            return LanguageModelOutput(
                logits=mx.distributed.all_sum(logits, group=group)
            )

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = TinyCollectiveLM()

    tokenizer = NS(
        decode=lambda tokens, **kw: " ".join(map(str, tokens)),
        all_special_ids=[],
        eos_token_id=None,
    )
    tokenizer.stopping_criteria = StoppingCriteria([], tokenizer)
    # Native prefill_step_size is an aggregate token budget: give every row
    # two tokens per step so the full requested batch reaches decoding.
    gen = _generator(prefill_step_size=2 * limit)

    def initialize():
        gen.model = TinyModel()
        gen.tokenizer = tokenizer
        gen.processor = NS(
            tokenizer=tokenizer, detokenizer=NaiveStreamingDetokenizer(tokenizer)
        )
        gen.config = NS(model_type="synthetic", eos_token_id=[])

    gen._initialize_model = initialize
    gen._gpu_embed = lambda raw, images=None, **kw: (
        raw["input_ids"],
        {"inputs_embeds": raw["input_ids"][..., None].astype(mx.float32)},
    )
    active_batches = []
    step_group = gen._step_group

    def observe(engine, active):
        active_batches.append(set(active))
        return step_group(engine, active)

    gen._step_group = observe

    queues = {}

    def enqueue(name, maximum):
        queue = Queue()
        queues[name] = queue
        gen.requests.put(
            generation.QueuedGenerationRequest(
                rqueue=queue,
                request_id=name + ":1",
                raw_inputs={"input_ids": mx.array([[1, 3, 4]])},
                prompt_tokens=3,
                args=generation.GenerationArguments(
                    max_tokens=maximum,
                    temperature=0.8,
                    seed=7,
                    top_n_sigma=2 if limit == 8 else 0,
                ),
            )
        )

    order = list(range(limit))
    if group.rank():
        order.reverse()
    for i in order:
        enqueue(str(i), 100 if i < 2 else 4)
    gen._thread = threading.Thread(target=gen._run, daemon=True)
    gen._thread.start()
    contexts = {name: queue.get(timeout=20) for name, queue in queues.items()}
    assert all(
        isinstance(context, generation.GenerationContext)
        for context in contexts.values()
    ), contexts
    first = queues["0"].get(timeout=20)
    assert isinstance(first, generation.StreamingToken), first
    if group.rank() == 0:
        gen._cancel(contexts["1"].uid)
    enqueue("late", 5)
    late = queues["late"].get(timeout=20)
    assert isinstance(late, generation.GenerationContext), late
    results = {"0": [first.token]}
    for name, queue in queues.items():
        tokens = results.setdefault(name, [])
        while (item := queue.get(timeout=20)) is not None:
            assert isinstance(item, generation.StreamingToken), item
            tokens.append(item.token)
    gen.stop_and_join(timeout=10)
    assert not gen._thread.is_alive() and gen._load_error is None
    assert max(sizes) == limit, sizes
    assert len(results["0"]) == 100 and len(results["late"]) == 5
    assert len(results["1"]) < 100
    assert any({contexts["0"].uid, late.uid} <= batch for batch in active_batches)
    all_results = gather_json(
        group,
        {"tokens": results, "uids": {name: ctx.uid for name, ctx in contexts.items()}},
    )
    assert all(value == all_results[0] for value in all_results)
    print(
        json.dumps(
            {
                "rank": group.rank(),
                "batch": max(sizes),
                "late_tokens": len(results["late"]),
                "passed": True,
            }
        ),
        flush=True,
    )


@pytest.mark.parametrize("limit", [4, 8])
def test_distributed_native_continuous_batching(limit):
    import secrets
    import signal
    import socket
    from contextlib import ExitStack

    import mlx.core as mx

    if not mx.distributed.is_available("ring"):
        pytest.skip("MLX ring backend is unavailable")
    for _ in range(100):
        with ExitStack() as stack:
            sockets = [stack.enter_context(socket.socket()) for _ in range(2)]
            port = 10000 + 2 * secrets.randbelow(10000)
            try:
                for rank, sock in enumerate(sockets):
                    sock.bind(("127.0.0.1", port + rank))
            except OSError:
                continue
            break
    else:
        pytest.fail("No free local ring ports")
    repo = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        "-c",
        "from mlx._distributed_utils.launch import main; main()",
        "--backend",
        "ring",
        "-n",
        "2",
        "--starting-port",
        str(port),
        "--env",
        f"PYTHONPATH={repo}",
        "--",
        sys.executable,
        "-c",
        "from mlx_vlm.tests.test_server_distributed import _native_batch_worker; "
        f"_native_batch_worker({limit})",
    ]
    process = subprocess.Popen(
        command,
        cwd=repo,
        env={**os.environ, "PYTHONPATH": str(repo)},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=90)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        pytest.fail("Distributed batching timed out:\n" + output)
    assert process.returncode == 0, output
    records = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
    assert len(records) == 2 and all(
        r["passed"] and r["batch"] == limit for r in records
    ), output


def test_progress_cadence_is_not_semantic_rank_divergence(cluster, monkeypatch):
    async def scenario():
        coordinator = Coordinator(cluster)
        monkeypatch.setenv(cluster.api_key_env, "key")
        monkeypatch.setattr(
            coordinator.stager, "prepare", lambda *_: "/models/snapshot"
        )

        async def handler(request):
            rank = int(request.url.host[-1])
            if request.method == "GET":
                return httpx.Response(200, json=_health(rank))

            async def events():
                for processed in ([0, 2, 4] if rank == 0 else [0, 1, 2, 3, 4]):
                    yield "event: response.prompt_progress\ndata: " + json.dumps(
                        {
                            "type": "response.prompt_progress",
                            "response_id": str(rank),
                            "processed_tokens": processed,
                            "total_tokens": 4,
                            "status": "completed" if processed == 4 else "in_progress",
                        }
                    ) + "\n\n"
                yield _sse(rank, "answer")
                yield "data: [DONE]\n\n"

            return httpx.Response(200, stream=EventStream(events))

        _transport(monkeypatch, handler)
        result = await coordinator.relay(
            "/v1/responses", {"model": "org/model", "stream": True}
        )
        output = "".join([value async for value in result.body_iterator])
        assert output.count('"type": "response.prompt_progress"') == 3
        assert output.index("response.prompt_progress") < output.index("answer")
        assert "[DONE]" in output and not coordinator.failure.exists()

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_resumable_disconnect_pauses_all_ranks_without_cancelling_decoder(
    cluster, monkeypatch, phase
):
    from mlx_vlm.server.distributed_sessions import SESSION_ROUTE

    async def scenario():
        coordinator = Coordinator(cluster)
        monkeypatch.setenv(cluster.api_key_env, "key")
        monkeypatch.setattr(
            coordinator.stager, "prepare", lambda *_: "/models/snapshot"
        )
        paused = set()

        async def handler(request):
            rank = int(request.url.host[-1])
            if request.method == "GET":
                return httpx.Response(200, json=_health(rank))
            assert request.url.path != CANCEL_ROUTE
            if request.url.path == SESSION_ROUTE:
                assert json.loads(request.content)["action"] == "pause"
                assert coordinator.active == 1
                paused.add(rank)
                return httpx.Response(200, json={"state": "paused"})

            async def events():
                if phase == "prefill":
                    yield 'event: response.prompt_progress\ndata: {"type":"response.prompt_progress","status":"started","processed_tokens":0,"total_tokens":100}\n\n'
                else:
                    yield _sse(rank, "first")
                await asyncio.Event().wait()

            return httpx.Response(200, stream=EventStream(events))

        _transport(monkeypatch, handler)
        response = await coordinator.relay(
            "/v1/responses",
            {"model": "org/model", "stream": True, "resumable": True},
            "tenant",
        )
        await asyncio.wait_for(anext(response.body_iterator), 2)
        await asyncio.wait_for(response.body_iterator.aclose(), 2)
        assert paused == {0, 1} and not coordinator.failure.exists()
        assert coordinator.active == 0 and coordinator.continuations
        assert all(
            not record["active"] for record in coordinator.continuations.values()
        )
        assert len(asyncio.all_tasks()) == 1

    asyncio.run(scenario())


def test_coordinated_session_pause_resume_and_expiry_use_existing_cache_engine(
    monkeypatch,
):
    from copy import deepcopy

    from mlx_vlm.server.distributed_schedule import DistributedBatchScheduler
    from mlx_vlm.server.sessions import InferenceSession

    generator = generation.ResponseGenerator.__new__(generation.ResponseGenerator)
    session = InferenceSession()
    engine = NS(
        suspend=Mock(return_value=NS(prompt_cache=[])),
        resume=Mock(),
        remove=Mock(),
        apc_manager=None,
    )
    active = {1: {"rqueue": session, "request_id": "request:1"}}
    groups, suspended = {0: (engine, active)}, {}
    other_paused = True
    other_error = None

    def exchange(value):
        other = deepcopy(value)
        if value["phase"] == "sessions":
            other["rows"][0][1] = other_paused
            other["rows"][0][3] = other_error
        return [value, other]

    scheduler = DistributedBatchScheduler(NS(), exchange=exchange)
    scheduler.continuations(generator, groups, suspended, 4)
    assert session in suspended and not active
    assert session.distributed_ack == "paused"
    other_paused = False
    scheduler.continuations(generator, groups, suspended, 4)
    assert 1 in active and not suspended
    assert session.distributed_ack == "running"
    engine.resume.assert_called_once()
    other_error = "Checkpoint expired on peer"
    scheduler.continuations(generator, groups, suspended, 4)
    assert not active and session.distributed_ack == "disposed"
    with pytest.raises(ValueError, match="expired"):
        session.claim()


def test_resume_uses_synchronized_decoder_history_when_http_consumer_lags():
    from mlx_vlm.server.continuation import Checkpoints, Generation
    from mlx_vlm.server.sessions import InferenceSession

    session = InferenceSession()
    for text in ("one", "two", "three"):
        session.put(
            generation.StreamingToken(
                text=text, token=1, logprobs=0, finish_reason=None
            )
        )
    store = Checkpoints()
    record = Generation(
        body={},
        tenant="t",
        path="/v1/responses",
        prompt="p",
        session=session,
        parts=["one"],
    )
    store.add(record)
    resumed = store.resume(
        {"id": record.id, "cursor": 2, "content": 6, "reasoning": 0},
        "t",
        "/v1/responses",
        record_id="a" * 32,
    )
    assert resumed.id == "a" * 32 and resumed.parts == ["one", "two"]
    assert resumed.session is session
    assert next(session.reader(2, resumed.lease)).text == "three"


def test_expiry_of_old_handle_does_not_dispose_active_resumed_session(cluster):
    async def scenario():
        coordinator = Coordinator(cluster)
        coordinator.continuations = {
            "old": {"session_id": "session", "active": False, "touched": 0, "ttl": 1},
            "new": {"session_id": "session", "active": True, "touched": 1, "ttl": 1},
        }
        coordinator.session_action = AsyncMock()
        await coordinator.expire_continuations(None)
        coordinator.session_action.assert_not_awaited()
        coordinator.continuations["new"]["active"] = False
        await coordinator.expire_continuations(None)
        coordinator.session_action.assert_awaited_once_with(None, "new", "dispose")

    asyncio.run(scenario())


def test_management_and_inference_staging_cannot_write_same_peer_concurrently(cluster):
    from concurrent.futures import ThreadPoolExecutor

    first, second = ModelStager(cluster), ModelStager(cluster)
    entered, release, peer_entered = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )

    def prepare_first(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return "first"

    def prepare_second(*args, **kwargs):
        peer_entered.set()
        return "second"

    first._prepare, second._prepare = prepare_first, prepare_second
    with ThreadPoolExecutor(2) as pool:
        a = pool.submit(first.prepare, "same-model")
        assert entered.wait(1)
        b = pool.submit(second.prepare, "same-model")
        assert not peer_entered.wait(0.05)
        release.set()
        assert a.result() == "first" and b.result() == "second"
