# Copyright © 2026 Neywa Labs Inc.
"""Reusable, isolated two-Mac transfer sessions with SSH-authenticated control."""

import atexit
import concurrent.futures
import errno
import json
import os
import queue
import secrets
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path

from . import checkpoint_store as store
from ._worker_supervisor import worker_environment
from .rsync_progress import run_rsync


class TransferUnavailable(RuntimeError):
    """A connection check failed before any payload was sent."""


def bootstrap():
    parent = Path(__file__).parent
    return (
        "import sys,types; "
        "m=types.ModuleType('mlx_checkpoint_store'); "
        "sys.modules[m.__name__]=m; "
        f"exec({(parent / 'checkpoint_store.py').read_text()!r},m.__dict__); "
        f"exec({(parent / '_checkpoint_worker.py').read_text()!r},{{'__name__':'__main__'}})"
    )


def ssh_command(config):
    return [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ConnectTimeout=10",
        *config.ssh_options,
    ]


class Helper:
    def __init__(self, command, *, timeout, physical_id=None):
        self.timeout = timeout
        self.events = queue.Queue()
        self.stderr = deque(maxlen=30)
        self.write_lock = threading.Lock()
        self.closed = threading.Event()
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
            env=worker_environment(os.environ),
        )
        self.output = threading.Thread(target=self._read, daemon=True)
        self.errors = threading.Thread(target=self._errors, daemon=True)
        self.output.start()
        self.errors.start()
        self.heartbeat = threading.Thread(target=self._renew, daemon=True)
        self.heartbeat.start()
        try:
            hello = self.wait(None, event="hello", timeout=20)
            if hello.get("protocol") != 1:
                raise RuntimeError("Incompatible checkpoint helper protocol")
            if physical_id is not None and hello["physical_id"] != physical_id:
                raise RuntimeError(
                    "SSH host does not match the authenticated worker identity"
                )
            self.pid = hello["pid"]
        except BaseException:
            self.close()
            raise

    def _read(self):
        try:
            for line in self.process.stdout:
                if line.startswith("MLX_TRANSFER "):
                    self.events.put(json.loads(line[len("MLX_TRANSFER ") :]))
                else:
                    self.stderr.append(line.rstrip())
        finally:
            self.events.put({"event": "exit"})

    def _errors(self):
        for line in self.process.stderr:
            self.stderr.append(line.rstrip())

    def _renew(self):
        while not self.closed.wait(5):
            try:
                self.send({"op": "ping"})
            except (OSError, ValueError):
                return

    def send(self, message):
        with self.write_lock:
            self.process.stdin.write(json.dumps(message) + "\n")
            self.process.stdin.flush()

    def submit(self, op, **data):
        request = uuid.uuid4().hex
        self.send({"op": op, "id": request, **data})
        return request

    def wait(
        self,
        request,
        *,
        event="result",
        timeout=None,
        progress=None,
        stop=None,
        connection_check=False,
    ):
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        while True:
            store.cancelled(stop)
            if time.monotonic() >= deadline:
                raise TimeoutError("Checkpoint helper timed out")
            try:
                value = self.events.get(
                    timeout=min(0.1, max(0.001, deadline - time.monotonic()))
                )
            except queue.Empty:
                continue
            if value["event"] == "exit":
                raise RuntimeError(
                    "Checkpoint helper exited: " + "\n".join(self.stderr)[-1500:]
                )
            if value.get("id") != request:
                raise RuntimeError("Unexpected checkpoint helper command sequence")
            if value["event"] == "error":
                if connection_check and value.get("code") == "rdma_unavailable":
                    raise TransferUnavailable(value["message"])
                raise RuntimeError(value["message"])
            if value["event"] == event:
                return value
            if value["event"] == "progress" and progress:
                progress({k: v for k, v in value.items() if k not in {"id", "event"}})

    def call(self, op, *, progress=None, stop=None, **data):
        return self.wait(self.submit(op, **data), progress=progress, stop=stop)

    def close(self):
        if self.closed.is_set():
            return
        self.closed.set()
        try:
            self.send({"op": "shutdown"})
        except (OSError, ValueError):
            pass
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.process.wait(timeout=5)
        try:
            self.process.stdin.close()
        except OSError:
            pass  # A failed remote rank may already have closed its input.
        self.output.join(timeout=2)
        self.errors.join(timeout=2)
        self.process.stdout.close()
        self.process.stderr.close()


class Pair:
    def __init__(self, config, worker):
        self.config, self.worker = config, worker
        self.remote = None
        self.local = None
        self.validated = False
        self.last_used = time.monotonic()
        command = [worker.python or config.python, "-u", "-c", bootstrap()]
        if worker.ssh:
            command = [*ssh_command(config), worker.ssh, shlex.join(command)]
        self.remote = Helper(
            command, timeout=config.timeout, physical_id=worker.physical_id
        )

    def connect(self, stop=None):
        if self.validated:
            return
        link = self.worker.rdma
        if not link or not self.worker.ssh:
            raise TransferUnavailable("No direct RDMA pair configured for this worker")
        self.local = Helper(
            [sys.executable, "-u", "-c", bootstrap()], timeout=self.config.timeout
        )
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            try:
                listener.bind((link["local_address"], 0))
            except OSError as error:
                if error.errno in {
                    errno.EADDRNOTAVAIL,
                    errno.ENETDOWN,
                    errno.ENETUNREACH,
                }:
                    raise TransferUnavailable(
                        f"JACCL coordinator address unavailable: {error}"
                    ) from error
                raise
            port = listener.getsockname()[1]
        common = dict(
            devices=[[None, link["local_device"]], [link["remote_device"], None]],
            coordinator=f"{link['local_address']}:{port}",
            challenge=secrets.token_hex(32),
        )
        first = self.local.submit("init", rank=0, **common)
        self.local.wait(
            first, event="initializing", timeout=20, stop=stop, connection_check=True
        )
        second = self.remote.submit("init", rank=1, **common)
        self.remote.wait(
            second, event="initializing", timeout=20, stop=stop, connection_check=True
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            pending = {
                pool.submit(
                    helper.wait, request, timeout=30, stop=stop, connection_check=True
                ): rank
                for helper, request, rank in (
                    (self.local, first, "sender"),
                    (self.remote, second, "receiver"),
                )
            }
            done, _ = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_EXCEPTION
            )
            try:
                # Prefer a real defect over a secondary connection failure if
                # both ranks have already reported errors.
                errors = [(pending[f], f.exception()) for f in done if f.exception()]
                for rank, error in errors:
                    if not isinstance(error, (TransferUnavailable, TimeoutError)):
                        raise error
                if errors:
                    rank, error = errors[0]
                    raise TransferUnavailable(
                        f"JACCL connection validation failed ({rank}): {error}"
                    ) from error
            except BaseException:
                self.close()
                raise
        self.validated = True

    def close(self):
        for helper in (self.local, self.remote):
            if helper:
                helper.close()


class TransferPool:
    """Serialized transfers, reusable per-peer helpers, bounded idle lifetime."""

    def __init__(self):
        self.pairs = {}
        self.lock = threading.Lock()
        self.closed = threading.Event()
        self.reaper = threading.Thread(target=self._reap, daemon=True)
        self.reaper.start()
        atexit.register(self.close)

    def _reap(self):
        while not self.closed.wait(10):
            if not self.lock.acquire(blocking=False):
                continue
            try:
                for key, pair in list(self.pairs.items()):
                    if time.monotonic() - pair.last_used > 300:
                        self.pairs.pop(key).close()
            finally:
                self.lock.release()

    def close(self):
        self.closed.set()
        with self.lock:
            for pair in self.pairs.values():
                pair.close()
            self.pairs.clear()
        atexit.unregister(self.close)

    def deliver(self, config, worker, source, manifest, transport, report, stop=None):
        with self.lock:
            started = time.monotonic()
            store.cancelled(stop)
            if self.closed.is_set():
                raise RuntimeError("Checkpoint transfer pool is closed")
            key = (
                worker.url,
                worker.ssh,
                worker.python,
                worker.physical_id,
                json.dumps(worker.rdma, sort_keys=True),
                tuple(config.ssh_options),
                str(worker.model_dir),
            )
            pair = self.pairs.get(key)
            if pair and pair.remote.process.poll() is not None:
                self.pairs.pop(key).close()
                pair = None
            if pair is None:
                if len(self.pairs) >= 8:
                    oldest = min(self.pairs, key=lambda k: self.pairs[k].last_used)
                    self.pairs.pop(oldest).close()
                pair = self.pairs[key] = Pair(config, worker)
            try:
                result = self._deliver(
                    pair, source, manifest, transport, report, stop, started
                )
                pair.last_used = time.monotonic()
                return result
            except TransferUnavailable as error:
                self.pairs.pop(key).close()
                if transport != "auto":
                    raise
                report(
                    "transport_fallback",
                    transport="ssh_rsync",
                    fallback_reason=str(error),
                )
                pair = self.pairs[key] = Pair(config, worker)
                try:
                    result = self._deliver(
                        pair, source, manifest, "ssh_rsync", report, stop, started
                    )
                    result["fallback_reason"] = str(error)
                    return result
                except BaseException:
                    self.pairs.pop(key).close()
                    raise
            except BaseException:
                self.pairs.pop(key).close()
                raise

    def _deliver(self, pair, source, manifest, transport, report, stop, started):
        worker, config = pair.worker, pair.config
        total = sum(r["size"] for r in manifest["files"].values())
        counters = dict(
            bytes_read=0,
            bytes_received=0,
            bytes_verified=0,
            bytes_reused=0,
            bytes_transferred=0,
            total_bytes=total,
        )
        selected = "jaccl_rdma" if transport in {"auto", "jaccl_rdma"} else "ssh_rsync"
        if transport == "auto" and not worker.rdma:
            selected = "ssh_rsync"
            report(
                "transport_fallback",
                transport=selected,
                fallback_reason="No direct RDMA pair configured for this worker",
            )

        def update(phase="transferring", **data):
            counters.update(data)
            elapsed = time.monotonic() - started
            report(
                phase,
                transport=selected,
                **counters,
                elapsed_seconds=elapsed,
                throughput_bytes_per_second=counters["bytes_transferred"]
                / max(elapsed, 1e-9),
                rate_kind=(
                    "logical_payload"
                    if selected == "jaccl_rdma"
                    else "logical_copy_progress"
                ),
            )

        root = worker.model_dir or str(config.model_dir)
        checked = 0

        def checking(data):
            nonlocal checked
            checked = data.get("bytes_checked", 0)
            report(
                "checking_partial",
                transport=selected,
                bytes_checked=checked,
                total_bytes=total,
            )

        state = pair.remote.call(
            "begin", root=root, manifest=manifest, stop=stop, progress=checking
        )
        if state["ready"]:
            selected = "verified_cache"
            update("worker_ready", bytes_reused=total, bytes_verified=total)
            return {"transport": selected, "verified": True, "path": state["path"]}
        counters["bytes_reused"] = state["bytes_reused"]
        counters["bytes_verified"] = state["bytes_reused"]
        if selected == "jaccl_rdma":
            self._rdma(pair, source, manifest, state, counters, update, stop)
        else:
            self._rsync(pair, source, manifest, state, total, update, stop)
        store.cancelled(stop)
        update("publishing")
        store.cancelled(stop)
        # Publication is the commit point. Once sent, await its definite result;
        # cancellation cannot undo a fully verified, atomically published snapshot.
        final = pair.remote.call("commit", manifest=manifest)
        update("worker_ready", bytes_verified=total)
        return {
            "transport": selected,
            "verified": True,
            "path": final["path"],
            "rdma_validated": pair.validated,
            "helper_pids": [p.pid for p in (pair.local, pair.remote) if p],
        }

    @staticmethod
    def _rdma(pair, source, manifest, state, counters, update, stop):
        update("connecting")
        pair.connect(stop)
        update("connected")
        common = dict(manifest=manifest, missing=state["missing"])
        receiver = pair.remote.submit("receive", path=state["path"], **common)
        sender = pair.local.submit(
            "send", root=str(Path(source).parent), path=str(source), **common
        )

        # Drain both streams while the file is in flight; never buffer a
        # checkpoint's worth of progress waiting for the other rank.
        def receiving(data):
            update(
                bytes_received=data["bytes_received"],
                bytes_verified=state["bytes_reused"] + data["bytes_verified"],
                bytes_transferred=data["bytes_received"],
            )

        def sending(data):
            counters["bytes_read"] = data["bytes_read"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            sent = pool.submit(pair.local.wait, sender, progress=sending, stop=stop)
            received = pool.submit(
                pair.remote.wait, receiver, progress=receiving, stop=stop
            )
            try:
                done, _ = concurrent.futures.wait(
                    [sent, received], return_when=concurrent.futures.FIRST_EXCEPTION
                )
                for future in done:
                    future.result()
            except BaseException:
                pair.close()  # Unblock the other rank's native collective too.
                raise

    @staticmethod
    def _rsync(pair, source, manifest, state, total, update, stop):
        worker, config = pair.worker, pair.config
        update()
        args = [
            "rsync",
            "-aL",
            "--checksum",
            "--partial",
            "--files-from=-",
            "-e",
            shlex.join(ssh_command(config)),
            str(source) + "/",
            (f"{worker.ssh}:{state['path']}/" if worker.ssh else state["path"] + "/"),
        ]
        copied = run_rsync(
            args,
            input="\n".join(manifest["files"]) + "\n",
            timeout=config.timeout,
            total_bytes=total,
            progress=lambda **data: update(**data),
            stop=stop,
        )
        if copied.returncode:
            raise RuntimeError("Checkpoint rsync failed: " + copied.stderr[-1000:])
        store.cancelled(stop)
        update("verifying_worker")
        pair.remote.call(
            "verify",
            manifest=manifest,
            stop=stop,
            progress=lambda data: update(
                "verifying_worker", bytes_verified=data["bytes_verified"]
            ),
        )
