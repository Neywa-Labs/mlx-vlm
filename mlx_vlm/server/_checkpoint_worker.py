# Copyright © 2026 Neywa Labs Inc.
"""SSH-controlled, leased transfer helper. Never joins an inference group.

This file and checkpoint_store are bootstrapped as source, allowing approved
workers to use their own Python/MLX installation without identical repo paths.
Weight bytes use JACCL; commands, manifests and progress use the SSH channel.
"""

import concurrent.futures
import fcntl
import hashlib
import json
import os
import platform
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

try:
    from . import checkpoint_store as store
except ImportError:  # Standalone, authenticated source bootstrap.
    import mlx_checkpoint_store as store

PREFIX = "MLX_TRANSFER "


class RDMAUnavailable(RuntimeError):
    """A recognized JACCL connection failure, before checkpoint delivery."""


def connection_call(function, *args, **kwargs):
    # JACCL currently exposes native connection errors as RuntimeError strings.
    # Match only known connection failures, never arbitrary backend exceptions.
    try:
        return function(*args, **kwargs)
    except RuntimeError as error:
        if re.fullmatch(
            r"\[jaccl\] (?:(?:Send|Recv) failed with errno=\d+|"
            r"(?:Couldn't connect|Accept failed) \(error: \d+\)|"
            r"Could not find device rdma_\w+ \(\d+ available\))",
            str(error),
        ):
            raise RDMAUnavailable(str(error)) from error
        raise


def emit(event, **data):
    print(PREFIX + json.dumps({"event": event, **data}), flush=True)


def physical_id():
    # Same physical identity contract as cluster_identity, with stdlib-only startup.
    if platform.system() == "Darwin":
        output = subprocess.check_output(
            ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"], text=True, timeout=3
        )
        raw = next(
            line.split("=")[1].strip()
            for line in output.splitlines()
            if '"IOPlatformUUID"' in line
        )
    else:
        raw = Path("/etc/machine-id").read_text().strip()
    return hashlib.sha256(raw.encode()).hexdigest()


def copy_chunks(group, command, report):
    import mlx.core as mx
    import numpy as np

    manifest = store.validate(command["manifest"])
    root = Path(command["path"])
    sending = group.rank() == 0
    stream = mx.new_stream(mx.cpu)
    chunk, depth = manifest["chunk_bytes"], 2
    idle, ready = queue.Queue(depth), queue.Queue(depth)
    errors, signatures = [], {}
    halted = threading.Event()
    arrays = []
    read = received = verified = 0
    started = time.monotonic()
    last = 0

    def progress(force=False):
        nonlocal last
        now = time.monotonic()
        if force or now - last >= 0.2:
            last = now
            report(
                bytes_read=read,
                bytes_received=received,
                bytes_verified=verified,
                elapsed_seconds=now - started,
            )

    def put(q, value):
        while not halted.is_set():
            try:
                q.put(value, timeout=0.1)
                return
            except queue.Full:
                pass
        raise RuntimeError("Transfer pipeline failed")

    def get(q):
        while not halted.is_set():
            try:
                return q.get(timeout=0.1)
            except queue.Empty:
                pass
        raise RuntimeError("Transfer pipeline failed")

    # CPU buffers must remain alive and unmodified through explicit send eval.
    if sending:
        for slot in range(depth):
            value = mx.zeros((chunk,), mx.uint8, stream=mx.cpu)
            mx.eval(value)
            view = memoryview(np.asarray(value))
            if view.readonly:
                raise RuntimeError("MLX CPU buffers are not writable in this version")
            arrays.append((value, view))
            idle.put(slot)

    def reader():
        nonlocal read
        try:
            for name, record in manifest["files"].items():
                path = store.safe_path(root, name)
                before = store.signature(path)
                with path.open("rb", buffering=0) as source:
                    for index in command["missing"][name]:
                        n = min(chunk, record["size"] - index * chunk)
                        slot = get(idle)
                        view = arrays[slot][1]
                        source.seek(index * chunk)
                        got = 0
                        while got < n:
                            count = source.readinto(view[got:n])
                            if not count:
                                raise OSError("Short checkpoint read")
                            got += count
                        read += n
                        progress()
                        put(ready, (slot, n))
                if store.signature(path) != before:
                    raise ValueError("Source checkpoint changed during transfer")
        except BaseException as error:
            errors.append(error)
            halted.set()

    def writer():
        nonlocal verified
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                pending = deque()

                def check(view, expected):
                    if hashlib.sha256(view).hexdigest() != expected:
                        raise ValueError("Checkpoint chunk checksum mismatch")
                    return len(view)

                for name, record in manifest["files"].items():
                    path = store.safe_path(root, name)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open(
                        "r+b" if path.exists() else "xb", buffering=0
                    ) as destination:
                        destination.truncate(record["size"])
                        for index in command["missing"][name]:
                            value, view = get(ready)
                            pending.append(
                                pool.submit(check, view, record["chunks"][index])
                            )
                            destination.seek(index * chunk)
                            remaining = view
                            while remaining:
                                count = destination.write(remaining)
                                if not count:
                                    raise OSError("Short checkpoint write")
                                remaining = remaining[count:]
                            if len(pending) >= 4:
                                verified += pending.popleft().result()
                            progress()
                            del view, value
                        while pending:
                            verified += pending.popleft().result()
                        store.flush_file(destination)
                    signatures[name] = store.signature(path)
        except BaseException as error:
            errors.append(error)
            halted.set()

    thread = threading.Thread(target=reader if sending else writer, daemon=True)
    thread.start()
    try:
        if sending:
            for name, record in manifest["files"].items():
                for index in command["missing"][name]:
                    slot, n = get(ready)
                    value = arrays[slot][0]
                    mx.eval(
                        mx.distributed.send(
                            value if n == chunk else value[:n],
                            1,
                            group=group,
                            stream=stream,
                        )
                    )
                    put(idle, slot)
        else:
            pending = deque()
            for name, record in manifest["files"].items():
                for index in command["missing"][name]:
                    n = min(chunk, record["size"] - index * chunk)
                    value = mx.distributed.recv(
                        (n,), mx.uint8, 0, group=group, stream=stream
                    )
                    mx.async_eval(value)
                    pending.append(value)
                    if len(pending) >= depth:
                        value = pending.popleft()
                        mx.eval(value)
                        view = memoryview(np.asarray(value))
                        received += len(view)
                        put(ready, (value, view))
            while pending:
                value = pending.popleft()
                mx.eval(value)
                view = memoryview(np.asarray(value))
                received += len(view)
                put(ready, (value, view))
        thread.join()
        if errors:
            raise errors[0]
        progress(True)
        return signatures
    finally:
        halted.set()


def main():
    commands = queue.Queue()
    renewed = [time.monotonic()]

    # Native RDMA may block. EOF/lease expiry must kill this helper independently.
    def control():
        try:
            for line in sys.stdin:
                message = json.loads(line)
                renewed[0] = time.monotonic()
                if message["op"] == "shutdown":
                    os._exit(0)
                if message["op"] != "ping":
                    commands.put(message)
        finally:
            os._exit(0)

    def watchdog():
        while True:
            time.sleep(1)
            if time.monotonic() - renewed[0] > 30:
                os._exit(124)

    threading.Thread(target=control, daemon=True).start()
    threading.Thread(target=watchdog, daemon=True).start()
    group = None
    lock = None
    current = None
    signatures = None
    emit("hello", physical_id=physical_id(), pid=os.getpid(), protocol=1)
    with tempfile.TemporaryDirectory(prefix="mlx-transfer-") as scratch:
        while True:
            command = commands.get()
            request = command["id"]
            op = command["op"]
            report = lambda **data: emit("progress", id=request, **data)
            try:
                result = {}
                if op == "init":
                    import mlx.core as mx
                    import numpy as np

                    if group is not None:
                        raise ValueError("Transfer group is already initialized")
                    if not mx.distributed.is_available("jaccl"):
                        raise RDMAUnavailable("JACCL is unavailable on this worker")
                    for name in list(os.environ):
                        if name.startswith(
                            ("MLX_JACCL_", "MLX_RING_", "JACCL_")
                        ) or name in {"MLX_HOSTFILE", "MLX_RANK", "MLX_IBV_DEVICES"}:
                            del os.environ[name]
                    devices = Path(scratch) / "devices.json"
                    devices.write_text(json.dumps(command["devices"]))
                    os.environ.update(
                        MLX_RANK=str(command["rank"]),
                        MLX_IBV_DEVICES=str(devices),
                        MLX_JACCL_COORDINATOR=command["coordinator"],
                    )
                    emit("initializing", id=request)
                    group = connection_call(
                        mx.distributed.init, backend="jaccl", strict=True
                    )
                    if group.size() != 2 or group.rank() != command["rank"]:
                        raise ValueError("Unexpected transfer group membership")
                    # Verify this pair can communicate using a session-specific challenge.
                    token = np.frombuffer(
                        bytes.fromhex(command["challenge"]), dtype=np.uint8
                    ).astype(np.int32)
                    probe = mx.distributed.all_sum(
                        mx.array(token), group=group, stream=mx.cpu
                    )
                    connection_call(mx.eval, probe)
                    if not np.array_equal(np.asarray(probe), token * 2):
                        raise ValueError("RDMA connection challenge mismatch")
                    result = {"rdma_validated": True, "mlx_version": mx.__version__}
                elif op == "begin":
                    if lock is not None:
                        raise ValueError("A checkpoint transaction is already active")
                    current = store.SnapshotStore(command["root"])
                    manifest = store.validate(command["manifest"])
                    lock = (
                        current.receipts / (store.snapshot_id(manifest) + ".lock")
                    ).open("a")
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    final = current.existing(manifest)
                    if final:
                        lock.close()
                        lock = None
                        result = {"ready": True, "path": str(final)}
                    else:
                        current.writable_partial(manifest)
                        checked, last_report = 0, 0

                        def checking(n):
                            nonlocal checked, last_report
                            checked += n
                            now = time.monotonic()
                            if now - last_report >= 0.2:
                                report(bytes_checked=checked)
                                last_report = now

                        missing, reused = current.missing(manifest, progress=checking)
                        report(bytes_checked=checked)
                        result = {
                            "ready": False,
                            "missing": missing,
                            "bytes_reused": reused,
                            "path": str(current.partial(manifest)),
                        }
                elif op in {"send", "receive"}:
                    if group is None:
                        raise ValueError("No validated transfer group")
                    if op == "send":
                        source = store.SnapshotStore(command["root"])
                        final = source.existing(command["manifest"])
                        if final is None or str(final) != command["path"]:
                            raise ValueError("Unverified transfer source")
                    elif (
                        lock is None
                        or str(current.partial(command["manifest"])) != command["path"]
                    ):
                        raise ValueError("No matching checkpoint transaction")
                    signatures = copy_chunks(group, command, report)
                elif op == "commit":
                    if lock is None or signatures is None:
                        raise ValueError("No verified transfer to commit")
                    result = {
                        "path": str(current.publish(command["manifest"], signatures))
                    }
                    lock.close()
                    lock = None
                    signatures = None
                elif op == "verify":
                    if lock is None:
                        raise ValueError("No checkpoint transaction")
                    manifest = store.validate(command["manifest"])
                    partial = current.writable_partial(manifest)
                    checked = 0
                    last_report = 0

                    def verified(n):
                        nonlocal checked, last_report
                        checked += n
                        now = time.monotonic()
                        if now - last_report >= 0.2:
                            report(bytes_verified=checked)
                            last_report = now

                    signatures = {}
                    for name, record in manifest["files"].items():
                        path = store.safe_path(partial, name)
                        store.verify_chunks(
                            path, record, manifest["chunk_bytes"], progress=verified
                        )
                        with path.open("rb") as stream:
                            store.flush_file(stream)
                        signatures[name] = store.signature(path)
                    report(bytes_verified=checked)
                else:
                    raise ValueError("Unknown transfer command")
                emit("result", id=request, **result)
            except BaseException as error:
                emit(
                    "error",
                    id=request,
                    message=str(error),
                    code=(
                        "rdma_unavailable"
                        if isinstance(error, RDMAUnavailable)
                        else "helper_error"
                    ),
                )
                # Never reuse a collective after an incomplete send/recv sequence.
                os._exit(1)


if __name__ == "__main__":
    main()
