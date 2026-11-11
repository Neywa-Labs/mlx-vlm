# Copyright © 2026 Neywa Labs Inc.
# SPDX-License-Identifier: MIT

"""Rank-zero admission order for the native continuous batch scheduler.

Only the generation thread uses the collective group. HTTP threads may arrive
in any order; a row is admitted only when every rank has prepared that row.
"""

import hashlib
import json
import time
from contextvars import ContextVar
from dataclasses import dataclass

import numpy as np


@dataclass
class RequestScope:
    request_id: str
    sequence: int = 0

    def next_id(self):
        self.sequence += 1
        return f"{self.request_id}:{self.sequence}"


request_scope = ContextVar("mlx_distributed_request", default=None)


def gather_json(group, value):
    """Exchange CPU control messages between completed model steps."""
    import mlx.core as mx

    # Native decoding prefetches the next token asynchronously. Finish that
    # work before issuing control collectives on the same group from a CPU
    # stream, so different rank timings cannot reorder collective operations.
    mx.synchronize()
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    lengths = mx.distributed.all_gather(
        mx.array([len(data)], dtype=mx.int32), group=group, stream=mx.cpu
    ).tolist()
    width = max(lengths)
    if width > 4 * 1024 * 1024:
        raise RuntimeError("Distributed scheduler control message exceeds 4 MiB")
    packed = mx.array(list(data) + [0] * (width - len(data)), dtype=mx.uint8)
    rows = mx.distributed.all_gather(packed, group=group, stream=mx.cpu)
    rows = np.asarray(rows).reshape(group.size(), width)
    return [json.loads(row[:length].tobytes()) for row, length in zip(rows, lengths)]


def request_signature(request, batch_key):
    digest = hashlib.sha256()

    def visit(value):
        if isinstance(value, dict):
            for key, item in sorted(value.items()):
                digest.update(str(key).encode())
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        elif hasattr(value, "shape") and hasattr(value, "dtype"):
            array = np.asarray(value)
            digest.update(str((array.shape, array.dtype)).encode())
            digest.update(array.tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True, default=str).encode())

    visit(request.raw_inputs)
    visit([batch_key, request.prompt_tokens, request.args.max_tokens])
    return digest.hexdigest()


def schedule(states, capacity):
    """Use rank 0's order, readiness intersection, and cancellation union."""
    if any(state.get("phase") != "schedule" for state in states):
        raise RuntimeError("Distributed generation phases diverged")
    if any(state["error"] for state in states):
        raise RuntimeError(
            "A distributed scheduler failed: "
            + str(next(state["error"] for state in states if state["error"]))
        )
    if any(state["active"] != states[0]["active"] for state in states[1:]):
        raise RuntimeError("Distributed active batches diverged")
    cancelled = set().union(*(state["cancelled"] for state in states))
    cancelled_requests = set().union(*(state["cancelled_requests"] for state in states))
    ready = [dict(state["ready"]) for state in states]
    selected = []
    for request_id, signature in states[0]["ready"]:
        if request_id.split(":", 1)[0] in cancelled_requests:
            continue
        if not all(request_id in rank for rank in ready):
            continue
        if any(rank[request_id] != signature for rank in ready):
            raise RuntimeError(
                "Distributed preprocessed inputs or sampling settings differ"
            )
        if capacity is None or len(selected) < capacity:
            selected.append(request_id)
    return (
        selected,
        cancelled,
        cancelled_requests,
        any(state["stop"] for state in states),
    )


class DistributedBatchScheduler:
    def __init__(self, worker, exchange=None):
        self.worker = worker
        self.exchange = exchange or (lambda value: gather_json(worker.group, value))
        self.pending = {}
        self.signatures = {}
        self.error = None

    def fail(self, error):
        self.error = str(error)

    def next(self, generator, groups, limit):
        items, _ = generator._collect_pending_requests(active=True)
        for item in items:
            if not item.request_id or ":" not in item.request_id:
                self.fail("Missing coordinated generation request ID")
            self.pending[item.request_id] = item
            try:
                self.signatures[item.request_id] = request_signature(
                    item, generator._batch_key(item)
                )
            except Exception as error:
                self.fail(error)
        active = [
            [list(key), [[uid, info["request_id"]] for uid, info in rows.items()]]
            for key, (_, rows) in groups.items()
            if rows
        ]
        running = sum(len(rows) for _, rows in groups.values())
        capacity = None if limit is None else max(0, limit - running)
        states = self.exchange(
            {
                "phase": "schedule",
                "ready": list(self.signatures.items()),
                "active": active,
                "cancelled": sorted(generator._drain_cancellations()),
                "cancelled_requests": self.worker.cancelled_requests(),
                "stop": generator._stop,
                "error": self.error,
            }
        )
        chosen, cancelled, request_cancels, stop = schedule(states, capacity)
        self.worker.remember_cancellations(request_cancels)
        for request_id in list(self.pending):
            if request_id.split(":", 1)[0] in request_cancels:
                item = self.pending.pop(request_id)
                self.signatures.pop(request_id)
                item.rqueue.put(RuntimeError("Distributed request cancelled"))
        for _, rows in groups.values():
            cancelled.update(
                uid
                for uid, info in rows.items()
                if info["request_id"].split(":", 1)[0] in request_cancels
            )
        requests = [self.pending.pop(request_id) for request_id in chosen]
        for request_id in chosen:
            self.signatures.pop(request_id)
        if not running and not requests:
            time.sleep(0.005)
        return requests, cancelled, stop and not running and not self.pending

    def verify_step(self, responses):
        values = [
            [
                response.uid,
                None if response.token is None else int(response.token),
                response.finish_reason,
            ]
            for response in responses
        ]
        states = self.exchange({"phase": "tokens", "values": values})
        if any(
            state.get("phase") != "tokens" or state != states[0] for state in states
        ):
            raise RuntimeError("Distributed token or completion decisions diverged")

    def abort(self, error, generator, groups):
        generator._load_error = error
        queued, _ = generator._collect_pending_requests(active=True)
        queues = [item.rqueue for item in [*self.pending.values(), *queued]]
        queues += [
            info["rqueue"] for _, rows in groups.values() for info in rows.values()
        ]
        for queue in queues:
            queue.put(error)
            queue.put(None)
        self.pending.clear()
        self.signatures.clear()
