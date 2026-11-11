# Copyright © 2026 Neywa Labs Inc.
# SPDX-License-Identifier: MIT
"""Bridge real prefill callbacks to async SSE without blocking the event loop."""

import asyncio
from queue import Empty, Queue

from .generation import StreamingToken


class ProgressChannel:
    def __init__(self):
        self.updates = Queue()

    def __call__(self, processed, total, status):
        self.updates.put(
            StreamingToken(
                text="",
                token=0,
                logprobs=0,
                finish_reason=None,
                token_count=0,
                prompt_progress={
                    "status": status,
                    "processed_tokens": processed,
                    "total_tokens": total,
                },
            )
        )

    async def consume(self, iterator):
        end = object()

        def advance():
            return next(iterator, end)

        task = None
        try:
            while True:
                task = asyncio.create_task(asyncio.to_thread(advance))
                while not task.done():
                    try:
                        yield self.updates.get_nowait()
                    except Empty:
                        await asyncio.sleep(0.005)
                while not self.updates.empty():
                    yield self.updates.get_nowait()
                value = await task
                if value is end:
                    break
                yield value
        finally:
            # A Python generator cannot be closed while its producer is inside
            # a native model call. Wait for that bounded step before closing.
            if task is not None:
                await asyncio.shield(task)
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
