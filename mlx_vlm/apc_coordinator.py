"""Model-agnostic coordination for Automatic Prefix Caching.

The storage manager owns hashes, blocks, eviction and persistence.  This
coordinator owns the model cache plan and is the only layer generation code
needs to call.  It follows vLLM's split between cache specs/groups and a
coordinator, adapted to MLX's contiguous runtime cache objects.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Callable, List, Optional, Sequence, Tuple

import mlx.core as mx

from .apc_adapters import (
    PrefixCachePlan,
    build_prefix_cache_plan,
    build_prefix_cache_plan_from_caches,
    cache_memory_components,
    clone_cache_entry,
)
from .models.cache import (
    AttentionSinkKVCache,
    BatchKVCache,
    CacheList,
    CacheMemory,
    KVCache,
)

logger = logging.getLogger(__name__)


class _NotDeferrable(Exception):
    pass


class _RowPrefix:
    """The first ``n`` tokens of one row of an append-only K/V cache."""

    __slots__ = ("n",)

    def __init__(self, n: int):
        self.n = n


def _append_only(c: Any) -> bool:
    """Dense K/V whose positions are never rewritten once written (exact
    types: not sliding windows, quantized or custom caches)."""
    return (
        type(c) in (BatchKVCache, KVCache)
        or (type(c) is AttentionSinkKVCache and c.max_size is None)
    ) and c.keys is not None


def _row_span(c: Any, idx: int) -> Optional[Tuple[int, int]]:
    """``(start, end)`` of a row's tokens in an append-only cache."""
    if type(c) is BatchKVCache:
        return int(c.left_padding[idx].item()), c._idx
    if idx != 0 or getattr(c, "_end", c.offset) != c.offset:
        return None
    return 0, c.offset


def _defer_entry(c: Any, idx: int, n: int, eval_targets: List[mx.array]) -> Any:
    if isinstance(c, CacheList):
        return CacheList(*(_defer_entry(ch, idx, n, eval_targets) for ch in c.caches))
    if _append_only(c):
        span = _row_span(c, idx)
        if span is None or span[1] - span[0] != n:
            raise _NotDeferrable
        return _RowPrefix(n)
    # A sliding window or recurrent state moves on: copy it now (it is small).
    extract = getattr(c, "extract", None)
    row = (
        extract(idx)
        if callable(extract)
        else clone_cache_entry(c, min_capacity_tokens=None, eval_targets=eval_targets)
    )
    if row is None:
        raise _NotDeferrable
    from .apc import _collect_mx_arrays

    _collect_mx_arrays(row.state, eval_targets)
    return row


def _prefix_bytes(live: Any, tmpl: Any, *, copied: bool = False) -> int:
    """Bytes of the K/V prefixes still to be cut from ``live``; with
    ``copied``, plus the state already copied at the boundary."""
    if isinstance(tmpl, CacheList):
        return sum(
            _prefix_bytes(l, t, copied=copied) for l, t in zip(live.caches, tmpl.caches)
        )
    if not isinstance(tmpl, _RowPrefix):
        if not copied:
            return 0
        from .apc import _cache_nbytes

        return _cache_nbytes(tmpl)
    if live.keys is None:
        return 0
    k, v = live.keys, live.values
    per_token = k.shape[1] * k.shape[3] * k.itemsize + v.shape[1] * v.shape[3] * (
        v.itemsize
    )
    return per_token * tmpl.n


def _finishable(live: Any, tmpl: Any, idx: int) -> bool:
    """Whether ``_finish_entry`` can cut the row's prefix from ``live``."""
    if isinstance(tmpl, CacheList):
        return (
            isinstance(live, CacheList)
            and len(live.caches) == len(tmpl.caches)
            and all(_finishable(l, t, idx) for l, t in zip(live.caches, tmpl.caches))
        )
    if not isinstance(tmpl, _RowPrefix):
        return True
    # The row may have been merged into a batch (or the batch filtered) since:
    # read it from its current position.
    span = _row_span(live, idx) if _append_only(live) else None
    return span is not None and span[0] >= 0 and span[0] + tmpl.n <= span[1]


def _finish_entry(live: Any, tmpl: Any, idx: int, release: bool) -> Any:
    if isinstance(tmpl, CacheList):
        return CacheList(
            *(
                _finish_entry(l, t, idx, release)
                for l, t in zip(live.caches, tmpl.caches)
            )
        )
    if not isinstance(tmpl, _RowPrefix):
        return tmpl
    start = _row_span(live, idx)[0]
    out = AttentionSinkKVCache() if type(live) is AttentionSinkKVCache else KVCache()
    out.state = (
        mx.contiguous(live.keys[idx : idx + 1, :, start : start + tmpl.n]),
        mx.contiguous(live.values[idx : idx + 1, :, start : start + tmpl.n]),
    )
    mx.eval(out.keys, out.values)
    if release:
        # Layer by layer: the snapshot replaces the live cache as it is built.
        live.keys = live.values = None
    return out


class DeferredCheckpoint:
    """A prefill checkpoint whose dense K/V is still the request's (APC_SINGLE_COPY).

    Only the state that moves on (a sliding window, recurrent or convolution
    state) is copied at the boundary. The append-only K/V rows are read from
    the live cache when the row leaves the batch; when it is the batch's last
    row (``release``), each layer's buffers are freed as its snapshot is made,
    so the turn never holds a second copy of the context. ``taken``: the
    lease of the checkpoint the request restored by taking it.
    """

    def __init__(self, coordinator, token_ids, extra_hash, template, taken=None):
        self.coordinator = coordinator
        self.token_ids = list(token_ids)
        self.extra_hash = extra_hash
        self.template = template
        self.taken = taken

    def finish(
        self, prompt_cache: Sequence[Any], batch_idx: int, *, release: bool
    ) -> bool:
        from .apc import apc_trace

        manager = self.coordinator.manager
        if len(prompt_cache) != len(self.template) or not all(
            _finishable(l, t, batch_idx) for l, t in zip(prompt_cache, self.template)
        ):
            return False
        size = sum(
            _prefix_bytes(l, t, copied=True)
            for l, t in zip(prompt_cache, self.template)
        )
        if release:
            # The snapshot takes the live buffers' place. Decide before they
            # are freed, so a refused store leaves them for a shorter one.
            admitted = manager.admits_retained(size, allocation=0, prefill_reserve=0)
            # A disk tier still takes what memory cannot.
            proceed = admitted or manager.disk is not None
        else:
            # The batch keeps its rows: this snapshot is a copy. Admit its new
            # bytes on top of the whole reserve of any prefill still running.
            admitted = proceed = manager.admits_retained(
                size,
                allocation=sum(
                    _prefix_bytes(l, t) for l, t in zip(prompt_cache, self.template)
                ),
            )
        if not admitted:
            apc_trace(
                "store",
                mode="deferred",
                ok=False,
                reason="memory",
                token_len=len(self.token_ids),
                size_mb=size >> 20,
            )
            with manager.lock:
                manager.stats.memory_skips += 1
        if not proceed:
            return False
        snapshot = [
            _finish_entry(live, tmpl, batch_idx, release)
            for live, tmpl in zip(prompt_cache, self.template)
        ]
        apc_trace(
            "store", mode="deferred", release=release, token_len=len(self.token_ids)
        )
        return manager.store_exact_cache(
            self.token_ids,
            snapshot,
            extra_hash=self.extra_hash,
            owned=True,
            allocated=True,
        )


def finish_deferred_checkpoints(
    deferred: Sequence[DeferredCheckpoint],
    prompt_cache: Sequence[Any],
    batch_idx: int,
    *,
    release: bool,
) -> None:
    """Store a leaving row's deferred checkpoints, the longest last (it may take
    the row's buffers). With APC_EVICT_SUPERSEDED the longest would evict the
    others at once, so only it is stored, or the next shorter while one is
    refused. The lease of a taken checkpoint ends first: once stored, another
    request may take it."""
    ordered = sorted(deferred, key=lambda d: len(d.token_ids))
    if not ordered:
        return
    manager = ordered[-1].coordinator.manager
    for d in ordered:
        if d.taken is not None:
            manager.forget_taken(d.token_ids, d.extra_hash, d.taken)

    def finish(d: DeferredCheckpoint, release: bool) -> bool:
        try:
            return d.finish(prompt_cache, batch_idx, release=release)
        except Exception as e:  # a lost checkpoint must not fail the request
            logger.warning("APC deferred checkpoint store failed: %s", e)
            return False

    if manager.evict_superseded:
        for d in reversed(ordered):
            if finish(d, release):
                break
    else:
        for i, d in enumerate(ordered):
            finish(d, release and i == len(ordered) - 1)


class PrefillMemoryPlan:
    """Per-manager budgets from cache layouts or conservative opaque bounds.

    Empty caches use the device reserve until dimensions are known.
    """

    def __init__(self):
        self.components: List[CacheMemory] = []
        self.lengths: List[int] = []
        self.prefix_lengths: List[int] = []
        self.chunk_size: Optional[int] = None
        self.extra_reserve_bytes = 0

    def prepare(
        self, lengths, *, chunk_size=None, prefix_lengths=None, extra_reserve_bytes=0
    ) -> int:
        self.extra_reserve_bytes = max(0, int(extra_reserve_bytes))
        self.lengths = [max(0, int(n)) for n in lengths]
        self.prefix_lengths = (
            list(prefix_lengths)
            if prefix_lengths is not None
            else [0] * len(self.lengths)
        )
        self.chunk_size = chunk_size if chunk_size and chunk_size > 0 else None
        return self.reserve_bytes()

    def _capacity(self) -> int:
        # Prefixes and suffixes are padded separately; cold prefixes are zero.
        return max(self.prefix_lengths) + max(
            n - p for n, p in zip(self.lengths, self.prefix_lengths)
        )

    def reserve_bytes(self, live_bytes: int = 0) -> int:
        if not self.lengths or not max(self.lengths):
            return 0
        capacity = self._capacity()
        row_bytes = sum(c.footprint(capacity, self.chunk_size) for c in self.components)
        return self.extra_reserve_bytes + max(
            0, 2 * len(self.lengths) * row_bytes - live_bytes
        )

    def remaining_reserve_bytes(self, prompt_cache, token_count) -> Optional[int]:
        """What the planned prefill still allocates once a snapshot of
        ``prompt_cache`` at ``token_count`` tokens is stored, or None for the
        whole reserve (``reserve_bytes``).

        With no update left (a finished prefill) only the merge reserve
        remains. Otherwise dense K/V whose buffers already hold every planned
        token (a restored prefix is allocated at prompt capacity) is written in
        place; dense K/V that must still grow, and layouts without a known
        profile, keep the whole reserve; sliding windows and recurrent state
        allocate replacement buffers on each update, so those are kept."""
        if not self.lengths or not max(self.lengths):
            return 0
        capacity = self._capacity()
        if capacity <= token_count:
            return self.extra_reserve_bytes
        replacements = 0
        for c in cache_memory_components(prompt_cache, token_count):
            if c.fallback:
                return None
            if c.window_size is not None or not c.bytes_per_token:
                # A replacement is allocated at the buffers' rounded capacity,
                # or at what they hold already (a compaction keeps it).
                replacements += max(
                    c.source_bytes, c.footprint(capacity, self.chunk_size)
                )
            elif (
                # Dense buffers grow by steps from wherever they were, so they
                # need not be step-aligned: compare the tokens they hold,
                # unrounded (half a token absorbs the per-row rounding).
                replace(c, step=1).footprint(capacity, self.chunk_size) - c.source_bytes
                > c.bytes_per_token / 2
            ):
                return None
        return self.extra_reserve_bytes + replacements

    def observe(self, components, *, live_bytes: int) -> int:
        previous = self.components
        self.components = list(components)
        if len(previous) == len(self.components):
            for i, (old, new) in enumerate(zip(previous, self.components)):
                if not new.source_bytes:
                    self.components[i] = old
                elif old.fallback and new.fallback:
                    self.components[i] = replace(
                        new,
                        bytes_per_token=max(old.bytes_per_token, new.bytes_per_token),
                    )
        return self.reserve_bytes(live_bytes)

    def observe_cache(
        self, prompt_cache, token_count, *, batch_size=1, live_bytes=None
    ) -> int:
        """Update dimensions and return the reserve beyond live buffers."""
        from .apc import _cache_nbytes

        if live_bytes is None:
            live_bytes = _cache_nbytes(prompt_cache)
        return self.observe(
            cache_memory_components(prompt_cache, token_count, batch_size=batch_size),
            live_bytes=live_bytes,
        )

    def observe_kv(self, keys, values, *, live_bytes: int) -> int:
        # Raw float KV slabs expose growth through their token axis.
        return self.observe(
            (
                CacheMemory(
                    source_bytes=k.nbytes + v.nbytes,
                    bytes_per_token=(k.nbytes + v.nbytes) / k.shape[2],
                )
                for k, v in zip(keys, values)
                if k.shape[2]
            ),
            live_bytes=live_bytes,
        )

    def restore_bytes(self, prompt_cache, token_count, capacity) -> int:
        components = cache_memory_components(prompt_cache, token_count)
        return sum(
            max(c.source_bytes, c.footprint(capacity, self.chunk_size))
            for c in components
        )


class APCCoordinator:
    """Coordinate prefix reuse across every cache group in a model.

    Native dense K/V layouts use the block pool.  Windowed, recurrent,
    composite and custom layouts use restorable checkpoints at the same token
    boundary.  The distinction is deliberately private to this class.
    """

    def __init__(self, manager: Any, model: Any):
        self.manager = manager
        self.model = model
        self.plan: PrefixCachePlan = build_prefix_cache_plan(model)

    def prepare_prefill(
        self,
        prompt_lengths,
        *,
        prefill_step_size=None,
        prefix_lengths=None,
        extra_reserve_bytes=0,
    ) -> bool:
        if self.enabled:
            lengths = (
                [prompt_lengths] if isinstance(prompt_lengths, int) else prompt_lengths
            )
            reserve = self.manager.memory_plan.prepare(
                lengths,
                chunk_size=prefill_step_size,
                prefix_lengths=prefix_lengths,
                extra_reserve_bytes=extra_reserve_bytes,
            )
            return self.manager.prepare_prefill(reserve)
        return True

    def finish_prefill(self) -> None:
        """Release the transient plan after admission is deferred or completed."""
        if self.enabled:
            self.manager.memory_plan.prepare([])
            self.manager._prefill_reserve_bytes = 0

    def observe_cache(self, prompt_cache, token_count, *, batch_size=1) -> None:
        """Update runtime dimensions independently of snapshot persistence."""
        reserve = self.manager.memory_plan.observe_cache(
            prompt_cache, token_count, batch_size=batch_size
        )
        self.manager.prepare_prefill(reserve)

    @property
    def enabled(self) -> bool:
        return self.manager is not None and self.plan.restorable

    @property
    def strategy(self) -> Optional[str]:
        return self.plan.strategy if self.enabled else None

    @property
    def is_checkpoint(self) -> bool:
        return self.strategy == "checkpoint"

    @property
    def legacy_mode(self) -> Optional[str]:
        return self.plan.legacy_mode if self.enabled else None

    def fresh_cache(self) -> List[Any]:
        language_model = getattr(self.model, "language_model", self.model)
        make_cache = getattr(language_model, "make_cache", None) or getattr(
            self.model, "make_cache", None
        )
        if callable(make_cache):
            return list(make_cache())
        from .models.cache import make_prompt_cache

        return list(make_prompt_cache(language_model))

    def lookup(
        self,
        token_ids: Sequence[int],
        *,
        extra_hash: int,
        safe_lookup_min: int,
        suffix_is_text_only: Callable[[int], bool],
        prefix_has_media: Callable[[int], bool],
        defer_on_pressure: bool = False,
        take: bool = False,
    ) -> Optional[dict]:
        """``take``: the caller hands back or replaces a checkpoint it takes
        (APC_SINGLE_COPY), as the batch generator does. Only checkpoint
        layouts defer their stores, so other layouts always copy."""
        if not self.enabled:
            return None
        from .apc import apc_lookup_plan

        hit = apc_lookup_plan(
            self.manager,
            token_ids,
            extra_hash=extra_hash,
            apc_mode=self.legacy_mode,
            safe_lookup_min=safe_lookup_min,
            suffix_is_text_only=suffix_is_text_only,
            prefix_has_media=prefix_has_media,
            defer_on_pressure=defer_on_pressure,
            take=take and self.is_checkpoint,
        )
        if hit is not None:
            hit["cache_plan"] = self.plan
        return hit

    def checkpoint_len(
        self, token_ids: Sequence[int], media_token_ids: set[int]
    ) -> int:
        """Reusable checkpoint before the final guard token(s)."""
        if not self.enabled or not self.is_checkpoint:
            return 0
        from .apc import adjust_prefix_to_text_suffix_boundary

        return adjust_prefix_to_text_suffix_boundary(
            token_ids,
            len(token_ids) - self.manager.exact_cache_guard_tokens,
            media_token_ids,
            max_prefix_tokens=len(token_ids) - 1,
        )

    def checkpoint_lengths(
        self,
        token_ids: Sequence[int],
        media_token_ids: set[int],
        *,
        extra_hash: int = 0,
        stable_prefix_len: int = 0,
    ) -> List[int]:
        """Bounded intermediate states plus the final conversation checkpoint.

        Stateful caches cannot roll back the final snapshot to a divergence.
        Capture earlier states while prefilling, aligned across requests. Limit
        captures to the resident entry budget (two for a disk-only manager),
        rather than copying an ever-growing cache at every prefill chunk.
        """
        final = self.checkpoint_len(token_ids, media_token_ids)
        if final <= 0:
            return []
        interval = self.manager.checkpoint_interval_tokens
        budget = self.manager._exact_cache_max or (2 if self.manager.disk else 1)
        if interval <= 0 or budget <= 1:
            return [final]
        from .apc import adjust_prefix_to_text_suffix_boundary

        block_size = self.manager.block_size
        interval = ((interval + block_size - 1) // block_size) * block_size
        last = ((final - 1) // interval) * interval
        first = max(interval, last - (budget - 2) * interval)
        lengths = {final}
        # Prefer the observed stable prefix over positions inside a changing
        # suffix. The final checkpoint still serves exact prompt extensions.
        stable = self.manager.checkpoint_prefix_length(token_ids, extra_hash=extra_hash)
        candidates = [stable_prefix_len, stable] + list(
            range(last, first - 1, -interval)
        )
        for boundary in candidates:
            if boundary < self.manager.exact_cache_min_tokens:
                continue
            boundary = adjust_prefix_to_text_suffix_boundary(
                token_ids, boundary, media_token_ids, max_prefix_tokens=final
            )
            if self.manager.exact_cache_min_tokens <= boundary < final:
                lengths.add(boundary)
            if len(lengths) >= budget:
                break
        return sorted(lengths)

    def merge_rows(
        self,
        picks: Sequence[Optional[dict]],
        prefix_lens: Sequence[int],
        *,
        kv_quant_config: Optional[dict] = None,
    ) -> Tuple[Optional[List[Any]], int]:
        """Materialize a mixed batch while preserving the caller's cache rows."""
        source = (
            picks[0]["warm_cache"]
            if self.is_checkpoint and len(picks) == 1 and picks[0] is not None
            else None
        )
        if self.is_checkpoint:
            picks = [
                (
                    {**pick, "warm_cache": list(pick["warm_cache"])}
                    if pick is not None
                    else None
                )
                for pick in picks
            ]
        caches, prefix_len = self._merge_and_release_cache_rows(
            picks, prefix_lens, kv_quant_config=kv_quant_config
        )
        if caches is not None and source is not None:
            from .apc import _detach_borrowed_cache_entries

            caches = _detach_borrowed_cache_entries(caches, source)
        return caches, prefix_len if caches is not None else 0

    def _merge_and_release_cache_rows(
        self,
        picks: Sequence[Optional[dict]],
        prefix_lens: Sequence[int],
        *,
        kv_quant_config: Optional[dict] = None,
    ) -> Tuple[Optional[List[Any]], int]:
        """Merge scheduler-owned rows; discard picks after success or failure.

        Exact-cache input entries are released layer by layer. Block leases
        retain their existing lifecycle in the scheduler.
        """
        from .apc import _merge_and_release_cache_rows, make_warm_batch_kv_cache_multi

        memory = self.manager.memory_plan
        if len(prefix_lens) == len(memory.lengths):
            if not self.prepare_prefill(
                memory.lengths,
                prefill_step_size=memory.chunk_size,
                prefix_lengths=prefix_lens,
                extra_reserve_bytes=memory.extra_reserve_bytes,
            ):
                from .apc import APCAdmissionError

                for pick in picks:  # the request waits: its checkpoints go back
                    self._return_taken(pick)
                raise APCAdmissionError("APC batch merge deferred by memory admission")

        if self.is_checkpoint:
            row_caches = [
                pick["warm_cache"] if pick is not None else self.fresh_cache()
                for pick in picks
            ]
            caches, max_prefix = _merge_and_release_cache_rows(
                row_caches,
                prefix_lens,
                kv_quant_config=kv_quant_config,
            )
        else:
            caches, max_prefix = make_warm_batch_kv_cache_multi(
                list(picks),
                num_layers=len(self.plan.components),
                kv_quant_config=kv_quant_config,
            )
        if caches is not None:
            with self.manager.lock:
                self.manager.stats.restored_tokens += sum(prefix_lens)
        return caches, max_prefix

    def materialize_single(
        self,
        hit: dict,
        *,
        min_capacity_tokens: int,
        kv_quant_config: Optional[dict] = None,
    ) -> List[Any]:
        warm_cache = hit.get("warm_cache")
        if warm_cache is None:
            from .apc import make_warm_kv_cache

            warm_cache = make_warm_kv_cache(
                hit.get("matched_blocks", []),
                min_capacity_tokens=min_capacity_tokens,
                kv_quant_config=kv_quant_config,
            )
        if warm_cache:
            with self.manager.lock:
                self.manager.stats.restored_tokens += hit["prefix_len"]
        return warm_cache

    def defer_checkpoint(
        self,
        token_ids: Sequence[int],
        prompt_cache: Sequence[Any],
        *,
        extra_hash: int = 0,
        batch_idx: int = 0,
    ) -> Optional[DeferredCheckpoint]:
        """With APC_SINGLE_COPY, a checkpoint of a row's first
        ``len(token_ids)`` tokens to store when the row leaves its batch
        (``DeferredCheckpoint``), or None: store it now with ``store_checkpoint``."""
        if not (self.enabled and self.is_checkpoint and self.manager.single_copy):
            return None
        if len(token_ids) < self.manager.exact_cache_min_tokens:
            return None
        if not build_prefix_cache_plan_from_caches(prompt_cache).restorable:
            return None
        eval_targets: List[mx.array] = []
        try:
            template = [
                _defer_entry(c, batch_idx, len(token_ids), eval_targets)
                for c in prompt_cache
            ]
        except _NotDeferrable:
            return None
        if eval_targets:
            mx.eval(eval_targets)
        return DeferredCheckpoint(self, token_ids, extra_hash, template)

    def store_checkpoint(
        self,
        token_ids: Sequence[int],
        prompt_cache: Sequence[Any],
        *,
        extra_hash: int = 0,
        batch_idx: Optional[int] = None,
    ) -> bool:
        if not self.enabled or not self.is_checkpoint:
            return False
        from .apc import (
            _cache_nbytes,
            _prompt_cache_is_batch_shaped,
            snapshot_prompt_cache_row,
        )

        # Decide before making room evicts the checkpoint this one continues,
        # and from the live buffers: a row snapshot loses their capacity.
        prefill_reserve = self.manager.store_prefill_reserve(
            token_ids, prompt_cache, extra_hash
        )
        # Batch extraction can itself allocate a full row before store_exact_cache
        # decides whether it can afford another retained snapshot.
        if _prompt_cache_is_batch_shaped(prompt_cache):
            if self.manager.disk is not None:
                self.manager.disk.flush()
            if not self.manager._make_room(
                _cache_nbytes(prompt_cache), prefill_reserve=prefill_reserve
            ):
                with self.manager.lock:
                    self.manager.stats.memory_skips += 1
                return False
        snapshot = snapshot_prompt_cache_row(prompt_cache, batch_idx or 0, clone=False)
        if snapshot is None:
            return False
        return self.manager.store_exact_cache(
            token_ids, snapshot, extra_hash=extra_hash, prefill_reserve=prefill_reserve
        )

    def commit(
        self,
        prompt_cache: Sequence[Any],
        token_ids: Sequence[int],
        *,
        batch_idx: Optional[int] = None,
        extra_hash: int = 0,
        skip_first_n_tokens: int = 0,
        blocks_in_use: Sequence[Any] = (),
    ) -> bool:
        """Store one completed prefix and release any block leases."""
        if not self.enabled:
            return False
        if self.is_checkpoint:
            try:
                return self.store_checkpoint(
                    token_ids,
                    prompt_cache,
                    batch_idx=batch_idx,
                    extra_hash=extra_hash,
                )
            finally:
                self.manager.release(blocks_in_use)

        from .apc import commit_prefix_blocks

        commit_prefix_blocks(
            self.manager,
            list(prompt_cache),
            token_ids,
            batch_idx=batch_idx,
            extra_hash=extra_hash,
            skip_first_n_tokens=skip_first_n_tokens,
            blocks_in_use=blocks_in_use,
        )
        return True

    def _return_taken(self, hit: Optional[dict]) -> None:
        if hit is not None and hit.get("warm_cache") is not None:
            self.manager.return_taken(
                hit["full_input_ids"][: hit["prefix_len"]],
                hit["warm_cache"],
                hit.get("extra_hash", 0),
            )

    def copy_taken(self, hit: Optional[dict]) -> None:
        """Copy instead, as without APC_SINGLE_COPY, for a row that cannot
        store from its own cache later (a right-padded cohort): a compact clone
        of the checkpoint the hit took goes back to the store, and the row keeps
        the cache that lookup already expanded to its capacity."""
        if hit is None:
            return
        cache = hit.get("warm_cache")
        lease = getattr(cache, "lease", None)
        prefix = hit["full_input_ids"][: hit["prefix_len"]]
        extra_hash = hit.get("extra_hash", 0)
        if lease is None or not self.manager.is_taken(prefix, extra_hash, lease):
            return
        from .apc import _clone_prompt_cache_for_apc

        clone = _clone_prompt_cache_for_apc(cache)
        if clone is not None:
            self.manager.return_taken(prefix, clone, extra_hash, lease=lease)

    def release_hit(self, hit: Optional[dict]) -> None:
        """Discard an unused lookup result: release its blocks and hand back a
        checkpoint it took (APC_SINGLE_COPY)."""
        if hit is not None and self.manager is not None:
            self.manager.release(hit.get("matched_blocks", ()))
            self._return_taken(hit)
