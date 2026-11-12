# Reusable checkpoint transfer validation — 2026-11-12

This implementation is based on PR #2712 commit
`a40965c35d943d58e35209947292590be91573fb`. Validation used uncommitted MLX-VLM
changes. No Nativ code, macOS network settings or unrelated services were changed.

## Hardware and workload

- Two directly connected M3 Ultra Macs, 512 GiB memory each, negotiated 80 Gb/s
  TB5 link, actual JACCL connection challenge passed.
- MLX `0.32.5.dev20261030+1ba7f96`, macOS 27.0 / 27.0.1.
- Released `GLM-4.7-Flash-4bit` safetensors shard: **5,357,453,035 bytes**,
  SHA-256 `540741ab088c32f7338636c82c601399c0db6c11c965ac2c2865e9400d32dd8b`.
  A two-byte configuration fixture accompanied this shard. This is delivery
  validation, not a new model-inference or model-quality claim.
- Source data was warm in the filesystem cache. Destinations were fresh for
  each transfer trial, except explicitly labelled destination-cache hits.
- 16 MiB chunks, send/receive depth two, four receiver hash workers; bounded
  CPU buffers with explicit MLX evaluation. Every received chunk was hashed.
- Timed transfers include read/write, checksum verification, disk flush, atomic
  publication and receipt persistence. Fresh-helper trials also include helper
  startup and group initialization. Persistent helpers are deliberately retained;
  shutdown is outside these timings, unlike the earlier one-shot prototype.
- No cache purges, compression, or source-weight modifications. APFS cloning
  provides an independent source snapshot without another full physical copy.

## Repeated measurements

Three measurements per row after one warmup, measured through `TransferPool`:

| Case | Median | Range |
|---|---:|---:|
| Verified source manifest, persistent RDMA helpers, fresh destination | 0.940 s | 0.931–1.001 s |
| Verified source manifest, fresh RDMA helpers, fresh destination | 1.509 s | 1.505–1.519 s |
| Verified source manifest, SSH/rsync, fresh destination | 30.052 s | 29.898–30.146 s |
| Already verified destination, persistent helper, no payload copy | 0.00185 s | 0.00175–0.00203 s |

The SSH measurements use the existing default cipher; they do not represent the
fastest possible SSH configuration. Rates are logical payload/copy-progress
rates, not cable-level wire counters. The earlier **75.17 Gb/s** result was a
separate memory-to-memory benchmark and is not the verified file-delivery rate.

An additional run through `ModelStager.prepare`, including source resolution,
manifest/receipt validation and staging locking, measured 1.512 s with fresh
helpers followed by **0.937, 0.922, 0.932 s** using the same helper PIDs (median
**0.932 s**). Source hashes record the revision used for these timing runs;
the subsequent fallback restrictions were checked separately below. Public
HTTP authentication/probing and application UI latency are outside this timing.

**Initial preparation is not free:** creating the private source snapshot and
its full-file/chunk manifest took **2.876 s** in a separate measurement. That
cost precedes delivery on the first preparation and is excluded from the cached
manifest rows above. Do not describe 0.94 s as the total first-ever preparation
time, or compare it directly with a cold full-checkpoint preparation without
accounting for this cost. A destination cache hit performs no transfer at all.

## Correctness and failure handling

- Cancelled an active transfer after measured received bytes: operation failed,
  partial snapshot remained, no final snapshot was published. Retry reused
  **1,409,286,146 bytes** after checking the actual partial contents.
- Terminated the owned worker's SSH control connection during delivery: both
  transfer helpers were disposed, no final snapshot was published. A new pair
  resumed successfully, reusing **1,308,622,850 bytes**.
- Independent whole-file disk readback after resumed publication matched the
  source SHA-256 above.
- Deliberately supplied an incorrect final-chunk digest to the receiver while
  sending the valid source: checksum failure prevented publication.
- Deliberately configured an unavailable RDMA interface: `auto` reported its
  connection-failure reason, then successfully delivered and verified using
  SSH/rsync. Explicit `jaccl_rdma` never silently falls back.
- Fresh and repeated transfers used separate groups and ephemeral ports from
  inference. Cancellation and failures did not stop unrelated processes.

Automated checks cover sealed receipt reuse without weight reads, source
independence, same-size corruption with restored mtime, ordered chunk/tail
validation, partial reuse, path traversal and symlink rejection, unexpected
unverified files, mutation before publication, identity mismatch, cancellation,
API authentication and idempotency, fallback and persistent-helper cleanup.
Existing server, launcher, continuous-batching and generation regressions remain
part of the full validation command. The initial implementation passed
**1,942 tests, with 23 skipped**. Black, isort,
autoflake and `git diff --check` also passed.

```sh
python -m pytest \
  mlx_vlm/tests/test_checkpoint_transfer.py \
  mlx_vlm/tests/test_server.py \
  mlx_vlm/tests/test_server_distributed.py \
  mlx_vlm/tests/test_server_distributed_launch.py \
  mlx_vlm/tests/test_server_cluster.py \
  mlx_vlm/tests/test_generate.py \
  -q -n 3 --disable-warnings \
  -k 'not test_diarization_decodes_and_resamples_wav'
```

The WAV test exclusion is the previously reproduced local audio-backend issue,
unrelated to checkpoint delivery. Existing skips include optional tests; CPU
ring coverage is not presented as evidence for TB5 performance.

## Limits and artifacts

The coordinator-to-third-Mac direct JACCL check failed (`Recv failed with
errno=2`, consistent with the earlier unavailable second direct pair).
**Three-Mac RDMA delivery is not validated.** Additional workers are scheduled
as separate direct two-Mac sessions, not a three-rank transfer collective.

Physical cable removal was not tested because the links carry other services.
Control-connection loss and process cancellation are not substitutes for
physical cable-loss validation. No 610 GB checkpoint was transferred; bounded
queues and streaming hashes avoid checkpoint-sized allocations, but this test
does not establish cold-disk performance at that size.

## Fallback restrictions after review

Connection setup no longer converts arbitrary exceptions to transport
unavailability. Helpers explicitly classify unavailable JACCL backends/devices
and recognized native connection errors; unknown errors, protocol failures,
challenge mismatches and helper startup failures remain fatal. Only a timeout
after both helpers acknowledge native initialization permits automatic fallback.
An unavailable local network address also permits fallback, while socket
permission/resource errors do not.

APFS cloning falls back to a streamed copy only for cross-filesystem or
unsupported-operation errors (`EXDEV`, `ENOTSUP`/`EOPNOTSUPP`, `ENOSYS`). Permission,
disk-full, I/O and invalid-argument failures abort preparation without copying.

A separate two-Mac JACCL check transferred a 2 MiB fixture using the final
fallback handling, then verified all of the following:

- An unavailable configured RDMA device triggered an observable `auto` fallback
  to a verified SSH/rsync transfer.
- The same failure with explicit `jaccl_rdma` failed without fallback.
- An injected incompatible backend API call failed with its original diagnostic,
  rather than switching to SSH/rsync.

These are correctness checks, not additional throughput measurements. Their
events and exact source hashes are in `fallback-review/results.json` under the
artifact root below. Automated regressions also cover clone errors, startup
timeouts, typed helper errors, connection timeouts, cancellation and unknown
backend failures.

The full regression command above passed **1,987 tests, with 23 skipped**, after
these restrictions. The same unrelated WAV test exclusion remains in place;
Black, isort, autoflake and whitespace checks also passed.

## Evidence files

Detailed local evidence: `/Users/Shared/mlx-checkpoint-persistent-20261112/`.
`speed-results.json` contains repeated timings, progress and helper PIDs;
`fault-results.json` contains interruption/resume, readback, corruption, fallback
and third-Mac results; `final-stager-results.json` records the staging-path trials
and measured source-file hashes. Small test-only helper scripts and configuration
remain alongside those reports for reproduction. These paths are validation
artifacts, not defaults or dependencies of the server implementation.
