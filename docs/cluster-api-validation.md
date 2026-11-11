# Cluster API validation — 2026-11-11

Validation ran against working-tree changes based on PR #2712 revision
`1d145b0cf0fbb0d30ab80f4357e43c450f095ae7`, before committing the implementation.
Only MLX-VLM was modified. The final transport is SSH/rsync; there are no HTTP
weight-upload endpoints.

## Automated checks

- Server, distributed coordinator, launcher, cluster API, and generation suites:
  **1,893 passed, 23 skipped**. The independently reproduced
  `test_diarization_decodes_and_resamples_wav` failure was excluded: this local
  environment's audio backend rejects the generated WAV with “unsupported file
  format.” This change does not modify the audio decoder.
- Subsequent focused cluster/launcher checks: **59 passed**, including empty
  group identity, launcher/backend mismatch, and failed-group checkpoint cleanup.
- Black, isort, autoflake and `git diff --check` passed.

Lifecycle follow-up after revision `8411e7e5`: **1,907 passed, 23 skipped** with
the same audio-backend test excluded. Added coverage for repeated start on
API-owned and imported CLI groups, retained-continuation rejection, drain timeout
and cancellation, cleanup of only a failed start's new launcher, and preservation
of real node/disposal failures. Both continuous and fallback Responses generation
paths now test `timings.predicted_per_second` against measured token intervals and
regular text events. Hardware checks below were not rerun for this follow-up.

Coverage includes authenticated registration, identity-preserving address
updates, persistence, duplicate devices, selected-only memory and freshness,
operation serialization/idempotency, ownership, node loss, SSE replay/reset,
SSH staging at different paths, failed delivery isolation, concurrent staging,
native prefill callbacks, MiniMax delimiter splits, continuation pause/resume,
expiry of replaced handles, cancellation, and semantic comparison independent
of progress cadence. Existing CPU-ring capacity-four/eight tests remain in the
suite; they are separate from the hardware evidence below.

## Real Macs and JACCL

Tests used isolated ports/processes and matching source on all ranks. Unrelated
Mistral, application, and web services were left running. Hardware was three
M3 Ultra Macs with 512 GiB per device and an existing TB5/JACCL configuration.
The runtime reported MLX `0.32.5.dev20261030+1ba7f96`. No cable was disconnected.

| Test | Result |
|---|---|
| Public `models/prepare`, SSH/rsync, immutable publication | Passed with 16,872,847,972 bytes of checkpoint files; verified on coordinator/local worker and remote worker |
| Released model, two Macs, JACCL | Passed with `mlx-community/GLM-4.7-Flash-4bit`, revision `1454cffb1a21737e162f508e5bc70be9def89276` |
| Two-Mac prefill progress | Measured counts `0 → 221 → 222`, before first output token |
| Two-Mac resumable streaming | Passed after disconnect during decoding and during prefill |
| Capacity four, released weights | Six concurrent requests completed; native rank checks reported no divergence |
| Three Macs, JACCL, native fixture | Passed with a seeded tiny IQuestLoopCoder checkpoint using its normal loader and `Model.shard` |
| Three-Mac prefill progress | Measured counts `0 → 218 → 219`, before first output token |
| Three-Mac resumable streaming | Passed after disconnect during decoding and during prefill |
| Capacity eight, fixture | Ten concurrent requests completed without rank divergence |
| Incompatible released model at three ranks | API rejected GLM's 20 attention heads with `unsupported_member_count` before model loading |
| Membership change while running | Returned structured busy response |
| Drain and model-ready startup | Stop followed by start/load succeeded; readiness required actual worker model loading |
| Controlled worker failure | Terminated only the isolated rank-one worker; distributed requests returned 503 |
| Local inference during distributed failure | Normal local Responses request returned 200 |
| Failure recovery | Failed-group stop preserved recovery requirements; fresh owned workers started; distributed inference returned 200; final stop left no failure marker |

The first SSH transfer attempt hit the smoke test's 300-second limit; its
operation failed explicitly and remained retryable. The successful retry used
the already trusted SSH identity over the Thunderbolt IP and a 900-second test
limit. Checksum checks and atomic publication remained enabled. The public
preparation operation and inference staging now share a transfer lock to avoid
writing the same peer snapshot concurrently.

Hardware checks also caught and fixed two lifecycle issues: an intentional API
stop could be mistaken for an unexpected launcher exit, and a model-ready start
used an empty token-counting request that the existing Responses API rejects.
Failure cleanup explicitly distinguishes unacknowledged cache disposal on dead
ranks from normal synchronized disposal.

## Limits of this evidence

- **Released-weight inference on three Macs was not validated.** The released
  model used here does not support that count. The three-Mac native fixture
  validates coordination/transport, not released-model output quality.
- **Physical cable loss was not tested**, because those links also carried
  unrelated running services. Controlled process failure is not equivalent to
  cable-loss/RDMA timeout validation.
- This is functional integration validation, not a throughput benchmark or a
  numerical-parity study. Compaction and reasoning/tool preservation have
  automated coverage; no released-model tool-calling quality claim is made.
- Automatic SSH launch still consumes the existing hostfile/topology. The API
  reports reconfiguration requirements when selected endpoints or backend do
  not match it; it does not infer RDMA wiring or configure macOS SSH.

Detailed local artifacts are under
`/Users/Shared/mlx-vlm-cluster-api-20261111`: preparation operation results,
`released-two-validation.json`, `released-two-provenance.json`,
`fixture-three-final-validation.json`, `fixture-three-provenance.json`,
`hardware-recovery.json`, SSE event captures, and process logs. Provenance files
record the exact Python source digest used for each hardware run. Test workers
and their coordinator were stopped after validation. The redundant test delivery
cache was removed; verified worker snapshots and the Hub checkpoint were retained.
