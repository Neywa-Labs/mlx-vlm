# ERNIE 5.5 Flash Base

MLX implementation of [`baidu/ERNIE-5.5-Flash-Base`](https://huggingface.co/baidu/ERNIE-5.5-Flash-Base), a text-only base language model with approximately 280B parameters and 14B active per token. It completes text; the released checkpoint has no instruction/chat template.

## Use

Convert the released FP8 checkpoint with the standard converter on a machine with sufficient disk and memory:

```sh
python -m mlx_vlm.convert \
  --hf-path baidu/ERNIE-5.5-Flash-Base \
  --mlx-path ./ERNIE-5.5-Flash-Base-MXFP8
```

The shared FP8 loader converts block-FP8 projections to native MLX MXFP8. The grouped HCA output and fused MLA KV projection are reconstructed for their model-specific layouts. Adding `-q` does not requantize existing MXFP8 projections to 4-bit. The dense vocabulary head is promoted to FP32 once at loading, matching its FP32 projection arithmetic without repeated weight conversion. The original download alone is approximately 288 GB; conversion also needs space for the output checkpoint.

```sh
python -m mlx_vlm.generate \
  --model ./ERNIE-5.5-Flash-Base-MXFP8 \
  --prompt "Large language models are" \
  --max-tokens 128 --temperature 0
```

The ordinary `load`, `generate`, `stream_generate`, and continuous batching paths apply. Use plain completion prompts with the released tokenizer.

### ECHO speculative decoding (experimental)

ECHO uses the prediction weights bundled in the Base checkpoint. Point both arguments at the same checkpoint; a separate draft-model download is unnecessary:

```sh
python -m mlx_vlm.generate \
  --model ./ERNIE-5.5-Flash-Base-MXFP8 \
  --draft-model ./ERNIE-5.5-Flash-Base-MXFP8 \
  --draft-kind mtp --draft-block-size 3 \
  --prompt "Large language models are" \
  --max-tokens 128 --temperature 0
```

The block size includes the target bonus token: `3` proposes two draft tokens. The loader automatically identifies ECHO as an MTP drafter. Standard conversion preserves its auxiliary weights; older conversions that discarded them must be regenerated from a checkpoint containing those tensors.

ECHO is opt-in. The fresh checkpoint check below improves throughput at batch 1; ordinary decoding is faster in both batch-4 cases. Mixed-precision block evaluation can still change greedy outputs relative to ordinary decoding.

ECHO shares the final MLA/MoE decoder layer, token embeddings, and vocabulary projection with the target. Its own projections, normalization, residual head, and expert-routing bias add approximately 67 MB of parameters. It maintains a separate latent KV cache. Verification uses shared cache transactions to retain accepted HCA pooling history and rewind rejected KV entries. Accepted draft cache entries are rebuilt using the verified target hidden states.

Single-request streaming and continuous batching are supported, including left padding, finished requests, and queued arrivals. Batches use the shortest accepted draft prefix across active rows. ECHO currently requires full-prompt prefill, so speculative runs disable chunked prefill and retain the prompt's final residual states for draft initialization. This can increase prefill memory at long contexts. Use a fresh prompt cache: suffix-only prefix reuse does not supply the residual history needed by the drafter. Distributed execution is not implemented.

## Implementation

- Heavily Compressed Attention (HCA), with a local window and learned 128-token compression, per-head query normalization, YaRN, and inverse output RoPE.
- MLA with a latent KV cache and a DSA indexer. Decode gathers selected older tokens plus the recent window. Prefill uses a masked attention reference path; it does not provide a fused sparse-attention kernel.
- Virtual head mixing and gates computed from the query latent.
- Two sigmoid routers, bias-corrected expert selection, learned expert scales, SiTU-GLU, latent output normalization, and a full-width shared expert.
- Manifold-constrained hyper-connections and the learned MultiMax vocabulary transform.
- Released BF16/block-FP8 weight mapping and native MLX quantized checkpoint loading.

The indexer stores its cache in the backbone dtype, without the FP8/FP4 activation and index-cache quantization used by GPU serving engines. Exact token agreement with those kernels is not guaranteed. Tensor/pipeline parallel execution is not integrated.

## Validation

Tiny configurations live in the shared model test catalog. Regressions cover compression boundaries, chunked prefill, sparse decode, causal masking, cache merging/extraction/filtering, padding, state restoration, quantized loading, and continuous batching with late arrivals and cancellation. A tiny checkpoint with the released FP8 naming and scale format exercises strict loading through the shared loader, MLA layout conversion, ECHO projections, and a native MXFP8 save/reload round trip. Component equations are compared with independent NumPy references.

The released checkpoint also passes strict loading and text-generation smoke tests. Independent PyTorch FP32 checks cover a complete tiny model and the released first layer across a compression boundary; these are separate from full-checkpoint generation checks.

ECHO regressions cover forced acceptance/rejection, compressor rollback across boundaries, repeated generation, padded batches, early completion, and auxiliary-weight conversion/loading. An independent PyTorch check of ECHO's equations matched full prefill and token-by-token decode with maximum FP32 logit error below `1e-6` and no top-token disagreements across 38 comparisons.

```sh
python -m pytest mlx_vlm/tests/test_models.py -k 'ernie55 or ernie5_5'
python -m pytest mlx_vlm/tests/test_generate.py -k ernie55
python -m pytest mlx_vlm/tests/test_speculative.py -k ernie_echo
```

## Measured performance

These profiling measurements were collected before rebasing the port onto the current shared cache-transaction implementation. The first table records the initial port before one-time vocabulary-head promotion, with a 2 GiB MLX allocator-cache limit. Profiling found that this limit amplified repeated BF16-to-FP32 head conversion. The ECHO comparison below reports the updated implementation.

Measured on November 10, 2026, on one Apple M3 Ultra with 512 GiB of unified memory, using MLX 0.32.4 and the shared `BatchGenerator`. This checkpoint uses 4-bit affine quantization with group size 64 for routed experts only; remaining tensors retain BF16/FP32 precision. It occupies approximately 174.4 GB on disk. These measurements use a separately prepared checkpoint; the converter command above produces MXFP8 projections and has not been benchmarked here.

Each request generated exactly 32 tokens with greedy sampling and EOS stopping disabled for the throughput measurement. Results are medians of three runs after model/kernel warmup at each batch size. Every run used a fresh KV cache, no prefix reuse or draft model, and a prefill chunk size of 128. Both throughput columns are aggregate rates across the batch; batch 1 is the latency baseline. TTFT is measured in-process. Memory is the maximum MLX allocator peak across the three runs, including model weights, in decimal GB.

| Batch | Input tokens per request | Prefill tok/s | Decode tok/s | TTFT (s) | Peak memory (GB) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 221.6 | 4.28 | 0.80 | 174.30 |
| 1 | 512 | 275.9 | 4.26 | 2.11 | 174.20 |
| 1 | 2,048 | 288.8 | 4.25 | 7.34 | 174.22 |
| 4 | 128 | 440.7 | 17.68 | 1.37 | 175.72 |
| 4 | 512 | 511.9 | 17.59 | 4.22 | 175.76 |
| 4 | 2,048 | 516.8 | 17.48 | 16.06 | 175.86 |

All 18 measured runs completed with the requested token counts, full decode batch occupancy, zero prefix-cache hits, and identical output tokens across repetitions of each case. A separate batch-4 smoke test with 2,304 input tokens and 8 output tokens per request exercised the sparse-indexer path beyond its 2,048 + 128 threshold; its peak was 176.14 GB. The largest tested input was therefore 2,048 tokens for batch 1 and 2,304 for batch 4. These measurements do not establish maximum context capacity or long-context answer quality. Another model remained resident but idle, and system swap usage stayed unchanged.

### ECHO comparison

After the head-promotion fix, a paired comparison on the same machine and checkpoint used two draft tokens, 32 generated tokens per request, batch sizes 1 and 4, and three repetitions per configuration. The model file remains unchanged; the dense head is held in FP32 at runtime. The MLX allocator-cache limit was 2 GiB. Both variants used fresh caches, greedy sampling, and disabled EOS stopping. AR used 128-token prefill chunks; ECHO used full-prompt prefill. Decode timing includes ECHO's draft initialization after the first target token. Rates are aggregate medians.

| Batch | Input tokens per request | AR decode tok/s | ECHO decode tok/s | ECHO / AR | Draft acceptance |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 9.34 | 19.70 | 2.11× | 58.6% |
| 1 | 512 | 9.32 | 17.95 | 1.93× | 53.3% |
| 4 | 128 | 41.19 | 47.63 | 1.16× | 41.2% |
| 4 | 512 | 40.39 | 30.80 | 0.76× | 20.5% |

All 24 runs completed with the requested token counts and full batch occupancy. Outputs repeated exactly within each variant. Six of the ten distinct benchmark request sequences matched AR exactly; four diverged, first at output positions 0, 4, 9, or 18 (zero-based). Two additional 32-token natural-language streaming smoke tests matched exactly. A teacher-forced comparison of block verification with singleton forwards from identical initial caches found 22 top-token differences across 249 proposed positions, including rejected positions. These released mixed-precision results do not establish exact greedy parity. The FP32 reference checks above validate the implementation's equations separately. For batch 4 at 512 input tokens, ordinary decoding remains faster than the tested ECHO configurations.

### Check after rebasing onto current main

A fresh strict load on the rebased implementation repeated the same eight AR/ECHO configurations once each, after warmup. The hardware, checkpoint, allocator limit, prompt tokens, and 32-token output limits were unchanged. These are single-run checks, not three-run medians:

| Batch | Input tokens per request | AR decode tok/s | ECHO decode tok/s | ECHO / AR |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 9.41 | 19.13 | 2.03× |
| 1 | 512 | 9.42 | 17.38 | 1.84× |
| 4 | 128 | 41.24 | 35.84 | 0.87× |
| 4 | 512 | 40.76 | 24.35 | 0.60× |

All cases completed with the requested token counts. Both natural-language streaming smoke tests matched AR for 32 tokens; six of ten benchmark request sequences matched between AR and ECHO. Peak MLX memory reached 180.70 GB, and swap usage stayed unchanged. The current shared runtime has lower batch-4 ECHO throughput than the earlier profiling implementation; that difference has not been isolated. Prefer AR for these batch-4 workloads.

### Profiling findings

The released vocabulary matrix is BF16 with shape `[201216, 4096]`, while every projection uses FP32. Repeated implicit promotion creates a roughly 3.30 GB temporary. In synchronized phase measurements with the original 2 GiB allocator cache, the vocabulary head took about 128 ms per verification, versus 103 ms for the full target backbone at batch 1. Keeping the head in FP32 reduced its phase time to about 7 ms and preserves the tested logits and generated tokens exactly. It adds 1.65 GB of persistent parameter storage relative to BF16. Raising the allocator cache to 8 GiB also mitigates the allocation cost; persistent head promotion still improved ECHO throughput by 18–34% against that better-configured baseline.

After this fix, target verification accounts for most measured decode time. The next opportunities are:

- **Draft length by workload:** one draft token (`--draft-block-size 2`) reached 36.42 aggregate tok/s at batch 4 / 512 input tokens, versus 30.80 for two drafts. At batch 1 / 128, one and two drafts were similar (19.58 versus 19.70 tok/s); three drafts were slower. These settings can change outputs because verification block shapes change mixed-precision arithmetic.
- **Independent row acceptance:** at batch 4 / 512, rows individually accepted 43.2% of proposed tokens, but the uniform acceptance rule retained only 20.5%. Per-row HCA rollback and draft-cache positions are required before removing this constraint. The observed acceptance gap is not a measured speedup for such an implementation.
- **Only project consumed draft logits:** prefill and cache reconditioning use only the final prediction. A prototype that projects only that position improved 512-token-prompt throughput by 4.9% at batch 1 and 8.4% at batch 4, with identical tokens in those tests. This prototype is not part of the runtime implementation.

Phase timings explicitly synchronize lazy MLX work and add 6–15% overhead, depending on the configuration; throughput figures come from separate uninstrumented runs. Representative attention/MoE probes used actual inputs from layers 1 and 42, and are not a complete GPU kernel profile. In the initial Python profile, scheduler functions were a small fraction of wall time. The earlier short-block absorbed-attention experiment had no consistent throughput benefit.

Architecture and checkpoint mapping follow [Baidu's reference implementation](https://github.com/baidu/vLLM-ERNIE/tree/main/src/vllm_ernie/models/ernie5_5). Model weights remain subject to the [upstream model license](https://huggingface.co/baidu/ERNIE-5.5-Flash-Base/blob/main/LICENSE).
