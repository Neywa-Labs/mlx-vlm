# Ripple

Native MLX support for the text-only
[Ricursive Ripple-65B-Preview](https://huggingface.co/ricursiveAI/Ripple-65B-Preview)
architecture (`model_type: ricursive`). The implementation follows the
[upstream model code](https://huggingface.co/ricursiveAI/Ripple-65B-Preview/blob/da94cc013a8fda04a502f0a185083c7099ea06c1/modeling_ricursive.py).

Ripple uses four RMS norms per decoder block, per-head Q/K normalization,
elementwise gated attention, clipped SwiGLU, and value residuals derived from
the scaled input embeddings. Sliding-window layers apply RoPE and use rotating
KV caches; every fifth layer uses global attention without positional embeddings.
Both scaled and folded embedding checkpoints are supported through `embed_scale`.

## Convert

The BF16 source weights occupy approximately 130 GB. A 4-bit affine conversion
uses approximately 36 GB of weight storage, plus runtime memory for activations
and KV caches. Use a Mac with sufficient unified memory and disk space.

```bash
python -m mlx_vlm.convert \
  --hf-path ricursiveAI/Ripple-65B-Preview \
  --mlx-path Ripple-65B-Preview-4bit \
  -q --q-bits 4 --q-group-size 64
```

## Generate

```bash
python -m mlx_vlm.generate \
  --model Ripple-65B-Preview-4bit \
  --prompt "Write a Python function that merges overlapping intervals." \
  --max-tokens 512 --temperature 0
```

The original tokenizer and chat template are used directly. Native loading
does not require executing the checkpoint's Python code. The checkpoint's
`<|im_end|>` and `<|endoftext|>` stop tokens are both honored.

## Speculative decoding

The official
[Ripple DFlash2 drafter](https://huggingface.co/ricursiveAI/Ripple-65B-Preview-DFlash2)
works with the same target checkpoint through the standard generation and server
interfaces:

```bash
python -m mlx_vlm.generate \
  --model Ripple-65B-Preview-4bit \
  --draft-model ricursiveAI/Ripple-65B-Preview-DFlash2 \
  --prompt "Write a Python function that merges overlapping intervals." \
  --max-tokens 512 --temperature 0
```

The separate BF16 drafter adds approximately 8.08 GB of weights and shares the
target's unscaled token embedding and output head. It consumes the outputs of
target layers 1, 20, 39, 58, and 77. Its first four layers use sliding-window RoPE
attention; its final layer uses global attention without RoPE. Four-tap dynamic
convolutions and the candidate-path selector use the shared DFlash2 implementation.

Verification uses the normal target layers and decode-equivalent short-block
projections. Following the Mistral Large 4 verifier, affine projections share
weights across draft positions while preserving native small-batch reductions.
Full-attention caches retain per-request append positions, and sliding caches
retain per-request ring cursors. Transactions discard rejected writes without
moving accepted keys, so each request can accept a different draft length.
Full and sliding verification group all block queries into one attention call
per layer. Cache updates and rollback operate on complete blocks, including
empty, partially filled, and wrapped sliding windows.
The drafter shares dense projections across requests and keeps independent
attention caches. Chunked prefill retains all hidden features needed by the drafter.
The default adaptive runtime block contains three to five tokens, including
the target anchor; `--draft-block-size` can override the ceiling up to the
checkpoint's trained block size of 16. Speed depends on acceptance and batch size.

## Validation

Run the small-model architecture and checkpoint tests with:

```bash
python -m pytest -q mlx_vlm/tests/test_models.py \
  mlx_vlm/tests/test_processors.py mlx_vlm/tests/test_speculative.py -k ricursive
```

These cover sliding-window rollover with single-token and chunked decoding,
embedding inputs, value residuals, FFN clipping, repeated weight sanitization,
tied embeddings, quantized checkpoint reload, and tokenizer/stop-token loading.
Float32 logits were also compared against the pinned upstream PyTorch model
with nonuniform norm weights, nonzero residual coefficients, and enabled or
disabled gating/value residuals. Maximum error was below `7e-7` with MLX TF32
disabled.

The full 65B checkpoint was also converted to 4-bit affine weights (group size
64; 36.44 GB) and validated on an Apple M5 Max with MLX 0.32.4. Arithmetic and
Python-generation smoke tests returned correct responses and stopped at EOS.
A 2,331-token retrieval prompt returned the correct code from the start of the
context, with identical output using full prefill and 256-token chunks. Peak
memory for that test was 38.42 GB.

On a separate 1,100-token cache test, BF16 execution produced a maximum logit
difference of `1.0` between full and chunked prefill, while the next token agreed
and the output distributions differed by less than `4e-6` in total variation.
Repeating the test with float32 computation on the same quantized weights reduced
the maximum logit difference to `2.33e-4` for 256-token chunks and `8.40e-5` for
511-token chunks. These results are consistent with BF16 rounding rather than a
cache-position error. They are smoke and cache checks, not a quantization-quality
benchmark or validation of the advertised maximum context length.

### DFlash2 validation and performance

The official BF16 drafter was loaded strictly from revision
`35609b3be3dd08a822b3ba1257c1ea5dd6f87632`. Arithmetic, Python generation, and a
2,331-token retrieval prompt matched the same 4-bit target without speculation.
After the batching changes, all 8 speculative benchmark runs (2,048 generated
tokens) matched their normal-decoding references token for token. A separate
full-model batch-shrink test matched all 160 tokens across 32- and 128-token
requests and confirmed active drafting at both batch sizes 2 and 1. Two additional
64-token runs starting at context lengths 1,022 and 8,190 matched normal decoding
while crossing attention dispatch boundaries.

The shared model, processor, speculative, generation, and cache suites passed
3,262 tests, with 2 skips and 126 passing subtests. Tests cover native affine
projection parity, independent acceptance, sliding-window rollback and rollover,
cache serialization, padded prefixes, filtering, extension, and causal block
views before and after sliding-window rollover.

Apple M5 Max (128 GiB), MLX 0.32.4, affine 4-bit target (group size 64), BF16 official
drafter, activations, and KV caches. Medians of two runs, 128 generated tokens per
request, greedy sampling, a 2,048-token shared prefill budget, and adaptive 3–5-token
draft blocks. Equal-length synthetic prompts; EOS stopping, prefix caching, and
logprobs disabled. Each setting uses normal/speculative/speculative/normal order in
one process, with the drafter unloaded for normal runs. Throughput is aggregate
decode speed after the first token; peak memory is maximum MLX allocation in decimal GB.

| Batch | Context | Normal tok/s | DFlash2 tok/s | Speedup | Peak GB (normal / DFlash2) |
|---:|---:|---:|---:|---:|---:|
| 1 | 2,048 | 12.51 | 16.98 | 1.36x | 38.25 / 46.51 |
| 1 | 8,192 | 11.88 | 14.45 | 1.22x | 39.01 / 47.59 |
| 2 | 2,048 | 24.58 | 27.24 | 1.11x | 39.04 / 47.41 |
| 4 | 2,048 | 32.18 | 24.98 | 0.78x | 39.90 / 48.69 |

DFlash2 improves single-request and batch-2 decode throughput in this workload.
Batch 4 remains slower than normal decoding; speculative decoding stays active
when a drafter is supplied, with independent acceptance for each request.
Including prefill, median request time is 23.90 / 22.78 seconds at
batch 2 and 46.72 / 53.97 seconds at batch 4 (normal / DFlash2).
At 8,192 context tokens, the corresponding single-request times are
45.28 / 43.54 seconds. Decode throughput alone does not describe end-to-end latency.
These are greedy inference parity checks and configuration-specific measurements,
not a quantization-quality evaluation or validation of the advertised maximum context length.
