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

Verification uses the normal target layers, decode-equivalent short-block
projections, and shared cache transactions. Rejected tokens are rolled back in
both full-attention and rotating caches. With multiple active requests, Ripple
uses ordinary target decoding: exact batched verification costs more than the
accepted tokens save. When one request remains, DFlash2 resumes with its full
pending context. The drafter remains resident for this transition. Explicitly
setting `model.language_model.dflash_batch_size_limit = None` enables batched
speculation for experiments; it uses uniform acceptance to preserve the attention
arithmetic of normal batched decoding.
Chunked prefill retains all hidden features needed by the drafter.
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
All 10 forced/automatic runs in the batch-policy benchmark (3,328 generated tokens)
matched their normal-decoding references token for token. A full-model test with
32- and 128-token requests also matched all 160 tokens: the shared batch used
ordinary decoding, then the survivor resumed DFlash2 with its pending context.

Apple M5 Max (128 GiB), MLX 0.32.4, affine 4-bit target (group size 64), BF16
official drafter, activations, and KV caches. Medians of two runs with 2,048 prompt
tokens and 128 generated tokens per request. Greedy sampling, a 2,048-token shared
prefill budget, and adaptive 3–5-token draft blocks. Equal-length synthetic prompts;
EOS stopping, prefix caching, and logprobs disabled. Each batch setting uses
normal/forced/automatic/automatic/forced/normal order in one process; batch 1 omits
the forced runs because it uses the same speculative path. The drafter is unloaded
for normal runs. Throughput is aggregate decode tokens per second after the first
token; peak memory is MLX allocation in decimal GB.

| Batch | Normal tok/s | Forced DFlash2 tok/s | Automatic tok/s | Gain over forced | Peak GB (normal / automatic) |
|---:|---:|---:|---:|---:|---:|
| 1 | 12.05 | — | 16.48 | same path | 38.25 / 46.51 |
| 2 | 22.72 | 13.99 | 22.75 | 1.63x | 39.04 / 47.41 |
| 4 | 29.45 | 14.51 | 29.02 | 2.00x | 39.90 / 48.48 |

The automatic path uses ordinary target decoding for multiple active requests and
DFlash2 for a single request. It retains pending context so speculation can resume
when the batch shrinks. The drafter stays resident; peak memory at batches 2 and 4
is unchanged from forced speculation. Total request time, including prefill, falls
from 32.26 to 25.65 seconds at batch 2 and from 67.32 to 49.98 seconds at batch 4.
These measurements validate greedy token parity and throughput, not quantization
quality or maximum-context behavior.

An earlier two-run single-request context sweep measured 12.37 → 19.77 tok/s at
512 prompt tokens and 11.98 → 14.51 tok/s at 8,192 prompt tokens. Those use the same
unchanged single-request speculative path but were measured in a separate session.
At 8,192 tokens, total time was 44.16 seconds normally and 42.42 seconds with DFlash2;
prefill limits the end-to-end benefit for long prompts.
