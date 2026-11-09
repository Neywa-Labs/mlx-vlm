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

## Validation

Run the small-model architecture and checkpoint tests with:

```bash
python -m pytest -q mlx_vlm/tests/test_models.py \
  mlx_vlm/tests/test_processors.py -k ricursive
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
