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
python -m pytest -q mlx_vlm/tests/test_ricursive.py \
  mlx_vlm/tests/test_models.py -k ricursive
```

These cover sliding-window rollover with single-token and chunked decoding,
embedding inputs, value residuals, FFN clipping, repeated weight sanitization,
tied embeddings, quantized checkpoint reload, and tokenizer/stop-token loading.
Float32 logits were also compared against the pinned upstream PyTorch model
with nonuniform norm weights, nonzero residual coefficients, and enabled or
disabled gating/value residuals. Maximum error was below `7e-7` with MLX TF32
disabled. This small-model parity check does not establish quantized-model
quality or validate the advertised maximum context length.
