# Omoikane

Text generation support for the `omoikane` architecture (Omoikane-50M-A27M:
48.3M parameters, 27.1M active per token). The decoder mixes Gated DeltaNet
layers with multi-head latent attention (MLA), and uses shared and routed
experts:

- Gated DeltaNet with a short causal convolution on q, k and v, L2-normalized
  q/k (readout scale 1), and a gated RMSNorm on the output. The delta rule runs
  in float32.
- MLA with decoupled RoPE (halves rotated, not interleaved pairs) and a sigmoid
  output gate. The cache holds the latent and the RoPE key. `rope_theta` comes
  from `mla_config` (1e5 for Base, 2e6 for the 4K-context model).
- Dense SwiGLU at `dense_layer_indices`, MoE elsewhere: sigmoid scoring with a
  correction bias, group-limited top-k (excluded groups get `-inf`),
  normalized weights and a shared expert.

The checkpoint stores a multi-token-prediction (MTP) module as the layer after
the last one. Generation does not use it, so `sanitize` drops it.

Hugging Face repos: `Goekdeniz-Guelmez/Omoikane-50M-A27M-Base`, `-Base-4K`,
`-Instruct` and `-Chat`.

```sh
python -m mlx_vlm.generate \
  --model Goekdeniz-Guelmez/Omoikane-50M-A27M-Instruct \
  --trust-remote-code \
  --prompt "Explain why the sky is blue." --max-tokens 128
```

The CLI applies the chat template. For the Base models, which are not trained
on chat, pass a raw prompt through the Python API:

```python
from mlx_vlm import load, stream_generate

model, processor = load("Goekdeniz-Guelmez/Omoikane-50M-A27M-Base", trust_remote_code=True)
for response in stream_generate(model, processor, "The history of Rome", max_tokens=64):
    print(response.text, end="", flush=True)
```

`--trust-remote-code` only lets the tokenizer read the checkpoint's
`config.json`, which names the custom Transformers classes. The MLX model does
not run that code.
