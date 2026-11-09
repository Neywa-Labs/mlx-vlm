# Ling 3.1 Flash

Text generation support for the `bailing_hybrid` architecture released in
[`inclusionAI/Ling-3.1-flash`](https://huggingface.co/inclusionAI/Ling-3.1-flash).
The decoder combines Kimi Delta Attention with gated multi-head latent
attention, shared and routed experts, and layer-specific SwiGLU clipping.

Convert the official block-FP8 checkpoint with the standard converter:

```sh
python -m mlx_vlm.convert \
  --hf-path inclusionAI/Ling-3.1-flash \
  --mlx-path ./Ling-3.1-flash-4bit \
  -q --q-bits 4 --q-group-size 64

python -m mlx_vlm.generate \
  --model ./Ling-3.1-flash-4bit \
  --trust-remote-code \
  --prompt "Explain why the sky is blue." --max-tokens 128
```

The source checkpoint occupies approximately 574 GB. Conversion requires
additional disk space for the MLX weights. The sanitizer decodes 128-by-128
FP8 blocks, stacks routed experts, converts convolution layouts, and splits
MLA projections for latent caching. Routers and recurrent gate parameters
retain their source precision.

The 4-bit conversion with group size 64 occupies approximately 320.5 GB
(4.547 effective bits per weight). Normal resident inference needs that much
available unified memory plus headroom for activations and caches. Run these
commands from a checkout containing the `bailing_hybrid` port.

Ordinary generation supports chunked prefill, incremental decoding, and padded
batches through the shared cache implementations. Use the default unquantized
KV cache. The bundled NEXTN prediction layer is excluded from target loading;
MTP speculative decoding and distributed sharding are not implemented by this
port. The downloaded source checkpoint retains the NEXTN tensors.

Validation covers synthetic generation, checkpoint conversion/reloading,
padding, chunked and incremental cache parity, and independent KDA/MLA
numerical references. Released FP8 weights were checked against NumPy
references, and the complete 4-bit target passed a text-generation smoke test
through the public generation API using per-layer weight eviction. Resident
inference performance and long-context quality have not been benchmarked.
