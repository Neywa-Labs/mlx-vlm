# ERNIE 5.5 Flash Base

MLX support for [baidu/ERNIE-5.5-Flash-Base](https://huggingface.co/baidu/ERNIE-5.5-Flash-Base). This is a text completion model; use plain prompts rather than chat messages.

## Convert

```sh
python -m mlx_vlm.convert \
  --hf-path baidu/ERNIE-5.5-Flash-Base \
  --mlx-path ./ERNIE-5.5-Flash-Base-MXFP8
```

This converts the released FP8 projections to native MLX MXFP8. The source download is about 288 GB; allow additional space for the output.

Add `-q --q-bits 4` to convert routed experts to affine 4-bit, keeping other eligible projections in MXFP8. Without `-q`, the released FP8 projections stay MXFP8.

## Generate

```sh
python -m mlx_vlm.generate \
  --model ./ERNIE-5.5-Flash-Base-MXFP8 \
  --prompt "Large language models are" \
  --max-tokens 128 --temperature 0
```

## ECHO (experimental)

ECHO uses the draft weights bundled in the same checkpoint:

```sh
python -m mlx_vlm.generate \
  --model ./ERNIE-5.5-Flash-Base-MXFP8 \
  --draft-model ./ERNIE-5.5-Flash-Base-MXFP8 \
  --draft-kind mtp --draft-block-size 3 \
  --prompt "Large language models are" \
  --max-tokens 128 --temperature 0
```

Block size `3` proposes two draft tokens. ECHO requires full-prompt prefill and a fresh cache. Outputs can differ from ordinary decoding, and speedups depend on the workload. Distributed execution is not supported.

[Reference implementation](https://github.com/baidu/vLLM-ERNIE/tree/main/src/vllm_ernie/models/ernie5_5) · [Model license](https://huggingface.co/baidu/ERNIE-5.5-Flash-Base/blob/main/LICENSE)
