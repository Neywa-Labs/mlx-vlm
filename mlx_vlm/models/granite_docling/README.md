# Granite for Docling

Support for the `granite_docling` architecture
(`GraniteDoclingForConditionalGeneration`). The model converts a page image to
DocLang markup: text, layout boxes, tables, and pictures.

The model has a SigLIP-style encoder for 512x512 tiles, a pixel-shuffle
connector with fixed 2D positions, and a Granite text decoder. The connector
also makes DeepStack features, which are added to the image tokens after the
first decoder layers.

Supported checkpoints:

- [`ibm-granite/granite-for-docling-2b`](https://huggingface.co/ibm-granite/granite-for-docling-2b)
- `ibm-granite/granite-for-docling-500m` (fine connector path and density router)

The local Granite Docling processor uses Pillow for page tiling and resizing.

## Usage

```sh
python -m mlx_vlm.convert \
  --hf-path ibm-granite/granite-for-docling-2b \
  --mlx-path ./granite-for-docling-2b-8bit \
  -q --q-bits 8

python -m mlx_vlm.generate \
  --model ./granite-for-docling-2b-8bit \
  --image page.png --prompt "<doclang>" \
  --max-tokens 4096 --temperature 0
```

Use `<doclang>` to convert the full page. To select tasks, give them in
brackets, for example `<doclang> [<table>, <layout>]`.

## Fine connector path

The 500M checkpoint has a fine connector path, which gives four times as many
image tokens per tile, for dense pages. Ask the processor for it:

```sh
python -m mlx_vlm.generate --model ./granite-for-docling-500m-mlx \
  --image page.png --prompt "<doclang>" --max-tokens 4096 --temperature 0 \
  --processor-kwargs '{"fine_route": true}'
```

In Python, pass `fine_route=True` to `generate`. For `batch_generate`, set
`processor.image_processor.fine_route = True` before the call.

The 500M checkpoint also has a density router, which tells from the vision
encoder output if a page needs the fine path. Pass `fine_route="auto"`
(`--processor-kwargs '{"fine_route": "auto"}'`) to let it choose for each page:
`generate` and `stream_generate` run the vision encoder once, ask the router
and use the image tokens of the path it chooses. `batch_generate` does not
support `"auto"`; there, `Model.predict_fine_route(pixel_values,
pixel_attention_mask)` returns the router's choice for each page. The 2B
checkpoint has no fine path and no router.

## Speculative decoding

Checkpoints with `mtp.layers.*` tensors have multi-token prediction heads.
They draft up to three tokens for each step of the model. The model verifies
the draft tokens, so greedy output does not change, except when two tokens
are equal within the precision of the weights. Give the original checkpoint as
the draft model:

```sh
python -m mlx_vlm.generate --model ./granite-for-docling-2b-8bit \
  --draft-model ibm-granite/granite-for-docling-2b \
  --image page.png --prompt "<doclang>" --max-tokens 4096 --temperature 0
```

`mlx_vlm.convert --mtp` also writes the heads to `<mlx-path>-mtp`, which you
can give as the draft model.

The gain is largest for greedy decoding of one page at a time. In a batch,
each page accepts its own number of draft tokens, but the gain is smaller.
With sampling, the model accepts fewer draft tokens. The heads of the 500M
checkpoint accept fewer draft tokens, so they give little or no gain.

## Limits

- No tool calling: the chat template has no tool format.
