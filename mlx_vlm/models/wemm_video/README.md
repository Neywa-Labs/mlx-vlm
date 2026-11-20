# WeMM-Video

Native MLX inference for Tencent's [WeMM-Video](https://github.com/Tencent/WeMM-Video)
family: Qwen3.5 vision/text, a Whisper encoder, and a windowed Q-Former. Inputs
can contain text, images, audio, and video with its soundtrack; output is text.

The loader recognizes `WeMMVideoForConditionalGeneration`, including dense
checkpoints whose `model_type` is `qwen3_5`, and `wemm_qwen3_5_moe` checkpoints.
All four sizes (2B, 4B, 9B, and 35B-A3B) have been exercised end to end in
BF16 and converted MLX 8-bit and 4-bit form. Loading the native processor does
not require `trust_remote_code=True`.

## Text, images, and audio

```bash
mlx_vlm.generate --model tencent/WeMM-Video-2B \
  --prompt "What is in this picture?" --image picture.jpg --max-tokens 128

mlx_vlm.generate --model tencent/WeMM-Video-2B \
  --prompt "Transcribe the speech." --audio speech.wav --max-tokens 128
```

## Video with audio

FFmpeg must be installed for soundtrack extraction. The processor handles
frame sampling itself to preserve the original video path and align audio
tokens with the timestamps of the sampled frames.

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("tencent/WeMM-Video-2B")
clip = "clip.mp4"
prompt = apply_chat_template(
    processor, model.config,
    "Describe the scene and summarize the speech.",
    num_images=0, video=[clip], enable_thinking=False,
)
result = generate(
    model, processor, prompt, video=[clip], use_audio_in_video=True,
    fps=1.0, min_frames=4, max_frames=16, total_pixels=262144,
    max_tokens=256, temperature=0.0,
)
print(result.text)
```

Omit `use_audio_in_video` for frames alone. A requested soundtrack that cannot
be decoded raises an error. For already-decoded frames, provide an explicit
waveform in a video item: `{"video": frames, "audio": waveform,
"use_audio_in_video": True}`. Waveform arrays are mono at 16 kHz.

`processor(question=..., images=..., audio=..., videos=...)` and
`processor.apply_chat_template(messages, tokenize=True, return_dict=True)`
also prepare inputs directly. Processing accepts one conversation, with
multiple media items, per call. Use `enable_thinking=True` to request reasoning.

## Conversion

```bash
mlx_vlm.convert --hf-path tencent/WeMM-Video-2B \
  --mlx-path WeMM-Video-2B-MLX-4bit -q --q-bits 4 --q-group-size 64
```

Use `--q-bits 8` for an 8-bit export. The normal conversion policy quantizes
the language backbone and keeps the vision and audio encoders in BF16. The
2B 4-bit export therefore contains about 3.86 GB of weights rather than four
bits per parameter across the entire model. Quantized generation works, but
output-quality parity is not established; see the observed limitations below.

## Validation

Tests cover architecture dispatch, dense/MoE forwards, idempotent checkpoint
conversion, independent image/video/audio insertion, timestamp grouping,
mixed audio ordering, preserving video paths for soundtrack extraction, and
processor save/reload without an upstream-code prompt for the MoE checkpoint.
Shared model cases exercise audio segment independence, overlapping windows,
and multiple queries without stored reference fixtures. Full-size 2B audio
embeddings were separately compared
with the reference using the same mel inputs: float32 mean cosine similarity
was 1.0 (maximum absolute difference about `6.2e-4`). BF16 arithmetic differs
across MLX and PyTorch MPS and is not bitwise equivalent.

All four checkpoints and their 8-bit/4-bit exports ran text, image, audio,
video, and video-with-audio smoke checks on an M5 Max with 128 GB memory.
Source weight shards were SHA-256 verified. The checks used Tencent's Apollo 11
media and the same prompts, greedy decoding, reasoning disabled, and a
512-token generation limit. Video sampling used 1 FPS, 4–16 frames, and a
262,144-pixel total budget.

The table counts natural stopping within 512 tokens out of five input modes;
it does **not** score answer correctness.

| Checkpoint revision | BF16 | MLX 8-bit | MLX 4-bit |
| --- | --- | --- | --- |
| [2B](https://huggingface.co/tencent/WeMM-Video-2B/tree/3f4faa80d947769f032b04f105317e1f25844082) | 5/5 | 5/5 | 5/5 |
| [4B](https://huggingface.co/tencent/WeMM-Video-4B/tree/04d1e7374f761a66951a6b47e9a84e320b9d73fe) | 5/5 | 5/5 | 4/5 |
| [9B](https://huggingface.co/tencent/WeMM-Video-9B/tree/33c084c098d97adbf174a1586a01dfb86a553f00) | 5/5 | 5/5 | 4/5 |
| [35B-A3B](https://huggingface.co/tencent/WeMM-Video-35BA3B/tree/4375278c69e03c6ee3af6d34124c2888417cdbe4) | 5/5 | 5/5 | 5/5 |

The 2B 4-bit export answered the arithmetic prompt incorrectly. The 4B 4-bit
video-with-audio response repeated itself, including when rerun with a
2,048-token limit. The 9B 4-bit video-with-audio response was merely longer:
it stopped naturally at 575 tokens with the larger limit. BF16 and 8-bit
avoided the arithmetic and repetition failures on these examples. Transcription and scene-description
errors still occur, and these checks do not establish benchmark accuracy or
full generation parity.

Reference-generation comparisons were performed on 2B only. Matched BF16
greedy runs in Tencent's PyTorch
reference on MPS reproduced the standalone-audio transcript and the first 128
video-with-audio output tokens, including their recognition errors. Increasing
the generation limit from 128 to 512 let all five examples finish in both
implementations. Image/video wording still differs.

## Source and licenses

Architecture and timestamp behavior were ported from
[`tencent/WeMM-Video-2B` at `3f4faa80`](https://huggingface.co/tencent/WeMM-Video-2B/tree/3f4faa80d947769f032b04f105317e1f25844082).
Tencent's source is [Apache-2.0 licensed](https://github.com/Tencent/WeMM-Video/blob/main/LICENSE)
(Copyright 2026 Tencent). The upstream Whisper code credits OpenAI and
Hugging Face (Apache-2.0); its Q-Former derives from Salesforce LAVIS
([BSD-3-Clause](https://github.com/salesforce/LAVIS/blob/main/LICENSE.txt)).
This native implementation reuses mlx-vlm's Qwen3.5 and Whisper blocks.
