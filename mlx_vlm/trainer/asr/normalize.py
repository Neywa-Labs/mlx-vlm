"""Normalize loaded ASR models to one training interface.

Speech libraries expose inference interfaces that differ per architecture.
normalize_model inspects a loaded model and attaches the interface the
trainer relies on, so nothing else in the trainer knows an architecture:

- training_objective: "sft" (next-token cross-entropy on the
  transcript) or "ctc".
- prepare_example(audio, text, language): one utterance as a mapping whose
  model inputs are named after forward_logits arguments (the preprocessed
  audio is always input_features), plus input_ids and
  completion_mask for sft or labels for ctc.
- forward_logits(**inputs): differentiable logits for one example,
  [1, tokens, vocab] for sft or [1, frames, vocab] for ctc.
- blank_id: the CTC blank class (ctc only).

A model that already defines this interface is returned unchanged. Each family
below is the whole description of one architecture.
"""

from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def load_audio(audio, sample_rate: int = 16000) -> mx.array:
    """Return a mono waveform, loading paths at sample_rate."""
    if isinstance(audio, (str, Path)):
        from mlx_audio.stt.utils import load_audio as load

        return load(str(audio), sr=sample_rate)
    return audio if isinstance(audio, mx.array) else mx.array(audio)


def sft_example(prompt, transcript, **inputs) -> dict:
    """Join prompt and transcript token ids; only the transcript is supervised."""
    prompt = np.asarray(prompt, dtype=np.int32).reshape(-1)
    transcript = np.asarray(transcript, dtype=np.int32).reshape(-1)
    return {
        "input_ids": np.concatenate([prompt, transcript]),
        "completion_mask": np.concatenate(
            [np.zeros_like(prompt), np.ones_like(transcript)]
        ),
        **inputs,
    }


def end_token_id(model, tokenizer=None) -> int:
    """Return the end-of-transcript id from the tokenizer or model config."""
    for owner in (tokenizer, getattr(model, "config", None)):
        for name in ("eos_token_id", "eot", "eos_id"):
            value = getattr(owner, name, None)
            if value is not None:
                return int(value[0] if isinstance(value, (list, tuple)) else value)
    raise ValueError("The tokenizer and config define no end-of-transcript id.")


def language_name(language: str) -> str:
    """Return the English name for an ISO code ("en" -> "English")."""
    from mlx_audio.stt.models.whisper.tokenizer import LANGUAGES

    return LANGUAGES.get(language.lower(), language).title()


def bpe_encode(text: str, vocabulary) -> list[int]:
    """Encode with a SentencePiece BPE vocabulary listed in merge order.

    Pairs merge in vocabulary order, which reproduces SentencePiece BPE
    segmentation; characters the vocabulary cannot spell are dropped.
    """
    pieces = {piece: index for index, piece in enumerate(vocabulary)}
    if not any(piece != piece.lower() for piece in pieces):
        text = text.lower()
    symbols = list("▁" + "▁".join(text.split()))
    while True:
        rank, index = min(
            (
                (pieces.get(left + right, len(pieces)), index)
                for index, (left, right) in enumerate(zip(symbols, symbols[1:]))
            ),
            default=(len(pieces), 0),
        )
        if rank == len(pieces):
            return [pieces[symbol] for symbol in symbols if symbol in pieces]
        symbols[index : index + 2] = [symbols[index] + symbols[index + 1]]


def inject_audio(embedding, input_ids, audio_embeddings, positions) -> mx.array:
    """Differentiably place [1, N, H] audio embeddings at token positions."""
    hidden = embedding(input_ids)
    positions = mx.array(positions)
    if positions.shape[0] != audio_embeddings.shape[1]:
        raise ValueError("Audio placeholder count does not match encoder output.")
    mask = mx.any(mx.arange(input_ids.shape[1])[:, None] == positions[None, :], axis=1)
    indices = mx.clip(
        mx.cumsum(mask.astype(mx.int32)) - 1, 0, audio_embeddings.shape[1] - 1
    )
    return mx.where(
        mask[None, :, None],
        audio_embeddings[:, indices, :].astype(hidden.dtype),
        hidden,
    )


class Whisper:
    objective = "sft"

    @staticmethod
    def prepare_example(model, audio, text: str, language: str = "en") -> dict:
        from mlx_audio.stt.models.whisper.audio import (
            N_FRAMES,
            log_mel_spectrogram,
            pad_or_trim,
        )

        tokenizer = model.get_tokenizer(language=language)
        mel = log_mel_spectrogram(load_audio(audio), n_mels=model.dims.n_mels)
        if mel.shape[0] > N_FRAMES:
            raise ValueError("Split Whisper training audio into clips of at most 30 s.")
        return sft_example(
            tokenizer.sot_sequence_including_notimestamps,
            [*tokenizer.encode(" " + text.strip()), tokenizer.eot],
            input_features=pad_or_trim(mel, N_FRAMES, axis=0)[None],
        )

    @staticmethod
    def forward_logits(model, input_ids, input_features):
        dtype = model.encoder.conv1.weight.dtype
        return model(input_features.astype(dtype), input_ids)


class Qwen3Asr:
    objective = "sft"

    @staticmethod
    def prepare_example(model, audio, text: str, language: str = "en") -> dict:
        features, mask, count = model._preprocess_audio(
            str(audio) if isinstance(audio, Path) else audio
        )
        prompt = model._build_prompt(count, language_name(language))
        transcript = model._tokenizer.encode(text, add_special_tokens=False)
        return sft_example(
            np.asarray(prompt),
            [*transcript, model._tokenizer.eos_token_id],
            input_features=features,
            feature_attention_mask=mask,
        )

    @staticmethod
    def forward_logits(model, input_ids, input_features, feature_attention_mask=None):
        return model(
            input_ids,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
        )


class Moonshine:
    objective = "sft"

    @staticmethod
    def prepare_example(model, audio, text: str, language: str = "en") -> dict:
        transcript = model._tokenizer.encode(text, add_special_tokens=False)
        return sft_example(
            [model.config.decoder_start_token_id],
            [*transcript, end_token_id(model, model._tokenizer)],
            input_features=load_audio(audio, model.sample_rate),
        )

    @staticmethod
    def forward_logits(model, input_ids, input_features):
        hidden, _ = model.decoder(input_ids, model.encoder(input_features))
        return model._get_logits(hidden)


class Canary:
    objective = "sft"

    @staticmethod
    def prepare_example(model, audio, text: str, language: str = "en") -> dict:
        tokenizer = model._tokenizer
        prompt = tokenizer.build_prompt_tokens(
            source_lang=language, target_lang=language, use_pnc=True
        )
        return sft_example(
            prompt,
            [*tokenizer.encode(text), end_token_id(model, tokenizer)],
            input_features=model._preprocess_audio(load_audio(audio)),
        )

    @staticmethod
    def forward_logits(model, input_ids, input_features):
        encoded, lengths = model.encoder(
            input_features, mx.array([input_features.shape[1]])
        )
        mask = mx.arange(encoded.shape[1])[None] < lengths[:, None]
        return model.decoder(input_ids, encoded, encoder_mask=mask)[0]


class CohereAsr:
    objective = "sft"

    @staticmethod
    def prepare_example(model, audio, text: str, language: str = "en") -> dict:
        tokenizer = model._tokenizer
        model._validate_language(language)
        features, lengths = model.audio_frontend([load_audio(audio)])
        return sft_example(
            tokenizer.build_prompt_tokens(language),
            [
                *tokenizer.encode(text, add_special_tokens=False),
                end_token_id(model, tokenizer),
            ],
            input_features=features,
            feature_lengths=lengths,
        )

    @staticmethod
    def forward_logits(model, input_ids, input_features, feature_lengths):
        encoded, lengths = model.encoder(input_features, feature_lengths)
        if model.encoder_decoder_proj is not None:
            encoded = model.encoder_decoder_proj(encoded)
        mask = mx.arange(encoded.shape[1])[None] < lengths[:, None]
        hidden, _ = model.transf_decoder(input_ids, encoded, encoder_mask=mask)
        return model.log_softmax.mlp.layer0(hidden)


class Fireredasr2:
    objective = "sft"

    @staticmethod
    def _encode_text(model, text: str) -> list:
        """Split into CJK characters and SentencePiece (or character) pieces."""
        import re

        vocabulary = {piece: index for index, piece in enumerate(model._tokenizer)}
        pieces = []
        for part in re.findall(r"[\u4e00-\u9fff]|[^\u4e00-\u9fff]+", text.upper()):
            if len(part) == 1 and "\u4e00" <= part <= "\u9fff":
                pieces.append(part)
            elif getattr(model, "_sp", None) is not None:
                pieces.extend(model._sp.encode(part, out_type=str))
            else:
                pieces.extend(part)
        unknown = set(pieces) - vocabulary.keys()
        if unknown:
            raise ValueError(f"Transcript contains unknown tokens: {unknown}")
        return [vocabulary[piece] for piece in pieces]

    @staticmethod
    def prepare_example(model, audio, text: str, language: str = "en") -> dict:
        features = model._extract_fbank(load_audio(audio))
        if model._cmvn is not None:
            means, istd = model._cmvn
            features = (features - means) * istd
        return sft_example(
            [model.config.sos_id],
            [*Fireredasr2._encode_text(model, text), model.config.eos_id],
            input_features=features[None],
        )

    @staticmethod
    def forward_logits(model, input_ids, input_features):
        encoded = model.encoder(input_features)
        decoder = model.decoder
        hidden = decoder.tgt_word_emb(
            input_ids
        ) * decoder.scale + decoder.positional_encoding(input_ids.shape[1])
        # This decoder takes a keep-mask (nonzero = attend), not an additive one.
        length = input_ids.shape[1]
        mask = mx.expand_dims(mx.tril(mx.ones((length, length))), axis=0)
        for layer in decoder.layer_stack:
            hidden = layer(hidden, encoded, mask)
        return decoder.tgt_word_prj(decoder.layer_norm_out(hidden))


class Glmasr:
    objective = "sft"

    @staticmethod
    def prepare_example(model, audio, text: str, language: str = "en") -> dict:
        tokenizer = model._tokenizer
        features = model._preprocess_audio(audio)
        _, count = model.audio_encoder(features)
        prefix = tokenizer.encode("<|user|>\n<|begin_of_audio|>")
        suffix = tokenizer.encode(
            "<|end_of_audio|>\nPlease transcribe this audio into text<|assistant|>\n"
        )
        return sft_example(
            prefix + [0] * count + suffix,
            [
                *tokenizer.encode(text, add_special_tokens=False),
                end_token_id(model, tokenizer),
            ],
            input_features=features,
            audio_offsets=[[len(prefix)]],
            audio_length=[[count]],
        )

    @staticmethod
    def forward_logits(model, input_ids, input_features, audio_offsets, audio_length):
        # The merge indexes with Python lists of offsets.
        return model(
            input_ids,
            audios=input_features,
            audio_offsets=np.asarray(audio_offsets).tolist(),
            audio_length=np.asarray(audio_length).tolist(),
        )


class Qwen2Audio:
    objective = "sft"

    @staticmethod
    def prepare_example(model, audio, text: str, language: str = "en") -> dict:
        tokenizer = model._processor.tokenizer
        features, count = model._extract_features(load_audio(audio))
        return sft_example(
            np.asarray(model._build_prompt([count])),
            [
                *tokenizer.encode(text, add_special_tokens=False),
                end_token_id(model, tokenizer),
            ],
            input_features=features,
        )

    @staticmethod
    def forward_logits(model, input_ids, input_features):
        audio = model.get_audio_features(input_features)
        positions = np.flatnonzero(np.asarray(input_ids[0]) == model.audio_token_id)
        embeddings = inject_audio(
            model.language_model.model.embed_tokens, input_ids, audio, positions
        )
        return model(input_ids, input_embeddings=embeddings)


class GraniteSpeech:
    objective = "sft"

    @staticmethod
    def prepare_example(model, audio, text: str, language: str = "en") -> dict:
        tokenizer = model._tokenizer
        features, count = model._extract_features(load_audio(audio))
        return sft_example(
            np.asarray(model._build_prompt(count)),
            [
                *tokenizer.encode(text, add_special_tokens=False),
                end_token_id(model, tokenizer),
            ],
            input_features=features,
        )

    @staticmethod
    def forward_logits(model, input_ids, input_features):
        audio = model.get_audio_features(input_features)
        positions = np.flatnonzero(np.asarray(input_ids[0]) == model.audio_token_id)
        embeddings = inject_audio(
            model.language_model.model.embed_tokens, input_ids, audio, positions
        )
        return model(input_ids, input_embeddings=embeddings)


class FunAsrNano:
    objective = "sft"

    @staticmethod
    def prepare_example(model, audio, text: str, language: str = "en") -> dict:
        from mlx_audio.stt.models.fun_asr_nano.audio import prepare_audio

        tokenizer = model._tokenizer
        features, lengths, count = prepare_audio(audio, model.config.frontend_conf)
        prompt, offset = model._build_prompt_ids(
            count, language=language_name(language), hotwords=None, itn=True
        )
        return sft_example(
            prompt,
            [
                *tokenizer.encode(text, add_special_tokens=False),
                end_token_id(model, tokenizer),
            ],
            input_features=features,
            feature_lengths=lengths,
            audio_positions=np.arange(offset, offset + count, dtype=np.int32),
        )

    @staticmethod
    def forward_logits(
        model, input_ids, input_features, feature_lengths, audio_positions
    ):
        encoded, lengths = model.audio_encoder(input_features, feature_lengths)
        audio, _ = model.audio_adaptor(encoded, lengths)
        embeddings = inject_audio(
            model.llm.model.embed_tokens, input_ids, audio, audio_positions
        )
        return model(input_ids, input_embeddings=embeddings)


class Voxtral:
    objective = "sft"

    @staticmethod
    def prepare_example(model, audio, text: str, language: str = "en") -> dict:
        tokenizer = model._processor.tokenizer
        processed = model._processor.apply_transcription_request(
            language=language,
            audio=np.asarray(load_audio(audio)),
            model_id=model.config.model_repo,
            sampling_rate=16_000,
            format="wav",
        )
        return sft_example(
            np.asarray(processed["input_ids"]),
            [
                *tokenizer.encode(text, add_special_tokens=False),
                end_token_id(model, tokenizer),
            ],
            input_features=np.asarray(processed["input_features"]).transpose(0, 2, 1),
        )

    @staticmethod
    def forward_logits(model, input_ids, input_features):
        audio = model.get_audio_embeds(input_features)[None]
        positions = np.flatnonzero(
            np.asarray(input_ids[0]) == model.config.audio_token_id
        )
        embeddings = inject_audio(
            model.language_model.model.embed_tokens, input_ids, audio, positions
        )
        return model.language_model(input_embeddings=embeddings)


class MossTranscribeDiarize:
    objective = "sft"

    @staticmethod
    def prepare_example(model, audio, text: str, language: str = "en") -> dict:
        tokenizer = model._tokenizer
        features, lengths, mapping, counts, _ = model._preprocess_audio(audio)
        prompt = model._build_prompt(
            sum(counts), prompt="Transcribe the speech into plain text."
        )
        return sft_example(
            np.asarray(prompt),
            [
                *tokenizer.encode(text, add_special_tokens=False),
                end_token_id(model, tokenizer),
            ],
            input_features=features,
            audio_feature_lengths=lengths,
            audio_chunk_mapping=mapping,
        )

    @staticmethod
    def forward_logits(
        model, input_ids, input_features, audio_feature_lengths, audio_chunk_mapping
    ):
        return model(
            input_ids=input_ids,
            input_features=input_features,
            audio_feature_lengths=audio_feature_lengths,
            audio_chunk_mapping=audio_chunk_mapping,
        )


class VibevoiceAsr:
    objective = "sft"

    @staticmethod
    def prepare_example(model, audio, text: str, language: str = "en") -> dict:
        tokenizer = model.tokenizer
        speech = model._preprocess_audio(audio, sampling_rate=24_000)
        frames = model.encode_speech(speech).shape[1]
        # Request plain transcripts rather than the inference JSON schema.
        prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": "<|object_ref_start|>"
                    + "<|box_start|>" * frames
                    + "<|object_ref_end|>\nTranscribe the speech into plain text.",
                }
            ],
            tokenize=True,
            add_generation_prompt=True,
        )
        return sft_example(
            prompt,
            [
                *tokenizer.encode(text, add_special_tokens=False),
                end_token_id(model, tokenizer),
            ],
            input_features=speech,
        )

    @staticmethod
    def forward_logits(model, input_ids, input_features):
        return model(
            input_ids,
            speech_tensors=input_features,
            acoustic_input_mask=input_ids == model._speech_pad_id,
        )


class HiggsAudio3:
    objective = "sft"

    @staticmethod
    def prepare_example(model, audio, text: str, language: str = "en") -> dict:
        tokenizer = model._tokenizer
        audio = load_audio(audio, model.sample_rate)
        # Keep each clip intact; VAD segmentation is an inference policy.
        if audio.shape[0] > model.config.chunk_size_seconds * model.sample_rate:
            raise ValueError(
                "Split training audio into config.chunk_size_seconds clips."
            )
        features = model._extract_features(audio)
        count = model.get_audio_features(features).shape[1]
        prefix = tokenizer.encode(
            "<|im_start|>user\nTranscribe the speech.<|audio_bos|>",
            add_special_tokens=False,
        )
        suffix = tokenizer.encode(
            "<|audio_eos|><|im_end|>\n<|im_start|>assistant\n",
            add_special_tokens=False,
        )
        return sft_example(
            prefix + [model.audio_in_token_idx] * count + suffix,
            [
                *tokenizer.encode(text, add_special_tokens=False),
                end_token_id(model, tokenizer),
            ],
            input_features=features,
        )

    @staticmethod
    def forward_logits(model, input_ids, input_features):
        audio = model.get_audio_features(input_features)
        positions = np.flatnonzero(np.asarray(input_ids[0]) == model.audio_in_token_idx)
        embeddings = inject_audio(model.model.embed_tokens, input_ids, audio, positions)
        return model(input_ids, input_embeddings=embeddings)


class Mms:
    objective = "ctc"

    @staticmethod
    def blank_id(model):
        return model.config.pad_token_id

    @staticmethod
    def prepare_example(model, audio, text: str, language: str = "en") -> dict:
        if model._vocab is None:
            raise RuntimeError("Vocabulary not loaded; load the model with stt.load().")
        audio = load_audio(audio, model.sample_rate)[None, :]
        audio = (audio - mx.mean(audio, axis=-1, keepdims=True)) / (
            mx.std(audio, axis=-1, keepdims=True) + 1e-7
        )
        ids = {token: index for index, token in model._vocab.items()}
        if not any(token != token.lower() for token in ids):
            text = text.lower()
        elif not any(token != token.upper() for token in ids):
            text = text.upper()
        # Characters outside the vocabulary are dropped; "|" separates words.
        labels = [ids[char] for char in "|".join(text.split()) if char in ids]
        return {"input_features": audio, "labels": labels}

    @staticmethod
    def forward_logits(model, input_features):
        return model(input_features)


class GraniteSpeech5Ctc:
    objective = "ctc"

    @staticmethod
    def blank_id(model):
        return model.config.pad_token_id

    @staticmethod
    def prepare_example(model, audio, text: str, language: str = "en") -> dict:
        if model._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded; load the model with stt.load().")
        from mlx_audio.stt.models.granite_speech5_ctc.granite_speech5 import (
            compute_features,
        )

        features = compute_features(
            model._load_audio(audio),
            num_mel_bins=model.config.encoder_config.num_mel_bins,
        )[None]
        labels = model._tokenizer.encode(text, add_special_tokens=False).ids
        return {"input_features": features, "labels": labels}

    @staticmethod
    def forward_logits(model, input_features):
        return model(input_features)


class Sensevoice:
    objective = "ctc"

    @staticmethod
    def blank_id(model):
        return model.blank_id

    @staticmethod
    def prepare_example(model, audio, text: str, language: str = "en") -> dict:
        if model._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded; load the model with stt.load().")
        features = model._extract_features(
            load_audio(audio, model.config.frontend_conf.fs)
        )
        return {
            "input_features": features[None, :, :],
            "labels": model._tokenizer.encode(text),
        }

    @staticmethod
    def forward_logits(model, input_features):
        # The first four frames carry language, event, emotion and ITN tags.
        # ITN mode emits the cased, punctuated text that transcripts contain.
        return model(input_features, use_itn=True)[:, 4:]


class Parakeet:
    @staticmethod
    def head(model):
        """The CTC output layer; transducer-only checkpoints have none."""
        head = getattr(model, "ctc_decoder", None) or getattr(model, "decoder", None)
        return head if type(head).__name__ == "ConvASRDecoder" else None

    @staticmethod
    def objective(model):
        return "ctc" if Parakeet.head(model) is not None else None

    @staticmethod
    def blank_id(model):
        return len(model.vocabulary)

    @staticmethod
    def prepare_example(model, audio, text: str, language: str = "en") -> dict:
        audio = load_audio(audio, model.preprocessor_config.sample_rate)
        return {
            "input_features": model._log_mel_spectrogram(audio),
            "labels": bpe_encode(text, model.vocabulary),
        }

    @staticmethod
    def forward_logits(model, input_features):
        features, _ = model.encoder(input_features)
        return Parakeet.head(model)(features)


# Keyed by the package that defines the model class (...models.<family>).
FAMILIES = {
    "whisper": Whisper,
    "qwen3_asr": Qwen3Asr,
    "moonshine": Moonshine,
    "canary": Canary,
    "cohere_asr": CohereAsr,
    "fireredasr2": Fireredasr2,
    "glmasr": Glmasr,
    "qwen2_audio": Qwen2Audio,
    "granite_speech": GraniteSpeech,
    "fun_asr_nano": FunAsrNano,
    "voxtral": Voxtral,
    "moss_transcribe_diarize": MossTranscribeDiarize,
    "vibevoice_asr": VibevoiceAsr,
    "higgs_audio_3": HiggsAudio3,
    "mms": Mms,
    "granite_speech5_ctc": GraniteSpeech5Ctc,
    "sensevoice": Sensevoice,
    "parakeet": Parakeet,
}


def _family(model):
    """Find the family of the model class or the nearest base class."""
    for cls in type(model).__mro__:
        parts = cls.__module__.split(".")
        if "models" in parts[:-1]:
            family = FAMILIES.get(parts[parts.index("models") + 1])
            if family is not None:
                return family
    return None


def normalize_model(model):
    """Return the trainable module with the standard training interface attached.

    Inference facades are unwrapped to the module that owns the parameters.
    """
    for core in ("_model", "_asr"):
        if not isinstance(model, nn.Module) and hasattr(model, core):
            model = getattr(model, core)
    if getattr(model, "training_objective", None) is not None:
        return model
    family = _family(model)
    objective = getattr(family, "objective", None)
    if callable(objective):
        objective = objective(model)
    if objective is None:
        raise ValueError(
            f"{type(model).__module__}.{type(model).__name__} has no training "
            f"normalization. Supported families: {', '.join(sorted(FAMILIES))}."
        )
    model.training_objective = objective
    model.prepare_example = partial(family.prepare_example, model)
    model.forward_logits = partial(family.forward_logits, model)
    if objective == "ctc":
        model.blank_id = family.blank_id(model)
    return model
