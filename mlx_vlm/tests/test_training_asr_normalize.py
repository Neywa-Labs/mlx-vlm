"""Normalization of real mlx-audio architectures (small random models, no downloads)."""

import math
import unittest
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest
from mlx.utils import tree_flatten

pytest.importorskip("mlx_audio")

from mlx_vlm.trainer import SFTTrainConfig, Trainer  # noqa: E402
from mlx_vlm.trainer.asr.normalize import (  # noqa: E402
    FAMILIES,
    bpe_encode,
    normalize_model,
    sft_example,
)


class HelpersTest(unittest.TestCase):
    def test_sft_example_supervises_only_the_transcript(self):
        example = sft_example([5, 6], [1, 2, 3], input_features=mx.zeros((1, 2)))
        self.assertEqual(example["input_ids"].tolist(), [5, 6, 1, 2, 3])
        self.assertEqual(example["completion_mask"].tolist(), [0, 0, 1, 1, 1])

    def test_bpe_encode_merges_in_vocabulary_order(self):
        vocabulary = ["<unk>", "▁", "a", "b", "c", "▁a", "ab", "▁ab"]
        self.assertEqual(bpe_encode("ab c", vocabulary), [7, 1, 4])
        self.assertEqual(bpe_encode("AB?", vocabulary), [7])

    def test_every_family_describes_a_complete_interface(self):
        for name, family in FAMILIES.items():
            with self.subTest(family=name):
                self.assertTrue(hasattr(family, "objective"))
                self.assertTrue(callable(family.prepare_example))
                self.assertTrue(callable(family.forward_logits))

    def test_unknown_models_are_rejected_and_native_ones_pass_through(self):
        with self.assertRaisesRegex(ValueError, "no training normalization"):
            normalize_model(nn.Linear(2, 2))
        native = nn.Linear(2, 2)
        native.training_objective = "ctc"
        self.assertIs(normalize_model(native), native)


VOCAB = 16
CASES = {}


@dataclass
class Case:
    """A small randomly initialized model with one utterance's inputs."""

    model: nn.Module
    inputs: dict
    audio_count: int = 0  # audio placeholder tokens in the prompt (sft)
    audio_id: int = 7
    encoder_prefix: str = "audio_tower."


def case(family):
    def register(build):
        CASES[family] = build
        return build

    return register


@case("qwen3_asr")
def _qwen3_asr():
    from mlx_audio.stt.models.qwen3_asr.config import ModelConfig
    from mlx_audio.stt.models.qwen3_asr.qwen3_asr import Qwen3ASRModel

    model = Qwen3ASRModel(
        ModelConfig(
            audio_token_id=7,
            audio_config=dict(
                num_mel_bins=16,
                encoder_layers=1,
                encoder_attention_heads=2,
                encoder_ffn_dim=32,
                d_model=16,
                output_dim=16,
                downsample_hidden_size=4,
            ),
            text_config=dict(
                vocab_size=16,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=8,
            ),
        )
    )
    inputs = dict(
        input_features=mx.random.normal((1, 16, 32)),
        feature_attention_mask=mx.ones((1, 32), dtype=mx.int32),
    )
    count = model.get_audio_features(
        inputs["input_features"], inputs["feature_attention_mask"]
    ).shape[0]
    return Case(model, inputs, count)


@case("qwen2_audio")
def _qwen2_audio():
    from mlx_audio.stt.models.qwen2_audio.config import ModelConfig
    from mlx_audio.stt.models.qwen2_audio.qwen2_audio import Model

    model = Model(
        ModelConfig(
            audio_token_id=7,
            audio_config=dict(
                num_mel_bins=16,
                encoder_layers=1,
                encoder_attention_heads=2,
                encoder_ffn_dim=32,
                d_model=16,
                max_source_positions=64,
            ),
            text_config=dict(
                vocab_size=16,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
            ),
        )
    )
    inputs = dict(input_features=mx.random.normal((1, 16, 32)))
    return Case(
        model, inputs, model.get_audio_features(inputs["input_features"]).shape[1]
    )


@case("higgs_audio_3")
def _higgs_audio_3():
    from mlx_audio.stt.models.higgs_audio_3.config import ModelConfig
    from mlx_audio.stt.models.higgs_audio_3.higgs_audio_3 import Model

    model = Model(
        ModelConfig(
            audio_in_token_idx=7,
            audio_encoder_config=dict(
                num_mel_bins=16,
                encoder_layers=1,
                encoder_attention_heads=2,
                encoder_ffn_dim=32,
                d_model=16,
                max_source_positions=64,
            ),
            text_config=dict(
                vocab_size=16,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=8,
            ),
        )
    )
    inputs = dict(input_features=mx.random.normal((1, 16, 32)))
    return Case(
        model, inputs, model.get_audio_features(inputs["input_features"]).shape[1]
    )


@case("whisper")
def _whisper():
    from mlx_audio.stt.models.whisper.whisper import Model, ModelDimensions

    model = Model(ModelDimensions(16, 16, 16, 2, 1, 16, 32, 16, 2, 1), dtype=mx.float32)
    return Case(
        model,
        dict(input_features=mx.random.normal((1, 32, 16))),
        0,
        encoder_prefix="encoder.",
    )


@case("canary")
def _canary():
    from mlx_audio.stt.models.canary.canary import Model
    from mlx_audio.stt.models.canary.config import ModelConfig

    model = Model(
        ModelConfig(
            vocab_size=16,
            enc_output_dim=16,
            encoder=dict(
                feat_in=16,
                n_layers=1,
                d_model=16,
                n_heads=2,
                subsampling_conv_channels=4,
                conv_kernel_size=3,
            ),
            transf_decoder=dict(
                num_layers=1, hidden_size=16, num_attention_heads=2, inner_size=32
            ),
        )
    )
    return Case(
        model,
        dict(input_features=mx.random.normal((1, 64, 16))),
        0,
        encoder_prefix="encoder.",
    )


@case("glmasr")
def _glmasr():
    from mlx_audio.stt.models.glmasr.config import ModelConfig
    from mlx_audio.stt.models.glmasr.glmasr import Model

    model = Model(
        ModelConfig(
            whisper_config=dict(
                num_mel_bins=16,
                encoder_layers=1,
                encoder_attention_heads=2,
                encoder_ffn_dim=32,
                d_model=16,
                max_source_positions=64,
            ),
            lm_config=dict(
                vocab_size=16,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=8,
            ),
        )
    )
    features = mx.random.normal((1, 32, 16))
    _, count = model.audio_encoder(features)
    return Case(
        model,
        dict(input_features=features, audio_offsets=[[1]], audio_length=[[count]]),
        count,
        encoder_prefix="audio_encoder.",
    )


@case("granite_speech")
def _granite_speech():
    from mlx_audio.stt.models.granite_speech.config import ModelConfig
    from mlx_audio.stt.models.granite_speech.granite_speech import Model

    model = Model(
        ModelConfig(
            audio_token_index=7,
            window_size=4,
            downsample_rate=2,
            encoder_config=dict(
                input_dim=16,
                num_layers=1,
                hidden_dim=16,
                num_heads=2,
                dim_head=8,
                context_size=8,
                conv_kernel_size=3,
                dropout=0,
            ),
            projector_config=dict(
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=32,
                encoder_hidden_size=16,
            ),
            text_config=dict(
                vocab_size=16,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
            ),
        )
    )
    inputs = dict(input_features=mx.random.normal((1, 16, 16)))
    return Case(
        model,
        inputs,
        model.get_audio_features(inputs["input_features"]).shape[1],
        encoder_prefix="encoder.",
    )


@case("cohere_asr")
def _cohere_asr():
    from mlx_audio.stt.models.cohere_asr.cohere_asr import Model
    from mlx_audio.stt.models.cohere_asr.config import ModelConfig

    model = Model(
        ModelConfig(
            vocab_size=16,
            encoder=dict(
                feat_in=16,
                n_layers=1,
                d_model=16,
                n_heads=2,
                subsampling_conv_channels=4,
                conv_kernel_size=3,
            ),
            transf_decoder=dict(
                config_dict=dict(
                    num_layers=1,
                    hidden_size=16,
                    num_attention_heads=2,
                    inner_size=32,
                )
            ),
            head=dict(hidden_size=16, num_classes=16),
        )
    )
    return Case(
        model,
        dict(
            input_features=mx.random.normal((1, 64, 16)),
            feature_lengths=mx.array([64]),
        ),
        0,
        encoder_prefix="encoder.",
    )


@case("fun_asr_nano")
def _fun_asr_nano():
    from mlx_audio.stt.models.fun_asr_nano import Model
    from mlx_audio.stt.models.fun_asr_nano.config import FunASRNanoConfig

    model = Model(
        FunASRNanoConfig(
            input_size=16,
            audio_encoder_conf=dict(
                output_size=16,
                attention_heads=2,
                linear_units=32,
                num_blocks=1,
                tp_blocks=1,
                kernel_size=3,
            ),
            audio_adaptor_conf=dict(
                encoder_dim=16, llm_dim=16, ffn_dim=32, n_layer=1, attention_heads=2
            ),
            text_config=dict(
                vocab_size=16,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=8,
            ),
        )
    )
    features, lengths = mx.random.normal((1, 32, 16)), mx.array([32])
    encoded, lens = model.audio_encoder(features, lengths)
    audio, _ = model.audio_adaptor(encoded, lens)
    count = audio.shape[1]
    return Case(
        model,
        dict(
            input_features=features,
            feature_lengths=lengths,
            audio_positions=mx.arange(1, count + 1),
        ),
        count,
        encoder_prefix="audio_encoder.",
    )


@case("moss_transcribe_diarize")
def _moss_transcribe_diarize():
    from mlx_audio.stt.models.moss_transcribe_diarize.config import ModelConfig
    from mlx_audio.stt.models.moss_transcribe_diarize.moss_transcribe_diarize import (
        Model,
    )

    model = Model(
        ModelConfig(
            audio_token_id=7,
            audio_config=dict(
                num_mel_bins=16,
                encoder_layers=1,
                encoder_attention_heads=2,
                encoder_ffn_dim=32,
                d_model=16,
                max_source_positions=64,
            ),
            text_config=dict(
                vocab_size=16,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=8,
            ),
        )
    )
    inputs = dict(
        input_features=mx.random.normal((1, 32, 16)),
        audio_feature_lengths=mx.array([4]),
        audio_chunk_mapping=mx.array([0]),
    )
    return Case(model, inputs, 4, encoder_prefix="model.whisper_encoder.")


@case("voxtral")
def _voxtral():
    from mlx_audio.stt.models.voxtral.config import ModelConfig
    from mlx_audio.stt.models.voxtral.voxtral import Model

    model = Model(
        ModelConfig(
            audio_token_id=7,
            audio_config=dict(
                hidden_size=16,
                num_hidden_layers=1,
                intermediate_size=64,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=8,
                num_mel_bins=16,
                encoder_layers=1,
                encoder_attention_heads=2,
                encoder_ffn_dim=32,
                d_model=16,
                max_source_positions=64,
            ),
            text_config=dict(
                vocab_size=16,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=8,
            ),
        )
    )
    inputs = dict(input_features=mx.random.normal((1, 128, 16)))
    count = model.get_audio_embeds(inputs["input_features"]).shape[0]
    return Case(model, inputs, count)


@case("vibevoice_asr")
def _vibevoice_asr():
    from mlx_audio.stt.models.vibevoice_asr.config import ModelConfig
    from mlx_audio.stt.models.vibevoice_asr.vibevoice_asr import Model

    tokenizer_config = dict(
        vae_dim=4,
        encoder_n_filters=4,
        encoder_ratios=[2, 2],
        encoder_depths="1-1-1",
    )
    model = Model(
        ModelConfig.from_dict(
            dict(
                acoustic_tokenizer_config=tokenizer_config,
                semantic_tokenizer_config=tokenizer_config,
                acoustic_vae_dim=4,
                semantic_vae_dim=4,
                decoder_config=dict(
                    vocab_size=16,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    layer_types=["full_attention"],
                ),
            )
        )
    )
    model._speech_pad_id = 7
    inputs = dict(input_features=mx.random.normal((1, 64)))
    count = model.encode_speech(inputs["input_features"]).shape[1]
    return Case(model, inputs, count, encoder_prefix="acoustic_tokenizer.")


@case("moonshine")
def _moonshine():
    from mlx_audio.stt.models.moonshine.config import ModelConfig
    from mlx_audio.stt.models.moonshine.moonshine import Model

    model = Model(
        ModelConfig(
            vocab_size=VOCAB,
            hidden_size=16,
            intermediate_size=32,
            encoder_num_hidden_layers=1,
            decoder_num_hidden_layers=1,
            encoder_num_attention_heads=2,
            decoder_num_attention_heads=2,
        )
    )
    return Case(
        model, dict(input_features=mx.random.normal((4000,))), encoder_prefix="encoder."
    )


@case("fireredasr2")
def _fireredasr2():
    from mlx_audio.stt.models.fireredasr2.config import ModelConfig
    from mlx_audio.stt.models.fireredasr2.fireredasr2 import FireRedASR2

    model = FireRedASR2(
        ModelConfig.from_dict(
            dict(
                idim=16,
                odim=VOCAB,
                d_model=16,
                encoder=dict(n_layers=1, n_head=2, d_model=16, kernel_size=3),
                decoder=dict(n_layers=1, n_head=2, d_model=16),
            )
        )
    )
    return Case(
        model,
        dict(input_features=mx.random.normal((1, 64, 16))),
        encoder_prefix="encoder.",
    )


@case("mms")
def _mms():
    from mlx_audio.stt.models.mms.mms import Model
    from mlx_audio.stt.models.wav2vec.wav2vec import ModelConfig

    model = Model(
        ModelConfig(
            vocab_size=VOCAB,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            conv_dim=(8, 8),
            conv_stride=(2, 2),
            conv_kernel=(3, 3),
            num_conv_pos_embeddings=4,
            num_conv_pos_embedding_groups=2,
            num_feat_extract_layers=2,
        )
    )
    return Case(
        model,
        dict(input_features=mx.random.normal((1, 400))),
        encoder_prefix="wav2vec2.",
    )


@case("granite_speech5_ctc")
def _granite_speech5_ctc():
    from mlx_audio.stt.models.granite_speech5_ctc.config import ModelConfig
    from mlx_audio.stt.models.granite_speech5_ctc.granite_speech5 import Model

    model = Model(
        ModelConfig(
            vocab_size=VOCAB,
            encoder_config=dict(
                vocab_size=VOCAB,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                num_mel_bins=16,
                head_dim=8,
                context_size=8,
                conv_kernel_size=3,
            ),
        )
    )
    # The encoder stacks four mel frames per input step.
    return Case(
        model,
        dict(input_features=mx.random.normal((1, 32, 64))),
        encoder_prefix="encoder.",
    )


@case("sensevoice")
def _sensevoice():
    from mlx_audio.stt.models.sensevoice.config import ModelConfig
    from mlx_audio.stt.models.sensevoice.sensevoice import SenseVoiceSmall

    model = SenseVoiceSmall(
        ModelConfig(
            vocab_size=VOCAB,
            input_size=16,
            encoder_conf=dict(
                output_size=16,
                attention_heads=2,
                linear_units=32,
                num_blocks=1,
                tp_blocks=1,
                kernel_size=3,
            ),
        )
    )
    return Case(
        model,
        dict(input_features=mx.random.normal((1, 24, 16))),
        encoder_prefix="encoder.",
    )


@case("parakeet")
def _parakeet():
    from mlx_audio.stt.models.parakeet.audio import PreprocessArgs
    from mlx_audio.stt.models.parakeet.conformer import ConformerArgs
    from mlx_audio.stt.models.parakeet.ctc import ConvASRDecoderArgs
    from mlx_audio.stt.models.parakeet.parakeet import (
        CTCDecodingArgs,
        ParakeetCTC,
        ParakeetCTCArgs,
    )

    vocabulary = [f"piece{index}" for index in range(VOCAB - 1)]
    model = ParakeetCTC(
        ParakeetCTCArgs(
            preprocessor=PreprocessArgs(
                16000, "per_feature", 0.025, 0.01, "hann", 16, 512, 0.0
            ),
            encoder=ConformerArgs(
                feat_in=16,
                n_layers=1,
                d_model=16,
                n_heads=2,
                ff_expansion_factor=2,
                subsampling_factor=4,
                self_attention_model="rel_pos",
                subsampling="dw_striding",
                conv_kernel_size=3,
                subsampling_conv_channels=4,
                pos_emb_max_len=128,
            ),
            decoder=ConvASRDecoderArgs(
                feat_in=16, num_classes=-1, vocabulary=vocabulary
            ),
            decoding=CTCDecodingArgs(greedy=None),
        )
    )
    return Case(
        model,
        dict(input_features=mx.random.normal((1, 64, 16))),
        encoder_prefix="encoder.",
    )


def prompt(case):
    return [1] + [case.audio_id] * case.audio_count + [3, 4]


def examples(case, model):
    """Two prepared utterances sharing the audio but not the transcript."""
    if model.training_objective == "ctc":
        return [{**case.inputs, "labels": labels} for labels in ([3, 4, 5], [5, 3])]
    return [
        {
            **case.inputs,
            "input_ids": np.array(prompt(case) + transcript),
            "completion_mask": np.array(
                [0] * len(prompt(case)) + [1] * len(transcript)
            ),
        }
        for transcript in ([5, 6, 2], [6, 5, 5, 2])
    ]


def test_every_family_has_a_random_model():
    assert set(CASES) == set(FAMILIES)


@pytest.mark.parametrize("family", sorted(CASES))
def test_normalized_logits_are_causal_and_reach_the_encoder(family):
    case = CASES[family]()
    model = normalize_model(case.model)
    tokens = {}
    if model.training_objective == "sft":
        tokens = {"input_ids": mx.array([prompt(case)])}
    logits = model.forward_logits(**tokens, **case.inputs)
    assert logits.shape[0] == 1 and logits.shape[-1] == VOCAB
    if tokens:
        assert logits.shape[1] == tokens["input_ids"].shape[1]
        # Teacher forcing must not let a future target affect earlier logits.
        changed = {"input_ids": tokens["input_ids"].at[:, -1].add(1)}
        again = model.forward_logits(**changed, **case.inputs)
        assert mx.allclose(logits[:, :-1], again[:, :-1], atol=1e-4).item()

    def loss(m):
        return m.forward_logits(**tokens, **case.inputs).astype(mx.float32).sum()

    value, grads = nn.value_and_grad(model, loss)(model)
    assert mx.isfinite(value).item()
    encoder = [
        mx.abs(gradient).sum().item()
        for name, gradient in tree_flatten(grads)
        if name.startswith(case.encoder_prefix)
    ]
    assert encoder and any(total > 0 for total in encoder)


@pytest.mark.parametrize("family", sorted(CASES))
def test_trainer_reduces_loss_on_a_random_model(family, tmp_path):
    mx.random.seed(0)
    case = CASES[family]()
    model = normalize_model(case.model)
    model.train()
    data = examples(case, model)
    trainer = Trainer(
        model,
        SFTTrainConfig(
            batch_size=2,
            iters=12,
            steps_per_report=12,
            steps_per_eval=12,
            adapter_file=str(tmp_path / "weights.safetensors"),
        ),
        task="asr",
        train_dataset=data,
        eval_dataset=data,
        prepared=True,
        optimizer=optim.Adam(learning_rate=3e-3),
    )
    before = trainer.evaluate()["val_loss"]
    metrics = trainer.train()
    assert math.isfinite(metrics["val_loss"]) and metrics["val_loss"] < before
