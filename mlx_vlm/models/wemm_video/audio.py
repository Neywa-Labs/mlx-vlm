"""WeMM's Whisper encoder and windowed query transformer in MLX.

Architecture reference: Tencent/WeMM-Video (Apache-2.0), h5_modeling_whisper.py
and h5_modeling_bert.py. The Whisper blocks are shared with MiniCPM-o.
"""

import mlx.core as mx
import mlx.nn as nn

from ..minicpmo.audio import AudioAttention, AudioEncoderLayer
from .config import AudioConfig, QFormerConfig


class WhisperAttention(AudioAttention):
    def __call__(self, hidden_states, attention_mask=None):
        batch, length, width = hidden_states.shape
        # Whisper scales the projected queries before attention. Preserve that
        # rounding point for BF16 instead of scaling inside the fused kernel.
        queries = self.q_proj(hidden_states) * self.scale
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)
        queries, keys, values = [
            value.reshape(batch, length, self.num_heads, self.head_dim).transpose(
                0, 2, 1, 3
            )
            for value in (queries, keys, values)
        ]
        output = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=1.0)
        return self.out_proj(output.transpose(0, 2, 1, 3).reshape(batch, length, width))


class QueryAttention(nn.Module):
    def __init__(self, config: QFormerConfig, cross=False):
        super().__init__()
        width = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = width // self.num_heads
        source_width = config.encoder_width if cross else width
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(source_width, width)
        self.value = nn.Linear(source_width, width)
        self.dense = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width, eps=config.layer_norm_eps)

    def __call__(self, x, source=None):
        source = x if source is None else source
        batch, length, width = x.shape
        q = self.query(x).reshape(batch, length, self.num_heads, self.head_dim)
        k = self.key(source).reshape(batch, -1, self.num_heads, self.head_dim)
        v = self.value(source).reshape(batch, -1, self.num_heads, self.head_dim)
        y = (
            mx.fast.scaled_dot_product_attention(
                q.transpose(0, 2, 1, 3),
                k.transpose(0, 2, 1, 3),
                v.transpose(0, 2, 1, 3),
                scale=self.head_dim**-0.5,
            )
            .transpose(0, 2, 1, 3)
            .reshape(batch, length, width)
        )
        return self.norm(x + self.dense(y))


class QueryLayer(nn.Module):
    def __init__(self, config: QFormerConfig, index: int):
        super().__init__()
        self.attention = QueryAttention(config)
        self.crossattention = (
            QueryAttention(config, cross=True)
            if index % config.cross_attention_freq == 0
            else None
        )
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, queries, source):
        x = self.attention(queries)
        if self.crossattention is not None:
            x = self.crossattention(x, source)
        return self.norm(x + self.fc2(nn.gelu(self.fc1(x))))


class QFormer(nn.Module):
    def __init__(self, config: QFormerConfig):
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = [QueryLayer(config, i) for i in range(config.num_hidden_layers)]

    def __call__(self, queries, source):
        x = self.norm(queries)
        for layer in self.layers:
            x = layer(x, source)
        return x


class AudioEncoder(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv1d(config.num_mel_bins, config.d_model, 3, padding=1)
        self.conv2 = nn.Conv1d(config.d_model, config.d_model, 3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(config.max_source_positions, config.d_model)
        self.layers = [AudioEncoderLayer(config) for _ in range(config.encoder_layers)]
        for layer in self.layers:
            layer.self_attn = WhisperAttention(config)
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        qconfig = config.qformer_config
        self.q_tokens = mx.zeros((1, qconfig.query_length, qconfig.hidden_size))
        self.qformer = QFormer(qconfig)
        self.audio_proj = nn.Linear(qconfig.hidden_size, qconfig.out_hidden_size)

    def __call__(self, features):
        expected = (self.config.num_mel_bins, self.config.max_source_positions * 2)
        if features.ndim != 3 or features.shape[1:] != expected:
            raise ValueError(
                f"Expected audio features (segments, {expected[0]}, {expected[1]}), got {features.shape}."
            )
        x = features.transpose(0, 2, 1)
        x = nn.gelu(self.conv1(x))
        x = nn.gelu(self.conv2(x)) + self.embed_positions.weight
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        batch, length, width = x.shape
        window, stride = self.config.window_size, self.config.window_stride
        indices = mx.arange(0, length - window + 1, stride)[:, None] + mx.arange(window)
        windows = x[:, indices, :].reshape(-1, window, width)
        queries = mx.broadcast_to(
            self.q_tokens, (windows.shape[0], *self.q_tokens.shape[1:])
        )
        return self.audio_proj(self.qformer(queries, windows)).reshape(
            batch, -1, self.config.qformer_config.out_hidden_size
        )


def sanitize_audio_weights(weights):
    result = {}
    for key, value in weights.items():
        if key.startswith("model.audio."):
            key = key.replace("model.audio.", "audio_tower.", 1)
            if key.endswith("position_ids"):
                continue
            if key.endswith(("conv1.weight", "conv2.weight")):
                value = value.transpose(0, 2, 1)
            key = key.replace(".gamma", ".weight").replace(".beta", ".bias")
            key = key.replace("qformer.bert.embeddings.LayerNorm", "qformer.norm")
            key = key.replace("qformer.bert.encoder.layer.", "qformer.layers.")
            key = key.replace(".attention.self.", ".attention.")
            key = key.replace(".crossattention.self.", ".crossattention.")
            key = key.replace(".output.dense.", ".dense.")
            key = key.replace(".output.LayerNorm.", ".norm.")
            key = key.replace(".intermediate_query.dense.", ".fc1.")
            key = key.replace(".output_query.dense.", ".fc2.")
            key = key.replace(".output_query.LayerNorm.", ".norm.")
        result[key] = value
    return result
