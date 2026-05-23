import math

import torch
import torch.nn as nn
from einops import rearrange

from .layers import Mlp, get_config


class ModalitySelectionGate(nn.Module):
    def __init__(self, hidden_size, num_modalities):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, modality_embeddings):
        weights = []

        for embedding in modality_embeddings:
            weight = self.gate(embedding.mean(dim=1))
            weights.append(weight)

        weights = torch.stack(weights, dim=1)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

        weighted_embeddings = [
            embedding * weights[:, i:i + 1, :]
            for i, embedding in enumerate(modality_embeddings)
        ]

        return torch.cat(weighted_embeddings, dim=1)


class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        img_size = config.img_size
        patch_size = config.patch_size
        num_patches = (
            (img_size[0] // patch_size[0])
            * (img_size[1] // patch_size[1])
            * (img_size[2] // patch_size[2])
        )

        self.patch_embeddings = nn.Conv3d(
            config.in_channels,
            config.hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, num_patches, config.hidden_size)
        )
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.position_embeddings
        return self.dropout(x)


class CrossModalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.all_head_size = self.head_dim * self.num_heads

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)

        self.attn_dropout = nn.Dropout(config.attention_dropout_rate)
        self.proj_dropout = nn.Dropout(config.attention_dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        batch_size, num_tokens, _ = x.shape
        x = x.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, query_tokens, key_value_tokens):
        query = self.transpose_for_scores(self.query(query_tokens))
        key = self.transpose_for_scores(self.key(key_value_tokens))
        value = self.transpose_for_scores(self.value(key_value_tokens))

        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        probs = self.softmax(scores)
        probs = self.attn_dropout(probs)

        context = torch.matmul(probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(query.size(0), -1, self.all_head_size)

        out = self.out(context)
        return self.proj_dropout(out)


class CrossAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attn_norm = nn.LayerNorm(config.hidden_size)
        self.ffn_norm = nn.LayerNorm(config.hidden_size)
        self.attn = CrossModalAttention(config)
        self.ffn = Mlp(config)

    def forward(self, query_tokens, key_value_tokens):
        x = self.attn(query_tokens, key_value_tokens) + query_tokens
        x = self.attn_norm(x)
        x = self.ffn(x) + x
        x = self.ffn_norm(x)
        return x


class TokenLearner(nn.Module):
    def __init__(self, in_channels, num_tokens):
        super().__init__()
        self.token_conv = nn.Conv3d(in_channels, num_tokens, kernel_size=3, padding=1)

    def forward(self, x):
        selected = torch.sigmoid(self.token_conv(x))
        selected = rearrange(selected, "b s d h w -> b s (d h w) 1")
        x = rearrange(x, "b c d h w -> b 1 (d h w) c")
        return (x * selected).mean(dim=2)


class CrossModalityFusion(nn.Module):
    def __init__(
        self,
        model_num,
        in_channels,
        hidden_size,
        img_size,
        mlp_size=256,
        token_mixer_size=32,
        token_learner=False,
        use_msg=True,
    ):
        super().__init__()

        patch_size = (1, 1, 1)
        self.config = get_config(
            in_channels=in_channels,
            hidden_size=hidden_size,
            patch_size=patch_size,
            img_size=img_size,
            mlp_dim=mlp_size,
        )

        self.model_num = model_num
        self.img_size = img_size
        self.token_learner = token_learner
        self.use_msg = use_msg

        patch_num = (
            (img_size[0] // patch_size[0])
            * (img_size[1] // patch_size[1])
            * (img_size[2] // patch_size[2])
        )

        self.embeddings = nn.ModuleList(
            [PatchEmbedding(self.config) for _ in range(model_num)]
        )

        if token_learner:
            self.token_mixer = TokenLearner(
                in_channels=in_channels,
                num_tokens=token_mixer_size,
            )
        else:
            self.token_mixer = nn.Linear(patch_num, token_mixer_size)

        if use_msg:
            self.modality_gate = ModalitySelectionGate(hidden_size, model_num)

        self.cross_attention = CrossAttentionBlock(self.config)

    def forward(self, query_feature, modality_features):
        query_tokens = rearrange(query_feature, "b c d h w -> b (d h w) c")
        modality_tokens = []

        for i in range(self.model_num):
            tokens = self.embeddings[i](modality_features[:, i])

            if self.token_learner:
                tokens = rearrange(
                    tokens,
                    "b (d h w) c -> b c d h w",
                    d=self.img_size[0],
                    h=self.img_size[1],
                    w=self.img_size[2],
                )
                tokens = self.token_mixer(tokens)
            else:
                tokens = tokens.transpose(1, 2)
                tokens = self.token_mixer(tokens)
                tokens = tokens.transpose(1, 2)

            modality_tokens.append(tokens)

        if self.use_msg:
            fused_tokens = self.modality_gate(modality_tokens)
        else:
            fused_tokens = torch.cat(modality_tokens, dim=1)

        x = self.cross_attention(query_tokens, fused_tokens)
        x = x.transpose(1, 2)
        x = x.view(query_feature.size(0), self.config.hidden_size, *self.img_size)

        return x
