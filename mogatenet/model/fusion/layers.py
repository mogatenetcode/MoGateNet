import math

import ml_collections
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_config(
    in_channels=1,
    hidden_size=128,
    img_size=(1, 1, 1),
    patch_size=(1, 1, 1),
    mlp_dim=256,
    num_heads=8,
    window_size=(8, 8, 8),
):
    config = ml_collections.ConfigDict()

    config.in_channels = in_channels
    config.hidden_size = hidden_size
    config.mlp_dim = mlp_dim
    config.num_heads = num_heads
    config.num_layers = 1
    config.attention_dropout_rate = 0.0
    config.dropout_rate = 0.1
    config.patch_size = patch_size
    config.img_size = img_size
    config.window_size = window_size

    return config


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {
    "gelu": F.gelu,
    "relu": F.relu,
    "swish": swish,
}


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_heads
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.attention_dropout_rate)
        self.proj_dropout = nn.Dropout(config.attention_dropout_rate)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, return_attention=False):
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        query = self.transpose_for_scores(query)
        key = self.transpose_for_scores(key)
        value = self.transpose_for_scores(value)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context = torch.matmul(attention_probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()

        new_context_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*new_context_shape)

        attention_output = self.out(context)
        attention_output = self.proj_dropout(attention_output)

        if return_attention:
            return attention_output, attention_probs

        return attention_output


class PositionEmbedding(nn.Module):
    def __init__(self, config, types=0):
        super().__init__()

        img_size = config.img_size
        patch_size = config.patch_size

        if types == 0:
            num_patches = (
                (img_size[0] // patch_size[0])
                * (img_size[1] // patch_size[1])
                * (img_size[2] // patch_size[2])
            )
        elif types == 1:
            num_patches = (
                (img_size[1] // patch_size[1])
                * (img_size[2] // patch_size[2])
            )
        elif types == 2:
            num_patches = img_size[0] // patch_size[0]
        else:
            raise ValueError(f"Unsupported position embedding type: {types}")

        self.position_embeddings = nn.Parameter(
            torch.zeros(1, num_patches, config.hidden_size)
        )

    def forward(self, x):
        return x + self.position_embeddings


class Mlp(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.fc1 = nn.Linear(config.hidden_size, config.mlp_dim)
        self.fc2 = nn.Linear(config.mlp_dim, config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x
