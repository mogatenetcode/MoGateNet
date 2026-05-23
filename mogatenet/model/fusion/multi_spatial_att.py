import math

import torch
import torch.nn as nn
from einops import rearrange
from torch import einsum

from .layers import Attention, PositionEmbedding, get_config, Mlp


class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.patch_embeddings = nn.Conv3d(
            in_channels=config.in_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = self.dropout(x)
        return x


def get_relative_distances(window_size):
    indices = torch.tensor(
        [
            [x, y, z]
            for x in range(window_size[0])
            for y in range(window_size[1])
            for z in range(window_size[2])
        ],
        dtype=torch.long,
    )
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, window_size, relative_pos_embedding=True):
        super().__init__()

        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        if self.relative_pos_embedding:
            relative_indices = get_relative_distances(window_size)
            relative_indices = relative_indices - relative_indices.min()
            max_index = relative_indices.max().item()

            self.register_buffer("relative_indices", relative_indices, persistent=False)
            self.pos_embedding = nn.Parameter(
                torch.randn(max_index + 1, max_index + 1, max_index + 1)
            )
        else:
            window_volume = window_size[0] * window_size[1] * window_size[2]
            self.pos_embedding = nn.Parameter(
                torch.randn(window_volume, window_volume)
            )

    def forward(self, x):
        batch_size, depth, height, width, _, num_heads = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        num_windows_d = depth // self.window_size[0]
        num_windows_h = height // self.window_size[1]
        num_windows_w = width // self.window_size[2]

        q, k, v = map(
            lambda t: rearrange(
                t,
                "b (nd wd) (nh wh) (nw ww) (h c) -> b h (nd nh nw) (wd wh ww) c",
                h=num_heads,
                wd=self.window_size[0],
                wh=self.window_size[1],
                ww=self.window_size[2],
            ),
            qkv,
        )

        dots = einsum("b h w i c, b h w j c -> b h w i j", q, k) * self.scale

        if self.relative_pos_embedding:
            dots = dots + self.pos_embedding[
                self.relative_indices[:, :, 0],
                self.relative_indices[:, :, 1],
                self.relative_indices[:, :, 2],
            ]
        else:
            dots = dots + self.pos_embedding

        attn = dots.softmax(dim=-1)

        out = einsum("b h w i j, b h w j c -> b h w i c", attn, v)
        out = rearrange(
            out,
            "b h (nd nh nw) (wd wh ww) c -> b (nd wd) (nh wh) (nw ww) (h c)",
            h=num_heads,
            wd=self.window_size[0],
            wh=self.window_size[1],
            ww=self.window_size[2],
            nd=num_windows_d,
            nh=num_windows_h,
            nw=num_windows_w,
        )

        out = self.to_out(out)
        return out


class MultiAxisAttention(nn.Module):
    def __init__(self, config, use_position=False):
        super().__init__()

        self.use_position = use_position

        self.depth_plane_attention = Attention(config)
        self.slice_attention = Attention(config)
        self.window_attention = WindowAttention(
            dim=config.hidden_size,
            heads=config.num_heads,
            head_dim=config.hidden_size // config.num_heads,
            window_size=config.window_size,
            relative_pos_embedding=True,
        )

        if use_position:
            self.pos_embedding_1 = PositionEmbedding(config, types=1)
            self.pos_embedding_2 = PositionEmbedding(config, types=2)

    def forward(self, x):
        batch_size, hidden_size, depth, height, width = x.shape

        x_1 = rearrange(x, "b c d h w -> (b d) (h w) c")
        x_2 = rearrange(x, "b c d h w -> (b h w) d c")
        x_3 = x.permute(0, 2, 3, 4, 1)

        if self.use_position:
            x_1 = self.pos_embedding_1(x_1)
            x_2 = self.pos_embedding_2(x_2)

        x_1 = self.depth_plane_attention(x_1)
        x_2 = self.slice_attention(x_2)
        x_3 = self.window_attention(x_3)

        x_1 = rearrange(
            x_1,
            "(b d) (h w) c -> b (d h w) c",
            b=batch_size,
            d=depth,
            h=height,
            w=width,
        )
        x_2 = rearrange(
            x_2,
            "(b h w) d c -> b (d h w) c",
            b=batch_size,
            d=depth,
            h=height,
            w=width,
        )
        x_3 = rearrange(
            x_3,
            "b d h w c -> b (d h w) c",
            d=depth,
            h=height,
            w=width,
        )

        return x_1 + x_2 + x_3


class MultiSpatialBlock(nn.Module):
    def __init__(self, config, use_position=False):
        super().__init__()

        self.config = config
        self.input_shape = config.img_size

        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

        self.attn = MultiAxisAttention(config, use_position=use_position)
        self.ffn = Mlp(config)

    def forward(self, x):
        batch_size, hidden_size, depth, height, width = x.shape

        x = rearrange(x, "b c d h w -> b (d h w) c")

        residual = x
        x = self.attention_norm(x)
        x = rearrange(
            x,
            "b (d h w) c -> b c d h w",
            d=depth,
            h=height,
            w=width,
        )

        x = self.attn(x)
        x = x + residual

        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + residual

        x = x.transpose(-1, -2)

        out_size = (
            self.input_shape[0] // self.config.patch_size[0],
            self.input_shape[1] // self.config.patch_size[1],
            self.input_shape[2] // self.config.patch_size[2],
        )

        x = x.view(
            batch_size,
            self.config.hidden_size,
            out_size[0],
            out_size[1],
            out_size[2],
        ).contiguous()

        return x


class MultiSpatialFusion(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_size,
        img_size,
        num_heads=8,
        mlp_size=256,
        num_layers=1,
        window_size=(8, 8, 8),
        out_hidden=False,
    ):
        super().__init__()

        self.config = get_config(
            in_channels=in_channels,
            hidden_size=hidden_size,
            patch_size=(1, 1, 1),
            img_size=img_size,
            mlp_dim=mlp_size,
            num_heads=num_heads,
            window_size=window_size,
        )

        self.embeddings = Embeddings(self.config)
        self.blocks = nn.ModuleList(
            [
                MultiSpatialBlock(self.config, use_position=(i == 0))
                for i in range(num_layers)
            ]
        )
        self.out_hidden = out_hidden

    def forward(self, x):
        x = self.embeddings(x)

        hidden_states = []

        for block in self.blocks:
            x = block(x)
            if self.out_hidden:
                hidden_states.append(x)

        if self.out_hidden:
            return x, hidden_states

        return x
