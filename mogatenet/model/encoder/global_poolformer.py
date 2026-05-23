from typing import Sequence

import torch
import torch.nn as nn
from einops import rearrange

from ..fusion.layers import get_config


class Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride_size, padding_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride_size, padding_size),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class TwoConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride_size, padding_size):
        super().__init__()
        self.conv_1 = Convolution(in_channels, out_channels, kernel_size, stride_size, padding_size)
        self.conv_2 = Convolution(out_channels, out_channels, kernel_size, stride_size, padding_size)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x


class UpCat(nn.Module):
    def __init__(self, in_chns, cat_chns, out_chns, pool_size=(2, 2, 2)):
        super().__init__()
        up_chns = in_chns // 2
        self.upsample = nn.ConvTranspose3d(
            in_chns,
            up_chns,
            kernel_size=pool_size,
            stride=pool_size,
            padding=0,
        )
        self.convs = TwoConv(cat_chns + up_chns, out_chns, 3, 1, 1)

    def forward(self, x, encoder_feature):
        x = self.upsample(x)
        x = torch.cat([encoder_feature, x], dim=1)
        return self.convs(x)


class MlpChannel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Conv3d(config.hidden_size, config.mlp_dim, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(config.mlp_dim, config.hidden_size, kernel_size=1)
        self.drop = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LayerNormChannel(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embeddings = nn.Conv3d(
            in_channels=config.in_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.norm = LayerNormChannel(num_channels=config.hidden_size)

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = self.norm(x)
        return x


class GlobalPool(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.img_size = config.img_size
        all_size = self.img_size[0] * self.img_size[1] * self.img_size[2]
        self.global_layer = nn.Linear(1, all_size)

    def forward(self, x):
        x = rearrange(x, "b c d h w -> b c (d h w)")
        x = x.mean(dim=-1, keepdim=True)
        x = self.global_layer(x)
        x = rearrange(
            x,
            "b c (d h w) -> b c d h w",
            d=self.img_size[0],
            h=self.img_size[1],
            w=self.img_size[2],
        )
        return x


class GlobalPoolFormerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = LayerNormChannel(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNormChannel(config.hidden_size, eps=1e-6)
        self.attn = GlobalPool(config)
        self.ffn = MlpChannel(config)

    def forward(self, x):
        residual = x
        x = self.attention_norm(x)
        x = self.attn(x) + x
        x = x + residual

        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + residual

        return x


class GlobalPoolFormer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        img_size,
        patch_size,
        mlp_size=256,
        num_layers=1,
    ):
        super().__init__()

        self.config = get_config(
            in_channels=in_channels,
            hidden_size=out_channels,
            patch_size=patch_size,
            mlp_dim=mlp_size,
            img_size=img_size,
        )

        self.embeddings = Embeddings(self.config)
        self.blocks = nn.ModuleList(
            [GlobalPoolFormerBlock(self.config) for _ in range(num_layers)]
        )

    def forward(self, x, out_hidden=False):
        x = self.embeddings(x)
        hidden_states = []

        for block in self.blocks:
            x = block(x)
            hidden_states.append(x)

        if out_hidden:
            return x, hidden_states

        return x


class GlobalPoolFormerEncoder(nn.Module):
    def __init__(self, img_size, in_channels, features: Sequence[int], pool_size):
        super().__init__()

        self.conv_0 = TwoConv(in_channels, features[0], 3, 1, 1)

        self.down_1 = GlobalPoolFormer(
            features[0],
            features[1],
            img_size=img_size[0],
            patch_size=pool_size[0],
            mlp_size=features[1] * 2,
            num_layers=2,
        )
        self.down_2 = GlobalPoolFormer(
            features[1],
            features[2],
            img_size=img_size[1],
            patch_size=pool_size[1],
            mlp_size=features[2] * 2,
            num_layers=2,
        )
        self.down_3 = GlobalPoolFormer(
            features[2],
            features[3],
            img_size=img_size[2],
            patch_size=pool_size[2],
            mlp_size=features[3] * 2,
            num_layers=2,
        )
        self.down_4 = GlobalPoolFormer(
            features[3],
            features[4],
            img_size=img_size[3],
            patch_size=pool_size[3],
            mlp_size=features[4] * 2,
            num_layers=2,
        )

    def forward(self, x):
        x0 = self.conv_0(x)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)

        return x4, x3, x2, x1, x0


class Encoder(nn.Module):
    def __init__(self, model_num, img_size, fea, pool_size):
        super().__init__()
        self.model_num = model_num
        self.encoders = nn.ModuleList(
            [
                GlobalPoolFormerEncoder(
                    img_size=img_size,
                    in_channels=1,
                    features=fea,
                    pool_size=pool_size,
                )
                for _ in range(model_num)
            ]
        )

    def forward(self, x):
        encoder_outputs = []
        x = x.unsqueeze(dim=2)

        for i in range(self.model_num):
            encoder_outputs.append(self.encoders[i](x[:, i]))

        return encoder_outputs
