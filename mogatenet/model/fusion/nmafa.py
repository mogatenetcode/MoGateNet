import torch.nn as nn
from einops import rearrange

from .multi_spatial_att import MultiSpatialFusion
from .cross_modality_att import CrossModalityFusion


class NMaFaLayer(nn.Module):
    def __init__(
        self,
        model_num,
        in_channels,
        hidden_size,
        img_size,
        mlp_size=256,
        self_num_layer=2,
        window_size=(2, 4, 4),
        token_mixer_size=32,
        token_learner=True,
    ):
        super().__init__()

        self.model_num = model_num
        self.img_size = img_size
        self.hidden_size = hidden_size

        self.spatial_att = MultiSpatialFusion(
            in_channels=model_num * in_channels,
            hidden_size=hidden_size,
            img_size=img_size,
            mlp_size=mlp_size,
            num_layers=self_num_layer,
            window_size=window_size,
        )

        self.modality_att = CrossModalityFusion(
            model_num=model_num,
            in_channels=in_channels,
            hidden_size=hidden_size,
            img_size=img_size,
            token_learner=token_learner,
            token_mixer_size=token_mixer_size,
        )

    def forward(self, x):
        spatial_input = rearrange(x, "b m c d h w -> b (m c) d h w")
        spatial_feature = self.spatial_att(spatial_input)
        fusion_out = self.modality_att(spatial_feature, x)

        return fusion_out
