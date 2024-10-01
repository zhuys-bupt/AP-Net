import torch
from torch import nn, Tensor
from torch.nn.utils import weight_norm
from models.submodule import Mish


class Refine(nn.Module):
    def __init__(self, num_blocks=8, feature_dim=16, expansion=4):
        super().__init__()
        self.num_blocks = num_blocks

        # disp head
        self.in_conv = nn.Conv2d(5, feature_dim, kernel_size=3, padding=1)
        self.layers = nn.ModuleList([ResBlock(feature_dim, expansion) for _ in range(num_blocks)])
        self.out_conv = nn.Conv2d(feature_dim, 1, kernel_size=3, padding=1)

    def forward(self, info_entr: Tensor, disp_raw: Tensor, img: Tensor):
        feat = self.in_conv(torch.cat([info_entr, disp_raw, img], dim=1))
        for layer in self.layers:
            feat = layer(feat)
        disp_res = self.out_conv(feat)
        disp_final = disp_raw + disp_res

        return disp_final


class ResBlock(nn.Module):
    def __init__(self, n_feats: int, expansion_ratio: int):
        super(ResBlock, self).__init__()
        self.module = nn.Sequential(
            weight_norm(nn.Conv2d(n_feats, n_feats * expansion_ratio, kernel_size=3, padding=1)),
            Mish(),
            weight_norm(nn.Conv2d(n_feats * expansion_ratio, n_feats, kernel_size=3, padding=1))
        )

    def forward(self, x: torch.Tensor):
        return x + self.module(x)

