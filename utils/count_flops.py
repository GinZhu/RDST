import torch
from torch import nn


def count_layer_norm(m, x, y):
    x = x[0]
    flops = torch.DoubleTensor([2*x.numel()])
    m.total_ops += flops

class LNT(nn.Module):

    def __init__(self, colors, feats):
        super(LNT, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(colors, feats, kernel_size=3),
            nn.LayerNorm((64, 14, 14)),
        )

    def forward(self, x):
        return self.body(x)
