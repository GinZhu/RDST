from networks.common import BasicBlock
import torch.nn as nn


class Discriminator(nn.Module):
    """
    GAN loss:
        1. vanilla: with bn, and with sigmoid at the end;
        2. WGAN: with bn, without sigmoid;
        3. WGAN_GP: without layer-norm, without sigmoid.
    """
    def __init__(self, paras):
        super(Discriminator, self).__init__()

        self.gan_type = paras.gan_type
        in_channels = paras.input_channel
        out_channels = 64
        depth = 7
        if 'GP' in self.gan_type:
            bn = False
        else:
            bn = True

        if paras.d_act == 'relu':
            act = nn.ReLU(inplace=False)
        elif paras.d_act == 'leaky_relu':
            slope = paras.leaky_relu_slope
            act = nn.LeakyReLU(negative_slope=slope, inplace=False)

        m_features = [
            BasicBlock(in_channels, out_channels, 3, bn=bn, act=act)
        ]
        for i in range(depth):
            in_channels = out_channels
            if i % 2 == 1:
                stride = 1
                out_channels *= 2
            else:
                stride = 2
            m_features.append(BasicBlock(
                in_channels, out_channels, 3, stride=stride, bn=bn, act=act
            ))

        self.features = nn.Sequential(*m_features)

        patch_size = int(paras.patch_size * paras.sr_scale) // (2**((depth + 1) // 2))
        m_classifier = [
            nn.Linear(out_channels * patch_size**2, 1024),
            act,
            nn.Linear(1024, 1)
        ]

        self.classifier = nn.Sequential(*m_classifier)

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features.contiguous().view(features.size(0), -1))

        return output
