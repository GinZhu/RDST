from collections import OrderedDict

from torch import nn
import numpy as np
import torch


class SelectiveSequential(nn.Module):
    def __init__(self, modules_dict):
        super().__init__()
        self.modules_dict = OrderedDict(modules_dict.items())
        for key, module in modules_dict.items():
            self.add_module(key, module)

    def forward(self, x):
        for name, module in self.modules_dict.items():
            x = module(x)
        return x


class VGG19(nn.Module):
    def __init__(self, pool_module=nn.MaxPool2d, mode='Minc_VGG22', pre_activation=True):
        super().__init__()

        self.mode = mode
        assert self.mode in ['Minc_VGG22', 'Minc_VGG54']
        self.pre_activation = pre_activation

        layers = [
            ('conv1_1', nn.Conv2d(3, 64, kernel_size=3, padding=1)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_2', nn.Conv2d(64, 64, kernel_size=3, padding=1)),
            ('relu1_2', nn.ReLU(inplace=True)),
            ('pool1', pool_module(kernel_size=2, stride=2)),

            ('conv2_1', nn.Conv2d(64, 128, kernel_size=3, padding=1)),
            ('relu2_1', nn.ReLU(inplace=True)),
            ('conv2_2', nn.Conv2d(128, 128, kernel_size=3, padding=1)),
        ]

        if not self.pre_activation:
            layers += [('relu2_2', nn.ReLU(inplace=True)), ]

        if self.mode == 'Minc_VGG54':
            layers += [
                ('pool2', pool_module(kernel_size=2, stride=2)),

                ('conv3_1', nn.Conv2d(128, 256, kernel_size=3, padding=1)),
                ('relu3_1', nn.ReLU(inplace=True)),
                ('conv3_2', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                ('relu3_2', nn.ReLU(inplace=True)),
                ('conv3_3', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                ('relu3_3', nn.ReLU(inplace=True)),
                ('conv3_4', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                ('relu3_4', nn.ReLU(inplace=True)),
                ('pool3', pool_module(kernel_size=2, stride=2)),

                ('conv4_1', nn.Conv2d(256, 512, kernel_size=3, padding=1)),
                ('relu4_1', nn.ReLU(inplace=True)),
                ('conv4_2', nn.Conv2d(512, 512, kernel_size=3, padding=1)),
                ('relu4_2', nn.ReLU(inplace=True)),
                ('conv4_3', nn.Conv2d(512, 512, kernel_size=3, padding=1)),
                ('relu4_3', nn.ReLU(inplace=True)),
                ('conv4_4', nn.Conv2d(512, 512, kernel_size=3, padding=1)),
                ('relu4_4', nn.ReLU(inplace=True)),
                ('pool4', pool_module(kernel_size=2, stride=2)),

                ('conv5_1', nn.Conv2d(512, 512, kernel_size=3, padding=1)),
                ('relu5_1', nn.ReLU(inplace=True)),
                ('conv5_2', nn.Conv2d(512, 512, kernel_size=3, padding=1)),
                ('relu5_2', nn.ReLU(inplace=True)),
                ('conv5_3', nn.Conv2d(512, 512, kernel_size=3, padding=1)),
                ('relu5_3', nn.ReLU(inplace=True)),
                ('conv5_4', nn.Conv2d(512, 512, kernel_size=3, padding=1)),
            ]

        if not self.pre_activation:
            layers += [('relu5_4', nn.ReLU(inplace=True)), ]

        self.features = SelectiveSequential(OrderedDict(layers))

    def forward(self, x):
        return self.features(x)

    def load_module_npy(self, path):
        data = np.load(path, allow_pickle=True)[()]
        for name, child in self.features._modules.items():
            if name in data:
                print("Loading {} => {}".format(name, child))
                weight_shape = tuple(child.weight.size())
                weights = data[name]['weights']
                if weight_shape != weights.shape:
                    print("\tReshaping weight {} => {}".format(weights.shape, weight_shape))
                    weights = weights.reshape(weight_shape)
                weights = torch.from_numpy(weights)
                bias = data[name]['biases']
                bias = torch.from_numpy(bias)
                child.weight.data.copy_(weights)
                child.bias.data.copy_(bias)
