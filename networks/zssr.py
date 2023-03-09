from torch import nn


class ZSSRNet_ori(nn.Module):
    """
    We use a simple, fully convolutional network, with 8 hidden layers, each has 64 channels.
    We use ReLU activations on each layer. The network input is interpolated to the output size.
    we only learn the residual between the interpolated LR and its HR parent.

    Donghao suggested to use BatchNorm.
    ! based on Jin's experiments, Batch Norm led to bad performance

    Jin is thinking that probably we can use a deeper neural network, or probably densenet?
    """
    def __init__(self, input_channel=3, kernel_size=3, inside_channel=64, num_layers=8, norm='BN', residual=True,
                 activation='relu'):
        super(ZSSRNet_ori, self).__init__()

        self.norm = norm
        self.residual = residual

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        else:
            self.activation = nn.ReLU()

        # ## make layers
        layers = []

        # ## input layer
        layers.append(nn.Conv2d(input_channel, inside_channel, kernel_size=kernel_size,
                                padding=kernel_size // 2, bias=True))
        if self.norm == 'BN':
            layers.append(nn.BatchNorm2d(inside_channel, momentum=0.001))
        layers.append(self.activation)

        for i in range(num_layers-2):
            layers.append(nn.Conv2d(inside_channel, inside_channel, kernel_size=kernel_size,
                                    padding=kernel_size // 2, bias=True))
            if self.norm == 'BN':
                layers.append(nn.BatchNorm2d(inside_channel, momentum=0.001))
            layers.append(self.activation)

        layers.append(nn.Conv2d(inside_channel, input_channel, kernel_size=kernel_size,
                                padding=kernel_size // 2, bias=True))

        self.model = nn.Sequential(*layers)

        # ## initial
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        temp = x
        x = self.model(x)
        if self.residual:
            x += temp

        return x

