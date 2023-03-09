import torch
import torch.nn as nn


class UpProjectionUnit(nn.Module):
    """
    up-projection unit from paper DBPN
        http://openaccess.thecvf.com/content_cvpr_2018/papers/Haris_Deep_Back-Projection_Networks_CVPR_2018_paper.pdf

    In D-DBPN, the input for each unit is the concatenation of the outputs from all previous units. Let the Lt˜ and Ht˜
    be the input for dense up- and down-projection unit, respectively. They are generated using conv(1, nR) which is
    used to merge all previous outputs from each unit as shown in Fig. 4. This improvement enables us to generate the
    feature maps effectively, as shown in the experimental results.

    In the proposed networks, the filter size in the projection unit is various with respect to the scaling factor.
    For 2× enlargement, we use 6 × 6 convolutional layer with two striding and two padding. Then, 4× enlargement use
    8 × 8 convolutional layer with four striding and two padding. Finally, the 8× enlargement use 12 × 12 convolutional
    layer with eight striding and two padding.

    Our final network, D-DBPN, uses conv(3, 256) then conv(1, 64) for the initial feature extraction and t = 7 for
    the back-projection stages. In the reconstruction, we use conv(3, 3). RGB color channels are used for input and
    output image. It takes less than four days to train.
    """

    def __init__(self, ic=64, oc=64, sr_factor=2, up_sample='Deconv'):
        super(UpProjectionUnit, self).__init__()
        self.sr_factor = sr_factor
        self.up_sample_mode = up_sample

        self.deconv_0 = None
        self.deconv_1 = None

        self.activation = nn.PReLU()

        self.conv = None


        self.input = None
        self.dense_input = (ic != oc)
        if self.dense_input:
            self.input = nn.Conv2d(ic, oc, 1)

        conv_paras = {2: [6, 2, 2, 0],   # ## when lr.size < 32, shoud be [3, 2, 1, 1] large:[6, 2, 2, 0]
                      4: [8, 4, 2, 0],
                      8: [12, 8, 2, 0]}

        k, s, p, out_padding = conv_paras[self.sr_factor]

        if self.up_sample_mode is 'Deconv':
            self.deconv_0 = nn.ConvTranspose2d(oc, oc, k, s, p, output_padding=out_padding)
            self.deconv_1 = nn.ConvTranspose2d(oc, oc, k, s, p, output_padding=out_padding)
            self.conv = nn.Conv2d(oc, oc, k, s, p)

        elif self.up_sample_mode is 'SubPixel':
            pass

    def forward(self, lt_1):
        """
        Input: L.t-1
        Output: H.t

        :param x:
        :return:
        """

        if self.dense_input:
            lt_1 = self.input(lt_1)
            lt_1 = self.activation(lt_1)

        ht0 = self.deconv_0(lt_1)
        ht0 = self.activation(ht0)

        lt0 = self.conv(ht0)
        lt0 = self.activation(lt0)

        etl = lt0 - lt_1
        ht1 = self.deconv_1(etl)
        ht1 = self.activation(ht1)

        ht = ht0+ht1

        return ht


class DownProjectionUnit(nn.Module):
    """
    down-projection unit from paper DBPN
        http://openaccess.thecvf.com/content_cvpr_2018/papers/Haris_Deep_Back-Projection_Networks_CVPR_2018_paper.pdf

    """

    def __init__(self, ic=64, oc=64, sr_factor=2, up_sample='Deconv'):
        super(DownProjectionUnit, self).__init__()
        self.sr_factor = sr_factor
        self.up_sample_mode = up_sample

        self.conv_0 = None
        self.conv_1 = None

        self.activation = nn.PReLU()

        self.deconv = None

        self.dense_input = (ic != oc)
        self.input = None
        if self.dense_input:
            self.input = nn.Conv2d(ic, oc, 1)

        conv_paras = {2: [6, 2, 2, 0],  # ## when lr.size < 32, shoud be [3, 2, 1, 1], large: [6, 2, 2, 0]
                      4: [8, 4, 2, 0],
                      8: [12, 8, 2, 0]}

        k, s, p, out_padding = conv_paras[self.sr_factor]

        if self.up_sample_mode is 'Deconv':
            self.conv_0 = nn.Conv2d(oc, oc, k, s, p)
            self.conv_1 = nn.Conv2d(oc, oc, k, s, p)
            self.deconv = nn.ConvTranspose2d(oc, oc, k, s, p, output_padding=out_padding)
        elif self.up_sample_mode is 'SubPixel':
            pass

    def forward(self, ht):
        """
        Input: L.t-1
        Output: H.t

        :param x:
        :return:
        """

        if self.dense_input:
            ht = self.input(ht)
            ht = self.activation(ht)

        lt0 = self.conv_0(ht)
        lt0 = self.activation(lt0)

        ht0 = self.deconv(lt0)
        ht0 = self.activation(ht0)

        eth = ht0 - ht

        lt1 = self.conv_1(eth)
        lt1 = self.activation(lt1)

        lt = lt0 + lt1

        return lt


class DeepBackProjectionNet(nn.Module):
    """
    Deep back projection Networks for Super-Resolution

    Our final network, D-DBPN, uses conv(3, 256) then conv(1, 64) for the initial feature extraction and t = 7 for
    the back-projection stages. In the reconstruction, we use conv(3, 3). RGB color channels are used for input and
    output image. It takes less than four days to train.

    Reconstruction:
    I.sr = f.rec([H.1, H.2, ..., H.t])
    where f.rec is conv(k=3, oc=image_c)

    Parameters from paper:
        n0 = 256
        nr = 64
        t = 7
    """

    def __init__(self, image_c, n0, nr, t=2, sr_factor=2, up_sample='Deconv', dense=False):
        super(DeepBackProjectionNet, self).__init__()

        self.dense = dense
        self.up_sample_mode = up_sample
        self.T = t
        self.sr_factor = sr_factor
        self.n0 = n0
        self.nr = nr
        self.image_c = image_c

        # ## input
        self.input_conv_0 = nn.Conv2d(self.image_c, self.n0, 3, padding=1)  # ## padding same
        self.input_conv_1 = nn.Conv2d(self.n0, self.nr, 1)

        # ## back projection
        self.up_units = nn.ModuleList()
        self.down_units = nn.ModuleList()

        for i in range(self.T):
            if i and self.dense:
                up_ic = nr * i
            else:
                up_ic = nr
            up_oc = nr

            self.up_units.append(UpProjectionUnit(up_ic, up_oc, self.sr_factor, self.up_sample_mode))
            if i != self.T-1:
                if self.dense:
                    dp_ic = nr * (i + 1)
                else:
                    dp_ic = nr
                dp_oc = nr
                self.down_units.append(DownProjectionUnit(dp_ic, dp_oc, self.sr_factor, self.up_sample_mode))

        # ## reconstruction
        re_ic = nr * self.T
        self.reconstruction = nn.Conv2d(re_ic, image_c, 3, padding=1)  # ##padding same

        # ## activation
        self.activation = nn.PReLU()

        # ## weights initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, lr):
        # ## input feature extraction
        features = self.input_conv_0(lr)
        features = self.activation(features)
        features = self.input_conv_1(features)
        features = self.activation(features)

        # ## back projection
        Hs = []
        Ls = []
        for i in range(self.T):
            if i and self.dense:
                features = torch.cat(Ls, 1)
            features = self.up_units[i](features)
            Hs.append(features)
            if i != self.T-1:
                if self.dense:
                    features = torch.cat(Hs, 1)
                features = self.down_units[i](features)
                Ls.append(features)

        # ## reconstruction
        Hs = torch.cat(Hs, 1)  # ## dim 1 is C
        sr = self.reconstruction(Hs)

        return sr


class DeepBackProjectionNet2(nn.Module):
    """
    Deep back projection Networks for Super-Resolution

    Our final network, D-DBPN, uses conv(3, 256) then conv(1, 64) for the initial feature extraction and t = 7 for
    the back-projection stages. In the reconstruction, we use conv(3, 3). RGB color channels are used for input and
    output image. It takes less than four days to train.

    Reconstruction:
    I.sr = f.rec([H.1, H.2, ..., H.t])
    where f.rec is conv(k=3, oc=image_c)

    Parameters from paper:
        n0 = 256
        nr = 64
        t = 7
    """

    def __init__(self, image_c, n0, nr, t=2, sr_factor=2, up_sample='Deconv', dense=False):
        super(DeepBackProjectionNet2, self).__init__()

        self.dense = dense
        self.up_sample_mode = up_sample
        self.T = t
        self.sr_factor = sr_factor
        self.n0 = n0
        self.nr = nr
        self.image_c = image_c

        # ## input
        self.input_conv_0 = nn.Conv2d(self.image_c, self.n0, 3, padding=1)  # ## padding same
        self.input_conv_1 = nn.Conv2d(self.n0, self.nr, 1)

        # ## back projection
        self.up_units = nn.ModuleList()
        self.down_units = nn.ModuleList()

        for i in range(self.T):
            if i and self.dense:
                dp_ic = nr * (i + 1)
            else:
                dp_ic = nr
            dp_oc = nr

            self.down_units.append(DownProjectionUnit(dp_ic, dp_oc, self.sr_factor, self.up_sample_mode))

            if i and self.dense:
                up_ic = nr * (i + 1)
            else:
                up_ic = nr
            up_oc = nr

            self.up_units.append(UpProjectionUnit(up_ic, up_oc, self.sr_factor, self.up_sample_mode))

        # ## reconstruction
        re_ic = nr * (self.T + 1)
        self.reconstruction = nn.Conv2d(re_ic, image_c, 3, padding=1)  # ##padding same

        # ## activation
        self.activation = nn.PReLU()

        # ## weights initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, lr):
        # ## input feature extraction
        features = self.input_conv_0(lr)
        features = self.activation(features)
        features = self.input_conv_1(features)
        features = self.activation(features)

        # ## back projection
        Hs = [features]
        Ls = []
        for i in range(self.T):
            # # dp
            if self.dense:
                features = torch.cat(Hs, 1)
                ls = self.down_units[i](features)
                Ls.append(ls)
            else:
                ls = self.down_units[i](features)

            if self.dense:
                features = torch.cat(Ls, 1)
                hs = self.up_units[i](features)
                Hs.append(hs)
            else:
                hs = self.up_units[i](ls)
                Hs.append(hs)
            features = hs

        # ## reconstruction
        Hs = torch.cat(Hs, 1)  # ## dim 1 is C
        sr = self.reconstruction(Hs)

        return sr


class DeepBackProjectionNet3(nn.Module):
    """
    Deep back projection Networks for Super-Resolution

    Our final network, D-DBPN, uses conv(3, 256) then conv(1, 64) for the initial feature extraction and t = 7 for
    the back-projection stages. In the reconstruction, we use conv(3, 3). RGB color channels are used for input and
    output image. It takes less than four days to train.

    Reconstruction:
    I.sr = f.rec([H.1, H.2, ..., H.t])
    where f.rec is conv(k=3, oc=image_c)

    Parameters from paper:
        n0 = 256
        nr = 64
        t = 7
    """

    def __init__(self, image_c, n0, nr, t=2, sr_factor=2, up_sample='Deconv', dense=False):
        super(DeepBackProjectionNet3, self).__init__()

        self.dense = dense
        self.up_sample_mode = up_sample
        self.T = t
        self.sr_factor = sr_factor
        self.n0 = n0
        self.nr = nr
        self.image_c = image_c

        # ## input
        self.input_conv_0 = nn.Conv2d(self.image_c, self.n0, 3, padding=1)  # ## padding same
        self.input_conv_1 = nn.Conv2d(self.n0, self.nr, 1)

        # ## back projection
        self.up_units = nn.ModuleList()
        self.down_units = nn.ModuleList()

        for i in range(self.T):
            if i and self.dense:
                dp_ic = nr * (i + 1)
            else:
                dp_ic = nr
            dp_oc = nr

            self.down_units.append(DownProjectionUnit(dp_ic, dp_oc, self.sr_factor, self.up_sample_mode))

            if i and self.dense:
                up_ic = nr * (i + 1)
            else:
                up_ic = nr
            up_oc = nr

            self.up_units.append(UpProjectionUnit(up_ic, up_oc, self.sr_factor, self.up_sample_mode))

        # ## reconstruction
        re_ic = nr * (self.T + 1)
        self.reconstruction = nn.Conv2d(re_ic, image_c, 3, padding=1)  # ##padding same

        # ## activation
        self.activation = nn.PReLU()

        # ## weights initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, lr):
        # ## input feature extraction
        features = self.input_conv_0(lr)
        features = self.activation(features)
        features = self.input_conv_1(features)
        features = self.activation(features)

        # ## back projection
        Hs = [features]
        Ls = []
        for i in range(self.T):
            # # dp
            if self.dense:
                features = torch.cat(Hs, 1)
                ls = self.down_units[i](features)
                Ls.append(ls)
            else:
                ls = self.down_units[i](features)

            if self.dense:
                features = torch.cat(Ls, 1)
                hs = self.up_units[i](features)
                Hs.append(hs)
            else:
                hs = self.up_units[i](ls)
                Hs.append(hs)
            features = hs

        # ## reconstruction
        Hs = torch.cat(Hs, 1)  # ## dim 1 is C
        sr = self.reconstruction(Hs)

        sr += lr

        return sr


def conv_get_size(input_size, kernel_size=3, stride=1, padding=0, dilation=1):
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    output_h = int((input_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    output_w = int((input_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[0] + 1)
    return output_h, output_w


def conv_with_size(input_size, ic, oc, k=3, s=1, p=0, d=1, bias=True):
    size = conv_get_size(input_size, k, s, p, d)
    return nn.Conv2d(ic, oc, k, s, p, d, bias=bias), size


def deconv_get_size(input_size, kernel_size=3, stride=1, padding=0, out_padding=0, dilation=1):
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    output_h = int((input_size[0] - 1) * stride[0] - 2 * padding[0] + kernel_size[0] + out_padding)
    output_w = int((input_size[1] - 1) * stride[1] - 2 * padding[1] + kernel_size[1] + out_padding)
    return output_h, output_w


def deconv_with_size(input_size, ic, oc, k=3, s=1, p=0, d=1, bias=True):
    size = deconv_get_size(input_size, k, s, p, d)
    return nn.ConvTranspose2d(ic, oc, k, s, p, bias=bias, dilation=d), size

