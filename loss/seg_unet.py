from networks.common import MeanShift

import torch
import torch.nn as nn
import torch.nn.functional as F

from os.path import exists

import segmentation_models_pytorch as smp


class SegUNet_F(nn.Module):
    """
    loss_mode:
        label -> dice loss of segmentation results
    """
    def __init__(self, loss_layers, mode='OASIS'):
        super(SegUNet_F, self).__init__()

        unet_path = None
        in_channels = None
        classes = None
        if 'OASIS' in mode:
            unet_path = 'loss/unet_oasis.pt'
            in_channels = 1
            classes = 4
        elif 'BraTS' in mode:
            unet_path = 'loss/unet_brats.pt'
            in_channels = 4
            classes = 4
        elif 'ACDC' in mode:
            unet_path = 'loss/unet_acdc.pt'
            in_channels = 1
            classes = 4
        elif 'COVID' in mode:
            unet_path = 'loss/unet_covid.pt'
            in_channels = 1
            classes = 4

        dice_classes = [0, 1, 2, 3]
        if 'tumor_only' in mode:
            dice_classes = [1, 2, 3]
        if 'lesion_only' in mode:
            dice_classes = [1, 2, 3]

        unet = smp.Unet(in_channels=in_channels, classes=classes)
        if not exists(unet_path):
            raise ValueError('Pre-trained UNet not exist: {}'.format(unet_path))
        unet.load_state_dict(torch.load(unet_path, map_location='cpu'))

        for k in loss_layers:
            self.loss_mode = k
        self.loss_layers = loss_layers[self.loss_mode]

        self.encoder = unet.encoder
        self.decoder = unet.decoder
        self.tail = unet.segmentation_head

        self.encoder.requires_grad = False
        self.decoder.requires_grad = False
        self.tail.requires_grad = False

        self.loss_names = ['SegUNet({})'.format(self.loss_mode)]

        # reflection padding
        # padding (96, 96) to (160, 128), padding -> (32, 32, 16, 16)
        self.padding = nn.ReflectionPad2d((16, 16, 32, 32))
        self.padding_flag = False

        # dice loss if necessary
        if 'label' in self.loss_mode:
            self.loss = smp.losses.DiceLoss('multiclass', dice_classes)
        elif 'L1' in self.loss_mode:
            self.loss = torch.nn.MSELoss()
        elif 'L2' in self.loss_mode:
            self.loss = torch.nn.L1Loss()
        else:
            self.loss = torch.nn.L1Loss()

    def unet_forward(self, x):
        # padding sr/hr to [160, 128], as the trained UNet
        if self.padding_flag:
            x = self.padding(x)
        features = self.encoder(x)
        if 'encoder' in self.loss_mode:
            return features
        decoder_output = self.decoder(*features)
        if 'decoder' in self.loss_mode:
            return decoder_output
        label = self.tail(decoder_output)
        if self.loss_mode in ['label-hr', 'label-gt']:
            return label

    def forward(self, sr, hr, gt_label=None):

        assert sr.shape == hr.shape, 'Seg UNet Loss invalid SR({}) and HR({}) shape!'.format(
            sr.shape, hr.shape
        )

        sr_features = self.unet_forward(sr)
        if 'encoder' in self.loss_mode:
            with torch.no_grad():
                hr_features = self.unet_forward(hr)
            loss = 0
            for l in self.loss_layers:
                loss += self.loss(sr_features[l], hr_features[l])
                loss /= len(self.loss_layers)
        elif 'decoder' in self.loss_mode:
            with torch.no_grad():
                hr_features = self.unet_forward(hr)
            loss = self.loss(sr_features, hr_features)
        elif self.loss_mode == 'label-hr':
            with torch.no_grad():
                hr_label = self.unet_forward(hr)
            hr_label = torch.argmax(hr_label, dim=1)
            loss = self.loss(sr_features, hr_label)
        elif self.loss_mode == 'label-gt':
            if gt_label.dim() == 4:
                gt_label = gt_label[:, 0]
            gt_label = gt_label.to(torch.long)
            if self.padding_flag:
                gt_label = self.padding(gt_label)
            loss = self.loss(sr_features, gt_label)
        else:
            raise ValueError('Invalid UNet Seg Loss Mode: {}'.format(self.loss_mode))

        return loss, {self.loss_names[0]: loss.item()}


