from loss.basic_loss import BasicLoss
from loss.adversarial import ScaleAdversarial
from loss.vgg import VGG
from loss.esrgan_vgg.minc_vgg_loss import MincVGG
from torch import nn

from loss.seg_unet import SegUNet_F


class SRLoss(BasicLoss):

    def __init__(self, paras):
        super(SRLoss, self).__init__(paras)

        self.use_seg_loss_flag = False

        for l in self.training_loss_names:
            if l in ['L1', 'L2', 'MSE']:
                f = RecLoss(l)
            elif l in ['VGG22', 'VGG54']:
                f = VGG(l[3:]).to(self.device)
            elif l in ['Minc_VGG22', 'Minc_VGG54']:
                minc_vgg_model_path = paras.minc_vgg_model_path
                f = MincVGG(mode=l, pre_activation=True, model_path=minc_vgg_model_path).to(self.device)
            elif 'GAN' in l:
                f = ScaleAdversarial(paras, self.device)

            # add segmentation loss
            elif l in ['UNet-F']:
                self.use_seg_loss_flag = True
                f = SegUNet_F(paras.unet_loss_layers, paras.unet_loss_mode).to(self.device)
            self.loss_components += f.loss_names
            self.loss_functions[l] = f

    def __call__(self, pred, gt, sr_scales=None, gt_label=None):
        repo = {}
        scalars = self.training_loss_scalars[self.current_training_state]
        loss = 0.
        for n in scalars:
            s = scalars[n]
            loss_function = self.loss_functions[n]
            if 'GAN' in n:
                l, r = loss_function(pred, gt, sr_scales)
            elif 'UNet' in n:
                l, r = loss_function(pred, gt, gt_label)
            else:
                l, r = loss_function(pred, gt)
            for k in r:
                repo[k] = r[k]
            loss += l * s
        return loss, repo

    def apply(self, fn):
        for l in self.loss_functions:
            if 'GAN' in l:
                f = self.loss_functions[l]
                f.apply(fn)


class RecLoss(object):

    def __init__(self, type='L1'):
        if type is 'L1':
            self.loss_names = ['Rec_L1']
            self.function = nn.L1Loss()
        elif type in ['L2', 'MSE']:
            self.loss_names = ['Rec_MSE']
            self.function = nn.MSELoss()

    def __call__(self, rec, gt):
        loss = self.function(rec, gt)
        return loss, {self.loss_names[0]: loss.item()}
