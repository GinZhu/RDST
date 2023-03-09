from loss.wgan import Discriminator
from utils.optim import make_scheduler, make_optimizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from loss.trans_wgan import make_STD


class ScaleAdversarial(nn.Module):
    """
    paras: hyper-parameters from the config file
    device: torch device if None will be cpu
    pre_trained_model: a state_dict of the same discriminator

    Behaviours:
        1. loss_names
        2. return [loss, loss_names, loss_item]

    ScaleGAN:

    """
    def __init__(self, paras, device=None):
        if device is None:
            device = torch.device('cpu')
        super(ScaleAdversarial, self).__init__()
        self.gan_type = paras.gan_type
        self.gan_k = paras.gan_k
        self.wgan_clip_value = paras.wgan_clip_value

        if 'ST' in self.gan_type or 'st' in self.gan_type:
            self.discriminator = make_STD(paras)
        else:
            self.discriminator = Discriminator(paras)
        # if pre_trained_model is not None:
        #     ptm = torch.load(pre_trained_model)
        #     if 'discriminator' in ptm:
        #         self.discriminator.load_state_dict(ptm['discriminator'])
        #     else:
        #         self.discriminator.load_state_dict(ptm)
        self.discriminator = self.discriminator.to(device)
        if 'GP' in self.gan_type:
            self.optimizer = optim.Adam(
                self.discriminator.parameters(),
                betas=(0, 0.9), eps=1e-8, lr=1e-5
            )
        else:
            self.optimizer = make_optimizer(paras, self.discriminator)

        self.scheduler = make_scheduler(paras, self.optimizer)

        self.loss_names = ['Adv_G', 'Adv_D', 'Adv_D Real', 'Adv_D Fake']

        self.l1 = nn.L1Loss()

    def forward(self, fake, real, scales=None):
        fake_detach = fake.detach()

        loss_d_item = 0
        loss_d_real_item = 0
        loss_d_fake_item = 0
        for _ in range(self.gan_k):
            self.optimizer.zero_grad()
            d_fake = self.discriminator(fake_detach)
            d_real = self.discriminator(real)

            # ## scale gan
            if 'ScaleGAN' in self.gan_type:
                label_real = torch.ones_like(d_real)
                label_fake = 1. / scales
                loss_d_real = self.l1(d_real, label_real)
                loss_d_fake = self.l1(d_fake, label_fake)

            elif 'WGAN' in self.gan_type:
                loss_d_fake = d_fake.mean()
                loss_d_real = - d_real.mean()
                # loss_d = (d_fake - d_real).mean()
            else:
                label_fake = torch.zeros_like(d_fake)
                label_real = torch.ones_like(d_real)
                # loss_d = loss_d_fake + loss_d_real
                # ## RaGAN:
                if 'RaGAN' in self.gan_type:
                    loss_d_fake = F.binary_cross_entropy_with_logits(
                        d_fake - d_real.mean(), label_fake
                    )
                    loss_d_real = F.binary_cross_entropy_with_logits(
                        d_real - d_fake.mean(), label_real
                    )
                else:
                    loss_d_fake = F.binary_cross_entropy_with_logits(d_fake, label_fake)
                    loss_d_real = F.binary_cross_entropy_with_logits(d_real, label_real)

            loss_d = loss_d_fake + loss_d_real

            # gradient penalty
            if 'GP' in self.gan_type:
                alpha = torch.rand_like(fake[:, 0, 0, 0]).view(-1, 1, 1, 1)
                alpha = alpha.expand_as(fake)
                hat = fake_detach.mul(1 - alpha) + real.mul(alpha)
                hat.requires_grad = True
                d_hat = self.discriminator(hat)
                gradients = torch.autograd.grad(
                    outputs=d_hat.sum(), inputs=hat,
                    retain_graph=True, create_graph=True, only_inputs=True
                )[0]
                gradients = gradients.view(gradients.size(0), -1)
                gradient_norm = gradients.norm(2, dim=1)
                gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
                loss_d += gradient_penalty

            # Discriminator update
            loss_d_fake_item += loss_d_fake.item()
            loss_d_real_item += loss_d_real.item()
            loss_d_item += loss_d.item()
            loss_d.backward()
            self.optimizer.step()

            if self.gan_type == 'WGAN':
                for p in self.discriminator.parameters():
                    p.data.clamp_(-self.wgan_clip_value, self.wgan_clip_value)

        loss_d_item /= self.gan_k
        loss_d_fake_item /= self.gan_k
        loss_d_real_item /= self.gan_k

        # ## Generator loss
        d_fake_for_g = self.discriminator(fake)
        if self.gan_type == 'GAN':
            label_real = torch.ones_like(d_fake_for_g)
            # ## sigmoid is implemented inside the loss function
            loss_g = F.binary_cross_entropy_with_logits(
                d_fake_for_g, label_real
            )
        elif 'RaGAN' in self.gan_type:
            label_real = torch.ones_like(d_fake_for_g)
            d_real = self.discriminator(real)
            label_fake = torch.zeros_like(d_real)
            loss_g = (F.binary_cross_entropy_with_logits(
                d_fake_for_g - d_real.mean(), label_real
            ) + F.binary_cross_entropy_with_logits(
                d_real - d_fake_for_g.mean(), label_fake
            )) / 2

        elif 'WGAN' in self.gan_type:
            loss_g = -d_fake_for_g.mean()

        elif 'ScaleGAN' in self.gan_type:
            label_g_fake = torch.ones_like(d_fake_for_g)
            loss_g = self.l1(label_g_fake, d_fake_for_g)

        if self.scheduler is not None:
            self.scheduler.step()

        all_loss_items = [loss_g.item(), loss_d_item, loss_d_real_item, loss_d_fake_item]

        all_loss_items = {n: item for n, item in zip(self.loss_names, all_loss_items)}
        return loss_g, all_loss_items

    def state_dict(self, *args, **kwargs):
        state_discriminator = self.discriminator.state_dict(*args, **kwargs)
        state_optimizer = self.optimizer.state_dict()
        if self.scheduler is not None:
            state_scheduler = self.scheduler.state_dict()
        else:
            state_scheduler = None

        state = {
            'discriminator': state_discriminator,
            'optimizer': state_optimizer,
            'scheduler': state_scheduler,
        }

        return state

    def load_state_dict(self, state_dict, strict: bool = True):
        state_discriminator = state_dict['discriminator']
        state_optimizer = state_dict['optimizer']
        report = self.discriminator.load_state_dict(state_discriminator, strict)
        self.optimizer.load_state_dict(state_optimizer)
        if self.scheduler and state_dict['scheduler']:
            self.scheduler.load_state_dict(state_dict['scheduler'])
        return report

