from collections import deque
from pytorch_wavelets import DWTForward, DWTInverse
import torch
from torch import nn


class PytorchDWT(nn.Module):
    def __init__(self, level, kernel='haar', mode='full'):
        """

        :param level:
        :param kernel:
        :param mode:
                full -> all coeffs will be decomposed, so all tokens are the same size
                part -> only the left-top coeff is decomposed, so the tokens are with various sizes
        """
        super(PytorchDWT, self).__init__()
        # valid_kernels = ['haar', 'db1']
        # assert kernel in valid_kernels, 'Wavelet Kernel {} not supported. Should be {}.'.format(kernel, valid_kernels)
        self.level = level
        self.mode = mode
        self.kernel = kernel

        if self.mode == 'full':
            self.xfm = DWTForward(J=1, wave=self.kernel)
        elif self.mode == 'part':
            self.xfm = DWTForward(J=self.level, wave=self.kernel)

    def forward(self, x):
        """

        :param x: N x C x H x W tensor as a batch of images
        :return:
            'full': N x P x C x h x w
            'part': [N x 1 x C x hn x wn,
                     N x 3 x C x hn x wn,
                     ...,
                     N x 3 x C x h1 x w1]
        """
        if self.mode == 'full':
            return self.__full_dwt2__(x)
        elif self.mode == 'part':
            return self.__part_dwt2__(x)
        return x

    def __full_dwt2__(self, image):
        coeffs = deque([image])
        for i in range(self.level):
            for _ in range(len(coeffs)):
                img = coeffs.popleft()
                cl, ch = self.xfm(img)
                coeffs.append(cl)
                for c in torch.unbind(ch[0], dim=2):
                    coeffs.append(c)
        coeffs = torch.stack(list(coeffs), dim=1)
        return coeffs

    def __part_dwt2__(self, image):
        coeffs = self.xfm(image)
        tokens = [coeffs[0].unsqueeze(1)]
        for c in coeffs[1][::-1]:
            tokens.append(torch.transpose(c, 1, 2))
        return tokens


class PytorchDWTInverse(nn.Module):
    def __init__(self, kernel='haar', mode='full'):
        """

        :param level:
        :param kernel:
        :param mode:
                full -> all coeffs will be decomposed, so all tokens are the same size
                part -> only the left-top coeff is decomposed, so the tokens are with various sizes
        """
        super(PytorchDWTInverse, self).__init__()
        # valid_kernels = ['haar', 'db1']
        # assert kernel in valid_kernels, 'Wavelet Kernel {} not supported. Should be {}.'.format(kernel, valid_kernels)
        self.mode = mode
        self.kernel = kernel

        self.ifm = DWTInverse(wave=self.kernel)

    def forward(self, x):
        """

        :param x:
            'full': N x P x C x h x w
            'part': [N x 1 x C x hn x wn,
                     N x 3 x C x hn x wn,
                     ...,
                     N x 3 x C x h1 x w1]
        :return: N x C x H x W tensor as a batch of images
        """
        if self.mode == 'full':
            return self.__full_idwt2__(x)
        elif self.mode == 'part':
            return self.__part_idwt2__(x)
        return x

    def __full_idwt2__(self, tokens):
        tokens = deque(torch.unbind(tokens, 1))
        while len(tokens) > 1:
            ca = tokens.popleft()
            ch = tokens.popleft()
            cv = tokens.popleft()
            cd = tokens.popleft()
            coeffs = [ca, [torch.stack([ch, cv, cd], dim=2)]]
            tokens.append(self.ifm(coeffs))
        return tokens[0]

    def __part_idwt2__(self, tokens):
        coeffs = [tokens[0][:, 0], []]
        for t in tokens[:0:-1]:
            coeffs[1].append(torch.transpose(t, 1, 2))
        return self.ifm(coeffs)
