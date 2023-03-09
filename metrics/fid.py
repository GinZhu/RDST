import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from metrics.fid_inception import InceptionV3


class FID(object):

    def __init__(self, gpu_id=-1, inception_block_idx=3, batch_size=16):
        """
        Calculates the Frechet Inception Distance (FID) to evalulate GANs
        The FID metric calculates the distance between two distributions of images.
        Typically, we have summary statistics (mean & covariance matrix) of one
        of these distributions, while the 2nd distribution is given by a GAN.
        When run as a stand-alone program, it compares the distribution of
        images that are stored as PNG/JPEG at a specified location with a
        distribution given by summary statistics (in pickle format).
        The FID is calculated by assuming that X_1 and X_2 are the activations of
        the pool_3 layer of the inception net for generated samples and real world
        samples respectively.
        See --help to see further details.
        Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
        of Tensorflow
        Copyright 2018 Institute of Bioinformatics, JKU Linz
        Licensed under the Apache License, Version 2.0 (the "License");
        you may not use this file except in compliance with the License.
        You may obtain a copy of the License at
           http://www.apache.org/licenses/LICENSE-2.0
        Unless required by applicable law or agreed to in writing, software
        distributed under the License is distributed on an "AS IS" BASIS,
        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        See the License for the specific language governing permissions and
        limitations under the License.
        :param gpu_id: if -1 by cpu (default), otherwise using one specific GPU
        :param inception_block_idx: by default 3: which activation layer of Inception Net will be used;
        :param batch_size: 16 (default), modify this to fit GPU's memory limitation.
        """

        if gpu_id >= 0:
            self.device = torch.device('cuda:{}'.format(gpu_id))
        else:
            self.device = torch.device('cpu')

        self.batch_size = batch_size

        self.model = InceptionV3([inception_block_idx])
        self.model = self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)

    @staticmethod
    def numpy_2_tensor(a):
        if isinstance(a, list):
            a = np.array(a)
        if a.ndim == 3:
            return torch.tensor(a.transpose(2, 0, 1), dtype=torch.float)
        elif a.ndim == 4:
            return torch.tensor(a.transpose(0, 3, 1, 2), dtype=torch.float)
        else:
            raise ValueError('Image should have 3 or 4 channles')

    def _compute_statistics_of_imgs(self, imgs):

        pred_acts = []

        for i in range(0, len(imgs), self.batch_size):
            batch_imgs = imgs[i:i+self.batch_size]

            with torch.no_grad():
                acts = self.model(batch_imgs)[0]

            if acts.size(2) != 1 or acts.size(3) != 1:
                acts = adaptive_avg_pool2d(acts, output_size=(1, 1))

            pred_acts.append(acts.cpu().data.numpy().reshape(acts.size(0), -1))

        pred_acts = np.concatenate(pred_acts, axis=0)

        mu = np.mean(pred_acts, axis=0)
        sigma = np.cov(pred_acts, rowvar=False)

        return mu, sigma

    def __call__(self, imgs_1, imgs_2):
        """
        Caculate the FID distance between two image sets, either of which could be:
            1. Pytorch Tensor: should be (N * C * H * W);
            2. Numpy array: should be (N * H * W * C);
        N could be 1 if there is one GT / Pred image pair.
        The data range should be [0, 1]

        If C == 3 (e.g. RGB images), the function performs by default;
        otherwise, each channel will be expanded to 3 channels and the channel-wise FID score will be
        calculated separately then the mean of FID of all channels will be returned.

        :param imgs_1: numpy array or tensor
        :param imgs_2: numpy array or tensor
        :return: FID
        """
        # imgs are
        imgs_1 = self.prepare_data(imgs_1)
        imgs_2 = self.prepare_data(imgs_2)

        fid_scores = []

        for img1, img2 in zip(imgs_1, imgs_2):

            m1, s1 = self._compute_statistics_of_imgs(img1)
            m2, s2 = self._compute_statistics_of_imgs(img2)

            fid_scores.append(self.calculate_frechet_distance(m1, s1, m2, s2))

        return np.mean(fid_scores)

    def prepare_data(self, imgs):
        if isinstance(imgs, np.ndarray):
            assert imgs.ndim == 4, 'images should have 4 dimensions (N H W C) as numpy array'
            if imgs.shape[-1] == 3:
                imgs = self.numpy_2_tensor(imgs)
                imgs = imgs.to(self.device)
                return [imgs]
            else:
                channels = []
                for c in range(imgs.shape[-1]):
                    one_channel = imgs[:, :, :, c]
                    one_channel = np.stack([one_channel]*3, axis=-1)
                    channels.append(self.numpy_2_tensor(one_channel).to(self.device))
                return channels

        elif isinstance(imgs, torch.Tensor):
            assert imgs.ndim == 4, 'images should have 4 dimensions (N C H W) as a Tensor'
            if imgs.size(1) == 3:
                imgs = imgs.to(self.device)
                return [imgs]
            else:
                channels = []
                for c in range(imgs.size(1)):
                    one_channel = imgs[:, c, :, :]
                    one_channel = torch.stack([one_channel]*3, dim=1)
                    channels.append(one_channel.to(self.device))
            return channels

