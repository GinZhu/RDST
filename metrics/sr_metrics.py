from sewar.full_ref import mse, rmse, rmse_sw, uqi, ergas, scc, rase, sam, vifp, psnrb
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from metrics.fid import FID


def psnr(GT, P):
    return peak_signal_noise_ratio(GT, P, data_range=1)


def ssim(GT, P):
    return structural_similarity(GT, P, data_range=1, multichannel=True)


class SRMetrics(object):
    """
    Example useage:
        import numpy as np
        a = np.random.rand(64, 256, 256, 1)
        b = np.random.rand(64, 256, 256, 1)
        sr_metrics = SRMetrics('mse psnr ssim rmse rmse_sw uqi ergas scc rase sam vifp psnrb fid', 0, 'mean')
        repo = sr_metrics(a, b)
        print(repo.keys())
        for k in repo:
            print(k, repo[k])
    """
    def __init__(self, metrics='', gpu_id=-1, return_mode='full', fid_paras=[3, 16],):
        self.metrics = metrics.split()
        self.functions = {}
        self.fid_functions = {}

        fid_gpu = gpu_id
        fid_block_idx, fid_batch_size = fid_paras

        pixel_based_metrics = ['mse', 'rmse', 'rmse_sw', 'psnr', 'uqi', 'ssim', 'ergas', 'scc', 'rase', 'sam', 'msssim',
                               'vifp', 'psnrb']

        cnn_based_metrics = ['fid']

        for m in self.metrics:
            if m in pixel_based_metrics:
                self.functions[m] = eval(m)
            elif m == 'fid':
                self.fid_functions[m] = FID(fid_gpu, fid_block_idx, fid_batch_size)
            else:
                raise ValueError('Do not support this metric: {}'.format(m))

        self.margin = 0
        if return_mode not in ['full', 'mean']:
            raise ValueError('return mode must be one of [mean, full]')
        self.return_mode = return_mode

    def __call__(self, gts, preds, margin=0):
        """
        Calculate various metrics for SR task.
        :param gts: groud truth high resolution images, could be tensor / array with 3 or 4 dims (one or more images)
        :param preds: Generated super-resolution images.
        :return:
            reports: a dict of {'metric': [score mean, score std] / 'FID': [score]};
            records: image-wise score (no fid) as a dictionary; If self.records == False then return {}
        """
        self.margin = margin
        gts = self.prepare_data(gts)
        preds = self.prepare_data(preds)

        reports = {}
        for m in self.functions:
            func = self.functions[m]
            scores = []
            for g, p in zip(gts, preds):
                s = func(g, p)
                if m in ['rmse_sw']:
                    s = s[0]
                scores.append(s)
            reports[m] = scores

        # ## fid based metrics
        for m in self.fid_functions:
            func = self.fid_functions[m]
            score = func(gts, preds)
            reports[m] = [score]

        if self.return_mode == 'mean':
            for m in reports:
                reports[m] = np.mean(reports[m])
        return reports

    def prepare_data(self, imgs):
        """
        Input images could be in either a Tensor or a numpy array, could be either H W C or N H W C.
        :param imgs: Tensor / Array, 3 or 4 dimensions
        :return: a list of numpy array images with format (H W C)
        """
        if isinstance(imgs, (list, tuple)):
            if isinstance(imgs[0], torch.Tensor):
                imgs = torch.stack(imgs)
            elif isinstance(imgs[0], np.ndarray):
                imgs = np.stack(imgs)

        assert imgs.ndim == 3 or imgs.ndim == 4, 'images should have 3 or 4 dimensions'

        if isinstance(imgs, torch.Tensor):
            imgs = self.tensor_2_numpy(imgs)

        assert isinstance(imgs, np.ndarray), 'Images should be either numpy array or PyTorch Tensor.'

        H, W = imgs.shape[-3:-1]

        if imgs.ndim == 4:
            imgs = imgs[:, self.margin:H - self.margin, self.margin:W - self.margin, :]
            return imgs
        elif imgs.ndim == 3:
            imgs = imgs[self.margin:H - self.margin, self.margin:W - self.margin, :]
            return [imgs]

    @staticmethod
    def numpy_2_tensor(a):
        if isinstance(a, list):
            a = np.array(a)
        if a.ndim == 3:
            return torch.tensor(a.transpose(2, 0, 1), dtype=torch.float)
        elif a.ndim == 4:
            return torch.tensor(a.transpose(0, 3, 1, 2), dtype=torch.float)
        else:
            raise ValueError('Images should be either (H, W, C) or (N, H, W, C)')

    @staticmethod
    def tensor_2_numpy(t):
        if t.ndim == 3:
            return t.detach().cpu().numpy().transpose(1, 2, 0)
        elif t.ndim == 4:
            return t.detach().cpu().numpy().transpose(0, 2, 3, 1)
        else:
            raise ValueError('Images should be either (C, H, W) or (N, C, H, W)')
