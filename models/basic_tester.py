from abc import ABC, abstractmethod
from os.path import isdir, join, exists
from os import makedirs
from skimage import io
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch

from datetime import datetime, timedelta
import time


"""
todo: fill in this part with an example

@Jin (jin.zhu@cl.cam.ac.uk) June 22 2020
"""


class BasicTester(ABC):
    """
    To use:
        tester = BasicTester(paras)
        tester.setup()
        tester.evaluation(id, DS)
    """

    def __init__(self, paras):
        super(BasicTester, self).__init__()

        # ## basic information
        self.paras = paras
        self.name = paras.model_name
        self.verbose = paras.verbose
        if paras.gpu_id == -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:{}'.format(paras.gpu_id))

        # ## training precision
        self.precision = paras.precision

        # ## dirs for output
        self.output_dir = join(paras.output_dir, 'Final_Predictions')
        self.plots_dir = None
        self.inferences_dir = None
        self.reports_dir = None
        self.testing_log = None

        # ## well-trained model
        self.model_names = []
        self.ptm_paths = {}

        # ## evaluation
        self.test_sr_factors = paras.sr_scales_for_final_testing
        self.eva_func = None

        # ## save results
        self.save_gts = False
        self.sr_factors_for_saving = paras.sr_scales_for_saving

    def _creat_dirs(self):
        self.output_dir = join(self.output_dir, self.name)

        # creat some dirs for output
        self.output_dir = self.exist_or_make(self.output_dir)

        self.plots_dir = self.exist_or_make(join(
            self.output_dir, 'plots'))
        self.inferences_dir = self.exist_or_make(join(
            self.output_dir, 'inferences'))
        self.reports_dir = self.exist_or_make(join(
            self.output_dir, 'reports'
        ))

        # testing log
        self.testing_log = join(self.output_dir, 'testing_log.txt')

    def setup(self):
        """
        This function must be called first
        """
        # creat dirs
        self._creat_dirs()

        # save paras for this experiment
        plog = self.fancy_print(
            'Prediction starts @ {}, with paras:'.format(self.current_time())
        )
        plog += '\n' + str(self.paras) + '\n\n\n'

        # load well-trained model
        for m in self.model_names:
            ptm_path = self.ptm_paths[m]
            ptm = torch.load(ptm_path, map_location=self.device)
            self.__getattribute__(m).load_state_dict(ptm)
            plog += self.fancy_print('Well trained model {}:'.format(m))
            plog += str(self.__getattribute__(m)) + '\n'

        self.write_log(plog)

    def inference(self, D, return_sample=False):
        # given a Dataset, do inference on it
        preds = []
        ori_samples = []
        for i in range(D.test_len()):
            s = D.get_test_pair(i)
            preds.append(self.__inference_one__(s))
            if return_sample:
                ori_samples.append(s)
        if return_sample:
            return preds, ori_samples
        return preds

    @abstractmethod
    def __inference_one__(self, sample):
        # given a data sample, do inference on it and return
        pass

    @abstractmethod
    def test(self):
        """
        Do testing on testing datasets
        :return:
        """
        pass

    @abstractmethod
    def get_gt_images(self, samples):
        """
        Extract GT images in samples and return.
        :param samples: a list of test samples
        :return: GT images, the format should be as the same as rec_imgs
        """
        pass

    def select_images_to_save(self, imgs):
        """
        Select only part of the rec_imgs and gt_imgs to be stored.
        :param imgs: rec_imgs or gt_imgs
        :return: subset of input images
        """
        return imgs

    def evaluation(self, case_name, DS):
        eva_report_file = join(self.reports_dir, '{}_eva_reports.tar'.format(case_name))
        if exists(eva_report_file):
            record = torch.load(eva_report_file)
            eva_report = record['eva_report']
            plog = self.fancy_print('Loading evaluation results of {}'.format(case_name))
            plog += '\nEva results loaded from {}'.format(eva_report_file)
            inference_time_cost_float = record['inference_time_cost']
        else:
            # ## inference
            inference_result_path = join(self.inferences_dir, '{}_inference_results.tar'.format(case_name))
            inference_start_time = self.current_time('float')
            rec_imgs, ori_samples = self.inference(DS, return_sample=True)
            inference_time_cost, inference_time_cost_float = self.time_cost(self.current_time('float') - inference_start_time)
            # save only part of images
            inference_result = {
                'rec_imgs': self.select_images_to_save(rec_imgs),
                'inference_cost': inference_time_cost_float,
            }
            if self.save_gts:
                gt_imgs = self.get_gt_images(ori_samples)
                inference_result['gt_imgs'] = self.select_images_to_save(gt_imgs)
            torch.save(inference_result, inference_result_path)

            # evaluation
            eva_report = self.eva_func(rec_imgs, ori_samples)

            flag = self.fancy_print('{} inference costs {}'.format(case_name, inference_time_cost))
            plog = flag + '\n' + self.eva_func.print(eva_report)

            # save evaluation results
            eva_results = {
                'inference_time_cost': inference_time_cost_float,
                'id': case_name,
                'eva_report': eva_report,
                'Time': self.current_time()
            }
            torch.save(eva_results, join(self.reports_dir, '{}_eva_reports.tar'.format(case_name)))
            plog += '\n{} eva results save to {}'.format(case_name, self.reports_dir)

        self.write_log(plog)

        return eva_report, inference_time_cost_float

        # self.eva_func.plot_final_evas(
        #     eva_report,
        #     self.plots_dir,
        #     prefix
        # )

    @staticmethod
    def stack_eva_reports(reports):
        # stack each element in eva_report separately
        stacked_report = {}
        for k in reports[0].keys():
            stacked_report[k] = []
            for r in reports:
                stacked_report[k].append(r[k])
        return stacked_report

    @staticmethod
    def exist_or_make(path):
        if not isdir(path):
            makedirs(path)
        return path

    @staticmethod
    def save_images(path, imgs, N_R=None, single_img=False):
        if single_img:
            if imgs.ndim == 3 and imgs.shape[2] == 1:
                io.imsave(path, imgs[:, :, 0])
            else:
                io.imsave(path, imgs)
        else:
            if isinstance(imgs, list):
                imgs = np.stack(imgs, axis=0)

            if imgs.ndim == 4 and imgs.shape[3] == 1:
                imgs = imgs[:, :, :, 0]
            num = imgs.shape[0]
            # try to re-arrange the imgs similar as a square
            if N_R is None:
                N_R = int(np.sqrt(num))

            if num % N_R:
                N_C = np.floor(num / N_R)
                sub_imgs_without_lastline = np.array_split(imgs[:N_R * N_C], N_C)

                # fill the last line with blank imgs
                last_line = imgs[N_R * N_C:]
                lack_num = int(N_R - last_line.shape[0])
                blank_img = np.zeros_like(last_line[0], dtype=last_line[0].dtype)
                blank_imgs = np.stack([blank_img] * lack_num, axis=0)
                last_line = np.concatenate([last_line, blank_imgs], axis=0)
                sub_imgs = sub_imgs_without_lastline + [last_line]
            else:
                sub_imgs = np.array_split(imgs, N_R)

            merged_img = np.concatenate([np.concatenate(_, axis=1) for _ in sub_imgs], axis=0)

            # clip
            merged_img = np.clip(merged_img, 0., 1.)

            io.imsave(path, (merged_img * 255).astype('uint8'))

            if merged_img.ndim == 2:
                merged_img = merged_img[:, :, np.newaxis]

            return merged_img

    def prepare(self, *args):
        """
        Move tensors to device, and change che precision.
        If para is not a tensor, nothing will happen
        :param args: a list of tensors (or not)
        :return: a list of processed tensor
        """
        tensors = []
        for t in args:
            if isinstance(t, torch.Tensor):
                # todo: add self.precision convert function
                t = t.to(self.device)
            tensors.append(t)
        # for t in args:
        #     if self.precision == 'half':
        #         t = t.half()
        #     tensors.append(t.to(self.device))
        if len(tensors) == 1:
            tensors = tensors[0]
        if len(tensors) == 0:
            tensors = None
        return tensors

    @staticmethod
    def fancy_print(m):
        l = len(m)
        return '#' * (l + 50) + '\n' + '#' * 5 + ' ' * 20 + m + ' ' * 20 + '#' * 5 + '\n' + '#' * (l + 50)

    @staticmethod
    def tensor_2_numpy(t):
        """
        :param t: a tensor, either on GPU or CPU
        :return: a numpy array
            if t is a stack of images (t.dim() == 3 or 4), it will transpose the channels
            if not, will return t as the same shape.
        """
        if t.dim() == 3:
            return t.detach().cpu().numpy().transpose(1, 2, 0)
        elif t.dim() == 4:
            return t.detach().cpu().numpy().transpose(0, 2, 3, 1)
        else:
            return t.detach().cpu().numpy()

    def write_log(self, plog):
        if self.verbose:
            print(plog)
        with open(self.testing_log, 'a') as f:
            f.write(plog + '\n')

    def plot_final_results(self):
        # ## basic: plot the results
        pass

    @staticmethod
    def current_time(mode='str'):
        """
        Return current system time.
        :param mode:
                str: return a string to print, current date and time;
                float: return time.time(), for time cost
        :return: a str or a float, depends on mode
        """
        if mode == 'str':
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if mode == 'float':
            return time.time()

    @staticmethod
    def time_cost(t):
        """
        To convert the time cost to day - hour - minute - second mode
        :param t: seconds
        :return: a string
        """
        return str(timedelta(seconds=int(t))), t

    def training_complete(self):
        """
        This function should be called at the end or .train(), to:
            1. save necessary records, such as loss, time costs, validation performances;
            2. print a summary of training;
        :return: None
        """
        summary = {
            'Time': self.current_time(),
            'training_loss_records': self.training_loss_records,
            'quick_validation_reports': self.quick_validation_reports,
            'training_epoch_costs': self.training_epoch_costs,
        }
        torch.save(summary, join(self.inferences_dir, 'training_records.tar'))

        plog = self.fancy_print(
            '{} training completed @ {}. {} epochs trained with {:.4}s/epoch.'.format(
                self.name, self.current_time(),
                len(self.training_epoch_costs), np.mean(self.training_epoch_costs)
            )
        )
        plog += '\n' + 'All records and results saved in {}'.format(self.output_dir)
        self.write_log(plog)

    @staticmethod
    def resize(data):
        """
        data:
          [img, size, interpolation_method, blur_method, blur_kernel, blur_sigma]
        cv2 coordinates:
          [horizontal, vertical], which is different as numpy array image.shape
          'cubic': cv2.INTER_LINEAR
          'linear': cv2.INTER_CUBIC
          'nearest' or None(default): cv2.INTER_NEAREST
        Caution: cubic interpolation may generate values out of original data range (e.g. negative values)

        """
        data += [None, ] * (6 - len(data))

        img, size, interpolation_method, blur_method, blur_kernel, blur_sigma = data

        #
        if interpolation_method == 'nearest':
            interpolation_method = cv2.INTER_NEAREST
        elif interpolation_method is None or interpolation_method == 'cubic':
            interpolation_method = cv2.INTER_CUBIC
        elif interpolation_method == 'linear':
            interpolation_method = cv2.INTER_LINEAR
        else:
            raise ValueError('cv2 Interpolation methods: None, nearest, cubic, linear')

        if blur_kernel is None:
            blur_kernel = 5
        if blur_sigma is None:
            blur_sigma = 0

        # calculate the output size
        if isinstance(size, (float, int)):
            size = [size, size]
        if not isinstance(size, (list, tuple)):
            raise TypeError('The input Size of LR image should be (float, int, list or tuple)')
        if isinstance(size[0], float):
            size = int(img.shape[0] * size[0]), int(img.shape[1] * size[1])
        if size[0] <= 0 or size[1] <= 0:
            raise ValueError('Size of output image should be positive')

        # resize the image
        if size[0] == img.shape[0] and size[1] == img.shape[1]:
            output_img = img
        else:
            # opencv2 is [horizontal, vertical], so the output_size should be reversed
            size = size[1], size[0]
            output_img = cv2.resize(img, dsize=size, interpolation=interpolation_method)

        # blur the image if necessary
        if blur_method == 'gaussian':
            output_img = cv2.GaussianBlur(output_img, (blur_kernel, blur_kernel), blur_sigma)
        else:
            # add more blur methods
            pass
        if img.ndim != output_img.ndim:
            output_img = output_img[:, :, np.newaxis]
        return output_img



