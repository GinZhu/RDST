from abc import ABC, abstractmethod
from os.path import isdir, join, exists
from os import makedirs
from skimage import io
import numpy as np
import matplotlib.pyplot as plt

import torch

from datetime import datetime, timedelta
import time


"""
todo: fill in this part with an example

@Jin (jin.zhu@cl.cam.ac.uk) June 22 2020
"""


class BasicTrainer(ABC):

    def __init__(self, paras):
        super(BasicTrainer, self).__init__()

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
        self.output_dir = paras.output_dir
        self.models_dir = None
        self.records_dir = None
        self.plots_dir = None
        self.final_results_dir = None
        self.inference_dir = None
        self.checkpoint_path = None
        self.training_log = None

        # ## data
        self.DS_train = None
        self.DS_valid = None

        # ## training
        self.batch_size = paras.batch_size
        self.training_loss_components = []
        self.training_epochs = paras.epochs_in_total
        self.training_states = paras.training_states

        # ## loss
        self.loss_threshold = paras.loss_threshold

        # ## training records
        self.check_every = paras.check_every
        self.training_loss_records = {}
        # ## current training stage
        self.current_training_state_id = None
        self.current_epoch = 0
        # ## time cost
        self.training_epoch_costs = []

        # ## validation evaluations
        self.quick_validation_reports = []

        # ## evaluation functions, from DataLoader
        self.quick_eva_func = None
        self.final_eva_func = None
        self.quick_eva_num_samples = paras.quick_eva_num_samples
        self.quick_eva_num_images_to_save = paras.quick_eva_num_images_to_save

        # ## models and optimizers, and losses
        self.module_names = []

    def _creat_dirs(self):
        """
        Testing passed.

        :return:
        """
        self.output_dir = join(self.output_dir, self.name)

        # creat some dirs for output
        self.output_dir = self.exist_or_make(self.output_dir)

        self.models_dir = self.exist_or_make(join(
            self.output_dir, 'models'))
        self.records_dir = self.exist_or_make(join(
            self.output_dir, 'records'))
        self.plots_dir = self.exist_or_make(join(
            self.output_dir, 'plots'))
        self.final_results_dir = self.exist_or_make(join(
            self.output_dir, 'final_results'))
        self.inference_dir = self.exist_or_make(join(
            self.output_dir, 'inferences'))

        self.checkpoint_path = join(
            self.output_dir, 'checkpoint.tar'
        )
        # training log
        self.training_log = join(self.output_dir, 'training_log.txt')

    def setup(self):
        """
        This function must be called first
        """
        # creat dirs
        self._creat_dirs()

        # save paras for this experiment
        plog = self.fancy_print(
            'Experiment starts @ {}, with paras:'.format(self.current_time())
        )
        plog += '\n' + str(self.paras) + '\n\n\n'

        for m in self.module_names:
            if 'model' in m:
                plog += self.fancy_print('Generator {}:') + '\n'
                plog += str(self.__getattribute__(m))

        # if training from checkpoint, load
        if exists(self.checkpoint_path):
            message = self.load_checkpoint()

        # if from scratch
        else:
            self.current_training_state_id = 0
            self.current_epoch = 0

            # training records:
            for l in self.training_loss_components:
                self.training_loss_records[l] = []

            message = self.fancy_print(
                'New Training with {}, Epochs {}'.format(
                    self.training_states, self.training_epochs
                )
            )

            message += '\n' + self.weights_init()

        plog += message

        self.write_log(plog)

    @abstractmethod
    def weights_init(self):
        """
        To implement this function, the trainer will initialize the weights of all models / Modules:
            1. random initialization;
            2. kaiming_normal or other initialization methods;
            3. pre-trained models initialization.
        :return: None
        """
        return ''

    def load_checkpoint(self):
        # load the training states from checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        # load models, optimizers, losses
        for n in self.module_names:
            self.__getattribute__(n).load_state_dict(checkpoint[n])
        # load training states
        self.training_loss_components = checkpoint['training_loss_names']
        self.training_loss_records = checkpoint['training_loss_records']
        self.quick_validation_reports = checkpoint['quick_validation_reports']
        self.current_training_state_id = checkpoint['current_training_state_id']
        self.current_epoch = checkpoint['current_epoch']
        self.training_epoch_costs = checkpoint['training_epoch_costs']

        plog = self.fancy_print(
            'Resuming training with {}, Epoch {}'.format(
                self.training_states[self.current_training_state_id],
                self.current_epoch
            )
        )

        return plog

    def save_checkpoint(self):
        # save the training states to checkpoint
        checkpoint = {'Time': self.current_time('str')}
        # save models
        for n in self.module_names:
            checkpoint[n] = self.__getattribute__(n).state_dict()
        # save optimizers
        for n in self.module_names:
            checkpoint[n] = self.__getattribute__(n).state_dict()
        # save training states
        checkpoint['training_loss_names'] = self.training_loss_components
        checkpoint['training_loss_records'] = self.training_loss_records
        checkpoint['quick_validation_reports'] = self.quick_validation_reports
        checkpoint['current_training_state_id'] = self.current_training_state_id
        checkpoint['current_epoch'] = self.current_epoch
        checkpoint['training_epoch_costs'] = self.training_epoch_costs

        torch.save(checkpoint, self.checkpoint_path)
        plog = 'Checkpoint saved to {}'.format(self.checkpoint_path)
        if self.verbose:
            print(plog)
        self.write_log(plog)

    def save_models(self, prefix=''):
        # save trained models to dir, normally be called at the end of each training stage
        # save: 1. models; 2. loss (when GAN is used)

        for n in self.module_names:
            if 'model' in n or 'loss' in n:
                torch.save(
                    self.__getattribute__(n).state_dict(),
                    join(self.models_dir, '{}_{}.pt'.format(prefix, n))
                )
                plog = 'Model {}_{} saved.'.format(prefix, n)

                self.write_log(plog)
            # elif 'loss' in n:
            #     plog = self.__getattribute__(n).save_models(self.models_dir, prefix)

            #     self.write_log(plog)

    @abstractmethod
    def train(self):
        # train the model
        pass

    def inference(self, D):
        # given a Dataset, do inference on it
        preds = []
        for i in range(D.test_len()):
            s = D.get_test_pair(i)
            preds.append(self.__inference_one__(s))
        return preds

    @abstractmethod
    def __inference_one__(self, sample):
        # given a data sample, do inference on it and return
        pass

    def __evaluation__(self, eva_func, sample_ids):
        rec_imgs = []
        samples = []
        for i in sample_ids:
            s = self.DS_valid.get_test_pair(i)
            pred = self.__inference_one__(s)
            rec_imgs.append(pred)
            samples.append(s)
        repo = eva_func(rec_imgs, samples)
        return repo, rec_imgs, samples

    def quick_eva(self, save_imgs=True):
        """
        Do a quick validation during training. Some of the results will be saved as images for visualization.
        :param save_imgs: If true, save images
        :return: None
        """
        # should call self.evaluation(), self.save_images, and self.plot_training()
        test_pair_len = self.DS_valid.test_len()
        eva_img_ids = list(range(test_pair_len))
        np.random.shuffle(eva_img_ids)
        eva_img_ids = eva_img_ids[:self.quick_eva_num_samples]

        eva_start_time = self.current_time('float')

        eva_report, rec_imgs, ori_samples = self.__evaluation__(self.quick_eva_func, eva_img_ids)

        eva_time_cost = self.time_cost(self.current_time('float') - eva_start_time)

        # record validation performance
        self.quick_validation_reports.append(eva_report)

        # print the validation performance
        flag = '{}_Epoch_{} Validation performance, with time cost {}'.format(
            self.training_states[self.current_training_state_id],
            self.current_epoch,
            eva_time_cost
        )
        plog = flag + ':\n' + self.quick_eva_func.print(eva_report)

        self.write_log(plog)

        if save_imgs:
            # ## record images to perceptually compare
            rec_imgs = rec_imgs[:self.quick_eva_num_images_to_save]
            ori_samples = ori_samples[:self.quick_eva_num_images_to_save]
            imgs_for_recording = self.quick_eva_func.display_images(rec_imgs, ori_samples)

            for k in imgs_for_recording:
                imgs = imgs_for_recording[k]
                self.save_images(
                    join(self.records_dir, '{}_{}.png'.format(flag, k)),
                    imgs,
                    self.quick_eva_num_images_to_save
                )

        # ## plot training process, with training losses and validation scores
        self.plot_training_process()

    def final_eva(self, prefix):
        # do the evaluation on all the data in DS_valid
        # should call self.save_images, and self.plot_results
        sample_ids = list(range(self.DS_valid.test_len()))

        eva_start_time = self.current_time('float')

        eva_report, rec_imgs, ori_samples = self.__evaluation__(self.final_eva_func, sample_ids)

        eva_time_cost = self.time_cost(self.current_time('float') - eva_start_time)

        flag = self.fancy_print('{} Final Evaluation costs {}'.format(prefix, eva_time_cost))
        plog = flag + '\n' + self.final_eva_func.print(eva_report)

        self.write_log(plog)

        self.final_eva_func.plot_final_evas(
            eva_report,
            self.plots_dir,
            flag
        )

    # @staticmethod
    # def stack_eva_reports(reports):
    #     # stack each element in eva_report separately
    #     stacked_report = {}
    #     for k in reports[0].keys():
    #         stacked_report[k] = []
    #         for r in reports:
    #             stacked_report[k].append(r[k])
    #     return stacked_report

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
        with open(self.training_log, 'a') as f:
            f.write(plog + '\n')

    def plot_training_process(self,):

        # plot each loss
        for k in self.training_loss_components:
            ls = self.training_loss_records[k]
            if len(ls) == 0:
                continue
            ls = np.stack(ls)
            plt.plot(ls, label=k)
            plt.xlabel('Training Step')
            plt.ylabel(k)
            plt.grid(True)
            plt.legend()
            plt.savefig(join(self.plots_dir, 'Training_{}.png'.format(k)))
            plt.close()

        self.quick_eva_func.plot_process(
            self.quick_validation_reports,
            self.plots_dir,
            'Quick Validation Performance'
        )

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
        return str(timedelta(seconds=int(t)))

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
        torch.save(summary, join(self.final_results_dir, 'training_records.tar'))

        plog = self.fancy_print(
            '{} training completed @ {}. {} epochs trained with {:.4}s/epoch.'.format(
                self.name, self.current_time(),
                len(self.training_epoch_costs), np.mean(self.training_epoch_costs)
            )
        )
        plog += '\n' + 'All records and results saved in {}'.format(self.output_dir)
        self.write_log(plog)




