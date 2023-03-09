from abc import ABC, abstractmethod
import torch
from tabulate import tabulate
import numpy as np

"""
todo: fill in this part with an example

@Jin (jin.zhu@cl.cam.ac.uk) Sep 2 2020
"""


class BasicLoss(ABC):

    def __init__(self, paras):
        super(BasicLoss, self).__init__()

        # ## basic information
        self.paras = paras
        if paras.gpu_id == -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:{}'.format(paras.gpu_id))

        # ## training precision
        self.precision = paras.precision

        # ## training
        self.training_loss_names = paras.training_losses
        self.training_loss_scalars = paras.loss_scalars
        self.current_training_state = paras.training_states[0]

        # ## models and [optimizers, learning rate schedulers]
        self.loss_components = []
        self.loss_functions = {}

    @abstractmethod
    def __call__(self, *args, **kwargs):
        # ## return the loss, and loss names, and items
        pass

    @staticmethod
    def print(repo):
        assert isinstance(repo, (dict, list, tuple)), '{} is not a valid report type.'.format(repo.__class__)
        if isinstance(repo, dict):
            repo = [repo]

        headers = list(repo[0].keys())

        row = []
        for k in headers:
            values = []
            for r in repo:
                values.append(r[k])
            mean_v = np.mean(values)
            row.append('{:.4}'.format(mean_v))

        table = [row]
        plog = tabulate(table, headers=headers)
        return plog

    def load_state_dict(self, checkpoint):
        # load the training states from checkpoint
        # if adversarial loss, there should be models and optimizers
        for n in self.loss_functions:
            f = self.loss_functions[n]
            if isinstance(f, torch.nn.Module):
                f.load_state_dict(checkpoint[n])
        # load training states
        # self.current_training_state = checkpoint['current_training_state']

    def state_dict(self):
        # # save the training states to checkpoint
        # states = {
        #     'current_training_state': self.current_training_state
        # }
        states = {}
        # save models
        for n in self.loss_functions:
            f = self.loss_functions[n]
            if isinstance(f, torch.nn.Module):
                states[n] = f.state_dict()
        return states

    @abstractmethod
    def apply(self, fn):
        pass

    def set_training_state(self, ts):
        self.current_training_state = ts


