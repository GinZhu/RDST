from models.basic_trainer import *
from networks.edsr import EDSR
from networks.srresnet import SRResNet
from networks.srdensenet import SRDenseNet
from networks.rdn import RDN
from networks.esrgan import ESRGAN
from utils.optim import make_optimizer, make_scheduler
from loss.sr_loss import SRLoss
import torch
from torch.utils.data import DataLoader, RandomSampler

"""
Stage II Trainer:
# fill in this part

@Jin (jin.zhu@cl.cam.ac.uk) June xx 2020
"""


class SRTrainer(BasicTrainer):

    def __init__(self, paras, DS_train, DS_valid):
        super(SRTrainer, self).__init__(paras)

        # data
        self.DS_train = DS_train
        self.DS_valid = DS_valid

        self.name = '{}_{}_x{}'.format(self.name, paras.sr_generator, paras.sr_scale)

        # models
        valid_methods = ['EDSR', 'SRResNet', 'SRDenseNet', 'RDN', 'ESRGAN']
        self.module_names.append('model_g')
        if paras.sr_generator == 'EDSR':
            self.model_g = EDSR(paras, DS_train.mean, DS_train.std).to(self.device)
        elif paras.sr_generator == 'SRResNet':
            self.model_g = SRResNet(paras, DS_train.mean, DS_train.std).to(self.device)
        elif paras.sr_generator == 'SRDenseNet':
            self.model_g = SRDenseNet(paras, DS_train.mean, DS_train.std).to(self.device)
        elif paras.sr_generator == 'RDN':
            self.model_g = RDN(paras, DS_train.mean, DS_train.std).to(self.device)
        elif paras.sr_generator == 'ESRGAN':
            self.model_g = ESRGAN(paras, DS_train.mean, DS_train.std).to(self.device)
        else:
            raise ValueError(
                'Invalid SR generator {}, supports {}'.format(paras.sr_generator, valid_methods)
            )

        # model embedding
        self.residual_scale = paras.residual_scale

        # optimizers
        self.module_names.append('optimizer_g')
        self.optimizer_g = make_optimizer(paras, self.model_g)
        self.module_names.append('scheduler_g')
        self.scheduler_g = make_scheduler(paras, self.optimizer_g)

        # loss functions
        self.module_names.append('loss')
        self.loss = SRLoss(paras)
        self.training_loss_components = self.loss.loss_components

        # evaluation functions
        self.quick_eva_func = DS_valid.get_quick_eva_func()
        self.final_eva_func = DS_valid.get_final_eva_func()
        # evaluation metrics
        self.quick_eva_metrics = DS_valid.get_quick_eva_metrics()
        self.final_eva_metrics = DS_valid.get_final_eva_metrics()

    def train(self):
        for ts_i in range(self.current_training_state_id, len(self.training_states)):
            ts = self.training_states[ts_i]

            plog = self.fancy_print(
                'Training State {} start @ {}'.format(ts, self.current_time())
            )

            self.write_log(plog)

            self.current_training_state_id = ts_i
            epochs = self.training_epochs[ts]
            left_epochs = epochs - self.current_epoch + 1
            if left_epochs == 0:
                self.current_epoch = 0
                plog = self.fancy_print(
                    'Training State {} completed before.'.format(ts)
                )

                self.write_log(plog)
                continue

            self.loss.set_training_state(ts)

            # ## data loader
            DL = DataLoader(
                self.DS_train, self.batch_size,
                sampler=RandomSampler(self.DS_train, True, left_epochs * self.batch_size)
            )

            # ## mean loss
            temp_loss_reports = []
            for i, training_batch in enumerate(DL, self.current_epoch+1):

                # timer
                epoch_start_timepoint = time.time()

                self.current_epoch = i
                self.model_g.train()
                input_imgs = training_batch['in']
                target_imgs = training_batch['out']
                res_imgs = training_batch['res']

                input_imgs, target_imgs, res_imgs = self.prepare(
                    input_imgs, target_imgs, res_imgs
                )

                rec_imgs = self.model_g(input_imgs)

                # model embedding
                if self.residual_scale > 0.:
                    rec_imgs = rec_imgs * (1 - self.residual_scale) + res_imgs * self.residual_scale

                loss, repo = self.loss(rec_imgs, target_imgs)

                # loss threshold
                if loss.item() < self.loss_threshold:
                    # record loss
                    temp_loss_reports.append(repo)
                    for n in repo:
                        l = repo[n]
                        self.training_loss_records[n].append(l)

                    # update G
                    self.optimizer_g.zero_grad()
                    loss.backward()
                    self.optimizer_g.step()
                    if self.scheduler_g is not None:
                        self.scheduler_g.step()

                # timer
                epoch_time_cost = time.time() - epoch_start_timepoint
                self.training_epoch_costs.append(epoch_time_cost)

                if i % self.check_every == 0 or i == epochs:
                    self.quick_eva(save_imgs=True)
                    self.save_checkpoint()
                    plog = 'Training stage {} Epoch {} - {}, mean losses:\n'.format(
                        ts, i - len(temp_loss_reports), i
                    )
                    plog += self.loss.print(temp_loss_reports)
                    self.write_log(plog)
                    temp_loss_reports = []

            # at the end of each training state, re-set epoch, and save trained models
            self.current_epoch = 0
            self.save_models(ts)

            self.final_eva(ts)

            plog = self.fancy_print(
                'Training State {} completed @ {}.'.format(ts, self.current_time())
            )

            self.write_log(plog)

        self.final_eva('All Completed')

        self.training_complete()

    def __inference_one__(self, sample):
        img = sample['in']
        res_img = sample['res']
        img, res_img = self.prepare(img, res_img)
        self.model_g.eval()
        with torch.no_grad():
            rec_img = self.model_g(img)[0]
            if self.residual_scale > 0.:
                rec_img = rec_img * (1. - self.residual_scale) + res_img * self.residual_scale
        rec_img = self.tensor_2_numpy(rec_img)

        return rec_img

    def weights_init(self):
        """
        To initialize the weights:
            1. kaiming_normal
            2. with pre_trained model
        :return: plog a message about what has happend
        """
        plog = ''
        ptm_g_path = self.paras.pre_trained_g
        ptm_d_path = self.paras.pre_trained_d
        if isinstance(ptm_g_path, str) and exists(ptm_g_path):
            ptm = torch.load(ptm_g_path, map_location=self.device)
            self.model_g.load_state_dict(ptm)
            plog += 'Init G with pre-trained model\n'
        else:
            # init_func = WeightsInitializer(
            #     act=self.paras.act,
            #     leaky_relu_slope=self.paras.leaky_relu_slope
            # )
            # self.model_g.apply(init_func)
            plog += 'Initialize G by default(he uniform)\n'

        if isinstance(ptm_d_path, str) and exists(ptm_d_path):
            ptm = torch.load(ptm_d_path, map_location=self.device)
            self.loss.load_state_dict(ptm)
            plog += 'Init Adversarial Loss with pre-trained model\n'

        else:
            # init_func = WeightsInitializer(
            #     act=self.paras.d_act,
            #     leaky_relu_slope=self.paras.leaky_relu_slope
            # )
            # self.loss.apply(init_func)
            plog += 'Init Adversarial Loss by default(he_uniform)\n'

        return plog


