from models.basic_trainer import *
from networks.swin_transformer_sr import swinir_make_model
from networks.swinIR_variations import make_RDSTSR
from networks.edsr import EDSR
from networks.rdn import RDN
from networks.han import han_make_model
from networks.rcan import rcan_make_model
from networks.convnet import ConvNetSR_model_lite, ConvNetSR_model_large
from utils.optim import make_optimizer, make_scheduler
from loss.sr_loss import SRLoss
import torch
from torch.utils.data import DataLoader, RandomSampler

"""
Wavelet Transformer for SR Trainer
# fill in this part

@Jin (jin.zhu@cl.cam.ac.uk) Sep 2021
"""


class TransSRTrainer(BasicTrainer):

    def __init__(self, paras, DS_train, DS_valid):
        super(TransSRTrainer, self).__init__(paras)

        # data
        self.DS_train = DS_train
        self.DS_valid = DS_valid
        self.sr_generator = paras.feature_generator
        self.name = '{}_{}'.format(self.name, paras.gan_type)

        self.batch_in_dataloader_flag = True

        # models
        self.model_input_with_scale_flag = 'with_scales'
        self.module_names.append('model_g')

        if self.sr_generator in ['swin', 'SwinIR', 'swinir', 'swinIR']:
            self.model_g = swinir_make_model(paras).to(self.device)
        elif self.sr_generator in ['rdst']:
            self.model_g = make_RDSTSR(paras, self.DS_train.mean, self.DS_train.std).to(self.device)
        elif self.sr_generator in ['edsr']:
            self.model_g = EDSR(paras, self.DS_train.mean, self.DS_train.std).to(self.device)
        elif self.sr_generator in ['rdn']:
            self.model_g = RDN(paras, self.DS_train.mean, self.DS_train.std).to(self.device)
        elif self.sr_generator in ['han', 'HAN', 'Han']:
            self.model_g = han_make_model(paras, self.DS_train.mean, self.DS_train.std).to(self.device)
        elif self.sr_generator in ['rcan', 'RCAN', 'Rcan', 'RCan']:
            self.model_g = rcan_make_model(paras, self.DS_train.mean, self.DS_train.std).to(self.device)
        elif self.sr_generator in ['convnet-large', 'ConvNet-Large']:
            self.model_g = ConvNetSR_model_large(paras, self.DS_train.mean, self.DS_train.std).to(self.device)
        elif self.sr_generator in ['convnet-lite', 'ConvNet-Lite']:
            self.model_g = ConvNetSR_model_lite(paras, self.DS_train.mean, self.DS_train.std).to(self.device)
        else:
            valid_models = ['swinir', 'rdst', 'edsr', 'rdn', 'han', 'rcan', 'convnet']
            raise ValueError('Invalid model type, only support: {}'.format(valid_models))

        # generate batch in dataset / dataloader flag
        multi_scale = paras.scale_free
        if multi_scale:
            self.batch_in_dataloader_flag = False
            self.model_input_with_scale_flag = 'with_scales'
        else:
            self.batch_in_dataloader_flag = False
            self.model_input_with_scale_flag = 'no'

        assert self.model_input_with_scale_flag in ['with_scales', 'no']

        # optimizers
        self.module_names.append('optimizer_g')
        self.optimizer_g = make_optimizer(paras, self.model_g)
        self.module_names.append('scheduler_g')
        self.scheduler_g = make_scheduler(paras, self.optimizer_g)

        # loss functions
        self.module_names.append('loss')
        self.loss = SRLoss(paras)
        self.training_loss_components = self.loss.loss_components
        # self.use_seg_loss_flag = self.loss.use_seg_loss_flag

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
            # ## batch_in_dataloader_flag must be False, so the batch_size must be 1
            batch_size = self.batch_size if self.batch_in_dataloader_flag else 1
            DL = DataLoader(
                self.DS_train, batch_size,
                sampler=RandomSampler(self.DS_train, True, int(batch_size * left_epochs)),
                collate_fn=self.DS_train.get_collate_func(),
                num_workers=8
            )

            # ## mean loss
            temp_loss_reports = []
            for i, training_batch in enumerate(DL, self.current_epoch+1):

                # ipt / mdsr / metasr / multi scale model
                if not self.batch_in_dataloader_flag:
                    training_batch = {k: training_batch[k][0] for k in training_batch}

                # timer
                epoch_start_timepoint = time.time()

                self.current_epoch = i
                self.model_g.train()
                input_imgs = training_batch['in']
                target_imgs = training_batch['out']
                sr_scales = training_batch['sr_factor'].item()
                # if self.use_seg_loss_flag:
                #     gt_labels = training_batch['seg_gt']
                #     gt_labels = self.prepare(gt_labels)

                input_imgs, target_imgs = self.prepare(
                    input_imgs, target_imgs
                )

                # ## modify inputs
                if self.model_input_with_scale_flag in ['no']:  # edsr / swinir
                    rec_imgs = self.model_g(input_imgs)
                elif self.model_input_with_scale_flag in ['with_scales']:   # my wt models
                    rec_imgs = self.model_g(input_imgs, sr_scales)
                else:   # todo: add ipt / metasr support
                    raise ValueError('Invalid input flag')

                # if self.use_seg_loss_flag:
                #     loss, repo = self.loss(rec_imgs, target_imgs, gt_label=gt_labels)
                # else:
                #     loss, repo = self.loss(rec_imgs, target_imgs)
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

        # self.final_eva('All Completed')

        self.training_complete()

    def __inference_one__(self, sample):
        rec_imgs = {}
        for s in sample:
            case = sample[s]
            lr_img = case['in']
            lr_img = self.prepare(lr_img)     # precision and device

            # # processing flag
            # no_processing = True
            # # padding = self.sr_generator in ['rdst', 'swinir']
            # padding = False

            # lr_tokens = self.DS_valid.pre_processing(lr_img, s, no_processing=no_processing, padding=padding)
            lr_tokens = lr_img
            self.model_g.eval()
            with torch.no_grad():
                # generate sr patches
                sr_tokens = []
                for p in lr_tokens.split(self.batch_size * 4):
                    sr_scale = case['sr_factor']
                    # sr_scale = torch.tensor(sr_scale).repeat(p.size(0), 1)
                    # sr_scale = self.prepare(sr_scale)   # precision and device

                    if self.model_input_with_scale_flag in ['no']:  # edsr / swinir
                        rec_p = self.model_g(p)
                    elif self.model_input_with_scale_flag in ['with_scales']:  # my wt models
                        rec_p = self.model_g(p, sr_scale)
                    else:  # todo: add ipt / metasr support
                        raise ValueError('Invalid input flag')

                    sr_tokens.append(rec_p)
                sr_tokens = torch.cat(sr_tokens, dim=0)
                # reconstruct the sr image
                rec_img = sr_tokens
                # rec_img = self.DS_valid.post_processing(sr_tokens, s, no_processing=no_processing)
            rec_img = self.tensor_2_numpy(rec_img)[0]
            rec_imgs[s] = rec_img

        return rec_imgs

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
            # plog += 'Initialize G by he_normal\n'
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
            # plog += 'Init Adversarial Loss by he_normal\n'
            plog += 'Init Adversarial Loss by default(he_uniform)\n'

        return plog

