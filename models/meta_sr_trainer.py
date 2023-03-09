from models.basic_trainer import *
from networks.meta_sr import MetaSR
from networks.mdsr import MDSR
from utils.optim import make_optimizer, make_scheduler
from loss.sr_loss import SRLoss
import torch
from torch.utils.data import DataLoader, RandomSampler
from networks.common import WeightsInitializer

"""
Meta SR Trainer:
# fill in this part

@Jin (jin.zhu@cl.cam.ac.uk) June xx 2020
"""


class MetaSRTrainer(BasicTrainer):

    def __init__(self, paras, DS_train, DS_valid):
        super(MetaSRTrainer, self).__init__(paras)

        # data
        self.DS_train = DS_train
        self.DS_valid = DS_valid

        self.name = '{}_{}_{}'.format(self.name, paras.feature_generator, paras.gan_type)

        # models
        self.module_names.append('model_g')
        if paras.feature_generator == 'MDSR':
            self.model_g = MDSR(paras, DS_train.mean, DS_train.std)
        else:
            self.model_g = MetaSR(paras, DS_train.mean, DS_train.std)
        self.model_g.to(self.device)

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
                self.DS_train, 1,
                sampler=RandomSampler(self.DS_train, True, left_epochs)
            )

            # ## mean loss
            temp_loss_reports = []
            for i, training_batch in enumerate(DL, self.current_epoch+1):

                # timer
                epoch_start_timepoint = time.time()

                training_batch = {k: training_batch[k][0] for k in training_batch}

                self.current_epoch = i
                self.model_g.train()
                input_imgs = training_batch['in']
                target_imgs = training_batch['out']
                res_imgs = training_batch['res']

                input_imgs, target_imgs, res_imgs = self.prepare(
                    input_imgs, target_imgs, res_imgs
                )
                sr_factor = training_batch['sr_factor'].item()

                rec_imgs = self.model_g(input_imgs, sr_factor)

                # model embedding
                if self.residual_scale > 0.:
                    rec_imgs = rec_imgs * (1-self.residual_scale) + res_imgs * self.residual_scale

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
        rec_imgs = {}
        for s in sample:
            case = sample[s]
            img = case['in']
            res_img = case['res']
            img, res_img = self.prepare(img, res_img)
            self.model_g.eval()
            with torch.no_grad():
                rec_img = self.model_g(img, s)[0]
                if self.residual_scale > 0.:
                    rec_img = rec_img * (1. - self.residual_scale) + res_img * self.residual_scale
            rec_img = self.tensor_2_numpy(rec_img)
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

    # def inference(self, D, inference_only=True):
    #     rst_dir = self.exist_or_make(
    #         join(self.inference_dir, D.patient_id)
    #     )
    #     preds = super(MetaSRTrainer, self).inference(D)
    #     if inference_only:
    #         return preds
    #     eva_func = D.get_final_eva_func()
    #     eva_metrics = D.get_final_eva_metrics()
    #     all_eva_reports = []
    #     for i, pred in enumerate(preds):
    #         sample = D.get_test_pair(i)
    #         eva_report = eva_func(pred, sample)
    #         all_eva_reports.append(eva_report)
    #
    #     # ## plot and save inference
    #     for i in range(0, len(all_eva_reports), 10):
    #         imgs = all_eva_reports[i]['imgs']
    #         self.save_images(
    #             join(rst_dir, '{}.png'.format(i)),
    #             imgs,
    #             N_R=1
    #         )
    #
    #     np.savez_compressed(
    #         join(rst_dir, 'preds_and_reports.npz'),
    #         all_eva_reports=all_eva_reports,
    #         preds=preds
    #     )
    #
    #     # ## analysis results
    #     all_eva_reports = self.stack_eva_reports(all_eva_reports)
    #     plog = 'Inference on {}: '.format(D.patient_id)
    #     for m in eva_metrics:
    #         m_mean = np.mean(all_eva_reports[m])
    #         m_std = np.std(all_eva_reports[m])
    #         plog += ' {} {:.4}(+/-{:.2})'.format(m, m_mean, m_std)
    #
    #     self.write_log(plog)
