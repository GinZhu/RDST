from models.basic_tester import BasicTester
from networks.swin_transformer_sr import swinir_make_model
from networks.swinIR_variations import make_RDSTSR
from networks.edsr import EDSR
from networks.rdn import RDN
from networks.han import han_make_model
from networks.rcan import rcan_make_model
from networks.convnet import ConvNetSR_model_lite, ConvNetSR_model_large


import torch
import copy
import numpy as np

from metrics.sr_evaluation import MetaSREvaluation, MultiModalityMetaSREvaluation
from datasets.OASIS_dataset import OASISMultiSRTest
from datasets.BraTS_dataset import BraTSMultiSRTest
from datasets.ACDC_dataset import ACDCMultiSRTest
from datasets.CovidCT_dataset import CovidCTMultiSRTest

"""
This one is for multi-scales SR tasks evaluation.

@Jin Zhu (jin.zhu@cl.cam.ac.uk) Nov 14 2020
"""


class TransSRTester(BasicTester):

    def __init__(self, paras):
        super(TransSRTester, self).__init__(paras)

        # data
        valid_datasets = ['OASIS',]     # 'BraTS', 'ACDC', 'COVID']
        data_folder = self.paras.data_folder
        self.which_data = None
        if 'OASIS' in data_folder:
            self.testing_patient_ids = self.paras.testing_patient_ids_oasis
            self.which_data = 'OASIS'
        elif 'BraTS' in data_folder:
            self.testing_patient_ids = self.paras.testing_patient_ids_brats
            self.which_data = 'BraTS'
        elif 'ACDC' in data_folder:
            self.testing_patient_ids = self.paras.testing_patient_ids_acdc
            self.which_data = 'ACDC'
        elif 'COVID' in data_folder:
            self.testing_patient_ids = self.paras.testing_patient_ids_covid
            self.which_data = 'COVID'
        # elif 'ACDC' in data_folder:
        #     self.testing_patient_ids = self.paras.testing_patient_ids_acdc
        #     self.which_data = 'ACDC'
        # elif 'COVID' in data_folder:
        #     self.testing_patient_ids = self.paras.testing_patient_ids_covid
        #     self.which_data = 'COVID'
        else:
            # add more
            raise ValueError('Invalid data, {}, only support {}'.format(data_folder, valid_datasets))
        self.DS = None
        self.name = '{}_{}'.format('FT', self.name)
        self.batch_size = paras.batch_size

        # model & input settings
        self.sr_generator = paras.feature_generator

        self.batch_in_dataloader_flag = True

        self.model_input_with_scale_flag = 'with_scales'

        if self.sr_generator in ['swin', 'SwinIR', 'swinir', 'swinIR']:
            self.single_scale_model = swinir_make_model(paras).to(self.device)
        elif self.sr_generator in ['rdst']:
            self.single_scale_model = make_RDSTSR(paras).to(self.device)
        elif self.sr_generator in ['edsr']:
            self.single_scale_model = EDSR(paras).to(self.device)
        elif self.sr_generator in ['rdn']:
            self.single_scale_model = RDN(paras).to(self.device)
        elif self.sr_generator in ['rcan']:
            self.single_scale_model = rcan_make_model(paras).to(self.device)
        elif self.sr_generator in ['han']:
            self.single_scale_model = han_make_model(paras).to(self.device)
        elif self.sr_generator in ['convnet-large']:
            self.single_scale_model = ConvNetSR_model_large(paras).to(self.device)
        elif self.sr_generator in ['convnet-lite']:
            self.single_scale_model = ConvNetSR_model_lite(paras).to(self.device)
        elif self.sr_generator in ['bicubic']:
            self.single_scale_model = None
            pass
        else:
            valid_models = ['swinir', 'rdst', 'edsr', 'rdn']
            raise ValueError('Invalid model type, only support: {}'.format(valid_models))

        # if not bicubic, load model
        if self.sr_generator in ['bicubic']:
            self.save_gts = True
        else:
            self.save_gts = False
            self.model_names = ['single_scale_model']
            self.ptm_paths['single_scale_model'] = paras.well_trained_single_scale_model_g

        # generate batch in dataset / dataloader flag
        multi_scale = paras.scale_free
        if multi_scale:
            self.batch_in_dataloader_flag = False
            self.model_input_with_scale_flag = 'with_scales'
        else:
            self.batch_in_dataloader_flag = False
            self.model_input_with_scale_flag = 'no'

        assert self.model_input_with_scale_flag in ['with_scales', 'no']

        # model embedding, which is not used for now
        self.residual_scale = paras.residual_scale

        # evaluation
        eva_metrics = self.paras.eva_metrics_for_testing
        eva_gpu = self.paras.gpu_id
        if self.which_data == 'BraTS':
            self.eva_func = MultiModalityMetaSREvaluation(
                paras.modalities_brats, eva_metrics, self.test_sr_factors, eva_gpu, 'full'
            )
        else:
            self.eva_func = MetaSREvaluation(eva_metrics, self.test_sr_factors, eva_gpu, 'full')

    def __inference_one__(self, sample):
        rec_imgs = {}
        for s in sample:
            case = sample[s]
            lr_img = case['in']
            res_img = case['res']
            lr_img, res_img = self.prepare(lr_img, res_img)     # precision and device

            if self.sr_generator in ['bicubic']:
                rec_img = res_img[0]
                rec_img = self.tensor_2_numpy(rec_img)
            else:
                # # processing flag
                # no_processing = True
                # # padding = self.sr_generator in ['rdst', 'swinir']
                # padding = False

                # lr_tokens = self.DS.pre_processing(lr_img, s, no_processing=no_processing, padding=padding)
                self.single_scale_model.eval()
                with torch.no_grad():
                    # generate sr patches
                    rec_img = []
                    for p in lr_img.split(self.batch_size * 4):
                        sr_scale = case['sr_factor']
                        # sr_scale = torch.tensor(sr_scale).repeat(p.size(0), 1)
                        # sr_scale = self.prepare(sr_scale)   # precision and device

                        if self.model_input_with_scale_flag in ['no']:  # edsr / swinir
                            rec_p = self.single_scale_model(p)
                        elif self.model_input_with_scale_flag in ['with_scales']:  # my wt models
                            rec_p = self.single_scale_model(p, sr_scale)
                        else:
                            raise ValueError('Invalid input flag')

                        rec_img.append(rec_p)
                    rec_img = torch.cat(rec_img, dim=0)
                    # reconstruct the sr image
                    # rec_img = self.DS.post_processing(rec_img, s, no_processing=no_processing)
                rec_img = self.tensor_2_numpy(rec_img)[0]

            rec_imgs[s] = rec_img

        return rec_imgs

    def modify_image_shape(self, img, s):
        int_s = np.ceil(s)
        H, W = img.shape[:2]
        return self.resize([img, [int(H//int_s*s), int(W//int_s*s)]])

    def test(self):
        all_eva_reports = []
        all_inference_costs = []
        # inference & evaluate case-by-case
        case_i = 0
        case_n = len(self.testing_patient_ids)
        for pid in self.testing_patient_ids:
            case_i += 1
            plog = self.fancy_print('[{}/{}] Inference & Evaluation on case {} start @ {}'.format(
                case_i, case_n, pid, self.current_time()))
            self.write_log(plog)

            if self.which_data == 'OASIS':
                self.DS = OASISMultiSRTest(self.paras, [pid,])
            elif self.which_data == 'BraTS':
                self.DS = BraTSMultiSRTest(self.paras, [pid,])
            elif self.which_data == 'ACDC':
                self.DS = ACDCMultiSRTest(self.paras, [pid,])
            elif self.which_data == 'COVID':
                self.DS = CovidCTMultiSRTest(self.paras, [pid,])
            # elif self.which_data == 'ACDC':
            #     DS = ACDCMetaSRTest(self.paras, pid, self.test_sr_factors)
            # elif self.which_data == 'COVID':
            #     DS = COVIDMetaSRTest(self.paras, pid, self.test_sr_factors)
            else:
                pass

            eva_report, inference_cost = self.evaluation(pid, self.DS)
            all_eva_reports.append(eva_report)
            all_inference_costs.append(inference_cost)

        all_eva_reports = self.eva_func.stack_eva_reports(all_eva_reports)

        # summary the all reports
        flag = self.fancy_print(
            'Summary evaluation performance on {} with {} cases @ {}, mean inference cost {}'.format(
                self.which_data, len(self.testing_patient_ids), self.current_time(), np.mean(all_inference_costs)
            )
        )
        plog = flag + '\n' + 'Case IDs: {}\n'.format(self.testing_patient_ids)
        plog += self.eva_func.print(all_eva_reports)

        self.write_log(plog)

    def select_images_to_save(self, all_images):
        selected_imgs = []
        for img in all_images:
            case = {}
            for s in self.sr_factors_for_saving:
                case[s] = img[s]
            selected_imgs.append(case)
        return selected_imgs

    def get_gt_images(self, samples):
        """

        :param samples: a list of ori_samples from the dataset
        :return gt_imgs: a dict with the same format of rec_imgs
        """
        gt_imgs = []
        for case in samples:
            gt_case = {}
            for s in case:
                gt_case[s] = case[s]['gt']
            gt_imgs.append(gt_case)
        return gt_imgs


