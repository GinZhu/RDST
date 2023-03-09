from models.basic_tester import BasicTester
from networks.meta_sr import MetaSR
from networks.edsr import EDSR
from networks.mdsr import MDSR
from networks.srresnet import SRResNet
from networks.srdensenet import SRDenseNet
from networks.esrgan import ESRGAN
from networks.rdn import RDN
import torch
import copy
import numpy as np

from metrics.sr_evaluation import MetaSREvaluation, MultiModalityMetaSREvaluation
from datasets.OASIS_dataset import OASISMetaSRTest
from datasets.BraTS_dataset import BraTSMetaSRTest
from datasets.ACDC_dataset import ACDCMetaSRTest
from datasets.COVID_dataset import COVIDMetaSRTest

"""
This one is for multi-scales SR tasks evaluation.

@Jin Zhu (jin.zhu@cl.cam.ac.uk) Nov 14 2020
"""


class MetaSRTester(BasicTester):

    def __init__(self, paras):
        super(MetaSRTester, self).__init__(paras)

        self.name = '{}_{}'.format(self.name, paras.trained_model_mode)

        self.trained_model_mode = paras.trained_model_mode

        self.valid_sota_methods = [
            # 'SRResNet',
            'SRGAN',
            'SRDenseNet',
            'RDN',
            'EDSR',
            'ESRGANL1',
            # 'ESRGAN',
        ]
        self.sota_model_name_prefix = 'model_g_x{}'
        if self.trained_model_mode == 'MDSR':
            self.model_names.append('mdsr')
            self.mdsr = MDSR(paras).to(self.device)
            self.ptm_paths['mdsr'] = paras.well_trained_model_mdsr
        elif self.trained_model_mode in self.valid_sota_methods:
            for s in [2, 3, 4]:
                m = self.sota_model_name_prefix.format(s)
                self.model_names.append(m)
                model = self.create_sr_model(s).to(self.device)
                self.__setattr__(m, model)
                ptm_path = self.paras.well_trained_model_g_single_scale
                self.ptm_paths[m] = ptm_path.format(self.trained_model_mode, s)
        elif 'MetaSR' in self.trained_model_mode:
            self.model_names.append('meta_sr')
            self.meta_sr = MetaSR(paras).to(self.device)
            self.ptm_paths['meta_sr'] = paras.well_trained_model_metasr
        elif self.trained_model_mode == 'bicubic':
            pass
        else:
            raise ValueError('Invalid trained model mode {}, only support {}'.format(
                self.trained_model_mode, self.valid_sota_methods + ['MDSR', 'MetaSR', 'bicubic']
            ))

        # model embedding, which is not used for now
        self.residual_scale = paras.residual_scale

        # bicubic image, and save gts
        if self.trained_model_mode == 'bicubic':
            self.save_gts = True
        else:
            self.save_gts = False

        # evaluation
        eva_metrics = self.paras.eva_metrics_for_testing
        eva_gpu = self.paras.gpu_id
        self.eva_func = MetaSREvaluation(eva_metrics, self.test_sr_factors, eva_gpu, 'full')

        # testing data
        #
        valid_datasets = ['OASIS', 'BraTS', 'ACDC', 'COVID']
        data_folder = self.paras.data_folder
        self.which_data = None
        if 'OASIS' in data_folder:
            self.testing_patient_ids = self.paras.testing_patient_ids_oasis
            self.which_data = 'OASIS'
        elif 'BraTS' in data_folder:
            self.testing_patient_ids = self.paras.testing_patient_ids_brats
            self.which_data = 'BraTS'
            # for multi-modality
            if len(self.paras.modalities_brats) > 1:
                self.eva_func = MultiModalityMetaSREvaluation(
                    self.paras.modalities_brats,
                    eva_metrics, self.test_sr_factors, eva_gpu, 'full'
                )
        elif 'ACDC' in data_folder:
            self.testing_patient_ids = self.paras.testing_patient_ids_acdc
            self.which_data = 'ACDC'
        elif 'COVID' in data_folder:
            self.testing_patient_ids = self.paras.testing_patient_ids_covid
            self.which_data = 'COVID'
        else:
            # add more
            raise ValueError('Invalid data, {}, only support {}'.format(data_folder, valid_datasets))

    def __inference_one__(self, sample):
        rec_imgs = {}

        for s in sample:
            case = sample[s]
            img = case['in']
            res_img = case['res']
            img, res_img = self.prepare(img, res_img)

            if self.trained_model_mode == 'bicubic':
                rec_img = res_img[0]
                rec_img = self.tensor_2_numpy(rec_img)
            elif self.trained_model_mode in self.valid_sota_methods + ['MDSR', ]:
                # increse the sr scale to make the model work
                int_s = int(np.ceil(s))
                if self.trained_model_mode in self.valid_sota_methods:
                    model = self.__getattribute__(self.sota_model_name_prefix.format(int_s))
                    model.eval()
                    with torch.no_grad():
                        rec_img = model(img)[0]
                else:
                    self.mdsr.eval()
                    with torch.no_grad():
                        rec_img = self.mdsr(img, int_s)[0]
                rec_img = self.tensor_2_numpy(rec_img)
                # reduce the image (normally down-sample) sr scale to correct size
                rec_img = self.modify_image_shape(rec_img, s)
            elif 'MetaSR' in self.trained_model_mode:
                self.meta_sr.eval()
                with torch.no_grad():
                    rec_img = self.meta_sr(img, s)[0]
                rec_img = self.tensor_2_numpy(rec_img)
            else:
                raise NotImplementedError('Invalid model mode {}'.format(self.trained_model_mode))
            # if necessary, do model embedding here

            rec_imgs[s] = rec_img

        return rec_imgs

    def modify_image_shape(self, img, s):
        int_s = np.ceil(s)
        H, W = img.shape[:2]
        return self.resize([img, [int(H//int_s*s), int(W//int_s*s)]])

    def test(self):
        all_eva_reports = []
        # inference & evaluate case-by-case
        for pid in self.testing_patient_ids:
            plog = self.fancy_print('Inference & Evaluation on case {} start @ {}'.format(pid, self.current_time()))
            self.write_log(plog)

            if self.which_data == 'OASIS':
                DS = OASISMetaSRTest(self.paras, pid, self.test_sr_factors)
            elif self.which_data == 'BraTS':
                DS = BraTSMetaSRTest(self.paras, pid, self.test_sr_factors)
            elif self.which_data == 'ACDC':
                DS = ACDCMetaSRTest(self.paras, pid, self.test_sr_factors)
            elif self.which_data == 'COVID':
                DS = COVIDMetaSRTest(self.paras, pid, self.test_sr_factors)
            else:
                pass
            eva_report = self.evaluation(pid, DS)
            all_eva_reports.append(eva_report)

        all_eva_reports = self.eva_func.stack_eva_reports(all_eva_reports)

        # summary the all reports
        flag = self.fancy_print(
            'Summary evaluation performance on {} with {} cases @ {}'.format(
                self.which_data, len(self.testing_patient_ids), self.current_time()
            )
        )
        plog = flag + '\n' + 'Case IDs: {}\n'.format(self.testing_patient_ids)
        plog += self.eva_func.print(all_eva_reports)

        self.write_log(plog)

    def create_sr_model(self, sr_scale):
        paras = copy.deepcopy(self.paras)
        paras.sr_scale = sr_scale

        if self.trained_model_mode in ['SRResNet', 'SRGAN']:
            model = SRResNet(paras)
        elif self.trained_model_mode in ['SRDenseNet']:
            model = SRDenseNet(paras)
        elif self.trained_model_mode in ['EDSR']:
            model = EDSR(paras)
        elif self.trained_model_mode in ['RDN']:
            model = RDN(paras)
        elif self.trained_model_mode in ['ESRGAN', 'ESRGANL1']:
            model = ESRGAN(paras)
        else:
            raise ValueError('Invalid model mode {}'.format(self.trained_model_mode))
        return model

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


