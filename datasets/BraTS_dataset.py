from datasets.basic_dataset import MedicalImageBasicDataset, BasicMultiSRTrain, BasicMultiSRTest
from datasets.basic_dataset import ImagePadding
from datasets.basic_dataset import SingleImageRandomCrop
from metrics.sr_evaluation import MetaSREvaluation, MultiModalityMetaSREvaluation
from datasets.basic_dataset import ImageFolder

import numpy as np
import nibabel as nib

from os.path import join

from multiprocessing import Pool

"""

@Jin (jin.zhu@cl.cam.ac.uk) Aug 2021
"""


class BraTSReader(MedicalImageBasicDataset):
    """
    Loading data from the OASIS dataset for training / validation
    Image data example information:
        OAS1_0041_MR1 (176, 208, 176, 1) 3046.0 0.0 231.66056320277733
    To pre-process:
        0. reshape to 3D
        1. select slices (remove background on three dimensions);
        2. normalise;
        3. merge to a list
    """

    def __init__(self,):
        super(BraTSReader, self).__init__()

        # data folder
        self.raw_data_folder = ''

        self.modalities = []

        self.dim = 2

        self.margin = 20

        self.toy_problem = True

        self.multi_pool = Pool(8)

        self.patient_ids = None

        self.masks = {}
        self.norm = ''
        self.norm_paras = {}
        self.img_ids = []

        self.remove_margin = None

    def loading(self):
        """
        Load data into self.hr_images
            swap axe;
            select_slice
            Crop
        Load image id to image ids
        Calculate mean/std of each patient into self.norm_paras
        :return:
        """
        # ## loading training data and validation data
        # ## Training dataset should be merged and shuffled, while validation dataset should be patient-wise
        if self.toy_problem:
            self.patient_ids = self.patient_ids[:2]
        for pid in self.patient_ids:
            image_data, _ = self.load_data(pid)
            for img in image_data:
                self.hr_images.append(img)

            # pid as image ids
            self.img_ids += [pid] * len(image_data)

        # ## crop image with margin
        self.remove_margin = SingleImageRandomCrop(0, self.margin)
        self.hr_images = self.multi_pool.map(self.remove_margin, self.hr_images)

    def load_data(self, pid):
        p_folder, p_name = self.encode_pid(pid)

        # label first to get mask
        # label
        label_path = join(p_folder, '{}_{}.nii.gz'.format(p_name, 'seg'))
        label_data = nib.load(label_path).get_fdata()
        label_data = np.swapaxes(label_data, 0, self.dim)
        label_data, mask = self.select_slice(label_data, threshold=100)
        self.masks[pid] = mask

        # convert label == 3 to label=4
        label_data[label_data == 4] = 3
        label_data = np.expand_dims(label_data, axis=-1)

        pid_data = []
        pid_data_ranges = []
        for m in self.modalities:
            image_path = join(p_folder, '{}_{}.nii.gz'.format(p_name, m))
            image_data = nib.load(image_path).get_fdata()
            image_data = np.swapaxes(image_data, 0, self.dim)
            if pid not in self.masks:
                image_data, mask = self.select_slice(image_data)
                self.masks[pid] = mask
            else:
                image_data, mask = self.select_slice(image_data, mask=self.masks[pid])
            image_data, image_min, image_max = self.normalize(image_data)
            pid_data.append(image_data)
            pid_data_ranges.append([image_min, image_max])
        pid_data = np.stack(pid_data, axis=-1)
        self.norm_paras[pid] = pid_data_ranges
        return pid_data, label_data

    def encode_pid(self, pid):
        sub_dir = pid.split('_')[0]
        pid = pid.replace('{}_'.format(sub_dir), '')
        p_folder = join(self.raw_data_folder, sub_dir, pid)
        return p_folder, pid

    @staticmethod
    def select_slice(imgs, mask=None, threshold=100):
        # ## get brain slices only
        if mask is None:
            if imgs.ndim == 4:
                mask = np.sum(imgs, axis=(1, 2, 3)) > threshold
            elif imgs.ndim == 3:
                mask = np.sum(imgs, axis=(1, 2)) > threshold
        selected_imgs = imgs[mask]

        return selected_imgs, mask


class BraTSMultiSRTrain(BraTSReader, BasicMultiSRTrain):
    """
    Behaviours:
        1. generate training patches in a batch;
        2. patches in each batch have the same sr_factor;
        3. how to design validation process?
        4. how to define SR-Evaluation?
    """

    def __init__(self, paras):
        super(BraTSMultiSRTrain, self).__init__()

        # ## data loading <- RGBReader
        self.toy_problem = paras.toy_problem
        self.dim = paras.medical_image_dim_brats
        self.patient_ids = paras.training_patient_ids_brats
        self.margin = paras.margin_brats
        self.multi_pool = Pool(paras.multi_threads)
        self.raw_data_folder = paras.data_folder
        self.norm = paras.normal_inputs
        self.modalities = paras.modalities_brats

        self.loading()

        # ## training patch cropping <- MultiSRDataset
        self.sr_scales = paras.all_sr_scales
        # lr image size remain
        self.lr_image_size_remain = paras.lr_image_size_remain
        self.cal_sr_scale_index()
        self.batch_size = paras.batch_size
        self.lr_patch_size = paras.patch_size
        self.return_res_image = paras.return_res_image

        # ## Padding
        input_shape = self.hr_images[0].shape[:2]
        Pad = ImagePadding(input_shape, self.get_hr_patch_size(max(self.sr_scales)))
        self.hr_images = self.multi_pool.map(Pad.pad, self.hr_images)

        # cropping, if remain the image size, only one crop function is necessary
        if self.lr_image_size_remain:
            self.batch_size = 1
            self.crops = [SingleImageRandomCrop(self.get_hr_patch_size(0), 0)]
            self.return_res_image = True
        else:
            self.crops = [SingleImageRandomCrop(self.get_hr_patch_size(s), 0) for s in self.sr_scales]

        self.mean = [0.] * len(self.modalities)
        self.std = [1.] * len(self.modalities)
        if 'zero_mean' in self.norm and len(self.hr_images):
            self.mean = np.mean(self.hr_images, axis=(0, 1, 2))
        if 'unit_std' in self.norm and len(self.hr_images):
            self.std = np.std(self.hr_images, axis=(0, 1, 2))


class BraTSMultiSRTest(BraTSReader, BasicMultiSRTest):
    """
    Processing in two steps:
        1. Unfold image into patches;
        2. Decompose patches into wavelet domain;
    Reconstruct image in two steps:
        1. Reconstruct patches from wavelet domain;
        2. Fold patches to generate SR image.
    """
    def __init__(self, paras, patient_ids: list):
        super(BraTSMultiSRTest, self).__init__()

        # ## data loading <- RGBReader
        self.toy_problem = paras.toy_problem
        self.dim = paras.medical_image_dim_brats
        self.patient_ids = patient_ids
        self.margin = paras.margin_brats
        self.multi_pool = Pool(paras.multi_threads)
        self.raw_data_folder = paras.data_folder
        self.norm = paras.normal_inputs
        self.modalities = paras.modalities_brats

        self.loading()

        # ## image shape
        self.input_channels = self.hr_images[0].shape[-1]
        self.hr_image_region = self.hr_images[0].shape[:2]

        # ## patch cropping <- MultiSRDataset
        self.test_sr_scales = paras.test_sr_scales
        self.lr_patch_size = paras.patch_size
        self.lr_patch_stride = paras.test_lr_patch_stride
        self.return_res_image = paras.return_res_image
        # lr image size remain
        self.lr_image_size_remain = paras.lr_image_size_remain

        # ## eva function
        quick_eva_metrics = paras.quick_eva_metrics
        final_eva_metrics = paras.eva_metrics
        eva_gpu = paras.eva_gpu_id
        self.quick_eva_func = MultiModalityMetaSREvaluation(self.modalities, quick_eva_metrics,
                                                            self.test_sr_scales, eva_gpu, 'mean')
        self.final_eva_func = MultiModalityMetaSREvaluation(self.modalities, final_eva_metrics,
                                                            self.test_sr_scales, eva_gpu, 'full')
        # self.quick_eva_func = MetaSREvaluation(quick_eva_metrics, self.test_sr_scales, eva_gpu, 'mean')
        # self.final_eva_func = MetaSREvaluation(final_eva_metrics, self.test_sr_scales, eva_gpu, 'full')

        #
        # self.crop_func = CentreCrop(self.hr_image_region)
        self.crop_func = lambda x: x

        # get image unfolder and folders
        if self.lr_image_size_remain:
            # calculate the lr image sizes
            H, W = self.hr_image_region
            lr_image_shape = (1, self.input_channels, H, W)
            img_folder = ImageFolder(
                lr_image_shape, self.lr_patch_size, stride=self.lr_patch_stride
            )
            self.lr_unfolders = {0: img_folder.get_unfolder()}
            self.hr_folders = {0: img_folder.get_folder()}
            # return res image for following operation
            self.return_res_image = True

        else:
            self.lr_unfolders = {}
            self.hr_folders = {}
            for s in self.test_sr_scales:
                lr_h, lr_w = int(self.hr_image_region[0] / s), int(self.hr_image_region[1] / s)
                lr_image_shape = (1, self.input_channels, lr_h, lr_w)
                imf_lr = ImageFolder(lr_image_shape, self.lr_patch_size, stride=self.lr_patch_stride)
                self.lr_unfolders[s] = imf_lr.get_unfolder()
                imf_hr = ImageFolder(
                    (1, self.input_channels, self.hr_image_region[0], self.hr_image_region[1]),
                    int(self.lr_patch_size * s),
                    stride=int(self.lr_patch_stride * s)
                )
                self.hr_folders[s] = imf_hr.get_folder()

    def get_test_pair(self, item):
        """

        :param item:
        :return:
            'in': 1 x C x H x W (lr_size_remain=True) tensor
            'gt': H x W x C numpy array
            'res': 1 x C x H x W tensor
            'sr_factor': float
            'real_sr_scale': float
        """
        sample = super(BraTSMultiSRTest, self).get_test_pair(item)
        for s in sample:
            if self.lr_image_size_remain:
                sample[s]['in'] = sample[s]['res']

            sample[s]['real_sr_scale'] = sample[s]['real_sr_scale'][0]
        return sample

    def post_processing(self, *args, **kwargs):
        pass

    def pre_processing(self, *args, **kwargs):
        pass

    def test_len(self):
        return len(self.hr_images)




