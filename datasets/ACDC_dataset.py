from datasets.basic_dataset import BasicMultiSRTrain, BasicMultiSRTest, MedicalImageBasicDataset
from datasets.basic_dataset import SingleImageRandomCrop, CentreCrop, ImageFolder
from metrics.sr_evaluation import MetaSREvaluation

import numpy as np
import nibabel as nib

from os.path import join
from os import listdir
from glob import glob
import copy

from multiprocessing import Pool

"""
Dataset for medical image segmentation:
    1. loading data with nib from the .nii / .mnc / ... file
    2. feed patch (image and label) to networks
    3. validation dataset:
    4. post processing
    
Dataset for medical image super-resolution:
    1. loading data with nib from the .mnc file
    2. feed patch to 
    
Training / Validation / Testing (GT + SR results)

Under working ... :
    OASISSRTest -> patient wise dataset for inference
    OASISMetaSRDataset 
    OASISMetaSRTest -> patient wise dataset for inference
    OASISSegTest -> patient wise dataset for inference

Test passed:
    OASISRawDataset
    OASISSegDataset
    OASISSRDataset

Todo: @ Oct 23 2020
    1. re-organise
    2. re-test the datasets
    
@Jin (jin.zhu@cl.cam.ac.uk) Aug 11 2020
"""


class ACDCReader(MedicalImageBasicDataset):
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

    def __init__(self, ):
        super(ACDCReader, self).__init__()

        self.raw_data_folder = ''

        self.image_path_template = '{}_frame*.nii.gz'
        self.label_path_template = '{}_frame*_gt.nii.gz'

        self.dim = 2
        self.centre_crop_size = 128
        self.centre_crop = None

        self.toy_problem = True

        self.multi_pool = Pool(8)

        self.patient_ids = None

        self.masks = {}
        self.norm = ''
        self.norm_paras = {}
        self.img_ids = []

    def loading(self):

        if self.toy_problem:
            self.patient_ids = self.patient_ids[:2]

        if self.toy_problem:
            self.patient_ids = self.patient_ids[:2]
        for pid in self.patient_ids:
            image_data = self.load_data(pid)
            for img in image_data:
                self.hr_images.append(img)

            # pid as image ids
            self.img_ids += [pid] * len(image_data)

        # ## crop image with margin
        self.centre_crop = CentreCrop(self.centre_crop_size)
        self.hr_images = self.multi_pool.map(self.centre_crop, self.hr_images)

    def load_data(self, pid):
        all_label_paths = glob(join(
            self.raw_data_folder, pid,
            self.label_path_template.format(pid)))

        pid_data = []
        pid_data_ranges = {}
        for label_path in all_label_paths:
            # load label and select slice based on label
            label_data = nib.load(label_path).get_fdata()
            label_data = np.swapaxes(label_data, 0, self.dim)
            label_data, mask = self.select_slice(label_data, threshold=100)
            
            # load slices
            frame_path = label_path.replace('_gt', '')
            frame_data = nib.load(frame_path).get_fdata()
            frame_data = np.swapaxes(frame_data, 0, self.dim)
            # remove slices with no label
            frame_data, _ = self.select_slice(frame_data, mask=mask)
            # norm
            frame_data, image_min, image_max = self.normalize(frame_data)
            pid_data_ranges[frame_path.split('/')[-1]] = [image_min, image_max]
            pid_data.append(frame_data)
        pid_data = np.concatenate(pid_data, axis=0)
        if pid_data.ndim == 3:
            pid_data = pid_data[:, :, :, np.newaxis]
        self.norm_paras[pid] = pid_data_ranges
        return pid_data

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


class ACDCMultiSRTrain(ACDCReader, BasicMultiSRTrain):
    """
    Behaviours:
        1. generate training patches in a batch;
        2. patches in each batch have the same sr_factor;
        3. how to design validation process?
        4. how to define SR-Evaluation?
    """

    def __init__(self, paras):
        super(ACDCMultiSRTrain, self).__init__()

        self.raw_data_folder = paras.data_folder
        self.toy_problem = paras.toy_problem
        self.dim = paras.medical_image_dim_acdc
        self.patient_ids = paras.training_patient_ids_acdc
        self.centre_crop_size = paras.crop_size_acdc
        
        self.norm = paras.normal_inputs

        self.loading()

        # ## training patch cropping <- MultiSRDataset
        self.sr_scales = paras.all_sr_scales
        # lr image size remain
        self.lr_image_size_remain = paras.lr_image_size_remain
        self.cal_sr_scale_index()
        self.batch_size = paras.batch_size
        self.lr_patch_size = paras.patch_size
        self.return_res_image = paras.return_res_image

        # cropping, if remain the image size, only one crop function is necessary
        if self.lr_image_size_remain:
            self.batch_size = 1
            self.crops = [SingleImageRandomCrop(self.get_hr_patch_size(0), 0)]
            self.return_res_image = True
        else:
            self.crops = [SingleImageRandomCrop(self.get_hr_patch_size(s), 0) for s in self.sr_scales]

        self.mean = [0.] 
        self.std = [1.] 
        if 'zero_mean' in self.norm and len(self.hr_images):
            self.mean = np.mean(self.hr_images, axis=(0, 1, 2))
        if 'unit_std' in self.norm and len(self.hr_images):
            self.std = np.std(self.hr_images, axis=(0, 1, 2))


class ACDCMultiSRTest(ACDCReader, BasicMultiSRTest):

    def __init__(self, paras, patient_ids,):
        super(ACDCMultiSRTest, self).__init__()

        self.raw_data_folder = paras.data_folder
        self.toy_problem = paras.toy_problem
        self.dim = paras.medical_image_dim_acdc
        self.patient_ids = patient_ids
        self.centre_crop_size = paras.crop_size_acdc

        self.norm = paras.normal_inputs

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
        self.quick_eva_func = MetaSREvaluation(quick_eva_metrics, self.test_sr_scales, eva_gpu, 'mean')
        self.final_eva_func = MetaSREvaluation(final_eva_metrics, self.test_sr_scales, eva_gpu, 'full')
        
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
        sample = super(ACDCMultiSRTest, self).get_test_pair(item)
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

