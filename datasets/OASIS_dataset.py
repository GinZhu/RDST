from datasets.basic_dataset import BasicMultiSRTrain, BasicMultiSRTest
from datasets.basic_dataset import MedicalImageBasicDataset
from datasets.basic_dataset import SingleImageRandomCrop, ImagePadding, SRImagePairRandomCrop
from metrics.sr_evaluation import MetaSREvaluation
from datasets.basic_dataset import ImageFolder

import numpy as np
import nibabel as nib

from os.path import join
from glob import glob

from multiprocessing import Pool

"""
    
@Jin (jin.zhu@cl.cam.ac.uk) Aug 2021
"""


class OASISReader(MedicalImageBasicDataset):
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
        super(OASISReader, self).__init__()

        # data folder
        self.raw_data_folder = ''
        self.image_folder = 'PROCESSED/MPRAGE/T88_111'

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
            image_path = glob(join(self.raw_data_folder, pid, self.image_folder, '*masked_gfc.img'))[0]
            image_data = nib.load(image_path).get_fdata()
            image_data = np.swapaxes(image_data, 0, self.dim)
            image_data, mask = self.select_slice(image_data)
            self.masks[pid] = mask
            image_data, image_min, image_max = self.normalize(image_data)
            # patient wise normalise
            self.norm_paras[pid] = [image_min, image_max]
            for img in image_data:
                self.hr_images.append(img)
            # pid as image ids
            self.img_ids += [pid] * mask.sum()

        # ## crop image with margin
        self.remove_margin = SingleImageRandomCrop(0, self.margin)
        self.hr_images = self.multi_pool.map(self.remove_margin, self.hr_images)

    @staticmethod
    def select_slice(imgs, mask=None):
        # ## get brain slices only
        if mask is None:
            mask = np.sum(imgs, axis=(1, 2, 3)) > 0
        selected_imgs = imgs[mask]

        return selected_imgs, mask


class OASISMultiSRTrain(OASISReader, BasicMultiSRTrain):
    """
    Behaviours:
        1. generate training patches in a batch;
        2. patches in each batch have the same sr_factor;
        3. how to design validation process?
        4. how to define SR-Evaluation?
    """

    def __init__(self, paras):
        super(OASISMultiSRTrain, self).__init__()

        # ## data loading <- RGBReader
        self.toy_problem = paras.toy_problem
        self.dim = paras.medical_image_dim_oasis
        self.patient_ids = paras.training_patient_ids_oasis
        self.margin = paras.margin_oasis
        self.multi_pool = Pool(paras.multi_threads)
        self.raw_data_folder = paras.data_folder
        self.norm = paras.normal_inputs
        # blur
        self.blur_method = paras.blur_method

        self.loading()

        # ## training patch cropping <- MultiSRDataset
        self.sr_scales = paras.all_sr_scales
        # lr image size remain
        self.lr_image_size_remain = paras.lr_image_size_remain
        self.cal_sr_scale_index()
        self.batch_size = paras.batch_size
        self.lr_patch_size = paras.patch_size
        # self.wavelet_hr_patch_size = paras.wavelet_hr_patch_size
        self.return_res_image = paras.return_res_image

        # ## wavelet related settings
        # self.wavelet_level = paras.wavelet_level
        # self.wavelet_kernel = paras.wavelet_kernel
        # self.wavelet_mode = paras.wavelet_mode

        # self.wt_func = Wavelet(self.wavelet_level, self.wavelet_kernel, self.wavelet_mode)

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

        # ## zero-mean-unit-variance
        self.mean = [0.]
        self.std = [1.]
        if 'zero_mean' in self.norm and len(self.hr_images):
            self.mean = np.mean(self.hr_images, axis=(0, 1, 2))
        if 'unit_std' in self.norm and len(self.hr_images):
            self.std = np.std(self.hr_images, axis=(0, 1, 2))


class OASISMultiSRTest(OASISReader, BasicMultiSRTest):
    """
    Processing in two steps:
        1. Unfold image into patches;
        2. Decompose patches into wavelet domain;
    Reconstruct image in two steps:
        1. Reconstruct patches from wavelet domain;
        2. Fold patches to generate SR image.
    To use:
        paras.patch_size = paras.wavelet_hr_patch_size
        paras.test_lr_patch_stride = paras.wavelet_hr_patch_size

        wtds = RGBMultiSRWaveletTest(paras)

        device = torch.device('cuda:0')
        samples = wtds.get_test_pair(2)
        for s in samples:  # [2.0, 3.0, 4.0]
            sample = samples[s]
            lr = sample['in']   # (1 x 1 x 768 x 768) tensor
            res = sample['res'] # (1 x 1 x 768 x 768) tensor or [[]]
            hr = sample['gt']   # (768 x 768 x 1) numpy array
            srf = sample['sr_factor']   # float
            rsf = sample['real_sr_scale']   # float

            lr = lr.to(device)
            # lr_tokens: N x P x C x h x w tensor on device
            lr_tokens = wtds.pre_processing(lr, srf)

            sr_tokens = model(lr_tokens, torch.tensor([rsf] * N).reshape(N, 1)

            # image: 1 x C x H x W tensor on device
            image = wtds.post_processing(sr_tokens, srf)
    """
    def __init__(self, paras, patient_ids: list):
        super(OASISMultiSRTest, self).__init__()

        # ## data loading <- RGBReader
        self.toy_problem = paras.toy_problem
        self.dim = paras.medical_image_dim_oasis
        self.patient_ids = patient_ids
        self.margin = paras.margin_oasis
        self.multi_pool = Pool(paras.multi_threads)
        self.raw_data_folder = paras.data_folder
        self.norm = paras.normal_inputs
        # blur
        self.blur_method = paras.blur_method

        self.loading()

        # ## image shape
        self.input_channels = self.hr_images[0].shape[-1]
        self.hr_image_region = self.hr_images[0].shape[:2]

        # ## patch cropping <- MultiSRDataset
        # assert paras.patch_size == paras.wavelet_hr_patch_size, \
        #     'In wavelet transformer, patch_size must equal wavelet_hr_patch_size'
        self.test_sr_scales = paras.test_sr_scales
        self.lr_patch_size = paras.patch_size
        self.lr_patch_stride = paras.test_lr_patch_stride
        self.return_res_image = paras.return_res_image
        # lr image size remain
        self.lr_image_size_remain = paras.lr_image_size_remain

        # # data mode: 'image' -> return image pairs; 'coeffs': return wavelet tokens
        # self.data_mode = paras.wt_data_mode

        # # decompose image to wavelet tokens and inverse
        # self.wavelet_hr_patch_size = paras.wavelet_hr_patch_size
        # self.wavelet_level = paras.wavelet_level
        # self.wavelet_kernel = paras.wavelet_kernel
        # self.wavelet_mode = paras.wavelet_mode
        # self.wt_func = Wavelet(self.wavelet_level, self.wavelet_kernel, self.wavelet_mode)

        # ## eva function
        quick_eva_metrics = paras.quick_eva_metrics
        final_eva_metrics = paras.eva_metrics
        eva_gpu = paras.eva_gpu_id
        self.quick_eva_func = MetaSREvaluation(quick_eva_metrics, self.test_sr_scales, eva_gpu, 'mean')
        self.final_eva_func = MetaSREvaluation(final_eva_metrics, self.test_sr_scales, eva_gpu, 'full')

        # crop if necessary
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

        # # ## pre/post-processing
        # self.collate_fn = WaveletTokenCollateFunc(self.wavelet_mode)

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
        sample = super(OASISMultiSRTest, self).get_test_pair(item)
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


class OASISSegSRTrain(OASISMultiSRTrain):
    """
    Only support x4 SR tasks.
    return GT labels when 'getitem'
    """
    def __init__(self, paras):
        super(OASISSegSRTrain, self).__init__(paras)

        self.seg_classes = ['gray', 'white', 'CSF']
        self.label_folder = 'FSL_SEG'

        # loading segmentation labels
        self.segmentation_labels = []
        for pid in self.patient_ids:
            label_path = glob(join(self.raw_data_folder, pid, self.label_folder, '*masked_gfc_fseg.img'))[0]
            label_data = nib.load(label_path).get_fdata()
            label_data = np.swapaxes(label_data, 0, self.dim)
            label_data = label_data[self.masks[pid]]
            for l in label_data:
                self.segmentation_labels.append(l)
        self.segmentation_labels = self.multi_pool.map(
            self.remove_margin, self.segmentation_labels
        )
        # cropping
        patch_size = int(self.lr_patch_size * self.sr_scales[0])
        self.dual_crop_func = SRImagePairRandomCrop(patch_size, sr_factor=1)

    def __getitem__(self, item):

        ids = np.random.choice(self.__len__(), self.batch_size, False)
        sr_factor = np.random.choice(self.sr_scales)    # only one sr_scale exist

        # ## sr_factor is by setting, but we need to know the real sr_scale.
        # ## rs should be the same as sr_factor when they are int.
        # ## but when sr scales are float such as 2.1 or 2.2, rs will be different
        rs = self.get_hr_patch_size(sr_factor) / self.get_lr_patch_size(sr_factor)

        img_outputs = []
        labels = []
        for i in ids:
            img = self.hr_images[i]
            label = self.segmentation_labels[i]
            img, label = self.dual_crop_func([img, label])
            img_outputs.append(img)
            labels.append(label)

        img_inputs = [self.resize([_, self.get_lr_patch_size(sr_factor), 'cubic', self.blur_method]) for _ in img_outputs]
        img_outputs = img_outputs

        if self.return_res_image:
            res_imgs = [self.resize([_, self.get_hr_patch_size(sr_factor)]) for _ in img_inputs]
            res_imgs = self.numpy_2_tensor(res_imgs)
        else:
            res_imgs = [[]] * self.batch_size

        img_inputs = self.numpy_2_tensor(img_inputs)
        img_outputs = self.numpy_2_tensor(img_outputs)
        labels = self.numpy_2_tensor(labels)

        return {'in': img_inputs, 'out': img_outputs, 'sr_factor': sr_factor,
                'res': res_imgs, 'real_sr_scale': rs, 'seg_gt': labels}




