import numpy as np
import cv2
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import torch
import math

"""
Basic Dataset:
    1. MedicalImageBasicDataset;
    2. MIBasicTrain;
    3. MIBasicValid;

Basic Evaluation

Test passed.

@Jin (jin.zhu@cl.cam.ac.uk) June 22 2020

@Jin Zhu (jin.zhu@cl.cam.ac.uk) Modified at Oct 23 2020
"""


class MedicalImageBasicDataset(Dataset):
    """
    Basic Dataset for medical images, by default it provides four functions:
        1, 2 numpy_2_tensor() and tensor_2_numpy();
        3. normalize;
        4. resize()
    """
    def __init__(self):
        self.hr_images = []

    def __len__(self):
        return len(self.hr_images)

    @staticmethod
    def numpy_2_tensor(a):
        if isinstance(a, list):
            a = np.array(a)
        if a.ndim == 3:
            return torch.tensor(a.transpose(2, 0, 1), dtype=torch.float)
        elif a.ndim == 4:
            return torch.tensor(a.transpose(0, 3, 1, 2), dtype=torch.float)
        else:
            raise ValueError('Image should have 3 or 4 channles')

    @staticmethod
    def tensor_2_numpy(t):
        if t.ndim == 3:
            return t.detach().cpu().numpy().transpose(1, 2, 0)
        elif t.ndim == 4:
            return t.detach().cpu().numpy().transpose(0, 2, 3, 1)
        else:
            return t.detach().cpu().numpy()

    @staticmethod
    def normalize(imgs):
        min_val = np.min(imgs)
        max_val = np.max(imgs)
        imgs_norm = (imgs - min_val) / (max_val - min_val)
        return imgs_norm, min_val, max_val

    @staticmethod
    def resize(data):
        """
        data:
          [img, size, interpolation_method, blur_method, blur_kernel, blur_sigma]
        cv2 coordinates:
          [horizontal, vertical], which is different as numpy array image.shape
          'cubic': cv2.INTER_LINEAR
          'linear': cv2.INTER_CUBIC
          'nearest' or None(default): cv2.INTER_NEAREST
        Caution: cubic interpolation may generate values out of original data range (e.g. negative values)

        """
        data += [None, ] * (6 - len(data))

        img, size, interpolation_method, blur_method, blur_kernel, blur_sigma = data

        #
        if interpolation_method == 'nearest':
            interpolation_method = cv2.INTER_NEAREST
        elif interpolation_method is None or interpolation_method == 'cubic':
            interpolation_method = cv2.INTER_CUBIC
        elif interpolation_method == 'linear':
            interpolation_method = cv2.INTER_LINEAR
        else:
            raise ValueError('cv2 Interpolation methods: None, nearest, cubic, linear')

        if blur_kernel is None:
            blur_kernel = 3
        if blur_sigma is None:
            blur_sigma = 0

        # calculate the output size
        if isinstance(size, (float, int)):
            size = [size, size]
        if not isinstance(size, (list, tuple)):
            raise TypeError('The input Size of LR image should be (float, int, list or tuple)')
        if isinstance(size[0], float):
            size = int(img.shape[0] * size[0]), int(img.shape[1] * size[1])
        if size[0] <= 0 or size[1] <= 0:
            raise ValueError('Size of output image should be positive')

        # resize the image
        if size[0] == img.shape[0] and size[1] == img.shape[1]:
            output_img = img
        else:
            # opencv2 is [horizontal, vertical], so the output_size should be reversed
            size = size[1], size[0]
            output_img = cv2.resize(img, dsize=size, interpolation=interpolation_method)

        # blur the image if necessary
        if blur_method == 'gaussian':
            output_img = cv2.GaussianBlur(output_img, (blur_kernel, blur_kernel), blur_sigma)
        else:
            # todo: add more blur methods
            pass

        if img.ndim != output_img.ndim:
            output_img = output_img[:, :, np.newaxis]
        return output_img


class MIBasicValid(MedicalImageBasicDataset, ABC):

    """
    Abstract Dataset, for validation.
    As a subclass of pytorch Dataset, but instead using __len__ and __getitem__,
    we should call test_len() and get_test_pair(), to avoid confusion.

    This dataset will also provide two evaluation functions:
        1. quick_eva_func: to valid the training process;
        2. final_eva_func: to know the final performance of model;
    """

    def __init__(self):
        super(MIBasicValid, self).__init__()
        self.quick_eva_func = None
        self.final_eva_func = None

    def __len__(self):
        return self.test_len()

    def __getitem__(self, item):
        return self.get_test_pair(item)

    @abstractmethod
    def test_len(self):
        # return the length of all data
        pass

    @abstractmethod
    def get_test_pair(self, item):
        # return a sample of all data
        pass

    def get_quick_eva_func(self):
        return self.quick_eva_func

    def get_final_eva_func(self):
        return self.final_eva_func

    def get_quick_eva_metrics(self):
        return self.quick_eva_func.get_metrics()

    def get_final_eva_metrics(self):
        return self.final_eva_func.get_metrics()


class BasicMultiSRTrain(MedicalImageBasicDataset):
    def __init__(self):
        super(BasicMultiSRTrain, self).__init__()

        # ## sr scales
        self.sr_scales = []
        self.sr_scale_index = {}
        # ## batch & patch
        self.batch_size = 0
        self.lr_patch_size = 0
        self.crops = {}
        self.return_res_image = False

        self.blur_method = None

    def cal_sr_scale_index(self):
        self.sr_scale_index = {s: i for i, s in enumerate(self.sr_scales)}

    def __getitem__(self, item):
        # ## return a batch every time
        ids = np.random.choice(self.__len__(), self.batch_size, False)
        sr_factor = np.random.choice(self.sr_scales)

        # ## sr_factor is by setting, but we need to know the real sr_scale.
        # ## rs should be the same as sr_factor when they are int.
        # ## but when sr scales are float such as 2.1 or 2.2, rs will be different
        rs = self.get_hr_patch_size(sr_factor) / self.get_lr_patch_size(sr_factor)

        img_outputs = []
        for i in ids:
            img = self.hr_images[i]
            img = self.crops[self.sr_scale_index[sr_factor]](img)
            img_outputs.append(img)
        img_inputs = [self.resize([_, self.get_lr_patch_size(sr_factor), 'cubic', self.blur_method]) for _ in img_outputs]
        img_outputs = img_outputs

        if self.return_res_image:
            res_imgs = [self.resize([_, self.get_hr_patch_size(sr_factor)]) for _ in img_inputs]
            res_imgs = self.numpy_2_tensor(res_imgs)
        else:
            res_imgs = [[]] * self.batch_size

        img_inputs = self.numpy_2_tensor(img_inputs)
        img_outputs = self.numpy_2_tensor(img_outputs)

        return {'in': img_inputs, 'out': img_outputs, 'sr_factor': sr_factor, 'res': res_imgs, 'real_sr_scale': rs}

    def get_lr_patch_size(self, s):
        return self.lr_patch_size

    def get_hr_patch_size(self, s):
        return int(self.lr_patch_size * s)

    def get_collate_func(self):
        return None


class BasicMultiSRTest(MedicalImageBasicDataset, ABC):
    def __init__(self):
        """
        HR GT image is with a single size, while the lr input size depends on the sr scale
        """
        super(BasicMultiSRTest, self).__init__()

        # ## evaluation functions
        self.quick_eva_func = None
        self.final_eva_func = None

        # ## sr scales. This is only the settings, but need to be updated once LR image size is decided.
        # ##
        self.test_sr_scales = []

        # ## input lr patches, LR image need to be cut as patches
        self.lr_patch_size = 0

        # ## output settings
        self.return_res_image = False

        # ## blur
        self.blur_method = None

    # def cal_sr_scale_index(self):
    #     self.sr_scale_index = {s: i for i, s in enumerate(self.sr_scales)}
    def crop(self, img):
        return img

    def get_test_pair(self, item):
        # # return gt and LR images with various sr_factors
        ori_img = self.hr_images[item]

        # ## crop if necessary
        ori_img = self.crop(ori_img)

        H, W = ori_img.shape[:2]

        # ## keep the input lr image as the same shape
        #    for example, HR / 4
        s = max(self.test_sr_scales)
        lr_image = self.resize([ori_img, (int(H // s), int(W // s)), 'cubic', self.blur_method])
        img_inputs = [lr_image for _ in self.test_sr_scales]

        lr_h, lr_w = lr_image.shape[:2]

        # generate hr images
        img_outputs = [self.resize([ori_img, (int(lr_h*s), int(lr_w*s))]) for s in self.test_sr_scales]

        # # modify the image shape for meta-upscale module
        # img_outputs = []
        # for s in self.test_sr_scales:
        #     new_H = int(int(H // s) * s)
        #     new_W = int(int(W // s) * s)
        #     img_outputs.append(self.resize([ori_img, (new_H, new_W)]))

        # ## the real sr_scale
        real_sr_scale = [(int(lr_h*s)/lr_h, int(lr_w*s)/lr_w) for s in self.test_sr_scales]

        if self.return_res_image:
            res_imgs = [self.resize([lr, hr.shape[:2]]) for lr, hr in zip(img_inputs, img_outputs)]
            res_imgs = [self.numpy_2_tensor(_).unsqueeze(0) for _ in res_imgs]
        else:
            res_imgs = [[]] * len(self.test_sr_scales)

        img_inputs = [self.numpy_2_tensor(_) for _ in img_inputs]
        img_inputs = [img.unsqueeze(0) for img in img_inputs]

        sample = {}
        for img_in, img_out, s, res, rs in zip(img_inputs, img_outputs, self.test_sr_scales, res_imgs, real_sr_scale):
            sample[s] = {'in': img_in, 'gt': img_out, 'sr_factor': s, 'res': res, 'real_sr_scale': rs}

        return sample

    @abstractmethod
    def post_processing(self, *args, **kwargs):
        pass

    @abstractmethod
    def pre_processing(self, *args, **kwargs):
        pass

    @abstractmethod
    def test_len(self):
        # return the length of all data
        pass

    def get_quick_eva_func(self):
        return self.quick_eva_func

    def get_final_eva_func(self):
        return self.final_eva_func

    def get_quick_eva_metrics(self):
        return self.quick_eva_func.get_metrics()

    def get_final_eva_metrics(self):
        return self.final_eva_func.get_metrics()


class SizeAlign:
    def __init__(self):
        pass

    @staticmethod
    def size_align(s, dim=2, message=''):
        if isinstance(s, int):
            size = tuple([s for _ in range(dim)])
        elif isinstance(s, (list, tuple)):
            if all(isinstance(_, int) for _ in s) and len(s) == dim:
                size = tuple(s)
            else:
                raise TypeError(message)
        else:
            raise TypeError(message)
        return size


class UnFolder:
    def __init__(self, kernel_size, dilation=1, padding=0, stride=1):
        """
        Crop image to patches.
            Input size: 1 x C x H x W, where N should be 1 by default.
            Output size: P*1 x C x pH x pW,
            where P is the number of patches, depending on patch_size, stride, dilation
        all parameters are like torch.nn.Unfold
        """
        self.patch_size = kernel_size
        self.unfold = torch.nn.Unfold(kernel_size, dilation, padding, stride)

    def __call__(self, x):
        patches = self.unfold(x).transpose(1, 2).view(
            -1, x.shape[1], self.patch_size[0], self.patch_size[1])
        return patches


class Folder(SizeAlign):
    def __init__(self, output_size, kernel_size, dilation=1, padding=0, stride=1):
        """
        Reconstruct image from patches.
            Input size: P*N x C x pH x pW, N should be 1.
            Output size: N x C x H x W,
            where P is the number of patches, depending on kernel_size, stride, dilation
        :param output_size: should be N x C x H x W, the original image size
        :param kernel_size: same as unfold
        :param dilation: same as unfold
        :param padding: same as unfold
        :param stride: same as unfold
        """

        self.fold = torch.nn.Fold(output_size[-2:], kernel_size, dilation, padding, stride)
        self.unfold = torch.nn.Unfold(kernel_size, dilation, padding, stride)

        self.patch_size = kernel_size
        self.channel = output_size[1]

        # ## calculate divisor and its inverse
        input_ones = torch.ones(output_size, dtype=torch.float32)
        divisor = self.fold(self.unfold(input_ones))
        self.d = 1. / divisor

    def __call__(self, patches):
        slides = patches.view(1, -1, self.patch_size[0]*self.patch_size[1]*self.channel).transpose(1, 2)
        # ensure that patches and d is on the same device
        d = self.d.cuda(patches.get_device()) if patches.is_cuda else self.d
        image = self.fold(slides) * d
        return image


class ImageFolder(SizeAlign):
    def __init__(self, image_size, patch_size, dilation=1, stride=1):
        """
        Unfold image to patches and fold the patches as an image.
        :param image_size: 1 x C x H x W
        :param patch_size: int or 2D-tuple (ph, pw)
        :param dilation: int or 2D-tuple
        :param stride: int or 2D-tuple

        unfolder: 1 x C x H x W -> P x C x ph x pw
        folder: P x C x ph x pw -> 1 x C x H x W

        To use: unfolder should be used once get the image, folder should be used as a post-processing
        """
        super(ImageFolder, self).__init__()
        assert isinstance(image_size, tuple), 'Image size must be a 4D-tuple of int'
        assert all(isinstance(_, int) for _ in image_size) and len(image_size) == 4, 'Image size must be a 4D-tuple of int'
        H, W = image_size[-2:]

        patch_size = self.size_align(
            patch_size, dim=2, message='Patch size should be int, list(int, int) or tuple(int, int)')

        stride = self.size_align(
            stride, dim=2, message='stride should be int, list(int, int) or tuple(int, int)'
        )
        dilation = self.size_align(
            dilation, dim=2, message='dilation should be int, list(int, int) or tuple(int, int)'
        )

        # ## calculate padding
        margin = [
            H - int((H - 1 - dilation[0] * (patch_size[0] - 1)) / stride[0] + 1) * stride[0],
            W - int((W - 1 - dilation[1] * (patch_size[1] - 1)) / stride[1] + 1) * stride[1]
        ]

        padding = tuple([0 if m == 0 else math.ceil((p - m) / 2) for m, p in zip(margin, patch_size)])

        self.fold_parameters = dict(
            kernel_size=patch_size,
            dilation=dilation,
            stride=stride,
            padding=padding
        )

        self.folder = Folder(image_size, **self.fold_parameters)
        self.unfolder = UnFolder(**self.fold_parameters)

    def get_folder(self):
        return self.folder

    def get_unfolder(self):
        return self.unfolder


class BasicCropTransform(ABC):
    def __init__(self, size, margin):
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, (list, tuple)):
            if all(isinstance(_, int) for _ in size):
                self.size = size
            else:
                raise TypeError('Crop size should be int, list(int) or tuple(int)')
        else:
            raise TypeError('Crop size should be int, list(int), or tuple(int)')

        if self.size[0] == 0 and self.size[1] == 0:
            self.size = None

        if isinstance(margin, int):
            self.margin = (margin, margin)
        elif isinstance(margin, (list, tuple)):
            if all(isinstance(_, int) for _ in margin):
                self.margin = margin
            else:
                raise TypeError('Crop margin should be int, list(int) or tuple(int)')
        else:
            raise TypeError('Crop margin should be int, list(int), or tuple(int)')

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class SingleImageRandomCrop(BasicCropTransform):

    def __init__(self, size, margin=0):
        super(SingleImageRandomCrop, self).__init__(size, margin)

    def __call__(self, in_img):
        if self.size is None:
            return in_img[self.margin[0]:-self.margin[0], self.margin[1]:-self.margin[1]]
        else:
            ori_H, ori_W = in_img.shape[:2]
            x_top_left = np.random.randint(
                self.margin[0], ori_H - self.size[0] - self.margin[0] + 1
            )
            y_top_left = np.random.randint(
                self.margin[1], ori_W - self.size[1] - self.margin[1] + 1
            )

            return in_img[x_top_left:x_top_left + self.size[0], y_top_left:y_top_left + self.size[1]]


class SRImagePairRandomCrop(BasicCropTransform):

    def __init__(self, size, sr_factor, margin=0):
        """
        Randomly crop a [lr, hr] pair correspondingly and return patches.
        :param size: patch size, if 0 return the full image without boundaries (margins)
        :param sr_factor: should be int
        :param margin: This margin is corresponded to the HR image, boundaries of the image
        """
        super(SRImagePairRandomCrop, self).__init__(size, margin)
        self.sr_factor = int(sr_factor)

        self.margin = [_//self.sr_factor for _ in self.margin]

    def __call__(self, data):
        in_img, out_img = data
        if self.size is None:
            cropped_data = [
                in_img[self.margin[0]:-self.margin[0], self.margin[1]:-self.margin[1]],
                out_img[self.margin[0]*self.sr_factor:-self.margin[0]*self.sr_factor,
                        self.margin[1]*self.sr_factor:-self.margin[1]*self.sr_factor]
            ]
        else:
            ori_H, ori_W = in_img.shape[:2]
            x_top_left = np.random.randint(
                self.margin[0], ori_H - self.size[0] - self.margin[0]
            )
            y_top_left = np.random.randint(
                self.margin[1], ori_W - self.size[1] - self.margin[1]
            )
            cropped_data = [
                in_img[x_top_left:x_top_left + self.size[0], y_top_left:y_top_left + self.size[1]],
                out_img[
                    x_top_left*self.sr_factor:(x_top_left+self.size[0])*self.sr_factor,
                    y_top_left*self.sr_factor:(y_top_left+self.size[1])*self.sr_factor
                ]
            ]
        return cropped_data


class CentreCrop(BasicCropTransform):
    """
    Crop a patch from the center of the input.
    """
    def __init__(self, size):
        super(CentreCrop, self).__init__(size, 0)

    def __call__(self, in_img):
        ori_H, ori_W = in_img.shape[:2]
        x_top_left = (ori_H - self.size[0]) // 2
        x_top_left = 0 if x_top_left < 0 else x_top_left
        y_top_left = (ori_W - self.size[1]) // 2
        y_top_left = 0 if y_top_left < 0 else y_top_left
        return in_img[x_top_left:x_top_left + self.size[0], y_top_left:y_top_left + self.size[1]]


class ImagePadding(SizeAlign):
    """
    If the input image is smaller than the aimed size on one / two dimension(s), do zero padding
    Input_shape: 2D tuple or int
    output_shape: 2D tuple or int

    pad(x): x -> 3d / 2d numpy array
        assert x.shape = input_shape
        if x.shape[i] < output_shape[i], return output_shape[i]
        if x.shape[i] > output_shape[i], return x.shape[i]

    ipad(x): xp -> 3d / 2d numpy array
        xr = input_shape

    test:
        h, w = [144, 200]
        size = 192
        ip = ImagePadding([h, w], size)
        x = np.random.randn(h, w, 1)
        xp = ip.pad(x)
        xr = ip.ipad(xp)
        print('x', x.shape)
        print('xp', xp.shape)
        print('xr', xr.shape)
        print('ip padding', ip.padding)
        print('x = xr', np.array_equal(x, xr))
    """
    def __init__(self, input_shape, output_shape):
        super(ImagePadding, self).__init__()
        self.input_shape = self.size_align(input_shape, 2)
        self.outputs_shape = self.size_align(output_shape, 2)

        padding = []
        for i, o in zip(self.input_shape, self.outputs_shape):
            padding.append(math.ceil((o - i) / 2))
            padding.append(math.floor((o - i) / 2))
        self.padding = [p if p > 0 else 0 for p in padding]

    def pad(self, x):
        if x.ndim == 3:
            return np.pad(x, (self.padding[:2], self.padding[2:], [0, 0]), 'edge')
        elif x.ndim == 2:
            return np.pad(x, (self.padding[:2], self.padding[2:]), 'edge')

    def ipad(self, x):
        h, w = x.shape[:2]
        return x[self.padding[0]:h-self.padding[1], self.padding[2]:w-self.padding[3]]

