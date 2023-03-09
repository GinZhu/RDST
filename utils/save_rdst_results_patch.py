import numpy as np
from skimage import io
from os.path import exists, join
from os import makedirs
import torch
import torch.nn.functional as F

from skimage.metrics import peak_signal_noise_ratio
import glob


pid = 'OAS1_0033_MR1'
sid = 90
cx, cy = eval(input('input cx, cy'))
hwx, hwy = eval(input('input hwx, hwy'))
rst_folder = input('input rst_folder')
# pid = input('input pid\n')
# sid = int(input('input sid\n'))
# cx, cy = (22, 64)   # 80,24; 22,64;
# hwx, hwy = (20, 32)   # 15, 24; 20, 32

# output_dir = join('/Users/Jin/Code/dev_output/rdst_figures', '{}_{}_yellow'.format(pid, sid))
# if not exists(output_dir):
#     makedirs(output_dir)
#
# root_dir = '/Users/Jin/Code/dev_output/rdst_figures'
# raw_paths = glob.glob(join(root_dir, pid, '{}*.png'.format(sid)))
#
# for rp in raw_paths:
#     img = io.imread(rp)
#     patch = img[cx-hwx:cx+hwx, cy-hwy:cy+hwy]
#     rst_path = join(output_dir, rp.split('/')[-1])
#     io.imsave(rst_path, patch)


patient_ids = ['OAS1_0009_MR1', 'OAS1_0033_MR1', 'OAS1_0023_MR1', 'OAS1_0004_MR1', 'OAS1_0019_MR1',
               'OAS1_0032_MR1', 'OAS1_0029_MR1', 'OAS1_0010_MR1', 'OAS1_0003_MR1']

image_data_folders = {
    'bicubic': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_OASIS_bicubic_x4/inferences',
    'GT': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_OASIS_bicubic_x4/inferences',
    'EDSR': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_OASIS_EDSR_full_x4_p24_b32/inferences',
    'RDN': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_OASIS_RDNx4_p24_b32/inferences',
    'RCAN': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_OASIS_RCANx4_p24_b32/inferences',
    'HAN': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_OASIS_HANx4_p24_b32/inferences',
    'SwinIR_lite': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_OASIS_SwinIRx4_lite_p24_b32/inferences',
    'SwinIR': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_OASIS_SwinIRx4_p24_b32_v2/inferences',
    'RDST-E': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_OASIS_RDST_lite_x4_p24_b32_pn_finetune_UNet_E1x10L1_nopadding/inferences',
    'RDST-L': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_OASIS_RDST_lite_x4_p24_b32_pn_finetune_UNet_labelHRx10L1_nopadding/inferences',
    'HR': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_OASIS_bicubic_x4/inferences',
    'RDST8-E': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_OASIS_RDST_x4_RDSTB_8_E1/inferences',
    'RDST8-L': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_OASIS_RDST_x4_RDSTB_8_HRL/inferences',
}

label_data_folders = {
    'GT': '/local/scratch/jz426/TransSR/Segmentation/OASIS_Seg_GTs',
    'bicubic': '/local/scratch/jz426/TransSR/Segmentation/output/Final_Predictions/Segmentation/OASIS_Segmentation_bicubic_x4_UNet/inferences',
    'EDSR': '/local/scratch/jz426/TransSR/Segmentation/output/Final_Predictions/Segmentation/OASIS_Segmentation_EDSR_full_x4_p24_b32_UNet/inferences',
    'RDN': '/local/scratch/jz426/TransSR/Segmentation/output/Final_Predictions/Segmentation/OASIS_Segmentation_RDNx4_p24_b32_UNet/inferences',
    'RCAN': '/local/scratch/jz426/TransSR/Segmentation/output/Final_Predictions/Segmentation/OASIS_Segmentation_RCANx4_p24_b32_UNet/inferences',
    'HAN': '/local/scratch/jz426/TransSR/Segmentation/output/Final_Predictions/Segmentation/OASIS_Segmentation_HANx4_p24_b32_UNet/inferences',
    'SwinIR_lite': '/local/scratch/jz426/TransSR/Segmentation/output/Final_Predictions/Segmentation/OASIS_Segmentation_SwinIRx4_lite_p24_b32_UNet/inferences',
    'SwinIR': '/local/scratch/jz426/TransSR/Segmentation/output/Final_Predictions/Segmentation/OASIS_Segmentation_SwinIRx4_p24_b32_v2_UNet/inferences',
    'RDST-E': '/local/scratch/jz426/TransSR/Segmentation/output/Final_Predictions/Segmentation/OASIS_Segmentation_RDST_lite_x4_p24_b32_pn_finetune_UNet_E1x10L1_nopadding_UNet/inferences',
    'RDST-L': '/local/scratch/jz426/TransSR/Segmentation/output/Final_Predictions/Segmentation/OASIS_Segmentation_RDST_lite_x4_p24_b32_pn_finetune_UNet_labelHRx10L1_nopadding_UNet/inferences',
    'HR': '/local/scratch/jz426/TransSR/Segmentation/output/Final_Predictions/Segmentation/OASIS_Segmentation_GT_UNet/inferences',

    'RDST8-E': '/local/scratch/jz426/TransSR/Segmentation/output/Final_Predictions/Segmentation/OASIS_Segmentation_RDST_x4_RDSTB_8_E1_UNet/inferences',
    'RDST8-L': '/local/scratch/jz426/TransSR/Segmentation/output/Final_Predictions/Segmentation/OASIS_Segmentation_RDST_x4_RDSTB_8_HRL_UNet/inferences',

}

color_map = {
    3.0: [141, 211, 199],
    2.0: [255, 255, 179],
    1.0: [190, 186, 218]
}


def label_render(l):
    """
    :param l: with shape H x W x 1, float type of range [0, C]
    :return:
    """
    l = l[:, :, 0]
    H, W = l.shape
    map = np.zeros((H, W, 3), 'uint8')
    for c in [1.0, 2.0, 3.0]:
        map[l==c] = color_map[c]
    return map


def dice_coef(gt, pred, eps=1e-6):
    """
    Dice coefficient for segmentation
    gt, pred shoud be 2d numpy arrays
    :param gt: 0 for background 1 for label, H x W
    :param pred: 0 for background 1 for label, H x W
    :param eps: 1e-6 by default
    :return: score between [0, 1]
    """
    return (2*(gt * pred).sum() + eps) / (gt.sum() + pred.sum() + eps)


def dice_T(gt, pred, C=4):
    """
    Dice coefficient for segmentation
    gt, pred shoud be 2d numpy arrays
    :param gt: 0 for background 1 for label, H x W
    :param pred: 0 for background 1 for label, H x W
    :param eps: 1e-6 by default
    :return: score between [0, 1]
    """
    gt_one_hot = F.one_hot(torch.from_numpy(gt).to(torch.long), C).numpy()
    pred_one_hot = F.one_hot(torch.from_numpy(pred).to(torch.long), C).numpy()

    return dice_coef(gt_one_hot[:, :, :, 1:], pred_one_hot[:, :, :, 1:])



sr_scale = 4.0
slice_indexes = [10, 30, 50, 70, 90, 110, 130]

output_dir = '/local/scratch/jz426/TransSR/output/{}'.format(rst_folder)

# create the output folder
if not exists(output_dir):
    makedirs(output_dir)
pid_output_dirs = {}
for pid in patient_ids:
    pid_output_dir = join(output_dir, pid)
    pid_output_dirs[pid] = pid_output_dir
    if not exists(pid_output_dir):
        makedirs(pid_output_dir)

# save GT images and labels
for pid in patient_ids:
    p_output_dir = pid_output_dirs[pid]
    p_data_path = join(
        image_data_folders['GT'], '{}_inference_results.tar'.format(pid)
    )
    p_data = torch.load(p_data_path)
    gt_imgs_data = p_data['gt_imgs']

    # label
    l_data_path = join(
        label_data_folders['GT'], '{}_gt.npz'.format(pid)
    )
    l_data = np.load(l_data_path)['arr_0']

    for sid in slice_indexes:
        gt_img = gt_imgs_data[sid][sr_scale]
        img_path = join(p_output_dir, '{}_GT_IMG.png'.format(sid))
        io.imsave(img_path, (gt_img*255).astype('uint8'))

        # label
        label = l_data[sid]
        label_map = label_render(label)
        img_3 = np.concatenate([gt_img, gt_img, gt_img], axis=-1)
        label_on_img = label_map/255 * 0.7 + img_3 * 0.7
        label_on_img = np.clip(label_on_img, 0.0, 1.0)
        label_path = join(p_output_dir, '{}_GT_Label.png'.format(sid))
        io.imsave(label_path, (label_on_img*255).astype('uint8'))

# save SR images and the predicted labels
SR_cases = [
    'bicubic', 'EDSR', 'RDN', 'RCAN', 'HAN', 'SwinIR_lite', 'SwinIR', 'RDST-E', 'RDST-L', 'HR', 'RDST8-E', 'RDST8-L'
]
for case in SR_cases:
    for pid in patient_ids:
        p_output_dir = pid_output_dirs[pid]
        p_data_path = join(
            image_data_folders[case], '{}_inference_results.tar'.format(pid)
        )
        p_data = torch.load(p_data_path)
        imgs_data = p_data['rec_imgs']

        gt_imgs = torch.load(join(image_data_folders['GT'], '{}_inference_results.tar'.format(pid)))['gt_imgs']

        # label
        l_data_path = join(
            label_data_folders[case], '{}_inference_results.tar'.format(pid)
        )
        l_data = torch.load(l_data_path)['rec_imgs']

        gt_labels = np.load(
            join(label_data_folders['GT'], '{}_gt.npz'.format(pid))
        )['arr_0']

        # metrics
        f = open(join(p_output_dir, '{}.txt'.format(case)), 'w')
        seg_reports = torch.load(
            l_data_path.replace('inferences', 'reports').replace('inference_results', 'eva_reports')
        )['eva_report']

        for sid in slice_indexes:
            p_img = imgs_data[sid][sr_scale]
            gt_img = gt_imgs[sid][sr_scale]

            # to patch
            p_img = p_img[cx-hwx:cx+hwx, cy-hwy:cy+hwy]
            gt_img = gt_img[cx-hwx:cx+hwx, cy-hwy:cy+hwy]

            img_path = join(p_output_dir, '{}_{}_IMG.png'.format(sid, case))
            io.imsave(img_path, (np.clip(p_img, 0, 1)*255).astype('uint8'))

            # label
            label = l_data[sid]
            # to patch
            label = label[cx-hwx:cx+hwx, cy-hwy:cy+hwy]

            label_map = label_render(label)
            img_3 = np.concatenate([p_img, p_img, p_img], axis=-1)
            label_on_img = label_map / 255 * 0.7 + img_3 * 0.7
            label_on_img = np.clip(label_on_img, 0.0, 1.0)
            label_path = join(p_output_dir, '{}_{}_Label.png'.format(sid, case))

            # label the segmentation error
            gt_label = gt_labels[sid]
            # to patch
            gt_label = gt_label[cx-hwx:cx+hwx, cy-hwy:cy+hwy]

            seg_error = gt_label != label
            label_on_img[seg_error[:, :, 0]] = [1.0, 0, 0]

            io.imsave(label_path, (label_on_img*255).astype('uint8'))

            # metrics
            psnr = peak_signal_noise_ratio(gt_img, p_img, data_range=1)
            dice = dice_T(gt_label, label)
            f.write('{}:\t {}\t {}\n'.format(sid, psnr, dice))
        f.close()
# save metrics

