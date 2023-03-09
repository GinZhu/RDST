import numpy as np
from skimage import io
from os.path import exists, join
from os import makedirs
import torch
import torch.nn.functional as F

from skimage.metrics import peak_signal_noise_ratio
import glob


# pid = 'OAS1_0033_MR1'
# sid = 90
cx, cy = eval(input('input cx, cy'))
hwx, hwy = eval(input('input hwx, hwy'))
rst_folder = input('input rst_folder')
info = '0: radiopaedia_10_85902_1, 1: coronacases_004, 2: radiopaedia_40_86625_0, 3: coronacases_001'
pid = int(input('input pid index\n {}\n'.format(info)))
sid = int(input('input sid\n'))
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


patient_ids = ['radiopaedia_10_85902_1', 'coronacases_004', 'radiopaedia_40_86625_0', 'coronacases_001']

patient_ids = [patient_ids[pid]]
slice_indexes = [sid]

image_data_folders = {
    'bicubic': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_COVID_CT_bicubic_x4/inferences',
    'GT': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_COVID_CT_bicubic_x4/inferences',
    'EDSR': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_COVID_CT_EDSRx4/inferences',
    'RDN': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_COVID_CT_RDNx4/inferences',
    'RCAN': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_COVID_CT_RCANx4/inferences',
    'HAN': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_COVID_CT_HANx4/inferences',

    'SwinIR': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_COVID_CT_SwinIRx4/inferences',
    'RDST-E': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_COVID_CT_RDSTx4_E1/inferences',
    'RDST-L': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_COVID_CT_RDSTx4_HRL/inferences',
    'HR': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_COVID_CT_bicubic_x4/inferences',
    'RDST8-E': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_COVID_CT_RDSTx4_RDSTB_8_E1_v2/inferences',
    'RDST8-L': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_COVID_CT_RDSTx4_RDSTB_8_HRL_v2/inferences',

    # 'RDST-base': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_COVID_CT_RDSTx4/inferences',
    # 'RDST-E-acdc': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_COVID_CT_RDSTx4_E1_acdc/inferences',
    # 'RDST-E-oasis': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_COVID_CT_RDSTx4_E1_oasis/inferences',

}

label_data_folders = {
    'GT': '/local/scratch/jz426/TransSR/Segmentation/COVID_CT_Seg_GTs',
    'bicubic': '/local/scratch/jz426/TransSR/Segmentation/output/Final_Predictions/Segmentation/COVID_CT_Segmentation_bicubic_x4_UNet/inferences',

    'EDSR': '/local/scratch/jz426/TransSR/Segmentation/output/Final_Predictions/Segmentation/COVID_CT_Segmentation_EDSRx4_UNet/inferences',
    'RDN': '/local/scratch/jz426/TransSR/Segmentation/output/Final_Predictions/Segmentation/COVID_CT_Segmentation_RDNx4_UNet/inferences',
    'RCAN': '/local/scratch/jz426/TransSR/Segmentation/output/Final_Predictions/Segmentation/COVID_CT_Segmentation_RCANx4_UNet/inferences',
    'HAN': '/local/scratch/jz426/TransSR/Segmentation/output/Final_Predictions/Segmentation/COVID_CT_Segmentation_HANx4_UNet/inferences',


    'SwinIR': '/local/scratch/jz426/TransSR/Segmentation/output/Final_Predictions/Segmentation/COVID_CT_Segmentation_SwinIRx4_UNet/inferences',
    'RDST-E': '/local/scratch/jz426/TransSR/Segmentation/output/Final_Predictions/Segmentation/COVID_CT_Segmentation_RDSTx4_E1_UNet/inferences',
    'RDST-L': '/local/scratch/jz426/TransSR/Segmentation/output/Final_Predictions/Segmentation/COVID_CT_Segmentation_RDSTx4_HRL_UNet/inferences',
    'HR': '/local/scratch/jz426/TransSR/Segmentation/output/Final_Predictions/Segmentation/COVID_CT_Segmentation_GT_UNet/inferences',

    'RDST8-E': '/local/scratch/jz426/TransSR/Segmentation/output/Final_Predictions/Segmentation/COVID_CT_Segmentation_RDSTx4_RDSTB_8_E1_v2_UNet/inferences',
    'RDST8-L': '/local/scratch/jz426/TransSR/Segmentation/output/Final_Predictions/Segmentation/COVID_CT_Segmentation_RDSTx4_RDSTB_8_HRL_v2_UNet/inferences',

    # 'RDST-base': '/local/scratch/jz426/TransSR/Segmentation/output/Final_Predictions/Segmentation/COVID_CT_Segmentation_RDSTx4_UNet/inferences',
    # 'RDST-E-acdc': '',
    # 'RDST-E-oasis': '/local/scratch/jz426/TransSR/Segmentation/output/Final_Predictions/Segmentation/COVID_CT_Segmentation_RDSTx4_E1_oasis_UNet/inferences',
}
color_map = {
    3.0: [141, 211, 199],   # cyan
    2.0: [255, 255, 179],   # yellow
    1.0: [190, 186, 218]    # grey
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

def dice_lesion(gt, pred):
    gtl = gt == 3
    predl = pred==3
    return dice_coef(gtl, predl)


sr_scale = 4.0

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

        gt_patch_path = join(p_output_dir, '{}_GT_patch.png'.format(sid))
        gt_patch = gt_img[cx - hwx:cx + hwx, cy - hwy:cy + hwy]
        io.imsave(gt_patch_path, (gt_patch*255).astype('uint8'))

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
    'bicubic', 'EDSR', 'RDN', 'RCAN', 'HAN', 'SwinIR_lite', 'SwinIR', 'RDST-E', 'RDST-L', 'HR', 'RDST-base',
    'RDST-E-acdc', 'RDST-E-oasis', 'RDST8-E', 'RDST8-L'
]
for case in SR_cases:
    if case not in image_data_folders:
        continue
    if case not in label_data_folders:
        continue
    print('Case:', case)
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
            if case == 'HR':
                img_3 = np.concatenate([gt_img, gt_img, gt_img], axis=-1)
            else:
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
            dice = dice_lesion(gt_label, label)
            f.write('{}:\t {}\t {}\n'.format(sid, psnr, dice))
        f.close()
# save metrics

