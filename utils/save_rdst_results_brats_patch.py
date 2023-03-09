import numpy as np
from skimage import io
from os.path import exists, join
from os import makedirs
import torch
import torch.nn.functional as F

from skimage.metrics import peak_signal_noise_ratio


cx, cy = eval(input('input cx, cy'))
hwx, hwy = eval(input('input hwx, hwy'))
rst_folder = input('input rst_folder')
info = '0: LGG_Brats17_TCIA_241_1, 1: HGG_Brats17_CBICA_ASV_1, 2: HGG_Brats17_TCIA_274_1, 3: LGG_Brats17_2013_28_1'
pid = int(input('input pid index\n {}\n'.format(info)))
sid = int(input('input sid\n'))



patient_ids = ['LGG_Brats17_TCIA_241_1', 'HGG_Brats17_CBICA_ASV_1', 'HGG_Brats17_TCIA_274_1', 'LGG_Brats17_2013_28_1',]# 'LGG_Brats17_TCIA_654_1', 'LGG_Brats17_TCIA_462_1', 'LGG_Brats17_2013_1_1', 'LGG_Brats17_TCIA_141_1', 'LGG_Brats17_TCIA_310_1', 'LGG_Brats17_2013_9_1', 'HGG_Brats17_2013_10_1', 'HGG_Brats17_TCIA_149_1', 'HGG_Brats17_CBICA_ATF_1', 'HGG_Brats17_TCIA_322_1', 'HGG_Brats17_CBICA_ANG_1', 'HGG_Brats17_TCIA_296_1', 'HGG_Brats17_CBICA_ABN_1', 'HGG_Brats17_TCIA_117_1', 'HGG_Brats17_TCIA_429_1', 'HGG_Brats17_TCIA_321_1', 'HGG_Brats17_CBICA_ALX_1', 'HGG_Brats17_CBICA_AXO_1', 'HGG_Brats17_TCIA_150_1', 'HGG_Brats17_TCIA_370_1', 'HGG_Brats17_TCIA_314_1', 'HGG_Brats17_2013_21_1', 'HGG_Brats17_2013_7_1', 'HGG_Brats17_TCIA_280_1', 'HGG_Brats17_CBICA_ALN_1', 'HGG_Brats17_CBICA_AME_1']

modalities_brats = ['t1ce', 't1', 't2', 'flair']

patient_ids = [patient_ids[pid]]
slice_indexes = [sid]

image_data_folders = {
    'bicubic': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_BraTS_multi_bicubic_x4/inferences',
    'GT': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_BraTS_multi_bicubic_x4/inferences',

    'EDSR': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_BraTS_multi_EDSRx4/inferences',
    'RDN': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_BraTS_multi_RDNx4/inferences',
    'RCAN': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_BraTS_multi_RCANx4/inferences',
    'HAN': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_BraTS_multi_HANx4/inferences',

    'SwinIR': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_BraTS_multi_SwinIRx4/inferences',
    'RDST-E': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_BraTS_multi_RDSTx4_E1/inferences',
    'RDST-L': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_BraTS_multi_RDSTx4_HRL/inferences',
    'HR': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_BraTS_multi_bicubic_x4/inferences',
    'RDST8-E': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_BraTS_multi_RDSTx4_RDSTB_8_E1/inferences',
    'RDST8-L': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_BraTS_multi_RDSTx4_RDSTB_8_HRL/inferences',

    # 'RDST-base': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_BraTS_multi_RDSTx4/inferences',
    # 'RDST-E-acdc': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_BraTS_multi_RDSTx4_E1_acdc/inferences',
    # 'RDST-E-oasis': '/local/scratch/jz426/TransSR/output/Final_Predictions/FT_SR_BraTS_multi_RDSTx4_E1_oasis/inferences',

}

label_data_folders = {
    'GT': '/local/scratch/jz426/TransSR/Segmentation/BraTS_Seg_GTs',
    'bicubic': '/local/scratch/jz426/TransSR/Segmentation/output/Final_Predictions/Segmentation/BraTS_Segmentation_multi_bicubic_x4_UNet/inferences',

    'EDSR': '/local/scratch/jz426/TransSR/Segmentation/output/Final_Predictions/Segmentation/BraTS_Segmentation_multi_EDSRx4_UNet/inferences',
    'RDN': '/local/scratch/jz426/TransSR/Segmentation/output/Final_Predictions/Segmentation/BraTS_Segmentation_multi_RDNx4_UNet/inferences',
    'RCAN': '/local/scratch/jz426/TransSR/Segmentation/output/Final_Predictions/Segmentation/BraTS_Segmentation_multi_RCANx4_UNet/inferences',
    'HAN': '/local/scratch/jz426/TransSR/Segmentation/output/Final_Predictions/Segmentation/BraTS_Segmentation_multi_HANx4_UNet/inferences',

    'SwinIR': '/local/scratch/jz426/TransSR/Segmentation/output/Final_Predictions/Segmentation/BraTS_Segmentation_multi_SwinIRx4_UNet/inferences',
    'RDST-E': '/local/scratch/jz426/TransSR/Segmentation/output/Final_Predictions/Segmentation/BraTS_Segmentation_multi_RDSTx4_E1_UNet/inferences',
    'RDST-L': '/local/scratch/jz426/TransSR/Segmentation/output/Final_Predictions/Segmentation/BraTS_Segmentation_multi_RDSTx4_HRL_UNet/inferences',
    'HR': '/local/scratch/jz426/TransSR/Segmentation/output/Final_Predictions/Segmentation/BraTS_Segmentation_GT_UNet/inferences',
    'RDST8-E': '/local/scratch/jz426/TransSR/Segmentation/output/Final_Predictions/Segmentation/BraTS_Segmentation_multi_RDSTx4_RDSTB_8_E1_UNet/inferences',
    'RDST8-L': '/local/scratch/jz426/TransSR/Segmentation/output/Final_Predictions/Segmentation/BraTS_Segmentation_multi_RDSTx4_RDSTB_8_HRL_UNet/inferences',

    # 'RDST-base': '/local/scratch/jz426/TransSR/Segmentation/output/Final_Predictions/Segmentation/BraTS_Segmentation_multi_RDSTx4_UNet/inferences',
    # 'RDST-E-acdc': '',
    # 'RDST-E-oasis': '/local/scratch/jz426/TransSR/Segmentation/output/Final_Predictions/Segmentation/BraTS_Segmentation_multi_RDSTx4_E1_oasis_UNet/inferences',
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


def dice_brats(gt_label, pred_label):
    info = '\t'
    for m in ['ET', 'TC', 'WT']:
        if 'ET' in m:
            gt = gt_label == 3
            pred = pred_label == 3
            dice = dice_coef(gt, pred)
            info += 'Dice-ET: {}\t'.format(dice)
        elif 'TC' in m:
            gt = (gt_label == 1) + (gt_label == 3)
            pred = (pred_label == 1) + (pred_label == 3)
            dice = dice_coef(gt, pred)
            info += 'Dice-TC: {}\t'.format(dice)
        elif 'WT' in m:
            gt = (gt_label == 1) + (gt_label == 2) + (gt_label == 3)
            pred = (pred_label == 1) + (pred_label == 2) + (pred_label == 3)
            dice = dice_coef(gt, pred)
            info += 'Dice-WT: {}\t'.format(dice)
    return info


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


def generate_slice_indexes(l, n=5):
    return [_ for _ in range(l//n, l, l//n)]


sr_scale = 4.0
# slice_indexes = [10, 30, 50, 70, 90, 110, 130]

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

    # slice_indexes = generate_slice_indexes(len(l_data))

    print('GT:', pid)

    for sid in slice_indexes:
        print('GT, slice:', pid, sid)
        gt_img_m = gt_imgs_data[sid][sr_scale]

        # multi-modality
        for i, m in enumerate(modalities_brats):
            gt_img = gt_img_m[:, :, i:i+1]
            img_path = join(p_output_dir, '{}_{}_GT_IMG.png'.format(sid, m))
            io.imsave(img_path, (gt_img*255).astype('uint8'))

            gt_patch_path = join(p_output_dir, '{}_{}_GT_patch.png'.format(sid, m))
            gt_patch = gt_img[cx - hwx:cx + hwx, cy - hwy:cy + hwy]
            io.imsave(gt_patch_path, (gt_patch * 255).astype('uint8'))

        # label
        label = l_data[sid]
        label_map = label_render(label)

        gt_img = gt_img_m[:, :, 0:1]

        img_3 = np.concatenate([gt_img, gt_img, gt_img], axis=-1)
        label_on_img = label_map/255 * 0.7 + img_3 * 0.7
        label_on_img = np.clip(label_on_img, 0.0, 1.0)
        label_path = join(p_output_dir, '{}_GT_Label.png'.format(sid))
        io.imsave(label_path, (label_on_img*255).astype('uint8'))

# save SR images and the predicted labels
SR_cases = [
    'bicubic', 'EDSR', 'RDN', 'RCAN', 'HAN', 'SwinIR_lite', 'SwinIR', 'RDST-E', 'RDST-L', 'HR', 'RDST-base',
    'RDST8-E', 'RDST8-L'
    # 'RDST-E-acdc', 'RDST-E-oasis'
]
for case in SR_cases:
    if case not in image_data_folders:
        continue
    if case not in label_data_folders:
        continue
    print('Case:', case)
    for pid in patient_ids:
        print('Case, pid', case, pid)
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

        # slice_indexes = generate_slice_indexes(len(l_data))

        for sid in slice_indexes:
            print('Case, pid, sid', case, pid, sid)
            p_img_m = imgs_data[sid][sr_scale]
            gt_img_m = gt_imgs[sid][sr_scale]

            # multi-modality
            for i, m in enumerate(modalities_brats):
                p_img = p_img_m[:, :, i:i+1]
                gt_img = gt_img_m[:, :, i:i+1]

                # to patch
                p_img = p_img[cx - hwx:cx + hwx, cy - hwy:cy + hwy]
                gt_img = gt_img[cx - hwx:cx + hwx, cy - hwy:cy + hwy]

                img_path = join(p_output_dir, '{}_{}_{}_IMG.png'.format(sid, m, case))
                io.imsave(img_path, (np.clip(p_img, 0, 1)*255).astype('uint8'))

                # metrics
                psnr = peak_signal_noise_ratio(gt_img, p_img, data_range=1)
                f.write('{}:\t {}\t {}\n'.format(sid, m, psnr))

            # label
            label = l_data[sid]
            # to patch
            label = label[cx - hwx:cx + hwx, cy - hwy:cy + hwy]

            label_map = label_render(label)
            # t1ce
            gt_img = gt_img_m[:, :, 0:1]
            p_img = p_img_m[:, :, 0:1]
            # to patch
            p_img = p_img[cx - hwx:cx + hwx, cy - hwy:cy + hwy]
            gt_img = gt_img[cx - hwx:cx + hwx, cy - hwy:cy + hwy]

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
            gt_label = gt_label[cx - hwx:cx + hwx, cy - hwy:cy + hwy]

            seg_error = gt_label != label
            label_on_img[seg_error[:, :, 0]] = [1.0, 0, 0]

            io.imsave(label_path, (label_on_img*255).astype('uint8'))

            dice_repo = dice_brats(gt_label, label)
            f.write('{}:\t'.format(sid) + dice_repo + '\n\n')

        f.close()
print('All figures save at: ', output_dir)
# save metrics

