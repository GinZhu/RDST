from datasets.basic_dataset import MIBasicTrain, MIBasicValid, BasicEvaluation
import numpy as np

"""
Dataset for Stage II:
    1. loading VAE generated data
    2. feeding to ResNet
    3. same function for quick / final evaluation

Test passed.

@Jin (jin.zhu@cl.cam.ac.uk) June 22 2020
"""


class StageIIDataset(MIBasicValid, MIBasicTrain):
    """
    For the training of stage II, need to load the saved data
    """
    def __init__(self, data_path):
        super(StageIIDataset, self).__init__()
        data = np.load(data_path, allow_pickle=True)

        self.training_inputs = []
        self.training_outputs = []
        self.training_ids = []

        self.testing_inputs = []
        self.testing_gts = []
        self.testing_ids = []

        for sample in data:
            if sample['for_training']:
                self.training_inputs.append(sample['vae_output'])
                self.training_outputs.append(sample['gt_img'])
                self.training_ids.append(sample['id'])
            else:
                self.testing_inputs.append(sample['vae_output'])
                self.testing_gts.append(sample['gt_img'])
                self.testing_ids.append(sample['id'])
        # ## input mean or output mean?
        self.mean = np.mean(self.training_outputs, axis=(0, 1, 2))
        self.std = np.std(self.training_outputs, axis=(0, 1, 2))

        # ## evaluation
        self.quick_eva_func = StageIIEvaluation()
        self.final_eva_func = StageIIEvaluation()

    def __len__(self):
        return len(self.training_inputs)

    def __getitem__(self, item):
        img_input = self.training_inputs[item]
        img_output = self.training_outputs[item]

        img_input = self.numpy_2_tensor(img_input)
        img_output = self.numpy_2_tensor(img_output)

        return {'in': img_input, 'out': img_output}

    def test_len(self):
        return len(self.testing_inputs)

    def get_test_pair(self, item):
        img_input = self.testing_inputs[item]
        img_output = self.testing_gts[item]
        img_id = self.testing_ids[item]

        img_input = self.numpy_2_tensor(img_input).unsqueeze(0)

        return {'in': img_input, 'gt': img_output, 'id': img_id}


class StageIIEvaluation(BasicEvaluation):
    def __init__(self):
        super(StageIIEvaluation, self).__init__()
        self.metrics = [
            'rec_psnr',
            'rec_ssim',
        ]

    def __call__(self, rec_img, sample):
        gt_img = sample['gt']
        psnr, ssim = self.psnr_ssim(rec_img, gt_img)
        report = {
            'imgs': [rec_img, gt_img],
            'rec_psnr': psnr,
            'rec_ssim': ssim,
            'id': sample['id']
        }
        return report





