from utils.param_loader import ParametersLoader

from datasets.OASIS_dataset import OASISMultiSRTrain, OASISMultiSRTest
from datasets.BraTS_dataset import BraTSMultiSRTrain, BraTSMultiSRTest
from datasets.ACDC_dataset import ACDCMultiSRTrain, ACDCMultiSRTest
from datasets.CovidCT_dataset import CovidCTMultiSRTrain, CovidCTMultiSRTest

from models.trans_sr_trainer import TransSRTrainer
import argparse

# seg loss
# from datasets.OASIS_dataset import OASISSegSRTrain

"""
Example:
    python -W ignore train.py --seg-loss --config-file config_files/colab_sota_sr_example.ini
"""


parser = argparse.ArgumentParser(description='Training Parameters')
parser.add_argument('--config-file', type=str, required=True, metavar='CONFIG',
                    help='Path to config file.')
parser.add_argument('--gpu-id', type=int, metavar='GPU',
                    help='Which gpu to use.')
parser.add_argument('--seg-loss', action='store_true')


args = parser.parse_args()
# do distributed training here
config_file = args.config_file
gpu_id = args.gpu_id
seg_loss = args.seg_loss

paras = ParametersLoader(config_file)

if gpu_id is not None:
    paras.gpu_id = gpu_id
    paras.eva_gpu_id = gpu_id

data_folder = paras.data_folder

# if 'OASIS' in data_folder:
#     ds_train = ds_valid = OASISMetaSRDataset(paras)
# elif 'BraTS' in data_folder:
#     ds_train = ds_valid = BraTSMetaSRDataset(paras)
# elif 'ACDC' in data_folder:
#     ds_train = ds_valid = ACDCMetaSRDataset(paras)
# elif 'COVID' in data_folder:
#     ds_train = ds_valid = COVIDMetaSRDataset(paras)
if 'DIV2K' in data_folder:
    pass
elif 'OASIS' in data_folder:
    if seg_loss:
        pass
        # ds_train = OASISSegSRTrain(paras)
    else:
        ds_train = OASISMultiSRTrain(paras)
        # ds_train = OASISMultiSRWaveletTrain(paras)
    ds_valid = OASISMultiSRTest(paras, paras.validation_patient_ids_oasis)
    # ds_valid = OASISMultiSRWaveletTest(paras, paras.validation_patient_ids_oasis)
elif 'BraTS' in data_folder:
    ds_train = BraTSMultiSRTrain(paras)
    ds_valid = BraTSMultiSRTest(paras, paras.validation_patient_ids_brats)
elif 'ACDC' in data_folder:
    ds_train = ACDCMultiSRTrain(paras)
    ds_valid = ACDCMultiSRTest(paras, paras.validation_patient_ids_acdc)
elif 'COVID' in data_folder:
    ds_train = CovidCTMultiSRTrain(paras)
    ds_valid = CovidCTMultiSRTest(paras, paras.validation_patient_ids_covid)
else:
    raise ValueError('Only support data: [OASIS, DIV2K, BraTS, ACDC, COVID]')

print('DS info:', len(ds_train), 'training samples, and', ds_valid.test_len(), 'testing cases.')

# ## training
trainer = TransSRTrainer(paras, ds_train, ds_valid)

trainer.setup()
trainer.train()

# # ## testing / inference
# for pid in paras.testing_patient_ids:
#     ds_test = OASISSRTest(paras.data_folder, pid, paras.dim, paras.sr_factor)
#     trainer.inference(ds_test, False)

