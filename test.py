from utils.param_loader import ParametersLoader
from models.trans_sr_tester import TransSRTester
import argparse

"""
Example:
    python -W ignore test.py --config-file 
"""


parser = argparse.ArgumentParser(description='Training Parameters')
parser.add_argument('--config-file', type=str, required=True, metavar='CONFIG',
                    help='Path to config file.')
parser.add_argument('--gpu-id', type=int, metavar='GPU',
                    help='Which gpu to use.')

args = parser.parse_args()
# do distributed training here
config_file = args.config_file
gpu_id = args.gpu_id

paras = ParametersLoader(config_file)

if gpu_id is not None:
    paras.gpu_id = gpu_id
    paras.eva_gpu_id = gpu_id

tester = TransSRTester(paras)

tester.setup()
tester.test()

