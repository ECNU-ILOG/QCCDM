import os
from pprint import pprint
import math
import wandb as wb
import random
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

current_path = os.path.abspath('.')
tmp = os.path.dirname(current_path)
path_CDM_ILOG = os.path.dirname(tmp)
path_CDM_QCCDM_runner = path_CDM_ILOG + '\\runners\\QCCDM'
sys.path.insert(0, path_CDM_ILOG)
sys.path.insert(0, path_CDM_QCCDM_runner)
from runners.commonutils.datautils import data_params
from runners.commonutils.util import set_seeds
from runners.QCCDM.utils import epochs_dict
from runners.QCCDM.cdm_runners import get_runner
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--method', default='qccdm', type=str, help='Q-Augmented Causal Cognitive Diagnosis Model',
                    required=True)
parser.add_argument('--datatype', default='junyi', type=str, help='benchmark', required=True)
parser.add_argument('--test_size', default=0.2, type=float, help='test size of benchmark')
parser.add_argument('--epoch', type=int, help='epoch of method')
parser.add_argument('--seed', default=0, type=int, help='seed for exp', required=True)
parser.add_argument('--dtype', default=torch.float64, help='dtype of tensor')
parser.add_argument('--device', default='cuda', type=str, help='device for exp')
parser.add_argument('--batch_size', type=int, help='batch size of benchmark')
parser.add_argument('--exp_type', help='experiment type', type=str, default='cdm')
parser.add_argument('--num_layers', default=2, type=int, help='numer of interactive blocks')
parser.add_argument('--lambda', default=0.01, type=float, help='the coefficient of regulation of Q-augmented')
parser.add_argument('--nonlinear', default='sigmoid', type=str, help='nonlinear function for SCM')
parser.add_argument('--q_aug', default='mf', type=str, help='the augment type of Q-augmentation')
config_dict = vars(parser.parse_args())
if config_dict['method'] == 'qccdm':
    name = f"{config_dict['method']}-{config_dict['nonlinear']}-{config_dict['q_aug']}-{config_dict['datatype']}-seed{config_dict['seed']}"
    tags = [config_dict['method'], config_dict['datatype'], str(config_dict['seed'])]
    config_dict['name'] = name
    config_dict['method'] = f"{config_dict['method']}-{config_dict['nonlinear']}-{config_dict['q_aug']}"
    method = config_dict['method']
else:
    name = f"{config_dict['method']}-{config_dict['datatype']}-seed{config_dict['seed']}"
    method = config_dict['method']
    tags = [config_dict['method'], config_dict['datatype'], str(config_dict['seed'])]
    config_dict['name'] = name


if method == 'qccdm-c':
    config_dict['mode'] = '1'
elif method == 'qccdm-q':
    config_dict['mode'] = '2'
else:
    config_dict['mode'] = '12'


datatype = config_dict['datatype']
if config_dict.get('epoch', None) is None:
    config_dict['epoch'] = epochs_dict[datatype][method]
if config_dict.get('batch_size', None) is None:
    config_dict['batch_size'] = data_params[datatype]['batch_size']

pprint(config_dict)
run = wb.init(project="QCCDM", name=name,
              tags=tags,
              config=config_dict)
config_dict['id'] = run.id


def main(config):
    method = config['method']
    runner = get_runner(method)
    datatype = config['datatype']
    device = config['device']
    dtype = config['dtype']
    torch.set_default_dtype(dtype)
    config.update({
        'stu_num': data_params[datatype]['stu_num'],
        'prob_num': data_params[datatype]['prob_num'],
        'know_num': data_params[datatype]['know_num'],
    })
    set_seeds(config['seed'])
    np_data = pd.read_csv('../../data/{}/{}TotalData.csv'.format(datatype, datatype),
                          header=None).to_numpy()
    q_np = pd.read_csv('../../data/{}/q.csv'.format(datatype),
                       header=None).to_numpy()
    np_train, np_test = train_test_split(np_data, test_size=config['test_size'], random_state=config['seed'])
    exp_type = config['exp_type']
    config['graph'] = pd.read_excel('../../data/{}/ground_truth.xlsx'.format(datatype), header=None).to_numpy()
    config['hier'] = pd.read_csv('../../data/{}/{}hier.csv'.format(datatype, datatype))
    if not os.path.exists(f'logs/{exp_type}'):
        os.makedirs(f'logs/{exp_type}')
    config['np_train'] = np_train
    config['np_test'] = np_test
    config['q'] = torch.tensor(q_np).to(device)
    runner(config)


if __name__ == '__main__':
    sys.exit(main(config_dict))
