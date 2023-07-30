import torch
import argparse
import numpy as np
from joblib import Parallel, delayed
from torch.utils.data import TensorDataset, DataLoader

from method.Baselines.RCD.build_graph import build_graph4ke, build_graph4se, build_graph4di


class CommonArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(CommonArgParser, self).__init__()
        self.add_argument('--datatype', help='benchmark', default='junyi')
        self.add_argument('--seed', type=int, help='experiment seed', default=1)
        self.add_argument('--device', type=str, default='cuda')
        self.add_argument('--epoch', type=int, default=5,
                          help='The epoch number of training')
        self.add_argument('--lr', type=float, default=0.0001,
                          help='Learning rate')
        self.add_argument('--test', action='store_true',
                          help='Evaluate the model on the testing set in the training process.')
        self.add_argument('--test_size', type=float, default=0.2, help='the test size of benchmark')
        self.add_argument('--exp_type', type=str)



def construct_local_map(config: dict):
    local_map = {
        'k_from_e': build_graph4ke(config, from_e=True),
        'e_from_k': build_graph4ke(config, from_e=False),
        'e_from_s': build_graph4se(config, from_s=True),
        's_from_e': build_graph4se(config, from_s=False),
        # 'directed_graph': build_graph4di(config)
    }
    return local_map
