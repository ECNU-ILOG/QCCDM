import os
import pickle
import torch
import torch.nn as nn
import numpy as np

epochs_dict = {
    'Math2': {
        'qccdm': 3,
        'qccdm-c': 1,
        'qccdm-q': 3,
        'ncdm': 3,
        'ncdmq': 3,
        'hiercdf': 1,
        'kancd': 1,
        'kancdq': 1,
        'mirt': 8,
        'dina': 10,
        'kscd':5,
        'rcd':5
    },
    'Math1': {
        'qccdm': 3,
        'qccdm-c': 1,
        'qccdm-q': 1,
        'ncdm': 3,
        'ncdmq': 3,
        'hiercdf': 1,
        'kancd': 1,
        'kancdq': 1,
        'mirt': 15,
        'dina': 10,
        'kscd': 5,
        'rcd': 5
    }, 'junyi': {
        'qccdm': 4,
        'qccdm-c': 1,
        'qccdm-q': 4,
        'ncdm': 5,
        'hiercdf': 5,
        'kancd': 1,
        'mirt': 25,
        'dina': 20,
        'rcd': 5
    },
}


def save(config: dict, mas):
    exp_type = config['exp_type']
    method = config['method']
    name = config['name']
    if exp_type == 'gcnlayers':
        name += '-' + str(config['gcnlayers'])
    elif exp_type == 'test':
        name += '-' + str(config['test_size'])
    elif exp_type == 'dim':
        name += '-' + str(config['dim'])
    elif exp_type == 'keep':
        name += '-' + str(config['keep_prob'])
    elif exp_type == 'reg':
        name += '-' + str(config['weight_reg'])
    elif exp_type == 'leaky':
        name += '-' + str(config['leaky'])
    elif exp_type == 'noise':
        name += '-' + str(config['noise_ratio'])
    elif exp_type == 'sparse':
        name += '-' + 'sparse'
    if not os.path.exists(f'logs/{exp_type}/{method}'):
        os.makedirs(f'logs/{exp_type}/{method}')
    if exp_type == 'cdm':
        mas_file_path = f'logs/{exp_type}/{method}' + '/' + name  +'-Mas' + '.pkl'
        with open(mas_file_path, 'wb') as f:
            pickle.dump(mas, f)
    id_file_path = f'logs/{exp_type}/{method}' + '/' + name + '-id' + '.pkl'
    with open(id_file_path, 'wb') as f:
        pickle.dump(config['id'], f)


class InvertiblePWL(nn.Module):
    """docstring for InvertiblePrior"""

    def __init__(self, vmin=-5, vmax=5, n=100, use_bias=True):
        super(InvertiblePWL, self).__init__()
        self.p = nn.Parameter(torch.randn([n + 1]) / 5)
        self.int_length = (vmax - vmin) / (n - 1)

        self.n = n
        if use_bias:
            self.b = nn.Parameter(torch.randn([1]) + vmin)
        else:
            self.b = vmin
        self.points = nn.Parameter(torch.from_numpy(np.linspace(vmin, vmax, n).astype('float32')).view(1, n),
                                   requires_grad=False)

    def to_positive(self, x):
        return torch.exp(x) + 1e-3

    def forward(self, eps):
        delta_h = self.int_length * self.to_positive(self.p[1:self.n]).detach()
        delta_bias = torch.zeros([self.n]).to(eps.device)

        delta_bias[0] = self.b
        for i in range(self.n - 1):
            delta_bias[i + 1] = delta_bias[i] + delta_h[i]
        index = torch.sum(((eps - self.points) >= 0).long(), 1).detach()  # b * 1 from 0 to n

        start_points = index - 1
        start_points[start_points < 0] = 0
        delta_bias = delta_bias[start_points]
        start_points = torch.squeeze(self.points)[torch.squeeze(start_points)].detach()
        delta_x = eps - start_points.view(-1, 1)

        k = self.to_positive(self.p[index])
        delta_fx = delta_x * k.view(-1, 1)

        o = delta_fx + delta_bias.view(-1, 1)

        return o

    def inverse(self, o):
        delta_h = self.int_length * self.to_positive(self.p[1:self.n]).detach()
        delta_bias = torch.zeros([self.n]).to(o.device)
        delta_bias[0] = self.b
        for i in range(self.n - 1):
            delta_bias[i + 1] = delta_bias[i] + delta_h[i]
        index = torch.sum(((o - delta_bias) >= 0).long(), 1).detach()  # b * 1 from 0 to n
        start_points = index - 1
        start_points[start_points < 0] = 0
        delta_bias = delta_bias[start_points]
        intervel_incre = o - delta_bias.view(-1, 1)
        start_points = torch.squeeze(self.points)[torch.squeeze(start_points)].detach()
        k = self.to_positive(self.p[index])
        delta_x = intervel_incre / k.view(-1, 1)
        eps = delta_x + start_points.view(-1, 1)
        return eps

