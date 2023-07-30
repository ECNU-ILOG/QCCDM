import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pickle
import wandb as wb
import pandas as pd
from model import Net
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error
from tqdm import tqdm

from runners.commonutils.datautils import transform, get_r_matrix, get_doa_function
from utils import CommonArgParser, construct_local_map


def train(config: dict, local_map):
    device = config['device']
    q = config['q']
    data_train, data_valid = [
        transform(q, _[:, 0], _[:, 1], _[:, 2], config['batch_size'])
        for _ in [np_train, np_test]
    ]
    config['mas_list'] = []

    net = Net(config, local_map)
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    print('training model...')

    loss_function = nn.NLLLoss()
    for epoch in range(config['epoch']):
        running_loss = 0.0
        batch_count = 0
        for batch_data in tqdm(data_train, "Epoch %s" % epoch):
            batch_count += 1
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = batch_data
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
                device), input_knowledge_embs.to(device), labels.to(device)
            optimizer.zero_grad()
            output_1 = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
            output_0 = torch.ones(output_1.size()).to(device) - output_1
            output = torch.cat((output_0, output_1), 1)
            loss = loss_function(torch.log(output + 1e-10), labels.long())
            loss.backward()
            optimizer.step()
            net.apply_clipper()

            running_loss += loss.item()

        auc, acc, rmse, f1, doa = predict(config, net, test_data=data_valid)
        print("[Epoch %d] auc: %.6f, accuracy: %.6f, rmse: %.6f, f1: %.6f, DOA: %.6f" % (
            epoch, auc, acc, rmse, f1, doa))
        wb.define_metric("epoch")
        wb.log({
            'epoch': epoch,
            'auc': auc,
            'acc': acc,
            'rmse': rmse,
            'f1': f1,
            'doa': doa
        })
    save(config, config['mas_list'])


def predict(config, net, test_data):
    print('predicting model...')
    net.eval()
    q = config['q']
    r = config['r']
    batch_count, batch_avg_loss = 0, 0.0
    y_true, y_pred = [], []
    mas = net.get_mastery_level()
    tmp = config['mas_lisat'].append(mas)
    config['mas_list'] =tmp
    doa_func = get_doa_function(know_num=config['know_num'])
    for batch_data in tqdm(test_data, "Evaluating"):
        batch_count += 1
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = batch_data
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
            device), input_knowledge_embs.to(device), labels
        output = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
        output = output.view(-1)
        y_pred.extend(output.detach().cpu().tolist())
        y_true.extend(labels.tolist())
    return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5), \
        np.sqrt(mean_squared_error(y_true, y_pred)), f1_score(y_true, np.array(y_pred) >= 0.5,
                                                              ), doa_func(mas, q.detach().cpu().numpy(), r)


def save(config: dict, mas):
    exp_type = config['exp_type']
    method = config['method']
    name = config['name']
    if exp_type == 'gcnlayernum':
        name += '-' + str(config['gcnlayers'])
    elif exp_type == 'test':
        name += '-' + str(config['test_size'])
    elif exp_type == 'parameter':
        name += '-' + str(config['weight_reg']) + '-' + str(config['weight_ssl']) + '-' + str(
            config['temp']) + '-' + str(config['ratio'])
    elif exp_type == 'noise':
        name += '-' + str(config['noise_ratio'])
    elif exp_type == 'tempature':
        name += '-' + str(config['temp'])
    elif exp_type == 'dim':
        name += '-' + str(config['dim'])
    elif exp_type == 'ratio':
        name += '-' + str(config['ratio'])
    if not os.path.exists(f'../exps/logs/{exp_type}/{method}'):
        os.makedirs(f'../exps/logs/{exp_type}/{method}')
    if exp_type == 'cdm':
        mas_file_path = f'../exps/logs/{exp_type}/{method}' + '/' + name + '-Mas' + '.pkl'
        with open(mas_file_path, 'wb') as f:
            pickle.dump(mas, f)
    id_file_path = f'../exps/logs/{exp_type}/{method}' + '/' + name + '-id' + '.pkl'
    with open(id_file_path, 'wb') as f:
        pickle.dump(config['id'], f)


if __name__ == '__main__':
    config = vars(CommonArgParser().parse_args())
    datatype = config['datatype']
    device = config['device']
    config['method'] = 'RCD'
    config.update({
        'stu_num': data_params[datatype]['stu_num'],
        'exer_num': data_params[datatype]['exer_num'],
        'know_num': data_params[datatype]['know_num'],
        'batch_size': data_params[datatype]['batch_size']
    })
    np_data = pd.read_csv('../data/{}/{}TotalData.csv'.format(datatype, datatype),
                          header=None).to_numpy()
    q_np = pd.read_csv('../data/{}/q.csv'.format(datatype),
                       header=None).to_numpy()
    np_train, np_test = train_test_split(np_data, test_size=config['test_size'], random_state=config['seed'])
    config['np_train'] = np_train
    config['np_test'] = np_test
    r = get_r_matrix(np_test, config['stu_num'], config['exer_num'])
    q_tensor = torch.tensor(q_np).to(device)
    config['q'] = q_tensor
    config['r'] = r
    name = f"RCD-{config['datatype']}-seed{config['seed']}"
    config['name'] = name
    tags = ['RCD', config['datatype'], str(config['seed'])]
    run = wb.init(project="LGCDF", name=name,
                  tags=tags,
                  config=config)
    config['id'] = run.id
    train(config, construct_local_map(config))
