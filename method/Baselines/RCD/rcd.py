import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pickle
import wandb as wb
from method.Baselines.RCD.model import Net
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error
from tqdm import tqdm
from runners.commonutils.datautils import transform, get_r_matrix, get_doa_function
from method.Baselines.RCD.utils import  construct_local_map
from runners.commonutils.util import get_number_of_params
from EduCDM import CDM


class RCD(CDM):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = config['device']
        self.config = config
        self.config['local_map'] = construct_local_map(self.config)
        self.net = Net(self.config, self.config['local_map'])
        self.net = self.net.to(self.device)
        self.mas_list = []

    def train(self):
        q = self.config['q']
        data_train, data_valid = [
            transform(q, _[:, 0], _[:, 1], _[:, 2], self.config['batch_size'])
            for _ in [self.config['np_train'], self.config['np_test']
                      ]]
        optimizer = optim.Adam(self.net.parameters(), lr=0.0001)
        print('training RCD...')
        get_number_of_params(self.config['method'], self.net)

        loss_function = nn.NLLLoss()
        for epoch in range(self.config['epoch']):
            running_loss = 0.0
            batch_count = 0
            for batch_data in tqdm(data_train, "Epoch %s" % epoch):
                batch_count += 1
                input_stu_ids, input_exer_ids, input_knowledge_embs, labels = batch_data
                input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(
                    self.device), input_exer_ids.to(
                    self.device), input_knowledge_embs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                output_1 = self.net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
                output_0 = torch.ones(output_1.size()).to(self.device) - output_1
                output = torch.cat((output_0, output_1), 1)
                loss = loss_function(torch.log(output + 1e-10), labels.long())
                loss.backward()
                optimizer.step()
                self.net.apply_clipper()

                running_loss += loss.item()

            auc, acc, rmse, f1, doa = self.predict(test_data=data_valid)
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

    def predict(self, test_data):
        print('predicting...')
        self.net.eval()
        q = self.config['q']
        r = get_r_matrix(self.config['np_test'], self.config['stu_num'], self.config['prob_num'])
        batch_count, batch_avg_loss = 0, 0.0
        y_true, y_pred = [], []
        mas = self.net.get_mastery_level()
        self.mas_list.append(mas)
        doa_func = get_doa_function(know_num=self.config['know_num'])
        for batch_data in tqdm(test_data, "Evaluating"):
            batch_count += 1
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = batch_data
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(
                self.device), input_exer_ids.to(
                self.device), input_knowledge_embs.to(self.device), labels
            output = self.net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
            output = output.view(-1)
            y_pred.extend(output.detach().cpu().tolist())
            y_true.extend(labels.tolist())
        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5), \
            np.sqrt(mean_squared_error(y_true, y_pred)), f1_score(y_true, np.array(y_pred) >= 0.5,
                                                                  ), doa_func(mas, q.detach().cpu().numpy(), r)
