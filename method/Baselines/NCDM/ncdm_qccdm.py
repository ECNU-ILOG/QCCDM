# coding: utf-8
# 2021/4/1 @ WangFei
import torch
import logging
import wandb as wb
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, f1_score

from EduCDM import CDM
from runners.commonutils.util import PosLinear
from runners.commonutils.datautils import get_doa_function, get_r_matrix, transform


class Net(nn.Module):

    def __init__(self, knowledge_n, exer_n, student_n, Q_matrix=None, device='cuda', dtype=torch.float64):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(Net, self).__init__()
        self.Q_matrix = Q_matrix
        if self.Q_matrix is not None:
            if not isinstance(Q_matrix, torch.Tensor):
                self.q_mask = torch.tensor(Q_matrix, dtype=dtype).to(device=device)
            else:
                self.q_mask = Q_matrix.to(device=device)
            self.q_neural = torch.randn(exer_n, knowledge_n).to(device)
            torch.nn.init.xavier_normal_(self.q_neural)
            self.q_neural = torch.sigmoid(self.q_neural)
            self.q_neural = nn.Parameter(self.q_neural)

        # prediction sub-net
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, input_exercise, input_knowledge_point):
        # before prednet
        stu_emb = self.student_emb(stu_id)
        stat_emb = torch.sigmoid(stu_emb)
        k_difficulty = torch.sigmoid(self.k_difficulty(input_exercise))
        e_difficulty = torch.sigmoid(self.e_difficulty(input_exercise))  # * 10
        # prednet
        if self.Q_matrix is None:
            input_x = e_difficulty * (stat_emb) * input_knowledge_point
        else:
            input_x = e_difficulty * (stat_emb - k_difficulty) * (self.q_neural * (1 - self.q_mask) + self.q_mask)[
                input_exercise]
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.view(-1)

    def get_mastery_level(self):
        return torch.sigmoid(self.student_emb.weight.detach().cpu()).numpy()


class NCDM(CDM):
    '''Neural Cognitive Diagnosis Model'''

    def __init__(self, knowledge_n, exer_n, student_n, Q_matrix=None, dtype=torch.float64, lambda_reg=0.01, wandb=True, device='cuda'):
        super(NCDM, self).__init__()
        self.ncdm_net = Net(knowledge_n, exer_n, student_n, Q_matrix=Q_matrix, dtype=dtype)
        self.Q_matrix = Q_matrix
        self.stu_num = student_n
        self.know_num = knowledge_n
        self.prob_num = exer_n
        self.lambda_reg = lambda_reg
        self.wb = wandb
        self.device = device


    def train(self, np_train, np_test, epoch=10, q=None, batch_size=None, lr=0.002, silence=False):
        self.ncdm_net = self.ncdm_net.to(self.device)
        self.ncdm_net.train()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.ncdm_net.parameters(), lr=lr)
        l1_lambda = self.lambda_reg / self.know_num / self.prob_num
        train_data, test_data = [
            transform(q, _[:, 0], _[:, 1], _[:, 2], batch_size)
            for _ in [np_train, np_test]
        ]
        r = get_r_matrix(np_test, self.stu_num, self.prob_num)
        for epoch_i in range(epoch):
            epoch_losses = []
            batch_count = 0
            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i):
                batch_count += 1
                user_id, item_id, knowledge_emb, y = batch_data
                user_id: torch.Tensor = user_id.to(self.device)
                item_id: torch.Tensor = item_id.to(self.device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(self.device)
                y: torch.Tensor = y.to(self.device)
                pred: torch.Tensor = self.ncdm_net(user_id, item_id, knowledge_emb)
                bce_loss = loss_function(pred, y)
                if self.Q_matrix is not None:
                    l1_reg = self.ncdm_net.q_neural * (torch.ones_like(self.ncdm_net.q_mask) - self.ncdm_net.q_mask)
                    total_loss = bce_loss + l1_lambda * l1_reg.abs().sum()
                else:
                    total_loss = bce_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_losses.append(total_loss.mean().item())

            print("[Epoch %d] average loss: %.6f" % (epoch_i, float(np.mean(epoch_losses))))

            if test_data is not None:
                auc, acc, rmse, f1, doa = self.eval(test_data, q=q, r=r)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f, rmse: %.6f, f1: %.6f, doa: %.6f" %
                      (epoch_i, auc, acc, rmse, f1, doa))

                if self.wb:
                    wb.define_metric("epoch")
                    wb.log({
                        'epoch': epoch_i,
                        'auc': auc,
                        'acc': acc,
                        'rmse': rmse,
                        'f1': f1,
                        'doa': doa
                    })

    def eval(self, test_data, q=None, r=None):
        self.ncdm_net = self.ncdm_net.to(self.device)
        self.ncdm_net.eval()
        y_true, y_pred = [], []
        mas = self.ncdm_net.get_mastery_level()
        doa_func = get_doa_function(self.know_num)
        for batch_data in tqdm(test_data, "Evaluating"):
            user_id, item_id, knowledge_emb, y = batch_data
            user_id: torch.Tensor = user_id.to(self.device)
            item_id: torch.Tensor = item_id.to(self.device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(self.device)
            pred: torch.Tensor = self.ncdm_net(user_id, item_id, knowledge_emb)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5), \
            np.sqrt(mean_squared_error(y_true, y_pred)), f1_score(y_true, np.array(y_pred) >= 0.5), doa_func(mas, q.detach().cpu().numpy(), r)