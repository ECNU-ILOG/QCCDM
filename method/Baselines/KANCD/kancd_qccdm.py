# coding: utf-8
# 2023/7/3 @ WangFei

import logging
import torch
import wandb as wb
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, f1_score
from EduCDM import CDM
from runners.commonutils.datautils import transform, get_r_matrix, get_doa_function, get_group_acc
from runners.commonutils.util import PosLinear


class Net(nn.Module):

    def __init__(self, exer_n, student_n, knowledge_n, mf_type, dim, device='cuda', dtype=torch.float64, q_matrix=None):
        self.knowledge_n = knowledge_n
        self.exer_n = exer_n
        self.student_n = student_n
        self.emb_dim = dim
        self.mf_type = mf_type
        self.prednet_input_len = self.knowledge_n
        self.prednet_len1, self.prednet_len2 = 256, 128  # changeable

        super(Net, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.student_n, self.emb_dim)
        self.exercise_emb = nn.Embedding(self.exer_n, self.emb_dim)
        self.knowledge_emb = nn.Parameter(torch.zeros(self.knowledge_n, self.emb_dim))
        self.e_discrimination = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        if mf_type == 'gmf':
            self.k_diff_full = nn.Linear(self.emb_dim, 1)
            self.stat_full = nn.Linear(self.emb_dim, 1)
        elif mf_type == 'ncf1':
            self.k_diff_full = nn.Linear(2 * self.emb_dim, 1)
            self.stat_full = nn.Linear(2 * self.emb_dim, 1)
        elif mf_type == 'ncf2':
            self.k_diff_full1 = nn.Linear(2 * self.emb_dim, self.emb_dim)
            self.k_diff_full2 = nn.Linear(self.emb_dim, 1)
            self.stat_full1 = nn.Linear(2 * self.emb_dim, self.emb_dim)
            self.stat_full2 = nn.Linear(self.emb_dim, 1)

        self.q_matrix = q_matrix
        if q_matrix is not None:
            if not isinstance(q_matrix, torch.Tensor):
                self.q_mask = torch.tensor(q_matrix, dtype=dtype).to(device=device)
            else:
                self.q_mask = q_matrix
            self.q_mask.requires_grad = False

        self.q_neural = torch.randn(self.exer_n, self.knowledge_n).to(device)
        torch.nn.init.xavier_normal_(self.q_neural)
        self.q_neural = torch.sigmoid(self.q_neural)
        self.q_neural = nn.Parameter(self.q_neural)
        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        nn.init.xavier_normal_(self.knowledge_emb)

    def forward(self, stu_id, input_exercise, input_knowledge_point=None):
        # before prednet
        stu_emb = self.student_emb(stu_id)
        exer_emb = self.exercise_emb(input_exercise)
        # get knowledge proficiency
        batch, dim = stu_emb.size()
        stu_emb = stu_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)  # batch, know_n, dim
        knowledge_emb = self.knowledge_emb.repeat(batch, 1).view(batch, self.knowledge_n, -1)  # batch, know_n, dim
        if self.mf_type == 'mf':  # simply inner product
            stat_emb = torch.sigmoid((stu_emb * knowledge_emb).sum(dim=-1, keepdim=False))  # batch, know_n
        elif self.mf_type == 'gmf':
            stat_emb = torch.sigmoid(self.stat_full(stu_emb * knowledge_emb)).view(batch, -1)
        elif self.mf_type == 'ncf1':
            stat_emb = torch.sigmoid(self.stat_full(torch.cat((stu_emb, knowledge_emb), dim=-1))).view(batch, -1)
        elif self.mf_type == 'ncf2':
            stat_emb = torch.sigmoid(self.stat_full1(torch.cat((stu_emb, knowledge_emb), dim=-1)))
            stat_emb = torch.sigmoid(self.stat_full2(stat_emb)).view(batch, -1)
        batch, dim = exer_emb.size()
        exer_emb = exer_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        if self.mf_type == 'mf':
            k_difficulty = torch.sigmoid((exer_emb * knowledge_emb).sum(dim=-1, keepdim=False))  # batch, knowledge_n
        elif self.mf_type == 'gmf':
            k_difficulty = torch.sigmoid(self.k_diff_full(exer_emb * knowledge_emb)).view(batch, -1)
        elif self.mf_type == 'ncf1':
            k_difficulty = torch.sigmoid(self.k_diff_full(torch.cat((exer_emb, knowledge_emb), dim=-1))).view(batch, -1)
        elif self.mf_type == 'ncf2':
            k_difficulty = torch.sigmoid(self.k_diff_full1(torch.cat((exer_emb, knowledge_emb), dim=-1)))
            k_difficulty = torch.sigmoid(self.k_diff_full2(k_difficulty)).view(batch, -1)
        # get exercise discrimination
        e_discrimination = torch.sigmoid(self.e_discrimination(input_exercise))

        # prednet
        if self.q_matrix is not None:
            input_x = e_discrimination * (stat_emb - k_difficulty) * (self.q_neural * (1 - self.q_mask) + self.q_mask)[
                input_exercise]
        else:
            input_x = e_discrimination * (stat_emb - k_difficulty) * input_knowledge_point
        # f = input_x[input_knowledge_point == 1]
        input_x = self.drop_1(torch.tanh(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.tanh(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.view(-1)

    def get_mastery_level(self):
        with torch.no_grad():
            blocks = torch.split(torch.arange(self.student_n).to(device='cuda'), 5)
            mas = []
            for block in blocks:
                stu_emb = self.student_emb(block)
                batch, dim = stu_emb.size()
                stu_emb = stu_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
                knowledge_emb = self.knowledge_emb.repeat(batch, 1).view(batch, self.knowledge_n, -1)
                stat_emb = torch.sigmoid(self.stat_full(stu_emb * knowledge_emb)).view(batch, -1)
                mas.append(stat_emb.detach().cpu().numpy())
        return np.vstack(mas)


class KaNCD(CDM):
    def __init__(self, **kwargs):
        super(KaNCD, self).__init__()
        mf_type = kwargs['mf_type'] if 'mf_type' in kwargs else 'gmf'
        self.q_matrix = kwargs.get('q_matrix', None)
        self.dtype = kwargs.get('dtype', torch.float64)
        self.lambda_reg = kwargs.get('lambda_reg', 0.01)
        self.stu_num = kwargs['student_n']
        self.know_num = kwargs['knowledge_n']
        self.prob_num = kwargs['exer_n']
        self.device = kwargs.get('device', 'cuda')
        self.net = Net(exer_n=kwargs['exer_n'], student_n=kwargs['student_n'], knowledge_n=kwargs['knowledge_n'],
                       mf_type=mf_type, dim=kwargs['dim'],
                       q_matrix=self.q_matrix, device=self.dtype)

    def train(self, np_train, np_test, lr=0.002, device='cpu', epoch=15, batch_size=None, q=None):
        logging.info("traing... (lr={})".format(lr))
        self.net = self.net.to(device)
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        train_data, test_data = [
            transform(q, _[:, 0], _[:, 1], _[:, 2], batch_size)
            for _ in [np_train, np_test]
        ]
        r = get_r_matrix(np_test, self.stu_num, self.prob_num)
        l1_lambda = self.lambda_reg / self.know_num / self.prob_num
        for epoch_i in range(epoch):
            self.net.train()
            epoch_losses = []
            batch_count = 0
            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i):
                batch_count += 1
                user_info, item_info, knowledge_emb, y = batch_data
                user_info: torch.Tensor = user_info.to(device)
                item_info: torch.Tensor = item_info.to(device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(device)
                y: torch.Tensor = y.to(device)
                pred = self.net(user_info, item_info, knowledge_emb)
                bce_loss = loss_function(pred, y)
                if self.q_matrix is not None:
                    l1_reg = self.net.q_neural * (torch.ones_like(self.net.q_mask) - self.net.q_mask)
                    total_loss = bce_loss + l1_lambda * l1_reg.abs().sum()
                else:
                    total_loss = bce_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_losses.append(total_loss.mean().item())

            print("[Epoch %d] average loss: %.6f" % (epoch_i, float(np.mean(epoch_losses))))
            auc, accuracy, rmse, f1, doa = self.eval(test_data, q=q,
                                                     r=r)
            wb.define_metric("epoch")
            wb.log({
                'epoch': epoch_i,
                'auc': auc,
                'acc': accuracy,
                'rmse': rmse,
                'f1': f1,
                'doa': doa,

            })
            print("[Epoch %d] auc: %.6f, accuracy: %.6f, rmse: %.6f, f1: %.6f, DOA: %.6f" % (
                epoch_i, auc, accuracy, rmse, f1, doa))

    def eval(self, test_data, q=None, r=None):
        logging.info('eval ... ')
        self.net = self.net.to(self.device)
        self.net.eval()
        y_true, y_pred = [], []
        mas = self.net.get_mastery_level()
        doa_func = get_doa_function(self.know_num)
        for batch_data in tqdm(test_data, "Evaluating"):
            user_id, item_id, knowledge_emb, y = batch_data
            user_id: torch.Tensor = user_id.to(self.device)
            item_id: torch.Tensor = item_id.to(self.device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(self.device)
            pred = self.net(user_id, item_id, knowledge_emb)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5), \
            np.sqrt(mean_squared_error(y_true, y_pred)), f1_score(y_true, np.array(y_pred) >= 0.5), doa_func(mas,
                                                                                                             q.detach().cpu().numpy(),
                                                                                                             r)
