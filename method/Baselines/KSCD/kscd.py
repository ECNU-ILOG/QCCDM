# coding: utf-8
# 2023/7/3 @ WangFei
import wandb as wb
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, f1_score
from EduCDM import CDM
from runners.commonutils.datautils import transform, get_doa_function, get_r_matrix
from runners.commonutils.util import get_number_of_params


class Net(nn.Module):

    def __init__(self, exer_n, student_n, knowledge_n, dim, device='cuda', dtype=torch.float64):
        self.knowledge_n = knowledge_n
        self.exer_n = exer_n
        self.student_n = student_n
        self.emb_dim = dim
        self.prednet_input_len = self.knowledge_n
        self.device = device
        super(Net, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.student_n, self.emb_dim)
        self.exercise_emb = nn.Embedding(self.exer_n, self.emb_dim)
        self.knowledge_emb = nn.Parameter(torch.zeros(self.knowledge_n, self.emb_dim))
        self.disc_mlp = nn.Linear(self.emb_dim, 1)
        self.f_sk = nn.Linear(self.knowledge_n + self.emb_dim, self.knowledge_n)
        self.f_ek = nn.Linear(self.knowledge_n + self.emb_dim, self.knowledge_n)
        self.f_se = nn.Linear(self.knowledge_n, 1)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        nn.init.xavier_normal_(self.knowledge_emb)

    def forward(self, stu_id, input_exercise, input_knowledge_point=None):
        # before prednet
        stu_emb = self.student_emb(stu_id)
        exer_emb = self.exercise_emb(input_exercise)
        stu_ability = torch.sigmoid(stu_emb @ self.knowledge_emb.T)
        diff_emb = torch.sigmoid(exer_emb @ self.knowledge_emb.T)
        disc = torch.sigmoid(self.disc_mlp(exer_emb))
        batch, dim = stu_emb.size()
        stu_emb = stu_ability.unsqueeze(1).repeat(1, self.knowledge_n, 1)
        diff_emb = diff_emb.unsqueeze(1).repeat(1, self.knowledge_n, 1)
        Q_relevant = input_knowledge_point.unsqueeze(2).repeat(1, 1, self.knowledge_n)
        knowledge_emb = self.knowledge_emb.repeat(batch, 1).view(batch, self.knowledge_n, -1)
        s_k_concat = torch.sigmoid(self.f_sk(torch.cat([stu_emb, knowledge_emb], dim=-1)))
        e_k_concat = torch.sigmoid(self.f_ek(torch.cat([diff_emb, knowledge_emb], dim=-1)))
        return torch.sigmoid(disc * self.f_se(torch.mean((s_k_concat - e_k_concat) * Q_relevant, dim=1))).view(-1)

        # res = []
        # for index in range(input_knowledge_point.shape[0]):
        #     q_vector = input_knowledge_point[index, :]
        #     sum = torch.tensor(0.0).view(-1).to(device=self.device)
        #     for know in torch.where(q_vector != 0)[0]:
        #         alpha_ic_prime = torch.sigmoid(self.f_sk(torch.cat([stu_ability[index].view(1, -1), know_emb[know].view(1, -1)], dim=1)))
        #         beta_ic_prime = torch.sigmoid(self.f_ek(torch.cat([diff[index].view(1, -1), know_emb[know].view(1, -1)], dim=1)))
        #         sum += self.f_se(alpha_ic_prime - beta_ic_prime).view(-1)
        #     sum /= torch.where(q_vector != 0)[0].shape[0]
        #     res.append(torch.sigmoid(sum))
        # return torch.tensor(res).to(self.device)

    def get_mastery_level(self):
        return torch.sigmoid(self.student_emb.weight @ self.knowledge_emb.T).detach().cpu().numpy()


class KSCD(CDM):
    def __init__(self, stu_num, prob_num, know_num, dim=20, dtype=torch.float64, device='cuda', wandb=True):
        super(KSCD, self).__init__()
        self.auc_list = []
        self.acc_list = []
        self.rmse_list = []
        self.f1_list = []
        self.dtype = dtype
        self.stu_num = stu_num
        self.know_num = know_num
        self.prob_num = prob_num
        self.device = device
        self.wandb = wandb
        self.net = Net(exer_n=prob_num, student_n=stu_num, knowledge_n=know_num, dim=dim,
                       device=self.device)
        self.mas_list = []

    def train(self, np_train, np_test, q, batch_size, lr=0.01, epoch=10):
        # lr=0.01
        logging.info("traing... (lr={})".format(lr))
        self.net = self.net.to(self.device)
        train_set, valid_set = [
            transform(q, _[:, 0], _[:, 1], _[:, 2], batch_size)
            for _ in [np_train, np_test]
        ]
        r = get_r_matrix(np_test, self.stu_num, self.prob_num)
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        get_number_of_params('kscd', self.net)
        for epoch_i in range(epoch):
            self.net.train()
            epoch_losses = []
            batch_count = 0
            for batch_data in tqdm(train_set, "Epoch %s" % epoch_i):
                batch_count += 1
                user_info, item_info, knowledge_emb, y = batch_data
                user_info: torch.Tensor = user_info.to(self.device)
                item_info: torch.Tensor = item_info.to(self.device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(self.device)
                y: torch.Tensor = y.to(self.device)
                pred = self.net(user_info, item_info, knowledge_emb)
                bce_loss = loss_function(pred, y).requires_grad_(True)
                optimizer.zero_grad()
                bce_loss.backward()
                optimizer.step()

                epoch_losses.append(bce_loss.mean().item())

            print("[Epoch %d] average loss: %.6f" % (epoch_i, float(np.mean(epoch_losses))))
            logging.info("[Epoch %d] average loss: %.6f" % (epoch_i, float(np.mean(epoch_losses))))
            auc, acc, rmse, f1, doa = self.eval(valid_set, q=q, r=r)
            if self.wandb:
                wb.define_metric("epoch")
                wb.log({
                    'epoch': epoch_i,
                    'auc': auc,
                    'acc': acc,
                    'rmse': rmse,
                    'f1': f1,
                    'doa': doa
                })

            self.auc_list.append(auc)
            self.acc_list.append(acc)
            self.rmse_list.append(rmse)
            self.f1_list.append(f1)
            print("[Epoch %d] auc: %.6f, acc: %.6f rmse: %.6f, f1: %.6f, doa %.6f" % (epoch_i, auc, acc, rmse, f1, doa))
            logging.info(
                "[Epoch %d] auc: %.6f, acc: %.6f rmse: %.6f, f1: %.6f, doa %.6f" % (epoch_i, auc, acc, rmse, f1, doa))

    def eval(self, test_data, q=None, r=None):
        logging.info('eval ... ')
        self.net = self.net.to(self.device)
        self.net.eval()
        y_true, y_pred = [], []
        mas = self.net.get_mastery_level()
        self.mas_list.append(mas)
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

    def save(self, filepath):
        torch.save(self.net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.net.load_state_dict(torch.load(filepath, map_location=lambda s, loc: s))
        logging.info("load parameters from %s" % filepath)
