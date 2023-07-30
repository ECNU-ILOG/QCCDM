import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, f1_score
from EduCDM import CDM
import wandb as wb
from runners.commonutils.datautils import transform, get_r_matrix, get_doa_function, get_group_acc
from runners.commonutils.util import PosLinear, get_number_of_params



class Net(nn.Module):

    def __init__(self, exer_n, student_n, knowledge_n, mf_type, dim, device='cuda'):
        self.knowledge_n = knowledge_n
        self.exer_n = exer_n
        self.student_n = student_n
        self.emb_dim = dim
        self.mf_type = mf_type
        self.device = device
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
        input_x = e_discrimination * (stat_emb - k_difficulty) * input_knowledge_point
        # f = input_x[input_knowledge_point == 1]
        input_x = self.drop_1(torch.tanh(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.tanh(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.view(-1)

    def get_mastery_level(self):
        with torch.no_grad():
            blocks = torch.split(torch.arange(self.student_n).to(device=self.device), 5)
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
        self.dtype = kwargs.get('dtype', torch.float64)
        self.stu_num = kwargs['stu_num']
        self.know_num = kwargs['know_num']
        self.prob_num = kwargs['prob_num']
        self.device = kwargs.get('device', 'cuda')
        self.dim = kwargs.get('dim', 20)
        self.net = Net(exer_n=self.prob_num, student_n=self.stu_num, knowledge_n=self.know_num,
                       mf_type=mf_type, dim=self.dim, device=self.device)
        self.wandb = kwargs.get('wandb', True)
        self.mas_list = []

    def train(self, np_train, np_test, lr=0.002, epoch=None, q=None, batch_size=None, sp=False):
        self.net = self.net.to(self.device)
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        train_data, test_data = [
            transform(q, _[:, 0], _[:, 1], _[:, 2], batch_size)
            for _ in [np_train, np_test]
        ]
        get_number_of_params('kancd', self.net)
        r = get_r_matrix(np_test, self.stu_num, self.prob_num)
        for epoch_i in range(epoch):
            self.net.train()
            epoch_losses = []
            batch_count = 0
            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i):
                batch_count += 1
                user_info, item_info, knowledge_emb, y = batch_data
                user_info: torch.Tensor = user_info.to(self.device)
                item_info: torch.Tensor = item_info.to(self.device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(self.device)
                y: torch.Tensor = y.to(self.device)
                pred = self.net(user_info, item_info, knowledge_emb)
                bce_loss = loss_function(pred, y)
                total_loss = bce_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_losses.append(total_loss.mean().item())
            if self.wandb:
                if sp:
                    auc, accuracy, rmse, f1, doa, high, middle, low = self.eval(test_data, q=q,
                                                                                r=r, sp=sp)
                    if self.wandb:
                        wb.define_metric("epoch")
                        wb.log({
                            'epoch': epoch_i,
                            'auc': auc,
                            'acc': accuracy,
                            'rmse': rmse,
                            'f1': f1,
                            'doa': doa,
                            'high': high,
                            'middle': middle,
                            'low': low
                        })
                    print("[Epoch %d] auc: %.6f, accuracy: %.6f, rmse: %.6f, f1: %.6f, DOA: %.6f, high acc: %.6f, "
                          "middle acc: %.6f, low acc: %.6f," % (
                              epoch_i, auc, accuracy, rmse, f1, doa, high, middle, low))
                else:
                    auc, accuracy, rmse, f1, doa = self.eval(test_data, q=q,
                                                             r=r, sp=sp)
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


    def eval(self, test_data, q=None, r=None, sp=False):
        self.net = self.net.to(self.device)
        self.net.eval()
        y_true, y_pred = [], []
        y_true_high, y_pred_high = [], []
        y_true_middle, y_pred_middle = [], []
        y_true_low, y_pred_low = [], []
        if sp:
            high, middle, low = get_group_acc(know_num=self.know_num)
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
            if sp:
                for index, user in enumerate(user_id.detach().cpu().tolist()):
                    if user in high:
                        y_true_high.append(y.tolist()[index])
                        y_pred_high.append(pred.detach().cpu().tolist()[index])
                    elif user in middle:
                        y_true_middle.append(y.tolist()[index])
                        y_pred_middle.append(pred.detach().cpu().tolist()[index])
                    else:
                        y_true_low.append(y.tolist()[index])
                        y_pred_low.append(pred.detach().cpu().tolist()[index])

        if sp:
            return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5), \
            np.sqrt(mean_squared_error(y_true, y_pred)), f1_score(y_true, np.array(y_pred) >= 0.5,
                                                                  ), doa_func(mas, q.detach().cpu().numpy(), r), \
            accuracy_score(y_true_high, np.array(y_pred_high) >= 0.5), accuracy_score(y_true_middle, np.array(y_pred_middle) >= 0.5),accuracy_score(y_true_low, np.array(y_pred_low) >= 0.5)
        else:
            return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5), \
            np.sqrt(mean_squared_error(y_true, y_pred)), f1_score(y_true, np.array(y_pred) >= 0.5,
                                                                  ), doa_func(mas, q.detach().cpu().numpy(),
                                                                              r)

