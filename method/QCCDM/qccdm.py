import numpy as np
import torch
import wandb as wb
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, f1_score
from tqdm import tqdm
from EduCDM import CDM
from runners.commonutils.datautils import transform, get_r_matrix, get_doa_function
from runners.commonutils.util import PosLinear, NoneNegClipper


class NET(nn.Module):
    def __init__(self, stu_num, prob_num, know_num, mask, q_matrix, mode, device='cpu', num_layers=2, hidden_dim=512,
                 dropout=0.5, dtype=torch.float32, nonlinear='sigmoid', q_aug='single', dim=32):
        super(NET, self).__init__()
        self.stu_num = stu_num
        self.prob_num = prob_num
        self.know_num = know_num
        self.device = device
        self.mode = mode
        self.nonlinear = nonlinear
        self.q_aug = q_aug
        self.dim = dim
        torch.set_default_dtype(dtype)

        if not isinstance(mask, torch.Tensor):
            self.g_mask = torch.tensor(mask, dtype=dtype).to(device)
        else:
            self.g_mask = mask
        self.g_mask.requires_grad = False

        if not isinstance(q_matrix, torch.Tensor):
            self.q_mask = torch.tensor(q_matrix, dtype=dtype).to(device=device)
        else:
            self.q_mask = q_matrix
        self.q_mask.requires_grad = False

        if '1' in mode:
            self.graph = torch.randn(self.know_num, self.know_num).to(device)
            torch.nn.init.xavier_normal_(self.graph)
            # self.graph = 2 * torch.relu(torch.neg(self.graph)) + self.graph
            self.graph = torch.sigmoid(self.graph)
            self.graph = nn.Parameter(self.graph)

        if '2' in mode:
            if self.q_aug == 'single':
                self.q_neural = torch.randn(self.prob_num, self.know_num).to(device)
                torch.nn.init.xavier_normal_(self.q_neural)
                self.q_neural = torch.sigmoid(self.q_neural)
                self.q_neural = nn.Parameter(self.q_neural)
            elif self.q_aug == 'mf':
                self.A = nn.Embedding(self.prob_num, self.dim)
                self.B = nn.Embedding(self.know_num, self.dim)


        self.latent_Zm_emb = nn.Embedding(self.stu_num, self.know_num)
        self.latent_Zd_emb = nn.Embedding(self.prob_num, self.know_num)
        self.e_discrimination = nn.Embedding(self.prob_num, 1)
        if self.nonlinear == 'sigmoid':
            self.nonlinear_func = F.sigmoid
        elif self.nonlinear == 'softplus':
            self.nonlinear_func = F.softplus
        elif self.nonlinear == 'tanh':
            self.nonlinear_func = F.tanh
        else:
            raise ValueError('We do not support such nonlinear function')
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(self.know_num if i == 0 else hidden_dim // pow(2, i - 1), hidden_dim // pow(2, i)))
            layers.append(nn.BatchNorm1d(hidden_dim // pow(2, i)))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim // pow(2, num_layers - 1), 1))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)
        BatchNorm_names = ['layers.{}.weight'.format(4 * i + 1) for i in range(num_layers)]

        for index, (name, param) in enumerate(self.named_parameters()):
            if 'weight' in name:
                if name not in BatchNorm_names:
                    nn.init.xavier_normal_(param)

    def forward(self, stu_id, prob_id, knowledge_point=None):
        latent_zm = self.nonlinear_func(self.latent_Zm_emb(stu_id))
        latend_zd = self.nonlinear_func(self.latent_Zd_emb(prob_id))
        e_disc = torch.sigmoid(self.e_discrimination(prob_id))
        identity = torch.eye(self.know_num).to(self.device)

        if '1' in self.mode:
            if self.nonlinear != 'sigmoid':
                Mas = latent_zm @ (torch.inverse(identity - torch.mul(self.graph, self.g_mask)))
                Mas = torch.sigmoid(self.nonlinear_func(Mas))
                Diff = latend_zd @ (torch.inverse(identity - torch.mul(self.graph, self.g_mask)))
                Diff = torch.sigmoid(self.nonlinear_func(Diff))
                input_ability = Mas - Diff
            else:
                Mas = latent_zm @ (torch.inverse(identity - torch.mul(self.graph, self.g_mask)))
                Mas = torch.sigmoid(Mas)
                Diff = latend_zd @ (torch.inverse(identity - torch.mul(self.graph, self.g_mask)))
                Diff = torch.sigmoid(Diff)
                input_ability = Mas - Diff

        if '2' in self.mode:
            if self.q_aug == 'single':
                input_data = e_disc * input_ability * (self.q_neural * (1 - self.q_mask) + self.q_mask)[prob_id]
            elif self.q_aug == 'mf':
                q_neural = self.A.weight @ self.B.weight.T
                input_data = e_disc * input_ability * (torch.sigmoid(q_neural) * (1 - self.q_mask) + self.q_mask)[prob_id]
        else:
            input_data = e_disc * input_ability * knowledge_point

        return self.layers(input_data).view(-1)

    def apply_clipper(self):
        clipper = NoneNegClipper()
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                layer.apply(clipper)

    def get_mastery_level(self):
        if '1' in self.mode:
            identity = torch.eye(self.know_num)
            if self.nonlinear != 'sigmoid':
                return torch.sigmoid(
                self.nonlinear_func(self.nonlinear_func(self.latent_Zm_emb.weight.detach().cpu()) @ torch.linalg.inv(
                    identity - self.graph.data.detach().cpu() * self.g_mask.detach().cpu()))).numpy()
            else:
                return torch.sigmoid(
                self.nonlinear_func(self.latent_Zm_emb.weight.detach().cpu()) @ torch.linalg.inv(
                    identity - self.graph.data.detach().cpu() * self.g_mask.detach().cpu())).numpy()
        else:
            return torch.sigmoid(self.dil_emb.weight.detach().cpu()).numpy()

    def get_Q_Pseudo(self):
        return (self.q_neural * (1 - self.q_mask) + self.q_mask).detach().cpu().numpy()

    def get_intervention_result(self, dil_emb):
        prob_id = torch.arange(self.prob_num).to(self.device)
        diff_emb = torch.sigmoid(self.diff_emb(prob_id))
        e_diff = torch.sigmoid(self.e_diff(prob_id))
        if '1' in self.mode:
            identity = torch.eye(self.know_num).to(self.device)
            diff_emb = diff_emb @ (torch.linalg.inv(identity - self.graph * self.g_mask))

        input_ability = dil_emb - diff_emb

        if '2' in self.mode:
            input_data = torch.mul(e_diff,
                                   torch.mul(input_ability, (self.q_neural * (1 - self.q_mask) + self.q_mask)[prob_id]))
        else:
            input_data = torch.mul(e_diff, torch.mul(input_ability, self.q_mask[prob_id]))

        return self.layers(input_data).view(-1)


class QCCDM(CDM):
    def __init__(self, stu_num, prob_num, know_num, q_matrix, device='cpu',
                 graph=None, lambda_reg=0.01, mode='12', dtype=torch.float64, num_layers=2, nonlinear='sigmoid',
                 q_aug='single'):
        """
        :param stu_num: number of Student
        :param prob_num: number of Problem
        :param know_num: number of Knowledge Attributes
        :param q_matrix: q_matrix of benchmark
        :param device: running device
        :param graph: causal graph of benchmark
        :param lambda_reg: regulation hyperparameter
        :param mode: '1' only SCM '2' only Q-augmented '12' both
        :param dtype: dtype of tensor
        :param num_layers: number of interactive block
        :param nonlinear: the nonlinear function of SCM
        :param q_aug: the augmentation of Q-Matrix
        """
        super(QCCDM, self).__init__()
        self.lambda_reg = lambda_reg
        self.know_num = know_num
        self.prob_num = prob_num
        self.stu_num = stu_num
        self.mode = mode
        self.causal_graph = graph
        self.device = device
        self.q_aug = q_aug
        self.net = NET(stu_num, prob_num, know_num, self.causal_graph, q_matrix, device=device, mode=mode,
                       dtype=dtype, num_layers=num_layers, nonlinear=nonlinear, q_aug=q_aug).to(device)
        self.mode = mode
        self.mas_list = []

    def train(self, np_train, np_test, batch_size=128, epoch=10, lr=0.002, q=None):
        self.net.train()
        bce_loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        train_data, test_data = [
            transform(q, _[:, 0], _[:, 1], _[:, 2], batch_size)
            for _ in [np_train, np_test]
        ]
        r = get_r_matrix(np_test, self.stu_num, self.prob_num)
        l1_lambda = self.lambda_reg / self.know_num / self.prob_num
        for epoch_i in range(epoch):
            epoch_losses = []
            bce_losses = []
            l1_losses = []
            batch_count = 0
            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i):
                batch_count += 1
                user_id, item_id, knowledge_emb, y = batch_data
                user_id: torch.Tensor = user_id.to(self.device)
                item_id: torch.Tensor = item_id.to(self.device)
                knowledge_emb = knowledge_emb.to(self.device)
                y: torch.Tensor = y.to(self.device)
                pred: torch.Tensor = self.net(user_id, item_id, knowledge_emb)

                bce_loss = bce_loss_function(pred, y)
                bce_losses.append(bce_loss.mean().item())
                if '2' in self.mode:
                    if self.q_aug == 'single':
                        l1_reg = self.net.q_neural * (torch.ones_like(self.net.q_mask) - self.net.q_mask)
                    elif self.q_aug == 'mf':
                        q_neural = torch.sigmoid(self.net.A.weight @ self.net.B.weight.T)
                        l1_reg = q_neural * (torch.ones_like(self.net.q_mask) - self.net.q_mask)
                    l1_losses.append(l1_lambda * l1_reg.abs().sum().mean().item())
                    total_loss = bce_loss + l1_lambda * l1_reg.abs().sum()
                else:
                    total_loss = bce_loss

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=20, norm_type=2)
                optimizer.step()

                if '1' in self.mode:
                    self.net.graph.data = torch.clamp(self.net.graph.data, 0., 1.)
                if '2' in self.mode:
                    if self.q_aug == 'single':
                        self.net.q_neural.data = torch.clamp(self.net.q_neural.data, 0., 1.)

                self.net.apply_clipper()
                epoch_losses.append(total_loss.mean().item())

            print("[Epoch %d] average loss: %.6f, bce loss: %.6f, l1 loss: %.6f" % (
                epoch_i, float(np.mean(epoch_losses)), float(np.mean(bce_losses)), float(np.mean(l1_losses))))

            if test_data is not None:
                auc, accuracy, rmse, f1, doa = self.eval(test_data, q=q, r=r)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f, rmse: %.6f, f1: %.6f, DOA: %.6f" % (
                    epoch_i, auc, accuracy, rmse, f1, doa))

                wb.define_metric("epoch")
                wb.log({
                    'epoch': epoch_i,
                    'auc': auc,
                    'acc': accuracy,
                    'rmse': rmse,
                    'f1': f1,
                    'doa': doa,

                })
    def eval(self, test_data, q=None, r=None):
        self.net = self.net.to(self.device)
        self.net.eval()
        y_true, y_pred = [], []
        mas = self.net.get_mastery_level()
        self.mas_list.append(mas)
        doa_func = get_doa_function(self.know_num)
        for batch_data in tqdm(test_data, "Evaluating"):
            user_id, item_id, know_emb, y = batch_data
            user_id: torch.Tensor = user_id.to(self.device)
            item_id: torch.Tensor = item_id.to(self.device)
            know_emb = know_emb.to(self.device)
            pred: torch.Tensor = self.net(user_id, item_id, know_emb)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())
        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5), \
            np.sqrt(mean_squared_error(y_true, y_pred)), f1_score(y_true, np.array(y_pred) >= 0.5,
                                                                  ), doa_func(mas, q.detach().cpu().numpy(), r)
