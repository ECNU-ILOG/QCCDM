import sys
import os
from tqdm import tqdm
import gc
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error
import torch
import torch.nn as nn
import warnings
from runners.commonutils.datautils import transform, get_doa_function, get_r_matrix
warnings.filterwarnings('ignore')
torch.set_default_tensor_type(torch.DoubleTensor)
import wandb as wb

class HierCDF(nn.Module):
    '''
    The hierarchical cognitive diagnosis model
    '''

    def __init__(self, n_user, n_item, n_know, hidden_dim, know_graph: pd.DataFrame, device='cuda'):

        super(HierCDF, self).__init__()

        self.n_user = n_user
        self.n_item = n_item
        self.n_know = n_know
        self.hidden_dim = hidden_dim
        self.know_graph = know_graph

        self.know_edge = nx.DiGraph()  # nx.DiGraph(know_graph.values.tolist())
        for k in range(n_know):
            self.know_edge.add_node(k)
        for edge in know_graph.values.tolist():
            self.know_edge.add_edge(edge[0], edge[1])

        self.topo_order = list(nx.topological_sort(self.know_edge))

        # the conditional mastery degree when parent is mastered
        condi_p = torch.Tensor(n_user, know_graph.shape[0])
        self.condi_p = nn.Parameter(condi_p)

        # the conditional mastery degree when parent is non-mastered
        condi_n = torch.Tensor(n_user, know_graph.shape[0])
        self.condi_n = nn.Parameter(condi_n)

        # the priori mastery degree of parent
        priori = torch.Tensor(n_user, n_know)
        self.priori = nn.Parameter(priori)

        # item representation
        self.item_diff = nn.Embedding(n_item, n_know)
        self.item_disc = nn.Embedding(n_item, 1)

        # embedding transformation
        self.user_contract = nn.Linear(n_know, hidden_dim)
        self.item_contract = nn.Linear(n_know, hidden_dim)

        # Neural Interaction Module (used only in ncd)
        self.cross_layer1 = nn.Linear(hidden_dim, max(int(hidden_dim / 2), 1))
        self.cross_layer2 = nn.Linear(max(int(hidden_dim / 2), 1), 1)

        # layer for featrue cross module
        self.itf = self.ncd

        # param initialization
        nn.init.xavier_normal_(self.priori)
        nn.init.xavier_normal_(self.condi_p)
        nn.init.xavier_normal_(self.condi_n)
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def ncd(self, user_emb: torch.Tensor, item_emb: torch.Tensor, item_offset: torch.Tensor):
        input_vec = (user_emb - item_emb) * item_offset
        x_vec = torch.sigmoid(self.cross_layer1(input_vec))
        x_vec = torch.sigmoid(self.cross_layer2(x_vec))
        return x_vec

    def get_posterior(self, user_ids: torch.LongTensor, device='cpu') -> torch.Tensor:
        n_batch = user_ids.shape[0]
        posterior = torch.rand(n_batch, self.n_know).to(device)
        batch_priori = torch.sigmoid(self.priori[user_ids, :])
        batch_condi_p = torch.sigmoid(self.condi_p[user_ids, :])
        batch_condi_n = torch.sigmoid(self.condi_n[user_ids, :])

        # self.logger.write('batch_priori:{}'.format(batch_priori.requires_grad),'console')

        # for k in range(self.n_know):
        for k in self.topo_order:
            # get predecessor list
            predecessors = list(self.know_edge.predecessors(k))
            predecessors.sort()
            len_p = len(predecessors)

            # for each knowledge k, do:
            if len_p == 0:
                priori = batch_priori[:, k]
                posterior[:, k] = priori.reshape(-1)
                continue

            # format of masks
            fmt = '{0:0%db}' % (len_p)
            # number of parent master condition
            n_condi = 2 ** len_p

            # sigmoid to limit priori to (0,1)
            # priori = batch_priori[:,predecessors]
            priori = posterior[:, predecessors].to(device)

            # self.logger.write('priori:{}'.format(priori.requires_grad),'console')

            pred_idx = self.know_graph[self.know_graph['to'] == k].sort_values(by='from').index
            condi_p = torch.pow(batch_condi_p[:, pred_idx], 1 / len_p).to(device)
            condi_n = torch.pow(batch_condi_n[:, pred_idx], 1 / len_p).to(device)
            margin_p = condi_p * priori
            margin_n = condi_n * (1.0 - priori)

            posterior_k = torch.zeros((1, n_batch)).to(device)

            for idx in range(n_condi):
                # for each parent mastery condition, do:
                mask = fmt.format(idx)
                mask = torch.Tensor(np.array(list(mask)).astype(int)).to(device)

                margin = mask * margin_p + (1 - mask) * margin_n
                margin = torch.prod(margin, dim=1).unsqueeze(dim=0)

                posterior_k = torch.cat([posterior_k, margin], dim=0)
            posterior_k = (torch.sum(posterior_k, dim=0)).squeeze()

            posterior[:, k] = posterior_k.reshape(-1)

        return posterior

    def get_condi_p(self, user_ids: torch.LongTensor, device='cpu') -> torch.Tensor:
        n_batch = user_ids.shape[0]
        result_tensor = torch.rand(n_batch, self.n_know).to(device)
        batch_priori = torch.sigmoid(self.priori[user_ids, :])
        batch_condi_p = torch.sigmoid(self.condi_p[user_ids, :])

        # for k in range(self.n_know):
        for k in self.topo_order:
            # get predecessor list
            predecessors = list(self.know_edge.predecessors(k))
            predecessors.sort()
            len_p = len(predecessors)
            if len_p == 0:
                priori = batch_priori[:, k]
                result_tensor[:, k] = priori.reshape(-1)
                continue
            pred_idx = self.know_graph[self.know_graph['to'] == k].sort_values(by='from').index
            condi_p = torch.pow(batch_condi_p[:, pred_idx], 1 / len_p)
            result_tensor[:, k] = torch.prod(condi_p, dim=1).reshape(-1)

        return result_tensor

    def get_condi_n(self, user_ids: torch.LongTensor, device='cpu') -> torch.Tensor:
        n_batch = user_ids.shape[0]
        result_tensor = torch.rand(n_batch, self.n_know).to(device)
        batch_priori = torch.sigmoid(self.priori[user_ids, :])
        batch_condi_n = torch.sigmoid(self.condi_n[user_ids, :])

        # for k in range(self.n_know):
        for k in self.topo_order:
            # get predecessor list
            predecessors = list(self.know_edge.predecessors(k))
            predecessors.sort()
            len_p = len(predecessors)
            if len_p == 0:
                priori = batch_priori[:, k]
                result_tensor[:, k] = priori.reshape(-1)
                continue
            pred_idx = self.know_graph[self.know_graph['to'] == k].sort_values(by='from').index
            condi_n = torch.pow(batch_condi_n[:, pred_idx], 1 / len_p)
            result_tensor[:, k] = torch.prod(condi_n, dim=1).reshape(-1)

        return result_tensor

    def concat(self, a, b, dim=0):
        if a is None:
            return b.reshape(-1, 1)
        else:
            return torch.cat([a, b], dim=dim)

    def forward(self, user_ids: torch.LongTensor, item_ids: torch.LongTensor, item_know: torch.Tensor,
                device='cpu') -> torch.Tensor:
        '''
        @Param item_know: the item q matrix of the batch
        '''
        user_mastery = self.get_posterior(user_ids, device)
        item_diff = torch.sigmoid(self.item_diff(item_ids))
        item_disc = torch.sigmoid(self.item_disc(item_ids))
        user_factor = torch.tanh(
                self.user_contract(user_mastery * item_know))
        item_factor = torch.sigmoid(
                self.item_contract(item_diff * item_know))

        output = self.itf(user_factor, item_factor, item_disc)

        return output

    def get_mastery_level(self):
        return torch.sigmoid(self.get_posterior(torch.arange(self.n_user), device='cuda').detach().cpu()).numpy()


from EduCDM import CDM


class HierCDM(CDM):
    def __init__(self, n_user, n_item, n_know, hidden_dim, know_graph: pd.DataFrame, device='cuda'):
        self.net = HierCDF(n_user, n_item, n_know, hidden_dim, know_graph, device=device)
        self.know_num = n_know
        self.prob_num = n_item
        self.stu_num = n_user
        self.device = device
        self.losses = []
        self.mas_list = []

    def train(self, np_train, np_test, lr=0.01, epoch=5, loss_factor=1.0, alpha=0.01, q=None, batch_size=None):
        self.net = self.net.to(self.device)
        self._to_device(self.device)
        loss_fn = MyLoss(self.net, nn.NLLLoss, loss_factor)
        optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
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
                optimizer.zero_grad()
                user_ids, item_ids, item_know, y_target = batch_data
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                item_know = item_know.to(self.device)

                y_target = y_target.to(self.device).type(torch.cuda.LongTensor)
                y_pred = self.net.forward(user_ids, item_ids, item_know, self.device)

                output_1 = y_pred
                output_0 = torch.ones(output_1.size()).to(self.device) - output_1
                output = (torch.cat((output_0, output_1), 1) + 1e-10).type(torch.cuda.FloatTensor)
                loss = loss_fn(torch.log(output), y_target, user_ids)
                loss.backward()
                optimizer.step()

                self.pos_clipper([self.net.user_contract, self.net.item_contract])
                self.pos_clipper([self.net.cross_layer1, self.net.cross_layer2])
                epoch_losses.append(loss.mean().item())

            print('epoch = {}, loss={}'.format(epoch_i, float(np.mean(epoch_losses))))
            self.losses.append(float(np.mean(epoch_losses)))
            if test_data is not None:
                auc, acc, rmse, f1, doa = self.eval(test_data, q, r)
                wb.define_metric("epoch")
                wb.log({
                    'epoch': epoch_i,
                    'auc': auc,
                    'acc': acc,
                    'rmse': rmse,
                    'f1': f1,
                    'doa': doa,

                })
                print("[Epoch %d] auc: %.6f, accuracy: %.6f, rmse: %.6f, f1: %.6f, DOA: %.6f" % (
                    epoch_i, auc, acc, rmse, f1, doa))
                print("[Epoch %d] auc: %.6f, accuracy: %.6f, rmse: %.6f, f1: %.6f, doa: %.6f" % (epoch_i, auc, acc, rmse, f1, doa))

    '''
    clip the parameters of each module in the moduleList to nonnegative
    '''

    def pos_clipper(self, module_list: list):
        for module in module_list:
            module.weight.data = module.weight.clamp_min(0)
        return

    def neg_clipper(self, module_list: list):
        for module in module_list:
            module.weight.data = module.weight.clamp_max(0)
        return

    def eval(self, test_data, q=None, r=None) -> pd.DataFrame:
        self.net = self.net.to(self.device)
        self._to_device(self.device)
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
            pred: torch.Tensor = self.net.forward(user_id, item_id, knowledge_emb, self.device)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())
        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5), \
               np.sqrt(mean_squared_error(y_true, y_pred)), f1_score(y_true, np.array(y_pred) >= 0.5), doa_func(mas, q, r)

    def _to_device(self, device):
        self.net.priori = nn.Parameter(self.net.priori.to(device))
        self.net.condi_p = nn.Parameter(self.net.condi_p.to(device))
        self.net.condi_n = nn.Parameter(self.net.condi_n.to(device))
        self.net.user_contract = self.net.user_contract.to(device)
        self.net.item_contract = self.net.item_contract.to(device)
        self.net.item_diff = self.net.item_diff.to(device)
        self.net.item_disc = self.net.item_disc.to(device)
        self.net.cross_layer1 = self.net.cross_layer1.to(device)
        self.net.cross_layer2 = self.net.cross_layer2.to(device)


class MyLoss(nn.Module):
    '''
    The loss function of HierCDM
    '''

    def __init__(self, net: HierCDF, loss_fn: nn.Module, factor=1.0):
        super(MyLoss, self).__init__()
        self.net = net
        self.factor = factor
        self.loss_fn = loss_fn()

    def forward(self, y_pred, y_target, user_ids):
        return self.loss_fn(y_pred, y_target) + self.factor * torch.sum(
            torch.relu(self.net.condi_n[user_ids, :] - self.net.condi_p[user_ids, :]))
