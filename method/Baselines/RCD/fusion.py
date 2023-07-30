import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from method.Baselines.RCD.GraphLayer import GraphLayer


class Fusion(nn.Module):
    def __init__(self, config, local_map):
        self.know_num = config['know_num']
        self.exer_num = config['prob_num']
        self.emb_num = config['stu_num']
        self.device = config['device']

        self.k_from_e = local_map['k_from_e'].to(self.device)
        self.e_from_k = local_map['e_from_k'].to(self.device)
        self.s_from_e = local_map['s_from_e'].to(self.device)
        self.e_from_s = local_map['e_from_s'].to(self.device)

        super(Fusion, self).__init__()
        self.k_from_e = GraphLayer(self.k_from_e, self.know_num, self.know_num)
        self.e_from_k = GraphLayer(self.e_from_k, self.know_num, self.know_num)

        self.s_from_e = GraphLayer(self.s_from_e, self.know_num, self.know_num)
        self.e_from_s = GraphLayer(self.e_from_s, self.know_num, self.know_num)

        self.k_attn_fc1 = nn.Linear(2 * self.know_num, 1, bias=True)
        self.k_attn_fc2 = nn.Linear(2 * self.know_num, 1, bias=True)
        self.k_attn_fc3 = nn.Linear(2 * self.know_num, 1, bias=True)

        self.e_attn_fc1 = nn.Linear(2 * self.know_num, 1, bias=True)
        self.e_attn_fc2 = nn.Linear(2 * self.know_num, 1, bias=True)

    def forward(self, kn_emb, exer_emb, all_stu_emb):
        e_k_graph = torch.cat((exer_emb, kn_emb), dim=0)

        k_from_e_graph = self.k_from_e(e_k_graph)
        e_from_k_graph = self.e_from_k(e_k_graph)

        e_s_graph = torch.cat((exer_emb, all_stu_emb), dim=0)
        u_from_e_graph = self.s_from_e(e_s_graph)
        e_from_u_graph = self.e_from_s(e_s_graph)

        # update concepts
        A = kn_emb
        D = k_from_e_graph[self.exer_num:]

        concat_c_3 = torch.cat([A, D], dim=1)

        score3 = self.k_attn_fc3(concat_c_3)
        score3 = F.softmax(score3, dim=1)

        kn_emb = A + score3[:, 0].unsqueeze(1) * D

        # updated exercises
        A = exer_emb
        B = e_from_k_graph[0: self.exer_num]
        C = e_from_u_graph[0: self.exer_num]
        concat_e_1 = torch.cat([A, B], dim=1)
        concat_e_2 = torch.cat([A, C], dim=1)
        score1 = self.e_attn_fc1(concat_e_1)
        score2 = self.e_attn_fc2(concat_e_2)
        score = F.softmax(torch.cat([score1, score2], dim=1), dim=1)  # dim = 1, 按行SoftMax, 行和为1
        exer_emb = exer_emb + score[:, 0].unsqueeze(1) * B + score[:, 1].unsqueeze(1) * C

        # updated students
        all_stu_emb = all_stu_emb + u_from_e_graph[self.exer_num:]

        return kn_emb, exer_emb, all_stu_emb
