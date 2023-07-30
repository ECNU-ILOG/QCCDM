import torch
import torch.nn as nn
import torch.nn.functional as F
from method.Baselines.RCD.fusion import Fusion
import dgl

class Net(nn.Module):
    def __init__(self, config, local_map):
        self.device = config['device']
        self.know_num = config['know_num']
        self.exer_num = config['prob_num']
        self.stu_num = config['stu_num']
        self.prednet_input_len = self.know_num
        self.prednet_len1, self.prednet_len2 = 512, 256
        self.k_from_e = local_map['k_from_e'].to(self.device)
        self.e_from_k = local_map['e_from_k'].to(self.device)
        self.s_from_e = local_map['s_from_e'].to(self.device)
        self.e_from_s = local_map['e_from_s'].to(self.device)

        super(Net, self).__init__()

        # network structure
        self.student_emb = nn.Embedding(self.stu_num, self.know_num)
        self.knowledge_emb = nn.Embedding(self.know_num, self.know_num)
        self.exercise_emb = nn.Embedding(self.exer_num, self.know_num)

        self.k_index = torch.LongTensor(list(range(self.know_num))).to(self.device)
        self.stu_index = torch.LongTensor(list(range(self.stu_num))).to(self.device)
        self.exer_index = torch.LongTensor(list(range(self.exer_num))).to(self.device)

        self.FusionLayer1 = Fusion(config, local_map)
        self.FusionLayer2 = Fusion(config, local_map)

        self.prednet_full1 = nn.Linear(2 * self.know_num, self.know_num, bias=False)
        # self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(2 * self.know_num, self.know_num, bias=False)
        # self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(1 * self.know_num, 1)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, kn_r):
        all_stu_emb = self.student_emb(self.stu_index).to(self.device)
        exer_emb = self.exercise_emb(self.exer_index).to(self.device)
        kn_emb = self.knowledge_emb(self.k_index).to(self.device)
        # Fusion layer 1
        kn_emb1, exer_emb1, all_stu_emb1 = self.FusionLayer1(kn_emb, exer_emb, all_stu_emb)
        # Fusion layer 2
        kn_emb2, exer_emb2, all_stu_emb2 = self.FusionLayer2(kn_emb1, exer_emb1, all_stu_emb1)

        # get batch student data
        batch_stu_emb = all_stu_emb2[stu_id]  # 32 123
        batch_stu_vector = batch_stu_emb.repeat(1, batch_stu_emb.shape[1]).reshape(batch_stu_emb.shape[0],
                                                                                   batch_stu_emb.shape[1],
                                                                                   batch_stu_emb.shape[1])

        # get batch exercise data
        batch_exer_emb = exer_emb2[exer_id]  # 32 123
        batch_exer_vector = batch_exer_emb.repeat(1, batch_exer_emb.shape[1]).reshape(batch_exer_emb.shape[0],
                                                                                      batch_exer_emb.shape[1],
                                                                                      batch_exer_emb.shape[1])

        # get batch knowledge concept data
        kn_vector = kn_emb2.repeat(batch_stu_emb.shape[0], 1).reshape(batch_stu_emb.shape[0], kn_emb2.shape[0],
                                                                      kn_emb2.shape[1])

        # Cognitive diagnosis
        preference = torch.sigmoid(self.prednet_full1(torch.cat((batch_stu_vector, kn_vector), dim=2)))
        diff = torch.sigmoid(self.prednet_full2(torch.cat((batch_exer_vector, kn_vector), dim=2)))
        o = torch.sigmoid(self.prednet_full3(preference - diff))

        sum_out = torch.sum(o * kn_r.unsqueeze(2), dim=1)
        count_of_concept = torch.sum(kn_r, dim=1).unsqueeze(1)
        output = sum_out / count_of_concept
        return output

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

    def get_mastery_level(self):
        all_stu_emb = self.student_emb(self.stu_index).to(self.device)
        exer_emb = self.exercise_emb(self.exer_index).to(self.device)
        kn_emb = self.knowledge_emb(self.k_index).to(self.device)
        # Fusion layer 1
        kn_emb1, exer_emb1, all_stu_emb1 = self.FusionLayer1(kn_emb, exer_emb, all_stu_emb)
        # Fusion layer 2
        kn_emb2, exer_emb2, all_stu_emb2 = self.FusionLayer2(kn_emb1, exer_emb1, all_stu_emb1)
        batch_stu_emb = all_stu_emb2[self.stu_index]
        # get batch knowledge concept data
        kn_vector = kn_emb2.repeat(batch_stu_emb.shape[0], 1).reshape(batch_stu_emb.shape[0], kn_emb2.shape[0],
                                                                      kn_emb2.shape[1])
        batch_stu_vector = batch_stu_emb.repeat(1, batch_stu_emb.shape[1]).reshape(batch_stu_emb.shape[0],
                                                                                   batch_stu_emb.shape[1],
                                                                                   batch_stu_emb.shape[1])
        preference = torch.sigmoid(self.prednet_full1(torch.cat((batch_stu_vector, kn_vector), dim=2)))
        o = torch.sigmoid(self.prednet_full3(preference))
        kn_r = torch.ones(size=(self.stu_num, self.know_num)).to(self.device)
        sum_out = torch.sum(o * kn_r.unsqueeze(2), dim=1)
        count_of_concept = torch.sum(kn_r, dim=1).unsqueeze(1)
        output = sum_out / count_of_concept
        return torch.sigmoid(self.student_emb.weight.detach().cpu()).numpy()


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)




