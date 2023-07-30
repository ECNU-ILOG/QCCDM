import logging
import numpy as np
import torch
from EduCDM import CDM
from torch import nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, f1_score
import torch.autograd as autograd
import torch.nn.functional as F
from runners.commonutils.datautils import transform
import wandb as wb
class DINANet(nn.Module):
    def __init__(self, user_num, item_num, hidden_dim, max_slip=0.4, max_guess=0.4, device='cuda',
                 dtype=torch.float64, *args, **kwargs):
        super(DINANet, self).__init__()
        self._user_num = user_num
        self._item_num = item_num
        self.step = 0
        self.max_step = 1000
        self.max_slip = max_slip
        self.max_guess = max_guess
        self.know_num = hidden_dim
        self.guess = nn.Embedding(self._item_num, 1)
        self.slip = nn.Embedding(self._item_num, 1)
        self.theta = nn.Embedding(self._user_num, hidden_dim)

    def forward(self, user, item, knowledge, *args):
        theta = self.theta(user)
        slip = torch.squeeze(torch.sigmoid(self.slip(item)) * self.max_slip)
        guess = torch.squeeze(torch.sigmoid(self.guess(item)) * self.max_guess)
        if self.training:
            n = torch.sum(knowledge * (torch.sigmoid(theta) - 0.5), dim=1)
            t, self.step = max((np.sin(2 * np.pi * self.step / self.max_step) + 1) / 2 * 100,
                               1e-6), self.step + 1 if self.step < self.max_step else 0
            return torch.sum(
                torch.stack([1 - slip, guess]).T * torch.softmax(torch.stack([n, torch.zeros_like(n)]).T / t, dim=-1),
                dim=1
            )
        else:
            n = torch.prod(knowledge * (theta >= 0) + (1 - knowledge), dim=1)
            return (1 - slip) ** n * guess ** (1 - n)

    def get_mastery_level(self):
        return (torch.sigmoid(self.theta.weight.detach().cpu()) - 0.5).numpy()


class STEFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x


class STEDINANet(DINANet):
    def __init__(self, user_num, item_num, hidden_dim, max_slip=0.4, max_guess=0.4, *args, **kwargs):
        super(STEDINANet, self).__init__(user_num, item_num, hidden_dim, max_slip, max_guess, *args, **kwargs)
        self.sign = StraightThroughEstimator()

    def forward(self, user, item, knowledge, *args):
        theta = self.sign(self.theta(user))
        slip = torch.squeeze(torch.sigmoid(self.slip(item)) * self.max_slip)
        guess = torch.squeeze(torch.sigmoid(self.guess(item)) * self.max_guess)
        mask_theta = (knowledge == 0) + (knowledge == 1) * theta
        n = torch.prod((mask_theta + 1) / 2, dim=-1)
        return torch.pow(1 - slip, n) * torch.pow(guess, 1 - n)


class DINA(CDM):
    def __init__(self, user_num, item_num, hidden_dim, ste=False, dtype=torch.float64, device='cuda', wandb=True):
        super(DINA, self).__init__()
        if ste:
            self.dina_net = STEDINANet(user_num, item_num, hidden_dim, device=device)
        else:
            self.dina_net = DINANet(user_num, item_num, hidden_dim, dtype=dtype, device=device)

        self.know_num = hidden_dim
        self.prob_num = item_num
        self.device = device
        self.wandb = wandb

    def train(self, np_train, np_test, q, batch_size, epoch=10, lr=0.001) -> ...:
        self.dina_net = self.dina_net.to(self.device)
        loss_function = nn.BCELoss()
        trainer = torch.optim.Adam(self.dina_net.parameters(), lr)
        train_data, test_data = [
            transform(q, _[:, 0], _[:, 1], _[:, 2], batch_size)
            for _ in [np_train, np_test]
        ]
        for e in range(epoch):
            losses = []
            for batch_data in tqdm(train_data, "Epoch %s" % e):
                user_id, item_id, knowledge, response = batch_data
                user_id: torch.Tensor = user_id.to(self.device)
                item_id: torch.Tensor = item_id.to(self.device)
                knowledge: torch.Tensor = knowledge.to(self.device)
                predicted_response: torch.Tensor = self.dina_net(user_id, item_id, knowledge)
                response: torch.Tensor = response.to(self.device)
                bce_loss = loss_function(predicted_response, response)
                total_loss = bce_loss
                # back propagation
                trainer.zero_grad()
                total_loss.backward()
                trainer.step()

                losses.append(total_loss.mean().item())
            print("[Epoch %d] LogisticLoss: %.6f" % (e, float(np.mean(losses))))

            if test_data is not None:
                auc, accuracy, rmse, f1 = self.eval(test_data)
                if self.wandb:
                    wb.define_metric("epoch")
                    wb.log({
                        'epoch': e,
                        'auc': auc,
                        'acc': accuracy,
                        'rmse': rmse,
                        'f1': f1,
                    })
                print("[Epoch %d] auc: %.6f, accuracy: %.6f, rmse: %.6f, f1: %.6f" % (e, auc, accuracy, rmse, f1))

    def eval(self, test_data) -> tuple:
        self.dina_net = self.dina_net.to(self.device)
        self.dina_net.eval()
        y_pred = []
        y_true = []
        for batch_data in tqdm(test_data, "evaluating"):
            user_id, item_id, knowledge, response = batch_data
            user_id: torch.Tensor = user_id.to(self.device)
            item_id: torch.Tensor = item_id.to(self.device)
            knowledge: torch.Tensor = knowledge.to(self.device)
            pred: torch.Tensor = self.dina_net(user_id, item_id, knowledge)
            y_pred.extend(pred.tolist())
            y_true.extend(response.tolist())
        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5), \
            np.sqrt(mean_squared_error(y_true, y_pred)), f1_score(y_true, np.array(y_pred) >= 0.5)

    def save(self, filepath):
        torch.save(self.dina_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.dina_net.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)
