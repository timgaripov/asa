# Adapted from
# https://github.com/microsoft/Domain-Adaptation-with-Conditional-Distribution-Matching-and-Generalized-Label-Shift/

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from cvxopt import matrix, solvers

solvers.options['show_progress'] = False


class ImportanceWeighting(nn.Module):
    """
        Implements IWDAN/IWCDAN weighting procedure
        described in https://arxiv.org/abs/2003.04475
        hparams: {
            ma -- moving average coefficient
        }
    """

    def __init__(self, num_classes, hparams):
        super(ImportanceWeighting, self).__init__()
        self.num_classes = num_classes
        self.hparams = hparams
        self.ma = self.hparams['ma']

        self.register_buffer('importance_weights', torch.ones(num_classes))

        self.register_buffer('stat_num_samples', torch.zeros(1, dtype=torch.long))
        self.register_buffer('stat_cov_mat_sum', torch.zeros(num_classes, num_classes))
        self.register_buffer('stat_pseudo_target_label_sum', torch.zeros(num_classes))

    def get_sample_weights(self, source_label):
        source_label_onehot = F.one_hot(source_label, self.num_classes).float()
        return source_label_onehot @ self.importance_weights

    def update_stats(self, source_pred_logits, source_label, target_pred_logits):
        batch_size = source_label.size(0)
        source_label_onehot = F.one_hot(source_label, self.num_classes).float()

        self.stat_pseudo_target_label_sum += torch.sum(F.softmax(target_pred_logits.detach(), dim=-1), dim=0)
        self.stat_cov_mat_sum += torch.mm(F.softmax(source_pred_logits.detach(), dim=-1).transpose(1, 0),
                                          source_label_onehot)
        self.stat_num_samples += batch_size

    # Adapted from
    # https://github.com/microsoft/Domain-Adaptation-with-Conditional-Distribution-Matching-and-Generalized-Label-Shift/network.py
    def update_weights(self, source_y):
        source_y = source_y.detach().cpu().numpy().reshape(-1, 1).astype(np.double)
        target_y = (self.stat_pseudo_target_label_sum / self.stat_num_samples).detach().cpu().numpy()
        target_y = target_y.reshape(-1, 1).astype(np.double)

        cov = (self.stat_cov_mat_sum / self.stat_num_samples).detach().cpu().numpy().astype(np.double)

        P = matrix(np.dot(cov.T, cov), tc="d")
        q = -matrix(np.dot(cov, target_y), tc="d")
        G = matrix(-np.eye(self.num_classes), tc="d")
        h = matrix(np.zeros(self.num_classes), tc="d")
        A = matrix(source_y.reshape(1, -1), tc="d")
        b = matrix([1.0], tc="d")
        sol = solvers.qp(P, q, G, h, A, b)
        new_importance_weights = self.importance_weights.new_tensor(np.array(sol["x"])).view(-1)

        # EMA for the weights
        self.importance_weights = (1 - self.ma) * new_importance_weights + self.ma * self.importance_weights

        # Reset stats
        self.stat_pseudo_target_label_sum.zero_()
        self.stat_cov_mat_sum.zero_()
        self.stat_num_samples.zero_()
