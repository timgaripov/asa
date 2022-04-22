# Adapted from https://github.com/lyakaap/VAT-pytorch/blob/master/vat.py

import contextlib

import torch
import torch.nn.functional as F


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_bn_stats_off(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats = False

    def switch_bn_stats_on(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats = True

    model.apply(switch_bn_stats_off)
    yield
    model.apply(switch_bn_stats_on)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATLoss(object):
    def __init__(self, xi, radius):
        """VAT loss

        :param xi: hyperparameter of VAT (default: 10.0) [VADA: 1e-6]
        :param eps: hyperparameter of VAT (default: 1.0) [VADA: 3.5]
        :param ip: iteration times of computing adv noise (default: 1) [VADA: 1]
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.radius = radius
        self.iters = 1

    def __call__(self, network, x):
        with torch.no_grad():
            logits = network(x)
            logp = F.log_softmax(logits, dim=1)

        # prepare random unit tensor
        d = torch.randn_like(x)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(network):
            # calc adversarial direction
            for _ in range(self.iters):
                d.requires_grad_()
                logits_hat = network(x + self.xi * d)
                logp_hat = F.log_softmax(logits_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, logp, reduction='batchmean', log_target=True)
                grad_d = torch.autograd.grad(outputs=adv_distance, inputs=d,
                                             only_inputs=True, create_graph=False, retain_graph=False)[0]
                d = _l2_normalize(grad_d)

            # calc LDS
            r_adv = d * self.radius
            x_adv = x + r_adv
            logits_hat = network(x_adv)
            logp_hat = F.log_softmax(logits_hat, dim=1)
            lds = F.kl_div(logp_hat, logp, reduction='batchmean', log_target=True)

        return lds, x_adv.detach()
