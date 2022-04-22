from collections import OrderedDict

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_hidden, depth=3, spectral=False):
        super(MLP, self).__init__()
        self.n_input = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        sizes = [n_inputs] + [n_hidden] * (depth - 1) + [n_outputs]
        self.net = nn.Sequential()
        for i in range(depth):
            n_in, n_out = sizes[i:i + 2]
            layer = nn.Linear(n_in, n_out)
            name = 'FC'
            if spectral:
                name = 'FC (spectral)'
                layer = nn.utils.spectral_norm(layer)
            self.net.add_module(f'{name} [{i}]', layer)
            if i != depth - 1:
                self.net.add_module(f'Leaky ReLU [{i}]', nn.LeakyReLU())

    def forward(self, x):
        return self.net(x)


class DiscriminatorOutputHistory(nn.Module):
    def __init__(self, max_size):
        super(DiscriminatorOutputHistory, self).__init__()
        self.max_size = max_size
        self.register_buffer('buffer', torch.zeros(max(1, max_size)))
        self.register_buffer('size', torch.zeros(1, dtype=torch.long))

        self.size[0] = 0

    def update(self, batch):
        batch_size = batch.size(0)
        excess = max(0, self.size.item() + batch_size - self.max_size)
        if excess > 0:
            if excess > self.max_size:
                self.size[0] = 0
            else:
                self.buffer[:-excess] = self.buffer.clone()[excess:]
                self.size[0] -= excess
        num_new_items = min(batch_size, self.max_size)
        if num_new_items > 0:
            self.buffer[self.size.item():self.size.item() + num_new_items] = batch[-num_new_items:]
        self.size[0] += num_new_items

    def forward(self):
        return self.buffer[:self.size]


class Discriminator(nn.Module):
    """
        hparams: {
            num_hidden -- number of units in the hidden layer;
            depth -- depth of the discriminator network;
            spectral -- if True, apply spectral norm to linear layers;
            history_size -- size of the history buffers;
            grad_penalty_weight [optinal] -- weight of the gradient penalty (used by subclasses);
        }
    """
    def __init__(self, in_dim, hparams):
        super(Discriminator, self).__init__()
        self.in_dim = in_dim
        self.hparams = hparams
        self.network = MLP(in_dim, 1, self.hparams['num_hidden'], self.hparams['depth'],
                           self.hparams['spectral'])

        self.history_src = DiscriminatorOutputHistory(self.hparams['history_size'])
        self.history_trg = DiscriminatorOutputHistory(self.hparams['history_size'])

        self.grad_penalty_weight = self.hparams.get('grad_penalty_weight', 0.0)

    def forward(self, x):
        return self.network(x)

    def alignment_loss(self, z_src, z_trg, update_history=False):
        raise NotImplementedError('subclass of Discriminator must implement alignment loss')

    def disc_loss(self, z_src, z_trg, update_history=False):
        disc_stats = OrderedDict()
        disc_extra_stats = OrderedDict()

        disc_out_src = self.network(z_src).view(-1)
        disc_out_trg = self.network(z_trg).view(-1)

        disc_out = torch.cat((disc_out_src, disc_out_trg), dim=0)
        disc_labels = torch.cat((
            torch.zeros_like(disc_out_src),
            torch.ones_like(disc_out_trg)
        ))
        disc_log_loss = F.binary_cross_entropy_with_logits(disc_out, disc_labels)

        disc_grad_loss = torch.zeros_like(disc_log_loss)

        if update_history:
            self.history_src.update(disc_out_src.detach())
            self.history_trg.update(disc_out_trg.detach())

        history_values_src = self.history_src()
        history_values_trg = self.history_trg()

        disc_stats['d_loss'] = disc_log_loss.item()

        disc_extra_stats['d_src_hmin'] = history_values_src.min().item() if history_values_src.size(0) > 0 else 0.0
        disc_extra_stats['d_src_min'] = disc_out_src.min().item()
        disc_extra_stats['d_src_mean'] = disc_out_src.mean().item()
        disc_extra_stats['d_src_max'] = disc_out_src.max().item()
        disc_extra_stats['d_src_hmax'] = history_values_src.max().item() if history_values_src.size(0) > 0 else 0.0

        disc_extra_stats['d_trg_hmin'] = history_values_trg.min().item() if history_values_trg.size(0) > 0 else 0.0
        disc_extra_stats['d_trg_min'] = disc_out_trg.min().item()
        disc_extra_stats['d_trg_mean'] = disc_out_trg.mean().item()
        disc_extra_stats['d_trg_max'] = disc_out_trg.max().item()
        disc_extra_stats['d_trg_hmax'] = history_values_trg.max().item() if history_values_trg.size(0) > 0 else 0.0

        return disc_log_loss, disc_grad_loss, disc_stats, disc_extra_stats


class LogLossZSDiscriminator(Discriminator):
    def alignment_loss(self, z_src, z_trg, update_history=False):
        disc_out_src = self.network(z_src).view(-1)
        disc_out_trg = self.network(z_trg).view(-1)

        disc_out = torch.cat((disc_out_src, disc_out_trg), dim=0)
        disc_labels = torch.cat((
            torch.zeros_like(disc_out_src),
            torch.ones_like(disc_out_trg),
        ))

        return -F.binary_cross_entropy_with_logits(disc_out, disc_labels), dict()


class LogLossNSDiscriminator(Discriminator):
    def alignment_loss(self, z_src, z_trg, update_history=False):
        disc_out_src = self.network(z_src).view(-1)
        disc_out_trg = self.network(z_trg).view(-1)

        disc_out = torch.cat((disc_out_src, disc_out_trg), dim=0)
        disc_labels = torch.cat((
            torch.ones_like(disc_out_src),
            torch.zeros_like(disc_out_trg),
        ))

        return F.binary_cross_entropy_with_logits(disc_out, disc_labels), dict()


class OptimalTransportLossDiscriminator(Discriminator):
    def __init__(self, in_dim, hparams, dist_fn):
        super(OptimalTransportLossDiscriminator, self).__init__(in_dim, hparams)
        self.dist_fn = dist_fn

    def alignment_loss(self, z_src, z_trg, update_history=False):
        disc_out_src, disc_out_trg = self.network(z_src).view(-1), self.network(z_trg).view(-1)

        history_values_src = self.history_src()
        history_values_trg = self.history_trg()

        v_sort_src, _ = torch.sort(torch.cat((history_values_src, disc_out_src), dim=0))
        v_sort_trg, _ = torch.sort(torch.cat((history_values_trg, disc_out_trg), dim=0))

        loss = torch.mean(self.dist_fn(v_sort_src - v_sort_trg))

        stats = OrderedDict()

        if update_history:
            self.history_src.update(disc_out_src.detach())
            self.history_trg.update(disc_out_trg.detach())

        return loss, stats


class OptimalTransportLossAbsDiscriminator(OptimalTransportLossDiscriminator):
    def __init__(self, in_dim, hparams):
        super(OptimalTransportLossAbsDiscriminator, self).__init__(in_dim, hparams, dist_fn=torch.abs)


class OptimalTransportLossSqDiscriminator(OptimalTransportLossDiscriminator):
    def __init__(self, in_dim, hparams):
        super(OptimalTransportLossSqDiscriminator, self).__init__(in_dim, hparams, dist_fn=torch.square)


class SupportLossDiscriminator(Discriminator):
    def __init__(self, in_dim, hparams, dist_fn):
        super(SupportLossDiscriminator, self).__init__(in_dim, hparams)
        self.dist_fn = dist_fn

    def alignment_loss(self, z_src, z_trg, update_history=False):
        disc_out_src = self.network(z_src).view(-1)
        disc_out_trg = self.network(z_trg).view(-1)

        history_values_src = self.history_src()
        history_values_trg = self.history_trg()

        v_src = torch.cat((history_values_src, disc_out_src), dim=0)
        v_trg = torch.cat((history_values_trg, disc_out_trg), dim=0)

        dist_matrix = self.dist_fn(v_src[:, None] - v_trg[None, :])

        src_dist_min, _ = torch.min(dist_matrix, dim=1)
        trg_dist_min, _ = torch.min(dist_matrix, dim=0)

        loss_src = torch.mean(src_dist_min)
        loss_trg = torch.mean(trg_dist_min)
        loss = loss_src + loss_trg

        stats = OrderedDict()
        stats['a_l_src'] = loss_src.item()
        stats['a_l_trg'] = loss_trg.item()

        if update_history:
            self.history_src.update(disc_out_src.detach())
            self.history_trg.update(disc_out_trg.detach())

        return loss, stats


class SupportLossAbsDiscriminator(SupportLossDiscriminator):
    def __init__(self, in_dim, hparams):
        super(SupportLossAbsDiscriminator, self).__init__(in_dim, hparams, dist_fn=torch.abs)


class SupportLossSqDiscriminator(SupportLossDiscriminator):
    def __init__(self, in_dim, hparams):
        super(SupportLossSqDiscriminator, self).__init__(in_dim, hparams, dist_fn=torch.square)


class SBetaDiscriminator(Discriminator):
    """
        Implements sDANN discriminator
        proposed in "Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment"
        http://proceedings.mlr.press/v97/wu19f.html

        https://github.com/yifan12wu/da-relax
        extra_hparams: {
            beta
        }
    """

    def __init__(self, in_dim, hparams):
        super(SBetaDiscriminator, self).__init__(in_dim, hparams)
        self.beta = self.hparams['beta']

    @staticmethod
    def soft_relu(x):
        """Compute log(1 + exp(x)) with numerical stability.

        Can be used for getting differentiable nonnegative outputs.
        Might also be useful in other cases, e.g.:
            log(sigmoid(x)) = x - soft_relu(x) = - soft_relu(-x).
            log(1 - sigmoid(x)) = - soft_relu(x)
        """
        return ((-x.abs()).exp() + 1.0).log() + torch.nn.functional.relu(x)

    @staticmethod
    def js_div(v_src, v_trg):
        part1 = - SBetaDiscriminator.soft_relu(v_src).mean()
        part2 = - SBetaDiscriminator.soft_relu(-v_trg).mean()
        return part1 + part2 + math.log(4.0), part1, part2

    def disc_loss(self, z_src, z_trg, update_history=False):
        disc_stats = OrderedDict()
        disc_extra_stats = OrderedDict()

        v_src = self.network(z_src).view(-1)
        v_trg = self.network(z_trg).view(-1)

        n_src = v_src.size(0)
        n_src_selected = int(n_src / (1.0 + self.beta))
        v_src_selected = torch.topk(v_src, n_src_selected, largest=True, sorted=False)[0]

        div, div_src, div_trg = SBetaDiscriminator.js_div(v_src_selected, v_trg)
        loss = -div

        grad_loss = torch.zeros_like(loss)

        disc_stats['d_loss'] = loss.item()
        disc_extra_stats['d_l_src'] = div_src.item()
        disc_extra_stats['d_l_trg'] = div_trg.item()
        disc_extra_stats['d_src_m'] = v_src_selected.size(0) * 1.0 / v_src.size(0)

        return loss, grad_loss, disc_stats, disc_extra_stats

    def alignment_loss(self, z_src, z_trg, update_history=False):
        stats = OrderedDict()

        v_src = self.network(z_src).view(-1)
        v_trg = self.network(z_trg).view(-1)

        n_src = v_src.size(0)
        n_src_selected = int(n_src / (1.0 + self.beta))
        v_src_selected = torch.topk(v_src, n_src_selected, largest=True, sorted=False)[0]

        loss, div_src, div_trg = SBetaDiscriminator.js_div(v_src_selected, v_trg)

        return loss, stats


class IWDiscriminator(nn.Module):
    """
        Implements IWDAN/IWCDAN discriminator
        proposed in "Domain Adaptation with Conditional Distribution Matching and Generalized Label Shift"
        https://arxiv.org/abs/2003.04475
        https://github.com/microsoft/Domain-Adaptation-with-Conditional-Distribution-Matching-and-Generalized-Label-Shift
    """

    def __init__(self, in_dim, hparams):
        super(IWDiscriminator, self).__init__()
        self.in_dim = in_dim
        self.hparams = hparams
        self.network = MLP(in_dim, 1, self.hparams['num_hidden'], self.hparams['depth'],
                           self.hparams['spectral'])

    def disc_loss(self, z_src, z_trg, weights_src):
        disc_stats = OrderedDict()
        disc_extra_stats = OrderedDict()

        disc_out_src = self.network(z_src).view(-1)
        disc_out_trg = self.network(z_trg).view(-1)

        disc_out = torch.cat((disc_out_src, disc_out_trg), dim=0)
        disc_labels = torch.cat((
            torch.zeros_like(disc_out_src),
            torch.ones_like(disc_out_trg)
        ))
        disc_weights = torch.cat((weights_src, torch.ones_like(disc_out_trg)), dim=0)
        loss = F.binary_cross_entropy_with_logits(disc_out, disc_labels, weight=disc_weights)

        disc_stats['d_loss'] = loss.item()

        return loss, disc_stats, disc_extra_stats

    def alignment_loss(self, z_src, z_trg, weights_src):
        stats = OrderedDict()
        loss = -self.disc_loss(z_src, z_trg, weights_src)[0]

        return loss, stats

    def forward(self, x):
        return self.network(x)


# Wasserstein gradient penalty discriminator
class WGPDiscriminator(Discriminator):
    """
        extra_hparams: {
          grad_penalty_weight -- gradient penalty weight
        }
    """

    def __init__(self, in_dim, hparams):
        super(WGPDiscriminator, self).__init__(in_dim, hparams)

    def disc_loss(self, z_src, z_trg, update_history=False):
        disc_stats = OrderedDict()
        disc_extra_stats = OrderedDict()

        disc_out_src = self.network(z_src).view(-1)
        disc_out_trg = self.network(z_trg).view(-1)

        disc_w_loss = -torch.mean(disc_out_src) + torch.mean(disc_out_trg)

        # WGAN-GP gradient penalty
        alpha = torch.rand(z_src.size(0), device=z_src.device)[:, None]
        z_int = z_src * alpha + z_trg * (1.0 - alpha)
        disc_out_int = self.network(z_int).view(-1)
        grad = torch.autograd.grad(outputs=torch.sum(disc_out_int), inputs=z_int,
                                   only_inputs=True, create_graph=True, retain_graph=True)[0]
        grad_norm = torch.sqrt(torch.sum(torch.square(grad), dim=1) + 1e-12)
        disc_grad_loss = torch.mean(torch.square(grad_norm - 1.0))
        disc_grad_max_norm = torch.max(grad_norm).item()

        disc_stats['d_loss'] = disc_w_loss.item()
        disc_stats['d_grad_loss'] = disc_grad_loss.item()
        disc_stats['d_grad_max'] = disc_grad_max_norm

        disc_extra_stats['d_src_min'] = disc_out_src.min().item()
        disc_extra_stats['d_src_mean'] = disc_out_src.mean().item()
        disc_extra_stats['d_src_max'] = disc_out_src.max().item()

        disc_extra_stats['d_trg_min'] = disc_out_trg.min().item()
        disc_extra_stats['d_trg_mean'] = disc_out_trg.mean().item()
        disc_extra_stats['d_trg_max'] = disc_out_trg.max().item()

        return disc_w_loss, disc_grad_loss, disc_stats, disc_extra_stats

    def alignment_loss(self, z_src, z_trg, update_history=False):
        stats = OrderedDict()

        disc_out_src = self.network(z_src).view(-1)
        disc_out_trg = self.network(z_trg).view(-1)

        w_loss = torch.mean(disc_out_src) - torch.mean(disc_out_trg)

        return w_loss, stats
