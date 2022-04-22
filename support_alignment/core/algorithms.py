from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from support_alignment.core import discriminators, classifiers, vat, feature_extractors, utils, importance_weighting


def create_linear_decay_lr_schedule(decay_start, decay_steps, decay_factor):
  def schedule(step):
    return 1.0 + min(max(0.0, (step - decay_start) / decay_steps), 1.0) * (decay_factor - 1.0)
  return schedule


# Adapted from https://github.com/facebookresearch/DomainBed/blob/master/domainbed/algorithms.py
class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    def __init__(self, input_shape, num_classes, hparams, data_params):
        super(Algorithm, self).__init__()
        self.hparams = hparams
        self.data_params = data_params
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.optimizers = []

    def update(self, iterator):
        """
        Perform one update step, given an iterator which yields
        a labeled mini-batch from source environment
        and an unlabeled mini-batch from target environment
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def optimizers_state(self):
        return [optim.state_dict() for optim in self.optimizers]

    def load_optimizers_state(self, optimizers_state):
        for optim, optim_state in zip(self.optimizers, optimizers_state):
            optim.load_state_dict(optim_state)


class DANetwork(nn.Module):
    """
        config: {
            feature_extractor: {
                name -- feature extractor name
                hparams -- feature extractor hparams dict
            }
            classifier: {
                name -- classifier name
                hparams -- classifier hparams dict

            }
        }
    """
    def __init__(self, input_shape, num_classes, config):
        super(DANetwork, self).__init__()
        feature_extractor_config = config['feature_extractor']
        classifier_config = config['classifier']

        feature_extractor_name = feature_extractor_config['name']
        feature_extractor_fn = getattr(feature_extractors, feature_extractor_name, None)
        if feature_extractor_fn is None:
            raise ValueError(f'Unknown feature_extractor {feature_extractor_name}')
        self.feature_extractor = feature_extractor_fn(input_shape, hparams=feature_extractor_config['hparams'])

        classifier_name = classifier_config['name']
        classifier_fn = getattr(classifiers, classifier_name, None)
        if classifier_fn is None:
            raise ValueError(f'Unknown classifier {classifier_name}')
        self.classifier = classifier_fn(self.feature_extractor.n_outputs, num_classes,
                                        hparams=classifier_config['hparams'])

    def forward(self, x):
        return self.classifier(self.feature_extractor(x))


def process_param_groups(param_groups, optim_config):
    base_lr = optim_config['kwargs']['lr']
    base_wd = optim_config['kwargs'].get('weight_decay', 0.0)
    result_param_groups = []
    for param_group in param_groups:
        lr_factor = param_group.get('lr_factor', 1.0)
        wd_factor = param_group.get('wd_factor', 1.0)
        result_param_groups.append({
            'params': param_group['params'],
            'lr': base_lr * lr_factor,
            'weight_decay': base_wd * wd_factor,
        })
        print(len(result_param_groups[-1]['params']), result_param_groups[-1]['lr'],
              result_param_groups[-1]['weight_decay'])
    return result_param_groups


def get_optimizer(params, optim_config):
    optim_name = optim_config['name']
    optim_fn = getattr(torch.optim, optim_name, None)
    if optim_fn is None:
        raise ValueError(f'Unknown optimizer {optim_name}')
    return optim_fn(params, **optim_config['kwargs'])


# Adapted from https://github.com/facebookresearch/DomainBed/blob/master/domainbed/algorithms.py
class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)

    hparams: {
        da_network -- DANetwork config dict

        fx_opt {
            name -- feature extractor optimizer name
            kwargs -- feature extractor kwargs dict
        }

        cls_opt {
            name -- classifier optimizer name
            kwargs -- classifier kwargs dict
        }

        ema:
        ema_momentum -- momentum of the exponential weight averaging (EWA)
                        applied to feature_extractor + classifier (None => no EMA)

        fx_lr_decay_start -- start of feature extractor learning rate decay
        fx_lr_decay_steps -- length of feature extractor learning rate decay
        fx_lr_decay_factor -- feature extractor learning rate decay factor
    }
    """
    def __init__(self, input_shape, num_classes, hparams, data_params):
        super(ERM, self).__init__(input_shape, num_classes, hparams, data_params)

        self.network = DANetwork(input_shape, num_classes, self.hparams['da_network'])

        self.ema_network = None
        if self.hparams['ema_momentum'] is not None:
            ema_momentum = self.hparams['ema_momentum']

            def ema_avg_fn(averaged_model_parameter, model_parameter, num_averaged):
                return ema_momentum * averaged_model_parameter + (1.0 - ema_momentum) * model_parameter

            self.ema_network = torch.optim.swa_utils.AveragedModel(self.network, avg_fn=ema_avg_fn)
            for param in self.ema_network.parameters():
                param.requires_grad = False

        if hasattr(self.network.feature_extractor, 'param_groups'):
            fx_params = process_param_groups(self.network.feature_extractor.param_groups, self.hparams['fx_opt'])
        else:
            fx_params = [param for param in self.network.feature_extractor.parameters() if param.requires_grad]

        self.fx_opt = get_optimizer(fx_params, self.hparams['fx_opt'])
        self.optimizers.append(self.fx_opt)

        cls_params = [param for param in self.network.classifier.parameters() if param.requires_grad]
        self.cls_opt = get_optimizer(cls_params, self.hparams['cls_opt'])
        self.optimizers.append(self.cls_opt)

        self.fx_lr_decay_start = self.hparams['fx_lr_decay_start']
        self.fx_lr_decay_steps = self.hparams['fx_lr_decay_steps']
        self.fx_lr_decay_factor = self.hparams['fx_lr_decay_factor']

        lr_schedule = lambda step: 1.0
        if self.fx_lr_decay_start is not None:
            lr_schedule = create_linear_decay_lr_schedule(self.fx_lr_decay_start,
                                                          self.fx_lr_decay_steps,
                                                          self.fx_lr_decay_factor)

        self.fx_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.fx_opt, lr_schedule)
        self.cls_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.cls_opt, lr_schedule)

    def update(self, iterator):
        x_src, y_src, x_trg = next(iterator)
        z_src = self.network.feature_extractor(x_src)
        logits_src = self.network.classifier(z_src)

        loss = torch.nn.functional.cross_entropy(logits_src, y_src)

        self.fx_opt.zero_grad()
        self.cls_opt.zero_grad()
        loss.backward()
        self.fx_opt.step()
        self.fx_lr_scheduler.step()

        self.cls_opt.step()
        self.cls_lr_scheduler.step()

        if self.ema_network is not None:
            self.ema_network.update_parameters(self.network)
        stats = OrderedDict()
        extra_stats = OrderedDict()
        stats['lr'] = self.fx_lr_scheduler.get_last_lr()[0]
        stats['cls_lr'] = self.cls_lr_scheduler.get_last_lr()[0]
        stats['c_loss'] = loss.item()
        return stats, extra_stats

    def predict(self, x):
        network = self.network if self.ema_network is None else self.ema_network
        return network(x)

    def update_bn(self, loader, device):
        network = self.network if self.ema_network is None else self.ema_network
        utils.update_bn(loader, network, device)


def entropy(logits):
    return torch.mean(-torch.sum(torch.softmax(logits, dim=1) * torch.log_softmax(logits, dim=1), dim=1))


class AbstrtactDiscriminatorAlignment(Algorithm):
    """
        DiscriminatorAlignment base class.

        hparams: {
            da_network -- DANetwork config dict

            discriminator: {
                hparams -- discriminator hparams dict
            }

            fx_opt {
                name -- feature extractor optimizer name
                kwargs -- feature extractor kwargs dict
            }

            cls_opt {
                name -- classifier optimizer name
                kwargs -- classifier kwargs dict
            }


            ema:
            ema_momentum -- momentum of the exponential weight averaging (EWA)
                            applied to feature_extractor + classifier (None => no EMA)

            fx_lr_decay_start -- start of feature extractor learning rate decay
            fx_lr_decay_steps -- length of feature extractor learning rate decay
            fx_lr_decay_factor -- feature extractor learning rate decay factor

            cls_weight -- weight of the classifier labeled loss (labeled source)
                          in the feature extractor's objective

            cls_trg_weight -- weight of the classifier unlabeled loss in the feature extractor's objective

            alignment_weight -- weight of the alignment loss in the feature extractor's objective
            alignment_w_steps -- alignment weight annealing steps

            disc_opt {
                name -- discriminator optimizer name
                kwargs -- discriminator optimizer kwargs dict
            }
            disc_steps -- number of discriminator training steps per one feature extractor step

            l2_weight -- weight of l2-norm regularizer on z (for both source and target)

            vat:   [if vat=True]
            cls_vat_src_weight -- weight of the VAT loss on source domain
            cls_vat_trg_weight -- weight of the VAT loss on target domain
            vat_radius, vat_xi -- VAT loss parameters
        }

    """
    def __init__(self, input_shape, num_classes, hparams, data_params, discriminator_fn, use_vat=False):
        super(AbstrtactDiscriminatorAlignment, self).__init__(input_shape, num_classes, hparams, data_params)
        self.register_buffer('update_count', torch.tensor([0]))

        self.network = DANetwork(input_shape, num_classes, self.hparams['da_network'])

        self.ema_network = None
        if self.hparams['ema_momentum'] is not None:
            ema_momentum = self.hparams['ema_momentum']

            def ema_avg_fn(averaged_model_parameter, model_parameter, num_averaged):
                return ema_momentum * averaged_model_parameter + (1.0 - ema_momentum) * model_parameter

            self.ema_network = torch.optim.swa_utils.AveragedModel(self.network, avg_fn=ema_avg_fn)
            for param in self.ema_network.parameters():
                param.requires_grad = False

        if hasattr(self.network.feature_extractor, 'param_groups'):
            fx_params = process_param_groups(self.network.feature_extractor.param_groups, self.hparams['fx_opt'])
        else:
            fx_params = [param for param in self.network.feature_extractor.parameters() if param.requires_grad]

        self.fx_opt = get_optimizer(fx_params, self.hparams['fx_opt'])
        self.optimizers.append(self.fx_opt)

        cls_params = [param for param in self.network.classifier.parameters() if param.requires_grad]
        self.cls_opt = get_optimizer(cls_params, self.hparams['cls_opt'])
        self.optimizers.append(self.cls_opt)

        self.fx_lr_decay_start = self.hparams['fx_lr_decay_start']
        self.fx_lr_decay_steps = self.hparams['fx_lr_decay_steps']
        self.fx_lr_decay_factor = self.hparams['fx_lr_decay_factor']

        lr_schedule = lambda step: 1.0
        if self.fx_lr_decay_start is not None:
            lr_schedule = create_linear_decay_lr_schedule(self.fx_lr_decay_start,
                                                          self.fx_lr_decay_steps,
                                                          self.fx_lr_decay_factor)

        self.fx_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.fx_opt, lr_schedule)
        self.cls_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.cls_opt, lr_schedule)

        self.cls_weight = self.hparams['cls_weight']

        self.cls_trg_weight = self.hparams['cls_trg_weight']

        self.alignment_weight = self.hparams['alignment_weight']
        self.alignment_w_steps = self.hparams['alignment_w_steps']

        self.discriminator = discriminator_fn(self.network.feature_extractor.n_outputs,
                                              self.hparams['discriminator']['hparams'])

        self.disc_steps = self.hparams['disc_steps']
        self.disc_opt = get_optimizer(self.discriminator.parameters(), self.hparams['disc_opt'])
        self.optimizers.append(self.disc_opt)

        self.l2_weight = self.hparams['l2_weight']

        self.vat_loss = None
        if use_vat:
            self.cls_vat_src_weight = self.hparams['cls_vat_src_weight']
            self.cls_vat_trg_weight = self.hparams['cls_vat_trg_weight']
            self.vat_loss = vat.VATLoss(radius=self.hparams['vat_radius'], xi=self.hparams['vat_xi'])

    def update(self, iterator):
        batches = None

        disc_stats = OrderedDict()
        disc_extra_stats = OrderedDict()
        for i in range(self.disc_steps):
            disc_stats = OrderedDict()
            disc_extra_stats = OrderedDict()
            batches = next(iterator)
            x_src, y_src, x_trg = batches

            # Detach feature_extractor's outputs so that the gradients w.r.t. feature_extractor are not computed
            with torch.no_grad():
                z_src = self.network.feature_extractor(x_src).detach()
                z_trg = self.network.feature_extractor(x_trg).detach()

            # Set require grad for detached copies of z_src and z_trg for gradient penalty
            z_src.requires_grad_()
            z_trg.requires_grad_()

            disc_log_loss, disc_grad_loss, disc_stats, disc_extra_stats = \
                self.discriminator.disc_loss(z_src, z_trg, update_history=i != self.disc_steps - 1)

            disc_loss = disc_log_loss + self.discriminator.grad_penalty_weight * disc_grad_loss

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()

        x_src, y_src, x_trg = batches if batches is not None else next(iterator)

        z_src = self.network.feature_extractor(x_src)
        z_trg = self.network.feature_extractor(x_trg)
        z_all = torch.cat((z_src, z_trg), dim=0)

        alignment_weight = self.alignment_weight
        if self.alignment_w_steps is not None:
            t = min(self.update_count.item() / self.alignment_w_steps, 1.0)
            alignment_weight = t * self.alignment_weight

        alignment_loss, alignment_stats = \
          self.discriminator.alignment_loss(z_src, z_trg, update_history=True)

        logits_src = self.network.classifier(z_src)
        logits_trg = self.network.classifier(z_trg)

        cls_loss_src = torch.nn.functional.cross_entropy(logits_src, y_src)

        cls_loss_trg = entropy(logits_trg)

        cls_w_norm = self.network.classifier.w_norm()

        l2_loss = torch.mean(torch.sum(torch.square(z_all), dim=1))

        fx_loss = self.cls_weight * cls_loss_src \
            + self.cls_trg_weight * cls_loss_trg \
            + alignment_weight * alignment_loss \
            + self.l2_weight * l2_loss

        cls_vat_src = None
        cls_vat_trg = None
        if self.vat_loss is not None:
            cls_vat_src, _ = self.vat_loss(self.network, x_src)
            cls_vat_trg, _ = self.vat_loss(self.network, x_trg)

            fx_loss += self.cls_vat_src_weight * cls_vat_src
            fx_loss += self.cls_vat_trg_weight * cls_vat_trg

        self.fx_opt.zero_grad()
        self.cls_opt.zero_grad()

        fx_loss.backward()
        self.fx_opt.step()
        self.fx_lr_scheduler.step()
        self.cls_opt.step()
        self.cls_lr_scheduler.step()

        if self.ema_network is not None:
            self.ema_network.update_parameters(self.network)

        stats = OrderedDict()
        extra_stats = OrderedDict()
        stats['lr'] = self.fx_lr_scheduler.get_last_lr()[0]
        stats['cls_lr'] = self.cls_lr_scheduler.get_last_lr()[0]
        stats['fx_loss'] = fx_loss.item()
        stats['z_dev'] = torch.sqrt(l2_loss).item()
        stats['c_loss'] = cls_loss_src.item()
        stats['c_trg'] = cls_loss_trg.item()
        stats['c_w'] = cls_w_norm
        if cls_vat_src is not None:
            stats['c_vat_src'] = cls_vat_src.item()
        if cls_vat_trg is not None:
            stats['c_vat_trg'] = cls_vat_trg.item()

        stats.update(disc_stats)
        stats['a_loss'] = alignment_loss.item()
        stats['a_w'] = alignment_weight

        extra_stats['z_max'] = torch.max(torch.abs(z_all)).item()
        extra_stats.update(disc_extra_stats)
        extra_stats.update(alignment_stats)

        self.update_count += 1
        return stats, extra_stats

    def predict(self, x):
        network = self.network if self.ema_network is None else self.ema_network
        return network(x)

    def update_bn(self, loader, device):
        network = self.network if self.ema_network is None else self.ema_network
        utils.update_bn(loader, network, device)


class DANN(AbstrtactDiscriminatorAlignment):
    def __init__(self, input_shape, num_classes, hparams, data_params):
        super(DANN, self).__init__(input_shape, num_classes, hparams, data_params,
                                   discriminator_fn=discriminators.LogLossZSDiscriminator)


class DANN_NS(AbstrtactDiscriminatorAlignment):
    def __init__(self, input_shape, num_classes, hparams, data_params):
        super(DANN_NS, self).__init__(input_shape, num_classes, hparams, data_params,
                                      discriminator_fn=discriminators.LogLossNSDiscriminator)


class VADA(AbstrtactDiscriminatorAlignment):
    def __init__(self, input_shape, num_classes, hparams, data_params):
        super(VADA, self).__init__(input_shape, num_classes, hparams, data_params,
                                   discriminator_fn=discriminators.LogLossNSDiscriminator, use_vat=True)


class DANN_OTD_SQ(AbstrtactDiscriminatorAlignment):
    def __init__(self, input_shape, num_classes, hparams, data_params):
        super(DANN_OTD_SQ, self).__init__(input_shape, num_classes, hparams, data_params,
                                          discriminator_fn=discriminators.OptimalTransportLossSqDiscriminator)


class DANN_SUPP_SQ(AbstrtactDiscriminatorAlignment):
    def __init__(self, input_shape, num_classes, hparams, data_params):
        super(DANN_SUPP_SQ, self).__init__(input_shape, num_classes, hparams, data_params,
                                           discriminator_fn=discriminators.SupportLossSqDiscriminator)


class DANN_SUPP_SQ_VAT(AbstrtactDiscriminatorAlignment):
    def __init__(self, input_shape, num_classes, hparams, data_params):
        super(DANN_SUPP_SQ_VAT, self).__init__(input_shape, num_classes, hparams, data_params,
                                               discriminator_fn=discriminators.SupportLossSqDiscriminator, use_vat=True)


class DANN_SUPP_ABS(AbstrtactDiscriminatorAlignment):
    def __init__(self, input_shape, num_classes, hparams, data_params):
        super(DANN_SUPP_ABS, self).__init__(input_shape, num_classes, hparams, data_params,
                                            discriminator_fn=discriminators.SupportLossAbsDiscriminator)


class DANN_SUPP_ABS_VAT(AbstrtactDiscriminatorAlignment):
    def __init__(self, input_shape, num_classes, hparams, data_params):
        super(DANN_SUPP_ABS_VAT, self).__init__(input_shape, num_classes, hparams, data_params,
                                                discriminator_fn=discriminators.SupportLossAbsDiscriminator,
                                                use_vat=True)


class SDANN(AbstrtactDiscriminatorAlignment):
    def __init__(self, input_shape, num_classes, hparams, data_params):
        super(SDANN, self).__init__(input_shape, num_classes, hparams, data_params,
                                    discriminator_fn=discriminators.SBetaDiscriminator)


class WGPDANN(AbstrtactDiscriminatorAlignment):
    def __init__(self, input_shape, num_classes, hparams, data_params):
        super(WGPDANN, self).__init__(input_shape, num_classes, hparams, data_params,
                                      discriminator_fn=discriminators.WGPDiscriminator)


class IWBase(Algorithm):
    """
        Implements IWDAN/IWCDAN proposed in
        Domain Adaptation with Conditional Distribution Matching and Generalized Label Shift
        https://arxiv.org/abs/2003.04475

        Notes:
            * does not use unlabeled classifier loss (e.g. entropy)

        hparams: {
            da_network -- DANetwork config dict classifier.name must be IWClassifier

            discriminator: {
                hparams -- discriminator hparams dict
            }

            fx_opt {
                name -- feature extractor optimizer name
                kwargs -- feature extractor kwargs dict
            }

            cls_opt {
                name -- classifier optimizer name
                kwargs -- classifier kwargs dict
            }

            fx_lr_decay_start -- start of feature extractor learning rate decay
            fx_lr_decay_steps -- length of feature extractor learning rate decay
            fx_lr_decay_factor -- feature extractor learning rate decay factor

            cls_weight -- weight of the classifier loss (labeled source)
                          in the feature extractor's objective

            importance_weighting -- importance_weighting hparams dict
            iw_update_period -- importance weight update period (in steps)

            alignment_weight -- weight of the alignment loss in the feature extractor's objective
            alignment_w_steps -- alignment weight annealing steps

            disc_opt {
                name -- discriminator optimizer name
                kwargs -- discriminator optimizer kwargs dict
            }
            disc_steps -- number of discriminator training steps per one feature extractor step
        }

    """
    def __init__(self, input_shape, num_classes, hparams, data_params, conditional=False):
        super(IWBase, self).__init__(input_shape, num_classes, hparams, data_params)
        self.register_buffer('update_count', torch.tensor([0]))
        self.conditional = conditional

        self.network = DANetwork(input_shape, num_classes, self.hparams['da_network'])

        if hasattr(self.network.feature_extractor, 'param_groups'):
            fx_params = process_param_groups(self.network.feature_extractor.param_groups, self.hparams['fx_opt'])
        else:
            fx_params = [param for param in self.network.feature_extractor.parameters() if param.requires_grad]
        self.fx_opt = get_optimizer(fx_params, self.hparams['fx_opt'])
        self.optimizers.append(self.fx_opt)

        cls_params = [param for param in self.network.classifier.parameters() if param.requires_grad]
        self.cls_opt = get_optimizer(cls_params, self.hparams['cls_opt'])
        self.optimizers.append(self.cls_opt)

        self.fx_lr_decay_start = self.hparams['fx_lr_decay_start']
        self.fx_lr_decay_steps = self.hparams['fx_lr_decay_steps']
        self.fx_lr_decay_factor = self.hparams['fx_lr_decay_factor']

        lr_schedule = lambda step: 1.0
        if self.fx_lr_decay_start is not None:
            lr_schedule = create_linear_decay_lr_schedule(self.fx_lr_decay_start,
                                                          self.fx_lr_decay_steps,
                                                          self.fx_lr_decay_factor)

        self.fx_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.fx_opt, lr_schedule)
        self.cls_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.cls_opt, lr_schedule)

        self.cls_weight = self.hparams['cls_weight']

        self.alignment_weight = self.hparams['alignment_weight']
        self.alignment_w_steps = self.hparams['alignment_w_steps']

        disc_in_dim = self.network.feature_extractor.n_outputs
        if conditional:
            disc_in_dim *= self.num_classes

        self.discriminator = discriminators.IWDiscriminator(disc_in_dim, self.hparams['discriminator']['hparams'])

        self.disc_steps = self.hparams['disc_steps']
        self.disc_opt = get_optimizer(self.discriminator.parameters(), self.hparams['disc_opt'])
        self.optimizers.append(self.disc_opt)

        self.iw = importance_weighting.ImportanceWeighting(num_classes, self.hparams['importance_weighting'])
        self.iw_update_period = self.hparams['iw_update_period']
        self.register_buffer('source_class_distribution', torch.tensor(data_params['source_class_distribution']))
        self.register_buffer('source_class_distribution_inv', 1.0 / self.source_class_distribution)

    def update(self, iterator):
        batches = None
        disc_stats = OrderedDict()
        disc_extra_stats = OrderedDict()
        for i in range(self.disc_steps):
            disc_stats = OrderedDict()
            disc_extra_stats = OrderedDict()
            batches = next(iterator)
            x_src, y_src, x_trg = batches
            weights_src = self.iw.get_sample_weights(y_src)

            # Detach feature_extractor's outputs so that the gradients w.r.t. feature_extractor are not computed
            with torch.no_grad():
                z_src = self.network.feature_extractor(x_src).detach()
                z_trg = self.network.feature_extractor(x_trg).detach()
                if self.conditional:
                    logits_src = self.network.classifier(z_src).detach()
                    logits_trg = self.network.classifier(z_trg).detach()
                    softmax_src = F.softmax(logits_src, dim=-1)
                    softmax_trg = F.softmax(logits_trg, dim=-1)
                    disc_in_src = torch.bmm(softmax_src.unsqueeze(2), z_src.unsqueeze(1)).view(
                      -1, softmax_src.size(1) * z_src.size(1))
                    disc_in_trg = torch.bmm(softmax_trg.unsqueeze(2), z_trg.unsqueeze(1)).view(
                      -1, softmax_trg.size(1) * z_trg.size(1))
                else:
                    disc_in_src = z_src
                    disc_in_trg = z_trg

            disc_loss, disc_stats, disc_extra_stats = \
                self.discriminator.disc_loss(disc_in_src, disc_in_trg, weights_src)

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()

        x_src, y_src, x_trg = batches if batches is not None else next(iterator)
        weights_src = self.iw.get_sample_weights(y_src)

        z_src = self.network.feature_extractor(x_src)
        z_trg = self.network.feature_extractor(x_trg)
        z_all = torch.cat((z_src, z_trg), dim=0)

        logits_src = self.network.classifier(z_src)
        logits_trg = self.network.classifier(z_trg)
        self.iw.update_stats(logits_src, y_src, logits_trg)

        # Alignment loss
        if self.conditional:
            softmax_src = F.softmax(logits_src, dim=-1)
            softmax_trg = F.softmax(logits_trg, dim=-1)
            disc_in_src = torch.bmm(softmax_src.unsqueeze(2), z_src.unsqueeze(1)).view(-1, softmax_src.size(
                1) * z_src.size(1))
            disc_in_trg = torch.bmm(softmax_trg.unsqueeze(2), z_trg.unsqueeze(1)).view(-1, softmax_trg.size(
                1) * z_trg.size(1))
        else:
            disc_in_src = z_src
            disc_in_trg = z_trg

        alignment_weight = self.alignment_weight
        if self.alignment_w_steps is not None:
            t = min(self.update_count.item() / self.alignment_w_steps, 1.0)
            alignment_weight = t * self.alignment_weight

        alignment_loss, alignment_stats = self.discriminator.alignment_loss(disc_in_src, disc_in_trg, weights_src)

        # Classifier loss
        class_weights = self.source_class_distribution_inv
        cls_loss_src = torch.mean(
            F.cross_entropy(logits_src, y_src, weight=class_weights, reduction='none') * weights_src) / self.num_classes

        cls_w_norm = self.network.classifier.w_norm()

        fx_loss = self.cls_weight * cls_loss_src \
            + alignment_weight * alignment_loss

        self.fx_opt.zero_grad()
        self.cls_opt.zero_grad()

        fx_loss.backward()
        self.fx_opt.step()
        self.fx_lr_scheduler.step()
        self.cls_opt.step()
        self.cls_lr_scheduler.step()

        if self.update_count % self.iw_update_period == self.iw_update_period - 1:
            self.iw.update_weights(self.source_class_distribution)

        stats = OrderedDict()
        extra_stats = OrderedDict()

        stats['lr'] = self.fx_lr_scheduler.get_last_lr()[0]
        stats['cls_lr'] = self.cls_lr_scheduler.get_last_lr()[0]
        stats['fx_loss'] = fx_loss.item()
        stats['z_dev'] = torch.sqrt(torch.mean(torch.sum(torch.square(z_all), dim=1))).item()

        stats['c_loss'] = cls_loss_src.item()
        stats['c_w'] = cls_w_norm
        stats.update(disc_stats)
        stats['a_loss'] = alignment_loss.item()
        stats['a_w'] = alignment_weight

        extra_stats['z_max'] = torch.max(torch.abs(z_all)).item()
        extra_stats.update(disc_extra_stats)
        extra_stats.update(alignment_stats)

        self.update_count += 1
        return stats, extra_stats

    def predict(self, x):
        return self.network(x)

    def update_bn(self, loader, device):
        utils.update_bn(loader, self.network, device)


class IWDAN(IWBase):
    def __init__(self, input_shape, num_classes, hparams, data_params):
        super(IWDAN, self).__init__(input_shape, num_classes, hparams, data_params, conditional=False)


class IWCDAN(IWBase):
    def __init__(self, input_shape, num_classes, hparams, data_params):
        super(IWCDAN, self).__init__(input_shape, num_classes, hparams, data_params, conditional=True)
