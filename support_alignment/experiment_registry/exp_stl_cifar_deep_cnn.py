"""Configs for STL->CIFAR experiments."""


def get_weighting_config_class_pareto(alpha, reverse, seed):
    return {
        'name': 'class_pareto',
        'kwargs': {
            'alpha': alpha,
            'reverse': reverse,
            'seed': seed
        },
    }


def get_dataset_config_cifar_stl_pareto_target_imbalance(alpha, seed=None):
    return {
        'name': 'CIFAR_STL',
        'val_fraction': 1.0 / 6.0,
        'mods': [],
        'source': {
            'index': 1,
            'weighting': {
                'name': 'class_uniform',
                'kwargs': dict(),
            },
            'subsample': True,
        },
        'target': {
            'index': 0,
            'weighting': get_weighting_config_class_pareto(alpha, True, seed=seed),
            'subsample': True,
        },
    }


def get_algorithm_config(algorithm, extra_hparams=None, extra_discriminator_hparams=None):
    # Common configs of all algorithms
    config = {
        'name': algorithm,
        'hparams': {
            'da_network': {
                'feature_extractor': {
                    'name': 'DeepCNN',
                    'hparams': {
                        'num_features': 192,
                        'gaussian_noise': 1.0,
                    }
                },
                'classifier': {
                    'name': 'LogLossClassifier',
                    'hparams': {
                        'num_hidden': None,
                    }
                },
            },
            'discriminator': {
                'hparams': {
                    'num_hidden': 512,
                    'depth': 3,
                    'spectral': False,
                    'history_size': 0,
                }
            },

            'ema_momentum': 0.998,
            'fx_opt': {
                'name': 'Adam',
                'kwargs': {
                    'lr': 1e-3,
                    'weight_decay': 0.0,
                    'amsgrad': False,
                    'betas': (0.5, 0.999),
                }
            },
            'fx_lr_decay_start': None,
            'fx_lr_decay_steps': None,
            'fx_lr_decay_factor': None,

            'cls_opt': {
                'name': 'Adam',
                'kwargs': {
                    'lr': 1e-3,
                    'weight_decay': 0.0,
                    'amsgrad': False,
                    'betas': (0.5, 0.999),
                }
            },
            'cls_weight': 1.0,
            'cls_trg_weight': 0.1,

            'alignment_weight': None,
            'alignment_w_steps': None,

            'disc_opt': {
                'name': 'Adam',
                'kwargs': {
                    'lr': 1e-3,
                    'weight_decay': 0.0,
                    'amsgrad': False,
                    'betas': (0.5, 0.999),
                }
            },
            'disc_steps': 1,

            'l2_weight': 0.0,
        }
    }

    if extra_hparams is not None:
        config['hparams'].update(extra_hparams)

    if extra_discriminator_hparams is not None:
        config['hparams']['discriminator']['hparams'].update(extra_discriminator_hparams)

    return config


def register_experiments(registry):
    # Algorithm configs format:
    # nickname, algorithm_name, extra_hparams, extra_discriminator_hparams
    algorithms = [
      ('source_only', 'ERM', None, None),
      ('dann_zero', 'DANN_NS', {'alignment_weight': 0.0}, None),
      ('dann', 'DANN_NS', {'alignment_weight': 0.1}, None),
    ]

    iwdan_extra_hparams = {'alignment_weight': 0.1, 'iw_update_period': 5000,
                           'importance_weighting': {'ma': 0.5}}
    algorithms.extend([
      ('iwdan', 'IWDAN', iwdan_extra_hparams, None),
      ('iwcdan', 'IWCDAN', iwdan_extra_hparams, None),
    ])

    algorithms.append(
      (f'sdann_4', 'SDANN', {'alignment_weight': 0.1}, {'beta': 4.0})
    )

    vada_extra_hparams = {
      'alignment_weight': 0.1,
      'cls_vat_src_weight': 0.0,
      'cls_vat_trg_weight': 0.1,
      'vat_xi': 1e-6,
      'vat_radius': 3.5,
    }
    algorithms.append(('vada', 'VADA', vada_extra_hparams, None))

    algorithms.extend([
      ('asa_abs', 'DANN_SUPP_ABS', {'alignment_weight': 0.1}, {'history_size': 1000}),
      ('support_sq', 'DANN_SUPP_SQ', {'alignment_weight': 0.1}, {'history_size': 1000}),
    ])

    # Target conditional entropy loss ablation

    algorithms.extend([
      ('dann_zero_no_trg_ent', 'DANN_NS', {'alignment_weight': 0.0, 'cls_trg_weight': 0.0}, None),
      ('dann_no_trg_ent', 'DANN_NS', {'alignment_weight': 0.1, 'cls_trg_weight': 0.0}, None),
      ('asa_abs_no_trg_ent', 'DANN_SUPP_ABS', {'alignment_weight': 0.1, 'cls_trg_weight': 0.0},
       {'history_size': 1000}),
      ('asa_sq_no_trg_ent', 'DANN_SUPP_SQ', {'alignment_weight': 0.1, 'cls_trg_weight': 0.0},
       {'history_size': 1000}),
    ])

    iwdan_no_trg_ent_extra_hparams = {'alignment_weight': 0.1, 'cls_trg_weight': 0.0, 'iw_update_period': 5000,
                                      'importance_weighting': {'ma': 0.5}}
    algorithms.extend([
      ('iwdan_no_trg_ent', 'IWDAN', iwdan_no_trg_ent_extra_hparams, None),
      ('iwcdan_no_trg_ent', 'IWCDAN', iwdan_no_trg_ent_extra_hparams, None),
    ])

    algorithms.extend([
      (f'sdann_{beta}_no_trg_ent', 'SDANN', {'alignment_weight': 0.1, 'cls_trg_weight': 0.0},
       {'beta': float(beta)}) for beta in [2, 4, 16]
    ])

    # Optimal transport-based baselines

    algorithms.extend([
      ('dann_ot', 'DANN_OTD_SQ', {'alignment_weight': 0.1}, None),
      ('wgp_dann', 'WGPDANN', {'alignment_weight': 0.1, 'disc_steps': 5}, {'grad_penalty_weight': 10.0}),
    ])

    # Alignment weight ablation

    algorithms.extend([
      ('dann_aw_001', 'DANN_NS', {'alignment_weight': 0.01}, None),
      ('dann_aw_1', 'DANN_NS', {'alignment_weight': 1.0}, None),
    ])

    vada_001_extra_hparams = {
      'alignment_weight': 0.01,
      'cls_vat_src_weight': 0.0,
      'cls_vat_trg_weight': 0.1,
      'vat_xi': 1e-6,
      'vat_radius': 3.5,
    }
    algorithms.append(('vada_001', 'VADA', vada_001_extra_hparams, None))

    asa_sq_vat_trg_aw_1_hparams = {
      'alignment_weight': 1.0,
      'cls_vat_src_weight': 0.0,
      'cls_vat_trg_weight': 0.1,
      'vat_xi': 1e-6,
      'vat_radius': 3.5,
    }
    algorithms.append(('asa_sq_vat_trg_aw_1', 'DANN_SUPP_SQ_VAT',
                       asa_sq_vat_trg_aw_1_hparams, {'history_size': 1000}))

    for imbalance_alpha in [0.0, 1.0, 1.5, 2.0]:
        for seed in range(1, 6):
            stl_cifar_config = get_dataset_config_cifar_stl_pareto_target_imbalance(
              imbalance_alpha, seed=seed)

            training_config = {
                'seed': seed,
                'num_steps': 40000,
                'batch_size': 64,
                'num_workers': 4,
                'eval_period': 800,
                'log_period': 50,
                'eval_bn_update': True,

                'save_model': False,
                'save_period': 1,

                'disc_eval_period': 15,
            }

            for alg_nickname, algorithm_name, extra_hparams, extra_discriminator_hparms in algorithms:
                algorithm_config = get_algorithm_config(algorithm_name, extra_hparams, extra_discriminator_hparms)

                experiment_name = (f'stl_cifar/deep_cnn/seed_{seed}/'
                                   f's_alpha_{int(imbalance_alpha * 10):02d}/{alg_nickname}')
                experiment_config = {
                    'dataset': stl_cifar_config,
                    'algorithm': algorithm_config,
                    'training': training_config,
                }
                registry.register(experiment_name, experiment_config)
