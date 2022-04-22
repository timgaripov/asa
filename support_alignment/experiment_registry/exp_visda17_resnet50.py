"""Configs for VisDA17 experiments."""


def get_weighting_config_class_pareto(alpha, reverse, seed):
    return {
        'name': 'class_pareto',
        'kwargs': {
            'alpha': alpha,
            'reverse': reverse,
            'seed': seed
        },
    }


def get_dataset_config_visda17_pareto_target_imbalance(alpha, seed=None):
    return {
        'name': 'VisDA17',
        'val_fraction': 0.15,
        'mods': [],
        'source': {
            'index': 0,
            'weighting': {
                'name': 'class_uniform',
                'kwargs': dict(),
            },
            'subsample': True,
        },
        'target': {
            'index': 1,
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
                    'name': 'ResNet',
                    'hparams': {
                        'feature_dim': 256,
                        'pretrained': True,
                        'freeze_bn': False,
                        'resnet18': False,
                        'resnet_dropout': 0.0,
                        'fc_lr_factor': 1.0,
                        'fc_wd_factor': 1.0,
                    }
                },
                'classifier': {
                    'name': 'LogLossClassifier',
                    'hparams': {
                        'num_hidden': None,
                        'special_init': True,
                    }
                },
            },
            'discriminator': {
                'hparams': {
                    'num_hidden': 1024,
                    'depth': 3,
                    'spectral': False,
                    'history_size': 0,
                }
            },

            'ema_momentum': None,
            'fx_opt': {
                'name': 'SGD',
                'kwargs': {
                    'lr': 0.001,
                    'momentum': 0.9,
                    'weight_decay': 0.001,
                    'nesterov': True,
                }
            },
            'fx_lr_decay_start': 0,
            'fx_lr_decay_steps': 50000,
            'fx_lr_decay_factor': 0.05,

            'cls_opt': {
                'name': 'SGD',
                'kwargs': {
                    'lr': 0.01,
                    'momentum': 0.9,
                    'weight_decay': 0.001,
                    'nesterov': True,
                }
            },
            'cls_weight': 1.0,
            'cls_trg_weight': 0.0,

            'alignment_weight': None,
            'alignment_w_steps': 10000,

            'disc_opt': {
                'name': 'SGD',
                'kwargs': {
                    'lr': 0.005,
                    'momentum': 0.9,
                    'weight_decay': 0.001,
                    'nesterov': True,
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

    iwdan_extra_hparams = {'alignment_weight': 0.1, 'iw_update_period': 4000, 'importance_weighting': {'ma': 0.5}}
    algorithms.extend([
      ('iwdan', 'IWDAN', iwdan_extra_hparams, None),
      ('iwcdan', 'IWCDAN', iwdan_extra_hparams, None),
    ])

    algorithms.append(
      (f'sdann_4', 'SDANN', {'alignment_weight': 0.1}, {'beta': 4.0})
    )

    algorithms.extend([
        ('asa_abs', 'DANN_SUPP_ABS', {'alignment_weight': 0.1},
         {'history_size': 1000}),
        ('asa_sq', 'DANN_SUPP_SQ', {'alignment_weight': 0.1},
         {'history_size': 1000}),
    ])

    for imbalance_alpha in [0.0, 1.0, 1.5, 2.0]:
        for seed in range(1, 6):
            dataset_config = get_dataset_config_visda17_pareto_target_imbalance(imbalance_alpha, seed=seed)

            training_config = {
                'seed': seed,
                'num_steps': 50000,
                'batch_size': 36,
                'num_workers': 4,
                'eval_period': 2500,
                'log_period': 50,
                'eval_bn_update': True,

                'save_model': False,
                'save_period': 1,

                'disc_eval_period': 4,
            }

            for alg_nickname, algorithm_name, extra_hparams, extra_discriminator_hparams in algorithms:
                algorithm_config = get_algorithm_config(algorithm_name, extra_hparams, extra_discriminator_hparams)

                experiment_name = f'visda17/resnet50/seed_{seed}/s_alpha_{int(imbalance_alpha * 10):02d}/{alg_nickname}'
                experiment_config = {
                    'dataset': dataset_config,
                    'algorithm': algorithm_config,
                    'training': training_config,
                }
                registry.register(experiment_name, experiment_config)
