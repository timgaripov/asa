"""Configs for USPS->MNIST experiments with 2D feature representations."""


def get_weighting_config_class_pareto(alpha, reverse, seed):
    return {
        'name': 'class_pareto',
        'kwargs': {
            'alpha': alpha,
            'reverse': reverse,
            'seed': seed
        },
    }


def get_dataset_config_mnist_usps_pareto_target_imbalance(alpha, class_subset=None, seed=None):
    mods = []
    if class_subset is not None:
        mods = [{
            'name': 'class_subset',
            'args': [class_subset],
        }]

    return {
        'name': 'MNIST_USPS',
        'val_fraction': 0.15,
        'mods': mods,
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


def get_algorithm_config(algorithm, num_steps, feature_dim=2, extra_hparams=None, extra_discriminator_hparams=None):
    # Common configs of all algorithms
    config = {
        'name': algorithm,
        'hparams': {
            'da_network': {
                'feature_extractor': {
                    'name': 'LeNetNoReLU',
                    'hparams': {
                        'feature_dim': feature_dim,
                        'output_dropout_rate': 0.0,
                        'conv_dropout_rate': 0.0,
                    },
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

            'ema_momentum': None,
            'fx_opt': {
                'name': 'SGD',
                'kwargs': {
                    'lr': 0.02,
                    'momentum': 0.9,
                    'weight_decay': 0.0005,
                }
            },
            'fx_lr_decay_start': min(30000, int(num_steps * 0.5)),
            'fx_lr_decay_steps': min(30000, int(num_steps * 0.5)),
            'fx_lr_decay_factor': 0.001,

            'cls_opt': {
                'name': 'SGD',
                'kwargs': {
                    'lr': 0.02,
                    'momentum': 0.9,
                    'weight_decay': 0.0005,
                }
            },
            'cls_weight': 1.0,
            'cls_trg_weight': 0.0,

            'alignment_weight': None,
            'alignment_w_steps': min(10000, 0.16 * num_steps),

            'disc_opt': {
                'name': 'SGD',
                'kwargs': {
                    'lr': 0.02,
                    'weight_decay': 0.0005,
                    'momentum': 0.9,
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
        ('dann', 'DANN_NS', {'alignment_weight': 1.0}, None),
    ]

    iwdan_extra_hparams = {'alignment_weight': 1.0, 'iw_update_period': 1000, 'importance_weighting': {'ma': 0.5}}
    algorithms.extend([
        ('iwdan', 'IWDAN', iwdan_extra_hparams, None),
        ('iwcdan', 'IWCDAN', iwdan_extra_hparams, None),
    ])

    algorithms.append(
      (f'sdann_4', 'SDANN', {'alignment_weight': 1.0}, {'beta': 4.0})
    )

    vada_extra_hparams = {
        'alignment_weight': 1.0,
        'cls_vat_src_weight': 1.0,
        'cls_vat_trg_weight': 0.1,
        'cls_trg_weight': 0.1,
        'vat_xi': 10.0,
        'vat_radius': 3.5,
    }
    algorithms.extend([
        ('vada', 'VADA', vada_extra_hparams, None),
    ])

    algorithms.extend([
        ('asa_sq', 'DANN_SUPP_SQ', {'alignment_weight': 1.0},
         {'history_size': 1000}),
        ('asa_abs', 'DANN_SUPP_ABS', {'alignment_weight': 1.0},
         {'history_size': 1000}),
        ('asa_abs_h0', 'DANN_SUPP_ABS', {'alignment_weight': 1.0},
         {'history_size': 0}),
        ('asa_abs_h100', 'DANN_SUPP_ABS', {'alignment_weight': 1.0},
         {'history_size': 100}),
        ('asa_abs_h500', 'DANN_SUPP_ABS', {'alignment_weight': 1.0},
         {'history_size': 500}),
        ('asa_abs_h2000', 'DANN_SUPP_ABS', {'alignment_weight': 1.0},
         {'history_size': 2000}),
        ('asa_abs_h5000', 'DANN_SUPP_ABS', {'alignment_weight': 1.0},
         {'history_size': 5000}),
    ])

    feature_dim = 2
    class_subset = [3, 5, 9]
    step_factor = 30

    for imbalance_alpha in [0.0, 1.0, 1.5, 2.0]:
        for seed in range(1, 6):
            usps_mnist_config = get_dataset_config_mnist_usps_pareto_target_imbalance(
              imbalance_alpha, class_subset=class_subset, seed=seed)

            training_config = {
                'seed': seed,
                'num_steps': 1000 * step_factor,
                'batch_size': 64,
                'num_workers': 4,
                'eval_period': 50 * step_factor,
                'log_period': 1 * step_factor,
                'eval_bn_update': False,

                'save_model': False,
                'save_period': 1,

                'disc_eval_period': 4,
                'save_features_period': 4,
            }

            for alg_nickname, algorithm_name, extra_hparams, extra_discriminator_hparams in algorithms:
                algorithm_config = get_algorithm_config(algorithm_name, training_config['num_steps'],
                                                        feature_dim, extra_hparams,
                                                        extra_discriminator_hparams)

                experiment_name = (f'usps_mnist_3c/lenet_2d/seed_{seed}/'
                                   f's_alpha_{int(imbalance_alpha * 10):02d}/{alg_nickname}')
                experiment_config = {
                    'dataset': usps_mnist_config,
                    'algorithm': algorithm_config,
                    'training': training_config,
                }
                registry.register(experiment_name, experiment_config)
