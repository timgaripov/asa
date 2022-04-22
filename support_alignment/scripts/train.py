"""
    Domain adaptation training.

    Config structure:
        dataset: {
            name -- dataset name (e.g. MNIST_USPS)
            val_fraction -- size of validation splits as a fraction of number of available examples
            mods -- list of mods to apply on the dataset
            source: {
                index -- index of the source environment
                weighting: {
                    name -- name of the weighting function
                    kwargs -- weighting function kwargs dict
                }
                subsample -- if True use subsampling instead of re-weighting
            }
            target: {
                index -- index of the source environment
                weighting: {
                    name -- name of the weighting function
                    kwargs -- weighting function kwargs dict
                }
                subsample -- if True use subsampling instead of re-weighting
            }
        }

        algorithm: {
            name -- algorithm name
            hparams -- algorithm hyperparameters dict
        }

        training: {
            seed -- seed used to generate dataset splits and model parameter initialization
            num_steps -- number of training steps

            batch_size -- batch size
            num_workers -- number of workers in data loader

            eval_period -- model evaluation period (eval each N steps)
            log_period -- logging period (log info each N steps)

            eval_bn_update -- if True update BN-statistics before each evaluation

            save_model -- if True save model checkpoints
            save_period -- model saving period (save each N evals)

            disc_eval_period -- discriminator evaluation period (evaluate each N evals)
            save_features_period -- feature saving period (save each N evals)
        }
"""

import argparse
import copy
from collections import OrderedDict
import json
import os
import pprint
import random
import shutil
import sys
import time
import tabulate

import torch

from support_alignment.experiment_registry import registry
from support_alignment.core import algorithms, utils
from support_alignment.core.data import sampling, datasets

import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str, required=True, help='name of experiment to run')
parser.add_argument('--experiment_name_override', type=str, default='',
                    help='by default experiment name is config name, this argument can be used to override default')

parser.add_argument('--experiment_root', type=str, required=True, help='folder to output results and checkpoints')
parser.add_argument('--force_overwrite', action='store_true', help='disables experiment folder overwrite prompt')
parser.add_argument('--data_root', type=str, required=True, help='path to datasets')

parser.add_argument('--cuda', action='store_true', help='enables cuda')

parser.add_argument('--wb_log', action='store_true', help='enables wandb logging')
parser.add_argument('--wb_project', default='support_alignment', help='wandb project name')
parser.add_argument('--wb_extra_tags', type=str, nargs='*', help='extra tags to be added to wandb run info')
parser.add_argument('--wb_save_model', action='store_true', help='enables checkpoint saving in wandb')


def train():
  args = parser.parse_args()

  experiment_config = registry.get_experiment_config(args.config_name)
  training_config = experiment_config['training']
  dataset_config = experiment_config['dataset']
  dataset_name = dataset_config['name']
  algorithm_config = experiment_config['algorithm']
  algorithm_name = algorithm_config['name']

  args.experiment_name = args.config_name
  if args.experiment_name_override:
      args.experiment_name = args.experiment_name_override

  seed = training_config['seed']
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.benchmark = True

  experiment_path = os.path.join(args.experiment_root, args.experiment_name)
  if os.path.exists(experiment_path):
      if not args.force_overwrite:
          print(f'Experiment directory {experiment_path} already exists, do you want to overwrite it? (y/n)')
          response = input()
          if response[:1] != 'y':
              print('Terminating')
              exit()
      shutil.rmtree(experiment_path)

  os.makedirs(experiment_path, exist_ok=True)
  with open(os.path.join(experiment_path, 'command_seed.txt'), 'w') as command_file:
      command_file.write(' '.join(sys.argv))
      command_file.write('\n')
      command_file.write(str(seed))
      command_file.write('\n')

  logger = utils.Logger(experiment_path, verbose=True)

  device = torch.device('cuda:0' if args.cuda else 'cpu')
  if torch.cuda.is_available() and not args.cuda:
      print('WARNING: a CUDA device is available, run with --cuda to use the CUDA device')

  logger.print('Args:')
  for k, v in sorted(vars(args).items()):
      logger.print(f'\t{k}: {v}')
  with open(os.path.join(experiment_path, 'args.json'), 'w') as args_file:
      args_file.write(json.dumps(vars(args), sort_keys=True))

  logger.print('Experiment config:')
  config_str = pprint.pformat(experiment_config)
  logger.print(config_str)

  if args.wb_log:
      # Initialize wandb logging
      wandb_config = {'experiment_name': args.experiment_name}
      wandb_config.update({f'arg/{k}': v for k, v in vars(args).items()})
      wandb_config.update({f'config/{k}': v for k, v in utils.flatten_config_dict(experiment_config)})

      tags = args.experiment_name.split('/')
      if args.wb_extra_tags:
          tags.extend(args.wb_extra_tags)
      wandb.init(project=args.wb_project, config=wandb_config, tags=tags,
                 settings=wandb.Settings(start_method="thread"))

  algorithm_class = getattr(algorithms, algorithm_name, None)
  if algorithm_class is None:
      raise RuntimeError(f'Unknown algorithm {algorithm_name}')

  dataset_class = getattr(datasets, dataset_name, None)
  if dataset_class is None:
      raise RuntimeError(f'Unknown dataset {dataset_name}')
  logger.print(f'Dataset: {dataset_name}')
  dataset = dataset_class(args.data_root)

  for mod_config in dataset_config['mods']:
      logger.print(f"Dataset mod: {mod_config['name']}({mod_config['args']})")
      dataset.apply_mod(mod_config['name'], mod_config['args'])

  def format_list(string_list):
      if len(string_list) > 16:
          tail = min(len(string_list) - 15, 5)
          string_list = string_list[:15] + ['...'] + string_list[-tail:]
      return ' '.join(string_list)

  def preprocess_splits(train_split, val_split, weighting_config=None, subsample=False, subsample_seed=None):
      class_probs = None

      if weighting_config is not None:
          logger.print(f"\tWeighting: {weighting_config['name']}({weighting_config['kwargs']})")
          weighting_fn = getattr(sampling.DatasetWeighting, weighting_config['name'])
          class_probs = weighting_fn(dataset.num_classes, **weighting_config['kwargs'])

          probs_fmt_list = [f'{p:0.2f}' for p in (class_probs * 100.0)]
          probs_str = format_list(probs_fmt_list)
          logger.print(f'\tClass probabilities (%): [{probs_str}]')
      else:
          logger.print('\tNo weighting')
      logger.print('')

      # Process train split
      train_class_distribution = None
      if subsample:
          if class_probs is None:
              raise ValueError('Dataset can not be subsampled: weights are not specified')
          logger.print('\tUsing subsampling for training')
          logger.print(f'\tTrain samples before subsampling: {len(train_split)}')
          subsample_indices, class_samples, original_class_samples = \
              sampling.dataset_class_subsampling(train_split, class_probs, seed=subsample_seed)
          train_split = datasets.SubsetDatasetWrapper(train_split, subsample_indices)
          train_weights = None

          ocsamples_fmt_list = [f'{n}' for n in original_class_samples]
          ocsamples_str = format_list(ocsamples_fmt_list)
          logger.print('\tOriginal Class samples: [{0}] (min = {1}, max = {2})'.format(ocsamples_str,
                                                                                       min(original_class_samples),
                                                                                       max(original_class_samples)))

          num_samples = len(train_split)
          csamples_fmt_list = [f'{n}' for n in class_samples]
          csamples_str = format_list(csamples_fmt_list)
          train_class_distribution = [n / num_samples for n in class_samples]
          probs_fmt_list = [f'{p * 100.0:0.2f}' for p in train_class_distribution]
          probs_str = format_list(probs_fmt_list)
          logger.print(f'\tTrain samples after subsampling: {num_samples}')
          logger.print('\tClass samples: [{0}] (min = {1}, max = {2})'.format(csamples_str,
                                                                              min(class_samples), max(class_samples)))
          logger.print(f'\tClass probabilities (%): [{probs_str}]')

      else:
          logger.print('\tUsing weighted sampling for training')
          train_weights, train_class_distribution, original_class_samples = sampling.dataset_class_weighting(train_split,
                                                                                                             class_probs)
          num_samples = len(train_split)
          csamples_fmt_list = [f'{n}' for n in original_class_samples]
          csamples_str = format_list(csamples_fmt_list)
          probs_fmt_list = [f'{n / num_samples * 100.0:0.2f}'for n in original_class_samples]
          probs_str = format_list(probs_fmt_list)
          logger.print(f'\tTrain samples: {num_samples}')
          logger.print('\tOriginal class samples: [{0}] (min = {1}, max = {2})'.format(csamples_str,
                                                                                       min(original_class_samples),
                                                                                       max(original_class_samples)))
          logger.print(f'\tOriginal class probabilities (%): [{probs_str}]')
      logger.print('')

      # Process validation split
      val_weights, _, original_class_samples = sampling.dataset_class_weighting(val_split, class_probs)

      num_samples = len(val_split)
      csamples_fmt_list = [f'{n}' for n in original_class_samples]
      csamples_str = format_list(csamples_fmt_list)
      probs_fmt_list = [f'{n / num_samples * 100.0:0.2f}' for n in original_class_samples]
      probs_str = format_list(probs_fmt_list)
      logger.print(f'\tValidation samples: {num_samples}')
      logger.print('\tOriginal class samples: [{0}] (min = {1}, max = {2})'.format(csamples_str,
                                                                                   min(original_class_samples),
                                                                                   max(original_class_samples)))
      logger.print(f'\tOriginal class probabilities (%): [{probs_str}]')

      return train_split, train_weights, train_class_distribution, val_split, val_weights

  val_fraction = dataset_config['val_fraction']
  # Source dataset splits
  source_index = dataset_config['source']['index']
  source_dataset, source_name = dataset[source_index], dataset.ENVIRONMENTS[source_index]
  source_train_split, source_val_split = datasets.split_dataset(source_dataset, split_fraction=val_fraction,
                                                                seed=utils.seed_hash(seed, 0))
  logger.print(f'Source: {source_name} [{source_index}]')

  source_train_split, source_train_weights, source_train_class_distribution, source_val_split, source_val_weights = \
      preprocess_splits(source_train_split, source_val_split, dataset_config['source']['weighting'],
                        dataset_config['source']['subsample'], subsample_seed=seed)

  # Source batchnorm update split
  source_bn_upd_split = source_train_split

  # Target dataset splits
  target_index = dataset_config['target']['index']
  target_dataset, target_name = dataset[target_index], dataset.ENVIRONMENTS[target_index]
  target_train_split, target_val_split = datasets.split_dataset(target_dataset, split_fraction=val_fraction,
                                                                seed=utils.seed_hash(seed, 1))

  logger.print(f'Target: {target_name} [{target_index}]')

  target_train_split, target_train_weights, target_train_class_distribution, target_val_split, target_val_weights = \
      preprocess_splits(target_train_split, target_val_split, dataset_config['target']['weighting'],
                        dataset_config['target']['subsample'], subsample_seed=seed)

  # Target batchnorm update split
  target_bn_upd_split = target_train_split

  # Apply transforms
  if hasattr(dataset, 'train_transform'):
      source_train_split = datasets.TransformDatasetWrapper(source_train_split, dataset.train_transform)
      target_train_split = datasets.TransformDatasetWrapper(target_train_split, dataset.train_transform)
  if hasattr(dataset, 'eval_transform'):
      source_val_split = datasets.TransformDatasetWrapper(source_val_split, dataset.eval_transform)
      target_val_split = datasets.TransformDatasetWrapper(target_val_split, dataset.eval_transform)

      source_bn_upd_split = datasets.TransformDatasetWrapper(source_bn_upd_split, dataset.eval_transform)
      target_bn_upd_split = datasets.TransformDatasetWrapper(target_bn_upd_split, dataset.eval_transform)

  # Create iterator and loaders
  batch_size = training_config['batch_size']
  num_workers = training_config['num_workers']
  train_iterator = sampling.DATrainIterator(source_train_split, source_train_weights,
                                            target_train_split, target_train_weights,
                                            batch_size, num_workers, device)

  source_val_loader = torch.utils.data.DataLoader(source_val_split,
                                                  shuffle=False,
                                                  drop_last=False,
                                                  batch_size=batch_size,
                                                  num_workers=num_workers)
  target_val_loader = torch.utils.data.DataLoader(target_val_split,
                                                  shuffle=False,
                                                  drop_last=False,
                                                  batch_size=batch_size,
                                                  num_workers=num_workers)

  # Apply weights
  if source_train_weights is None:
      source_bn_upd_loader = torch.utils.data.DataLoader(source_bn_upd_split,
                                                         shuffle=False,
                                                         drop_last=False,
                                                         batch_size=batch_size,
                                                         num_workers=num_workers)
  else:
      sampler = torch.utils.data.WeightedRandomSampler(source_train_weights,
                                                       replacement=True,
                                                       num_samples=source_train_weights.size(0))
      batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=batch_size, drop_last=False)
      source_bn_upd_loader = torch.utils.data.DataLoader(source_bn_upd_split,
                                                         num_workers=num_workers,
                                                         batch_sampler=batch_sampler)

  if target_train_weights is None:
      target_bn_upd_loader = torch.utils.data.DataLoader(target_bn_upd_split,
                                                         shuffle=False,
                                                         drop_last=False,
                                                         batch_size=batch_size,
                                                         num_workers=num_workers)
  else:
      sampler = torch.utils.data.WeightedRandomSampler(target_train_weights,
                                                       replacement=True,
                                                       num_samples=target_train_weights.size(0))
      batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=batch_size, drop_last=False)
      target_bn_upd_loader = torch.utils.data.DataLoader(target_bn_upd_split,
                                                         num_workers=num_workers,
                                                         batch_sampler=batch_sampler)

  data_params = getattr(dataset, 'data_params', dict())
  data_params.update({'source_class_distribution': source_train_class_distribution})

  eval_names = ['source_val', 'target_val']
  eval_loaders = [source_val_loader, target_val_loader]
  bn_upd_loaders = [source_bn_upd_loader, target_bn_upd_loader]
  eval_weights = [source_val_weights, target_val_weights]

  # Create and initialize algorithm
  algorithm = algorithm_class(dataset.input_shape, dataset.num_classes, algorithm_config['hparams'], data_params)
  logger.print(algorithm)
  algorithm.to(device)

  algorithm.train()

  def save_checkpoint(filename):
      if not training_config['save_model']:
          return

      save_dict = {
          'args': vars(args),
          'config': experiment_config,
          'model_input_shape': dataset.input_shape,
          'model_num_classes': dataset.num_classes,
          'model_state': algorithm.state_dict(),
          'optimizers_state': algorithm.optimizers_state(),
      }

      checkpoint_path = os.path.join(experiment_path, filename)
      torch.save(save_dict, checkpoint_path)
      if args.wb_log and args.wb_save_model:
          wandb.save(checkpoint_path, base_path=experiment_path)

  stats = OrderedDict()
  eval_results = OrderedDict()
  log_eval_results = False
  log_rows = 0

  alignment_eval_batches = (max(len(source_train_split), len(target_train_split)) + batch_size - 1) // batch_size

  num_steps = training_config['num_steps']
  eval_period = training_config['eval_period']
  log_period = training_config['log_period']
  save_features_period = training_config.get('save_features_period', None)

  for step in range(0, num_steps):
      step_start_time = time.time()

      # Perform training step
      alg_step_stats, alg_step_extra_stats = algorithm.update(train_iterator)

      # Evaluate algorithm
      if step == 0 or (step + 1) % eval_period == 0:
          eval_results = OrderedDict()
          eval_results['step'] = step + 1
          for eval_name, loader, bn_upd_loader, weights in zip(eval_names, eval_loaders, bn_upd_loaders, eval_weights):
              if training_config['eval_bn_update'] and hasattr(algorithm, 'update_bn'):
                  # Update batchnorm statistics
                  algorithm.update_bn(bn_upd_loader, device)

              # Evaluate model
              eval_stats, predictions, labels, eval_samples = utils.evaluate(algorithm, loader, weights, device)
              eval_results.update({'{0}_{1}'.format(eval_name, k): v for k, v in eval_stats.items()})

              if args.wb_log:
                  class_names = getattr(dataset, 'class_names', None)
                  if class_names is None:
                      nzeros = len(str(dataset.num_classes))
                      class_names = ['class_{0:0{nzeros}d}'.format(i, nzeros=nzeros) for i in range(dataset.num_classes)]

                  if dataset.num_classes <= 12:
                      confusion_matrix = wandb.plot.confusion_matrix(probs=None, y_true=labels, preds=predictions,
                                                                     class_names=class_names)

                      wandb.log({'eval/{}/conf_mat'.format(eval_name): confusion_matrix}, step=step+1)
                  wandb.log({'eval/{0}/{1}'.format(eval_name, k): v for k, v in eval_stats.items()}, step=step+1)

                  # Save eval image samples on the first step
                  if step == 0:
                      image_grids = utils.batch_to_grid(eval_samples, data_params=data_params)
                      for grid_name, np_image in image_grids:
                          wandb.log({'images/{0}/{1}'.format(eval_name, grid_name): [wandb.Image(np_image)]}, step=step+1)

          # Save evaluation results (append to results.jsnol)
          results_path = os.path.join(experiment_path, 'results.jsonl')
          with open(results_path, 'a') as f:
              f.write(json.dumps(eval_results))
              f.write('\n')

          if args.wb_log and step == 0:
              # Save train image samples
              x_src, y_src, x_trg = next(train_iterator)
              src_grids = utils.batch_to_grid(x_src, data_params=data_params)
              for name, np_image in src_grids:
                  wandb.log({'images/source_train/{}'.format(name): [wandb.Image(np_image)]}, step=step+1)
              trg_grids = utils.batch_to_grid(x_trg, data_params=data_params)
              for name, np_image in trg_grids:
                  wandb.log({'images/target_train/{}'.format(name): [wandb.Image(np_image)]}, step=step+1)
              if getattr(algorithm, 'vat_loss', None) is not None and hasattr(algorithm, 'network'):
                  adv_loss_src, x_adv_src = algorithm.vat_loss(algorithm.network, x_src)
                  adv_loss_trg, x_adv_trg = algorithm.vat_loss(algorithm.network, x_trg)

                  adv_src_grids = utils.batch_to_grid(x_adv_src, data_params=data_params)
                  adv_trg_grids = utils.batch_to_grid(x_adv_trg, data_params=data_params)
                  for name, np_image in adv_src_grids:
                      img = wandb.Image(np_image, caption='adv_loss: {:.2f}'.format(adv_loss_src.item()))
                      wandb.log({'images/source_adversarial/{}'.format(name): [img]}, step=step+1)
                  for name, np_image in adv_trg_grids:
                      img = wandb.Image(np_image, caption='adv_loss: {:.2f}'.format(adv_loss_trg.item()))
                      wandb.log({'images/target_adversarial/{}'.format(name): [img]}, step=step+1)

          if args.wb_log and hasattr(algorithm, 'network') and hasattr(algorithm, 'discriminator') and \
                  ((step + 1) // eval_period) % training_config['disc_eval_period'] == 0 and \
                  not isinstance(algorithm, algorithms.IWCDAN):
              # Save discriminator output statistics and histogram
              outputs, metrics = utils.evaluate_alignment(algorithm.network.feature_extractor,
                                                          [algorithm.discriminator], ['disc_alignment'],
                                                          train_iterator,
                                                          num_batches=min(alignment_eval_batches, 60000 // 32))
              wandb.log(metrics, step=step+1)

              src_output, trg_output = outputs[0]
              histplot = utils.alignment_histplot(src_output, trg_output,
                                                  title='Disc alignment (step: {})'.format(step+1))
              wandb.log({'disc_alignment/hist': wandb.Image(histplot)}, step=step+1)

          if hasattr(algorithm, 'network') and (save_features_period is not None) and \
                  (((step + 1) // eval_period) % save_features_period == 0):
              # Save features for analysis
              features_src_tr, labels_src_tr = utils.extract_features(algorithm.network.feature_extractor,
                                                                      source_bn_upd_loader,
                                                                      device=device)
              features_trg_tr, labels_trg_tr = utils.extract_features(algorithm.network.feature_extractor,
                                                                      target_bn_upd_loader,
                                                                      device=device)
              features_src_val, labels_src_val = utils.extract_features(algorithm.network.feature_extractor,
                                                                        source_val_loader,
                                                                        device=device)
              features_trg_val, labels_trg_val = utils.extract_features(algorithm.network.feature_extractor,
                                                                        target_val_loader,
                                                                        device=device)
              file_path = os.path.join(experiment_path, 'features_step_{0:06d}.pkl'.format(step + 1))
              torch.save({
                  'features_src_tr': features_src_tr,
                  'labels_src_tr': labels_src_tr,
                  'features_trg_tr': features_trg_tr,
                  'labels_trg_tr': labels_trg_tr,
                  'features_src_val': features_src_val,
                  'labels_src_val': labels_src_val,
                  'features_trg_val': features_trg_val,
                  'labels_trg_val': labels_trg_val
              }, file_path)
              if args.wb_log:
                  wandb.save(file_path, base_path=experiment_path)

              if args.wb_log and dataset.num_classes <= 3:
                  # Visualize features
                  title = 'Step {0}\n S val acc min: {1:.2f}% T val acc min: {2:.2f}'.format(
                      step + 1, eval_results['source_val_accuracy_class_min'],
                      eval_results['target_val_accuracy_class_min'])
                  feature_plot = utils.feature_plot(algorithm, features_src_tr, labels_src_tr,
                                                    features_trg_tr, labels_trg_tr,
                                                    num_classes=dataset.num_classes, title=title, device=device)
                  wandb.log({'feature_vis/plot': wandb.Image(feature_plot)}, step=step+1)

          if isinstance(algorithm, algorithms.IWBase):
              # Log importance weights
              importance_weights = algorithm.iw.importance_weights.detach().cpu().numpy()
              logger.print('Importance weights')
              logger.print([f'{val:.2f}' for val in importance_weights])
              logger.print('')
              if args.wb_log:
                  wandb.log({'iw/importance_weights': importance_weights}, step=step+1)
                  for i in range(min(importance_weights.shape[0], 15)):
                      wandb.log({f'iw/importance_weights_{i}': importance_weights[i]}, step=step+1)

          # Save checkpoint
          if ((step + 1) // eval_period) % training_config['save_period'] == 0:
              save_checkpoint('model_step_{0:06d}.pkl'.format(step + 1))
          log_eval_results = True

      stats['step'] = step + 1
      stats.update({key: val for key, val in alg_step_stats.items() if (key not in stats) or (val is not None)})
      if log_eval_results:
          stats['src_acc_w'] = eval_results['source_val_accuracy_weight']
          stats['trg_acc_w'] = eval_results['target_val_accuracy_weight']
          stats['src_acc_mn'] = eval_results['source_val_accuracy_class_min']
          stats['trg_acc_mn'] = eval_results['target_val_accuracy_class_min']
      else:
          for name in ['src_acc_w', 'trg_acc_w', 'src_acc_mn', 'trg_acc_mn']:
              stats[name] = None
      stats['time'] = time.time() - step_start_time

      if (step == 0) or ((step + 1) % log_period == 0) or (step < 200) or (step < 1000 and (step + 1) % 5 == 0):
          # Log training stats
          if args.wb_log:
              wb_stats = copy.deepcopy(stats)
              # Save extra stats in wandb
              wb_stats.update(alg_step_extra_stats)
              wandb.log({k: v for k, v in wb_stats.items() if v is not None}, step=step+1)
          columns = stats.keys()
          values = stats.values()
          table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
          if log_rows % 40 == 0:
              table = table.split('\n')
              table = '\n'.join([table[1]] + table)
          else:
              table = table.split('\n')[2]
          logger.print(table)
          log_rows += 1
          log_eval_results = False

  save_checkpoint('model.pkl')

  if hasattr(algorithm, 'network') and hasattr(algorithm, 'discriminator') and \
          not isinstance(algorithm, algorithms.IWCDAN):
      # Evaluate alignment in discriminator output space
      logger.print('Alignment evaluation.')

      names = ['alignment_eval_original']
      discriminator_list = [algorithm.discriminator]

      logger.print('Evaluating discriminators')
      outputs, metrics = utils.evaluate_alignment(algorithm.network.feature_extractor, discriminator_list, names,
                                                  train_iterator, num_batches=alignment_eval_batches)

      metrics_fname = os.path.join(experiment_path, 'alignment_eval_metrics.json')
      outputs_fname = os.path.join(experiment_path, 'alignment_eval_outputs.pt')

      with open(metrics_fname, 'wt') as metrics_file:
          metrics_file.write(json.dumps(metrics))

      torch.save({
          'disc_names': names,
          'disc_outputs': outputs
      }, outputs_fname)

      if args.wb_log:
          for key, val in metrics.items():
              wandb.run.summary[key] = val
          wandb.save(metrics_fname, base_path=experiment_path)
          wandb.save(outputs_fname, base_path=experiment_path)

          for name, out in zip(names, outputs):
              histplot = utils.alignment_histplot(*out, title=name)
              wandb.log({'{}/histplot'.format(name): wandb.Image(histplot)})


if __name__ == "__main__":
  train()
