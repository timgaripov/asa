import argparse
import pprint
import re

from . import registry


parser = argparse.ArgumentParser()

parser.add_argument('experiment_str', type=str, nargs='?', default=None)
parser.add_argument('--action', choices=['list_all', 'regex', 'print_config'], default='list_all')


args = parser.parse_args()

if args.action == 'list_all':
    print('All experiments:')
    experiment_list = registry.list_experiments()
    for i, experiment_name in enumerate(experiment_list):
        print(f'{i}: {experiment_name}')
elif args.action == 'regex':
    regex = args.experiment_str
    print(f'Regex {regex} filter results:')
    experiment_list = [name for name in registry.list_experiments() if re.match(regex, name)]
    for i, experiment_name in enumerate(experiment_list):
        print(f'{i}: {experiment_name}')
else:   # args.action == 'print_config'
    experiment_name = args.experiment_str
    print(f'Experiment {experiment_name} config:')
    experiment_config = registry.get_experiment_config(experiment_name)
    pprint.pprint(experiment_config)
