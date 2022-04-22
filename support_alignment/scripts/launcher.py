import argparse
import re
import subprocess
import sys

from support_alignment.experiment_registry import registry

parser = argparse.ArgumentParser()
parser.add_argument('args', nargs='*')
parser.add_argument('--regex', type=str, required=True, help='experiment name regex')

args = parser.parse_args()

regex = args.regex
print(f'Regex {regex} filter results:')
experiment_list = [name for name in registry.list_experiments() if re.match(regex, name)]
for i, experiment_name in enumerate(experiment_list):
    print(f'{i}: {experiment_name}')
print()

for experiment_name in experiment_list:
    print(f'Starting {experiment_name}')
    command = ['python3', '-m', 'support_alignment.scripts.train_da', f'--config_name={experiment_name}']
    command.extend(args.args or [])
    print(f'Command: {command}')
    subprocess.run(command, stdout=sys.stdout, stderr=sys.stderr)
    print()
