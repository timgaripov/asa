import os
import importlib


class ExperimentRegistry(object):
    def __init__(self):
        self.experiment_dict = None

    def _build(self):
        if self.experiment_dict is None:
            self.experiment_dict = dict()
            for file in os.listdir(os.path.dirname(__file__)):
                if not file.startswith('exp_'):
                    continue
                mod_name = file[:-3]  # strip .py at the end
                module = importlib.import_module('.' + mod_name, package=__name__)
                module.register_experiments(self)

    def register(self, name, config):
        if name in self.experiment_dict:
            raise RuntimeError(f'Experiment "{name}" is already registered')
        self.experiment_dict[name] = config

    def list_experiments(self):
        self._build()
        return list(sorted(self.experiment_dict.keys()))

    def get_experiment_config(self, name):
        self._build()
        return self.experiment_dict[name]


registry = ExperimentRegistry()
