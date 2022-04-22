import collections

import torch


# Adapted from https://github.com/facebookresearch/DomainBed/blob/master/domainbed/lib/fast_data_loader.py
class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


def create_infinite_iterator(dataset, batch_size, num_workers, weights=None):
    if weights is not None:
        sampler = torch.utils.data.WeightedRandomSampler(weights, replacement=True, num_samples=batch_size)
    else:
        sampler = torch.utils.data.RandomSampler(dataset, replacement=True)
    batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=batch_size, drop_last=True)
    infinite_iterator = iter(torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=_InfiniteSampler(batch_sampler),
    ))
    return infinite_iterator


class DatasetWeighting:
    @staticmethod
    def class_uniform(num_classes, *_):
        return torch.ones(num_classes) / num_classes

    @staticmethod
    def class_prob(_, class_probs):
        probs = torch.tensor(class_probs)
        probs = probs / torch.sum(probs)
        return probs

    @staticmethod
    def class_pareto(num_classes, alpha, reverse, seed=None):
        class_probs = torch.arange(1, num_classes + 1) ** (-alpha)
        class_probs /= torch.sum(class_probs)
        if reverse:
            class_probs = torch.flip(class_probs, [0])
        if seed is not None:
            with torch.random.fork_rng():
                torch.random.manual_seed(seed)
                permutation = torch.randperm(class_probs.numel())
                class_probs = class_probs[permutation]
        return class_probs


def dataset_class_weighting(dataset, class_probs=None):
    classes = None
    if hasattr(dataset, 'attributes'):
        classes = dataset.attributes.get('classes', None)
    if classes is None:
        classes = [int(y) for _, y in dataset]
    counter = collections.Counter(classes)

    num_classes = len(counter)
    original_class_samples = [counter[y] for y in range(num_classes)]

    if class_probs is None:
        num_samples = len(classes)
        class_distribution = [counter[y] / num_samples for y in range(num_classes)]
        return None, class_distribution, original_class_samples

    class_weights = [0.0 for _ in range(num_classes)]
    for y, n_y in counter.items():
        class_weights[y] = 1.0 / n_y * class_probs[y]

    weights = torch.tensor([class_weights[y] for y in classes])
    class_distribution = class_probs

    return weights, class_distribution, original_class_samples


def dataset_class_subsampling(dataset, class_probs, seed=None):
    classes = None
    if hasattr(dataset, 'attributes'):
        classes = dataset.attributes.get('classes', None)
    if classes is None:
        classes = [int(y) for _, y in dataset]
    counter = collections.Counter(classes)

    num_classes = len(counter)
    original_class_samples = [counter[y] for y in range(num_classes)]

    num_samples = min([int(counter[y] / class_probs[y]) for y in range(num_classes)])
    class_samples = [max(int(num_samples * prob), 1) for prob in class_probs]

    indices = []
    with torch.random.fork_rng(enabled=seed is not None):
        if seed is not None:
            torch.random.manual_seed(seed)

        for y in range(num_classes):
            class_indices = torch.tensor([i for i, label in enumerate(classes) if label == y])

            class_indices = class_indices[torch.randperm(class_indices.size(0))[:class_samples[y]]]
            indices.append(class_indices)

    indices = torch.cat(indices, dim=0)

    return indices, class_samples, original_class_samples


class DATrainIterator(object):
    def __init__(self, source_dataset, source_weights, target_dataset, target_weights, batch_size, num_workers, device):
        self.device = device
        self.source_iterator = create_infinite_iterator(source_dataset, batch_size, num_workers, weights=source_weights)
        self.target_iterator = create_infinite_iterator(target_dataset, batch_size, num_workers, weights=target_weights)

    def __next__(self):
        x_src, y_src = next(self.source_iterator)
        x_trg, _ = next(self.target_iterator)
        return x_src.to(self.device, non_blocking=True), \
               y_src.to(self.device, non_blocking=True), \
               x_trg.to(self.device, non_blocking=True)
