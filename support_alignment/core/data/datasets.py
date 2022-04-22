import os

import torch
import torchvision
import torchvision.transforms as transforms


class DatasetWithAttributesWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.attributes = dict()

    def __getitem__(self, item):
        return self.base_dataset[item]

    def __len__(self):
        return len(self.base_dataset)


class ClassSubsetDatasetWrapper(DatasetWithAttributesWrapper):
    def __init__(self, base_dataset, class_subset):
        super(ClassSubsetDatasetWrapper, self).__init__(base_dataset)

        classes = None
        if hasattr(base_dataset, 'attributes'):
            classes = base_dataset.attributes.get('classes', None)
        if classes is None:
            classes = [int(y) for _, y in base_dataset]

        self.class_mapping = {class_id: i for i, class_id in enumerate(class_subset)}
        self.indices = [i for i, y in enumerate(classes) if int(y) in self.class_mapping]
        if isinstance(base_dataset, DatasetWithAttributesWrapper):
            for key, attr_list in base_dataset.attributes.items():
                self.attributes[key] = [attr_list[i] for i in self.indices]
        self.attributes['classes'] = [self.class_mapping[classes[i]] for i in self.indices]

    def __getitem__(self, item):
        x, y = self.base_dataset[self.indices[item]]
        y = self.class_mapping[int(y)]
        return x, y

    def __len__(self):
        return len(self.indices)


class ClassGroupingDatasetWrapper(DatasetWithAttributesWrapper):
    def __init__(self, base_dataset, class_grouping):
        super(ClassGroupingDatasetWrapper, self).__init__(base_dataset)

        classes = None
        if hasattr(base_dataset, 'attributes'):
            classes = base_dataset.attributes.get('classes', None)
        if classes is None:
            classes = [int(y) for _, y in base_dataset]

        self.class_mapping = dict()
        for new_class, class_group in enumerate(class_grouping):
            for old_class in class_group:
                self.class_mapping[old_class] = new_class
        if isinstance(base_dataset, DatasetWithAttributesWrapper):
            self.attributes = {key: attr_list for key, attr_list in base_dataset.attributes.items()}
        self.attributes['subclasses'] = classes
        self.attributes['classes'] = [self.class_mapping[y] for y in classes]

    def __getitem__(self, item):
        x, y = self.base_dataset[item]
        y = self.class_mapping[int(y)]
        return x, y

    def __len__(self):
        return len(self.base_dataset)


class SubsetDatasetWrapper(DatasetWithAttributesWrapper):
    def __init__(self, base_dataset, indices):
        super(SubsetDatasetWrapper, self).__init__(base_dataset)
        self.indices = indices
        if isinstance(base_dataset, DatasetWithAttributesWrapper):
            for key, attr_list in base_dataset.attributes.items():
                self.attributes[key] = [attr_list[i] for i in self.indices]

    def __getitem__(self, item):
        return self.base_dataset[self.indices[item]]

    def __len__(self):
        return len(self.indices)


def split_dataset(dataset, split_fraction=0.2, seed=None):
    with torch.random.fork_rng(enabled=seed is not None):
        if seed is not None:
            torch.random.manual_seed(seed)
        n = len(dataset)
        n_cut = int((1.0 - split_fraction) * n)
        if seed is None:
            permutation = torch.arange(n)
        else:
            permutation = torch.randperm(n)

        indices_1 = permutation[:n_cut]
        indices_2 = permutation[n_cut:]

        return SubsetDatasetWrapper(dataset, indices_1), \
               SubsetDatasetWrapper(dataset, indices_2)


class TransformDatasetWrapper(DatasetWithAttributesWrapper):
    def __init__(self, base_dataset, transform):
        super(TransformDatasetWrapper, self).__init__(base_dataset)
        self.transform = transform
        if isinstance(base_dataset, DatasetWithAttributesWrapper):
            self.attributes = {key: attr_list for key, attr_list in base_dataset.attributes.items()}

    def __getitem__(self, item):
        x, y = self.base_dataset[item]
        x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.base_dataset)


# Adapted from https://github.com/facebookresearch/DomainBed/blob/master/domainbed/datasets.py

class MultipleDomainDataset:
    # Adapted from https://github.com/facebookresearch/DomainBed/blob/master/domainbed/datasets.py
    # attributes:
    #   input_shape
    #   num_classes
    #   datasets
    ENVIRONMENTS = tuple()

    def __init__(self):
        self.input_shape = tuple()
        self.num_classes = 0
        self.datasets = list()
        self.class_names = None

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)

    def mod_class_subset(self, class_subset):
        for class_index in class_subset:
            if class_index < 0 or class_index >= self.num_classes:
                raise ValueError("Class subset index {0} is not in range [0, {1})".format(
                    class_index, self.num_classes))

        self.num_classes = len(class_subset)
        self.datasets = [ClassSubsetDatasetWrapper(dataset, class_subset) if dataset is not None else None
                         for dataset in self.datasets]
        if self.class_names is not None:
            self.class_names = [self.class_names[i] for i in class_subset]

    def mod_class_grouping(self, class_grouping):
        used_classes = set()
        for class_group in class_grouping:
            for class_index in class_group:
                if class_index < 0 or class_index >= self.num_classes:
                    raise ValueError("Class index {0} is not in range [0, {1})".format(
                        class_index, self.num_classes))
                used_classes.add(class_index)
        for class_index in range(self.num_classes):
            if class_index not in used_classes:
                raise ValueError("Class {0} is not present in the class grouping".format(class_index))
        if sum([len(class_group) for class_group in class_grouping]) != self.num_classes:
            raise ValueError("Class grouping {0} total size is different from the number of classes {1}".format(
                class_grouping, self.num_classes))

        self.num_classes = len(class_grouping)
        self.datasets = [ClassGroupingDatasetWrapper(dataset, class_grouping) if dataset is not None else None
                         for dataset in self.datasets]
        if self.class_names is not None:
            self.class_names = [' | '.join([self.class_names[i] for i in class_group])
                                for class_group in class_grouping]

    # MOD DESCRIPTION structure:
    # {'name': <name>, 'args': <argument_list>}
    # when applying a mod the method with the name 'mod_<name>' will be called

    def apply_mod(self, mod_name, mod_args):
        mod_fn = getattr(self, 'mod_{0}'.format(mod_name))
        mod_fn(*mod_args)


class MNIST_USPS(MultipleDomainDataset):
    ENVIRONMENTS = ('MNIST', 'USPS')

    def __init__(self, root):
        super().__init__()
        self.input_shape = (1, 28, 28)
        self.num_classes = 10
        self.class_names = ['{0}'.format(i) for i in range(10)]

        norm_transform = transforms.Normalize(mean=[0.5], std=[0.5])

        inv_norm_transform = transforms.Normalize(mean=[-1.0], std=[2.0])

        mnist_transform = transforms.Compose([
            transforms.ToTensor(),
            norm_transform
        ])

        mnist_tr = torchvision.datasets.MNIST(root, train=True, download=True, transform=mnist_transform)
        mnist_te = torchvision.datasets.MNIST(root, train=False, download=True, transform=mnist_transform)
        mnist_combined = torch.utils.data.ConcatDataset([mnist_tr, mnist_te])
        mnist_classes = torch.cat((mnist_tr.targets, mnist_te.targets))
        mnist_combined = DatasetWithAttributesWrapper(mnist_combined)
        mnist_combined.attributes['classes'] = [int(y) for y in mnist_classes]

        usps_transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            norm_transform
        ])
        usps_tr = torchvision.datasets.USPS(root, train=True, download=True, transform=usps_transform)
        usps_te = torchvision.datasets.USPS(root, train=False, download=True, transform=usps_transform)
        usps_combined = torch.utils.data.ConcatDataset([usps_tr, usps_te])
        usps_classes = torch.cat((torch.tensor(usps_tr.targets), torch.tensor(usps_te.targets)))
        usps_combined = DatasetWithAttributesWrapper(usps_combined)
        usps_combined.attributes['classes'] = [int(y) for y in usps_classes]

        self.datasets = [mnist_combined, usps_combined]

        self.data_params = {
            'inv_norm_transform': inv_norm_transform
        }


class CIFAR_STL(MultipleDomainDataset):
    ENVIRONMENTS = ('CIFAR', 'STL')

    def __init__(self, root):
        super().__init__()
        self.input_shape = (3, 32, 32)
        self.num_classes = 9
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'horse', 'ship', 'truck']

        norm_transform = transforms.Normalize(mean=[0.5], std=[0.5])
        inv_norm_transform = transforms.Normalize(mean=[-1.0], std=[2.0])

        cifar_transform = transforms.Compose([
            transforms.ToTensor(),
            norm_transform
        ])

        cifar_tr = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=cifar_transform)
        cifar_te = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=cifar_transform)

        cifar_class_mapping = torch.tensor([0, 1, 2, 3, 4, 5, -1, 6, 7, 8], dtype=torch.long)
        cifar_tr_classes = cifar_class_mapping[cifar_tr.targets]
        cifar_te_classes = cifar_class_mapping[cifar_te.targets]

        cifar_tr_subset_index = torch.nonzero(cifar_tr_classes >= 0, as_tuple=True)[0]
        cifar_te_subset_index = torch.nonzero(cifar_te_classes >= 0, as_tuple=True)[0]
        cifar_tr.data = cifar_tr.data[cifar_tr_subset_index.numpy()]
        cifar_te.data = cifar_te.data[cifar_te_subset_index.numpy()]
        cifar_tr_classes = cifar_tr_classes[cifar_tr_subset_index]
        cifar_te_classes = cifar_te_classes[cifar_te_subset_index]
        cifar_tr.targets = cifar_tr_classes.tolist()
        cifar_te.targets = cifar_te_classes.tolist()

        cifar_combined = torch.utils.data.ConcatDataset([cifar_tr, cifar_te])
        cifar_classes = torch.cat((cifar_tr_classes, cifar_te_classes))
        cifar_combined = DatasetWithAttributesWrapper(cifar_combined)
        cifar_combined.attributes['classes'] = [int(y) for y in cifar_classes]

        stl_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            norm_transform
        ])

        stl_tr = torchvision.datasets.STL10(root, split='train', download=True, transform=stl_transform)
        stl_te = torchvision.datasets.STL10(root, split='test', download=True, transform=stl_transform)

        stl_class_mapping = torch.tensor([0, 2, 1, 3, 4, 5, 6, -1, 7, 8], dtype=torch.long)
        stl_tr_classes = stl_class_mapping[stl_tr.labels.tolist()]
        stl_te_classes = stl_class_mapping[stl_te.labels.tolist()]

        stl_tr_subset_index = torch.nonzero(stl_tr_classes >= 0, as_tuple=True)[0]
        stl_te_subset_index = torch.nonzero(stl_te_classes >= 0, as_tuple=True)[0]
        stl_tr.data = stl_tr.data[stl_tr_subset_index.numpy()]
        stl_te.data = stl_te.data[stl_te_subset_index.numpy()]
        stl_tr_classes = stl_tr_classes[stl_tr_subset_index]
        stl_te_classes = stl_te_classes[stl_te_subset_index]
        stl_tr.labels = stl_tr_classes.numpy()
        stl_te.labels = stl_te_classes.numpy()

        stl_combined = torch.utils.data.ConcatDataset([stl_tr, stl_te])
        stl_classes = torch.cat((stl_tr_classes, stl_te_classes))
        stl_combined = DatasetWithAttributesWrapper(stl_combined)
        stl_combined.attributes['classes'] = [int(y) for y in stl_classes]

        self.datasets = [cifar_combined, stl_combined]

        self.data_params = {
            'inv_norm_transform': inv_norm_transform
        }


# Adapted from https://github.com/facebookresearch/DomainBed/blob/master/domainbed/datasets.py
class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, train_aug=None, eval_aug=None):
        super().__init__()
        env_dirs = [f.name for f in os.scandir(root) if f.is_dir()]
        self.env_dirs = sorted(env_dirs)

        norm_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        inv_norm_transform = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225]
        )

        if train_aug is None:
            train_aug = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                transforms.RandomGrayscale(),
            ])

        self.train_transform = transforms.Compose([
            train_aug,
            transforms.ToTensor(),
            norm_transform
        ])

        if eval_aug is None:
            eval_aug = transforms.Resize((224, 224))

        self.eval_transform = transforms.Compose([
            eval_aug,
            transforms.ToTensor(),
            norm_transform
        ])

        self.data_params = {
            'inv_norm_transform': inv_norm_transform
        }

        self.datasets = []
        num_classes = 0
        self.class_names = None
        for i, environment in enumerate(self.env_dirs):
            path = os.path.join(root, environment)
            base_dataset = torchvision.datasets.ImageFolder(path, transform=None)
            num_classes = len(base_dataset.classes)
            self.class_names = base_dataset.classes
            # Wrap dataset and extract class labels for all samples to save time
            env_dataset = base_dataset
            env_dataset = DatasetWithAttributesWrapper(env_dataset)
            env_dataset.attributes['classes'] = [int(label) for _, label in base_dataset.samples]

            self.datasets.append(env_dataset)

        self.input_shape = None
        self.num_classes = num_classes


class VisDA17(MultipleEnvironmentImageFolder):
    ENVIRONMENTS = ['Train', 'Val']

    # Dataset root must contain visda17 folder with only train and validation directory
    # test split should be saved in a separated directory
    def __init__(self, root):
        self.dir = os.path.join(root, 'visda17')

        train_aug = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])

        eval_aug = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
        ])

        super().__init__(self.dir, train_aug=train_aug, eval_aug=eval_aug)
        if self.env_dirs != ['train', 'validation']:
            raise ValueError(
                """VisDA17 folder must contain train and validation folders and no other folders.
                Found: {0}""".format(self.env_dirs)
            )
        self.input_shape = (3, 224, 224)
