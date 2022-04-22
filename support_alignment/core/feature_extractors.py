import torch
import torch.nn as nn
import torchvision


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# Adapted from
# https://github.com/microsoft/Domain-Adaptation-with-Conditional-Distribution-Matching-and-Generalized-Label-Shift/
# blob/main/network.py
class LeNet(nn.Module):
    def __init__(self, input_shape, hparams):
        super(LeNet, self).__init__()
        self.n_outputs = hparams.get('feature_dim', 500)
        self.conv_dropout_rate = hparams.get('conv_dropout_rate', 0.5)
        self.output_dropout_rate = hparams.get('output_dropout_rate', 0.5)
        self.conv_params = nn.Sequential(
            nn.Conv2d(input_shape[0], 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=self.conv_dropout_rate),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )

        h, w = input_shape[1:]

        h_f = (((((h - 4) - 2) // 2 + 1) - 4) - 2) // 2 + 1
        w_f = (((((w - 4) - 2) // 2 + 1) - 4) - 2) // 2 + 1

        self.fc_params = nn.Sequential(nn.Linear(50*h_f*w_f, self.n_outputs), nn.ReLU(),
                                       nn.Dropout(p=self.output_dropout_rate))

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.fc_params(x)
        return x


class LeNetNoReLU(nn.Module):
    def __init__(self, input_shape, hparams):
        super(LeNetNoReLU, self).__init__()
        self.n_outputs = hparams.get('feature_dim', 500)
        self.conv_dropout_rate = hparams.get('conv_dropout_rate', 0.5)
        self.output_dropout_rate = hparams.get('output_dropout_rate', 0.5)
        self.conv_params = nn.Sequential(
            nn.Conv2d(input_shape[0], 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=self.conv_dropout_rate),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )

        h, w = input_shape[1:]

        h_f = (((((h - 4) - 2) // 2 + 1) - 4) - 2) // 2 + 1
        w_f = (((((w - 4) - 2) // 2 + 1) - 4) - 2) // 2 + 1

        self.fc_params = nn.Sequential(nn.Linear(50*h_f*w_f, self.n_outputs),
                                       nn.Dropout(p=self.output_dropout_rate))

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.fc_params(x)
        return x


# Adapted from https://github.com/ozanciga/dirt-t/blob/master/models.py
class GaussianNoise(nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma
        self.noise = torch.tensor(0.0).cuda()

    def forward(self, x):
        if self.training:
            sampled_noise = self.noise.repeat(*x.size()).normal_(mean=0, std=self.sigma)
            x = x + sampled_noise
        return x


# Adapted from https://github.com/ozanciga/dirt-t/blob/master/models.py
class DeepCNN(torch.nn.Module):
    def __init__(self, input_shape, hparams):
        """

        haprams: {
            num_features -- number of features [64 - small, 192 - large]
            gaussian_noise -- level of Gaussian noise [suggested 1.0]
        }
        """
        super().__init__()
        n_features = hparams['num_features']
        gaussian_noise = hparams['gaussian_noise']
        self.n_outputs = n_features

        self.net = nn.Sequential(
            nn.InstanceNorm2d(input_shape[0], momentum=1, eps=1e-3),  # L-17
            nn.Conv2d(input_shape[0], n_features, 3, 1, 1),  # L-16
            nn.BatchNorm2d(n_features, momentum=0.99, eps=1e-3),  # L-16
            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-16
            nn.Conv2d(n_features, n_features, 3, 1, 1),  # L-15
            nn.BatchNorm2d(n_features, momentum=0.99, eps=1e-3),  # L-15
            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-15
            nn.Conv2d(n_features, n_features, 3, 1, 1),  # L-14
            nn.BatchNorm2d(n_features, momentum=0.99, eps=1e-3),  # L-14
            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-14
            nn.MaxPool2d(2),  # L-13
            nn.Dropout(0.5),  # L-12
            GaussianNoise(gaussian_noise),  # L-11
            nn.Conv2d(n_features, n_features, 3, 1, 1),  # L-10
            nn.BatchNorm2d(n_features, momentum=0.99, eps=1e-3),  # L-10
            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-10
            nn.Conv2d(n_features, n_features, 3, 1, 1),  # L-9
            nn.BatchNorm2d(n_features, momentum=0.99, eps=1e-3),  # L-9
            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-9
            nn.Conv2d(n_features, n_features, 3, 1, 1),  # L-8
            nn.BatchNorm2d(n_features, momentum=0.99, eps=1e-3),  # L-8
            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-8
            nn.MaxPool2d(2),  # L-7
            nn.Dropout(0.5),  # L-6
            GaussianNoise(gaussian_noise),  # L-5
            nn.Conv2d(n_features, n_features, 3, 1, 1),  # L-4
            nn.BatchNorm2d(n_features, momentum=0.99, eps=1e-3),  # L-4
            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-4
            nn.Conv2d(n_features, n_features, 3, 1, 1),  # L-3
            nn.BatchNorm2d(n_features, momentum=0.99, eps=1e-3),  # L-3
            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-3
            nn.Conv2d(n_features, n_features, 3, 1, 1),  # L-2
            nn.BatchNorm2d(n_features, momentum=0.99, eps=1e-3),  # L-2
            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-2
            nn.AdaptiveAvgPool2d(1),  # L-1
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                m.track_running_stats = False

    def forward(self, x):
        return self.net(x).view(x.size(0), self.n_outputs)


# Adapted from https://github.com/facebookresearch/DomainBed/blob/master/domainbed/networks.py
class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen (optinal)"""
    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        pretrained = hparams['pretrained']
        fc_dim = hparams['feature_dim']
        self.freeeze_bn_flag = hparams['freeze_bn']
        if hparams['resnet18']:
            self.network = torchvision.models.resnet18(pretrained=pretrained)
            conv_dim = 512
        else:
            self.network = torchvision.models.resnet50(pretrained=pretrained)
            conv_dim = 2048

        self.n_outputs = fc_dim if fc_dim else conv_dim

        # save memory
        del self.network.fc

        self.param_groups = [
            {
                'params': list(self.network.parameters()),
                'lr_factor': 1.0,
                'wd_factor': 1.0,
            }
        ]

        if fc_dim:
            self.network.fc = nn.Linear(conv_dim, fc_dim)
            nn.init.xavier_normal_(self.network.fc.weight)
            nn.init.zeros_(self.network.fc.bias)
            self.param_groups.append({
                'params': list(self.network.parameters())[-2:],
                'lr_factor': hparams['fc_lr_factor'],
                'wd_factor': hparams['fc_wd_factor'],
            })
        else:
            self.network.fc = Identity()

        self._freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self._freeze_bn()

    def _freeze_bn(self):
        if self.freeeze_bn_flag:
            for m in self.network.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def freeze_bn(self):
        self.freeeze_bn_flag = True

    def unfreeze_bn(self):
        self.freeeze_bn_flag = False
