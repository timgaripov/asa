import math

import torch
import torch.nn as nn


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class LogLossClassifier(nn.Module):
    """
        Log loss classifier

        hparams: {
            num_hidden -- number of units in the hidden layer; if None the classifier network is linear
            special_init [optional] -- use special weight init (default: False)
        }
    """
    def __init__(self, in_dim, num_classes, hparams):
        super(LogLossClassifier, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.hparams = hparams
        self.num_hidden = self.hparams['num_hidden']
        if self.num_hidden is not None:
            self.net = nn.Sequential(
                nn.Linear(in_dim, self.num_hidden),
                nn.LeakyReLU(0.2),
                nn.Linear(self.num_hidden, num_classes)
            )
        else:
            self.net = nn.Linear(in_dim, num_classes)

        special_init = self.hparams.get('special_init', False)
        if special_init:
            self.net.apply(init_weights)

    def forward(self, x):
        return self.net(x)

    def w_norm(self):
        w_norm_sq = 0.0
        with torch.no_grad():
            for mod in self.modules():
                if isinstance(mod, nn.Linear):
                    w_norm_sq += torch.sum(torch.square(mod.weight))
        return math.sqrt(w_norm_sq)
