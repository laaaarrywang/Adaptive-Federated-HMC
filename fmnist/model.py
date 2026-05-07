from __future__ import absolute_import

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc = nn.Linear(28*28*1, 10)
    def forward(self, x):
        x = x.view(-1, 28*28*1)
        x = self.fc(x)
        return x
