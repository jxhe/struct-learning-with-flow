from __future__ import print_function
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReLUNet(nn.Module):
    def __init__(self, hidden_layers, hidden_units, in_features, out_features):
        super(ReLUNet, self).__init__()

        self.hidden_layers = hidden_layers
        self.in_layer = nn.Linear(in_features, hidden_units, bias=True)
        self.out_layer = nn.Linear(hidden_units, out_features, bias=True)
        for i in range(hidden_layers):
            name = 'cell{}'.format(i)
            cell = nn.Linear(hidden_units, hidden_units, bias=True)
            setattr(self, name, cell)

    def reset_parameters(self):
        self.in_layer.reset_parameters()
        self.out_layer.reset_parameters()
        for i in range(self.hidden_layers):
            name = 'cell{}'.format(i)
            getattr(self, name).reset_parameters()

    def init_identity(self):
        self.in_layer.weight.data.zero_()
        self.in_layer.bias.data.zero_()
        self.out_layer.weight.data.zero_()
        self.out_layer.bias.data.zero_()
        for i in range(self.hidden_layers):
            name = 'cell{}'.format(i)
            getattr(self, name).weight.data.zero_()
            getattr(self, name).bias.data.zero_()

    def forward(self, input):
        """
        input: (batch_size, seq_length, in_features)
        output: (batch_size, seq_length, out_features)

        """
        h = self.in_layer(input)
        h = F.relu(h)
        for i in range(self.hidden_layers):
            name = 'cell{}'.format(i)
            h = getattr(self, name)(h)
            h = F.relu(h)
        return self.out_layer(h)


class NICETrans(nn.Module):
    def __init__(self,
                 couple_layers,
                 cell_layers,
                 hidden_units,
                 features,
                 device):
        super(NICETrans, self).__init__()

        self.device = device
        self.couple_layers = couple_layers

        for i in range(couple_layers):
            name = 'cell{}'.format(i)
            cell = ReLUNet(cell_layers, hidden_units, features//2, features//2)
            setattr(self, name, cell)

    def reset_parameters(self):
        for i in range(self.couple_layers):
            name = 'cell{}'.format(i)
            getattr(self, name).reset_parameters()

    def init_identity(self):
        for i in range(self.couple_layers):
            name = 'cell{}'.format(i)
            getattr(self, name).init_identity()


    def forward(self, input):
        """
        input: (seq_length, batch_size, features)
        h: (seq_length, batch_size, features)

        """

        # For NICE it is a constant
        jacobian_loss = torch.zeros(1, device=self.device,
                                    requires_grad=False)

        ep_size = input.size()
        features = ep_size[-1]
        # h = odd_input
        h = input
        for i in range(self.couple_layers):
            name = 'cell{}'.format(i)
            h1, h2 = torch.split(h, features//2, dim=-1)
            if i%2 == 0:
                h = torch.cat((h1, h2 + getattr(self, name)(h1)), dim=-1)
            else:
                h = torch.cat((h1 + getattr(self, name)(h2), h2), dim=-1)
        return h, jacobian_loss
