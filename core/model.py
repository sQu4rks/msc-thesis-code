#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn


class DOTEModel(nn.Module):

    def __init__(self, input_size, output_size, group_sizes=None):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.group_sizes = group_sizes

        # Prepare softmax over all tunnel pairs
        self.register_buffer('_group_indices', torch.tensor([0] + list(np.cumsum(self.group_sizes))))

        # Build MLP
        HIDDEN_SIZE=256
        DROPOUT = 0.1

        self.input = nn.Linear(input_size, HIDDEN_SIZE)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(DROPOUT)

        self.linear2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(DROPOUT)

        self.linear3 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(DROPOUT)

        self.output = nn.Linear(HIDDEN_SIZE, output_size)

        self.backbone = nn.Sequential(self.input, self.relu1, self.dropout1,
                                       self.linear2, self.relu2, self.dropout2,
                                       self.linear3, self.relu3, self.dropout3,
                                       self.output)

    def forward(self, x):
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        logits = self.backbone(x_flat)

        outputs = []
        indices = self._group_indices

        ratios = None
        # Either normalize softmax across groups of tunnels (if group size is not none) or use simple softmax for single tunnel
        if self.group_sizes is not None:
            for i in range(len(self.group_sizes)):
                start = indices[i]
                end = indices[i + 1]
                group_logits = logits[:, start:end]
                group_probs = torch.softmax(group_logits, dim=1)
                outputs.append(group_probs)
            ratios = torch.cat(outputs, dim=1)
        else:
            softm = nn.Softmax(dim=1)
            ratios = softm(logits)

        return ratios


class TEFederatedModel(nn.Module):

    HIDDEN_SIZE = 256
    DROPOUT = 0.1

    def __init__(self, input_size, output_size, group_sizes):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.group_sizes = group_sizes

        # Prepare softmax group indices
        self.register_buffer(
            '_group_indices',
            torch.tensor([0] + list(np.cumsum(self.group_sizes)))
        )

        # backbone shared to server
        self.backbone = nn.Sequential(
            nn.Linear(input_size, self.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(self.DROPOUT),
            nn.Linear(self.HIDDEN_SIZE, self.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(self.DROPOUT),
            nn.Linear(self.HIDDEN_SIZE, self.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(self.DROPOUT),
        )

        # per node head - not send to server for aggregation
        self.head = nn.Linear(self.HIDDEN_SIZE, output_size)

    def forward(self, x):
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)

        features = self.backbone(x_flat)

        logits = self.head(features)

        # aggregate with softmax over group
        outputs = []
        indices = self._group_indices
        for i in range(len(self.group_sizes)):
            start = indices[i]
            end = indices[i + 1]
            group_logits = logits[:, start:end]
            group_probs = torch.softmax(group_logits, dim=1)
            outputs.append(group_probs)

        return torch.cat(outputs, dim=1)

    def get_backbone_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if k.startswith('backbone.')}

    def load_backbone_state_dict(self, state_dict):
        current_state = self.state_dict()
        for k, v in state_dict.items():
            if k.startswith('backbone.'):
                current_state[k] = v
        self.load_state_dict(current_state)
