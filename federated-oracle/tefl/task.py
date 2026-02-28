import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Import from core directory for model and topology 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'core'))

from model import TEFederatedModel
from topology import NODE_LIST

# Output dimensions per node
# todo: Compute these instead of hardcoding here
NODE_OUTPUT_DIMS = {
    "SEAT": 38,
    "SNVA": 34,
    "LOSA": 35,
    "DENV": 34,
    "KSCY": 29,
    "HOUS": 35,
    "CHIN": 36,
    "IPLS": 29,
    "ATLA": 37,
    "WASH": 35,
    "NYCM": 36,
}

# Group sizes per node
NODE_GROUP_SIZES = {
    "SEAT": [3, 4, 3, 4, 4, 4, 4, 4, 4, 4],
    "SNVA": [3, 3, 3, 3, 3, 4, 3, 4, 4, 4],
    "LOSA": [4, 3, 3, 3, 3, 4, 3, 4, 4, 4],
    "DENV": [3, 3, 3, 3, 3, 4, 3, 4, 4, 4],
    "KSCY": [4, 3, 3, 3, 3, 3, 1, 3, 3, 3],
    "HOUS": [4, 3, 3, 3, 3, 4, 3, 4, 4, 4],
    "CHIN": [4, 4, 4, 4, 3, 4, 3, 4, 3, 3],
    "IPLS": [4, 3, 3, 3, 1, 3, 3, 3, 3, 3],
    "ATLA": [4, 4, 4, 4, 3, 4, 4, 3, 3, 4],
    "WASH": [4, 4, 4, 4, 3, 4, 3, 3, 3, 3],
    "NYCM": [4, 4, 4, 4, 3, 4, 3, 3, 4, 3],
}

# Input dimensions: 2 channels (incoming, outgoing) x H history x (N-1) nodes
HISTORY_LEN = 12
NUM_NODES = 11
INPUT_SIZE = 2 * HISTORY_LEN * (NUM_NODES - 1)  # 2 * 12 * 10 = 240

# Set last experiment dir as data directory
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'federated_data', 'federal-2026-02-05-22:31:03')

def partition_to_node(partition_id):
    return NODE_LIST[partition_id]


def get_node_output_dim(node_name):
    return NODE_OUTPUT_DIMS[node_name]


def get_node_group_sizes(node_name):
    return NODE_GROUP_SIZES[node_name]


class TEFederatedDataset(Dataset):
    def __init__(self, node_name, split = "train", data_dir = None):
        if data_dir is None:
            data_dir = DEFAULT_DATA_DIR

        node_dir = os.path.join(data_dir, node_name, split)

        # Load data for th node only
        self.inputs_incoming = np.load(os.path.join(node_dir, "inputs_incoming.npy"))
        self.inputs_outgoing = np.load(os.path.join(node_dir, "inputs_outgoing.npy"))
        self.outputs = np.load(os.path.join(node_dir, "outputs.npy"))

        # Stack incoming and outgoing into 2 channels
        self.inputs = np.stack([self.inputs_incoming, self.inputs_outgoing], axis=1)

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, idx):
        return {
            "input": torch.tensor(self.inputs[idx], dtype=torch.float32),
            "output": torch.tensor(self.outputs[idx], dtype=torch.float32),
        }


# KL Loss - same as central model
class TELoss(nn.Module):
    def __init__(self, group_sizes):
        super().__init__()
        self.group_sizes = group_sizes
        self.register_buffer(
            '_group_indices',
            torch.tensor([0] + list(np.cumsum(group_sizes)))
        )

    def forward(self, pred, target):
        eps = 1e-8
        total_loss = 0.0
        indices = self._group_indices

        for i in range(len(self.group_sizes)):
            start = indices[i]
            end = indices[i + 1]
            p = target[:, start:end] + eps
            q = pred[:, start:end] + eps
            kl = torch.sum(p * torch.log(p / q), dim=1)
            total_loss = total_loss + kl.mean()

        return total_loss / len(self.group_sizes)


def create_model(node_name):
    output_size = get_node_output_dim(node_name)
    group_sizes = get_node_group_sizes(node_name)

    return TEFederatedModel(INPUT_SIZE, output_size, group_sizes)


def load_data(partition_id, batch_size, data_dir= None):
    node_name = partition_to_node(partition_id)

    train_dataset = TEFederatedDataset(node_name, split="train", data_dir=data_dir)
    val_dataset = TEFederatedDataset(node_name, split="val", data_dir=data_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train(model, trainloader, epochs, lr, device):
    model.to(device)
    model.train()

    criterion = TELoss(model.group_sizes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    running_loss = 0.0
    num_batches = 0

    for _ in range(epochs):
        for batch in trainloader:
            inputs = batch["input"].to(device)
            targets = batch["output"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

    return running_loss / num_batches if num_batches > 0 else 0.0


def test(model, testloader, device):
    model.to(device)
    model.eval()

    criterion = TELoss(model.group_sizes).to(device)

    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in testloader:
            inputs = batch["input"].to(device)
            targets = batch["output"].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            total_mae += torch.mean(torch.abs(outputs - targets)).item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_mae = total_mae / num_batches if num_batches > 0 else 0.0

    return avg_loss, avg_mae
