# Only optimize on localy observable traffic

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Import from core directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'core'))
from model import TEFederatedModel
from topology import LINKS, NODE_LIST

# Output dimensions per node (total number of splitting ratios)
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

# Group sizes per node (tunnels per destination for grouped softmax)
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

# Default data directory (same as oracle-trained version)
DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'federated_data', 'federal-2026-02-05-22:31:03'
)


def _build_link_info():
    outgoing_links = {node: [] for node in NODE_LIST}
    link_capacity = {}

    for node_a, node_b, _, capacity in LINKS:
        # Links are bidirectional
        outgoing_links[node_a].append((node_b, capacity))
        outgoing_links[node_b].append((node_a, capacity))
        link_capacity[(node_a, node_b)] = capacity
        link_capacity[(node_b, node_a)] = capacity

    return outgoing_links, link_capacity


def _load_tunnels():
    tunnels_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'data', 'training_data', 'tunnels.json'
    )
    with open(tunnels_path, 'r') as f:
        return json.load(f)


def _build_tunnel_first_hop_mapping(node_name, tunnels):
    first_hops = []
    group_sizes = []

    destinations = [n for n in NODE_LIST if n != node_name]

    for dst in destinations:
        key = f"{node_name}->{dst}"
        tunnel_list = tunnels.get(key, [])
        group_sizes.append(len(tunnel_list))

        for tunnel in tunnel_list:
            path = tunnel["path"]
            first_hop = path[1] if len(path) > 1 else node_name
            first_hops.append(first_hop)

    return first_hops, group_sizes


# Build global topo
OUTGOING_LINKS, LINK_CAPACITY = _build_link_info()
TUNNELS = _load_tunnels()


def partition_to_node(partition_id):
    return NODE_LIST[partition_id]


def get_node_output_dim(node_name):
    return NODE_OUTPUT_DIMS[node_name]


def get_node_group_sizes(node_name):
    return NODE_GROUP_SIZES[node_name]


class TEFederatedDataset(Dataset):
    def __init__(self, node_name, split="train", data_dir=None):
        if data_dir is None:
            data_dir = DEFAULT_DATA_DIR

        node_dir = os.path.join(data_dir, node_name, split)

        self.node_name = node_name

        # Load data
        self.inputs_incoming = np.load(os.path.join(node_dir, "inputs_incoming.npy"))
        self.inputs_outgoing = np.load(os.path.join(node_dir, "inputs_outgoing.npy"))

        self.outputs = np.load(os.path.join(node_dir, "outputs.npy"))

        # Stack incoming and outgoing into 2 channels
        self.inputs = np.stack([self.inputs_incoming, self.inputs_outgoing], axis=1)

        # Extract current demands from last outgoing timestamp
        self.current_demands = self.inputs_outgoing[:, -1, :]

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, idx):
        return {
            "input": torch.tensor(self.inputs[idx], dtype=torch.float32),
            "output": torch.tensor(self.outputs[idx], dtype=torch.float32),
            "demands": torch.tensor(self.current_demands[idx], dtype=torch.float32),
        }


class LocalLoadBalancingLoss(nn.Module):
    def __init__(self, node_name, group_sizes):
        super().__init__()
        self.node_name = node_name
        self.group_sizes = group_sizes

        # Get outgoing links for this node
        self.outgoing_links = OUTGOING_LINKS[node_name]
        self.num_links = len(self.outgoing_links)

        # Build first-hop mapping
        first_hops, _ = _build_tunnel_first_hop_mapping(node_name, TUNNELS)

        # Create mapping from tunnel index to link index
        link_to_idx = {neighbor: i for i, (neighbor, _) in enumerate(self.outgoing_links)}

        tunnel_to_link = []
        for first_hop in first_hops:
            tunnel_to_link.append(link_to_idx[first_hop])

        self.register_buffer('tunnel_to_link', torch.tensor(tunnel_to_link, dtype=torch.long))

        # Link capacities
        capacities = [cap for _, cap in self.outgoing_links]
        self.register_buffer('link_capacities', torch.tensor(capacities, dtype=torch.float32))

        # Group indices for mapping destinations to tunnel ranges
        group_indices = [0] + list(np.cumsum(group_sizes))
        self.register_buffer('group_indices', torch.tensor(group_indices, dtype=torch.long))

    def forward(self, pred_ratios, demands):
        batch_size = pred_ratios.shape[0]

        # Compute traffic on each link
        # link_traffic[b, l] = sum over tunnels using link l of (demand * ratio)
        link_traffic = torch.zeros(batch_size, self.num_links, device=pred_ratios.device)

        # Iterate over destinations
        num_destinations = demands.shape[1]
        for dst_idx in range(num_destinations):
            start = self.group_indices[dst_idx]
            end = self.group_indices[dst_idx + 1]

            # Get ratios for tunnels to this destination
            dst_ratios = pred_ratios[:, start:end]  

            # Get demand to this destination
            dst_demand = demands[:, dst_idx:dst_idx+1]

            # Traffic per tunnel = demand * ratio
            tunnel_traffic = dst_demand * dst_ratios 

            # Add to link traffic based on first hop
            tunnel_links = self.tunnel_to_link[start:end]
            for t_idx, link_idx in enumerate(tunnel_links):
                link_traffic[:, link_idx] += tunnel_traffic[:, t_idx]

        # Compute link utilization
        link_utilization = link_traffic / (self.link_capacities.unsqueeze(0) + 1e-8)

        # Comibine two losses: variance of utilization to spread evenly across available
        # links and max utilization to not oversubscribe a link
        variance_loss = torch.var(link_utilization, dim=1).mean()
        max_util_loss = torch.max(link_utilization, dim=1).values.mean()

        # Combined loss: variance + weighted max utilization
        # todo: Weighting is chosen randomly. Should be a hyperparameter for proper investigation
        loss = variance_loss + 0.5 * max_util_loss

        return loss


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


def load_data(partition_id, batch_size, data_dir=None):
    node_name = partition_to_node(partition_id)

    train_dataset = TEFederatedDataset(node_name, split="train", data_dir=data_dir)
    val_dataset = TEFederatedDataset(node_name, split="val", data_dir=data_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train(model, trainloader, epochs, lr, device, node_name=None):
    if node_name is None:
        node_name = trainloader.dataset.node_name

    model.to(device)
    model.train()

    # Use local load balancing loss instead of kl loss
    criterion = LocalLoadBalancingLoss(node_name, model.group_sizes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    running_loss = 0.0
    num_batches = 0

    for _ in range(epochs):
        for batch in trainloader:
            inputs = batch["input"].to(device)
            demands = batch["demands"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, demands)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

    return running_loss / num_batches if num_batches > 0 else 0.0


def test(model, testloader, device, node_name=None):
    if node_name is None:
        node_name = testloader.dataset.node_name

    model.to(device)
    model.eval()

    local_criterion = LocalLoadBalancingLoss(node_name, model.group_sizes).to(device)
    kl_criterion = TELoss(model.group_sizes).to(device)

    total_local_loss = 0.0
    total_kl_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in testloader:
            inputs = batch["input"].to(device)
            demands = batch["demands"].to(device)
            targets = batch["output"].to(device)

            outputs = model(inputs)

            local_loss = local_criterion(outputs, demands)
            kl_loss = kl_criterion(outputs, targets)

            total_local_loss += local_loss.item()
            total_kl_loss += kl_loss.item()
            num_batches += 1

    avg_local_loss = total_local_loss / num_batches if num_batches > 0 else 0.0
    avg_kl_loss = total_kl_loss / num_batches if num_batches > 0 else 0.0

    return avg_local_loss, avg_kl_loss
