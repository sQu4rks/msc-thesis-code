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

# Output dimensions per node
# todo: compute them instead of hardcoding
NODE_OUTPUT_DIMS = {
    "SEAT": 38, "SNVA": 34, "LOSA": 35, "DENV": 34, "KSCY": 29,
    "HOUS": 35, "CHIN": 36, "IPLS": 29, "ATLA": 37, "WASH": 35, "NYCM": 36,
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

# Input dimensions
HISTORY_LEN = 12
NUM_NODES = 11
NODE_IDX = {name: i for i, name in enumerate(NODE_LIST)}
MAX_NEIGHBORS = 4 

# Base input: 2 channels (incoming, outgoing) x H x (N-1)
BASE_INPUT_SIZE = 2 * HISTORY_LEN * (NUM_NODES - 1) 
# Total input: base + neighbor loads (padded to MAX_NEIGHBORS)
INPUT_SIZE = BASE_INPUT_SIZE + HISTORY_LEN * MAX_NEIGHBORS

# Default data directory
DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'training_data'
)


def _build_link_info():
    outgoing_links = {node: [] for node in NODE_LIST}
    link_capacity = {}
    neighbors = {node: [] for node in NODE_LIST}

    for node_a, node_b, _, capacity in LINKS:
        outgoing_links[node_a].append((node_b, capacity))
        outgoing_links[node_b].append((node_a, capacity))
        link_capacity[(node_a, node_b)] = capacity
        link_capacity[(node_b, node_a)] = capacity
        neighbors[node_a].append(node_b)
        neighbors[node_b].append(node_a)

    return outgoing_links, link_capacity, neighbors


def _load_tunnels():
    tunnels_path = os.path.join(DEFAULT_DATA_DIR, 'tunnels.json')
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


# Build topo structures
OUTGOING_LINKS, LINK_CAPACITY, NEIGHBORS = _build_link_info()
TUNNELS = _load_tunnels()


def get_input_size(node_name):
    return INPUT_SIZE # fixed because backbone needs to aggregate across all nodes

def partition_to_node(partition_id):
    return NODE_LIST[partition_id]


def get_node_output_dim(node_name):
    return NODE_OUTPUT_DIMS[node_name]


def get_node_group_sizes(node_name):
    return NODE_GROUP_SIZES[node_name]


def _load_raw_data(data_dir):
    raw_path = os.path.join(data_dir, "raw_data.json")
    with open(raw_path) as f:
        return json.load(f)


def _compute_node_total_loads(raw_data):
    num_epochs = len(raw_data)
    node_loads = {node: np.zeros(num_epochs) for node in NODE_LIST}

    for epoch_idx, epoch_data in enumerate(raw_data):
        demands = epoch_data["demands"]
        for key, value in demands.items():
            src, dst = key.split("->")
            node_loads[src][epoch_idx] += value

    return node_loads


class TENeighborDataset(Dataset):
    def __init__(self, node_name, split="train", data_dir=None):
        if data_dir is None:
            data_dir = DEFAULT_DATA_DIR

        self.node_name = node_name
        self.node_idx = NODE_IDX[node_name]
        self.neighbors = NEIGHBORS[node_name]
        self.num_neighbors = len(self.neighbors)

        # Load base federated data
        fed_data_dir = os.path.join(
            os.path.dirname(__file__), '..', '..', 'data', 'federated_data',
            'federal-2026-02-05-22:31:03'
        )
        node_dir = os.path.join(fed_data_dir, node_name, split)

        self.inputs_incoming = np.load(os.path.join(node_dir, "inputs_incoming.npy"))
        self.inputs_outgoing = np.load(os.path.join(node_dir, "inputs_outgoing.npy"))
        self.outputs = np.load(os.path.join(node_dir, "outputs.npy"))

        # Load raw data to compute neighbor loads
        raw_data = _load_raw_data(data_dir)
        all_node_loads = _compute_node_total_loads(raw_data)

        # Get split indices (same logic as federated data generation)
        H = HISTORY_LEN  # 12
        n_total = len(raw_data)
        train_end = int((n_total - H) * 0.7) + H
        val_end = int((n_total - H) * 0.85) + H

        if split == "train":
            start_idx, end_idx = H, train_end
        elif split == "val":
            start_idx, end_idx = train_end, val_end
        else:  # test
            start_idx, end_idx = val_end, n_total

        # Extract neighbor loads for this split
        num_samples = end_idx - start_idx
        self.neighbor_loads = np.zeros((num_samples, H, MAX_NEIGHBORS))

        for sample_idx in range(num_samples):
            epoch_idx = start_idx + sample_idx
            for t in range(H):
                hist_epoch = epoch_idx - H + 1 + t
                if hist_epoch >= 0:
                    for n_idx, neighbor in enumerate(self.neighbors):
                        self.neighbor_loads[sample_idx, t, n_idx] = all_node_loads[neighbor][hist_epoch]

        # Normalize neighbor loads
        non_zero_mask = self.neighbor_loads > 0
        if non_zero_mask.any():
            self.neighbor_loads_mean = np.mean(self.neighbor_loads[non_zero_mask])
            self.neighbor_loads_std = np.std(self.neighbor_loads[non_zero_mask]) + 1e-8
        else:
            self.neighbor_loads_mean = 0.0
            self.neighbor_loads_std = 1.0
        self.neighbor_loads_normalized = (
            self.neighbor_loads - self.neighbor_loads_mean
        ) / self.neighbor_loads_std

        # Keep padding as 0
        self.neighbor_loads_normalized[~non_zero_mask] = 0.0

        # Stack base inputs
        self.inputs_base = np.stack([self.inputs_incoming, self.inputs_outgoing], axis=1)

        self.current_demands = self.inputs_outgoing[:, -1, :]

        # Current neighbor loads
        self.current_neighbor_loads = self.neighbor_loads[:, -1, :]

        # store mask to determine which positions are real neighbors
        self.neighbor_mask = np.zeros(MAX_NEIGHBORS, dtype=bool)
        self.neighbor_mask[:self.num_neighbors] = True

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, idx):
        # Combine base demand inputs with neighbor loads
        base_flat = self.inputs_base[idx].flatten()

        neighbor_flat = self.neighbor_loads_normalized[idx].flatten()
        combined_input = np.concatenate([base_flat, neighbor_flat])

        return {
            "input": torch.tensor(combined_input, dtype=torch.float32),
            "output": torch.tensor(self.outputs[idx], dtype=torch.float32),
            "demands": torch.tensor(self.current_demands[idx], dtype=torch.float32),
            "neighbor_loads": torch.tensor(self.current_neighbor_loads[idx], dtype=torch.float32),
            "neighbor_mask": torch.tensor(self.neighbor_mask, dtype=torch.bool),
        }


class NeighborAwareLoss(nn.Module):
    def __init__(self, node_name, group_sizes):
        super().__init__()
        self.node_name = node_name
        self.group_sizes = group_sizes
        self.neighbors = NEIGHBORS[node_name]
        self.num_neighbors = len(self.neighbors)

        # Get outgoing links for this node
        self.outgoing_links = OUTGOING_LINKS[node_name]
        self.num_links = len(self.outgoing_links)

        # Build mapping from neighbor to link index
        self.neighbor_to_link_idx = {
            neighbor: i for i, (neighbor, _) in enumerate(self.outgoing_links)
        }

        # Build first-hop mapping
        first_hops, _ = _build_tunnel_first_hop_mapping(node_name, TUNNELS)
        link_to_idx = {neighbor: i for i, (neighbor, _) in enumerate(self.outgoing_links)}
        tunnel_to_link = [link_to_idx[fh] for fh in first_hops]

        self.register_buffer('tunnel_to_link', torch.tensor(tunnel_to_link, dtype=torch.long))

        # link capacities
        capacities = [cap for _, cap in self.outgoing_links]
        self.register_buffer('link_capacities', torch.tensor(capacities, dtype=torch.float32))

        # Group indices
        group_indices = [0] + list(np.cumsum(group_sizes))
        self.register_buffer('group_indices', torch.tensor(group_indices, dtype=torch.long))

    def forward(self, pred_ratios, demands, neighbor_loads, neighbor_mask=None):
        batch_size = pred_ratios.shape[0]

        # Compute traffic on each local link
        link_traffic = torch.zeros(batch_size, self.num_links, device=pred_ratios.device)

        num_destinations = demands.shape[1]
        for dst_idx in range(num_destinations):
            start = self.group_indices[dst_idx]
            end = self.group_indices[dst_idx + 1]
            dst_ratios = pred_ratios[:, start:end]
            dst_demand = demands[:, dst_idx:dst_idx+1]
            tunnel_traffic = dst_demand * dst_ratios

            tunnel_links = self.tunnel_to_link[start:end]
            for t_idx, link_idx in enumerate(tunnel_links):
                link_traffic[:, link_idx] += tunnel_traffic[:, t_idx]

        # Combine four losses: node local util, variance loss for even spread over links
        # max utilization loss to not oversubscribe links and apply the penalty for 
        # going via congested neighbors
        link_utilization = link_traffic / (self.link_capacities.unsqueeze(0) + 1e-8)

        variance_loss = torch.var(link_utilization, dim=1).mean()
        max_util_loss = torch.max(link_utilization, dim=1).values.mean()

        real_neighbor_loads = neighbor_loads[:, :self.num_links]  # (batch, num_links)

        # Normalize neighbor loads
        neighbor_load_sum = real_neighbor_loads.sum(dim=1, keepdim=True) + 1e-8
        neighbor_load_normalized = real_neighbor_loads / neighbor_load_sum

        traffic_sum = link_traffic.sum(dim=1, keepdim=True) + 1e-8
        traffic_normalized = link_traffic / traffic_sum

        # Penalize sending traffic to a link from a busy neighbor
        congestion_penalty = (traffic_normalized * neighbor_load_normalized).sum(dim=1).mean()

        # Combined loss
        # todo: These weight factors are pretty much randomly chosen. Should be a hyper parameter and properly investigated
        loss = (
            variance_loss +
            0.5 * max_util_loss +
            0.3 * congestion_penalty
        )

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


class TENeighborModel(nn.Module):
    HIDDEN_SIZE = 256
    DROPOUT = 0.1

    def __init__(self, input_size, output_size, group_sizes):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.group_sizes = group_sizes

        self.register_buffer('_group_indices', torch.tensor([0] + list(np.cumsum(group_sizes))))

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

        self.head = nn.Linear(self.HIDDEN_SIZE, output_size)

    def forward(self, x):
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
        elif x.dim() == 4:
            x = x.view(x.size(0), -1)

        features = self.backbone(x)
        logits = self.head(features)

        # Apply grouped softmax
        outputs = []
        for i in range(len(self.group_sizes)):
            start = self._group_indices[i]
            end = self._group_indices[i + 1]
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


def create_model(node_name):
    input_size = get_input_size(node_name)
    output_size = get_node_output_dim(node_name)
    group_sizes = get_node_group_sizes(node_name)
    return TENeighborModel(input_size, output_size, group_sizes)


def load_data(partition_id, batch_size, data_dir=None):
    node_name = partition_to_node(partition_id)

    train_dataset = TENeighborDataset(node_name, split="train", data_dir=data_dir)
    val_dataset = TENeighborDataset(node_name, split="val", data_dir=data_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train(model, trainloader, epochs, lr, device, node_name=None):
    if node_name is None:
        node_name = trainloader.dataset.node_name

    model.to(device)
    model.train()

    criterion = NeighborAwareLoss(node_name, model.group_sizes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    running_loss = 0.0
    num_batches = 0

    for _ in range(epochs):
        for batch in trainloader:
            inputs = batch["input"].to(device)
            demands = batch["demands"].to(device)
            neighbor_loads = batch["neighbor_loads"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, demands, neighbor_loads)
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

    neighbor_criterion = NeighborAwareLoss(node_name, model.group_sizes).to(device)
    kl_criterion = TELoss(model.group_sizes).to(device)

    total_neighbor_loss = 0.0
    total_kl_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in testloader:
            inputs = batch["input"].to(device)
            demands = batch["demands"].to(device)
            neighbor_loads = batch["neighbor_loads"].to(device)
            targets = batch["output"].to(device)

            outputs = model(inputs)

            neighbor_loss = neighbor_criterion(outputs, demands, neighbor_loads)
            kl_loss = kl_criterion(outputs, targets)

            total_neighbor_loss += neighbor_loss.item()
            total_kl_loss += kl_loss.item()
            num_batches += 1

    avg_neighbor_loss = total_neighbor_loss / num_batches if num_batches > 0 else 0.0
    avg_kl_loss = total_kl_loss / num_batches if num_batches > 0 else 0.0

    return avg_neighbor_loss, avg_kl_loss
