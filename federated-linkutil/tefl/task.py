# Use observed link utilization (based on ECMP baseline)

import json
import os
import sys
from collections import defaultdict

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
MAX_OUTGOING_LINKS = 4

BASE_INPUT_SIZE = 2 * HISTORY_LEN * (NUM_NODES - 1) 
INPUT_SIZE = BASE_INPUT_SIZE + MAX_OUTGOING_LINKS

# Default data directory
DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'training_data'
)


def _build_topology_info():
    outgoing_links = {node: [] for node in NODE_LIST}
    link_capacity = {}

    for node_a, node_b, _, capacity in LINKS:
        outgoing_links[node_a].append((node_b, capacity))
        outgoing_links[node_b].append((node_a, capacity))
        link_capacity[(node_a, node_b)] = capacity
        link_capacity[(node_b, node_a)] = capacity

    return outgoing_links, link_capacity


def _load_tunnels():
    tunnels_path = os.path.join(DEFAULT_DATA_DIR, 'tunnels.json')
    with open(tunnels_path, 'r') as f:
        raw = json.load(f)

    tunnels = {}
    for key, paths in raw.items():
        src, dst = key.split("->")
        tunnels[(src, dst)] = [p["path"] for p in paths]
    return tunnels


def _build_tunnel_first_hop_mapping(node_name, tunnels):
    first_hops = []
    destinations = [n for n in NODE_LIST if n != node_name]

    for dst in destinations:
        tunnel_list = tunnels.get((node_name, dst), [])
        for path in tunnel_list:
            first_hop = path[1] if len(path) > 1 else node_name
            first_hops.append(first_hop)

    return first_hops


OUTGOING_LINKS, LINK_CAPACITY = _build_topology_info()
TUNNELS = _load_tunnels()


def compute_ecmp_link_utilizations(demands_dict):
    link_loads = defaultdict(float)

    for (src, dst), demand in demands_dict.items():
        if demand <= 0:
            continue

        paths = TUNNELS.get((src, dst), [])
        if not paths:
            continue

        # Split equally across all available paths
        per_path_traffic = demand / len(paths)

        for path in paths:
            for i in range(len(path) - 1):
                link = (path[i], path[i + 1])
                link_loads[link] += per_path_traffic

    # Convert to utilization
    link_utils = {}
    for link, load in link_loads.items():
        capacity = LINK_CAPACITY.get(link, 1000)
        link_utils[link] = load / capacity

    return link_utils


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


class TELinkUtilDataset(Dataset):
    def __init__(self, node_name, split="train", data_dir=None):
        if data_dir is None:
            data_dir = DEFAULT_DATA_DIR

        self.node_name = node_name
        self.node_idx = NODE_IDX[node_name]
        self.outgoing_links = OUTGOING_LINKS[node_name]
        self.num_links = len(self.outgoing_links)

        # Load base federated data
        fed_data_dir = os.path.join(
            os.path.dirname(__file__), '..', '..', 'data', 'federated_data',
            'federal-2026-02-05-22:31:03'
        )
        node_dir = os.path.join(fed_data_dir, node_name, split)

        self.inputs_incoming = np.load(os.path.join(node_dir, "inputs_incoming.npy"))
        self.inputs_outgoing = np.load(os.path.join(node_dir, "inputs_outgoing.npy"))
        self.outputs = np.load(os.path.join(node_dir, "outputs.npy"))

        # Load raw data to compute ECMP
        raw_data = _load_raw_data(data_dir)

        # Get split indices
        H = HISTORY_LEN
        n_total = len(raw_data)
        train_end = int((n_total - H) * 0.7) + H
        val_end = int((n_total - H) * 0.85) + H

        if split == "train":
            start_idx, end_idx = H, train_end
        elif split == "val":
            start_idx, end_idx = train_end, val_end
        else:
            start_idx, end_idx = val_end, n_total

        # Compute ECMP link util for each sample
        num_samples = end_idx - start_idx
        self.link_utils = np.zeros((num_samples, MAX_OUTGOING_LINKS))

        for sample_idx in range(num_samples):
            epoch_idx = start_idx + sample_idx
            demands = raw_data[epoch_idx]["demands"]

            demands_dict = {}
            for key, value in demands.items():
                src, dst = key.split("->")
                demands_dict[(src, dst)] = value

            # Compute ECMP utilizations
            all_link_utils = compute_ecmp_link_utilizations(demands_dict)

            # Extract outgoing link util
            for link_idx, (neighbor, _) in enumerate(self.outgoing_links):
                link = (node_name, neighbor)
                self.link_utils[sample_idx, link_idx] = all_link_utils.get(link, 0.0)

        self.inputs_base = np.stack([self.inputs_incoming, self.inputs_outgoing], axis=1)
        self.current_demands = self.inputs_outgoing[:, -1, :]
        self.link_capacities = np.array([cap for _, cap in self.outgoing_links])

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, idx):
        base_flat = self.inputs_base[idx].flatten()
        link_utils = self.link_utils[idx]

        combined_input = np.concatenate([base_flat, link_utils])

        return {
            "input": torch.tensor(combined_input, dtype=torch.float32),
            "output": torch.tensor(self.outputs[idx], dtype=torch.float32),
            "demands": torch.tensor(self.current_demands[idx], dtype=torch.float32),
            "link_utils": torch.tensor(link_utils[:self.num_links], dtype=torch.float32),
            "link_capacities": torch.tensor(self.link_capacities, dtype=torch.float32),
        }


class LinkUtilAwareLoss(nn.Module):
    def __init__(self, node_name, group_sizes):
        super().__init__()
        self.node_name = node_name
        self.group_sizes = group_sizes

        self.outgoing_links = OUTGOING_LINKS[node_name]
        self.num_links = len(self.outgoing_links)

        # Build tunnel to link mapping
        first_hops = _build_tunnel_first_hop_mapping(node_name, TUNNELS)
        link_to_idx = {neighbor: i for i, (neighbor, _) in enumerate(self.outgoing_links)}
        tunnel_to_link = [link_to_idx[fh] for fh in first_hops]

        self.register_buffer('tunnel_to_link', torch.tensor(tunnel_to_link, dtype=torch.long))

        capacities = [cap for _, cap in self.outgoing_links]
        self.register_buffer('link_capacities', torch.tensor(capacities, dtype=torch.float32))

        group_indices = [0] + list(np.cumsum(group_sizes))
        self.register_buffer('group_indices', torch.tensor(group_indices, dtype=torch.long))

    def forward(self, pred_ratios, demands, current_link_utils):
        batch_size = pred_ratios.shape[0]

        # Compute traffic this node is adding to the links
        node_link_traffic = torch.zeros(batch_size, self.num_links, device=pred_ratios.device)

        num_destinations = demands.shape[1]
        for dst_idx in range(num_destinations):
            start = self.group_indices[dst_idx]
            end = self.group_indices[dst_idx + 1]
            dst_ratios = pred_ratios[:, start:end]
            dst_demand = demands[:, dst_idx:dst_idx+1]
            tunnel_traffic = dst_demand * dst_ratios

            tunnel_links = self.tunnel_to_link[start:end]
            for t_idx, link_idx in enumerate(tunnel_links):
                node_link_traffic[:, link_idx] += tunnel_traffic[:, t_idx]

        # Nodes contribution to link utilization
        node_link_util = node_link_traffic / (self.link_capacities.unsqueeze(0) + 1e-8)

        # Three part loss: Split evenly across local links with varriance loss, penalize 
        # using high-utilization links and avoid oversubscription
        variance_loss = torch.var(node_link_util, dim=1).mean()
        congestion_penalty = (node_link_util * current_link_utils).sum(dim=1).mean()
        max_util_loss = torch.max(node_link_util, dim=1).values.mean()

        # Combined loss
        # todo: Weighting is chosen randomly. Should be a hyperparameter for proper investigation
        loss = (
            0.3 * variance_loss + 0.5 * congestion_penalty + 0.2 * max_util_loss     
        )

        return loss


class TELoss(nn.Module):
    def __init__(self, group_sizes):
        super().__init__()
        self.group_sizes = group_sizes
        self.register_buffer('_group_indices',
                            torch.tensor([0] + list(np.cumsum(group_sizes))))

    def forward(self, pred, target):
        eps = 1e-8
        total_loss = 0.0

        for i in range(len(self.group_sizes)):
            start = self._group_indices[i]
            end = self._group_indices[i + 1]
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

    train_dataset = TELinkUtilDataset(node_name, split="train", data_dir=data_dir)
    val_dataset = TELinkUtilDataset(node_name, split="val", data_dir=data_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train(model, trainloader, epochs, lr, device, node_name=None):
    if node_name is None:
        node_name = trainloader.dataset.node_name

    model.to(device)
    model.train()

    criterion = LinkUtilAwareLoss(node_name, model.group_sizes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    running_loss = 0.0
    num_batches = 0

    for _ in range(epochs):
        for batch in trainloader:
            inputs = batch["input"].to(device)
            demands = batch["demands"].to(device)
            link_utils = batch["link_utils"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, demands, link_utils)
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

    linkutil_criterion = LinkUtilAwareLoss(node_name, model.group_sizes).to(device)
    kl_criterion = TELoss(model.group_sizes).to(device)

    total_linkutil_loss = 0.0
    total_kl_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in testloader:
            inputs = batch["input"].to(device)
            demands = batch["demands"].to(device)
            link_utils = batch["link_utils"].to(device)
            targets = batch["output"].to(device)

            outputs = model(inputs)

            linkutil_loss = linkutil_criterion(outputs, demands, link_utils)
            kl_loss = kl_criterion(outputs, targets)

            total_linkutil_loss += linkutil_loss.item()
            total_kl_loss += kl_loss.item()
            num_batches += 1

    return (total_linkutil_loss / num_batches if num_batches > 0 else 0.0,
            total_kl_loss / num_batches if num_batches > 0 else 0.0)
