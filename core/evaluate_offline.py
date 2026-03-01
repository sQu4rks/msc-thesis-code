#!/usr/bin/env python3

import argparse
import json
import os
import heapq
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linprog


from topology import LINKS, NODE_LIST
from model import DOTEModel, TEFederatedModel

NODE_IDX = {name: i for i, name in enumerate(NODE_LIST)}
NUM_NODES = len(NODE_LIST)

# link capacity dict from topology
LINK_CAPACITY = {}
for src, dst, delay, cap in LINKS:
    LINK_CAPACITY[(src, dst)] = cap
    LINK_CAPACITY[(dst, src)] = cap

def compute_mlu(demands, ratios, tunnels):
    link_loads = defaultdict(float)

    for (src, dst), paths in tunnels.items():
        demand = demands.get((src, dst), 0)
        if demand <= 0:
            continue

        path_ratios = ratios.get((src, dst), [])
        if not path_ratios:
            # Default to equal splitting
            path_ratios = [1.0 / len(paths)] * len(paths)

        # Ensure ratios match path count
        while len(path_ratios) < len(paths):
            path_ratios.append(0.0)
        path_ratios = path_ratios[:len(paths)]

        # Normalize
        total = sum(path_ratios)
        if total > 0:
            path_ratios = [r / total for r in path_ratios]
        else:
            path_ratios = [1.0 / len(paths)] * len(paths)

        for path, ratio in zip(paths, path_ratios):
            flow = demand * ratio
            for i in range(len(path) - 1):
                link = (path[i], path[i + 1])
                link_loads[link] += flow

    # Compute utilization
    link_utils = {}
    for link, load in link_loads.items():
        cap = LINK_CAPACITY.get(link, 1000)
        link_utils[link] = load / cap

    mlu = max(link_utils.values()) if link_utils else 0.0

    return {
        "mlu": mlu,
        "link_loads": dict(link_loads),
        "link_utils": link_utils,
    }


def compute_ecmp_ratios(tunnels):
    ratios = {}
    for (src, dst), paths in tunnels.items():
        if paths:
            ratios[(src, dst)] = [1.0 / len(paths)] * len(paths)
    return ratios


def load_tunnels(data_dir):
    tunnel_path = os.path.join(data_dir, "tunnels.json")
    with open(tunnel_path) as f:
        raw = json.load(f)

    tunnels = {}
    for key, paths in raw.items():
        src, dst = key.split("->")
        tunnels[(src, dst)] = [p["path"] for p in paths]
    return tunnels


def load_raw_data(data_dir):
    raw_path = os.path.join(data_dir, "raw_data.json")
    with open(raw_path) as f:
        raw = json.load(f)
    return raw


def load_metadata(data_dir):
    meta_path = os.path.join(data_dir, "metadata.json")
    with open(meta_path) as f:
        return json.load(f)

def array_to_ratios(ratio_array, metadata, tunnels):
    tunnels_per_pair = metadata.get("tunnels_per_pair", {})
    sorted_pairs = sorted(tunnels_per_pair.keys())

    ratios = {}
    idx = 0
    for pair_key in sorted_pairs:
        src, dst = pair_key.split("->")
        k = tunnels_per_pair[pair_key]
        pair_ratios = ratio_array[idx:idx + k].tolist()
        ratios[(src, dst)] = pair_ratios
        idx += k

    return ratios


def raw_demands_to_dict(raw_demands):
    dm = {}
    for key, value in raw_demands.items():
        src, dst = key.split("->")
        dm[(src, dst)] = value
    return dm


def raw_ratios_to_dict(raw_ratios):
    ratios = {}
    for key, values in raw_ratios.items():
        src, dst = key.split("->")
        ratios[(src, dst)] = values
    return ratios


# Per-node outputs for federated model
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

FEDERATED_INPUT_SIZE = 2 * 12 * 10  # 2 channels * H * (N-1)
MAX_NEIGHBORS = 4 # todo:maybe parse dict?
MAX_OUTGOING_LINKS = 4
NEIGHBOR_INPUT_SIZE = FEDERATED_INPUT_SIZE + 12 * MAX_NEIGHBORS  # 288
LINKUTIL_INPUT_SIZE = FEDERATED_INPUT_SIZE + MAX_OUTGOING_LINKS  # 244


def load_centralized_model(ckpt_path):

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["model_config"]

    model = DOTEModel(
        input_size=config["input_size"],
        output_size=config["output_size"],
        group_sizes=config["group_sizes"]
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, ckpt.get("metadata", {})


def load_federated_models(fed_dir, input_size=None):
    backbone_path = os.path.join(fed_dir, "final_backbone.pt")
    heads_dir = os.path.join(fed_dir, "heads")

    if not os.path.exists(backbone_path):
        raise FileNotFoundError(f"Backbone not found: {backbone_path}")
    if not os.path.exists(heads_dir):
        raise FileNotFoundError(f"Heads directory not found: {heads_dir}")

    backbone_state = torch.load(backbone_path, map_location="cpu", weights_only=True)

    # check input from backbone because federated models have differing input
    if input_size is None:
        first_layer_key = "backbone.0.weight"
        if first_layer_key in backbone_state:
            input_size = backbone_state[first_layer_key].shape[1]
        else:
            input_size = FEDERATED_INPUT_SIZE

    models = {}
    for node in NODE_LIST:
        head_path = os.path.join(heads_dir, f"{node}_head.pt")
        if not os.path.exists(head_path):
            raise FileNotFoundError(f"Head not found: {head_path}")

        # Create model for this node
        model = TEFederatedModel(
            input_size=input_size,
            output_size=NODE_OUTPUT_DIMS[node],
            group_sizes=NODE_GROUP_SIZES[node]
        )

        # Load backbone
        model.load_backbone_state_dict(backbone_state)

        # Load head
        head_state = torch.load(head_path, map_location="cpu", weights_only=True)
        current_state = model.state_dict()
        for k, v in head_state.items():
            current_state[k] = v
        model.load_state_dict(current_state)
        model.eval()

        models[node] = model

    return models, input_size


def demands_dict_to_matrix(demands_dict):
    matrix = np.zeros((NUM_NODES, NUM_NODES))
    for (src, dst), value in demands_dict.items():
        i = NODE_IDX[src]
        j = NODE_IDX[dst]
        matrix[i, j] = value
    return matrix


def extract_local_view(demands_matrix, node_idx):
    N = demands_matrix.shape[0]

    # Incoming: column node_idx, excluding diagonal
    incoming = np.delete(demands_matrix[:, node_idx], node_idx)

    # Outgoing: row node_idx, excluding diagonal
    outgoing = np.delete(demands_matrix[node_idx, :], node_idx)

    return np.stack([incoming, outgoing], axis=0)  # (2, N-1)


def run_centralized_inference(model, demands_matrix, H=12):
    # Create (H, N, N) input by replicating current demand
    input_tensor = torch.tensor(
        np.tile(demands_matrix, (H, 1, 1)),
        dtype=torch.float32
    ).unsqueeze(0)  # (1, H, N, N)

    with torch.no_grad():
        output = model(input_tensor)

    return output.squeeze(0).numpy()


def _get_neighbors():
    neighbors = {node: [] for node in NODE_LIST}
    for node_a, node_b, _, _ in LINKS:
        neighbors[node_a].append(node_b)
        neighbors[node_b].append(node_a)
    return neighbors

NEIGHBORS = _get_neighbors()


def _get_outgoing_links():
    outgoing = {node: [] for node in NODE_LIST}
    for node_a, node_b, _, cap in LINKS:
        outgoing[node_a].append((node_b, cap))
        outgoing[node_b].append((node_a, cap))
    return outgoing

OUTGOING_LINKS = _get_outgoing_links()


def compute_ecmp_link_utilizations(demands_dict, tunnels):
    link_loads = defaultdict(float)

    for (src, dst), demand in demands_dict.items():
        if demand <= 0:
            continue

        paths = tunnels.get((src, dst), [])
        if not paths:
            continue

        per_path_traffic = demand / len(paths)

        for path in paths:
            for i in range(len(path) - 1):
                link = (path[i], path[i + 1])
                link_loads[link] += per_path_traffic

    link_utils = {}
    for link, load in link_loads.items():
        cap = LINK_CAPACITY.get(link, 1000)
        link_utils[link] = load / cap

    return link_utils


def compute_node_total_loads(demands_matrix):
    return {node: demands_matrix[NODE_IDX[node], :].sum() for node in NODE_LIST}


def run_federated_inference(models, demands_matrix, H=12, include_neighbor_loads=False,
                            include_link_utils=False, tunnels=None):
    all_ratios = {}

    if include_neighbor_loads:
        node_loads = compute_node_total_loads(demands_matrix)

    if include_link_utils:
        demands_dict = {}
        for i, src in enumerate(NODE_LIST):
            for j, dst in enumerate(NODE_LIST):
                if i != j and demands_matrix[i, j] > 0:
                    demands_dict[(src, dst)] = demands_matrix[i, j]
        all_link_utils = compute_ecmp_link_utilizations(demands_dict, tunnels)

    for node_idx, node in enumerate(NODE_LIST):
        model = models[node]

        
        local_view = extract_local_view(demands_matrix, node_idx)

        # Create (2, H, N-1) input by replicating
        local_input = np.tile(local_view[:, np.newaxis, :], (1, H, 1))

        if include_neighbor_loads:
            # Add neighbor loads 
            neighbor_loads = np.zeros((H, MAX_NEIGHBORS))
            for n_idx, neighbor in enumerate(NEIGHBORS[node]):
                neighbor_loads[:, n_idx] = node_loads[neighbor]

            neighbor_loads_mean = np.mean(neighbor_loads[neighbor_loads > 0]) if (neighbor_loads > 0).any() else 0
            neighbor_loads_std = np.std(neighbor_loads[neighbor_loads > 0]) + 1e-8 if (neighbor_loads > 0).any() else 1.0
            neighbor_loads_norm = (neighbor_loads - neighbor_loads_mean) / neighbor_loads_std
            neighbor_loads_norm[neighbor_loads == 0] = 0 

            # Flatten and concatenate
            base_flat = local_input.flatten()
            neighbor_flat = neighbor_loads_norm.flatten()
            combined_input = np.concatenate([base_flat, neighbor_flat])

            input_tensor = torch.tensor(
                combined_input, dtype=torch.float32
            ).unsqueeze(0)  # (1, input_size)

        elif include_link_utils:
            # Add ECMP link util
            link_utils = np.zeros(MAX_OUTGOING_LINKS)
            for link_idx, (neighbor, _) in enumerate(OUTGOING_LINKS[node]):
                link = (node, neighbor)
                link_utils[link_idx] = all_link_utils.get(link, 0.0)

            base_flat = local_input.flatten()
            combined_input = np.concatenate([base_flat, link_utils])

            input_tensor = torch.tensor(
                combined_input, dtype=torch.float32
            ).unsqueeze(0)  # (1, input_size)

        else:
            input_tensor = torch.tensor(
                local_input, dtype=torch.float32
            ).unsqueeze(0)  # (1, 2, H, N-1)

        with torch.no_grad():
            output = model(input_tensor)

        ratios = output.squeeze(0).numpy()

        # Map to (src, dst) pairs for this node
        idx = 0
        group_idx = 0
        for dst in NODE_LIST:
            if dst == node:
                continue
            k = NODE_GROUP_SIZES[node][group_idx]
            all_ratios[(node, dst)] = ratios[idx:idx + k].tolist()
            idx += k
            group_idx += 1

    return all_ratios

def evaluate_strategy(strategy_name, demands_list, ratios_list, tunnels):
    mlus = []
    all_link_utils = []

    for dm, ratios in zip(demands_list, ratios_list):
        result = compute_mlu(dm, ratios, tunnels)
        mlus.append(result["mlu"])
        all_link_utils.append(result["link_utils"])

    mlus = np.array(mlus)

    metrics = {
        "strategy": strategy_name,
        "mean_mlu": float(np.mean(mlus)),
        "median_mlu": float(np.median(mlus)),
        "p90_mlu": float(np.percentile(mlus, 90)),
        "p95_mlu": float(np.percentile(mlus, 95)),
        "p99_mlu": float(np.percentile(mlus, 99)),
        "max_mlu": float(np.max(mlus)),
        "min_mlu": float(np.min(mlus)),
        "std_mlu": float(np.std(mlus)),
        "num_samples": len(mlus),
        "mlus": mlus.tolist(),
    }

    return metrics


def evaluate_all(args):
    # Load topo
    tunnels = load_tunnels(args.data_dir)
    metadata = load_metadata(args.data_dir)
    raw_data = load_raw_data(args.data_dir)
    H = metadata["history_length"]

    print(f"Loaded {len(raw_data)} epochs, H={H}")
    print(f"Tunnels: {len(tunnels)} pairs\n")

    # Extract test set demands and oracle solutions
    n_total = len(raw_data)
    train_end = int((n_total - H) * args.train_ratio) + H
    val_end = int((n_total - H) * (args.train_ratio + args.val_ratio)) + H

    test_epochs = raw_data[val_end:]
    print(f"Test set: {len(test_epochs)} epochs "
          f"(indices {val_end}-{n_total-1})\n")

    # Build demand and ratio lists for test set
    test_demands = []
    oracle_ratios_list = []
    oracle_mlus = []

    for epoch_data in test_epochs:
        dm = raw_demands_to_dict(epoch_data["demands"])
        test_demands.append(dm)

        if epoch_data["oracle_ratios"]:
            ratios = raw_ratios_to_dict(epoch_data["oracle_ratios"])
            oracle_ratios_list.append(ratios)
        else:
            oracle_ratios_list.append(compute_ecmp_ratios(tunnels))

        oracle_mlus.append(epoch_data["oracle_mlu"])

    # Combine results
    results = {}

    # always oracle
    print("Evaluating Oracle")
    oracle_metrics = evaluate_strategy("oracle", test_demands,
                                       oracle_ratios_list, tunnels)
    results["oracle"] = oracle_metrics

    # always ecmp
    print("Evaluating ECMP")
    ecmp_ratios = compute_ecmp_ratios(tunnels)
    ecmp_ratios_list = [ecmp_ratios] * len(test_demands)
    ecmp_metrics = evaluate_strategy("ecmp", test_demands,
                                     ecmp_ratios_list, tunnels)
    results["ecmp"] = ecmp_metrics


    # Now load all configured models with torch and run against data
    if args.centralized_model:
        print(f"Evaluating Centralized model ({args.centralized_model})")
        central_model, central_meta = load_centralized_model(args.centralized_model)
        central_ratios_list = []

        for dm in test_demands:
            dm_matrix = demands_dict_to_matrix(dm)
            ratios_array = run_centralized_inference(central_model, dm_matrix)
            ratios = array_to_ratios(ratios_array, central_meta, tunnels)
            central_ratios_list.append(ratios)

        central_metrics = evaluate_strategy("centralized", test_demands,
                                            central_ratios_list, tunnels)
        results["centralized"] = central_metrics

    # Federated model inference
    if args.federated_dir:
        print(f"Evaluating Federated model ({args.federated_dir})")
        fed_models, fed_input_size = load_federated_models(args.federated_dir)
        fed_ratios_list = []

        for dm in test_demands:
            dm_matrix = demands_dict_to_matrix(dm)
            ratios = run_federated_inference(fed_models, dm_matrix)
            fed_ratios_list.append(ratios)

        fed_metrics = evaluate_strategy("federated", test_demands,
                                        fed_ratios_list, tunnels)
        results["federated"] = fed_metrics

    # Federated model with local loss
    if args.federated_local_dir:
        print(f"Evaluating Federated-Local model ({args.federated_local_dir})")
        fed_local_models, _ = load_federated_models(args.federated_local_dir)
        fed_local_ratios_list = []

        for dm in test_demands:
            dm_matrix = demands_dict_to_matrix(dm)
            ratios = run_federated_inference(fed_local_models, dm_matrix)
            fed_local_ratios_list.append(ratios)

        fed_local_metrics = evaluate_strategy("federated_local", test_demands,
                                              fed_local_ratios_list, tunnels)
        results["federated_local"] = fed_local_metrics

    # Federated model with neighbor feedback
    if args.federated_neighbor_dir:
        print(f"Evaluating Federated-Neighbor model ({args.federated_neighbor_dir})")
        fed_neighbor_models, neighbor_input_size = load_federated_models(args.federated_neighbor_dir)
        fed_neighbor_ratios_list = []

        use_neighbor_loads = neighbor_input_size > FEDERATED_INPUT_SIZE

        for dm in test_demands:
            dm_matrix = demands_dict_to_matrix(dm)
            ratios = run_federated_inference(fed_neighbor_models, dm_matrix,
                                             include_neighbor_loads=use_neighbor_loads)
            fed_neighbor_ratios_list.append(ratios)

        fed_neighbor_metrics = evaluate_strategy("federated_neighbor", test_demands,
                                                 fed_neighbor_ratios_list, tunnels)
        results["federated_neighbor"] = fed_neighbor_metrics

    # Federated model with link utilization feedback
    if args.federated_linkutil_dir:
        print(f"Evaluating Federated-LinkUtil model ({args.federated_linkutil_dir})")
        fed_linkutil_models, linkutil_input_size = load_federated_models(args.federated_linkutil_dir)
        fed_linkutil_ratios_list = []

        use_link_utils = linkutil_input_size == LINKUTIL_INPUT_SIZE

        for dm in test_demands:
            dm_matrix = demands_dict_to_matrix(dm)
            ratios = run_federated_inference(fed_linkutil_models, dm_matrix,
                                             include_link_utils=use_link_utils,
                                             tunnels=tunnels)
            fed_linkutil_ratios_list.append(ratios)

        fed_linkutil_metrics = evaluate_strategy("federated_linkutil", test_demands,
                                                  fed_linkutil_ratios_list, tunnels)
        results["federated_linkutil"] = fed_linkutil_metrics

    # Print resultsa
    print(f"{'Strategy':>12} {'Mean MLU':>10} {'P90 MLU':>10} {'P99 MLU':>10} "
          f"{'Max MLU':>10} {'vs Oracle':>10}")

    oracle_mean = results["oracle"]["mean_mlu"]
    for name, metrics in results.items():
        ratio = metrics["mean_mlu"] / oracle_mean if oracle_mean > 0 else 0
        print(f"{name:>12} {metrics['mean_mlu']:10.4f} "
              f"{metrics['p90_mlu']:10.4f} {metrics['p99_mlu']:10.4f} "
              f"{metrics['max_mlu']:10.4f} {ratio:10.2f}x")

    # Per epoch
    if "model" in results:
        model_mlus = np.array(results["model"]["mlus"])
        oracle_mlu_arr = np.array(results["oracle"]["mlus"][:len(model_mlus)])
        ecmp_mlus = np.array(results["ecmp"]["mlus"][:len(model_mlus)])

        # vs ecmp
        improvement = (ecmp_mlus - model_mlus) / ecmp_mlus * 100
        valid = ecmp_mlus > 0

        if valid.any():
            print(f"\nModel vs ECMP improvement:")
            print(f"  Mean:   {np.mean(improvement[valid]):+.1f}%")
            print(f"  Median: {np.median(improvement[valid]):+.1f}%")
            print(f"  Min:    {np.min(improvement[valid]):+.1f}%")

        # vs oracle
        gap = model_mlus / oracle_mlu_arr
        valid_gap = oracle_mlu_arr > 0

        if valid_gap.any():
            print(f"\nModel optimality ratio (model_mlu / oracle_mlu):")
            print(f"  Mean:   {np.mean(gap[valid_gap]):.3f}x")
            print(f"  Median: {np.median(gap[valid_gap]):.3f}x")
            print(f"  P95:    {np.percentile(gap[valid_gap], 95):.3f}x")

    # Save results
    output_path = os.path.join(args.data_dir, "evaluation_results.json")

    # Convert numpy arrays for JSON serialization
    save_results = {}
    for name, metrics in results.items():
        save_results[name] = {k: v for k, v in metrics.items() if k != "mlus"}
        save_results[name]["mlus"] = [round(m, 6) for m in metrics["mlus"]]

    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Eval script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--data-dir", type=str, required=True,
        help="data directory (with tunnels.json, raw_data.json)")
    # train and val ratio must match training script config
    parser.add_argument("--train-ratio", type=float, default=0.7,
        help="Training set ratio") 
    parser.add_argument("--val-ratio", type=float, default=0.15,
        help="Validation set ratio")
    
    parser.add_argument("--centralized-model", type=str, default=None,
        help="centralized model checkpoint")
    parser.add_argument("--federated-dir", type=str, default=None,
        help="federated model checkpoint dir")
    parser.add_argument("--federated-local-dir", type=str, default=None,
        help="federated-local model directory ")
    parser.add_argument("--federated-neighbor-dir", type=str, default=None,
        help="federated-neighbor model directory")
    parser.add_argument("--federated-linkutil-dir", type=str, default=None,
        help="federated-linkutil model directory")

    args = parser.parse_args()
    evaluate_all(args)
