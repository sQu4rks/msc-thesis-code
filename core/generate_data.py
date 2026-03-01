#!/usr/bin/env python3

import argparse
import json
import math
import os
import random
import heapq
from collections import defaultdict
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.optimize import linprog

from topology import LINKS, NODES, NODE_WEIGHTS

NODE_LIST = list(NODES.keys())
NODE_IDX = {name: i for i, name in enumerate(NODE_LIST)}
NUM_NODES = len(NODE_LIST)

class NetworkGraph:

    def __init__(self):
        self.adj = defaultdict(list)
        self.link_capacity = {}
        self.link_delay = {}

        for src, dst, delay, cap in LINKS:
            self.adj[src].append((dst, delay))
            self.adj[dst].append((src, delay))
            self.link_capacity[(src, dst)] = cap
            self.link_capacity[(dst, src)] = cap
            self.link_delay[(src, dst)] = delay
            self.link_delay[(dst, src)] = delay

    def k_shortest_paths(self, src, dst, k=4):
        # Based on pseudo code in https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm and https://gist.github.com/kachayev/5990802
        def dijkstra(source, target, excluded_edges=set(), excluded_nodes=set()):
            dist = {source: 0}
            prev = {source: None}
            pq = [(0, source)]

            while pq:
                d, u = heapq.heappop(pq)
                if u == target:
                    path = []
                    while u is not None:
                        path.append(u)
                        u = prev[u]
                    return list(reversed(path)), d
                if d > dist.get(u, float('inf')):
                    continue
                for v, w in self.adj[u]:
                    if v in excluded_nodes and v != target:
                        continue
                    if (u, v) in excluded_edges:
                        continue
                    new_d = d + w
                    if new_d < dist.get(v, float('inf')):
                        dist[v] = new_d
                        prev[v] = u
                        heapq.heappush(pq, (new_d, v))
            return None, float('inf')

        shortest, cost = dijkstra(src, dst)
        if shortest is None:
            return []

        A = [(cost, shortest)]
        B = []

        for i in range(1, k):
            if i - 1 >= len(A):
                break
            prev_path = A[i - 1][1]

            for j in range(len(prev_path) - 1):
                spur_node = prev_path[j]
                root_path = prev_path[:j + 1]

                excluded_edges = set()
                for _, p in A:
                    if p[:j + 1] == root_path and j + 1 < len(p):
                        excluded_edges.add((p[j], p[j + 1]))

                excluded_nodes = set(root_path[:-1])

                spur_path, spur_cost = dijkstra(
                    spur_node, dst, excluded_edges, excluded_nodes
                )
                if spur_path is not None:
                    total_path = root_path[:-1] + spur_path
                    total_cost = sum(
                        self.link_delay.get((total_path[x], total_path[x+1]), 0)
                        for x in range(len(total_path) - 1)
                    )
                    candidate = (total_cost, total_path)
                    if candidate not in B:
                        heapq.heappush(B, candidate)

            if not B:
                break
            best = heapq.heappop(B)
            A.append(best)

        return [path for _, path in A]

    def compute_all_tunnels(self, k=4):
        tunnels = {}
        for src in NODE_LIST:
            for dst in NODE_LIST:
                if src == dst:
                    continue
                paths = self.k_shortest_paths(src, dst, k)
                tunnels[(src, dst)] = paths
        return tunnels

class DemandGenerator:

    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    def generate(self, hour, total_traffic=100.0, noise=0.15):
        weights = [NODE_WEIGHTS[n] for n in NODE_LIST]
        total_w = sum(weights)

        # peak around noon, lowest at midnight 
        # todo: make configurable for diff patterns?
        factor = 0.5 + 0.5 * math.sin(math.pi * (hour - 6) / 12)
        factor = max(0.2, factor)

        dm = {}
        for i, src in enumerate(NODE_LIST):
            for j, dst in enumerate(NODE_LIST):
                if i == j:
                    continue
                base = total_traffic * weights[i] * weights[j] / (total_w ** 2)
                noised = base * factor * (1 + self.rng.uniform(-noise, noise))
                dm[(src, dst)] = max(0, noised)
        return dm

    def generate_sequence(self, num_epochs, epoch_minutes=5, total_traffic=100.0,
                          noise=0.15, start_hour=0.0):
        demands = []
        for epoch in range(num_epochs):
            hour = (start_hour + epoch * epoch_minutes / 60) % 24
            dm = self.generate(hour=hour, total_traffic=total_traffic, noise=noise)
            demands.append(dm)
        return demands

    def to_matrix(self, dm):
        matrix = np.zeros((NUM_NODES, NUM_NODES))
        for (src, dst), value in dm.items():
            i = NODE_IDX[src]
            j = NODE_IDX[dst]
            matrix[i, j] = value
        return matrix

    def from_matrix(self, matrix):
        dm = {}
        for i, src in enumerate(NODE_LIST):
            for j, dst in enumerate(NODE_LIST):
                if i != j and matrix[i, j] > 0:
                    dm[(src, dst)] = matrix[i, j]
        return dm

class OmniscientOracle:

    def __init__(self, graph, tunnels):
        self.graph = graph
        self.tunnels = tunnels
        self.setup_lp_structure()

    def setup_lp_structure(self):
        self.tunnel_list = []
        for (src, dst), paths in self.tunnels.items():
            for idx, path in enumerate(paths):
                self.tunnel_list.append((src, dst, idx, path))

        self.n_tunnels = len(self.tunnel_list)
        self.n_vars = self.n_tunnels + 1  # tunnels + MLU variable
        self.z_idx = self.n_tunnels

        # Group tunnels by (src, dst) pair
        self.pairs = {}
        for t_idx, (src, dst, p_idx, path) in enumerate(self.tunnel_list):
            key = (src, dst)
            if key not in self.pairs:
                self.pairs[key] = []
            self.pairs[key].append(t_idx)

        # Pre-compute which tunnels use which edges
        self.tunnel_edges = []
        for t_idx, (src, dst, p_idx, path) in enumerate(self.tunnel_list):
            edges = set()
            for h in range(len(path) - 1):
                edges.add((path[h], path[h+1]))
            self.tunnel_edges.append(edges)

        # All directed edges
        self.all_edges = []
        for src, dst, delay, cap in LINKS:
            self.all_edges.append((src, dst, cap))
            self.all_edges.append((dst, src, cap))

    def solve(self, demand_matrix):
        # Objective: minimize MLU - represented by variable z
        c = np.zeros(self.n_vars)
        c[self.z_idx] = 1.0

        # Equality constraints: sum of ratios = 1 for each (src, dst)
        A_eq = []
        b_eq = []
        for (src, dst), t_indices in self.pairs.items():
            row = np.zeros(self.n_vars)
            for t_idx in t_indices:
                row[t_idx] = 1.0
            A_eq.append(row)
            b_eq.append(1.0)

        # Inequality constraints: link capacity
        A_ub = []
        b_ub = []
        for edge_src, edge_dst, cap in self.all_edges:
            row = np.zeros(self.n_vars)
            row[self.z_idx] = -cap  # -z * capacity

            for t_idx, (src, dst, p_idx, path) in enumerate(self.tunnel_list):
                demand = demand_matrix.get((src, dst), 0)
                if demand <= 0:
                    continue
                if (edge_src, edge_dst) in self.tunnel_edges[t_idx]:
                    row[t_idx] = demand

            A_ub.append(row)
            b_ub.append(0.0)

        # Bounds: all variables >= 0
        bounds = [(0, None)] * self.n_vars

        try:
            result = linprog(
                c,
                A_ub=np.array(A_ub) if A_ub else None,
                b_ub=np.array(b_ub) if b_ub else None,
                A_eq=np.array(A_eq) if A_eq else None,
                b_eq=np.array(b_eq) if b_eq else None,
                bounds=bounds,
                method='highs'
            )

            if result.success:
                opt_mlu = result.x[self.z_idx]
                ratios = {}
                for (src, dst), t_indices in self.pairs.items():
                    ratios[(src, dst)] = [result.x[t_idx] for t_idx in t_indices]
                return opt_mlu, ratios
            else:
                return None, None

        except Exception as e:
            print(f"solve error: {e}")
            return None, None

class TrainingDataFormatter:

    def __init__(self, tunnels, history_length=12):
        self.tunnels = tunnels
        self.history_length = history_length

        # Create ordered list of tunnel indices for consistent output format
        self.tunnel_pairs = sorted(tunnels.keys())
        self.num_tunnels_per_pair = {k: len(v) for k, v in tunnels.items()}

    def format_centralized(self, demand_history, optimal_ratios):
        # Input: stack of demand matrices
        H = len(demand_history)
        input_tensor = np.zeros((H, NUM_NODES, NUM_NODES))
        for t, dm in enumerate(demand_history):
            for (src, dst), value in dm.items():
                i = NODE_IDX[src]
                j = NODE_IDX[dst]
                input_tensor[t, i, j] = value

        # Output: flattened splitting ratios
        output_list = []
        for pair in self.tunnel_pairs:
            ratios = optimal_ratios.get(pair, [])
            k = self.num_tunnels_per_pair[pair]
            # Pad with zeros if needed
            padded = list(ratios) + [0.0] * (k - len(ratios))
            output_list.extend(padded[:k])

        output_tensor = np.array(output_list)

        return {
            "input": input_tensor,
            "output": output_tensor,
            "metadata": {
                "history_length": H,
                "num_nodes": NUM_NODES,
                "num_pairs": len(self.tunnel_pairs),
            }
        }

    def format_federated(self, demand_history, optimal_ratios, node_name):
        node_idx = NODE_IDX[node_name]
        other_nodes = [n for n in NODE_LIST if n != node_name]
        H = len(demand_history)

        # Input: local view of demands
        outgoing = np.zeros((H, NUM_NODES - 1))  # Traffic from one node to other nodes
        incoming = np.zeros((H, NUM_NODES - 1))  # Traffic from other nodes to this node

        for t, dm in enumerate(demand_history):
            for j, other in enumerate(other_nodes):
                outgoing[t, j] = dm.get((node_name, other), 0)
                incoming[t, j] = dm.get((other, node_name), 0)

        total_outgoing = outgoing.sum(axis=1)
        total_incoming = incoming.sum(axis=1)

        # Output: splitting ratios for flows originating from this node
        output_list = []
        local_pairs = [(node_name, dst) for dst in other_nodes]
        for pair in local_pairs:
            ratios = optimal_ratios.get(pair, [])
            k = self.num_tunnels_per_pair.get(pair, 0)
            if k > 0:
                padded = list(ratios) + [0.0] * (k - len(ratios))
                output_list.extend(padded[:k])

        output_tensor = np.array(output_list) if output_list else np.array([])

        return {
            "input": {
                "outgoing": outgoing,
                "incoming": incoming,
                "total_outgoing": total_outgoing,
                "total_incoming": total_incoming,
            },
            "output": output_tensor,
            "metadata": {
                "node": node_name,
                "node_idx": node_idx,
                "history_length": H,
                "num_destinations": len(other_nodes),
                "other_nodes": other_nodes,
            }
        }

def generate_training_data(args):
    # Init graph, tunnels, oracle and demand generator and exporter for training data
    graph = NetworkGraph()
    tunnels = graph.compute_all_tunnels(k=args.k_paths)
    oracle = OmniscientOracle(graph, tunnels)
    demand_gen = DemandGenerator(seed=args.seed)
    formatter = TrainingDataFormatter(tunnels, history_length=args.history_length)

    # Generate demand sequence
    print(f"Generating {args.epochs} demand matrices")
    demands = demand_gen.generate_sequence(
        num_epochs=args.epochs,
        epoch_minutes=5,
        total_traffic=args.total_traffic,
        noise=args.noise,
        start_hour=args.start_hour
    )

    # Compute oracle solutions
    print("Computing oracle solutions")
    oracle_solutions = []
    for i, dm in enumerate(demands):
        mlu, ratios = oracle.solve(dm)
        oracle_solutions.append({
            "mlu": mlu,
            "ratios": ratios
        })
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{args.epochs} epochs")

    # Create training samples
    print("\nFormatting training data")
    H = args.history_length

    os.makedirs(args.output_dir, exist_ok=True)

    if args.federated:
        # separate dataset per node
        node_datasets = {node: [] for node in NODE_LIST}

        for t in range(H, len(demands)):
            demand_history = demands[t - H:t]
            current_ratios = oracle_solutions[t]["ratios"]

            if current_ratios is None:
                continue

            for node in NODE_LIST:
                sample = formatter.format_federated(
                    demand_history, current_ratios, node
                )
                sample["epoch"] = t
                sample["oracle_mlu"] = oracle_solutions[t]["mlu"]
                node_datasets[node].append(sample)

        # Save per-node datasets
        for node, samples in node_datasets.items():
            node_dir = os.path.join(args.output_dir, node)
            os.makedirs(node_dir, exist_ok=True)

            # Save as numpy arrays
            if samples:
                inputs_outgoing = np.stack([s["input"]["outgoing"] for s in samples])
                inputs_incoming = np.stack([s["input"]["incoming"] for s in samples])
                outputs = np.stack([s["output"] for s in samples])
                mlus = np.array([s["oracle_mlu"] for s in samples])

                np.save(os.path.join(node_dir, "inputs_outgoing.npy"), inputs_outgoing)
                np.save(os.path.join(node_dir, "inputs_incoming.npy"), inputs_incoming)
                np.save(os.path.join(node_dir, "outputs.npy"), outputs)
                np.save(os.path.join(node_dir, "oracle_mlus.npy"), mlus)

                with open(os.path.join(node_dir, "metadata.json"), "w") as f:
                    json.dump(samples[0]["metadata"], f, indent=2)

            print(f"  {node}: {len(samples)} samples")

    else:
        # single dataset
        samples = []

        for t in range(H, len(demands)):
            demand_history = demands[t - H:t]
            current_ratios = oracle_solutions[t]["ratios"]

            if current_ratios is None:
                continue

            sample = formatter.format_centralized(demand_history, current_ratios)
            sample["epoch"] = t
            sample["oracle_mlu"] = oracle_solutions[t]["mlu"]
            samples.append(sample)

        inputs = np.stack([s["input"] for s in samples])
        outputs = np.stack([s["output"] for s in samples])
        mlus = np.array([s["oracle_mlu"] for s in samples])

        np.save(os.path.join(args.output_dir, "inputs.npy"), inputs)
        np.save(os.path.join(args.output_dir, "outputs.npy"), outputs)
        np.save(os.path.join(args.output_dir, "oracle_mlus.npy"), mlus)

        metadata = {
            "num_samples": len(samples),
            "input_shape": list(inputs.shape),
            "output_shape": list(outputs.shape),
            "history_length": H,
            "k_paths": args.k_paths,
            "num_nodes": NUM_NODES,
            "node_list": NODE_LIST,
            "tunnel_pairs": [(s, d) for s, d in formatter.tunnel_pairs],
            "tunnels_per_pair": {f"{s}->{d}": k for (s, d), k in formatter.num_tunnels_per_pair.items()},
        }
        with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    # Save tunnel definitions for loading in heads on federated models
    tunnel_info = {}
    for (src, dst), paths in tunnels.items():
        key = f"{src}->{dst}"
        tunnel_info[key] = [
            {"path": path, "hops": len(path) - 1}
            for path in paths
        ]
    with open(os.path.join(args.output_dir, "tunnels.json"), "w") as f:
        json.dump(tunnel_info, f, indent=2)

    # Save raw data as well
    if args.save_raw:
        raw_data = []
        for t, (dm, sol) in enumerate(zip(demands, oracle_solutions)):
            hour = (args.start_hour + t * 5 / 60) % 24
            raw_data.append({
                "epoch": t,
                "hour": round(hour, 2),
                "demands": {f"{s}->{d}": round(v, 4) for (s, d), v in dm.items()},
                "oracle_mlu": round(sol["mlu"], 6) if sol["mlu"] else None,
                "oracle_ratios": {
                    f"{s}->{d}": [round(r, 6) for r in ratios]
                    for (s, d), ratios in (sol["ratios"] or {}).items()
                }
            })
        with open(os.path.join(args.output_dir, "raw_data.json"), "w") as f:
            json.dump(raw_data, f, indent=2)

    print(f"\nData saved to {args.output_dir}/")

def create_splits(args):

    data_dir = args.output_dir

    if args.federated:
        # Split each node's data separately
        for node in NODE_LIST:
            node_dir = os.path.join(data_dir, node)
            if not os.path.exists(node_dir):
                print(f"No data dir for {node}")
                continue

            inputs_out = np.load(os.path.join(node_dir, "inputs_outgoing.npy"))
            inputs_in = np.load(os.path.join(node_dir, "inputs_incoming.npy"))
            outputs = np.load(os.path.join(node_dir, "outputs.npy"))
            mlus = np.load(os.path.join(node_dir, "oracle_mlus.npy"))

            n = len(outputs)
            train_end = int(n * args.train_ratio)
            val_end = int(n * (args.train_ratio + args.val_ratio))

            # split in order for time dependnc
            splits = {
                "train": (0, train_end),
                "val": (train_end, val_end),
                "test": (val_end, n)
            }

            for split_name, (start, end) in splits.items():
                split_dir = os.path.join(node_dir, split_name)
                os.makedirs(split_dir, exist_ok=True)

                np.save(os.path.join(split_dir, "inputs_outgoing.npy"), inputs_out[start:end])
                np.save(os.path.join(split_dir, "inputs_incoming.npy"), inputs_in[start:end])
                np.save(os.path.join(split_dir, "outputs.npy"), outputs[start:end])
                np.save(os.path.join(split_dir, "oracle_mlus.npy"), mlus[start:end])

            print(f"{node}: train={train_end}, val={val_end-train_end}, test={n-val_end}")
    else:
        # all in one data file for full model
        inputs = np.load(os.path.join(data_dir, "inputs.npy"))
        outputs = np.load(os.path.join(data_dir, "outputs.npy"))
        mlus = np.load(os.path.join(data_dir, "oracle_mlus.npy"))

        n = len(outputs)
        train_end = int(n * args.train_ratio)
        val_end = int(n * (args.train_ratio + args.val_ratio))

        splits = {
            "train": (0, train_end),
            "val": (train_end, val_end),
            "test": (val_end, n)
        }

        for split_name, (start, end) in splits.items():
            split_dir = os.path.join(data_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)

            np.save(os.path.join(split_dir, "inputs.npy"), inputs[start:end])
            np.save(os.path.join(split_dir, "outputs.npy"), outputs[start:end])
            np.save(os.path.join(split_dir, "oracle_mlus.npy"), mlus[start:end])

        print(f"Splits: train={train_end}, val={val_end-train_end}, test={n-val_end}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data generator"
    )

    parser.add_argument("--epochs", type=int, default=288,
        help="Number of epochs to generate (default: 288 = 1 day)")
    parser.add_argument("--history-length", type=int, default=12,
        help="Number of historical epochs for input (default: 12 = 1 hour)")
    parser.add_argument("--k-paths", type=int, default=4,
        help="Number of shortest paths per pair (default: 4)")
    parser.add_argument("--total-traffic", type=float, default=100.0,
        help="Base traffic level in Mbps (default: 100)")
    parser.add_argument("--noise", type=float, default=0.15,
        help="Demand noise factor (default: 0.15)")
    parser.add_argument("--start-hour", type=float, default=0.0,
        help="Starting hour of day (default: 0.0)")
    parser.add_argument("--seed", type=int, default=42,
        help="Random seed (default: 42)")
    parser.add_argument("--output-dir", type=str, default="./training_data",
        help="Output directory (default: ./training_data)")
    parser.add_argument("--federated", action="store_true",
        help="Generate federated (per-node) format")
    parser.add_argument("--save-raw", action="store_true",
        help="Save raw demands and oracle solutions as JSON")
    parser.add_argument("--split", action="store_true",
        help="Create train/val/test splits after generation")
    parser.add_argument("--train-ratio", type=float, default=0.7,
        help="Training set ratio (default: 0.7)")
    parser.add_argument("--val-ratio", type=float, default=0.15,
        help="Validation set ratio (default: 0.15)")

    args = parser.parse_args()
    output_type_prefix = "central"

    if args.federated:
        output_type_prefix = "federal"

    timestamp = datetime.strftime(datetime.now(), "%Y-%m-%d-%H:%M:%S")
    prefix = f"{output_type_prefix}-{timestamp}"
    args.output_dir = os.path.join(args.output_dir, prefix)

    generate_training_data(args)

    if args.split:
        print("\nCreating train/val/test splits...")
        create_splits(args)
