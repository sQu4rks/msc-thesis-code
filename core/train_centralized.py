#!/usr/bin/env python3

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import DOTEModel

class TEDataset(Dataset):

    def __init__(self, data_dir, split=None):
        load_dir = os.path.join(data_dir, split) if split else data_dir

        self.inputs = np.load(os.path.join(load_dir, "inputs.npy"))
        self.outputs = np.load(os.path.join(load_dir, "outputs.npy"))
        self.oracle_mlus = np.load(os.path.join(load_dir, "oracle_mlus.npy"))

        # Load metadata from parent directory
        meta_path = os.path.join(data_dir, "metadata.json")
        with open(meta_path) as f:
            self.metadata = json.load(f)

        print(f"Loaded {len(self.inputs)} samples from {load_dir}")
        print(f"  Input shape:  {self.inputs.shape}")
        print(f"  Output shape: {self.outputs.shape}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.inputs[idx])
        y = torch.FloatTensor(self.outputs[idx])
        mlu = torch.FloatTensor([self.oracle_mlus[idx]])
        return x, y, mlu

class TELoss(nn.Module):

    def __init__(self, mlu_weight=0.0):
        super().__init__()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.mlu_weight = mlu_weight

    def forward(self, predicted, target, oracle_mlu=None):
        # KL divergence: target * log(target / predicted)
        # pytorch KLDivLoss expects log-probabilities as input
        log_pred = torch.log(predicted + 1e-8)
        loss = self.kl_loss(log_pred, target + 1e-8)

        return loss


class MSELoss(nn.Module):

    def forward(self, predicted, target, oracle_mlu=None):
        return torch.mean((predicted - target) ** 2)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n_batches = 0

    for x, y, mlu in loader:
        x, y, mlu = x.to(device), y.to(device), mlu.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y, mlu)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for x, y, mlu in loader:
            x, y, mlu = x.to(device), y.to(device), mlu.to(device)
            pred = model(x)
            loss = criterion(pred, y, mlu)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def train(args):
    device = torch.device("cpu") # "cuda" for gpu but right now cpu seems to be enough
    print(f"Using device: {device}\n")

    # Load data
    train_dataset = TEDataset(args.data_dir, split="train")
    val_dataset = TEDataset(args.data_dir, split="val")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # Model dimensions
    metadata = train_dataset.metadata
    H = metadata["history_length"]
    N = metadata["num_nodes"]
    input_size = H * N * N
    output_size = train_dataset.outputs.shape[1]

    # Get group sizes for grouped softmax
    tunnels_per_pair = metadata.get("tunnels_per_pair", {})
    if tunnels_per_pair:
        # Sort pairs consistently with data generation
        sorted_pairs = sorted(tunnels_per_pair.keys())
        group_sizes = [tunnels_per_pair[p] for p in sorted_pairs]
    else:
        group_sizes = None

    # Create model
    model = DOTEModel(
        input_size=input_size,
        output_size=output_size,
        group_sizes=group_sizes,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    if args.loss == "kl":
        criterion = TELoss()
    else:
        criterion = MSELoss()

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} {'LR':>10} {'Time':>8}")
    print("-" * 52)

    for epoch in range(args.epochs):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch+1:6d} {train_loss:12.6f} {val_loss:12.6f} "
              f"{current_lr:10.6f} {dt:7.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "metadata": metadata,
                "model_config": {
                    "input_size": input_size,
                    "output_size": output_size,
                    "group_sizes": group_sizes,
                }
            }, os.path.join(args.checkpoint_dir, "model_best.pt"))
        else:
            # Abort training if model is not improving after 20 episodes
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"stopping early")
                break

        # Periodic checkpoint in case something aborts
        if (epoch + 1) % 50 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
            }, os.path.join(args.checkpoint_dir, f"model_epoch_{epoch+1}.pt"))

    # Save training history
    with open(os.path.join(args.checkpoint_dir, "training_history.json"), "w") as f:
        json.dump(history, f)

    return model


def evaluate(args):
    device = torch.device("cpu")

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    metadata = checkpoint["metadata"]

    # Reconstruct model with same architecture as training
    model = DOTEModel(
        input_size=config["input_size"],
        output_size=config["output_size"],
        group_sizes=config["group_sizes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load test data
    test_dataset = TEDataset(args.data_dir, split="test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)

    # Run inference
    all_preds = []
    all_targets = []
    all_mlus = []

    with torch.no_grad():
        for x, y, mlu in test_loader:
            x = x.to(device)
            pred = model(x)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.numpy())
            all_mlus.append(mlu.numpy())

    predictions = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    oracle_mlus = np.concatenate(all_mlus, axis=0).flatten()

    # save as structured output for evaluate_offline.py
    export_dir = os.path.join(args.data_dir, "predictions")
    os.makedirs(export_dir, exist_ok=True)
    np.save(os.path.join(export_dir, "predicted_ratios.npy"), predictions)
    np.save(os.path.join(export_dir, "oracle_ratios.npy"), targets)
    np.save(os.path.join(export_dir, "oracle_mlus.npy"), oracle_mlus)
    print(f"predictions saved to {export_dir}/")

    return predictions, targets, oracle_mlus

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Centralized trainig"
    )

    parser.add_argument("--evaluate", action="store_true",
        help="evaluate instead of train. default is train")
    parser.add_argument("--data-dir", type=str, required=True,
        help="Path to training data")
    parser.add_argument("--epochs", type=int, default=100,
        help="training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
        help="batch size")
    parser.add_argument("--lr", type=float, default=0.001,
        help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
        help="weight decay")
    parser.add_argument("--patience", type=int, default=20,
        help="stop if no better loss is achieved after x epochs")
    parser.add_argument("--loss", choices=["kl", "mse"], default="kl",
        help="loss function")

    # Checkpoints
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
        help="Checkpoint directory")
    parser.add_argument("--checkpoint", type=str, default=None,
        help="Path to specific checkpoint for eval. Otherwise use best model from training run")

    args = parser.parse_args()

    if args.evaluate:
        if args.checkpoint is None:
            args.checkpoint = os.path.join(args.checkpoint_dir, "model_best.pt")
        evaluate(args)
    else:
        train(args)
