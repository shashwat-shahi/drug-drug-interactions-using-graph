"""
Main training script for DDI prediction using Graph Neural Networks with PyTorch.

Based on Chapter 4: Understanding Drug–Drug Interactions Using Graphs
from "Deep Learning for the Life Sciences"
"""

import torch
import torch.optim as optim
import numpy as np
import argparse
import os
from pathlib import Path

from dataset import DatasetBuilder, optimal_batch_size
from model import DdiModel
from training import train, auc_loss, binary_log_loss, TrainingMetrics


def setup_device():
    """Setup device (GPU if available, else CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def load_dataset(path, node_limit=500, seed=42):
    """Load and prepare dataset."""
    print(f"Loading dataset from {path}...")
    builder = DatasetBuilder(path=path)
    dataset_splits = builder.build(node_limit=node_limit, rng_seed=seed)

    print(f"Dataset loaded:")
    for split_name, dataset in dataset_splits.items():
        n_pos = dataset.pairs.pos.shape[0]
        n_neg = dataset.pairs.neg.shape[0]
        print(f"  {split_name}: {dataset.n_nodes} nodes, {n_pos} pos pairs, {n_neg} neg pairs")

    return dataset_splits


def create_model(dataset_splits, embedding_dim=128, dropout_rate=0.3,
                 last_layer_self=False, degree_norm=False, n_mlp_layers=2):
    """Create the DDI model."""
    n_nodes = dataset_splits["train"].n_nodes
    model = DdiModel(
        n_nodes=n_nodes,
        embedding_dim=embedding_dim,
        dropout_rate=dropout_rate,
        last_layer_self=last_layer_self,
        degree_norm=degree_norm,
        n_mlp_layers=n_mlp_layers,
    )
    return model


def train_simple_model(dataset_splits, device, save_dir="./models"):
    """Train the simplest model from the paper."""
    print("\n" + "="*50)
    print("Training Simplest Model (Binary Log Loss)")
    print("="*50)

    # Create model
    model = create_model(
        dataset_splits,
        embedding_dim=128,
        dropout_rate=0.3,
        last_layer_self=False,
        degree_norm=False,
        n_mlp_layers=2,
    ).to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Optimal batch size
    batch_size = optimal_batch_size(dataset_splits)
    print(f"Optimal batch size: {batch_size}")

    # Training parameters
    save_path = os.path.join(save_dir, "simple_model.pt")
    os.makedirs(save_dir, exist_ok=True)

    # Train
    metrics, best_val_hits = train(
        model=model,
        optimizer=optimizer,
        device=device,
        dataset_splits=dataset_splits,
        batch_size=batch_size,
        num_epochs=500,
        loss_fn=binary_log_loss,
        norm_loss=False,
        eval_every=1,
        save_path=save_path,
    )

    print(f"\nBest validation Hits@20: {best_val_hits:.4f}")
    return model, metrics


def train_auc_model(dataset_splits, device, save_dir="./models"):
    """Train model with AUC loss."""
    print("\n" + "="*50)
    print("Training AUC-Optimized Model")
    print("="*50)

    # Create model
    model = create_model(
        dataset_splits,
        embedding_dim=128,
        dropout_rate=0.3,
        last_layer_self=False,
        degree_norm=False,
        n_mlp_layers=2,
    ).to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Optimal batch size
    batch_size = optimal_batch_size(dataset_splits)
    print(f"Optimal batch size: {batch_size}")

    # Training parameters
    save_path = os.path.join(save_dir, "auc_model.pt")
    os.makedirs(save_dir, exist_ok=True)

    # Train with AUC loss
    metrics, best_val_hits = train(
        model=model,
        optimizer=optimizer,
        device=device,
        dataset_splits=dataset_splits,
        batch_size=batch_size,
        num_epochs=500,
        loss_fn=auc_loss,
        norm_loss=True,
        eval_every=1,
        save_path=save_path,
    )

    print(f"\nBest validation Hits@20: {best_val_hits:.4f}")
    return model, metrics


def train_large_scale_model(dataset_splits, device, save_dir="./models"):
    """Train model on larger dataset."""
    print("\n" + "="*50)
    print("Training Large-Scale Model")
    print("="*50)

    # Create model with optimized hyperparameters
    model = create_model(
        dataset_splits,
        embedding_dim=512,
        dropout_rate=0.3,
        last_layer_self=True,
        degree_norm=True,
        n_mlp_layers=2,
    ).to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Optimal batch size
    batch_size = optimal_batch_size(dataset_splits)
    print(f"Optimal batch size: {batch_size}")

    # Training parameters
    save_path = os.path.join(save_dir, "large_scale_model.pt")
    os.makedirs(save_dir, exist_ok=True)

    # Train
    metrics, best_val_hits = train(
        model=model,
        optimizer=optimizer,
        device=device,
        dataset_splits=dataset_splits,
        batch_size=batch_size,
        num_epochs=1000,
        loss_fn=auc_loss,
        norm_loss=True,
        eval_every=25,
        save_path=save_path,
    )

    print(f"\nBest validation Hits@20: {best_val_hits:.4f}")
    return model, metrics


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train DDI prediction model using Graph Neural Networks"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--node-limit",
        type=int,
        default=500,
        help="Number of nodes for small-scale training (default: 500 for 10% of data)"
    )
    parser.add_argument(
        "--large-scale",
        action="store_true",
        help="Train on larger dataset (50% of data, ~2134 nodes)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./models",
        help="Directory to save models"
    )

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device
    device = setup_device()

    # Load dataset
    if args.large_scale:
        node_limit = 2134  # ~50% of data
        print("\n*** Large-Scale Dataset Mode ***")
    else:
        node_limit = args.node_limit
        print(f"\n*** Small-Scale Dataset Mode ({node_limit} nodes) ***")

    dataset_splits = load_dataset(args.data_path, node_limit=node_limit, seed=args.seed)

    # Train simple model
    print("\n" + "="*80)
    print("PHASE 1: SIMPLE BASELINE MODEL")
    print("="*80)
    simple_model, simple_metrics = train_simple_model(dataset_splits, device, args.save_dir)

    # Train AUC-optimized model
    print("\n" + "="*80)
    print("PHASE 2: AUC-OPTIMIZED MODEL")
    print("="*80)
    auc_model, auc_metrics = train_auc_model(dataset_splits, device, args.save_dir)

    # Optionally train on larger dataset
    if args.large_scale:
        print("\n" + "="*80)
        print("PHASE 3: LARGE-SCALE MODEL")
        print("="*80)
        # Reload larger dataset
        dataset_splits_large = load_dataset(args.data_path, node_limit=2134, seed=args.seed)
        large_model, large_metrics = train_large_scale_model(
            dataset_splits_large, device, args.save_dir
        )

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Models saved to {args.save_dir}")

    return {
        "simple": (simple_model, simple_metrics),
        "auc": (auc_model, auc_metrics),
    }


if __name__ == "__main__":
    main()
