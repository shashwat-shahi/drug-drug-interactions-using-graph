"""Training utilities for DDI model."""

import torch
import torch.nn.functional as F
from typing import Dict, Callable, Tuple, List
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
import json
import os


@dataclass
class TrainingMetrics:
    """Stores training metrics across epochs."""
    metrics: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)

    def log_step(self, split: str, **kwargs):
        """Log metrics for a single step."""
        if split not in self.metrics:
            self.metrics[split] = {k: [] for k in kwargs.keys()}

        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.metrics[split][key].append(float(value))

    def flush(self, epoch: int = None):
        """Print latest metrics."""
        for split, metrics in self.metrics.items():
            msg = f"{split.upper()}: "
            for key, values in metrics.items():
                if values:
                    msg += f"{key}={values[-1]:.4f} "
            if epoch is not None:
                msg = f"Epoch {epoch}: " + msg
            print(msg)

    def latest(self, keys: List[str]):
        """Get latest values for specified keys."""
        result = []
        for split, metrics in self.metrics.items():
            for key in keys:
                if key in metrics and metrics[key]:
                    result.append(f"{split}_{key}={metrics[key][-1]:.4f}")
        return " | ".join(result)

    def export(self) -> Dict:
        """Export metrics as dictionary."""
        return self.metrics

    def get_best_value(self, split: str, metric: str) -> float:
        """Get best value for a metric."""
        if split in self.metrics and metric in self.metrics[split]:
            values = self.metrics[split][metric]
            if values:
                return max(values)
        return 0.0


# Loss functions
def binary_log_loss(scores: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Computes the binary log loss for positive and negative drug pairs."""
    # Sigmoid transformation
    probs = {
        k: torch.sigmoid(v).clamp(1e-7, 1 - 1e-7)
        for k, v in scores.items()
    }

    # Compute positive and negative losses
    pos_loss = -torch.log(probs["pos"]).mean()
    neg_loss = -torch.log(1 - probs["neg"]).mean()

    return pos_loss + neg_loss


def auc_loss(scores: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Computes AUC-based loss for positive and negative drug pairs."""
    return torch.square(1 - (scores["pos"] - scores["neg"])).sum()


# Evaluation metrics
def evaluate_hits_at_20(scores: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Computes the hits@20 metric capturing positive pairs ranking."""
    # Find the 20th highest score among negative edges
    if scores["neg"].shape[0] >= 20:
        kth_score_in_negative_edges = torch.sort(scores["neg"])[0][-20]
    else:
        kth_score_in_negative_edges = torch.min(scores["neg"])

    # Compute the proportion of positive scores greater than the threshold
    return (
        torch.sum(scores["pos"] > kth_score_in_negative_edges).float()
        / scores["pos"].shape[0]
    )


def evaluate_hits_at_k(scores: Dict[str, torch.Tensor], k: int = 20) -> torch.Tensor:
    """Computes the hits@k metric."""
    if scores["neg"].shape[0] >= k:
        kth_score = torch.sort(scores["neg"])[0][-k]
    else:
        kth_score = torch.min(scores["neg"])

    return (
        torch.sum(scores["pos"] > kth_score).float()
        / scores["pos"].shape[0]
    )


def evaluate_auc(scores: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Computes AUC for link prediction."""
    from sklearn.metrics import roc_auc_score

    pos_scores = scores["pos"].cpu().detach().numpy()
    neg_scores = scores["neg"].cpu().detach().numpy()

    y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    y_pred = np.concatenate([pos_scores, neg_scores])

    auc = roc_auc_score(y_true, y_pred)
    return torch.tensor(auc)


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    graph_data: Dict,
    pairs_batch: Dict,
    loss_fn: Callable = auc_loss,
    norm_loss: bool = False,
) -> Dict[str, float]:
    """Performs a single training step, updating model parameters."""
    model.train()

    # Prepare data for device
    node_ids = torch.arange(model.n_nodes, device=device)
    senders = torch.tensor(graph_data["senders"], dtype=torch.long, device=device)
    receivers = torch.tensor(graph_data["receivers"], dtype=torch.long, device=device)

    pos_pairs = torch.tensor(pairs_batch["pos"], dtype=torch.long, device=device)
    neg_pairs = torch.tensor(pairs_batch["neg"], dtype=torch.long, device=device)

    # Forward pass
    scores = model(
        node_ids, senders, receivers,
        {"pos": pos_pairs, "neg": neg_pairs},
        training=True,
        is_pred=False
    )

    # Compute loss
    loss = loss_fn(scores)

    if norm_loss:
        loss = loss / (pos_pairs.shape[0] + neg_pairs.shape[0])

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Compute metrics
    with torch.no_grad():
        hits_at_20 = evaluate_hits_at_20(scores)

    metrics = {
        "loss": loss.item(),
        "hits@20": hits_at_20.item(),
    }

    return metrics


def eval_step(
    model: torch.nn.Module,
    device: torch.device,
    graph_data: Dict,
    pairs_batch: Dict,
    loss_fn: Callable = auc_loss,
    norm_loss: bool = False,
) -> Dict[str, float]:
    """Performs an evaluation step, computing loss and hits@20 metric."""
    model.eval()

    with torch.no_grad():
        # Prepare data for device
        node_ids = torch.arange(model.n_nodes, device=device)
        senders = torch.tensor(graph_data["senders"], dtype=torch.long, device=device)
        receivers = torch.tensor(graph_data["receivers"], dtype=torch.long, device=device)

        pos_pairs = torch.tensor(pairs_batch["pos"], dtype=torch.long, device=device)
        neg_pairs = torch.tensor(pairs_batch["neg"], dtype=torch.long, device=device)

        # Forward pass
        scores = model(
            node_ids, senders, receivers,
            {"pos": pos_pairs, "neg": neg_pairs},
            training=False,
            is_pred=False
        )

        # Compute loss
        loss = loss_fn(scores)

        if norm_loss:
            loss = loss / (pos_pairs.shape[0] + neg_pairs.shape[0])

        # Compute metrics
        hits_at_20 = evaluate_hits_at_20(scores)

    metrics = {
        "loss": loss.item(),
        "hits@20": hits_at_20.item(),
    }

    return metrics


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    dataset_splits: Dict,
    batch_size: int,
    num_epochs: int,
    loss_fn: Callable = auc_loss,
    norm_loss: bool = False,
    eval_every: int = 10,
    save_path: str = None,
) -> Tuple[TrainingMetrics, float]:
    """
    Training loop for the drug-drug interaction model.

    Args:
        model: The model to train
        optimizer: Optimizer for training
        device: Device to train on (CPU/GPU)
        dataset_splits: Dictionary of train/valid/test datasets
        batch_size: Batch size for training
        num_epochs: Number of epochs to train
        loss_fn: Loss function to use
        norm_loss: Whether to normalize loss
        eval_every: Evaluate every N epochs
        save_path: Path to save the best model

    Returns:
        Tuple of (metrics, best_validation_hits_at_20)
    """
    metrics = TrainingMetrics()
    best_val_hits = 0.0
    best_model_state = None

    # Progress bar for epochs
    epochs_bar = tqdm(range(num_epochs), desc="Training")

    for epoch in epochs_bar:
        rng = np.random.default_rng(epoch)

        # Training loop
        for pairs_batch in dataset_splits["train"].pairs.get_train_batches(
            batch_size, rng
        ):
            batch_metrics = train_step(
                model, optimizer, device,
                dataset_splits["train"].graph,
                pairs_batch,
                loss_fn=loss_fn,
                norm_loss=norm_loss
            )
            metrics.log_step(split="train", **batch_metrics)

        # Evaluation loop
        if epoch % eval_every == 0:
            for pairs_batch in dataset_splits["valid"].pairs.get_eval_batches(batch_size):
                batch_metrics = eval_step(
                    model, device,
                    dataset_splits["valid"].graph,
                    pairs_batch,
                    loss_fn=loss_fn,
                    norm_loss=norm_loss
                )
                metrics.log_step(split="valid", **batch_metrics)

            # Track best model
            val_hits = metrics.metrics["valid"]["hits@20"][-1]
            if val_hits > best_val_hits:
                best_val_hits = val_hits
                best_model_state = model.state_dict().copy()

        # Update progress bar
        latest_str = metrics.latest(["loss", "hits@20"])
        epochs_bar.set_postfix_str(latest_str)

    # Save best model if path provided
    if save_path and best_model_state:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(best_model_state, save_path)
        print(f"Best model saved to {save_path}")

    return metrics, best_val_hits


def predict(
    model: torch.nn.Module,
    device: torch.device,
    graph_data: Dict,
    node_pairs: np.ndarray,
) -> np.ndarray:
    """
    Make predictions on node pairs.

    Args:
        model: Trained model
        device: Device to use
        graph_data: Graph structure data
        node_pairs: Array of node pairs to predict on [n_pairs, 2]

    Returns:
        Predictions for each node pair
    """
    model.eval()

    with torch.no_grad():
        node_ids = torch.arange(model.n_nodes, device=device)
        senders = torch.tensor(graph_data["senders"], dtype=torch.long, device=device)
        receivers = torch.tensor(graph_data["receivers"], dtype=torch.long, device=device)
        pairs = torch.tensor(node_pairs, dtype=torch.long, device=device)

        scores = model(
            node_ids, senders, receivers,
            pairs,
            training=False,
            is_pred=True
        )

    return scores.cpu().numpy()
