"""
utils.py — Shared utilities for the CSE599 contact detection pipeline.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Return the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EarlyStopping:
    """Simple early stopping based on validation loss."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, val_loss: float, model) -> bool:
        """
        Call after each epoch. Returns True if training should stop.
        Saves the best model state internally.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model):
        """Load the best model weights back into model."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def load_npz(path: str):
    """
    Load a preprocessed .npz dataset.

    Returns
    -------
    X : np.ndarray  shape (N, window, features)
    y : np.ndarray  shape (N,)
    feature_names : list[str]
    """
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    feature_names = list(data["feature_names"]) if "feature_names" in data else []
    return X, y, feature_names


def make_dataloaders(X_train, y_train, X_val, y_val,
                     batch_size: int = 64, num_workers: int = 0):
    """Create PyTorch DataLoaders for train and val splits."""
    from torch.utils.data import TensorDataset, DataLoader

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False)
    return train_loader, val_loader
