"""
train.py — Train ContactLSTM with 5-fold stratified cross-validation.

Key design choices (documented for the paper):
  - StratifiedKFold (5 folds) to maintain label balance across splits
  - BCEWithLogitsLoss with pos_weight to handle class imbalance
  - Adam optimizer, lr=1e-3, weight decay=1e-4
  - Early stopping (patience=7) per fold
  - Best fold models saved to checkpoints/<tier>_fold<k>.pt
  - Final model trained on full train split saved to checkpoints/<tier>_final.pt
  - CV results (per-fold accuracy, F1, loss curves) saved to results/<tier>_cv_results.json

IMPORTANT: --data should point to the *_train.npz file produced by preprocess.py.
The *_test.npz file is reserved exclusively for evaluate.py and must never be
touched during training or cross-validation.

Usage
-----
python train.py --data processed_low_train.npz --tier low [options]

Options
-------
--epochs     int   Max training epochs per fold        (default: 50)
--folds      int   Number of CV folds                  (default: 5)
--hidden     int   LSTM hidden size                    (default: 128)
--batch      int   Batch size                          (default: 64)
--lr         float Learning rate                       (default: 1e-3)
--patience   int   Early stopping patience in epochs   (default: 7)
--seed       int   Random seed                         (default: 42)
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from model import ContactLSTM
from utils import EarlyStopping, load_npz, make_dataloaders, set_seed, get_device


def compute_pos_weight(y: np.ndarray) -> float:
    """
    Compute BCEWithLogitsLoss pos_weight = n_negative / n_positive.
    Handles degenerate cases gracefully.
    """
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0:
        return 1.0
    return float(n_neg / n_pos)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * X_batch.size(0)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= threshold).astype(int)
        all_preds.extend(preds.tolist())
        all_labels.extend(y_batch.cpu().numpy().astype(int).tolist())
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return avg_loss, acc, f1


def train_fold(X_train, y_train, X_val, y_val,
               input_size, args, device, ckpt_path):
    """Train a single fold, return best val metrics and loss history."""
    pos_weight = torch.tensor([compute_pos_weight(y_train)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = ContactLSTM(
        input_size=input_size,
        hidden_size=args.hidden,
        num_layers=2,
        dropout=0.3,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )
    stopper = EarlyStopping(patience=args.patience)

    train_loader, val_loader = make_dataloaders(X_train, y_train, X_val, y_val,
                                                 batch_size=args.batch)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(args.epochs):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    ep {epoch+1:3d}/{args.epochs}  "
                  f"tr_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"val_acc={val_acc:.3f}  val_f1={val_f1:.3f}")

        if stopper.step(val_loss, model):
            print(f"    Early stop at epoch {epoch+1}")
            break

    stopper.restore_best(model)

    # Re-evaluate with best weights
    _, best_acc, best_f1 = evaluate_epoch(model, val_loader, criterion, device)

    # Save checkpoint
    torch.save({
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": args.hidden,
        "num_layers": 2,
        "dropout": 0.3,
        "val_acc": best_acc,
        "val_f1": best_f1,
    }, ckpt_path)

    return best_acc, best_f1, history, model


def train_final(X, y, input_size, args, device, ckpt_path):
    """Train on full dataset (no validation split) for final model saved for eval."""
    pos_weight = torch.tensor([compute_pos_weight(y)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = ContactLSTM(
        input_size=input_size,
        hidden_size=args.hidden,
        num_layers=2,
        dropout=0.3,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    from torch.utils.data import TensorDataset, DataLoader
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True)

    # Use the mean number of epochs across folds (set externally) × 0.8 as heuristic
    # We just train for args.epochs directly on the full set
    for epoch in range(args.epochs):
        train_one_epoch(model, loader, optimizer, criterion, device)

    torch.save({
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": args.hidden,
        "num_layers": 2,
        "dropout": 0.3,
        "tier": args.tier,
    }, ckpt_path)
    print(f"  Final model saved -> {ckpt_path}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    required=True,  help="Path to .npz preprocessed dataset")
    parser.add_argument("--tier",    required=True,  help="Variability tier label (low/medium/high)")
    parser.add_argument("--epochs",  type=int, default=50)
    parser.add_argument("--folds",   type=int, default=5)
    parser.add_argument("--hidden",  type=int, default=128)
    parser.add_argument("--batch",   type=int, default=64)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--patience",type=int, default=7)
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"\n=== Training: tier={args.tier}  device={device} ===")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    X, y, feature_names = load_npz(args.data)
    input_size = X.shape[2]
    print(f"  Dataset: {X.shape}  label balance: {y.mean()*100:.1f}% contact")
    print(f"  Features ({input_size}): {feature_names[:5]}...")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_results = []
    fold_histories = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold_idx + 1}/{args.folds} ---")
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        ckpt_path = f"checkpoints/{args.tier}_fold{fold_idx + 1}.pt"
        acc, f1, history, _ = train_fold(
            X_tr, y_tr, X_val, y_val,
            input_size, args, device, ckpt_path
        )
        print(f"  Fold {fold_idx + 1} best: acc={acc:.4f}  f1={f1:.4f}")
        fold_results.append({"fold": fold_idx + 1, "val_acc": acc, "val_f1": f1})
        fold_histories.append(history)

    # Summary
    accs = [r["val_acc"] for r in fold_results]
    f1s  = [r["val_f1"]  for r in fold_results]
    print(f"\n=== CV Summary ({args.tier}) ===")
    print(f"  Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  F1 Score : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    # Save CV results
    results = {
        "tier": args.tier,
        "folds": fold_results,
        "cv_acc_mean": float(np.mean(accs)),
        "cv_acc_std":  float(np.std(accs)),
        "cv_f1_mean":  float(np.mean(f1s)),
        "cv_f1_std":   float(np.std(f1s)),
        "histories": fold_histories,
    }
    results_path = f"results/{args.tier}_cv_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  CV results -> {results_path}")

    # Train final model on full data
    print(f"\n--- Training final model on full dataset ({args.tier}) ---")
    final_ckpt = f"checkpoints/{args.tier}_final.pt"
    train_final(X, y, input_size, args, device, final_ckpt)

    print(f"\n=== Done training tier={args.tier} ===\n")


if __name__ == "__main__":
    main()
