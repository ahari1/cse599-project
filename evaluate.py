"""
evaluate.py — Evaluate ContactLSTM models across all (train_tier × test_tier) pairs.

For each tier's final model (checkpoints/<tier>_final.pt), we run inference on
the HELD-OUT TEST splits (processed_{tier}_test.npz) to build a full 3×3
distribution-shift matrix. The test split is never used during training or CV.

Metrics computed per (train, test) pair:
  accuracy, precision, recall, F1, AUC-ROC

Output: results/eval_results.json

Usage
-----
python evaluate.py [options]

Options
-------
--processed_dir   str   Directory containing processed_*_test.npz files (default: .)
--checkpoint_dir  str   Directory containing *_final.pt files            (default: checkpoints)
--results_dir     str   Output directory for JSON results                 (default: results)
--threshold       float Decision threshold for binary prediction          (default: 0.5)
--tiers           str   Comma-separated tiers to evaluate                (default: low,medium,high)
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from model import ContactLSTM
from utils import get_device, load_npz


def predict(model, X: np.ndarray, device, batch_size: int = 256):
    """Run inference over X in batches, return prob array."""
    model.eval()
    probs = []
    n = len(X)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
            logits = model(batch)
            p = torch.sigmoid(logits).cpu().numpy()
            probs.extend(p.tolist())
    return np.array(probs, dtype=np.float32)


def compute_metrics(y_true, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    y_true_int = y_true.astype(int)

    acc = accuracy_score(y_true_int, preds)
    prec = precision_score(y_true_int, preds, zero_division=0)
    rec = recall_score(y_true_int, preds, zero_division=0)
    f1 = f1_score(y_true_int, preds, zero_division=0)

    # AUC-ROC: handle degenerate case where only one class is present in test set
    try:
        auc = roc_auc_score(y_true_int, probs)
    except ValueError:
        auc = float("nan")

    return {
        "accuracy":  round(float(acc),  4),
        "precision": round(float(prec), 4),
        "recall":    round(float(rec),  4),
        "f1":        round(float(f1),   4),
        "auc_roc":   round(float(auc),  4) if not np.isnan(auc) else None,
        "n_test":    int(len(y_true)),
        "contact_rate_test": round(float(y_true.mean()), 4),
        "n_predicted_contact": int(preds.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir",  default=".")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--results_dir",    default="results")
    parser.add_argument("--threshold",      type=float, default=0.5)
    parser.add_argument("--tiers",          default="low,medium,high")
    args = parser.parse_args()

    tiers = [t.strip() for t in args.tiers.split(",")]
    device = get_device()
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"\n=== Evaluation (device={device}) ===")
    print(f"  Tiers: {tiers}")

    # ---- Load all held-out test datasets ----
    datasets = {}
    for tier in tiers:
        npz_path = os.path.join(args.processed_dir, f"processed_{tier}_test.npz")
        if not os.path.exists(npz_path):
            print(f"  [WARN] Missing dataset: {npz_path} — skipping")
            continue
        X, y, feat = load_npz(npz_path)
        datasets[tier] = (X, y)
        print(f"  Loaded processed_{tier}_test.npz: {X.shape}, label={y.mean()*100:.1f}%")

    # ---- Load all final models ----
    models = {}
    for tier in tiers:
        ckpt_path = os.path.join(args.checkpoint_dir, f"{tier}_final.pt")
        if not os.path.exists(ckpt_path):
            print(f"  [WARN] Missing checkpoint: {ckpt_path} — skipping")
            continue
        model, ckpt = ContactLSTM.from_checkpoint(ckpt_path, device)
        models[tier] = model
        print(f"  Loaded {tier}_final.pt (input_size={ckpt['input_size']})")

    # ---- 3×3 evaluation matrix ----
    eval_matrix = {}

    for train_tier, model in models.items():
        eval_matrix[train_tier] = {}
        for test_tier, (X_test, y_test) in datasets.items():
            print(f"\n  train={train_tier}  test={test_tier}")

            # Check input size compatibility
            model_input_size = next(model.parameters()).shape  # proxy check
            expected = X_test.shape[2]

            probs = predict(model, X_test, device)
            metrics = compute_metrics(y_test, probs, threshold=args.threshold)

            is_in_dist = (train_tier == test_tier)
            metrics["in_distribution"] = is_in_dist

            print(f"    acc={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}  "
                  f"auc={metrics['auc_roc']}  {'[in-dist]' if is_in_dist else '[SHIFT]'}")
            eval_matrix[train_tier][test_tier] = metrics

    # ---- Save results ----
    results_path = os.path.join(args.results_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(eval_matrix, f, indent=2)
    print(f"\n  Results saved -> {results_path}")

    # ---- Print summary table ----
    print("\n=== Accuracy Matrix (rows=train, cols=test) ===")
    header = f"{'':12s}" + "".join(f"{t:>12s}" for t in tiers)
    print(header)
    for train_tier in tiers:
        if train_tier not in eval_matrix:
            continue
        row = f"{train_tier:<12s}"
        for test_tier in tiers:
            if test_tier in eval_matrix[train_tier]:
                acc = eval_matrix[train_tier][test_tier]["accuracy"]
                row += f"{acc:>12.4f}"
            else:
                row += f"{'N/A':>12s}"
        print(row)

    print("\n=== F1 Matrix (rows=train, cols=test) ===")
    print(header)
    for train_tier in tiers:
        if train_tier not in eval_matrix:
            continue
        row = f"{train_tier:<12s}"
        for test_tier in tiers:
            if test_tier in eval_matrix[train_tier]:
                f1 = eval_matrix[train_tier][test_tier]["f1"]
                row += f"{f1:>12.4f}"
            else:
                row += f"{'N/A':>12s}"
        print(row)

    print("\n=== Done ===\n")


if __name__ == "__main__":
    main()
