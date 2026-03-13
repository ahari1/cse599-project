"""
visualize.py — Generate all research figures for the CSE599 contact detection project.

Reads from:
  results/<tier>_cv_results.json       (from train.py)
  results/eval_results.json            (from evaluate.py)

Produces (saved to figures/):
  1. cv_accuracy.png          — Per-tier CV accuracy mean ± std bar chart
  2. cv_f1.png                — Per-tier CV F1 mean ± std bar chart
  3. loss_curves.png          — Train/val loss curves per fold per tier
  4. degradation_heatmap.png  — 3×3 F1 heatmap (train tier × test tier)
  5. roc_curves.png           — In-distribution ROC curves per tier
  6. confusion_matrices.png   — In-distribution confusion matrices per tier

Usage
-----
python visualize.py [--results_dir results] [--figures_dir figures] [--tiers low,medium,high]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay

# ── Style ──────────────────────────────────────────────────────────────────────
TIER_COLORS = {"low": "#4C72B0", "medium": "#DD8452", "high": "#55A868"}
_FALLBACK_COLORS = ["#8172B2", "#C44E52", "#64B5CD", "#CCB974", "#B47CC7"]
TIER_ORDER  = ["low", "medium", "high"]
FIG_DPI     = 150


def _tier_color(tier: str, idx: int = 0) -> str:
    return TIER_COLORS.get(tier, _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)])

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def savefig(fig, path):
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 1 & 2: CV Accuracy / F1 bar charts ──────────────────────────────────

def _ordered_tiers(cv_data):
    """Return tiers from cv_data in TIER_ORDER where possible, then any extras."""
    ordered = [t for t in TIER_ORDER if t in cv_data]
    extras  = [t for t in cv_data   if t not in TIER_ORDER]
    return ordered + extras


def plot_cv_metric(cv_data, metric_key, ylabel, title, out_path):
    tiers = _ordered_tiers(cv_data)
    means = [cv_data[t][f"cv_{metric_key}_mean"] for t in tiers]
    stds  = [cv_data[t][f"cv_{metric_key}_std"]  for t in tiers]
    colors = [_tier_color(t, i) for i, t in enumerate(tiers)]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(tiers, means, yerr=stds, color=colors,
                  capsize=6, width=0.5, error_kw={"linewidth": 1.8, "ecolor": "#333"})

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                mean + std + 0.008,
                f"{mean:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylim(0, 1.1)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("Training Variability Tier", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    savefig(fig, out_path)


# ── Plot 3: Loss Curves ───────────────────────────────────────────────────────

def plot_loss_curves(cv_data, out_path):
    tiers = _ordered_tiers(cv_data)
    n_tiers = len(tiers)
    fig, axes = plt.subplots(1, n_tiers, figsize=(6 * n_tiers, 4.5), sharey=False)
    if n_tiers == 1:
        axes = [axes]

    for i, (ax, tier) in enumerate(zip(axes, tiers)):
        histories = cv_data[tier].get("histories", [])
        color = _tier_color(tier, i)
        for fold_idx, h in enumerate(histories):
            epochs = range(1, len(h["train_loss"]) + 1)
            label_tr = f"Fold {fold_idx+1} train" if fold_idx == 0 else None
            label_val = f"Fold {fold_idx+1} val" if fold_idx == 0 else None
            ax.plot(epochs, h["train_loss"], color=color, alpha=0.35, linewidth=1.2)
            ax.plot(epochs, h["val_loss"],   color=color, alpha=0.80, linewidth=1.5,
                    linestyle="--")

        # Mean train/val across folds (align to shortest fold)
        min_len_tr  = min(len(h["train_loss"]) for h in histories)
        min_len_val = min(len(h["val_loss"])   for h in histories)
        mean_tr  = np.mean([h["train_loss"][:min_len_tr]  for h in histories], axis=0)
        mean_val = np.mean([h["val_loss"][:min_len_val]   for h in histories], axis=0)
        ax.plot(range(1, min_len_tr + 1),  mean_tr,  color=_tier_color(tier, i), linewidth=2.5,
                label="Mean train")
        ax.plot(range(1, min_len_val + 1), mean_val, color=_tier_color(tier, i), linewidth=2.5,
                linestyle="--", label="Mean val")

        ax.set_title(f"{tier.capitalize()} Variability", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCE Loss")
        ax.legend(fontsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Training & Validation Loss Curves (all folds)", fontsize=13,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    savefig(fig, out_path)


# ── Plot 4: Degradation Heatmap ───────────────────────────────────────────────

def plot_degradation_heatmap(eval_data, metric, out_path):
    tiers = [t for t in TIER_ORDER if t in eval_data]
    n = len(tiers)
    matrix = np.full((n, n), np.nan)
    for i, train_t in enumerate(tiers):
        for j, test_t in enumerate(tiers):
            if test_t in eval_data.get(train_t, {}):
                matrix[i, j] = eval_data[train_t][test_t][metric]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label=metric.upper())

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"{t}" for t in tiers], fontsize=11)
    ax.set_yticklabels([f"{t}" for t in tiers], fontsize=11)
    ax.set_xlabel("Test Tier", fontsize=12)
    ax.set_ylabel("Train Tier", fontsize=12)
    ax.set_title(f"Distribution Shift: {metric.upper()} Matrix\n"
                 f"(rows=trained on, cols=tested on)", fontsize=12, fontweight="bold")

    for i in range(n):
        for j in range(n):
            if not np.isnan(matrix[i, j]):
                val = matrix[i, j]
                color = "black" if 0.35 < val < 0.75 else "white"
                border = " ★" if i == j else ""
                ax.text(j, i, f"{val:.3f}{border}", ha="center", va="center",
                        fontsize=11, fontweight="bold", color=color)

    fig.tight_layout()
    savefig(fig, out_path)


# ── Plot 5: ROC Curves (in-distribution only) ─────────────────────────────────

def plot_roc_curves(eval_data, processed_dir, checkpoint_dir, tiers, out_path, device):
    """
    Re-run inference to get per-sample probabilities for ROC computation.
    Falls back to just printing AUC from eval_results if needed.
    """
    import torch
    from model import ContactLSTM
    from utils import load_npz

    fig, ax = plt.subplots(figsize=(6, 5))
    plotted_any = False

    for tier in tiers:
        if tier not in eval_data:
            continue
        auc = eval_data[tier].get(tier, {}).get("auc_roc")
        npz_path  = os.path.join(processed_dir,  f"processed_{tier}_test.npz")
        ckpt_path = os.path.join(checkpoint_dir, f"{tier}_final.pt")

        if not os.path.exists(npz_path) or not os.path.exists(ckpt_path):
            if auc is not None:
                ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.3)
                ax.text(0.5, 0.5, f"{tier}: AUC={auc:.3f} (approx.)",
                        color=TIER_COLORS[tier], ha="center")
            continue

        X, y, _ = load_npz(npz_path)
        model, _ = ContactLSTM.from_checkpoint(ckpt_path, device)
        model.eval()

        # Get probabilities
        all_probs = []
        with torch.no_grad():
            bs = 256
            for i in range(0, len(X), bs):
                xb = torch.tensor(X[i:i+bs], dtype=torch.float32).to(device)
                p  = torch.sigmoid(model(xb)).cpu().numpy()
                all_probs.extend(p.tolist())
        probs = np.array(all_probs)

        fpr, tpr, _ = roc_curve(y.astype(int), probs)
        from sklearn.metrics import auc as sk_auc
        roc_auc = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, color=_tier_color(tier), linewidth=2.2,
                label=f"{tier.capitalize()} (AUC={roc_auc:.3f})")
        plotted_any = True

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("In-Distribution ROC Curves", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)

    if not plotted_any:
        ax.text(0.5, 0.5, "No data available\n(run evaluate.py first)",
                ha="center", va="center", transform=ax.transAxes, fontsize=12, alpha=0.5)

    fig.tight_layout()
    savefig(fig, out_path)


# ── Plot 6: Confusion Matrices ────────────────────────────────────────────────

def plot_confusion_matrices(eval_data, processed_dir, checkpoint_dir, tiers, out_path, device):
    import torch
    from model import ContactLSTM
    from utils import load_npz

    valid_tiers = [t for t in tiers if t in eval_data]
    n = len(valid_tiers)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, tier in zip(axes, valid_tiers):
        npz_path  = os.path.join(processed_dir,  f"processed_{tier}_test.npz")
        ckpt_path = os.path.join(checkpoint_dir, f"{tier}_final.pt")

        if not os.path.exists(npz_path) or not os.path.exists(ckpt_path):
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(f"{tier.capitalize()} — no data")
            continue

        X, y, _ = load_npz(npz_path)
        model, _ = ContactLSTM.from_checkpoint(ckpt_path, device)
        model.eval()

        all_preds = []
        with torch.no_grad():
            bs = 256
            for i in range(0, len(X), bs):
                xb = torch.tensor(X[i:i+bs], dtype=torch.float32).to(device)
                p  = torch.sigmoid(model(xb)).cpu().numpy()
                all_preds.extend((p >= 0.5).astype(int).tolist())

        cm = confusion_matrix(y.astype(int), all_preds, labels=[0, 1])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No Contact", "Contact"]
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{tier.capitalize()} Variability\n(in-distribution)",
                     fontsize=11, fontweight="bold")

    fig.suptitle("Confusion Matrices — Final Models (In-Distribution)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    savefig(fig, out_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir",     default="results")
    parser.add_argument("--figures_dir",     default="figures")
    parser.add_argument("--processed_dir",   default=".")
    parser.add_argument("--checkpoint_dir",  default="checkpoints")
    parser.add_argument("--tiers",           default="low,medium,high")
    args = parser.parse_args()

    tiers = [t.strip() for t in args.tiers.split(",")]
    os.makedirs(args.figures_dir, exist_ok=True)

    from utils import get_device
    device = get_device()

    print(f"\n=== Visualization (device={device}) ===")

    # --- Load result JSONs ---
    eval_data = load_json(os.path.join(args.results_dir, "eval_results.json"))
    cv_data   = {}
    for tier in tiers:
        d = load_json(os.path.join(args.results_dir, f"{tier}_cv_results.json"))
        if d:
            cv_data[tier] = d

    # --- Plot CV metrics ---
    if cv_data:
        plot_cv_metric(cv_data, "acc", "Accuracy",
                       "Cross-Validation Accuracy by Variability Tier",
                       os.path.join(args.figures_dir, "cv_accuracy.png"))
        plot_cv_metric(cv_data, "f1", "F1 Score",
                       "Cross-Validation F1 Score by Variability Tier",
                       os.path.join(args.figures_dir, "cv_f1.png"))
        plot_loss_curves(cv_data,
                         os.path.join(args.figures_dir, "loss_curves.png"))
    else:
        print("  [WARN] No CV results found — skipping cv_accuracy, cv_f1, loss_curves")

    # --- Degradation heatmap ---
    if eval_data:
        plot_degradation_heatmap(eval_data, "f1",
                                 os.path.join(args.figures_dir, "degradation_heatmap_f1.png"))
        plot_degradation_heatmap(eval_data, "accuracy",
                                 os.path.join(args.figures_dir, "degradation_heatmap_acc.png"))
    else:
        print("  [WARN] No eval_results.json — skipping heatmaps")

    # --- ROC curves ---
    plot_roc_curves(eval_data or {}, args.processed_dir, args.checkpoint_dir,
                    tiers, os.path.join(args.figures_dir, "roc_curves.png"), device)

    # --- Confusion matrices ---
    plot_confusion_matrices(eval_data or {}, args.processed_dir, args.checkpoint_dir,
                            tiers, os.path.join(args.figures_dir, "confusion_matrices.png"), device)

    print("\n=== All figures saved to:", args.figures_dir, "===\n")
    for fname in sorted(os.listdir(args.figures_dir)):
        if fname.endswith(".png"):
            print(f"  {fname}")


if __name__ == "__main__":
    main()
