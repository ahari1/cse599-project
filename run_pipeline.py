"""
run_pipeline.py — End-to-end pipeline runner for CSE599 contact detection.

Runs the full pipeline in order:
  1. [optional] Data collection (data_collection.py)
  2. Preprocessing (preprocess.py) for all tiers
  3. Training (train.py) for all tiers
  4. Evaluation (evaluate.py)
  5. Visualization (visualize.py)

Usage
-----
# Run everything (assumes data CSVs already exist):
python run_pipeline.py

# Also collect data first:
python run_pipeline.py --collect --iterations 500 --steps 150

# Run full experiment with tuned BCE weight:
python run_pipeline.py --collect --iterations 2500 --pos_weight 2

# Skip certain stages:
python run_pipeline.py --skip collect train

Options
-------
--collect       Flag: also run data collection (requires PyBullet)
--iterations    int   Iterations per tier for data collection (default: 500)
--steps         int   Steps per iteration for data collection  (default: 150)
--window_size   int   Preprocessing window size                (default: 50)
--stride        int   Preprocessing stride                     (default: 10)
--smooth        int   Smoothing window                         (default: 5)
--epochs        int   Training epochs per fold                 (default: 50)
--folds         int   CV folds                                 (default: 5)
--pos_weight    float Positive class weight for BCE loss       (default: 1.0)
--skip          list  Stages to skip: collect preprocess train evaluate visualize
--tiers         str   Comma-separated tiers                    (default: low,medium,high)
--seed          int   Global random seed                       (default: 42)
"""

import argparse
import os
import subprocess
import sys
import time


def run(cmd, desc):
    """Run a shell command, stream output, and raise on error."""
    print(f"\n{'='*60}")
    print(f"  STAGE: {desc}")
    print(f"  CMD:   {' '.join(cmd)}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    print(f"  ✓ Completed in {elapsed:.1f}s\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect",      action="store_true")
    parser.add_argument("--iterations",   type=int,   default=500)
    parser.add_argument("--steps",        type=int,   default=150)
    parser.add_argument("--window_size",  type=int,   default=50)
    parser.add_argument("--stride",       type=int,   default=10)
    parser.add_argument("--smooth",       type=int,   default=5)
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--folds",        type=int,   default=5)
    parser.add_argument("--pos_weight",   type=float, default=1.0)
    parser.add_argument("--skip",         nargs="*",  default=[])
    parser.add_argument("--tiers",        default="low,medium,high")
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    tiers = [t.strip() for t in args.tiers.split(",")]
    skip  = set(args.skip or [])
    py    = sys.executable

    print(f"\n{'#'*60}")
    print(f"# CSE599 — Full Pipeline")
    print(f"# Tiers: {tiers}")
    print(f"# Skip:  {skip or 'none'}")
    print(f"# BCE pos_weight: {args.pos_weight}")
    print(f"{'#'*60}")

    # ── Stage 1: Data Collection ──────────────────────────────────────────────
    if args.collect and "collect" not in skip:
        for tier in tiers:
            run([
                py, "data_collection.py",
                "--variability", tier,
                "--iterations",  str(args.iterations),
                "--steps",       str(args.steps),
                "--output",      f"data_{tier}.csv",
            ], f"Collect data [{tier}]")
    else:
        print("\n  [SKIP] Data collection")

    # ── Stage 2: Preprocessing ────────────────────────────────────────────────
    if "preprocess" not in skip:
        for tier in tiers:
            csv_path = f"data_{tier}.csv"
            if not os.path.exists(csv_path):
                print(f"  [WARN] {csv_path} not found, skipping preprocess for {tier}")
                continue

            output_prefix = f"processed_{tier}"

            run([
                py, "preprocess.py",
                "--input",         csv_path,
                "--output_prefix", output_prefix,
                "--window_size",   str(args.window_size),
                "--stride",        str(args.stride),
                "--smooth",        str(args.smooth),
            ], f"Preprocess [{tier}]")
    else:
        print("\n  [SKIP] Preprocessing")

    # ── Stage 3: Training ─────────────────────────────────────────────────────
    if "train" not in skip:
        for tier in tiers:
            npz_path = f"processed_{tier}_train.npz"

            if not os.path.exists(npz_path):
                print(f"  [WARN] {npz_path} not found, skipping train for {tier}")
                continue

            run([
                py, "train.py",
                "--data",       npz_path,
                "--tier",       tier,
                "--epochs",     str(args.epochs),
                "--folds",      str(args.folds),
                "--seed",       str(args.seed),
                "--pos_weight", str(args.pos_weight),
            ], f"Train [{tier}]")
    else:
        print("\n  [SKIP] Training")

    # ── Stage 4: Evaluation ───────────────────────────────────────────────────
    if "evaluate" not in skip:
        run([
            py, "evaluate.py",
            "--tiers", args.tiers,
        ], "Cross-tier evaluation")
    else:
        print("\n  [SKIP] Evaluation")

    # ── Stage 5: Visualization ────────────────────────────────────────────────
    if "visualize" not in skip:
        run([
            py, "visualize.py",
            "--tiers", args.tiers,
        ], "Visualization")
    else:
        print("\n  [SKIP] Visualization")

    print(f"\n{'#'*60}")
    print("# Pipeline complete!")
    print(f"# Figures -> figures/")
    print(f"# Results -> results/")
    print(f"# Models  -> checkpoints/")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
