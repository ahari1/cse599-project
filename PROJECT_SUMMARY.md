# CSE599 Project: LSTM Contact Detection in Robot Telemetry

## What This Project Does

This project investigates whether a machine learning model — specifically a stacked LSTM (Long Short-Term Memory) neural network — can reliably detect **contact events** (i.e., when a robot arm's hand physically touches an object) from raw joint telemetry alone, and crucially, how well that detection ability **holds up when the robot's operating conditions change**.

The core research question is about **distribution shift**: if you train a model on data where picking up an object is easy and predictable (low variability), does it still work when the robot encounters harder, more chaotic conditions (high variability)? This is a fundamental question in robot learning robustness.

To study this, the entire pipeline — from simulation to plots — is self-contained and runs with a single command.

---

## The Big Picture: Pipeline Flow

```
PyBullet Simulation
       │
       ▼
data_collection.py  ──→  data_low.csv / data_medium.csv / data_high.csv
       │
       ▼
preprocess.py  ──→  processed_{tier}_train.npz  (80% of episodes, used for CV + training)
               └──→  processed_{tier}_test.npz   (20% of episodes, held out — never seen during training)
               └──→  processed_{tier}_scaler.pkl (fit on train only)
       │
       ▼
train.py ──→  checkpoints/{tier}_fold{k}.pt  +  checkpoints/{tier}_final.pt
           └→  results/{tier}_cv_results.json
       │
       ▼
evaluate.py  ──→  results/eval_results.json   (3×3 cross-tier matrix, all on _test.npz)
       │
       ▼
visualize.py  ──→  figures/*.png  (bar charts, heatmaps, ROC curves, confusion matrices)
```

All of these stages are orchestrated by `run_pipeline.py`.

---

## File-by-File Breakdown

### `data_collection.py` — The Simulator & Data Recorder

**What it does:** Runs a PyBullet physics simulation where a Kuka iiwa robot arm reaches down to touch a block on a table. At every step of the simulation, it records the robot's full joint state and labels whether the hand is in contact with the block.

**Key concepts:**
- **Three variability tiers** are defined (`low`, `medium`, `high`). These control how much the environment varies between episodes:
  - *Block placement*: how far the block can be from the robot's default reach
  - *Arm jitter*: random noise added to the arm's starting joint angles
  - *Path noise*: Gaussian noise added to the IK-computed waypoints the arm moves through
- Each **episode** resets the block to a new random pose, resets the arm, then moves through two phases: first hovering above the block, then descending to touch it.
- **Per-step data recorded** (into a CSV): joint positions, velocities, and torques for all 7 joints (21 values), the hand's 3D position and quaternion orientation (7 values), and the binary `in_contact` label + raw contact force.
- The output is one CSV per tier: `data_low.csv`, `data_medium.csv`, `data_high.csv`.

**Why it's designed this way:** Using three variability levels lets us later ask "does a model trained in easy conditions transfer to hard conditions?" — the central research question.

---

### `collect_all.sh` — Shell Script for Batch Data Collection

**What it does:** A convenience Bash script that calls `data_collection.py` three times (once per tier) with the same iteration/step count. Takes optional arguments for number of iterations and steps per iteration.

**Usage:** `bash collect_all.sh 500 150`

---

### `preprocess.py` — Data Cleaning, Windowing & Normalization

**What it does:** Transforms raw per-step CSVs into a form the LSTM can learn from, with a clean train/test boundary.

**Steps it performs in order:**
1. **Load & clean**: Reads the CSV, drops `step`, `variability_level`, and `contact_force` columns. Keeps `iteration` temporarily for the split. `contact_force` is dropped because it's a direct proxy for the label — keeping it would be cheating.
2. **Episode-level train/test split**: Shuffles the unique episode IDs (with a fixed seed for reproducibility) and reserves the last 20% as the held-out test set. **This split happens before any windowing**, which is critical — splitting after windowing would allow overlapping windows from the same episode to appear in both train and test (data leakage).
3. **Drop iteration column**: After the split is done, `iteration` is dropped from both halves.
4. **Smooth**: Applies a rolling mean (default window=5 steps) across all feature columns independently in each split. This reduces high-frequency sensor noise.
5. **Window**: Cuts each split's time series into overlapping fixed-length windows (default: 50 steps long, sliding by 10 steps). Each window becomes one sample.
6. **Label**: A window is labeled `1` (contact) if **any** single step within it was in contact; otherwise `0`. This "any-frame" labeling makes the task binary.
7. **Normalize**: Fits a `StandardScaler` on the **train** windows only, then applies it to both train and test. The scaler is also saved to a `.pkl` file.
8. **Save**: Outputs two compressed `.npz` files per tier.

**Output per tier:** `processed_{tier}_train.npz` + `processed_{tier}_test.npz` + `processed_{tier}_scaler.pkl`

---

### `model.py` — The Neural Network Architecture

**What it does:** Defines `ContactLSTM`, the PyTorch model used throughout the project.

**Architecture:**
```
Input (batch, 50 timesteps, 28 features)
    │
    ▼
LSTM Layer 1  (hidden_size=128)
    │  dropout=0.3 between layers
LSTM Layer 2  (hidden_size=128)
    │
    ▼  (take only the last timestep's output)
Dropout(0.3)
    │
    ▼
Linear(128 → 1)
    │
    ▼
Output logit  (sigmoid applied at inference)
```

**Why LSTM?** The robot's state at any given moment depends on where it was a moment ago — velocity, torque, and position are all temporally correlated. LSTMs are designed to model exactly this kind of sequential dependency.

**Key methods:**
- `forward(x)` — returns raw logits (pre-sigmoid); loss is computed with `BCEWithLogitsLoss` which is numerically more stable than applying sigmoid first.
- `predict_proba(x)` — returns sigmoid probabilities for inference.
- `from_checkpoint(path)` — class method to load a saved model from a `.pt` file, reading the hyperparameters stored inside the checkpoint.

---

### `utils.py` — Shared Helper Utilities

**What it does:** Houses small utilities used across multiple scripts so they don't need to be re-implemented everywhere.

**Contents:**
- `set_seed(seed)` — Sets seeds for Python's `random`, NumPy, and PyTorch (including CUDA) to ensure reproducible runs.
- `get_device()` — Returns the best available PyTorch device: CUDA GPU if available, Apple MPS (Metal) if on a Mac with Apple Silicon, otherwise CPU.
- `EarlyStopping` — A class that monitors per-epoch validation loss and triggers training to stop when the loss hasn't improved for `patience` epochs. It also saves the best model weights internally and can restore them after stopping.
- `load_npz(path)` — Loads an `.npz` file produced by `preprocess.py` and returns `X`, `y`, and `feature_names`.
- `make_dataloaders(...)` — Wraps NumPy arrays into PyTorch `TensorDataset`/`DataLoader` objects for batched training and validation.

---

### `train.py` — Model Training with Cross-Validation

**What it does:** Takes a preprocessed `.npz` dataset for one tier and trains the `ContactLSTM` using 5-fold stratified cross-validation, then trains a final model on the entire dataset.

**Training procedure:**
1. **Stratified K-Fold (k=5)**: The data is split into 5 folds while preserving the contact/no-contact class ratio in each fold. This is important because contact events are a minority class.
2. **Class imbalance handling**: Computes `pos_weight = n_negative / n_positive` and passes it to `BCEWithLogitsLoss`. This upweights the loss from positive (contact) samples, preventing the model from ignoring the minority class.
3. **Optimizer**: Adam with `lr=1e-3` and `weight_decay=1e-4` (L2 regularization).
4. **Learning rate schedule**: `ReduceLROnPlateau` — halves the LR if val loss hasn't improved for 3 epochs.
5. **Early stopping**: Stops training and restores the best weights if val loss hasn't improved for 7 consecutive epochs.
6. **Gradient clipping**: Clips gradient norm to 5.0 to prevent exploding gradients in the LSTM.
7. **Per-fold checkpoints**: Each fold's best model is saved to `checkpoints/{tier}_fold{k}.pt`.
8. **Final model**: After CV, a separate model is trained on **all** the data (no validation split) and saved to `checkpoints/{tier}_final.pt`. This is the model used for cross-tier evaluation.

**Output:**
- `checkpoints/{tier}_fold{1-5}.pt` — per-fold best models
- `checkpoints/{tier}_final.pt` — full-data final model
- `results/{tier}_cv_results.json` — per-fold metrics + full loss curves (used by `visualize.py`)

---

### `evaluate.py` — Cross-Tier Distribution Shift Evaluation

**What it does:** Loads every final model (one per tier) and runs it against every dataset (one per tier), building a **3×3 evaluation matrix** that directly shows how well each model generalizes across conditions.

**The matrix:**

|  | Test: Low | Test: Medium | Test: High |
|---|---|---|---|
| **Train: Low** | in-distribution | shift ↑ | shift ↑↑ |
| **Train: Medium** | shift ↓ | in-distribution | shift ↑ |
| **Train: High** | shift ↓↓ | shift ↓ | in-distribution |

The diagonal cells are **in-distribution** performance (training and test conditions match). Off-diagonal cells reveal how badly (or how gracefully) each model degrades when the test variability differs from training.

**Metrics computed per cell:** Accuracy, Precision, Recall, F1, AUC-ROC.

**Output:** `results/eval_results.json`

---

### `visualize.py` — Research Figure Generation

**What it does:** Reads the JSON files from `train.py` and `evaluate.py` and produces all the paper-ready plots, saved as PNGs to the `figures/` directory.

**Figures produced:**

| File | What it shows |
|------|---------------|
| `cv_accuracy.png` | Bar chart: mean ± std CV accuracy per tier |
| `cv_f1.png` | Bar chart: mean ± std CV F1 per tier |
| `loss_curves.png` | Train + val loss curves for each fold and each tier (with mean overlay) |
| `degradation_heatmap_f1.png` | 3×3 color heatmap of F1 scores from the eval matrix |
| `degradation_heatmap_acc.png` | Same heatmap but for accuracy |
| `roc_curves.png` | In-distribution ROC curves per tier, with AUC in the legend |
| `confusion_matrices.png` | In-distribution confusion matrices per tier |

For the ROC curves and confusion matrices, `visualize.py` re-runs inference directly (it loads the final models and preprocessed datasets itself) in order to get per-sample probabilities, not just summary statistics.

---

### `run_pipeline.py` — End-to-End Pipeline Orchestrator

**What it does:** Ties all five stages together so the entire project can be reproduced with a single command. It calls each script as a subprocess in order, streaming output and timing each stage.

**Stages (all skippable via `--skip`):**
1. Data collection (optional, only if `--collect` flag is passed)
2. Preprocessing (one run per tier)
3. Training (one run per tier)
4. Evaluation (single run, all tiers together)
5. Visualization (single run, all tiers together)

**Key design:** Uses `sys.executable` to call the same Python interpreter that launched `run_pipeline.py`, ensuring the conda environment is respected.

---

## How the Parts Connect

The data flows strictly forward through the pipeline — each stage consumes the output of the previous one:

```
Simulation (PyBullet)
    produces → raw CSVs (one per tier)

Preprocessing
    consumes → raw CSVs
    produces → _train.npz, _test.npz, _scaler.pkl (per tier)

Training
    consumes → _train.npz arrays only
    produces → .pt model checkpoints + CV result JSONs

Evaluation
    consumes → .pt checkpoints + _test.npz arrays (held-out, never seen during training)
    produces → eval_results.json

Visualization
    consumes → eval_results.json + CV result JSONs
             + (re-loads .pt + _test.npz for ROC/confusion figures)
    produces → figures/*.png
```

**`model.py`** is the shared definition used by `train.py`, `evaluate.py`, and `visualize.py`.

**`utils.py`** is the shared toolkit used by `train.py`, `evaluate.py`, and `visualize.py`.

**`run_pipeline.py`** + **`collect_all.sh`** are the two entry points that wire everything together.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| 3 variability tiers (low/medium/high) | Enables systematic study of distribution shift |
| LSTM over simpler models | Robot state is temporally correlated; LSTMs capture this naturally |
| Episode-level train/test split (before windowing) | Prevents overlapping windows from bleeding across the boundary; ensures the test set is truly unseen |
| 20% held-out test set — never touched during training | In-distribution eval numbers accurately reflect real generalization, not memorization |
| Windowed labeling ("any-frame" = contact) | A contact label anywhere in a window signals the arm is near the object |
| BCEWithLogitsLoss + pos_weight | Contact is a minority class; upweighting prevents the model from predicting "no contact" always |
| Final model trained on full train split | Cross-tier eval should use a maximally-informed model, not a held-out fold |
| Scaler fitted only on training episodes | Prevents test set statistics from leaking into normalization |
| 5-fold stratified CV | Reliable estimate of in-distribution performance that preserves class balance |

---

## How to Run It

```bash
# Run full pipeline (collect data + everything else)
python run_pipeline.py --collect --iterations 500 --steps 150

# Or, if data CSVs already exist:
python run_pipeline.py

# Results land in:
#   figures/      ← PNG plots
#   results/      ← JSON metrics
#   checkpoints/  ← saved model weights
#
# Preprocessed data files:
#   processed_{tier}_train.npz  ← used for training + CV
#   processed_{tier}_test.npz   ← held-out, used only for final evaluation
```
