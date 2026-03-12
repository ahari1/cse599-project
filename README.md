### Prereqs

- Python 3.12
- Conda

### Install

```bash
conda create -y -p ./conda_env python=3.12
conda activate ./conda_env

conda install -y -c conda-forge pybullet

pip install numpy scipy torch matplotlib pandas scikit-learn
```

---

## Project Structure

```
cse599-project/
├── data_collection.py   # PyBullet simulation data collector
├── collect_all.sh       # Collects all 3 variability tiers
├── preprocess.py        # Smoothing, normalization, windowing
├── train.py             # 5-fold stratified CV + LSTM training
├── evaluate.py          # 3×3 cross-tier distribution shift evaluation
├── visualize.py         # All research figures
├── run_pipeline.py      # End-to-end pipeline runner
├── model.py             # ContactLSTM architecture
├── utils.py             # Shared utilities
├── checkpoints/         # Saved model weights (generated)
├── results/             # JSON metrics (generated)
└── figures/             # PNG plots (generated)
```

---

## Usage

### Variability Tiers

| Tier   | Block placement | Arm jitter | Path noise |
|--------|----------------|------------|------------|
| low    | narrow range   | ±0.02 rad  | none       |
| medium | moderate range | ±0.15 rad  | σ=0.015 m  |
| high   | wide range     | ±0.40 rad  | σ=0.04 m   |

---

### Option A — Run Everything at Once

```bash
# Collect data + run full pipeline (500 iterations per tier)
python run_pipeline.py --collect --iterations 500 --steps 150

# If data CSVs already exist, skip collection:
python run_pipeline.py
```

### Option B — Step by Step

**1. Collect data** (one tier at a time, or use the shell script):

```bash
# All tiers at once:
bash collect_all.sh 500 150

# Or individually:
python data_collection.py --variability low    --iterations 500 --steps 150 --output data_low.csv
python data_collection.py --variability medium --iterations 500 --steps 150 --output data_medium.csv
python data_collection.py --variability high   --iterations 500 --steps 150 --output data_high.csv
```

**2. Preprocess** (episode-level train/test split → smooth → normalize → window):

Each tier produces two files: `processed_{tier}_train.npz` (used for training)
and `processed_{tier}_test.npz` (held-out, never touched during training/CV).

```bash
python preprocess.py --input data_low.csv    --output_prefix processed_low
python preprocess.py --input data_medium.csv --output_prefix processed_medium
python preprocess.py --input data_high.csv   --output_prefix processed_high
```

**3. Train** (5-fold CV on train split + final model per tier):

```bash
python train.py --data processed_low_train.npz    --tier low
python train.py --data processed_medium_train.npz --tier medium
python train.py --data processed_high_train.npz   --tier high
```

**4. Evaluate** (3×3 cross-tier matrix on held-out test splits):

```bash
python evaluate.py
```

**5. Visualize** (all figures to `figures/`):

```bash
python visualize.py
```

---

## Figures Produced

| File | Description |
|------|-------------|
| `figures/cv_accuracy.png` | Per-tier CV accuracy mean ± std |
| `figures/cv_f1.png` | Per-tier CV F1 mean ± std |
| `figures/loss_curves.png` | Train/val loss per fold per tier |
| `figures/degradation_heatmap_f1.png` | 3×3 F1 distribution shift heatmap |
| `figures/degradation_heatmap_acc.png` | 3×3 accuracy distribution shift heatmap |
| `figures/roc_curves.png` | In-distribution ROC curves |
| `figures/confusion_matrices.png` | In-distribution confusion matrices |

---

## Model Architecture

- **Input**: windowed joint telemetry (positions, velocities, torques × 7 joints + hand pose = 28 features)
- **Window**: 50 timesteps, stride 10 (default)
- **Model**: 2-layer stacked LSTM (hidden=128) → Dropout(0.3) → Linear(1) → Sigmoid
- **Label**: window = contact if **any** frame in window has contact
- **Training**: BCEWithLogitsLoss + pos_weight (class imbalance), Adam lr=1e-3, early stopping patience=7
- **Evaluation**: Final model trained on full tier dataset used for cross-tier distribution shift evaluation