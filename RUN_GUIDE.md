# Contact Detection Pipeline: Execution Guide

This guide provides the commands to run each step of the pipeline manually.

## 0. Environment Setup
Ensure your conda environment is active before running any scripts.
```bash
conda activate ./conda_env
```

## Option A: Automated Pipeline (Recommended)
You can run the entire pipeline—from data collection to visualization—with a single command. Use the `--collect` flag if you haven't generated the CSV data yet.

```bash
# Collect data + Run Full Pipeline (500 iterations per tier)
python run_pipeline.py --collect --iterations 500 --steps 150

# If CSV data already exists, simply run:
python run_pipeline.py
```

## Option B: Manual Step-by-Step
If you prefer to run or debug individual stages, follow these steps:

### 1. Data Generation
Collect robot telemetry data for all variability tiers. The arguments are `<iterations>` and `<steps>`.
```bash
# Collect 500 episodes per tier
bash collect_all.sh 500 150
```

## 2. Preprocessing
Transform raw CSVs into windowed, normalized NumPy arrays.
```bash
python preprocess.py --input data_low.csv    --output_prefix processed_low
python preprocess.py --input data_medium.csv --output_prefix processed_medium
python preprocess.py --input data_high.csv   --output_prefix processed_high
```

## 3. Training
Train the LSTM models using 5-fold stratified cross-validation.
```bash
python train.py --data processed_low_train.npz    --tier low
python train.py --data processed_medium_train.npz --tier medium
python train.py --data processed_high_train.npz   --tier high
```

## 4. Evaluation
Generate the 3x3 cross-tier distribution shift matrix.
```bash
python evaluate.py
```

## 5. Visualization
Generate all research figures (heatmaps, ROC curves, etc.).
```bash
python visualize.py
```

---

## Artifacts Created
- **Figures**: `figures/` (PNG plots)
- **Metrics**: `results/eval_results.json`
- **Models**: `checkpoints/` (.pt weights)
- **Data**: `processed_*.npz`
