"""
preprocess.py — Preprocessing pipeline for CSE599 contact detection.

Steps
-----
1. Load raw CSV produced by data_collection.py
2. Split episodes (by iteration ID) into train and held-out test sets
   before any windowing — prevents window-level data leakage
3. Drop metadata columns (iteration, step, variability_level, contact_force)
4. Apply rolling mean smoothing to telemetry features (per split)
5. Normalize features with StandardScaler (fit on TRAIN split only, applied to both)
6. Segment each split into fixed-length overlapping windows
7. Label each window: 1 if any frame in window has in_contact==1, else 0
8. Save X (N, window, features) and y (N,) to separate _train.npz and _test.npz
   Plus the fitted scaler to _scaler.pkl

Usage
-----
python preprocess.py --input data_low.csv --output_prefix processed_low [options]

Options
-------
--window_size   int   Length of each time window in steps         (default: 50)
--stride        int   Step between consecutive windows            (default: 10)
--smooth        int   Rolling mean window size; 1=disabled        (default: 5)
--test_frac     float Fraction of episodes held out for testing   (default: 0.2)
--seed          int   Random seed for episode shuffle             (default: 42)
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Columns we always drop before building feature matrix.
# contact_force / obstacle_contact / obstacle_force are ground-truth signals
# that would cause label leakage if left as input features.
_DROP_COLS = ["step", "variability_level",
              "contact_force", "obstacle_contact", "obstacle_force"]
_ITER_COL  = "iteration"
_LABEL_COL = "in_contact"


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load CSV and report basic stats. Keeps 'iteration' for episode splitting."""
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows from {csv_path}")
    print(f"  Columns: {list(df.columns)}")

    drop = [c for c in _DROP_COLS if c in df.columns]
    df = df.drop(columns=drop)

    contact_rate = df[_LABEL_COL].mean() * 100
    print(f"  Contact rate: {contact_rate:.2f}%")
    return df


def split_episodes(df: pd.DataFrame, test_frac: float, seed: int):
    """
    Split by unique episode (iteration) IDs into train and test DataFrames.

    Splitting at the episode level prevents overlapping windows from the same
    episode appearing in both train and test sets.

    Returns
    -------
    df_train, df_test : pd.DataFrame  (both still contain _ITER_COL)
    """
    episode_ids = df[_ITER_COL].unique()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(episode_ids)

    n_test  = max(1, int(len(shuffled) * test_frac))
    test_ids  = set(shuffled[-n_test:])
    train_ids = set(shuffled[:-n_test])

    df_train = df[df[_ITER_COL].isin(train_ids)].copy()
    df_test  = df[df[_ITER_COL].isin(test_ids)].copy()

    print(f"  Episodes  — train: {len(train_ids)}  test: {len(test_ids)}")
    print(f"  Rows      — train: {len(df_train)}    test: {len(df_test)}")
    print(f"  Contact % — train: {df_train[_LABEL_COL].mean()*100:.2f}%  "
          f"test: {df_test[_LABEL_COL].mean()*100:.2f}%")
    return df_train, df_test


def drop_iter_col(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the iteration column after the episode split has been done."""
    return df.drop(columns=[_ITER_COL], errors="ignore")


def smooth_signals(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Apply rolling mean to all telemetry features (not the label)."""
    if window <= 1:
        return df
    feature_cols = [c for c in df.columns if c != _LABEL_COL]
    df[feature_cols] = (
        df[feature_cols]
        .rolling(window=window, min_periods=1, center=False)
        .mean()
    )
    return df


def make_windows(df: pd.DataFrame, window_size: int, stride: int):
    """
    Segment the DataFrame into overlapping windows.

    Returns
    -------
    X : np.ndarray  (N, window_size, n_features)
    y : np.ndarray  (N,)  binary label per window
    feature_names : list[str]
    """
    feature_cols = [c for c in df.columns if c != _LABEL_COL]
    features = df[feature_cols].values.astype(np.float32)
    labels   = df[_LABEL_COL].values.astype(np.float32)

    n = len(features)
    X_list, y_list = [], []

    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        X_list.append(features[start:end])
        # "any-frame" labeling: window is contact if any step in it is contact
        y_list.append(1.0 if labels[start:end].any() else 0.0)

    X = np.stack(X_list, axis=0)               # (N, window, features)
    y = np.array(y_list, dtype=np.float32)      # (N,)
    return X, y, feature_cols


def normalize(X: np.ndarray, scaler: StandardScaler = None):
    """
    Normalize features using StandardScaler.

    If scaler is None, fit a new one on this data (use for train split only).
    If scaler is provided, apply it as-is (use for test split — no refitting).

    Parameters
    ----------
    X      : (N, window, features)
    scaler : fitted StandardScaler or None

    Returns
    -------
    X_norm : (N, window, features)
    scaler : fitted scaler
    """
    N, W, F = X.shape
    X_2d = X.reshape(-1, F)
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(X_2d)
    X_norm = scaler.transform(X_2d).reshape(N, W, F).astype(np.float32)
    return X_norm, scaler


def save_split(X, y, feature_names, window_size, stride, path):
    np.savez_compressed(
        path,
        X=X,
        y=y,
        feature_names=np.array(feature_names),
        window_size=np.array(window_size),
        stride=np.array(stride),
    )
    print(f"  Saved: {path}  (shape {X.shape}, label balance {y.mean()*100:.1f}% contact)")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw telemetry CSV into windowed numpy arrays with a held-out test set."
    )
    parser.add_argument("--input",         required=True,  help="Path to raw CSV (e.g. data_low.csv)")
    parser.add_argument("--output_prefix", required=True,  help="Output prefix (e.g. processed_low → processed_low_train.npz + processed_low_test.npz)")
    parser.add_argument("--window_size",   type=int,   default=50,  help="Window length in timesteps (default: 50)")
    parser.add_argument("--stride",        type=int,   default=10,  help="Step between windows (default: 10)")
    parser.add_argument("--smooth",        type=int,   default=5,   help="Rolling mean size; 1=off (default: 5)")
    parser.add_argument("--test_frac",     type=float, default=0.2, help="Fraction of episodes held out for testing (default: 0.2)")
    parser.add_argument("--seed",          type=int,   default=42,  help="Random seed for episode shuffle (default: 42)")
    args = parser.parse_args()

    print(f"\n=== Preprocessing: {args.input} ===")

    # 1. Load
    df = load_and_clean(args.input)

    # 2. Episode-level train/test split (before any windowing)
    df_train, df_test = split_episodes(df, test_frac=args.test_frac, seed=args.seed)

    # 3. Drop iteration column now that split is done
    df_train = drop_iter_col(df_train)
    df_test  = drop_iter_col(df_test)

    # 4. Smooth each split independently
    df_train = smooth_signals(df_train, window=args.smooth)
    df_test  = smooth_signals(df_test,  window=args.smooth)

    # 5. Window each split
    X_train, y_train, feature_names = make_windows(df_train, args.window_size, args.stride)
    X_test,  y_test,  _             = make_windows(df_test,  args.window_size, args.stride)

    # 6. Normalize: fit scaler ONLY on train, apply to both
    X_train, scaler = normalize(X_train, scaler=None)
    X_test,  _      = normalize(X_test,  scaler=scaler)

    print(f"\n  Features ({len(feature_names)}): {feature_names}")
    print(f"  Train windows: {X_train.shape[0]}  |  Test windows: {X_test.shape[0]}")

    # 7. Save
    train_path = args.output_prefix + "_train.npz"
    test_path  = args.output_prefix + "_test.npz"
    save_split(X_train, y_train, feature_names, args.window_size, args.stride, train_path)
    save_split(X_test,  y_test,  feature_names, args.window_size, args.stride, test_path)

    # Save scaler (fit on train only) for future use
    scaler_path = args.output_prefix + "_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Scaler (fit on train) saved -> {scaler_path}")

    print("=== Done ===\n")


if __name__ == "__main__":
    main()
