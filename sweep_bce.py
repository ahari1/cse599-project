import subprocess
import sys
import re

PYTHON = sys.executable

weights = [1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5]
medium_results = {}
high_results = {}

# Early stopping settings
MIN_EPOCHS = 6           # must run at least this many epochs
MAX_EPOCHS = 10          # max epochs for this sweep
NO_IMPROVE_EPOCHS = 3    # stop if no F1 improvement over these many epochs
F1_ABS_FLOOR = 0.25      # absolute minimum F1 to continue

def run_train_with_smart_early_stop(data_file, tier, pos_weight):
    best_f1 = 0.0
    last_improvement_ep = 0
    final_f1 = None

    for ep in range(1, MAX_EPOCHS + 1):
        cmd = [
            PYTHON, "train.py",
            "--data", data_file,
            "--tier", tier,
            "--epochs", str(ep),
            "--folds", "2",
            "--pos_weight", str(pos_weight)
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(result.stdout)

        # Parse last F1 from logs
        match = re.findall(r"f1=([0-9.]+)", result.stdout)
        if match:
            f1 = float(match[-1])
            final_f1 = f1
            if f1 > best_f1:
                best_f1 = f1
                last_improvement_ep = ep

        # Smart early stopping
        if ep >= MIN_EPOCHS:
            # 1) Absolute floor check
            if final_f1 is not None and final_f1 < F1_ABS_FLOOR:
                print(f"Early stopping tier={tier} pos_weight={pos_weight} at epoch {ep}, F1={final_f1:.4f} below absolute floor")
                break
            # 2) No improvement over last NO_IMPROVE_EPOCHS
            if ep - last_improvement_ep >= NO_IMPROVE_EPOCHS:
                print(f"Early stopping tier={tier} pos_weight={pos_weight} at epoch {ep}, no F1 improvement for {NO_IMPROVE_EPOCHS} epochs")
                break

    return final_f1

# --------------------------------------------------
# Generate smaller datasets (50 iterations) for 2-hour test
# --------------------------------------------------
for tier, fname in [("medium", "data_medium_test.csv"), ("high", "data_high_test.csv")]:
    subprocess.run([
        PYTHON, "data_collection.py",
        "--variability", tier,
        "--iterations", "50",
        "--output", fname
    ], check=True)
    subprocess.run([
        PYTHON, "preprocess.py",
        "--input", fname,
        "--output_prefix", f"processed_{tier}_test",
        "--window_size", "50",
        "--stride", "10",
        "--smooth", "5"
    ], check=True)

# --------------------------------------------------
# Sweep BCE weights with smart early stopping
# --------------------------------------------------
for w in weights:
    print(f"\n########## TESTING pos_weight = {w} ##########\n")

    # medium
    f1_medium = run_train_with_smart_early_stop("processed_medium_test_train.npz", "medium", w)
    medium_results[w] = f1_medium

    # high
    f1_high = run_train_with_smart_early_stop("processed_high_test_train.npz", "high", w)
    high_results[w] = f1_high

# --------------------------------------------------
# Print results
# --------------------------------------------------
print("\n\n==============================")
print(" BCE WEIGHT RESULTS (MEDIUM)")
print("==============================")
medium_sorted = sorted(medium_results.items(), key=lambda x: x[1] if x[1] is not None else -1, reverse=True)
for w, score in medium_sorted:
    print(f"pos_weight={w}  ->  F1={score:.4f}" if score is not None else f"pos_weight={w}  ->  F1=NA")

print("\n\n==============================")
print(" BCE WEIGHT RESULTS (HIGH)")
print("==============================")
high_sorted = sorted(high_results.items(), key=lambda x: x[1] if x[1] is not None else -1, reverse=True)
for w, score in high_sorted:
    print(f"pos_weight={w}  ->  F1={score:.4f}" if score is not None else f"pos_weight={w}  ->  F1=NA")

print("\nSweep complete with smart early stopping!")