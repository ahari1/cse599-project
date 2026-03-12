#!/usr/bin/env bash
# collect_all.sh — Collect data for all three variability tiers
# Usage: bash collect_all.sh [iterations] [steps]
# Defaults: 500 iterations, 150 steps per iteration

set -e

ITERATIONS=${1:-500}
STEPS=${2:-150}

echo "============================================="
echo "CSE599 — Data Collection"
echo "  iterations / tier : $ITERATIONS"
echo "  steps / iteration  : $STEPS"
echo "============================================="

for TIER in low medium high; do
    OUTPUT="data_${TIER}.csv"
    echo ""
    echo ">>> Collecting tier: $TIER  ->  $OUTPUT"
    python data_collection.py \
        --variability "$TIER" \
        --iterations "$ITERATIONS" \
        --steps "$STEPS" \
        --output "$OUTPUT"
done

echo ""
echo "============================================="
echo "All tiers collected:"
ls -lh data_low.csv data_medium.csv data_high.csv 2>/dev/null || true
echo "============================================="
