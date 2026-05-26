#!/usr/bin/env bash
# reproduce.sh — Full experiment suite for the Muon review paper.
#
# Usage:
#   bash reproduce.sh          # download data + run all experiments + figures
#   bash reproduce.sh --figs   # skip experiments, regenerate figures only
#
# Results are cached: if a JSON file already exists it is skipped, so you
# can resume an interrupted run safely.
#
# Expected runtime: ~45–60 minutes on a modern 8-core CPU (all experiments
# are CPU-only and use small models by design).
#
# Prerequisites:
#   pip install -r requirements.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR/src"
RESULTS="$SCRIPT_DIR/results"
FIGS="$SCRIPT_DIR/figures"
SCRIPTS="$SCRIPT_DIR/scripts"

mkdir -p "$RESULTS" "$FIGS"

# ── Resolve python executable ────────────────────────────────────────────────
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "ERROR: python not found. Install Python 3.9+ and run: pip install -r requirements.txt"
    exit 1
fi

# ── Download data ─────────────────────────────────────────────────────────────
if [ "${1:-}" != "--figs" ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo " STEP 0/4 — Download datasets (skipped if already present)"
    echo "════════════════════════════════════════════════════════════════"
    $PYTHON "$SCRIPTS/download_data.py"
fi

# ── Run experiments (skip if --figs only) ────────────────────────────────────
if [ "${1:-}" != "--figs" ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo " STEP 1/4 — Main comparison: AdamW / Muon / AdaMuon"
    echo "            CIFAR-10 (400 steps) + TinyShakespeare (300 steps)"
    echo "            3 seeds each"
    echo "════════════════════════════════════════════════════════════════"
    RESULTS_DIR="$RESULTS" $PYTHON "$SRC/run_standard.py" --exp main

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo " STEP 2/4 — Variants: Dion + MuonW (both tasks, 3 seeds)"
    echo "════════════════════════════════════════════════════════════════"
    RESULTS_DIR="$RESULTS" $PYTHON "$SRC/run_variants.py"

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo " STEP 3/4 — LR sensitivity + NS iteration sweeps"
    echo "            MuonClip tau=10 and tau=30 (TinyShakespeare, 3 seeds)"
    echo "════════════════════════════════════════════════════════════════"
    RESULTS_DIR="$RESULTS" $PYTHON "$SRC/run_standard.py" --exp lr
    RESULTS_DIR="$RESULTS" $PYTHON "$SRC/run_standard.py" --exp ns
    RESULTS_DIR="$RESULTS" $PYTHON "$SRC/run_muonclip.py"
fi

# ── Regenerate figures ────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo " STEP 4/4 — Generating figures"
echo "════════════════════════════════════════════════════════════════"
RESULTS_DIR="$RESULTS" FIGS_DIR="$FIGS" $PYTHON "$SRC/make_figs.py"

echo ""
echo "Done. Figures written to $FIGS/"
echo "      Results in        $RESULTS/"
