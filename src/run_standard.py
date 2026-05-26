"""
run_standard.py — Main experiment driver.

Runs up to three experiment types, saving one JSON file per run to results/:

  main  Main comparison: AdamW / Muon / AdaMuon on CIFAR-10 + TinyShakespeare
        400 steps (CIFAR) / 300 steps (Shakespeare), 3 seeds each. (18 runs)

  lr    Learning-rate sensitivity sweep on TinyShakespeare: 5 LRs × 3 optimizers
        × 1 seed = 15 runs, 200 steps each.

  ns    Newton-Schulz iteration count sweep on CIFAR-10: ns ∈ {2, 3, 5, 7, 10}
        × 1 seed = 5 runs, 250 steps each.

All runs are idempotent: if the output JSON already exists the run is skipped.
This makes it safe to resume an interrupted experiment suite.

Usage:
    python src/run_standard.py             # runs all three (default)
    python src/run_standard.py --exp main
    python src/run_standard.py --exp lr
    python src/run_standard.py --exp ns

Results directory: controlled by RESULTS_DIR env var (default: ./results/).
"""

import os
import sys
import json
import time
import argparse

# Allow running from the repo root: python src/run_standard.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_standard import train_one

RESULTS_DIR = os.environ.get(
    "RESULTS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"),
)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Default learning rates used in the main comparison.
# Muon-family entries are (aux_lr, muon_lr); AdamW is a scalar.
_DEFAULT_LR = {
    "adamw":   {"cifar": 3e-3,            "shake": 3e-3},
    "muon":    {"cifar": (3e-3, 2e-2),    "shake": (3e-3, 2e-2)},
    "adamuon": {"cifar": (3e-3, 2e-2),    "shake": (3e-3, 2e-2)},
}


def _kw(opt: str, task: str) -> dict:
    cfg = _DEFAULT_LR[opt][task]
    if opt == "adamw":
        return dict(lr=cfg, muon_lr=None)
    return dict(lr=cfg[0], muon_lr=cfg[1])


def main_comparison(seeds=(0, 1, 2)):
    """AdamW vs Muon vs AdaMuon on both tasks, 3 seeds each."""
    print("=" * 60)
    print("MAIN COMPARISON (CIFAR-10 + TinyShakespeare)")
    print("=" * 60)
    n_steps_map  = {"cifar": 400, "shake": 300}
    eval_map     = {"cifar": 25,  "shake": 20}

    for task in ["cifar", "shake"]:
        for opt in ["adamw", "muon", "adamuon"]:
            for seed in seeds:
                tag = f"main_{task}_{opt}_s{seed}"
                out = os.path.join(RESULTS_DIR, f"{tag}.json")
                if os.path.exists(out):
                    print(f"skip {tag}")
                    continue
                print(f"\n--- {tag} ---")
                t = time.time()
                log = train_one(
                    task, opt,
                    n_steps=n_steps_map[task],
                    eval_every=eval_map[task],
                    seed=seed, verbose=True,
                    **_kw(opt, task),
                )
                with open(out, "w") as f:
                    json.dump(log, f, indent=2)
                print(f"  -> wrote {out}  ({time.time() - t:.1f}s)")


def lr_sweep(seed: int = 0):
    """LR sensitivity sweep on TinyShakespeare (5 LRs × 3 optimizers)."""
    print("=" * 60)
    print("LR SWEEP (TinyShakespeare, 200 steps)")
    print("=" * 60)
    lr_grid = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

    for opt in ["adamw", "muon", "adamuon"]:
        for lr in lr_grid:
            tag = f"lr_{opt}_lr{lr:.0e}_s{seed}"
            out = os.path.join(RESULTS_DIR, f"{tag}.json")
            if os.path.exists(out):
                print(f"skip {tag}")
                continue
            print(f"\n--- {tag} ---")
            kw = (dict(lr=lr, muon_lr=None) if opt == "adamw"
                  else dict(lr=3e-3, muon_lr=lr))
            log = train_one("shake", opt, n_steps=200, eval_every=25,
                            seed=seed, verbose=False, **kw)
            with open(out, "w") as f:
                json.dump(log, f, indent=2)
            print(f"  -> final vloss {log['val'][-1]['loss']:.3f}")


def ns_sweep(seed: int = 0):
    """Newton-Schulz iteration sweep on CIFAR-10 (ns ∈ {2, 3, 5, 7, 10})."""
    print("=" * 60)
    print("NS ITERATION SWEEP (CIFAR-10, Muon, 250 steps)")
    print("=" * 60)

    for ns in [2, 3, 5, 7, 10]:
        tag = f"ns_muon_ns{ns}_s{seed}"
        out = os.path.join(RESULTS_DIR, f"{tag}.json")
        if os.path.exists(out):
            print(f"skip {tag}")
            continue
        print(f"\n--- {tag} ---")
        log = train_one("cifar", "muon", lr=3e-3, muon_lr=2e-2,
                        ns_steps=ns, n_steps=250, eval_every=25,
                        seed=seed, verbose=False)
        with open(out, "w") as f:
            json.dump(log, f, indent=2)
        # Compute per-step time excluding startup overhead.
        steps = log["steps"]
        wc    = log["wallclock"]
        if len(steps) >= 2:
            per_step = (wc[-1] - wc[0]) / (steps[-1] - steps[0])
        else:
            per_step = wc[-1] / steps[-1]
        acc = log["val"][-1]["acc"]
        print(f"  -> final acc {acc*100:.1f}%  per-step {per_step:.3f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", choices=["main", "lr", "ns", "all"], default="all",
                   help="Which experiment to run (default: all).")
    args = p.parse_args()

    t0 = time.time()
    if args.exp in ("main", "all"):
        main_comparison()
    if args.exp in ("lr", "all"):
        lr_sweep()
    if args.exp in ("ns", "all"):
        ns_sweep()
    print(f"\nTotal wallclock: {(time.time() - t0) / 60:.1f} min")
