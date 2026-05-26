"""
run_variants.py — Run Dion and MuonW experiments.

Produces main comparison results for Dion and MuonW on both CIFAR-10 and
TinyShakespeare, using the same hyperparameters and step budgets as run_standard.py
so results can be combined in a single comparison plot.

MuonClip is handled separately in run_muonclip.py (Shakespeare-only, because
the CIFAR-10 CNN has no attention mechanism).

All runs are idempotent; existing JSON files are skipped.

Usage:
    python src/run_variants.py

Results directory: controlled by RESULTS_DIR env var (default: ./results/).
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_standard import train_one

RESULTS_DIR = os.environ.get(
    "RESULTS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"),
)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Hyperparameters for variant experiments.
# Same structure as run_standard.py: (aux_lr, muon_lr) per task.
_LR = {
    "dion":  {"cifar": (3e-3, 2e-2), "shake": (3e-3, 2e-2)},
    "muonw": {"cifar": (3e-3, 2e-2), "shake": (3e-3, 2e-2)},
}
_N_STEPS  = {"cifar": 400, "shake": 300}
_EVAL_MAP = {"cifar": 25,  "shake": 20}


def run_variants(seeds=(0, 1, 2)):
    """Run Dion and MuonW on both tasks, 3 seeds each."""
    for opt in ["dion", "muonw"]:
        for task in ["cifar", "shake"]:
            aux_lr, mu_lr = _LR[opt][task]
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
                    lr=aux_lr, muon_lr=mu_lr,
                    n_steps=_N_STEPS[task],
                    eval_every=_EVAL_MAP[task],
                    seed=seed, verbose=False,
                )
                with open(out, "w") as f:
                    json.dump(log, f, indent=2)
                v = log["val"][-1]
                msg = (f"acc={v['acc']*100:.1f}%"
                       if "acc" in v else f"vloss={v['loss']:.3f}")
                print(f"{tag}: {msg}  ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    t0 = time.time()
    run_variants()
    print(f"\nTotal: {(time.time() - t0) / 60:.1f} min")
