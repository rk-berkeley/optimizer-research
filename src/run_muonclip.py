"""
run_muonclip.py — MuonClip experiments on TinyShakespeare.

Tests two QK-Clip thresholds: τ=10 (clips frequently, aggressive) and τ=30
(clips rarely, conservative). CIFAR-10 uses a CNN with no attention, so
MuonClip is identical to Muon there and is not run.

Key diagnostic recorded alongside accuracy: cumulative clip events (how many
block-steps fired the clip) and the max attention logit at each eval point.
This lets you verify the clip is active and see how logit growth is bounded.

All runs are idempotent; existing JSON files are skipped.

Usage:
    python src/run_muonclip.py

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


def run_muonclip(seeds=(0, 1, 2)):
    """Run MuonClip at τ ∈ {10, 30} on TinyShakespeare, 3 seeds each."""
    for tau in [10.0, 30.0]:
        for seed in seeds:
            tag = f"main_shake_muonclip_tau{int(tau)}_s{seed}"
            out = os.path.join(RESULTS_DIR, f"{tag}.json")
            if os.path.exists(out):
                print(f"skip {tag}")
                continue
            print(f"\n--- {tag} ---")
            t = time.time()
            log = train_one(
                "shake", "muonclip",
                lr=3e-3, muon_lr=2e-2,
                n_steps=300, eval_every=20,
                seed=seed, qk_clip_tau=tau, verbose=False,
            )
            with open(out, "w") as f:
                json.dump(log, f, indent=2)
            vloss       = log["val"][-1]["loss"]
            clips       = log["clip_events"][-1]
            max_logit   = log["max_logits"][-1]
            print(f"{tag}: vloss={vloss:.3f}  clips={clips}  "
                  f"max_logit_end={max_logit:.2f}  ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    t0 = time.time()
    run_muonclip()
    print(f"\nTotal: {(time.time() - t0) / 60:.1f} min")
