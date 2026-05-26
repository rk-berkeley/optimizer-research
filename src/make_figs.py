"""
make_figs.py — Generate all paper figures from results/ JSON logs.

Produces (in figures/):
  main_comparison.{png,pdf}  — 2×2 grid: training loss + val metric on
                                CIFAR-10 (top) and TinyShakespeare (bottom),
                                all optimizers, ±1 SE bands over 3 seeds.
  lr_sweep.{png,pdf}         — LR sensitivity: final val loss + train loss
                                vs learning rate (TinyShakespeare, log scale).
  ns_sweep.{png,pdf}         — NS iteration count: val acc trajectory + final
                                acc / per-step time trade-off (CIFAR-10).
  logit_growth.{png,pdf}     — Max attention logit over training steps,
                                comparing plain Muon vs MuonClip τ=10
                                (TinyShakespeare).

Also prints a summary statistics table to stdout.

Usage:
    python src/make_figs.py

Override paths via environment variables:
    RESULTS_DIR   path to JSON logs   (default: ./results/)
    FIGS_DIR      output directory    (default: ./figures/)
"""

import os
import sys
import json
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(_ROOT, "results"))
FIGS_DIR    = os.environ.get("FIGS_DIR",    os.path.join(_ROOT, "figures"))
os.makedirs(FIGS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 8, "figure.dpi": 130, "savefig.dpi": 200,
    "lines.linewidth": 1.4,
    "axes.spines.top": False, "axes.spines.right": False,
})

COLORS = {
    "adamw":    "#1f77b4",
    "muon":     "#d62728",
    "adamuon":  "#2ca02c",
    "dion":     "#9467bd",
    "muonw":    "#ff7f0e",
    "muonclip": "#8c564b",
}
LABELS = {
    "adamw":    "AdamW",
    "muon":     "Muon",
    "adamuon":  "AdaMuon",
    "dion":     "Dion",
    "muonw":    "MuonW",
    "muonclip": "MuonClip",
}
ALL_OPTS = ["adamw", "muon", "adamuon", "dion", "muonw", "muonclip"]
TASKS    = ("cifar", "shake")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(pattern: str) -> dict:
    """Load all JSON files matching pattern in RESULTS_DIR."""
    out = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, pattern))):
        tag = os.path.basename(path).replace(".json", "")
        with open(path) as f:
            out[tag] = json.load(f)
    return out


def _agg(logs: dict, task: str, opt: str, key: str):
    """Aggregate runs over seeds: return (steps, mean, stderr) or None."""
    keys = [t for t in logs if f"main_{task}_{opt}_" in t]
    if not keys:
        return None
    n = min(len(logs[k]["steps"]) for k in keys)
    steps = np.array(logs[keys[0]]["steps"][:n])
    vals = []
    for k in keys:
        lg = logs[k]
        if key == "train_loss":
            vals.append(lg["train_loss"][:n])
        elif key == "val_loss":
            vals.append([v["loss"] for v in lg["val"][:n]])
        elif key == "val_acc":
            if "acc" not in lg["val"][0]:
                return None
            vals.append([v["acc"] for v in lg["val"][:n]])
        elif key == "max_logits":
            vals.append([x for x in lg.get("max_logits", [None] * n)[:n]])
    v = np.array(vals, dtype=float)
    return steps, v.mean(0), v.std(0) / np.sqrt(max(v.shape[0], 1))


# ---------------------------------------------------------------------------
# Figure 1: Main comparison (2x2 grid)
# ---------------------------------------------------------------------------

def fig_main_comparison():
    logs = _load("main_*.json")
    if not logs:
        print("No main_* logs found in", RESULTS_DIR); return

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.5))

    # CIFAR-10 — training loss
    ax = axes[0, 0]
    for opt in ALL_OPTS:
        r = _agg(logs, "cifar", opt, "train_loss")
        if r is None: continue
        s, m, se = r
        ax.plot(s, m, label=LABELS[opt], color=COLORS[opt])
        ax.fill_between(s, m - se, m + se, alpha=0.2, color=COLORS[opt])
    ax.set_title("CIFAR-10: training loss")
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.legend()

    # CIFAR-10 — validation accuracy
    ax = axes[0, 1]
    for opt in ALL_OPTS:
        r = _agg(logs, "cifar", opt, "val_acc")
        if r is None: continue
        s, m, se = r
        ax.plot(s, m * 100, label=LABELS[opt], color=COLORS[opt])
        ax.fill_between(s, (m - se) * 100, (m + se) * 100,
                         alpha=0.2, color=COLORS[opt])
    ax.set_title("CIFAR-10: validation accuracy")
    ax.set_xlabel("step"); ax.set_ylabel("acc (%)"); ax.legend()

    # TinyShakespeare — training loss
    ax = axes[1, 0]
    for opt in ALL_OPTS:
        r = _agg(logs, "shake", opt, "train_loss")
        if r is None: continue
        s, m, se = r
        ax.plot(s, m, label=LABELS[opt], color=COLORS[opt])
        ax.fill_between(s, m - se, m + se, alpha=0.2, color=COLORS[opt])
    ax.set_title("TinyShakespeare: training loss")
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.legend()

    # TinyShakespeare — validation loss
    ax = axes[1, 1]
    for opt in ALL_OPTS:
        r = _agg(logs, "shake", opt, "val_loss")
        if r is None: continue
        s, m, se = r
        ax.plot(s, m, label=LABELS[opt], color=COLORS[opt])
        ax.fill_between(s, m - se, m + se, alpha=0.2, color=COLORS[opt])
    ax.set_title("TinyShakespeare: validation loss")
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.legend()

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS_DIR, f"main_comparison.{ext}"))
    plt.close(fig)
    print("wrote main_comparison.{pdf,png}")


# ---------------------------------------------------------------------------
# Figure 2: LR sweep
# ---------------------------------------------------------------------------

def fig_lr_sweep():
    logs = _load("lr_*.json")
    if not logs:
        print("No lr_* logs found in", RESULTS_DIR); return

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    by_opt: dict = {}
    for tag, lg in logs.items():
        opt = lg["config"]["opt"]
        x = lg["config"]["lr"] if opt == "adamw" else lg["config"]["muon_lr"]
        if not lg["val"]:
            continue
        final_vloss  = lg["val"][-1]["loss"]
        final_train  = lg["train_loss"][-1]
        by_opt.setdefault(opt, []).append((x, final_vloss, final_train))

    for ax_idx, (ylabel, val_idx) in enumerate(
        [("final val loss", 1), ("final train loss", 2)]
    ):
        ax = axes[ax_idx]
        for opt in ALL_OPTS:
            if opt not in by_opt:
                continue
            pts = sorted(by_opt[opt])
            ax.plot([p[0] for p in pts], [p[val_idx] for p in pts],
                    label=LABELS[opt], color=COLORS[opt], marker="o", markersize=4)
        ax.set_xscale("log")
        ax.set_xlabel("learning rate"); ax.set_ylabel(ylabel); ax.legend()

    fig.suptitle("LR sensitivity (TinyShakespeare, 200 steps)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS_DIR, f"lr_sweep.{ext}"))
    plt.close(fig)
    print("wrote lr_sweep.{pdf,png}")


# ---------------------------------------------------------------------------
# Figure 3: NS iteration sweep
# ---------------------------------------------------------------------------

def fig_ns_sweep():
    logs = _load("ns_muon_ns*.json")
    if not logs:
        print("No ns_* logs found in", RESULTS_DIR); return

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))

    # Left: val acc trajectories coloured by NS count.
    ax = axes[0]
    pts = []
    for tag, lg in sorted(logs.items(), key=lambda kv: kv[1]["config"]["ns_steps"]):
        ns    = lg["config"]["ns_steps"]
        steps = lg["steps"]
        acc   = [v["acc"] for v in lg["val"]]
        ax.plot(steps, [a * 100 for a in acc], label=f"NS={ns}",
                color=plt.cm.viridis(ns / 11))
        if len(lg["wallclock"]) >= 2:
            per_step = ((lg["wallclock"][-1] - lg["wallclock"][0])
                        / (lg["steps"][-1] - lg["steps"][0]))
        else:
            per_step = lg["wallclock"][-1] / lg["steps"][-1]
        pts.append((ns, acc[-1] * 100, per_step))
    ax.set_xlabel("step"); ax.set_ylabel("val acc (%)")
    ax.set_title("CIFAR-10 val acc by NS iterations")
    ax.legend(ncol=2, fontsize=7)

    # Right: final acc and per-step time vs NS count (dual y-axis).
    ax = axes[1]
    pts.sort()
    xs   = [p[0] for p in pts]
    accs = [p[1] for p in pts]
    tpts = [p[2] for p in pts]
    ax.plot(xs, accs, marker="o", color="#d62728", label="final acc")
    ax.set_xlabel("Newton-Schulz iterations")
    ax.set_ylabel("final val acc (%)", color="#d62728")
    ax.tick_params(axis="y", labelcolor="#d62728")
    ax2 = ax.twinx()
    ax2.plot(xs, tpts, marker="s", color="#666666")
    ax2.set_ylabel("time per step (s)", color="#666666")
    ax2.tick_params(axis="y", labelcolor="#666666")
    ax2.spines["top"].set_visible(False)
    ax.set_title("NS iterations: accuracy vs compute")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS_DIR, f"ns_sweep.{ext}"))
    plt.close(fig)
    print("wrote ns_sweep.{pdf,png}")


# ---------------------------------------------------------------------------
# Figure 4: Attention logit growth (MuonClip diagnostic)
# ---------------------------------------------------------------------------

def fig_logit_growth():
    """Plot max attention logit over training for Muon vs MuonClip tau=10."""
    logs = _load("main_shake_*.json")
    if not logs:
        print("No main_shake_* logs found in", RESULTS_DIR); return

    fig, ax = plt.subplots(figsize=(5.0, 2.8))

    for label, match_fn, style in [
        ("Muon (no clip)",
         lambda k: "main_shake_muon_" in k and "muonw" not in k and "muonclip" not in k,
         dict(color=COLORS["muon"], linestyle="-")),
        ("MuonClip tau=10",
         lambda k: "main_shake_muonclip_tau10" in k,
         dict(color=COLORS["muonclip"], linestyle="--")),
    ]:
        keys = [k for k in logs if match_fn(k)]
        # Filter to runs that actually recorded max_logits (older logs may lack it).
        keys = [k for k in keys if logs[k].get("max_logits") and
                any(x is not None for x in logs[k]["max_logits"])]
        if not keys:
            continue
        n = min(len(logs[k]["max_logits"]) for k in keys)
        steps  = np.array(logs[keys[0]]["steps"][:n])
        logits = np.array([[x if x is not None else float("nan")
                            for x in logs[k]["max_logits"][:n]]
                           for k in keys], dtype=float)
        m  = np.nanmean(logits, axis=0)
        se = np.nanstd(logits, axis=0) / np.sqrt(max(len(keys), 1))
        ax.plot(steps, m, label=label, **style)
        ax.fill_between(steps, m - se, m + se, alpha=0.15, color=style["color"])

    ax.set_xlabel("step"); ax.set_ylabel("max pre-softmax logit")
    ax.set_title("Attention logit growth: Muon vs MuonClip")
    ax.legend()
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS_DIR, f"logit_growth.{ext}"))
    plt.close(fig)
    print("wrote logit_growth.{pdf,png}")


# ---------------------------------------------------------------------------
# Summary table (stdout)
# ---------------------------------------------------------------------------

def summary_table():
    logs = _load("main_*.json")
    print(f"\n{'task':<8} {'opt':<10} {'train_loss':<20} {'val_metric':<22} {'wall_s':<10}")
    print("-" * 72)
    for task in TASKS:
        for opt in ALL_OPTS:
            keys = [k for k in logs if f"main_{task}_{opt}_" in k]
            if not keys:
                continue
            train_l, vals, wcs = [], [], []
            vlabel = "?"
            for k in keys:
                lg = logs[k]
                if not lg["val"]:
                    continue
                train_l.append(lg["train_loss"][-1])
                if "acc" in lg["val"][-1]:
                    vals.append(lg["val"][-1]["acc"] * 100); vlabel = "acc%"
                else:
                    vals.append(lg["val"][-1]["loss"]);      vlabel = "vloss"
                wcs.append(lg["wallclock"][-1])
            if not vals:
                continue
            print(
                f"{task:<8} {opt:<10} "
                f"{np.mean(train_l):.3f}±{np.std(train_l):.3f}       "
                f"{np.mean(vals):.3f}±{np.std(vals):.3f} ({vlabel})  "
                f"{np.mean(wcs):.1f}±{np.std(wcs):.1f}"
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fig_main_comparison()
    fig_lr_sweep()
    fig_ns_sweep()
    fig_logit_growth()
    summary_table()
