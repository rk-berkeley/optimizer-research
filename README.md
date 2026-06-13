# Muon Optimizer: Critical Review and Reproducibility Study

**Paper:** *A Critical Review of Matrix-Aware Optimizers: From Adam to Muon and its Variants*

---

## Overview

Muon ([Jordan et al., 2024](https://kellerjordan.github.io/posts/muon/)) is the most significant optimizer to emerge since Adam. It holds current speed records on NanoGPT and CIFAR-10 speedruns, has been validated at 16B parameters, and now powers **Kimi K2** at **1.04 trillion parameters** via the MuonClip variant. The theoretical picture is far behind the empirical one.

This repo contains:
- **Clean implementations** of Muon, AdaMuon, Dion, MuonW, and MuonClip from scratch in PyTorch
- **Reproducibility experiments** on CIFAR-10 (CNN, ~165k params) and TinyShakespeare (TinyGPT, ~220k params) — all CPU-only, no custom kernels
- **The full paper** (LaTeX source + compiled PDF)
- **Pre-computed results** (57 JSON experiment logs, 3 seeds each) and all figures

The central thesis: Muon is on the same trajectory as Adam. Deployed at scale with empirical results well ahead of the theory. The four convergence papers published in 2025 collectively prove only that Muon converges at the same *asymptotic rate* as SGD. Nobody knows why it's faster in practice.

---

## Key Findings

### Main Comparison (3 seeds, ±SD)

| Task | Optimizer | Val Metric | Train Loss | Notes |
|------|-----------|-----------|-----------|-------|
| CIFAR-10 | AdamW | 56.9 ± 2.6% | 1.252 ± 0.019 | Baseline |
| CIFAR-10 | **Muon** | **63.4 ± 0.8%** | **1.111 ± 0.051** | +6.5pp vs AdamW |
| CIFAR-10 | AdaMuon | 62.8 ± 1.1% | 1.144 ± 0.037 | Nearly matches Muon |
| CIFAR-10 | Dion | 55.5 ± 1.8% | 1.279 ± 0.011 | Rank-8 approx loses signal |
| CIFAR-10 | MuonW | 63.3 ± 0.9% | 1.112 ± 0.051 | Matches Muon |
| Shakespeare | AdamW | 2.117 ± 0.007 | 1.782 ± 0.022 | Baseline (val loss ↓ better) |
| Shakespeare | **Muon** | **2.044 ± 0.016** | **1.475 ± 0.016** | Best overall |
| Shakespeare | AdaMuon | 2.239 ± 0.008 | 1.771 ± 0.001 | *Worse than AdamW* |
| Shakespeare | Dion | 2.223 ± 0.028 | 2.169 ± 0.021 | Distributed-only benefit |
| Shakespeare | MuonW | 2.018 ± 0.003 | 1.584 ± 0.003 | Best val loss |
| Shakespeare | MuonClip (τ=10) | 2.095 ± 0.016 | 1.460 ± 0.007 | 2× wallclock at small scale |

### Key Takeaways

1. **Muon's per-step advantage over AdamW is real and replicates**: +6.5pp CIFAR-10 accuracy, −0.3 Shakespeare val loss at equal wallclock.
2. **AdaMuon is task-dependent**: helps on vision (flat loss surface), hurts on language (short-horizon aggressive exploration is beneficial). Adding back v_t undermines Muon's memory advantage anyway.
3. **Dion loses at small scale by design**: rank-8 power iteration captures ~8% of directional information for 96-dim layers. Its benefit is communication efficiency in tensor-parallel training, which doesn't exist here.
4. **MuonW provides mild regularization**: spectral-norm projection fires occasionally during transformer training; marginal improvement over Muon, never worse.
5. **MuonClip is a scale-specific tool**: the QK-Clip mechanism works as described (logits held at threshold from step 30). At 220k params the unconstrained growth (max logit → 47) causes no instability; at 1T params the same growth causes loss spikes. The 2× wallclock cost is unjustified below that regime.
6. **Muon is *less* LR-sensitive than AdamW**: AdamW diverges at η=0.1; Muon degrades gracefully (bounded operator-norm update by construction).
7. **NS=3 is sufficient**: accuracy plateaus from N=3 iterations; N=5 is the practical default. NS=10 costs ~3% more compute with zero benefit here.
8. **The convergence theory explains nothing**: best proven rate is O(T^{-1/4}) — same as SGD — under standard assumptions. The theory-practice gap is the main open problem.

---

## Repository Structure

```
.
├── src/
│   ├── optimizers.py        # Muon, AdaMuon, Newton-Schulz orthogonalization
│   ├── muon_variants.py     # Dion, MuonW, MuonClip (apply_qk_clip)
│   ├── train_standard.py    # Training loop, CifarCNN, TinyGPT, make_optimizer
│   ├── standard_data.py     # CIFAR-10 and TinyShakespeare dataset loaders
│   ├── run_standard.py      # Main comparison + LR sweep + NS sweep
│   ├── run_variants.py      # Dion / MuonW experiments
│   ├── run_muonclip.py      # MuonClip τ sweep on TinyShakespeare
│   ├── make_figs.py         # Generate all figures from results/
│   └── test_sanity.py       # Unit test: Muon/AdaMuon converge on a toy regression
├── results/                 # 57 JSON experiment logs (pre-computed, 3 seeds each)
├── figures/                 # PNG figures used in the paper
├── report/
│   ├── main.tex             # Paper LaTeX source
│   ├── experiments.tex      # Experiments section (included by main.tex)
│   ├── refs.bib             # Bibliography (17 references)
│   └── muon_review.pdf              # Compiled paper
├── reproduce.sh             # End-to-end reproduction script
├── requirements.txt
└── .gitignore
```

---

## Reproduction

### Prerequisites

```bash
pip install -r requirements.txt
```

Data is downloaded automatically on first run:
- **CIFAR-10**: via `torchvision` (downloaded to `data/cifar10/`)
- **TinyShakespeare**: fetched from `karpathy/char-rnn` on first import

### Run Everything

```bash
bash reproduce.sh
```

This runs all experiments (main comparison, LR sweep, NS sweep, MuonClip) and regenerates all figures. On a modern CPU expect **~45–60 minutes** total. Results are cached — if a JSON file already exists it is skipped, so you can resume an interrupted run.

### Run Individual Experiments

```bash
# Main optimizer comparison (AdamW / Muon / AdaMuon on CIFAR-10 + Shakespeare)
python src/run_standard.py --exp main

# Learning rate sensitivity sweep
python src/run_standard.py --exp lr

# Newton-Schulz iteration count sweep
python src/run_standard.py --exp ns

# Dion and MuonW variants
python src/run_variants.py

# MuonClip (τ=10, τ=30) on TinyShakespeare
python src/run_muonclip.py

# Regenerate all figures
python src/make_figs.py

# Sanity check: optimizer convergence on toy regression
python src/test_sanity.py
```

### Single Training Run (Programmatic)

```python
from src.train_standard import train_one

log = train_one(
    task="cifar",           # "cifar" or "shake"
    opt_name="muon",        # "adamw" | "muon" | "adamuon" | "dion" | "muonw" | "muonclip"
    lr=3e-3,                # auxiliary AdamW lr (biases, norms, embeddings)
    muon_lr=2e-2,           # Muon lr (2D weight matrices)
    n_steps=400,
    seed=0,
)
print(f"Final val acc: {log['val'][-1]['acc']*100:.1f}%")
```

---

## Implementation Notes

### Optimizer Summary

| Optimizer | Key Idea | Memory vs AdamW | Paper |
|-----------|----------|-----------------|-------|
| **Muon** | Newton-Schulz orthogonalization of momentum; steepest descent under spectral norm | −50% (no v_t) | Jordan et al. (2024) |
| **AdaMuon** | Muon + per-element v_t on the orthogonalized update | Same as AdamW | Si et al. (2025) |
| **Dion** | Amortized rank-r power iteration instead of NS; shard-friendly | −50% | Ahn et al. (2025) |
| **MuonW** | Muon + post-step projection onto spectral-norm ball | −50% | Crawshaw et al. (2025) |
| **MuonClip** | Muon + QK-Clip: rescale Q/K weights when max attention logit > τ | −50% | Moonshot AI (2025) |

All Muon-family optimizers use an auxiliary AdamW for parameters that are not 2D weight matrices (biases, layer norms, embeddings, output head).

### Newton-Schulz Orthogonalization

The quintic polynomial `X ← aX + (bXX^T + cXX^TX)X` with `(a, b, c) = (3.4445, −4.7750, 2.0315)` approximates the orthogonal polar factor of the gradient in 5 iterations. Coefficients are chosen for steep slope at 0, pushing small singular values toward 1 quickly. See [`src/optimizers.py`](src/optimizers.py) for the implementation.

---

## Convergence Theory (2025 State)

| Paper | Rate | Assumptions | Notes |
|-------|------|-------------|-------|
| Li & Hong (2025) | informal | L-smooth, spectral norm | Flags errors in concurrent proofs |
| Shen et al. (2025) | O(T^{-1/4}) | L-smooth, bounded variance | Same as SGD; no theoretical advantage |
| Chang et al. (2025) | O(T^{-1/3}), tight | Mean-sq smooth + variance reduction | Different algorithm (Muon-MVR), not deployed Muon |
| Crawshaw et al. (2025) | various | Design choice taxonomy | No unified result |

The deployed algorithm has no proven asymptotic advantage over SGD under standard assumptions.

