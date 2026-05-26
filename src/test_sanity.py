"""
test_sanity.py — Quick convergence check for Muon and AdaMuon.

Minimizes ||X @ W - Y||^2 over a 2D weight matrix W using each optimizer and
verifies that loss decreases to near-zero. Runs in ~2 seconds on CPU.

Usage:
    python src/test_sanity.py
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optimizers import Muon, AdaMuon

torch.manual_seed(0)

N, D = 256, 32
X = torch.randn(N, D)
W_true = torch.randn(D, D) / (D ** 0.5)
Y = X @ W_true

print("Sanity check: minimize ||X @ W - Y||^2 over 2D weight matrix W")
print(f"  N={N}, D={D}, 200 steps\n")

THRESHOLD = 0.01   # loss should reach this within 200 steps

all_passed = True
for name, OptCls, kw in [
    ("Muon",    Muon,    dict(lr=0.05, momentum=0.95)),
    ("AdaMuon", AdaMuon, dict(lr=0.05)),
    ("AdamW",   torch.optim.AdamW, dict(lr=0.05)),
]:
    W = torch.zeros(D, D, requires_grad=True)
    opt = OptCls([W], **kw)
    losses = []
    for _ in range(200):
        pred = X @ W
        loss = ((pred - Y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    passed = losses[-1] < THRESHOLD
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_passed = False
    print(f"  {name:10s}  loss: {losses[0]:.4f} -> {losses[-1]:.6f}  [{status}]")

print()
if all_passed:
    print("All checks passed.")
else:
    print("One or more checks FAILED.")
    sys.exit(1)
