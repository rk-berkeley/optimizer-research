"""
train_standard.py — Training loop, models, and optimizer dispatch.

Defines:
  - CifarCNN:   small CNN (~165k params) for CIFAR-10 classification.
  - TinyGPT:    3-layer decoder-only transformer (~220k params) for
                character-level language modeling on TinyShakespeare.
  - make_optimizer: constructs the (main, aux) optimizer pair for any
                    supported optimizer name.
  - train_one:  single experiment: one task × one optimizer × one seed.

Supported optimizers:
  "adamw"    — standard PyTorch AdamW (all parameters)
  "muon"     — Muon (2D/4D weights) + AdamW (rest)
  "adamuon"  — AdaMuon (2D/4D weights) + AdamW (rest)
  "dion"     — Dion (2D/4D weights) + AdamW (rest)
  "muonw"    — MuonW (2D/4D weights) + AdamW (rest)
  "muonclip" — Muon (2D/4D weights) + AdamW (rest) + apply_qk_clip() hook

All Muon-family optimizers apply only to 2D and 4D weight matrices; biases,
layer norms, embeddings, and the output head are handled by auxiliary AdamW.

Typical usage (see run_standard.py for CLI):

    from train_standard import train_one
    log = train_one("cifar", "muon", lr=3e-3, muon_lr=2e-2, n_steps=400)
    # log["val"][-1]["acc"] -> final validation accuracy
"""

import os
import sys
import time
import json
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from optimizers import Muon, AdaMuon
from muon_variants import Dion, MuonW, apply_qk_clip
from standard_data import (
    CIFAR10Dataset, ShakespeareDataset, SHAKESPEARE_VOCAB_SIZE,
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class CifarCNN(nn.Module):
    """Three-block CNN for CIFAR-10 (32×32 RGB, 10 classes).

    Architecture: Conv-BN-ReLU-MaxPool × 3, then a linear classifier.
    Three max-pools bring 32 → 16 → 8 → 4, so the flatten dimension is
    width*4 * 4 * 4. At width=24: ~165k parameters.

    Conv weights are 4D (out, in, kH, kW); Muon handles them by reshaping
    to (out, in*kH*kW) for orthogonalization, then reshaping back.
    """

    def __init__(self, num_classes: int = 10, width: int = 24):
        super().__init__()
        self.conv1 = nn.Conv2d(3,       width,   3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width,   width*2, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(width * 2)
        self.conv3 = nn.Conv2d(width*2, width*4, 3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(width * 4)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc    = nn.Linear(width * 4 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        return self.fc(x.flatten(1))


class TinyGPT(nn.Module):
    """Decoder-only transformer for character-level language modeling.

    Default config (d_model=96, n_layers=3, n_heads=4): ~220k parameters.
    Large enough that the optimizer choice is meaningful; small enough to
    train in minutes on CPU.

    During the forward pass, records per-block max pre-softmax attention
    logits in self._max_logits for use by apply_qk_clip().

    Args:
        vocab_size: number of characters in the vocabulary.
        d_model: hidden dimension.
        n_heads: number of attention heads.
        n_layers: number of transformer blocks.
        max_len: maximum sequence length (positional embedding table size).
        dropout: dropout probability (0 = off, default for these experiments).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 96,
        n_heads: int = 4,
        n_layers: int = 3,
        max_len: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleDict({
                "ln1":     nn.LayerNorm(d_model),
                "qkv":     nn.Linear(d_model, 3 * d_model, bias=False),
                "proj":    nn.Linear(d_model, d_model, bias=False),
                "ln2":     nn.LayerNorm(d_model),
                "mlp_in":  nn.Linear(d_model, 4 * d_model, bias=False),
                "mlp_out": nn.Linear(4 * d_model, d_model, bias=False),
            }))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.n_heads = n_heads
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.tok(x) + self.pos(pos)
        self._max_logits = []   # populated for MuonClip diagnostics

        for blk in self.blocks:
            # --- Self-attention ---
            n = blk["ln1"](h)
            qkv = blk["qkv"](n)
            q, k, v = qkv.chunk(3, dim=-1)
            B, T, D = q.shape
            H = self.n_heads
            Dh = D // H
            q = q.view(B, T, H, Dh).transpose(1, 2)
            k = k.view(B, T, H, Dh).transpose(1, 2)
            v = v.view(B, T, H, Dh).transpose(1, 2)
            att_raw = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)
            # Record peak pre-softmax logit magnitude (used by apply_qk_clip).
            with torch.no_grad():
                self._max_logits.append(att_raw.detach().abs().max().item())
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            att = att_raw.masked_fill(mask, float("-inf")).softmax(dim=-1)
            out = (att @ v).transpose(1, 2).contiguous().view(B, T, D)
            h = h + blk["proj"](out)
            # --- MLP ---
            n = blk["ln2"](h)
            h = h + blk["mlp_out"](F.gelu(blk["mlp_in"](n)))

        h = self.ln_f(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Optimizer construction
# ---------------------------------------------------------------------------

def _split_for_muon(model: nn.Module):
    """Partition model parameters into Muon-eligible and AdamW-only sets.

    Muon handles: 2D and 4D weight matrices in hidden layers.
    AdamW handles: the output head ("head"/"fc"), embeddings ("tok"/"pos"),
                   layer norms ("ln"), and any 1D or scalar parameters.

    Returns:
        (muon_params, adam_params): two lists of nn.Parameter.
    """
    muon_p, adam_p = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        excluded = (
            "head" in name or "fc" in name
            or "tok" in name or "pos" in name
            or "ln" in name or p.ndim < 2
        )
        (adam_p if excluded else muon_p).append(p)
    return muon_p, adam_p


def make_optimizer(
    opt_name: str,
    model: nn.Module,
    lr: float,
    muon_lr: float = None,
    ns_steps: int = 5,
    weight_decay: float = 0.0,
) -> list:
    """Construct the optimizer list for a given optimizer name.

    Returns a list of optimizers that should all be stepped together.
    For AdamW this is a single-element list; for Muon-family optimizers
    this is [main_optimizer, aux_adamw].

    Args:
        opt_name: one of "adamw", "muon", "adamuon", "dion", "muonw",
                  "muonclip".
        model: the model whose parameters to optimize.
        lr: AdamW / auxiliary lr (biases, norms, embeddings, output head).
        muon_lr: lr for the Muon-family optimizer. Defaults to lr if None.
        ns_steps: Newton-Schulz iteration count (Muon/AdaMuon/MuonW).
        weight_decay: applied to both main and auxiliary optimizers.

    Returns:
        list of torch.optim.Optimizer instances.
    """
    if muon_lr is None:
        muon_lr = lr

    if opt_name == "adamw":
        return [torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )]

    muon_p, adam_p = _split_for_muon(model)
    aux = torch.optim.AdamW(adam_p, lr=lr, weight_decay=weight_decay)

    if opt_name == "muon":
        main = Muon(muon_p, lr=muon_lr, momentum=0.95,
                    ns_steps=ns_steps, weight_decay=weight_decay)
    elif opt_name == "adamuon":
        main = AdaMuon(muon_p, lr=muon_lr, ns_steps=ns_steps,
                       weight_decay=weight_decay)
    elif opt_name == "dion":
        main = Dion(muon_p, lr=muon_lr, momentum=0.95, weight_decay=weight_decay)
    elif opt_name == "muonw":
        main = MuonW(muon_p, lr=muon_lr, momentum=0.95,
                     ns_steps=ns_steps, sigma_max=1.5, weight_decay=weight_decay)
    elif opt_name == "muonclip":
        # MuonClip = Muon optimizer + apply_qk_clip() called in the training loop.
        main = Muon(muon_p, lr=muon_lr, momentum=0.95,
                    ns_steps=ns_steps, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name!r}. "
                         "Choose from: adamw, muon, adamuon, dion, muonw, muonclip.")

    return [main, aux]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_cifar(model, loader, max_batches: int = 15):
    """Evaluate CifarCNN on the validation set.

    Returns:
        (accuracy, mean_loss) over at most max_batches * batch_size samples.
    """
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    model.train()
    return correct / max(total, 1), loss_sum / max(total, 1)


@torch.no_grad()
def eval_shake(model, loader, max_batches: int = 10):
    """Evaluate TinyGPT on the validation set.

    Returns:
        Per-token cross-entropy loss (lower is better).
    """
    model.eval()
    loss_sum = 0.0
    n_tokens = 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        logits = model(x)
        loss_sum += F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
        ).item()
        n_tokens += y.numel()
    model.train()
    return loss_sum / max(n_tokens, 1)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_one(
    task: str,
    opt_name: str,
    lr: float,
    *,
    muon_lr: float = None,
    ns_steps: int = 5,
    weight_decay: float = 1e-4,
    n_steps: int = 500,
    batch_size: int = 64,
    eval_every: int = 50,
    seed: int = 0,
    cifar_subset: int = 10_000,
    shake_seq_len: int = 64,
    qk_clip_tau: float = 10.0,
    verbose: bool = True,
) -> dict:
    """Run one training experiment and return a result log.

    Args:
        task: "cifar" (CIFAR-10 classification) or "shake" (TinyShakespeare LM).
        opt_name: optimizer name. See make_optimizer() for valid choices.
        lr: AdamW / auxiliary lr.
        muon_lr: Muon-family lr for 2D weights. Defaults to lr.
        ns_steps: Newton-Schulz iterations (Muon/AdaMuon/MuonW).
        weight_decay: weight decay applied to all parameter groups.
        n_steps: total number of gradient steps.
        batch_size: training batch size.
        eval_every: evaluate every this many steps (and at step n_steps).
        seed: random seed (controls data sampling and model init).
        cifar_subset: number of CIFAR-10 training images to use (max 50000).
        shake_seq_len: context length for TinyShakespeare in characters.
        qk_clip_tau: QK-Clip threshold τ (only used when opt_name="muonclip").
        verbose: if True, print progress to stdout.

    Returns:
        dict with keys:
          "steps":        list of int — eval step indices
          "train_loss":   list of float — EMA training loss at each eval
          "val":          list of dict — {"acc": float} (cifar) or {"loss": float} (shake)
          "wallclock":    list of float — elapsed seconds at each eval
          "max_logits":   list of float|None — peak attention logit per eval (shake only)
          "clip_events":  list of int — cumulative QK-Clip fires per eval
          "config":       dict — all hyperparameters for this run
    """
    torch.manual_seed(seed)

    if task == "cifar":
        train_ds = CIFAR10Dataset(split="train", subset=cifar_subset,
                                  augment=True, seed=seed)
        val_ds   = CIFAR10Dataset(split="test",  subset=2000, augment=False)
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, drop_last=True)
        val_loader   = DataLoader(val_ds, batch_size=128, shuffle=False)
        model = CifarCNN()
    elif task == "shake":
        train_ds = ShakespeareDataset(n_samples=2000, seq_len=shake_seq_len,
                                      split="train", seed=seed)
        val_ds   = ShakespeareDataset(n_samples=300,  seq_len=shake_seq_len,
                                      split="val",    seed=seed)
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, drop_last=True)
        val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)
        model = TinyGPT(vocab_size=SHAKESPEARE_VOCAB_SIZE)
    else:
        raise ValueError(f"Unknown task: {task!r}. Choose 'cifar' or 'shake'.")

    opts = make_optimizer(opt_name, model, lr=lr, muon_lr=muon_lr,
                          ns_steps=ns_steps, weight_decay=weight_decay)
    n_params = sum(p.numel() for p in model.parameters())

    log = {
        "steps": [], "train_loss": [], "val": [],
        "wallclock": [], "max_logits": [], "clip_events": [],
        "config": dict(
            task=task, opt=opt_name, lr=lr, muon_lr=muon_lr,
            ns_steps=ns_steps, weight_decay=weight_decay,
            n_params=n_params, seed=seed, n_steps=n_steps,
            batch_size=batch_size, cifar_subset=cifar_subset,
            shake_seq_len=shake_seq_len,
            qk_clip_tau=qk_clip_tau if opt_name == "muonclip" else None,
        ),
    }

    t0 = time.time()
    step = 0
    ema_loss = None
    cumulative_clips = 0
    train_iter = iter(train_loader)

    while step < n_steps:
        # Infinite data loop.
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x, y = batch
        for o in opts:
            o.zero_grad()
        logits = model(x)

        if task == "cifar":
            loss = F.cross_entropy(logits, y)
        else:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)
            )

        loss.backward()
        # Gradient clipping for stability (especially helpful for transformers).
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        for o in opts:
            o.step()

        # MuonClip: apply QK-Clip after the optimizer step.
        if opt_name == "muonclip" and task == "shake":
            cumulative_clips += apply_qk_clip(model, tau=qk_clip_tau)

        ema_loss = (loss.item() if ema_loss is None
                    else 0.9 * ema_loss + 0.1 * loss.item())
        step += 1

        if step % eval_every == 0 or step == n_steps:
            if task == "cifar":
                va, vl = eval_cifar(model, val_loader)
                val_record = {"acc": va, "loss": vl}
            else:
                vl = eval_shake(model, val_loader)
                val_record = {"loss": vl}

            wt = time.time() - t0
            log["steps"].append(step)
            log["train_loss"].append(ema_loss)
            log["val"].append(val_record)
            log["wallclock"].append(wt)
            log["max_logits"].append(
                max(model._max_logits) if (
                    hasattr(model, "_max_logits") and model._max_logits
                ) else None
            )
            log["clip_events"].append(cumulative_clips)

            if verbose:
                v_msg = (f"acc={val_record['acc']*100:.1f}% "
                         if "acc" in val_record else "")
                clip_msg = (f" clips={cumulative_clips}"
                            if opt_name == "muonclip" else "")
                print(f"  [{task} {opt_name:>8s} lr={lr:.0e}] "
                      f"s{step:4d}  loss={ema_loss:.3f}  "
                      f"{v_msg}vloss={val_record['loss']:.3f}  "
                      f"t={wt:.1f}s{clip_msg}")

    return log


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train one optimizer on one task and save results to JSON."
    )
    p.add_argument("--task",     required=True, choices=["cifar", "shake"])
    p.add_argument("--opt",      required=True,
                   choices=["adamw", "muon", "adamuon", "dion", "muonw", "muonclip"])
    p.add_argument("--lr",       type=float, default=3e-3)
    p.add_argument("--muon_lr", type=float, default=None)
    p.add_argument("--ns_steps", type=int,   default=5)
    p.add_argument("--n_steps",  type=int,   default=500)
    p.add_argument("--seed",     type=int,   default=0)
    p.add_argument("--out",      type=str,   required=True,
                   help="Path to output JSON file.")
    args = p.parse_args()

    log = train_one(
        args.task, args.opt,
        lr=args.lr, muon_lr=args.muon_lr,
        ns_steps=args.ns_steps, n_steps=args.n_steps, seed=args.seed,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(log, f, indent=2)
    print("Wrote", args.out)
