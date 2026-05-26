"""
muon_variants.py — Three 2025 Muon variants.

Implements:
  - Dion: amortized low-rank power iteration (Ahn et al., 2025)
  - MuonW: Muon + post-step spectral-norm projection (Crawshaw et al., 2025)
  - apply_qk_clip: QK-Clip post-step hook for MuonClip (Moonshot AI, 2025)

Design notes:
  Dion targets distributed (tensor-parallel) training. Newton-Schulz requires
  dense matmuls on the full weight matrix, which is communication-expensive
  when W is sharded across devices. Dion maintains a running rank-r estimate of
  the top left singular vectors and updates it with one power iteration per
  step, reducing both FLOPs and communication. At small scale (single device)
  the rank-r approximation loses directional information; expect lower accuracy
  than full Muon unless r ≈ min(n, m).

  MuonW adds a cheap post-step projection: if the resulting weight matrix has
  spectral norm > σ_max, rescale it back. Muon controls the *update* magnitude
  but not the cumulative weight norm, which can drift during long training runs.

  MuonClip (used to pre-train Kimi K2, 1.04T parameters) is not a separate
  optimizer class — it is plain Muon plus the apply_qk_clip() post-step hook.
  The hook reads per-block max attention logits recorded during the most recent
  forward pass and rescales Q/K weight rows whenever max_logit > τ. This
  prevents softmax saturation at scale; below ~1B parameters the growth of
  max logits is benign and the clip is unnecessary.

References:
  Ahn et al. "Dion: Distributed Orthonormalized Updates." arXiv:2504.05295 (2025)
  Crawshaw et al. "An Exploration of Non-Euclidean Gradient Descent."
    arXiv:2510.09827 (2025)
  Kimi Team et al. "Kimi K2: Open Agentic Intelligence." arXiv:2507.20534 (2025)
"""

import torch
from torch.optim.optimizer import Optimizer
from optimizers import zeropower_via_newtonschulz5


# ---------------------------------------------------------------------------
# Dion
# ---------------------------------------------------------------------------

class Dion(Optimizer):
    """Dion: amortized low-rank orthogonalization via power iteration.

    Maintains a per-parameter buffer U ∈ R^{n × r} approximating the top-r
    left singular vectors of the momentum matrix. Each step:
      1. Update momentum: B = μ * B_prev + G
      2. Power iteration: Y = B @ B^T @ U; QR-orthonormalize to get U_new.
      3. Project:  V = (U_new^T @ B)^T, normalize each column.
      4. Update:   W <- W - lr * scale * U_new @ V^T

    For r = min(n, m) this is equivalent to full orthogonalization at higher
    cost; for smaller r it is a deliberate low-rank approximation suited to
    tensor-parallel distributed training where each device holds a shard of W.

    Args:
        params: 2D/4D weight tensors.
        lr: learning rate.
        momentum: momentum coefficient μ.
        rank: rank r of the low-rank approximation. Defaults to
              max(2, min(8, min(n, m) // 2)) if None.
        weight_decay: decoupled weight decay.

    Reference:
        Ahn et al. "Dion: Distributed Orthonormalized Updates."
        arXiv:2504.05295 (2025)
    """

    def __init__(self, params, lr=2e-2, momentum=0.95, rank=None, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, rank=rank, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            r_cfg = group["rank"]

            for p in group["params"]:
                if p.grad is None or p.grad.ndim not in (2, 4):
                    continue

                g = p.grad
                orig_shape = g.shape
                if g.ndim == 4:
                    g = g.reshape(orig_shape[0], -1)

                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(g)
                    n, m = g.shape
                    r = r_cfg if r_cfg is not None else max(2, min(8, min(n, m) // 2))
                    # Initialize U with random orthonormal columns.
                    U0 = torch.randn(n, r, dtype=g.dtype, device=g.device)
                    state["U"], _ = torch.linalg.qr(U0)

                # Update momentum buffer.
                state["buf"].mul_(mu).add_(g)
                B = state["buf"]
                U = state["U"]

                # One power iteration step: Y = B @ B^T @ U
                Y = B @ (B.t() @ U)  # (n, r)
                U_new, _ = torch.linalg.qr(Y)
                state["U"] = U_new

                # Project B onto the updated U basis.
                V = (U_new.t() @ B).t()                        # (m, r)
                V = V / V.norm(dim=0, keepdim=True).clamp(min=1e-7)

                # Rank-r approximate orthogonal update.
                update = U_new @ V.t()                         # (n, m)

                n, m = update.shape
                scale = 0.2 * (max(n, m) ** 0.5)
                update = update.reshape(orig_shape)

                if wd != 0.0:
                    p.data.mul_(1.0 - lr * wd)
                p.data.add_(update, alpha=-lr * scale)

        return loss


# ---------------------------------------------------------------------------
# MuonW
# ---------------------------------------------------------------------------

class MuonW(Optimizer):
    """MuonW: Muon with post-step weight spectral-norm projection.

    After each standard Muon step, estimates the spectral norm of the updated
    weight matrix (via a few steps of power iteration) and rescales it back
    down if it exceeds σ_max. For 4D conv weights, uses a Frobenius-norm
    bound as a cheaper proxy.

    The motivation: Muon bounds the operator norm of each *update* (≈ lr *
    0.2 * sqrt(max(n, m))), but the weight matrix itself can drift over many
    steps. Explicit projection provides a mild regularizer and prevents
    singular-value blowup in long training runs.

    Args:
        params: 2D/4D weight tensors.
        lr: learning rate.
        momentum: Nesterov momentum coefficient.
        ns_steps: Newton-Schulz iterations for orthogonalization.
        sigma_max: spectral-norm upper bound for weight projection.
        weight_decay: decoupled weight decay.

    Reference:
        Crawshaw et al. "An Exploration of Non-Euclidean Gradient Descent:
        Muon and its Many Variants." arXiv:2510.09827 (2025)
    """

    def __init__(self, params, lr=2e-2, momentum=0.95, ns_steps=5,
                 sigma_max=2.0, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps,
                        sigma_max=sigma_max, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            ns = group["ns_steps"]
            sigma_max = group["sigma_max"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None or p.grad.ndim not in (2, 4):
                    continue

                g = p.grad
                orig_shape = g.shape
                if g.ndim == 4:
                    g = g.reshape(orig_shape[0], -1)

                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(g)

                # Nesterov momentum + Newton-Schulz orthogonalization.
                state["buf"].mul_(mu).add_(g)
                buf = state["buf"]
                update = g.add(buf, alpha=mu)         # Nesterov lookahead
                ortho = zeropower_via_newtonschulz5(update, steps=ns)

                n, m = ortho.shape
                scale = 0.2 * (max(n, m) ** 0.5)
                ortho = ortho.reshape(orig_shape)

                if wd != 0.0:
                    p.data.mul_(1.0 - lr * wd)
                p.data.add_(ortho, alpha=-lr * scale)

                # --- Spectral-norm projection ---
                if p.data.ndim == 2:
                    # Approximate σ_max(W) via 2 steps of power iteration.
                    W = p.data
                    if "v_powiter" not in state:
                        state["v_powiter"] = torch.randn(
                            W.size(1), dtype=W.dtype, device=W.device
                        )
                        state["v_powiter"] /= state["v_powiter"].norm()
                    v = state["v_powiter"]
                    for _ in range(2):
                        u = W @ v
                        u = u / u.norm().clamp(min=1e-8)
                        v = W.t() @ u
                        v = v / v.norm().clamp(min=1e-8)
                    sigma_est = (W @ v).norm()
                    state["v_powiter"] = v
                    if sigma_est > sigma_max:
                        p.data.mul_(sigma_max / sigma_est)
                else:
                    # Conv 4D: cheaper Frobenius-norm bound.
                    fnorm = p.data.norm()
                    target = sigma_max * (p.data.numel() ** 0.5) * 0.3
                    if fnorm > target:
                        p.data.mul_(target / fnorm)

        return loss


# ---------------------------------------------------------------------------
# MuonClip: QK-Clip hook (used with plain Muon as the optimizer)
# ---------------------------------------------------------------------------

def apply_qk_clip(model, tau: float) -> int:
    """Apply QK-Clip to a TinyGPT model after a Muon optimizer step.

    Reads model._max_logits (a list of per-block max pre-softmax attention
    logits populated during the most recent forward pass). For any block
    where max_logit > τ, rescales both the Q and K row slices of that
    block's qkv weight matrix by sqrt(τ / max_logit), so that the resulting
    maximum logit is exactly τ.

    MuonClip was used to train Kimi K2 (1.04T parameters, 15.5T tokens) with
    zero loss spikes. The clip addresses a specific failure mode: Muon's
    bounded-norm updates do not prevent Q/K weights from growing, and once
    max logits exceed ~100 the softmax saturates and gradient flow through
    attention nearly stops. At small scale (≤ 1B params) max logits typically
    stay below 50 and cause no instability; the clip becomes a mild regularizer
    rather than a stability mechanism.

    Args:
        model: TinyGPT instance. Must have model.blocks (ModuleList of ModuleDict
               with a "qkv" Linear) and model.d_model (int).
        tau: logit clipping threshold. Use τ ≈ 10–30 for short training runs;
             the Kimi K2 paper uses τ = 15.

    Returns:
        Number of blocks that were actually clipped this step (useful diagnostic).
    """
    if not hasattr(model, "_max_logits") or not model._max_logits:
        return 0

    n_clipped = 0
    d_model = model.d_model

    with torch.no_grad():
        for blk, max_logit in zip(model.blocks, model._max_logits):
            if max_logit > tau:
                scale = (tau / max_logit) ** 0.5
                W = blk["qkv"].weight
                # Rows 0:d_model = Q projection; d_model:2*d_model = K projection.
                W.data[:d_model].mul_(scale)
                W.data[d_model : 2 * d_model].mul_(scale)
                n_clipped += 1

    return n_clipped
