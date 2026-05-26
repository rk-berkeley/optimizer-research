"""
optimizers.py — Core optimizer implementations.

Implements:
  - zeropower_via_newtonschulz5: quintic Newton-Schulz orthogonalization
  - Muon: Newton-Schulz orthogonalization on 2D weight matrices (Jordan et al., 2024)
  - AdaMuon: Muon + per-element second moment (Si et al., 2025)
  - split_params_for_muon: utility to partition model parameters

Usage note: Muon and AdaMuon handle only 2D (and 4D conv) weight matrices.
All other parameters (biases, layer norms, embeddings, output head) should
be given to a separate AdamW optimizer. See train_standard.py for an example.

References:
  Jordan et al. "Muon: An optimizer for hidden layers in neural networks." (2024)
    https://kellerjordan.github.io/posts/muon/
  Si et al. "AdaMuon: Adaptive Muon Optimizer." arXiv:2507.11005 (2025)
  Liu et al. "Muon is Scalable for LLM Training." arXiv:2502.16982 (2025)
"""

import torch
from torch.optim.optimizer import Optimizer


# ---------------------------------------------------------------------------
# Newton-Schulz orthogonalization
# ---------------------------------------------------------------------------

@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Quintic Newton-Schulz iteration approximating the orthogonal polar factor.

    Given G = U S V^T (SVD), returns a matrix close to U V^T — the component
    of G that has all singular values equal to 1.

    The iteration is:
        X <- X / ||X||_F           (normalize for numerical stability)
        X <- a*X + (b*X*X^T + c*(X*X^T)^2) * X    (repeated `steps` times)
    with coefficients (a, b, c) = (3.4445, -4.7750, 2.0315) chosen so the
    quintic polynomial has a steep slope at zero, pushing small singular
    values toward 1 quickly.

    Args:
        G: 2D tensor (n x m). For n > m the computation is transposed
           internally so the inner products are over the smaller dimension.
        steps: number of Newton-Schulz iterations. 5 is sufficient for most
               weight matrices; 3 is also fine in practice (see experiments).

    Returns:
        Tensor of the same shape and dtype as G, approximately orthogonal:
        O^T O ≈ I  (when n >= m)  or  O O^T ≈ I  (when n < m).
    """
    assert G.ndim == 2, "Newton-Schulz orthogonalization expects a 2D tensor."
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.float32)
    # Frobenius norm >= spectral norm, so this normalization ensures stability.
    X = X / (X.norm() + 1e-7)
    # Operate on the smaller dimension for efficiency.
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


# ---------------------------------------------------------------------------
# Muon
# ---------------------------------------------------------------------------

class Muon(Optimizer):
    """Muon: Momentum Orthogonalized by Newton-Schulz.

    Applies Nesterov momentum, orthogonalizes the resulting update via
    Newton-Schulz iteration, then takes a scaled gradient step. All singular
    values of each update are forced to approximately 1, so no single
    direction in weight space receives a disproportionately large update.

    Only handles 2D and 4D (conv) weight tensors. Pair with an auxiliary
    AdamW for biases, layer norms, embeddings, and the output head.

    Args:
        params: 2D/4D weight tensors to optimize.
        lr: learning rate applied to the orthogonalized update.
        momentum: momentum coefficient μ (default 0.95, from reference impl).
        nesterov: use Nesterov momentum (default True, matches reference impl).
        ns_steps: number of Newton-Schulz iterations (default 5).
        weight_decay: decoupled weight decay coefficient.

    Reference:
        Jordan et al. "Muon: An optimizer for hidden layers in neural networks."
        https://kellerjordan.github.io/posts/muon/ (2024)
    """

    def __init__(
        self,
        params,
        lr: float = 2e-2,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim not in (2, 4):
                    # Only 2D matrices and 4D conv weights are handled.
                    continue

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                # Nesterov: look ahead by one momentum step.
                update = g.add(buf, alpha=momentum) if nesterov else buf

                # Reshape conv weights (out, in, kH, kW) -> (out, in*kH*kW).
                orig_shape = update.shape
                flat = update.reshape(orig_shape[0], -1) if update.ndim == 4 else update

                ortho = zeropower_via_newtonschulz5(flat, steps=ns_steps)

                # Scale factor from Liu et al. (2025): aligns update RMS across
                # layers of different sizes so the same lr works everywhere.
                n, m = ortho.shape
                scale = 0.2 * (max(n, m) ** 0.5)
                ortho = ortho.reshape(orig_shape)

                if wd != 0.0:
                    p.data.mul_(1.0 - lr * wd)
                p.data.add_(ortho, alpha=-lr * scale)

        return loss


# ---------------------------------------------------------------------------
# AdaMuon
# ---------------------------------------------------------------------------

class AdaMuon(Optimizer):
    """AdaMuon: Adaptive Muon.

    Augments Muon with a per-element second-moment estimator on the
    orthogonalized update, then applies RMS-aligned rescaling to preserve
    Muon's scale behavior. Reduces to Muon when β2 → 1.

    Note: adding v_t restores AdamW-level memory usage, undermining Muon's
    main advantage at scale. Whether the second moment helps depends on task
    and training regime (see experiments).

    Args:
        params: 2D/4D weight tensors to optimize.
        lr: learning rate.
        betas: (β1, β2) — momentum and second-moment decay rates.
        eps: numerical stability term.
        ns_steps: Newton-Schulz iteration count.
        weight_decay: decoupled weight decay.

    Reference:
        Si et al. "AdaMuon: Adaptive Muon Optimizer." arXiv:2507.11005 (2025)
    """

    def __init__(
        self,
        params,
        lr: float = 2e-2,
        betas: tuple = (0.95, 0.999),
        eps: float = 1e-8,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim not in (2, 4):
                    continue

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(g)   # first moment
                    state["v"] = torch.zeros_like(g)   # second moment

                state["step"] += 1
                t = state["step"]
                m, v = state["m"], state["v"]

                # Update first moment (momentum buffer).
                m.mul_(b1).add_(g, alpha=1.0 - b1)

                # Orthogonalize the first moment (Muon's core operation).
                orig_shape = m.shape
                m_flat = m.reshape(orig_shape[0], -1) if m.ndim == 4 else m
                ortho_flat = zeropower_via_newtonschulz5(m_flat, steps=ns_steps)
                ortho = ortho_flat.reshape(orig_shape)

                # Update second moment on the *orthogonalized* direction.
                v.mul_(b2).addcmul_(ortho, ortho, value=1.0 - b2)
                v_hat = v / (1.0 - b2 ** t)  # bias correction

                update = ortho / (v_hat.sqrt() + eps)

                # RMS-aligned rescaling: match total update norm to Muon's
                # 0.2 * sqrt(max(n, m)) so the same lr is compatible with Muon.
                shape = update.shape
                if update.ndim == 4:
                    n, mm = shape[0], shape[1] * shape[2] * shape[3]
                else:
                    n, mm = shape
                target_norm = 0.2 * (max(n, mm) ** 0.5)
                cur_norm = update.norm() + 1e-12
                update = update * (target_norm / cur_norm)

                if wd != 0.0:
                    p.data.mul_(1.0 - lr * wd)
                p.data.add_(update, alpha=-lr)

        return loss


# ---------------------------------------------------------------------------
# Parameter splitting utility
# ---------------------------------------------------------------------------

def split_params_for_muon(model):
    """Split model parameters into (muon_params, adamw_params).

    2D and 4D (conv) weight matrices go to Muon; all other parameters
    (biases, layer norms, embeddings, 1D tensors, output head) go to AdamW.

    Following the convention in the reference implementation (Jordan, 2024),
    the output head is left for AdamW even if it is 2D. For models where
    the output head is large relative to hidden layers, excluding it from
    Muon avoids orthogonalizing a potentially very wide matrix.

    Args:
        model: nn.Module.

    Returns:
        (muon_params, adamw_params): two lists of nn.Parameter.
    """
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim in (2, 4):
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return muon_params, adamw_params
