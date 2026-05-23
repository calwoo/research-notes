# 🤖 LLM Pruning: Movement Pruning, SparseGPT, and Wanda

## Table of Contents

- [[#1. 💡 The LLM Compression Challenge|1. The LLM Compression Challenge]]
- [[#2. 🚶 Movement Pruning|2. Movement Pruning]]
  - [[#2.1 The Fine-Tuning Saliency Problem|2.1 The Fine-Tuning Saliency Problem]]
  - [[#2.2 Movement Score Derivation|2.2 Movement Score Derivation]]
  - [[#2.3 💻 PyTorch: Movement Pruning Mask|2.3 PyTorch: Movement Pruning Mask]]
- [[#3. ⚡ SparseGPT: OBS at Scale|3. SparseGPT: OBS at Scale]]
  - [[#3.1 The Layer-Wise Reconstruction Objective|3.1 The Layer-Wise Reconstruction Objective]]
  - [[#3.2 The OBC Algorithm|3.2 The OBC Algorithm]]
  - [[#3.3 Cholesky-Based Efficient Implementation|3.3 Cholesky-Based Efficient Implementation]]
  - [[#3.4 💻 PyTorch: SparseGPT Layer Pruning|3.4 PyTorch: SparseGPT Layer Pruning]]
- [[#4. 🎯 Wanda: Weights AND Activations|4. Wanda: Weights AND Activations]]
  - [[#4.1 Activation-Weighted Saliency|4.1 Activation-Weighted Saliency]]
  - [[#4.2 Why Activations Matter|4.2 Why Activations Matter]]
  - [[#4.3 💻 PyTorch: Wanda Pruning|4.3 PyTorch: Wanda Pruning]]
- [[#5. 📊 Comparison at Scale|5. Comparison at Scale]]
- [[#6. 📚 References|6. References]]

---

## 1. 💡 The LLM Compression Challenge

Classical pruning methods face a fundamental obstacle at LLM scale: *retraining is infeasible*.

- GPT-3 (175B parameters) took $\approx$ 3,600 petaflop-days to train once. An IMP pipeline requiring 3–10 train–prune–retrain cycles would cost tens of thousands of GPU-years.
- The iterative magnitude pruning of [[concepts/deep-learning-engineering/sparsity-pruning/classical-pruning|Han et al.]] depends critically on the retrain step to recover accuracy after each pruning round. Without retraining, naive magnitude pruning on LLMs causes catastrophic perplexity spikes.

The LLM-era pruning question is: **Can we prune a pre-trained LLM to high sparsity in a single pass, with no weight updates, using only a small calibration dataset?**

Three methods define the state of the art:

| Method | Saliency Signal | Weight Update? | Calibration Data? | Year |
|--------|----------------|---------------|------------------|------|
| Movement Pruning | Weight × gradient during fine-tuning | Yes (fine-tuning) | Task data | 2020 |
| SparseGPT | OBS inverse-Hessian (per-layer) | Yes (one-shot layer-wise) | 128 calibration seqs | 2023 |
| Wanda | Weight magnitude × activation ℓ₂ norm | No | 128 calibration seqs | 2023 |

---

## 2. 🚶 Movement Pruning

*Sanh, Wolf, Rush (2020). "Movement Pruning: Adaptive Sparsity by Fine-Tuning." NeurIPS 2020.*

### 2.1 The Fine-Tuning Saliency Problem

When fine-tuning a pre-trained BERT-like model on a downstream task, the pre-trained weights $\hat{w}$ are not the optimal solution for the task — they are *moved* by fine-tuning toward the task optimum. Magnitude pruning (with saliency $s_j = |w_j|$) is problematic here because it reflects the *pre-training* weight distribution, not the task-specific importance.

A weight may be large because it was important for pre-training (masked language modeling) but irrelevant for the target task (e.g., sentiment classification). Conversely, a small weight may be growing — *moving* toward a large value during fine-tuning — indicating it is becoming important for the task.

**Key insight:** During fine-tuning, the *direction of weight movement* (not its magnitude) signals task importance.

### 2.2 Movement Score Derivation

**Zeroth-order movement (soft movement pruning):** Define the movement score as:

$$s_j = \alpha_j \cdot w_j$$

where $\alpha_j$ is a learned continuous gate (initialized to 0), and $w_j$ is the weight. The total network output uses gated weights $\tilde{w}_j = \alpha_j \cdot w_j$. During fine-tuning, both $w_j$ and $\alpha_j$ are updated. At the end of fine-tuning, weights with $\alpha_j < 0$ (gate is negative) are pruned.

**First-order movement (hard movement pruning):** The movement score is the running sum of gradient-times-weight:

$$s_j^{(t)} = s_j^{(t-1)} + \frac{\partial L^{(t)}}{\partial w_j} \cdot w_j^{(t)}$$

This is the first-order Taylor approximation of the loss change from moving $w_j$ by one gradient step. Integrating over training: $\sum_t \frac{\partial L}{\partial w_j} \cdot w_j$ measures the total "work done" on weight $j$ — how much the gradient is trying to move it (in the direction it's currently pointing). Positive = moving toward larger; negative = moving toward zero.

**Magnitude vs. movement:**
- Magnitude: $s_j \propto |w_j|$ — reflects *current* size regardless of direction.
- Movement: $s_j \propto \sum_t g_j \cdot w_j$ — reflects *accumulated task-relevant signal*.

A weight that is currently large but whose gradient is negative (optimizer is pushing it toward zero) has low movement score and should be pruned.

> [!QUESTION] Exercise 1: Comparing magnitude and movement saliencies
> *This exercise constructs a case where magnitude and movement disagree.*
>
> > **Prerequisites:** [[#2.2 Movement Score Derivation|2.2 Movement Score Derivation]]
>
> Two weights after $T = 3$ fine-tuning steps:
>
> - $w_A$: initial $= 2.0$, gradients $= [-0.5, -0.5, -0.5]$, final $= 2.0 - 3 \times 0.5 \times \eta$ with $\eta = 0.1$: final $= 1.85$.
> - $w_B$: initial $= 0.1$, gradients $= [+0.3, +0.3, +0.3]$, final $= 0.1 + 3 \times 0.3 \times 0.1 = 0.19$.
>
> (a) Compute magnitude saliency at the *end* of fine-tuning for both weights.
>
> (b) Compute hard movement scores $\sum_t g_t \cdot w_t$ for both weights (use the weight value at the start of each step for simplicity).
>
> (c) Which weight does each method prune? Which is more appropriate for a downstream task?

> [!TIP]- Solution to Exercise 1
> **Key insight:** $w_A$ is large but moving *away* from the pre-trained value in the direction the optimizer wants (toward zero) — it's becoming less task-relevant. $w_B$ is small but the optimizer is growing it — it's becoming more task-relevant.
>
> **(a)** Magnitude: $|w_A| = 1.85 > |w_B| = 0.19$. Magnitude prunes $w_B$ (smaller).
>
> **(b)** Approximate weight trajectories (start of each step): $w_A^{(1)} = 2.0, w_A^{(2)} = 1.95, w_A^{(3)} = 1.90$. Movement score $w_A$: $(-0.5)(2.0) + (-0.5)(1.95) + (-0.5)(1.90) = -1.0 - 0.975 - 0.95 = -2.925$. Saliency $= |-2.925| = 2.925$.
>
> $w_B^{(1)} = 0.1, w_B^{(2)} = 0.13, w_B^{(3)} = 0.16$. Movement score $w_B$: $(0.3)(0.1) + (0.3)(0.13) + (0.3)(0.16) = 0.03 + 0.039 + 0.048 = 0.117$. Saliency $= 0.117$.
>
> **(c)** Movement prunes $w_B$ (lower movement saliency). Both methods prune $w_B$ here! But the magnitude of $w_A$'s movement score ($2.925 \gg 0.117$) correctly identifies $w_A$ as the more "active" weight — the optimizer is strongly moving it. Whether $w_A$ should be kept (because it's being used) or pruned (because it's being *shrunk toward zero*) depends on the sign: $w_A$ has negative movement (being pushed to zero) — it's a candidate for pruning by the task's signal. This is the nuance movement pruning captures with soft gates ($\alpha$).

### 2.3 💻 PyTorch: Movement Pruning Mask

```python
import torch
import torch.nn as nn


class MovementPrunedLinear(nn.Module):
    """
    Linear layer with movement-based pruning via learned soft gates.

    The effective weight is: w_eff = sigmoid(alpha) * w
    Alpha is jointly trained with w. After fine-tuning, weights with
    alpha < 0 (sigmoid(alpha) < 0.5) are pruned.

    Regularizer: L1 on sigmoid(alpha) drives gates toward 0 (sparse).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.alpha = nn.Parameter(torch.zeros(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.alpha)
        return nn.functional.linear(x, gate * self.weight, self.bias)

    def movement_regularizer(self) -> torch.Tensor:
        """ℓ₁ on gates — drives some alpha → -∞ (gate → 0), pruning those weights."""
        return torch.sigmoid(self.alpha).sum()

    def apply_mask(self, threshold: float = 0.5) -> None:
        """Hard-prune: zero out weights where sigmoid(alpha) < threshold."""
        with torch.no_grad():
            gate = torch.sigmoid(self.alpha)
            self.weight.data.mul_(gate.ge(threshold).float())
            self.alpha.data.fill_(float("inf"))  # freeze surviving gates at 1


def movement_prune_training_loss(
    model: nn.Module,
    task_loss: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Add movement regularizer to the task loss during fine-tuning."""
    reg = sum(
        m.movement_regularizer()
        for m in model.modules()
        if isinstance(m, MovementPrunedLinear)
    )
    return task_loss + lam * reg
```

---

## 3. ⚡ SparseGPT: OBS at Scale

*Frantar and Alistarh (2023). "SparseGPT: Massive Language Models Can be Accurately Pruned in One Shot." ICML 2023.*

### 3.1 The Layer-Wise Reconstruction Objective

SparseGPT reformulates pruning as a *layer-wise weight reconstruction* problem. For each linear layer $\ell$ with weight $W \in \mathbb{R}^{d_{out} \times d_{in}}$, given a calibration set of $n$ input activations $X \in \mathbb{R}^{n \times d_{in}}$:

$$\min_{W' \in \mathcal{S}_s}\; \|WX^\top - W'X^\top\|_F^2$$

where $\mathcal{S}_s$ is the set of $s$-sparse matrices. This says: find a sparse $W'$ that best approximates the *outputs* of the original layer, not just the weights themselves.

**The key simplification:** This is equivalent (after expanding) to minimizing:

$$\|W - W'\|_{H}^2 = \text{tr}\!\left((W - W') H (W - W')^\top\right), \quad H = \frac{X^\top X}{n}$$

where $\|\cdot\|_H$ is the Hessian-weighted norm. Each *row* $W_i$ of $W$ faces an independent problem:

$$\min_{W'_i : \text{sparsity}(W'_i) = s}\; (W_i - W'_i) H (W_i - W'_i)^\top$$

This is exactly the per-row OBS formulation from [[concepts/deep-learning-engineering/sparsity-pruning/classical-pruning|Classical Pruning]] §4 — the OBD/OBS math from 1993, applied to each transformer linear layer with $H = X^\top X / n$ computed from calibration data.

### 3.2 The OBC Algorithm

SparseGPT implements OBS column-by-column via the **Optimal Brain Compression (OBC)** algorithm:

**Setup:** For row $i$, let $w = W_i \in \mathbb{R}^{d_{in}}$ and $H^{-1}$ be the inverse row Hessian.

**Iterative pruning:** For each step $t = 1, \ldots, k$ (where $k = \lfloor s \cdot d_{in} \rfloor$ weights to prune):

1. Compute OBS saliency for all remaining active weights: $s_j = w_j^2 / (2 [H^{-1}]_{jj})$
2. Select $j^* = \arg\min_j s_j$ (minimum saliency)
3. Apply OBS weight correction to all remaining active weights:

$$w_j \mathrel{-}= \frac{w_{j^*}}{[H^{-1}]_{j^* j^*}} \cdot [H^{-1}]_{j j^*}, \quad \forall j \neq j^*$$

4. Zero out $w_{j^*}$ and update $H^{-1}$ via Woodbury:

$$[H^{-1}_{\text{new}}]_{ab} = [H^{-1}]_{ab} - \frac{[H^{-1}]_{a j^*} [H^{-1}]_{j^* b}}{[H^{-1}]_{j^* j^*}}$$

This is $O(d_{in}^2)$ per pruning step, $O(d_{in}^3)$ total per row — the bottleneck for large $d_{in}$.

### 3.3 Cholesky-Based Efficient Implementation

Frantar & Alistarh observe that the OBC weight updates and Hessian inverse updates can be expressed in terms of the *Cholesky factor* of $H$. Specifically, if $H = L L^\top$ (Cholesky decomposition), then the OBS weight corrections for pruning column $j$ can be read off the $j$-th column of $L^{-1}$ without ever materializing $H^{-1}$ in full.

**Practical algorithm:**
1. Compute $H = X^\top X / n$ using $n \approx 128$ calibration sequences (forward hooks).
2. Add Tikhonov damping: $H \mathrel{+}= \delta I$ for numerical stability.
3. Compute Cholesky: $H = L L^\top$.
4. Compute $H^{-1}$ via Cholesky solve (or store $L^{-1}$ directly).
5. For each row $i$ of $W$: apply sequential OBS pruning using the diagonal and columns of $H^{-1}$.

**Complexity:** $O(d_{in}^2 \cdot d_{out})$ total — dominated by the per-row sequential pruning. For OPT-175B ($d_{in} = 12{,}288$, $d_{out} = 49{,}152$ for the MLP), this takes $\approx 4.5$ hours on a single A100.

> [!INFO] Column grouping for speedup
> SparseGPT optionally prunes columns in groups of $B$ (e.g., $B = 128$) simultaneously rather than one by one. Within a group, the OBS update is applied jointly. This reduces the number of $H^{-1}$ updates from $k$ to $k/B$, providing $B\times$ speedup at the cost of a slightly less optimal pruning order. The paper reports that $B = 128$ recovers $>99\%$ of the quality of sequential pruning.

> [!QUESTION] Exercise 2: SparseGPT saliency vs. magnitude at an attention layer
> *This exercise computes SparseGPT and magnitude saliencies for a small attention projection.*
>
> > **Prerequisites:** [[#3.2 The OBC Algorithm|3.2 The OBC Algorithm]], [[concepts/deep-learning-engineering/sparsity-pruning/classical-pruning#4.3 OBS Saliency and the Weight Correction|OBS Saliency]]
>
> A value projection $W_V \in \mathbb{R}^{1 \times 3}$ (single output, 3 inputs) has weight $w = (0.8,\; 0.1,\; 0.5)^\top$. The calibration Hessian (from input activations) is:
>
> $$H = \begin{pmatrix} 4 & 0 & 0 \\ 0 & 0.01 & 0 \\ 0 & 0 & 1 \end{pmatrix}$$
>
> (a) Compute magnitude, OBD, and OBS saliencies for each of the three weights. (Since $H$ is diagonal, OBD = OBS here.)
>
> (b) For 33% sparsity (prune 1 weight), which weight does each method prune?
>
> (c) After pruning $w_2$ (the SparseGPT/OBD choice), compute the OBS weight correction for $w_1$ and $w_3$.

> [!TIP]- Solution to Exercise 2
> **Key insight:** Even with a diagonal Hessian, SparseGPT and magnitude disagree — $w_1 = 0.8$ is large but sits in the sharpest curvature direction ($H_{11} = 4$), making it the most important to keep.
>
> **(a)** Saliencies:
> - Magnitude: $|w_1| = 0.8$, $|w_2| = 0.1$, $|w_3| = 0.5$.
> - OBD: $s_1 = 0.5 \cdot 4 \cdot 0.64 = 1.28$, $s_2 = 0.5 \cdot 0.01 \cdot 0.01 = 5 \times 10^{-5}$, $s_3 = 0.5 \cdot 1 \cdot 0.25 = 0.125$.
> - OBS: same as OBD (diagonal $H$): $[H^{-1}]_{jj} = 1/H_{jj}$, so saliency $= w_j^2 H_{jj} / 2 = $ OBD.
>
> **(b)** Magnitude prunes $w_2$ (smallest magnitude). OBD/OBS also prunes $w_2$ (saliency $= 5 \times 10^{-5}$, by far the smallest). Agreement here because $w_2$ is small in *both* magnitude and curvature.
>
> **(c)** Since $H$ is diagonal, $H^{-1}$ is diagonal and $[H^{-1}]_{1,2} = [H^{-1}]_{3,2} = 0$. The OBS weight correction is $\delta w_j = -(w_2 / [H^{-1}]_{22}) \cdot [H^{-1}]_{j2} = 0$ for $j \neq 2$. **No correction needed** — for a diagonal Hessian, the off-diagonal terms vanish and the other weights don't change. This demonstrates why OBD ≈ OBS for decorrelated inputs, but they diverge when inputs are correlated ($H$ has large off-diagonal entries).

### 3.4 💻 PyTorch: SparseGPT Layer Pruning

```python
import torch
import torch.nn as nn
from typing import Optional


class SparseGPTPruner:
    """
    One-shot layer-wise pruning following SparseGPT (Frantar & Alistarh, 2023).

    Usage:
        pruner = SparseGPTPruner(layer)
        # Register hook and run calibration forward passes
        for batch in calibration_loader:
            _ = model(batch)  # triggers hook
        pruner.prune(sparsity=0.5)
        pruner.remove_hook()
    """

    def __init__(self, layer: nn.Linear, damping_factor: float = 1e-2):
        self.layer = layer
        self.damping = damping_factor
        self.H: Optional[torch.Tensor] = None
        self.n_samples = 0
        self._hook = layer.register_forward_hook(self._accumulate_H)

    def _accumulate_H(
        self,
        module: nn.Module,
        inp: tuple,
        out: torch.Tensor,
    ) -> None:
        x = inp[0].detach().float()
        if x.dim() == 3:
            x = x.reshape(-1, x.size(-1))  # (batch * seq, d_in)
        n = x.size(0)
        if self.H is None:
            d = x.size(1)
            self.H = torch.zeros(d, d, device=x.device, dtype=torch.float32)
        self.H.addmm_(x.T, x)
        self.n_samples += n

    def remove_hook(self) -> None:
        self._hook.remove()

    def prune(self, sparsity: float, block_size: int = 128) -> None:
        """
        Apply SparseGPT pruning: sequential OBS column-wise within blocks.

        Args:
            sparsity: fraction of weights to zero out
            block_size: number of columns to process jointly (trade accuracy for speed)
        """
        assert self.H is not None, "Run calibration forward passes first"
        self.remove_hook()

        d_in = self.H.size(0)
        W = self.layer.weight.data.float().clone()  # (d_out, d_in)

        # Normalize and damp
        H = self.H / self.n_samples
        H += self.damping * torch.eye(d_in, device=H.device)

        # Cholesky decomposition for stable inversion
        try:
            L = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(L)
        except torch.linalg.LinAlgError:
            H_inv = torch.linalg.pinv(H)

        n_prune_per_row = round(sparsity * d_in)
        mask = torch.ones(W.shape, dtype=torch.bool, device=W.device)

        # Process each output row independently
        for row in range(W.size(0)):
            w = W[row].clone()
            H_inv_local = H_inv.clone()
            active = torch.ones(d_in, dtype=torch.bool, device=w.device)

            for _ in range(n_prune_per_row):
                # OBS saliency for active weights
                diag = H_inv_local.diag().clamp(min=1e-8)
                saliency = w.pow(2) / (2.0 * diag)
                saliency[~active] = float("inf")
                j = int(saliency.argmin())

                # OBS weight correction
                scale = w[j] / H_inv_local[j, j].clamp(min=1e-8)
                w -= scale * H_inv_local[:, j]
                w[j] = 0.0
                active[j] = False

                # Rank-1 Woodbury update of H_inv (Schur complement formula)
                hjj = H_inv_local[j, j].clamp(min=1e-8)
                H_inv_local -= torch.outer(
                    H_inv_local[:, j], H_inv_local[j, :]
                ) / hjj

            W[row] = w
            mask[row] = active

        self.layer.weight.data = W.to(self.layer.weight.dtype)
        # Store mask for potential downstream use
        self.mask = mask

    def prune_2_4(self) -> None:
        """
        Apply NVIDIA 2:4 structured sparsity: exactly 2 zeros per 4 consecutive weights.
        Uses OBS saliency within each group of 4 to select which 2 to prune.
        """
        assert self.H is not None
        self.remove_hook()

        d_in = self.H.size(0)
        W = self.layer.weight.data.float().clone()
        H = self.H / self.n_samples
        H += self.damping * torch.eye(d_in, device=H.device)
        H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))

        for row in range(W.size(0)):
            w = W[row]
            # Process in groups of 4
            for g in range(0, d_in, 4):
                grp = slice(g, min(g + 4, d_in))
                w_grp = w[grp]
                H_inv_grp = H_inv[grp, :][:, grp]
                diag_grp = H_inv_grp.diag().clamp(min=1e-8)
                saliency = w_grp.pow(2) / (2.0 * diag_grp)
                # Prune 2 lowest-saliency within group
                n_grp = len(w_grp)
                n_prune = max(0, n_grp - 2)
                if n_prune > 0:
                    prune_local = saliency.topk(n_prune, largest=False).indices
                    w_grp[prune_local] = 0.0
            W[row] = w

        self.layer.weight.data = W.to(self.layer.weight.dtype)
```

---

## 4. 🎯 Wanda: Weights AND Activations

*Sun, Liu, Bair, Kolter (2023). "A Simple and Effective Pruning Approach for Large Language Models." arXiv:2306.11695.*

### 4.1 Activation-Weighted Saliency

Wanda (Weights AND Activations) introduces a saliency score that combines the weight magnitude with the ℓ₂ norm of the corresponding input activation:

$$s_{ij} = |W_{ij}| \cdot \|X_{:,j}\|_2$$

where $X \in \mathbb{R}^{n \times d_{in}}$ contains the input activations from the calibration set (rows = tokens, columns = features), and $\|X_{:,j}\|_2 = \sqrt{\sum_t X_{tj}^2}$ is the ℓ₂ norm of the $j$-th input feature across all calibration tokens.

**Pruning rule:** For each row $i$ (output neuron), prune the bottom-$k$ weights by $s_{ij}$, independently per row.

Crucially, Wanda requires **no weight updates** — just one forward pass to collect activations, then a single threshold operation.

### 4.2 Why Activations Matter

The motivation comes from analyzing when a weight $W_{ij}$ contributes significantly to the output:

$$y_i = \sum_j W_{ij} x_j$$

The contribution of weight $W_{ij}$ on example $t$ is $W_{ij} \cdot x_{tj}$. Its *expected contribution* magnitude (over the calibration set) is $|W_{ij}| \cdot \mathbb{E}[|x_{tj}|] \approx |W_{ij}| \cdot \|X_{:,j}\|_2 / \sqrt{n}$.

So Wanda's saliency is proportional to the *expected output contribution* — the first-order importance signal that SparseGPT's Hessian also captures (since $H_{jj} = \|X_{:,j}\|_2^2 / n$).

**Comparison with SparseGPT saliency:**
- SparseGPT: $s_{ij} = W_{ij}^2 / (2 [H^{-1}]_{jj})$. Second-order, accounts for off-diagonal correlations, applies weight correction.
- Wanda: $s_{ij} = |W_{ij}| \cdot \|X_{:,j}\|_2$. First-order, no correction, independent per-weight.

*Surprisingly*, Wanda matches or exceeds SparseGPT at 50% sparsity on most LLMs, at a fraction of the compute cost. The weight correction in SparseGPT matters most at very high sparsity (70%+) or for structured 2:4 patterns.

> [!NOTE] Connection to OBD diagonal Hessian
> Since $H_{jj} = \|X_{:,j}\|_2^2 / n$ for a linear layer, OBD saliency is:
> $$s_{ij}^{OBD} = \frac{1}{2} H_{jj} W_{ij}^2 = \frac{W_{ij}^2 \|X_{:,j}\|_2^2}{2n}$$
>
> Wanda saliency: $s_{ij}^{Wanda} = |W_{ij}| \cdot \|X_{:,j}\|_2$.
>
> The relationship: $s_{ij}^{Wanda} = \sqrt{2n \cdot s_{ij}^{OBD}}$ — Wanda is the *square root* of OBD saliency (up to a constant), which changes the pruning order when weights in the same row have varying magnitudes. Wanda is effectively a softened version of OBD that is less aggressive about removing small-weight high-curvature connections.

> [!QUESTION] Exercise 3: Wanda vs. magnitude vs. OBD
> *This exercise compares all three saliency methods on a concrete LLM weight.*
>
> > **Prerequisites:** [[#4.1 Activation-Weighted Saliency|4.1 Activation-Weighted Saliency]]
>
> A single row of an attention $W_Q$ layer has weights $w = (0.6, 0.05, 0.3)$ and calibration activation norms $\|X_{:,j}\|_2 = (0.5, 20.0, 2.0)$ for $j = 1, 2, 3$.
>
> (a) Compute magnitude, Wanda, and OBD saliencies. For OBD, use $H_{jj} = \|X_{:,j}\|_2^2 / n$ with $n = 128$.
>
> (b) For 33% sparsity (prune 1 weight), which weight does each method prune?
>
> (c) Weight $w_2 = 0.05$ is small but paired with a very large activation norm ($\|X_{:,2}\|_2 = 20$). Interpret this physically: what kind of transformer feature might have this profile?

> [!TIP]- Solution to Exercise 3
> **Key insight:** A small weight paired with a large activation norm can be critically important — the product $w_j \cdot x_j$ may dominate the output even if $w_j$ alone is small.
>
> **(a)** Magnitude: $(0.6, 0.05, 0.3)$. Wanda: $(0.6 \times 0.5, 0.05 \times 20, 0.3 \times 2) = (0.30, 1.00, 0.60)$. OBD ($H_{jj} = \|X_{:,j}\|_2^2/128$): $s_j = 0.5 \times (||X_{:,j}||_2^2/128) \times w_j^2$:
> $s_1 = 0.5 \times (0.25/128) \times 0.36 = 3.52 \times 10^{-4}$, $s_2 = 0.5 \times (400/128) \times 0.0025 = 3.91 \times 10^{-3}$, $s_3 = 0.5 \times (4/128) \times 0.09 = 1.41 \times 10^{-3}$.
>
> **(b)** Magnitude prunes $w_2$ (smallest $|w|$). Wanda prunes $w_1$ (smallest Wanda score $= 0.30$). OBD prunes $w_1$ as well (smallest OBD saliency).
>
> **(c)** A small weight $w_2 = 0.05$ paired with large activations $\|X_{:,2}\|_2 = 20$ is characteristic of *outlier dimensions* in transformer representations — a well-known phenomenon in LLMs (Dettmers et al., "LLM.int8()"). Certain hidden dimensions have persistently large activation magnitudes across all tokens, likely encoding global context or positional information. Even a small weight on such a dimension contributes $0.05 \times 20 = 1.0$ to the output on average — comparable to the contribution of $w_1 = 0.6 \times 0.5 = 0.30$. Pruning $w_2$ based on magnitude alone would be catastrophic.

### 4.3 💻 PyTorch: Wanda Pruning

```python
import torch
import torch.nn as nn
from typing import Optional


class WandaPruner:
    """
    Wanda pruning: saliency = |W_ij| * ||X_{:,j}||_2.
    Requires one forward pass on calibration data; no weight updates.
    """

    def __init__(self, layer: nn.Linear):
        self.layer = layer
        self._activation_norms: Optional[torch.Tensor] = None
        self._n_samples = 0
        self._hook = layer.register_forward_hook(self._accumulate_activations)

    def _accumulate_activations(
        self,
        module: nn.Module,
        inp: tuple,
        out: torch.Tensor,
    ) -> None:
        x = inp[0].detach().float()
        if x.dim() == 3:
            x = x.reshape(-1, x.size(-1))  # (n_tokens, d_in)
        if self._activation_norms is None:
            self._activation_norms = torch.zeros(x.size(1), device=x.device)
        # Accumulate sum of squares: ||X_{:,j}||_2^2 = sum_t X_{tj}^2
        self._activation_norms.add_(x.pow(2).sum(dim=0))
        self._n_samples += x.size(0)

    def remove_hook(self) -> None:
        self._hook.remove()

    def prune(self, sparsity: float) -> None:
        """
        Prune `sparsity` fraction of weights per output row using Wanda saliency.
        Pruning is row-local: each output neuron has its own threshold.
        """
        assert self._activation_norms is not None, "Run calibration data first"
        self.remove_hook()

        # ||X_{:,j}||_2 = sqrt(sum of squares)
        act_norms = self._activation_norms.sqrt()  # (d_in,)

        W = self.layer.weight.data.float()  # (d_out, d_in)
        n_prune = round(sparsity * W.size(1))

        # Saliency: |W_ij| * ||X_{:,j}||_2
        # Broadcast act_norms (d_in,) across output rows
        saliency = W.abs() * act_norms.unsqueeze(0)  # (d_out, d_in)

        # Per-row threshold (row-local pruning)
        thresholds = saliency.kthvalue(n_prune, dim=1).values  # (d_out,)
        mask = saliency.gt(thresholds.unsqueeze(1))  # (d_out, d_in)

        self.layer.weight.data = (W * mask.float()).to(self.layer.weight.dtype)


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: prune an entire model's linear layers with Wanda
# ──────────────────────────────────────────────────────────────────────────────

def wanda_prune_model(
    model: nn.Module,
    calibration_loader: torch.utils.data.DataLoader,
    sparsity: float,
    device: str = "cuda",
    n_calibration_batches: int = 4,
) -> None:
    """
    Prune all Linear layers in `model` using Wanda at the given sparsity.
    Runs calibration forward passes, then prunes each layer in one shot.
    """
    model.eval()
    model.to(device)

    # Register Wanda hooks on all Linear layers
    pruners = {
        name: WandaPruner(module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    }

    # Calibration forward passes (no grad needed)
    with torch.no_grad():
        for i, batch in enumerate(calibration_loader):
            if i >= n_calibration_batches:
                break
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            model(batch.to(device))

    # Prune each layer
    for name, pruner in pruners.items():
        pruner.prune(sparsity=sparsity)

    print(f"Pruned {len(pruners)} linear layers to {sparsity:.0%} sparsity")
```

---

## 5. 📊 Comparison at Scale

Frantar & Alistarh (SparseGPT) and Sun et al. (Wanda) both evaluate on OPT and LLaMA families. Key findings:

| Model | Sparsity | Method | Perplexity (WikiText2) | Dense Baseline |
|-------|----------|--------|----------------------|---------------|
| OPT-30B | 50% | Magnitude | 32.1 | 9.56 |
| OPT-30B | 50% | Wanda | 11.2 | 9.56 |
| OPT-30B | 50% | SparseGPT | 10.3 | 9.56 |
| LLaMA-7B | 50% | Magnitude | 77.6 | 5.68 |
| LLaMA-7B | 50% | Wanda | 7.26 | 5.68 |
| LLaMA-7B | 50% | SparseGPT | 6.51 | 5.68 |
| OPT-175B | 50% | SparseGPT | 9.92 | 9.56 |

**Key observations:**
1. **Magnitude pruning catastrophically fails** on LLMs at 50% sparsity (perplexity explodes). The outlier-dimension phenomenon makes raw magnitude a misleading signal.
2. **Wanda nearly matches SparseGPT** at much lower computational cost ($\sim 300\times$ faster than SparseGPT). For LLaMA-7B: Wanda $= 7.26$ vs. SparseGPT $= 6.51$ perplexity.
3. **SparseGPT scales to 175B parameters** in $< 4.5$ hours on a single A100 — demonstrating that the layerwise OBS formulation is tractable at this scale.
4. At **2:4 structured sparsity**, SparseGPT's weight correction provides a larger advantage over Wanda because the structural constraint makes the pruning problem harder and compensation more valuable.

---

## 6. 📚 References

| Reference Name | Brief Summary | Link |
|----------------|---------------|------|
| Sanh, Wolf, Rush (2020). "Movement Pruning" | Fine-tuning-adaptive saliency via learned gates; task-distribution-aware pruning | [arXiv:2005.07683](https://arxiv.org/abs/2005.07683) |
| Frantar & Alistarh (2023). "SparseGPT" | One-shot OBS at 175B scale via Cholesky inverse; 50% sparsity with minimal perplexity increase | [arXiv:2301.00774](https://arxiv.org/abs/2301.00774) |
| Sun et al. (2023). "Wanda" | Weight × activation-norm saliency; no weight update; near-SparseGPT quality at much lower cost | [arXiv:2306.11695](https://arxiv.org/abs/2306.11695) |
| Dettmers et al. (2022). "LLM.int8()" | Outlier dimensions in LLM activations; motivates activation-weighted saliency | [arXiv:2208.07339](https://arxiv.org/abs/2208.07339) |
| Blalock et al. (2020). "What is the State of Neural Network Pruning?" | Meta-survey; community lacks reproducible benchmarks; ShrinkBench | [arXiv:2003.03033](https://arxiv.org/abs/2003.03033) |
