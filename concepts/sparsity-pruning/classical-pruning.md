# 🔪 Classical Pruning: OBD, OBS, and Iterative Magnitude Pruning

## Table of Contents

- [[#1. 💡 The Over-Parameterization Puzzle|1. The Over-Parameterization Puzzle]]
- [[#2. 📐 The Second-Order Taylor View of Weight Saliency|2. The Second-Order Taylor View of Weight Saliency]]
  - [[#2.1 Setup and Notation|2.1 Setup and Notation]]
  - [[#2.2 The Taylor Approximation at a Minimum|2.2 The Taylor Approximation at a Minimum]]
  - [[#2.3 The Pruning Constraint|2.3 The Pruning Constraint]]
- [[#3. 🔬 Optimal Brain Damage|3. Optimal Brain Damage]]
  - [[#3.1 The Diagonal Hessian Approximation|3.1 The Diagonal Hessian Approximation]]
  - [[#3.2 OBD Saliency Criterion|3.2 OBD Saliency Criterion]]
  - [[#3.3 Computing the Diagonal Hessian|3.3 Computing the Diagonal Hessian]]
  - [[#3.4 💻 PyTorch: OBD Saliency Scorer|3.4 PyTorch: OBD Saliency Scorer]]
- [[#4. 🧠 Optimal Brain Surgeon|4. Optimal Brain Surgeon]]
  - [[#4.1 The Full Inverse-Hessian Framework|4.1 The Full Inverse-Hessian Framework]]
  - [[#4.2 KKT Derivation of the Weight Update|4.2 KKT Derivation of the Weight Update]]
  - [[#4.3 OBS Saliency and the Weight Correction|4.3 OBS Saliency and the Weight Correction]]
  - [[#4.4 When OBD and OBS Diverge|4.4 When OBD and OBS Diverge]]
  - [[#4.5 💻 PyTorch: Layer-wise OBS|4.5 PyTorch: Layer-wise OBS]]
- [[#5. 📏 Magnitude-Based Pruning|5. Magnitude-Based Pruning]]
  - [[#5.1 The Zeroth-Order Approximation|5.1 The Zeroth-Order Approximation]]
  - [[#5.2 Why It Works Anyway|5.2 Why It Works Anyway]]
  - [[#5.3 💻 PyTorch: Magnitude Pruner|5.3 PyTorch: Magnitude Pruner]]
- [[#6. 🔄 Iterative Magnitude Pruning|6. Iterative Magnitude Pruning]]
  - [[#6.1 The Train → Prune → Retrain Loop|6.1 The Train → Prune → Retrain Loop]]
  - [[#6.2 Global vs. Layer-wise Thresholds|6.2 Global vs. Layer-wise Thresholds]]
  - [[#6.3 💻 PyTorch: Full IMP Pipeline|6.3 PyTorch: Full IMP Pipeline]]
- [[#7. 📚 References|7. References]]

---

## 1. 💡 The Over-Parameterization Puzzle

Modern neural networks are wildly over-parameterized. ResNet-50 has $\approx 25$M weights to learn a mapping from ImageNet images ($224\times224\times3 \approx 150$K real-valued inputs) to 1000 class probabilities. GPT-3 has 175B parameters. The *effective* degrees of freedom needed to express these functions is almost certainly orders of magnitude smaller.

The empirical observation motivating pruning is stark: after training, a large fraction of weights are either zero or negligibly small, and *removing* them — if done carefully — causes little or no accuracy drop. [Han et al. (2015)](https://arxiv.org/abs/1506.02626) removed 80–90% of weights from AlexNet and VGG-16 with no loss in top-5 accuracy. Why?

The answer lies in the geometry of the loss surface near the trained minimum. Weights that are unimportant correspond to *flat directions* of the loss — the Hessian has small eigenvalues in those directions, and the loss is insensitive to perturbations along them. The formal machinery of this observation is a second-order Taylor expansion.

> [!INFO] Connection to double descent and generalization
> Over-parameterization is not a bug — it is the mechanism behind benign overfitting and the double-descent phenomenon. Over-parameterized models interpolate training data while generalizing, partly because their implicit bias (gradient descent on smooth loss) selects solutions with small weight norms, which live in flat regions of the loss surface. Pruning exploits exactly this flatness.

---

## 2. 📐 The Second-Order Taylor View of Weight Saliency

### 2.1 Setup and Notation

Let $w^* \in \mathbb{R}^P$ denote the trained weight vector ($P$ is total parameter count). We want to remove a single weight $w_q^*$ — i.e., hard-constrain it to zero — and ask: how much does the loss increase?

After the deletion, the weight vector shifts by $\delta w \in \mathbb{R}^P$, where $\delta w_q = -w_q^*$ (the $q$-th weight must become zero) and the remaining components $\delta w_i, i \neq q$ represent any compensatory adjustments we make to other weights.

### 2.2 The Taylor Approximation at a Minimum

The loss change is:

$$\delta L = L(w^* + \delta w) - L(w^*) \approx \underbrace{g^\top \delta w}_{\text{linear}} + \underbrace{\frac{1}{2}\, \delta w^\top H\, \delta w}_{\text{quadratic}}$$

where $g = \nabla_w L\big|_{w^*}$ and $H = \nabla^2_w L\big|_{w^*}$ is the Hessian.

**Key simplification.** At a converged training minimum, the first-order optimality condition gives $g \approx 0$. The linear term vanishes, leaving:

$$\boxed{\delta L \approx \frac{1}{2}\, \delta w^\top H\, \delta w}$$

*All classical pruning criteria are approximations of this single expression under different assumptions about* $H$ *and* $\delta w$.

> [!WARNING] Finite-time training
> In practice, $g \neq 0$ — SGD never reaches an exact minimum. The linear term contributes $g_q \delta w_q$ to the loss change when we prune weight $q$. For well-trained models, this is small relative to the quadratic term, but it is non-negligible for early-stopped or partially-trained models. OBD/OBS are most reliable when training has converged.

### 2.3 The Pruning Constraint

*Pruning weight* $q$ *means setting* $w_q^* + \delta w_q = 0$, i.e.:

$$e_q^\top \delta w = -w_q^*$$

where $e_q \in \mathbb{R}^P$ is the $q$-th standard basis vector. Different pruning methods correspond to different choices about what to do with the remaining $\delta w_i, i \neq q$:

| Method | Assumption on $\delta w$ | Assumption on $H$ |
|--------|------------------------|-------------------|
| OBD | $\delta w_i = 0$ for $i \neq q$ | $H$ diagonal |
| OBS | $\delta w$ minimizes $\delta L$ | $H$ full (exact) |
| Magnitude | ignore $H$ entirely | — |

> [!QUESTION] Exercise 1: The linear term matters
> *This exercise quantifies when the $g \approx 0$ assumption fails.*
>
> > **Prerequisites:** [[#2.2 The Taylor Approximation at a Minimum|2.2 The Taylor Approximation at a Minimum]]
>
> A model is partially trained. Weight $q$ has gradient $g_q = -0.5$, Hessian diagonal $H_{qq} = 2.0$, and current value $w_q^* = 0.3$. Assuming no weight compensation and diagonal $H$, compute (a) the linear term's contribution to $\delta L_q$, (b) the quadratic term's contribution, and (c) the ratio. What does this say about applying OBD to under-trained models?

> [!TIP]- Solution to Exercise 1
> **Key insight:** At a true minimum $g = 0$ and the linear term vanishes; away from a minimum, the linear term can dominate.
>
> **(a)** Linear term: $g_q \delta w_q = (-0.5) \cdot (-0.3) = +0.15$ (pruning *decreases* loss — the weight is pushing us in the wrong direction!)
>
> **(b)** Quadratic term: $\frac{1}{2} H_{qq} w_q^2 = \frac{1}{2} \cdot 2.0 \cdot 0.09 = +0.09$
>
> **(c)** Ratio: linear/quadratic $= 0.15/0.09 \approx 1.67$. The linear term is $\sim 67\%$ larger than the quadratic term.
>
> **Implication:** OBD saliency $s_q = \frac{1}{2} H_{qq} w_q^2 = 0.09$ underestimates the actual damage (or in this case, the actual *benefit*) from pruning. For under-trained models, the gradient dominates and second-order methods lose their theoretical justification.

---

## 3. 🔬 Optimal Brain Damage

*LeCun, Denker, and Solla (1990). "Optimal Brain Damage." NeurIPS 1989.*

### 3.1 The Diagonal Hessian Approximation

OBD makes two simplifying assumptions to make $\delta L$ tractable:

1. **Diagonal Hessian:** $H_{ij} = 0$ for $i \neq j$. Weights are assumed to be independent — the curvature of the loss in the direction of weight $i$ does not depend on weight $j$.
2. **No weight compensation:** $\delta w_i = 0$ for all $i \neq q$. When we delete weight $q$, all other weights are frozen.

Under these assumptions, the pruning perturbation is simply $\delta w = -w_q^* e_q$, and:

$$\delta L_q^{OBD} = \frac{1}{2} (-w_q^*)^\top H (-w_q^*) = \frac{1}{2} w_q^{*2} H_{qq}$$

### 3.2 OBD Saliency Criterion

**Definition (OBD Saliency).** The *saliency* of weight $q$ under OBD is:

$$\boxed{s_q^{OBD} = \frac{1}{2} H_{qq}\, w_q^{*2}}$$

Weights with *low* saliency are safe to remove — they are either small ($w_q^{*2} \to 0$) or in flat regions of the loss ($H_{qq} \to 0$). Note that pure magnitude pruning ($s_q \propto |w_q|$) misses the second factor: a large weight in a flat direction is actually *less* important than a small weight in a sharp direction.

**The OBD algorithm:**
1. Train network to convergence.
2. Compute diagonal Hessian $\{H_{qq}\}$ via backprop.
3. Compute saliencies $\{s_q^{OBD}\}$.
4. Prune weights with lowest saliency (globally).
5. Retrain (fine-tune) the pruned network.
6. Repeat until target sparsity.

> [!EXAMPLE] Toy illustration: OBD vs. magnitude pruning
> Consider two weights: $w_1 = 1.0$ in a flat region ($H_{11} = 0.01$) and $w_2 = 0.1$ in a sharp region ($H_{22} = 100$).
>
> - Magnitude: prune $w_2$ (smaller magnitude). This removes the weight in the *sharp* curvature direction — a catastrophic choice.
> - OBD: $s_1 = 0.005$, $s_2 = 0.5$. Prune $w_1$. Correct: the large-but-flat weight is the safe one to delete.

### 3.3 Computing the Diagonal Hessian

Computing the full Hessian is $O(P^2)$ in memory — completely intractable for modern networks. For the diagonal we have three options of increasing approximation:

**Method 1: Double backprop (exact).** For each weight $w_i$:

$$H_{ii} = \frac{\partial^2 L}{\partial w_i^2} = \frac{\partial}{\partial w_i}\left(\frac{\partial L}{\partial w_i}\right)$$

This requires $O(P)$ forward-backward passes — still expensive.

**Method 2: Hutchinson estimator (stochastic diagonal).** Sample random Rademacher vectors $v \sim \{\pm 1\}^P$ and use:

$$H_{ii} \approx \frac{1}{K} \sum_{k=1}^K v_i^{(k)}\, (Hv^{(k)})_i$$

Hessian-vector products $Hv$ can be computed in $O(P)$ time via the Pearlmutter trick (second-order reverse-mode AD). $K \approx 30$–100 vectors gives a good estimate.

**Method 3: Empirical Fisher diagonal (practical standard).** The *empirical Fisher information* is:

$$F_{ii} = \frac{1}{N} \sum_{j=1}^N \left(\frac{\partial L_j}{\partial w_i}\right)^2 = \mathbb{E}\left[g_i^2\right]$$

At the optimum of an exponential family model, $H \approx F$ (the Fisher = Hessian identity). In practice, simply square the per-sample gradients and accumulate. This is $O(P)$ memory and $O(1)$ passes per batch — the standard choice for OBD at scale.

> [!NOTE] Fisher vs. Hessian
> The empirical Fisher is a *positive semi-definite* approximation to $H$. It is exact at a maximum-likelihood optimum for models in the exponential family (logistic regression, softmax classifiers). For MSE loss on regression, $H = X^\top X / n$ and the Fisher gives the same answer. For general architectures and non-convergence, they can differ. See [[concepts/optimization-theory/second-order-methods|Second-Order Methods]] for a rigorous treatment.

> [!QUESTION] Exercise 2: OBD for linear regression
> *This exercise derives the exact OBD saliency for a model where the Hessian has a closed form.*
>
> > **Prerequisites:** [[#3.2 OBD Saliency Criterion|3.2 OBD Saliency Criterion]]
>
> Consider linear regression: $L(w) = \frac{1}{2n}\|Xw - y\|^2$ with $X \in \mathbb{R}^{n \times d}$, $w \in \mathbb{R}^d$.
>
> (a) Compute the exact Hessian $H = \nabla^2_w L$.
> (b) Write down the diagonal entries $H_{ii}$.
> (c) State the OBD saliency. Under what condition on $X$ is OBD *exact* (i.e., the diagonal assumption introduces no error)?
> (d) Show that the Fisher diagonal $F_{ii} = \mathbb{E}[g_i^2]$ equals $H_{ii}$ for this model.

> [!TIP]- Solution to Exercise 2
> **Key insight:** For quadratic loss, the Hessian is constant (independent of $w$) and equals the data covariance, so OBD is exact when features are orthogonal.
>
> **(a)** $\nabla_w L = \frac{1}{n} X^\top (Xw - y)$. Differentiating again: $H = \frac{1}{n} X^\top X$.
>
> **(b)** $H_{ii} = \frac{1}{n}\|X_{:,i}\|^2$ — the normalized squared $\ell_2$ norm of the $i$-th feature column.
>
> **(c)** OBD saliency: $s_i^{OBD} = \frac{w_i^2}{2n}\|X_{:,i}\|^2$. The diagonal assumption introduces no error when $X^\top X$ is diagonal — i.e., when the feature columns are *orthogonal*. In that case, OBD = OBS (exact pruning with zero cost from off-diagonal terms).
>
> **(d)** At the optimum $Xw^* = \hat{y}$, the residual $r = y - Xw^*$ and the gradient on example $j$ is $g^{(j)} = \frac{1}{n} X_{j,:}^\top (X_{j,:} w^* - y_j) = \frac{1}{n} X_{j,:}^\top r_j$. Squaring and summing: $F_{ii} = \frac{1}{n^2}\sum_j r_j^2 X_{ji}^2$. At the optimum of a linear model under Gaussian noise, $\mathbb{E}[r_j^2] = \sigma^2$ and $F_{ii} = \sigma^2 H_{ii}$ — they are proportional, with equality when the noise is unit variance.

### 3.4 💻 PyTorch: OBD Saliency Scorer

`★ Insight ─────────────────────────────────────`
The Fisher diagonal accumulation below is equivalent to computing $\mathbb{E}[g_i^2]$ — the expected squared per-parameter gradient. This is exactly what Adam's second moment estimate tracks! Adam's $v_t$ buffer *is* a running Fisher diagonal. If you have an Adam-trained model, you can read off the OBD saliency from the optimizer state: `saliency = 0.5 * v_t * param^2`.
`─────────────────────────────────────────────────`

```python
import torch
import torch.nn as nn
from collections import defaultdict


class OBDPruner:
    """
    Optimal Brain Damage pruner using the empirical Fisher diagonal
    as a Hessian approximation.

    Saliency: s_i = 0.5 * F_ii * w_i^2,  F_ii = E[g_i^2].
    """

    def __init__(
        self,
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        n_batches: int = 64,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.loader = loader
        self.criterion = criterion
        self.n_batches = n_batches
        self.device = device

    # ------------------------------------------------------------------
    # Fisher diagonal: F_ii = E[g_i^2]
    # ------------------------------------------------------------------

    def _accumulate_fisher(self) -> dict[str, torch.Tensor]:
        fisher: dict[str, torch.Tensor] = {}
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                fisher[name] = torch.zeros_like(p.data)

        self.model.eval()
        n_samples = 0

        for i, (inputs, targets) in enumerate(self.loader):
            if i >= self.n_batches:
                break
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.model.zero_grad()
            loss = self.criterion(self.model(inputs), targets)
            loss.backward()

            batch_size = inputs.size(0)
            for name, p in self.model.named_parameters():
                if p.grad is not None:
                    # Accumulate sum of squared gradients (unnormalized)
                    fisher[name].add_(p.grad.data.pow(2).mul_(batch_size))
            n_samples += batch_size

        # Normalize to get the expectation
        return {name: f.div_(n_samples) for name, f in fisher.items()}

    # ------------------------------------------------------------------
    # Saliency and pruning
    # ------------------------------------------------------------------

    def saliency(self) -> dict[str, torch.Tensor]:
        """Per-weight OBD saliency s_i = 0.5 * F_ii * w_i^2."""
        fisher = self._accumulate_fisher()
        return {
            name: fisher[name].mul(p.data.pow(2)).mul_(0.5)
            for name, p in self.model.named_parameters()
            if name in fisher
        }

    def prune(self, sparsity: float) -> None:
        """
        Globally prune fraction `sparsity` of weights by lowest OBD saliency.
        Uses a hard mask (sets weights to zero in-place).
        """
        sal = self.saliency()

        # Global threshold over all saliency scores
        all_scores = torch.cat([s.flatten() for s in sal.values()])
        threshold = torch.quantile(all_scores, sparsity)

        for name, p in self.model.named_parameters():
            if name in sal:
                mask = sal[name].gt(threshold)
                p.data.mul_(mask.float())

    def current_sparsity(self) -> float:
        total = numel = 0
        for p in self.model.parameters():
            total += p.numel()
            numel += p.eq(0).sum().item()
        return numel / total
```

> [!TIP] Adam-based OBD saliency (zero extra compute)
> If the model was trained with Adam, the optimizer already stores $v_t \approx \mathbb{E}[g_i^2]$. You can skip the Fisher accumulation pass entirely:
> ```python
> def adam_obd_saliency(model, optimizer):
>     sal = {}
>     for group in optimizer.param_groups:
>         for p in group['params']:
>             state = optimizer.state[p]
>             if 'exp_avg_sq' in state:
>                 sal[p] = 0.5 * state['exp_avg_sq'] * p.data.pow(2)
>     return sal
> ```

---

## 4. 🧠 Optimal Brain Surgeon

*Hassibi and Stork (1993). "Optimal Brain Surgeon and General Network Pruning." NeurIPS 1992.*

### 4.1 The Full Inverse-Hessian Framework

OBS drops both of OBD's approximations simultaneously:

1. **Full Hessian:** No diagonal assumption. Off-diagonal curvature (weight correlations) is accounted for.
2. **Optimal weight compensation:** When weight $q$ is deleted, all remaining weights $\{w_i\}_{i \neq q}$ adjust *optimally* to minimize the resulting loss increase.

This turns pruning into a *constrained optimization problem*: find the $\delta w$ that minimizes the quadratic loss increase subject to the constraint that weight $q$ is zeroed.

### 4.2 KKT Derivation of the Weight Update

We solve:

$$\min_{\delta w}\; \frac{1}{2}\, \delta w^\top H\, \delta w \qquad \text{subject to}\quad e_q^\top \delta w = -w_q^*$$

**Lagrangian:**

$$\mathcal{L}(\delta w, \lambda) = \frac{1}{2}\,\delta w^\top H\,\delta w + \lambda\bigl(e_q^\top \delta w + w_q^*\bigr)$$

**KKT stationarity** ($\nabla_{\delta w}\,\mathcal{L} = 0$):

$$H\, \delta w + \lambda\, e_q = 0 \implies \delta w = -\lambda\, H^{-1} e_q$$

**Substituting into the constraint:**

$$e_q^\top (-\lambda\, H^{-1} e_q) = -w_q^* \implies -\lambda\, [H^{-1}]_{qq} = -w_q^* \implies \lambda = \frac{w_q^*}{[H^{-1}]_{qq}}$$

**Optimal weight perturbation:**

$$\boxed{\delta w^* = -\frac{w_q^*}{[H^{-1}]_{qq}}\, H^{-1} e_q}$$

The $j$-th component of this update is $\delta w_j^* = -\frac{w_q^*}{[H^{-1}]_{qq}}\,[H^{-1}]_{jq}$. Weight $j$ is adjusted in proportion to its Hessian-inverse correlation with the deleted weight $q$.

### 4.3 OBS Saliency and the Weight Correction

The minimum achievable loss increase when pruning weight $q$ — the **OBS saliency** — is:

$$\delta L_q^{OBS} = \frac{1}{2}\,(\delta w^*)^\top H\,(\delta w^*) = \frac{1}{2} \cdot \frac{w_q^{*2}}{[H^{-1}]_{qq}^2}\,(H^{-1} e_q)^\top H\,(H^{-1} e_q)$$

Simplifying the quadratic form:

$$(H^{-1} e_q)^\top H\,(H^{-1} e_q) = e_q^\top H^{-\top} H\, H^{-1} e_q = e_q^\top H^{-1} e_q = [H^{-1}]_{qq}$$

Therefore:

$$\boxed{\delta L_q^{OBS} = \frac{w_q^{*2}}{2\,[H^{-1}]_{qq}}}$$

**Summary of the OBS algorithm:**
1. Train network. Compute the full Hessian $H$ (or its inverse $H^{-1}$) at $w^*$.
2. For each candidate weight $q$, compute OBS saliency $s_q^{OBS} = w_q^{*2} / (2[H^{-1}]_{qq})$.
3. Select $q^* = \arg\min_q s_q^{OBS}$.
4. Delete $w_{q^*}$ and apply the weight correction $\delta w^*$ to all remaining weights.
5. Update $H^{-1}$ using the rank-1 Woodbury identity (see callout below).
6. Repeat until target sparsity.

> [!INFO] Efficient Hessian inverse update (Woodbury)
> After pruning weight $q$, the effective Hessian changes (the $q$-th row/column is removed). Recomputing $H^{-1}$ from scratch is $O(P^3)$. The *Woodbury matrix identity* gives a rank-1 update in $O(P^2)$:
>
> $$[H^{-1}_{\text{new}}]_{ij} = [H^{-1}]_{ij} - \frac{[H^{-1}]_{iq}\,[H^{-1}]_{qj}}{[H^{-1}]_{qq}}$$
>
> This is a Schur complement formula — removing a row/column from a matrix and updating its inverse. SparseGPT exploits this structure by operating on one weight per step and updating the inverse iteratively, reducing the full $O(P^3)$ OBS cost to something tractable at layer scale.

### 4.4 When OBD and OBS Diverge

**Theorem.** For any positive definite $H$ and any index $q$:

$$s_q^{OBS} = \frac{w_q^{*2}}{2[H^{-1}]_{qq}} \leq \frac{w_q^{*2}}{2} H_{qq} = s_q^{OBD}$$

*Proof.* By the positive definiteness of $H$, the Schur complement argument gives $[H^{-1}]_{qq} \geq 1/H_{qq}$, with equality iff $H$ is diagonal. Therefore $1/[H^{-1}]_{qq} \leq H_{qq}$, which yields the inequality. $\square$

*Interpretation:* OBD always *overestimates* the damage of pruning any weight. It cannot account for the compensatory slack in other weights (the off-diagonal terms of $H^{-1}$). The gap between OBD and OBS is largest when weights are highly correlated — when the Hessian has large off-diagonal entries. In convolutional layers with strong inter-filter correlations, this gap can be substantial.

> [!DANGER] Rank ordering can flip
> OBD and OBS can disagree on *which* weight to prune next. Consider two weights $w_1, w_2$ with:
> - $H_{11} = 1$, $H_{22} = 1$, $H_{12} = H_{21} = 0.9$ (strongly correlated)
> - $w_1 = 0.2$, $w_2 = 0.3$
>
> OBD says prune $w_1$ (lower $H_{ii} w_i^2$). But $H^{-1}$ has large diagonal entries because of the near-singularity of $H$ (the two weights almost lie in the same direction in loss space). OBS may identify a completely different weight to prune. Getting the order wrong leads to suboptimal pruned networks.

> [!QUESTION] Exercise 3: OBS ≤ OBD
> *This exercise proves the key inequality relating OBD and OBS saliencies.*
>
> > **Prerequisites:** [[#4.3 OBS Saliency and the Weight Correction|4.3 OBS Saliency and the Weight Correction]]
>
> Let $H \in \mathbb{R}^{P \times P}$ be positive definite and let $q \in \{1, \ldots, P\}$.
>
> (a) Using the block-matrix inversion formula (Schur complement), show that $[H^{-1}]_{qq} = 1/(H_{qq} - H_{q,-q} H_{-q,-q}^{-1} H_{-q,q})$, where $H_{-q,-q}$ denotes the submatrix with row and column $q$ deleted.
>
> (b) Hence show $[H^{-1}]_{qq} \geq 1/H_{qq}$, with equality iff $H_{qi} = 0$ for all $i \neq q$ (i.e., $H$ is block-diagonal with $q$ isolated).
>
> (c) Conclude $s_q^{OBS} \leq s_q^{OBD}$.

> [!TIP]- Solution to Exercise 3
> **Key insight:** The Schur complement formula for a single element is just the formula for the diagonal of an inverse, and the Schur complement is always $\leq H_{qq}$ for PD matrices.
>
> **(a)** Partition $H$ as $\begin{pmatrix} H_{qq} & h^\top \\ h & \hat{H} \end{pmatrix}$ where $h = H_{-q,q}$ and $\hat{H} = H_{-q,-q}$. The $(1,1)$ entry of $H^{-1}$ via block inversion is $[H^{-1}]_{qq} = (H_{qq} - h^\top \hat{H}^{-1} h)^{-1}$.
>
> **(b)** Since $\hat{H} \succ 0$, the Schur complement $S = H_{qq} - h^\top \hat{H}^{-1} h > 0$. Also $h^\top \hat{H}^{-1} h \geq 0$ (since $\hat{H}^{-1} \succ 0$), so $S \leq H_{qq}$. Therefore $[H^{-1}]_{qq} = S^{-1} \geq H_{qq}^{-1}$. Equality holds iff $h = 0$, i.e., all off-diagonal entries involving $q$ are zero.
>
> **(c)** $s_q^{OBS} = \frac{w_q^2}{2[H^{-1}]_{qq}} \leq \frac{w_q^2 H_{qq}}{2} = s_q^{OBD}$. $\square$

### 4.5 💻 PyTorch: Layer-wise OBS

For large networks, computing the full $P \times P$ Hessian inverse is intractable. The practical approach, used in [[concepts/sparsity-pruning/llm-pruning|SparseGPT]], applies OBS *layer by layer*: for a linear layer $y = Wx$, the Hessian of the reconstruction loss w.r.t. $W$ has a Kronecker structure enabling efficient row-wise computation.

**The layer Hessian.** For a linear layer with input activations $X \in \mathbb{R}^{n \times d_{in}}$ (rows are examples) and squared reconstruction loss:

$$L = \frac{1}{n}\|WX^\top - Y^\top\|_F^2$$

The Hessian w.r.t. $\text{vec}(W)$ is:

$$H = \frac{2}{n}(XX^\top) \otimes I_{d_{out}}$$

*Each row* $W_{i,:}$ has the same *row Hessian* $H_\text{row} = \frac{2}{n} X^\top X \in \mathbb{R}^{d_{in} \times d_{in}}$. This decouples the OBS problem: we can prune each row independently.

`★ Insight ─────────────────────────────────────`
The Kronecker structure $H = H_{row} \otimes I$ means the rows of $W$ are *independent* in the loss landscape — the loss doesn't couple $W_{1,:}$ to $W_{2,:}$. This is why SparseGPT processes each output neuron separately. In contrast, the columns of $W$ *are* coupled via $H_{row} = X^\top X / n$ — that's what the OBS correction $\delta w^* \propto H_{row}^{-1} e_j$ captures.
`─────────────────────────────────────────────────`

```python
import torch
import torch.nn as nn
from typing import Optional


class LayerOBSPruner:
    """
    Layer-wise OBS pruner for nn.Linear (following the SparseGPT approach).

    For a linear layer y = Wx, each row of W shares the row Hessian:
        H_row = (2/n) * X^T X  (X: n x d_in input activations)

    OBS saliency for weight W[i, j]:  s_{ij} = W[i,j]^2 / (2 * H_inv[j, j])
    OBS weight update for row i after pruning column j:
        delta_W[i, :] -= (W[i, j] / H_inv[j, j]) * H_inv[j, :]
    """

    def __init__(self, layer: nn.Linear, damping: float = 1e-2):
        self.layer = layer
        self.damping = damping
        self.H_row: Optional[torch.Tensor] = None
        self.H_inv: Optional[torch.Tensor] = None
        self._n_samples = 0
        self._hook = layer.register_forward_hook(self._collect_activations)

    # ------------------------------------------------------------------
    # Activation collection
    # ------------------------------------------------------------------

    def _collect_activations(
        self,
        module: nn.Module,
        inp: tuple,
        out: torch.Tensor,
    ) -> None:
        x = inp[0].detach()
        if x.dim() == 3:
            x = x.reshape(-1, x.size(-1))  # (batch * seq, d_in)
        n = x.size(0)
        if self.H_row is None:
            d = x.size(1)
            self.H_row = torch.zeros(d, d, device=x.device, dtype=x.dtype)
        self.H_row.addmm_(x.T, x)  # accumulate X^T X
        self._n_samples += n

    def prepare(self) -> None:
        """Finalize H_row and compute its inverse (with Tikhonov damping)."""
        self._hook.remove()
        # Normalize to get (2/n) X^T X, then add damping
        H = self.H_row.div_(self._n_samples / 2)
        d = H.size(0)
        H.add_(torch.eye(d, device=H.device, dtype=H.dtype).mul_(self.damping))
        # Cholesky-based inversion for numerical stability
        try:
            L = torch.linalg.cholesky(H)
            self.H_inv = torch.cholesky_inverse(L)
        except torch.linalg.LinAlgError:
            # Fallback: increase damping
            H.add_(torch.eye(d, device=H.device, dtype=H.dtype).mul_(1.0))
            self.H_inv = torch.linalg.inv(H)

    # ------------------------------------------------------------------
    # Row-wise OBS pruning
    # ------------------------------------------------------------------

    def prune_row(self, row_idx: int, sparsity: float) -> None:
        """
        Prune `sparsity` fraction of weights in row `row_idx` using OBS.
        Iteratively deletes the minimum-saliency weight and applies the
        weight correction to the remaining weights in the row.
        """
        assert self.H_inv is not None, "Call prepare() before prune_row()"
        W_row = self.layer.weight.data[row_idx].clone()  # (d_in,)
        H_inv = self.H_inv
        d = W_row.numel()
        n_prune = round(sparsity * d)
        active = torch.ones(d, dtype=torch.bool, device=W_row.device)

        for _ in range(n_prune):
            # Saliency for active weights: w_j^2 / (2 * H_inv[j,j])
            diag = H_inv.diag()
            saliency = W_row.pow(2).div(diag.clamp(min=1e-8).mul(2))
            saliency[~active] = float("inf")
            j = int(saliency.argmin())

            # OBS weight correction for remaining active weights
            # delta_W_row = - (W_row[j] / H_inv[j,j]) * H_inv[:, j]
            correction = H_inv[:, j].mul(-W_row[j] / H_inv[j, j].clamp(min=1e-8))
            W_row.add_(correction)
            W_row[j] = 0.0
            active[j] = False

        self.layer.weight.data[row_idx] = W_row

    def prune_all_rows(self, sparsity: float) -> None:
        """Apply OBS pruning to every row of the weight matrix."""
        n_rows = self.layer.weight.size(0)
        for i in range(n_rows):
            self.prune_row(i, sparsity)
```

> [!WARNING] Scalability note
> The iterative row-wise OBS loop above is $O(d_{in}^2 \cdot k)$ per row where $k = \lfloor\text{sparsity} \cdot d_{in}\rfloor$. For large hidden dimensions ($d_{in} \sim 4096$), this is expensive. SparseGPT amortizes this by pruning multiple weights per iteration (column-wise grouping) and updating $H^{-1}$ using the Cholesky factor directly — see [[concepts/sparsity-pruning/llm-pruning|LLM Pruning]] for the full treatment.

---

## 5. 📏 Magnitude-Based Pruning

*Han, Pool, Tran, and Dally (2015). "Learning both Weights and Connections for Efficient Neural Networks." NeurIPS 2015.*

### 5.1 The Zeroth-Order Approximation

Magnitude pruning ignores the Hessian entirely and uses the raw weight magnitude as the saliency:

$$s_q^{mag} = |w_q|$$

In the Taylor expansion framework, this corresponds to a zeroth-order approximation. There is no theoretical justification from the loss surface perspective: a large weight in a flat direction is more important under magnitude than a small weight in a sharp direction, which is the *wrong* answer.

### 5.2 Why It Works Anyway

Despite the theoretical gap, magnitude pruning is empirically competitive with Hessian-based methods at scale. Several explanations:

**1. Implicit regularization aligns magnitude with importance.** $\ell_2$ regularization (weight decay) penalizes large weights, so the optimizer implicitly minimizes weight magnitudes while maintaining task performance. After training with weight decay, large-magnitude weights are *disproportionately retained* by the gradient signal — small weights are ones the optimizer has pushed toward zero because they weren't needed. Magnitude becomes a *proxy* for gradient signal.

**2. The flat-direction effect is weaker at scale.** For large networks, the Hessian eigenspectrum is highly concentrated near zero (bulk of eigenvalues $\approx 0$, with a few large "spiked" eigenvalues). Most weights live in nearly-flat directions, so the curvature factor $H_{qq}$ is nearly constant across most weights. In this regime, magnitude and saliency are proportional.

**3. Global thresholding exploits over-parameterization.** When the model has far more parameters than needed, *most* weights are unimportant regardless of how we measure importance. Any crude criterion finds most of the prunable weights. The differences between methods matter most in the high-sparsity regime (90%+ pruning).

> [!INFO] Gale et al. (2019): State of Sparsity
> A large-scale empirical study by [Gale, Elsen, and Hooker (2019)](https://arxiv.org/abs/1902.09574) compared magnitude pruning, $\ell_0$ regularization, and variational dropout across ResNet-50 and Transformer on standard benchmarks. *Surprisingly,* the study found that magnitude pruning was competitive with or superior to the more sophisticated methods — reinforcing the "empirical regularization aligns magnitude with importance" hypothesis.

> [!QUESTION] Exercise 4: Magnitude vs. OBD on a toy example
> *This exercise shows concretely when magnitude pruning gives the wrong answer.*
>
> > **Prerequisites:** [[#3.2 OBD Saliency Criterion|3.2 OBD Saliency Criterion]], [[#5.1 The Zeroth-Order Approximation|5.1 The Zeroth-Order Approximation]]
>
> A two-weight model has weights $w_1 = 1.0$ and $w_2 = 0.1$. The Hessian (exact) is $H = \begin{pmatrix} 0.01 & 0 \\ 0 & 100 \end{pmatrix}$.
>
> (a) Compute magnitude, OBD, and OBS saliencies for each weight. (Note: since $H$ is diagonal, OBD = OBS.)
>
> (b) Magnitude pruning removes $w_2$. What is the actual loss increase from each choice?
>
> (c) How much larger (in relative terms) is the loss increase from magnitude's choice vs. the correct choice?

> [!TIP]- Solution to Exercise 4
> **Key insight:** Magnitude pruning removes the *geometrically important* weight (small magnitude, sharp curvature) while leaving the *geometrically irrelevant* weight (large magnitude, flat curvature).
>
> **(a)**
> - Magnitude: $|w_1| = 1.0$, $|w_2| = 0.1$ → prune $w_2$
> - OBD: $s_1 = 0.5 \cdot 0.01 \cdot 1.0 = 0.005$, $s_2 = 0.5 \cdot 100 \cdot 0.01 = 0.5$ → prune $w_1$
> - OBS: since $H$ is diagonal, $[H^{-1}]_{11} = 100$, $[H^{-1}]_{22} = 0.01$: $s_1^{OBS} = 1.0/(2 \cdot 100) = 0.005$, $s_2^{OBS} = 0.01/(2 \cdot 0.01) = 0.5$. Same as OBD.
>
> **(b)** Loss increases: $\delta L_1 = 0.005$, $\delta L_2 = 0.5$.
>
> **(c)** Magnitude's choice ($w_2$) costs $0.5$ in loss. The correct choice ($w_1$) costs $0.005$. The ratio is $0.5/0.005 = 100\times$. Magnitude pruning causes 100× more damage in this example.

### 5.3 💻 PyTorch: Magnitude Pruner

PyTorch provides `torch.nn.utils.prune` with built-in magnitude methods. The key distinction is *local* (per-layer threshold) vs. *global* (single threshold over all parameters) pruning.

```python
import torch
import torch.nn as nn
from torch.nn.utils import prune


def magnitude_prune_global(model: nn.Module, sparsity: float) -> None:
    """
    Global unstructured ℓ₁-magnitude pruning.

    A single threshold is computed over all weight tensors jointly so that
    exactly `sparsity` fraction of all weights (by count) are zeroed.
    This is preferable to per-layer pruning because different layers have
    different magnitude distributions — forcing uniform sparsity per layer
    under-prunes redundant layers and over-prunes critical ones.
    """
    params_to_prune = [
        (module, "weight")
        for module in model.modules()
        if isinstance(module, (nn.Linear, nn.Conv2d))
    ]
    prune.global_unstructured(
        params_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity,
    )


def remove_pruning_reparameterization(model: nn.Module) -> None:
    """
    torch.nn.utils.prune works by adding a 'weight_mask' buffer and a
    'weight_orig' parameter — the actual 'weight' is computed on-the-fly
    as weight_orig * weight_mask. After retraining, call this to make
    the mask permanent and restore standard weight tensors.
    """
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            try:
                prune.remove(module, "weight")
            except ValueError:
                pass  # not pruned


def current_sparsity(model: nn.Module) -> float:
    """Fraction of zero weights across all Linear and Conv2d layers."""
    total = zeros = 0
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            w = module.weight.data
            total += w.numel()
            zeros += w.eq(0).sum().item()
    return zeros / total if total > 0 else 0.0
```

> [!NOTE] The mask reparameterization
> `torch.nn.utils.prune` implements pruning as a *reparameterization*: it registers `weight_orig` (the unpruned weights) and `weight_mask` (a binary tensor), and `weight` becomes a forward hook that computes `weight_orig * weight_mask`. This means gradients flow through `weight_orig` but masked weights are zeroed each forward pass. You must call `prune.remove()` after retraining to collapse this back to a standard weight tensor for deployment.

---

## 6. 🔄 Iterative Magnitude Pruning

*Han et al. (2015). "Learning both Weights and Connections." NeurIPS 2015.*

### 6.1 The Train → Prune → Retrain Loop

*One-shot pruning* — train once, prune once — degrades significantly at high sparsity. The key insight of Han et al. is that the network needs a *recovery phase* after each pruning step to redistribute the representational load from deleted weights to surviving ones.

**Iterative Magnitude Pruning (IMP):**

```
for round r = 1, ..., R:
    1. Train for E epochs (full training in round 1; fine-tuning in subsequent rounds)
    2. Prune: zero out the bottom p% of weights by |w| (globally)
    3. (Optional) Reset pruned weights' masks but keep surviving weights
```

After $R$ rounds at rate $p$, the total sparsity is $1 - (1-p)^R$. For example, $R = 10$ rounds at $p = 20\%$ reaches $1 - 0.8^{10} \approx 89\%$ sparsity.

> [!NOTE] IMP vs. Lottery Ticket weight rewinding
> The IMP described here (prune + retrain from current weights) is *different* from the Lottery Ticket Hypothesis variant, where pruned weights are reset to their *initialization values* after each round. The LTH variant is described in [[concepts/sparsity-pruning/sparse-training|Sparse Training]].

### 6.2 Global vs. Layer-wise Thresholds

**Layer-wise thresholding:** Each layer has its own sparsity target (e.g., 50% sparsity everywhere). This treats all layers as equally compressible — which is wrong. Fully-connected layers are generally more compressible than the first/last convolutional layers; the softmax output layer is usually not pruned at all.

**Global thresholding:** A single magnitude threshold is applied across all weights jointly. Redundant layers (large layers far from the input/output) end up with higher sparsity; critical layers retain more weights. This is *empirically superior* and is what Han et al. use.

> [!WARNING] Don't prune the final classification layer
> The softmax layer's weights directly encode class-conditional output. Its saliency scores are uniformly high, and global thresholding naturally spares it. If you force uniform sparsity across all layers, you risk catastrophic accuracy collapse on the final layer.

> [!QUESTION] Exercise 5: Geometric sparsity schedule
> *This exercise derives the per-round pruning rate needed to hit a target sparsity in R rounds.*
>
> > **Prerequisites:** [[#6.1 The Train → Prune → Retrain Loop|6.1 The Train → Prune → Retrain Loop]]
>
> You want to reach 90% total sparsity ($s_{final} = 0.9$) in $R = 5$ rounds using a constant per-round pruning fraction $p$ (fraction of *remaining* weights removed each round).
>
> (a) Derive the formula for $p$ in terms of $s_{final}$ and $R$.
>
> (b) Compute the numerical value of $p$ for this case.
>
> (c) After round 3, what is the cumulative sparsity?

> [!TIP]- Solution to Exercise 5
> **Key insight:** Iterative pruning at a constant *fraction of remaining weights* is a geometric series in the *surviving* weight count.
>
> **(a)** After $R$ rounds, the fraction of weights surviving is $(1-p)^R = 1 - s_{final}$. Solving: $p = 1 - (1 - s_{final})^{1/R}$.
>
> **(b)** $p = 1 - 0.1^{1/5} = 1 - 0.631 \approx 36.9\%$ pruned per round.
>
> **(c)** After round 3: fraction surviving = $(1-p)^3 = 0.631^3 \approx 0.251$, so cumulative sparsity $\approx 74.9\%$.

### 6.3 💻 PyTorch: Full IMP Pipeline

`★ Insight ─────────────────────────────────────`
The IMP loop below uses PyTorch's mask reparameterization instead of directly zeroing weights. This is crucial: if you zero weights in-place and the optimizer has momentum state (SGD) or second-moment estimates (Adam) for those weights, the optimizer will try to "un-prune" them via accumulated gradient updates. The mask ensures zeroed weights stay zero regardless of gradient updates.
`─────────────────────────────────────────────────`

```python
import copy
import torch
import torch.nn as nn
from torch.nn.utils import prune
from torch.utils.data import DataLoader
from typing import Callable


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(loader.dataset)


def iterative_magnitude_prune(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer],
    criterion: nn.Module,
    n_rounds: int,
    target_sparsity: float,
    epochs_per_round: int,
    device: str = "cuda",
) -> nn.Module:
    """
    Iterative magnitude pruning: (train → prune) × n_rounds.

    Uses a geometric sparsity schedule: each round removes a constant
    fraction p of the *remaining* unpruned weights, reaching
    `target_sparsity` exactly after `n_rounds` rounds.

    Returns the pruned model with permanent weight masks applied.
    """
    model = model.to(device)

    # Per-round pruning fraction of REMAINING weights
    per_round_fraction = 1.0 - (1.0 - target_sparsity) ** (1.0 / n_rounds)

    prunable = [
        (module, "weight")
        for module in model.modules()
        if isinstance(module, (nn.Linear, nn.Conv2d))
    ]

    for round_idx in range(n_rounds):
        optimizer = optimizer_factory(model)

        # Train (full training in round 0, fine-tuning thereafter)
        for epoch in range(epochs_per_round):
            loss = _train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Global unstructured magnitude pruning
        # amount= is fraction of CURRENTLY UNPRUNED weights to remove
        prune.global_unstructured(
            prunable,
            pruning_method=prune.L1Unstructured,
            amount=per_round_fraction,
        )

        sparsity = current_sparsity(model)
        print(f"Round {round_idx + 1}/{n_rounds}  sparsity={sparsity:.1%}")

    # Collapse weight_orig * weight_mask → weight (permanent mask)
    remove_pruning_reparameterization(model)
    return model


# -----------------------------------------------------------------------
# Usage example
# -----------------------------------------------------------------------

if __name__ == "__main__":
    import torchvision.models as tvm
    import torchvision.datasets as tvd
    import torchvision.transforms as T

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)

    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    dataset = tvd.FakeData(size=1024, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

    def make_optimizer(m):
        return torch.optim.SGD(m.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

    pruned = iterative_magnitude_prune(
        model=model,
        train_loader=loader,
        optimizer_factory=make_optimizer,
        criterion=nn.CrossEntropyLoss(),
        n_rounds=5,
        target_sparsity=0.90,
        epochs_per_round=3,
        device=device,
    )
    print(f"Final sparsity: {current_sparsity(pruned):.1%}")
```

> [!EXAMPLE] Han et al. (2015) results
> Applying IMP to AlexNet on ImageNet:
> - FC layers: 90–95% sparsity (layers 6, 7, 8 pruned from 9216→4096→4096→1000)
> - Conv layers: 65–84% sparsity
> - Overall: **9× compression** of AlexNet, **13× compression** of VGG-16
> - Top-1 accuracy: unchanged (within noise of repeated runs)
>
> The key observation is that *most of the compressible mass* is in the fully-connected layers. ConvNet convolutional layers are already relatively parameter-efficient. This motivates the compression pipeline in [[concepts/sparsity-pruning/compression-pipelines|Deep Compression]], which chains IMP with quantization and Huffman coding to reach 35–49× total compression.

---

## 7. 📚 References

| Reference Name | Brief Summary | Link |
|----------------|---------------|------|
| LeCun, Denker, Solla (1990). "Optimal Brain Damage" | Introduced second-order (diagonal Hessian) saliency scores; first rigorous framework for principled weight deletion | [NeurIPS 1989](https://proceedings.neurips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html) |
| Hassibi & Stork (1993). "Optimal Brain Surgeon" | Extended OBD to full inverse Hessian with closed-form weight compensation; proved OBD overestimates damage | [NeurIPS 1992](https://proceedings.neurips.cc/paper/1992/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html) |
| Han, Pool, Tran, Dally (2015). "Learning both Weights and Connections" | Canonical IMP pipeline (train → threshold → retrain); 9×/13× compression of AlexNet/VGG-16 | [arXiv:1506.02626](https://arxiv.org/abs/1506.02626) |
| Gale, Elsen, Hooker (2019). "The State of Sparsity in Deep Neural Networks" | Large-scale comparison of pruning methods; magnitude pruning competitive with complex methods | [arXiv:1902.09574](https://arxiv.org/abs/1902.09574) |
| Blalock et al. (2020). "What is the State of Neural Network Pruning?" | 81-paper meta-survey; reproducibility crisis in pruning; introduced ShrinkBench | [arXiv:2003.03033](https://arxiv.org/abs/2003.03033) |
| Liu et al. (2019). "Rethinking the Value of Network Pruning" | For structured pruning, architecture matters more than inherited weights | [arXiv:1810.05270](https://arxiv.org/abs/1810.05270) |
