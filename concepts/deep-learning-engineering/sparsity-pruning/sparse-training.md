# 🎰 Sparse Training: Lottery Tickets, SNIP, and RigL

## Table of Contents

- [[#1. 💡 The Dense-to-Sparse Paradigm Shift|1. The Dense-to-Sparse Paradigm Shift]]
- [[#2. 🎰 The Lottery Ticket Hypothesis|2. The Lottery Ticket Hypothesis]]
  - [[#2.1 Formal Statement|2.1 Formal Statement]]
  - [[#2.2 Finding Winning Tickets: IMP + Weight Rewinding|2.2 Finding Winning Tickets: IMP + Weight Rewinding]]
  - [[#2.3 Linear Mode Connectivity and the Stability Fix|2.3 Linear Mode Connectivity and the Stability Fix]]
  - [[#2.4 💻 PyTorch: LTH Weight Rewinding|2.4 PyTorch: LTH Weight Rewinding]]
- [[#3. ✂️ SNIP: Single-Shot Pruning Before Training|3. SNIP: Single-Shot Pruning Before Training]]
  - [[#3.1 Connection Sensitivity at Initialization|3.1 Connection Sensitivity at Initialization]]
  - [[#3.2 💻 PyTorch: SNIP Saliency|3.2 PyTorch: SNIP Saliency]]
- [[#4. 🌱 Dynamic Sparse Training|4. Dynamic Sparse Training]]
  - [[#4.1 SET: Sparse Evolutionary Training|4.1 SET: Sparse Evolutionary Training]]
  - [[#4.2 RigL: Rigging the Lottery|4.2 RigL: Rigging the Lottery]]
  - [[#4.3 The RigL Update Rule|4.3 The RigL Update Rule]]
  - [[#4.4 💻 PyTorch: RigL Mask Update Step|4.4 PyTorch: RigL Mask Update Step]]
- [[#5. 📊 Empirical Comparison|5. Empirical Comparison]]
- [[#6. 📚 References|6. References]]

---

## 1. 💡 The Dense-to-Sparse Paradigm Shift

All methods in [[concepts/deep-learning-engineering/sparsity-pruning/classical-pruning|Classical Pruning]] and [[concepts/deep-learning-engineering/sparsity-pruning/structured-pruning|Structured Pruning]] follow the same paradigm: *train dense, then prune*. This has a fundamental inefficiency: you pay the full computational cost of training a dense model just to discard 80–90% of it.

The *sparse training* question is more radical: **Can we train a sparse network from scratch — or online during training — and match the dense baseline's accuracy?**

Two subquestions emerge:

1. **Sparse from initialization (SNIP, GRASP):** Find a good sparse mask *before* any training. Use it as a fixed mask throughout.
2. **Dynamic sparse training (SET, RigL):** Start sparse. Let the mask evolve *during* training — grow connections in useful directions, prune others.

The Lottery Ticket Hypothesis motivates both: if winning ticket subnetworks exist, perhaps we can find them without the expensive dense training phase.

---

## 2. 🎰 The Lottery Ticket Hypothesis

*Frankle and Carlin (2019). "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks." ICLR 2019 Best Paper.*

### 2.1 Formal Statement

**Definition (Winning Ticket).** Let $f(x; w)$ be a network initialized at $w_0 \sim \mathcal{D}_{init}$ and trained for $T$ steps to accuracy $a$ with mask $m = \mathbf{1}$ (dense). A *winning ticket* is a subnetwork $f(x; m \odot w_0)$ — same initialization $w_0$, sparse mask $m$ — that when trained for $T$ steps achieves accuracy $a' \geq a$ at parameter count $|m| \ll |w|$.

**Lottery Ticket Hypothesis (LTH).** Dense, randomly-initialized networks contain sparse subnetworks that, when trained in isolation from their original initialization, can match the full network's test accuracy in at most the same number of training iterations.

*Surprising implication:* The specific initial weight values $w_0$ matter — they are the "lottery ticket." Reinitializing the sparse subnetwork to fresh random values (different from $w_0$) destroys the winning property. The ticket is not just the architecture, it is the architecture *plus* the specific initial weights.

### 2.2 Finding Winning Tickets: IMP + Weight Rewinding

The procedure to find winning tickets is *Iterative Magnitude Pruning with weight rewinding*:

```
w_0 ~ D_init                          # record the initialization
for round r = 1, ..., R:
    w_T = SGD(w_{T-1}, T steps)       # train for T steps
    m_r = {i : |w_T[i]| > threshold}  # prune by magnitude
    w_{r+1} = m_r ⊙ w_0              # rewind: reset to ORIGINAL init
```

The key step is **weight rewinding**: after pruning, the surviving weights are reset to $w_0$ (not the trained values $w_T$). The surviving *mask* $m_r$ is kept but the *values* are rewound. This is what distinguishes LTH from standard IMP (which keeps the trained weights).

**Why rewinding?** The hypothesis claims the winning property resides in the *combination* of mask and initialization, not just the mask. Empirically, rewinding to $w_0$ consistently outperforms random reinitialization of the masked subnetwork.

> [!WARNING] LTH at scale fails without late rewinding
> Frankle et al. (2020) showed that the original LTH (rewind to step 0) only works for small networks (LeNet, small ResNets). For large networks (ResNet-50, wide ResNets), the winning tickets only emerge after a *few hundred training steps* — not at initialization. The fix is **late rewinding**: instead of rewinding to $w_0$, rewind to $w_k$ (the weights at step $k \approx 1\%$–$5\%$ of total training). The subnetwork is *linearly stable* to SGD noise at this point.

### 2.3 Linear Mode Connectivity and the Stability Fix

*Frankle, Dziugaite, Roy, Carlin (2020). "Linear Mode Connectivity and the Lottery Ticket Hypothesis." ICML 2020.*

**Definition (Linear Mode Connectivity).** Two solutions $w_A$ and $w_B$ are *linearly connected at error $\epsilon$* if the interpolated network $w(\alpha) = (1-\alpha)w_A + \alpha w_B$ achieves test error $\leq \epsilon + \max(\text{err}(w_A), \text{err}(w_B))$ for all $\alpha \in [0, 1]$.

**Key result.** A sparse subnetwork $m \odot w$ (found by IMP) constitutes a winning ticket *if and only if* its solutions (found by training from $w_k$ with two different random SGD noise sequences) are linearly connected at low error. This occurs reliably at step $k > 0$ for large-scale networks, but rarely at $k = 0$.

*Interpretation:* At step $k = 0$ (random init), the loss landscape around the sparse subnetwork is still rough — tiny perturbations to SGD noise land in different basins. After $k$ steps, gradient descent has "oriented" the subnetwork in a direction where the loss basin is smooth and wide enough that two training runs converge to linearly connected solutions.

> [!QUESTION] Exercise 1: Rewinding vs. random reinit
> *This exercise quantifies the difference between rewinding and random reinitialization.*
>
> > **Prerequisites:** [[#2.2 Finding Winning Tickets: IMP + Weight Rewinding|2.2 Finding Winning Tickets: IMP + Weight Rewinding]]
>
> Frankle & Carlin find that on MNIST with a 2-layer FC network at 90% sparsity:
> - IMP + rewind to $w_0$: 98.2% test accuracy
> - IMP + random reinit: 96.1% test accuracy
> - Dense baseline: 98.3% test accuracy
>
> (a) What is the accuracy gap between rewinding and random reinit at 90% sparsity?
>
> (b) Suppose the gap grows as sparsity increases. At 99% sparsity, rewinding achieves 97.5% and random reinit achieves 91.2%. What does this tell you about where the "lottery" information lives at extreme sparsity?
>
> (c) The gap between rewinding and the dense baseline is only 0.1% at 90% sparsity. What does this say about the compression-accuracy tradeoff?

> [!TIP]- Solution to Exercise 1
> **Key insight:** The initialization values encode the "winning" information, not just the sparse structure. At extreme sparsity, the gap grows dramatically — the specific initial weights become *critical*.
>
> **(a)** Gap at 90%: 98.2% − 96.1% = 2.1 percentage points.
>
> **(b)** At 99% sparsity, gap = 97.5% − 91.2% = 6.3 pp. The gap nearly triples as sparsity goes from 90% to 99%. This means the initial weight values carry increasingly critical information as the ticket gets more sparse: when very few weights survive, *which values they start at* becomes the dominant factor in final performance.
>
> **(c)** The dense network's 98.3% is matched (within 0.1%) by a 10% sparse subnetwork trained from the original init. This means 90% of parameters are essentially redundant given the right sparse structure and init — a striking demonstration of over-parameterization.

### 2.4 💻 PyTorch: LTH Weight Rewinding

```python
import copy
import torch
import torch.nn as nn
from torch.nn.utils import prune


class LotteryTicketFinder:
    """
    Finds winning tickets via Iterative Magnitude Pruning + weight rewinding.

    Usage:
        finder = LotteryTicketFinder(model, rewind_step=0)
        finder.record_init()                    # save w_0
        finder.train(loader, optimizer, T)      # train T steps
        finder.prune(sparsity=0.2)              # prune 20% of remaining
        finder.rewind()                          # reset surviving weights to w_0
        # Repeat train/prune/rewind n_rounds times
    """

    def __init__(self, model: nn.Module, rewind_step: int = 0):
        self.model = model
        self.rewind_step = rewind_step
        self._init_weights: dict[str, torch.Tensor] = {}
        self._rewind_weights: dict[str, torch.Tensor] = {}
        self._step = 0

    def record_init(self) -> None:
        """Save w_0 (initialization) for later rewinding."""
        self._init_weights = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }

    def _record_rewind_checkpoint(self) -> None:
        """Save weights at the current step as the rewind target."""
        self._rewind_weights = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }

    def step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """Single training step; records rewind checkpoint at rewind_step."""
        self.model.train()
        optimizer.zero_grad()
        loss = criterion(self.model(inputs), targets)
        loss.backward()
        optimizer.step()
        self._step += 1

        if self._step == self.rewind_step:
            self._record_rewind_checkpoint()

        return loss.item()

    def prune(self, amount: float) -> None:
        """
        Global magnitude pruning: zero out `amount` fraction of
        the currently-unmasked weights.
        """
        params = [
            (m, "weight")
            for m in self.model.modules()
            if isinstance(m, (nn.Linear, nn.Conv2d))
        ]
        prune.global_unstructured(params, prune.L1Unstructured, amount=amount)

    def rewind(self) -> None:
        """
        Reset surviving (non-zero) weights to their values at rewind_step.
        The pruning mask is preserved — only values are rewound.
        """
        rewind_target = (
            self._rewind_weights if self._rewind_weights else self._init_weights
        )
        for name, param in self.model.named_parameters():
            base_name = name.replace("_orig", "")
            if base_name in rewind_target:
                # Preserve the mask: only rewind values where mask is 1
                if hasattr(param, "_mask"):
                    param.data.copy_(rewind_target[base_name] * param._mask)
                else:
                    param.data.copy_(rewind_target[base_name])

    def current_sparsity(self) -> float:
        total = zeros = 0
        for p in self.model.parameters():
            total += p.numel()
            zeros += p.eq(0).sum().item()
        return zeros / total
```

---

## 3. ✂️ SNIP: Single-Shot Pruning Before Training

*Lee, Ajanthan, Torr (2019). "SNIP: Single-shot Network Pruning based on Connection Sensitivity." ICLR 2019.*

### 3.1 Connection Sensitivity at Initialization

SNIP asks: can we identify unimportant weights *before any training* using a single mini-batch? The key quantity is the *connection sensitivity* $c_j$ — the magnitude of the loss change when weight $w_j$ is dropped at initialization:

$$c_j = \left|\frac{\partial L}{\partial w_j}\bigg|_{w=w_0} \cdot w_j\right|$$

This is the first-order Taylor term for removing $w_j$ (keeping $g \neq 0$ at init). Note this differs from OBD/OBS saliency (which uses second-order terms at convergence). At initialization, the loss is far from a minimum, so the gradient dominates.

**Normalization:** SNIP normalizes to make connection sensitivities comparable across layers:

$$\tilde{c}_j = \frac{c_j}{\sum_k c_k}$$

**Pruning rule:** Prune the bottom $(1-\kappa)$ fraction of connections by $\tilde{c}_j$, retaining fraction $\kappa$. The pruned mask is then *fixed* for the entire training run.

> [!NOTE] SNIP saliency vs. OBD saliency
> - SNIP: first-order, computed at initialization ($g \neq 0$). Saliency $= |g_j \cdot w_j|$.
> - OBD: second-order, computed at convergence ($g \approx 0$). Saliency $= \frac{1}{2} H_{jj} w_j^2$.
>
> SNIP is tractable before training (one mini-batch). OBD requires a fully trained model. The two are complementary: SNIP finds cheap masks, OBD finds accurate masks.

> [!QUESTION] Exercise 2: SNIP saliency for a linear layer
> *This exercise computes SNIP saliency explicitly for a linear model.*
>
> > **Prerequisites:** [[#3.1 Connection Sensitivity at Initialization|3.1 Connection Sensitivity at Initialization]]
>
> A single linear layer $y = Wx$ has weights $W \in \mathbb{R}^{2 \times 2}$ initialized as $W_0 = \begin{pmatrix} 0.5 & -0.3 \\ 0.1 & 0.8 \end{pmatrix}$. On a single example $x = (1, 1)^\top$, $y^* = (0, 1)^\top$, the MSE loss is $L = \|Wx - y^*\|^2$.
>
> (a) Compute $y = W_0 x$ and $L$.
>
> (b) Compute $\partial L / \partial W_{ij}$ for all $i, j$.
>
> (c) Compute the unnormalized SNIP saliencies $c_{ij} = |(\partial L/\partial W_{ij}) \cdot W_{ij}|$. Which weight has the highest saliency? Which would be pruned first?

> [!TIP]- Solution to Exercise 2
> **Key insight:** SNIP saliency combines gradient magnitude with weight magnitude — neither alone determines importance.
>
> **(a)** $y = W_0 x = (0.5 - 0.3,\; 0.1 + 0.8)^\top = (0.2,\; 0.9)^\top$. $L = (0.2-0)^2 + (0.9-1)^2 = 0.04 + 0.01 = 0.05$.
>
> **(b)** $\partial L/\partial W_{ij} = 2(y_i - y^*_i) x_j$. With $r = y - y^* = (0.2, -0.1)^\top$ and $x = (1, 1)^\top$:
> $$\frac{\partial L}{\partial W} = 2 r x^\top = 2 \begin{pmatrix} 0.2 \\ -0.1 \end{pmatrix} \begin{pmatrix} 1 & 1 \end{pmatrix} = \begin{pmatrix} 0.4 & 0.4 \\ -0.2 & -0.2 \end{pmatrix}$$
>
> **(c)** $c_{ij} = |g_{ij} \cdot W_{ij}|$:
> $c_{11} = |0.4 \times 0.5| = 0.20$, $c_{12} = |0.4 \times (-0.3)| = 0.12$, $c_{21} = |(-0.2)(0.1)| = 0.02$, $c_{22} = |(-0.2)(0.8)| = 0.16$.
>
> Highest saliency: $W_{11} = 0.20$. **Prune first:** $W_{21}$ (lowest saliency $= 0.02$).

### 3.2 💻 PyTorch: SNIP Saliency

```python
import torch
import torch.nn as nn
from torch.nn.utils import prune


def snip_saliency(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
) -> dict[str, torch.Tensor]:
    """
    Compute SNIP connection sensitivity c_j = |g_j * w_j| for all weights.
    Uses a single forward-backward pass on the provided mini-batch.

    Returns dict mapping parameter name -> saliency tensor (same shape as param).
    """
    model.zero_grad()
    loss = criterion(model(inputs), targets)
    loss.backward()

    saliency = {}
    for name, param in model.named_parameters():
        if param.grad is not None and param.requires_grad:
            # c_j = |dL/dw_j * w_j|
            saliency[name] = (param.grad * param.data).abs()

    return saliency


def snip_prune(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    sparsity: float,
    device: str = "cuda",
) -> None:
    """
    Apply SNIP pruning: compute sensitivity on one mini-batch,
    then globally prune the bottom `sparsity` fraction of connections.
    The mask is applied in-place and fixed for subsequent training.
    """
    inputs = inputs.to(device)
    targets = targets.to(device)
    sal = snip_saliency(model, inputs, targets, criterion)

    # Global threshold
    all_scores = torch.cat([s.flatten() for s in sal.values()])
    threshold = torch.quantile(all_scores, sparsity)

    for name, param in model.named_parameters():
        if name in sal:
            mask = sal[name].gt(threshold).float()
            param.data.mul_(mask)
            # Register as a permanent pruning mask via torch.nn.utils.prune
            # (Optional: use a custom mask hook to keep zeros during training)
```

---

## 4. 🌱 Dynamic Sparse Training

### 4.1 SET: Sparse Evolutionary Training

*Mocanu et al. (2018). "Scalable Training of Artificial Neural Networks with Adaptive Sparse Connectivity Inspired by Network Science." Nature Communications.*

SET replaces dense FC layers with sparse *Erdős–Rényi random graphs* at initialization. The sparsity level per layer follows:

$$s_l = 1 - \frac{\epsilon (n_l + n_{l-1})}{n_l n_{l-1}}$$

where $\epsilon$ controls the overall connectivity. This sets the number of connections per layer proportional to $(n_l + n_{l-1})$ — the same scaling as a *random sparse graph* of degree $\epsilon$.

**Training loop:**

```
Initialize: sparse random mask M_0 (Erdős–Rényi)
for each epoch:
    Train: forward/backward with weights W ⊙ M
    Evolve topology:
        Prune: zero out fraction p of weights with |w| < τ
        Grow:  randomly activate the same number of new weights
```

The grow step is *uniform random* — new connections are selected uniformly among the currently-zero connections. No gradient information is used to guide growth.

**Key finding:** Despite no dense model, SET trains sparse ResNets that match or approach dense accuracy on CIFAR-10/100, at a fraction of the training FLOPs.

### 4.2 RigL: Rigging the Lottery

*Evci, Gale, Menick, Castro, Elsen (2020). "Rigging the Lottery: Making All Tickets Winners." ICML 2020.*

RigL improves on SET's random growth by using *gradient magnitudes* to select which new connections to activate:

$$\text{Grow}: \text{activate} \left\{j \in \text{inactive}: \left|\frac{\partial L}{\partial w_j}\right| \text{ is among top-}k \text{ inactive weights}\right\}$$

The gradient $\partial L / \partial w_j$ for a currently-zero weight $w_j = 0$ is well-defined — it measures the first-order improvement from activating that connection. High-gradient inactive weights are the most beneficial to grow.

**The FLOP budget constraint.** RigL maintains a fixed sparsity $s$ throughout training — every grow step activates exactly as many connections as the prune step removes. Total training FLOPs are therefore constant, regardless of the update frequency $\Delta T$.

### 4.3 The RigL Update Rule

Let $M^{(t)} \in \{0, 1\}^P$ be the mask at step $t$. The RigL update at interval $\Delta T$ is:

**Prune:** Remove connections with smallest weight magnitude:
$$\mathcal{D}^{(t)} = \text{bottom-}k \left\{|w_j| : M^{(t)}_j = 1\right\}$$

**Grow:** Activate connections with largest gradient magnitude:
$$\mathcal{G}^{(t)} = \text{top-}k \left\{\left|\frac{\partial L}{\partial w_j}\right| : M^{(t)}_j = 0\right\}$$

**Update mask:**
$$M^{(t+1)}_j = \begin{cases} 0 & j \in \mathcal{D}^{(t)} \\ 1 & j \in \mathcal{G}^{(t)} \\ M^{(t)}_j & \text{otherwise} \end{cases}$$

The fraction of weights updated per step is controlled by a *cosine drop schedule*:
$$k^{(t)} = \left\lfloor k_0 \cdot \frac{1 + \cos\!\left(\pi t / T_{end}\right)}{2} \right\rfloor$$

This starts with large topology updates (high $k_0$) and tapers to zero at $T_{end}$, freezing the final mask.

> [!NOTE] Why gradient for grow, magnitude for prune?
> Pruning: weights with small $|w|$ contribute little to the current forward pass — magnitude is the right criterion for *current* utility. Growing: weights currently at zero have no magnitude signal; their gradient measures the *potential* utility of activating them — the first-order improvement from turning them on. Using magnitude to grow is meaningless (all inactive weights have $w = 0$); using gradient to prune is expensive (requires dense backward) and less stable.

> [!QUESTION] Exercise 3: RigL's gradient computation for zero weights
> *This exercise derives why gradients of zero weights are well-defined.*
>
> > **Prerequisites:** [[#4.3 The RigL Update Rule|4.3 The RigL Update Rule]]
>
> Consider a linear layer $y = Wx$ where weight $W_{ij} = 0$ (currently pruned). The loss is $L = f(y)$.
>
> (a) Write down $\partial L / \partial W_{ij}$ via the chain rule.
>
> (b) Explain why this is nonzero even though $W_{ij} = 0$.
>
> (c) In RigL, to compute the gradient for all inactive connections, we would need to perform a forward pass *as if all connections were active* (dense backward). Why is this computationally expensive, and how does RigL avoid it?

> [!TIP]- Solution to Exercise 3
> **Key insight:** The gradient $\partial L / \partial W_{ij}$ depends on $x_j$ and $\partial L/\partial y_i$ — neither of which requires $W_{ij} \neq 0$. The gradient is a property of the *surrounding computation*, not the weight value itself.
>
> **(a)** $\partial L / \partial W_{ij} = (\partial L / \partial y_i) \cdot x_j$.
>
> **(b)** $\partial L / \partial y_i$ is the backpropagated error signal at neuron $i$, determined by all *other* weights in the network and the loss. $x_j$ is the input activation at position $j$, determined by upstream weights. Both are computed during the forward/backward pass regardless of $W_{ij}$'s value. The fact that $W_{ij} = 0$ affects $y_i$ (which is missing the contribution $W_{ij} x_j$) but not the gradient formula itself.
>
> **(c)** A dense backward pass through the sparse layer would require computing activations for *all* $W_{ij}$ (including zeros), which costs $O(n_{in} \times n_{out})$ — the same as a dense layer. RigL avoids this by observing that $\partial L / \partial W_{ij} = (\partial L / \partial y_i) \cdot x_j$ can be computed from the already-available backpropagated signal $\partial L / \partial y_i$ (sparse, $O(n_{out})$) and input $x_j$ (also available). The cost is $O(n_{in} \times n_{out})$ only for the grow computation, not the main forward/backward — and this is amortized over $\Delta T$ training steps.

### 4.4 💻 PyTorch: RigL Mask Update Step

`★ Insight ─────────────────────────────────────`
The trick for computing the gradient of inactive weights without a full dense pass is to store the sparse layer's *input activations* and *output gradients* during the regular backward pass, then compute their outer product: $\partial L / \partial W = (\partial L / \partial y)^T \otimes x$. This is $O(n_{in} + n_{out})$ to accumulate but $O(n_{in} \times n_{out})$ to materialize. RigL only materializes the top-k entries, making the actual grow step $O(n_{in} \times n_{out})$ but well-parallelized on GPU.
`─────────────────────────────────────────────────`

```python
import torch
import torch.nn as nn
import math


class RigLLayer(nn.Module):
    """
    A sparse linear layer with a RigL-updatable mask.
    Stores input activations and output gradients for the grow step.
    """

    def __init__(self, in_features: int, out_features: int, sparsity: float):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity

        # Initialize with a dense weight; mask out `sparsity` fraction
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        mask = torch.rand(out_features, in_features) > sparsity
        self.register_buffer("mask", mask.float())
        self.weight.data.mul_(self.mask)

        # Storage for gradient computation of inactive connections
        self._last_input: torch.Tensor | None = None
        self._last_grad_output: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._last_input = x.detach()

        def _save_grad(grad: torch.Tensor) -> None:
            self._last_grad_output = grad.detach()

        out = torch.nn.functional.linear(x, self.weight * self.mask)
        if out.requires_grad:
            out.register_hook(_save_grad)
        return out

    # ------------------------------------------------------------------
    # RigL update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def rigl_update(self, drop_fraction: float) -> None:
        """
        Perform one RigL topology update:
          1. Prune: drop `drop_fraction` of active connections by |w|
          2. Grow: activate the same number by |grad_W| for inactive connections
        """
        active = self.mask.bool()
        n_active = active.sum().item()
        k = max(1, round(drop_fraction * n_active))

        # ── Prune ──────────────────────────────────────────────────────
        active_magnitudes = self.weight.data.abs() * self.mask
        # Set inactive weights' magnitudes to infinity so they aren't pruned
        active_magnitudes[~active] = float("inf")
        flat_mag = active_magnitudes.flatten()
        _, prune_idx = flat_mag.topk(k, largest=False)

        # ── Grow ───────────────────────────────────────────────────────
        # Compute gradient for ALL connections: dL/dW_ij = grad_out_i * x_j
        # Use stored activations from the last forward pass
        if self._last_input is not None and self._last_grad_output is not None:
            x = self._last_input
            g = self._last_grad_output
            if x.dim() == 3:
                x = x.reshape(-1, x.size(-1))
                g = g.reshape(-1, g.size(-1))
            # Approximate grad_W = g^T x / batch_size
            grad_W = (g.T @ x) / x.size(0)
        else:
            grad_W = torch.zeros_like(self.weight.data)

        inactive_grad = grad_W.abs() * (1 - self.mask)
        inactive_grad[active] = -float("inf")  # exclude already-active
        flat_grad = inactive_grad.flatten()
        _, grow_idx = flat_grad.topk(k, largest=True)

        # ── Apply mask update ──────────────────────────────────────────
        flat_mask = self.mask.flatten()
        flat_mask[prune_idx] = 0.0
        flat_mask[grow_idx] = 1.0
        self.mask = flat_mask.reshape(self.mask.shape)

        # Zero out newly grown weights (they start at 0)
        flat_weight = self.weight.data.flatten()
        flat_weight[grow_idx] = 0.0
        # Zero out pruned weights
        flat_weight[prune_idx] = 0.0
        self.weight.data = flat_weight.reshape(self.weight.shape)


def cosine_drop_schedule(step: int, total_steps: int, k0: float) -> float:
    """RigL's cosine annealing schedule for drop_fraction."""
    return k0 * (1 + math.cos(math.pi * step / total_steps)) / 2
```

---

## 5. 📊 Empirical Comparison

Gale et al. (2019) and Evci et al. (2020) provide the most comprehensive comparisons. Key takeaways:

| Method | Mask fixed? | Dense model needed? | Achieves dense accuracy at 90% sparsity? |
|--------|------------|--------------------|-----------------------------------------|
| IMP (standard) | Yes (after each round) | Yes (to find mask) | ✅ With fine-tuning |
| LTH (IMP + rewind) | Yes | Yes | ✅ (small models); ❌ (large, needs late rewind) |
| SNIP | Yes (from init) | No | ⚠️ Close, but 1–2% gap at high sparsity |
| SET | No (evolves) | No | ⚠️ Competitive on small models |
| RigL | No (evolves) | No | ✅ Matches IMP at same FLOP budget |

**RigL vs. IMP:** Evci et al. show that RigL, trained for $\alpha$× more steps (same total FLOPs as dense training), matches or exceeds IMP's accuracy at equivalent sparsity. **The key insight:** dynamic sparse training is more FLOP-efficient because it never wastes FLOPs computing gradients for weights the mask will delete — it reallocates them to useful connections as training progresses.

---

## 6. 📚 References

| Reference Name | Brief Summary | Link |
|----------------|---------------|------|
| Frankle & Carlin (2019). "The Lottery Ticket Hypothesis" | Winning ticket subnetworks; IMP + weight rewinding; ICLR 2019 Best Paper | [arXiv:1803.03635](https://arxiv.org/abs/1803.03635) |
| Frankle et al. (2020). "Linear Mode Connectivity and LTH" | LTH at scale requires late rewinding to $w_k$, not $w_0$; stability criterion | [arXiv:1912.05671](https://arxiv.org/abs/1912.05671) |
| Lee, Ajanthan, Torr (2019). "SNIP" | One-shot pre-training pruning via connection sensitivity; no dense model needed | [arXiv:1810.02340](https://arxiv.org/abs/1810.02340) |
| Mocanu et al. (2018). "SET" | Sparse Erdős–Rényi topology evolved during training; Nature Communications | [arXiv:1707.04780](https://arxiv.org/abs/1707.04780) |
| Dettmers & Zettlemoyer (2019). "SNFS" | Gradient-momentum topology reallocation; 5× faster sparse training | [arXiv:1907.04840](https://arxiv.org/abs/1907.04840) |
| Evci et al. (2020). "RigL" | Gradient-magnitude guided growth; fixed-FLOP dynamic training; ICML 2020 | [arXiv:1911.11134](https://arxiv.org/abs/1911.11134) |
| Gale, Elsen, Hooker (2019). "State of Sparsity" | Empirical comparison across methods; magnitude pruning competitive at scale | [arXiv:1902.09574](https://arxiv.org/abs/1902.09574) |
