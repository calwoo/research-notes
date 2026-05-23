# 📐 Second-Order Optimization Methods

## Table of Contents

- [[#1. 💡 Why Second-Order Methods|1. Why Second-Order Methods]]
- [[#2. 📐 Newton's Method|2. Newton's Method]]
  - [[#2.1 Derivation from Taylor Expansion|2.1 Derivation from Taylor Expansion]]
  - [[#2.2 Convergence and Limitations|2.2 Convergence and Limitations]]
  - [[#2.3 💻 PyTorch: Newton Step|2.3 PyTorch: Newton Step]]
- [[#3. 🔮 The Gauss-Newton Approximation|3. The Gauss-Newton Approximation]]
  - [[#3.1 Factoring the Hessian via the Jacobian|3.1 Factoring the Hessian via the Jacobian]]
  - [[#3.2 The Gauss-Newton Matrix|3.2 The Gauss-Newton Matrix]]
  - [[#3.3 💻 PyTorch: Gauss-Newton Step|3.3 PyTorch: Gauss-Newton Step]]
- [[#4. 🐟 Fisher Information and the Natural Gradient|4. Fisher Information and the Natural Gradient]]
  - [[#4.1 The Fisher Information Matrix|4.1 The Fisher Information Matrix]]
  - [[#4.2 Fisher = Gauss-Newton at the MLE|4.2 Fisher = Gauss-Newton at the MLE]]
  - [[#4.3 The Natural Gradient|4.3 The Natural Gradient]]
  - [[#4.4 💻 PyTorch: Empirical Fisher Diagonal|4.4 PyTorch: Empirical Fisher Diagonal]]
- [[#5. 🔗 Connection to Pruning (OBD and OBS)|5. Connection to Pruning (OBD and OBS)]]
- [[#6. 📚 References|6. References]]

---

## 1. 💡 Why Second-Order Methods

First-order methods (SGD, Adam) use only the gradient $g = \nabla_w L$. They treat the loss landscape as locally linear: the best step direction is $-g$. But the curvature of the loss surface (encoded in the Hessian $H = \nabla^2_w L$) matters:

- In a flat direction ($H_{ii} \approx 0$), a large step is safe.
- In a sharp direction ($H_{ii}$ large), even a small step overshoots.

SGD with fixed learning rate $\eta$ takes steps $\delta w = -\eta g$ — the same size in all directions regardless of curvature. This forces a conservative $\eta$ globally, causing slow convergence in flat directions.

**Second-order methods** use $H$ to scale the step in proportion to local curvature: step size $\propto 1/H_{ii}$ in each direction, automatically moving quickly in flat directions and cautiously in sharp ones. This is the same curvature information that [[concepts/deep-learning-engineering/sparsity-pruning/classical-pruning|Optimal Brain Damage and OBS]] exploit for pruning.

---

## 2. 📐 Newton's Method

### 2.1 Derivation from Taylor Expansion

Minimize $L(w)$ starting from $w^{(t)}$. Approximate $L$ by its second-order Taylor expansion around $w^{(t)}$:

$$L(w^{(t)} + \delta w) \approx L(w^{(t)}) + g^\top \delta w + \frac{1}{2}\,\delta w^\top H\, \delta w$$

Setting $\nabla_{\delta w} = 0$:

$$H\, \delta w + g = 0 \implies \delta w^* = -H^{-1} g$$

**Newton step:** $w^{(t+1)} = w^{(t)} - H^{-1} g$.

For a strictly convex quadratic loss, this reaches the exact minimum in one step. For general smooth losses, Newton's method has *quadratic convergence* near a minimum: $\|w^{(t+1)} - w^*\| = O(\|w^{(t)} - w^*\|^2)$.

### 2.2 Convergence and Limitations

**Quadratic convergence** (formal). Under smoothness and strong convexity, after one Newton step:

$$\|w^{(t+1)} - w^*\|_H \leq \frac{L_H}{2\mu}\|w^{(t)} - w^*\|_H^2$$

where $\|\cdot\|_H = \sqrt{(\cdot)^\top H (\cdot)}$ is the Hessian norm, $L_H$ is the Lipschitz constant of $H$, and $\mu$ is the strong convexity constant.

**Practical limitations:**
1. **$O(P^2)$ memory:** The Hessian $H \in \mathbb{R}^{P \times P}$ requires $P^2$ storage — infeasible for $P \sim 10^8$ parameters.
2. **$O(P^3)$ solve:** Solving $H^{-1} g$ via LU factorization costs $O(P^3)$.
3. **Non-convexity:** Neural network losses are non-convex; $H$ may be indefinite or near-singular.

All practical second-order methods are approximations that make Newton tractable.

> [!QUESTION] Exercise 1: Newton step on a quadratic
> *This exercise computes the Newton step explicitly on a 2D quadratic loss.*
>
> > **Prerequisites:** [[#2.1 Derivation from Taylor Expansion|2.1 Derivation from Taylor Expansion]]
>
> Let $L(w) = \frac{1}{2} w^\top A w - b^\top w$ with $A = \begin{pmatrix} 4 & 1 \\ 1 & 2 \end{pmatrix}$, $b = (3, 2)^\top$. Starting from $w^{(0)} = (0, 0)^\top$:
>
> (a) Compute $g = Aw^{(0)} - b$ and $H = A$.
>
> (b) Compute the Newton step $\delta w^* = -H^{-1} g$.
>
> (c) Verify that $w^{(1)} = w^{(0)} + \delta w^*$ is the exact minimum of $L$.

> [!TIP]- Solution to Exercise 1
> **Key insight:** For a quadratic loss, Newton reaches the exact minimum in one step regardless of initial point — this is the "one-step convergence" property that makes Newton so powerful for nearly-quadratic losses (near a minimum).
>
> **(a)** $g = A \cdot (0,0)^\top - (3,2)^\top = (-3, -2)^\top$. $H = A$.
>
> **(b)** $H^{-1} = \frac{1}{4 \cdot 2 - 1 \cdot 1}\begin{pmatrix} 2 & -1 \\ -1 & 4 \end{pmatrix} = \frac{1}{7}\begin{pmatrix} 2 & -1 \\ -1 & 4 \end{pmatrix}$.
> $\delta w^* = -H^{-1}(-3, -2)^\top = \frac{1}{7}\begin{pmatrix} 2 & -1 \\ -1 & 4 \end{pmatrix}\begin{pmatrix} 3 \\ 2 \end{pmatrix} = \frac{1}{7}(4, 5)^\top \approx (0.571, 0.714)^\top$.
>
> **(c)** Check: $g(w^{(1)}) = A w^{(1)} - b = \frac{1}{7}(4 \cdot 4 + 5, 4 + 2 \cdot 5) - (3,2) = (3,2) - (3,2) = 0$. ✓ Exact minimum.

### 2.3 💻 PyTorch: Newton Step

```python
import torch
import torch.nn as nn


def newton_step(
    model: nn.Module,
    loss: torch.Tensor,
    damping: float = 1e-3,
) -> dict[str, torch.Tensor]:
    """
    Compute the Newton update delta_w = -H^{-1} g for a small model.

    WARNING: O(P^2) memory and O(P^3) compute — only feasible for tiny models.
    For large models, use diagonal or Kronecker approximations.

    Returns dict of per-parameter update tensors.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, create_graph=True)
    g = torch.cat([gr.flatten() for gr in grads])

    P = g.numel()
    H = torch.zeros(P, P, device=g.device)

    # Build Hessian row by row via second-order backprop
    for i in range(P):
        h_row = torch.autograd.grad(g[i], params, retain_graph=(i < P - 1))
        H[i] = torch.cat([h.flatten() for h in h_row])

    # Damped Newton step
    H += damping * torch.eye(P, device=H.device)
    delta = torch.linalg.solve(H, -g)

    # Split back into per-parameter updates
    updates = {}
    offset = 0
    for name, p in model.named_parameters():
        n = p.numel()
        updates[name] = delta[offset:offset + n].reshape(p.shape)
        offset += n

    return updates
```

---

## 3. 🔮 The Gauss-Newton Approximation

### 3.1 Factoring the Hessian via the Jacobian

For a loss of the form $L(w) = \ell(f(w))$ where $\ell$ is a scalar loss and $f: \mathbb{R}^P \to \mathbb{R}^m$ is the model output, the Hessian decomposes as:

$$H = J^\top \nabla^2_f \ell\; J + \sum_k \frac{\partial \ell}{\partial f_k} \nabla^2_w f_k$$

where $J = \partial f / \partial w \in \mathbb{R}^{m \times P}$ is the Jacobian of the model output w.r.t. parameters.

The second term involves $\nabla^2_w f_k$ — the Hessian of each output component w.r.t. parameters. Near a solution where $\ell$ is small (residuals $\approx 0$), this term is negligible.

### 3.2 The Gauss-Newton Matrix

Dropping the second term and assuming $\nabla^2_f \ell = I$ (MSE loss) gives the **Gauss-Newton matrix**:

$$G = J^\top J \in \mathbb{R}^{P \times P}$$

The **Gauss-Newton step:** $\delta w^* = -G^{-\!1} g = -(J^\top J)^{-1} J^\top \nabla_f \ell$.

**Why it works:**
- $G = J^\top J$ is always positive semi-definite (no negative curvature issues).
- Computing $G v$ (a matrix-vector product) requires only one forward-backward pass via automatic differentiation — no explicit $H$ needed.
- For least-squares problems, $G$ is the exact Hessian; for general losses, it's a positive-definite approximation.

**For a linear layer $f = Wx$:** $J = I \otimes x^\top$, so $G = (x x^\top) \otimes I$ — the Kronecker product structure that enables layer-wise second-order methods like [[concepts/deep-learning-engineering/sparsity-pruning/classical-pruning|OBS]] and [[concepts/deep-learning-engineering/sparsity-pruning/llm-pruning|SparseGPT]].

> [!QUESTION] Exercise 2: Gauss-Newton for a linear layer
> *This exercise derives the Gauss-Newton matrix for a linear layer with MSE loss.*
>
> > **Prerequisites:** [[#3.2 The Gauss-Newton Matrix|3.2 The Gauss-Newton Matrix]]
>
> Let $y = Wx \in \mathbb{R}^m$ be a linear layer ($W \in \mathbb{R}^{m \times d}$) with MSE loss $L = \frac{1}{2n}\|Wx - y^*\|^2$.
>
> (a) Compute the Jacobian $J = \partial \text{vec}(y) / \partial \text{vec}(W) \in \mathbb{R}^{m \times md}$.
>
> (b) Show that $G = J^\top J = (x x^\top) \otimes I_m / n$.
>
> (c) Explain the Kronecker structure: what does it mean that $G$ is block-diagonal with identical blocks?

> [!TIP]- Solution to Exercise 2
> **Key insight:** The Kronecker structure of $G$ means each output row $W_i$ has the same curvature landscape — the curvature is determined entirely by the input $x$, not by which output neuron we're looking at.
>
> **(a)** $\text{vec}(y) = (I_m \otimes x^\top) \text{vec}(W)$, so $J = I_m \otimes x^\top \in \mathbb{R}^{m \times md}$.
>
> **(b)** $G = J^\top J = (I_m \otimes x)(I_m \otimes x^\top) = I_m \otimes (x x^\top)$. For $n$ examples: $G = I_m \otimes (XX^\top/n) = (XX^\top/n) \otimes I_m$ (Kronecker commutativity). Either way, the block structure has $m$ identical diagonal blocks of size $d \times d$, each equal to $XX^\top/n$.
>
> **(c)** The Kronecker structure $G = (XX^\top/n) \otimes I_m$ means: (1) the curvature for any output neuron $i$ is the same $d \times d$ matrix $XX^\top/n$ — no inter-output coupling; (2) the rows of $W$ (output neurons) are *independent* in the loss landscape. This justifies the row-by-row OBS updates in SparseGPT: each row can be pruned without affecting any other row's optimal solution.

### 3.3 💻 PyTorch: Gauss-Newton Step

```python
import torch
import torch.nn as nn


def gauss_newton_hvp(
    model: nn.Module,
    loss: torch.Tensor,
    outputs: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Gauss-Newton matrix-vector product G v = J^T (J v)
    via two autodiff passes (Pearlmutter trick).

    v: (P,) vector to multiply by G
    Returns: (P,) result of G v
    """
    params = list(model.parameters())
    grads = torch.autograd.grad(outputs.sum(), params, create_graph=True)
    g = torch.cat([gr.flatten() for gr in grads])

    # Forward pass: compute J v = d(outputs)/dw · v
    Jv = (g * v).sum()

    # Backward pass: compute J^T (J v)
    Jt_Jv = torch.autograd.grad(Jv, params)
    return torch.cat([h.flatten() for h in Jt_Jv])
```

---

## 4. 🐟 Fisher Information and the Natural Gradient

### 4.1 The Fisher Information Matrix

For a probabilistic model $p(y | x, w)$ parameterized by $w$, the **Fisher information matrix** is:

$$F = \mathbb{E}_{x, y \sim p(y|x,w)}\!\left[\nabla_w \log p(y | x, w)\; (\nabla_w \log p(y | x, w))^\top\right]$$

The Fisher matrix measures how sensitive the model's predictions are to parameter changes: $F_{ij}$ is the expected product of partial log-likelihood derivatives w.r.t. $w_i$ and $w_j$.

Equivalently (by the Fisher identity, at the MLE):

$$F = -\mathbb{E}\!\left[\nabla^2_w \log p(y | x, w)\right]$$

So the Fisher is the *negative expected Hessian* of the log-likelihood — a positive semi-definite approximation to the (possibly indefinite) Hessian $H$ of the empirical loss.

### 4.2 Fisher = Gauss-Newton at the MLE

For models in the exponential family (logistic regression, softmax classifiers, Gaussian regression):

$$F = G \quad \text{at the maximum likelihood estimator}$$

*Intuition:* At the MLE, the Gauss-Newton matrix $G = J^\top \nabla^2_f \ell\, J$ (using the exact loss Hessian $\nabla^2_f \ell$) equals the Fisher $F = \mathbb{E}[g g^\top]$. This means the Fisher diagonal $F_{ii} = \mathbb{E}[g_i^2]$ is an approximation to the Hessian diagonal $H_{ii}$, which is why it is used in [[concepts/deep-learning-engineering/sparsity-pruning/classical-pruning|OBD's]] empirical Fisher approximation.

> [!WARNING] Empirical vs. true Fisher
> The *empirical Fisher* uses the observed labels $y$ (from the training set) instead of samples from the model's predictive distribution:
> $$\hat{F} = \frac{1}{n}\sum_{i=1}^n \nabla_w \log p(y_i | x_i, w)\; (\nabla_w \log p(y_i | x_i, w))^\top$$
> The true Fisher samples $y \sim p(y | x, w)$ (the model's own predictions). The two are equal at the MLE but can differ substantially off-manifold. The empirical Fisher is cheaper to compute (no sampling) and is standard in practice.

### 4.3 The Natural Gradient

Standard gradient descent moves in the Euclidean metric on parameter space:
$$w^{(t+1)} = w^{(t)} - \eta g$$

The **natural gradient** moves in the *Fisher information metric* — the geometry induced by the KL divergence between model distributions:

$$\tilde{g} = F^{-1} g$$

$$w^{(t+1)} = w^{(t)} - \eta F^{-1} g$$

The natural gradient is invariant to reparameterization: if you change coordinates in parameter space, the natural gradient update produces the same change in the *function* (model distribution), unlike the Euclidean gradient.

**K-FAC (Kronecker-Factored Approximate Curvature)** approximates $F^{-1}$ using the Kronecker factorization of the Fisher for each layer, making natural gradient descent practical for neural networks.

> [!QUESTION] Exercise 3: Natural gradient vs. gradient descent on a curved surface
> *This exercise shows why the natural gradient converges faster for ill-conditioned problems.*
>
> > **Prerequisites:** [[#4.3 The Natural Gradient|4.3 The Natural Gradient]]
>
> Minimize $L(w) = w_1^2 + 100 w_2^2$ (a "valley" with very different curvatures in the two directions). The Fisher/Hessian is $F = H = \text{diag}(2, 200)$.
>
> (a) Compute the gradient descent step from $w^{(0)} = (1, 1)^\top$ with $\eta = 0.01$.
>
> (b) Compute the natural gradient step from the same point (use the same $\eta$).
>
> (c) How many gradient descent steps does it take to converge to $|w| < 0.01$ if $\eta = 0.01$? How many natural gradient steps?

> [!TIP]- Solution to Exercise 3
> **Key insight:** Gradient descent must use a conservative learning rate (limited by the sharpest direction), causing slow convergence in flat directions. Natural gradient automatically scales by $1/F_{ii}$, converging uniformly in all directions.
>
> **(a)** $g = (2 w_1, 200 w_2) = (2, 200)$. GD step: $w^{(1)} = (1 - 0.01 \times 2, 1 - 0.01 \times 200) = (0.98, -1.0)$. Overshoots in $w_2$!
>
> **(b)** $\tilde{g} = F^{-1} g = \text{diag}(1/2, 1/200)(2, 200)^\top = (1, 1)^\top$. Natural gradient step: $w^{(1)} = (1 - 0.01 \times 1, 1 - 0.01 \times 1) = (0.99, 0.99)$. No overshoot.
>
> **(c)** GD with $\eta = 0.01$: convergence rate in $w_1$: $|w_1^{(t)}| = (1 - 0.02)^t$. In $w_2$: $|w_2^{(t)}| = (1 - 2.0)^t$ — diverges! Must use $\eta < 1/100 = 0.01$ to keep $w_2$ stable. With $\eta = 0.005$: rate in $w_1$ is $0.99^t \approx e^{-0.01t}$; needs $t \approx 460$ steps. Natural gradient with $\eta = 0.01$: both directions converge at rate $0.99^t$; needs $t \approx 460$ steps in both — but doesn't require a smaller $\eta$.

### 4.4 💻 PyTorch: Empirical Fisher Diagonal

```python
import torch
import torch.nn as nn


def empirical_fisher_diagonal(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    n_batches: int = 64,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """
    Compute empirical Fisher diagonal: F_ii = E[g_i^2].

    This approximates the Hessian diagonal and is the quantity used in
    Optimal Brain Damage (LeCun et al. 1990) saliency computation.

    Returns dict: parameter_name -> F_ii tensor (same shape as parameter).
    """
    fisher = {
        name: torch.zeros_like(p)
        for name, p in model.named_parameters()
        if p.requires_grad
    }
    model.eval()
    n_samples = 0

    for i, (inputs, targets) in enumerate(loader):
        if i >= n_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        model.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()

        batch_size = inputs.size(0)
        for name, p in model.named_parameters():
            if p.grad is not None:
                fisher[name].add_(p.grad.data.pow(2).mul_(batch_size))
        n_samples += batch_size

    return {name: f.div_(n_samples) for name, f in fisher.items()}


def obd_saliency_from_fisher(
    model: nn.Module,
    fisher: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Compute OBD saliency s_i = 0.5 * F_ii * w_i^2 from precomputed Fisher.
    This is the connection between second-order optimization and pruning.
    """
    return {
        name: 0.5 * fisher[name] * p.data.pow(2)
        for name, p in model.named_parameters()
        if name in fisher
    }
```

---

## 5. 🔗 Connection to Pruning (OBD and OBS)

The second-order methods described here are the mathematical prerequisites for [[concepts/deep-learning-engineering/sparsity-pruning/classical-pruning|classical pruning]]:

| Concept | Used by | Connection |
|---------|---------|-----------|
| Taylor expansion | OBD, OBS | Pruning = constrained minimization of $\delta w^\top H \delta w$ |
| Diagonal Hessian approximation | OBD | $H_{ii} \approx F_{ii} = \mathbb{E}[g_i^2]$ |
| Fisher = Hessian identity | OBD practical implementation | Fisher diagonal as surrogate for diagonal Hessian |
| Gauss-Newton matrix | SparseGPT | Layer Hessian $H = XX^\top/n$ is the Gauss-Newton matrix for the layer's reconstruction loss |
| Woodbury identity | OBS update | Rank-1 update of $H^{-1}$ after each weight deletion |
| KKT conditions | OBS | Constrained optimization derivation of the weight correction |

The Adam optimizer's second moment $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$ is a running estimate of the Fisher diagonal $F_{ii} = \mathbb{E}[g_i^2]$. An Adam-trained model therefore *already contains an estimate of the OBD saliency* in its optimizer state — no extra computation needed.

---

## 6. 📚 References

| Reference Name | Brief Summary | Link |
|----------------|---------------|------|
| Nocedal & Wright (2006). "Numerical Optimization" | Comprehensive reference for Newton, quasi-Newton, and conjugate gradient methods | [Springer](https://link.springer.com/book/10.1007/978-0-387-40065-5) |
| Amari (1998). "Natural Gradient Works Efficiently in Learning" | Introduced natural gradient for neural network training | [Neural Computation](https://direct.mit.edu/neco/article/10/2/251/6143) |
| Martens & Grosse (2015). "Optimizing Neural Networks with Kronecker-Factored Approximate Curvature" | K-FAC: practical natural gradient via Kronecker Fisher approximation | [arXiv:1503.05671](https://arxiv.org/abs/1503.05671) |
| LeCun et al. (1990). "Optimal Brain Damage" | First application of second-order Hessian saliency to neural network pruning | [NeurIPS 1989](https://proceedings.neurips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html) |
| Hassibi & Stork (1993). "Optimal Brain Surgeon" | Full inverse-Hessian pruning with KKT-derived weight correction | [NeurIPS 1992](https://proceedings.neurips.cc/paper/1992/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html) |
| Martens (2014). "New Insights and Perspectives on the Natural Gradient Method" | Survey connecting Fisher, Gauss-Newton, and natural gradient | [arXiv:1412.1193](https://arxiv.org/abs/1412.1193) |
