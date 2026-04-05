# Muon: A Scalable Matrix Orthogonalization Optimizer for LLM Training

*Jingyuan Liu, Jianlin Su, et al. (Kimi Team, Moonshot AI) — arXiv 2502.16982, February 2025*

| Dimension | Prior State | This Paper | Key Result |
|-----------|-------------|------------|------------|
| Scalability | Original Muon untested beyond small LMs | Weight decay + per-parameter update scaling | Validated on 3B/16B MoE at 5.7T tokens |
| Compute efficiency | AdamW is the de facto standard | Muon under compute-optimal training | ~2x fewer FLOPs to match AdamW loss |
| Theoretical grounding | Heuristic motivation for orthogonalization | Steepest descent under RMS-to-RMS operator norm | Update = UV^T from SVD is the exact dual solution |
| Hyperparameter transfer | Optimizer-specific tuning required | Scaling factor ties update RMS to parameter shape | Muon reuses AdamW learning rate and weight decay |
| Model quality | DeepSeek-V2-Lite MMLU 58.3, MATH 17.1 | Moonlight (same arch/tokens) | MMLU 70.0, MATH 45.3, HumanEval 48.1 |

## Relations

**Builds on:** [[papers/shampoo|Shampoo]] *(no note yet)*, [[papers/adam|Adam / AdamW]] *(no note yet)*
**Extended by:** [[papers/muon-hyperball|Hyperball / HyperMuon]] *(no note yet)*
**Concepts used:** [[concepts/randomized-algorithms/metric-geometry-and-dimension-reduction|Metric Geometry and Dimension Reduction]], [[concepts/optimization/steepest-descent|Steepest Descent and Dual Norms]] *(no note yet)*

---

## Table of Contents

- [[#1. Overview and Motivation|1. Overview and Motivation]]
- [[#2. The Muon Algorithm|2. The Muon Algorithm]]
  - [[#2.1 Core Update Rule|2.1 Core Update Rule]]
  - [[#2.2 Newton-Schulz Orthogonalization|2.2 Newton-Schulz Orthogonalization]]
  - [[#2.3 Scaled Muon for Large Models|2.3 Scaled Muon for Large Models]]
- [[#3. Derivation from First Principles|3. Derivation from First Principles]]
  - [[#3.1 Metrizing the Linear Layer|3.1 Metrizing the Linear Layer]]
  - [[#3.2 Steepest Descent Under the Operator Norm|3.2 Steepest Descent Under the Operator Norm]]
  - [[#3.3 The Dual Solution is UV^T|3.3 The Dual Solution is UV^T]]
  - [[#3.4 Learning Rate Transfer Across Width|3.4 Learning Rate Transfer Across Width]]
- [[#4. Connection to Shampoo and Second-Order Methods|4. Connection to Shampoo and Second-Order Methods]]
- [[#5. The Geometric Interpretation: Hyperball Optimization|5. The Geometric Interpretation: Hyperball Optimization]]
- [[#6. The Explore-Exploit Perspective|6. The Explore-Exploit Perspective]]
- [[#7. Su Jianlin's Spectral Norm Perspective|7. Su Jianlin's Spectral Norm Perspective]]
- [[#8. Why Orthogonalization Helps in Practice|8. Why Orthogonalization Helps in Practice]]
- [[#9. Distributed Implementation|9. Distributed Implementation]]
- [[#10. Practical Considerations and Hyperparameters|10. Practical Considerations and Hyperparameters]]
- [[#11. Experimental Results|11. Experimental Results]]
- [[#References|References]]

---

## 1. Overview and Motivation

💡 The *Muon optimizer* (Momentum + Update Orthogonalization via Newton-Schulz) is a matrix-aware optimizer for neural network weight matrices. It was originally introduced by Keller Jordan for small-scale transformer training and scaled to production LLM pretraining by the Kimi/Moonshot AI team in the paper "Muon is Scalable for LLM Training" (Liu et al., 2025).

The central operation is simple to state: after accumulating a standard Nesterov momentum gradient, Muon replaces the momentum buffer with its nearest *semi-orthogonal* matrix before applying the weight update. If the momentum matrix $\mathbf{M} \in \mathbb{R}^{m \times n}$ has *singular value decomposition* (SVD) $\mathbf{M} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$, then Muon uses:

$$\mathbf{O} = \mathbf{U}\mathbf{V}^\top$$

as the update direction, discarding all singular value magnitude information entirely.

The central question is: why should this be a good idea? The answer, developed in the sections below, connects to:

1. **Steepest descent** under the natural operator norm for linear layers.
2. **Approximate second-order optimization** via the connection to Shampoo.
3. **Geometric** intuitions about update isotropy across singular directions.
4. **Explore-exploit trade-offs** in traversing the loss landscape.

> [!INFO] Historical note
> The original Muon implementation was released by Keller Jordan as a NanoGPT training speedrun optimizer in 2024. The name was coined then. The theoretical derivation connecting it to steepest descent under the RMS-to-RMS operator norm was later worked out by Jeremy Bernstein and further analyzed by Jianlin Su (Su Jianlin, known for the RoPE positional embedding). The Kimi Team's 2025 paper constitutes the first rigorous large-scale validation.

---

## 2. The Muon Algorithm

### 2.1 Core Update Rule

🔧 The Muon update for a weight matrix $\mathbf{W}_t \in \mathbb{R}^{m \times n}$ proceeds in three steps:

**Step 1 — Momentum accumulation:**
$$\mathbf{M}_t = \mu \mathbf{M}_{t-1} + \nabla_\mathbf{W} \mathcal{L}_t(\mathbf{W}_{t-1})$$

where $\mu \in [0, 1)$ is the momentum coefficient (typically $\mu = 0.95$).

**Step 2 — Newton-Schulz orthogonalization:**
$$\mathbf{O}_t = \operatorname{NS}(\mathbf{M}_t) \approx \mathbf{U}\mathbf{V}^\top \quad \text{where } \mathbf{M}_t = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$$

**Step 3 — Weight update:**
$$\mathbf{W}_t = \mathbf{W}_{t-1} - \eta_t \mathbf{O}_t$$

The key insight is that $\operatorname{NS}(\cdot)$ is a *matrix function* — it can be evaluated without computing an explicit SVD by iterating a polynomial approximation directly in matrix arithmetic.

![Muon algorithm pseudocode from Keller Jordan's blog post](figures/muon-optimizer/kellerjordan-muon-algo.png)
*The Muon update rule as presented by Keller Jordan: momentum accumulation followed by Newton-Schulz orthogonalization, applied only to 2D weight matrices.*

> [!NOTE] What does "nearest semi-orthogonal" mean?
> A matrix $\mathbf{O} \in \mathbb{R}^{m \times n}$ with $m \geq n$ is *semi-orthogonal* if $\mathbf{O}^\top \mathbf{O} = \mathbf{I}_n$. Equivalently, all singular values equal 1. The nearest semi-orthogonal matrix to $\mathbf{M}$ (in Frobenius norm) is $\mathbf{U}\mathbf{V}^\top$ where $\mathbf{M} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$. This is the *polar factor* of $\mathbf{M}$.

### 2.2 Newton-Schulz Orthogonalization

📐 Computing the full SVD of $\mathbf{M}_t$ at every step is prohibitively expensive for large matrices ($O(mn\min(m,n))$ per step). The Newton-Schulz iteration provides a polynomial approximation that converges to the polar factor using only matrix multiplications.

**Definition (Newton-Schulz Iteration).** Given an initial matrix $\mathbf{X}_0 = \mathbf{M}_t / \|\mathbf{M}_t\|_F$, the quintic Newton-Schulz update is:

$$\mathbf{X}_{k+1} = a\mathbf{X}_k + b(\mathbf{X}_k\mathbf{X}_k^\top)\mathbf{X}_k + c(\mathbf{X}_k\mathbf{X}_k^\top)^2\mathbf{X}_k$$

with coefficients $(a, b, c) = (3.4445, -4.7750, 2.0315)$.

*Convergence mechanism:* Observe that if $\mathbf{X}_k = \mathbf{U}\boldsymbol{\Sigma}_k\mathbf{V}^\top$, then $\mathbf{X}_{k+1} = \mathbf{U}\phi(\boldsymbol{\Sigma}_k)\mathbf{V}^\top$ where $\phi(s) = as + bs^3 + cs^5$ acts scalar-wise on each singular value. The singular vectors $\mathbf{U}, \mathbf{V}$ are preserved. The polynomial $\phi$ is chosen so that:

$$\phi^{(N)}(s) \to 1 \quad \forall s \in (0, 1] \text{ as } N \to \infty$$

The Frobenius-normalization of $\mathbf{M}_t$ ensures all singular values land in $[0, 1]$, so $\phi$ acts as a contracting map toward 1 on the singular value spectrum.

**Proposition.** After $N = 5$ iterations with the optimized quintic coefficients, the approximation error is within $\pm 0.3$ for all singular values, which is empirically sufficient (loss curves are unaffected by errors of this magnitude).

![Baseline Newton-Schulz polynomial convergence (Keller Jordan)](figures/muon-optimizer/kellerjordan-ns-poly-baseline.png)
*Baseline quintic polynomial $\phi(s) = as + bs^3 + cs^5$ applied iteratively: singular values converge toward 1 but require several iterations with the default coefficients.*

![Tuned Newton-Schulz polynomial convergence (Keller Jordan)](figures/muon-optimizer/kellerjordan-ns-poly-tuned.png)
*Optimized Newton-Schulz coefficients exhibiting steeper growth near $s = 0$, achieving faster convergence to 1 in fewer iterations — the coefficients $(a, b, c) = (3.4445, -4.7750, 2.0315)$ used in practice.*

> [!TIP] Why quintic and not cubic?
> A cubic polynomial $\phi(s) = as + bs^3$ only has two degrees of freedom, limiting convergence speed. The quintic adds a fifth-degree term, giving faster convergence toward sign$(s) = 1$ for $s > 0$. The coefficients $(a, b, c)$ are optimized numerically to minimize the number of iterations required for convergence to within 0.3 over $[0, 1]$.

**FLOP overhead.** Each Newton-Schulz step requires computing $\mathbf{X}_k\mathbf{X}_k^\top \in \mathbb{R}^{m \times m}$ and then two subsequent multiplications. The total FLOP overhead relative to the forward/backward pass is:

$$\text{overhead} = \frac{T \cdot m}{B}$$

where $T = 5$ is the number of NS iterations, $m$ is the model dimension, and $B$ is the batch size in tokens. For typical settings (NanoGPT: $\sim 0.7\%$; Llama 405B: $\sim 0.5\%$).

### 2.3 Scaled Muon for Large Models

⚠️ The basic Muon algorithm produces updates of inconsistent *root-mean-square* (RMS) magnitude depending on the shape of the parameter matrix. The Kimi team identifies this as the primary obstacle to scalability.

**Lemma (Liu et al., 2025).** For a full-rank weight matrix $\mathbf{W} \in \mathbb{R}^{A \times B}$, the theoretical update RMS under the basic Muon rule is:

$$\operatorname{RMS}(\mathbf{O}) = \frac{1}{\sqrt{\max(A, B)}}$$

*Proof sketch.* $\mathbf{O} = \mathbf{U}\mathbf{V}^\top$ has $\min(A, B)$ singular values all equal to 1. The Frobenius norm is $\|\mathbf{O}\|_F = \sqrt{\min(A, B)}$. The RMS is $\|\mathbf{O}\|_F / \sqrt{AB} = \sqrt{\min(A,B)/AB} = 1/\sqrt{\max(A,B)}$.

This means Muon naturally applies smaller-RMS updates to larger matrices — a shape-dependent bias. The fix is a rescaling:

**The scaled Muon update (MuonW):**

$$\mathbf{W}_t = \mathbf{W}_{t-1} - \eta_t \left(0.2 \cdot \mathbf{O}_t \cdot \sqrt{\max(A, B)} + \lambda \mathbf{W}_{t-1}\right)$$

The factor $0.2\sqrt{\max(A,B)}$ restores a target RMS of $0.2$, matching AdamW's typical effective update scale. The $\lambda \mathbf{W}_{t-1}$ term is standard *decoupled weight decay*.

> [!WARNING] Which parameters get Muon?
> Muon is designed for 2D weight matrices in linear layers. Embedding tables, classifier heads, 1D bias vectors, and layer norm parameters should use AdamW instead. The QKV projection matrices should have their Q, K, V portions treated as separate matrices (not concatenated) if their feature spaces are semantically distinct.

---

## 3. Derivation from First Principles

📐 The following derivation, due to Jeremy Bernstein, shows that the Muon update is *not* heuristic — it is the exact solution to a principled constrained optimization problem.

### 3.1 Metrizing the Linear Layer

**Definition (RMS norm).** For a vector $\mathbf{v} \in \mathbb{R}^d$, the *root-mean-square norm* is:
$$\|\mathbf{v}\|_{\text{RMS}} := \sqrt{\frac{1}{d} \sum_{i=1}^d v_i^2} = \frac{\|\mathbf{v}\|_2}{\sqrt{d}}$$

The RMS norm measures "average entry size" rather than total magnitude. For activations in a well-initialized network, $\|\mathbf{x}\|_{\text{RMS}} \approx 1$ throughout training.

**Definition (RMS-to-RMS operator norm).** For a linear map $\mathbf{W} : \mathbb{R}^{d_{\text{in}}} \to \mathbb{R}^{d_{\text{out}}}$, define:
$$\|\mathbf{W}\|_{\text{RMS}\to\text{RMS}} := \max_{\mathbf{x} \neq 0} \frac{\|\mathbf{W}\mathbf{x}\|_{\text{RMS}}}{\|\mathbf{x}\|_{\text{RMS}}} = \sqrt{\frac{d_{\text{in}}}{d_{\text{out}}}} \cdot \|\mathbf{W}\|_{\text{op}}$$

where $\|\mathbf{W}\|_{\text{op}} = \sigma_{\max}(\mathbf{W})$ is the *spectral norm* (largest singular value).

*Derivation.* Write $\|\mathbf{W}\mathbf{x}\|_{\text{RMS}} = \|\mathbf{W}\mathbf{x}\|_2 / \sqrt{d_{\text{out}}}$ and $\|\mathbf{x}\|_{\text{RMS}} = \|\mathbf{x}\|_2 / \sqrt{d_{\text{in}}}$. The ratio is $({\sqrt{d_{\text{in}}}}/{\sqrt{d_{\text{out}}}}) \cdot (\|\mathbf{W}\mathbf{x}\|_2 / \|\mathbf{x}\|_2)$. Maximizing over $\mathbf{x}$ gives $\sqrt{d_{\text{in}}/d_{\text{out}}} \cdot \sigma_{\max}(\mathbf{W})$.

![Operator norm as maximum stretch factor (Bernstein, 2025)](figures/muon-optimizer/bernstein-standoff.png)
*The operator norm (spectral norm) measures the maximum factor by which a matrix can stretch an input vector — geometrically, the radius of the ellipsoid image of the unit sphere.*

### 3.2 Steepest Descent Under the Operator Norm

The standard gradient descent update $\mathbf{W} \leftarrow \mathbf{W} - \eta \nabla_\mathbf{W} \mathcal{L}$ is steepest descent under the *Frobenius norm* constraint $\|\Delta\mathbf{W}\|_F \leq \eta$. Muon instead uses steepest descent under the RMS-to-RMS operator norm constraint.

![Different distance notions lead to different optimization theories (Bernstein, 2025)](figures/muon-optimizer/bernstein-geometry.png)
*Choosing the right geometry for weight updates: Frobenius norm (SGD), element-wise $\ell^\infty$ (Adam), or operator norm (Muon) each induce different steepest descent directions and distinct optimization theories.*

**Problem formulation.** Given the linearized loss $\langle \nabla_\mathbf{W}\mathcal{L}, \Delta\mathbf{W}\rangle$, find:

$$\Delta\mathbf{W}^* = \arg\min_{\Delta\mathbf{W}} \langle \nabla_\mathbf{W}\mathcal{L}, \Delta\mathbf{W}\rangle \quad \text{subject to} \quad \|\Delta\mathbf{W}\|_{\text{RMS}\to\text{RMS}} \leq \eta$$

Since $\|\Delta\mathbf{W}\|_{\text{RMS}\to\text{RMS}} = \sqrt{d_{\text{in}}/d_{\text{out}}} \cdot \sigma_{\max}(\Delta\mathbf{W})$, this is equivalent to constraining the spectral norm $\sigma_{\max}(\Delta\mathbf{W}) \leq \eta \sqrt{d_{\text{out}}/d_{\text{in}}}$.

### 3.3 The Dual Solution is UV^T

**Proposition.** Let $\mathbf{G} = \nabla_\mathbf{W}\mathcal{L} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ with SVD $\mathbf{G} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$. The solution to the constrained steepest descent problem is:

$$\Delta\mathbf{W}^* = -\eta \sqrt{\frac{d_{\text{out}}}{d_{\text{in}}}} \cdot \mathbf{U}\mathbf{V}^\top$$

**Proof.** The dual norm of $\|\cdot\|_{\text{op}}$ (spectral norm) is the *nuclear norm* $\|\cdot\|_* = \sum_i \sigma_i$. By the duality of norms, the steepest descent direction under spectral norm constraint is the maximizer of:

$$\langle \mathbf{G}, \Delta\mathbf{W}\rangle \quad \text{subject to} \quad \|\Delta\mathbf{W}\|_{\text{op}} \leq c$$

The nuclear norm/spectral norm duality gives: $\langle \mathbf{G}, \Delta\mathbf{W}\rangle \leq \|\mathbf{G}\|_* \cdot \|\Delta\mathbf{W}\|_{\text{op}}$, with equality when $\Delta\mathbf{W} \propto \mathbf{U}\mathbf{V}^\top$. To see this: $\langle \mathbf{G}, \mathbf{U}\mathbf{V}^\top\rangle = \operatorname{tr}(\mathbf{V}\mathbf{U}^\top \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top) = \operatorname{tr}(\boldsymbol{\Sigma}) = \|\mathbf{G}\|_*$, which is maximal. The scaling $\sqrt{d_{\text{out}}/d_{\text{in}}}$ comes from the RMS-to-RMS normalization factor. $\square$

**Key conclusion: the Muon update direction $\mathbf{U}\mathbf{V}^\top$ is the exact steepest descent direction under the RMS-to-RMS operator norm constraint on weight perturbations.**

> [!EXAMPLE] Intuition via 2x2 matrices
> Suppose $\mathbf{G} = \begin{pmatrix} 10 & 0 \\ 0 & 1 \end{pmatrix}$. Standard gradient descent takes a large step in the first singular direction and a tiny step in the second. Muon takes $\mathbf{O} = \mathbf{I}$, an equal step in both directions. Under the spectral norm constraint, the second direction is "free" — we can take a full unit step there at no cost to the constraint budget.

### 3.4 Learning Rate Transfer Across Width

🔑 A critical practical consequence: **steepest descent under the RMS-to-RMS operator norm automatically achieves maximal update parameterization (muP) learning rate transfer across model width.**

*Heuristic argument.* In muP, the correct learning rate for a weight matrix scales as $1/\text{fan\_in}$. The RMS-to-RMS operator norm includes a factor $\sqrt{d_{\text{in}}/d_{\text{out}}}$, which under the steepest descent normalization produces a $\sqrt{d_{\text{out}}/d_{\text{in}}}$ scaling in $\Delta\mathbf{W}^*$. This matches the muP prescription: updates should be $O(1/\sqrt{d_{\text{in}}})$ in RMS, independent of width, which is achieved by the polar factor $\mathbf{U}\mathbf{V}^\top$ (whose RMS is $\sim 1/\sqrt{\max(d_{\text{out}}, d_{\text{in}})}$ — the Lemma from §2.3).

![Dualized vs. conventional training on CIFAR across model widths (Bernstein, 2025)](figures/muon-optimizer/bernstein-transfer.png)
*Empirical validation of learning rate transfer: dualized training (operator-norm steepest descent, equivalent to Muon) maintains consistent loss across model widths without retuning, while conventional training requires width-specific learning rates.*

---

## 4. Connection to Shampoo and Second-Order Methods

🔗 The *Shampoo* optimizer maintains Kronecker-factored approximations of the gradient covariance and applies their inverse square-root as a preconditioner. Concretely, for $\mathbf{G} \in \mathbb{R}^{m \times n}$, Shampoo computes:

![Shampoo algorithm (Keller Jordan's blog)](figures/muon-optimizer/kellerjordan-shampoo-algo.png)
*The Shampoo update rule: accumulate gradient outer products $\mathbf{L}_t, \mathbf{R}_t$ and precondition with their inverse fourth roots. Muon recovers this update in the limiting case where all gradient history is discarded.*

$$\mathbf{L}_t = \sum_{s \leq t} \mathbf{G}_s \mathbf{G}_s^\top \in \mathbb{R}^{m \times m}, \quad \mathbf{R}_t = \sum_{s \leq t} \mathbf{G}_s^\top \mathbf{G}_s \in \mathbb{R}^{n \times n}$$

and applies the update:

$$\Delta\mathbf{W} \propto -\mathbf{L}_t^{-1/4} \mathbf{G}_t \mathbf{R}_t^{-1/4}$$

**Connection to Muon.** Now suppose the gradient has no memory across steps: $\mathbf{G}_s \approx \mathbf{G}_t$ for all $s$. Then:

$$\mathbf{L}_t \approx T \cdot \mathbf{G}_t\mathbf{G}_t^\top = T \cdot \mathbf{U}\boldsymbol{\Sigma}^2\mathbf{U}^\top$$

$$\mathbf{L}_t^{-1/4} \approx T^{-1/4} \cdot \mathbf{U}\boldsymbol{\Sigma}^{-1/2}\mathbf{U}^\top$$

Similarly, $\mathbf{R}_t^{-1/4} \approx T^{-1/4} \cdot \mathbf{V}\boldsymbol{\Sigma}^{-1/2}\mathbf{V}^\top$. Substituting:

$$\mathbf{L}_t^{-1/4}\mathbf{G}_t\mathbf{R}_t^{-1/4} \approx T^{-1/2} \cdot \mathbf{U}\boldsymbol{\Sigma}^{-1/2}\mathbf{U}^\top \cdot \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top \cdot \mathbf{V}\boldsymbol{\Sigma}^{-1/2}\mathbf{V}^\top = T^{-1/2} \cdot \mathbf{U}\mathbf{V}^\top$$

*Muon is a limiting case of Shampoo* where gradient memory is reset each step. This connection shows that Muon is implicitly doing approximate second-order preconditioning.

> [!NOTE] Why Muon is faster than Shampoo
> Shampoo requires maintaining and inverting the $m \times m$ and $n \times n$ covariance matrices, costing $O(m^3 + n^3)$ per step (or using approximations). Muon's Newton-Schulz iteration acts directly on $\mathbf{G} \in \mathbb{R}^{m \times n}$ using matrix multiplications, with no explicit covariance accumulation. The computational savings are substantial for large matrices.

> [!QUESTION] Open question: can Muon recover Shampoo's benefits from gradient memory?
> Muon discards all history of the gradient direction and uses only the current momentum buffer. Shampoo accumulates gradient outer products over many steps. Whether a variant of Muon that accumulates some spectral memory across steps could improve further is an open research direction.

---

## 5. The Geometric Interpretation: Hyperball Optimization

🔵 The *hyperball* perspective, developed by Wen (2025), frames Muon as optimizing on a constrained manifold to achieve *designed* rather than *accidental* update-to-weight ratios.

**The instability of standard optimizers.** In AdamW and SGD, the ratio $\|\mathbf{u}_t\|_F / \|\mathbf{W}_t\|_F$ (where $\mathbf{u}_t$ is the raw update before scaling by $\eta$) evolves unpredictably across layers, widths, and depths. The effective step size along each eigen-direction depends on accumulated gradient statistics in ways that are hard to control.

**The hyperball normalization scheme.** Define normalization to a sphere of radius $R$ as:

$$\operatorname{Normalize}_R(\mathbf{X}) := R \cdot \frac{\mathbf{X}}{\|\mathbf{X}\|_F}$$

The hyperball update rule is:

$$\mathbf{W}_{t+1} = \operatorname{Normalize}_R\!\left(\mathbf{W}_t - \eta \cdot \operatorname{Normalize}_R(\mathbf{u}_t)\right)$$

where $R = \|\mathbf{W}_0\|_F$ is fixed at initialization. This ensures:

1. **Weight norms are pinned:** $\|\mathbf{W}_t\|_F = R$ for all $t$.
2. **Update norms are pinned:** $\|\text{applied update}\|_F = \eta$ for all $t$.
3. **The ratio $\eta / R$ is the designed step size**, predictable and transferable.

**Connection to Muon.** The "normalize the update" step $\operatorname{Normalize}_R(\mathbf{u}_t)$ applied to a matrix generalizes naturally to *matching the singular vectors* rather than just the Frobenius norm. Specifically, if one further constrains the update to lie on the *Stiefel manifold* (the set of semi-orthogonal matrices), the minimum-distortion lift from a vector update to a matrix update is exactly $\mathbf{U}\mathbf{V}^\top$.

**Practical benefit.** By pinning the update-to-weight ratio to a designed constant, hyperball optimization enables hyperparameter transfer across model width and depth without muP reparameterization.

![Hyperball optimizer speedup vs. AdamW and MuonW baselines (Wen, 2025)](figures/muon-optimizer/hyperball-speedup.png)
*Empirical performance of the Hyperball optimizer relative to AdamW and MuonW: by enforcing a hard norm constraint on both weights and updates, Hyperball achieves faster convergence and improved sample efficiency.*

> [!WARNING] Hyperball vs. Muon
> Hyperball normalization also normalizes the *weights* themselves (projecting $\mathbf{W}_t$ back onto a sphere at each step). Standard Muon does not do this — it only orthogonalizes the update, not the weights. The weight decay term in MuonW provides a soft analogue: it prevents weight norms from growing unboundedly, but does not enforce a hard norm constraint.

---

## 6. The Explore-Exploit Perspective

🗺️ An alternative lens, developed by Paperplanet (2025), interprets Muon's update as balancing *exploration* of gradient directions against *exploitation* of the steepest-descent path.

**The exploit-only baseline.** Standard gradient descent (and Adam) follow the gradient direction, concentrating the update budget on singular directions with large singular values. If $\mathbf{G} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$, then:

$$\text{Adam update} \approx -\mathbf{U}\operatorname{sign}(\boldsymbol{\Sigma})\mathbf{V}^\top \quad \text{(element-wise sign adaptation)}$$

In practice, Adam's sign operation on individual entries does not align with the SVD structure, but the effect is qualitatively similar: it amplifies directions according to gradient magnitude.

**Muon's exploration.** By normalizing *all* singular values to 1:

$$\mathbf{O}_t = \sum_{i=1}^r \mathbf{u}_i \mathbf{v}_i^\top$$

Muon assigns *equal priority* to every singular direction, regardless of whether the corresponding singular value is large or small. This forces the optimizer to explore directions where the current gradient signal is weak — which may correspond to important but currently dormant features.

**Why exploitation is still present.** The gradient $\mathbf{G}$ determines the singular *vectors* $\mathbf{u}_i, \mathbf{v}_i$. These encode the *direction* of the gradient in the input/output feature spaces. Muon exploits this directional information fully while ignoring the magnitude information. Subsequent gradient computations (at the next step) naturally correct any overshooting from over-weighted minor directions.

**Empirical signature.** The Kimi team observes that "SVD entropy of Muon is higher than that of AdamW" across training checkpoints — confirming that Muon uses more of the available singular directions, consistent with the exploration interpretation.

![Muon optimizer computational flow: gradient to orthogonalized update (Paperplanet, 2025)](figures/muon-optimizer/paperplanet-muon-diagram.png)
*The end-to-end computational process of Muon: the gradient is smoothed via momentum, normalized to unit Frobenius norm, passed through Newton-Schulz iteration to recover the polar factor $\mathbf{U}\mathbf{V}^\top$, and scaled before application.*

> [!NOTE] Why does exploration help for MoE models?
> In mixture-of-experts architectures, the router weight matrices have strongly peaked singular value spectra: a few directions dominate expert routing. Adam reinforces these dominant directions, while Muon forces exploration of alternative routing patterns. This may explain why Moonlight (trained with Muon) shows particularly large gains on reasoning tasks like MATH, which may require activating diverse expert combinations.

---

## 7. Su Jianlin's Spectral Norm Perspective

📐 Jianlin Su's analysis (kexue.fm) provides an independent angle: characterizing *what norm Muon is steepest descent under* and contrasting it with AdamW.

**AdamW's implicit norm.** AdamW with element-wise second-moment normalization is approximately steepest descent under the *max-of-max norm* — which normalizes each parameter element individually. This is a very weak norm that treats all matrix entries symmetrically.

**Muon's implicit norm.** As derived in §3, Muon is steepest descent under the spectral norm (or equivalently, the RMS-to-RMS operator norm). The spectral norm is *the natural norm for linear operators acting on Euclidean space*, since it measures the maximum amplification factor applied to inputs.

**Why the spectral norm is the right choice.** Weight matrices in neural networks function as *linear operators*: they map activation vectors from one representation space to another. The spectral norm measures the worst-case distortion of this mapping. An update constrained to be small in spectral norm guarantees that the network's input-output behavior changes in a controlled way, regardless of the direction.

The Frobenius norm (used by SGD) treats all singular directions equally and may allow large changes in the network's behavior (by making a large change in a dominant singular direction). The spectral norm provides a tighter coupling between weight change and behavioral change.

> [!INFO] The nuclear norm dual
> The dual norm of the spectral norm is the *nuclear norm* $\|\mathbf{G}\|_* = \sum_i \sigma_i(\mathbf{G})$. The steepest descent direction under a norm constraint $\|\Delta\mathbf{W}\|_{\text{op}} \leq \eta$ is the subdifferential of the dual norm at $\mathbf{G}$, which is exactly $\mathbf{U}\mathbf{V}^\top$ (a subgradient of the nuclear norm). This is a standard result in convex analysis.

**Frobenius vs. spectral norm comparison:**

| Constraint | Dual norm | Steepest descent direction | Method |
|------------|-----------|---------------------------|--------|
| $\|\Delta\mathbf{W}\|_F \leq \eta$ | Frobenius | $-\mathbf{G}$ | SGD |
| $\|\Delta\mathbf{W}\|_{\text{op}} \leq \eta$ | Nuclear | $-\mathbf{U}\mathbf{V}^\top$ | Muon |
| Element-wise $\ell^\infty$ | $\ell^1$ | $-\operatorname{sign}(\mathbf{G})$ | Adam (approx.) |

---

## 8. Why Orthogonalization Helps in Practice

💡 Several independent empirical and theoretical observations explain why replacing the gradient with its polar factor improves training.

**High condition numbers of transformer gradients.** Gradient matrices in transformers tend to have *high condition numbers* — the ratio of largest to smallest singular value is very large. This means gradient descent allocates almost all of its update budget to a few dominant directions, making extremely slow progress in the remaining directions.

Orthogonalization effectively sets the condition number of the update to 1. Every singular direction receives equal treatment. This counters the "gradient concentration" pathology.

**Rare but important directions.** The minor singular directions of the gradient (those with small $\sigma_i$) correspond to weight perturbations that the loss is currently insensitive to — but they may be highly relevant for long-range optimization. By amplifying these directions equally with the dominant ones, Muon ensures they are not neglected.

**Watermark erasure as a diagnostic.** Bernstein's analysis notes that models trained with Muon show faster erasure of training data watermarks — meaning the weights move substantially from initialization. This indicates that Muon is genuinely exploring the weight space rather than staying near the NTK (neural tangent kernel) regime.

**SVD entropy.** The Kimi team measures that Muon-trained models consistently have higher SVD entropy (defined as $-\sum_i \bar\sigma_i \log \bar\sigma_i$ where $\bar\sigma_i = \sigma_i / \sum_j \sigma_j$ are normalized singular values) in their weight matrices. Higher entropy means more evenly distributed singular value spectra — the model is using more of its representational capacity.

![Empirical singular value distribution of attention weight matrices (Wen, 2025)](figures/muon-optimizer/hyperball-svd-dist.png)
*Singular value distributions of transformer attention weight matrices across training: weights remain close to full rank rather than collapsing to a low-rank structure, supporting the claim that orthogonalization encourages uniform use of representational capacity.*

> [!DANGER] Common misconception: Muon does not improve conditioning of the weights
> Muon orthogonalizes the *update*, not the *weights*. After one Muon step, $\mathbf{W}_t - \eta \mathbf{U}\mathbf{V}^\top$ is not in general a well-conditioned matrix. What Muon controls is the *update direction* — ensuring that the step taken in weight space is balanced across singular directions.

---

## 9. Distributed Implementation

🖥️ A significant engineering contribution of Liu et al. (2025) is a memory-optimal, communication-efficient distributed implementation for Megatron-LM.

The challenge is that Newton-Schulz requires the *full* gradient matrix $\mathbf{M}_t \in \mathbb{R}^{m \times n}$ for orthogonalization, but in distributed training each device holds only a *shard* of each parameter matrix. Naive approaches would either (a) replicate the full gradient on all devices (memory-prohibitive) or (b) apply NS to each shard independently (mathematically incorrect — the polar factor of a submatrix is not a submatrix of the polar factor).

**Algorithm (Distributed Muon):**

```mermaid
flowchart TD
    A["Gradient shards G_i<br/>on each DP rank"] --> B["AllGather across DP group<br/>assembles full G"]
    B --> C["Newton-Schulz on full G<br/>produces full O = UV^T"]
    C --> D["Each rank discards<br/>non-local rows/cols of O"]
    D --> E["Apply local shard<br/>of O to local W shard"]
```

1. **DP Gather:** Assemble partitioned gradient shards into the full gradient $\mathbf{G}$.
2. **Full NS:** Compute $\mathbf{O} = \operatorname{NS}(\mathbf{G})$ on each device (redundant but necessary for correctness).
3. **Selective discard:** Each device retains only the rows/columns of $\mathbf{O}$ corresponding to its local parameter shard.

**Communication cost.** The only extra communication is the AllGather of gradients within DP groups. The additional overhead is $[1, 1.25]\times$ that of AdamW, with practical overhead near the lower bound $1\times$ when multiple DP groups are used.

> [!NOTE] Memory comparison with AdamW
> AdamW requires storing two optimizer states (first and second moment) per parameter. Muon requires storing only the momentum buffer (one state per parameter). The orthogonalized update $\mathbf{O}$ is computed on-the-fly and not stored. **Muon therefore uses approximately half the optimizer state memory of AdamW.**

---

## 10. Practical Considerations and Hyperparameters

🔧 The following table summarizes recommended hyperparameters for MuonW (the scaled, weight-decayed Muon variant):

| Hyperparameter | Recommended Value | Notes |
|----------------|-------------------|-------|
| Momentum $\mu$ | 0.95 | Higher than typical 0.9 for SGD |
| NS iterations $N$ | 5 | $N = 10$ gives negligible improvement |
| NS coefficients $(a,b,c)$ | $(3.4445, -4.7750, 2.0315)$ | Optimized quintic |
| Update RMS scale | 0.2 | Matches AdamW's typical effective scale |
| Weight decay $\lambda$ | 0.1 | Critical for stability at scale |
| Learning rate | Reuse AdamW's | Transfers directly with the RMS scaling |

**Which parameters to apply Muon to:**

```mermaid
flowchart TD
    P["Parameter tensor"] --> D{"2D weight matrix?"}
    D -->|"yes"| E{"Embedding or<br/>classifier head?"}
    E -->|"no"| F["Apply MuonW"]
    E -->|"yes"| G["Apply AdamW"]
    D -->|"no"| H["Apply AdamW<br/>(biases, norms, 1D params)"]
```

**QKV handling.** For attention QKV projections stored as a single concatenated matrix $[\mathbf{W}_Q; \mathbf{W}_K; \mathbf{W}_V]$, apply NS to each of the three sub-matrices independently. Treating the full concatenation as a single matrix is incorrect because Q, K, V operate in different semantic spaces.

**SFT compatibility.** *Models pretrained with Muon should also be fine-tuned with Muon*. The Kimi team observes that switching from Muon pretraining to AdamW fine-tuning yields no significant advantage over AdamW-pretrained models. The optimizer mismatch degrades the Muon pretraining benefits.

> [!TIP] Implementation tip: NS normalization
> Always normalize $\mathbf{M}_t$ to unit Frobenius norm before running the NS iteration. Without normalization, singular values outside $[0, 1]$ cause the polynomial to diverge. The normalization factor is discarded after NS (the output $\mathbf{O}$ has Frobenius norm $\approx \sqrt{\min(m,n)}$, which is then corrected by the $0.2\sqrt{\max(m,n)}$ scaling).

---

## 11. Experimental Results

📊 The Kimi team validates Muon on two axes: scaling law experiments (controlled FLOP comparisons) and the Moonlight production model.

**Scaling law comparison.** Under compute-optimal training (following Chinchilla scaling), Muon achieves:

$$\mathcal{L}_{\text{Muon}}(C) = 2.506 \times C^{-0.052}$$
$$\mathcal{L}_{\text{AdamW}}(C) = 2.608 \times C^{-0.054}$$

**At any fixed compute budget $C$, Muon requires approximately 52% of AdamW's FLOPs to reach the same loss.** This is the headline 2x efficiency result.

![NanoGPT speedrun: optimizer comparison by sample efficiency (Keller Jordan)](figures/muon-optimizer/kellerjordan-nanogpt-sample-efficiency.png)
*Sample efficiency comparison across optimizers on the NanoGPT speedrun benchmark: Muon reaches target validation loss with significantly fewer training tokens than AdamW, SGD, and other baselines.*

![NanoGPT speedrun: optimizer comparison by wallclock time (Keller Jordan)](figures/muon-optimizer/kellerjordan-nanogpt-wallclock.png)
*Wallclock training time comparison: despite the overhead of Newton-Schulz iterations (~0.7% extra FLOPs at NanoGPT scale), Muon reaches target loss faster in wall time due to its superior per-token progress.*

![Muon vs. AdamW at 1.5B parameter scale (Keller Jordan)](figures/muon-optimizer/kellerjordan-muon-vs-adamw-1b5.jpeg)
*Scaling validation at 1.5B parameters: Muon maintains its loss advantage over AdamW at larger scale, establishing that the efficiency gain is not limited to small models.*

**Moonlight results (3B active / 16B total MoE, 5.7T tokens):**

| Benchmark | DeepSeek-V2-Lite (AdamW) | Moonlight (Muon) |
|-----------|--------------------------|-------------------|
| MMLU | 58.3 | 70.0 |
| MATH | 17.1 | 45.3 |
| HumanEval | 29.9 | 48.1 |

The comparison is same architecture, same number of training tokens.

**Spectral analysis.** Muon-trained models exhibit higher SVD entropy in weight matrices throughout training, with the effect most pronounced in MoE router weights. This is consistent with the explore-exploit analysis: Muon encourages more diverse routing patterns, which may underlie the large MATH and reasoning gains.

> [!NOTE] Caveat on the 2x claim
> The 2x compute efficiency is measured under compute-*optimal* training, where both model size and token count are simultaneously optimized for a given FLOP budget. At fixed model size with increasing tokens (beyond the compute-optimal point), the advantage may differ. The paper does not report this regime.

---

## References

| Reference Name | Brief Summary | Link to Reference |
|----------------|---------------|-------------------|
| [Muon is Scalable for LLM Training (Liu et al., 2025)](https://arxiv.org/abs/2502.16982) | Primary paper; introduces MuonW with weight decay and per-parameter scaling, validates on Moonlight 3B/16B MoE, establishes 2x compute efficiency over AdamW | [https://arxiv.org/abs/2502.16982](https://arxiv.org/abs/2502.16982) |
| [Keller Jordan: Muon blog post](https://kellerjordan.github.io/posts/muon/) | Original Muon description; explains Newton-Schulz iteration, why orthogonalization helps, connection to Shampoo, FLOP overhead analysis | [https://kellerjordan.github.io/posts/muon/](https://kellerjordan.github.io/posts/muon/) |
| [Jeremy Bernstein: Deriving Muon](https://jeremybernste.in/writing/deriving-muon) | First-principles derivation of Muon as steepest descent under the RMS-to-RMS operator norm; proves UV^T is the exact dual solution; derives learning rate transfer | [https://jeremybernste.in/writing/deriving-muon](https://jeremybernste.in/writing/deriving-muon) |
| [Wen: Hyperball Optimizer (Part 1)](https://whenwen.github.io/wd_blog/public/hyperball-part-1.html) | Geometric interpretation of Muon as optimization on a hypersphere; normalization scheme for designed update-to-weight ratios; hyperparameter transfer results | [https://whenwen.github.io/wd_blog/public/hyperball-part-1.html](https://whenwen.github.io/wd_blog/public/hyperball-part-1.html) |
| [Paperplanet: Muon — An Explore-Exploit Perspective](https://paperplanet.github.io/posts/muon-a-explore-exploit-perspective/) | Interprets singular value normalization as exploration of gradient directions; explains SVD entropy observations; practical notes on QKV handling | [https://paperplanet.github.io/posts/muon-a-explore-exploit-perspective/](https://paperplanet.github.io/posts/muon-a-explore-exploit-perspective/) |
| [Su Jianlin: Muon Analysis (kexue.fm/10592)](https://kexue.fm/archives/10592) | Characterizes Muon as steepest descent under spectral norm; contrasts with AdamW's max-of-max norm; nuclear norm duality perspective | [https://kexue.fm/archives/10592](https://kexue.fm/archives/10592) |
| [Su Jianlin: Muon Analysis II (kexue.fm/10739)](https://kexue.fm/archives/10739) | Extended analysis and connections to other optimizers from Su Jianlin | [https://kexue.fm/archives/10739](https://kexue.fm/archives/10739) |
| [Shampoo: Preconditioned Stochastic Tensor Optimization (Gupta et al., 2018)](https://arxiv.org/abs/1802.09568) | Introduces Kronecker-factored gradient preconditioning; Muon is a limiting case of Shampoo with single-step gradient memory | [https://arxiv.org/abs/1802.09568](https://arxiv.org/abs/1802.09568) |
| [Scalable Second Order Optimization for Deep Learning (Anil et al., 2020)](https://arxiv.org/abs/2002.09018) | Practical Shampoo implementation at scale; comparison point for Muon's computational efficiency | [https://arxiv.org/abs/2002.09019](https://arxiv.org/abs/2002.09019) |
