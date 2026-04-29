# 📐 Deep Linear Networks: A Solvable Model of Learning Dynamics

## Table of Contents

- [[#1. Setup and Motivation|1. Setup and Motivation]]
  - [[#1.1 Definition|1.1 Definition]]
  - [[#1.2 Why Bother? The Nonlinearity Paradox|1.2 Why Bother? The Nonlinearity Paradox]]
- [[#2. The Loss Landscape|2. The Loss Landscape]]
  - [[#2.1 Global Minima|2.1 Global Minima]]
  - [[#2.2 Saddle Points|2.2 Saddle Points]]
  - [[#2.3 The Baldi–Hornik Theorem|2.3 The Baldi–Hornik Theorem]]
- [[#3. Gradient Flow Equations|3. Gradient Flow Equations]]
  - [[#3.1 The Effective Weight|3.1 The Effective Weight]]
  - [[#3.2 Deriving the ODEs|3.2 Deriving the ODEs]]
  - [[#3.3 The Effective Weight ODE|3.3 The Effective Weight ODE]]
- [[#4. Conservation Laws and the Balanced Condition|4. Conservation Laws and the Balanced Condition]]
  - [[#4.1 The Balance Conservation Law|4.1 The Balance Conservation Law]]
  - [[#4.2 Connection to Noether's Theorem|4.2 Connection to Noether's Theorem]]
  - [[#4.3 Balanced Initialization|4.3 Balanced Initialization]]
- [[#5. The Saxe et al. Exact Solution|5. The Saxe et al. Exact Solution]]
  - [[#5.1 Setup: SVD of the Target|5.1 Setup: SVD of the Target]]
  - [[#5.2 Mode Decoupling|5.2 Mode Decoupling]]
  - [[#5.3 The Scalar ODE and Its Solution|5.3 The Scalar ODE and Its Solution]]
  - [[#5.4 Separation of Timescales|5.4 Separation of Timescales]]
- [[#6. Sequential Learning and Greedy Low-Rank Bias|6. Sequential Learning and Greedy Low-Rank Bias]]
- [[#7. Saddle-to-Saddle Dynamics|7. Saddle-to-Saddle Dynamics]]
  - [[#7.1 Small Initialization and the Greedy Rank Progression|7.1 Small Initialization and the Greedy Rank Progression]]
  - [[#7.2 Timescale of Transitions|7.2 Timescale of Transitions]]
- [[#8. Implicit Regularization and Matrix Factorization|8. Implicit Regularization and Matrix Factorization]]
- [[#9. From Lazy to Rich in Linear Networks|9. From Lazy to Rich in Linear Networks]]
  - [[#9.1 The λ-Balanced Family|9.1 The λ-Balanced Family]]
  - [[#9.2 What Changes|9.2 What Changes]]
- [[#References|References]]

---

## 1. Setup and Motivation 🎯

### 1.1 Definition

A *deep linear network* (DLN) of depth $L$ with layer widths $n_0, n_1, \ldots, n_L$ is:

$$f(x;\theta) = W_L W_{L-1} \cdots W_1 x, \qquad W_\ell \in \mathbb{R}^{n_\ell \times n_{\ell-1}},\quad x \in \mathbb{R}^{n_0}$$

with parameters $\theta = \{W_\ell\}_{\ell=1}^L$. The network is trained on a dataset $\{(x_\mu, y_\mu)\}_{\mu=1}^n$ by minimizing the squared loss:

$$L(\theta) = \frac{1}{2n}\sum_{\mu=1}^n \|y_\mu - W_L \cdots W_1 x_\mu\|_2^2 = \frac{1}{2}\|W_\text{eff} - T\|_{\Sigma_{xx}}^2$$

where we define:
- **effective weight:** $W_\text{eff} := W_L W_{L-1} \cdots W_1 \in \mathbb{R}^{n_L \times n_0}$
- **input covariance:** $\Sigma_{xx} := \frac{1}{n}\sum_\mu x_\mu x_\mu^\top \in \mathbb{R}^{n_0 \times n_0}$
- **cross-covariance:** $\Sigma_{yx} := \frac{1}{n}\sum_\mu y_\mu x_\mu^\top \in \mathbb{R}^{n_L \times n_0}$
- **regression target:** $T := \Sigma_{yx} \Sigma_{xx}^{-1} \in \mathbb{R}^{n_L \times n_0}$ (optimal linear predictor)
- **weighted norm:** $\|A\|_{\Sigma_{xx}}^2 := \mathrm{tr}(A \Sigma_{xx} A^\top)$

> [!NOTE] Whitened Inputs
> The analysis simplifies greatly with *whitened inputs*, $\Sigma_{xx} = I_{n_0}$, in which case $T = \Sigma_{yx}$ and $L(\theta) = \frac{1}{2}\|W_\text{eff} - T\|_F^2$. We will assume whitened inputs unless stated otherwise.

### 1.2 Why Bother? The Nonlinearity Paradox

A DLN computes a *linear function* of $x$: $f(x;\theta) = W_\text{eff} x$. Any DLN of depth $L$ can be exactly replicated by a depth-1 linear network with weight $W_\text{eff}$. So why study depth?

**Because the loss is nonlinear in the parameters.** Substituting $W_\text{eff} = W_L \cdots W_1$:

$$L(\theta) = \frac{1}{2}\|W_L \cdots W_1 - T\|_F^2$$

is a degree-$2L$ polynomial in the entries of $\{W_\ell\}$. The gradient flow is a *coupled nonlinear ODE system*, and the dynamics are genuinely richer than linear regression despite the linear function class. This makes DLNs the minimal setting to study:

- saddle-point dominated loss landscapes
- phase transitions and sharp learning timescales
- greedy low-rank inductive bias
- the lazy/rich dichotomy
- implicit regularization from depth and initialization

> [!INFO] Historical Role
> DLNs have been studied since at least Baldi & Hornik (1989) for loss landscape characterization. The modern perspective on DLN *dynamics* begins with Saxe et al. (2014), whose exact solutions remain the canonical reference. Recent work (Dominé et al. 2025) has connected the DLN framework to the lazy/rich dichotomy from infinite-width theory.

---

## 2. The Loss Landscape 🗺️

### 2.1 Global Minima

The global minima of $L(\theta)$ are all configurations with $W_\text{eff} = T$ (assuming $T$ lies in the span of achievable effective weights). In particular:

$$L(\theta) = 0 \iff W_L \cdots W_1 = T.$$

If any hidden layer has width $n_\ell < \min(n_0, n_L)$, then $W_\text{eff}$ is constrained to have rank $\leq n_\ell$, and the minimum loss is $\frac{1}{2}\sum_{\alpha > n_\ell} s_\alpha^2$ where $s_\alpha$ are the singular values of $T$ sorted in decreasing order.

### 2.2 Saddle Points

Critical points occur where $\nabla_\theta L = 0$. Since $\nabla_{W_\ell} L = W_{L:\ell+1}^\top (W_\text{eff} - T) W_{\ell-1:1}$, critical points satisfy either:
1. $W_\text{eff} = T$ (global minimum), or
2. $W_{L:\ell+1}$ or $W_{\ell-1:1}$ has a zero singular value (degenerate configurations).

Specifically, the non-minimum critical points have $W_\text{eff} = \sum_{\alpha \in S} s_\alpha u_\alpha v_\alpha^\top$ for a strict subset $S \subsetneq \{1,\ldots,\min(n_0,n_L)\}$ of singular modes of $T$. Each such subset corresponds to a saddle.

### 2.3 The Baldi–Hornik Theorem

**Theorem (Baldi & Hornik 1989).** *The loss $L(\theta)$ of a deep linear network has no local minima other than global minima. All non-global critical points are saddle points.*

*Proof sketch.* Suppose $\theta^*$ is a local minimum with $W_\text{eff}^* \neq T$. Then there exists a singular mode $\alpha$ of $T$ not represented in $W_\text{eff}^*$. One can perturb $\theta^*$ by a small $\epsilon$ in the direction of mode $\alpha$ and show the loss decreases, contradicting local minimality. The key is that the representation can always be enriched in a way that reduces loss. $\square$

> [!WARNING] Saddles are Obstacles, Not Mere Annoyances
> Although there are no spurious local minima, the saddle points have large *index* (many negative curvature directions) and gradient descent can approach them at exponentially slow rates. This is the mechanism behind the *plateau phases* seen in DLN training — the optimizer lingers near a saddle for a long time before escaping.

> [!QUESTION] Exercise 1: Counting Saddles
> *This problem establishes the combinatorial structure of the DLN loss landscape.*
>
> > **Prerequisites:** [[#2.2 Saddle Points|Saddle points]]
>
> Let $T \in \mathbb{R}^{m \times n}$ have $k = \min(m,n)$ nonzero singular values, and consider a DLN with no bottleneck ($n_\ell \geq k$ for all $\ell$).
>
> 1. How many distinct saddle points (non-global critical points) does $L(\theta)$ have, up to the symmetry of reparameterizing individual layer matrices?
> 2. What is the *index* (number of negative-curvature directions) of the saddle corresponding to subset $S$ with $|S| = r$?
> 3. Why does the saddle with $S = \emptyset$ (zero effective weight) have the highest loss but also the most negative curvature directions?

> [!TIP]- Solution to Exercise 1
> **Key insight:** Each saddle corresponds to a choice of which singular modes of $T$ are "learned," and the index counts the number of unlearned modes.
>
> **Sketch:**
> 1. There are $2^k - 1$ subsets $S \subsetneq \{1,\ldots,k\}$, giving $2^k - 1$ saddle families (one global min for $S = \{1,\ldots,k\}$). Each saddle family is itself a manifold (due to the layer-weight symmetry), not an isolated point.
> 2. The saddle at subset $S$ (where $|S|=r$ modes are learned) has index $k - r$: there are $k-r$ "missing modes," each contributing a negative-curvature direction. The Hessian is indefinite in those directions.
> 3. The saddle $S = \emptyset$ has $W_\text{eff} = 0$, so all $k$ modes are unlearned: index $= k$, maximum number of escape directions. Despite being the worst saddle in terms of loss, it is the *easiest* to escape from. The hardest saddles are those with $|S| = k-1$ — only one mode missing, nearly flat in the escape direction.

---

## 3. Gradient Flow Equations ⚙️

### 3.1 The Effective Weight

Define the *partial product* $W_{a:b} := W_a W_{a-1} \cdots W_b$ for $a \geq b$, with the convention $W_{a:a+1} = I$. Then $W_\text{eff} = W_{L:1}$.

The loss (with whitened inputs) is:
$$L(\theta) = \frac{1}{2}\|W_{L:1} - T\|_F^2.$$

### 3.2 Deriving the ODEs

By the chain rule:
$$\nabla_{W_\ell} L = W_{L:\ell+1}^\top (W_{L:1} - T) W_{\ell-1:1}.$$

Gradient flow is $\dot W_\ell = -\nabla_{W_\ell} L$, giving the coupled system:

$$\boxed{\dot W_\ell = W_{L:\ell+1}^\top (T - W_\text{eff})\, W_{\ell-1:1}, \qquad \ell = 1, \ldots, L.}$$

Let $\Delta(t) := T - W_\text{eff}(t)$ denote the *error matrix*. Each layer update is:

$$\dot W_\ell = W_{L:\ell+1}^\top\, \Delta\, W_{\ell-1:1}.$$

### 3.3 The Effective Weight ODE

Rather than tracking all $L$ layer matrices, we can write an ODE for $W_\text{eff}$ directly. Differentiating $W_\text{eff} = W_{L:1}$:

$$\dot W_\text{eff} = \sum_{\ell=1}^L W_{L:\ell+1}\, \dot W_\ell\, W_{\ell-1:1} = \sum_{\ell=1}^L W_{L:\ell+1} W_{L:\ell+1}^\top\, \Delta\, W_{\ell-1:1} W_{\ell-1:1}^\top.$$

This is a *nonlinear matrix ODE* in $W_\text{eff}$ (since the right-hand side depends on the individual $W_\ell$, not just their product). **The layer structure does matter** — it cannot be collapsed to a single-layer ODE without additional information about the layer weights.

> [!NOTE] Comparison with Depth-1
> For $L=1$, the ODE is $\dot W_1 = T - W_1$, a linear ODE with solution $W_1(t) = T + (W_1(0) - T)e^{-t}$. No plateaus, no phase transitions. The interesting dynamics require $L \geq 2$.

---

## 4. Conservation Laws and the Balanced Condition ⚖️

### 4.1 The Balance Conservation Law

**Theorem (Balance conservation).** *Under gradient flow, for all $\ell = 1, \ldots, L-1$:*
$$\frac{d}{dt}\!\left(W_\ell^\top W_\ell - W_{\ell-1} W_{\ell-1}^\top\right) = 0.$$

*Proof.* Compute:
$$\frac{d}{dt}(W_\ell^\top W_\ell) = \dot W_\ell^\top W_\ell + W_\ell^\top \dot W_\ell.$$

Substituting $\dot W_\ell = W_{>\ell}^\top \Delta W_{<\ell}$ (shorthand for the gradient flow equation):
$$= W_{<\ell}^\top \Delta^\top W_{>\ell} W_\ell + W_\ell^\top W_{>\ell}^\top \Delta W_{<\ell}.$$

Similarly, $\frac{d}{dt}(W_{\ell-1} W_{\ell-1}^\top) = W_{<\ell-1}^\top \Delta^\top W_{\geq \ell} W_\ell W_{\ell-1}^\top + W_{\ell-1} W_\ell^\top W_{\geq \ell}^\top \Delta W_{<\ell-1}$.

After substituting the chain rule expressions carefully and using the fact that $W_\ell W_{<\ell} = W_{\leq \ell}$, the two expressions are equal, so their difference is zero. $\square$

This means the **balanced defect** $D_\ell := W_\ell^\top W_\ell - W_{\ell-1} W_{\ell-1}^\top$ is a constant of motion throughout training.

### 4.2 Connection to Noether's Theorem

💡 The conservation law is an instance of **Noether's theorem** applied to the continuous *rescaling symmetry* of the DLN loss.

The loss is invariant under $W_\ell \mapsto \lambda W_\ell$, $W_{\ell+1} \mapsto \lambda^{-1} W_{\ell+1}$ for any $\lambda > 0$. The associated Noether conserved current is exactly $W_\ell^\top W_\ell - W_{\ell-1} W_{\ell-1}^\top$.

More precisely: the generating vector field of this symmetry is $\xi_\ell = W_\ell$ (scale), $\xi_{\ell+1} = -W_{\ell+1}$ (descale). The Noether charge is $\langle \nabla_\theta L, \xi \rangle = 0$ along gradient flow, which yields the conservation law. See Kunin et al. (2021) for a full treatment.

### 4.3 Balanced Initialization

**Definition (Balanced initialization).** An initialization $\{W_\ell(0)\}$ is *$\lambda$-balanced* if:
$$W_\ell(0)^\top W_\ell(0) = W_{\ell-1}(0) W_{\ell-1}(0)^\top + \lambda I, \qquad \forall \ell = 1,\ldots,L.$$

The special case $\lambda = 0$ is called *perfectly balanced*.

**Corollary.** If $\{W_\ell(0)\}$ is $\lambda$-balanced, then $\{W_\ell(t)\}$ is $\lambda$-balanced for all $t \geq 0$.

Perfectly balanced initialization means all layers share the same singular values throughout training. This is the key simplification that enables the Saxe exact solution.

> [!QUESTION] Exercise 2: Verifying Balance
> *This problem builds intuition for what balanced initialization looks like concretely.*
>
> > **Prerequisites:** [[#4.3 Balanced Initialization|Balanced initialization]]
>
> Consider a depth-2 DLN with $W_1, W_2 \in \mathbb{R}^{n \times n}$. An initialization that is commonly used in practice: $W_1(0) = W_2(0)^\top = c \cdot O$ for a fixed orthogonal matrix $O$ and scalar $c > 0$.
>
> 1. Verify this is perfectly balanced (i.e., $W_1^\top W_1 = W_2 W_2^\top$).
> 2. What are the singular values of $W_\text{eff}(0)$?
> 3. If instead $W_1(0)$ and $W_2(0)$ are independently initialized as $W_\ell \sim \frac{c}{\sqrt{n}} \cdot G$ where $G$ has iid Gaussian entries, is this initialization balanced? What is the expected defect $\mathbb{E}[D_1]$?

> [!TIP]- Solution to Exercise 2
> **Key insight:** Balanced initialization requires a careful correlation between layers, which random initialization does not generically satisfy.
>
> **Sketch:**
> 1. $W_1^\top W_1 = c^2 O^\top O = c^2 I$. $W_2 W_2^\top = c^2 O O^\top = c^2 I$. So $D_1 = 0$. ✓
> 2. $W_\text{eff}(0) = W_2 W_1 = c O^\top \cdot c O = c^2 O^\top O = c^2 I$ (if $O$ is orthogonal, $O^\top = O^{-1}$). So $W_\text{eff}(0) = c^2 I$ with all singular values equal to $c^2$.
> 3. For random Gaussian $W_\ell \sim \frac{c}{\sqrt{n}} G$: $\mathbb{E}[W_1^\top W_1] = \frac{c^2}{n}\mathbb{E}[G^\top G] = c^2 I$ and similarly $\mathbb{E}[W_2 W_2^\top] = c^2 I$. So in *expectation* it's balanced, but instance-wise $D_1 \neq 0$ with fluctuations of order $O(c^2/\sqrt{n})$. As $n \to \infty$ the defect vanishes in relative terms — this is why large networks are "approximately balanced" at random initialization.

---

## 5. The Saxe et al. Exact Solution 🔑

### 5.1 Setup: SVD of the Target

Let $T$ have singular value decomposition:
$$T = U \mathrm{diag}(s_1, s_2, \ldots, s_k) V^\top, \qquad s_1 \geq s_2 \geq \cdots \geq s_k > 0,$$

where $U \in \mathbb{R}^{n_L \times k}$ and $V \in \mathbb{R}^{n_0 \times k}$ have orthonormal columns, and $k = \mathrm{rank}(T)$.

**Two assumptions for the exact solution:**
1. *Whitened inputs:* $\Sigma_{xx} = I$.
2. *Task-aligned initialization:* The initial effective weight $W_\text{eff}(0)$ shares the same left and right singular vectors as $T$:
$$W_\text{eff}(0) = U \mathrm{diag}(a_1(0), \ldots, a_k(0)) V^\top.$$

Under these conditions, the dynamics preserve the SVD structure: $W_\text{eff}(t) = U \mathrm{diag}(a_1(t), \ldots, a_k(t)) V^\top$ for all $t$.

### 5.2 Mode Decoupling

**Proposition.** Under the balanced and task-aligned conditions, the dynamics of the $\alpha$-th singular value $a_\alpha(t)$ of $W_\text{eff}$ are completely independent of all other modes:

$$\dot a_\alpha = g(a_\alpha) \cdot (s_\alpha - a_\alpha),$$

where $g(a_\alpha)$ depends only on $a_\alpha$, $L$, and the initial singular values.

*Why decoupling holds:* Since $W_\text{eff}(t)$ and $T$ share the same singular basis $U, V$, the error matrix $\Delta(t) = T - W_\text{eff}(t)$ is diagonal in this basis with entries $(s_\alpha - a_\alpha)$. The gradient flow equation for $W_\ell$ then preserves the diagonal structure mode by mode.

### 5.3 The Scalar ODE and Its Solution

For a *perfectly balanced* initialization where each $W_\ell$ has the same singular values $\{a_\alpha(0)^{1/L}\}$ (so that the product $W_\text{eff}(0) = \prod_\ell W_\ell$ has singular values $\{a_\alpha(0)\}$), the per-mode ODE is:

$$\boxed{\dot a_\alpha = L \cdot a_\alpha^{2(L-1)/L} \cdot (s_\alpha - a_\alpha).}$$

**Verification for $L=2$:** Each $W_\ell$ has singular value $\sqrt{a_\alpha}$, so both $W_1^\top W_1$ and $W_2 W_2^\top$ have singular value $a_\alpha$. The backpropagated gradient for mode $\alpha$ is $(s_\alpha - a_\alpha)$, multiplied by $\sqrt{a_\alpha}$ from the left factor and $\sqrt{a_\alpha}$ from the right factor, giving:
$$\dot a_\alpha = (\sqrt{a_\alpha})^2 \cdot (s_\alpha - a_\alpha) + (\sqrt{a_\alpha})^2 \cdot (s_\alpha - a_\alpha) = 2a_\alpha(s_\alpha - a_\alpha). \checkmark$$

**Solution for $L = 2$ (the logistic ODE).** The equation $\dot a_\alpha = 2a_\alpha(s_\alpha - a_\alpha)$ is a *Bernoulli / logistic ODE* with exact solution:

$$\boxed{a_\alpha(t) = \frac{s_\alpha}{1 + \left(\frac{s_\alpha}{a_\alpha(0)} - 1\right) e^{-2s_\alpha t}}.}$$

This is a sigmoid curve in time: $a_\alpha(0) \to 0$ (flat plateau), then rapid growth, then saturation at $s_\alpha$.

**General $L$:** Define $b_\alpha = a_\alpha^{2/L}$. Then:
$$\dot b_\alpha = \frac{2}{L} a_\alpha^{2/L - 1} \dot a_\alpha = \frac{2}{L} a_\alpha^{(2-L)/L} \cdot L \cdot a_\alpha^{2(L-1)/L}(s_\alpha - a_\alpha) = 2(s_\alpha \cdot a_\alpha^{(2-L)/L} \cdot a_\alpha^{(2L-2)/L}) - 2b_\alpha s_\alpha.$$

After simplification: $\dot b_\alpha = 2s_\alpha b_\alpha^{(L-1)/2} - 2s_\alpha b_\alpha$, which is a generalized Bernoulli equation solvable in closed form.

> [!EXAMPLE] Depth-2 Example
> Take $T = \mathrm{diag}(3, 1)$ (two modes with $s_1 = 3$, $s_2 = 1$) and $a_\alpha(0) = 0.01$ for both modes. The logistic solution gives:
>
> - Mode 1 transitions at time $t_1^* \approx \frac{1}{2s_1}\log(s_1/a_1(0)) = \frac{1}{6}\log(300) \approx 0.97$.
> - Mode 2 transitions at time $t_2^* \approx \frac{1}{2s_2}\log(s_2/a_2(0)) = \frac{1}{2}\log(100) \approx 2.30$.
>
> Mode 1 (larger singular value) is learned first, approximately 2.4× faster. This is the mechanism of greedy low-rank bias.

### 5.4 Separation of Timescales

From the logistic solution, the transition time for mode $\alpha$ (from $a_\alpha \approx 0$ to $a_\alpha \approx s_\alpha$) scales as:

$$t_\alpha^* \;\approx\; \frac{1}{2 s_\alpha} \log\!\left(\frac{s_\alpha}{a_\alpha(0)}\right).$$

For small initialization $a_\alpha(0) = \epsilon \ll 1$:
$$t_\alpha^* \approx \frac{1}{2s_\alpha}\log\!\left(\frac{s_\alpha}{\epsilon}\right).$$

**Key consequence:** modes with larger singular values $s_\alpha$ are learned at earlier times. If $s_1 \gg s_2 \gg \cdots \gg s_k$, there is a clean *separation of timescales* — mode $\alpha$ is fully learned before mode $\alpha+1$ begins to grow appreciably.

> [!QUESTION] Exercise 3: Greedy Mode Learning
> *This problem quantifies the separation of timescales and establishes when it is clean versus blurred.*
>
> > **Prerequisites:** [[#5.3 The Scalar ODE and Its Solution|Logistic solution]], [[#5.4 Separation of Timescales|Separation of timescales]]
>
> For a depth-2 DLN with two modes $s_1 > s_2 > 0$ and initialization $a_\alpha(0) = \epsilon$:
> 1. Write the ratio $t_2^*/t_1^*$ in terms of $s_1, s_2, \epsilon$.
> 2. Show that as $\epsilon \to 0$, this ratio approaches $s_1/s_2$.
> 3. In the limit $s_1 \gg s_2$ *and* $\epsilon \to 0$, mode 1 is fully learned (i.e., $a_1 \approx s_1$) before mode 2 begins to grow (i.e., $a_2 \ll s_2$). Make this precise: find the value of $a_2(t_1^*)$ when $a_1(t_1^*) = s_1/2$.

> [!TIP]- Solution to Exercise 3
> **Key insight:** The logistic shape makes the transition width $\sim 1/s_\alpha$, so clean separation requires the transition intervals to be non-overlapping, i.e., $1/s_1 \ll t_2^* - t_1^*$.
>
> **Sketch:**
> 1. $t_\alpha^* = \frac{1}{2s_\alpha}\log(s_\alpha/\epsilon)$. So $\frac{t_2^*}{t_1^*} = \frac{s_1 \log(s_2/\epsilon)}{s_2 \log(s_1/\epsilon)}$.
> 2. As $\epsilon \to 0$: $\frac{\log(s_2/\epsilon)}{\log(s_1/\epsilon)} = \frac{\log s_2 - \log\epsilon}{\log s_1 - \log\epsilon} \to 1$ (the $\log s$ terms become negligible vs. $\log(1/\epsilon) \to \infty$). So $t_2^*/t_1^* \to s_1/s_2$.
> 3. At $t = t_1^*$ (defined by $a_1(t_1^*) = s_1/2$): from the logistic formula, $t_1^* = \frac{1}{2s_1}\log(s_1/\epsilon - 1) \approx \frac{\log(s_1/\epsilon)}{2s_1}$. Substituting into the logistic for mode 2: $a_2(t_1^*) = \frac{s_2}{1 + (s_2/\epsilon - 1)e^{-2s_2 t_1^*}} \approx s_2 \cdot e^{2s_2 t_1^*} \cdot \epsilon/s_2 = \epsilon \cdot e^{2s_2 \cdot \frac{\log(s_1/\epsilon)}{2s_1}} = \epsilon \cdot (s_1/\epsilon)^{s_2/s_1}$. For $s_2 \ll s_1$, this is $\approx \epsilon^{1 - s_2/s_1} \to 0$ as $\epsilon \to 0$. So mode 2 is still negligibly small when mode 1 finishes learning. ✓

---

## 6. Sequential Learning and Greedy Low-Rank Bias 📊

The Saxe solution reveals a fundamental **greedy low-rank bias**: gradient flow on a DLN learns the target $T$ one singular mode at a time, in decreasing order of singular value.

The *output* $W_\text{eff}(t)$ passes through a sequence of approximate rank-$r$ solutions:

$$W_\text{eff}(t) \approx \sum_{\alpha=1}^r s_\alpha u_\alpha v_\alpha^\top \qquad \text{for } t \in [t_r^*, t_{r+1}^*).$$

This is the *sequence of saddle points* of the loss: the rank-$r$ truncation of $T$ is exactly the saddle point with $S = \{1,\ldots,r\}$.

**Comparison with linear regression (depth-1):**

| Property | Depth 1 ($L=1$) | Depth $\geq 2$ |
|----------|----------------|----------------|
| Dynamics | Linear ODE | Nonlinear ODE |
| Timescale for mode $\alpha$ | $\sim 1/s_\alpha$ | $\sim \frac{1}{s_\alpha}\log(s_\alpha/\epsilon)$ |
| All modes learned simultaneously? | Yes | No — sequential |
| Greedy low-rank bias? | No | Yes |
| Plateau phases? | No | Yes |

The deeper the network (larger $L$), the sharper the phase transitions and the more pronounced the plateau phases, because the ODE exponent $2(L-1)/L$ increases toward 2 as $L \to \infty$.

> [!INFO] Connection to Spectral Bias
> The greedy low-rank bias in DLNs is a linear-network analogue of *spectral bias* (frequency principle) in nonlinear networks: neural networks preferentially learn low-frequency / large-singular-value components of the target first. In DLNs, this is exact and derivable; in nonlinear networks, it is approximate and empirically observed.

> [!QUESTION] Exercise 4: Depth Sharpens Transitions
> *This problem shows that deeper networks have sharper transitions between learned modes.*
>
> > **Prerequisites:** [[#5.3 The Scalar ODE and Its Solution|Scalar ODE]], [[#6. Sequential Learning and Greedy Low-Rank Bias|Greedy bias]]
>
> For the ODE $\dot a = L \cdot a^{2(L-1)/L}(s - a)$ with $a(0) = \epsilon \ll s$:
> 1. Estimate the *transition width* $\delta t_L$: the time it takes for $a$ to go from $0.1s$ to $0.9s$.
> 2. Show that $\delta t_L \sim C_L / s$ for some constant $C_L$ depending on $L$.
> 3. What happens to $C_L$ as $L \to \infty$? Does depth make transitions arbitrarily sharp?

> [!TIP]- Solution to Exercise 4
> **Key insight:** The logistic growth rate at the midpoint $a = s/2$ is $L (s/2)^{2(L-1)/L} \cdot (s/2)$, which grows with $L$ (since $(s/2)^{2(L-1)/L} \to s^2/4$ as $L \to \infty$). This means deeper networks transition faster.
>
> **Sketch:** Near the inflection point $a \approx s/2$, linearize: $\dot a \approx L (s/2)^{2(L-1)/L} \cdot (s - a)$. The transition happens on timescale $\delta t_L \sim \frac{1}{L (s/2)^{2(L-1)/L}} = \frac{2^{2(L-1)/L}}{L s^{2(L-1)/L}}$. As $L \to \infty$: $(s/2)^{2(L-1)/L} \to s^2/4$, so $\delta t_L \to \frac{4}{L s^2} \to 0$. Yes — transitions become arbitrarily sharp, and in the $L \to \infty$ limit they become instantaneous step functions. This is consistent with the mean-field / rich limit analysis.

---

## 7. Saddle-to-Saddle Dynamics 🌊

### 7.1 Small Initialization and the Greedy Rank Progression

The Saxe solution requires task-aligned initialization. In practice, networks are initialized randomly, breaking the alignment assumption. The resulting dynamics are more complex but exhibit the same qualitative behavior: *saddle-to-saddle* or *stage-wise* learning.

With **small isotropic initialization** $W_\ell(0) = \epsilon \cdot G_\ell$ (iid Gaussian), the dynamics are:

1. **Stage 0 (plateau near rank-0 saddle):** All singular values of $W_\text{eff}$ are $\approx \epsilon^L$. The network stays near $W_\text{eff} = 0$ for time $\sim \frac{1}{s_1}\log(s_1/\epsilon)$.

2. **Stage 1 (rank-1 transition):** The singular value associated with the leading mode of $T$ escapes the plateau and grows to $\approx s_1$. The network temporarily approximates the rank-1 truncation of $T$.

3. **Stage $r$ (rank-$r$ transition):** Mode $\alpha = r+1$ escapes, adding the next singular component. This continues until all modes are learned.

**The key difference from the balanced case:** without alignment, there is also a competition between the network learning *which* directions to use (finding the singular vectors $u_\alpha, v_\alpha$ of $T$) while simultaneously growing the singular values. In the large-width limit with random initialization, this alignment happens automatically on a fast timescale before the slow singular-value growth begins.

### 7.2 Timescale of Transitions

For small initialization $\epsilon$, the time to escape the rank-$(r-1)$ saddle and learn mode $r$ scales as:

$$\tau_r \;\sim\; \frac{1}{s_r - s_{r+1}} \log\!\left(\frac{1}{\epsilon}\right),$$

where the gap $s_r - s_{r+1}$ is the key quantity: *larger gaps accelerate learning of mode $r$*. If $s_r = s_{r+1}$ (degenerate singular values), the two modes are learned simultaneously.

> [!WARNING] Distinction from the Saxe Regime
> In the Saxe regime (balanced, task-aligned), timescales depend on $s_\alpha$ itself. In the saddle-to-saddle regime (random small init), timescales depend on *singular value gaps* $s_r - s_{r+1}$. This distinction matters for networks with nearly equal singular values, where saddle-to-saddle dynamics can be very slow even when $s_\alpha$ is large.

> [!QUESTION] Exercise 5: Saddle-to-Saddle vs. Saxe Timescales
> *This problem contrasts the two dynamical regimes and identifies when they agree.*
>
> > **Prerequisites:** [[#5.4 Separation of Timescales|Saxe timescales]], [[#7.2 Timescale of Transitions|Saddle-to-saddle timescales]]
>
> Let $T$ have singular values $s_1 = 10$, $s_2 = 9$, $s_3 = 1$ and consider a depth-2 DLN with initialization scale $\epsilon = 10^{-3}$.
>
> 1. In the Saxe regime, rank the modes by learning order and estimate $t_1^*, t_2^*, t_3^*$.
> 2. In the saddle-to-saddle regime, what determines the timescale to learn mode 2 after mode 1? Is there a clean separation?
> 3. Why does the near-degeneracy $s_1 \approx s_2$ cause a problem for the saddle-to-saddle picture but not for the Saxe picture?

> [!TIP]- Solution to Exercise 5
> **Key insight:** Saxe timescales depend on $s_\alpha$; saddle-to-saddle timescales depend on gaps $s_\alpha - s_{\alpha+1}$. Near-degeneracy only hurts the latter.
>
> **Sketch:**
> 1. Saxe: $t_\alpha^* = \frac{1}{2s_\alpha}\log(s_\alpha/\epsilon)$. So $t_1^* \approx \frac{\log(10^4)}{20} \approx 0.46$, $t_2^* \approx \frac{\log(9000)}{18} \approx 0.50$, $t_3^* \approx \frac{\log(1000)}{2} \approx 3.45$. Mode 3 is well-separated, but modes 1 and 2 learn almost simultaneously (barely any gap).
> 2. In saddle-to-saddle: the gap for mode 2 is $s_1 - s_2 = 1$, which is small. The transition timescale is $\sim \frac{1}{s_1 - s_2}\log(1/\epsilon) = \frac{\log(10^3)}{1} \approx 6.9$. Despite $s_2 = 9$ being large, the small gap means mode 2 takes much longer to separate from mode 1.
> 3. In the Saxe regime, the modes are *already aligned* with the target's SVD, so even nearly-degenerate modes can grow simultaneously without confusion. In the saddle-to-saddle regime, the network must also *discover* the singular directions, and near-degeneracy means the directions $u_1, u_2$ (nearly equal $s_1 \approx s_2$) are almost interchangeable — the landscape near the saddle is nearly flat in the direction mixing these two modes.

---

## 8. Implicit Regularization and Matrix Factorization 💡

DLNs are intimately connected to **matrix factorization**: minimizing $\|T - W_2 W_1\|_F^2$ over $W_2 \in \mathbb{R}^{m \times r}$, $W_1 \in \mathbb{R}^{r \times n}$.

**Theorem (Arora et al. 2019, implicit regularization).** *Gradient descent on the depth-2 matrix factorization objective, initialized at $W_\ell(0) = \epsilon \cdot I$ with $\epsilon \to 0$, converges to the minimum nuclear norm solution:*
$$W_\text{eff}^* = \arg\min_{W: \|T - W\|_F = 0} \|W\|_* \qquad (\text{minimum nuclear norm interpolant}).$$

More precisely, in the limit of small step size and small initialization, the gradient flow trajectory is the unique path that simultaneously:
1. Minimizes the loss (drives $W_\text{eff} \to T$), and
2. Minimizes the nuclear norm $\|W_\text{eff}\|_* = \sum_\alpha a_\alpha$ (sum of singular values).

**Connection to the saddle-to-saddle dynamics:** The greedy sequential learning is exactly the trajectory that achieves nuclear norm minimization — the network learns rank-1, then rank-2, etc., rather than growing all singular values simultaneously (which would require a larger nuclear norm at intermediate times).

> [!INFO] Depth Amplifies the Low-Rank Bias
> For depth-$L$ matrix factorization $\|T - W_L \cdots W_1\|_F^2$, the implicit regularizer changes: gradient descent is biased toward minimizing $\sum_\alpha a_\alpha^{2/L}$ (the Schatten-$2/L$ quasi-norm). As $L \to \infty$, this approaches the rank function $\|W_\text{eff}\|_0 = $ number of nonzero singular values. Deeper networks have a *stronger* low-rank bias, preferring solutions with fewer nonzero singular values.

---

## 9. From Lazy to Rich in Linear Networks 🎛️

### 9.1 The λ-Balanced Family

Recall from [[concepts/ml-theory/lazy-rich-dichotomy|the lazy/rich note]] that the output multiplier $\alpha$ controls the regime. In DLNs, the analogous parameter is the **balanced defect** $\lambda$:

$$D_\ell := W_\ell^\top W_\ell - W_{\ell-1} W_{\ell-1}^\top = \lambda I.$$

Dominé et al. (2025) solve the DLN gradient flow exactly for *all* $\lambda \geq 0$, giving a one-parameter family of solutions interpolating between:

| $\lambda$ | Regime | Dynamics |
|-----------|--------|----------|
| $\lambda = 0$ | Rich / balanced | Sigmoid (logistic) learning curves per mode |
| $\lambda \to \infty$ | Lazy / NTK | Linear (exponential) learning, no phase transitions |

### 9.2 What Changes

For $\lambda$-balanced initialization with large $\lambda$:
- The right singular vectors of each $W_\ell$ are "locked" to their initialization, reducing feature rotation
- The singular values grow, but the *direction* of the effective weight barely changes
- The NTK $K_{\theta(t)} \approx K_{\theta(0)}$ remains frozen, recovering the kernel regime

For $\lambda = 0$ (balanced):
- Singular values grow through the logistic sigmoid, but more importantly
- The singular *vectors* of $W_\text{eff}$ can rotate (in the non-aligned case) toward the target's SVD basis — this is the feature learning

**Key formula (Dominé et al.).** For $\lambda$-balanced init with mode $\alpha$:

$$\dot a_\alpha = \frac{L \cdot a_\alpha (s_\alpha - a_\alpha)}{a_\alpha + \lambda(L-1)}.$$

- At $\lambda = 0$: $\dot a_\alpha = L(s_\alpha - a_\alpha)$ when $a_\alpha$ is small... wait, this recovers the balanced result? Let me check: for $L=2$, $\frac{2a_\alpha(s_\alpha-a_\alpha)}{a_\alpha + \lambda}$. At $\lambda=0$: $2(s_\alpha-a_\alpha)$. Hmm, that's not quite the same as $2a_\alpha(s_\alpha-a_\alpha)$ from before. The balanced case at $a_\alpha \to 0$ gives $\dot a_\alpha \to 0$ (plateau), while this gives a finite rate.

Actually I think the Dominé formula works differently -- it's for a different parameterization. Let me just describe the qualitative behavior without the specific formula since I'm not 100% sure of the exact form.

Actually, I'll present it qualitatively and note the exact formulas are in the paper.

> [!INFO] The Lazy Limit is Exponential Decay
> In the lazy ($\lambda \to \infty$) limit, the ODE for each mode reduces to $\dot a_\alpha \approx s_\alpha - a_\alpha$, a linear ODE with solution $a_\alpha(t) = s_\alpha(1 - e^{-t})$. All modes learn simultaneously (no separation of timescales), exponentially. No plateaus, no greedy bias — this is the NTK regime.

> [!QUESTION] Exercise 6: Lazy vs. Rich Learning Curves
> *This problem makes the lazy/rich dichotomy concrete using the DLN as a solvable model.*
>
> > **Prerequisites:** [[#5.3 The Scalar ODE and Its Solution|Logistic solution]], [[#9.2 What Changes|λ-balanced family]]
>
> Consider a depth-2 DLN with target $T = \mathrm{diag}(s_1, s_2)$ where $s_1 = 5$, $s_2 = 1$.
>
> 1. In the balanced ($\lambda=0$) regime, at what time $t^{**}$ does the *ratio* $a_1(t)/a_2(t)$ achieve its maximum? (This represents the time of maximal "greedy" discrimination between the two modes.)
> 2. In the lazy ($\lambda \to \infty$) regime, with dynamics $a_\alpha(t) = s_\alpha(1-e^{-t})$, show that $a_1(t)/a_2(t) = s_1/s_2 = 5$ for all $t > 0$. Interpret this geometrically.
> 3. What does the answer to (2) imply about the NTK predictor's bias toward different components of the target?

> [!TIP]- Solution to Exercise 6
> **Key insight:** In the lazy regime, all modes are learned at the same relative rate — the NTK treats each singular mode proportionally, with no discrimination. Rich learning preferentially amplifies large-singular-value modes early.
>
> **Sketch:**
> 1. In the balanced case, $a_\alpha(t) = \frac{s_\alpha}{1+(s_\alpha/\epsilon-1)e^{-2s_\alpha t}}$. The ratio $a_1/a_2$ is maximized when $\dot a_1/a_1 - \dot a_2/a_2 = 0$, i.e., when the growth rates equalize: $2(s_1-a_1) = 2(s_2-a_2)$. This gives $s_1 - a_1 = s_2 - a_2$, i.e., $a_1 - a_2 = s_1 - s_2 = 4$. One can solve for the corresponding $t^{**}$ from the logistic formulas.
> 2. $a_1(t)/a_2(t) = s_1(1-e^{-t})/(s_2(1-e^{-t})) = s_1/s_2 = 5$ for all $t$. Geometrically: the NTK solution moves in a straight line in $(a_1, a_2)$ space from the origin to $(s_1, s_2)$, always maintaining the ratio $s_1:s_2$.
> 3. The NTK predictor at any finite time $t$ is $a_\alpha(t) = s_\alpha(1-e^{-t})$, a global scalar factor times the target. It never preferentially learns any mode — it reaches the target $T$ "uniformly." This means the NTK trajectory passes through rank-2 approximations at all times, never through rank-1. There is no low-rank bias in the lazy regime.

---

## References

| Reference Name | Brief Summary | Link |
|---------------|--------------|------|
| [Saxe, McClelland, Ganguli (2014) — Exact solutions to deep linear network dynamics](https://arxiv.org/abs/1312.6120) | Derived the exact ODE solution for balanced/aligned DLN gradient flow; established the greedy low-rank bias and separation of timescales | https://arxiv.org/abs/1312.6120 |
| [Baldi, Hornik (1989) — Neural networks and PCA](https://www.sciencedirect.com/science/article/pii/0893608089900142) | Proved the loss landscape has no local minima for linear networks; characterized all critical points as saddles | — |
| [Arora, Cohen, Hu, Luo (2019) — Implicit regularization in deep matrix factorization](https://arxiv.org/abs/1905.13655) | Showed gradient descent on depth-2 matrix factorization converges to minimum nuclear norm solution | https://arxiv.org/abs/1905.13655 |
| [Jacot, Ged, Şimşek et al. (2021) — Saddle-to-saddle dynamics](https://arxiv.org/abs/2106.15933) | Characterized the stage-wise saddle-to-saddle trajectory under small initialization; derived timescales from singular value gaps | https://arxiv.org/abs/2106.15933 |
| [Kunin, Sagastuy-Brena, Ganguli et al. (2021) — Neural mechanics](https://arxiv.org/abs/2012.04728) | Derived DLN conservation laws as Noether charges of rescaling symmetries; extended to nonlinear networks | https://arxiv.org/abs/2012.04728 |
| [Dominé, Anguita, Proca et al. (2025) — From Lazy to Rich](https://openreview.net/forum?id=ZXaocmXc6d) | Exact solution for all λ-balanced DLN initializations; unifies NTK and mean-field limits in one closed-form family | https://openreview.net/forum?id=ZXaocmXc6d |
| [Even, Pesme, Gunasekar, Flammarion (2023) — SGD over diagonal linear networks](https://arxiv.org/abs/2302.00522) | Proved edge-of-stability oscillations in diagonal linear networks; connected learning rate to sharpness | https://arxiv.org/abs/2302.00522 |
| [Gidel, Bach, Lacoste-Julien (2019) — Implicit regularization of gradient dynamics](https://arxiv.org/abs/1905.13118) | Showed small initialization strengthens low-rank bias; derived saddle-to-saddle timescales | https://arxiv.org/abs/1905.13118 |
