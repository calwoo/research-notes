# Mixture of Experts: Solutions

## Table of Contents

- [[#Mathematical Development|Mathematical Development]]
  - [[#Problem 1 Soft Gating Gradient and the Expert Weighting Update|Problem 1: Soft Gating Gradient and the Expert Weighting Update]]
  - [[#Problem 2 Top-k Gating as a Constrained Argmax|Problem 2: Top-k Gating as a Constrained Argmax]]
  - [[#Problem 3 Gradient Variance in Sparse vs Soft Gating|Problem 3: Gradient Variance in Sparse vs Soft Gating]]
  - [[#Problem 4 Load Balancing Loss Derivation|Problem 4: Load Balancing Loss Derivation]]
  - [[#Problem 5 Minimizers of the Importance and Auxiliary Losses|Problem 5: Minimizers of the Importance and Auxiliary Losses]]
  - [[#Problem 6 Expert Capacity and Token Drop Probability|Problem 6: Expert Capacity and Token Drop Probability]]
  - [[#Problem 7 Capacity Factor Sensitivity|Problem 7: Capacity Factor Sensitivity]]
  - [[#Problem 8 Birthday Problem Analogy for Token Overflow|Problem 8: Birthday Problem Analogy for Token Overflow]]
  - [[#Problem 9 Expert-Choice Routing as Bipartite Matching|Problem 9: Expert-Choice Routing as Bipartite Matching]]
  - [[#Problem 10 Expert-Choice Dual Formulation and LP Relaxation|Problem 10: Expert-Choice Dual Formulation and LP Relaxation]]
  - [[#Problem 11 MoE vs Ensemble vs Product of Experts|Problem 11: MoE vs Ensemble vs Product of Experts]]
  - [[#Problem 12 Top-1 Routing and Voronoi Specialization|Problem 12: Top-1 Routing and Voronoi Specialization]]
  - [[#Problem 13 Expert Collapse as a Fixed-Point Instability|Problem 13: Expert Collapse as a Fixed-Point Instability]]
  - [[#Problem 14 Router Z-Loss Gradient Analysis|Problem 14: Router Z-Loss Gradient Analysis]]
  - [[#Problem 15 Entropy Regularization vs Z-Loss|Problem 15: Entropy Regularization vs Z-Loss]]
  - [[#Problem 16 MoE Scaling Laws vs Dense Scaling Laws|Problem 16: MoE Scaling Laws vs Dense Scaling Laws]]
  - [[#Problem 17 Soft-Gating Fixed-Point Stability|Problem 17: Soft-Gating Fixed-Point Stability]]
  - [[#Problem 18 Effective Parameter Count and Routing Efficiency|Problem 18: Effective Parameter Count and Routing Efficiency]]
- [[#Algorithmic Applications|Algorithmic Applications]]
  - [[#Problem 19 Noisy Top-k Gating Forward Pass|Problem 19: Noisy Top-k Gating Forward Pass]]
  - [[#Problem 20 Expert Dispatch with Capacity Enforcement|Problem 20: Expert Dispatch with Capacity Enforcement]]
  - [[#Problem 21 Router Z-Loss Numerically Stable Implementation|Problem 21: Router Z-Loss Numerically Stable Implementation]]
  - [[#Problem 22 Expert-Choice Routing Forward Pass|Problem 22: Expert-Choice Routing Forward Pass]]
  - [[#Problem 23 Distributed Expert Parallelism Protocol|Problem 23: Distributed Expert Parallelism Protocol]]

---

## Mathematical Development

### Problem 1: Soft Gating Gradient and the Expert Weighting Update

**Key insight:** The softmax Jacobian $\partial g_i / \partial z_j = g_i(\delta_{ij} - g_j)$ causes the gradient of $\mathcal{L}$ with respect to the router to depend on the residual $f_i(x) - y$, so experts that match the current ensemble mean contribute zero gradient pressure on the router.

**Sketch:**

(a) By chain rule: $\frac{\partial \mathcal{L}}{\partial W_g} = \sum_k \frac{\partial \mathcal{L}}{\partial g_k} \frac{\partial g_k}{\partial W_g}$. The softmax Jacobian gives $\partial g_k / \partial (W_g)_{jk} = g_k(\delta_{kk} - g_k) x_j$, producing after assembly a rank-1 outer product:

$$\frac{\partial \mathcal{L}}{\partial W_g} = x \cdot v^\top \in \mathbb{R}^{d \times E}$$

where the $k$-th component of $v$ is:

$$v_k = \frac{\partial \mathcal{L}}{\partial y} \cdot g_k(x)\bigl(f_k(x) - y\bigr) \in \mathbb{R}$$

a scalar, since $\partial\mathcal{L}/\partial y$, $f_k(x)$, and $y$ are all in $\mathbb{R}^d$ and their inner product is a scalar.

(b) Stationary condition: $\sum_i g_i(x)(f_i(x) - y) = 0$ for all $x$. An expert with $f_i(x) = y$ contributes zero residual; its gradient pressure is exactly zero regardless of $g_i(x)$.

(c) Clamping unselected logits to $-\infty$ before softmax makes those entries constant with zero gradient. The gradient of the resulting modified objective with respect to selected logits is algebraically identical to the straight-through estimator — confirming the straight-through interpretation is exact.

---

### Problem 2: Top-k Gating as a Constrained Argmax

**Key insight:** The KeepTopK operator interpolates between two extremes: at $k = E$ no logits are masked so standard softmax is recovered, while at $k = 1$ the softmax over a single surviving finite logit is identically 1, recovering the argmax one-hot.

**Sketch:**

(a) For $k = E$: KeepTopK leaves all entries finite, so softmax is unchanged. For $k = 1$: only $\max_i h_i$ survives; $\operatorname{softmax}$ of a single finite value is 1.

(b) Where all logits are distinct, the top-$k$ set $\mathcal{T}(x)$ is locally constant as a function of $W_g$. So $\partial/\partial W_g$ passes only through the $k$ surviving softmax logits, and the Jacobian restricted to $\mathcal{T}(x)$ has the same $g_i(\delta_{ij} - g_j)$ form as the dense softmax Jacobian.

(c) Define $\widetilde{h}_i = h_i$ for $i \in \mathcal{T}$, $\widetilde{h}_i = -\infty$ for $i \notin \mathcal{T}$. The gradient $\partial \mathcal{L} / \partial h_i$ for $i \in \mathcal{T}$ under $\operatorname{softmax}(\widetilde{h})$ is identical to the straight-through estimator, confirming equivalence with a modified objective.

---

### Problem 3: Gradient Variance in Sparse vs Soft Gating

**Key insight:** Restricting gradient flow to $k < E$ experts increases variance by a factor of $E/k$ relative to soft gating because fewer terms contribute to the Monte Carlo estimator, reducing the variance-reduction from averaging.

**Sketch:**

(a) Under uniform weights $g_i = 1/E$, the gradient involves a sum over $E$ terms each contributing variance $\sigma^2/E^2$ (since each expert is weighted $1/E$). Total variance $= E \cdot \sigma^2/E^2 = \sigma^2/E$. (This treats $y \approx \mu$ as approximately constant, valid for large $E$ where the law of large numbers makes $y$ concentrate around $\mu$. For finite $E$, the exact variance has additional cross-terms from the dependence of $y$ on each $f_i$.)

(b) Under top-$k$ gating, only $k$ experts contribute with weights $\approx 1/k$ each. Total variance $\approx k \cdot \sigma^2/k^2 = \sigma^2/k$. Since $k < E$, this is higher than soft gating.

(c) The noise $\epsilon_i \sim \mathcal{N}(0, s_i^2)$ perturbs which $k$ experts are selected, adding a selection-variance term on top of the output variance. A large $s_i$ increases exploration for underused experts at the cost of noisier gradients; learning $s_i$ allows the model to tune this tradeoff per expert.

---

### Problem 4: Load Balancing Loss Derivation

**Key insight:** Both losses are zero exactly at uniform routing, and their gradients penalize the product of discrete load ($f_i$) and soft probability ($P_i$), with only the soft term providing a gradient, so the loss acts purely through the softmax probabilities.

**Sketch:**

(a) $\operatorname{CV}^2 = 0$ iff $\sigma(\operatorname{Importance}) = 0$ iff all $\operatorname{Importance}_i$ are equal. Non-negativity follows from $\sigma \geq 0$ and $\mu > 0$. It is unbounded because $\sigma$ can grow without bound if one expert captures all weight.

(b) Under uniform routing: $f_i = P_i = 1/E$, so $\sum_i f_i P_i = E \cdot (1/E)^2 = 1/E$ and $\mathcal{L}_{\text{aux}} = \alpha$. By Cauchy-Schwarz, $\sum_i f_i P_i \geq (\sum_i \sqrt{f_i P_i})^2 / E$, with equality iff $f_i/P_i$ is constant. Concentration of $f_i$ and $P_i$ on the same expert increases the inner product strictly above $1/E$.

(c) $\partial \mathcal{L}_{\text{aux}} / \partial g_i(x) = \alpha E \cdot f_i / T$. Large when $f_i$ is large (over-loaded expert): gradient pushes $g_i(x)$ down, reducing probability mass sent to that expert. Small when $f_i$ is small: under-loaded experts receive relatively more probability mass. This is the anti-collapse direction.

---

### Problem 5: Minimizers of the Importance and Auxiliary Losses

**Key insight:** Both losses share the same global minimizer (uniform routing), but $\mathcal{L}_{\text{imp}}$ is blind to the discrete argmax routing pattern, so it is possible to have $\mathcal{L}_{\text{aux}} = \alpha$ (minimized) while $\mathcal{L}_{\text{imp}} > 0$ if soft probabilities are concentrated within tokens even when balanced in aggregate.

**Sketch:**

(a) $\mathcal{L}_{\text{imp}} = 0$ iff $\sigma(\operatorname{Importance}) = 0$ iff $\sum_x g_i(x) = Tk/E$ for all $i$. This holds for any per-token distribution satisfying $\mathbb{E}_x[g_i(x)] = k/E$; it does not require uniform per-token gates.

(b) By Cauchy-Schwarz: $\sum_i f_i P_i \geq \frac{(\sum_i f_i)(\sum_i P_i)}{E} = 1/E$ with equality iff $f_i/P_i$ is constant. The constrained minimum with $\sum_i f_i = 1$ and $\sum_i P_i = 1$ is $1/E$, achieved at $f_i = P_i = 1/E$.

(c) Let each token assign gate weight 1 to one of two experts cyclically, so $f_i = P_i = 1/E$ for all $i$ (both losses minimized). Now perturb: give half the tokens gate weight $1-\epsilon$ for expert 1 and $\epsilon$ for expert 2 (and vice versa for the other half). Then $P_1 = P_2 = 1/2$ still, $f_i$ unchanged — but $\operatorname{Importance}_1 \neq \operatorname{Importance}_2$ if $\epsilon \neq 1-\epsilon$, so $\mathcal{L}_{\text{imp}} > 0$. This demonstrates $\mathcal{L}_{\text{aux}} = \alpha$ while $\mathcal{L}_{\text{imp}} > 0$ is achievable.

---

### Problem 6: Expert Capacity and Token Drop Probability

**Key insight:** Under the normal approximation the overflow probability is a complementary Gaussian CDF evaluated at a threshold proportional to $(\phi - 1)\sqrt{T}$, so the sign of $\phi - 1$ determines whether the limit as $T \to \infty$ is 0 or 1/2.

**Sketch:**

(a) $\mathbb{E}[\text{Load}_i] = T \cdot (k/E) = kT/E$. For $\phi < 1$: capacity $C \approx \phi \cdot kT/E < kT/E$, so overflow fraction $= 1 - \phi > 0$.

(b) Let $\mu = kT/E$, $\sigma^2 = T(k/E)(1-k/E)$. By the CLT:

$$P(\text{Load}_i > C) \approx 1 - \Phi\!\left(\frac{C - \mu}{\sigma}\right) \approx 1 - \Phi\!\left((\phi - 1)\sqrt{\frac{Tk/E}{1 - k/E}}\right)$$

(c) As $T \to \infty$: for $\phi > 1$, the argument $\to +\infty$, so $P \to 0$. For $\phi = 1$, the argument $= 0$ for all $T$, giving $P \to 1 - \Phi(0) = 1/2$.

---

### Problem 7: Capacity Factor Sensitivity

**Key insight:** Near $\phi = 1$, the drop probability changes at a rate proportional to $\sqrt{Tk/E}$ — the standard deviation of the load distribution — so large batches are dramatically more sensitive to small increases in $\phi$.

**Sketch:**

(a) From Problem 6(b): $P_{\text{drop}}(\phi) = 1 - \Phi((\phi-1) \cdot r)$ where $r = \sqrt{Tk/E \cdot (1-k/E)^{-1}}$. The rate parameter governing how fast $P_{\text{drop}}$ falls is $r \propto \sqrt{T}$.

(b) $\frac{dP_{\text{drop}}}{d\phi}\big|_{\phi=1} = -r \cdot \phi_{\text{std}}(0) = -r/\sqrt{2\pi}$, where $\phi_{\text{std}}$ is the standard normal density. Sensitivity grows as $\sqrt{Tk/E}$: at large $T$, even $\Delta\phi = 0.01$ above 1 sharply reduces drops.

(c) Fix $E \cdot C \approx kT\phi = \text{const}$, so $\phi \propto 1/E$ as $E$ varies. As $E$ decreases (fewer larger experts), $k/E$ increases (each expert handles a larger share), reducing $r$ relative to the case with many small experts. The optimal $(\phi, E)$ tradeoff balances the drop reduction from larger $\phi$ against the routing-resolution cost of smaller $E$.

---

### Problem 8: Birthday Problem Analogy for Token Overflow

**Key insight:** In the $E = T$ limit, the probability that at least one expert receives zero tokens approaches $1 - e^{-1}$ via the Poisson approximation to the birthday problem, establishing a universal overflow baseline under random routing.

**Sketch:**

(a) Assigning $T$ tokens uniformly to $E$ experts is identical to placing $T$ balls in $E$ bins. Overflow (bin count $> C$) is the complement of "all bins within capacity" — the exact birthday problem structure with capacity threshold $C = \lceil T/E \rceil$.

(b) $P(\text{expert } i \text{ receives 0}) = (1-1/E)^T$. For $E = T$: $(1-1/E)^E \to e^{-1}$. By inclusion-exclusion and the union bound: $P(\text{at least one empty}) \to 1 - e^{-1} \approx 0.632$.

(c) For $E \ll T$: load $\sim \operatorname{Poisson}(\lambda)$, $\lambda = T/E$. For overflow by $r = \sqrt{\lambda}$ above mean: by the CLT approximation to the Poisson, $P(\text{Load}_i > \lambda + r) \approx 1 - \Phi(r/\sqrt{\lambda}) = 1 - \Phi(1) \approx 0.16$. The exact Poisson large-deviation bound gives $P \sim e^{-r^2/(2\lambda)} = e^{-1/2}$ to leading order for $r = \sqrt{\lambda}$.

---

### Problem 9: Expert-Choice Routing as Bipartite Matching

**Key insight:** Expert-choice routing solves the one-sided (expert-constrained) LP relaxation of the bipartite $b$-matching exactly via greedy column-wise sorting, with perfect expert-side balance following as a structural consequence, at the cost of variable per-token coverage.

**Sketch:**

(a) ILP: maximize $\sum_{t,i} s_{t,i} x_{t,i}$ subject to $x_{t,i} \in \{0,1\}$, $\sum_t x_{t,i} = m$ for all $i$, $\sum_i x_{t,i} \leq k'$ for all $t$.

(b) Relax to $x_{t,i} \in [0,1]$ and drop the per-token constraint. The LP decomposes by expert: for each $i$, maximize $\sum_t s_{t,i} x_{t,i}$ subject to $\sum_t x_{t,i} = m$, $x_{t,i} \geq 0$. The optimal solution sets $x_{t,i} = 1$ for the top-$m$ tokens by $s_{t,i}$ — the greedy expert-choice rule. Any duality gap arises only from the per-token integrality constraint.

(c) $\operatorname{Coverage} \leq 1$ trivially. Under independent selection with probability $m/T$: $P(t \text{ unselected}) = (1 - m/T)^E$. Expected coverage $= 1 - (1 - m/T)^E$. Significantly below 1 when $mE \ll T$.

---

### Problem 10: Expert-Choice Dual Formulation and LP Relaxation

**Key insight:** The dual variables of the expert capacity constraints equal the $(m+1)$-th highest routing score per expert — the market-clearing price at which expert $i$'s capacity is exactly exhausted — with complementary slackness confirming that only tokens above this threshold are selected.

**Sketch:**

(a) LP: max $\sum_{t,i} s_{t,i} x_{t,i}$ s.t. $\sum_t x_{t,i} = m$, $x_{t,i} \geq 0$. Dual: min $m \sum_i \lambda_i + \sum_{t,i} \mu_{t,i}$ s.t. $\lambda_i + \mu_{t,i} \geq s_{t,i}$, $\mu_{t,i} \geq 0$. The dual variables $\lambda_i$ are the shadow prices (capacity rents) for expert $i$.

(b) Set $x_{t,i}^* = 1$ for $t$ in top-$m$ by $s_{t,i}$, else 0. Dual prices $\lambda_i^* = s_{(m+1),i}$ (the $(m+1)$-th largest score). Complementary slackness: for selected $t$, $\mu_{t,i} = s_{t,i} - \lambda_i^* \geq 0$; for unselected $t$, $x_{t,i} = 0$ and $\mu_{t,i} = 0$. Both conditions are satisfied, confirming optimality.

(c) The LP is an instance of the Kantorovich optimal transport problem with uniform expert marginal $\mathbf{1}_E/E$ and free token marginal. The optimal value is the Wasserstein-1 cost between these marginals under the score kernel. Sinkhorn iterations with marginals $(\mathbf{1}_T/T, \mathbf{1}_E/E)$ solve the doubly-constrained version, recovering both token- and expert-balanced assignment.

---

### Problem 11: MoE vs Ensemble vs Product of Experts

**Key insight:** The distinguishing structural feature of MoE is input-conditional gating, which couples all expert gradients through the shared gate and creates the joint training dynamics absent in static ensembles.

**Sketch:**

(a) MoE $\to$ committee when $g_i(x) = 1/E$ for all $x$. In a committee, $\partial \mathcal{L}/\partial W_i = \frac{1}{E}\frac{\partial \mathcal{L}}{\partial y} x^\top$, independent of all other experts. In MoE, $g_i(x)$ depends on all logits via softmax, so $\partial g_i / \partial z_j \neq 0$: gradients for $W_i$ are coupled to all other experts' outputs through the gate.

(b) Gaussian PoE with common variance: $\mu_{\text{PoE}} = \sum_i g_i \mu_i / \sum_i g_i = \sum_i g_i \mu_i$ (coincidentally matching MoE). For unequal variances, PoE precision-weights: $\mu_{\text{PoE}} = (\sum_i g_i \sigma_i^{-2} \mu_i)/(\sum_i g_i \sigma_i^{-2})$. When experts strongly disagree: PoE produces a high-confidence narrow prediction near the majority; MoE averages with gate weights, potentially bimodal.

(c) $\mathcal{R}_i = \bigcap_{j \neq i} \{x : (W_g[:, i] - W_g[:, j])^\top x > 0\}$ — a finite intersection of half-spaces through the origin, hence a convex polyhedral cone. The boundaries are hyperplanes $\{x : (W_g[:, i] - W_g[:, j])^\top x = 0\}$, forming a linear Voronoi diagram with sites at the columns of $W_g^\top$.

---

### Problem 12: Top-1 Routing and Voronoi Specialization

**Key insight:** For fixed label $y$, the assignment boundary is affine; linearity of the expected-loss boundary requires $\mathbb{E}[y|x]$ to be linear in $x$ (i.e., the regression function is linear). The linear router can recover the oracle partition exactly when and only when the oracle's decision boundaries are piecewise-linear.

**Sketch:**

(a) $\mathcal{R}_i = \bigcap_{j \neq i} \{x : (W_g[:, i] - W_g[:, j])^\top x > 0\}$ — a finite intersection of open half-spaces through the origin. Each half-space is convex; the intersection of convex sets is convex. The cone structure follows from the homogeneity of the inequalities.

(b) The linear router can recover oracle assignment iff there exists $W_g$ such that $(W_g^\top x)_i > (W_g^\top x)_j$ for all $j \neq i$ whenever $i^*(x) = i$. This is possible iff the oracle's regions are linearly separable — each region can be separated from all others by a linear classifier. Impossible for oracle boundaries with non-linear (e.g., curved or non-convex) structure.

(c) For fixed label $y$: $\|W_i x - y\|^2 - \|W_j x - y\|^2 = (W_i x - W_j x) \cdot (W_i x + W_j x - 2y) = 0$, an affine constraint in $x$ for fixed $y$ — so the fixed-$y$ boundary is an affine hyperplane. Taking expectation: $\mathbb{E}_y[\cdot] = 0$ iff $(W_i - W_j)x \cdot ((W_i + W_j)x - 2\mathbb{E}[y|x]) = 0$. This is linear in $x$ only if $\mathbb{E}[y|x]$ is linear in $x$ (e.g., $y = Wx + \varepsilon$). If $p(y|x)$ is linear-Gaussian, the partition is a true Voronoi tessellation in feature space and the linear router $W_g$ is expressive enough to represent the optimal routing.

---

### Problem 13: Expert Collapse as a Fixed-Point Instability

**Key insight:** The uniform fixed point is always linearly unstable for positive parameters: the symmetry-breaking Jacobian eigenvalue is $1 + \eta\beta\delta(E-1)/E$, which exceeds 1 for any $\eta, \beta, \delta > 0$. The quantity $\eta\beta\delta(E-1)/E$ controls the *rate* of divergence, not the threshold.

**Sketch:**

(a) Fixed-point: $\rho_i^* = 1/E$, $q_i^* = q^*$ with $\eta(1/E)\delta = 0$ at convergence (quality improvement ceases). Dynamic system: $\rho_i^{(t)} = \operatorname{softmax}(\beta q^{(t)})_i$, $q_i^{(t+1)} = q_i^{(t)} + \eta \rho_i^{(t)} \delta$.

(b) Linearize around $q^* = \frac{1}{E}\mathbf{1}$: let $q_i = 1/E + \epsilon_i$ with $\sum \epsilon_i = 0$. Softmax linearization: $\rho_i \approx 1/E + \beta(\epsilon_i - \bar{\epsilon})$ (first-order, where $\bar{\epsilon} = 0$ by constraint). Update: $\epsilon_i^{(t+1)} = \epsilon_i + \eta\delta \cdot \beta \epsilon_i (E-1)/E = \epsilon_i(1 + \eta\beta\delta(E-1)/E)$. Eigenvalue $\lambda = 1 + \eta\beta\delta(E-1)/E > 1$ for all positive parameters. **The uniform fixed point is unconditionally unstable.** $\eta\beta\delta(E-1)/E$ is the per-step growth rate, not a threshold.

(c) Large $\beta$ (sharp routing) maps small quality differences to large routing differences. Large $\eta$ (fast learning) amplifies each step. Large $E$ stabilizes: more experts dilute any individual's advantage. Practical implications: initialize $W_g$ with small variance (equivalent to small $\beta$) and use LR warmup (small $\eta$ early), keeping the system subcritical during early training.

---

### Problem 14: Router Z-Loss Gradient Analysis

**Key insight:** The z-loss gradient is proportional to $g_i(x)$ (the softmax probability), so the dominant expert — with the largest logit and thus the largest $g_i$ — receives the strongest shrinkage signal, directly counteracting the feedback loop driving collapse.

**Sketch:**

(a) Let $\ell_t = \operatorname{LSE}(h(x_t))$. Then $\mathcal{L}_z = \frac{1}{T}\sum_t \ell_t^2$. By chain rule:

$$\frac{\partial \mathcal{L}_z}{\partial h_i(x_t)} = \frac{2\ell_t}{T} \cdot \frac{\partial \ell_t}{\partial h_i(x_t)} = \frac{2\ell_t}{T} \cdot g_i(x_t) = \frac{2}{T} \cdot \operatorname{LSE}(h(x_t)) \cdot g_i(x_t)$$

(b) LSE bounds: $\max_j h_j \leq \ell_t \leq \max_j h_j + \log E$. For the dominant expert $i^* = \arg\max_j h_j$: $g_{i^*}$ is maximal and $\ell_t \geq h_{i^*}$. So $|\partial \mathcal{L}_z/\partial h_{i^*}| \geq (2/T) h_{i^*} g_{i^*}$, which grows with the logit magnitude — the self-correcting mechanism.

(c) Z-loss gradient: $\propto g_i \cdot \operatorname{LSE}(h)$; acts on logit magnitude. Switch auxiliary loss gradient: $\propto f_i/T$; acts on relative overloading regardless of logit scale. Both apply downward pressure on over-represented experts, but z-loss provides stronger correction when logits are already large — exactly when collapse is most imminent.

---

### Problem 15: Entropy Regularization vs Z-Loss

**Key insight:** Z-loss dominates entropy regularization as a stabilizer in the sharp-routing regime because LSE grows unboundedly with logit magnitude while $|\log g_i|$ is bounded by $\log E$, making z-loss a stronger corrective force precisely when collapse is most dangerous.

**Sketch:**

(a) $\partial(-H)/\partial h_i = g_i(\log g_i + H(g))$. For an above-average expert ($g_i > 1/E$): $\log g_i > -\log E$. When routing is sharp ($H(g)$ small), $\log g_i + H(g) > 0$, so the gradient of the entropy penalty pushes $h_i$ downward, reducing the dominant expert's logit.

(b) Entropy gradient scales as $g_i|\log g_i| \leq g_i \log E$ (bounded by $\log E$). Z-loss gradient scales as $g_i \cdot \operatorname{LSE}(h) \geq g_i \cdot \max h_j$, which is unbounded. In the sharp-routing regime with large logits, z-loss dominates.

(c) LSE is already computed in the numerically stable softmax (the max-subtract step produces $\log\sum e^{h_i - m} + m = \text{LSE}$). Z-loss requires only squaring and averaging this cached value — $O(T)$ extra work. Entropy regularization requires computing $\log g_i$ for every expert-token pair — an additional $O(TE)$ pass. Z-loss is therefore cheaper to fuse with the existing routing computation.

---

### Problem 16: MoE Scaling Laws vs Dense Scaling Laws

**Key insight:** The FLOP-parameter decoupling ratio is exactly $E/k$, and the MoE advantage over the FLOP-matched dense baseline vanishes when experts are redundant and shrinks at large data scale because both models enter the data-limited regime.

**Sketch:**

(a) FLOP-matched: $kN_e = N$. Total MoE: $EN_e = (E/k) \cdot N$. The parameter advantage is exactly $E/k$. For $10\times$ advantage: need $E/k = 10$, e.g., $k=2, E=20$ or $k=1, E=10$.

(b) Redundant experts ($f_i \approx f_j$ for all $i,j,x$): $y = \sum_i g_i f_i \approx f_1$ regardless of gating. The MoE is functionally a single expert, achieving the same loss as a dense model with $kN_e$ active parameters — the FLOP-matched baseline.

(c) Chinchilla form: $L(N, D) = A/N^\alpha + B/D^\beta$. MoE advantage: $\Delta L = L(kN_e, D) - L_{\text{MoE}}(EN_e, D)$. As $D \to \infty$, both terms $\to 0$ and $\Delta L \to 0$. The advantage is largest when $N$ binds (small $D$ relative to $N$) and disappears when $D$ binds.

---

### Problem 17: Soft-Gating Fixed-Point Stability

**Key insight:** The uniform fixed point of two-expert soft gating is neutrally stable because gate saturation (sigmoid $\to 0$ or 1) kills the gradient, so fully specialized fixed points are zero-gradient configurations that are local minima when each expert is genuinely better on its region.

**Sketch:**

(a) $\partial \mathcal{L}/\partial w = \mathbb{E}_x\left[\frac{\partial \mathcal{L}}{\partial y}(f_1(x) - f_2(x)) g_1(x)(1-g_1(x)) x\right]$. Zero gradient when $f_1 = f_2$ everywhere, or when the gate is saturated ($g_1 \in \{0,1\}$).

(b) At $g_1 = 1/2$: the factor $g_1(1-g_1) = 1/4$ is maximal, so any $f_1 - f_2 \neq 0$ generates a nonzero gradient. The fixed point is attractive when $\mathbb{E}[\partial\mathcal{L}/\partial y \cdot (f_1-f_2) x] = 0$ (uncorrelated), repulsive when expert 1 genuinely outperforms expert 2 in expectation.

(c) At a specialized fixed point ($g_1 = 1$ on $\mathcal{R}_1$, $g_1 = 0$ elsewhere): $g_1(1-g_1) = 0$ everywhere, so $\partial\mathcal{L}/\partial w = 0$ — it is a fixed point. It is a local minimum when moving toward $g_1 = 1/2$ increases expected loss, i.e., when expert 1 achieves strictly lower loss on $\mathcal{R}_1$ than expert 2 and vice versa on $\mathcal{R}_2$.

---

### Problem 18: Effective Parameter Count and Routing Efficiency

**Key insight:** The effective parameter count is bounded between active and total parameters, and Mixtral's empirical dominance over a 13B dense model while falling short of a 46.7B dense model bounds routing efficiency to a non-trivial interval, confirming partial but incomplete expert specialization.

**Sketch:**

(a) Left bound ($N_{\text{eff}} \geq kN_e$): at minimum the model utilizes its active parameters, matching the FLOP-matched dense baseline. Right bound ($N_{\text{eff}} \leq EN_e$): perfect oracle routing cannot exceed the capacity of all parameters. Both extremes are achieved only in degenerate (redundant or perfect) routing.

(b) $\eta = 0$ ($N_{\text{eff}} = kN_e$): all experts identical, zero gain from additional parameters. $\eta = 1$ ($N_{\text{eff}} = EN_e$): each expert holds fully non-overlapping knowledge. For Gaussian experts with routing mutual information $I$: $N_{\text{eff}} \approx kN_e \cdot (E/k)^\eta$ (power-law interpolation), giving $\eta = \log(N_{\text{eff}}/kN_e) / \log(E/k)$.

(c) Mixtral: $kN_e \approx 12.9\text{B}$, $EN_e \approx 46.7\text{B}$, $E/k = 4$. If $N_{\text{eff}} > 13\text{B}$ and $N_{\text{eff}} < 46.7\text{B}$: $\eta = \log(N_{\text{eff}}/12.9)/\log 4 \in (0, 1)$. This confirms experts are partially specialized but not perfectly non-overlapping — routing efficiency is positive but meaningfully below 1.

---

## Algorithmic Applications

### Problem 19: Noisy Top-k Gating Forward Pass

**Key insight:** The noisy top-k forward pass is a three-stage pipeline — logit perturbation, sparse masking, weighted aggregation — with only the top-k selection indices, gate weights, noise values, and noise scales needing to be cached for the backward pass.

**Sketch:**

```
# Inputs: x (d,), W_g (d, E), W_n (d, E), k
# Forward:
z = W_g.T @ x                        # (E,)  clean logits
s = softplus(W_n.T @ x)              # (E,)  learned noise scales
eps = sample_normal(E)               # (E,)  fresh noise per token
H = z + eps * s                      # (E,)  noisy logits

# KeepTopK
topk_vals, topk_idx = topk(H, k)     # (k,)  top-k values and indices
mask = full(E, -inf)
mask[topk_idx] = topk_vals           # (E,)  sparse logits

# Numerically stable sparse softmax
# Mask non-finite entries before exponentiation to avoid NaN
finite = mask != -inf
g = zeros(E)
g[finite] = softmax(mask[finite])    # (k,)  gate weights over selected experts

# Expert dispatch and aggregation
y = zeros(d)
for i in topk_idx:
    y += g[i] * expert_i.forward(x) # (d,)  weighted expert output

# Cache for backward: z, s, eps, topk_idx, g
# Time complexity: O(E*d) routing projections + O(E log k) top-k + O(k*d) experts
```

---

### Problem 20: Expert Dispatch with Capacity Enforcement

**Key insight:** Capacity enforcement requires sorting per-expert token lists by routing score before truncating at $C$, and dropped tokens receive zero output — blocking gradient flow to expert FFN weights while the router still receives gradient through the kept tokens' routing scores.

**Sketch:**

```
# Inputs: X (T, d), W_g (d, E), C = floor(T/E * phi)
S = softmax(X @ W_g)                 # (T, E) routing scores
assign = argmax(S, axis=1)           # (T,)  expert index per token
score_assigned = S[arange(T), assign]# (T,)  score for assigned expert

output = zeros(T, d)
for i in range(E):
    ids = where(assign == i)         # tokens routed to expert i
    sorted_ids = sort_desc(ids, key=score_assigned)  # sort by score
    kept = sorted_ids[:C]            # first C tokens retained
    # dropped = sorted_ids[C:]      # remainder: output stays zero

    if len(kept) > 0:
        expert_out = expert_i.forward(X[kept])       # (|kept|, d)
        output[kept] = S[kept, i:i+1] * expert_out   # weighted scatter

# Gradient analysis:
# - kept tokens: gradient flows to expert FFN weights and through S to router
# - dropped tokens: output = 0 => no gradient to expert FFN weights
#   but S[t, assign[t]] participates in score computation => router still
#   receives gradient for dropped tokens (via S), just not expert params
# Correction for gradient bias: scale loss by T / sum(len(kept[i]) for i in E)
```

---

### Problem 21: Router Z-Loss Numerically Stable Implementation

**Key insight:** The LSE value is already computed as a byproduct of numerically stable softmax, so z-loss can be fused into the routing computation with $O(T)$ additional work — negligible compared to the $O(TE)$ cost of computing the score matrix.

**Sketch:**

```
# Forward: H is (T, E) logits (same as routing logits)
def zloss_forward(H):
    m = max(H, axis=1, keepdim=True)          # (T, 1)  per-token max
    shifted = H - m                            # (T, E)  numerically stable
    log_sum = log(sum(exp(shifted), axis=1))   # (T,)    log sum of shifted
    lse = log_sum + m.squeeze()               # (T,)    true LSE(h_t)
    Lz = mean(lse ** 2)                        # scalar
    return Lz, lse                             # lse cached for backward

# Backward: reuse lse from forward and g (softmax) already computed for routing
def zloss_backward(lse, g, T):
    # d Lz / d H[t,i] = (2/T) * lse[t] * g[t,i]
    dH = (2.0 / T) * lse[:, None] * g         # (T, E)  no redundant computation

# Combined loss
Ltask  = cross_entropy(logits, labels)
f_i    = compute_dispatch_fractions(routing_decisions)   # stop-gradient
P_i    = mean_gate_probs(g)                              # differentiable
Laux   = alpha * E * sum(f_i * P_i)
Lz_val, lse = zloss_forward(H)
L = Ltask + alpha * Laux + beta * Lz_val
# alpha = 1e-2 (Switch Transformer): controls load balance incentive
# beta in [1e-4, 1e-2] (ST-MoE): controls logit magnitude / training stability
# Tradeoff: larger alpha => stronger load balance but may hurt task quality;
#           larger beta => smaller logits (stabler training) but may restrict routing sharpness
```

---

### Problem 22: Expert-Choice Routing Forward Pass

**Key insight:** Expert-choice routing guarantees $|\mathcal{B}_i| = m$ for all $i$ by construction — no auxiliary loss needed for expert-side balance — but tokens selected by zero experts produce zero output, requiring explicit coverage tracking and residual passthrough.

**Sketch:**

```
# Inputs: X (T, d), W_g (d, E), m = floor(phi * T / E)
S = softmax(X @ W_g)                  # (T, E)  token-expert scores

# Expert-side top-m selection
selected = {}                          # expert -> list of token indices
for i in range(E):
    col = S[:, i]                      # (T,)  scores for expert i
    top_ids = argsort(col)[-m:]        # top-m indices (exactly m, guaranteed)
    selected[i] = top_ids

# Output assembly
output = zeros(T, d)                   # (T, d)  default: zero for unselected
for i in range(E):
    ids = selected[i]                  # (m,)
    out_i = expert_i.forward(X[ids])   # (m, d)
    for t, o in zip(ids, out_i):
        output[t] += S[t, i] * o       # accumulate weighted contributions

# Coverage computation
covered_set = union(selected[i] for i in range(E))
coverage = len(covered_set) / T
# Expected coverage under independence: 1 - (1 - m/T)^E

# Gradient flow:
# - (t, i) pairs NOT in any selected[i]: S[t,i] does not enter output
#   => S[t,i] receives NO gradient from this forward pass
# - Contrast with token-choice: router always receives gradient from all E scores per token
# - Implication: expert-choice router only learns from token-expert pairs
#   that experts decide are worth selecting -- unselected pairs are invisible to training
```

---

### Problem 23: Distributed Expert Parallelism Protocol

**Key insight:** Each MoE layer requires two all-to-all collectives communicating $O(Td/E)$ bytes per device per collective; for typical Transformer dimensions this communication volume is smaller than tensor-parallel all-reduce only when $E$ is large relative to $T/d$.

**Sketch:**

```
# Setup: E devices, each with T/E tokens, dimension d, top-1 routing
# Notation: bytes_fp = 2 (bfloat16) or 4 (float32)

# All-to-all 1 (dispatch):
# Each device sends C tokens to each of E-1 peer devices.
# C = floor((T/E) * phi) tokens per expert slot.
# Bytes per device per collective (upper bound):
bytes_dispatch = C * (E - 1) * d * bytes_fp  # ~ T * phi * d * bytes_fp

# Expert FFN computation (local, no communication)
# Each device processes its received C tokens through its expert.

# All-to-all 2 (gather):
# Expert outputs returned to originating devices. Same volume as dispatch.
bytes_gather = bytes_dispatch

# Total per MoE layer: 2 * C * (E-1) * d * bytes_fp per device

# Comparison: tensor parallelism (2-device row-parallel linear, W in R^{d x d})
# Each device holds half the columns; computes partial Y = X W_local (T, d/2)
# All-reduce (sum) over 2 devices: each device sends T*(d/2) elements.
bytes_tensor_parallel = T * d / 2 * bytes_fp  # per device per layer

# Expert parallelism dominates when:
# C * (E-1) >> T/2  =>  phi*(T/E)*(E-1) >> T/2  =>  phi*(E-1)/E >> 1/2
# For E >= 2 and phi >= 1 this is almost always true => expert parallelism
# communicates more than 2-device tensor parallelism in typical settings.
# Tensor parallelism is better when T >> C*(E-1): very large batch, few experts.

# Gradient bias correction for dropped tokens:
# Problem: dropped tokens have labels but zero expert output
#          => zero gradient contribution to expert FFN weights
#          => systematic under-training for overloaded experts
# Correction: importance weighting
kept_count = sum(min(count(assign==i), C) for i in range(E))  # total kept
loss_scale = T / kept_count                                     # reweighting factor
L_corrected = loss_scale * L_task_over_kept_tokens
# This is unbiased in expectation for loss magnitude, but does NOT restore
# the directional gradient from missing token types -- it reduces magnitude
# bias without eliminating the selection bias on expert parameters.
```
