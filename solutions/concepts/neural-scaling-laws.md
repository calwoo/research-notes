# Solutions: Neural Scaling Laws

---

## 1. Derivation Problems

---

### Problem 1 — Compute-Optimal Scaling via Direct Substitution

**Setup.** The loss decomposition is

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta},$$

with compute constraint $C = 6ND$, where $C$ is the total FLOP budget, $N$ is the number of non-embedding parameters, and $D$ is the number of training tokens.

**Step 1: Eliminate $D$.** Solve the constraint for $D$:

$$D = \frac{C}{6N}.$$

**Step 2: Substitute into $L$.** Replace $D$ in the loss:

$$L(N) = E + \frac{A}{N^\alpha} + \frac{B}{\left(\frac{C}{6N}\right)^\beta} = E + \frac{A}{N^\alpha} + \frac{B \cdot (6N)^\beta}{C^\beta}.$$

Expanding $(6N)^\beta = 6^\beta N^\beta$:

$$L(N) = E + \frac{A}{N^\alpha} + \frac{6^\beta B}{C^\beta} N^\beta.$$

**Step 3: Differentiate and set to zero.** Since $E$ is a constant and $C$ is fixed, minimize over $N$:

$$\frac{dL}{dN} = -\frac{\alpha A}{N^{\alpha+1}} + \frac{6^\beta \beta B}{C^\beta} N^{\beta - 1} = 0.$$

**Step 4: Solve for $N^*$.** Rearrange:

$$\frac{6^\beta \beta B}{C^\beta} N^{\beta - 1} = \frac{\alpha A}{N^{\alpha + 1}}.$$

Multiply both sides by $N^{\alpha+1}$:

$$\frac{6^\beta \beta B}{C^\beta} N^{\alpha + \beta} = \alpha A.$$

Isolate $N^{\alpha+\beta}$:

$$N^{\alpha+\beta} = \frac{\alpha A \cdot C^\beta}{6^\beta \beta B}.$$

Taking both sides to the power $\frac{1}{\alpha+\beta}$:

$$\boxed{N^* = \left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha+\beta}} \cdot \left(\frac{C}{6}\right)^{\frac{\beta}{\alpha+\beta}} \propto C^{\frac{\beta}{\alpha+\beta}}.}$$

**Step 5: Recover $D^*$.** From $D = C/(6N)$ and $N^* \propto C^{\beta/(\alpha+\beta)}$:

$$D^* = \frac{C}{6 N^*} \propto \frac{C}{C^{\beta/(\alpha+\beta)}} = C^{1 - \beta/(\alpha+\beta)} = C^{\alpha/(\alpha+\beta)}.$$

$$\boxed{D^* \propto C^{\frac{\alpha}{\alpha+\beta}}.}$$

The exponents $\beta/(\alpha+\beta)$ for $N^*$ and $\alpha/(\alpha+\beta)$ for $D^*$ sum to 1, which is consistent: $C = 6N^* D^*$ must scale as $C^1$.

---

### Problem 2 — Compute-Optimal Scaling via Lagrangian Optimization

**Setup.** Form the Lagrangian with multiplier $\lambda$ enforcing the constraint $6ND = C$:

$$\mathcal{L}(N, D, \lambda) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta} + \lambda(6ND - C).$$

**Step 1: First-order condition in $N$.** Differentiate $\mathcal{L}$ with respect to $N$ and set to zero:

$$\frac{\partial \mathcal{L}}{\partial N} = -\frac{\alpha A}{N^{\alpha+1}} + 6\lambda D = 0.$$

Solve for $\lambda$:

$$\lambda = \frac{\alpha A}{6 D N^{\alpha+1}}. \tag{FOC-N}$$

**Step 2: First-order condition in $D$.** Differentiate $\mathcal{L}$ with respect to $D$ and set to zero:

$$\frac{\partial \mathcal{L}}{\partial D} = -\frac{\beta B}{D^{\beta+1}} + 6\lambda N = 0.$$

Solve for $\lambda$:

$$\lambda = \frac{\beta B}{6 N D^{\beta+1}}. \tag{FOC-D}$$

**Step 3: Eliminate $\lambda$.** Set (FOC-N) equal to (FOC-D):

$$\frac{\alpha A}{6 D N^{\alpha+1}} = \frac{\beta B}{6 N D^{\beta+1}}.$$

Multiply both sides by $6 D N^{\alpha+1} D^{\beta+1} / 1$:

$$\alpha A \cdot D^\beta = \beta B \cdot N^\alpha. \tag{Optimality}$$

This is the **optimality condition**: at the constrained optimum, the marginal loss reduction from a fractional increase in $N$ equals that from a fractional increase in $D$ (after accounting for the constraint).

**Step 4: Solve for $D$ in terms of $N$.** From (Optimality):

$$D^\beta = \frac{\beta B}{\alpha A} N^\alpha \implies D = \left(\frac{\beta B}{\alpha A}\right)^{1/\beta} N^{\alpha/\beta}. \tag{D-N relation}$$

**Step 5: Substitute into the constraint $6ND = C$.** Replace $D$:

$$6N \cdot \left(\frac{\beta B}{\alpha A}\right)^{1/\beta} N^{\alpha/\beta} = C.$$

$$6 \left(\frac{\beta B}{\alpha A}\right)^{1/\beta} N^{1 + \alpha/\beta} = C.$$

The exponent on $N$ is $1 + \alpha/\beta = (\alpha + \beta)/\beta$. Solving:

$$N^{(\alpha+\beta)/\beta} = \frac{C}{6} \cdot \left(\frac{\alpha A}{\beta B}\right)^{1/\beta}.$$

$$\boxed{N^* \propto C^{\beta/(\alpha+\beta)}.}$$

By symmetry (swap $N \leftrightarrow D$, $A \leftrightarrow B$, $\alpha \leftrightarrow \beta$):

$$\boxed{D^* \propto C^{\alpha/(\alpha+\beta)}.}$$

This matches the result from Problem 1 exactly. The Lagrangian method confirms the answer while additionally yielding the optimality condition (Optimality), which will be used in Problems 3 and 4.

---

### Problem 3 — Equal Scaling Condition

**Claim.** $N^* \propto D^*$ if and only if $\alpha = \beta$.

**Algebraic proof.** From Problem 1:

$$N^* \propto C^{\beta/(\alpha+\beta)}, \qquad D^* \propto C^{\alpha/(\alpha+\beta)}.$$

For $N^* \propto D^*$ we require the two exponents to be equal:

$$\frac{\beta}{\alpha+\beta} = \frac{\alpha}{\alpha+\beta}.$$

Since $\alpha + \beta > 0$, this reduces to:

$$\boxed{\beta = \alpha.}$$

This is the necessary and sufficient condition for $N^* \propto D^*$.

**Verification via the optimality condition.** When $\alpha = \beta$, the optimality condition (Optimality) from Problem 2 becomes:

$$\alpha A \cdot D^\alpha = \alpha B \cdot N^\alpha \implies \left(\frac{D}{N}\right)^\alpha = \frac{B}{A} \implies \frac{D^*}{N^*} = \left(\frac{B}{A}\right)^{1/\alpha},$$

which is a fixed constant (independent of $C$), confirming $N^* \propto D^*$.

**Geometric interpretation.** In $(\log N, \log D)$ space, the loss surface $L(N,D) = E + A e^{-\alpha \log N} + B e^{-\beta \log D}$ has iso-loss contours that are convex curves. Near the compute-optimal point on the constraint line $\log N + \log D = \text{const}$, the curvature of the loss in the $\log N$ direction is governed by $\alpha$ and the curvature in the $\log D$ direction by $\beta$. When $\alpha = \beta$, the loss surface has equal curvature (equal "stiffness") in both log-directions, so the gradient of $L$ points equally toward reducing $N$ and $D$ — the constraint line therefore intersects the gradient field symmetrically, yielding $N^* = D^*$ up to the constant $(B/A)^{1/\alpha}$. When $\alpha \neq \beta$, the loss surface is elongated in one direction, tilting the optimum.

---

### Problem 4 — Deriving $C \approx 6ND$ from FLOP Counting

Let $W \in \mathbb{R}^{m \times n}$ and input vector $x \in \mathbb{R}^n$, with output $y = Wx \in \mathbb{R}^m$.

#### Part (a): Forward Pass FLOPs

Computing $y = Wx$ requires forming $m$ dot products, each of length $n$. Each dot product requires $n$ multiplications and $n-1$ additions $\approx n$ multiply-add operations. Each multiply-add counts as **2 FLOPs** (1 multiply + 1 add).

$$\text{Forward FLOPs} = m \cdot n \cdot 2 = 2mn. \tag{Fwd}$$

#### Part (b): Backward Pass FLOPs

**Gradient w.r.t. input** $\partial \mathcal{L}/\partial x = W^\top (\partial \mathcal{L}/\partial y)$. Here $W^\top \in \mathbb{R}^{n \times m}$ acts on $\partial \mathcal{L}/\partial y \in \mathbb{R}^m$, costing:

$$\text{FLOPs}(\partial x) = 2mn. \tag{Bwd-x}$$

**Gradient w.r.t. weights** $\partial \mathcal{L}/\partial W = (\partial \mathcal{L}/\partial y) x^\top$. This is an outer product of vectors in $\mathbb{R}^m$ and $\mathbb{R}^n$, producing a matrix in $\mathbb{R}^{m \times n}$, costing:

$$\text{FLOPs}(\partial W) = 2mn. \tag{Bwd-W}$$

**Total per layer per token:**

$$\text{FLOPs} = 2mn + 2mn + 2mn = 6mn. \tag{Total}$$

#### Part (c): Scaling to a Full Transformer

A transformer's parameter count $N$ is dominated by the weight matrices in its attention and MLP layers (excluding embedding matrices, which are tied and not trained per-step). Across all $L$ layers, the total parameter count is approximately $N \approx \sum_\ell m_\ell n_\ell$ (summing over all linear layers per layer, times the number of layers).

From (Total), each linear sublayer with $m_\ell n_\ell$ parameters contributes $6 m_\ell n_\ell$ FLOPs per token. Summing over all layers:

$$\text{FLOPs per token} \approx 6 \sum_\ell m_\ell n_\ell \approx 6N.$$

Training on $D$ tokens gives:

$$\boxed{C \approx 6ND.}$$

The factor 6 decomposes as $2 \times 3$: factor 2 from multiply-add counting, and factor 3 from the three passes (forward, backward-input, backward-weight). This is an approximation because: (i) attention softmax and layer norm have negligible FLOPs relative to matmuls; (ii) embedding layers are excluded; (iii) the $-1$ in $n-1$ additions is dropped.

---

### Problem 5 — Symmetric Exponents: $\alpha = \beta$

**Step 1: Apply the general formulas.** From Problem 1, setting $\alpha = \beta$:

$$N^* \propto C^{\beta/(\alpha+\beta)} = C^{\alpha/(2\alpha)} = C^{1/2}.$$

$$D^* \propto C^{\alpha/(\alpha+\beta)} = C^{\alpha/(2\alpha)} = C^{1/2}.$$

$$\boxed{N^* \propto C^{1/2}, \qquad D^* \propto C^{1/2},}$$

regardless of the specific values of $A$, $B$, and $E$. (The constants $A$, $B$, $E$ affect the prefactors of $N^*$ and $D^*$, specifically the ratio $D^*/N^* = (B/A)^{1/\alpha}$, but not the scaling exponent with $C$.)

**Step 2: Verify the exponents sum correctly.** We need $N^* D^* \propto C^1$. Indeed: $C^{1/2} \cdot C^{1/2} = C^1$. The constraint $C = 6ND$ is satisfied.

**Interpretation of $\alpha = \beta$.** The power-law terms in the loss are $A/N^\alpha$ (capacity-limited excess loss) and $B/D^\alpha$ (data-limited excess loss). Both decay with the same exponent $\alpha$. This means:

- Doubling $N$ reduces the capacity term by factor $2^{-\alpha}$.
- Doubling $D$ reduces the data term by factor $2^{-\alpha}$.

So parameters and data have **equal marginal returns** in log-scale: spending a fraction of compute on parameters yields the same loss reduction as spending the same fraction on data. When $\alpha < \beta$ (as Kaplan found), parameters have lower marginal returns — a given fractional increase in $N$ reduces loss by less than the same fractional increase in $D$ — so the optimal allocation tilts toward data. When $\alpha > \beta$, the opposite holds.

---

## 2. Conceptual Questions

---

### Question 1 — The Irreducible Loss $E = L_\infty$

The irreducible loss $E$ is the **entropy of the true data-generating distribution** — the minimum expected cross-entropy loss achievable by any predictor, regardless of its size or the amount of data it has seen. Formally, if $p(x_{t+1} \mid x_1, \ldots, x_t)$ is the true conditional distribution of the next token, then $E = \mathbb{E}[-\log p(x_{t+1} \mid x_1,\ldots,x_t)]$ is the Shannon entropy rate of the source. No model can achieve loss below $E$ because the Bayes-optimal predictor — which has access to the exact true distribution — already achieves exactly $E$; any other predictor either matches or exceeds it. The irreducibility comes from genuine stochasticity in the data: even a perfect model that has memorized all training data and knows the true conditional distribution cannot predict the outcome of an inherently random event.

A concrete example in language modeling: given the context "She opened the door and saw a", the next word could be any of dozens of plausible continuations ("cat", "man", "stranger", "police officer", ...). The true distribution assigns nonzero probability to many continuations. The expected $-\log$ probability under the true distribution is nonzero — this is the irreducible entropy from genuine lexical and semantic ambiguity in language, not from any model limitation.

The quantity $E$ is **not practically computable**: it would require knowing the true distribution $p$ over all possible text sequences, which is precisely what a language model is attempting to learn. We can only estimate upper bounds on $E$ via increasingly capable models, but we cannot evaluate the true Shannon entropy of natural language directly.

---

### Question 2 — Interpreting $\alpha_D > \alpha_N$

The exponents $\alpha_N \approx 0.076$ and $\alpha_D \approx 0.095$ (Kaplan et al.) appear in $L \approx E + A/N^{\alpha_N} + B/D^{\alpha_D}$. A larger exponent means the corresponding term decays faster per unit of log-scaling: increasing $D$ by a factor $r$ reduces the data term by $r^{-\alpha_D}$, while increasing $N$ by the same factor $r$ reduces the capacity term by only $r^{-\alpha_N}$. Since $\alpha_D > \alpha_N$, **data is more efficient than parameters** at reducing excess loss per unit of log-scaling: for the same multiplicative increase, data reduces its term by a larger factor.

The **reversal in the Kaplan compute-optimal conclusion** (which says parameters should grow faster than data) follows directly from the compute-optimal exponents. Plugging $\alpha = \alpha_N$, $\beta = \alpha_D$ into the formulas from Problem 1:

$$N^* \propto C^{\alpha_D/(\alpha_N + \alpha_D)}, \qquad D^* \propto C^{\alpha_N/(\alpha_N + \alpha_D)}.$$

Since $\alpha_D > \alpha_N$, the exponent on $N^*$ exceeds $1/2$ and the exponent on $D^*$ is less than $1/2$: **parameters are allocated more compute than data**. This seems paradoxical given that data reduces loss more efficiently per unit, but it follows from the constrained optimization: the resource with the *slower marginal return* (parameters) must be scaled more aggressively to keep both terms balanced at the optimum. Equivalently, since data is more efficient, you need less of it to match a given capacity level; the constraint $C = 6ND$ then forces the remaining compute into more parameters.

This is why Kaplan recommended investing compute primarily in model size (with fixed dataset size), while Chinchilla revised the exponents using the IsoFLOP design and found $\alpha_N \approx \alpha_D$, yielding a more balanced $N^* \approx D^*$ scaling.

---

### Question 3 — Why IsoFLOP Eliminates Undertrained-Model Bias

#### Part (a): Bias mechanism in Kaplan's $L(N)$ measurement

Kaplan's approach fits $L(N)$ by training each model of size $N$ to convergence on the full available dataset — that is, $D$ is held fixed at a large constant across all $N$. When $N$ is large, however, the model has more capacity than the fixed dataset can exploit: the model is data-saturated (undertrained in the sense of not having enough tokens per parameter). In the loss decomposition $L = E + A/N^{\alpha_N} + B/D^{\alpha_D}$, the data term $B/D^{\alpha_D}$ is constant (since $D$ is fixed), so the measured $L(N)$ only reflects changes in the capacity term $A/N^{\alpha_N}$. But a large undertrained model has effectively "used up" most of the capacity term reduction it can achieve given the fixed $D$; further increases in $N$ yield diminishing returns not because $\alpha_N$ is small, but because the data term dominates and is not improving. The measured slope of $\log L$ vs $\log N$ is therefore flatter than the true $\alpha_N$, biasing $\hat{\alpha}_N$ downward.

#### Part (b): Why fixing $C$ and sweeping $(N, D)$ removes this bias

In the IsoFLOP design, for each compute budget $C_i$, a range of $(N_j, D_j)$ pairs are trained with $6 N_j D_j = C_i$. As $N_j$ increases, $D_j = C_i / (6 N_j)$ decreases correspondingly, so the token-to-parameter ratio $D_j / N_j$ is always coupled to model size. A larger model always trains on proportionally fewer tokens per parameter, keeping the model near the computationally optimal regime rather than the data-saturated regime. The measured loss at each $(N_j, D_j)$ reflects the true trade-off between capacity and data at that compute level: neither term dominates, and the estimated slope of $\log L$ vs $\log N$ (across different $N_j$ within the same $C_i$ budget) captures the true $\alpha_N$. The undertrained-model bias is absent because we never fix $D$ while varying $N$; instead, $D$ always adjusts to keep the model on the IsoFLOP pareto frontier.

---

### Question 4 — Why $D^*/N^*$ Requires $A$ and $B$

From the optimality condition derived in Problem 2 (equation (Optimality)):

$$\alpha A \cdot (D^*)^\beta = \beta B \cdot (N^*)^\alpha.$$

Rearranging:

$$\frac{(D^*)^\beta}{(N^*)^\alpha} = \frac{\beta B}{\alpha A}.$$

The ratio $D^*/N^*$ (the optimal token-to-parameter ratio) is a function of both the exponents $(\alpha, \beta)$ and the amplitude constants $(A, B)$. Even knowing $\alpha = \beta$ exactly, the ratio becomes $(D^*/N^*)^\alpha = B/A$, so $D^*/N^* = (B/A)^{1/\alpha}$, which depends on the empirically fitted amplitudes $A$ and $B$.

The exponents $\alpha$ and $\beta$ only determine **how the optimal ratio scales with compute** (i.e., whether $N^*$ grows faster or slower than $D^*$ as $C$ increases). They say nothing about the absolute level of the ratio at any given compute budget. The constants $A$ and $B$ encode the scale of the capacity-limited and data-limited loss components — which depend on model architecture, tokenizer, dataset characteristics, and other factors — and must be estimated from data. Chinchilla's "20 tokens per parameter" rule is obtained by fitting $A$, $B$, $\alpha$, $\beta$ simultaneously to the observed loss surface, then evaluating $D^*/N^*$ at the fitted optimality condition for realistic compute budgets. Without $A$ and $B$, scaling exponents alone cannot pin down the prefactor.

---

### Question 5 — Compute-Optimal Strategy with a Constrained Model Size $N_{\max}$

In the unconstrained case, the compute-optimal model size grows as $N^*(C) \propto C^{\beta/(\alpha+\beta)}$. For large enough $C$, $N^*(C)$ will exceed the hardware-imposed ceiling $N_{\max}$. Once the unconstrained optimum exceeds the constraint — i.e., $N^*(C) > N_{\max}$ — the constrained optimum is simply to set $N = N_{\max}$ and allocate all remaining compute to data:

$$D^* = \frac{C}{6 N_{\max}}.$$

In this regime, $D^*$ grows linearly with $C$ rather than sublinearly ($D^* \propto C^{\alpha/(\alpha+\beta)} < C^1$ in the unconstrained case). The model is trained for more and more tokens as $C$ increases, but the capacity is capped.

The achievable loss in the constrained regime is

$$L(N_{\max},\, C/(6 N_{\max})) = E + \frac{A}{N_{\max}^\alpha} + \frac{B \cdot (6 N_{\max})^\beta}{C^\beta}.$$

As $C \to \infty$, the data term $\propto C^{-\beta}$ vanishes, and the loss approaches $E + A/N_{\max}^\alpha > E + A/(N^*(C))^\alpha$. Thus:

1. The achievable loss is **strictly higher** than the unconstrained optimum for all large $C$, because the capacity term is frozen at $A/N_{\max}^\alpha$ rather than continuing to shrink.
2. The loss **does continue to decrease** with $C$ (via the data term $B/D^\beta = B(6 N_{\max}/C)^\beta \to 0$), but converges to a floor $E + A/N_{\max}^\alpha$ rather than approaching $E$.
3. The gap between constrained and unconstrained loss grows with $C$: the unconstrained loss continues to decrease in both the capacity and data terms, while the constrained loss can only improve the data term.

In practice this means: if you are hardware-constrained, invest all additional compute in training tokens (more epochs or more data), but be aware you are operating suboptimally and that there is a hard loss floor imposed by the model size cap.

---

## 3. Implementation Sketches

---

### Sketch 1 — IsoFLOP Sweep

**Goal.** For each compute budget $C_i$, find the compute-optimal model size $N^*(C_i)$ by training multiple $(N, D)$ pairs along the IsoFLOP curve and fitting a minimum.

```
# Inputs
compute_budgets = [C_1, C_2, ..., C_k]   # in FLOPs, e.g. logspaced from 1e18 to 1e24
models_per_budget = 8                      # number of (N, D) pairs per budget
flop_constant = 6                          # C ≈ 6ND

# For each compute budget C_i:
results = {}  # maps C_i -> list of (N_j, D_j, loss_j)

for C_i in compute_budgets:

    # Step (a): Generate (N_j, D_j) pairs satisfying 6 * N_j * D_j = C_i.
    # Choose N_j log-uniformly spanning a wide range around an initial guess N_guess.
    # N_guess = (C_i / 6) ** 0.5  (midpoint assuming N* ~ D*)
    # Span roughly 0.25x to 4x the guess in log-space.

    N_guess = (C_i / flop_constant) ** 0.5
    N_values = logspace(N_guess * 0.25, N_guess * 4.0, num=models_per_budget)

    pairs = []
    for N_j in N_values:
        D_j = C_i / (flop_constant * N_j)
        if D_j < min_tokens_threshold:      # e.g. D_j < 1e8; skip degenerate runs
            continue
        pairs.append((N_j, D_j))

    # Step (b): Number of models per budget.
    # Use 6–10 models per budget to get a smooth parabola in (log N, loss) space.
    # More models per budget → better $N^*$ estimate; fewer → more budgets covered.
    # 8 is a reasonable default; reduce to 4–5 for very expensive budgets.

    # Step (c): Train each model and measure loss.
    for (N_j, D_j) in pairs:
        model = initialize_transformer(num_params=N_j)        # architecture search
        loss_j = train(model, num_tokens=D_j, track_eval_loss=True)
        # loss_j = final held-out cross-entropy (nats) on a fixed validation set
        results[C_i].append((N_j, D_j, loss_j))

# Step (d): Estimate N*(C_i) for each budget.
N_opt = {}
for C_i in compute_budgets:
    data = results[C_i]
    N_vals = [r[0] for r in data]
    L_vals = [r[2] for r in data]

    # Fit a quadratic to (log N, loss) to find minimum:
    log_N = [log(N) for N in N_vals]
    coeffs = fit_quadratic(log_N, L_vals)   # [a, b, c] s.t. loss ~ a*(logN)^2 + b*logN + c
    # Minimum at log_N* = -b / (2a)
    log_N_star = -coeffs[1] / (2 * coeffs[0])
    N_opt[C_i] = exp(log_N_star)

# Final output: a table of (C_i, N_opt[C_i], D_opt[C_i])
# D_opt[C_i] = C_i / (6 * N_opt[C_i])

# Fit the scaling exponent from (C_i, N_opt[C_i]):
# log N_opt = (beta/(alpha+beta)) * log C + const
# OLS regression of log(N_opt) on log(C_i) gives the exponent beta/(alpha+beta).
log_C = [log(C) for C in compute_budgets]
log_N_opt = [log(N_opt[C]) for C in compute_budgets]
slope, intercept = ols_fit(log_C, log_N_opt)
# slope = beta / (alpha + beta)
# Similarly for D: slope_D = 1 - slope = alpha / (alpha + beta)
```

**Design choices.**

- Log-uniform spacing of $N_j$ ensures adequate coverage of the loss parabola on both sides of the minimum.
- The quadratic fit in $\log N$ is justified because near the optimum, $L(N) \approx \text{const} + \frac{1}{2} L''(\log N^*)(\log N - \log N^*)^2$ to second order.
- The minimum-tokens threshold guards against configurations where $D$ is too small for reliable loss estimation (high variance from data scarcity).
- Using a held-out validation set (not training loss) avoids confounding the loss measurement with memorization of training data.

---

### Sketch 2 — Log-Linear Regression for Scaling Exponents

**Setup.** We have observations $\{(N_i, L_i)\}_{i=1}^n$ where $D$ is held fixed, so the model is $L_i \approx E + A / N_i^\alpha$. Assuming $E$ and $A$ are known (or have been subtracted), we work with the excess loss $\ell_i = L_i - E > 0$ and model

$$\ell_i = A / N_i^\alpha \implies \log \ell_i = \log A - \alpha \log N_i.$$

**Step 1: Transform variables.** Define:

$$y_i = \log \ell_i = \log(L_i - E), \qquad x_i = \log N_i.$$

The model becomes the linear regression $y_i = \beta_0 + \beta_1 x_i + \varepsilon_i$, where $\beta_0 = \log A$ and $\beta_1 = -\alpha$.

**Step 2: OLS normal equations.** Let $\bar{x} = \frac{1}{n}\sum x_i$ and $\bar{y} = \frac{1}{n}\sum y_i$. The OLS estimators are:

$$\hat{\beta}_1 = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2} = \frac{S_{xy}}{S_{xx}},$$

$$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}.$$

The scaling exponent estimate is:

$$\hat{\alpha} = -\hat{\beta}_1 = -\frac{S_{xy}}{S_{xx}}.$$

**Step 3: Confidence interval for $\hat{\alpha}$.** The residual variance is estimated by:

$$\hat{\sigma}^2 = \frac{1}{n-2} \sum_{i=1}^n (y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i)^2 = \frac{\text{RSS}}{n-2}.$$

The standard error of $\hat{\beta}_1$ is:

$$\text{SE}(\hat{\beta}_1) = \hat{\sigma} / \sqrt{S_{xx}}.$$

A $(1-\delta)$ confidence interval for $\alpha = -\beta_1$ is:

$$\hat{\alpha} \pm t_{n-2,\, 1-\delta/2} \cdot \hat{\sigma} / \sqrt{S_{xx}},$$

where $t_{n-2,\, 1-\delta/2}$ is the $(1-\delta/2)$ quantile of the Student-$t$ distribution with $n-2$ degrees of freedom.

**Required assumption.** The confidence interval is valid under the assumption that the residuals $\varepsilon_i = y_i - (\beta_0 + \beta_1 x_i)$ are i.i.d. normal with mean zero and constant variance $\sigma^2$ (homoscedastic Gaussian noise in log-space). In practice this means we assume the power-law model $\ell = A/N^\alpha$ is exactly correct up to multiplicative log-normal noise — i.e., $\ell_i = A/N_i^\alpha \cdot \exp(\varepsilon_i)$ with $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$. Violations (e.g., heteroscedasticity, model misspecification, correlation between runs from shared compute) will invalidate the nominal coverage of the interval.

**Practical caveat.** If $E$ is not known precisely, subtracting a misestimated $\hat{E}$ will bias $\hat{\alpha}$. A common approach is to jointly fit $E$, $A$, $\alpha$ via nonlinear least squares (see Sketch 3).

---

### Sketch 3 — Parametric Fit via L-BFGS (Chinchilla Approach 2)

**Objective.** Fit the 5-parameter model $L(N, D) = E + A/N^\alpha + B/D^\beta$ to a dataset of observations $\{(N_i, D_i, L_i)\}_{i=1}^M$ by minimizing the Huber loss (robust to outliers) in linear loss space:

$$\mathcal{J}(e, a, b, \alpha, \beta) = \sum_{i=1}^M \text{Huber}_\delta\!\left(L_i - \exp(e) - \exp(a) \cdot N_i^{-\alpha} - \exp(b) \cdot D_i^{-\beta}\right),$$

where the reparametrization uses $e = \log E$, $a = \log A$, $b = \log B$ to enforce positivity constraints $E, A, B > 0$.

#### Part (a): Why fit in log-space (reparametrize $E \to e = \log E$, etc.)

Fitting $E$, $A$, $B$ directly in linear space is problematic for two reasons:

1. **Positivity.** The parameters $E$, $A$, $B$ are physically required to be positive. An unconstrained optimizer can produce negative values, making the loss undefined (can't take the log of a negative number in subsequent steps) or physically meaningless. Reparametrizing as $E = \exp(e)$ enforces $E > 0$ automatically without constrained optimization.

2. **Scale.** The parameters can span many orders of magnitude. For example, $E$ might be $\sim 1$ nat while $A$ and $B$ might be $\sim 10^3$ to $10^6$ for standard architectures. Gradient-based optimizers behave poorly when parameters have very different scales. In log-space, the optimizer works with $e$, $a$, $b$ which are all $O(1)$ to $O(10)$, giving roughly equal gradient magnitudes and better-conditioned Hessians.

#### Part (b): Gradient $\nabla_{e,\, a,\, b,\, \alpha,\, \beta}$ of the objective

Let $r_i = L_i - e^e - e^a N_i^{-\alpha} - e^b D_i^{-\beta}$ be the residual for observation $i$, and let $\psi_\delta(r) = d\,\text{Huber}_\delta(r)/dr$ be the Huber pseudo-residual ($\psi_\delta(r) = r$ if $|r| \le \delta$, else $\delta \cdot \text{sign}(r)$). Then:

$$\frac{\partial \mathcal{J}}{\partial e} = -\sum_i \psi_\delta(r_i) \cdot e^e,$$

$$\frac{\partial \mathcal{J}}{\partial a} = -\sum_i \psi_\delta(r_i) \cdot e^a N_i^{-\alpha},$$

$$\frac{\partial \mathcal{J}}{\partial b} = -\sum_i \psi_\delta(r_i) \cdot e^b D_i^{-\beta},$$

$$\frac{\partial \mathcal{J}}{\partial \alpha} = \sum_i \psi_\delta(r_i) \cdot e^a \cdot (\log N_i) \cdot N_i^{-\alpha},$$

$$\frac{\partial \mathcal{J}}{\partial \beta} = \sum_i \psi_\delta(r_i) \cdot e^b \cdot (\log D_i) \cdot D_i^{-\beta}.$$

These can be computed in a single forward pass over the dataset, making L-BFGS efficient here.

#### Part (c): Initialization strategy

Poor initialization leads to slow convergence or convergence to bad local minima. A principled strategy:

```
# Stage 1: Estimate E from the best (largest N, largest D) run in the dataset.
E_init = min(L_i for all i) * 0.9     # slightly below best observed loss
e_init = log(E_init)

# Stage 2: With E fixed, fit the power-law terms separately.
# For alpha: regress log(L_i - E_init) on log(N_i) for runs with large D (data-saturated).
# For beta: regress log(L_i - E_init) on log(D_i) for runs with large N (model-saturated).
alpha_init = -slope from OLS of log(L_i - E_init) vs log(N_i)  [for large-D subset]
beta_init  = -slope from OLS of log(L_i - E_init) vs log(D_i)  [for large-N subset]

# Stage 3: Given alpha_init, beta_init, E_init, fit A and B by OLS in log-space.
# Solve: log(L_i - E_init) ≈ log A - alpha * log N_i (approximately, for data-saturated runs)
A_init = exp(mean(log(L_i - E_init) + alpha_init * log(N_i)))
B_init = exp(mean(log(L_i - E_init) + beta_init  * log(D_i)))
a_init = log(A_init)
b_init = log(B_init)

# Initial parameter vector for L-BFGS:
theta_0 = [e_init, a_init, b_init, alpha_init, beta_init]
```

#### Part (d): Detecting and handling local minima

The objective $\mathcal{J}$ is non-convex in $(\alpha, \beta)$ (though convex in $(e, a, b)$ for fixed $\alpha, \beta$), so L-BFGS may converge to local minima.

```
# Multi-start strategy:
n_restarts = 50
best_loss = inf
best_theta = None

for restart in range(n_restarts):
    # Perturb initialization with random noise in log-space
    theta_init = theta_0 + random_normal(scale=0.5, size=5)
    # Constrain alpha, beta to physically reasonable range [0.01, 2.0]
    theta_init[3] = clip(theta_init[3], log(0.01), log(2.0))  # if fitting log(alpha)
    theta_init[4] = clip(theta_init[4], log(0.01), log(2.0))

    theta_hat, loss_val = lbfgs_minimize(objective=J, grad=grad_J, x0=theta_init,
                                          max_iter=10000, gtol=1e-8)

    if loss_val < best_loss:
        best_loss = loss_val
        best_theta = theta_hat

# Post-hoc validation:
# 1. Inspect the loss surface in (alpha, beta) by grid search around best_theta.
# 2. Compare fitted vs. observed losses; residuals should be symmetric and small.
# 3. Check that E_hat < min(L_i): if not, E is overfit and should be constrained.
# 4. Run on held-out (N, D) pairs not used in fitting; large held-out error indicates
#    overfitting or a bad local minimum.
```

The Huber loss (with $\delta \approx 0.001$ nats) is preferable to squared error here because a few outlier runs (e.g., diverged training runs, incorrectly logged losses) can dominate the squared-error objective and badly skew the fitted exponents. Huber loss down-weights large residuals, making the fit robust to such anomalies.
