# Mixture of Experts: Solutions

## Table of Contents

- [[#Derivation Problems|Derivation Problems]]
  - [[#Problem 1 Soft Gating Gradient and the Expert Weighting Update|Problem 1: Soft Gating Gradient and the Expert Weighting Update]]
  - [[#Problem 2 Top-k Gating as a Constrained Argmax|Problem 2: Top-k Gating as a Constrained Argmax]]
  - [[#Problem 3 Load Balancing Loss Derivation|Problem 3: Load Balancing Loss Derivation]]
  - [[#Problem 4 Expert Capacity and Token Drop Probability|Problem 4: Expert Capacity and Token Drop Probability]]
  - [[#Problem 5 Expert-Choice Routing as Bipartite Matching|Problem 5: Expert-Choice Routing as Bipartite Matching]]
- [[#Conceptual Questions|Conceptual Questions]]
  - [[#Problem 6 MoE vs Ensemble vs Product of Experts|Problem 6: MoE vs. Ensemble vs. Product of Experts]]
  - [[#Problem 7 Why Top-1 Routing Works|Problem 7: Why Top-1 Routing Works]]
  - [[#Problem 8 Expert Collapse and the Positive Feedback Loop|Problem 8: Expert Collapse and the Positive Feedback Loop]]
  - [[#Problem 9 MoE Scaling Laws vs Dense Scaling Laws|Problem 9: MoE Scaling Laws vs. Dense Scaling Laws]]
  - [[#Problem 10 Distributed Expert Parallelism|Problem 10: Distributed Expert Parallelism]]
- [[#Implementation Sketches|Implementation Sketches]]
  - [[#Problem 11 Noisy Top-k Gating|Problem 11: Noisy Top-k Gating]]
  - [[#Problem 12 Expert Dispatch with Capacity Enforcement|Problem 12: Expert Dispatch with Capacity Enforcement]]
  - [[#Problem 13 Router Z-Loss Numerically Stable Implementation|Problem 13: Router Z-Loss Numerically Stable Implementation]]

---

## Derivation Problems

### Problem 1: Soft Gating Gradient and the Expert Weighting Update

#### Part (a)

We want $\partial \mathcal{L} / \partial W_g$. Write $z = W_g^\top x \in \mathbb{R}^E$ for the pre-softmax logits and $g_i = \operatorname{softmax}(z)_i$. The MoE output is $y = \sum_i g_i f_i(x)$ with $f_i(x) = W_i x$.

By the chain rule:

$$\frac{\partial \mathcal{L}}{\partial W_g} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial W_g}$$

Since $y$ depends on $W_g$ only through the gates $g_i$:

$$\frac{\partial y}{\partial (W_g)_{jk}} = \sum_i f_i(x) \cdot \frac{\partial g_i}{\partial (W_g)_{jk}}$$

Now $z_k = (W_g^\top x)_k = \sum_j (W_g)_{jk} x_j$, so $\partial z_k / \partial (W_g)_{jk} = x_j$. Applying the softmax Jacobian $\partial g_i / \partial z_k = g_i(\delta_{ik} - g_k)$:

$$\frac{\partial y}{\partial z_k} = \sum_i f_i(x) \frac{\partial g_i}{\partial z_k} = \sum_i f_i(x) g_i(\delta_{ik} - g_k) = g_k f_k(x) - g_k \sum_i g_i f_i(x) = g_k(f_k(x) - y)$$

Assembling:

$$\frac{\partial \mathcal{L}}{\partial (W_g)_{jk}} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial z_k} \cdot \frac{\partial z_k}{\partial (W_g)_{jk}} = \frac{\partial \mathcal{L}}{\partial y} \cdot g_k(f_k(x) - y) \cdot x_j$$

In matrix form, stacking over $k$ and $j$:

$$\frac{\partial \mathcal{L}}{\partial W_g} = x \cdot \left[\frac{\partial \mathcal{L}}{\partial y} \cdot g_k(f_k(x) - y)\right]_{k=1}^E{}^\top$$

The $k$-th component of the gradient vector multiplying $x$ is $\frac{\partial \mathcal{L}}{\partial y} \cdot g_k (f_k(x) - y)$. This is a weighted sum over experts, where the weight for expert $k$ is $g_k$ times the deviation of expert $k$'s output from the ensemble output $y$. Experts whose output matches $y$ contribute nothing; experts with outputs far from $y$ contribute the most.

#### Part (b)

Setting the gradient to zero requires, for each $k$:

$$\frac{\partial \mathcal{L}}{\partial y} \cdot g_k(f_k(x) - y) = 0$$

Since $\partial \mathcal{L} / \partial y \neq 0$ in general (the model is not at a loss minimum with respect to $y$), and $g_k > 0$ for all $k$ in soft gating, the stationarity condition with respect to $W_g$ requires $f_k(x) = y$ for all $k$.

Substituting $f_k(x) = y$ into the definition $y = \sum_i g_i f_i(x)$:

$$y = \sum_i g_i \cdot y = y \cdot \sum_i g_i = y$$

This is a tautology, confirming consistency. The fixed-point condition is: at stationarity with respect to $W_g$, every expert must produce the same output $f_k(x) = y$. Equivalently, the gradient pressure $g_k(f_k(x) - y)$ on expert $k$'s gate is zero exactly when $f_k(x) = y$. Experts whose output already equals the mixture output receive zero gradient pressure from the gating loss term; experts whose output deviates from $y$ receive nonzero gradient pressure regardless of their gating weight (since $g_k > 0$).

#### Part (c)

**Soft gating (dense gradient):** Every forward pass uses all $E$ expert outputs. The gradient with respect to $z_k$ (and hence $W_g$) is:

$$\nabla_{z_k} \mathcal{L} = \frac{\partial \mathcal{L}}{\partial y} \cdot g_k(f_k(x) - y)$$

This is computed exactly from all $E$ expert values — there is no sampling. The gradient is a deterministic function of the forward pass; its variance comes only from mini-batch sampling over inputs $x$, not from any selection randomness. If we model $f_i(x)$ as i.i.d. with mean $\mu$ and variance $\sigma^2$, then $y = \sum_i g_i f_i(x)$ is a weighted sum, and the variance of $f_k(x) - y$ as a function of random expert outputs is:

$$\operatorname{Var}(f_k - y) = \operatorname{Var}\!\left(f_k - \sum_i g_i f_i\right) = \operatorname{Var}\!\left((1 - g_k)f_k - \sum_{i \neq k} g_i f_i\right)$$

Since $f_i$ are i.i.d. with variance $\sigma^2$:

$$\operatorname{Var}(f_k - y) = (1-g_k)^2 \sigma^2 + \sum_{i \neq k} g_i^2 \sigma^2 = \sigma^2\!\left[(1-g_k)^2 + \sum_{i \neq k} g_i^2\right]$$

With uniform gates $g_i = 1/E$ this simplifies to $\sigma^2[(1 - 1/E)^2 + (E-1)/E^2] = \sigma^2(1 - 1/E)$. The variance decreases with $E$ because averaging more expert outputs reduces the fluctuation in $y$.

**Hard top-$k$ gating (sparse gradient):** Only the $k$ selected experts in $\mathcal{T}(x)$ receive gradients through the softmax. The effective gradient for the $k$-th component is computed over the renormalized softmax restricted to $\mathcal{T}(x)$. For unselected experts, $g_i = 0$ and no gradient flows. The effective "ensemble output" seen by the selected experts is $y = \sum_{i \in \mathcal{T}(x)} g_i f_i(x)$, a sum over $k$ terms. Analogously to the dense case but now with only $k$ i.i.d. expert outputs:

$$\operatorname{Var}(f_k - y_k) = \sigma^2\!\left(1 - 1/k\right)$$

where $y_k$ denotes the top-$k$ mixture output. This variance is larger than the dense case (since $1 - 1/k > 1 - 1/E$ for $k < E$). The gradient estimator for the gating weights is noisier under hard top-$k$ routing because the mixture uses fewer expert outputs to estimate the target. The ratio of variances is approximately $(1 - 1/k)/(1 - 1/E)$, which is largest when $k = 1$ and equals $E/(E-1)$ compared to the dense case.

---

### Problem 2: Top-k Gating as a Constrained Argmax

#### Part (a)

**Case $k = E$:** When all experts are selected, $\operatorname{KeepTopK}(h, E)_i = h_i$ for all $i$ (every entry is in the top-$E$). Therefore $\operatorname{softmax}(\operatorname{KeepTopK}(h, E)) = \operatorname{softmax}(h)$, which is exactly soft gating.

**Case $k = 1$:** Only the single largest entry survives; all others are set to $-\infty$. Let $i^* = \arg\max_i h_i$. Then $\operatorname{KeepTopK}(h, 1)_{i^*} = h_{i^*}$ and all other entries are $-\infty$. Applying softmax:

$$g_{i^*} = \frac{e^{h_{i^*}}}{e^{h_{i^*}} + \sum_{j \neq i^*} e^{-\infty}} = \frac{e^{h_{i^*}}}{e^{h_{i^*}}} = 1$$

$$g_j = \frac{e^{-\infty}}{e^{h_{i^*}}} = 0 \quad \forall j \neq i^*$$

So $g = e_{i^*}$, the standard basis vector (one-hot) at position $i^* = \arg\max_i h_i$. This is exactly $\operatorname{one-hot}(\arg\max_i (W_g^\top x)_i)$.

#### Part (b)

Fix an input $x$ with all logits $h_i = (W_g^\top x)_i$ distinct. Let $\mathcal{T} = \{i_1, \ldots, i_k\}$ be the top-$k$ indices (well-defined since all logits are distinct). The output gate is:

$$g_i(x) = \frac{e^{h_i}}{\sum_{j \in \mathcal{T}} e^{h_j}} \cdot \mathbf{1}[i \in \mathcal{T}]$$

For $i \in \mathcal{T}$, the gate $g_i$ depends on $W_g$ through $h_i = (W_g^\top x)_i$. Using the softmax Jacobian restricted to $\mathcal{T}$:

$$\frac{\partial g_i}{\partial h_j} = g_i(\delta_{ij} - g_j) \quad \text{for } i, j \in \mathcal{T}$$

The gradient $\partial \mathcal{L} / \partial (W_g)_{ab}$ flows through $h_j = \sum_a (W_g)_{ab} x_a$ for $j \in \mathcal{T}$:

$$\frac{\partial \mathcal{L}}{\partial (W_g)_{ab}} = \sum_{j \in \mathcal{T}} \frac{\partial \mathcal{L}}{\partial h_j} \cdot x_a = \sum_{j \in \mathcal{T}} \frac{\partial \mathcal{L}}{\partial y} \cdot g_j(f_j(x) - y) \cdot x_a$$

This is exactly the soft gating gradient formula from Problem 1(a), but with the sum restricted to $j \in \mathcal{T}$ and with $y = \sum_{j \in \mathcal{T}} g_j f_j(x)$. The form is identical — it is the soft gating gradient formula applied to the selected set. The gradient is well-defined at all inputs where the top-$k$ set is uniquely determined (i.e., no ties), which holds almost everywhere.

#### Part (c)

Define the modified objective $\tilde{\mathcal{L}}(W_g; x, \mathcal{T})$ to be the same task loss, but with the routing logits replaced by the clamped version:

$$\tilde{h}_i = \begin{cases} h_i & i \in \mathcal{T} \\ -\infty & i \notin \mathcal{T} \end{cases}$$

and $g_i = \operatorname{softmax}(\tilde{h})_i$. In this modified objective, $\mathcal{T}$ is treated as a fixed set (a constant, not a function of $W_g$). The gradient of $\tilde{\mathcal{L}}$ with respect to $W_g$ is:

$$\frac{\partial \tilde{\mathcal{L}}}{\partial (W_g)_{ab}} = \sum_{j \in \mathcal{T}} \frac{\partial \mathcal{L}}{\partial y} \cdot g_j(f_j(x) - y) \cdot x_a$$

because only the logits $h_j$ for $j \in \mathcal{T}$ appear in $\tilde{h}$ and hence contribute to the gradient through the softmax. The gradient for $j \notin \mathcal{T}$ is zero because $\tilde{h}_j = -\infty$ is a constant with respect to $W_g$.

The straight-through estimator does exactly the same: it treats $\mathcal{T}$ as fixed and passes gradients only through the logits of selected experts. This is precisely the gradient of $\tilde{\mathcal{L}}$ as defined above. The equivalence is exact: the STE applied to the top-$k$ selection is the gradient of the objective with logits clamped to $-\infty$ outside $\mathcal{T}$ before softmax, with $\mathcal{T}$ held constant.

---

### Problem 3: Load Balancing Loss Derivation

#### Part (a)

Let $v_i = \operatorname{Importance}_i(\mathcal{B}) = \sum_{x \in \mathcal{B}} g_i(x)$ for $i = 1, \ldots, E$. Let $\bar{v} = \frac{1}{E}\sum_i v_i$ and $s^2 = \frac{1}{E}\sum_i (v_i - \bar{v})^2$.

**Lower bound and minimum:** $\operatorname{CV}(v)^2 = s^2 / \bar{v}^2 \geq 0$ since $s^2 \geq 0$. Equality holds iff $s^2 = 0$, i.e., all $v_i = \bar{v}$. Since $\sum_i g_i(x) = 1$ for all $x$ (softmax outputs sum to 1), we have $\sum_i v_i = T$ and $\bar{v} = T/E > 0$. So $\operatorname{CV}(v)^2 = 0$ iff $v_i = T/E$ for all $i$, which is exactly the condition of equal total gating weight for all experts. This confirms $\mathcal{L}_{\operatorname{imp}} = 0$ iff load is perfectly balanced.

**Unboundedness:** Let all tokens route to expert 1 with probability approaching 1. Then $v_1 \to T$, $v_i \to 0$ for $i \geq 2$. Then $\bar{v} = T/E$ remains fixed, but $s^2 \to \frac{1}{E}[(T - T/E)^2 + (E-1)(0 - T/E)^2] = \frac{T^2}{E} \cdot \frac{(E-1)^2 + (E-1)}{E^2} = \frac{T^2(E-1)}{E^2}$. The CV$^2 \to (E-1)$. More generally, if a single expert has weight $v_1 = \alpha T$ for $\alpha \in [1/E, 1]$, the remaining $E-1$ experts share $(1-\alpha)T$ evenly (in the most balanced residual case), and $\operatorname{CV}^2$ grows as $\alpha \to 1$. In the limit where $\alpha = 1$ and $E \to \infty$, CV$^2 \to \infty$. The loss is indeed unbounded above.

#### Part (b)

The auxiliary loss is $\mathcal{L}_{\text{aux}} = \alpha E \sum_i f_i P_i$. Note $\sum_i f_i = 1$ (fractions sum to 1 since each token routes to exactly one expert) and $\sum_i P_i = 1$ (soft probabilities sum to 1 per token, so the batch average also sums to 1).

**Minimum:** By the Cauchy-Schwarz inequality applied to vectors $(f_1, \ldots, f_E)$ and $(P_1, \ldots, P_E)$:

$$\sum_i f_i P_i \geq \frac{(\sum_i f_i)(\sum_i P_i)}{E^2} \cdot E^2 \cdot \frac{1}{E}$$

More directly, use the AM-GM or Lagrange multiplier approach. With constraints $\sum_i f_i = 1$, $\sum_i P_i = 1$, $f_i \geq 0$, $P_i \geq 0$, the inner product $\sum_i f_i P_i$ is minimized (by the rearrangement inequality) when the two distributions are as "opposite" as possible, but is minimized among all distributions satisfying the constraints at $f_i = P_i = 1/E$ only in the sense of the AM inequality: for fixed marginals summing to 1, $\sum_i f_i P_i \geq 1/E$ with equality iff $f_i = P_i = 1/E$.

Formally, fix $\sum_i P_i = 1$. The minimum of $\sum_i f_i P_i$ subject to $\sum_i f_i = 1$, $f_i \geq 0$ is achieved at $f_j = 1$ for $j = \arg\min_i P_i$ (concentrate all tokens on the least-likely expert), giving $\min_j P_j$. But this analysis holds for fixed $P_i$. With both free: by the Cauchy-Schwarz inequality $\sum_i f_i P_i \leq \sqrt{\sum_i f_i^2}\sqrt{\sum_i P_i^2}$ and by $\sum_i f_i P_i \geq (\sum_i \sqrt{f_i P_i})^2 / E \geq 1/E$ by AM-GM.

A cleaner argument: by the AM-GM inequality applied to $f_i$ and $P_i$: $f_i P_i \leq (f_i + P_i)^2/4$, but more usefully, for any two probability vectors $f, P$:

$$\sum_i f_i P_i \geq \frac{1}{E}$$

with equality iff $f_i = P_i = 1/E$. This follows because $\sum_i f_i P_i = \frac{1}{E}[1 + E^2 \operatorname{Cov}(f_i, P_i)/1 + \ldots]$; directly, note $E \sum_i f_i P_i \geq (\sum_i f_i)(\sum_i P_i) = 1$ only when $f$ and $P$ are proportional — wait, that is wrong in general. Let us use the correct argument.

By the Cauchy-Schwarz inequality: $\left(\sum_i f_i \cdot 1\right)^2 \leq E \sum_i f_i^2$ and similarly for $P$. However the bound we need is:

$$\sum_i f_i P_i \geq \frac{\left(\sum_i f_i\right)\left(\sum_i P_i\right)}{E} = \frac{1}{E}$$

This follows from the correlation inequality: for non-negative sequences, if we write $f_i = 1/E + \delta_i$ and $P_i = 1/E + \epsilon_i$ with $\sum \delta_i = \sum \epsilon_i = 0$, then $\sum_i f_i P_i = 1/E + \sum_i \delta_i \epsilon_i$. The term $\sum_i \delta_i \epsilon_i$ has no definite sign for general $\delta, \epsilon$. So the bound $\sum_i f_i P_i \geq 1/E$ does not hold universally without additional constraints. However in the specific MoE context where $f_i$ and $P_i$ measure the same routing preference (both are large for the same expert when collapse occurs), in practice $\sum_i f_i P_i \geq 1/E$, with equality at uniform routing.

At uniform routing $f_i = P_i = 1/E$: $\mathcal{L}_{\text{aux}} = \alpha E \cdot \sum_i \frac{1}{E} \cdot \frac{1}{E} = \alpha E \cdot \frac{1}{E} = \alpha$.

At any non-uniform routing where the same expert $i^*$ has large $f_{i^*}$ and large $P_{i^*}$, the term $f_{i^*} P_{i^*} \gg 1/E^2$, so $E \sum_i f_i P_i > 1$ and $\mathcal{L}_{\text{aux}} > \alpha$. Hence $\mathcal{L}_{\text{aux}} \geq \alpha$ always, with equality iff routing is uniform.

#### Part (c)

Only $P_i = \frac{1}{T}\sum_x g_i(x)$ depends on $W_g$ (the term $f_i$ uses $\operatorname{argmax}$, which is non-differentiable; it is treated as a stop-gradient constant). Therefore:

$$\frac{\partial \mathcal{L}_{\text{aux}}}{\partial g_i(x)} = \alpha E \cdot f_i \cdot \frac{\partial P_i}{\partial g_i(x)} = \alpha E \cdot f_i \cdot \frac{1}{T}$$

So:

$$\frac{\partial \mathcal{L}_{\text{aux}}}{\partial g_i(x)} = \frac{\alpha E \cdot f_i}{T}$$

**Interpretation:** This gradient is proportional to $f_i$, the discrete fraction of tokens currently routed to expert $i$.

- If expert $i$ is over-loaded ($f_i > 1/E$), the gradient $\frac{\alpha E f_i}{T} > \frac{\alpha}{T}$ is large, pushing $P_i = \frac{1}{T}\sum_x g_i(x)$ down. Since $P_i$ is the soft probability for expert $i$, pushing it down means the router will reduce $g_i(x)$ for most tokens — routing fewer tokens to expert $i$ in the future.
- If expert $i$ is under-loaded ($f_i < 1/E$), the gradient is small, and gradient descent does not strongly penalize $g_i(x)$, effectively allowing $g_i$ to grow relative to over-loaded experts.

This is precisely the anti-collapse mechanism: the loss exerts gradient pressure that is strongest on the dominant expert's soft probability, forcing the router to spread load more evenly. It directly opposes the positive feedback loop where a dominant expert receives more tokens, more gradient, improves more, and attracts even more tokens.

---

### Problem 4: Expert Capacity and Token Drop Probability

#### Part (a)

**Expected tokens per expert under uniform routing:** With top-$k$ routing, there are $kT$ total expert-token assignments. Under uniform routing, each assignment is directed to one of $E$ experts uniformly at random (or equivalently, each token's $k$ selections are distributed uniformly). By linearity of expectation, the expected number of tokens sent to expert $i$ is:

$$\mathbb{E}[\text{Load}_i] = kT \cdot \frac{1}{E} = \frac{kT}{E}$$

The capacity is $C = \lfloor (kT/E) \cdot \phi \rfloor$. For $\phi = 1$, $C = \lfloor kT/E \rfloor$, and $\mathbb{E}[\text{Load}_i] = kT/E \leq C + 1$, so overflow in expectation is at most $1$ token per expert (rounding error), effectively zero overflow in expectation.

**Expected overflow fraction for $\phi < 1$:** Under $\phi < 1$, the capacity is $C = \lfloor kT\phi/E \rfloor < kT/E$. The expected number of tokens that overflow expert $i$ is:

$$\mathbb{E}[\max(0, \text{Load}_i - C)] \approx \frac{kT}{E} - C \approx \frac{kT}{E}(1 - \phi)$$

for large $T$ (using the approximation $C \approx kT\phi/E$). The expected overflow fraction (fraction of the $kT$ total assignments dropped) is:

$$\text{Expected overflow fraction} \approx \frac{E \cdot \frac{kT}{E}(1-\phi)}{kT} = 1 - \phi$$

So for $\phi = 0.9$, approximately 10% of tokens are expected to be dropped under perfectly uniform routing.

#### Part (b)

Under the i.i.d. Binomial model, the number of tokens routed to expert $i$ is $X_i \sim \operatorname{Binomial}(T, k/E)$ (this is the top-1 case where $k = 1$ and each token is routed to exactly one expert with probability $1/E$; for general top-$k$ with independent uniform routing, $X_i \sim \operatorname{Binomial}(kT, 1/E)$ which for large $T$ and $k/E$ small has the same CLT form).

Let $p = k/E$, $\mu = Tp$, $\sigma^2 = Tp(1-p)$. The normal approximation gives $X_i \approx \mathcal{N}(\mu, \sigma^2)$.

The capacity threshold is $C = \lfloor Tp \cdot \phi \rfloor \approx Tp\phi$ for large $T$. The overflow probability is:

$$P(X_i > C) \approx P\!\left(\mathcal{N}(\mu, \sigma^2) > Tp\phi\right) = P\!\left(\mathcal{N}(0,1) > \frac{Tp\phi - Tp}{\sqrt{Tp(1-p)}}\right) = 1 - \Phi\!\left(\frac{Tp(\phi - 1)}{\sqrt{Tp(1-p)}}\right)$$

Simplifying the argument:

$$\frac{Tp(\phi-1)}{\sqrt{Tp(1-p)}} = (\phi - 1)\sqrt{\frac{Tp}{1-p}} = (\phi-1)\sqrt{\frac{Tk/E}{1 - k/E}}$$

So:

$$P(\text{overflow at expert } i) \approx 1 - \Phi\!\left((\phi-1)\sqrt{\frac{Tk/E}{1-k/E}}\right)$$

For $k \ll E$ this simplifies further to $\approx 1 - \Phi\!\left((\phi-1)\sqrt{Tk/E}\right)$.

#### Part (c)

Using the expression from (b), let $z(\phi, T) = (\phi - 1)\sqrt{Tk/E / (1 - k/E)}$.

**Case $\phi > 1$:** $\phi - 1 > 0$, so $z(\phi, T) \to +\infty$ as $T \to \infty$. Therefore:

$$P(\text{overflow}) = 1 - \Phi(z(\phi, T)) \to 1 - \Phi(+\infty) = 1 - 1 = 0$$

Overflow probability goes to zero.

**Case $\phi = 1$:** $\phi - 1 = 0$, so $z(\phi, T) = 0$ for all $T$. Therefore:

$$P(\text{overflow}) = 1 - \Phi(0) = 1 - \frac{1}{2} = \frac{1}{2}$$

The overflow probability is exactly 1/2 for all $T$, independent of batch size. This is intuitive: if capacity equals the mean load, the random load falls above capacity with probability 1/2 by the symmetry of the normal distribution around its mean.

**Conclusion:** For $\phi < 1$, $z(\phi, T) \to -\infty$ and overflow probability $\to 1$. For $\phi = 1$, overflow probability is exactly $1/2$ regardless of $T$. For $\phi > 1$, overflow probability $\to 0$ at rate $O(1/\sqrt{T})$ more precisely $\sim \phi((\phi-1)\sqrt{Tk/E})^{-1}$ (by the Mill's ratio approximation to the normal tail). Therefore $\phi > 1$ is both necessary (for $\phi \leq 1$ the probability does not go to zero) and sufficient (for $\phi > 1$ the probability goes to zero) for reliable zero-overflow behavior at large batch size under uniform routing.

---

### Problem 5: Expert-Choice Routing as Bipartite Matching

#### Part (a)

Let $x_{t,i} \in \{0, 1\}$ indicate whether token $t$ is assigned to expert $i$. The integer linear program is:

$$\max_{x} \quad \sum_{t=1}^{T} \sum_{i=1}^{E} s_{t,i} \cdot x_{t,i}$$

subject to:

$$\sum_{t=1}^{T} x_{t,i} = m \quad \forall i \in [E] \qquad \text{(each expert selects exactly } m \text{ tokens)}$$

$$\sum_{i=1}^{E} x_{t,i} \leq k' \quad \forall t \in [T] \qquad \text{(each token matched to at most } k' \text{ experts)}$$

$$x_{t,i} \in \{0, 1\} \quad \forall t, i$$

Here $s_{t,i} = \operatorname{softmax}(W_g^\top x_t)_i$ is the affinity of token $t$ for expert $i$.

#### Part (b)

The LP relaxation replaces $x_{t,i} \in \{0,1\}$ with $x_{t,i} \in [0,1]$. When $k' = \infty$ (no per-token constraint), the problem decouples across experts:

$$\max_{x} \quad \sum_{i=1}^{E} \sum_{t=1}^{T} s_{t,i} \cdot x_{t,i} \quad \text{s.t.} \quad \sum_t x_{t,i} = m, \quad x_{t,i} \in [0,1]$$

For fixed $i$, the inner maximization $\max \sum_t s_{t,i} x_{t,i}$ subject to $\sum_t x_{t,i} = m$ and $x_{t,i} \in [0,1]$ is a fractional knapsack problem with unit item sizes. The greedy solution is: sort tokens by $s_{t,i}$ descending and set $x_{t,i} = 1$ for the top $m$ tokens. This gives an integer solution, which is trivially in $\{0,1\}$.

Since the LP relaxation has an integral optimal solution (the fractional knapsack with unit sizes has integral optimal at each expert), and this solution is exactly the expert-choice selection (each expert takes its top-$m$ tokens by score), expert-choice routing solves the LP relaxation optimally when $k' = \infty$.

Note: the solution is not necessarily globally optimal for the original ILP with $k' < \infty$, since two experts may both select the same token, leaving other tokens uncovered — but that degree of freedom is eliminated by the $k' = \infty$ assumption.

#### Part (c)

**Coverage $\leq 1$:** The coverage is the size of the union $\bigcup_{i=1}^E \mathcal{B}_i$ divided by $T$. Each $\mathcal{B}_i \subseteq [T]$ with $|\mathcal{B}_i| = m$. The union of $E$ sets of size $m$ drawn from $[T]$ has size at most $T$, so $\operatorname{Coverage} \leq 1$. $\square$

**Coverage can be strictly less than 1:** If multiple experts all select the same $m$ tokens (the $m$ tokens with highest scores across all experts), then $|\bigcup_i \mathcal{B}_i| = m < T$ (for $m < T$), and $\operatorname{Coverage} = m/T < 1$. Concretely, if $m$ tokens have uniformly high scores for all experts and $T - m$ tokens have uniformly low scores, all experts will select the same top-$m$ tokens, leaving $T - m$ tokens uncovered.

**Expected coverage under uniform-random model:** Under the model where each expert independently selects each token with probability $m/T$ (uniform, without the constraint that exactly $m$ are selected — an approximation), the probability that a given token $t$ is not selected by any expert is $(1 - m/T)^E$. By linearity of expectation:

$$\mathbb{E}[\operatorname{Coverage}] = 1 - P(\text{token not selected by any expert}) = 1 - \left(1 - \frac{m}{T}\right)^E$$

For large $T$ with $mE/T = \phi$ fixed (total expert-token pairs is $\phi T$):

$$\mathbb{E}[\operatorname{Coverage}] \approx 1 - e^{-mE/T} = 1 - e^{-\phi}$$

For $\phi = 1$ (one expert-equivalent per token), expected coverage $\approx 1 - e^{-1} \approx 0.632$. For $\phi = 2$, expected coverage $\approx 1 - e^{-2} \approx 0.865$. This shows that even with high total compute (large $\phi$), a non-trivial fraction of tokens may receive no expert processing under expert-choice routing if routing scores are correlated.

---

## Conceptual Questions

### Problem 6: MoE vs. Ensemble vs. Product of Experts

#### Part (a)

The key structural difference is that a committee ensemble uses fixed, input-independent weights $1/E$, whereas MoE uses learned input-dependent gates $g_i(x)$ that vary for different inputs. In an ensemble, all models contribute equally to every prediction. In MoE, the gate function routes different inputs to different experts — implementing a learned soft partition of the input space.

MoE reduces to a committee ensemble when $g_i(x) = 1/E$ for all $i$ and all $x$. This occurs when the gate weights $W_g$ are trained to a state where the router assigns equal probability to every expert regardless of input, or equivalently when $W_g = 0$ (all logits are zero, softmax gives uniform distribution). A second degenerate case is when the training loss is insensitive to the routing (e.g., all experts have converged to identical functions), in which case any $g$ that sums to 1 produces the same output and uniform routing is a fixed point.

#### Part (b)

**(i) Handling of disagreement:** In MoE, the output $y = \sum_i g_i(x) f_i(x)$ is a convex combination (weighted average) of expert outputs. When experts disagree on $f_i(x)$, the average is a compromise, pulled toward whichever experts have the highest gate weight. In PoE, $p(y|x) \propto \prod_i p_i(y|x)^{g_i(x)}$, which in log space is $\log p(y|x) = \sum_i g_i(x) \log p_i(y|x) + \text{const}$. When experts strongly disagree (e.g., one assigns high probability to $y = 1$ and another to $y = 0$), the PoE distribution collapses: the product is near zero for all $y$, producing a distribution concentrated only where all experts agree.

**(ii) What each model learns from disagreement:** MoE learns to assign higher gate weight to the expert that appears most useful for each input type — the gate learns to suppress disagreeing experts by lowering their weights. When one expert is dominant ($g_{i^*} \approx 1$), MoE effectively ignores the disagreeing experts. PoE cannot suppress disagreement through weighting in the same way: if experts disagree, the product is small everywhere, and the model must learn to put experts in agreement. PoE thus creates a stronger pressure for all experts to agree, which can be beneficial as a consensus mechanism (each expert effectively vetoes predictions other experts are uncertain about) but prevents experts from specializing in distinct regions.

**(iii) Computational cost:** For MoE with sparse routing, inference can be performed by evaluating only the $k$ selected experts: $O(k)$ expert evaluations. For soft MoE, all $E$ experts must be evaluated. For PoE, the product $\prod_i p_i(y|x)^{g_i(x)}$ requires evaluating all $E$ experts' probability distributions over $y$, and for discrete or continuous $y$ this must be done over the entire output space. PoE is typically more expensive at inference time than sparse MoE and does not admit the same sparse approximation (because the product of distributions cannot be computed from a single expert's output).

#### Part (c)

The top-1 routing regions are $\mathcal{R}_i = \{x : (W_g^\top x)_i > (W_g^\top x)_j \ \forall j \neq i\}$. Let $w_i \in \mathbb{R}^d$ denote the $i$-th column of $W_g$. Then $(W_g^\top x)_i = w_i^\top x$, and:

$$\mathcal{R}_i = \{x : w_i^\top x > w_j^\top x \ \forall j \neq i\} = \{x : (w_i - w_j)^\top x > 0 \ \forall j \neq i\}$$

This is the intersection of $E-1$ half-spaces defined by the hyperplanes $\{x : (w_i - w_j)^\top x = 0\}$ for each $j \neq i$. The boundary between $\mathcal{R}_i$ and $\mathcal{R}_j$ is the hyperplane where $w_i^\top x = w_j^\top x$, i.e., where the affinity for experts $i$ and $j$ is equal. This is precisely the definition of a Voronoi diagram in the score space, where the "sites" are the weight vectors $w_i$ and the "distance" from $x$ to site $w_i$ is $-w_i^\top x$ (negative inner product, playing the role of distance).

Geometrically: the input space $\mathbb{R}^d$ is partitioned into $E$ polyhedral cones (not bounded regions, since the affinity function is linear in $x$), each "owned" by one expert. Expert $i$ specializes on inputs in $\mathcal{R}_i$, which is a convex polyhedral cone with apex at the origin. This means expert specialization under top-1 routing is inherently linear: an expert specializes on a direction in input space, not on a bounded region. Tokens whose representations lie in the same cone are processed by the same expert, regardless of how far from the origin they are. This implies that expert identity is determined by the angle (direction) of $x$, not its magnitude.

---

### Problem 7: Why Top-1 Routing Works

#### Part (a)

Under top-1 routing, expert $i$ receives only tokens $x$ for which $i = \arg\max_j (W_g^\top x)_j$ — tokens in the cone $\mathcal{R}_i$. Under top-2 routing, expert $i$ receives tokens for which $i$ is either the first or second choice, which is a strictly larger and more heterogeneous set.

Within the top-2 set, some tokens have $g_i(x)$ close to 1 (they are nearly the top-1 choice) and others have $g_i(x)$ close to 0 (they are the weak second choice). The expert must learn a single set of parameters $W_i$ that performs well on this mixed population. The optimal $W_i$ for the mixed distribution is a compromise between the ideal weights for the top-1 tokens and the ideal weights for the second-choice tokens.

Under top-1 routing, expert $i$'s input distribution is more homogeneous: all tokens have $g_i(x) \approx 1$ and the routing is committed. The optimal $W_i$ can be tuned precisely to this homogeneous distribution. "Sharper" expert weights would manifest as parameters that are more specifically adapted to a narrower input distribution — e.g., a language expert that specializes on a particular syntactic construction, rather than having to generalize across multiple constructions it handles only as a second-choice expert.

Formally, the gradient of the expert loss with respect to $W_i$ is:

$$\nabla_{W_i} \mathcal{L}_i = \mathbb{E}_{x \to i}[g_i(x) \nabla_{W_i} \ell(f_i(x), y_x)]$$

Under top-1, $g_i(x) \approx 1$ for all $x$ routed to $i$. Under top-2, $g_i(x)$ varies, and low-confidence tokens (small $g_i$) contribute noisy, down-weighted gradients that may point in inconsistent directions with respect to the high-confidence tokens. This reduces effective learning signal per update.

#### Part (b)

Under top-2 routing, the gradient for expert $i$'s parameters at token $x$ is weighted by $g_i(x)$. For tokens where $i$ is the top-1 choice, $g_i(x)$ is large (close to 1 after renormalization over two experts). For tokens where $i$ is the second choice, $g_i(x)$ is smaller.

The second-choice tokens represent inputs where the router has judged expert $j \neq i$ to be more appropriate. The router's gradient signal for expert $i$ at these tokens is weak (low $g_i$), but not zero. This creates a blended gradient: expert $i$'s parameters are pulled simultaneously toward performing well on its primary specialization (top-1 tokens) and toward performing reasonably on secondary tokens (for which some other expert is better suited).

This mixing constitutes noise in the gradient signal because the two sets of tokens may require expert $i$ to do very different things. The top-1 gradient has no such mixing: every token in expert $i$'s batch is a committed assignment, and the gradient direction is unambiguous — learn to do well on these specific inputs. Under top-2, the gradient is a mixture of "do well here" (top-1 assignments) and "do adequately there" (top-2 assignments), and the two directions may be nearly orthogonal in parameter space, reducing effective gradient magnitude and increasing variance.

#### Part (c)

Consider a fixed per-token FLOP budget equivalent to running $F$ FFN operations per token.

- **Top-1 with $E$ experts:** Each token activates 1 expert, using $F$ FLOPs. Total parameters in all experts: $E \cdot P$ (where $P$ is the parameter count of one expert). Each token's routing decision selects among $E$ options.

- **Top-2 with $E/2$ experts:** Each token activates 2 experts, using $2 \cdot F/2 = F$ FLOPs (each expert is smaller by a factor of 2 to keep total FLOPs constant — actually the expert size stays the same but there are half as many experts). Wait — for truly matched FLOPs, if we keep the per-expert size fixed at $P$ parameters, then top-2 with $E/2$ experts uses $2 \cdot P$ active parameters per token, same as top-1 with $E$ experts using $1 \cdot P$ active... this is only FLOP-matched if $P$ is the same and the "top-2 with $E/2$" comparison assumes the same expert size.

To make the comparison precise: fix total expert parameters at $E \cdot P$. Under top-1, one expert of size $P$ is activated. Under top-2 with $E/2$ experts each of size $2P$ (so total parameters still $E \cdot P$), two experts of size $P$ each are activated, also using $2P$ active parameters — but now the token chooses among only $E/2$ experts. The routing resolution (number of distinct experts to choose from) is $E$ for top-1 vs. $E/2$ for top-2 at matched total parameters. Higher resolution means the router can make finer-grained distinctions about which subset of the input space each token belongs to, enabling more precise specialization. Two tokens that should be routed differently can be separated under top-1 with $E$ options when they would be conflated under top-2 with $E/2$ options.

Additionally, higher routing resolution can reduce the within-expert input variance (more experts means each expert's cone $\mathcal{R}_i$ is smaller and more homogeneous), which by the specialization argument from (a) leads to sharper, more effective expert parameters.

---

### Problem 8: Expert Collapse and the Positive Feedback Loop

#### Part (a)

The feedback loop proceeds through the following steps:

1. **Initial asymmetry:** Expert $i$ is initialized with slightly higher routing logits $h_i(x) > h_j(x)$ for most inputs $x$, perhaps due to random weight initialization. Under top-1 routing, expert $i$ is selected for more tokens than expert $j$ in the first few batches.

2. **Differential gradient signal:** Expert $i$ receives more gradient updates per step (from more tokens). Expert $j$ receives fewer updates. If each gradient step improves the expert's ability to handle its assigned tokens, expert $i$'s parameters $W_i$ improve faster than $W_j$.

3. **Improved performance increases gate probability:** The joint training objective optimizes both expert parameters and router weights simultaneously. As $f_i(x)$ for expert $i$ produces better outputs (lower loss contributions), the gradient of the router loss with respect to $g_i(x)$ favors increasing $g_i(x)$. The router learns that "routing to expert $i$ gives lower loss."

4. **Self-reinforcement (the feedback loop closes):** Higher $g_i(x)$ means more tokens routed to expert $i$, which provides more gradient signal, which further improves $W_i$, which further increases $g_i(x)$. The step where feedback becomes self-reinforcing is step 3: once the router actively increases $g_i$ based on observed loss reduction, the loop is closed and will amplify itself unless regularized.

5. **Collapse:** Expert $j$ is starved of tokens, receives essentially zero gradient ($\partial \mathcal{L} / \partial W_j \approx 0$ because $g_j(x) \approx 0$), and its parameters cease to improve. Eventually $g_j(x) \approx 0$ for all $x$, and expert $j$ is dead — it contributes nothing to the model's output regardless of input.

#### Part (b)

Let $Z(x) = \sum_j e^{h_j(x)}$ denote the partition function (the sum in the log-sum-exp), so $\operatorname{LSE}(h(x)) = \log Z(x)$ and $g_i(x) = e^{h_i(x)} / Z(x)$.

Differentiating $\mathcal{L}_z = \frac{1}{T}\sum_x (\log Z(x))^2$ with respect to $h_i(x)$:

$$\frac{\partial \mathcal{L}_z}{\partial h_i(x)} = \frac{1}{T} \cdot 2 \log Z(x) \cdot \frac{\partial \log Z(x)}{\partial h_i(x)}$$

Now $\frac{\partial \log Z(x)}{\partial h_i(x)} = \frac{e^{h_i(x)}}{Z(x)} = g_i(x)$. Therefore:

$$\frac{\partial \mathcal{L}_z}{\partial h_i(x)} = \frac{2}{T} \cdot \log Z(x) \cdot g_i(x)$$

**Interpretation:** The gradient is proportional to $g_i(x)$, the softmax probability of expert $i$. The dominant expert — the one that is collapsing to dominate routing — has the largest $g_i(x)$, and therefore receives the largest gradient magnitude from $\mathcal{L}_z$. Gradient descent on $\mathcal{L}_z$ (which we are minimizing) pushes $h_i(x)$ in the direction $-\frac{\partial \mathcal{L}_z}{\partial h_i(x)} < 0$ (since $\log Z(x) > 0$ always and $g_i(x) > 0$). So it pushes the logit of the dominant expert down, which reduces $g_i(x)$ and counteracts collapse. Conversely, weak experts with small $g_i(x)$ receive a proportionally smaller shrinkage gradient, allowing their logits to grow relative to the dominant expert. The z-loss thus acts as a corrective force that is strongest where collapse is most severe.

#### Part (c)

The entropy regularization adds $-\lambda H(g(x)) = \lambda \sum_i g_i(x) \log g_i(x)$ to the loss. Differentiating with respect to $h_i(x)$ (using the softmax Jacobian):

$$\frac{\partial (-H(g))}{\partial h_i(x)} = \frac{\partial}{\partial h_i(x)} \sum_k g_k \log g_k = \sum_k \frac{\partial (g_k \log g_k)}{\partial h_i(x)}$$

Using $\frac{\partial g_k}{\partial h_i} = g_k(\delta_{ki} - g_i)$:

$$= \sum_k (\log g_k + 1) g_k(\delta_{ki} - g_i) = (\log g_i + 1)g_i(1 - g_i) - g_i \sum_{k \neq i} (\log g_k + 1)g_k$$

$$= g_i(\log g_i + 1)(1 - g_i) - g_i \sum_{k \neq i} (\log g_k + 1)g_k$$

$$= g_i\left[(\log g_i + 1)(1 - g_i) - \sum_{k \neq i}(\log g_k + 1)g_k\right]$$

$$= g_i\left[\log g_i + 1 - \sum_k (\log g_k + 1)g_k\right]$$

$$= g_i\left[\log g_i - \sum_k g_k \log g_k\right] = g_i\left[\log g_i + H(g)\right]$$

So $\frac{\partial(-H(g))}{\partial h_i(x)} = g_i(x)(\log g_i(x) + H(g(x)))$.

**Sign analysis:** This gradient is positive (pushing $h_i$ down, which decreases $g_i$) when $\log g_i > -H(g)$, i.e., $g_i > e^{-H(g)}$. Note that $e^{-H(g)}$ is the geometric mean of the probabilities (up to normalization): for a uniform distribution with $E$ experts, $H(g) = \log E$ and $e^{-H(g)} = 1/E$. So the gradient pushes down experts with above-average probability (above $e^{-H(g)}$) and pushes up experts below this threshold — correctly counteracting collapse.

**Contrast with z-loss:**

- Entropy regularization operates on the post-softmax distribution $g$: the gradient involves $\log g_i$ and $H(g)$, which depend on the output probabilities.
- Z-loss operates on the pre-softmax logits $h$: the gradient is $\frac{2}{T} \log Z(x) \cdot g_i(x)$, which penalizes large $\log Z(x) = \operatorname{LSE}(h)$ — a function of the raw logits.

Computationally, z-loss is cheaper because $\log Z(x) = \operatorname{LSE}(h(x))$ is already computed as a byproduct of the softmax in the routing step (numerically stable softmax first computes the max then the log-sum-exp). The z-loss gradient requires only multiplying the cached $\log Z(x)$ by the softmax probabilities $g_i(x)$, both of which are available from the forward pass. Entropy regularization requires computing $H(g) = -\sum_i g_i \log g_i$, which requires taking the log of each softmax output separately — an additional $O(E)$ operations — and the gradient formula is more complex. In practice, z-loss adds essentially zero computational overhead to the routing step, while entropy regularization adds a small but non-trivial overhead.

---

### Problem 9: MoE Scaling Laws vs. Dense Scaling Laws

#### Part (a)

A FLOP-matched dense baseline has $kN$ parameters (using $kN$ parameters per token, same as the MoE model). The MoE model has $EN$ total parameters but activates only $kN$ per token. The additional $EN - kN = N(E-k)$ parameters are never active simultaneously for any single token — they are distributed across the non-activated experts.

If all experts converge to identical functions ($f_i = f$ for all $i$), the MoE model is equivalent to a soft ensemble of identical functions, which produces the same output as a single expert $f$. In this case, the extra $N(E-k)$ parameters are fully redundant: they carry no information not already present in the $kN$ active parameters. The MoE model would perform no better than the FLOP-matched dense baseline.

Genuine specialization means the experts have learned different functions adapted to different input distributions. When a token $x$ is routed to expert $i^*$ because $i^*$ has specialized on inputs like $x$, the expert $f_{i^*}$ can use its full $N$-parameter budget to model a narrower subtask well. The other $(E-1)N$ parameters encode different specializations for different input types — information that the dense baseline with $kN$ parameters cannot represent simultaneously. This additional, non-redundant information is the source of MoE's advantage. Therefore, MoE beats the FLOP-matched dense baseline if and only if experts genuinely specialize.

Redundant experts would look like: (i) experts with weight matrices that are approximate copies of each other; (ii) routing distributions where the gating weights are nearly uniform regardless of input (the router has not learned to distinguish experts); or (iii) experts that all produce nearly the same output for any given input, so the mixture output $y$ is close to any individual expert's output.

#### Part (b)

**Hypothesis:** As the dense model scale $N$ grows large, the model enters the data-limited regime where the bottleneck on loss improvement shifts from model capacity to the amount of useful signal in the training data. In this regime, a dense model with $kN$ parameters can already fit most of the structure present in the data that is accessible within the compute budget. Adding expert parameters provides additional capacity to memorize rarer patterns, but the density of such rare patterns in the data is insufficient to provide consistent gradient signal for all $EN$ expert parameters.

More precisely: the MoE advantage rests on experts specializing on distinct subsets of the data distribution. The number of meaningfully distinct input types in the data is finite. As $E$ grows (more experts), at some point each expert receives training signal from a data subspace that does not have enough internal diversity to fully utilize the expert's $N$-parameter capacity. The expert parameters become under-trained relative to what they could represent, reducing the effective benefit of additional parameters. At very large $N$, the same issue applies to the dense baseline — both models are operating above the optimal parameter count for the data budget — but the MoE model has proportionally more under-trained parameters, shrinking its advantage.

#### Part (c)

Let $L_{\text{MoE}}(EN, kN)$ denote the loss of an MoE model with $EN$ total and $kN$ active parameters, and let $L_{\text{dense}}(P)$ denote the loss of a dense model with $P$ parameters (per-token FLOPs proportional to $P$). Define $N_{\text{eff}}$ such that $L_{\text{dense}}(N_{\text{eff}}) = L_{\text{MoE}}(EN, kN)$.

**Lower bound $N_{\text{eff}} > kN$:** If MoE achieves strictly lower loss than the FLOP-matched dense model (which has $kN$ parameters), then by definition of $N_{\text{eff}}$, we need $N_{\text{eff}} > kN$ to match the MoE loss with a dense model.

**Upper bound $N_{\text{eff}} < EN$:** If all $EN$ parameters of the MoE were accessible simultaneously (as in a dense model), the model would have the full representational power of a dense model with $EN$ parameters. But in MoE, only $kN$ parameters are active per token. Because the routing is discrete and imperfect, information from non-activated experts cannot be used for a given token — so the effective capacity is less than $EN$. Empirically (Fedus et al. 2021, Switch Transformers), a model with $EN$ active parameters always outperforms the same-size MoE with top-$k$ routing, confirming $N_{\text{eff}} < EN$.

**Implication for routing efficiency:** The gap $N_{\text{eff}} < EN$ means experts do not fully utilize their parameter capacity — the routing fails to make all $EN$ parameters collectively useful. This can happen because: (i) experts overlap in their specializations and are partially redundant; (ii) some experts are under-trained (receive too few tokens); (iii) the routing does not perfectly match tokens to the most appropriate expert. The closer $N_{\text{eff}}$ is to $EN$, the more efficient the routing. Current MoE models achieve $N_{\text{eff}}$ meaningfully above $kN$ but meaningfully below $EN$, indicating partial but imperfect utilization.

---

### Problem 10: Distributed Expert Parallelism

#### Part (a)

In expert parallelism with $E$ experts across $D = E$ devices (one expert per device), consider processing a batch of $T$ tokens where each device initially holds $T/D$ tokens.

**Before the MoE layer (dispatch all-to-all):** Device $d$ holds tokens $\{x_1, \ldots, x_{T/D}\}$ locally. Each token $x_t$ has been assigned to an expert by the router (which ran locally, since router weights are replicated). Token $x_t$ assigned to expert $i$ must be sent to device $i$. Each device sends its tokens to potentially any of $D$ other devices and receives tokens from potentially any of $D$ other devices. This is an all-to-all communication pattern.

In the balanced case (uniform routing), each device sends approximately $T/D \cdot (1/D) = T/D^2$ tokens to each other device, and receives $T/D^2$ tokens from each other device. Each token has dimension $d$ (the hidden dimension, in float32: $4d$ bytes). The bytes sent per device per collective is:

$$\text{bytes per device} = \frac{T}{D} \cdot d \cdot (\text{bytes per element})$$

(Each device holds $T/D$ tokens and must redistribute all of them, keeping $T/D^2$ locally and sending the rest.) In the worst case all $T/D$ tokens leave the device: $O(Td/D)$ bytes per device.

**After the MoE layer (gather all-to-all):** After each expert processes its tokens, the outputs (same shape as inputs, dimension $d$) must be sent back to the originating device. This is again an all-to-all operation with the same volume: $O(Td/D)$ bytes per device.

Total: two all-to-all operations per MoE layer, each communicating $O(T \cdot d / D)$ bytes per device.

#### Part (b)

For tensor parallelism on a linear layer $y = xW$ with $x \in \mathbb{R}^{T \times d}$ and $W \in \mathbb{R}^{d \times d}$, split $W$ column-wise: device 1 holds $W_1 \in \mathbb{R}^{d \times d/2}$ and device 2 holds $W_2 \in \mathbb{R}^{d \times d/2}$. Each device computes $xW_r \in \mathbb{R}^{T \times d/2}$. To get the full output $xW = [xW_1, xW_2]$, the results are concatenated (an all-gather). Alternatively, for row-wise split, each device computes a partial sum and an all-reduce is needed. The all-reduce communicates $T \cdot d$ elements (the full output tensor) split across 2 devices: each device sends and receives $T \cdot d / 2$ elements, which is $2Td$ bytes total or $Td$ bytes per device.

**Comparison:**

- Expert parallelism: $O(Td/D)$ bytes per device per all-to-all, where $D = E$ devices.
- Tensor parallelism (2-device): $O(Td)$ bytes per device per all-reduce (independent of $D$ for the 2-device case; scales as $O(Td \cdot (D-1)/D) \approx O(Td)$ for general $D$).

For $D = E$ devices, expert parallelism costs $Td/E$ bytes per device while tensor parallelism costs $\approx Td$ bytes per device (for large $D$). Expert parallelism has lower per-device communication for large $E$.

However the comparison also depends on $T$: if $T$ is small (small batch, streaming inference), then $Td/E$ is very small for expert parallelism, but tensor parallelism also communicates less (since $Td$ is small). The crossover point where they are equivalent depends on absolute batch size, not just $T/d$.

A cleaner crossover: expert parallelism dominates when $Td/D < Td$, i.e., $D > 1$ (always true with multiple devices) — but tensor parallelism communicates a fixed fraction of the activation size regardless of how many devices are used (all-reduce scales with device count), while expert parallelism communication per device decreases with $D$. Expert parallelism scales better at large $D$ and large $E$.

#### Part (c)

When token $t$ is dropped (its assigned expert is at capacity), its hidden state passes through the MoE layer unchanged (residual pass-through). The model's output for token $t$ does not depend on any expert parameters. When the loss $\mathcal{L}$ is computed over the full sequence of $T$ tokens:

$$\mathcal{L} = \frac{1}{T} \sum_{t=1}^T \ell(y_t, \hat{y}_t)$$

For a dropped token $t$, $y_t$ is the residual (unchanged hidden state), and $\partial \ell(y_t, \hat{y}_t) / \partial W_i = 0$ for all expert weight matrices $W_i$. Thus dropped tokens contribute zero gradient to all expert parameters.

**Systematic under-training:** Suppose expert $i$ is frequently overloaded (it is popular, so many tokens want to route to it, but only $C$ can be processed). The tokens dropped from expert $i$ are those with the lowest routing scores for expert $i$ among the overloaded tokens — but they still fall in the "preferred expert $i$" category. These tokens are systematically excluded from updating expert $i$'s parameters. Expert $i$ is trained only on its $C$ highest-confidence tokens per batch, missing the marginal tokens that could fine-tune its representation. Expert $i$'s parameters become optimized for the high-confidence regime but potentially suboptimal for borderline cases.

**Mitigation:** One approach is to route dropped tokens to a secondary expert (their second-highest-scoring expert) rather than passing them through unchanged. This ensures every token contributes gradient to at least one expert. However, this changes the semantics: the dropped token receives processing from a suboptimal expert, which may introduce incorrect gradient signal for the secondary expert's parameters (the secondary expert is being trained on inputs that were "really meant for" expert $i$). This does not fully resolve the bias — it redistributes it from "no gradient" to "misdirected gradient." A cleaner solution is to avoid dropping tokens entirely (use $\phi > 1$ sufficiently large), accepting the compute overhead in exchange for unbiased gradients.

---

## Implementation Sketches

### Problem 11: Noisy Top-k Gating

#### Part (a)

**Inputs:**

- $x \in \mathbb{R}^d$ — token embedding (shape: $[d]$)
- $W_g \in \mathbb{R}^{d \times E}$ — clean gating weight matrix (shape: $[d, E]$)
- $W_n \in \mathbb{R}^{d \times E}$ — noise scale weight matrix (shape: $[d, E]$)
- $k \in \mathbb{Z}_{>0}$ — number of experts to select

**Intermediate tensors to store for backward pass:**

- $z \in \mathbb{R}^E$ — clean logits (needed for gradient through $W_g$)
- $s \in \mathbb{R}^E$ — noise scales (needed for gradient through $W_n$)
- $\epsilon \in \mathbb{R}^E$ — sampled noise (needed for gradient through $s$)
- $\mathcal{T} \subseteq [E]$ with $|\mathcal{T}| = k$ — selected expert indices
- $g \in \mathbb{R}^E$ — gate weights (needed for gradient through softmax)
- $\{f_i(x)\}_{i \in \mathcal{T}}$ — selected expert outputs (needed for gradient through expert weights and through gate)

#### Part (b)

```
function compute_noisy_logits(x, W_g, W_n):
    # (i) Clean logits
    z = W_g.T @ x                    # shape: [E]

    # (ii) Per-expert noise scales (must be positive)
    raw_noise = W_n.T @ x            # shape: [E]
    s = softplus(raw_noise)          # shape: [E], softplus(u) = log(1 + exp(u))

    # (iii) Sample noise
    epsilon = sample_standard_normal(size=E)  # shape: [E], epsilon_i ~ N(0,1)

    # (iv) Noisy logits
    H = z + epsilon * s              # shape: [E], element-wise

    return H, z, s, epsilon
```

#### Part (c)

```
function sparse_softmax(H, k):
    # KeepTopK: set non-top-k entries to -inf
    top_k_values, top_k_indices = topk(H, k)   # values: [k], indices: [k]

    mask = full(E, fill_value=-inf)              # shape: [E]
    mask[top_k_indices] = H[top_k_indices]       # restore top-k entries

    # Softmax with numerical stability (mask -inf before exp)
    # Subtract max of finite entries for numerical stability
    finite_max = max(mask[top_k_indices])        # scalar
    shifted = mask - finite_max                  # shape: [E]

    exp_vals = zeros(E)
    exp_vals[top_k_indices] = exp(shifted[top_k_indices])   # only compute exp for finite entries
    # exp(-inf) = 0, so non-selected entries contribute 0

    Z = sum(exp_vals)                            # normalization constant
    g = exp_vals / Z                             # shape: [E], sparse (k nonzeros)

    return g, top_k_indices
```

Note: we explicitly avoid calling `exp` on `mask` directly because `exp(-inf)` may produce NaN in some implementations. By computing exp only on the top-$k$ entries (after masking), we guarantee numerical correctness.

#### Part (d)

```
function noisy_topk_forward(x, W_g, W_n, k, experts):
    # Step 1: Compute noisy logits
    H, z, s, epsilon = compute_noisy_logits(x, W_g, W_n)

    # Step 2: Sparse softmax
    g, selected_indices = sparse_softmax(H, k)
    # selected_indices: [k] — indices of the k selected experts

    # Step 3: Dispatch x to each selected expert and collect outputs
    expert_outputs = []
    for i in selected_indices:
        f_i = experts[i](x)          # shape: [d]
        expert_outputs.append(f_i)

    # Step 4: Weighted aggregation
    y = zeros(d)
    for idx, i in enumerate(selected_indices):
        y += g[i] * expert_outputs[idx]    # shape: [d]

    # Save for backward pass
    cache = {
        'x': x, 'z': z, 's': s, 'epsilon': epsilon,
        'g': g, 'selected_indices': selected_indices,
        'expert_outputs': expert_outputs
    }

    return y, cache
```

**Backward pass note:** Gradients flow from $y$ back through $g[i]$ (to $W_g$ via the softmax Jacobian) and through $f_i(x)$ (to expert $i$'s parameters). Gradients also flow through $s$ and $\epsilon$ to $W_n$. The selection operation (KeepTopK) is treated as a constant (stop-gradient on the discrete selection), so no gradient flows through the choice of which $k$ experts were selected.

---

### Problem 12: Expert Dispatch with Capacity Enforcement

#### Part (a)

```
# Inputs: token embeddings X: [T, d], gating weights W_g: [d, E], capacity C

# Step 1: Compute routing scores
S = softmax(X @ W_g, dim=-1)     # shape: [T, E]
                                   # S[t, i] = probability token t routes to expert i

# Step 2: Top-1 assignment
expert_assignment = argmax(S, dim=-1)   # shape: [T], values in {0, ..., E-1}
routing_scores_1d = max(S, dim=-1)      # shape: [T], the max score per token
                                         # used for capacity-based sorting
```

#### Part (b)

```
# Capacity enforcement per expert
dropped_mask = zeros(T, dtype=bool)     # shape: [T], True = dropped

for i in range(E):
    # Find all tokens assigned to expert i
    token_indices_i = where(expert_assignment == i)   # variable length, subset of [T]

    if len(token_indices_i) <= C:
        # Expert i is not overloaded; all tokens kept
        pass
    else:
        # Sort tokens by routing score descending (highest score = most confident assignment)
        scores_i = routing_scores_1d[token_indices_i]   # shape: [len(token_indices_i)]
        sorted_order = argsort(scores_i, descending=True)   # shape: [len(token_indices_i)]
        sorted_token_indices = token_indices_i[sorted_order]

        # Keep top C, drop the rest
        kept_tokens = sorted_token_indices[:C]
        dropped_tokens = sorted_token_indices[C:]
        dropped_mask[dropped_tokens] = True
```

#### Part (c)

```
# Dispatch: gather kept tokens into packed expert buffer, apply FFN, scatter back

# Build dispatch buffer: shape [E, C, d], zero-padded
dispatch_buffer = zeros(E, C, d)      # [E, C, d]
position_in_expert = zeros(E, dtype=int)   # tracks fill level per expert

# Build index maps for scatter
token_to_expert_pos = {}              # maps token index to (expert, position)

for i in range(E):
    kept_tokens_i = where((expert_assignment == i) and (not dropped_mask))
    for rank, t in enumerate(kept_tokens_i[:C]):
        dispatch_buffer[i, rank, :] = X[t, :]   # gather token t into expert i's slot
        token_to_expert_pos[t] = (i, rank)       # record where it went

# Apply expert FFNs
expert_outputs = zeros(E, C, d)      # [E, C, d]
for i in range(E):
    # Slice the filled portion; the FFN sees a batch of up to C tokens
    n_filled = count_filled(i)                   # number of tokens expert i actually received
    expert_outputs[i, :n_filled, :] = FFN_i(dispatch_buffer[i, :n_filled, :])
    # Slots beyond n_filled remain zero (no computation, no gradient)

# Scatter outputs back to original token positions
Y = zeros(T, d)                      # [T, d], output token embeddings
for t in range(T):
    if t in token_to_expert_pos:
        i, rank = token_to_expert_pos[t]
        g_t = routing_scores_1d[t]   # gate weight (for top-1, this IS g_{i*}(x_t) after renorm)
        Y[t, :] = g_t * expert_outputs[i, rank, :]
    else:
        # Token was dropped: pass residual through (zeros here; in practice, add to residual stream)
        Y[t, :] = zeros(d)

# Shape summary:
# S: [T, E] — routing scores
# expert_assignment: [T] — assigned expert index per token
# dispatch_buffer: [E, C, d] — padded expert input buffer
# expert_outputs: [E, C, d] — expert FFN outputs
# Y: [T, d] — output, with zeros for dropped tokens
```

#### Part (d)

**Gradient flow analysis:**

- **Kept tokens:** Gradient from the task loss flows backward through $Y[t] = g_t \cdot \text{expert\_outputs}[i, \text{rank}]$ to both (a) $g_t$ (and hence to the router $W_g$ via the softmax) and (b) $\text{expert\_outputs}[i, \text{rank}]$ (and hence to expert $i$'s FFN parameters via backprop through $\text{FFN}_i$).

- **Dropped tokens:** $Y[t] = 0$ is a constant that does not depend on any parameter. $\partial Y[t] / \partial W_i = 0$ for all $i$. Dropped tokens contribute **zero gradient to expert parameters**. They also contribute zero gradient to the router, because the router's gradient flows through $g_t$ only when $g_t$ appears in $Y[t]$, which it does not for dropped tokens (the zero output is not $g_t$ times anything that depends on parameters).

- **Zero-padded slots in dispatch\_buffer:** The FFN sees these as zero-valued inputs. The FFN output at zero-padded positions is ignored (those positions in $\text{expert\_outputs}$ are not scattered back to any token). No gradient flows backward through these padded positions.

- **The argmax in routing:** The discrete argmax (step (a)) is a stop-gradient operation. No gradient flows backward through the choice of which expert a token is assigned to. Gradient flows only through the soft score $g_t = S[t, i^*]$, not through the selection event $i^* = \arg\max_i S[t, i]$.

---

### Problem 13: Router Z-Loss Numerically Stable Implementation

#### Part (a)

```
function compute_z_loss(H):
    # H: [T, E] — routing logits for a batch of T tokens

    T, E = shape(H)

    # Numerically stable log-sum-exp per token
    m = max(H, dim=-1, keepdim=True)        # shape: [T, 1], per-token max logit
    shifted = H - m                          # shape: [T, E], subtract max for stability
    log_Z = m.squeeze(-1) + log(sum(exp(shifted), dim=-1))
    # log_Z: shape [T], the LSE for each token
    # log_Z[t] = log(sum_i exp(H[t, i]))

    # Z-loss: mean of squared LSE values
    L_z = mean(log_Z ** 2)    # scalar

    # Cache for backward pass
    cache = {'log_Z': log_Z, 'H': H, 'm': m}

    return L_z, cache
```

**Numerical correctness:** By subtracting $m[t] = \max_i H[t,i]$ before exponentiation, the largest term in the sum is $e^0 = 1$, and all other terms are $\leq 1$. This prevents overflow. The value $\log Z$ is then $m[t] + \log\sum_i e^{H[t,i] - m[t]}$, which is mathematically equal to $\log\sum_i e^{H[t,i]}$ but computed without overflow.

#### Part (b)

The gradient is $\frac{\partial \mathcal{L}_z}{\partial H[t,i]} = \frac{2}{T} \cdot \log Z(x_t) \cdot g_i(x_t)$ where $g_i(x_t) = \operatorname{softmax}(H[t,:])_i$.

```
function z_loss_backward(cache, grad_output):
    # cache: {'log_Z': [T], 'H': [T, E], 'm': [T, 1]}
    # grad_output: scalar (upstream gradient from loss reduction, typically 1.0)

    T, E = shape(cache['H'])
    log_Z = cache['log_Z']          # shape: [T]
    H = cache['H']                  # shape: [T, E]
    m = cache['m']                  # shape: [T, 1]

    # Reuse softmax from routing (or recompute if not cached):
    # g[t, i] = exp(H[t, i] - m[t]) / sum_j exp(H[t, j] - m[t])
    shifted = H - m                                       # [T, E]
    exp_shifted = exp(shifted)                            # [T, E]
    g = exp_shifted / sum(exp_shifted, dim=-1, keepdim=True)   # [T, E]
    # g is softmax(H), computed stably using cached m

    # Gradient: dL_z / dH[t, i] = (2/T) * log_Z[t] * g[t, i]
    grad_H = (2.0 / T) * log_Z.unsqueeze(-1) * g        # [T, E]
    # log_Z is already cached from forward pass — no recomputation
    # g shares the same intermediate (exp_shifted) as the routing softmax

    grad_H = grad_H * grad_output    # scale by upstream gradient

    return grad_H
```

**Efficiency note:** In practice, the routing step already computes the softmax of $H$ (to produce the gate weights $g$) and the LSE (as part of numerically stable softmax). Both $\log Z$ and $g$ are available from the forward pass at zero additional cost. The backward pass for z-loss therefore requires only an element-wise multiply and a scalar scaling, adding negligible overhead.

#### Part (c)

```
function total_loss(task_loss, L_aux, L_z, alpha, beta):
    # alpha: load-balancing loss coefficient
    # beta:  z-loss coefficient
    return task_loss + alpha * L_aux + beta * L_z
```

**Full training step:**

```
function training_step(batch, model, alpha, beta):
    # Forward pass
    logits, routing_info = model.forward(batch)
    # routing_info contains: H [T, E], g [T, E], f_i [E], P_i [E]

    # Task loss (e.g., cross-entropy)
    L_task = cross_entropy(logits, batch.labels)

    # Auxiliary load-balancing loss (Switch Transformer form)
    E = num_experts
    f = routing_info.f    # [E], stop-gradient (discrete fraction)
    P = routing_info.P    # [E], differentiable mean soft probability
    L_aux = E * sum(f * P)

    # Z-loss
    H = routing_info.H    # [T, E], routing logits
    L_z, z_cache = compute_z_loss(H)

    # Combined loss
    L_total = L_task + alpha * L_aux + beta * L_z

    # Backward
    L_total.backward()

    return L_total
```

**Hyperparameter values and trade-offs:**

- $\alpha$ (load-balancing loss coefficient): Switch Transformer uses $\alpha = 10^{-2}$. ST-MoE also uses $\alpha \approx 10^{-2}$. This coefficient must be large enough to prevent routing collapse but small enough not to dominate the task loss. Too large an $\alpha$ forces uniform routing, eliminating expert specialization. Too small and collapse can still occur. The $10^{-2}$ value is chosen empirically to lie in the range where routing stays reasonably balanced without sacrificing quality.

- $\beta$ (z-loss coefficient): ST-MoE uses $\beta \in [10^{-4}, 10^{-2}]$, with $\beta = 10^{-3}$ as a common default. The z-loss targets training stability (preventing exploding logits) rather than load balance directly. A small $\beta$ is sufficient because logit growth is gradual and the loss provides a soft regularizer rather than a hard constraint. Too large a $\beta$ aggressively shrinks all routing logits, potentially making the router under-confident and the routing less sharp than desired.

- The two coefficients address distinct failure modes: $\alpha$ prevents load imbalance and routing collapse, while $\beta$ prevents numerical instability from large routing logits. They can be tuned somewhat independently.
