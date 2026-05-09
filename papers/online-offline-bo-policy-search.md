# Bayesian Optimization for Policy Search via Online-Offline Experimentation

*Benjamin Letham, Eytan Bakshy — Facebook AI Research, JMLR 2019 · [arXiv:1904.01049](https://arxiv.org/abs/1904.01049)*

| Dimension | Prior State | This Paper | Key Result |
|---|---|---|---|
| Online experiment throughput | Each policy config needs its own A/B test; 10–20d spaces intractable | MTGP jointly models biased offline simulator + sparse online tests | Tuned 10–20d value model policy with **40 total online A/B tests** |
| Offline simulator utility | Considered too biased to use directly for optimization | ICM kernel treats bias as a smooth GP; models it away | MTGP + 100 offline obs rescues 10d policy search that fails completely with online-only GP |
| Source of multi-fidelity gain | Assumed: more data in posterior = better predictions | Ablation separates kernel inference from posterior conditioning | **Improved kernel lengthscale inference** (via shared spatial kernel) accounts for most of the MTGP gain |
| Simulator bias direction | No systematic characterization | Simulators systematically *overestimate* effect sizes | Attributed to unmodeled attention/time elasticities; bias is smooth and ICM-correctable |

## Relations

**Builds on:** Swersky et al. 2013 *(Multi-task Bayesian Optimization, NIPS)*, Chai 2009 *(MTGP generalization bound, NIPS)*, Letham et al. 2019 *(Constrained BO with noisy EI, Bayesian Analysis)*
**Concepts used:** [[curricula/multi-task-bayesian-optimization/curriculum.md|MTBO Prerequisites Curriculum]] (GP regression, ICM kernels, EI acquisition)

---

## Table of Contents

- [[#1. The Value Model as a Policy Space|1. The Value Model as a Policy Space]]
- [[#2. The Offline Simulator and Its Biases|2. The Offline Simulator and Its Biases]]
  - [[#2.1 How the Naive Simulator Works|2.1 How the Naive Simulator Works]]
  - [[#2.2 Why Simulators Systematically Overestimate|2.2 Why Simulators Systematically Overestimate]]
- [[#3. Multi-Task Gaussian Process with ICM Kernel|3. Multi-Task Gaussian Process with ICM Kernel]]
  - [[#3.1 From Single-Task to Multi-Task GP|3.1 From Single-Task to Multi-Task GP]]
  - [[#3.2 The ICM Kernel|3.2 The ICM Kernel]]
  - [[#3.3 Bias as a GP: The Key Structural Insight|3.3 Bias as a GP: The Key Structural Insight]]
  - [[#3.4 Hyperparameter Inference|3.4 Hyperparameter Inference]]
- [[#4. The Bayesian Optimization Loop|4. The Bayesian Optimization Loop]]
  - [[#4.1 Noisy EI under Constraints|4.1 Noisy EI under Constraints]]
  - [[#4.2 Multi-Task Acquisition and the Interleaved Loop|4.2 Multi-Task Acquisition and the Interleaved Loop]]
  - [[#4.3 Simulator Non-Stationarity|4.3 Simulator Non-Stationarity]]
- [[#5. Reading the Empirical Results|5. Reading the Empirical Results]]
  - [[#5.1 LOO-CV Scatter Plots (Figs. 2–3)|5.1 LOO-CV Scatter Plots (Figs. 2–3)]]
  - [[#5.2 Learning Curves (Fig. 7)|5.2 Learning Curves (Fig. 7)]]
  - [[#5.3 The Kernel Inference Ablation (Fig. 10)|5.3 The Kernel Inference Ablation (Fig. 10)]]
- [[#6. The ρ² Bound on Multi-Task Generalization|6. The ρ² Bound on Multi-Task Generalization]]
  - [[#6.1 Chai (2009) Bound|6.1 Chai (2009) Bound]]
  - [[#6.2 Bound vs. Empirical (Fig. 9)|6.2 Bound vs. Empirical (Fig. 9)]]
- [[#References|References]]

---

## 1. The Value Model as a Policy Space

A modern industrial recommender system (e.g., Facebook News Feed) relies on a cascade of ML models, each predicting a specific *event probability*: probability of click, share, comment, long view, etc. The *value model* is the function that combines these predictions into a single scalar score used to rank items:

$$s(i) = \sum_j w_j \cdot \hat{p}_j(i) + \text{conditional rules}(\mathbf{x}_i)$$

Here $\hat{p}_j(i)$ is the event model prediction for item $i$ on event type $j$, and the weights $w_j$ plus any conditional rules (e.g., "apply weight $\alpha$ to videos with engagement rate $> \theta$") constitute the **policy** $\pi \in \mathbb{R}^d$.

> [!NOTE] Value model terminology
> At LinkedIn, Yahoo!, and Meta this object goes by different names (blending function, scoring formula, value model), but the structure is the same: a parameterized linear-or-piecewise-linear combination of ML-predicted signals. The policy space dimension $d$ ranges from ~5 (simple systems) to 20+ (mature production ranking).

The key consequence: **outcomes of interest are nonlinear functions of the policy $\pi$**. Changing $w_\text{comment}$ by 0.1 might increase comments by 2% but decrease long-views by 1%, with the trade-off depending on the current operating point. This nonlinear response surface is what Bayesian optimization is designed to navigate.

📐 Critically, *the response surface can only be observed online* — event models are trained on historical data and optimized for prediction, not for evaluating counterfactual policy changes. Only a live A/B test reveals how humans actually respond to a new ranking order. This is the bottleneck that the paper addresses.

---

## 2. The Offline Simulator and Its Biases

### 2.1 How the Naive Simulator Works

The simulator constructs a counterfactual estimate by *replaying* logged user sessions through the modified value model:

```python
def simulate_policy(sessions, new_policy, event_models):
    outcomes = []
    for session in sessions:
        # Re-rank items with new policy weights
        scored_items = [(item, score(item, new_policy)) for item in session.candidates]
        ranked = sorted(scored_items, key=lambda x: -x[1])
        
        # Apply event models to selected items (e.g., top-k shown)
        shown = ranked[:K]
        predicted_events = sum(
            event_models[j].predict(item) for item in shown for j in OUTCOMES
        )
        outcomes.append(aggregate(predicted_events))
    return mean(outcomes)
```

This requires *zero additional infrastructure* — it reuses the existing event models and the ranking system itself. But it has two major sources of bias.

**Source 1: Off-policy distribution shift.** The event models $\hat{p}_j$ were trained on data generated by the *current* policy $\pi_0$. When the simulator evaluates $\pi \neq \pi_0$, it shows items that the current policy would not have shown, and applies event models that were trained on a different item distribution. Predictions for out-of-distribution items are unreliable.

**Source 2: Missing session dynamics.** The simulator assumes independence across items and across sessions. In reality:
- What a user does on item $i$ at position $k$ depends on items $1, \ldots, k-1$ (cascade / attention effects)
- Interactions in one session influence behavior in future sessions (habit formation, satiation)

### 2.2 Why Simulators Systematically Overestimate

⚠️ The paper makes a subtle empirical claim: simulators don't just introduce *noise* — they systematically *overestimate* the magnitude of policy effects. The authors attribute this to **attention and time elasticities** (citing Simon 1971, 1996): humans have finite attention capacity, so showing "better" content does not produce proportionally more engagement. If the simulator predicts that switching to policy $\pi$ increases comments by $\Delta$, the true online effect is typically $\alpha \cdot \Delta$ with $\alpha < 1$.

> [!WARNING] Why this matters for optimization
> A biased-but-correlated simulator is useful for *ordering* policies (which is better?). But if the bias is not corrected, constraints calibrated in the simulator will be violated online. You cannot simply use a scalar correction factor because $\alpha$ depends on *which parameters are being tuned* — the paper shows the bias is experiment-specific.

The ICM model's response to this: don't correct the bias explicitly. Instead, model the bias *as a function of policy parameters* using the flexibility of the GP. The MTGP learns how the offline-to-online discrepancy varies across the design space.

> [!INFO] The correlation survives the bias
> From Figure 1 in the paper: the simulator outputs and online results are clearly correlated (points cluster around a line through the origin), even though the slope is not 1 and the scatter grows far from the status quo. This correlated-but-biased structure is exactly what the ICM model is designed to exploit.

---

## 3. Multi-Task Gaussian Process with ICM Kernel

### 3.1 From Single-Task to Multi-Task GP

Recall the GP posterior (kriging equations). Given observed data $\{X, y\}$ with noise $\eta^2$, the predictive distribution at $x^*$ is

$$f(x^*) \mid x^*, X, y \;\sim\; \mathcal{N}\!\left(\mu(x^*, X, y),\; \sigma^2(x^*, X)\right)$$

$$\mu(x^*, X, y) = m(x^*) + K(x^*, X)\left[K(X,X) + \operatorname{diag}(\eta^2)\right]^{-1}(y - m(X)) \tag{kriging mean}$$

$$\sigma^2(x^*, X) = k(x^*, x^*) - K(x^*, X)\left[K(X,X) + \operatorname{diag}(\eta^2)\right]^{-1}K(X, x^*) \tag{kriging variance}$$

> [!NOTE] Variance is independent of observations $y$
> Notice $\sigma^2(x^*, X)$ depends only on the *locations* $X$, not on the values $y$. This is a structural consequence of Gaussian conditioning and is what makes GP-based learning curves (Section 6) tractable — you can compute expected predictive variance from the design alone.

The **Multi-Task GP** extends this to $D$ tasks by letting each task $d$ have its own function $f_d : \mathcal{X} \to \mathbb{R}$. Points in the combined dataset are indexed by $(d, x)$ pairs. We need a covariance function over the product space $\{1,\ldots,D\} \times \mathcal{X}$.

### 3.2 The ICM Kernel

The *Intrinsic Coregionalization Model* (ICM) makes two assumptions:

**Assumption 1 (Shared spatial structure).** All tasks share the same base spatial covariance $\kappa(x, x')$. In this paper, $\kappa$ is the ARD RBF kernel:

$$\kappa(x, x') = \tau^2 \exp\!\left(-\frac{1}{2}\sum_{j=1}^m \frac{(x_j - x'_j)^2}{\ell_j^2}\right)$$

with per-dimension lengthscales $\ell_j$ and output scale $\tau^2$.

**Assumption 2 (Separability).** The cross-task covariance factorizes as a product of task covariance and spatial covariance:

$$\operatorname{Cov}[f_d(x), f_{d'}(x')] = B_{d,d'} \cdot \kappa(x, x') \tag{ICM}$$

where $B \in \mathbb{R}^{D \times D}$ is a positive semi-definite *task covariance matrix*.

The ICM kernel has an equivalent characterization in terms of latent functions. For $D = 2$ tasks, each task function is a linear combination of $P$ independent latent GPs $u_1, \ldots, u_P$ (each with covariance $\kappa$):

$$f_1(x) = \sum_{p=1}^P a_{1p}\, u_p(x), \qquad f_2(x) = \sum_{p=1}^P a_{2p}\, u_p(x)$$

The task covariance matrix is then $B = AA^\top$ where $A_{dp} = a_{dp}$. The rank of $B$ equals $P$, the number of latent functions. Using a *low-rank* decomposition $B = \tilde{L}\tilde{L}^\top$ with $\tilde{L} \in \mathbb{R}^{D \times P}$, $P < D$, reduces the number of task covariance parameters from $O(D^2)$ to $O(DP)$ — important when $D$ grows as new simulator batches are added.

### 3.3 Bias as a GP: The Key Structural Insight

💡 From the latent decomposition we get an elegant characterization of the simulator bias. Rearranging for $P = 2$:

$$f_2(x) = f_1(x) + u'(x) \qquad \text{where } u'(x) = (a_{21} - a_{11})u_1(x) + (a_{22} - a_{12})u_2(x)$$

Since $u_1, u_2$ are independent GPs with covariance $\kappa$, their linear combination $u'$ is also a GP with covariance

$$k_{u'}(x, x') = \left[(a_{21}-a_{11})^2 + (a_{22}-a_{12})^2\right] \kappa(x, x')$$

**The ICM model implicitly assumes the offline-to-online bias is a GP with the same smoothness as the response surface itself.** This is precisely the right assumption when the bias arises from aggregate behavioral dynamics: smooth, structured, but not zero.

> [!TIP] What "same smoothness" buys you
> If the true response surface has long lengthscales (the value model is smooth in its parameters), then the bias function is also smooth. This means: near the status quo, the bias is small and predictable; far from the status quo, the bias grows but remains structured. The ICM can learn to correct for it given a few online calibration points.

> [!EXAMPLE] Extreme cases of B
> - If $B_{12} = \sqrt{B_{11} B_{22}}$ (rank-1): both tasks concentrate on *the same* latent function, i.e., $f_2 = c \cdot f_1$ — a pure scale bias. This is the "linear correction" often used in engineering multi-fidelity models.
> - If $B_{12} = 0$ (diagonal): tasks are modeled as independent GPs with the same lengthscales — the ICM reduces to two separate GPs sharing only kernel hyperparameters.

> [!QUESTION] Exercise 1: Bias Covariance Under ICM
> *The ICM latent decomposition characterizes the structure of the simulator bias.*
>
> > **Prerequisites:** [[#3.2 The ICM Kernel|3.2 The ICM Kernel]]
>
> For the two-task ICM with $P = 2$ latent functions, let task 1 be the online task and task 2 the offline simulator. Define the bias process $u'(x) = f_2(x) - f_1(x)$.
>
> (a) Derive the covariance $k_{u'}(x, x')$ of the bias process in terms of the ICM weights $a_{dp}$ and the base kernel $\kappa(x, x')$.
>
> (b) Under what conditions on the weights is the bias identically zero? What is the interpretation in terms of task covariance matrix $B$?

> [!TIP]- Solution to Exercise 1
> **Key insight:** $u' = f_2 - f_1$ is a GP because it is a linear combination of GPs; its covariance is computed via bilinearity of covariance.
>
> **Sketch:**
> $u'(x) = (a_{21} - a_{11})u_1(x) + (a_{22} - a_{12})u_2(x)$
>
> Since $u_1 \perp u_2$ with $\text{Cov}[u_p(x), u_p(x')] = \kappa(x, x')$:
>
> $$k_{u'}(x, x') = \left[(a_{21}-a_{11})^2 + (a_{22}-a_{12})^2\right]\kappa(x, x')$$
>
> (b) Bias is identically zero iff $a_{21} = a_{11}$ and $a_{22} = a_{12}$, i.e., iff $f_1 = f_2$. In terms of $B$: $B_{11} = B_{12} = B_{22}$ (all tasks identical). This is the $\rho = 1$ case from Section 6.

### 3.4 Hyperparameter Inference

The MTGP has hyperparameters $\Theta = \{B, \tau^2, \ell_1, \ldots, \ell_m\}$. Inference is by marginal likelihood maximization (or slice sampling for $B$):

$$\log p(y \mid X, \Theta) = -\frac{1}{2}y^\top (K_\Theta + \text{diag}(\eta^2))^{-1}y - \frac{1}{2}\log|K_\Theta + \text{diag}(\eta^2)| - \frac{n}{2}\log 2\pi$$

where $K_\Theta$ is the full $n \times n$ kernel matrix over all tasks and points. To ensure $B \succ 0$, it is reparameterized as $B = LL^\top$ with $L$ lower-triangular; then $L$ is optimized.

🔑 **The critical advantage of the shared spatial kernel**: the lengthscales $\ell_j$ are inferred from *all* observations across tasks. Since the simulator provides $n_S \gg n_T$ observations, the lengthscales are driven almost entirely by offline data. Online experiments can then focus on learning the task relationship $B$ rather than kernel hyperparameters, which are notoriously hard to infer from $O(20)$ points in 10–20 dimensions.

---

## 4. The Bayesian Optimization Loop

### 4.1 Noisy EI under Constraints

The optimization problem is:
$$\max_{x \in \mathcal{X}} f(x) \quad \text{subject to } c_j(x) \geq 0, \; j = 1, \ldots, J$$

where $f$ is the objective (e.g., meaningful interactions) and $c_j$ are constraint metrics (e.g., ecosystem health). All are observed with noise.

The *improvement* over the current best feasible policy at a new point $x$ is:

$$I(x, X) = \max\!\left(0,\; f(x) - \max_{x^* \in X: c(x^*) \geq 0} f(x^*)\right) \cdot \mathbf{1}_{c(x) \geq 0}$$

The *Noisy Expected Improvement* (NEI) acquisition function averages over both observation noise at $x$ and noise at the existing best feasible:

$$\alpha_\text{NEI}(x \mid X, \mathcal{D}) = \mathbb{E}_{f(X), c(X)}\!\left[\mathbb{E}_{f(x)}\!\left[I(x,X) \mid f(X), c(X)\right] \Bigg\vert \mathcal{D}\right]$$

The inner expectation is analytic (standard EI formula). The outer expectation over the noisy current-best is handled with quasi-Monte Carlo integration.

### 4.2 Multi-Task Acquisition and the Interleaved Loop

In the multi-task setting, the acquisition function is modified so that **only online observations count toward "best feasible"**:

$$\alpha_\text{NEI}(x \mid X_T, \mathcal{D}_T \cup \mathcal{D}_S) \tag{MT-NEI}$$

Simulator observations improve the posterior over $f(X_T)$ and $c(X_T)$ but are never treated as candidate optimal solutions. This is principled: the goal is to find the best *online* policy.

The optimization loop interleaves offline and online batches:

```python
# Algorithm 1: Online-Offline Bayesian Optimization
X_T, X_S = quasi_random_batch(n_T=20), quasi_random_batch(n_S=100)
D = run_online(X_T) + run_simulator(X_S)

for k in range(K):
    model = fit_MTGP(D)                         # fit ICM kernel to all data
    X_opt = optimize_acqf(MT_NEI, model, n=30)  # 30 optimized candidates
    
    D_S_k = run_simulator(X_opt)               # test candidates offline
    D = D ∪ D_S_k
    model = fit_MTGP(D)                        # re-fit with offline results
    
    X_T_k = thompson_sample(X_opt, model, n=8) # select best 8–10 for online
    D_T_k = run_online(X_T_k)
    D = D ∪ D_T_k

return best_online_policy(D)
```

> [!TIP] Why the offline pre-test matters
> After optimizing MT-NEI, some of the 30 candidates have high EI because of high *uncertainty* (exploration), not because they're likely good. Running them on the simulator resolves that uncertainty cheaply. When the MTGP is re-fit with simulator results, the updated uncertainty often de-prioritizes those candidates. The online batch of 8–10 is thus more *exploitative* while exploration happened "for free" offline. This is the key practical contribution of the interleaved design.

### 4.3 Simulator Non-Stationarity

The simulator replays live sessions during a short window. Sessions vary by time of day, day of week, and content freshness — so two simulator batches run days apart on the *same* policy produce different outcomes (Figure 11 in the paper shows this clearly: linear shifts between batches).

The fix: treat each simulator batch as a *separate task* in the MTGP, using a low-rank ICM kernel to learn the batch-to-batch adjustment. Since the biases are empirically near rank-1 (linear shifts), a rank-1 $\tilde{L}$ is usually sufficient. The online-offline covariance uses a full-rank (rank-2) $B$.

---

## 5. Reading the Empirical Results

### 5.1 LOO-CV Scatter Plots (Figs. 2–3)

These are *leave-one-out cross-validation* scatter plots. Each point corresponds to a single policy that was:
1. Held out from the training set
2. Predicted by the GP/MTGP fit to the remaining observations
3. Actually tested online

The **x-axis** is the observed online outcome (ground truth); the **y-axis** is the model's LOO prediction. A model with perfect predictions concentrates all points on the diagonal $y = x$. A model that predicts the mean for every point produces a horizontal band around $y = 0$ (since outcomes are standardized).

> [!EXAMPLE] Reading Fig. 3
> - **Left panel (single-task GP, 20 online obs, 10d space):** Points are scattered in a roughly horizontal band with no visible diagonal structure. The GP is effectively predicting the mean. *Why?* With 20 points in 10 dimensions, the kernel lengthscales cannot be reliably estimated — the MLE of the lengthscale collapses to the prior or to extreme values, producing near-constant predictions.
> - **Right panel (MTGP, 20 online + 100 offline obs):** Points cluster tightly around the diagonal. The same 20 online observations, but now the shared ICM spatial kernel can use 100 offline points to estimate lengthscales, transforming an underdetermined problem into a well-conditioned one.

### 5.2 Learning Curves (Fig. 7)

Figure 7 shows **mean squared prediction error (MSE) as a function of number of observations**, estimated via 500-subsample cross-validation. There are two panels per column:
- **Top row:** MSE vs. number of *online* observations $n_T$, for fixed $n_S \in \{0, 10, 30, 90\}$
- **Bottom row:** MSE vs. number of *offline* observations $n_S$, for fixed $n_T \in \{4, 10, 18\}$

The three columns correspond to $\rho^2 \approx \{0.95, 0.80, 0.09\}$.

| Region of Fig. 7 | What to look for | What it shows |
|---|---|---|
| Top row, high $\rho^2$, $n_S = 90$ vs. $n_S = 0$ | Vertical gap between curves | Each offline observation substitutes for ~1/3–1/2 of an online observation |
| Top row, low $\rho^2$, any $n_S$ | Curves nearly overlap | Offline observations add almost nothing when tasks are uncorrelated |
| Bottom row, any column | Rate of decrease with $n_S$ | Shows diminishing returns: gains concentrated at low $n_S$ |
| Crossover point | Where $n_S = 0$ curve reaches same MSE as $n_S = 90$ | Read off the "equivalent online observations" a full offline batch provides |

📐 *Reading the ρ² = 0.95 panel closely:* at $n_T = 2$ online observations, adding 90 offline observations reduces MSE from ~1.4 to ~0.2 — comparable to 18 online observations with no offline. So at high correlation, the offline-to-online information ratio is roughly 90:18 ≈ 5:1 in terms of equivalent samples. This ratio is determined by $\rho^2$ and the shape of the learning curve, not by the simulator accuracy per se.

> [!QUESTION] Exercise 2: Equivalent Sample Count
> *Learning curves quantify the relative value of online vs. offline observations.*
>
> > **Prerequisites:** [[#5.2 Learning Curves (Fig. 7)|5.2 Learning Curves (Fig. 7)]], [[#6.1 Chai (2009) Bound|6.1 Chai (2009) Bound]]
>
> Using the Chai bound $\mathcal{E}_T \geq \rho^2 \kappa(n_T + n_S) + (1-\rho^2)\kappa(n_T)$, define the *effective online equivalent* of $n_S$ offline observations as the value $\Delta n$ such that:
>
> $$\kappa(n_T + \Delta n) \approx \mathcal{E}_T(\rho, n_T, n_S) / \rho^2$$
>
> (a) Show that for a power-law learning curve $\kappa(n) = c \cdot n^{-\beta}$, the effective equivalent satisfies $\Delta n \approx \rho^2 n_S$ when $n_T \ll n_S$.
>
> (b) What does this imply about the maximum useful number of simulator observations once $n_S \gg n_T / \rho^2$?

> [!TIP]- Solution to Exercise 2
> **Key insight:** When the bound is tight, offline observations scale the effective $n$ by $\rho^2$.
>
> **Sketch:**
> (a) $\mathcal{E}_T \geq \rho^2 \kappa(n_T + n_S) + (1-\rho^2)\kappa(n_T) \approx \rho^2 \kappa(n_S)$ for $n_S \gg n_T$.
> Setting $\kappa(n_T + \Delta n) = \kappa(n_S)$, we get $n_T + \Delta n = n_S$, i.e., $\Delta n = n_S - n_T$.
> More carefully with the $\rho^2$ weight: $\kappa(n_T + \Delta n) = \kappa(n_T + n_S)$ only when $\rho^2 = 1$; for general $\rho^2$, the effective contribution of the $(n_T + n_S)$-point learning curve is discounted by $\rho^2$. For power law $\kappa(n) = c n^{-\beta}$:
> $$\Delta n \approx \rho^2 n_S \quad (\text{for } n_T \ll n_S)$$
>
> (b) Returns diminish rapidly once $n_S > n_T / \rho^2$: the learning curve $\kappa$ flattens, so additional simulator observations provide negligible further reduction in the bound. At high $\rho^2$, the saturation point is early; at low $\rho^2$, you never gain much regardless.

### 5.3 The Kernel Inference Ablation (Fig. 10)

This is the most surprising result in the paper. Figure 10 compares three models:

| Model | Kernel | Posterior data |
|---|---|---|
| GP, online kernel | Inferred from online data only | Online only |
| GP, offline kernel | Fixed at offline-inferred values | Online only |
| MTGP | Inferred jointly | Online + offline |

🔑 **The offline kernel model (middle) nearly closes the gap between single-task GP and MTGP.** At $n_T = 18$ online observations, for $\rho^2 = 0.8$ and $\rho^2 = 0.09$, the performance difference between "GP + offline kernel" and MTGP is small — while the gap between "GP + online kernel" and MTGP is large.

*Interpretation:* In 10 dimensions with only 20 online observations, the MLE for ARD lengthscales is highly underdetermined. The optimizer may collapse to near-zero lengthscales (overfitting) or very long lengthscales (underfitting). The offline data, which provides 100 observations, gives a much better-conditioned MLE problem. Sharing the kernel across tasks means this well-estimated kernel gets used for online predictions too — essentially free regularization from the simulator.

> [!WARNING] Implication for sim design
> You don't need a *calibrated* simulator to benefit from MTGP. You need a simulator whose response surface has approximately the same *smoothness structure* (lengthscales) as the online surface. A simulator that's systematically biased but has the right spatial structure will still improve kernel inference significantly.

---

## 6. The ρ² Bound on Multi-Task Generalization

### 6.1 Chai (2009) Bound

Define the **single-task learning curve** as the expected predictive variance with $n$ i.i.d. training points:

$$\kappa(n) = \mathbb{E}_{X, x^*}\!\left[\sigma^2(x^*, X)\right] \tag{single-task}$$

For the MTGP with $n_T$ primary (online) and $n_S$ secondary (offline) observations, the **multi-task learning curve** is:

$$\mathcal{E}_T(\rho, n_T, n_S) = \mathbb{E}_{X_T, X_S, x^*}\!\left[\sigma_T^2(x^*, \rho, X_T, X_S)\right]$$

Under the normalized ICM with $B = \begin{pmatrix}1 & \rho \\ \rho & 1\end{pmatrix}$, Chai (2009) proved:

$$\sigma_T^2(x^*, \rho, X_T, X_S) \geq \rho^2 \sigma_T^2(x^*, \rho=1, X_T, X_S) + (1-\rho^2)\sigma_T^2(x^*, \rho=0, X_T, X_S)$$

Taking expectations gives **Corollary 2**:

$$\boxed{\mathcal{E}_T(\rho, n_T, n_S) \geq \rho^2 \kappa(n_T + n_S) + (1-\rho^2)\kappa(n_T)}$$

The two extreme cases recover known single-task results:
- $\rho = 1$ (perfect task correlation): bound is $\kappa(n_T + n_S)$ — equivalent to a single task with all observations combined
- $\rho = 0$ (independent tasks): bound is $\kappa(n_T)$ — secondary task observations are ignored

The bound is a *lower bound* on the multi-task learning curve — i.e., an upper bound on how well the MTGP can do. The key practical property: **the single-task curve $\kappa(n)$ can be estimated from secondary task observations only** (Figure 8). Combined with an estimate of $\rho^2$ from past experiments, this lets you predict MTGP performance before collecting any primary observations.

> [!QUESTION] Exercise 3: Bound Verification at Extremes
> *The bound should recover standard single-task results at $\rho \in \{0, 1\}$.*
>
> > **Prerequisites:** [[#6.1 Chai (2009) Bound|6.1 Chai (2009) Bound]]
>
> (a) Starting from $\mathcal{E}_T \geq \rho^2 \kappa(n_T + n_S) + (1-\rho^2)\kappa(n_T)$, verify that at $\rho = 1$ and $\rho = 0$ you recover the interpretations above.
>
> (b) Show that the bound is monotonically non-decreasing in $n_S$ (i.e., adding offline observations can only *improve* — or at worst not change — the bound on the multi-task learning curve).
>
> (c) For a fixed total budget $n = n_T + n_S$, at what ratio $n_T/n_S$ does the bound become independent of $\rho^2$?

> [!TIP]- Solution to Exercise 3
> **Key insight:** The bound is a convex combination of two single-task learning curves; $\rho^2$ controls the mixing weight.
>
> **Sketch:**
> (a) $\rho = 1$: $\mathcal{E}_T \geq 1 \cdot \kappa(n_T+n_S) + 0 \cdot \kappa(n_T) = \kappa(n_T+n_S)$. This is the joint single-task bound — all observations are equivalent. $\rho = 0$: $\mathcal{E}_T \geq 0 + 1 \cdot \kappa(n_T) = \kappa(n_T)$ — offline ignored.
>
> (b) $\kappa$ is non-increasing in $n$ (more data $\Rightarrow$ lower expected variance). Adding $\Delta n_S > 0$ changes the bound's first term from $\rho^2 \kappa(n_T + n_S)$ to $\rho^2 \kappa(n_T + n_S + \Delta n_S) \leq \rho^2 \kappa(n_T + n_S)$. Second term unchanged. So the bound is non-increasing in $n_S$, i.e., the *bound on prediction error* decreases, meaning predictions can only improve.
>
> (c) The bound equals $\kappa(n)$ (independent of $\rho^2$) when both terms are equal: $\kappa(n_T+n_S) = \kappa(n_T)$, i.e., when $n_S = 0$ — trivially. For $n_S > 0$, the bound always depends on $\rho^2$. The bound becomes *insensitive* to $\rho^2$ in the limit $\rho^2 \to 0$ (offline observations become useless) or when $\kappa$ is flat (saturation regime, learning curve has converged).

### 6.2 Bound vs. Empirical (Fig. 9)

Figure 9 overlays the theoretical Chai bound (computed using offline-estimated $\kappa$ and $\rho^2$) against the empirical MTGP learning curves. Key observations:

1. **The bound is consistently optimistic** (empirical is below the bound = better than predicted), especially at low $n_T$. The paper attributes this to improved *kernel inference* with the MTGP — a mechanism not captured by Chai's bound, which assumes fixed true kernel parameters.

2. **The shape and $\rho^2$-ordering are well-predicted.** The theory correctly predicts that the $\rho^2 = 0.95$ curve descends much faster than the $\rho^2 = 0.09$ curve as $n_S$ grows.

3. **Practical use:** Since $\kappa(n)$ can be estimated from offline data alone, and $\rho^2$ can be estimated from past experiments, the bound gives a *pre-deployment* estimate of how many online vs. offline observations you need. This is a real operational tool for experiment sizing.

---

## References

| Reference | Brief Summary | Link |
|---|---|---|
| Letham & Bakshy (2019) | *This paper.* MTGP + noisy EI for value model policy search at Facebook | [arXiv:1904.01049](https://arxiv.org/abs/1904.01049) |
| Chai (2009) | Theoretical learning curve bounds for multi-task GP regression; the $\rho^2$ bound used throughout | [NIPS 2009](https://papers.nips.cc/paper/2009/hash/3a15c7d0bbe60300a39f76f8a5ba6896-Abstract.html) |
| Swersky, Snoek & Adams (2013) | Multi-task BO for hyperparameter optimization; introduced MTGP for downsampled datasets | [NIPS 2013](https://papers.nips.cc/paper/2013/hash/f33ba15effa5c10e873bf3842afb46a6-Abstract.html) |
| Letham, Karrer et al. (2019) | Constrained BO with noisy EI (noisy EI formula and the acquisition function used here) | [Bayesian Analysis](https://projecteuclid.org/journals/bayesian-analysis/volume-14/issue-2/Constrained-Bayesian-Optimization-with-Noisy-Experiments/10.1214/18-BA1110.full) |
| Álvarez & Lawrence (2011) | ICM kernel derivation and computationally efficient multi-output GP | [JMLR](https://jmlr.org/papers/v12/alvarez11a.html) |
| Bottou et al. (2013) | Counterfactual reasoning in production ad systems; motivates bias sources in simulators | [JMLR](https://jmlr.org/papers/v14/bottou13a.html) |
| Simon (1971, 1996) | Information-rich environments and attention capacity; theoretical basis for simulator overestimation | [Book: The Sciences of the Artificial](https://mitpress.mit.edu/9780262691918/) |
