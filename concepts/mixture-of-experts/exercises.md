# Mixture of Experts: Exercises

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

*This problem establishes the full gradient of the soft-gated MoE output with respect to the router weight matrix $W_g$, and characterizes the stationary-point condition that the router satisfies at convergence. The result illuminates why soft gating always propagates signal to every expert.*

> **Prerequisites:** cf. note [[note#Soft Gating|§2 — Soft Gating]]

(a) Let $y = \sum_{i=1}^E g_i(x) f_i(x)$ with $g_i(x) = \operatorname{softmax}(W_g^\top x)_i$ and $f_i(x) = W_i x$ (linear experts). Given a scalar loss $\mathcal{L}$ with upstream gradient $\partial \mathcal{L}/\partial y$ known, compute $\partial \mathcal{L}/\partial W_g$ via the chain rule. Show the gradient takes the form of a weighted outer product involving $(f_i(x) - y)$ for each expert $i$.

(b) At a stationary point $\partial \mathcal{L}/\partial W_g = 0$, characterize the fixed-point condition on the gate weights $g_i(x)$. Identify which experts exert zero gradient pressure on $W_g$ from this term, and explain why expert outputs equal to the ensemble mean contribute no training signal to the router.

(c) The soft gating gradient is dense: all $E$ experts receive a gradient signal on every forward pass. Contrast with hard top-$k$ gating, where only the $k$ selected experts receive a gradient through the softmax weights. Show that restricting the gradient to $k$ experts is equivalent to computing the exact gradient of a modified objective where unselected experts have their logits clamped to $-\infty$ before the softmax.

---

### Problem 2: Top-k Gating as a Constrained Argmax

*This problem analyzes the boundary behavior of the KeepTopK operator and establishes that hard top-k gating interpolates between soft gating and one-hot selection as $k$ varies. Understanding these limits clarifies the computational-statistical tradeoff of sparsity.*

> **Prerequisites:** cf. note [[note#Hard Top-k Gating|§2 — Hard Top-k Gating]]

(a) Define $\operatorname{KeepTopK}(h, k)_i = h_i$ if $h_i$ is among the $k$ largest values of $h$, and $-\infty$ otherwise. Show that as $k \to E$, the sparse gate $g = \operatorname{softmax}(\operatorname{KeepTopK}(h, k))$ recovers standard softmax. Show that as $k \to 1$, $g \to \operatorname{one-hot}(\arg\max_i h_i)$.

(b) The KeepTopK operation is not differentiable at inputs where two logits are tied. For generic inputs where all logits are distinct, show that the gradient of the $k$ selected logits with respect to $W_g$ is well-defined and has the same algebraic form as the soft gating gradient, restricted to the top-$k$ experts.

(c) Formalize the straight-through estimator for KeepTopK: fix the selected set $\mathcal{T}(x)$ during backpropagation and pass gradients only through the selected experts' softmax logits. Show this equals the exact gradient of a modified objective $\widetilde{\mathcal{L}}$ in which all non-selected logits are replaced by $-\infty$ before the softmax.

---

### Problem 3: Gradient Variance in Sparse vs Soft Gating

*This problem quantifies the variance of the gradient estimator under soft vs hard gating, establishing a formal bias-variance tradeoff that underlies the choice of routing mechanism.*

> **Prerequisites:** cf. note [[note#Hard Top-k Gating|§2 — Hard Top-k Gating]]; requires Problem 1

(a) Suppose expert outputs $f_i(x)$ are i.i.d. with mean $\mu$ and variance $\sigma^2$. Under soft gating with uniform weights $g_i = 1/E$, compute the variance of the Monte Carlo gradient estimator for $W_g$ due to stochasticity in expert outputs. Express your answer in terms of $E$ and $\sigma^2$.

(b) Under hard top-$k$ gating, only $k$ experts are selected. Repeat the calculation from (a) for the gradient estimator restricted to the $k$ selected experts. Show that the variance scales as $\sigma^2 / k$ and thus the variance is higher than soft gating when $k < E$.

(c) The noisy top-$k$ mechanism adds $\epsilon_i \sim \mathcal{N}(0, s_i^2)$ to logits before selection. Explain qualitatively how this noise introduces an additional variance term in the gradient, and identify the mechanism by which learned noise scale $s_i$ allows the model to trade off exploration vs gradient quality.

---

### Problem 4: Load Balancing Loss Derivation

*This problem derives the two auxiliary load-balancing losses of Shazeer et al. (2017) and the Switch Transformer auxiliary loss from first principles, establishing when each equals zero and characterizing its gradient.*

> **Prerequisites:** cf. note [[note#Importance Loss|§4 — Importance Loss]]; cf. note [[note#Simplified Auxiliary Loss|§5 — Simplified Auxiliary Loss]]

(a) Define $\operatorname{Importance}_i(\mathcal{B}) = \sum_{x \in \mathcal{B}} g_i(x)$ and $\mathcal{L}_{\text{imp}} = \operatorname{CV}(\operatorname{Importance}(\mathcal{B}))^2$. Show that $\mathcal{L}_{\text{imp}} = 0$ if and only if all experts have equal total gating weight. Show $\mathcal{L}_{\text{imp}} \geq 0$ always and that it is unbounded above.

(b) The Switch Transformer uses $\mathcal{L}_{\text{aux}} = \alpha E \sum_{i=1}^E f_i \cdot P_i$ where $f_i = \frac{1}{T}\sum_{x} \mathbf{1}[i^*(x) = i]$ and $P_i = \frac{1}{T}\sum_x g_i(x)$. Show that under uniform routing ($f_i = P_i = 1/E$ for all $i$), the loss evaluates to $\alpha$. Show that any concentration (large $f_i$ and $P_i$ for the same $i$) strictly increases $\mathcal{L}_{\text{aux}}$ above $\alpha$ via the Cauchy-Schwarz inequality.

(c) Compute $\partial \mathcal{L}_{\text{aux}} / \partial g_i(x)$ treating $f_i$ as a stop-gradient constant. Show that this gradient increases $g_i(x)$ for under-loaded experts and decreases it for over-loaded ones, directly counteracting the routing collapse feedback loop.

---

### Problem 5: Minimizers of the Importance and Auxiliary Losses

*This problem proves that both the Shazeer importance loss and the Switch auxiliary loss share the same global minimizer (uniform routing) but have different landscapes, with implications for the speed and stability of convergence.*

> **Prerequisites:** cf. note [[note#Load Loss|§4 — Load Loss]]; requires Problem 4

(a) Prove that the global minimum of $\mathcal{L}_{\text{imp}} = \operatorname{CV}(\operatorname{Importance}(\mathcal{B}))^2$ over all distributions of gate weights $\{g_i(x)\}_{x,i}$ with $\sum_i g_i(x) = 1$ is achieved exactly when $\operatorname{Importance}_i(\mathcal{B}) = T k / E$ for all $i$. State what distributional condition on the $g_i(x)$ guarantees this.

(b) For $\mathcal{L}_{\text{aux}}$ from Problem 4(b), prove that the minimum over all routing distributions (treating $f_i$ as determined by the argmax of gate weights) is achieved when $f_i = 1/E$ for all $i$. Use the AM-GM or Cauchy-Schwarz inequality to show $\sum_i f_i P_i \geq (\sum_i f_i \sqrt{P_i} / \sqrt{E})^2$ and conclude.

(c) Identify a routing configuration where $\mathcal{L}_{\text{aux}}$ is minimized ($f_i = P_i = 1/E$) but $\mathcal{L}_{\text{imp}}$ is not minimized. What does this imply about the relative strength of the two losses as load balancing incentives?

---

### Problem 6: Expert Capacity and Token Drop Probability

*This problem derives the drop probability under the capacity-factor mechanism, establishing the quantitative relationship between $\phi$, batch size $T$, and the fraction of lost tokens.*

> **Prerequisites:** cf. note [[note#Capacity Factor and Token Dropping|§4 — Capacity Factor and Token Dropping]]

(a) With $E$ experts, batch size $T$, top-$k$ routing, and capacity factor $\phi$, each expert has capacity $C = \lfloor (kT/E) \cdot \phi \rfloor$. Under perfectly uniform routing, show the expected token count per expert is exactly $kT/E$. For $\phi < 1$, compute the expected overflow fraction per expert.

(b) Assume token routing is i.i.d. with each of the $T$ tokens assigned to expert $i$ with probability $p = k/E$ (top-1 uniform routing). The load on expert $i$ follows $\operatorname{Binomial}(T, k/E)$. Using the normal approximation, derive an expression for the overflow probability $P(\text{Load}_i > C)$ as a function of $T$, $E$, $k$, and $\phi$.

(c) Analyze the asymptotic behavior of the overflow probability as $T \to \infty$ with $E$, $k$, $\phi$ fixed. Show that for $\phi > 1$ the overflow probability $\to 0$, and for $\phi = 1$ it $\to 1/2$. Conclude that $\phi > 1$ is necessary and sufficient for reliable zero-overflow behavior under uniform routing at large $T$.

---

### Problem 7: Capacity Factor Sensitivity

*This problem derives how the token drop rate changes as the capacity factor $\phi$ varies near $\phi = 1$, establishing the sensitivity of overflow to small changes in buffer size.*

> **Prerequisites:** cf. note [[note#Capacity Factor and Token Dropping|§4 — Capacity Factor and Token Dropping]]; requires Problem 6

(a) Using the normal approximation from Problem 6(b), write the drop probability as $P_{\text{drop}}(\phi) = \Phi\!\left(\frac{(\phi - 1)\sqrt{T k/E}}{\sqrt{(k/E)(1 - k/E)}}\right)^c$ for appropriate constants. Simplify and identify the rate parameter governing how fast $P_{\text{drop}}$ falls as $\phi$ increases from 1.

(b) Differentiate $P_{\text{drop}}(\phi)$ with respect to $\phi$ at $\phi = 1^+$. Show the sensitivity scales as $\sqrt{Tk/E}$ and interpret: for large batches $T$, a small increase in $\phi$ above 1 produces a large reduction in drop rate. What does this imply for the choice of $\phi$ in practice?

(c) For fixed total compute budget (i.e., $E \cdot C$ expert-token slots fixed), show that increasing $\phi$ above 1 while decreasing $E$ (fewer but larger experts) maintains the same total compute but changes the drop probability. Derive the $(\phi, E)$ tradeoff curve that holds total compute constant and characterize when reducing $E$ is preferable to increasing $\phi$.

---

### Problem 8: Birthday Problem Analogy for Token Overflow

*This problem establishes a formal analogy between the MoE token assignment problem and the classical birthday problem, yielding an exact expression for the probability that at least one expert overflows under random routing.*

> **Prerequisites:** cf. note [[note#Capacity Factor and Token Dropping|§4 — Capacity Factor and Token Dropping]]; cf. note [[note#The Load Imbalance Problem|§3 — The Load Imbalance Problem]]

(a) Consider top-1 routing where each of $T$ tokens is independently assigned uniformly at random to one of $E$ experts (so $k = 1$, $p = 1/E$). Capacity is $C = \lceil T/E \rceil$ (the ceiling of the fair share). Show that the event "at least one expert overflows" is exactly the complementary birthday problem: the probability that a random assignment of $T$ items into $E$ bins has at least one bin with more than $C$ items.

(b) For $E = T$ (one token per expert on average), compute the probability that at least one expert receives zero tokens using the inclusion-exclusion formula. Show this approaches $1 - e^{-1} \approx 0.632$ as $E = T \to \infty$ by analogy with the coupon-collector argument.

(c) For $E \ll T$ (many tokens per expert), use the Poisson approximation (each expert's load $\sim \operatorname{Poisson}(T/E)$) to derive the probability that expert $i$ overflows by more than $r$ tokens above capacity $C = T/E$. Show that for $r = \sqrt{T/E}$, the overflow probability decays as $e^{-r^2/2}$ to leading order.

---

### Problem 9: Expert-Choice Routing as Bipartite Matching

*This problem formalizes expert-choice routing as a constrained optimization problem on a bipartite graph, and establishes when the greedy expert-choice solution is globally optimal.*

> **Prerequisites:** cf. note [[note#Connection to Bipartite Matching|§6 — Connection to Bipartite Matching]]; cf. note [[note#Formal Specification|§6 — Formal Specification]]

(a) Formulate expert-choice routing as a bipartite $b$-matching integer linear program: tokens on one side, experts on the other; edge weight $s_{t,i} = g_i(x_t)$; each expert is matched to exactly $m$ tokens; each token may be matched to at most $k'$ experts. Write the full ILP.

(b) Show that the greedy expert-choice solution (each expert independently takes its top-$m$ tokens by score) solves the LP relaxation of the ILP from (a) when the per-token constraint is $k' = \infty$. Hence characterize the duality gap introduced by the integrality and per-token constraints.

(c) Define $\operatorname{Coverage} = \frac{1}{T}|\{t : \exists i,\ t \in \mathcal{B}_i\}|$. Show $\operatorname{Coverage} \leq 1$. Under a model where each expert independently selects each token with probability $m/T$, derive the expected coverage and identify the parameter regime where coverage is significantly less than 1.

---

### Problem 10: Expert-Choice Dual Formulation and LP Relaxation

*This problem derives the dual of the expert-choice LP, establishing the economic interpretation of routing as a market where tokens are priced by scarcity, and connects expert-choice to optimal transport.*

> **Prerequisites:** cf. note [[note#Connection to Bipartite Matching|§6 — Connection to Bipartite Matching]]; requires Problem 9

(a) Write the LP relaxation of the expert-choice ILP: maximize $\sum_{t,i} s_{t,i} x_{t,i}$ subject to $\sum_t x_{t,i} = m$ for all $i$ (expert capacity), $x_{t,i} \geq 0$. Derive its dual. Identify the dual variables as "prices" for each expert's capacity and interpret the complementary slackness conditions.

(b) Show that the optimal primal solution to this LP assigns each expert its top-$m$ tokens by score (the greedy expert-choice rule), and that the corresponding dual prices are given by the $(m+1)$-th highest score for each expert. Verify primal and dual feasibility.

(c) Identify this LP as an instance of the Kantorovich optimal transport problem with uniform marginals over experts and non-uniform marginals over tokens. State the Wasserstein distance interpretation of the optimal value and explain how Sinkhorn iterations could be used to approximately solve the doubly-constrained version (balanced token-expert assignment).

---

### Problem 11: MoE vs Ensemble vs Product of Experts

*This problem establishes the precise structural distinctions between MoE, static ensembles, and Product-of-Experts models — differences that become mathematically sharp when one analyzes the gradient signal each architecture sends to individual sub-networks.*

> **Prerequisites:** cf. note [[note#Relation to Ensemble Methods|§2 — Relation to Ensemble Methods]]

(a) A committee ensemble computes $y = \frac{1}{E}\sum_i f_i(x)$; MoE computes $y = \sum_i g_i(x) f_i(x)$ with $g_i(x)$ input-dependent. Show that MoE reduces to a committee when $g_i(x) = 1/E$ for all $x$. Prove that in a committee, the gradient of $\mathcal{L}$ with respect to $W_i$ (expert $i$'s weights) is independent of all other experts' outputs, whereas in MoE it is not.

(b) The Product of Experts defines $p(y|x) \propto \prod_i p_i(y|x)^{g_i(x)}$. Taking log, the log-probability of a Gaussian PoE is a weighted average of log-normalizing constants minus a precision-weighted mean. Show that for Gaussian experts $p_i(y|x) = \mathcal{N}(\mu_i(x), \sigma^2)$, the PoE predictive mean is not $\sum_i g_i(x)\mu_i(x)$ — contrast with MoE. Describe qualitatively how PoE vs MoE handles the case where two experts strongly disagree.

(c) Under top-1 MoE routing, define the region $\mathcal{R}_i = \{x : i = \arg\max_j (W_g^\top x)_j\}$. Show these regions form a polyhedral partition of $\mathbb{R}^d$ determined by the hyperplanes $\{x : (W_g)_i^\top x = (W_g)_j^\top x\}$ for pairs $(i,j)$. Interpret this as a Voronoi diagram in the logit space and state what it implies about the geometric structure of expert specialization.

---

### Problem 12: Top-1 Routing and Voronoi Specialization

*This problem proves that the partitioning of the input space under top-1 routing is a linear Voronoi partition, and derives formal conditions under which the router's learned partition aligns with the optimal partition given the experts' functional forms.*

> **Prerequisites:** cf. note [[note#Top-1 Routing|§5 — Top-1 Routing]]; requires Problem 11

(a) Let $W_g \in \mathbb{R}^{d \times E}$ be fixed. Prove that the regions $\mathcal{R}_i = \{x : (W_g^\top x)_i > (W_g^\top x)_j \ \forall j \neq i\}$ are convex polyhedral cones in $\mathbb{R}^d$.

(b) A natural oracle router assigns token $x$ to the expert $i^*$ that minimizes loss: $i^*(x) = \arg\min_i \mathcal{L}(f_i(x), y)$. Derive the condition under which the linear router $W_g$ can perfectly recover this oracle assignment. When is recovery impossible with a linear router?

(c) Suppose the $E$ experts are linear: $f_i(x) = W_i x$. Show that the optimal routing partition (minimizing the expected loss over a dataset) is indeed a linear Voronoi partition in a suitable feature space, confirming that the linear router $W_g$ has sufficient expressive power to represent the optimal routing when experts are linear.

---

### Problem 13: Expert Collapse as a Fixed-Point Instability

*This problem formalizes the expert collapse positive feedback loop as a discrete-time dynamical system and derives the stability condition separating healthy specialization from degenerate collapse.*

> **Prerequisites:** cf. note [[note#Expert Collapse|§8 — Expert Collapse]]

(a) Model the routing dynamics as follows: at step $t$, expert $i$ has quality $q_i^{(t)}$ (measured by negative loss on tokens it receives). The router assigns fraction $\rho_i^{(t)} = \operatorname{softmax}(\beta q^{(t)})_i$ of tokens to expert $i$, where $\beta > 0$ is a temperature. Expert quality updates as $q_i^{(t+1)} = q_i^{(t)} + \eta \rho_i^{(t)} \delta_i$, where $\delta_i > 0$ is a fixed per-token improvement rate and $\eta$ is the learning rate. Write the fixed-point equations for this system.

(b) Linearize the dynamics around the uniform fixed point $\rho_i = 1/E$, $q_i = q^*$ for all $i$. Derive the Jacobian of the update map and find its eigenvalues. Show that the uniform fixed point is unstable when $\eta \beta \delta / E > 1/(E-1)$ for any perturbation that breaks symmetry.

(c) Interpret the instability condition from (b): which factors (large $\beta$, large $\eta$, large $\delta$) drive collapse, and which (large $E$) provide stability through diversification? Connect this to the practical recommendations for router initialization (small initial logit magnitude) and learning rate warmup described in the note.

---

### Problem 14: Router Z-Loss Gradient Analysis

*This problem computes the gradient of the router z-loss and proves that it applies the strongest shrinkage to the dominant expert, thereby directly counteracting the collapse feedback loop.*

> **Prerequisites:** cf. note [[note#Router Z-Loss|§8 — Router Z-Loss]]

(a) Define $\mathcal{L}_z = \frac{1}{T}\sum_{t=1}^T \left(\log \sum_i e^{h_i(x_t)}\right)^2$. Compute $\partial \mathcal{L}_z / \partial h_i(x_t)$ and show it equals $\frac{2}{T} \cdot \operatorname{LSE}(h(x_t)) \cdot \operatorname{softmax}(h(x_t))_i$.

(b) Show that the LSE satisfies $\max_i h_i \leq \operatorname{LSE}(h) \leq \max_i h_i + \log E$. Use this to bound the magnitude of $\partial \mathcal{L}_z / \partial h_i$ from below for the dominant expert (the one with the largest logit), and show this bound grows with the logit magnitude — confirming the self-correcting nature of z-loss.

(c) Compare the gradient directions of z-loss and the Switch auxiliary loss $\mathcal{L}_{\text{aux}}$ with respect to the routing logits $h_i$. Show that both apply downward pressure on over-represented experts, but z-loss acts on the magnitude of logits while $\mathcal{L}_{\text{aux}}$ acts on the relative distribution of softmax probabilities.

---

### Problem 15: Entropy Regularization vs Z-Loss

*This problem derives the gradient of entropy regularization on the routing distribution, compares it formally with the z-loss gradient, and establishes when each is a stronger regularizer.*

> **Prerequisites:** cf. note [[note#Entropy Regularization|§8 — Entropy Regularization]]; requires Problem 14

(a) Define the routing entropy $H(g(x)) = -\sum_i g_i(x) \log g_i(x)$. Compute $\partial(-H(g(x)))/\partial h_i(x)$ and show it equals $g_i(x)(\log g_i(x) + H(g(x)))$. Identify the sign of this gradient for an expert with above-average weight.

(b) Compare the gradients from (a) and Problem 14(a). Show that entropy regularization applies pressure proportional to $g_i \log g_i$, while z-loss applies pressure proportional to $\operatorname{LSE}(h) \cdot g_i$. Identify the regime (sharp vs flat routing distribution) where each is relatively stronger.

(c) Entropy regularization operates on the post-softmax distribution $g$, while z-loss operates on the pre-softmax logits $h$. Show that z-loss can be computed as a side effect of the log-sum-exp already computed during numerically stable softmax, while entropy requires an extra $\log g_i$ pass. Conclude which is cheaper to add to an existing routing implementation.

---

### Problem 16: MoE Scaling Laws vs Dense Scaling Laws

*This problem derives the FLOP-parameter decoupling argument for MoE models and formalizes the condition under which MoE achieves better loss than a FLOP-matched dense baseline.*

> **Prerequisites:** cf. note [[note#Scaling Laws for MoE|§7 — Scaling Laws for MoE]]; cf. note [[note#FLOP-Matched Comparisons|§7 — FLOP-Matched Comparisons]]

(a) A dense model with $N$ parameters costs $C \propto N$ FLOPs per forward pass. An MoE model with $E$ experts of size $N_e$ each uses top-$k$ routing, giving active parameters $k N_e$ and total parameters $E N_e$. For the two models to be FLOP-matched, we need $k N_e = N$. Express the ratio of total MoE parameters to dense parameters as $E/k$ and interpret: under what routing sparsity does MoE hold a $10\times$ parameter advantage?

(b) The MoE model achieves lower loss than the FLOP-matched dense model if and only if experts genuinely specialize. Formalize this: define "redundant experts" as those satisfying $f_i(x) \approx f_j(x)$ for all $x$. Show that if all experts are redundant, the MoE collapses to the performance of a single expert, recovering the FLOP-matched dense baseline.

(c) Empirically, the MoE advantage narrows at very large scale. Propose a hypothesis grounded in the data distribution: argue that as the dense model grows, it enters a data-limited regime where additional parameters (MoE or dense) have diminishing returns. Formalize by sketching a loss curve $L(N, D)$ (the Chinchilla scaling law form) and explain why the MoE advantage $\Delta L = L(kN_e, D) - L_{\text{MoE}}$ shrinks when $D$ is the binding constraint rather than $N$.

---

### Problem 17: Soft-Gating Fixed-Point Stability

*This problem analyzes the joint training dynamics of soft-gated MoE as a coupled dynamical system for gate and expert parameters, and derives the conditions under which expert specialization is a stable fixed point.*

> **Prerequisites:** cf. note [[note#Soft Gating|§2 — Soft Gating]]; cf. note [[note#Expert Collapse|§8 — Expert Collapse]]; requires Problem 13

(a) Consider a two-expert soft MoE with experts $f_1, f_2$ and gate $g_1 = \sigma(w^\top x)$, $g_2 = 1 - g_1$, where $\sigma$ is the sigmoid and $w \in \mathbb{R}^d$ is the gating parameter. Show that the gradient of $\mathcal{L}$ with respect to $w$ is proportional to $(f_1(x) - f_2(x))$ times a factor involving $g_1(1 - g_1)$.

(b) At a fixed point where $g_1(x) = g_2(x) = 1/2$ for all $x$, the two experts receive equal weight. Show this fixed point is neutrally stable: any small perturbation in $w$ causes a small nonzero gradient, but does not grow. Identify the condition on $(f_1, f_2)$ under which the fixed point becomes attractive vs repulsive.

(c) Show that a fully specialized fixed point, where $g_1(x) = 1$ for $x \in \mathcal{R}_1$ and $g_2(x) = 0$ for $x \notin \mathcal{R}_1$, is a fixed point of the gradient dynamics. Under what conditions on the task loss $\mathcal{L}$ and expert functions $f_1, f_2$ is this specialized configuration a local minimum?

---

### Problem 18: Effective Parameter Count and Routing Efficiency

*This problem formalizes the "effective parameter count" of an MoE model and proves that it is strictly bounded between the active and total parameter counts, with the bound depending on how well the routing concentrates tokens.*

> **Prerequisites:** cf. note [[note#Scaling Laws for MoE|§7 — Scaling Laws for MoE]]; requires Problem 16

(a) Define the effective parameter count $N_{\text{eff}}$ of an MoE model as the parameter count of the dense model achieving the same validation loss. Given a MoE with $E$ experts, top-$k$ routing, and $N_e$ parameters per expert ($EN_e$ total, $kN_e$ active), argue that $kN_e \leq N_{\text{eff}} \leq EN_e$, with equality at the left when experts are redundant and equality at the right when routing is perfectly oracle-optimal.

(b) Define routing efficiency $\eta = (N_{\text{eff}} - kN_e) / ((E-k)N_e) \in [0,1]$. Show that $\eta = 0$ corresponds to all experts being identical copies and $\eta = 1$ corresponds to fully non-overlapping expert specializations. Express $\eta$ as a function of the mutual information between the routing decision and the expert's output, assuming a simple Gaussian expert model.

(c) For the Mixtral 8x7B model ($E = 8$, $k = 2$, total 46.7B parameters, active 12.9B), if the model outperforms a FLOP-matched 13B dense model and underperforms a 46.7B dense model, bound $\eta$ and state what this implies about the specialization of Mixtral's experts.

---

## Algorithmic Applications

### Problem 19: Noisy Top-k Gating Forward Pass

*This problem designs the full forward pass for noisy top-k gating as introduced by Shazeer et al. (2017), including numerically stable sparse softmax and the bookkeeping required for the backward pass.*

> **Prerequisites:** cf. note [[note#Noisy Top-k Gating|§3 — Noisy Top-k Gating]]

(a) **Inputs and data structures**: Specify the inputs ($x \in \mathbb{R}^d$, $W_g \in \mathbb{R}^{d \times E}$, $W_n \in \mathbb{R}^{d \times E}$, $k \in \mathbb{Z}_{>0}$) and list all intermediate tensors that must be stored for the backward pass.

(b) **Noisy logit computation**: Write pseudocode that: (i) computes clean logits $z = W_g^\top x$; (ii) computes learned noise scales $s = \operatorname{Softplus}(W_n^\top x)$; (iii) samples $\epsilon \sim \mathcal{N}(0, I_E)$; (iv) forms noisy logits $H = z + \epsilon \odot s$. Annotate shapes.

(c) **Top-k selection and sparse softmax**: Write pseudocode applying KeepTopK to $H$ and computing the softmax over non-$-\infty$ entries only. Specify the numerical mask required to avoid NaN from $e^{-\infty}$ in the denominator.

(d) **Expert dispatch and aggregation**: Write pseudocode dispatching $x$ to the $k$ selected experts, collecting their outputs $\{f_i(x)\}_{i \in \mathcal{T}}$, and computing the weighted sum $y = \sum_{i \in \mathcal{T}} g_i(x) f_i(x)$. State what must be cached for the backward pass and give the time complexity in terms of $E$, $k$, and $d$.

---

### Problem 20: Expert Dispatch with Capacity Enforcement

*This problem designs the token dispatch algorithm for a single MoE layer with capacity-enforced top-1 routing on a batch, including the padding and masking needed for efficient batched expert computation.*

> **Prerequisites:** cf. note [[note#Capacity Factor and Token Dropping|§4 — Capacity Factor and Token Dropping]]; cf. note [[note#Top-1 Routing|§5 — Top-1 Routing]]

(a) **Routing**: Describe how to compute routing scores $S \in \mathbb{R}^{T \times E}$ and produce an assignment index tensor of shape $(T,)$ for top-1 routing. Specify how tie-breaking is handled.

(b) **Capacity enforcement**: For each expert $i$, describe how to sort its assigned tokens by routing score (descending), retain the first $C$, and mark the remainder as dropped. Write pseudocode for this step and state its time complexity.

(c) **Dispatch, compute, and scatter**: Write pseudocode to gather kept tokens per expert into a padded tensor of shape $(E, C, d)$, apply each expert's FFN, and scatter outputs back to token positions, writing zeros for dropped tokens. Annotate all tensor shapes.

(d) Explicitly identify where gradients flow and where they are blocked. Do dropped tokens contribute gradient to the router weights? Do they contribute gradient to the expert FFN weights? Propose a correction term to reduce the gradient bias introduced by dropped tokens.

---

### Problem 21: Router Z-Loss Numerically Stable Implementation

*This problem designs a numerically stable forward-backward implementation of the router z-loss, exploiting the fact that the log-sum-exp value is already computed during the softmax routing step.*

> **Prerequisites:** cf. note [[note#Router Z-Loss|§8 — Router Z-Loss]]

(a) **Numerically stable forward pass**: Write pseudocode for computing $\mathcal{L}_z = \frac{1}{T}\sum_t (\operatorname{LSE}(h(x_t)))^2$ using the max-subtract trick: $\operatorname{LSE}(h) = m + \log\sum_i e^{h_i - m}$, $m = \max_i h_i$. Annotate intermediate shapes.

(b) **Efficient backward pass**: Show that $\partial \mathcal{L}_z / \partial h_i(x_t) = \frac{2}{T} \cdot \operatorname{LSE}(h(x_t)) \cdot \operatorname{softmax}(h(x_t))_i$. Write pseudocode for the backward pass that reuses the cached LSE values and softmax probabilities from the routing computation, avoiding any redundant passes over the logits.

(c) **Combined loss assembly**: Write pseudocode for the full objective $\mathcal{L} = \mathcal{L}_{\text{task}} + \alpha \mathcal{L}_{\text{aux}} + \beta \mathcal{L}_z$. State the typical values of $\alpha$ and $\beta$ from Switch Transformer and ST-MoE respectively, and explain the tradeoff each hyperparameter controls.

---

### Problem 22: Expert-Choice Routing Forward Pass

*This problem designs the forward pass for expert-choice routing, emphasizing the structural differences from token-choice routing in terms of dispatch, coverage tracking, and handling of unselected tokens.*

> **Prerequisites:** cf. note [[note#Formal Specification|§6 — Formal Specification]]; cf. note [[note#Guaranteed Load Balance|§6 — Guaranteed Load Balance]]

(a) **Inputs and score matrix**: Specify the inputs and describe how to compute the token-expert score matrix $S \in \mathbb{R}^{T \times E}$ using the routing softmax. Annotate shapes.

(b) **Expert-side top-m selection**: For each expert $i$, describe how to select the top-$m$ tokens by column $S_{:,i}$ and produce the selection tensor $\mathcal{B}_i \subseteq [T]$ with $|\mathcal{B}_i| = m$. Write pseudocode and give the complexity. Confirm that this step guarantees $|\mathcal{B}_i| = m$ for all $i$ without any auxiliary loss.

(c) **Output assembly and coverage**: Write pseudocode for computing the output $y_t = \sum_{i : t \in \mathcal{B}_i} g_i(x_t) f_i(x_t)$ for each token. Handle the case where token $t$ is selected by zero experts (set $y_t = 0$ or pass through residual). Compute the batch coverage $\operatorname{Coverage} = |\{t : y_t \neq 0\}| / T$ and state its expected value as a function of $m$, $E$, $T$.

(d) **Gradient flow**: Identify which quantities receive gradient from this forward pass. Does the routing score $S_{t,i}$ for an unselected $(t,i)$ pair receive gradient? Contrast with token-choice routing and discuss the implication for router training.

---

### Problem 23: Distributed Expert Parallelism Protocol

*This problem designs the all-to-all communication protocol for expert-parallel MoE training and derives the communication volume, enabling comparison with tensor-parallel alternatives.*

> **Prerequisites:** cf. note [[note#The Load Imbalance Problem|§3 — The Load Imbalance Problem]]

(a) **Setup**: Assume $E$ experts, each on a separate device, processing a batch of $T$ tokens with hidden dimension $d$ using top-1 routing. Each device initially holds $T/E$ tokens (data parallel). Describe the two all-to-all collectives required per MoE layer and bound the bytes communicated per device per collective in terms of $T$, $d$, and $E$.

(b) **Comparison with tensor parallelism**: For a two-device tensor-parallel split of a linear layer $W \in \mathbb{R}^{d \times d}$ applied to a batch of $T$ tokens, derive the bytes communicated per device per layer. Identify the regime in $(T, d, E)$ space where expert parallelism dominates vs tensor parallelism in communication volume.

(c) **Token dropping in distributed training**: In expert parallelism with capacity $C < T/E$ (aggressive capacity), some tokens are dropped. Explain why dropped tokens introduce a systematic gradient bias on expert parameters. Write pseudocode for a correction that re-weights the loss of kept tokens to compensate for dropped ones, and state whether this correction fully eliminates the bias or merely reduces it.
