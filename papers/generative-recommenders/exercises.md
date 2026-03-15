# Exercises: Wukong & HSTU

## Table of Contents

- [[#Mathematical Development|Mathematical Development]]
  - [[#Problem 1 FM Pairwise Interaction as a Gram Matrix|Problem 1: FM Pairwise Interaction as a Gram Matrix]]
  - [[#Problem 2 Low-Rank FM Projection and Associativity|Problem 2: Low-Rank FM Projection and Associativity]]
  - [[#Problem 3 Complexity Comparison: Naive vs. Optimized FM|Problem 3: Complexity Comparison: Naive vs. Optimized FM]]
  - [[#Problem 4 Interaction Order: Base Case and Inductive Step|Problem 4: Interaction Order: Base Case and Inductive Step]]
  - [[#Problem 5 Exponential vs. Linear Order Growth|Problem 5: Exponential vs. Linear Order Growth]]
  - [[#Problem 6 LCB Does Not Raise Interaction Order|Problem 6: LCB Does Not Raise Interaction Order]]
  - [[#Problem 7 Residual and LCB as Complementary Linear Pathways|Problem 7: Residual and LCB as Complementary Linear Pathways]]
  - [[#Problem 8 Total Complexity of the Interaction Stack|Problem 8: Total Complexity of the Interaction Stack]]
  - [[#Problem 9 Impression-Level vs. Generative Training Complexity|Problem 9: Impression-Level vs. Generative Training Complexity]]
  - [[#Problem 10 Stochastic Length Truncation: Expected Cost|Problem 10: Stochastic Length Truncation: Expected Cost]]
  - [[#Problem 11 Stochastic Length Probability Calibration|Problem 11: Stochastic Length Probability Calibration]]
  - [[#Problem 12 M-FALCON Amortization Analysis|Problem 12: M-FALCON Amortization Analysis]]
  - [[#Problem 13 Softmax Normalization and Non-Stationarity|Problem 13: Softmax Normalization and Non-Stationarity]]
  - [[#Problem 14 Pointwise Attention as an Unnormalized Kernel|Problem 14: Pointwise Attention as an Unnormalized Kernel]]
  - [[#Problem 15 Temporal Relative Bias and Sequence-Length Invariance|Problem 15: Temporal Relative Bias and Sequence-Length Invariance]]
  - [[#Problem 16 Supervision Density and the Scaling Law|Problem 16: Supervision Density and the Scaling Law]]
  - [[#Problem 17 Power-Law Exponent Stability|Problem 17: Power-Law Exponent Stability]]
  - [[#Problem 18 Memory Footprint: HSTU vs. Standard Transformer|Problem 18: Memory Footprint: HSTU vs. Standard Transformer]]
- [[#Algorithmic Applications|Algorithmic Applications]]
  - [[#Problem 19 Optimized FM Block: Pseudocode and Shape Annotations|Problem 19: Optimized FM Block: Pseudocode and Shape Annotations]]
  - [[#Problem 20 HSTU Forward Pass: Pseudocode and Complexity|Problem 20: HSTU Forward Pass: Pseudocode and Complexity]]
  - [[#Problem 21 Stochastic Length Sampling: Implementation Sketch|Problem 21: Stochastic Length Sampling: Implementation Sketch]]
  - [[#Problem 22 M-FALCON Attention Mask Construction|Problem 22: M-FALCON Attention Mask Construction]]
  - [[#Problem 23 Ablation Study: Gradient Flow Through LCB and Residual|Problem 23: Ablation Study: Gradient Flow Through LCB and Residual]]

---

## Mathematical Development

### Problem 1: FM Pairwise Interaction as a Gram Matrix

*The Factorization Machine Block (FMB) computes all pairwise inner products via a Gram matrix. This problem establishes that the resulting matrix is positive semidefinite, characterizes its rank, and derives a closed-form expression for the sum of all pairwise interactions from the FM identity.*

> **Prerequisites:** cf. note [[wukong#2.2 Factorization Machine Block|§2.2 — Factorization Machine Block]]

(a) Let $X \in \mathbb{R}^{n \times d}$. The FM interaction matrix is $G = X X^\top \in \mathbb{R}^{n \times n}$. Prove that $G$ is positive semidefinite (PSD) and that $\text{rank}(G) \leq \min(n, d)$.

(b) The standard FM scalar output for a feature matrix $X$ is
$$\hat{y}_{\text{FM}} = \sum_{j < k} \langle \mathbf{x}_j, \mathbf{x}_k \rangle.$$
Using the identity $\|\sum_j \mathbf{x}_j\|^2 = \sum_j \|\mathbf{x}_j\|^2 + 2\sum_{j<k}\langle \mathbf{x}_j, \mathbf{x}_k \rangle$, derive a formula for $\hat{y}_{\text{FM}}$ that avoids explicitly computing $G$, and state its complexity.

(c) Prove that the naive computation of all $\binom{n}{2}$ pairwise scores via $G = X X^\top$ followed by reading off off-diagonal entries has complexity $O(n^2 d)$. Contrast with the complexity of the formula from part (b).

---

### Problem 2: Low-Rank FM Projection and Associativity

*The optimized FM in Wukong projects the full $n \times n$ Gram matrix to rank $k$ via a right-multiplication $Y \in \mathbb{R}^{n \times k}$. This problem proves the key associativity that reduces complexity, and characterizes what information the projection retains.*

> **Prerequisites:** cf. note [[wukong#2.2 Factorization Machine Block|§2.2 — Factorization Machine Block]]

(a) The optimized FM computes $\text{FM}_{\text{opt}}(X) = X X^\top Y$ with $Y \in \mathbb{R}^{n \times k}$, $k \ll n$. Show that by evaluating right-to-left as $X(X^\top Y)$, the total operation has complexity $O(ndk)$ rather than $O(n^2 d + n^2 k)$. State all intermediate tensor shapes.

(b) Interpret $\text{FM}_{\text{opt}}(X)$ as projecting each row of the Gram matrix $G$ into a $k$-dimensional subspace. Prove that if $Y$ has orthonormal columns, then $\text{FM}_{\text{opt}}(X) \text{FM}_{\text{opt}}(X)^\top = G Y Y^\top G^\top$, and show this equals $G P_Y G$ where $P_Y = YY^\top$ is the projection onto the column space of $Y$.

(c) Suppose $Y$ is drawn randomly with i.i.d. $\mathcal{N}(0, 1/k)$ entries. Using the Johnson-Lindenstrauss lemma (state it), argue informally that for $k = O(\log n / \epsilon^2)$, the pairwise distances between rows of $\text{FM}_{\text{opt}}(X)$ approximate those of $G$ with relative error $\epsilon$.

---

### Problem 3: Complexity Comparison: Naive vs. Optimized FM

*This problem derives the complexity of the full Wukong FMB pipeline, tracking the compute cost of each stage and showing that the bottleneck shifts from the FM step to the subsequent MLP.*

> **Prerequisites:** cf. note [[wukong#4. Complexity Analysis|§4 — Complexity Analysis]]

(a) Write out the complete sequence of operations in FMB$(X_i)$ with $X_i \in \mathbb{R}^{n \times d}$, projection rank $k$, and MLP hidden size $h$. For each step (FM, flatten, layer norm, MLP, reshape), state the operation, its input/output shapes, and its FLOPs cost.

(b) Sum the costs from (a) and show the total is $O(ndk + n^2 k + nkh + h^2 + n_F dh)$. Under the assumption $k = O(n)$ and $n_F, n_L = O(n)$, simplify to the form stated in the note: $O(ndh + h^2)$.

(c) Compare the total cost per layer of Wukong to the naive FM baseline (which requires $O(n^2 d)$ for the FM alone). For what value of $k$ does the optimized FM cost equal the naive FM cost? Derive the break-even condition.

---

### Problem 4: Interaction Order: Base Case and Inductive Step

*The exponential interaction order theorem is the core theoretical contribution of Wukong. This problem asks you to make the inductive proof fully rigorous by formalizing the notion of "interaction order" and verifying each component of the update.*

> **Prerequisites:** cf. note [[wukong#3. Theory: Exponential Interaction Orders|§3 — Theory: Exponential Interaction Orders]]

(a) Define formally: an embedding $\mathbf{v} \in \mathbb{R}^d$ has *interaction order* at most $p$ if every component $v_j$ is a polynomial of degree at most $p$ in the raw input features $\{x_{i,j}\}_{i,j}$. Using this definition, prove the base case: $X_0$ has order 1.

(b) Under the hypothesis that $X_i$ contains embeddings of orders in $\{1, \ldots, 2^i\}$, prove that $\text{LCB}(X_i)$ has order at most $2^i$. (A linear combination of degree-$p$ polynomials has degree at most $p$.)

(c) Under the same hypothesis, prove that the FM output $G_i = X_i X_i^\top$ contains entries of order at most $2^{i+1}$. Explicitly identify the maximizing pair of rows and their orders.

(d) Explain why the MLP inside FMB does not raise the interaction order beyond $2^{i+1}$. What assumption on the MLP is required for this claim to hold? (Hint: consider what happens if the MLP is a degree-2 polynomial network rather than a ReLU network.)

---

### Problem 5: Exponential vs. Linear Order Growth

*DCNv2 and xDeepFM achieve at most linear interaction order growth with depth. This problem quantifies the gap, derives how many Wukong layers are needed to match a given DCNv2 depth, and shows that the parameter count grows much more slowly for Wukong to reach a fixed target order.*

> **Prerequisites:** cf. note [[wukong#3.2 Inductive Proof|§3.2 — Inductive Proof]]; requires Problem 4

(a) DCNv2 with $l$ layers achieves at most interaction order $l+1$. Wukong with $l$ layers achieves order $2^l$. Derive the number of Wukong layers $l_W$ needed to match or exceed the order achieved by $l_D$ DCNv2 layers, and simplify to $l_W = \lceil \log_2(l_D + 1) \rceil$.

(b) Let the parameter count per layer be $\Theta$ for both models (assume equal per-layer cost). To reach interaction order $q$, DCNv2 requires $q - 1$ layers and Wukong requires $\lceil \log_2 q \rceil$ layers. Compute the ratio of total parameter counts as a function of $q$, and show that for $q = 256$, Wukong uses $12.5\times$ fewer layers.

(c) Consider the following claim: "The exponential growth of interaction order means Wukong must be expressive enough to overfit any training dataset after a logarithmic number of layers." Evaluate this claim mathematically. Does interaction order control generalization? What quantity does it control?

---

### Problem 6: LCB Does Not Raise Interaction Order

*The Linear Compression Block (LCB) applies a linear map to the embedding matrix. This problem proves that linear operations cannot generate new polynomial interactions, making the LCB safe to include without disrupting the inductive order argument.*

> **Prerequisites:** cf. note [[wukong#2.3 Linear Compression Block|§2.3 — Linear Compression Block]]; requires Problem 4

(a) Let $f: \mathbb{R}^{n \times d} \to \mathbb{R}^{m \times d}$ be defined by $f(X) = WX$ for $W \in \mathbb{R}^{m \times n}$. Prove that if each row of $X$ has interaction order at most $p$, then each row of $f(X)$ has interaction order at most $p$.

(b) Contrast the LCB with a hypothetical "Quadratic Compression Block" $f_Q(X)_{ij} = \sum_{k,l} W_{ij,kl} X_{ik} X_{jl}$. What interaction order would $f_Q(X)$ introduce if $X$ has order $p$? Does adding a QCB in place of the LCB break the exponential order theorem?

(c) The residual update is $X_{i+1} = \text{LN}(\text{concat}(\text{FMB}_i(X_i), \text{LCB}_i(X_i)) + X_i)$. Layer normalization normalizes each feature embedding to zero mean and unit variance. Does layer normalization raise the interaction order? Justify your answer carefully.

---

### Problem 7: Residual and LCB as Complementary Linear Pathways

*The ablation study shows that removing both LCB and residual causes substantially more degradation than removing either alone. This problem formalizes the "complementary linear pathways" argument in terms of gradient flow.*

> **Prerequisites:** cf. note [[wukong#6.3 Ablation Studies|§6.3 — Ablation Studies]]; requires Problem 6

(a) Consider a simplified two-layer stack where $X_1 = F(X_0) + L(X_0) + X_0$, with $F$ nonlinear (FMB) and $L$ linear (LCB). Compute $\partial X_1 / \partial X_0$ and identify the terms that survive when $F$ is saturated (i.e., $\partial F / \partial X_0 \approx 0$).

(b) Extend to an $l$-layer stack. Let $G_i = \partial X_i / \partial X_{i-1}$. Express the gradient $\partial \mathcal{L} / \partial X_0$ as a product $\prod_{i=1}^l G_i \cdot \partial \mathcal{L} / \partial X_l$. Under the approximation $\partial F_i / \partial X_{i-1} \approx 0$, show that the gradient product reduces to a sum of paths through LCB and residual matrices only.

(c) Suppose the LCB is removed but the residual remains. How does your answer to (b) change? Suppose both are removed. What is $\partial \mathcal{L} / \partial X_0$ in that case, assuming $\partial F_i / \partial X_{i-1} \approx 0$? Relate this to the observed "substantial degradation" in the ablation.

---

### Problem 8: Total Complexity of the Interaction Stack

*Wukong achieves near-linear complexity in $n$ (with a $\log n$ factor) despite generating $2^l$-order interactions. This problem derives the total cost across all layers and shows how the $O(ndh \log n + h^2)$ bound arises.*

> **Prerequisites:** cf. note [[wukong#4. Complexity Analysis|§4 — Complexity Analysis]]; requires Problem 3

(a) Let the number of layers be $l$. Assume $n_i \approx n' = n_F + n_L$ is constant across all layers (embedding count is stabilized by the LCB). Using the per-layer cost from Problem 3, write down the total cost across $l$ layers.

(b) To reach interaction order $q = 2^l$ from depth-1 order, we need $l = \log_2 q$ layers. Substitute this into the per-layer cost and show that the total across $l = \log_2 n$ layers (to cover order up to $n$) gives $O(ndh \log n + h^2 \log n)$. Under the assumption $h^2 \gg ndh$, simplify to $O(ndh \log n + h^2)$.

(c) How does this compare to the cost of a naive all-order FM that explicitly computes all interactions up to order $q$ via a complete polynomial kernel? (A degree-$q$ polynomial kernel on $\mathbb{R}^n$ has $\binom{n+q}{q}$ monomials.) Why is Wukong's cost dramatically lower, and what is the tradeoff?

---

### Problem 9: Impression-Level vs. Generative Training Complexity

*HSTU's generative training reduces the per-user complexity from $O(N^3 d)$ (impression-level) to $O(N^2 d)$. This problem re-derives this reduction from first principles and characterizes when the saving is most pronounced.*

> **Prerequisites:** cf. note [[hstu#4.1 Generative Training vs. Impression-Level Training|§4.1 — Generative Training vs. Impression-Level Training]]

(a) Suppose a user has $n_i$ historical interactions. In impression-level training, each interaction generates one training example requiring a full attention-based forward pass over $n_i$ tokens. Derive the total FLOPs for all $n_i$ passes, counting only the self-attention cost $O(n^2 d)$ per pass.

(b) In generative training with sampling rate $s_u(n_i) = 1/n_i$, a single forward pass over the full sequence of length $n_i$ produces $n_i$ supervision signals but only $s_u(n_i) \cdot n_i = 1$ target per step is used. Show that the expected FLOPs per user per training step is $O(n_i^2 d)$, a factor of $n_i$ cheaper than impression-level.

(c) Let the user sequence length follow a power-law distribution $P(n_i = n) \propto n^{-\gamma}$ with $\gamma > 1$ and support $[1, N]$. Compute the expected cost per user under both paradigms as a function of $N$ and $\gamma$. For $\gamma = 2$ (a realistic exponent for user activity), how does the expected cost ratio scale with $N$?

---

### Problem 10: Stochastic Length Truncation: Expected Cost

*Stochastic length (SL) truncation controls the per-user attention cost by randomly retaining full sequences with probability proportional to the budget. This problem derives the expected cost under SL and verifies the claimed $O(N^\alpha d)$ bound.*

> **Prerequisites:** cf. note [[hstu#4.2 Stochastic Length Sparsity|§4.2 — Stochastic Length Sparsity]]

(a) For a user with sequence length $n_i > N^{\alpha/2}$, the SL scheme retains the full sequence with probability $p_{\text{full}} = N^\alpha / n_i^2$ and truncates to length $N^{\alpha/2}$ with probability $1 - N^\alpha / n_i^2$. Let the attention cost for a sequence of length $m$ be $c \cdot m^2 d$. Compute the expected cost $\mathbb{E}[\text{Cost}_i]$ under SL.

(b) Show that $\mathbb{E}[\text{Cost}_i] = O(N^\alpha d)$ for all $n_i \geq 1$. Verify that this bound is also achieved (trivially) for $n_i \leq N^{\alpha/2}$.

(c) The bound in (b) applies per user. Suppose there are $M$ users, each with i.i.d. length $n_i \sim \text{Power-law}(N, \gamma)$. What is the total expected training cost? Compare to the impression-level cost from Problem 9 and quantify the improvement for $N = 10^4$, $\alpha = 1.5$.

---

### Problem 11: Stochastic Length Probability Calibration

*The probability $p_{\text{full}} = N^\alpha / n_i^2$ is a specific design choice. This problem derives the constraint it must satisfy to achieve the target complexity, and shows that no other functional form achieves $O(N^\alpha d)$ without sacrificing unbiasedness.*

> **Prerequisites:** cf. note [[hstu#4.2 Stochastic Length Sparsity|§4.2 — Stochastic Length Sparsity]]; requires Problem 10

(a) Let $p(n_i)$ be the probability of retaining the full sequence for a user of length $n_i > N^{\alpha/2}$. For the expected cost $\mathbb{E}[\text{Cost}_i] = p(n_i) \cdot n_i^2 d + (1 - p(n_i)) \cdot N^\alpha d \leq C \cdot N^\alpha d$ for some constant $C$, derive the constraint on $p(n_i)$.

(b) Show that $p(n_i) = N^\alpha / n_i^2$ is the unique solution (up to constants) satisfying the constraint from (a) with equality, i.e., it makes both terms contribute $O(N^\alpha d)$ simultaneously.

(c) Compute the bias introduced by SL truncation. Let $\hat{g}_{\text{SL}}$ be the gradient estimate from one SL training step. Is $\mathbb{E}[\hat{g}_{\text{SL}}] = g$ (the full-sequence gradient)? If not, characterize the bias and state under what conditions it vanishes.

---

### Problem 12: M-FALCON Amortization Analysis

*M-FALCON processes $b_m$ candidates simultaneously by sharing user history computation. This problem derives the exact complexity savings and identifies the break-even batch size.*

> **Prerequisites:** cf. note [[hstu#4.3 M-FALCON: Inference Amortization|§4.3 — M-FALCON: Inference Amortization]]

(a) Naive inference for ranking $b_m$ candidates, each requiring a separate attention forward pass over user history of length $n$, costs $O(b_m \cdot n^2 d)$. Write out this cost in full, accounting for: (i) self-attention over the history, (ii) cross-attention of the candidate over the history.

(b) In M-FALCON, the user history of length $n$ is encoded once at cost $O(n^2 d)$, and each candidate attends to the encoded history at cost $O(nd)$ per candidate. Derive the total M-FALCON cost as $O(n^2 d + b_m n d)$ and express the speedup ratio $S(b_m, n)$ relative to naive inference.

(c) Find the value of $b_m$ at which M-FALCON achieves at least a $10\times$ speedup. For what regime of $(b_m, n)$ does M-FALCON degenerate to the same cost as naive inference? Interpret: does M-FALCON help when $n \gg b_m$ or when $b_m \gg n$?

---

### Problem 13: Softmax Normalization and Non-Stationarity

*HSTU replaces softmax with pointwise SiLU to handle non-stationary item vocabularies. This problem formalizes the argument that softmax loses absolute preference magnitude under non-stationarity.*

> **Prerequisites:** cf. note [[hstu#3.4 Why Pointwise Attention Instead of Softmax|§3.4 — Why Pointwise Attention Instead of Softmax]]

(a) Let $s_j = \mathbf{q}^\top \mathbf{k}_j$ be the unnormalized attention score for item $j$. The softmax attention weight is $\alpha_j = e^{s_j} / \sum_k e^{s_k}$. Prove that for any constant $c$, shifting all scores by $c$ (i.e., $s_j \to s_j + c$) leaves $\alpha_j$ unchanged. What information about the absolute magnitude of preference does this destroy?

(b) Under a Dirichlet Process (DP) model of item popularity, new items are continually added to the vocabulary with positive probability. Suppose at time $t$ the vocabulary has size $V_t$, growing as $V_t = V_0 \cdot t^\theta$. Show that as $V_t \to \infty$, the softmax denominator $Z_t = \sum_{j=1}^{V_t} e^{s_j}$ grows unboundedly, driving all attention weights $\alpha_j \to 0$.

(c) Pointwise attention replaces softmax with $\tilde{\alpha}_j = \varphi_2(s_j)$ (applied elementwise, $\varphi_2 = \text{SiLU}$). Prove that $\tilde{\alpha}_j$ is invariant to the vocabulary size $V_t$. What is the cost of this choice — specifically, what property of standard attention does pointwise attention violate, and why is that property less important for recommendation?

---

### Problem 14: Pointwise Attention as an Unnormalized Kernel

*Pointwise SiLU attention can be interpreted as an unnormalized kernel attention. This problem derives the implied kernel and shows that the resulting attention map is data-dependent in a way that softmax attention is not.*

> **Prerequisites:** cf. note [[hstu#3.2 Sub-Layer 2: Spatial Aggregation|§3.2 — Sub-Layer 2: Spatial Aggregation]]; requires Problem 13

(a) Standard softmax attention computes $A = \text{softmax}(QK^\top / \sqrt{d})V$. Write down the HSTU spatial aggregation as $A = \varphi_2(QK^\top + \text{rab}^{p,t})V$. For the case $\text{rab}^{p,t} = 0$, show that the $(i,j)$ entry of the pre-aggregation matrix is $\varphi_2(\mathbf{q}_i^\top \mathbf{k}_j)$.

(b) Define the attention kernel $\kappa(\mathbf{q}, \mathbf{k}) = \varphi_2(\mathbf{q}^\top \mathbf{k})$. For $\varphi_2 = \text{SiLU}$, show that $\kappa$ is not a positive-definite kernel in general (construct a counterexample using a $2 \times 2$ Gram matrix).

(c) Despite not being PD, pointwise attention has a useful monotonicity property: if $\mathbf{q}^\top \mathbf{k}_j > \mathbf{q}^\top \mathbf{k}_l$ then $\kappa(\mathbf{q}, \mathbf{k}_j) > \kappa(\mathbf{q}, \mathbf{k}_l)$ (since $\varphi_2$ is monotone for large positive arguments). Prove this monotonicity for $\varphi_2(z) = z \sigma(z)$ when $z > 0$. What does this imply about ranking quality relative to softmax?

---

### Problem 15: Temporal Relative Bias and Sequence-Length Invariance

*The temporal relative attention bias (rab) uses log-bucketed time differences rather than absolute positions. This problem proves that the rab is invariant to absolute position shifts and derives the number of distinct buckets needed to cover the range of observed time differences.*

> **Prerequisites:** cf. note [[hstu#3.5 Temporal Relative Attention Bias|§3.5 — Temporal Relative Attention Bias]]

(a) The rab is $\text{rab}^{p,t}_{ij} = b^p_{|i-j|} + b^t_{\lfloor \log_2 \Delta t_{ij} \rfloor}$. Prove that $\text{rab}^{p,t}_{ij}$ depends only on relative position $|i - j|$ and relative time $\Delta t_{ij} = |t_i - t_j|$, not on the absolute positions $i, j$ or absolute timestamps $t_i, t_j$.

(b) Absolute sinusoidal positional encodings $\text{PE}(i) = [\sin(\omega_k i), \cos(\omega_k i)]_{k=1}^{d/2}$ are not sequence-length invariant: a sequence of length 100 and a sequence of length 10,000 assign different encodings to position 50. Formalize why this is problematic for recommendation (where sequence lengths vary from 10 to 100,000) and prove that rab does not share this pathology.

(c) Suppose timestamps are measured in seconds and user history spans at most $T_{\max}$ seconds. How many distinct temporal buckets does $\lfloor \log_2 \Delta t_{ij} \rfloor$ produce for $\Delta t_{ij} \in [1, T_{\max}]$? If $T_{\max}$ = 1 year $\approx 3.15 \times 10^7$ seconds, compute the number of buckets. How does this compare to the number of buckets needed for linear bucketing to achieve the same resolution ratio?

---

### Problem 16: Supervision Density and the Scaling Law

*HSTU's generative training extracts $n_i$ supervision signals per user sequence. This problem formalizes the relationship between supervision density, parameter count, and the scaling law exponent.*

> **Prerequisites:** cf. note [[hstu#5. Scaling Laws|§5 — Scaling Laws]]; requires Problem 9

(a) Define *supervision density* as the number of training targets per model parameter per epoch: $\rho = T_{\text{targets}} / P$ where $T_{\text{targets}}$ is total training targets and $P$ is parameter count. For impression-level training with $M$ users each of average length $\bar{n}$, compute $T_{\text{targets}}^{\text{imp}}$. For generative training, compute $T_{\text{targets}}^{\text{gen}}$ and show $T_{\text{targets}}^{\text{gen}} = \bar{n} \cdot T_{\text{targets}}^{\text{imp}}$.

(b) Assume the scaling law $\text{HR@100} \propto C^\beta$ holds when $\rho$ is held approximately constant as $P$ scales. If impression-level training has supervision density $\rho_0$ and generative training has density $\bar{n} \rho_0$, derive the compute $C'$ at which impression-level training first achieves the same quality as generative training at compute $C$. Express $C'/C$ as a function of $\bar{n}$ and $\beta$.

(c) The note states that DLRMs plateau around 200B parameters while HSTU scales to 1.5 trillion. Interpret this observation through the lens of supervision density. At what supervision density $\rho$ does a power-law cease to hold, and what would a density-based model of the plateau predict?

---

### Problem 17: Power-Law Exponent Stability

*A genuine scaling law has a stable power-law exponent across the measured compute range. This problem analyzes what conditions are necessary for the exponent to be stable, and characterizes the difference between a power law and a sigmoid-shaped saturation curve.*

> **Prerequisites:** cf. note [[wukong#5. Scaling Strategy and Hyperparameters|§5 — Scaling Strategy and Hyperparameters]]; cf. note [[hstu#5. Scaling Laws|§5 — Scaling Laws]]

(a) A power law is $L(C) = A \cdot C^{-\alpha}$. A saturating model is $L(C) = L_\infty + (L_0 - L_\infty) e^{-\lambda C}$. On a log-log plot of $L$ vs. $C$, derive the slope (i.e., $d \log L / d \log C$) for both models. Show that the power law has constant slope $-\alpha$, while the saturating model has slope that asymptotically approaches 0.

(b) Suppose you observe quality improvement over $C \in [C_{\min}, C_{\max}]$ with $C_{\max}/C_{\min} = 100$ (two orders of magnitude, as in Wukong). A saturating model with $\lambda = 1/C_{\min}$ fits the data in $[C_{\min}, 2C_{\min}]$ with apparent exponent $\hat{\alpha}$. Show that extrapolating this fit to $C_{\max}$ overestimates the quality improvement. Derive the overestimation factor.

(c) What architectural property of Wukong and HSTU prevents saturation within the tested compute range? Relate your answer to the interaction order theorem (Wukong) and supervision density (HSTU).

---

### Problem 18: Memory Footprint: HSTU vs. Standard Transformer

*HSTU reduces per-layer activation memory from $33d$ to $14d$. This problem derives these bounds from the computation graph, explains the sources of savings, and quantifies the implication for maximum achievable depth.*

> **Prerequisites:** cf. note [[hstu#4.4 Memory Efficiency vs. Standard Transformers|§4.4 — Memory Efficiency vs. Standard Transformers]]

(a) For a standard Transformer block with embedding dimension $d$ and sequence length $n$, enumerate the activations that must be stored for backpropagation: input ($nd$), attention scores ($n^2$), softmax output ($n^2$), value projection ($nd$), attention output ($nd$), FFN intermediate ($nd \cdot 4$), etc. Show the dominant term is proportional to $33d$ (per token, in units of $d$-dimensional vectors).

(b) For an HSTU block, enumerate the same activations. The SiLU gating fuses $U$ and $AV$; there is no separate FFN sublayer. Show the total is proportional to $14d$ per token. Which specific activations account for the $19d$ reduction?

(c) A GPU with memory $\mathcal{M}$ bytes can store $\mathcal{M} / (33d \cdot n \cdot \text{sizeof(bf16)})$ Transformer layers simultaneously for a sequence of length $n$. Derive the analogous expression for HSTU. For $\mathcal{M} = 80$ GB, $d = 1024$, $n = 4096$, compute the maximum number of layers for each architecture.

---

## Algorithmic Applications

### Problem 19: Optimized FM Block: Pseudocode and Shape Annotations

*This problem implements the FMB pipeline from input embeddings to output embeddings, tracking all tensor shapes and verifying that the low-rank associativity trick is applied correctly.*

> **Prerequisites:** cf. note [[wukong#2.2 Factorization Machine Block|§2.2 — Factorization Machine Block]]; requires Problem 2

(a) **Inputs and data structures**: Write pseudocode for `FMB(X, Y, W_mlp, n_F, d)` where `X` $\in \mathbb{R}^{n \times d}$ is the input embedding matrix, `Y` $\in \mathbb{R}^{n \times k}$ is the low-rank projection, and `W_mlp` is a list of MLP weight matrices. Annotate each intermediate tensor with its shape.

(b) **The associativity optimization**: In your pseudocode, explicitly show the two orderings `(X @ X.T) @ Y` and `X @ (X.T @ Y)`. Add a complexity count (in FLOPs) for each ordering and verify that the right-to-left ordering is strictly cheaper when $k < n$.

(c) **Output reshaping and correctness check**: After the MLP, the output must be reshaped from a flat vector of length $n_F \cdot d$ into $\mathbb{R}^{n_F \times d}$. Write a shape assertion that fails if the MLP output size is inconsistent with $n_F$ and $d$, and explain why this assertion is necessary for the subsequent layer's FM to work correctly.

---

### Problem 20: HSTU Forward Pass: Pseudocode and Complexity

*This problem implements a single HSTU layer, tracking the three sub-layers and verifying that the attention computation is causal (lower-triangular mask for autoregressive generation).*

> **Prerequisites:** cf. note [[hstu#3. HSTU Architecture|§3 — HSTU Architecture]]

(a) **Sub-layer structure**: Write pseudocode for `HSTU_layer(X, W1, W2, rab)` where `X` $\in \mathbb{R}^{n \times d}$, `W1` $\in \mathbb{R}^{d \times 4d}$ (projection), `W2` $\in \mathbb{R}^{d \times d}$ (output), and `rab` $\in \mathbb{R}^{n \times n}$ is the precomputed relative attention bias. Annotate shapes throughout.

(b) **Causal mask**: For autoregressive generation (predicting $x_{t+1}$ from $x_{\leq t}$), the attention matrix must be lower-triangular. Add a causal mask to your pseudocode. Show explicitly what changes in the attention computation, and verify that the masked HSTU still has $O(n^2 d)$ complexity (not $O(n^2)$ due to the mask sparsity).

(c) **Complexity accounting**: Count FLOPs for each sub-layer separately: (i) the linear projection $W_1$, (ii) the QK dot-product attention, (iii) the value aggregation, (iv) the gating step, (v) the output projection $W_2$. Which sub-layer dominates for large $n$? For large $d$?

---

### Problem 21: Stochastic Length Sampling: Implementation Sketch

*This problem implements the stochastic length truncation procedure and verifies that it is unbiased in expectation when used with appropriate importance weighting.*

> **Prerequisites:** cf. note [[hstu#4.2 Stochastic Length Sparsity|§4.2 — Stochastic Length Sparsity]]; requires Problem 11

(a) **Sampling procedure**: Write pseudocode for `stochastic_length(sequence, N, alpha)` that takes a user sequence and returns either the full sequence or a randomly truncated subsequence, according to the SL scheme. The output must also return a scalar `weight` for importance weighting.

(b) **Importance weighting for unbiasedness**: If the loss on the full sequence is $\mathcal{L}(x_{0:n})$ and on the truncated sequence is $\mathcal{L}(x_{\text{sub}})$, write the importance-weighted loss $\tilde{\mathcal{L}}$ such that $\mathbb{E}[\tilde{\mathcal{L}}] = \mathcal{L}(x_{0:n})$ in expectation over the SL randomness. Derive the weight analytically from your answer to Problem 11(c).

(c) **Batch-level implementation**: In practice, a minibatch contains users with variable-length sequences. Write pseudocode for `batch_stochastic_length(batch, N, alpha)` that applies SL independently per user and packs the resulting (possibly shorter) sequences into a ragged batch. Identify the key implementation challenge (hint: it involves attention mask construction for sequences of different lengths).

---

### Problem 22: M-FALCON Attention Mask Construction

*M-FALCON scores $b_m$ candidates simultaneously by constructing a combined attention mask that prevents candidates from attending to each other. This problem derives the mask and verifies the resulting complexity.*

> **Prerequisites:** cf. note [[hstu#4.3 M-FALCON: Inference Amortization|§4.3 — M-FALCON: Inference Amortization]]; requires Problem 20

(a) **Mask structure**: Consider a combined sequence $[h_1, \ldots, h_n, c_1, \ldots, c_{b_m}]$ of total length $n + b_m$, where $h_i$ are history tokens and $c_j$ are candidate tokens. Define a binary attention mask $M \in \{0, 1\}^{(n+b_m) \times (n+b_m)}$ such that:
   - History tokens attend to previous history tokens (causal).
   - Each candidate $c_j$ attends to all history tokens.
   - No candidate attends to any other candidate.
   Draw the mask matrix as a block diagram with four labeled quadrants.

(b) **Complexity verification**: Using the mask from (a), count the number of non-zero entries. Show that the attention cost is proportional to $n^2/2 + b_m \cdot n$ (ignoring constants). Verify this matches the $O(n^2 d + b_m n d)$ bound from Problem 12.

(c) **rab modification**: The temporal relative attention bias must assign consistent temporal positions to candidates. Write pseudocode for `compute_rab_mfalcon(history_times, candidate_time, n, b_m)` that constructs the rab for the combined sequence, placing all candidates at `candidate_time` (a fixed inference timestamp).

---

### Problem 23: Ablation Study: Gradient Flow Through LCB and Residual

*This problem implements a simplified two-layer Wukong stack and numerically verifies the gradient flow argument from Problem 7 by measuring the gradient norm at the input layer under different ablation conditions.*

> **Prerequisites:** cf. note [[wukong#6.3 Ablation Studies|§6.3 — Ablation Studies]]; requires Problem 7

(a) **Model definition**: Write pseudocode for a two-layer simplified Wukong stack with configurable ablation flags `use_lcb` (bool) and `use_residual` (bool). The FMB is simplified to $F(X) = \text{ReLU}(X X^\top W_F)$ for $W_F \in \mathbb{R}^{n \times n}$. Include the LCB as $L(X) = W_L X$ and the residual update.

(b) **Gradient norm experiment**: Write pseudocode for `measure_gradient_norm(model, X, use_lcb, use_residual)` that: (i) initializes $W_F$ near zero (so $\partial F / \partial X \approx 0$), (ii) computes a forward pass, (iii) backpropagates a unit loss, and (iv) returns $\|\partial \mathcal{L} / \partial X_0\|_F$. Predict the expected output for each of the four ablation conditions $(\pm \text{LCB}, \pm \text{residual})$ using your theory from Problem 7.

(c) **Interpretation**: Given the predicted gradient norms from (b), rank the four ablation conditions by expected training quality (best to worst). Does the ranking match the ablation results reported in the note? Identify any condition that the gradient-flow argument alone cannot fully explain.
