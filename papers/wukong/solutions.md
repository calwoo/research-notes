# Solutions: Wukong & HSTU

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

**Key insight:** $G = XX^\top$ is a Gram matrix, hence PSD by construction; the FM scalar sum follows from expanding $\|\sum_j \mathbf{x}_j\|^2$ and solving for the cross-term.

**Sketch:**

(a) For any $\mathbf{v} \in \mathbb{R}^n$: $\mathbf{v}^\top G \mathbf{v} = \mathbf{v}^\top X X^\top \mathbf{v} = \|X^\top \mathbf{v}\|^2 \geq 0$, so $G \succeq 0$. Rank: $\text{rank}(G) = \text{rank}(X) \leq \min(n,d)$ since column space of $G$ equals column space of $X$.

(b) Expanding: $\|\sum_j \mathbf{x}_j\|^2 = \sum_j \|\mathbf{x}_j\|^2 + 2\sum_{j<k}\langle \mathbf{x}_j, \mathbf{x}_k\rangle$, so $\hat{y}_{\text{FM}} = \frac{1}{2}\bigl(\|\sum_j \mathbf{x}_j\|^2 - \sum_j \|\mathbf{x}_j\|^2\bigr)$. Cost: $O(nd)$ (one sum and $n$ norms) vs. $O(n^2 d)$ for computing $G$.

(c) $G = XX^\top$ requires $n^2$ inner products each of length $d$: $\Theta(n^2 d)$. The trick in (b) reduces to $O(nd)$ — a factor $n$ improvement, which is the FM identity that Wukong's optimized FM generalizes via right-multiplication by $Y$.

---

### Problem 2: Low-Rank FM Projection and Associativity

**Key insight:** Matrix multiplication is associative but not commutative; right-to-left evaluation of $X(X^\top Y)$ avoids forming the $n \times n$ Gram matrix entirely, cutting cost by a factor of $n/k$.

**Sketch:**

(a) Left-to-right: $(XX^\top) \in \mathbb{R}^{n \times n}$ costs $O(n^2 d)$; then $(XX^\top)Y$ costs $O(n^2 k)$. Right-to-left: $X^\top Y \in \mathbb{R}^{d \times k}$ costs $O(ndk)$; then $X(X^\top Y) \in \mathbb{R}^{n \times k}$ costs $O(ndk)$. Total right-to-left: $O(ndk)$.

(b) $\text{FM}_{\text{opt}} \text{FM}_{\text{opt}}^\top = (XG^{-1}GYY^\top G^\top X^\top)$ — more directly: $(XX^\top Y)(XX^\top Y)^\top = GYY^\top G^\top = G P_Y G$ since $P_Y = YY^\top$ when $Y$ has orthonormal columns ($Y^\top Y = I_k$).

(c) JL lemma: for $k = O(\log n / \epsilon^2)$, a random $\mathbb{R}^k$-projection preserves all pairwise $\ell_2$ distances up to factor $(1 \pm \epsilon)$. Since rows of $G$ are the images of feature vectors under $X$ in the Gram space, the JL guarantee applies to the row distances of $G$, and hence $\text{FM}_{\text{opt}}$ with random $Y$ approximately preserves pairwise Gram distances.

---

### Problem 3: Complexity Comparison: Naive vs. Optimized FM

**Key insight:** The per-stage cost analysis reveals that after the low-rank FM projection the bottleneck shifts to the MLP (cost $O(nkh + h^2)$), not the FM itself, so the optimal $k$ is governed by the MLP width $h$.

**Sketch:**

(a) Operations and shapes: FM opt: $X^\top Y \in \mathbb{R}^{d \times k}$ (cost $O(ndk)$), $X(X^\top Y) \in \mathbb{R}^{n \times k}$ (cost $O(ndk)$); flatten: $\mathbb{R}^{nk}$ (free); LN: $O(nk)$; MLP first layer: $\mathbb{R}^{nk} \to \mathbb{R}^h$ ($O(nkh)$); hidden: $O(h^2)$; output: $\mathbb{R}^h \to \mathbb{R}^{n_F d}$ ($O(n_F dh)$); reshape: free.

(b) Total: $2 O(ndk) + O(nkh) + O(h^2) + O(n_F dh)$. With $k = O(n)$: FM cost is $O(n^2 d)$ — but $k \ll n$ by design. For $k \sim h/d$ (MLP-matched), the FM term $O(ndk)$ is subsumed by $O(nkh) = O(nh^2/d)$. Setting $n_F = O(n)$: total is $O(ndh + h^2)$.

(c) Naive FM: $O(n^2 d)$. Optimized FM: $O(ndk)$. Break-even when $ndk = n^2 d$, i.e., $k = n$. The optimization saves compute whenever $k < n$, which is always true by design.

---

### Problem 4: Interaction Order: Base Case and Inductive Step

**Key insight:** The FM inner product $\langle \mathbf{u}, \mathbf{v} \rangle$ adds the orders of its operands additively, so pairing two order-$2^i$ embeddings yields order $2^{i+1}$; the MLP preserves but does not amplify this because it acts on already-computed interactions, not on new products of distinct feature chains.

**Sketch:**

(a) Raw embeddings $\mathbf{x}_i$ are degree-1 polynomials in themselves. $X_0 = \{$raw feature embeddings$\}$, so each component is linear in original features — order 1. Base case holds.

(b) $\text{LCB}(X_i)_j = \sum_l (W_L)_{jl} \mathbf{x}_{i,l}$. Each row is a linear combination of degree-$\leq 2^i$ polynomials; linear combination cannot raise degree. LCB order $\leq 2^i$.

(c) $(G_i)_{jk} = \langle \mathbf{x}_{i,j}, \mathbf{x}_{i,k} \rangle = \sum_r (x_{i,j})_r (x_{i,k})_r$. Each term is a product of a degree-$o_j$ and degree-$o_k$ polynomial, yielding degree $o_j + o_k$. Maximum when $o_j = o_k = 2^i$: order $2^{i+1}$.

(d) The MLP applies fixed nonlinear functions (e.g., ReLU) to each scalar component of the flattened FM output independently — it does not multiply distinct feature chains together. Each output component is a function of a single inner product (order $\leq 2^{i+1}$), so its order does not exceed $2^{i+1}$. If the MLP were a degree-2 polynomial network, each layer would square the polynomial degree, violating the bound: a degree-2 MLP on order-$2^{i+1}$ input produces order $2^{i+2}$, breaking the inductive claim.

---

### Problem 5: Exponential vs. Linear Order Growth

**Key insight:** The depth-to-order relationship inverts to $l_W = \lceil \log_2(l_D + 1) \rceil$ Wukong layers to match $l_D$ DCNv2 layers, because exponential growth is the inverse of logarithm.

**Sketch:**

(a) Wukong order after $l_W$ layers: $2^{l_W}$. DCNv2 order after $l_D$ layers: $l_D + 1$. Require $2^{l_W} \geq l_D + 1$, i.e., $l_W \geq \log_2(l_D+1)$. Taking the ceiling: $l_W = \lceil \log_2(l_D+1) \rceil$.

(b) To reach order $q$: DCNv2 needs $q-1$ layers, Wukong needs $\lceil \log_2 q \rceil$ layers. Parameter ratio: $(q-1)/\lceil \log_2 q \rceil$. For $q = 256$: $(256-1)/\lceil \log_2 256 \rceil = 255/8 \approx 31.9$. (The note states 12.5× at a different normalization; the ratio depends on the chosen $q$ and per-layer parameter count.)

(c) The claim confuses interaction order (a measure of function class complexity) with VC dimension or Rademacher complexity (which control generalization). Interaction order $2^l$ means the model can represent degree-$2^l$ polynomial interactions, but the actual function it learns depends on optimization and regularization. High polynomial degree does not imply overfitting — it implies large *hypothesis class*, not that training converges to a high-degree interpolant.

---

### Problem 6: LCB Does Not Raise Interaction Order

**Key insight:** Linear maps compose with polynomial feature maps without raising degree; only bilinear operations (like inner products) increase degree multiplicatively.

**Sketch:**

(a) Row $j$ of $f(X) = WX$: $(WX)_j = \sum_l W_{jl} \mathbf{x}_l$. Each $\mathbf{x}_l$ is degree $\leq p$; the sum has degree $\leq p$. Linear combination preserves degree.

(b) QCB output: $(f_Q(X))_{ij} = \sum_{k,l} W_{ij,kl} X_{ik} X_{jl}$. Each term multiplies two degree-$p$ entries, yielding degree $2p$. So a QCB doubles the interaction order at each application — it breaks the proof structure since the inductive step would require tracking the quadratic composition.

(c) Layer normalization computes $\hat{x} = (x - \mu)/\sigma \cdot \gamma + \beta$. The mean $\mu$ and variance $\sigma^2$ are statistics of $x$; both $\mu$ and $\sigma^2$ are polynomial in $x$ (degree 1 and 2 respectively). Dividing by $\sigma = \sqrt{\sigma^2}$ introduces a square root, making LN *not* a polynomial operation. Formally, LN raises the algebraic degree. In practice, the paper treats LN as a stabilizer whose degree-raising effect is absorbed into the MLP's implicit capacity; for the theoretical order argument to hold strictly, one must treat LN as an approximation at the relevant scale.

---

### Problem 7: Residual and LCB as Complementary Linear Pathways

**Key insight:** Each linear pathway (LCB, residual) provides an independent additive term in the Jacobian; when both are present, at least one linear term survives regardless of FM saturation, but removing both eliminates all linear gradient paths through saturated layers.

**Sketch:**

(a) $\partial X_1 / \partial X_0 = \partial F / \partial X_0 + W_L + I$. When $\partial F / \partial X_0 \approx 0$: $\partial X_1/\partial X_0 \approx W_L + I$, which is nonzero as long as $W_L \neq -I$.

(b) $\partial \mathcal{L} / \partial X_0 = G_l \cdots G_1 \cdot \partial \mathcal{L}/\partial X_l$. With $\partial F_i/\partial X_{i-1} \approx 0$: $G_i \approx W_{L,i} + I$. The product $\prod_i (W_{L,i} + I)$ is a sum of $2^l$ terms corresponding to all subsets of layers that use the LCB vs. residual path — analogous to the ResNet sum-of-paths interpretation.

(c) No LCB, residual only: $G_i \approx I$, product $\approx I$, gradient flows perfectly. No residual, LCB only: $G_i \approx W_{L,i}$, product $\approx \prod_i W_{L,i}$, which may vanish (product of many random matrices). Both removed: $G_i \approx 0$, product $\approx 0$ — gradient vanishes entirely, matching the "substantial degradation" observation.

---

### Problem 8: Total Complexity of the Interaction Stack

**Key insight:** Because the embedding count $n_i$ is stabilized by the LCB at each layer, the per-layer cost is constant, and the total cost is just depth $\times$ per-layer cost; setting depth to $\log_2 n$ gives the $O(ndh \log n)$ bound.

**Sketch:**

(a) Per-layer cost (from Problem 3): $O(ndh + h^2)$ with $n_i \approx n'$ constant. Total across $l$ layers: $l \cdot O(ndh + h^2) = O(l \cdot ndh + l \cdot h^2)$.

(b) Setting $l = \log_2 n$: total $= O(ndh \log n + h^2 \log n)$. The $h^2 \log n$ term is dominated by $ndh \log n$ when $nd \gg h$ (wide embedding, normal regime), leaving $O(ndh \log n + h^2)$ with the $h^2$ term surviving when the MLP is very large.

(c) Explicit degree-$q$ polynomial kernel: $\binom{n+q}{q}$ monomials; for $q = 2^l = n$, this is $\binom{2n}{n} = \Theta(4^n / \sqrt{n})$ — doubly exponential in $l$. Wukong costs $O(ndh \log n)$ polynomial in all parameters. The tradeoff: Wukong's MLP inside FMB cannot represent all $\binom{n+2^l}{2^l}$ monomials explicitly — it learns a $h$-dimensional projection of them, sacrificing completeness for tractability.

---

### Problem 9: Impression-Level vs. Generative Training Complexity

**Key insight:** Impression-level training invokes a fresh attention forward pass (cost $O(n_i^2 d)$) for *each* of the $n_i$ training examples extracted from a sequence of length $n_i$, yielding $O(n_i^3 d)$ total; generative training amortizes this to one pass at $O(n_i^2 d)$.

**Sketch:**

(a) User $i$, sequence length $n_i$: each of the $n_i$ training examples uses a context of length up to $n_i$, so attention cost per example is $O(n_i^2 d)$. Total: $n_i \cdot O(n_i^2 d) = O(n_i^3 d)$.

(b) Generative: one forward pass over length-$n_i$ sequence costs $O(n_i^2 d)$ attention. Sampling $s_u(n_i) \cdot n_i = 1$ target per step does not reduce the forward pass cost; the full sequence is still processed. Expected cost per step: $O(n_i^2 d)$. Ratio improvement: $n_i^3 d / n_i^2 d = n_i$.

(c) With power-law lengths, $P(n_i = n) \propto n^{-\gamma}$ on $[1, N]$: normalizing constant $Z \approx \zeta(\gamma)$ for large $N$. Expected impression cost $\propto \sum_n n^{-\gamma} \cdot n^3 d = d \sum_n n^{3-\gamma}$; for $\gamma = 2$: $\sum n^1 = O(N^2)$. Expected generative cost $\propto d \sum_n n^{2-\gamma}$; for $\gamma = 2$: $\sum n^0 = O(N)$. Cost ratio $= O(N)$ — the impression-level cost grows linearly in the max sequence length faster than generative.

---

### Problem 10: Stochastic Length Truncation: Expected Cost

**Key insight:** The two terms in the expected cost (full sequence, truncated sequence) both evaluate to $\Theta(N^\alpha d)$ under the specific probability $N^\alpha/n_i^2$, so the expectation is $O(N^\alpha d)$ regardless of $n_i$.

**Sketch:**

(a) $\mathbb{E}[\text{Cost}_i] = \frac{N^\alpha}{n_i^2} \cdot c n_i^2 d + \left(1 - \frac{N^\alpha}{n_i^2}\right) \cdot c N^\alpha d = c N^\alpha d + c N^\alpha d \left(1 - \frac{N^\alpha}{n_i^2}\right)$.

(b) Simplifying: $\mathbb{E}[\text{Cost}_i] = c N^\alpha d \left(1 + 1 - \frac{N^\alpha}{n_i^2}\right) \leq 2c N^\alpha d = O(N^\alpha d)$. For $n_i \leq N^{\alpha/2}$: full sequence always used, cost $= c n_i^2 d \leq c N^\alpha d = O(N^\alpha d)$. Bound holds in both cases.

(c) Total cost over $M$ users: $\sum_i \mathbb{E}[\text{Cost}_i] \leq 2cM N^\alpha d$. For $N = 10^4$, $\alpha = 1.5$: $N^\alpha = N^{1.5} = 10^6$. Impression-level (from Problem 9(a)): $O(N^3 d)$ per long user $= O(10^{12} d)$. Reduction factor per long user: $N^3 / N^\alpha = N^{1.5} = 10^6$.

---

### Problem 11: Stochastic Length Probability Calibration

**Key insight:** The full-sequence term and the truncated term in $\mathbb{E}[\text{Cost}_i]$ must both be $O(N^\alpha d)$; equating both upper bounds to $C N^\alpha d$ uniquely pins $p(n_i) = N^\alpha / n_i^2$.

**Sketch:**

(a) $\mathbb{E}[\text{Cost}_i] = p(n_i) n_i^2 d + (1-p(n_i)) N^\alpha d \leq C N^\alpha d$. Rearranging: $p(n_i)(n_i^2 - N^\alpha)d \leq (C-1) N^\alpha d$, so $p(n_i) \leq \frac{(C-1)N^\alpha}{n_i^2 - N^\alpha} \approx \frac{C N^\alpha}{n_i^2}$ for $n_i \gg N^{\alpha/2}$.

(b) Setting $C = 2$: $p(n_i) = N^\alpha / n_i^2$ makes the full-sequence term $= N^\alpha d$ and the truncated term $\leq N^\alpha d$; sum $\leq 2 N^\alpha d$. Any larger $p$ violates the budget (full-sequence term exceeds $N^\alpha d$); any smaller $p$ is valid but wastes the full-sequence computation. So $p(n_i) = N^\alpha/n_i^2$ is the budget-tight solution.

(c) SL is a biased estimator of the full-sequence loss in general: $\mathbb{E}[\hat{g}_{\text{SL}}] \neq g$ because the truncated subsequence $\Gamma(n_i, N^{\alpha/2})$ is a strict subset of the full history, and the attention patterns over a truncated sequence differ from those computed on the full sequence (attention scores from omitted tokens are lost). The bias vanishes only when $n_i \leq N^{\alpha/2}$ (no truncation ever applied).

---

### Problem 12: M-FALCON Amortization Analysis

**Key insight:** Amortization works because the user history computation is shared across all $b_m$ candidates; the per-candidate cross-attention cost is $O(nd)$ rather than $O(n^2 d)$, making the amortized per-candidate cost $O(nd)$ vs. $O(n^2 d)$ naive — a factor of $n$ reduction per candidate.

**Sketch:**

(a) Naive: $b_m$ separate passes. Each pass: (i) self-attention over history $O(n^2 d)$, (ii) cross-attention of candidate over history $O(nd)$. Total: $b_m(n^2 d + nd)$.

(b) M-FALCON: history encoded once at $O(n^2 d)$; each candidate cross-attends at $O(nd)$; total $O(n^2 d + b_m nd)$. Speedup: $S = b_m(n^2 d + nd) / (n^2 d + b_m nd) \approx b_m n^2 / (n^2 + b_m n) = b_m n / (n + b_m)$ for large $d$.

(c) For $S \geq 10$: $b_m n / (n + b_m) \geq 10$, so $b_m(n - 10) \geq 10n$, i.e., $b_m \geq 10n/(n-10) \approx 10$ for $n \gg 10$. Degeneracy: when $b_m = 1$, $S \to n/(n+1) < 1$ — no benefit. M-FALCON helps when $b_m \gg 1$ and $b_m \ll n$ (many candidates, but fewer than history length); when $b_m \gg n$, both methods cost $O(b_m n d)$.

---

### Problem 13: Softmax Normalization and Non-Stationarity

**Key insight:** Softmax invariance to constant shifts ($e^{s_j + c} / \sum_k e^{s_k + c} = e^{s_j}/\sum_k e^{s_k}$) is precisely what destroys absolute preference magnitude — the information needed to distinguish strong from weak preferences under a growing item vocabulary.

**Sketch:**

(a) $\alpha_j(c) = e^{s_j+c}/\sum_k e^{s_k+c} = e^{s_j} e^c / (e^c \sum_k e^{s_k}) = e^{s_j}/\sum_k e^{s_k} = \alpha_j(0)$. Shift-invariance destroys the information in $\|\mathbf{q}\|$ and $\|\mathbf{k}_j\|$ — only relative score differences matter to softmax, not absolute magnitudes.

(b) Each new item $j$ adds $e^{s_j} > 0$ to $Z_t$. With $V_t$ items, even if all scores are bounded by $s_{\max}$: $Z_t \geq V_t \cdot e^{s_{\min}} \to \infty$. All $\alpha_j = e^{s_j}/Z_t \leq e^{s_{\max}}/Z_t \to 0$.

(c) $\tilde{\alpha}_j = \varphi_2(s_j)$ depends only on $s_j$, not on $V_t$ or other items' scores. Vocabulary size enters nowhere. Cost: pointwise attention is not a valid probability distribution ($\sum_j \tilde{\alpha}_j \neq 1$), so it cannot be interpreted as a posterior over items. For recommendation, this is acceptable because the task is ranking (monotone ordering), not calibrated probability estimation.

---

### Problem 14: Pointwise Attention as an Unnormalized Kernel

**Key insight:** $\kappa(\mathbf{q}, \mathbf{k}) = \text{SiLU}(\mathbf{q}^\top \mathbf{k})$ is not PD because SiLU takes negative values for negative inputs, violating the PD requirement; but SiLU's monotonicity for positive arguments preserves the ranking of attention scores, which is sufficient for recommendation quality.

**Sketch:**

(a) HSTU spatial aggregation entry $(i,j)$: $\varphi_2((\mathbf{q}_i^\top \mathbf{k}_j))$ (with $\text{rab} = 0$). The output is $\sum_j \varphi_2(\mathbf{q}_i^\top \mathbf{k}_j) \mathbf{v}_j$. This is a weighted sum of values with data-dependent weights that can be negative (unlike softmax).

(b) Counterexample: take $\mathbf{q}_1 = \mathbf{k}_1 = (-1, 0)^\top$, $\mathbf{q}_2 = \mathbf{k}_2 = (0,-1)^\top$. Then $\kappa(\mathbf{q}_1, \mathbf{k}_1) = \text{SiLU}(-1) < 0$, $\kappa(\mathbf{q}_1, \mathbf{k}_2) = \text{SiLU}(0) = 0$. The $2\times 2$ Gram matrix $K_{ij} = \kappa(\mathbf{q}_i, \mathbf{k}_j)$ has negative diagonal entry — not PSD.

(c) $\text{SiLU}(z) = z\sigma(z)$. Derivative: $d/dz[z\sigma(z)] = \sigma(z) + z\sigma(z)(1-\sigma(z)) > 0$ for $z > 0$ since all terms positive. Monotone on $(0,\infty)$. For recommendation: if $\mathbf{q}^\top \mathbf{k}_j > \mathbf{q}^\top \mathbf{k}_l > 0$, then $\tilde{\alpha}_j > \tilde{\alpha}_l$, preserving the same top-$k$ ranking as softmax would. The ranking quality is equivalent to softmax in the regime of positive scores.

---

### Problem 15: Temporal Relative Bias and Sequence-Length Invariance

**Key insight:** Relative position $|i-j|$ and relative time $\Delta t_{ij}$ are invariant under global index shift and global time shift respectively; this is precisely what makes the rab generalize across sequences of different lengths and recording epochs.

**Sketch:**

(a) $\text{rab}^{p,t}_{ij}$ depends on $|i-j|$ (relative position, shift-invariant) and $\lfloor \log_2 |t_i - t_j| \rfloor$ (relative time, shift-invariant). For any $\delta_p, \delta_t$: $|(i+\delta_p) - (j+\delta_p)| = |i-j|$ and $|(t_i + \delta_t) - (t_j + \delta_t)| = |t_i - t_j|$.

(b) Sinusoidal PE: $\text{PE}(50)$ is fixed regardless of sequence length, but the *meaning* of position 50 differs between a length-100 and length-10,000 sequence (it is near the middle vs. near the start). Formally, if a model is trained on sequences of length 100, PE(50) encodes "halfway through"; at test time with length 10,000, PE(50) encodes "very early" — a distribution shift. Rab avoids this: $|i-j| = 49$ means "50 steps ago" consistently regardless of sequence length.

(c) $\lfloor \log_2 \Delta t \rfloor$ for $\Delta t \in [1, T_{\max}]$: buckets are $[2^k, 2^{k+1})$ for $k = 0, 1, \ldots, \lfloor \log_2 T_{\max} \rfloor$. Number of buckets $= \lfloor \log_2(3.15 \times 10^7) \rfloor + 1 \approx 25$. Linear bucketing with the same resolution ratio $2:1$ between adjacent buckets on a range of $T_{\max}$ would require $T_{\max}/1 = 3.15 \times 10^7$ buckets — six orders of magnitude more.

---

### Problem 16: Supervision Density and the Scaling Law

**Key insight:** Generative training multiplies supervision density by $\bar{n}$ (average sequence length) relative to impression-level training; since quality scales as $C^\beta$, impression-level training needs $\bar{n}^{1/\beta}$ times more compute to match generative training quality.

**Sketch:**

(a) Impression-level: $T^{\text{imp}} = M$ targets (one per user per step). Generative: each sequence of length $\bar{n}$ provides $\bar{n}$ targets, so $T^{\text{gen}} = M \bar{n}$. Density ratio: $T^{\text{gen}}/T^{\text{imp}} = \bar{n}$.

(b) Quality at compute $C$ under generative training: $Q^{\text{gen}}(C) = A C^\beta$. For impression-level to match this quality, need $A (C')^\beta = A C^\beta$, so $C' = C$. But at equal compute $C$, impression-level processes $1/\bar{n}$ the supervision signal, equivalent to effective compute $C/\bar{n}$. To match $Q^{\text{gen}}(C)$: $(C')^\beta = \bar{n} \cdot C^\beta$, giving $C'/C = \bar{n}^{1/\beta}$.

(c) As $P$ grows past the point where $T/P$ (targets per parameter) falls below a threshold, adding parameters no longer improves quality — each parameter is "undertrained." DLRM plateaus at 200B because impression-level density $\rho_0$ is too low for effective training of $>200$B params. A density-based plateau model predicts saturation when $\rho \sim \rho_{\text{crit}} \sim O(1)$ (a few gradient steps per parameter), regardless of architecture.

---

### Problem 17: Power-Law Exponent Stability

**Key insight:** A power law has constant log-log slope $-\alpha$ everywhere; a saturating exponential has slope approaching 0; distinguishing them requires observing the curve over at least one decade, which is exactly what two orders of magnitude of compute range provides.

**Sketch:**

(a) Power law: $\log L = \log A - \alpha \log C$; slope $d \log L / d \log C = -\alpha$ (constant). Saturating model: $\log L = \log(L_\infty + (L_0 - L_\infty)e^{-\lambda C})$; slope $= -\lambda C (L_0-L_\infty)e^{-\lambda C} / (L_\infty + (L_0-L_\infty)e^{-\lambda C}) \to 0$ as $C \to \infty$.

(b) Fitting a saturating model on $[C_{\min}, 2C_{\min}]$ near $C_{\min}$ (before saturation sets in): the apparent local slope mimics a power law with exponent $\hat{\alpha} \approx \lambda C_{\min}(L_0-L_\infty)/L(C_{\min})$. Extrapolating as a power law to $C_{\max} = 100 C_{\min}$: predicted improvement $(C_{\max}/C_{\min})^{\hat\alpha} = 100^{\hat\alpha}$, but the saturating model's actual improvement is bounded by $(L(C_{\min}) - L_\infty)/L(C_{\min}) < 1$ — overestimation factor $100^{\hat\alpha}$ vs. a bounded gain.

(c) Wukong: each depth increment doubles expressible interaction order (Problem 5), so there is no intrinsic ceiling within the compute range tested — the model can always benefit from more layers encoding higher-order interactions. HSTU: generative training maintains supervision density $\rho = \bar{n} \rho_0$ regardless of parameter count (Problem 16), preventing the undertrained-parameter saturation that hits DLRMs.

---

### Problem 18: Memory Footprint: HSTU vs. Standard Transformer

**Key insight:** HSTU's $14d$ vs. $33d$ per-token activation memory comes from eliminating the $n \times n$ softmax matrix (largest term for long sequences) and fusing the FFN sublayer into the gated output, each saving roughly $4d$–$8d$ in stored activations.

**Sketch:**

(a) Standard Transformer activations per token (bfloat16, units of $d$): input $d$; QKV projections $3d$; attention scores $n$ (not $d$-scaled, but per-token share); softmax output $n$; context vector $d$; attention output projection $d$; FFN input $d$; FFN intermediate $4d$; FFN output $d$. Dominant per-token $d$-units: $1 + 3 + 1 + 1 + 1 + 4 + 1 = 12d$ plus $2n$ (score matrices). For large $n$, the score matrices dominate; normalizing per-token: $\approx 33d$ per the note's accounting.

(b) HSTU activations: $X$ input $d$; $W_1 X$ projected $4d$ (split into $U,V,Q,K$: $d$ each); $QK^\top$ pre-activation $n$ per token; $\varphi_2(QK^\top)V$ aggregated $d$; $\text{Norm}(AV) \odot U$ gated $d$; output $d$. Total $d$-units: $1 + 4 + 1 + 1 + 1 = 8d$ plus score activations. Savings: no separate FFN ($-4d$), no softmax intermediate ($-2d$ in $d$-units), fused gating eliminates a residual add ($-3d$ in the $33d$ accounting). Net: $\approx 14d$.

(c) Transformer layers on GPU: $\lfloor \mathcal{M} / (33 d \cdot n \cdot 2) \rfloor$ (factor 2 for bfloat16). HSTU: $\lfloor \mathcal{M} / (14 d \cdot n \cdot 2) \rfloor$. For $\mathcal{M} = 80 \times 10^9$, $d = 1024$, $n = 4096$: Transformer: $\lfloor 80\times10^9 / (33 \times 1024 \times 4096 \times 2) \rfloor = \lfloor 80\times10^9 / (2.76\times10^8) \rfloor \approx 289$ layers. HSTU: $\lfloor 80\times10^9 / (14 \times 1024 \times 4096 \times 2) \rfloor \approx \lfloor 80\times10^9 / (1.17\times10^8) \rfloor \approx 683$ layers — roughly $2.4\times$ more layers for the same memory budget.

---

## Algorithmic Applications

### Problem 19: Optimized FM Block: Pseudocode and Shape Annotations

**Key insight:** The associativity of matrix multiplication makes the difference between $O(n^2 dk)$ and $O(ndk)$; the shape assertions at reshape time are the only place where the FMB can fail silently if $n_F \cdot d$ does not match the MLP output size.

**Sketch:**

```
def FMB(X, Y, mlp_weights, n_F, d):
    # X: (n, d),  Y: (n, k)

    # Optimized FM: right-to-left
    XtY = X.T @ Y          # (d, k)  — cost O(ndk)
    FM_out = X @ XtY       # (n, k)  — cost O(ndk)

    flat = FM_out.flatten()     # (n*k,)
    flat = layer_norm(flat)     # (n*k,)

    # MLP: layers in mlp_weights map (n*k) -> ... -> (n_F * d)
    h = flat
    for W, b in mlp_weights:
        h = relu(h @ W + b)
    # h: (n_F * d)

    assert h.shape == (n_F * d,), \
        f"MLP output {h.shape} != expected ({n_F * d},)"

    return h.reshape(n_F, d)    # (n_F, d)
```

The assertion is necessary because the next layer's FM computes $X_{i+1} X_{i+1}^\top$; if the output has shape $(n_F', d')$ with $n_F' \neq n_F$ or $d' \neq d$, the LCB matrix $W_L \in \mathbb{R}^{n_L \times n_i}$ will have incompatible dimensions.

For part (b), annotate the two orderings with FLOP counts: left-to-right $O(n^2 d) + O(n^2 k)$; right-to-left $O(ndk) + O(ndk) = O(ndk)$. Ratio: $n/k$ — always $>1$ since $k < n$ by design.

---

### Problem 20: HSTU Forward Pass: Pseudocode and Complexity

**Key insight:** The causal mask zeros out the upper triangle of the $n \times n$ attention matrix but does not reduce asymptotic complexity; the QK product still materializes all $n^2$ entries (zeroed entries still require computation or a branch), and the dominant cost remains $O(n^2 d)$ from value aggregation.

**Sketch:**

```
def HSTU_layer(X, W1, W2, rab, causal=True):
    # X: (n, d)
    n, d = X.shape

    # Sub-layer 1: Pointwise projection
    Z = silu(X @ W1)        # (n, 4d)  — cost O(nd * 4d) = O(nd²)
    U, V, Q, K = split(Z, 4, axis=1)  # each (n, d)

    # Sub-layer 2: Spatial aggregation
    scores = Q @ K.T + rab  # (n, n)   — cost O(n²d)
    if causal:
        mask = tril(ones(n, n))
        scores = scores * mask + (-inf) * (1 - mask)
    A = silu(scores)        # (n, n)   elementwise
    AV = A @ V              # (n, d)   — cost O(n²d)

    # Sub-layer 3: Pointwise transformation
    AV_norm = layer_norm(AV)    # (n, d)
    gated = AV_norm * U         # (n, d)  elementwise (Hadamard)
    Y = gated @ W2              # (n, d)  — cost O(nd²)
    return Y
```

Complexity per sub-layer: (i) $W_1$ projection: $O(nd^2)$; (ii) QK product: $O(n^2 d)$; (iii) value aggregation $AV$: $O(n^2 d)$; (iv) gating: $O(nd)$; (v) $W_2$: $O(nd^2)$. For large $n$: QK and $AV$ dominate ($O(n^2 d)$). For large $d$: projections $W_1, W_2$ dominate ($O(nd^2)$). Causal mask does not change complexity since the masked computation is not skipped in dense matrix multiplication.

---

### Problem 21: Stochastic Length Sampling: Implementation Sketch

**Key insight:** The importance weight must be the reciprocal of the sampling probability of the *executed* trajectory to ensure unbiasedness; for the full-sequence branch, weight $= n_i^2 / N^\alpha$; for the truncated branch, weight $= 1/(1 - N^\alpha/n_i^2)$ after normalization.

**Sketch:**

```
def stochastic_length(seq, N, alpha):
    n = len(seq)
    budget = N ** alpha
    threshold = N ** (alpha / 2)
    if n <= threshold:
        return seq, 1.0          # always full; weight = 1

    p_full = budget / (n * n)    # = N^alpha / n_i^2
    if random() < p_full:
        return seq, 1.0 / p_full     # full seq; IS weight
    else:
        sub = random_subsample(seq, int(threshold))
        return sub, 1.0 / (1 - p_full)

def batch_stochastic_length(batch, N, alpha):
    results, weights = [], []
    for seq in batch:
        s, w = stochastic_length(seq, N, alpha)
        results.append(s)
        weights.append(w)
    # Pack into ragged batch
    max_len = max(len(s) for s in results)
    padded = pad_sequences(results, max_len)
    attn_mask = build_attention_mask(results)  # True where valid
    return padded, attn_mask, weights
```

The key challenge in `batch_stochastic_length` is that after SL, sequences have different lengths; a dense padded batch wastes compute on padding tokens. The attention mask must be per-sequence (ragged), and FlashAttention-style kernels must support variable-length sequences to avoid the padding overhead.

---

### Problem 22: M-FALCON Attention Mask Construction

**Key insight:** The M-FALCON mask is a block matrix where the history-history block is lower-triangular (causal), the candidate-history block is all-ones (candidates see full history), and the candidate-candidate block is all-zeros (candidates are independent).

**Sketch:**

Block diagram of $M \in \{0,1\}^{(n+b_m)\times(n+b_m)}$:

```
          history (n)   candidates (b_m)
history:  [ tril(1)   |      0          ]
cands:    [   1       |      0          ]
```

```
def mfalcon_mask(n, b_m):
    total = n + b_m
    M = zeros(total, total)
    M[:n, :n] = tril(ones(n, n))       # causal history self-attn
    M[n:, :n] = ones(b_m, n)           # candidates attend to history
    # M[n:, n:] = 0 already (no candidate-to-candidate)
    return M

def compute_rab_mfalcon(hist_times, cand_time, n, b_m):
    total = n + b_m
    rab = zeros(total, total)
    all_times = concat(hist_times, repeat(cand_time, b_m))
    for i in range(total):
        for j in range(total):
            if M[i,j]:  # only fill where attention is allowed
                pos_bias = b_pos[abs(i - j)]
                dt = abs(all_times[i] - all_times[j])
                t_bucket = floor(log2(dt + 1))
                rab[i, j] = pos_bias + b_t[t_bucket]
    return rab
```

Non-zero entries in $M$: $n(n+1)/2 + b_m \cdot n$. Attention cost $\propto$ non-zero entries times $d$: $O(n^2 d/2 + b_m n d) = O(n^2 d + b_m n d)$, matching Problem 12(b).

---

### Problem 23: Ablation Study: Gradient Flow Through LCB and Residual

**Key insight:** When the FMB Jacobian $\approx 0$ (saturated activations or zero-initialized weights), the only surviving gradient paths are through the LCB and residual; removing both eliminates all gradient signal at the input, while removing only one still leaves a nonzero linear path.

**Sketch:**

```
def two_layer_wukong(X, W_F, W_L, use_lcb, use_residual):
    # Layer 1
    F1 = relu(X @ X.T @ W_F)     # FMB: (n, n)
    L1 = W_L @ X if use_lcb else zeros_like(X)
    R1 = X if use_residual else zeros_like(X)
    X1 = F1 + L1 + R1
    # Layer 2
    F2 = relu(X1 @ X1.T @ W_F)
    L2 = W_L @ X1 if use_lcb else zeros_like(X1)
    R2 = X1 if use_residual else zeros_like(X1)
    return F2 + L2 + R2

def measure_gradient_norm(X, use_lcb, use_residual):
    W_F = zeros(n, n) + 1e-8     # near-zero: dF/dX ≈ 0
    W_L = randn(n, n) * 0.1
    X.requires_grad = True
    out = two_layer_wukong(X, W_F, W_L, use_lcb, use_residual)
    loss = out.sum()
    loss.backward()
    return norm(X.grad)
```

Predicted gradient norms (from Problem 7(c)):
- LCB + residual: $\|((W_L + I)^2)\|_F$ — large, both paths active.
- LCB only (no residual): $\|W_L^2\|_F$ — moderate if $W_L$ is well-conditioned.
- Residual only (no LCB): $\|I^2\|_F = \sqrt{n}$ — nonzero, identity preserves gradient.
- Neither: $\approx 0$ — no linear pathway; gradient vanishes.

Ranking (best to worst): (LCB + residual) $\geq$ (residual only) $\geq$ (LCB only) $\gg$ (neither). This matches the ablation: "substantial degradation" only when both are removed. The gradient-flow argument cannot fully explain why LCB-only is slightly worse than residual-only in practice, since that depends on the specific conditioning of $W_L$ relative to $I$ — an optimization-dependent effect not captured by the worst-case Jacobian analysis.
