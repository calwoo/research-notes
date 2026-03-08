# DHEN: Solutions

## Table of Contents

- [[#Mathematical Development|Mathematical Development]]
  - [[#Problem 1 FM Complexity Reduction via the Kernel Trick|Problem 1: FM Complexity Reduction via the Kernel Trick]]
  - [[#Problem 2 DCN Cross Network Polynomial Degree via Induction|Problem 2: DCN Cross Network Polynomial Degree via Induction]]
  - [[#Problem 3 The Full Jacobian of a DCN Cross Layer|Problem 3: The Full Jacobian of a DCN Cross Layer]]
  - [[#Problem 4 Self-Attention as Data-Dependent Feature Recombination|Problem 4: Self-Attention as Data-Dependent Feature Recombination]]
  - [[#Problem 5 Bit-Level vs. Feature-Level Interaction|Problem 5: Bit-Level vs. Feature-Level Interaction]]
  - [[#Problem 6 Gradient Flow Through Residual DHEN Layers|Problem 6: Gradient Flow Through Residual DHEN Layers]]
  - [[#Problem 7 Vanishing Gradient Bound for Residual vs. Plain Stacks|Problem 7: Vanishing Gradient Bound for Residual vs. Plain Stacks]]
  - [[#Problem 8 Normalized Entropy as a Calibration Measure|Problem 8: Normalized Entropy as a Calibration Measure]]
  - [[#Problem 9 NE and Mutual Information|Problem 9: NE and Mutual Information]]
  - [[#Problem 10 The k^N Interaction Composition Count|Problem 10: The k^N Interaction Composition Count]]
  - [[#Problem 11 Ensemble Variance Reduction|Problem 11: Ensemble Variance Reduction]]
  - [[#Problem 12 Ensemble Stabilization of Slow-Converging Modules|Problem 12: Ensemble Stabilization of Slow-Converging Modules]]
  - [[#Problem 13 Why Linear Can Outperform DCN|Problem 13: Why Linear Can Outperform DCN]]
  - [[#Problem 14 DHEN vs. MoE: Interaction Diversity vs. Capacity|Problem 14: DHEN vs. MoE: Interaction Diversity vs. Capacity]]
  - [[#Problem 15 HSDP Communication Latency Analysis|Problem 15: HSDP Communication Latency Analysis]]
  - [[#Problem 16 Shortcut Aggregation and Dimension Matching|Problem 16: Shortcut Aggregation and Dimension Matching]]
  - [[#Problem 17 Convolution as Local Co-occurrence Capture|Problem 17: Convolution as Local Co-occurrence Capture]]
- [[#Algorithmic Applications|Algorithmic Applications]]
  - [[#Problem 18 DCN Cross Layer as a Bilinear Operation|Problem 18: DCN Cross Layer as a Bilinear Operation]]
  - [[#Problem 19 DHEN Forward Pass with Tensor Shapes|Problem 19: DHEN Forward Pass with Tensor Shapes]]
  - [[#Problem 20 Numerically Stable Distributed NE Computation|Problem 20: Numerically Stable Distributed NE Computation]]
  - [[#Problem 21 HSDP Forward and Backward Pass Pseudocode|Problem 21: HSDP Forward and Backward Pass Pseudocode]]
  - [[#Problem 22 MoE-DHEN Hybrid Architecture|Problem 22: MoE-DHEN Hybrid Architecture]]

---

## Mathematical Development

### Problem 1: FM Complexity Reduction via the Kernel Trick

**Key insight:** The squared-norm identity converts the $O(m^2)$ sum over pairs into a single $O(m)$ accumulation by exploiting the bilinearity of the inner product.

**Sketch:**

(a) Expand $\|\sum_i x_i v_i\|^2 = \sum_i x_i^2 \|v_i\|^2 + 2\sum_{i < j} x_i x_j \langle v_i, v_j \rangle$. Rearranging gives the identity. The right-hand side requires one pass to compute $s = \sum_i x_i v_i$ ($O(md)$), one pass for $\sum_i x_i^2 \|v_i\|^2$ ($O(md)$), and one norm ($O(d)$): total $O(md)$ vs. the naive $O(m^2 d)$.

(b) For $x_i \in \{0,1\}$, $x_i^2 = x_i$, so the identity becomes $\sum_{i < j, i,j \in \mathcal{A}} \langle v_i, v_j \rangle = \frac{1}{2}[\|\sum_{i \in \mathcal{A}} v_i\|^2 - \sum_{i \in \mathcal{A}} \|v_i\|^2]$. This is exactly "sum active embeddings, square, subtract self-norms," recovering DLRM's sum-pooling interaction.

(c) The FM trick output is $\sum_{i<j} s_{ij} = \frac{1}{2}[\|\sum x_i v_i\|^2 - \sum x_i^2 \|v_i\|^2]$, which is a fixed linear combination of $\{s_{ij}\}$ with all coefficients 1. Retaining the full vector $(s_{ij})_{i<j} \in \mathbb{R}^{\binom{m}{2}}$ and applying an MLP allows learning any function $f: \mathbb{R}^{\binom{m}{2}} \to \mathbb{R}$, a strictly richer function class.

---

### Problem 2: DCN Cross Network Polynomial Degree via Induction

**Key insight:** Each cross step multiplies the current iterate by the scalar $x_\ell^\top w_\ell$, raising the polynomial degree by exactly 1, so the degree grows linearly with layer count and is tight.

**Sketch:**

(a) Base case: $x_1 = x_0(x_0^\top w_0) + b_0 + x_0$; the term $x_0(x_0^\top w_0)$ has degree 2 in $x_0$ components. Inductive step: $x_{\ell+1} = x_0(x_\ell^\top w_\ell) + b_\ell + x_\ell$. The product $x_0 \cdot (x_\ell^\top w_\ell)$ multiplies degree-1 ($x_0$) by degree-$\ell$ ($x_\ell^\top w_\ell$), giving degree $\ell+1$. The term $x_\ell$ has degree $\ell$. So $\deg(x_{\ell+1}) = \ell + 1$.

(b) For degree $\ell+2$ to appear in $x_{\ell+1}$, we would need $\deg(x_0) \cdot \deg(x_\ell^\top w_\ell) \geq \ell + 2$, i.e., $1 \cdot \ell \geq \ell+1$, which is false. So the upper bound $\ell+1$ is tight.

(c) In an $N$-layer DHEN, the DCN path at layer $n$ takes a degree-$D_n$ input and outputs degree $D_n (L+1)$ (each of its $L$ cross layers multiplies the degree by one more factor of the input). So $D_{n+1} = D_n \cdot (L+1)$, giving $D_N = (L+1)^N$. A plain depth-$NL$ DCN achieves degree $NL + 1$. Since $(L+1)^N \gg NL + 1$ for $N, L \geq 2$, hierarchical stacking yields super-linear degree growth.

---

### Problem 3: The Full Jacobian of a DCN Cross Layer

**Key insight:** The rank-1 update structure $J_\ell = I + x_0 w_\ell^\top$ follows directly from differentiating the cross recurrence, and the Sherman-Morrison formula then gives the inverse and singular-value structure in closed form.

**Sketch:**

(a) $x_{\ell+1} = x_0(x_\ell^\top w_\ell) + b_\ell + x_\ell$. Differentiating with respect to $x_\ell$: $\frac{\partial x_{\ell+1}}{\partial x_\ell} = x_0 w_\ell^\top + I$. This is exactly $J_\ell = I + x_0 w_\ell^\top$.

(b) Matrix determinant lemma: $\det(I + uv^\top) = 1 + v^\top u$. So $\det(J_\ell) = 1 + w_\ell^\top x_0$; $J_\ell$ is invertible iff $w_\ell^\top x_0 \neq -1$. By Sherman-Morrison: $J_\ell^{-1} = I - \frac{x_0 w_\ell^\top}{1 + w_\ell^\top x_0}$.

(c) $J_\ell = I + x_0 w_\ell^\top$ is a rank-1 perturbation of the identity. All singular vectors orthogonal to both $x_0$ and $w_\ell$ have singular value 1. The remaining two-dimensional subspace has singular values $\approx 1 \pm \|x_0\|\|w_\ell\|$ (to first order). Since all singular values are $O(1)$ for bounded weights, the cross layer cannot cause exponential gradient explosion or vanishing.

---

### Problem 4: Self-Attention as Data-Dependent Feature Recombination

**Key insight:** The attention weight $A_{ij}$ is a softmax-normalized inner product of projected features, making the output a rational function of the input — not a polynomial — which puts attention outside the expressive class of any DCN with finite depth.

**Sketch:**

(a) $A_{ij} = \frac{\exp(q_i k_j^\top / \sqrt{d_k})}{\sum_{j'} \exp(q_i k_{j'}^\top / \sqrt{d_k})}$ where $q_i = x_i W^Q$ and $k_j = x_j W^K$. Each $q_i k_j^\top = x_i W^Q (W^K)^\top x_j^\top$ is a scalar quadratic in the input, making $A_{ij}$ a ratio of exponentials of quadratics in $\{x_j\}$.

(b) $z_i = \sum_j A_{ij} x_j W^V$ is a convex combination (weights sum to 1) of value vectors. The weights $A_{ij}$ depend on all inputs $\{x_{j'}\}$ through the softmax denominator, making $z_i$ a non-polynomial rational function of the entire input matrix $X$.

(c) DCN computes $u = W(XX^\top) + b$ where entry $(i,j)$ of $XX^\top$ is $\langle x_i, x_j \rangle = \sum_r x_i[r] x_j[r]$ — a sum over all $d$ embedding dimensions. Attention computes $q_i k_j^\top = \sum_r (x_i W^Q)_r (x_j W^K)_r$ — a sum in the projected subspace of dimension $d_k < d$. DCN's interaction weight is a fixed bilinear form in the full embedding space; attention's is a learned bilinear form in a lower-dimensional subspace, but made adaptive via softmax normalization.

---

### Problem 5: Bit-Level vs. Feature-Level Interaction

**Key insight:** DCN's scalar $x^\top w$ scales the entire vector $x_0$ uniformly (bit-level: all embedding dimensions get the same multiplicative update), while attention's $A_{ij}$ scales entire feature vectors $x_j$ (feature-level: all embedding dimensions of feature $j$ are weighted together).

**Sketch:**

(a) DCN cross: $x_{\ell+1}[r] = x_0[r] \cdot (x_\ell^\top w) + x_\ell[r]$ for all dimensions $r$. The scalar $x_\ell^\top w = \sum_s x_\ell[s] w[s]$ mixes all dimensions $s$ of $x_\ell$ into one number, then this single number multiplies every dimension $r$ of $x_0$. In bilinear form: $x_0^\top M x$ with $M = w \mathbf{1}^\top$ (rank 1, same row for all output dimensions).

(b) Attention: $[z_i]_r = \sum_j A_{ij} \sum_s x_j[s] W^V[s,r]$. The embedding dimension mixing $\sum_s x_j[s] W^V[s,r]$ is governed by $W^V$ and is the same for every feature $j$; the per-feature weighting $A_{ij}$ does not mix embedding dimensions. So attention weights whole feature vectors, not individual scalars.

(c) DCN can represent: "global importance weighting" — scaling all output dimensions by the same feature cross score. Attention cannot do this in a single layer (it must use $W^V$ uniformly). Attention can represent: "select a specific feature subset based on content" (e.g., $A_{ij} \approx 1$ for one $j$ and 0 elsewhere), which DCN cannot do (DCN's output is always a function of $x_0$ scaled by a global scalar). These examples confirm the modules are incomparable and complementary.

---

### Problem 6: Gradient Flow Through Residual DHEN Layers

**Key insight:** The identity shortcut means $\frac{\partial X_{n+1}}{\partial X_n} = I + J_n$, so the product of Jacobians across all layers always contains the identity path ($S = \emptyset$ term) which by itself transmits the full gradient signal from output to input.

**Sketch:**

(a) $\frac{\partial X_{n+1}}{\partial X_n} = \frac{\partial}{\partial X_n}[F_n(X_n) + X_n] = J_n + I$.

(b) $\prod_{n=0}^{N-1}(I + J_n) = \sum_{S \subseteq [N]} \prod_{n \in S} J_n$ holds when expanding the product by choosing either $I$ or $J_n$ at each factor (valid for any matrices, not just commuting ones — the sum over subsets exactly enumerates all $2^N$ choices). The $S = \emptyset$ term is $I$, contributing $\frac{\partial \mathcal{L}}{\partial X_N}$ directly to $\frac{\partial \mathcal{L}}{\partial X_0}$.

(c) LayerNorm Jacobian: $\frac{\partial \operatorname{Norm}(z)}{\partial z} = \frac{1}{\sigma}P_\perp$ where $P_\perp = I - \frac{1}{d}\mathbf{1}\mathbf{1}^\top - \hat{z}\hat{z}^\top$ projects out the mean and $\hat{z}$ directions. This has $d-2$ nonzero eigenvalues equal to $1/\sigma$. The identity shortcut bypasses LayerNorm, so the $S = \emptyset$ path still carries the full gradient even when LayerNorm attenuates the ensemble-path contribution.

---

### Problem 7: Vanishing Gradient Bound for Residual vs. Plain Stacks

**Key insight:** For contractive maps ($\|J_n\|_\text{op} < 1$), the plain stack gradient decays as $\sigma^N \to 0$, while the residual stack's $S = \emptyset$ term gives a norm-1 lower bound, making the gradient ratio at least 1 regardless of depth.

**Sketch:**

(a) $\|\prod_{n=0}^{N-1} J_n\|_F \leq \prod_n \|J_n\|_\text{op} \leq \sigma^N$. So $\|\frac{\partial \mathcal{L}}{\partial X_0}\|_F \leq \sigma^N \|\frac{\partial \mathcal{L}}{\partial X_N}\|_F \to 0$ as $N \to \infty$.

(b) The $S = \emptyset$ term in the residual expansion is $I$, contributing a term of Frobenius norm $\|\frac{\partial \mathcal{L}}{\partial X_N}\|_F$ to $\|\frac{\partial \mathcal{L}}{\partial X_0}\|_F$. By the triangle inequality applied to the sum: $\|\frac{\partial \mathcal{L}}{\partial X_0}\|_F \geq \|\frac{\partial \mathcal{L}}{\partial X_N}\|_F - \|\text{other terms}\|_F$. More precisely, the identity path alone gives $\|\frac{\partial \mathcal{L}}{\partial X_0}\|_F \geq \|\frac{\partial \mathcal{L}}{\partial X_N}\|_F$ only as an upper bound on the $S = \emptyset$ contribution; the key point is this term never vanishes.

(c) $\sigma^8 = 0.9^8 \approx 0.43$. So a plain 8-layer stack attenuates gradients to 43% of their original magnitude; 22-layer stacks give $0.9^{22} \approx 0.098$ (10% attenuation). With residual shortcuts the gradient from the identity path remains at 100%, explaining why DHEN can train stably at $N = 8$ and the paper's HSDP experiments push to $N = 22+$.

---

### Problem 8: Normalized Entropy as a Calibration Measure

**Key insight:** NE normalizes binary cross-entropy by the entropy of the marginal click rate, so NE = 1 exactly when the model's conditional predictions add zero mutual information over the base rate predictor.

**Sketch:**

(a) Numerator of NE: $-\frac{1}{n}\sum_i[y_i \log p + (1-y_i)\log(1-p)]$. Since $\frac{1}{n}\sum_i y_i = p$: this equals $-(p \log p + (1-p)\log(1-p)) = H(p)$. Denominator is $H(p)$. So $\text{NE} = H(p)/H(p) = 1$.

(b) Cross-entropy decomposes as $H(\hat{p}, y) = H(p) + D_\text{KL}(p_\text{data} \| p_\theta) + \text{calibration gap}$ (standard result from information theory). Dividing by $H(p)$ gives $\text{NE} = 1 + (D_\text{KL} + \text{calib. gap})/H(p)$. Thus $\text{NE} < 1$ iff $D_\text{KL} + \text{calib. gap} < 0$, which is impossible for a calibrated model — actually $\text{NE} < 1$ iff $D_\text{KL} > 0$ and calibration gap is handled correctly, meaning the model genuinely discriminates labels.

(c) $\text{NE} = 1$ exactly when $\hat{p}_i = p$ for all $i$ (from part a), equivalently when the model's predictions are independent of all input features. This is the condition $I(y; \hat{p} \mid \text{context}) = 0$: predictions carry zero information about labels beyond the marginal.

---

### Problem 9: NE and Mutual Information

**Key insight:** For a perfectly calibrated model, the NE equals $1 - I(y;\hat{p})/H(p)$, so NE improvement directly measures the gain in normalized mutual information between predictions and labels.

**Sketch:**

(a) For a perfectly calibrated model, the calibration gap is zero. The cross-entropy decomposition gives $H(\hat{p}, y) = H(p) - I(y; \hat{p})$ (using $H(y \mid \hat{p}) = H(y) - I(y;\hat{p})$ and calibration meaning $H(y|\hat{p}_\theta) = H(\hat{p}_\theta)$). Dividing by $H(p)$: $\text{NE} = 1 - I(y;\hat{p})/H(p)$.

(b) $\text{NE}_1 - \text{NE}_2 = (1 - I_1/H(p)) - (1 - I_2/H(p)) = (I_2 - I_1)/H(p)$, directly the normalized mutual information gain.

(c) $H(0.05) = -(0.05 \ln 0.05 + 0.95 \ln 0.95) \approx 0.199$ nats. DHEN's 0.27% NE improvement corresponds to $\Delta I = 0.0027 \times 0.199 \approx 0.000537$ nats of additional mutual information. This is a small absolute gain, but relative to the information-theoretic ceiling $H(p) = 0.199$ nats, it represents about 0.27% of the maximum achievable gain — meaningful at the scale of industrial CTR systems.

---

### Problem 10: The k^N Interaction Composition Count

**Key insight:** In a $k$-module, $N$-layer DHEN, each of the $k^N$ length-$N$ sequences of module choices corresponds to a distinct functional composition, and sum aggregation mixes them in a fixed linear combination while concatenation aggregation exposes them individually to the next layer.

**Sketch:**

(a) $X_1 = I_1(X_0) + I_2(X_0) + X_0$ (with shortcut). $X_2 = I_1(X_1) + I_2(X_1) + X_1$. Substituting: $X_2 = I_1(I_1(X_0)+I_2(X_0)+X_0) + I_2(I_1(X_0)+I_2(X_0)+X_0) + I_1(X_0)+I_2(X_0)+X_0$. The distinct composition terms are $I_1 \circ I_1$, $I_1 \circ I_2$, $I_2 \circ I_1$, $I_2 \circ I_2$ plus lower-order terms: 4 distinct pairwise compositions plus $I_1, I_2$, identity = 7 terms total (more if shortcut interactions count separately).

(b) With concatenation: $X_1 = [I_1(X_0) \| I_2(X_0)]$ presents both module outputs as distinct channels. When $I_1$ at layer 2 receives $X_1$, it jointly processes both $I_1(X_0)$ and $I_2(X_0)$ as separate features — it can learn to weight them differently. With sum, $I_1$ at layer 2 sees only $I_1(X_0) + I_2(X_0)$ and cannot distinguish which part came from which module.

(c) By induction: with $k$ modules and sum aggregation at each layer, layer $n$'s output is a sum of all $k^n$ distinct $n$-deep compositions. The count doubles each time we add a layer: $k^N$ distinct paths. A homogeneous stack with 1 module type produces only the single composition $I^N$ at depth $N$. For $k=5$, $N=4$: $5^4 = 625$ distinct composition paths.

---

### Problem 11: Ensemble Variance Reduction

**Key insight:** Ensemble variance equals the average pairwise covariance of module predictions; uncorrelated modules reduce variance by $1/k$, and lower inter-module correlation (the non-overlapping information hypothesis) is directly what drives variance reduction.

**Sketch:**

(a) $\hat{y}_\text{ens} = \frac{1}{k}\sum_i \hat{y}_i$. $\mathbb{E}[(y - \hat{y}_\text{ens})^2] = \mathbb{E}[(y - \frac{1}{k}\sum_i \hat{y}_i)^2]$. Expand: $= \mathbb{E}[y^2] - \frac{2}{k}\sum_i \mathbb{E}[y \hat{y}_i] + \frac{1}{k^2}\sum_{i,j}\mathbb{E}[\hat{y}_i \hat{y}_j]$. Recognizing that $\mathbb{E}[\hat{y}_i \hat{y}_j] = \operatorname{Cov}(\hat{y}_i, \hat{y}_j) + \mathbb{E}[\hat{y}_i]\mathbb{E}[\hat{y}_j]$ and collecting terms gives the stated form.

(b) With $\operatorname{Cov}(\hat{y}_i, \hat{y}_j) = 0$ for $i \neq j$ and $\operatorname{Var}(\hat{y}_i) = V$: ensemble variance $= V/k$. For negative correlation $\rho < 0$, the cross-terms decrease the ensemble variance further; for $\rho = 1$ (identical predictions), ensemble variance $= V$ (no reduction).

(c) $\operatorname{Cov}(\hat{y}_i, \hat{y}_j) = \rho V$ for $i \neq j$. Ensemble variance $= \frac{1}{k^2}[kV + k(k-1)\rho V] = V\frac{1 + (k-1)\rho}{k}$. For $\rho=0.5$, $k=2$: variance ratio $= \frac{1 + 0.5}{2} = 0.75$. So the ensemble has 75% the variance of a single module — consistent with the ablation showing attention+linear outperforms either alone by combining two modules with complementary predictions.

---

### Problem 12: Ensemble Stabilization of Slow-Converging Modules

**Key insight:** The shared gradient $\frac{\partial \mathcal{L}}{\partial Y}$ becomes smoother when the linear module provides an accurate baseline prediction early in training, reducing gradient variance and allowing the attention module to learn from a cleaner signal.

**Sketch:**

(a) With $Y = I_\text{attn}(X) + I_\text{lin}(X) + \text{ShortCut}(X)$, the gradient to each module is $\frac{\partial \mathcal{L}}{\partial I_j(X)} = \frac{\partial \mathcal{L}}{\partial Y}$ (same for all modules by chain rule). The variance of $\frac{\partial \mathcal{L}}{\partial Y}$ is lower when $Y$ is a better predictor (lower loss residuals). Since $I_\text{lin}$ contributes a reasonable early-training prediction, the ensemble $Y$ is closer to the truth than $I_\text{attn}(X)$ alone, reducing $\operatorname{Var}(\frac{\partial \mathcal{L}}{\partial Y})$.

(b) At initialization with small $W^V$: $I_\text{attn}(X) \approx 0$ (uniform attention gives $z_i \approx \frac{1}{m}\sum_j x_j W^V \approx 0$). Near-identity linear: $I_\text{lin}(X) \approx X$. So $Y \approx X + \text{ShortCut}(X) \approx 2X$, a reasonable baseline. The gradient to $\theta_\text{attn}$ starts from a meaningful signal.

(c) ResNet analogy: at initialization, ResNet behaves like a shallow network because $F_n \approx 0$ (small random init) and $X_{n+1} \approx X_n$. The effective depth gradually increases as $F_n$ learns. DHEN analogously behaves like "linear module + shortcut = identity" at init; attention gradually becomes non-trivial. Formal correspondence: DHEN's $I_\text{lin}$ plays the role of ResNet's identity shortcut; DHEN's $I_\text{attn}$ plays the role of ResNet's residual branch $F_n$.

---

### Problem 13: Why Linear Can Outperform DCN

**Key insight:** In production feature spaces with pre-computed crosses already present in the input embeddings, DCN recomputes information already in the input (feature redundancy), while its bilinear coupling also creates ill-conditioned gradients for sparse embeddings (optimization landscape).

**Sketch:**

(a) If $X_0$ contains embedding entries corresponding to feature pairs $(i,j)$ — i.e., $X_0 = [v_1, \ldots, v_m, v_{12}, v_{13}, \ldots]$ with $v_{ij}$ a pre-computed cross embedding — then DCN's degree-2 term $x_0(x_0^\top w) \in \text{span}(x_0)$, and cross terms in $x_0^\top w$ are linear combinations of the already-available $v_{ij}$ entries. DCN adds nothing not already in $\text{span}(X_0)$, while the linear module $WX_0$ accesses all of $\text{span}(X_0)$ directly.

(b) Hessian of loss w.r.t. $w_\ell$: $\frac{\partial^2 \mathcal{L}}{\partial w_\ell^2} \propto x_0 x_0^\top$. The condition number of this Hessian is $\|x_0\|^2 / \lambda_\text{min}(x_0 x_0^\top)$. For sparse embeddings, $x_0$ varies across features (some near zero, some large), making the effective Hessian highly ill-conditioned. The linear module's Hessian is $X_0 X_0^\top$ (a full-rank matrix in expectation), better conditioned.

(c) DCN would outperform linear when: features are raw one-hot encodings with no pre-computed crosses; the corpus is small enough that the crossed features have not appeared frequently enough to be estimated well; and the optimizer is heavily regularized (forcing the linear module toward near-zero weights while DCN can still learn the explicit crossing via $x_0 x_0^\top w$).

---

### Problem 14: DHEN vs. MoE: Interaction Diversity vs. Capacity

**Key insight:** MoE uses sparsity (each example sees one expert) to amortize capacity across sub-populations; DHEN uses density (each example sees all modules) to maximize interaction diversity per example — these are complementary strategies suited to different data distributions.

**Sketch:**

(a) MoE top-1 routing: expected FLOPs per example $= F_\text{expert}$ (only one expert is active). DHEN: FLOPs per example $= k \cdot F_\text{module}$. At matched total FLOPs budget $F$: MoE can instantiate $E = F / F_\text{expert}$ experts (with each example seeing $F_\text{expert}$ FLOPs); DHEN can instantiate $k = F / F_\text{module}$ modules (each example sees all $k$ modules for $k \cdot F_\text{module} = F$ FLOPs). MoE adds capacity ($E$ distinct functions); DHEN adds diversity (all $k$ functions applied per example).

(b) Formal scenario: suppose $p(\text{data}) = \sum_e \pi_e p_e(\text{data})$ with $E$ sub-populations, each with optimal predictor $f_e^*$ satisfying $\|f_e^* - \sum_j \alpha_j f_j^*\|^2 > \epsilon > 0$ for all $\alpha \in \Delta^E$ ($f_e^*$ are not convex combinations of each other). Then no fixed ensemble of $\{f_e^*\}$ achieves the per-sub-population optimal; only routing (MoE) does.

(c) For $E=2$, $k=2$ (DCN + Linear): MoE-DHEN routes each example to one expert; that expert runs DCN + Linear ensemble. Per-example interaction paths: 2 (one per expert's DCN and Linear). Shared parameters: gating network $g$. Expert-specific parameters: all DCN and Linear weights within each expert. Total distinct paths: $2 \times 2 = 4$ per routing decision, vs. DHEN's $2^N$ across layers.

---

### Problem 15: HSDP Communication Latency Analysis

**Key insight:** HSDP moves the allgather from the slow cross-host network (RoCEv2) to the fast intra-host network (NVLink), reducing critical-path communication latency by the bandwidth ratio $B_\text{NV}/B_\text{net} \approx 24$.

**Sketch:**

(a) FSDP allgather: each GPU must receive all $GH - 1$ other shards, each of size $P/(GH)$ bytes, for a total of $(GH-1) \cdot P/(GH) \approx P$ bytes per GPU. The bottleneck is the cross-host RoCEv2 links (capacity $B_\text{net}$ per GPU). So $L_\text{FSDP} \approx P / B_\text{net}$.

(b) HSDP allgather: confined to $G$ GPUs within one host over NVLink. Each GPU receives $G-1$ shards each of size $P/G$, totalling $(G-1) \cdot P/G \approx P$ bytes, communicated at $B_\text{NV}$. So $L_\text{HSDP} \approx P/B_\text{NV}$. Ratio: $L_\text{FSDP}/L_\text{HSDP} = B_\text{NV}/B_\text{net} \approx 24$.

(c) Peak memory per GPU:
- DP: $sP$ (full parameters + optimizer state) + activations. Total: $sP + \text{activations}$.
- FSDP: $sP/(GH)$ (sharded parameters + optimizer state) + one layer's unsharded weights (materialized during allgather): $\approx sP/(GH) + P_\text{layer}$. In practice $\approx sP/(GH)$ amortized.
- HSDP: $sP/G$ (parameters sharded across $G$ intra-host GPUs) + gradient buffer for async allreduce: $P/G$ extra. Total: $(s+1)P/G$.
- HSDP requires $(s+1)P/G$ vs. FSDP's $sP/(GH)$: HSDP stores $H(s+1)/s$ times more than FSDP, traded for $24\times$ lower critical-path latency.

---

### Problem 16: Shortcut Aggregation and Dimension Matching

**Key insight:** Concatenation aggregation always triggers the projection shortcut (unless $kl = m$ by coincidence), requiring $ml \cdot k$ parameters, while sum aggregation only triggers projection when $l \neq m$, and the projection has $ml$ parameters — a $k$-fold difference.

**Sketch:**

(a) Concatenation output: $[u_1 \| \cdots \| u_k] \in \mathbb{R}^{d \times kl}$. Input: $X_n \in \mathbb{R}^{d \times m}$. Mismatch when $kl \neq m$. Projection $W_n \in \mathbb{R}^{m \times kl}$ has $m \cdot kl$ parameters.

(b) Sum output: $\sum_i u_i \in \mathbb{R}^{d \times l}$. Identity shortcut possible when $l = m$. When $l \neq m$, projection $W_n \in \mathbb{R}^{m \times l}$ has $ml$ parameters (factor $k$ fewer than concatenation case).

(c) When concatenation grows the embedding count ($kl > m$): the shortcut projects $\mathbb{R}^m \to \mathbb{R}^{kl}$, mapping a lower-dimensional input to a higher-dimensional ensemble space — an information expansion (the shortcut creates new embedding slots). ResNet's 1x1 projection shortcut maps between same-dimension feature maps (identity case) or reduces depth in bottleneck blocks. The DHEN expansion shortcut is most analogous to the ResNet learned projection used to match channel dimensions when stride-2 downsampling changes spatial size — both add parameters to handle dimension changes, not to compress information.

---

### Problem 17: Convolution as Local Co-occurrence Capture

**Key insight:** Conv2d applied to the embedding matrix $X_n \in \mathbb{R}^{d \times m}$ captures local correlations among neighboring features and adjacent embedding dimensions, with parameter count independent of $m$ (unlike attention), making it efficient when the number of features is large.

**Sketch:**

(a) A $(k_d, k_m)$ filter computes $\text{out}[r,s] = \sum_{p=0}^{k_d-1}\sum_{q=0}^{k_m-1} w_{pq} X_n[r+p, s+q]$, a weighted sum of a $k_d \times k_m$ patch. This mixes embedding dimensions $r, \ldots, r+k_d-1$ with feature slots $s, \ldots, s+k_m-1$ jointly in the same filter computation.

(b) Locality assumption: features at adjacent columns of $X_n$ have correlated interaction patterns. This holds when features are ordered by semantic similarity or by co-occurrence frequency. Maximally valid ordering: place highly correlated feature groups (e.g., same entity type: all user features together, all ad features together) in contiguous columns. If features are randomly ordered, the locality assumption is violated and convolution degrades toward a global interaction.

(c) Conv2d parameters: $k_d \times k_m \times C_\text{in} \times C_\text{out}$ — independent of $m$ and $d$ (assuming $k_d, k_m \ll d, m$). Self-attention parameters: $3d \cdot d_k \cdot h$ (for $Q, K, V$ projections, $h$ heads) + $d_v h \cdot d$ (output projection) — also independent of $m$. Both scale quadratically with $d$ and linearly with head count/channel count, but convolution scales with filter size not sequence length. For very large $m$, the key difference is computational: attention costs $O(m^2 d_k)$ per forward pass, convolution costs $O(m d k_m k_d)$ — convolution is linear in $m$ while attention is quadratic.

---

## Algorithmic Applications

### Problem 18: DCN Cross Layer as a Bilinear Operation

**Key insight:** The low-rank factorization $W_\ell = U_\ell V_\ell^\top$ reduces the dominant $O(d^2)$ matrix-vector product to two sequential $O(dr)$ projections, cutting cost by factor $d/r$ while preserving the rank-$r$ interaction subspace.

**Sketch:**

(a) **Full cross layer pseudocode:**
```
def cross_layer_full(x0, xl, W, b):
    # W: (d, d), x0, xl, b: (d,)
    s = W @ xl          # O(d^2) -- dominant cost
    s = s + b           # O(d)
    out = x0 * s        # O(d)  element-wise
    out = out + xl      # O(d)  residual
    return out          # shape: (d,)
```
Dominant FLOP cost: $O(d^2)$ for the matrix-vector product.

(b) **Low-rank cross layer:**
```
def cross_layer_lowrank(x0, xl, U, V, b):
    # U, V: (d, r), r << d
    t = V.T @ xl        # O(dr): project to rank-r subspace
    t = U @ t           # O(dr): project back to d-dim
    t = t + b           # O(d)
    out = x0 * t        # O(d)
    out = out + xl      # O(d)
    return out          # shape: (d,)
```
Total: $O(dr)$ vs. $O(d^2)$; speedup factor $d/r$.

(c) **Full DCN module in DHEN:**
```
def dcn_module(Xn, x0_flat, cross_layers, Wm):
    # Xn: (d, m), x0_flat = flatten(Xn): (md,)
    # cross_layers: list of L (U_l, V_l, b_l) tuples
    # Wm: (md, d*l) projection back to embedding list
    xl = x0_flat                            # (md,)
    for (U_l, V_l, b_l) in cross_layers:   # L iterations
        xl = cross_layer_lowrank(x0_flat, xl, U_l, V_l, b_l)
    u_flat = Wm.T @ xl                      # (d*l,): O(md * dl)
    u = reshape(u_flat, (d, l))             # (d, l)
    return u
# Parameters: {U_l in R^{md x r}, V_l in R^{md x r}, b_l in R^{md}} x L; Wm in R^{md x dl}
```

---

### Problem 19: DHEN Forward Pass with Tensor Shapes

**Key insight:** Concatenation aggregation forces the shortcut to apply a learned projection whenever $kl \neq m$; tracking shapes explicitly at each step reveals exactly which operations require projection parameters.

**Sketch:**

(a) **Parameters for 2-layer DHEN (self-attention + linear, concat aggregation):**
```
# Attention: W^Q, W^K, W^V each (d, d_k * h); W^O: (d_v * h, d)
# Linear: W_lin: (m, l)  -- maps embedding count m -> l
# Concatenation output per layer: d x (l_attn + l_lin) = d x 2l  (if l_attn = l_lin = l)
# Shortcut projection (if 2l != m): W_sc: (m, 2l)
# LayerNorm: gamma, beta: (d,) each
# Prediction head: W_head: (d * 2l, hidden), ...
```

(b) **One DHEN layer pseudocode (batch dim B suppressed):**
```
def dhen_layer(Xn, params):
    # Xn: (B, d, m)
    u_attn = attention(Xn, W_Q, W_K, W_V, W_O)  # (B, d, l)
    u_lin  = linear(Xn, W_lin)                   # (B, d, l)
    ensemble = concat([u_attn, u_lin], dim=-1)   # (B, d, 2l)
    if 2l == m:
        shortcut = Xn                             # (B, d, m)
    else:
        shortcut = Xn @ W_sc                      # (B, d, 2l): W_sc (m, 2l)
    Y = LayerNorm(ensemble + shortcut)            # (B, d, 2l)
    return Y
```

(c) **Full 2-layer forward pass:**
```
X0: (B, d, m)          # feature processing output
X1 = dhen_layer(X0)    # (B, d, 2l) -- projection needed if 2l != m
X2 = dhen_layer(X1)    # (B, d, 2l) -- identity shortcut if dims match
flat = reshape(X2, (B, d*2l))
logits = MLP(flat)     # (B, 1)
p_hat = sigmoid(logits)  # (B,)
# Projection required: at layer 1 shortcut (if 2l != m);
# layer 2 shortcut is identity if output dim of layer 1 = 2l.
```

---

### Problem 20: Numerically Stable Distributed NE Computation

**Key insight:** Replace $\log \sigma(s)$ with $-\operatorname{softplus}(-s)$ to avoid numerical underflow near 0/1; then allreduce only scalar sums (CE sum and label sum) to compute NE in $O(1)$ communication volume regardless of local batch size.

**Sketch:**

(a) Each worker $w$ computes local sums $(S_y^{(w)}, S_n^{(w)}) = (\sum_\text{local} y_i, n/W)$. AllReduce-sum gives $S_y = \sum_w S_y^{(w)}$ and $n = \sum_w S_n^{(w)}$. Then $p = S_y / n$. This requires a single AllReduce of 2 scalars.

(b) For logit $s_i$ with $\hat{p}_i = \sigma(s_i)$:
$\log \hat{p}_i = \log \sigma(s_i) = -\log(1 + e^{-s_i}) = -\operatorname{softplus}(-s_i)$
$\log(1-\hat{p}_i) = \log(1 - \sigma(s_i)) = -\log(1 + e^{s_i}) = -\operatorname{softplus}(s_i)$
CE$(y_i, s_i) = \operatorname{softplus}(s_i) - y_i \cdot s_i$ (numerically stable for all $s_i$).

(c) **Complete pseudocode:**
```
def distributed_NE(local_logits, local_labels, n_global):
    # local_logits, local_labels: (n_local,)
    local_ce = softplus(local_logits) - local_labels * local_logits  # (n_local,)
    local_ce_sum = sum(local_ce)            # scalar
    local_label_sum = sum(local_labels)     # scalar

    # AllReduce: sum CE and label counts across all workers
    global_ce_sum = allreduce_sum(local_ce_sum)    # 1 scalar communicated
    global_label_sum = allreduce_sum(local_label_sum)  # 1 scalar communicated

    p = global_label_sum / n_global                # empirical CTR
    H_p = -(p * log(p) + (1-p) * log(1-p))        # NE denominator (nats)
    numerator = global_ce_sum / n_global            # mean cross-entropy
    return numerator / H_p                          # NE
```

---

### Problem 21: HSDP Forward and Backward Pass Pseudocode

**Key insight:** HSDP restricts the critical-path allgather to fast NVLink (intra-host), then asynchronously allreduces gradient shards across hosts over slow RoCEv2, overlapping the network communication with computation of earlier backward layers.

**Sketch:**

(a) **HSDP data structures:**
```
# Each GPU (host h, local rank g) holds:
#   param_shard[g]: P/G parameters (its local shard of all dense DHEN weights)
#   grad_shard[g]:  P/G gradient buffer (for async cross-host allreduce)
#   full_params:    P parameters (materialized during forward via allgather; freed after use)
# No parameters are replicated across hosts; all are uniquely held by one GPU per host.
```

(b) **HSDP forward pass for one layer:**
```
def hsdp_forward_layer(layer_idx, input_activation):
    # Step 1: allgather weight shards from G GPUs on same host (NVLink -- critical path)
    full_weights = intra_host_allgather(param_shard[local_rank])
    # full_weights: P/num_layers parameters for this layer, shape (d_out, d_in)
    # Bandwidth: B_NV; latency: ~P_layer / B_NV

    # Step 2: forward computation with full weights
    output = layer_forward(input_activation, full_weights)

    # Step 3: free full_weights (not stored; recomputed in backward if activation ckpt)
    del full_weights

    return output
```

(c) **HSDP backward pass:**
```
def hsdp_backward_layer(layer_idx, grad_output):
    # Step 1: re-allgather weights (or use activation checkpointing recompute)
    full_weights = intra_host_allgather(param_shard[local_rank])  # NVLink

    # Step 2: compute gradients
    grad_input = layer_backward_input(grad_output, full_weights)
    grad_weight = layer_backward_weight(grad_output, saved_input)  # (P_layer,)

    # Step 3: intra-host reducescatter: each GPU accumulates its shard of grad_weight
    grad_shard_local = intra_host_reducescatter(grad_weight)  # NVLink, critical path

    # Step 4: register async allreduce across hosts (RoCEv2, OFF critical path)
    async_handle = allreduce_async(grad_shard_local, group=same_local_rank_across_hosts)
    pending_allreduces.append(async_handle)

    # Step 5: continue backward (allreduce overlaps with next layer's backward)
    return grad_input

# Before optimizer step: synchronize all pending allreduces
for handle in pending_allreduces:
    handle.wait()
optimizer.step()  # grad_shard now contains globally averaged gradient
```

---

### Problem 22: MoE-DHEN Hybrid Architecture

**Key insight:** MoE-DHEN separates routing specialization (the gating network selects which expert's interaction structure to apply) from interaction diversity (each expert's DHEN ensemble applies multiple module types), combining both benefits at the cost of $E \times$ the parameter count of a single DHEN.

**Sketch:**

(a) **Architecture specification:**
```
# Shared: gating network g: R^{d x m} -> Delta^E
#   g(Xn) = softmax(W_gate * flatten(Xn))  -- W_gate: (E, dm), shared
# Expert-specific (for e = 1..E):
#   DHEN_e: k interaction modules {I_1^e, ..., I_k^e} with their own weights
#   Each module: DCN weights {U_l^e, V_l^e, b_l^e} or Linear weights W_lin^e
# No sharing of interaction module weights across experts.
```

(b) **Forward pass (top-1 routing):**
```
def moe_dhen_layer(Xn, gate, experts):
    # Xn: (B, d, m)
    scores = gate(flatten(Xn))              # (B, E): gating logits
    expert_idx = argmax(scores, dim=1)      # (B,): top-1 selection

    output = zeros_like(Xn)
    for e in range(E):
        mask = (expert_idx == e)            # (B,) boolean
        if mask.any():
            Xn_e = Xn[mask]                # (B_e, d, m): routed sub-batch
            u_dcn  = experts[e].dcn(Xn_e)  # (B_e, d, l)
            u_lin  = experts[e].linear(Xn_e)  # (B_e, d, l)
            u_ens  = concat([u_dcn, u_lin], dim=-1)  # (B_e, d, 2l)
            output[mask] = u_ens + shortcut(Xn_e)    # residual
    return LayerNorm(output)                # (B, d, 2l)
```

(c) **FLOPs and parameter analysis:**
- FLOPs per example (top-1): $F_\text{gate} + k \cdot F_\text{module}$ (one expert's ensemble + gating). Same compute as single DHEN + gating overhead.
- Total parameters: $P_\text{gate} + E \cdot k \cdot P_\text{module}$ (each expert has its own $k$ modules). $E$ times more than plain DHEN.
- vs. plain DHEN ($k$ modules, no MoE): MoE-DHEN has $E \times$ more parameters for the same per-example FLOPs.
- vs. plain MoE ($E$ experts, 1 module each): MoE-DHEN has $k$ times more per-example compute (all $k$ modules run within chosen expert) and same parameter count if each MoE expert $\equiv$ one module. MoE-DHEN dominates plain MoE when interaction diversity per example matters; MoE-DHEN dominates plain DHEN when sub-population specialization matters.
