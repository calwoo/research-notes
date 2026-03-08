# DHEN: Exercises

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

*The Factorization Machine second-order term naively requires $O(m^2 d)$ operations for $m$ features of dimension $d$. This problem establishes the $O(md)$ identity that makes FM-style interactions tractable, and traces its consequence for the AdvancedDLRM computation.*

> **Prerequisites:** cf. note [[note#3.1 AdvancedDLRM|§3.1 — AdvancedDLRM]]

(a) Let $v_i \in \mathbb{R}^d$ for $i = 1, \ldots, m$ and let $x_i \in \mathbb{R}$ be scalar feature values. Prove the identity

$$\sum_{i < j} \langle v_i, v_j \rangle x_i x_j = \frac{1}{2}\left[\left\|\sum_{i=1}^m x_i v_i\right\|^2 - \sum_{i=1}^m x_i^2 \|v_i\|^2\right]$$

and show this reduces the computation from $O(m^2 d)$ to $O(md)$.

(b) For binary features ($x_i \in \{0, 1\}$), let $\mathcal{A} = \{i : x_i = 1\}$ be the set of active features. Show that the identity from (a) reduces to

$$\sum_{i < j,\, i,j \in \mathcal{A}} \langle v_i, v_j \rangle = \frac{1}{2}\left[\left\|\sum_{i \in \mathcal{A}} v_i\right\|^2 - \sum_{i \in \mathcal{A}} \|v_i\|^2\right]$$

and interpret this as "sum pooling then squared norm minus self-norms," which is the computation used in DLRM's embedding interaction step.

(c) AdvancedDLRM retains all individual pairwise scores $s_{ij} = \langle v_i, v_j \rangle$ before passing them to an MLP. Show that the FM trick output is a fixed linear combination of $\{s_{ij}\}$ (specifically with coefficient 1 for all pairs), whereas retaining individual scores allows the downstream MLP to learn an arbitrary function $f(s_{11}, \ldots, s_{mm})$. Conclude that the FM trick is a strict computational relaxation of the full pairwise approach, trading expressivity for efficiency.

---

### Problem 2: DCN Cross Network Polynomial Degree via Induction

*The DCN cross network produces feature interactions of bounded polynomial degree. This problem proves the tight degree bound by induction and identifies the exact leading-degree term at each layer, making precise why stacking more cross layers yields higher-order interactions.*

> **Prerequisites:** cf. note [[note#3.3 Deep Cross Net|§3.3 — Deep Cross Net]]

(a) The DCN cross recurrence is $x_{\ell+1} = x_0 (x_\ell^\top w_\ell) + b_\ell + x_\ell$, where $x_\ell, x_0, b_\ell \in \mathbb{R}^d$ and $w_\ell \in \mathbb{R}^d$. Prove by induction on $\ell$ that $x_\ell$ is a polynomial in the components of $x_0$ of degree exactly $\ell + 1$. Explicitly identify the degree-$(\ell+1)$ monomial term at each step of the induction.

(b) Show that no degree-$(\ell+2)$ term can appear in $x_\ell$: the inductive step multiplies degree-$\ell$ terms by $x_0$ (degree 1), giving degree $\ell+1$ at most. Conclude that a DCN with $L$ cross layers captures feature interactions of degree at most $L+1$, establishing the tight upper bound.

(c) Extend the degree argument to depth $L$ and $k$ modules per DHEN layer. For a $N$-layer DHEN where each layer contains one DCN module with $L$ cross layers, show that the maximum interaction degree reachable by the DCN path through all $N$ layers grows as $(L+1)^N$. Contrast this with a plain depth-$NL$ DCN (single module), which achieves degree $NL + 1$. Conclude that hierarchical stacking yields super-linear growth in achievable degree versus simply deepening a single module.

---

### Problem 3: The Full Jacobian of a DCN Cross Layer

*The DCN cross operation $x_{\ell+1} = x_0(x_\ell^\top w_\ell) + b_\ell + x_\ell$ has a Jacobian with a specific rank-1 plus identity structure. Computing this Jacobian exactly clarifies why DCN gradients are well-conditioned by construction.*

> **Prerequisites:** cf. note [[note#3.3 Deep Cross Net|§3.3 — Deep Cross Net]]; requires Problem 2

(a) Treating $x_0$ as a fixed input and $x_\ell$ as the variable, compute the Jacobian $J_\ell = \frac{\partial x_{\ell+1}}{\partial x_\ell} \in \mathbb{R}^{d \times d}$. Show that it factors as

$$J_\ell = I + x_0 w_\ell^\top$$

a rank-1 update of the identity matrix.

(b) Using the matrix determinant lemma, show that $\det(J_\ell) = 1 + w_\ell^\top x_0$. Derive the condition on $w_\ell$ and $x_0$ under which $J_\ell$ is invertible, and write the inverse explicitly using the Sherman-Morrison formula.

(c) The singular values of $J_\ell = I + x_0 w_\ell^\top$ determine gradient magnification or attenuation. Show that $J_\ell$ has $d-1$ singular values equal to 1 and one singular value equal to $\sqrt{1 + \|x_0\|^2 \|w_\ell\|^2 + (w_\ell^\top x_0)^2}$ (approximately). Conclude that the DCN cross layer cannot cause exponential gradient vanishing even without explicit residual shortcuts — the Jacobian is inherently near-identity.

---

### Problem 4: Self-Attention as Data-Dependent Feature Recombination

*Self-attention produces outputs that are non-linear data-dependent linear combinations of value vectors. This problem makes precise what "feature-level interaction" means for the attention module and contrasts it with bit-level interactions from DCN.*

> **Prerequisites:** cf. note [[note#3.2 Self-Attention|§3.2 — Self-Attention]]

(a) For a single attention head with input $X \in \mathbb{R}^{m \times d}$ (rows are feature embeddings), define $Q = XW^Q$, $K = XW^K$, $V = XW^V$. Write the output at feature position $i$ as

$$z_i = \sum_{j=1}^m A_{ij} (x_j W^V), \qquad A_{ij} = \operatorname{softmax}_j\!\left(\frac{q_i k_j^\top}{\sqrt{d_k}}\right)$$

Show that $A_{ij}$ depends on $x_i$ and $x_j$ only through the scalar $q_i k_j^\top = x_i W^Q (x_j W^K)^\top$, making each attention weight a function of a single inner product between projected features.

(b) Show that $z_i$ is a weighted average of value vectors $\{x_j W^V\}$, where the weights $A_{ij}$ sum to 1 and depend on the entire input $X$ (not just $x_i$). Conclude that the output at position $i$ is not a polynomial of bounded degree in the entries of $X$, because the softmax normalization makes $A_{ij}$ a rational function of all inner products $\{q_i k_{j'}^\top\}_{j'=1}^m$.

(c) The DCN cross module in DHEN computes $u = W(X X^\top) + b$. Show that this operates on the outer product $X X^\top \in \mathbb{R}^{m \times m}$, where entry $(i,j)$ is the inner product $\langle x_i, x_j \rangle$ across all embedding dimensions simultaneously. Contrast: in DCN, the interaction weight between features $i$ and $j$ is fixed (given by $\langle x_i, x_j \rangle$ summed over all dimensions); in attention, the interaction weight $A_{ij}$ is computed in a subspace determined by $W^Q, W^K$. Formalize this distinction.

---

### Problem 5: Bit-Level vs. Feature-Level Interaction

*DHEN combines modules that operate at different granularities: DCN mixes individual scalar dimensions ("bit-level"), while self-attention treats each embedding vector as an atomic unit ("feature-level"). This problem formalizes the distinction and shows neither class subsumes the other.*

> **Prerequisites:** cf. note [[note#3.3 Deep Cross Net|§3.3 — Deep Cross Net]]; cf. note [[note#3.2 Self-Attention|§3.2 — Self-Attention]]

(a) DCN's cross operation applied to $x \in \mathbb{R}^{md}$ (a flattened embedding matrix) produces a scalar interaction $x^\top w$ and multiplies it back into $x_0$. Show that this is equivalent to a bilinear form $x_0^\top M x$ with $M = w \mathbf{1}^\top$ (rank-1). In particular, every scalar coordinate of $x_0$ gets scaled by the same global scalar $x^\top w$, mixing all embedding dimensions uniformly.

(b) Self-attention produces, at position $i$, an output $z_i = \sum_j A_{ij} x_j W^V$. Show that $z_i$ mixes only the embedding dimension via the projection $W^V$, and that the combination coefficients $A_{ij}$ act at the level of whole feature vectors (not individual scalar dimensions). Specifically, $[z_i]_r = \sum_j A_{ij} \sum_s x_j[s] W^V[s,r]$: the mixing of embedding dimension $s$ into output dimension $r$ is the same for every feature $j$ (governed by $W^V$).

(c) Give a concrete example of an interaction that DCN can represent but attention cannot (in a single layer), and one that attention can represent but DCN cannot. Conclude that the two are complementary, providing formal justification for including both in an ensemble.

---

### Problem 6: Gradient Flow Through Residual DHEN Layers

*The DHEN shortcut connection $X_{n+1} = F_n(X_n) + X_n$ has a specific effect on gradient flow. This problem derives the exact gradient formula and identifies the term that prevents vanishing gradients regardless of the ensemble Jacobian.*

> **Prerequisites:** cf. note [[note#4.2 Residual Shortcut and Dimension Matching|§4.2 — Residual Shortcut and Dimension Matching]]

(a) For an $N$-layer DHEN with identity shortcuts and no normalization, $X_{n+1} = F_n(X_n) + X_n$. By the chain rule, write

$$\frac{\partial \mathcal{L}}{\partial X_0} = \prod_{n=0}^{N-1} \frac{\partial X_{n+1}}{\partial X_n}$$

and show that $\frac{\partial X_{n+1}}{\partial X_n} = I + J_n$ where $J_n = \frac{\partial F_n(X_n)}{\partial X_n}$ is the Jacobian of the ensemble at layer $n$.

(b) Expand the product $\prod_{n=0}^{N-1}(I + J_n)$ as a sum over subsets $S \subseteq \{0, \ldots, N-1\}$:

$$\prod_{n=0}^{N-1}(I + J_n) = \sum_{S \subseteq [N]} \prod_{n \in S} J_n$$

(state precisely in what sense this expansion holds for non-commuting matrix factors). Identify the $S = \emptyset$ term and explain why it guarantees a non-vanishing gradient path.

(c) In the presence of layer normalization, the Jacobian of Norm is $\frac{\partial \operatorname{Norm}(z)}{\partial z} = \frac{1}{\sigma}(I - \frac{1}{d}\mathbf{1}\mathbf{1}^\top - \hat{z}\hat{z}^\top)$ where $\hat{z}$ is the normalized vector. Show that this projection matrix has rank $d-2$ and all nonzero eigenvalues equal to $1/\sigma$. Argue that while LayerNorm introduces some gradient attenuation, the identity shortcut term still prevents complete gradient vanishing.

---

### Problem 7: Vanishing Gradient Bound for Residual vs. Plain Stacks

*This problem proves quantitatively that plain (non-residual) deep stacks suffer exponential gradient decay under mild spectral assumptions, while the residual shortcut bounds the gradient away from zero.*

> **Prerequisites:** cf. note [[note#4.2 Residual Shortcut and Dimension Matching|§4.2 — Residual Shortcut and Dimension Matching]]; requires Problem 6

(a) For a plain non-residual $N$-layer stack $X_{n+1} = F_n(X_n)$, the gradient is $\frac{\partial \mathcal{L}}{\partial X_0} = \prod_{n=0}^{N-1} J_n$. Suppose all singular values of each $J_n$ are bounded above by $\sigma < 1$ (contractive maps). Show that $\left\|\frac{\partial \mathcal{L}}{\partial X_0}\right\|_F \leq \sigma^N \left\|\frac{\partial \mathcal{L}}{\partial X_N}\right\|_F$, so the gradient vanishes geometrically as $N \to \infty$.

(b) For the residual stack with the same contractive assumption $\|J_n\|_\text{op} \leq \sigma < 1$, show that the $S = \emptyset$ term alone gives $\left\|\frac{\partial \mathcal{L}}{\partial X_0}\right\|_F \geq \left\|\frac{\partial \mathcal{L}}{\partial X_N}\right\|_F$, establishing a lower bound of 1 on the gradient ratio. This is the vanishing gradient theorem for deep residual networks.

(c) Empirically, DHEN achieves positive NE gains at $N = 8$ layers (Table 2 in the paper). Under the contractive assumption, a plain 8-layer stack would attenuate gradients by factor $\sigma^8$. For $\sigma = 0.9$, compute $\sigma^8$ and compare it to the residual lower bound of 1. Argue that without residual shortcuts, training an 8-layer DHEN would require significantly larger learning rates or careful initialization.

---

### Problem 8: Normalized Entropy as a Calibration Measure

*Normalized entropy (NE) normalizes binary cross-entropy by the entropy of the empirical click rate, making it comparable across datasets with different base click rates. This problem establishes the calibration interpretation: NE = 1 means the model adds no information beyond the marginal rate.*

> **Prerequisites:** cf. note [[note#7.1 Setup|§7.1 — Setup]]

(a) Show that a constant predictor $\hat{p}_i = p$ for all $i$ achieves exactly $\text{NE} = 1$, where $p$ is the empirical click rate, regardless of the dataset. Compute the numerator and denominator separately and show they are equal.

(b) The cross-entropy decomposes as $H(\hat{p}, y) = H(p) + D_\text{KL}(p_\text{data} \| p_\theta) + \text{(calibration gap)}$ where $H(p)$ is the entropy of the marginal distribution and $D_\text{KL}$ captures the model's discriminative power. Show that $\text{NE} = 1 + \frac{D_\text{KL}(p_\text{data} \| p_\theta)}{H(p)} + \frac{\text{(calibration gap)}}{H(p)}$, and conclude that $\text{NE} < 1$ iff the model's conditional predictions $\hat{p}_i$ carry information beyond the marginal rate $p$.

(c) Derive the exact condition on the predicted probabilities $\{\hat{p}_i\}$ under which $\text{NE} = 1$ exactly. Show that this condition is equivalent to: the model's predictions carry zero conditional mutual information $I(y ; \hat{p} \mid \text{context}) = 0$, i.e., the model is no better than chance given any feature context.

---

### Problem 9: NE and Mutual Information

*This problem formalizes the connection between NE improvement and mutual information, giving a precise information-theoretic interpretation to the 0.27% NE gain reported for DHEN over AdvancedDLRM.*

> **Prerequisites:** cf. note [[note#7.3 DHEN vs. AdvancedDLRM|§7.3 — DHEN vs. AdvancedDLRM]]; requires Problem 8

(a) Show that the NE of a model $p_\theta$ equals $1 - I(y; \hat{p}_\theta) / H(p)$ in expectation, where $I(y; \hat{p}_\theta)$ is the mutual information between the label $y$ and the model's prediction $\hat{p}_\theta$. Use the decomposition from Problem 8(b) with zero calibration gap for a perfectly calibrated model.

(b) For two models with NE values $\text{NE}_1$ and $\text{NE}_2 < \text{NE}_1$, show that

$$\text{NE}_1 - \text{NE}_2 = \frac{I(y; \hat{p}_{\theta_2}) - I(y; \hat{p}_{\theta_1})}{H(p)}$$

i.e., the NE improvement is proportional to the gain in mutual information, normalized by the marginal entropy.

(c) The paper reports a 0.27% NE improvement for DHEN over AdvancedDLRM. If the empirical click rate is $p = 0.05$ (a typical display ad CTR), compute $H(p)$ in nats. Then compute the additional mutual information (in nats) that DHEN's predictions capture about click outcomes compared to AdvancedDLRM. Interpret the result.

---

### Problem 10: The k^N Interaction Composition Count

*Stacking $N$ DHEN layers with $k$ modules per layer produces $k^N$ distinct interaction compositions. This problem proves the count rigorously, distinguishes sum from concatenation aggregation, and contrasts with homogeneous stacks.*

> **Prerequisites:** cf. note [[note#4.3 Stacking and Mixture of High-Order Interactions|§4.3 — Stacking and Mixture of High-Order Interactions]]

(a) For a 2-layer, 2-module DHEN with modules $I_1$ and $I_2$ and sum aggregation at each layer, write $X_1 = I_1(X_0) + I_2(X_0) + X_0$ (including shortcut). Then write $X_2$ explicitly as a function of $X_0$ by substituting the expression for $X_1$. Enumerate all the distinct functional compositions of $I_1$ and $I_2$ that appear in $X_2$, and count them.

(b) With concatenation aggregation, $X_1 = [I_1(X_0) \| I_2(X_0)]$ (ignoring shortcut for clarity). Each second-layer module takes $X_1$ as input; e.g., $I_1(X_1) = I_1([I_1(X_0) \| I_2(X_0)])$. Explain why this produces a strictly richer set of interactions than sum aggregation: the concatenation preserves the individual module outputs separately, allowing each second-layer module to distinguish contributions from different first-layer modules.

(c) Prove by induction that for an $N$-layer DHEN with $k$ modules per layer (sum aggregation, no shortcuts for simplicity), the number of distinct functional compositions reachable at the output grows as at least $k^N$. Contrast: a depth-$N$ homogeneous stack with a single module type produces exactly 1 distinct functional form at each layer (composition of the same function $N$ times). Compute $k^N$ for $k = 5$, $N = 4$ and state what this means for expressivity.

---

### Problem 11: Ensemble Variance Reduction

*Combining heterogeneous modules reduces prediction variance relative to any single module, under the standard bias-variance decomposition. This problem formalizes why ensemble diversity — not just model capacity — drives DHEN's gains.*

> **Prerequisites:** cf. note [[note#4.1 Ensemble Aggregation|§4.1 — Ensemble Aggregation]]; cf. note [[note#1.2 The Non-Overlapping Information Hypothesis|§1.2 — The Non-Overlapping Information Hypothesis]]

(a) Let $\hat{y}_1, \ldots, \hat{y}_k$ be the predictions from $k$ modules with individual expected squared errors $\mathbb{E}[(y - \hat{y}_i)^2] = B_i^2 + V_i$, where $B_i$ is bias and $V_i$ is variance. For the average ensemble $\hat{y}_\text{ens} = \frac{1}{k}\sum_i \hat{y}_i$, show that

$$\mathbb{E}\left[(y - \hat{y}_\text{ens})^2\right] = \left(\frac{1}{k}\sum_i B_i\right)^2 + \frac{1}{k^2}\sum_{i,j} \operatorname{Cov}(\hat{y}_i, \hat{y}_j)$$

(b) Show that if the modules are uncorrelated ($\operatorname{Cov}(\hat{y}_i, \hat{y}_j) = 0$ for $i \neq j$) and have equal variance $V$, the ensemble variance is $V/k$, a $k$-fold reduction. Show further that the variance reduction is maximized when module predictions are negatively correlated, and is eliminated when modules are perfectly correlated.

(c) The non-overlapping information hypothesis asserts that different DHEN modules capture complementary (low-correlation) information. Formalize: if $\operatorname{Cov}(\hat{y}_i, \hat{y}_j) = \rho V$ for $i \neq j$ with $0 \leq \rho < 1$, show that the ensemble variance is $V(1 + (k-1)\rho)/k$. For $\rho = 0.5$, $k = 2$, compute the variance ratio and interpret it in the context of the ablation (self-attention + linear vs. either alone).

---

### Problem 12: Ensemble Stabilization of Slow-Converging Modules

*Self-attention alone converges slowly in the DHEN ablation, but attention + linear converges faster and achieves better final NE. This problem derives the gradient-sharing mechanism that explains this stabilization effect.*

> **Prerequisites:** cf. note [[note#7.2 Interaction Module Ablation|§7.2 — Interaction Module Ablation]]; requires Problem 6

(a) With sum aggregation, $Y = I_\text{attn}(X) + I_\text{lin}(X) + \operatorname{ShortCut}(X)$. The gradient with respect to $I_\text{attn}$'s parameters $\theta_\text{attn}$ is

$$\frac{\partial \mathcal{L}}{\partial \theta_\text{attn}} = \frac{\partial \mathcal{L}}{\partial Y} \cdot \frac{\partial I_\text{attn}(X)}{\partial \theta_\text{attn}}$$

Show that $\frac{\partial \mathcal{L}}{\partial Y}$ is shared across all modules. Argue that adding $I_\text{lin}(X)$ reduces the variance of $\frac{\partial \mathcal{L}}{\partial Y}$ compared to the case where attention is the sole module (the loss surface the ensemble navigates is smoother).

(b) At initialization, self-attention with small random weights produces outputs near zero (approximately uniform attention weights $A_{ij} \approx 1/m$, giving $z_i \approx \frac{1}{m}\sum_j x_j W^V \approx 0$ for small $W^V$). The linear module with near-identity initialization produces $I_\text{lin}(X) \approx X$, a reasonable initial predictor. Show that the ensemble's prediction at initialization is therefore dominated by the linear module, providing a stable gradient signal even before attention has learned.

(c) Draw the explicit parallel to ResNet training dynamics: explain how the DHEN ensemble at initialization behaves like an "effective shallow model" and gradually shifts to exploit the attention module as training progresses. State this as a formal analogy (not just a verbal description): identify the correspondence between DHEN components and ResNet components.

---

### Problem 13: Why Linear Can Outperform DCN

*The ablation shows that a simple linear projection module outperforms the DCN cross-network module in isolation. This problem derives two formal hypotheses — feature redundancy and optimization landscape — explaining this counterintuitive result.*

> **Prerequisites:** cf. note [[note#7.2 Interaction Module Ablation|§7.2 — Interaction Module Ablation]]; cf. note [[note#3.3 Deep Cross Net|§3.3 — Deep Cross Net]]; cf. note [[note#3.4 Linear|§3.4 — Linear]]

(a) **Feature redundancy hypothesis.** Suppose the input embeddings $X_0$ already contain explicit feature crosses: the embedding table contains entries $v_{ij}$ for feature pairs $(i,j)$ alongside individual feature embeddings $v_i$. Show that DCN's degree-2 term $x_0(x_0^\top w)$ in the cross network is then in the linear span of the input embeddings, meaning DCN's crossing adds no information not already present in $X_0$. By contrast, the linear module $u = WX_0$ simply re-projects all available information, including the pre-computed crosses.

(b) **Optimization landscape hypothesis.** The cross network recurrence $x_{\ell+1} = x_0(x_\ell^\top w_\ell) + b_\ell + x_\ell$ introduces bilinear coupling between $x_0$ and the parameter $w_\ell$ via the product $x_\ell^\top w_\ell$. Show that the Hessian of the loss with respect to $w_\ell$ contains second-order terms $\propto x_0 x_0^\top$, creating curvature that grows with embedding magnitude. For sparse embedding inputs where $\|x_0\|$ varies wildly across features, argue that this creates an ill-conditioned optimization problem relative to the linear module's constant-curvature landscape.

(c) Given both hypotheses, identify the dataset conditions under which DCN would outperform the linear module. Specifically, state a concrete property of the feature space (e.g., features are raw one-hot categoricals with no pre-computed crosses) and a training regime property (e.g., small learning rate, strong regularization) where DCN's explicit crossing should dominate.

---

### Problem 14: DHEN vs. MoE: Interaction Diversity vs. Capacity

*At matched FLOPs, DHEN consistently outperforms MoE-based scaling. This problem formalizes the distinction between interaction diversity (DHEN) and model capacity (MoE), and identifies conditions under which MoE could be preferable.*

> **Prerequisites:** cf. note [[note#7.4 Scaling Efficiency vs. Mixture of Experts|§7.4 — Scaling Efficiency vs. Mixture of Experts]]

(a) MoE with $E$ experts and top-1 routing processes each example through exactly one expert. Show that the expected FLOPs per example equal the FLOPs of a single expert (not $E$ times). DHEN with $k$ modules processes each example through all $k$ modules, so its FLOPs per example scale exactly as $k$. For matched FLOPs budget $F$: MoE can have $E = F / F_\text{expert}$ experts (increasing capacity), while DHEN can have $k = F / F_\text{module}$ modules (increasing diversity). Formalize the tradeoff.

(b) Construct a formal scenario in which MoE outperforms DHEN: suppose the data distribution is a mixture of $E$ sub-populations with qualitatively different optimal interaction structures $f_1^*, \ldots, f_E^*$, such that $f_e^*$ is not well-approximated by any convex combination of $\{f_1^*, \ldots, f_E^*\}$. Show that DHEN's ensemble (a single model applied to all examples) cannot match MoE's per-example specialization in this setting.

(c) Propose a hybrid "MoE-DHEN" architecture: an MoE layer where each expert is itself a small DHEN ensemble with $k_\text{expert}$ modules. Write the forward pass formally, specifying which parameters are shared across experts and which are expert-specific. For $E = 2$ experts and $k_\text{expert} = 2$ modules (DCN + Linear), count the total distinct interaction paths per example.

---

### Problem 15: HSDP Communication Latency Analysis

*HSDP exploits a 24x bandwidth asymmetry between NVLink and RoCEv2 to reduce critical-path communication latency. This problem derives the exact latency savings and the memory tradeoffs.*

> **Prerequisites:** cf. note [[note#6.3 Hybrid Sharded Data Parallel|§6.3 — Hybrid Sharded Data Parallel]]; cf. note [[note#6.2 Fully Sharded Data Parallel|§6.2 — Fully Sharded Data Parallel]]

(a) A cluster has $H$ hosts with $G$ GPUs each; NVLink bandwidth is $B_\text{NV}$ per GPU and cross-host bandwidth is $B_\text{net}$ per GPU with $B_\text{NV}/B_\text{net} \approx 24$. The dense model has $P$ parameters (BF16 = 2 bytes each). In FSDP across all $GH$ GPUs, each GPU holds a shard of size $P/(GH)$. The allgather on the forward-pass critical path communicates approximately $P$ bytes per GPU over the bottleneck cross-host links. Estimate the FSDP allgather critical-path latency as $L_\text{FSDP} \approx P / B_\text{net}$.

(b) In HSDP, the forward-pass allgather is confined to the $G$ GPUs within a single host (over NVLink). Each GPU's shard is $P/G$, and the total bytes communicated per GPU in the intra-host allgather is approximately $P(G-1)/G \approx P$. Show that the HSDP allgather critical-path latency is $L_\text{HSDP} \approx P / B_\text{NV}$, and conclude that HSDP reduces critical-path latency by a factor of $B_\text{NV}/B_\text{net} \approx 24$ compared to FSDP.

(c) Compute the peak memory per GPU in bytes for: (i) plain data parallel (DP); (ii) FSDP across $GH$ GPUs; (iii) HSDP. Express answers in terms of $P$, $G$, $H$, and the optimizer state multiplier $s$ (e.g., $s = 3$ for Adam: 1 for parameters + 2 for first and second moments). Note which strategy HSDP uses for gradient buffers — specifically that the asynchronous cross-host allreduce requires buffering a full $P/G$-parameter gradient shard during the backward pass.

---

### Problem 16: Shortcut Aggregation and Dimension Matching

*The DHEN shortcut applies a learned projection $W_n$ only when the ensemble output dimension differs from the input dimension, analogous to the 1x1 convolution shortcut in ResNet. This problem characterizes when dimension mismatch occurs and what the projection learns.*

> **Prerequisites:** cf. note [[note#4.2 Residual Shortcut and Dimension Matching|§4.2 — Residual Shortcut and Dimension Matching]]; cf. note [[note#4.1 Ensemble Aggregation|§4.1 — Ensemble Aggregation]]

(a) Suppose the DHEN layer uses concatenation aggregation with $k$ modules, each producing output $u_i \in \mathbb{R}^{d \times l}$. The ensemble output is $[u_1 \| \cdots \| u_k] \in \mathbb{R}^{d \times (kl)}$. The input is $X_n \in \mathbb{R}^{d \times m}$. Show that a dimension mismatch occurs unless $kl = m$, and that the shortcut projection $W_n \in \mathbb{R}^{m \times kl}$ must be applied. Compute the number of parameters in $W_n$.

(b) For sum aggregation with $k$ modules each producing $u_i \in \mathbb{R}^{d \times l}$, show that the ensemble output has dimension $d \times l$. Identify the condition on $l$ and $m$ under which the identity shortcut can be used. When $l \neq m$, how many parameters does the projection require compared to the concatenation case?

(c) The shortcut projection $W_n$ plays a dual role: dimension matching and information compression/expansion. Argue that when concatenation grows the embedding count ($kl > m$), the shortcut projects from a lower-dimensional input space to a higher-dimensional ensemble output space — this is an information expansion, not a bottleneck. Contrast with the ResNet 1x1 projection shortcut, which always projects between equal-dimensional spaces (identity) or reduces dimensions (bottleneck). Identify which DHEN regime is more analogous to which ResNet case.

---

### Problem 17: Convolution as Local Co-occurrence Capture

*The convolution module applies 2D filters to the embedding matrix $X_n \in \mathbb{R}^{d \times m}$, treating the embedding dimension $d$ and feature count $m$ as spatial axes. This problem analyzes what interactions a convolutional filter captures and why locality in the embedding matrix is a meaningful inductive bias.*

> **Prerequisites:** cf. note [[note#3.5 Convolution|§3.5 — Convolution]]

(a) A Conv2d filter of kernel size $(k_d, k_m)$ applied to $X_n \in \mathbb{R}^{d \times m}$ produces an output at position $(r, s)$ as $\sum_{p=0}^{k_d-1}\sum_{q=0}^{k_m-1} w_{pq} X_n[r+p, s+q]$. Show that this computes a weighted sum of $k_d \times k_m$ scalar entries from a local patch of the embedding matrix, mixing embedding dimensions $r$ through $r+k_d-1$ with features $s$ through $s+k_m-1$.

(b) The DHEN embedding matrix $X_n$ has rows corresponding to embedding dimensions (not features) and columns corresponding to feature slots. The ordering of features in the matrix is determined by the feature processing layer (embedding table ordering). Argue that the convolutional module's locality assumption — that adjacent features in the matrix have correlated interaction patterns — is an inductive bias that may or may not match the true feature correlation structure. Formalize: what ordering of features would make this assumption maximally valid?

(c) Compare the parameter count of a Conv2d module (kernel size $(k_d, k_m)$, $C_\text{out}$ output channels) with the self-attention module (head dimension $d_k$, $h$ heads). Show that for large $m$ and $d$, convolution is significantly more parameter-efficient if $k_d \ll d$ and $k_m \ll m$, while attention's parameter count is independent of $m$ (the number of features). Conclude that convolution and attention represent different parameter-efficiency/expressivity tradeoffs.

---

## Algorithmic Applications

### Problem 18: DCN Cross Layer as a Bilinear Operation

*The DCN-V2 cross layer has a low-rank factorization that reduces its dominant $O(d^2)$ cost to $O(dr)$. This problem develops the pseudocode for both the full and low-rank variants and assembles the complete DCN module used in DHEN.*

> **Prerequisites:** cf. note [[note#3.3 Deep Cross Net|§3.3 — Deep Cross Net]]; requires Problem 2

(a) **Full cross layer pseudocode.** Write pseudocode for the DCN-V2 cross layer $x_{\ell+1} = x_0 \odot (W_\ell x_\ell + b_\ell) + x_\ell$ where $W_\ell \in \mathbb{R}^{d \times d}$, $x_0, x_\ell, b_\ell \in \mathbb{R}^d$. Annotate every operation with its FLOP count. Identify the dominant cost.

(b) **Low-rank factorization.** For $W_\ell = U_\ell V_\ell^\top$ with $U_\ell, V_\ell \in \mathbb{R}^{d \times r}$ ($r \ll d$), write pseudocode for the low-rank cross layer. Show that the dominant cost reduces from $O(d^2)$ to $O(dr)$. Annotate all shapes and FLOPs.

(c) **Full DCN module in DHEN.** Sketch the complete DCN module: $L$ stacked cross layers applied to the flattened input $x_0 = \operatorname{flatten}(X_n) \in \mathbb{R}^{md}$, followed by a projection $W_m \in \mathbb{R}^{md \times dl}$ back to embedding list format $\mathbb{R}^{d \times l}$. List all learned parameters with their shapes. Write the forward pass pseudocode with shape annotations at each step.

---

### Problem 19: DHEN Forward Pass with Tensor Shapes

*A complete DHEN forward pass with shape tracking reveals where tensor dimensions change and where projections are required. This problem builds the full forward pass pseudocode for a 2-layer DHEN with self-attention and linear modules.*

> **Prerequisites:** cf. note [[note#2.2 The DHEN Layer: Formal Definition|§2.2 — The DHEN Layer: Formal Definition]]; cf. note [[note#4.1 Ensemble Aggregation|§4.1 — Ensemble Aggregation]]

(a) **Data structures and inputs.** Define all parameters and inputs needed for a 2-layer DHEN with self-attention + linear modules, concatenation aggregation, and layer normalization. For batch size $B$, $m$ feature slots, embedding dimension $d$, attention head dimension $d_k$, and $h$ attention heads: list every parameter tensor with its shape.

(b) **Per-layer pseudocode.** Write the pseudocode for one DHEN layer, annotating every intermediate tensor with shape $(B, d, m)$ or equivalent. Include: (i) self-attention module forward; (ii) linear module forward; (iii) concatenation ensemble; (iv) shortcut (identity or projection); (v) LayerNorm.

(c) **Full 2-layer forward pass.** Chain two DHEN layers and add the prediction head (flatten + MLP + sigmoid). Annotate the complete data flow from input $X_0 \in \mathbb{R}^{B \times d \times m}$ to output $\hat{p} \in \mathbb{R}^B$. Identify all points where a projection is required due to dimension mismatch.

---

### Problem 20: Numerically Stable Distributed NE Computation

*Computing NE in distributed training requires an allreduce for the global click rate and numerically stable cross-entropy computation for the numerator. This problem develops the complete distributed NE pseudocode.*

> **Prerequisites:** cf. note [[note#7.1 Setup|§7.1 — Setup]]

(a) **Allreduce for global click rate.** In distributed training with $W$ workers, each holding a local shard of $n/W$ examples, describe the allreduce operations needed to compute the global empirical click rate $p = \frac{1}{n}\sum_i y_i$. Show that allreducing the pair $(\sum_\text{local} y_i, n/W)$ and computing $p = \frac{\text{global sum of } y_i}{n}$ gives the correct result.

(b) **Numerically stable cross-entropy.** The binary cross-entropy $-[y \log \hat{p} + (1-y)\log(1-\hat{p})]$ is unstable when $\hat{p}$ is near 0 or 1. Using logits $s_i$ where $\hat{p}_i = \sigma(s_i)$, show that $\log \hat{p}_i = -\operatorname{softplus}(-s_i)$ and $\log(1-\hat{p}_i) = -\operatorname{softplus}(s_i)$, giving the stable formula $\operatorname{CE}(y_i, s_i) = \operatorname{softplus}(s_i) - y_i \cdot s_i$.

(c) **Complete distributed NE pseudocode.** Write `distributed_NE(local_logits, local_labels, n_global)` that: (i) computes local cross-entropy in a numerically stable way using the formula from (b); (ii) allreduces to obtain global cross-entropy sum and global label sum; (iii) computes $p$ and the NE denominator $H(p)$; (iv) returns NE. Annotate each allreduce with what is communicated and why.

---

### Problem 21: HSDP Forward and Backward Pass Pseudocode

*HSDP restricts the critical-path allgather to NVLink-connected intra-host GPUs and moves cross-host gradient averaging off the critical path via asynchronous allreduce. This problem traces the exact communication schedule through the forward and backward passes.*

> **Prerequisites:** cf. note [[note#6.3 Hybrid Sharded Data Parallel|§6.3 — Hybrid Sharded Data Parallel]]; requires Problem 15

(a) **HSDP data structures.** Define the parameter sharding scheme: each GPU holds a shard of size $P/G$ parameters (for $G$ GPUs per host). Specify which parameters are sharded (dense DHEN weights) and which are replicated (none, in the full HSDP model). List the communication buffers each GPU maintains.

(b) **Forward pass pseudocode.** Write the HSDP forward pass for a single DHEN layer. Include: (i) intra-host allgather of weight shards over NVLink to reconstruct full layer weights; (ii) forward computation using full weights; (iii) intra-host reducescatter of activations (if used). Annotate each communication step with: operation type, participants (which GPUs), bandwidth used (NVLink or RoCEv2), and whether it is on the critical path.

(c) **Backward pass and async allreduce.** Write the HSDP backward pass. Include: (i) gradient computation; (ii) intra-host reducescatter of gradients; (iii) registration of an async allreduce hook to average the local gradient shard across hosts over RoCEv2; (iv) overlap of the allreduce with subsequent backward layers. Explain why this async overlap is valid and what synchronization is required before the optimizer step.

---

### Problem 22: MoE-DHEN Hybrid Architecture

*A hybrid architecture combining MoE routing with DHEN ensembles within each expert could capture both per-example routing specialization and interaction diversity. This problem develops the complete forward pass and analyzes the computational cost.*

> **Prerequisites:** cf. note [[note#7.4 Scaling Efficiency vs. Mixture of Experts|§7.4 — Scaling Efficiency vs. Mixture of Experts]]; requires Problem 14

(a) **Architecture specification.** Design a "MoE-DHEN" layer: an MoE layer with $E$ experts, where each expert is a DHEN ensemble with $k$ modules (DCN + Linear). Define the gating network $g: \mathbb{R}^{d \times m} \to \Delta^E$ (maps input to a probability simplex over experts), the expert networks $\{\text{DHEN}_e\}_{e=1}^E$, and the output aggregation. Specify which parameters are shared across experts and which are expert-specific.

(b) **Forward pass pseudocode.** Write pseudocode for the MoE-DHEN layer forward pass with top-1 routing. Include: (i) gating score computation; (ii) expert selection and dispatch; (iii) DHEN ensemble computation within the selected expert; (iv) output assembly. Annotate shapes at each step for batch size $B$, $m$ features, embedding dimension $d$.

(c) **FLOPs and parameter count analysis.** For $E$ experts with $k$ modules per expert, each module with $F_\text{module}$ FLOPs per example: compute total FLOPs per example (top-1 routing) and total parameter count. Compare to: (i) a plain $k$-module DHEN (no MoE); (ii) a plain $E$-expert MoE with single-module experts. Identify the parameter-efficiency and compute-efficiency regimes where MoE-DHEN dominates each baseline.
