# DHEN: Exercises

## Table of Contents

- [[#derivation-problems|Derivation Problems]]
  - [[#problem-1-fm-pairwise-dot-products-and-the-dlrm-computation|Problem 1: FM Pairwise Dot Products and the DLRM Computation]]
  - [[#problem-2-dcn-cross-network-polynomial-degree|Problem 2: DCN Cross Network Polynomial Degree]]
  - [[#problem-3-self-attention-as-data-dependent-feature-recombination|Problem 3: Self-Attention as Data-Dependent Feature Recombination]]
  - [[#problem-4-gradient-flow-through-residual-dhen-layers|Problem 4: Gradient Flow Through Residual DHEN Layers]]
  - [[#problem-5-normalized-entropy-as-a-calibration-measure|Problem 5: Normalized Entropy as a Calibration Measure]]
- [[#conceptual-questions|Conceptual Questions]]
  - [[#problem-6-the-kn-interaction-path-argument|Problem 6: The k^N Interaction Path Argument]]
  - [[#problem-7-why-linear-outperforms-dcn-in-the-ablation|Problem 7: Why Linear Outperforms DCN in the Ablation]]
  - [[#problem-8-ensemble-stabilization-of-slow-converging-modules|Problem 8: Ensemble Stabilization of Slow-Converging Modules]]
  - [[#problem-9-dhen-vs-moe-interaction-diversity-vs-capacity|Problem 9: DHEN vs. MoE — Interaction Diversity vs. Capacity]]
  - [[#problem-10-hsdp-communication-analysis|Problem 10: HSDP Communication Analysis]]
- [[#implementation-sketches|Implementation Sketches]]
  - [[#problem-11-dcn-cross-layer-as-a-bilinear-operation|Problem 11: DCN Cross Layer as a Bilinear Operation]]
  - [[#problem-12-dhen-forward-pass-with-tensor-shapes|Problem 12: DHEN Forward Pass with Tensor Shapes]]
  - [[#problem-13-numerically-stable-distributed-ne-computation|Problem 13: Numerically Stable Distributed NE Computation]]

---

## Derivation Problems

### Problem 1: FM Pairwise Dot Products and the DLRM Computation

The Factorization Machine second-order term is $\sum_{i < j} \langle v_i, v_j \rangle x_i x_j$ for feature values $x_i$ and embedding vectors $v_i \in \mathbb{R}^d$. AdvancedDLRM computes all pairwise dot products $\langle x_i, x_j \rangle$ explicitly.

**(a)** Show that

$$\sum_{i < j} \langle v_i, v_j \rangle = \frac{1}{2}\left[\left\|\sum_i v_i\right\|^2 - \sum_i \|v_i\|^2\right]$$

giving an $O(md)$ algorithm, where $m$ is the number of features.

**(b)** For the binary-feature FM case ($x_i \in \{0,1\}$, active features only), show that the identity from (a) reduces to summing over active feature embeddings, recovering the "sum pooling then inner product" used in DLRM.

**(c)** The explicit $O(m^2 d)$ pairwise computation in AdvancedDLRM retains the individual interaction scores $s_{ij} = \langle v_i, v_j \rangle$ before aggregation, whereas the FM trick collapses them. Argue that retaining individual $s_{ij}$ allows the subsequent MLP to learn which pairs are important — something the FM trick destroys. Formalize by showing that the FM trick output is a fixed linear combination of $\{s_{ij}\}$, while retaining individual scores allows an arbitrary function.

---

### Problem 2: DCN Cross Network Polynomial Degree

The DCN cross network applies the recurrence $x_{l+1} = x_0 (x_l^\top w_l) + b_l + x_l$ where $x_l, x_0, b_l \in \mathbb{R}^d$ and $w_l \in \mathbb{R}^d$.

**(a)** Show by induction that $x_L$ is a polynomial in the components of $x_0$ of degree at most $L+1$. Identify the degree-$(L+1)$ term explicitly.

**(b)** Show that no degree-$(L+2)$ term can appear, establishing that DCN with $L$ cross layers captures at most degree-$(L+1)$ feature interactions.

**(c)** The cross network is therefore "depth-bounded" in expressive power. For a fixed interaction budget (total FLOPs), compare the expressivity of: (i) a single DCN with $L = 5$ cross layers, versus (ii) a 2-layer DHEN with DCN and linear modules, where each DCN has $L = 2$ cross layers. What does the DHEN composition add that the deep single DCN cannot provide?

---

### Problem 3: Self-Attention as Data-Dependent Feature Recombination

For a single attention head with input $X \in \mathbb{R}^{m \times d}$ (rows are feature embeddings), define $Q = XW^Q$, $K = XW^K$, $V = XW^V$ with $W^Q, W^K \in \mathbb{R}^{d \times d_k}$ and $W^V \in \mathbb{R}^{d \times d_v}$.

**(a)** Write the output at feature position $i$ as

$$z_i = \sum_{j=1}^m A_{ij} (x_j W^V)$$

where $A_{ij} = \text{softmax}_j(q_i k_j^\top / \sqrt{d_k})$. Show that $A_{ij}$ depends on $x_i$ and $x_j$ through a dot product of their projections, making the output $z_i$ a non-linear function of all input features simultaneously.

**(b)** Show that DCN's cross operation $x_0(x_l^\top w_l)$ is a bit-level interaction: it scales the entire vector $x_0$ by the scalar $x_l^\top w_l$, effectively mixing all scalar components of $x_l$ into a single weight. By contrast, show that attention's $A_{ij}$ is a feature-level interaction: it produces a scalar weight for each feature pair $(i,j)$ without mixing the embedding dimensions of $i$ with those of $j$ (except through the global projections $W^Q, W^K$).

**(c)** Show that the attention output $z_i$ is NOT a polynomial of bounded degree in the components $\{x_j[r]\}$ (due to the softmax normalization). This means attention captures interactions that no finite-degree polynomial (such as DCN's output) can express — establishing that the two modules are not comparable in expressive power.

---

### Problem 4: Gradient Flow Through Residual DHEN Layers

Consider a simplified $N$-layer DHEN with identity shortcuts and no normalization: $X_{n+1} = F_n(X_n) + X_n$ where $F_n$ is the ensemble function at layer $n$.

**(a)** By the chain rule, write $\frac{\partial \mathcal{L}}{\partial X_0} = \prod_{n=0}^{N-1} \frac{\partial X_{n+1}}{\partial X_n}$. Show that $\frac{\partial X_{n+1}}{\partial X_n} = I + J_n$ where $J_n = \frac{\partial F_n(X_n)}{\partial X_n}$ is the Jacobian of the ensemble.

**(b)** Expand the product $\prod_{n=0}^{N-1}(I + J_n) = \sum_{S \subseteq [N]} \prod_{n \in S} J_n$ (using the fact that $I$ and $J_n$ "commute" in the expansion — note carefully when this expansion is valid for matrix products). Identify the $S = \emptyset$ term and explain why it guarantees a non-vanishing gradient path regardless of the $J_n$ values.

**(c)** Compare to a plain (non-residual) $N$-layer stack $X_{n+1} = F_n(X_n)$, where $\frac{\partial \mathcal{L}}{\partial X_0} = \prod_{n=0}^{N-1} J_n$. Show that if all singular values of $J_n$ are less than 1 (contractive), the gradient vanishes geometrically as $N \to \infty$ in the plain case, while in the residual case the gradient is bounded below by 1 (the identity path). This is the vanishing gradient theorem for deep residual networks.

---

### Problem 5: Normalized Entropy as a Calibration Measure

The NE metric is

$$\text{NE} = \frac{-\frac{1}{n}\sum_i [y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)]}{H(p)}$$

where $p$ is the empirical click rate and $H(p) = -(p \log p + (1-p)\log(1-p))$.

**(a)** Show that a constant predictor $\hat{p}_i = p$ for all $i$ achieves exactly $\text{NE} = 1$, regardless of the dataset.

**(b)** Decompose the cross-entropy $H(\hat{p}, y) = H(p) + D_{\text{KL}}(p_\text{data} \| p_\theta) + \text{calibration error}$ using the standard cross-entropy decomposition into entropy, KL divergence, and calibration terms. Hence show that $\text{NE} < 1$ iff the model's conditional predictions $\hat{p}_i$ carry information beyond the marginal rate $p$.

**(c)** Show that the NE improvement $\Delta\text{NE} = \text{NE}_\text{baseline} - \text{NE}_\text{model}$ is proportional (up to a dataset-dependent constant) to the mutual information $I(y; \hat{p})$ between label and prediction. Conclude: a 0.27% NE improvement means the DHEN model provides 0.27% more normalized mutual information about click outcomes than the AdvancedDLRM baseline.

---

## Conceptual Questions

### Problem 6: The k^N Interaction Path Argument

**(a)** For a 2-layer, 2-module DHEN with modules $I_1$ (DCN) and $I_2$ (Linear) and sum aggregation, write out the explicit computation of $X_2$ in terms of $X_0$. Show that $X_2$ depends on all four compositions $I_j(I_i(X_0) + I_{i'}(X_0))$ for $j, i, i' \in \{1,2\}$. Enumerate and count the distinct functional forms.

**(b)** With concatenation aggregation, $X_1 = [I_1(X_0) \| I_2(X_0)]$. Now $X_2 = [I_1(X_1) \| I_2(X_1)]$, where each $I_j$ takes as input the concatenation of both module outputs from layer 1. Explain why this is structurally richer than sum aggregation: each second-layer module jointly processes all first-layer outputs simultaneously, rather than their sum.

**(c)** The paper claims $k^N$ distinct interaction compositions for $N$ layers and $k$ modules. For $k = 5$ and $N = 4$ (the ablation setting), compute $k^N$. Argue that this exponential richness is qualitatively different from what a depth-$N$ homogeneous stack can produce (which generates at most $N+1$ interaction orders, not $k^N$ distinct compositions).

---

### Problem 7: Why Linear Outperforms DCN in the Ablation

The ablation (Table 1) shows the linear module alone outperforms DCN alone at all training checkpoints. Give two distinct hypotheses:

**(a) Feature redundancy hypothesis.** If the feature space already contains explicit feature crosses as input features (a common practice in production systems), then DCN's cross network is recomputing information already present in the input. The linear module, by simply projecting embeddings, avoids this redundancy. Formalize: if the input $X_0$ already contains $v_i \odot v_j$ terms for all pairs, show that DCN's degree-2 term in the cross network is linearly dependent on the input, adding zero new information.

**(b) Optimization landscape hypothesis.** The cross network introduces multiplicative coupling between $x_0$ and $x_l$ in the recurrence $x_{l+1} = x_0(x_l^\top w) + b + x_l$. Argue that this creates a rougher optimization landscape than a pure linear transformation $u = Wx$, particularly for large sparse embedding inputs where gradients can vary wildly across feature pairs.

**(c)** Given both hypotheses, predict under what conditions DCN would outperform linear: name a dataset property (e.g., features are raw uncrossed categoricals, small corpus, dense features only) where DCN's explicit crossing should dominate.

---

### Problem 8: Ensemble Stabilization of Slow-Converging Modules

The ablation shows that self-attention alone improves NE only after many examples (slow convergence), but attention + linear converges faster and achieves better final performance.

**(a)** With sum aggregation, the gradient with respect to the attention module parameters $\theta_\text{attn}$ is $\frac{\partial \mathcal{L}}{\partial \theta_\text{attn}} = \frac{\partial \mathcal{L}}{\partial Y} \cdot \frac{\partial Y}{\partial \theta_\text{attn}}$ where $Y = I_\text{attn}(X) + I_\text{lin}(X) + \text{ShortCut}(X)$. Show that the gradient $\frac{\partial \mathcal{L}}{\partial Y}$ is shared across both modules, and that the linear module provides an "alternative path" that makes the gradient signal to the attention module smoother (less variance) than if attention were the only module.

**(b)** Argue that at initialization, the linear module's output is a reasonable predictor (weights initialized near identity or small random), while attention's output is nearly uniform across features. The ensemble's early-training predictions therefore rely primarily on the linear module, providing stable gradient estimates that guide the attention module's warmup.

**(c)** This is structurally analogous to what phenomenon in ResNets? Draw the parallel explicitly.

---

### Problem 9: DHEN vs. MoE — Interaction Diversity vs. Capacity

At matched FLOPs, DHEN consistently outperforms MoE (Table 3). DHEN uses $k$ heterogeneous modules per layer; MoE routes each example to one of $E$ expert MLPs.

**(a)** Show that MoE with $E$ experts and top-1 routing processes each example through exactly one expert's computation path. The effective computation per example is the same as a single MLP (same FLOPs), but the model has $E$-times more parameters. DHEN with $k$ modules processes each example through all $k$ modules (full FLOPs from each). The key difference: DHEN increases interaction diversity per example; MoE increases model capacity across examples.

**(b)** Under what user-data distribution would MoE be expected to outperform DHEN? Argue: if different user segments require qualitatively different interaction structures (e.g., mobile users vs. desktop users have fundamentally different click patterns), MoE's routing can specialize experts per segment. DHEN's ensemble is always applied in full and cannot route.

**(c)** Propose a hybrid architecture: "MoE-DHEN" where each expert in an MoE layer is itself a small DHEN ensemble. Write the forward pass for a 2-expert MoE-DHEN with 2 modules per expert (DCN + Linear), and identify the parameters that would be shared vs. expert-specific.

---

### Problem 10: HSDP Communication Analysis

A cluster has $H$ hosts, each with $G$ GPUs connected via NVLink at bandwidth $B_\text{NV}$. Cross-host bandwidth is $B_\text{net}$ per GPU (much lower: $B_\text{NV}/B_\text{net} \approx 24$). The dense model has $P$ parameters (each a BF16 scalar = 2 bytes).

**(a)** In standard FSDP across all $GH$ GPUs: the allgather on the forward pass critical path must deliver each GPU's $P/(GH)$-parameter shard to all others. The total bytes communicated per GPU is $P \cdot \frac{GH-1}{GH} \approx P$ bytes. For the allgather to complete, the bottleneck is the cross-host links. Estimate the allgather latency as $P / B_\text{net}$ (ignoring startup overhead) for a single GPU.

**(b)** In HSDP: the intra-host allgather communicates $P/G \cdot (G-1) \approx P$ bytes per GPU over NVLink (latency $P/B_\text{NV}$). The cross-host allreduce communicates $2P(H-1)/H \approx 2P$ bytes per GPU over RoCEv2, but asynchronously (overlapped with backward pass of subsequent layers). Show that the critical-path latency of HSDP is $P/B_\text{NV}$ (the NVLink allgather), whereas FSDP's critical-path latency is $P/B_\text{net}$. Hence HSDP reduces critical-path communication latency by a factor of $B_\text{NV}/B_\text{net} \approx 24$.

**(c)** HSDP incurs additional memory overhead: each GPU holds the full model shard (as in FSDP) but must also buffer the allreduce gradient. Compute the peak memory per GPU in HSDP vs. FSDP vs. plain data parallel, in terms of $P$, $G$, $H$, and the optimizer state multiplier $s$ (e.g., $s = 3$ for Adam: parameters + first moment + second moment).

---

## Implementation Sketches

### Problem 11: DCN Cross Layer as a Bilinear Operation

**(a)** Write the DCN-V2 cross layer $x_{l+1} = x_0 \odot (W_l x_l + b_l) + x_l$ (element-wise product version, where $W_l \in \mathbb{R}^{d \times d}$) as pseudocode. Identify the dominant FLOP cost: the matrix-vector product $W_l x_l$ costs $O(d^2)$.

**(b)** For the low-rank version $W_l = U_l V_l^\top$ with $U_l, V_l \in \mathbb{R}^{d \times r}$ (rank $r \ll d$), show that the computation reduces from $O(d^2)$ to $O(dr)$. Write the pseudocode for the low-rank cross layer.

**(c)** Sketch the full DCN module used in DHEN: $L$ stacked cross layers applied to the flattened embedding input $x_0 = \text{flatten}(X_n) \in \mathbb{R}^{md}$, followed by a projection $W_m$ back to the embedding list format $\mathbb{R}^{d \times l}$. Identify all learned parameters and their shapes.

---

### Problem 12: DHEN Forward Pass with Tensor Shapes

Sketch the complete forward pass of a 2-layer DHEN with self-attention and linear modules, concatenation aggregation, identity shortcut (when dimensions allow), and layer normalization. Use batch size $B$, $m$ features, embedding dimension $d$, and attention head dimension $d_k$, $d_v$.

Track tensor shapes at every step:
- Input $X_0$
- Attention module output
- Linear module output
- Concatenation ensemble output
- Shortcut (identity or projection)
- After addition and LayerNorm: $X_1$
- Prediction head input

Write pseudocode with shape annotations.

---

### Problem 13: Numerically Stable Distributed NE Computation

The NE denominator requires the global empirical click rate $p = \frac{1}{n}\sum_i y_i$. In distributed training with $W$ workers each holding a local shard of size $n/W$:

**(a)** Describe the AllReduce operations needed to compute $p$ correctly across workers (hint: allreduce sum of $(y_i, 1)$ pairs).

**(b)** The NE numerator cross-entropy $H(\hat{p}, y) = -\frac{1}{n}\sum_i [y_i \log\hat{p}_i + (1-y_i)\log(1-\hat{p}_i)]$ is numerically unstable when $\hat{p}_i \approx 0$ or $\hat{p}_i \approx 1$. Describe a numerically stable implementation using the log-sigmoid function: $\log \hat{p}_i = \log\sigma(s_i) = -\text{softplus}(-s_i)$ and $\log(1-\hat{p}_i) = -\text{softplus}(s_i)$.

**(c)** Write complete pseudocode for `distributed_NE(local_logits, local_labels, worker_id)` that: (i) computes local cross-entropy in a numerically stable way; (ii) allreduces to get global cross-entropy numerator and denominator; (iii) computes the global empirical click rate and NE denominator; (iv) returns the final NE.
