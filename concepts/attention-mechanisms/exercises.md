# Attention Mechanisms: Exercises

## Table of Contents

- [[#Mathematical Development|Mathematical Development]]
  - [[#Problem 1 Variance of Scaled Dot Products|Problem 1: Variance of Scaled Dot Products]]
  - [[#Problem 2 Causal Mask and Zero Attention Weight|Problem 2: Causal Mask and Zero Attention Weight]]
  - [[#Problem 3 Equivalence of Per-Token and Matrix Attention|Problem 3: Equivalence of Per-Token and Matrix Attention]]
  - [[#Problem 4 Multi-Head Attention Parameter Count|Problem 4: Multi-Head Attention Parameter Count]]
  - [[#Problem 5 Softmax Saturation and Gradient Vanishing|Problem 5: Softmax Saturation and Gradient Vanishing]]
  - [[#Problem 6 Attention as Associative Memory Retrieval|Problem 6: Attention as Associative Memory Retrieval]]
  - [[#Problem 7 KV Cache Memory Formula|Problem 7: KV Cache Memory Formula]]
  - [[#Problem 8 Linear Attention State Recurrence from First Principles|Problem 8: Linear Attention State Recurrence from First Principles]]
  - [[#Problem 9 Linear Attention Output Formula|Problem 9: Linear Attention Output Formula]]
  - [[#Problem 10 Memory Complexity: State Matrix vs KV Cache|Problem 10: Memory Complexity: State Matrix vs KV Cache]]
  - [[#Problem 11 Chunkwise Output Decomposition|Problem 11: Chunkwise Output Decomposition]]
  - [[#Problem 12 Inter-Chunk State Recurrence|Problem 12: Inter-Chunk State Recurrence]]
  - [[#Problem 13 Exactness of the Chunkwise Form|Problem 13: Exactness of the Chunkwise Form]]
  - [[#Problem 14 Constant Decay and Effective Memory Horizon|Problem 14: Constant Decay and Effective Memory Horizon]]
  - [[#Problem 15 Negative Dot-Product Loss and Linear Attention Gradient|Problem 15: Negative Dot-Product Loss and Linear Attention Gradient]]
  - [[#Problem 16 Delta Rule from Squared-Error Loss|Problem 16: Delta Rule from Squared-Error Loss]]
  - [[#Problem 17 Delta Rule vs Linear Attention: The Erase Term|Problem 17: Delta Rule vs Linear Attention: The Erase Term]]
  - [[#Problem 18 GLA Gate Rank Constraint|Problem 18: GLA Gate Rank Constraint]]
- [[#Algorithmic Applications|Algorithmic Applications]]
  - [[#Problem 19 KV Cache Prefill and Decode Pseudocode|Problem 19: KV Cache Prefill and Decode Pseudocode]]
  - [[#Problem 20 Complexity Comparison: Softmax vs Linear Attention|Problem 20: Complexity Comparison: Softmax vs Linear Attention]]
  - [[#Problem 21 Chunkwise-Parallel Forward Pass Pseudocode|Problem 21: Chunkwise-Parallel Forward Pass Pseudocode]]
  - [[#Problem 22 Chunk Size Tradeoffs|Problem 22: Chunk Size Tradeoffs]]
  - [[#Problem 23 Linear Attention State Memory at Inference|Problem 23: Linear Attention State Memory at Inference]]
  - [[#Problem 24 Decay Variants: Parameters and Expressiveness|Problem 24: Decay Variants: Parameters and Expressiveness]]

---

## Mathematical Development

### Problem 1: Variance of Scaled Dot Products

*This problem establishes the statistical motivation for the $1/\sqrt{d_k}$ scaling factor by deriving the variance of the unscaled attention score and showing that scaling restores unit variance.*

> **Prerequisites:** cf. note [[note#2.2 Attention Score Computation and Causal Masking|§2.2 — Attention Score Computation and Causal Masking]]

(a) Let $\mathbf{q}, \mathbf{k} \in \mathbb{R}^{d_k}$ with i.i.d. components satisfying $\mathbb{E}[q_i] = \mathbb{E}[k_i] = 0$ and $\operatorname{Var}(q_i) = \operatorname{Var}(k_i) = 1$, and suppose all $q_i, k_j$ are mutually independent. Compute $\mathbb{E}[\mathbf{q}^\top \mathbf{k}]$ and $\operatorname{Var}(\mathbf{q}^\top \mathbf{k})$.

(b) Show that dividing by $\sqrt{d_k}$ gives the rescaled score $\tilde{e} = \mathbf{q}^\top \mathbf{k} / \sqrt{d_k}$ with $\mathbb{E}[\tilde{e}] = 0$ and $\operatorname{Var}(\tilde{e}) = 1$ for all $d_k$.

(c) The Jacobian of the softmax with respect to its inputs evaluated at a one-hot configuration $\mathbf{p} = \mathbf{e}_j$ (where $\mathbf{e}_j$ is a standard basis vector) equals $\mathbf{0}$. Prove this, and explain what it implies about gradient flow when $d_k$ is large and no scaling is applied.

### Problem 2: Causal Mask and Zero Attention Weight

*This problem makes rigorous the claim that adding $-\infty$ to a softmax input drives that entry's weight exactly to zero, and extends the result to the numerical floating-point regime.*

> **Prerequisites:** cf. note [[note#2.2 Attention Score Computation and Causal Masking|§2.2 — Attention Score Computation and Causal Masking]]

(a) Let $\mathbf{s} \in \mathbb{R}^T$ and let $\tilde{s}_j = s_j + M_j$ where $M_j \in \{0, -\infty\}$. Show that for any $j$ with $M_j = -\infty$, the $j$-th softmax output equals exactly $0$ regardless of the values $\{s_{j'}\}_{j' \neq j}$.

(b) In finite precision, $-\infty$ is approximated by a large negative constant $-B$ (e.g., $B = 10^9$). Derive an upper bound on the spurious attention weight $\alpha_j$ assigned to a masked position when $s_i \in [-C, C]$ for all $i$ and the mask constant is $-B$.

(c) Now suppose a practitioner applies the mask as $s_j \leftarrow 0$ for future tokens (rather than $-\infty$). Show by example that this does not implement causal masking correctly: construct a score vector where a future-token position receives nonzero weight under this scheme.

### Problem 3: Equivalence of Per-Token and Matrix Attention

*This problem proves that the matrix formula $\operatorname{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}({\mathbf{Q}\mathbf{K}^\top}/{\sqrt{d_k}} + \mathbf{M})\mathbf{V}$ is row-for-row identical to the per-token computation.*

> **Prerequisites:** cf. note [[note#3. Matrix Form of Attention|§3 — Matrix Form of Attention]]

(a) Let $A = \operatorname{softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{d_k} + \mathbf{M})$, where softmax is applied row-wise. Show that the $t$-th row of $A$, denoted $\mathbf{a}_t^\top$, equals $(\alpha_{t1}, \ldots, \alpha_{tT})$ as defined in the per-token formula of §2.3.

(b) Show that the $t$-th row of $A\mathbf{V}$ equals $\mathbf{o}_t = \sum_{j=1}^{t} \alpha_{tj} \mathbf{v}_j^\top$.

(c) Derive the shapes of each intermediate quantity in the computation $\mathbf{Q}\mathbf{K}^\top \to \mathbf{S} \to A \to A\mathbf{V}$, and state the total floating-point operation count (in terms of $T$, $d_k$, $d_v$) for each matrix multiply.

### Problem 4: Multi-Head Attention Parameter Count

*This problem re-derives the parameter count claimed in note.md — that multi-head attention has the same $4D^2$ parameters as four $D \times D$ matrices — and investigates what changes under non-standard dimensionality.*

> **Prerequisites:** cf. note [[note#4. Multi-Head Attention|§4 — Multi-Head Attention]]

(a) With $H$ heads, $d_k = d_v = D/H$, and a shared output projection $\mathbf{W}_O \in \mathbb{R}^{D \times D}$, verify that the total parameter count across all projection matrices $\{\mathbf{W}_Q^{(h)}, \mathbf{W}_K^{(h)}, \mathbf{W}_V^{(h)}\}_{h=1}^H$ and $\mathbf{W}_O$ equals $4D^2$, independent of $H$.

(b) Suppose $d_v = 2D/H$ while $d_k = D/H$ is kept standard. Derive the new total parameter count and find the ratio to the standard $4D^2$.

(c) In Grouped-Query Attention (GQA) with $G$ KV groups ($1 \leq G \leq H$), the $G$ key projection matrices $\mathbf{W}_K^{(g)} \in \mathbb{R}^{d_k \times D}$ and $G$ value matrices $\mathbf{W}_V^{(g)} \in \mathbb{R}^{d_v \times D}$ are shared across all $H/G$ query heads in the group. Derive the total parameter count for GQA as a function of $H$, $G$, and $D$, and verify the two extreme cases $G = H$ (MHA) and $G = 1$ (MQA).

### Problem 5: Softmax Saturation and Gradient Vanishing

*This problem establishes that large-magnitude inputs cause the softmax Jacobian to become rank-deficient, providing a rigorous foundation for why scaling is necessary beyond the variance argument.*

> **Prerequisites:** cf. note [[note#2.2 Attention Score Computation and Causal Masking|§2.2 — Attention Score Computation and Causal Masking]]

(a) Let $\mathbf{p} = \operatorname{softmax}(\mathbf{s})$ for $\mathbf{s} \in \mathbb{R}^n$. Show that the Jacobian $J \in \mathbb{R}^{n \times n}$ has entries $J_{ij} = p_i(\delta_{ij} - p_j)$, and prove that $J$ is always singular (has a zero eigenvalue).

(b) Suppose $s_1 \gg s_j$ for all $j \neq 1$, so that $p_1 \to 1$ and $p_j \to 0$ for $j \geq 2$. Show that all entries of $J$ converge to $0$ in this limit. Conclude that $\|J\|_F \to 0$ as $\max_j s_j - \min_j s_j \to \infty$.

(c) Compute the Frobenius norm $\|J\|_F$ for the uniform distribution $p_i = 1/n$ for all $i$, and show it equals $\sqrt{(n-1)/n}$. Use this to argue that the scaled inputs ($\operatorname{Var}(\tilde{e}) = 1$) keep the attention layer in the non-saturated regime for large $n = t$ (number of tokens attended to).

### Problem 6: Attention as Associative Memory Retrieval

*This problem formalizes the correspondence between softmax attention and content-addressable memory, showing that attention in the zero-temperature limit reduces to exact key lookup.*

> **Prerequisites:** cf. note [[linear-attention#6.1 Standard Attention as Associative Memory|§6.1 — Standard Attention as Associative Memory]]

(a) Consider the one-parameter family of attention outputs $\mathbf{o}_t(\beta) = \sum_{j=1}^{t} \alpha_{tj}(\beta) \mathbf{v}_j$ where $\alpha_{tj}(\beta) = \operatorname{softmax}_j(\beta \mathbf{q}_t^\top \mathbf{k}_j / \sqrt{d_k})$ and $\beta > 0$ is an inverse temperature. Show that as $\beta \to \infty$, if the scores $\{\mathbf{q}_t^\top \mathbf{k}_j\}$ have a unique maximum at $j = j^*$, then $\mathbf{o}_t(\beta) \to \mathbf{v}_{j^*}$.

(b) Show that as $\beta \to 0$, the output converges to the uniform average $\mathbf{o}_t \to \frac{1}{t}\sum_{j=1}^{t} \mathbf{v}_j$.

(c) Hopfield networks retrieve stored patterns by energy minimization. Standard attention can be seen as a single step of a continuous Hopfield update. Identify the correspondence: what plays the role of the "stored patterns," the "probe," and the "retrieved pattern" in the attention setting?

### Problem 7: KV Cache Memory Formula

*This problem derives the KV cache memory formula and instantiates it for a concrete large model to establish that the cache often dominates the weight memory.*

> **Prerequisites:** cf. note [[note#5.2 Memory Growth with Sequence Length|§5.2 — Memory Growth with Sequence Length]]

(a) For a transformer with $L$ layers, $H$ heads, head dimension $d = d_k = d_v$, and sequence length $T$, show that the total KV cache requires $2LHdT$ stored scalars. Simplify using $Hd = D$ to obtain $2LDT$.

(b) In float16 (2 bytes per scalar), compute the KV cache size in gigabytes for the following configuration: $L = 80$, $D = 8192$, $T = 128{,}000$. Compare this to the model weight memory, which for a 70B-parameter model at float16 is approximately $140\,\text{GB}$.

(c) In MQA, all $H$ heads share a single KV head ($G = 1$ in GQA notation). Show that the KV cache reduces to $2LdT$ scalars, a factor of $H$ smaller. For the configuration above with $H = 64$ heads, compute the MQA cache size and compare it to the MHA result from (b).

### Problem 8: Linear Attention State Recurrence from First Principles

*This problem derives the linear attention recurrence by factoring the key-similarity kernel and shows that the resulting state update is a rank-1 outer-product accumulation.*

> **Prerequisites:** cf. note [[linear-attention#2.1 Factored Dot-Product Attention|§2.1 — Factored Dot-Product Attention]]; requires Problem 6

(a) Starting from the causal softmax attention formula $\mathbf{o}_t = \sum_{j=1}^{t} \exp(\mathbf{q}_t \mathbf{k}_j^\top / \sqrt{d_k}) \mathbf{v}_j \,/\, Z_t$, explain why $\mathbf{q}_t$ cannot be factored out of the sum even though it does not depend on $j$. What property of $\exp(\cdot)$ prevents factorization?

(b) Now replace $\exp(\mathbf{q}_t \mathbf{k}_j^\top / \sqrt{d_k})$ with the identity feature map $\phi(\mathbf{q}_t)\phi(\mathbf{k}_j)^\top = \mathbf{q}_t \mathbf{k}_j^\top$ and drop the normalization. Show that $\mathbf{q}_t$ factors out of the sum over $j$, yielding $\mathbf{o}_t = \mathbf{q}_t \left(\sum_{j=1}^{t} \mathbf{k}_j^\top \mathbf{v}_j\right)$.

(c) Define $S_t = \sum_{j=1}^{t} \mathbf{k}_j^\top \mathbf{v}_j$. Show that $S_t$ satisfies the recurrence $S_t = S_{t-1} + \mathbf{k}_t^\top \mathbf{v}_t$ with $S_0 = \mathbf{0}$, and state the dimensions of $S_t$.

### Problem 9: Linear Attention Output Formula

*This problem verifies that the recurrent form $\mathbf{o}_t = \mathbf{q}_t S_t$ is consistent with the unrolled sum definition, and computes the per-step arithmetic cost.*

> **Prerequisites:** cf. note [[linear-attention#2.2 The State Matrix and Recurrence Relation|§2.2 — The State Matrix and Recurrence Relation]]; requires Problem 8

(a) Using the recurrence from Problem 8(c), prove by induction on $t$ that $S_t = \sum_{j=1}^{t} \mathbf{k}_j^\top \mathbf{v}_j$ for all $t \geq 1$.

(b) Show that $\mathbf{o}_t = \mathbf{q}_t S_t$ gives $\mathbf{o}_t = \sum_{j=1}^{t} (\mathbf{q}_t \mathbf{k}_j^\top) \mathbf{v}_j$, confirming consistency with the per-token linear attention formula in §2.1 of linear-attention.md.

(c) Count the floating-point operations required to compute: (i) the state update $S_t \leftarrow S_{t-1} + \mathbf{k}_t^\top \mathbf{v}_t$, and (ii) the output $\mathbf{o}_t = \mathbf{q}_t S_t$. Express both as functions of $d_k$ and $d_v$. Confirm that neither depends on $T$.

### Problem 10: Memory Complexity: State Matrix vs KV Cache

*This problem quantifies the concrete memory advantage of linear attention at inference time, comparing the fixed-size state matrix to the linearly growing KV cache.*

> **Prerequisites:** cf. note [[linear-attention#2.4 Memory Complexity Comparison|§2.4 — Memory Complexity Comparison]]; requires Problems 7 and 9

(a) Show that during autoregressive inference with linear attention, only $S_t \in \mathbb{R}^{d_k \times d_v}$ needs to be stored (not the history $\{S_j\}_{j < t}$). What recurrence property makes this possible?

(b) For the 70B-model configuration from Problem 7 ($L = 80$, $D = 8192$, $H = 64$ heads, $d = 128$), compute the linear attention state memory per layer (one matrix $S_t$ per head) in megabytes at float16. Then compute the total across all $L$ layers and $H$ heads.

(c) Compare: at what sequence length $T^*$ does the KV cache memory (from Problem 7) equal the total linear attention state memory from (b)? Express $T^*$ in terms of $d_k$, $d_v$, $L$, $H$.

### Problem 11: Chunkwise Output Decomposition

*This problem proves the two-part output decomposition $O_i = O_i^{\text{inter}} + O_i^{\text{intra}}$ for chunkwise-parallel linear attention by expanding the state at every token in a chunk.*

> **Prerequisites:** cf. note [[linear-attention#4.4 Output Decomposition into Two Parts|§4.4 — Output Decomposition into Two Parts]]; requires Problem 8

(a) Let chunk $i$ contain absolute token indices $\{(i-1)C+1, \ldots, iC\}$ with local indices $s \in \{1, \ldots, C\}$. Let $S_{i-1}^{\text{end}}$ be the state entering the chunk. Write $S_t$ for the absolute token at position $t = (i-1)C + s$ as the sum of $S_{i-1}^{\text{end}}$ and a partial state accumulated within the chunk.

(b) Substitute the decomposition from (a) into $\mathbf{o}_t = \mathbf{q}_t S_t$ to obtain two terms. Identify each as a contribution from outside the chunk (inter) and from within the chunk (intra).

(c) Stack over all $s \in \{1, \ldots, C\}$ to obtain the matrix form $O_i = Q_i S_{i-1}^{\text{end}} + (Q_i K_i^\top \odot M) V_i$, where $M \in \{0,1\}^{C \times C}$ is the lower-triangular causal mask. Carefully justify why the intra-chunk term takes precisely this form.

### Problem 12: Inter-Chunk State Recurrence

*This problem derives the inter-chunk state update $S_i^{\text{end}} = S_{i-1}^{\text{end}} + K_i^\top V_i$ and shows it is exactly the token-level recurrence aggregated over one chunk.*

> **Prerequisites:** cf. note [[linear-attention#4.3 Inter-Chunk: Recurrent State Passing|§4.3 — Inter-Chunk: Recurrent State Passing]]; requires Problem 8

(a) Using the token-level recurrence $S_t = S_{t-1} + \mathbf{k}_t^\top \mathbf{v}_t$, unroll the recurrence from position $(i-1)C+1$ to $iC$ (the full extent of chunk $i$) to obtain an expression for $S_{iC}$ in terms of $S_{(i-1)C}$ and the keys/values within chunk $i$.

(b) Show that $K_i^\top V_i = \sum_{s=1}^{C} \mathbf{k}_{(i-1)C+s}^\top \mathbf{v}_{(i-1)C+s}$, confirming that the chunk-level update aggregates all $C$ rank-1 outer products. State the shape of $K_i^\top V_i$.

(c) The computation $K_i^\top V_i$ can be performed as a single matrix multiply. Identify the shapes of the two operands and the result, and count the floating-point operations. Express the total cost over all $N = T/C$ chunks and compare to the $O(T d_k d_v)$ cost of the sequential recurrence.

### Problem 13: Exactness of the Chunkwise Form

*This problem provides a self-contained proof that the chunkwise-parallel algorithm produces outputs identical to the sequential token-level recurrence, with no approximation.*

> **Prerequisites:** cf. note [[linear-attention#4.4 Output Decomposition into Two Parts|§4.4 — Output Decomposition into Two Parts]]; requires Problems 11 and 12

(a) State precisely what "exactness" means here: for every token $t$ in the sequence, the output $\mathbf{o}_t$ produced by the chunkwise algorithm equals the output $\mathbf{o}_t$ produced by running the sequential recurrence $S_t = S_{t-1} + \mathbf{k}_t^\top \mathbf{v}_t$, $\mathbf{o}_t = \mathbf{q}_t S_t$.

(b) Prove exactness for an arbitrary token $t = (i-1)C + s$ (in chunk $i$, local index $s$) by: (i) expanding $S_t$ using the sequential recurrence, (ii) splitting the sum at the chunk boundary, and (iii) matching the two resulting terms to $O_i^{\text{inter}}$ and $O_i^{\text{intra}}$.

(c) Identify the key algebraic property of linear attention (absent in softmax attention) that makes this exact decomposition possible. Why does the analogous argument fail for softmax attention?

### Problem 14: Constant Decay and Effective Memory Horizon

*This problem analyzes how a constant scalar decay factor $\gamma \in (0, 1)$ induces exponential forgetting and defines the effective memory horizon.*

> **Prerequisites:** cf. note [[linear-attention#5.2 Constant Scalar Decay: RetNet|§5.2 — Constant Scalar Decay: RetNet]]; requires Problem 8

(a) Consider the decayed state recurrence $S_t = \gamma S_{t-1} + \mathbf{k}_t^\top \mathbf{v}_t$ with $\gamma \in (0,1)$ and $S_0 = \mathbf{0}$. Unroll the recurrence to show $S_t = \sum_{j=1}^{t} \gamma^{t-j} \mathbf{k}_j^\top \mathbf{v}_j$.

(b) Define the *effective weight* of token $j$ as $w_j = \gamma^{t-j}$, normalized so that $\sum_{j=1}^{t} w_j = 1$ (take $t \to \infty$). Compute the expected "age" $\bar{\tau} = \sum_{j=0}^{\infty} j \cdot \tilde{w}_j$ where $\tilde{w}_j = (1-\gamma)\gamma^j$ is the normalized geometric weight on a token $j$ steps in the past. Show $\bar{\tau} = \gamma / (1 - \gamma)$.

(c) RetNet uses head-specific decays $\gamma_h = 1 - 2^{-(5+h)}$ for head $h = 1, 2, \ldots, H$. Compute the effective memory horizon $\bar{\tau}_h$ for $h = 1$ and $h = H$, and interpret the geometric spread of horizons across heads.

### Problem 15: Negative Dot-Product Loss and Linear Attention Gradient

*This problem derives the connection between linear attention and online gradient descent on a negative dot-product loss, showing that the state update is a gradient step.*

> **Prerequisites:** cf. note [[linear-attention#6.2 Linear Attention as Online Linear Regression|§6.2 — Linear Attention as Online Linear Regression]]; requires Problem 8

(a) Define the loss $\mathcal{L}_t(S) = -\langle S \mathbf{k}_t^\top, \mathbf{v}_t^\top \rangle$ where $\langle \cdot, \cdot \rangle$ denotes the Frobenius inner product. Expand this expression and compute $\nabla_S \mathcal{L}_t$.

(b) Perform one step of gradient descent with step size $\eta = 1$ starting from $S_{t-1}$:
$$S_t = S_{t-1} - \eta \nabla_S \mathcal{L}_t\big|_{S=S_{t-1}}$$
Show this equals the linear attention update $S_t = S_{t-1} + \mathbf{k}_t^\top \mathbf{v}_t$.

(c) Show that $\mathcal{L}_t(S)$ is linear in $S$ (not quadratic) and therefore unbounded below. Explain why this means the "loss" interpretation is degenerate: the gradient step does not converge to a minimum, but instead increases $\mathcal{L}_t$ indefinitely. What does this imply about the stability of linear attention as a memory model?

### Problem 16: Delta Rule from Squared-Error Loss

*This problem derives the delta rule update by computing the gradient of the squared-error loss and shows it introduces a prediction-error correction absent in vanilla linear attention.*

> **Prerequisites:** cf. note [[linear-attention#6.4 The Delta Rule: Squared-Error Loss and Its Gradient|§6.4 — The Delta Rule: Squared-Error Loss and Its Gradient]]; requires Problem 15

(a) Define the squared-error loss $\mathcal{L}_t(S) = \frac{1}{2}\|S_{t-1}\mathbf{k}_t^\top - \mathbf{v}_t^\top\|^2$, where the norm is the Euclidean norm on $\mathbb{R}^{d_v}$. Compute $\nabla_S \mathcal{L}_t$ evaluated at $S = S_{t-1}$. Show your work using the chain rule for matrix calculus.

(b) Apply gradient descent with step size $\beta_t$:
$$S_t = S_{t-1} - \beta_t \nabla_S \mathcal{L}_t\big|_{S=S_{t-1}}$$
Factor the result to obtain $S_t = S_{t-1} + \beta_t \mathbf{k}_t^\top (\mathbf{v}_t - \mathbf{k}_t S_{t-1})$.

(c) Identify the "prediction error" term in the delta rule. What does the model "predict" when it applies $\mathbf{k}_t$ to the current state $S_{t-1}$? Under what condition on $S_{t-1}$ and $\mathbf{k}_t$ does the delta rule reduce exactly to the linear attention update?

### Problem 17: Delta Rule vs Linear Attention: The Erase Term

*This problem isolates the structural difference between delta rule and linear attention updates and analyzes when the extra "erase" term matters.*

> **Prerequisites:** cf. note [[linear-attention#6.4 The Delta Rule: Squared-Error Loss and Its Gradient|§6.4 — The Delta Rule: Squared-Error Loss and Its Gradient]]; requires Problem 16

(a) Write the delta rule as $S_t = S_{t-1} + \beta_t \mathbf{k}_t^\top \mathbf{v}_t - \beta_t \mathbf{k}_t^\top (\mathbf{k}_t S_{t-1})$. The third term $-\beta_t \mathbf{k}_t^\top \mathbf{k}_t S_{t-1}$ is the "erase" term. Show that it acts as a rank-1 projection of the previous state onto the subspace spanned by $\mathbf{k}_t$, scaled by $\beta_t \|\mathbf{k}_t\|^2$.

(b) Consider the setting where keys are orthonormal: $\mathbf{k}_t \mathbf{k}_{t'}^\top = \delta_{tt'}$ for all $t \neq t'$. Show that in this case the erase term precisely removes the old association $\mathbf{k}_{t'}^\top \mathbf{v}_{t'}$ (if $\mathbf{k}_t = \mathbf{k}_{t'}$ for some earlier step $t'$) before writing the new one. Why is this beneficial compared to the purely additive linear attention update?

(c) In contrast, suppose $\mathbf{k}_t$ is orthogonal to all past keys: $\mathbf{k}_t \mathbf{k}_{j}^\top = 0$ for all $j < t$. Show that the erase term $\mathbf{k}_t^\top \mathbf{k}_t S_{t-1}$ vanishes in this case (assuming $S_{t-1}$ only contains contributions from past keys). Conclude: under what distributional assumption about the keys does the delta rule provide the most benefit over linear attention?

### Problem 18: GLA Gate Rank Constraint

*This problem justifies why the Gated Linear Attention gate $\boldsymbol{\alpha}_t \boldsymbol{\beta}_t^\top$ must be rank-1 for the chunkwise algorithm to remain hardware-efficient.*

> **Prerequisites:** cf. note [[linear-attention#5.4 Vector Data-Dependent Gating: GLA|§5.4 — Vector Data-Dependent Gating: GLA]]; requires Problem 12

(a) The GLA recurrence is $S_t = (\boldsymbol{\alpha}_t \boldsymbol{\beta}_t^\top) \odot S_{t-1} + \mathbf{k}_t^\top \mathbf{v}_t$. Unroll this recurrence over chunk $i$ to express $S_{iC}$ in terms of $S_{(i-1)C}$ and the keys, values, and gate vectors within the chunk. Show the cumulative gate across the chunk factors as an outer product.

(b) Suppose instead the gate is an arbitrary matrix $G_t \in \mathbb{R}^{d_k \times d_v}$ (not necessarily rank-1). Write the unrolled expression for $S_{iC}$. Explain why this form cannot be expressed as a single matrix multiply of the form $(d_k \times C)(C \times d_v)$.

(c) Show that the rank-1 constraint on the gate preserves the form $K_i^\top \tilde{V}_i$ for some modified value matrix $\tilde{V}_i \in \mathbb{R}^{C \times d_v}$, and identify what $\tilde{V}_i$ is. Why does this make the inter-chunk state update GPU-efficient?

---

## Algorithmic Applications

### Problem 19: KV Cache Prefill and Decode Pseudocode

*This problem forces precise formulation of the two-phase KV cache algorithm and establishes the asymptotic cost of each phase.*

> **Prerequisites:** cf. note [[note#5.1 Motivation: Avoiding Redundant Computation During Decoding|§5.1 — Motivation: Avoiding Redundant Computation During Decoding]]

(a) **Prefill phase pseudocode**: Write pseudocode for processing a prompt of $T_0$ tokens. The pseudocode should: accept the prompt token matrix $\mathbf{X} \in \mathbb{R}^{T_0 \times D}$; compute $\mathbf{Q}, \mathbf{K}, \mathbf{V}$; run full causal attention; populate the cache $\mathcal{C}$ with all $T_0$ key-value pairs. Annotate with shapes at each step.

(b) **Decode phase pseudocode**: Write pseudocode for generating a single new token $t = T_0 + s$ given the current cache $\mathcal{C}$. The pseudocode should: compute only $\mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t$; retrieve the full key and value tensors from cache; compute the attention output; append $(\mathbf{k}_t, \mathbf{v}_t)$ to $\mathcal{C}$.

(c) **Complexity**: State the time complexity (in multiply-adds) and memory complexity for (i) the prefill phase and (ii) a single decode step at position $t$. For a single decode step, show that the complexity is $O(tD)$ not $O(t^2)$ — why does the cache make this possible?

### Problem 20: Complexity Comparison: Softmax vs Linear Attention

*This problem tabulates training and inference complexity for softmax and linear attention, making explicit under what sequence lengths each architecture is preferred.*

> **Prerequisites:** cf. note [[linear-attention#2.4 Memory Complexity Comparison|§2.4 — Memory Complexity Comparison]]; requires Problems 7 and 10

(a) **Training complexity table**: Fill in the following table for one layer, one head, sequence length $T$, key/value dimension $d$:

| Metric | Softmax Attention | Linear Attention (sequential) | Linear Attention (chunkwise, chunk size $C$) |
|---|---|---|---|
| Time complexity | | | |
| Memory (activations) | | | |
| Sequential depth | | | |

(b) **Inference complexity**: For autoregressive decoding at step $t$ (generating the $t$-th token), compare: (i) softmax with KV cache, (ii) linear attention (recurrent form). State the time and memory cost for each. At what value of $t$ (if any) does the softmax cost exceed the linear attention cost?

(c) **Crossover analysis**: For fixed $d$ and varying $T$, plot the asymptotic regimes. Identify the sequence length $T^* = T^*(d)$ at which the chunkwise linear attention training cost (in FLOPs) equals the softmax attention training cost. Express $T^*$ in terms of $d$ and $C$.

### Problem 21: Chunkwise-Parallel Forward Pass Pseudocode

*This problem requires writing a complete, annotated pseudocode for the chunkwise-parallel linear attention forward pass, making the three-matrix-multiply structure explicit.*

> **Prerequisites:** cf. note [[linear-attention#4. Chunkwise-Parallel Form|§4 — Chunkwise-Parallel Form]]; requires Problems 11 and 12

(a) **Inputs and data structures**: Specify the inputs to the procedure (stacked $Q, K, V$ matrices, chunk size $C$), the output (stacked $O$ matrix), and the intermediate variables needed (state $S$, chunk matrices, intra/inter outputs).

(b) **Core loop**: Write the main loop over chunks $i = 1, \ldots, N = T/C$. For each chunk, perform: (i) slice out $Q_i, K_i, V_i$; (ii) compute $O_i^{\text{inter}} = Q_i S$; (iii) compute $O_i^{\text{intra}} = (Q_i K_i^\top \odot M) V_i$ using a lower-triangular mask $M$; (iv) update $S \leftarrow S + K_i^\top V_i$; (v) accumulate $O_i = O_i^{\text{inter}} + O_i^{\text{intra}}$. Annotate each line with the shape of every matrix involved.

(c) **Correctness check**: Verify the state update in (b)(iv) is performed *after* computing $O_i^{\text{inter}}$ and *before* computing $O_{i+1}^{\text{inter}}$. Explain what goes wrong if the update order is reversed.

### Problem 22: Chunk Size Tradeoffs

*This problem quantifies the memory-compute tradeoff controlled by the chunk size $C$ and identifies the regime where chunkwise form outperforms both extremes.*

> **Prerequisites:** cf. note [[linear-attention#4.5 IO-Aware Implementation|§4.5 — IO-Aware Implementation]]; requires Problem 20

(a) **Sequential depth**: Show that the chunkwise algorithm has sequential depth $N = T/C$ (in the inter-chunk recurrence). At the extremes $C = 1$ and $C = T$, identify what the algorithm reduces to. What is the sequential depth at each extreme?

(b) **Memory for intermediate states**: During the forward pass of the chunkwise algorithm, which matrices must reside in SRAM simultaneously? Express the SRAM requirement as a function of $C$, $d_k$, and $d_v$. For fixed SRAM capacity $M_{\text{SRAM}}$, derive the maximum chunk size $C_{\max}$.

(c) **FLOP analysis**: The intra-chunk attention $Q_i K_i^\top$ costs $O(C^2 d_k)$ FLOPs per chunk, and the inter-chunk state update $K_i^\top V_i$ costs $O(C d_k d_v)$. Over all $N = T/C$ chunks, compute the total FLOPs for each term. Identify the $C$ at which the two terms contribute equally, and argue why $C \approx d_v$ is often near-optimal.

### Problem 23: Linear Attention State Memory at Inference

*This problem instantiates the linear attention state memory formula for a concrete 7B-parameter model and computes the total per-layer and aggregate state sizes.*

> **Prerequisites:** cf. note [[linear-attention#2.4 Memory Complexity Comparison|§2.4 — Memory Complexity Comparison]]; requires Problem 10

(a) **Model configuration**: Suppose a 7B-parameter linear attention model has $L = 32$ layers, $H = 32$ heads, and $D = 4096$ (so $d_k = d_v = d = D/H = 128$). For one head, compute the size of the state matrix $S \in \mathbb{R}^{d_k \times d_v}$ in float16 (kilobytes).

(b) **Total state memory**: Compute the total state memory across all $L \times H$ heads in megabytes. How does this compare to the model weight memory of approximately $14\,\text{GB}$ (7B parameters at float16)?

(c) **Sequence-length independence**: At inference, a softmax attention model with the same architecture would require a KV cache of size $2LDT$ float16 elements. Compute the sequence length $T^*$ at which the KV cache equals the linear attention state memory computed in (b). For $T > T^*$, which model is more memory-efficient at inference?

### Problem 24: Decay Variants: Parameters and Expressiveness

*This problem compares constant decay (RetNet), scalar data-dependent decay (Mamba-2/SSD), and vector data-dependent gating (GLA) in terms of the extra parameters each introduces and the expressiveness gained.*

> **Prerequisites:** cf. note [[linear-attention#5. Decay and Gating Mechanisms|§5 — Decay and Gating Mechanisms]]; requires Problems 14 and 18

(a) **Parameter count overhead**: For a single layer with $H$ heads, $D$-dimensional input, and $d = D/H$:
  - Constant decay (RetNet): the decay $\gamma_h$ per head is a fixed constant, not a learned parameter. How many extra trainable parameters does the decay introduce?
  - Scalar data-dependent decay (SSD): $\gamma_t = \sigma(\mathbf{w}_a^\top \mathbf{x}_t)$ for a learned vector $\mathbf{w}_a \in \mathbb{R}^D$. How many extra parameters per head?
  - Vector data-dependent gating (GLA): $\boldsymbol{\alpha}_t \in \mathbb{R}^{d_k}$ and $\boldsymbol{\beta}_t \in \mathbb{R}^{d_v}$ are linear projections of $\mathbf{x}_t$. How many extra parameters per head?

(b) **Expressiveness**: For each variant, describe the set of effective "decay masks" it can produce. Specifically: (i) what is the form of the mask matrix $D \in \mathbb{R}^{T \times T}$ (with $D_{t,j}$ being the effective weight of token $j$ at time $t$) for each variant? (ii) Under what conditions does the scalar data-dependent decay reduce to the constant decay?

(c) **State update structure**: For each variant, write the state update equation and identify whether the update preserves the outer-product (rank-1 per step) structure that enables the chunkwise GEMM. Explain why GLA's rank-1 gate is necessary for hardware efficiency, citing the result from Problem 18.
