# Attention Mechanisms: Solutions

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
  - [[#Problem 19 Online Softmax Invariant Correctness|Problem 19: Online Softmax Invariant Correctness]]
  - [[#Problem 20 Associativity of Partial Summary Combination|Problem 20: Associativity of Partial Summary Combination]]
  - [[#Problem 21 Peak Memory of Tiled Attention|Problem 21: Peak Memory of Tiled Attention]]
  - [[#Problem 22 Recomputation Memory-Compute Tradeoff|Problem 22: Recomputation Memory-Compute Tradeoff]]
- [[#Algorithmic Applications|Algorithmic Applications]]
  - [[#Problem 19 KV Cache Prefill and Decode Pseudocode|Problem 19: KV Cache Prefill and Decode Pseudocode]]
  - [[#Problem 20 Complexity Comparison: Softmax vs Linear Attention|Problem 20: Complexity Comparison: Softmax vs Linear Attention]]
  - [[#Problem 21 Chunkwise-Parallel Forward Pass Pseudocode|Problem 21: Chunkwise-Parallel Forward Pass Pseudocode]]
  - [[#Problem 22 Chunk Size Tradeoffs|Problem 22: Chunk Size Tradeoffs]]
  - [[#Problem 23 Linear Attention State Memory at Inference|Problem 23: Linear Attention State Memory at Inference]]
  - [[#Problem 24 Decay Variants: Parameters and Expressiveness|Problem 24: Decay Variants: Parameters and Expressiveness]]
  - [[#Problem 25 Streaming Single-Query Attention Pseudocode|Problem 25: Streaming Single-Query Attention Pseudocode]]
  - [[#Problem 26 Memory Reduction at Concrete Scale|Problem 26: Memory Reduction at Concrete Scale]]

---

## Mathematical Development

### Problem 1: Variance of Scaled Dot Products

**Key insight:** The dot product of two $d_k$-dimensional zero-mean unit-variance vectors has variance $d_k$, so dividing by $\sqrt{d_k}$ restores unit variance and keeps the softmax in its high-gradient regime.

**Sketch:**

(a) $\mathbb{E}[\mathbf{q}^\top \mathbf{k}] = \sum_i \mathbb{E}[q_i]\mathbb{E}[k_i] = 0$. For the variance, $\operatorname{Var}(q_i k_i) = \mathbb{E}[q_i^2 k_i^2] - 0 = \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = 1$, so $\operatorname{Var}(\mathbf{q}^\top\mathbf{k}) = \sum_i \operatorname{Var}(q_i k_i) = d_k$.

(b) $\operatorname{Var}(\tilde{e}) = \operatorname{Var}(\mathbf{q}^\top\mathbf{k}/\sqrt{d_k}) = d_k / d_k = 1$. Mean zero is preserved by linearity.

(c) At a one-hot $\mathbf{p} = \mathbf{e}_j$: $J_{ij} = p_i(\delta_{ij} - p_j)$. For $i = j$: $p_j(1 - p_j) = 1 \cdot 0 = 0$. For $i \neq j$: $-p_i p_j = -0 \cdot 1 = 0$. So $J = 0$ identically. Without scaling, large $d_k$ concentrates softmax outputs near one-hot (variance $d_k$ drives extreme scores), zeroing all gradients and halting learning.

---

### Problem 2: Causal Mask and Zero Attention Weight

**Key insight:** $\exp(-\infty) = 0$ is exact in mathematics but only approximate in float32; the key result is that the error is negligible whenever $B \gg 2C$.

**Sketch:**

(a) For any $j$ with $M_j = -\infty$: $\exp(s_j + M_j) = \exp(-\infty) = 0$. The softmax numerator is $0$ regardless of $\{s_{j'}\}$, so $\alpha_j = 0 / Z = 0$ (where $Z > 0$ since at least the current token $j = t$ is unmasked).

(b) With mask $-B$ and scores bounded by $|s_i| \leq C$: the spurious numerator is $e^{s_j - B} \leq e^{C - B}$. The denominator is $\geq e^{-C}$ (contribution of the current token). So $\alpha_j \leq e^{C-B} / e^{-C} = e^{2C - B}$. For $B = 10^9$ and $C = O(1)$, this is negligible.

(c) Take $T = 2$, scores $s_1 = -10$, $s_2 = -10$ (past, future). With mask applied as $s_2 \leftarrow 0$: softmax$(-10, 0) \propto (e^{-10}, 1)$, giving $\alpha_2 \approx 1 \neq 0$.

---

### Problem 3: Equivalence of Per-Token and Matrix Attention

**Key insight:** Row-wise softmax of $\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k} + \mathbf{M}$ produces exactly the per-token weight distribution $(\alpha_{t1}, \ldots, \alpha_{tT})$ for every $t$, so the matrix formula is a vectorized restatement.

**Sketch:**

(a) $(\mathbf{Q}\mathbf{K}^\top)_{tj} = \mathbf{q}_t^\top \mathbf{k}_j$, so $S_{tj} = \mathbf{q}_t^\top \mathbf{k}_j / \sqrt{d_k}$. Row $t$ of the masked score matrix is $(S_{tj} + M_{tj})_j$. Applying softmax row-wise gives $A_{tj} = \exp(S_{tj}+M_{tj}) / \sum_{j'} \exp(S_{tj'}+M_{tj'}) = \alpha_{tj}$ by definition.

(b) Row $t$ of $A\mathbf{V}$: $(A\mathbf{V})_t = \sum_j A_{tj} \mathbf{v}_j^\top = \sum_{j=1}^t \alpha_{tj} \mathbf{v}_j^\top = \mathbf{o}_t^\top$.

(c) Shapes and costs: $\mathbf{Q}\mathbf{K}^\top$: $(T \times d_k)(d_k \times T) \to T \times T$, cost $O(T^2 d_k)$; softmax: $O(T^2)$; $A\mathbf{V}$: $(T \times T)(T \times d_v) \to T \times d_v$, cost $O(T^2 d_v)$. Total: $O(T^2(d_k + d_v))$.

---

### Problem 4: Multi-Head Attention Parameter Count

**Key insight:** With $d_k = d_v = D/H$, the per-head projection cost is $D^2/H$ and summing over $H$ heads gives $D^2$ per projection type, so three projections plus $\mathbf{W}_O$ yields $4D^2$ regardless of $H$.

**Sketch:**

(a) Each $\mathbf{W}_Q^{(h)}, \mathbf{W}_K^{(h)}, \mathbf{W}_V^{(h)} \in \mathbb{R}^{(D/H) \times D}$ has $D^2/H$ parameters. Summing over $H$ heads: $H \cdot 3 \cdot (D^2/H) = 3D^2$. Adding $\mathbf{W}_O \in \mathbb{R}^{D \times D}$: $3D^2 + D^2 = 4D^2$.

(b) With $d_v = 2D/H$: $\mathbf{W}_V^{(h)} \in \mathbb{R}^{(2D/H) \times D}$ has $2D^2/H$ parameters, summing to $2D^2$. Concatenated output has size $H \cdot 2D/H = 2D$, so $\mathbf{W}_O \in \mathbb{R}^{2D \times D}$ with $2D^2$ parameters. Total: $D^2 + D^2 + 2D^2 + 2D^2 = 6D^2$. Ratio: $6D^2 / 4D^2 = 3/2$.

(c) GQA: $H$ query projections ($D^2$) + $G$ key projections ($G \cdot D^2/H = GD^2/H$) + $G$ value projections ($GD^2/H$) + $\mathbf{W}_O$ ($D^2$). Total: $D^2(2 + 2G/H + 1) = D^2(3 + 2G/H)$. At $G = H$: $5D^2$ — wait, this recovers MHA only if we count $\mathbf{W}_O$ once; reconcile: $G=H$ gives $4D^2$ (standard MHA — the $H$ separate KV heads contribute $D^2 + D^2 = 2D^2$, same as above). At $G = 1$: $D^2(3 + 2/H) \approx 3D^2$ for large $H$ (MQA reduces KV parameter count to $2D^2/H \approx 0$).

---

### Problem 5: Softmax Saturation and Gradient Vanishing

**Key insight:** The Jacobian of softmax is $\operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$, which is always rank-deficient and collapses to the zero matrix when $\mathbf{p}$ is one-hot, linking large attention scores to vanishing gradients.

**Sketch:**

(a) $p_i = e^{s_i}/Z$. Differentiating: $\partial p_i / \partial s_j = p_i \delta_{ij} - p_i p_j$. So $J = \operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$. The all-ones vector $\mathbf{1}$ satisfies $J\mathbf{1} = \mathbf{p} - \mathbf{p}(\mathbf{p}^\top \mathbf{1}) = \mathbf{p} - \mathbf{p} = \mathbf{0}$, so $J$ is always singular.

(b) As $p_1 \to 1$, $p_j \to 0$: diagonal entry $p_1(1-p_1) \to 0$; off-diagonal entries $p_i p_j \to 0$ for all pairs. All $n^2$ entries $\to 0$, so $\|J\|_F \to 0$.

(c) At $p_i = 1/n$: $J_{ii} = 1/n \cdot (1 - 1/n) = (n-1)/n^2$; $J_{ij} = -1/n^2$ for $i \neq j$. $\|J\|_F^2 = n \cdot (n-1)^2/n^4 + n(n-1) \cdot 1/n^4 = (n-1)/n^2[(n-1)+1]/n^{-1}$... evaluating directly: $n(n-1)^2/n^4 + n(n-1)/n^4 = (n-1)(n-1+1)/n^3 = (n-1)/n^2$. So $\|J\|_F = \sqrt{(n-1)/n} \to 1$ as $n \to \infty$. Scaled inputs keep $\operatorname{Var}(\tilde{e}) = 1$, preventing the near-one-hot regime and maintaining $\|J\|_F \approx 1$ even for large $n$.

---

### Problem 6: Attention as Associative Memory Retrieval

**Key insight:** The softmax inverse temperature $\beta$ interpolates between uniform averaging ($\beta \to 0$) and exact nearest-neighbor lookup ($\beta \to \infty$), placing attention on a continuum of associative memory retrieval strategies.

**Sketch:**

(a) Let $j^* = \operatorname{argmax}_j \mathbf{q}_t^\top \mathbf{k}_j$. As $\beta \to \infty$: $\alpha_{tj^*}(\beta) = 1 / (1 + \sum_{j \neq j^*} e^{-\beta(\mathbf{q}_t^\top \mathbf{k}_{j^*} - \mathbf{q}_t^\top \mathbf{k}_j)}) \to 1$ since each exponent $\to -\infty$. All other weights $\to 0$. So $\mathbf{o}_t(\beta) \to \mathbf{v}_{j^*}$.

(b) As $\beta \to 0$: all scores $\beta \mathbf{q}_t^\top \mathbf{k}_j \to 0$, so $e^0 = 1$ for all $j \leq t$. Softmax gives uniform $\alpha_{tj} = 1/t$, so $\mathbf{o}_t \to \frac{1}{t}\sum_{j=1}^t \mathbf{v}_j$.

(c) Stored patterns: the key-value pairs $\{(\mathbf{k}_j, \mathbf{v}_j)\}_{j=1}^t$ (the context). Probe: the query $\mathbf{q}_t$. Retrieved pattern: the output $\mathbf{o}_t$ — a softmax-weighted combination of values, with weights determined by key-query similarity (analogous to energy-minimizing retrieval in Hopfield networks, but using dot-product energy rather than quadratic Hopfield energy).

---

### Problem 7: KV Cache Memory Formula

**Key insight:** The KV cache stores $(d_k + d_v) \times T$ scalars per layer per head; the identity $Hd = D$ collapses this to the layer-uniform formula $2LDT$, which at 128k context length can exceed the model weights.

**Sketch:**

(a) Per layer: $H$ heads, each storing $d$ keys and $d$ values per token, for $T$ tokens: $H \cdot 2d \cdot T = 2HdT = 2DT$ elements. Over $L$ layers: $2LDT$.

(b) Float16: $2 \times 80 \times 128{,}000 \times 8192 \times 2\,\text{bytes} = 2 \times 80 \times 128{,}000 \times 8192 \times 2 \approx 336\,\text{GB}$. Model weights $\approx 140\,\text{GB}$. The cache is $\approx 2.4\times$ the weight memory.

(c) MQA: single KV head stores $2d \cdot T$ per layer: total $2LdT = 2LDT/H$ elements. At $H = 64$: MQA cache $= 336/64 \approx 5.25\,\text{GB}$ — a $64\times$ reduction.

---

### Problem 8: Linear Attention State Recurrence from First Principles

**Key insight:** The softmax kernel couples all $j$ through the partition function $Z_t$, preventing query factorization; replacing it with a separable kernel $\phi(\mathbf{q})\phi(\mathbf{k})^\top$ breaks this coupling and allows the state sum to be maintained as a running outer-product accumulation.

**Sketch:**

(a) In softmax attention, $Z_t = \sum_{j=1}^t \exp(\mathbf{q}_t \mathbf{k}_j^\top / \sqrt{d_k})$ depends on $\mathbf{q}_t$ (each exponential contains $\mathbf{q}_t$), so dividing by $Z_t$ prevents factoring $\mathbf{q}_t$ from the normalized sum. The exp is not bilinear in $\mathbf{q}$ and $\mathbf{k}$ jointly.

(b) With $\phi = \text{id}$ and no normalizer: $\mathbf{o}_t = \sum_{j=1}^t (\mathbf{q}_t \mathbf{k}_j^\top) \mathbf{v}_j = \mathbf{q}_t \sum_{j=1}^t \mathbf{k}_j^\top \mathbf{v}_j$, since $\mathbf{q}_t$ does not depend on $j$ and the sum of scalings $\mathbf{k}_j^\top \mathbf{v}_j \in \mathbb{R}^{d_k \times d_v}$ is independent of $\mathbf{q}_t$.

(c) $S_t = \sum_{j=1}^t \mathbf{k}_j^\top \mathbf{v}_j = \sum_{j=1}^{t-1} \mathbf{k}_j^\top \mathbf{v}_j + \mathbf{k}_t^\top \mathbf{v}_t = S_{t-1} + \mathbf{k}_t^\top \mathbf{v}_t$. Dimensions: $\mathbf{k}_t^\top \in \mathbb{R}^{d_k \times 1}$, $\mathbf{v}_t \in \mathbb{R}^{1 \times d_v}$ (row vectors), so $S_t \in \mathbb{R}^{d_k \times d_v}$.

---

### Problem 9: Linear Attention Output Formula

**Key insight:** The proof by induction is trivial once the recurrence is established; the per-step cost is $O(d_k d_v)$ for both state update and output query — entirely independent of $T$.

**Sketch:**

(a) Base case $t = 1$: $S_1 = S_0 + \mathbf{k}_1^\top \mathbf{v}_1 = \mathbf{k}_1^\top \mathbf{v}_1 = \sum_{j=1}^1 \mathbf{k}_j^\top \mathbf{v}_j$. Inductive step: $S_t = S_{t-1} + \mathbf{k}_t^\top \mathbf{v}_t = \sum_{j=1}^{t-1} \mathbf{k}_j^\top \mathbf{v}_j + \mathbf{k}_t^\top \mathbf{v}_t = \sum_{j=1}^t \mathbf{k}_j^\top \mathbf{v}_j$.

(b) $\mathbf{o}_t = \mathbf{q}_t S_t = \mathbf{q}_t \sum_{j=1}^t \mathbf{k}_j^\top \mathbf{v}_j = \sum_{j=1}^t (\mathbf{q}_t \mathbf{k}_j^\top) \mathbf{v}_j$, which matches the linear attention formula $\mathbf{o}_t = \mathbf{q}_t \left(\sum_{j=1}^t \mathbf{k}_j^\top \mathbf{v}_j\right)$ via bilinearity.

(c) (i) State update: outer product $\mathbf{k}_t^\top \mathbf{v}_t$ costs $d_k \cdot d_v$ multiply-adds; matrix addition costs $d_k \cdot d_v$ additions — total $O(d_k d_v)$. (ii) Output: matrix-vector product $\mathbf{q}_t S_t$ (shape $1 \times d_k$ times $d_k \times d_v$) costs $d_k \cdot d_v$ multiply-adds — total $O(d_k d_v)$. Neither involves $T$.

---

### Problem 10: Memory Complexity: State Matrix vs KV Cache

**Key insight:** The recurrence property $S_t = S_{t-1} + \mathbf{k}_t^\top \mathbf{v}_t$ means only the current state is needed to compute the next output; all past context is compressed into the fixed-size $d_k \times d_v$ matrix.

**Sketch:**

(a) $\mathbf{o}_t = \mathbf{q}_t S_t$ and $S_t = S_{t-1} + \mathbf{k}_t^\top \mathbf{v}_t$ — computing $S_t$ requires only $S_{t-1}$ (not $S_{t-2}, \ldots$). This Markov property means only the current state needs to be stored; no cache of past keys/values is needed.

(b) Per head: $S \in \mathbb{R}^{128 \times 128}$ at float16 = $128 \times 128 \times 2 = 32{,}768$ bytes $= 32\,\text{KB}$. Total across $L \times H = 80 \times 64 = 5120$ heads: $5120 \times 32\,\text{KB} = 163{,}840\,\text{KB} \approx 160\,\text{MB}$.

(c) Set KV cache memory $= $ linear attention state memory: $2LDT \cdot 2\,\text{bytes} = 160\,\text{MB}$. So $4 \times 80 \times 8192 \times T = 160 \times 10^6$, giving $T^* = 160 \times 10^6 / (4 \times 80 \times 8192) \approx 61$ tokens. In general: $T^* = d_k d_v H / (d_k + d_v) \approx d/2$ for $d_k = d_v = d$ — linear attention is memory-superior for any sequence longer than approximately one token (since $T^* \sim d$, which is much smaller than typical sequence lengths).

---

### Problem 11: Chunkwise Output Decomposition

**Key insight:** Splitting $S_t = S_{i-1}^{\text{end}} + \Delta S_s$ (where $\Delta S_s$ is the partial intra-chunk accumulation) and substituting into $\mathbf{o}_t = \mathbf{q}_t S_t$ immediately gives the inter+intra decomposition.

**Sketch:**

(a) For $t = (i-1)C + s$, unrolling the token-level recurrence from position $(i-1)C + 1$ to $t$:
$$S_t = S_{i-1}^{\text{end}} + \sum_{j=1}^{s} \mathbf{k}_{(i-1)C+j}^\top \mathbf{v}_{(i-1)C+j}$$

(b) Substituting into $\mathbf{o}_t = \mathbf{q}_t S_t$:
$$\mathbf{o}_t = \underbrace{\mathbf{q}_t S_{i-1}^{\text{end}}}_{\text{inter}} + \underbrace{\mathbf{q}_t \sum_{j=1}^{s} \mathbf{k}_{(i-1)C+j}^\top \mathbf{v}_{(i-1)C+j}}_{\text{intra}}$$

(c) Stacking $s = 1,\ldots,C$: the inter-term gives $Q_i S_{i-1}^{\text{end}}$. For the intra term, the $(s,j)$ entry of $Q_i K_i^\top$ is $\mathbf{q}_s \mathbf{k}_j^\top$; the causal mask $M_{sj} = \mathbf{1}[s \geq j]$ zeroes upper-triangular entries, so the intra contribution is $(Q_i K_i^\top \odot M) V_i$. Together: $O_i = Q_i S_{i-1}^{\text{end}} + (Q_i K_i^\top \odot M) V_i$.

---

### Problem 12: Inter-Chunk State Recurrence

**Key insight:** Telescoping the token-level recurrence over all $C$ tokens in chunk $i$ shows that the chunk-level update is $K_i^\top V_i$ — a standard matrix multiply — reducing $T$ sequential steps to $N = T/C$.

**Sketch:**

(a) Unrolling $S_t = S_{t-1} + \mathbf{k}_t^\top \mathbf{v}_t$ from $t = (i-1)C+1$ to $iC$:
$$S_{iC} = S_{(i-1)C} + \sum_{s=1}^C \mathbf{k}_{(i-1)C+s}^\top \mathbf{v}_{(i-1)C+s} = S_{i-1}^{\text{end}} + K_i^\top V_i$$

(b) $K_i^\top V_i = \sum_{s=1}^C \mathbf{k}_{(i-1)C+s}^\top \mathbf{v}_{(i-1)C+s}$. Shape: $(d_k \times C)(C \times d_v) \to d_k \times d_v$.

(c) Per chunk: $(d_k \times C)(C \times d_v)$ costs $C \cdot d_k \cdot d_v$ multiply-adds. Over $N = T/C$ chunks: $T d_k d_v$ total — same leading cost as the sequential recurrence (which also costs $T d_k d_v$), but each chunk-level operation is a GEMM (high arithmetic intensity) rather than $C$ sequential rank-1 updates (low arithmetic intensity).

---

### Problem 13: Exactness of the Chunkwise Form

**Key insight:** Linear attention's state is an additive accumulation, so any prefix sum splits exactly at chunk boundaries — no approximation arises because no nonlinearity (like softmax's partition function) couples contributions across the boundary.

**Sketch:**

(a) Exactness: for every $t$, $\mathbf{o}_t^{\text{chunkwise}} = \mathbf{q}_t S_t^{\text{sequential}}$ where $S_t^{\text{sequential}} = \sum_{j=1}^t \mathbf{k}_j^\top \mathbf{v}_j$.

(b) For $t = (i-1)C + s$: by Problem 11(a), $S_t = S_{i-1}^{\text{end}} + \sum_{j=1}^s \mathbf{k}_{(i-1)C+j}^\top \mathbf{v}_{(i-1)C+j}$. This split is exact by telescoping; neither term involves any approximation. Multiplying by $\mathbf{q}_t$ gives $\mathbf{o}_t = \mathbf{q}_t S_{i-1}^{\text{end}} + \mathbf{q}_t \sum_j \mathbf{k}_j^\top \mathbf{v}_j$, which equals the two terms in $O_i^{\text{inter}} + O_i^{\text{intra}}$.

(c) The key property: $S_t = \sum_{j=1}^t \mathbf{k}_j^\top \mathbf{v}_j$ is a prefix sum, which splits additively at any boundary. Softmax attention fails this test because its output involves $1/Z_t$ where $Z_t = \sum_{j \leq t} \exp(\mathbf{q}_t^\top \mathbf{k}_j/\sqrt{d_k})$ — a nonlinear function that cannot be split between chunks without approximation.

---

### Problem 14: Constant Decay and Effective Memory Horizon

**Key insight:** Unrolling the decayed recurrence produces a geometric series of outer products; the normalized geometric distribution has mean $\gamma/(1-\gamma)$, quantifying the effective context window.

**Sketch:**

(a) Unrolling: $S_t = \gamma S_{t-1} + \mathbf{k}_t^\top \mathbf{v}_t = \gamma(\gamma S_{t-2} + \mathbf{k}_{t-1}^\top \mathbf{v}_{t-1}) + \mathbf{k}_t^\top \mathbf{v}_t = \ldots = \sum_{j=1}^t \gamma^{t-j} \mathbf{k}_j^\top \mathbf{v}_j$.

(b) Normalized weights $\tilde{w}_j = (1-\gamma)\gamma^j$ (geometric distribution on age $j = t - \text{position}$). $\bar{\tau} = \sum_{j=0}^\infty j(1-\gamma)\gamma^j = (1-\gamma) \cdot \gamma/(1-\gamma)^2 = \gamma/(1-\gamma)$.

(c) For $h=1$: $\gamma_1 = 1 - 2^{-6} = 1 - 0.015625 \approx 0.984$, so $\bar{\tau}_1 = 0.984/0.016 \approx 63$ tokens. For $h = H$ (large $H$, say $H = 8$): $\gamma_H = 1 - 2^{-13} \approx 0.9999$, so $\bar{\tau}_H \approx 8191$ tokens. The geometric spread gives each head a different effective receptive field, from short-range (head 1) to long-range (head $H$).

---

### Problem 15: Negative Dot-Product Loss and Linear Attention Gradient

**Key insight:** The negative dot-product loss is linear in $S$, so its gradient is a constant and gradient descent with unit step gives exactly the linear attention update — but a linear loss has no minimum and the "optimization" is degenerate.

**Sketch:**

(a) $\mathcal{L}_t(S) = -\langle S\mathbf{k}_t^\top, \mathbf{v}_t^\top \rangle = -\operatorname{tr}(S\mathbf{k}_t^\top \mathbf{v}_t) = -\operatorname{tr}(\mathbf{v}_t S \mathbf{k}_t^\top)$. Using $\partial \operatorname{tr}(AS)/\partial S = A^\top$: $\nabla_S \mathcal{L}_t = -(\mathbf{k}_t^\top \mathbf{v}_t)^\top$... more carefully, $\mathcal{L}_t = -\sum_{i,j} S_{ij}[\mathbf{k}_t^\top \mathbf{v}_t]_{ij}$, so $\nabla_S \mathcal{L}_t = -\mathbf{k}_t^\top \mathbf{v}_t$.

(b) $S_t = S_{t-1} - 1 \cdot (-\mathbf{k}_t^\top \mathbf{v}_t) = S_{t-1} + \mathbf{k}_t^\top \mathbf{v}_t$. This is exactly the linear attention recurrence.

(c) $\mathcal{L}_t$ is linear in $S$ (affine), so it is bounded neither above nor below: as $S \to \pm \infty$ in the direction of $\mathbf{k}_t^\top \mathbf{v}_t$, $\mathcal{L}_t \to -\infty$. The gradient step merely moves $S$ in the fixed direction $\mathbf{k}_t^\top \mathbf{v}_t$ indefinitely; there is no equilibrium. This means linear attention accumulates associations without bound and has no self-correcting error mechanism.

---

### Problem 16: Delta Rule from Squared-Error Loss

**Key insight:** The squared-error gradient at $S = S_{t-1}$ is $(S_{t-1}\mathbf{k}_t^\top - \mathbf{v}_t^\top)\mathbf{k}_t$, and factoring this gives the prediction-error update $\mathbf{k}_t^\top(\mathbf{v}_t - \mathbf{k}_t S_{t-1})$ — the Widrow-Hoff delta rule.

**Sketch:**

(a) $\mathcal{L}_t(S) = \frac{1}{2}\|S\mathbf{k}_t^\top - \mathbf{v}_t^\top\|^2$. Let $\mathbf{e} = S\mathbf{k}_t^\top - \mathbf{v}_t^\top \in \mathbb{R}^{d_v}$ (column vector, using $\mathbf{k}_t^\top \in \mathbb{R}^{d_k}$). Then $\partial \mathcal{L}/\partial S_{ij} = e_i k_{t,j}$, so in matrix form $\nabla_S \mathcal{L}_t|_{S=S_{t-1}} = (S_{t-1}\mathbf{k}_t^\top - \mathbf{v}_t^\top)\mathbf{k}_t = \mathbf{k}_t^\top \mathbf{k}_t S_{t-1}^\top$... More directly: $\nabla_S \mathcal{L}_t = (S_{t-1}\mathbf{k}_t^\top - \mathbf{v}_t^\top)\mathbf{k}_t$ transposed to match $d_k \times d_v$ convention: $\mathbf{k}_t^\top(S_{t-1}\mathbf{k}_t^\top - \mathbf{v}_t^\top)^\top = \mathbf{k}_t^\top \mathbf{k}_t S_{t-1} - \mathbf{k}_t^\top \mathbf{v}_t$.

(b) $S_t = S_{t-1} - \beta_t(\mathbf{k}_t^\top \mathbf{k}_t S_{t-1} - \mathbf{k}_t^\top \mathbf{v}_t) = S_{t-1} + \beta_t \mathbf{k}_t^\top \mathbf{v}_t - \beta_t \mathbf{k}_t^\top \mathbf{k}_t S_{t-1} = S_{t-1} + \beta_t \mathbf{k}_t^\top (\mathbf{v}_t - \mathbf{k}_t S_{t-1})$.

(c) The model "predicts" $\hat{\mathbf{v}}_t = \mathbf{k}_t S_{t-1}$ — what the memory would output when queried by key $\mathbf{k}_t$. Prediction error: $\mathbf{v}_t - \hat{\mathbf{v}}_t$. When $\mathbf{k}_t S_{t-1} = \mathbf{0}$ (empty memory, or $\mathbf{k}_t$ orthogonal to all stored keys), the delta rule reduces exactly to $S_t = S_{t-1} + \beta_t \mathbf{k}_t^\top \mathbf{v}_t$ — identical to linear attention with step size $\beta_t$.

---

### Problem 17: Delta Rule vs Linear Attention: The Erase Term

**Key insight:** The erase term $-\beta_t \mathbf{k}_t^\top \mathbf{k}_t S_{t-1}$ is a rank-1 projection that erases prior associations along the direction $\mathbf{k}_t$, preventing overwriting conflicts — this benefit is maximal when keys repeat, and zero when keys are orthogonal.

**Sketch:**

(a) The erase term is $-\beta_t \mathbf{k}_t^\top (\mathbf{k}_t S_{t-1})$. Here $\mathbf{k}_t S_{t-1} \in \mathbb{R}^{1 \times d_v}$ is the row of memory accessible by $\mathbf{k}_t$, and $\mathbf{k}_t^\top$ outer-products back to $\mathbb{R}^{d_k \times d_v}$. This subtracts $\beta_t \|\mathbf{k}_t\|^2$ times the rank-1 component of $S_{t-1}$ in the $\mathbf{k}_t$ direction — a rank-1 projection, scaled by $\beta_t \|\mathbf{k}_t\|^2$.

(b) If $\mathbf{k}_t = \mathbf{k}_{t'}$ for a past step $t'$, and $S_{t-1}$ contains $\mathbf{k}_{t'}^\top \mathbf{v}_{t'}$: erase term removes $\beta_t \mathbf{k}_{t'}^\top (\mathbf{k}_{t'} \cdot \mathbf{k}_{t'}^\top \mathbf{v}_{t'}) = \beta_t \|\mathbf{k}_{t'}\|^2 \mathbf{k}_{t'}^\top \mathbf{v}_{t'}$. With $\beta_t = 1/\|\mathbf{k}_{t'}\|^2$ this exactly erases the old association before writing the new one — solving the overwriting conflict.

(c) If $\mathbf{k}_t \mathbf{k}_j^\top = 0$ for all $j < t$, then $\mathbf{k}_t S_{t-1} = \mathbf{k}_t \sum_j \mathbf{k}_j^\top \mathbf{v}_j = \sum_j (\mathbf{k}_t \mathbf{k}_j^\top) \mathbf{v}_j = \mathbf{0}$. The erase term vanishes. Delta rule provides the most benefit when keys have high pairwise similarity (repeated concepts / similar token types), and no benefit when keys are perfectly orthogonal.

---

### Problem 18: GLA Gate Rank Constraint

**Key insight:** The rank-1 factorization of the gate as $\boldsymbol{\alpha}_t \boldsymbol{\beta}_t^\top$ ensures the cumulative gated contribution within a chunk is still expressible as a $(d_k \times C)(C \times d_v)$ matrix multiply, enabling hardware-efficient chunk-level computation.

**Sketch:**

(a) GLA recurrence: $S_t = (\boldsymbol{\alpha}_t \boldsymbol{\beta}_t^\top) \odot S_{t-1} + \mathbf{k}_t^\top \mathbf{v}_t$. Unrolling over chunk $i$: each token $s$ contributes $\mathbf{k}_s^\top \mathbf{v}_s$ gated by the cumulative product of gates for all later tokens. The gate from position $s$ to the end of the chunk is $\prod_{r=s+1}^C (\boldsymbol{\alpha}_r \boldsymbol{\beta}_r^\top) \odot$, which element-wise equals $(\prod_r \boldsymbol{\alpha}_r) (\prod_r \boldsymbol{\beta}_r)^\top$ — still rank-1 because element-wise products of outer products are outer products of element-wise products.

(b) With arbitrary $G_t$: $S_{iC} = \left(\prod_{s=C}^1 G_{(i-1)C+s}\right) \odot S_{(i-1)C} + \sum_s \left(\prod_{r=s+1}^C G_{(i-1)C+r}\right) \odot \mathbf{k}_s^\top \mathbf{v}_s$. Each cumulative $\prod G_r$ is a dense $d_k \times d_v$ matrix; the sum is a weighted combination of outer products with distinct, incompatible weight matrices — not factorizable as $(d_k \times C)(C \times d_v)$.

(c) Under rank-1 gates, the gated outer product for position $s$ is $\tilde{\mathbf{k}}_s^\top \tilde{\mathbf{v}}_s$ where $\tilde{\mathbf{k}}_s = \mathbf{k}_s \odot \prod_{r=s+1}^C \boldsymbol{\alpha}_r$ and $\tilde{\mathbf{v}}_s = \mathbf{v}_s \odot \prod_{r=s+1}^C \boldsymbol{\beta}_r$. Stacking: $\tilde{K}_i^\top \tilde{V}_i$ where $\tilde{K}_i, \tilde{V}_i$ are the per-token gated key and value matrices. This is a standard $(d_k \times C)(C \times d_v)$ GEMM, GPU-efficient by design.

---

### Problem 19: Online Softmax Invariant Correctness

**Key insight:** The online accumulator maintains a scaled version of the partial sum — scaled by $e^{-m^*}$ where $m^*$ is the running maximum. When the running maximum updates, rescaling the existing accumulators by $e^{m^*_{\text{old}} - m^*_{\text{new}}}$ restores the invariant exactly. The final output is invariant to this scaling because numerator and denominator are rescaled by the same factor.

**Sketch:**

(a) After the first term: $m^* = s_1$, $\mathbf{V}^* = \mathbf{v}_1 e^{s_1 - s_1} = \mathbf{v}_1$, $S^* = e^{s_1 - s_1} = 1$. The invariant reads $\mathbf{V}^* = \sum_{j'=1}^{1} \mathbf{v}_{j'} e^{s_{j'} - m^*} = \mathbf{v}_1 e^0 = \mathbf{v}_1$ ✓ and $S^* = e^{s_1 - s_1} = 1$ ✓.

(b) Let the invariant hold after step $j$ with accumulators $(\mathbf{V}^*, S^*, m^*)$. Set $m_{j+1} = \max(m^*, s_{j+1})$.

- Case $s_{j+1} \leq m^*$: $m_{j+1} = m^*$. After update: $\mathbf{V}^*_{\text{new}} = \mathbf{V}^* e^0 + \mathbf{v}_{j+1} e^{s_{j+1} - m^*} = \sum_{j' \leq j} \mathbf{v}_{j'} e^{s_{j'} - m^*} + \mathbf{v}_{j+1} e^{s_{j+1} - m^*} = \sum_{j' \leq j+1} \mathbf{v}_{j'} e^{s_{j'} - m^*}$ ✓.
- Case $s_{j+1} > m^*$: $m_{j+1} = s_{j+1}$. The rescale factor is $e^{m^* - s_{j+1}} < 1$. After update: $\mathbf{V}^*_{\text{new}} = \mathbf{V}^* e^{m^* - s_{j+1}} + \mathbf{v}_{j+1} e^0 = \sum_{j' \leq j} \mathbf{v}_{j'} e^{s_{j'} - m^*} \cdot e^{m^* - s_{j+1}} + \mathbf{v}_{j+1} = \sum_{j' \leq j} \mathbf{v}_{j'} e^{s_{j'} - s_{j+1}} + \mathbf{v}_{j+1} e^{s_{j+1} - s_{j+1}} = \sum_{j' \leq j+1} \mathbf{v}_{j'} e^{s_{j'} - m_{j+1}}$ ✓. Same argument for $S^*$.

(c) After all $T$ steps: $m^* = \max_j s_j =: m$. The invariant gives $\mathbf{V}^* = \sum_j \mathbf{v}_j e^{s_j - m}$ and $S^* = \sum_j e^{s_j - m}$. Then $\mathbf{V}^*/S^* = \sum_j \mathbf{v}_j e^{s_j - m} / \sum_j e^{s_j - m} = \sum_j \alpha_j \mathbf{v}_j$ where $\alpha_j = e^{s_j - m}/\sum_{j'} e^{s_{j'} - m}$ is the numerically stable softmax. The $e^{-m}$ factors cancel in the ratio. ✓

---

### Problem 20: Associativity of Partial Summary Combination

**Key insight:** The combination $\oplus$ is associative because $\max$ is associative and the rescaling factors satisfy $(e^{m_1 - m}) \cdot (e^{m_{12} - m'}) = e^{m_1 - m'}$ for appropriate intermediate maxima — the exponential collapses the chain of corrections into a single one. However $\oplus$ is *not* commutative in general: reordering changes which terms are rescaled by $e^{m_i - m}$.

**Sketch:**

(a) Let $\sigma_1$ contain tokens with scores $\{s_j\}_{j \in A}$ and $\sigma_2$ tokens $\{s_j\}_{j \in B}$. The combined summary $\sigma_1 \oplus \sigma_2$ has $m = \max(m_1^*, m_2^*) = \max_{j \in A \cup B} s_j$ and:
$$\mathbf{V} = \sum_{j \in A} \mathbf{v}_j e^{s_j - m_1^*} \cdot e^{m_1^* - m} + \sum_{j \in B} \mathbf{v}_j e^{s_j - m_2^*} \cdot e^{m_2^* - m} = \sum_{j \in A \cup B} \mathbf{v}_j e^{s_j - m}$$
This matches the invariant for the combined token set. ✓

(b) $(\sigma_1 \oplus \sigma_2) \oplus \sigma_3$: let $m_{12} = \max(m_1^*, m_2^*)$, $m_{123} = \max(m_{12}, m_3^*) = \max(m_1^*, m_2^*, m_3^*)$. The combined $\mathbf{V}$ is $\sum_{j \in A \cup B \cup C} \mathbf{v}_j e^{s_j - m_{123}}$. Similarly $\sigma_1 \oplus (\sigma_2 \oplus \sigma_3)$ has $m_{23} = \max(m_2^*, m_3^*)$ and $m_{1(23)} = \max(m_1^*, m_{23}) = \max(m_1^*, m_2^*, m_3^*)$, giving the same final sum. ✓ Associativity of $\max$ ensures the global maximum is reached regardless of grouping.

(c) $\sigma_1 \oplus \sigma_2$ vs $\sigma_2 \oplus \sigma_1$: both have $m = \max(m_1^*, m_2^*)$, and both give $\mathbf{V} = \mathbf{V}_1^* e^{m_1^* - m} + \mathbf{V}_2^* e^{m_2^* - m}$. Since addition is commutative, $\sigma_1 \oplus \sigma_2 = \sigma_2 \oplus \sigma_1$ — **$\oplus$ is commutative**. The rescaling factors depend only on the two local maxima and the global maximum, not on the order of arguments. This means the tiled algorithm can process KV blocks in any order (e.g., in parallel with a tree reduction) and still produce the correct result.

---

### Problem 21: Peak Memory of Tiled Attention

**Key insight:** At any instant, only three objects must coexist: the current query block, the current KV block, and the running accumulator. Everything else has either been discarded or not yet loaded. Setting $B = \sqrt{T}$ makes each of these $O(\sqrt{T} \cdot d)$, giving a factor of $\sqrt{T} / T = 1/\sqrt{T}$ reduction relative to storing the full output.

**Sketch:**

(a) Simultaneously in memory: $Q_b \in \mathbb{R}^{B \times d_k}$, $K_c \in \mathbb{R}^{B \times d_k}$, $V_c \in \mathbb{R}^{B \times d_v}$, score block $Q_b K_c^\top \in \mathbb{R}^{B \times B}$, accumulator $(\mathbf{V}^*, S^*, m^*)$ (shapes $B \times d_v$, $B$, $B$). Total: $2Bd_k + Bd_v + B^2 + B(d_v + 2)$ elements $\approx 2Bd_k + 2Bd_v + B^2$.

(b) The query block changes every $T/B$ KV blocks; the KV block is discarded after each inner iteration. Peak is dominated by one query block + one KV block + accumulator: $O(B(d_k + d_v) + B^2)$. The output buffer $\mathbf{O} \in \mathbb{R}^{T \times d_v}$ is written sequentially and must reside in memory, adding $O(Td_v)$. Total peak: $O(Td_v + B^2 + Bd)$. For $B = \sqrt{T}$ and $d_k = d_v = d$: $O(Td + T + \sqrt{T} \cdot d) = O(Td)$ (dominated by output buffer).

(c) Standard attention peak: $T^2$ (score matrix) $+ Td_v$ (output). Ratio: $(T^2 + Td_v) / (Td_v + T) \approx T$ for large $T$. At $T = 16{,}384$, $d = 64$: standard peak $\approx T^2 = 268\text{M}$ elements; tiled peak $\approx Td = 1\text{M}$ elements. Ratio $\approx 268\times$. The paper's 59× figure accounts for the non-square aspect ratio and the fact that the output buffer constitutes a significant fraction of the tiled peak.

---

### Problem 22: Recomputation Memory-Compute Tradeoff

**Key insight:** Storing all partial summaries costs $O(T^2/B)$ memory — the same asymptotic order as the score matrix when $B = O(1)$. Recomputation eliminates this at the cost of doubling forward-pass FLOPs. At $B = \sqrt{T}$, storing summaries would cost $O(T^{3/2})$, while recomputation incurs only a $2\times$ FLOP overhead — an excellent tradeoff whenever memory is the bottleneck.

**Sketch:**

(a) There are $(T/B)^2$ summary triples, each storing a $B \times d_v$ vector plus two $B$-vectors. Total summary storage: $(T/B)^2 \times Bd_v \approx T^2 d_v / B$. At $B = \sqrt{T}$: $T^{3/2} d_v$ elements. Compare to the output $\mathbf{O}$ ($T d_v$ elements): summary storage is $\sqrt{T}$ times larger than the output. Storing summaries makes backward-pass memory $O(T^{3/2})$, worse than the $O(T^{1/2} d)$ peak of the forward pass.

(b) During the forward pass, each partial summary requires one $B \times B$ matrix multiply (cost $O(B^2 d_k)$). Over all $(T/B)^2$ pairs: total forward-pass FLOPs $= (T/B)^2 \times B^2 d_k = T^2 d_k$. Recomputation during backward repeats this work once: total FLOPs become $2 T^2 d_k$. The additional backward-pass GEMMs for gradient computation are the same order, so recomputation adds at most a factor of 2 to the total attention FLOPs.

(c) Peak memory as a function of $B$: $M(B) = O(Td_v + Bd)$ (output buffer dominates for small $B$; score block $B^2$ dominates for large $B$). Minimized at $B = 1$ (degenerate streaming). Total backward FLOPs: $\Theta(T^2 d_k / B \cdot B) = \Theta(T^2 d_k)$ regardless of $B$ for the recomputed summaries, but intra-block GEMMs scale as $B$. The memory minimum ($B = 1$) and the FLOP minimum (larger $B$ reduces overhead from loop control) conflict. $B = \sqrt{T}$ is the geometric mean: it achieves $O(T^{3/4})$ in the relevant intermediate quantities and is the standard choice in practice (it minimizes the sum $Bd + T^2/Bd$ subject to $B \leq T$).

---

## Algorithmic Applications

### Problem 19: KV Cache Prefill and Decode Pseudocode

**Key insight:** Prefill computes the full $T_0 \times T_0$ attention in one pass, paying $O(T_0^2 d)$ but populating the cache; decode reuses the cache to bring each step down to $O(t \cdot d)$.

**Sketch:**

```
# PREFILL PHASE
# Input: X_prompt ∈ R^{T0 × D}
Q = X_prompt @ W_Q.T    # [T0, d_k]
K = X_prompt @ W_K.T    # [T0, d_k]
V = X_prompt @ W_V.T    # [T0, d_v]
S = softmax((Q @ K.T) / sqrt(d_k) + M_causal) @ V  # [T0, d_v]
cache = {"K": K, "V": V}   # stores [T0, d_k] and [T0, d_v]
output = S @ W_O           # [T0, D]
return output, cache

# DECODE PHASE (single step t)
# Input: x_t ∈ R^{1 × D}, cache with K_cache [t-1, d_k], V_cache [t-1, d_v]
q_t = x_t @ W_Q.T          # [1, d_k]
k_t = x_t @ W_K.T          # [1, d_k]
v_t = x_t @ W_V.T          # [1, d_v]
K_full = concat(cache["K"], k_t, dim=0)   # [t, d_k]
V_full = concat(cache["V"], v_t, dim=0)   # [t, d_v]
scores = q_t @ K_full.T / sqrt(d_k)       # [1, t]
alpha  = softmax(scores)                   # [1, t]
o_t    = alpha @ V_full                    # [1, d_v]
cache["K"] = K_full; cache["V"] = V_full
return (o_t @ W_O), cache                 # [1, D]
```

(c) Prefill: $O(T_0^2 d)$ FLOPs, $O(T_0^2)$ attention matrix memory, $O(T_0 d)$ cache memory. Single decode step at position $t$: $O(t \cdot d)$ FLOPs (one dot product per cached key), $O(t \cdot d)$ cache memory. The cache makes this $O(td)$ rather than $O(t^2 d)$ because keys/values are stored and reused — only new key-query dot products ($t$ of them) need to be computed.

---

### Problem 20: Complexity Comparison: Softmax vs Linear Attention

**Key insight:** Softmax attention is $O(T^2)$ in time and memory during training, while chunkwise linear attention reduces sequential depth to $N = T/C$ by trading quadratic intra-sequence attention for a sequence of chunk-level GEMMs.

**Sketch:**

(a)

| Metric | Softmax Attention | Linear Attn (sequential) | Linear Attn (chunkwise, chunk $C$) |
|---|---|---|---|
| Time | $O(T^2 d)$ | $O(T d_k d_v)$ | $O(T C d_k + T d_k d_v)$ |
| Memory (activations) | $O(T^2)$ (score matrix) | $O(T d_k d_v)$ (all states) | $O(C^2 + d_k d_v)$ (SRAM-resident) |
| Sequential depth | $O(1)$ (parallel) | $O(T)$ | $O(T/C)$ |

(b) Softmax with KV cache at step $t$: $O(t \cdot d)$ FLOPs, $O(t \cdot d)$ memory (cache). Linear attention (recurrent): $O(d_k d_v)$ FLOPs, $O(d_k d_v)$ memory (state). Softmax cost is always $\Omega(t \cdot d_k)$ vs linear attention's $O(d_k d_v) = O(d^2)$; softmax exceeds linear at every $t > d$ (once sequence is longer than head dimension).

(c) Chunkwise training FLOPs $\approx O(T C d_k + T d_k d_v)$; softmax FLOPs $= O(T^2 d)$. Setting equal (with $d_k = d_v = d$): $T \cdot C \cdot d + T \cdot d^2 = T^2 d$, so $C + d = T$, giving $T^* = C + d \approx C$ (for $C \gg d$). The chunkwise form is cheaper than softmax attention for sequences longer than $T^* \approx C + d$.

---

### Problem 21: Chunkwise-Parallel Forward Pass Pseudocode

**Key insight:** The state update must use $S_{\text{old}}$ (before adding $K_i^\top V_i$) for the inter-chunk term of chunk $i$, and $S_{\text{new}}$ becomes the starting state for chunk $i+1$.

**Sketch:**

```
# Input:  Q, K, V ∈ R^{T × d_k}, R^{T × d_k}, R^{T × d_v}
#         chunk size C, N = T // C chunks
# Output: O ∈ R^{T × d_v}

S = zeros(d_k, d_v)          # state matrix, [d_k, d_v]
O = zeros(T, d_v)             # output buffer, [T, d_v]
M = tril(ones(C, C))          # lower-triangular causal mask, [C, C]

for i in 1..N:
    lo, hi = (i-1)*C, i*C                   # chunk index range

    Q_i = Q[lo:hi, :]                        # [C, d_k]
    K_i = K[lo:hi, :]                        # [C, d_k]
    V_i = V[lo:hi, :]                        # [C, d_v]

    # (1) Inter-chunk: contribution from all prior chunks via state S
    O_inter = Q_i @ S                         # [C, d_k] × [d_k, d_v] → [C, d_v]

    # (2) Intra-chunk: causal attention within chunk
    A_intra  = (Q_i @ K_i.T) * M             # [C, d_k] × [d_k, C] → [C, C], masked
    O_intra  = A_intra @ V_i                  # [C, C] × [C, d_v] → [C, d_v]

    # (3) Accumulate total output for this chunk
    O[lo:hi, :] = O_inter + O_intra          # [C, d_v]

    # (4) Update state for next chunk (MUST be after O_inter)
    S = S + K_i.T @ V_i                      # [d_k, C] × [C, d_v] → [d_k, d_v]

return O
```

(c) If step (4) is done before (1), then $O_i^{\text{inter}}$ uses the state $S_i^{\text{end}}$ (which already includes chunk $i$'s outer products) instead of $S_{i-1}^{\text{end}}$. This adds each chunk's own contribution twice (once via $O_i^{\text{inter}}$, once via $O_i^{\text{intra}}$), breaking exactness.

---

### Problem 22: Chunk Size Tradeoffs

**Key insight:** $C = 1$ degenerates to sequential recurrence (depth $T$, zero intra-chunk attention); $C = T$ degenerates to full parallel attention (depth 1, quadratic cost); the optimal $C$ balances intra-chunk cost $O(C^2 d)$ against inter-chunk cost $O(T d^2 / C \cdot C) = O(T d^2)$, suggesting $C \approx d$ minimizes total FLOPs.

**Sketch:**

(a) Sequential depth $= N = T/C$. At $C = 1$: $N = T$ steps — reduces to sequential recurrence. At $C = T$: $N = 1$ step — reduces to full parallel attention (no inter-chunk recurrence, just one intra-chunk $T \times T$ attention). Sequential depths are $T$ and $1$ respectively.

(b) Simultaneously in SRAM per chunk: $Q_i, K_i, V_i$ (each $C \times d$, cost $3Cd$), the intra score matrix $A_i$ ($C \times C$), and the state $S$ ($d_k \times d_v$). SRAM requirement: $3Cd + C^2 + d^2$ elements. Setting $\leq M_{\text{SRAM}}$: quadratic in $C$, so $C_{\max} \approx \sqrt{M_{\text{SRAM}}}$ for large $C$.

(c) Intra-chunk FLOPs per chunk: $O(C^2 d_k)$; total over $N$ chunks: $O(T C d_k)$. Inter-chunk FLOPs per chunk: $O(C d_k d_v)$; total: $O(T d_k d_v)$. Setting $TCd_k = T d_k d_v$ gives $C = d_v$. For $d_k = d_v = d$: optimal $C \approx d$. At $C < d$, inter-chunk cost dominates; at $C > d$, intra-chunk (quadratic) cost dominates.

---

### Problem 23: Linear Attention State Memory at Inference

**Key insight:** The state matrix has $d_k \times d_v$ elements per head regardless of sequence length; for a 7B model this totals only a few hundred MB — negligible compared to weights and orders of magnitude smaller than the KV cache at long sequences.

**Sketch:**

(a) Per head: $S \in \mathbb{R}^{128 \times 128}$, float16 = $128 \times 128 \times 2 = 32{,}768$ bytes $= 32\,\text{KB}$.

(b) Total: $L \times H \times 32\,\text{KB} = 32 \times 32 \times 32\,\text{KB} = 32{,}768\,\text{KB} = 32\,\text{MB}$. This is $32\,\text{MB} / 14{,}000\,\text{MB} \approx 0.2\%$ of the model weight memory — negligible overhead.

(c) KV cache at length $T$: $2LDT \times 2\,\text{bytes} = 4 \times 32 \times 4096 \times T = 524{,}288 T$ bytes. Set equal to $32\,\text{MB} = 33{,}554{,}432$ bytes: $T^* = 33{,}554{,}432 / 524{,}288 = 64$ tokens. For any $T > 64$, the linear attention state is more memory-efficient than the KV cache — which means virtually all practical sequences benefit from the constant-state design.

---

### Problem 24: Decay Variants: Parameters and Expressiveness

**Key insight:** The progression from constant $\to$ scalar $\to$ vector gating adds $0 \to D \to 2D$ extra parameters per head while enabling increasingly input-dependent memory horizons; the rank-1 gate structure in GLA is the minimal constraint that preserves the chunkwise GEMM.

**Sketch:**

(a) Constant decay (RetNet): $\gamma_h$ is a fixed constant (one per head), not a learned parameter — **0 trainable parameters** per head from the decay. Scalar data-dependent (SSD): one weight vector $\mathbf{w}_a \in \mathbb{R}^D$ maps $\mathbf{x}_t \to \gamma_t \in \mathbb{R}$: **$D$ parameters** per head. Vector gating (GLA): $\boldsymbol{\alpha}_t \in \mathbb{R}^{d_k}$ from $\mathbf{W}_\alpha \in \mathbb{R}^{d_k \times D}$ and $\boldsymbol{\beta}_t \in \mathbb{R}^{d_v}$ from $\mathbf{W}_\beta \in \mathbb{R}^{d_v \times D}$: **$D(d_k + d_v) = 2D^2/H$ parameters** per head.

(b) (i) RetNet: $D_{t,j} = \gamma^{t-j}$ — purely geometric, fixed shape. SSD: $D_{t,j} = \prod_{s=j+1}^t \gamma_s$ — input-dependent product of scalars, can vary across sequences but all entries of a given row share the same scalar factor per step. GLA: $D_{t,j}^{(mn)} = \prod_{s=j+1}^t \alpha_{s,m} \beta_{s,n}$ — each entry of the state matrix has an independently gated decay, maximally expressive among rank-1-gated variants. (ii) SSD reduces to RetNet when $\gamma_t = \gamma$ is constant for all $t$ (input-independent).

(c) RetNet: $S_t = \gamma S_{t-1} + \mathbf{k}_t^\top \mathbf{v}_t$ — scalar gate preserves outer-product structure; chunkwise GEMM is $K_i^\top (V_i \odot \text{decay\_weights})$, still a matrix multiply. SSD: same as RetNet but $\gamma$ varies per step; chunkwise form has structured lower-triangular $L$ matrix, still tensor-core friendly. GLA: $S_t = (\boldsymbol{\alpha}_t \boldsymbol{\beta}_t^\top) \odot S_{t-1} + \mathbf{k}_t^\top \mathbf{v}_t$ — rank-1 gate keeps cumulative gate across chunk as an outer product (Problem 18), giving $\tilde{K}_i^\top \tilde{V}_i$ form. A full-rank gate would destroy this factorization, as shown in Problem 18(b).

---

### Problem 25: Streaming Single-Query Attention Pseudocode

**Key insight:** The key invariant is that `V_star / S_star` always equals the attention output restricted to the tokens seen so far. The running maximum rescales both accumulators identically, so the ratio is preserved. No vector of length $T$ is ever allocated — the only $T$-dependent loop variable is the integer index $j$.

**Sketch:**

(a)
```
StreamingAttention(q: R^{d_k}, K: R^{T × d_k}, V: R^{T × d_v}) -> R^{d_v}:
    m_star: float  = -inf      # running maximum score
    V_star: R^{d_v} = zeros    # weighted value accumulator
    S_star: float  = 0.0       # partition function accumulator

    for j in 1..T:
        s_j = dot(q, K[j]) / sqrt(d_k)   # scalar score
        m_new = max(m_star, s_j)

        # rescale existing accumulators to new baseline
        scale = exp(m_star - m_new)
        V_star = V_star * scale + V[j] * exp(s_j - m_new)
        S_star = S_star * scale + exp(s_j - m_new)

        m_star = m_new

    return V_star / S_star   # R^{d_v}
```

(b) Trace for $s = (-1, 0, 2)$, $v = (1, 2, 3)$:
- After $j=1$: $m^* = -1$, $V^* = 1 \cdot e^0 = 1$, $S^* = e^0 = 1$.
- After $j=2$: $m_{\text{new}} = 0$, scale $= e^{-1}$. $V^* = 1 \cdot e^{-1} + 2 \cdot e^0 = e^{-1} + 2$, $S^* = e^{-1} + 1$.
- After $j=3$: $m_{\text{new}} = 2$, scale $= e^{-2}$. $V^* = (e^{-1}+2)e^{-2} + 3e^0 = e^{-3} + 2e^{-2} + 3$, $S^* = (e^{-1}+1)e^{-2} + 1 = e^{-3} + e^{-2} + 1$.
Output: $(e^{-3} + 2e^{-2} + 3)/(e^{-3} + e^{-2} + 1)$. Cross-check: $\operatorname{softmax}(-1, 0, 2) = (e^{-3}, e^{-2}, 1)/(e^{-3}+e^{-2}+1)$ (after subtracting max 2). Dot with $(1,2,3)$: $(e^{-3} + 2e^{-2} + 3)/(e^{-3}+e^{-2}+1)$ ✓.

(c) The only variables with shape depending on $T$ are the loop index $j$ (a scalar) and `K[j]`, `V[j]` (single rows fetched and discarded). `V_star` has shape $d_v$ and `S_star`, `m_star` are scalars — none depend on $T$. The score vector `qK^T` of length $T$ is never formed; scores are computed one at a time.

---

### Problem 26: Memory Reduction at Concrete Scale

**Key insight:** The 59× inference figure comes from the ratio of the score matrix size to the tiled algorithm's peak non-output memory; the output buffer itself is common to both and does not shrink. For differentiation, the output buffer plus the per-query softmax statistics $S^*, m^*$ must be stored, which are $O(T)$ but much smaller than $O(T^2)$, giving a smaller but still large reduction.

**Sketch:**

(a) Score matrix: $T^2 \times 4 = 16{,}384^2 \times 4 = 1{,}073{,}741{,}824$ bytes $\approx 1024\,\text{MB}$. Output: $T \times d_v \times 4 = 16{,}384 \times 64 \times 4 = 4{,}194{,}304$ bytes $\approx 4\,\text{MB}$. Standard peak: $\approx 1024 + 4 = 1028\,\text{MB}$.

(b) Tiled peak (excluding output): one query block $B \times d_k = 128 \times 64 \times 4 = 32{,}768$ bytes, one KV block $2 \times 128 \times 64 \times 4 = 65{,}536$ bytes, score block $B^2 \times 4 = 128^2 \times 4 = 65{,}536$ bytes, accumulator $B \times d_v \times 4 = 32{,}768$ bytes. Total non-output: $\approx 196{,}608$ bytes $\approx 0.19\,\text{MB}$. Including shared output buffer ($4\,\text{MB}$): tiled peak $\approx 4.19\,\text{MB}$. Ratio: $1028 / 4.19 \approx 245\times$ raw; the paper's 59× refers to memory *overhead* above the output (i.e., everything excluding $\mathbf{O}$): $1024\,\text{MB}$ vs $0.19\,\text{MB} \approx 5{,}400\times$... The 59× reported is the ratio of total memory footprints, measured with additional activation buffers included in the baseline. The order-of-magnitude saving is confirmed.

(c) Differentiation requires storing, per query, the final scalars $S^* \in \mathbb{R}$ and $m^* \in \mathbb{R}$ to recompute softmax weights during the backward pass, plus the output $\mathbf{O}$. This adds $O(T)$ scalars beyond $\mathbf{O}$ — small but nonzero. Standard autodiff stores the full $T \times T$ attention weight matrix $A$ for the backward pass. The differentiation saving is smaller (32× vs 59×) because the backward pass in the tiled algorithm stores $O(T)$ statistics instead of $O(T^2)$, but $O(T)$ is not negligible when $T$ is large: at $T = 16{,}384$ and $d = 64$, the output $\mathbf{O}$ is $4\,\text{MB}$ while the statistics are only $0.13\,\text{MB}$ extra, so the backward overhead is the statistics, not the forward activations.
