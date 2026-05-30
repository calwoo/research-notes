# Multi-Probe Zero Collision Hash (MPZCH): Mitigating Embedding Collisions and Enhancing Model Freshness in Large-Scale Recommenders

*Zhao et al., ByteDance, 2026 — [arXiv:2602.17050](https://arxiv.org/abs/2602.17050)*

| Dimension | Prior State | This Paper | Key Result |
|---|---|---|---|
| Collision rate | Sigrid Hash: 29.65% at 1.33× table capacity | MPZCH at P=256: 0.0000% | Zero collisions for 3B MAU production deployment |
| Stale embedding inheritance | Evicted slots reused with old weights | Evict + reset embedding + optimizer state | +0.83% impressions for videos posted within 48h |
| Lookup latency overhead | O(1) baseline | O(P) with P up to 512 | 0.80–0.90 ms — negligible vs. baseline |
| Embedding cluster quality | Intra-creator similarity 25% | Intra-creator similarity 38% | Better same-day video clustering (t-SNE) |

## Relations

**Builds on:** [[papers/collisionless-embeddings/monolith|Monolith (Liu et al., ByteDance, 2022)]]

---

## Table of Contents

- [[#1. Motivation Two Problems With Prior Collisionless Methods|1. Motivation: Two Problems With Prior Collisionless Methods]]
- [[#2. MPZCH The Algorithm|2. MPZCH: The Algorithm]]
  - [[#2.1 Data Structures|2.1 Data Structures]]
  - [[#2.2 Two-Pass Lookup|2.2 Two-Pass Lookup]]
  - [[#2.3 Eviction Policies|2.3 Eviction Policies]]
  - [[#2.4 Optimizer State Reset|2.4 Optimizer State Reset]]
- [[#3. System Integration|3. System Integration]]
  - [[#3.1 Sharding and CUDA Kernels|3.1 Sharding and CUDA Kernels]]
  - [[#3.2 Inference and Online Training|3.2 Inference and Online Training]]
- [[#4. Experiments|4. Experiments]]
  - [[#4.1 Collision Rate|4.1 Collision Rate]]
  - [[#4.2 Model Quality|4.2 Model Quality]]
  - [[#4.3 Latency|4.3 Latency]]
- [[#References|References]]

---

## 1. Motivation: Two Problems With Prior Collisionless Methods

[[papers/collisionless-embeddings/monolith|Monolith]] introduced cuckoo hashing for collisionless embedding tables, but it left two problems unresolved:

**Problem 1 — Remaining collisions in dynamic settings.** Cuckoo hashing is collision-free *at insertion time* — but only if the table has spare capacity. In a production system with billions of IDs and tight memory budgets, the table eventually fills to its load factor limit, and new IDs fall back to standard hashing (with collisions). Furthermore, Monolith's TTL-based eviction frees slots for reuse, but the reuse is uncontrolled: a new ID may collide with an existing one in the available slot pool.

**Problem 2 — Stale embedding inheritance.** When a TTL-expired slot is evicted and a new ID is assigned that slot, the new ID *inherits the previous occupant's embedding weights and optimizer state* (Adam moment estimates, etc.). This *negative transfer* is especially damaging for fresh content: a newly posted video that happens to reuse an old video's embedding slot starts its training with a strongly biased prior, causing erratic learning and poor early recommendations.

> [!INFO] Why stale inheritance matters for fresh content
> Recommendation systems measure success partly by their ability to surface *fresh* content (videos posted within 24–48 hours). A video with a stale-inherited embedding appears "already trained" to the model — it starts with a non-zero gradient history that reflects another video's interaction pattern. This suppresses the fresh video's ability to organically accumulate signal, harming the model's freshness capability.

MPZCH addresses both problems with a single mechanism: *multi-probe linear probing with active eviction and full slot reset*.

---

## 2. MPZCH: The Algorithm

### 2.1 Data Structures

MPZCH augments the standard embedding table with two auxiliary tensors:

- **Identities tensor** $\mathcal{I} \in \mathbb{Z}^N$: $\mathcal{I}[s]$ stores the ID currently occupying slot $s$. Initialized to $-1$ (empty).
- **Metadata tensor** $\mathcal{M} \in \mathbb{R}^N$: $\mathcal{M}[s]$ stores temporal information for slot $s$ — either a TTL timestamp (last-seen time) or an LRU counter.

The *probe range* for an ID $x$ with base hash $h = \text{hash}(x) \bmod N$ is the window $[h, h+P)$ (with wraparound), where $P$ is the *probe depth* hyperparameter.

### 2.2 Two-Pass Lookup

MPZCH uses a *read-before-write* two-pass strategy to handle each lookup/insertion:

**Pass 1 — Discovery (read-only):** Scan slots $h, h+1, \ldots, h+P-1$ without writing. Record:
- Whether $x$ already exists (found at some slot $s^*$)
- The best eviction candidate $s_\text{evict}$ (empty slot, or slot with minimum metadata value)

```python
def lookup(I, M, embed, x, P, policy):
    h = hash(x) % len(I)
    found_slot = None
    evict_slot = None
    evict_score = float('inf')

    # Pass 1: scan, no writes
    for i in range(P):
        s = (h + i) % len(I)
        if I[s] == x:
            found_slot = s
            break
        score = eviction_score(M[s], policy)  # 0 if empty, TTL or LRU score otherwise
        if score < evict_score:
            evict_score, evict_slot = score, s

    # Pass 2: act
    if found_slot is not None:
        M[found_slot] = current_time()        # refresh metadata
        return embed[found_slot]
    elif evict_slot is not None and is_evictable(M[evict_slot], policy):
        reset_slot(I, M, embed, evict_slot, x)  # evict + assign to x
        return embed[evict_slot]
    else:
        return embed[h]                        # fallback: return base slot (collision)
```

**Pass 2 — Action:** Based on Pass 1 findings, exactly one of three outcomes:
1. **Update**: ID found at $s^*$ → refresh $\mathcal{M}[s^*]$, return embedding.
2. **Allocate**: ID not found, evictable slot found → reset slot and assign to $x$.
3. **Fallback**: No evictable slot in probe range → return base hash slot (collision, rare at appropriate capacity).

> [!NOTE] Why two passes prevent duplicate storage
> A single-pass scan could insert $x$ at the first empty slot *before* discovering that $x$ already exists later in the probe window. The result: two slots for the same ID, creating inconsistency and wasted memory. Pass 1 ensures we only allocate if the ID is genuinely absent.

> [!QUESTION] Exercise 2: Probe Depth vs. Collision Rate Trade-off
> *This exercise quantifies how probe depth affects zero-collision guarantees.*
>
> > **Prerequisites:** [[#2.2 Two-Pass Lookup|§2.2 Two-Pass Lookup]]
>
> Suppose the embedding table has $N$ slots and $n$ distinct IDs are active ($n < N$). Treat each ID's base hash $h(x)$ as uniformly random in $[0, N)$.
>
> (a) With probe depth $P = 1$ (standard hashing), what is the expected number of IDs that collide with at least one other ID? Express in terms of $n$ and $N$.
>
> (b) With probe depth $P$, an ID $x$ can be placed collision-free if at least one of the $P$ slots $\{h(x), h(x)+1, \ldots, h(x)+P-1\}$ is unoccupied. Assuming independent occupancy with probability $\alpha = n/N$, what is the probability that all $P$ slots are occupied (a true collision that forces fallback)?
>
> (c) For the MPZCH production setting: $N = 200\text{M}$, $n = 150\text{M}$ (capacity ratio 1.33×), $P = 256$. Estimate the expected number of fallback collisions. Compare to the empirically reported 0.0000% collision rate.

> [!TIP]- Solution to Exercise 2
> **Key insight:** Linear probing with depth $P$ reduces collision probability exponentially in $P$.
>
> **(a)** With $P=1$, collision occurs for $x$ when another ID $y \neq x$ satisfies $h(y) = h(x)$. The probability that a specific slot $h(x)$ is occupied by another ID is $1 - (1 - 1/N)^{n-1} \approx n/N = \alpha$. Expected number of colliding IDs: $n \cdot \alpha = n^2/N$.
>
> For MovieLens numbers: $n = 162\text{K}$, $N \approx 2^{24} \approx 16\text{M}$: expected $\approx 162000^2 / 16\text{M} \approx 1641$ collisions, ≈ 1% rate. (The paper reports 7.73% with a smaller table — the smaller $N$, the higher the rate.)
>
> **(b)** Probability all $P$ slots are occupied: $\alpha^P$. With $\alpha = n/N$ and assuming independence (approximate for large $N$):
> $$P(\text{fallback for ID } x) = \alpha^P$$
>
> **(c)** With $\alpha = 150/200 = 0.75$ and $P = 256$:
> $$P(\text{fallback}) = 0.75^{256} \approx 10^{-32}$$
> Expected fallbacks out of $150\text{M}$ IDs: $150\text{M} \times 10^{-32} \approx 10^{-24}$. Effectively zero — consistent with the reported 0.0000% rate. *The exponential decay in $\alpha^P$ makes collision probability negligible well before $P$ is large.*

### 2.3 Eviction Policies

MPZCH supports two eviction policies, selected per embedding table:

**TTL (Time-To-Live):** A slot is evictable if $\mathcal{M}[s] < t_\text{current}$ (its last-seen timestamp has expired). Among all evictable slots in the probe range, MPZCH selects the *most expired* one (smallest $\mathcal{M}[s]$). TTL is appropriate for item embeddings with well-defined lifetimes (e.g., video post IDs expire after 24h; creator IDs expire after 72h).

**LRU (Least Recently Used):** The slot with the minimum $\mathcal{M}[s]$ (oldest last-access time) is evicted, regardless of absolute age. LRU is appropriate for user IDs, where no hard expiry makes semantic sense but inactive users' slots can be reclaimed.

### 2.4 Optimizer State Reset

*This is the key novelty over Monolith's cuckoo approach.* When a slot is evicted and assigned to a new ID, MPZCH performs a **full reset**:

1. $\mathcal{I}[s] \leftarrow x_\text{new}$
2. $\mathcal{M}[s] \leftarrow t_\text{current}$
3. **Embedding weights**: $\mathbf{e}[s] \leftarrow \mathbf{0}$ (reset to zero)
4. **Optimizer state**: Adam moments $m_1[s], m_2[s] \leftarrow \mathbf{0}$ (reset both first and second moment estimates)

Step 4 is the critical difference. Without optimizer state reset, the new ID's embedding inherits the previous occupant's Adam moment history — which encodes a "velocity" in the direction of the old ID's gradients. Even if the embedding weights are reset to zero, the optimizer will immediately push the weights in the old ID's direction on the first gradient step, recreating the stale embedding.

> [!DANGER] Adam state inheritance is subtler than weight inheritance
> Resetting weights but not Adam state is *worse than not resetting*: the embedding starts at zero but the first gradient update applies a large corrected step $\hat{m}_1 / (\sqrt{\hat{m}_2} + \epsilon)$ in the old ID's direction, potentially placing the embedding further from the new ID's true optimum than if it had simply inherited the old weights.

---

## 3. System Integration

### 3.1 Sharding and CUDA Kernels

MPZCH uses *row-wise sharding*: embedding table rows are distributed across ranks such that the full probe window $[h, h+P)$ for any ID falls within a single rank's local memory. This eliminates inter-rank communication during probe lookups — each lookup is a purely local CUDA kernel operation.

The CUDA kernel implements the two-pass scan in a single thread block per ID, with auxiliary tensors $\mathcal{I}$ and $\mathcal{M}$ stored in HBM alongside the embedding table. Key performance property: *probe depth $P$ has negligible effect on latency* (0.80 ms at $P=256$ vs. 0.90 ms at $P=8$) because the bottleneck is HBM memory bandwidth, not the sequential probe scan — modern HBM can read a 512-element contiguous probe window in a single burst.

### 3.2 Inference and Online Training

**At inference:** The identities tensor $\mathcal{I}$ is published alongside embedding weights and frozen as read-only. The metadata tensor $\mathcal{M}$ is *omitted* from the published model (reducing model size). Inference lookups use $\mathcal{I}$ to identify slot occupancy but skip all eviction logic.

**Online training:** Training updates stream through the same two-pass probe mechanism, updating $\mathcal{M}$ timestamps on access. New IDs seen in the stream trigger allocation; expired IDs are evicted as their slots are needed. Delta synchronization (as in [[papers/collisionless-embeddings/monolith|Monolith]]) transfers updated embeddings to the serving system, including newly allocated and reset slots.

---

## 4. Experiments

### 4.1 Collision Rate

**Test:** 150M unique user IDs, varying table sizes, probe depth $P=256$.

| Table Size | Capacity Ratio | Sigrid Hash | MPZCH (P=256) |
|---|---|---|---|
| 200M | 1.33× | 29.65% | **0.0000%** |
| 300M | 2.00× | 21.31% | **0.0000%** |

**Production deployment:** 4-billion-row table for 3 billion MAU, $P=512$ — zero collisions achieved.

### 4.2 Model Quality

**User embedding task** (Share, video view metrics):

| Metric | Normalized Entropy Improvement |
|---|---|
| Share | +0.38% |
| Video view duration | +0.12% |
| Video view percentage (100%) | +0.12% |
| Skip prediction | +0.09% |

**Item embedding freshness** (HSTU architecture, 24h Post ID TTL, 72h Owner ID TTL):

- **+0.83% impressions** for videos posted within 48 hours — the freshness metric most sensitive to stale embedding inheritance.
- Intra-creator embedding similarity (same-day videos): baseline 25% → MPZCH 38% (t-SNE cosine similarity). Same-day videos by the same creator now cluster correctly.
- Embedding trajectory analysis: MPZCH trajectories are smooth and monotone; baseline trajectories show erratic shifts from collision-induced gradient interference.

### 4.3 Latency

*Probe depth has negligible impact on training throughput:*

| Probe Depth $P$ | Latency (ms) |
|---|---|
| 8 | 0.90 |
| 256 | 0.80 |
| 512 | 0.90 |

The non-monotone trend ($P=256$ slightly faster than $P=8$) is likely a CUDA kernel scheduling artifact; all values are within measurement noise. **Production training QPS is maintained.**

> [!QUESTION] Exercise 3: Optimizer State Reset and Adam Convergence
> *This exercise shows why Adam state inheritance causes erratic embedding updates.*
>
> > **Prerequisites:** [[#2.4 Optimizer State Reset|§2.4 Optimizer State Reset]]
>
> Consider an embedding $\mathbf{e} \in \mathbb{R}^d$ optimized with Adam (learning rate $\eta$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$). The slot was previously trained for $T$ steps by ID $x_\text{old}$, accumulating moments $m_1^{(T)}, m_2^{(T)}$.
>
> (a) After a full reset ($\mathbf{e} \leftarrow \mathbf{0}$, $m_1 \leftarrow \mathbf{0}$, $m_2 \leftarrow \mathbf{0}$), the first gradient step for new ID $x_\text{new}$ with gradient $g_1$ gives an update $\Delta\mathbf{e}$. Write the expression for $\Delta\mathbf{e}$ using the bias-corrected Adam update. Show that $\|\Delta\mathbf{e}\|$ is bounded by $\eta\sqrt{d}$ regardless of $g_1$.
>
> (b) After a weight-only reset ($\mathbf{e} \leftarrow \mathbf{0}$, but $m_1 = m_1^{(T)}, m_2 = m_2^{(T)}$ inherited), the first gradient step for $x_\text{new}$ with gradient $g_1$ applies the bias-corrected update with step $t = T+1$. Show that if $m_1^{(T)}$ and $g_1$ point in opposite directions, the weight-only reset can produce $\|\Delta\mathbf{e}\| \gg \eta\sqrt{d}$, potentially larger than under no reset at all.
>
> (c) Argue qualitatively: for a fresh video with no interaction history, what does a large erratic first update do to early recommendation quality?

> [!TIP]- Solution to Exercise 3
> **Key insight:** Inherited Adam moments amplify the first gradient step by the accumulated magnitude of the old ID's gradient history, potentially dwarfing the new ID's true learning signal.
>
> **(a)** After full reset, $m_1 = \beta_1 g_1$, $m_2 = (1-\beta_2)g_1^2$ (element-wise). Bias-corrected: $\hat{m}_1 = m_1 / (1-\beta_1) = g_1$, $\hat{m}_2 = m_2 / (1-\beta_2) = g_1^2$. Update:
> $$\Delta\mathbf{e} = -\eta \frac{\hat{m}_1}{\sqrt{\hat{m}_2} + \epsilon} = -\eta \frac{g_1}{|g_1| + \epsilon} \approx -\eta \cdot \text{sign}(g_1)$$
> Each coordinate updates by at most $\pm\eta$, so $\|\Delta\mathbf{e}\| \leq \eta\sqrt{d}$. The full reset degrades to sign-gradient descent on the first step — safe and bounded.
>
> **(b)** With inherited moments at step $t = T+1$: $m_1^{(T+1)} = \beta_1 m_1^{(T)} + (1-\beta_1)g_1$, $m_2^{(T+1)} = \beta_2 m_2^{(T)} + (1-\beta_2)g_1^2$. The bias-corrected denominator $\sqrt{\hat{m}_2^{(T+1)}}$ is inflated by the accumulated second moment from $T$ old steps. If $m_1^{(T)}$ points strongly opposite to $g_1$, the numerator is suppressed while the denominator reflects the old ID's gradient magnitude. The effective step size is $\eta \cdot \|m_1^{(T)}\| / \sqrt{\|m_2^{(T)}\|}$ — which can be arbitrarily large relative to the correct step $\eta\sqrt{d}$.
>
> **(c)** A fresh video needs its early gradient steps to accurately reflect the actual user interactions it receives (clicks, watches, skips). A large erratic first update moves the embedding far from initialization in an arbitrary direction (inherited from another video), making the model's early predictions highly inaccurate for this video. Early inaccuracy → poor early recommendations → the video receives less traffic → fewer gradient updates → the model never corrects the bad initialization. This is a *feedback loop* that systematically suppresses fresh content.

---

## References

| Reference | Brief Summary | Link |
|---|---|---|
| [MPZCH: Multi-Probe Zero Collision Hash](https://arxiv.org/abs/2602.17050) | Linear probing with TTL/LRU eviction + optimizer state reset; zero collisions at 3B MAU; +0.83% freshness | [arXiv:2602.17050](https://arxiv.org/abs/2602.17050) |
| [Monolith: Real-Time Recommendation System With Collisionless Embedding Table](https://arxiv.org/abs/2209.07663) | Foundational ByteDance paper: cuckoo-hash embedding tables + online training architecture | [arXiv:2209.07663](https://arxiv.org/abs/2209.07663) |
| [Adam: A Method for Stochastic Optimization (Kingma & Ba, 2015)](https://arxiv.org/abs/1412.6980) | Introduces the Adam optimizer with first and second moment estimates; optimizer state semantics are critical for understanding stale inheritance | [arXiv:1412.6980](https://arxiv.org/abs/1412.6980) |
| [HSTU: Hierarchical Sequential Transduction Units for Large-Scale Recommendation](https://arxiv.org/abs/2402.17152) | Architecture used in MPZCH's item embedding experiments | [arXiv:2402.17152](https://arxiv.org/abs/2402.17152) |
