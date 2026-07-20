# SlimPer: Make Personalization Model Slim and Smart

Siqi Wang, Xianjie Chen, and 66+ co-authors. Meta Platforms, Inc. arXiv:2607.12281v1, July 2026.

| Dimension | Prior State (HSTU + Wukong hybrid) | This Paper | Key Result |
|---|---|---|---|
| Paradigm | Generative sequence modeling: user history treated as a growing token sequence, processed by transformer-style attention over the full sequence | Discriminative iterative refinement: a fixed-size $K \times d$ knowledge base is repeatedly refined via cross-attention into the raw multi-modal token set | Reframes ranking as $O(N)$-per-layer refinement rather than $O(N^2)$ sequence transduction |
| Per-layer complexity | $O(N^2)$ self-attention (or $O(N^2/B)$ with request-batch amortization) | $O(N \cdot q)$ cross-attention into a compact query set, $q \ll N$ | $\sim 8\times$ FLOPs reduction at $N=2048$, $\sim 25\times$ at $N=6144$ |
| Max practical sequence length | Throughput collapses as $N$ grows past a few thousand events (quadratic wall) | Near-flat throughput growth out to 6k+ events | $<20\%$ QPS regression at $N=6\text{k}$ vs. $48\%$ regression for the baseline |
| Memory | $O(L \cdot N)$ per-layer activations, growing with history length | $O(L \cdot K)$ per-layer, independent of $N$ | $-9.32\%$ to $-18.12\%$ memory at matched sequence length |
| Headline NE / QPS tradeoff | baseline $(0, 0)$ | Instagram Reels, $N=2\text{k}$: $-0.51\%$ NE, $+11\%$ QPS; Feed, $N=4\text{k}$: $-0.94\%$ NE, $-16.07\%$ QPS | $-0.51\%$ to $-0.94\%$ NE improvement across production surfaces |

## Relations

**Builds on:** [[papers/generative-recommenders/hstu|HSTU]], [[papers/generative-recommenders/wukong|Wukong]]
**Concepts used:** [[concepts/attention-mechanisms/standard-attention|Standard Attention]], [[concepts/attention-mechanisms/linear-attention|Linear Attention]]

## Table of Contents

- [[#1. Motivation and Background|1. Motivation and Background]]
  - [[#1.1 The Discriminative vs. Generative Mismatch|1.1 The Discriminative vs. Generative Mismatch]]
  - [[#1.2 Prior Art: HSTU, Wukong, and the Transformer-Mimicking Paradigm|1.2 Prior Art: HSTU, Wukong, and the Transformer-Mimicking Paradigm]]
  - [[#1.3 SlimPer's Reframing: Iterative Knowledge-Base Refinement|1.3 SlimPer's Reframing: Iterative Knowledge-Base Refinement]]
- [[#2. Problem Formulation and Notation|2. Problem Formulation and Notation]]
  - [[#2.1 Multi-Modal Inputs and the Scoring Function|2.1 Multi-Modal Inputs and the Scoring Function]]
  - [[#2.2 Tokenization and the Knowledge Base State|2.2 Tokenization and the Knowledge Base State]]
- [[#3. The SlimPer Layer: Select, Match, Refine|3. The SlimPer Layer: Select, Match, Refine]]
  - [[#3.1 Step 1: Selection via Modality-Dependent Attention Kernels|3.1 Step 1: Selection via Modality-Dependent Attention Kernels]]
  - [[#3.2 Step 2: Explicit Multifaceted Relevance Matching|3.2 Step 2: Explicit Multifaceted Relevance Matching]]
  - [[#3.3 Step 3: Knowledge Base Refinement|3.3 Step 3: Knowledge Base Refinement]]
  - [[#3.4 Stacking Layers and Task Heads|3.4 Stacking Layers and Task Heads]]
- [[#4. Complexity Analysis|4. Complexity Analysis]]
  - [[#4.1 Per-Layer Cost Comparison|4.1 Per-Layer Cost Comparison]]
  - [[#4.2 Resolving Memory versus Interaction Capacity|4.2 Resolving Memory versus Interaction Capacity]]
- [[#5. Information-Theoretic Justification|5. Information-Theoretic Justification]]
  - [[#5.1 The Output Entropy Bound|5.1 The Output Entropy Bound]]
  - [[#5.2 Repeated Raw-Token Access and the Data Processing Inequality|5.2 Repeated Raw-Token Access and the Data Processing Inequality]]
  - [[#5.3 A Tension: Capacity versus Per-Layer Bandwidth|5.3 A Tension: Capacity versus Per-Layer Bandwidth]]
- [[#6. Request-Only Optimization|6. Request-Only Optimization]]
- [[#7. Experiments|7. Experiments]]
  - [[#7.1 Offline Results on Instagram Reels and Feed|7.1 Offline Results on Instagram Reels and Feed]]
  - [[#7.2 Scaling with Sequence Length|7.2 Scaling with Sequence Length]]
  - [[#7.3 Ablations|7.3 Ablations]]
  - [[#7.4 Online A/B Tests|7.4 Online A/B Tests]]
  - [[#7.5 Interpretability|7.5 Interpretability]]
- [[#8. Limitations|8. Limitations]]
- [[#9. Relation to HSTU and Wukong|9. Relation to HSTU and Wukong]]
- [[#10. References|10. References]]

---

## 1. Motivation and Background

### 1.1 The Discriminative vs. Generative Mismatch

Personalized ranking is, at its core, a *discriminative* task: given a user $u$ and a candidate item $i$, the model must produce a small, fixed number $\tau$ of relevance scores — one per prediction task (click, like, share, watch-time bucket, etc.). This is categorically different from *generative* sequence modeling, where the objective is to predict a full distribution over the next token at every position of a growing sequence.

Sequential transducers such as [[papers/generative-recommenders/hstu|HSTU]] adopt the generative framing regardless: user history is treated as an unbounded time series $h = (x_0, \ldots, x_{n-1})$, and a Transformer-style block computes pairwise self-attention $QK^\top \in \mathbb{R}^{n \times n}$ over the entire history at every layer. SlimPer's starting observation is that this is architecturally a mismatch: the *output* of a ranking model is $\tau$ scalars, not a per-token distribution, so there is no intrinsic reason the *intermediate representation* needs to grow with $N$ (the history length) the way it would in an autoregressive language model, where every position genuinely needs its own output.

*This is the central thesis SlimPer builds on:* if the final answer is low-dimensional, an architecture whose per-layer state also stays low-dimensional — while still retaining full read access to the raw history — should suffice, and should be cheaper.

> [!NOTE] Formalizing the "structural difference" argument
> The paper motivates this reframing by contrasting *why* full-width state through depth is justified for autoregressive LLMs but not here. Consider the autoregressive factorization
> $$
> \log p(x_1, \ldots, x_N) = \sum_{i=1}^N \log p(x_i \mid x_{<i}),
> $$
> a sum of $N$ *independently supervised* terms. Under teacher forcing, this entire sum is computed in a **single parallel forward pass**: a causal mask lets position $i$ attend only to positions $\le i$, so one pass over $x_1, \ldots, x_N$ yields a contextualized hidden state $h_i^{(\ell)}$ at *every* position $i$ and *every* layer $\ell$ simultaneously, and the final-layer states $h_1^{(L)}, \ldots, h_N^{(L)}$ are read out in parallel to supply all $N$ terms of the loss at once — rather than requiring $N$ separate forward passes, one per prefix length. Maintaining $O(N \times d)$ activations through all $L$ layers is therefore *forced* by this training-time parallelism: the model owes $N$ independent predictions per forward pass, and the resulting $O(N^2)$ self-attention cost is amortized over those $N$ outputs, i.e. $O(N)$ compute per prediction. (At autoregressive *generation* time the same $O(N)$ state is instead built up incrementally, one token per step, and cached as KV pairs — sequential in order of construction, but identical in asymptotic footprint by the time $N$ tokens have been produced.)
>
> Personalized ranking supplies exactly *one* supervised quantity per forward pass — $f(u,i;\mathbf S,\mathbf D,\mathbf E) \to \mathbb R^\tau$ ([[#2.1 Multi-Modal Inputs and the Scoring Function|§2.1]]), not $N$. No loss term ever asks the network to reconstruct or predict an individual historical event, so nothing forces per-event representations to survive intact through depth the way per-position states must in the LLM case. If a Transformer backbone is used anyway, it inherits an $O(N)$-width, $O(N^2)$-attention cost structure — but that cost now buys **one** prediction instead of $N$, degrading the cost-per-prediction ratio from $O(N)$ to $O(N^2)$.
>
> This is what makes the mismatch a genuine *dilemma* rather than mere inefficiency — there are exactly two ways to fit a Transformer to a single-output task, and both are bad:
> - **Compress early** — shrink representation width fast across depth. With no per-position loss supplying gradient pressure for what a later layer might need, compression is *unsupervised* by anything except the single final score: information can be discarded that no later layer can recover.
> - **Never compress** — carry full $O(N)$ state through all $L$ layers. This avoids that information loss but pays $O(L \cdot N)$ memory and $O(L \cdot N^2)$ compute ([[#4.1 Per-Layer Cost Comparison|§4.1]]) for an output of only $\tau \ll N$ scalars.
>
> The paper's diagnosis is that neither horn is inherent to *ranking* — both are artifacts of importing an architectural premise (uniform full-width state across depth) that a different supervision structure (per-token loss) had licensed. SlimPer's Select-Match-Refine cycle ([[#3. The SlimPer Layer: Select, Match, Refine|§3]]) is precisely the design that dodges both horns at once: the *carried* state stays $O(K)$ (avoiding horn 2), while every layer re-reads the *uncompressed* raw tokens rather than only the previous layer's summary (avoiding horn 1) — this is exactly the "no single bottleneck" argument formalized in [[#5.2 Repeated Raw-Token Access and the Data Processing Inequality|§5.2]].

### 1.2 Prior Art: HSTU, Wukong, and the Transformer-Mimicking Paradigm

SlimPer positions its baseline as a **mature hybrid of [[papers/generative-recommenders/hstu|HSTU]] and [[papers/generative-recommenders/wukong|Wukong]]** — HSTU's sequential transduction handling the event history, Wukong's factorization-machine stack handling dense/sparse feature interactions. This hybrid is representative of what the paper calls the *transformer-mimicking paradigm*: irrespective of whether the interaction mechanism is pointwise attention (HSTU) or explicit factorized dot products (Wukong), both process a token set whose size scales with $N$, and both therefore inherit at least one $O(N)$-or-worse cost term per layer.

HSTU further amortizes candidate scoring via M-FALCON (see [[papers/generative-recommenders/hstu#4.3 M-FALCON: Inference Amortization|HSTU §4.3]]), which computes the $O(N^2 d)$ user-history self-attention once per request and reuses it across the $b_m$ candidates scored in that request — this is a *sequence-only* instance of what SlimPer calls Request-Only Optimization (see [[#6. Request-Only Optimization|§6]]).

### 1.3 SlimPer's Reframing: Iterative Knowledge-Base Refinement

SlimPer replaces the growing token sequence with a **fixed-size knowledge base** $\mathcal{X} \in \mathbb{R}^{K \times d}$, $K = 64 \ll N$, that is refined across $L$ layers. Each layer does not attend to a compressed summary of the previous layer alone — it cross-attends *directly into the raw multi-modal token set* (sparse features $\mathbf{S}$, dense features $\mathbf{D}$, and event tokens $\mathbf{E}$) every time. The state carried between layers stays $O(K)$; the *compute* touches all $N$ raw tokens at every layer, but only through a small number of query rows, giving an overall $O(N)$-per-layer (not $O(N^2)$) cost. Sections [[#3. The SlimPer Layer: Select, Match, Refine|§3]]–[[#5. Information-Theoretic Justification|§5]] formalize this design and its complexity and information-theoretic consequences.

---

## 2. Problem Formulation and Notation

### 2.1 Multi-Modal Inputs and the Scoring Function

**Definition (Personalized Ranking Scoring Function).** Given a user $u$ and candidate item $i$, along with three heterogeneous input modalities:
- $\mathbf{S}$ — *sparse features*: high-cardinality categorical identifiers (user ID, item ID, category, etc.),
- $\mathbf{D}$ — *dense features*: continuous numerical signals,
- $\mathbf{E}$ — *event tokens*: the structured sequence of the user's past interactions, of length $N$,

the model learns
$$
f(u, i; \mathbf{S}, \mathbf{D}, \mathbf{E}) \to \mathbb{R}^\tau,
$$
producing $\tau$ task-specific relevance scores (e.g. $P(\text{click})$, $P(\text{like})$, $P(\text{share})$, …).

This formulation is deliberately agnostic to *how* $\mathbf{E}$ is processed — it does not presuppose a sequence-transduction architecture the way HSTU's next-action objective does. That degree of freedom is exactly what SlimPer exploits.

### 2.2 Tokenization and the Knowledge Base State

Each event $x_t \in \mathbf{E}$ is embedded via an *event modeling module*:
$$
\mathbf{E}^{(t)} = \mathrm{MLP}_E^s\big(\mathbf{e}^{(t)}_{\text{sparse}}\big) + \mathbf{e}^{(t)}_{\text{temporal}} + \mathrm{MLP}_E^e\big(\mathbf{e}^{(t)}_{\text{type}}\big) + \mathbf{e}^{(t)}_{\text{context}}, \tag{Eq. 16}
$$
summing content, temporal (position + delta-time), action-type, and local-context embeddings, where the *contextual encoding* term is itself computed from a window of neighboring events,
$$
\mathbf{e}^{(t)}_{\text{context}} = \Psi_{\text{context}}(x_{t-w+1}, \ldots, x_t). \tag{Eq. 1}
$$
Dense features are concatenated and projected, $\mathbf{D} = \mathrm{MLP}(\mathrm{Concat}(\mathbf{d}_1, \ldots, \mathbf{d}_{|\mathcal{D}|}))$ (Eq. 17), and sparse features $\mathbf{S}$ use standard embedding-table lookups.

**Definition (Knowledge Base).** The persistent state carried between layers is
$$
\mathcal{X}^k \in \mathbb{R}^{K \times d}, \qquad K = 64,\ d = 256,
$$
i.e. $K$ fixed *slots*, each a $d$-dimensional embedding, initialized (e.g. from learned or task-conditioned priors) before the first refinement layer and updated $L$ times. All query, template, and task-head vectors used inside a SlimPer layer are produced from $\mathcal{X}^k$ by a *segment-wise linear projection*
$$
\mathcal{L}(\alpha) = \rho\, \alpha, \qquad \rho \in \mathbb{R}^{L_{\text{out}} \times L_{\text{in}}},
$$
so that, e.g., the query set is $\mathbf{Q} = \mathcal{L}(\mathcal{X}^k) \in \mathbb{R}^{q \times d}$ with $q = 16$, and the template set is $\mathbf{T} = \mathcal{L}(\mathcal{X}^k) \in \mathbb{R}^{t \times d}$ with $t = 32$. Crucially, $\mathbf{Q}$ and $\mathbf{T}$ are *derived from the current knowledge base*, not from the raw tokens — the raw tokens only ever play the role of keys/values being read from, never queries.

---

## 3. The SlimPer Layer: Select, Match, Refine

Each of the $L$ SlimPer layers applies a three-step cycle to update $\mathcal{X}^k \to \mathcal{X}^{k+1}$.

### 3.1 Step 1: Selection via Modality-Dependent Attention Kernels

The knowledge base state generates a compact query set $\mathbf{Q} = \mathcal{L}(\mathcal{X}^k) \in \mathbb{R}^{q \times d}$, which cross-attends into each raw modality using a kernel $\Phi$ chosen by modality type:

$$
\Phi(\mathbf{Q}, \mathbf{K}) =
\begin{cases}
\mathrm{MLP}\big(\mathrm{Concat}(\mathcal{L}(\mathbf{Q}), \mathcal{L}(\mathbf{K}))\big), & \text{sparse features} \\[4pt]
\mathrm{Softmax}(\gamma\, \mathbf{Q}\mathbf{K}^\top), & \text{sequence features}
\end{cases}
\tag{Eq. 4}
$$

with dense features fused directly (no attention kernel — they enter Eq. 10 as an additive term). The sequence-feature branch is exactly [[concepts/attention-mechanisms/standard-attention|standard scaled dot-product attention]] restricted to $q \ll N$ query rows rather than $N$; the sparse-feature branch replaces the dot-product similarity with a learned MLP over the concatenated (rather than multiplied) query/key pair — a design closer to the target-aware local activation unit used in DIN and inherited conceptually by HSTU (see [[papers/generative-recommenders/hstu#1.3 The Generative Recommender Paradigm|HSTU §1.3]]) than to vanilla softmax attention.

The resulting retrieved-evidence vectors are
$$
\mathbf{R}_s = \Phi_s(\mathbf{Q}, \mathbf{S})\, \mathbf{S} \in \mathbb{R}^{q \times d}, \qquad
\mathbf{R}_e = \Phi_e(\mathbf{Q}, \mathbf{E})\, \mathbf{E} \in \mathbb{R}^{q \times d}. \tag{Eqs. 5–6}
$$

*Note the shape:* both outputs have $q$ rows regardless of how large $N = |\mathbf{E}|$ is — the raw event count is entirely absorbed into the attention sum, not into the output dimensionality. This is the mechanism that keeps the per-layer state size independent of history length.

### 3.2 Step 2: Explicit Multifaceted Relevance Matching

A second, independent projection of the knowledge base produces a *template* set,
$$
\mathbf{T} = \mathcal{L}(\mathcal{X}^k) \in \mathbb{R}^{t \times d}, \qquad t = 32, \tag{Eq. 7}
$$
which is matched against the retrieved evidence via explicit dot products,
$$
\lambda_s = \mathrm{DotProduct}(\mathbf{R}_s, \mathbf{T}) \in \mathbb{R}^{q \times t}, \qquad
\lambda_e = \mathrm{DotProduct}(\mathbf{R}_e, \mathbf{T}) \in \mathbb{R}^{q \times t}. \tag{Eqs. 8–9}
$$

**Remark.** Where Step 1 asks "what evidence is relevant to my current queries?", Step 2 asks "how well does that evidence match each of $t$ learned *facets* of interest?" — decoupling *retrieval* from *scoring against interpretable templates*. This separation is also the mechanism exploited for interpretability in [[#7.5 Interpretability|§7.5]].

### 3.3 Step 3: Knowledge Base Refinement

The knowledge base is additively updated using all evidence gathered in Steps 1–2, plus the dense-feature vector and the (normalized) previous state:

$$
\mathcal{X}^{k+1} = \mathcal{X}^k + \mathrm{MLP}_\mu\Big(\mathrm{Concat}\big(\mathrm{RMSNorm}(\lambda_s),\ \mathrm{RMSNorm}(\lambda_e),\ \mathcal{L}(\mathcal{X}^k),\ \mathbf{D}\big)\Big). \tag{Eq. 10}
$$

This is a residual update in the style of a Transformer block's feed-forward sublayer, but the "sequence position" being updated is a *knowledge-base slot*, not a raw token. Because $\mathcal{X}^{k+1}$ depends on $\mathcal{X}^k$ through $\mathbf{Q}$, $\mathbf{T}$, and the residual term, and each of those in turn attended over the *full, un-compressed* $\mathbf{S}, \mathbf{D}, \mathbf{E}$, no information is permanently discarded between layers — layer $k+1$ can, in principle, retrieve evidence that layer $k$ did not (see [[#5. Information-Theoretic Justification|§5]]).

### 3.4 Stacking Layers and Task Heads

After $L$ layers, per-task query vectors are read out from the final state:
$$
\mathbf{P}_p^L = \mathrm{MLP}_p\big(\mathcal{L}(\mathcal{X}^L)\big), \qquad p \in [1, \tau], \tag{Eq. 11}
$$
producing the $\tau$-dimensional output of [[#2.1 Multi-Modal Inputs and the Scoring Function|§2.1]]'s scoring function. Production configurations use $L = 5$ layers for Instagram Feed and $L = 7$ for Instagram Reels, with $K = 64$, $d = 256$, $q = 16$, $t = 32$ throughout (Table 4 of the paper).

---

## 4. Complexity Analysis

### 4.1 Per-Layer Cost Comparison

The paper's Table 2 compares five architecture families along three axes: tokenization cost, per-layer stacked-computation cost, and *user-item interaction capacity* (defined below). $N$ is the history length, $L$ the depth, $B$ the request-batch size used for candidate amortization, $K = 64$ the knowledge-base size, and $q = 16$ the query-set size.

| Model | Tokenization | Stacked-layer cost | Interaction capacity |
|---|---|---|---|
| Conventional DLRM | $O(N)$ mem, $O(N)$ compute | $O(L \cdot 64)$ mem, $O(L \cdot N \cdot 16)$ compute | $O(L \cdot N \cdot 64)$ |
| ROO-aware Transformer (HSTU) | $O(N/B)$ mem, $O(N/B)$ compute | $O(L \cdot N/B)$ mem, $O(L \cdot (N+1)^2/B)$ compute | $O(L \cdot N)$ |
| Non-ROO Transformer (OneTrans-style) | $O(N)$ mem, $O(N)$ compute | $O(L \cdot N)$ mem, $O(L \cdot N^2)$ compute | $O(L \cdot N)$ |
| Sub-quadratic Transformer ([[concepts/attention-mechanisms/linear-attention|linear attention]]) | $O(N)$ mem, $O(N)$ compute | $O(L \cdot N)$ mem, $O(L \cdot N)$ compute | $O(L \cdot N)$ |
| **SlimPer** | $O(N/B)$ mem, $O(N/B)$ compute | $O(L \cdot K)$ mem, $O(L \cdot N \cdot q)$ compute | $O(L \cdot N \cdot K)$ |

The headline contrast is against the two Transformer rows the paper actually deploys against (the HSTU + Wukong hybrid corresponds most closely to the ROO-aware Transformer row): quadratic-in-$N$ per-layer compute $O(L(N+1)^2/B)$ versus SlimPer's linear-in-$N$ $O(L \cdot N \cdot q)$, and $O(L \cdot N/B)$ memory versus SlimPer's *history-independent* $O(L \cdot K)$.

### 4.2 Resolving Memory versus Interaction Capacity

*At first glance* the last column looks inconsistent with the second: SlimPer has **lower** per-layer memory ($O(L \cdot K)$, a constant in $N$) than the Transformer rows ($O(L \cdot N/B)$ or $O(L \cdot N)$), yet **higher** interaction capacity ($O(L \cdot N \cdot K)$, a factor of $K$ above the Transformer rows' $O(L \cdot N)$). Lower memory and higher capacity pulling in opposite directions from the same $N$-dependence is worth resolving carefully rather than taking on faith.

**Remark (the two quantities measure different things).** *Memory* measures the size of the tensor that is actually *retained* across the layer boundary — for SlimPer this is exactly $\mathcal{X}^{k} \in \mathbb{R}^{K \times d}$ (Eq. 10); the $N$ raw tokens are re-read from a shared, request-amortized cache at every layer but never re-materialized into the carried state. *Interaction capacity* instead counts the number of raw-token read-accesses that occur, even transiently, during the forward pass — i.e. how many distinct (query-slot, event) pairs the model is capable of directly conditioning on before they are reduced away. For the ROO-aware Transformer row, only the *target item's* query row attends over the $N$ history positions (target-aware attention, cf. [[papers/generative-recommenders/hstu#2.2 Retrieval vs. Ranking Formulation|HSTU §2.2]]) — a single query row, hence $O(N)$ capacity per layer. SlimPer instead issues $q = 16$ independent query rows per layer (Eq. 4–6) whose outputs, after Step 2's matching, feed into an update that touches all $K = 64$ knowledge-base slots (Eq. 10) — so the *effective* number of independent (slot, event) channels through which raw history can influence the final prediction is $O(N \cdot K)$, even though the attention operation computing them costs only $O(N \cdot q)$ per layer (note $q=16 < K=64$: the compute-cost row and the capacity row use different constants for exactly this reason — capacity is a property of the full knowledge base that Step 3 broadcasts the matched evidence into, not of the smaller query set that produces it).

This is the same asymmetric-attention trick used by Perceiver / Perceiver IO (Jaegle et al., 2021): a small latent array of size $K$ cross-attends into a large input array of size $N$ at cost $O(NK)$, while only the $O(K)$ latent array — not the $O(N)$ input — is carried forward as state. **Cheap-to-store and rich-to-compute are not in tension; they are exactly what asymmetric cross-attention is designed to decouple.**

---

## 5. Information-Theoretic Justification

### 5.1 The Output Entropy Bound

The paper's argument for why a fixed-size $K \times d$ bottleneck need not be lossy begins with an entropy bound on the *label*, not the input. If $Y = (Y_1, \ldots, Y_\tau)$ are the $\tau$ (e.g. binary) task labels, then
$$
H(Y) \le \tau \text{ bits},
$$
with equality only in the degenerate worst case where every $Y_p$ is an independent, unbiased coin flip; in practice $H(Y)$ is typically far smaller due to label correlation and skewed base rates (most impressions are not clicked). The knowledge base has raw representational capacity $K \times d = 64 \times 256 = 16{,}384$ real numbers — at even a conservative 16-bit quantization this is $2.6 \times 10^5$ bits, several orders of magnitude above $\tau$.

*This bound alone is not sufficient to justify the architecture.* $H(Y) \le \tau$ bounds how much information is needed to represent the *label*, but the quantity that actually matters for whether compressing $\mathbf{S}, \mathbf{D}, \mathbf{E}$ into $\mathcal{X}^L$ is safe is the *task-relevant* mutual information $I(\mathbf{S}, \mathbf{D}, \mathbf{E}; Y)$, not $H(Y)$ itself — a sufficient statistic $T(\mathbf{S}, \mathbf{D}, \mathbf{E})$ need only satisfy $I(T; Y) = I(\mathbf{S}, \mathbf{D}, \mathbf{E}; Y) \le H(Y) \le \tau$ bits. The paper's own presentation elides this distinction between $H(Y)$ and $I(X;Y)$; the two coincide only when the label is a deterministic function of the input, which is not the case in ranking.

### 5.2 Repeated Raw-Token Access and the Data Processing Inequality

**Proposition (Data Processing Inequality, DPI).** For any Markov chain $X \to Z \to Y'$ — meaning $Y'$ depends on $X$ only through $Z$ — the mutual information satisfies $I(X; Y') \le I(X; Z)$. No downstream processing of $Z$ can recover information about $X$ that $Z$ itself does not retain.

If a model compressed the raw input once, $X \to Z_1$, and then discarded $X$ — computing every subsequent layer as a function of $Z_1$ alone, $Z_1 \to Z_2 \to \cdots \to Z_L \to \hat{Y}$ — the chain $X \to Z_1 \to \hat Y$ would be a genuine Markov chain, and $I(X; \hat Y) \le I(X; Z_1)$ would be an *irreversible* ceiling: whatever $Z_1$ failed to preserve about $X$ is permanently unrecoverable, regardless of how much additional compute layers 2 through $L$ apply.

SlimPer's Step 1 (Eqs. 4–6) is executed against $\mathbf{S}, \mathbf{D}, \mathbf{E}$ *directly, at every layer* — not against $\mathcal{X}^{k-1}$'s compression of them. Formally, the computational graph is not $X \to Z_1 \to Z_2 \to \cdots$; it is closer to $\hat Y = g(X, \mathcal{X}^1, \ldots, \mathcal{X}^L)$, where $X$ re-enters the computation at every layer rather than being routed exclusively through a single compressed intermediate. Consequently there is no single $Z_1$ through which *all* of $X$'s influence on $\hat Y$ must flow, so the single-bottleneck form of the DPI bound above does not apply to the *architecture as a whole* — each layer gets a fresh, independent opportunity to extract information from the raw tokens that a previous layer's queries happened not to retrieve.

> [!WARNING]
> This argument shows only that SlimPer's architecture avoids the specific failure mode of a *single, irreversible* compress-then-discard step — it does **not** by itself prove the model achieves any information-theoretic optimum, nor that no relevant signal is lost in practice. Losses can still occur through (a) the attention kernels $\Phi$ themselves being imperfect estimators of relevance, (b) $\mathrm{MLP}_\mu$'s limited capacity to fold new evidence into $\mathcal{X}^k$ without overwriting previously retained facets, or (c) too few layers $L$ for the iterative retrieval process to converge on all task-relevant evidence before the readout in Eq. 11. The paper's own ablations (§5.3 below) show exactly this second failure mode empirically.

### 5.3 A Tension: Capacity versus Per-Layer Bandwidth

If the $K \times d$ capacity argument of [[#5.1 The Output Entropy Bound|§5.1]] were the whole story, reducing $K$ should not matter much, since even $K=4$, $d=256$ gives $1{,}024$ real numbers — still orders of magnitude above $\tau$ bits. Yet the ablation in [[#7.3 Ablations|§7.3]] shows that shrinking $K$ from 64 to 4 measurably *degrades* NE by $+1.2\%$.

**Remark.** This is not a contradiction of §5.1–§5.2, but it does sharpen what those arguments actually establish. $H(Y) \le \tau$ bounds the *total information capacity* of the knowledge base needed at convergence — a static, information-theoretic quantity. It says nothing about the *per-round bandwidth* available to move evidence from the raw tokens into the base: with $K = 4$ slots, Step 1 produces only $q$ query rows derived from a 4-slot state, and Step 3 can only write updates into 4 slots per layer (Eq. 10). A smaller $K$ is a smaller *per-layer communication channel*, not necessarily a smaller *eventual capacity* — but with a fixed budget of $L$ layers, a narrower per-round channel may simply not have enough rounds to converge on a sufficient statistic before the readout in Eq. 11 fires. This is a distinction between total information capacity and per-round communication complexity, and the ablation is better read as evidence about the latter, not a counterexample to the former.

---

## 6. Request-Only Optimization

**Definition (Request-Only Optimization, ROO).** For a single request scoring $B$ candidate items against the same user, the user-side tokens ($\mathbf{S}$, $\mathbf{D}$, $\mathbf{E}$ tokenization, and — for SlimPer — the shared portions of the knowledge-base refinement that do not depend on the candidate item) are computed **once** per request and shared across all $B$ candidates, rather than recomputed per candidate.

This is the mechanism behind the $O(N/B)$ tokenization-cost rows in Table 2 ([[#4.1 Per-Layer Cost Comparison|§4.1]]). HSTU's M-FALCON ([[papers/generative-recommenders/hstu#4.3 M-FALCON: Inference Amortization|HSTU §4.3]]) applies exactly this idea, but only to the sequence/attention computation — the user-history self-attention is shared across the microbatch, while sparse and dense feature paths are, in the paper's characterization, not uniformly amortized in the same way. SlimPer's claimed contribution here is to extend request-only sharing to *all three modalities* — sparse, dense, and sequence — since all three participate symmetrically in Steps 1–3 of every layer (Eqs. 4–10), rather than being handled by separate, heterogeneously-amortized subsystems.

---

## 7. Experiments

### 7.1 Offline Results on Instagram Reels and Feed

| Surface | Config | NE (%) | QPS (%) | Memory (%) |
|---|---|---|---|---|
| Reels | Baseline, $N=2\text{k}$ | 0.00 | 0.00 | 0.00 |
| Reels | SlimPer, $N=2\text{k}$ | $-0.51$ | $+11.0$ | $-9.32$ |
| Reels | SlimPer, $N=5\text{k}$ | $-0.80$ | $-10.0$ | $+4.59$ |
| Feed | Baseline, $N=1\text{k}$ | 0.00 | 0.00 | 0.00 |
| Feed | SlimPer, $N=1\text{k}$ | $-0.49$ | $+12.5$ | $-18.12$ |
| Feed | SlimPer, $N=4\text{k}$ | $-0.94$ | $-16.07$ | $+2.04$ |

NE (Normalized Entropy) is lower-is-better; QPS is throughput relative to baseline (positive = faster); memory is relative footprint (negative = smaller). At matched, moderate sequence lengths ($N=1\text{–}2\text{k}$), SlimPer is simultaneously better on quality, throughput, *and* memory. At longer lengths, the throughput advantage narrows (Reels $N=5\text{k}$: $-10\%$ QPS) as the linear-in-$N$ term $O(L \cdot N \cdot q)$ starts to dominate the request-amortized $O(N/B)$ tokenization cost — but quality continues to improve monotonically with $N$, motivating the dedicated scaling analysis below.

### 7.2 Scaling with Sequence Length

The paper reports a scaling comparison holding both models' architecture fixed and varying $N$ from $2\text{k}$ to $6\text{k}$, measuring NE and QPS regression *relative to each model's own $N=2\text{k}$ configuration* (not relative to each other):

| Sequence length | SlimPer NE | SlimPer QPS regression | Baseline NE | Baseline QPS regression |
|---|---|---|---|---|
| $2\text{k}$ | 0.00 (ref.) | 0% (ref.) | 0.00 (ref.) | 0% (ref.) |
| $6\text{k}$ | $-0.86\%$ | $<20\%$ | $-0.13\%$ | $-48\%$ |

**Key conclusion: SlimPer's relative advantage over the Transformer baseline grows with sequence length, and the compute-complexity gap explains why.** Consider a simplified two-term serving-cost model $C(N) = A + c \cdot N^{p}$, where $A$ captures the $N$-independent portion of inference cost (embedding lookups, dense/sparse paths) and $c \cdot N^p$ the sequence-length-dependent attention term, with $p = 2$ for the quadratic Transformer row and $p = 1$ for SlimPer's linear cross-attention (Eqs. 4–6). Scaling the sequence length by a factor $s = N / N_0$, throughput regression is
$$
1 - \frac{\mathrm{QPS}(sN_0)}{\mathrm{QPS}(N_0)} = 1 - \frac{C(N_0)}{C(sN_0)} = 1 - \frac{A + cN_0^p}{A + c\,s^p N_0^p} \;\xrightarrow{s \gg 1}\; 1 - O(s^{-p}).
$$
For $p=2$ the surviving throughput fraction decays as $s^{-2}$; for $p=1$, only as $s^{-1}$ — so the *gap* between the two regressions grows at least as fast as $s^2 - s = s(s-1)$, i.e. **superlinearly** in the scaling factor $s$.

*This is a heuristic, order-of-magnitude argument, not an exact fit to the reported numbers.* Plugging $s = 3$ (from $N_0 = 2\text{k}$ to $N=6\text{k}$) into the pure quadratic model predicts a regression of $1 - 3^{-2} \approx 88.9\%$ for the Transformer baseline, well above the observed $48\%$; the pure linear model predicts $1 - 3^{-1} \approx 66.7\%$ for SlimPer, far above the observed $<20\%$. The discrepancy is expected: real serving cost has a substantial $N$-independent term $A$ (fixed model components, memory bandwidth, batching overhead) that the observed regressions suggest dominates total cost at these sequence lengths for *both* systems — which is exactly why the *quadratic* term still visibly dominates the Transformer's regression while the *linear* term remains comparatively small for SlimPer. The qualitative conclusion — that the compute-complexity gap ($O(N^2)$ vs. $O(N)$) predicts a widening throughput advantage as $N$ grows — is well supported; the specific decay exponents above should be read as a simplified model, not a precise reproduction of the paper's production benchmarking pipeline.

This matches the reported FLOPs comparison (paper's Table 8, Appendix F):

| Sequence length | Baseline (HSTU) GFLOPs | SlimPer GFLOPs | Ratio |
|---|---|---|---|
| 2048 | $\sim 24$ | $\sim 3$ | $\sim 8\times$ |
| 4096 | $\sim 74$ | $\sim 5$ | $\sim 15\times$ |
| 6144 | $\sim 150$ | $\sim 6$ | $\sim 25\times$ |

The baseline's FLOPs grow by $\sim 3.1\times$ and then $\sim 2.0\times$ across the two doublings — sub-quadratic in practice, likely due to HSTU's own stochastic-length sparsity (cf. [[papers/generative-recommenders/hstu#4.2 Stochastic Length Sparsity|HSTU §4.2]]) — while SlimPer's FLOPs grow only $\sim 1.7\times$ and $\sim 1.2\times$, sub-linearly, consistent with a fixed per-layer overhead increasingly dominating the small $O(N \cdot q)$ term as $N$ grows. Both trends are directionally consistent with, though not an exact numerical match for, the two-term model above.

### 7.3 Ablations

**Knowledge base size** ($K, q, t$), relative to the $K=64$ production configuration:

| $(K, q, t)$ | NE (%) | QPS (%) | Memory (%) |
|---|---|---|---|
| $(64, 16, 32)$ — baseline | 0.0 | 0.0 | 0.0 |
| $(32, 8, 16)$ | $+0.18$ | $+4.25$ | $-1.31$ |
| $(16, 4, 8)$ | $+0.36$ | $+17.8$ | $-7.43$ |
| $(8, 2, 4)$ | $+0.53$ | $+21.2$ | $-8.19$ |
| $(4, 1, 2)$ | $+1.2$ | $+23.2$ | $-9.53$ |

*Positive* NE here means *worse* quality relative to $K=64$. $K=4$ trades a large throughput gain ($+23.2\%$ QPS) for a $1.2\%$ NE degradation — a direct illustration of the capacity/bandwidth tradeoff discussed in [[#5.3 A Tension: Capacity versus Per-Layer Bandwidth|§5.3]]: shrinking $K$ narrows the per-layer channel faster than it saves compute, since QPS gains diminish (from $+23.2\%$ at $K=4$ down to $+4.25\%$ at $K=32$) while NE loss grows roughly monotonically. $K=64$ is the paper's chosen operating point.

**Number of refinement layers**, relative to the $L=7$ baseline:

| Layers $L$ | NE (%) | QPS (%) | Memory (%) |
|---|---|---|---|
| 7 — baseline | 0.0 | 0.0 | 0.0 |
| 3 | $+0.32$ | $+21$ | $-8.60$ |
| 5 | $+0.14$ | $+8.8$ | $-5.50$ |
| 9 | $-0.12$ | $-7.16$ | $+4.92$ |

Quality continues to improve monotonically out to $L=9$, consistent with the iterative-refinement framing: more rounds give the knowledge base more opportunities to retrieve evidence it missed in earlier rounds (§5.2). The paper settles on $L=7$ (Reels) / $L=5$ (Feed) as its quality/throughput operating point rather than the strictly-best-quality $L=9$.

**Contextual encoding** ($\Psi_{\text{context}}$ in Eq. 1): comparing no context, event-type-only context, and full (event-type + event-content) context shows event-type context alone accounts for essentially all of a $\sim -0.15\%$ NE gain on the reshare task, with event content contributing $<0.02\%$ additional improvement — i.e. *knowing what kind of action a neighboring event was* matters far more than *what specific item it involved*.

### 7.4 Online A/B Tests

SlimPer was deployed at full traffic on Reels ($N=5\text{k}$) and Feed ($N=4\text{k}$), yielding statistically significant gains across multiple major engagement metrics, with the paper reporting an aggregate topline impact "roughly $10\times$ that of a typical statistically significant launch," no regressions on ecosystem guardrails (integrity, diversity, recency), and training/inference GPU capacity held approximately neutral. Deployed variants have since scaled to $10\text{k}$+ event sequences in production.

### 7.5 Interpretability

Because Step 2's relevance matching ([[#3.2 Step 2: Explicit Multifaceted Relevance Matching|§3.2]]) produces explicit dot-product scores $\lambda_s, \lambda_e$ against a fixed template set $\mathbf{T}$ (rather than an opaque softmax-weighted sum with no fixed reference points), attention patterns are directly inspectable per layer. The paper reports a hierarchical pattern across depth: lower layers attend broadly across the full multi-thousand-event history, middle layers concentrate on the first few hundred (i.e. most recent) positions, and upper layers narrow further to roughly the most recent 400 events — a coarse-to-fine refinement trajectory consistent with the iterative-refinement design of [[#1.3 SlimPer's Reframing: Iterative Knowledge-Base Refinement|§1.3]].

---

## 8. Limitations

The paper acknowledges two limitations directly:

1. **Baseline choice.** Comparisons are against a mature HSTU + Wukong hybrid rather than more recent unified transformer-style architectures (OneTrans, HHFT, RankMixer); the authors justify this by claiming the hybrid outperforms any single unified architecture they are aware of, but this leaves open whether SlimPer's advantage would persist against a stronger, more recent single-architecture baseline.
2. **Reproducibility.** All experiments use internal Instagram training and evaluation data at volumes not available in public benchmarks, limiting independent reproduction of the reported numbers.

*A further caveat, not raised by the paper itself:* the information-theoretic argument of [[#5. Information-Theoretic Justification|§5]] is, by the paper's own ablations, incomplete as an explanation for why $K=64$ specifically is needed (see [[#5.3 A Tension: Capacity versus Per-Layer Bandwidth|§5.3]]) — the empirical operating point is determined by a quality/throughput tradeoff curve, not derived from the capacity argument.

---

## 9. Relation to HSTU and Wukong

| Aspect | HSTU | Wukong | SlimPer |
|---|---|---|---|
| Core operation | Pointwise (SiLU) self-attention over history | Factorization machine stack over feature set | Fixed-size cross-attention into raw multi-modal tokens |
| Per-layer state size | $O(N)$, grows with history | $O(n_F + n_L)$, grows across layers via stacking | $O(K)$, fixed at $64$ regardless of $N$ |
| Per-layer compute | $O(N^2 d)$ (or $O(N^2/B)$ amortized) | $O(ndh)$ via low-rank FM | $O(N \cdot q \cdot d)$ |
| Candidate amortization | M-FALCON, sequence-only | Not addressed | ROO, all modalities |
| Relation to SlimPer | Explicitly critiqued baseline; SlimPer departs from sequential transduction as the modeling primitive | Its dense/sparse feature-interaction role is absorbed into SlimPer's unified Select-Match-Refine cycle | — |

SlimPer is best read not as a strict extension of either paper but as a **departure from the shared assumption both make** — that the token count processed per layer should scale with $N$. Where [[papers/generative-recommenders/hstu|HSTU]] scales by making attention over a growing sequence as cheap as possible (stochastic length, M-FALCON), and [[papers/generative-recommenders/wukong|Wukong]] scales feature-interaction *order* through depth while holding per-layer feature count roughly fixed, SlimPer's move is to decouple *history length* from *per-layer state size* entirely — treating the $N$-scaling that both prior papers work hard to make cheap as a a cost that need not be paid by the persistent representation at all, only by the (amortizable) act of reading from raw history.

---

## 10. References

| Reference Name | Brief Summary | Link to Reference |
|---|---|---|
| SlimPer: Make Personalization Model Slim and Smart | Main paper: iterative refinement of a fixed-size knowledge base for personalized ranking | https://arxiv.org/abs/2607.12281 |
| Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations | HSTU; the sequential-transduction baseline SlimPer departs from | https://arxiv.org/abs/2402.17152 |
| Wukong: Towards a Scaling Law for Large-Scale Recommendation | FM-stack architecture; the dense/sparse-feature-interaction baseline SlimPer's unified cycle absorbs | https://arxiv.org/abs/2403.02545 |
| Perceiver: General Perception with Iterative Attention (Jaegle et al., 2021) | Introduces asymmetric latent-array cross-attention: $O(NK)$ compute, $O(K)$ carried state — the architectural pattern SlimPer's knowledge base instantiates | https://arxiv.org/abs/2103.03206 |
| Perceiver IO: A General Architecture for Structured Inputs & Outputs (Jaegle et al., 2021) | Extends Perceiver's fixed-size latent bottleneck to structured, multi-modal outputs | https://arxiv.org/abs/2107.14795 |
| Deep Interest Network for Click-Through Rate Prediction (Zhou et al., KDD 2018) | Target-aware local activation unit; conceptual precursor to SlimPer's modality-dependent MLP attention kernel for sparse features | https://arxiv.org/abs/1706.06978 |
| Elements of Information Theory (Cover & Thomas) | Standard reference for the Data Processing Inequality used in §5.2 | https://www.wiley.com/en-us/Elements+of+Information+Theory%2C+2nd+Edition-p-9780471241959 |
