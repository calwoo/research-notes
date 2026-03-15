# ML Systems for Generative Recommenders

*See also [[note|Generative Recommender Systems]] for the algorithmic foundations this note builds upon.*

## Table of Contents

- [[#1. The Systems Challenge|1. The Systems Challenge]]
- [[#2. Distributed Training|2. Distributed Training]]
  - [[#2.1 Heterogeneous Parallelism: Embeddings vs. Dense Layers|2.1 Heterogeneous Parallelism: Embeddings vs. Dense Layers]]
  - [[#2.2 Ragged Attention Kernels|2.2 Ragged Attention Kernels]]
  - [[#2.3 Sparse Embedding Gradients|2.3 Sparse Embedding Gradients]]
  - [[#2.4 Incremental Checkpointing|2.4 Incremental Checkpointing]]
- [[#3. Inference Serving|3. Inference Serving]]
  - [[#3.1 The Prefill and Decode Asymmetry|3.1 The Prefill and Decode Asymmetry]]
  - [[#3.2 KV Cache Reuse for Returning Users|3.2 KV Cache Reuse for Returning Users]]
  - [[#3.3 M-FALCON and Candidate Amortization|3.3 M-FALCON and Candidate Amortization]]
  - [[#3.4 Semantic ID Beam Search: The Hourglass Problem|3.4 Semantic ID Beam Search: The Hourglass Problem]]
  - [[#3.5 Long-History Compression: VISTA|3.5 Long-History Compression: VISTA]]
- [[#4. Embedding Infrastructure|4. Embedding Infrastructure]]
  - [[#4.1 Memory Hierarchy for Embedding Tables|4.1 Memory Hierarchy for Embedding Tables]]
  - [[#4.2 Dynamic Vocabulary Management|4.2 Dynamic Vocabulary Management]]
  - [[#4.3 RQ-VAE Codebook Lifecycle|4.3 RQ-VAE Codebook Lifecycle]]
- [[#5. Online Learning and Model Freshness|5. Online Learning and Model Freshness]]
  - [[#5.1 The Freshness Hierarchy|5.1 The Freshness Hierarchy]]
  - [[#5.2 Parameter-Efficient Daily Updates|5.2 Parameter-Efficient Daily Updates]]
  - [[#5.3 Continual Learning Without Catastrophic Forgetting|5.3 Continual Learning Without Catastrophic Forgetting]]
- [[#6. References|6. References]]

---

## 1. The Systems Challenge

The algorithmic design of generative recommenders (covered in [[note|note.md]]) creates a distinct set of infrastructure problems that do not arise in either standard DLRM training or LLM serving. Three structural properties drive the difficulty:

**Heterogeneous computation.** A generative recommender combines two qualitatively different components: a massive *embedding layer* (parameter count proportional to item inventory, $|\mathcal{I}| \sim 10^9$, dominated by memory bandwidth) and a *Transformer backbone* (compute-intensive, amenable to tensor parallelism). Standard 3D parallelism for LLMs assumes uniform computation per layer and fails on this heterogeneous workload.

**Ragged sequences.** User interaction histories follow a power-law length distribution — most users have tens of interactions, a small fraction have millions. Padding all sequences to uniform length wastes $O(L_{\max}^2)$ attention compute when $L_{\max} \gg$ the typical length. Efficient training requires native support for variable-length (ragged) tensor operations.

**Non-stationary item vocabulary.** New items appear daily; old items expire. The item embedding table must grow and the retrieval index must be rebuilt without full model retraining. Serving infrastructure must handle zero-shot embedding of new items and frequent index refresh while maintaining sub-10ms P99 latency.

---

## 2. Distributed Training

### 2.1 Heterogeneous Parallelism: Embeddings vs. Dense Layers

**The fundamental mismatch.** In LLM training, all layers have the same parameter type (dense weight matrices) and similar compute profiles. Standard 3D parallelism (tensor × pipeline × data) assumes this uniformity. Recommendation models violate it:

- *Embedding layers*: $|\mathcal{I}| \times d$ parameters, each accessed only for items present in the current batch. Gradients are structured-sparse: in a batch of $B$ users each seeing $k$ items, at most $Bk$ of the $|\mathcal{I}|$ embedding rows receive non-zero gradient. AllReduce — which forces dense gradient tensors — wastes bandwidth proportional to $|\mathcal{I}| / (Bk)$ on zero-gradient rows.
- *Dense Transformer layers*: fully dense gradients, uniform compute, well-served by standard AllReduce with NCCL.

**The hybrid parallelism solution.** Production systems (ZionEX, Meta 2021; Monolith, ByteDance 2022) adopt a *hybrid* strategy that treats the two components differently:

| Component | Parallelism | Communication | Update schedule |
|---|---|---|---|
| Embedding tables | Row-wise model parallel sharding | AlltoAll (butterfly pattern) | Async or sync |
| Transformer backbone | Data parallel replicas | AllReduce (NCCL) | Synchronous |

**Sharding.** With $D$ devices, device $k$ owns embedding rows $\{i : i \bmod D = k\}$. During the forward pass, each worker sends its batch's embedding lookup indices to the owning device and receives the corresponding embeddings — a *personalized AlltoAll* with communication volume proportional to the number of distinct items in the batch, not $|\mathcal{I}|$.

**ZionEX** (Meta, arXiv:2104.05158) eliminates the parameter server entirely by making embedding AlltoAll synchronous, achieving **40× speedup** over parameter-server-based baselines on 12-trillion-parameter models using custom RDMA/RoCE networking between nodes.

**Monolith** (ByteDance, arXiv:2209.07663) takes the opposite direction for online training: maintain a *hash set of touched embedding keys* per training step and synchronize only updated rows at minute intervals, decoupling the dense parameter update cadence from the embedding update cadence for real-time streaming scenarios.

### 2.2 Ragged Attention Kernels

**The padding waste.** In a batch of users with sequence lengths $n_1, n_2, \ldots, n_B$, padding all to $N = \max_i n_i$ wastes:

$$\text{wasted FLOPs} = O\!\left(B N^2 d - \sum_i n_i^2 d\right)$$

Under a power-law length distribution, $\sum_i n_i^2 \ll B N^2$, so wasted compute can dominate.

**Ragged (jagged) tensor format.** Sequences are stored as a single flat concatenated tensor with an auxiliary offsets array:

$$\mathbf{X}_{\text{flat}} = [\mathbf{x}^{(1)}_1, \ldots, \mathbf{x}^{(1)}_{n_1},\ \mathbf{x}^{(2)}_1, \ldots, \mathbf{x}^{(2)}_{n_2},\ \ldots], \quad \text{offsets} = [0, n_1, n_1 + n_2, \ldots, \sum_i n_i]$$

Custom CUDA kernels use the offsets to mask attention computations so queries cannot attend across sequence boundaries. Memory complexity reduces from $O(B N^2 d)$ to $O(\sum_i n_i^2 d)$ — proportional to actual sequence lengths.

**Jagged Flash Attention** (arXiv:2409.15373) integrates this ragged format with FlashAttention's IO-aware tiling strategy. On variable-length recommendation batches it achieves up to **9× memory reduction** and **3× throughput improvement** over dense FlashAttention-2. HSTU's production implementation achieves **5.3–15.2× speedup** over FlashAttention-2-based Transformers at sequence length 8,192 through this combination of ragged kernels and stochastic length sparsity.

**Context Parallelism for ragged sequences** (arXiv:2508.04711) extends sequence-length parallelism — already used in LLM training — to jagged tensors, further splitting each user's sequence across GPUs. This enables training on sequences **5.3× longer** than single-GPU memory allows, with a **1.55× scaling factor** when composed with data parallelism.

### 2.3 Sparse Embedding Gradients

In a batch, only embeddings for items actually seen receive non-zero gradients. For a catalog of $|\mathcal{I}| = 10^9$ items and a batch touching $B k \sim 10^6$ distinct items, the gradient is 0.1% dense. This sparsity has critical implications for distributed training.

**Why AllReduce fails.** AllReduce operates on dense tensors — NCCL's ring-allreduce has no sparse mode. Using it for embedding gradients forces $O(|\mathcal{I}| d)$ communication volume regardless of batch sparsity. For a $10^9 \times 128$-dim embedding table in float16, this is 256GB per step across all workers.

**HET** (arXiv:2112.07221) exploits the power-law popularity distribution of items: a small fraction of items account for most gradient updates. HET maintains a *distributed embedding cache* for hot items (updated in-place locally) and synchronizes only cold embeddings via AlltoAll. This achieves **88% communication reduction** and **20.68× throughput improvement** over baseline distributed embedding training.

**The EmbRace hybrid** (arXiv:2110.09132) combines AlltoAll (for sparse embedding communication) with model-parallel scheduling to overlap embedding communication with dense layer computation, achieving a further **2.41× speedup** over state-of-the-art.

### 2.4 Incremental Checkpointing

At 1.5T parameters, writing a full checkpoint costs hundreds of gigabytes and significant I/O bandwidth. Since most embedding rows are unchanged between checkpoints (sparse gradient updates), full checkpoints are wasteful.

**Check-N-Run** (Facebook, arXiv:2010.08679) maintains a *dirty bit* per embedding row: only rows that received a gradient update in the current training window are written to the checkpoint. Combined with quantization of stored embedding values, this achieves **6–17× write bandwidth reduction** and **2.5–8× checkpoint size reduction** in production Facebook recommendation models.

---

## 3. Inference Serving

### 3.1 The Prefill and Decode Asymmetry

A request to a generative recommender has two computationally distinct phases:

**Prefill:** Process the user's full interaction history $(x_0, a_0, \ldots, x_{n-1}, a_{n-1})$ to produce the hidden state $u_{n-1} \in \mathbb{R}^d$. This is a batch matrix multiply — highly parallel, compute-bound, near-peak FLOP utilization on modern GPUs.

**Decode:** Generate the recommendation (a semantic ID token sequence, or a scored candidate list). For generative retrieval this is sequential beam search; for ranking it is a single scored forward pass per candidate. The bottleneck is GPU *memory bandwidth* (loading KV cache weights), not arithmetic throughput.

These profiles are incompatible: mixing prefill and decode requests in the same batch causes prefill to stall on decode's memory-bandwidth bottleneck and decode to wait for prefill's large compute kernel. **Disaggregated prefill-decode** separates them onto dedicated GPU clusters:

- *Prefill workers*: large batch size, high throughput, optimized for matrix multiply
- *Decode workers*: low batch size, low latency, optimized for memory bandwidth

Disaggregated serving achieves up to **4.48× goodput improvement** or **10.2× tighter latency SLO** vs. mixed batching. **PRISM** (NSDI 2025) demonstrates this architecture at production scale for DLRM-class recommendation models on GPU clusters.

### 3.2 KV Cache Reuse for Returning Users

**The repeated prefill problem.** A returning user whose history has not changed since their last request would have their history encoded identically every time. The prefill cost $O(n^2 d)$ is paid in full on every request even when the user's sequence is unchanged.

**Persistent KV cache.** The attention key-value tensors $(\mathbf{K}, \mathbf{V}) \in \mathbb{R}^{n \times d}$ produced during prefill are stored in a distributed cache (Redis/Memcached tier) keyed by user ID and sequence hash. On a cache hit, the prefill step is skipped entirely and the cached tensors are loaded directly into GPU SRAM.

The cache invalidation policy is straightforward: when the user has a new interaction (new item in history), the cached KV tensors are stale and must be recomputed. In practice, the sequence changes with every new interaction, but the *prefix* of the sequence may be unchanged — *prefix caching* stores the KV cache for a common prefix (e.g., the user's history from 7 days ago) and only recomputes the recent suffix.

On systems with high user return rates, persistent KV caching achieves approximately **2× reduction in time-to-first-token (TTFT)** for returning users.

### 3.3 M-FALCON and Candidate Amortization

The HSTU paper introduces *M-FALCON* (Microbatch FALCON) to amortize the user history encoding cost across multiple ranking candidates. This is covered in [[note#4.3 M-FALCON: Inference Amortization|the algorithmic note]]; the systems perspective is:

- User history KV cache computed once: $O(n^2 d)$
- Each candidate $x_c$ appended to sequence and scored: $O(nd)$ incremental cost
- Total for $b_m$ candidates: $O(n^2 d + b_m n d)$

Despite HSTU being **285× more computationally complex** than the DLRM baseline, M-FALCON achieves **1.5–2.99× higher QPS** when scoring 1,024–16,384 candidates per request by amortizing the $O(n^2 d)$ prefix computation. Combined with persistent KV caching, the $O(n^2 d)$ term can be eliminated entirely for returning users.

### 3.4 Semantic ID Beam Search: The Hourglass Problem

Generative retrieval (TIGER, COBRA) decodes the next item's semantic ID $(c_1, c_2, \ldots, c_K)$ autoregressively via beam search over the RQ-VAE code tree. An underappreciated pathology appears in this decoding process.

**Definition (Hourglass Phenomenon).** In residual quantization, the intermediate codebook levels ($k = 2, \ldots, K-1$) receive residuals $\mathbf{r}_k$ that are increasingly concentrated — the first level captures most of the variance, leaving residuals with low magnitude. Codes at intermediate levels cluster around a small region of the codebook, causing severe *hourglass narrowing*: beam diversity collapses at intermediate levels even when beams diverge at level 1. The recovered candidate set after $K$ decode steps is far smaller than the beam width suggests.

**Mitigation strategies:**

1. *Constrained beam search via trie.* Build a prefix trie over all valid item codes. At each decode step, restrict the vocabulary to valid next codewords — those leading to at least one real item. This prevents wasted beam slots on invalid code prefixes and ensures all $B$ beams yield retrievable items.

2. *Parallel semantic ID generation* (arXiv:2506.05781). Train the decoder with a multi-token prediction objective so that multiple ID tokens are generated in a single forward pass rather than sequentially. Reduces the number of autoregressive steps from $K$ to 1–2, trading sequential latency for parallel compute.

3. *Random Last Level* (Liang et al., GR2 2026). As discussed in [[note#4.1 Semantic IDs via RQ-VAE: TIGER|the algorithmic note]], assigning the final quantization level randomly recovers uniqueness at the cost of fine-grained reconstruction fidelity.

### 3.5 Long-History Compression: VISTA

For users with histories of $10^5$–$10^6$ interactions, even $O(n^2 d)$ with $n = 10^5$ is infeasible at serving time (a single forward pass would require $10^{10}$ operations). HSTU's stochastic length truncation addresses this during *training* but not necessarily inference.

**VISTA** (arXiv:2510.22049) addresses inference scalability via *two-stage target attention*:

1. *History summarization.* Compress the full interaction history of length $n$ into a fixed-size summary $\mathbf{S} \in \mathbb{R}^{m \times d}$ with $m \ll n$ via learnable pooling. Cost: $O(nm d)$, once per user.
2. *Candidate attention over summary.* Each candidate item $x_c$ attends only over $\mathbf{S}$, not the full history. Cost: $O(m d)$ per candidate — constant in $n$.

Total serving cost: $O(nm d + b_m m d)$, independent of history length after the one-time summarization. Deployed in production at billion-user scale with significant offline and online metric improvements.

**GEMs** (arXiv:2602.13631) proposes a complementary *multi-stream decoder* that decomposes the user history into parallel behavioral streams (e.g., by action type: clicks, watches, purchases) and processes each stream independently before fusing. This reduces the effective attention sequence length and allows longer histories without quadratic cost.

---

## 4. Embedding Infrastructure

### 4.1 Memory Hierarchy for Embedding Tables

A $10^9 \times 128$-dimensional float16 embedding table requires 256GB — far exceeding GPU HBM capacity (80GB on H100). Production systems tier embedding storage across the memory hierarchy:

| Tier | Capacity | Bandwidth | Access latency |
|---|---|---|---|
| GPU HBM | 40–80 GB | ~3 TB/s | ~10 ns |
| CPU DRAM | 1–8 TB | ~200 GB/s | ~100 ns |
| NVMe SSD | 10–100 TB | ~10 GB/s | ~100 μs |

**TorchRec 2D embedding parallelism** (Meta) provides *table-wise, row-wise, and column-wise* sharding options with automatic selection of the optimal strategy per table based on access patterns and device topology. The *Table Batched Embedding (TBE)* operator batches all embedding lookups in a forward pass into a single CUDA kernel call, amortizing kernel launch overhead and improving memory access locality.

**GPU Memory Tiering (GMT)** (2024) enables GPUs to orchestrate their own multi-tier memory via *GPUDirect Storage*, bypassing CPU in the data path. Hot embeddings reside in HBM; warm embeddings in CPU DRAM; cold embeddings on NVMe SSD. Async prefetching pipelines storage latency behind GPU computation. Empirically: **3.9× latency reduction** and **68% throughput improvement** for cold-embedding workloads.

**Frequency-aware caching.** Since item popularity follows a power law, the top-1% of items account for the majority of embedding accesses. Static frequency-aware caches keep hot embeddings in HBM; cold embeddings are fetched from lower tiers on demand. HET (Section 2.3) extends this principle to the distributed setting.

### 4.2 Dynamic Vocabulary Management

On creator-economy platforms, $O(10^6)$ new items are created daily and a comparable number expire. This creates two problems: (1) new items have no trained embedding, and (2) the embedding table must grow without triggering full retraining.

**Hash-backed dynamic tables.** Rather than a dense pre-allocated table, use a hash map keyed by item ID. New items are inserted on-demand; expired items are evicted by a TTL policy. *Monolith* implements this as a *cuckoo hash map* with collision-free guarantees: each item has a unique slot, eliminating the embedding collision problem that plagues fixed-size hash tables. *Expirable embeddings* allow automatic eviction of items not seen in recent training batches, keeping the table from growing unboundedly.

**Cold-start warm-starting.** New items have no interaction history to learn from. Embedding initialization strategies in increasing order of sophistication:

1. *Random initialization*: simplest; the embedding converges only once the item accumulates interactions.
2. *Average pooling of similar items*: initialize the new item's embedding as the mean of embeddings of semantically similar items (by content features or category).
3. *Content-feature bridging*: train a separate encoder $f_{\text{content}} : \text{text/image features} \to \mathbb{R}^d$ and initialize $\mathbf{e}_{\text{new}} = f_{\text{content}}(\text{attributes}_{\text{new}})$. Meta Scaling and Shifting Networks (arXiv:2105.04790) learn item-specific affine transformations of a shared content embedding to produce task-specific item representations without interaction history.

### 4.3 RQ-VAE Codebook Lifecycle

The RQ-VAE tokenizer (Section 4.1 of [[note|note.md]]) maps items to semantic IDs. As new items arrive, two questions arise: (1) can a trained codebook generalize to new items, and (2) how often must the codebook be retrained?

**Codebook generalization.** YouTube/Google (arXiv:2306.08121) froze an RQ-VAE trained on older data and evaluated it on items created subsequently. Performance was comparable to using a fresh codebook — *frozen codebooks generalize without retraining*. The intuition: the codebook encodes semantic clustering structure (which items are similar) that remains stable even as specific items enter and leave. New items are mapped to the nearest existing code cluster, which is semantically appropriate.

**Codebook retraining.** When the item distribution shifts substantially (new content categories, major platform changes), the codebook should be retrained. **SOLID** (2024) proposes *dynamic semantic codebook learning* that jointly optimizes the codebook with the recommendation model, continuously updating code assignments as user-item interaction patterns evolve. **RQ-Kmeans** (OneRec v2, Kuaishou 2025) incorporates collaborative signals — co-interacted item pairs — directly into codebook construction, ensuring semantically similar items in the embedding space also receive similar codes.

---

## 5. Online Learning and Model Freshness

### 5.1 The Freshness Hierarchy

Generative recommender models degrade rapidly after their training cutoff: GR2 (arXiv:2602.07774) documents significant quality degradation within **24 hours** of training cutoff. A 1.5T parameter model cannot be fully retrained daily — the compute cost is prohibitive. The solution is a *freshness hierarchy* that matches the refresh mechanism to the timescale of the underlying distribution shift:

| Layer | What it captures | Refresh frequency | Mechanism |
|---|---|---|---|
| Real-time features | Platform-wide trends (CTR last hour, velocity, seasonal) | Minutes | Streaming feature computation; no model update |
| Retrieval index | New item embeddings and ANN structure | Daily to 4× daily | Candidate tower inference + index rebuild |
| LoRA adapters | Recent user preference drift, new item interactions | Daily | Fine-tune only low-rank adapter matrices |
| Base model | Long-range behavioral patterns, item co-occurrence | Weekly to monthly | Full retraining with warm-start |

**The feature vs. model freshness insight.** GenRank (arXiv:2505.04180) finds that *real-time statistical features* — sliding-window CTR by category, trending scores, item velocity — provide significant AUC improvement even in a generative model that was not designed around them. These features capture what is trending *right now* on the platform, independent of the model's training-time knowledge. *This means a stale base model can be partially compensated by fresh features*, buying more time between full retraining cycles.

### 5.2 Parameter-Efficient Daily Updates

**LoRA** (arXiv:2106.09685) provides the most practical mechanism for daily model updates. Rather than retraining all parameters, inject trainable *rank-$r$ decomposition matrices* into each Transformer layer:

$$W_{\text{updated}} = W_{\text{frozen}} + \Delta W, \quad \Delta W = W_{\uparrow} W_{\downarrow}$$

where $W_{\uparrow} \in \mathbb{R}^{d \times r}$, $W_{\downarrow} \in \mathbb{R}^{r \times d}$, and $r \ll d$. For a 1.5T model with $d = 768$ and $r = 8$, the trainable parameter count reduces by **10,000×**. Daily LoRA training requires a fraction of the GPU resources needed for full retraining; the resulting adapter can be merged with the frozen base weights at serving time with zero inference overhead.

**Prompt-based Continual Learning (PCL)** (arXiv:2502.19628) takes the freeze further: the base model weights are entirely frozen and per-task *learnable prompt vectors* are prepended to the input sequence. Each task (click prediction, watch-time prediction, purchase prediction) gets its own prompt; the model reads the prompt and adapts its behavior without weight updates. This is compatible with any frozen large model and requires only the prompt parameters to be stored and updated per day.

**Warm-starting and knowledge distillation.** Naive warm-start from the previous checkpoint can cause the model to overfit to outdated patterns. *Adaptive Knowledge Distillation (AdaKD)* blends new and old supervision:

$$\mathcal{L}_{\text{AdaKD}} = \alpha\, \mathcal{L}_{\text{new}} + (1 - \alpha)\, D_{\text{KL}}\!\left(\pi_{\text{new}}(\cdot) \,\|\, \pi_{\text{old}}(\cdot)\right)$$

where $\alpha$ is scheduled from 0 (trust the old model) to 1 (trust only new data) over the update window. This prevents catastrophic forgetting of well-learned patterns while adapting to the new distribution.

### 5.3 Continual Learning Without Catastrophic Forgetting

When LoRA or full retraining is applied to new data, the model may forget patterns learned on older data — the *catastrophic forgetting* problem.

**Elastic Weight Consolidation (EWC)** (arXiv:1612.00796) adds a regularization penalty to the loss function:

$$\mathcal{L}_{\text{EWC}} = \mathcal{L}_{\text{new}}(\theta) + \frac{\lambda}{2} \sum_i F_i \left(\theta_i - \theta^*_i\right)^2$$

where $F_i$ is the diagonal of the *Fisher Information Matrix* — a measure of how important parameter $\theta_i$ is to the previous task. Parameters with high $F_i$ are constrained to stay close to their previous values $\theta^*_i$; parameters with low $F_i$ can drift freely. In knowledge graph continual learning settings, EWC reduces catastrophic forgetting from 12.62% to 6.85%.

*Limitation*: Computing $F_i$ requires access to old task data or a full backward pass over historical interactions — expensive at recommendation scale.

**Experience replay.** Maintain a buffer of historical (user, item, label) triples sampled from past training windows. At each training step, mix replay samples with fresh data:

$$\mathcal{L} = \mathcal{L}_{\text{fresh}} + \beta\, \mathcal{L}_{\text{replay}}$$

The buffer acts as a compressed representation of the old data distribution, preventing the model from drifting too far from learned historical patterns. *Adaptive Experience Replay (AdaER)* prioritizes replay of examples that caused the most interference in recent training steps, directing the rehearsal budget toward the most forgotten patterns.

**Distribution drift detection.** *ADWIN* (Adaptive WINdowing) maintains a dynamic sliding window of recent model errors. If the error distribution in a recent sub-window deviates significantly from the full window, a drift alarm is triggered, prompting a faster model update or learning rate reset. Applied to recommendation systems, ADWIN-triggered updates improved accuracy by **15%** in e-commerce settings (compared to fixed-schedule updates) by detecting distribution shifts earlier.

---

## 6. References

| Reference Name | Brief Summary | Link |
|---|---|---|
| ZionEX: Software-Hardware Co-design for Fast and Scalable Training (Meta, 2021) | Synchronous hybrid parallelism for 12T-param DLRM; eliminates parameter server; 40× speedup | https://arxiv.org/abs/2104.05158 |
| Monolith: Real Time Recommendation System with Collisionless Embedding Table (ByteDance, 2022) | Cuckoo hash for collision-free online embeddings; sparse synchronization of touched keys; expirable embeddings | https://arxiv.org/abs/2209.07663 |
| Jagged Flash Attention (arXiv:2409.15373, 2024) | Custom CUDA kernels for ragged/variable-length attention; 9× memory reduction, 3× throughput over dense FlashAttention-2 | https://arxiv.org/abs/2409.15373 |
| Scaling Generative Recommendations with Context Parallelism on HSTU (Meta, 2025) | Extends sequence-length parallelism to ragged tensors; 5.3× longer sequences, 1.55× scaling with DDP | https://arxiv.org/abs/2508.04711 |
| HET: Scaling out Huge Embedding Model Training (2021) | Power-law-aware distributed embedding cache; 88% communication reduction, 20.68× throughput improvement | https://arxiv.org/abs/2112.07221 |
| EmbRace: Accelerating Sparse Communication for Distributed Training (2021) | Sparsity-aware hybrid AlltoAll + model-parallel scheduling; 2.41× speedup over SOTA | https://arxiv.org/abs/2110.09132 |
| Check-N-Run: Checkpointing for Deep Learning Recommendation Models (Facebook, 2020) | Dirty-bit incremental checkpointing; 6–17× write bandwidth reduction, 2.5–8× size reduction | https://arxiv.org/abs/2010.08679 |
| PRISM: GPU-Disaggregated Serving for Deep Learning Recommendation (NSDI 2025) | GPU-disaggregated prefill/decode architecture for production DLRM serving | https://www.lingyunyang.com/assets/files/prism-nsdi25.pdf |
| Massive Memorization with VISTA (arXiv:2510.22049, 2024) | Two-stage target attention for lifelong histories; fixed-size summary enables O(1) candidate scoring in history length | https://arxiv.org/abs/2510.22049 |
| GEMs: Multi-Stream Decoder for Long-Sequence Recommendation (arXiv:2602.13631, 2026) | Decomposes history into parallel behavioral streams; reduces effective attention sequence length | https://arxiv.org/abs/2602.13631 |
| Breaking the Hourglass Phenomenon in Residual Quantization (arXiv:2407.21488, 2024) | Identifies and mitigates codebook concentration in intermediate RQ-VAE levels during beam search | https://arxiv.org/abs/2407.21488 |
| Generating Long Semantic IDs in Parallel (arXiv:2506.05781, 2025) | Multi-token prediction for parallel semantic ID decoding; reduces sequential beam search steps | https://arxiv.org/abs/2506.05781 |
| Better Generalization with Semantic IDs (Google/YouTube, arXiv:2306.08121, 2023) | Demonstrates frozen RQ-VAE codebooks generalize to new items without retraining | https://arxiv.org/abs/2306.08121 |
| GMT: GPU Orchestrated Memory Tiering (2024) | GPU-direct 3-tier HBM/DRAM/NVMe hierarchy for embedding tables; 3.9× latency reduction | https://dl.acm.org/doi/10.1145/3620666.3651353 |
| Learning to Warm Up Cold Item Embeddings (arXiv:2105.04790, 2021) | Meta Scaling and Shifting Networks for zero-interaction item embedding initialization | https://arxiv.org/abs/2105.04790 |
| LoRA: Low-Rank Adaptation of Large Language Models (arXiv:2106.09685, 2021) | 10,000× parameter reduction for fine-tuning; zero inference overhead via weight merging | https://arxiv.org/abs/2106.09685 |
| PCL: Prompt-based Continual Learning for User Modeling (arXiv:2502.19628, 2025) | Freezes base model; learns per-task prompt vectors as external memory for daily updates | https://arxiv.org/abs/2502.19628 |
| Elastic Weight Consolidation (arXiv:1612.00796, 2017) | Fisher Information Matrix regularization to prevent catastrophic forgetting during continual learning | https://arxiv.org/abs/1612.00796 |
