# Historical Development of Recommender Systems

*A survey of the intellectual lineage from collaborative filtering to generative recommenders. For formal architecture derivations, see [[note|Generative Recommender Systems]]. For production systems, see [[systems|ML Systems for Generative Recommenders]].*

## Table of Contents

- [[#Era I — Collaborative Filtering and Matrix Factorization (1992–2012)|Era I — Collaborative Filtering and Matrix Factorization (1992–2012)]]
  - [[#1.1 Memory-Based Collaborative Filtering|1.1 Memory-Based Collaborative Filtering]]
  - [[#1.2 The Netflix Prize and Model-Based CF|1.2 The Netflix Prize and Model-Based CF]]
  - [[#1.3 Factorization Machines|1.3 Factorization Machines]]
  - [[#1.4 What CF Cannot Capture|1.4 What CF Cannot Capture]]
- [[#Era II — The Deep Learning Turn (2013–2017)|Era II — The Deep Learning Turn (2013–2017)]]
  - [[#2.1 Embedding Everything|2.1 Embedding Everything]]
  - [[#2.2 Wide & Deep and the Two-Tower Paradigm|2.2 Wide & Deep and the Two-Tower Paradigm]]
  - [[#2.3 Deep Interest Networks: The First Crack|2.3 Deep Interest Networks: The First Crack]]
  - [[#2.4 The DLRM Architecture|2.4 The DLRM Architecture]]
- [[#Era III — Sequential Modeling (2016–2022)|Era III — Sequential Modeling (2016–2022)]]
  - [[#3.1 Session-Based Recommendation with RNNs|3.1 Session-Based Recommendation with RNNs]]
  - [[#3.2 SASRec: Self-Attention Comes to Recommendation|3.2 SASRec: Self-Attention Comes to Recommendation]]
  - [[#3.3 BERT4Rec and Bidirectional Pretraining|3.3 BERT4Rec and Bidirectional Pretraining]]
  - [[#3.4 Why Sequential DLRMs Still Hit a Wall|3.4 Why Sequential DLRMs Still Hit a Wall]]
- [[#Era IV — The Generative Turn (2023–Present)|Era IV — The Generative Turn (2023–Present)]]
  - [[#4.1 Quality Saturation Crystallizes the Problem|4.1 Quality Saturation Crystallizes the Problem]]
  - [[#4.2 Generative Retrieval: TIGER and Semantic IDs|4.2 Generative Retrieval: TIGER and Semantic IDs]]
  - [[#4.3 HSTU: Scaling Laws Arrive in Recommendation|4.3 HSTU: Scaling Laws Arrive in Recommendation]]
  - [[#4.4 Production Refinements|4.4 Production Refinements]]
  - [[#4.5 Reasoning at the Top: GR2|4.5 Reasoning at the Top: GR2]]
- [[#The Through-Line|The Through-Line]]
- [[#References|References]]

---

## Era I — Collaborative Filtering and Matrix Factorization (1992–2012)

### 1.1 Memory-Based Collaborative Filtering

The recommender systems field effectively begins with the 1992 Tapestry system at Xerox PARC (Goldberg et al.), which introduced *collaborative filtering* as a concept: use the opinions of many users to filter information for a given user, based on the assumption that users who agreed in the past are likely to agree again.

The core idea hardened into two forms:

**User-based CF.** To predict user $u$'s rating for item $i$, find the $k$ most similar users to $u$ who have rated $i$, and aggregate their ratings:

$$\hat{r}_{ui} = \bar{r}_u + \frac{\sum_{v \in N_k(u)} \text{sim}(u, v)\,(r_{vi} - \bar{r}_v)}{\sum_{v \in N_k(u)} |\text{sim}(u, v)|}$$

Similarity was typically Pearson correlation or cosine similarity over the shared item space.

**Item-based CF.** Amazon's 2003 deployment (Linden et al.) flipped the perspective: instead of finding similar users, find items similar to those the target user has already rated. Item-item similarities are more stable than user-user similarities (items don't change their "personality"; users do), and can be pre-computed offline. This insight drove Amazon's recommender engine to industrial scale.

> [!INFO] Why collaborative filtering works
> CF exploits the *low effective dimensionality* of taste: users differ along a small number of latent preference axes (action vs. drama, price-sensitive vs. premium, etc.), even if the item catalog is enormous. A user who likes items $A$ and $B$ is likely to like $C$ if other users with the same A-B preference pattern also liked $C$ — without needing to know *what* A, B, or C actually are.

**Limits.** Memory-based CF has three structural problems that motivate everything that follows: (1) *sparsity* — most user-item pairs are unobserved, and similarity estimates are unreliable in sparse data; (2) *scalability* — $k$-NN search over $|\mathcal{U}|$ users at query time is expensive for large catalogs; and (3) *cold start* — new users and items have no history to establish similarity.

### 1.2 The Netflix Prize and Model-Based CF

The 2006 Netflix Prize ($1M for a 10% improvement in RMSE over Netflix's Cinematch baseline) catalyzed the development of *model-based* collaborative filtering. The key advance was *Singular Value Decomposition* (Simon Funk's blog post, November 2006; Koren et al., 2009): decompose the observed rating matrix $R \in \mathbb{R}^{|\mathcal{U}| \times |\mathcal{I}|}$ as a product of low-rank factors:

$$\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{p}_u^\top \mathbf{q}_i$$

where $\mu$ is the global mean, $b_u$ is a user bias, $b_i$ is an item bias, and $\mathbf{p}_u, \mathbf{q}_i \in \mathbb{R}^f$ are latent factor vectors. Minimizing the regularized squared error over observed ratings:

$$\min_{\mathbf{p}, \mathbf{q}, b} \sum_{(u,i) \in \mathcal{K}} \left(r_{ui} - \hat{r}_{ui}\right)^2 + \lambda\!\left(\|\mathbf{p}_u\|^2 + \|\mathbf{q}_i\|^2 + b_u^2 + b_i^2\right)$$

trains both user and item representations jointly from data, without handcrafting any features. The winning solution (BellKor's Pragmatic Chaos) combined over 100 such models, but matrix factorization was the engine.

> [!NOTE] SVD vs. "SVD" in recsys
> In recommendation literature, "SVD" typically refers not to the true matrix SVD (which requires filling missing entries) but to Simon Funk's stochastic gradient descent over observed entries — a factorization that avoids the explicit decomposition. The naming is historical and slightly misleading.

**ALS.** Alternating Least Squares (Hu et al., 2008) provided an important variation suited to *implicit feedback* (clicks, watches, purchases) rather than explicit ratings. ALS treats unobserved interactions as negative evidence with low confidence, and alternates between solving closed-form updates for user factors (holding item factors fixed) and vice versa. This gave an efficient algorithm for binary/count data, which is closer to the industrial setting than explicit star ratings.

### 1.3 Factorization Machines

Steffen Rendle's *Factorization Machines* (FM, ICDM 2010) generalized matrix factorization to work over arbitrary feature vectors rather than just user and item IDs. For a feature vector $\mathbf{x} \in \mathbb{R}^n$, an FM of degree 2 models:

$$\hat{y}(\mathbf{x}) = w_0 + \sum_{i=1}^n w_i x_i + \sum_{i=1}^n \sum_{j=i+1}^n \langle \mathbf{v}_i, \mathbf{v}_j \rangle\, x_i x_j$$

where each feature $i$ is associated with a latent vector $\mathbf{v}_i \in \mathbb{R}^k$. The pairwise interaction $\langle \mathbf{v}_i, \mathbf{v}_j \rangle$ can be computed in $O(nk)$ rather than $O(n^2 k)$ via the identity:

$$\sum_{i < j} \langle \mathbf{v}_i, \mathbf{v}_j \rangle\, x_i x_j = \frac{1}{2}\left[\left\|\sum_i x_i \mathbf{v}_i\right\|^2 - \sum_i x_i^2 \|\mathbf{v}_i\|^2\right]$$

This allowed recommendation models to incorporate *any* input features (user demographics, item categories, context) while still capturing interaction effects between them. FMs became the go-to tool for CTR prediction in display advertising and remain a component of many modern architectures.

### 1.4 What CF Cannot Capture

By 2012, collaborative filtering and its descendants handled static rating/click prediction well. Three phenomena remained structurally difficult:

**Temporal dynamics.** User preferences evolve over time — someone binge-watches a genre for a week, then moves on. Static latent factor models have no mechanism to represent how preferences change within or between sessions.

**Sequential dependency.** The order of interactions carries meaning. A user who bought a camera is likely to want a memory card next, independent of long-term preference structure. CF models treat interactions as an exchangeable set, losing this sequential signal.

**Content semantics.** Two items can be semantically similar (both crime thrillers) without sharing interaction history on a new platform. CF cannot bootstrap recommendations for items with no co-interaction signal — the cold-start problem is fundamental to the paradigm.

---

## Era II — The Deep Learning Turn (2013–2017)

### 2.1 Embedding Everything

The key insight from the deep learning era is that item and user IDs can be treated as discrete tokens and learned as dense embeddings — the same way words are embedded in word2vec (Mikolov et al., 2013). Item2vec (Barkan & Koenigstein, 2016) applied the skip-gram objective directly to item co-occurrence in user histories: items that appear in similar contexts (nearby in purchase sequences, or bought together) receive similar embeddings.

This shift had a subtle but important effect on the field's mental model: items were no longer just row/column indices in a matrix — they were *points in a semantic space* where proximity meant something. Neural networks could now learn to project items and users into a shared space and compute compatibility scores via dot products or more complex functions.

> [!INFO] 💡 From IDs to embeddings
> The embedding table lookup — `nn.Embedding(num_items, d)` in modern frameworks — is mathematically identical to a matrix factorization layer. What changed was the *framing*: treating items as tokens opened the door to borrowing the entire NLP toolkit (attention, positional encoding, masked pretraining) for recommendation.

### 2.2 Wide & Deep and the Two-Tower Paradigm

Google's *Wide & Deep* (Cheng et al., 2016) formalized the standard industrial recommendation architecture: a *wide* component (logistic regression over manually crossed features) captures memorization (specific feature conjunctions known to predict clicks), while a *deep* component (an MLP over embeddings) captures generalization (smooth interpolation in embedding space). The two components are trained jointly and their outputs summed.

The *two-tower* model (Yi et al., 2019) specialized this for retrieval: train a user tower $f_u : u \mapsto \mathbf{h}_u \in \mathbb{R}^d$ and an item tower $f_i : i \mapsto \mathbf{h}_i \in \mathbb{R}^d$ separately, then use approximate nearest-neighbor search over pre-computed item embeddings at serving time. This decoupling of user and item encoders — train together, serve independently — is still the dominant industrial retrieval paradigm.

### 2.3 Deep Interest Networks: The First Crack

*Deep Interest Networks* (DIN, Zhou et al., KDD 2018) introduced the most important architectural idea of the pre-generative era: *target-aware attention* over user history. Rather than summarizing the user into a fixed vector independently of the candidate item, DIN computes a candidate-conditioned user representation:

$$\mathbf{v}_U(A) = \sum_{j=1}^H a(\mathbf{e}_j, \mathbf{v}_A) \cdot \mathbf{e}_j$$

where $a(\mathbf{e}_j, \mathbf{v}_A)$ is a small MLP applied to the concatenation of history item embedding, candidate embedding, their elementwise product, and their outer product. Critically, there is **no softmax normalization**: the unnormalized magnitude of $\mathbf{v}_U(A)$ encodes how *strongly* the user has historically preferred content similar to $A$, not just which history items are most relevant.

> [!TIP] Why DIN matters historically
> DIN's no-softmax design is not an accident — it's a principled choice that reappears, generalized, in HSTU's softmax-free attention. Both architectures recognize that recommendation requires encoding *preference intensity* (how much does this user care about this type of content?) not just *relative relevance* (which history item is most like the candidate?). Softmax normalization destroys the former in favor of the latter.

DIN showed that the static aggregate features that DLRM relied on — 7-day CTR by category, session engagement counts — could be replaced by learned attention over raw history. This was the first formal argument that feature engineering was not fundamental to the problem.

### 2.4 The DLRM Architecture

Meta's *Deep Learning Recommendation Model* (DLRM, Naumov et al., 2019) codified the standard industrial architecture that dominated from 2016 to 2023:

1. **Embedding lookup** for each categorical feature (user ID, item ID, category, device, etc.)
2. **FM-style pairwise interactions** over all feature embedding pairs
3. **MLP** combining interaction outputs and dense features into a click probability

DLRM's distinguishing characteristic is *impression-level training*: each (user, item, label) triple is processed as an independent example. A user with 1,000 interactions contributes 1,000 binary labels, but the model sees no dependency structure between them — each prediction is conditioned on the current (user, item) pair and a bag-of-history features, not on the ordered sequence of past interactions.

At moderate item catalog sizes (stable inventory, millions of items rather than billions), DLRM works extremely well: per-item ID embeddings converge, FM interactions capture cross-feature dependencies efficiently, and the architecture scales to hundreds of millions of parameters.

---

## Era III — Sequential Modeling (2016–2022)

### 3.1 Session-Based Recommendation with RNNs

The first systematic treatment of recommendation as a sequence prediction problem came from *GRU4Rec* (Hidasi et al., ICLR 2016): model each user session as a sequence of item interactions and train a GRU to predict the next item. The key insight was to frame recommendation as a *language modeling* problem over the "language" of user behavior.

GRU4Rec used a session-parallel mini-batch scheme to handle variable-length sessions efficiently and trained with ranking losses (TOP1, BPR) rather than pointwise cross-entropy. It outperformed matrix factorization methods significantly on session-based benchmarks, establishing that temporal order in interaction sequences contains signal that bag-of-interactions models miss.

> [!WARNING] ⚠️ Session vs. full-history modeling
> GRU4Rec operates on *within-session* sequences (a single browsing session), not a user's full interaction history. This is both a simplification (sessions are shorter and more coherent) and a limitation (cross-session preference patterns are invisible). The transition to full-history modeling requires dealing with histories of thousands or millions of interactions — a systems problem as much as a modeling one.

**Caser** (Tang & Wang, 2018) framed sequential recommendation as a convolutional image problem: arrange the most recent $L$ item embeddings as a "temporal image" and apply horizontal and vertical convolutions to capture both point-level and union-level sequential patterns. While less scalable than attention-based approaches, Caser illustrated that local patterns in sequences (e.g., "buy A then B within 3 days") carry recommendation-relevant signal.

### 3.2 SASRec: Self-Attention Comes to Recommendation

*Self-Attentive Sequential Recommendation* (SASRec, Kang & McAuley, ICDM 2018) applied the Transformer decoder architecture — specifically, causal self-attention — to sequential recommendation. The model processes the user's last $N$ interactions as a sequence, applies $b$ Transformer blocks with causal masking, and predicts the next item from the final hidden state via a dot product with all item embeddings.

The key design choices were:
- **Causal (unidirectional) masking**: position $t$ attends only to positions $\leq t$, preserving temporal causality
- **Learned positional embedding** added to each item embedding
- **Item embedding tying**: the prediction head uses the same embedding table as the input, reducing parameter count
- **Next-item prediction via inner product**: at position $t$, compute $\text{score}(x) = \mathbf{h}_t^\top \mathbf{e}_x$ for all items $x$

SASRec substantially outperformed RNN-based and CNN-based sequential recommenders across multiple datasets, and established the Transformer as the natural architecture for sequential recommendation.

> [!INFO] SASRec vs. HSTU
> SASRec is the direct architectural ancestor of HSTU. Both use causal self-attention over interaction sequences. The differences are in the details that matter at industrial scale: SASRec uses standard softmax attention and sinusoidal positional embeddings; HSTU replaces these with SiLU-gated attention (no softmax), temporal relative biases, and stochastic length sparsity. Each modification is a direct response to production-scale failure modes that SASRec's research-scale design never encountered.

### 3.3 BERT4Rec and Bidirectional Pretraining

*BERT4Rec* (Sun et al., CIKM 2019) adapted BERT's masked language model pretraining to recommendation: randomly mask items in a user history and train the model to predict the masked items given bidirectional context. At inference, mask the last item position and predict.

BERT4Rec's bidirectional attention allows each position to attend to both past and future items in the history, which gives richer context for predicting masked positions during training. It outperformed SASRec on several benchmarks.

However, bidirectional attention creates a fundamental train-inference mismatch: at inference time, there is no "future" context available, so the model predicts from a truncated context that it was not trained to handle. *This mismatch is one reason that later work (GenRank in particular) finds that causal/autoregressive masking is not just a convenience but an architecturally essential choice for ranking.*

### 3.4 Why Sequential DLRMs Still Hit a Wall

By 2022, the field had established that sequential Transformer models outperform static DLRM on sequential recommendation benchmarks. However, industrial deployment of sequential models at scale revealed a new problem that neither SASRec nor BERT4Rec was designed to address: the *quality saturation* of per-item ID embeddings on creator-economy platforms.

The crux: on platforms where new items are created continuously (short video feeds, news, user-generated content), the item catalog grows at the same rate as training data. A platform adding $10^6$ new videos daily adds $10^6$ new learnable ID embedding vectors. The training signal per parameter stays bounded — the error per item embedding does not decrease with more data. The model gets bigger, but not better.

This was the structural failure that motivated the complete paradigm shift of 2023.

---

## Era IV — The Generative Turn (2023–Present)

### 4.1 Quality Saturation Crystallizes the Problem

Zhao et al. (KDD 2023) formalized the failure mode precisely. For an *item-centric ranking* (ICR) model with per-item embeddings on a creator-economy platform where $|\mathcal{I}(t)| = O(t)$ and total interactions $T(t) = O(t)$, the expected estimation error satisfies:

$$\mathbb{E}\!\left[\|\hat{\theta}_t - \theta^*\|^2\right]_{\text{ICR}} = \Omega(1) \quad \text{as } t \to \infty$$

The training signal-to-parameter ratio is $T(t) / (|\mathcal{I}(t)| \cdot d) = O(1)$ — it never grows. A *user-centric ranking* (UCR) model whose parameter count is $O(1)$ in $t$ achieves $O(1/t)$ error, simply because the fixed parameter count receives growing signal.

Empirically, at 60 days of training data, an ICR model is **21× larger** in parameters than its UCR counterpart trained on identical data, with *worse* quality per parameter. Increasing embedding dimension degrades ICR performance (random initialization in underfed embeddings adds noise) while improving UCR performance. The problem is structural, not a tuning issue.

> [!DANGER] 🔴 The parameter inflation trap
> The natural instinct when quality plateaus is to add capacity — larger embeddings, more layers. For DLRMs on creator-economy platforms, this makes things worse: more embedding capacity with the same training signal means more underfitted parameters. The model is not data-starved; it is *signal-per-parameter starved*. The fix requires decoupling parameter count from item inventory size.

Separately, the *Wukong* paper (Zhang et al., 2024) showed that even the best FM-stack DLRM architecture saturates around 31 GFLOP/example, with no quality improvement from additional compute beyond that point. These two results together made the case for a paradigm change.

### 4.2 Generative Retrieval: TIGER and Semantic IDs

The **TIGER** paper (Rajput et al., NeurIPS 2023) addressed the generative retrieval problem: generating the next item directly from the user's history, without a pre-specified candidate set. The obstacle is that the item "vocabulary" has $|\mathcal{I}| \sim 10^9$ entries — a softmax output layer is computationally infeasible.

TIGER's solution: encode each item as a short tuple of discrete codewords (*semantic ID*) via a Residual Quantized VAE (RQ-VAE), then treat recommendation as a seq2seq generation problem. A Transformer encoder processes the user's history of semantic IDs; a decoder autoregressively generates the semantic ID $(c_1, c_2, \ldots, c_K)$ of the next item. At each decode step, the output vocabulary is the codebook size $V \sim 256$, not $|\mathcal{I}|$. Beam search over the code tree retrieves candidates efficiently in $O(B \cdot K \cdot V)$.

*Historically, this was the first paper to demonstrate that next-item recommendation can be cast as a pure generation problem at industrial vocabulary sizes.* The RQ-VAE construction also provides a natural hierarchy — similar items share prefix codes — that makes beam search semantically meaningful.

**COBRA** (Yang et al., 2025) later extended TIGER by noting that the discrete semantic ID necessarily loses information via quantization. COBRA generates a complementary dense vector *conditioned on* the predicted semantic ID:

$$P(\text{ID}_{t+1},\, \mathbf{v}_{t+1} \mid S_{1:t}) = P(\text{ID}_{t+1} \mid S_{1:t}) \cdot P(\mathbf{v}_{t+1} \mid \text{ID}_{t+1},\, S_{1:t})$$

The cascade — coarse discrete token first, then fine-grained dense vector conditioned on it — recovers information that neither branch alone could represent. Deployed at Baidu (200M DAU): +3.60% conversion, +4.15% ARPU.

> [!NOTE] 📐 The tokenization problem as the central technical challenge of generative retrieval
> The key difficulty that TIGER and COBRA both address is: *how do you make a $10^9$-item vocabulary tractable for autoregressive generation?* RQ-VAE solves this by hierarchical quantization, turning an intractable softmax over $10^9$ items into $K$ sequential softmaxes each over $V \sim 256$ codewords. The tradeoff — information loss via quantization, invalid codes in beam search, codebook collapse — defined the research agenda for generative retrieval for the next two years. See [[note#4. Generative Retrieval: The Tokenization Problem|§4 of the concept note]] for full derivations.

### 4.3 HSTU: Scaling Laws Arrive in Recommendation

**HSTU** (*Hierarchical Sequential Transduction Units*, Zhai et al., 2024) is the architectural backbone that makes generative ranking work at trillion-parameter scale. It is the paper that establishes, for the first time, a clean power-law scaling relationship in recommendation:

$$\text{HR@100} = 0.15 + 0.0195 \ln C, \qquad \text{NE} = 0.549 - 5.3 \times 10^{-3} \ln C$$

across three orders of magnitude in training compute. At $1.5 \times 10^{12}$ parameters, the deployed model continues improving. DLRMs cluster at HR@100 $\approx 0.28$–$0.29$ regardless of compute budget.

The key architectural departures from standard Transformers:

**No softmax.** HSTU replaces softmax-normalized attention with SiLU-gated pointwise attention:

$$A(X) = \text{SiLU}\!\left(Q(X)\, K(X)^\top + \text{rab}^{p,t}\right)$$

where SiLU is applied *elementwise* to the score matrix. This preserves the absolute magnitude of preference signals, which softmax normalization would destroy. Under a Dirichlet Process model of non-stationary user preferences, softmax-free attention outperforms the softmax variant by **44.7% relative HR@10** on synthetic data.

**Temporal relative attention bias.** Rather than absolute positional encodings (which do not generalize across sequence lengths), HSTU uses a learned bias $\text{rab}^{p,t}_{ij} = b^p_{|i-j|} + b^t_{\lfloor \log_2(\Delta t_{ij}) \rfloor}$ based on relative position and real elapsed time. This encodes both recency (position) and temporal proximity (wall-clock time) as inductive biases.

**Stochastic length sparsity.** For users with very long histories, HSTU subsamples the sequence to control attention cost. A sequence of length $n$ is processed with expected cost $O(N^\alpha d)$ for a tunable $\alpha$, with empirically less than 0.2% metric degradation at 80%+ sparsity.

**M-FALCON.** At inference, instead of scoring each candidate independently (cost $O(b_m n^2 d)$), HSTU computes the user history prefix once and amortizes its $O(n^2 d)$ cost across $b_m$ candidates. Total cost: $O(n^2 d + b_m n d)$ — an $O(b_m)$ improvement for large candidate sets.

> [!EXAMPLE] 🔑 Why this is historically significant
> HSTU's scaling law result is the recommendation analogue of Kaplan et al. (2020) for language models. The observation that a recommendation model's quality continues improving log-linearly with compute across three orders of magnitude — and that this improvement is *architectural* (DLRMs on the same data plateau) — established that the generative paradigm change is not incremental but *fundamental*. The largest deployed model is the best model, with no sign of saturation.

### 4.4 Production Refinements

**GenRank** (Huang et al., 2025) adapted HSTU for production at Xiaohongshu (hundreds of millions of users), addressing two efficiency bottlenecks:

*Action-oriented sequence organization.* HSTU interleaves item and action tokens; GenRank argues that items serve mainly as positional context for actions and consolidates each (item, action) pair into a single token, halving sequence length and reducing attention cost by 75%.

*ALiBi position bias.* HSTU's learnable $\text{rab}^{p,t}$ requires an $O(n^2)$ look-up table. GenRank replaces it with ALiBi (Press et al., 2022) — a parameter-free linear distance penalty on attention scores that adds zero learned parameters and integrates into FlashAttention at negligible cost. Combined with halved sequence length, training throughput improves by **94.8%**.

The most intellectually interesting finding from GenRank is an ablation: replacing HSTU's causal mask with bidirectional attention causes AUC drops exceeding 0.0100, with the gap growing with model size. *The causal autoregressive structure is an architectural property of generative ranking, not merely a training convenience.* This retrospectively validates BERT4Rec's train-inference mismatch as a principled concern, not just a practical one.

### 4.5 Reasoning at the Top: GR2

**GR2** (Liang et al., Meta, 2026) introduces a fourth stage to the industrial recommendation pipeline — *reasoning-based reranking* — built on a large language model. Where ranking (HSTU/GenRank) scores hundreds of candidates via a sequence model, GR2 applies an 8B-parameter LLM (Qwen3-8B) to the top-$K$ ranked candidates ($K = 10$) and asks it to reason explicitly about their relative ordering.

Items are represented as RQ-VAE semantic IDs injected into the LLM's vocabulary. A three-stage training pipeline — tokenized mid-training, rejection-sampling SFT on reasoning traces, DAPO RL with rank-promotion reward — produces a model that reasons about cross-candidate comparisons ("the user bought conditioner last week, so among these candidates the complementary item is X rather than Y") that the ranking model, which scores candidates independently, structurally cannot.

The RL stage is necessary: SFT alone often *degrades* R@1 by improving reasoning fluency at the expense of ranking accuracy. The reward function is carefully designed with a conditional format reward to prevent the degenerate strategy of copying the pre-ranked order.

*GR2 represents the current frontier: the full NLP toolkit (LLMs, chain-of-thought, RLHF) applied to the top of the recommendation funnel.*

---

## The Through-Line

The 30-year arc from collaborative filtering to generative recommenders is a story about progressively *relaxing assumptions* that turned out to be unnecessary:

| Assumption relaxed | When | What replaced it |
|---|---|---|
| Users and items are atomic (no content) | 2013–2016 | Learned embeddings from IDs and content features |
| User preference is static | 2016–2018 | RNN/Transformer sequential models |
| User preference is target-independent | 2018 | Target-aware attention (DIN) |
| User preference can be summarized by aggregate features | 2018–2022 | Raw sequence processing |
| Items need per-item ID embeddings | 2023 | User-centric models, semantic IDs |
| Recommendation requires discriminative scoring | 2023 | Generative next-event modeling |
| Recommendation doesn't scale | 2024 | Scaling laws (HSTU) |
| Reranking must be implicit | 2026 | Explicit reasoning traces (GR2) |

Each relaxation was resisted — feature engineers argued that handcrafted features captured domain knowledge that models couldn't learn; ID embeddings were argued to encode item-specific signal that shared parameters couldn't represent; causal generation was argued to be too expensive for production latency. In every case, the assumption collapsed when empirical evidence at scale made the tradeoff clear.

The common driver across all transitions: **more compute and data reveal which constraints are real and which are artifacts of working at insufficient scale.**

---

## References

| Reference Name | Brief Summary | Link to Reference |
|---|---|---|
| [Using Collaborative Filtering to Weave an Information Tapestry (Goldberg et al., 1992)](https://dl.acm.org/doi/10.1145/138859.138867) | First use of the term "collaborative filtering"; Tapestry system at Xerox PARC | https://dl.acm.org/doi/10.1145/138859.138867 |
| [Amazon.com Recommendations: Item-to-Item Collaborative Filtering (Linden et al., 2003)](https://ieeexplore.ieee.org/document/1167344) | Item-based CF at industrial scale; stable item-item similarities pre-computable offline | https://ieeexplore.ieee.org/document/1167344 |
| [Collaborative Filtering for Implicit Feedback Datasets (Hu et al., 2008)](https://ieeexplore.ieee.org/document/4781121) | ALS for implicit feedback (clicks/watches rather than explicit ratings) | https://ieeexplore.ieee.org/document/4781121 |
| [Matrix Factorization Techniques for Recommender Systems (Koren et al., 2009)](https://ieeexplore.ieee.org/document/5197422) | Summary of Netflix Prize MF methods; biased SVD, temporal dynamics extensions | https://ieeexplore.ieee.org/document/5197422 |
| [Factorization Machines (Rendle, ICDM 2010)](https://ieeexplore.ieee.org/document/5694074) | Generalizes MF to arbitrary feature vectors; $O(nk)$ pairwise interaction via identity trick | https://ieeexplore.ieee.org/document/5694074 |
| [Wide & Deep Learning for Recommender Systems (Cheng et al., RecSys 2016)](https://arxiv.org/abs/1606.07792) | Wide (memorization via feature crosses) + deep (generalization via embeddings); Google Play deployment | https://arxiv.org/abs/1606.07792 |
| [Session-Based Recommendations with Recurrent Neural Networks — GRU4Rec (Hidasi et al., ICLR 2016)](https://arxiv.org/abs/1511.06939) | First RNN (GRU) applied to session-based recommendation; session-parallel mini-batch training | https://arxiv.org/abs/1511.06939 |
| [Deep Interest Network for Click-Through Rate Prediction — DIN (Zhou et al., KDD 2018)](https://arxiv.org/abs/1706.06978) | Target-aware local activation over user history; no softmax normalization; supersedes static aggregate features | https://arxiv.org/abs/1706.06978 |
| [Self-Attentive Sequential Recommendation — SASRec (Kang & McAuley, ICDM 2018)](https://arxiv.org/abs/1808.09781) | First Transformer for sequential recommendation; causal masking; item embedding tying | https://arxiv.org/abs/1808.09781 |
| [BERT4Rec: Sequential Recommendation with BERT (Sun et al., CIKM 2019)](https://arxiv.org/abs/1904.06690) | Masked-item pretraining with bidirectional attention; train-inference mismatch later identified as critical flaw | https://arxiv.org/abs/1904.06690 |
| [Breaking the Curse of Quality Saturation with User-Centric Ranking (Zhao et al., KDD 2023)](https://arxiv.org/abs/2305.15333) | Formal proof of ICR error floor; 21× parameter inflation; UCR achieves O(1/t) error | https://arxiv.org/abs/2305.15333 |
| [Recommender Systems with Generative Retrieval — TIGER (Rajput et al., NeurIPS 2023)](https://arxiv.org/abs/2305.05065) | First generative retrieval via RQ-VAE semantic IDs; seq2seq beam search over code tree | https://arxiv.org/abs/2305.05065 |
| [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations — HSTU (Zhai et al., 2024)](https://arxiv.org/abs/2402.17152) | HSTU architecture; scaling laws for GR; 1.5T deployed model; M-FALCON amortization | https://arxiv.org/abs/2402.17152 |
| [Wukong: Towards a Scaling Law for Large-Scale Recommendation (Zhang et al., 2024)](https://arxiv.org/abs/2403.02545) | FM-stack DLRM scaling study; DLRM plateaus at 31 GFLOP/example | https://arxiv.org/abs/2403.02545 |
| [COBRA: Sparse Meets Dense for Generative Recommendation (Yang et al., 2025)](https://arxiv.org/abs/2503.02453) | Cascaded sparse-dense generation; probabilistic factorization; BeamFusion; +3.60% conversion at Baidu | https://arxiv.org/abs/2503.02453 |
| [Towards Large-scale Generative Ranking — GenRank (Huang et al., 2025)](https://arxiv.org/abs/2505.04180) | Action-oriented sequences; ALiBi bias; 94.8% training speedup; ablation proving causal masking is essential | https://arxiv.org/abs/2505.04180 |
| [Generative Reasoning Re-ranker — GR2 (Liang et al., Meta 2026)](https://arxiv.org/abs/2602.07774) | LLM-based reranking with chain-of-thought; rejection-sampling SFT; DAPO RL; rank-promotion reward | https://arxiv.org/abs/2602.07774 |
| [Scaling Laws for Neural Language Models (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361) | Original power-law scaling for LLMs; motivates the HSTU scaling law analogy | https://arxiv.org/abs/2001.08361 |
