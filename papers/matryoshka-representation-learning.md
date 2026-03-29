# Matryoshka Representation Learning

**Kusupati et al., NeurIPS 2022** — Aditya Kusupati, Gantavya Bhatt, Aniket Rege, Matthew Wallingford, Aditya Sinha, Vivek Ramanujan, William Howard-Snyder, Kaifeng Chen, Sham Kakade, Prateek Jain, Ali Farhadi

| Dimension | Prior State | This Paper | Key Result |
|---|---|---|---|
| Embedding flexibility | Fixed-size embeddings; separate models per dimension | Single model producing nested embeddings at $O(\log d)$ granularities | Adaptive cascade: **14× smaller representation** (37-dim vs. 512-dim) at same 76.3% ImageNet top-1 |
| Low-dim quality | Post-hoc PCA/SVD: 2.3% top-1 at 8-dim | MRL trained prefix: 66.6% top-1 at 8-dim | **29× accuracy gain** over SVD at minimum granularity |
| Retrieval throughput | Single-shot 2048-dim ANN search | Adaptive shortlisting (16-dim) + re-ranking (2048-dim) | **128× theoretical FLOP reduction**, 14× wall-clock speedup, no mAP@10 loss |
| Backbone compatibility | Task-specific training pipelines | Loss-level modification; any backbone | Works with ResNet, ViT, BERT, ALIGN with no extra hyperparameter tuning |

## Relations

**Extended by:** [[papers/2d-matryoshka-sentence-embeddings|2D Matryoshka Sentence Embeddings]] *(no note yet)*, [[papers/matryoshka-adaptor|Matryoshka-Adaptor (EMNLP 2024)]] *(no note yet)*, [[papers/matformer|MatFormer (NeurIPS 2024)]] *(no note yet)*
**Concepts used:** [[concepts/neural-scaling-laws/note|Neural Scaling Laws]]

---

## Table of Contents

- [[#1. Motivation and Background|1. Motivation and Background]]
  - [[#1.1 The Fixed-Capacity Problem|1.1 The Fixed-Capacity Problem]]
  - [[#1.2 The Matryoshka Doll Intuition|1.2 The Matryoshka Doll Intuition]]
- [[#2. The MRL Objective|2. The MRL Objective]]
  - [[#2.1 Notation and Setup|2.1 Notation and Setup]]
  - [[#2.2 The Nesting Constraint and Multi-Granularity Loss|2.2 The Nesting Constraint and Multi-Granularity Loss]]
  - [[#2.3 Efficient MRL via Weight Tying|2.3 Efficient MRL via Weight Tying]]
  - [[#2.4 Relationship to Standard Representation Learning|2.4 Relationship to Standard Representation Learning]]
- [[#3. Training MRL Models|3. Training MRL Models]]
  - [[#3.1 What Changes in the Training Loop|3.1 What Changes in the Training Loop]]
  - [[#3.2 Importance Weights and Hyperparameter Choices|3.2 Importance Weights and Hyperparameter Choices]]
  - [[#3.3 Contrastive and Generative Extensions|3.3 Contrastive and Generative Extensions]]
- [[#4. Inference-Time Flexibility|4. Inference-Time Flexibility]]
  - [[#4.1 Adaptive Classification|4.1 Adaptive Classification]]
  - [[#4.2 Adaptive Retrieval: Shortlisting and Re-ranking|4.2 Adaptive Retrieval: Shortlisting and Re-ranking]]
  - [[#4.3 The FLOP/Accuracy Pareto Frontier|4.3 The FLOP/Accuracy Pareto Frontier]]
- [[#5. Theoretical Grounding|5. Theoretical Grounding]]
  - [[#5.1 Why Prefixes Retain Information Under MRL|5.1 Why Prefixes Retain Information Under MRL]]
  - [[#5.2 Interpolation Across Intermediate Dimensions|5.2 Interpolation Across Intermediate Dimensions]]
  - [[#5.3 Capacity Arguments and the Information Bottleneck View|5.3 Capacity Arguments and the Information Bottleneck View]]
- [[#6. Empirical Results|6. Empirical Results]]
  - [[#6.1 ImageNet Classification|6.1 ImageNet Classification]]
  - [[#6.2 Image Retrieval|6.2 Image Retrieval]]
  - [[#6.3 Few-Shot and Long-Tail Learning|6.3 Few-Shot and Long-Tail Learning]]
  - [[#6.4 Robustness Benchmarks|6.4 Robustness Benchmarks]]
- [[#7. Extensions and Follow-up Work|7. Extensions and Follow-up Work]]
  - [[#7.1 2D Matryoshka Sentence Embeddings|7.1 2D Matryoshka Sentence Embeddings]]
  - [[#7.2 Matryoshka-Adaptor|7.2 Matryoshka-Adaptor]]
  - [[#7.3 Multimodal Extensions|7.3 Multimodal Extensions]]
  - [[#7.4 MRL in Production: OpenAI text-embedding-3|7.4 MRL in Production: OpenAI text-embedding-3]]
  - [[#7.5 MatFormer: Nesting at the Architecture Level|7.5 MatFormer: Nesting at the Architecture Level]]
- [[#8. References|8. References]]

---

## 1. Motivation and Background 🎯

### 1.1 The Fixed-Capacity Problem

Modern ML pipelines decouple representation learning from downstream use. A model $F(\cdot\,;\theta_F): \mathcal{X} \to \mathbb{R}^d$ is trained once — on ImageNet, a web-crawled corpus, or a contrastive multimodal dataset — and its frozen embeddings are reused across an open-ended collection of tasks. This decoupling is efficient but hides a structural mismatch: the embedding dimension $d$ is chosen at training time, yet downstream tasks have heterogeneous and often unknown computational budgets.

Consider two concrete failure modes:

1. **Over-provisioning.** A product search system ingests 100 billion items, each embedded at $d = 2048$. Even at float16, the index occupies $\sim$400 GB. Halving $d$ would cut memory and ANN query cost by 2×, but no post-hoc compression method does so without catastrophic accuracy loss (see §5.1 for why).

2. **Under-provisioning.** A low-latency real-time retrieval path needs $d = 32$, while a high-accuracy offline re-ranking path can afford $d = 512$. Training two separate models wastes compute and introduces representation drift between the stages.

The paper identifies this as a *capacity-vs-cost* tradeoff: fixed-size embeddings are either over- or under-accommodating relative to any given deployment context.

> [!INFO] The fundamental asymmetry
> Training is done once; inference is done billions of times. The correct design point is to amortize the training cost of learning flexible representations over the many downstream deployments that benefit from different granularities.

### 1.2 The Matryoshka Doll Intuition

The paper's central metaphor is the *Matryoshka doll* (Russian nesting doll): a set of wooden figures each concealed inside the next larger one. MRL encodes information at multiple levels of granularity within a single vector $z \in \mathbb{R}^d$, arranged so that:

$$z_{1:8} \;\subset\; z_{1:16} \;\subset\; z_{1:32} \;\subset\; \cdots \;\subset\; z_{1:d}$$

where $z_{1:m}$ denotes the first $m$ coordinates of $z$. Each prefix $z_{1:m}$ is a *standalone* representation — independently useful for classification, retrieval, or any downstream task — and the prefix relationship is not imposed post-hoc but baked into training via the loss.

> [!TIP] Coarse-to-fine semantics
> Lower-dimensional prefixes capture coarse semantic structure (superclass identity, rough topic), while higher-dimensional extensions refine this into finer distinctions (fine-grained class, subtle sentiment). This mirrors the classical *information bottleneck* view: tight bottlenecks preserve class-level signal and discard instance-level noise.

![Figure 1 from Kusupati et al. (2022): MRL overview showing adaptive inference and training](https://arxiv.org/html/2205.13147v4/x3.png)
*Figure 1 (Kusupati et al., 2022): Left — adaptive inference and retrieval pipeline at varying representation sizes. Center — a single MRL-trained embedding vector acts as a nested set of dolls; each prefix $z_{1:m}$ is independently usable. Right — the MRL training objective applies the loss at each of $O(\log d)$ nested dimensions.*

> [!QUESTION] Exercise 1: Nesting as a geometric constraint
> *This exercise establishes that the nesting structure imposes an inclusion constraint on the subspaces used by each granularity.*
>
> Let $V_m = \text{span}(e_1, \ldots, e_m) \subseteq \mathbb{R}^d$ denote the coordinate subspace corresponding to the first $m$ dimensions. Show that MRL's prefix structure enforces $V_{m_1} \subseteq V_{m_2}$ whenever $m_1 \leq m_2$. Contrast this with PCA, where the $m_1$-dimensional subspace is not generally a subspace of the $m_2$-dimensional subspace when $m_2 > m_1$. Why does this difference make MRL suitable for adaptive retrieval pipelines while PCA is not?

---

## 2. The MRL Objective 📐

### 2.1 Notation and Setup

**Definition (MRL Setup).** Let:
- $\mathcal{X}$ be the input space (e.g., images, text).
- $F(\cdot\,; \theta_F): \mathcal{X} \to \mathbb{R}^d$ be a neural encoder parameterized by $\theta_F$, producing a $d$-dimensional embedding $z = F(x; \theta_F)$.
- $\mathcal{M} \subset [d]$ be a set of *nesting dimensions* with $|\mathcal{M}| \leq \lfloor \log_2 d \rfloor$. The standard choice is $\mathcal{M} = \{8, 16, 32, 64, 128, 256, 512, 1024, 2048\}$ for $d = 2048$.
- $z_{1:m} \in \mathbb{R}^m$ denote the first $m$ coordinates of $z$ — the *$m$-dimensional prefix* embedding.
- $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ be a labeled training dataset with $L$ classes.
- $\mathbf{W}^{(m)} \in \mathbb{R}^{L \times m}$ be a linear classification head for dimension $m$.
- $\ell: \mathbb{R}^L \times [L] \to \mathbb{R}$ be the multi-class softmax cross-entropy loss.
- $c_m \geq 0$ be a scalar *importance weight* for dimension $m \in \mathcal{M}$.

### 2.2 The Nesting Constraint and Multi-Granularity Loss

**Definition (MRL Loss).** The Matryoshka Representation Learning objective jointly optimizes over the encoder parameters $\theta_F$ and all classification heads $\{\mathbf{W}^{(m)}\}_{m \in \mathcal{M}}$:

$$\min_{\{\mathbf{W}^{(m)}\}_{m \in \mathcal{M}},\, \theta_F} \;\frac{1}{N} \sum_{i=1}^{N} \sum_{m \in \mathcal{M}} c_m \cdot \ell\!\left(\mathbf{W}^{(m)} \cdot F(x_i;\theta_F)_{1:m};\; y_i\right) \tag{1}$$

The nesting constraint is implicit: by always taking the *prefix* $z_{1:m}$ (not an arbitrary $m$-dimensional projection), the loss forces the encoder to arrange semantically useful information in the first coordinates of $z$. There is no explicit regularizer enforcing nesting — it emerges from the shared-encoder, prefix-indexing structure of the loss itself.

> [!INFO] Why prefixes and not arbitrary subsets?
> One could, in principle, train with arbitrary $m$-element subsets of coordinates. Prefixes are the specific choice that enables zero-overhead inference: to obtain a $m$-dimensional representation, simply truncate the embedding. No secondary projection matrix needs to be stored or applied at query time.

The key structural insight is that **a single forward pass through $F$ produces all granularities simultaneously.** The $O(\log d)$ heads $\mathbf{W}^{(m)}$ are small and add negligible overhead; they are discarded after training. At inference, one uses $z_{1:m}$ directly.

> [!QUESTION] Exercise 2: MRL loss upper-bounds the standard loss
> *This exercise shows that the standard single-granularity training objective is a special case of MRL, so MRL is no harder to optimize in a minimax sense.*
>
> Let $\mathcal{M}_{\text{std}} = \{d\}$ (the full dimension only) and $c_d = 1$. Show that the MRL loss (Eq. 1) reduces exactly to the standard cross-entropy loss on $z_{1:d} = z$. Now let $\mathcal{M}$ contain $|\mathcal{M}|$ elements, all with $c_m = 1/|\mathcal{M}|$. Show that the MRL loss is a convex combination of per-dimension losses. Conclude that any solution to the standard problem is also feasible for MRL, and that the MRL optimum achieves loss $\leq$ the average of the per-dimension optima.

### 2.3 Efficient MRL via Weight Tying

The *Efficient MRL* (MRL-E) variant ties the classification heads across dimensions:

$$\mathbf{W}^{(m)} = \mathbf{W}_{1:m} \quad \text{for all } m \in \mathcal{M}$$

where $\mathbf{W} \in \mathbb{R}^{L \times d}$ is a single full-dimensional head and $\mathbf{W}_{1:m}$ denotes its first $m$ columns. This reduces classifier memory from $O(L \cdot \sum_{m \in \mathcal{M}} m)$ to $O(L \cdot d)$, approximately a 2× reduction for the standard logarithmically-spaced $\mathcal{M}$.

*Weight tying introduces a coupling between the gradient signals from different granularities*: the gradient of $\ell(\mathbf{W}_{1:m} \cdot z_{1:m}; y)$ with respect to $\mathbf{W}_{1:m}$ propagates into the smaller-$m$ columns of $\mathbf{W}$, while larger-$m$ losses add updates to the additional columns. The paper reports that MRL-E matches MRL accuracy at mid-to-high dimensions but degrades slightly at the extreme low end (e.g., 56.7% vs. 66.6% at $d=8$ for ResNet50 on ImageNet).

> [!QUESTION] Exercise 3: Gradient decomposition in MRL-E
> *This exercise makes explicit how the weight-tying in MRL-E creates gradient interference between granularities.*
>
> In MRL-E, write out $\frac{\partial}{\partial \mathbf{W}_{1:m_1}} \sum_{m \in \mathcal{M}} c_m \cdot \ell(\mathbf{W}_{1:m} \cdot z_{1:m}; y)$ for a fixed $m_1 \in \mathcal{M}$. Show that this gradient receives contributions from all $m \geq m_1$. Does larger-$m$ signal help or hurt the $m_1$-dimensional objective? Argue heuristically for why this explains the accuracy gap between MRL-E and MRL at small $m$.

### 2.4 Relationship to Standard Representation Learning

The MRL loss (Eq. 1) differs from standard representation learning in exactly one way: it sums the classification loss over $|\mathcal{M}|$ prefix lengths, each with its own head. Everything else — the backbone $F$, the optimizer, the data pipeline, the augmentation schedule — remains unchanged.

This minimal-modification property is a design goal. The paper explicitly validates that using the same hyperparameters as the corresponding independently-trained baseline suffices; no additional tuning is required.

> [!EXAMPLE] MRL loss for ResNet50 on ImageNet
> For ResNet50 ($d = 2048$) with $\mathcal{M} = \{8, 16, 32, 64, 128, 256, 512, 1024, 2048\}$ and uniform $c_m = 1$:
>
> $$\mathcal{L}_{\text{MRL}} = \frac{1}{N} \sum_{i=1}^N \left[ \ell(\mathbf{W}^{(8)} z_{1:8}^{(i)}; y_i) + \ell(\mathbf{W}^{(16)} z_{1:16}^{(i)}; y_i) + \cdots + \ell(\mathbf{W}^{(2048)} z_{1:2048}^{(i)}; y_i) \right]$$
>
> Nine classification heads are added. The 8-dimensional head has $1000 \times 8 = 8000$ parameters; the full-dimensional head has $1000 \times 2048 \approx 2\text{M}$ parameters. Total added parameters are dominated by the largest head — negligible relative to ResNet50's 25M.

---

## 3. Training MRL Models 🔧

### 3.1 What Changes in the Training Loop

The training loop modification is surgical. Below is a PyTorch pseudocode sketch:

```python
import torch
import torch.nn as nn

class MRLLoss(nn.Module):
    def __init__(self, nesting_dims, num_classes, importance_weights=None):
        super().__init__()
        self.nesting_dims = nesting_dims  # e.g., [8, 16, 32, ..., 2048]
        self.heads = nn.ModuleList([
            nn.Linear(m, num_classes, bias=False) for m in nesting_dims
        ])
        if importance_weights is None:
            # Uniform weighting by default
            self.c = [1.0] * len(nesting_dims)
        else:
            self.c = importance_weights
        self.ce = nn.CrossEntropyLoss()

    def forward(self, z, targets):
        """
        z: (batch_size, d) full-dimensional embedding from backbone
        targets: (batch_size,) integer class labels
        """
        total_loss = 0.0
        for m, head, c_m in zip(self.nesting_dims, self.heads, self.c):
            # Take the prefix of the embedding
            z_prefix = z[:, :m]
            logits = head(z_prefix)
            total_loss += c_m * self.ce(logits, targets)
        return total_loss / len(self.nesting_dims)
```

The training loop itself is unchanged: compute loss, call `.backward()`, step the optimizer. The only modification is replacing the standard cross-entropy call with `MRLLoss.forward(z, targets)`.

> [!WARNING] Normalization of the loss scale
> When $c_m = 1$ for all $m$, the MRL loss is $|\mathcal{M}|$ times larger in magnitude than the standard loss. If learning rate was tuned for the standard loss, scale $c_m \leftarrow 1/|\mathcal{M}|$ or reduce the learning rate accordingly to avoid instability.

### 3.2 Importance Weights and Hyperparameter Choices

The weights $c_m$ control the relative emphasis on each granularity. Three natural choices:

| Scheme | Definition | Effect |
|---|---|---|
| Uniform | $c_m = 1/\|\mathcal{M}\|$ | Equal training pressure at all granularities |
| Geometric | $c_m \propto m$ | Emphasizes high-dimensional, high-capacity representations |
| Endpoint | $c_m = 1$ for $m = d$, else $0$ | Degenerates to standard training |

The paper uses **uniform weights ($c_m = 1$)** throughout and reports competitive results without sweeping this hyperparameter. This suggests the multi-granularity signal is robust to weighting.

**Choice of $\mathcal{M}$:** The paper recommends logarithmically-spaced sizes, halving at each step. This ensures the nesting hierarchy is geometrically uniform: each step doubles the representational capacity, so the marginal information gain at each level is roughly equal. The constraint $|\mathcal{M}| \leq \lfloor \log_2 d \rfloor$ formalizes this.

### 3.3 Contrastive and Generative Extensions

MRL generalizes beyond supervised classification:

- **Contrastive learning (ALIGN):** The loss $\ell$ is replaced by an InfoNCE contrastive loss. For a matched image-text pair $(x^v, x^t)$, the MRL loss applies the contrastive objective to each prefix pair $(z^v_{1:m}, z^t_{1:m})$ independently:

$$\mathcal{L}_{\text{MRL-contrastive}} = \sum_{m \in \mathcal{M}} c_m \cdot \ell_{\text{NCE}}\!\left(z^v_{1:m} / \|z^v_{1:m}\|,\; z^t_{1:m} / \|z^t_{1:m}\|\right)$$

  Note the per-prefix $\ell_2$-normalization — necessary because cosine similarity is not scale-invariant.

- **Masked language modeling (BERT):** MRL-E with weight-tied heads reduces naturally to the standard MLM objective on the prefix. Specifically, BERT's output embedding matrix plays the role of $\mathbf{W}$; taking $\mathbf{W}_{1:m}$ produces the $m$-dimensional head automatically.

> [!QUESTION] Exercise 4: MRL contrastive loss
> *This exercise derives the gradient signal from the contrastive MRL loss at a small granularity $m$.*
>
> Let $s_m = (z^v_{1:m})^\top (z^t_{1:m}) / (\|z^v_{1:m}\| \|z^t_{1:m}\|)$ be the cosine similarity at dimension $m$. Write the InfoNCE loss $\ell_{\text{NCE}}$ for a batch of $B$ pairs. Compute $\partial \ell_{\text{NCE}} / \partial z^v_{1:m}$ and show that this gradient lives in $\mathbb{R}^m$ and therefore only updates the first $m$ dimensions of $z^v$ via backpropagation through the prefix slice. What does this imply about the independence of gradient signals across different $m$ values?

---

## 4. Inference-Time Flexibility 🚀

### 4.1 Adaptive Classification

At inference, a single forward pass through $F$ yields $z \in \mathbb{R}^d$. One then selects the appropriate prefix $z_{1:m}$ based on the available budget. Because the model was trained with the MRL loss, every prefix is a valid, calibrated representation — not a degraded truncation of an over-parameterized one.

**Adaptive classification cascade:** Rather than committing to a single dimension upfront, one can route individual examples to the appropriate dimension. The paper demonstrates that:

$$\text{accuracy at 37-dim} \approx \text{accuracy at 512-dim (fixed-feature)} \approx 76.3\%$$

A representation 13.8× smaller achieves the same accuracy, because easy examples (those with large margins at low dimension) never need the additional capacity.

![Figure 6 from Kusupati et al. (2022): Adaptive classification accuracy vs. expected representation size for MRL-AC vs. FF 2048](https://arxiv.org/html/2205.13147v4/x13.png)
*Figure 6 (Kusupati et al., 2022): Adaptive classification (MRL-AC) achieves the same top-1 accuracy as the FF-2048 baseline (dashed line) while using a dramatically smaller expected representation size — the Pareto-optimal tradeoff that motivates MRL for resource-constrained deployments.*

> [!QUESTION] Exercise 5: Oracle accuracy gain from perfect routing
> *This exercise quantifies the upper bound on accuracy achievable by an oracle adaptive classifier.*
>
> Let $\hat{y}_m(x)$ denote the prediction of the $m$-dimensional MRL head on input $x$. Define the *oracle adaptive classifier* as the one that, for each $x$, uses $\hat{y}_{m^*}(x)$ where $m^* = \min\{m \in \mathcal{M} : \hat{y}_m(x) = y\}$ (i.e., the smallest dimension that gets it right). The paper reports a 4.6% accuracy gain over the full-dimensional baseline for this oracle. Formalize why this gain is positive: argue that there must exist examples where $\hat{y}_d(x) \neq y$ but $\hat{y}_m(x) = y$ for some $m < d$. What does this suggest about the relationship between embedding capacity and difficulty of classification?

### 4.2 Adaptive Retrieval: Shortlisting and Re-ranking

Nearest-neighbor retrieval over a database of $N$ items with $d$-dimensional embeddings costs $O(Nd)$ per query for exact search (or dominates the ANN index building cost). MRL enables a *funnel retrieval* strategy:

**Definition (Adaptive Retrieval).** Given query $q$ with embedding $z^q \in \mathbb{R}^d$, shortlisting dimension $D_s \in \mathcal{M}$, re-ranking dimension $D_r \in \mathcal{M}$ ($D_s < D_r$), and shortlist size $K$:

1. **Shortlist:** Retrieve the top-$K$ candidates by $\ell_2$ distance in $\mathbb{R}^{D_s}$ using the prefix embeddings $\{z^{(j)}_{1:D_s}\}_{j=1}^N$.
2. **Re-rank:** Score the $K$ shortlisted items using $\{z^{(j)}_{1:D_r}\}_{j=1}^K$; return the top result.

The key observation: because $z^{(j)}_{1:D_s}$ and $z^{(j)}_{1:D_r}$ are prefixes of the same stored vector $z^{(j)}$, **a single database index suffices.** No separate low-dimensional and high-dimensional indices are needed.

```mermaid
flowchart LR
    Q["Query z^q_1:D_s<br/>low-dim prefix"] --> S["ANN shortlist<br/>top-K from N items"]
    S --> R["Re-rank with z^q_1:D_r<br/>high-dim prefix"]
    R --> Out["Final top-k<br/>results"]
```

**Funnel retrieval** extends this to a multi-stage cascade, halving the shortlist size at each step while doubling the dimension:

| Stage | Shortlist size | Dimension |
|---|---|---|
| 1 | 200 | 16 |
| 2 | 100 | 32 |
| 3 | 50 | 64 |
| 4 | 25 | 128 |
| 5 | 10 | 256 |
| Final | 10 | 2048 |

> [!QUESTION] Exercise 6: FLOP analysis of funnel retrieval
> *This exercise derives the theoretical speedup of funnel retrieval relative to single-shot full-dimensional search.*
>
> Assume a database of $N = 10^6$ items. Single-shot retrieval at dimension $d = 2048$ costs $2Nd = 4 \times 10^9$ multiply-add operations per query. Compute the total cost of the 6-stage funnel above (assume each stage computes exact inner products over its shortlist). Show the claimed $\sim$128× reduction. Under what conditions on the shortlist size $K$ relative to $N$ does the leading-order term come from the first stage?

### 4.3 The FLOP/Accuracy Pareto Frontier

The MRL family traces a Pareto frontier in the (FLOPs, accuracy) plane that dominates both:
- **Fixed-feature (FF) baselines:** independently trained $m$-dimensional models.
- **Post-hoc compression (SVD/PCA):** principal components of the full-dimensional embedding.

The fundamental reason for MRL's dominance over SVD/PCA is elaborated in §5.1. The fundamental reason for MRL's dominance over FF models is that the multi-granularity training signal acts as an *implicit regularizer* on the low-dimensional prefixes, making them more structured than what a standalone low-dimensional model learns from scratch.

---

## 5. Theoretical Grounding 🔬

### 5.1 Why Prefixes Retain Information Under MRL

The catastrophic failure of SVD/PCA at low dimensions ($2.3\%$ top-1 at $d=8$ vs. $66.6\%$ for MRL) is not simply a matter of information loss — it reflects a *structural mismatch*.

**PCA is not hierarchically nested.** Given a $d$-dimensional embedding $z$, PCA finds the $m$-dimensional subspace of maximum variance. Call it $V^{\text{PCA}}_m$. For $m' > m$, the PCA subspace $V^{\text{PCA}}_{m'}$ contains $V^{\text{PCA}}_m$ (by the nested eigenspace property of PCA). So PCA *is* nested — but the basis is data-dependent and requires a separate projection matrix $P_m \in \mathbb{R}^{d \times m}$ to apply.

The problem: the PCA projection $P_m$ is computed from the covariance of an embedding $z$ that was trained with *no knowledge* that it would be truncated. High-variance directions in $z$ are not necessarily semantically meaningful; they may reflect scale differences between coordinate dimensions induced by the arbitrary parameterization of $F$.

> [!EXAMPLE] Why PCA fails
> Suppose the encoder places most class-discriminative information in the last 100 dimensions of $z$ (a natural outcome of a randomly initialized final linear layer). PCA on the full embedding finds high-variance directions that are linear combinations of all dimensions. At $m = 8$, the top-8 PCA components are dominated by whatever directions happen to have large variance — which may encode coordinate scaling, not class identity. MRL explicitly penalizes *prefix accuracy*, forcing the encoder to place class-relevant information in early coordinates.

**Formal statement (informal):** MRL's objective directly minimizes classification loss using $z_{1:m}$ for all $m \in \mathcal{M}$. Any local optimum therefore satisfies: $z_{1:m}$ is a sufficient statistic for $y$ at granularity $m$, in the sense that the optimal $m$-dimensional linear classifier applied to $z_{1:m}$ achieves minimum Bayes-consistent loss for that dimension. Post-hoc SVD has no such guarantee because the embedding was trained only for $z_{1:d} = z$.

> [!QUESTION] Exercise 7: MRL vs. PCA on a toy dataset
> *This exercise demonstrates empirically that MRL prefixes outperform PCA truncation at low dimension.*
>
> Generate a 2D Gaussian mixture with 4 classes in $\mathbb{R}^{10}$ where the discriminative signal lies in the last 2 dimensions. Train a linear encoder $z = Wx + b \in \mathbb{R}^{10}$ with (a) standard cross-entropy on $z$ and (b) MRL loss on prefixes $z_{1:2}, z_{1:4}, z_{1:6}, z_{1:8}, z_{1:10}$. Compare the 2-dimensional test accuracy of: (i) PCA applied to the standard embedding, (ii) the MRL prefix $z_{1:2}$. Show that MRL prefix dominates PCA.
>
> ```python
> import numpy as np
> from sklearn.decomposition import PCA
> from sklearn.linear_model import LogisticRegression
> import torch, torch.nn as nn
>
> # TODO: generate data, train both models, compare 2-dim accuracy
> ```

### 5.2 Interpolation Across Intermediate Dimensions

A remarkable empirical finding: though MRL explicitly optimizes only $|\mathcal{M}| = O(\log d)$ nesting sizes, *accuracy interpolates smoothly at all intermediate dimensions* (i.e., dimensions not in $\mathcal{M}$).

This implies the encoder does not place information in discrete "packets" aligned with the elements of $\mathcal{M}$. Instead, the gradient signal from the $O(\log d)$ objectives diffuses across all $d$ coordinates, producing a smooth coarse-to-fine hierarchy. The paper verifies this empirically (Figure 5): for a ResNet50 trained with $\mathcal{M} = \{8, 16, \ldots, 2048\}$, accuracy at dimension 96 (not in $\mathcal{M}$, lying between 64 and 128) matches the value predicted by linear interpolation in log-dimension space.

*This interpolation property is not theoretically guaranteed* — it is an empirical regularity. One heuristic explanation: the SGD dynamics that minimize the sum of losses at $O(\log d)$ checkpoints implicitly regularize the loss landscape to be smooth between those checkpoints, analogous to how polynomial fitting at $k$ points produces a smooth curve.

![Figure 4 from Kusupati et al. (2022): 1-NN accuracy on ImageNet-1K for MRL models at varying representation sizes including ViT-JFT and ViT-ALIGN](https://arxiv.org/html/2205.13147v4/x12.png)
*Figure 4 (Kusupati et al., 2022): 1-NN accuracy on ImageNet-1K as a function of representation size, comparing MRL (solid lines) against independently trained fixed-feature baselines (dashed "Int" lines). Results span ResNet50-IN1K, ViT-B/16-JFT, and ViT-B/16-ALIGN — the smooth interpolation holds across all scales and architectures.*

### 5.3 Capacity Arguments and the Information Bottleneck View

**Definition (Representational capacity at dimension $m$).** The $m$-dimensional prefix $z_{1:m}$ is an information bottleneck: it is the only information available to the classifier $\mathbf{W}^{(m)}$. The bottleneck forces the encoder to maximize $I(z_{1:m}; y)$ — the mutual information between the prefix and the label.

By the data-processing inequality, $I(z_{1:m}; y) \leq I(z_{1:m'}; y)$ for $m \leq m'$ if $z_{1:m}$ is obtained deterministically from $z_{1:m'}$ (which it is, as a prefix). **The MRL training objective enforces that this chain of mutual information values is as large as possible at each checkpoint $m \in \mathcal{M}$.**

A formal capacity bound: for a linear classification head $\mathbf{W}^{(m)} \in \mathbb{R}^{L \times m}$ and softmax loss, the minimum achievable loss is lower bounded by $H(y | z_{1:m}) \geq H(y) - I(z_{1:m}; y)$. The MRL objective is thus equivalent (at the optimum) to maximizing $\sum_{m \in \mathcal{M}} c_m \cdot I(z_{1:m}; y)$ subject to the prefix structure.

> [!WARNING] Linear classifier limitation
> The capacity bound assumes a *linear* classifier head $\mathbf{W}^{(m)}$. Nonlinear downstream tasks may extract more information from $z_{1:m}$. The empirical results in §6 use linear probes throughout, which is the conventional evaluation but may underestimate the value of low-dimensional MRL representations for nonlinear tasks.

> [!QUESTION] Exercise 8: Capacity monotonicity under the prefix constraint
> *This exercise formalizes why adding dimensions cannot hurt under the MRL nesting structure.*
>
> Let $f^*(m)$ denote the optimal MRL classification loss using only $z_{1:m}$. Prove that $f^*(m)$ is non-increasing in $m$ for MRL (i.e., adding more prefix dimensions cannot increase the optimal loss). Does this guarantee hold for SVD/PCA truncation of a standard fixed-feature embedding? Explain why or why not. (*Hint:* Consider whether the feasible set for the $(m+1)$-dimensional problem contains the $m$-dimensional optimum as a special case.)

---

## 6. Empirical Results 📊

### 6.1 ImageNet Classification

The primary benchmark is linear probing on ImageNet-1K (1000 classes, 1.28M training images). Results for ResNet50 ($d = 2048$) with MRL vs. fixed-feature (FF) independently trained baselines and SVD compression:

| Embedding dim | FF top-1 | MRL top-1 | MRL-E top-1 | SVD of full FF |
|---|---|---|---|---|
| 8 | 65.3% | **66.6%** | 56.7% | 2.3% |
| 16 | 67.5% | **68.3%** | 67.5% | 14.1% |
| 32 | 70.2% | **71.4%** | 70.9% | 41.7% |
| 64 | 75.3% | **75.8%** | 75.4% | 62.0% |
| 128 | 76.0% | **76.4%** | 76.2% | 70.2% |
| 512 | 76.2% | **76.7%** | 76.4% | 75.8% |
| 2048 | **76.9%** | 76.8% | 76.5% | 76.9% |

Key observations:
- **MRL matches or exceeds FF at every dimension.** At small $m$, the MRL signal improves prefix quality; at full $d$, the extra objectives act as mild regularization.
- **SVD collapses at small $m$.** Without structured training, post-hoc compression is catastrophic at $d < 128$.
- **MRL-E is competitive above $m = 32$ but weaker at $m = 8$**, consistent with the gradient interference analysis in §2.3.

![Figure 3 from Kusupati et al. (2022): Top-1 linear classification accuracy on ImageNet-1K for ResNet50](https://arxiv.org/html/2205.13147v4/x9.png)
*Figure 3 (Kusupati et al., 2022): Top-1 linear classification accuracy on ImageNet-1K as a function of representation size for ResNet50. MRL and MRL-E (labeled "Int" variants) consistently match or exceed FF baselines, while SVD and random projections collapse at low dimension.*

> [!QUESTION] Exercise 9: Accuracy-dimension scaling
> *This exercise explores the empirical scaling of accuracy with dimension and relates it to the information-theoretic bound from §5.3.*
>
> Using the table above, fit a log-linear model $\text{accuracy}(m) \approx \alpha \log(m) + \beta$ to the MRL top-1 results. Estimate $\alpha$ and $\beta$ via least squares. Does this fit hold for SVD? At what dimension does SVD "catch up" to the log-linear MRL trend? Interpret the difference in terms of the capacity arguments from §5.3.

### 6.2 Image Retrieval

For 1-NN retrieval (find the nearest neighbor in embedding space and report its class label), MRL outperforms FF baselines by up to 2% at low dimensions while remaining competitive at full capacity. The adaptive retrieval results are particularly striking:

- **Shortlist $D_s = 16$, re-rank $D_r = 2048$:** achieves the same mAP@10 as single-shot 2048-dim retrieval, while reducing theoretical FLOPs by $\sim$128× and wall-clock time by 14×.
- **Funnel retrieval** (multi-stage cascade, see §4.2) achieves the same accuracy at approximately 1% of the computational cost of brute-force exact search.

**Why does shortlisting at $D_s = 16$ work?** Because MRL's 16-dimensional prefix captures coarse semantic structure (superclass identity), sufficient to eliminate 99.9% of clearly irrelevant items from the database. The re-ranking step then applies fine-grained discrimination only to the small shortlist.

![Figure 5 from Kusupati et al. (2022): mAP@10 image retrieval on ImageNet-1K across representation sizes](https://arxiv.org/html/2205.13147v4/x14.png)
*Figure 5 (Kusupati et al., 2022): mAP@10 for image retrieval on ImageNet-1K as a function of representation size. MRL significantly outperforms FF, SVD, and random baselines at low dimensions. The MRL-Int (interpolated) curve confirms smooth accuracy across all intermediate dimensions.*

### 6.3 Few-Shot and Long-Tail Learning

MRL is evaluated on the FLUID benchmark (long-tail distribution, few-shot novel classes):

- **Head classes:** MRL matches FF accuracy, with no degradation from the multi-granularity objective.
- **Novel tail classes:** MRL gains up to 2% over FF. Hypothesis: the coarse-to-fine structure learned by MRL generalizes better to low-shot regimes, where fine-grained distinctions are less learnable from few examples.

### 6.4 Robustness Benchmarks

MRL representations are tested on distribution-shifted ImageNet variants:

| Benchmark | Shift type | MRL vs. FF |
|---|---|---|
| ImageNet-V2 | Natural variation | Comparable |
| ImageNet-A | Adversarial examples | +0.6% |
| ImageNet-R | Renditions | Comparable |
| ImageNet-Sketch | Sketch domain | Comparable |

*The +0.6% improvement on ImageNet-A suggests that multi-granularity training produces representations that are more robust to adversarial perturbations*, possibly because the lower-dimensional prefix training implicitly encourages smoother, more class-aligned intermediate representations.

---

## 7. Extensions and Follow-up Work 🌐

### 7.1 2D Matryoshka Sentence Embeddings

**[2D-MRL (Chen et al., 2024)](https://arxiv.org/abs/2402.14776)** identifies a limitation in the original MRL formulation: even with a 32-dimensional prefix, the encoder $F$ still requires a full forward pass through all transformer layers, which dominates inference time and memory. 2D-MRL extends nesting to a second axis: *both embedding dimension and transformer depth.*

**Definition (2D Nesting).** Let $\mathcal{L} = \{l_1 < l_2 < \cdots < l_T\} \subset [L]$ be a set of intermediate transformer layer indices. The 2D-MRL loss is:

$$\mathcal{L}_{\text{2D-MRL}} = \sum_{l \in \mathcal{L}} \sum_{m \in \mathcal{M}} c_{l,m} \cdot \ell\!\left(\mathbf{W}^{(l,m)} \cdot h^{(l)}_{1:m};\; y\right)$$

where $h^{(l)} \in \mathbb{R}^d$ is the CLS-token hidden state at transformer layer $l$. This allows deployment at $(l, m)$ pairs: e.g., use only the first 6 layers and 128-dimensional prefix for fast inference, or all 12 layers and 768-dimensional embedding for maximum accuracy.

### 7.2 Matryoshka-Adaptor

**[Matryoshka-Adaptor (Zhu et al., EMNLP 2024)](https://arxiv.org/abs/2407.20243)** addresses the case where a practitioner has a fixed, already-trained LLM embedding model (e.g., a deployed production model) and wants to add MRL flexibility without full retraining. The approach inserts a lightweight adapter module after the frozen LLM, trained with the MRL objective.

Results: $\sim$2× dimensionality reduction (unsupervised) and $\sim$6× reduction (supervised) with no loss in retrieval performance on BEIR benchmarks, demonstrating that the MRL signal can be injected post-hoc with substantially less compute than full retraining.

### 7.3 Multimodal Extensions

**Matryoshka Multimodal Models (M3, Cai et al., 2024)** and the **Matryoshka Query Transformer (MQT, NeurIPS 2024)** extend MRL to the visual token dimension of vision-language models (VLMs). Rather than nesting embedding *coordinates*, these methods nest the *number of visual tokens*: a VLM can process an image as $m$ tokens (for any $m$ up to a maximum), enabling adaptive compute based on image complexity.

The nesting structure is analogous: the first $m$ visual tokens capture coarse spatial structure, and additional tokens refine the representation. This enables a 4× or greater reduction in visual token count for many queries with minimal answer quality degradation.

### 7.4 MRL in Production: OpenAI text-embedding-3

OpenAI's **text-embedding-3-small** and **text-embedding-3-large** models (announced January 2024) explicitly adopt MRL training to enable dimension shortening at inference. Key facts:

- text-embedding-3-large has native dimension $d = 3072$ but can be shortened to any $m \leq 3072$.
- *Surprisingly,* text-embedding-3-large truncated to $m = 256$ **outperforms** text-embedding-ada-002 at $m = 1536$ on the [MTEB benchmark](https://huggingface.co/spaces/mteb/leaderboard) — a 6× dimension reduction with improved accuracy, achieved purely through MRL training.
- Nomic's `nomic-embed-text-v1` and Alibaba's `gte-multilingual-base` have since adopted the same approach.

**This industrial adoption validates MRL's core thesis at billion-parameter scale.** The flexibility to shorten embeddings reduces storage costs for large-scale retrieval systems (e.g., 6× reduction in vector database storage for equivalent accuracy).

### 7.5 MatFormer: Nesting at the Architecture Level

**[MatFormer (Devvrit et al., NeurIPS 2024)](https://arxiv.org/abs/2310.07707)** from Google DeepMind extends the Matryoshka nesting principle to transformer *widths* rather than embedding prefixes. A large E4B model is trained so that the E2B, E1B, etc. sub-models are nested inside it — enabling elastic deployment from a single set of weights. This is architecturally distinct from MRL (which operates on the output embedding space) but shares the same nesting philosophy.

> [!QUESTION] Exercise 10: Matryoshka principle unification
> *This exercise situates MRL within a broader framework of nested model families.*
>
> Identify the common abstract structure shared by: (a) MRL on embedding prefixes, (b) 2D-MRL on transformer layers × embedding dimensions, (c) MatFormer on transformer widths. Formalize this as a partial order on computational budgets $(b_1, b_2, \ldots)$ and a family of models indexed by that order. What property must the training loss satisfy for "nesting" to be beneficial? (Hint: consider what happens to the gradient when a larger model subsumes a smaller one.)

> [!QUESTION] Exercise 11: Implementing adaptive retrieval in NumPy
> *This exercise implements the shortlisting and re-ranking pipeline and measures its empirical speedup.*
>
> ```python
> import numpy as np
> import time
>
> def adaptive_retrieval(query, database, D_s, D_r, K, top_k=10):
>     """
>     query: (d,) float array — full-dimensional query embedding
>     database: (N, d) float array — N full-dimensional database embeddings
>     D_s: int — shortlisting dimension (< D_r)
>     D_r: int — re-ranking dimension
>     K: int — shortlist size
>     Returns: indices of top_k nearest neighbors by D_r distance
>     """
>     # TODO: implement shortlisting phase (use query[:D_s] and database[:, :D_s])
>     # TODO: implement re-ranking phase (use query[:D_r] and shortlist[:, :D_r])
>     pass
>
> # Generate random data (N=100000, d=2048)
> N, d = 100_000, 2048
> db = np.random.randn(N, d).astype(np.float32)
> q = np.random.randn(d).astype(np.float32)
>
> # Measure and compare wall-clock times for:
> # (a) Brute-force search at d=2048
> # (b) Adaptive retrieval with D_s=32, D_r=2048, K=1000
> # Report the speedup ratio
> ```

> [!QUESTION] Exercise 12: Sensitivity of MRL to the choice of M
> *This exercise investigates whether the geometric halving schedule for M is necessary or merely conventional.*
>
> Consider training MRL on a small dataset (e.g., CIFAR-10 with a ResNet-18 backbone, $d = 512$) with three choices of $\mathcal{M}$:
> (a) Geometric: $\mathcal{M} = \{8, 16, 32, 64, 128, 256, 512\}$
> (b) Arithmetic: $\mathcal{M} = \{73, 146, 219, 292, 365, 438, 512\}$ (linear spacing)
> (c) Endpoint-heavy: $\mathcal{M} = \{8, 16, 256, 512\}$ (only 4 checkpoints)
>
> For each, measure test accuracy at all intermediate dimensions $m \notin \mathcal{M}$ (e.g., $m = 24, 48, \ldots$). Does the interpolation property from §5.2 hold for all three? Which schedule produces the smoothest interpolation, and why?

---

## 8. References

| Reference Name | Brief Summary | Link to Reference |
|---|---|---|
| [Kusupati et al., NeurIPS 2022 — Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) | Original MRL paper; introduces the nested embedding framework, MRL-E, adaptive retrieval, and empirical benchmarks across vision, language, and multimodal models | [arxiv.org/abs/2205.13147](https://arxiv.org/abs/2205.13147) |
| [Rege, 2024 — MRL from the Ground Up (blog)](https://aniketrege.github.io/blog/2024/mrl/) | Pedagogical blog post with the Matryoshka doll analogy, t-SNE visualizations, and intuitions for why MRL outperforms PCA | [aniketrege.github.io/blog/2024/mrl/](https://aniketrege.github.io/blog/2024/mrl/) |
| [Chen et al., 2024 — 2D Matryoshka Sentence Embeddings](https://arxiv.org/abs/2402.14776) | Extends MRL to nest across both embedding dimension and transformer depth; enables layer-level adaptive inference | [arxiv.org/abs/2402.14776](https://arxiv.org/abs/2402.14776) |
| [Zhu et al., EMNLP 2024 — Matryoshka-Adaptor](https://arxiv.org/abs/2407.20243) | Post-hoc MRL adapter for frozen LLMs; achieves 2–6× dimensionality reduction with no retrieval performance loss | [arxiv.org/abs/2407.20243](https://arxiv.org/abs/2407.20243) |
| [Cai et al., 2024 — Matryoshka Multimodal Models (M3)](https://arxiv.org/abs/2405.17430) | Extends nesting to visual token count in VLMs; 4× token reduction with minimal quality loss | [arxiv.org/abs/2405.17430](https://arxiv.org/abs/2405.17430) |
| [MQT — Matryoshka Query Transformer, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/59c147c7d4fdb732daea3064eab949bf-Paper-Conference.pdf) | Encodes images into variable numbers of visual tokens; nests token count for compute-adaptive VLM inference | [openreview.net/forum?id=B1vGiSgELw](https://openreview.net/forum?id=B1vGiSgELw) |
| [Devvrit et al., NeurIPS 2024 — MatFormer](https://arxiv.org/abs/2310.07707) | Nests transformer widths; trains E4B so that E2B, E1B sub-models are embedded inside it for elastic deployment | [arxiv.org/abs/2310.07707](https://arxiv.org/abs/2310.07707) |
| [OpenAI text-embedding-3 announcement, 2024](https://openai.com/blog/new-embedding-models-and-api-updates) | Production deployment of MRL in OpenAI's embedding API; text-embedding-3-large at 256-dim outperforms ada-002 at 1536-dim | [openai.com/blog/new-embedding-models-and-api-updates](https://openai.com/blog/new-embedding-models-and-api-updates) |
| [RAIVNLab/MRL — Official GitHub](https://github.com/RAIVNLab/MRL) | Official code and pretrained models for MRL | [github.com/RAIVNLab/MRL](https://github.com/RAIVNLab/MRL) |
| [HuggingFace — Introduction to Matryoshka Embedding Models](https://huggingface.co/blog/matryoshka) | Tutorial on training Matryoshka embedding models with sentence-transformers; practical guide with code | [huggingface.co/blog/matryoshka](https://huggingface.co/blog/matryoshka) |
