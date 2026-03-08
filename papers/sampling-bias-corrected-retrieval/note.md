# Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations

Yi, Yang, Hong, Cheng, Heldt, Kumthekar, Zhao, Wei, Chi. RecSys 2019. Google.

## TL;DR

| Dimension | Prior State | This Paper |
|-----------|-------------|------------|
| **Training objective** | Batch softmax treats in-batch items as uniform negatives; implicitly minimizes loss under frequency-reweighted partition function $\sum_j q_j e^{s_j}$ | Corrects to target full softmax partition function $\sum_j e^{s_j}$ via logQ correction: subtract $\log \hat{p}_j$ from each item logit before softmax |
| **Bias source** | Never identified; practitioners observed that popular items are over-penalized but attributed it to data imbalance | Formally derived: uncorrected batch softmax is an unbiased estimator of the wrong objective; bias scales with variance of $\log q_j$ across items |
| **Frequency estimation** | Requires precomputed per-item frequency table over fixed vocabulary | Online streaming estimator via exponential moving average on inter-arrival times; handles non-stationary distributions and new items without recomputation |
| **Architecture** | Two-tower inner product model (unchanged) | Two-tower inner product model (unchanged) — correction is training-only, zero inference cost |
| **Deployment** | ANN index on raw item embeddings | ANN index on corrected-training item embeddings — same serving infrastructure |
| **Empirical gains** | — | +8% Recall@1 on Wikipedia link prediction; statistically significant live metric gains on YouTube retrieval |

---

## Table of Contents

1. [[#1. Motivation and Problem Setting|Motivation and Problem Setting]]
   - [[#The Two-Phase Retrieval-Ranking Architecture|The Two-Phase Retrieval-Ranking Architecture]]
   - [[#Challenges at Industrial Scale|Challenges at Industrial Scale]]
2. [[#2. The Two-Tower Model|The Two-Tower Model]]
   - [[#Formal Setup|Formal Setup]]
   - [[#Full Softmax Objective|Full Softmax Objective]]
   - [[#Batch Softmax and the Sampling Bias Problem|Batch Softmax and the Sampling Bias Problem]]
3. [[#3. Sampling Bias Correction|Sampling Bias Correction]]
   - [[#The LogQ Correction|The LogQ Correction]]
   - [[#Corrected Batch Loss|Corrected Batch Loss]]
   - [[#Connection to Importance Sampling|Connection to Importance Sampling]]
4. [[#4. Streaming Frequency Estimation|Streaming Frequency Estimation]]
   - [[#Problem Formulation|Problem Formulation]]
   - [[#Algorithm 2: Single Hash Estimation|Algorithm 2: Single Hash Estimation]]
   - [[#Proposition 4.1: Bias and Variance Analysis|Proposition 4.1: Bias and Variance Analysis]]
   - [[#Algorithm 3: Multiple Hashings|Algorithm 3: Multiple Hashings]]
5. [[#5. YouTube Neural Retrieval System|YouTube Neural Retrieval System]]
   - [[#Model Architecture|Model Architecture]]
   - [[#Sequential Training|Sequential Training]]
   - [[#Indexing and Serving|Indexing and Serving]]
6. [[#6. Experiments|Experiments]]
   - [[#Frequency Estimation Simulation|Frequency Estimation Simulation]]
   - [[#Wikipedia Link Prediction|Wikipedia Link Prediction]]
   - [[#YouTube Offline and Live Experiments|YouTube Offline and Live Experiments]]
7. [[#References|References]]

---

## 1. Motivation and Problem Setting

### The Two-Phase Retrieval-Ranking Architecture

Industrial recommendation systems connecting billions of users to corpora of tens of millions of items cannot score every candidate at query time — the latency would be prohibitive. The standard solution is a two-phase cascade:

1. **Retrieval (nomination)**: a lightweight *retrieval model* retrieves $O(10^2)$ candidates from $O(10^7)$ items using *approximate nearest neighbor* search on learned embeddings. Speed dominates — the model must be indexable into a static *embedding index*.
2. **Ranking**: a fully-featured *ranking model* re-ranks the retrieved candidates using richer features and multiple objectives (clicks, watch time, etc.). Latency allows more computation but only over the small candidate set.

This paper addresses the retrieval stage. The key architectural constraint is that the scoring function at retrieval time must decompose as an inner product of independently computed query and item vectors:

$$s(x, y) = \langle u(x, \theta), v(y, \theta) \rangle$$

This decomposition is what enables the item embeddings to be precomputed and indexed offline, supporting sub-linear MIPS (*maximum inner product search*) at inference time. Any scoring function that couples $x$ and $y$ in a non-decomposable way — such as concatenating them before a feedforward network — cannot be indexed and is thus incompatible with large-corpus retrieval.

### Challenges at Industrial Scale

Three challenges make this setting qualitatively harder than small-scale retrieval:

1. **Power-law item distribution**: a small fraction of popular items accounts for most user interactions. Training data is extremely sparse for long-tail items. Naive *in-batch negatives* sampling over-represents popular items as negatives, introducing a systematic bias toward penalizing popular items — precisely the opposite of what a well-calibrated ranking should do.

2. **Vocabulary shift**: the item corpus changes daily (new videos uploaded, old videos removed). Unlike language modeling where the vocabulary is fixed, item frequencies and identities are non-stationary. Any algorithm that requires a fixed item vocabulary or a static frequency table is inappropriate for this setting.

3. **Cold-start**: new items have no interaction history and thus no training signal from ID-based features alone. Content features (visual embeddings, audio features, metadata) are essential for generalizing to unseen items. This motivates deep item towers that can process structured content features, not just ID embeddings.

---

## 2. The Two-Tower Model

### Formal Setup

Let the training dataset be $\mathcal{T} = \{(x_i, y_i, r_i)\}_{i=1}^T$ where:
- $x_i \in \mathcal{X}$ is the **query** (user + context features)
- $y_i \in \mathcal{Y}$ is the **positive item** (the item the user interacted with)
- $r_i \in \mathbb{R}$ is the **reward** (degree of user engagement, e.g., watch fraction)

Two parameterized embedding functions map queries and items to a shared $k$-dimensional space:

$$u : \mathcal{X} \times \mathbb{R}^d \to \mathbb{R}^k, \qquad v : \mathcal{Y} \times \mathbb{R}^d \to \mathbb{R}^k$$

where $\theta \in \mathbb{R}^d$ are the shared model parameters. Both functions are realized as multi-layer feedforward networks (the "left" and "right" towers). The score between query $x$ and item $y$ is:

$$s(x, y; \theta) = \langle u(x, \theta),\, v(y, \theta) \rangle$$

**Normalization and temperature.** Empirically, L2 normalizing both embeddings before the inner product stabilizes training and improves retrieval quality:

$$u \leftarrow \frac{u}{\|u\|_2}, \qquad v \leftarrow \frac{v}{\|v\|_2}$$

A temperature parameter $\tau > 0$ sharpens or softens the score distribution:

$$s(x, y) = \frac{\langle u(x, \theta), v(y, \theta) \rangle}{\tau}$$

Temperature is treated as a hyperparameter tuned to maximize recall. With L2 normalization, the inner product equals the cosine similarity, so $s(x,y) \in [-1/\tau, 1/\tau]$.

### Full Softmax Objective

Given $M$ candidate items $\{y_j\}_{j=1}^M$, the probability of item $y$ given query $x$ under the softmax model is:

$$P(y \mid x;\, \theta) = \frac{e^{s(x,y)}}{\sum_{j=1}^{M} e^{s(x, y_j)}}$$

The training objective is a weighted log-likelihood (reward-weighted cross-entropy):

$$\mathcal{L}_T(\theta) = -\frac{1}{T} \sum_{i=1}^{T} r_i \cdot \log P(y_i \mid x_i;\, \theta)$$

When $r_i = 1$ for all $i$ this reduces to standard cross-entropy. The reward weighting allows the model to upweight high-quality positive interactions (e.g., a fully watched video) and downweight low-quality ones (e.g., a click with immediate abandonment).

The full softmax over $M = O(10^7)$ items is computationally intractable for each gradient step. The *partition function* $\sum_{j=1}^M e^{s(x, y_j)}$ requires computing $v(y_j, \theta)$ for all $M$ items per step — infeasible.

### Batch Softmax and the Sampling Bias Problem

The standard scalable approximation is *batch softmax*: treat the $B-1$ other items in the current mini-batch as negatives for each query. Given a batch $\{(x_i, y_i, r_i)\}_{i=1}^B$:

$$P_B(y_i \mid x_i;\, \theta) = \frac{e^{s(x_i, y_i)}}{\sum_{j=1}^{B} e^{s(x_i, y_j)}}$$

**The bias.** In-batch items are drawn from the training distribution, which is typically *power-law distribution* skewed: popular items appear in far more training pairs $(x, y)$ and therefore in far more batches. Item $j$'s probability of appearing as a negative in a batch of size $B$ is approximately $p_j \approx B \cdot q_j$, where $q_j \propto \text{item frequency}$.

This means popular items are systematically over-represented as negatives. The batch softmax effectively trains with the wrong negative distribution — it approximates full softmax with negative sampling distribution $\propto q_j$ rather than uniform. **This introduces a bias: the model is penalized heavily for assigning high scores to popular items, even when those scores are appropriate, because popular items almost always appear as negatives.**

---

## 3. Sampling Bias Correction

### The LogQ Correction

The correction is inspired by the *logQ correction* for sampled softmax (Bengio and Sénécal 2008). If items are sampled as negatives with probability $p_j$, the corrected logit subtracts the log sampling probability:

$$s^c(x_i, y_j) = s(x_i, y_j) - \log p_j$$

**Intuition.** The full softmax with sampling correction approximates the denominator of the full softmax over $M$ items:

$$\sum_{j=1}^M e^{s(x,y_j)} \approx \frac{1}{K} \sum_{j \in S} \frac{e^{s(x,y_j)}}{p_j} = \frac{1}{K} \sum_{j \in S} e^{s(x,y_j) - \log p_j} = \frac{1}{K} \sum_{j \in S} e^{s^c(x,y_j)}$$

where $S$ is the set of sampled negatives and the approximation is an *importance-weighted* Monte Carlo estimate. The log-correction is thus a form of *importance sampling* reweighting that debiases the partition function estimate.

A popular item with $p_j$ close to 1 receives a large negative correction $-\log p_j \approx 0$ (little correction needed, already expected to appear). A rare item with small $p_j$ receives a large negative correction $-\log p_j \gg 0$, boosting its logit to compensate for being under-sampled. This restores calibration: after correction, the effective score reflects intrinsic relevance, not sampling artifacts.

### Corrected Batch Loss

With the corrected logits, the corrected batch softmax probability is:

$$P_B^c(y_i \mid x_i;\, \theta) = \frac{e^{s^c(x_i, y_i)}}{e^{s^c(x_i, y_i)} + \displaystyle\sum_{\substack{j \in [B] \\ j \neq i}} e^{s^c(x_i, y_j)}}$$

The batch training loss is:

$$\mathcal{L}_B(\theta) = -\frac{1}{B} \sum_{i=1}^{B} r_i \cdot \log P_B^c(y_i \mid x_i;\, \theta) \tag{4}$$

Gradient descent on $\mathcal{L}_B(\theta)$ with the corrected logits forms the inner loop of Algorithm 1.

### Connection to Importance Sampling

The correction can also be understood from first principles via the sampled softmax identity. For a softmax classifier with $M$ classes, sampling $K$ negatives from distribution $\{p_j\}$ and reweighting by $1/p_j$ gives an unbiased estimate of the partition function (in expectation over the sampling). The batch softmax is a special case with $K = B-1$ and $p_j$ proportional to item frequency. The logQ subtraction implements this reweighting in log-space, which is more numerically stable than multiplying by $1/p_j$ directly.

**Key difference from sampled softmax.** Classical sampled softmax (Bengio and Sénécal 2008) requires a fixed vocabulary and a predefined sampling distribution (typically unigram). Here the item vocabulary is non-stationary (new videos appear daily) and the sampling distribution changes over time. The frequency estimator in Section 4 addresses this.

---

## 4. Streaming Frequency Estimation

### Problem Formulation

The correction requires the sampling probability $p_j$ for each item $y_j$ appearing in the batch. Since $p_j \approx B \cdot q_j$ (batch size times item marginal frequency in the data stream), estimating $p_j$ reduces to estimating the item frequency $q_j$.

The requirements are stringent:
- **No fixed vocabulary**: cannot precompute a frequency table for all items, since new items appear continuously.
- **Adaptive to distribution shift**: item frequencies change over time (viral videos spike, old content decays). A static frequency table becomes stale.
- **Distributed**: training runs across many workers; the estimator must be consistent across workers without expensive global synchronization.

The key insight: instead of estimating the marginal frequency $q_j$ directly, estimate $\delta_j = $ the average number of training steps between two consecutive appearances of item $j$ in any batch. Then $p_j \approx 1/\delta_j$.

### Algorithm 2: Single Hash Estimation

Maintain two hash arrays $A$ and $B$ of size $H$ and a hash function $h : \mathcal{Y} \to [H]$:
- $A[h(y)]$: the global step at which item $y$ was last seen
- $B[h(y)]$: the running *exponential moving average* of *inter-arrival time*s $\delta_y$

At global step $t$, when item $y$ appears in a batch:

$$B[h(y)] \leftarrow (1-\alpha) \cdot B[h(y)] + \alpha \cdot (t - A[h(y)]) \tag{6}$$

$$A[h(y)] \leftarrow t$$

The update rule is online SGD with fixed learning rate $\alpha$ applied to the loss $\frac{1}{2}(\delta - \Delta)^2$ where $\Delta = t - A[h(y)]$ is the observed inter-arrival time. The estimated sampling probability is $\hat{p}(y) = 1/B[h(y)]$.

**Why global step, not wall clock?** The global step is synchronized across all workers via parameter servers (each worker increments a shared counter after processing one batch). This makes the estimator implicitly consistent across workers without explicit communication — each worker reads and updates the same $A$, $B$ arrays stored on parameter servers.

### Proposition 4.1: Bias and Variance Analysis

Let $\{\Delta_1, \Delta_2, \ldots, \Delta_t\}$ be i.i.d. samples of the inter-arrival time $\Delta$ with mean $\delta = \mathbb{E}[\Delta]$. The SGD update with learning rate $\alpha$ gives:

$$\delta_i = (1-\alpha) \delta_{i-1} + \alpha \Delta_i$$

**Bias:**

$$\mathbb{E}[\delta_t] - \delta = (1-\alpha)^t \delta_0 - (1-\alpha)^{t-1} \delta \tag{7}$$

The bias decays geometrically to zero: $|\mathbb{E}[\delta_t] - \delta| \to 0$ as $t \to \infty$. The rate of decay is controlled by $\alpha$: larger $\alpha$ means faster convergence but higher steady-state variance.

**Proof sketch of (7):** By linearity of expectation, $\mathbb{E}[\delta_i] = (1-\alpha)\mathbb{E}[\delta_{i-1}] + \alpha\delta$. This is a linear recurrence with fixed point $\delta$ and decay factor $(1-\alpha)$. By induction from the initial condition $\delta_0$:
$$\mathbb{E}[\delta_t] = (1-\alpha)^t \delta_0 + \delta \cdot \sum_{k=0}^{t-1}(1-\alpha)^k \alpha = (1-\alpha)^t \delta_0 + \delta \cdot [1 - (1-\alpha)^t]$$

Subtracting $\delta$ gives (7). An ideal initialization $\delta_0 = \delta/(1-\alpha)$ makes the estimator unbiased at every step.

**Variance:**

$$\mathbb{E}\left[(\delta_t - \mathbb{E}[\delta_t])^2\right] \leq (1-\alpha)^{2t}(\delta_0 - \delta)^2 + \alpha \cdot \mathbb{E}\left[(\Delta_1 - \delta)^2\right] \tag{8}$$

The first term decays geometrically (effect of initialization error). The second term is a constant floor of order $\alpha \cdot \text{Var}(\Delta)$, which decreases with smaller $\alpha$. *The variance-adaptability tradeoff: $\alpha$ must be large enough to track distribution shift but small enough to keep estimation variance low.* In practice $\alpha = 0.01$ is used.

### Algorithm 3: Multiple Hashings

*Hash collisions* in Algorithm 2 cause $B[h(y)]$ to track the union of items sharing the same bucket, under-estimating $\delta$ and over-estimating frequency. Multiple hashings mitigate this via a count-min-sketch-style approach:

Maintain $m$ independent pairs $(A_i, B_i)$ with independent hash functions $\{h_i\}_{i=1}^m$. Each array is updated as in Algorithm 2. At inference:

$$\hat{p}(y) = \frac{1}{\max_{i \in [m]} B_i[h_i(y)]}$$

Taking the maximum over $m$ independent estimates exploits the fact that the true inter-arrival time $\delta_y$ is the maximum possible value (collisions only pull the estimate down). The maximum of $m$ independent estimates upper-bounds the true value, which translates to a lower-bound on the frequency estimate — directionally correcting the collision bias.

Total parameter budget is held fixed (arrays of size $H/m$ each with $m$ hash functions, same total as $H$ with $m=1$), so multiple hashings reduce variance at no additional memory cost.

---

## 5. YouTube Neural Retrieval System

### Model Architecture

The YouTube retrieval model follows the *two-tower model* structure of Figure 2. Embedding sharing is a key architectural decision: the same video ID embedding table is used for seed video features, candidate video features, and past watch history features. This three-way sharing serves two purposes: (1) parameter efficiency — a single table covers all uses of video IDs; (2) improved generalization — the model learns a single consistent representation of each video across all contexts in which it appears.

**Query (left) tower** receives:
- Seed video features: video ID, channel ID, categorical metadata (sparse IDs → embeddings)
- User watch history: bag-of-words over past $k$ video IDs (mean-pooled video ID embeddings)
- User profile features: views, likes, other engagement signals

**Candidate (right) tower** receives:
- Candidate video features: video ID, channel ID, categorical and dense features

Both towers are three-layer feedforward networks with hidden dimensions $[1024, 512, 128]$ and ReLU activations. The final 128-dimensional outputs are L2-normalized and scored by inner product. Out-of-vocabulary video IDs are mapped to random hash buckets, which have learnable embeddings — this handles fresh content without requiring a vocabulary expansion.

**Training label**: clicks are positive pairs, with reward $r_i$ reflecting engagement quality (e.g., $r_i = 0$ for immediately abandoned clicks, $r_i = 1$ for fully watched videos). The reward is used as the example weight in Equation (4).

### Sequential Training

*Sequential training* organizes training data by day. The trainer processes days sequentially from oldest to most recent. Once caught up to the latest day, it waits for the next day's data. This sequential structure has two benefits:

1. The model continuously adapts to distribution shift (new videos, changing user preferences, seasonal trends).
2. The *streaming frequency estimation* in Algorithm 2 naturally integrates into this framework — the moving average adapts online as the data distribution shifts day by day.

### Indexing and Serving

The offline indexing pipeline runs periodically (every few hours) in three stages:

1. **Candidate generation**: select videos from the YouTube corpus meeting certain criteria (e.g., not removed, not violating policy).
2. **Embedding inference**: apply the candidate tower to compute $v(y, \theta)$ for each candidate video. This is a batch inference job using the trained model's right tower as a SavedModel.
3. **Index construction**: build an approximate MIPS index using tree-based and quantization-based methods (product quantization, coarse and product quantizers). The index supports sub-linear retrieval at serving time.

At serving time, the query tower computes $u(x, \theta)$ for the live user-context query and the index returns the top-$K$ candidates by approximate inner product. The index covers approximately 10M videos and is rebuilt frequently enough to surface fresh content.

---

## 6. Experiments

### Frequency Estimation Simulation

Simulation setup: $M = 1000$ items, batch size $B = 128$, array size $H = 5000$. Item sampling probabilities follow $q_i \propto i^2$ for the first $t = 10000$ steps, then switch to $q_i \propto (M-1-i)^2$.

Evaluation metric: total variation distance $\frac{1}{2|B|} \sum_i |\hat{p}_i - p_i|$ between estimated and true batch sampling probabilities.

**Learning rate effect**: all learning rates converge; higher $\alpha$ adapts faster after the distribution switch but settles at higher variance, consistent with the bound in Proposition 4.1.

**Multiple hashings**: Algorithm 3 with $m = 2$ or $m = 4$ hash functions (same total parameter budget as $m = 1$) consistently achieves lower estimation error across all steps — fewer collisions outweigh the reduced array size per hash function.

### Wikipedia Link Prediction

Dataset: English Wikipedia graph, 5.3M pages, 430M links. Task: given a source page, retrieve destination pages (intra-site links). Both towers share input feature embeddings (page URL, title n-grams, categories). Each tower has two ReLU layers of dimensions $[512, 128]$.

Three methods:
- `mse-gramian`: MSE loss on observed pairs + Gramian regularization for unseen pairs [Krichene et al. 2019]
- `plain-sfx`: batch softmax without correction (Equation 3)
- `correct-sfx`: sampling-bias-corrected batch softmax (Equation 4)

| Method | Recall@10 | Recall@50 | Recall@100 | Recall@300 |
|--------|-----------|-----------|------------|------------|
| mse-gramian | 0.0432 | 0.1326 | 0.2027 | 0.3530 |
| plain-sfx ($\tau = 0.07$) | 0.0643 | 0.2423 | 0.3746 | 0.5991 |
| correct-sfx ($\tau = 0.07$) | **0.1065** | **0.3079** | **0.4664** | **0.7234** |

The corrected softmax improves over uncorrected by $65\%$ in Recall@10. *Temperature must be tuned carefully — the optimal $\tau$ for plain-sfx and correct-sfx differ, and the ranking of methods is consistent across temperatures.*

### YouTube Offline and Live Experiments

**Offline**: 10M video corpus; model trained on billions of daily clicks across 15+ days of sequential data. Evaluation on 10% holdout. $r_i = 1$ for all clicks (simplified reward).

| Method | Recall@5 | Recall@10 | Recall@30 | Recall@50 |
|--------|----------|-----------|-----------|-----------|
| mse-gramian | 0.0554 | 0.0768 | 0.1149 | 0.1338 |
| plain-sfx ($\tau = 0.05$) | 0.2069 | 0.2728 | 0.3964 | 0.4586 |
| correct-sfx ($\tau = 0.05$) | **0.2150** | **0.2960** | **0.4537** | **0.5322** |

**Live A/B experiment**: treatment group receives recommendations augmented by candidates from the neural retrieval system (added to the nomination stage alongside existing nominators).

| System | Engagement Metric Improvement |
|--------|-------------------------------|
| plain-sfx | +0.20% |
| correct-sfx | +0.37% |

Both outperform the production baseline. The sampling-bias correction nearly doubles the engagement gain over plain-sfx in the live test, confirming that the bias is practically significant — not just a metric artifact.

---

## References

| Reference Name | Brief Summary | Link to Reference |
|---|---|---|
| Yi et al. (2019), "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations" | The paper itself; introduces two-tower retrieval with logQ bias correction and streaming frequency estimation for YouTube recommendations | https://doi.org/10.1145/3298689.3346996 |
| Bengio and Sénécal (2008), "Adaptive Importance Sampling to Accelerate Training of a Neural Probabilistic Language Model" | Original logQ correction for sampled softmax in language modeling; the bias correction in this paper is directly inspired by this work | https://doi.org/10.1109/TNN.2007.912312 |
| Covington, Adams, Sargin (2016), "Deep Neural Networks for YouTube Recommendations" | Predecessor YouTube recommendation system; establishes the two-phase retrieval-ranking pipeline and the use of deep networks for candidate generation | https://dl.acm.org/doi/10.1145/2959100.2959190 |
| Cheng et al. (2016), "Wide and Deep Learning for Recommender Systems" | Wide-and-deep framework for ranking; context for the retrieval-ranking cascade in which this paper's retrieval model operates | https://arxiv.org/abs/1606.07792 |
| Cormode and Muthukrishnan (2005), "Count-Min Sketch and Its Applications" | Foundational data structure for frequency estimation in streams with hash collisions; Algorithm 3's multiple-hashing design is directly inspired by the count-min sketch | https://doi.org/10.1016/j.jalgor.2003.12.001 |
| Krichene et al. (2019), "Efficient Training on Very Large Corpora via Gramian Estimation" | Baseline method (mse-gramian) used in experiments; extends MSE recommendation loss to non-linear models via Gramian computation | https://openreview.net/forum?id=rJe4ShAcF7 |
