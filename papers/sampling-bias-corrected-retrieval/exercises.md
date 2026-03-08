# Sampling-Bias-Corrected Retrieval: Exercises

## Table of Contents

1. [Derivation Problems](#derivation-problems)
   - [Problem 1: Gradient of the Softmax Loss and Its Contrastive Structure](#problem-1-gradient-of-the-softmax-loss-and-its-contrastive-structure)
   - [Problem 2: Bias of Uncorrected Batch Softmax](#problem-2-bias-of-uncorrected-batch-softmax)
   - [Problem 3: LogQ Correction as Importance-Weighted Estimation](#problem-3-logq-correction-as-importance-weighted-estimation)
   - [Problem 4: Streaming Estimator Bias and Variance](#problem-4-streaming-estimator-bias-and-variance)
   - [Problem 5: Temperature and Softmax Entropy](#problem-5-temperature-and-softmax-entropy)
2. [Conceptual Questions](#conceptual-questions)
   - [Problem 6: Inner Product Decomposability and MIPS Indexing](#problem-6-inner-product-decomposability-and-mips-indexing)
   - [Problem 7: When Does Sampling Bias Matter](#problem-7-when-does-sampling-bias-matter)
   - [Problem 8: Sequential vs. Shuffled Training Under Distribution Shift](#problem-8-sequential-vs-shuffled-training-under-distribution-shift)
   - [Problem 9: Reward Weighting and Its Effect on the Learned Distribution](#problem-9-reward-weighting-and-its-effect-on-the-learned-distribution)
   - [Problem 10: Multiple Hashings vs. Larger Single Array](#problem-10-multiple-hashings-vs-larger-single-array)
3. [Implementation Sketches](#implementation-sketches)
   - [Problem 11: Corrected Batch Softmax Loss](#problem-11-corrected-batch-softmax-loss)
   - [Problem 12: Distributed Streaming Frequency Estimation](#problem-12-distributed-streaming-frequency-estimation)
   - [Problem 13: Recall at K Evaluation Against a Full Corpus](#problem-13-recall-at-k-evaluation-against-a-full-corpus)

---

## Derivation Problems

### Problem 1: Gradient of the Softmax Loss and Its Contrastive Structure

Let the full softmax probability and weighted log-likelihood loss be as defined in the paper:

$$P(y \mid x;\, \theta) = \frac{e^{s(x,y)}}{\sum_{j=1}^M e^{s(x, y_j)}}, \qquad \mathcal{L}(\theta) = -\frac{1}{T}\sum_{i=1}^T r_i \log P(y_i \mid x_i;\, \theta)$$

where $s(x, y) = \langle u(x, \theta), v(y, \theta) \rangle$ and all embeddings are computed by neural networks with shared parameter vector $\theta$.

**(a)** Fix a single training example $(x, y^+, r)$ with $r = 1$ and $M$ candidate items $\{y_j\}_{j=1}^M$. Define the shorthand $p_j = P(y_j \mid x;\, \theta)$ for all $j$. Show that the gradient of the per-example loss $\ell = -\log P(y^+ \mid x;\, \theta)$ with respect to the score $s_j := s(x, y_j)$ is:

$$\frac{\partial \ell}{\partial s_j} = p_j - \mathbf{1}[y_j = y^+]$$

**(b)** Hence show that the gradient with respect to the item embedding $v(y^+, \theta)$ (treating $u(x, \theta)$ as fixed) can be written as:

$$\frac{\partial \ell}{\partial v(y^+, \theta)} = -u(x, \theta) + \sum_{j=1}^M p_j \cdot u(x, \theta) \cdot \mathbb{1}[\ldots]$$

More carefully: via the chain rule $\frac{\partial \ell}{\partial v(y_j)} = \frac{\partial \ell}{\partial s_j} \cdot u(x, \theta)$, show that the total gradient of $\ell$ with respect to the **query embedding** $u(x, \theta)$ is:

$$\frac{\partial \ell}{\partial u(x, \theta)} = \mathbb{E}_{j \sim P(\cdot \mid x;\, \theta)}[v(y_j, \theta)] - v(y^+, \theta)$$

Interpret this contrastive form: the gradient pushes $u$ toward the positive item and away from the **model-distribution-weighted average** of all item embeddings.

**(c)** Now suppose you have access to a set of $K$ negative samples $\{y_j^-\}_{j=1}^K$ drawn i.i.d. from some proposal distribution $q$ over items. Define the Monte Carlo approximation to the partition function:

$$\hat{Z}(x) = e^{s(x, y^+)} + \frac{1}{K} \sum_{j=1}^K \frac{e^{s(x, y_j^-)}}{q(y_j^-)}$$

Show that $\hat{Z}(x)$ is an unbiased estimator of $\sum_{y} e^{s(x,y)}$ (up to the normalization constant $M$) when $q$ is the uniform distribution over items. When $q = P(\cdot \mid x;\, \theta)$ (self-normalized importance sampling), what happens?

---

### Problem 2: Bias of Uncorrected Batch Softmax

Let $\{(x_i, y_i)\}_{i=1}^B$ be a random mini-batch where positive pairs $(x_i, y_i)$ are drawn i.i.d. from the training distribution. In-batch negatives are the items $\{y_j : j \neq i\}$ for each query $x_i$.

**(a)** Let $q_j$ denote the marginal probability of item $y_j$ appearing as a positive in the training distribution (item frequency). Explain why, in a batch of size $B$, item $j$ appears as a negative for query $x_i$ with probability approximately $p_j^{\text{batch}} \approx (B-1) q_j$ (for large $M \gg B$). Hence the effective negative distribution of the uncorrected batch softmax is proportional to $\{q_j\}$, not uniform.

**(b)** The uncorrected batch softmax approximates the full softmax loss as:

$$-\log P_B(y_i \mid x_i;\,\theta) \approx -\log \frac{e^{s_i^+}}{\frac{1}{B-1}\sum_{j \neq i} \frac{e^{s_{ij}}}{q_j} \cdot q_j}$$

where we write $s_i^+ = s(x_i, y_i)$ and $s_{ij} = s(x_i, y_j)$. Rewrite the uncorrected batch softmax loss as a biased estimator of the full softmax loss and identify the bias term explicitly. Specifically, show that:

$$\mathbb{E}\left[-\log P_B(y_i \mid x_i)\right] \approx -\log P_M^{(q)}(y_i \mid x_i) + \text{const}$$

where $P_M^{(q)}$ is a softmax with a **reweighted** partition function $\sum_j e^{s_{ij}} \cdot (q_j / \bar{q})$ for some $\bar{q}$. What is the effect of this reweighting on the trained model?

**(c)** Suppose the item distribution is Zipf: $q_j \propto j^{-\alpha}$ for $j = 1, \ldots, M$ and some $\alpha > 0$. Define the "popularity ratio" as the ratio of the most popular item's sampling probability to the median item's. Show that as $\alpha$ increases (heavier power law), this ratio grows, and hence the bias in the uncorrected estimator grows. For $\alpha = 0$ (uniform distribution), show that the bias vanishes.

---

### Problem 3: LogQ Correction as Importance-Weighted Estimation

The logQ correction replaces $s(x, y_j)$ with $s^c(x, y_j) = s(x, y_j) - \log p_j$ where $p_j$ is the probability of item $j$ appearing in a random batch.

**(a)** The full softmax partition function is $Z(x) = \sum_{j=1}^M e^{s(x, y_j)}$. In-batch items are sampled with probabilities $\{p_j\}$ (not necessarily uniform). Show that the importance-weighted estimator:

$$\hat{Z}_{\text{IS}}(x) = \frac{1}{B} \sum_{j \in \text{batch}} \frac{e^{s(x, y_j)}}{p_j}$$

satisfies $\mathbb{E}[\hat{Z}_{\text{IS}}(x)] = Z(x)$ (unbiased). Write $\frac{e^{s(x,y_j)}}{p_j} = e^{s(x,y_j) - \log p_j} = e^{s^c(x, y_j)}$ to show the logQ correction implements this importance weighting in log-space.

**(b)** Show that the corrected batch softmax probability:

$$P_B^c(y_i \mid x_i;\, \theta) = \frac{e^{s^c(x_i, y_i)}}{e^{s^c(x_i, y_i)} + \sum_{j \neq i} e^{s^c(x_i, y_j)}}$$

approximates the full softmax $P(y_i \mid x_i;\, \theta)$ with lower bias than $P_B$. Identify the residual source of bias (hint: Jensen's inequality and the concavity of $\log$).

**(c)** Consider the case where all items are equally frequent: $p_j = B/M$ for all $j$. Show that the logQ correction reduces to subtracting the constant $\log(B/M)$ from all logits, which does not change the softmax output at all. Conclude that the correction is only non-trivial when the item distribution is non-uniform — and that the magnitude of the correction scales with the variance of $\log p_j$ across items in the batch.

**(d)** The correction requires knowing $p_j$ exactly. In practice, $p_j$ is estimated with noise $\hat{p}_j = p_j + \epsilon_j$. Show that the corrected logit with noisy estimate is $\hat{s}^c_j = s_j - \log(p_j + \epsilon_j) \approx s_j - \log p_j - \epsilon_j/p_j$ for small $\epsilon_j$. What is the effect of estimation noise on the variance of the corrected batch loss? How does the sensitivity scale with $p_j$ (rare vs. popular items)?

---

### Problem 4: Streaming Estimator Bias and Variance

This problem fills in the details of Proposition 4.1 from the paper.

Let $\{\Delta_1, \Delta_2, \ldots\}$ be i.i.d. with mean $\delta = \mathbb{E}[\Delta]$ and variance $\sigma^2 = \text{Var}(\Delta)$. The online estimator with learning rate $\alpha \in (0,1)$ and initial value $\delta_0$ is:

$$\delta_t = (1 - \alpha)\,\delta_{t-1} + \alpha\,\Delta_t, \qquad t \geq 1$$

**(a)** Prove by induction that:

$$\delta_t = (1-\alpha)^t \delta_0 + \alpha \sum_{k=1}^t (1-\alpha)^{t-k} \Delta_k$$

Hence compute $\mathbb{E}[\delta_t]$ and derive the bias formula:

$$\mathbb{E}[\delta_t] - \delta = (1-\alpha)^t(\delta_0 - \delta)$$

Compare this to equation (7) in the paper and reconcile any difference in form.

**(b)** Compute $\text{Var}(\delta_t)$ exactly using the representation from part (a), exploiting the independence of $\Delta_1, \ldots, \Delta_t$. Show that:

$$\text{Var}(\delta_t) = \alpha^2 \sigma^2 \sum_{k=1}^t (1-\alpha)^{2(t-k)} = \frac{\alpha \sigma^2}{2-\alpha}\left[1 - (1-\alpha)^{2t}\right]$$

As $t \to \infty$, the variance converges to the **steady-state variance** $\sigma_\infty^2 = \frac{\alpha \sigma^2}{2 - \alpha} \approx \frac{\alpha \sigma^2}{2}$ for small $\alpha$. Confirm this is consistent with the bound in equation (8) of the paper.

**(c)** The learning rate $\alpha$ governs a bias-variance tradeoff. Define the **mean squared error** $\text{MSE}(\delta_t) = \text{Bias}(\delta_t)^2 + \text{Var}(\delta_t)$. Using the results of (a) and (b), find the value of $\alpha^*$ that minimizes $\text{MSE}(\delta_t)$ at steady state (as $t \to \infty$). Show that $\alpha^* \to 0$ as $\sigma^2 \to 0$ (low-noise setting) and $\alpha^* \to 1$ as $\sigma^2 \to \infty$.

**(d)** Suppose the true mean $\delta$ undergoes a step change at step $t_0$: $\delta \to \delta + \Delta\delta$ for $t > t_0$. Derive the expected tracking lag — the number of steps $\tau_{1/2}$ after which $|\mathbb{E}[\delta_t] - (\delta + \Delta\delta)|$ has halved. Show $\tau_{1/2} = \log(2)/|\log(1-\alpha)| \approx \log(2)/\alpha$ for small $\alpha$. Interpret: smaller $\alpha$ means slower adaptation to distribution shift.

---

### Problem 5: Temperature and Softmax Entropy

Let $P_\tau(j) = \frac{e^{s_j / \tau}}{\sum_{k} e^{s_k/\tau}}$ be the softmax distribution with temperature $\tau > 0$ over scores $\{s_j\}_{j=1}^M$.

**(a)** Compute the entropy $H(P_\tau) = -\sum_j P_\tau(j) \log P_\tau(j)$. Show that $H(P_\tau)$ is a decreasing function of $1/\tau$ (i.e., increasing in $\tau$). Verify the limiting cases: $H(P_\tau) \to \log M$ as $\tau \to \infty$ (maximum entropy, uniform distribution) and $H(P_\tau) \to 0$ as $\tau \to 0$ (degenerate distribution concentrated at $\arg\max_j s_j$).

**(b)** Show that the gradient of the loss $\ell = -\log P_\tau(y^+)$ with respect to the temperature $\tau$ is:

$$\frac{\partial \ell}{\partial \tau} = \frac{1}{\tau^2}\left(s_{y^+} - \mathbb{E}_{j \sim P_\tau}[s_j]\right) \cdot (-1)$$

Interpret: when $s_{y^+} > \mathbb{E}[s_j]$ (positive item has above-average score), decreasing $\tau$ decreases the loss. When does decreasing $\tau$ increase the loss?

**(c)** Consider a retrieval evaluation metric: Recall@$K$ defined as $\mathbb{P}[\text{positive item in top-}K]$. Show that Recall@$K$ is a non-decreasing function of $s_{y^+} - \max_{j \neq y^+} s_j$ (the margin). Argue that the temperature $\tau$ does not affect which item has the highest score (ranking is invariant to positive scaling of logits), so Recall@$K$ is independent of $\tau$ for a fixed trained model. However, explain why $\tau$ still matters **during training**: it controls the sharpness of the training signal (gradient magnitude for hard vs. easy negatives).

**(d)** The **InfoNCE loss** (a generalization used in contrastive learning) is:

$$\mathcal{L}_{\text{InfoNCE}} = -\mathbb{E}\left[\log \frac{e^{s(x, y^+)/\tau}}{e^{s(x, y^+)/\tau} + \sum_{j=1}^K e^{s(x, y_j^-)/\tau}}\right]$$

Show that as $K \to \infty$ with negatives drawn uniformly, this converges to the full softmax cross-entropy loss (up to a constant). This establishes the batch softmax as a special case of InfoNCE with $K = B - 1$ and non-uniform sampling.

---

## Conceptual Questions

### Problem 6: Inner Product Decomposability and MIPS Indexing

**(a)** A general similarity model scores query $x$ and item $y$ by $f(x, y)$ where $f$ is computed by concatenating $x$ and $y$ and passing them through a feedforward network. Explain precisely why this model is incompatible with precomputed item embeddings. What computational property of the inner product $s(x,y) = \langle u(x), v(y) \rangle$ makes it indexable, and what property of general $f(x,y)$ breaks this?

**(b)** The Maximum Inner Product Search (MIPS) problem is: given a query vector $u \in \mathbb{R}^k$ and a set of item vectors $\{v_j\}_{j=1}^M$, find $\arg\max_j \langle u, v_j \rangle$. Exact MIPS is solved in $O(kM)$ time by exhaustive search. Approximate MIPS algorithms (tree-based, quantization-based, LSH-based) reduce this to sublinear time at the cost of recall. Explain the fundamental tradeoff: why can approximate MIPS guarantee exact top-1 retrieval only probabilistically, and how does the approximation quality relate to the geometry of the embedding space?

**(c)** The paper uses L2 normalization before inner product, converting MIPS into maximum cosine similarity search (MCSS). Show algebraically that for L2-normalized vectors, $\langle u, v \rangle = \cos(u, v)$ and that $\arg\max_j \langle u, v_j \rangle = \arg\min_j \|u - v_j\|_2^2$ (nearest neighbor in Euclidean distance). This equivalence is significant: nearest neighbor search has more efficient exact algorithms (e.g., $k$-d trees in low dimensions) and better-understood approximation theory than general MIPS.

**(d)** Suppose you want to go beyond inner products and use a general kernel $K(u, v)$ as the similarity score. Under what conditions on $K$ does a decomposition $K(u, v) = \langle \phi(u), \phi(v) \rangle$ exist for some feature map $\phi$? (Hint: Mercer's theorem.) What does this imply about the class of ranking models that are in principle compatible with precomputed item representations?

---

### Problem 7: When Does Sampling Bias Matter

**(a)** The sampling bias in the batch softmax arises because in-batch negatives are drawn from the training distribution $\{q_j\}$ rather than uniformly. Define a formal measure of the bias magnitude as the total variation distance between the effective negative distribution $q$ and the uniform distribution: $\text{TV}(q, \text{Uniform}) = \frac{1}{2}\sum_j |q_j - 1/M|$. For a Zipf distribution with exponent $\alpha$, derive (or estimate) how $\text{TV}(q, \text{Uniform})$ scales with $M$ and $\alpha$.

**(b)** Argue that in the limit $M \to \infty$ with $B$ fixed, almost all in-batch items are distinct (the batch contains a vanishing fraction of the corpus), and the chance of any given popular item appearing in a batch approaches its marginal frequency $q_j \cdot B$. In this limit, explain why the uncorrected batch softmax is a consistent estimator of what objective — and whether that objective is the intended full-corpus softmax.

**(c)** Consider the opposite extreme: all items are equally popular ($q_j = 1/M$ for all $j$). Show from Problem 3(c) that in this case, the logQ correction reduces to a constant shift of all logits, leaving the softmax output unchanged. Conclude that the correction is needed only when item frequency is heterogeneous. Relate the required magnitude of correction to the entropy of the item distribution $H(q) = -\sum_j q_j \log q_j$.

**(d)** The paper uses YouTube data with power-law item frequencies. Give two real-world examples of recommendation domains where the power-law exponent $\alpha$ would be expected to be (i) large (highly concentrated, few popular items dominate) and (ii) small (relatively flat distribution). For each, predict whether the sampling bias correction would yield a larger or smaller empirical gain.

---

### Problem 8: Sequential vs. Shuffled Training Under Distribution Shift

**(a)** Standard practice in supervised learning is to shuffle training data to break temporal correlations. This paper does the opposite — it trains sequentially in day order. Explain the tradeoff: what does shuffling buy in terms of gradient variance, and what does it destroy in terms of the model's ability to track distribution shift? Formalize using the bias-variance framing: under what assumptions does sequential training have lower expected loss at deployment time?

**(b)** The streaming frequency estimator (Algorithm 2) uses a moving average with decay $1 - \alpha$, which naturally upweights recent observations. Show that if training data were randomly shuffled, the exponential moving average on the global step would give the same weight to all past observations regardless of their original temporal position — the "recency" signal is destroyed. This is one concrete reason why sequential training is coupled to the streaming estimator design.

**(c)** Let $\mathcal{D}_t$ be the data distribution at time $t$ and let $\mathcal{D}_t = (1-\beta)\mathcal{D}_{t-1} + \beta \mathcal{D}^{\text{new}}$ model gradual drift at rate $\beta$ per day. Show that a model trained on a sliding window of $W$ days (equally weighted) incurs a distribution shift bias of order $O(\beta W)$ relative to the current distribution $\mathcal{D}_t$. How should $W$ be chosen to balance this drift bias against the variance of training on too few examples?

---

### Problem 9: Reward Weighting and Its Effect on the Learned Distribution

**(a)** The training objective is $\mathcal{L}(\theta) = -\frac{1}{T}\sum_i r_i \log P(y_i \mid x_i;\, \theta)$ where $r_i \in [0, 1]$. Show that this is equivalent to training on a reweighted dataset where each positive pair $(x_i, y_i)$ appears with effective count proportional to $r_i$. Describe the stationary distribution that the model converges to: what joint distribution $p^*(x, y)$ does maximizing the reward-weighted likelihood implicitly target?

**(b)** With binary rewards $r_i \in \{0, 1\}$ (click or no-click), the model learns $P(y \mid x)$ proportional to the click distribution. With continuous rewards, the model learns a distribution that upweights high-reward items. Show that in the limit of a perfect model (zero approximation error), $P(y \mid x;\, \theta^*) \propto r(x, y) \cdot P_{\text{data}}(y \mid x)$ where $r(x,y)$ is the expected reward and $P_{\text{data}}$ is the data-generating distribution. Is this the ideal recommendation distribution? Under what conditions would you want $r(x,y) \cdot P_\text{data}(y|x)$ vs. simply $P_\text{data}(y \mid x)$?

**(c)** The paper sets $r_i = 0$ for clicked videos with little watch time and $r_i = 1$ for fully watched videos. Explain in terms of your answer to (b) how this shifts the learned retrieval distribution relative to a binary-click model. What type of content would the reward-weighted model retrieve more (or less) often than the click-only model?

---

### Problem 10: Multiple Hashings vs. Larger Single Array

**(a)** In Algorithm 2, each bucket $B[h(y)]$ tracks the exponential moving average of inter-arrival times for all items that hash to the same bucket. A collision between items $a$ and $b$ (i.e., $h(a) = h(b)$) causes $B[h(a)]$ to underestimate the true $\delta_a$ (the bucket conflates the arrivals of both items). Explain in one sentence why underestimating $\delta$ leads to **overestimating** the sampling probability $\hat{p} = 1/B[h(y)]$.

**(b)** Algorithm 3 uses $m$ independent hash functions and takes $\hat{p}(y) = 1/\max_i B_i[h_i(y)]$. Explain the directional argument for why taking the maximum of $m$ estimates reduces the over-estimation bias: each $B_i[h_i(y)]$ is a biased underestimate of $\delta_y$ (due to collisions pulling the average down), so the maximum of $m$ independent underestimates is the least-biased among them.

**(c)** An alternative approach: use a single hash array of size $mH$ (same total memory as $m$ arrays of size $H$). Compare this alternative to Algorithm 3 on two dimensions: (i) expected collision probability for a corpus of $M$ items, and (ii) adaptability to distribution shift (how quickly does each approach track a change in item frequency?). Which approach do you expect to perform better, and under what conditions might the answer reverse?

**(d)** The count-min sketch (Cormode and Muthukrishnan 2005) is a data structure for frequency estimation in streams that uses exactly the multiple-hashing approach of Algorithm 3, but estimates frequency as $\min_i B_i[h_i(y)]$ rather than $\max_i B_i[h_i(y)]$. Explain why the paper takes the max instead of the min: in the count-min sketch, each bucket over-counts (due to collisions adding to the count), so the min is the best estimate. In Algorithm 3, each bucket under-estimates $\delta$ (conflated arrivals appear more frequent, making $\delta$ appear shorter), so the max of estimated $\delta$ values (equivalently the min of estimated frequencies) is the best estimate. Reconcile: show that both the count-min sketch and Algorithm 3 are applying the same directional correction, just in opposite directions.

---

## Implementation Sketches

### Problem 11: Corrected Batch Softmax Loss

Sketch a complete, numerically stable algorithm for computing the corrected batch softmax loss $\mathcal{L}_B(\theta)$ from Equation (4) of the paper, given a batch of $B$ query-item pairs and their estimated sampling probabilities.

**(a)** **Input specification.** Define the inputs: score matrix $S \in \mathbb{R}^{B \times B}$ where $S_{ij} = s(x_i, y_j)$, estimated log-probabilities $\log \hat{p} \in \mathbb{R}^B$ (one per item in the batch), and reward vector $r \in \mathbb{R}^B$.

**(b)** **Logit correction.** Describe how to form the corrected score matrix $S^c$ by broadcasting $\log \hat{p}$ correctly. Note that the correction applies to the item dimension (columns of $S$), not the query dimension (rows). Write the element-wise operation explicitly.

**(c)** **Numerically stable softmax.** The naive computation $\log \text{softmax}(z)_i = z_i - \log \sum_j e^{z_j}$ is numerically unstable for large $z$. Describe the standard log-sum-exp stabilization: subtract $\max_j z_j$ before exponentiation. Write pseudocode for `log_softmax(z)` that is numerically stable and apply it to the corrected scores.

**(d)** **Diagonal extraction and loss.** The positive pair for query $i$ is item $i$ (the diagonal of $S^c$). Extract the diagonal, apply log-softmax over each row, multiply by rewards, and sum. Write the full pseudocode for `corrected_batch_loss(S, log_p_hat, r)` and state its time complexity in terms of $B$.

**(e)** **Diagonal masking.** For each query $i$, item $i$ appears in the denominator of its own softmax (as the positive) and should not appear as a negative. Explain whether this "self-negative" issue requires special treatment in the formulation above, or whether the standard softmax already handles it correctly.

---

### Problem 12: Distributed Streaming Frequency Estimation

Sketch Algorithm 2 adapted for a distributed training setting with $W$ workers and a shared parameter server.

**(a)** **Data structure on the parameter server.** Describe the shared state: two integer arrays $A[0..H-1]$ and $B[0..H-1]$, a shared global step counter $t$, and a hash function $h$. Specify the data types (e.g., $A$ stores integer step indices, $B$ stores floating-point estimates of $\delta$).

**(b)** **Worker update protocol.** Each worker processes one batch per step. For each item $y$ in the batch, the worker must: (i) read $A[h(y)]$ and $B[h(y)]$ from the parameter server, (ii) compute the update, (iii) write back. Write pseudocode for this protocol. Note that the read and write are not atomic — a race condition can occur if two workers update the same bucket simultaneously. Argue that the effect of such races is bounded: the worst case is that one update is overwritten, losing one sample, not corrupting the estimator.

**(c)** **Inference.** At training time, after updating, the worker needs $\hat{p}(y) = 1/B[h(y)]$ to compute the corrected logit. Describe a caching strategy: after computing the update, use the locally computed $B[h(y)]$ (before writing back to the server) to get an estimate without an additional round trip. What is the staleness of this estimate?

**(d)** **Initialization.** Proposition 4.1 shows that an ideal initialization $\delta_0 = \delta/(1-\alpha)$ makes the estimator unbiased from step 1. In practice, $\delta$ is unknown at initialization. Describe two practical initialization strategies and their tradeoffs: (i) initialize $B[k] = 1$ (assumes all items appear every step — a severe overestimate); (ii) initialize $B[k] = B_{\text{expected}}$ where $B_{\text{expected}}$ is a rough estimate based on total corpus size and batch size.

---

### Problem 13: Recall at K Evaluation Against a Full Corpus

Sketch an evaluation pipeline for computing Recall@$K$ for a retrieval model against a full corpus of $M$ items, given $N$ test queries.

**(a)** **Embedding computation.** Describe how to compute query embeddings $\{u_i\}_{i=1}^N$ and item embeddings $\{v_j\}_{j=1}^M$ using the trained model. Identify the computational bottleneck (hint: $M \gg N$ for large corpora) and explain why the item embeddings are computed offline and cached.

**(b)** **Exact Recall@K.** For each test query $i$ with positive item $y_i^+$: (i) compute scores $s_{ij} = \langle u_i, v_j \rangle$ for all $j \in [M]$; (ii) find the rank of $y_i^+$ among all $M$ scores; (iii) define $\text{Hit}_i = \mathbf{1}[\text{rank}(y_i^+) \leq K]$. Recall@$K = \frac{1}{N}\sum_i \text{Hit}_i$. What is the time complexity of this exact computation in terms of $N$, $M$, and $k$ (embedding dimension)?

**(c)** **Approximate Recall@K via ANN.** For large $M$ and $N$, exact computation is infeasible. Describe how to use an approximate nearest neighbor (ANN) index: build the index on $\{v_j\}_{j=1}^M$ offline, then for each query $u_i$, retrieve the approximate top-$K$ candidates. The approximate Recall@$K$ may be lower than exact Recall@$K$ due to retrieval errors. Define the **ANN recall gap** as $\text{Recall@}K^{\text{exact}} - \text{Recall@}K^{\text{ANN}}$ and explain the two sources of this gap: (i) the ANN index misses the true top-$K$ items; (ii) the positive item itself may not be in the index (out-of-index items).

**(d)** **Multiple positives.** In YouTube, a query (user-context pair) may have multiple positive items (multiple videos the user engaged with in one session). Generalize the Recall@$K$ definition to the multi-positive setting: define $\text{Recall@}K$ as the fraction of positive items retrieved in the top-$K$, averaged over queries. Write pseudocode for computing this metric and identify when the single-positive and multi-positive definitions coincide.
