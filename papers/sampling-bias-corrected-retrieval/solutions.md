# Sampling-Bias-Corrected Retrieval: Solutions

## Table of Contents

1. [Derivation Problems](#derivation-problems)
   - [Solution 1: Gradient of the Softmax Loss and Its Contrastive Structure](#solution-1-gradient-of-the-softmax-loss-and-its-contrastive-structure)
   - [Solution 2: Bias of Uncorrected Batch Softmax](#solution-2-bias-of-uncorrected-batch-softmax)
   - [Solution 3: LogQ Correction as Importance-Weighted Estimation](#solution-3-logq-correction-as-importance-weighted-estimation)
   - [Solution 4: Streaming Estimator Bias and Variance](#solution-4-streaming-estimator-bias-and-variance)
   - [Solution 5: Temperature and Softmax Entropy](#solution-5-temperature-and-softmax-entropy)
2. [Conceptual Questions](#conceptual-questions)
   - [Solution 6: Inner Product Decomposability and MIPS Indexing](#solution-6-inner-product-decomposability-and-mips-indexing)
   - [Solution 7: When Does Sampling Bias Matter](#solution-7-when-does-sampling-bias-matter)
   - [Solution 8: Sequential vs. Shuffled Training Under Distribution Shift](#solution-8-sequential-vs-shuffled-training-under-distribution-shift)
   - [Solution 9: Reward Weighting and Its Effect on the Learned Distribution](#solution-9-reward-weighting-and-its-effect-on-the-learned-distribution)
   - [Solution 10: Multiple Hashings vs. Larger Single Array](#solution-10-multiple-hashings-vs-larger-single-array)
3. [Implementation Sketches](#implementation-sketches)
   - [Solution 11: Corrected Batch Softmax Loss](#solution-11-corrected-batch-softmax-loss)
   - [Solution 12: Distributed Streaming Frequency Estimation](#solution-12-distributed-streaming-frequency-estimation)
   - [Solution 13: Recall at K Evaluation Against a Full Corpus](#solution-13-recall-at-k-evaluation-against-a-full-corpus)

---

## Derivation Problems

### Solution 1: Gradient of the Softmax Loss and Its Contrastive Structure

**(a)** Let $\ell = -\log P(y^+ \mid x;\,\theta) = -s_{y^+} + \log \sum_{k=1}^M e^{s_k}$ where $s_k := s(x, y_k)$. Differentiate with respect to $s_j$:

$$\frac{\partial \ell}{\partial s_j} = -\mathbf{1}[y_j = y^+] + \frac{e^{s_j}}{\sum_k e^{s_k}} = p_j - \mathbf{1}[y_j = y^+]$$

where the first term arises because only the $y^+$ logit appears in the numerator.

**(b)** By the chain rule, the gradient with respect to the query embedding $u := u(x, \theta)$ is:

$$\frac{\partial \ell}{\partial u} = \sum_{j=1}^M \frac{\partial \ell}{\partial s_j} \cdot \frac{\partial s_j}{\partial u} = \sum_{j=1}^M \left(p_j - \mathbf{1}[y_j = y^+]\right) v(y_j, \theta)$$

$$= \sum_{j=1}^M p_j\, v(y_j, \theta) - v(y^+, \theta) = \mathbb{E}_{j \sim P(\cdot \mid x;\,\theta)}[v(y_j, \theta)] - v(y^+, \theta)$$

Interpretation: the gradient moves $u$ away from the model-weighted average of all item embeddings and toward the positive item embedding $v(y^+)$. Equivalently, gradient descent on $\ell$ pushes the query closer to the positive and away from the centroid of the current model distribution over items.

**(c)** When negatives $y_1^-, \ldots, y_K^-$ are drawn i.i.d. from proposal $q$ and $q = \text{Uniform}$ (so $q(y_j^-) = 1/M$ for all $j$):

$$\mathbb{E}\left[\hat{Z}(x)\right] = e^{s(x, y^+)} + \frac{1}{K}\sum_{j=1}^K \frac{\mathbb{E}[e^{s(x, y_j^-)}]}{q(y_j^-)}$$

$$= e^{s(x, y^+)} + \frac{1}{K} \cdot K \cdot M \cdot \frac{1}{M}\sum_{y} e^{s(x,y)} = \sum_{y} e^{s(x,y)} = Z(x)$$

so $\hat{Z}$ is unbiased (the $e^{s(x,y^+)}$ term and the IS-corrected negative term together reconstruct the full partition function).

When $q = P(\cdot \mid x;\,\theta)$ (self-normalized IS), the IS weight $e^{s(x,y_j^-)}/q(y_j^-)$ becomes $e^{s(x,y_j^-)}/P(y_j^- \mid x;\,\theta) = Z(x)$, a constant. So every negative contributes exactly $Z(x)$ to $\hat{Z}$, making the variance of the estimator zero — but the estimator is no longer useful for learning, since the gradient with respect to $\theta$ of a constant partition function is zero. This is a degenerate case where the proposal exactly matches the model, and all gradient information vanishes.

---

### Solution 2: Bias of Uncorrected Batch Softmax

**(a)** In a batch of size $B$, each of the $B$ positive items is drawn i.i.d. from the training distribution. For query $x_i$, item $y_j$ ($j \neq i$) appears as an in-batch negative if and only if it is the positive item for some other query $x_j$. The probability that a randomly drawn positive pair has item $y$ is $q_y$ (the item frequency). So:

$$\mathbb{P}[y_j = y \text{ for some } j \neq i] \approx (B-1)\,q_y$$

for large $M \gg B$ (the balls-in-bins approximation: each draw is approximately independent since collisions are rare). Therefore the effective distribution of in-batch negatives is $\tilde{q}(y) = (B-1)q_y / \sum_{y'} q_{y'} = q_y$, proportional to item frequency — not uniform.

**(b)** The uncorrected batch softmax loss for query $x_i$ is:

$$\ell_i^{\text{batch}} = -\log\frac{e^{s_i^+}}{\sum_{j=1}^B e^{s_{ij}}} \approx -s_i^+ + \log\sum_{j=1}^B e^{s_{ij}}$$

The expected denominator approximates $\mathbb{E}\left[\sum_{j=1}^B e^{s_{ij}}\right] = e^{s_i^+} + (B-1)\sum_y q_y e^{s(x_i, y)}$, which is proportional to $\sum_y q_y e^{s(x_i,y)}$, not $\sum_y e^{s(x_i,y)}$.

Hence the uncorrected batch softmax is an unbiased estimator of:

$$-\log\frac{e^{s_i^+}}{\sum_y q_y e^{s(x_i,y)}} = -\log P^{(q)}(y_i \mid x_i;\,\theta)$$

where $P^{(q)}$ is a softmax with partition function $\sum_y q_y e^{s(x_i,y)}$. The bias term is:

$$\mathbb{E}[\ell_i^{\text{batch}}] - \ell_i^{\text{full}} \approx \log\frac{\sum_y q_y e^{s(x_i,y)}}{\frac{1}{M}\sum_y e^{s(x_i,y)}}$$

This reweights the partition function by $q_y$, boosting the contribution of high-frequency items. At convergence, the model is incentivized to make popular items even more distinguishable (since they appear more in the denominator), at the expense of rare items.

**(c)** For Zipf distribution $q_j = C j^{-\alpha}$ with normalizing constant $C = \left(\sum_{j=1}^M j^{-\alpha}\right)^{-1}$:

$$\text{popularity ratio} = \frac{q_1}{q_{\lfloor M/2 \rfloor}} = \frac{1}{(\lfloor M/2 \rfloor)^{-\alpha}} = \left(\frac{M}{2}\right)^\alpha$$

As $\alpha$ increases, this ratio grows as $(M/2)^\alpha$, unboundedly. The bias $\log(\sum_y q_y e^s) - \log(\frac{1}{M}\sum_y e^s)$ scales with the covariance $\text{Cov}_y(q_y, e^{s(x,y)})$ under the empirical item distribution; higher $\alpha$ increases the variance of $q_y$ across items, increasing this covariance term and hence the bias.

For $\alpha = 0$: $q_j = 1/M$ for all $j$, so $\sum_y q_y e^{s_y} = \frac{1}{M}\sum_y e^{s_y}$, and the bias term is $\log 1 = 0$. The correction vanishes exactly.

---

### Solution 3: LogQ Correction as Importance-Weighted Estimation

**(a)** The full partition function is $Z(x) = \sum_{j=1}^M e^{s(x,y_j)}$. For items $y_j$ sampled into the batch with probabilities $p_j$ (not necessarily uniform), the importance-weighted estimator is:

$$\mathbb{E}\left[\hat{Z}_{\text{IS}}(x)\right] = \mathbb{E}\left[\frac{1}{B}\sum_{j \in \text{batch}} \frac{e^{s(x,y_j)}}{p_j}\right] = \frac{1}{B} \cdot B \cdot \sum_{j=1}^M p_j \cdot \frac{e^{s(x,y_j)}}{p_j} = \sum_{j=1}^M e^{s(x,y_j)} = Z(x)$$

(using linearity of expectation and the fact that each item enters the batch with probability $p_j$). Now observe:

$$\frac{e^{s(x,y_j)}}{p_j} = e^{s(x,y_j) - \log p_j} = e^{s^c(x,y_j)}$$

so the logQ correction $s^c_j = s_j - \log p_j$ implements IS in log-space. Each corrected logit $s^c_j$ has the same exponentiated value as the IS-corrected score, and the batch softmax denominator with corrected logits is an unbiased estimator of $Z(x)$ (up to a factor of $1/B$).

**(b)** The corrected batch softmax $P_B^c(y_i \mid x_i;\,\theta) = \text{softmax}(s^c_{i,\cdot})_i$ approximates $P(y_i \mid x_i;\,\theta) = e^{s_i^+}/Z(x_i)$.

The residual bias comes from $\log$ of an expectation vs. expectation of a $\log$:

$$\mathbb{E}[-\log P_B^c(y_i \mid x_i)] = \mathbb{E}\left[-s^c_i + \log\sum_{j} e^{s^c_{ij}}\right]$$

The denominator $\sum_j e^{s^c_{ij}}$ is an unbiased estimator of $Z(x_i)/B$, but because $\log$ is concave, Jensen's inequality gives:

$$\mathbb{E}\left[\log\sum_j e^{s^c_{ij}}\right] \leq \log\mathbb{E}\left[\sum_j e^{s^c_{ij}}\right] = \log(Z(x_i)/B)$$

wait — the inequality goes the other way: since $\log$ is concave, $\mathbb{E}[\log \hat{Z}] \leq \log \mathbb{E}[\hat{Z}] = \log(Z/B)$. So the expected corrected batch loss is at least $-s^c_i + \log(Z/B) = -\log P(y_i \mid x_i)$ up to the $\log B$ constant. The residual bias is $\log(Z/B) - \mathbb{E}[\log \hat{Z}] \geq 0$, driven by the variance of $\hat{Z}$: higher variance of in-batch scores means larger Jensen gap, which is reduced by increasing batch size $B$.

**(c)** If $p_j = B/M$ for all $j$ (uniform sampling into a batch of size $B$), then $\log p_j = \log(B/M)$ is a constant. The corrected logit is:

$$s^c_j = s_j - \log(B/M) = s_j + \log(M/B)$$

Subtracting a constant from all logits shifts the softmax numerator and denominator equally:

$$\text{softmax}(s^c)_i = \frac{e^{s_i + c}}{\sum_j e^{s_j + c}} = \frac{e^{s_i}}{\sum_j e^{s_j}} = \text{softmax}(s)_i$$

so the output is unchanged. The correction is trivially constant and contributes nothing. Non-triviality requires $\text{Var}_j(\log p_j) > 0$, i.e., the item sampling probabilities must be heterogeneous. When $\text{Var}_j(\log p_j) = 0$, correction = no-op; the magnitude of the correction scales as $\text{std}_j(\log p_j)$, which is the entropy-like spread of the log-probability distribution.

**(d)** With $\hat{p}_j = p_j + \epsilon_j$ for small perturbation $\epsilon_j$:

$$\hat{s}^c_j = s_j - \log(p_j + \epsilon_j) \approx s_j - \log p_j - \frac{\epsilon_j}{p_j}$$

The noise term $-\epsilon_j/p_j$ has variance $\text{Var}(\epsilon_j)/p_j^2$. For **rare items** with small $p_j$, the noise is amplified by $1/p_j^2$: a small absolute error in estimating the frequency becomes a large error in the corrected logit. For **popular items** with large $p_j$, the same absolute estimation error has a smaller relative impact.

Consequence: the streaming estimator must be most accurate for rare items — which are also the hardest to estimate precisely (few observations, high inter-arrival variance). This is the fundamental tension in the method: the correction matters most where it is hardest to estimate.

---

### Solution 4: Streaming Estimator Bias and Variance

**(a)** Define $a = 1-\alpha$. The recurrence is $\delta_t = a\,\delta_{t-1} + \alpha\,\Delta_t$.

**Base case** ($t = 1$): $\delta_1 = a\,\delta_0 + \alpha\,\Delta_1$, which matches the formula with the sum containing only the $k=1$ term: $a^0\,\Delta_1 = \Delta_1$.

**Inductive step**: Assume $\delta_{t-1} = a^{t-1}\delta_0 + \alpha\sum_{k=1}^{t-1} a^{t-1-k}\Delta_k$. Then:

$$\delta_t = a\cdot\delta_{t-1} + \alpha\,\Delta_t = a^t\delta_0 + \alpha\sum_{k=1}^{t-1}a^{t-k}\Delta_k + \alpha\,\Delta_t = a^t\delta_0 + \alpha\sum_{k=1}^t a^{t-k}\Delta_k$$

This completes the induction. Taking expectations (using $\mathbb{E}[\Delta_k] = \delta$ and independence):

$$\mathbb{E}[\delta_t] = a^t\delta_0 + \alpha\,\delta\sum_{k=1}^t a^{t-k} = a^t\delta_0 + \alpha\,\delta\cdot\frac{1-a^t}{1-a} = a^t\delta_0 + (1-a^t)\delta$$

so the bias is:

$$\mathbb{E}[\delta_t] - \delta = a^t(\delta_0 - \delta) = (1-\alpha)^t(\delta_0 - \delta)$$

This matches equation (7) in the paper (the paper writes the inter-arrival time in place of $\delta$ and uses the same geometric decay). The bias decays geometrically to zero regardless of $\delta_0$, but the rate of decay depends on $\alpha$: larger $\alpha$ means faster convergence.

**(b)** From the representation $\delta_t = a^t\delta_0 + \alpha\sum_{k=1}^t a^{t-k}\Delta_k$, the $\delta_0$ term is deterministic, so:

$$\text{Var}(\delta_t) = \alpha^2 \sum_{k=1}^t a^{2(t-k)} \text{Var}(\Delta_k) = \alpha^2\sigma^2\sum_{k=1}^t a^{2(t-k)}$$

This is a geometric series with ratio $a^2 = (1-\alpha)^2$:

$$\text{Var}(\delta_t) = \alpha^2\sigma^2 \cdot \frac{1 - a^{2t}}{1 - a^2} = \frac{\alpha^2\sigma^2}{(1-a)(1+a)}\left(1-a^{2t}\right) = \frac{\alpha\sigma^2}{2-\alpha}\left(1-(1-\alpha)^{2t}\right)$$

As $t \to \infty$: $\sigma_\infty^2 = \frac{\alpha\sigma^2}{2-\alpha}$. For $\alpha \ll 1$: $\sigma_\infty^2 \approx \frac{\alpha\sigma^2}{2}$. This matches the paper's equation (8) (stated as an upper bound $\alpha\sigma^2/2$, which is the small-$\alpha$ approximation).

**(c)** At steady state ($t \to \infty$), bias $= 0$, so $\text{MSE}_\infty = \sigma_\infty^2 = \frac{\alpha\sigma^2}{2-\alpha}$. This is an increasing function of $\alpha$: lower $\alpha$ always yields lower steady-state variance. The bias-variance tradeoff only manifests at finite $t$.

To minimize $\text{MSE}(\delta_t)$ at finite $t$:

$$\text{MSE}(\delta_t) = (1-\alpha)^{2t}(\delta_0-\delta)^2 + \frac{\alpha\sigma^2}{2-\alpha}\left(1-(1-\alpha)^{2t}\right)$$

Differentiating with respect to $\alpha$ and setting to zero gives an implicit equation for $\alpha^*$. In the limit $\sigma^2 \to 0$: the variance term vanishes, so $\alpha^* \to 0$ (any smaller $\alpha$ reduces bias faster... actually bias decays as $(1-\alpha)^t$, which decays **faster** for larger $\alpha$). More carefully: for $\sigma^2 \to 0$, $\text{MSE} \approx (1-\alpha)^{2t}(\delta_0-\delta)^2$, minimized at $\alpha^* = 1$ (set to the true mean in one step). For $\sigma^2 \to \infty$: variance dominates, minimized at $\alpha^* \to 0$ (reduce variance by averaging). The tradeoff reverses: with high noise, slow adaptation beats fast convergence.

**(d)** At step $t_0$ the true mean shifts from $\delta$ to $\delta' = \delta + \Delta\delta$. For $t > t_0$, starting with initial condition $\delta_{t_0}$:

$$\mathbb{E}[\delta_t] = (1-\alpha)^{t-t_0}\mathbb{E}[\delta_{t_0}] + \left(1-(1-\alpha)^{t-t_0}\right)\delta'$$

The error after the shift is:

$$|\mathbb{E}[\delta_t] - \delta'| = (1-\alpha)^{t-t_0}|\delta_{t_0} - \delta'|$$

The half-life $\tau_{1/2}$ satisfies $(1-\alpha)^{\tau_{1/2}} = 1/2$, giving:

$$\tau_{1/2} = \frac{\log 2}{|\log(1-\alpha)|} \approx \frac{\log 2}{\alpha} \quad \text{for small } \alpha$$

For $\alpha = 0.01$: $\tau_{1/2} \approx 69$ steps — nearly 70 gradient steps before the estimator catches up halfway to the new frequency. Smaller $\alpha$ trades faster variance reduction for slower adaptation to non-stationarity. In YouTube's setting where video popularity can shift overnight, this lag must be bounded by setting $\alpha$ large enough to track changes within a single day's training steps.

---

### Solution 5: Temperature and Softmax Entropy

**(a)** With $P_\tau(j) = e^{s_j/\tau}/Z_\tau$ where $Z_\tau = \sum_k e^{s_k/\tau}$:

$$H(P_\tau) = -\sum_j P_\tau(j)\log P_\tau(j) = -\sum_j P_\tau(j)\left(\frac{s_j}{\tau} - \log Z_\tau\right) = -\frac{\mathbb{E}[s]}{\tau} + \log Z_\tau$$

To show $H$ is increasing in $\tau$: note that $H(P_\tau) = \log Z_\tau - \frac{1}{\tau}\mathbb{E}_{P_\tau}[s]$. The function $\log Z_\tau$ is convex in $1/\tau$ (as a log-sum-exp), and $H$ can be written as the Legendre conjugate structure. More directly: as $\tau \to \infty$, $s_j/\tau \to 0$ for all $j$, so $P_\tau \to \text{Uniform}(M)$ and $H \to \log M$. As $\tau \to 0$, $P_\tau(j) \to \mathbf{1}[j = \arg\max_k s_k]$ (argmax distribution), giving $H \to 0$. Since $H$ is continuous and has these boundary values, and $P_\tau$ interpolates smoothly, $H$ is monotone increasing in $\tau$.

**(b)** Write $\ell = -s_{y^+}/\tau + \log Z_\tau$. Then:

$$\frac{\partial \ell}{\partial \tau} = \frac{s_{y^+}}{\tau^2} + \frac{\partial \log Z_\tau}{\partial \tau}$$

Now $\frac{\partial \log Z_\tau}{\partial \tau} = \frac{1}{Z_\tau}\sum_j e^{s_j/\tau}\cdot\left(-\frac{s_j}{\tau^2}\right) = -\frac{\mathbb{E}_{P_\tau}[s]}{\tau^2}$.

Therefore:

$$\frac{\partial \ell}{\partial \tau} = \frac{1}{\tau^2}\left(s_{y^+} - \mathbb{E}_{P_\tau}[s]\right)$$

When $s_{y^+} > \mathbb{E}_{P_\tau}[s]$ (positive item above average), $\partial\ell/\partial\tau > 0$: the loss decreases as $\tau$ decreases (sharper distribution concentrates probability on high-scoring items). The loss increases with decreasing $\tau$ when the positive item has below-average score — the model is confused, and sharpening the distribution concentrates probability away from the positive.

**(c)** Recall@$K$ depends only on the ranking of $y^+$ among all items, which is determined by $\{s_j\}$. Multiplying all scores by $1/\tau$ is a positive scaling that preserves ranks. Therefore Recall@$K$ is invariant to $\tau$ for a fixed trained model.

However, during training, $\tau$ controls how sharply the gradient distinguishes hard vs. easy negatives. At low $\tau$, the softmax probabilities $\{p_j\}$ are concentrated near the top-scoring items: the gradient $p_j - \mathbf{1}[j=y^+]$ is large (in magnitude) only for the few high-scoring negatives, making training focus on hard negatives. At high $\tau$, the gradient is spread across all negatives equally, making every negative contribute a similar learning signal regardless of score. This is why temperature is a crucial training hyperparameter even though it is irrelevant for evaluation.

**(d)** With $K$ uniform negatives $y_j^- \sim \text{Uniform}(M)$, the InfoNCE loss is:

$$\mathcal{L}_{\text{InfoNCE}} = -\mathbb{E}\left[\log\frac{e^{s^+/\tau}}{e^{s^+/\tau} + \sum_{j=1}^K e^{s_j^-/\tau}}\right]$$

For large $K$, the denominator $\frac{1}{K}\sum_{j=1}^K e^{s_j^-/\tau} \to \frac{M}{K}\cdot\frac{1}{M}\sum_j e^{s_j/\tau} = \frac{Z_\tau}{K}$ by the law of large numbers. So:

$$\mathcal{L}_{\text{InfoNCE}} \to -\mathbb{E}\left[\log\frac{e^{s^+/\tau}}{e^{s^+/\tau} + K\cdot\frac{Z_\tau}{K}}\right] = -\mathbb{E}\left[\log\frac{e^{s^+/\tau}}{e^{s^+/\tau} + Z_\tau}\right] \approx -\mathbb{E}[\log P_\tau(y^+\mid x)]$$

up to the $e^{s^+/\tau}$ correction in the denominator (which is negligible for large $M$ since the positive contributes a single term to the sum of $M$ terms). This recovers the full softmax cross-entropy. The batch softmax is InfoNCE with $K = B-1$ and $q =$ training distribution (non-uniform), making it a biased special case.

---

## Conceptual Questions

### Solution 6: Inner Product Decomposability and MIPS Indexing

**(a)** A general interaction model $f(x, y)$ requires both $x$ and $y$ to be available at scoring time, making it impossible to precompute item representations. Specifically: the top-$K$ retrieval step must score all $M$ items against a given query at inference time. If $f$ decomposes as $\langle u(x), v(y)\rangle$, then $v(y)$ can be computed offline for all $y \in [M]$ and stored; at inference time, only $u(x)$ is computed, and MIPS retrieves the top items using the precomputed item index. If $f$ is non-decomposable (e.g., takes concatenation of $x$ and $y$ as input), then every $(x, y)$ pair must be evaluated fresh at query time — cost $O(M)$ per query, prohibitive for $M \sim 10^6$.

The critical property is **bilinearity**: $\langle u(x), v(y)\rangle$ is linear in $v(y)$ for fixed $u(x)$, which enables precomputation. General functions $f$ lack this linearity.

**(b)** Exact MIPS requires $O(kM)$ work (inner product with each item). ANN guarantees are probabilistic because the index structure (e.g., graph-based like HNSW, or quantization-based like IVF-PQ) trades completeness for speed: it searches only a fraction of the item space. Whether the true top-$K$ items lie in the searched region depends on the geometry — specifically, whether the query lies in a "cluster" with the top items or whether the top items are scattered across the embedding space. High-dimensional geometry is adversarial to many ANN methods (curse of dimensionality), so guarantees are typically in expectation over queries or with probability $1-\delta$ over randomness in the index structure.

**(c)** For L2-normalized vectors $\|u\|_2 = \|v\|_2 = 1$:

$$\langle u, v\rangle = \cos\angle(u,v)$$

by definition of cosine similarity. Furthermore:

$$\|u - v\|_2^2 = \|u\|^2 - 2\langle u, v\rangle + \|v\|^2 = 2 - 2\langle u, v\rangle$$

Since $2$ is a constant, $\arg\max_j\langle u, v_j\rangle = \arg\min_j\|u - v_j\|_2^2$: maximizing cosine similarity is exactly nearest neighbor search. This equivalence is significant because Euclidean NN has a richer theoretical and practical toolkit (e.g., exact algorithms in low dimensions via $k$-d trees, well-understood random projection bounds for LSH).

**(d)** By Mercer's theorem, a symmetric positive semi-definite kernel $K: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ admits a feature map $\phi: \mathcal{X} \to \mathcal{H}$ (possibly infinite-dimensional Hilbert space) such that $K(u,v) = \langle \phi(u), \phi(v)\rangle_\mathcal{H}$. The condition is that $K$ must be PSD: $\sum_{i,j} c_i c_j K(x_i, x_j) \geq 0$ for all finite sets and real coefficients. Any PSD kernel yields a feature-map decomposition and is thus compatible with precomputed item representations (in the feature space). Non-PSD kernels (e.g., cosine distance on non-normalized vectors with a sign flip) are incompatible with the two-tower retrieval architecture unless approximated.

---

### Solution 7: When Does Sampling Bias Matter

**(a)** For $q_j = C j^{-\alpha}$ with $C = \left(\sum_{j=1}^M j^{-\alpha}\right)^{-1}$ and Uniform distribution $u_j = 1/M$:

$$\text{TV}(q, \text{Uniform}) = \frac{1}{2}\sum_j |q_j - 1/M|$$

For large $M$ and $\alpha > 1$ (summable), $C \to \zeta(\alpha)^{-1}$ (Riemann zeta function), and the sum is dominated by $j=1$: $|q_1 - 1/M| \approx C$ since $1/M \to 0$. So $\text{TV} \to \frac{1}{2}\sum_j q_j = \frac{1}{2}$ (since $q$ is a probability measure and $1/M$ is negligible for small $j$). For $0 < \alpha < 1$ (non-summable), $C \to 0$ and the distribution flattens, so $\text{TV} \to 0$. The phase transition at $\alpha = 1$ separates the concentrated regime (substantial bias) from the flat regime (negligible bias).

**(b)** As $M \to \infty$ with $B$ fixed: the fraction of the corpus in any batch is $B/M \to 0$. Items in the batch are essentially distinct (collision probability $O(B^2/M) \to 0$). Item $j$ appears in a batch with probability $\approx B q_j$. The uncorrected batch softmax therefore converges to a consistent estimator of:

$$-\log\frac{e^{s^+}}{\sum_j q_j e^{s_j}}$$

not the intended $-\log\frac{e^{s^+}}{\frac{1}{M}\sum_j e^{s_j}}$. So the uncorrected estimator is consistent — but for the wrong objective: it converges to the maximum likelihood estimator under the **frequency-reweighted** item distribution $q$, not the uniform distribution.

**(c)** Covered in Solution 3(c): uniform $q_j = 1/M$ implies $\log p_j = \log(B/M)$ is a constant, giving a constant logit shift that cancels in the softmax. The magnitude of the required correction is $\log p_j - \log(1/M) = \log(M p_j)$, which is zero for all $j$ when $q$ is uniform.

The entropy of $q$ is maximized at $H(q) = \log M$ for uniform $q$ and minimized at $H(q) = 0$ for a point mass. The correction magnitude scales with $\max_j |\log p_j - \log(B/M)| = \max_j |\log(M q_j)|$, which is large when $q$ is concentrated ($H(q) \ll \log M$). So the correction is most important in the low-entropy regime — which is precisely the power-law setting of YouTube recommendations.

**(d)** Two examples:

**(i) High $\alpha$ (concentrated): News article recommendation.** A handful of breaking news stories dominate all clicks during a 24-hour period. The item distribution is extremely concentrated: a few items have $q_j \approx 0.01$–$0.1$, while millions of older articles have $q_j \approx 10^{-8}$. The sampling bias correction is large and critical: without it, the model conflates "this item is frequently clicked" with "this item is relevant to this query," and will embed popular items near all queries.

**(ii) Low $\alpha$ (flat): Music streaming for niche genres.** In a genre-specific playlist, listening counts across tracks follow a near-uniform distribution — all songs are approximately equally popular. The sampling bias correction has negligible magnitude and may not be worth the added complexity of streaming frequency estimation.

---

### Solution 8: Sequential vs. Shuffled Training Under Distribution Shift

**(a)** **What shuffling buys**: i.i.d. mini-batches have lower gradient variance. Shuffling ensures each batch is a representative sample of the training set, minimizing the variance of the stochastic gradient estimator. Under shuffled training, the model at convergence minimizes the time-averaged loss $\mathbb{E}_{(x,y)\sim \bar{\mathcal{D}}}[\ell(x,y;\theta)]$ where $\bar{\mathcal{D}}$ is the mixture of all daily distributions.

**What shuffling destroys**: the temporal ordering that allows the model to reflect the current distribution $\mathcal{D}_T$ at deployment time $T$. Under sequential training, recent data has higher influence on model parameters, because the optimizer's state (weights) is last updated on the most recent data.

**Formal condition**: sequential training has lower expected loss at deployment if the distribution shift is substantial relative to the training variance. Let $\mathcal{D}_T$ be the current distribution and $\bar{\mathcal{D}}$ the mixture. If $\|\mathcal{D}_T - \bar{\mathcal{D}}\|_{\text{TV}} > O(1/\sqrt{T_{\text{dataset}}})$ (the shift is larger than the statistical noise), sequential training wins.

**(b)** Under shuffled training, the order of examples reaching the optimizer is i.i.d. from $\bar{\mathcal{D}}$. The exponential moving average on global step $t$ assigns weight $(1-\alpha)^{t-k}$ to the observation at step $k$ — but since the observations are permuted, step $k$ may correspond to data from any day in the training window. The temporal recency information is uniformly destroyed: the EMA now simply tracks the stationary mean of the shuffled stream, not the changing mean of the sequential stream. The estimator loses its ability to upweight recent arrivals.

**(c)** Under gradual drift $\mathcal{D}_t = (1-\beta)\mathcal{D}_{t-1} + \beta\mathcal{D}^{\text{new}}$, after $W$ days the distribution has drifted to $\mathcal{D}_T$ from $\mathcal{D}_{T-W}$. The $W$-day sliding window trains equally on all $W$ distributions; the effective training distribution is $\hat{\mathcal{D}} = \frac{1}{W}\sum_{t=T-W}^T \mathcal{D}_t$.

The drift bias is:

$$\|\mathcal{D}_T - \hat{\mathcal{D}}\|_{\text{TV}} = \left\|\mathcal{D}_T - \frac{1}{W}\sum_{t=T-W}^T \mathcal{D}_t\right\|_{\text{TV}} \leq \frac{1}{W}\sum_{t=T-W}^T \|\mathcal{D}_T - \mathcal{D}_t\|_{\text{TV}} = O(\beta W)$$

since $\|\mathcal{D}_T - \mathcal{D}_t\|_{\text{TV}} \leq \beta(T-t) \leq \beta W$ for all $t$ in the window.

To balance: the variance of learning on $W$ days of data scales as $O(1/\sqrt{W n_{\text{day}}})$ where $n_{\text{day}}$ is examples per day. Setting drift bias $\sim$ variance: $\beta W \sim 1/\sqrt{Wn}$, giving $W^* \sim (\beta\sqrt{n})^{-2/3}$. Shorter window for fast drift ($\beta$ large) or abundant data ($n$ large).

---

### Solution 9: Reward Weighting and Its Effect on the Learned Distribution

**(a)** The reward-weighted loss $\mathcal{L}(\theta) = -\frac{1}{T}\sum_i r_i \log P(y_i \mid x_i;\,\theta)$ is the cross-entropy of $P(\cdot \mid \cdot;\,\theta)$ against the empirical distribution where example $i$ has weight $r_i$. Equivalently, it is the cross-entropy against the weighted empirical measure $\tilde{\mathbb{P}} = \frac{\sum_i r_i \delta_{(x_i, y_i)}}{\sum_i r_i}$.

In the limit of a perfect model and infinite data, the maximizer minimizes $\mathbb{E}_{(x,y) \sim \tilde{P}}[-\log P(y \mid x;\,\theta)]$, which is achieved at $P^*(y \mid x) \propto \tilde{P}(y \mid x)$. With $r_i = r(x_i, y_i)$ and the data drawn from $P_{\text{data}}$:

$$P^*(y \mid x) \propto r(x,y) \cdot P_{\text{data}}(y \mid x)$$

This is the reward-reweighted conditional: items with higher reward are upweighted proportional to their reward.

**(b)** Whether $r(x,y) P_{\text{data}}(y\mid x)$ is the ideal distribution depends on the deployment goal. If the goal is to serve the item a user would most enjoy (independent of how it was presented historically), then $P_{\text{data}}(y \mid x)$ already encodes the user-item affinity. Multiplying by $r(x,y)$ further upweights items that generate high reward conditioned on being shown — desirable if reward is a good proxy for long-term user satisfaction, undesirable if reward is noisy or if popular items have inflated reward due to exposure bias.

For retrieval specifically (first-stage of ranking), one wants to retrieve items that have potential for high reward; the reward-weighted model achieves this by concentrating retrieval mass on high-utility items at the cost of recall on medium-utility items.

**(c)** With $r_i = 0$ for low-watchtime clicks and $r_i = 1$ for full-watchtime watches, the model targets $P^*(y \mid x) \propto P_{\text{data}}^{\text{full watch}}(y \mid x)$: the distribution over videos that users watch completely conditioned on context $x$.

Compared to a binary click model targeting $P_{\text{data}}^{\text{click}}(y \mid x)$, the watch-time model retrieves:

- **More**: long-form informational content (tutorials, documentaries) that users tend to finish once they click
- **Less**: clickbait titles with low completion rate; content with high CTR but high abandon rate; viral short clips that get clicked but skimped

This is intentional: the retrieval distribution shifts toward content with genuine engagement quality, reducing optimizing for superficial click patterns.

---

### Solution 10: Multiple Hashings vs. Larger Single Array

**(a)** If items $a$ and $b$ collide to the same bucket $h(a) = h(b)$, then $B[h(a)]$ tracks the moving average of inter-arrival times across both $a$'s and $b$'s arrivals. Since arrivals of $b$ occur more frequently (both appear in the same bucket), the inter-arrival time $\delta_{\text{bucket}}$ measured is shorter than the true $\delta_a$. A shorter estimated $\delta$ gives $\hat{p} = 1/\hat{\delta}$ larger than the true $p_a = 1/\delta_a$: underestimating the inter-arrival time causes overestimating the sampling probability.

**(b)** Each hash array $B_i$ provides an independent estimate of $\delta_y$ that is biased downward (too-short inter-arrival time due to collisions from other items). Among $m$ independent downward-biased estimates, the maximum is the closest to the truth (the one estimate where $y$ happened to collide with few other items). Taking $\max_i B_i[h_i(y)]$ selects the least-biased (largest) inter-arrival time estimate, reducing the net overestimation of $\hat{p} = 1/\max_i B_i[h_i(y)]$.

**(c)** **Single array of size $mH$**: expected collision probability for $M$ items is $\approx M/mH$ (by birthday paradox, roughly $M$ items into $mH$ buckets). **Multiple arrays with $m$ hashings on arrays of size $H$ each**: collision probability in each array is $\approx M/H$; but $B_i[h_i(y)]$ is the max over $m$ independent arrays, so the effective reduction in bias is $O(m)$.

**Comparison**:
- **(i) Collision probability**: single array has $M/(mH)$ expected collision rate per bucket; multiple arrays have $M/H$ per bucket but $m$ independent views. For the collision to corrupt the estimate, it must occur in the array that gives the max — this happens with probability $(1-1/H)^{m-1} \approx e^{-(m-1)/H}$ (other arrays give even worse estimates). The max-over-$m$ strategy effectively reduces bias more aggressively than the single large array for skewed distributions.
- **(ii) Adaptability to shift**: both approaches have the same total number of buckets $mH$ and use the same $\alpha$, so the per-bucket EMA decay is identical. However, with multiple arrays, each update touches $m$ buckets simultaneously, giving $m$ times more opportunities to update. The single large array updates exactly one bucket per item, so adaptation speed is equivalent per item occurrence.

**Conclusion**: multiple hashings typically outperforms a single large array for heavily skewed distributions (YouTube), because the max-over-$m$ strategy exploits independence to cancel collision bias in a way that simply widening one array cannot. The single array would need $H \gg M$ (near-zero collisions) to match the bias reduction, which is storage-prohibitive.

**(d)** **Count-min sketch**: each counter $C_i[h_i(y)]$ accumulates collision arrivals, causing **over-counting**. The true frequency of $y$ is $\leq C_i[h_i(y)]$ for every $i$, so the min over $i$ is the best (least-biased) upper estimate.

**Algorithm 3**: each $B_i[h_i(y)]$ tracks inter-arrival times and is biased downward (collisions make $\hat{\delta}$ shorter, hence $\hat{p} = 1/\hat{\delta}$ larger). The true $\delta_y$ is $\geq B_i[h_i(y)]$ for every $i$, so the max over $i$ is the best (least-biased) estimate of $\delta_y$.

**Reconciliation**: both methods apply a min/max correction to remove the directional bias from hash collisions. The count-min sketch's frequencies and Algorithm 3's inter-arrival times are related by $p \propto 1/\delta$. Taking $\min$ of frequencies $\equiv$ taking $\max$ of inter-arrival times: $\min_i (1/B_i) = 1/\max_i B_i$. Both are the same operation under the frequency-to-interval transformation.

---

## Implementation Sketches

### Solution 11: Corrected Batch Softmax Loss

**(a)** **Inputs**:
- Score matrix $S \in \mathbb{R}^{B \times B}$: $S_{ij} = s(x_i, y_j) = \langle u(x_i), v(y_j)\rangle$
- Log-probability vector $\hat{l} \in \mathbb{R}^B$: $\hat{l}_j = \log\hat{p}_j$ (estimated log-sampling-probability of item $j$)
- Reward vector $r \in \mathbb{R}^B$: $r_i \in [0,1]$

**(b)** **Logit correction**: subtract $\hat{l}$ from each row of $S$ along the item (column) dimension:

```
S_corrected[i, j] = S[i, j] - log_p_hat[j]   # broadcast over rows
```

The correction is applied column-wise (each item $j$'s correction $\log\hat{p}_j$ is subtracted from all queries' scores for that item).

**(c)** **Numerically stable log-softmax** per row:

```
function log_softmax(z):
    m = max(z)
    return z - m - log(sum(exp(z - m)))   # log-sum-exp trick
```

The subtraction of `m` prevents overflow in `exp(z)` without changing the output (since `m` cancels in the numerator and the `log(sum(exp(z-m)))` term).

**(d)** **Full corrected batch loss**:

```
function corrected_batch_loss(S, log_p_hat, r):
    B = len(r)
    # Step 1: apply logQ correction
    S_c = S - log_p_hat[None, :]       # shape (B, B), broadcast over rows
    # Step 2: log-softmax over items (columns) for each query row
    lse = log_softmax_per_row(S_c)     # shape (B, B)
    # Step 3: extract diagonal (positive pair for each query)
    log_probs = lse[range(B), range(B)]  # shape (B,)
    # Step 4: reward-weighted sum
    loss = -sum(r * log_probs) / sum(r + 1e-8)
    return loss
```

Time complexity: $O(B^2)$ for the score matrix and correction, $O(B)$ for the log-sum-exp (amortized over $B$ rows). Dominant cost is the $B \times B$ matrix multiplication $U V^\top$ to form $S$, which is $O(Bk)$ per row or $O(B^2 k)$ total.

**(e)** **Self-negative issue**: the diagonal entry $S^c_{ii}$ represents item $y_i$ scored as a negative for query $x_i$ — but $y_i$ is actually the positive. In the standard formulation above, $S^c_{ii}$ appears in both the numerator (the positive logit) and the denominator of the softmax. This is correct: the full softmax $P(y^+ \mid x) = e^{s^+}/\sum_{j=1}^M e^{s_j}$ includes $e^{s^+}$ in the denominator. If we wanted to exclude the positive from the denominator (modeling the probability over negatives only), we would need to mask the diagonal. The paper includes the positive in the denominator (matching the full softmax semantics), so no special masking is required.

---

### Solution 12: Distributed Streaming Frequency Estimation

**(a)** **Shared state on parameter server**:
- `A[0..H-1]`: integer array; `A[k]` stores the global step index when bucket $k$ was last updated
- `B[0..H-1]`: float array; `B[k]` stores the current EMA estimate of the inter-arrival time $\hat{\delta}$ for items hashing to bucket $k$
- `t`: global step counter (atomic integer, incremented per batch across all workers)
- `h`: shared hash function (e.g., MurmurHash mod $H$)

**(b)** **Worker update protocol**:

```
for each item y in batch:
    k = h(y)
    # Read from parameter server
    last_step = A[k]
    delta_hat = B[k]
    current_step = t  # read global step
    # Compute new inter-arrival time observation
    Delta_new = current_step - last_step
    # EMA update
    delta_hat_new = (1 - alpha) * delta_hat + alpha * Delta_new
    # Write back
    A[k] = current_step
    B[k] = delta_hat_new
```

**Race condition analysis**: if two workers simultaneously read `A[k]` and `B[k]` for the same bucket and then both write, one write will be overwritten. The effect is that one EMA update is lost — the estimator behaves as if that arrival was not observed. This is a bounded degradation: the estimator still converges since the step index update and EMA update are not safety-critical; missing one update is equivalent to a slightly higher effective $\alpha$ for that step.

**(c)** **Caching strategy**: after computing `delta_hat_new`, use it immediately (before the write-back round trip) to compute the corrected logit:

```
log_p_hat = -log(delta_hat_new)   # use locally computed, not yet written
```

The staleness of this estimate is zero within the current step (it uses the just-computed value). Across subsequent steps, the worker caches `log_p_hat` for items seen in previous batches; staleness is at most one training step (the time between consecutive batches for this worker). For high-throughput workers processing thousands of steps per second, staleness is negligible relative to the EMA decay timescale $1/\alpha$.

**(d)** **Initialization strategies**:

**(i) Initialize $B[k] = 1$ (aggressive overestimate)**: every item is assumed to arrive every step. This overestimates $\delta$ by a factor of $M/B$ (if items have mean arrival time $M/B$ steps). The resulting $\hat{p} = 1/\hat{\delta}$ is initially underestimated by $M/B$, giving logQ corrections that are too large in magnitude. The estimator converges to the true $\delta$ in $O(1/\alpha)$ steps, so this bias decays quickly. It is simple but causes artifacts in early training where the logQ correction over-corrects for rare items.

**(ii) Initialize $B[k] = M/(B \cdot H)$ (corpus-size-aware estimate)**: with a corpus of $M$ items and $H$ buckets in a batch of size $B$, the expected inter-arrival time for a bucket is roughly $M/(B \cdot H)$ (assuming uniform hash distribution). This provides a better-calibrated starting point. The estimator still requires $O(1/\alpha)$ steps to converge to item-specific values, but the initial correction is closer to the population mean, reducing early-training instability.

---

### Solution 13: Recall at K Evaluation Against a Full Corpus

**(a)** **Embedding computation**:
- Query embeddings: compute $u_i = u(x_i;\,\theta^*)$ for each test query $x_i$ ($i = 1,\ldots,N$). These require an online forward pass at evaluation time.
- Item embeddings: compute $v_j = v(y_j;\,\theta^*)$ for each corpus item $y_j$ ($j = 1,\ldots,M$). These are computed offline and cached as a matrix $V \in \mathbb{R}^{M \times k}$.

**Computational bottleneck**: $M \gg N$ (e.g., YouTube has $M \sim 10^6$ videos, $N \sim 10^4$ queries per eval batch). Computing item embeddings takes $O(Mk)$ time (model forward pass for each item, embarrassingly parallel). This is done once and cached; the cache is invalidated only when model weights change.

**(b)** **Exact Recall@K**:

```
for i in 1..N:
    scores = [dot(u[i], v[j]) for j in 1..M]   # shape (M,)
    rank_of_positive = rank(scores[y_i_plus])   # position in sorted order
    Hit[i] = (rank_of_positive <= K)
Recall_at_K = mean(Hit)
```

Time complexity: for each query, $O(kM)$ to compute all scores + $O(M\log M)$ or $O(M)$ (using partial sort) to find rank. Total: $O(N(kM + M)) = O(NkM)$.

For $N = 10^4$, $k = 256$, $M = 10^6$: $\approx 2.56 \times 10^{12}$ FLOPs — infeasible without batching or approximation. In practice: batch the score computation as $U V^\top$ (matrix multiply, $O(NkM)$) and use BLAS-optimized GEMM.

**(c)** **ANN-based Recall@K**:

```
# Offline: build ANN index on {v[j]}
index = ANN_Index(V)   # e.g., HNSW or IVF-PQ

# Online: for each query
for i in 1..N:
    approx_top_K = index.query(u[i], K)   # sublinear time
    Hit[i] = (y_i_plus in approx_top_K)
Approx_Recall_at_K = mean(Hit)
```

**ANN recall gap** $= \text{Recall@}K^{\text{exact}} - \text{Recall@}K^{\text{ANN}}$ has two sources:

- **(i) Index retrieval error**: the ANN index returns approximate top-$K$; the true positive may have a high exact score but not be returned because the index search terminates before reaching its region of the embedding space. This is the primary source and is controlled by the index's `ef_search` or `nprobe` parameter.
- **(ii) Out-of-index items**: if the corpus grows after the index is built, new items are not indexed. For YouTube where new videos are continuously uploaded, the index must be updated periodically; videos uploaded after the last index build cannot be retrieved.

**(d)** **Multi-positive Recall@K**: let $\mathcal{P}_i = \{y \in \mathcal{Y} : (x_i, y) \text{ is a positive pair}\}$ be the set of positive items for query $i$.

```
for i in 1..N:
    scores = [dot(u[i], v[j]) for j in 1..M]
    top_K_items = argsort_descending(scores)[:K]
    hits = |top_K_items ∩ P_i| / |P_i|
MultiRecall_at_K = mean(hits over i)
```

This computes the fraction of each query's positive set that is recovered in the top-$K$. When $|\mathcal{P}_i| = 1$ for all $i$, the multi-positive metric reduces to the standard Recall@$K$ (0 or 1 per query, averaged). The two definitions coincide exactly in the single-positive case. In the multi-positive case, the metric is more informative: a model that retrieves 5 out of 10 positives scores 0.5 rather than 0 (which it would under any single-positive metric).
