# Sampling-Bias-Corrected Retrieval: Solutions

## Table of Contents

- [[#Mathematical Development|Mathematical Development]]
  - [[#Problem 1 Gradient of the Full Softmax Loss|Problem 1: Gradient of the Full Softmax Loss]]
  - [[#Problem 2 Bias of the Uncorrected Batch Softmax|Problem 2: Bias of the Uncorrected Batch Softmax]]
  - [[#Problem 3 Formal Derivation of the Expected Batch Softmax Objective|Problem 3: Formal Derivation of the Expected Batch Softmax Objective]]
  - [[#Problem 4 LogQ Correction as Importance-Weighted Estimation|Problem 4: LogQ Correction as Importance-Weighted Estimation]]
  - [[#Problem 5 Consistency of the SBC Estimator|Problem 5: Consistency of the SBC Estimator]]
  - [[#Problem 6 Variance of the LogQ Correction Term|Problem 6: Variance of the LogQ Correction Term]]
  - [[#Problem 7 Streaming Estimator Bias and Variance|Problem 7: Streaming Estimator Bias and Variance]]
  - [[#Problem 8 Bias-Variance Tradeoff and Optimal Learning Rate|Problem 8: Bias-Variance Tradeoff and Optimal Learning Rate]]
  - [[#Problem 9 Tracking Lag Under Distribution Shift|Problem 9: Tracking Lag Under Distribution Shift]]
  - [[#Problem 10 Equivalence Between SBC and Importance Sampling from Uniform|Problem 10: Equivalence Between SBC and Importance Sampling from Uniform]]
  - [[#Problem 11 Effect of In-Batch Negatives on Gradient Variance|Problem 11: Effect of In-Batch Negatives on Gradient Variance]]
  - [[#Problem 12 Temperature and Softmax Entropy|Problem 12: Temperature and Softmax Entropy]]
  - [[#Problem 13 Lower Bound on Recall at K from Bias Magnitude|Problem 13: Lower Bound on Recall at K from Bias Magnitude]]
  - [[#Problem 14 Inner Product Decomposability and MIPS Indexing|Problem 14: Inner Product Decomposability and MIPS Indexing]]
  - [[#Problem 15 Multiple Hashings vs. Larger Single Array|Problem 15: Multiple Hashings vs. Larger Single Array]]
  - [[#Problem 16 Reward Weighting and the Learned Distribution|Problem 16: Reward Weighting and the Learned Distribution]]
  - [[#Problem 17 Sequential Training Under Distribution Shift|Problem 17: Sequential Training Under Distribution Shift]]
- [[#Algorithmic Applications|Algorithmic Applications]]
  - [[#Problem 18 Corrected Batch Softmax Loss|Problem 18: Corrected Batch Softmax Loss]]
  - [[#Problem 19 Distributed Streaming Frequency Estimation|Problem 19: Distributed Streaming Frequency Estimation]]
  - [[#Problem 20 Multiple-Hashing Frequency Estimator|Problem 20: Multiple-Hashing Frequency Estimator]]
  - [[#Problem 21 Recall at K Evaluation Against a Full Corpus|Problem 21: Recall at K Evaluation Against a Full Corpus]]
  - [[#Problem 22 Sequential Training with Online Frequency Updates|Problem 22: Sequential Training with Online Frequency Updates]]

---

## Mathematical Development

### Problem 1: Gradient of the Full Softmax Loss

**Key insight:** The softmax gradient has an exact contrastive decomposition — a "push toward the positive" term and a "pull toward the model-weighted centroid" — so any bias in the negative distribution directly distorts the centroid and corrupts the update direction.

**Sketch:**

(a) Write $\ell = -s_{y^+} + \log\sum_k e^{s_k}$. Differentiating: $\partial\ell/\partial s_j = -\mathbf{1}[j=y^+] + e^{s_j}/\sum_k e^{s_k} = p_j - \mathbf{1}[j=y^+]$.

(b) Chain rule over all $j$: $\partial\ell/\partial u = \sum_j (p_j - \mathbf{1}[j=y^+])v_j = \mathbb{E}_{j\sim P}[v_j] - v_{y^+}$. Gradient descent pushes $u$ toward $v_{y^+}$ and away from the model-weighted centroid of all items.

(c) MC estimator with $K$ negatives from $q$: $\hat{\nabla} \approx \frac{1}{K}\sum_k p^B_k v_k - v_{y^+}$. When $q \propto$ item frequency, popular items appear with high weight in the estimated centroid, biasing $\hat{\nabla}$ to push $u$ away from popular items even when their scores are appropriate.

---

### Problem 2: Bias of the Uncorrected Batch Softmax

**Key insight:** The probability of item $j$ entering a batch is proportional to $q_j$, so the batch denominator is an unbiased estimator of the frequency-weighted partition function $\sum_j q_j e^{s_j}$ — not the uniform-weighted $\sum_j e^{s_j}$ that the full softmax requires.

**Sketch:**

(a) Each of the $B-1$ negative slots draws from $q$ independently; $\mathbb{P}[y_j = y] \approx (B-1)q_y$ for large $M \gg B$. Effective negative distribution is $q$, not Uniform.

(b) $\mathbb{E}[\sum_{j\neq i} e^{s_{ij}}] = (B-1)\sum_j q_j e^{s_{ij}}$. Using $\log\mathbb{E}[Z]\approx\mathbb{E}[\log Z]$: $\mathbb{E}[-\log P_B]\approx -s_{ii}+\log\sum_j q_j e^{s_{ij}} + C$. The effective partition function is frequency-weighted, not uniform-weighted.

(c) For $\alpha=0$: $q_j = 1/M$, so $\mathrm{TV}(q,\mathrm{Uniform}) = 0$ and bias = 0. As $\alpha\to\infty$: all mass concentrates on item 1, so $q_1\to 1$ and $\mathrm{TV}\to 1/2$. The bias is a monotone increasing function of the power-law concentration $\alpha$.

---

### Problem 3: Formal Derivation of the Expected Batch Softmax Objective

**Key insight:** Taking expectation over random batch sampling is linear and directly yields $\sum_j q_j e^{s_{ij}}$ as the effective partition function, confirming that batch softmax is consistent to the wrong target — the frequency-reweighted softmax — no matter how large $B$ is.

**Sketch:**

(a) $\mathbb{E}[\hat{Z}_i] = \mathbb{E}[\sum_{j\neq i} e^{s_{ij}}] = (B-1)\sum_j q_j e^{s_{ij}}$, since each negative slot draws from $q$ independently.

(b) Applying $\log\mathbb{E}[Z] \approx \mathbb{E}[\log Z]$: $\mathbb{E}[-\log P_B(y_i|x_i)] \approx -s_{ii} + \log(B-1) + \log\sum_j q_j e^{s_{ij}}$. The effective objective is the softmax loss with partition function $\sum_j q_j e^{s_{ij}}$ (not $\sum_j e^{s_{ij}}$).

(c) The ratio $\sum_j q_j e^{s_{ij}}/\sum_j e^{s_{ij}}$ equals 1 when $q$ is uniform (bias = 0) and deviates as $q$ becomes concentrated. The corrected objective replaces $q_j$ with $1/M$ in the partition function, restoring the intended target.

---

### Problem 4: LogQ Correction as Importance-Weighted Estimation

**Key insight:** Subtracting $\log p_j$ from a logit is algebraically identical to multiplying the exponentiated score by $1/p_j$, which is the importance weight needed to convert samples from the training distribution into an unbiased estimator of the uniform-negative partition function.

**Sketch:**

(a) $\mathbb{E}[\hat{Z}_\mathrm{IS}] = \mathbb{E}_{j\sim q}[e^{s_j}/p_j] = \sum_j q_j \cdot e^{s_j}/p_j = \sum_j e^{s_j} = Z(x)$. Log-space: $e^{s_j}/p_j = e^{s_j - \log p_j} = e^{s^c_j}$.

(b) Residual bias from Jensen's inequality (concave $\log$): $\mathbb{E}[\log\hat{Z}^c] \leq \log\mathbb{E}[\hat{Z}^c] = \log Z(x) + C$. The gap is $O(\mathrm{Var}(\hat{Z}^c)/\mathbb{E}[\hat{Z}^c]^2) = O(1/B)$, vanishing as $B\to\infty$.

(c) If $p_j = B/M$ for all $j$: $s^c_j = s_j - \log(B/M) = s_j + \log(M/B)$. Adding a constant to all logits leaves the softmax unchanged. The correction is non-trivial only when $\mathrm{Var}(\log p_j) > 0$.

(d) Taylor: $\log(p_j+\epsilon_j) \approx \log p_j + \epsilon_j/p_j$, so $\hat{s}^c_j \approx s_j - \log p_j - \epsilon_j/p_j$. Estimation noise is amplified by $1/p_j$: a fixed absolute error in $\hat{p}_j$ has $1/p_j^2$ larger variance impact for rare items than for popular items.

---

### Problem 5: Consistency of the SBC Estimator

**Key insight:** The corrected denominator is a sum of i.i.d. importance weights that are individually unbiased for $Z(x_i)$, so the law of large numbers gives $\hat{Z}^c/B \to Z(x_i)$ and the continuous mapping theorem extends this to the log-loss.

**Sketch:**

(a) $\mathbb{E}[e^{s_{ij}-\log p_j}] = Z(x_i)$ for each negative slot. By LLN: $(B-1)^{-1}\sum_{j\neq i} e^{s^c_{ij}} \xrightarrow{p} Z(x_i)$. Hence $\hat{Z}^c/B \xrightarrow{p} Z(x_i)$.

(b) By the continuous mapping theorem applied to $\log(\cdot)$: $-\log P_B^c = -s^c_{ii} + \log\hat{Z}^c \xrightarrow{p} -s^c_{ii} + \log(BZ(x_i)) = -\log P(y_i|x_i) + \log B$. The $\log B$ additive constant cancels in the gradient.

(c) Uncorrected: $\mathbb{E}[\hat{Z}]/B \to \sum_j q_j e^{s_{ij}} \neq Z(x_i)$ (unless $q$ is uniform). The uncorrected estimator converges to the wrong quantity regardless of $B$.

---

### Problem 6: Variance of the LogQ Correction Term

**Key insight:** The correction term adds variance proportional to $\mathrm{Var}(e^{s_{ij}}/p_j)$ across items — this cost grows with the heterogeneity of $q$ but decreases as $O(1/B)$, making the correction always worthwhile when bias dominates at realistic batch sizes.

**Sketch:**

(a) Let $W_j = e^{s_{ij}}/p_j$; by unbiasedness $\mathbb{E}[W_j] = Z(x_i)$. $\mathrm{Var}(W_j) = \sum_k q_k e^{2s_{ik}}/p_k^2 - Z(x_i)^2$, which is larger when $q$ is concentrated (high $q_k$ for a few $k$ with potentially large $e^{s_{ik}}/p_k$).

(b) $\mathrm{Var}(\hat{Z}^c) = (B-1)\mathrm{Var}(W_j)$. Delta method: $\mathrm{Var}(-\log P_B^c) \approx \mathrm{Var}(\hat{Z}^c)/(\mathbb{E}[\hat{Z}^c])^2 = \mathrm{Var}(W_j)/((B-1)Z(x_i)^2)$.

(c) Variance increase from correction $= \mathrm{Var}(W_j) - \mathrm{Var}(e^{s_{ij}})$, which is non-negative. But bias reduction (targeting correct $Z$) eliminates $O(1)$ systematic error while variance increase is $O(1/B)$; for any reasonable $B$, bias correction dominates for heavily skewed $q$.

---

### Problem 7: Streaming Estimator Bias and Variance

**Key insight:** The EMA recurrence is a linear first-order system with fixed point $\delta$, so its mean and variance both admit closed-form geometric series expressions — bias decays geometrically at rate $(1-\alpha)^t$ and variance converges to a floor $\alpha\sigma^2/(2-\alpha)$.

**Sketch:**

(a) Induction: base $\delta_1 = (1-\alpha)\delta_0 + \alpha\Delta_1$. Step: $\delta_t = (1-\alpha)\delta_{t-1}+\alpha\Delta_t$, substitute inductive hypothesis, collect geometric series. $\mathbb{E}[\delta_t] = (1-\alpha)^t\delta_0 + \delta[1-(1-\alpha)^t]$; bias $= (1-\alpha)^t(\delta_0-\delta)$.

(b) $\mathrm{Var}(\delta_t) = \alpha^2\sigma^2\sum_{k=0}^{t-1}(1-\alpha)^{2k} = \frac{\alpha\sigma^2}{2-\alpha}[1-(1-\alpha)^{2t}]$; steady state $\sigma_\infty^2 = \alpha\sigma^2/(2-\alpha) \approx \alpha\sigma^2/2$ for small $\alpha$.

(c) Unbiased at every $t$: need $(1-\alpha)^t(\delta_0-\delta) = 0$ for all $t > 0$, which requires $\delta_0 = \delta$. The paper's phrasing $\delta_0 = \delta/(1-\alpha)$ removes the bias under its specific form of equation (7); both are consistent with initialization at the true mean when interpreted correctly.

---

### Problem 8: Bias-Variance Tradeoff and Optimal Learning Rate

**Key insight:** At steady state the bias is zero and MSE equals variance (monotone increasing in $\alpha$), so absent distribution shift the optimal $\alpha$ is as small as possible — distribution shift introduces a minimum effective $\alpha$ determined by the drift rate.

**Sketch:**

(a) Steady-state bias $= 0$; $\mathrm{MSE}_\infty(\alpha) = \alpha\sigma^2/(2-\alpha)$, monotone increasing in $\alpha$. Without shift, any smaller $\alpha$ is strictly better.

(b) Post-shift transient bias at lag $\tau$: $(1-\alpha)^\tau(\delta-\delta')$. Total MSE $= (1-\alpha)^{2\tau}(\delta-\delta')^2 + \alpha\sigma^2/(2-\alpha)$. Minimizing over $\alpha$ at fixed $\tau$ gives a positive interior optimum balancing the two terms.

(c) Continuous drift at rate $\beta$ per step: steady-state tracking bias $\approx \beta/\alpha$, variance $\approx \alpha\sigma^2/2$. Minimizing $(\beta/\alpha)^2 + \alpha\sigma^2/2$ over $\alpha$: $\alpha^* \propto (\beta^2/\sigma^2)^{1/3}$. As $\beta\to 0$: $\alpha^*\to 0$; as $\sigma^2\to\infty$: $\alpha^*\to 0$. The paper's $\alpha = 0.01$ is conservative, suitable for slow daily drift and moderate noise.

---

### Problem 9: Tracking Lag Under Distribution Shift

**Key insight:** After a step change the EMA satisfies a first-order exponential relaxation with decay constant $(1-\alpha)$ — the half-life is $\log(2)/\alpha$ steps, making large $\alpha$ the only way to achieve rapid adaptation at the cost of higher steady-state variance.

**Sketch:**

(a) Post-change at lag $\tau$: $\mathbb{E}[\delta_{t_0+\tau}] = (\delta+\Delta\delta) - (1-\alpha)^\tau\Delta\delta$. Tracking error $= (1-\alpha)^\tau|\Delta\delta|$.

(b) $(1-\alpha)^{\tau_{1/2}} = 1/2 \Rightarrow \tau_{1/2} = \log(2)/|\log(1-\alpha)| \approx \log(2)/\alpha$ for small $\alpha$.

(c) $\alpha = 0.01$: $\tau_{1/2} \approx 69$ steps. With millions of steps per day, the EMA adapts within seconds of wall-clock time — well within a single day. With $\alpha = 0.1$: $\tau_{1/2} \approx 7$ steps (much faster) but steady-state variance is $\approx 5.6\times$ larger ($\sigma_\infty^2 = 0.1\sigma^2/1.9$ vs. $0.01\sigma^2/1.99$).

---

### Problem 10: Equivalence Between SBC and Importance Sampling from Uniform

**Key insight:** The logQ correction converts samples from the training distribution $q$ into an unbiased MC estimator of the uniform-distribution expectation via the classical unnormalized IS identity — the two descriptions are algebraically identical.

**Sketch:**

(a) SNIS estimator $\hat{Z}_\mathrm{SNIS} = \sum_j e^{s_j}/q_j / \sum_j 1/q_j$ is biased $O(1/B)$ (from self-normalization). The unnormalized IS estimator $\hat{Z}_\mathrm{IS} = B^{-1}\sum_j e^{s_j}/q_j$ is exactly unbiased but has potentially larger variance.

(b) The corrected logit weight $e^{s_{ij}-\log p_j} = e^{s_{ij}}/p_j$ with $p_j = Bq_j$ equals $e^{s_{ij}}/(Bq_j)$, proportional to the IS weight $(1/M)/q_j$ for converting from $q$ to uniform. The SBC correction is IS reweighting in log-space.

(c) Items with $q_j\approx 0$ almost never appear in a batch ($\mathbb{P}[\text{item in batch}] = Bq_j\to 0$). The variance contribution from such items is $Bq_j\cdot(e^{s_j}/q_j - Z)^2 \to 0$, since the large IS weight $1/q_j$ is offset by the small appearance probability $Bq_j$. The infinite-variance concern is naturally bounded in the batch setting.

---

### Problem 11: Effect of In-Batch Negatives on Gradient Variance

**Key insight:** Gradient variance from batch negatives has two components: a $O(1/B)$ estimation term and a mismatch term that grows when the negative sampling distribution is far from the model distribution — the mismatch dominates early in training under heavy-tailed item frequencies.

**Sketch:**

(a) Full gradient: $g = \mathbb{E}_{j\sim P}[v_j] - v_{y^+}$. Batch MC approximation $\hat{g}$ is unbiased for the full gradient only if the effective batch weights $p_j^B$ equal the model weights $p_j$ in expectation, which requires the negative distribution to equal $P(\cdot|x_i;\theta)$.

(b) $\mathrm{Var}(\hat{g}) = O(1/B)$ by i.i.d. summation over $B-1$ negatives. The constant factor involves $\mathrm{Var}(W_j) = \mathrm{Var}(p_j^B)$, which grows when the batch softmax weights $p_j^B$ are volatile (large score variance across items).

(c) Early in training, $P(\cdot|x_i;\theta)\approx$ Uniform. Power-law $q$ concentrates the batch on popular items, making $p_j^B$ highly concentrated on a few items and nearly zero for the rest. This mismatch inflates $\mathrm{Var}(W_j)$ and raises gradient variance well above the uniform-sampling baseline.

---

### Problem 12: Temperature and Softmax Entropy

**Key insight:** Temperature controls the entropy of $P_\tau$ monotonically and has no effect on rankings (hence no effect on Recall@K at evaluation), but during training lower $\tau$ concentrates gradient mass on hard negatives, making temperature a training-dynamics knob rather than a model-capacity knob.

**Sketch:**

(a) $H(P_\tau) = \log Z_\tau - \tau^{-1}\mathbb{E}_{P_\tau}[s]$. As $\tau\to\infty$: $P_\tau\to$ Uniform, $H\to\log M$. As $\tau\to 0^+$: $P_\tau\to\delta_{j^*}$, $H\to 0$. Monotonicity follows from $\partial H/\partial\tau > 0$.

(b) $\partial\ell/\partial\tau = s_{y^+}/\tau^2 - \mathbb{E}_{P_\tau}[s]/\tau^2 = (1/\tau^2)(s_{y^+}-\mathbb{E}_{P_\tau}[s])$. When $s_{y^+} < \mathbb{E}[s]$ (positive below average), decreasing $\tau$ concentrates probability further away from the positive, increasing loss.

(c) Rankings depend only on the order of $\{s_j\}$, preserved under positive rescaling $s_j\mapsto s_j/\tau$. So Recall@K is $\tau$-invariant at evaluation. During training, smaller $\tau$ produces larger $p_j/\tau$ for high-scoring negatives, creating sharper learning signal focused on hard negatives near the decision boundary.

(d) As $K\to\infty$ uniform: $K^{-1}\sum_j e^{s_j/\tau}\xrightarrow{a.s.} M^{-1}\sum_j e^{s_j/\tau} = Z_\tau/M$. The InfoNCE denominator $\to e^{s^+/\tau} + K\cdot Z_\tau/M \approx K Z_\tau/M$ for large $K$ and $M$. So $\mathcal{L}_\mathrm{InfoNCE}\to-\mathbb{E}[\log P_\tau(y^+|x)] + \log(K/M)$, which matches full softmax cross-entropy up to the constant $\log(K/M)$.

---

### Problem 13: Lower Bound on Recall at K from Bias Magnitude

**Key insight:** The optimal biased model converges to a score function that additively subtracts $\log q_y$ from each item's true score, systematically under-ranking rare positives relative to popular negatives by an amount equal to the log-frequency gap.

**Sketch:**

(a) The biased loss $\approx -s(x,y) + \log\sum_j q_j e^{s(x,y_j)}$ is minimized by $P^B(y|x) \propto q_y e^{s^*(x,y)}$, giving $s^B(x,y) = s^*(x,y) + \log q_y - C(x)$ where $C(x)$ absorbs the query-dependent normalizer.

(b) Item $y^-$ overtakes $y^+$ under the biased model when $s^*(x,y^-) - s^*(x,y^+) > \log q_{y^+} - \log q_{y^-}$. The number of such items equals the rank degradation; the gap $\log q_{y^+} - \log q_{y^-}$ is the bias penalty proportional to the log-frequency advantage of the negative over the positive.

(c) $\Delta_\mathrm{bias}(y^+) = \mathbb{E}_{y^-\sim q}[\log q_{y^-}] - \log q_{y^+}$. Recall@1 under the biased model fails when $\Delta_\mathrm{bias}(y^+) > s^*(x,y^+) - \max_{y^-}s^*(x,y^-)$ (the true margin). Rare positives facing popular negatives have the largest $\Delta_\mathrm{bias}$ and suffer the most degradation.

---

### Problem 14: Inner Product Decomposability and MIPS Indexing

**Key insight:** Decomposability is the property that the scoring function factors into independently computed query and item functions — the inner product satisfies this by definition, enabling offline item indexing; any coupling of query and item features before aggregation breaks decomposability.

**Sketch:**

(a) $\langle u(x), v(y)\rangle$ depends on $x$ only through $u(x)$ and $y$ only through $v(y)$: fully decomposable. A model with concatenation $[u;v]$ inside a nonlinearity couples $x$ and $y$ before the final aggregation and is not decomposable: $v(y)$ cannot be precomputed independently of $u(x)$.

(b) For L2-normalized $u, v$: $\langle u,v\rangle = \cos\theta$. Also $\|u-v\|^2 = \|u\|^2 - 2\langle u,v\rangle + \|v\|^2 = 2 - 2\langle u,v\rangle$, so $\arg\max_j\langle u,v_j\rangle = \arg\min_j\|u-v_j\|^2_2$. MIPS reduces to exact nearest-neighbor search for L2-normalized embeddings.

(c) By Mercer's theorem, any symmetric PSD kernel $K(u,v) = \langle\phi(u),\phi(v)\rangle_\mathcal{H}$ yields a feature-map decomposition. Any PSD kernel is compatible with precomputed item representations (in the feature space). Non-PSD kernels and asymmetric scoring functions fail the symmetry or positive-definiteness condition and cannot be expressed as inner products.

---

### Problem 15: Multiple Hashings vs. Larger Single Array

**Key insight:** Collisions bias each EMA bucket downward (mixed inter-arrivals appear shorter), and taking the maximum over $m$ independent hash estimates selects the least-biased estimate — the debiasing from the max operation outweighs the higher per-array collision rate from smaller array sizes.

**Sketch:**

(a) $\mathbb{P}[\text{collision}] = 1-(1-1/H)^{M-1} \approx 1-e^{-M/H}$. For $M=10^7$, $H=5000$: $e^{-2000}\approx 0$. Every bucket has collisions; essentially all single-hash estimates are biased.

(b) Each $B_i[h_i(y)]$ is biased downward (collisions pull the EMA below $\delta_y$). The true $\delta_y$ upper-bounds each $B_i[h_i(y)]$ in expectation. Taking $\max_i$ selects the estimate with the fewest collisions, stochastically closest to $\delta_y$.

(c) Single array of size $mH$: collision probability $\approx 1-e^{-M/(mH)}$ (lower per bucket), but the max-over-hashings debiasing trick does not apply. With $m$ arrays of size $H/m$: per-array collision probability $\approx 1-e^{-Mm/H}$ (higher), but the max over $m$ independent estimates reduces directional bias by $O(m)$. Empirically the max-hashing strategy dominates for $m\geq 2$ under skewed distributions; the single large array would need $H\gg M$ (storage-prohibitive) to match.

---

### Problem 16: Reward Weighting and the Learned Distribution

**Key insight:** Reward-weighted cross-entropy is exactly cross-entropy against the reward-reweighted data distribution $p_r(x,y) \propto r(x,y)\cdot p_\mathrm{data}(x,y)$, so the optimal model recovers $p_r(y|x)$, not $p_\mathrm{data}(y|x)$.

**Sketch:**

(a) $\mathcal{L}(\theta) = -\sum_i r_i\log P(y_i|x_i;\theta) = -\sum_{x,y} n_r(x,y)\log P(y|x;\theta)$ where $n_r(x,y) \propto r(x,y)\cdot p_\mathrm{data}(x,y)$. Normalizing: $p_r(x,y) = r(x,y)\cdot p_\mathrm{data}(x,y)/\mathbb{E}[r]$. Minimizing $\mathcal{L}$ is equivalent to minimizing $\mathrm{KL}(p_r\|P_\theta)$.

(b) Optimal model: $P^*(y|x) \propto r(x,y)\cdot p_\mathrm{data}(y|x)$. This equals $p_\mathrm{data}(y|x)$ when $r(x,y)$ is constant in $y$ for each $x$ (reward independent of item given query).

(c) Fully-watched positives ($r=1$) are retained; abandoned clicks ($r=0$) are discarded. The model targets $P^*(y|x)\propto p_\mathrm{data}^\mathrm{watched}(y|x)$: items users watch to completion. Compared to binary-click model: clickbait with low completion rate is down-weighted to zero; long-form content with genuine engagement is up-weighted; content with moderate watch fractions occupies intermediate mass.

---

### Problem 17: Sequential Training Under Distribution Shift

**Key insight:** The EMA's recency weighting is only coherent when the temporal ordering of training examples is preserved — shuffling destroys the correspondence between global step $t$ and actual arrival time, making the EMA track a stationary mixture rather than the current distribution.

**Sketch:**

(a) With shuffled data, global step $t$ no longer corresponds to temporal order. The inter-arrival time $t - A[h(y)]$ measures steps since the last shuffled occurrence of $y$, which reflects the time-averaged frequency rather than the current frequency. The EMA converges to the stationary average distribution, not the most recent day's distribution.

(b) $\|\bar{\mathcal{D}}_W - \mathcal{D}_t\|_1 \leq \frac{1}{W}\sum_{\tau=0}^{W-1}\|\mathcal{D}_t - \mathcal{D}_{t-\tau}\|_1 = \frac{1}{W}\sum_{\tau=0}^{W-1}O(\beta\tau) = O(\beta W)$. To balance drift bias against statistical variance ($O(1/\sqrt{Wn})$): $\beta W \sim 1/\sqrt{Wn}$, giving $W^* \sim (\beta\sqrt{n})^{-2/3}$.

(c) Effective EMA weight given to an observation $\tau$ steps ago is $(1-\alpha)^\tau$; effective window $\approx \sum_\tau\tau(1-\alpha)^\tau\alpha = (1-\alpha)/\alpha \approx 1/\alpha$. For $\alpha=0.01$: window $\approx 100$ steps. With millions of steps per day over 15 days, the EMA window is a small fraction of a day — the estimator tracks item frequencies at sub-day resolution, far finer than the day-level distribution shift.

---

## Algorithmic Applications

### Problem 18: Corrected Batch Softmax Loss

**Key insight:** The corrected batch loss is structurally identical to standard batch softmax but with column-wise logit shifts applied before the log-sum-exp — a one-line change to the standard cross-entropy implementation.

**Sketch:**

```
function corrected_batch_loss(S, log_p_hat, r):
    # S: [B, B], S[i,j] = score(x_i, y_j)
    # log_p_hat: [B], log_p_hat[j] = log(p_hat_j)
    # r: [B], reward weights

    # (b) Column-wise logit correction (item dimension = columns)
    S_c = S - log_p_hat[newaxis, :]    # broadcast over rows; shape [B, B]

    # (c) Numerically stable row-wise log-softmax
    m = max(S_c, axis=1, keepdims=True)           # [B, 1]
    log_Z = m + log(sum(exp(S_c - m), axis=1))    # [B]
    log_probs = diag(S_c) - log_Z                 # [B] diagonal = positive logits

    # (d) Reward-weighted mean negative log-likelihood
    loss = -mean(r * log_probs)
    return loss
    # Time: O(B^2) dominated by score matrix; correction and loss are O(B)
```

(e) The positive item $i$ appears in both numerator (diagonal of $S^c$) and denominator (row sum of $S^c$). This matches the full softmax semantics where the positive is included in the partition function. No diagonal masking is needed — the paper's Eq. (4) includes the positive in the denominator, and the gradient correctly flows through both terms.

---

### Problem 19: Distributed Streaming Frequency Estimation

**Key insight:** The shared-state EMA update tolerates non-atomic read-write races because each conflicting write merely replaces the other's update rather than corrupting shared structure — at worst, one inter-arrival observation is lost, equivalent to a temporary increase in effective learning-rate variance.

**Sketch:**

```
# Parameter server shared state:
#   A[0..H-1]: int64, last global step for each bucket
#   B[0..H-1]: float32, EMA of inter-arrival times
#   t: global step counter (coordinator-managed)

function worker_update_and_estimate(y, alpha):
    k = h(y)
    # (b) Read (may be stale by at most one concurrent worker's write)
    a_k, b_k = PS.read(A[k]), PS.read(B[k])
    delta = t - a_k
    b_new = (1 - alpha) * b_k + alpha * delta
    # Write back (race: another worker may overwrite; we lose one observation)
    PS.write(A[k], t)
    PS.write(B[k], b_new)
    # (c) Use locally computed value — zero extra round-trip, staleness <= 1 step
    return 1.0 / b_new
```

(d) Strategy (i): $B[k]=1$ (assumes every item arrives every step). Severe overestimate of frequency; $\hat{p}$ starts much too large, under-correcting rare items. Bias decays in $O(1/\alpha)$ steps. Strategy (ii): $B[k] = M/B$ (uniform prior — each item appears on average every $M/B$ steps). Better-calibrated initial bias for all items; preferred when $M$ is known. Both converge to item-specific values in $O(1/\alpha)$ steps.

---

### Problem 20: Multiple-Hashing Frequency Estimator

**Key insight:** Algorithm 3 applies the same directional debiasing as the count-min sketch but in the opposite direction — collisions under-count inter-arrivals (over-estimate frequency), so taking the max of inter-arrival estimates (equivalently, min of frequency estimates) corrects toward the true, less-frequent value.

**Sketch:**

```
# Data structure: m arrays A_i[0..H/m-1], B_i[0..H/m-1], hash functions h_i
# Total memory: m * 2 * (H/m) = 2H entries, same as single-hash baseline

function multi_hash_update(y, t, alpha):
    for i in 1..m:                                 # independent updates
        k = h_i(y)
        B_i[k] = (1-alpha)*B_i[k] + alpha*(t - A_i[k])
        A_i[k] = t

function multi_hash_estimate(y):
    # (c) Max inter-arrival = least-biased estimate (collisions bias B_i downward)
    delta_hat = max(B_i[h_i(y)] for i in 1..m)
    return 1.0 / delta_hat
    # Contrast: count-min sketch uses min(counts) because collisions inflate counts
    # Here: collisions deflate delta -> max(delta) corrects in opposite direction
    # Both are the same operation: min(1/B_i) = 1/max(B_i)
```

(d) Update: $O(mB)$ per batch. Inference: $O(m)$ per item. For $m=4$, $H=5000$: each array has 1250 buckets. Collision probability per array: $1-e^{-10^7/1250}\approx 1$ (fully saturated). However the probability that all 4 hashes collide in the same direction is lower, so $\max_i B_i[h_i(y)]$ is closer to $\delta_y$ than any single $B_i$. Empirically this reduces total-variation error by $40$–$60\%$ relative to $m=1$ at the same memory budget.

---

### Problem 21: Recall at K Evaluation Against a Full Corpus

**Key insight:** Exact Recall@K has $O(NMk)$ complexity that is infeasible at YouTube scale — the ANN approximation trades a bounded recall gap (from index error and out-of-index items) for sub-linear query time, and the multi-positive generalization is a straightforward fraction of positives retrieved.

**Sketch:**

```
function evaluate_recall_at_k(U, V, test_pairs, K):
    # U: [N, k] query embeddings (computed online)
    # V: [M, k] item embeddings (precomputed offline, cached)
    # test_pairs: list of (query_idx, set_of_positive_item_indices)

    hits = []
    for (i, pos_set) in test_pairs:
        # (b) Exact: compute all M scores
        scores = U[i] @ V.T                        # [M]
        top_k = argsort_descending(scores)[:K]     # O(M log K)
        # (d) Multi-positive: fraction of positives retrieved
        n_retrieved = |set(top_k) & pos_set|
        hits.append(n_retrieved / |pos_set|)
    return mean(hits)
    # Time: O(NkM) for matrix multiply + O(NM log K) for sort
    # Single-positive special case: |pos_set| = 1, metric is 0 or 1
```

(c) ANN pipeline: build index over $\{v_j\}$ offline; retrieve approximate top-$K$ per $u_i$ in sub-linear time. ANN recall gap $= \mathrm{Recall@}K^\mathrm{exact} - \mathrm{Recall@}K^\mathrm{ANN}$ from two sources: (i) index retrieval error — ANN returns approximate top-$K$, missing the true positive when it lies in an unvisited region of the index; (ii) out-of-index items — videos uploaded after the last index build cannot be retrieved regardless of ANN quality.

---

### Problem 22: Sequential Training with Online Frequency Updates

**Key insight:** Frequency estimator updates must precede the logQ correction within each batch step — the correction uses the newly observed inter-arrival information, so updating first ensures the correction reflects the current batch's items rather than stale estimates.

**Sketch:**

```
function sequential_train(days, alpha, lr):
    for day in chronological_order(days):          # (a) outer loop: day-by-day
        for batch in day.mini_batches():
            t += 1

            # (b) Inner loop — ORDER MATTERS: update frequencies first
            log_p_hat = zeros(B)
            for j, y_j in enumerate(batch.items):
                # Step ii: EMA update with new inter-arrival observation
                k = h(y_j)
                delta = t - A[k]
                B[k] = (1-alpha)*B[k] + alpha*delta
                A[k] = t
                log_p_hat[j] = -log(B[k])        # step iii: read corrected estimate

            # Steps iv-v: compute corrected loss
            S = score_matrix(batch)               # [B, B], O(B^2 k)
            loss = corrected_batch_loss(S, log_p_hat, batch.rewards)

            # Step vi: gradient update
            theta -= lr * backprop(loss)

    wait_for_next_day()                           # (a) stay live after catch-up
```

(c) If frequency update happened after logit correction: items appearing for the first time in this batch would use $B[k]$ from their previous occurrence (potentially many steps ago). The correction $-\log(1/B[k])$ would reflect stale frequency, introducing a systematic error for newly popular items. Updating first ensures the correction uses the most current inter-arrival observation.

(d) Per-iteration complexity: $O(B^2 k)$ for score matrix and forward/backward pass (dominant); $O(Bm)$ for frequency updates (hash lookups and EMA, with $m\leq 4$). The ratio is $B^2 k / Bm = Bk/m \approx 256\cdot 128/4 = 8192$: frequency estimation overhead is less than 0.01% of the training computation.
