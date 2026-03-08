# Sampling-Bias-Corrected Retrieval: Exercises

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

*This problem establishes the contrastive structure of the softmax gradient that motivates the entire retrieval training framework. Understanding this structure clarifies why popular negatives distort training.*

> **Prerequisites:** cf. note [[note#Full Softmax Objective|§2 — Full Softmax Objective]]

(a) Fix a single training example $(x, y^+, r)$ with $r = 1$ and $M$ candidate items $\{y_j\}_{j=1}^M$. Let $p_j = P(y_j \mid x;\, \theta)$ and $s_j = s(x, y_j)$. Show that the gradient of the per-example loss $\ell = -\log P(y^+ \mid x;\, \theta)$ with respect to the score $s_j$ is:

$$\frac{\partial \ell}{\partial s_j} = p_j - \mathbf{1}[y_j = y^+]$$

(b) Using the chain rule $\partial \ell / \partial v(y_j) = (\partial \ell / \partial s_j) \cdot u(x, \theta)$, show that the total gradient of $\ell$ with respect to the query embedding $u(x, \theta)$ is:

$$\frac{\partial \ell}{\partial u(x, \theta)} = \mathbb{E}_{j \sim P(\cdot \mid x;\,\theta)}[v(y_j, \theta)] - v(y^+, \theta)$$

Interpret this contrastive form: what does the gradient push $u$ toward and away from?

(c) Suppose negatives are drawn from a biased distribution $q \neq \text{Uniform}$ and you form the approximate gradient using only $K$ sampled negatives. Write the Monte Carlo gradient estimator and state its expected value. Show that when $q$ is proportional to item frequency, the estimated gradient is biased toward over-penalizing popular items.

---

### Problem 2: Bias of the Uncorrected Batch Softmax

*This problem formally derives the bias introduced by using in-batch negatives sampled from the training distribution, establishing that batch softmax implicitly minimizes a reweighted objective rather than the intended full-corpus softmax.*

> **Prerequisites:** cf. note [[note#Batch Softmax and the Sampling Bias Problem|§2 — Batch Softmax and the Sampling Bias Problem]]

(a) Let $q_j$ denote the marginal probability of item $y_j$ in the training distribution. In a batch of size $B$ drawn i.i.d. from the training distribution, show that the probability of item $j$ appearing as a negative for query $x_i$ is approximately $(B-1) q_j$ for large $M \gg B$. Hence state the effective negative distribution of the uncorrected batch softmax.

(b) The uncorrected batch softmax log-loss for query $i$ is $-\log P_B(y_i \mid x_i;\,\theta)$ where:

$$P_B(y_i \mid x_i;\,\theta) = \frac{e^{s_{ii}}}{\sum_{j=1}^B e^{s_{ij}}}$$

Taking expectation over random batches, show that:

$$\mathbb{E}\!\left[-\log P_B(y_i \mid x_i)\right] \approx -s_{ii} + \log\!\left(\frac{1}{B}\sum_{j} e^{s_{ij}} \cdot \frac{q_j}{\bar{q}}\right) + C$$

for some constant $C$ and normalization $\bar{q}$, identifying the effective partition function being estimated. This shows that batch softmax is an unbiased estimator of the wrong (reweighted) objective.

(c) Suppose item frequencies follow a Zipf law $q_j \propto j^{-\alpha}$ for $j = 1, \ldots, M$. Define the bias magnitude as $\mathrm{TV}(q, \mathrm{Uniform}) = \frac{1}{2}\sum_j |q_j - 1/M|$. Show that $\mathrm{TV}(q, \mathrm{Uniform}) \to 1/2$ as $\alpha \to \infty$ (all mass on one item) and $\mathrm{TV}(q, \mathrm{Uniform}) = 0$ for $\alpha = 0$ (uniform). Conclude that the bias is a monotone increasing function of the power-law concentration.

---

### Problem 3: Formal Derivation of the Expected Batch Softmax Objective

*This problem makes precise what objective the uncorrected batch softmax actually minimizes in expectation, confirming that it targets a frequency-reweighted partition function rather than the uniform-negative full softmax.*

> **Prerequisites:** cf. note [[note#Batch Softmax and the Sampling Bias Problem|§2 — Batch Softmax and the Sampling Bias Problem]]; requires Problem 2

(a) For a fixed query $x_i$ with positive item $y_i$, let the batch items $\{y_j : j \neq i\}$ be drawn i.i.d. from the item frequency distribution $q$. Define the random variable $\hat{Z}_i = \sum_{j \neq i} e^{s_{ij}}$. Show that $\mathbb{E}[\hat{Z}_i] = (B-1)\sum_j q_j e^{s_{ij}}$, identifying the (weighted) partition function being estimated.

(b) Using the approximation $\log \mathbb{E}[\hat{Z}_i] \approx \mathbb{E}[\log \hat{Z}_i]$ (valid when $\mathrm{Var}(\hat{Z}_i)/\mathbb{E}[\hat{Z}_i]^2 \ll 1$), show that the expected batch loss satisfies:

$$\mathbb{E}[-\log P_B(y_i \mid x_i)] \approx -s_{ii} + \log\!\left(\sum_j q_j e^{s_{ij}}\right) + C$$

where $C$ is a constant independent of $\theta$. Identify this as a softmax loss with partition function $\sum_j q_j e^{s_{ij}}$ (frequency-weighted, not uniform-weighted).

(c) The corrected objective targets the partition function $\sum_j e^{s_{ij}}$ (uniform weights). Show that the ratio of the two partition functions is $\sum_j q_j e^{s_{ij}} / \sum_j e^{s_{ij}}$, and interpret what the model must learn in each case. When does the gap between the two objectives vanish?

---

### Problem 4: LogQ Correction as Importance-Weighted Estimation

*This problem derives the logQ correction from the importance-sampling identity, showing it implements an unbiased estimator of the full-corpus partition function in log-space.*

> **Prerequisites:** cf. note [[note#The LogQ Correction|§3 — The LogQ Correction]]; cf. note [[note#Connection to Importance Sampling|§3 — Connection to Importance Sampling]]

(a) The full softmax partition function is $Z(x) = \sum_{j=1}^M e^{s(x, y_j)}$. In-batch items are sampled with probabilities $\{p_j\}$. Show that the importance-weighted estimator:

$$\hat{Z}_{\mathrm{IS}}(x) = \frac{1}{B}\sum_{j \in \mathrm{batch}} \frac{e^{s(x, y_j)}}{p_j}$$

satisfies $\mathbb{E}[\hat{Z}_{\mathrm{IS}}(x)] = Z(x)$. Write $e^{s(x,y_j)}/p_j = e^{s(x,y_j) - \log p_j}$ to show the logQ correction implements this in log-space.

(b) Show that the corrected batch softmax probability:

$$P_B^c(y_i \mid x_i;\,\theta) = \frac{e^{s^c_{ii}}}{e^{s^c_{ii}} + \sum_{j \neq i} e^{s^c_{ij}}}$$

where $s^c_{ij} = s_{ij} - \log p_j$, is an approximately unbiased estimator of the full softmax $P(y_i \mid x_i;\,\theta)$. Identify the residual source of bias: the log of an average is not the average of logs (Jensen's inequality for the concave $\log$ function).

(c) Consider the case where all items are equally frequent: $p_j = B/M$ for all $j$. Show that the logQ correction reduces to subtracting a constant from all logits, which does not change the softmax output. Conclude that the correction is only non-trivial when $\{p_j\}$ is non-uniform, and that the effective magnitude of the correction scales with the variance of $\log p_j$ across batch items.

(d) In practice, $p_j$ is estimated with noise: $\hat{p}_j = p_j + \epsilon_j$. Using a first-order Taylor expansion, show that the corrected logit with noisy $\hat{p}_j$ is $\hat{s}^c_j \approx s_j - \log p_j - \epsilon_j / p_j$. Conclude that estimation error in the correction is amplified by $1/p_j$: rare items (small $p_j$) are far more sensitive to frequency estimation error than popular items.

---

### Problem 5: Consistency of the SBC Estimator

*This problem proves that the sampling-bias-corrected batch loss is a consistent estimator of the full softmax loss as batch size grows, formalizing the claim that the correction asymptotically eliminates the sampling bias.*

> **Prerequisites:** cf. note [[note#Corrected Batch Loss|§3 — Corrected Batch Loss]]; requires Problem 4

(a) Let $\hat{Z}^c(x_i) = e^{s^c_{ii}} + \sum_{j \neq i} e^{s^c_{ij}}$ denote the corrected denominator for query $i$ in a batch of size $B$. Show that $\mathbb{E}[\hat{Z}^c(x_i)]$ equals $e^{s^c_{ii}} + (B-1) Z(x_i)$ where $Z(x_i) = \sum_j e^{s_{ij}}$ is the full partition function. Hence $\hat{Z}^c(x_i)/B \to Z(x_i)$ in probability as $B \to \infty$ by the law of large numbers.

(b) Using part (a) and the continuous mapping theorem applied to $\log(\cdot)$, show that:

$$-\log P_B^c(y_i \mid x_i;\,\theta) \xrightarrow{p} -\log P(y_i \mid x_i;\,\theta) \quad \text{as } B \to \infty$$

This confirms consistency: the corrected batch loss converges in probability to the full softmax loss as batch size increases.

(c) For the uncorrected batch softmax, show that $\hat{Z}(x_i)/B \to \sum_j q_j e^{s_{ij}}$ rather than $Z(x_i)$, confirming that the uncorrected estimator is inconsistent (converges to the wrong quantity even as $B \to \infty$).

---

### Problem 6: Variance of the LogQ Correction Term

*This problem analyzes how the logQ correction term itself contributes to gradient variance, revealing the intrinsic cost of bias correction.*

> **Prerequisites:** cf. note [[note#The LogQ Correction|§3 — The LogQ Correction]]; requires Problem 4

(a) In a batch of size $B$, the corrected denominator for query $i$ is $\sum_{j \neq i} e^{s_{ij} - \log p_j} = \sum_{j \neq i} e^{s_{ij}}/p_j$. Let $W_j = e^{s_{ij}}/p_j$ for the random batch item $j$. Show that $\mathrm{Var}(W_j) = \mathbb{E}[e^{2s_{ij}}/p_j^2] \cdot \mathrm{Var}(p_j) + \ldots$ and that $\mathrm{Var}(W_j)$ is larger when item frequency is more heterogeneous.

(b) The variance of the corrected partition function estimate is $\mathrm{Var}(\hat{Z}^c) = (B-1)\,\mathrm{Var}(W_j)$. Show that the variance of the corrected log-loss is approximately:

$$\mathrm{Var}(-\log P_B^c) \approx \frac{\mathrm{Var}(\hat{Z}^c)}{(\mathbb{E}[\hat{Z}^c])^2} = \frac{\mathrm{Var}(W_j)}{(B-1)\,Z(x_i)^2}$$

using the delta method for the variance of $\log(\cdot)$.

(c) Compare the variance of the corrected and uncorrected batch loss. In what regime (large $B$, very non-uniform $q$, or very uniform $q$) does the correction substantially increase variance? Argue that the bias-reduction benefit outweighs the variance increase when the item distribution is heavily skewed.

---

### Problem 7: Streaming Estimator Bias and Variance

*This problem fills in the details of Proposition 4.1, providing exact closed forms for the bias and variance of the exponential moving average estimator of inter-arrival times.*

> **Prerequisites:** cf. note [[note#Proposition 4.1: Bias and Variance Analysis|§4 — Proposition 4.1: Bias and Variance Analysis]]

(a) Let $\delta_t = (1-\alpha)\delta_{t-1} + \alpha\Delta_t$ with $\{\Delta_t\}$ i.i.d., mean $\delta$, variance $\sigma^2$, and initial value $\delta_0$. Prove by induction that:

$$\delta_t = (1-\alpha)^t \delta_0 + \alpha \sum_{k=1}^t (1-\alpha)^{t-k} \Delta_k$$

Hence compute $\mathbb{E}[\delta_t]$ and derive the exact bias $\mathbb{E}[\delta_t] - \delta = (1-\alpha)^t(\delta_0 - \delta)$. Reconcile this with equation (7) of the paper, which states $\mathbb{E}[\delta_t] - \delta = (1-\alpha)^t \delta_0 - (1-\alpha)^{t-1}\delta$.

(b) Using the representation from (a) and independence of $\Delta_1, \ldots, \Delta_t$, compute $\mathrm{Var}(\delta_t)$ exactly. Show that it equals:

$$\mathrm{Var}(\delta_t) = \frac{\alpha\sigma^2}{2-\alpha}\left[1-(1-\alpha)^{2t}\right]$$

As $t \to \infty$, identify the steady-state variance $\sigma_\infty^2 = \alpha\sigma^2/(2-\alpha) \approx \alpha\sigma^2/2$ for small $\alpha$.

(c) Find the ideal initialization $\delta_0^*$ that makes $\mathbb{E}[\delta_t] = \delta$ for all $t \geq 0$ (unbiased at every step, not just in the limit). Show that $\delta_0^* = \delta/(1-\alpha)$. Since $\delta$ is unknown in practice, discuss which direction of initialization error (over vs. under) is less harmful and why.

---

### Problem 8: Bias-Variance Tradeoff and Optimal Learning Rate

*This problem characterizes the optimal learning rate for the streaming estimator, showing it depends on the signal-to-noise ratio of the inter-arrival time distribution.*

> **Prerequisites:** cf. note [[note#Proposition 4.1: Bias and Variance Analysis|§4 — Proposition 4.1: Bias and Variance Analysis]]; requires Problem 7

(a) Define the steady-state mean squared error of the estimator as $\mathrm{MSE}_\infty(\alpha) = \lim_{t \to \infty} \mathrm{MSE}(\delta_t)$. Using your results from Problem 7, show that the steady-state bias is zero and $\mathrm{MSE}_\infty(\alpha) = \alpha\sigma^2/(2-\alpha)$.

(b) Now suppose the true mean undergoes a step change at step $t_0$: $\delta \to \delta'$ for $t > t_0$. After the change, the estimator has a transient bias. Show that the post-change tracking bias at step $t_0 + \tau$ is $(1-\alpha)^\tau(\delta - \delta')$ and that the total MSE (bias squared plus variance) is minimized by choosing $\alpha$ to balance these terms.

(c) In the regime where distribution shift occurs at rate $\beta$ per step (i.e., the true $\delta$ changes by $\beta$ per step), show that the optimal $\alpha^*$ scales as $(\beta^2/\sigma^2)^{1/3}$ (heuristic). Interpret: when the signal changes slowly ($\beta \to 0$), optimal $\alpha^* \to 0$; when the signal is very noisy ($\sigma^2 \to \infty$), $\alpha^* \to 0$ as well. Conclude that $\alpha = 0.01$ as used in the paper is conservative in both directions.

---

### Problem 9: Tracking Lag Under Distribution Shift

*This problem quantifies the delay with which the streaming estimator adapts after a distributional change, establishing the half-life formula for the exponential moving average.*

> **Prerequisites:** cf. note [[note#Algorithm 2: Single Hash Estimation|§4 — Algorithm 2: Single Hash Estimation]]; requires Problem 7

(a) At step $t_0$ the true mean changes from $\delta$ to $\delta + \Delta\delta$. Starting from $\mathbb{E}[\delta_{t_0}] = \delta$, derive the expected value $\mathbb{E}[\delta_{t_0+\tau}]$ for $\tau \geq 1$. Show that the tracking error $|\mathbb{E}[\delta_{t_0+\tau}] - (\delta + \Delta\delta)|$ decays as $(1-\alpha)^\tau |\Delta\delta|$.

(b) Define the half-life $\tau_{1/2}$ as the number of steps after which the tracking error has halved: $(1-\alpha)^{\tau_{1/2}} = 1/2$. Solve to get $\tau_{1/2} = \log(2)/|\log(1-\alpha)|$. Show that for small $\alpha$, $\tau_{1/2} \approx \log(2)/\alpha \approx 0.693/\alpha$.

(c) For the paper's choice of $\alpha = 0.01$, compute $\tau_{1/2}$. Given that the paper's system processes many items per batch and many batches per day, interpret whether this half-life is short or long relative to the timescale of distribution shift in YouTube (which operates on a timescale of days). What would happen if $\alpha$ were set to $0.1$?

---

### Problem 10: Equivalence Between SBC and Importance Sampling from Uniform

*This problem establishes that sampling-bias-corrected batch softmax and importance sampling from the uniform distribution over items are two descriptions of the same estimator, grounding the correction in classical statistical theory.*

> **Prerequisites:** cf. note [[note#Connection to Importance Sampling|§3 — Connection to Importance Sampling]]; requires Problem 4

(a) Let $U$ denote the uniform distribution over $M$ items. An importance sampling estimator of $Z(x) = \sum_j e^{s(x,y_j)}$ using samples from item frequency distribution $q$ reweights by $u_j/q_j = (1/M)/q_j$. Show that the self-normalized importance sampling (SNIS) estimator from $q$:

$$\hat{Z}_{\mathrm{SNIS}}(x) = \frac{\sum_{j \in S} e^{s_{ij}}/q_j}{\sum_{j \in S} 1/q_j}$$

is a consistent but biased estimator of $Z(x)$. Contrast with the (unnormalized) importance sampling estimator $\hat{Z}_{\mathrm{IS}}$ from Problem 4(a).

(b) Show that the SBC correction (subtracting $\log p_j \approx \log(B \cdot q_j)$ from each logit) is equivalent to reweighting from the training distribution $q$ back toward uniform. Specifically, the corrected weight $e^{s_{ij} - \log p_j} = e^{s_{ij}}/p_j$ matches the importance weight $e^{s_{ij}} \cdot (1/p_j)$ from $p$ to uniform.

(c) The unnormalized IS estimator from (a) is unbiased but has potentially infinite variance when $q_j \to 0$ for some items. Argue that in the batch setting this infinite-variance issue is naturally bounded by the batch size $B$: items with $q_j \approx 0$ almost never appear in the batch, so their infinite importance weights have negligible probability of appearing. Formalize using the truncation: the effective IS weights in a batch are bounded by $1/q_{\min}$ where $q_{\min} = \min_j q_j$.

---

### Problem 11: Effect of In-Batch Negatives on Gradient Variance

*This problem analyzes how the choice of negative sampling distribution affects gradient variance, showing the fundamental tension between computational efficiency and estimation quality.*

> **Prerequisites:** cf. note [[note#Batch Softmax and the Sampling Bias Problem|§2 — Batch Softmax and the Sampling Bias Problem]]; requires Problem 1

(a) For a fixed query $x_i$, consider the gradient of the per-example loss with respect to the query embedding. In the full softmax, this gradient is $\nabla_{u_i}\ell_i = \mathbb{E}_{j \sim P(\cdot|x_i)}[v_j] - v_{y_i}$. With batch negatives, it is approximated by $\hat{\nabla}_{u_i}\ell_i = \frac{1}{B-1}\sum_{j \neq i} p_j^B v_j - v_{y_i}$ where $p_j^B$ is the batch softmax probability. Show that this is an unbiased estimator of the full gradient only when the negative sampling distribution equals $P(\cdot|x_i;\,\theta)$.

(b) The gradient variance for a single query is $\mathrm{Var}(\hat{\nabla}) = \mathrm{Var}\!\left(\frac{1}{B-1}\sum_{j \neq i} W_j v_j\right)$ where $W_j = p_j^B - \mathbf{1}[j = i]$ are stochastic weights. Show that this variance decreases as $O(1/B)$ and increases when the negative sampling distribution is far from the model distribution (more variance is introduced by the mismatch in weights).

(c) Sampling negatives from the training distribution $q$ rather than from the model distribution $P(\cdot|x_i;\,\theta)$ introduces bias (as shown in Problem 2) but may reduce variance if $q$ is close to $P$. Argue that early in training when $P$ is close to uniform, sampling from $q$ with power-law concentration can substantially increase gradient variance relative to uniform sampling, because very popular items dominate the denominator.

---

### Problem 12: Temperature and Softmax Entropy

*This problem characterizes how temperature controls the entropy of the softmax distribution and, through that, the sharpness of the gradient signal during training.*

> **Prerequisites:** cf. note [[note#Formal Setup|§2 — Formal Setup]]

(a) Let $P_\tau(j) = e^{s_j/\tau} / \sum_k e^{s_k/\tau}$ over scores $\{s_j\}_{j=1}^M$. Compute the entropy $H(P_\tau)$ and show it is increasing in $\tau$. Verify the limiting cases: $H(P_\tau) \to \log M$ as $\tau \to \infty$ and $H(P_\tau) \to 0$ as $\tau \to 0^+$.

(b) Show that the gradient of the per-example loss $\ell = -\log P_\tau(y^+)$ with respect to $\tau$ is:

$$\frac{\partial \ell}{\partial \tau} = \frac{1}{\tau^2}\left(\mathbb{E}_{j \sim P_\tau}[s_j] - s_{y^+}\right)$$

Interpret: when the positive item score $s_{y^+}$ is above the model-weighted average, reducing $\tau$ reduces the loss. Identify when decreasing $\tau$ increases the loss.

(c) Recall@$K$ is a deterministic function of item rankings. Show that ranking is invariant to positive rescaling of all logits by $1/\tau$, so Recall@$K$ at evaluation time is independent of $\tau$ for a fixed trained model. However, explain why $\tau$ matters during training: it controls the relative gradient magnitude for hard negatives (items with scores close to $s_{y^+}$) versus easy negatives (items with scores far below $s_{y^+}$).

(d) Show that the InfoNCE loss $\mathcal{L}_{\mathrm{InfoNCE}} = -\mathbb{E}[\log \frac{e^{s(x,y^+)/\tau}}{e^{s(x,y^+)/\tau} + \sum_{j=1}^K e^{s(x,y_j^-)/\tau}}]$ with $K$ negatives drawn uniformly converges as $K \to \infty$ to the full softmax cross-entropy loss (up to a constant), establishing batch softmax as a special case of InfoNCE.

---

### Problem 13: Lower Bound on Recall at K from Bias Magnitude

*This problem derives a formal relationship between the magnitude of the sampling bias and the degradation in Recall@K, providing a theoretical justification for why the correction improves retrieval metrics.*

> **Prerequisites:** cf. note [[note#Batch Softmax and the Sampling Bias Problem|§2 — Batch Softmax and the Sampling Bias Problem]]; cf. note [[note#Wikipedia Link Prediction|§6 — Wikipedia Link Prediction]]; requires Problem 3

(a) Let $s_j^*(\theta)$ be the score of the $j$-th item under the model trained with full softmax, and let $s_j^B(\theta^B)$ be the score under the model trained with uncorrected batch softmax. The bias in the objective causes the two models to converge to different parameter values. Formalize: the uncorrected model converges to $\theta^B = \arg\min_\theta \mathbb{E}[-\log P_B(y \mid x)]$, which by Problem 3 targets a frequency-weighted loss. Show that the optimal score function under the biased objective satisfies $s^B(x,y) = s^*(x,y) - \log q_y + C(x)$ for some query-dependent constant $C(x)$, where $q_y$ is item frequency.

(b) For a query $x$ with true positive $y^+$, the rank of $y^+$ under the biased model is degraded relative to the unbiased model when the correction term $\log q_{y^+}$ differs significantly from $\log q_{y^-}$ for competing items $y^-$. Show that the rank of $y^+$ under the biased model is at most its rank under the corrected model plus the number of items $y^-$ satisfying $s^*(x,y^-) > s^*(x,y^+) - (\log q_{y^+} - \log q_{y^-})$.

(c) Define the bias gap for a positive item as $\Delta_{\mathrm{bias}}(y^+) = \mathbb{E}_{y^- \sim q}[\log q_{y^-}] - \log q_{y^+}$ (the expected log-frequency advantage of negatives over the positive). Show that Recall@1 under the biased model is upper-bounded by $\mathbb{P}[\Delta_{\mathrm{bias}}(y^+) < s^*(x,y^+) - \max_{y^-} s^*(x,y^-)]$. Interpret: if the positive item is rare and competing negatives are popular, the bias gap is large and Recall@1 degrades.

---

### Problem 14: Inner Product Decomposability and MIPS Indexing

*This problem derives the mathematical condition under which a scoring function is compatible with precomputed item embeddings, establishing the inner product as the canonical indexable scoring function.*

> **Prerequisites:** cf. note [[note#The Two-Phase Retrieval-Ranking Architecture|§1 — The Two-Phase Retrieval-Ranking Architecture]]; cf. note [[note#Formal Setup|§2 — Formal Setup]]

(a) A scoring function $f(x, y)$ is called *decomposable* if it can be written as $f(x,y) = g(u(x), v(y))$ for independently computed maps $u$ and $v$ and some bivariate function $g$. Show that $f(x,y) = \langle u(x), v(y) \rangle$ is decomposable, but $f(x,y) = \text{ReLU}([u(x); v(y)]^\top W [u(x); v(y)])$ (a bilinear form with concatenated features) is not decomposable in general.

(b) For L2-normalized embeddings, show algebraically that $\langle u, v \rangle = \cos\theta(u,v)$ (cosine similarity) and that $\arg\max_j \langle u, v_j \rangle = \arg\min_j \|u - v_j\|_2^2$ (exact nearest neighbor). This equivalence means MIPS reduces to nearest neighbor search for L2-normalized embeddings.

(c) Mercer's theorem states that a symmetric positive-definite kernel $K(u,v)$ admits a representation $K(u,v) = \langle \phi(u), \phi(v) \rangle_\mathcal{H}$ in some (possibly infinite-dimensional) Hilbert space $\mathcal{H}$. Conclude that any such kernel can in principle be approximated by a finite-dimensional inner product, making the two-tower inner product model a universal approximator for symmetric PD scoring functions. State the condition that fails for asymmetric scoring functions.

---

### Problem 15: Multiple Hashings vs. Larger Single Array

*This problem analyzes the count-min-sketch-style design of Algorithm 3, proving that multiple independent hashings reduce collision bias more efficiently than enlarging a single hash array.*

> **Prerequisites:** cf. note [[note#Algorithm 3: Multiple Hashings|§4 — Algorithm 3: Multiple Hashings]]

(a) With a single hash array of size $H$ and corpus of $M$ items, the expected number of items colliding into a given bucket is $M/H$. Show that the probability that item $y$ collides with at least one other item is $1 - (1 - 1/H)^{M-1} \approx 1 - e^{-M/H}$. For $M = 10^7$ and $H = 5000$, compute this probability.

(b) With $m$ arrays each of size $H/m$ (same total memory), each array has collision probability $1 - e^{-Mm/H}$ (higher per array, since arrays are smaller). However, Algorithm 3 takes $\hat{\delta}(y) = \max_i B_i[h_i(y)]$. Show that the maximum of $m$ independent estimates is less biased than any single estimate when all estimates are biased downward (collisions pull $B_i[h_i(y)]$ below $\delta_y$).

(c) Contrast Algorithm 3 with using a single array of size $mH$ (all the memory in one table). Compare on: (i) expected number of collisions per item (state formulas for both), and (ii) whether the maximum-over-hashings trick applies. Conclude which strategy is preferable when $m \geq 2$ and justify.

---

### Problem 16: Reward Weighting and the Learned Distribution

*This problem characterizes the implicit target distribution of the reward-weighted training objective, showing it differs from the raw data distribution in a manner controlled by the reward function.*

> **Prerequisites:** cf. note [[note#Full Softmax Objective|§2 — Full Softmax Objective]]; cf. note [[note#Sequential Training|§5 — Sequential Training]]

(a) The reward-weighted log-likelihood is $\mathcal{L}(\theta) = -\frac{1}{T}\sum_i r_i \log P(y_i \mid x_i;\,\theta)$. Show that minimizing this is equivalent to minimizing the KL divergence $\mathrm{KL}(p_r \| P_\theta)$ where $p_r(x,y) \propto r(x,y) \cdot p_{\mathrm{data}}(x,y)$ is a reward-reweighted distribution. Identify $p_r$ explicitly.

(b) In the limit of zero approximation error (expressive model), show that the optimal model satisfies $P^*(y \mid x) \propto r(x,y) \cdot p_{\mathrm{data}}(y \mid x)$. When is this proportional to the pure conditional data distribution $p_{\mathrm{data}}(y \mid x)$?

(c) The paper sets $r_i = 0$ for abandoned clicks and $r_i = 1$ for fully watched videos. Using your answer to (b), describe qualitatively how the retrieval distribution $P^*(y \mid x)$ differs between the reward-weighted model and a binary-click model. What types of content are relatively up-weighted or down-weighted?

---

### Problem 17: Sequential Training Under Distribution Shift

*This problem formalizes why sequential (non-shuffled) training is necessary for the streaming frequency estimator to work correctly, and quantifies the distribution-shift bias of a sliding-window approach.*

> **Prerequisites:** cf. note [[note#Sequential Training|§5 — Sequential Training]]; cf. note [[note#Algorithm 2: Single Hash Estimation|§4 — Algorithm 2: Single Hash Estimation]]

(a) If training data were randomly shuffled (mixing all days uniformly), the global step counter $t$ in Algorithm 2 would no longer correspond to temporal ordering. Show that the exponential moving average update $B[h(y)] \leftarrow (1-\alpha)B[h(y)] + \alpha(t - A[h(y)])$ would then track the average inter-arrival time under the time-averaged distribution, not the current distribution. Conclude that the recency weighting built into the EMA is destroyed by shuffling.

(b) Let $\mathcal{D}_t$ denote the item frequency distribution at day $t$, and model drift as $\mathcal{D}_t = (1-\beta)\mathcal{D}_{t-1} + \beta\mathcal{D}^{\mathrm{new}}$ for some drift rate $\beta \in (0,1)$ per day. A sliding-window model trained equally on the past $W$ days uses the time-averaged distribution $\bar{\mathcal{D}}_W = \frac{1}{W}\sum_{t'=t-W+1}^t \mathcal{D}_{t'}$. Show that the distribution shift bias relative to the current day is $\|\bar{\mathcal{D}}_W - \mathcal{D}_t\|_1 = O(\beta W)$.

(c) The paper's sequential training with EMA frequency estimation automatically upweights recent observations. Show that the effective weight given to observations from $\tau$ steps ago is $(1-\alpha)^\tau$, making the effective window length approximately $1/\alpha$ steps. For $\alpha = 0.01$, compute this effective window length and compare to the paper's description of training on 15+ days sequentially.

---

## Algorithmic Applications

### Problem 18: Corrected Batch Softmax Loss

*Sketch a complete, numerically stable algorithm for computing the corrected batch softmax loss from Equation (4) of the paper.*

> **Prerequisites:** cf. note [[note#Corrected Batch Loss|§3 — Corrected Batch Loss]]

(a) **Inputs and data structures.** Define the inputs: score matrix $S \in \mathbb{R}^{B \times B}$ where $S_{ij} = s(x_i, y_j)$, log-sampling-probability vector $\ell_p \in \mathbb{R}^B$ where $\ell_p[j] = \log \hat{p}_j$, and reward vector $r \in \mathbb{R}^B$. State what each dimension of $S$ represents.

(b) **Logit correction.** Describe how to form the corrected score matrix $S^c$ by broadcasting $\ell_p$ over the item dimension (columns). Write the element-wise operation: $S^c[i,j] = S[i,j] - \ell_p[j]$ for all $i, j$. Explain why the subtraction is along the item (column) axis, not the query (row) axis.

(c) **Numerically stable log-softmax.** Describe the log-sum-exp stabilization: subtract $\max_j S^c[i,j]$ from each row before exponentiation. Write pseudocode for `row_log_softmax(S_c)` and state why this is numerically equivalent to the naive formula but avoids overflow.

(d) **Diagonal extraction and loss.** The positive for query $i$ is item $i$ (diagonal of $S^c$). Extract the diagonal log-softmax values, multiply by rewards, negate, and average. Write complete pseudocode for `corrected_batch_loss(S, log_p_hat, r)` and state its time complexity in $B$.

(e) **Self-negative handling.** For each query $i$, item $i$ appears in both the numerator and the denominator of the softmax. Explain whether this is correct (the self-negative is included in the denominator) or whether it should be masked out, and what the paper does.

---

### Problem 19: Distributed Streaming Frequency Estimation

*Sketch Algorithm 2 adapted for a distributed training setting with multiple workers sharing state via a parameter server.*

> **Prerequisites:** cf. note [[note#Algorithm 2: Single Hash Estimation|§4 — Algorithm 2: Single Hash Estimation]]

(a) **Data structure.** Describe the shared state on the parameter server: arrays $A[0..H-1]$ (integer, last seen step) and $B[0..H-1]$ (float, EMA of inter-arrival times), global step counter $t$, hash function $h : \mathcal{Y} \to [H]$, and learning rate $\alpha$.

(b) **Worker update protocol.** Each worker processes one batch per step. Write pseudocode for the per-item update a worker performs for each item $y$ in its batch, including read, compute, and write steps. Note the non-atomic read-write and argue that race conditions only lose one sample rather than corrupting the estimate.

(c) **Inference.** After updating, the worker needs $\hat{p}(y) = 1/B[h(y)]$ immediately for the logQ correction. Describe a caching strategy: use the locally computed new value of $B[h(y)]$ (before sending the write to the server) to avoid a second round-trip. State the staleness of this estimate.

(d) **Initialization.** The note shows that $\delta_0 = \delta/(1-\alpha)$ makes the estimator unbiased from step one. Since $\delta$ is unknown, describe two practical initialization strategies: (i) initialize $B[k] = 1$ (assumes frequency = 1 per step); (ii) initialize $B[k] = M/B$ (uniform prior: each item appears on average every $M/B$ steps). Compare their initial bias.

---

### Problem 20: Multiple-Hashing Frequency Estimator

*Sketch Algorithm 3 (multiple hashings for collision-robust frequency estimation) including data structure layout, update rule, and inference.*

> **Prerequisites:** cf. note [[note#Algorithm 3: Multiple Hashings|§4 — Algorithm 3: Multiple Hashings]]; requires Problem 19

(a) **Data structure.** Maintain $m$ independent pairs of arrays $(A_i[0..H/m-1], B_i[0..H/m-1])$ with independent hash functions $h_i : \mathcal{Y} \to [H/m]$ for $i = 1, \ldots, m$. State total memory (in array entries) and confirm it equals the single-hashing baseline.

(b) **Update rule.** For each item $y$ in the batch at step $t$, update all $m$ arrays independently using the same rule as Algorithm 2. Write pseudocode. Explain why the $m$ updates are independent (no shared state across hash functions).

(c) **Inference.** Compute $\hat{p}(y) = 1/\max_i B_i[h_i(y)]$. Explain the directional argument: each $B_i[h_i(y)]$ underestimates $\delta_y$ due to collisions pulling the EMA downward, so the maximum over $m$ estimates is the least-biased. Relate this to the count-min sketch which takes the minimum of bucket counts (opposite direction because counts are over-estimated by collisions).

(d) **Complexity and tradeoffs.** State the time complexity of the update for a batch of $B$ items with $m$ hash functions and the time complexity of inference. For $m = 4$ and $H = 5000$, each array has $H/m = 1250$ buckets. Compute the expected collision probability for $M = 10^7$ items and compare to $m = 1$ with $H = 5000$ (using the formula from Problem 15(a)).

---

### Problem 21: Recall at K Evaluation Against a Full Corpus

*Sketch an exact and approximate evaluation pipeline for Recall@K that handles both single-positive and multi-positive settings.*

> **Prerequisites:** cf. note [[note#Wikipedia Link Prediction|§6 — Wikipedia Link Prediction]]; cf. note [[note#Indexing and Serving|§5 — Indexing and Serving]]

(a) **Embedding computation.** Describe how to compute query embeddings $\{u_i\}_{i=1}^N$ and item embeddings $\{v_j\}_{j=1}^M$ using the trained towers. Identify the computational bottleneck (hint: $M \gg N$) and explain why item embeddings are precomputed and cached. State the total memory for embedding storage.

(b) **Exact Recall@K.** For each query $i$ with positive item $y_i^+$: compute scores $\langle u_i, v_j \rangle$ for all $j$, find the rank of $y_i^+$, and compute $\mathrm{Hit}_i = \mathbf{1}[\mathrm{rank}(y_i^+) \leq K]$. State the time complexity in terms of $N$, $M$, $k$.

(c) **Approximate Recall@K via ANN.** Describe the ANN-based pipeline: build an index over $\{v_j\}$, retrieve approximate top-$K$ for each $u_i$, check whether $y_i^+$ is among the retrieved set. Define the ANN recall gap $= \mathrm{Recall@}K^{\mathrm{exact}} - \mathrm{Recall@}K^{\mathrm{ANN}}$ and name its two sources.

(d) **Multi-positive generalization.** In YouTube, a query may have a set of positive items $\mathcal{P}_i$. Define multi-positive Recall@$K$ as $\frac{1}{N}\sum_i \frac{|\mathcal{P}_i \cap \mathrm{top}_K(u_i)|}{|\mathcal{P}_i|}$. Write pseudocode. State when this reduces to the single-positive definition.

---

### Problem 22: Sequential Training with Online Frequency Updates

*Sketch the full end-to-end training loop of the YouTube system combining sequential data ingestion, online frequency estimation, and corrected batch softmax.*

> **Prerequisites:** cf. note [[note#Sequential Training|§5 — Sequential Training]]; cf. note [[note#Algorithm 2: Single Hash Estimation|§4 — Algorithm 2: Single Hash Estimation]]; cf. note [[note#Corrected Batch Loss|§3 — Corrected Batch Loss]]

(a) **Outer loop.** Describe the day-level training loop: iterate over days from oldest to most recent; within each day, iterate over mini-batches. State what happens once the model catches up to the current day.

(b) **Inner loop.** For each mini-batch of size $B$ at global step $t$: (i) extract item IDs from the batch; (ii) update the frequency estimator arrays for all items (Algorithm 2); (iii) read $\hat{p}(y_j)$ for each item; (iv) compute corrected logits $s^c_{ij} = s_{ij} - \log\hat{p}(y_j)$; (v) compute the corrected batch loss (Problem 18); (vi) backpropagate and update $\theta$. Write pseudocode for this inner loop.

(c) **Ordering constraint.** Explain why frequency estimator updates (step ii) must happen before logit correction (step iv) within the same batch. What would happen if the estimator were updated after the logit correction?

(d) **Complexity.** State the time complexity of one inner-loop iteration in terms of $B$, $k$ (embedding dimension), $m$ (number of hash functions), and $H$. Identify the dominant term and argue that the frequency estimation overhead is negligible compared to the forward/backward pass through the two towers.
