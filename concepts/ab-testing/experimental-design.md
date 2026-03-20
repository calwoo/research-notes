# A/B Testing: Experimental Design

## Table of Contents

1. [[#1. Randomization Mechanisms|Randomization Mechanisms]]
   - [[#1.1 Bernoulli Randomization|1.1 Bernoulli Randomization]]
   - [[#1.2 Complete Randomization|1.2 Complete Randomization]]
   - [[#1.3 Variance Comparison: Bernoulli vs. Complete|1.3 Variance Comparison: Bernoulli vs. Complete]]
   - [[#1.4 Hash-Based Assignment in Production Systems|1.4 Hash-Based Assignment in Production Systems]]
   - [[#1.5 Cluster Randomization|1.5 Cluster Randomization]]
2. [[#2. Stratification and Blocking|Stratification and Blocking]]
   - [[#2.1 Stratified Randomization|2.1 Stratified Randomization]]
   - [[#2.2 The Stratified ATE Estimator|2.2 The Stratified ATE Estimator]]
   - [[#2.3 Variance of the Stratified Estimator|2.3 Variance of the Stratified Estimator]]
   - [[#2.4 Stratification Never Increases Variance|2.4 Stratification Never Increases Variance]]
   - [[#2.5 The Ideal Stratification Variable|2.5 The Ideal Stratification Variable]]
3. [[#3. CUPED: Covariate-Adjusted Pre-Experiment Data|CUPED: Covariate-Adjusted Pre-Experiment Data]]
   - [[#3.1 Motivation and Setup|3.1 Motivation and Setup]]
   - [[#3.2 The Adjusted Outcome|3.2 The Adjusted Outcome]]
   - [[#3.3 Optimal Theta: Derivation|3.3 Optimal Theta: Derivation]]
   - [[#3.4 Variance Reduction in Terms of Correlation|3.4 Variance Reduction in Terms of Correlation]]
   - [[#3.5 Unbiasedness of the CUPED Estimator|3.5 Unbiasedness of the CUPED Estimator]]
   - [[#3.6 Connection to OLS and Frisch-Waugh-Lovell|3.6 Connection to OLS and Frisch-Waugh-Lovell]]
   - [[#3.7 Practical Caveats|3.7 Practical Caveats]]
4. [[#4. Multiple Testing: FWER Control|Multiple Testing: FWER Control]]
   - [[#4.1 The Multiple Comparisons Problem|4.1 The Multiple Comparisons Problem]]
   - [[#4.2 Family-Wise Error Rate|4.2 Family-Wise Error Rate]]
   - [[#4.3 Bonferroni Correction|4.3 Bonferroni Correction]]
   - [[#4.4 Holm-Bonferroni Procedure|4.4 Holm-Bonferroni Procedure]]
   - [[#4.5 FWER Control of Holm via Closed Testing|4.5 FWER Control of Holm via Closed Testing]]
5. [[#5. Multiple Testing: FDR Control|Multiple Testing: FDR Control]]
   - [[#5.1 False Discovery Rate|5.1 False Discovery Rate]]
   - [[#5.2 Benjamini-Hochberg Procedure|5.2 Benjamini-Hochberg Procedure]]
   - [[#5.3 FDR Control Level of BH under Independence|5.3 FDR Control Level of BH under Independence]]
   - [[#5.4 FDR vs. FWER: When to Use Each|5.4 FDR vs. FWER: When to Use Each]]
   - [[#5.5 Worked Example: Bonferroni, Holm, and BH|5.5 Worked Example: Bonferroni, Holm, and BH]]
6. [[#6. Metric Selection|Metric Selection]]
   - [[#6.1 Primary Metrics|6.1 Primary Metrics]]
   - [[#6.2 Guardrail Metrics|6.2 Guardrail Metrics]]
   - [[#6.3 Sensitivity and Signal-to-Noise Ratio|6.3 Sensitivity and Signal-to-Noise Ratio]]
   - [[#6.4 Directionality and Metric Hierarchies|6.4 Directionality and Metric Hierarchies]]
7. [[#References|References]]

---

## 1. Randomization Mechanisms

*Randomization* is the mechanism that transforms a controlled experiment from an observational study into one that admits causal identification. The choice of randomization scheme determines both the distribution of group sizes and the variance of the treatment effect estimator. We work throughout in the *potential outcomes* framework: each unit $i \in \{1, \ldots, N\}$ has potential outcomes $Y_i(1)$ and $Y_i(0)$, and treatment assignment $T_i \in \{0, 1\}$.

### 1.1 Bernoulli Randomization

**Definition (Bernoulli Randomization).** In *Bernoulli randomization*, each unit $i$ is assigned independently to treatment with probability $p \in (0,1)$:
$$T_i \overset{\text{i.i.d.}}{\sim} \text{Bernoulli}(p), \quad i = 1, \ldots, N.$$

The treatment group size $n_1 = \sum_{i=1}^N T_i$ is a random variable with distribution $n_1 \sim \text{Binomial}(N, p)$, so $\mathbb{E}[n_1] = Np$ and $\text{Var}(n_1) = Np(1-p)$. In particular, group sizes are not fixed — in any realized experiment, $n_1$ may deviate substantially from $Np$, especially for small $N$.

The difference-in-means estimator of the *average treatment effect* $\tau = \mathbb{E}[Y_i(1) - Y_i(0)]$ is:
$$\hat{\tau}_{\text{DM}} = \bar{Y}_1 - \bar{Y}_0 = \frac{1}{n_1}\sum_{i: T_i=1} Y_i - \frac{1}{n_0}\sum_{i: T_i=0} Y_i,$$
where $n_0 = N - n_1$.

### 1.2 Complete Randomization

**Definition (Complete Randomization).** In *complete randomization*, exactly $m$ of the $N$ units are assigned to treatment, chosen uniformly at random from all $\binom{N}{m}$ subsets of size $m$. The treatment assignment vector $\mathbf{T} = (T_1, \ldots, T_N)$ is drawn uniformly from the set $\{\mathbf{t} \in \{0,1\}^N : \sum_i t_i = m\}$.

Group sizes are now deterministic: $n_1 = m$ and $n_0 = N - m$ with probability one. This is a key structural difference from Bernoulli randomization.

### 1.3 Variance Comparison: Bernoulli vs. Complete

We derive the variance of $\hat{\tau}_{\text{DM}}$ under each scheme, following the Neyman repeated-sampling framework. Write $S_1^2 = \frac{1}{N-1}\sum_i (Y_i(1) - \bar{Y}(1))^2$ and $S_0^2 = \frac{1}{N-1}\sum_i (Y_i(0) - \bar{Y}(0))^2$ for the finite-population variances of the potential outcomes, and $S_{10}^2 = \frac{1}{N-1}\sum_i ((Y_i(1) - Y_i(0)) - \tau)^2$ for the variance of unit-level treatment effects.

**Proposition (Variance under Complete Randomization).** Under complete randomization with $n_1 = m$ units treated:
$$\text{Var}_{\text{CR}}(\hat{\tau}_{\text{DM}}) = \frac{S_1^2}{m} + \frac{S_0^2}{N-m} - \frac{S_{10}^2}{N}.$$

*Proof sketch.* The potential outcomes are fixed; randomness enters only through $\mathbf{T}$. Writing $\bar{Y}_1 = N^{-1}\sum_i T_i Y_i(1) \cdot (N/m)$ and expanding, one obtains the Neyman variance formula. The term $-S_{10}^2/N$ arises because units cannot simultaneously be in both groups; it represents the negative covariance between $\bar{Y}_1$ and $\bar{Y}_0$ induced by the fixed-sum constraint.

**Proposition (Variance under Bernoulli Randomization).** Conditioning on $n_1 > 0$ and $n_0 > 0$, and approximating for large $N$:
$$\text{Var}_{\text{Bern}}(\hat{\tau}_{\text{DM}}) \approx \frac{S_1^2}{Np} + \frac{S_0^2}{N(1-p)}.$$

The term $-S_{10}^2/N$ is absent because under Bernoulli randomization, $T_i$ and $T_j$ are independent, so there is no negative covariance between the group means arising from the assignment mechanism. **Complete randomization always has variance no greater than Bernoulli randomization** — the difference is exactly $S_{10}^2 / N \geq 0$.

### 1.4 Hash-Based Assignment in Production Systems

In production systems, true random assignment is often replaced by *deterministic hashing*. The assignment of user $u$ to experiment $e$ is computed as:
$$T_u = \mathbf{1}\!\left[h(u,\, e) \bmod 100 < \lfloor 100p \rfloor\right],$$
where $h : \mathcal{U} \times \mathcal{E} \to \{0, 1, \ldots, 2^k - 1\}$ is a hash function (e.g., MD5, MurmurHash). The experiment ID is incorporated into the hash input so that the same user receives independent-looking assignments across different experiments. This approach achieves two goals:

1. **Reproducibility:** repeated calls return the same assignment, enabling consistent user experience and post-hoc audit.
2. **Cross-experiment independence:** for two experiments $e_1 \neq e_2$, the joint distribution of $(T_u^{e_1}, T_u^{e_2})$ is approximately uniform over $\{0,1\}^2$, preventing spurious correlations between simultaneous experiments.

*This is a Bernoulli scheme, not complete randomization — group sizes are random.* However, for $N$ in the millions, the standard deviation of $n_1$ is $O(\sqrt{N})$, making the fractional deviation $O(N^{-1/2})$ negligible.

### 1.5 Cluster Randomization

When units in the same cluster can influence one another — violating the *stable unit treatment value assumption* (SUTVA) — treatment must be assigned at the cluster level. Let $\mathcal{C} = \{C_1, \ldots, C_K\}$ be a partition of units into $K$ clusters, with cluster $C_k$ containing $n_k$ units. Randomize at the cluster level: each cluster $C_k$ is assigned $T_k^{\text{cluster}} \in \{0,1\}$ independently.

The effective sample size for variance computation drops from $N$ to $K$. Formally, the variance of the cluster-randomized ATE estimator is of order $K^{-1}$ rather than $N^{-1}$, inflated by the *intracluster correlation* (ICC) $\rho_{\text{ICC}}$:
$$\text{Var}(\hat{\tau}_{\text{cluster}}) \approx \frac{\sigma^2}{N}\left[1 + (n_{\text{avg}} - 1)\rho_{\text{ICC}}\right],$$
where $n_{\text{avg}} = N/K$ is the average cluster size. The factor $[1 + (n_{\text{avg}} - 1)\rho_{\text{ICC}}]$ is the *design effect* — it quantifies how much larger the effective variance is relative to individual-level randomization. *Cluster randomization should be used only when interference genuinely threatens validity; the power loss can be severe.*

---

## 2. Stratification and Blocking

### 2.1 Stratified Randomization

**Definition (Stratified Randomization).** Partition the $N$ units into $L$ *strata* $S_1, \ldots, S_L$ defined by a pre-experiment covariate $X$ (or a discretization thereof), with $|S_\ell| = N_\ell$ and $\sum_\ell N_\ell = N$. Within each stratum $\ell$, apply complete randomization: assign exactly $m_\ell$ units to treatment, where $m_\ell / N_\ell = p$ (or as close as integer constraints allow).

This enforces *balance* on $X$: the empirical distribution of $X$ is identical in treatment and control by construction, not merely in expectation.

### 2.2 The Stratified ATE Estimator

Within stratum $\ell$, the difference-in-means estimator is:
$$\hat{\tau}_\ell = \bar{Y}_{1,\ell} - \bar{Y}_{0,\ell}.$$

The *stratified ATE estimator* combines these stratum-level estimates with population weights:
$$\hat{\tau}_{\text{strat}} = \sum_{\ell=1}^L w_\ell \hat{\tau}_\ell, \quad w_\ell = \frac{N_\ell}{N}.$$

This estimator is unbiased for $\tau = \mathbb{E}[Y_i(1) - Y_i(0)]$, since within each stratum $\mathbb{E}[\hat{\tau}_\ell] = \tau_\ell = \mathbb{E}[Y_i(1) - Y_i(0) \mid i \in S_\ell]$, and $\sum_\ell w_\ell \tau_\ell = \tau$.

### 2.3 Variance of the Stratified Estimator

Under complete randomization within each stratum, the variance of $\hat{\tau}_\ell$ is, by the Neyman formula, approximately $S_{1,\ell}^2 / m_\ell + S_{0,\ell}^2 / (N_\ell - m_\ell)$. Setting $m_\ell = pN_\ell$ and using independence across strata:

$$\text{Var}(\hat{\tau}_{\text{strat}}) = \sum_{\ell=1}^L w_\ell^2 \text{Var}(\hat{\tau}_\ell) \approx \sum_{\ell=1}^L \frac{N_\ell^2}{N^2} \left(\frac{S_{1,\ell}^2}{pN_\ell} + \frac{S_{0,\ell}^2}{(1-p)N_\ell}\right) = \frac{1}{N}\sum_{\ell=1}^L w_\ell \left(\frac{S_{1,\ell}^2}{p} + \frac{S_{0,\ell}^2}{1-p}\right).$$

### 2.4 Stratification Never Increases Variance

**Proposition.** $\text{Var}(\hat{\tau}_{\text{strat}}) \leq \text{Var}(\hat{\tau}_{\text{unstrat}})$.

*Proof sketch.* Let $\mu_{1,\ell} = \mathbb{E}[Y_i(1) \mid i \in S_\ell]$ and $\mu_1 = \mathbb{E}[Y_i(1)]$. By the law of total variance:
$$S_1^2 = \underbrace{\sum_\ell w_\ell S_{1,\ell}^2}_{\text{within-stratum variance}} + \underbrace{\sum_\ell w_\ell (\mu_{1,\ell} - \mu_1)^2}_{\text{between-stratum variance}} \geq \sum_\ell w_\ell S_{1,\ell}^2.$$

The unstratified variance (ignoring the $-S_{10}^2/N$ term for simplicity) is approximately $S_1^2/(pN) + S_0^2/((1-p)N)$. The stratified variance replaces $S_k^2$ with its within-stratum average $\sum_\ell w_\ell S_{k,\ell}^2 \leq S_k^2$. Hence $\text{Var}(\hat{\tau}_{\text{strat}}) \leq \text{Var}(\hat{\tau}_{\text{unstrat}})$, with equality if and only if the potential outcome means are constant across strata (i.e., $X$ is uninformative).

### 2.5 The Ideal Stratification Variable

The variance reduction from stratification is proportional to the *between-stratum variance* in potential outcomes. This is maximized when strata are as homogeneous as possible within and as different as possible between. The optimal stratification variable is therefore the pre-experiment value of the outcome itself, $Y_{i,\text{pre}}$. Units with similar pre-experiment outcomes have similar potential outcomes, so within-stratum variance is minimized. This observation is also the motivation for CUPED (Section 3), which achieves a similar variance reduction continuously rather than discretely.

---

## 3. CUPED: Covariate-Adjusted Pre-Experiment Data

### 3.1 Motivation and Setup

The *CUPED* (Controlled-experiment Using Pre-Experiment Data) method, introduced by Deng, Xu, Kohavi, and Walker (2013), reduces the variance of the ATE estimator by exploiting a pre-experiment covariate $X_i$ correlated with the outcome $Y_i$.

**Setup.** Let $Y_i$ be the outcome observed during the experiment (e.g., revenue per user in the experiment week), and let $X_i$ be a pre-experiment covariate observed before randomization (e.g., revenue per user in the week prior). Let $T_i \in \{0,1\}$ be the treatment indicator. Because $X_i$ is observed before randomization, it is independent of $T_i$:
$$X_i \perp T_i.$$

### 3.2 The Adjusted Outcome

**Definition (CUPED Adjusted Outcome).** For a scalar $\theta \in \mathbb{R}$, define:
$$Y_i^{\text{CUPED}}(\theta) = Y_i - \theta(X_i - \mathbb{E}[X_i]).$$

The centering by $\mathbb{E}[X_i]$ is conventional and does not affect the estimator (it only shifts the outcome by a constant). The CUPED ATE estimator is the difference-in-means applied to $Y_i^{\text{CUPED}}$:
$$\hat{\tau}^{\text{CUPED}} = \bar{Y}^{\text{CUPED}}_{1} - \bar{Y}^{\text{CUPED}}_{0}.$$

### 3.3 Optimal Theta: Derivation

We choose $\theta$ to minimize the variance of $Y_i^{\text{CUPED}}(\theta)$. By the variance of a linear combination:
$$\text{Var}(Y_i - \theta X_i) = \text{Var}(Y_i) - 2\theta\,\text{Cov}(Y_i, X_i) + \theta^2\,\text{Var}(X_i).$$

This is a quadratic in $\theta$, opening upward. Differentiating and setting to zero:
$$\frac{d}{d\theta}\text{Var}(Y_i - \theta X_i) = -2\,\text{Cov}(Y_i, X_i) + 2\theta\,\text{Var}(X_i) = 0,$$
$$\theta^* = \frac{\text{Cov}(Y_i, X_i)}{\text{Var}(X_i)}.$$

This is precisely the ordinary least squares (OLS) coefficient of regressing $Y$ on $X$. In practice, $\theta^*$ is estimated from the pooled sample (or from pre-experiment data) and treated as fixed; the resulting estimator remains consistent.

### 3.4 Variance Reduction in Terms of Correlation

Substituting $\theta^*$ back into the variance formula:
$$\text{Var}(Y_i^{\text{CUPED}}) = \text{Var}(Y_i) - \frac{[\text{Cov}(Y_i, X_i)]^2}{\text{Var}(X_i)} = \text{Var}(Y_i)\left(1 - \frac{[\text{Cov}(Y_i, X_i)]^2}{\text{Var}(Y_i)\,\text{Var}(X_i)}\right) = \text{Var}(Y_i)(1 - \rho^2),$$

where $\rho = \text{Corr}(Y_i, X_i)$. **The variance reduction factor is $\rho^2$: CUPED reduces outcome variance by a fraction $\rho^2$.** Since the variance of $\hat{\tau}^{\text{CUPED}}$ is proportional to $\text{Var}(Y_i^{\text{CUPED}})$, the same factor applies to the estimator variance. For $\rho = 0.6$, variance drops by $36\%$, equivalently reducing the required sample size by $36\%$ for fixed power.

### 3.5 Unbiasedness of the CUPED Estimator

**Proposition.** $\mathbb{E}[\hat{\tau}^{\text{CUPED}}] = \tau$.

*Proof.* Compute the conditional expectation of $Y_i^{\text{CUPED}}$ given treatment status:
$$\mathbb{E}[Y_i^{\text{CUPED}} \mid T_i = t] = \mathbb{E}[Y_i \mid T_i = t] - \theta^*(\mathbb{E}[X_i \mid T_i = t] - \mathbb{E}[X_i]).$$

Since $X_i \perp T_i$, we have $\mathbb{E}[X_i \mid T_i = t] = \mathbb{E}[X_i]$ for all $t$. Therefore:
$$\mathbb{E}[Y_i^{\text{CUPED}} \mid T_i = t] = \mathbb{E}[Y_i \mid T_i = t].$$

It follows that:
$$\hat{\tau}^{\text{CUPED}} = \bar{Y}_1^{\text{CUPED}} - \bar{Y}_0^{\text{CUPED}} \xrightarrow{p} \mathbb{E}[Y_i \mid T_i=1] - \mathbb{E}[Y_i \mid T_i=0] = \tau.\quad\square$$

The adjustment $\theta^* X_i$ cancels in the difference precisely because $X_i$ has the same conditional mean under treatment and control. *Using a post-experiment covariate would violate $X_i \perp T_i$ and introduce bias.*

### 3.6 Connection to OLS and Frisch-Waugh-Lovell

The CUPED estimator is algebraically equivalent to the OLS coefficient on $T$ in the regression:
$$Y_i = \alpha + \tau T_i + \beta X_i + \varepsilon_i.$$

This equivalence is a direct consequence of the *Frisch-Waugh-Lovell (FWL) theorem*. The FWL theorem states that the coefficient $\hat{\tau}$ in the above regression equals the coefficient from regressing the residuals of $Y$ on $X$ against the residuals of $T$ on $X$. Since $T \perp X$ by randomization, the projection of $T$ on $X$ has zero slope, so the residuals of $T$ on $X$ are simply $T_i - \bar{T}$. Thus, the FWL coefficient on $T$ reduces to regressing $Y_i - \hat{\beta}_{\text{OLS}} X_i$ on $T_i$, which is exactly $\hat{\tau}^{\text{CUPED}}$.

*Importantly*, the FWL theorem also implies that the standard error of $\hat{\tau}^{\text{CUPED}}$ from the joint regression is asymptotically the same as the standard error from the two-step CUPED procedure (up to a degrees-of-freedom correction that vanishes as $N \to \infty$). In practice, running the joint regression is the simplest implementation.

### 3.7 Practical Caveats

- $X_i$ must be measured before the experiment begins. Any covariate that could be affected by the treatment introduces *post-treatment bias*.
- $\theta^*$ is typically estimated from the pooled experimental data or from a holdout pre-period, and treated as a fixed constant in variance calculations. This introduces a small plug-in approximation error that is $O(N^{-1})$.
- CUPED can be combined with stratification: stratify on $X$ and apply CUPED within each stratum, though in practice CUPED alone often captures most of the variance reduction.

---

## 4. Multiple Testing: FWER Control

### 4.1 The Multiple Comparisons Problem

In a production experiment, it is common to test $m > 1$ hypotheses simultaneously — e.g., a primary metric and several secondary metrics, or a primary metric across multiple user segments. Even when each individual test controls Type I error at level $\alpha$, the probability of making *at least one* false rejection inflates rapidly with $m$.

**Under independence of the $m$ test statistics, all of which are null:**
$$P(\text{at least one false rejection}) = 1 - (1 - \alpha)^m.$$

For $m = 20$ and $\alpha = 0.05$: $1 - (0.95)^{20} \approx 0.64$. *An experiment testing 20 metrics at $\alpha = 0.05$ has roughly a 64% chance of producing at least one spurious significant result under the global null.*

### 4.2 Family-Wise Error Rate

**Definition (FWER).** Let $\mathcal{H}_0 \subseteq \{H_0^1, \ldots, H_0^m\}$ be the set of true null hypotheses, with $|\mathcal{H}_0| = m_0$. The *family-wise error rate* is:
$$\text{FWER} = P(V \geq 1),$$
where $V$ is the number of false rejections (Type I errors). A procedure *controls FWER at level $\alpha$* in the strong sense if $\text{FWER} \leq \alpha$ for any configuration of true and false nulls.

### 4.3 Bonferroni Correction

**Procedure (Bonferroni).** Reject $H_0^j$ if and only if $p_j \leq \alpha/m$.

**Theorem (Bonferroni controls FWER).** $\text{FWER} \leq \alpha$ under arbitrary dependence of the test statistics.

*Proof.* By the union bound:
$$\text{FWER} = P\!\left(\bigcup_{j \in \mathcal{H}_0} \{p_j \leq \alpha/m\}\right) \leq \sum_{j \in \mathcal{H}_0} P(p_j \leq \alpha/m) = m_0 \cdot \frac{\alpha}{m} \leq m \cdot \frac{\alpha}{m} = \alpha. \quad\square$$

The Bonferroni correction requires no assumption about the dependence structure of the tests. Its conservatism — using $m$ in the denominator rather than $m_0$ — comes from upper-bounding $m_0$ by $m$.

### 4.4 Holm-Bonferroni Procedure

**Procedure (Holm, 1979).** Order the $m$ p-values from smallest to largest: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$, with corresponding null hypotheses $H_0^{(1)}, H_0^{(2)}, \ldots, H_0^{(m)}$.

For $j = 1, 2, \ldots, m$:
- If $p_{(j)} \leq \alpha / (m - j + 1)$, reject $H_0^{(j)}$ and continue to $j+1$.
- Otherwise, retain $H_0^{(j)}, H_0^{(j+1)}, \ldots, H_0^{(m)}$ and stop.

At step $j$, the threshold is $\alpha/(m-j+1)$, which starts at $\alpha/m$ (same as Bonferroni for the smallest p-value) and grows as more hypotheses are rejected.

**Theorem (Holm controls FWER; Holm, 1979).** The Holm procedure controls FWER at level $\alpha$ under arbitrary dependence. Moreover, it is *uniformly more powerful* than Bonferroni: whenever Bonferroni rejects $H_0^j$, so does Holm.

*Proof of uniform dominance.* At step $j = 1$, the Holm threshold $\alpha/m$ equals the Bonferroni threshold. If $H_0^{(1)}$ is rejected, the Holm threshold for subsequent tests increases to $\alpha/(m-1) > \alpha/m$. Since both procedures compare the same ordered p-values to non-decreasing thresholds (Holm) versus a constant threshold (Bonferroni), every rejection by Bonferroni corresponds to a rejection by Holm. $\square$

### 4.5 FWER Control of Holm via Closed Testing

The *closed testing principle* (Marcus, Peritz, and Gabriel, 1976) provides the formal underpinning for Holm's FWER control. A *closed testing procedure* rejects $H_0^j$ only if all intersection hypotheses $H_0^J = \bigcap_{k \in J} H_0^k$ for every $J \supseteq \{j\}$ are also rejected at level $\alpha$.

For Holm, each intersection hypothesis $H_0^J$ ($|J| = q$) is tested using a Bonferroni test at level $q\alpha/m$: reject $H_0^J$ if $\min_{k \in J} p_k \leq q\alpha/m$, equivalently if $p_{(1)}^J \leq q\alpha/m$ where $p_{(1)}^J$ is the smallest p-value in $J$.

**FWER control.** Under the closed testing framework, any closed procedure controls FWER at level $\alpha$. The Holm step-down procedure is equivalent to this closed procedure, establishing its FWER control. The key step: if we make a false rejection of $H_0^j$, then by closure, the intersection $H_0^{\mathcal{H}_0}$ (the global null over all true nulls) was also rejected. The probability of this event is at most $\alpha$ by the validity of the Bonferroni test applied to the true nulls. Hence $\text{FWER} \leq \alpha$.

---

## 5. Multiple Testing: FDR Control

### 5.1 False Discovery Rate

When $m$ is large (e.g., testing hundreds of metrics or hundreds of user segments), controlling FWER becomes extremely conservative — the adjusted per-comparison threshold $\alpha/m$ may be so small that virtually no true effects are detected.

**Definition (FDR).** Let $R$ be the total number of rejections and $V$ the number of false rejections. The *false discovery rate* is:
$$\text{FDR} = \mathbb{E}\!\left[\frac{V}{\max(R, 1)}\right] = \mathbb{E}\!\left[\frac{V}{R} \,\Big|\, R > 0\right] P(R > 0).$$

FDR is the expected *proportion* of rejected hypotheses that are false positives, rather than the probability of any false positive. When all nulls are true, $V = R$, so FDR $= P(R > 0) = \text{FWER}$. When some nulls are false, FDR $\leq$ FWER.

### 5.2 Benjamini-Hochberg Procedure

**Procedure (Benjamini-Hochberg, 1995).** Order the $m$ p-values: $p_{(1)} \leq \cdots \leq p_{(m)}$. Define:
$$k^* = \max\!\left\{j \in \{1,\ldots,m\} : p_{(j)} \leq \frac{j\alpha}{m}\right\},$$
with the convention $k^* = 0$ if no such $j$ exists. Reject $H_0^{(1)}, \ldots, H_0^{(k^*)}$ (i.e., all hypotheses with p-value at most $p_{(k^*)}$).

The BH procedure compares each ordered p-value to a linearly increasing threshold $j\alpha/m$, and rejects everything up to the last crossing.

### 5.3 FDR Control Level of BH under Independence

**Theorem (Benjamini and Hochberg, 1995).** Under the assumption that the test statistics are mutually independent, the BH procedure controls FDR at level:
$$\text{FDR} \leq \frac{m_0}{m} \cdot \alpha \leq \alpha,$$
where $m_0$ is the number of true null hypotheses.

*Proof sketch.* Index the true null hypotheses as $j = 1, \ldots, m_0$ (with p-values uniform on $[0,1]$ by assumption) and false nulls as $j = m_0+1, \ldots, m$. Write the FDR as:
$$\text{FDR} = \mathbb{E}\!\left[\frac{V}{R}\right] = \sum_{j=1}^{m_0} \mathbb{E}\!\left[\frac{\mathbf{1}[p_j \leq p_{(k^*)}]}{R}\right].$$

For a fixed true null $j$, condition on all other p-values. The key observation is that for $p_j \sim \text{Uniform}(0,1)$, independent of the other p-values, the contribution of hypothesis $j$ to the FDR satisfies:
$$\mathbb{E}\!\left[\frac{\mathbf{1}[p_j \leq p_{(k^*)}]}{R}\right] \leq \frac{\alpha}{m}.$$

This follows because: if $j$ is the $\ell$-th smallest p-value among all $m$ p-values, then $j$ is rejected only if $p_j \leq \ell\alpha/m$, and there are at least $\ell$ rejections, so $V/R \leq 1/\ell$. Averaging over the rank $\ell$ of a uniform p-value:
$$\mathbb{E}\!\left[\frac{\mathbf{1}[H_0^j \text{ rejected}]}{R}\right] \leq \mathbb{E}\!\left[\frac{p_j \cdot m}{\alpha \cdot R}\right] \cdot \frac{\alpha}{m} \leq \frac{\alpha}{m}.$$

Summing over $m_0$ true nulls: $\text{FDR} \leq m_0 \alpha / m \leq \alpha$. $\square$

*This bound is tight: if all $m$ null hypotheses are true ($m_0 = m$), BH controls FDR at exactly $\alpha$, coinciding with FWER control in that case.*

### 5.4 FDR vs. FWER: When to Use Each

| Criterion | FWER | FDR |
|-----------|------|-----|
| Controls | $P(\text{any false positive})$ | $\mathbb{E}[\text{FP fraction among rejections}]$ |
| Appropriate when | Any false positive is unacceptable (safety, regulatory) | Exploring many hypotheses; cost of FP scales with discoveries |
| Power at large $m$ | Low (threshold $\sim \alpha/m$) | High (threshold $\sim j\alpha/m$, grows with rank) |
| Dependence requirement | Arbitrary (Bonferroni, Holm) | Independence (BH); extensions exist for dependence |

In A/B testing at scale, FWER control is typically applied to a small set of primary metrics (e.g., 2–5), while FDR control may be appropriate for exploratory analyses across hundreds of segments.

### 5.5 Worked Example: Bonferroni, Holm, and BH

**Setup.** $m = 10$ hypotheses, $\alpha = 0.05$. Observed p-values (already sorted):

| Rank $j$ | $p_{(j)}$ | Bonferroni threshold $\alpha/m$ | Holm threshold $\alpha/(m-j+1)$ | BH threshold $j\alpha/m$ |
|----------|-----------|----------------------------------|----------------------------------|--------------------------|
| 1 | 0.001 | 0.005 | 0.005 | 0.005 |
| 2 | 0.008 | 0.005 | 0.0056 | 0.010 |
| 3 | 0.012 | 0.005 | 0.0063 | 0.015 |
| 4 | 0.020 | 0.005 | 0.0071 | 0.020 |
| 5 | 0.035 | 0.005 | 0.0083 | 0.025 |
| 6 | 0.048 | 0.005 | 0.010 | 0.030 |
| 7 | 0.062 | 0.005 | 0.0125 | 0.035 |
| 8 | 0.200 | 0.005 | 0.0167 | 0.040 |
| 9 | 0.380 | 0.005 | 0.025 | 0.045 |
| 10 | 0.740 | 0.005 | 0.050 | 0.050 |

**Bonferroni:** Threshold $= 0.05/10 = 0.005$. Only $p_{(1)} = 0.001 \leq 0.005$. **Reject $H_0^{(1)}$ only.** (1 rejection)

**Holm:** Start at $j=1$: $p_{(1)} = 0.001 \leq 0.005$ — reject. $j=2$: $p_{(2)} = 0.008 \leq 0.0056$? No, $0.008 > 0.0056$ — stop. **Reject $H_0^{(1)}$ only.** (1 rejection)

**BH:** Find the largest $j$ such that $p_{(j)} \leq j\alpha/m$:
- $j=4$: $p_{(4)} = 0.020 \leq 0.020$ — yes.
- $j=5$: $p_{(5)} = 0.035 \leq 0.025$ — no.

So $k^* = 4$. **Reject $H_0^{(1)}, H_0^{(2)}, H_0^{(3)}, H_0^{(4)}$.** (4 rejections)

**Conclusion.** Bonferroni and Holm both yield 1 rejection. BH yields 4, demonstrating its substantially higher power at the cost of permitting a controlled expected false discovery rate rather than a zero-false-positive guarantee. *Note that Holm here gives the same result as Bonferroni; the gain from Holm is most pronounced when multiple p-values are small.*

---

## 6. Metric Selection

### 6.1 Primary Metrics

A *primary metric* is the main quantity the experiment is designed to move, directly tied to a business or product objective. Examples include revenue per user, click-through rate, and conversion rate. All statistical corrections (sample size, multiple testing) are designed with the primary metric in mind. *The number of primary metrics should be small* — one or two — to avoid diluting statistical power across the correction budget.

### 6.2 Guardrail Metrics

*Guardrail metrics* are metrics that must not degrade, even if the primary metric improves. Common examples: page load latency, crash rate, ad revenue (for product experiments), and customer support contact rate. They are tested one-tailed (degradation direction only) with a pre-specified degradation threshold. **If any guardrail metric fails its threshold, the experiment is declared invalid regardless of primary metric results.** This prevents Goodharting — optimizing a target metric at the expense of overall system health.

Formally, for guardrail metric $G$ with threshold $\delta_G > 0$ (maximum tolerable degradation), the guardrail test is:
$$H_0^G: \tau_G \geq -\delta_G \quad \text{vs.} \quad H_a^G: \tau_G < -\delta_G.$$

### 6.3 Sensitivity and Signal-to-Noise Ratio

The *sensitivity* of a metric is its ability to detect a true treatment effect. Formally, sensitivity is determined by the signal-to-noise ratio:
$$\text{SNR} = \frac{|\tau|}{\sigma_Y / \sqrt{n}},$$
where $\sigma_Y^2 = \text{Var}(Y_i)$ and $n$ is the per-group sample size. For fixed $n$ and effect size $\tau$, sensitivity increases as $\sigma_Y$ decreases. Strategies for reducing $\sigma_Y$:

- **Capping:** cap $Y_i$ at a high percentile (e.g., 99th) to remove outliers that dominate variance. This introduces a small bias but often substantially reduces variance.
- **Longer averaging windows:** if $Y_i = T^{-1}\sum_{t=1}^T Z_{i,t}$ is an average over $T$ time steps, $\text{Var}(Y_i) = O(T^{-1})$ under weak dependence.
- **CUPED:** subtract the pre-experiment covariate as in Section 3.

### 6.4 Directionality and Metric Hierarchies

Every metric must have a clear *directionality*: a definition of which direction of change is beneficial. This must be specified before the experiment, not inferred from data. Metrics with ambiguous directionality (e.g., session time — is more engagement good or are users struggling to find content?) should be decomposed into directional components (e.g., session time conditional on task completion vs. session time without completion).

The *Overall Evaluation Criterion* (OEC) is a single scalar score combining multiple metrics with fixed weights:
$$\text{OEC} = \sum_{k=1}^K \lambda_k \cdot \text{standardized}(M_k),$$
where $\lambda_k$ are predetermined weights summing to one. The OEC reduces the multiple testing problem to a single primary test, at the cost of making explicit value judgments about metric tradeoffs. *The choice of weights is not statistically determined and reflects organizational priorities.* Changes to the OEC formula should be versioned and tracked to ensure historical comparisons remain valid.

---

## References

| Reference Name | Brief Summary | Link to Reference |
|----------------|---------------|-------------------|
| Deng, Xu, Kohavi, Walker (2013) — "Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data" | Introduces CUPED: using pre-experiment covariates to reduce variance; derives the optimal adjustment and connects it to regression control variates | [ACM DL](https://dl.acm.org/doi/10.1145/2433396.2433413) |
| Benjamini and Hochberg (1995) — "Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing" | Introduces the FDR and the BH step-up procedure; proves FDR control at level $\alpha m_0/m$ under independence | [Wiley Online Library](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.2517-6161.1995.tb02031.x) |
| Holm (1979) — "A Simple Sequentially Rejective Multiple Test Procedure" | Introduces the Holm step-down procedure; proves strong FWER control under arbitrary dependence, dominates Bonferroni uniformly | [Scandinavian Journal of Statistics](https://www.ime.usp.br/~abe/lista/pdf4R8xPVzCnX.pdf) |
| Kohavi, Crook, Longbotham (2009) — "Online Experimentation at Microsoft" | Practical overview of large-scale A/B testing infrastructure, metric selection, and experimental pitfalls at Microsoft | [exp-platform.com](https://exp-platform.com/) |
| Kohavi, Tang, Xu (2020) — "Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing" | Comprehensive textbook on online experimentation: design, analysis, metrics, pitfalls, and organizational processes | [Cambridge University Press](https://www.cambridge.org/core/books/trustworthy-online-controlled-experiments/D97B26382EB0EB2DC2019A7A7B518F59) |
| Imbens and Rubin (2015) — "Causal Inference for Statistics, Social, and Biomedical Sciences" | Rigorous textbook treatment of the potential outcomes framework, complete randomization, stratification, and Neyman variance formulas | [Cambridge University Press](https://www.cambridge.org/core/books/causal-inference-for-statistics-social-and-biomedical-sciences/71126BE90C58F1A431FE9B2DD07938AB) |
| Frisch and Waugh (1933) — "Partial Time Regressions as Compared with Individual Trends" | Original paper proving the Frisch-Waugh theorem on partialling out regressors in OLS | [Econometrica](https://www.jstor.org/stable/1907330) |
| Marcus, Peritz, Gabriel (1976) — "On closed testing procedures with special reference to ordered analysis of variance" | Introduces the closed testing principle, the formal framework used to prove FWER control of Holm and other step-down procedures | [Biometrika](https://academic.oup.com/biomet/article-abstract/63/3/655/270440) |
