# A/B Testing: Frequentist Tests

## Table of Contents

1. [[#1. The Test Statistic Framework|1. The Test Statistic Framework]]
   - [[#1.1 Definition and Requirements|1.1 Definition and Requirements]]
   - [[#1.2 Sufficient Statistics|1.2 Sufficient Statistics]]
   - [[#1.3 Pivotal Quantities|1.3 Pivotal Quantities]]
   - [[#1.4 Rejection Regions and Critical Values|1.4 Rejection Regions and Critical Values]]
2. [[#2. Proportions: The Z-test|2. Proportions: The Z-test]]
   - [[#2.1 Setup and Null Hypothesis|2.1 Setup and Null Hypothesis]]
   - [[#2.2 The Pooled Z-statistic and Its Asymptotic Distribution|2.2 The Pooled Z-statistic and Its Asymptotic Distribution]]
   - [[#2.3 Unpooled Variance for Composite Nulls|2.3 Unpooled Variance for Composite Nulls]]
   - [[#2.4 Normal Approximation Validity|2.4 Normal Approximation Validity]]
3. [[#3. Continuous Metrics: The T-test|3. Continuous Metrics: The T-test]]
   - [[#3.1 Setup|3.1 Setup]]
   - [[#3.2 Student's T-statistic: Equal Variances|3.2 Student's T-statistic: Equal Variances]]
   - [[#3.3 Welch's T-test: Unequal Variances|3.3 Welch's T-test: Unequal Variances]]
   - [[#3.4 The Welch-Satterthwaite Degrees of Freedom|3.4 The Welch-Satterthwaite Degrees of Freedom]]
   - [[#3.5 Assumptions and Violations|3.5 Assumptions and Violations]]
4. [[#4. Confidence Intervals|4. Confidence Intervals]]
   - [[#4.1 Duality with Hypothesis Tests|4.1 Duality with Hypothesis Tests]]
   - [[#4.2 CI for the Difference in Proportions|4.2 CI for the Difference in Proportions]]
   - [[#4.3 Interpretation: Correct and Incorrect|4.3 Interpretation: Correct and Incorrect]]
   - [[#4.4 Width and the Precision-Cost Tradeoff|4.4 Width and the Precision-Cost Tradeoff]]
5. [[#5. Sample Size Formulas|5. Sample Size Formulas]]
   - [[#5.1 Derivation from the Power Equation|5.1 Derivation from the Power Equation]]
   - [[#5.2 Special Case: Proportions|5.2 Special Case: Proportions]]
   - [[#5.3 Unequal Allocation|5.3 Unequal Allocation]]
   - [[#5.4 Rule of Thumb|5.4 Rule of Thumb]]
6. [[#6. Multiple Variants: One-Way ANOVA|6. Multiple Variants: One-Way ANOVA]]
   - [[#6.1 Setup and Sum of Squares Decomposition|6.1 Setup and Sum of Squares Decomposition]]
   - [[#6.2 The F-statistic and Its Distribution|6.2 The F-statistic and Its Distribution]]
   - [[#6.3 Scope and Limitations|6.3 Scope and Limitations]]
7. [[#7. References|7. References]]

---

## 1. The Test Statistic Framework

### 1.1 Definition and Requirements

This note focuses on the mechanics of frequentist hypothesis tests used in A/B experiments. For the hypothesis framework, Type I/II error tradeoffs, p-values, power, and MDE, see `foundations.md`.

**Definition (Test Statistic).** A *test statistic* $W_n = W_n(X_1, \ldots, X_n)$ is a measurable function of the observed data whose distribution under the null hypothesis $H_0$ is known — either exactly or asymptotically as $n \to \infty$.

Three requirements make a test statistic useful:

1. **Known null distribution.** We must be able to compute $P_{H_0}(W_n > c)$ for any threshold $c$ in order to calibrate false positive rates.
2. **Sensitivity to the alternative.** Under $H_1$, the distribution of $W_n$ should shift in a detectable direction. The power of the test depends directly on this shift.
3. **Computability.** The statistic must be computable from observed data without knowledge of unknown parameters (nuisance parameters must be eliminated or estimated consistently).

### 1.2 Sufficient Statistics

**Definition (Sufficient Statistic).** A statistic $T(X_1, \ldots, X_n)$ is *sufficient* for a parameter $\theta$ if the conditional distribution of the data given $T$ does not depend on $\theta$. That is, $T$ captures all information in the data relevant to $\theta$.

The relevance to testing: by the *Fisher-Neyman factorization theorem*, the likelihood factors as $f(\mathbf{x} \mid \theta) = g(T(\mathbf{x}), \theta) \cdot h(\mathbf{x})$, so any optimal test — in the sense of maximizing power for a given size — should depend on the data only through $T$. More precisely, the Neyman-Pearson lemma ensures that a most powerful test of a simple null against a simple alternative is a function of the likelihood ratio, which for exponential families is a function of the sufficient statistic. Basing $W_n$ on the sufficient statistic is therefore not merely convenient; it is necessary for optimality.

For two Bernoulli samples of sizes $n_A$ and $n_B$, the sufficient statistic for $(p_A, p_B)$ is $(\sum X_i^A, \sum X_j^B) = (n_A \hat{p}_A,\, n_B \hat{p}_B)$. The Z-test below depends on the data only through these counts.

### 1.3 Pivotal Quantities

**Definition (Pivotal Quantity).** A test statistic $W_n$ is a *pivotal quantity* (or *pivot*) if its distribution under $H_0$ does not depend on any unknown nuisance parameters.

This matters practically: if the null distribution of $W_n$ depends on, say, the common variance $\sigma^2$ — which is unknown — then we cannot compute critical values without estimating $\sigma^2$ first, which introduces additional randomness. Pivots avoid this problem by construction.

The t-statistics discussed in Section 3 are exact pivots under normality: the unknown variance $\sigma^2$ cancels in the ratio. The Z-statistics in Section 2 are *asymptotic* pivots: they depend on unknown proportions, but consistently estimating those proportions and invoking Slutsky's theorem makes the limiting distribution standard normal regardless of the true $p$.

### 1.4 Rejection Regions and Critical Values

For a test at significance level $\alpha \in (0, 1)$, the *rejection region* for a one-sided upper-tail test is:

$$\mathcal{R} = \{W_n > c_\alpha\}$$

where the *critical value* $c_\alpha$ satisfies $P_{H_0}(W_n > c_\alpha) = \alpha$. For a two-sided test at level $\alpha$:

$$\mathcal{R} = \{|W_n| > c_{\alpha/2}\}$$

with $P_{H_0}(|W_n| > c_{\alpha/2}) = \alpha$. In A/B testing, two-sided tests are standard because we typically have no prior directional commitment (a new feature could hurt as well as help). The critical value $c_{\alpha/2} = z_{\alpha/2}$ when $W_n$ is asymptotically standard normal, and $c_{\alpha/2} = t_{\alpha/2, \nu}$ when $W_n$ follows a t-distribution with $\nu$ degrees of freedom.

---

## 2. Proportions: The Z-test

### 2.1 Setup and Null Hypothesis

Suppose we observe two independent samples of binary outcomes:

$$X_1^A, \ldots, X_{n_A}^A \overset{\text{iid}}{\sim} \text{Bernoulli}(p_A), \qquad X_1^B, \ldots, X_{n_B}^B \overset{\text{iid}}{\sim} \text{Bernoulli}(p_B)$$

The natural estimators of the group rates are the sample proportions:

$$\hat{p}_A = \frac{1}{n_A}\sum_{i=1}^{n_A} X_i^A, \qquad \hat{p}_B = \frac{1}{n_B}\sum_{j=1}^{n_B} X_j^B$$

The null and two-sided alternative are:

$$H_0: p_A = p_B \qquad \text{vs.} \qquad H_1: p_A \neq p_B$$

### 2.2 The Pooled Z-statistic and Its Asymptotic Distribution

**Why pool?** Under $H_0$, both groups share a common success probability $p := p_A = p_B$. The most efficient estimate of this common $p$ under $H_0$ is the *pooled proportion*, which combines both samples:

$$\hat{p} = \frac{n_A \hat{p}_A + n_B \hat{p}_B}{n_A + n_B} = \frac{\text{total successes}}{\text{total observations}}$$

Using $\hat{p}$ rather than group-specific estimates is not arbitrary: it is the maximum likelihood estimate of $p$ under the constraint $p_A = p_B$, and hence the most efficient estimator of the nuisance parameter under $H_0$.

**Definition (Pooled Z-statistic).**

$$Z = \frac{\hat{p}_B - \hat{p}_A}{\sqrt{\hat{p}(1-\hat{p})\left(\dfrac{1}{n_A}+\dfrac{1}{n_B}\right)}}$$

**Proposition.** Under $H_0: p_A = p_B = p$ and as $n_A, n_B \to \infty$ with $n_A/(n_A + n_B) \to \lambda \in (0,1)$,

$$Z \xrightarrow{d} \mathcal{N}(0, 1)$$

**Derivation.** We proceed in two steps.

*Step 1: CLT for the numerator.* The CLT is applied separately to each group: $\sqrt{n_A}(\hat{p}_A - p) \xrightarrow{d} \mathcal{N}(0, p(1-p))$ and $\sqrt{n_B}(\hat{p}_B - p) \xrightarrow{d} \mathcal{N}(0, p(1-p))$. Since $\hat{p}_A$ and $\hat{p}_B$ are independent, the difference $\hat{p}_B - \hat{p}_A$ satisfies:

$$\frac{\hat{p}_B - \hat{p}_A}{\sqrt{p(1-p)(1/n_A + 1/n_B)}} \xrightarrow{d} \mathcal{N}(0, 1)$$

This follows because $\text{Var}(\hat{p}_B - \hat{p}_A) = \text{Var}(\hat{p}_B) + \text{Var}(\hat{p}_A) = p(1-p)/n_B + p(1-p)/n_A$.

*Step 2: Replace true $p$ with $\hat{p}$ via Slutsky.* The denominator above contains the unknown $p$. We replace it with $\hat{p}$. Since $\hat{p}$ is a consistent estimator of $p$ under $H_0$ (it is the MLE), we have $\hat{p} \xrightarrow{p} p$, and by the continuous mapping theorem $\hat{p}(1-\hat{p}) \xrightarrow{p} p(1-p)$. Slutsky's theorem then guarantees that replacing the denominator's $p$ with the consistent estimator $\hat{p}$ does not change the limiting distribution:

$$Z = \frac{\hat{p}_B - \hat{p}_A}{\sqrt{p(1-p)(1/n_A + 1/n_B)}} \cdot \frac{\sqrt{p(1-p)(1/n_A+1/n_B)}}{\sqrt{\hat{p}(1-\hat{p})(1/n_A+1/n_B)}} \xrightarrow{d} \mathcal{N}(0,1) \cdot 1 = \mathcal{N}(0,1)$$

where the second factor converges in probability to 1. $\square$

### 2.3 Unpooled Variance for Composite Nulls

The pooled estimator is only valid under the simple null $H_0: p_A = p_B$. When testing a *shifted null* $H_0: p_B - p_A = \delta_0$ for some $\delta_0 \neq 0$, the null does not force a common probability, so pooling is invalid — it produces a biased estimator of the true variance.

In this case, we use the *unpooled* standard error, estimating each group's variance separately:

$$\text{SE}_{\text{unpooled}} = \sqrt{\frac{\hat{p}_A(1-\hat{p}_A)}{n_A} + \frac{\hat{p}_B(1-\hat{p}_B)}{n_B}}$$

This is simply the estimated standard deviation of $\hat{p}_B - \hat{p}_A$ when $p_A$ and $p_B$ are allowed to differ. The corresponding test statistic

$$Z_{\text{unpooled}} = \frac{(\hat{p}_B - \hat{p}_A) - \delta_0}{\text{SE}_{\text{unpooled}}}$$

is again asymptotically $\mathcal{N}(0,1)$ under $H_0: p_B - p_A = \delta_0$, by the same CLT-plus-Slutsky argument. As we will see in Section 4, confidence intervals for $p_B - p_A$ use the unpooled SE, not the pooled SE, because they invert the family of unpooled tests.

### 2.4 Normal Approximation Validity

The CLT applies in finite samples only approximately. A standard rule of thumb for adequate approximation quality is that both

$$n_A \hat{p}_A (1 - \hat{p}_A) \geq 10 \qquad \text{and} \qquad n_B \hat{p}_B(1-\hat{p}_B) \geq 10$$

This threshold ensures each group has at least 10 expected successes and 10 expected failures, providing a reasonable approximation to normality. The condition is most binding when $p$ is close to 0 or 1 (rare events), which is common in practice: click rates of $0.5\%$, conversion rates of $2\%$, etc.

When this condition fails — typically because $p$ is very small or sample sizes are limited — the normal approximation is unreliable and *Fisher's exact test* should be used instead. Fisher's test conditions on both marginal totals of the $2 \times 2$ contingency table and computes an exact p-value from the hypergeometric distribution, at the cost of being conservative (i.e., having power below the nominal level).

---

## 3. Continuous Metrics: The T-test

### 3.1 Setup

Suppose we observe two independent samples of a continuous outcome $Y$:

$$Y_1^A, \ldots, Y_{n_A}^A \overset{\text{iid}}{\sim} F_A \qquad \text{and} \qquad Y_1^B, \ldots, Y_{n_B}^B \overset{\text{iid}}{\sim} F_B$$

Let $\mu_A = \mathbb{E}[Y^A]$, $\mu_B = \mathbb{E}[Y^B]$, $\sigma_A^2 = \text{Var}(Y^A)$, $\sigma_B^2 = \text{Var}(Y^B)$. The null hypothesis is $H_0: \mu_A = \mu_B$ (equivalently $\mu_B - \mu_A = 0$). The sample means and variances are:

$$\bar{Y}_A = \frac{1}{n_A}\sum_{i=1}^{n_A} Y_i^A, \qquad s_A^2 = \frac{1}{n_A - 1}\sum_{i=1}^{n_A}(Y_i^A - \bar{Y}_A)^2$$

and analogously for group B.

### 3.2 Student's T-statistic: Equal Variances

When we assume $\sigma_A^2 = \sigma_B^2 =: \sigma^2$ (homoskedasticity) and $F_A, F_B$ are both normal, we can construct an exact pivot. The two sample variances $s_A^2$ and $s_B^2$ each estimate $\sigma^2$, so the most efficient combined estimate is the *pooled sample variance*, which is a weighted average:

$$s_p^2 = \frac{(n_A - 1)s_A^2 + (n_B - 1)s_B^2}{n_A + n_B - 2}$$

The weights $(n_k - 1)$ are the degrees of freedom contributed by each group to the within-group sum of squares. Under normality, $(n_A - 1)s_A^2/\sigma^2 \sim \chi^2_{n_A - 1}$ and $(n_B - 1)s_B^2/\sigma^2 \sim \chi^2_{n_B - 1}$, independently, so their sum $(n_A + n_B - 2)s_p^2/\sigma^2 \sim \chi^2_{n_A + n_B - 2}$.

**Definition (Student's T-statistic).**

$$t = \frac{\bar{Y}_B - \bar{Y}_A}{s_p\sqrt{1/n_A + 1/n_B}}$$

**Proposition.** Under normality, equal variances, and $H_0: \mu_A = \mu_B$,

$$t \sim t_{n_A + n_B - 2}$$

**Proof sketch.** The numerator $\bar{Y}_B - \bar{Y}_A \sim \mathcal{N}(0, \sigma^2(1/n_A + 1/n_B))$ under $H_0$, because both group means are normal and independent. Dividing by $\sigma\sqrt{1/n_A + 1/n_B}$ gives a standard normal $Z$. The denominator $s_p\sqrt{1/n_A + 1/n_B}$ equals $\sigma\sqrt{1/n_A + 1/n_B}$ times a factor $s_p / \sigma$, where $(n_A + n_B - 2)(s_p/\sigma)^2 \sim \chi^2_{n_A + n_B - 2}$. Since $Z$ and this $\chi^2$ are independent (by the independence of the sample mean and sample variance in a normal population — Cochran's theorem),

$$t = \frac{Z}{\sqrt{\chi^2_{n_A+n_B-2}/(n_A+n_B-2)}} \sim t_{n_A+n_B-2}$$

by the definition of the t-distribution. $\square$

### 3.3 Welch's T-test: Unequal Variances

When $\sigma_A^2 \neq \sigma_B^2$, pooling is no longer valid — pooling imposes a constraint (equal variances) that is false under the alternative model, and the resulting statistic is no longer pivotal. Instead we estimate each variance separately.

**Definition (Welch's T-statistic).**

$$t_W = \frac{\bar{Y}_B - \bar{Y}_A}{\sqrt{s_A^2/n_A + s_B^2/n_B}}$$

The denominator is a consistent estimator of $\sqrt{\text{Var}(\bar{Y}_B - \bar{Y}_A)} = \sqrt{\sigma_A^2/n_A + \sigma_B^2/n_B}$ because $s_k^2 \xrightarrow{p} \sigma_k^2$ for each group. Under $H_0$ and normality, the numerator is $\mathcal{N}(0, \sigma_A^2/n_A + \sigma_B^2/n_B)$. Under these conditions, $t_W$ is approximately t-distributed, but the exact degrees of freedom depends on the variance ratio $\sigma_A^2/\sigma_B^2$, which is unknown.

*Surprisingly,* Welch's t-test is nearly uniformly preferable to Student's t-test, even when variances truly are equal. The efficiency loss from not pooling is negligible for moderate sample sizes, while the robustness gain against variance heterogeneity is substantial. In practice, always use Welch's test by default.

### 3.4 The Welch-Satterthwaite Degrees of Freedom

The key distributional problem for $t_W$ is that its denominator is a sum of two random terms, $s_A^2/n_A$ and $s_B^2/n_B$, which follow scaled chi-squared distributions with different degrees of freedom. We need an effective degrees of freedom $\nu$ such that $t_W \approx t_\nu$.

**The moment matching argument.** Define $U = s_A^2/n_A$ and $V = s_B^2/n_B$. Under normality, $(n_A - 1)s_A^2/\sigma_A^2 \sim \chi^2_{n_A-1}$, so $U = \sigma_A^2 \chi^2_{n_A-1} / (n_A(n_A-1))$. Similarly for $V$. Thus:

$$\mathbb{E}[U] = \frac{\sigma_A^2}{n_A}, \qquad \text{Var}(U) = \frac{2\sigma_A^4}{n_A^2(n_A - 1)}$$

and analogously for $V$. We want to find $\nu$ such that $W := U + V$ is well approximated by $\frac{\sigma_W^2}{d}\chi^2_d$ for some $d = \nu$ and $\sigma_W^2 = \mathbb{E}[W]$. Matching the first two moments of $W$ to those of $c \cdot \chi^2_\nu$:

- $\mathbb{E}[W] = \mathbb{E}[c\chi^2_\nu] \Rightarrow c\nu = \mathbb{E}[U] + \mathbb{E}[V]$
- $\text{Var}(W) = \text{Var}(c\chi^2_\nu) \Rightarrow 2c^2\nu = \text{Var}(U) + \text{Var}(V)$ (by independence)

Dividing the second equation by the square of the first:

$$\frac{2c^2\nu}{c^2\nu^2} = \frac{\text{Var}(U) + \text{Var}(V)}{(\mathbb{E}[U] + \mathbb{E}[V])^2} \implies \frac{2}{\nu} = \frac{\text{Var}(W)}{\mathbb{E}[W]^2}$$

Solving for $\nu$ and substituting the expressions for $\text{Var}(U)$ and $\text{Var}(V)$:

$$\nu \approx \frac{(s_A^2/n_A + s_B^2/n_B)^2}{\dfrac{(s_A^2/n_A)^2}{n_A - 1} + \dfrac{(s_B^2/n_B)^2}{n_B - 1}}$$

This is the *Welch-Satterthwaite* effective degrees of freedom. We take the floor $\lfloor\nu\rfloor$ and use quantiles from $t_{\lfloor\nu\rfloor}$. Note that $\nu$ lies between $\min(n_A, n_B) - 1$ and $n_A + n_B - 2$, with the upper bound achieved when $\sigma_A^2/n_A = \sigma_B^2/n_B$ (the equal-variance case).

### 3.5 Assumptions and Violations

| Assumption | Consequence of violation | Remedy |
|---|---|---|
| Normality of $Y$ | Exact t-distribution fails | CLT saves us for $n \gtrsim 30$; for heavy tails use bootstrap |
| Independence within and across groups | Variance is underestimated, Type I error inflates | Use cluster-robust SE if units are clustered |
| Homoskedasticity ($\sigma_A^2 = \sigma_B^2$) | Student's t is invalid; pooled SE is biased | Use Welch's t-test |

The normality assumption is the least concerning in A/B testing: by the CLT, $\bar{Y}_A$ and $\bar{Y}_B$ are approximately normal for large $n$, so the numerator of $t_W$ is approximately normal regardless of $F_A, F_B$. *The CLT typically kicks in around $n \geq 30$ for moderately skewed distributions, but for highly skewed metrics like revenue-per-user, $n$ in the hundreds may be necessary.*

The independence assumption is more fragile. In practice, users in the same geographic cluster, or who interact socially, can produce correlated outcomes. Violating independence inflates the effective sample size, underestimates the standard error, and inflates Type I error. Clustering corrections (e.g., the delta method with cluster-robust variance estimators) are necessary in such settings.

---

## 4. Confidence Intervals

### 4.1 Duality with Hypothesis Tests

The *test-inversion* principle establishes a formal equivalence between hypothesis tests and confidence intervals. Given a family of level-$\alpha$ tests $\{\phi_{\delta_0}\}$ indexed by null value $\delta_0$, the corresponding $1-\alpha$ confidence interval for $\tau = \mu_B - \mu_A$ is:

$$\text{CI}_{1-\alpha} = \{\delta_0 : \phi_{\delta_0} \text{ does not reject}\}$$

This is not a computational convenience but a logical identity: the CI contains precisely those parameter values that would not be rejected as null hypotheses by the test.

For the two-sample z-test with asymptotically normal $\hat{\tau} = \bar{Y}_B - \bar{Y}_A$ and standard error $\text{SE}(\hat{\tau})$, the test at level $\alpha$ rejects $H_0: \tau = \delta_0$ when $|(\hat{\tau} - \delta_0)/\text{SE}| > z_{\alpha/2}$. Inverting this condition — finding all $\delta_0$ for which the test does not reject — yields:

$$\text{CI}_{1-\alpha} = \left[\hat{\tau} - z_{\alpha/2} \cdot \text{SE},\; \hat{\tau} + z_{\alpha/2} \cdot \text{SE}\right]$$

### 4.2 CI for the Difference in Proportions

For the difference $\hat{p}_B - \hat{p}_A$, the standard error to use in a confidence interval is the *unpooled* SE. This is because the CI inverts the family of unpooled tests: for each $\delta_0$, the relevant test of $H_0: p_B - p_A = \delta_0$ does not pool (since the null does not force $p_A = p_B$ unless $\delta_0 = 0$). Using the pooled SE in a CI would be internally inconsistent — pooling is only valid at the single point $\delta_0 = 0$, not across the range of $\delta_0$ values whose inclusion in the CI we are checking.

The resulting *Wald confidence interval* for the difference in proportions is:

$$(\hat{p}_B - \hat{p}_A) \pm z_{\alpha/2}\sqrt{\frac{\hat{p}_A(1-\hat{p}_A)}{n_A} + \frac{\hat{p}_B(1-\hat{p}_B)}{n_B}}$$

*Caveat: the Wald interval can undercover when $p$ is near 0 or 1 or when sample sizes are small. The Wilson score interval or the Agresti-Caffo interval for differences offer better coverage in those regimes.*

### 4.3 Interpretation: Correct and Incorrect

**Correct interpretation.** The confidence interval is a statement about the procedure, not about a single realized interval. If we were to repeat the experiment many times, $100(1-\alpha)\%$ of the constructed intervals would contain the true $\tau$. The true $\tau$ is fixed (it is not random); the interval endpoints are random because they depend on the data.

**Incorrect interpretation.** "There is a $95\%$ probability that the true effect lies in $[L, U]$" — this is a Bayesian posterior probability statement, not a frequentist confidence statement. Once the data are observed and $[L, U]$ is computed, the interval either does or does not contain $\tau$; no probability is involved. The $95\%$ refers to the long-run coverage frequency of the procedure, not to a posterior.

### 4.4 Width and the Precision-Cost Tradeoff

The width of the confidence interval is:

$$\text{Width} = 2 z_{\alpha/2} \cdot \text{SE} = 2 z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$

(in the equal-variance, equal-allocation case). This shows:

- **Width decreases as $1/\sqrt{n}$.** Halving the width requires quadrupling the sample size. This is the fundamental precision-cost tradeoff of frequentist inference.
- **Width increases with $z_{\alpha/2}$.** A 99% CI is wider than a 95% CI: requiring more coverage demands more uncertainty. Specifically, $z_{0.005} \approx 2.576$ vs. $z_{0.025} \approx 1.960$.
- **Width is proportional to $\sigma$.** High-variance metrics (e.g., revenue per user) require larger samples to achieve the same precision as lower-variance metrics (e.g., binary conversion).

**The CI width equals twice the margin of error, which in turn equals the MDE at the threshold power level** (see `foundations.md`, Section 7 for the MDE derivation).

---

## 5. Sample Size Formulas

### 5.1 Derivation from the Power Equation

We derive the required per-group sample size for a two-tailed z-test with equal allocation $n_A = n_B = n$. Let $\delta = \mu_B - \mu_A$ denote the true effect under the alternative, and let $\sigma_A^2, \sigma_B^2$ be the group variances. Denote $\hat{\tau} = \bar{Y}_B - \bar{Y}_A$.

**Step 1: Distribution of the test statistic under the alternative.** Under $H_1: \mu_B - \mu_A = \delta$, the test statistic

$$Z = \frac{\hat{\tau}}{\sqrt{\sigma_A^2/n + \sigma_B^2/n}}$$

follows (asymptotically) a normal distribution with mean

$$\mu_Z = \frac{\delta}{\sqrt{(\sigma_A^2 + \sigma_B^2)/n}} = \frac{\delta\sqrt{n}}{\sqrt{\sigma_A^2 + \sigma_B^2}}$$

and variance 1. So $Z \sim \mathcal{N}(\mu_Z, 1)$ under $H_1$.

**Step 2: Express power in terms of $n$.** The test rejects $H_0$ when $|Z| > z_{\alpha/2}$. Power is $P_{H_1}(|Z| > z_{\alpha/2})$. Because $\mu_Z > 0$ for $\delta > 0$, the dominant rejection event is $Z > z_{\alpha/2}$, and the lower-tail contribution $P(Z < -z_{\alpha/2})$ is negligible for $\delta > 0$ (an $O(\Phi(-2z_{\alpha/2}))$ error). *This approximation is accurate when $\delta$ is not too small relative to the noise; for very small effects (near the MDE boundary), the omitted term can be non-negligible and the formula underestimates the true required $n$.* The power is approximately:

$$1 - \beta \approx P\!\left(Z > z_{\alpha/2} \mid Z \sim \mathcal{N}(\mu_Z, 1)\right) = P\!\left(\mathcal{N}(0,1) > z_{\alpha/2} - \mu_Z\right) = \Phi\!\left(\mu_Z - z_{\alpha/2}\right)$$

**Step 3: Set power equal to $1 - \beta$ and solve.** Require $\Phi(\mu_Z - z_{\alpha/2}) = 1 - \beta$. Since $\Phi(z_\beta) = 1 - \beta$, we need:

$$\mu_Z - z_{\alpha/2} = z_\beta \implies \mu_Z = z_{\alpha/2} + z_\beta$$

Substituting $\mu_Z = \delta\sqrt{n}/\sqrt{\sigma_A^2 + \sigma_B^2}$:

$$\frac{\delta\sqrt{n}}{\sqrt{\sigma_A^2 + \sigma_B^2}} = z_{\alpha/2} + z_\beta$$

Squaring both sides and solving for $n$:

$$n = \frac{(z_{\alpha/2} + z_\beta)^2(\sigma_A^2 + \sigma_B^2)}{\delta^2}$$

**Step 4: Apply ceiling.** Since $n$ must be an integer, the required per-group size is $\lceil n \rceil$.

**The two key drivers are the effect size $\delta$ (in the denominator, quadratically) and the sum of variances (in the numerator). Doubling the MDE reduces the required sample size by a factor of four.**

### 5.2 Special Case: Proportions

For binary outcomes, $\sigma_A^2 = p_A(1-p_A)$ and $\sigma_B^2 = p_B(1-p_B)$. Substituting:

$$n = \frac{(z_{\alpha/2} + z_\beta)^2[p_A(1-p_A) + p_B(1-p_B)]}{\delta^2}$$

where $\delta = p_B - p_A$. In terms of relative lift $\ell = \delta/p_A = (p_B - p_A)/p_A$, so $p_B = p_A(1+\ell)$ and $\delta = p_A \ell$:

$$n = \frac{(z_{\alpha/2} + z_\beta)^2[p_A(1-p_A) + p_A(1+\ell)(1 - p_A(1+\ell))]}{p_A^2 \ell^2}$$

This highlights that for rare events ($p_A \ll 1$), both variances are approximately $p_A$ and $p_A(1+\ell)$, so $n \propto p_A^{-1}\ell^{-2}$: rarer base rates and smaller relative lifts require dramatically larger samples.

### 5.3 Unequal Allocation

Let $n_B = k \cdot n_A$ for some ratio $k > 0$. The variance of $\hat{\tau}$ is:

$$\text{Var}(\hat{\tau}) = \frac{\sigma_A^2}{n_A} + \frac{\sigma_B^2}{k n_A} = \frac{1}{n_A}\left(\sigma_A^2 + \frac{\sigma_B^2}{k}\right)$$

For fixed total $N = n_A + n_B = n_A(1+k)$, we have $n_A = N/(1+k)$, so:

$$\text{Var}(\hat{\tau}) = \frac{1+k}{N}\left(\sigma_A^2 + \frac{\sigma_B^2}{k}\right)$$

To minimize this over $k$, differentiate with respect to $k$ and set to zero:

$$\frac{d}{dk}\left[(1+k)\left(\sigma_A^2 + \frac{\sigma_B^2}{k}\right)\right] = \sigma_A^2 + \frac{\sigma_B^2}{k} - \frac{\sigma_B^2(1+k)}{k^2} = 0$$

Solving: $\sigma_A^2 k^2 = \sigma_B^2$, hence:

$$k^* = \frac{\sigma_B}{\sigma_A}$$

**The optimal allocation ratio equals the ratio of standard deviations.** If group B has higher variance, it should receive more units. For equal variances, $k^* = 1$ (equal allocation) is optimal. *In practice, unequal allocation is used when one group is more costly to measure, or when exposing users to a risky treatment variant should be minimized (underpowering the treatment arm is acceptable in early-phase tests).*

### 5.4 Rule of Thumb

For standard parameters — two-tailed test at $\alpha = 0.05$ (so $z_{0.025} \approx 1.96$), power $= 80\%$ (so $z_{0.20} \approx 0.84$), and equal variances $\sigma_A^2 = \sigma_B^2 = \sigma^2$ — the formula gives:

$$n = \frac{(1.96 + 0.84)^2 \cdot 2\sigma^2}{\delta^2} = \frac{7.84 \cdot 2\sigma^2}{\delta^2} \approx \frac{15.68\,\sigma^2}{\delta^2}$$

**Rule of thumb: $n \approx 16\sigma^2/\delta^2$ per group** for a two-tailed 5%-level test at 80% power with equal variances.

For 90% power ($z_{0.10} \approx 1.28$):

$$n \approx \frac{(1.96 + 1.28)^2 \cdot 2\sigma^2}{\delta^2} \approx \frac{21\,\sigma^2}{\delta^2}$$

The increase from $16\sigma^2/\delta^2$ to $21\sigma^2/\delta^2$ reflects the cost of the extra 10 percentage points of power.

---

## 6. Multiple Variants: One-Way ANOVA

### 6.1 Setup and Sum of Squares Decomposition

Suppose we run an A/B/n test with $K \geq 3$ variants. Let $n_k$ denote the number of units in group $k$, and $N = \sum_{k=1}^K n_k$ the total. The outcome for unit $i$ in group $k$ is $Y_{ik}$. The group means and grand mean are:

$$\bar{Y}_k = \frac{1}{n_k}\sum_{i=1}^{n_k} Y_{ik}, \qquad \bar{Y} = \frac{1}{N}\sum_{k=1}^K \sum_{i=1}^{n_k} Y_{ik} = \frac{\sum_k n_k \bar{Y}_k}{N}$$

The null hypothesis is the *omnibus null*: $H_0: \mu_1 = \mu_2 = \cdots = \mu_K$.

The one-way ANOVA framework is built on a decomposition of the *total sum of squares* ($\text{SST}$):

$$\text{SST} = \sum_{k=1}^K\sum_{i=1}^{n_k}(Y_{ik} - \bar{Y})^2$$

Write $Y_{ik} - \bar{Y} = (Y_{ik} - \bar{Y}_k) + (\bar{Y}_k - \bar{Y})$. Squaring and summing:

$$\text{SST} = \underbrace{\sum_{k=1}^K\sum_{i=1}^{n_k}(Y_{ik} - \bar{Y}_k)^2}_{\text{SSW}} + \underbrace{\sum_{k=1}^K n_k(\bar{Y}_k - \bar{Y})^2}_{\text{SSB}} + 2\underbrace{\sum_{k=1}^K(\bar{Y}_k - \bar{Y})\sum_{i=1}^{n_k}(Y_{ik}-\bar{Y}_k)}_{ = 0}$$

The cross-term vanishes because $\sum_{i=1}^{n_k}(Y_{ik} - \bar{Y}_k) = 0$ for each $k$. Therefore:

$$\text{SST} = \text{SSB} + \text{SSW}$$

where:

- $\text{SSB} = \sum_{k=1}^K n_k(\bar{Y}_k - \bar{Y})^2$ is the *between-group sum of squares*, measuring how much group means vary around the grand mean.
- $\text{SSW} = \sum_{k=1}^K\sum_{i=1}^{n_k}(Y_{ik} - \bar{Y}_k)^2$ is the *within-group sum of squares*, measuring variation within each group.

### 6.2 The F-statistic and Its Distribution

The *mean squares* normalize each sum of squares by its degrees of freedom:

$$\text{MSB} = \frac{\text{SSB}}{K-1}, \qquad \text{MSW} = \frac{\text{SSW}}{N-K}$$

The degrees of freedom $K-1$ for SSB reflects that $K$ group means are constrained to one relation (they sum to $N\bar{Y}$), so there are $K-1$ free contrasts. The degrees of freedom $N-K$ for SSW reflects that within each group $k$, we lose 1 degree of freedom for estimating $\bar{Y}_k$, leaving $n_k - 1$ per group, totaling $\sum_k(n_k - 1) = N - K$.

**Definition (F-statistic).**

$$F = \frac{\text{MSB}}{\text{MSW}} = \frac{\text{SSB}/(K-1)}{\text{SSW}/(N-K)}$$

**Proposition.** Under $H_0: \mu_1 = \cdots = \mu_K$, normality, and homoskedasticity ($\sigma_k^2 = \sigma^2$ for all $k$):

$$F \sim F(K-1,\, N-K)$$

**Conceptual derivation.** The $F$-distribution with $(d_1, d_2)$ degrees of freedom is defined as the ratio of two independent chi-squared random variables, each divided by its degrees of freedom: $F = (\chi^2_{d_1}/d_1) / (\chi^2_{d_2}/d_2)$.

Under $H_0$ and normality, both MSB and MSW are unbiased estimators of $\sigma^2$:

- $\mathbb{E}[\text{MSW}] = \sigma^2$ always (regardless of whether $H_0$ holds), because $\text{SSW}/\sigma^2 \sim \chi^2_{N-K}$ — within each group, $(n_k-1)s_k^2/\sigma^2 \sim \chi^2_{n_k-1}$, and these are independent across groups.
- $\mathbb{E}[\text{MSB}] = \sigma^2$ only under $H_0$. This follows because each group mean $\bar{Y}_k \sim \mathcal{N}(\mu_k, \sigma^2/n_k)$, so under $H_0$, the group means are i.i.d. $\mathcal{N}(\mu, \sigma^2/n_k)$ (up to the weighting), and $\text{SSB}/\sigma^2 \sim \chi^2_{K-1}$ by a similar argument. *The chi-squared distribution for $\text{SSB}/\sigma^2$ is exact for balanced designs ($n_k = n$ for all $k$); unbalanced designs require the general Cochran theorem.*

Furthermore, SSB and SSW are independent under normality (this follows from Cochran's theorem on the independence of quadratic forms in normal random vectors). Therefore, under $H_0$:

$$F = \frac{\text{SSB}/(\sigma^2(K-1))}{\text{SSW}/(\sigma^2(N-K))} = \frac{\chi^2_{K-1}/(K-1)}{\chi^2_{N-K}/(N-K)} \sim F(K-1, N-K)$$

Under $H_1$ (at least one $\mu_k$ differs), $\mathbb{E}[\text{MSB}] > \sigma^2$, so $F$ tends to be inflated, and the rejection region is $\{F > F_{\alpha, K-1, N-K}\}$ (one-sided upper tail only, since only a large $F$ is evidence against $H_0$).

### 6.3 Scope and Limitations

ANOVA tests the omnibus null $H_0: \mu_1 = \cdots = \mu_K$. Rejecting it tells us only that at least one group differs from the others — it does not identify which pairs differ or by how much. Pairwise comparisons after a significant ANOVA require *post-hoc correction* for multiple testing (e.g., Bonferroni, Tukey's HSD, Holm-Bonferroni) to control the familywise error rate; see `experimental-design.md` for details.

Homoskedasticity ($\sigma_1^2 = \cdots = \sigma_K^2$) can be tested with Levene's test before running ANOVA. When it fails, *Welch's one-way ANOVA* — which generalizes Welch's two-sample t-test to $K$ groups — provides a robust alternative, using a modified F-statistic with Satterthwaite-adjusted degrees of freedom.

*ANOVA also requires independence of observations across all $NK$ units, not just within groups. Clustering, time effects, or carry-over between test variants all violate this assumption and require mixed-effects models or stratified analysis.*

---

## 7. References

| Reference Name | Brief Summary | Link to Reference |
|---|---|---|
| Casella & Berger, *Statistical Inference* (2nd ed.) | Graduate-level reference for sufficient statistics, pivotal quantities, UMP tests, and the theory of test inversion for confidence intervals | [Routledge](https://www.routledge.com/Statistical-Inference/Casella-Berger/p/book/9781032593036) |
| Wikipedia: Welch-Satterthwaite equation | Statement of the Satterthwaite approximation and moment-matching derivation for effective degrees of freedom | [Wikipedia](https://en.wikipedia.org/wiki/Welch%E2%80%93Satterthwaite_equation) |
| Wikipedia: Two-proportion Z-test | Derivation of the pooled and unpooled z-statistics for comparing two proportions, with discussion of when each applies | [Wikipedia](https://en.wikipedia.org/wiki/Two-proportion_Z-test) |
| PSU STAT 415, Lesson 25.3: Sample Size Calculations | Step-by-step derivation of two-sample z-test sample size formulas with worked examples | [PSU Online](https://online.stat.psu.edu/stat415/lesson/25/25.3) |
| PSU STAT 415, Lesson 13.2: The ANOVA Table | Derivation of the SST = SSB + SSW decomposition and the F-distribution result under the omnibus null | [PSU Online](https://online.stat.psu.edu/stat415/lesson/13/13.2) |
| Kohavi, Tang & Xu, *Trustworthy Online Controlled Experiments* (2020) | Practitioner reference for A/B testing at scale; covers variance reduction, multiple testing, and metric selection | [Cambridge UP](https://www.cambridge.org/core/books/trustworthy-online-controlled-experiments/D97B26382EB0EB2DC2019A7A7B518F59) |
| Powell, *Elements of Asymptotic Theory* (Berkeley lecture notes) | Formal treatment of Slutsky's theorem and its application to replacing nuisance parameters with consistent estimates in hypothesis tests | [Berkeley](https://eml.berkeley.edu/~powell/e240b_sp10/asynotes.pdf) |
| Wei (2019), "Probing into Minimum Sample Size Formula" (*Towards Data Science*) | Step-by-step derivation of the two-sample sample size formula from the power equation | [Medium](https://medium.com/data-science/probing-into-minimum-sample-size-formula-derivation-and-usage-8db9a556280b) |
