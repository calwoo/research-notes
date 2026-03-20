# A/B Testing: Exercises

## Table of Contents

- [[#Mathematical Development|Mathematical Development]]
  - [[#Problem 1 Unbiasedness of the Difference-in-Means Estimator|Problem 1: Unbiasedness of the Difference-in-Means Estimator]]
  - [[#Problem 2 Selection Bias Decomposition under Observational Data|Problem 2: Selection Bias Decomposition under Observational Data]]
  - [[#Problem 3 P-value Uniformity under the Null|Problem 3: P-value Uniformity under the Null]]
  - [[#Problem 4 Power Function for the Two-Sided Z-test|Problem 4: Power Function for the Two-Sided Z-test]]
  - [[#Problem 5 MDE Derivation for a Two-Sample Z-test|Problem 5: MDE Derivation for a Two-Sample Z-test]]
  - [[#Problem 6 Sample Size Formula All Four Steps|Problem 6: Sample Size Formula: All Four Steps]]
  - [[#Problem 7 Optimal Unequal Allocation Ratio|Problem 7: Optimal Unequal Allocation Ratio]]
  - [[#Problem 8 Welch-Satterthwaite Degrees of Freedom via Moment Matching|Problem 8: Welch-Satterthwaite Degrees of Freedom via Moment Matching]]
  - [[#Problem 9 ANOVA Sum of Squares Decomposition and F-statistic Distribution|Problem 9: ANOVA Sum of Squares Decomposition and F-statistic Distribution]]
  - [[#Problem 10 Beta-Binomial Conjugate Update|Problem 10: Beta-Binomial Conjugate Update]]
  - [[#Problem 11 Jeffreys Prior from Fisher Information|Problem 11: Jeffreys Prior from Fisher Information]]
  - [[#Problem 12 Posterior Mean as Weighted Average|Problem 12: Posterior Mean as Weighted Average]]
  - [[#Problem 13 Expected Loss for Deploying a Variant|Problem 13: Expected Loss for Deploying a Variant]]
  - [[#Problem 14 CUPED Variance Reduction Factor|Problem 14: CUPED Variance Reduction Factor]]
  - [[#Problem 15 Unbiasedness of the CUPED Estimator|Problem 15: Unbiasedness of the CUPED Estimator]]
  - [[#Problem 16 Bonferroni FWER Control via the Union Bound|Problem 16: Bonferroni FWER Control via the Union Bound]]
  - [[#Problem 17 BH Procedure FDR Control Level|Problem 17: BH Procedure FDR Control Level]]
  - [[#Problem 18 Regret Decomposition for Multi-Armed Bandits|Problem 18: Regret Decomposition for Multi-Armed Bandits]]
  - [[#Problem 19 UCB1 Expected Pulls Bound|Problem 19: UCB1 Expected Pulls Bound]]
  - [[#Problem 20 Covariance of Sequential Z-Statistics|Problem 20: Covariance of Sequential Z-Statistics]]
  - [[#Problem 21 Thompson Sampling Probability Matching Property|Problem 21: Thompson Sampling Probability Matching Property]]
- [[#Algorithmic Applications|Algorithmic Applications]]
  - [[#Problem 22 Sample Size Calculator via Binary Search|Problem 22: Sample Size Calculator via Binary Search]]
  - [[#Problem 23 Beta-Binomial Thompson Sampling Implementation|Problem 23: Beta-Binomial Thompson Sampling Implementation]]
  - [[#Problem 24 Benjamini-Hochberg Procedure Implementation|Problem 24: Benjamini-Hochberg Procedure Implementation]]
  - [[#Problem 25 CUPED Estimator Implementation|Problem 25: CUPED Estimator Implementation]]
  - [[#Problem 26 UCB1 Bandit with Regret Tracking|Problem 26: UCB1 Bandit with Regret Tracking]]

---

## Mathematical Development

### Problem 1: Unbiasedness of the Difference-in-Means Estimator

*This problem establishes the foundational result that the difference-in-means estimator is an unbiased estimator of the ATE, and identifies the precise step where random assignment is invoked.*

> **Prerequisites:** cf. note [[foundations#2.6 Unbiasedness of the Difference-in-Means Estimator|foundations §2.6]]

(a) State the potential outcomes framework formally. Define $Y_i^{\text{obs}}$, $Y_i(0)$, $Y_i(1)$, and the ATE $\tau$.

(b) Prove that under complete random assignment, $\mathbb{E}[Y_i(1) \mid T_i = 1] = \mathbb{E}[Y_i(1)]$. Identify the single property of random assignment that makes this step valid.

(c) Using (b), prove that $\mathbb{E}[\hat{\tau}] = \tau$ where $\hat{\tau} = \bar{Y}_1 - \bar{Y}_0$. Is the unbiasedness result specific to complete randomization, or does it hold for Bernoulli randomization as well? Justify.

### Problem 2: Selection Bias Decomposition under Observational Data

*This problem derives the decomposition of the naive difference-in-means estimator under an observational (non-randomized) design, making precise why observational studies cannot identify the ATE without further assumptions.*

> **Prerequisites:** cf. note [[foundations#1.2 Why Observational Data Is Insufficient|foundations §1.2]]

(a) Show that the observed difference in means decomposes as:
$$\mathbb{E}[Y_i \mid T_i = 1] - \mathbb{E}[Y_i \mid T_i = 0] = \underbrace{\mathbb{E}[Y_i(1) - Y_i(0) \mid T_i = 1]}_{\text{ATT}} + \underbrace{\mathbb{E}[Y_i(0) \mid T_i = 1] - \mathbb{E}[Y_i(0) \mid T_i = 0]}_{\text{selection bias}}$$
Justify each algebraic step using the switching equation $Y_i^{\text{obs}} = T_i Y_i(1) + (1-T_i)Y_i(0)$.

(b) Prove that under random assignment $(Y_i(0), Y_i(1)) \perp\!\!\!\perp T_i$, the selection bias term is exactly zero and the ATT equals the ATE.

(c) Give a concrete data-generating process (specifying distributions of $Y_i(0)$, $Y_i(1)$, and a selection rule for $T_i$) in which the selection bias term is strictly positive. Compute it numerically for your example.

### Problem 3: P-value Uniformity under the Null

*This problem proves that the p-value, as a pre-data random variable, is stochastically uniform under the null — the mathematical foundation of the multiple comparisons problem.*

> **Prerequisites:** cf. note [[foundations#5.2 The P-value Is Uniform Under the Null|foundations §5.2]]

(a) Let $W_n$ be a test statistic with continuous CDF $F_{H_0}$ under $H_0$. Define $U = F_{H_0}(W_n)$ and prove $U \sim \text{Uniform}(0,1)$ using the probability integral transform. State explicitly where continuity is required.

(b) Conclude that the one-tailed p-value $P = 1 - F_{H_0}(W_n)$ satisfies $P \sim \text{Uniform}(0,1)$ under $H_0$.

(c) Suppose we run $m$ independent tests simultaneously, all under their respective null hypotheses, each at level $\alpha$. Show that the probability of at least one false rejection is $1 - (1-\alpha)^m$, and derive the asymptotic approximation $\approx m\alpha$ for small $\alpha$.

(d) Prove that for a discrete test statistic $W_n$ (e.g., an exact binomial test), the p-value is stochastically larger than $\text{Uniform}(0,1)$, i.e., $P(P \leq u) \leq u$ for all $u \in [0,1]$. What does this imply about test conservatism?

### Problem 4: Power Function for the Two-Sided Z-test

*This problem derives the power function $\pi(\delta)$ and verifies three structural properties that characterize how a test's sensitivity varies with true effect size.*

> **Prerequisites:** cf. note [[foundations#6.1 Definition and the Power Function|foundations §6.1]]

(a) For the two-sample z-test with equal group sizes $n$ and common variance $\sigma^2$, derive the distribution of the test statistic $Z$ under the alternative $H_1: \tau = \delta$. Express the non-centrality parameter $\lambda$ as a function of $\delta$, $n$, and $\sigma^2$.

(b) Show that the power function is approximately:
$$\pi(\delta) \approx \Phi(\lambda - z_{\alpha/2}) + \Phi(-\lambda - z_{\alpha/2})$$
and explain under what condition the second term is negligible. When does the approximation $\pi(\delta) \approx \Phi(\lambda - z_{\alpha/2})$ fail non-negligibly?

(c) Verify the three structural properties: (i) $\pi(0) = \alpha$, (ii) $\pi(\delta) \to 1$ as $|\delta| \to \infty$, and (iii) $\pi(\delta) = \pi(-\delta)$ for the two-tailed test.

(d) Show that at the MDE (i.e., at $\delta = \text{MDE}$), the power exactly equals $1 - \beta$ by construction. Derive a formula for MDE as a function of $n$, $\sigma^2$, $\alpha$, and $\beta$.

### Problem 5: MDE Derivation for a Two-Sample Z-test

*This problem derives the closed-form MDE formula and exhibits the quadratic relationship between MDE and sample size, quantifying the cost of detecting smaller effects.*

> **Prerequisites:** cf. note [[foundations#7.4 Derivation of the MDE for a Two-Sample Z-test|foundations §7.4]]; requires Problem 4

(a) For a two-tailed z-test at level $\alpha$ with $n$ observations per group and outcome variance $\sigma^2$, set up the power equation $\pi(\text{MDE}) = 1 - \beta$ and solve for:
$$\text{MDE} = (z_{\alpha/2} + z_\beta)\sqrt{\frac{2\sigma^2}{n}}$$
Identify the source of the factor of 2 in $2\sigma^2/n$.

(b) Invert the MDE formula to express the required sample size $n$ as a function of $\delta$ (the desired detectable effect), $\sigma^2$, $\alpha$, and $\beta$:
$$n = \frac{(z_{\alpha/2} + z_\beta)^2 \cdot 2\sigma^2}{\delta^2}$$
Show that doubling the MDE threshold reduces the required $n$ by a factor of 4.

(c) For binary outcomes with baseline conversion rate $p_0$ and treatment rate $p_1 = p_0 + \delta$, express $n$ in terms of $p_0$, $\delta$, $z_{\alpha/2}$, and $z_\beta$ using the formula from (b). Show that $n \propto p_0^{-1}$ for rare events ($p_0 \ll 1$, $\delta \ll p_0$).

### Problem 6: Sample Size Formula: All Four Steps

*This problem reconstructs the four-step derivation of the per-group sample size formula for the two-sample z-test from first principles, including the key approximation step.*

> **Prerequisites:** cf. note [[frequentist-testing#5.1 Derivation from the Power Equation|frequentist-testing §5.1]]

(a) **Step 1.** Show that under the alternative $H_1: \mu_B - \mu_A = \delta$, the z-statistic $Z = \hat{\tau}/\sqrt{(\sigma_A^2 + \sigma_B^2)/n}$ satisfies $Z \sim \mathcal{N}(\mu_Z, 1)$ asymptotically, and compute $\mu_Z$ in terms of $\delta$, $n$, $\sigma_A^2$, $\sigma_B^2$.

(b) **Step 2.** Express the power $1 - \beta = P_{H_1}(|Z| > z_{\alpha/2})$ in terms of $\mu_Z$ and $z_{\alpha/2}$. State and justify the approximation that drops the lower-tail probability $P(Z < -z_{\alpha/2})$.

(c) **Step 3.** Set the power equation equal to $1 - \beta$ and solve for $n$. What is the resulting formula?

(d) **Step 4.** Explain why the final answer requires a ceiling operation $\lceil n \rceil$. In a simulation study, when would you expect the ceiling to matter most for the actual achieved power?

### Problem 7: Optimal Unequal Allocation Ratio

*This problem derives the variance-minimizing allocation ratio $k^* = \sigma_B / \sigma_A$ for a two-sample experiment with fixed total sample size, and identifies when equal allocation is suboptimal.*

> **Prerequisites:** cf. note [[frequentist-testing#5.3 Unequal Allocation|frequentist-testing §5.3]]

(a) Let $n_B = k \cdot n_A$ and $N = n_A + n_B$ (fixed). Express $\text{Var}(\hat{\tau}) = \sigma_A^2/n_A + \sigma_B^2/n_B$ as a function of $k$ and $N$.

(b) Minimize $\text{Var}(\hat{\tau})$ over $k > 0$ by differentiating and setting to zero. Show that the optimal ratio is $k^* = \sigma_B/\sigma_A$.

(c) Suppose $\sigma_A = 1$ and $\sigma_B = 3$. Compute the optimal $k^*$ and the variance $\text{Var}(\hat{\tau})$ at $k^*$. Compare to the variance under equal allocation $k = 1$. By what factor does unequal allocation reduce variance?

(d) Prove that for equal variances $\sigma_A = \sigma_B$, the optimum is $k^* = 1$ (equal allocation), and verify the minimum variance formula.

### Problem 8: Welch-Satterthwaite Degrees of Freedom via Moment Matching

*This problem derives the Welch-Satterthwaite effective degrees of freedom by matching the first two moments of $U + V$ to those of a scaled chi-squared distribution, motivating the formula used in practice.*

> **Prerequisites:** cf. note [[frequentist-testing#3.4 The Welch-Satterthwaite Degrees of Freedom|frequentist-testing §3.4]]

(a) Let $U = s_A^2/n_A$ and $V = s_B^2/n_B$. Under normality, show that:
$$\mathbb{E}[U] = \frac{\sigma_A^2}{n_A}, \quad \text{Var}(U) = \frac{2\sigma_A^4}{n_A^2(n_A - 1)}$$
and analogously for $V$. (Hint: use $(n_A - 1)s_A^2/\sigma_A^2 \sim \chi^2_{n_A-1}$.)

(b) We approximate $W = U + V \approx c \cdot \chi^2_\nu$ for some $c$ and $\nu$. Write the two moment-matching equations (equating $\mathbb{E}[W]$ and $\text{Var}(W)$ to those of $c\chi^2_\nu$) and solve for $\nu$ by dividing the variance equation by the square of the mean equation.

(c) Substitute the expressions from (a) into the result of (b) to derive the Welch-Satterthwaite formula:
$$\nu = \frac{(s_A^2/n_A + s_B^2/n_B)^2}{\dfrac{(s_A^2/n_A)^2}{n_A - 1} + \dfrac{(s_B^2/n_B)^2}{n_B - 1}}$$

(d) Show that $\nu$ is bounded: $\min(n_A - 1, n_B - 1) \leq \nu \leq n_A + n_B - 2$. When is the upper bound achieved exactly?

### Problem 9: ANOVA Sum of Squares Decomposition and F-statistic Distribution

*This problem proves the SST = SSB + SSW decomposition and derives the distribution of the F-statistic under the omnibus null, identifying the role of Cochran's theorem.*

> **Prerequisites:** cf. note [[frequentist-testing#6.1 Setup and Sum of Squares Decomposition|frequentist-testing §6.1]]; cf. note [[frequentist-testing#6.2 The F-statistic and Its Distribution|frequentist-testing §6.2]]

(a) For $K$ groups with $n_k$ observations each, prove the identity $\text{SST} = \text{SSB} + \text{SSW}$ by writing $Y_{ik} - \bar{Y} = (Y_{ik} - \bar{Y}_k) + (\bar{Y}_k - \bar{Y})$, squaring, summing, and showing the cross-term vanishes.

(b) State the degrees of freedom for SSB and SSW, and give the intuitive argument for each count. What constraint causes SSB to have $K-1$ rather than $K$ degrees of freedom?

(c) Under $H_0: \mu_1 = \cdots = \mu_K$, normality, and homoskedasticity, show that $\text{SSB}/\sigma^2 \sim \chi^2_{K-1}$ and $\text{SSW}/\sigma^2 \sim \chi^2_{N-K}$ (for balanced designs). Why does the distribution of SSW hold even when $H_0$ is false?

(d) Assuming the independence of SSB and SSW (which follows from Cochran's theorem), conclude $F = \text{MSB}/\text{MSW} \sim F(K-1, N-K)$ under $H_0$. Show that under $H_1$, $\mathbb{E}[\text{MSB}] > \sigma^2$, making large $F$ evidence against $H_0$.

### Problem 10: Beta-Binomial Conjugate Update

*This problem derives the Beta-Binomial conjugacy result from first principles, establishing that the posterior shape parameters increment by observed successes and failures respectively.*

> **Prerequisites:** cf. note [[bayesian-testing#2.3 Conjugate Posterior Update|bayesian-testing §2.3]]

(a) Let $p \sim \text{Beta}(\alpha_0, \beta_0)$ and $X \mid p \sim \text{Binomial}(n, p)$ with $X = k$. Write out $\pi(p \mid k) \propto p(k \mid p)\pi(p)$ explicitly and identify the kernel of the resulting distribution.

(b) Prove that $p \mid X = k \sim \text{Beta}(\alpha_0 + k, \beta_0 + n - k)$ by appealing to the uniqueness of the normalizing constant of a probability density.

(c) Interpret the pseudo-counts $\alpha_0$ and $\beta_0$ in the prior as encoding a prior belief based on $n_0 = \alpha_0 + \beta_0$ hypothetical observations. Show that the posterior mean interpolates between the prior mean $\alpha_0/n_0$ and the MLE $k/n$.

(d) Derive the posterior mean from (b) and express it as the weighted average:
$$\mathbb{E}[p \mid k] = \frac{n_0}{n_0 + n} \cdot \frac{\alpha_0}{n_0} + \frac{n}{n_0 + n} \cdot \frac{k}{n}$$
Verify that as $n \to \infty$, the posterior mean converges to the MLE regardless of the prior.

### Problem 11: Jeffreys Prior from Fisher Information

*This problem derives the Jeffreys prior for the Bernoulli model and proves its reparameterization invariance, distinguishing it from the uniform prior.*

> **Prerequisites:** cf. note [[bayesian-testing#5.2 Jeffreys Prior|bayesian-testing §5.2]]

(a) For $X \sim \text{Bernoulli}(p)$, compute the Fisher information $I(p) = -\mathbb{E}[\partial^2 \ell / \partial p^2]$ where $\ell(p; x) = x\log p + (1-x)\log(1-p)$. Show $I(p) = 1/(p(1-p))$.

(b) Show that the Jeffreys prior $\pi_J(p) \propto \sqrt{I(p)} = (p(1-p))^{-1/2}$ is the $\text{Beta}(1/2, 1/2)$ density.

(c) Prove the reparameterization invariance property: if $\phi = g(p)$ is a smooth bijection, then the Jeffreys prior on $\phi$ is $\pi_J(\phi) \propto \sqrt{I_\phi(\phi)}$, where $I_\phi(\phi)$ is the Fisher information in the $\phi$ parameterization. (Hint: use $I_\phi(\phi) = I(p)(dp/d\phi)^2$ and the change-of-variables formula.)

(d) Apply (c) to the logit parameterization $\phi = \log(p/(1-p))$. Compute $\pi_J(\phi)$ and verify it is the logistic distribution, confirming invariance.

### Problem 12: Posterior Mean as Weighted Average

*This problem proves the shrinkage structure of the Beta-Binomial posterior mean and derives the asymptotic rate at which the prior is overwhelmed by data.*

> **Prerequisites:** cf. note [[bayesian-testing#2.4 Posterior Mean as a Shrinkage Estimator|bayesian-testing §2.4]]; requires Problem 10

(a) Using the conjugate update $p \mid k \sim \text{Beta}(\alpha_0 + k, \beta_0 + n - k)$, write the posterior mean as a weighted combination of the prior mean $\mu_0 = \alpha_0/(\alpha_0+\beta_0)$ and the MLE $\hat{p} = k/n$, with weights $n_0/(n_0 + n)$ and $n/(n_0 + n)$ where $n_0 = \alpha_0 + \beta_0$.

(b) Show that the posterior mean is always shrunk toward the prior mean: $|\mathbb{E}[p \mid k] - \hat{p}| \leq |\mu_0 - \hat{p}|$. Under what condition does the posterior mean equal the MLE?

(c) Quantify the rate of convergence: show that $|\mathbb{E}[p \mid k] - \hat{p}| = O(n^{-1})$ as $n \to \infty$ for fixed prior parameters $\alpha_0$, $\beta_0$.

(d) Suppose a practitioner has historical data suggesting $p \approx 0.05$ with effective sample size $n_0 = 1000$. If an experiment yields 10 successes out of 100 trials, compute the posterior mean and compare to the MLE $k/n = 0.10$.

### Problem 13: Expected Loss for Deploying a Variant

*This problem formulates the expected loss criterion for the Bayesian A/B decision problem and shows how the Monte Carlo estimator arises from the integral definition.*

> **Prerequisites:** cf. note [[bayesian-testing#4.2 Expected Loss|bayesian-testing §4.2]]

(a) Define the expected loss for deploying variant B as $\mathcal{L}_B = \mathbb{E}[\max(p_A - p_B, 0) \mid \text{data}]$ with independent Beta posteriors $p_A \sim \text{Beta}(\alpha_A', \beta_A')$ and $p_B \sim \text{Beta}(\alpha_B', \beta_B')$. Write the double integral expression for $\mathcal{L}_B$ explicitly.

(b) Decompose $\mathcal{L}_B = T_1 - T_2$ where:
$$T_1 = \int_0^1 \int_{p_B}^1 p_A\, \pi(p_A)\, dp_A\, \pi(p_B)\, dp_B, \quad T_2 = \int_0^1 p_B\, \pi(p_B) \int_{p_B}^1 \pi(p_A)\, dp_A\, dp_B$$
Show this decomposition is exact.

(c) Show that $T_1 = \mathbb{E}[p_A \cdot \mathbf{1}[p_A > p_B]]$ and express $T_1$ in terms of the posterior mean of $p_A$ and a probability of superiority computed under a shifted Beta distribution $\text{Beta}(\alpha_A' + 1, \beta_A')$.

(d) Give the Monte Carlo estimator:
$$\hat{\mathcal{L}}_B = \frac{1}{S}\sum_{s=1}^S \max(p_A^{(s)} - p_B^{(s)}, 0)$$
where $p_A^{(s)} \sim \text{Beta}(\alpha_A', \beta_A')$ and $p_B^{(s)} \sim \text{Beta}(\alpha_B', \beta_B')$. By the law of large numbers, show this converges to $\mathcal{L}_B$. Bound the Monte Carlo standard error as a function of $S$.

### Problem 14: CUPED Variance Reduction Factor

*This problem derives the variance reduction achieved by CUPED as a fraction $\rho^2$ of the original outcome variance, and identifies the optimal adjustment coefficient.*

> **Prerequisites:** cf. note [[experimental-design#3.3 Optimal Theta: Derivation|experimental-design §3.3]]; cf. note [[experimental-design#3.4 Variance Reduction in Terms of Correlation|experimental-design §3.4]]

(a) For the adjusted outcome $Y_i^{\text{CUPED}} = Y_i - \theta(X_i - \mathbb{E}[X_i])$, compute $\text{Var}(Y_i^{\text{CUPED}})$ as a quadratic in $\theta$. Show it is minimized at $\theta^* = \text{Cov}(Y_i, X_i)/\text{Var}(X_i)$.

(b) Substitute $\theta^*$ back into the variance formula and show:
$$\text{Var}(Y_i^{\text{CUPED}}) = \text{Var}(Y_i)(1 - \rho^2)$$
where $\rho = \text{Corr}(Y_i, X_i)$.

(c) Express the variance reduction in sample size terms: if we reduce outcome variance by factor $(1 - \rho^2)$, by what factor is the required sample size reduced (at fixed power and $\alpha$)?

(d) For $\rho = 0$, $\rho = 0.5$, and $\rho = 0.8$, compute the variance reduction fraction $\rho^2$ and the corresponding sample size reduction. What correlation is needed to halve the required sample size?

### Problem 15: Unbiasedness of the CUPED Estimator

*This problem proves that CUPED preserves unbiasedness of the ATE estimator by exploiting the pre-randomization independence $X_i \perp T_i$, and identifies the condition that would introduce bias.*

> **Prerequisites:** cf. note [[experimental-design#3.5 Unbiasedness of the CUPED Estimator|experimental-design §3.5]]; requires Problem 14

(a) Compute $\mathbb{E}[Y_i^{\text{CUPED}} \mid T_i = t]$ for $t \in \{0, 1\}$ and show the adjustment term $\theta^*(X_i - \mathbb{E}[X_i])$ has conditional mean zero given $T_i = t$.

(b) Conclude that $\mathbb{E}[\hat{\tau}^{\text{CUPED}}] = \mathbb{E}[Y_i \mid T_i = 1] - \mathbb{E}[Y_i \mid T_i = 0] = \tau$.

(c) Suppose instead that $X_i$ is a *post-experiment* covariate (measured during the experiment window). Explain precisely why the step in (a) breaks down. Under what condition on $\text{Cov}(X_i, T_i)$ is the resulting bias zero despite using a post-experiment covariate?

(d) Show that the CUPED estimator $\hat{\tau}^{\text{CUPED}}$ is algebraically equivalent to the OLS coefficient on $T_i$ in the regression $Y_i = \alpha + \tau T_i + \beta X_i + \varepsilon_i$. (Appeal to the Frisch-Waugh-Lovell theorem and the key property $T_i \perp X_i$.)

### Problem 16: Bonferroni FWER Control via the Union Bound

*This problem proves that the Bonferroni correction controls the family-wise error rate at level $\alpha$ under arbitrary dependence, and quantifies the conservatism introduced by using $m$ rather than $m_0$.*

> **Prerequisites:** cf. note [[experimental-design#4.3 Bonferroni Correction|experimental-design §4.3]]

(a) Let $\{H_0^1, \ldots, H_0^m\}$ be $m$ null hypotheses tested at threshold $\alpha/m$. Let $\mathcal{H}_0 \subseteq \{1,\ldots,m\}$ index the true nulls ($|\mathcal{H}_0| = m_0$). Prove $\text{FWER} \leq \alpha$ using the union bound, without assuming independence of the test statistics.

(b) Show that the Bonferroni bound is tight: give a construction (with $m_0 = m$, all tests independent) under which $\text{FWER} = 1 - (1-\alpha/m)^m$, and show that $1 - (1-\alpha/m)^m \to 1 - e^{-\alpha}$ as $m \to \infty$. Is FWER bounded away from $\alpha$ in this limit?

(c) Prove that $\text{FWER} \leq m_0\alpha/m \leq \alpha$. Under what condition does $\text{FWER} = m_0\alpha/m$ hold exactly? What does this imply about the power of Bonferroni when $m_0 \ll m$?

(d) Show that the Holm-Bonferroni procedure uniformly dominates Bonferroni: at step $j$, if $p_{(j)} \leq \alpha/(m-j+1)$ then $p_{(j)} \leq \alpha/m$ only if $j = 1$. For $j \geq 2$, the Holm threshold is strictly larger than the Bonferroni threshold, so Holm makes at least as many rejections.

### Problem 17: BH Procedure FDR Control Level

*This problem establishes that the Benjamini-Hochberg procedure controls FDR at level $m_0\alpha/m$ under independence, and identifies why the bound is tight when all nulls are true.*

> **Prerequisites:** cf. note [[experimental-design#5.3 FDR Control Level of BH under Independence|experimental-design §5.3]]

(a) State the BH procedure: given ordered p-values $p_{(1)} \leq \cdots \leq p_{(m)}$, define $k^* = \max\{j : p_{(j)} \leq j\alpha/m\}$ and reject $H_0^{(1)}, \ldots, H_0^{(k^*)}$.

(b) Write FDR $= \sum_{j=1}^{m_0} \mathbb{E}[\mathbf{1}[p_j \leq p_{(k^*)}]/R]$ and argue why the sum runs over only the true null indices.

(c) For a single true null $p_j \sim \text{Uniform}(0,1)$ (independent of all other p-values), show that its contribution satisfies $\mathbb{E}[\mathbf{1}[p_j \leq p_{(k^*)}]/R] \leq \alpha/m$ using the argument that if $j$ is the $\ell$-th smallest p-value and is rejected, then $p_j \leq \ell\alpha/m$ and $R \geq \ell$.

(d) Sum over $m_0$ true nulls to conclude $\text{FDR} \leq m_0\alpha/m$. Show this bound is tight when $m_0 = m$ (all nulls true), and that in this case FDR $=$ FWER.

### Problem 18: Regret Decomposition for Multi-Armed Bandits

*This problem derives the regret decomposition $\text{Regret}(T) = \sum_k \Delta_k \mathbb{E}[N_k(T)]$ from the definition of cumulative regret, establishing the fundamental role of suboptimality gaps.*

> **Prerequisites:** cf. note [[sequential-and-adaptive#4.2 Regret and Its Decomposition|sequential-and-adaptive §4.2]]

(a) Define cumulative regret $\text{Regret}(T) = T\mu^* - \mathbb{E}[\sum_{t=1}^T R_{a_t,t}]$ and write $\mathbb{E}[R_{a_t,t}] = \mu_{a_t}$ using the tower property.

(b) Show that $\text{Regret}(T) = \mathbb{E}[\sum_{t=1}^T (\mu^* - \mu_{a_t})]$ and rewrite the inner sum as $\sum_k \Delta_k N_k(T)$ where $\Delta_k = \mu^* - \mu_k$ and $N_k(T) = \sum_{t=1}^T \mathbf{1}[a_t = k]$.

(c) Conclude $\text{Regret}(T) = \sum_k \Delta_k \mathbb{E}[N_k(T)]$ by linearity of expectation. Note that the $k^*$ term contributes zero. What does this imply about the algorithm design problem?

(d) For a two-armed bandit with $\mu^* = 0.6$ and $\mu_2 = 0.4$, and a policy that pulls arm 2 exactly $\lceil C \ln T \rceil$ times (for some constant $C$), compute $\text{Regret}(T)$ to leading order in $T$. Is this policy consistent (i.e., $\text{Regret}(T) = o(T^\alpha)$ for all $\alpha > 0$)?

### Problem 19: UCB1 Expected Pulls Bound

*This problem outlines the key concentration argument showing that UCB1 pulls each suboptimal arm at most $O(\ln T / \Delta_k^2)$ times in expectation, establishing the logarithmic regret rate.*

> **Prerequisites:** cf. note [[sequential-and-adaptive#5.3 Regret Bound for UCB1|sequential-and-adaptive §5.3]]; requires Problem 18

(a) State the UCB1 selection rule. Arm $k$ is pulled at round $t$ only if $\hat{\mu}_k(t-1) + \sqrt{2\ln t / N_k(t-1)} \geq \hat{\mu}_{k^*}(t-1) + \sqrt{2\ln t / N_{k^*}(t-1)}$. Show that if $N_k(t-1) \geq l$ for some threshold $l$, then this event implies either the UCB of $k^*$ is below $\mu^*$ or the UCB of $k$ is above $\mu^*$.

(b) Define $l = \lceil 8 \ln T / \Delta_k^2 \rceil$. By the Hoeffding inequality, bound the probability that $\hat{\mu}_k + \sqrt{2\ln t / N_k} \geq \mu^*$ when $N_k \geq l$, and show it is at most $t^{-4}$.

(c) Using (a) and (b), bound $\mathbb{E}[N_k(T)] \leq l + \sum_{t=1}^T P(\text{UCB}(k) \geq \mu^*$ with $N_k \geq l)$ and conclude $\mathbb{E}[N_k(T)] \leq 8\ln T/\Delta_k^2 + 1 + \pi^2/3$.

(d) Using the regret decomposition from Problem 18, state the resulting bound $\mathbb{E}[\text{Regret}(T)] \leq \sum_{k:\Delta_k>0} 8\ln T/\Delta_k + (1 + \pi^2/3)\sum_k \Delta_k$. Verify this is $O(\ln T)$ in $T$.

### Problem 20: Covariance of Sequential Z-Statistics

*This problem derives the canonical covariance structure $\text{Cov}(Z_j, Z_k) = \sqrt{t_j/t_k}$ of group sequential test statistics, which underlies all group sequential boundary computations.*

> **Prerequisites:** cf. note [[sequential-and-adaptive#2.2 The Covariance Structure of Sequential Z-Statistics|sequential-and-adaptive §2.2]]

(a) Let $X_1, X_2, \ldots \overset{\text{iid}}{\sim} (\mu, \sigma^2)$ and define $Z_k = S_{n_k}/(\sigma\sqrt{n_k})$ where $S_{n_k} = \sum_{i=1}^{n_k} X_i$. For $j \leq k$, write $S_{n_k} = S_{n_j} + (S_{n_k} - S_{n_j})$ and argue that $S_{n_k} - S_{n_j}$ is independent of $S_{n_j}$.

(b) Compute $\text{Cov}(S_{n_j}, S_{n_k}) = \text{Var}(S_{n_j}) = \sigma^2 n_j$ using the independence of the increment.

(c) Conclude $\text{Cov}(Z_j, Z_k) = \sqrt{n_j/n_k} = \sqrt{t_j/t_k}$. Since $\text{Var}(Z_k) = 1$, note that $\text{Cov} = \text{Corr}$ here.

(d) For the two-look procedure with $t_1 = 1/2$ and $t_2 = 1$, the correlation is $\text{Corr}(Z_1, Z_2) = 1/\sqrt{2}$. Show that the FWER of the naive procedure (rejecting at either look using critical value $z_{0.025} = 1.96$) is approximately $0.083$, using the bivariate normal CDF with this correlation.

### Problem 21: Thompson Sampling Probability Matching Property

*This problem proves that the Thompson sampling selection probability for each arm equals the posterior probability that arm is optimal, establishing the algorithm's self-regulating exploration mechanism.*

> **Prerequisites:** cf. note [[sequential-and-adaptive#6.4 The Probability Matching Property|sequential-and-adaptive §6.4]]; requires Problem 10

(a) Let $\theta_k^{(t)} \sim \text{Beta}(\alpha_k, \beta_k)$ independently for $k = 1, \ldots, K$, representing posterior samples at round $t$. Define $a_t = \argmax_k \theta_k^{(t)}$.

(b) Show that $P(a_t = k \mid \mathcal{F}_{t-1}) = P(\theta_k^{(t)} > \theta_j^{(t)} \text{ for all } j \neq k \mid \mathcal{F}_{t-1})$ directly from the definition of $a_t$.

(c) Argue that the probability in (b) equals $P(\mu_k = \mu^* \mid \mathcal{F}_{t-1})$ — the posterior probability that arm $k$ is the true optimal arm. (Hint: $(\theta_1^{(t)}, \ldots, \theta_K^{(t)})$ is a joint draw from the posterior over $(\mu_1, \ldots, \mu_K)$, and the event $\{\theta_k > \theta_j \text{ for all } j\}$ has the same probability as $\{\mu_k > \mu_j \text{ for all } j\}$ under the posterior.)

(d) Explain the self-regulating character: why does probability matching naturally balance exploration and exploitation without an explicit exploration bonus term? Under what condition on the posteriors does Thompson sampling converge to pure exploitation?

---

## Algorithmic Applications

### Problem 22: Sample Size Calculator via Binary Search

*This problem constructs a numerical sample size calculator that finds the minimum per-group $n$ achieving target power for arbitrary (possibly non-symmetric) variance settings, using binary search on the power function.*

> **Prerequisites:** cf. note [[frequentist-testing#5.1 Derivation from the Power Equation|frequentist-testing §5.1]]; cf. note [[foundations#7.4 Derivation of the MDE for a Two-Sample Z-test|foundations §7.4]]

(a) **Inputs and power function**: Define the inputs: $\sigma_A^2$, $\sigma_B^2$, target MDE $\delta$, $\alpha$ (two-sided), power target $1 - \beta$. Write pseudocode for `compute_power(n, delta, sigma_A_sq, sigma_B_sq, alpha)` that computes $\pi(\delta)$ using the formula from Problem 4.

(b) **Binary search structure**: Write pseudocode for `find_sample_size(delta, sigma_A_sq, sigma_B_sq, alpha, target_power)` that binary-searches over $n \in [n_{\min}, n_{\max}]$ (with $n_{\min} = 1$ and $n_{\max} = 10^7$) to find the smallest integer $n$ such that `compute_power(n, ...) >= target_power`. Specify the termination condition.

(c) **Closed-form check**: For the symmetric case $\sigma_A^2 = \sigma_B^2 = \sigma^2$, verify the binary search result matches the closed-form formula $n = \lceil (z_{\alpha/2} + z_\beta)^2 \cdot 2\sigma^2 / \delta^2 \rceil$. What is the computational complexity of the binary search relative to directly evaluating the closed form?

(d) **Edge cases**: Describe what happens when $\delta = 0$ or when the target power exceeds 1. Add guards to handle these in the pseudocode.

### Problem 23: Beta-Binomial Thompson Sampling Implementation

*This problem implements the Thompson Sampling algorithm for Bernoulli bandits with Beta-Binomial conjugate updates, and shows how to estimate the probability of superiority and expected loss via Monte Carlo.*

> **Prerequisites:** cf. note [[sequential-and-adaptive#6.1 Algorithm|sequential-and-adaptive §6.1]]; cf. note [[bayesian-testing#4.2 Expected Loss|bayesian-testing §4.2]]; requires Problem 10

(a) **Data structures**: Define the state for a $K$-armed Thompson Sampling bandit. Specify the shape parameters $(\alpha_k, \beta_k)$ for each arm and their initialization.

(b) **Main loop**: Write pseudocode for one round of Thompson Sampling: sample $\theta_k \sim \text{Beta}(\alpha_k, \beta_k)$ for all $k$, select $a = \argmax_k \theta_k$, observe $R \in \{0,1\}$, and update the posterior of arm $a$.

(c) **Metrics after $T$ rounds**: Write pseudocode for `estimate_probability_of_superiority(alpha, beta, S)` that uses $S$ Monte Carlo samples to estimate $P(p_B > p_A \mid \text{data})$ for a two-arm bandit. Annotate the sampling and aggregation steps.

(d) **Expected loss**: Extend (c) to write `estimate_expected_loss(alpha_A, beta_A, alpha_B, beta_B, S)` returning $\hat{\mathcal{L}}_B = S^{-1}\sum_s \max(p_A^{(s)} - p_B^{(s)}, 0)$. What is the Monte Carlo standard error, and how large does $S$ need to be for 3 significant figures of accuracy?

### Problem 24: Benjamini-Hochberg Procedure Implementation

*This problem implements the BH procedure and compares it against Bonferroni and Holm on a concrete example, illustrating the power advantage of FDR over FWER control.*

> **Prerequisites:** cf. note [[experimental-design#5.2 Benjamini-Hochberg Procedure|experimental-design §5.2]]; cf. note [[experimental-design#4.3 Bonferroni Correction|experimental-design §4.3]]

(a) **BH implementation**: Write pseudocode for `bh_procedure(p_values, alpha)` that: (i) sorts the p-values, (ii) computes thresholds $j\alpha/m$ for $j = 1, \ldots, m$, (iii) finds $k^* = \max\{j : p_{(j)} \leq j\alpha/m\}$, and (iv) returns the set of rejected hypotheses. Handle the case $k^* = 0$ (no rejections).

(b) **Bonferroni and Holm**: Write pseudocode for `bonferroni(p_values, alpha)` and `holm(p_values, alpha)` as baselines. The Holm procedure is step-down: sort, compare $p_{(j)}$ to $\alpha/(m-j+1)$, stop at first non-rejection.

(c) **Worked example**: Apply all three procedures to the $m = 10$ p-values from the note's worked example. Report the rejection sets for each and verify the counts match those in the note: Bonferroni (1), Holm (1), BH (4).

(d) **Complexity**: State the time complexity of each procedure. What is the bottleneck, and can it be parallelized?

### Problem 25: CUPED Estimator Implementation

*This problem implements the CUPED variance reduction procedure end-to-end, including estimation of the adjustment coefficient $\theta^*$ and construction of the adjusted ATE estimate with its standard error.*

> **Prerequisites:** cf. note [[experimental-design#3.2 The Adjusted Outcome|experimental-design §3.2]]; cf. note [[experimental-design#3.3 Optimal Theta: Derivation|experimental-design §3.3]]

(a) **Inputs and preprocessing**: Define the inputs: arrays `Y` (outcome), `X` (pre-experiment covariate), `T` (treatment indicator). Write pseudocode to estimate $\theta^* = \hat{\text{Cov}}(Y, X)/\hat{\text{Var}}(X)$ using the pooled sample (all units regardless of treatment).

(b) **Adjusted outcome and ATE estimate**: Write pseudocode to compute `Y_cuped = Y - theta_star * (X - mean(X))` and then compute `tau_hat = mean(Y_cuped[T==1]) - mean(Y_cuped[T==0])`.

(c) **Standard error**: Write pseudocode to compute the standard error of `tau_hat` as $\sqrt{\text{Var}(Y_i^{\text{CUPED}} \mid T_i=1)/n_1 + \text{Var}(Y_i^{\text{CUPED}} \mid T_i=0)/n_0}$ where the variances are estimated within-arm.

(d) **Variance reduction check**: Write pseudocode to compute the empirical correlation $\hat{\rho}$ between `Y` and `X` and verify that `Var(Y_cuped) ≈ Var(Y) * (1 - rho_hat^2)` to within sampling error. Annotate the expected relative reduction.

### Problem 26: UCB1 Bandit with Regret Tracking

*This problem implements the UCB1 algorithm and produces a regret curve, verifying empirically that cumulative regret grows as $O(\ln T)$ for a two-armed Bernoulli bandit.*

> **Prerequisites:** cf. note [[sequential-and-adaptive#5.2 UCB1|sequential-and-adaptive §5.2]]; cf. note [[sequential-and-adaptive#5.3 Regret Bound for UCB1|sequential-and-adaptive §5.3]]; requires Problem 18

(a) **Data structures**: Define the bandit state: true arm means $\mu_1, \ldots, \mu_K$, empirical means $\hat{\mu}_k$, pull counts $N_k$, total rounds $t$. Specify the initialization (pull each arm once before the main loop).

(b) **UCB1 selection and update**: Write pseudocode for one round of UCB1: compute $\text{UCB}(k) = \hat{\mu}_k + \sqrt{2\ln t / N_k}$ for all $k$, select $a = \argmax_k \text{UCB}(k)$, observe reward, update $\hat{\mu}_a$ and $N_a$.

(c) **Regret tracking**: After $T$ rounds, compute cumulative regret as $\text{Regret}(T) = T \cdot \mu^* - \sum_{t=1}^T R_{a_t,t}$. Write pseudocode to log the regret at each round and compute the ratio $\text{Regret}(T) / \ln T$ at $T = 100, 1000, 10000$.

(d) **Theoretical check**: For a two-armed bandit with $\mu^* = 0.6$ and $\mu_2 = 0.4$ (so $\Delta_2 = 0.2$), the UCB1 bound gives $\mathbb{E}[N_2(T)] \leq 8\ln T / (0.2)^2 = 200\ln T$. Verify that this is consistent with the empirical $N_2(T)$ from your pseudocode simulation. What constant in front of $\ln T$ does the empirical $\mathbb{E}[\text{Regret}(T)]/\ln T$ converge to as $T \to \infty$?
