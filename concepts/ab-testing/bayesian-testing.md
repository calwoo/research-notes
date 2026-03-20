# A/B Testing: Bayesian Approach

## Table of Contents

1. [[#1. The Bayesian Framework|1. The Bayesian Framework]]
   - [[#1.1 Bayes' Theorem and the Posterior|1.1 Bayes' Theorem and the Posterior]]
   - [[#1.2 The Posterior as an Information Compression|1.2 The Posterior as an Information Compression]]
   - [[#1.3 Bayesian vs. Frequentist Paradigm|1.3 Bayesian vs. Frequentist Paradigm]]
   - [[#1.4 Coherence of Bayesian Inference|1.4 Coherence of Bayesian Inference]]
2. [[#2. The Beta-Binomial Model|2. The Beta-Binomial Model]]
   - [[#2.1 Setup|2.1 Setup]]
   - [[#2.2 The Beta Distribution|2.2 The Beta Distribution]]
   - [[#2.3 Conjugate Posterior Update|2.3 Conjugate Posterior Update]]
   - [[#2.4 Posterior Mean as a Shrinkage Estimator|2.4 Posterior Mean as a Shrinkage Estimator]]
3. [[#3. Posterior Inference|3. Posterior Inference]]
   - [[#3.1 Credible Intervals|3.1 Credible Intervals]]
   - [[#3.2 Highest Posterior Density Intervals|3.2 Highest Posterior Density Intervals]]
   - [[#3.3 Posterior Predictive Distribution|3.3 Posterior Predictive Distribution]]
   - [[#3.4 Point Estimates and Loss Functions|3.4 Point Estimates and Loss Functions]]
4. [[#4. Decision Rules|4. Decision Rules]]
   - [[#4.1 Probability of Superiority|4.1 Probability of Superiority]]
   - [[#4.2 Expected Loss|4.2 Expected Loss]]
   - [[#4.3 The Decision Criterion|4.3 The Decision Criterion]]
5. [[#5. Prior Selection|5. Prior Selection]]
   - [[#5.1 Uniform Prior|5.1 Uniform Prior]]
   - [[#5.2 Jeffreys Prior|5.2 Jeffreys Prior]]
   - [[#5.3 Informative Priors and Effective Sample Size|5.3 Informative Priors and Effective Sample Size]]
   - [[#5.4 Prior Sensitivity Analysis|5.4 Prior Sensitivity Analysis]]
6. [[#6. Bayesian vs. Frequentist: A Structured Comparison|6. Bayesian vs. Frequentist: A Structured Comparison]]
   - [[#6.1 Interpretability of Uncertainty Statements|6.1 Interpretability of Uncertainty Statements]]
   - [[#6.2 Sequential Validity|6.2 Sequential Validity]]
   - [[#6.3 Decision Metric|6.3 Decision Metric]]
   - [[#6.4 Computational Cost|6.4 Computational Cost]]
   - [[#6.5 When to Use Which|6.5 When to Use Which]]
7. [[#7. References|7. References]]

---

## 1. The Bayesian Framework

### 1.1 Bayes' Theorem and the Posterior

Let $\theta \in \Theta$ be a parameter of interest and $x \in \mathcal{X}$ observed data. The Bayesian framework requires two ingredients:

- The *prior distribution* $\pi(\theta)$, encoding beliefs about $\theta$ before observing any data.
- The *likelihood* $p(x \mid \theta)$, the probability (density) of observing $x$ given parameter value $\theta$.

**Definition (Posterior Distribution).** The *posterior distribution* of $\theta$ given observed data $x$ is

$$\pi(\theta \mid x) = \frac{p(x \mid \theta)\, \pi(\theta)}{p(x)},$$

where $p(x) = \int_\Theta p(x \mid \theta)\, \pi(\theta)\, d\theta$ is the *marginal likelihood* (also called the *evidence*). Since $p(x)$ does not depend on $\theta$, it is a normalizing constant, and we may write the proportionality

$$\pi(\theta \mid x) \propto p(x \mid \theta)\, \pi(\theta).$$

The verbal reading of this proportionality is: **posterior is proportional to likelihood times prior**.

### 1.2 The Posterior as an Information Compression

The posterior $\pi(\theta \mid x)$ is a complete summary of all information about $\theta$ after observing $x$. It combines the prior knowledge $\pi(\theta)$ with the information in the data through the likelihood, and every subsequent inference — point estimates, intervals, predictions, decisions — is derived from $\pi(\theta \mid x)$ alone. In this sense the posterior is not just one of many outputs; it is the output of Bayesian analysis.

When new data $x'$ arrives independently of $x$, the posterior $\pi(\theta \mid x)$ becomes the new prior, and the updated posterior is $\pi(\theta \mid x, x') \propto p(x' \mid \theta)\, \pi(\theta \mid x)$. Sequential updating and batch updating are equivalent; the order of data presentation is irrelevant.

### 1.3 Bayesian vs. Frequentist Paradigm

The two frameworks differ at the level of what probability means.

In the *frequentist* paradigm, $\theta$ is a fixed but unknown constant. Probability statements refer to the behavior of procedures over hypothetical repeated experiments. The statement "$\hat{\theta}$ estimates $\theta$" is meaningful; the statement "$P(\theta \in [a, b]) = 0.95$" is not, because $\theta$ is fixed and either falls in $[a, b]$ or does not.

In the *Bayesian* paradigm, $\theta$ is modeled as a random variable with distribution $\pi(\theta)$ expressing the analyst's uncertainty about its true value. Probability statements about $\theta$ are valid by construction: "$P(\theta \in [a, b] \mid x) = 0.95$" means exactly that 95% of posterior probability mass lies in $[a, b]$.

This distinction has practical consequences for A/B testing: the frequentist confidence interval is a statement about the interval-generating procedure, not about $\theta$; the Bayesian credible interval is a direct statement about $\theta$.

### 1.4 Coherence of Bayesian Inference

**Proposition (Coherence).** Given a prior $\pi(\theta)$ and a likelihood $p(x \mid \theta)$, the posterior $\pi(\theta \mid x) \propto p(x \mid \theta)\, \pi(\theta)$ is the unique update that satisfies the axioms of probability.

*Sketch.* Any update rule that respects the product rule of probability, $P(A \cap B) = P(A \mid B) P(B)$, must yield exactly $\pi(\theta \mid x) = p(x \mid \theta)\, \pi(\theta) / p(x)$. Any other assignment of probabilities to $\theta$ after seeing $x$ would create a Dutch book — a set of bets that guarantees a loss to the agent regardless of the true value of $\theta$. This *Dutch book argument* (de Finetti, Cox) establishes that Bayesian updating is not merely one possible rule; it is the only coherent one.

---

## 2. The Beta-Binomial Model

### 2.1 Setup

In the binary conversion A/B setting, each user either converts (success) or does not (failure). Formally, let $X_1, \ldots, X_n \overset{\mathrm{iid}}{\sim} \text{Bernoulli}(p)$, where $p \in [0, 1]$ is the unknown true conversion rate. After $n$ observations, let $k = \sum_{i=1}^n X_i$ be the number of successes. The likelihood is

$$p(k \mid p) = \binom{n}{k} p^k (1-p)^{n-k}.$$

We place a prior on $p \in [0, 1]$. The natural choice — and the one that yields a tractable closed-form posterior — is the Beta distribution.

### 2.2 The Beta Distribution

**Definition (Beta Distribution).** For shape parameters $\alpha, \beta > 0$, the Beta distribution has probability density function

$$f(p \mid \alpha, \beta) = \frac{p^{\alpha - 1}(1-p)^{\beta - 1}}{B(\alpha, \beta)}, \quad p \in [0, 1],$$

where the *Beta function* $B(\alpha, \beta)$ is the normalizing constant

$$B(\alpha, \beta) = \int_0^1 p^{\alpha-1}(1-p)^{\beta-1}\, dp = \frac{\Gamma(\alpha)\,\Gamma(\beta)}{\Gamma(\alpha + \beta)}.$$

Here $\Gamma$ is the Gamma function satisfying $\Gamma(n) = (n-1)!$ for positive integers. Key moments:

$$\mathbb{E}[p] = \frac{\alpha}{\alpha + \beta}, \quad \text{Mode}(p) = \frac{\alpha - 1}{\alpha + \beta - 2} \text{ (for } \alpha, \beta > 1\text{)}, \quad \mathrm{Var}(p) = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}.$$

The Beta family is closed under many operations and — crucially — is the *conjugate prior* for the Bernoulli and Binomial likelihoods, meaning the posterior belongs to the same Beta family.

### 2.3 Conjugate Posterior Update

**Theorem (Beta-Binomial Conjugacy).** If $p \sim \text{Beta}(\alpha_0, \beta_0)$ and $X \mid p \sim \text{Binomial}(n, p)$ with observed value $X = k$, then

$$p \mid X = k \;\sim\; \text{Beta}(\alpha_0 + k,\; \beta_0 + n - k).$$

*Derivation.* By Bayes' theorem,

$$\pi(p \mid k) \propto p(k \mid p)\, \pi(p) = \binom{n}{k} p^k (1-p)^{n-k} \cdot \frac{p^{\alpha_0 - 1}(1-p)^{\beta_0 - 1}}{B(\alpha_0, \beta_0)}.$$

The binomial coefficient $\binom{n}{k}$ and the Beta normalizing constant $B(\alpha_0, \beta_0)$ do not depend on $p$, so they fold into the proportionality constant. What remains is

$$\pi(p \mid k) \propto p^k (1-p)^{n-k} \cdot p^{\alpha_0 - 1}(1-p)^{\beta_0 - 1} = p^{(\alpha_0 + k) - 1}(1-p)^{(\beta_0 + n - k) - 1}.$$

This is the kernel of a $\text{Beta}(\alpha_0 + k,\, \beta_0 + n - k)$ density. Since probability densities must integrate to one, the normalizing constant is uniquely determined, giving

$$\pi(p \mid k) = \frac{p^{(\alpha_0 + k) - 1}(1-p)^{(\beta_0 + n - k) - 1}}{B(\alpha_0 + k,\, \beta_0 + n - k)}. \quad \square$$

The update rule is strikingly simple: $\alpha \leftarrow \alpha_0 + k$ (add observed successes) and $\beta \leftarrow \beta_0 + (n - k)$ (add observed failures). The parameters $\alpha_0$ and $\beta_0$ are interpretable as pseudo-observations: $\alpha_0$ prior pseudo-successes and $\beta_0$ prior pseudo-failures, as if the prior encodes $\alpha_0 + \beta_0$ past trials.

### 2.4 Posterior Mean as a Shrinkage Estimator

Let $\alpha' = \alpha_0 + k$ and $\beta' = \beta_0 + n - k$ denote the posterior parameters. The posterior mean is

$$\mathbb{E}[p \mid k] = \frac{\alpha'}{\alpha' + \beta'} = \frac{\alpha_0 + k}{\alpha_0 + \beta_0 + n}.$$

Writing $n_0 = \alpha_0 + \beta_0$ and $\hat{p} = k/n$ (the MLE), we can decompose this as a convex combination:

$$\mathbb{E}[p \mid k] = \frac{n_0}{n_0 + n} \cdot \underbrace{\frac{\alpha_0}{n_0}}_{\text{prior mean}} + \frac{n}{n_0 + n} \cdot \underbrace{\hat{p}}_{\text{MLE}}.$$

**The posterior mean is a weighted average of the prior mean and the MLE, with weights proportional to prior pseudo-sample-size and observed sample size.** As $n \to \infty$, the data weight $n/(n_0 + n) \to 1$ and $\mathbb{E}[p \mid k] \to \hat{p} = k/n$. The prior is *washed out* by sufficient data regardless of its specification.

---

## 3. Posterior Inference

### 3.1 Credible Intervals

**Definition (Credible Interval).** A $100(1-\alpha)\%$ *credible interval* (also called a *Bayesian confidence interval* or *posterior interval*) is any interval $[L, U]$ satisfying *(here $\alpha$ is the coverage complement, distinct from the Beta shape parameters $\alpha_A', \alpha_B'$)*

$$P(L \leq \theta \leq U \mid x) = 1 - \alpha.$$

This statement has the direct probabilistic interpretation that frequentist confidence intervals cannot support: given the observed data, the probability that $\theta$ lies in $[L, U]$ is $1 - \alpha$. In the frequentist framework, $\theta$ is fixed; the interval is random, and a 95% CI is one that covers the true $\theta$ in 95% of repeated experiments. In the Bayesian framework, both the interval and the statement "$\theta \in [L, U]$ with probability $1-\alpha$" are directly meaningful.

Credible intervals are not unique: for a given coverage level, infinitely many intervals satisfy the definition. Two canonical choices arise in practice: *equal-tailed* intervals (the $\alpha/2$ and $1-\alpha/2$ quantiles of the posterior) and *highest posterior density* intervals.

### 3.2 Highest Posterior Density Intervals

**Definition (HPD Interval).** The $100(1-\alpha)\%$ *highest posterior density* (HPD) interval is the shortest interval $[L, U]$ such that $P(L \leq \theta \leq U \mid x) = 1 - \alpha$. Equivalently, it is the set

$$C_{1-\alpha} = \{\theta : \pi(\theta \mid x) \geq c_\alpha\},$$

where $c_\alpha$ is the largest constant such that $P(\theta \in C_{1-\alpha} \mid x) \geq 1 - \alpha$.

Every point inside an HPD region has higher posterior density than every point outside it. For unimodal, symmetric posteriors the HPD interval coincides with the equal-tailed interval and is centered on the mode. For asymmetric unimodal posteriors (such as Beta distributions far from symmetric), the HPD interval is shorter than the equal-tailed interval while maintaining the same coverage. *For multimodal posteriors, the HPD region need not be a connected interval.*

For the Beta distribution, the HPD interval is computed numerically (there is no closed-form expression in general), but it is easily obtained via the inverse CDF of the Beta distribution.

### 3.3 Posterior Predictive Distribution

**Definition (Posterior Predictive Distribution).** Given observed data $x$ and a new hypothetical dataset $\tilde{x}$ of size $m$, the *posterior predictive distribution* integrates out the unknown parameter:

$$p(\tilde{x} \mid x) = \int_0^1 p(\tilde{x} \mid p)\, \pi(p \mid x)\, dp.$$

For the Beta-Binomial model with posterior $p \mid x \sim \text{Beta}(\alpha', \beta')$, and new data $\tilde{X} \mid p \sim \text{Binomial}(m, p)$:

$$P(\tilde{X} = j \mid x) = \int_0^1 \binom{m}{j} p^j (1-p)^{m-j} \cdot \frac{p^{\alpha'-1}(1-p)^{\beta'-1}}{B(\alpha',\beta')}\, dp = \binom{m}{j}\frac{B(\alpha' + j,\, \beta' + m - j)}{B(\alpha',\beta')}.$$

This is the *Beta-Binomial distribution*, written $\tilde{X} \mid x \sim \text{BetaBinomial}(\alpha', \beta', m)$. Its mean is $m\alpha'/(\alpha' + \beta')$ and its variance exceeds $m\hat{p}(1-\hat{p})$ (overdispersion relative to a plain Binomial), reflecting additional uncertainty about the true $p$.

### 3.4 Point Estimates and Loss Functions

The choice of point estimate from a posterior is not arbitrary — it corresponds to minimizing expected loss under a particular loss function.

**Proposition (Bayes Estimators).** Let $\ell(\hat{\theta}, \theta)$ be a loss function and let the *posterior expected loss* of estimator $\hat{\theta}$ be $\rho(\hat{\theta}) = \mathbb{E}_{\theta \mid x}[\ell(\hat{\theta}, \theta)]$. The estimator minimizing $\rho(\hat{\theta})$ is:

| Loss function | $\ell(\hat{\theta}, \theta)$ | Optimal estimator |
|---------------|------------------------------|-------------------|
| Squared error | $(\hat{\theta} - \theta)^2$ | Posterior mean |
| Absolute error | $\lvert \hat{\theta} - \theta \rvert$ | Posterior median |
| 0-1 loss | $\mathbf{1}[\hat{\theta} \neq \theta]$ | Posterior mode (MAP) |

*Sketch for squared error.* Expanding $\mathbb{E}[(\hat{\theta} - \theta)^2 \mid x] = (\hat{\theta} - \mathbb{E}[\theta \mid x])^2 + \mathrm{Var}(\theta \mid x)$, the variance term is constant in $\hat{\theta}$, so the minimum is achieved at $\hat{\theta} = \mathbb{E}[\theta \mid x]$.

---

## 4. Decision Rules

### 4.1 Probability of Superiority

The primary Bayesian criterion for concluding that variant B outperforms variant A is the *probability of superiority*:

$$P(p_B > p_A \mid \text{data}) = \int_0^1 \int_0^{p_B} \pi(p_A \mid \text{data})\, \pi(p_B \mid \text{data})\, dp_A\, dp_B.$$

Since $p_A$ and $p_B$ have independent posteriors $\text{Beta}(\alpha_A', \beta_A')$ and $\text{Beta}(\alpha_B', \beta_B')$, this integral does not factor into a simple closed form for general parameters, but two approaches are available.

**Monte Carlo estimate.** Draw $S$ independent samples:

$$p_A^{(s)} \sim \text{Beta}(\alpha_A', \beta_A'), \quad p_B^{(s)} \sim \text{Beta}(\alpha_B', \beta_B'), \quad s = 1, \ldots, S.$$

Then

$$P(p_B > p_A \mid \text{data}) \approx \frac{1}{S} \sum_{s=1}^S \mathbf{1}[p_B^{(s)} > p_A^{(s)}].$$

By the law of large numbers this converges to the true probability as $S \to \infty$. For $S = 10{,}000$ the Monte Carlo error is on the order of $1/\sqrt{S} \approx 0.01$.

**Closed-form result.** Integrating the joint density analytically yields (Miller, 2015):

$$P(p_B > p_A \mid \text{data}) = \sum_{i=0}^{\alpha_B' - 1} \frac{B\!\left(\alpha_A' + i,\; \beta_A' + \beta_B'\right)}{(\beta_B' + i)\, B(1 + i,\, \beta_B')\, B(\alpha_A',\, \beta_A')},$$

where $B(\cdot, \cdot)$ denotes the Beta function. This sum has $\alpha_B'$ terms, each computable via $\log \Gamma$ evaluations, and avoids any simulation variance.

*Derivation sketch.* Fix $p_B$ and integrate out $p_A$: $\int_0^{p_B} \pi(p_A)\, dp_A = I_{p_B}(\alpha_A', \beta_A')$, the regularized incomplete Beta function. Then integrate over $p_B$ weighted by $\pi(p_B)$. A series expansion of $I_{p_B}(\alpha_A', \beta_A')$ in powers of $p_B$ and term-by-term integration against the $\text{Beta}(\alpha_B', \beta_B')$ density yields the sum above.

### 4.2 Expected Loss

The probability of superiority alone does not encode the *magnitude* of the risk of a wrong decision. A complementary criterion is the *expected loss*, also called the expected opportunity cost.

**Definition (Expected Loss for Deploying B).** Suppose we deploy B. If B is in fact inferior to A (i.e., $p_A > p_B$), the opportunity cost is $p_A - p_B$ — the conversion rate we sacrificed by choosing the wrong variant. The expected loss is

$$\mathcal{L}_B = \mathbb{E}[\max(p_A - p_B,\, 0) \mid \text{data}].$$

*Derivation.* With independent Beta posteriors,

$$\mathcal{L}_B = \int_0^1 \int_0^1 \max(p_A - p_B, 0)\, \pi(p_A)\, \pi(p_B)\, dp_A\, dp_B.$$

Restrict to the region $p_A > p_B$ (where the max is positive) and split:

$$\mathcal{L}_B = \int_0^1 \int_{p_B}^1 (p_A - p_B)\, \pi(p_A)\, dp_A\, \pi(p_B)\, dp_B = \underbrace{\int_0^1\!\!\int_{p_B}^1 p_A\, \pi(p_A)\, dp_A\, \pi(p_B)\, dp_B}_{T_1} - \underbrace{\int_0^1 p_B \pi(p_B)\!\int_{p_B}^1 \pi(p_A)\, dp_A\, dp_B}_{T_2}.$$

Each term involves the incomplete Beta function $I_x(\alpha, \beta) = B(x;\alpha,\beta)/B(\alpha,\beta)$ and the first moment of a truncated Beta. Evaluating each term gives:

$$T_1 = \frac{\alpha_A'}{\alpha_A' + \beta_A'} \sum_{i=0}^{\alpha_B'-1} \frac{B(\alpha_A'+1+i,\; \beta_A'+\beta_B')}{(\beta_B'+i)\,B(1+i,\,\beta_B')\,B(\alpha_A'+1,\,\beta_A')},$$

$$T_2 = \frac{\alpha_B'}{\alpha_B' + \beta_B'} \sum_{i=0}^{\alpha_B'-1} \frac{B(\alpha_A'+i,\; \beta_A'+\beta_B')}{(\beta_B'+i)\,B(1+i,\,\beta_B')\,B(\alpha_A',\,\beta_A')},$$

so $\mathcal{L}_B = T_1 - T_2$. The derivation of $T_1$ follows by writing $\mathbb{E}[p_A \mathbf{1}[p_A > p_B]] = \frac{\alpha_A'}{\alpha_A'+\beta_A'} \cdot P(p_A > p_B;\, \alpha_A' \to \alpha_A'+1)$, where the right-hand factor is a probability of superiority computed with $\text{Beta}(\alpha_A'+1, \beta_A')$ in place of $\text{Beta}(\alpha_A', \beta_A')$; applying the closed-form superiority formula from Section 4.1 to that shifted distribution yields the sum above. The expression for $T_2$ follows symmetrically. *These sums converge for integer-valued Beta shape parameters, in which case each term is a ratio of Gamma function evaluations.*

In practice the Monte Carlo estimator

$$\mathcal{L}_B \approx \frac{1}{S}\sum_{s=1}^S \max(p_A^{(s)} - p_B^{(s)},\, 0)$$

is simpler to implement and sufficiently accurate for experimental decisions.

Similarly, the expected loss for deploying A (i.e., the risk of A being inferior) is

$$\mathcal{L}_A = \mathbb{E}[\max(p_B - p_A,\, 0) \mid \text{data}].$$

### 4.3 The Decision Criterion

In practice, both the probability of superiority and the expected loss are used jointly:

**Definition (Decision Rule).** Deploy variant B when both:
1. $P(p_B > p_A \mid \text{data}) > 1 - \delta$ for some threshold $\delta$ (e.g., $\delta = 0.05$, so 95% posterior probability), **and**
2. $\mathcal{L}_B < \epsilon$ for a business-meaningful loss threshold $\epsilon$ (e.g., $\epsilon = 0.001$, representing 0.1 percentage points of conversion rate).

Criterion 1 guards against high-uncertainty decisions. Criterion 2 guards against declaring a winner when the posterior mass on "B is worse" is small but the magnitude of that inferiority could be large. Together they provide a calibrated, continuous-valued decision rule without binary rejection logic.

*Importantly,* unlike frequentist tests, this rule can be evaluated at any time during the experiment — see Section 6.2.

---

## 5. Prior Selection

### 5.1 Uniform Prior

**Definition (Uniform Prior).** The uniform prior is $\text{Beta}(1, 1)$, with density $f(p) = 1$ for $p \in [0, 1]$. It assigns equal density to every conversion rate a priori.

With $\text{Beta}(1, 1)$ prior and $k$ successes in $n$ trials, the posterior is $\text{Beta}(1 + k, 1 + n - k)$ and the posterior mean is

$$\mathbb{E}[p \mid k] = \frac{1 + k}{2 + n}.$$

This is the *Laplace estimator*, which avoids the edge cases $k = 0$ or $k = n$ that make the MLE $k/n$ degenerate (0 or 1, respectively). The effective pseudo-sample-size of the uniform prior is $\alpha_0 + \beta_0 = 2$, a minimal regularization.

*The uniform prior is not as "uninformative" as it appears*: it places higher prior probability on conversion rates near 0.5 than a Jeffreys prior does, and it is not invariant to reparameterization of $p$.

### 5.2 Jeffreys Prior

**Definition (Jeffreys Prior).** The *Jeffreys prior* for a model parametrized by $\theta$ is

$$\pi_J(\theta) \propto \sqrt{\det I(\theta)},$$

where $I(\theta)$ is the *Fisher information matrix* (scalar $I(\theta)$ for one-dimensional $\theta$).

*Derivation for the Bernoulli model.* For $X \sim \text{Bernoulli}(p)$, the log-likelihood for a single observation is

$$\ell(p; x) = x \log p + (1-x)\log(1-p).$$

The Fisher information is

$$I(p) = -\mathbb{E}\!\left[\frac{\partial^2 \ell}{\partial p^2}\right] = -\mathbb{E}\!\left[-\frac{X}{p^2} - \frac{1-X}{(1-p)^2}\right] = \frac{\mathbb{E}[X]}{p^2} + \frac{1 - \mathbb{E}[X]}{(1-p)^2} = \frac{p}{p^2} + \frac{1-p}{(1-p)^2} = \frac{1}{p} + \frac{1}{1-p} = \frac{1}{p(1-p)}.$$

The Jeffreys prior is therefore

$$\pi_J(p) \propto \sqrt{I(p)} = \frac{1}{\sqrt{p(1-p)}} = p^{-1/2}(1-p)^{-1/2} = p^{1/2 - 1}(1-p)^{1/2 - 1},$$

which is the kernel of $\text{Beta}(1/2, 1/2)$.

**The Jeffreys prior for a Bernoulli likelihood is $\text{Beta}(1/2, 1/2)$.**

*Why this prior is preferred over the uniform.* The key property is *reparameterization invariance*. If $\phi = g(p)$ is a smooth bijection, then by the change-of-variables formula the Jeffreys prior on $\phi$ is $\pi_J(\phi) \propto \sqrt{I_\phi(\phi)}$ where $I_\phi(\phi) = I(p) \cdot (dp/d\phi)^2$. Because $\pi_J(p) \propto \sqrt{I(p)}$, after the change of variables one obtains exactly $\pi_J(\phi) \propto \sqrt{I_\phi(\phi)}$: the Jeffreys prior on $\phi$ is the same as the prior one would have derived by working in the $\phi$ parameterization from the start. No other prior satisfies this invariance in general.

The $\text{Beta}(1/2, 1/2)$ density has a U-shape on $[0,1]$, placing more weight near 0 and 1 than the uniform prior. In A/B testing, where conversion rates can legitimately be very low or very high depending on the funnel stage, this behavior is often more realistic than the uniform.

### 5.3 Informative Priors and Effective Sample Size

When historical data is available — for example, from previous experiments on a similar product feature — the prior should encode that information. If historical records suggest the conversion rate is approximately $\mu_0$ with some confidence, one can set

$$\alpha_0 = \mu_0 \cdot n_0, \quad \beta_0 = (1 - \mu_0) \cdot n_0,$$

where $n_0 = \alpha_0 + \beta_0$ is the *effective sample size* of the prior: it is interpretable as the number of hypothetical prior observations carrying equivalent weight to $\alpha_0$ pseudo-successes and $\beta_0$ pseudo-failures.

The choice of $n_0$ controls how strongly the prior resists updates from new data. A prior with $n_0 = 100$ is equivalent in influence to 100 observed trials; it will dominate the likelihood until the experiment accumulates substantially more than 100 observations. Setting $n_0$ too large encodes overconfidence; setting it too small is nearly equivalent to an uninformative prior.

### 5.4 Prior Sensitivity Analysis

No single prior choice is universally correct. Best practice is to run the analysis under multiple priors — at minimum, the Jeffreys prior, the uniform prior, and any informative prior encoding domain knowledge — and verify that the posterior conclusions are qualitatively similar. When posteriors are robust to prior choice, the data is informative enough to dominate. *This robustness check only holds for large $n$; for small experiments, the prior can substantially alter conclusions, and the choice should be documented and justified explicitly.*

---

## 6. Bayesian vs. Frequentist: A Structured Comparison

### 6.1 Interpretability of Uncertainty Statements

The frequentist 95% confidence interval $[\hat{L}, \hat{U}]$ satisfies $P(\hat{L} \leq \theta \leq \hat{U}) = 0.95$, where the probability is over repeated sampling of the interval-generating procedure — not over $\theta$. It is incorrect to say "there is a 95% probability that $\theta \in [\hat{L}, \hat{U}]$" given a single observed interval (because $\theta$ is fixed).

The Bayesian 95% credible interval $[L, U]$ satisfies $P(\theta \in [L, U] \mid x) = 0.95$ exactly as written. Communicating uncertainty to non-statistical stakeholders is therefore more natural with credible intervals, and the statement maps directly to the intuitive meaning people assign to "95% confidence."

### 6.2 Sequential Validity

*Bayesian tests are valid at any stopping time.* Because the posterior $\pi(\theta \mid x)$ is computed via Bayes' theorem regardless of when or why the analyst chose to look at the data, the posterior probability $P(p_B > p_A \mid \text{data})$ is a valid probability statement at any point during data collection. There is no inflation of error analogous to the frequentist *peeking problem* (see `sequential-and-adaptive.md`).

Frequentist p-values are valid only when the stopping rule is specified in advance and the sample size is fixed. An analyst who peeks at a frequentist p-value, finds $p = 0.06$, and continues collecting data has implicitly changed the test's Type I error rate without correcting for it. The Bayesian updates through the same mechanism whether the analyst peeks or not: the posterior is just the posterior.

*Caveat: this sequential validity is relative to the prior. If the prior is severely misspecified, the posterior can be misleading for any number of observations.*

### 6.3 Decision Metric

Frequentist hypothesis testing produces a binary decision: reject $H_0$ or fail to reject $H_0$ at a pre-specified significance level $\alpha$. This binary framing discards information about how far $p$-values are from the threshold and conflates statistical significance with practical importance.

The Bayesian expected loss criterion $\mathcal{L}_B < \epsilon$ is continuous. It directly quantifies the business cost of a wrong decision in the units of the metric (e.g., percentage points of conversion rate). Stopping when $\mathcal{L}_B < 0.001$ means the analyst is comfortable that, under the posterior, the expected cost of deploying B when A is better is less than 0.1 percentage points. This framing is directly actionable.

### 6.4 Computational Cost

For the Beta-Binomial model, Bayesian inference is computationally trivial: the posterior update is two additions ($\alpha \leftarrow \alpha + k$, $\beta \leftarrow \beta + n - k$). Evaluating $P(p_B > p_A)$ and $\mathcal{L}_B$ requires either a short loop over Beta function evaluations (closed form) or $S$ samples from two Beta distributions (Monte Carlo).

For models beyond the conjugate case — e.g., continuous revenue metrics with non-conjugate priors, or hierarchical models with treatment heterogeneity — Bayesian inference requires *Markov Chain Monte Carlo* (MCMC) or *variational inference* (VI), which can be orders of magnitude more expensive than frequentist point estimates and standard errors.

### 6.5 When to Use Which

| Criterion | Prefer Frequentist | Prefer Bayesian |
|-----------|-------------------|-----------------|
| Regulatory environment | Fixed Type I error required (FDA, clinical trials) | Industry product experiments |
| Stopping rule | Pre-specified, fixed $n$ | Need to stop early, sequential monitoring |
| Prior knowledge | None available or distrust prior | Strong historical data, informative prior |
| Audience | Statistical reviewers | Product managers, business stakeholders |
| Metric type | Continuous, well-understood | Rare binary events, sparse data |
| Multiple experiments | FWER/FDR correction needed | Hierarchical prior shares strength |

---

## 7. References

| Reference Name | Brief Summary | Link to Reference |
|----------------|---------------|-------------------|
| Gelman, Carlin, Stern, Dunson, Vehtari, Rubin — Bayesian Data Analysis (3rd ed.) | Comprehensive graduate textbook on Bayesian inference; covers posterior summarization, conjugate models, prior selection, and HPD intervals in depth | [BDA3 (online draft)](http://www.stat.columbia.edu/~gelman/book/) |
| VanderPlas (2014) — Frequentism and Bayesianism: A Python-driven Primer | Accessible comparison of the frequentist and Bayesian paradigms with worked examples and code; clarifies the credible interval vs. confidence interval distinction | [arXiv:1411.5018](https://arxiv.org/abs/1411.5018) |
| Miller, E. (2015) — Formulas for Bayesian A/B Testing | Derives the closed-form sum for $P(p_B > p_A)$ when posteriors are Beta; also covers count data (Poisson/Gamma) and three-way tests | [evanmiller.org](https://www.evanmiller.org/bayesian-ab-testing.html) |
| Thompson, W. R. (1933) — On the Likelihood That One Unknown Probability Exceeds Another | Foundational paper introducing what is now called Thompson Sampling; the original Bayesian treatment of comparing two unknown Bernoulli probabilities | [Biometrika 25(3-4):285–294](https://academic.oup.com/biomet/article-abstract/25/3-4/285/200862) |
| VWO SmartStats Technical Whitepaper | Industry whitepaper describing VWO's Bayesian A/B testing engine including expected loss criterion and stopping rules | [vwo.com/downloads](https://vwo.com/downloads/VWO_SmartStats_technical_whitepaper.pdf) |
| Wikipedia — Jeffreys Prior | Statement of the reparameterization invariance theorem and examples for common likelihoods including Bernoulli | [en.wikipedia.org/wiki/Jeffreys_prior](https://en.wikipedia.org/wiki/Jeffreys_prior) |
| Wikipedia — Conjugate Prior | Table of conjugate prior families including Beta-Binomial; derivation sketches | [en.wikipedia.org/wiki/Conjugate_prior](https://en.wikipedia.org/wiki/Conjugate_prior) |
| Jordan (2010) — Lecture 7: Jeffreys Priors and Reference Priors (UC Berkeley) | Lecture notes deriving the Jeffreys prior from Fisher information and proving reparameterization invariance | [Berkeley EECS 260](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/lectures/lecture7.pdf) |
| Duke STA 114 — The Jeffreys Prior | Short technical note deriving Jeffreys prior, proving invariance, and discussing its use in binomial models | [stat.duke.edu](https://www2.stat.duke.edu/courses/Fall11/sta114/jeffreys.pdf) |
| Variance Explained — Understanding Bayesian A/B Testing | Applied walkthrough using baseball batting statistics; illustrates credible intervals, posterior updating, and sequential stopping in a concrete domain | [varianceexplained.org](http://varianceexplained.org/r/bayesian_ab_baseball/) |
