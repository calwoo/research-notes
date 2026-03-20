# A/B Testing: Solutions

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

**Problem 1.** Prove that the difference-in-means estimator $\hat{\tau} = \bar{Y}_1 - \bar{Y}_0$ is unbiased for the ATE under random assignment.

**Key insight:** Random assignment $(Y_i(0), Y_i(1)) \perp\!\!\!\perp T_i$ makes conditioning on $T_i = 1$ uninformative about $Y_i(1)$, so the treated subsample is a representative draw from the population with respect to potential outcomes.

**Sketch:**
$$\mathbb{E}[\bar{Y}_1] = \mathbb{E}\!\left[\frac{1}{n_1}\sum_{i:T_i=1} Y_i(1)\right] = \mathbb{E}[Y_i(1) \mid T_i = 1] = \mathbb{E}[Y_i(1)]$$
The last step uses $T_i \perp\!\!\!\perp Y_i(1)$. Symmetrically $\mathbb{E}[\bar{Y}_0] = \mathbb{E}[Y_i(0)]$, giving $\mathbb{E}[\hat{\tau}] = \mathbb{E}[Y_i(1)] - \mathbb{E}[Y_i(0)] = \tau$. The result holds for both complete and Bernoulli randomization — both satisfy the independence condition, which is the only property used.

---

### Problem 2: Selection Bias Decomposition under Observational Data

**Problem 2.** Decompose the naive observational estimand $\mathbb{E}[Y \mid T=1] - \mathbb{E}[Y \mid T=0]$ into the ATT plus a selection bias term, and show both terms vanish to ATE under random assignment.

**Key insight:** The fundamental decomposition arises by adding and subtracting $\mathbb{E}[Y_i(0) \mid T_i = 1]$; random assignment forces the added term to zero, collapsing ATT to ATE.

**Sketch:**
$$\mathbb{E}[Y \mid T=1] - \mathbb{E}[Y \mid T=0] = \mathbb{E}[Y(1) \mid T=1] - \mathbb{E}[Y(0) \mid T=0]$$
Add and subtract $\mathbb{E}[Y(0) \mid T=1]$:
$$= \underbrace{\mathbb{E}[Y(1) - Y(0) \mid T=1]}_{\text{ATT}} + \underbrace{\mathbb{E}[Y(0) \mid T=1] - \mathbb{E}[Y(0) \mid T=0]}_{\text{selection bias}}$$
Under $(Y(0),Y(1)) \perp\!\!\!\perp T$: $\mathbb{E}[Y(0)\mid T=1] = \mathbb{E}[Y(0)\mid T=0]$, so selection bias $= 0$ and ATT $=$ ATE. A concrete example: let $Y_i(0) = Z_i$, $Y_i(1) = Z_i + 1$, $T_i = \mathbf{1}[Z_i > 0]$ with $Z_i \sim \mathcal{N}(0,1)$; then selection bias $= \mathbb{E}[Z_i \mid Z_i > 0] - \mathbb{E}[Z_i] = \sqrt{2/\pi} > 0$.

---

### Problem 3: P-value Uniformity under the Null

**Problem 3.** Prove that the p-value $P = 1 - F_{H_0}(W_n)$ is $\mathrm{Uniform}(0,1)$ under a continuous null distribution, and characterize the inflation of FWER for $m$ independent tests.

**Key insight:** The probability integral transform converts any continuous CDF-applied random variable to a uniform — this is purely a property of continuous CDFs, independent of the underlying test.

**Sketch:**
For $U = F_{H_0}(W_n)$: $P(U \leq u) = P(F_{H_0}(W_n) \leq u) = P(W_n \leq F_{H_0}^{-1}(u)) = F_{H_0}(F_{H_0}^{-1}(u)) = u$, so $U \sim \text{Uniform}(0,1)$. Continuity is required to invert $F_{H_0}$. Then $P = 1 - U$ is also Uniform. For $m$ independent tests: $P(\text{no false rejections}) = (1-\alpha)^m$, so $P(\text{at least one}) = 1-(1-\alpha)^m \approx m\alpha$ by first-order Taylor expansion for small $\alpha$. For discrete $W_n$: $F_{H_0}$ has jumps, so $F_{H_0}(F_{H_0}^{-1}(u)) \geq u$, giving $P(U \leq u) \leq u$ — the p-value is stochastically larger than uniform, making the test conservative.

---

### Problem 4: Power Function for the Two-Sided Z-test

**Problem 4.** Derive the power function $\pi(\delta) = \Phi(\lambda - z_{\alpha/2}) + \Phi(-\lambda - z_{\alpha/2})$ for the two-sided z-test, and verify its properties at $\delta = 0$ and $\delta \to \infty$.

**Key insight:** Under the alternative, the test statistic $Z$ gains a non-centrality $\lambda = \delta\sqrt{n/(2\sigma^2)}$; the power is $\Phi(\lambda - z_{\alpha/2})$ after dropping the negligible lower tail.

**Sketch:**
Under $H_1: \tau = \delta$: $Z = \hat{\tau}/\sqrt{2\sigma^2/n} \sim \mathcal{N}(\lambda, 1)$ with $\lambda = \delta\sqrt{n/(2\sigma^2)}$.
$$\pi(\delta) = P(|Z| > z_{\alpha/2}) = \Phi(\lambda - z_{\alpha/2}) + \Phi(-\lambda - z_{\alpha/2})$$
The second term $\Phi(-\lambda - z_{\alpha/2}) < \Phi(-z_{\alpha/2}) = \alpha/2$ and is negligible for practical $\alpha$. Properties: (i) $\pi(0) = \Phi(-z_{\alpha/2}) + \Phi(-z_{\alpha/2}) = 2(1-\Phi(z_{\alpha/2})) = \alpha$. (ii) $\lambda \to \infty$ as $\delta \to \infty$, so $\Phi(\lambda - z_{\alpha/2}) \to 1$. (iii) $\lambda$ enters symmetrically through $|\delta|$, so $\pi(\delta) = \pi(-\delta)$. At $\delta = \text{MDE}$: set $\pi(\text{MDE}) = 1-\beta$, giving $\lambda - z_{\alpha/2} = z_\beta$, so $\text{MDE} = (z_{\alpha/2}+z_\beta)\sqrt{2\sigma^2/n}$.

---

### Problem 5: MDE Derivation for a Two-Sample Z-test

**Problem 5.** Derive the minimum detectable effect $\mathrm{MDE} = (z_{\alpha/2}+z_\beta)\sqrt{2\sigma^2/n}$ from the power equation, and compute how MDE scales with sample size for binary outcomes under rare events.

**Key insight:** The MDE is the inverse of the sample size formula — fixing power and solving for effect size rather than fixing effect size and solving for $n$.

**Sketch:**
Power equation: $\Phi(\lambda - z_{\alpha/2}) = 1-\beta$ where $\lambda = \delta/\sqrt{2\sigma^2/n}$. So $\lambda = z_{\alpha/2} + z_\beta$, giving $\delta = (z_{\alpha/2}+z_\beta)\sqrt{2\sigma^2/n}$. The factor 2 in $2\sigma^2/n$ arises from $\text{Var}(\hat{\tau}) = \sigma^2/n + \sigma^2/n = 2\sigma^2/n$ with equal-variance, equal-allocation groups. Inverting: $n = (z_{\alpha/2}+z_\beta)^2 \cdot 2\sigma^2/\delta^2$; doubling $\delta$ multiplies the denominator by 4, dividing $n$ by 4. For binary outcomes: $\sigma_k^2 = p_k(1-p_k)$; for rare events $p_0 \ll 1$ and $\delta \ll p_0$: $p_0(1-p_0) + p_1(1-p_1) \approx 2p_0$, so $n \approx (z_{\alpha/2}+z_\beta)^2 \cdot 2p_0/\delta^2 \propto p_0/\delta^2 \propto p_0^{-1}$ for fixed relative lift $\ell = \delta/p_0$.

---

### Problem 6: Sample Size Formula: All Four Steps

**Problem 6.** Derive the two-sample z-test sample size formula $n = (z_{\alpha/2}+z_\beta)^2(\sigma_A^2+\sigma_B^2)/\delta^2$ step by step, justifying each algebraic move and the final ceiling operation.

**Key insight:** The four steps are: compute the non-centrality, express power as a standard normal probability, set it to $1-\beta$ and solve, then ceiling to an integer — the ceiling matters most near the MDE boundary where power is sensitive to $n$.

**Sketch:**
**Step 1.** $Z = \hat{\tau}/\sqrt{(\sigma_A^2+\sigma_B^2)/n}$. Under $H_1$: $\mathbb{E}[Z] = \delta/\sqrt{(\sigma_A^2+\sigma_B^2)/n} = \delta\sqrt{n}/\sqrt{\sigma_A^2+\sigma_B^2} =: \mu_Z$.

**Step 2.** $1-\beta \approx P(Z > z_{\alpha/2}) = \Phi(\mu_Z - z_{\alpha/2})$. The lower tail $P(Z < -z_{\alpha/2})$ is $O(\Phi(-2z_{\alpha/2})) = O(\alpha^2)$, negligible for $\alpha \leq 0.05$.

**Step 3.** $\Phi(\mu_Z - z_{\alpha/2}) = 1-\beta \Rightarrow \mu_Z = z_{\alpha/2}+z_\beta$. Squaring: $n = (z_{\alpha/2}+z_\beta)^2(\sigma_A^2+\sigma_B^2)/\delta^2$.

**Step 4.** $n$ must be an integer; $\lceil n \rceil$ ensures the constraint is satisfied. The ceiling matters most when $n$ from the formula is non-integer and close to a whole number where power transitions sharply — this is most pronounced near the MDE boundary.

---

### Problem 7: Optimal Unequal Allocation Ratio

**Problem 7.** Find the allocation ratio $k^* = n_B/n_A$ that minimizes $\mathrm{Var}(\hat{\tau})$ for fixed total sample size $N$, and compute the variance reduction relative to equal allocation when $\sigma_A = 1$ and $\sigma_B = 3$.

**Key insight:** For fixed total $N$, $\text{Var}(\hat{\tau})$ is convex in $k$ and minimized by setting $d\text{Var}/dk = 0$, yielding $k^* = \sigma_B/\sigma_A$ — allocate more to the higher-variance arm.

**Sketch:**
$$\text{Var}(\hat{\tau}) = \frac{1+k}{N}\left(\sigma_A^2 + \frac{\sigma_B^2}{k}\right)$$
Differentiating: $d\text{Var}/dk = N^{-1}[\sigma_A^2 + \sigma_B^2/k - \sigma_B^2(1+k)/k^2] = 0 \Rightarrow \sigma_A^2 k^2 = \sigma_B^2 \Rightarrow k^* = \sigma_B/\sigma_A$.

For $\sigma_A=1$, $\sigma_B=3$: $k^*=3$, so $n_B = 3n_A$. With $N = n_A + 3n_A = 4n_A$: $\text{Var}^* = (4/N)(1 + 9/3) = (4/N)\cdot 4 = 16/N \cdot (1/4) = 4/N \cdot 1 = ...$; direct computation gives $\text{Var}^* = (1+3)/N \cdot(1 + 9/3) = (4/N)(1+3) = 16/N$. Under $k=1$: $\text{Var} = (2/N)(1+9) = 20/N$. Reduction factor: $16/20 = 0.8$; unequal allocation saves 20\%.

---

### Problem 8: Welch-Satterthwaite Degrees of Freedom via Moment Matching

**Problem 8.** Derive the Welch-Satterthwaite effective degrees of freedom $\nu$ by approximating the distribution of $s_A^2/n_A + s_B^2/n_B$ with a scaled chi-squared via moment matching.

**Key insight:** Approximating $U + V$ by $c\chi^2_\nu$ requires matching two moments; dividing the variance equation by the square of the mean equation eliminates $c$ and directly yields $\nu$.

**Sketch:**
$(n_A-1)s_A^2/\sigma_A^2 \sim \chi^2_{n_A-1}$, so $\mathbb{E}[U] = \sigma_A^2/n_A$ and $\text{Var}(U) = 2\sigma_A^4/(n_A^2(n_A-1))$.

Matching: $c\nu = \mathbb{E}[U]+\mathbb{E}[V]$ and $2c^2\nu = \text{Var}(U)+\text{Var}(V)$ (independence). Dividing: $2/\nu = (\text{Var}(U)+\text{Var}(V))/(\mathbb{E}[U]+\mathbb{E}[V])^2$. Substituting:
$$\nu = \frac{(s_A^2/n_A + s_B^2/n_B)^2}{(s_A^2/n_A)^2/(n_A-1) + (s_B^2/n_B)^2/(n_B-1)}$$
Bounds: by AM-QM, $\nu \geq \min(n_A-1, n_B-1)$; equality $\sigma_A^2/n_A = \sigma_B^2/n_B$ gives $\nu = n_A+n_B-2$.

---

### Problem 9: ANOVA Sum of Squares Decomposition and F-statistic Distribution

**Problem 9.** Prove the identity $\mathrm{SST} = \mathrm{SSB} + \mathrm{SSW}$, establish the degrees of freedom for each term, and derive the null distribution $F \sim F(K-1, N-K)$ via Cochran's theorem.

**Key insight:** The cross-term in the expansion of $\|Y - \bar{Y}\|^2$ vanishes because the within-group deviations $Y_{ik} - \bar{Y}_k$ sum to zero; independence of SSB and SSW under normality follows from Cochran's theorem on quadratic forms.

**Sketch:**
Write $Y_{ik}-\bar{Y} = (Y_{ik}-\bar{Y}_k) + (\bar{Y}_k-\bar{Y})$, square, sum over all $i,k$. The cross-term is $2\sum_k(\bar{Y}_k-\bar{Y})\sum_i(Y_{ik}-\bar{Y}_k) = 0$ since $\sum_i(Y_{ik}-\bar{Y}_k) = 0$ for each $k$. SSB has $K-1$ df: $K$ group means constrained to satisfy $\sum_k n_k\bar{Y}_k = N\bar{Y}$, leaving $K-1$ free. Under $H_0$ and normality: $\text{SSB}/\sigma^2 \sim \chi^2_{K-1}$ (balanced case by direct chi-squared argument on group means); $\text{SSW}/\sigma^2 = \sum_k(n_k-1)s_k^2/\sigma^2 \sim \chi^2_{N-K}$ always (this is an unbiased estimate of $\sigma^2$ regardless of $H_0$). Cochran's theorem: SSB and SSW are independent (orthogonal quadratic forms in a normal vector), giving $F = (\chi^2_{K-1}/(K-1))/(\chi^2_{N-K}/(N-K)) \sim F(K-1,N-K)$ under $H_0$.

---

### Problem 10: Beta-Binomial Conjugate Update

**Problem 10.** Derive the posterior $p \mid k \sim \mathrm{Beta}(\alpha_0 + k,\, \beta_0 + n - k)$ from a $\mathrm{Beta}(\alpha_0, \beta_0)$ prior and binomial likelihood, and show the posterior mean converges to the MLE as $n \to \infty$.

**Key insight:** The Beta kernel absorbs both the likelihood and prior into a single power-form expression; the unique normalizing constant of a Beta density identifies the posterior parameters.

**Sketch:**
$$\pi(p \mid k) \propto p^k(1-p)^{n-k} \cdot p^{\alpha_0-1}(1-p)^{\beta_0-1} = p^{(\alpha_0+k)-1}(1-p)^{(\beta_0+n-k)-1}$$
This is the kernel of $\text{Beta}(\alpha_0+k, \beta_0+n-k)$. Since the Beta function $B(\alpha_0+k, \beta_0+n-k)$ is the unique normalizing constant, the posterior is exactly $\text{Beta}(\alpha_0+k, \beta_0+n-k)$.

Posterior mean: $(\alpha_0+k)/(\alpha_0+\beta_0+n) = (n_0/(n_0+n)) \cdot (\alpha_0/n_0) + (n/(n_0+n)) \cdot (k/n)$. As $n\to\infty$: weight $n/(n_0+n) \to 1$, so posterior mean $\to k/n = \hat{p}$ regardless of prior.

---

### Problem 11: Jeffreys Prior from Fisher Information

**Problem 11.** Compute the Jeffreys prior for a Bernoulli parameter $p$, identify it as $\mathrm{Beta}(1/2, 1/2)$, and verify its invariance under the logit reparameterization.

**Key insight:** The Fisher information for Bernoulli is $1/(p(1-p))$, so $\pi_J(p) \propto (p(1-p))^{-1/2}$ — the $\text{Beta}(1/2,1/2)$ density; invariance follows because the Jacobian squared cancels precisely with the information ratio under reparameterization.

**Sketch:**
$\ell(p;x) = x\log p + (1-x)\log(1-p)$. Second derivative: $\partial^2\ell/\partial p^2 = -X/p^2 - (1-X)/(1-p)^2$. Taking expectation: $I(p) = p/p^2 + (1-p)/(1-p)^2 = 1/p + 1/(1-p) = 1/(p(1-p))$.

$\pi_J(p) \propto \sqrt{I(p)} = p^{-1/2}(1-p)^{-1/2} = p^{1/2-1}(1-p)^{1/2-1} = \text{Beta}(1/2,1/2)$ kernel.

Reparameterization: $I_\phi(\phi) = I(p(g^{-1}(\phi)))\cdot(dp/d\phi)^2$. Then $\pi_J(\phi) = \pi_J(p)|dp/d\phi| \propto \sqrt{I(p)}|dp/d\phi| = \sqrt{I(p)(dp/d\phi)^2} = \sqrt{I_\phi(\phi)}$.

For logit $\phi = \log(p/(1-p))$: $dp/d\phi = e^\phi/(1+e^\phi)^2 = p(1-p)$, so $I_\phi(\phi) = 1/(p(1-p)) \cdot p^2(1-p)^2 = p(1-p)$, giving $\pi_J(\phi) \propto \sqrt{p(1-p)} = (1+e^\phi)^{-1}(1+e^{-\phi})^{-1}$ — the standard logistic density.

---

### Problem 12: Posterior Mean as Weighted Average

**Problem 12.** Show that the Beta-Binomial posterior mean is a convex combination of the prior mean and the MLE with weights proportional to pseudo-sample-size and data size, and bound the gap to the MLE.

**Key insight:** The posterior mean is a convex combination of prior mean and MLE with weights proportional to prior pseudo-sample-size and data size; the prior contribution shrinks as $O(n^{-1})$.

**Sketch:**
From $p\mid k \sim \text{Beta}(\alpha_0+k, \beta_0+n-k)$: posterior mean $= (\alpha_0+k)/(\alpha_0+\beta_0+n) = (n_0\mu_0 + n\hat{p})/(n_0+n)$ where $\mu_0 = \alpha_0/n_0$. This is a convex combination with weights $n_0/(n_0+n)$ and $n/(n_0+n)$.

Since $n_0/(n_0+n) \in (0,1)$, the posterior mean lies strictly between $\mu_0$ and $\hat{p}$, so $|\mathbb{E}[p\mid k] - \hat{p}| = (n_0/(n_0+n))|\mu_0 - \hat{p}| < |\mu_0 - \hat{p}|$. Equals $\hat{p}$ only if $n_0 = 0$ (improper limit) or $\mu_0 = \hat{p}$. The gap is $n_0(\mu_0-\hat{p})/(n_0+n) = O(n^{-1})$.

Numerical: $n_0=1000$, $\mu_0=0.05$, $n=100$, $k=10$: posterior mean $= (1000\cdot0.05 + 100\cdot0.1)/1100 = 60/1100 \approx 0.0545$, much closer to $0.05$ than to the MLE $0.10$.

---

### Problem 13: Expected Loss for Deploying a Variant

**Problem 13.** Express the expected loss $\mathcal{L}_B = \mathbb{E}[\max(p_A - p_B, 0)]$ as an integral over posterior Beta distributions, decompose it into two tractable terms, and bound the Monte Carlo standard error.

**Key insight:** The expected loss integral splits into two terms involving the first moment of a truncated Beta distribution, each expressible as a probability of superiority under a shifted Beta parameter.

**Sketch:**
$$\mathcal{L}_B = \int_0^1\!\int_0^1 \max(p_A-p_B,0)\pi(p_A)\pi(p_B)\,dp_A\,dp_B = \int_0^1\!\int_{p_B}^1 (p_A-p_B)\pi(p_A)\,dp_A\,\pi(p_B)\,dp_B$$
Splitting: $T_1 = \mathbb{E}[p_A\mathbf{1}[p_A > p_B]]$ and $T_2 = \mathbb{E}[p_B\mathbf{1}[p_A > p_B]]$. For $T_1$: write $\mathbb{E}[p_A\mathbf{1}[p_A>p_B]] = \frac{\alpha_A'}{\alpha_A'+\beta_A'} \cdot P(\tilde{p}_A > p_B)$ where $\tilde{p}_A \sim \text{Beta}(\alpha_A'+1,\beta_A')$, using the identity $\mathbb{E}[p\mathbf{1}[p>x]] = \mu_A \cdot P(p_A>x; \alpha_A'+1, \beta_A')$. The Monte Carlo estimator $\hat{\mathcal{L}}_B = S^{-1}\sum_s\max(p_A^{(s)}-p_B^{(s)},0) \xrightarrow{a.s.} \mathcal{L}_B$ by the strong LLN; standard error $\leq 1/(2\sqrt{S})$ (since the integrand is bounded in $[0,1]$), so $S = 10^4$ gives accuracy $\approx 0.005$.

---

### Problem 14: CUPED Variance Reduction Factor

**Problem 14.** Derive the optimal control coefficient $\theta^* = \mathrm{Cov}(Y, X)/\mathrm{Var}(X)$ and show the residual variance of the CUPED outcome is $\sigma_Y^2(1 - \rho^2)$.

**Key insight:** Completing the square in the variance quadratic shows the residual variance at the optimal $\theta^*$ is $\text{Var}(Y)(1 - \rho^2)$, the irreducible variance after projecting out $X$.

**Sketch:**
$$\text{Var}(Y - \theta X) = \sigma_Y^2 - 2\theta\,\text{Cov}(Y,X) + \theta^2\sigma_X^2$$
Setting $d/d\theta = 0$: $\theta^* = \text{Cov}(Y,X)/\sigma_X^2$. Substituting:
$$\text{Var}(Y^{\text{CUPED}}) = \sigma_Y^2 - \frac{[\text{Cov}(Y,X)]^2}{\sigma_X^2} = \sigma_Y^2\!\left(1 - \frac{[\text{Cov}(Y,X)]^2}{\sigma_Y^2\sigma_X^2}\right) = \sigma_Y^2(1-\rho^2)$$
Sample size scales linearly with variance, so required $n$ drops by factor $(1-\rho^2)$: $\rho=0.5 \Rightarrow 25\%$ reduction; $\rho=0.8 \Rightarrow 64\%$ reduction. Halving sample size requires $\rho^2 = 1/2$, i.e., $\rho = 1/\sqrt{2} \approx 0.707$.

---

### Problem 15: Unbiasedness of the CUPED Estimator

**Problem 15.** Prove that $\hat{\tau}^{\mathrm{CUPED}}$ is unbiased for the ATE when the covariate $X_i$ is pre-experiment, and characterize the bias when $X_i$ is a post-experiment variable.

**Key insight:** The bias of CUPED adjustments cancels in the treatment-minus-control difference precisely because $X_i \perp T_i$ forces the conditional mean of $X_i$ to be the same in both groups.

**Sketch:**
$$\mathbb{E}[Y_i^{\text{CUPED}} \mid T_i = t] = \mathbb{E}[Y_i \mid T_i=t] - \theta^*(\mathbb{E}[X_i \mid T_i=t] - \mathbb{E}[X_i])$$
Since $X_i \perp T_i$: $\mathbb{E}[X_i \mid T_i=t] = \mathbb{E}[X_i]$, so the adjustment term is zero. Hence $\mathbb{E}[\hat{\tau}^{\text{CUPED}}] = \mathbb{E}[Y_i\mid T_i=1] - \mathbb{E}[Y_i\mid T_i=0] = \tau$. If $X_i$ is post-experiment, $\mathbb{E}[X_i \mid T_i=1] \neq \mathbb{E}[X_i]$ in general; bias equals $\theta^*(\mathbb{E}[X_i\mid T_i=1]-\mathbb{E}[X_i\mid T_i=0]) = \theta^*\,\text{(treatment effect on }X\text{)} \neq 0$ unless $\text{Cov}(X_i,T_i) = 0$.

FWL equivalence: since $T_i \perp X_i$, the OLS residual of $T$ on $X$ is $T_i - \bar{T}$; the FWL coefficient on $T$ in the joint regression equals the coefficient from regressing $Y_i - \hat{\beta}X_i$ on $T_i$, which is $\hat{\tau}^{\text{CUPED}}$.

---

### Problem 16: Bonferroni FWER Control via the Union Bound

**Problem 16.** Prove that the Bonferroni procedure with per-test threshold $\alpha/m$ controls the FWER at level $\alpha$ under any dependence structure, and analyze the conservatism of the bound as $m \to \infty$.

**Key insight:** The Boole inequality (union bound) converts a probability over a union of events into a sum of individual probabilities, giving a valid upper bound under any dependence structure.

**Sketch:**
$$\text{FWER} = P\!\left(\bigcup_{j \in \mathcal{H}_0}\{p_j \leq \alpha/m\}\right) \leq \sum_{j \in \mathcal{H}_0} P(p_j \leq \alpha/m) = m_0 \cdot \frac{\alpha}{m} \leq \alpha$$
No assumption on dependence needed; the union bound holds universally.

Tightness: with $m_0 = m$ independent tests, FWER $= 1-(1-\alpha/m)^m$. As $m \to \infty$: $(1-\alpha/m)^m \to e^{-\alpha}$, so FWER $\to 1-e^{-\alpha} < \alpha$ (strictly less, so Bonferroni is conservative in the limit). FWER $= m_0\alpha/m$ exactly only when all true-null p-values are at the boundary $\alpha/m$ simultaneously — this holds if all true nulls are independent and each $p_j$ is exactly $\alpha/m$ (a degenerate case). When $m_0 \ll m$, the per-test threshold $\alpha/m$ is far below $\alpha/m_0$, severely reducing power.

---

### Problem 17: BH Procedure FDR Control Level

**Problem 17.** Prove that the Benjamini-Hochberg procedure controls the FDR at level $m_0 \alpha / m \leq \alpha$ under independent test statistics, and identify when the bound is tight.

**Key insight:** The per-hypothesis FDR contribution is bounded by $\alpha/m$ by conditioning on the rank of a uniform p-value and using the BH threshold structure; summing over $m_0$ true nulls gives $m_0\alpha/m$.

**Sketch:**
FDR $= \mathbb{E}[V/\max(R,1)] = \sum_{j\in\mathcal{H}_0}\mathbb{E}[\mathbf{1}[p_j \leq p_{(k^*)}]/R]$. For true null $j$: if $j$ has rank $\ell$ among all p-values, it is rejected only if $p_j \leq \ell\alpha/m$ (BH threshold at rank $\ell$) and $R \geq \ell$. So:
$$\mathbb{E}[\mathbf{1}[p_j \text{ rejected}]/R] \leq \mathbb{E}[p_j\cdot m/(\alpha\cdot R) \cdot \mathbf{1}[p_j\text{ rejected}]] \leq \frac{m}{\alpha}\mathbb{E}[p_j/R] \cdot P(\text{rejected}) \leq \frac{\alpha}{m}$$
using $p_j \sim \text{Uniform}(0,1)$ and $p_j/R \leq p_j\cdot(\alpha/m)/p_j = \alpha/m$ whenever $j$ is rejected. Summing: FDR $\leq m_0\alpha/m \leq \alpha$. Tight when $m_0 = m$: then FDR $= P(R > 0) = \text{FWER}$; all tests are null, any rejection is a false discovery, and BH at $m_0 = m$ coincides with its worst-case FWER control.

---

### Problem 18: Regret Decomposition for Multi-Armed Bandits

**Problem 18.** Derive the arm-by-arm regret decomposition $\mathrm{Regret}(T) = \sum_k \Delta_k \mathbb{E}[N_k(T)]$ by swapping the order of summation over rounds and arms.

**Key insight:** Swapping the order of summation (over time rounds and arms) converts the total regret into arm-by-arm contributions, revealing that only pulls of suboptimal arms ($\Delta_k > 0$) accumulate regret.

**Sketch:**
$$\text{Regret}(T) = T\mu^* - \mathbb{E}\!\left[\sum_{t=1}^T R_{a_t,t}\right] = \mathbb{E}\!\left[\sum_{t=1}^T(\mu^* - R_{a_t,t})\right] = \mathbb{E}\!\left[\sum_{t=1}^T(\mu^* - \mu_{a_t})\right]$$
The last step uses $\mathbb{E}[R_{a_t,t} \mid a_t] = \mu_{a_t}$ and tower property. Now swap the sum:
$$\sum_{t=1}^T(\mu^* - \mu_{a_t}) = \sum_{k=1}^K (\mu^* - \mu_k)\sum_{t=1}^T\mathbf{1}[a_t=k] = \sum_k\Delta_k N_k(T)$$
By linearity: $\text{Regret}(T) = \sum_k \Delta_k\mathbb{E}[N_k(T)]$. The $k^*$ term is $\Delta_{k^*}\mathbb{E}[N_{k^*}(T)] = 0$. Design objective: minimize $\mathbb{E}[N_k(T)]$ for all $k$ with $\Delta_k > 0$, while identifying $k^*$ quickly enough. For $\Delta_2 = 0.2$ and $\mathbb{E}[N_2(T)] = C\ln T$: Regret$(T) = 0.2C\ln T = O(\ln T)$, consistent ($o(T^\alpha)$ for any $\alpha > 0$).

---

### Problem 19: UCB1 Expected Pulls Bound

**Problem 19.** Prove that UCB1 satisfies $\mathbb{E}[N_k(T)] \leq 8\ln T/\Delta_k^2 + 1 + \pi^2/3$ for each suboptimal arm $k$, using the Hoeffding concentration inequality.

**Key insight:** The Hoeffding bound shows that once arm $k$ has been pulled $l = \lceil 8\ln T/\Delta_k^2 \rceil$ times, the probability its UCB still exceeds $\mu^*$ is at most $t^{-4}$, making the expected additional pulls negligible.

**Sketch:**
Arm $k$ pulled at $t$ requires $\hat{\mu}_k + \sqrt{2\ln t/N_k} \geq \hat{\mu}_{k^*} + \sqrt{2\ln t/N_{k^*}}$, which implies either $\hat{\mu}_{k^*} \leq \mu^* - \sqrt{2\ln t/N_{k^*}}$ (low UCB for $k^*$) or $\hat{\mu}_k + \sqrt{2\ln t/N_k} \geq \mu^*$ (high UCB for $k$). By Hoeffding: $P(\hat{\mu}_k + \sqrt{2\ln t/N_k} \geq \mu^*$ with $N_k \geq l) \leq P(\hat{\mu}_k \geq \mu_k + \Delta_k/2) \leq e^{-2N_k(\Delta_k/2)^2}$. For $l = \lceil 8\ln T/\Delta_k^2\rceil$: $e^{-2l\Delta_k^2/4} \leq e^{-2\ln T} = t^{-4}$.

Therefore $\mathbb{E}[N_k(T)] \leq l + \sum_{t=1}^T t^{-4} \leq l + \pi^2/6 \cdot 2 = l + \pi^2/3$. With $l \leq 8\ln T/\Delta_k^2 + 1$:
$$\mathbb{E}[N_k(T)] \leq \frac{8\ln T}{\Delta_k^2} + 1 + \frac{\pi^2}{3}$$
Multiplying by $\Delta_k$ and summing: $\mathbb{E}[\text{Regret}(T)] \leq \sum_k 8\ln T/\Delta_k + (1+\pi^2/3)\sum_k\Delta_k = O(\ln T)$.

---

### Problem 20: Covariance of Sequential Z-Statistics

**Problem 20.** Compute $\mathrm{Corr}(Z_j, Z_k) = \sqrt{t_j/t_k}$ for sequential z-statistics formed from cumulative i.i.d. sums, and quantify the FWER inflation from applying a naive $z_{0.025}$ threshold at two interim looks.

**Key insight:** The cumulative sum $S_{n_k}$ contains $S_{n_j}$ as a sub-sum; the i.i.d. increment $S_{n_k} - S_{n_j}$ is independent of $S_{n_j}$, so $\text{Cov}(S_{n_j}, S_{n_k}) = \text{Var}(S_{n_j})$ exactly.

**Sketch:**
Write $S_{n_k} = S_{n_j} + (S_{n_k} - S_{n_j})$. Independence of i.i.d. increments: $S_{n_k} - S_{n_j} \perp S_{n_j}$.
$$\text{Cov}(S_{n_j}, S_{n_k}) = \text{Cov}(S_{n_j}, S_{n_j}) + \text{Cov}(S_{n_j}, S_{n_k}-S_{n_j}) = \text{Var}(S_{n_j}) + 0 = \sigma^2 n_j$$
$$\text{Cov}(Z_j, Z_k) = \frac{\sigma^2 n_j}{\sigma^2\sqrt{n_j n_k}} = \sqrt{\frac{n_j}{n_k}} = \sqrt{\frac{t_j}{t_k}}$$

Since $\text{Var}(Z_k) = 1$, Cov = Corr. For $t_1 = 1/2$, $t_2 = 1$: Corr$(Z_1, Z_2) = 1/\sqrt{2}$. FWER of naive procedure: $P(|Z_1|>1.96 \text{ or } |Z_2|>1.96) = 0.05 + 0.05 - P(|Z_1|>1.96 \text{ and } |Z_2|>1.96) \approx 0.10 - 0.017 = 0.083$ (the joint probability evaluated numerically from the bivariate normal with correlation $1/\sqrt{2}$).

---

### Problem 21: Thompson Sampling Probability Matching Property

**Problem 21.** Show that Thompson Sampling's selection probability $P(a_t = k \mid \mathcal{F}_{t-1})$ equals the posterior probability that arm $k$ is optimal, and explain how this drives automatic exploration-exploitation balance.

**Key insight:** Since $(\theta_1, \ldots, \theta_K)$ is a joint draw from the current posterior over $(\mu_1, \ldots, \mu_K)$, the probability that arm $k$'s sample is the largest equals the posterior probability that $\mu_k$ is the true maximum.

**Sketch:**
$$P(a_t = k \mid \mathcal{F}_{t-1}) = P(\theta_k^{(t)} > \theta_j^{(t)}\;\forall j \neq k \mid \mathcal{F}_{t-1})$$
This is the probability that a posterior sample from arm $k$ dominates all others. Since $(\theta_1^{(t)},\ldots,\theta_K^{(t)})$ is drawn jointly from the posterior over $(\mu_1,\ldots,\mu_K)$, the probability of the event $\{\theta_k > \max_{j\neq k}\theta_j\}$ equals the posterior probability that $\mu_k$ is the maximum: $P(\mu_k = \mu^* \mid \mathcal{F}_{t-1})$. The algorithm is self-regulating: arms with high posterior probability of optimality are sampled often (exploitation), and arms with high posterior variance still have non-negligible probability of generating a large sample (exploration). As data accumulates, all posteriors concentrate on their true means; posterior probability on the suboptimal arms goes to zero, and Thompson sampling converges to pure exploitation of arm $k^*$.

---

## Algorithmic Applications

### Problem 22: Sample Size Calculator via Binary Search

**Problem 22.** Implement a binary search over $n$ that finds the smallest sample size achieving a target power, using the power function as a monotone predicate, and handle edge cases for $\delta = 0$ and unachievable targets.

**Key insight:** The power function $\pi(n; \delta, \sigma^2, \alpha)$ is monotone increasing in $n$, making binary search correct and efficient; the closed form should serve as a validation check.

**Sketch:**
```
function compute_power(n, delta, sigma_A_sq, sigma_B_sq, alpha):
    se = sqrt((sigma_A_sq + sigma_B_sq) / n)
    lambda_ = delta / se          # non-centrality
    z = z_quantile(1 - alpha/2)   # e.g., 1.96 for alpha=0.05
    power = Phi(lambda_ - z) + Phi(-lambda_ - z)
    return power

function find_sample_size(delta, sigma_A_sq, sigma_B_sq, alpha, target_power):
    if delta <= 0: raise ValueError("delta must be positive")
    if target_power >= 1: raise ValueError("target_power must be < 1")

    lo, hi = 1, 10_000_000
    while lo < hi:
        mid = (lo + hi) // 2
        if compute_power(mid, delta, sigma_A_sq, sigma_B_sq, alpha) >= target_power:
            hi = mid
        else:
            lo = mid + 1
    return lo   # smallest n achieving target power
```

Closed-form check (symmetric case): `n_closed = ceil((z_alpha2 + z_beta)^2 * 2 * sigma^2 / delta^2)`. Binary search over $[1, 10^7]$ takes $\lceil\log_2(10^7)\rceil = 24$ iterations vs. one closed-form evaluation — negligible difference in practice, but binary search generalizes to non-symmetric and non-Gaussian settings.

Edge cases: `delta = 0` makes `lambda_ = 0` and power $= \alpha$ for all $n$ — guard with an early return. `target_power > compute_power(10^7, ...)` indicates the target is unachievable with reasonable sample sizes — raise an error or return `None`.

---

### Problem 23: Beta-Binomial Thompson Sampling Implementation

**Problem 23.** Implement Thompson Sampling with Beta-Binomial conjugate updates, including routines to estimate the probability of superiority and expected loss via Monte Carlo sampling.

**Key insight:** The Beta-Binomial conjugacy means the posterior update is two integer increments; all Monte Carlo quantities follow directly from posterior samples, keeping the implementation compact and parallelizable.

**Sketch:**
```
# State initialization
alpha = [1, 1, ..., 1]   # K arms, uniform prior
beta_  = [1, 1, ..., 1]

function thompson_round(alpha, beta_):
    theta = [sample_beta(alpha[k], beta_[k]) for k in 1..K]
    a = argmax(theta)
    R = observe_reward(a)          # R in {0, 1}
    alpha[a] += R
    beta_[a]  += (1 - R)
    return a, R

function estimate_probability_of_superiority(alpha_A, beta_A, alpha_B, beta_B, S):
    count = 0
    for s in 1..S:
        p_A = sample_beta(alpha_A, beta_A)
        p_B = sample_beta(alpha_B, beta_B)
        if p_B > p_A: count += 1
    return count / S          # Monte Carlo SE ≈ 1 / (2*sqrt(S))

function estimate_expected_loss(alpha_A, beta_A, alpha_B, beta_B, S):
    total = 0
    for s in 1..S:
        p_A = sample_beta(alpha_A, beta_A)
        p_B = sample_beta(alpha_B, beta_B)
        total += max(p_A - p_B, 0)
    return total / S
```

Monte Carlo SE for expected loss: since $\max(p_A - p_B, 0) \in [0,1]$, Var of each term $\leq 1/4$, so SE $\leq 1/(2\sqrt{S})$. For 3 significant figures at $\mathcal{L}_B \sim 0.01$: need SE $\leq 0.00005$, requiring $S \geq 10^8$. In practice $S = 10^4$ to $10^5$ suffices for business decisions.

---

### Problem 24: Benjamini-Hochberg Procedure Implementation

**Problem 24.** Implement the BH procedure, Bonferroni, and Holm corrections, compare their rejection sets on a worked 10-hypothesis example, and verify that all three run in $O(m \log m)$ time.

**Key insight:** The BH procedure requires only a sort and a scan for the last threshold crossing; both Bonferroni and Holm are also linear after sorting, making all three $O(m \log m)$ overall.

**Sketch:**
```
function bh_procedure(p_values, alpha):
    m = len(p_values)
    order = argsort(p_values)             # ascending
    p_sorted = p_values[order]
    thresholds = [(j+1)*alpha/m for j in 0..m-1]
    k_star = -1
    for j in 0..m-1:
        if p_sorted[j] <= thresholds[j]: k_star = j
    if k_star == -1: return {}            # no rejections
    return set(order[0..k_star])          # reject first k_star+1

function bonferroni(p_values, alpha):
    return {j for j, p in enumerate(p_values) if p <= alpha/len(p_values)}

function holm(p_values, alpha):
    m = len(p_values)
    order = argsort(p_values)
    rejected = {}
    for rank, j in enumerate(order):
        if p_values[j] <= alpha/(m - rank):
            rejected.add(j)
        else:
            break      # step-down: stop at first non-rejection
    return rejected
```

Worked example (10 p-values from note): Bonferroni rejects $\{H_0^{(1)}\}$ (p=0.001 $\leq$ 0.005); Holm stops at rank 2 (p=0.008 $>$ 0.0056), also $\{H_0^{(1)}\}$; BH finds $k^* = 4$ (last $j$ with $p_{(j)} \leq j\alpha/m$: $p_{(4)}=0.020 \leq 0.020$), rejecting $\{H_0^{(1)}, H_0^{(2)}, H_0^{(3)}, H_0^{(4)}\}$. All three procedures run in $O(m\log m)$ (sort dominates). The aggregation scan is $O(m)$ and trivially parallelizable over $j$.

---

### Problem 25: CUPED Estimator Implementation

**Problem 25.** Implement the CUPED estimator by computing the pooled OLS coefficient $\hat{\theta}^*$, constructing the adjusted outcome, and reporting the ATE estimate, Welch standard error, and empirical variance reduction.

**Key insight:** The CUPED estimator reduces to computing one OLS coefficient ($\theta^*$) from the pooled data and then computing a standard difference-in-means on the adjusted outcome — equivalent to the FWL projection.

**Sketch:**
```
function cuped_estimator(Y, X, T):
    # Estimate theta* from pooled sample
    X_centered = X - mean(X)
    theta_star = cov(Y, X) / var(X)     # OLS coefficient (pooled)

    # Adjusted outcome
    Y_cuped = Y - theta_star * X_centered

    # Group-split
    Y1 = Y_cuped[T == 1];  n1 = len(Y1)
    Y0 = Y_cuped[T == 0];  n0 = len(Y0)

    # ATE estimate
    tau_hat = mean(Y1) - mean(Y0)

    # Standard error (Welch-style, within-arm variances)
    se = sqrt(var(Y1)/n1 + var(Y0)/n0)

    # Variance reduction check
    rho_hat = corr(Y, X)
    expected_var_reduction = 1 - rho_hat**2
    actual_var_reduction   = var(Y_cuped) / var(Y)
    # Should satisfy: actual_var_reduction ≈ expected_var_reduction

    return tau_hat, se, rho_hat
```

For $\rho = 0$: `Y_cuped == Y`, no reduction. For $\rho = 0.8$: variance drops by 64%, SE drops by $\sqrt{0.36} = 0.6$, equivalent to 2.78$\times$ more data. The plug-in estimator $\hat{\theta}^*$ introduces $O(n^{-1})$ bias in variance estimates, negligible for large $n$.

---

### Problem 26: UCB1 Bandit with Regret Tracking

**Problem 26.** Implement UCB1 with online mean updates and cumulative regret tracking, then verify empirically that $\mathrm{Regret}(T)/\ln T$ converges to a finite constant consistent with the theoretical $O(\ln T)$ bound.

**Key insight:** UCB1's $O(\ln T)$ regret manifests as a stabilizing ratio $\text{Regret}(T)/\ln T \to \sum_k 8/\Delta_k$ empirically, matching the theoretical bound up to constants.

**Sketch:**
```
# Initialization
mu = [mu_1, ..., mu_K]         # true means (unknown to agent)
mu_star = max(mu)
alpha_hat = [0.0] * K          # empirical means
N = [0] * K; t = 0
# Pull each arm once
for k in 0..K-1:
    R = bernoulli(mu[k]); N[k] = 1; alpha_hat[k] = R; t += 1

cumulative_reward = sum(alpha_hat)
regret_log = []

for t in K..T:
    # UCB selection
    ucb = [alpha_hat[k] + sqrt(2*log(t) / N[k]) for k in 0..K-1]
    a = argmax(ucb)
    R = bernoulli(mu[a])

    # Update
    N[a] += 1
    alpha_hat[a] += (R - alpha_hat[a]) / N[a]   # online mean update
    cumulative_reward += R
    t += 1

    regret = t * mu_star - cumulative_reward
    regret_log.append((t, regret, regret / log(t)))
```

Theoretical check for $K=2$, $\mu^*=0.6$, $\mu_2=0.4$ ($\Delta_2=0.2$): bound gives $\mathbb{E}[N_2(T)] \leq 8\ln T/(0.04) = 200\ln T$, so $\mathbb{E}[\text{Regret}(T)] \leq 0.2 \cdot 200\ln T + \text{const} = 40\ln T + \text{const}$. Empirically, `regret_log[-1][2]` (the ratio $\text{Regret}(T)/\ln T$) converges to a constant near $\Delta_2/\text{KL}(\mu_2, \mu^*) \approx 0.2/0.020 = 10$ by the Lai-Robbins lower bound (since UCB1 is not asymptotically optimal but tracks within a constant factor of the KL-UCB rate).
