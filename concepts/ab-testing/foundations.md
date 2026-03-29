# A/B Testing: Foundations

## Table of Contents

1. [[#1. Motivation|1. Motivation]]
   - [[#1.1 The Fundamental Problem of Causal Inference|1.1 The Fundamental Problem of Causal Inference]]
   - [[#1.2 Why Observational Data Is Insufficient|1.2 Why Observational Data Is Insufficient]]
   - [[#1.3 The Randomized Experiment as a Solution|1.3 The Randomized Experiment as a Solution]]
2. [[#2. The Causal Estimand|2. The Causal Estimand]]
   - [[#2.1 Potential Outcomes and the Rubin Causal Model|2.1 Potential Outcomes and the Rubin Causal Model]]
   - [[#2.2 The Individual Treatment Effect|2.2 The Individual Treatment Effect]]
   - [[#2.3 The Average Treatment Effect|2.3 The Average Treatment Effect]]
   - [[#2.4 SUTVA|2.4 SUTVA]]
   - [[#2.5 Identification Under Random Assignment|2.5 Identification Under Random Assignment]]
   - [[#2.6 Unbiasedness of the Difference-in-Means Estimator|2.6 Unbiasedness of the Difference-in-Means Estimator]]
3. [[#3. Statistical Hypotheses|3. Statistical Hypotheses]]
   - [[#3.1 Formulating the Null and Alternative|3.1 Formulating the Null and Alternative]]
   - [[#3.2 The Test Statistic Framework|3.2 The Test Statistic Framework]]
   - [[#3.3 Significance Level|3.3 Significance Level]]
4. [[#4. Type I and Type II Errors|4. Type I and Type II Errors]]
   - [[#4.1 Definitions|4.1 Definitions]]
   - [[#4.2 The Error Table|4.2 The Error Table]]
   - [[#4.3 The Tradeoff and Sample Size Constraint|4.3 The Tradeoff and Sample Size Constraint]]
5. [[#5. P-values|5. P-values]]
   - [[#5.1 Formal Definition|5.1 Formal Definition]]
   - [[#5.2 The P-value Is Uniform Under the Null|5.2 The P-value Is Uniform Under the Null]]
   - [[#5.3 Common Misconceptions|5.3 Common Misconceptions]]
6. [[#6. Statistical Power|6. Statistical Power]]
   - [[#6.1 Definition and the Power Function|6.1 Definition and the Power Function]]
   - [[#6.2 Factors Governing Power|6.2 Factors Governing Power]]
   - [[#6.3 Standard Power Targets|6.3 Standard Power Targets]]
7. [[#7. Effect Size and MDE|7. Effect Size and MDE]]
   - [[#7.1 Standardized Effect Sizes|7.1 Standardized Effect Sizes]]
   - [[#7.2 Relative Lift|7.2 Relative Lift]]
   - [[#7.3 Minimum Detectable Effect: Definition|7.3 Minimum Detectable Effect: Definition]]
   - [[#7.4 Derivation of the MDE for a Two-Sample Z-test|7.4 Derivation of the MDE for a Two-Sample Z-test]]
   - [[#7.5 Practical Implications|7.5 Practical Implications]]
8. [[#8. References|8. References]]

---

## 1. Motivation 🎯

Science frequently asks questions of the form: "Does intervention $X$ cause outcome $Y$?" Answering such questions rigorously requires more than correlation — it requires a *causal* claim. This section establishes why randomized experiments occupy a privileged position in the hierarchy of evidence, and how an A/B test fits into that framework.

### 1.1 The Fundamental Problem of Causal Inference

Let unit $i$ denote a single experimental subject — a user, a patient, a webpage visit. Suppose we wish to understand the causal effect of a binary intervention $T \in \{0, 1\}$ on an outcome $Y$. The *fundamental problem of causal inference* is:

> For any unit $i$, we can observe $Y_i$ under at most one value of $T$. The outcome under the counterfactual assignment is permanently unobserved.

This is not a limitation of measurement technology or data availability. It is a logical impossibility: a user either sees the new feature or does not, never both simultaneously. Because causal effects are defined as comparisons between two states of the world — one factual, one counterfactual — they are inherently unobservable at the individual level.

### 1.2 Why Observational Data Is Insufficient

In an observational study, the treatment $T_i$ is not under the experimenter's control. Units self-select into treatment, or are assigned by a mechanism we do not observe. This introduces two canonical threats to causal identification:

⚠️ **Confounding.** A *confounder* is a variable $Z_i$ that causes both $T_i$ and $Y_i$. If $Z_i$ is unobserved, we cannot separate its effect from that of $T_i$. For example: users who voluntarily adopt a new product feature may already be more engaged, so a naive comparison of their outcomes to non-adopters conflates the effect of the feature with the pre-existing engagement difference.

⚠️ **Selection bias.** Units that end up in the treatment group may have systematically different potential outcomes than units in the control group, even after conditioning on observed covariates. Selection bias is precisely the failure of $\mathbb{E}[Y_i(0) \mid T_i = 1] = \mathbb{E}[Y_i(0) \mid T_i = 0]$: the untreated potential outcomes of the treated group differ from those of the control group. Any difference in observed outcomes then mixes the treatment effect with this pre-existing gap.

Formally, the observed difference-in-means decomposes as:

$$\mathbb{E}[Y_i \mid T_i = 1] - \mathbb{E}[Y_i \mid T_i = 0] = \underbrace{\mathbb{E}[Y_i(1) - Y_i(0) \mid T_i = 1]}_{\text{ATT}} + \underbrace{\mathbb{E}[Y_i(0) \mid T_i = 1] - \mathbb{E}[Y_i(0) \mid T_i = 0]}_{\text{selection bias}}$$

💡 The second term is in general nonzero in observational data, and unobservable without further assumptions. This is precisely the gap that randomization closes.

### 1.3 The Randomized Experiment as a Solution

A *randomized controlled trial* (RCT) assigns treatment $T_i$ independently of all unit characteristics — observed or unobserved. Under random assignment, by construction:

$$\bigl(Y_i(0),\, Y_i(1)\bigr) \perp\!\!\!\perp T_i$$

> **Aside: the $\perp\!\!\!\perp$ notation.**
> The symbol $\perp\!\!\!\perp$ denotes *statistical independence*. We write $A \perp\!\!\!\perp B$ to mean that $A$ and $B$ are independent random variables: $P(A \in S \mid B = b) = P(A \in S)$ for all measurable $S$ and all $b$, i.e., knowing $B$ provides no information about $A$. The notation extends to *conditional independence*: $A \perp\!\!\!\perp B \mid C$ means $P(A \in S \mid B = b, C = c) = P(A \in S \mid C = c)$ for all $b, c$.
>
> The double-perpendicular $\perp\!\!\!\perp$ is used rather than the single $\perp$ (geometric orthogonality) to signal a strictly stronger condition. Two mean-zero random variables with $\mathbb{E}[AB] = 0$ are *orthogonal* ($A \perp B$) but not necessarily independent — orthogonality only kills the linear relationship. Independence kills every statistical relationship. In the causal inference context, we need the stronger condition: randomization must make treatment assignment independent of *all* potential outcome structure, not just uncorrelated with it.

This independence annihilates the selection bias term above. The observed difference-in-means becomes equal to the *average treatment effect* (ATE) over the population. An *A/B test* is an RCT applied in a digital product context: users are randomly assigned to variant A (control) or variant B (treatment), and a metric of interest is compared across groups.

---

## 2. The Causal Estimand 🧮

### 2.1 Potential Outcomes and the Rubin Causal Model

The *Rubin causal model* (RCM), also called the *Neyman–Rubin potential outcomes framework*, provides the formal scaffolding for causal inference. We work with the following primitives.

📐 **Definition (Population and Units).** Let $\mathcal{U} = \{1, \ldots, N\}$ denote a finite population of $N$ units. Each unit $i \in \mathcal{U}$ is characterized by a pair of *potential outcomes*:

$$Y_i(0) \in \mathbb{R}, \qquad Y_i(1) \in \mathbb{R}$$

where $Y_i(t)$ is the outcome unit $i$ would exhibit if assigned to treatment $t \in \{0, 1\}$. The value $t = 0$ denotes control and $t = 1$ denotes treatment.

The observed outcome for unit $i$, given actual assignment $T_i \in \{0, 1\}$, is:

$$Y_i^{\text{obs}} = Y_i(T_i) = T_i \cdot Y_i(1) + (1 - T_i) \cdot Y_i(0)$$

The counterfactual outcome $Y_i(1 - T_i)$ is never observed. This is the fundamental source of difficulty.

### 2.2 The Individual Treatment Effect

📐 **Definition (ITE).** The *individual treatment effect* for unit $i$ is:

$$\tau_i = Y_i(1) - Y_i(0)$$

*It is never directly observable*, because we observe only one of $Y_i(0)$ and $Y_i(1)$ for any given unit. Statistical inference for causal effects must therefore operate on population-level summaries.

### 2.3 The Average Treatment Effect

📐 **Definition (ATE).** The *average treatment effect* is the expectation of the individual treatment effect over the population:

$$\tau = \mathbb{E}[\tau_i] = \mathbb{E}[Y_i(1) - Y_i(0)] = \mathbb{E}[Y_i(1)] - \mathbb{E}[Y_i(0)]$$

where the expectation is taken with respect to the distribution over units in $\mathcal{U}$ (or the superpopulation from which they are drawn). The linearity of expectation allows the second equality; the ATE is the difference of two population means of potential outcomes.

Two related estimands appear frequently:

- *Average treatment effect on the treated* (ATT): $\tau_{\text{ATT}} = \mathbb{E}[Y_i(1) - Y_i(0) \mid T_i = 1]$
- *Average treatment effect on the control* (ATC): $\tau_{\text{ATC}} = \mathbb{E}[Y_i(1) - Y_i(0) \mid T_i = 0]$

💡 Under complete random assignment, $\tau = \tau_{\text{ATT}} = \tau_{\text{ATC}}$, since treatment and control groups are exchangeable. A/B tests target the ATE.

### 2.4 SUTVA

📐 **Definition (SUTVA).** The *Stable Unit Treatment Value Assumption* comprises two conditions:

1. **No interference:** The potential outcome $Y_i(t)$ for unit $i$ depends only on $i$'s own treatment assignment $t$, not on the treatment assignments of any other unit $j \neq i$. Formally, for all $\mathbf{t} = (t_1, \ldots, t_N)$:
$$Y_i(\mathbf{t}) = Y_i(t_i)$$

2. **No hidden versions of treatment:** There is only one version of each treatment level. Receiving $T_i = 1$ means receiving a uniquely defined intervention, not one of several heterogeneous variants that happen to share a label.

⚠️ *SUTVA can fail in realistic A/B tests.* Network interference (e.g., one user's experience affects another's via a social graph) and marketplace competition effects (a supply platform serving both experiment arms simultaneously) are standard violations. When SUTVA fails, the potential outcomes $Y_i(t)$ are not well-defined without specifying the full assignment vector $\mathbf{t}$, and the ATE as defined above is not the right estimand. See [Kohavi, Tang & Xu (2020)](https://www.cambridge.org/core/books/trustworthy-online-controlled-experiments/D97B26382EB0EB2DC2019A7A7B518F59) for practical mitigations.

### 2.5 Identification Under Random Assignment

The identification problem is: how do we connect the unobservable causal estimand $\tau = \mathbb{E}[Y_i(1)] - \mathbb{E}[Y_i(0)]$ to observable quantities?

🔑 **Proposition (Identification by Randomization).** Suppose treatment is assigned completely at random, so that $T_i \perp\!\!\!\perp (Y_i(0), Y_i(1))$ for all $i$. Then:

$$\mathbb{E}[Y_i(1)] = \mathbb{E}[Y_i \mid T_i = 1], \qquad \mathbb{E}[Y_i(0)] = \mathbb{E}[Y_i \mid T_i = 0]$$

and therefore $\tau = \mathbb{E}[Y_i \mid T_i = 1] - \mathbb{E}[Y_i \mid T_i = 0]$.

*Proof.* We derive the first equality; the second is symmetric.

$$\begin{align}
\mathbb{E}[Y_i \mid T_i = 1] &= \mathbb{E}[Y_i(T_i) \mid T_i = 1] \\
&= \mathbb{E}[Y_i(1) \mid T_i = 1] \quad \text{(since } T_i = 1 \text{ on the event } \{T_i = 1\}) \\
&= \mathbb{E}[Y_i(1)] \quad \text{(by independence } T_i \perp\!\!\!\perp Y_i(1))
\end{align}$$

The key step is the last equality: because $T_i$ is independent of the pair $(Y_i(0), Y_i(1))$, conditioning on $T_i = 1$ does not change the distribution of $Y_i(1)$. This is exactly what random assignment guarantees — the treated group is a representative sample of the population with respect to potential outcomes.

Combining both equalities:

$$\tau = \mathbb{E}[Y_i(1)] - \mathbb{E}[Y_i(0)] = \mathbb{E}[Y_i \mid T_i = 1] - \mathbb{E}[Y_i \mid T_i = 0]$$

**The right-hand side involves only observed outcomes.** $\square$

### 2.6 Unbiasedness of the Difference-in-Means Estimator

In practice, we observe samples $\{(Y_i, T_i)\}_{i=1}^{n}$ with $n_1 = \sum_i T_i$ treated units and $n_0 = n - n_1$ control units. Define the *difference-in-means estimator*:

$$\hat{\tau} = \bar{Y}_1 - \bar{Y}_0 = \frac{1}{n_1} \sum_{i: T_i = 1} Y_i - \frac{1}{n_0} \sum_{i: T_i = 0} Y_i$$

🔑 **Proposition (Unbiasedness).** Under random assignment with fixed $n_1$ and $n_0$, $\mathbb{E}[\hat{\tau}] = \tau$.

*Proof.*

$$\begin{align}
\mathbb{E}[\hat{\tau}] &= \mathbb{E}[\bar{Y}_1] - \mathbb{E}[\bar{Y}_0] \\
&= \mathbb{E}\!\left[\frac{1}{n_1}\sum_{i:T_i=1} Y_i(1)\right] - \mathbb{E}\!\left[\frac{1}{n_0}\sum_{i:T_i=0} Y_i(0)\right]
\end{align}$$

We now show $\mathbb{E}[\bar{Y}_1] = \mathbb{E}[Y_i(1)]$ explicitly; the control term is symmetric. Under random assignment, $T_i \perp\!\!\!\perp (Y_i(0), Y_i(1))$, so each unit that ends up with $T_i = 1$ is an independent, representative draw from the population with respect to potential outcomes. Formally, for each $i$ in the treated subsample:

$$\mathbb{E}[Y_i(1) \mid T_i = 1] = \mathbb{E}[Y_i(1)]$$

by independence. Taking the average over the $n_1$ treated units:

$$\mathbb{E}[\bar{Y}_1] = \mathbb{E}\!\left[\frac{1}{n_1}\sum_{i:T_i=1} Y_i(1)\right] = \frac{1}{n_1} \cdot n_1 \cdot \mathbb{E}[Y_i(1)] = \mathbb{E}[Y_i(1)]$$

where the second equality uses the fact that each term in the sum has the same expectation $\mathbb{E}[Y_i(1)]$ under random assignment, and $n_1$ is fixed by design. By the same argument, $\mathbb{E}[\bar{Y}_0] = \mathbb{E}[Y_i(0)]$. Therefore:

$$\mathbb{E}[\hat{\tau}] = \mathbb{E}[Y_i(1)] - \mathbb{E}[Y_i(0)] = \tau \qquad \square$$

**The difference-in-means estimator is unbiased for the ATE under any random assignment mechanism.**

> **TL;DR (Sections 1–2).** We want to measure a causal effect $\tau = \mathbb{E}[Y_i(1)] - \mathbb{E}[Y_i(0)]$, but can never observe both potential outcomes for the same unit. Observational data fails because selection bias contaminates the difference-in-means with pre-existing group differences. Random assignment fixes this: it makes treatment $T_i$ independent of all potential outcomes ($T_i \perp\!\!\!\perp (Y_i(0), Y_i(1))$), collapsing the selection bias term to zero and making $\hat{\tau} = \bar{Y}_1 - \bar{Y}_0$ an unbiased estimator of $\tau$. Two assumptions must hold for this to work cleanly: SUTVA (no interference between units, no hidden treatment versions) and complete randomization. When SUTVA fails — e.g. network effects, marketplace competition — $\tau$ is no longer well-defined without the full assignment vector.

---

## 3. Statistical Hypotheses 🔬

### 3.1 Formulating the Null and Alternative

Having established that $\hat{\tau}$ is an unbiased estimator of $\tau$, we now set up the inferential machinery for deciding whether the observed $\hat{\tau}$ is consistent with zero or with a meaningful nonzero effect.

📐 **Definition (Null Hypothesis).** The *null hypothesis* $H_0$ is the hypothesis of no treatment effect:

$$H_0 : \tau = 0$$

This is the *weak null* in the potential outcomes framework. A stronger version is *Fisher's sharp null*: $Y_i(1) = Y_i(0)$ for all $i$, which asserts that the treatment effect is zero for every individual unit, not just on average. The sharp null is what justifies the permutation test (see [[p-values.md#7. The Permutation Test: Downey's Canonical Example|Section 7 of p-values.md]]).

*Throughout the remainder of this note, we work under the Neyman framing: $H_0: \tau = 0$ refers to the weak null on the ATE, and inference proceeds via the CLT-based normal approximation.*

📐 **Definition (Alternative Hypothesis).** The *alternative hypothesis* $H_1$ specifies the direction of departure from $H_0$. Two forms are standard:

- **Two-tailed:** $H_1 : \tau \neq 0$. Appropriate when we have no prior knowledge of the sign of the effect and wish to detect deviations in either direction.
- **One-tailed:** $H_1 : \tau > 0$. Appropriate when the treatment can only plausibly improve the outcome (e.g., a new ranking algorithm cannot degrade engagement by construction).

⚠️ *The choice between one-tailed and two-tailed tests must be made before data collection.* Post-hoc switching from two-tailed to one-tailed upon seeing the data direction inflates Type I error — this is a form of p-hacking.

### 3.2 The Test Statistic Framework

A *test statistic* $W_n = W_n(Y_1, \ldots, Y_n, T_1, \ldots, T_n)$ is a real-valued function of the data, chosen so that large values of $|W_n|$ (or large values of $W_n$ for one-tailed tests) are inconsistent with $H_0$. We reject $H_0$ when $W_n$ falls in the *rejection region* $\mathcal{R}$:

$$\text{Reject } H_0 \iff W_n \in \mathcal{R}$$

The rejection region is defined so that $\mathbb{P}(W_n \in \mathcal{R} \mid H_0) \leq \alpha$, where $\alpha$ is the significance level defined below. The standard test statistic for a two-sample comparison of means under known variance is:

$$Z = \frac{\hat{\tau}}{\sqrt{\sigma_1^2 / n_1 + \sigma_0^2 / n_0}}$$

where $\sigma_t^2 = \text{Var}(Y_i(t))$ for $t \in \{0, 1\}$. Under $H_0$ and by the central limit theorem (CLT), $Z \xrightarrow{d} \mathcal{N}(0, 1)$ as $n \to \infty$.

💡 For the full treatment of how different choices of test statistic and null distribution give rise to named tests (Z-test, Welch's t, permutation test), see [[p-values.md#3. The Unified Structure of Hypothesis Tests|Section 3 of p-values.md]].

### 3.3 Significance Level

📐 **Definition (Significance Level).** The *significance level* $\alpha \in (0, 1)$ is a pre-specified upper bound on the probability of rejecting $H_0$ when $H_0$ is true:

$$\alpha = \sup_{\theta \in H_0} \mathbb{P}_\theta(W_n \in \mathcal{R})$$

Common choices are $\alpha = 0.05$ and $\alpha = 0.01$. ⚠️ *The significance level must be fixed before the experiment begins*; it is not a free parameter to be tuned after observing results.

---

## 4. Type I and Type II Errors ⚖️

### 4.1 Definitions

Any decision procedure based on a test statistic can make two distinct types of error.

📐 **Definition (Type I Error).** A *Type I error* (false positive) occurs when $H_0$ is true but we reject it. The Type I error rate is:

$$\alpha = \mathbb{P}(\text{reject } H_0 \mid H_0 \text{ true})$$

By construction, our choice of rejection region ensures $\alpha$ is controlled at the pre-specified level.

📐 **Definition (Type II Error).** A *Type II error* (false negative) occurs when $H_1$ is true but we fail to reject $H_0$. The Type II error rate is:

$$\beta = \mathbb{P}(\text{fail to reject } H_0 \mid H_1 \text{ true})$$

Note that $\beta$ is not a single number — it depends on the specific value of $\tau$ under $H_1$. We discuss this further in Section 6.

### 4.2 The Error Table 📋

The complete classification of outcomes is:

|  | $H_0$ True | $H_1$ True |
|---|---|---|
| **Reject $H_0$** | Type I error (rate $\alpha$) | Correct rejection (rate $1 - \beta$) |
| **Fail to reject $H_0$** | Correct retention (rate $1 - \alpha$) | Type II error (rate $\beta$) |

The rate $1 - \beta$ is the *statistical power* of the test (Section 6). The rate $1 - \alpha$ is sometimes called the *specificity* or *true negative rate*.

### 4.3 The Tradeoff and Sample Size Constraint

For a fixed test statistic and data-generating process, $\alpha$ and $\beta$ are not simultaneously free. Decreasing $\alpha$ (tightening the rejection threshold) increases $\beta$ and vice versa, holding $n$ fixed. The only way to simultaneously reduce both error rates is to increase the sample size $n$.

More precisely, for the two-sample z-test under a two-tailed alternative ($H_1: \tau \neq 0$), the required per-group sample size to achieve error rates $(\alpha, \beta)$ at true effect size $\delta$ is:

$$n = \frac{2\sigma^2 (z_{\alpha/2} + z_\beta)^2}{\delta^2}$$

where $z_{\alpha/2} = \Phi^{-1}(1-\alpha/2)$ and $z_\beta = \Phi^{-1}(1-\beta)$, and $\Phi$ is the standard normal CDF. The half-$\alpha$ quantile appears because the two-tailed test splits the rejection region equally between both tails. **Simultaneously halving $\alpha$ and $\beta$ requires increasing $n$ by a constant factor determined by the new quantiles — it cannot be done for free.**

---

## 5. P-values 📊

### 5.1 Formal Definition

📐 **Definition (P-value).** Let $t_{\text{obs}}$ be the realized value of the test statistic $W_n$ from the observed data. The *p-value* is the probability, computed under $H_0$, of observing a test statistic at least as extreme as $t_{\text{obs}}$:

- **One-tailed (upper):**

$$p = \mathbb{P}(W_n \geq t_{\text{obs}} \mid H_0)$$

- **Two-tailed:**

$$p = \mathbb{P}(|W_n| \geq |t_{\text{obs}}| \mid H_0) = 2 \cdot \mathbb{P}(W_n \geq |t_{\text{obs}}| \mid H_0)$$

where the second equality holds when the null distribution of $W_n$ is symmetric about zero (as is the case for $W_n \sim \mathcal{N}(0,1)$ under $H_0$).

For the standard normal test statistic, these become:

$$p_{\text{one}} = 1 - \Phi(t_{\text{obs}}), \qquad p_{\text{two}} = 2\bigl(1 - \Phi(|t_{\text{obs}}|)\bigr)$$

The decision rule "reject $H_0$ if $p \leq \alpha$" is equivalent to "reject $H_0$ if $W_n$ falls in the rejection region $\mathcal{R}$," by construction of the p-value. For a much deeper treatment of p-values — including the uniformity proof, the unified test structure, and the permutation test — see [[p-values.md]].

### 5.2 The P-value Is Uniform Under the Null

A fundamental and often underappreciated fact is that the p-value, viewed as a random variable before data are collected, is *stochastically* well-behaved under $H_0$.

🔑 **Proposition (Uniformity of P-value Under $H_0$).** Let $W_n$ be a test statistic with continuous CDF $F_{H_0}$ under $H_0$. Define the (one-tailed) p-value as the random variable $P = 1 - F_{H_0}(W_n)$. Then under $H_0$:

$$P \sim \mathrm{Uniform}(0, 1)$$

*Proof.* Let $U = F_{H_0}(W_n)$. We compute the CDF of $U$:

$$\mathbb{P}(U \leq u) = \mathbb{P}(F_{H_0}(W_n) \leq u) = \mathbb{P}(W_n \leq F_{H_0}^{-1}(u)) = F_{H_0}(F_{H_0}^{-1}(u)) = u$$

for $u \in [0,1]$, where the third equality uses the definition of the CDF. Thus $U \sim \mathrm{Uniform}(0,1)$, and since $P = 1 - U$, we also have $P \sim \mathrm{Uniform}(0,1)$. $\square$

💡 This result — the *probability integral transform* — has an immediate consequence: under $H_0$, we expect roughly $5\%$ of p-values to fall below $0.05$ by chance. Running many tests and reporting only those with $p < 0.05$ therefore inflates the false discovery rate. This is the statistical basis for the *multiple comparisons problem*.

⚠️ *This uniformity holds only for continuous test statistics and point null hypotheses. Discrete distributions produce p-values that are stochastically larger than uniform, so the test is conservative.*

### 5.3 Common Misconceptions

The p-value is one of the most frequently misinterpreted quantities in applied statistics. We state the correct interpretation and catalog the principal errors. A more detailed treatment with formal diagnoses is in [[p-values.md#6. Common Misinterpretations|Section 6 of p-values.md]].

**Correct interpretation.** The p-value $p$ is the probability, under the assumption that $H_0$ is true, of observing data at least as extreme as what was actually observed. It measures the compatibility of the data with $H_0$.

⚠️ **Misconception 1: "The p-value is the probability that $H_0$ is true."**

This is the *prosecutor's fallacy*, a confusion of $\mathbb{P}(\text{data} \mid H_0)$ with $\mathbb{P}(H_0 \mid \text{data})$. The p-value is a frequentist object — $H_0$ is either true or false, not a random event with a probability. Making probabilistic statements about $H_0$ requires a prior, which is the domain of Bayesian inference (see [[bayesian-testing.md]]).

⚠️ **Misconception 2: "A small p-value means the effect is large."**

The p-value conflates effect size with sample size. With $n = 10^7$, a negligible effect $\delta = 0.0001$ can produce $p < 10^{-10}$. Conversely, a large effect in a small sample may yield $p = 0.15$. Effect size must be assessed separately (Section 7). The [ASA statement on p-values](https://doi.org/10.1080/00031305.2016.1154108) explicitly flags this as one of the most consequential misuses in practice.

⚠️ **Misconception 3: "The p-value is the probability the result occurred by chance."**

This is an informal gloss that is technically incoherent: probability is a property of random events under a model, not a property of an observed result. The observed data are fixed; the randomness lives in the hypothetical sampling distribution under $H_0$.

---

## 6. Statistical Power ⚡

### 6.1 Definition and the Power Function

📐 **Definition (Power).** The *statistical power* of a test is the probability of correctly rejecting $H_0$ when $H_1$ is true:

$$1 - \beta = \mathbb{P}(\text{reject } H_0 \mid H_1 \text{ true})$$

Because "H_1 true" is not a single hypothesis but a family of alternatives parameterized by the true effect $\delta = \tau$, power is properly a function of $\delta$.

📐 **Definition (Power Function).** The *power function* $\pi : \mathbb{R} \to [0,1]$ is defined as:

$$\pi(\delta) = \mathbb{P}_\delta(\text{reject } H_0)$$

where $\mathbb{P}_\delta$ denotes probability computed under the distribution with true effect size $\delta$. For the two-sample z-test with significance level $\alpha$ and per-group sample size $n$, the test statistic under true effect $\delta$ is:

$$Z = \frac{\hat{\tau} - 0}{\sqrt{2\sigma^2/n}} \sim \mathcal{N}\!\left(\frac{\delta}{\sqrt{2\sigma^2/n}},\, 1\right)$$

The non-centrality parameter is $\lambda = \delta\sqrt{n/(2\sigma^2)}$. For a two-tailed test with critical value $z_{\alpha/2} = \Phi^{-1}(1 - \alpha/2)$:

$$\pi(\delta) = \mathbb{P}(|Z| > z_{\alpha/2} \mid \delta) = 1 - \Phi(z_{\alpha/2} - \lambda) + \Phi(-z_{\alpha/2} - \lambda)$$

For $\delta > 0$ sufficiently large, the second term is negligible: since $\lambda > 0$ and $z_{\alpha/2} > 0$, we have $\Phi(-z_{\alpha/2} - \lambda) < \Phi(-z_{\alpha/2}) = \alpha/2$. Dropping it:

$$\pi(\delta) \approx 1 - \Phi(z_{\alpha/2} - \lambda) = \Phi(\lambda - z_{\alpha/2})$$

Three properties of $\pi(\delta)$ follow directly:

1. $\pi(0) = \alpha$ — at the null, power equals the Type I error rate.
2. $\pi(\delta) \to 1$ as $|\delta| \to \infty$ — a sufficiently large effect is always detected.
3. $\pi(\delta)$ is symmetric about $\delta = 0$ for the two-tailed test.

### 6.2 Factors Governing Power 📋

From the expression $\lambda = \delta\sqrt{n/(2\sigma^2)}$, power increases monotonically in $\lambda$. We can read off the dependence on each design parameter:

| Parameter | Effect on $\pi(\delta)$ | Mechanism |
|-----------|------------------------|-----------|
| $n$ (sample size) | ↑ increases | Larger $n$ shrinks $\text{SE}(\hat{\tau})$, increasing the signal-to-noise ratio $\lambda$ |
| $|\delta|$ (effect size) | ↑ increases | Larger true effect makes the departure from $H_0$ easier to detect |
| $\alpha$ (significance level) | ↑ increases | A laxer threshold makes rejection easier, at the cost of more Type I errors |
| $\sigma^2$ (outcome variance) | ↓ decreases | More noise in $Y$ inflates $\text{SE}(\hat{\tau})$ and reduces signal-to-noise |

💡 *Reducing $\sigma^2$ through variance reduction techniques — stratification, [CUPED](https://dl.acm.org/doi/10.1145/2487575.2488217), regression adjustment — is often more practical than increasing $n$, especially when sample acquisition is costly.*

### 6.3 Standard Power Targets

The industry and clinical trial convention is to target $1 - \beta = 0.80$, corresponding to a $20\%$ chance of missing a true effect. More stringent applications (e.g., safety-critical trials, confirmatory experiments) use $1 - \beta = 0.90$. These thresholds are conventions, not laws: the appropriate target depends on the relative costs of Type I and Type II errors in the specific decision context.

---

## 7. Effect Size and MDE 📏

### 7.1 Standardized Effect Sizes

Raw differences in means $\hat{\tau} = \bar{Y}_1 - \bar{Y}_0$ are not scale-invariant: a difference of 5 seconds is large for page load time but negligible for session duration. *Cohen's d* provides a dimensionless standardization.

📐 **Definition (Cohen's d).** For two populations with means $\mu_1, \mu_0$ and common standard deviation $\sigma$, *Cohen's d* is:

$$d = \frac{\mu_1 - \mu_0}{\sigma}$$

[Cohen's (1988)](https://www.taylorfrancis.com/books/mono/10.4324/9780203771587/statistical-power-analysis-behavioral-sciences-jacob-cohen) interpretive benchmarks — small ($d \approx 0.2$), medium ($d \approx 0.5$), large ($d \approx 0.8$) — are heuristics derived from behavioral science. In technology product experimentation, practically significant effects are often much smaller (e.g., $d < 0.05$), which is why sample sizes in industry A/B tests are often in the millions.

When the two groups have unequal variances $\sigma_0^2$ and $\sigma_1^2$, the pooled standard deviation:

$$\sigma_{\text{pooled}} = \sqrt{\frac{(n_0 - 1)\sigma_0^2 + (n_1 - 1)\sigma_1^2}{n_0 + n_1 - 2}}$$

is used in place of $\sigma$ in the denominator of $d$.

### 7.2 Relative Lift

For metrics with a natural baseline (click-through rate, revenue, conversion rate), the *relative lift* is often more interpretable than the absolute difference:

📐 **Definition (Relative Lift).** Given a baseline mean $\mu_0 > 0$ and treatment mean $\mu_1$:

$$\delta_r = \frac{\mu_1 - \mu_0}{\mu_0} = \frac{\mu_1}{\mu_0} - 1$$

This is the fractional change in the metric due to treatment. A relative lift of $\delta_r = 0.02$ means a $2\%$ improvement over baseline. ⚠️ *When $\mu_0$ is estimated from data rather than known exactly, $\delta_r$ becomes a ratio estimator and its variance calculation requires the delta method.*

### 7.3 Minimum Detectable Effect: Definition

📐 **Definition (MDE).** Given a fixed experimental design — significance level $\alpha$, power target $1 - \beta$, per-group sample size $n$, and outcome variance $\sigma^2$ — the *minimum detectable effect* (MDE) is the smallest absolute effect size $\delta > 0$ such that the test achieves power at least $1 - \beta$:

$$\text{MDE} = \inf\{\delta > 0 : \pi(\delta) \geq 1 - \beta\}$$

The MDE provides a concrete, pre-experiment statement about the resolution of the test: effects smaller than the MDE will be detected less than $100(1-\beta)\%$ of the time.

### 7.4 Derivation of the MDE for a Two-Sample Z-test 📐

We derive a closed-form expression for the MDE under the two-sample z-test with equal group sizes.

**Setup.** Suppose the outcome $Y_i$ has variance $\sigma^2$ in both groups (homoskedastic case). With $n$ observations per group, the estimator $\hat{\tau} = \bar{Y}_1 - \bar{Y}_0$ has variance:

$$\text{Var}(\hat{\tau}) = \frac{\sigma^2}{n} + \frac{\sigma^2}{n} = \frac{2\sigma^2}{n}$$

The test statistic is:

$$Z = \frac{\hat{\tau}}{\sqrt{2\sigma^2/n}} \sim \mathcal{N}(0, 1) \text{ under } H_0$$

**Step 1: Rejection condition.** For a two-tailed test at level $\alpha$, we reject $H_0$ when $|Z| > z_{\alpha/2}$, where $z_{\alpha/2} = \Phi^{-1}(1 - \alpha/2)$.

**Step 2: Distribution under a true effect $\delta$.** When the true ATE is $\delta \neq 0$:

$$Z = \frac{\hat{\tau}}{\sqrt{2\sigma^2/n}} \sim \mathcal{N}\!\left(\frac{\delta}{\sqrt{2\sigma^2/n}},\, 1\right)$$

The non-centrality parameter is $\lambda(\delta) = \delta / \sqrt{2\sigma^2/n} = \delta\sqrt{n/(2\sigma^2)}$.

**Step 3: Power equation.** For $\delta > 0$, the full power expression is $\pi(\delta) = \Phi(\lambda - z_{\alpha/2}) + \Phi(-\lambda - z_{\alpha/2})$. The left-tail term $\Phi(-\lambda - z_{\alpha/2})$ is negligible: since $\lambda > 0$ and $z_{\alpha/2} > 0$, we have $\Phi(-z_{\alpha/2} - \lambda) < \Phi(-z_{\alpha/2}) = \alpha/2$. Dropping it:

$$\pi(\delta) = \mathbb{P}(Z > z_{\alpha/2} \mid \delta) = \mathbb{P}\!\left(\mathcal{N}(\lambda, 1) > z_{\alpha/2}\right) = \Phi\bigl(\lambda - z_{\alpha/2}\bigr)$$

**Step 4: Solve for MDE.** The MDE is the smallest $\delta$ achieving $\pi(\delta) = 1 - \beta$:

$$\Phi(\lambda - z_{\alpha/2}) = 1 - \beta$$

Since $\Phi^{-1}(1 - \beta) = z_\beta$ (where we use the shorthand $z_p = \Phi^{-1}(1-p)$ for tail quantiles):

$$\lambda - z_{\alpha/2} = z_\beta \implies \frac{\delta}{\sqrt{2\sigma^2/n}} = z_{\alpha/2} + z_\beta$$

Solving for $\delta$:

$$\boxed{\text{MDE} = (z_{\alpha/2} + z_\beta)\sqrt{\frac{2\sigma^2}{n}}}$$

For a one-tailed test at level $\alpha$, replace $z_{\alpha/2}$ with $z_\alpha$:

$$\text{MDE}_{\text{one-tailed}} = (z_\alpha + z_\beta)\sqrt{\frac{2\sigma^2}{n}}$$

**Numerical example.** With $\alpha = 0.05$ (two-tailed), $1 - \beta = 0.80$, $\sigma^2 = 1$:

$$z_{0.025} = 1.96, \quad z_{0.20} = 0.842$$

$$\text{MDE} = (1.96 + 0.842)\sqrt{2/n} = 2.802\sqrt{2/n} \approx \frac{3.96}{\sqrt{n}}$$

### 7.5 Practical Implications 💡

From the MDE formula, several structural observations follow:

1. **MDE scales as $1/\sqrt{n}$.** Quadrupling the sample size halves the MDE. This is the square-root law of statistics: detecting effects twice as small requires four times the sample. **Investments in sample size exhibit strongly diminishing returns for effect size resolution.**

2. **Equivalently, solving for $n$:**
$$n = \frac{2\sigma^2(z_{\alpha/2} + z_\beta)^2}{\delta^2}$$
The required $n$ scales as $\delta^{-2}$: detecting effects half as large requires four times the sample.

3. **Variance reduction directly improves MDE.** Any technique that reduces $\sigma^2$ — stratified randomization, [CUPED](https://dl.acm.org/doi/10.1145/2487575.2488217), regression adjustment — reduces the MDE proportionally to $\sqrt{\sigma^2_{\text{new}}/\sigma^2_{\text{old}}}$.

4. **The MDE is a property of the experiment, not the data.** It should be computed before running the test and used to determine whether the experimental design is capable of detecting effects of practical significance. ⚠️ *Running an underpowered experiment is a form of resource waste: it cannot reliably detect true effects and its null results are uninformative.*

---

## 8. References 📚

| Reference Name | Brief Summary | Link to Reference |
|---|---|---|
| [Imbens & Rubin, *Causal Inference for Statistics, Social, and Biomedical Sciences* (2015)](https://www.cambridge.org/core/books/causal-inference-for-statistics-social-and-biomedical-sciences/71126BE90C58F1A431FE9B2DD07938AB) | Definitive textbook treatment of the potential outcomes framework, SUTVA, and identification in randomized experiments | [Cambridge University Press](https://www.cambridge.org/core/books/causal-inference-for-statistics-social-and-biomedical-sciences/71126BE90C58F1A431FE9B2DD07938AB) |
| [Rubin, "Estimating Causal Effects of Treatments in Randomized and Nonrandomized Studies" (1974)](https://doi.org/10.1037/h0037350) | Original paper introducing the potential outcomes notation $Y_i(0), Y_i(1)$ and the modern causal inference framework | [Journal of Educational Psychology](https://doi.org/10.1037/h0037350) |
| [Neyman, "On the Application of Probability Theory to Agricultural Experiments" (1923/1990)](https://doi.org/10.1214/ss/1177012031) | Foundational paper introducing the finite-population potential outcomes framework and the difference-in-means estimator | [Statistical Science (reprint)](https://doi.org/10.1214/ss/1177012031) |
| [Cohen, *Statistical Power Analysis for the Behavioral Sciences*, 2nd ed. (1988)](https://www.taylorfrancis.com/books/mono/10.4324/9780203771587/statistical-power-analysis-behavioral-sciences-jacob-cohen) | Canonical reference for power analysis, effect size conventions (Cohen's d), and MDE; standard industry reference | [Taylor & Francis](https://www.taylorfrancis.com/books/mono/10.4324/9780203771587/statistical-power-analysis-behavioral-sciences-jacob-cohen) |
| [Lehmann & Romano, *Testing Statistical Hypotheses*, 3rd ed. (2005)](https://link.springer.com/book/10.1007/0-387-27605-X) | Rigorous mathematical treatment of hypothesis testing, p-values, power functions, and uniformly most powerful tests | [Springer](https://link.springer.com/book/10.1007/0-387-27605-X) |
| [Deng et al., "Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data" (2013)](https://dl.acm.org/doi/10.1145/2487575.2488217) | Introduces CUPED for variance reduction in A/B tests; shows how to reduce effective $\sigma^2$ without increasing $n$ | [ACM KDD](https://dl.acm.org/doi/10.1145/2487575.2488217) |
| [Kohavi, Tang & Xu, *Trustworthy Online Controlled Experiments* (2020)](https://www.cambridge.org/core/books/trustworthy-online-controlled-experiments/D97B26382EB0EB2DC2019A7A7B518F59) | Comprehensive practitioner guide covering A/B test design, SUTVA violations, network effects, and metric design | [Cambridge University Press](https://www.cambridge.org/core/books/trustworthy-online-controlled-experiments/D97B26382EB0EB2DC2019A7A7B518F59) |
| [Alex Deng, "Potential Outcomes Framework — Causal Inference and Its Applications in Online Industry"](https://alexdeng.github.io/causal/rcm.html) | Online lecture notes linking the RCM to industry A/B testing practice; concise treatment of identification | [alexdeng.github.io](https://alexdeng.github.io/causal/rcm.html) |
| [The Book of Statistical Proofs, "P-value follows Uniform(0,1) under $H_0$"](https://statproofbook.github.io/P/pval-h0.html) | Formal proof of the probability integral transform argument establishing p-value uniformity | [statproofbook.github.io](https://statproofbook.github.io/P/pval-h0.html) |
