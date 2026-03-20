# A/B Testing: Sequential and Adaptive Methods

## Table of Contents

1. [[#1. The Peeking Problem|1. The Peeking Problem]]
   - [[#1.1 Setup and Type I Error Inflation|1.1 Setup and Type I Error Inflation]]
   - [[#1.2 The Correlation Structure Does Not Save You|1.2 The Correlation Structure Does Not Save You]]
   - [[#1.3 The Fix: Joint Error Control|1.3 The Fix: Joint Error Control]]
2. [[#2. Group Sequential Designs|2. Group Sequential Designs]]
   - [[#2.1 Setup and Information Fractions|2.1 Setup and Information Fractions]]
   - [[#2.2 The Covariance Structure of Sequential Z-Statistics|2.2 The Covariance Structure of Sequential Z-Statistics]]
   - [[#2.3 Pocock Boundaries|2.3 Pocock Boundaries]]
   - [[#2.4 O'Brien-Fleming Boundaries|2.4 O'Brien-Fleming Boundaries]]
   - [[#2.5 Sample Size Inflation|2.5 Sample Size Inflation]]
3. [[#3. Alpha-Spending Functions|3. Alpha-Spending Functions]]
   - [[#3.1 The Lan-DeMets Framework|3.1 The Lan-DeMets Framework]]
   - [[#3.2 Incremental Spending and Boundary Computation|3.2 Incremental Spending and Boundary Computation]]
   - [[#3.3 Common Spending Functions|3.3 Common Spending Functions]]
   - [[#3.4 Key Property: FWER Preservation|3.4 Key Property: FWER Preservation]]
4. [[#4. The Multi-Armed Bandit Problem|4. The Multi-Armed Bandit Problem]]
   - [[#4.1 Formal Setup|4.1 Formal Setup]]
   - [[#4.2 Regret and Its Decomposition|4.2 Regret and Its Decomposition]]
   - [[#4.3 The Exploration-Exploitation Tradeoff|4.3 The Exploration-Exploitation Tradeoff]]
   - [[#4.4 Lower Bound: Lai and Robbins (1985)|4.4 Lower Bound: Lai and Robbins (1985)]]
5. [[#5. Epsilon-Greedy and UCB|5. Epsilon-Greedy and UCB]]
   - [[#5.1 Epsilon-Greedy|5.1 Epsilon-Greedy]]
   - [[#5.2 UCB1|5.2 UCB1]]
   - [[#5.3 Regret Bound for UCB1|5.3 Regret Bound for UCB1]]
   - [[#5.4 KL-UCB|5.4 KL-UCB]]
6. [[#6. Thompson Sampling|6. Thompson Sampling]]
   - [[#6.1 Algorithm|6.1 Algorithm]]
   - [[#6.2 Bayesian Regret|6.2 Bayesian Regret]]
   - [[#6.3 Frequentist Regret and Asymptotic Optimality|6.3 Frequentist Regret and Asymptotic Optimality]]
   - [[#6.4 The Probability Matching Property|6.4 The Probability Matching Property]]
   - [[#6.5 Implementation Advantages|6.5 Implementation Advantages]]
7. [[#7. Bandits vs. Fixed A/B Tests|7. Bandits vs. Fixed A/B Tests]]
   - [[#7.1 Fixed A/B Test: Strengths and Regime|7.1 Fixed A/B Test: Strengths and Regime]]
   - [[#7.2 Bandits: Strengths and Regime|7.2 Bandits: Strengths and Regime]]
   - [[#7.3 The Opportunity Cost Framing|7.3 The Opportunity Cost Framing]]
   - [[#7.4 Hybrid Approaches|7.4 Hybrid Approaches]]
8. [[#8. References|8. References]]

---

## 1. The Peeking Problem

### 1.1 Setup and Type I Error Inflation

Consider a standard two-sided $z$-test designed with significance level $\alpha = 0.05$ and a pre-specified sample size $n = 1000$ per arm. A single look at the final data controls the Type I error at exactly $\alpha$: we reject $H_0: \mu_A = \mu_B$ if $|Z| > z_{\alpha/2} = 1.96$.

Now suppose an analyst *peeks* at $n_1 = 500$ observations per arm and again at $n_2 = 1000$, and rejects $H_0$ at either look if the corresponding $p$-value is below $0.05$. This is a two-look procedure that applies the fixed-horizon critical value at every look. What is the actual *family-wise error rate* (FWER)?

Let $Z_1$ and $Z_2$ be the $z$-statistics at the two looks. Under $H_0$, both are standard normal marginally. The FWER of the procedure is

$$\text{FWER} = P\!\left(|Z_1| > 1.96 \text{ or } |Z_2| > 1.96 \;\middle|\; H_0\right).$$

By inclusion-exclusion,

$$\text{FWER} = P(|Z_1| > 1.96) + P(|Z_2| > 1.96) - P(|Z_1| > 1.96 \text{ and } |Z_2| > 1.96).$$

The first two terms each equal $0.05$. To evaluate the intersection term we need the joint distribution of $(Z_1, Z_2)$.

As derived in Section 2.2 below, for equal information fractions $t_1 = 1/2$ and $t_2 = 1$, the correlation is

$$\text{Corr}(Z_1, Z_2) = \sqrt{t_1/t_2} = \sqrt{1/2} = 1/\sqrt{2} \approx 0.707.$$

Numerically integrating the bivariate normal with this correlation gives $P(|Z_1| > 1.96 \text{ and } |Z_2| > 1.96) \approx 0.017$, so

$$\text{FWER} \approx 0.05 + 0.05 - 0.017 = 0.083.$$

**The FWER is inflated from 5% to approximately 8.3% — a 66% increase.** This inflation worsens with more looks. For $k$ equally spaced looks all tested at level $\alpha$, the FWER grows approximately as

$$\text{FWER} \approx \alpha \ln(k) \quad \text{(up to multiplicative constants)}.$$

With $k = 5$ looks the nominal-$5\%$ procedure has actual FWER near $8\%$; with $k = 25$ looks, near $16\%$.

### 1.2 The Correlation Structure Does Not Save You

The positive correlation between $Z_1$ and $Z_2$ does reduce the FWER compared to the independent case (where FWER $= 1 - 0.95^2 = 0.0975$). One might hope that high correlation keeps the FWER near $\alpha$. However, this reasoning fails for two reasons.

First, the analyst *stops at the first rejection*. The observed data at the stopping time is a mixture of: the case where $H_0$ was rejected at look 1 (using $Z_1$ alone) and the case where it was not rejected at look 1 but was at look 2 (using $Z_2$ conditional on $|Z_1| \leq 1.96$). The p-value from either single look does not account for this selection effect — it treats the stopping time as if it were pre-specified.

Second, as $k \to \infty$ the correlation between adjacent looks approaches 1, yet the supremum of $k$ correlated standard normals over $k$ looks still drifts to infinity. The FWER grows without bound as $k$ grows, regardless of correlation.

### 1.3 The Fix: Joint Error Control

The correct remedy is a procedure that controls the FWER *jointly* over all planned looks — that is, a procedure that pre-specifies the critical values $c_1, \ldots, c_K$ for $K$ looks such that

$$P\!\left(\exists\, k \in \{1,\ldots,K\} : |Z_k| > c_k \;\middle|\; H_0\right) = \alpha.$$

Such procedures are called *group sequential designs*, and they are the subject of Section 2.

---

## 2. Group Sequential Designs

### 2.1 Setup and Information Fractions

**Definition (Group Sequential Design).** A group sequential design specifies:
- A maximum sample size $n_{\max}$ per arm.
- $K$ planned *interim analyses* at sample sizes $n_1 < n_2 < \cdots < n_K = n_{\max}$.
- *Information fractions* $t_k = n_k / n_{\max} \in (0, 1]$ for $k = 1, \ldots, K$, with $t_K = 1$.
- Critical values $c_1, \ldots, c_K$ (possibly varying across looks) chosen to satisfy the FWER constraint at level $\alpha$.

The trial stops at the first look $k$ where $|Z_k| > c_k$ (efficacy stopping), or continues to look $K$ if no boundary is crossed.

### 2.2 The Covariance Structure of Sequential Z-Statistics

Let $X_1, X_2, \ldots$ be i.i.d. observations with mean $\mu$ and variance $\sigma^2$. Define the cumulative sum $S_{n} = \sum_{i=1}^{n} X_i$. At look $k$ with sample size $n_k$, the standardized test statistic is

$$Z_k = \frac{S_{n_k}}{\sigma \sqrt{n_k}}.$$

**Proposition (Covariance of Sequential Statistics).** For $j \leq k$,

$$\text{Cov}(Z_j, Z_k) = \sqrt{\frac{t_j}{t_k}},$$

where $t_j = n_j / n_{\max}$ and $t_k = n_k / n_{\max}$.

*Proof sketch.* Since $j \leq k$, we have $n_j \leq n_k$, and the partial sum $S_{n_j}$ is contained in $S_{n_k}$:

$$\text{Cov}(Z_j, Z_k) = \text{Cov}\!\left(\frac{S_{n_j}}{\sigma\sqrt{n_j}},\, \frac{S_{n_k}}{\sigma\sqrt{n_k}}\right).$$

Write $S_{n_k} = S_{n_j} + (S_{n_k} - S_{n_j})$. Since the observations are i.i.d., $S_{n_k} - S_{n_j}$ is independent of $S_{n_j}$. Therefore

$$\text{Cov}(Z_j, Z_k) = \frac{1}{\sigma^2 \sqrt{n_j n_k}}\, \text{Cov}(S_{n_j}, S_{n_k}) = \frac{1}{\sigma^2 \sqrt{n_j n_k}}\, \text{Var}(S_{n_j}) = \frac{\sigma^2 n_j}{\sigma^2 \sqrt{n_j n_k}} = \sqrt{\frac{n_j}{n_k}}.$$

Since $n_j/n_k = (n_j/n_{\max})/(n_k/n_{\max}) = t_j/t_k$, we conclude $\text{Cov}(Z_j, Z_k) = \sqrt{t_j/t_k}$. $\square$

*Remark.* Because $\text{Var}(Z_k) = 1$ for all $k$, the covariance and correlation coincide: $\text{Corr}(Z_j, Z_k) = \sqrt{t_j/t_k}$. This structure is the *canonical joint distribution* of group sequential statistics, and it holds asymptotically even for non-normal $X_i$ by the multivariate CLT.

### 2.3 Pocock Boundaries

**Definition (Pocock Boundary).** The Pocock procedure uses a *constant* critical value $c_P$ at every look: stop if $|Z_k| > c_P$ for any $k$. The value $c_P$ is the unique solution to

$$P\!\left(\max_{k=1,\ldots,K} |Z_k| > c_P \;\middle|\; H_0\right) = \alpha,$$

where $(Z_1, \ldots, Z_K)$ is multivariate normal with the covariance structure derived in Section 2.2.

For $K = 2$ equally-spaced looks with $\alpha = 0.05$ (two-sided), the fixed-horizon critical value is $z_{0.025} = 1.96$, whereas the Pocock boundary yields $c_P \approx 2.178$. The constant boundary is more lenient at interim looks — it allows early stopping more readily than O'Brien-Fleming — but it pays for this with a higher final critical value. A study that proceeds to look $K$ must clear a bar of $2.178$ rather than $1.96$, reducing power.

The Pocock boundary is appropriate when early stopping is scientifically desirable and the analyst is willing to pay a modest power penalty throughout.

### 2.4 O'Brien-Fleming Boundaries

**Definition (O'Brien-Fleming Boundary).** The O'Brien-Fleming procedure uses a *decreasing* sequence of critical values that are very large early (making early stopping rare) and converge to approximately $z_{\alpha/2}$ at the final look. The boundary at look $k$ takes the approximate form

$$c_k^{\text{OBF}} \approx c_{\text{OBF}} \cdot \sqrt{K/k},$$

where $c_{\text{OBF}}$ is the constant chosen so that the joint probability of crossing the boundary at any look equals $\alpha$.

*Surprisingly,* the O'Brien-Fleming boundary at the final look ($k = K$) is very close to the fixed-horizon $z_{\alpha/2} = 1.96$ even with many interim looks. For $K = 5$ looks with $\alpha = 0.05$, the final critical value is approximately $2.040$ — barely higher than $1.96$. This means the O'Brien-Fleming design imposes almost no penalty at the final analysis; the cost of interim monitoring is borne almost entirely at the interim looks (where the boundaries are very high and early stopping is unlikely).

This property makes O'Brien-Fleming the dominant choice in clinical trials: the final decision is made under essentially the same standard as a fixed-horizon test, while still permitting early stopping for extreme early evidence.

### 2.5 Sample Size Inflation

A group sequential design requires a larger maximum sample size $n_{\max}$ than a fixed-horizon test $n_{\text{fixed}}$ to achieve the same power, because the elevated critical values at interim looks (and slightly elevated final critical value) reduce the probability of rejection. The *inflation factor* is $R = n_{\max}/n_{\text{fixed}}$.

For O'Brien-Fleming with $K = 5$ equally-spaced looks and $\alpha = 0.05$, $\beta = 0.20$ (power 80%), the inflation factor is approximately

$$R \approx 1.028.$$

**The sample size inflation for O'Brien-Fleming with $K = 5$ is only about 3%, making it an almost cost-free addition to a fixed-horizon design.** Pocock boundaries incur a larger inflation (around 6–10% for $K = 5$) in exchange for more aggressive early stopping.

---

## 3. Alpha-Spending Functions

### 3.1 The Lan-DeMets Framework

Group sequential designs as described in Section 2 require the number and timing of interim analyses to be fixed in advance. In practice, looks are often triggered by logistical events (a batch of data arriving, a monthly review meeting) rather than by pre-specified sample sizes. The *alpha-spending* framework of Lan and DeMets (1983) accommodates this flexibility.

**Definition (Alpha-Spending Function).** An alpha-spending function is a function $\alpha^*: [0,1] \to [0,\alpha]$ satisfying:
1. $\alpha^*(0) = 0$.
2. $\alpha^*(1) = \alpha$.
3. $\alpha^*$ is non-decreasing in the information fraction $t$.

The function $\alpha^*(t)$ specifies the cumulative amount of the Type I error budget that has been "spent" by information fraction $t$.

The key insight is that $\alpha^*$ is chosen once and fixed before the trial. The number and timing of looks need not be pre-specified; only $\alpha^*$ must be pre-committed.

### 3.2 Incremental Spending and Boundary Computation

At look $k$ (reached at information fraction $t_k$), the incremental alpha to be spent is

$$\delta_k = \alpha^*(t_k) - \alpha^*(t_{k-1}), \quad \alpha^*(t_0) \equiv 0.$$

The critical value $c_k$ at look $k$ is chosen as the unique value satisfying

$$P\!\left(|Z_k| > c_k \;\middle|\; |Z_j| \leq c_j \text{ for all } j < k,\; H_0\right) = \delta_k.$$

That is, conditional on not having stopped at any earlier look, the probability of stopping at look $k$ under $H_0$ is exactly $\delta_k$.

*Remark.* The $c_k$ values are computed recursively. At look 1, $c_1 = z_{\delta_1/2}$ (two-sided). At look $k > 1$, $c_k$ is determined by a $k$-dimensional normal integral over the region $\{|Z_j| \leq c_j : j < k\}$, using the covariance structure $\text{Cov}(Z_j, Z_k) = \sqrt{t_j/t_k}$.

### 3.3 Common Spending Functions

Two spending functions are standard, designed to approximate the Pocock and O'Brien-Fleming boundaries respectively.

**Pocock-like spending (Lan-DeMets).** The spending function

$$\alpha^*(t) = \alpha \ln\!\bigl(1 + (e-1)\,t\bigr)$$

spends approximately $\alpha \cdot t$ of the total budget by information fraction $t$ — it spends liberally early. The increments $\delta_k$ are roughly equal across looks, mimicking the constant critical value of Pocock boundaries.

**O'Brien-Fleming-like spending (Lan-DeMets).** The spending function

$$\alpha^*(t) = 2\!\left[1 - \Phi\!\left(\frac{z_{\alpha/2}}{\sqrt{t}}\right)\right]$$

spends very little early (when $t$ is small, $z_{\alpha/2}/\sqrt{t}$ is large, so $1 - \Phi(\cdot)$ is tiny) and concentrates spending near $t = 1$. This mimics O'Brien-Fleming: early critical values are high, and the final critical value is close to $z_{\alpha/2}$.

### 3.4 Key Property: FWER Preservation

**Proposition (FWER control under alpha-spending).** For any number and timing of looks, if each look $k$ spends exactly $\delta_k = \alpha^*(t_k) - \alpha^*(t_{k-1})$ of the Type I error budget, then

$$P\!\left(\exists\, k : |Z_k| > c_k \;\middle|\; H_0\right) = \sum_{k=1}^{K} \delta_k = \alpha^*(t_K) = \alpha.$$

*Proof sketch.* The events $A_k = \{|Z_k| > c_k, |Z_j| \leq c_j \text{ for all } j < k\}$ are mutually disjoint (the trial stops at the first crossing). Therefore

$$P\!\left(\bigcup_k A_k \;\middle|\; H_0\right) = \sum_k P(A_k \mid H_0) = \sum_k \delta_k = \alpha.$$

The FWER is maintained at exactly $\alpha$ regardless of the number or timing of looks, provided only the pre-committed spending function is used. $\square$

---

## 4. The Multi-Armed Bandit Problem

### 4.1 Formal Setup

The *multi-armed bandit* problem is a sequential decision problem. At each round $t = 1, 2, \ldots, T$, an agent selects an arm $a_t \in \mathcal{A} = \{1, \ldots, K\}$ and receives a stochastic reward $R_{a_t, t}$. The rewards $\{R_{k,t}\}_{t \geq 1}$ for arm $k$ are i.i.d. with mean $\mu_k \in [0,1]$ and are independent across arms. The means $\mu_1, \ldots, \mu_K$ are unknown to the agent.

Denote $\mu^* = \max_{k} \mu_k$ the optimal arm mean and $k^* \in \argmax_k \mu_k$ the optimal arm (assumed unique for simplicity). Let $N_k(t) = \sum_{s=1}^{t} \mathbf{1}[a_s = k]$ denote the number of times arm $k$ has been pulled through round $t$.

### 4.2 Regret and Its Decomposition

**Definition (Regret).** The *cumulative regret* of a policy $\pi$ over $T$ rounds is

$$\text{Regret}(T) = T\mu^* - \mathbb{E}\!\left[\sum_{t=1}^{T} R_{a_t, t}\right].$$

This is the expected reward foregone by not always pulling the optimal arm. Since $\mathbb{E}[R_{a_t,t}] = \mu_{a_t}$ and $T\mu^* = \sum_{t=1}^T \mu^*$, we may write

$$\text{Regret}(T) = \mathbb{E}\!\left[\sum_{t=1}^{T} (\mu^* - \mu_{a_t})\right] = \sum_{k=1}^{K} \Delta_k\, \mathbb{E}[N_k(T)],$$

where $\Delta_k = \mu^* - \mu_k \geq 0$ is the *suboptimality gap* of arm $k$. The decomposition shows that regret accumulates only from pulls of suboptimal arms ($\Delta_k > 0$), weighted by how frequently each suboptimal arm is pulled.

### 4.3 The Exploration-Exploitation Tradeoff

A greedy policy that always pulls the arm with the highest empirical mean $\hat{\mu}_k(t)$ exploits current knowledge but may never explore arms whose true mean is higher than the current estimate. Conversely, uniform exploration wastes pulls on clearly suboptimal arms.

The tension between these objectives is the *exploration-exploitation tradeoff*. A policy must explore enough to identify the optimal arm but not so much that it incurs large regret from suboptimal pulls. The optimal resolution yields *minimax regret*:

$$\min_\pi \max_{\mu} \text{Regret}(T) = \Theta\!\left(\sqrt{KT\log K}\right).$$

### 4.4 Lower Bound: Lai and Robbins (1985)

The fundamental impossibility result for bandits, due to Lai and Robbins (1985), establishes that no consistent policy can avoid logarithmic regret.

**Theorem (Lai-Robbins Lower Bound).** For any policy that is *consistent* (i.e., $\text{Regret}(T) = o(T^\alpha)$ for all $\alpha > 0$ and all bandit instances), we have

$$\liminf_{T \to \infty} \frac{\text{Regret}(T)}{\ln T} \geq \sum_{k:\, \Delta_k > 0} \frac{\Delta_k}{\text{KL}(\mu_k,\, \mu^*)},$$

where $\text{KL}(\mu_k, \mu^*)$ is the Kullback-Leibler divergence between the reward distributions of arm $k$ and the optimal arm $k^*$ evaluated at their means.

For Bernoulli rewards with parameters $p$ (suboptimal arm) and $q$ (optimal arm, $q > p$), the KL divergence is

$$\text{KL}(p, q) = p \ln\frac{p}{q} + (1-p)\ln\frac{1-p}{1-q}.$$

*Intuition.* To confidently identify that arm $k$ is suboptimal, the agent must pull arm $k$ enough times to statistically distinguish $\mu_k$ from $\mu^*$. The number of pulls required to distinguish two Bernoulli distributions grows as $\ln T / \text{KL}(\mu_k, \mu^*)$. Any fewer pulls and the policy cannot be consistent; hence the $\ln T$ lower bound on expected pulls of each suboptimal arm, and by the regret decomposition, on regret.

---

## 5. Epsilon-Greedy and UCB

### 5.1 Epsilon-Greedy

The *epsilon-greedy* policy at round $t$ selects a uniformly random arm with probability $\epsilon_t$ (exploration) and the arm with the highest current empirical mean $\hat{\mu}_k(t-1)$ with probability $1 - \epsilon_t$ (exploitation).

With a constant $\epsilon$, the policy incurs linear regret $\Theta(\epsilon T)$ from forced exploration. With the decaying schedule $\epsilon_t = c/t$ for an appropriate constant $c$, the regret satisfies $\text{Regret}(T) = O(T^{2/3})$. *This is strictly suboptimal* relative to the $\Theta(\ln T)$ lower bound of Lai and Robbins, because the schedule $c/t$ is too slow to adapt: early estimates are poor, yet the exploration rate decays regardless of whether the optimal arm has been identified.

### 5.2 UCB1

The *Upper Confidence Bound* strategy (Auer, Cesa-Bianchi, and Fischer, 2002) maintains, for each arm $k$, an *optimistic* upper confidence bound on $\mu_k$ and always selects the arm with the highest upper bound.

**Definition (UCB1).** At round $t$, after each arm has been pulled at least once, UCB1 selects

$$a_t = \argmax_{k \in \{1,\ldots,K\}} \left[\hat{\mu}_k(t-1) + \sqrt{\frac{2 \ln t}{N_k(t-1)}}\right],$$

where $\hat{\mu}_k(t-1) = \frac{1}{N_k(t-1)} \sum_{s < t} R_{k,s} \mathbf{1}[a_s = k]$ is the empirical mean of arm $k$ after $t-1$ rounds.

The term $\sqrt{2\ln t / N_k(t-1)}$ is the *exploration bonus*: it is large when arm $k$ has been pulled rarely (large uncertainty) and shrinks as $N_k(t-1)$ grows. An arm is pulled whenever it plausibly could be optimal given the current data.

### 5.3 Regret Bound for UCB1

**Theorem (UCB1 Regret Bound, Auer et al. 2002).** For a $K$-armed bandit with rewards in $[0,1]$, UCB1 satisfies

$$\mathbb{E}[\text{Regret}(T)] \leq \sum_{k:\,\Delta_k > 0} \frac{8 \ln T}{\Delta_k} + \left(1 + \frac{\pi^2}{3}\right) \sum_{k=1}^K \Delta_k.$$

The first term dominates for large $T$; the second is a constant independent of $T$.

*Key step in the derivation.* Fix a suboptimal arm $k$ with gap $\Delta_k > 0$. Arm $k$ is pulled at round $t$ only if its UCB exceeds the UCB of arm $k^*$, which means either the UCB of $k^*$ is below $\mu^*$ (a low-probability event) or the UCB of $k$ is above $\mu^*$ (also low-probability once $k$ has been pulled sufficiently often). Specifically, for $l = \lceil 8 \ln T / \Delta_k^2 \rceil$, the event $\{N_k(T) > l\}$ requires that for some round $t$ with $N_k(t-1) \geq l$, the UCB of arm $k$ still exceeded that of arm $k^*$. By a Hoeffding-type concentration argument, the probability that $\hat{\mu}_k + \sqrt{2\ln t / N_k} \geq \mu^*$ when $N_k \geq l$ is at most $t^{-4}$. Summing over $t = 1, \ldots, T$ gives

$$\mathbb{E}[N_k(T)] \leq \frac{8 \ln T}{\Delta_k^2} + 1 + \frac{\pi^2}{3},$$

yielding the bound after multiplication by $\Delta_k$ and summation over suboptimal arms.

**The UCB1 bound is $O(\ln T)$ per suboptimal arm and $O(\sqrt{KT\log T})$ in the minimax sense** — matching the Lai-Robbins lower bound up to the $\Delta_k$ factors.

### 5.4 KL-UCB

The *KL-UCB* algorithm (Garivier and Cappé, 2011) replaces the Euclidean exploration bonus $\sqrt{2\ln t / N_k}$ with a KL-divergence-based upper confidence bound:

$$\text{UCB}_k^{\text{KL}}(t) = \sup\!\left\{q \in [0,1] : N_k(t-1)\, \text{KL}\!\left(\hat{\mu}_k(t-1), q\right) \leq \ln t + 3\ln\ln t\right\}.$$

This upper bound is tighter than the Euclidean bound because it uses the actual geometry of the Bernoulli family. KL-UCB achieves the Lai-Robbins lower bound asymptotically: $\text{Regret}(T)/\ln T \to \sum_k \Delta_k/\text{KL}(\mu_k, \mu^*)$ as $T \to \infty$.

---

## 6. Thompson Sampling

### 6.1 Algorithm

*Thompson sampling* (also called *posterior sampling* or *probability matching*) is a Bayesian algorithm that selects arms in proportion to the probability that each arm is optimal under the current posterior.

**Algorithm (Thompson Sampling for Bernoulli Bandits).**

Initialize $\alpha_k = 1$, $\beta_k = 1$ (uniform prior) for all $k$. At each round $t$:
1. For each arm $k$, sample $\theta_k^{(t)} \sim \text{Beta}(\alpha_k, \beta_k)$.
2. Pull arm $a_t = \argmax_k \theta_k^{(t)}$.
3. Observe reward $R_t \in \{0, 1\}$.
4. Update: $\alpha_{a_t} \leftarrow \alpha_{a_t} + R_t$ and $\beta_{a_t} \leftarrow \beta_{a_t} + (1 - R_t)$.

The Beta distribution is the conjugate prior for the Bernoulli likelihood (see the companion note `bayesian-testing.md`): after $s$ successes and $f$ failures, the posterior is $\text{Beta}(\alpha_k + s, \beta_k + f)$.

### 6.2 Bayesian Regret

In the Bayesian setting, the arm means $\mu_1, \ldots, \mu_K$ are themselves drawn from a prior distribution $\pi$.

**Definition (Bayesian Regret).** The Bayesian regret of a policy $\pi$ is

$$\text{BayesRegret}(T) = \mathbb{E}_{\mu \sim \pi}\!\left[\text{Regret}_\mu(T)\right],$$

where $\text{Regret}_\mu(T)$ is the frequentist regret for the specific instance $\mu = (\mu_1, \ldots, \mu_K)$.

**Theorem (Russo and Van Roy, 2014).** Thompson sampling with a prior $\pi$ over arm means achieves Bayesian regret

$$\text{BayesRegret}(T) = O\!\left(\sqrt{KT \ln K}\right).$$

The proof uses an information-theoretic argument: at each round, the expected information gain about the optimal arm from the pulled arm's reward bounds the expected instantaneous regret.

### 6.3 Frequentist Regret and Asymptotic Optimality

A remarkable property of Thompson sampling is that its frequentist regret guarantees are equally strong.

**Theorem (Agrawal and Goyal, 2012/2013).** Thompson sampling with a Beta prior for Bernoulli bandits satisfies, for each suboptimal arm $k$,

$$\limsup_{T \to \infty} \frac{\mathbb{E}[N_k(T)]}{\ln T} \leq \frac{1}{\text{KL}(\mu_k, \mu^*)},$$

which, by the regret decomposition, implies

$$\limsup_{T \to \infty} \frac{\text{Regret}(T)}{\ln T} \leq \sum_{k:\,\Delta_k > 0} \frac{\Delta_k}{\text{KL}(\mu_k, \mu^*)}.$$

**This matches the Lai-Robbins lower bound exactly — Thompson sampling is asymptotically optimal.** The result is striking: a Bayesian algorithm designed for Bayesian regret minimization achieves the information-theoretic frequentist lower bound.

### 6.4 The Probability Matching Property

The key conceptual insight behind Thompson sampling is the *probability matching* property.

**Proposition (Probability Matching).** Under Thompson sampling, at each round $t$, the probability of selecting arm $k$ equals the posterior probability that arm $k$ is optimal:

$$P(a_t = k \mid \mathcal{F}_{t-1}) = P\!\left(\theta_k^{(t)} > \theta_j^{(t)} \text{ for all } j \neq k \;\middle|\; \mathcal{F}_{t-1}\right) = P(\mu_k = \mu^* \mid \mathcal{F}_{t-1}),$$

where $\mathcal{F}_{t-1}$ is the sigma-algebra generated by all observations through round $t-1$.

*Proof.* By definition, $a_t = \argmax_k \theta_k^{(t)}$ where $\theta_k^{(t)} \sim \text{posterior}_k$. The probability of selecting arm $k$ is therefore the probability that the sample from arm $k$'s posterior exceeds all others, which equals the posterior probability that $\mu_k$ is the true maximum (since $(\theta_1^{(t)}, \ldots, \theta_K^{(t)})$ is a joint sample from the posterior). $\square$

The probability matching property gives Thompson sampling its self-regulating character: arms that are likely optimal are pulled often (exploitation), while arms with wide posteriors (uncertain, possibly optimal) are also pulled when their posterior sample happens to be high (exploration). The exploration bonus is implicit in the posterior variance rather than explicitly computed as in UCB.

### 6.5 Implementation Advantages

Thompson sampling has several practical advantages over UCB-type algorithms:

- **Simplicity:** the update rule is closed-form for conjugate priors (Beta-Bernoulli, Normal-Normal, Gamma-Poisson).
- **Non-stationarity:** decaying the posterior (e.g., multiplying $(\alpha_k, \beta_k)$ by a discount factor $\gamma < 1$ at each round) allows the algorithm to track drifting arm means.
- **Contextual bandits:** Thompson sampling extends naturally to contextual settings where arm means depend on observed covariates, by placing a prior over the regression weights.

*This bound only holds for fixed prior $\pi$; misspecified priors can degrade frequentist performance.*

---

## 7. Bandits vs. Fixed A/B Tests

### 7.1 Fixed A/B Test: Strengths and Regime

A fixed A/B test (with or without group sequential interim looks) has the following properties:

- **Pre-specified sample size:** power and Type I/II error are guaranteed before the experiment begins.
- **Equal allocation:** each arm receives $n/K$ observations, maximizing statistical efficiency for the "which arm is best" question.
- **Strong error control:** FWER (with multiple comparisons) and FDR (with BH correction, see `experimental-design.md`) are precisely controlled.
- **No opportunity cost exploitation:** the fixed test does not shift traffic toward better-performing arms mid-experiment.

Use a fixed A/B test when: regulatory requirements demand pre-specified error bounds, a formal hypothesis test is needed at the end (e.g., for product launch decisions requiring statistical proof), multiple hypotheses must be tested with joint error control, or the experiment window is short and fixed.

### 7.2 Bandits: Strengths and Regime

A bandit algorithm has the following properties:

- **Adaptive allocation:** traffic is shifted toward better-performing arms over time, minimizing opportunity cost.
- **No fixed horizon:** the algorithm can run indefinitely, adapting as arm means drift.
- **Weaker testing guarantees:** the arm selected most often at time $T$ is a biased estimator of the best arm; formal confidence intervals for $\mu_k$ are distorted by the adaptive sampling design.
- **Regret minimization:** the objective is cumulative reward maximization, not error rate control.

Use a bandit when: the experiment horizon is long and the opportunity cost of showing an inferior treatment is large, no formal hypothesis test is required at the end (e.g., a recommendation system that just needs to serve the best content), traffic is streaming with no natural end date, or $K$ is large and many arms need simultaneous exploration.

### 7.3 The Opportunity Cost Framing

The opportunity cost of running a fixed A/B test with $n$ total observations per arm and true effect size $\Delta = \mu_B - \mu_A > 0$ (where arm $B$ is better) is the expected reward lost by not immediately deploying arm $B$:

$$\text{Opportunity Cost}_{\text{fixed}} = \frac{n \Delta}{2}.$$

This is because, on average, half the traffic ($n/2$ observations) is assigned to the inferior arm $A$.

For a bandit algorithm that achieves regret $\text{Regret}(T) \approx C \ln T / \Delta$, the opportunity cost equals the regret, which grows only logarithmically in $T$. **When $\Delta$ is large, the bandit's opportunity cost is dramatically smaller than the fixed test's.** When $\Delta$ is small, $n$ must be large (recall $n \propto 1/\Delta^2$ for fixed power), and the bandit's $\ln T$ growth may not compensate — the comparison depends on $\Delta$ and $T$.

*In the limit of small $\Delta$, fixed tests can be more sample-efficient for the hypothesis-testing objective, but bandits minimize total opportunity cost when regret is the right objective.*

### 7.4 Hybrid Approaches

Two principal approaches combine sequential validity with hypothesis-testing guarantees.

**Always-valid inference (Johari, Pekelis, and Walsh, 2015).** Constructs a *mixture sequential probability ratio test* (mSPRT) that yields a $p$-value process $\{p_t\}_{t \geq 1}$ satisfying

$$P\!\left(\exists\, t : p_t \leq \alpha \;\middle|\; H_0\right) \leq \alpha$$

for all stopping times. The analyst may stop and reject at any time the $p$-value crosses $\alpha$, without inflating the Type I error. This is a sequential version of the fixed-horizon test that supports continuous monitoring with valid inference at any peek.

**mSPRT construction.** Fix a mixing distribution $G$ over effect sizes $\theta$. The test statistic is

$$\Lambda_t = \int \prod_{s=1}^{t} \frac{p(X_s \mid \theta)}{p(X_s \mid 0)}\, dG(\theta).$$

Under $H_0$, $\Lambda_t$ is a non-negative martingale with $\mathbb{E}[\Lambda_t \mid H_0] = 1$. By Ville's inequality (a uniform version of Markov's inequality for supermartingales), $P(\exists\, t : \Lambda_t \geq 1/\alpha \mid H_0) \leq \alpha$, so the always-valid $p$-value $p_t = 1/\Lambda_t$ controls the Type I error at any stopping time.

---

## 8. References

| Reference Name | Brief Summary | Link to Reference |
|---|---|---|
| Lai and Robbins (1985) | Establishes the asymptotic lower bound $\Omega(\ln T)$ on regret for consistent policies; the foundational impossibility result for multi-armed bandits | [Advances in Applied Mathematics](https://dl.acm.org/doi/10.1016/0196-8858(85)90002-8) |
| Auer, Cesa-Bianchi, and Fischer (2002) | Introduces UCB1 and proves the first finite-time logarithmic regret bound for multi-armed bandits; derives the $O(\ln T / \Delta_k^2)$ pulls bound for suboptimal arms | [Machine Learning, 47:235–256](https://link.springer.com/article/10.1023/A:1013689704352) |
| Russo and Van Roy (2014) | Proves $O(\sqrt{KT \ln K})$ Bayesian regret for Thompson sampling using an information-theoretic analysis; introduces the eluder dimension for general function classes | [JMLR, 17:1–52](https://www.jmlr.org/papers/volume17/14-087/14-087.pdf) |
| Agrawal and Goyal (2012/2013) | Proves that Thompson sampling achieves the Lai-Robbins lower bound asymptotically in the frequentist sense; first finite-time instance-dependent regret bound for Thompson sampling | [JACM, 64(5)](https://dl.acm.org/doi/10.1145/3088510) |
| Johari, Pekelis, and Walsh (2015) | Introduces always-valid inference via the mSPRT; enables continuous monitoring of A/B tests without Type I error inflation at any stopping time | [arXiv:1512.04922](https://arxiv.org/abs/1512.04922) |
| Lan and DeMets (1983) | Introduces the alpha-spending function framework for group sequential designs; allows flexible, unscheduled interim analyses while maintaining FWER $= \alpha$ | [Biometrika, 70(3):659–663](https://link.springer.com/chapter/10.1007/978-1-4615-2009-2_1) |
| O'Brien and Fleming (1979) | Proposes the group sequential boundary that is conservative at interim looks and close to the fixed-horizon critical value at the final look; the standard choice in clinical trials | [Biometrics, 35(3):549–556](https://www.semanticscholar.org/paper/A-multiple-testing-procedure-for-clinical-trials.-O'Brien-Fleming/d40fee01a7708099f9e9392f10ac0b370b7ed8a8) |
| Russo, Van Roy, Kazerouni, Osband, and Wen (2018) | Comprehensive tutorial on Thompson sampling covering theory, algorithms, and applications including contextual bandits and reinforcement learning | [Foundations and Trends in ML, 11(1):1–99](https://arxiv.org/abs/1707.02038) |
| Garivier and Cappé (2011) | Introduces KL-UCB, which achieves the Lai-Robbins lower bound asymptotically by using KL-divergence-based confidence bounds | [COLT 2011](https://arxiv.org/abs/1102.2490) |
