# Rejection Sampling as Reinforcement Learning

> **Anchored to:** Zeng et al. (2025), [*A Minimalist Approach to LLM Reasoning*](https://arxiv.org/abs/2504.11343)

## Table of Contents

- [[#1. The RL Framework for LLM Post-Training|1. The RL Framework for LLM Post-Training]]
  - [[#1.1 The KL-Regularized Objective|1.1 The KL-Regularized Objective]]
  - [[#1.2 The Closed-Form Optimal Policy|1.2 The Closed-Form Optimal Policy]]
- [[#2. Rejection Sampling RAFT|2. Rejection Sampling: RAFT]]
  - [[#2.1 The Algorithm|2.1 The Algorithm]]
  - [[#2.2 Why This Is RL|2.2 Why This Is RL]]
  - [[#2.3 The Imitation Learning Interpretation|2.3 The Imitation Learning Interpretation]]
- [[#3. The Distribution Shift Problem and RAFT++|3. The Distribution Shift Problem and RAFT++]]
  - [[#3.1 Why Shift Happens|3.1 Why Shift Happens]]
  - [[#3.2 Importance Sampling Correction|3.2 Importance Sampling Correction]]
- [[#4. Entropy Collapse|4. Entropy Collapse]]
- [[#5. Policy Gradient Methods|5. Policy Gradient Methods]]
  - [[#5.1 REINFORCE|5.1 REINFORCE]]
  - [[#5.2 GRPO Group Relative Policy Optimization|5.2 GRPO: Group Relative Policy Optimization]]
  - [[#5.3 The Role of Negative Samples|5.3 The Role of Negative Samples]]
- [[#6. Reinforce-Rej Bridging the Gap|6. Reinforce-Rej: Bridging the Gap]]
- [[#7. Summary Comparison|7. Summary Comparison]]
- [[#References|References]]

---

## 1. The RL Framework for LLM Post-Training 🎯

After supervised pre-training, a language model $\pi_\text{ref}$ has learned to predict text but not to optimize for a *specific goal* — e.g., solving math problems correctly or following human instructions. *Reinforcement learning* provides the framework for this second stage.

The setup:
- **State:** the prompt $x$ (e.g., a math problem)
- **Action:** a generated response $y = (a_1, a_2, \ldots, a_T)$ (a sequence of tokens)
- **Reward:** a scalar $r(x, y)$ evaluating the quality of the response (e.g., 1 if correct, 0 if not)
- **Policy:** $\pi_\theta(y|x)$, the language model parameterized by $\theta$

Unlike classical RL with short action sequences, the "action" here is an entire token sequence — the credit assignment problem is therefore collapsed: the reward arrives only at the end of the full generation.

### 1.1 The KL-Regularized Objective

A naive objective $\max_\theta \mathbb{E}_{y \sim \pi_\theta}[r(x,y)]$ is unstable: the model can exploit the reward function by drifting arbitrarily far from $\pi_\text{ref}$, producing degenerate outputs that score well but are incoherent (reward hacking).

The standard fix is to add a KL regularization penalty:

$$\mathcal{J}(\theta) = \mathbb{E}_{x \sim \mathcal{D}}\left[\mathbb{E}_{y \sim \pi_\theta(\cdot|x)}\left[r(x,y)\right] - \beta \cdot D_\text{KL}\!\left(\pi_\theta(\cdot|x) \,\|\, \pi_\text{ref}(\cdot|x)\right)\right]$$

where $\beta > 0$ is a temperature that controls the strength of the regularization. The KL term penalizes the policy for deviating from the reference model, which:

1. Prevents reward hacking
2. Preserves the language modeling prior (fluency, coherence)
3. Keeps the optimization well-conditioned

### 1.2 The Closed-Form Optimal Policy

The KL-regularized objective is a *convex functional* over the policy distribution. Taking the functional derivative and setting it to zero yields the **closed-form optimal policy**:

$$\boxed{\pi^*(y|x) = \frac{1}{Z(x)}\,\pi_\text{ref}(y|x)\,\exp\!\left(\frac{r(x,y)}{\beta}\right)}$$

where $Z(x) = \sum_y \pi_\text{ref}(y|x)\,\exp(r(x,y)/\beta)$ is the partition function (intractable to compute exactly).

> [!NOTE] Derivation sketch
> Expand the KL term: $D_\text{KL}(\pi \| \pi_\text{ref}) = \sum_y \pi(y|x)\log\frac{\pi(y|x)}{\pi_\text{ref}(y|x)}$. Setting $\frac{\delta \mathcal{J}}{\delta \pi} = 0$ with a Lagrange multiplier for normalization gives $\log \pi^*(y|x) = \log \pi_\text{ref}(y|x) + \frac{1}{\beta}r(x,y) - \log Z(x)$.

**Key result:** $\pi^*$ is a *Gibbs distribution* (Boltzmann distribution) over responses, where the reference model provides the base measure and the reward acts as an energy function. Higher reward → exponentially higher probability under $\pi^*$.

> [!TIP] Intuition for $\beta$
> - $\beta \to \infty$: the KL penalty dominates, so $\pi^* \approx \pi_\text{ref}$ — no learning happens.
> - $\beta \to 0$: the reward dominates, so $\pi^*$ concentrates all mass on the highest-reward response — mode collapse.
> - Intermediate $\beta$: balances exploration (staying broad like $\pi_\text{ref}$) with exploitation (concentrating on high-reward regions).

---

> [!QUESTION] Exercise 1: Partition Function Lower Bound
> *This problem shows that even without computing $Z(x)$, its structure constrains the optimal policy.*
>
> > **Prerequisites:** [[#1.2 The Closed-Form Optimal Policy|§1.2 The Closed-Form Optimal Policy]]
>
> Show that for binary reward $r \in \{0, 1\}$, the partition function satisfies $Z(x) \geq p_\text{ref}(x)$, where $p_\text{ref}(x)$ is the probability the reference model generates a correct answer. What does this imply about the normalization of $\pi^*$ relative to $\pi_\text{ref}$?

> [!TIP]- Solution to Exercise 1
> **Key insight:** $Z(x) = \sum_y \pi_\text{ref}(y|x) e^{r(x,y)/\beta} \geq \sum_{y: r=1} \pi_\text{ref}(y|x) e^{1/\beta} = e^{1/\beta} p_\text{ref}(x) \geq p_\text{ref}(x)$, with equality iff $p_\text{ref}(x) = 0$.
>
> **Sketch:** Since $Z(x) \geq 1 \cdot \pi_\text{ref}(\text{any single } y)$ and also $Z(x) \geq p_\text{ref}(x) e^{1/\beta}$, the normalization $Z(x)$ is at least as large as the reference model's success probability inflated by $e^{1/\beta}$. This means $\pi^*(y|x) = \pi_\text{ref}(y|x) e^{r/\beta} / Z(x) \leq \pi_\text{ref}(y|x) / p_\text{ref}(x)$ for correct $y$ — the optimal policy redistributes mass from incorrect to correct responses relative to $\pi_\text{ref}$.

---

## 2. Rejection Sampling: RAFT 🎲

*RAFT* (Reward-rAnked Fine-Tuning) is the simplest possible algorithm for post-training: generate candidates, keep the good ones, imitate them.

### 2.1 The Algorithm

For each prompt $x$ in the training set:

1. **Sample** $n$ responses $y_1, \ldots, y_n \sim \pi_\theta(\cdot|x)$ from the current policy
2. **Filter** to the accepted set $\mathcal{D}_+ = \{(x, y_i) : r(x, y_i) = 1\}$ (correct answers only)
3. **Fine-tune** on $\mathcal{D}_+$ via maximum likelihood:

$$\mathcal{L}^\text{RAFT}(\theta) = -\sum_{(x,y) \in \mathcal{D}_+} \log \pi_\theta(y|x)$$

This is just supervised fine-tuning (SFT) on filtered data.

### 2.2 Why This Is RL

At first glance, RAFT looks like supervised learning — it is just SFT on some data. But the data is *generated and filtered by a reward signal*, which makes it RL in a deep sense.

📐 **The formal connection:** Sampling $y \sim \pi_\theta$ and accepting iff $r(x,y)=1$ defines a distribution over accepted samples:

$$q(y|x) \propto \pi_\theta(y|x) \cdot \mathbf{1}[r(x,y)=1]$$

For binary reward, this is exactly the optimal policy $\pi^*$ restricted to the $\beta \to 0$ limit — the Gibbs distribution concentrates on the correct responses proportionally to $\pi_\theta$. *The accepted sample distribution is an approximation of $\pi^*$*, and RAFT trains $\pi_\theta$ to imitate $q$.

> [!INFO] Connection to classical rejection sampling
> In statistics, *rejection sampling* draws from a target distribution $p^*(y)$ by (1) sampling $y \sim q(y)$ from a proposal distribution and (2) accepting with probability $p^*(y) / (M \cdot q(y))$ where $M$ is a normalizing constant. Here, the "proposal" is $\pi_\theta$, the "target" is $\pi^*$, and the acceptance criterion $r=1$ is a binary approximation to the acceptance probability.

### 2.3 The Imitation Learning Interpretation

RAFT can be read as *behavioral cloning* from an improved demonstrator:

1. Run the current policy $\pi_\theta$ to collect rollouts
2. Filter rollouts using the reward as a success criterion
3. Imitate the successful rollouts with SFT

This is the *imitation learning* paradigm: learn from positive examples of the desired behavior. The contrast with policy gradient methods is stark:

| | RAFT | REINFORCE |
|---|---|---|
| Uses negative samples? | ❌ | ✅ |
| Gradient through reward? | ❌ | ✅ |
| Loss type | SFT (MLE) | Policy gradient |
| Credit assignment | Implicit (filter) | Explicit (gradient) |

---

> [!QUESTION] Exercise 2: RAFT as Maximum Likelihood of the Optimal Policy
> *This problem establishes the formal connection between RAFT and the KL-regularized RL objective.*
>
> > **Prerequisites:** [[#1.2 The Closed-Form Optimal Policy|§1.2 The Closed-Form Optimal Policy]], [[#2.1 The Algorithm|§2.1 The Algorithm]]
>
> Show that the RAFT objective $\mathcal{L}^\text{RAFT}(\theta) = -\mathbb{E}_{(x,y) \sim \mathcal{D}_+}[\log \pi_\theta(y|x)]$ is equivalent (up to a constant) to minimizing $D_\text{KL}(\hat{\pi}^* \| \pi_\theta)$, where $\hat{\pi}^*$ is the empirical approximation of the optimal policy $\pi^*$ supported on the accepted samples. What does this imply about what RAFT is "trying" to do?

> [!TIP]- Solution to Exercise 2
> **Key insight:** $D_\text{KL}(\hat\pi^* \| \pi_\theta) = \sum_{(x,y) \in \mathcal{D}_+} \hat\pi^*(y|x)\log\hat\pi^*(y|x) - \hat\pi^*(y|x)\log\pi_\theta(y|x)$. The first term is the entropy of $\hat\pi^*$ — constant w.r.t. $\theta$. So minimizing KL is equivalent to maximizing $\sum \hat\pi^* \log \pi_\theta$, which (with uniform weighting over $\mathcal{D}_+$) is exactly $\mathcal{L}^\text{RAFT}$.
>
> **Sketch:** RAFT is doing *maximum likelihood estimation* of $\pi^*$ — it is fitting $\pi_\theta$ to match the accepted sample distribution (a Monte Carlo approximation of $\pi^*$). This is RL via *distribution matching*, not via explicit gradient signals.

---

## 3. The Distribution Shift Problem and RAFT++ ⚠️

### 3.1 Why Shift Happens

RAFT has a subtle flaw: the accepted samples $\mathcal{D}_+$ are collected from policy $\pi_{\theta_\text{old}}$, but after each gradient update, $\pi_\theta$ changes. On the *next* iteration, samples are collected from the updated $\pi_\theta$, but if a *replay buffer* is used (reusing old data), the data distribution no longer matches the current policy.

This is the *distribution shift* (or *off-policy*) problem. Formally, the loss being minimized is:

$$\mathbb{E}_{y \sim \pi_{\theta_\text{old}}(\cdot|x), r(x,y)=1}[\log \pi_\theta(y|x)]$$

but the policy being updated is $\pi_\theta \neq \pi_{\theta_\text{old}}$. The gradient is a *biased* estimator of the on-policy gradient.

### 3.2 Importance Sampling Correction

RAFT++ corrects for this bias using *importance sampling*. Define the per-token importance ratio:

$$s_t(\theta) = \frac{\pi_\theta(a_t | x, a_{1:t-1})}{\pi_{\theta_\text{old}}(a_t | x, a_{1:t-1})}$$

The corrected loss clips the ratio to $[1-\epsilon, 1+\epsilon]$ (PPO-style) to prevent large updates:

$$\mathcal{L}^\text{RAFT++}(\theta) = \frac{1}{|\mathcal{D}|}\sum_{(x,a) \in \mathcal{D}}\frac{1}{|a|}\sum_{t=1}^{|a|}\min\!\left(s_t(\theta),\, \text{clip}(s_t(\theta), 1{-}\epsilon, 1{+}\epsilon)\right) \cdot \mathbf{1}[r(x,a) = 1]$$

> [!NOTE] Why importance sampling works
> The key identity is $\mathbb{E}_{y \sim \pi_\text{old}}[f(y)] = \mathbb{E}_{y \sim \pi}[f(y) \cdot \frac{\pi_\text{old}(y)}{\pi(y)}]$, which is trivially reversible. By reweighting samples from $\pi_{\theta_\text{old}}$ by $\pi_\theta / \pi_{\theta_\text{old}}$, we get an unbiased estimate of the expectation under $\pi_\theta$.

*RAFT++ is essentially PPO applied only to positive samples.* The clipping prevents the policy from moving too far from where the samples were collected, stabilizing training.

---

## 4. Entropy Collapse 📉

A critical problem with RAFT (and to a lesser degree RAFT++) is *entropy collapse*: the policy's output distribution rapidly narrows, concentrating mass on a small set of high-reward responses and losing diversity.

Formally, the policy entropy is:

$$\mathcal{H}(\pi_\theta(\cdot|x)) = -\sum_y \pi_\theta(y|x)\log\pi_\theta(y|x)$$

RAFT only applies *positive gradient* (increasing $\pi_\theta(y|x)$ for $y$ with $r=1$) with no corresponding *negative gradient* (decreasing $\pi_\theta(y|x)$ for $y$ with $r=0$). The normalization constraint $\sum_y \pi_\theta(y|x) = 1$ means that increasing mass on accepted responses implicitly decreases mass on all others — but this implicit decrease is uncontrolled and non-specific.

The consequence:
- Early in training: broad exploration, many diverse solutions found
- As training continues: the policy concentrates on a few known-good patterns
- Entropy falls sharply → exploration ceases → performance plateaus

> [!WARNING] Entropy collapse vs. reward maximization
> *High entropy is not a goal in itself* — you want the policy to concentrate on correct answers. But premature collapse means the model stops exploring and gets stuck at a local optimum. The key is that entropy collapse in RAFT happens too fast, before the policy has found high-quality diverse solutions.

---

> [!QUESTION] Exercise 3: Entropy Dynamics Under Positive-Only Training
> *This problem builds intuition for why positive-only updates collapse entropy.*
>
> > **Prerequisites:** [[#4. Entropy Collapse|§4 Entropy Collapse]]
>
> Consider a toy setting: binary output space $\{0, 1\}$ with reward $r(1)=1$, $r(0)=0$, and policy $\pi_\theta(1|x) = p$. The RAFT update maximizes $\log p$ (increases $p$). Show that the entropy $\mathcal{H}(p) = -p\log p - (1-p)\log(1-p)$ is monotonically decreasing under this update once $p > 0.5$. What does this say about the long-run behavior of RAFT?

> [!TIP]- Solution to Exercise 3
> **Key insight:** $\frac{d\mathcal{H}}{dp} = -\log p + \log(1-p) = \log\frac{1-p}{p}$. This is negative iff $p > 0.5$. The RAFT gradient always increases $p$ toward 1, so once $p > 0.5$, entropy monotonically decreases. In the limit, $p \to 1$ and $\mathcal{H} \to 0$ — the policy deterministically outputs the correct answer (or the answer it has come to believe is correct).
>
> **Sketch:** This explains entropy collapse: positive-only training pushes $p \to 1$, which drives entropy to zero. In a multi-dimensional setting with many possible responses, the same logic applies — without repulsive gradients from negative samples, the policy eventually degenerates to a near-deterministic distribution over a small set of learned patterns.

---

## 5. Policy Gradient Methods 📐

### 5.1 REINFORCE

The policy gradient theorem gives a tractable estimator of $\nabla_\theta \mathcal{J}$:

$$\nabla_\theta \mathcal{J}(\theta) = \mathbb{E}_{x \sim \mathcal{D}}\left[\mathbb{E}_{y \sim \pi_\theta(\cdot|x)}\left[\nabla_\theta \log \pi_\theta(y|x) \cdot r(x,y)\right]\right]$$

The key identity behind this is $\nabla_\theta \mathbb{E}[\cdot] = \mathbb{E}[\nabla_\theta \log \pi_\theta \cdot r]$ (the log-derivative trick). With PPO-style clipping for stability:

$$\mathcal{L}^\text{REINFORCE}(\theta) = \frac{1}{|\mathcal{D}|}\sum_{(x,a) \in \mathcal{D}}\frac{1}{|a|}\sum_{t=1}^{|a|}\min\!\left(s_t(\theta),\, \text{clip}(s_t(\theta), 1{-}\epsilon, 1{+}\epsilon)\right) \cdot r(x,a)$$

For binary reward, this produces:
- Positive gradient on correct responses ($r=1$): increase their probability
- Negative gradient on incorrect responses ($r=0$): **no gradient** (since $r=0$ zeroes out the term)

Wait — zero reward doesn't produce a negative gradient. To see the *real* mechanism by which negative samples help, we need to look at baselines.

> [!NOTE] Why a baseline matters
> With a *baseline* $b$, the gradient uses the *advantage* $r(x,y) - b$ instead of raw reward. For prompts where all responses are incorrect, the advantage is $0 - b < 0$ — a genuine negative gradient that decreases the probability of those responses. This is the key mechanism by which negative samples aid learning.

### 5.2 GRPO: Group Relative Policy Optimization

GRPO samples $n$ responses per prompt and computes a *within-group normalized advantage*:

$$A(x, y_i) = \frac{r_i - \mu_r}{\sigma_r}, \quad \mu_r = \frac{1}{n}\sum_{j=1}^n r_j, \quad \sigma_r = \sqrt{\frac{1}{n}\sum_{j=1}^n (r_j - \mu_r)^2}$$

This normalization has two effects:
1. **Centering:** correct responses in a group where most are correct get low positive advantage; correct responses in a mostly-incorrect group get high positive advantage
2. **Scaling:** the gradient magnitude is standardized across prompts

*Surprisingly*, the paper shows that this normalization contributes minimally to GRPO's empirical advantage over REINFORCE. The dominant benefit of GRPO over RAFT is the use of negative samples via the advantage function.

### 5.3 The Role of Negative Samples

The paper makes a nuanced distinction about negative samples:

| Prompt type | Effect of including |
|---|---|
| Mixed (some correct, some not) | ✅ Helpful — provides signal about what to avoid |
| All-incorrect | ❌ Harmful — high-variance gradient with no useful positive signal |
| All-correct | ❌ Harmful — trivially high reward, no learning signal |

*Filtering prompts where all responses are incorrect* is what actually drives GRPO's gains. This is the key insight of Reinforce-Rej.

---

## 6. Reinforce-Rej: Bridging the Gap 🔑

**Reinforce-Rej** is the paper's proposed algorithm: run REINFORCE with clipping, but filter out prompts where all responses are correct or all are incorrect:

$$\mathcal{D}^\text{Rej} = \{(x, \{y_i\}) : 0 < \sum_i r_i < n\}$$

This keeps only *informative* prompts — those with mixed outcomes — and trains REINFORCE on them with normalized advantages.

The result: **comparable performance to GRPO** with:
- Better KL efficiency (achieves the same reward with less divergence from $\pi_\text{ref}$)
- Better entropy stability than RAFT (no collapse)
- Simpler implementation than full GRPO

> [!INFO] Why all-incorrect prompts are harmful
> When all $n$ samples are incorrect ($r_i = 0$ for all $i$), after advantage normalization the advantages become $A_i = (0 - 0)/0$ — undefined (or numerically zero after smoothing). More importantly, these prompts provide no positive learning signal: there is no correct response to push toward. Yet they contribute gradient noise that destabilizes training.

---

## 7. Summary Comparison 📊

| Algorithm | Uses negatives | Distribution shift | Entropy stability | Sample efficiency | Complexity |
|---|---|---|---|---|---|
| RAFT | ❌ | Ignored | Poor (collapses) | High (SFT only) | Very low |
| RAFT++ | ❌ | Importance sampling | Moderate | High | Low |
| REINFORCE | ✅ | Importance sampling | Good | Moderate | Moderate |
| GRPO | ✅ | Importance sampling | Good | Moderate | Moderate |
| Reinforce-Rej | ✅ (filtered) | Importance sampling | Good | Moderate | Low |

**The big picture:** The gap between RAFT and GRPO is not about algorithmic sophistication — it is about whether negative samples provide a repulsive gradient. Reinforce-Rej shows this gap can be closed with a simple filtering rule.

---

## References

| Reference Name | Brief Summary | Link |
|---|---|---|
| Zeng et al. (2025), *A Minimalist Approach to LLM Reasoning* | Compares RAFT, RAFT++, REINFORCE, GRPO for LLM math reasoning; introduces Reinforce-Rej; shows entropy collapse and negative sample filtering are the key factors | [arXiv:2504.11343](https://arxiv.org/abs/2504.11343) |
| Dong et al. (2023), *RAFT: Reward rAnked FineTuning* | Original RAFT paper proposing rejection-sampling-based fine-tuning for alignment | [arXiv:2304.06767](https://arxiv.org/abs/2304.06767) |
| Shao et al. (2024), *DeepSeekMath* | Introduces GRPO (Group Relative Policy Optimization) for mathematical reasoning | [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) |
| Ziegler et al. (2019), *Fine-Tuning Language Models from Human Preferences* | Foundational RLHF paper establishing the KL-regularized RL objective for LLMs | [arXiv:1909.08593](https://arxiv.org/abs/1909.08593) |
| Rafailov et al. (2023), *Direct Preference Optimization* | Shows that the optimal policy for KL-regularized RL is the Gibbs distribution; derives DPO from this | [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) |
| Williams (1992), *Simple Statistical Gradient-Following Algorithms* | Original REINFORCE algorithm; establishes the log-derivative trick for policy gradient estimation | [Machine Learning 8(3-4)](https://link.springer.com/article/10.1007/BF00992696) |
