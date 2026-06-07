# Q-Learning and Value-Based Reinforcement Learning

## Table of Contents

- [[#1. The MDP Framework|1. The MDP Framework]]
- [[#2. Value Functions|2. Value Functions]]
  - [[#2.1 State-Value and Action-Value Functions|2.1 State-Value and Action-Value Functions]]
  - [[#2.2 Bellman Consistency Equations|2.2 Bellman Consistency Equations]]
- [[#3. Optimal Value Functions|3. Optimal Value Functions]]
  - [[#3.1 The Bellman Optimality Equations|3.1 The Bellman Optimality Equations]]
  - [[#3.2 The Bellman Operator Is a Contraction|3.2 The Bellman Operator Is a Contraction]]
- [[#4. Value Iteration|4. Value Iteration]]
- [[#5. Q-Learning Stochastic Approximation of Value Iteration|5. Q-Learning: Stochastic Approximation of Value Iteration]]
  - [[#5.1 The Update Rule|5.1 The Update Rule]]
  - [[#5.2 Convergence|5.2 Convergence]]
  - [[#5.3 SARSA On-Policy TD|5.3 SARSA: On-Policy TD]]
- [[#6. The Deadly Triad and Deep Q-Learning|6. The Deadly Triad and Deep Q-Learning]]
  - [[#6.1 The Deadly Triad|6.1 The Deadly Triad]]
  - [[#6.2 DQN|6.2 DQN]]
  - [[#6.3 Double DQN Fixing Overestimation|6.3 Double DQN: Fixing Overestimation]]
- [[#7. Scope and Limitations|7. Scope and Limitations]]
- [[#References|References]]

---

## 1. The MDP Framework 🎯

Q-learning operates inside the *Markov Decision Process* (MDP) formalism. An MDP is a tuple $(\mathcal{S}, \mathcal{A}, P, r, \gamma)$:

- $\mathcal{S}$: state space
- $\mathcal{A}$: action space
- $P(s' \mid s, a)$: transition kernel — probability of reaching $s'$ from $(s,a)$
- $r(s, a)$: bounded reward function
- $\gamma \in [0, 1)$: discount factor

A *policy* $\pi: \mathcal{S} \to \Delta(\mathcal{A})$ maps states to distributions over actions. The agent's goal is to find the policy $\pi^*$ that maximizes expected discounted return from every state simultaneously.

The Markov property — $s_{t+1} \perp s_{0:t-1}, a_{0:t-1} \mid s_t, a_t$ — is the assumption that the current state contains all information relevant to the future. This is what makes value functions well-defined.

---

## 2. Value Functions 📐

### 2.1 State-Value and Action-Value Functions

For a fixed policy $\pi$, define:

**State-value function:**
$$V^\pi(s) = \mathbb{E}_\pi\!\left[\sum_{t=0}^\infty \gamma^t r(s_t, a_t) \;\bigg|\; s_0 = s\right]$$

**Action-value function (Q-function):**
$$Q^\pi(s, a) = \mathbb{E}_\pi\!\left[\sum_{t=0}^\infty \gamma^t r(s_t, a_t) \;\bigg|\; s_0 = s,\, a_0 = a\right]$$

The Q-function conditions on a specific first action $a$, then follows $\pi$ thereafter. The two functions are related by:

$$V^\pi(s) = \mathbb{E}_{a \sim \pi(\cdot|s)}\!\left[Q^\pi(s, a)\right] = \sum_a \pi(a \,|\, s)\,Q^\pi(s, a)$$

The Q-function is the more useful object for *control*: given $Q^\pi$, improving the policy is trivial — just act greedily. Given only $V^\pi$, you need the model $P$ to compute $\mathbb{E}[V^\pi(s')]$ and evaluate actions.

### 2.2 Bellman Consistency Equations

$Q^\pi$ satisfies a *self-consistency* (Bellman) equation. Expanding the return by peeling off the first step:

$$Q^\pi(s, a) = \mathbb{E}_{s' \sim P(\cdot|s,a)}\!\left[r(s,a) + \gamma\,V^\pi(s')\right] = \mathbb{E}_{s'}\!\left[r(s,a) + \gamma\sum_{a'}\pi(a'|s')\,Q^\pi(s',a')\right]$$

Define the *Bellman operator for policy $\pi$*:

$$(\mathcal{T}^\pi Q)(s,a) = \mathbb{E}_{s'}\!\left[r(s,a) + \gamma\sum_{a'}\pi(a'|s')\,Q(s',a')\right]$$

Then $Q^\pi$ is the unique fixed point $\mathcal{T}^\pi Q^\pi = Q^\pi$.

> [!NOTE] Why the Bellman equation is useful
> It transforms the problem of computing an infinite sum into solving a fixed-point equation. The fixed-point can be found by iterating $\mathcal{T}^\pi$ (policy evaluation) or solved in closed form for tabular MDPs: $Q^\pi = (I - \gamma P^\pi)^{-1} r$ where $P^\pi_{(s,a),(s',a')} = P(s'|s,a)\pi(a'|s')$.

---

> [!QUESTION] Exercise 1: Bellman Equation from First Principles
> *This problem derives the Bellman consistency equation directly from the definition of $Q^\pi$.*
>
> > **Prerequisites:** [[#2.1 State-Value and Action-Value Functions|§2.1 State-Value and Action-Value Functions]]
>
> Starting from $Q^\pi(s,a) = \mathbb{E}_\pi[\sum_{t=0}^\infty \gamma^t r(s_t,a_t) \mid s_0=s, a_0=a]$, split the sum at $t=0$ and $t \geq 1$. Use the Markov property to show the $t \geq 1$ terms collapse to $\gamma Q^\pi(s', a')$ for some $(s', a')$, and average over the transition and policy to obtain the Bellman equation.

> [!TIP]- Solution to Exercise 1
> **Key insight:** The Markov property means the return from time 1 onward, conditioned on $(s_1, a_1)$, is independent of $(s_0, a_0)$ and has the same distribution as $Q^\pi(s_1, a_1)$.
>
> **Sketch:** $Q^\pi(s,a) = \mathbb{E}[r(s_0,a_0)] + \mathbb{E}[\sum_{t=1}^\infty \gamma^t r_t \mid s_0,a_0]$. Factor out $\gamma$: $= r(s,a) + \gamma\,\mathbb{E}_{s_1 \sim P}[\mathbb{E}_{a_1 \sim \pi}[\mathbb{E}[\sum_{t=0}^\infty \gamma^t r_{t+1} \mid s_1,a_1]]]$. By the Markov property the inner expectation is $Q^\pi(s_1,a_1)$, giving $r(s,a) + \gamma\,\mathbb{E}_{s',a'}[Q^\pi(s',a')]$.

---

## 3. Optimal Value Functions 🔑

### 3.1 The Bellman Optimality Equations

Define the *optimal* value functions:

$$V^*(s) = \max_\pi V^\pi(s), \qquad Q^*(s,a) = \max_\pi Q^\pi(s,a)$$

These exist and are unique for discounted MDPs (by compactness of the policy space and linearity of the value functional in $\pi$ for fixed $s,a$). They satisfy the *Bellman optimality equations*:

$$Q^*(s,a) = \mathbb{E}_{s' \sim P(\cdot|s,a)}\!\left[r(s,a) + \gamma\,\max_{a'}\,Q^*(s',a')\right]$$

$$V^*(s) = \max_a\,\mathbb{E}_{s'}\!\left[r(s,a) + \gamma\,V^*(s')\right]$$

**Key structural fact:** Any policy $\pi^*$ satisfying $\pi^*(s) \in \arg\max_a Q^*(s,a)$ is optimal. Once $Q^*$ is known, deriving the optimal policy requires no further computation — just a greedy table lookup.

> [!INFO] Comparison with the KL-regularized objective
> The standard Bellman optimality equation has a hard $\max$ over actions. The KL-regularized RL objective studied in [[concepts/reinforcement-learning/rejection-sampling-as-rl|rejection sampling as RL]] replaces $\max_a$ with a soft max (Boltzmann/Gibbs distribution), producing a *soft* Bellman equation:
> $$Q^*(s,a) = r(s,a) + \gamma\,\mathbb{E}_{s'}\!\left[\beta\log\sum_{a'}\exp\!\left(\frac{Q^*(s',a')}{\beta}\right)\right]$$
> The hard case ($\beta \to 0$) recovers standard Q-learning.

### 3.2 The Bellman Operator Is a Contraction

Define the *Bellman optimality operator*:

$$(\mathcal{T}Q)(s,a) = \mathbb{E}_{s'}\!\left[r(s,a) + \gamma\,\max_{a'}\,Q(s',a')\right]$$

**Theorem.** $\mathcal{T}$ is a contraction on $(\ell^\infty(\mathcal{S} \times \mathcal{A}), \|\cdot\|_\infty)$ with factor $\gamma$:

$$\|\mathcal{T}Q - \mathcal{T}Q'\|_\infty \leq \gamma\,\|Q - Q'\|_\infty$$

**Proof.** For any $(s,a)$:

$$|(\mathcal{T}Q)(s,a) - (\mathcal{T}Q')(s,a)| = \gamma\,\left|\mathbb{E}_{s'}\!\left[\max_{a'}Q(s',a') - \max_{a'}Q'(s',a')\right]\right|$$

Use the fact that $|\max_a f(a) - \max_a g(a)| \leq \max_a |f(a) - g(a)|$ and linearity of expectation:

$$\leq \gamma\,\mathbb{E}_{s'}\!\left[\max_{a'}\left|Q(s',a') - Q'(s',a')\right|\right] \leq \gamma\,\|Q - Q'\|_\infty$$

Taking the sup over $(s,a)$ gives $\|\mathcal{T}Q - \mathcal{T}Q'\|_\infty \leq \gamma\|Q-Q'\|_\infty$. $\square$

By Banach's fixed-point theorem: $Q^*$ is the unique fixed point of $\mathcal{T}$, and for any initialization $Q_0$:

$$\|Q_k - Q^*\|_\infty \leq \gamma^k\,\|Q_0 - Q^*\|_\infty$$

*convergence is geometric in $k$ with rate $\gamma$.*

---

> [!QUESTION] Exercise 2: Contraction Implies Unique Fixed Point
> *This problem derives the convergence rate directly from the contraction.*
>
> > **Prerequisites:** [[#3.2 The Bellman Operator Is a Contraction|§3.2 The Bellman Operator Is a Contraction]]
>
> Let $Q^* = \mathcal{T}Q^*$ and $Q_k = \mathcal{T}Q_{k-1}$. Using only the contraction property $\|\mathcal{T}Q - \mathcal{T}Q'\|_\infty \leq \gamma\|Q-Q'\|_\infty$, prove: (a) $Q^*$ is the unique fixed point; (b) $\|Q_k - Q^*\|_\infty \leq \gamma^k \|Q_0 - Q^*\|_\infty$; (c) how many iterations of $\mathcal{T}$ are needed to reduce the initial error by a factor of $\epsilon$?

> [!TIP]- Solution to Exercise 2
> **Key insight:** Contraction + completeness of $\ell^\infty$ → Banach fixed-point theorem directly.
>
> **Sketch:**
> (a) If $Q^* = \mathcal{T}Q^*$ and $Q^{**} = \mathcal{T}Q^{**}$, then $\|Q^* - Q^{**}\|_\infty = \|\mathcal{T}Q^* - \mathcal{T}Q^{**}\|_\infty \leq \gamma\|Q^*-Q^{**}\|_\infty$. Since $\gamma < 1$, this forces $\|Q^*-Q^{**}\|_\infty = 0$.
> (b) By induction: $\|Q_k - Q^*\|_\infty = \|\mathcal{T}Q_{k-1} - \mathcal{T}Q^*\|_\infty \leq \gamma\|Q_{k-1}-Q^*\|_\infty \leq \cdots \leq \gamma^k\|Q_0-Q^*\|_\infty$.
> (c) Need $\gamma^k \leq \epsilon$, so $k \geq \log(1/\epsilon)/\log(1/\gamma) = O\!\left(\frac{\log(1/\epsilon)}{1-\gamma}\right)$ — the $1/(1-\gamma)$ factor reflects that near $\gamma=1$ (long-horizon problems), many more iterations are needed.

---

## 4. Value Iteration 🔄

*Value iteration* is the direct application of the Bellman operator: start from any $Q_0$ and iterate $Q_{k+1} = \mathcal{T}Q_k$. Explicitly, for all $(s,a)$:

$$Q_{k+1}(s,a) = \sum_{s'} P(s'|s,a)\left[r(s,a) + \gamma\,\max_{a'} Q_k(s',a')\right]$$

By the contraction theorem, this converges to $Q^*$ geometrically. The greedy policy $\pi_{k+1}(s) = \arg\max_a Q_{k+1}(s,a)$ converges to $\pi^*$.

**Requirement:** computing $\mathcal{T}Q_k$ requires summing over all $s'$ weighted by $P(s'|s,a)$ — the *transition model*. Value iteration is a *model-based* planning algorithm.

**Complexity:** each sweep costs $O(|\mathcal{S}|^2 |\mathcal{A}|)$ — feasible for small tabular MDPs, intractable for large or continuous state spaces.

> [!EXAMPLE] Gridworld
> In a 5×5 grid with 4 actions (N/S/E/W), $|\mathcal{S}| = 25$ and $|\mathcal{A}| = 4$. Value iteration converges in $O(\log(1/\epsilon)/(1-\gamma))$ sweeps. Each sweep visits all 100 state-action pairs and looks up transition probabilities. This is feasible by hand; for Atari with $|\mathcal{S}| \approx 10^{70}$ pixel states, it is obviously not.

---

## 5. Q-Learning: Stochastic Approximation of Value Iteration 💡

Q-learning removes the model requirement by replacing the exact expectation in $\mathcal{T}$ with a single sampled transition.

### 5.1 The Update Rule

Given a transition $(s, a, r, s')$ observed from the environment:

$$Q(s,a) \leftarrow Q(s,a) + \alpha_t\,\underbrace{\bigl[r + \gamma\max_{a'} Q(s',a') - Q(s,a)\bigr]}_{\delta_t\;:\;\text{TD error}}$$

**The TD error $\delta_t$** is a noisy estimate of $(\mathcal{T}Q)(s,a) - Q(s,a)$: if $\delta_t > 0$, $Q(s,a)$ was too pessimistic about $(s,a)$ given this transition; if $\delta_t < 0$, too optimistic. The update moves $Q(s,a)$ in the direction of the *one-sample Bellman target* $r + \gamma\max_{a'} Q(s',a')$.

**Off-policy.** The update uses $\max_{a'} Q(s',a')$ — the greedy action — regardless of what the *behavior policy* (the policy used to collect the transition) actually did. This makes Q-learning *off-policy*: data can come from any distribution over $(s,a)$ pairs (e.g., a replay buffer, a random policy, a human demonstrator) as long as it covers all $(s,a)$.

**Contrast with SARSA.** SARSA replaces $\max_{a'} Q(s',a')$ with $Q(s', a'_\text{behavior})$, where $a'_\text{behavior} \sim \pi_\text{behavior}(\cdot|s')$ is the action actually taken next. This makes SARSA *on-policy* — it evaluates the behavior policy rather than the greedy policy.

| | Q-Learning | SARSA |
|---|---|---|
| Target | $r + \gamma\max_{a'} Q(s',a')$ | $r + \gamma Q(s', a')$, $a' \sim \pi_b$ |
| On/off-policy | Off-policy | On-policy |
| Converges to | $Q^*$ (optimal) | $Q^{\pi_b}$ (behavior policy) |
| Cliff-walking behavior | Takes risky optimal path | Takes safe suboptimal path |

### 5.2 Convergence

**Theorem (Watkins & Dayan, 1992).** Tabular Q-learning with learning rates $\{\alpha_t\}$ converges to $Q^*$ almost surely, provided:

1. **Exploration:** every $(s,a)$ pair is visited infinitely often
2. **Robbins–Monro:** $\sum_t \alpha_t(s,a) = \infty$ and $\sum_t \alpha_t(s,a)^2 < \infty$ for all $(s,a)$
3. **Bounded rewards:** $|r| \leq r_\text{max} < \infty$

The Robbins–Monro condition (e.g., $\alpha_t = 1/t$) ensures the updates are large enough to eventually correct any initialization, but small enough that noise averages out. The proof uses *stochastic approximation* theory (Borkar, Tsitsiklis): the Q-learning update is an *asynchronous* stochastic version of value iteration, and the noise (from single-sample transitions) has conditional mean zero given $Q_t$.

> [!WARNING] Exploration requirement is non-trivial
> *Every $(s,a)$ must be visited infinitely often* — this requires the behavior policy to be *GLIE* (Greedy in the Limit with Infinite Exploration): $\epsilon$-greedy with $\epsilon_t \to 0$ as $t \to \infty$, but $\sum_t \epsilon_t = \infty$. Without this, Q-learning converges to $Q^{\pi_b}$ (the value of the behavior policy), not $Q^*$.

### 5.3 SARSA: On-Policy TD

For completeness, SARSA (State-Action-Reward-State-Action) uses the *actual next action* in the target:

$$Q(s,a) \leftarrow Q(s,a) + \alpha\bigl[r + \gamma Q(s', a') - Q(s,a)\bigr], \quad a' \sim \pi(\cdot|s')$$

SARSA converges to $Q^{\pi}$ for the *current behavior policy* $\pi$. If $\pi$ is $\epsilon$-greedy and $\epsilon \to 0$, SARSA converges to $Q^*$ — but more slowly than Q-learning because it evaluates the exploratory policy, not the greedy one.

---

> [!QUESTION] Exercise 3: Q-Learning as Stochastic Value Iteration
> *This problem makes precise the sense in which Q-learning approximates value iteration.*
>
> > **Prerequisites:** [[#4. Value Iteration|§4 Value Iteration]], [[#5.1 The Update Rule|§5.1 The Update Rule]]
>
> Write the Q-learning update as $Q_{t+1}(s,a) = (1-\alpha_t) Q_t(s,a) + \alpha_t [r + \gamma\max_{a'} Q_t(s',a')]$. (a) Show this is a convex combination of the old estimate and the one-sample Bellman target. (b) Take $\mathbb{E}[\cdot | Q_t, s, a]$ of the update and show that the expected update direction is $\mathcal{T}Q_t - Q_t$ (the value iteration step). (c) What does this imply about the noise $\delta_t - \mathbb{E}[\delta_t | Q_t]$?

> [!TIP]- Solution to Exercise 3
> **Key insight:** Q-learning is exactly value iteration plus zero-mean noise — stochastic approximation in the sense of Robbins–Monro.
>
> **Sketch:**
> (a) $Q_{t+1}(s,a) = (1-\alpha_t)Q_t(s,a) + \alpha_t \cdot \text{target}$: this is an interpolation with weights $(1-\alpha_t, \alpha_t)$ summing to 1, so it's a convex combination.
> (b) $\mathbb{E}[\text{target}|Q_t,s,a] = \mathbb{E}_{s'}[r + \gamma\max_{a'}Q_t(s',a')] = (\mathcal{T}Q_t)(s,a)$. So $\mathbb{E}[Q_{t+1} - Q_t | Q_t] = \alpha_t [(\mathcal{T}Q_t)(s,a) - Q_t(s,a)]$, which is $\alpha_t$ times the value iteration update.
> (c) The noise $\xi_t = [r + \gamma\max_{a'}Q_t(s',a')] - (\mathcal{T}Q_t)(s,a)$ has $\mathbb{E}[\xi_t | Q_t] = 0$ — it's mean-zero, the central condition for stochastic approximation convergence. Robbins–Monro then ensures the noise averages out over time.

---

## 6. The Deadly Triad and Deep Q-Learning ⚠️

### 6.1 The Deadly Triad

Tabular Q-learning converges because $Q(s,a)$ for each $(s,a)$ is an independent parameter — updating one doesn't change others. With *function approximation* $Q_\phi(s,a)$ (e.g., a neural network), updating the parameters $\phi$ changes $Q_\phi$ at *all* $(s,a)$ simultaneously. Three ingredients combine to cause potential divergence:

**1. Function approximation.** $Q_\phi$ is a smooth function of $\phi$; a gradient step on one $(s,a)$ moves $Q_\phi$ at nearby states too. The update is no longer a local table write.

**2. Bootstrapping.** The TD target $r + \gamma\max_{a'} Q_\phi(s',a')$ *depends on $Q_\phi$ itself*. You are fitting $Q_\phi$ toward a target that moves as $\phi$ moves — chasing a moving goalposts. The semi-gradient update:

$$\phi \leftarrow \phi + \alpha\,\delta_t\,\nabla_\phi Q_\phi(s,a)$$

does *not* gradient-descend any fixed loss — $\delta_t$ depends on $\phi$, but the gradient is taken only through $Q_\phi(s,a)$, not through the target. This is not a true gradient step.

**3. Off-policy training.** Replay buffer samples come from policies different from $\pi_\text{greedy}$, creating a mismatch between the distribution the function approximator is trained on and the distribution it's evaluated on.

Any two of these three are benign; all three together can cause divergence (Baird's counterexample demonstrates this for linear function approximation).

### 6.2 DQN

*DQN* (Mnih et al., 2015) stabilizes deep Q-learning with two engineering interventions:

**Experience replay.** Store transitions $(s,a,r,s')$ in a replay buffer $\mathcal{B}$. Sample mini-batches uniformly at random for gradient updates. This breaks temporal correlations (which destabilize SGD) and allows data reuse.

**Target network.** Maintain a *target network* $Q_{\phi^-}$ with parameters $\phi^-$ frozen for $k$ steps. Use $Q_{\phi^-}$ for the Bellman target:

$$\mathcal{L}(\phi) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{B}}\!\left[\bigl(r + \gamma\max_{a'} Q_{\phi^-}(s',a') - Q_\phi(s,a)\bigr)^2\right]$$

This is a proper least-squares regression problem (for the $k$ steps the target is frozen). Every $k$ steps, $\phi^- \leftarrow \phi$.

The target network slows the bootstrap feedback loop — the goalposts move, but slowly. Neither intervention is theoretically principled; both are empirical stabilizers. DQN still provably can diverge on some MDPs.

### 6.3 Double DQN: Fixing Overestimation

Q-learning has a systematic *overestimation bias*. The issue: the target uses $\max_{a'} Q_\phi(s',a')$, which is the maximum over noisy estimates. For independent noise $\eta_{a'}$:

$$\mathbb{E}\!\left[\max_{a'} (Q^*(s',a') + \eta_{a'})\right] \geq \max_{a'} Q^*(s',a')$$

(the expectation of the max exceeds the max of the expectations). The bias grows with $|\mathcal{A}|$ and the noise magnitude.

*Double DQN* (van Hasselt et al., 2016) decouples *action selection* from *action evaluation*:

$$\text{target} = r + \gamma\,Q_{\phi^-}\!\!\left(s',\,\underbrace{\arg\max_{a'} Q_\phi(s',a')}_{\text{online net selects action}}\right)$$

The online network $Q_\phi$ selects the greedy action; the target network $Q_{\phi^-}$ evaluates it. Since these are different networks with different noise, the maximum is no longer over correlated estimates — the bias shrinks substantially.

---

> [!QUESTION] Exercise 4: Overestimation Bias in Q-Learning
> *This problem derives the overestimation bound quantitatively.*
>
> > **Prerequisites:** [[#6.3 Double DQN Fixing Overestimation|§6.3 Double DQN: Fixing Overestimation]]
>
> Suppose $Q_\phi(s',a') = Q^*(s',a') + \eta_{a'}$ where $\eta_{a'} \sim \mathcal{N}(0, \sigma^2)$ i.i.d. for each $a'$. Let $|\mathcal{A}| = n$ and denote $m^* = \max_{a'} Q^*(s',a')$. (a) Show $\mathbb{E}[\max_{a'} Q_\phi(s',a')] \geq m^*$. (b) For the special case $Q^*(s',a') = 0$ for all $a'$, compute $\mathbb{E}[\max_{a'} \eta_{a'}]$ in terms of $\sigma$ and $n$. What does this say about large action spaces?

> [!TIP]- Solution to Exercise 4
> **Key insight:** The expected maximum of $n$ i.i.d. Gaussians grows as $\Theta(\sigma\sqrt{\log n})$.
>
> **Sketch:**
> (a) $\mathbb{E}[\max_{a'} Q_\phi(s',a')] = \mathbb{E}[\max_{a'}(Q^*(s',a') + \eta_{a'})] \geq \max_{a'} \mathbb{E}[Q^*(s',a') + \eta_{a'}] = \max_{a'} Q^*(s',a') = m^*$ by Jensen (since max is convex).
> (b) With $Q^*=0$: $\mathbb{E}[\max_{a'}\eta_{a'}] = \sigma\,\mathbb{E}[\max_{a'} Z_{a'}]$ where $Z_{a'} \sim \mathcal{N}(0,1)$. By the Gaussian order-statistic result, $\mathbb{E}[\max_n Z_i] \approx \sqrt{2\log n}$. So the bias is $\approx \sigma\sqrt{2\log n}$ — it grows logarithmically in $|\mathcal{A}|$. For large action spaces (e.g., $n=1000$), the overestimation is $\approx 3.7\sigma$ — substantial enough to cause meaningful policy degradation.

---

## 7. Scope and Limitations 🗺️

Q-learning and value-based RL have a specific niche; they are not the universal RL framework.

**Works well when:**
- Action space is discrete and small enough to enumerate ($\arg\max$ must be computable)
- Good exploration is achievable
- A replay buffer can be maintained (off-policy is a feature, not a bug)

**Fails or struggles when:**
- **Continuous actions:** $\arg\max_a Q_\phi(s,a)$ over continuous $\mathcal{A}$ requires a nested optimization loop. DDPG and TD3 sidestep this by learning a separate actor network $\mu_\psi(s) \approx \arg\max_a Q_\phi(s,a)$ — making them actor-critic methods, not pure Q-learning.
- **High-dimensional state + function approximation + off-policy:** the deadly triad. Convergence is not guaranteed.
- **Reward shaping sensitivity:** the greedy policy is sensitive to small errors in $Q_\phi$. Policy gradient methods, which optimize a smoother objective, can be more robust.
- **Long horizon / sparse reward:** the $\gamma^T$ discounting means distant rewards contribute negligibly. Reward shaping or hierarchical methods are needed.

**The broader RL landscape** (for context):

| Paradigm | What it learns | Key strength | Key weakness |
|---|---|---|---|
| Value-based (Q-learning) | $Q^*(s,a)$ | Off-policy, data efficient | Discrete actions only |
| Policy gradient (REINFORCE, PPO) | $\pi_\theta$ directly | Continuous actions, stable | On-policy, sample inefficient |
| Actor-critic (SAC, TD3) | Both $Q_\phi$ and $\pi_\theta$ | Continuous actions + efficiency | Complex, many hyperparameters |
| Model-based (Dyna, MuZero) | $P(s'|s,a)$ and/or $Q^*$ | Sample efficient | Model errors compound |

Value-based methods remain the standard for discrete-action problems (Atari, board games, combinatorial optimization). Policy gradient and actor-critic methods dominate continuous control (robotics, locomotion, LLM post-training).

---

## References

| Reference Name | Brief Summary | Link |
|---|---|---|
| Watkins (1989), *Learning from Delayed Rewards* | Introduces Q-learning; proves tabular convergence under GLIE and Robbins–Monro conditions | [PhD thesis, Cambridge](https://www.cs.rhul.ac.uk/~chrisw/new_thesis.pdf) |
| Watkins & Dayan (1992), *Q-Learning* | Published journal version of the Q-learning convergence proof | [Machine Learning 8(3)](https://link.springer.com/article/10.1007/BF00992698) |
| Mnih et al. (2015), *Human-Level Control through Deep Reinforcement Learning* | DQN: combines Q-learning with deep nets, experience replay, and target networks; achieves human-level Atari performance | [Nature 518](https://www.nature.com/articles/nature14236) |
| van Hasselt, Guez & Silver (2016), *Deep Reinforcement Learning with Double Q-Learning* | Identifies overestimation bias in DQN; Double DQN decouples action selection and evaluation to reduce it | [AAAI 2016](https://arxiv.org/abs/1509.06461) |
| Sutton & Barto (2018), *Reinforcement Learning: An Introduction* | Standard textbook; chapters 6–8 cover TD learning, Q-learning, SARSA, and the deadly triad in depth | [Online edition](http://incompleteideas.net/book/the-book-2nd.html) |
| Baird (1995), *Residual Algorithms* | Constructs explicit counterexample (Baird's counterexample) showing Q-learning with linear function approximation can diverge | [ICML 1995](https://dl.acm.org/doi/10.5555/3091622.3091624) |
| Tsitsiklis & Van Roy (1997), *An Analysis of Temporal-Difference Learning* | Proves convergence of TD(0) with linear function approximation under on-policy sampling; identifies off-policy instability | [IEEE Transactions on Automatic Control](https://ieeexplore.ieee.org/document/580874) |
