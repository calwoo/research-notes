# Lyapunov Functions and Stability: KL Divergence for the Replicator Dynamic

## Table of Contents

- [[#1. Introduction|1. Introduction]]
- [[#2. Lyapunov Stability Theory|2. Lyapunov Stability Theory]]
  - [[#2.1 Equilibria and Stability Notions|2.1 Equilibria and Stability Notions]]
  - [[#2.2 Lyapunov Functions and the Direct Method|2.2 Lyapunov Functions and the Direct Method]]
  - [[#2.3 LaSalle's Invariance Principle|2.3 LaSalle's Invariance Principle]]
- [[#3. The Replicator Dynamic|3. The Replicator Dynamic]]
  - [[#3.1 Finite Symmetric Games and the Strategy Simplex|3.1 Finite Symmetric Games and the Strategy Simplex]]
  - [[#3.2 The Replicator ODE|3.2 The Replicator ODE]]
  - [[#3.3 Fixed Points|3.3 Fixed Points]]
  - [[#3.4 Evolutionarily Stable Strategies|3.4 Evolutionarily Stable Strategies]]
- [[#4. KL Divergence as a Lyapunov Function|4. KL Divergence as a Lyapunov Function]]
  - [[#4.1 Setup and Statement|4.1 Setup and Statement]]
  - [[#4.2 Positivity of V|4.2 Positivity of V]]
  - [[#4.3 Computing V-dot along Replicator Trajectories|4.3 Computing V-dot along Replicator Trajectories]]
  - [[#4.4 Signing V-dot via the ESS Inequality|4.4 Signing V-dot via the ESS Inequality]]
  - [[#4.5 Asymptotic Stability Conclusion|4.5 Asymptotic Stability Conclusion]]
- [[#5. Geometric Interpretation|5. Geometric Interpretation]]
- [[#6. References|6. References]]

---

## 1. Introduction 🎯

The central question of dynamical systems theory is: given a differential equation $\dot{x} = f(x)$, do trajectories converge to a specific equilibrium? For nonlinear systems, linearization only tells a local story, and even that fails at non-hyperbolic equilibria. *Lyapunov's direct method* sidesteps linearization entirely: rather than solving the ODE, one constructs an *energy-like* function $V$ whose value decreases along every trajectory. Existence of such a function implies stability.

This note builds Lyapunov theory from scratch and applies it to a canonical example from evolutionary game theory. The main result we will prove is:

> **If $\hat{x}$ is an evolutionarily stable strategy (ESS) of a finite symmetric game, then the relative entropy $V(x) = D_{\mathrm{KL}}(\hat{x} \| x)$ is a Lyapunov function for the replicator dynamic, and $\hat{x}$ is asymptotically stable.**

The elegance here is not accidental. The KL divergence measures how far the population state $x$ is from the ESS $\hat{x}$ in an information-geometric sense — and the replicator dynamic turns out to be *gradient flow* with respect to the Fisher information metric on the simplex. The ESS condition is precisely the statement that the KL divergence is being minimized.

> [!INFO] Prerequisites
> This note assumes: basic ODE theory (existence/uniqueness, the notion of a flow), and familiarity with the KL divergence $D_{\mathrm{KL}}(p \| q) = \sum_i p_i \log(p_i / q_i)$ from information theory. Game-theoretic prerequisites are built from scratch in Section 3.

---

## 2. Lyapunov Stability Theory 📐

### 2.1 Equilibria and Stability Notions

Consider the autonomous ODE

$$\dot{x} = f(x), \qquad x \in \mathbb{R}^n, \quad f: U \to \mathbb{R}^n, \tag{2.1}$$

where $U \subseteq \mathbb{R}^n$ is open and $f$ is continuously differentiable. Denote the flow by $\phi_t(x_0)$, i.e., the unique solution with $\phi_0(x_0) = x_0$.

**Definition 2.1 (Equilibrium).** A point $x^* \in U$ is an *equilibrium* (or *rest point*, or *fixed point*) of (2.1) if $f(x^*) = 0$.

*Without loss of generality*, we translate so that $x^* = 0$. This is purely notational — any equilibrium can be shifted to the origin.

**Definition 2.2 (Lyapunov stability).** The equilibrium $x^* = 0$ is *Lyapunov stable* if for every $\epsilon > 0$ there exists $\delta > 0$ such that

$$\|x_0\| < \delta \implies \|\phi_t(x_0)\| < \epsilon \quad \text{for all } t \geq 0.$$

**Definition 2.3 (Asymptotic stability).** The equilibrium $x^* = 0$ is *asymptotically stable* if it is Lyapunov stable and there exists $\delta > 0$ such that

$$\|x_0\| < \delta \implies \phi_t(x_0) \to 0 \quad \text{as } t \to \infty.$$

**Definition 2.4 (Global asymptotic stability).** The equilibrium is *globally asymptotically stable* if it is Lyapunov stable and $\phi_t(x_0) \to 0$ as $t \to \infty$ for every $x_0 \in U$.

> [!NOTE] Stability hierarchy
> Asymptotic stability strictly implies Lyapunov stability. The converse fails: a center (purely imaginary eigenvalues, e.g., $\dot{x}_1 = x_2$, $\dot{x}_2 = -x_1$) is Lyapunov stable but not asymptotically stable.

### 2.2 Lyapunov Functions and the Direct Method 🔑

The key idea is to certify stability without solving the ODE. One finds a scalar function $V$ that acts as a generalized energy: positive away from equilibrium, and non-increasing along trajectories.

**Definition 2.5 (Lyapunov function).** Let $U \subseteq \mathbb{R}^n$ be a neighborhood of $0$. A continuously differentiable function $V: U \to \mathbb{R}$ is a *Lyapunov function* for (2.1) at $x^* = 0$ if:
1. $V(0) = 0$.
2. $V(x) > 0$ for all $x \in U \setminus \{0\}$ (positive definiteness).
3. $\dot{V}(x) := \nabla V(x) \cdot f(x) \leq 0$ for all $x \in U$ (non-increasing along trajectories).

If condition (3) holds with strict inequality ($\dot{V}(x) < 0$ for $x \neq 0$), we call $V$ a *strict* Lyapunov function.

> [!NOTE] The orbital derivative
> The quantity $\dot{V}(x) = \nabla V(x) \cdot f(x)$ is called the *orbital derivative* or *Lie derivative* of $V$ along $f$. It gives the rate of change of $V$ along the solution $\phi_t(x)$ at time $t = 0$:
> $$\dot{V}(x) = \left.\frac{d}{dt}\right|_{t=0} V(\phi_t(x)) = \nabla V(x) \cdot \dot{x}\big|_{x} = \nabla V(x) \cdot f(x).$$
> The chain rule is the only ingredient; no solution is needed.

**Theorem 2.6 (Lyapunov's Direct Method).** Let $V$ be a Lyapunov function for (2.1) on a neighborhood $U$ of $0$.
1. If $\dot{V}(x) \leq 0$ for all $x \in U$, then $x^* = 0$ is Lyapunov stable.
2. If $\dot{V}(x) < 0$ for all $x \in U \setminus \{0\}$, then $x^* = 0$ is asymptotically stable.

*Proof sketch (Part 1 — Lyapunov stability).* Let $\epsilon > 0$ be small enough that $B_\epsilon(0) \subseteq U$. Let $c := \min_{\|x\| = \epsilon} V(x) > 0$ (the minimum over the sphere, which is positive by positive definiteness). The set $\Omega_c := \{x \in B_\epsilon(0) : V(x) < c\}$ is open, contains $0$, and is *positively invariant*: since $\dot{V} \leq 0$, any trajectory starting in $\Omega_c$ has non-increasing $V$, hence can never exit through $\partial B_\epsilon(0)$ (where $V \geq c$). Choose $\delta$ so that $B_\delta(0) \subseteq \Omega_c$. Then $\|x_0\| < \delta$ implies $\phi_t(x_0)$ stays in $\Omega_c \subseteq B_\epsilon(0)$ for all $t \geq 0$. $\square$

*Proof sketch (Part 2 — asymptotic stability).* Lyapunov stability follows from Part 1. For convergence: since $V$ is non-increasing and bounded below by 0, the limit $\ell := \lim_{t \to \infty} V(\phi_t(x_0)) \geq 0$ exists. If $\ell > 0$, then $\phi_t(x_0)$ stays in the compact region $\{V \geq \ell\} \cap \overline{B_\epsilon(0)}$ away from $0$, where $\dot{V} \leq -\alpha < 0$ for some $\alpha > 0$ (by compactness and strict negativity). But then $V(\phi_t(x_0)) \to -\infty$, contradicting $V \geq 0$. Hence $\ell = 0$, and positive definiteness forces $\phi_t(x_0) \to 0$. $\square$

> [!EXAMPLE]- Worked example: harmonic oscillator
> Consider $\dot{x}_1 = x_2$, $\dot{x}_2 = -x_1 - x_2$ (damped oscillator). Let $V(x) = \tfrac{1}{2}(x_1^2 + x_2^2)$. Then
> $$\dot{V} = x_1 \dot{x}_1 + x_2 \dot{x}_2 = x_1 x_2 + x_2(-x_1 - x_2) = -x_2^2 \leq 0.$$
> Since $\dot{V} = 0$ iff $x_2 = 0$, this is *not* a strict Lyapunov function — we cannot conclude asymptotic stability from Theorem 2.6 Part 2 alone. LaSalle's principle (Section 2.3) repairs this.

> [!QUESTION] Exercise 1: Verifying a Lyapunov Function
> *This exercise practices checking all three conditions in Definition 2.5 for a concrete nonlinear system.*
>
> > **Prerequisites:** [[#2.2 Lyapunov Functions and the Direct Method|2.2 Lyapunov Functions and the Direct Method]]
>
> Consider the system $\dot{x}_1 = -x_1^3$, $\dot{x}_2 = -x_2$. Define $V(x) = x_1^2 + x_2^2$.
>
> (a) Verify that $V$ satisfies conditions (1) and (2) of Definition 2.5.
>
> (b) Compute $\dot{V}(x)$ along trajectories and determine whether $V$ is a strict Lyapunov function.
>
> (c) What does Theorem 2.6 then tell you about the stability of $x^* = 0$?

> [!TIP]- Solution to Exercise 1
> **Key insight:** A sum of squares is always positive definite; the orbital derivative picks up a factor of $-2$ from each squared term here.
>
> **Sketch:**
>
> (a) $V(0) = 0$; $V(x) = x_1^2 + x_2^2 > 0$ for $(x_1, x_2) \neq (0,0)$. Conditions (1) and (2) hold.
>
> (b) $\dot{V} = 2x_1 \dot{x}_1 + 2x_2 \dot{x}_2 = 2x_1(-x_1^3) + 2x_2(-x_2) = -2x_1^4 - 2x_2^2$.
> Both terms are non-positive; their sum is zero iff $x_1 = 0$ and $x_2 = 0$. Thus $\dot{V}(x) < 0$ for all $x \neq 0$, so $V$ is a strict Lyapunov function.
>
> (c) By Theorem 2.6 Part 2, $x^* = 0$ is asymptotically stable.

### 2.3 LaSalle's Invariance Principle 💡

Theorem 2.6 Part 2 requires $\dot{V} < 0$ everywhere. This is often too strong — the damped oscillator example has $\dot{V} = 0$ on the entire $x_1$-axis. LaSalle's principle handles this by asking not where $\dot{V}$ vanishes, but what invariant structure lives in that set.

**Definition 2.7 (Positively invariant set).** A set $\Omega$ is *positively invariant* for (2.1) if $x_0 \in \Omega \implies \phi_t(x_0) \in \Omega$ for all $t \geq 0$.

**Definition 2.8 (Omega-limit set).** The *omega-limit set* of a point $x_0$ is
$$\omega(x_0) := \bigl\{ y \in \mathbb{R}^n : \exists\, t_k \to \infty \text{ with } \phi_{t_k}(x_0) \to y \bigr\}.$$

**Theorem 2.9 (LaSalle's Invariance Principle).** Let $\Omega \subseteq U$ be a compact, positively invariant set for (2.1). Let $V: U \to \mathbb{R}$ be $C^1$ with $\dot{V}(x) \leq 0$ for all $x \in \Omega$. Define

$$S := \{x \in \Omega : \dot{V}(x) = 0\}, \qquad M := \text{largest invariant subset of } S.$$

Then every solution starting in $\Omega$ converges to $M$ as $t \to \infty$.

*Proof sketch.* For $x_0 \in \Omega$: since $\dot{V} \leq 0$ on $\Omega$ and $\Omega$ is positively invariant, $V(\phi_t(x_0))$ is non-increasing and bounded below, so $V(\phi_t(x_0)) \to \ell$ for some $\ell$. The omega-limit set $\omega(x_0) \subseteq \Omega$ is non-empty, compact, and invariant (standard ODE theory). On $\omega(x_0)$, $V \equiv \ell$ (since $V$ is continuous and converges to $\ell$), so $\dot{V} \equiv 0$ on $\omega(x_0)$, meaning $\omega(x_0) \subseteq S$. Since $\omega(x_0)$ is invariant and lies in $S$, it lies in $M$. $\square$

> [!NOTE] LaSalle strengthens Lyapunov
> If $M = \{0\}$ — i.e., the only trajectory along which $\dot{V} = 0$ throughout is the equilibrium itself — then LaSalle gives asymptotic stability even when $\dot{V}$ is only non-positive. This resolves the damped oscillator example: $\dot{V} = -x_2^2 = 0$ requires $x_2 = 0$, and $\dot{x}_2 = -x_1 - x_2 = -x_1$ must also be zero, so $x_1 = 0$ too. Thus $M = \{0\}$, and asymptotic stability follows.

> [!QUESTION] Exercise 2: Applying LaSalle
> *This exercise shows how LaSalle's principle extends the reach of Lyapunov analysis to systems where no strict Lyapunov function exists.*
>
> > **Prerequisites:** [[#2.3 LaSalle's Invariance Principle|2.3 LaSalle's Invariance Principle]]
>
> Consider the damped nonlinear pendulum: $\dot{\theta} = \omega$, $\dot{\omega} = -\sin\theta - \omega$. Let $V(\theta, \omega) = 1 - \cos\theta + \tfrac{1}{2}\omega^2$.
>
> (a) Show $V$ is positive definite near $(0, 0)$ (note $V(0,0) = 0$).
>
> (b) Compute $\dot{V}$ and identify the set $S = \{\dot{V} = 0\}$.
>
> (c) Show the largest invariant subset $M \subseteq S$ is $\{(0, 0)\}$, and conclude asymptotic stability of the origin.

> [!TIP]- Solution to Exercise 2
> **Key insight:** The dissipation $-\omega^2$ forces $\omega = 0$ in $S$; then the pendulum equation forces $\theta = 0$ too.
>
> **Sketch:**
>
> (a) Near $\theta = 0$: $1 - \cos\theta \approx \theta^2/2 \geq 0$, with equality iff $\theta = 0$. Combined with $\tfrac{1}{2}\omega^2 \geq 0$, we have $V \geq 0$ with $V = 0$ iff $\theta = \omega = 0$.
>
> (b) $\dot{V} = \sin\theta \cdot \omega + \omega(-\sin\theta - \omega) = -\omega^2 \leq 0$. So $S = \{(\theta, \omega) : \omega = 0\}$.
>
> (c) On any trajectory in $S$: $\omega(t) = 0$ for all $t$, so $\dot{\omega} = 0 = -\sin\theta - 0 = -\sin\theta$. Near the origin this forces $\theta = 0$. Thus $M = \{(0,0)\}$, and LaSalle gives asymptotic stability.

---

## 3. The Replicator Dynamic 🧬

### 3.1 Finite Symmetric Games and the Strategy Simplex

A *finite symmetric game* is defined by a finite strategy set $\{1, 2, \ldots, n\}$ and a payoff matrix $A \in \mathbb{R}^{n \times n}$. The entry $A_{ij}$ is the payoff to a player using strategy $i$ against a player using strategy $j$.

The game is played not between two fixed individuals, but in a large *population*. At any time, a fraction $x_i \geq 0$ of the population uses strategy $i$. The state of the population is a vector

$$x = (x_1, \ldots, x_n) \in \Delta^n,$$

where $\Delta^n$ is the *standard $(n-1)$-simplex*:

$$\Delta^n := \left\{ x \in \mathbb{R}^n : x_i \geq 0 \text{ for all } i, \quad \sum_{i=1}^n x_i = 1 \right\}.$$

The interior is $\Delta^n_\circ := \{x \in \Delta^n : x_i > 0 \text{ for all } i\}$. We think of $x$ as a *mixed strategy* — a probability distribution over the pure strategies.

**Definition 3.1 (Fitness).** The *fitness* (expected payoff) of strategy $i$ against the population $x$ is

$$f_i(x) := (Ax)_i = \sum_{j=1}^n A_{ij} x_j.$$

The *mean fitness* of the population is

$$\bar{f}(x) := x \cdot Ax = \sum_{i=1}^n x_i f_i(x) = \sum_{i,j} x_i A_{ij} x_j.$$

> [!NOTE] Why $x \cdot Ax$?
> Each member of the population encounters a random opponent drawn from $x$. Strategy $i$ players earn $f_i(x) = (Ax)_i$. The population-weighted average is $\bar{f}(x) = \sum_i x_i (Ax)_i = x^T A x$.

### 3.2 The Replicator ODE

The *replicator dynamic* models a population that reproduces at a rate proportional to fitness relative to the mean. If strategy $i$ is above-average, its frequency grows; below-average, it shrinks.

**Definition 3.2 (Replicator dynamic).** The replicator ODE on $\Delta^n$ is

$$\dot{x}_i = x_i \bigl( f_i(x) - \bar{f}(x) \bigr) = x_i \bigl( (Ax)_i - x \cdot Ax \bigr), \qquad i = 1, \ldots, n. \tag{3.1}$$

**Proposition 3.3 (Simplex invariance).** If $x(0) \in \Delta^n$, then $x(t) \in \Delta^n$ for all $t \geq 0$.

*Proof.* First, $x_i(0) > 0 \implies x_i(t) > 0$ for all $t$ (by the ODE: $\dot{x}_i = x_i g_i$ has solution $x_i(t) = x_i(0) \exp(\int_0^t g_i(s)\,ds)$, which cannot cross zero). Second, differentiating $\sum_i x_i$:
$$\frac{d}{dt}\sum_i x_i = \sum_i \dot{x}_i = \sum_i x_i (f_i(x) - \bar{f}(x)) = \bar{f}(x) - \bar{f}(x) \cdot 1 = 0.$$
So $\sum_i x_i$ is constant, equaling 1 at $t=0$. $\square$

> [!NOTE] Effective dimension
> Because $\sum_i x_i = 1$ is preserved, the replicator dynamic on $\Delta^n$ has $n-1$ degrees of freedom. The constraint forces $\sum_i \dot{x}_i = 0$, so the $n$ equations (3.1) are not independent — the system lives on the $(n-1)$-dimensional simplex.

> [!EXAMPLE]- Example: Two-strategy game (coordination/hawk-dove)
> For $n = 2$, write $x_1 = x$, $x_2 = 1 - x$. Let $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$. Then
> $$f_1 = ax + b(1-x), \qquad f_2 = cx + d(1-x), \qquad \bar{f} = x f_1 + (1-x) f_2.$$
> The replicator reduces to a scalar ODE $\dot{x} = x(1-x)(f_1 - f_2)$. Interior fixed points occur where $f_1 = f_2$, i.e., $x^* = (d-b)/((a-c)+(d-b))$ (when this lies in $(0,1)$).

### 3.3 Fixed Points

**Proposition 3.4 (Fixed points of the replicator dynamic).** A state $x^* \in \Delta^n$ is a fixed point of (3.1) if and only if

$$x_i^* > 0 \implies f_i(x^*) = \bar{f}(x^*).$$

That is: all strategies in the *support* of $x^*$ must have equal fitness under $x^*$.

*Proof.* $\dot{x}_i^* = 0$ for all $i$. If $x_i^* > 0$, then $(f_i(x^*) - \bar{f}(x^*)) = 0$. If $x_i^* = 0$, the equation $\dot{x}_i^* = 0 \cdot (\ldots) = 0$ is automatic regardless of $f_i$. $\square$

> [!NOTE] Vertices and Nash equilibria
> Every vertex $e_i$ (pure strategy $i$) is a fixed point: the support is $\{i\}$, and $(Ae_i)_i - e_i \cdot Ae_i = A_{ii} - A_{ii} = 0$. Every Nash equilibrium of the underlying game is a fixed point of the replicator (though not conversely).

### 3.4 Evolutionarily Stable Strategies

The concept of an *evolutionarily stable strategy* (ESS) formalizes resistance to invasion by mutants. It is stronger than Nash equilibrium.

**Definition 3.5 (ESS — two-condition form).** A strategy $\hat{x} \in \Delta^n$ is an *evolutionarily stable strategy* if for all $x \neq \hat{x}$ in a neighborhood of $\hat{x}$:

1. $\hat{x} \cdot A\hat{x} \geq x \cdot A\hat{x}$ (Nash condition), and
2. if $\hat{x} \cdot A\hat{x} = x \cdot A\hat{x}$, then $\hat{x} \cdot Ax > x \cdot Ax$ (stability condition).

The first condition says $\hat{x}$ is a best reply to itself. The second says that among best replies to $\hat{x}$, the strategy $\hat{x}$ does strictly better against any mutant $x$ than $x$ does against itself.

**Proposition 3.6 (Interior ESS — one-condition form).** If $\hat{x} \in \Delta^n_\circ$ is an interior point, then $\hat{x}$ is an ESS if and only if there exists $\epsilon > 0$ such that for all $x \in B_\epsilon(\hat{x}) \setminus \{\hat{x}\}$:

$$\hat{x} \cdot Ax > x \cdot Ax. \tag{3.2}$$

*Proof sketch.* For an interior $\hat{x}$, condition (1) of Definition 3.5 is equivalent to $A\hat{x} = \bar{f}(\hat{x}) \mathbf{1}$ (all strategies earn the same fitness at $\hat{x}$). Substituting, $\hat{x} \cdot Ax = \bar{f}(\hat{x})$ for all $x$, so condition (2) becomes $\bar{f}(\hat{x}) > x \cdot Ax = \bar{f}(x)$ for nearby $x \neq \hat{x}$. This is precisely (3.2). $\square$

> [!WARNING] ESS implies Nash, not conversely
> Every ESS is a Nash equilibrium, but not every Nash equilibrium is an ESS. The ESS condition is strictly stronger. A symmetric Nash equilibrium that is not an ESS may be a fixed point of the replicator but need not be asymptotically stable.

> [!QUESTION] Exercise 3: Fixed Points of Rock-Paper-Scissors
> *This exercise identifies all fixed points of the replicator dynamic for a classical non-transitive game and illustrates the distinction between ESS and mere fixed point.*
>
> > **Prerequisites:** [[#3.3 Fixed Points|3.3 Fixed Points]], [[#3.4 Evolutionarily Stable Strategies|3.4 Evolutionarily Stable Strategies]]
>
> Rock-Paper-Scissors has payoff matrix (payoffs to row player):
>
> $$A = \begin{pmatrix} 0 & -1 & 1 \\ 1 & 0 & -1 \\ -1 & 1 & 0 \end{pmatrix}.$$
>
> (a) Verify that the three pure strategies $e_1, e_2, e_3$ are fixed points. Are they ESS?
>
> (b) Show that the mixed strategy $\hat{x} = (1/3, 1/3, 1/3)$ is a fixed point. Is it a Nash equilibrium?
>
> (c) Is $\hat{x}$ an ESS? (Hint: check whether $\hat{x} \cdot Ax > x \cdot Ax$ holds near $\hat{x}$. For zero-sum symmetric games, $x \cdot Ax = 0$ for all $x$ — verify this first.)

> [!TIP]- Solution to Exercise 3
> **Key insight:** For zero-sum symmetric games ($A + A^T = 0$), $x \cdot Ax = 0$ for all $x$, so the ESS condition $\hat{x} \cdot Ax > x \cdot Ax$ becomes $0 > 0$, which is impossible. Thus no ESS exists in RPS.
>
> **Sketch:**
>
> (a) Pure strategy vertices are always fixed points (Proposition 3.4). They are not ESS: e.g., $e_1$ (Rock) is defeated by $e_3$ (Scissors), so no invasion resistance.
>
> (b) At $\hat{x} = (1/3, 1/3, 1/3)$: $(A\hat{x})_i = (0 - 1 + 1)/3 = 0$ for all $i$ (by symmetry of $A$), so $f_i(\hat{x}) = 0 = \bar{f}(\hat{x})$. Thus $\hat{x}$ is a fixed point. It is also a Nash equilibrium since all strategies tie.
>
> (c) Not an ESS. For any $x$: $x \cdot Ax = \sum_{ij} x_i A_{ij} x_j$. Since $A_{ij} = -A_{ji}$, each pair $(i,j)$ contributes $x_i A_{ij} x_j + x_j A_{ji} x_i = 0$. Thus $x \cdot Ax = 0$ for all $x$. But $\hat{x} \cdot Ax = 0$ too (by the same argument), so $\hat{x} \cdot Ax = x \cdot Ax$ for all $x$ — the strict inequality in (3.2) never holds.

---

## 4. KL Divergence as a Lyapunov Function 🔑

### 4.1 Setup and Statement

We now prove the main result. Fix an interior ESS $\hat{x} \in \Delta^n_\circ$ of a finite symmetric game with payoff matrix $A$. Define the candidate Lyapunov function

$$V(x) = D_{\mathrm{KL}}(\hat{x} \| x) := \sum_{i=1}^n \hat{x}_i \log \frac{\hat{x}_i}{x_i}. \tag{4.1}$$

$V$ is defined on $\Delta^n_\circ$ (since $\hat{x}_i > 0$ and we need $x_i > 0$ for the logarithm).

**Theorem 4.1 (KL Lyapunov at an ESS).** Let $\hat{x} \in \Delta^n_\circ$ be an interior ESS. Then $V(x) = D_{\mathrm{KL}}(\hat{x} \| x)$ is a strict Lyapunov function for the replicator dynamic (3.1) in a neighborhood of $\hat{x}$. In particular, $\hat{x}$ is asymptotically stable.

The proof occupies Sections 4.2–4.5, building each piece in turn.

### 4.2 Positivity of V

**Lemma 4.2 (Positivity of KL divergence).** $D_{\mathrm{KL}}(\hat{x} \| x) \geq 0$ for all $x, \hat{x} \in \Delta^n_\circ$, with equality if and only if $x = \hat{x}$.

*Proof.* By Jensen's inequality and the concavity of $\log$:

$$-D_{\mathrm{KL}}(\hat{x} \| x) = \sum_i \hat{x}_i \log \frac{x_i}{\hat{x}_i} \leq \log \sum_i \hat{x}_i \cdot \frac{x_i}{\hat{x}_i} = \log \sum_i x_i = \log 1 = 0.$$

Equality in Jensen holds iff $x_i / \hat{x}_i$ is constant for all $i$ in the support of $\hat{x}$. Since $\sum_i x_i = \sum_i \hat{x}_i = 1$, the constant must be 1, so $x = \hat{x}$. $\square$

This establishes conditions (1) and (2) of Definition 2.5: $V(\hat{x}) = D_{\mathrm{KL}}(\hat{x} \| \hat{x}) = 0$ and $V(x) > 0$ for $x \neq \hat{x}$.

> [!NOTE] V is defined relative to hat-x
> We treat $V$ as a function of $x$ with $\hat{x}$ fixed as a parameter. The point $\hat{x}$ is translated to play the role of the equilibrium $x^* = 0$ in Definition 2.5 — we are certifying stability of $\hat{x}$, not of the origin.

### 4.3 Computing V-dot along Replicator Trajectories 📐

This is the heart of the proof. We differentiate $V(x(t)) = \sum_i \hat{x}_i \log(\hat{x}_i / x_i(t))$ with respect to $t$, using the replicator ODE (3.1).

**Computation.** Differentiating $V(x) = \sum_i \hat{x}_i \log \hat{x}_i - \sum_i \hat{x}_i \log x_i$ with respect to $t$ (the first sum is constant since $\hat{x}$ is fixed):

$$\dot{V}(x) = -\sum_{i=1}^n \hat{x}_i \cdot \frac{\dot{x}_i}{x_i}. \tag{4.2}$$

Substitute the replicator ODE $\dot{x}_i = x_i(f_i(x) - \bar{f}(x))$:

$$\dot{V}(x) = -\sum_{i=1}^n \hat{x}_i \cdot \frac{x_i(f_i(x) - \bar{f}(x))}{x_i} = -\sum_{i=1}^n \hat{x}_i \bigl( f_i(x) - \bar{f}(x) \bigr). \tag{4.3}$$

The $x_i$ in numerator and denominator cancel (this is where positivity $x_i > 0$ is used). Distributing:

$$\dot{V}(x) = -\sum_{i=1}^n \hat{x}_i f_i(x) + \bar{f}(x) \sum_{i=1}^n \hat{x}_i. \tag{4.4}$$

Since $\sum_i \hat{x}_i = 1$ and $\sum_i \hat{x}_i f_i(x) = \hat{x} \cdot Ax$ (by definition of fitness), this simplifies to:

$$\boxed{\dot{V}(x) = \bar{f}(x) - \hat{x} \cdot Ax.} \tag{4.5}$$

> [!NOTE] The formula in words
> $\dot{V}$ equals the *mean fitness of the current population* minus the *expected fitness of the ESS $\hat{x}$ against the current population*. This is a quantity entirely determined by the game matrix $A$ and the current state $x$.

### 4.4 Signing V-dot via the ESS Inequality 🔑

We now use the ESS condition to show $\dot{V}(x) < 0$ for $x \neq \hat{x}$ near $\hat{x}$.

**Lemma 4.3.** Let $\hat{x} \in \Delta^n_\circ$ be an interior ESS. Then for all $x \in B_\epsilon(\hat{x}) \setminus \{\hat{x}\}$:

$$\dot{V}(x) = \bar{f}(x) - \hat{x} \cdot Ax < 0.$$

*Proof.* By the interior ESS condition (Proposition 3.6, equation (3.2)):

$$\hat{x} \cdot Ax > x \cdot Ax = \bar{f}(x) \qquad \text{for all } x \in B_\epsilon(\hat{x}) \setminus \{\hat{x}\}.$$

Rearranging: $\bar{f}(x) - \hat{x} \cdot Ax < 0$. By (4.5), this is exactly $\dot{V}(x) < 0$. $\square$

**This is the key computation. The ESS inequality says exactly that the KL divergence from $\hat{x}$ to the current state is being reduced along every nearby trajectory.**

> [!WARNING] Locality
> *This argument is local: the ESS inequality (3.2) holds only in a neighborhood $B_\epsilon(\hat{x})$, so $\dot{V} < 0$ is guaranteed only near $\hat{x}$. If $\hat{x}$ were a global ESS (the strict inequality holds on all of $\Delta^n_\circ$), the conclusion would extend globally.*

### 4.5 Asymptotic Stability Conclusion

Collecting the pieces:

1. **$V(\hat{x}) = 0$:** immediate from $D_{\mathrm{KL}}(\hat{x} \| \hat{x}) = 0$.
2. **$V(x) > 0$ for $x \neq \hat{x}$:** Lemma 4.2 (non-negativity and equality condition of KL).
3. **$\dot{V}(x) < 0$ for $x \neq \hat{x}$ near $\hat{x}$:** Lemma 4.3 via the ESS inequality.

By Theorem 2.6 Part 2, the equilibrium $\hat{x}$ is asymptotically stable. This completes the proof of Theorem 4.1. $\square$

> [!NOTE] The converse also holds
> Harper (2009) proves the converse: $D_{\mathrm{KL}}(\hat{x} \| x)$ is a Lyapunov function for the replicator dynamic *if and only if* $\hat{x}$ is an interior ESS. The "only if" direction shows that the ESS condition is not merely sufficient but also necessary for this particular Lyapunov function to work. This is Theorem 5 in [Harper (2009)](https://arxiv.org/abs/0911.1383).

> [!QUESTION] Exercise 4: Two-Strategy Prisoner's Dilemma
> *This exercise works through the KL Lyapunov computation explicitly for a two-strategy game and checks every step of the derivation.*
>
> > **Prerequisites:** [[#4.3 Computing V-dot along Replicator Trajectories|4.3 Computing V-dot along Replicator Trajectories]], [[#4.4 Signing V-dot via the ESS Inequality|4.4 Signing V-dot via the ESS Inequality]]
>
> Consider a symmetric $2 \times 2$ game with
>
> $$A = \begin{pmatrix} 2 & 0 \\ 3 & 1 \end{pmatrix}.$$
>
> Write $x = (x, 1-x) \in \Delta^2$ (abusing notation to use $x$ for $x_1$).
>
> (a) Compute $f_1(x)$, $f_2(x)$, and $\bar{f}(x)$.
>
> (b) Find all fixed points of the replicator in $[0,1]$. Show that $x^* = 0$ (pure strategy 2) is the only stable one. Is it an ESS?
>
> (c) For $\hat{x} = (0, 1) = e_2$, the KL formula (4.1) formally requires $\hat{x}_i > 0$. Instead, take $\hat{x}_1 = 0$ and note $V(x) = \log(1/(1-x))$ (the $\hat{x}_1 = 0$ term vanishes). Compute $\dot{V}$ along the replicator and verify it is negative for $x \in (0,1)$.

> [!TIP]- Solution to Exercise 4
> **Key insight:** A pure ESS at a vertex uses a degenerate version of the KL formula (the boundary term vanishes), but $\dot{V} < 0$ still follows from direct computation.
>
> **Sketch:**
>
> (a) $f_1 = 2x$, $f_2 = 3x + (1-x) = 2x + 1$. Mean fitness: $\bar{f} = x \cdot 2x + (1-x)(2x+1) = 2x^2 + (1-x)(2x+1) = 2x+1-x^2$ (after simplification, or just $x f_1 + (1-x) f_2$).
>
> (b) Fixed points: $\dot{x} = x(f_1 - f_2) = x(2x - (2x+1)) = x(-1) = -x$. So $\dot{x} = 0$ only at $x = 0$. The unique fixed point in $[0,1]$ is $x^* = 0$ (pure strategy 2 dominates). Pure strategy 2 is an ESS since it is strictly dominant.
>
> (c) $V(x) = -\log(1-x)$ (for $x_1 = x$ small). Then $\dot{V} = \frac{\dot{x}}{1-x} = \frac{-x}{1-x} < 0$ for $x \in (0,1)$. This confirms $\dot{V} < 0$ away from $\hat{x}$.

---

## 5. Geometric Interpretation 🌐

> [!INFO] The Shahshahani Metric and Gradient Flow
>
> The match between the replicator dynamic and the KL Lyapunov function is not a coincidence — it reflects deep geometric structure on the simplex.
>
> **The Shahshahani metric.** Equip the interior $\Delta^n_\circ$ with the *Shahshahani metric*, a Riemannian metric given by
> $$g_{ij}(x) = \frac{\delta_{ij}}{x_i}.$$
> This is a diagonal metric whose coefficient in the $i$-th direction scales as $1/x_i$: directions toward the boundary (where $x_i \to 0$) become infinitely "expensive."
>
> **Equivalence with Fisher information.** The manifold $\Delta^n_\circ$ can be identified with the space of categorical distributions $\mathrm{Cat}(n)$ on $n$ outcomes. Under this identification, the Shahshahani metric equals the *Fisher information metric*:
> $$g_{ij}^{\mathrm{Fisher}}(x) = \mathbb{E}_x\!\left[\frac{\partial \log p(k;x)}{\partial x_i}\frac{\partial \log p(k;x)}{\partial x_j}\right] = \frac{\delta_{ij}}{x_i}.$$
> This is Theorem 1 of [Harper (2009)](https://arxiv.org/abs/0911.1383).
>
> **Replicator as gradient flow.** The replicator dynamic (3.1) is the *gradient flow* of the mean fitness $\bar{f}(x) = x \cdot Ax$ with respect to the Shahshahani metric. That is,
> $$\dot{x}_i = x_i(f_i(x) - \bar{f}(x)) = \bigl(\mathrm{grad}_g \bar{f}(x)\bigr)_i,$$
> where $\mathrm{grad}_g$ denotes the Riemannian gradient. Since the ESS is a maximum of $\bar{f}$ restricted to $\Delta^n_\circ$ (local maximum of mean fitness), gradient flow on $\bar{f}$ is attracted to it — and the KL divergence is the "distance" function in this geometry, since the Fisher information metric is the Hessian of the KL divergence at any point.

> [!INFO] Mirror Descent and Multiplicative Weights
>
> The connection to optimization is equally tight. *Mirror descent* with KL divergence as the Bregman potential on $\Delta^n$ produces the update:
>
> ```python
> # Continuous-time mirror descent (multiplicative weights)
> # x: current population state (probability vector)
> # f: fitness vector f_i = (Ax)_i
> # eta: step size / learning rate
>
> def mirror_descent_step(x, f, eta):
>     log_x_new = [log(x[i]) + eta * f[i] for i in range(n)]
>     # project back: normalize in log-space (log-sum-exp)
>     log_Z = log_sum_exp(log_x_new)
>     return [exp(log_x_new[i] - log_Z) for i in range(n)]
> ```
>
> Taking $\eta \to 0$ (continuous time limit) recovers precisely the replicator ODE. The *multiplicative weights update algorithm* (MWUA) in online learning is the discrete-time version. The connection is made explicit in [Raskutti & Mukherjee (2015)](https://arxiv.org/abs/1310.7780), which shows that mirror descent with a Bregman divergence is equivalent to natural gradient descent on the dual Riemannian manifold. When the Bregman potential is the negative entropy $\phi(x) = \sum_i x_i \log x_i$, the induced metric is the Fisher information metric and the dynamics are replicator-like.

---

## 6. References

| Reference Name | Brief Summary | Link to Reference |
|---|---|---|
| [Khalil (2002), *Nonlinear Systems*](https://www.egr.msu.edu/~khalil/NonlinearSystems/index.html) | Standard graduate reference for Lyapunov's direct method and LaSalle's invariance principle | [Link](https://www.egr.msu.edu/~khalil/NonlinearSystems/index.html) |
| [Perko (2001), *Differential Equations and Dynamical Systems*](https://link.springer.com/book/10.1007/978-1-4613-0003-8) | Graduate ODE and stability theory; equilibria, flows, phase portraits | [Link](https://link.springer.com/book/10.1007/978-1-4613-0003-8) |
| [Hofbauer & Sigmund (1998), *Evolutionary Games and Population Dynamics*](https://www.cambridge.org/core/books/evolutionary-games-and-population-dynamics/A8D94EBE6A16837E7CB3CED24E1948F8) | Definitive textbook treatment of replicator dynamics and KL Lyapunov at ESS (Chapter 7) | [Link](https://www.cambridge.org/core/books/evolutionary-games-and-population-dynamics/A8D94EBE6A16837E7CB3CED24E1948F8) |
| [Hofbauer & Sigmund (2003), *Evolutionary Game Dynamics*, AMS Bulletin](https://www.ams.org/journals/bull/2003-40-04/S0273-0979-03-00988-1/S0273-0979-03-00988-1.pdf) | Concise self-contained survey of replicator dynamics, fixed points, ESS, and Lyapunov methods | [Link](https://www.ams.org/journals/bull/2003-40-04/S0273-0979-03-00988-1/S0273-0979-03-00988-1.pdf) |
| [Weibull (1995), *Evolutionary Game Theory*](https://mitpress.mit.edu/9780262731218/evolutionary-game-theory/) | ESS definition, stability hierarchy, relationship to Nash equilibria | [Link](https://mitpress.mit.edu/9780262731218/evolutionary-game-theory/) |
| [Sandholm (2010), *Population Games and Evolutionary Dynamics*](https://mitpress.mit.edu/9780262195874/population-games-and-evolutionary-dynamics/) | Generalized evolutionary dynamics, potential functions, and global stability results | [Link](https://mitpress.mit.edu/9780262195874/population-games-and-evolutionary-dynamics/) |
| [Harper (2009), arXiv:0911.1383](https://arxiv.org/abs/0911.1383) | Proves KL divergence is a Lyapunov function iff ESS; establishes Shahshahani = Fisher information metric | [Link](https://arxiv.org/abs/0911.1383) |
| [Fryer (2012), arXiv:1207.0036](https://arxiv.org/abs/1207.0036) | Extends KL Lyapunov result to incentive dynamics, giving sufficient conditions for asymptotic stability beyond replicator | [Link](https://arxiv.org/abs/1207.0036) |
| [Raskutti & Mukherjee (2015), arXiv:1310.7780](https://arxiv.org/abs/1310.7780) | Mirror descent with Bregman divergences equals natural gradient descent; connects multiplicative weights to replicator dynamics | [Link](https://arxiv.org/abs/1310.7780) |
