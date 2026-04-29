# 🧠 Infinite-Width Limits: The Lazy/Rich Dichotomy

## Table of Contents

- [[#1. Setup|1. Setup]]
- [[#2. The Neural Tangent Kernel|2. The Neural Tangent Kernel]]
  - [[#2.1 Definition and Gradient Flow|2.1 Definition and Gradient Flow]]
  - [[#2.2 Infinite-Width Limit of the NTK|2.2 Infinite-Width Limit of the NTK]]
- [[#3. The Lazy Regime|3. The Lazy Regime]]
  - [[#3.1 Why Features Freeze|3.1 Why Features Freeze]]
  - [[#3.2 Frozen NTK and Linear Dynamics|3.2 Frozen NTK and Linear Dynamics]]
  - [[#3.3 Equivalence to Kernel Regression|3.3 Equivalence to Kernel Regression]]
- [[#4. The Rich Regime|4. The Rich Regime]]
  - [[#4.1 Mean-Field Scaling and Breakdown of Linearization|4.1 Mean-Field Scaling and Breakdown of Linearization]]
  - [[#4.2 The McKean-Vlasov Limit|4.2 The McKean-Vlasov Limit]]
  - [[#4.3 Feature Learning as Evolution of the Weight Distribution|4.3 Feature Learning as Evolution of the Weight Distribution]]
- [[#5. The Output Multiplier as Control Parameter|5. The Output Multiplier as Control Parameter]]
- [[#6. The Maximal Update Parameterization|6. The Maximal Update Parameterization]]
  - [[#6.1 Deep Networks and the Tensor Programs Framework|6.1 Deep Networks and the Tensor Programs Framework]]
  - [[#6.2 The Two Non-Trivial Limits|6.2 The Two Non-Trivial Limits]]
  - [[#6.3 Hyperparameter Transfer|6.3 Hyperparameter Transfer]]
- [[#7. Summary Comparison|7. Summary Comparison]]
- [[#References|References]]

---

## 1. Setup 🎯

Consider a *shallow neural network* of width $N$:
$$f(x;\theta) = \frac{\alpha}{\sqrt{N}} \sum_{i=1}^N a_i \,\sigma(w_i^\top x), \qquad \theta = \{a_i, w_i\}_{i=1}^N$$

where $x \in \mathbb{R}^d$, $a_i \in \mathbb{R}$, $w_i \in \mathbb{R}^d$, $\sigma$ is a pointwise nonlinearity (e.g. ReLU), and $\alpha > 0$ is an *output multiplier* — the key parameter controlling the lazy/rich dichotomy.

Initialize:
$$a_i \overset{\text{iid}}{\sim} \mathcal{N}(0,1), \qquad w_i \overset{\text{iid}}{\sim} \mathcal{N}(0, I_d).$$

Train by gradient flow on a dataset $\{(x_\mu, y_\mu)\}_{\mu=1}^n$:
$$\dot{\theta}(t) = -\nabla_\theta L(\theta(t)), \qquad L(\theta) = \frac{1}{n}\sum_{\mu=1}^n \ell\!\left(f(x_\mu;\theta),\, y_\mu\right).$$

**Goal.** Understand the $N \to \infty$ limit of these dynamics, and how $\alpha$ governs qualitatively distinct dynamical regimes.

> [!INFO] Why the $1/\sqrt{N}$ prefactor?
> By the central limit theorem, $\frac{1}{\sqrt{N}}\sum_{i=1}^N a_i \sigma(w_i^\top x) = O(1)$ at initialization (since each term has variance $\sim 1/N$ after the prefactor). This is the *LeCun initialization* rule: preserve $O(1)$ signal variance through the network. The $\alpha$ factor then independently controls the output scale.

---

## 2. The Neural Tangent Kernel 🔑

### 2.1 Definition and Gradient Flow

**Definition (Neural Tangent Kernel).** For a network $f(x;\theta)$, define:
$$K_\theta(x,x') \;:=\; \bigl\langle \nabla_\theta f(x;\theta),\, \nabla_\theta f(x';\theta) \bigr\rangle \;=\; \sum_k \frac{\partial f(x;\theta)}{\partial \theta_k} \frac{\partial f(x';\theta)}{\partial \theta_k}.$$

Under gradient flow, the output at any input $x$ evolves as:
$$\dot{f}(x;\theta(t)) = \nabla_\theta f(x;\theta(t))^\top \dot{\theta}(t) = -\sum_\mu K_{\theta(t)}(x,x_\mu)\,\frac{\partial L}{\partial f(x_\mu)}.$$

This is a coupled ODE in $f$-space, driven by the Gram matrix $K_{\theta(t)}$ evaluated on the training set. The fundamental question is: *does $K_{\theta(t)}$ change during training, or is it frozen?*

### 2.2 Infinite-Width Limit of the NTK

For our shallow network, compute the partial derivatives:
$$\nabla_{a_i} f(x;\theta) = \frac{\alpha}{\sqrt{N}} \sigma(w_i^\top x), \qquad \nabla_{w_i} f(x;\theta) = \frac{\alpha}{\sqrt{N}} a_i\, \sigma'(w_i^\top x)\, x.$$

The NTK is therefore:
$$K_\theta(x,x') = \frac{\alpha^2}{N} \sum_{i=1}^N \Bigl[ a_i^2\, \sigma'(w_i^\top x)\,\sigma'(w_i^\top x')\,\langle x,x'\rangle + \sigma(w_i^\top x)\,\sigma(w_i^\top x') \Bigr].$$

**Proposition (NTK convergence).** As $N \to \infty$, by the law of large numbers:
$$K_\theta(x,x') \xrightarrow{\;\mathrm{a.s.}\;} \alpha^2\, K^\infty(x,x'),$$

where $K^\infty$ is the *deterministic limiting kernel*:
$$K^\infty(x,x') = \underbrace{\mathbb{E}_{w}\!\left[\sigma'(w^\top x)\,\sigma'(w^\top x')\right]\langle x,x'\rangle}_{\text{hidden-layer contribution}} + \underbrace{\mathbb{E}_{w}\!\left[\sigma(w^\top x)\,\sigma(w^\top x')\right]}_{\text{output-layer contribution}}$$

with $w \sim \mathcal{N}(0,I_d)$ and $a \sim \mathcal{N}(0,1)$ (note $\mathbb{E}[a^2]=1$).

> [!NOTE] Recursive NTK for Deep Networks
> For an $L$-layer deep network, $K^\infty$ satisfies a recursion over layers. At each layer $\ell$, one computes a *kernel* and a *derivative kernel*, building the full NTK as a sum over layers. This is the Jacot et al. (2018) original computation.

> [!QUESTION] Exercise 1: NTK for a Linear Network
> *This problem establishes the NTK for the simplest case and connects it to the Gram kernel.*
>
> > **Prerequisites:** [[#2.2 Infinite-Width Limit of the NTK|NTK limit]]
>
> For a **linear** network $f(x;\theta) = \frac{\alpha}{\sqrt{N}} \sum_{i=1}^N a_i (w_i^\top x)$ (i.e. $\sigma = \mathrm{id}$), compute $K^\infty(x,x')$ in closed form. Show that it is proportional to $\langle x, x'\rangle$ and identify the proportionality constant.

> [!TIP]- Solution to Exercise 1
> **Key insight:** With $\sigma = \mathrm{id}$, $\sigma' = 1$ everywhere, so the NTK computation reduces to simple expectations over Gaussian weights.
>
> **Sketch:**
> $$K^\infty(x,x') = \mathbb{E}_w[1 \cdot 1]\langle x,x'\rangle + \mathbb{E}_w[(w^\top x)(w^\top x')]$$
> The first term is $\langle x,x'\rangle$. The second: $\mathbb{E}_w[(w^\top x)(w^\top x')] = x^\top \mathbb{E}[ww^\top] x' = x^\top I_d\, x' = \langle x,x'\rangle$ (since $w\sim\mathcal{N}(0,I_d)$).
>
> Therefore $K^\infty(x,x') = 2\langle x,x'\rangle$. Up to the factor of 2, the NTK for a linear network is just the **dot-product (Gram) kernel**, recovering the well-known connection between linearized networks and kernel methods.

---

## 3. The Lazy Regime 😴

### 3.1 Why Features Freeze

The *lazy regime* corresponds to $\alpha = O(1)$ — i.e., standard LeCun initialization with no extra output scaling.

Compute the gradient flow equation for the feature parameters $w_i$:
$$\dot{w}_i(t) = -\nabla_{w_i} L = -\frac{\alpha}{\sqrt{N}} a_i(t) \sum_\mu \frac{\partial \ell}{\partial f(x_\mu)} \sigma'(w_i(t)^\top x_\mu)\, x_\mu.$$

Since $a_i = O(1)$, $\frac{\partial \ell}{\partial f} = O(1)$, and $\sigma' = O(1)$, we have $\|\dot{w}_i\|_2 = O(\alpha/\sqrt{N})$.

After $T$ time units:
$$\|w_i(T) - w_i(0)\|_2 \;\leq\; \int_0^T \|\dot{w}_i(t)\|_2\,dt \;=\; O\!\left(\frac{\alpha T}{\sqrt{N}}\right) \;\xrightarrow{N\to\infty}\; 0.$$

The same calculation holds for $a_i$. In particular, the change in hidden activation at any input $x$:
$$|\Delta h_i(x)| \;:=\; |\sigma(w_i(T)^\top x) - \sigma(w_i(0)^\top x)| \;\leq\; \|\sigma'\|_\infty\, \|\Delta w_i\|\,\|x\| \;=\; O\!\left(\frac{\alpha T}{\sqrt{N}}\right).$$

**In the $N \to \infty$ limit, hidden representations are frozen at their initialization values.** This is the defining property of the *lazy* regime.

> [!WARNING] Output Still Changes!
> Although features are frozen, the *output* $f(x)$ still changes during training. The change is:
> $$\Delta f(x) = \frac{\alpha}{\sqrt{N}}\sum_i \bigl[\Delta a_i\,\sigma(h_i^0(x)) + a_i\,\sigma'(h_i^0(x))\,\Delta h_i(x)\bigr],$$
> where $h_i^0(x) = w_i(0)^\top x$. Each term is $O(1/N)$, but there are $N$ terms, so $\Delta f(x) = O(1)$ — the output changes by a finite amount through the accumulation of many tiny weight updates. This is the key mechanism of the lazy regime.

### 3.2 Frozen NTK and Linear Dynamics

Since $w_i(t) \approx w_i(0)$ and $a_i(t) \approx a_i(0)$ for all $t \leq T$, the NTK barely changes:
$$K_{\theta(t)}(x,x') = \alpha^2 K^\infty(x,x') + O\!\left(\frac{1}{\sqrt{N}}\right).$$

Substituting into the gradient flow equation for $f$:
$$\dot{f}(x;\theta(t)) \approx -\alpha^2 \sum_\mu K^\infty(x,x_\mu)\,\frac{\partial L}{\partial f(x_\mu)}.$$

For squared loss $\ell(f,y) = \frac{1}{2}(f-y)^2$, define the residual vector $r(t) \in \mathbb{R}^n$ with $(r)_\mu = f(x_\mu;t) - y_\mu$. Then:
$$\dot{r}(t) = -\alpha^2 \mathbf{K}\, r(t), \qquad \mathbf{K}_{\mu\nu} = K^\infty(x_\mu, x_\nu).$$

**This is a linear ODE with solution:**
$$r(t) = e^{-\alpha^2 \mathbf{K} t}\, r(0).$$

The loss decays exponentially at rate determined by the smallest eigenvalue $\lambda_{\min}(\mathbf{K})$:
$$L(\theta(t)) = \frac{1}{2}\|r(t)\|^2 \sim e^{-2\alpha^2 \lambda_{\min}(\mathbf{K})\, t} \cdot L(\theta(0)).$$

> [!INFO] Spectral Decomposition and Mode Learning
> Expanding $r(0)$ in the eigenbasis of $\mathbf{K}$ as $r(0) = \sum_k c_k v_k$ (with $\mathbf{K}v_k = \lambda_k v_k$), the solution is $r(t) = \sum_k c_k e^{-\alpha^2 \lambda_k t} v_k$. Eigencomponents with large $\lambda_k$ are learned first — this is the origin of the *spectral bias* / frequency principle observed in neural networks.

### 3.3 Equivalence to Kernel Regression

As $t \to \infty$ under the lazy dynamics, assuming $\mathbf{K} \succ 0$:
$$f(x;\theta(\infty)) = K^\infty(x, \mathbf{x})\,[\mathbf{K}]^{-1} y,$$

where $K^\infty(x,\mathbf{x}) \in \mathbb{R}^n$ has entries $K^\infty(x,x_\mu)$. This is exactly **kernel regression** with kernel $K^\infty$.

The NTK regime thus reduces the problem of training a neural network to a kernel method with a fixed architecture-dependent kernel. This is exact in the $N \to \infty$ limit and explains why the lazy regime *cannot* exhibit feature learning: the network's effective kernel $K^\infty$ is determined entirely by the initialization and architecture, and never adapts to the data.

> [!QUESTION] Exercise 2: Eigenspectrum Controls Generalization
> *This problem shows that the NTK eigenspectrum directly controls test-time generalization error.*
>
> > **Prerequisites:** [[#3.3 Equivalence to Kernel Regression|Kernel regression equivalence]], [[#3.2 Frozen NTK and Linear Dynamics|Linear dynamics]]
>
> Suppose the target function $f^*(x)$ has a kernel ridge regression expansion $f^* = \sum_k \hat{f}^*_k v_k$ in the NTK eigenbasis $\{v_k, \lambda_k\}$ (with $\lambda_1 \geq \lambda_2 \geq \ldots$). In the infinite-data limit ($n \to \infty$), show that the test MSE of the lazy network at convergence is exactly:
> $$\text{MSE}_\text{test} = \sum_k \hat{f}^{*2}_k \cdot \mathbf{1}[\lambda_k = 0].$$
> What does this imply about which target functions the lazy regime *cannot* learn?

> [!TIP]- Solution to Exercise 2
> **Key insight:** Kernel regression with kernel $K$ projects $f^*$ onto the RKHS of $K$; components in the null space of $K$ are unlearnable.
>
> **Sketch:** The kernel regression predictor in the infinite-data limit is the minimum-norm interpolant in the RKHS $\mathcal{H}_K$, i.e. $\hat{f} = P_{\mathcal{H}_K} f^*$. The component $\hat{f}^*_k v_k$ lies in $\mathcal{H}_K$ iff $\lambda_k > 0$. If $\lambda_k = 0$, the kernel cannot represent this component, and it contributes $\hat{f}^{*2}_k$ to the test MSE.
>
> **Implication:** The lazy regime cannot learn any target component that is in the null space of $K^\infty$. For ReLU networks, $K^\infty$ is typically a Laplace or arc-cosine kernel, which has non-trivial structure. In particular, high-frequency components are in the approximate null space (small eigenvalues), explaining why kernel methods are sample-inefficient on high-dimensional structured data compared to feature-learning networks.

---

## 4. The Rich Regime 🌊

### 4.1 Mean-Field Scaling and Breakdown of Linearization

The *rich regime* arises when we downscale the output by an extra factor of $1/\sqrt{N}$, changing the parameterization to:
$$f(x;\theta) = \frac{1}{N}\sum_{i=1}^N \Phi(\theta_i, x), \qquad \Phi(\theta_i, x) := a_i\,\sigma(w_i^\top x).$$

Compare with the lazy case: the prefactor is $1/N$ instead of $1/\sqrt{N}$, so $f(x;\theta_0) = O(1/\sqrt{N}) \to 0$ at initialization — the output starts near zero and must *grow* during training.

The NTK for this parameterization:
$$K^{MF}_\theta(x,x') = \frac{1}{N^2}\sum_{i=1}^N \bigl\langle \nabla_{\theta_i}\Phi(\theta_i,x),\, \nabla_{\theta_i}\Phi(\theta_i,x') \bigr\rangle = O\!\left(\frac{1}{N}\right) \xrightarrow{N\to\infty} 0.$$

*Critically, the NTK vanishes in the $N \to \infty$ limit.* This means the **linearization breaks down** — the network cannot be approximated by its first-order Taylor expansion. One must deal with the full nonlinear dynamics, which is why feature learning can occur.

> [!EXAMPLE] Analogy: Elastic vs. Plastic Deformation
> The lazy/rich dichotomy is analogous to elastic vs. plastic deformation in materials. A small applied force deforms a material linearly (elastic), and its internal atomic structure is unchanged. A large force causes plastic deformation — the internal structure reorganizes permanently. The output multiplier $\alpha$ plays the role of the applied force.

### 4.2 The McKean-Vlasov Limit

The correct $N \to \infty$ limit for the mean-field network is a *distributional* description. Define the *empirical measure* over the $N$ particles:
$$\mu^N_t \;:=\; \frac{1}{N}\sum_{i=1}^N \delta_{\theta_i(t)} \;\in\; \mathcal{P}(\mathbb{R}^{d+1}).$$

The mean-field output is:
$$f(x;\mu^N_t) = \int \Phi(\theta, x)\,d\mu^N_t(\theta).$$

Each particle evolves under gradient flow:
$$\dot{\theta}_i(t) = -\nabla_{\theta_i} L = -\mathbb{E}_{x,y}\!\left[\frac{\partial \ell}{\partial f}\bigl(f(x;\mu^N_t), y\bigr)\,\nabla_{\theta_i}\Phi(\theta_i,x)\right].$$

**Theorem (Mean-Field Limit — Mei et al. 2018, Chizat & Bach 2018).** As $N \to \infty$, $\mu^N_t \rightharpoonup \mu_t$ weakly, where $\mu_t$ satisfies the *McKean-Vlasov PDE*:
$$\partial_t \mu_t + \nabla_\theta\cdot\!\left(\mu_t\,\mathbf{v}_t\right) = 0,$$

$$\mathbf{v}_t(\theta) \;:=\; -\,\mathbb{E}_{(x,y)\sim\mathcal{D}}\!\left[\frac{\partial \ell}{\partial f}\bigl(f(x;\mu_t),\,y\bigr)\,\nabla_\theta\Phi(\theta,x)\right],$$

$$f(x;\mu_t) \;:=\; \int \Phi(\theta,x)\,d\mu_t(\theta).$$

This is a *continuity equation* — mass is conserved — with a velocity field $\mathbf{v}_t$ that depends nonlinearly on $\mu_t$ itself (through the output $f(x;\mu_t)$). The self-referential dependence is the "McKean" part.

> [!NOTE] Well-Posedness
> Under mild regularity conditions on $\sigma$ and $\ell$, the McKean-Vlasov equation has a unique weak solution. Moreover, the empirical measure $\mu^N_t$ converges to $\mu_t$ at rate $O(N^{-1/2})$ in the 2-Wasserstein distance, uniformly on finite time intervals.

> [!INFO] Gradient Flow on Wasserstein Space
> The McKean-Vlasov equation can be written as a gradient flow on the space $\mathcal{P}_2(\mathbb{R}^{d+1})$ of probability measures with finite second moment, equipped with the 2-Wasserstein metric $W_2$:
> $$\dot{\mu}_t = -\nabla_{W_2} \mathcal{L}(\mu_t), \qquad \mathcal{L}(\mu) = \mathbb{E}_{(x,y)}\!\left[\ell\!\left(\int \Phi(\theta,x)\,d\mu(\theta), y\right)\right].$$
> This formulation connects the mean-field limit to optimal transport theory.

### 4.3 Feature Learning as Evolution of the Weight Distribution

📐 The key quantity distinguishing the two regimes is the *marginal distribution over first-layer features*:
$$\rho_t(w) \;:=\; \int_{\mathbb{R}} \mu_t(a, w)\,da.$$

| Regime | $\rho_t$ evolution |
|--------|-------------------|
| Lazy | $\rho_t \approx \rho_0 = \mathcal{N}(0, I_d)$ frozen |
| Rich | $\rho_t$ concentrates toward task-relevant directions |

**This is the precise mathematical definition of feature learning:** the marginal weight distribution $\rho_t$ changes non-trivially during training.

> [!EXAMPLE] Neuron Specialization in Teacher-Student Models
> Consider a teacher $f^*(x) = \frac{1}{k}\sum_{j=1}^k a^*_j \sigma({w^*_j}^\top x)$ with $k$ features. In the rich regime, $\rho_t$ evolves from the isotropic Gaussian $\rho_0$ toward a distribution with $k$ clusters, each centered near one of the teacher directions $w^*_j$. Neurons *specialize*: each student neuron $\theta_i$ migrates toward one teacher direction. The output $f(x;\mu_t)$ improves not by changing the functional form (as in the lazy case) but by adapting the feature basis.

> [!QUESTION] Exercise 3: When Does the Mean-Field NTK Vanish?
> *This problem establishes the precise rate at which the mean-field NTK vanishes, confirming the breakdown of linearization.*
>
> > **Prerequisites:** [[#4.1 Mean-Field Scaling and Breakdown of Linearization|Mean-field scaling]]
>
> For the mean-field parameterization $f(x;\theta) = \frac{1}{N}\sum_i a_i \sigma(w_i^\top x)$:
> 1. Compute $K^{MF}_\theta(x,x')$ explicitly.
> 2. Show that $K^{MF}_\theta(x,x') = \frac{1}{N} K^\infty_\text{MF}(x,x') + O(N^{-3/2})$ and identify $K^\infty_\text{MF}$.
> 3. Explain why this scaling implies that any $O(1)$-time gradient flow trajectory in the mean-field parameterization is *not* approximated by the corresponding NTK dynamics (i.e., why the linearization error grows to $O(1)$ rather than vanishing).

> [!TIP]- Solution to Exercise 3
> **Key insight:** The NTK is $O(1/N)$, so gradient flow driven by the NTK predicts changes of size $O(1/N) \cdot O(1)$ time $= O(1/N)$ — but the actual output changes by $O(1)$. The linearization error is thus $O(1) - O(1/N) = O(1)$, not small.
>
> **Sketch:**
> 1. $K^{MF}_\theta(x,x') = \frac{1}{N^2}\sum_i\bigl[a_i^2 \sigma'(w_i^\top x)\sigma'(w_i^\top x')\langle x,x'\rangle + \sigma(w_i^\top x)\sigma(w_i^\top x')\bigr]$.
> 2. By LLN, this is $\frac{1}{N}\mathbb{E}[\ldots] + O(N^{-3/2})$, so $K^\infty_\text{MF} = K^\infty$ from before (same limiting kernel, different scaling).
> 3. The NTK ODE $\dot{r}(t) = -K^{MF}(t)\, r(t)$ has characteristic timescale $\sim 1/\lambda_{\min}(K^{MF}) = O(N)$. So on the $O(1)$ timescale relevant for mean-field training, the NTK-linearized dynamics predict almost *no* change in the output, while the true dynamics produce $O(1)$ changes. The approximation error is $O(1)$.

---

## 5. The Output Multiplier as Control Parameter 🎛️

The lazy and rich regimes are unified by the output multiplier $\alpha$. Write:
$$f(x;\theta) = \frac{\alpha}{\sqrt{N}}\sum_{i=1}^N a_i\,\sigma(w_i^\top x).$$

The gradient of $L$ w.r.t. $w_i$ is proportional to $\alpha/\sqrt{N}$, so cumulative feature change after time $T$:
$$\|w_i(T) - w_i(0)\|_2 = O\!\left(\frac{\alpha T}{\sqrt{N}}\right).$$

| $\alpha$ scaling | Feature change $\|{\Delta w_i}\|$ | Regime |
|-----------------|--------------------------------|--------|
| $\alpha = O(1)$ | $O(1/\sqrt{N}) \to 0$ | **Lazy** |
| $\alpha = O(\sqrt{N})$ | $O(1)$ | **Rich** |
| $\alpha \gg \sqrt{N}$ | $O(\alpha/\sqrt{N}) \to \infty$ | Explosive |

**Proposition (Chizat et al. 2019).** In the $N \to \infty$ limit:
- *Lazy ($\alpha = O(1)$)*: the NTK is frozen, training dynamics are linear, and the network converges to a kernel predictor.
- *Rich ($\alpha = \Theta(\sqrt{N})$)*: the NTK changes during training, linearization fails, and the network exhibits feature learning described by the McKean-Vlasov PDE.

*Proof sketch.* In the lazy case, the gradient satisfies $\|\nabla_\theta L\| = O(\alpha/\sqrt{N})$, so parameter changes $\|\Delta\theta\| = O(\alpha T/\sqrt{N}) \to 0$. Since $K_\theta$ is a smooth function of $\theta$, $\|K_{\theta(t)} - K_{\theta(0)}\| \to 0$ uniformly, and the NTK is frozen. Linearizing around $\theta(0)$ then introduces only $O(\|\Delta\theta\|^2) = O(\alpha^2 T^2/N) \to 0$ error.

In the rich case, write $\alpha = \beta\sqrt{N}$. Then $f = \frac{\beta}{N}\sum_i a_i \sigma(w_i^\top x)$, which is precisely the mean-field parameterization, and the McKean-Vlasov limit applies. $\square$

> [!QUESTION] Exercise 4: Interpolating Between Regimes at Finite Width
> *This problem explores how finite width softens the sharp lazy/rich boundary.*
>
> > **Prerequisites:** [[#5. The Output Multiplier as Control Parameter|Output multiplier]]
>
> At fixed finite width $N$, consider $\alpha = N^\beta$ for $\beta \in [0, 1/2]$.
> 1. Show that the cumulative feature change after $T$ steps is $\|w_i(T)-w_i(0)\| = O(N^{\beta - 1/2} T)$.
> 2. For what range of $\beta$ do features change by $O(1)$?
> 3. What does this suggest about the "width" of the transition between lazy and rich behavior at finite $N$?

> [!TIP]- Solution to Exercise 4
> **Key insight:** The parameter $\beta$ continuously interpolates between the two regimes; the transition becomes sharp only as $N \to \infty$.
>
> **Sketch:**
> 1. Since $\alpha = N^\beta$, the gradient is $O(\alpha/\sqrt{N}) = O(N^{\beta - 1/2})$, so $\|w_i(T)-w_i(0)\| = O(N^{\beta-1/2} T)$.
> 2. This is $O(1)$ when $\beta = 1/2$ (the rich regime boundary), $O(N^{-1/2})$ when $\beta=0$ (lazy), and $O(N^{\beta-1/2})$ in between.
> 3. The transition is not a sharp phase transition at finite $N$ — it is a crossover. The width of the crossover region shrinks as $N$ grows, converging to a sharp dichotomy only in the $N \to \infty$ limit. This is analogous to finite-size rounding of phase transitions in statistical mechanics.

---

## 6. The Maximal Update Parameterization 📐

### 6.1 Deep Networks and the Tensor Programs Framework

For deep networks, one cannot simply count $1/\sqrt{N}$ vs. $1/N$ prefactors — the multi-layer structure creates a web of interdependencies between initialization variances and learning rates. The **Tensor Programs** framework (Yang & Hu 2021) resolves this systematically.

Consider a depth-$L$ MLP with hidden width $N$:
$$h^0 = x, \quad z^\ell = W^\ell h^{\ell-1}, \quad h^\ell = \sigma(z^\ell), \quad f = v^\top h^L,$$

where $W^\ell \in \mathbb{R}^{N \times N}$ and $v \in \mathbb{R}^N$. Write the hyperparameters as:
$$W^\ell_{ij} \sim \mathcal{N}(0,\, \sigma^2_\ell / N), \qquad v_i \sim \mathcal{N}(0, \sigma^2_{\text{out}}/N), \qquad \eta_\ell = \eta \cdot N^{c_\ell},$$

where $\sigma^2_\ell$ and $c_\ell$ are per-layer constants to be chosen.

**The question:** For which exponents $c_\ell$ does the network exhibit non-trivial, non-explosive training dynamics as $N \to \infty$?

The Tensor Programs answer this by tracking the $N$-scaling of every intermediate quantity (pre-activations, activations, backpropagated signals, weight updates) through the forward and backward passes, using a set of master equations called *tensor programs*.

### 6.2 The Two Non-Trivial Limits

The framework shows that — up to degenerate cases — there are exactly **two** consistent, non-explosive parameterizations that survive the $N \to \infty$ limit with non-trivial dynamics:

> [!NOTE] NTP (Standard Parameterization) — The Lazy Limit
> - All layers: $\sigma^2_\ell = 1$ (so $W^\ell_{ij} \sim \mathcal{N}(0, 1/N)$)
> - Learning rate: $\eta_\ell = \eta$ for all $\ell$ (no $N$-scaling)
> - Pre-activations: $z^\ell = O(1)$ at all layers
> - Weight updates $\Delta W^\ell$: contribute $O(1/N) \cdot O(\sqrt{N}) = O(1/\sqrt{N})$ to hidden-layer changes — **vanishing**
> - Limit: NTK dynamics, kernel regression

> [!NOTE] μP (Maximal Update Parameterization) — The Rich Limit
> - Hidden layers: $W^\ell_{ij} \sim \mathcal{N}(0, 1/N)$ (same as NTP)
> - Output layer: $v_i \sim \mathcal{N}(0, 1/N^2)$ (extra $1/\sqrt{N}$ suppression)
> - Learning rate for output $v$: $\eta_{\text{out}} = \eta$ (unscaled)
> - Learning rate for hidden $W^\ell$: $\eta_\ell = \eta / N$ (scaled down)
> - Weight updates: each layer's update contributes $O(1)$ to the output — **maximal**
> - Limit: mean-field / feature-learning dynamics

The name "Maximal Update Parameterization" refers to the fact that $\mu P$ is the unique parameterization where every layer achieves the *maximum* possible contribution to the output change without causing divergence.

> [!DANGER] Common Misconception
> μP does *not* just mean "scale everything down by $1/N$." The key is that **different layers are scaled differently**: hidden layers use the same initialization as NTP, but the output layer is suppressed by an extra $1/\sqrt{N}$, and hidden-layer learning rates are scaled by $1/N$ while the output learning rate is not. Getting any of these wrong destroys the feature-learning property.

### 6.3 Hyperparameter Transfer

💡 The most practically important consequence of μP is **hyperparameter transfer across widths**.

**Claim:** In μP, the optimal learning rate $\eta^*$ is approximately *independent of width $N$* for large enough $N$.

*Why?* Under μP, the per-step output change is $O(\eta)$ regardless of $N$ (by construction — this is what "maximal updates" means). So the effective learning rate as experienced by the loss is $O(\eta)$ at all widths, meaning the optimal $\eta$ doesn't need to be re-tuned when $N$ changes.

Under NTP, the per-step output change is $O(\eta/\sqrt{N})$, so the optimal $\eta$ must scale as $\sqrt{N}$ — it is width-dependent.

| Parameterization | Optimal $\eta^*$ | Transfer? |
|-----------------|-----------------|-----------|
| NTP / SP | $\Theta(\sqrt{N})$ | ✗ — must retune |
| μP | $\Theta(1)$ | ✓ — transfers |

**Practical consequence:** Tune $\eta$ on a small proxy model of width $N_0 \ll N_\text{target}$ (cheap), then deploy the same $\eta$ on the full model. This reduces the cost of hyperparameter search by orders of magnitude for large models.

> [!QUESTION] Exercise 5: Why μP Learning Rate is Constant
> *This problem derives the width-independence of the optimal μP learning rate from the first-order Taylor expansion of the loss.*
>
> > **Prerequisites:** [[#6.2 The Two Non-Trivial Limits|μP definition]], [[#5. The Output Multiplier as Control Parameter|Output multiplier]]
>
> Consider a 2-layer μP network: $f(x) = \frac{1}{N} a^\top \sigma(Wx)$ with $W_{ij} \sim \mathcal{N}(0,1)$, $a_i \sim \mathcal{N}(0,1)$, learning rate $\eta_W$ for $W$ and $\eta_a$ for $a$.
>
> 1. Write the gradient flow equations for $W$ and $a$.
> 2. Show that for the output $f(x)$ to change by $O(1)$ per unit gradient-flow time, we need $\eta_W = O(N)$ and $\eta_a = O(1)$.
> 3. Explain why this means the μP prescription for a 2-layer network is $\eta_W = \eta N$ (or equivalently, a per-neuron learning rate of $\eta$) for the hidden layer.

> [!TIP]- Solution to Exercise 5
> **Key insight:** In μP, the $1/N$ prefactor in $f$ means each weight $W_{ij}$ contributes $O(1/N)$ to $f$, so its gradient is $O(1/N)$. To get $O(1)$ output change per unit time from updating $W$, we need learning rate $\eta_W = O(N)$.
>
> **Sketch:**
> 1. $\dot{W}_{ij} = -\eta_W \nabla_{W_{ij}} L = -\eta_W \cdot \frac{1}{N} a_i \sigma'(h_i(x))x_j \cdot \frac{\partial L}{\partial f}$, so $\|\dot{W}\| = O(\eta_W / N)$. Similarly $\dot{a}_i = O(\eta_a/N)$.
> 2. Change in output: $\dot{f}(x) = \frac{1}{N}\sum_i [a_i \sigma'(h_i)\dot{W}_i^\top x + \dot{a}_i \sigma(h_i)] = O(\eta_W/N^2) \cdot N + O(\eta_a/N^2)\cdot N = O(\eta_W/N + \eta_a/N)$. For this to be $O(1)$: $\eta_W = O(N)$ and $\eta_a = O(N)$.
> 3. Under μP, the canonical choice is $\eta_W = \eta \cdot N$ for hidden weights. Rescaling by $1/N$ per neuron (i.e. per-neuron learning rate $\eta$) gives the "maximal" output update. This is width-independent: the $N$ in $\eta_W = \eta N$ exactly cancels the $1/N$ from the output prefactor, leaving an $O(\eta)$ per-step loss decrease independent of $N$.

---

## 7. Summary Comparison 📊

| Property | Lazy (NTK) | Rich (Mean-Field / μP) |
|----------|-----------|------------------------|
| **Output scaling** | $\alpha/\sqrt{N}$, $\alpha = O(1)$ | $1/N$ or $\alpha = \Theta(\sqrt{N})$ |
| **Init output** | $O(1)$ | $O(1/\sqrt{N}) \to 0$ |
| **Feature change** $\|\Delta w_i\|$ | $O(1/\sqrt{N}) \to 0$ | $O(1)$ |
| **NTK during training** | Frozen: $K_t \approx K_0$ | Changes: $K_t \not\approx K_0$ |
| **$N\to\infty$ dynamics** | Linear ODE (kernel regression) | McKean-Vlasov PDE |
| **Feature learning** | No | Yes |
| **Sample efficiency** | Same as kernel methods | Better on structured tasks |
| **HP transfer across widths** | No ($\eta^* \propto \sqrt{N}$) | Yes ($\eta^*$ = const) |
| **Analytically tractable** | ✓ Exactly | Partially (mean-field limit) |

> [!TIP] Which Regime Are Real Networks In?
> Modern large language models trained with standard Adam and typical initialization are closer to the **rich** regime, but not strictly in either limit (finite width, finite learning rate). The μP analysis suggests they *should* be trained in the μP parameterization to maximize feature learning and enable hyperparameter transfer. Recent work (Yang et al. 2022, Bordelon et al. 2024) confirms that μP-trained models outperform SP-trained models at matched compute, supporting the rich-regime hypothesis for practical deep learning.

---

## References

| Reference Name | Brief Summary | Link |
|---------------|--------------|------|
| [Jacot, Gabriel, Hongler (2018) — Neural Tangent Kernel](https://arxiv.org/abs/1806.07572) | Introduced the NTK; proved that infinite-width networks trained by gradient descent converge to kernel regression | https://arxiv.org/abs/1806.07572 |
| [Chizat, Oyallon, Bach (2019) — On Lazy Training](https://arxiv.org/abs/1812.07956) | Coined "lazy training"; precisely characterized how output multiplier $\alpha$ controls the lazy/rich transition | https://arxiv.org/abs/1812.07956 |
| [Mei, Montanari, Nguyen (2018) — Mean Field View of SGD](https://arxiv.org/abs/1805.09538) | Derived the McKean-Vlasov PDE as the $N\to\infty$ limit of mean-field networks | https://arxiv.org/abs/1805.09538 |
| [Chizat, Bach (2018) — Global Convergence of Gradient Descent](https://arxiv.org/abs/1805.09545) | Proved global convergence for the mean-field (rich) limit using Wasserstein gradient flow | https://arxiv.org/abs/1805.09545 |
| [Yang, Hu (2021) — Feature Learning in Infinite-Width Networks](https://arxiv.org/abs/2011.14522) | Introduced μP via Tensor Programs IV; proved the dichotomy for deep networks | https://arxiv.org/abs/2011.14522 |
| [Yang et al. (2022) — Tensor Programs V: μTransfer](https://arxiv.org/abs/2203.03466) | Empirical validation of μP hyperparameter transfer across widths; introduced μTransfer | https://arxiv.org/abs/2203.03466 |
| [Simon et al. (2026) — There Will Be a Scientific Theory of Deep Learning](https://arxiv.org/abs/2604.21691) | Survey/perspective situating lazy/rich dichotomy within the broader program of "learning mechanics" | https://arxiv.org/abs/2604.21691 |
