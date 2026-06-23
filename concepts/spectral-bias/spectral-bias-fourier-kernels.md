# Spectral Bias in Neural Networks and Resolution via Fourier Kernel Methods

## Table of Contents

- [[#1. Introduction|1. Introduction]]
- [[#2. The Spectral Bias Phenomenon|2. The Spectral Bias Phenomenon]]
  - [[#2.1 Empirical Evidence|2.1 Empirical Evidence]]
  - [[#2.2 Formal Definition and the Frequency Principle|2.2 Formal Definition and the Frequency Principle]]
  - [[#2.3 Fourier Spectrum of ReLU Networks|2.3 Fourier Spectrum of ReLU Networks]]
- [[#3. NTK Analysis of Spectral Bias|3. NTK Analysis of Spectral Bias]]
  - [[#3.1 Neural Tangent Kernel as Kernel Regression|3.1 Neural Tangent Kernel as Kernel Regression]]
  - [[#3.2 Eigendecomposition and Frequency-Dependent Convergence|3.2 Eigendecomposition and Frequency-Dependent Convergence]]
  - [[#3.3 Spherical Harmonics and the Decay of NTK Eigenvalues|3.3 Spherical Harmonics and the Decay of NTK Eigenvalues]]
- [[#4. Fourier Feature Maps|4. Fourier Feature Maps]]
  - [[#4.1 Bochner's Theorem|4.1 Bochner's Theorem]]
  - [[#4.2 Rahimi and Recht: Random Fourier Features|4.2 Rahimi and Recht: Random Fourier Features]]
  - [[#4.3 Positional Encodings as Deterministic Fourier Features|4.3 Positional Encodings as Deterministic Fourier Features]]
- [[#5. Effect on the NTK Spectrum|5. Effect on the NTK Spectrum]]
  - [[#5.1 The Composed Kernel|5.1 The Composed Kernel]]
  - [[#5.2 Stationarity and Bandwidth Control|5.2 Stationarity and Bandwidth Control]]
  - [[#5.3 Eigenvalue Equalization and Kernel Preconditioning|5.3 Eigenvalue Equalization and Kernel Preconditioning]]
- [[#6. Convergence Rate Analysis|6. Convergence Rate Analysis]]
  - [[#6.1 Baseline Convergence in the NTK Regime|6.1 Baseline Convergence in the NTK Regime]]
  - [[#6.2 Convergence After Fourier Feature Lifting|6.2 Convergence After Fourier Feature Lifting]]
- [[#7. Applications|7. Applications]]
  - [[#7.1 NeRF Positional Encodings|7.1 NeRF Positional Encodings]]
  - [[#7.2 SIREN: Sinusoidal Activation Networks|7.2 SIREN: Sinusoidal Activation Networks]]
  - [[#7.3 Occupancy Networks and Other Implicit Representations|7.3 Occupancy Networks and Other Implicit Representations]]
- [[#8. References|8. References]]

---

## 1. Introduction 🗺️

*Implicit neural representations* (INRs) are networks that map coordinates $(x, y, z, t, \ldots)$ directly to signal values: pixel intensities, signed distances, density fields, or radiance. The appeal is compact parameterization — a single MLP encodes a whole scene or object.

The obstacle is fundamental: plain MLPs with smooth (e.g., ReLU, tanh) activations exhibit a strong *spectral bias* — they learn to represent low-frequency content orders of magnitude faster than high-frequency content. For natural signals (images, 3D geometry, audio), the high-frequency components carry texture, edges, and fine detail. A spectrally biased network trained on a 3D scene will render a blurry blob long before it can reproduce surface texture.

This note develops the spectral bias story from three angles:

1. **Empirical and Fourier-analytic** (Rahaman et al., 2019): what we observe in practice and what the Fourier transform of ReLU networks reveals.
2. **NTK-theoretic** (Jacot et al., 2018; Basri et al., 2019): how the neural tangent kernel's eigenspectrum directly encodes the frequency-learning schedule.
3. **Feature-map resolution** (Rahimi & Recht, 2007; Tancik et al., 2020): how *Fourier feature maps* reshape the induced kernel, equalizing eigenvalues across frequency modes and recovering high-frequency learning ability.

> [!INFO] Scope
> This note focuses on coordinate-based MLPs (low-dimensional inputs, $d \leq 3$ typically). The spectral bias analysis generalizes to higher dimensions but the resolution via Fourier features is most effective — and most studied — in the low-dimensional regime.

---

## 2. The Spectral Bias Phenomenon 📊

### 2.1 Empirical Evidence

The following experiment (following Rahaman et al., 2019) cleanly isolates the phenomenon.

**Setup.** Train a deep ReLU network $f_\theta : \mathbb{R} \to \mathbb{R}$ to regress a target $g(x) = \sum_{k} A_k \sin(2\pi k x)$ on $x \in [0,1]$, where frequencies $k \in \{1, 5, 10, 40\}$ are simultaneously present.

**Observation.** Monitoring the *residual power spectrum* $|\hat{r}(k)|^2$ where $\hat{r}(k) = \widehat{g - f_{\theta^{(t)}}}(k)$ at training iteration $t$:

- The $k=1$ component vanishes within a few hundred iterations.
- The $k=5$ component decays an order of magnitude later.
- $k=10, 40$ components remain present long into training.

*Crucially, this ordering holds even when higher-frequency amplitudes $A_k$ are larger.* The network's preference is not amplitude-driven but frequency-driven.

A secondary experiment perturbs the parameters of a trained network by additive Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$ and measures the effect on each frequency component. Low-frequency components are nearly unaffected; high-frequency components degrade rapidly. This shows that *high-frequency representations occupy a small volume in parameter space* — they require precise parameter coordination, making them both hard to learn and fragile.

### 2.2 Formal Definition and the Frequency Principle

**Definition (Spectral Bias).** Let $f^{(t)} : \mathbb{R}^d \to \mathbb{R}$ be the network function at training time $t$, and let $f^* : \mathbb{R}^d \to \mathbb{R}$ be the target function. The network exhibits *spectral bias* if the residual error $r^{(t)} = f^* - f^{(t)}$ has the property that, for any $\xi_1, \xi_2 \in \mathbb{R}^d$ with $\|\xi_1\| < \|\xi_2\|$:

$$|\hat{r}^{(t)}(\xi_2)| \gg |\hat{r}^{(t)}(\xi_1)|$$

throughout most of training, i.e., the high-frequency residual decays far slower than the low-frequency residual.

**Definition (Frequency Principle / F-Principle).** Training dynamics satisfy the *frequency principle* if the Fourier components of the training residual decay approximately in order of increasing frequency $\|\xi\|$.

The two definitions are closely related. The frequency principle is the dynamic statement (frequencies learned in order); spectral bias is the static description of the outcome (networks are biased toward low-frequency solutions).

### 2.3 Fourier Spectrum of ReLU Networks

Rahaman et al. (2019) provide a Fourier-analytic characterization of what ReLU networks can express and at what cost.

**Setup.** Let $f_\theta : \mathbb{R}^d \to \mathbb{R}$ be an $L$-layer ReLU network with width $n$ and weight matrices $\{W^{(l)}\}$. A ReLU network is *piecewise linear*, partitioning its input domain into a set of *linear regions* (convex polytopes) $\{P_i\}$, on each of which $f_\theta$ is an affine function.

**Theorem 1 (Rahaman et al., 2019 — Fourier Spectrum of ReLU Networks).** For a piecewise-linear function $f_\theta$ with linear regions $\{P_i\}$, the Fourier transform satisfies:

$$\hat{f}_\theta(\xi) = \sum_i \sum_{n=0}^{d} \frac{C_n(\theta, \xi) \cdot \mathbf{1}_{H^i_n}(\xi)}{\|\xi\|^{n+1}}$$

where $H^i_n$ is the set of frequency directions aligned with $n$-dimensional faces of polytope $P_i$, and $C_n(\theta, \xi)$ is a bounded coefficient depending on the local affine map and face geometry.

The upshot: the Fourier transform of a ReLU network decays as $\|\xi\|^{-(n+1)}$ for some $n \in \{0, \ldots, d\}$, so the spectral content is suppressed polynomially at high frequencies. For a generic direction, $n = 0$ and the decay is $\|\xi\|^{-1}$ — the mildest suppression possible. But for directions perpendicular to high-dimensional faces, $n = d-1$ and the decay is $\|\xi\|^{-d}$, much stronger.

**Corollary.** The rate at which the residual's $\xi$-component changes satisfies:

$$\left|\frac{d\hat{r}^{(t)}(\xi)}{dt}\right| = O\!\left(\|\xi\|^{-\Delta}\right), \quad \Delta \in [1, d]$$

meaning *high-frequency residual components change more slowly during gradient descent*, giving the formal basis for the empirical observations in §2.1.

> [!WARNING] NTK connection is implicit here
> Rahaman et al. (2019) predates the NTK literature's rigorous application to spectral bias. The $O(\|\xi\|^{-\Delta})$ rate is derived via Fourier analysis of piecewise-linear networks, not from NTK eigendecomposition. The NTK-based derivation (§3) provides a cleaner and more general framework.

> [!EXAMPLE]- Example: Manifold Complexity Lifts the Bias
> Rahaman et al. also observe that spectral bias is *relative to the intrinsic coordinates of the data manifold*. If inputs lie on a manifold $\mathcal{M}$ parameterized by $\gamma : \mathbb{R}^m \to \mathbb{R}^d$, then a frequency $l$ in the manifold's intrinsic coordinate $z$ corresponds to network frequency $k$ via:
>
> $$l = J_\gamma(\bar{z}) \cdot k$$
>
> where $J_\gamma$ is the Jacobian of the embedding. A complex, rapidly-varying embedding magnifies intrinsic frequencies — a low network-frequency $k$ corresponds to a high intrinsic frequency $l$. This is why, *counterintuitively*, networks trained on high-dimensional or geometrically complex data can appear to learn high-frequency functions more readily.

> [!QUESTION] Exercise 1: Spectral Decay from Piecewise Linearity
> *This exercise establishes why piecewise linear networks have limited high-frequency content.*
>
> > **Prerequisites:** [[#2.3 Fourier Spectrum of ReLU Networks|2.3 Fourier Spectrum of ReLU Networks]]
>
> Let $f : \mathbb{R} \to \mathbb{R}$ be piecewise linear with breakpoints at $x_1 < x_2 < \cdots < x_N$ and slopes $a_0, a_1, \ldots, a_N$ on successive intervals. Write $f$ as a sum of ramp functions and use this to derive the asymptotic behavior of $\hat{f}(\xi)$ as $|\xi| \to \infty$. What does this imply about the spectral content of any deep ReLU network?

> [!TIP]- Solution to Exercise 1
> **Key insight:** A piecewise linear function $f$ can be written as $f(x) = a_0 x + b + \sum_{i=1}^N (a_i - a_{i-1}) \max(x - x_i, 0)$. The second derivative $f''$ is a sum of Dirac deltas: $f''(x) = \sum_i (a_i - a_{i-1}) \delta(x - x_i)$.
>
> **Sketch:** Taking the Fourier transform of the relation $\widehat{f''}(\xi) = (2\pi i \xi)^2 \hat{f}(\xi)$:
>
> $$\hat{f}(\xi) = \frac{\widehat{f''}(\xi)}{(2\pi i \xi)^2} = \frac{\sum_i (a_i - a_{i-1}) e^{-2\pi i \xi x_i}}{(2\pi i \xi)^2}$$
>
> So $|\hat{f}(\xi)| = O(|\xi|^{-2})$ as $|\xi| \to \infty$. In $d$ dimensions, the analogous argument yields $O(\|\xi\|^{-(n+1)})$ decay where $n$ reflects the codimension of the discontinuity set of the highest-order derivative, matching Theorem 1. The implication: ReLU networks cannot place Fourier mass at high frequencies without a multiplicative cost of $\|\xi\|^{n+1}$ in the numerator — achievable only through very fine coordination of weights.

---

## 3. NTK Analysis of Spectral Bias 📐

### 3.1 Neural Tangent Kernel as Kernel Regression

The *neural tangent kernel* (NTK), introduced by Jacot et al. (2018), characterizes the infinite-width limit of neural network training dynamics.

**Definition (NTK).** For a network $f(\cdot;\theta) : \mathbb{R}^d \to \mathbb{R}$ with parameters $\theta \in \mathbb{R}^P$, the *neural tangent kernel* is:

$$K_{\mathrm{NTK}}(x, x') = \left\langle \frac{\partial f(x;\theta)}{\partial \theta},\, \frac{\partial f(x';\theta)}{\partial \theta} \right\rangle_{\mathbb{R}^P}$$

In the infinite-width limit under gradient flow on a squared loss $\mathcal{L} = \frac{1}{2}\|f_\theta(X) - y\|^2$, two facts hold:

1. **Constancy:** $K_{\mathrm{NTK}}$ remains constant throughout training (i.e., $\frac{d}{dt} K_{\mathrm{NTK}} = 0$).
2. **Kernel regression limit:** The network predictions $\hat{y}^{(t)} = f_\theta^{(t)}(X_{\mathrm{test}})$ converge to the kernel regression solution:

$$\hat{y}^{(t)}_{\mathrm{test}} = K_{\mathrm{test}} K^{-1} \left(I - e^{-\eta K t}\right) y$$

where $K = K_{\mathrm{NTK}}(X_{\mathrm{train}}, X_{\mathrm{train}}) \in \mathbb{R}^{n \times n}$ is the train–train kernel matrix, $K_{\mathrm{test}} = K_{\mathrm{NTK}}(X_{\mathrm{test}}, X_{\mathrm{train}}) \in \mathbb{R}^{m \times n}$, and $\eta$ is the learning rate.

### 3.2 Eigendecomposition and Frequency-Dependent Convergence

Let $K = Q \Lambda Q^\top$ be the eigendecomposition with $\Lambda = \mathrm{diag}(\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n > 0)$ and orthonormal eigenvectors $Q = [q_1 | q_2 | \cdots | q_n]$.

**Proposition (NTK Convergence in Eigenbasis).** Under gradient flow, the error in the eigenbasis satisfies:

$$Q^\top \!\left(\hat{y}^{(t)}_{\mathrm{train}} - y\right) = -e^{-\eta \Lambda t}\, Q^\top y$$

That is, the $i$-th eigenvector component of the error decays as $e^{-\eta \lambda_i t}$.

*Proof sketch.* The training residual $r^{(t)} = y - \hat{y}^{(t)}_{\mathrm{train}}$ evolves as $\dot{r}^{(t)} = -K r^{(t)}$, giving $r^{(t)} = e^{-Kt} r^{(0)}$. Rotating by $Q^\top$: $(Q^\top r^{(t)})_i = e^{-\lambda_i t} (Q^\top r^{(0)})_i$. $\square$

**Definition (Convergence Rate of Mode $i$).** The $i$-th mode converges at rate $\eta \lambda_i$; its *half-life* is $(\eta \lambda_i)^{-1} \log 2$.

This is the core of the NTK-based explanation of spectral bias:

💡 **If $\lambda_i \ll \lambda_j$ for eigenvectors $q_i, q_j$ corresponding to high and low frequencies respectively, then the high-frequency mode takes far longer to fit.**

**Key question:** Do the NTK eigenvectors actually align with frequency modes, and do low-frequency modes have larger eigenvalues?

### 3.3 Spherical Harmonics and the Decay of NTK Eigenvalues

For training data distributed uniformly on the $(d-1)$-sphere $\mathbb{S}^{d-1}$, the NTK matrix forms a *zonal kernel* — it depends only on the inner product $x^\top x'$:

$$K_{\mathrm{NTK}}(x, x') = h(x^\top x'), \quad \|x\| = \|x'\| = 1$$

By the theory of zonal kernels on spheres, the *eigenfunctions* of the associated integral operator

$$(\mathcal{K} f)(x) = \int_{\mathbb{S}^{d-1}} h(x^\top x') f(x')\, d\sigma(x')$$

are exactly the *spherical harmonics* $\{Y_{l,j}\}$ of degree $l = 0, 1, 2, \ldots$ and order $j = 1, \ldots, N(d,l)$, where $N(d,l)$ is the multiplicity of degree $l$.

**Theorem (Basri et al., 2019; Bietti & Mairal, 2019).** For a two-layer overparameterized ReLU network with weights initialized from a standard isotropic Gaussian, the NTK eigenvalue $\mu_l$ corresponding to degree-$l$ spherical harmonics satisfies:

$$\mu_l \sim l^{-(d+1)}$$

as $l \to \infty$. That is, **eigenvalues decay polynomially in frequency, with exponent $d+1$.**

**Corollary (Frequency-Dependent Convergence Rate).** In the NTK regime with uniform data on $\mathbb{S}^{d-1}$, the degree-$l$ spherical harmonic component of the training error decays as $e^{-\eta \mu_l t} \approx e^{-\eta l^{-(d+1)} t}$. For the error to reach half its initial value requires time:

$$t_{1/2}(l) \sim l^{d+1}$$

**This is the formal quantification of spectral bias:** high-frequency modes (large $l$) take polynomially longer to learn, with the gap growing with input dimension $d$.

> [!NOTE] Linear F-Principle Model
> For general (non-spherical) data distributions, an explicit model of the learning dynamics is available (Zhang et al., 2021; Xu et al., 2019). For two-layer ReLU networks, the residual evolves in Fourier space as:
>
> $$\frac{\partial}{\partial t}\hat{u}(\xi) = -\gamma(\xi)^2 \hat{u}(\xi)$$
>
> where the decay rate function is:
>
> $$\gamma(\xi)^2 = \frac{C_1}{\|\xi\|^{d+3}} + \frac{C_2}{\|\xi\|^{d+1}}$$
>
> The dominant term $\propto \|\xi\|^{-(d+1)}$ precisely matches the spherical harmonic eigenvalue decay rate, confirming the NTK picture holds beyond the uniform-on-sphere setting.

> [!QUESTION] Exercise 2: Convergence Time Scaling
> *This exercise makes the dimensional dependence of spectral bias concrete.*
>
> > **Prerequisites:** [[#3.2 Eigendecomposition and Frequency-Dependent Convergence|3.2 Eigendecomposition and Frequency-Dependent Convergence]], [[#3.3 Spherical Harmonics and the Decay of NTK Eigenvalues|3.3 Spherical Harmonics and the Decay of NTK Eigenvalues]]
>
> Suppose you have two frequency modes, $l_{\mathrm{lo}} = 1$ and $l_{\mathrm{hi}} = L$, in a problem with input dimension $d$. Using the result $t_{1/2}(l) \sim l^{d+1}$:
>
> (a) Compute the ratio $t_{1/2}(l_{\mathrm{hi}}) / t_{1/2}(l_{\mathrm{lo}})$ as a function of $L$ and $d$.
>
> (b) For $d = 3$ (3D spatial coordinates, as in NeRF) and $L = 10$, how many times longer does the high-frequency mode take to converge?
>
> (c) What does this imply about learning sharp edges (say, $l \approx 100$) in a 3D scene?

> [!TIP]- Solution to Exercise 2
> **Key insight:** The polynomial decay of NTK eigenvalues with frequency creates a time-to-convergence ratio that grows as a power law in the frequency ratio.
>
> **Sketch:**
>
> (a) $t_{1/2}(l_{\mathrm{hi}}) / t_{1/2}(l_{\mathrm{lo}}) = L^{d+1} / 1^{d+1} = L^{d+1}$.
>
> (b) $d=3, L=10$: ratio $= 10^{3+1} = 10^4 = 10000$. The $l=10$ mode requires $10{,}000\times$ more training time than the $l=1$ mode.
>
> (c) For $l \approx 100$ in $d=3$: ratio $= 100^4 = 10^8$. Even if the $l=1$ mode converges in 1000 steps, the $l=100$ mode would need $10^{11}$ steps — computationally infeasible. This is precisely why plain MLPs fail on NeRF-like tasks: the geometry can be learned but the fine texture cannot.

---

## 4. Fourier Feature Maps 🔑

### 4.1 Bochner's Theorem

The theoretical bridge between shift-invariant kernels and random feature maps is *Bochner's theorem* from harmonic analysis.

**Definition (Shift-Invariant Kernel).** A kernel $k : \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$ is *shift-invariant* (stationary) if $k(x, x') = k(x - x')$ for some function $k : \mathbb{R}^d \to \mathbb{R}$.

**Theorem (Bochner).** A continuous, bounded, shift-invariant kernel $k : \mathbb{R}^d \to \mathbb{R}$ is positive definite if and only if it is the Fourier transform of a finite non-negative Borel measure $p$ on $\mathbb{R}^d$:

$$k(x - x') = \int_{\mathbb{R}^d} p(\omega)\, e^{i\omega^\top(x - x')}\, d\omega$$

If $k(0) = 1$ (normalized), then $p$ is a probability density on $\mathbb{R}^d$ and is called the *spectral density* of the kernel.

*Proof sketch.* The forward direction (positive definite $\Rightarrow$ Fourier transform of measure) follows from the Herglotz–Bochner theorem in harmonic analysis. The converse is the Bochner–Schwartz theorem: if $p \geq 0$, then $k(x - x') = \mathbb{E}_\omega[e^{i\omega^\top(x-x')}]$ is a convex combination of positive definite functions $\omega \mapsto e^{i\omega^\top \cdot}$, hence positive definite. $\square$

**Corollary (Real-Valued Form).** Since $k$ is real and symmetric, the imaginary parts cancel in the integral:

$$k(x - x') = \int_{\mathbb{R}^d} p(\omega) \cos(\omega^\top(x - x'))\, d\omega = \mathbb{E}_{\omega \sim p}\!\left[\cos(\omega^\top(x - x'))\right]$$

This is the key identity: the kernel is an *expectation over cosine features*.

### 4.2 Rahimi and Recht: Random Fourier Features

Rahimi & Recht (2007) exploit Bochner's theorem to construct a randomized, finite-dimensional feature map that approximates any shift-invariant kernel.

**Definition (Random Fourier Feature Map).** Given a shift-invariant kernel $k$ with spectral density $p$, sample $\omega_1, \ldots, \omega_m \overset{\mathrm{iid}}{\sim} p$ and define the *random Fourier feature map* $z : \mathbb{R}^d \to \mathbb{R}^{2m}$:

$$z(x) = \frac{1}{\sqrt{m}}\!\left[\cos(\omega_1^\top x),\, \sin(\omega_1^\top x),\, \ldots,\, \cos(\omega_m^\top x),\, \sin(\omega_m^\top x)\right]^\top$$

**Proposition (Unbiased Kernel Approximation).** The inner product $z(x)^\top z(x')$ is an unbiased estimator of the kernel:

$$\mathbb{E}\!\left[z(x)^\top z(x')\right] = k(x - x')$$

*Proof.* Using the Euler identity $\cos(\alpha)\cos(\beta) + \sin(\alpha)\sin(\beta) = \cos(\alpha - \beta)$:

$$\mathbb{E}\!\left[z(x)^\top z(x')\right] = \mathbb{E}_{\omega \sim p}\!\left[\cos(\omega^\top x)\cos(\omega^\top x') + \sin(\omega^\top x)\sin(\omega^\top x')\right] = \mathbb{E}_\omega\!\left[\cos(\omega^\top(x - x'))\right] = k(x - x') \quad \square$$

**Concentration.** By Hoeffding's inequality applied to the bounded random variables (since $|\cos(\cdot)| \leq 1$), the approximation error concentrates:

$$\Pr\!\left[\sup_{x, x'} \left|z(x)^\top z(x') - k(x - x')\right| > \epsilon\right] \leq C \cdot e^{-m\epsilon^2/C'}$$

so $m = O(\epsilon^{-2} \log(1/\delta))$ features suffice for uniform $\epsilon$-approximation with probability $1-\delta$.

**Example: RBF Kernel.** For the Gaussian (RBF) kernel $k(x - x') = \exp(-\|x-x'\|^2 / 2\sigma^2)$, the spectral density is $p(\omega) = \mathcal{N}(0, \sigma^{-2} I_d)$. Sampling $\omega \sim \mathcal{N}(0, \sigma^{-2} I)$ and forming $z$ gives an unbiased approximation to the RBF kernel.

> [!NOTE] Scaling and the RFF Approximation Quality
> The approximation quality depends on $m$ but *not* on $n$ (the number of training examples). This is why RFFs scale kernel methods to large datasets: replacing the $n \times n$ kernel matrix with $n \times 2m$ feature matrices reduces kernel SVM training from $O(n^2)$ to $O(nm)$.

### 4.3 Positional Encodings as Deterministic Fourier Features

Tancik et al. (2020) make the connection between Fourier features and neural network conditioning explicit and replace Monte Carlo sampling with a fixed, deterministic feature map.

**Definition (Fourier Feature Map / Positional Encoding).** Given a frequency matrix $B \in \mathbb{R}^{m \times d}$ with rows $b_1, \ldots, b_m$, define:

$$\gamma(v) = \left[\cos(2\pi b_1^\top v),\, \sin(2\pi b_1^\top v),\, \ldots,\, \cos(2\pi b_m^\top v),\, \sin(2\pi b_m^\top v)\right]^\top \in \mathbb{R}^{2m}$$

The MLP then operates on $\gamma(v)$ rather than $v$ directly: $f_\theta(\gamma(v))$.

Three variants arise depending on how $B$ is chosen:

| Variant | How $B$ is chosen | Kernel approximated |
|---------|-------------------|---------------------|
| **Random Fourier** | $b_j \overset{\mathrm{iid}}{\sim} \mathcal{N}(0, \sigma^2 I)$ | Gaussian kernel, bandwidth $\sigma^{-1}$ |
| **Deterministic grid** | $b_j = j \cdot \mathbf{e}_k$ for integer $j$, standard basis $\mathbf{e}_k$ | Fourier series (band-limited approximation) |
| **NeRF encoding** | $b_j = 2^{j-1} \mathbf{e}_k$, powers of 2 | Log-spaced frequency grid |

The NeRF encoding maps $v \mapsto [\cos(v), \sin(v), \cos(2v), \sin(2v), \ldots, \cos(2^{L-1}v), \sin(2^{L-1}v)]$ — a deterministic frequency ladder spanning from fundamental to $2^{L-1}$ times the fundamental.

> [!QUESTION] Exercise 3: Feature Map Inner Product as a Kernel
> *This exercise shows directly that the positional encoding's inner product defines a shift-invariant kernel.*
>
> > **Prerequisites:** [[#4.1 Bochner's Theorem|4.1 Bochner's Theorem]], [[#4.3 Positional Encodings as Deterministic Fourier Features|4.3 Positional Encodings as Deterministic Fourier Features]]
>
> Let $\gamma(v) = [\cos(b_j^\top v), \sin(b_j^\top v)]_{j=1}^m \in \mathbb{R}^{2m}$ with fixed frequency vectors $b_1, \ldots, b_m$.
>
> (a) Compute $\gamma(v)^\top \gamma(v')$ in closed form and show it depends only on $v - v'$.
>
> (b) Express the result as a sum of cosines and identify it as a truncated Fourier series.
>
> (c) Under what conditions on $\{b_j\}$ does this kernel approximate a Gaussian RBF kernel?

> [!TIP]- Solution to Exercise 3
> **Key insight:** The positional encoding inner product is a cosine kernel, shift-invariant by construction.
>
> **Sketch:**
>
> (a) Using $\cos(\alpha)\cos(\beta) + \sin(\alpha)\sin(\beta) = \cos(\alpha - \beta)$:
>
> $$\gamma(v)^\top \gamma(v') = \sum_{j=1}^m \cos(b_j^\top v)\cos(b_j^\top v') + \sin(b_j^\top v)\sin(b_j^\top v') = \sum_{j=1}^m \cos(b_j^\top(v - v'))$$
>
> This depends only on $v - v'$, confirming shift-invariance.
>
> (b) This is exactly a truncated (finite) Fourier series with frequencies $\{b_j\}$ and uniform coefficients $a_j = 1$.
>
> (c) By Bochner's theorem, the Gaussian RBF kernel has spectral density $p(\omega) = \mathcal{N}(0, \sigma^{-2}I)$. Taking $b_j \overset{\mathrm{iid}}{\sim} \mathcal{N}(0, \sigma^{-2}I)$ makes the sum a Monte Carlo estimate of $m \cdot k_{\mathrm{RBF}}(v - v')$ (up to the $1/m$ normalization). Thus, i.i.d. Gaussian $b_j$ with variance $\sigma^{-2}$ approximate the Gaussian kernel of bandwidth $\sigma$.

---

## 5. Effect on the NTK Spectrum 🔬

### 5.1 The Composed Kernel

When inputs are lifted via $\gamma$, the effective NTK of the network $f_\theta(\gamma(\cdot))$ changes. Let $h_{\mathrm{NTK}} : \mathbb{R} \to \mathbb{R}$ denote the NTK of the *base* MLP (which, for typical architectures, is a function of the inner product $\gamma(v)^\top \gamma(v')$).

**Definition (Composed Kernel).** The *composed kernel* induced by the Fourier feature map is:

$$K_\gamma(v, v') = h_{\mathrm{NTK}}\!\left(\gamma(v)^\top \gamma(v')\right) = h_{\mathrm{NTK}}\!\left(\sum_{j=1}^m \cos(b_j^\top(v - v'))\right)$$

Since the argument $\gamma(v)^\top \gamma(v') = h_\gamma(v - v')$ depends only on $v - v'$, the composed kernel $K_\gamma$ is *shift-invariant (stationary)*.

**Theorem (Tancik et al., 2020 — Stationarity).** If $h_{\mathrm{NTK}}$ is a function of the inner product and $\gamma$ is a Fourier feature map, then the composed kernel $K_\gamma(v, v') = h_{\mathrm{NTK}}(h_\gamma(v - v'))$ is shift-invariant. As a composition of an isotropic function with a shift-invariant kernel, it is itself shift-invariant, and by Bochner's theorem has a spectral density $\tilde{p}$ satisfying:

$$K_\gamma(v - v') = \int_{\mathbb{R}^d} \tilde{p}(\omega) e^{i\omega^\top(v - v')}\, d\omega$$

### 5.2 Stationarity and Bandwidth Control

The original NTK of a coordinate MLP is *not* stationary — it decays as inputs move apart, with a decay rate that depends strongly on frequency. The feature map $\gamma$ converts it into a stationary kernel whose bandwidth is governed by the scale of $B$.

**Key mechanism:** For the Gaussian random variant with $b_j \sim \mathcal{N}(0, \sigma^2 I)$, the feature map inner product approximates a Gaussian kernel of bandwidth $\propto \sigma^{-1}$:

$$h_\gamma(v - v') \approx \exp\!\left(-\frac{\sigma^2 \|v - v'\|^2}{2}\right)$$

Composing with $h_{\mathrm{NTK}}$ (which is an increasing function near $1$ for typical normalized inputs), the composed kernel has a *width* set by $\sigma$. Larger $\sigma$ makes the feature map project to higher frequencies and narrows the kernel in spatial domain, corresponding to a kernel that resolves finer detail.

The training dynamics (§3.1) with the composed kernel are:

$$\hat{y}^{(t)}_{\mathrm{test}} = K^\gamma_{\mathrm{test}} (K^\gamma)^{-1}(I - e^{-\eta K^\gamma t}) y$$

where $K^\gamma$ is the $n \times n$ composed kernel matrix on training points.

### 5.3 Eigenvalue Equalization and Kernel Preconditioning

The central claim of Tancik et al. (2020) is that the Fourier feature map *equalizes* the eigenvalues of the induced kernel across frequency modes, allowing all modes to converge at comparable rates.

**Heuristic argument for equalization.** The original coordinate MLP NTK has eigenvalues decaying as $\mu_l \sim l^{-(d+1)}$ for spherical harmonic degree $l$ (§3.3). After lifting via $\gamma$, the composed kernel $K_\gamma$ is a stationary kernel whose spectral density $\tilde{p}$ is shaped by the distribution of frequencies $b_j$.

For the random Fourier feature map with $b_j \sim \mathcal{N}(0, \sigma^2 I)$:

- The spectral density $\tilde{p}$ is supported over a ball of radius $\sim \sigma$ in frequency space.
- Modes with $\|\xi\| \lesssim \sigma$ all receive comparable spectral weight.
- The eigenvalue decay, instead of being $l^{-(d+1)}$, is approximately *flat* for $l \lesssim \sigma$ and drops sharply for $l \gg \sigma$.

This is the sense in which the Fourier feature map preconditions the kernel: it converts the strongly non-uniform eigenvalue distribution of the raw NTK into a more uniform one over the frequency range $[0, \sigma]$.

**Connection to kernel preconditioning.** Preconditioned gradient descent on a kernel regression problem with kernel $K$ and preconditioner $P$ is equivalent to vanilla gradient descent with kernel $P^{1/2} K P^{1/2}$. The Fourier feature map effectively implements a preconditioning that boosts high-frequency eigenvalues, making the optimization landscape more isotropic.

> [!WARNING] Equalization is not exact
> The eigenvalue equalization is a heuristic/asymptotic claim. The exact eigenvalues of $K_\gamma$ depend on both the architecture (depth, width, activation) and the frequency distribution. For very deep networks or structured frequency matrices $B$, the composed NTK may still exhibit non-trivial spectral decay. The practical benefit is well-established empirically; the precise theoretical characterization is an active research area.

> [!QUESTION] Exercise 4: Bandwidth-Quality Tradeoff
> *This exercise explores why there is an optimal $\sigma$ rather than simply "larger is better."*
>
> > **Prerequisites:** [[#5.2 Stationarity and Bandwidth Control|5.2 Stationarity and Bandwidth Control]], [[#5.3 Eigenvalue Equalization and Kernel Preconditioning|5.3 Eigenvalue Equalization and Kernel Preconditioning]]
>
> Consider fitting a 1D target $f^*(x) = \sin(2\pi k^* x)$ for some target frequency $k^*$ on $n$ training points equally spaced in $[0,1]$. The composed kernel $K_\gamma$ approximates a Gaussian kernel of bandwidth $\sigma^{-1}$ (i.e., the kernel has characteristic length scale $\sigma^{-1}$).
>
> (a) For what range of $\sigma$ does the kernel have enough frequency content to represent $f^*$?
>
> (b) If $\sigma \gg k^*$, the composed kernel is very narrow spatially. What happens to kernel regression with a narrow kernel and only $n$ training points?
>
> (c) Explain qualitatively why there is an optimal $\sigma \approx k^*$ (or some multiple thereof) for minimizing test error.

> [!TIP]- Solution to Exercise 4
> **Key insight:** The bandwidth $\sigma$ sets the frequency range of the kernel; choosing $\sigma$ too small underfits (can't represent $f^*$), choosing it too large overfits (kernel is too narrow to regularize).
>
> **Sketch:**
>
> (a) The composed kernel with Gaussian spectral density $\mathcal{N}(0, \sigma^2)$ has most of its power at frequencies $\|\omega\| \lesssim \sigma$. To represent $f^*(x) = \sin(2\pi k^* x)$, we need the kernel to have power at $k^*$, so $\sigma \gtrsim k^*$.
>
> (b) A very narrow kernel (large $\sigma$) places effectively zero weight on training points far from each test point. With $n$ finite training points, this means the kernel regression estimate near a test point far from training data is close to zero — the model interpolates locally but cannot generalize across gaps. This is textbook overfitting in the kernel sense: low bias on training data, high variance off of it.
>
> (c) The optimal $\sigma$ matches the "bandwidth" of the target: large enough to capture $k^*$ but small enough that the kernel is wide enough to regularize effectively across the $n$ training points. In practice, $\sigma \approx k^*$ up to constants determined by $n$ and the geometry of training data.

---

## 6. Convergence Rate Analysis 📈

### 6.1 Baseline Convergence in the NTK Regime

Recall the NTK training dynamics (§3.1): the $i$-th eigenvector component of the training residual decays as

$$[Q^\top r^{(t)}]_i = e^{-\eta \lambda_i t} [Q^\top r^{(0)}]_i$$

**Definition (Frequency Mode Convergence Rate).** For a target $f^*$ with dominant Fourier mode $\omega$, let $\lambda_\omega$ be the NTK eigenvalue associated with the eigenvector most aligned with frequency $\omega$. The *convergence rate* of mode $\omega$ is $\eta \lambda_\omega$, and the mode fits to accuracy $\epsilon$ at time:

$$t_\epsilon(\omega) = \frac{\log(1/\epsilon)}{\eta \lambda_\omega}$$

For a plain coordinate MLP, $\lambda_\omega \sim \|\omega\|^{-(d+1)}$, so:

$$\boxed{t_\epsilon(\omega) \sim \frac{\|\omega\|^{d+1} \log(1/\epsilon)}{\eta}}$$

**The convergence time grows as the $(d+1)$-th power of frequency.** In 3D ($d=3$), doubling the spatial frequency increases the required training time by $2^4 = 16\times$.

### 6.2 Convergence After Fourier Feature Lifting

After lifting $v \mapsto \gamma(v)$ with $B \sim \mathcal{N}(0, \sigma^2 I)$, the composed kernel $K_\gamma$ has an approximately flat spectrum for $\|\omega\| \lesssim \sigma$. In this regime, all eigenvalues satisfy $\tilde\lambda_\omega \approx c$ for some constant $c$ (the plateau value of the spectral density), and:

$$t_\epsilon(\omega) \approx \frac{\log(1/\epsilon)}{\eta c}, \quad \|\omega\| \lesssim \sigma$$

*Convergence is frequency-independent for all modes within the bandwidth.*

**Theorem (Tancik et al., 2020 — informal).** For an MLP trained on coordinates mapped through $\gamma(v) = [\cos(Bv), \sin(Bv)]$ with $B \sim \mathcal{N}(0, \sigma^2 I)$:

1. The composed NTK is stationary with bandwidth parameter $\sigma$.
2. For target frequencies $\|\omega\| \leq \sigma$, the convergence rate is $\Theta(\tilde\lambda_\text{max})$, independent of $\|\omega\|$.
3. For target frequencies $\|\omega\| > \sigma$, the mode lies outside the kernel's effective support and cannot be learned.

**Quantitative comparison.** Consider frequency $\|\omega\| = L$ in $d$ dimensions:

| Method | Convergence time for mode $L$ |
|--------|-------------------------------|
| Plain MLP (NTK regime) | $\sim L^{d+1}$ |
| Fourier features ($\sigma \geq L$) | $\sim 1$ (constant in $L$) |

**The Fourier feature map converts polynomial-in-frequency convergence time to constant convergence time, for all modes within the kernel bandwidth.**

> [!WARNING] NTK regime caveat
> *These results hold in the NTK regime (infinite width, small learning rate, fixed kernel). For finite-width networks, the kernel changes during training, and the spectral equalization argument is approximate. The practical benefit observed in coordinate-based MLP applications suggests the argument is robust, but the formal theory for finite networks is less complete.*

> [!QUESTION] Exercise 5: Optimal Bandwidth Selection
> *This exercise derives a formal criterion for choosing $\sigma$ given knowledge of the target signal.*
>
> > **Prerequisites:** [[#6.1 Baseline Convergence in the NTK Regime|6.1 Baseline Convergence in the NTK Regime]], [[#6.2 Convergence After Fourier Feature Lifting|6.2 Convergence After Fourier Feature Lifting]]
>
> Suppose the target function $f^* : [0,1]^d \to \mathbb{R}$ has a known Fourier spectrum: $|\hat{f}^*(\omega)|$ is non-negligible only for $\|\omega\| \leq \omega_{\max}$.
>
> (a) What is the minimum $\sigma$ needed to ensure all non-negligible modes of $f^*$ are within the kernel's bandwidth?
>
> (b) Given a fixed training budget of $T$ gradient steps, and using both the plain MLP formula ($t \sim \|\omega\|^{d+1}$) and the Fourier features formula ($t \sim 1$), derive the maximum learnable frequency $\omega^*$ for each method.
>
> (c) For $d=3$ and $T = 10^4$ steps, compute $\omega^*_{\mathrm{plain}}$ and $\omega^*_{\mathrm{FF}}(\sigma = \omega_{\max})$.

> [!TIP]- Solution to Exercise 5
> **Key insight:** The Fourier features method changes the bottleneck from frequency-dependent convergence to bandwidth selection, making the "maximum learnable frequency" simply $\sigma$.
>
> **Sketch:**
>
> (a) Set $\sigma = \omega_{\max}$ — this ensures all non-negligible modes satisfy $\|\omega\| \leq \sigma$ and thus converge at the flat rate.
>
> (b) For the plain MLP: a mode $\omega$ converges within budget $T$ if $T \gtrsim \|\omega\|^{d+1}$, so $\omega^*_{\mathrm{plain}} = T^{1/(d+1)}$. For Fourier features: all modes $\|\omega\| \leq \sigma$ converge in $O(1)$ steps (where the constant depends on the plateau eigenvalue), so $\omega^*_{\mathrm{FF}} = \sigma$ regardless of $T$ (as long as $T$ is large enough to clear the constant).
>
> (c) $d=3$: $\omega^*_{\mathrm{plain}} = (10^4)^{1/4} = 10$. With Fourier features and $\sigma = \omega_{\max}$: $\omega^*_{\mathrm{FF}} = \omega_{\max}$, which can be set arbitrarily (within generalization constraints). If $\omega_{\max} = 100$, the Fourier features method can in principle learn $10\times$ higher frequencies within the same budget.

---

## 7. Applications 🖼️

### 7.1 NeRF Positional Encodings

*Neural Radiance Fields* (NeRF; Mildenhall et al., 2020) represent a 3D scene as a continuous volumetric function $F_\theta : (\mathbf{x}, \mathbf{d}) \to (\mathbf{c}, \sigma)$ mapping 3D coordinates $\mathbf{x} \in \mathbb{R}^3$ and viewing direction $\mathbf{d} \in \mathbb{S}^2$ to RGB color $\mathbf{c}$ and volume density $\sigma$.

Without positional encoding, an MLP trained on $(\mathbf{x}, \mathbf{d})$ directly suffers spectral bias and produces blurry, low-frequency renderings. The solution (motivated by, and empirically validated prior to, the Tancik et al. theory) is the NeRF positional encoding:

$$\gamma(p) = \left(\sin(2^0 \pi p),\, \cos(2^0 \pi p),\, \sin(2^1 \pi p),\, \cos(2^1 \pi p),\, \ldots,\, \sin(2^{L-1} \pi p),\, \cos(2^{L-1} \pi p)\right)$$

applied component-wise to each coordinate. With $L = 10$ for position and $L = 4$ for direction, NeRF can recover fine textures, specular highlights, and thin geometric details that a plain MLP cannot.

The Tancik et al. analysis provides the retrospective explanation: the log-spaced encoding ensures the composed kernel has substantial spectral support out to frequency $2^{L-1}$, with the geometric progression providing roughly uniform (in log-frequency) coverage.

### 7.2 SIREN: Sinusoidal Activation Networks

A distinct approach to the spectral bias problem is *SIREN* (Sitzmann et al., 2020): replace all ReLU activations with $\sin(\omega_0 \cdot)$ for a hyperparameter $\omega_0 > 0$.

**Mechanism.** A sine-activation network computes:

$$f_\theta(x) = W_L \sin(\omega_0 (W_{L-1} \sin(\omega_0 (\cdots W_1 \sin(\omega_0 W_0 x + b_0) \cdots)) + b_{L-1})) + b_L$$

Unlike piecewise-linear networks, sine activations are infinitely differentiable and have Fourier transforms supported *everywhere* — $\hat{\sigma}(\xi) = \delta(\xi - \omega_0) / 2 + \delta(\xi + \omega_0) / 2$. Compositions create superpositions of sinusoids at frequencies $\omega_0^k$ for $k = 1, \ldots, L$, enabling the network to represent high-frequency functions without needing fine-tuned weight cooperation.

*Surprisingly,* the Fourier-theoretic argument is complementary to the Fourier features argument: SIREN changes the *architecture* to natively produce high-frequency outputs, whereas Fourier features change the *input representation* to reshape the NTK.

**Practical considerations.** SIREN requires a careful initialization scheme (weights initialized as $\mathcal{U}(-\sqrt{6/n}, \sqrt{6/n})$ with $\omega_0$ chosen to control frequency support) and is sensitive to these choices. Fourier feature maps, by contrast, can be bolted onto any existing MLP architecture.

### 7.3 Occupancy Networks and Other Implicit Representations

*Occupancy networks* (Mescheder et al., 2019) represent 3D geometry as $f_\theta : \mathbb{R}^3 \to [0,1]$ predicting whether a point is inside the object. Without positional encoding, occupancy networks learn coarse geometry (large-scale shape) but miss fine surface details (cracks, ridges, engravings).

Fourier feature maps (or learned embeddings) applied to the input coordinates of occupancy networks enable representation of fine-grained geometry — thin structures, sharp creases — by elevating the NTK's effective bandwidth.

**Empirical comparison** across implicit representation tasks (Tancik et al., 2020):

| Task | Input dim | Plain MLP PSNR | + Fourier Features PSNR | Gain |
|------|-----------|---------------|------------------------|------|
| 2D Image regression | 2 | ~18 dB | ~35 dB | +17 dB |
| CT reconstruction | 2 | ~23 dB | ~40 dB | +17 dB |
| 3D shape (SDF) | 3 | ~22 dB | ~36 dB | +14 dB |
| NeRF-style 3D | 5 (+ dir) | blurry | sharp | qualitative |

> [!INFO] Why Low-Dimensional Inputs Matter
> The spectral bias problem is most acute for low-dimensional inputs ($d \leq 5$). For high-dimensional inputs (e.g., images as features), networks learn complex functions more easily because the NTK eigenvalue decay $l^{-(d+1)}$ is suppressed by the large $d$ — or more precisely, the "frequency" concept is less cleanly defined in high dimensions. Fourier feature maps are therefore most impactful precisely in the coordinate-based / implicit representation setting.

> [!EXAMPLE]- Example: NeRF Encoding Frequencies
> The standard NeRF encoding with $L = 10$ for position encodes frequencies $2^0 = 1, 2^1 = 2, \ldots, 2^9 = 512$ (in units of $\pi$). The composed kernel thus has spectral support from 1 to 512 in each coordinate direction. Without encoding, a 5-layer MLP with 256 hidden units on 3D coordinates would typically resolve features no finer than roughly length-scale 0.1 (frequency $\sim 10$) — matching $\omega^*_{\mathrm{plain}} \approx T^{1/4}$ for reasonable $T$. The encoding lifts this ceiling by over $50\times$.

---

## 8. References

| Reference Name | Brief Summary | Link to Reference |
|----------------|--------------|-------------------|
| Rahaman et al. (2019), "On the Spectral Bias of Neural Networks" | Establishes and formalizes spectral bias in deep ReLU networks via Fourier analysis; provides empirical evidence and the $O(\|\xi\|^{-\Delta})$ convergence rate argument | [arXiv:1806.08734](https://arxiv.org/abs/1806.08734) |
| Tancik et al. (2020), "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains" | Shows that lifting inputs via $\gamma(v)=[\cos(Bv),\sin(Bv)]$ converts the NTK to a stationary kernel with tunable bandwidth; provides NTK-based convergence analysis and empirical results on implicit representations | [arXiv:2006.10739](https://arxiv.org/abs/2006.10739) |
| Rahimi & Recht (2007), "Random Features for Large-Scale Kernel Machines" | Proposes random Fourier features based on Bochner's theorem to approximate shift-invariant kernels in $O(m)$ inner products instead of $O(n^2)$; foundational paper for the random feature approximation literature | [NIPS 2007 Proceedings](https://proceedings.neurips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html) |
| Jacot et al. (2018), "Neural Tangent Kernel: Convergence and Generalization in Neural Networks" | Introduces the NTK formalism; proves that infinite-width networks trained with gradient descent evolve as kernel regression; the theoretical foundation for §3 | [arXiv:1806.07572](https://arxiv.org/abs/1806.07572) |
| Basri et al. (2019/2020), "Frequency Bias in Neural Networks for Input of Non-Uniform Density" | Formally proves spectral bias via spherical harmonic decomposition of the NTK; establishes $\mu_l \sim l^{-(d+1)}$ eigenvalue decay; extends to non-uniform input distributions | [arXiv:2003.04560](https://arxiv.org/abs/2003.04560) |
| Bietti & Mairal (2019), "On the Inductive Bias of Neural Tangent Kernels" | Companion work to Basri et al.; establishes NTK eigenvalue decay via RKHS analysis; proves low-frequency eigenfunctions dominate for uniform sphere distributions | [arXiv:1905.12173](https://arxiv.org/abs/1905.12173) |
| Sitzmann et al. (2020), "Implicit Neural Representations with Periodic Activation Functions" (SIREN) | Proposes sine activations as an architectural alternative to Fourier feature maps; achieves high-frequency representation by changing activation rather than input encoding | [arXiv:2006.09661](https://arxiv.org/abs/2006.09661) |
| Mildenhall et al. (2020), "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" | Introduces Neural Radiance Fields with positional encoding; the applied context motivating the Tancik et al. theory | [arXiv:2003.08934](https://arxiv.org/abs/2003.08934) |
| Xu et al. (2019), "Frequency Principle: Fourier Analysis Sheds Light on Implicit Regularization of Gradient Descent" | Provides empirical and theoretical evidence for the frequency principle from a physics-inspired perspective; introduces the linear F-principle model | [arXiv:1901.06523](https://arxiv.org/abs/1901.06523) |
| Zhang et al. (2021), "Overview Frequency Principle/Spectral Bias in Deep Learning" | Comprehensive survey of theoretical and empirical results on the frequency principle; includes the general residual evolution equation $\partial_t \hat{u} = -\gamma(\xi)^2 \hat{u}$ | [arXiv:2201.07395](https://arxiv.org/abs/2201.07395) |
