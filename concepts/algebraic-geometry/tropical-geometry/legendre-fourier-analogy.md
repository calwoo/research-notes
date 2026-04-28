# The Legendre Transform as the Tropical Fourier Transform

## Table of Contents

- [[#1. The Tropical Semiring|1. The Tropical Semiring]]
  - [[#1.1 Definition and Motivation|1.1 Definition and Motivation]]
  - [[#1.2 Maslov's Dequantization|1.2 Maslov's Dequantization]]
- [[#2. Tropicalizing the Fourier Transform|2. Tropicalizing the Fourier Transform]]
  - [[#2.1 Classical Laplace Transform|2.1 Classical Laplace Transform]]
  - [[#2.2 The Tropical Substitution|2.2 The Tropical Substitution]]
  - [[#2.3 The Legendre–Fenchel Conjugate|2.3 The Legendre–Fenchel Conjugate]]
- [[#3. The Convolution Theorem|3. The Convolution Theorem]]
  - [[#3.1 Inf-Convolution as Tropical Convolution|3.1 Inf-Convolution as Tropical Convolution]]
  - [[#3.2 The Tropical Convolution Theorem|3.2 The Tropical Convolution Theorem]]
- [[#4. The Inversion Theorem|4. The Inversion Theorem]]
  - [[#4.1 Fourier Inversion and Its Tropical Analog|4.1 Fourier Inversion and Its Tropical Analog]]
  - [[#4.2 The Biconjugate Theorem|4.2 The Biconjugate Theorem]]
- [[#5. The Full Dictionary|5. The Full Dictionary]]
- [[#6. Connections and Applications|6. Connections and Applications]]
  - [[#6.1 Legendre Transform in Physics and Thermodynamics|6.1 Legendre Transform in Physics and Thermodynamics]]
  - [[#6.2 Tropical Characters and Affine Functions|6.2 Tropical Characters and Affine Functions]]
  - [[#6.3 Connection to Neuroalgebraic Geometry|6.3 Connection to Neuroalgebraic Geometry]]
- [[#7. References|7. References]]

---

## 1. The Tropical Semiring 📐

### 1.1 Definition and Motivation

**Definition (Tropical Semiring).** The *tropical semiring* (or *max-plus semiring*) is the set $\mathbb{T} = \mathbb{R} \cup \{-\infty\}$ equipped with:
$$a \oplus b = \max(a, b), \qquad a \odot b = a + b.$$

The element $-\infty$ is the additive identity ($a \oplus (-\infty) = a$) and $0$ is the multiplicative identity ($a \odot 0 = a$). The min-plus semiring $(\mathbb{R} \cup \{+\infty\}, \min, +)$ is the dual; the two are related by negation.

> [!NOTE] Why "Tropical"?
> The name honors the Brazilian mathematician Imre Simon, who introduced the semiring in the context of combinatorics. It was dubbed "tropical" by French mathematicians.

A *tropical polynomial* in one variable is a max-plus expression:

$$p(x) = \max(a_0, a_1 + x, a_2 + 2x, \ldots, a_d + dx) = \bigoplus_{k=0}^d (a_k \odot x^{\odot k}),$$

where $x^{\odot k} = kx$. This is a **piecewise-linear convex function** of $x$. Its *tropical roots* are the breakpoints of this piecewise-linear function, counted with multiplicity.

### 1.2 Maslov's Dequantization

The tropical semiring arises naturally as a *classical limit*. Define the deformed semiring $(\mathbb{R}, \oplus_\hbar, \odot_\hbar)$ by:

$$a \oplus_\hbar b = \hbar \log(e^{a/\hbar} + e^{b/\hbar}), \qquad a \odot_\hbar b = a + b.$$

At $\hbar = 1$ (with $e^a$ standing in for $a$), this recovers ordinary addition and multiplication. As $\hbar \to 0$:

$$a \oplus_\hbar b = \hbar \log\bigl(e^{a/\hbar}(1 + e^{(b-a)/\hbar})\bigr) \xrightarrow{\hbar \to 0} \max(a, b).$$

**The tropical semiring is the $\hbar \to 0$ (classical) limit of the usual semiring under the logarithmic change of variables $a \mapsto \hbar \log a$.** This is *Maslov dequantization*: $\hbar$ plays the role of Planck's constant, and the tropical world is the "classical mechanics" limit of "quantum" arithmetic.

> [!INFO] Amoebas and the Passage to Tropical
> For an algebraic variety $V \subset (\mathbb{C}^*)^n$, the *amoeba* $\mathcal{A}(V) = \{(\log|z_1|, \ldots, \log|z_n|) : z \in V\} \subset \mathbb{R}^n$ is a "tentacled" subset of $\mathbb{R}^n$. As $\hbar \to 0$, the rescaled amoeba $\hbar \cdot \mathcal{A}(V)$ converges to a polyhedral complex — the *tropical variety* associated to $V$. Tropical geometry is algebraic geometry in this classical limit.

---

> [!QUESTION] Exercise 1: Tropical Polynomial Roots
> *This exercise builds intuition for tropical polynomials as piecewise-linear functions.*
>
> > **Prerequisites:** [[#1.1 Definition and Motivation|1.1 Definition and Motivation]]
>
> Consider the tropical polynomial $p(x) = \max(0, x, 2x - 1)$ over $\mathbb{T}$. (a) Sketch $p$ as a function $\mathbb{R} \to \mathbb{R}$. (b) Find the breakpoints (tropical roots) and their multiplicities. (c) Write $p$ as a product of tropical linear factors $\max(0, x - r_i)$.

> [!TIP]- Solution to Exercise 1
> **Key insight:** Breakpoints occur where two branches of the max achieve the same value; multiplicity is the difference in slopes across the breakpoint.
>
> **Sketch:** (a) Three linear pieces: $p(x) = 0$ for $x \leq 0$, $p(x) = x$ for $0 \leq x \leq 1$, $p(x) = 2x-1$ for $x \geq 1$. (b) Breakpoints at $x = 0$ (slope jump $0 \to 1$, multiplicity 1) and $x = 1$ (slope jump $1 \to 2$, multiplicity 1). (c) $p(x) = \max(0, x) + \max(0, x-1) = \max(0, x) \odot \max(0, x-1)$, confirming two roots $r_1 = 0$, $r_2 = 1$.

---

## 2. Tropicalizing the Fourier Transform 📐

### 2.1 Classical Laplace Transform

The *Laplace transform* (real-variable version of the Fourier transform) is:

$$\mathcal{L}[f](\xi) = \int_{\mathbb{R}^n} f(x)\, e^{x \cdot \xi}\, dx,$$

for functions $f: \mathbb{R}^n \to \mathbb{R}_{\geq 0}$ (or more generally, $f$ in the appropriate $L^1$ class). The Fourier transform $\hat{f}(\xi) = \int f(x) e^{ix\cdot\xi} dx$ is obtained by replacing $\xi$ with $i\xi$; the two are related by Wick rotation.

The key structural property: the Laplace transform converts **convolution** into **pointwise multiplication**:

$$\mathcal{L}[f * g] = \mathcal{L}[f] \cdot \mathcal{L}[g].$$

### 2.2 The Tropical Substitution

To tropicalize the Laplace transform, make the substitution dictated by Maslov dequantization:
- Replace $f(x)$ with $e^{g(x)/\hbar}$ (writing $f$ in logarithmic coordinates, $g = \hbar \log f$)
- Replace $e^{x \cdot \xi}$ with $e^{x \cdot \xi / \hbar}$ (tropical multiplication $\odot$)
- Replace $\int$ (classical sum, $\oplus_{\hbar=1}$) with $\sup$ (tropical sum $\oplus_{\hbar=0}$)

The transform becomes:

$$\mathcal{L}_\hbar[g](\xi) = \hbar \log \int e^{g(x)/\hbar} e^{x \cdot \xi / \hbar}\, dx = \hbar \log \int e^{(g(x) + x \cdot \xi)/\hbar}\, dx.$$

By **Laplace's method** (steepest descent), as $\hbar \to 0$:

$$\mathcal{L}_\hbar[g](\xi) \xrightarrow{\hbar \to 0} \sup_{x \in \mathbb{R}^n}\bigl(g(x) + x \cdot \xi\bigr).$$

**The Legendre transform is the $\hbar \to 0$ limit of the Laplace transform in logarithmic coordinates.**

> [!WARNING] Fourier vs. Laplace
> The direct tropical analog is the **Laplace** transform, not the oscillatory Fourier transform. The Fourier transform $\int f e^{i\xi x} dx$ involves imaginary exponents; stationary phase gives Morse-theoretic data (critical points, indices), not the Legendre transform. The identification "Legendre = tropical Fourier" should be understood as "Legendre = tropical Laplace," with Laplace and Fourier related by $\xi \mapsto i\xi$.

### 2.3 The Legendre–Fenchel Conjugate

**Definition (Legendre–Fenchel Conjugate).** For $f: \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$, the *convex conjugate* (or *Legendre–Fenchel transform*) is:

$$f^*(\xi) = \sup_{x \in \mathbb{R}^n}\bigl(\langle x, \xi\rangle - f(x)\bigr).$$

This is the tropical Laplace transform of $-f$, or equivalently the tropical Laplace transform of $f$ in the min-plus convention. Both sign conventions appear in the literature; the key object is the supremum.

> [!EXAMPLE] Legendre Transform of a Quadratic
> For $f(x) = \frac{1}{2}\langle Ax, x\rangle$ with $A \succ 0$: the supremum $\sup_x(\langle x, \xi\rangle - \frac{1}{2}\langle Ax, x\rangle)$ is achieved at $x^* = A^{-1}\xi$, giving $f^*(\xi) = \frac{1}{2}\langle A^{-1}\xi, \xi\rangle$. So the Legendre transform of a quadratic form with matrix $A$ is a quadratic form with matrix $A^{-1}$ — dual in exactly the same sense as the Fourier transform of a Gaussian with variance $\sigma^2$ is a Gaussian with variance $1/\sigma^2$.

This Gaussian duality is the clearest finite-dimensional instance of the Fourier–Legendre analogy: $\mathcal{F}[e^{-\|x\|^2/2\sigma^2}] \propto e^{-\sigma^2\|\xi\|^2/2}$ mirrors $\left(\frac{1}{2\sigma^2}\|x\|^2\right)^* = \frac{\sigma^2}{2}\|\xi\|^2$.

---

> [!QUESTION] Exercise 2: Tropical Laplace of an Indicator
> *This exercise connects indicator functions to support functions, a classical duality in convex geometry.*
>
> > **Prerequisites:** [[#2.3 The Legendre–Fenchel Conjugate|2.3 The Legendre–Fenchel Conjugate]]
>
> Let $C \subset \mathbb{R}^n$ be a convex set and $f = \delta_C$ the indicator function ($\delta_C(x) = 0$ if $x \in C$, $+\infty$ otherwise). Show that $f^*(\xi) = \sup_{x \in C} \langle x, \xi\rangle$ — the *support function* of $C$ — and interpret this as the tropical Laplace transform of a "tropical indicator."

> [!TIP]- Solution to Exercise 2
> **Key insight:** The indicator function is the tropical analog of a characteristic function $\mathbf{1}_C$ (since $0$ is the multiplicative identity in tropical arithmetic and $-\infty$ is the additive identity).
>
> **Sketch:** $f^*(\xi) = \sup_x(\langle x,\xi\rangle - \delta_C(x)) = \sup_{x \in C}\langle x, \xi\rangle$ since $\delta_C(x) = +\infty$ kills all $x \notin C$. This is the support function $h_C(\xi)$. Tropically: $\delta_C$ is the "tropical indicator" (value $0$ on $C$, $+\infty$ off $C$), and its tropical Laplace transform is $\sup_{x \in C}\langle x, \xi\rangle$ — obtained by "integrating" (taking the tropical sum) $\langle x, \xi\rangle \odot \delta_C(x) = \langle x, \xi\rangle + 0$ over $x \in C$.

---

## 3. The Convolution Theorem 🔑

### 3.1 Inf-Convolution as Tropical Convolution

**Definition (Inf-Convolution).** For $f, g: \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$, the *inf-convolution* is:

$$(f \,\square\, g)(x) = \inf_{y \in \mathbb{R}^n}\bigl(f(y) + g(x - y)\bigr).$$

This is the **tropical analog of convolution**: replace integration $\int$ with $\inf$ and multiplication $f(y) g(x-y)$ with addition $f(y) + g(x-y)$ — exactly the $(\min, +)$ substitution.

> [!EXAMPLE] Geometric Interpretation
> If $f = \delta_A$ and $g = \delta_B$ are indicator functions of convex sets $A, B$, then $(f \,\square\, g)(x) = 0$ iff $x \in A + B = \{a + b : a \in A, b \in B\}$, and $+\infty$ otherwise. So inf-convolution of indicators produces the indicator of the *Minkowski sum* — the tropical analog of the fact that convolution of indicator functions produces a function supported on the sumset.

### 3.2 The Tropical Convolution Theorem

**Theorem (Tropical Convolution Theorem).** *For any $f, g: \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$:*

$$\boxed{(f \,\square\, g)^* = f^* + g^*.}$$

*The Legendre–Fenchel transform converts inf-convolution into pointwise addition.*

**Proof.** Compute directly:
$$
(f\,\square\, g)^*(\xi) = \sup_x\Bigl(\langle x,\xi\rangle - \inf_y\bigl(f(y) + g(x-y)\bigr)\Bigr)
= \sup_x \sup_y\Bigl(\langle x,\xi\rangle - f(y) - g(x-y)\Bigr).
$$
Substitute $z = x - y$, so $x = y + z$:
$$
= \sup_{y,z}\Bigl(\langle y,\xi\rangle + \langle z,\xi\rangle - f(y) - g(z)\Bigr)
= \sup_y\bigl(\langle y,\xi\rangle - f(y)\bigr) + \sup_z\bigl(\langle z,\xi\rangle - g(z)\bigr)
= f^*(\xi) + g^*(\xi). \quad \square
$$

Comparing with the classical convolution theorem $\mathcal{F}[f * g] = \mathcal{F}[f] \cdot \mathcal{F}[g]$:

| Classical | Tropical |
|-----------|---------|
| Convolution $f * g$ | Inf-convolution $f \,\square\, g$ |
| Pointwise multiplication $\hat{f} \cdot \hat{g}$ | Pointwise addition $f^* + g^*$ |
| $(+, \times)$ | $(\inf, +)$ = min-plus tropical |

**The proof is structurally identical** to the proof of the classical convolution theorem: both use Fubini to separate variables after a substitution.

---

> [!QUESTION] Exercise 3: Tropical Convolution Theorem for Quadratics
> *This exercise verifies the theorem in a case where both sides can be computed explicitly.*
>
> > **Prerequisites:** [[#3.2 The Tropical Convolution Theorem|3.2 The Tropical Convolution Theorem]], [[#2.3 The Legendre–Fenchel Conjugate|2.3 The Legendre–Fenchel Conjugate]]
>
> Let $f(x) = \frac{1}{2a}\|x\|^2$ and $g(x) = \frac{1}{2b}\|x\|^2$ for $a, b > 0$. (a) Compute $f^*$ and $g^*$. (b) Compute $(f \,\square\, g)$ directly by minimizing over $y$. (c) Compute $(f \,\square\, g)^*$ and verify it equals $f^* + g^*$.

> [!TIP]- Solution to Exercise 3
> **Key insight:** Inf-convolution of quadratics is a quadratic with the harmonic mean of the curvatures; the theorem makes this transparent.
>
> **Sketch:** (a) $f^*(\xi) = \frac{a}{2}\|\xi\|^2$, $g^*(\xi) = \frac{b}{2}\|\xi\|^2$. (b) $(f \,\square\, g)(x) = \inf_y \frac{1}{2a}\|y\|^2 + \frac{1}{2b}\|x-y\|^2$. Optimizing over $y$: $y^* = \frac{a}{a+b}x$, giving $(f\,\square\, g)(x) = \frac{1}{2(a+b)}\|x\|^2$. (c) $(f\,\square\, g)^*(\xi) = \frac{a+b}{2}\|\xi\|^2 = \frac{a}{2}\|\xi\|^2 + \frac{b}{2}\|\xi\|^2 = f^*(\xi) + g^*(\xi)$. ✓ The curvatures add under Legendre, just as variances add under Fourier.

---

## 4. The Inversion Theorem 📐

### 4.1 Fourier Inversion and Its Tropical Analog

The classical Fourier inversion theorem states: if $f \in L^1(\mathbb{R}^n)$ and $\hat{f} \in L^1(\mathbb{R}^n)$, then $f = \mathcal{F}^{-1}[\hat{f}]$. In the Laplace/Fourier context, the condition on $f$ (square-integrability, $L^1$, etc.) controls when inversion is valid.

The tropical analog asks: for which $f$ does $f^{**} = f$? (Applying the Legendre transform twice.)

### 4.2 The Biconjugate Theorem

**Theorem (Biconjugate / Fenchel–Moreau).** *For any $f: \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$ that is not identically $+\infty$:*

$$f^{**} = \overline{\mathrm{co}}\, f,$$

*the* lower semicontinuous convex hull *of $f$ (the largest closed convex function $\leq f$). In particular, $f^{**} = f$ if and only if $f$ is closed (lower semicontinuous) and convex.*

This is the **tropical Fourier inversion theorem**: you can recover $f$ from $f^*$ exactly when $f$ is convex, just as you can recover $f$ from $\hat{f}$ when $f \in L^2$ (or $L^1$). The convexity condition is the tropical analog of the $L^p$ condition.

> [!EXAMPLE] Non-Convex Function Loses Information
> Let $f: \mathbb{R} \to \mathbb{R}$ with $f(0) = 1$, $f(x) = |x|$ for $x \neq 0$ (a non-convex function dipping below the chord). Then $f^{**}(0) = 0 \neq 1 = f(0)$: the double conjugate fills in the "dip," replacing $f$ with its convex hull. Information about the non-convex part is lost under $(\cdot)^*$, just as a non-$L^2$ function is not recoverable from its Fourier transform.

The structure is exact:

| Classical Fourier | Tropical Legendre |
|-------------------|------------------|
| $f \in L^2$ $\Rightarrow$ $f = \mathcal{F}^{-1}[\hat{f}]$ | $f$ closed convex $\Rightarrow$ $f = (f^*)^*$ |
| Non-$L^2$ $f$: inversion fails | Non-convex $f$: $f^{**} \neq f$ |
| Plancherel: $\|\hat{f}\|_{L^2} = \|f\|_{L^2}$ | Analogue: Legendre is an isometry of convex functions |

---

> [!QUESTION] Exercise 4: Double Legendre of a Non-Convex Function
> *This exercise demonstrates concretely that the biconjugate is the convex hull.*
>
> > **Prerequisites:** [[#4.2 The Biconjugate Theorem|4.2 The Biconjugate Theorem]]
>
> Let $f: \mathbb{R} \to \mathbb{R}$ be defined by $f(x) = x^4 - 2x^2$ (a double-well potential with local maxima and minima). (a) Find the convex hull $\overline{\mathrm{co}}\, f$. (b) Without computing $f^*$ explicitly, state what $f^{**}$ must be by the biconjugate theorem. (c) On the interval where $f^{**} \neq f$, what does the Legendre transform "forget" about $f$?

> [!TIP]- Solution to Exercise 4
> **Key insight:** The convex hull connects the two global minima with a flat segment; the double Legendre recovers only this convex envelope.
>
> **Sketch:** (a) $f(x) = x^4 - 2x^2$ has global minima at $x = \pm 1$ with $f(\pm 1) = -1$ and a local max at $x=0$ with $f(0) = 0$. The convex hull is $\overline{\mathrm{co}}\,f(x) = -1$ for $x \in [-1,1]$ (the "flat bottom" connecting the two minima) and $f(x)$ for $|x| > 1$. (b) $f^{**} = \overline{\mathrm{co}}\,f$. (c) On $(-1, 1)$, the Legendre transform forgets the double-well structure (the local max at 0 and the specific shape of the wells) — it retains only the lowest achievable value and the convex envelope.

---

## 5. The Full Dictionary 🔑

The complete correspondence between classical harmonic analysis and tropical convex analysis:

| Classical | Tropical ($\min$-plus or $\max$-plus) |
|-----------|--------------------------------------|
| Semiring $(\mathbb{R}, +, \times)$ | Semiring $(\mathbb{R}, \max, +)$ or $(\mathbb{R}, \min, +)$ |
| Laplace transform $\int f(x) e^{x\cdot\xi} dx$ | Legendre conjugate $\sup_x(f(x) + x\cdot\xi)$ |
| Fourier transform (via Wick rotation) | Legendre conjugate (same, up to sign) |
| Convolution $(f * g)(x) = \int f(y)g(x-y) dy$ | Inf-convolution $(f \,\square\, g)(x) = \inf_y(f(y) + g(x-y))$ |
| Convolution theorem $\widehat{f*g} = \hat{f}\cdot\hat{g}$ | $(f \,\square\, g)^* = f^* + g^*$ |
| Inversion: $f = \mathcal{F}^{-1}[\hat{f}]$ for $f \in L^2$ | Inversion: $f^{**} = f$ for $f$ closed convex |
| Characters $x \mapsto e^{i\langle x,\xi\rangle}$ | Tropical characters $x \mapsto \langle x, \xi\rangle$ (affine functions) |
| Gaussian self-duality: $\mathcal{F}[e^{-\|x\|^2/2\sigma^2}] \propto e^{-\sigma^2\|\xi\|^2/2}$ | Quadratic self-duality: $(\frac{1}{2\sigma^2}\|x\|^2)^* = \frac{\sigma^2}{2}\|\xi\|^2$ |
| Plancherel: $\|f\|_{L^2} = \|\hat{f}\|_{L^2}$ | Legendre is an isometry on convex functions |
| $\hbar = 1$ in $a \oplus_\hbar b = \hbar\log(e^{a/\hbar}+e^{b/\hbar})$ | $\hbar \to 0$ limit |

The tropical characters $x \mapsto \langle x, \xi\rangle$ (linear/affine functions) play the same role in tropical harmonic analysis that complex exponentials $x \mapsto e^{i\langle x, \xi\rangle}$ play in classical harmonic analysis: they are the "eigenfunctions" of tropical convolution operators.

---

## 6. Connections and Applications 💡

### 6.1 Legendre Transform in Physics and Thermodynamics

The Legendre transform is ubiquitous in physics precisely because physics operates in the $\hbar \to 0$ limit. The correspondences:

- **Thermodynamics:** free energy $F(T) = E - TS$ is the Legendre transform of entropy $S(E)$ as a function of energy. The convolution theorem explains why free energies of independent subsystems add: their state densities convolve, so their free energies (Legendre transforms) add.
- **Classical mechanics:** the Hamiltonian $H(q, p)$ is the Legendre transform of the Lagrangian $L(q, \dot{q})$ in the velocity variable. This is the Maslov dequantization of the path integral (quantum mechanics, $\hbar > 0$) to Hamilton–Jacobi theory (classical mechanics, $\hbar \to 0$).
- **Large deviations:** the *rate function* $I(\xi)$ of a large deviations principle is the Legendre transform of the *log-moment generating function* $\Lambda(\lambda) = \log \mathbb{E}[e^{\lambda X}]$. This is exactly the tropical Laplace transform structure.

> [!INFO] Singular Learning Theory Connection
> In [[concepts/algebraic-geometry/neuroalgebraic-geometry/singular-learning-theory|Singular Learning Theory]], the Bayesian partition function $Z_n(\beta) = \int p(x^n | w)^\beta \varphi(w)\, dw$ is a Laplace-type transform (at inverse temperature $\beta$). Its $\beta \to \infty$ limit (zero-temperature) concentrates on the optimal parameter set $W_0 = \{K(w) = 0\}$, and the free energy $F_n = -\log Z_n(1)$ satisfies the asymptotic $F_n \sim \lambda \log n$ — where $\lambda$ is the RLCT. The tropical ($\beta \to \infty$) limit of the Bayesian free energy is exactly the Legendre transform of the KL divergence in the large-$n$ regime.

### 6.2 Tropical Characters and Affine Functions

In classical Fourier analysis, the characters $\chi_\xi(x) = e^{i\langle x, \xi\rangle}$ satisfy:
- They are **group homomorphisms**: $\chi_\xi(x+y) = \chi_\xi(x)\chi_\xi(y)$
- They are **eigenfunctions** of translation: $\tau_a \chi_\xi = e^{i\langle a,\xi\rangle} \chi_\xi$

The tropical characters $\ell_\xi(x) = \langle x, \xi\rangle$ (affine functions) satisfy:
- **Tropical homomorphism**: $\ell_\xi(x \odot y) = \ell_\xi(x) + \ell_\xi(y)$... but $x \odot y = x + y$ classically, so this just says linearity.
- More precisely: $\ell_\xi$ is a **tropical monomial** (monomial = linear function in the $\log$-coordinate world), and tropical monomials are the building blocks of tropical polynomials exactly as complex exponentials are the building blocks of classical functions via Fourier expansion.

The *tropical Fourier expansion* of a convex piecewise-linear function $f$ decomposes it as a max of affine functions:

$$f(x) = \max_{\xi \in S}\bigl(\langle x, \xi\rangle + c(\xi)\bigr),$$

where $S$ is the set of slopes (the *subdifferential image* of $f$) and $c(\xi) = -f^*(\xi)$ is determined by $f^*$. This is the tropical analog of the Fourier series.

### 6.3 Connection to Neuroalgebraic Geometry

ReLU neural networks compute **tropical polynomials**. A single ReLU unit computes $\max(0, w \cdot x + b)$, which is a tropical polynomial in $(x, w, b)$-space. A deep ReLU network computes iterated maxima and sums of linear functions — a composition of tropical polynomials — making its input-output map a *piecewise-linear convex function* on each activation region.

The consequence: the expressivity analysis of deep ReLU networks is a problem in **tropical geometry**. The number of linear regions of a depth-$L$ width-$N$ ReLU network equals (up to a polynomial factor) the number of vertices of a tropical variety — a tropical Grassmannian computation. This is why the `expressivity-and-complexity` note in [[concepts/algebraic-geometry/neuroalgebraic-geometry/overview|Neuroalgebraic Geometry]] lists tropical geometry as a key tool.

The Legendre–Fourier analogy plays a role here too: **the dual of a tropical polynomial** (in the Legendre sense) encodes the Newton polytope of the corresponding classical polynomial, connecting tropical convexity to classical algebraic geometry via the Gelfand–Kapranov–Zelevinsky theory of $A$-discriminants.

---

## 7. References 📚

| Reference | Brief Summary | Link |
|-----------|---------------|------|
| [Legendre Transform — Surya Teja](https://surya-teja.com/2011/02/04/legendre-transform/) | Original source for the statement "Legendre = tropical Fourier"; intuitive overview | surya-teja.com |
| [Introduction to Tropical Geometry](https://arxiv.org/abs/1502.05950) | Maclagan–Sturmfels: standard textbook covering dequantization, amoebas, tropical varieties | arXiv:1502.05950 |
| [Tropical Mathematics](https://arxiv.org/abs/math/0408099) | Speyer–Sturmfels: expository introduction to the tropical semiring and tropical geometry | arXiv:math/0408099 |
| [Maslov Dequantization and the Idempotent Mathematics](https://arxiv.org/abs/math/0507014) | Litvinov: the dequantization perspective ($\hbar \to 0$ limit) connecting tropical and classical | arXiv:math/0507014 |
| [Convex Analysis](https://www.degruyter.com/document/doi/10.1515/9781400873173/html) | Rockafellar: definitive treatment of Legendre–Fenchel conjugates, inf-convolution, biconjugate theorem | Princeton Univ. Press |
| [Large Deviations Techniques and Applications](https://link.springer.com/book/10.1007/978-3-642-03311-7) | Dembo–Zeitouni: rate functions as Legendre transforms of log-MGFs; tropical Laplace in probability | Springer |
| [Tropical Convexity](https://arxiv.org/abs/math/0408099) | Develin–Sturmfels: tropical analogs of convex geometry, tropical polytopes | arXiv:math/0408009 |
| [Idempotent Analysis and Applications](https://link.springer.com/book/9780792345091) | Maslov–Kolokoltsov–Maslova: the $(\min,+)$ semiring in optimization and PDEs | Springer |
| [Singular Learning Theory](https://www.cambridge.org/core/books/algebraic-geometry-and-statistical-learning-theory/00C7E7C8EFB48AF64E8E1F22B3C7FC91) | Watanabe: Bayesian free energy as tropical Laplace limit; RLCT from KL divergence | Cambridge Univ. Press |
