# Categorical Entropy

## Sources

| Source | Type | Key Contribution | Link |
|--------|------|-----------------|------|
| [Baez, Fritz & Leinster (2011)](https://arxiv.org/abs/1106.1791) | paper | Entropy characterized as the unique functor measuring information loss; operadic / categorical uniqueness theorem | arXiv:1106.1791 |
| [Baudot & Bennequin (2015)](https://www.mdpi.com/1099-4300/17/5/3253) | paper | Shannon entropy is a 1-cocycle in an explicitly constructed information cohomology; chain rule = cocycle condition | *Entropy* 17(5):3253–3318 |
| [Vigneaux (2017)](https://arxiv.org/abs/1709.07807) | paper | Extends information cohomology to generalized information structures; axiomatizes the coefficient module | arXiv:1709.07807 |
| [Vigneaux (2020)](https://arxiv.org/abs/2003.02021) | paper | Homological characterization of generalized multinomial coefficients via the entropic chain rule | arXiv:2003.02021 |
| [Vigneaux (2019)](https://arxiv.org/abs/1807.05152) | paper | Information theory over finite vector spaces; q-analogs of entropy in the finite-field setting | arXiv:1807.05152 |
| [Leinster, *Entropy and Diversity* (2021)](https://arxiv.org/abs/2012.02113) | textbook | Book-length treatment of entropy via category theory; cleanest exposition of the operad of probability distributions and the BFL theorem | arXiv:2012.02113 |
| [Baez, "An Operadic Introduction to Entropy" (2011)](https://golem.ph.utexas.edu/category/2011/05/an_operadic_introduction_to_en.html) | blog | Explicit internal $\mathcal{P}$-algebra definition; derivation formula $H = D(\Sigma) - \Sigma D$; binary cocycle equation bridging to cohomology | n-Category Café |
| Marcolli, Ma148b Winter 2025 | course | Comprehensive treatment: categorical, geometric, and quantum information theory | [course page](https://www.its.caltech.edu/~matilde/Ma148bWinter2025.html) |
| Marcolli, Ma148a Fall 2021 | course | Emphasis on categorical formulations of entropy and Hochschild cohomology | [course page](https://www.its.caltech.edu/~matilde/Ma148aFall2021.html) |

---

## Context and Motivation

💡 The central puzzle: Shannon entropy $H(p) = -\sum_i p_i \log p_i$ satisfies a *chain rule*

$$H(X, Y) = H(X) + H(Y \mid X),$$

which uniquely characterizes it (up to scalar) among continuous, symmetric, normalized functionals — this is the content of the Khinchin/Faddeev uniqueness theorem. But *why* does such a characterization exist? What is the "correct" mathematical home for entropy?

Two programs studied here give answers that turn out to be secretly equivalent:

1. **Information cohomology** (Baudot-Bennequin, Vigneaux): the chain rule is a *cocycle condition* $\delta H = 0$, Shannon entropy is the generator of $H^1$, and higher cohomology groups classify higher-order dependencies.

2. **Operad of probability distributions** (Baez-Fritz-Leinster, Leinster): entropy is the unique *internal $\mathcal{P}$-algebra* map, with the chain rule as the operadic composition identity. The bridge between these two is the binary cocycle equation.

For the thermodynamic / algebraic deformation perspective (Marcolli-Thorngren semirings, Connes-Kreimer Hopf algebras, Gamma spaces), see [[research/thermodynamic-semirings|Thermodynamic Semirings]].

> [!INFO] Why categorical machinery?
> The uniqueness theorems for entropy (Faddeev 1956, Baez-Fritz-Leinster 2011) say: entropy is the *unique* solution to a system of functional equations. Category theory makes this precise by identifying entropy as a *universal* object — either a unique cohomology class or an initial algebra. This is stronger than "the only solution": it says entropy is canonical in a structural sense.

---

## The Cohomological Approach

### Setup: the information category

Baudot and Bennequin construct a category $\mathcal{P}$ whose objects are finite probability spaces $(X, p)$ and whose morphisms encode *refinements* (conditioning). A *functional* $f$ on this category assigns a real number to each object. The key definition is a *coboundary operator* $\delta$ that encodes the chain rule:

$$(\delta f)(X, Y) := f(X) + f(Y \mid X) - f(X, Y).$$

The chain rule $H(X, Y) = H(X) + H(Y \mid X)$ then says exactly that $\delta H = 0$: **Shannon entropy is a 1-cocycle**.

> [!NOTE] Cocycles vs coboundaries
> In any cohomology theory, 1-cocycles $Z^1$ are closed 1-forms and 1-coboundaries $B^1 = \delta(C^0)$ are exact. The cohomology $H^1 = Z^1 / B^1$ measures the "non-trivial" closed forms. The claim is that $[H_\text{Shannon}]$ generates $H^1$ — and is the *only* generator up to scalar.

### The coefficient module

The cohomology depends on a choice of *coefficient module* $\mathcal{A}$ — a sheaf of abelian groups over $\mathcal{P}$ specifying what values the cochains take. Baudot-Bennequin use the module of *measurable functions* on probability spaces. Vigneaux (2017) axiomatizes which modules $\mathcal{A}$ give rise to entropy-like cocycles, showing that the structure of $\mathcal{A}$ determines which entropy *family* (Shannon, Tsallis, Rényi) appears as $H^1$.

**Key result (Baudot-Bennequin):** With the standard coefficient module, $H^1(\mathcal{P}; \mathcal{A}) \cong \mathbb{R}$, generated by $H_\text{Shannon}$. Shannon entropy is, up to scalar, the *unique* 1-cocycle.

### Higher cohomology and mutual information

The higher groups $H^n$ classify *$n$-point dependencies*. The 2-cocycle condition gives mutual information $I(X; Y)$, and $H^2$ being nontrivial would indicate irreducible three-way interactions. This connects to:

- The *interaction information* $I(X; Y; Z) = H(X) + H(Y) + H(Z) - H(X,Y) - H(X,Z) - H(Y,Z) + H(X,Y,Z)$, which can be negative (unlike pairwise MI), potentially signaling a nontrivial class in $H^2$.
- Vigneaux (2020): the *multinomial coefficients* $\binom{n}{k_1, \ldots, k_r}$ satisfy a cocycle condition in this framework, giving a purely combinatorial shadow of the cohomology.

> [!QUESTION] Open: what does $H^n$ classify for $n \geq 2$?
> The Baudot-Bennequin paper leaves the computation of $H^n$ for $n \geq 2$ largely open. Is $H^2$ related to specific multivariate dependencies? Can one construct an "entropy spectral sequence" for hierarchical models?

---

## The Operad Approach

### 📐 Baez-Fritz-Leinster: information loss as a functor

#### The category FinProb

**Definition (FinProb).** The category $\mathbf{FinProb}$ has:
- **Objects:** finite probability spaces $(X, p)$ where $X$ is a finite set and $p : X \to [0,1]$ with $\sum_{x} p(x) = 1$.
- **Morphisms:** measure-preserving maps $f : (X, p) \to (Y, q)$, i.e. functions $f : X \to Y$ with $q_j = \sum_{i \in f^{-1}(j)} p_i$ for all $j \in Y$.

A morphism $f$ represents a *deterministic process* that collapses the distribution $p$ onto $q$ by grouping outcomes.

> [!EXAMPLE] A simple morphism
> Let $X = \{H, T\}$ with $p = (1/2, 1/2)$ and $Y = \{*\}$ with $q = (1)$. The unique function $f : X \to Y$ is measure-preserving. This morphism represents *complete erasure* of a fair coin flip. The information lost is $H(p) - H(q) = 1 - 0 = 1$ bit.

#### Information loss as a functor

The key move of BFL is to study **information loss** $F(f) := H(p) - H(q)$ rather than entropy itself.

Critically, $F$ is **functorial**: for composable morphisms $g : (X, p) \to (Y, q)$ and $f : (Y, q) \to (Z, r)$,

$$F(f \circ g) = H(p) - H(r) = \bigl(H(p) - H(q)\bigr) + \bigl(H(q) - H(r)\bigr) = F(g) + F(f).$$

**This is the chain rule.** Entropy itself is recovered as the loss of the terminal morphism $!: (X, p) \to (\{*\}, 1)$:

$$H(p) = F\bigl(! : (X,p) \to (\{*\},1)\bigr).$$

So entropy is not an intrinsic property of an object but rather the *information lost in the maximally destructive process* — total erasure.

#### The main theorem

**Theorem (Baez-Fritz-Leinster 2011).** Suppose $F$ assigns a value in $[0, \infty)$ to each morphism in $\mathbf{FinProb}$, satisfying:
1. **Functoriality:** $F(f \circ g) = F(f) + F(g)$
2. **Convex-linearity:** $F(\lambda f \oplus (1{-}\lambda) g) = \lambda F(f) + (1{-}\lambda) F(g)$
3. **Continuity:** $F$ is continuous in the probabilities

Then there exists $c \geq 0$ such that $F(f) = c\bigl(H(p) - H(q)\bigr)$ for all morphisms $f : (X,p) \to (Y,q)$, where $H$ is Shannon entropy.

> [!NOTE] What convex-linearity says
> The operation $\lambda f \oplus (1-\lambda)g$ forms the *mixture* of two processes: with probability $\lambda$ run process $f$, with probability $1-\lambda$ run process $g$. Convex-linearity says information loss scales linearly under this mixing — which is what distinguishes Shannon entropy from Tsallis entropy (see below).

#### The operad of probability distributions

The theorem has a cleaner restatement in operadic language, spelled out in Leinster's *Entropy and Diversity*. Define the **operad of probability distributions** $\mathcal{P}$:

- **Arity-$n$ operations:** $\mathcal{P}(n) = \Delta^{n-1}$ = the standard $(n{-}1)$-simplex, i.e. probability distributions on $n$ outcomes.
- **Operadic composition:** given $(p_1, \ldots, p_n) \in \mathcal{P}(n)$ and $(q^{(i)}_1, \ldots, q^{(i)}_{k_i}) \in \mathcal{P}(k_i)$, the composite is the *joint distribution*:

$$\bigl(p_1 q^{(1)}_1,\ \ldots,\ p_1 q^{(1)}_{k_1},\ p_2 q^{(2)}_1,\ \ldots,\ p_n q^{(n)}_{k_n}\bigr) \in \mathcal{P}(k_1 + \cdots + k_n).$$

**Key identity:** Shannon entropy satisfies the operadic composition rule

$$H(p_1 q^{(1)}_1, \ldots, p_n q^{(n)}_{k_n}) = H(p_1, \ldots, p_n) + \sum_{i=1}^n p_i\, H(q^{(i)}_1, \ldots, q^{(i)}_{k_i}),$$

which is the **chain rule** written as a *morphism condition* for $\mathcal{P}$.

> [!NOTE] Why "derivation" and not "algebra map"?
> A true algebra map would satisfy $H(\text{composite}) = H(p)$ (degenerate). Instead, entropy satisfies a *twisted* composition rule — it measures the deviation from being a constant. This is analogous to a derivation $\partial(fg) = \partial(f)g + f\partial(g)$ rather than a homomorphism.

#### The internal $\mathcal{P}$-algebra formulation

An **internal $\mathcal{P}$-algebra** in $\mathbb{R}_{\geq 0}$ is a continuous family of maps $\alpha = \{\alpha_n : \mathcal{P}(n) \to \mathbb{R}_{\geq 0}\}_{n \geq 1}$ satisfying:

1. **Twisted composition:** $\alpha_k(p \circ (r_1, \ldots, r_n)) = \alpha_n(p) + \sum_i p_i\, \alpha_{k_i}(r_i)$
2. **Normalization:** $\alpha_1((1)) = 0$
3. **Symmetry:** $\alpha_n(\sigma \cdot p) = \alpha_n(p)$ for all $\sigma \in S_n$

**Faddeev's theorem (operadic form):** The only internal $\mathcal{P}$-algebra in $\mathbb{R}_{\geq 0}$ is $\alpha_n = c \cdot H$ for $c \geq 0$.

Axiom (1) is the chain rule verbatim; axiom (2) says a certain outcome contributes no entropy; axiom (3) says entropy is label-independent. These three axioms, with continuity, are necessary and sufficient. The BFL functor theorem derives this from the more primitive data of $\mathbf{FinProb}$ — the internal $\mathcal{P}$-algebra formulation is the distilled operadic result.

#### Entropy as an additivity defect

There is a completely explicit formula making the "derivation" intuition precise. Define

$$D : [0,1] \to \mathbb{R}, \qquad D(x) = x \ln x \quad (D(0) := 0).$$

Then Shannon entropy is exactly the **additivity defect** of $D$ with respect to $\mathcal{P}$-algebra structure:

$$H(p_1, \ldots, p_n) = D\!\left(\sum_i p_i\right) - \sum_i D(p_i) = D(1) - \sum_i D(p_i) = -\sum_i p_i \ln p_i.$$

$D$ itself fails the twisted composition condition; $H$ is precisely the correction term measuring that failure. *Entropy arises because $x \ln x$ is not linear.*

> [!INFO] Connection to the Leibniz rule
> In differential algebra, a derivation $\partial$ satisfies $\partial(ab) = a\partial(b) + \partial(a)b$. The function $D(x) = x\ln x$ nearly satisfies this, with the "defect" linear in $\ln$. Shannon entropy is the integrated version of this defect over a probability simplex.

#### The binary cocycle equation: bridge to cohomology

🔑 The single most important identity connecting the operad and cohomology approaches is the **binary cocycle equation**. For any $a, b, c \geq 0$ with $a + b + c = 1$:

$$H(a,\ b) + H(a+b,\ c) = H(b,\ c) + H(a,\ b+c).$$

This says: the two ways to sequentially coarsen a three-outcome distribution agree. It is the chain rule applied twice — but written this way, it is a **cocycle condition** $\delta H = 0$ for the 1-cochain $H$ on the simplicial complex of probability spaces, exactly the Baudot-Bennequin formulation.

**The internal $\mathcal{P}$-algebra axiom (twisted composition) implies the binary cocycle equation, and vice versa** — they are equivalent formulations of the same constraint on $H$. This makes the bridge between BFL and Baudot-Bennequin explicit rather than analogical.

> [!QUESTION] Does the binary cocycle equation determine a simplicial structure?
> The equation looks like a 1-cocycle on a 2-simplex. Is there a natural simplicial set $\mathcal{S}_\bullet$ built from probability simplices such that $H$ defines a class in $H^1(\mathcal{S}_\bullet; \mathbb{R})$? If so, the BFL-to-Baudot-Bennequin bridge becomes a theorem.

#### Tsallis entropy from relaxing convex-linearity

If axiom (2) is replaced by **$\alpha$-homogeneity:**

$$F(\lambda f \oplus (1{-}\lambda)g) = \lambda^\alpha F(f) + (1{-}\lambda)^\alpha F(g), \quad \alpha > 0,$$

then the unique solution (Theorem 7 of BFL) is the **Tsallis entropy** of order $\alpha$:

$$H_\alpha(p) = \frac{1}{\alpha - 1}\Bigl(1 - \sum_i p_i^\alpha\Bigr), \qquad \lim_{\alpha \to 1} H_\alpha = H_\text{Shannon}.$$

The parameter $\alpha$ controls how information loss scales under probabilistic mixing. Shannon entropy is singled out by *linear* scaling ($\alpha = 1$), which is what makes it additive over independent systems.

---

## Open Questions

> [!QUESTION] 1. Higher cohomology
> What does $H^n(\mathcal{P}; \mathcal{A})$ classify for $n \geq 2$? Are there natural probability models (Markov fields, exponential families) that produce nontrivial classes in $H^2$?

> [!QUESTION] 2. Quantum generalization
> Baudot-Bennequin work over classical probability spaces. Von Neumann entropy $S(\rho) = -\mathrm{tr}(\rho \log \rho)$ satisfies subadditivity rather than the chain rule. Does it define a *relative* cocycle? What cohomology does quantum entropy live in?

> [!QUESTION] 3. Is there an information-theoretic de Rham theorem?
> Do the Baudot-Bennequin cocycles and the Marcolli-Thorngren semiring deformations (see [[research/thermodynamic-semirings|Thermodynamic Semirings]]) compute the same "information cohomology"? A precise formulation would require identifying what plays the role of "singular chains" in the probability-space setting.
