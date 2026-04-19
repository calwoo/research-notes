# Categorical Entropy

## Sources

| Source | Type | Key Contribution | Link |
|--------|------|-----------------|------|
| [Baez, Fritz & Leinster (2011)](https://arxiv.org/abs/1106.1791) | paper | Entropy characterized as the unique functor measuring information loss; operadic / categorical uniqueness theorem | arXiv:1106.1791 |
| [Baudot & Bennequin (2015)](https://www.mdpi.com/1099-4300/17/5/3253) | paper | Shannon entropy is a 1-cocycle in an explicitly constructed information cohomology; chain rule = cocycle condition | *Entropy* 17(5):3253–3318 |
| [Vigneaux (2017)](https://arxiv.org/abs/1709.07807) | paper | Extends information cohomology to generalized information structures; axiomatizes the coefficient module | arXiv:1709.07807 |
| [Vigneaux (2020)](https://arxiv.org/abs/2003.02021) | paper | Homological characterization of generalized multinomial coefficients via the entropic chain rule | arXiv:2003.02021 |
| [Vigneaux (2019)](https://arxiv.org/abs/1807.05152) | paper | Information theory over finite vector spaces; q-analogs of entropy in the finite-field setting | arXiv:1807.05152 |
| [Marcolli & Thorngren (2011)](https://arxiv.org/abs/1108.2874) | paper | Thermodynamic semirings: entropy axioms as a deformation of the tropical semiring | arXiv:1108.2874 |
| [Marcolli & Tedeschi (2014)](https://arxiv.org/abs/1412.0247) | paper | Entropy algebras and Birkhoff factorization in Hopf algebras of rooted trees | arXiv:1412.0247 |
| [Marcolli (2018)](https://arxiv.org/abs/1807.05314) | paper | Gamma spaces and information loss; homotopy-theoretic perspective on entropy functors | arXiv:1807.05314 |
| Marcolli, Ma148b Winter 2025 | course | Comprehensive treatment: categorical, geometric, and quantum information theory | [course page](https://www.its.caltech.edu/~matilde/Ma148bWinter2025.html) |
| Marcolli, Ma148a Fall 2021 | course | Emphasis on categorical formulations of entropy and Hochschild cohomology | [course page](https://www.its.caltech.edu/~matilde/Ma148aFall2021.html) |
| [Leinster, *Entropy and Diversity* (2021)](https://arxiv.org/abs/2012.02113) | textbook | Book-length treatment of entropy via category theory; cleanest exposition of the operad of probability distributions and the BFL theorem | arXiv:2012.02113 |
| [Baez, "An Operadic Introduction to Entropy" (2011)](https://golem.ph.utexas.edu/category/2011/05/an_operadic_introduction_to_en.html) | blog | Explicit internal $\mathcal{P}$-algebra definition; derivation formula $H = D(\Sigma) - \Sigma D$; binary cocycle equation bridging to cohomology | n-Category Café |

---

## Context and Motivation

💡 The central puzzle: Shannon entropy $H(p) = -\sum_i p_i \log p_i$ satisfies a *chain rule*

$$H(X, Y) = H(X) + H(Y \mid X),$$

which uniquely characterizes it (up to scalar) among continuous, symmetric, normalized functionals — this is the content of the Khinchin/Faddeev uniqueness theorem. But *why* does such a characterization exist? What is the "correct" mathematical home for entropy?

Two research programs give different but potentially equivalent answers:

1. **Information cohomology** (Baudot-Bennequin, Vigneaux): the chain rule is a *cocycle condition*, Shannon entropy is the generator of $H^1$, and higher cohomology groups classify higher-order dependencies.

2. **Thermodynamic semirings / entropy operads** (Marcolli-Thorngren, Marcolli-Tedeschi): entropy axioms arise as *deformations* of algebraic structure (tropical semirings, Hopf algebras of trees), placing entropy in the world of algebraic geometry and renormalization.

> [!INFO] Why categorical machinery?
> The uniqueness theorems for entropy (Faddeev 1956, Baez-Fritz-Leinster 2011) say: entropy is the *unique* solution to a system of functional equations. Category theory makes this precise by identifying entropy as a *universal* object — either an initial algebra, a terminal coalgebra, or a unique cohomology class. This is stronger than "the only solution": it says entropy is canonical in a structural sense.

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
> The Baudot-Bennequin paper leaves the computation of $H^n$ for $n \geq 2$ largely open. Is $H^2$ related to specific multivariate dependencies? Can one construct a "entropy spectral sequence" for hierarchical models?

---

## The Algebraic/Operad Approach

### 📐 Baez-Fritz-Leinster: information loss as a functor

#### The category FinProb

**Definition (FinProb).** The category $\mathbf{FinProb}$ has:
- **Objects:** finite probability spaces $(X, p)$ where $X$ is a finite set and $p : X \to [0,1]$ with $\sum_{x} p(x) = 1$.
- **Morphisms:** measure-preserving maps $f : (X, p) \to (Y, q)$, i.e. functions $f : X \to Y$ with $q_j = \sum_{i \in f^{-1}(j)} p_i$ for all $j \in Y$.

A morphism $f$ represents a *deterministic process* that collapses the distribution $p$ onto $q$ by grouping outcomes. Note that $q$ is completely determined by $p$ and $f$ — so the morphism data is just the function $f$, but the measure-preserving condition forces $q = f_* p$.

> [!EXAMPLE] A simple morphism
> Let $X = \{H, T\}$ with $p = (1/2, 1/2)$ and $Y = \{*\}$ with $q = (1)$. The unique function $f : X \to Y$ is measure-preserving. This morphism represents *complete erasure* of a fair coin flip. The information lost is $H(p) - H(q) = 1 - 0 = 1$ bit.

#### Information loss as a functor

The key move of BFL is to study **information loss** $F(f) := H(p) - H(q)$ rather than entropy itself. This is a non-negative quantity measuring how much is forgotten by the process $f$.

Critically, $F$ is **functorial**: for composable morphisms $g : (X, p) \to (Y, q)$ and $f : (Y, q) \to (Z, r)$,

$$F(f \circ g) = H(p) - H(r) = \bigl(H(p) - H(q)\bigr) + \bigl(H(q) - H(r)\bigr) = F(g) + F(f).$$

**This is the chain rule.** Entropy itself is recovered as the loss of the terminal morphism $!: (X, p) \to (\{*\}, 1)$:

$$H(p) = F\bigl(! : (X,p) \to (\{*\},1)\bigr).$$

So entropy is not an intrinsic property of an object but rather the *information lost in the maximally destructive process* — total erasure. This reframing is what makes the categorical treatment clean: once $F$ is defined on morphisms, everything else follows.

#### The main theorem

**Theorem (Baez-Fritz-Leinster 2011).** Suppose $F$ assigns a value in $[0, \infty)$ to each morphism in $\mathbf{FinProb}$, satisfying:
1. **Functoriality:** $F(f \circ g) = F(f) + F(g)$
2. **Convex-linearity:** $F(\lambda f \oplus (1{-}\lambda) g) = \lambda F(f) + (1{-}\lambda) F(g)$
3. **Continuity:** $F$ is continuous in the probabilities

Then there exists $c \geq 0$ such that $F(f) = c\bigl(H(p) - H(q)\bigr)$ for all morphisms $f : (X,p) \to (Y,q)$, where $H$ is Shannon entropy.

*Uniqueness up to scalar.* The three axioms force $F$ to be Shannon information loss. There is no other consistent, convex-linear, continuous way to assign a "how much was forgotten" number to a measure-preserving process.

> [!NOTE] What convex-linearity says
> The operation $\lambda f \oplus (1-\lambda)g$ forms the *mixture* of two processes: with probability $\lambda$ run process $f$, with probability $1-\lambda$ run process $g$. Convex-linearity says information loss scales linearly under this mixing — which is what distinguishes Shannon entropy from Rényi entropy (see below).

#### The operad of probability distributions

The theorem has a cleaner restatement in operadic language, spelled out in Leinster's *Entropy and Diversity*. Define the **operad of probability distributions** $\mathcal{P}$:

- **Arity-$n$ operations:** $\mathcal{P}(n) = \Delta^{n-1}$ = the standard $(n{-}1)$-simplex, i.e. probability distributions on $n$ outcomes.
- **Operadic composition:** given $(p_1, \ldots, p_n) \in \mathcal{P}(n)$ and $(q^{(i)}_1, \ldots, q^{(i)}_{k_i}) \in \mathcal{P}(k_i)$ for $i = 1, \ldots, n$, the composite is the *joint distribution*:

$$\bigl(p_1 q^{(1)}_1,\ \ldots,\ p_1 q^{(1)}_{k_1},\ p_2 q^{(2)}_1,\ \ldots,\ p_n q^{(n)}_{k_n}\bigr) \in \mathcal{P}(k_1 + \cdots + k_n).$$

This is exactly *mixing*: first choose outcome $i$ with probability $p_i$, then outcome $j$ within group $i$ with probability $q^{(i)}_j$.

An **algebra** for $\mathcal{P}$ with values in a set $A$ is a family of maps $\alpha_n : \mathcal{P}(n) \times A^n \to A$ compatible with operadic composition. Taking $A = \mathbb{R}$ and $\alpha_n(p, h_1, \ldots, h_n) = \sum_i p_i h_i$ gives the *expected value* algebra. Shannon entropy is a **derivation** of this algebra: the deviation of $H$ from being an algebra map is precisely the entropic term.

**Key identity:** Shannon entropy satisfies the operadic composition rule

$$H(p_1 q^{(1)}_1, \ldots, p_n q^{(n)}_{k_n}) = H(p_1, \ldots, p_n) + \sum_{i=1}^n p_i\, H(q^{(i)}_1, \ldots, q^{(i)}_{k_i}),$$

which is the **chain rule** written as a *morphism condition* for the operad $\mathcal{P}$. This is the operadic content of BFL: entropy is the unique (up to scalar) continuous map $H : \mathcal{P}(n) \to \mathbb{R}$ satisfying this composition identity for all $n$ and all choices of distributions.

> [!NOTE] Why "derivation" and not "algebra map"?
> A true algebra map would satisfy $H(\text{composite}) = H(p)$ (ignoring the second argument entirely) or something similarly degenerate. Instead, entropy satisfies a *twisted* composition rule — it measures the deviation from being a constant. This is analogous to a *derivation* in algebra: $D(fg) = D(f)g + fD(g)$ rather than $D(fg) = D(f)D(g)$.

#### The internal $\mathcal{P}$-algebra formulation

The blog post by Baez gives the sharpest purely operadic statement. An **internal $\mathcal{P}$-algebra** in $\mathbb{R}_{\geq 0}$ is a continuous family of maps $\alpha = \{\alpha_n : \mathcal{P}(n) \to \mathbb{R}_{\geq 0}\}_{n \geq 1}$ satisfying:

1. **Twisted composition:** $\alpha_k(p \circ (r_1, \ldots, r_n)) = \alpha_n(p) + \sum_i p_i\, \alpha_{k_i}(r_i)$ where $k = \sum_i k_i$
2. **Normalization:** $\alpha_1((1)) = 0$
3. **Symmetry:** $\alpha_n(\sigma \cdot p) = \alpha_n(p)$ for all permutations $\sigma \in S_n$

**Faddeev's theorem (operadic form):** The only internal $\mathcal{P}$-algebra in $\mathbb{R}_{\geq 0}$ is $\alpha_n = c \cdot H$ for some constant $c \geq 0$.

Axiom (1) is the chain rule verbatim. Axiom (2) says: a certain outcome has no entropy. Axiom (3) says: entropy doesn't depend on labeling of outcomes. *These three axioms, with continuity, are necessary and sufficient.*

The relation to the BFL functor theorem is that BFL derives this from the more primitive data of $\mathbf{FinProb}$ — the internal $\mathcal{P}$-algebra formulation is the distilled result, with the categorical machinery stripped away.

#### Entropy as an additivity defect

There is a completely explicit construction of $H$ that makes the "derivation" intuition precise. Define

$$D : [0,1] \to \mathbb{R}, \qquad D(x) = x \ln x \quad (D(0) := 0).$$

Then Shannon entropy is exactly the **additivity defect** of $D$ with respect to $\mathcal{P}$-algebra structure:

$$H(p_1, \ldots, p_n) = D\!\left(\sum_i p_i\right) - \sum_i D(p_i) = D(1) - \sum_i D(p_i) = -\sum_i p_i \ln p_i.$$

Why is this interesting? Because $D$ itself is *not* an internal $\mathcal{P}$-algebra map — it fails the twisted composition condition. The entropy $H$ is precisely the *correction term* that measures this failure. In the language of algebra, if $A \xrightarrow{D} B$ is not a homomorphism, then $\partial(f,g) = D(fg) - D(f) - D(g)$ is a 2-cochain measuring the failure, and requiring $\partial$ to be a cocycle yields constraints. Here:

$$H(p_1, \ldots, p_n) = D(1) - \sum_i D(p_i)$$

is the failure of $D$ to be additive over $\mathcal{P}$-composition with the uniform distribution. *Entropy arises because $x \ln x$ is not linear.*

> [!INFO] Connection to the Leibniz rule
> In differential algebra, a *derivation* $\partial$ on a ring $R$ satisfies $\partial(ab) = a\partial(b) + \partial(a)b$. The function $D(x) = x\ln x$ satisfies $D(xy) = xD(y) + D(x)y + D(x)D(y)/\text{(lower order)}$ — not quite a derivation, but the leading "defect" is linear in $\ln$. Shannon entropy is the integrated version of this defect over a probability simplex.

#### The binary cocycle equation: bridge to cohomology

🔑 The single most important identity connecting the operad and cohomology approaches is the **binary cocycle equation** for Shannon entropy. For any $a, b, c \geq 0$ with $a + b + c = 1$:

$$H(a,\ b) + H(a+b,\ c) = H(b,\ c) + H(a,\ b+c).$$

This says: the two ways to *sequentially coarsen* a three-outcome distribution agree. Starting from $(a, b, c)$:
- *Left side:* first group $\{a, b\}$ vs $\{c\}$, then split $\{a,b\}$ — entropy of the first split plus conditional entropy of the second.
- *Right side:* first group $\{a\}$ vs $\{b, c\}$, then split $\{b,c\}$ — entropy of the first split plus conditional entropy of the second.

The equality is the chain rule applied twice. But written this way, it is a **cocycle condition** $\delta H = 0$ for the 1-cochain $H$ on the simplicial complex of probability spaces — exactly the Baudot-Bennequin formulation. 

**This is the explicit bridge between the BFL operad approach and information cohomology:** the internal $\mathcal{P}$-algebra axiom (twisted composition) implies the binary cocycle equation, and the binary cocycle equation, extended to all arities via symmetry and continuity, reconstructs the internal $\mathcal{P}$-algebra axiom. They are equivalent formulations of the same constraint on $H$.

> [!QUESTION] Does the binary cocycle equation determine a simplicial structure?
> The equation $H(a,b) + H(a+b,c) = H(b,c) + H(a,b+c)$ looks like a 1-cocycle on a 2-simplex (three vertices $a, b, c$, three edges). Is there a natural simplicial set $\mathcal{S}_\bullet$ built from probability simplices such that $H$ defines a class in $H^1(\mathcal{S}_\bullet; \mathbb{R})$? If so, this would make the BFL-to-Baudot-Bennequin bridge into a theorem rather than an analogy.

#### Tsallis entropy from relaxing convex-linearity

If axiom (2) is replaced by **$\alpha$-homogeneity:**

$$F(\lambda f \oplus (1{-}\lambda)g) = \lambda^\alpha F(f) + (1{-}\lambda)^\alpha F(g), \quad \alpha > 0,$$

then the unique solution (Theorem 7 of BFL) is the **Tsallis entropy** of order $\alpha$:

$$H_\alpha(p) = \frac{1}{\alpha - 1}\Bigl(1 - \sum_i p_i^\alpha\Bigr), \qquad H_1(p) = \lim_{\alpha \to 1} H_\alpha(p) = H_\text{Shannon}(p).$$

The parameter $\alpha$ measures how information loss scales under probabilistic mixing. Shannon entropy is the unique case where this scaling is linear ($\alpha = 1$), which is precisely what makes it additive over independent systems.

> [!QUESTION] What distinguishes $\alpha = 1$ physically?
> From the BFL perspective, Shannon entropy is singled out by *linear* scaling under mixing. From a physics perspective, $\alpha = 1$ corresponds to *extensive* thermodynamic systems (entropy scales with system size). Is there a categorical explanation for why extensivity forces $\alpha = 1$?

### Thermodynamic semirings

A *semiring* $(R, \oplus, \otimes)$ satisfies the ring axioms except subtraction. The *tropical semiring* is $(\mathbb{R} \cup \{+\infty\}, \min, +)$, which arises as the $\beta \to \infty$ (zero-temperature) limit of the *log-sum-exp* operation:

$$a \oplus_\beta b := -\frac{1}{\beta} \log(e^{-\beta a} + e^{-\beta b}).$$

Marcolli and Thorngren (2011) observe that the *Boltzmann entropy* emerges as the derivative of $\oplus_\beta$ with respect to $\beta$ at the tropical point $\beta = \infty$. More precisely: if we track the "correction" to the tropical semiring as $\beta^{-1} \to 0^+$, entropy appears as the first-order deformation term.

**Key structure:** The Witt vectors construction provides the algebraic framework — $\oplus_\beta$ is the addition law of a family of Witt vector semirings parametrized by temperature $\beta^{-1}$. The entropy axioms (positivity, symmetry, chain rule) correspond to the *axioms of a semiring homomorphism* from this deformed structure to $\mathbb{R}$.

> [!EXAMPLE] Rényi entropy from $q$-deformation
> Setting $\beta = 1/(1-q)$ and deforming the coefficient module gives the *Rényi entropy*
> $$H_q(p) = \frac{1}{1-q} \log \sum_i p_i^q.$$
> The $q \to 1$ limit recovers Shannon. In the semiring language, different entropy families correspond to different *deformation parameters* of the same algebraic structure.

### Entropy operads and rooted trees

An *operad* $\mathcal{O}$ consists of spaces $\mathcal{O}(n)$ of "$n$-ary operations" with composition maps satisfying associativity. The *operad of rooted trees* encodes iterated binary operations.

Marcolli-Tedeschi (2014) construct a *Hopf algebra* $\mathcal{H}$ on rooted trees (the Connes-Kreimer algebra from renormalization theory) and show:

1. Shannon entropy is a *character* of $\mathcal{H}$ — a multiplicative functional $\phi: \mathcal{H} \to \mathbb{R}$.
2. The *Birkhoff factorization* $\phi = \phi_- \star \phi_+$ (standard in renormalization) applied to the entropy character extracts the "entropic content" from a probability tree in a canonical way.

This connects information theory to the *algebraic renormalization* program (Connes-Kreimer), suggesting that entropy extraction from hierarchical models is structurally identical to the renormalization of divergences in QFT.

> [!WARNING] Caveat on the QFT analogy
> The Connes-Kreimer Hopf algebra controls *subdivergences* of Feynman diagrams. The Birkhoff factorization separates "pole part" from "finite part." In the entropy setting, the analogy is suggestive but the precise dictionary between subdivergences and conditional entropies is not fully worked out in the literature.

### Gamma spaces and homotopy-theoretic entropy

Marcolli (2018) takes a further step: *Gamma spaces* (in the sense of Segal) are functors $\Gamma^{op} \to \mathbf{Top}$ from the category of finite pointed sets, and they model *infinite loop spaces* — spaces with coherent $E_\infty$ multiplication. Segal showed that $\Gamma$-spaces provide a model for connective spectra (stable homotopy theory).

The claim is that information loss functors naturally define $\Gamma$-space structures, placing entropy in *stable homotopy theory*. This is the most structurally ambitious claim: entropy as a map of spectra.

> [!QUESTION] Open: what is the spectrum of entropy?
> If entropy defines a $\Gamma$-space, what is the associated spectrum? What do the stable homotopy groups $\pi_n$ of this spectrum compute? This connects to Waldhausen's algebraic K-theory of spaces and potentially to the $K$-theory of information structures.

---

## Synthesis: Two Faces of the Same Program?

Both approaches identify entropy via a *universal property*:

| | Cohomological (B-B/Vigneaux) | Categorical (BFL) | Algebraic (Marcolli) |
|--|------------------------------|-------------------|----------------------|
| **Setting** | Sheaves on $\mathcal{P}$ | $\mathbf{FinProb}$, operad $\mathcal{P}$ | Hopf algebras, semirings |
| **Entropy as** | Generator of $H^1$ | Unique functorial information loss | Character / deformation derivative |
| **Chain rule** | Cocycle condition $\delta H = 0$ | Functoriality $F(f \circ g) = F(f) + F(g)$ | Semiring homomorphism axiom |
| **Uniqueness** | $H^1 \cong \mathbb{R}$ | Functorial + convex-linear + continuous | Birkhoff factorization is unique |
| **Generalizations** | Different coefficient modules | $\alpha$-homogeneity → Tsallis $H_\alpha$ | Different deformation parameters |

The chain rule appears in both as the *central constraint*. The cohomological approach treats it as a *differential-geometric* datum (closed form), while the algebraic approach treats it as an *algebraic* datum (homomorphism condition). These should be related by a version of the *de Rham theorem* for information structures — cohomology computed via differential forms vs. singular cohomology.

> [!QUESTION] Is there an information-theoretic de Rham theorem?
> In differential geometry, de Rham cohomology (differential forms) and singular cohomology agree by de Rham's theorem. Is there a parallel statement here: do the Baudot-Bennequin cocycles and the Marcolli-Thorngren semiring deformations compute the same "information cohomology"? A precise formulation would require identifying what plays the role of "singular chains" in the probability-space setting.

---

## Open Questions

> [!QUESTION] 1. Higher cohomology
> What does $H^n(\mathcal{P}; \mathcal{A})$ classify for $n \geq 2$? Are there natural probability models (Markov fields, exponential families) that produce nontrivial classes in $H^2$?

> [!QUESTION] 2. The unification question
> Is there a single framework that contains both the cohomological (Baudot-Bennequin) and algebraic (Marcolli-Thorngren) approaches as special cases? Candidate: the *derived category* of sheaves on $\mathcal{P}$, with the semiring structure arising from a monoidal structure on this derived category.

> [!QUESTION] 3. Quantum generalization
> Baudot-Bennequin work over classical probability spaces. Von Neumann entropy $S(\rho) = -\text{tr}(\rho \log \rho)$ satisfies a subadditivity $S(\rho_{AB}) \leq S(\rho_A) + S(\rho_B)$ rather than the chain rule. Does it define a *relative* cocycle (coboundary up to a term)? What cohomology does quantum entropy live in?

> [!QUESTION] 4. Entropy and K-theory
> Marcolli's Gamma-space construction connects entropy to $K$-theory spectra. Is there a precise sense in which the *K-theory of a probability space* computes information-theoretic invariants? The work of Baas-Dundas-Richter-Rognes on 2-vector bundles might be relevant.
