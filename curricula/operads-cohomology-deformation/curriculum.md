# Curriculum: Operads, Hochschild Cohomology, Deformation Theory, and Witt Vectors

Prerequisites for the categorical entropy research thread. See [[research/categorical-entropy|Categorical Entropy]], [[research/entropy-operad-derivation|Entropy as an Operad Derivation]], and [[research/thermodynamic-semirings|Thermodynamic Semirings]].

---

## Operads: Definitions and Examples

- Symmetric sequences: collections $\{\mathcal{O}(n)\}_{n \geq 0}$ with $S_n$-actions; the composition product $\mathcal{O} \circ \mathcal{P}$ of two symmetric sequences; the monoidal category $(\mathbf{SymSeq}, \circ, \mathbf{1})$
- Definition of an operad as a monoid in $(\mathbf{SymSeq}, \circ)$: unit $\eta: \mathbf{1} \to \mathcal{O}$ and associative composition $\gamma: \mathcal{O} \circ \mathcal{O} \to \mathcal{O}$; equivalent definition via partial compositions $\circ_i$ and the associativity, unit, equivariance axioms
- Morphisms of operads; the category $\mathbf{Op}$; operads as a full subcategory of monoids in $\mathbf{SymSeq}$
- Key examples: $\mathrm{Ass}$ (all permutations, encodes associative algebras); $\mathrm{Com}$ (one operation per arity, encodes commutative algebras); $\mathrm{Lie}$ (antisymmetric bracket); $\mathrm{End}_V = \{\mathrm{Hom}(V^{\otimes n}, V)\}$ (the endomorphism operad of a vector space)
- The maps $\mathrm{Lie} \to \mathrm{Ass} \to \mathrm{Com}$ as operad morphisms; what they say about the categories of algebras
- Topological operads: operads in $\mathbf{Top}$ with continuous structure maps; the little disks operad $\mathcal{D}_n$ and its recognition principle for $n$-fold loop spaces
- The probability operad $\mathcal{P}$: $\mathcal{P}(n) = \Delta^{n-1}$ as a topological operad; operadic composition as distribution mixing; entropy as an internal $\mathcal{P}$-algebra; why $\mathcal{P}$ is not an algebraic operad (non-discrete arities)
- Colored operads and multicategories: operads with multiple types; small categories as colored operads with only unary operations; symmetric monoidal categories as colored operads

**References:** Loday & Vallette, *Algebraic Operads* Ch. 1–5; Markl, [*Operads and PROPs*](https://arxiv.org/abs/math/0601129); Leinster, [*Entropy and Diversity*](https://arxiv.org/abs/2012.02113) Ch. 2–3

---

## Operads: Algebras and Modules

- Algebras over an operad: a vector space $A$ with structure maps $\gamma_A: \mathcal{O}(n) \otimes A^{\otimes n} \to A$ compatible with $\gamma$; the category $\mathcal{O}\text{-}\mathbf{Alg}$; recovering associative, commutative, Lie algebras from $\mathrm{Ass}$, $\mathrm{Com}$, $\mathrm{Lie}$
- Free $\mathcal{O}$-algebras $\mathcal{O}(V) = \bigoplus_{n \geq 0} \mathcal{O}(n) \otimes_{S_n} V^{\otimes n}$; the free-forgetful adjunction $\mathcal{O}(-) \dashv U$
- Left $\mathcal{O}$-modules, right $\mathcal{O}$-modules, $(\mathcal{O}, \mathcal{O})$-bimodules; structure maps and compatibility with operadic composition; the category ${}^\mathcal{O}\mathbf{Mod}^\mathcal{O}$
- The enveloping algebra $U_\mathcal{O}(A)$ of an $\mathcal{O}$-algebra $A$: the universal associative algebra such that $\mathrm{Der}_\mathcal{O}(A, M) \cong \mathrm{Hom}_{U_\mathcal{O}(A)}(U_\mathcal{O}(A), M)$; explicit construction as a quotient of the tensor algebra on $\mathcal{O}$
- Derivations of $\mathcal{O}$-algebras: $\mathrm{Der}_\mathcal{O}(A, M)$ for an $A$-module $M$; inner derivations; the evaluation map $\mathrm{Der}(\mathcal{O}, -) \to \mathrm{Der}_\mathcal{O}(A, -)$ from operad derivations to algebra derivations
- Kähler differentials $\Omega^1_\mathcal{O}(A)$: the universal $A$-module representing $\mathrm{Der}_\mathcal{O}(A, -)$; construction as $U_\mathcal{O}(A) \otimes_A \Omega^1 / \text{(relations)}$; the operadic cotangent sequence
- $A_\infty$-algebras: algebras over the $A_\infty$ operad; structure maps $m_n: A^{\otimes n} \to A[2-n]$ satisfying the Stasheff relations $\sum_{i+j=n+1} \sum_k (-1)^\star m_i(\ldots, m_j(\ldots), \ldots) = 0$; $A_\infty$ as a minimal resolution of $\mathrm{Ass}$
- The probability operad revisited: the $\mathcal{P}$-algebra structure on $\mathbb{R}$ via weighted averages; entropy as a derivation of this algebra; the BFL twisted-composition rule as $\mathrm{Der}_\mathcal{P}(\mathbb{R}, \mathbb{R})$

**References:** Loday & Vallette, *Algebraic Operads* Ch. 5–6; Fresse, *Modules over Operads and Functors* Ch. 2–5; Voronov, *The $A_\infty$ operad and $A_\infty$ algebras*

---

## Operads: Koszul Duality and the Bar Construction

- The operadic bar construction $B(\mathcal{O})$: a dg-cooperad with underlying symmetric sequence $\mathcal{O}^{\circ+}$; the differential encoding operadic composition; $B(\mathcal{O})$ as the "derived" version of $\mathcal{O}$
- The cobar construction $\Omega(\mathcal{C})$ for a dg-cooperad $\mathcal{C}$: a dg-operad; the bar-cobar adjunction $\Omega \dashv B$; the counit $\Omega B(\mathcal{O}) \xrightarrow{\sim} \mathcal{O}$ as a cofibrant resolution
- The twisting morphism $\alpha: \mathcal{C} \to \mathcal{O}$ between a cooperad and an operad; the twisted composite product $\mathcal{C} \circ_\alpha \mathcal{O}$; the Maurer-Cartan equation for twisting morphisms
- Koszul duality for operads: the Koszul dual cooperad $\mathcal{O}^¡$ of a quadratic operad $\mathcal{O}$; the canonical twisting morphism $\kappa: \mathcal{O}^¡ \to \mathcal{O}$; a quadratic operad is Koszul iff $\kappa$ is a quasi-isomorphism
- The Koszul criterion: $\mathcal{O}$ is Koszul iff the bar construction $B(\mathcal{O})$ has homology concentrated in arity $= $ weight; examples $\mathrm{Ass}^! = \mathrm{Ass}$, $\mathrm{Com}^! = \mathrm{Lie}$, $\mathrm{Lie}^! = \mathrm{Com}$ (Koszul self-duality and the Lie-Com duality)
- $\mathcal{O}_\infty$-algebras from Koszul duality: for a Koszul operad $\mathcal{O}$, the $\infty$-version $\mathcal{O}_\infty = \Omega(\mathcal{O}^¡)$ gives the correct homotopy-coherent notion; $A_\infty = \Omega(\mathrm{Ass}^¡)$, $L_\infty = \Omega(\mathrm{Com}^¡)$
- Operadic cohomology via the bar construction: $H^\bullet_\mathcal{O}(A, M)$ computed by the complex $\mathrm{Hom}_{\mathcal{O}\text{-bimod}}(B(\mathcal{O}, A, A), M)$; the two-sided bar resolution $B(\mathcal{O}, A, A)$ as a free resolution of $A$ as an $\mathcal{O}$-algebra
- The deformation complex of an operad: $\mathrm{Def}(\mathcal{O}) = \mathrm{Hom}_{\mathbf{SymSeq}}(\mathcal{O}^¡, \mathcal{O})$ with the convolution $L_\infty$-algebra structure; Maurer-Cartan elements as deformed operad structures; $H^0(\mathrm{Def}(\mathcal{O}))$ as infinitesimal automorphisms, $H^1$ as infinitesimal deformations

**References:** Loday & Vallette, *Algebraic Operads* Ch. 6–7, 10–12; Ginzburg & Kapranov, *Koszul duality for operads* (Duke Math. J. 1994)

---

## Hochschild Cohomology

- The Hochschild cochain complex $C^n(A, M) = \mathrm{Hom}(A^{\otimes n}, M)$ for an associative $k$-algebra $A$ and $A$-bimodule $M$; the coboundary $\delta: C^n \to C^{n+1}$ via the explicit formula $((\delta f)(a_0, \ldots, a_n) = a_0 f(a_1, \ldots) + \sum_i (-1)^i f(\ldots, a_i a_{i+1}, \ldots) + (-1)^{n+1} f(\ldots, a_{n-1}) a_n)$
- $HH^0(A, M)$ as the center $Z(A, M) = \{m \in M : am = ma\}$; $HH^0(A, A) = Z(A)$
- $HH^1(A, M) = \mathrm{Der}(A, M) / \mathrm{InnDer}(A, M)$: derivations modulo inner derivations; proof that 1-cocycles are exactly derivations
- $HH^2(A, M)$ classifies infinitesimal deformations of $A$ over $k[\epsilon]/\epsilon^2$: the obstruction to extending a deformation from order $n$ to order $n+1$ lives in $HH^3$
- The cup product $\smile: HH^p \otimes HH^q \to HH^{p+q}$; the Gerstenhaber bracket $[-,-]: HH^p \otimes HH^q \to HH^{p+q-1}$; the resulting $G_\infty$ (Gerstenhaber) algebra structure on $HH^\bullet(A, A)$
- Cyclic cohomology $HC^\bullet(A)$: the cyclic cochain complex; the SBI long exact sequence $\cdots \to HC^{n-1} \xrightarrow{S} HC^{n+1} \xrightarrow{B} HH^n \xrightarrow{I} HC^n \to \cdots$; Connes' $B$-operator
- Operadic cohomology $H^\bullet_\mathcal{O}(A, M)$: the general construction for any operad $\mathcal{O}$; Hochschild as the $\mathrm{Ass}$ case; Harrison cohomology as the $\mathrm{Com}$ case; Chevalley-Eilenberg as the $\mathrm{Lie}$ case
- André-Quillen cohomology $AQ^\bullet(A, M)$ for commutative algebras: the derived functor of derivations; relation to Harrison via the Harrison-André-Quillen comparison
- Kähler differentials $\Omega^1_{A/k}$: the $A$-module generated by symbols $da$ with $d(ab) = a\,db + b\,da$; the universal property $\mathrm{Der}_k(A, M) \cong \mathrm{Hom}_A(\Omega^1_{A/k}, M)$; the cotangent sequence $\Omega^1_{B/k} \otimes_B A \to \Omega^1_{A/k} \to \Omega^1_{A/B} \to 0$
- The cotangent complex $\mathbb{L}_{A/k}$: the derived version of $\Omega^1$; André-Quillen cohomology as $\mathrm{Ext}^\bullet(\mathbb{L}_{A/k}, M)$; the cotangent complex for operadic algebras

**References:** Weibel, *Introduction to Homological Algebra* Ch. 9; Witherspoon, *Hochschild Cohomology for Algebras* Ch. 1–4; Loday, *Cyclic Homology* Ch. 1–2; Ginzburg, [*Lectures on Noncommutative Geometry*](https://arxiv.org/abs/math/0506603)

---

## Deformation Theory

- Formal deformations of a $k$-algebra $A$: a flat $k[[t]]$-algebra $A_t$ with $A_t / tA_t \cong A$; the first-order deformation $\mu_1 \in C^2(A, A)$ satisfying the Maurer-Cartan equation $\delta\mu_1 + \frac{1}{2}[\mu_1, \mu_1] = 0$ in $HH^\bullet$
- The Maurer-Cartan equation in a dg-Lie algebra $(\mathfrak{g}, d, [-,-])$: $d\gamma + \frac{1}{2}[\gamma, \gamma] = 0$; gauge equivalence; the Deligne groupoid $\mathrm{MC}(\mathfrak{g})$ as the moduli of deformations
- $L_\infty$-algebras: the homotopy-coherent generalization of dg-Lie algebras; higher brackets $\ell_n: \mathfrak{g}^{\otimes n} \to \mathfrak{g}[2-n]$; the $L_\infty$ Maurer-Cartan equation $\sum_{n \geq 1} \frac{1}{n!} \ell_n(\gamma^{\otimes n}) = 0$
- Rigidity: $HH^2(A, A) = 0$ implies $A$ is rigid (no nontrivial formal deformations); examples of rigid algebras
- Formality of a dg-algebra: quasi-isomorphic to its cohomology $H^\bullet(A)$; a formal dg-algebra has no higher Massey products; Kontsevich's formality theorem for $C^\infty(M)$ as a dg-Lie algebra
- Deformation quantization: deforming a commutative Poisson algebra $(A, \{-,-\})$ to a noncommutative $\star$-product $f \star g = fg + \hbar\{f,g\}/2 + O(\hbar^2)$; Kontsevich's theorem that every Poisson manifold admits a canonical deformation quantization
- Deformations of operads: the deformation complex $\mathrm{Def}(\mathcal{O})$ of an operad; the $L_\infty$-algebra structure on $\mathrm{Def}(\mathcal{O})$; Maurer-Cartan elements as deformed operad structures
- Thermodynamic semirings as deformations: the tropical semiring $(\mathbb{R}, \min, +)$ as the $\beta \to \infty$ limit; $\oplus_\beta$ as a formal deformation of $\min$; Shannon entropy as the first-order coefficient $\partial(\oplus_\beta)/\partial\beta^{-1}|_{\beta = \infty}$; the deformation complex of the tropical semiring

**References:** Manetti, [*Deformation Theory via Differential Graded Lie Algebras*](https://arxiv.org/abs/math/9907.179); Kontsevich & Soibelman, *Deformation Theory* (lecture notes); Loday & Vallette, *Algebraic Operads* Ch. 12–13; Kontsevich, [*Deformation Quantization of Poisson Manifolds*](https://arxiv.org/abs/q-alg/9709040)

---

## Witt Vectors

- Motivation: the problem of lifting a ring $R$ of characteristic $p$ to characteristic $0$; the non-canonicity of naive lifts; Witt vectors as the canonical solution
- Ghost components: a Witt vector $(a_0, a_1, a_2, \ldots) \in W(R)$ corresponds to ghost coordinates $w_n = \sum_{d \mid n} d\, a_d^{n/d}$; ring operations defined by requiring ghost maps to be ring homomorphisms $W(R) \to R^\mathbb{N}$
- $p$-typical Witt vectors $W(R)$: restricting to $p$-power ghost components $w_{p^n}$; the truncated Witt vectors $W_n(R)$; $W(\mathbb{F}_p) \cong \mathbb{Z}_p$
- Frobenius $F: W(R) \to W(R)$ shifting ghost coordinates by $F(w_n) = w_{pn}$; Verschiebung $V: W(R) \to W(R)$ with $V(w_n) = pw_{n/p}$ (zero if $p \nmid n$); the fundamental relations $FV = p$, $VF = \mathrm{mult. by } p$, $FV = VF$ on ghosts
- Big Witt vectors $\mathbb{W}(R)$: using all ghost components $w_n$ for $n \geq 1$; the $\lambda$-ring structure on $\mathbb{W}(R)$; Adams operations $\psi^n$
- The functor $W: \mathbf{CRing} \to \mathbf{CRing}$; right adjoint to the forgetful functor from $\delta$-rings; Witt vectors as the representing object for $p$-typical $\lambda$-ring structures
- $\delta$-rings: a ring $R$ with a map $\delta: R \to R$ satisfying $\delta(x+y) = \delta(x) + \delta(y) + \frac{x^p + y^p - (x+y)^p}{p}$ and $\delta(xy) = x^p\delta(y) + y^p\delta(x) + p\delta(x)\delta(y)$; the connection to Frobenius lifts $\phi(x) = x^p + p\delta(x)$; $\delta$-rings as the correct algebraic framework for $p$-adic Hodge theory
- Thermodynamic Witt vectors: the $\oplus_\beta$ family as a temperature-parametrized deformation of $(\mathbb{R}, \min)$; Boltzmann weights $e^{-\beta E_i}$ as the analogue of Teichmüller representatives; entropy as the $\delta$-ring derivation $\partial/\partial\beta$ at $\beta = \infty$
- Witt vectors and $p$-adic Hodge theory: $A_\mathrm{inf} = W(\mathcal{O}_{\mathbb{C}_p}^\flat)$ as the period ring; the Fontaine map; connection to prismatic cohomology (Bhatt-Scholze)

**References:** Rabinoff, [*The Theory of Witt Vectors*](https://math.uchicago.edu/~may/TQFT/Witt.pdf); Hazewinkel, [*Witt Vectors*](https://arxiv.org/abs/0804.3888); Borger, [*The Basic Geometry of Witt Vectors, I*](https://arxiv.org/abs/0801.1691); Joyal, *$\delta$-anneaux et vecteurs de Witt*

---

## Relating the Two Operadic Definitions of Entropy

The BFL internal $\mathcal{P}$-algebra condition and the Bradley operad derivation condition are derivations of *different objects* — an algebra over $\mathcal{P}$ vs. the operad $\mathcal{P}$ itself. The following topics are needed to understand how they relate.

- Modules over operads: left $\mathcal{O}$-modules $M$ with structure maps $\mathcal{O}(n) \otimes_{\mathcal{O}} M \to M$; bimodules; the category $_\mathcal{O}\mathbf{Mod}_\mathcal{O}$; examples for $\mathrm{Ass}$ recovering the classical notion of bimodule
- The enveloping algebra $U_\mathcal{O}(A)$: for an $\mathcal{O}$-algebra $A$, the universal associative algebra such that $\mathrm{Der}_\mathcal{O}(A, M) \cong \mathrm{Hom}_{U_\mathcal{O}(A)}(U_\mathcal{O}(A), M)$; the natural evaluation map $\mathrm{Der}(\mathcal{O}, -) \to \mathrm{Der}_\mathcal{O}(A, -)$ sending an operad derivation to an induced algebra derivation on every $\mathcal{O}$-algebra $A$
- Operadic Kähler differentials: the universal $A$-module $\Omega^1_\mathcal{O}(A)$ with $\mathrm{Der}_\mathcal{O}(A, M) \cong \mathrm{Hom}_A(\Omega^1_\mathcal{O}(A), M)$; the universal operad bimodule $\Omega^1(\mathcal{O})$ with $\mathrm{Der}(\mathcal{O}, M) \cong \mathrm{Hom}_{\mathcal{O}\text{-bimod}}(\Omega^1(\mathcal{O}), M)$; the comparison map $\Omega^1(\mathcal{O}) \otimes_\mathcal{O} A \to \Omega^1_\mathcal{O}(A)$ and when it is an isomorphism
- The bar construction and operadic cohomology: $B(\mathcal{O}, A, A)$ as the two-sided bar resolution computing $H^\bullet_\mathcal{O}(A, M)$; the bar construction $B(\mathcal{O})$ for the operad itself computing $H^\bullet_\mathrm{op}(\mathcal{O}, M)$; the map between the two complexes induced by evaluation
- The explicit calculation for $\mathcal{P}$: the evaluation map $\mathrm{Der}(\mathcal{P}, M) \to \mathrm{Der}_\mathcal{P}(\mathbb{R}, N)$ for Bradley's bimodule $M = C(\mathbb{R}^-, \mathbb{R})$ and the BFL bimodule $N = \mathbb{R}$; whether the Leibniz condition on $d$ maps to the twisted-composition condition on $\alpha$ under this map
- The operadic tangent and cotangent complex: $\mathbb{T}_\mathcal{O}(A) = \mathrm{Der}_\mathcal{O}(A, A)$ and $\mathbb{L}_\mathcal{O}(A) = \Omega^1_\mathcal{O}(A)$ as the cotangent complex; entropy as a class in $H^1(\mathbb{T}_\mathcal{O}(\mathcal{P}))$ unifying both derivation notions

**References:** Fresse, *Modules over Operads and Functors* (Springer LNM 1967) Ch. 4–5; Loday & Vallette, *Algebraic Operads* Ch. 6–7, 12; Hinich, [*Homological algebra of homotopy algebras*](https://arxiv.org/abs/q-alg/9702015); Bradley, [arXiv:2107.09581](https://arxiv.org/abs/2107.09581) §3–4
