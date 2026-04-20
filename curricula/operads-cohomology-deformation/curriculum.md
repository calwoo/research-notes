# Curriculum: Operads, Hochschild Cohomology, Deformation Theory, and Witt Vectors

Prerequisites for the categorical entropy research thread. See [[research/categorical-entropy|Categorical Entropy]], [[research/entropy-operad-derivation|Entropy as an Operad Derivation]], and [[research/thermodynamic-semirings|Thermodynamic Semirings]].

---

## Operads

**Core concepts to understand:**

- [ ] Symmetric sequences and the composition product $\circ$
- [ ] Definition of an operad (symmetric, non-symmetric); unit and associativity axioms
- [ ] Algebras over an operad; free algebras
- [ ] Bimodules over an operad; left and right module actions
- [ ] Morphisms of operads; the category of operads
- [ ] Key examples: $\text{Ass}$, $\text{Com}$, $\text{Lie}$, $\text{End}_V$, $A_\infty$, $E_\infty$
- [ ] The operad of probability distributions $\mathcal{P}$: $\mathcal{P}(n) = \Delta^{n-1}$, composition by mixing
- [ ] Topological operads; continuous families of operations
- [ ] The operadic bar construction and cobar construction
- [ ] Koszul duality for operads: the Koszul dual operad $\mathcal{O}^!$, Koszul operads
- [ ] $\infty$-operads (overview level): colored operads, multicategories

**Key references:**

- Markl, [*Operads and PROPs*](https://arxiv.org/abs/math/0601129) — concise, in the Marcolli course bibliography
- Loday & Vallette, *Algebraic Operads* (Springer, freely available) — the comprehensive reference; Ch. 1–5 cover the above
- Voronov, *The $A_\infty$ operad and $A_\infty$ algebras* (lecture notes) — in the Marcolli course bibliography
- Leinster, [*Entropy and Diversity*](https://arxiv.org/abs/2012.02113) Ch. 2–3 — for the probability operad $\mathcal{P}$ specifically

---

## Hochschild Cohomology

**Core concepts to understand:**

- [ ] The Hochschild cochain complex $C^\bullet(A, M)$ for an associative algebra $A$ and bimodule $M$
- [ ] The coboundary map $\delta: C^n \to C^{n+1}$; explicit formula
- [ ] $HH^0(A, M)$ = center / invariants
- [ ] $HH^1(A, M) = \mathrm{Der}(A, M) / \mathrm{InnDer}(A, M)$ — the derivation isomorphism
- [ ] $HH^2(A, M)$ = infinitesimal deformations of $A$ as an algebra
- [ ] $HH^3(A, M)$ = obstructions to extending deformations
- [ ] The cup product on $HH^\bullet$; the Gerstenhaber bracket; the $G_\infty$-structure
- [ ] Cyclic cohomology $HC^\bullet$ and the SBI sequence $HC^{n-1} \to HH^n \to HC^n$
- [ ] Operadic cohomology as a generalization: $H^\bullet_\mathcal{O}(A, M)$ for a general operad $\mathcal{O}$
- [ ] Harrison cohomology (commutative case); André-Quillen cohomology
- [ ] Chevalley-Eilenberg cohomology (Lie case)
- [ ] Kähler differentials $\Omega_{A/k}$; the universal derivation $d: A \to \Omega_{A/k}$; $\mathrm{Der}(A, M) \cong \mathrm{Hom}_A(\Omega_{A/k}, M)$

**Key references:**

- Weibel, *Introduction to Homological Algebra* Ch. 9 — Hochschild cohomology from scratch
- Witherspoon, *Hochschild Cohomology for Algebras* (Cambridge, 2019) Ch. 1–4
- Loday, *Cyclic Homology* (Springer) Ch. 1–2 — for cyclic cohomology and the full structure
- Ginzburg, [*Lectures on Noncommutative Geometry*](https://arxiv.org/abs/math/0506603) — broader context

---

## Deformation Theory

**Core concepts to understand:**

- [ ] Formal deformations of an algebra $A$ over $k[[t]]$; the Maurer-Cartan equation $d\gamma + \frac{1}{2}[\gamma, \gamma] = 0$
- [ ] The deformation complex and its role: $HH^2$ parametrizes infinitesimal deformations, $HH^3$ harbors obstructions
- [ ] Rigidity: an algebra with $HH^2 = 0$ has no nontrivial deformations
- [ ] $L_\infty$-algebras: the correct homotopy-coherent framework controlling deformation problems
- [ ] The Deligne conjecture / Deligne groupoid: $MC(L)$ as the moduli space of deformations
- [ ] Formality: a dg-algebra is formal if it is quasi-isomorphic to its cohomology; Kontsevich formality
- [ ] Deformation quantization: deforming a Poisson algebra to a noncommutative algebra; $\star$-products
- [ ] Deformations of operads; the deformation complex of an operad $\mathcal{O}$
- [ ] Application to thermodynamic semirings: the tropical semiring as the $t \to 0$ limit; entropy as $\partial/\partial t|_{t=0}$

**Key references:**

- Manetti, [*Deformation Theory via Differential Graded Lie Algebras*](https://arxiv.org/abs/math/9907.179) — clean and focused
- Kontsevich & Soibelman, *Deformation Theory* (lecture notes, available online) — comprehensive
- Loday & Vallette, *Algebraic Operads* Ch. 12–13 — deformations of operads specifically
- Kontsevich, [*Deformation Quantization of Poisson Manifolds*](https://arxiv.org/abs/q-alg/9709040) — the landmark paper

---

## Witt Vectors

**Core concepts to understand:**

- [ ] Motivation: lifting from characteristic $p$ to characteristic $0$; the Teichmüller lift
- [ ] $p$-typical Witt vectors $W(R)$: ghost components $w_n$, addition and multiplication via ghost coordinates
- [ ] The Frobenius $F: W(R) \to W(R)$ and Verschiebung $V: W(R) \to W(R)$; the relations $FV = p$, $VF = V(\cdot)p$
- [ ] Big Witt vectors $\mathbb{W}(R)$: the $\lambda$-ring structure; Adams operations
- [ ] Witt vectors as a functor $W: \mathbf{CRing} \to \mathbf{CRing}$; the universal property
- [ ] Witt vectors and $p$-adic numbers: $W(\mathbb{F}_p) \cong \mathbb{Z}_p$
- [ ] Witt vectors over non-commutative rings; the Hesselholt-Madsen construction
- [ ] Thermodynamic Witt vectors: the $\oplus_\beta$ family as a Witt-type deformation of the tropical semiring; Boltzmann weights as Teichmüller representatives
- [ ] $\delta$-rings: a $p$-derivation $\delta: R \to R$ satisfying $\delta(xy) = x^p\delta(y) + y^p\delta(x) + p\delta(x)\delta(y)$; the connection to Frobenius lifts

**Key references:**

- Rabinoff, [*The Theory of Witt Vectors*](https://math.uchicago.edu/~may/TQFT/Witt.pdf) — best expository notes, self-contained
- Hazewinkel, [*Witt Vectors*](https://arxiv.org/abs/0804.3888) — comprehensive reference
- Borger, [*The Basic Geometry of Witt Vectors, I*](https://arxiv.org/abs/0801.1691) — modern perspective via $\lambda$-rings
- Joyal, *$\delta$-anneaux et vecteurs de Witt* — for $\delta$-rings (in French; Buium's *Arithmetic Differential Equations* covers similar ground in English)
