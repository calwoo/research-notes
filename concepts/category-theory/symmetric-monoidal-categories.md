# Symmetric Monoidal Categories

## Table of Contents

- [[#1. Monoidal Categories|1. Monoidal Categories]]
  - [[#1.1 The Bifunctor and Structural Isomorphisms|1.1 The Bifunctor and Structural Isomorphisms]]
  - [[#1.2 The Pentagon and Triangle Axioms|1.2 The Pentagon and Triangle Axioms]]
  - [[#1.3 Strict Monoidal Categories|1.3 Strict Monoidal Categories]]
- [[#2. The Coherence Theorem|2. The Coherence Theorem]]
  - [[#2.1 Statement and Intuition|2.1 Statement and Intuition]]
  - [[#2.2 Strictification|2.2 Strictification]]
  - [[#2.3 Working Consequences|2.3 Working Consequences]]
- [[#3. Examples|3. Examples]]
  - [[#3.1 Cartesian Monoidal Categories|3.1 Cartesian Monoidal Categories]]
  - [[#3.2 The Tensor Product of Abelian Groups|3.2 The Tensor Product of Abelian Groups]]
  - [[#3.3 Vector Spaces over a Field|3.3 Vector Spaces over a Field]]
  - [[#3.4 Chain Complexes|3.4 Chain Complexes]]
  - [[#3.5 Endofunctor Categories|3.5 Endofunctor Categories]]
- [[#4. Braided and Symmetric Monoidal Categories|4. Braided and Symmetric Monoidal Categories]]
  - [[#4.1 The Braiding|4.1 The Braiding]]
  - [[#4.2 The Hexagon Axioms|4.2 The Hexagon Axioms]]
  - [[#4.3 Symmetric Monoidal Categories|4.3 Symmetric Monoidal Categories]]
  - [[#4.4 Braided versus Symmetric: the Braid Group|4.4 Braided versus Symmetric: the Braid Group]]
- [[#5. Monoidal Functors|5. Monoidal Functors]]
  - [[#5.1 Lax, Strong, and Strict Monoidal Functors|5.1 Lax, Strong, and Strict Monoidal Functors]]
  - [[#5.2 Monoidal Natural Transformations|5.2 Monoidal Natural Transformations]]
  - [[#5.3 The 2-Category of Monoidal Categories|5.3 The 2-Category of Monoidal Categories]]
- [[#6. Algebras in a Monoidal Category|6. Algebras in a Monoidal Category]]
  - [[#6.1 Monoid Objects|6.1 Monoid Objects]]
  - [[#6.2 Commutative Monoids in Symmetric Monoidal Categories|6.2 Commutative Monoids in Symmetric Monoidal Categories]]
  - [[#6.3 Classical Structures as Monoid Objects|6.3 Classical Structures as Monoid Objects]]
- [[#7. Duals and Compact Closed Categories|7. Duals and Compact Closed Categories]]
  - [[#7.1 Left and Right Duals|7.1 Left and Right Duals]]
  - [[#7.2 The Zig-Zag Identities|7.2 The Zig-Zag Identities]]
  - [[#7.3 Pivotal and Compact Closed Categories|7.3 Pivotal and Compact Closed Categories]]
  - [[#7.4 Graphical Calculus: a Hint|7.4 Graphical Calculus: a Hint]]
- [[#References|References]]

---

## 1. Monoidal Categories 📐

A *monoidal category* is the categorical distillation of the notion of a "tensor product." Informally, it is a category equipped with a binary operation on objects that is associative and unital — but only *up to coherent natural isomorphism*, not on the nose. This weakening is essential: the categories that arise in practice (modules, chain complexes, sets under Cartesian product) are almost never strictly associative, and forcing strict equality destroys functoriality.

The definition has six data and two axioms.

### 1.1 The Bifunctor and Structural Isomorphisms

**Definition (Monoidal Category).** A *monoidal category* is a tuple $(\mathcal{C}, \otimes, I, \alpha, \lambda, \rho)$ consisting of:

1. A category $\mathcal{C}$.
2. A functor $\otimes : \mathcal{C} \times \mathcal{C} \to \mathcal{C}$, called the *tensor product* or *monoidal product*.
3. An object $I \in \mathcal{C}$, called the *unit object*.
4. A natural isomorphism $\alpha$, called the *associator*, with components

$$\alpha_{A,B,C} : (A \otimes B) \otimes C \xrightarrow{\;\sim\;} A \otimes (B \otimes C),$$

natural in $A, B, C \in \mathcal{C}$.

5. A natural isomorphism $\lambda$, called the *left unitor*, with components

$$\lambda_A : I \otimes A \xrightarrow{\;\sim\;} A,$$

natural in $A \in \mathcal{C}$.

6. A natural isomorphism $\rho$, called the *right unitor*, with components

$$\rho_A : A \otimes I \xrightarrow{\;\sim\;} A,$$

natural in $A \in \mathcal{C}$.

These six data are subject to two coherence axioms given in the next subsection.

> [!NOTE] The bifunctor requirement
> The condition that $\otimes$ is a functor $\mathcal{C} \times \mathcal{C} \to \mathcal{C}$ is not vacuous: it requires, for morphisms $f : A \to A'$ and $g : B \to B'$, a morphism $f \otimes g : A \otimes B \to A' \otimes B'$ satisfying $(f' \circ f) \otimes (g' \circ g) = (f' \otimes g') \circ (f \otimes g)$ and $\mathrm{id}_A \otimes \mathrm{id}_B = \mathrm{id}_{A \otimes B}$. This *interchange law* is what makes $\otimes$ a bifunctor rather than just a function on objects.

### 1.2 The Pentagon and Triangle Axioms

The six pieces of data above are not free: they must satisfy two commutative diagrams. These diagrams are the *only* axioms; Mac Lane's coherence theorem (§2) then guarantees that every other diagram built from $\alpha$, $\lambda$, $\rho$ automatically commutes.

**Axiom 1 (Pentagon Identity).** For all $A, B, C, D \in \mathcal{C}$, the following diagram commutes:

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}[column sep=small, row sep=large]
& ((A \otimes B) \otimes C) \otimes D \arrow[dl, "\alpha_{A\otimes B,C,D}"'] \arrow[dr, "\alpha_{A,B,C} \otimes \mathrm{id}_D"] & \\
(A \otimes B) \otimes (C \otimes D) \arrow[d, "\alpha_{A,B,C\otimes D}"'] & & (A \otimes (B \otimes C)) \otimes D \arrow[d, "\alpha_{A,B\otimes C,D}"] \\
A \otimes (B \otimes (C \otimes D)) & & A \otimes ((B \otimes C) \otimes D) \arrow[ll, "\mathrm{id}_A \otimes \alpha_{B,C,D}"]
\end{tikzcd}
\end{document}
```

This diagram has five vertices (the five ways to parenthesize $A \otimes B \otimes C \otimes D$) and five edges. Its commutativity says that the two paths from $((A \otimes B) \otimes C) \otimes D$ to $A \otimes (B \otimes (C \otimes D))$ — one going left-and-down, the other going right-and-down — agree.

**Axiom 2 (Triangle Identity).** For all $A, B \in \mathcal{C}$, the following diagram commutes:

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}[column sep=large]
(A \otimes I) \otimes B \arrow[rr, "\alpha_{A,I,B}"] \arrow[dr, "\rho_A \otimes \mathrm{id}_B"'] & & A \otimes (I \otimes B) \arrow[dl, "\mathrm{id}_A \otimes \lambda_B"] \\
& A \otimes B &
\end{tikzcd}
\end{document}
```

This says that the two canonical ways of "deleting" the unit $I$ from the middle of $A \otimes I \otimes B$ agree.

> [!WARNING] Why only two axioms suffice
> At first glance it seems one would need infinitely many coherence axioms — one for every diagram. *Surprisingly,* Mac Lane proved that the pentagon and triangle together imply all further diagrams commute. This is the content of the coherence theorem (§2). The pentagon is necessary for four objects; the triangle handles the unit; no further axioms are needed.

> [!EXAMPLE] Checking the pentagon for Set
> In $(\mathbf{Set}, \times, \{*\})$, the associator is $\alpha_{A,B,C}((a,b),c) = (a,(b,c))$ and the unitors are the evident projections. The pentagon commutes because all five parenthesizations of a quadruple $(a,b,c,d)$ of elements are sent, via $\alpha$, to the canonical tuple $(a,b,c,d)$ and all paths agree. No element is created or destroyed; only the nesting of pairs changes.

> [!QUESTION] Exercise 1: Naturality of the Associator
> *This problem establishes that the associator is genuinely natural, not merely a family of isomorphisms.*
>
> > **Prerequisites:** [[#1.1 The Bifunctor and Structural Isomorphisms|1.1 The Bifunctor and Structural Isomorphisms]]
>
> Let $(\mathcal{C}, \otimes, I, \alpha, \lambda, \rho)$ be a monoidal category, and let $f : A \to A'$, $g : B \to B'$, $h : C \to C'$ be morphisms in $\mathcal{C}$. Write out explicitly what naturality of $\alpha$ requires, i.e., state the commutative square that must hold. Then verify it for $(\mathbf{Set}, \times, \{*\})$ using the explicit formula $\alpha_{A,B,C}((a,b),c)=(a,(b,c))$.

> [!TIP]- Solution to Exercise 1
> **Key insight:** Naturality of $\alpha$ means $\alpha$ is a natural isomorphism of functors $\mathcal{C}^3 \to \mathcal{C}$, so the relevant naturality square lives in $\mathcal{C}$.
>
> **Sketch:** Naturality requires the square
> $$\begin{array}{ccc} (A \otimes B) \otimes C & \xrightarrow{\alpha_{A,B,C}} & A \otimes (B \otimes C) \\ (f \otimes g) \otimes h \downarrow & & \downarrow f \otimes (g \otimes h) \\ (A' \otimes B') \otimes C' & \xrightarrow{\alpha_{A',B',C'}} & A' \otimes (B' \otimes C') \end{array}$$
> to commute. For $\mathbf{Set}$: going clockwise, $((a,b),c) \mapsto (a,(b,c)) \mapsto (f(a),(g(b),h(c)))$. Going counterclockwise, $((a,b),c) \mapsto ((f(a),g(b)),h(c)) \mapsto (f(a),(g(b),h(c)))$. These agree.

### 1.3 Strict Monoidal Categories

**Definition (Strict Monoidal Category).** A monoidal category $(\mathcal{C}, \otimes, I, \alpha, \lambda, \rho)$ is *strict* if $\alpha$, $\lambda$, and $\rho$ are all identity morphisms. Equivalently, $(A \otimes B) \otimes C = A \otimes (B \otimes C)$, $I \otimes A = A$, and $A \otimes I = A$ as equalities of objects, not merely isomorphisms.

Strict monoidal categories are conceptually simpler — all parenthesizations of a tensor product are literally equal — but they are rare in nature. The primary examples include:

- The *endofunctor category* $\mathrm{End}(\mathcal{C}) = [\mathcal{C}, \mathcal{C}]$ with composition $\circ$ and unit $\mathrm{Id}_\mathcal{C}$ (see §3.5).
- Monoids regarded as one-object categories.
- The free strict monoidal category on a set of objects.

> [!INFO] Historical note
> Benabou introduced the weakening from strict to non-strict associativity in 1963. Mac Lane's original 1963 paper on monoidal categories already contained the coherence theorem, showing the weakened notion was in a precise sense no weaker for purposes of diagram-chasing.

> [!QUESTION] Exercise 2: The Unit Object is Unique up to Isomorphism
> *This establishes a basic structural fact about monoidal categories analogous to uniqueness of identity elements in monoids.*
>
> > **Prerequisites:** [[#1.1 The Bifunctor and Structural Isomorphisms|1.1 The Bifunctor and Structural Isomorphisms]]
>
> Suppose $(\mathcal{C}, \otimes, I, \alpha, \lambda, \rho)$ and $(\mathcal{C}, \otimes, I', \alpha, \lambda', \rho')$ are two monoidal structures on $\mathcal{C}$ sharing the same $\otimes$ and $\alpha$. Prove that $I \cong I'$.

> [!TIP]- Solution to Exercise 2
> **Key insight:** The isomorphism arises from applying the unitors to $I \otimes I'$.
>
> **Sketch:** Using $\lambda'_{I} : I' \otimes I \xrightarrow{\sim} I$ and $\rho_{I'} : I' \otimes I \xrightarrow{\sim} I'$ (both are natural isomorphisms for their respective unit objects), we get $I' \cong I' \otimes I \cong I$ via the composite $\rho_{I'} \circ (\lambda'_I)^{-1}$ (up to the appropriate associator, which is the identity in $I' \otimes I$). One must verify this composite is indeed an isomorphism using the triangle identity.

---

## 2. The Coherence Theorem 🔑

The pentagon and triangle identities look like a small fragment of all the coherence conditions one could impose. The miracle — proved by Mac Lane in 1963 — is that they are sufficient.

### 2.1 Statement and Intuition

To state the theorem precisely, we need to make "formal diagram built from $\alpha$, $\lambda$, $\rho$" rigorous.

**Definition (Free Monoidal Category).** Let $S$ be a set of *generating objects*. The *free monoidal category* $\mathcal{F}(S)$ on $S$ has:
- Objects: all well-formed words over $S \cup \{I\}$ built by iterating $\otimes$.
- Morphisms: generated by $\alpha$, $\alpha^{-1}$, $\lambda$, $\lambda^{-1}$, $\rho$, $\rho^{-1}$ and their tensor products and composites, modulo the pentagon and triangle identities.

**Theorem (Mac Lane's Coherence Theorem).** *In the free monoidal category $\mathcal{F}(S)$, there is at most one morphism between any two objects.* Equivalently, every diagram in a monoidal category that is built entirely from instances of $\alpha^{\pm 1}$, $\lambda^{\pm 1}$, $\rho^{\pm 1}$ and tensor products and composites thereof commutes.

> [!NOTE] Precise meaning
> The theorem does NOT say every diagram in an arbitrary monoidal category commutes — only those whose morphisms are built formally from the structural isomorphisms. A monoidal category may have many other morphisms that do not commute in every possible diagram.

The intuition: think of a parenthesization of $A_1 \otimes \cdots \otimes A_n$ as a binary tree. The associator rebalances trees. Coherence says: any two paths of rebalancing operations connecting the same two trees yield the same composite. The proof proceeds by reducing every such diagram to a diagram in a free monoidal category and then exhibiting an explicit normal form (the *fully left-associated* bracketing) for each object, showing the reduction maps are well-defined.

### 2.2 Strictification

A stronger consequence of coherence is the *strictification theorem*:

**Theorem (Strictification).** *Every monoidal category $\mathcal{C}$ is monoidally equivalent to a strict monoidal category $\mathcal{C}^{\mathrm{st}}$.*

The construction of $\mathcal{C}^{\mathrm{st}}$ is explicit. Objects of $\mathcal{C}^{\mathrm{st}}$ are finite lists $[A_1, \ldots, A_n]$ of objects of $\mathcal{C}$ (including the empty list $[]$ as the unit). The tensor product of two lists is concatenation: $[A_1, \ldots, A_m] \otimes [B_1, \ldots, B_n] = [A_1, \ldots, A_m, B_1, \ldots, B_n]$. This is strictly associative because list concatenation is strictly associative. Morphisms from $[A_1, \ldots, A_n]$ to $[B_1, \ldots, B_m]$ are morphisms $A_1 \otimes \cdots \otimes A_n \to B_1 \otimes \cdots \otimes B_m$ in $\mathcal{C}$ (where the left and right sides use any fixed parenthesization, with coherence ensuring this is well-defined).

The inclusion functor $i : \mathcal{C} \to \mathcal{C}^{\mathrm{st}}$ sending $A \mapsto [A]$ is a strong monoidal equivalence.

> [!WARNING] Equivalence, not isomorphism
> Strictification gives a *monoidal equivalence*, not a monoidal isomorphism. The categories $\mathcal{C}$ and $\mathcal{C}^{\mathrm{st}}$ are generally not isomorphic as monoidal categories. The point is that for any specific diagram one wants to verify commutes, one may work in the strict model and import the conclusion back to $\mathcal{C}$ via the equivalence.

### 2.3 Working Consequences

In practice, Mac Lane's coherence theorem licenses the following conventions:

1. **Drop parentheses**: Write $A \otimes B \otimes C$ unambiguously, since all parenthesizations are canonically isomorphic and the isomorphisms are coherent.
2. **Drop unit instances**: Write $A$ for $I \otimes A$ or $A \otimes I$, since the unitors provide canonical isomorphisms and these are coherent with the associator.
3. **Reorder structural morphisms freely**: Any composite of $\alpha^{\pm 1}$, $\lambda^{\pm 1}$, $\rho^{\pm 1}$ connecting the same source and target is equal.

**Key conclusion: once the pentagon and triangle are verified, no further coherence work is needed for morphisms built from the structural isomorphisms.**

> [!QUESTION] Exercise 3: A Consequence of Coherence
> *This problem extracts a concrete identity that follows from coherence but is not one of the two axioms.*
>
> > **Prerequisites:** [[#2.1 Statement and Intuition|2.1 Statement and Intuition]]
>
> Using the coherence theorem (or by explicit diagram chase), prove that $\lambda_I = \rho_I : I \otimes I \to I$. That is, the left and right unitors agree on the unit object.

> [!TIP]- Solution to Exercise 3
> **Key insight:** Both $\lambda_I$ and $\rho_I$ are morphisms $I \otimes I \to I$ in the free monoidal category, and coherence implies there is at most one such morphism built from structural isomorphisms.
>
> **Sketch:** More explicitly (following Mac Lane's original proof): apply the triangle identity with $A = I, B = I$ to obtain $(\rho_I \otimes \mathrm{id}_I) = (\mathrm{id}_I \otimes \lambda_I) \circ \alpha_{I,I,I}^{-1} \circ \alpha_{I,I,I}$ after suitable manipulation. Also use that $\rho$ and $\lambda$ are natural and that $\alpha$ satisfies the pentagon. The identity $\lambda_I = \rho_I$ falls out as a consequence.

> [!QUESTION] Exercise 4: Strictification of a One-Object Monoidal Category
> *This problem makes the strictification theorem concrete in the simplest nontrivial case.*
>
> > **Prerequisites:** [[#2.2 Strictification|2.2 Strictification]]
>
> A monoidal category with one object $*$ is exactly a *monoid* (the set of endomorphisms of $*$ with composition as multiplication). What is the strictification $\mathcal{C}^{\mathrm{st}}$ in this case? What does the monoidal equivalence $i : \mathcal{C} \to \mathcal{C}^{\mathrm{st}}$ look like?

> [!TIP]- Solution to Exercise 4
> **Key insight:** List concatenation strictifies the associativity of a monoid.
>
> **Sketch:** Let the monoid be $(M, \cdot, e)$. Then $\mathcal{C}$ has one object $*$ and morphisms $M$ with composition given by multiplication. The strictification $\mathcal{C}^{\mathrm{st}}$ still has one object (the only list-type is $[*, *, \ldots, *]$ for $n \geq 0$, but concatenation of $n$-tuples of $*$ with $m$-tuples gives $(n+m)$-tuples — however, all these map to $*$ in $\mathcal{C}$, so the functor $i$ identifies all lists with $[*]$). Thus $\mathcal{C}^{\mathrm{st}}$ is equivalent to a one-object strict monoidal category, i.e., the free monoid on the elements of $M$ quotiented by the monoid relations — which is just the monoid $(M,\cdot,e)$ again. The strictification adds no new structure for one-object monoidal categories.

---

## 3. Examples 💡

### 3.1 Cartesian Monoidal Categories

**Example (Set, ×, 1).** The category $\mathbf{Set}$ with the Cartesian product $\times$ and the one-element set $\mathbf{1} = \{*\}$ is a monoidal category. Concretely:

- $\otimes = \times$: on objects, $A \times B = \{(a,b) : a \in A, b \in B\}$; on morphisms, $(f \times g)(a,b) = (f(a),g(b))$.
- Unit object: any singleton $\mathbf{1}$.
- Associator: $\alpha_{A,B,C}((a,b),c) = (a,(b,c))$.
- Left unitor: $\lambda_A(*,a) = a$.
- Right unitor: $\rho_A(a,*) = a$.

More generally, any category $\mathcal{C}$ with finite products and a terminal object $\mathbf{1}$ is a *Cartesian monoidal category* with $A \otimes B = A \times B$ and $I = \mathbf{1}$. Examples: $\mathbf{Top}$, $\mathbf{Cat}$, $\mathbf{Grp}$, any topos.

> [!NOTE] Cartesian versus non-Cartesian
> A monoidal structure is *Cartesian* if $\otimes = \times$ (the categorical product). In Cartesian monoidal categories every object carries a diagonal $\Delta_A : A \to A \times A$ and augmentation $\epsilon_A : A \to I$, making every object a comonoid. This extra structure is absent in non-Cartesian examples such as $(\mathbf{Vect}_k, \otimes_k, k)$.

### 3.2 The Tensor Product of Abelian Groups

**Example (Ab, tensor, Z).** The category $\mathbf{Ab}$ of abelian groups with the tensor product $\otimes_\mathbb{Z}$ and unit $\mathbb{Z}$ is a monoidal category. Recall that for abelian groups $A$ and $B$, the tensor product $A \otimes_\mathbb{Z} B$ is the abelian group generated by symbols $a \otimes b$ subject to bilinearity:

$$a \otimes (b + b') = a \otimes b + a \otimes b', \qquad (a + a') \otimes b = a \otimes b + a' \otimes b.$$

The associator $\alpha_{A,B,C} : (A \otimes B) \otimes C \to A \otimes (B \otimes C)$ is defined on generators by $(a \otimes b) \otimes c \mapsto a \otimes (b \otimes c)$ and extended by linearity. The left unitor $\lambda_A : \mathbb{Z} \otimes A \to A$ is $n \otimes a \mapsto na$, and similarly for $\rho$.

This monoidal structure is *closed*: there is an internal hom $[A, B] = \mathrm{Hom}_\mathbb{Z}(A, B)$ satisfying $\mathrm{Hom}(A \otimes B, C) \cong \mathrm{Hom}(A, [B,C])$ naturally.

### 3.3 Vector Spaces over a Field

**Example (Vect_k, tensor, k).** Let $k$ be a field. The category $\mathbf{Vect}_k$ of $k$-vector spaces with the tensor product $\otimes_k$ and unit $k$ is a monoidal category. For finite-dimensional vector spaces, $\dim(V \otimes W) = \dim(V) \cdot \dim(W)$.

This example is fundamental: it is the home of *representation theory* (representations of a group $G$ are monoid objects in a monoidal category of $G$-modules), *linear algebra* (matrix multiplication is composition in the enriched hom), and — with the symmetric monoidal structure added — commutative algebra (commutative algebras over $k$ are commutative monoid objects in $\mathbf{Vect}_k$).

### 3.4 Chain Complexes

**Example (Ch(R), tensor, R[0]).** Let $R$ be a commutative ring. The category $\mathbf{Ch}(R)$ of chain complexes of $R$-modules, with the *tensor product of chain complexes* and unit $R[0]$ (the complex with $R$ in degree $0$ and $0$ elsewhere), is a monoidal category.

The tensor product of complexes $(A_\bullet, d^A)$ and $(B_\bullet, d^B)$ is

$$(A \otimes B)_n = \bigoplus_{p+q=n} A_p \otimes_R B_q$$

with differential $d^{A \otimes B}(a \otimes b) = d^A(a) \otimes b + (-1)^{|a|} a \otimes d^B(b)$, where $|a|$ is the degree of $a$. The sign $(-1)^{|a|}$ is the *Koszul sign rule*, which is necessary for $d^2 = 0$ and for the associator and unitors to be chain maps.

> [!WARNING] The Koszul sign
> The sign $(-1)^{|a|}$ in the tensor differential is not optional. Without it, the differential does not square to zero. The Koszul sign rule is also responsible for the sign in the commutativity isomorphism $A \otimes B \cong B \otimes A$ (see §4.3 for the symmetric monoidal structure on $\mathbf{Ch}(R)$).

### 3.5 Endofunctor Categories

**Example (End(C), composition, Id).** For any category $\mathcal{C}$, the functor category $\mathrm{End}(\mathcal{C}) = [\mathcal{C}, \mathcal{C}]$ of endofunctors is a *strict* monoidal category with:

- Tensor product: functor composition $F \otimes G = F \circ G$.
- Unit: the identity functor $\mathrm{Id}_\mathcal{C}$.
- Associator, unitors: all identity natural transformations (composition of functors is strictly associative and $\mathrm{Id}$ is strictly a unit).

A *monad* on $\mathcal{C}$ is exactly a monoid object in $(\mathrm{End}(\mathcal{C}), \circ, \mathrm{Id})$ (see [[concepts/category-theory/foundations/04-adjoint-functor-theorems-monads|Adjoint Functor Theorems and Monads]]).

> [!QUESTION] Exercise 5: Checking the Triangle Identity in Vect
> *This problem grounds the abstract axioms in the concrete setting of linear algebra.*
>
> > **Prerequisites:** [[#1.2 The Pentagon and Triangle Axioms|1.2 The Pentagon and Triangle Axioms]], [[#3.3 Vector Spaces over a Field|3.3 Vector Spaces over a Field]]
>
> In $(\mathbf{Vect}_k, \otimes_k, k)$, write out explicit formulas for $\alpha_{V,k,W}$, $\rho_V$, and $\lambda_W$ on elementary tensors, and verify the triangle identity: $(\mathrm{id}_V \otimes \lambda_W) \circ \alpha_{V,k,W} = \rho_V \otimes \mathrm{id}_W$.

> [!TIP]- Solution to Exercise 5
> **Key insight:** On elementary tensors, all three structural isomorphisms act by "moving the scalar."
>
> **Sketch:** $\alpha_{V,k,W}((v \otimes c) \otimes w) = v \otimes (c \otimes w)$. Then $(\mathrm{id}_V \otimes \lambda_W)(v \otimes (c \otimes w)) = v \otimes (cw)$. On the other side, $(\rho_V \otimes \mathrm{id}_W)((v \otimes c) \otimes w) = (vc) \otimes w = (cv) \otimes w$. These agree since $cv = vc$ in $V$ ($c \in k$ is a scalar) and the identification $v \otimes (cw) = (cv) \otimes w$ holds by bilinearity of $\otimes_k$.

> [!QUESTION] Exercise 6: The Koszul Sign and d-squared Zero
> *This problem shows why the Koszul sign rule is forced by the requirement that the differential squares to zero on a tensor product.*
>
> > **Prerequisites:** [[#3.4 Chain Complexes|3.4 Chain Complexes]]
>
> Let $(A_\bullet, d^A)$ and $(B_\bullet, d^B)$ be chain complexes of $R$-modules with $d^2 = 0$. Define a differential on $(A \otimes B)_n = \bigoplus_{p+q=n} A_p \otimes B_q$ by $D(a \otimes b) = d^A(a) \otimes b + (-1)^{|a|} a \otimes d^B(b)$. Verify that $D^2 = 0$.

> [!TIP]- Solution to Exercise 6
> **Key insight:** The sign $(-1)^{|a|}$ creates cancellation when $D$ is applied twice.
>
> **Sketch:** $D^2(a \otimes b) = D(d^A a \otimes b + (-1)^{|a|} a \otimes d^B b)$. Expanding:
> $= d^A(d^A a) \otimes b + (-1)^{|a|+1} d^A a \otimes d^B b + (-1)^{|a|} d^A a \otimes d^B b + (-1)^{|a|}(-1)^{|a|} a \otimes d^B(d^B b)$.
> The first and last terms vanish by $d^2=0$, and the middle two terms cancel since they have opposite signs: $(-1)^{|a|+1}+(-1)^{|a|}=0$.

---

## 4. Braided and Symmetric Monoidal Categories 📐

A monoidal category lets us tensor objects together but gives no canonical way to *swap* the two factors. A *braided* monoidal category adds a natural isomorphism $\beta_{A,B}: A \otimes B \to B \otimes A$ subject to coherence. A *symmetric* monoidal category imposes the additional condition that swapping twice returns to the identity.

### 4.1 The Braiding

**Definition (Braided Monoidal Category).** A *braided monoidal category* is a monoidal category $(\mathcal{C}, \otimes, I, \alpha, \lambda, \rho)$ equipped with a natural isomorphism $\beta$, called the *braiding*, with components

$$\beta_{A,B} : A \otimes B \xrightarrow{\;\sim\;} B \otimes A,$$

natural in $A, B \in \mathcal{C}$, subject to the two hexagon axioms below.

Naturality of $\beta$ means that for any morphisms $f : A \to A'$ and $g : B \to B'$, the square

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}
A \otimes B \arrow[r, "\beta_{A,B}"] \arrow[d, "f \otimes g"'] & B \otimes A \arrow[d, "g \otimes f"] \\
A' \otimes B' \arrow[r, "\beta_{A',B'}"] & B' \otimes A'
\end{tikzcd}
\end{document}
```

commutes.

### 4.2 The Hexagon Axioms

The braiding must be compatible with the associator via two *hexagon identities*.

**Axiom (First Hexagon Identity).** For all $A, B, C \in \mathcal{C}$:

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}[column sep=small, row sep=large]
(A \otimes B) \otimes C \arrow[r, "\alpha_{A,B,C}"] \arrow[d, "\beta_{A,B} \otimes \mathrm{id}_C"'] & A \otimes (B \otimes C) \arrow[dr, "\beta_{A,B\otimes C}"] & \\
(B \otimes A) \otimes C \arrow[r, "\alpha_{B,A,C}"] & B \otimes (A \otimes C) \arrow[r, "\mathrm{id}_B \otimes \beta_{A,C}"] & B \otimes (C \otimes A)
\end{tikzcd}
\end{document}
```

**Axiom (Second Hexagon Identity).** For all $A, B, C \in \mathcal{C}$:

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}[column sep=small, row sep=large]
A \otimes (B \otimes C) \arrow[r, "\alpha_{A,B,C}^{-1}"] \arrow[d, "\mathrm{id}_A \otimes \beta_{B,C}"'] & (A \otimes B) \otimes C \arrow[dr, "\beta_{A\otimes B,C}"] & \\
A \otimes (C \otimes B) \arrow[r, "\alpha_{A,C,B}^{-1}"] & (A \otimes C) \otimes B \arrow[r, "\beta_{A,C} \otimes \mathrm{id}_B"] & (C \otimes A) \otimes B
\end{tikzcd}
\end{document}
```

The first hexagon says: braiding $A$ past $B \otimes C$ in one step equals braiding $A$ past $B$ and then past $C$ (with appropriate reassociation). The second is the mirror image.

> [!NOTE] Two hexagons are necessary
> In a symmetric monoidal category (below) one hexagon implies the other. In a braided monoidal category with $\beta_{B,A} \circ \beta_{A,B} \neq \mathrm{id}$, the two hexagons are genuinely independent.

### 4.3 Symmetric Monoidal Categories

**Definition (Symmetric Monoidal Category).** A *symmetric monoidal category* is a braided monoidal category $(\mathcal{C}, \otimes, I, \alpha, \lambda, \rho, \beta)$ such that for all objects $A, B \in \mathcal{C}$:

$$\beta_{B,A} \circ \beta_{A,B} = \mathrm{id}_{A \otimes B}.$$

This condition says: the braiding is its own inverse. Swapping $A \otimes B$ to $B \otimes A$ and then swapping back yields the identity. In a symmetric monoidal category, $\beta_{A,B}^{-1} = \beta_{B,A}$.

> [!EXAMPLE] The symmetry isomorphism in Vect
> In $(\mathbf{Vect}_k, \otimes_k, k)$ the braiding is $\beta_{V,W}(v \otimes w) = w \otimes v$. This satisfies $\beta_{W,V} \circ \beta_{V,W}(v \otimes w) = \beta_{W,V}(w \otimes v) = v \otimes w$. So $\mathbf{Vect}_k$ is symmetric monoidal.

For chain complexes $\mathbf{Ch}(R)$, the symmetry isomorphism includes a Koszul sign: $\beta_{A,B}(a \otimes b) = (-1)^{|a||b|} b \otimes a$. The sign is necessary for $\beta^2 = \mathrm{id}$ at the chain level (since $(-1)^{|a||b|} \cdot (-1)^{|b||a|} = (-1)^{2|a||b|} = 1$) and for the hexagon axioms.

**Coherence for Symmetric Monoidal Categories.** Mac Lane and Kelly proved an analogue of the coherence theorem: *every symmetric monoidal category is equivalent to a permutative category* (a strict symmetric monoidal category). The free symmetric monoidal category on objects $\{A_1, \ldots, A_n\}$ has morphisms that are permutations in $\Sigma_n$ (the symmetric group on $n$ letters), and every diagram built from $\alpha^{\pm 1}$, $\lambda^{\pm 1}$, $\rho^{\pm 1}$, $\beta^{\pm 1}$ commutes.

### 4.4 Braided versus Symmetric: the Braid Group

The distinction between braided and symmetric monoidal categories is captured by the action on $n$-fold tensor products.

In a braided monoidal category, the group acting on $A^{\otimes n}$ (via the braidings $\beta_{A,A}$) is the *braid group* $B_n$, generated by elementary braids $\sigma_i$ (swapping the $i$-th and $(i+1)$-th factor) subject to:
- $\sigma_i \sigma_j = \sigma_j \sigma_i$ for $|i - j| \geq 2$.
- $\sigma_i \sigma_{i+1} \sigma_i = \sigma_{i+1} \sigma_i \sigma_{i+1}$ (braid relation).

In a symmetric monoidal category, the additional condition $\beta^2 = \mathrm{id}$ forces $\sigma_i^2 = \mathrm{id}$, quotienting $B_n$ to the *symmetric group* $\Sigma_n$. The group acting on $A^{\otimes n}$ in a symmetric monoidal category is therefore $\Sigma_n$.

> [!INFO] Topological motivation
> The braid group $B_n$ is the fundamental group of the configuration space of $n$ ordered points in $\mathbb{R}^2$. The symmetric group $\Sigma_n$ is the fundamental group of the space of $n$ *unordered* points. This topological distinction — $\mathbb{R}^2$ vs. $\mathbb{R}^3$ (where loops can be unlinked) — explains why braided but non-symmetric monoidal categories arise naturally in low-dimensional topology (knot invariants, TFTs in dimension $\leq 3$) while symmetric ones govern higher-dimensional algebra.

> [!QUESTION] Exercise 7: The Hexagon Implies Compatibility with the Unit
> *This problem establishes that the braiding automatically satisfies $\beta_{I,A} = \lambda_A \circ \rho_A^{-1}$, a compatibility not listed as an axiom.*
>
> > **Prerequisites:** [[#4.2 The Hexagon Axioms|4.2 The Hexagon Axioms]]
>
> Using the first hexagon identity with $A = I$ and the triangle identity, show that $\beta_{I,A} = \lambda_A \circ \rho_A^{-1}$ (as morphisms $I \otimes A \to A \otimes I$, where one uses $\lambda_A$ and $\rho_A$ for the target identification).

> [!TIP]- Solution to Exercise 7
> **Key insight:** Setting $A = I$ in a hexagon collapses parts of the diagram using the triangle identity.
>
> **Sketch:** In the first hexagon, set $A = I$. The morphism $\beta_{I,B} \otimes \mathrm{id}_C : (I \otimes B) \otimes C \to (B \otimes I) \otimes C$ can be composed with $\rho_B \otimes \mathrm{id}_C$ on the right and $\lambda_B^{-1} \otimes \mathrm{id}_C$ on the left. Using the triangle identity and naturality of $\lambda$ and $\rho$, one derives $\beta_{I,A} = \lambda_A \circ \rho_A^{-1}$. The key calculation uses that $\lambda$ and $\rho$ are natural transformations with $\lambda_I = \rho_I$.

> [!QUESTION] Exercise 8: Koszul Sign and the Symmetry Condition
> *This problem shows that the Koszul sign in Ch(R) is the unique sign making the symmetry condition hold.*
>
> > **Prerequisites:** [[#4.3 Symmetric Monoidal Categories|4.3 Symmetric Monoidal Categories]], [[#3.4 Chain Complexes|3.4 Chain Complexes]]
>
> For chain complexes $A, B \in \mathbf{Ch}(R)$, define $\beta_{A,B}(a \otimes b) = \epsilon(|a|,|b|) \cdot b \otimes a$ where $\epsilon(p,q) \in \{+1,-1\}$ depends on the degrees. What conditions must $\epsilon$ satisfy for $\beta$ to be: (a) a chain map, and (b) satisfy $\beta_{B,A} \circ \beta_{A,B} = \mathrm{id}$? Show that $\epsilon(p,q) = (-1)^{pq}$ satisfies both.

> [!TIP]- Solution to Exercise 8
> **Key insight:** Being a chain map forces the sign to intertwine the Koszul-signed differential; the symmetry condition forces $\epsilon(p,q)\epsilon(q,p)=1$.
>
> **Sketch:** (a) $\beta_{A,B}$ is a chain map iff $\beta \circ d^{A\otimes B} = d^{B\otimes A} \circ \beta$. On $a \otimes b$ of degree $|a|=p$, $|b|=q$: LHS gives $\epsilon(p-1,q)d^A(a) \otimes b + (-1)^p \epsilon(p,q-1) a \otimes d^B(b)$ mapped via $\beta$, yielding terms with $\epsilon(p-1,q)$ and $(-1)^p\epsilon(p,q-1)$. Comparing with the RHS forces $\epsilon(p-1,q) = (-1)^q \epsilon(p,q)$ and $(-1)^p \epsilon(p,q-1) = (-1)^p \epsilon(p,q)$ — satisfied by $\epsilon = (-1)^{pq}$. (b) $\beta_{B,A}\circ\beta_{A,B}(a\otimes b) = \epsilon(p,q)\epsilon(q,p) \cdot a\otimes b$. For identity, need $\epsilon(p,q)\epsilon(q,p)=1$. Since $(-1)^{pq} \cdot (-1)^{qp} = (-1)^{2pq} = 1$, the Koszul sign works.

---

## 5. Monoidal Functors 📐

A functor between monoidal categories should "respect" the monoidal structure. But "respect" can mean several things: strictly preserve it, preserve it up to coherent isomorphism, or merely preserve it up to coherent morphism. These give three levels of monoidal functor.

### 5.1 Lax, Strong, and Strict Monoidal Functors

**Definition (Lax Monoidal Functor).** Let $(\mathcal{C}, \otimes, I)$ and $(\mathcal{D}, \otimes', I')$ be monoidal categories. A *lax monoidal functor* $(F, \varphi, \varphi_0)$ consists of:

1. A functor $F : \mathcal{C} \to \mathcal{D}$.
2. A natural transformation $\varphi$ with components
$$\varphi_{A,B} : F(A) \otimes' F(B) \to F(A \otimes B),$$
natural in $A, B \in \mathcal{C}$.
3. A morphism $\varphi_0 : I' \to F(I)$.

These must satisfy three coherence conditions:

**Associativity coherence:** For all $A, B, C \in \mathcal{C}$, the following diagram commutes:

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}[column sep=large]
(FA \otimes' FB) \otimes' FC \arrow[r, "\varphi_{A,B} \otimes' \mathrm{id}"] \arrow[d, "\alpha'_{FA,FB,FC}"'] & F(A\otimes B) \otimes' FC \arrow[r, "\varphi_{A\otimes B,C}"] & F((A\otimes B)\otimes C) \arrow[d, "F\alpha_{A,B,C}"] \\
FA \otimes' (FB \otimes' FC) \arrow[r, "\mathrm{id} \otimes' \varphi_{B,C}"] & FA \otimes' F(B\otimes C) \arrow[r, "\varphi_{A,B\otimes C}"] & F(A\otimes(B\otimes C))
\end{tikzcd}
\end{document}
```

**Left unit coherence:** For all $A \in \mathcal{C}$:

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}
I' \otimes' FA \arrow[r, "\varphi_0 \otimes' \mathrm{id}"] \arrow[dr, "\lambda'_{FA}"'] & F(I) \otimes' FA \arrow[r, "\varphi_{I,A}"] & F(I \otimes A) \arrow[dl, "F\lambda_A"] \\
& FA &
\end{tikzcd}
\end{document}
```

**Right unit coherence:** For all $A \in \mathcal{C}$:

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}
FA \otimes' I' \arrow[r, "\mathrm{id} \otimes' \varphi_0"] \arrow[dr, "\rho'_{FA}"'] & FA \otimes' F(I) \arrow[r, "\varphi_{A,I}"] & F(A \otimes I) \arrow[dl, "F\rho_A"] \\
& FA &
\end{tikzcd}
\end{document}
```

**Definition (Strong Monoidal Functor).** A lax monoidal functor $(F, \varphi, \varphi_0)$ is *strong* (or *monoidal* without further qualification) if $\varphi_{A,B}$ is an isomorphism for all $A, B$ and $\varphi_0$ is an isomorphism.

**Definition (Strict Monoidal Functor).** A lax monoidal functor is *strict* if $\varphi_{A,B} = \mathrm{id}_{FA \otimes' FB}$ and $\varphi_0 = \mathrm{id}_{I'}$ (which forces $F(A \otimes B) = F(A) \otimes' F(B)$ and $F(I) = I'$ on the nose).

> [!EXAMPLE] The forgetful functor as a lax monoidal functor
> The forgetful functor $U : \mathbf{Ring} \to \mathbf{Ab}$ sending a ring $R$ to its underlying abelian group $(R,+)$ is lax monoidal: the comparison map $\varphi_{R,S} : U(R) \otimes_\mathbb{Z} U(S) \to U(R \otimes_\mathbb{Z} S)$ is the canonical map from the free bilinear product to the tensor product, which is not an isomorphism in general (the tensor product of rings $R \otimes_\mathbb{Z} S$ as rings uses the same generators but the ring structure is richer).

> [!EXAMPLE] The free commutative algebra functor as a strong monoidal functor
> The symmetric algebra functor $\mathrm{Sym} : \mathbf{Vect}_k \to \mathbf{CAlg}_k$ (sending a vector space $V$ to the polynomial algebra $k[V] = \bigoplus_{n \geq 0} V^{\otimes n}/\Sigma_n$) is strong monoidal: $\mathrm{Sym}(V \oplus W) \cong \mathrm{Sym}(V) \otimes_k \mathrm{Sym}(W)$ as $k$-algebras. Here the tensor product on $\mathbf{CAlg}_k$ is the coproduct (pushout) of commutative algebras.

### 5.2 Monoidal Natural Transformations

**Definition (Monoidal Natural Transformation).** Let $(F, \varphi, \varphi_0)$ and $(G, \psi, \psi_0)$ be lax monoidal functors $\mathcal{C} \to \mathcal{D}$. A *monoidal natural transformation* $\eta : F \Rightarrow G$ is a natural transformation such that:

1. For all $A, B \in \mathcal{C}$: $\psi_{A,B} \circ (\eta_A \otimes' \eta_B) = \eta_{A \otimes B} \circ \varphi_{A,B}$.
2. $\psi_0 = \eta_I \circ \varphi_0$.

In diagrams, condition 1 is:

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}
FA \otimes' FB \arrow[r, "\varphi_{A,B}"] \arrow[d, "\eta_A \otimes' \eta_B"'] & F(A \otimes B) \arrow[d, "\eta_{A\otimes B}"] \\
GA \otimes' GB \arrow[r, "\psi_{A,B}"] & G(A \otimes B)
\end{tikzcd}
\end{document}
```

### 5.3 The 2-Category of Monoidal Categories

Monoidal categories, lax monoidal functors, and monoidal natural transformations organize into a *2-category* $\mathbf{MonCat}$. This 2-category has:
- Objects: monoidal categories.
- 1-morphisms: lax monoidal functors.
- 2-morphisms: monoidal natural transformations.

One similarly has $\mathbf{MonCat}_{\mathrm{str}}$ (strong monoidal functors) and $\mathbf{SymMonCat}$ (symmetric monoidal categories with symmetric monoidal functors).

**Proposition.** *A monoidal equivalence* (an equivalence in $\mathbf{MonCat}_{\mathrm{str}}$, i.e., a strong monoidal functor with a strong monoidal inverse up to monoidal natural isomorphism) is the appropriate notion of "isomorphism" of monoidal categories.

> [!QUESTION] Exercise 9: Composition of Lax Monoidal Functors
> *This problem shows that the composite of two lax monoidal functors is again lax monoidal, establishing that $\mathbf{MonCat}$ is well-defined.*
>
> > **Prerequisites:** [[#5.1 Lax, Strong, and Strict Monoidal Functors|5.1 Lax, Strong, and Strict Monoidal Functors]]
>
> Let $(F, \varphi, \varphi_0) : \mathcal{C} \to \mathcal{D}$ and $(G, \psi, \psi_0) : \mathcal{D} \to \mathcal{E}$ be lax monoidal functors. Define the composite comparison maps $\chi_{A,B} : G(FA) \otimes'' G(FB) \to GF(A \otimes B)$ and $\chi_0 : I'' \to GF(I)$. Write out the formulas and verify the associativity coherence condition.

> [!TIP]- Solution to Exercise 9
> **Key insight:** The composite comparison is defined by first applying $G$'s comparison to $GFA \otimes'' GFB$, then applying $G$ to $F$'s comparison.
>
> **Sketch:** Define $\chi_{A,B} = G(\varphi_{A,B}) \circ \psi_{FA,FB}$ and $\chi_0 = G(\varphi_0) \circ \psi_0$. For associativity coherence: the relevant diagram expands to a cube whose commutativity follows by pasting the associativity coherence squares for $\psi$ (outer face) and for $\varphi$ (inner face after applying $G$), using functoriality of $G$ to identify $G(\varphi_{A,B} \circ \text{something})$ with $G(\varphi_{A,B}) \circ G(\text{something})$.

> [!QUESTION] Exercise 10: Strong Monoidal Functors Preserve Dualizable Objects
> *This problem previews the section on duals by showing strong monoidal functors send dualizable objects to dualizable objects.*
>
> > **Prerequisites:** [[#5.1 Lax, Strong, and Strict Monoidal Functors|5.1 Lax, Strong, and Strict Monoidal Functors]]
>
> Suppose $(F, \varphi, \varphi_0) : \mathcal{C} \to \mathcal{D}$ is a strong monoidal functor, and $A \in \mathcal{C}$ is dualizable with dual $A^*$ and unit/counit $\eta : I \to A \otimes A^*$, $\varepsilon : A^* \otimes A \to I$. Show that $F(A)$ is dualizable in $\mathcal{D}$ with dual $F(A^*)$.

> [!TIP]- Solution to Exercise 10
> **Key insight:** Transport the unit and counit along the isomorphisms $\varphi_{A,A^*}$ and $\varphi_0^{-1}$.
>
> **Sketch:** Define $\tilde\eta : I' \xrightarrow{\psi_0^{-1}} F(I) \xrightarrow{F\eta} F(A \otimes A^*) \xrightarrow{\varphi_{A,A^*}^{-1}} F(A) \otimes' F(A^*)$ and similarly $\tilde\varepsilon$. The zig-zag identities for $(F(A), F(A^*), \tilde\eta, \tilde\varepsilon)$ follow from functoriality of $F$ and the coherence conditions on $\varphi$.

---

## 6. Algebras in a Monoidal Category 📐

The notion of an "algebra" or "monoid" can be internalized to any monoidal category. This single abstraction simultaneously captures associative algebras, rings, monoids, operads (of a sort), and monads.

### 6.1 Monoid Objects

**Definition (Monoid Object).** A *monoid object* (or *algebra object*, or *internal monoid*) in a monoidal category $(\mathcal{C}, \otimes, I)$ is a triple $(M, \mu, \eta)$ where:

- $M \in \mathcal{C}$ is an object.
- $\mu : M \otimes M \to M$ is a morphism, called *multiplication*.
- $\eta : I \to M$ is a morphism, called the *unit*.

These must satisfy:

**Associativity:** The following diagram commutes:

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}[column sep=large]
(M \otimes M) \otimes M \arrow[r, "\alpha_{M,M,M}"] \arrow[d, "\mu \otimes \mathrm{id}_M"'] & M \otimes (M \otimes M) \arrow[d, "\mathrm{id}_M \otimes \mu"] \\
M \otimes M \arrow[dr, "\mu"'] & M \otimes M \arrow[d, "\mu"] \\
& M
\end{tikzcd}
\end{document}
```

**Left unit law:** The following diagram commutes:

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}
I \otimes M \arrow[r, "\eta \otimes \mathrm{id}_M"] \arrow[dr, "\lambda_M"'] & M \otimes M \arrow[d, "\mu"] \\
& M
\end{tikzcd}
\end{document}
```

**Right unit law:** The following diagram commutes:

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}
M \otimes I \arrow[r, "\mathrm{id}_M \otimes \eta"] \arrow[dr, "\rho_M"'] & M \otimes M \arrow[d, "\mu"] \\
& M
\end{tikzcd}
\end{document}
```

**Definition (Monoid Morphism).** A *morphism of monoid objects* $f : (M, \mu, \eta) \to (M', \mu', \eta')$ is a morphism $f : M \to M'$ in $\mathcal{C}$ such that $f \circ \mu = \mu' \circ (f \otimes f)$ and $f \circ \eta = \eta'$.

The category of monoid objects in $(\mathcal{C}, \otimes, I)$ is denoted $\mathbf{Mon}(\mathcal{C})$.

> [!EXAMPLE] Monoids in Set and Ab
> - In $(\mathbf{Set}, \times, \mathbf{1})$: a monoid object is a monoid in the classical sense — a set $M$ with associative binary operation $\mu : M \times M \to M$ and unit element $e = \eta(*) \in M$.
> - In $(\mathbf{Ab}, \otimes_\mathbb{Z}, \mathbb{Z})$: a monoid object is a *ring* (not necessarily commutative). The multiplication $\mu : R \otimes_\mathbb{Z} R \to R$ is the ring multiplication extended by bilinearity, and the unit $\eta : \mathbb{Z} \to R$ sends $1 \mapsto 1_R$.

### 6.2 Commutative Monoids in Symmetric Monoidal Categories

In a symmetric monoidal category, there is a canonical isomorphism $\beta_{M,M} : M \otimes M \to M \otimes M$. This permits a notion of commutativity.

**Definition (Commutative Monoid Object).** A monoid object $(M, \mu, \eta)$ in a symmetric monoidal category $(\mathcal{C}, \otimes, I, \beta)$ is *commutative* if

$$\mu \circ \beta_{M,M} = \mu.$$

This says the multiplication is unchanged by swapping the two inputs.

> [!EXAMPLE] Commutative algebras in Vect
> In $(\mathbf{Vect}_k, \otimes_k, k, \beta)$, a commutative monoid object is a *commutative $k$-algebra* in the classical sense: an associative unital $k$-algebra $A$ with $ab = ba$ for all $a, b \in A$. The symmetry condition $\mu \circ \beta_{A,A} = \mu$ translates to $\mu(b \otimes a) = \mu(a \otimes b)$, i.e., $ba = ab$.

> [!NOTE] Why symmetry is needed
> Without the symmetry isomorphism $\beta_{M,M}$, there is no canonical morphism $M \otimes M \to M \otimes M$ to compose with $\mu$. In a merely braided monoidal category, one gets two a priori different notions of "commutativity" ($\mu \circ \beta = \mu$ and $\mu \circ \beta^{-1} = \mu$), which collapse in the symmetric case since $\beta^{-1} = \beta_{M,M}^{-1} = \beta_{M,M}$ (using $\beta^2 = \mathrm{id}$ and renaming indices).

### 6.3 Classical Structures as Monoid Objects

The following table records classical algebraic structures as monoid objects:

| Monoidal category | Monoid objects | Commutative monoid objects |
|---|---|---|
| $(\mathbf{Set}, \times, \mathbf{1})$ | Monoids | Commutative monoids |
| $(\mathbf{Ab}, \otimes_\mathbb{Z}, \mathbb{Z})$ | Rings (assoc., unital) | Commutative rings |
| $(\mathbf{Vect}_k, \otimes_k, k)$ | Associative $k$-algebras | Commutative $k$-algebras |
| $(\mathbf{Ch}(R), \otimes, R[0])$ | DG algebras | Commutative DG algebras |
| $(\mathrm{End}(\mathcal{C}), \circ, \mathrm{Id})$ | Monads on $\mathcal{C}$ | — |
| $(\mathbf{Top}, \times, \mathbf{1})$ | Topological monoids | Commutative top. monoids |

A *DG algebra* (differential graded algebra) is an associative algebra internal to $(\mathbf{Ch}(R), \otimes, R[0])$: it is a chain complex $A_\bullet$ equipped with a chain map $\mu : A \otimes A \to A$ that is associative and unital. A commutative DG algebra satisfies $\mu(a \otimes b) = (-1)^{|a||b|}\mu(b \otimes a)$ (the Koszul sign appears because $\beta_{A,A}$ in $\mathbf{Ch}(R)$ carries a sign).

> [!QUESTION] Exercise 11: Monads are Monoid Objects
> *This problem makes the fundamental connection between monads and monoid objects precise.*
>
> > **Prerequisites:** [[#6.1 Monoid Objects|6.1 Monoid Objects]], [[#3.5 Endofunctor Categories|3.5 Endofunctor Categories]]
>
> Let $T : \mathcal{C} \to \mathcal{C}$ be a monad with multiplication $\mu : T^2 \Rightarrow T$ and unit $\eta : \mathrm{Id} \Rightarrow T$ (in the sense of [[concepts/category-theory/foundations/04-adjoint-functor-theorems-monads|Adjoint Functor Theorems and Monads]]). Show explicitly that $(T, \mu, \eta)$ is a monoid object in $(\mathrm{End}(\mathcal{C}), \circ, \mathrm{Id})$ by matching the monad axioms against the monoid object axioms.

> [!TIP]- Solution to Exercise 11
> **Key insight:** Composition of functors is the tensor product; the monad axioms are exactly the monoid object axioms in this monoidal category.
>
> **Sketch:** The monad multiplication $\mu : T \circ T \Rightarrow T$ corresponds to the monoid multiplication $\mu : M \otimes M \to M$ with $M = T$ and $\otimes = \circ$. The monad unit $\eta : \mathrm{Id} \Rightarrow T$ corresponds to $\eta : I \to M$ with $I = \mathrm{Id}$. Associativity of the monad ($\mu \circ T\mu = \mu \circ \mu T$) is exactly the associativity diagram for a monoid object, since $T\mu = \mathrm{id}_T \circ \mu$ and $\mu T = \mu \circ \mathrm{id}_T$ in the strict monoidal category $\mathrm{End}(\mathcal{C})$ (no associator needed). Similarly the unit laws for the monad match the unit law diagrams.

> [!QUESTION] Exercise 12: The Category of Monoid Objects
> *This problem establishes that monoid morphisms compose and form a category.*
>
> > **Prerequisites:** [[#6.1 Monoid Objects|6.1 Monoid Objects]]
>
> (a) Show that the identity morphism $\mathrm{id}_M : M \to M$ is a monoid morphism $(M,\mu,\eta) \to (M,\mu,\eta)$. (b) Show that the composite of two monoid morphisms $f : M \to M'$ and $g : M' \to M''$ is a monoid morphism $gf : M \to M''$.

> [!TIP]- Solution to Exercise 12
> **Key insight:** Both conditions (compatibility with $\mu$ and with $\eta$) are preserved under composition.
>
> **Sketch:** (a) $\mathrm{id}_M \circ \mu = \mu = \mu \circ (\mathrm{id}_M \otimes \mathrm{id}_M)$ since $\mathrm{id} \otimes \mathrm{id} = \mathrm{id}$ by bifunctoriality. Similarly for $\eta$. (b) $(gf)\circ\mu = g\circ(f\circ\mu) = g\circ(\mu'\circ(f\otimes f)) = (g\circ\mu')\circ(f\otimes f) = (\mu''\circ(g\otimes g))\circ(f\otimes f) = \mu''\circ((gf)\otimes(gf))$, using that $g\otimes g$ and $f\otimes f$ compose to $(gf)\otimes(gf)$ by bifunctoriality.

> [!QUESTION] Exercise 13: Commutativity and the Symmetry
> *This problem shows that the commutativity condition for a monoid object genuinely requires the symmetric monoidal structure.*
>
> > **Prerequisites:** [[#6.2 Commutative Monoids in Symmetric Monoidal Categories|6.2 Commutative Monoids in Symmetric Monoidal Categories]]
>
> Let $(A, \mu, \eta)$ be a commutative monoid object in a symmetric monoidal category $(\mathcal{C}, \otimes, I, \beta)$. Show that the condition $\mu \circ \beta_{A,A} = \mu$ implies $\mu \circ \beta_{A,A}^{-1} = \mu$ as well. (Thus both braidings give the same condition in the symmetric case.)

> [!TIP]- Solution to Exercise 13
> **Key insight:** In a symmetric monoidal category $\beta_{A,A}^{-1} = \beta_{A,A}$.
>
> **Sketch:** By the symmetry condition $\beta_{B,A} \circ \beta_{A,B} = \mathrm{id}_{A\otimes B}$, setting $A=B$ gives $\beta_{A,A}^2 = \mathrm{id}_{A\otimes A}$, hence $\beta_{A,A}^{-1} = \beta_{A,A}$. Therefore $\mu \circ \beta_{A,A}^{-1} = \mu \circ \beta_{A,A} = \mu$.

---

## 7. Duals and Compact Closed Categories 🔑

In a monoidal category, some objects admit "duals" — objects that interact with them via unit and counit morphisms satisfying zig-zag identities. In finite-dimensional linear algebra, every vector space is dualizable, with dual $V^* = \mathrm{Hom}(V,k)$. Compact closed categories are those where every object is dualizable.

### 7.1 Left and Right Duals

In a general (non-symmetric) monoidal category, left and right duals are distinct.

**Definition (Right Dual).** Let $(\mathcal{C}, \otimes, I)$ be a monoidal category and $A \in \mathcal{C}$. A *right dual* of $A$ is an object $A^*$ together with morphisms

$$\eta : I \to A \otimes A^* \qquad \text{(unit, or coevaluation)}$$
$$\varepsilon : A^* \otimes A \to I \qquad \text{(counit, or evaluation)}$$

satisfying the *zig-zag identities* (see §7.2).

**Definition (Left Dual).** A *left dual* of $A$ is an object ${}^*A$ together with morphisms

$$\eta' : I \to {}^*A \otimes A \qquad \text{(unit)}$$
$$\varepsilon' : A \otimes {}^*A \to I \qquad \text{(counit)}$$

satisfying the zig-zag identities.

> [!NOTE] Left versus right in non-symmetric categories
> In a symmetric monoidal category, if $A^*$ is a right dual then it is also a left dual (via the braiding). In a merely braided monoidal category, the two notions can differ. In the fully general monoidal case, an object can have a right dual but no left dual, or vice versa.

### 7.2 The Zig-Zag Identities

The unit $\eta$ and counit $\varepsilon$ of a right duality $(A, A^*, \eta, \varepsilon)$ must satisfy the *zig-zag identities* (also called *snake identities* or *triangle identities for duals*):

**First zig-zag (for $A$):**

$$(\mathrm{id}_A \otimes \varepsilon) \circ \alpha_{A,A^*,A}^{-1} \circ (\eta \otimes \mathrm{id}_A) = \mathrm{id}_A$$

using left/right unitors to identify $I \otimes A \cong A \cong A \otimes I$.

**Second zig-zag (for $A^*$):**

$$(\varepsilon \otimes \mathrm{id}_{A^*}) \circ \alpha_{A^*,A,A^*} \circ (\mathrm{id}_{A^*} \otimes \eta) = \mathrm{id}_{A^*}$$

using unitors similarly.

In a strict monoidal category (dropping the unitors for readability), these become:

$$(\mathrm{id}_A \otimes \varepsilon) \circ (\eta \otimes \mathrm{id}_A) = \mathrm{id}_A, \qquad (\varepsilon \otimes \mathrm{id}_{A^*}) \circ (\mathrm{id}_{A^*} \otimes \eta) = \mathrm{id}_{A^*}.$$

> [!EXAMPLE] Duality in Vect_k
> In $(\mathbf{Vect}_k, \otimes_k, k)$ with a finite-dimensional vector space $V$, the dual is $V^* = \mathrm{Hom}_k(V,k)$. Fix a basis $\{e_i\}$ of $V$ with dual basis $\{e_i^*\}$.
> - Coevaluation: $\eta : k \to V \otimes_k V^*$, $1 \mapsto \sum_i e_i \otimes e_i^*$.
> - Evaluation: $\varepsilon : V^* \otimes_k V \to k$, $f \otimes v \mapsto f(v)$.
>
> First zig-zag: $(\mathrm{id}_V \otimes \varepsilon)(\eta \otimes \mathrm{id}_V)(v) = (\mathrm{id}_V \otimes \varepsilon)(\sum_i e_i \otimes e_i^* \otimes v) = \sum_i e_i \cdot e_i^*(v) = v$. ✓
>
> Second zig-zag: $(\varepsilon \otimes \mathrm{id}_{V^*})(\mathrm{id}_{V^*} \otimes \eta)(f) = (\varepsilon \otimes \mathrm{id}_{V^*})(f \otimes \sum_i e_i \otimes e_i^*) = \sum_i f(e_i) e_i^* = f$ (by definition of the dual basis). ✓

**Proposition.** *If a right dual $(A^*, \eta, \varepsilon)$ exists, it is unique up to unique isomorphism.*

*Proof sketch:* Suppose $(B, \eta_B, \varepsilon_B)$ is another right dual. Define $f : A^* \to B$ by $f = \lambda_B \circ (\varepsilon_B \otimes \mathrm{id}_{A^*}) \circ (\mathrm{id}_{A^*} \otimes \eta)$ and $g : B \to A^*$ similarly. The zig-zag identities imply $g \circ f = \mathrm{id}_{A^*}$ and $f \circ g = \mathrm{id}_B$. $\square$

### 7.3 Pivotal and Compact Closed Categories

**Definition (Rigid Monoidal Category).** A monoidal category is *left rigid* (resp. *right rigid*) if every object has a left dual (resp. right dual). It is *rigid* (or *autonomous*) if it is both.

**Definition (Compact Closed Category).** A *compact closed category* is a symmetric monoidal category in which every object has a dual (equivalently, left and right duals coincide in the symmetric case).

The following categories are compact closed:

| Category | Dual of $A$ | Evaluation $\varepsilon$ |
|---|---|---|
| $\mathbf{fdVect}_k$ | $A^* = \mathrm{Hom}(A,k)$ | $f \otimes v \mapsto f(v)$ |
| $\mathbf{fdRep}(G)$ | Contragredient representation | canonical pairing |
| $\mathbf{Rel}$ (sets and relations) | $A^* = A$ | diagonal relation |
| Cobordism category $n\mathbf{Cob}$ | same manifold, reversed orientation | pair-of-pants cobordism |

**Definition (Pivotal Category).** A *pivotal category* is a rigid monoidal category equipped with a monoidal natural isomorphism $A \cong A^{**}$ (the double dual isomorphism). In a pivotal category, one can define a *trace* for endomorphisms.

> [!INFO] Connection to topological field theory
> Compact closed categories are the categorical structure underlying *topological quantum field theory* (TQFT). Atiyah's axioms for a $d$-dimensional TQFT define it as a symmetric monoidal functor $Z : d\mathbf{Cob} \to \mathbf{Vect}_k$. The compactness of $d\mathbf{Cob}$ (every manifold has a dual: itself with reversed orientation) implies that the functor values $Z(M)$ are finite-dimensional and $Z(M^*)$ is the dual of $Z(M)$.

### 7.4 Graphical Calculus: a Hint

Compact closed categories admit a *graphical calculus* (also called *string diagrams*) in which:

- Objects $A$ are represented by oriented *strings* (wires), with $A^*$ by the same string oriented oppositely.
- Morphisms $f : A \to B$ are represented by *boxes* with input wire $A$ at the bottom and output wire $B$ at the top.
- The tensor product is horizontal juxtaposition of diagrams.
- Composition is vertical stacking.
- The unit $\eta : I \to A \otimes A^*$ is a *cup* (a U-shaped wire).
- The counit $\varepsilon : A^* \otimes A \to I$ is a *cap* (an inverted U-shaped wire).

The zig-zag identities say exactly that a cup followed by a cap (forming a zig-zag in the string) can be "straightened" to a straight wire. This graphical intuition is not merely heuristic — by a theorem of Joyal and Street, the graphical calculus is *sound and complete*: a diagram of string diagrams commutes if and only if the corresponding equation of morphisms holds in every compact closed category.

> [!WARNING] Graphical calculus is not informal
> String diagrams have a precise mathematical formalization as a form of *planar isotopy* of diagrams. The theorem of Joyal–Street (1991) shows that the category of string diagrams is the *free* compact closed category, making the graphical calculus an exact syntactic presentation of the algebra of compact closed categories.

> [!QUESTION] Exercise 14: Duals and Tensor Products
> *This problem shows that the dual of a tensor product is the tensor product of duals (in reverse order), a key structural fact.*
>
> > **Prerequisites:** [[#7.1 Left and Right Duals|7.1 Left and Right Duals]], [[#7.2 The Zig-Zag Identities|7.2 The Zig-Zag Identities]]
>
> Let $(\mathcal{C}, \otimes, I)$ be a monoidal category, and suppose $A$ has right dual $(A^*, \eta_A, \varepsilon_A)$ and $B$ has right dual $(B^*, \eta_B, \varepsilon_B)$. Construct a right dual for $A \otimes B$, i.e., exhibit an object and unit/counit morphisms satisfying the zig-zag identities. (Hint: the dual should be $B^* \otimes A^*$.)

> [!TIP]- Solution to Exercise 14
> **Key insight:** The dual of a tensor product reverses order, mirroring $(AB)^T = B^T A^T$ in linear algebra.
>
> **Sketch:** Set $(A \otimes B)^* = B^* \otimes A^*$. Define:
> - Unit: $\eta : I \xrightarrow{\eta_A} A \otimes A^* \xrightarrow{\mathrm{id}_A \otimes \eta_B \otimes \mathrm{id}_{A^*}} A \otimes (B \otimes B^*) \otimes A^*$, then reassociate to $A \otimes B \otimes B^* \otimes A^* = (A\otimes B) \otimes (B^* \otimes A^*)$.
> - Counit: $\varepsilon : (B^* \otimes A^*) \otimes (A \otimes B) \to B^* \otimes (A^* \otimes A) \otimes B \xrightarrow{\mathrm{id}_{B^*} \otimes \varepsilon_A \otimes \mathrm{id}_B} B^* \otimes I \otimes B \cong B^* \otimes B \xrightarrow{\varepsilon_B} I$.
> The zig-zag identities for $(A\otimes B, B^*\otimes A^*)$ follow from the zig-zag identities for each of $(A,A^*)$ and $(B,B^*)$ by chasing the diagram.

> [!QUESTION] Exercise 15: The Trace in a Compact Closed Category
> *This problem introduces the categorical trace, a key tool in representation theory and TQFT.*
>
> > **Prerequisites:** [[#7.3 Pivotal and Compact Closed Categories|7.3 Pivotal and Compact Closed Categories]]
>
> In a compact closed category $(\mathcal{C}, \otimes, I, \beta)$, define the *trace* of an endomorphism $f : A \to A$ as the composite
> $$\mathrm{tr}(f) : I \xrightarrow{\eta} A \otimes A^* \xrightarrow{f \otimes \mathrm{id}_{A^*}} A \otimes A^* \xrightarrow{\beta_{A,A^*}} A^* \otimes A \xrightarrow{\varepsilon} I.$$
> Compute $\mathrm{tr}(\mathrm{id}_A)$ in $(\mathbf{fdVect}_k, \otimes_k, k)$ and show it equals $\dim(A)$ (viewed as an element of $k$ via the isomorphism $\mathrm{End}(k) \cong k$).

> [!TIP]- Solution to Exercise 15
> **Key insight:** The trace of the identity gives the dimension; this is the categorical shadow of the classical formula $\mathrm{tr}(I_n) = n$.
>
> **Sketch:** In $\mathbf{fdVect}_k$ with basis $\{e_i\}$ for $A$ and dual basis $\{e_i^*\}$: $\eta(1) = \sum_i e_i \otimes e_i^*$. Apply $\mathrm{id}_A \otimes \mathrm{id}_{A^*}$: still $\sum_i e_i \otimes e_i^*$. Apply $\beta_{A,A^*}$: $\sum_i e_i^* \otimes e_i$. Apply $\varepsilon : e_i^* \otimes e_j \mapsto e_i^*(e_j) = \delta_{ij}$: result is $\sum_i e_i^*(e_i) = \sum_i 1 = n = \dim(A)$ in $k$. Thus $\mathrm{tr}(\mathrm{id}_A) = \dim(A) \cdot 1_k$.

---

## Mathematical Development Exercises (continued)

> [!QUESTION] Exercise 16: Lax Monoidal Functors and Monoid Objects
> *This problem establishes the key functoriality property: lax monoidal functors send monoid objects to monoid objects.*
>
> > **Prerequisites:** [[#5.1 Lax, Strong, and Strict Monoidal Functors|5.1 Lax, Strong, and Strict Monoidal Functors]], [[#6.1 Monoid Objects|6.1 Monoid Objects]]
>
> Let $(F, \varphi, \varphi_0) : (\mathcal{C}, \otimes, I) \to (\mathcal{D}, \otimes', I')$ be a lax monoidal functor, and let $(M, \mu, \eta)$ be a monoid object in $\mathcal{C}$. Define a monoid structure on $F(M)$ using $F$, $\varphi$, and $\varphi_0$. Verify the associativity axiom for the resulting monoid object in $\mathcal{D}$.

> [!TIP]- Solution to Exercise 16
> **Key insight:** The monoid structure on $F(M)$ is obtained by transporting $\mu$ and $\eta$ along the comparison maps.
>
> **Sketch:** Define $\tilde\mu = F(\mu) \circ \varphi_{M,M} : F(M) \otimes' F(M) \to F(M)$ and $\tilde\eta = F(\eta) \circ \varphi_0 : I' \to F(M)$. For associativity: the diagram for $(F(M), \tilde\mu, \tilde\eta)$ factors through the associativity coherence square for $\varphi$ (the naturality square relating $\varphi_{M\otimes M,M} \circ (\varphi_{M,M} \otimes' \mathrm{id})$ and $\varphi_{M,M\otimes M} \circ (\mathrm{id} \otimes' \varphi_{M,M})$, which commutes by the lax monoidal axiom), followed by the image under $F$ of the associativity diagram for $(M,\mu,\eta)$, which commutes by assumption.

> [!QUESTION] Exercise 17: The Pentagon from Four-Object Associativity
> *This problem shows that the pentagon identity is the minimal axiom capturing four-object associativity.*
>
> > **Prerequisites:** [[#1.2 The Pentagon and Triangle Axioms|1.2 The Pentagon and Triangle Axioms]], [[#2.1 Statement and Intuition|2.1 Statement and Intuition]]
>
> There are exactly five distinct fully parenthesized products of four objects $A,B,C,D$ (corresponding to the five binary trees with four leaves). List all five. Then write down the five edges of the pentagon as instances of $\alpha$ (with appropriate arguments), and verify that the pentagon indeed has five edges, one for each adjacent pair of parenthesizations connected by a single application of $\alpha$.

> [!TIP]- Solution to Exercise 17
> **Key insight:** The five parenthesizations correspond to binary trees; adjacent ones differ by one application of $\alpha$.
>
> **Sketch:** The five parenthesizations are: $((AB)C)D$, $(A(BC))D$, $A((BC)D)$, $A(B(CD))$, $(AB)(CD)$. Edges: $((AB)C)D \xrightarrow{\alpha_{AB,C,D}} (AB)(CD)$ (no, this regroups $((AB)C)D \to (AB)(CD)$... actually the five edges of the pentagon are: (1) $((AB)C)D \xrightarrow{\alpha_{A,B,C}\otimes\mathrm{id}_D} (A(BC))D$, (2) $(A(BC))D \xrightarrow{\alpha_{A,BC,D}} A((BC)D)$, (3) $A((BC)D) \xrightarrow{\mathrm{id}_A\otimes\alpha_{B,C,D}} A(B(CD))$, (4) $((AB)C)D \xrightarrow{\alpha_{AB,C,D}} (AB)(CD) \xrightarrow{\alpha_{A,B,CD}} A(B(CD))$. The five nodes and five edges correspond exactly to the Stasheff pentagon $K_4$ — one edge per single-step reassociation.

> [!QUESTION] Exercise 18: Rigidity and Closed Structure
> *This problem connects duality to the closed monoidal structure, showing that right duals give a right adjoint to tensoring.*
>
> > **Prerequisites:** [[#7.1 Left and Right Duals|7.1 Left and Right Duals]], [[#7.2 The Zig-Zag Identities|7.2 The Zig-Zag Identities]]
>
> In a right rigid monoidal category, show that the functor $(-) \otimes A$ is left adjoint to $A^* \otimes (-)$. That is, construct a natural bijection $\mathcal{C}(B \otimes A, C) \cong \mathcal{C}(B, A^* \otimes C)$ using the unit $\eta : I \to A \otimes A^*$ and counit $\varepsilon : A^* \otimes A \to I$.

> [!TIP]- Solution to Exercise 18
> **Key insight:** The adjunction unit and counit are built from the duality unit and counit via tensoring.
>
> **Sketch:** Define $\Phi : \mathcal{C}(B \otimes A, C) \to \mathcal{C}(B, A^* \otimes C)$ by $\Phi(f) = (\mathrm{id}_{A^*} \otimes f) \circ \alpha_{A^*,B,A}^{-1} \circ (\mathrm{id}_{A^*} \otimes \cdots) \circ$... more precisely: $\Phi(f) = (\mathrm{id}_{A^*} \otimes f) \circ \alpha \circ (\eta \otimes \mathrm{id}_B)$ after applying unitors. Define $\Psi$ in the other direction using $\varepsilon$. The zig-zag identities for $(A, A^*, \eta, \varepsilon)$ imply $\Phi$ and $\Psi$ are mutually inverse bijections. Naturality follows from naturality of $\eta$, $\varepsilon$, and $\alpha$.

---

## Algorithmic Applications

> [!QUESTION] Exercise 19: Computing in a Strict Monoidal Category
> *This problem develops a basic algorithm for reducing morphisms in a free strict monoidal category to normal form.*
>
> > **Prerequisites:** [[#1.3 Strict Monoidal Categories|1.3 Strict Monoidal Categories]], [[#2.2 Strictification|2.2 Strictification]]
>
> In the free strict monoidal category $\mathcal{F}^{\mathrm{st}}$ on generators $\{a, b, c\}$ (objects are finite words over $\{a,b,c\}$, morphisms are formal composites of generators), describe a normal form for objects and outline an algorithm to: (a) given two object-words, determine if they are equal; (b) given a morphism expressed as a composite of tensor products of atomic generators, reduce it to normal form.

> [!TIP]- Solution to Exercise 19
> **Key insight:** Objects are words in a free monoid; equality is word equality. Morphisms reduce by the interchange law.
>
> **Sketch:** (a) Objects are strings over $\{a,b,c\}$ with concatenation as $\otimes$ and empty string $\varepsilon$ as $I$. Two words are equal iff they are the same string — decidable by character-by-character comparison in $O(n)$ time. (b) A morphism is a grid of atomic morphisms (interchange law says horizontal and vertical can be reordered): reduce to a canonical form by sorting the "rows" (parallel tensor factors) and then composing vertically. Concretely: represent as a sequence of parallel pairs $(f_1 \otimes \cdots \otimes f_k)$ and apply composition. Normal form is a single composite of levels, each level a tensor product of atomic morphisms. This is $O(n \log n)$ in the number of atomic generators.

> [!QUESTION] Exercise 20: Implementing Tensor Products of Linear Maps
> *This problem translates the bifunctor structure of the tensor product into a concrete matrix algorithm.*
>
> > **Prerequisites:** [[#3.3 Vector Spaces over a Field|3.3 Vector Spaces over a Field]], [[#1.1 The Bifunctor and Structural Isomorphisms|1.1 The Bifunctor and Structural Isomorphisms]]
>
> Given finite-dimensional vector spaces $V, W, V', W'$ over $k$ with bases $\{e_i\}$, $\{f_j\}$, $\{e'_p\}$, $\{f'_q\}$ respectively, and linear maps $f : V \to V'$ and $g : W \to W'$ with matrices $F = (F_{pi})$ and $G = (G_{qj})$, write pseudocode (in Python) to compute the matrix of $f \otimes g : V \otimes W \to V' \otimes W'$ in the standard tensor basis $\{e_i \otimes f_j\}$.

> [!TIP]- Solution to Exercise 20
> **Key insight:** The Kronecker product of matrices represents $f \otimes g$ in the tensor basis.
>
> **Sketch:**
> ```python
> import numpy as np
>
> def tensor_of_maps(F: np.ndarray, G: np.ndarray) -> np.ndarray:
>     """
>     Compute the matrix of f ⊗ g.
>     F has shape (dim_V', dim_V); G has shape (dim_W', dim_W).
>     Output has shape (dim_V' * dim_W', dim_V * dim_W).
>     Basis ordering: (i,j) -> i * dim_W + j (row-major).
>     """
>     return np.kron(F, G)
> ```
> The matrix entry at row $(p, q)$ and column $(i, j)$ is $(f \otimes g)(e_i \otimes f_j) = f(e_i) \otimes g(f_j) = (\sum_p F_{pi} e'_p) \otimes (\sum_q G_{qj} f'_q) = \sum_{p,q} F_{pi} G_{qj} e'_p \otimes f'_q$, so the matrix entry is $F_{pi} G_{qj}$. Row index $(p,q) = p \cdot \dim(W') + q$, column index $(i,j) = i \cdot \dim(W) + j$ — exactly the Kronecker product.

> [!QUESTION] Exercise 21: Deciding Commutativity of a Monoid Object
> *This problem asks for an algorithmic test for the commutativity condition in finite monoidal categories.*
>
> > **Prerequisites:** [[#6.2 Commutative Monoids in Symmetric Monoidal Categories|6.2 Commutative Monoids in Symmetric Monoidal Categories]]
>
> Suppose $(\mathcal{C}, \otimes, I, \beta)$ is a symmetric monoidal category presented by generators and relations (finite), and $(M, \mu, \eta)$ is a monoid object given explicitly. Outline a decision procedure to test whether $(M, \mu, \eta)$ is commutative, assuming morphisms can be composed and compared for equality.

> [!TIP]- Solution to Exercise 21
> **Key insight:** Commutativity reduces to checking a single equation $\mu \circ \beta_{M,M} = \mu$, which is decidable when morphisms are finitely presented.
>
> **Sketch:**
> ```python
> def is_commutative_monoid(C, mu, beta_MM):
>     """
>     C: monoidal category with decidable morphism equality
>     mu: morphism M ⊗ M -> M (the multiplication)
>     beta_MM: morphism M ⊗ M -> M ⊗ M (the symmetry)
>     Returns True iff mu ∘ beta_MM == mu
>     """
>     lhs = compose(mu, beta_MM)   # mu ∘ beta_{M,M}
>     return morphisms_equal(C, lhs, mu)
> ```
> Complexity: one composition and one equality test. In practice, for categories like $\mathbf{fdVect}_k$ where morphisms are matrices, `morphisms_equal` is floating-point matrix comparison (or exact over $\mathbb{Q}$). For symbolic monoidal categories given by rewriting systems, equality of morphisms is in general undecidable (word problem), but for specific finite presentations it is decidable.

> [!QUESTION] Exercise 22: String Diagram Simplification as Graph Reduction
> *This problem frames the graphical calculus computationally as a graph rewriting system.*
>
> > **Prerequisites:** [[#7.4 Graphical Calculus: a Hint|7.4 Graphical Calculus: a Hint]]
>
> Model string diagrams as labeled directed hypergraphs: nodes are object-wires, hyperedges are morphism-boxes. Describe a graph rewriting rule corresponding to the zig-zag identity $(\mathrm{id}_A \otimes \varepsilon) \circ (\eta \otimes \mathrm{id}_A) = \mathrm{id}_A$, and outline an algorithm to simplify a string diagram by exhaustive application of zig-zag rules.

> [!TIP]- Solution to Exercise 22
> **Key insight:** The zig-zag identity is a local rewriting rule: replace a cup-cap pair on the same wire by a straight wire.
>
> **Sketch:**
> ```python
> def simplify_zig_zags(diagram):
>     """
>     diagram: list of (type, wires) pairs, where type in
>              {'box', 'cup', 'cap', 'id'} and wires are wire IDs.
>     Returns simplified diagram with all cup-cap (zig-zag) pairs removed.
>     """
>     changed = True
>     while changed:
>         changed = False
>         for i, node_i in enumerate(diagram):
>             if node_i.type == 'cup':
>                 # Look for a cap that immediately follows on the same wire
>                 for j, node_j in enumerate(diagram):
>                     if node_j.type == 'cap' and forms_zig_zag(node_i, node_j, diagram):
>                         diagram = remove_zig_zag(diagram, i, j)
>                         changed = True
>                         break
>     return diagram
> ```
> Termination is guaranteed because each step strictly reduces the number of nodes. Confluence (the order of rewrites doesn't matter) follows from the fact that zig-zag rewrites act on disjoint parts of the diagram. This is essentially a variant of string diagram normalization via the theory of compact closed categories.

> [!QUESTION] Exercise 23: Complexity of Tensor Contraction Order
> *This problem connects monoidal associativity to the classical matrix chain multiplication problem.*
>
> > **Prerequisites:** [[#1.2 The Pentagon and Triangle Axioms|1.2 The Pentagon and Triangle Axioms]], [[#3.3 Vector Spaces over a Field|3.3 Vector Spaces over a Field]]
>
> In $(\mathbf{fdVect}_k, \otimes_k, k)$, the pentagon identity says all parenthesizations of $V_1 \otimes \cdots \otimes V_n$ yield canonically isomorphic spaces. However, the *computational cost* of evaluating a linear map along different parenthesizations can differ vastly. Given linear maps $f_i : V_i \to W_i$ and their Kronecker product representations, describe how the problem of finding the optimal contraction order for $f_1 \otimes \cdots \otimes f_n$ relates to the matrix chain multiplication problem, and state its complexity.

> [!TIP]- Solution to Exercise 23
> **Key insight:** Optimal tensor contraction order is equivalent to optimal matrix chain multiplication, which is solvable in $O(n^3)$ by dynamic programming.
>
> **Sketch:** Write $d_i = \dim(V_i)$ and $d_i' = \dim(W_i)$. Contracting $f_i \otimes f_{i+1}$ (Kronecker product of $d_i' \times d_i$ and $d_{i+1}' \times d_{i+1}$ matrices) costs $O(d_i' d_{i+1}' \cdot d_i d_{i+1})$ operations. The total cost depends on the order in which pairs are contracted, exactly as in matrix chain multiplication. The Yao–Hu dynamic programming algorithm finds the optimal parenthesization in $O(n^3)$ time (or $O(n^2)$ space). In practice, this is the bottleneck in tensor network contraction algorithms used in quantum chemistry and machine learning.

---

## References

| Reference Name | Brief Summary | Link to Reference |
|---|---|---|
| [Riehl, *Category Theory in Context*](https://math.jhu.edu/~eriehl/context/) | Graduate-level introduction to category theory; Appendix E covers coherence for symmetric monoidal categories. Primary reference for this note. | [PDF](https://emilyriehl.github.io/files/context.pdf) |
| [Mac Lane, *Categories for the Working Mathematician* (2nd ed.)](https://link.springer.com/book/10.1007/978-1-4757-4721-8) | Original source for monoidal categories and Mac Lane's coherence theorem; Chapter VII covers monoids, monoidal categories, and the pentagon axiom. | [Springer](https://link.springer.com/book/10.1007/978-1-4757-4721-8) |
| [Etingof, Gelaki, Nikshych, Ostrik, *Tensor Categories*](https://math.mit.edu/~etingof/tenscat.pdf) | Comprehensive treatment of tensor categories including braided and symmetric monoidal categories, duals, and representation-theoretic applications; AMS 2015. | [MIT PDF](https://math.mit.edu/~etingof/tenscat.pdf) |
| [nLab: monoidal category](https://ncatlab.org/nlab/show/monoidal+category) | Encyclopedic reference with formal definitions, multiple equivalent formulations, and links to related structures. | [nLab](https://ncatlab.org/nlab/show/monoidal+category) |
| [nLab: symmetric monoidal category](https://ncatlab.org/nlab/show/symmetric+monoidal+category) | Precise definition, hexagon axioms, coherence for symmetric monoidal categories, and examples. | [nLab](https://ncatlab.org/nlab/show/symmetric+monoidal+category) |
| [nLab: braided monoidal category](https://ncatlab.org/nlab/show/braided+monoidal+category) | Definition of braiding, hexagon identities, distinction from symmetric case, braid group action. | [nLab](https://ncatlab.org/nlab/show/braided+monoidal+category) |
| [nLab: monoidal functor](https://ncatlab.org/nlab/show/monoidal+functor) | Formal definitions of lax, strong, and strict monoidal functors with all coherence diagrams. | [nLab](https://ncatlab.org/nlab/show/monoidal+functor) |
| [nLab: monoid in a monoidal category](https://ncatlab.org/nlab/show/monoid+in+a+monoidal+category) | Definition of monoid objects, commutative monoids, and the table of classical algebraic structures. | [nLab](https://ncatlab.org/nlab/show/monoid+in+a+monoidal+category) |
| [nLab: compact closed category](https://ncatlab.org/nlab/show/compact+closed+category) | Definition of compact closed categories, the graphical calculus, and connection to TFT. | [nLab](https://ncatlab.org/nlab/show/compact+closed+category) |
| [nLab: Mac Lane's coherence theorem](https://ncatlab.org/nlab/show/Mac+Lane%27s+proof+of+the+coherence+theorem+for+monoidal+categories) | Statement and proof sketch of Mac Lane's coherence theorem and the strictification construction. | [nLab](https://ncatlab.org/nlab/show/Mac+Lane%27s+proof+of+the+coherence+theorem+for+monoidal+categories) |
| [nLab: dualizable object](https://ncatlab.org/nlab/show/dualizable+object) | Definition of dualizable objects, unit and counit, zig-zag identities, and examples in fdVect. | [nLab](https://ncatlab.org/nlab/show/dualizable+object) |
| [Baez, *Higher-Dimensional Algebra and Topological Quantum Field Theory* (lecture notes)](https://math.ucr.edu/home/baez/qg-winter2001/definitions.pdf) | Concise lecture notes defining monoidal, braided monoidal, and symmetric monoidal categories with an eye toward TQFTs and higher-dimensional algebra. | [PDF](https://math.ucr.edu/home/baez/qg-winter2001/definitions.pdf) |
| [Baez, *Symmetric Monoidal Categories: a Rosetta Stone* (blog post)](https://johncarlosbaez.wordpress.com/2021/05/28/symmetric-monoidal-categories-a-rosetta-stone/) | Accessible blog exposition of the Rosetta Stone perspective: how the same SMC structure simultaneously encodes physics, logic, and computation. | [Blog](https://johncarlosbaez.wordpress.com/2021/05/28/symmetric-monoidal-categories-a-rosetta-stone/) |
| [Baez & Stay, *Physics, Topology, Logic and Computation: a Rosetta Stone*](https://math.ucr.edu/home/baez/rosetta/rosetta_topos_web.pdf) | Foundational paper unifying quantum mechanics, type theory, and computation via symmetric monoidal closed categories; introduces the compact closed / traced monoidal perspective. | [PDF](https://math.ucr.edu/home/baez/rosetta/rosetta_topos_web.pdf) |
| [Baez, *From Cartesian to Symmetric Monoidal Categories* (n-Category Café)](https://golem.ph.utexas.edu/category/2024/02/from_cartesian_to_symmetric_mo.html) | 2024 blog post explaining how every cartesian monoidal category is symmetric monoidal (via the diagonal) and what structure is gained or lost in passing between the two. | [Blog](https://golem.ph.utexas.edu/category/2024/02/from_cartesian_to_symmetric_mo.html) |
| [nLab: string diagram](https://ncatlab.org/nlab/show/string+diagram) | The graphical calculus for monoidal categories; Joyal–Street theorem on soundness and completeness. | [nLab](https://ncatlab.org/nlab/show/string+diagram) |
