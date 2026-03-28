# 📐 Category Theory I: Categories, Functors, and Natural Transformations

## Table of Contents

- [[#1. Categories|1. Categories]]
  - [[#1.1 The Definition|1.1 The Definition]]
  - [[#1.2 The Axioms|1.2 The Axioms]]
  - [[#1.3 Standard Examples|1.3 Standard Examples]]
  - [[#1.4 Categorical Structures as Categories|1.4 Categorical Structures as Categories]]
  - [[#1.5 Size: Small, Large, and Locally Small|1.5 Size: Small, Large, and Locally Small]]
  - [[#1.6 Discrete, Empty, and Terminal Categories|1.6 Discrete, Empty, and Terminal Categories]]
- [[#2. Morphisms: Monics, Epics, Isomorphisms|2. Morphisms: Monics, Epics, Isomorphisms]]
  - [[#2.1 Cancellation Properties|2.1 Cancellation Properties]]
  - [[#2.2 Inverses and Isomorphisms|2.2 Inverses and Isomorphisms]]
  - [[#2.3 Initial, Terminal, and Zero Objects|2.3 Initial, Terminal, and Zero Objects]]
- [[#3. Subcategories and Opposite Categories|3. Subcategories and Opposite Categories]]
  - [[#3.1 Subcategories|3.1 Subcategories]]
  - [[#3.2 The Opposite Category|3.2 The Opposite Category]]
  - [[#3.3 Self-Dual Categories|3.3 Self-Dual Categories]]
- [[#4. Functors|4. Functors]]
  - [[#4.1 Definition and First Examples|4.1 Definition and First Examples]]
  - [[#4.2 Contravariant Functors|4.2 Contravariant Functors]]
  - [[#4.3 Hom-Functors|4.3 Hom-Functors]]
  - [[#4.4 Properties of Functors|4.4 Properties of Functors]]
  - [[#4.5 The Category Cat|4.5 The Category Cat]]
- [[#5. Natural Transformations|5. Natural Transformations]]
  - [[#5.1 Definition and the Naturality Square|5.1 Definition and the Naturality Square]]
  - [[#5.2 Vertical Composition|5.2 Vertical Composition]]
  - [[#5.3 Natural Isomorphisms|5.3 Natural Isomorphisms]]
  - [[#5.4 Functor Categories and Horizontal Composition|5.4 Functor Categories and Horizontal Composition]]
- [[#6. Equivalence of Categories|6. Equivalence of Categories]]
  - [[#6.1 Definition of Equivalence|6.1 Definition of Equivalence]]
  - [[#6.2 Characterization via Fullness, Faithfulness, Essential Surjectivity|6.2 Characterization via Fullness, Faithfulness, Essential Surjectivity]]
  - [[#6.3 Comma Categories and Slice Categories|6.3 Comma Categories and Slice Categories]]
- [[#7. Products of Categories|7. Products of Categories]]
  - [[#7.1 The Product Category|7.1 The Product Category]]
  - [[#7.2 Functors of Two Variables and Currying|7.2 Functors of Two Variables and Currying]]
- [[#8. Adjoints: First Encounter|8. Adjoints: First Encounter]]
  - [[#8.1 Motivation via Universal Properties|8.1 Motivation via Universal Properties]]
  - [[#8.2 Definition via Hom-Set Bijection|8.2 Definition via Hom-Set Bijection]]
  - [[#8.3 Unit and Counit|8.3 Unit and Counit]]
  - [[#8.4 Triangle Identities|8.4 Triangle Identities]]
  - [[#8.5 Uniqueness of Adjoints|8.5 Uniqueness of Adjoints]]
  - [[#8.6 Examples|8.6 Examples]]
- [[#References|References]]

---

> [!INFO] Series context
> This is File 1 of 4 in a self-study series based on Tom Leinster's Part III Cambridge course (Michaelmas 2000). Subsequent files cover [[concepts/category-theory/02-adjoints-representables|adjoints and representables]], [[concepts/category-theory/03-limits-colimits|limits and colimits]], and [[concepts/category-theory/04-adjoint-functor-theorems-monads|adjoint functor theorems and monads]]. Exercises are taken verbatim from Sheet 1 of the course.

---

## 1. Categories 🏗️

### 1.1 The Definition

Category theory begins with an observation: across all of mathematics, we study not only objects in isolation but the *structure-preserving maps* between them. The abstraction that captures this uniformly is the notion of a *category*.

**Definition (Category).** A *category* $\mathcal{C}$ consists of the following data:

1. A collection $\mathrm{ob}(\mathcal{C})$ whose elements are called *objects*.
2. For each pair of objects $A, B \in \mathrm{ob}(\mathcal{C})$, a collection $\mathcal{C}(A, B)$ (also written $\mathrm{Hom}_{\mathcal{C}}(A,B)$ or $\mathrm{Hom}(A,B)$) whose elements are called *morphisms* (or *maps* or *arrows*) from $A$ to $B$. We write $f: A \to B$ to mean $f \in \mathcal{C}(A,B)$.
3. For each triple of objects $A, B, C$, a function
$$\circ \;:\; \mathcal{C}(B,C) \times \mathcal{C}(A,B) \;\longrightarrow\; \mathcal{C}(A,C),\quad (g,f) \mapsto g \circ f,$$
called *composition*.
4. For each object $A$, a distinguished morphism $\mathrm{id}_A \in \mathcal{C}(A,A)$ called the *identity* on $A$.

### 1.2 The Axioms

These data must satisfy two axioms.

**Axiom 1 (Associativity).** For all $f: A \to B$, $g: B \to C$, $h: C \to D$,
$$h \circ (g \circ f) = (h \circ g) \circ f.$$

**Axiom 2 (Unit laws).** For all $f: A \to B$,
$$\mathrm{id}_B \circ f = f \qquad \text{and} \qquad f \circ \mathrm{id}_A = f.$$

> [!WARNING] On the word "collection"
> We deliberately say "collection" rather than "set" to sidestep Russell's paradox. The objects of a category need not form a set (e.g., $\mathrm{ob}(\mathbf{Set})$ is a proper class). The precise set-theoretic foundation — whether via Grothendieck universes, class-set distinctions, or type theory — varies by author. Mac Lane adopts the distinction between *small* (set-sized) and *large* (class-sized) collections; see §1.5.

> [!NOTE] Notation conventions
> The collections $\mathcal{C}(A,B)$ are assumed to be *disjoint* for different pairs $(A,B)$: every morphism has a unique domain and codomain. We sometimes write the domain and codomain of $f$ as $\mathrm{dom}(f)$ and $\mathrm{cod}(f)$.

### 1.3 Standard Examples

The following are the canonical examples appearing throughout mathematics.

| Category | Objects | Morphisms |
|----------|---------|-----------|
| $\mathbf{Set}$ | Sets | Functions |
| $\mathbf{Grp}$ | Groups | Group homomorphisms |
| $\mathbf{Ab}$ | Abelian groups | Group homomorphisms |
| $\mathbf{Ring}$ | Rings (unital) | Ring homomorphisms |
| $\mathbf{Top}$ | Topological spaces | Continuous maps |
| $\mathbf{Vect}_k$ | $k$-vector spaces | $k$-linear maps |
| $\mathbf{Man}$ | Smooth manifolds | Smooth maps |

In each case, composition is ordinary function composition, and identities are identity functions. Verification of the axioms is routine.

> [!EXAMPLE] The homotopy category Hty
> The *homotopy category* $\mathbf{Hty}$ has topological spaces as objects but takes as morphisms the *homotopy classes* $[f]$ of continuous maps $f: X \to Y$. Two maps $f, g: X \to Y$ are *homotopic* if there exists a continuous $H: X \times [0,1] \to Y$ with $H(-,0) = f$ and $H(-,1) = g$.
>
> Composition is well-defined on homotopy classes: if $f_0 \simeq f_1: X \to Y$ and $g_0 \simeq g_1: Y \to Z$, then $g_0 \circ f_0 \simeq g_1 \circ f_1$. Associativity and unit laws are inherited from $\mathbf{Top}$. This is a genuinely important example because it is **not** concretely realizable: morphisms are equivalence classes, not functions between the underlying sets.

### 1.4 Categorical Structures as Categories

Two seemingly degenerate cases are fundamental.

**Definition (Poset as a category).** Let $(P, \leq)$ be a *poset* (partially ordered set). Define a category $\mathcal{P}$ by:
- Objects: elements of $P$.
- Morphisms: $\mathcal{P}(a,b)$ contains exactly one element (conventionally denoted $a \leq b$ or $*$) if $a \leq b$, and is empty otherwise.

The transitivity of $\leq$ supplies composition ($a \leq b$ and $b \leq c$ imply $a \leq c$), and reflexivity supplies identities ($a \leq a$). Associativity is trivial since there is at most one morphism between any two objects.

**Definition (Monoid as a one-object category).** A *monoid* $(M, \cdot, e)$ — a set with an associative binary operation and a unit element — determines a category $\mathbf{B}M$ with:
- A single object $*$.
- $\mathbf{B}M(*,*) = M$ (every element of $M$ is a morphism).
- Composition given by the monoid operation: $g \circ f := g \cdot f$.
- Identity: the unit element $e$.

The monoid axioms are exactly the category axioms. **A group is a monoid in which every element is invertible; viewed as a one-object category, it is a category in which every morphism is an isomorphism.**

> [!INFO] Historical note
> Categories were introduced by Eilenberg and Mac Lane in their 1945 paper "General Theory of Natural Equivalences" precisely to give a rigorous home to the notion of natural transformation. The definitions of category and functor were, as Mac Lane later wrote, "auxiliary" — the real prize was naturalness.

> [!NOTE] Exercise 1
> Write down three examples each of:
> (a) categories,
> (b) functors,
> (c) natural transformations, and
> (d) adjunctions,
>
> not covered in lectures.

### 1.5 Size: Small, Large, and Locally Small

**Definition (Small and locally small).** A category $\mathcal{C}$ is *small* if both $\mathrm{ob}(\mathcal{C})$ and $\coprod_{A,B} \mathcal{C}(A,B)$ are sets (not proper classes). It is *locally small* if each hom-collection $\mathcal{C}(A,B)$ is a set for every pair of objects $A, B$.

Most familiar categories ($\mathbf{Set}$, $\mathbf{Grp}$, $\mathbf{Top}$) are locally small but not small. Any poset and any monoid, viewed as a category, is small.

> [!WARNING] Why size matters
> Several constructions — notably the functor category $[\mathcal{C}, \mathcal{D}]$ and the Yoneda embedding — require local smallness to avoid set-theoretic pathology. The *adjoint functor theorems* (covered in [[concepts/category-theory/04-adjoint-functor-theorems-monads|File 4]]) impose additional smallness conditions.

### 1.6 Discrete, Empty, and Terminal Categories

**Definition.** The *discrete category* on a set $S$ has object set $S$ and only identity morphisms. The *empty category* $\mathbf{0}$ has no objects and no morphisms. The *terminal category* $\mathbf{1}$ has exactly one object $*$ and one morphism $\mathrm{id}_*$.

These serve as categorical analogues of the empty set, a singleton set, and a set of indiscrete points.

---

## 2. Morphisms: Monics, Epics, Isomorphisms 🔑

### 2.1 Cancellation Properties

**Definition (Monic).** A morphism $f: A \to B$ is *monic* (a *monomorphism*) if for all objects $C$ and all morphisms $g, h: C \to A$,
$$f \circ g = f \circ h \implies g = h.$$

**Definition (Epic).** A morphism $f: A \to B$ is *epic* (an *epimorphism*) if for all objects $C$ and all morphisms $g, h: B \to C$,
$$g \circ f = h \circ f \implies g = h.$$

> [!NOTE] Characterization via cancellation
> These definitions are equivalent to the following:
> - $f$ is monic $\iff$ $f$ is *left-cancellable*: $f \circ g = f \circ h \Rightarrow g = h$.
> - $f$ is epic $\iff$ $f$ is *right-cancellable*: $g \circ f = h \circ f \Rightarrow g = h$.
>
> In $\mathbf{Set}$, monics are exactly injections and epics are exactly surjections. In $\mathbf{Ring}$, the inclusion $\mathbb{Z} \hookrightarrow \mathbb{Q}$ is epic but not surjective — demonstrating that categorical epics need not be surjective.

### 2.2 Inverses and Isomorphisms

**Definition (Left and right inverses).** Given $f: A \to B$:
- A morphism $r: B \to A$ satisfying $f \circ r = \mathrm{id}_B$ is called a *right inverse* (or *retraction*) of $f$; $f$ is then a *section* of $r$.
- A morphism $s: B \to A$ satisfying $s \circ f = \mathrm{id}_A$ is called a *left inverse* (or *section*) of $f$.

**Definition (Isomorphism).** A morphism $f: A \to B$ is an *isomorphism* if there exists $g: B \to A$ such that $g \circ f = \mathrm{id}_A$ and $f \circ g = \mathrm{id}_B$. We write $A \cong B$ if such a morphism exists.

**Proposition.** In any category, if $f: A \to B$ has both a left inverse $s$ and a right inverse $r$, then $f$ is an isomorphism with $s = r$.

*Proof.* We compute:
$$s = s \circ \mathrm{id}_B = s \circ (f \circ r) = (s \circ f) \circ r = \mathrm{id}_A \circ r = r.$$
Since $s = r$, the morphism $f$ has a two-sided inverse, hence is an isomorphism. $\square$

> [!NOTE] Exercise 2
> (i) Prove that in any category, a map with both a left inverse and a right inverse is an isomorphism, and that the two inverses are equal.
>
> (ii) Show that functors preserve isomorphisms: if $F: \mathcal{C} \to \mathcal{D}$ is a functor and $f: A \to B$ is an isomorphism in $\mathcal{C}$, then $Ff: FA \to FB$ is an isomorphism in $\mathcal{D}$.

**Proposition.** Every isomorphism is both monic and epic.

*Proof sketch.* If $f$ has inverse $g$, and $f \circ h = f \circ k$, then $h = g \circ f \circ h = g \circ f \circ k = k$. Similarly for the epic direction. $\square$

*The converse fails in general.* The inclusion $\mathbb{Z} \hookrightarrow \mathbb{Q}$ in $\mathbf{Ring}$ is both monic and epic but not an isomorphism.

### 2.3 Initial, Terminal, and Zero Objects

**Definition.** An object $I \in \mathcal{C}$ is *initial* if for every object $A$ there is exactly one morphism $I \to A$. An object $T \in \mathcal{C}$ is *terminal* if for every object $A$ there is exactly one morphism $A \to T$. An object is a *zero object* if it is both initial and terminal.

**Proposition.** Any two initial objects are isomorphic (via a unique isomorphism). Similarly for terminal objects.

*Proof.* If $I$ and $I'$ are both initial, there exist unique morphisms $f: I \to I'$ and $g: I' \to I$. Then $g \circ f: I \to I$ is the unique morphism from $I$ to itself, which must be $\mathrm{id}_I$. Similarly $f \circ g = \mathrm{id}_{I'}$. $\square$

> [!EXAMPLE] Examples of initial and terminal objects
> - In $\mathbf{Set}$: $\emptyset$ is initial (there is exactly one function $\emptyset \to A$ for any set $A$); any singleton $\{*\}$ is terminal.
> - In $\mathbf{Grp}$: the trivial group $\{e\}$ is both initial and terminal (a zero object).
> - In $\mathbf{Ring}$: the zero ring $\{0\}$ (where $0 = 1$) is terminal; $\mathbb{Z}$ is initial (the unique ring map $\mathbb{Z} \to R$ sends $n \mapsto n \cdot 1_R$).
> - In a poset viewed as a category: an initial object is a least element; a terminal object is a greatest element.

---

## 3. Subcategories and Opposite Categories 🔁

### 3.1 Subcategories

**Definition (Subcategory).** A *subcategory* $\mathcal{S}$ of $\mathcal{C}$ consists of:
- A subcollection $\mathrm{ob}(\mathcal{S}) \subseteq \mathrm{ob}(\mathcal{C})$,
- For each $A, B \in \mathrm{ob}(\mathcal{S})$, a subcollection $\mathcal{S}(A,B) \subseteq \mathcal{C}(A,B)$,

such that $\mathcal{S}$ is closed under composition (if $f \in \mathcal{S}(A,B)$ and $g \in \mathcal{S}(B,C)$ then $g \circ f \in \mathcal{S}(A,C)$) and contains all identity morphisms of its objects ($\mathrm{id}_A \in \mathcal{S}(A,A)$ for all $A \in \mathrm{ob}(\mathcal{S})$).

**Definition (Full subcategory).** $\mathcal{S}$ is a *full* subcategory if $\mathcal{S}(A,B) = \mathcal{C}(A,B)$ for all $A, B \in \mathrm{ob}(\mathcal{S})$ — all morphisms between objects of $\mathcal{S}$ are included.

**Definition (Wide subcategory).** $\mathcal{S}$ is a *wide* (or *lluf*) subcategory if $\mathrm{ob}(\mathcal{S}) = \mathrm{ob}(\mathcal{C})$ — all objects are included.

> [!EXAMPLE] Examples
> - $\mathbf{Ab}$ is a full subcategory of $\mathbf{Grp}$.
> - $\mathbf{Grp}$ is a subcategory of $\mathbf{Mon}$ (monoids) that is wide but not full.
> - The category of fields is a full subcategory of $\mathbf{Ring}$.

> [!NOTE] Exercise 3
> (i) Characterize the subcategories of a poset $P$ viewed as a category: what are the possible subcategories?
>
> (ii) Characterize the subcategories of a group $G$ viewed as a one-object category: what structure on $G$ do they correspond to?

### 3.2 The Opposite Category

**Definition (Opposite category).** Given a category $\mathcal{C}$, the *opposite* (or *dual*) category $\mathcal{C}^{\mathrm{op}}$ is defined by:
- $\mathrm{ob}(\mathcal{C}^{\mathrm{op}}) = \mathrm{ob}(\mathcal{C})$.
- $\mathcal{C}^{\mathrm{op}}(A, B) = \mathcal{C}(B, A)$ for all objects $A, B$.
- Composition in $\mathcal{C}^{\mathrm{op}}$: given $f \in \mathcal{C}^{\mathrm{op}}(A,B)$ and $g \in \mathcal{C}^{\mathrm{op}}(B,C)$ (i.e., $f: B \to A$ and $g: C \to B$ in $\mathcal{C}$), define $g \circ^{\mathrm{op}} f := f \circ g \in \mathcal{C}(C,A) = \mathcal{C}^{\mathrm{op}}(A,C)$.

In other words, $\mathcal{C}^{\mathrm{op}}$ is obtained by formally reversing all arrows.

> [!DANGER] Duality principle
> The *duality principle* states: any theorem about all categories, when its proof is dualized (all arrows reversed, domain and codomain exchanged), yields another valid theorem. This means every categorical concept comes in a dual pair: monic/epic, initial/terminal, limit/colimit, etc. We get two theorems for the price of one proof.

> [!NOTE] Exercise 4
> (i) Determine the opposite category of a group $G$ (viewed as a one-object category) and establish an isomorphism $G \cong G^{\mathrm{op}}$.
>
> (ii) Give an example of (a) a poset and (b) a monoid that are **not** isomorphic to their opposite categories.

### 3.3 Self-Dual Categories

**Definition (Category isomorphism).** An *isomorphism of categories* is a functor $F: \mathcal{C} \to \mathcal{D}$ that has an inverse functor $G: \mathcal{D} \to \mathcal{C}$ with $GF = \mathrm{id}_{\mathcal{C}}$ and $FG = \mathrm{id}_{\mathcal{D}}$ (equality, not just natural isomorphism).

A category $\mathcal{C}$ is *self-dual* if $\mathcal{C} \cong \mathcal{C}^{\mathrm{op}}$.

For a group $G$ viewed as a one-object category, the map $g \mapsto g^{-1}$ defines an isomorphism $G \cong G^{\mathrm{op}}$ (since $(gh)^{-1} = h^{-1}g^{-1}$, which reverses composition). Every group, viewed as a category, is self-dual. The category $\mathbf{Ab}$ is also self-dual (via the identity-on-objects map, using commutativity).

> [!NOTE] Exercise 5
> Consider the homotopy category $\mathbf{Hty}$, whose objects are topological spaces and whose morphisms are homotopy classes of continuous maps. Characterize the objects that are isomorphic (in $\mathbf{Hty}$) to the one-element space $\{*\}$.

*Hint.* An object $X$ is isomorphic to $\{*\}$ in $\mathbf{Hty}$ iff there exist maps $f: X \to \{*\}$ and $g: \{*\} \to X$ with $f \circ g \simeq \mathrm{id}_{\{*\}}$ and $g \circ f \simeq \mathrm{id}_X$. The latter condition says $X$ is *contractible*.

---

## 4. Functors 🔭

### 4.1 Definition and First Examples

**Definition (Functor).** Let $\mathcal{C}$ and $\mathcal{D}$ be categories. A *(covariant) functor* $F: \mathcal{C} \to \mathcal{D}$ consists of:
- A function $F: \mathrm{ob}(\mathcal{C}) \to \mathrm{ob}(\mathcal{D})$.
- For each pair $A, B \in \mathrm{ob}(\mathcal{C})$, a function $F_{A,B}: \mathcal{C}(A,B) \to \mathcal{D}(FA, FB)$, written $f \mapsto Ff$.

These must satisfy:
1. **Preservation of composition:** $F(g \circ f) = Fg \circ Ff$ for all composable $f, g$.
2. **Preservation of identities:** $F(\mathrm{id}_A) = \mathrm{id}_{FA}$ for all $A$.

> [!EXAMPLE] Forgetful functors
> The *forgetful functor* $U: \mathbf{Grp} \to \mathbf{Set}$ sends a group $(G, \cdot, e, {}^{-1})$ to its underlying set $G$, and sends each homomorphism to the underlying function. It "forgets" the group structure. Similarly one has forgetful functors $\mathbf{Ring} \to \mathbf{Ab} \to \mathbf{Set}$, $\mathbf{Top} \to \mathbf{Set}$, etc.

> [!EXAMPLE] Free functors
> The *free group functor* $F: \mathbf{Set} \to \mathbf{Grp}$ sends a set $S$ to the free group $F(S)$ generated by $S$. A function $f: S \to T$ induces a homomorphism $F(f): F(S) \to F(T)$ by extending $f$ to a group homomorphism. Free functors are the left adjoints of forgetful functors.

> [!EXAMPLE] Power set functor
> The covariant *power set functor* $\mathcal{P}: \mathbf{Set} \to \mathbf{Set}$ sends a set $X$ to its power set $\mathcal{P}(X) = \{S \mid S \subseteq X\}$. For a function $f: X \to Y$, the map $\mathcal{P}(f): \mathcal{P}(X) \to \mathcal{P}(Y)$ is the direct image: $\mathcal{P}(f)(S) = f(S) = \{f(s) \mid s \in S\}$.

### 4.2 Contravariant Functors

**Definition (Contravariant functor).** A *contravariant functor* from $\mathcal{C}$ to $\mathcal{D}$ is a covariant functor $F: \mathcal{C}^{\mathrm{op}} \to \mathcal{D}$. Concretely, it assigns to each $f: A \to B$ in $\mathcal{C}$ a morphism $Ff: FB \to FA$ in $\mathcal{D}$, reversing the direction, with $F(g \circ f) = Ff \circ Fg$.

> [!EXAMPLE] Contravariant power set
> The contravariant power set functor sends $f: X \to Y$ to the preimage map $f^{-1}: \mathcal{P}(Y) \to \mathcal{P}(X)$. This is functorial because $(g \circ f)^{-1} = f^{-1} \circ g^{-1}$.

### 4.3 Hom-Functors

Fix a locally small category $\mathcal{C}$ and an object $A \in \mathcal{C}$.

**Definition (Covariant hom-functor).** $\mathcal{C}(A, -): \mathcal{C} \to \mathbf{Set}$ sends:
- An object $B \mapsto \mathcal{C}(A, B)$.
- A morphism $f: B \to C \mapsto$ the postcomposition map $f_*: \mathcal{C}(A,B) \to \mathcal{C}(A,C)$, $g \mapsto f \circ g$.

**Definition (Contravariant hom-functor).** $\mathcal{C}(-, B): \mathcal{C}^{\mathrm{op}} \to \mathbf{Set}$ sends:
- An object $A \mapsto \mathcal{C}(A, B)$.
- A morphism $f: A \to A'$ in $\mathcal{C}$ (i.e., $f: A' \to A$ in $\mathcal{C}^{\mathrm{op}}$) $\mapsto$ the precomposition map $f^*: \mathcal{C}(A',B) \to \mathcal{C}(A,B)$, $g \mapsto g \circ f$.

These hom-functors are central: the Yoneda lemma (covered in [[concepts/category-theory/02-adjoints-representables|File 2]]) says they determine a category completely.

### 4.4 Properties of Functors

**Definition.** A functor $F: \mathcal{C} \to \mathcal{D}$ is:
- *Full* if $F_{A,B}: \mathcal{C}(A,B) \to \mathcal{D}(FA,FB)$ is surjective for all $A, B$.
- *Faithful* if $F_{A,B}$ is injective for all $A, B$.
- *Essentially surjective* (or *dense*) if for every $D \in \mathcal{D}$ there exists $C \in \mathcal{C}$ with $FC \cong D$.

> [!NOTE] Exercise 6
> Is there a functor $Z: \mathbf{Grp} \to \mathbf{Grp}$ that sends each group $G$ to its center $Z(G)$? More precisely: given a group homomorphism $f: G \to H$, is there a natural candidate for a homomorphism $Z(G) \to Z(H)$ making $Z$ into a functor?

*Note.* The difficulty is that a homomorphism need not map the center into the center in a functorial way. Consider whether $f(Z(G)) \subseteq Z(H)$ always holds.

### 4.5 The Category Cat

Functors can be composed: if $F: \mathcal{A} \to \mathcal{B}$ and $G: \mathcal{B} \to \mathcal{C}$ are functors, the composite $G \circ F: \mathcal{A} \to \mathcal{C}$ is defined by $(G \circ F)(A) = G(FA)$ and $(G \circ F)(f) = G(Ff)$. The identity functor $\mathrm{id}_{\mathcal{C}}: \mathcal{C} \to \mathcal{C}$ acts as the identity on both objects and morphisms.

**Definition.** The category $\mathbf{Cat}$ has small categories as objects and functors as morphisms, with the composition and identities above.

> [!WARNING] Size caveat for Cat
> $\mathbf{Cat}$ is itself a large category (its objects form a proper class). One can restrict to a universe-relative version. In the 2-categorical sense, $\mathbf{Cat}$ has an additional layer of structure: natural transformations between functors form the 2-morphisms.

---

## 5. Natural Transformations 🌿

### 5.1 Definition and the Naturality Square

Natural transformations are the morphisms between functors. They encode the notion that a construction is "the same" across all objects of a category in a coherent way.

**Definition (Natural transformation).** Let $F, G: \mathcal{C} \to \mathcal{D}$ be functors. A *natural transformation* $\alpha: F \Rightarrow G$ is a family of morphisms
$$\alpha_A: FA \to GA \quad \text{(for each } A \in \mathrm{ob}(\mathcal{C})\text{)},$$
called the *components* of $\alpha$, such that for every morphism $f: A \to B$ in $\mathcal{C}$, the following *naturality square* commutes:

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}
FA \arrow[r, "Ff"] \arrow[d, "\alpha_A"'] & FB \arrow[d, "\alpha_B"] \\
GA \arrow[r, "Gf"'] & GB
\end{tikzcd}
\end{document}
```

That is: $\alpha_B \circ Ff = Gf \circ \alpha_A$ for every $f: A \to B$.

> [!EXAMPLE] A canonical natural transformation
> Let $V$ be a finite-dimensional real vector space. There is a natural transformation $\alpha: \mathrm{id}_{\mathbf{FDVect}_\mathbb{R}} \Rightarrow (-)^{**}$ (the double-dual functor) whose component at $V$ is the canonical map
> $$\alpha_V: V \to V^{**}, \quad v \mapsto \hat{v}, \quad \hat{v}(\phi) := \phi(v).$$
> Naturality: for any linear map $f: V \to W$, we need $\alpha_W \circ f = f^{**} \circ \alpha_V$. This holds because $\widehat{f(v)}(\psi) = \psi(f(v)) = (f^*\psi)(v) = \widehat{v}(f^*\psi) = (f^{**}\hat{v})(\psi)$.
>
> In contrast, there is a non-natural isomorphism $V \cong V^*$ (requiring a choice of basis), illustrating that naturality is a substantive condition.

### 5.2 Vertical Composition

**Definition (Vertical composition).** Given natural transformations $\alpha: F \Rightarrow G$ and $\beta: G \Rightarrow H$ between functors $F, G, H: \mathcal{C} \to \mathcal{D}$, their *vertical composite* $\beta \circ \alpha: F \Rightarrow H$ has components
$$(\beta \circ \alpha)_A := \beta_A \circ \alpha_A: FA \to HA.$$

*Naturality check.* For $f: A \to B$:
$$(\beta \circ \alpha)_B \circ Ff = \beta_B \circ \alpha_B \circ Ff = \beta_B \circ Gf \circ \alpha_A = Hf \circ \beta_A \circ \alpha_A = Hf \circ (\beta \circ \alpha)_A. \quad \square$$

The *identity natural transformation* $\mathrm{id}_F: F \Rightarrow F$ has components $(\mathrm{id}_F)_A = \mathrm{id}_{FA}$.

### 5.3 Natural Isomorphisms

**Definition (Natural isomorphism).** A natural transformation $\alpha: F \Rightarrow G$ is a *natural isomorphism* if there exists a natural transformation $\alpha^{-1}: G \Rightarrow F$ with $\alpha^{-1} \circ \alpha = \mathrm{id}_F$ and $\alpha \circ \alpha^{-1} = \mathrm{id}_G$. We write $F \cong G$.

> [!NOTE] Exercise 7
> Show that a natural transformation $\alpha: F \Rightarrow G$ is a natural isomorphism if and only if each component $\alpha_A: FA \to GA$ is an isomorphism in $\mathcal{D}$.

*Proof direction (component-wise iso $\Rightarrow$ natural iso).* Define $(\alpha^{-1})_A := (\alpha_A)^{-1}$. One must verify naturality of $\alpha^{-1}$: for $f: A \to B$,
$$(\alpha^{-1})_B \circ Gf = (\alpha_B)^{-1} \circ Gf = (\alpha_B)^{-1} \circ Gf \circ \alpha_A \circ (\alpha_A)^{-1} = (\alpha_B)^{-1} \circ \alpha_B \circ Ff \circ (\alpha_A)^{-1} = Ff \circ (\alpha^{-1})_A.$$

### 5.4 Functor Categories and Horizontal Composition

**Definition (Functor category).** Given categories $\mathcal{C}$ and $\mathcal{D}$ (with $\mathcal{C}$ small), the *functor category* $[\mathcal{C}, \mathcal{D}]$ (also written $\mathcal{D}^{\mathcal{C}}$) has:
- Objects: functors $F: \mathcal{C} \to \mathcal{D}$.
- Morphisms: natural transformations $\alpha: F \Rightarrow G$.
- Composition: vertical composition of natural transformations.

**Definition (Horizontal composition / whiskering).** Given $\alpha: F \Rightarrow G$ between $F, G: \mathcal{B} \to \mathcal{C}$ and $\beta: H \Rightarrow K$ between $H, K: \mathcal{C} \to \mathcal{D}$, the *horizontal composite* $\beta * \alpha: H \circ F \Rightarrow K \circ G$ has components
$$(\beta * \alpha)_A := \beta_{GA} \circ H(\alpha_A) = K(\alpha_A) \circ \beta_{FA}: HFA \to KGA.$$

These two expressions are equal by naturality of $\beta$ applied to $\alpha_A: FA \to GA$.

> [!TIP] Interchange law
> Horizontal and vertical composition satisfy the *interchange law*: $(\beta' \circ \beta) * (\alpha' \circ \alpha) = (\beta' * \alpha') \circ (\beta * \alpha)$ whenever the composites are defined. This makes $\mathbf{Cat}$ into a *2-category*.

---

## 6. Equivalence of Categories ≃

### 6.1 Definition of Equivalence

Isomorphism of categories (requiring $GF = \mathrm{id}$ and $FG = \mathrm{id}$ on the nose) is too rigid for most purposes. The correct notion is equivalence.

**Definition (Equivalence of categories).** A functor $F: \mathcal{C} \to \mathcal{D}$ is an *equivalence of categories* if there exists a functor $G: \mathcal{D} \to \mathcal{C}$ and natural isomorphisms
$$\eta: \mathrm{id}_{\mathcal{C}} \xrightarrow{\;\cong\;} G \circ F \qquad \text{and} \qquad \varepsilon: F \circ G \xrightarrow{\;\cong\;} \mathrm{id}_{\mathcal{D}}.$$

We say $\mathcal{C}$ and $\mathcal{D}$ are *equivalent*, written $\mathcal{C} \simeq \mathcal{D}$.

> [!DANGER] Equivalence is not isomorphism
> Isomorphism requires the functors to be mutually inverse on the nose. Equivalence only requires natural isomorphism. Most categories that "look the same" are equivalent but not isomorphic — e.g., $\mathbf{FDVect}_k \simeq \mathbf{Mat}_k$ (Exercise 9) but these are not isomorphic as categories.

### 6.2 Characterization via Fullness, Faithfulness, Essential Surjectivity

**Theorem.** A functor $F: \mathcal{C} \to \mathcal{D}$ is an equivalence of categories if and only if it is full, faithful, and essentially surjective.

*Proof sketch.*
- ($\Rightarrow$) If $F$ is an equivalence with quasi-inverse $G$ and natural isomorphisms $\eta, \varepsilon$: essentially surjectivity is immediate since $FGA \cong A$ for every $A \in \mathcal{D}$. Faithfulness: if $Ff = Fg$, then $Gff = GFf = $ (via $\eta$) $= GFg = Gg$, and applying the naturality of $\eta$ shows $f = g$. Fullness: given $h: FA \to FB$, define $f = \eta_B^{-1} \circ G(h) \circ \eta_A$ and verify $Ff = h$ using naturality of $\varepsilon$.
- ($\Leftarrow$) Requires choosing, for each $D \in \mathcal{D}$, an object $GD \in \mathcal{C}$ and an isomorphism $\varepsilon_D: FGD \xrightarrow{\sim} D$. *This direction requires the axiom of choice* to select the $GD$ and $\varepsilon_D$.

**Key distinction:** An isomorphism of categories does not require choice; an equivalence (in the $\Leftarrow$ direction) does.

### 6.3 Comma Categories and Slice Categories

**Definition (Comma category).** Given functors $F: \mathcal{A} \to \mathcal{C}$ and $G: \mathcal{B} \to \mathcal{C}$, the *comma category* $(F \downarrow G)$ (also written $F/G$) has:
- Objects: triples $(A, B, f)$ where $A \in \mathcal{A}$, $B \in \mathcal{B}$, and $f: FA \to GB$ is a morphism in $\mathcal{C}$.
- Morphisms $(A, B, f) \to (A', B', f')$: pairs $(a: A \to A', b: B \to B')$ such that $f' \circ Fa = Gb \circ f$ (the evident square commutes).

**Definition (Slice category).** Taking $F = \mathrm{id}_{\mathcal{C}}$ and $G$ the inclusion of a single object $A$: the *slice category* $\mathcal{C}/A$ has objects $(C, f)$ where $f: C \to A$, and morphisms $(C, f) \to (C', f')$ are maps $g: C \to C'$ with $f' \circ g = f$.

The *coslice* $A/\mathcal{C}$ is dual. Taking $\mathcal{A} = \mathbf{1}$ (the terminal category) and $G = \mathrm{id}_{\mathcal{C}}$, the coslice $*/\mathcal{C}$ is the category of *objects under $*$*: objects are pairs $(C, x)$ where $x: * \to C$ picks out an element of $C$.

> [!NOTE] Exercise 8
> (i) A *pointed set* is a pair $(X, x_0)$ where $X$ is a set and $x_0 \in X$ is a chosen basepoint. Let $\mathbf{Set}_*$ be the category of pointed sets, where morphisms are basepoint-preserving functions. Show that $\mathbf{Set}_* \cong (\mathbf{1} \downarrow \mathbf{Set})$ (the comma category of the unique functor $\mathbf{1} \to \mathbf{Set}$ picking out a one-element set and the identity functor).
>
> (ii) Define an equivalence of categories precisely. Prove that $\mathbf{Set}_*$ is equivalent to $\mathbf{Par}$, the category whose objects are sets and whose morphisms $X \to Y$ are *partial functions* from $X$ to $Y$ (functions defined on some subset of $X$).

> [!NOTE] Exercise 9
> Fix a field $k$. Let $\mathbf{Mat}_k$ be the category whose objects are the natural numbers $0, 1, 2, \ldots$ and whose hom-set $\mathbf{Mat}_k(m, n)$ is the set of $n \times m$ matrices over $k$ (equivalently, $k$-linear maps $k^m \to k^n$), with composition given by matrix multiplication. Prove that $\mathbf{Mat}_k \simeq \mathbf{FDVect}_k$ (the category of finite-dimensional $k$-vector spaces and $k$-linear maps).

*Proof sketch.* Define $F: \mathbf{Mat}_k \to \mathbf{FDVect}_k$ by $F(n) = k^n$ and $F(M) = $ the linear map with matrix $M$. Show $F$ is full (every linear map $k^m \to k^n$ has a matrix), faithful (different matrices give different maps), and essentially surjective (every f.d. vector space is isomorphic to some $k^n$).

---

## 7. Products of Categories ✖️

### 7.1 The Product Category

**Definition (Product category).** Given categories $\mathcal{C}$ and $\mathcal{D}$, their *product* $\mathcal{C} \times \mathcal{D}$ has:
- Objects: pairs $(C, D)$ with $C \in \mathcal{C}$ and $D \in \mathcal{D}$.
- Morphisms: $(C, D) \to (C', D')$ are pairs $(f, g)$ with $f: C \to C'$ and $g: D \to D'$.
- Composition: $(f', g') \circ (f, g) = (f' \circ f, g' \circ g)$.
- Identities: $\mathrm{id}_{(C,D)} = (\mathrm{id}_C, \mathrm{id}_D)$.

### 7.2 Functors of Two Variables and Currying

**Definition.** A *bifunctor* is a functor $F: \mathcal{A} \times \mathcal{B} \to \mathcal{C}$. Explicitly:
- $F$ assigns to each pair $(A, B)$ an object $F(A, B) \in \mathcal{C}$.
- To each pair $(f: A \to A', g: B \to B')$ a morphism $F(f,g): F(A,B) \to F(A',B')$.
- With $F(\mathrm{id}_A, \mathrm{id}_B) = \mathrm{id}_{F(A,B)}$ and $F(f' \circ f, g' \circ g) = F(f',g') \circ F(f,g)$.

**Currying.** A bifunctor $F: \mathcal{A} \times \mathcal{B} \to \mathcal{C}$ determines:
- For each fixed $A \in \mathcal{A}$: a functor $F(A, -): \mathcal{B} \to \mathcal{C}$, $B \mapsto F(A,B)$, $g \mapsto F(\mathrm{id}_A, g)$.
- For each fixed $B \in \mathcal{B}$: a functor $F(-, B): \mathcal{A} \to \mathcal{C}$, $A \mapsto F(A,B)$, $f \mapsto F(f, \mathrm{id}_B)$.

Moreover, $f: A \to A'$ determines a natural transformation $F(f,-): F(A,-) \Rightarrow F(A',-)$ with components $F(f, \mathrm{id}_B): F(A,B) \to F(A',B)$.

> [!EXAMPLE] The tensor product as a bifunctor
> For a commutative ring $R$, $- \otimes_R -: \mathbf{Mod}_R \times \mathbf{Mod}_R \to \mathbf{Mod}_R$ is a bifunctor. For fixed $M$, the functor $M \otimes_R -$ is left adjoint to $\mathrm{Hom}_R(M, -)$.

> [!NOTE] Exercise 10
> Let $F: \mathcal{A} \times \mathcal{B} \to \mathcal{C}$ be a functor. Suppose we are given:
> - For each $a \in \mathcal{A}$, a functor $F(a,-): \mathcal{B} \to \mathcal{C}$.
> - For each $b \in \mathcal{B}$, a functor $F(-,b): \mathcal{A} \to \mathcal{C}$.
>
> such that $F(a,b)$ agrees for both families at each pair $(a,b)$, and such that for each $f: a \to a'$ and $g: b \to b'$, the square
> $$F(a,b) \to F(a',b) \to F(a',b') \quad \text{and} \quad F(a,b) \to F(a,b') \to F(a',b')$$
> commute (i.e., $F(f,b') \circ F(a,g) = F(a',g) \circ F(f,b)$). Show that there is a unique functor $F: \mathcal{A} \times \mathcal{B} \to \mathcal{C}$ whose restrictions to each slice are as specified.

---

## 8. Adjoints: First Encounter 🔑

### 8.1 Motivation via Universal Properties

Universal properties pervade mathematics. When we say "the free group on $S$" or "the tensor product $M \otimes_R N$," we are specifying an object by what maps *into* or *out of* it look like. The notion of adjunction is the categorical formalization of this idea.

> [!INFO] The central philosophy
> Leinster's course synopsis states: the real objects of study in category theory are "not so much categories themselves as the maps between them." Adjunctions encode the ubiquitous phenomenon that two seemingly different mapping problems have the same solutions — a bijection of hom-sets that is natural in both variables.

### 8.2 Definition via Hom-Set Bijection

**Definition (Adjunction).** Let $\mathcal{C}$ and $\mathcal{D}$ be categories. An *adjunction* between a functor $F: \mathcal{C} \to \mathcal{D}$ and a functor $G: \mathcal{D} \to \mathcal{C}$ is a bijection
$$\phi_{A,B}: \mathcal{D}(FA, B) \xrightarrow{\;\sim\;} \mathcal{C}(A, GB)$$
that is natural in both $A \in \mathcal{C}$ and $B \in \mathcal{D}$. We say $F$ is *left adjoint* to $G$ and $G$ is *right adjoint* to $F$, written $F \dashv G$.

Naturality in $A$ means: for all $f: A' \to A$ in $\mathcal{C}$,
$$\phi_{A',B}(h \circ Ff) = \phi_{A,B}(h) \circ f \quad \text{for all } h \in \mathcal{D}(FA, B).$$

Naturality in $B$ means: for all $g: B \to B'$ in $\mathcal{D}$,
$$\phi_{A,B'}(g \circ h) = Gg \circ \phi_{A,B}(h) \quad \text{for all } h \in \mathcal{D}(FA, B).$$

Together, naturality in both variables is the condition that the assignment $(A, B) \mapsto \phi_{A,B}$ defines a natural isomorphism of functors $\mathcal{D}(F-, -) \cong \mathcal{C}(-, G-)$ from $\mathcal{C}^{\mathrm{op}} \times \mathcal{D}$ to $\mathbf{Set}$.

### 8.3 Unit and Counit

An adjunction $F \dashv G$ can equivalently be specified by its *unit* and *counit*.

**Definition (Unit and counit).** Given $F \dashv G$ with bijection $\phi$:
- The *unit* $\eta: \mathrm{id}_{\mathcal{C}} \Rightarrow GF$ is the natural transformation with components $\eta_A := \phi_{A, FA}(\mathrm{id}_{FA}) \in \mathcal{C}(A, GFA)$.
- The *counit* $\varepsilon: FG \Rightarrow \mathrm{id}_{\mathcal{D}}$ is the natural transformation with components $\varepsilon_B := \phi_{GB,B}^{-1}(\mathrm{id}_{GB}) \in \mathcal{D}(FGB, B)$.

These are natural because $\phi$ is natural. Every morphism $h \in \mathcal{D}(FA, B)$ can be recovered as $h = \varepsilon_B \circ F(\phi(h))$, and dually.

### 8.4 Triangle Identities

**Theorem (Triangle identities).** Given an adjunction $F \dashv G$ with unit $\eta$ and counit $\varepsilon$, the following diagrams commute:

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}
F \arrow[r, "F\eta"] \arrow[dr, "\mathrm{id}_F"'] & FGF \arrow[d, "\varepsilon F"] \\
& F
\end{tikzcd}
\end{document}
```

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}
G \arrow[r, "\eta G"] \arrow[dr, "\mathrm{id}_G"'] & GFG \arrow[d, "G\varepsilon"] \\
& G
\end{tikzcd}
\end{document}
```

That is:
$$(\varepsilon F) \circ (F\eta) = \mathrm{id}_F \qquad \text{and} \qquad (G\varepsilon) \circ (\eta G) = \mathrm{id}_G.$$

Here $\varepsilon F$ denotes the natural transformation with components $\varepsilon_{FA}: FGF A \to FA$, and $F\eta$ has components $F(\eta_A): FA \to FGFA$.

*Proof sketch.* At a fixed object $A$, the composite $(\varepsilon_{FA}) \circ F(\eta_A)$ must equal $\mathrm{id}_{FA}$. Under $\phi_{A, FA}$, the identity $\mathrm{id}_{FA}$ corresponds to $\eta_A$. The naturality of $\phi$ in $B$, applied to $\varepsilon_{FA}: FGA \to FA$ (wait — at object $FA$), gives $\phi(\varepsilon_{FA} \circ F(\eta_A)) = G(\varepsilon_{FA}) \circ \phi(F(\eta_A))$. Tracing through the definitions yields both identities.

> [!NOTE] Exercises 12 and 13
> **(Exercise 12)** For posets $A$ and $B$, describe explicitly what an adjunction $f \dashv g$ between order-preserving maps $f: A \to B$ and $g: B \to A$ consists of. (Recall: a poset is a category with at most one morphism between any two objects.)
>
> Show that the contravariant power set functor $\mathcal{P}: \mathbf{Set}^{\mathrm{op}} \to \mathbf{Set}$ is related to itself by an adjunction: exhibit a bijection $\mathbf{Set}^{\mathrm{op}}(\mathcal{P}(Y), X) \cong \mathbf{Set}(X, \mathcal{P}(Y))$ that is natural in $X$ and $Y$ (or formulate the correct adjunction statement involving $\mathcal{P}$).
>
> **(Exercise 13)** Let $F: \mathcal{C} \rightleftharpoons \mathcal{D}: G$ be an adjunction. Define the unit $\eta: \mathrm{id}_{\mathcal{C}} \Rightarrow GF$ and counit $\varepsilon: FG \Rightarrow \mathrm{id}_{\mathcal{D}}$ explicitly in terms of the hom-set bijection. Verify the triangle identities:
> $$(\varepsilon F) \circ (F\eta) = \mathrm{id}_F \qquad \text{and} \qquad (G\varepsilon) \circ (\eta G) = \mathrm{id}_G.$$
> Also prove uniqueness: show that the unit and counit together with the triangle identities uniquely determine the adjunction bijection $\phi$.

### 8.5 Uniqueness of Adjoints

**Proposition.** If $F \dashv G$ and $F \dashv G'$, then $G \cong G'$ via a natural isomorphism. Left adjoints are unique up to unique natural isomorphism.

*Proof.* For each $B \in \mathcal{D}$, the functor $\mathcal{C}(-, GB)$ and $\mathcal{C}(-, G'B)$ are both naturally isomorphic to $\mathcal{D}(F-, B)$, hence to each other. By the Yoneda lemma (File 2), $GB \cong G'B$ naturally in $B$. $\square$

### 8.6 Examples

> [!EXAMPLE] Free-forgetful adjunction
> The free group functor $F: \mathbf{Set} \to \mathbf{Grp}$ is left adjoint to the forgetful functor $U: \mathbf{Grp} \to \mathbf{Set}$:
> $$\mathbf{Grp}(F(S), G) \cong \mathbf{Set}(S, U(G)).$$
> A group homomorphism $F(S) \to G$ is determined freely by the images of the generators $S$ — any function $S \to U(G)$ extends uniquely to a homomorphism. The unit $\eta_S: S \to UF(S)$ is the inclusion of generators.

> [!EXAMPLE] Product-diagonal adjunction
> The diagonal functor $\Delta: \mathcal{C} \to \mathcal{C} \times \mathcal{C}$, $A \mapsto (A, A)$, has a right adjoint: the product functor $(A \times B): \mathcal{C} \times \mathcal{C} \to \mathcal{C}$ (when $\mathcal{C}$ has binary products). The adjunction is $\mathcal{C}(C, A \times B) \cong \mathcal{C}(C, A) \times \mathcal{C}(C, B)$, which is the universal property of the product.

> [!EXAMPLE] Tensor-hom adjunction
> For a commutative ring $R$ and $R$-modules $M$, $N$, $P$:
> $$\mathrm{Hom}_R(M \otimes_R N, P) \cong \mathrm{Hom}_R(M, \mathrm{Hom}_R(N, P)).$$
> This is the adjunction $- \otimes_R N \dashv \mathrm{Hom}_R(N, -)$.

> [!EXAMPLE] Initial objects as degenerate adjoints
> The unique functor $!: \mathcal{C} \to \mathbf{1}$ always has a right adjoint iff $\mathcal{C}$ has a terminal object, and a left adjoint iff $\mathcal{C}$ has an initial object. The functor picking out an initial object $I: \mathbf{1} \to \mathcal{C}$ is left adjoint to $!$ iff $I(*)$ is initial in $\mathcal{C}$, since $\mathcal{C}(I(*), A) \cong \mathbf{1}(*,*(= \{*\}))$ for all $A$ means exactly one morphism $I(*) \to A$.

> [!NOTE] Exercise 11
> (i) Define what it means for a functor $F: \mathcal{C} \to \mathcal{D}$ to be left adjoint to $G: \mathcal{D} \to \mathcal{C}$.
>
> (ii) Show that left adjoints preserve initial objects: if $I$ is initial in $\mathcal{C}$ and $F \dashv G$, then $FI$ is initial in $\mathcal{D}$.
>
> (iii) Show dually that right adjoints preserve terminal objects.

*Proof sketch for (ii).* We need to show $\mathcal{D}(FI, B)$ is a singleton for every $B \in \mathcal{D}$. By the adjunction bijection, $\mathcal{D}(FI, B) \cong \mathcal{C}(I, GB)$. Since $I$ is initial, $\mathcal{C}(I, GB)$ has exactly one element. Hence so does $\mathcal{D}(FI, B)$, making $FI$ initial. $\square$

> [!NOTE] Exercise 14
> Let $\mathbf{Bij}$ be the category whose objects are finite sets and whose morphisms are bijections. Define two functors:
> - $\mathrm{Sym}: \mathbf{Bij} \to \mathbf{Set}$, sending a finite set $X$ to its set of permutations $\mathrm{Sym}(X) = \{\sigma: X \xrightarrow{\sim} X\}$.
> - $\mathrm{Ord}: \mathbf{Bij} \to \mathbf{Set}$, sending a finite set $X$ to its set of total orderings.
>
> Show that $|\mathrm{Sym}(X)| = |\mathrm{Ord}(X)| = n!$ for every $n$-element set $X$, so the two functors are pointwise isomorphic. However, show that $\mathrm{Sym}$ and $\mathrm{Ord}$ are **not** naturally isomorphic as functors $\mathbf{Bij} \to \mathbf{Set}$.

*Hint.* A natural isomorphism would give, for each finite set $X$, a bijection $\alpha_X: \mathrm{Sym}(X) \to \mathrm{Ord}(X)$ natural in $X$. Naturality requires $\alpha_X$ to be equivariant with respect to the action of $\mathrm{Sym}(X)$ on itself by conjugation and on orderings by relabeling. Consider the case $|X| = 2$ or $|X| = 3$ and derive a contradiction.

---

## References

| Reference Name | Brief Summary | Link to Reference |
|---|---|---|
| Tom Leinster, Part III Category Theory (Cambridge, 2000) | Original Part III course: 24 lectures, 4 problem sheets; source of all 14 exercises in this note | [Course page](https://webhomes.maths.ed.ac.uk/~tl/categories/index.html) |
| Tom Leinster, *Basic Category Theory* (Cambridge UP, 2014; arXiv 2016) | Self-contained introduction to categories, adjunctions, representables, and limits; freely available | [arXiv:1612.09375](https://arxiv.org/abs/1612.09375) |
| Saunders Mac Lane, *Categories for the Working Mathematician* (Springer, 1971; 2nd ed. 1998) | The original comprehensive reference; Leinster calls it "still the best all-round book on the subject" | [Springer](https://link.springer.com/book/10.1007/978-1-4757-4721-8) |
| Francis Borceux, *Handbook of Categorical Algebra*, 3 vols. (Cambridge UP, 1994) | Comprehensive modern treatment; covers adjoint functor theorems, monads, enriched categories | [Cambridge UP](https://www.cambridge.org/core/books/handbook-of-categorical-algebra/AB4CF3C8DC4B1E59A97BF0CF7CCDA9EB) |
| Colin McLarty, *Elementary Categories, Elementary Toposes* (Oxford UP, 1992) | Accessible entry point; covers basic category theory en route to topos theory | [Oxford UP](https://global.oup.com/academic/product/elementary-categories-elementary-toposes-9780198514732) |
| F. William Lawvere & Stephen Schanuel, *Conceptual Mathematics* (Cambridge UP, 1997; 2nd ed. 2009) | Introductory approach; motivation from first principles | [Cambridge UP](https://www.cambridge.org/core/books/conceptual-mathematics/C1EC1E30E6B8A34D16FC96F58793D50E) |
| Emily Riehl, *Category Theory in Context* (Dover, 2016) | Graduate-level treatment with detailed proofs; freely available online | [math.jhu.edu](https://math.jhu.edu/~eriehl/context.pdf) |
| Samuel Eilenberg & Saunders Mac Lane, "General Theory of Natural Equivalences" (1945) | Founding paper introducing categories, functors, and natural transformations | [Trans. AMS](https://www.ams.org/journals/tran/1945-058-00/S0002-9947-1945-0013131-6/) |
