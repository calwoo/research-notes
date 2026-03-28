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
- [[#9. Leinster: Examples and Perspective|9. Leinster: Examples and Perspective]]
  - [[#9.1 Abelianisation as an Adjunction|9.1 Abelianisation as an Adjunction]]
  - [[#9.2 Stone-Cech Compactification|9.2 Stone-Cech Compactification]]
  - [[#9.3 What Category Theory Is About|9.3 What Category Theory Is About]]
- [[#10. Size Matters 📏|10. Size Matters 📏]]
  - [[#10.1 Small and Large Categories|10.1 Small and Large Categories]]
  - [[#10.2 Locally Small Categories|10.2 Locally Small Categories]]
  - [[#10.3 Essentially Small Categories|10.3 Essentially Small Categories]]
  - [[#10.4 Why Size Matters: Russell's Paradox and Adjunctions|10.4 Why Size Matters: Russell's Paradox and Adjunctions]]
  - [[#10.5 Grothendieck Universes|10.5 Grothendieck Universes]]
  - [[#10.6 The Local Smallness Convention|10.6 The Local Smallness Convention]]
- [[#11. The 2-Category of Categories 🔷|11. The 2-Category of Categories 🔷]]
  - [[#11.1 Motivation|11.1 Motivation]]
  - [[#11.2 Definition of a 2-Category|11.2 Definition of a 2-Category]]
  - [[#11.3 Cat as a 2-Category|11.3 Cat as a 2-Category]]
  - [[#11.4 2-Functors and 2-Natural Transformations|11.4 2-Functors and 2-Natural Transformations]]
  - [[#11.5 Adjunctions in a 2-Category|11.5 Adjunctions in a 2-Category]]
  - [[#11.6 Modifications|11.6 Modifications]]
  - [[#11.7 The Godement Product|11.7 The Godement Product]]
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

> [!INFO] Leinster's notation convention
> Leinster uses $\mathcal{A}$ (calligraphic) for a generic/variable category, and boldface $\mathbf{Set}$, $\mathbf{Grp}$, etc., for named categories. He writes $\mathcal{A}(A, B)$ uniformly for hom-sets rather than $\mathrm{Hom}_{\mathcal{A}}(A, B)$. These notes follow this convention. Leinster also uses the term *map* interchangeably with *morphism* and *arrow*, and emphasises that "map" need not mean a function between sets.

> [!EXAMPLE]- The category of topological spaces and homotopy classes
> Leinster uses $\mathbf{Top}$ in the usual sense (continuous maps as morphisms), but also highlights that one can form the *homotopy category* $h\mathbf{Top}$ by taking the same objects but replacing morphisms with homotopy classes $[X, Y]$ of continuous maps. In $h\mathbf{Top}$, two spaces are isomorphic if and only if they are *homotopy equivalent* — a much weaker relation than homeomorphism. For instance, $\mathbb{R}^n$ and the one-point space $\{*\}$ are isomorphic in $h\mathbf{Top}$ (both are contractible), but not homeomorphic.
>
> This illustrates a key lesson: changing the morphisms of a category changes what "sameness" means.

> [!EXAMPLE] The matrix category Mat_k
> 💡 Leinster uses this example to illustrate that objects need not be "sets with structure." Fix a field $k$. Define the category $\mathbf{Mat}_k$ by:
> - Objects: the natural numbers $0, 1, 2, \ldots$
> - Morphisms: $\mathbf{Mat}_k(m, n)$ is the set of $n \times m$ matrices over $k$.
> - Composition: matrix multiplication — if $A \in \mathbf{Mat}_k(m,n)$ and $B \in \mathbf{Mat}_k(n,p)$, then $B \circ A$ is the $p \times m$ matrix $BA$.
> - Identities: $\mathrm{id}_n$ is the $n \times n$ identity matrix $I_n$.
>
> Associativity follows from associativity of matrix multiplication. This category is equivalent to $\mathbf{FDVect}_k$ (Exercise 9), but the two are not *isomorphic* as categories — they differ in how many objects they have.

> [!EXAMPLE] The homotopy category Hty
> The *homotopy category* $\mathbf{Hty}$ has topological spaces as objects but takes as morphisms the *homotopy classes* $[f]$ of continuous maps $f: X \to Y$. Two maps $f, g: X \to Y$ are *homotopic* if there exists a continuous $H: X \times [0,1] \to Y$ with $H(-,0) = f$ and $H(-,1) = g$.
>
> Composition is well-defined on homotopy classes: if $f_0 \simeq f_1: X \to Y$ and $g_0 \simeq g_1: Y \to Z$, then $g_0 \circ f_0 \simeq g_1 \circ f_1$. Associativity and unit laws are inherited from $\mathbf{Top}$. This is a genuinely important example because it is **not** concretely realizable: morphisms are equivalence classes, not functions between the underlying sets.

> [!INFO] Leinster's notation
> Leinster consistently writes $\mathcal{C}(A, B)$ for hom-sets (rather than $\mathrm{Hom}_{\mathcal{C}}(A, B)$), and uses $\mathbf{A}, \mathbf{B}$ (bold) for named categories and $\mathcal{A}, \mathcal{B}$ (calligraphic) for variable categories. These notes follow his convention. When Leinster writes $\mathbf{C}(A, B) = \{f : A \to B\}$, he always means a *set* (assuming local smallness), not a proper class. This is why local smallness is silently assumed in most results.

### 1.4 Categorical Structures as Categories

Two seemingly degenerate cases are fundamental.

**Definition (Poset as a category).** Let $(P, \leq)$ be a *poset* (partially ordered set). Define a category $\mathcal{P}$ by:
- Objects: elements of $P$.
- Morphisms: $\mathcal{P}(a,b)$ contains exactly one element (conventionally denoted $a \leq b$ or $*$) if $a \leq b$, and is empty otherwise.

The transitivity of $\leq$ supplies composition ($a \leq b$ and $b \leq c$ imply $a \leq c$), and reflexivity supplies identities ($a \leq a$). Associativity is trivial since there is at most one morphism between any two objects.

> [!INFO] Leinster's formulation of poset categories
> Leinster emphasises that a *preorder* (where the antisymmetry axiom $a \leq b$ and $b \leq a \Rightarrow a = b$ is dropped) gives an equally valid category. In a preorder, there may be distinct objects $a \neq b$ with morphisms in both directions — corresponding categorically to two distinct objects that are both isomorphic to each other. A poset is a preorder in which the only isomorphisms are the identity morphisms.

**Definition (Monoid as a one-object category).** A *monoid* $(M, \cdot, e)$ — a set with an associative binary operation and a unit element — determines a category $\mathbf{B}M$ with:
- A single object $*$.
- $\mathbf{B}M(*,*) = M$ (every element of $M$ is a morphism).
- Composition given by the monoid operation: $g \circ f := g \cdot f$.
- Identity: the unit element $e$.

The monoid axioms are exactly the category axioms. **A group is a monoid in which every element is invertible; viewed as a one-object category, it is a category in which every morphism is an isomorphism.**

> [!TIP] The general principle: categories with one object
> Leinster uses this example to introduce an important general principle: a category with exactly one object is precisely a monoid. More generally, a *groupoid* (a category in which every morphism is an isomorphism) with one object is precisely a group. This prefigures the idea that functors from $\mathbf{B}G$ to $\mathbf{Set}$ are $G$-sets, which is the categorical formulation of group actions (see §11.1 of File 2).

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

> [!INFO] Leinster's emphasis on functors as the primary objects
> Leinster opens Chapter 1 of *Basic Category Theory* by saying that "category theory takes a bird's eye view of mathematics." A key point he makes is that a functor $F : \mathcal{A} \to \mathcal{B}$ must specify *both* what it does to objects and what it does to maps, and the two specifications must be compatible via the functoriality axioms. He cautions that specifying only the object-level map is not enough — the map-level action is essential data.

> [!EXAMPLE] Forgetful functors
> The *forgetful functor* $U: \mathbf{Grp} \to \mathbf{Set}$ sends a group $(G, \cdot, e, {}^{-1})$ to its underlying set $G$, and sends each homomorphism to the underlying function. It "forgets" the group structure. Similarly one has forgetful functors $\mathbf{Ring} \to \mathbf{Ab} \to \mathbf{Set}$, $\mathbf{Top} \to \mathbf{Set}$, etc.

> [!EXAMPLE] Free functors
> The *free group functor* $F: \mathbf{Set} \to \mathbf{Grp}$ sends a set $S$ to the free group $F(S)$ generated by $S$. A function $f: S \to T$ induces a homomorphism $F(f): F(S) \to F(T)$ by extending $f$ to a group homomorphism. Free functors are the left adjoints of forgetful functors.

> [!EXAMPLE] Power set functor
> The covariant *power set functor* $\mathcal{P}: \mathbf{Set} \to \mathbf{Set}$ sends a set $X$ to its power set $\mathcal{P}(X) = \{S \mid S \subseteq X\}$. For a function $f: X \to Y$, the map $\mathcal{P}(f): \mathcal{P}(X) \to \mathcal{P}(Y)$ is the direct image: $\mathcal{P}(f)(S) = f(S) = \{f(s) \mid s \in S\}$.

> [!EXAMPLE]- Functors between posets and monotone maps
> 💡 A functor $F : \mathcal{P} \to \mathcal{Q}$ between two posets (viewed as categories) is exactly a *monotone map*: a function $f : P \to Q$ with $a \leq_P b \Rightarrow f(a) \leq_Q f(b)$.
>
> *Verification.* A functor must send each morphism $a \leq b$ in $\mathcal{P}$ (i.e., each pair with $a \leq_P b$) to a morphism $f(a) \leq f(b)$ in $\mathcal{Q}$ — which is exactly monotonicity. Preservation of composition is automatic (there is at most one morphism between any two objects), as is preservation of identities.
>
> This shows that adjunctions between posets are exactly *Galois connections* — a classical concept in order theory. See File 4, §2 for details.

> [!EXAMPLE]- Functors out of a monoid: representations
> A functor $F : \mathbf{B}M \to \mathbf{Set}$ from a monoid $M$ (viewed as a one-object category) to $\mathbf{Set}$ is exactly a *left $M$-action* on the set $S = F(*)$: each element $m \in M$ acts as a function $F(m) : S \to S$, and functoriality gives $F(mn) = F(m) \circ F(n)$ and $F(e) = \mathrm{id}_S$.
>
> Similarly, a functor $\mathbf{B}G \to \mathbf{Set}$ from a group $G$ is a *$G$-set*. A functor $\mathbf{B}G \to \mathbf{Vect}_k$ is a *$k$-linear representation* of $G$. This is the categorical origin of representation theory.

> [!EXAMPLE] The identity and constant functors
> For any category $\mathcal{C}$: the *identity functor* $\mathrm{id}_\mathcal{C} : \mathcal{C} \to \mathcal{C}$ sends every object and morphism to itself. For any object $D \in \mathcal{D}$, the *constant functor* $\Delta_D : \mathcal{C} \to \mathcal{D}$ sends every object to $D$ and every morphism to $\mathrm{id}_D$. Constant functors play a central role in the definition of limits (File 3, §1).

### 4.2 Contravariant Functors

**Definition (Contravariant functor).** A *contravariant functor* from $\mathcal{C}$ to $\mathcal{D}$ is a covariant functor $F: \mathcal{C}^{\mathrm{op}} \to \mathcal{D}$. Concretely, it assigns to each $f: A \to B$ in $\mathcal{C}$ a morphism $Ff: FB \to FA$ in $\mathcal{D}$, reversing the direction, with $F(g \circ f) = Ff \circ Fg$.

> [!INFO] Leinster on contravariant functors
> Leinster consistently avoids using the term "contravariant functor" as a standalone concept, preferring instead to say "$F$ is a functor $\mathcal{A}^{\mathrm{op}} \to \mathcal{B}$." This has the advantage that all general results about functors apply automatically by viewing $\mathcal{A}^{\mathrm{op}}$ as the domain. The term "contravariant" is retained here for expository clarity, but beware that some modern treatments (following Leinster) omit it entirely.

> [!EXAMPLE] Contravariant power set
> The contravariant power set functor sends $f: X \to Y$ to the preimage map $f^{-1}: \mathcal{P}(Y) \to \mathcal{P}(X)$. This is functorial because $(g \circ f)^{-1} = f^{-1} \circ g^{-1}$.

> [!EXAMPLE]- The dual vector space functor
> The functor $(-)^* : \mathbf{Vect}_k^{\mathrm{op}} \to \mathbf{Vect}_k$ sending $V \mapsto V^* = \mathrm{Hom}_k(V, k)$ and $f : V \to W$ to $f^* : W^* \to V^*$ (precomposition with $f$) is a contravariant functor. It is *not* naturally isomorphic to the identity functor on $\mathbf{FDVect}_k$ (there is no canonical isomorphism $V \cong V^*$), but the double-dual $(-)^{**}$ *is* naturally isomorphic to the identity — this is the content of the canonical natural transformation $\alpha : \mathrm{id} \Rightarrow (-)^{**}$ described in §5.1.

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

> [!INFO] Leinster's phrasing of naturality
> Leinster phrases the naturality condition precisely as: "the square commutes for every map $f : A \to B$ in $\mathcal{A}$." He emphasises that a natural transformation $\alpha : F \Rightarrow G$ is not just a collection of morphisms — the commutativity of the naturality square for *every* morphism in $\mathcal{A}$ is what makes $\alpha$ "natural" (i.e., independent of arbitrary choices). The original motivation of Eilenberg and Mac Lane was exactly to give a precise definition distinguishing "natural" from "non-natural" isomorphisms.

> [!EXAMPLE] A canonical natural transformation
> Let $V$ be a finite-dimensional real vector space. There is a natural transformation $\alpha: \mathrm{id}_{\mathbf{FDVect}_\mathbb{R}} \Rightarrow (-)^{**}$ (the double-dual functor) whose component at $V$ is the canonical map
> $$\alpha_V: V \to V^{**}, \quad v \mapsto \hat{v}, \quad \hat{v}(\phi) := \phi(v).$$
> Naturality: for any linear map $f: V \to W$, we need $\alpha_W \circ f = f^{**} \circ \alpha_V$. This holds because $\widehat{f(v)}(\psi) = \psi(f(v)) = (f^*\psi)(v) = \widehat{v}(f^*\psi) = (f^{**}\hat{v})(\psi)$.
>
> In contrast, there is a non-natural isomorphism $V \cong V^*$ (requiring a choice of basis), illustrating that naturality is a substantive condition.

> [!EXAMPLE]- Natural transformations between functors from a group
> Let $G$ be a group viewed as a one-object category $\mathbf{B}G$, and let $F, H : \mathbf{B}G \to \mathbf{Set}$ be two functors, i.e., two $G$-sets $S = F(*)$ and $T = H(*)$. A natural transformation $\alpha : F \Rightarrow H$ consists of a single component $\alpha_* : S \to T$ (one component per object, and $\mathbf{B}G$ has one object). The naturality square for each $g \in G$ (viewed as a morphism $* \to *$) requires:
> $$\alpha_* \circ F(g) = H(g) \circ \alpha_*,$$
> i.e., $\alpha_*$ must commute with the $G$-action on both sides. This is precisely the condition that $\alpha_*$ is a *$G$-equivariant map* (a morphism of $G$-sets). **Natural transformations between $G$-set functors are exactly $G$-equivariant maps.**

> [!EXAMPLE]- Determinant as a natural transformation
> 💡 For each $n \geq 1$, the *determinant* defines a natural transformation $\det : \mathrm{GL}_n \Rightarrow (-)^\times$ from the general linear group functor to the units functor, where both are viewed as functors $\mathbf{CRing} \to \mathbf{Grp}$ from the category of commutative rings to the category of groups.
>
> Explicitly: for a commutative ring $R$, $\mathrm{GL}_n(R)$ is the group of $n \times n$ invertible matrices over $R$, and $R^\times$ is the group of units of $R$. The determinant $\det_R : \mathrm{GL}_n(R) \to R^\times$ is a group homomorphism for each $R$. Naturality: for any ring homomorphism $\phi : R \to S$, the diagram
> $$\det_S \circ \mathrm{GL}_n(\phi) = \phi^\times \circ \det_R$$
> commutes — applying $\phi$ entry-wise to a matrix and then taking the determinant equals first taking the determinant in $R$ and then applying $\phi$.

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
- ($\Rightarrow$) If $F$ is an equivalence with quasi-inverse $G$ and natural isomorphisms $\eta, \varepsilon$: essential surjectivity is immediate since $\varepsilon_D: FGD \xrightarrow{\sim} D$ for every $D \in \mathcal{D}$. Faithfulness: if $Ff = Fg$, apply $G$ to get $GFf = GFg$; since $\eta_A: A \xrightarrow{\sim} GFA$ is natural, $f = \eta_B^{-1} \circ GFf \circ \eta_A = \eta_B^{-1} \circ GFg \circ \eta_A = g$. Fullness: given $h: FA \to FB$, define $f = \eta_B^{-1} \circ G(h) \circ \eta_A: A \to B$; then $Ff = \varepsilon_{FB} \circ FG(h) \circ F(\eta_A)$, and using naturality of $\varepsilon$ ($\varepsilon_{FB} \circ FG(h) = h \circ \varepsilon_{FA}$) together with the triangle identity $\varepsilon_{FA} \circ F(\eta_A) = \mathrm{id}_{FA}$ gives $Ff = h$.
- ($\Leftarrow$) Requires choosing, for each $D \in \mathcal{D}$, an object $GD \in \mathcal{C}$ and an isomorphism $\varepsilon_D: FGD \xrightarrow{\sim} D$. *This direction requires the axiom of choice* to select the $GD$ and $\varepsilon_D$.

**Key distinction:** An isomorphism of categories does not require choice; an equivalence (in the $\Leftarrow$ direction) does.

To see why, contrast the two situations precisely.

In an **isomorphism** $F: \mathcal{C} \to \mathcal{D}$, the quasi-inverse $G$ is *given to you*: $G = F^{-1}$, determined on the nose by the equations $GF = \mathrm{id}_\mathcal{C}$ and $FG = \mathrm{id}_\mathcal{D}$. There is nothing to choose. In fact, $G(D)$ is the *unique* object $C$ with $FC = D$, because $F$ is bijective on objects.

In an **equivalence** constructed from the $(\Leftarrow)$ direction, you are handed only the property: *for every $D \in \mathcal{D}$ there exists some $C \in \mathcal{C}$ and some isomorphism $FС \xrightarrow{\sim} D$*. "There exists" is not "here it is." To define a functor $G: \mathcal{D} \to \mathcal{C}$, you must assign to *each* $D$ a specific object $G(D)$ and a specific isomorphism $\varepsilon_D: FG(D) \xrightarrow{\sim} D$. When $\mathcal{D}$ has infinitely many (or a proper class of) objects, making all these choices simultaneously requires the **axiom of choice**.

> [!EXAMPLE] Choosing bases in $\mathbf{FDVect}_k$
> The functor $F: \mathbf{Mat}_k \to \mathbf{FDVect}_k$ sending $n \mapsto k^n$ is full, faithful, and essentially surjective — every finite-dimensional $k$-vector space $V$ is *isomorphic* to some $k^n$. To build the quasi-inverse $G: \mathbf{FDVect}_k \to \mathbf{Mat}_k$, you must assign to *each* $V$ a specific natural number $G(V) = \dim V$ and a specific isomorphism $\varepsilon_V: k^{\dim V} \xrightarrow{\sim} V$. The isomorphism is just an ordered basis for $V$. So **constructing $G$ requires choosing a basis for every finite-dimensional vector space simultaneously** — a choice function over all of $\mathbf{FDVect}_k$, which is exactly an instance of the axiom of choice.
>
> Note the asymmetry: $F$ (the "canonical" direction, $\mathbf{Mat}_k \to \mathbf{FDVect}_k$) is completely explicit. The quasi-inverse $G$ (the "coordinatization" direction) is non-canonical — there is no preferred basis.

> [!INFO] Foundational aside: the axiom of choice and equivalences
> The theorem "every full, faithful, and essentially surjective functor is an equivalence" is — over a weak foundational base — *equivalent to* a form of the axiom of choice. In **constructive mathematics** (where AC is not available), one instead works with *split* or *cloven* equivalences: the quasi-inverse and the isomorphisms $\eta$, $\varepsilon$ are part of the data, not merely asserted to exist. This is the correct notion in homotopy type theory (where it goes by the name *adjoint equivalence*) and in internal category theory in a topos.

> [!TIP] "Evil" vs. invariant properties
> This asymmetry is the reason category theorists call isomorphism of categories an **"evil"** notion: it asks whether two categories are *equal as structured sets*, which is sensitive to the labeling of objects. Equivalence is the **invariant** notion — it asks only whether the categories have the same shape up to coherent isomorphism. A property or construction is considered well-behaved in category theory if it is *invariant under equivalence*. The principle of equivalence says: never write down a definition that would distinguish between equivalent categories.

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

---

## 9. Leinster: Examples and Perspective 📖

This section collects key examples and philosophical remarks from Leinster's *Basic Category Theory* that complement and enrich the material above.

### 9.1 Abelianisation as an Adjunction

> [!EXAMPLE]- Abelianisation: the most important free-forgetful adjunction
> The *abelianisation* of a group $G$ is the quotient $G^{\mathrm{ab}} = G / [G, G]$, where $[G, G]$ is the *commutator subgroup* (the subgroup generated by $\{ghg^{-1}h^{-1} : g, h \in G\}$). This construction defines a functor $\mathrm{ab} : \mathbf{Grp} \to \mathbf{Ab}$.
>
> The key property is that every group homomorphism from $G$ to an abelian group $A$ factors uniquely through $G^{\mathrm{ab}}$:
> $$\mathbf{Ab}(G^{\mathrm{ab}}, A) \cong \mathbf{Grp}(G, \iota A),$$
> where $\iota : \mathbf{Ab} \hookrightarrow \mathbf{Grp}$ is the inclusion (forgetful) functor. This is the adjunction $\mathrm{ab} \dashv \iota$.
>
> The unit $\eta_G : G \to G^{\mathrm{ab}}$ is the quotient map, and the counit is the identity $\mathrm{id}_A : \mathrm{ab}(\iota A) = A^{\mathrm{ab}} \xrightarrow{\sim} A$ (since $A$ is already abelian, $[A, A] = \{e\}$, so $A^{\mathrm{ab}} = A$).
>
> Leinster uses this as a canonical example of a *reflective subcategory*: $\mathbf{Ab}$ is a reflective full subcategory of $\mathbf{Grp}$, with the reflector being abelianisation.

### 9.2 Stone-Cech Compactification

> [!EXAMPLE]- Stone-Čech compactification as a universal property
> Let $\mathbf{KHaus}$ be the full subcategory of $\mathbf{Top}$ whose objects are compact Hausdorff topological spaces. The inclusion $\iota : \mathbf{KHaus} \hookrightarrow \mathbf{Top}$ has a left adjoint $\beta : \mathbf{Top} \to \mathbf{KHaus}$ called the *Stone-Čech compactification*: for any topological space $X$ and compact Hausdorff space $K$, there is a natural bijection
> $$\mathbf{KHaus}(\beta X, K) \cong \mathbf{Top}(X, K).$$
> The unit $\eta_X : X \to \beta X$ is the canonical dense embedding of $X$ into its Stone-Čech compactification. The universal property says: every continuous map from $X$ to a compact Hausdorff space extends uniquely to $\beta X$.
>
> Leinster uses this as an example of an adjunction that cannot be understood concretely without significant set-theoretic machinery — illustrating that the adjunction *is* the content, not an encoding of some prior construction.

### 9.3 What Category Theory Is About

> [!INFO] Leinster's perspective on the purpose of category theory
> In the preface to *Basic Category Theory*, Leinster writes: "Category theory takes a bird's eye view of mathematics. From high up, we can see the common threads that connect different areas of mathematics, and apply the same abstract structures and concepts across different fields."
>
> He identifies three recurring themes:
> 1. **Sameness:** Equivalences of categories are the right notion of "sameness" for categories — not isomorphism. Two equivalent categories share all categorical properties.
> 2. **Universal properties:** Constructions defined by universal properties are unique up to unique isomorphism, and are preserved by functors that respect the relevant structures.
> 3. **Functoriality:** The distinction between "functorial" and "non-functorial" constructions is deep. A construction that cannot be made into a functor (e.g., choosing a basis for a vector space) is fundamentally different from one that can.
>
> Leinster's book deliberately keeps the scope narrow — categories, functors, natural transformations, adjunctions, representables, and limits — to present these ideas in their cleanest form.

---

## 10. Size Matters 📏

Set-theoretic size is not a bureaucratic afterthought in category theory — it is the boundary condition that determines which theorems are even *statable*. This section makes precise what it means for a category to be small, locally small, or essentially small, and explains why each condition appears as a hypothesis in the major theorems.

### 10.1 Small and Large Categories

**Definition (Small and large categories).** A category $\mathcal{C}$ is *small* if both $\mathrm{ob}(\mathcal{C})$ and $\mathrm{mor}(\mathcal{C}) = \coprod_{A,B \in \mathcal{C}} \mathcal{C}(A,B)$ are *sets* (i.e., elements of a fixed set-theoretic universe, not proper classes). A category is *large* if either collection is a proper class.

> [!INFO] Examples of small and large categories
> **Large categories:** $\mathbf{Set}$, $\mathbf{Grp}$, $\mathbf{Top}$, $\mathbf{Vect}_k$, $\mathbf{Ring}$, $\mathbf{Ab}$ — their object collections are proper classes (there is no set of all sets, no set of all groups, etc.).
>
> **Small categories:** Any poset $(P, \leq)$ viewed as a category (when $P$ is a set); any monoid $M$ viewed as a one-object category $\mathbf{B}M$; the *ordinal categories* $\mathbf{n} = \{0 \to 1 \to \cdots \to n\}$ for $n \geq 0$; the *empty category* $\mathbf{0}$; the *terminal category* $\mathbf{1}$.
>
> 💡 Smallness is an absolute condition on the collection of objects and morphisms, whereas local smallness (§10.2) is a condition on each individual hom-collection.

### 10.2 Locally Small Categories

**Definition (Locally small).** A category $\mathcal{C}$ is *locally small* if for every pair of objects $A, B \in \mathcal{C}$, the hom-collection $\mathcal{C}(A, B)$ is a set.

Every small category is locally small. Among large categories, all the named examples — $\mathbf{Set}$, $\mathbf{Grp}$, $\mathbf{Top}$, $\mathbf{Vect}_k$ — are locally small, since morphisms between any two fixed objects form a set (e.g., the set of all functions between two sets, the set of all group homomorphisms, etc.).

> [!WARNING] Local smallness is needed for the Yoneda lemma
> The Yoneda lemma asserts that for $F: \mathcal{C} \to \mathbf{Set}$ and $A \in \mathcal{C}$, natural transformations $\mathcal{C}(A, -) \Rightarrow F$ are in bijection with $FA$. The domain functor $\mathcal{C}(A, -)$ is only well-defined as a functor to $\mathbf{Set}$ if $\mathcal{C}(A, B)$ is a *set* for each $B$ — i.e., if $\mathcal{C}$ is locally small. Without this, $\mathcal{C}(A, B)$ could be a proper class, and it would not be an object of $\mathbf{Set}$.

### 10.3 Essentially Small Categories

**Definition (Essentially small).** A category $\mathcal{C}$ is *essentially small* if it is *equivalent* (in the sense of §6) to a small category. Equivalently, $\mathcal{C}$ has a *small skeleton* — a full subcategory containing exactly one object from each isomorphism class.

> [!EXAMPLE] The category of finite sets
> The category $\mathbf{FinSet}$ of all finite sets is *not* small: there are class-many finite sets (e.g., $\{0\}, \{1\}, \{2\}, \ldots$ are all distinct one-element sets in ZFC). However, $\mathbf{FinSet}$ is essentially small: it is equivalent to the full subcategory $\{\mathbf{0}, \mathbf{1}, \mathbf{2}, \ldots\}$ where $\mathbf{n} = \{0, 1, \ldots, n-1\}$, which is a small category.
>
> This illustrates that essential smallness is the invariant notion of smallness under categorical equivalence.

### 10.4 Why Size Matters: Russell's Paradox and Adjunctions

📐 The need for size distinctions is not philosophical — it is forced by set theory.

**Russell's paradox and $\mathbf{Set}$.** There is no set of all sets: the collection $\{X : X \notin X\}$ leads to a contradiction. Thus $\mathrm{ob}(\mathbf{Set})$ is a proper class, and $\mathbf{Set}$ is large but not small.

**Adjunctions require sets.** The hom-set bijection in an adjunction,
$$\mathcal{D}(FA, B) \cong \mathcal{C}(A, GB),$$
is a natural bijection of *sets*. If $\mathcal{C}$ and $\mathcal{D}$ are not locally small, neither side is a set, and the statement is not well-formed. **Local smallness of $\mathcal{C}$ and $\mathcal{D}$ is a standing hypothesis in the definition of adjunction via hom-sets.**

> [!WARNING] The General Adjoint Functor Theorem and the solution set condition
> The *General Adjoint Functor Theorem* (Freyd's theorem) states: a functor $G: \mathcal{D} \to \mathcal{C}$ has a left adjoint if and only if it preserves all small limits and satisfies the *solution set condition* — for each $A \in \mathcal{C}$, there exists a *set* (not a proper class) of morphisms $A \to GB$ that "generates" all morphisms from $A$ to the image of $G$.
>
> The solution set condition is precisely a smallness hypothesis. Without it, one would need to take an initial object in a category of potentially class-many candidates, which ZFC set theory does not permit. This is why the theorem is not vacuous: most functors satisfy limit preservation, but fail the solution set condition in pathological cases.

### 10.5 Grothendieck Universes

One clean resolution of the size problem is via *Grothendieck universes*.

**Definition (Grothendieck universe).** A *Grothendieck universe* $\mathcal{U}$ is a set satisfying:
1. If $x \in \mathcal{U}$ and $y \in x$, then $y \in \mathcal{U}$ (transitivity).
2. If $x \in \mathcal{U}$, then $\mathcal{P}(x) \in \mathcal{U}$ (power sets).
3. If $I \in \mathcal{U}$ and $\{x_i\}_{i \in I} \subseteq \mathcal{U}$, then $\bigcup_{i \in I} x_i \in \mathcal{U}$ (unions).
4. $\mathbb{N} \in \mathcal{U}$ (contains the natural numbers).

> [!INFO] How Grothendieck universes resolve size issues
> Fix a universe $\mathcal{U}$. Call a set *$\mathcal{U}$-small* if it is an element of $\mathcal{U}$. Then:
> - "$\mathcal{U}$-sets" play the role of "small sets."
> - "$\mathcal{U}$-classes" (collections of $\mathcal{U}$-small sets) play the role of "large sets" / "proper classes."
> - The category $\mathbf{Set}_{\mathcal{U}}$ of all $\mathcal{U}$-small sets is an honest *set* from the perspective of a larger universe $\mathcal{U}'$, allowing us to form functor categories, slice categories, and presheaf categories without set-theoretic illegality.
>
> The existence of arbitrarily large Grothendieck universes is not provable in ZFC alone; it requires an additional axiom (the *universe axiom*, equivalent to the existence of inaccessible cardinals in ZFC). SGA and much of Grothendieck's algebraic geometry is written in this framework.

### 10.6 The Local Smallness Convention

🔑 **Throughout category theory, local smallness is the default standing hypothesis.** The major theorems — the Yoneda lemma (File 2), the adjoint functor theorems (File 4), representability criteria, and limits (File 3) — all require at minimum that the relevant categories be locally small. This hypothesis is frequently left implicit in the literature, but it is never vacuous.

**The practical upshot:** when defining a new category, one should immediately verify local smallness. For most naturally occurring categories this is immediate (morphisms are functions, homomorphisms, or homotopy classes, all of which form sets). The exceptions — such as functor categories $[\mathcal{C}, \mathcal{D}]$ where $\mathcal{C}$ is large — require more care.

> [!NOTE] Riehl Exercise: Size
> **(i)** Show that the category of all small categories $\mathbf{Cat}$ is large but locally small.
> *(Hint: the collection of all small categories is not a set by a cardinality argument, but for any two small categories $\mathcal{C}$ and $\mathcal{D}$, the collection of functors $\mathcal{C} \to \mathcal{D}$ is a set.)*
>
> **(ii)** Let $\mathcal{C}$ be a locally small category and fix $A \in \mathcal{C}$. Show that the functor $\mathcal{C}(A, -): \mathcal{C} \to \mathbf{Set}$ is well-defined — i.e., that it sends each object $B$ to an element of $\mathrm{ob}(\mathbf{Set})$ (a genuine set, not a proper class).
>
> **(iii)** Give an example of a category that is locally small but not essentially small.
> *(Hint: consider $\mathbf{Set}$ itself — it is locally small, but its isomorphism classes are not a set, since for each cardinality $\kappa$ there is a proper class of sets of cardinality $\kappa$.)*

---

## 11. The 2-Category of Categories 🔷

The structure of $\mathbf{Cat}$ — categories, functors, and natural transformations — exhibits a strict hierarchy: there are morphisms (functors) between objects (categories), and morphisms between morphisms (natural transformations). This is the prototype of a *2-category*, a structure with three levels of data.

### 11.1 Motivation

In §5, we defined functor categories $[\mathcal{C}, \mathcal{D}]$ whose objects are functors and whose morphisms are natural transformations. We found two composition operations:

- **Vertical composition** (§5.2): given $\alpha: F \Rightarrow G$ and $\beta: G \Rightarrow H$ in $[\mathcal{C}, \mathcal{D}]$, the composite $\beta \circ \alpha: F \Rightarrow H$ has components $(\beta \circ \alpha)_A = \beta_A \circ \alpha_A$.
- **Horizontal composition** (§5.4): given $\alpha: F \Rightarrow G: \mathcal{C} \to \mathcal{D}$ and $\beta: H \Rightarrow K: \mathcal{D} \to \mathcal{E}$, the *Godement product* $\beta * \alpha: HF \Rightarrow KG: \mathcal{C} \to \mathcal{E}$.

💡 The existence of two interacting composition operations, together with an interchange law relating them, is precisely the data of a *2-category*. The category $\mathbf{Cat}$ is the canonical example.

### 11.2 Definition of a 2-Category

**Definition (2-category).** A *2-category* $\mathcal{K}$ consists of:

- **0-cells** (objects): a collection $\mathrm{ob}(\mathcal{K})$, typically denoted $A, B, C, \ldots$
- **1-cells** (morphisms between 0-cells): for each pair $A, B$, a collection $\mathcal{K}(A, B)$ of 1-cells $f: A \to B$.
- **2-cells** (morphisms between 1-cells): for each pair of parallel 1-cells $f, g: A \to B$, a collection $\mathcal{K}(f, g)$ of 2-cells $\alpha: f \Rightarrow g$.
- **Vertical composition of 2-cells**: for $\alpha: f \Rightarrow g$ and $\beta: g \Rightarrow h$ (same 0-cell boundaries), a 2-cell $\beta \circ_v \alpha: f \Rightarrow h$, satisfying associativity and unit laws with respect to *identity 2-cells* $\mathrm{id}_f: f \Rightarrow f$.
- **Horizontal composition of 2-cells** (Godement product): for $\alpha: f \Rightarrow g: A \to B$ and $\beta: h \Rightarrow k: B \to C$, a 2-cell $\beta *\alpha: h \circ f \Rightarrow k \circ g: A \to C$.
- **Composition and identities for 1-cells**: as in an ordinary category.

These data must satisfy the **interchange law**:
$$(\beta' \circ_v \beta) * (\alpha' \circ_v \alpha) = (\beta' * \alpha') \circ_v (\beta * \alpha),$$
for composable pairs $(\alpha, \alpha')$ of 2-cells with $f \xRightarrow{\alpha} g \xRightarrow{\alpha'} h: A \to B$ and $k \xRightarrow{\beta} l \xRightarrow{\beta'} m: B \to C$.

> [!INFO] The interchange law visualized
> The interchange law says that two different ways of composing a $2 \times 2$ grid of 2-cells give the same result:
>
> $$\begin{array}{ccc} A & \xrightarrow{f} & B & \xrightarrow{h} & C \\ & \Downarrow\!\alpha & & \Downarrow\!\beta & \\ A & \xrightarrow{g} & B & \xrightarrow{k} & C \\ & \Downarrow\!\alpha' & & \Downarrow\!\beta' & \\ A & \xrightarrow{h'} & B & \xrightarrow{m} & C \end{array}$$
>
> One can first compose vertically in each column (getting $\alpha' \circ_v \alpha$ and $\beta' \circ_v \beta$) and then compose horizontally, or first compose horizontally in each row (getting $\beta * \alpha$ and $\beta' * \alpha'$) and then compose vertically. The law asserts these are equal.

> [!WARNING] Strict vs. weak 2-categories
> The definition above is for a *strict* 2-category, where all associativity and unit laws hold on the nose (as equations). A *bicategory* (also called a *weak 2-category*) relaxes these to coherent isomorphisms — 2-cells asserting that $f \circ (g \circ h) \cong (f \circ g) \circ h$, etc. Most naturally occurring examples ($\mathbf{Cat}$, the 2-category of rings and bimodules) are strict, but higher-dimensional analogues typically require weak structures.

### 11.3 Cat as a 2-Category

**Theorem.** $\mathbf{Cat}$ is a strict 2-category with:
- **0-cells:** (small) categories.
- **1-cells:** functors $F: \mathcal{C} \to \mathcal{D}$.
- **2-cells:** natural transformations $\alpha: F \Rightarrow G$.
- **Vertical composition:** $(\beta \circ_v \alpha)_A = \beta_A \circ \alpha_A$ (componentwise composition in $\mathcal{D}$).
- **Horizontal composition (Godement product):** $(\beta * \alpha)_A = \beta_{GA} \circ H(\alpha_A) = K(\alpha_A) \circ \beta_{FA}$ (both expressions agree by naturality of $\beta$).
- **Composition of 1-cells:** functor composition.

🔑 **The two expressions for horizontal composition are equal precisely because naturality of $\beta$ gives a commutative square.** This is the content of §11.7.

> [!TIP] Working with 2-cells in Cat
> When computing in $\mathbf{Cat}$, remember:
> - Vertical composition is just pointwise composition of natural transformation components.
> - Horizontal composition (whiskering) requires choosing whether to "whisk" with the left or right functor first — but naturality guarantees both choices coincide.
> - The identity 2-cell on a functor $F$ is the natural transformation $\mathrm{id}_F$ with $(\mathrm{id}_F)_A = \mathrm{id}_{FA}$.

### 11.4 2-Functors and 2-Natural Transformations

**Definition (2-functor).** A *strict 2-functor* $\Phi: \mathcal{K} \to \mathcal{L}$ between 2-categories assigns:
- To each 0-cell $A \in \mathcal{K}$, a 0-cell $\Phi A \in \mathcal{L}$.
- To each 1-cell $f: A \to B$, a 1-cell $\Phi f: \Phi A \to \Phi B$.
- To each 2-cell $\alpha: f \Rightarrow g$, a 2-cell $\Phi \alpha: \Phi f \Rightarrow \Phi g$.

preserving all compositions (vertical and horizontal) and all identities (for 0-cells, 1-cells, and 2-cells) strictly.

> [!EXAMPLE]- The presheaf 2-functor
> The assignment $\mathcal{C} \mapsto [\mathcal{C}^{\mathrm{op}}, \mathbf{Set}]$ (sending a small category to its presheaf category) extends to a 2-functor $\mathbf{Cat}^{\mathrm{op}} \to \mathbf{CAT}$ (where $\mathbf{CAT}$ is the very large 2-category of all locally small categories):
>
> - A functor $F: \mathcal{C} \to \mathcal{D}$ is sent to the *precomposition functor* $F^*: [\mathcal{D}^{\mathrm{op}}, \mathbf{Set}] \to [\mathcal{C}^{\mathrm{op}}, \mathbf{Set}]$, $P \mapsto P \circ F^{\mathrm{op}}$.
> - A natural transformation $\alpha: F \Rightarrow G$ is sent to a natural transformation $\alpha^*: G^* \Rightarrow F^*$ whose component at $P \in [\mathcal{D}^{\mathrm{op}}, \mathbf{Set}]$ is post-composition with $\alpha$.
>
> This is contravariant in 1-cells (hence $\mathbf{Cat}^{\mathrm{op}}$) and covariant in 2-cells. Preservation of compositions follows from associativity of functor composition.

**Definition (2-natural transformation).** A *2-natural transformation* $\Gamma: \Phi \Rightarrow \Psi$ between 2-functors $\Phi, \Psi: \mathcal{K} \to \mathcal{L}$ assigns to each 0-cell $A \in \mathcal{K}$ a 1-cell $\Gamma_A: \Phi A \to \Psi A$ in $\mathcal{L}$, such that for each 1-cell $f: A \to B$, there is a *specified 2-cell* $\Gamma_f: \Gamma_B \circ \Phi f \Rightarrow \Psi f \circ \Gamma_A$ (the 2-dimensional naturality condition), satisfying coherence conditions for 2-cell composition.

*Remark.* Unlike ordinary natural transformations — whose components are 0-cells (objects) — the components of a 2-natural transformation are 1-cells, and naturality is expressed by a 2-cell filling a square rather than an equation.

### 11.5 Adjunctions in a 2-Category

The notion of adjunction generalizes from $\mathbf{Cat}$ to any 2-category.

**Definition (Adjunction in a 2-category).** An *adjunction* $F \dashv G$ in a 2-category $\mathcal{K}$ consists of:
- 0-cells $A, B$.
- 1-cells $F: A \to B$ (the *left adjoint*) and $G: B \to A$ (the *right adjoint*).
- 2-cells $\eta: \mathrm{id}_A \Rightarrow G \circ F$ (the *unit*) and $\varepsilon: F \circ G \Rightarrow \mathrm{id}_B$ (the *counit*).

satisfying the *triangle identities* (expressed as equalities of 2-cells):
$$(\varepsilon * \mathrm{id}_F) \circ_v (\mathrm{id}_F * \eta) = \mathrm{id}_F, \qquad (\mathrm{id}_G * \varepsilon) \circ_v (\eta * \mathrm{id}_G) = \mathrm{id}_G.$$

When $\mathcal{K} = \mathbf{Cat}$, this recovers the usual definition: $F: \mathcal{C} \to \mathcal{D}$, $G: \mathcal{D} \to \mathcal{C}$, with unit $\eta: \mathrm{id}_{\mathcal{C}} \Rightarrow GF$ and counit $\varepsilon: FG \Rightarrow \mathrm{id}_{\mathcal{D}}$ satisfying the triangle identities of §8.4.

> [!TIP] Adjunctions in 2-categories: what generalizes and what does not
> In $\mathbf{Cat}$, an adjunction $F \dashv G$ is equivalently described by either the unit-counit data or the hom-set bijection $\mathcal{D}(FA, B) \cong \mathcal{C}(A, GB)$. In a general 2-category, only the unit-counit formulation generalizes directly (since there are no "hom-sets" for 1-cells in an abstract 2-category). The hom-set bijection requires the 2-category to be *locally small* (i.e., each hom-category $\mathcal{K}(A, B)$ must be a genuine category with a set of objects).

### 11.6 Modifications

Having natural transformations between functors and 2-natural transformations between 2-functors, one can go one level higher.

**Definition (Modification).** Given 2-functors $\Phi, \Psi: \mathcal{K} \to \mathcal{L}$ and 2-natural transformations $\Gamma, \Delta: \Phi \Rightarrow \Psi$, a *modification* $\Sigma: \Gamma \Rrightarrow \Delta$ assigns to each 0-cell $A \in \mathcal{K}$ a 2-cell $\Sigma_A: \Gamma_A \Rightarrow \Delta_A$ in $\mathcal{L}$, satisfying a 3-dimensional coherence condition.

*Remark.* Modifications are the 3-cells of the (strict) 3-category $\mathbf{2CAT}$ of 2-categories, 2-functors, 2-natural transformations, and modifications. This hierarchy — categories, 2-categories, 3-categories, ... — is the beginning of the theory of *$(\infty, n)$-categories*, where morphisms exist at all dimensions and become invertible above dimension $n$.

> [!QUESTION] Higher category theory and homotopy
> The relationship between $(\infty, 1)$-categories (where all $k$-morphisms for $k \geq 2$ are invertible) and homotopy theory is one of the deepest insights of modern mathematics: the homotopy hypothesis asserts that $(\infty, 1)$-categories are equivalent to *homotopy types* (Kan complexes / $\infty$-groupoids). This is the foundation of Lurie's *Higher Topos Theory* and the work of Voevodsky, Riehl, Verity, and others on homotopy type theory.

### 11.7 The Godement Product

📐 We make explicit the calculation that defines horizontal composition of natural transformations.

**Proposition (Godement product).** Let $\alpha: F \Rightarrow G: \mathcal{C} \to \mathcal{D}$ and $\beta: H \Rightarrow K: \mathcal{D} \to \mathcal{E}$ be natural transformations. The *Godement product* (horizontal composite) $\beta * \alpha: HF \Rightarrow KG: \mathcal{C} \to \mathcal{E}$ has components at $A \in \mathcal{C}$ given by either of the two equal expressions:
$$(\beta * \alpha)_A \;=\; K(\alpha_A) \circ \beta_{FA} \;=\; \beta_{GA} \circ H(\alpha_A).$$

*Proof that both expressions agree.* Naturality of $\beta: H \Rightarrow K$ applied to the morphism $\alpha_A: FA \to GA$ in $\mathcal{D}$ gives the commutative square:
$$K(\alpha_A) \circ \beta_{FA} = \beta_{GA} \circ H(\alpha_A). \qquad \square$$

**Corollary (Interchange law in Cat).** For composable pairs of 2-cells $\alpha: F \Rightarrow G \Rightarrow H: \mathcal{C} \to \mathcal{D}$ and $\beta: K \Rightarrow L \Rightarrow M: \mathcal{D} \to \mathcal{E}$, denoting vertical composites as $\alpha' \circ \alpha$ and $\beta' \circ \beta$,
$$(\beta' \circ \beta) * (\alpha' \circ \alpha) = (\beta' * \alpha') \circ (\beta * \alpha).$$

🔑 **The interchange law is not an additional axiom in $\mathbf{Cat}$ — it is a consequence of the definition of the Godement product via naturality squares.** It becomes an axiom only when defining abstract 2-categories.

> [!NOTE] Riehl Exercise: 2-Categories
> **(i)** Verify the interchange law for $\mathbf{Cat}$: for composable 2-cells
> $$F \xRightarrow{\alpha} G \xRightarrow{\alpha'} H : \mathcal{C} \to \mathcal{D}, \qquad K \xRightarrow{\beta} L \xRightarrow{\beta'} M : \mathcal{D} \to \mathcal{E},$$
> show that $(\beta' \circ \beta) * (\alpha' \circ \alpha) = (\beta' * \alpha') \circ (\beta * \alpha)$ by computing both sides componentwise using the Godement product formula.
>
> **(ii)** Define a strict 2-functor $\Phi: \mathbf{Cat}^{\mathrm{op}} \to \mathbf{CAT}$. Show that the assignment $\mathcal{C} \mapsto [\mathcal{C}^{\mathrm{op}}, \mathbf{Set}]$ (taking the presheaf category) extends to such a 2-functor, specifying the action on functors and natural transformations explicitly.
>
> **(iii)** Show that an adjunction $F \dashv G: \mathcal{C} \rightleftharpoons \mathcal{D}$, when viewed in the 2-category $\mathbf{Cat}$, satisfies the 2-categorical triangle identities as stated in §11.5. Conversely, verify that the definition of adjunction in any 2-category (via unit and counit 2-cells) implies that the unit $\eta$ and counit $\varepsilon$ are unique given the other, in the sense that a left adjoint $F$ has at most one right adjoint $G$ up to unique isomorphism.

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
