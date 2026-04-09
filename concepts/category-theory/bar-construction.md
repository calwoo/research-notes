# The Bar Construction

## Table of Contents

- [[#1. Motivation: The Resolution Problem|1. Motivation: The Resolution Problem]]
  - [[#1.1 Canonical Resolutions and Functoriality|1.1 Canonical Resolutions and Functoriality]]
  - [[#1.2 The Key Idea: Resolve Using the Algebra Itself|1.2 The Key Idea: Resolve Using the Algebra Itself]]
  - [[#1.3 A Unified Perspective|1.3 A Unified Perspective]]
- [[#2. The Algebraic Bar Construction|2. The Algebraic Bar Construction]]
  - [[#2.1 Setup: Augmented Algebras|2.1 Setup: Augmented Algebras]]
  - [[#2.2 The One-Sided Bar Complex B(k, A, k)|2.2 The One-Sided Bar Complex B(k, A, k)]]
  - [[#2.3 The Two-Sided Bar Construction B(M, A, N)|2.3 The Two-Sided Bar Construction B(M, A, N)]]
  - [[#2.4 Acyclicity: The Contracting Homotopy|2.4 Acyclicity: The Contracting Homotopy]]
  - [[#2.5 Computing Tor via the Bar Resolution|2.5 Computing Tor via the Bar Resolution]]
- [[#3. The Simplicial Perspective|3. The Simplicial Perspective]]
  - [[#3.1 The Bar Construction as a Simplicial Object|3.1 The Bar Construction as a Simplicial Object]]
  - [[#3.2 Face and Degeneracy Maps|3.2 Face and Degeneracy Maps]]
  - [[#3.3 The Normalized Chain Complex and Dold-Kan|3.3 The Normalized Chain Complex and Dold-Kan]]
- [[#4. The Topological Bar Construction: Classifying Spaces|4. The Topological Bar Construction: Classifying Spaces]]
  - [[#4.1 The Nerve of a Group|4.1 The Nerve of a Group]]
  - [[#4.2 The Classifying Space BG|4.2 The Classifying Space BG]]
  - [[#4.3 The Universal Bundle EG via the Two-Sided Bar|4.3 The Universal Bundle EG via the Two-Sided Bar]]
  - [[#4.4 Homotopy Colimits as Bar Constructions|4.4 Homotopy Colimits as Bar Constructions]]
- [[#5. The Monadic Bar Construction|5. The Monadic Bar Construction]]
  - [[#5.1 Setup and Definition|5.1 Setup and Definition]]
  - [[#5.2 The Canonical Simplicial Resolution|5.2 The Canonical Simplicial Resolution]]
  - [[#5.3 Connection to Beck Monadicity|5.3 Connection to Beck Monadicity]]
- [[#6. The Bar-Cobar Adjunction and Koszul Duality|6. The Bar-Cobar Adjunction and Koszul Duality]]
  - [[#6.1 The Cobar Construction|6.1 The Cobar Construction]]
  - [[#6.2 The Adjunction B Dashv Omega|6.2 The Adjunction B Dashv Omega]]
  - [[#6.3 Koszul Duality|6.3 Koszul Duality]]
  - [[#6.4 Adams' Cobar Construction for Loop Spaces|6.4 Adams' Cobar Construction for Loop Spaces]]
- [[#7. Hochschild Homology and the Cyclic Bar Construction|7. Hochschild Homology and the Cyclic Bar Construction]]
  - [[#7.1 The Hochschild Complex as a Bar Complex|7.1 The Hochschild Complex as a Bar Complex]]
  - [[#7.2 The Cyclic Bar Construction|7.2 The Cyclic Bar Construction]]
  - [[#7.3 Cyclic Homology and the Circle Action|7.3 Cyclic Homology and the Circle Action]]
- [[#8. The Categorical and Infinity-Categorical Perspective|8. The Categorical and Infinity-Categorical Perspective]]
  - [[#8.1 Bar as Colimit: The Coequalizer Interpretation|8.1 Bar as Colimit: The Coequalizer Interpretation]]
  - [[#8.2 The Infinity-Categorical Bar Construction|8.2 The Infinity-Categorical Bar Construction]]
  - [[#8.3 Everything Is One Construction|8.3 Everything Is One Construction]]
- [[#9. The Bar Construction in Spectra: B as Categorical Suspension|9. The Bar Construction in Spectra: B as Categorical Suspension]]
  - [[#9.1 Augmented Ring Spectra and the Spectral Bar Construction|9.1 Augmented Ring Spectra and the Spectral Bar Construction]]
  - [[#9.2 B as Categorical Suspension, Not Spectrum Suspension|9.2 B as Categorical Suspension, Not Spectrum Suspension]]
  - [[#9.3 The Free Algebra Case: When BA Simeq Sigma M|9.3 The Free Algebra Case: When BA Simeq Sigma M]]
  - [[#9.4 Iterated Bar and En-Algebras|9.4 Iterated Bar and En-Algebras]]
- [[#References|References]]

---

## 1. Motivation: The Resolution Problem 🔍

### 1.1 Canonical Resolutions and Functoriality

Let $A$ be a ring (or algebra over a field $k$) and let $M$, $N$ be $A$-modules. One of the central problems of homological algebra is the computation of the *derived functors* $\mathrm{Tor}_*^A(M, N)$ and $\mathrm{Ext}^*_A(M, N)$. By definition, these require choosing a projective (or free) resolution of one of the modules. For example, to compute $\mathrm{Tor}_*^A(M,N)$, one finds a free resolution

$$\cdots \to F_2 \to F_1 \to F_0 \to M \to 0,$$

then applies $(-) \otimes_A N$ and takes homology.

The trouble is that free resolutions are not unique. Different choices of resolution give the same answer — but constructing the comparison chain map and verifying independence requires work. For abstract existence theorems this is fine, but for explicit computation and for *functoriality in $A$* it is not.

**The question:** Is there a *canonical*, functorial free resolution — one that requires no choices, depends only on $A$ and $M$, and varies naturally with maps of rings or modules?

### 1.2 The Key Idea: Resolve Using the Algebra Itself

The answer is yes, and the key observation is almost embarrassingly simple: **use the algebra $A$ to build its own resolution.** Rather than choosing generators for a free module, observe that $A$ itself is free of rank one over $A$. We can then build a resolution of $k$ (the ground field, viewed as an $A$-module via the augmentation $\varepsilon: A \to k$) using only tensor products of $A$ with itself:

$$\cdots \to A \otimes_k A \otimes_k A \to A \otimes_k A \to A \to k \to 0.$$

The differential encodes the multiplication of $A$: it alternately multiplies adjacent tensor factors or removes boundary factors. This is the *bar resolution*, and it is entirely determined by the algebra structure of $A$.

> [!INFO] Why "bar"?
> The notation $[a_1 | a_2 | \cdots | a_n]$ — elements of $A$ separated by vertical bars — was introduced by Eilenberg and Mac Lane in their 1950s study of the cohomology of Eilenberg–MacLane spaces $K(\pi, n)$. The bars are literally typographic separators between tensor factors, and the name stuck.

### 1.3 A Unified Perspective

What makes the bar construction remarkable is that this same idea — *resolve an object by freely generated objects built from the generating structure* — appears identically in at least five different mathematical contexts:

| Context | Input | Bar construction gives |
|---------|-------|----------------------|
| Homological algebra | Ring $A$, module $M$ | Free resolution $B(M,A,N) \to M \otimes_A N$ |
| Group homology | Group $G$ | Resolution $\mathbb{Z}[G^{\bullet+1}] \to \mathbb{Z}$ |
| Topology | Topological group $G$ | Classifying space $BG = \lvert NG \rvert$ |
| Category theory | Monad $T$ on $\mathcal{C}$ | Simplicial resolution $B_\bullet(T,X) \to X$ |
| Koszul duality | dg-algebra $A$ | dg-coalgebra $BA$ with bar differential |

The unifying principle — section [[#8.3 Everything Is One Construction|8.3]] — is that every bar construction is the geometric realization of a simplicial object arising from a monad.

---

## 2. The Algebraic Bar Construction 🔩

### 2.1 Setup: Augmented Algebras

**Definition (Augmented algebra).** An *augmented $k$-algebra* is a $k$-algebra $A$ equipped with a $k$-algebra homomorphism $\varepsilon: A \to k$ called the *augmentation*. The *augmentation ideal* is $\bar{A} = \ker(\varepsilon)$, so $A \cong k \oplus \bar{A}$ as $k$-modules.

The prototypical examples are group algebras $k[G]$ with $\varepsilon(g) = 1$ for all $g \in G$, and free algebras $T(V) = \bigoplus_{n \geq 0} V^{\otimes n}$ with $\varepsilon$ projecting to $V^{\otimes 0} = k$.

> [!NOTE] Why augmentation?
> The augmentation allows us to view $k$ as a left (or right) $A$-module via $a \cdot \lambda = \varepsilon(a)\lambda$. This is the module we will resolve. Without an augmentation, there is no canonical choice of "trivial" module to resolve.

### 2.2 The One-Sided Bar Complex B(k, A, k)

**Definition (Bar complex).** Let $A$ be an augmented $k$-algebra with augmentation ideal $\bar{A}$. The *bar complex* $B(A) = B(k, A, k)$ is the chain complex with:

- **Underlying graded module:** $B_n(A) = \bar{A}^{\otimes_k n}$ in degree $n \geq 0$.
- **Notation:** An element $a_1 \otimes \cdots \otimes a_n \in \bar{A}^{\otimes n}$ is written $[a_1 | a_2 | \cdots | a_n]$.
- **Differential:** $\partial: B_n(A) \to B_{n-1}(A)$ is given by

$$\partial[a_1 | a_2 | \cdots | a_n] = \sum_{i=1}^{n-1} (-1)^{i-1} [a_1 | \cdots | a_i a_{i+1} | \cdots | a_n].$$

The map $\partial$ simply multiplies together adjacent pairs, alternating signs. It is degree $-1$, and the standard telescoping argument gives $\partial^2 = 0$ (adjacent terms cancel in pairs).

> [!TIP]- Proof that the differential squares to zero
> Expand $\partial^2 [a_1 | \cdots | a_n]$. The term $(-1)^{i-1}(-1)^{j-1}$ from first applying $\partial$ at position $i$ then at position $j$ equals $(-1)^{i+j}$. For $j < i$, the second application acts before the first, giving sign $(-1)^{j-1}(-1)^{i-2} = (-1)^{i+j-1}$. These two signs are opposite, so every pair of terms $(i,j)$ and $(j, i)$ cancels. (Here $1 \le j < i \le n-1$, and after applying $\partial$ at position $j$ the elements shift, giving the $i-1$ instead of $i$.) This is the standard sign-bookkeeping for the alternating sum of face maps of a simplicial object — see section [[#3.2 Face and Degeneracy Maps|3.2]].

The complex $B(A)$ is augmented by the map $\varepsilon: B_0(A) = \bar{A} \hookrightarrow A \xrightarrow{\varepsilon} k$ (here we use the inclusion $\bar{A} \subset A$, then augment; equivalently one appends $k$ in degree $-1$).

### 2.3 The Two-Sided Bar Construction B(M, A, N)

The one-sided bar complex computes the homology of $k$ over $A$. For computing $\mathrm{Tor}_*^A(M, N)$ for general modules, we need the *two-sided bar construction*.

**Definition (Two-sided bar construction).** Let $M$ be a right $A$-module and $N$ a left $A$-module. The *two-sided bar complex* $B(M, A, N)$ is the chain complex with:

- **Degree $n$ piece:** $B_n(M, A, N) = M \otimes_k \bar{A}^{\otimes_k n} \otimes_k N$.
- **Notation:** An element $m \otimes [a_1 | \cdots | a_n] \otimes n'$ is written $m[a_1 | \cdots | a_n]n'$.
- **Differential:** $\partial: B_n(M,A,N) \to B_{n-1}(M,A,N)$ is

$$\partial(m[a_1|\cdots|a_n]n') = ma_1[a_2|\cdots|a_n]n' + \sum_{i=1}^{n-1}(-1)^i m[a_1|\cdots|a_i a_{i+1}|\cdots|a_n]n' + (-1)^n m[a_1|\cdots|a_{n-1}]a_n n'.$$

The first term absorbs $a_1$ into the right $A$-action on $M$; the last term absorbs $a_n$ into the left $A$-action on $N$; the middle terms multiply adjacent elements of $\bar{A}$.

> [!EXAMPLE] The first few differentials
> - $\partial(m[\,]n') = mn'$ (the augmentation: degree $0$ maps to $M \otimes_k N$, then to $M \otimes_A N$ by the universal property)
> - $\partial(m[a]n') = ma[\,]n' - m[\,]an'$ (in $M \otimes_k N$, this is $ma \otimes n' - m \otimes an'$, which maps to zero in $M \otimes_A N$ — this is precisely the relation defining $\otimes_A$!)
> - $\partial(m[a|b]n') = ma[b]n' - m[ab]n' + m[a]bn'$

**Key observation:** The complex $B(M,A,N)$ is not over $A$; it is a complex of $k$-modules, and its homology in degree $-1$ (after appending $M \otimes_A N$ as the augmentation target) is precisely $\mathrm{Tor}_*^A(M,N)$.

### 2.4 Acyclicity: The Contracting Homotopy

The bar complex $B(k,A,k)$ is an acyclic complex of $k$-modules — its terms $\bar{A}^{\otimes n}$ are free over $k$ but not over $A$. The actual free left $A$-module resolution of $k$ is $B(A,A,k)$, with terms $A \otimes_k \bar{A}^{\otimes n}$ (free over $A$ on basis $\bar{A}^{\otimes n}$); one recovers $B(k,A,k)$ from it by applying $k \otimes_A -$. The proof of acyclicity is explicit:

**Proposition (Acyclicity).** The augmented bar complex

$$\cdots \to B_2(k,A,k) \xrightarrow{\partial} B_1(k,A,k) \xrightarrow{\partial} B_0(k,A,k) \xrightarrow{\varepsilon} k \to 0$$

is exact.

*Proof.* We exhibit a *contracting homotopy* $s: B_n \to B_{n+1}$ defined by

$$s[a_1 | a_2 | \cdots | a_n] = [1 | a_1 | a_2 | \cdots | a_n]$$

(prepend the unit $1 \in A$ to the bar sequence; note $1 \notin \bar{A}$, so we must work in the unreduced bar complex where the $a_i$ range over all of $A$, not just $\bar{A}$). A direct computation gives $\partial s + s \partial = \mathrm{id}$, hence $H_n(B(A)) = 0$ for $n > 0$ and $H_0 \cong k$. $\square$

> [!INFO] Why this homotopy is not $A$-linear
> The map $s$ is only a $k$-linear contracting homotopy, not a map of $A$-modules. This is exactly what is needed: acyclicity of $B(k,A,k)$ as a complex of $k$-modules implies that $B(A,A,k)$ — the free $A$-module resolution obtained by tensoring each term on the left with $A$ — is a valid projective resolution, since its exactness follows from the same contracting homotopy applied termwise.

The same argument applies to the two-sided bar complex $B(M,A,N)$: the contracting homotopy prepends a unit to the bar sequence, and one obtains:

**Corollary.** $B_\bullet(M,A,N) \to M \otimes_A N \to 0$ is a resolution of $M \otimes_A N$ by free $k$-modules.

*This exercise makes the contracting homotopy explicit by carrying out the identity $\partial s + s\partial = \mathrm{id}$ on a concrete element, and identifies the two-sided generalization with the extra degeneracy used in the topological setting.*

> [!QUESTION] Exercise 1
> Let $A$ be an augmented $k$-algebra and define the contracting homotopy $s: B_n(k,A,k) \to B_{n+1}(k,A,k)$ by $s[a_1|\cdots|a_n] = [1|a_1|\cdots|a_n]$.
>
> (a) Verify $(\partial s + s\partial)[a_1|a_2] = [a_1|a_2]$ by expanding both terms explicitly and checking cancellation.
>
> (b) Show the formula $\partial s + s\partial = \mathrm{id}$ holds in general by tracking the sign on each term produced by $\partial s[a_1|\cdots|a_n]$ and $s\partial[a_1|\cdots|a_n]$, and identifying which terms cancel and which survive.
>
> (c) Generalize: for the two-sided complex $B(M,A,N)$, define $s(m[a_1|\cdots|a_n]n') = m[1|a_1|\cdots|a_n]n'$ and verify this is still a contracting homotopy. Which step in (b) required the first factor to be $A$ itself (not a general $M$)?
>
> **Prerequisites:** [[#2.3 The Two-Sided Bar Construction B(M, A, N)|§2.3]], [[#2.4 Acyclicity: The Contracting Homotopy|§2.4]]

> [!TIP]- Solution to Exercise 1
> **Key insight:** The extra degeneracy $s$ exists because $A$ acts on itself — prepending $1$ works precisely because $1$ is a unit. The term $d_0 s[a_1|\cdots|a_n] = [1 \cdot a_1|\cdots] = [a_1|\cdots]$ uses unitality and cancels with all other terms.
>
> **Sketch:** (a) $\partial s[a_1|a_2] = \partial[1|a_1|a_2] = [1 \cdot a_1|a_2] - [1|a_1 a_2] + [1|a_1] \cdot a_2$... but we are in $B(k,A,k)$ so the outer terms hit the trivial module: $\partial s[a_1|a_2] = [a_1|a_2] - [a_1 a_2] + [a_1]a_2$ where the last term uses the right $A$-action on $k$, i.e. $a_2$ acts by $\varepsilon(a_2)$. Meanwhile $s\partial[a_1|a_2] = s(-[a_1 a_2] + \varepsilon(a_2)[a_1]) = -[1|a_1 a_2] + \varepsilon(a_2)[1|a_1]$. The sum telescopes to $[a_1|a_2]$. (b) In general, $\partial s$ produces $[a_1|\cdots|a_n]$ from $d_0 s$ plus terms $d_i s$ for $i \geq 1$; each $d_i s$ for $i \geq 1$ is $s d_{i-1}$, which exactly cancels the $s d_{i-1}$ terms in $s\partial$. (c) The step that fails for general $M$: $d_0 s(m[a_1|\cdots]n') = m \cdot 1 \cdot [a_1|\cdots]n' = m[a_1|\cdots]n'$ only because $m \cdot 1 = m$ requires $m \in A$ (or at least that the left $A$-action on $M$ satisfies $m \cdot 1_A = m$, which is the unit axiom for a module). So the argument goes through for $B(A,A,N)$ but not for $B(M,A,N)$ with $M \neq A$ unless $M$ has the unit property. The homotopy $s(m[a_1|\cdots]n') = m[1|\cdots]n'$ prepends $1$ into the $A$-strand, not before $m$, so the identity $d_0 s = \mathrm{id}$ actually holds for any $M$ — but $s$ is not $A$-linear. The key point is that $B(A,A,N)$ is contractible for *any* $N$, which is what makes $EA = |B(A,A,*)|$ contractible.

### 2.5 Computing Tor via the Bar Resolution

Since $B(A,A,k)$ is a free left $A$-module resolution of $k$, applying $M \otimes_A -$ gives a complex computing $\mathrm{Tor}^A_*(M, k)$. The two-sided complex $B(M,A,N)$ generalizes this: $B(M,A,A)$ is a free right $A$-module resolution of $M$ (terms $M \otimes_k \bar{A}^{\otimes n} \otimes_k A$, free over $A$), and

$$\mathrm{Tor}_n^A(M, N) \cong H_n(B(M,A,A) \otimes_A N) = H_n(B(M,A,N)).$$

Note also that $k \otimes_A k \cong k$ (since $A$ acts on $k$ via $\varepsilon$, the tensor relation becomes trivial), so $B(k,A,k)$ resolves $k \otimes_A k = k$ — consistent with the general formula for $M = N = k$. This is the *canonical* computation of Tor: no choices of resolution were made.

> [!WARNING] Efficiency vs. canonicity
> The bar resolution is almost never the most efficient resolution to work with. For a polynomial algebra $A = k[x]$, the bar resolution is an infinite complex, while $0 \to A \xrightarrow{x} A \to k \to 0$ is a two-term free resolution. The bar construction's value is not computational efficiency — it is functoriality, canonicity, and its role as a universal object.

*This exercise computes $\mathrm{Tor}$ via the bar resolution in the simplest nontrivial case and reveals the periodic structure that arises from the exterior algebra.*

> [!QUESTION] Exercise 2
> Let $k$ be a field and $A = k[x]/(x^2)$ (the exterior algebra on one generator). Write $\bar{A} = kx$ (one-dimensional, spanned by $x$).
>
> (a) Write out the bar complex $B_0, B_1, B_2, B_3$ as $k$-vector spaces and give the differentials $\partial: B_n \to B_{n-1}$ explicitly in terms of the basis elements $[x|\cdots|x]$.
>
> (b) Show that the complex is the periodic complex $\cdots \xrightarrow{0} k \xrightarrow{0} k \xrightarrow{0} k \to 0$. (Why are all differentials zero?)
>
> (c) Conclude $\mathrm{Tor}_n^A(k,k) \cong k$ for all $n \geq 0$, and compare with the Hilbert syzygy theorem (which says polynomial algebras have finite projective dimension).
>
> **Prerequisites:** [[#2.3 The Two-Sided Bar Construction B(M, A, N)|§2.3]], [[#2.5 Computing Tor via the Bar Resolution|§2.5]]

> [!TIP]- Solution to Exercise 2
> **Key insight:** $\bar{A} = kx$ is one-dimensional, so $B_n = \bar{A}^{\otimes n} = k[x|\cdots|x]$ is one-dimensional for each $n$. Every differential is $\partial[x|\cdots|x] = \sum (-1)^i [x|\cdots|x^2|\cdots|x]$, but $x^2 = 0$ in $A$, so every term vanishes.
>
> **Sketch:** (a) $B_0 = k$, $B_1 = kx$, $B_2 = k[x|x]$, $B_3 = k[x|x|x]$, all one-dimensional. $\partial[x] = 0$ since $x \cdot 1_k = \varepsilon(x) \cdot 1_k = 0$ (as $x \in \bar A$) and $1_k \cdot x = 0$ similarly. More generally $\partial[x|\cdots|x] = \sum (-1)^i [x|\cdots|x^2|\cdots|x] = 0$. (b) All differentials are zero because every internal multiplication produces $x^2 = 0$ and every boundary term hits the augmentation (which sends $x \mapsto 0$). (c) The homology is $k$ in every degree. Compare: $k[x]$ (no nilpotents) has the Koszul resolution $0 \to k[x] \xrightarrow{x} k[x] \to k \to 0$ with $\mathrm{Tor}_n^{k[x]}(k,k) = 0$ for $n > 1$. The nilpotence $x^2 = 0$ kills the differential and forces infinite Tor, reflecting the infinite projective dimension of $k$ over $A$.

---

## 3. The Simplicial Perspective 🔺

### 3.1 The Bar Construction as a Simplicial Object

The bar complex $B(M,A,N)$ is not just a chain complex — it is naturally a *simplicial* $k$-module. Understanding this simplicial structure is key to seeing why the bar construction generalizes so broadly.

Recall that a *simplicial object* in a category $\mathcal{C}$ is a functor $X_\bullet: \Delta^{\mathrm{op}} \to \mathcal{C}$, where $\Delta$ is the simplex category with objects $[n] = \{0 < 1 < \cdots < n\}$ and morphisms the order-preserving maps.

**Definition (Simplicial bar construction).** The simplicial $k$-module $B_\bullet(M,A,N)$ has:
- **Degree $n$ term:** $B_n(M,A,N) = M \otimes_k A^{\otimes_k n} \otimes_k N$

(note: here we use all of $A$, not just $\bar{A}$ — the augmentation ideal arises from the normalized chain complex).

### 3.2 Face and Degeneracy Maps

The structure maps are:

**Face maps** $d_i: B_n \to B_{n-1}$ for $0 \leq i \leq n$:

$$d_i(m \otimes a_1 \otimes \cdots \otimes a_n \otimes n') = \begin{cases} ma_1 \otimes a_2 \otimes \cdots \otimes a_n \otimes n' & i = 0 \\ m \otimes a_1 \otimes \cdots \otimes a_i a_{i+1} \otimes \cdots \otimes a_n \otimes n' & 0 < i < n \\ m \otimes a_1 \otimes \cdots \otimes a_{n-1} \otimes a_n n' & i = n \end{cases}$$

**Degeneracy maps** $s_j: B_n \to B_{n+1}$ for $0 \leq j \leq n$:

$$s_j(m \otimes a_1 \otimes \cdots \otimes a_n \otimes n') = m \otimes a_1 \otimes \cdots \otimes a_j \otimes 1_A \otimes a_{j+1} \otimes \cdots \otimes a_n \otimes n'$$

The degeneracy maps insert a unit $1_A \in A$ at position $j+1$.

> [!NOTE] Simplicial identities
> These maps satisfy the simplicial identities $d_i d_j = d_{j-1} d_i$ for $i < j$, and the analogous identities for $s_j$ and mixed $d_i s_j$. These identities encode the functoriality of $[n] \mapsto B_n$ with respect to order-preserving maps.

*This exercise verifies the simplicial identities directly from the definitions and identifies which algebraic axiom (associativity vs. unitality) each identity uses.*

> [!QUESTION] Exercise 3
> Work with $B_\bullet(M, A, N)$ as defined in [[#3.2 Face and Degeneracy Maps|§3.2]].
>
> (a) Verify the identity $d_1 d_2 = d_1 d_1$ on $B_3(M,A,N)$ (i.e., $d_i d_j = d_{j-1} d_i$ for $i=1, j=2$) by computing both sides on a general element $m \otimes a_1 \otimes a_2 \otimes a_3 \otimes n'$.
>
> (b) Verify $d_0 s_0 = \mathrm{id}$ (i.e., $d_i s_j = \mathrm{id}$ for $i = j$) on $B_1(M,A,N)$.
>
> (c) Verify $d_0 s_1 = s_0 d_0$ (i.e., $d_i s_j = s_{j-1} d_i$ for $i < j$) on $B_1(M,A,N)$.
>
> (d) For each identity in (a)–(c), state which axiom of an $A$-module the verification reduces to (associativity of multiplication in $A$, unitality $1_A \cdot a = a$, or associativity of the module action).
>
> **Prerequisites:** [[#3.2 Face and Degeneracy Maps|§3.2]]

> [!TIP]- Solution to Exercise 3
> **Key insight:** All simplicial identities reduce to either associativity $(a_i a_{i+1})a_{i+2} = a_i(a_{i+1} a_{i+2})$ or unitality $1 \cdot a = a = a \cdot 1$ — the coefficient objects $M$ and $N$ are spectators except at the boundary faces $d_0$ (which uses the $M$-action) and $d_n$ (which uses the $N$-action).
>
> **Sketch:** (a) $d_1 d_2(m \otimes a_1 \otimes a_2 \otimes a_3 \otimes n') = d_1(m \otimes a_1 \otimes a_2 a_3 \otimes n') = m \otimes a_1(a_2 a_3) \otimes n'$. And $d_1 d_1(m \otimes a_1 \otimes a_2 \otimes a_3 \otimes n') = d_1(m \otimes a_1 a_2 \otimes a_3 \otimes n') = m \otimes (a_1 a_2)a_3 \otimes n'$. These are equal by associativity in $A$. (b) $d_0 s_0(m \otimes a_1 \otimes n') = d_0(m \otimes 1 \otimes a_1 \otimes n') = m \cdot 1 \otimes a_1 \otimes n' = m \otimes a_1 \otimes n'$, using $m \cdot 1_A = m$ (right unit axiom for $M$). (c) $d_0 s_1(m \otimes a_1 \otimes n') = d_0(m \otimes a_1 \otimes 1 \otimes n') = ma_1 \otimes 1 \otimes n'$. And $s_0 d_0(m \otimes a_1 \otimes n') = s_0(ma_1 \otimes n') = ma_1 \otimes 1 \otimes n'$. Equal. (d): (a) uses associativity in $A$; (b) uses the right unit axiom for $M$; (c) uses neither — it is a tautological commutation of inserting $1$ at position $1$ vs. applying $d_0$ then inserting at position $0$.

### 3.3 The Normalized Chain Complex and Dold-Kan

The relationship between the simplicial bar construction and the chain complex bar construction is given by the *Dold-Kan correspondence*.

**Theorem (Dold-Kan, informal).** There is an equivalence of categories between simplicial abelian groups and non-negatively graded chain complexes of abelian groups. Under this equivalence, a simplicial module $X_\bullet$ corresponds to its *normalized chain complex* $N(X)_n = X_n / (\text{images of all degeneracy maps})$, with differential $\sum_{i=0}^{n} (-1)^i d_i$.

For the simplicial bar construction: the normalized chain complex of $B_\bullet(M,A,N)$ is precisely the two-sided bar complex $B(M,A,N)$ defined in section [[#2.3 The Two-Sided Bar Construction B(M, A, N)|2.3]]. (The images of the degeneracy maps — which insert units — exactly account for the difference between using $A$ and using $\bar{A}$ in each tensor factor.)

> [!INFO] The nerve interpretation
> The simplicial set $B_\bullet(*, G, *)$ for a discrete group $G$ (with $*$ the trivial module / single point) is the *nerve* $NG$ of the category $\mathbf{B}G$ (the one-object groupoid with morphisms $G$). This is the bridge to the topological bar construction — see section [[#4. The Topological Bar Construction: Classifying Spaces|4]].

---

## 4. The Topological Bar Construction: Classifying Spaces 🌐

### 4.1 The Nerve of a Group

Let $G$ be a topological group. The same simplicial formulas define a *simplicial topological space* $NG$ (the *nerve* of $G$, viewed as a one-object groupoid):

- $NG_n = G^n$ (the $n$-fold product)
- Face maps: $d_0(g_1, \ldots, g_n) = (g_2, \ldots, g_n)$; $d_i(g_1, \ldots, g_n) = (g_1, \ldots, g_i g_{i+1}, \ldots, g_n)$ for $0 < i < n$; $d_n(g_1, \ldots, g_n) = (g_1, \ldots, g_{n-1})$
- Degeneracy maps: $s_j(g_1, \ldots, g_n) = (g_1, \ldots, g_j, 1, g_{j+1}, \ldots, g_n)$

These are exactly the same algebraic formulas as the bar construction with $A = k[G]$, $M = N = k$, now interpreted topologically.

### 4.2 The Classifying Space BG

**Definition (Classifying space).** The *classifying space* of $G$ is

$$BG = |NG| = B(*, G, *)$$

the geometric realization of the nerve $NG$. Here $*$ denotes the one-point space (the trivial $G$-space).

*Geometric realization* replaces each $G^n$ with a copy of the standard topological $n$-simplex $\Delta^n$, glued together along the face and degeneracy maps:

$$|NG| = \left(\bigsqcup_{n \geq 0} G^n \times \Delta^n\right) \bigg/ \sim$$

where $(d_i(g), t) \sim (g, \delta_i(t))$ and $(s_j(g), t) \sim (g, \sigma_j(t))$, with $\delta_i, \sigma_j$ the standard face and degeneracy maps on simplices.

**Key properties:**
- When $G$ is a discrete group, $\pi_1(BG) \cong G$ and $\pi_k(BG) = 0$ for $k > 1$ — so $BG = K(G,1)$ is the Eilenberg-MacLane space.
- $H_*(BG; \mathbb{Z}) \cong H_*(G; \mathbb{Z})$ — the singular homology of $BG$ recovers group homology.
- A principal $G$-bundle $P \to X$ is classified by a map $X \to BG$; homotopy classes of such maps are in bijection with isomorphism classes of principal $G$-bundles.

> [!INFO] Historical note
> Milnor (1956) first constructed $BG$ via the infinite join construction $G * G * G * \cdots$, which is homeomorphic to $|NG|$ but less transparent. The simplicial/bar construction perspective was developed by Segal (1968), who showed $BG = |NG|$ and used it to construct spectra via $\Gamma$-spaces — see [[papers/matryoshka-representation-learning|Segal (1974)]] and [[concepts/category-theory/segal-categories-cohomology|the Segal note]].

*This exercise makes the bar construction completely concrete for the two most fundamental groups, connecting the simplicial machinery to familiar classifying spaces.*

> [!QUESTION] Exercise 4
> (a) **$G = \mathbb{Z}$:** Write down $B_0, B_1, B_2$ of $B(*,\mathbb{Z},*)$ as sets with all face maps. Show the 1-skeleton is a circle, the 2-cells fill in all composites $[m][n] = [m+n]$, and conclude $B\mathbb{Z} \simeq S^1 = K(\mathbb{Z},1)$.
>
> (b) **$G = \mathbb{Z}/2$:** Write down $B_0, B_1, B_2$ with face maps. Identify the unique nondegenerate cell in each degree and describe the attaching maps. Conclude that the $n$-skeleton of $B(\mathbb{Z}/2)$ is $\mathbb{RP}^n$, hence $B(\mathbb{Z}/2) \simeq \mathbb{RP}^\infty = K(\mathbb{Z}/2, 1)$.
>
> (c) How many nondegenerate $k$-cells does $B(\mathbb{Z}/n)$ have in each degree? (Watch for the off-by-one: the answer is not $n^k$.)
>
> **Prerequisites:** [[#4.2 The Classifying Space BG|§4.2]], [[#3.2 Face and Degeneracy Maps|§3.2]]

> [!TIP]- Solution to Exercise 4
> **Key insight:** A simplex $(g_1, \ldots, g_k) \in G^k$ is *nondegenerate* iff no $g_i = e$ — inserting any unit gives a degenerate simplex. So the count of nondegenerate $k$-cells is $|G \setminus \{e\}|^k$.
>
> **Sketch:** (a) $B_0 = \{*\}$, $B_1 = \mathbb{Z}$ (one 1-cell per integer), $B_2 = \mathbb{Z}^2$ with $d_0(m,n) = n$, $d_1(m,n) = m+n$, $d_2(m,n) = m$. Each 2-cell $(m,n)$ is a triangle with edges $[m]$, $[n]$, and $[m+n]$, gluing the path $[m][n]$ to $[m+n]$. After realization, all 1-cells become identified (since $[n] = [1]^n$ in $\pi_1$), and the 2-cells force $\pi_1 = \mathbb{Z}$ and kill higher $\pi_k$ inductively. Result: $S^1$. (b) $B_0 = \{*\}$, $B_1 = \{0,1\}$ — one nondegenerate 1-cell $[1]$. $B_2 = \{(0,0),(0,1),(1,0),(1,1)\}$; nondegenerate cells: $(1,1)$ only (both entries $\neq 0$). Face maps: $d_0(1,1) = 1$, $d_1(1,1) = 1+1 = 0 = *$, $d_2(1,1) = 1$. So the 2-cell attaches along $[1][1][1]^{-1}$, i.e. forces $[1]^2 = *$. This builds $\mathbb{RP}^2$; repeating gives $\mathbb{RP}^\infty$. (c) Nondegenerate $k$-cells: $(|G|-1)^k = (n-1)^k$.

*This exercise shows the bar construction is strictly compatible with products, giving a clean proof that $K(G \times H, 1) \simeq K(G,1) \times K(H,1)$.*

> [!QUESTION] Exercise 5
> For discrete groups $G$ and $H$, show there is a natural homeomorphism $B(G \times H) \cong BG \times BH$ by:
>
> (a) Identifying the simplicial space $B_\bullet(G \times H)$ with $B_\bullet(G) \times B_\bullet(H)$ (componentwise face and degeneracy maps).
>
> (b) Citing the fact that geometric realization commutes with finite products for "good" simplicial spaces (those whose degeneracy maps are cofibrations) to conclude $|B_\bullet(G \times H)| \cong |B_\bullet(G)| \times |B_\bullet(H)|$.
>
> (c) Deduce $K(G \times H, 1) \simeq K(G,1) \times K(H,1)$ for any two discrete groups.
>
> **Prerequisites:** [[#4.2 The Classifying Space BG|§4.2]]

> [!TIP]- Solution to Exercise 5
> **Key insight:** The bar construction is functorial in $G$ and sends products of groups to products of simplicial sets; the content is entirely in the geometric realization step, which requires the "goodness" condition to avoid point-set pitfalls.
>
> **Sketch:** (a) $(G \times H)^n \cong G^n \times H^n$ and face maps $d_i^{G \times H} = (d_i^G, d_i^H)$ — this is just the product of simplicial sets. (b) The simplicial spaces $B_\bullet(G)$ and $B_\bullet(H)$ are "good" since all degeneracy maps $s_j: G^n \to G^{n+1}$ are closed cofibrations (being homeomorphisms onto closed subsets). The Milnor–Geometric Realization theorem then gives $|X_\bullet \times Y_\bullet| \cong |X_\bullet| \times |Y_\bullet|$. (c) Immediate: $B(G \times H) \simeq BG \times BH$, and $\pi_1(BG \times BH) \cong G \times H$, $\pi_k = 0$ for $k > 1$.

*This exercise connects the combinatorics of the bar construction to classical topology via the Euler characteristic, and catches a common off-by-one error in cell counting.*

> [!QUESTION] Exercise 6
> For $G = \mathbb{Z}/n$, let $\mathrm{sk}_k B(\mathbb{Z}/n)$ denote the $k$-skeleton of $B(\mathbb{Z}/n)$.
>
> (a) Using the result of Exercise 4(c), write a formula for the number of nondegenerate $j$-cells for each $0 \leq j \leq k$.
>
> (b) Compute the Euler characteristic $\chi(\mathrm{sk}_k B(\mathbb{Z}/n)) = \sum_{j=0}^k (-1)^j c_j$ where $c_j$ is the number of nondegenerate $j$-cells.
>
> (c) For $n = 2$, verify the formula matches $\chi(\mathbb{RP}^k)$ for $k = 1, 2, 3, 4$.
>
> **Prerequisites:** Exercise 4, [[#4.2 The Classifying Space BG|§4.2]]

> [!TIP]- Solution to Exercise 6
> **Key insight:** The nondegenerate cell count is $(n-1)^j$ (not $n^j$), so the Euler characteristic is a geometric series in $-(n-1)$.
>
> **Sketch:** (a) $c_0 = 1$ (the unique basepoint), $c_j = (n-1)^j$ for $j \geq 1$. (b) $\chi(\mathrm{sk}_k) = 1 + \sum_{j=1}^k (-1)^j (n-1)^j = 1 - (n-1)\frac{1-(-1)^{k+1}(n-1)^k}{n}$. Simplifies to $\frac{1 + (-1)^k (n-1)^{k+1}}{n}$ after algebra. (c) For $n = 2$: $c_j = 1$ for all $j$. $\chi(\mathrm{sk}_k) = \sum_{j=0}^k (-1)^j = \begin{cases} 1 & k \text{ even} \\ 0 & k \text{ odd}\end{cases}$, which matches $\chi(\mathbb{RP}^k) = \frac{1+(-1)^k}{2}$. ✓

### 4.3 The Universal Bundle EG via the Two-Sided Bar

The *universal principal $G$-bundle* $EG \to BG$ also arises from the bar construction. Using $G$ itself as a left $G$-space (via left multiplication):

$$EG = B(G, G, *) = |[n \mapsto G^{n+1}]|$$

The face maps for $EG$ are: $d_0(g_0, g_1, \ldots, g_n) = (g_0 g_1^{-1} \cdot \text{something})$... more precisely, $EG$ is the realization of the simplicial space with $n$-simplices $G^{n+1}$ and face map $d_i$ multiplying the $i$-th and $(i+1)$-th entries.

The bundle map $EG \to BG$ is induced by the simplicial map $G^{n+1} \to G^n$ that forgets the first factor and records the ratios $g_0^{-1}g_1, g_1^{-1}g_2, \ldots$

**Theorem.** $EG$ is contractible and $G$ acts freely on $EG$ with quotient $EG/G \cong BG$.

*Proof sketch.* The contracting homotopy $s([g_0 | g_1 | \cdots | g_n]) = [e | g_0 | \cdots | g_n]$ (prepend the identity, exactly as in the algebraic case) shows $EG$ is contractible. The free $G$-action is by $(g \cdot (g_0, g_1, \ldots, g_n)) = (gg_0, gg_1, \ldots, gg_n)$. $\square$

More generally, the two-sided bar construction $B(Y, G, X)$ for $G$-spaces $Y$ (right) and $X$ (left) gives the *homotopy quotient* (Borel construction):

$$B(Y, G, X) \simeq Y \times_G EG \times_G X.$$

*This exercise verifies the contractibility of $EG$ and freeness of the $G$-action from the simplicial structure, giving a complete proof that $EG \to BG$ is a principal $G$-bundle.*

> [!QUESTION] Exercise 7
> Let $G$ be a discrete group and $EG = B(G,G,*) = |[n \mapsto G^{n+1}]|$.
>
> (a) Define the extra degeneracy $s_{-1}: G^{n+1} \to G^{n+2}$ by $s_{-1}(g_0, g_1, \ldots, g_n) = (e, g_0, g_1, \ldots, g_n)$. Verify $d_0 \circ s_{-1} = \mathrm{id}$ and $d_{i+1} \circ s_{-1} = s_{-1} \circ d_i$ for all $i \geq 0$. Conclude $EG \simeq *$.
>
> (b) The left $G$-action on $EG$ is $g \cdot (g_0, g_1, \ldots, g_n) = (gg_0, gg_1, \ldots, gg_n)$. Show this action is free: if $g \cdot (g_0, \ldots, g_n) = (g_0, \ldots, g_n)$ then $g = e$.
>
> (c) Show the simplicial map $G^{n+1} \to G^n$ given by $(g_0, g_1, \ldots, g_n) \mapsto (g_0^{-1}g_1, g_1^{-1}g_2, \ldots, g_{n-1}^{-1}g_n)$ is $G$-equivariant (for the $G$-action on $G^n$ by left multiplication on the first factor only) and induces $EG \to BG$ on geometric realizations.
>
> **Prerequisites:** [[#4.3 The Universal Bundle EG via the Two-Sided Bar|§4.3]], Exercise 1

> [!TIP]- Solution to Exercise 7
> **Key insight:** The extra degeneracy works because $G$ acts on itself — prepending $e$ is the unit. The freeness uses that $G$ acts on $G^{n+1}$ by changing every coordinate simultaneously, so a fixed point requires $g = e$.
>
> **Sketch:** (a) $d_0 s_{-1}(g_0,\ldots,g_n) = d_0(e,g_0,\ldots,g_n) = (e \cdot g_0, g_1, \ldots, g_n) = (g_0, g_1,\ldots,g_n)$ ✓. $d_{i+1} s_{-1}(g_0,\ldots,g_n) = d_{i+1}(e,g_0,\ldots,g_n)$ multiplies entries at positions $i+1$ and $i+2$ (0-indexed), which are $g_i$ and $g_{i+1}$, giving $(e,g_0,\ldots,g_ig_{i+1},\ldots,g_n) = s_{-1} d_i(g_0,\ldots,g_n)$ ✓. The extra degeneracy gives a simplicial null-homotopy $\mathrm{id} \simeq c_*$ (constant map), so $|B(G,G,*)| \simeq *$. (b) $g \cdot (g_0,\ldots,g_n) = (gg_0,\ldots,gg_n) = (g_0,\ldots,g_n)$ iff $gg_i = g_i$ for all $i$, iff $g = e$. (c) Under $(g_0,\ldots,g_n) \mapsto (g_0^{-1}g_1,\ldots)$: left $G$-action sends $g_i \mapsto gg_i$, so $g_i^{-1}g_{i+1} \mapsto (gg_i)^{-1}(gg_{i+1}) = g_i^{-1}g_{i+1}$ — the map is $G$-invariant. It is compatible with face maps: $d_i$ on $G^{n+1}$ corresponds to $d_i$ on $G^n$ via this change of coordinates. The quotient $EG/G = B(*,G,*) = BG$.

*This exercise deduces $\Omega BG \simeq G$ from the fiber sequence, providing the key structural result that explains why the bar construction deloops topological groups.*

> [!QUESTION] Exercise 8
> Let $G$ be a discrete group. Use the fiber sequence $G \to EG \to BG$ from §4.3.
>
> (a) Write out the long exact sequence of homotopy groups for the fibration $G \to EG \to BG$.
>
> (b) Use contractibility of $EG$ (Exercise 7(a)) to show $\pi_n(BG) \cong \pi_{n-1}(G)$ for all $n \geq 1$.
>
> (c) Conclude $BG = K(G,1)$ when $G$ is discrete. What goes wrong if $G$ is not discrete — say $G = S^1$? What does $BS^1$ look like?
>
> **Prerequisites:** [[#4.3 The Universal Bundle EG via the Two-Sided Bar|§4.3]], Exercise 7

> [!TIP]- Solution to Exercise 8
> **Key insight:** Contractibility of $EG$ makes it the "path space" of $BG$, forcing $\Omega BG \simeq G$ exactly as contractibility of $PX$ forces $\Omega X \simeq \Omega X$ in the based path-loop fibration.
>
> **Sketch:** (a) $\cdots \to \pi_n(G) \to \pi_n(EG) \to \pi_n(BG) \to \pi_{n-1}(G) \to \pi_{n-1}(EG) \to \cdots$. (b) Since $EG \simeq *$: $\pi_n(EG) = 0$ for all $n$, so the LES gives $0 \to \pi_n(BG) \xrightarrow{\cong} \pi_{n-1}(G) \to 0$. (c) For $G$ discrete: $\pi_0(G) = G$, $\pi_k(G) = 0$ for $k \geq 1$. So $\pi_1(BG) = G$ and $\pi_k(BG) = 0$ for $k \neq 1$: this is $K(G,1)$. For $G = S^1$: $\pi_1(S^1) = \mathbb{Z}$, $\pi_k(S^1) = 0$ for $k > 1$, so $BS^1$ has $\pi_2(BS^1) = \mathbb{Z}$ and all other $\pi_k = 0$: $BS^1 = K(\mathbb{Z},2) = \mathbb{CP}^\infty$.

### 4.4 Homotopy Colimits as Bar Constructions

The bar construction also computes *homotopy colimits* of diagrams. Let $\mathcal{D}$ be a small category and $F: \mathcal{D} \to \mathbf{Top}$ a diagram. The *homotopy colimit* of $F$ is

$$\mathrm{hocolim}_{\mathcal{D}}\, F = B(*, \mathcal{D}, F) = \left\lvert [n \mapsto \bigsqcup_{d_0 \to d_1 \to \cdots \to d_n} F(d_0)] \right\rvert$$

where the coproduct runs over all composable chains of $n$ morphisms in $\mathcal{D}$.

- Face map $d_0$ applies $F$ to the first morphism $d_0 \to d_1$ (functoriality).
- Face maps $d_i$ for $0 < i < n$ compose adjacent morphisms $d_i \to d_{i+1}$.
- Face map $d_n$ forgets the last object.
- Degeneracy maps insert identity morphisms.

This perspective (Malkiewich's notes on homotopy colimits) shows that $\mathrm{hocolim}$ is a systematic version of the bar construction with $G$ replaced by the category $\mathcal{D}$ and a point replaced by the functor $F$.

> [!EXAMPLE] Hocolim as homotopy pushout
> For $\mathcal{D} = \bullet \leftarrow \bullet \rightarrow \bullet$ (the span category) and $F = (A \leftarrow C \rightarrow B)$, the bar construction $B(*, \mathcal{D}, F)$ recovers the homotopy pushout $A \sqcup^h_C B = (A \sqcup (C \times [0,1]) \sqcup B)/(c,0) \sim f(c), (c,1) \sim g(c)$.

*This exercise identifies $B(*,G,X)$ as the Borel construction $EG \times_G X$, recovering equivariant homotopy theory as a special case of the bar machinery.*

> [!QUESTION] Exercise 9
> Let $G$ be a discrete group acting on a space $X$ from the left. Define the *twisted bar construction* with $B_n(*,G,X) = G^n \times X$ and face maps $d_0(g_1,\ldots,g_n,x) = (g_2,\ldots,g_n, g_1 \cdot x)$, $d_i(g_1,\ldots,g_n,x) = (g_1,\ldots,g_ig_{i+1},\ldots,g_n,x)$ for $0 < i < n$, and $d_n(g_1,\ldots,g_n,x) = (g_1,\ldots,g_{n-1},x)$.
>
> (a) Check the simplicial identity $d_0 d_1 = d_0 d_0$ on $B_2(*,G,X)$. Which axiom does it use?
>
> (b) Show $|B(*,G,X)| \cong EG \times_G X$ by identifying the simplicial map $B_n(G,G,X) \to B_n(*,G,X)$ (forgetting the leading $G$-factor) with the quotient by the free $G$-action of Exercise 7(b).
>
> (c) Take $G = \mathbb{Z}/2$ and $X = S^0 = \{+1,-1\}$ with the sign action $\tau \cdot x = -x$. Write out $B_0, B_1, B_2$ explicitly. Identify $|B(*,\mathbb{Z}/2, S^0)|$ as the mapping telescope of a degree-2 map, and conclude it is $\mathbb{RP}^\infty$ up to homotopy.
>
> **Prerequisites:** [[#4.3 The Universal Bundle EG via the Two-Sided Bar|§4.3]], [[#4.4 Homotopy Colimits as Bar Constructions|§4.4]], Exercises 7–8

> [!TIP]- Solution to Exercise 9
> **Key insight:** The twisted face map $d_0$ uses the $G$-action on $X$ rather than multiplication in $G$; this is the only place where the coefficient space $X$ plays a role, exactly as $d_0$ in $B(M,A,N)$ uses the $M$-action.
>
> **Sketch:** (a) $d_0 d_1(g_1,g_2,x) = d_0(g_1g_2,x) = g_1g_2 \cdot x$. $d_0 d_0(g_1,g_2,x) = d_0(g_2, g_1 \cdot x) = g_2 \cdot (g_1 \cdot x)$. These are equal iff $(g_1 g_2) \cdot x = g_1 \cdot (g_2 \cdot x)$... wait, this should be $d_0 d_1 = d_0 d_0$ for $i=0, j=1$ in $d_i d_j = d_{j-1} d_i$: $d_0 d_1 = d_0 d_0$. LHS: $d_0(g_1g_2, x) = g_1 g_2 \cdot x$. RHS: $d_0(g_2, g_1 x) = g_2 \cdot (g_1 \cdot x)$. *Not equal in general* — the correct identity is $d_0 d_1 = d_0 d_0$ which requires $(g_1 g_2)\cdot x = g_2(g_1 x)$... hmm. Actually the correct simplicial identity here is $d_i d_j = d_{j-1} d_i$ for $i < j$, so for $i=0, j=1$: $d_0 d_1 = d_0 d_0$. Let us recheck. $d_0 d_1(g_1,g_2,x) = d_0(g_1g_2,x) = (g_1g_2)\cdot x$. $d_0 d_0(g_1,g_2,x) = d_0(g_2, g_1 x) = g_2 \cdot (g_1 \cdot x)$. For the $G$-action to be a left action we need $(g_2 g_1)\cdot x = g_2 \cdot(g_1 \cdot x)$ — but $d_1(g_1,g_2,x) = (g_1g_2, x)$ puts the product in the opposite order. This reflects that $d_0$ on the twisted bar uses the *right* coset convention. *The simplicial identity does hold* because it uses associativity of the left $G$-action: $(g_1 g_2) x = g_1(g_2 x)$. (b) The map $B_n(G,G,X) \to B_n(*,G,X)$ is $(g_0, g_1,\ldots,g_n,x) \mapsto (g_0^{-1}g_1, \ldots, g_{n-1}^{-1}g_n, g_n^{-1}\cdot x)$ — the $G$-equivariant quotient. On realizations: $|B(G,G,X)|/G = |B(*,G,X)|$, and $|B(G,G,X)| = EG \times X$ (free $G$-action componentwise by Ex 7), so the quotient is $EG \times_G X$. (c) $B_0 = S^0 = \{+1,-1\}$; $B_1 = \{0,1\} \times S^0$ (4 elements: $(0,+1),(0,-1),(1,+1),(1,-1)$); $B_2 = (\mathbb{Z}/2)^2 \times S^0$ (8 elements). The realization identifies $(\mathbb{Z}/2)^n \times S^0$ cells: the two points of $S^0$ are connected by 1-cells via the $\tau$-action, and the pattern builds $\mathbb{RP}^\infty$ — the Borel construction $E(\mathbb{Z}/2) \times_{\mathbb{Z}/2} S^0 \simeq \mathbb{RP}^\infty$ since $S^0 \to \mathbb{RP}^\infty$ is a $2$-sheeted cover and $S^0/(\mathbb{Z}/2) \simeq *$ but homotopically the Borel quotient is nontrivial.

---

## 5. The Monadic Bar Construction 🏗️

### 5.1 Setup and Definition

The algebraic and topological bar constructions are both special cases of the *monadic bar construction*. See [[concepts/category-theory/foundations/04-adjoint-functor-theorems-monads|the monads note]] for background on monads.

**Definition (Monad).** A *monad* on a category $\mathcal{C}$ is a triple $(T, \mu, \eta)$ where $T: \mathcal{C} \to \mathcal{C}$ is a functor, $\mu: T^2 \Rightarrow T$ is the *multiplication* (a natural transformation), and $\eta: \mathrm{Id} \Rightarrow T$ is the *unit*, satisfying the associativity and unit axioms.

**Definition (Monadic bar construction).** Given a monad $(T, \mu, \eta)$ on $\mathcal{C}$ and an object $X \in \mathcal{C}$, the *bar construction* $B_\bullet(T, X)$ is the simplicial object with:

- **Degree $n$ term:** $B_n(T, X) = T^{n+1}X$
- **Face maps:** $d_i = T^i \mu T^{n-i}: T^{n+2}X \to T^{n+1}X$ for $0 \leq i \leq n$

  (apply the monad multiplication $\mu: T^2 \Rightarrow T$ at the $i$-th position)
- **Degeneracy maps:** $s_j = T^j \eta T^{n-j}: T^{n+1}X \to T^{n+2}X$ for $0 \leq j \leq n$

  (apply the monad unit $\eta: \mathrm{Id} \Rightarrow T$ at the $j$-th position)

If $X$ is a $T$-algebra with structure map $\xi: TX \to X$, then the augmentation $d_{n+1} = T^n \xi: T^{n+1}X \to T^n X$ extends this to an *augmented* simplicial object.

> [!NOTE] The algebra case
> When $T = A \otimes_k (-)$ is the monad for modules over an algebra $A$ (with $\eta$ the unit map and $\mu$ the multiplication of $A$), $B_\bullet(T, k)$ recovers exactly the simplicial bar construction $B_\bullet(k, A, k)$ of section [[#3. The Simplicial Perspective|3]].

### 5.2 The Canonical Simplicial Resolution

**Theorem.** For any $T$-algebra $(X, \xi)$, the augmented simplicial object

$$B_\bullet(T,X) \xrightarrow{\xi \circ \mu^n} X$$

is a *simplicial resolution* of $X$ by free $T$-algebras: each $T^{n+1}X$ is freely generated (as a $T$-algebra) by $T^n X$.

The *geometric realization* $|B_\bullet(T,X)|$ is, when it exists (e.g., when $\mathcal{C}$ is a model category with enough structure), the *cofibrant replacement* of $X$ in the model category of $T$-algebras.

The contracting homotopy of the algebraic bar complex has a monadic incarnation: the maps $s_{-1} = \eta_{T^{n+1}X}: T^{n+1}X \to T^{n+2}X$ (apply $\eta$ on the outside) provide a simplicial homotopy between the identity and the constant map, showing that $|B_\bullet(T,X)|$ is contractible before applying the algebra structure.

### 5.3 Connection to Beck Monadicity

The bar construction is intimately connected to Beck's monadicity theorem (see [[concepts/category-theory/descent-monadicity|Descent and Monadicity]]). Given an adjunction $F \dashv U: \mathcal{C} \to \mathcal{D}$ with induced monad $T = UF$ on $\mathcal{D}$:

- The bar construction $B_\bullet(T, X)$ for $X \in \mathcal{D}$ is the *canonical* resolution of $X$ by free $T$-algebras.
- The coequalizer of $UFUFX \rightrightarrows UFX$ is $X$ itself (when $U$ reflects isomorphisms).
- Beck's theorem says $\mathcal{C}$ is equivalent to $\mathcal{D}^T$ exactly when $U$ preserves and reflects coequalizers of $U$-split pairs — and the bar construction provides the canonical *split pair* witnessing this.

**The bar construction as the canonical splitting:** The pair $(UFUF X, UFX)$ with face maps $UFU(\varepsilon_X): UFUFX \to UFX$ (counit applied inside) and $\varepsilon_{UFX}: UFUFX \to UFX$ (counit applied outside) is always $U$-split by the bar construction. The geometric realization $|B_\bullet(T,X)|$ gives the free resolution of $X$ used in descent theory.

> [!INFO] Why the bar construction deloops
> The moral reason the bar construction deloops $G$ to $BG$ is the *fibration sequence*
> $$G \longrightarrow EG \longrightarrow BG$$
> with $EG = B(G, G, *)$ contractible. Since $EG$ is contractible, the long exact sequence of homotopy groups collapses to $\pi_k(BG) \cong \pi_{k-1}(G)$ for all $k$ — so $BG$ is, by definition, the delooping: $\Omega BG \simeq G$.
>
> **Why is $EG$ contractible?** The "prepend the identity" map $s_{-1}(g_0, \ldots, g_n) = (e, g_0, \ldots, g_n)$ is a *simplicial contraction*: the face map $d_0 \circ s_{-1} = \mathrm{id}$, and $s_{-1}$ homotopes everything to a cone. This is the same algebraic trick as the acyclicity of the bar complex — *prepend the unit* — but now running topologically.
>
> **The simplicial picture**: The $n$-simplices $G^{n+1}$ of $EG$ are *chains* $(g_0, g_1, \ldots, g_n)$ — think of them as paths of length $n$ in $G$. The face maps multiply adjacent elements, collapsing paths. The geometric realization "fills in" all possible interpolations by stretching each chain over a topological $n$-simplex. The quotient by the free $G$-action $(g \cdot (g_0, \ldots, g_n)) = (gg_0, \ldots, gg_n)$ forgets the "starting point" and records only the *ratios* $g_i^{-1}g_{i+1}$ — which is exactly $BG = B(*,G,*)$.
>
> **A_∞ recognition theorem**: More generally, Stasheff (1963) and May (1972) showed that a connected, well-pointed space $X$ has the homotopy type of $\Omega Y$ for some $Y$ *if and only if* $X$ is an *$A_\infty$-space* (a space with a homotopy-coherently associative multiplication). When $X = G$ is a topological *group*, the $A_\infty$ structure is strict, so the recognition theorem applies cleanly: the delooping is $Y = |B_\bullet(*,G,*)| = BG$, the bar construction. For *iterated* deloopings ($X \simeq \Omega^k Y$), one needs a $k$-fold structure — the little $k$-cubes operad — and the iterated bar construction $B^k$; see [Mathew (2012)](https://amathew.wordpress.com/2012/01/09/delooping-and-the-bar-construction/) for a lucid exposition.
>
> **Summary**: The bar construction deloops because it *is* the path space construction in disguise — $EG$ is the contractible total space, $BG$ the base, and $G$ the fiber. Every delooping machine ultimately builds a contractible "path object" one dimension up and takes the quotient.

> [!INFO] Godement's standard construction
> The monadic bar construction is sometimes called *Godement's standard construction* or *the standard resolution* of a $T$-algebra, reflecting its role as the canonical (choice-free) resolution in categorical algebra.

---

## 6. The Bar-Cobar Adjunction and Koszul Duality 🔄

### 6.1 The Cobar Construction

Every bar construction has a dual: the *cobar construction*, which turns coalgebras into algebras.

**Definition (Coaugmented coalgebra).** A *coaugmented dg-coalgebra* is a dg-coalgebra $(C, \Delta, \varepsilon)$ with a map $\eta: k \to C$ (the *coaugmentation*) such that $\varepsilon \circ \eta = \mathrm{id}_k$. The *coaugmentation coideal* is $\bar{C} = \ker(\varepsilon: C \to k)$.

**Definition (Cobar construction).** The *cobar construction* $\Omega C$ of a coaugmented dg-coalgebra $C$ is the dg-algebra:

- **Underlying graded algebra:** $T(s^{-1}\bar{C})$, the free associative algebra on the *desuspension* $s^{-1}\bar{C}$ (shift of grading by $-1$).
- **Notation:** An element $(s^{-1}c_1) \otimes \cdots \otimes (s^{-1}c_n)$ is written $[c_1 | \cdots | c_n]$ (same bar notation, different meaning).
- **Differential:** The differential on $\Omega C$ has two parts:
  1. The internal differential of $C$, applied to each $c_i$.
  2. The *cobar differential* $\delta[c_1|\cdots|c_n] = \sum_i (-1)^{|c_1|+\cdots+|c_{i-1}|} [c_1|\cdots|\Delta(c_i)|\cdots|c_n]$, using the reduced coproduct $\bar{\Delta}: \bar{C} \to \bar{C} \otimes \bar{C}$.

### 6.2 The Adjunction B Dashv Omega

There is an *adjunction* between the category of augmented dg-algebras and the category of coaugmented conilpotent dg-coalgebras:

$$B \dashv \Omega: \mathbf{dgAlg}_{\mathrm{aug}} \rightleftharpoons \mathbf{dgCoalg}_{\mathrm{coaug}}^{\mathrm{conil}}$$

The *bar construction* $BA$ of an augmented dg-algebra $A$ is the dg-coalgebra with:
- **Underlying coalgebra:** $T^c(s\bar{A})$, the cofree conilpotent coalgebra on the suspension $s\bar{A}$.
- **Differential:** The bar differential from section [[#2.2 The One-Sided Bar Complex B(k, A, k)|2.2]], now incorporating any internal differential on $A$.
- **Coproduct:** The *deconcatenation coproduct* $\Delta[a_1|\cdots|a_n] = \sum_{i=0}^n [a_1|\cdots|a_i] \otimes [a_{i+1}|\cdots|a_n]$.

The adjunction unit $A \to \Omega BA$ is the *bar-cobar resolution* of $A$.

> [!WARNING] Conilpotency
> The coalgebra $BA$ is *conilpotent*: every element is annihilated by sufficiently many applications of the reduced coproduct $\bar{\Delta}$. Without this condition, the adjunction breaks down. For non-conilpotent coalgebras, one needs the *extended bar construction*.

### 6.3 Koszul Duality

For a *quadratic algebra* $A = T(V)/(R)$ where $R \subset V^{\otimes 2}$, the *Koszul dual coalgebra* is $A^¡ = T^c(sV)/(sR)^{\perp}$. The algebra $A$ is *Koszul* if the natural map

$$\Omega(A^¡) \xrightarrow{\sim} A$$

is a quasi-isomorphism — i.e., the bar-cobar resolution through the Koszul dual is *minimal* (the bar-cobar resolution of a general algebra is always a resolution, but typically far from minimal).

**Examples of Koszul algebras:**
- Polynomial algebras $k[x_1, \ldots, n]$ (Koszul dual: exterior algebra $\Lambda[x_1^*, \ldots, x_n^*]$)
- Exterior algebras (Koszul dual: polynomial algebra)
- The universal enveloping algebra $U(\mathfrak{g})$ for a Lie algebra $\mathfrak{g}$ (Koszul dual: the Chevalley-Eilenberg algebra)

> [!INFO] Koszul duality in homotopy theory
> Koszul duality has a deep homotopy-theoretic interpretation: the bar construction $BA$ is a model for the *suspension* of $A$ in an appropriate sense, and the cobar construction $\Omega C$ models the *loop space* functor. The Koszul duality $\Omega(A^!) \simeq A$ says that for Koszul algebras, one loop of one suspension returns you home — exactly as $\Omega\Sigma X \simeq X$ for simply-connected spaces under suitable conditions.

### 6.4 Adams' Cobar Construction for Loop Spaces

The original motivation for the cobar construction was purely topological. *J. Frank Adams* (1956) showed:

**Theorem (Adams).** For a simply-connected topological space $X$ with basepoint, there is a quasi-isomorphism of dg-algebras

$$\Omega C_*(X, *) \xrightarrow{\sim} C_*(\Omega X)$$

where $C_*(X, *)$ is the singular chain coalgebra of $X$ (with coproduct given by the Alexander-Whitney map), and $\Omega X$ is the based loop space with its Pontryagin product.

This remarkable theorem says: *the cobar construction on the chains of a space models the chains on its loop space.* The bar-cobar adjunction is the algebraic shadow of the geometric suspension-loop adjunction $\Sigma \dashv \Omega$.

---

## 7. Hochschild Homology and the Cyclic Bar Construction ♻️

### 7.1 The Hochschild Complex as a Bar Complex

Let $A$ be a $k$-algebra and $M$ an $A$-bimodule (equivalently, a left $A \otimes_k A^{\mathrm{op}}$-module). The *Hochschild complex* $C_*(A, M)$ is:

$$C_n(A, M) = M \otimes_k A^{\otimes_k n}, \quad n \geq 0$$

with differential $b: C_n \to C_{n-1}$ given by

$$b(m \otimes a_1 \otimes \cdots \otimes a_n) = ma_1 \otimes a_2 \otimes \cdots \otimes a_n + \sum_{i=1}^{n-1}(-1)^i m \otimes a_1 \otimes \cdots \otimes a_i a_{i+1} \otimes \cdots \otimes a_n + (-1)^n a_n m \otimes a_1 \otimes \cdots \otimes a_{n-1}.$$

**Observation:** $C_*(A, M) = B(M, A, k)$ — the two-sided bar complex with $N = k$ and the bimodule action folding into the two-sided structure. The *Hochschild homology* is therefore

$$HH_n(A, M) = H_n(C_*(A, M)) = \mathrm{Tor}_n^{A \otimes A^{\mathrm{op}}}(A, M).$$

For $M = A$ (with the natural bimodule structure), $HH_*(A, A) = \mathrm{Tor}_*^{A^e}(A, A)$ where $A^e = A \otimes_k A^{\mathrm{op}}$.

### 7.2 The Cyclic Bar Construction

Hochschild homology carries an extra structure: a *cyclic symmetry*. The key observation is that $C_n(A,A) = A^{\otimes(n+1)}$ admits a cyclic action by the cyclic group $C_{n+1}$ via

$$t_n(a_0 \otimes a_1 \otimes \cdots \otimes a_n) = (-1)^n a_n \otimes a_0 \otimes a_1 \otimes \cdots \otimes a_{n-1}$$

(cyclic rotation with sign). This is the *cyclic operator* of Connes.

**Definition (Cyclic bar construction).** The *cyclic bar construction* $B^{\mathrm{cy}}_\bullet(A)$ is the cyclic object with $B^{\mathrm{cy}}_n(A) = A^{\otimes(n+1)}$, face maps $d_i$ as in the Hochschild complex, and the cyclic operator $t_n$ above.

A *cyclic object* in $\mathcal{C}$ is a simplicial object with compatible cyclic actions — formally, a functor from Connes' cyclic category $\Lambda$ to $\mathcal{C}$.

### 7.3 Cyclic Homology and the Circle Action

The geometric realization $|B^{\mathrm{cy}}_\bullet(A)|$ carries an $S^1$-action (since cyclic objects realize to spaces with circle actions), and:

$$HH_*(A) = \pi_*(|B^{\mathrm{cy}}_\bullet(A)|)$$

The circle action on $|B^{\mathrm{cy}}(A)|$ gives, via the fibration $ES^1 \to BS^1 = \mathbb{CP}^\infty$, the *cyclic homology* $HC_*(A)$ as the $S^1$-equivariant version:

$$HC_*(A) = H_*^{S^1}(|B^{\mathrm{cy}}(A)|)$$

and the *periodic cyclic homology* $HP_*(A)$ is the corresponding Tate construction.

> [!INFO] Topological Hochschild Homology
> In stable homotopy theory, replacing $k$-modules with spectra and the tensor product $\otimes_k$ with the smash product $\wedge$ yields *topological Hochschild homology* $THH(A) = |B^{\mathrm{cy}}_\bullet(A)|$ for a ring spectrum $A$. This is the starting point for trace methods in algebraic $K$-theory (via the cyclotomic trace $K(A) \to TC(A)$).

---

## 8. The Categorical and Infinity-Categorical Perspective 🌌

### 8.1 Bar as Colimit: The Coequalizer Interpretation

The geometric realization of the bar construction is a *colimit*. The simplest case: for a $T$-algebra $X$, the realization $|B_\bullet(T,X)|$ is the *coequalizer* in $\mathcal{C}$:

$$T^2 X \underset{T\xi}{\overset{\mu_X}{\rightrightarrows}} TX \to X$$

where $\mu_X: T^2X \to TX$ is the monad multiplication and $T\xi: T^2X \to TX$ applies $T$ to the algebra structure $\xi: TX \to X$. This coequalizer is the *Beck coequalizer* and is always split (by the $T$-algebra axioms), hence absolute.

More precisely, $|B_\bullet(T,X)|$ is the *geometric realization* (= colimit over $\Delta^{\mathrm{op}}$) of the entire simplicial diagram, not just the coequalizer of the first two face maps. But for good enough monads, these agree.

**Proposition.** When $T$ preserves reflexive coequalizers and $\mathcal{C}$ has such coequalizers, $|B_\bullet(T,X)| \cong X$ for any $T$-algebra $(X, \xi)$.

*Proof sketch.* The colimit of $B_\bullet(T,X)$ is the coequalizer of $T^2X \rightrightarrows TX$, which by the split pair argument is $X$ itself. $\square$

### 8.2 The Infinity-Categorical Bar Construction

In the $\infty$-categorical setting (Lurie's Higher Algebra), the bar construction works even better because *geometric realization* (= $\infty$-colimit over $\Delta^{\mathrm{op}}$) is better behaved than ordinary coequalizers.

**Theorem (Barr-Beck-Lurie).** Let $f: \mathcal{C} \to \mathcal{D}$ be a functor of $\infty$-categories admitting a left adjoint $g$. Then $f$ is monadic if and only if:
1. $f$ is conservative (reflects equivalences).
2. $\mathcal{C}$ admits geometric realizations of $f$-split simplicial objects, and $f$ preserves them.

Under monadicity, the canonical comparison $|B_\bullet(T, X)| \simeq X$ holds for every $T$-algebra $X$.

The $\infty$-categorical bar construction also gives the functor

$$B: \mathbf{Alg}_{\mathcal{O}}(\mathcal{C}) \to \mathbf{CoAlg}_{\mathcal{O}^!}(\mathcal{C})$$

from $\mathcal{O}$-algebras to $\mathcal{O}^!$-coalgebras for an operad $\mathcal{O}$ and its Koszul dual $\mathcal{O}^!$ — Koszul duality for $\infty$-operads (Lurie HA §4.4).

> [!INFO] The bar construction and deformation theory
> In derived algebraic geometry, the bar construction controls deformation theory: the formal moduli problem of deforming an $\mathcal{O}$-algebra structure is controlled by the bar construction $B$ and its Koszul dual $\Omega$. This is the content of Lurie's theorem on formal moduli problems (HA §5.1).

### 8.3 Everything Is One Construction

We can now state the unifying principle:

**Every bar construction is the geometric realization of a simplicial object arising from a monad, with the face maps given by the monad multiplication and the augmentation given by an algebra structure map.**

| Version | Category $\mathcal{C}$ | Monad $T$ | Algebra $(X, \xi)$ | Result |
|---------|----------------------|-----------|-------------------|--------|
| Algebraic bar | $k$-modules | $A \otimes_k (-)$ | $(k, \varepsilon)$ | $B(k,A,k) \to k$ |
| Two-sided bar | $k$-modules | $A \otimes_k (-)$ | $(N, \text{action})$ | $B(M,A,N) \to M \otimes_A N$ |
| Group homology | $\mathbb{Z}$-modules | $\mathbb{Z}[G] \otimes (-)$ | $(\mathbb{Z}, \text{trivial})$ | $B(k,G,k)$ |
| Classifying space | Topological spaces | $G \times (-)$ | $(\{*\}, \text{trivial})$ | $BG = |NG|$ |
| Homotopy colimit | Topological spaces | $\bigsqcup_{d \to -} F(d)$ | (functor $F$) | $\mathrm{hocolim}_{\mathcal{D}} F$ |
| Hochschild | $k$-modules | $A \otimes_k (-) \otimes_k A$ | $(A, \mu)$ | $HH_*(A)$ |
| Cobar/Koszul | dg-modules | $T(s^{-1}\bar{C} \otimes -)$ | $(A, \text{alg. str.})$ | $\Omega C \to A$ |

*Surprisingly,* the contracting homotopy that proves acyclicity is always the same map: *prepend the unit* (of the monad, of the algebra, of the group, ...). The sign conventions vary but the idea is universal.

> [!QUESTION] Open question: operadic bar constructions
> The bar construction generalizes further to *operads*: the bar construction of an operad $\mathcal{P}$ is the cooperad $B\mathcal{P}$, and Koszul duality for operads says $\mathcal{P}$ is Koszul iff $\Omega(B\mathcal{P}) \simeq \mathcal{P}$ (quasi-isomorphism). Loday-Vallette give a comprehensive treatment. Much of derived algebraic geometry and deformation theory reduces to questions about these operadic bar constructions.

---

## 9. The Bar Construction in Spectra: B as Categorical Suspension 🌠

### 9.1 Augmented Ring Spectra and the Spectral Bar Construction

In the ∞-categorical setting, the bar construction generalizes to **ring spectra** — $\mathbb{E}_1$- or $\mathbb{E}_\infty$-algebra objects in the stable ∞-category of spectra $\mathrm{Sp}$. The algebraic field $k$ is replaced by the *sphere spectrum* $\mathbb{S}$ (the unit for the smash product $\wedge$), and the tensor product $\otimes_k$ is replaced by $\wedge$.

The input is an *augmented* $\mathbb{E}_n$-ring spectrum: an $\mathbb{E}_n$-algebra $A$ with a map of $\mathbb{E}_n$-algebras $\varepsilon: A \to \mathbb{S}$. The *augmentation ideal* is the fiber:

$$\bar{A} = \mathrm{fib}(\varepsilon: A \to \mathbb{S}), \quad \text{fitting into } \bar{A} \to A \xrightarrow{\varepsilon} \mathbb{S}.$$

**Definition (Spectral bar construction).** The *bar construction* of an augmented $\mathbb{E}_1$-ring spectrum $A$ is the *relative smash product*:

$$BA = \mathbb{S} \wedge_A \mathbb{S}$$

i.e., the geometric realization of the simplicial spectrum $B_\bullet(\mathbb{S}, A, \mathbb{S})$ with $B_n = \mathbb{S} \wedge A^{\wedge n} \wedge \mathbb{S}$, face maps using the multiplication of $A$ and the augmentation maps at the ends, degeneracy maps inserting the unit $\eta: \mathbb{S} \to A$.

### 9.2 B as Categorical Suspension, Not Spectrum Suspension

The key clarification: **$BA$ is not $\Sigma A$ as a spectrum**, but rather the suspension of $A$ *in the ∞-category of augmented $\mathbb{E}_\infty$-algebras*.

The *suspension functor* in any pointed ∞-category $\mathcal{C}$ (with zero object $0$) is:

$$\Sigma_\mathcal{C}\, X = 0 \sqcup_X 0 \quad (\text{pushout of two maps } X \to 0).$$

| Category $\mathcal{C}$ | Zero object | $\Sigma_\mathcal{C}\, X$ |
|---|---|---|
| Pointed spaces $\mathcal{S}_*$ | $*$ | $S^1 \wedge X$ (unreduced suspension) |
| Spectra $\mathrm{Sp}$ | $*$ | $\Sigma X$ (shift by 1) |
| $\mathrm{Alg}^{\mathrm{aug}}_{\mathbb{E}_\infty}$ | $\mathbb{S}$ (initial and terminal) | $\mathbb{S} \sqcup_A \mathbb{S} = \mathbb{S} \wedge_A \mathbb{S} = BA$ |

So $BA$ **is** the suspension of $A$ — but only in the ∞-category of augmented $\mathbb{E}_\infty$-algebras. The forgetful functor $\mathrm{Alg}^{\mathrm{aug}}_{\mathbb{E}_\infty} \to \mathrm{Sp}$ does not preserve pushouts, so this algebraic suspension does not coincide with the spectrum-level suspension.

> [!WARNING] $BA \neq \Sigma A$ as spectra
> The homotopy groups of $BA$ are computed by the *bar spectral sequence*:
>
> $$E^2_{p,q} = \mathrm{Tor}^{\pi_*A}_{p,q}(\pi_*\mathbb{S},\ \pi_*\mathbb{S}) \implies \pi_{p+q}(BA).$$
>
> For a non-free algebra, this spectral sequence has multiple non-zero rows and the result is far from $\Sigma A$. For example, for $A = H\mathbb{F}_p$ (the Eilenberg-MacLane spectrum), $BA \simeq H\mathbb{F}_p \wedge_{H\mathbb{F}_p \wedge H\mathbb{F}_p} H\mathbb{F}_p$ computes the dual Steenrod algebra, not $\Sigma H\mathbb{F}_p$.

### 9.3 The Free Algebra Case: When BA Simeq Sigma M

The one case where spectral bar and spectrum suspension coincide is for **free algebras**.

**Theorem.** Let $M$ be a spectrum. For the free $\mathbb{E}_\infty$-algebra $\mathrm{Sym}(M) = \bigoplus_{n \geq 0} (M^{\wedge n})_{h\Sigma_n}$ on $M$:

$$B(\mathrm{Sym}(M)) \simeq \Sigma M \quad \text{(as spectra)}.$$

*Why.* The bar spectral sequence degenerates: $\mathrm{Tor}^{\pi_*\mathrm{Sym}(M)}_{p,*}(\pi_*\mathbb{S}, \pi_*\mathbb{S}) = 0$ for $p \geq 2$, because the Tor groups of a polynomial algebra over itself vanish above the indecomposables (homological degree 1). The first column is $\bar{A}/\bar{A}^{\otimes 2} \simeq \Sigma M$ and everything else is zero.

This is the spectral version of the classical fact: the *indecomposables* of $k[V] = \mathrm{Sym}(V)$ are just $V$ (all higher-order terms are decomposable). The bar construction strips away everything but the generators.

> [!EXAMPLE] Polynomial algebra over a field
> Let $A = k[x]$ with $|x| = n$, augmented by $\varepsilon(x) = 0$. The classical bar complex $B(k, k[x], k)$ has $E^2_{1,*} = \bar{A}/\bar{A}^2 = k\cdot x$ concentrated in homological degree 1. All higher Tor groups vanish (polynomial algebras are Koszul). So $B(k[x]) \simeq \Sigma k\cdot x$ — a single class shifted by one. This is the $k$-linear version of $B(\mathrm{Sym}(M)) \simeq \Sigma M$.

### 9.4 Iterated Bar and En-Algebras

The bar construction can be iterated, and each application "uses up" one level of commutativity. For an augmented $\mathbb{E}_n$-algebra $A$ (for $n \geq 1$), the *$n$-fold bar construction* is:

$$B^n A = \underbrace{B \circ B \circ \cdots \circ B}_{n}(A)$$

Each $B$ lowers the $\mathbb{E}_k$-level by one: $B$ sends augmented $\mathbb{E}_k$-algebras to augmented $\mathbb{E}_{k-1}$-coalgebras. So $B^n A$ requires $A$ to be an $\mathbb{E}_n$-algebra (at minimum), and produces an $\mathbb{E}_0$-coalgebra (a coaugmented cochain complex / cospectra).

**The iterated bar construction is the $\mathbb{E}_n$-suspension:**

$$B^n A = \Sigma_{\mathrm{Alg}^{\mathrm{aug}}_{\mathbb{E}_n}} A.$$

For free algebras, this degenerates cleanly:

$$B^n(\mathrm{Free}_{\mathbb{E}_n}(M)) \simeq \Sigma^n M.$$

The full hierarchy of structures and their delooping machines:

| Algebra structure on $A$ | $B^n A$ | Geometric meaning |
|---|---|---|
| $\mathbb{E}_1$ (associative) | $BA = \mathbb{S} \wedge_A \mathbb{S}$ | Classifying space / one delooping |
| $\mathbb{E}_2$ (braided) | $B^2 A = B(BA)$ | Two-fold delooping |
| $\mathbb{E}_n$ | $B^n A$ | $n$-fold delooping |
| $\mathbb{E}_\infty$ (commutative) | $B^\infty A$ | Suspension spectrum $\Sigma^\infty_+ A$ |

In the $\mathbb{E}_\infty$ (fully commutative / stable) case, the ∞-fold bar construction of a grouplike $\mathbb{E}_\infty$-space $A$ is its *suspension spectrum* $\Sigma^\infty_+ A$ — the stable homotopy type encoding only the additive structure of $A$.

> [!INFO] Koszul duality as $B^n \dashv \Omega^n$
> The iterated cobar construction $\Omega^n C$ inverts $B^n$: $\Omega^n(B^n A) \simeq A$ for Koszul $\mathbb{E}_n$-algebras (those for which the bar-cobar resolution is minimal). This is Francis-Gaitsgory's *$\mathbb{E}_n$-Koszul duality*. For $n = 1$, it reduces to classical Koszul duality of §6.3. For $n = \infty$, it recovers the equivalence between connective $\mathbb{E}_\infty$-algebras and connective cocommutative coalgebras — a spectral version of the Milnor-Moore theorem.

---

## References

| Reference Name | Brief Summary | Link to Reference |
|---|---|---|
| [Eilenberg & Mac Lane (1953)](https://doi.org/10.2307/1969820) | Original bar notation for homology of $K(\pi,n)$ | https://doi.org/10.2307/1969820 |
| [Milnor (1956)](https://doi.org/10.2307/1969983) | Universal bundle $EG \to BG$ via infinite join | https://doi.org/10.2307/1969983 |
| [Adams (1956)](https://doi.org/10.1073/pnas.42.7.409) | Cobar construction $\Omega C$ for loop spaces | https://doi.org/10.1073/pnas.42.7.409 |
| [Stasheff (1963)](https://doi.org/10.1090/S0002-9947-1963-99936-3) | Associahedra, $A_\infty$-spaces, bar for $H$-spaces | https://doi.org/10.1090/S0002-9947-1963-99936-3 |
| [Segal (1968)](https://link.springer.com/article/10.1007/BF02684591) | Nerve $BG = |NG|$; classifying spaces | https://link.springer.com/article/10.1007/BF02684591 |
| [May (1972), Geometry of Iterated Loop Spaces](https://www.math.uchicago.edu/~may/BOOKS/gils.pdf) | Two-sided bar $B(Y,M,X)$, monadic bar | https://www.math.uchicago.edu/~may/BOOKS/gils.pdf |
| [May (1975), Classifying Spaces and Fibrations](https://doi.org/10.1090/memo/0155) | Geometric bar for fibrations | https://doi.org/10.1090/memo/0155 |
| [Weibel (1994)](https://www.sas.rochester.edu/mth/sites/doug-ravenel/otherpapers/weibel-homv2.pdf) | Algebraic bar resolution, Tor, Ext | https://www.sas.rochester.edu/mth/sites/doug-ravenel/otherpapers/weibel-homv2.pdf |
| [McCleary (2001)](https://www.cambridge.org/core/books/users-guide-to-spectral-sequences/524E3BD5DDD6D387E3C609910142F6E6) | Eilenberg-Moore SS, cobar construction | https://www.cambridge.org/core/books/users-guide-to-spectral-sequences/524E3BD5DDD6D387E3C609910142F6E6 |
| [Loday (1992), Cyclic Homology](https://link.springer.com/book/10.1007/978-3-662-11389-9) | Cyclic bar construction, $HC_*$ | https://link.springer.com/book/10.1007/978-3-662-11389-9 |
| [Loday & Vallette (2012), Algebraic Operads](https://www.math.univ-paris13.fr/~vallette/Operads.pdf) | Bar/cobar for operads, Koszul duality | https://www.math.univ-paris13.fr/~vallette/Operads.pdf |
| [May & Ponto (2012)](https://webhomes.maths.ed.ac.uk/~v1ranick/papers/mayponto.pdf) | Modern classifying spaces, model categories | https://webhomes.maths.ed.ac.uk/~v1ranick/papers/mayponto.pdf |
| [Lurie (2017), Higher Algebra](https://www.math.ias.edu/~lurie/papers/HA.pdf) | $\infty$-bar construction, Barr-Beck-Lurie | https://www.math.ias.edu/~lurie/papers/HA.pdf |
| [Trimble/Schreiber (2007), n-Café](https://golem.ph.utexas.edu/category/2007/05/on_the_bar_construction.html) | Categorical acyclicity of bar complex | https://golem.ph.utexas.edu/category/2007/05/on_the_bar_construction.html |
| [Malkiewich, "The Bar Construction"](https://people.math.binghamton.edu/malkiewich/bar.pdf) | Self-contained notes on algebraic bar | https://people.math.binghamton.edu/malkiewich/bar.pdf |
| [Malkiewich, "Homotopy Colimits via Bar"](https://people.math.binghamton.edu/malkiewich/hocolim_bar.pdf) | Hocolim $= B(*,\mathcal{D},F)$ | https://people.math.binghamton.edu/malkiewich/hocolim_bar.pdf |
| [nLab, "two-sided bar construction"](https://ncatlab.org/nlab/show/two-sided+bar+construction) | Categorical unification of all versions | https://ncatlab.org/nlab/show/two-sided+bar+construction |
| [Segal (1974)](https://www.sciencedirect.com/science/article/pii/0040938374900226) | $\Gamma$-spaces; bar construction as classifying space of $\Gamma$-category | https://www.sciencedirect.com/science/article/pii/0040938374900226 |
| [Zhang (2019), Two-Sided Bar Construction](http://math.uchicago.edu/~may/REU2019/REUPapers/Zhang,Ruoqi(Rachel).pdf) | REU paper: two-sided bar, Tor, homotopy colimits | http://math.uchicago.edu/~may/REU2019/REUPapers/Zhang,Ruoqi(Rachel).pdf |
| [Mathew (2012), "Delooping and the Bar Construction"](https://amathew.wordpress.com/2012/01/09/delooping-and-the-bar-construction/) | Intuitive explanation of why bar deloops; $A_\infty$ recognition, iterated deloopings, little cubes operads | https://amathew.wordpress.com/2012/01/09/delooping-and-the-bar-construction/ |
