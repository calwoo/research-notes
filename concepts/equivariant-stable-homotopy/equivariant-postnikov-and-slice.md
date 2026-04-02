# Equivariant Postnikov Towers and the Slice Filtration

## Table of Contents

- [[#1. Non-Equivariant Postnikov Towers|1. Non-Equivariant Postnikov Towers]]
  - [[#1.1 Postnikov Sections: Definition and Characterization|1.1 Postnikov Sections: Definition and Characterization]]
  - [[#1.2 Construction via the Small Object Argument|1.2 Construction via the Small Object Argument]]
  - [[#1.3 The Fiber Sequences and k-Invariants|1.3 The Fiber Sequences and k-Invariants]]
  - [[#1.4 Postnikov Towers as Localization|1.4 Postnikov Towers as Localization]]
  - [[#1.5 The Atiyah-Hirzebruch Spectral Sequence|1.5 The Atiyah-Hirzebruch Spectral Sequence]]
- [[#2. Equivariant Postnikov Towers: Unstable|2. Equivariant Postnikov Towers: Unstable]]
  - [[#2.1 Equivariant Homotopy Groups as Coefficient Systems|2.1 Equivariant Homotopy Groups as Coefficient Systems]]
  - [[#2.2 Construction via Elmendorf's Theorem|2.2 Construction via Elmendorf's Theorem]]
  - [[#2.3 Equivariant Eilenberg-Mac Lane Spaces|2.3 Equivariant Eilenberg-Mac Lane Spaces]]
  - [[#2.4 Fiber Sequences and Bredon k-Invariants|2.4 Fiber Sequences and Bredon k-Invariants]]
  - [[#2.5 Non-Equivariant vs. Equivariant Comparison|2.5 Non-Equivariant vs. Equivariant Comparison]]
- [[#3. Why the Naive Stable Analogue Fails|3. Why the Naive Stable Analogue Fails]]
  - [[#3.1 The Integer-Indexed Stable Postnikov Tower|3.1 The Integer-Indexed Stable Postnikov Tower]]
  - [[#3.2 Non-Nullhomotopic Maps Between Representation Spheres|3.2 Non-Nullhomotopic Maps Between Representation Spheres]]
  - [[#3.3 Mackey Functors vs. Coefficient Systems|3.3 Mackey Functors vs. Coefficient Systems]]
  - [[#3.4 The Remedy: Representation-Indexed Filtration|3.4 The Remedy: Representation-Indexed Filtration]]
- [[#4. The Slice Filtration|4. The Slice Filtration]]
  - [[#4.1 Slice Cells|4.1 Slice Cells]]
  - [[#4.2 Slice Null and Slice Positive Spectra|4.2 Slice Null and Slice Positive Spectra]]
  - [[#4.3 Relationship to Ordinary Connectivity|4.3 Relationship to Ordinary Connectivity]]
  - [[#4.4 Characterization via Filtrations|4.4 Characterization via Filtrations]]
  - [[#4.5 The Slice Sections and the Slice Tower|4.5 The Slice Sections and the Slice Tower]]
  - [[#4.6 The n-Slice and an Explicit Example|4.6 The n-Slice and an Explicit Example]]
  - [[#4.7 Behavior Under Change Functors|4.7 Behavior Under Change Functors]]
- [[#5. The Slice Spectral Sequence|5. The Slice Spectral Sequence]]
  - [[#5.1 Construction from the Slice Tower|5.1 Construction from the Slice Tower]]
  - [[#5.2 Dugger's C2 Construction and KR-Theory|5.2 Dugger's C2 Construction and KR-Theory]]
  - [[#5.3 The General HHR Slice Spectral Sequence|5.3 The General HHR Slice Spectral Sequence]]
- [[#6. Comparison and the HHR Application|6. Comparison and the HHR Application]]
  - [[#6.1 The Three-Way Comparison Table|6.1 The Three-Way Comparison Table]]
  - [[#6.2 The Slices of the Real Cobordism Spectrum|6.2 The Slices of the Real Cobordism Spectrum]]
- [[#References|References]]

---

## 1. Non-Equivariant Postnikov Towers 📐

We begin with a careful review of the classical theory. This section establishes the precise statements that will be *upgraded* in the equivariant and stable settings.

### 1.1 Postnikov Sections: Definition and Characterization

Throughout this section, $X$ is a based, connected topological space (or spectrum, when noted).

**Definition (Postnikov Section).** A *Postnikov section* of $X$ at level $n$ is a space $P_n X$ equipped with a map $p_n: X \to P_n X$ such that:

1. $(p_n)_*: \pi_k(X) \xrightarrow{\sim} \pi_k(P_n X)$ is an isomorphism for all $k \leq n$,
2. $\pi_k(P_n X) = 0$ for all $k > n$,
3. $X \simeq \operatorname{holim}_n P_n X$ (convergence of the tower).

The natural transformations $P_n X \to P_{n-1} X$ are induced by the universal property: since $\pi_k(P_{n-1} X) = 0$ for $k \geq n$ and $\pi_k(P_n X) = \pi_k(X)$ for $k \leq n-1$, there is an essentially unique compatible map between sections.

The collection $\{\cdots \to P_2 X \to P_1 X \to P_0 X\}$ is the *Postnikov tower* of $X$.

> [!INFO] Duality with CW complexes
> The Postnikov tower is, in a precise sense, dual to the cellular (CW) filtration. The CW filtration is a colimit of skeleta $X^0 \hookrightarrow X^1 \hookrightarrow X^2 \hookrightarrow \cdots$ where each attachment kills only finitely many cells; the Postnikov tower is a limit of stages $\cdots \to P_2 X \to P_1 X \to P_0 X$ where each stage kills infinitely many cells. The axiomatization of Postnikov towers in a triangulated category is called a *t-structure*; the dual axiomatization of CW filtrations is a *weight structure*.

### 1.2 Construction via the Small Object Argument

There is a canonical point-set construction. The naive attempt — given $\alpha \in \pi_{n+1}(X)$, form a pushout
$$\begin{tikzcd}
S^{n+1} \arrow[r, "\alpha"] \arrow[d, hook] & X \arrow[d] \\
D^{n+2} \arrow[r] & X \cup_\alpha D^{n+2}
\end{tikzcd}$$
to kill $\alpha$ — is not functorial: it depends on the choice of representative $\alpha$.

The fix is the *small object argument*. Fix $n \geq 0$. Define:

$$X^{(n+1)} := X \cup \left( \coprod_{\alpha: S^{n+1} \to X} D^{n+2} \right),$$

where we attach a disk along every map $S^{n+1} \to X$ simultaneously. Iterate: form $X^{(N)}$ by attaching disks along all maps $S^N \to X^{(N-1)}$ for each $N > n+1$. Set

$$P_n X := \operatorname{colim}_{N \geq n+1} X^{(N)}.$$

This is functorial in $X$: a map $f: X \to Y$ induces maps between the pushout diagrams at each stage by post-composition.

> [!NOTE] Why the iteration is necessary
> Attaching a disk to kill $\alpha \in \pi_{n+1}(X)$ may create new elements in $\pi_{n+2}$ (via the attaching map $S^{n+1} \to X$, viewed as a class in the relative homotopy). One must iterate to kill all these new contributions. The colimit converges because at each stage we only affect homotopy groups in degrees $> n$.

**Proposition.** The map $X \to P_n X$ constructed above satisfies the Postnikov section axioms. *The proof follows by induction on the relative homotopy long exact sequence and the observation that the colimit is filtered, so the homotopy groups of $P_n X$ are computed as the colimit of the homotopy groups.*

### 1.3 The Fiber Sequences and k-Invariants

**Proposition (Fiber Sequence).** For each $n \geq 1$, there is a homotopy fiber sequence

$$K(\pi_n X, n) \longrightarrow P_n X \longrightarrow P_{n-1} X.$$

*Proof sketch.* Let $F_n$ be the homotopy fiber of $P_n X \to P_{n-1} X$. From the long exact sequence on homotopy groups:

$$\cdots \to \pi_k(F_n) \to \pi_k(P_n X) \to \pi_k(P_{n-1} X) \to \pi_{k-1}(F_n) \to \cdots$$

By the Postnikov section axioms:
- For $k < n$: $\pi_k(P_n X) \cong \pi_k(X) \cong \pi_k(P_{n-1} X)$, so $\pi_k(F_n) = 0$.
- For $k = n$: $\pi_n(P_n X) \cong \pi_n(X)$ and $\pi_n(P_{n-1} X) = 0$, so $\pi_n(F_n) \cong \pi_n(X)$.
- For $k > n$: $\pi_k(P_n X) = 0$ and $\pi_k(P_{n-1} X) = 0$, so $\pi_k(F_n) = 0$.

Thus $F_n$ has exactly one nonzero homotopy group, $\pi_n(F_n) \cong \pi_n(X)$, in degree $n$, so $F_n \simeq K(\pi_n X, n)$. $\square$

The map $P_n X \to P_{n-1} X$ is classified by an element

$$k^{n+2}(X) \in H^{n+2}(P_{n-1} X;\, \pi_n X),$$

called the *(n+2)-th) k-invariant* of $X$. Here we use the bijection $[P_{n-1} X, K(\pi_n X, n+1)] \cong H^{n+2}(P_{n-1} X; \pi_n X)$ (noting that $K(\pi_n X, n+1)$ is the classifying space for $H^{n+1}(-; \pi_n X)$, so a map $P_{n-1} X \to K(\pi_n X, n+1)$ — whose homotopy fiber is $K(\pi_n X, n)$ — is classified by a class in $H^{n+2}$). *Iterating the k-invariants completely determines the Postnikov tower, and hence the homotopy type of $X$, up to extensions.*

> [!EXAMPLE]- k-invariant for the 2-sphere
> For $X = S^2$, the first nontrivial k-invariant lives in $H^4(K(\mathbb{Z},2); \mathbb{Z}) \cong \mathbb{Z}$, generated by the fundamental class. This k-invariant records the Hopf fibration $S^3 \to S^2$; its nontriviality reflects $\pi_3(S^2) \cong \mathbb{Z}$.

### 1.4 Postnikov Towers as Localization

The Postnikov section $P_n$ has a clean categorical description: it is *$n$-truncation*, or localization with respect to the class of maps that are $\pi_k$-isomorphisms for $k \leq n$ and $\pi_k$-surjections for $k = n+1$.

More precisely, $P_n$ is the left adjoint (in the homotopy category) to the inclusion of $n$-truncated spaces (spaces with $\pi_k = 0$ for $k > n$) into all spaces. The unit of this adjunction is the map $X \to P_n X$.

This localization perspective is crucial: it will generalize directly to the equivariant stable setting, where the slice sections $P^n$ are defined as localizations with respect to an appropriate class of spectra.

### 1.5 The Atiyah-Hirzebruch Spectral Sequence

For a generalized (co)homology theory $E_*$, the Postnikov tower of a space $X$ gives rise to the *Atiyah-Hirzebruch spectral sequence* (AHSS). The key observation is that the fiber sequences

$$\Sigma^n H\pi_n(X) \to \Sigma^\infty P_n X \to \Sigma^\infty P_{n-1} X$$

(after suspending to spectra) form an exact couple whose associated spectral sequence has:

$$E_2^{p,q} = H^p(X;\, E_q(*)) \Implies E^{p+q}(X).$$

*The $E_2$-page is the Bredon cohomology of $X$ with coefficients in the homotopy groups of $E$; the spectral sequence converges to the $E$-cohomology of $X$.* The analogous construction in the equivariant stable setting — replacing the Postnikov tower with the slice tower — produces the slice spectral sequence.

> [!NOTE] t-structures
> The abstract framework for Postnikov towers in a triangulated (or stable $\infty$-) category is a *t-structure* $(\mathcal{C}_{\geq 0}, \mathcal{C}_{\leq 0})$: a pair of full subcategories closed under appropriate suspensions with the property that every object $X$ fits into a unique fiber sequence $X_{\geq n} \to X \to X_{\leq n-1}$. The slice filtration will be a t-structure on $\mathrm{Sp}^G$ (though with integer indexing replaced by a more refined dimension function).

---

## 2. Equivariant Postnikov Towers: Unstable 💡

We now pass to the equivariant setting, working in $G\mathbf{Top}$ (see [[concepts/equivariant-stable-homotopy/g-spaces-and-equivariant-maps|G-Spaces and Equivariant Maps]] for foundations). The key insight — due to Elmendorf — is that the equivariant Postnikov tower is completely determined by the ordinary Postnikov towers of all the fixed-point spaces simultaneously.

### 2.1 Equivariant Homotopy Groups as Coefficient Systems

Recall from [[concepts/equivariant-stable-homotopy/g-spaces-and-equivariant-maps|G-Spaces and Equivariant Maps]] that the *equivariant homotopy groups* of a based $G$-space $X$ are defined, for each closed subgroup $H \leq G$, as

$$\pi_n^H(X) := \pi_n(X^H),$$

the $n$-th homotopy group of the $H$-fixed-point space.

**Definition (Coefficient System).** A *coefficient system* for $G$ is a functor $\underline{M}: \mathcal{O}_G^{op} \to \mathbf{Ab}$, where $\mathcal{O}_G$ is the *orbit category* of $G$ (objects: orbits $G/H$ for closed $H \leq G$; morphisms: $G$-equivariant maps $G/H \to G/K$, corresponding to elements $g \in G$ with $gHg^{-1} \leq K$).

The equivariant homotopy groups assemble into a coefficient system: define

$$\underline{\pi}_n X: \mathcal{O}_G^{op} \to \mathbf{Ab}, \qquad G/H \mapsto \pi_n(X^H).$$

The functoriality is as follows: a $G$-map $G/H \to G/K$ (corresponding to conjugation by some $g$) induces a map $X^K \to X^H$ on fixed-point spaces (by precomposition), and hence a map $\pi_n(X^K) \to \pi_n(X^H)$ — these are the *restriction maps* of the coefficient system.

> [!WARNING] Coefficient systems vs. Mackey functors
> A coefficient system has only *restriction maps* (contravariant functoriality). A *Mackey functor* additionally has *transfer maps* (covariant functoriality) satisfying a double-coset formula. Stably, homotopy groups always form Mackey functors; unstably, they only form coefficient systems. This distinction is invisible non-equivariantly (where there are no proper subgroups to transfer along) but is fundamental in the equivariant stable setting. See Section 3.3 for more.

### 2.2 Construction via Elmendorf's Theorem

**Elmendorf's Theorem** (recalled from [[concepts/equivariant-stable-homotopy/g-spaces-and-equivariant-maps|G-Spaces and Equivariant Maps]]): There is a Quillen equivalence

$$G\mathbf{Top} \simeq \mathrm{Fun}(\mathcal{O}_G^{op}, \mathbf{Spaces}),$$

under which a $G$-space $X$ corresponds to the presheaf $G/H \mapsto X^H$.

This equivalence immediately gives us equivariant Postnikov sections for free:

**Definition (Equivariant Postnikov Section).** The *equivariant Postnikov section* $P_n^G X$ is the $G$-space corresponding, under Elmendorf's theorem, to the presheaf

$$G/H \mapsto P_n(X^H).$$

Equivalently:

**Proposition.** $P_n^G X$ is uniquely characterized (up to weak $G$-equivalence) by:
1. $\pi_k^H(P_n^G X) \cong \pi_k^H(X)$ for all closed $H \leq G$ and $k \leq n$,
2. $\pi_k^H(P_n^G X) = 0$ for all closed $H \leq G$ and $k > n$,
3. $X \simeq \operatorname{holim}_n P_n^G X$.

The fixed-point formula $(P_n^G X)^H \simeq P_n(X^H)$ is the key identity: the equivariant Postnikov section is computed objectwise on fixed-point spaces.

> [!INFO] As an application of Elmendorf
> Blumberg's lecture notes explicitly identify the equivariant Postnikov tower as "another application of Elmendorf's theorem." The equivalence $G\mathbf{Top} \simeq \mathrm{Fun}(\mathcal{O}_G^{op}, \mathbf{Spaces})$ reduces any homotopy-theoretic construction on $G$-spaces — such as Postnikov sections — to the corresponding objectwise construction on presheaves of spaces. This is the power of the Elmendorf perspective.

The construction of Section 1.2 applies objectwise: form $(P_n^G X)^H$ by the small object argument applied to $X^H$, using all maps $S^k \to X^H$ for $k > n$. The equivariance is automatic because the construction is functorial in $X^H$, and the fixed-point functors $X \mapsto X^H$ are functorial in $G$-equivariant maps.

### 2.3 Equivariant Eilenberg-Mac Lane Spaces

**Definition (Equivariant Eilenberg-Mac Lane Space).** For a coefficient system $\underline{M}: \mathcal{O}_G^{op} \to \mathbf{Ab}$ and an integer $n \geq 1$, the *equivariant Eilenberg-Mac Lane space* $K(\underline{M}, n)$ is the $G$-space corresponding under Elmendorf's theorem to the presheaf

$$G/H \mapsto K(\underline{M}(G/H),\, n).$$

That is, $(K(\underline{M},n))^H = K(\underline{M}(G/H), n)$.

**Theorem (Representability).** For a $G$-CW complex $X$ and a coefficient system $\underline{M}$:

$$[X,\, K(\underline{M}, n)]_G \cong H_G^n(X;\, \underline{M}),$$

where $H_G^*(X; \underline{M})$ denotes *Bredon cohomology* with coefficients in $\underline{M}$.

*Proof sketch.* Under Elmendorf's equivalence, $G$-equivariant maps $X \to K(\underline{M},n)$ correspond to natural transformations between the corresponding presheaves. Since $K(\underline{M}(G/H),n)$ represents $H^n(-; \underline{M}(G/H))$ objectwise, and Bredon cohomology is defined as the cohomology of the chain complex $\mathrm{Hom}_{\mathcal{O}_G}(C_*^{cell}(X), \underline{M})$ (where $C_*^{cell}(X)(G/H) = H_*(X^H, (X^H)^{n-1})$ are the cellular chains), the result follows from the objectwise representability and the Elmendorf equivalence. $\square$

> [!NOTE] Bredon cohomology
> Bredon cohomology, introduced by Bredon in 1967, is the "correct" cohomology theory for $G$-spaces. Its coefficient systems take values in all of $\mathcal{O}_G$, not just individual subgroups. The equivariant Eilenberg-Mac Lane spaces show that Bredon cohomology classes have geometric representatives, exactly as in the non-equivariant case.

### 2.4 Fiber Sequences and Bredon k-Invariants

**Proposition (Equivariant Fiber Sequence).** For each $n \geq 1$, there is a homotopy fiber sequence of $G$-spaces:

$$K(\underline{\pi}_n X, n) \longrightarrow P_n^G X \longrightarrow P_{n-1}^G X.$$

*Proof.* Taking $H$-fixed points, we get the ordinary fiber sequence $K(\pi_n(X^H), n) \to P_n(X^H) \to P_{n-1}(X^H)$ for each $H$. Since Elmendorf's theorem is a Quillen equivalence (in particular, it preserves fiber sequences), the objectwise fiber sequences assemble into a $G$-equivariant fiber sequence. $\square$

The fiber sequence is classified by a *Bredon k-invariant*

$$k^{n+2}(X) \in H_G^{n+2}(P_{n-1}^G X;\, \underline{\pi}_n X),$$

which lives in Bredon cohomology (not ordinary cohomology). This follows from the representability theorem: the map $P_{n-1}^G X \to K(\underline{\pi}_n X, n+1)$ classifying the fiber sequence corresponds to a class in $[P_{n-1}^G X, K(\underline{\pi}_n X, n+1)]_G \cong H_G^{n+2}(P_{n-1}^G X; \underline{\pi}_n X)$.

> [!WARNING] The role of coefficient systems vs. Mackey functors here
> In the unstable equivariant setting, the k-invariants live in Bredon cohomology with coefficient system coefficients — not Mackey functor coefficients. This is because the coefficient system $\underline{\pi}_n X$ only has restriction maps (it sees $X^K \to X^H$ for inclusions $H \hookrightarrow K$), not transfer maps. Stably, this upgrades: the $n$-slice involves Mackey functors with genuine transfer maps. See Section 3.3.

### 2.5 Non-Equivariant vs. Equivariant Comparison

The following table summarizes the parallel structure between the non-equivariant and equivariant unstable Postnikov towers.

| Feature | Non-equivariant $\mathbf{Top}$ | Equivariant $G\mathbf{Top}$ |
|---|---|---|
| Homotopy groups | $\pi_n(X) \in \mathbf{Ab}$ | $\underline{\pi}_n X: \mathcal{O}_G^{op} \to \mathbf{Ab}$ (coefficient system) |
| Postnikov section | $P_n X$, $\pi_k(P_n X) = \begin{cases}\pi_k X & k\leq n \\ 0 & k > n\end{cases}$ | $P_n^G X$, same for each $\pi_k^H$ |
| Construction | Small object argument on $X$ | Small object argument objectwise on each $X^H$ |
| Eilenberg-Mac Lane | $K(A, n)$, $A \in \mathbf{Ab}$ | $K(\underline{M}, n)$, $\underline{M}: \mathcal{O}_G^{op} \to \mathbf{Ab}$ |
| Fiber sequence | $K(\pi_n X, n) \to P_n X \to P_{n-1} X$ | $K(\underline{\pi}_n X, n) \to P_n^G X \to P_{n-1}^G X$ |
| k-invariant | $k^{n+2} \in H^{n+2}(P_{n-1} X; \pi_n X)$ | $k^{n+2} \in H_G^{n+2}(P_{n-1}^G X; \underline{\pi}_n X)$ (Bredon) |
| Key tool | Whitehead theorem | Elmendorf's theorem + objectwise Whitehead |

**The formal structure is identical; the coefficients upgrade from abelian groups to coefficient systems, and ordinary cohomology upgrades to Bredon cohomology.**

---

## 3. Why the Naive Stable Analogue Fails ⚠️

Passing to genuine $G$-spectra indexed by a complete universe $\mathcal{U}$, the situation becomes subtler. The integer-indexed Postnikov filtration fails to capture the correct equivariant structure.

### 3.1 The Integer-Indexed Stable Postnikov Tower

In the non-equivariant stable category $\mathrm{Sp}$, the Postnikov tower works as follows. Given a spectrum $E$, define $P^n E$ by attaching $\Sigma^{n+1}$-cells (cones on Moore spaces $M(\pi_{n+1}E, n+1)$) to kill $\pi_{n+1}$, then iterating. The construction is indexed by the integers: attaching $\Sigma^k$-cells for $k > n$.

Naively, one might attempt the same in $\mathrm{Sp}^G$ (genuine $G$-spectra): for each subgroup $H \leq G$, attach cells of the form $\Sigma^\infty_G G/H_+ \wedge \Sigma^{n+1} M$ to kill $\pi_{n+1}^H$. This defines an integer-indexed filtration

$$\cdots \to \tau_{\leq n} E \to \tau_{\leq n-1} E \to \cdots$$

where $\pi_k^H(\tau_{\leq n} E) = \begin{cases} \pi_k^H(E) & k \leq n \\ 0 & k > n \end{cases}$ for all $H$.

### 3.2 Non-Nullhomotopic Maps Between Representation Spheres

The integer-indexed Postnikov filtration has a fundamental deficiency: it is not compatible with the richer sphere spectrum structure of $\mathrm{Sp}^G$.

**The Key Issue.** In $\mathrm{Sp}^G$, the sphere spectrum is $S = \{S^V\}_{V \in \mathcal{U}}$ (the sphere $G$-spectrum indexed over the complete universe $\mathcal{U}$ of representations). For representations $V \subset W$, there is a canonical map $S^V \to S^W$ induced by the inclusion $V \hookrightarrow W$. *This map need not be null-homotopic in the $G$-equivariant stable category, even when $\dim W > \dim V$.*

**Example.** Let $G = C_2$ and let $\sigma$ denote the sign representation of $C_2$ (the 1-dimensional real representation where the generator acts by $-1$). The map

$$S^\sigma \longrightarrow S^{2\sigma}$$

induced by the inclusion $\sigma \hookrightarrow 2\sigma = \sigma \oplus \sigma$ is *not* null-homotopic in $\mathrm{Sp}^{C_2}$.

*Why?* Nonequivariantly, $S^\sigma \simeq S^1$ and $S^{2\sigma} \simeq S^2$, so the map is a degree-1 map $S^1 \to S^2$, which is null-homotopic. But equivariantly, one must check the $C_2$-fixed-point level: $(S^\sigma)^{C_2} = S^0$ and $(S^{2\sigma})^{C_2} = S^0$, and the induced map $(S^\sigma)^{C_2} \to (S^{2\sigma})^{C_2}$ is a degree-1 map $S^0 \to S^0$, which is not null-homotopic. Since a $G$-equivariant map is null-homotopic iff its fixed-point maps are all null-homotopic, this map is non-null.

More generally: *any map $S^V \to S^W$ with $V \subseteq W$ is non-null-homotopic at the $H$-fixed-point level precisely when $W^H \neq V^H$, i.e., when $W - V$ has $H$-fixed summands.* If $V$ contains a trivial summand, all such maps are null-homotopic (since we can use the trivial direction to contract); if $V$ has no trivial summands, non-null maps abound.

> [!DANGER] Why this breaks the integer Postnikov tower
> Suppose we build an integer-indexed Postnikov section $\tau_{\leq n} E$ by attaching integer-suspension cells. A slice cell $G_+ \wedge_H S^{m\rho_H}$ of dimension $m|H|$ might not be "seen" by any integer-suspension cell of the right dimension, because the representation sphere $S^{m\rho_H}$ has non-trivial equivariant structure that $\Sigma^{m|H|}$ (with trivial $G$-action) does not capture. The cells of the integer Postnikov tower are "too coarse" — they see the underlying non-equivariant homotopy but miss the representation-theoretic data.

### 3.3 Mackey Functors vs. Coefficient Systems

Stably, equivariant homotopy groups always form *Mackey functors*, not merely coefficient systems.

**Definition (Mackey Functor).** A *Mackey functor* $\underline{M}$ for $G$ is a pair of functors

$$\underline{M}^*: \mathcal{O}_G^{op} \to \mathbf{Ab}, \qquad \underline{M}_*: \mathcal{O}_G \to \mathbf{Ab}$$

(i.e., a contravariant part encoding restriction maps $\mathrm{res}_H^K: \underline{M}(G/K) \to \underline{M}(G/H)$ and a covariant part encoding transfer maps $\mathrm{tr}_H^K: \underline{M}(G/H) \to \underline{M}(G/K)$) satisfying:
1. $\underline{M}^*(G/H) = \underline{M}_*(G/H)$ for all $H$ (same underlying abelian groups),
2. The *Mackey double-coset formula*: for $H, K \leq G$,
$$\mathrm{res}_H^G \circ \mathrm{tr}_K^G = \sum_{[g] \in H \backslash G / K} \mathrm{tr}_{H \cap gKg^{-1}}^H \circ c_g \circ \mathrm{res}_{g^{-1}Hg \cap K}^K,$$
where $c_g$ is conjugation by $g$.

**Theorem.** For a genuine $G$-spectrum $E \in \mathrm{Sp}^G$, the homotopy groups $\underline{\pi}_n^{Mky}(E)(G/H) := \pi_n^H(E) = [G/H_+ \wedge S^n, E]_G$ assemble into a Mackey functor, with:
- *Restriction*: induced by the $G$-map $G/H_+ \to G/K_+$ (for $H \leq K$),
- *Transfer*: induced by the norm map / umkehr map in the stable category.

*The transfer maps arise from the covariant functoriality of the smash product in $\mathrm{Sp}^G$ — they have no counterpart in the unstable setting because the suspension spectrum functor $\Sigma^\infty_G$ does not preserve covariant functoriality of orbit sets.*

The integer Postnikov filtration only captures the coefficient system structure (restriction maps); it misses the transfer maps. The slice filtration, built from representation spheres, is sensitive to the full Mackey functor structure.

### 3.4 The Remedy: Representation-Indexed Filtration

The resolution, due to Hill-Hopkins-Ravenel, is to replace the integer indexing with a *representation-dimension* indexing. Instead of attaching cells $G/H_+ \wedge \Sigma^n D^1$ to kill integer-degree homotopy, one uses *slice cells* — cells built from representation spheres $S^{m\rho_H}$ — to filter by representation dimension.

The philosophy: *the "correct" notion of dimension for the slice filtration is not the integer dimension of a cell, but the representation-theoretic dimension, which counts each irreducible representation with multiplicity according to its role in the regular representation.*

---

## 4. The Slice Filtration 🔑

We now develop the slice filtration of Hill-Hopkins-Ravenel. Throughout, $G$ is a finite group and $\mathrm{Sp}^G$ denotes the category of genuine $G$-spectra.

### 4.1 Slice Cells

**Definition (Regular Representation).** For a finite group $H$, the *regular representation* $\rho_H$ is the real representation $\mathbb{R}[H]$ with $H$ acting by left multiplication. Its dimension is $|\rho_H| = |H|$.

**Definition (Slice Cells, HHR §4).** For a subgroup $H \leq G$ and integer $m \geq 0$, the *slice cells* are the $G$-spectra:

- **Even slice cell of dimension $m|H|$:**
$$\widetilde{S}^{m|H|} := G_+ \wedge_H S^{m\rho_H},$$
where $S^{m\rho_H}$ is the one-point compactification of $m\rho_H = \rho_H^{\oplus m}$, and $G_+ \wedge_H -$ is the induction from $H$ to $G$.

- **Odd slice cell of dimension $m|H| - 1$:**
$$\widetilde{S}^{m|H|-1} := \Sigma^{-1}(G_+ \wedge_H S^{m\rho_H}).$$

The *dimension* of $G_+ \wedge_H S^{m\rho_H}$ is $m|H|$ because $\dim_\mathbb{R}(m\rho_H) = m \cdot |H|$, and suspension by a representation of dimension $d$ raises underlying (non-equivariant) dimension by $d$.

> [!EXAMPLE]- Slice cells for G = C2
> Let $G = C_2$, $H = C_2$. The regular representation is $\rho_{C_2} = \mathbb{R} \oplus \sigma$ where $\sigma$ is the sign representation ($\dim \rho_{C_2} = 2$). The even slice cells are:
> - $m=0$: $G_+ \wedge_{C_2} S^0 = S^0$ (dimension 0)
> - $m=1$: $G_+ \wedge_{C_2} S^{\rho_{C_2}} = G_+ \wedge_{C_2} S^{1+\sigma}$ (dimension 2)
> - $m=2$: $G_+ \wedge_{C_2} S^{2\rho_{C_2}}$ (dimension 4)
>
> For $H = e$ (trivial subgroup), $\rho_e = \mathbb{R}$ (trivial 1-dimensional representation), $|H| = 1$. The slice cells are $G_+ \wedge_e S^m = \Sigma^\infty_G G_+ \wedge S^m$ — just the free $G$-spectrum on $S^m$. These are the "ordinary" suspension cells.
>
> *Surprisingly,* the slice cells for $H = e$ are exactly the cells that appear in the ordinary (integer-indexed) Postnikov tower. The extra structure comes from the nontrivial subgroup cells.

> [!NOTE] Why the regular representation?
> The choice of $\rho_H$ (rather than some other representation) is not arbitrary. The regular representation has the property that $\rho_H^{C_2} = 0$ for all proper subgroups $H' < H$ of $G$ that don't fix $H$ — this ensures that the slice cells behave well under restriction and induction. More precisely, the regular representation is the unique (up to multiples) representation whose character is the delta function at the identity of $H$, which makes the slice cells "concentrated" at the right level of the subgroup lattice.

### 4.2 Slice Null and Slice Positive Spectra

**Definition (Blumberg/HHR, Definition 5.2.2).** Let $X \in \mathrm{Sp}^G$.

- $X$ is *slice $n$-null* (also written: slice $\leq n-1$, or slice $< n$) if $\mathrm{Map}(\widetilde{S}, X)$ is contractible as a $G$-space for every slice cell $\widetilde{S}$ of dimension greater than $n$.

- $X$ is *slice $n$-positive* (also written: slice $> n$, or slice $\geq n+1$) if $\mathrm{Map}(\widetilde{S}, X)$ is contractible as a $G$-space for every slice cell $\widetilde{S}$ of dimension at most $n$.

Here $\mathrm{Map}(\widetilde{S}, X)$ is the $G$-equivariant mapping space (the internal hom in $\mathrm{Sp}^G$, evaluated as a $G$-space).

> [!WARNING] Contractibility is as G-spaces
> The condition is that $\mathrm{Map}(\widetilde{S}, X)$ is contractible *as a $G$-space*, meaning $\mathrm{Map}(\widetilde{S}, X)^H \simeq *$ for all closed $H \leq G$. This is stronger than just asking the underlying (non-equivariant) mapping space to be contractible. In particular, it encodes the mapping spaces out of all representations simultaneously.

Note the parallel with the non-equivariant case: a spectrum $E$ is $n$-truncated (Postnikov $\leq n$) iff $\mathrm{Map}(S^k, E) \simeq *$ for all $k > n$, and $n$-connective iff $\mathrm{Map}(S^k, E) \simeq *$ for all $k \leq n$. The slice conditions replace $\{S^k\}$ with $\{\widetilde{S}^k\}$ (slice cells).

### 4.3 Relationship to Ordinary Connectivity

**Proposition (HHR, Proposition 4.11; Blumberg Exercise 5.2.3).** Let $X$ be a $G$-spectrum. Then:

1. $X$ is slice $0$-positive if and only if $X$ is $(-1)$-connected (i.e., $\pi_k^H(X) = 0$ for $k \leq -1$ and all $H \leq G$).
2. $X$ is slice $(-1)$-positive if and only if $X$ is $(-2)$-connected.
3. $X$ is slice $0$-null if and only if $X$ is $0$-coconnected (i.e., $\pi_k^H(X) = 0$ for $k \geq 1$ and all $H \leq G$).
4. $X$ is slice $(-1)$-null if and only if $X$ is $(-1)$-coconnected.

*Proof sketch for (1).* $X$ is slice $0$-positive iff $\mathrm{Map}(\widetilde{S}, X) \simeq *$ for all slice cells of dimension $\leq 0$. The slice cells of dimension $\leq 0$ are: $\widetilde{S}^0 = G_+ \wedge_H S^0$ (the $0$-dimensional even cells, $m=0$) and $\widetilde{S}^{-1} = \Sigma^{-1}(G_+ \wedge_H S^0)$ (for $m=0$ and various $H$). Mapping out of $G_+ \wedge_H S^0$ gives $\mathrm{Map}_G(G_+ \wedge_H S^0, X) \simeq \mathrm{Map}_H(S^0, i_H^* X) \simeq X^H$, and mapping out of $\Sigma^{-1}(G_+ \wedge_H S^0)$ gives $\Omega X^H$. These are contractible iff $X^H \simeq *$ and $\pi_0(X^H) = 0$, i.e., iff $\pi_k^H(X) = 0$ for $k \leq -1$. $\square$

**This proposition shows that, for small dimensions, the slice conditions precisely recover ordinary connectivity conditions. The slice filtration is a genuine generalization of the connectivity filtration.**

> [!NOTE] The general pattern
> The connectivity statements hold in the range where slice cells are "the same" as ordinary suspension cells — i.e., for $H = e$, where $G_+ \wedge_e S^m = \Sigma^\infty_G G_+ \wedge S^m$ are just free $G$-spectra. At higher dimensions, the slice cells for nontrivial $H$ introduce additional conditions beyond ordinary connectivity.

### 4.4 Characterization via Filtrations

**Proposition (HHR, Proposition 4.15; Blumberg Proposition 5.2.4).** A $G$-spectrum $X$ is slice $n$-positive if and only if, up to weak $G$-equivalence, $X$ admits a filtration

$$X^0 \subseteq X^1 \subseteq X^2 \subseteq \cdots \subseteq X$$

such that:
- $X \simeq \operatorname{hocolim}_i X^i$,
- each cofiber $X^i / X^{i-1}$ is a wedge of slice cells of dimension $> n$.

*This is the analogue of the CW approximation theorem: slice $n$-positive spectra are exactly those built from slice cells in dimensions $> n$.*

> [!INFO] Parallels with Postnikov and CW
> Compare: in $\mathrm{Sp}$, an $n$-connective spectrum is one built from cells $\Sigma^k$ for $k > n$. The slice $n$-positivity replaces $\{\Sigma^k : k > n\}$ with $\{\widetilde{S}^k : k > n\}$. Since slice cells include the ordinary suspension cells (via $H = e$), slice $n$-positivity is in general a stronger condition than $n$-connectivity.

### 4.5 The Slice Sections and the Slice Tower

**Definition (Slice Section $P^n X$).** For $X \in \mathrm{Sp}^G$, the *$n$-th slice section* $P^n X$ is the *localization* of $X$ with respect to the class of slice $n$-null spectra. Concretely:

- There is a natural map $X \to P^n X$,
- $P^n X$ is slice $n$-null,
- The homotopy fiber $P_n^{n+1} X := \mathrm{fib}(X \to P^n X)$ is slice $n$-positive.

The existence of $P^n X$ follows from Bousfield localization theory applied in $\mathrm{Sp}^G$; the class of slice $n$-null spectra is a localizing subcategory (closed under arbitrary homotopy colimits and retracts).

> [!WARNING] Subtlety in the construction
> The naive attempt — kill all mapping spaces from slice cells of dimension $> n$ — runs into the same issue as Section 1.2: one must iterate. An additional subtlety in the equivariant case is the existence of nontrivial maps $S^V \to S^W$ for $V \subset W$ representations. *If $V$ contains a trivial summand, all such maps are null-homotopic* (one can deform via the trivial direction), but if $V$ has no trivial summands, these maps are genuinely nontrivial and must be handled separately in the small object argument. This is the precise obstruction identified in Section 3.2, and it forces the use of representation spheres rather than integer spheres in the construction.

**Definition (Slice Tower).** The *slice tower* of $X$ is the sequence of maps

$$\cdots \longrightarrow P^{n+1} X \longrightarrow P^n X \longrightarrow P^{n-1} X \longrightarrow \cdots$$

obtained from the natural maps $X \to P^n X$ and the functoriality of localization. The maps $P^{n+1} X \to P^n X$ arise because any slice $n$-null spectrum is also slice $(n+1)$-null (trivially: if all mapping spaces from cells of dimension $> n$ are contractible, then a fortiori all mapping spaces from cells of dimension $> n+1$ are contractible).

**Theorem.** $X \simeq \operatorname{holim}_n P^n X$.

*This is the convergence theorem for the slice tower, analogous to $X \simeq \operatorname{holim}_n P_n X$ for the Postnikov tower. The proof uses the fact that $X$ is built from slice cells (by Proposition 4.4) and that the localization maps are compatible with the filtration.*

### 4.6 The n-Slice and an Explicit Example

**Definition (n-Slice).** The *$n$-slice* of $X$ is

$$P_n^n X := \mathrm{fib}(P^n X \to P^{n-1} X).$$

This is the "associated graded piece" of the slice tower at level $n$: it captures the "purely dimension-$n$" part of $X$.

**Proposition (Blumberg Exercise 5.2.5; HHR Proposition 4.20).** The $(-1)$-slice of $X$ is

$$P_{-1}^{-1} X \simeq \Sigma^{-1} H\underline{\pi}_{-1}(X),$$

where $\underline{\pi}_{-1}(X)$ is the $(-1)$-st homotopy group of $X$ viewed as a Mackey functor (with $\underline{\pi}_{-1}(X)(G/H) = \pi_{-1}^H(X)$, with restriction and transfer maps), and $H\underline{M}$ is the *Eilenberg-Mac Lane $G$-spectrum* for a Mackey functor $\underline{M}$.

*Proof sketch.* From the long exact sequence on homotopy Mackey functors associated to the fibration $P_{-1}^{-1} X \to P^{-1} X \to P^{-2} X$:
- $P^{-1} X$ is slice $(-1)$-null, so $\pi_k^H(P^{-1} X) = 0$ for $k \geq 0$ (by Proposition 4.3, part 3).
- $P^{-2} X$ is slice $(-2)$-null, so $\pi_k^H(P^{-2} X) = 0$ for $k \geq -1$.
- From the long exact sequence, $\pi_k^H(P_{-1}^{-1} X) \cong \begin{cases} \pi_{-1}^H(X) & k = -1 \\ 0 & k \neq -1 \end{cases}$.

A $G$-spectrum with exactly one nonzero homotopy Mackey functor (in degree $-1$) is, by the equivariant Whitehead theorem, equivalent to $\Sigma^{-1} H\underline{\pi}_{-1}(X)$, where $H\underline{M}$ denotes the Eilenberg-Mac Lane spectrum for the Mackey functor $\underline{M}$. $\square$

> [!NOTE] Eilenberg-Mac Lane G-spectra
> For a Mackey functor $\underline{M}$, the Eilenberg-Mac Lane $G$-spectrum $H\underline{M}$ is the unique (up to weak equivalence) $G$-spectrum with $\underline{\pi}_0(H\underline{M}) \cong \underline{M}$ and $\underline{\pi}_k(H\underline{M}) = 0$ for $k \neq 0$. It represents *Bredon cohomology with Mackey functor coefficients*: $[X, H\underline{M}]_G \cong H_G^0(X; \underline{M})$. The stable Eilenberg-Mac Lane spectra extend the unstable $K(\underline{M}, n)$ to the full stable category and encode both restriction and transfer maps.

> [!EXAMPLE] The 0-slice
> For a connective $G$-spectrum $X$, the $0$-slice is $P_0^0 X \simeq H\underline{\pi}_0(X)$ — the Eilenberg-Mac Lane spectrum on the $0$-th homotopy Mackey functor. This is the exact stable analogue of the fact that the $0$-th Postnikov section $P_0 X$ is a $K(\pi_0 X, 0)$.

### 4.7 Behavior Under Change Functors

**Proposition (HHR).** The slice cells, and hence the slice filtration, are preserved by the following change-of-group functors:
- *Restriction*: $i_K^*: \mathrm{Sp}^G \to \mathrm{Sp}^K$ (restrict to a subgroup $K \leq G$),
- *Induction*: $G_+ \wedge_K -: \mathrm{Sp}^K \to \mathrm{Sp}^G$ (induce from $K$ to $G$),
- *Norm*: $N_K^G: \mathrm{Sp}^K \to \mathrm{Sp}^G$ (the Hill-Hopkins-Ravenel norm).

*Proof sketch for restriction.* A slice cell $\widetilde{S} = G_+ \wedge_H S^{m\rho_H}$ restricts under $i_K^*$ to $i_K^*(G_+ \wedge_H S^{m\rho_H}) \cong (G/H)_+ \wedge_\emptyset S^{m\rho_H|_K}$. By Mackey's double coset formula, $(i_K^* G_+ \wedge_H -) \cong \bigvee_{[g] \in K\backslash G/H} K_+ \wedge_{K \cap gHg^{-1}} (-)^{c_g}$. Each term $K_+ \wedge_{K \cap gHg^{-1}} S^{m \cdot g \cdot \rho_H|_{K \cap gHg^{-1}}}$ is a $K$-slice cell (since $g \cdot \rho_H|_{K \cap gHg^{-1}}$ contains the regular representation of $K \cap gHg^{-1}$ as a summand). $\square$

**The compatibility of the slice filtration with all three change functors is crucial for HHR's proof: it allows slice computations to be moved between groups and norms to be taken without losing the slice structure.**

> [!TIP]- Intuition for norm compatibility
> The norm $N_K^G$ is the equivariant analogue of the tensor product (not the direct sum). For the slice filtration to be compatible with norms, one needs the dimensions to multiply correctly: a $K$-spectrum of slice dimension $d$ has norm of slice dimension $[G:K] \cdot d$ (since the norm raises the "representation dimension" by the index). This multiplicativity is exactly what the regular representation provides: $\rho_{G|_K}^{\otimes [G:K]}$ contributes to $\rho_G$, matching dimensions.

---

## 5. The Slice Spectral Sequence 📐

The slice tower, like any filtered object in a stable $\infty$-category, gives rise to a spectral sequence. This is the *slice spectral sequence*, the equivariant analogue of the Atiyah-Hirzebruch spectral sequence.

### 5.1 Construction from the Slice Tower

Let $X \in \mathrm{Sp}^G$ and fix a subgroup $H \leq G$. The slice tower

$$\cdots \to P^{n+1} X \to P^n X \to P^{n-1} X \to \cdots$$

is a filtration of $X$ in $\mathrm{Sp}^G$. Applying $\pi_*^H(-)$ and using the long exact sequences associated to the fiber sequences

$$P_n^n X \longrightarrow P^n X \longrightarrow P^{n-1} X,$$

we obtain an exact couple, hence a spectral sequence.

**Definition (Slice Spectral Sequence).** The *slice spectral sequence* for $X$ is the spectral sequence with:

$$E_1^{n,*} = \pi_*^G(P_n^n X) \Implies \pi_*^G(X),$$

where $P_n^n X$ is the $n$-slice of $X$. The differentials $d_r: E_r^{n,*} \to E_r^{n-r, *+r-1}$ are induced by the attaching maps in the slice tower.

More precisely, since the slices $P_n^n X$ are Eilenberg-Mac Lane-type spectra (built from $H\underline{M}$'s), the $E_2$-page can often be computed in terms of Bredon cohomology of the slices. One writes

$$E_2^{p,q} = \pi_p^G(P_q^q X) \Implies \pi_{p+q}^G(X).$$

> [!WARNING] RO(G)-grading
> For genuine $G$-spectra, the homotopy groups are *$RO(G)$-graded*: $\pi_\alpha^H(X) = [S^\alpha, X]_G$ for $\alpha \in RO(G)$ (the real representation ring of $G$). The slice spectral sequence is naturally $RO(G)$-graded, and its full strength requires tracking contributions from all representation spheres. The integer-graded version (collapsing $RO(G)$ to $\mathbb{Z}$) is a shadow of the full spectral sequence.

### 5.2 Dugger's C2 Construction and KR-Theory

The first instance of the slice spectral sequence was constructed by Dugger for $G = C_2$ and Atiyah's *KR-theory*.

**KR-theory** is the $C_2$-equivariant cohomology theory $KR^{*,*}(X)$ defined by Atiyah, with $C_2$ acting by complex conjugation. Non-equivariantly, $KR^*(X) \cong K^*(X)$; at the $C_2$-fixed level, $KR^*(X^{C_2}) \cong KO^*(X^{C_2})$ (real K-theory).

Dugger's motivation was the deep analogy between $C_2$-equivariant homotopy theory and *motivic homotopy theory* over $\mathbb{R}$: both have a "weight" grading (corresponding to the sign representation $\sigma$ for $C_2$, or the Tate twist for motivic) in addition to the topological grading.

**Theorem (Dugger).** There is a conditionally convergent spectral sequence

$$E_2^{p,q} = H^{p,\, r-q/2}(X;\, \underline{\mathbb{Z}}) \Implies KR^{p+q,\, r}(X),$$

where $X$ is a $C_2$-space, $\underline{\mathbb{Z}}$ is the constant Mackey functor valued in $\mathbb{Z}$, and $KR^{p+q, r}$ is the $(p+q)$-th $KR$-group with motivic weight $r$. The left-hand side uses bigraded cohomology reflecting the $C_2$-equivariant structure.

> [!INFO] The motivic analogy
> Dugger's construction was explicitly motivated by the analogous spectral sequence in motivic homotopy theory:
> $$E_2^{p,q} = H^{p,q}(X; \mathbb{Z}) \Implies K^{p+q}(X)_\mathbb{Z}^{\wedge}$$
> (the motivic AHSS for algebraic K-theory). The $C_2$-equivariant slice spectral sequence is the "topological realization" of the motivic one, via the Betti realization functor sending motivic spectra over $\mathbb{R}$ to $C_2$-equivariant spectra.

### 5.3 The General HHR Slice Spectral Sequence

The full generality of the slice spectral sequence, for $G = C_{2^n}$ and arbitrary $G$-spectra, was developed by Hill-Hopkins-Ravenel in their proof of the Kervaire invariant one theorem.

**Theorem (HHR).** For $G = C_{2^n}$ and $X \in \mathrm{Sp}^G$, the slice tower gives a conditionally convergent spectral sequence

$$E_2^{p,q} = \pi_p^G(P_q^q X) \Implies \pi_{p+q}^G(X),$$

which is the *slice spectral sequence* for $X$.

The key properties established by HHR:
1. *Convergence*: the tower converges conditionally (and strongly under finiteness hypotheses).
2. *$E_2$-page identification*: the slices $P_q^q X$ are Eilenberg-Mac Lane spectra (wedges of $H\underline{M}$'s) in favorable cases, making the $E_2$-page computable in terms of Mackey functor cohomology.
3. *Multiplicative structure*: when $X$ is a commutative $G$-ring spectrum, the slice spectral sequence is multiplicative (a spectral sequence of rings).

**The key feature distinguishing the slice spectral sequence from the ordinary AHSS is its sensitivity to the $RO(G)$-graded homotopy groups.** Non-equivariantly (with trivial $G$-action), the slice tower and integer Postnikov tower coincide, and the spectral sequence reduces to the ordinary AHSS:

$$E_2^{p,q} = H^p(X; \pi_q E) \Implies E^{p+q}(X).$$

---

## 6. Comparison and the HHR Application 🔑

### 6.1 The Three-Way Comparison Table

The following table compares the three tiers of Postnikov/slice theory.

| Feature | Non-equivariant $\mathrm{Sp}$ | Equivariant unstable $G\mathbf{Top}$ | Equivariant stable $\mathrm{Sp}^G$ |
|---|---|---|---|
| Cells | $S^n$ (integer spheres) | $G/H_+ \times S^n$ (orbit cells) | $G_+ \wedge_H S^{m\rho_H}$ (slice cells) |
| Dimension indexing | Integer $n$ | Integer $n$ | Representation dim $m|H|$ |
| Filtration sections | $P_n X$ (Postnikov) | $P_n^G X$ (equivariant Postnikov) | $P^n X$ (slice section) |
| Fiber / slice | $K(\pi_n X, n)$ | $K(\underline{\pi}_n X, n)$ | $P_n^n X \simeq \Sigma^n H\underline{M}$ |
| Coefficients | $\pi_n(X) \in \mathbf{Ab}$ | $\underline{\pi}_n X$ (coeff. system) | $\underline{\pi}_n^{Mky}(X)$ (Mackey functor) |
| k-invariants | $H^{n+2}(-; \pi_n X)$ | $H_G^{n+2}(-; \underline{\pi}_n X)$ (Bredon) | Mackey functor cohomology |
| Spectral sequence | — | Bredon AHSS | Slice SS |
| $E_2$-page | $H^p(X; \pi_q E)$ | $H_G^p(X; \underline{\pi}_q E)$ | $\pi_p^G(P_q^q X)$ |
| Construction tool | Small object argument | Elmendorf + objectwise | Bousfield localization |
| Key theorem | Whitehead | Elmendorf's theorem | HHR (Prop. 4.11, 4.15, 4.20) |

> [!INFO] Summary of the upgrade chain
> As one passes from non-equivariant to equivariant unstable to equivariant stable, each layer adds structure:
> - Non-equivariant → equivariant unstable: coefficients upgrade from abelian groups to coefficient systems (add restriction maps); k-invariants upgrade from ordinary to Bredon cohomology.
> - Equivariant unstable → equivariant stable: coefficients upgrade from coefficient systems to Mackey functors (add transfer maps); cells upgrade from orbit cells to slice cells (add representation spheres); the indexing upgrades from integers to representation dimensions.

### 6.2 The Slices of the Real Cobordism Spectrum

The central computation in HHR's proof of the Kervaire invariant one theorem is the identification of the slices of the *Real cobordism spectrum* $MU_\mathbb{R}$.

**Construction.** The Real cobordism spectrum $MU_\mathbb{R}$ is a $C_2$-equivariant spectrum; it is the Thom spectrum of the $C_2$-equivariant complex cobordism, with $C_2$ acting by complex conjugation. The full $C_{2^n}$-equivariant spectrum used by HHR is the *norm*:

$$MU^{((C_{2^n}))} := N_{C_2}^{C_{2^n}} MU_\mathbb{R},$$

the Hill-Hopkins-Ravenel norm of $MU_\mathbb{R}$ from $C_2$ to $C_{2^n}$.

**Theorem (HHR, Theorem 6.1).** The slices of $MU^{((C_{2^n}))}$ are:

$$P_{2k}^{2k} MU^{((C_{2^n}))} \simeq \bigvee H\underline{\mathbb{Z}} \wedge S^{k\rho_G},$$

where the wedge is over an explicit finite set of representation spheres, and $H\underline{\mathbb{Z}}$ is the Eilenberg-Mac Lane $G$-spectrum for the constant Mackey functor $\underline{\mathbb{Z}}$. Odd slices vanish.

*This explicit identification is the "hard part" of the HHR proof.* Once the slices are known, the slice spectral sequence converges to $\pi_*^G(MU^{((C_{2^n}))})$, and the differentials can be analyzed to prove the vanishing of Kervaire invariant one elements $\theta_j$ for $j \geq 7$.

> [!QUESTION] Open: the general slice computation
> The full slice structure of $MU^{((G))}$ for groups $G$ beyond $C_{2^n}$ is not known in general. The slice machinery extends to all finite groups $G$ (and more generally to compact Lie groups), but the explicit slice computations depend on detailed knowledge of the representation theory of $G$. This remains an active area of research.

> [!INFO] Historical note: Dugger's priority
> The first piece of the slice spectral sequence was worked out by Dugger for $C_2$. His motivation was purely the $C_2$-motivic analogy: he sought a spectral sequence converging to $KR$-theory whose $E_2$-page was motivic cohomology. HHR recognized that this construction generalized to a filtration of all $G$-spectra indexed by representation dimension, and that the resulting spectral sequence for $MU^{((C_{2^n}))}$ would be powerful enough to settle the Kervaire invariant problem.

---

## References

| Reference Name | Brief Summary | Link to Reference |
|---|---|---|
| [M392C Lecture Notes (Blumberg, notes by Debray)](https://adebray.github.io/lecture_notes/m392c_EHT_notes.pdf) | Primary source; §1.3 for equivariant Postnikov towers, §5.2 for the full slice filtration and slice spectral sequence | [https://adebray.github.io/lecture_notes/m392c_EHT_notes.pdf](https://adebray.github.io/lecture_notes/m392c_EHT_notes.pdf) |
| [On the Non-Existence of Elements of Kervaire Invariant One (Hill, Hopkins, Ravenel)](https://arxiv.org/abs/0908.3724) | Original slice filtration paper; Propositions 4.11, 4.15, 4.20 are the foundational results on slice null/positive and the slice tower | [https://arxiv.org/abs/0908.3724](https://arxiv.org/abs/0908.3724) |
| [The Equivariant Slice Filtration: A Primer (Hill)](https://arxiv.org/abs/1107.3582) | Best expository introduction to slice cells and the slice tower; highly recommended as a companion to HHR | [https://arxiv.org/abs/1107.3582](https://arxiv.org/abs/1107.3582) |
| [Equivariant Homotopy and Cohomology Theory (May et al.)](https://www.math.uchicago.edu/~may/BOOKS/alaska.pdf) | Classical treatment of equivariant Postnikov towers, coefficient systems, Bredon cohomology, and equivariant Eilenberg-Mac Lane spaces | [https://www.math.uchicago.edu/~may/BOOKS/alaska.pdf](https://www.math.uchicago.edu/~may/BOOKS/alaska.pdf) |
| [Equivariant Stable Homotopy Theory — Handbook (Greenlees, May)](https://www.math.uchicago.edu/~may/PAPERS/Newthird.pdf) | Survey covering Mackey functors, RO(G)-grading, the stable equivariant category, and change-of-group functors | [https://www.math.uchicago.edu/~may/PAPERS/Newthird.pdf](https://www.math.uchicago.edu/~may/PAPERS/Newthird.pdf) |
| [Equivariant Cohomology Theories (Bredon)](https://link.springer.com/book/10.1007/BFb0082690) | Original definition of Bredon cohomology and coefficient systems; foundational for the equivariant Postnikov theory | [https://link.springer.com/book/10.1007/BFb0082690](https://link.springer.com/book/10.1007/BFb0082690) |
