# Thermodynamic Semirings and Entropy Algebras

## Sources

| Source | Type | Key Contribution | Link |
|--------|------|-----------------|------|
| [Marcolli & Thorngren (2011)](https://arxiv.org/abs/1108.2874) | paper | Thermodynamic semirings: entropy emerges as a first-order deformation of the tropical semiring | arXiv:1108.2874 |
| [Marcolli & Tedeschi (2014)](https://arxiv.org/abs/1412.0247) | paper | Entropy algebras and Birkhoff factorization in Connes-Kreimer Hopf algebras of rooted trees | arXiv:1412.0247 |
| [Marcolli (2018)](https://arxiv.org/abs/1807.05314) | paper | Gamma spaces and information loss; homotopy-theoretic / stable-homotopy perspective on entropy functors | arXiv:1807.05314 |
| Marcolli, Ma148b Winter 2025 | course | Comprehensive treatment including thermodynamic and algebraic approaches | [course page](https://www.its.caltech.edu/~matilde/Ma148bWinter2025.html) |
| Marcolli, Ma148a Fall 2021 | course | Thermodynamic semirings and entropy algebras as lecture topics | [course page](https://www.its.caltech.edu/~matilde/Ma148aFall2021.html) |

---

## Context and Motivation

The cohomological and operadic approaches (see [[research/categorical-entropy|Categorical Entropy]]) characterize entropy via universal properties in categories of probability spaces. This thread asks a different question: can entropy be *constructed* as the output of a canonical algebraic operation, rather than characterized axiomatically?

The answer from Marcolli-Thorngren is yes: entropy is the **first-order deformation of tropical arithmetic** at zero temperature. The tropical semiring $(\mathbb{R}, \min, +)$ governs zero-temperature statistical mechanics; raising the temperature deforms $\min$ into log-sum-exp, and the derivative of this deformation with respect to temperature is Shannon entropy.

This places entropy inside *algebraic geometry* (Witt vectors, deformation theory) and *renormalization theory* (Connes-Kreimer Hopf algebras, Birkhoff factorization) rather than in sheaf theory or operadic algebra.

---

## Thermodynamic Semirings

### The tropical semiring and its deformation

A *semiring* $(R, \oplus, \otimes)$ satisfies all ring axioms except the existence of additive inverses. The **tropical semiring** is

$$(\mathbb{R} \cup \{+\infty\},\ \min,\ +),$$

which encodes zero-temperature statistical mechanics: the "sum" of two energies $a, b$ is the energy of the ground state $\min(a, b)$.

At finite inverse temperature $\beta > 0$, the ground-state selection is softened into the **log-sum-exp** operation:

$$a \oplus_\beta b := -\frac{1}{\beta} \log\bigl(e^{-\beta a} + e^{-\beta b}\bigr).$$

As $\beta \to \infty$ (zero temperature), $a \oplus_\beta b \to \min(a, b)$, recovering the tropical semiring. As $\beta \to 0$ (infinite temperature), $a \oplus_\beta b \to -\frac{1}{\beta}\log 2 + \frac{a+b}{2}$, which diverges — the high-temperature limit degenerates.

### Entropy as a deformation derivative

🔑 **The key observation of Marcolli-Thorngren:** expand $\oplus_\beta$ as a perturbation of the tropical $\oplus_\infty = \min$ as $\beta^{-1} \to 0^+$. Setting $T = \beta^{-1}$:

$$a \oplus_T b = \min(a, b) - T \cdot H_2\!\left(\frac{e^{-a/T}}{e^{-a/T} + e^{-b/T}}\right) + O(T^2),$$

where $H_2(p) = -p\log p - (1-p)\log(1-p)$ is the *binary entropy function*. **Shannon entropy is the coefficient of the first-order temperature correction to tropical addition.**

More generally, for a distribution $p = (p_1, \ldots, p_n)$ at inverse temperature $\beta$, the free energy

$$F_\beta(p) = -\frac{1}{\beta} \log \sum_i e^{-\beta E_i}$$

satisfies $\partial F_\beta / \partial T\big|_{T=0} = H(p_\beta)$ where $p_\beta$ is the Boltzmann distribution. Entropy appears as the thermodynamic derivative $\partial F / \partial T$ — which is its classical definition. Marcolli-Thorngren observe this is not just a physics coincidence but an algebraic statement about the deformation theory of semirings.

### Witt vectors and the algebraic framework

The family $\{(\mathbb{R}, \oplus_\beta, +)\}_{\beta > 0}$ forms a one-parameter family of semirings. The algebraic object encoding all of these simultaneously is a **Witt vector semiring**: a lift of the tropical semiring from characteristic $\infty$ (the tropical limit) to finite characteristic (finite temperature).

In the classical Witt vectors construction, one lifts a ring $R$ of characteristic $p$ to a characteristic-zero ring $W(R)$ carrying a Frobenius lift. Here, the analogous construction replaces:

| Classical | Thermodynamic |
|-----------|---------------|
| Characteristic $p$ | Inverse temperature $\beta = \infty$ (tropical) |
| Characteristic $0$ | Finite temperature $\beta < \infty$ |
| Frobenius lift | Temperature deformation $\oplus_\beta$ |
| Teichmüller representatives | Boltzmann weights $e^{-\beta E_i}$ |

**Key structure:** The entropy axioms — positivity, symmetry, chain rule — correspond to the axioms of a *semiring homomorphism* from the Witt vector semiring $W_\beta(\mathbb{R})$ to $(\mathbb{R}, +, \cdot)$. This is the Marcolli-Thorngren analogue of the Baudot-Bennequin cocycle condition: entropy is "flat" in the Witt vector sense.

> [!EXAMPLE] Rényi entropy from $q$-deformation
> Setting $\beta = 1/(1-q)$ and using the $q$-logarithm $\log_q(x) = (x^{1-q} - 1)/(1-q)$ in place of $\log$ gives the **Rényi entropy**
> $$H_q(p) = \frac{1}{1-q} \log \sum_i p_i^q.$$
> Different entropy families correspond to different choices of *deformation parameter* — they are different points in the moduli space of thermodynamic semirings.

> [!QUESTION] What is the moduli space of thermodynamic semirings?
> The parameter $\beta$ (or equivalently $q$ for Rényi) suggests a one-dimensional moduli space. But Vigneaux's work on coefficient modules (see [[research/categorical-entropy|Categorical Entropy]]) suggests the space of entropy families is richer. Is there a precise algebraic-geometric description of the moduli space of "entropy-producing" semiring deformations?

---

## Entropy Algebras and Birkhoff Factorization

### The Connes-Kreimer Hopf algebra

In perturbative QFT, *renormalization* is the systematic removal of UV divergences from Feynman diagram integrals. Connes and Kreimer showed this process is controlled by a **Hopf algebra $\mathcal{H}_{CK}$** of rooted forests, where:

- **Multiplication:** disjoint union of forests
- **Comultiplication:** $\Delta(T) = \sum_{C \subseteq T} P_C(T) \otimes R_C(T)$, summing over *admissible cuts* $C$ of the tree $T$, where $P_C(T)$ is the "pruned" part above the cut and $R_C(T)$ is the remaining trunk.
- **Antipode:** encodes the BPHZ counterterms

The key theorem is that divergences in QFT are organized by the combinatorics of nested subgraphs, and the Hopf algebra captures this nesting structure algebraically.

### Entropy as a Hopf algebra character

Marcolli and Tedeschi (2014) transplant this structure to information theory. A **character** of $\mathcal{H}_{CK}$ is an algebra homomorphism $\phi: \mathcal{H}_{CK} \to \mathbb{C}$. Their main observation:

**Shannon entropy is a character of $\mathcal{H}_{CK}$.** Explicitly, given a rooted probability tree $T$ encoding a hierarchical probability model — with each internal node $v$ carrying a conditional distribution $(p_1^v, \ldots, p_{k_v}^v)$ over its children — define

$$\phi_H(T) := \sum_{v \in T} H(p_1^v, \ldots, p_{k_v}^v).$$

This assigns to each tree the *total entropy of all conditional distributions* in the hierarchy. The claim is that $\phi_H$ is multiplicative with respect to the Hopf algebra structure — i.e., a character.

> [!WARNING] Caveat: the precise dictionary
> The Connes-Kreimer comultiplication cuts trees at admissible cuts (roughly, cuts that don't separate a node from all its ancestors). The precise statement that $\phi_H$ is a character requires checking that this multiplicativity holds across all admissible cuts — which is essentially the *chain rule for conditional entropy* applied recursively. The details of this correspondence are worked out in Marcolli-Tedeschi but the conceptual content is: *the chain rule is the Hopf algebra multiplicativity condition*.

### Birkhoff factorization of entropy

A central theorem of the Connes-Kreimer theory is **Birkhoff factorization**: every character $\phi: \mathcal{H}_{CK} \to \mathbb{C}$ admits a unique decomposition

$$\phi = \phi_- \star \phi_+$$

where $\star$ is convolution in the Hopf algebra, $\phi_+$ is a "renormalized" character (finite, no poles), and $\phi_-$ is the counterterm character (encodes the divergences being removed). In QFT, $\phi_-$ is the BPHZ counterterm and $\phi_+$ is the renormalized Feynman amplitude.

Applied to $\phi_H$: the Birkhoff factorization of the entropy character extracts, from a hierarchical probability model, a canonical "renormalized" entropy $(\phi_H)_+$ and a "counterterm" $(\phi_H)_-$.

The physical interpretation: in a tree of conditional distributions, there is redundant entropic content (correlations that appear at multiple levels of the hierarchy). Birkhoff factorization *removes this double-counting* canonically — just as renormalization removes divergences from nested Feynman diagrams.

> [!QUESTION] What does the entropy counterterm $(\phi_H)_-$ measure?
> In QFT, $\phi_-$ measures the divergences — the "bad" part. In the entropy setting, $(\phi_H)_-$ measures the redundant or shared information across levels of the hierarchy. Is there a clean information-theoretic interpretation? Could this be related to *common information* (Gács-Körner) or *synergistic information* in the partial information decomposition literature?

---

## Gamma Spaces and Stable Homotopy

### Segal's $\Gamma$-spaces

Marcolli (2018) lifts the entropy story to stable homotopy theory. Recall: a **$\Gamma$-space** (Segal 1974) is a functor

$$X : \Gamma^{op} \longrightarrow \mathbf{Top}$$

where $\Gamma$ is the category of finite pointed sets $\{0, 1, \ldots, n\}$ with basepoint $0$. $\Gamma$-spaces model *$E_\infty$ spaces* — spaces with a multiplication that is commutative and associative up to all higher coherences. Segal's theorem: every $\Gamma$-space with a group-completion gives a connective spectrum (an object of stable homotopy theory).

The slogan: $\Gamma$-spaces are the correct homotopy-theoretic notion of "commutative monoid."

### Information loss as a $\Gamma$-space

The claim of Marcolli (2018) is that the information loss functor

$$F : \mathbf{FinProb} \to (\mathbb{R}_{\geq 0}, +)$$

naturally extends to a $\Gamma$-space structure. The rough idea:

- The category $\mathbf{FinProb}$ of finite probability spaces carries a symmetric monoidal structure (product of probability spaces).
- A symmetric monoidal functor to an abelian group is the data of a $\Gamma$-space.
- The BFL information loss functor is (essentially) symmetric monoidal, hence defines a $\Gamma$-space.

The associated spectrum $\mathbf{I}$ has:
- $\pi_0(\mathbf{I})$ = the group completion of $(\mathbb{R}_{\geq 0}, +)$, i.e. $\mathbb{R}$
- Higher $\pi_n(\mathbf{I})$ = (conjecturally) higher information-theoretic invariants

This is the most structurally ambitious claim in this thread: **entropy as a map of spectra** in the sense of stable homotopy theory.

> [!QUESTION] What are the higher homotopy groups of the entropy spectrum?
> If $\pi_0 = \mathbb{R}$ (classical entropy) and $\pi_1$ = (some secondary entropy invariant), what does $\pi_n$ compute for $n \geq 2$? Candidates: higher mutual informations, topological complexity of probability distributions, or K-theoretic invariants of coding schemes.

> [!QUESTION] Entropy and algebraic K-theory
> Waldhausen's algebraic K-theory $A(X)$ of a space $X$ is built from $\Gamma$-spaces associated to categories of retractive spaces. Is there a precise functor from $\mathbf{FinProb}$ to Waldhausen's input data such that the resulting K-theory spectrum has $\pi_0 = \mathbb{R}$ (entropy) and higher groups classifying information-theoretic structure? The work of Baas-Dundas-Richter-Rognes on 2-vector bundles might be relevant here.

---

## Open Questions

> [!QUESTION] 1. Moduli of entropy families
> The deformation parameter $\beta$ (or $q$) parametrizes the Rényi/Tsallis family of entropies. Is there an algebraic-geometric moduli space of "thermodynamic semiring deformations" whose points correspond to entropy families, and whose tangent space at $\beta = \infty$ is one-dimensional (spanned by Shannon entropy)?

> [!QUESTION] 2. Information-theoretic counterterms
> What does the Birkhoff factorization counterterm $(\phi_H)_-$ measure for the entropy character? Is it related to redundancy, common information, or synergy in information-theoretic terms?

> [!QUESTION] 3. The unification question
> Is there a single framework containing both the cohomological (Baudot-Bennequin) and thermodynamic (Marcolli-Thorngren) approaches? Candidate: the *derived category* of sheaves on the probability simplex, with the semiring structure arising from a monoidal structure. The de Rham theorem analogy from [[research/categorical-entropy|Categorical Entropy]] would then be a statement about different resolutions of the same derived object.

> [!QUESTION] 4. What distinguishes $\alpha = 1$ thermodynamically?
> Shannon entropy is singled out by linear scaling under mixing (BFL) and corresponds to $\beta$-derivative at $\beta = \infty$. From a physics perspective, $\alpha = 1$ corresponds to *extensive* systems (entropy scales with system size). Is there a categorical or algebraic explanation — within the semiring framework — for why extensivity selects $\alpha = 1$?
