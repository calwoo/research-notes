# Self-Organized Criticality: Exercises

## Table of Contents

- [[#Mathematical Development|Mathematical Development]]
  - [[#Problem 1 Abelian Property for a Two-Site System|Problem 1: Abelian Property for a Two-Site System]]
  - [[#Problem 2 Toppling Lemma via Monotonicity|Problem 2: Toppling Lemma via Monotonicity]]
  - [[#Problem 3 Positive Definiteness and Invertibility of the Toppling Matrix|Problem 3: Positive Definiteness and Invertibility of the Toppling Matrix]]
  - [[#Problem 4 Mean Avalanche Size from Finite-Size Scaling|Problem 4: Mean Avalanche Size from Finite-Size Scaling]]
  - [[#Problem 5 Spectral Exponent from Avalanche Duration Distribution|Problem 5: Spectral Exponent from Avalanche Duration Distribution]]
  - [[#Problem 6 SOC vs. Tuned Criticality|Problem 6: SOC vs. Tuned Criticality]]
  - [[#Problem 7 The Attractor Argument and Its Prerequisites|Problem 7: The Attractor Argument and Its Prerequisites]]
  - [[#Problem 8 Interpreting the Finite-Size Scaling Function|Problem 8: Interpreting the Finite-Size Scaling Function]]
  - [[#Problem 9 Universality Classes and Model Symmetries|Problem 9: Universality Classes and Model Symmetries]]
  - [[#Problem 10 Empirical Power Laws and the SOC Attribution Problem|Problem 10: Empirical Power Laws and the SOC Attribution Problem]]
  - [[#Problem 11 Recurrent vs. Transient Configurations and det of Delta|Problem 11: Recurrent vs. Transient Configurations and det of Delta]]
  - [[#Problem 12 The Sandpile Group is Abelian|Problem 12: The Sandpile Group is Abelian]]
  - [[#Problem 13 Green's Function Symmetry|Problem 13: Green's Function Symmetry]]
  - [[#Problem 14 Power-Law Tail Conditions and Moment Existence|Problem 14: Power-Law Tail Conditions and Moment Existence]]
  - [[#Problem 15 Upper Critical Dimension via Dimensional Analysis|Problem 15: Upper Critical Dimension via Dimensional Analysis]]
  - [[#Problem 16 Fractal Dimension of the Avalanche Boundary|Problem 16: Fractal Dimension of the Avalanche Boundary]]
  - [[#Problem 17 Eigenvalue Bounds on the Spectral Gap of Delta|Problem 17: Eigenvalue Bounds on the Spectral Gap of Delta]]
  - [[#Problem 18 Gutenberg-Richter Law as a Power-Law Derivation|Problem 18: Gutenberg-Richter Law as a Power-Law Derivation]]
- [[#Algorithmic Applications|Algorithmic Applications]]
  - [[#Problem 19 BTW Sandpile Simulation|Problem 19: BTW Sandpile Simulation]]
  - [[#Problem 20 Dhar's Burning Algorithm|Problem 20: Dhar's Burning Algorithm]]
  - [[#Problem 21 Scaling Exponent Estimation via Data Collapse|Problem 21: Scaling Exponent Estimation via Data Collapse]]
  - [[#Problem 22 Sandpile Group Enumeration on Small Graphs|Problem 22: Sandpile Group Enumeration on Small Graphs]]
  - [[#Problem 23 Branching Process Simulation for Mean-Field SOC|Problem 23: Branching Process Simulation for Mean-Field SOC]]

---

## Mathematical Development

### Problem 1: Abelian Property for a Two-Site System

*This problem establishes the Abelian property concretely on the smallest non-trivial sandpile graph, verifying that the order of toppling is immaterial and that the toppling vector is uniquely determined by the initial configuration and the toppling matrix.*

> **Prerequisites:** cf. note [[note#4.2 The Abelian Property|§4.2 — The Abelian Property]]

Consider an Abelian sandpile on the graph with two non-sink vertices $\{1, 2\}$ connected by a single edge, with both vertices also connected to a sink $s$. Vertex 1 has degree 2 (one edge to vertex 2, one edge to $s$) and vertex 2 has degree 2 (one edge to vertex 1, one edge to $s$).

(a) Write down the $2 \times 2$ toppling matrix $\Delta$ for this graph.

(b) Suppose the initial configuration is $\eta = (2, 2)$ (both sites unstable since $z_c = \deg(i) = 2$). Exhibit two distinct legal toppling sequences — one starting by toppling site 1, one starting by toppling site 2 — and show that both sequences terminate in the same stable configuration $\eta'$ with the same toppling vector $\mathbf{n} = (n_1, n_2)$.

(c) Using the identity $\eta' = \eta - \Delta \mathbf{n}$, verify your result algebraically. Confirm that $\det(\Delta) \neq 0$ and explain why this implies $\mathbf{n}$ is unique.

---

### Problem 2: Toppling Lemma via Monotonicity

*The toppling lemma is the key monotonicity property that underlies the Abelian sandpile's tractability. This problem derives the lemma carefully and examines its limits.*

> **Prerequisites:** cf. note [[note#4.3 The Toppling Lemma|§4.3 — The Toppling Lemma]]; requires Problem 1

Let $G = (V \cup \{s\}, E)$ be the sandpile graph. Let $\eta$ be a configuration that stabilizes to $\text{Stab}(\eta)$ with toppling vector $\mathbf{n}(\eta)$.

(a) Let $S = (i_1, i_2, \ldots, i_N)$ be a fixed legal toppling sequence that stabilizes $\eta$. Show that $S$ is also a legal toppling sequence for the configuration $\eta + \mathbf{e}_j$ (one extra grain at site $j$). Here "legal" means each site is toppled only when unstable.

(b) Argue from (a) that $n_i(\eta + \mathbf{e}_j) \geq n_i(\eta)$ for all $i \in V$. State explicitly which property of the Abelian sandpile you invoke.

(c) Suppose instead you remove a grain: consider $\eta - \mathbf{e}_j$ (with $\eta_j \geq 1$). Does the analogous statement $n_i(\eta - \mathbf{e}_j) \leq n_i(\eta)$ hold? Give a brief argument or counterexample.

(d) Using the toppling lemma, argue that the Green's function $G_{ij} = n_i(\eta + \mathbf{e}_j) - n_i(\eta)$ is well-defined and independent of the base configuration $\eta$ within the recurrent class. [Hint: use the fact that toppling numbers satisfy $\mathbf{n} = \Delta^{-1}(\eta - \eta')$ and linearity of $\Delta^{-1}$.]

---

### Problem 3: Positive Definiteness and Invertibility of the Toppling Matrix

*This problem establishes the foundational linear-algebraic fact behind the Abelian property: $\Delta$ is positive definite, so $\ker(\Delta) = \{\mathbf{0}\}$, giving uniqueness of stabilization.*

> **Prerequisites:** cf. note [[note#4.1 Setup on a General Graph|§4.1 — Setup on a General Graph]]

Let $G = (V \cup \{s\}, E)$ be a finite connected undirected graph with at least one edge between $V$ and the sink $s$.

(a) For any $\mathbf{x} \in \mathbb{R}^{|V|}$, show that the quadratic form decomposes as:
$$\mathbf{x}^\top \Delta \mathbf{x} = \sum_{\{i,j\} \in E,\ i,j \in V} (x_i - x_j)^2 + \sum_{\substack{i \in V \\ \{i,s\} \in E}} x_i^2$$

(b) Show that $\mathbf{x}^\top \Delta \mathbf{x} = 0$ implies $\mathbf{x} = \mathbf{0}$, using connectivity of $G$ and the existence of at least one edge to $s$.

(c) Conclude that $\Delta$ is positive definite and hence invertible. Explain what invertibility implies for uniqueness of the toppling vector $\mathbf{n}$ — and hence for the uniqueness of the final stable configuration after any sequence of topplings.

(d) The unrestricted graph Laplacian $L$ on $V \cup \{s\}$ is only positive semidefinite, with a zero eigenvector (the all-ones vector). Explain precisely why deleting the row and column corresponding to $s$ removes the zero eigenvalue and promotes $\Delta$ to positive definiteness.

---

### Problem 4: Mean Avalanche Size from Finite-Size Scaling

*This problem extracts quantitative predictions from the finite-size scaling ansatz by computing moments of the avalanche size distribution. The divergence of the mean with $L$ is the precise statement that there is no characteristic scale.*

> **Prerequisites:** cf. note [[note#5.1 Finite-Size Scaling|§5.1 — Finite-Size Scaling]]

Assume the finite-size scaling ansatz $P(s, L) = s^{-\tau_s} g(s/L^D)$, where $g(u) \to g_0 > 0$ for $u \to 0^+$ and $g(u) \to 0$ rapidly for $u \to \infty$.

(a) Show that the mean avalanche size satisfies $\langle s \rangle_L \sim L^{D(2-\tau_s)}$ for $\tau_s < 2$, by evaluating $\langle s \rangle_L = \int_1^\infty s\, P(s,L)\, ds$ with the substitution $u = s/L^D$.

(b) For the 2D BTW sandpile ($\tau_s \approx 1.20$, $D \approx 2.75$), compute the exponent $D(2 - \tau_s)$ and interpret its physical meaning.

(c) Show that for $\tau_s < 2$ the variance $\text{Var}(s)_L = \langle s^2 \rangle_L - \langle s \rangle_L^2$ also diverges as $L \to \infty$, and find its scaling exponent.

(d) Derive the condition on $\tau_s$ under which $\langle s \rangle_L$ remains finite as $L \to \infty$, and explain why that condition is equivalent to the power law $P(s) \sim s^{-\tau_s}$ on $(1, \infty)$ having a finite mean.

---

### Problem 5: Spectral Exponent from Avalanche Duration Distribution

*This problem derives the 1/f noise connection rigorously from the rectangular-pulse model of avalanches, pinning down exactly which value of the duration exponent gives true pink noise.*

> **Prerequisites:** cf. note [[note#2.4 The 1/f Noise Connection|§2.4 — The 1/f Noise Connection]]

Model the output signal from an SOC system as a superposition of rectangular pulses; the $k$-th avalanche contributes a pulse of height $h_0$ and duration $T_k$ drawn i.i.d. from $P(T) \sim T^{-\tau_T}$ for $1 \leq \tau_T < 3$.

(a) Compute the Fourier transform $\hat{\phi}_T(f)$ of a single rectangular pulse $\phi_T(t) = h_0 \cdot \mathbf{1}[0 \leq t \leq T]$. Show that $|\hat{\phi}_T(f)|^2 \approx h_0^2 T^2$ for $fT \ll 1$ and $|\hat{\phi}_T(f)|^2 \ll h_0^2 T^2$ for $fT \gg 1$.

(b) Writing $S(f) \approx \int_1^\infty P(T)\, |\hat{\phi}_T(f)|^2\, dT$, split the integral at $T^* = 1/f$ and show that the dominant contribution for small $f$ comes from $T > 1/f$, giving:
$$S(f) \sim \int_{1/f}^\infty T^{-\tau_T} \cdot T^2\, dT$$

(c) Evaluate the integral in (b) for $\tau_T < 3$ and show $S(f) \sim f^{-(3-\tau_T)}$.

(d) For what value of $\tau_T$ does this give exact $1/f$ noise ($\beta = 1$)? Compare this to the 2D BTW value $\tau_T \approx 1.37$ and compute the predicted spectral exponent.

(e) Identify one assumption in this derivation that may fail for real SOC systems and explain how it could shift $\beta$.

---

### Problem 6: SOC vs. Tuned Criticality

*This problem sharpens the conceptual distinction between self-organized and externally-tuned criticality by casting both in the language of control parameters and fixed points.*

> **Prerequisites:** cf. note [[note#Ordinary Criticality vs. SOC|§1 — Ordinary Criticality vs. SOC]]; cf. note [[note#3.1 The Attractor Argument|§3.1 — The Attractor Argument]]

(a) In an Ising ferromagnet, $\xi \sim |T - T_c|^{-\nu}$ diverges only at $T = T_c$. Define precisely what "fine-tuning" means: what agent fixes $T$, and what is the qualitative effect of a perturbation $T \to T_c + \delta T$ on the correlation length and on the avalanche-size distribution?

(b) In the BTW sandpile, identify the quantity that plays the role of the control parameter analogous to $T$. Derive a schematic differential equation for this quantity that shows it is a stable fixed point of the dynamics, not an externally imposed value.

(c) Some authors argue that SOC self-tuning is merely an internal feedback loop equivalent in principle to an external controller. Evaluate this argument: does the mechanism of tuning (internal vs. external) have observable physical consequences, or is it a distinction without a difference?

(d) Provide one concrete example of a system with a heavy-tailed or power-law distribution that is definitively not at a second-order phase transition. Derive or cite the mechanism responsible for the heavy tail, and explain why it cannot be attributed to SOC.

---

### Problem 7: The Attractor Argument and Its Prerequisites

*The attractor argument explains why the BTW dynamics self-tunes to criticality, but it relies on precise conditions. This problem makes those conditions explicit and examines what breaks when they fail.*

> **Prerequisites:** cf. note [[note#3.1 The Attractor Argument|§3.1 — The Attractor Argument]]; cf. note [[note#2.2 Time-Scale Separation|§2.2 — Time-Scale Separation]]

(a) Define the input flux $J_\text{in}$ and output flux $J_\text{out}$ formally in the BTW model, and write a differential equation for $d\langle z \rangle / dt$ in terms of these fluxes. Use this to show that $\langle z \rangle_c$ is a stable fixed point.

(b) Show by reasoning what happens in each degenerate limit: (i) periodic boundary conditions with $h > 0$; (ii) no drive ($h = 0$), open boundaries, starting from a random stable configuration.

(c) The attractor argument controls only $\langle z \rangle$, not the full distribution. Explain why $\langle z \rangle = \langle z \rangle_c$ does not alone imply power-law fluctuations, and what additional structure — specifically the Abelian property and Dhar's recurrence characterization — is required to conclude that the stationary measure has power-law tails.

---

### Problem 8: Interpreting the Finite-Size Scaling Function

*The scaling function $g$ encodes all finite-size corrections to pure criticality. This problem develops geometric and statistical intuition for what $g$ looks like and how data collapse extracts the exponents.*

> **Prerequisites:** cf. note [[note#5.1 Finite-Size Scaling|§5.1 — Finite-Size Scaling]]; cf. note [[note#5.2 Scaling Relations|§5.2 — Scaling Relations]]

(a) Sketch $g : (0,\infty) \to (0,\infty)$ consistent with $g(u) \to g_0 > 0$ as $u \to 0^+$ and rapid decay for $u \to \infty$. Describe two physically plausible functional forms for the decay (e.g., exponential, Gaussian) and explain which is more consistent with a sharp cutoff at $s \sim L^D$.

(b) For fixed $L$, sketch $\log P(s, L)$ vs. $\log s$ and identify: (i) the power-law regime; (ii) the crossover scale $s^* \sim L^D$; (iii) the cutoff regime. Describe qualitatively how the plot evolves as $L \to \infty$.

(c) Explain data collapse: given empirical histograms $\hat{P}(s, L_k)$ for sizes $L_1 < L_2 < L_3$, describe the procedure for extracting $\tau_s$ and $D$ by collapsing the rescaled curves $s^{\tau_s} \hat{P}(s, L_k)$ plotted against $s/L_k^D$.

(d) The ansatz assumes a single relevant length scale $L$. Identify one additional length scale present in a realistic SOC system that could spoil the collapse, and explain how it would manifest in the data.

---

### Problem 9: Universality Classes and Model Symmetries

*Universality class membership — not qualitative SOC phenomenology — is the sharp physical distinction between models. This problem develops the classification criteria and their consequences for exponents.*

> **Prerequisites:** cf. note [[note#5.4 Universality|§5.4 — Universality]]

(a) State the Manna model toppling rule precisely and identify the symmetry it breaks relative to BTW. Argue why breaking this symmetry changes the universality class and hence the numerical values of the critical exponents.

(b) In equilibrium critical phenomena, universality classes are labeled by $(d, \text{order-parameter symmetry}, \text{interaction range})$. Identify three analogous determinants for SOC universality classes and, for each, give a pair of models that differ in that determinant and exhibit different exponents.

(c) Logarithmic corrections to scaling are a hallmark of a system at or above its upper critical dimension $d_c$. Explain conceptually why these corrections appear at $d_c$, and argue from the known mean-field exponent $\tau_s^\text{MF} = 3/2$ and the 2D BTW value $\tau_s \approx 1.20$ that the 2D BTW sandpile may be anomalous (or at its upper critical dimension).

(d) The Gutenberg-Richter law gives $\tau_E \approx 5/3$ for seismic energy. Check whether this is consistent with any known SOC universality class (BTW 2D, BTW mean-field, Manna 2D). State what geometric or physical assumption on earthquake fault systems would be required for the mean-field exponent to apply.

---

### Problem 10: Empirical Power Laws and the SOC Attribution Problem

*Observing a power law is necessary but not sufficient for SOC. This problem develops statistical tools for distinguishing SOC from competing mechanisms.*

> **Prerequisites:** cf. note [[note#Natural Examples and Connection to Scaling Laws|§1 — Natural Examples and Connection to Scaling Laws]]

(a) List three mechanisms other than SOC that generate power-law or heavy-tailed distributions. For each, name a specific natural example and state what distinguishes the mechanism from SOC at the level of the underlying dynamics.

(b) Neural avalanches in cortical tissue show $P(s) \sim s^{-3/2}$, consistent with a mean-field branching process at criticality. Describe two additional measurements beyond the power-law exponent that would strengthen or weaken the SOC hypothesis for neural tissue.

(c) In the Bak-Sneppen model, identify the slow drive and dissipation mechanism by analogy with the BTW sandpile. What plays the roles of "grains," "height," and "toppling"?

(d) Power laws are often fit by log-log linear regression. Identify two statistical pitfalls of this approach. State the maximum likelihood estimator for the discrete power-law exponent and explain the role of the lower cutoff $s_\text{min}$ in the estimator.

---

### Problem 11: Recurrent vs. Transient Configurations and det of Delta

*This problem proves the remarkable identity $|\mathcal{R}| = \det(\Delta)$, which connects the combinatorics of recurrent sandpile configurations to a purely algebraic quantity.*

> **Prerequisites:** cf. note [[note#4.4 Recurrent Configurations and the Sandpile Group|§4.4 — Recurrent Configurations and the Sandpile Group]]; requires Problem 3

(a) Define recurrent and transient configurations precisely in terms of the stationary distribution of the Markov chain. Show that the set of transient configurations $\mathcal{T}$ is visited at most finitely often with probability 1.

(b) The matrix-tree theorem states that for a connected graph $G$ with Laplacian $L$, $\det(\Delta) = $ (number of spanning trees of $G$ rooted at $s$). State this theorem and verify it for the two-site graph from Problem 1 by explicitly counting spanning trees.

(c) Dhar's result asserts $|\mathcal{R}| = \det(\Delta)$. Sketch the bijection between recurrent configurations and spanning trees of $G$ rooted at $s$ (you may describe the bijection in words using the burning algorithm, without full proof).

(d) For the $L \times L$ grid graph with a corner sink, $\det(\Delta)$ grows as $e^{c L^2}$ where $c = 4G/\pi$ and $G$ is Catalan's constant $\approx 0.9159$. Compute $|\mathcal{R}|$ for $L = 2$ (a $2 \times 2$ grid with a corner sink) by explicitly enumerating all recurrent configurations using Dhar's burning algorithm.

---

### Problem 12: The Sandpile Group is Abelian

*The set of recurrent configurations under componentwise addition followed by stabilization forms an abelian group — one of the rare algebraic structures arising from a purely dynamical system.*

> **Prerequisites:** cf. note [[note#4.4 Recurrent Configurations and the Sandpile Group|§4.4 — Recurrent Configurations and the Sandpile Group]]; cf. note [[note#4.2 The Abelian Property|§4.2 — The Abelian Property]]

Define $\mathcal{R}$ as the set of recurrent stable configurations and $\eta_1 \oplus \eta_2 = \text{Stab}(\eta_1 + \eta_2)$ for $\eta_1, \eta_2 \in \mathcal{R}$.

(a) Prove that $\oplus$ is commutative: $\eta_1 \oplus \eta_2 = \eta_2 \oplus \eta_1$. Identify exactly where the Abelian property of the sandpile (Section 4.2) is used.

(b) Prove that $\oplus$ is associative: $(\eta_1 \oplus \eta_2) \oplus \eta_3 = \eta_1 \oplus (\eta_2 \oplus \eta_3)$. [Hint: use the fact that stabilization can be decomposed into independent toppling sequences.]

(c) Prove closure: if $\eta_1, \eta_2 \in \mathcal{R}$, then $\eta_1 \oplus \eta_2 \in \mathcal{R}$. [Hint: adding a recurrent configuration to any configuration and stabilizing produces a recurrent configuration — argue this from the definition of recurrence.]

(d) Identify the identity element of $(\mathcal{R}, \oplus)$. Describe it for the $2 \times 2$ grid example: what stable configuration $e$ satisfies $\eta \oplus e = \eta$ for all $\eta \in \mathcal{R}$?

---

### Problem 13: Green's Function Symmetry

*The Green's function $G_{ij}$ measures how many times site $i$ topples when a grain is added at $j$. Proving its symmetry $G_{ij} = G_{ji}$ reveals a deep reciprocity in the sandpile dynamics.*

> **Prerequisites:** cf. note [[note#4.2 The Abelian Property|§4.2 — The Abelian Property]]; cf. note [[note#4.3 The Toppling Lemma|§4.3 — The Toppling Lemma]]; requires Problem 2

For a recurrent configuration $\eta$, define the Green's function $G_{ij} = n_i(\eta + \mathbf{e}_j) - n_i(\eta)$, where $n_i(\cdot)$ denotes the number of topplings at site $i$ during stabilization.

(a) The toppling relation states that $\mathbf{n}(\eta)$ satisfies $\Delta \mathbf{n}(\eta) = \eta - \text{Stab}(\eta)$ for any configuration $\eta$. Consider two configurations $\eta$ and $\eta + \mathbf{e}_j$. Applying the toppling relation to each:
$$\Delta \mathbf{n}(\eta + \mathbf{e}_j) = (\eta + \mathbf{e}_j) - \text{Stab}(\eta + \mathbf{e}_j)$$
$$\Delta \mathbf{n}(\eta) = \eta - \text{Stab}(\eta)$$
Subtract and rearrange to get:
$$\Delta(\mathbf{n}(\eta + \mathbf{e}_j) - \mathbf{n}(\eta)) = \mathbf{e}_j - (\text{Stab}(\eta + \mathbf{e}_j) - \text{Stab}(\eta))$$
Argue that for recurrent configurations in the stationary regime, on average over $\eta \in \mathcal{R}$ the final configurations $\text{Stab}(\eta + \mathbf{e}_j)$ and $\text{Stab}(\eta)$ are both uniformly distributed on $\mathcal{R}$, so $\mathbb{E}[\text{Stab}(\eta + \mathbf{e}_j) - \text{Stab}(\eta)] = 0$. Hence $\mathbb{E}[\Delta G_{\cdot j}] = \mathbf{e}_j$, giving $G_{\cdot j} = \Delta^{-1} \mathbf{e}_j$ in expectation.

(b) Conclude that $G_{ij} = (\Delta^{-1})_{ij}$.

(c) Since $\Delta$ is a real symmetric matrix (verify this from its definition), $\Delta^{-1}$ is also symmetric. Conclude $G_{ij} = G_{ji}$.

(d) Interpret the symmetry $G_{ij} = G_{ji}$ physically: what does it mean that adding a grain at $j$ causes $i$ to topple exactly as many times as adding a grain at $i$ causes $j$ to topple?

---

### Problem 14: Power-Law Tail Conditions and Moment Existence

*This problem derives the precise conditions on the exponent $\tau$ under which a power-law distribution has finite moments, connecting the abstract mathematics to the physical requirement that SOC produces divergent fluctuations.*

> **Prerequisites:** cf. note [[note#The Three Empirical Hallmarks|§1 — The Three Empirical Hallmarks]]; cf. note [[note#5.1 Finite-Size Scaling|§5.1 — Finite-Size Scaling]]

Let $P(s) = C s^{-\tau}$ for $s \geq s_0 > 0$ be a normalized power-law distribution.

(a) Find the normalization constant $C$ as a function of $\tau$ and $s_0$. For what values of $\tau$ is the distribution normalizable on $[s_0, \infty)$?

(b) Compute $\langle s \rangle = \int_{s_0}^\infty s\, P(s)\, ds$ and determine the condition on $\tau$ for which the mean is finite.

(c) Compute $\langle s^k \rangle$ for integer $k \geq 1$ and determine for each $k$ the threshold $\tau_k^*$ below which the $k$-th moment diverges.

(d) In the context of SOC, the avalanche size exponent $\tau_s \approx 1.20 < 2$ for the 2D BTW sandpile in infinite volume. Show that this implies the mean avalanche size is infinite in the thermodynamic limit, and explain why this is precisely the statement that "there is no characteristic scale."

---

### Problem 15: Upper Critical Dimension via Dimensional Analysis

*This problem derives $d_c = 4$ for the BTW sandpile from a mean-field (branching process) argument, mirroring the derivation of $d_c = 4$ for directed percolation.*

> **Prerequisites:** cf. note [[note#2.3 Avalanche Observables|§2.3 — Avalanche Observables]]; cf. note [[note#5.4 Universality|§5.4 — Universality]]

A mean-field theory of avalanche dynamics models each toppling site as a branching process: a site that topples sends grains to $k$ neighbors independently, each with probability $p$ of causing a topple. The process is critical when the mean number of offspring per toppling equals 1.

(a) In the mean-field branching process at criticality, show that $P(s) \sim s^{-3/2}$ (the mean-field exponent $\tau_s^\text{MF} = 3/2$). [Hint: use the generating function for the total progeny of a critical Galton-Watson process.]

(b) In $d$ spatial dimensions, the characteristic linear size of an avalanche of size $s$ scales as $r \sim s^{1/d}$ (if avalanches are compact). The mean-field approximation neglects return visits and spatial correlations. The correction to mean-field becomes relevant when the probability that two branches of an avalanche revisit the same site is of order 1. Show that this condition is equivalent to $s \cdot r^{-d} \sim 1$, i.e., $s \sim r^d$.

(c) Two branches of a critical avalanche of total size $s$ each perform random walks of $O(s)$ steps. The probability that two independent random walkers of length $n$ in $d$ dimensions share at least one site scales as $O(s^2 \cdot n^{-d/2})$ (there are $O(s^2)$ pairs of time steps, and the probability that two walkers are at the same site at a given pair of times is $O(n^{-d/2})$ from the Gaussian Green's function). With $n \sim s$, this gives $O(s^{2-d/2})$. Determine the upper critical dimension $d_c$ above which mean-field theory is exact by setting the expected number of self-intersections to $O(1)$.

(d) State the predicted mean-field exponents ($\tau_s = 3/2$, $\tau_T = 2$, $D = 4$, $z = 2$) and verify they are consistent with the scaling relations in Section 5.2 of the note.

---

### Problem 16: Fractal Dimension of the Avalanche Boundary

*The fractal dimension of the avalanche footprint is not an independent quantity — it is determined by the other scaling exponents. This problem derives that relationship.*

> **Prerequisites:** cf. note [[note#5.2 Scaling Relations|§5.2 — Scaling Relations]]; cf. note [[note#2.3 Avalanche Observables|§2.3 — Avalanche Observables]]

(a) Recall the definitions: $a$ is the number of distinct sites that topple at least once (area), $s$ is the total topplings, $D$ is the avalanche fractal dimension relating $s_\text{max} \sim L^D$, and $d_f$ is the spatial fractal dimension of the avalanche footprint (area $a \sim r^{d_f}$ where $r$ is the linear extent).

(b) Using the scaling relation $a \sim s^{d_f/D}$ and the finite-size scaling forms for $P(s,L)$ and $P(a,L)$, derive the relation:
$$\tau_a = 1 + \frac{D}{d_f}(\tau_s - 1)$$

(c) For the 2D BTW sandpile, the note states $d_f \approx 2$ (space-filling footprints). Verify that the tabulated exponents ($\tau_s \approx 1.20$, $D \approx 2.75$, $\tau_a \approx 1.14$) satisfy the relation in (b) to within stated uncertainties.

(d) If an SOC model in 3D had compact avalanches ($d_f = 3$) and $D = 3$, what would the scaling relation predict for $\tau_a$ in terms of $\tau_s$? Interpret this result.

---

### Problem 17: Eigenvalue Bounds on the Spectral Gap of Delta

*The spectral gap of $\Delta$ controls the mixing time of the sandpile Markov chain and the decay of correlations between successive avalanches. This problem derives elementary bounds on the gap.*

> **Prerequisites:** cf. note [[note#4.1 Setup on a General Graph|§4.1 — Setup on a General Graph]]; requires Problem 3

Let $\lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_{|V|}$ be the eigenvalues of $\Delta$.

(a) By the variational characterization (Courant-Fischer), show that $\lambda_1 = \min_{\mathbf{x} \neq 0} \frac{\mathbf{x}^\top \Delta \mathbf{x}}{\|\mathbf{x}\|^2}$. Use this to show $\lambda_1 > 0$ (which you already know from Problem 3), and give a lower bound in terms of graph properties (e.g., the minimum number of edges from $V$ to $s$).

(b) Show that $\lambda_{|V|} \leq 2 \Delta_\text{max}$ where $\Delta_\text{max} = \max_{i \in V} \deg(i)$, using the fact that $\Delta$ is diagonally dominant.

(c) For the $L \times L$ grid with a corner sink, argue (without exact computation) that $\lambda_1 = O(L^{-2})$ as $L \to \infty$. [Hint: construct a test vector $\mathbf{x}$ that is slowly varying and achieves small Rayleigh quotient.]

(d) The mixing time of the sandpile Markov chain is related to $1/\lambda_1$ (the inverse spectral gap). What does $\lambda_1 = O(L^{-2})$ imply for how long the chain must run before observing stationary-regime statistics? Relate this to the burn-in period required in simulation.

---

### Problem 18: Gutenberg-Richter Law as a Power-Law Derivation

*This problem works through the full derivation connecting the Gutenberg-Richter magnitude-frequency relation to a power law for seismic energy, pinning down the exponent $\tau_E$.*

> **Prerequisites:** cf. note [[note#Natural Examples and Connection to Scaling Laws|§1 — Natural Examples and Connection to Scaling Laws]]

The Gutenberg-Richter law states $\log_{10} N(\geq M) = a - bM$ with $b \approx 1$, where $N(\geq M)$ is the number of earthquakes with moment magnitude at least $M$.

(a) Recall the moment magnitude scale: $M = \frac{2}{3}\log_{10} E - \text{const}$ where $E$ is seismic energy. Invert this to express $E$ as a function of $M$.

(b) Substitute into the Gutenberg-Richter law to show that the cumulative distribution of seismic energy satisfies $N(\geq E) \sim E^{-\beta}$ for some $\beta$ that you should express in terms of $b$.

(c) Differentiate to obtain the probability density $P(E) \sim E^{-\tau_E}$ and compute $\tau_E$ in terms of $b$. For $b = 1$, evaluate $\tau_E$ numerically.

(d) Compare $\tau_E$ to the BTW mean-field value $\tau_s = 3/2$ and to the 2D BTW value $\tau_s \approx 1.20$. Discuss what dimension or universality class of SOC model could produce the observed earthquake exponent.

---

## Algorithmic Applications

### Problem 19: BTW Sandpile Simulation

*This problem sketches a complete simulation of the BTW sandpile including the queue-based relaxation and the MLE power-law exponent estimator.*

> **Prerequisites:** cf. note [[note#2.1 Formal Definition|§2.1 — Formal Definition]]; cf. note [[note#2.3 Avalanche Observables|§2.3 — Avalanche Observables]]

(a) **Inputs and data structures:** Represent the height field as a 2D integer array `z[i][j]` for $0 \leq i, j < L$ with open boundaries. Write pseudocode for `topple(z, i, j)` that performs one toppling at $(i,j)$, distributes grains to valid neighbors, discards off-grid grains, and returns the number of grains dissipated.

(b) **Relaxation loop:** Write pseudocode for `relax(z)` using a queue of unstable sites. After each toppling, newly unstabilized neighbors are enqueued. Return the total topplings $s$ and the set of toppled sites. Compare the time complexity of this queue-based approach to the naive full-scan approach.

(c) **Main simulation loop:** Write pseudocode for the full simulation:
1. Initialize `z` to all zeros.
2. Burn in $N_\text{burn}$ steps: add a grain at a uniformly random site, call `relax`.
3. Measure $N_\text{measure}$ steps: add a grain, call `relax`, record $s$.

(d) **MLE exponent estimator:** Given samples $\{s_k\}$ with $s_k \geq s_\text{min}$, write pseudocode to compute:
$$\hat\tau_s = 1 + n\left[\sum_{k=1}^n \ln \frac{s_k}{s_\text{min} - 1/2}\right]^{-1}$$
Explain the role of $s_\text{min}$ and outline a criterion for selecting it (e.g., minimizing the Kolmogorov-Smirnov statistic over candidate values).

---

### Problem 20: Dhar's Burning Algorithm

*This problem sketches a queue-based implementation of Dhar's burning algorithm, which decides recurrence in $O(|V|)$ time, and connects the algorithm to enumeration of the sandpile group.*

> **Prerequisites:** cf. note [[note#4.4 Recurrent Configurations and the Sandpile Group|§4.4 — Recurrent Configurations and the Sandpile Group]]

(a) **Inputs and data structures:** Represent the configuration as `z[i][j]` with $0 \leq z[i][j] \leq 3$, a boolean array `burned[i][j]` initialized to `False`, and an integer array `unburned_neighbors[i][j]` counting currently unburned non-sink neighbors. Initialize the sink as burned and adjust `unburned_neighbors` accordingly.

(b) **Burning rule:** A site $(i,j)$ is burnable if $z[i][j] \geq \text{unburned\_neighbors}[i][j]$. Write pseudocode for `process(i, j)` that: marks $(i,j)$ burned, decrements `unburned_neighbors` of all unburned neighbors of $(i,j)$, and enqueues any neighbor that becomes burnable.

(c) **Main loop:** Write pseudocode using a queue initialized with all boundary sites adjacent to the sink. Process each site; after the queue empties, return `True` iff all sites are burned.

(d) **Complexity:** State the time complexity of the queue-based algorithm in terms of $|V|$ and compare to the naive repeated-pass algorithm.

(e) **Application:** Describe how to use this algorithm to: (i) verify empirically that the BTW simulation of Problem 19 visits only recurrent configurations in stationarity; (ii) enumerate all recurrent configurations for small $L$ and verify the count equals $\det(\Delta)$.

---

### Problem 21: Scaling Exponent Estimation via Data Collapse

*This problem sketches the grid-search data collapse algorithm for jointly estimating $(\tau_s, D)$ from multi-size simulation data.*

> **Prerequisites:** cf. note [[note#5.1 Finite-Size Scaling|§5.1 — Finite-Size Scaling]]; requires Problem 19

(a) **Data collection:** Assume simulations at $K$ system sizes $L_1 < \cdots < L_K$, each yielding $N_k$ avalanche size samples. Describe how to construct empirical histograms $\hat{P}(s, L_k)$ in log-spaced bins and explain why log-spacing is preferred over linear spacing.

(b) **Collapse objective:** The rescaled data $Q(u_k, L_k) = (s_k)^{\tau_s} \hat{P}(s_k, L_k)$ should collapse to a universal curve $g(u)$ with $u = s/L^D$. Write pseudocode for a grid search over $(\tau_s, D)$ that minimizes the mean squared deviation of $Q$ values at equal $u$ across different sizes.

(c) **Collapse quality metric:** Define a specific functional form for the collapse quality $\mathcal{Q}(\tau_s, D)$. Discuss one advantage and one disadvantage of least-squares vs. maximum likelihood as the objective for collapse quality.

(d) **Finite-size corrections:** If corrections of the form $L^{-\omega} h(s/L^D)$ are present, describe how they would manifest as systematic residuals in the collapse and how you would detect them from the size dependence of the residuals.

---

### Problem 22: Sandpile Group Enumeration on Small Graphs

*This problem sketches an algorithm to enumerate all elements of the sandpile group $(\mathcal{R}, \oplus)$ for small graphs and verify the identity $|\mathcal{R}| = \det(\Delta)$.*

> **Prerequisites:** cf. note [[note#4.4 Recurrent Configurations and the Sandpile Group|§4.4 — Recurrent Configurations and the Sandpile Group]]; requires Problems 11 and 12

(a) **Inputs and data structures:** For a graph $G = (V \cup \{s\}, E)$ with $|V|$ small (say $\leq 9$), represent each configuration as a tuple of integers. Describe a data structure for storing the set of all recurrent configurations $\mathcal{R}$ and the group multiplication table.

(b) **Enumeration:** Write pseudocode to enumerate all stable configurations $\{0,\ldots,\deg(i)-1\}^{|V|}$, apply Dhar's burning test from Problem 20 to each, and collect those that pass into $\mathcal{R}$.

(c) **Group table construction:** Write pseudocode to compute the full Cayley table of $(\mathcal{R}, \oplus)$: for each pair $(\eta_1, \eta_2) \in \mathcal{R}^2$, compute $\eta_1 \oplus \eta_2$ by componentwise addition followed by relaxation (using `relax` from Problem 19). Verify commutativity by checking symmetry of the table.

(d) **Verification:** Compute $\det(\Delta)$ for the $2 \times 2$ grid with a corner sink and verify $|\mathcal{R}| = \det(\Delta)$. Identify the identity element and one non-trivial element and its inverse.

---

### Problem 23: Branching Process Simulation for Mean-Field SOC

*The mean-field limit of SOC is equivalent to a critical branching process. This problem sketches a simulation that generates mean-field avalanche statistics and verifies $\tau_s^\text{MF} = 3/2$.*

> **Prerequisites:** cf. note [[note#2.3 Avalanche Observables|§2.3 — Avalanche Observables]]; requires Problem 15

(a) **Inputs and data structures:** A critical Galton-Watson process has offspring distribution $\{p_k\}$ with mean $\mu = \sum_k k\, p_k = 1$. Represent the population at each generation as an integer. Choose a specific offspring distribution (e.g., Poisson with $\lambda = 1$, or Bernoulli where each particle produces 0 or 2 offspring each with probability $1/2$) and justify that it is critical.

(b) **Simulation loop:** Write pseudocode to simulate a single avalanche (branching process run until extinction or indefinite growth). The avalanche size $s$ is the total number of individuals across all generations. Run until extinction or until $s$ exceeds a cutoff $s_\text{max}$.

(c) **Main loop and measurement:** Write pseudocode to generate $N$ avalanches (conditioned on finite total size), collect sizes $\{s_k\}$, and estimate $\hat\tau_s$ using the MLE from Problem 19(d). What cutoff $s_\text{max}$ and how many samples $N$ would be needed to distinguish $\tau_s = 3/2$ from $\tau_s = 1.20$ with high confidence?

(d) **Comparison:** Describe how you would use the simulated mean-field data as a reference to test whether BTW simulation data at large $L$ converges to mean-field behavior (as expected for $d \geq 4$) or departs from it (as observed for $d = 2$).
