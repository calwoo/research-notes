# Self-Organized Criticality: Exercises

## Table of Contents

1. [Derivation Problems](#derivation-problems)
   - [Problem 1: Abelian Property for a Two-Site System](#problem-1-abelian-property-for-a-two-site-system)
   - [Problem 2: Toppling Lemma via Monotonicity](#problem-2-toppling-lemma-via-monotonicity)
   - [Problem 3: Positive Definiteness and Invertibility of Δ](#problem-3-positive-definiteness-and-invertibility-of-delta)
   - [Problem 4: Mean Avalanche Size from Finite-Size Scaling](#problem-4-mean-avalanche-size-from-finite-size-scaling)
   - [Problem 5: Spectral Exponent from Avalanche Duration Distribution](#problem-5-spectral-exponent-from-avalanche-duration-distribution)
2. [Conceptual Questions](#conceptual-questions)
   - [Problem 6: SOC vs. Tuned Criticality](#problem-6-soc-vs-tuned-criticality)
   - [Problem 7: The Attractor Argument and Its Prerequisites](#problem-7-the-attractor-argument-and-its-prerequisites)
   - [Problem 8: Interpreting the Finite-Size Scaling Function](#problem-8-interpreting-the-finite-size-scaling-function)
   - [Problem 9: Universality Classes and Model Symmetries](#problem-9-universality-classes-and-model-symmetries)
   - [Problem 10: Empirical Power Laws and the SOC Attribution Problem](#problem-10-empirical-power-laws-and-the-soc-attribution-problem)
3. [Implementation Sketches](#implementation-sketches)
   - [Problem 11: BTW Sandpile Simulation](#problem-11-btw-sandpile-simulation)
   - [Problem 12: Dhar's Burning Algorithm](#problem-12-dhars-burning-algorithm)
   - [Problem 13: Scaling Exponent Estimation via Data Collapse](#problem-13-scaling-exponent-estimation-via-data-collapse)

---

## Derivation Problems

### Problem 1: Abelian Property for a Two-Site System

Consider an Abelian sandpile on the simplest non-trivial graph: two non-sink vertices $\{1, 2\}$ connected by a single edge, with both vertices also connected to a sink $s$. Suppose vertex 1 has degree 2 (one edge to vertex 2, one edge to $s$) and vertex 2 has degree 2 (one edge to vertex 1, one edge to $s$). The toppling threshold at each vertex equals its degree.

**(a)** Write down the $2 \times 2$ toppling matrix $\Delta$ for this graph.

**(b)** Suppose the initial configuration is $\eta = (2, 2)$ (both sites have height 2, hence both are unstable since $z_c = \deg(i) = 2$). Exhibit two distinct legal toppling sequences — one starting by toppling site 1, one starting by toppling site 2 — and show that both sequences:
  - Terminate in a stable configuration (all $z_i < \deg(i)$)
  - Produce the same final configuration $\eta'$
  - Result in the same toppling vector $\mathbf{n} = (n_1, n_2)$

**(c)** Using the identity $\eta' = \eta - \Delta \mathbf{n}$, verify your result algebraically. Confirm that $\det(\Delta) \neq 0$ for this graph and explain why this implies $\mathbf{n}$ is unique.

---

### Problem 2: Toppling Lemma via Monotonicity

Let $G = (V \cup \{s\}, E)$ be the sandpile graph as in Section 4. Let $\eta$ be a configuration that stabilizes to $\eta' = \text{Stab}(\eta)$ with toppling vector $\mathbf{n}(\eta)$.

**(a)** Let $\sigma^{(0)} = \eta$ and define the sequence of toppling steps. Consider a fixed legal toppling sequence $S = (i_1, i_2, \ldots, i_N)$ that stabilizes $\eta$. Show that the same sequence $S$ is also legal for the configuration $\eta + \mathbf{e}_j$ (adding one grain at site $j$). Here, "legal" means that each site is toppled only when it is unstable.

**(b)** Argue from (a) that $n_i(\eta + \mathbf{e}_j) \geq n_i(\eta)$ for all $i \in V$. Be precise about which property of the Abelian sandpile you use.

**(c)** Suppose instead you remove a grain: consider $\eta - \mathbf{e}_j$ (assuming $\eta_j \geq 1$). Does the analogous statement $n_i(\eta - \mathbf{e}_j) \leq n_i(\eta)$ hold? Give a brief argument or counterexample.

**(d)** Using the toppling lemma, show that the map $j \mapsto n_i(\eta + \mathbf{e}_j)$ is well-defined for all $j \in V$ and $i \in V$, and that the Green's function $G_{ij} = n_i(\eta + \mathbf{e}_j) - n_i(\eta)$ is independent of the base configuration $\eta$ (at least for $\eta$ in the recurrent class). [Hint: use linearity properties of the toppling matrix.]

---

### Problem 3: Positive Definiteness and Invertibility of $\Delta$

Let $G = (V \cup \{s\}, E)$ be a finite connected undirected graph with at least one edge between $V$ and the sink $s$. Let $\Delta$ be the $|V| \times |V|$ toppling matrix (restricted graph Laplacian).

**(a)** For any $\mathbf{x} \in \mathbb{R}^{|V|}$, expand $\mathbf{x}^\top \Delta \mathbf{x}$ as a sum over edges. Show that the quadratic form decomposes as:
$$\mathbf{x}^\top \Delta \mathbf{x} = \sum_{\{i,j\} \in E,\ i,j \in V} (x_i - x_j)^2 + \sum_{\substack{i \in V \\ \{i,s\} \in E}} x_i^2$$

**(b)** Show that $\mathbf{x}^\top \Delta \mathbf{x} = 0$ implies $\mathbf{x} = \mathbf{0}$. Use the fact that $G$ is connected and that at least one vertex in $V$ is adjacent to $s$.

**(c)** Conclude that $\Delta$ is positive definite, hence invertible. What does the invertibility of $\Delta$ imply for the uniqueness of stabilization? Relate this back to the Abelian property.

**(d)** The standard (unrestricted) graph Laplacian $L$ on $V \cup \{s\}$ is positive semidefinite, not positive definite (it has a zero eigenvalue for each connected component). Explain precisely why grounding the graph at the sink $s$ — i.e., deleting the row and column of $L$ corresponding to $s$ — promotes the matrix from semidefinite to definite.

---

### Problem 4: Mean Avalanche Size from Finite-Size Scaling

Assume the finite-size scaling ansatz for the avalanche size distribution:
$$P(s, L) = s^{-\tau_s}\, g\!\left(\frac{s}{L^D}\right)$$
where $g(u) \to g_0 > 0$ as $u \to 0^+$ and $g(u) \to 0$ rapidly for $u \to \infty$. Assume the normalization $\int_0^\infty P(s,L)\,ds = 1$ is satisfied (absorb any proportionality constant into $g_0$).

**(a)** Show that the mean avalanche size satisfies:
$$\langle s \rangle_L \sim L^{D(2 - \tau_s)}, \quad \tau_s < 2$$
by evaluating $\langle s \rangle_L = \int_1^\infty s\, P(s,L)\, ds$ using the substitution $u = s/L^D$.

**(b)** For the 2D BTW sandpile, $\tau_s \approx 1.20$ and $D \approx 2.75$. Compute the predicted exponent $D(2-\tau_s)$ and interpret: how does the mean avalanche size grow with system size?

**(c)** Repeat for the variance $\langle s^2 \rangle_L - \langle s \rangle_L^2$. Show that for $\tau_s < 2$, the variance also diverges as $L \to \infty$, and find the exponent. This divergence reflects the absence of a characteristic avalanche size — the defining feature of criticality.

**(d)** Derive the condition on $\tau_s$ for which $\langle s \rangle_L$ is finite as $L \to \infty$. Compare to the condition for a power-law distribution $P(s) \sim s^{-\tau_s}$ on $(1,\infty)$ to have a finite mean.

---

### Problem 5: Spectral Exponent from Avalanche Duration Distribution

Model the output signal $\phi(t)$ from an SOC system as a superposition of rectangular pulses, one per grain addition. The $k$-th avalanche produces a pulse of height $h_0$ and duration $T_k$, where $T_k$ is drawn i.i.d. from $P(T) \sim T^{-\tau_T}$ for $1 \leq \tau_T < 3$.

**(a)** Compute the Fourier transform of a single rectangular pulse $\phi_T(t) = h_0 \cdot \mathbf{1}[0 \leq t \leq T]$:
$$\hat{\phi}_T(f) = \int_{-\infty}^\infty \phi_T(t)\, e^{-2\pi i f t}\, dt$$
Show that $|\hat{\phi}_T(f)|^2 \approx h_0^2 T^2$ for $f \ll 1/T$ and $|\hat{\phi}_T(f)|^2 \ll h_0^2 T^2$ for $f \gg 1/T$.

**(b)** The aggregate power spectral density is:
$$S(f) \approx \int_1^{\infty} P(T)\, |\hat{\phi}_T(f)|^2\, dT$$
Split the integral at $T^* = 1/f$ and show that the dominant contribution comes from $T > 1/f$:
$$S(f) \sim \int_{1/f}^{\infty} T^{-\tau_T} \cdot h_0^2 T^2\, dT$$

**(c)** Evaluate the integral in (b) (assuming $\tau_T < 3$ so it converges) and show:
$$S(f) \sim f^{-(3 - \tau_T)}$$

**(d)** For what value of $\tau_T$ does this give exact $1/f$ noise ($\beta = 1$)? This is the mean-field value of the duration exponent; compare to the 2D BTW value $\tau_T \approx 1.37$ and compute the predicted spectral exponent.

**(e)** Identify one assumption in this derivation that may fail in practice for real SOC systems, and explain how it could affect the predicted $\beta$.

---

## Conceptual Questions

### Problem 6: SOC vs. Tuned Criticality

**(a)** In an Ising ferromagnet on a finite lattice, the correlation length $\xi \sim |T - T_c|^{-\nu}$ diverges only at $T = T_c$. Explain precisely what "fine-tuning" means here: what external agent controls $T$, and what happens if $T$ deviates slightly from $T_c$?

**(b)** In the BTW sandpile, identify the quantity that plays the role of the "control parameter" (analogous to $T$ in the Ising model). Explain how this quantity is determined by the dynamics rather than by an external agent. What prevents it from drifting away from the critical value?

**(c)** Some authors argue that SOC is not fundamentally different from tuned criticality — that the "self-tuning" is just a consequence of the feedback loop implemented by the drive/dissipation mechanism, which could in principle be replicated by an external controller. Evaluate this argument: does it undermine the conceptual novelty of SOC? Does the mechanism of tuning (internal feedback vs. external control) matter physically?

**(d)** Give one example of a system that exhibits power-law statistics but is definitely not at a second-order phase transition. Explain why power-law statistics alone are insufficient evidence for SOC.

---

### Problem 7: The Attractor Argument and Its Prerequisites

**(a)** The attractor argument (Section 3.1) claims that the subcritical state is not stationary because the output flux is less than the input flux. Make this precise: define "output flux" and "input flux" formally, and write a differential equation for $\frac{d\langle z \rangle}{dt}$ in terms of these fluxes.

**(b)** The argument requires both slow drive ($h \to 0$) and open boundaries (dissipation). Show by example or reasoning what goes wrong in each of the following degenerate limits:
  - (i) Periodic boundary conditions (no sink): the model with $h > 0$ constant
  - (ii) No drive ($h = 0$), open boundaries, starting from a random initial configuration

**(c)** The attractor argument describes the dynamics of $\langle z \rangle$, but criticality is a statement about fluctuations (power-law distributions), not just the mean. Explain why having $\langle z \rangle = \langle z \rangle_c$ does not by itself guarantee critical fluctuations, and what additional structure (e.g., the Abelian property) is needed to establish that the stationary measure has power-law tails.

---

### Problem 8: Interpreting the Finite-Size Scaling Function

The finite-size scaling ansatz $P(s, L) = s^{-\tau_s} g(s/L^D)$ encodes both the critical behavior and finite-size corrections in the single scaling function $g$.

**(a)** Sketch (qualitatively) the function $g : (0,\infty) \to (0,\infty)$ consistent with: $g(u) \to g_0 > 0$ for $u \to 0$, and $g(u) \to 0$ rapidly for $u \to \infty$. Describe two physically reasonable functional forms for the decay of $g(u)$ at large $u$ (e.g., exponential, Gaussian) and explain which would be more consistent with an exponential cutoff at $s \sim L^D$.

**(b)** Fix a system size $L$ and plot $\log P(s, L)$ vs. $\log s$ schematically. Identify: (i) the power-law regime, (ii) the crossover scale $s^* \sim L^D$, and (iii) the cutoff regime. How does the plot change as $L \to \infty$?

**(c)** Explain the data collapse technique: if you measure $P(s, L)$ for several system sizes $L_1 < L_2 < L_3$, how do you extract $\tau_s$ and $D$ from the data? What would a successful collapse of the rescaled plots $s^{\tau_s} P(s,L)$ vs. $s/L^D$ look like?

**(d)** The ansatz assumes a single relevant length scale $L$. In a real SOC system, are there other length scales that might complicate the collapse? Give one example.

---

### Problem 9: Universality Classes and Model Symmetries

The BTW sandpile (deterministic toppling) and the Manna model (stochastic toppling) both exhibit power-law avalanche statistics but belong to different universality classes.

**(a)** State precisely the toppling rule of the Manna model and identify the symmetry broken relative to the BTW model. Why does this change in symmetry lead to different critical exponents?

**(b)** In equilibrium critical phenomena, universality classes are determined by spatial dimension $d$, order-parameter symmetry, and range of interactions. Identify the three analogous determinants for SOC universality classes. For each, give an example of two models that differ in that determinant and have different exponents.

**(c)** The BTW sandpile is believed to have logarithmic corrections to power-law scaling in $d = 2$. Explain conceptually why logarithmic corrections arise at the upper critical dimension $d_c$ of a phase transition, and speculate on why the 2D BTW sandpile might be at or near its upper critical dimension.

**(d)** The Gutenberg-Richter law for earthquakes gives $\tau_E \approx 5/3$. The BTW sandpile gives $\tau_s \approx 1.20$ in 2D and $\tau_s = 3/2$ in mean field. Are these consistent with the earthquake data? What would need to be true about the earthquake fault system for the BTW mean-field exponent to apply?

---

### Problem 10: Empirical Power Laws and the SOC Attribution Problem

**(a)** List three mechanisms (other than SOC) that can generate power-law or heavy-tailed distributions in natural systems. For each, give a specific example and identify what distinguishes it from SOC.

**(b)** Neural avalanches in cortical tissue show $P(s) \sim s^{-3/2}$, consistent with a mean-field branching process at criticality. Describe two additional measurements (beyond the power-law exponent) that would strengthen or weaken the case for SOC in neural tissue.

**(c)** The Bak-Sneppen model of biological evolution produces power laws in extinction cascade sizes. Identify the slow drive and the dissipation mechanism in this model (by analogy to the BTW sandpile). What plays the role of "grains" and "toppling"?

**(d)** Power laws are often fit to empirical data using log-log linear regression. Explain two statistical pitfalls of this approach. What is the standard alternative (hint: maximum likelihood on the power-law family), and what does the Kolmogorov-Smirnov test measure in this context?

---

## Implementation Sketches

### Problem 11: BTW Sandpile Simulation

Sketch a complete algorithm for simulating the BTW sandpile on an $L \times L$ grid and measuring the avalanche size distribution.

**(a)** **Data structure.** Represent the height field as a 2D integer array $z[i][j]$ for $0 \leq i,j < L$. The boundary condition is open: sites with $i \in \{0, L-1\}$ or $j \in \{0, L-1\}$ have fewer than 4 neighbors; grains sent off-grid are discarded. Write pseudocode for a function `topple(z, i, j)` that performs one toppling at site $(i,j)$ and returns the number of grains dissipated.

**(b)** **Relaxation loop.** Write pseudocode for `relax(z)` that repeatedly topples any unstable site until the configuration is stable. Discuss two implementation strategies:
  - Naive scan: iterate over all sites in each pass, topple any unstable one, repeat until no unstable site remains. What is the worst-case time complexity per avalanche in terms of $L$ and avalanche size $s$?
  - Queue-based: maintain a queue of currently unstable sites; after each toppling, enqueue any newly destabilized neighbors. Why is this more efficient?

**(c)** **Main simulation loop.** Write pseudocode for the full simulation:
  1. Initialize $z$ to the all-zeros configuration (or random stable configuration).
  2. Repeat $N_{\text{burn}}$ times: add a grain at a random site, relax. (Burn-in to reach stationarity.)
  3. Repeat $N_{\text{measure}}$ times: add a grain at a random site, relax while recording $s$ (total topplings). Store $s$ in a list.

**(d)** **Estimating the power-law exponent.** Given a list of avalanche sizes $\{s_k\}$, outline an algorithm to estimate $\tau_s$ using the maximum likelihood estimator for a discrete power law with lower cutoff $s_{\min}$:
$$\hat{\tau}_s = 1 + n \left[ \sum_{k=1}^n \ln \frac{s_k}{s_{\min} - 1/2} \right]^{-1}$$
where $n$ is the number of samples with $s_k \geq s_{\min}$. Explain the role of $s_{\min}$ and how one would choose it.

---

### Problem 12: Dhar's Burning Algorithm

Sketch an algorithm implementing Dhar's burning algorithm to test whether a given stable configuration $\eta$ on an $L \times L$ grid (with a single sink at a corner, say) is recurrent.

**(a)** **Setup.** Represent the configuration as a 2D array $z[i][j]$ with $0 \leq z[i][j] \leq 3$, and a boolean array `burned[i][j]` initialized to `False`. The sink is a virtual vertex adjacent to all boundary sites; initialize it as burned.

**(b)** **Burning rule.** A site $(i,j)$ can be burned if it has at least as many grains as it has unburned non-sink neighbors. Formally: site $(i,j)$ is burnable if
$$z[i][j] \geq |\{(i',j') \in \mathcal{N}(i,j) : \text{burned}[i'][j'] = \texttt{False}\}|$$
Write pseudocode for a single pass that burns all currently burnable unburned sites.

**(c)** **Main loop.** Write pseudocode for the full algorithm using a queue:
  1. Initialize the queue with all boundary sites (which are adjacent to the sink and may be immediately burnable).
  2. Process each site in the queue: if burnable and unburned, mark as burned, then add its unburned neighbors to the queue for rechecking.
  3. After the queue empties, check whether all sites are burned.
  Return `True` iff all sites are burned.

**(d)** **Complexity.** What is the time complexity of this algorithm in terms of $|V| = L^2$? Compare to the naive algorithm that reruns the full pass until no new site is burned.

**(e)** **Application.** Explain how you would use this algorithm to: (i) verify that the stationary distribution of the BTW simulation from Problem 11 is supported only on recurrent configurations, and (ii) count the number of recurrent configurations for small $L$ and compare to $\det(\Delta)$.

---

### Problem 13: Scaling Exponent Estimation via Data Collapse

Sketch an algorithm to estimate the exponents $(\tau_s, D)$ of the finite-size scaling ansatz $P(s,L) = s^{-\tau_s} g(s/L^D)$ from simulation data.

**(a)** **Data collection.** Assume you have run $K$ system sizes $L_1 < L_2 < \cdots < L_K$ and for each size $L_k$ have collected $N_k$ avalanche sizes. Describe how to construct empirical histograms $\hat{P}(s, L_k)$ in log-spaced bins. Why use log-spaced rather than linearly-spaced bins?

**(b)** **Data collapse.** The rescaled quantity $Q(s, L) = s^{\tau_s} P(s, L)$ should, under the ansatz, collapse to a universal function of $u = s/L^D$:
$$Q(s, L) = g(s/L^D)$$
Sketch a grid-search algorithm over $(\tau_s, D)$ that minimizes a measure of collapse quality — for example, the mean squared deviation between $Q(s, L_k)$ and the mean curve $\bar{g}(u)$ computed from all sizes. What grid resolution is needed to detect exponent differences of $\pm 0.05$?

**(c)** **Collapse quality metric.** Propose a specific functional form for the collapse quality $\mathcal{Q}(\tau_s, D)$ in terms of the empirical data and the estimated mean curve $\bar{g}$. Discuss one advantage and one disadvantage of using least-squares vs. maximum likelihood as the objective.

**(d)** **Finite-size corrections.** The ansatz $P(s,L) = s^{-\tau_s} g(s/L^D)$ is exact only in the scaling limit. For small $L$, corrections of the form $P(s,L) = s^{-\tau_s}[g(s/L^D) + L^{-\omega} h(s/L^D) + \cdots]$ may be present ($\omega > 0$ is the correction-to-scaling exponent). How would you detect the presence of such corrections from your data, and how would they bias the estimate of $\tau_s$ and $D$ from the uncorrected collapse?
