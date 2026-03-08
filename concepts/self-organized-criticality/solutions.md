# Self-Organized Criticality: Solutions

## Table of Contents

1. [Derivation Problems](#derivation-problems)
   - [Solution 1: Abelian Property for a Two-Site System](#solution-1-abelian-property-for-a-two-site-system)
   - [Solution 2: Toppling Lemma via Monotonicity](#solution-2-toppling-lemma-via-monotonicity)
   - [Solution 3: Positive Definiteness and Invertibility of Δ](#solution-3-positive-definiteness-and-invertibility-of-delta)
   - [Solution 4: Mean Avalanche Size from Finite-Size Scaling](#solution-4-mean-avalanche-size-from-finite-size-scaling)
   - [Solution 5: Spectral Exponent from Avalanche Duration Distribution](#solution-5-spectral-exponent-from-avalanche-duration-distribution)
2. [Conceptual Questions](#conceptual-questions)
   - [Solution 6: SOC vs. Tuned Criticality](#solution-6-soc-vs-tuned-criticality)
   - [Solution 7: The Attractor Argument and Its Prerequisites](#solution-7-the-attractor-argument-and-its-prerequisites)
   - [Solution 8: Interpreting the Finite-Size Scaling Function](#solution-8-interpreting-the-finite-size-scaling-function)
   - [Solution 9: Universality Classes and Model Symmetries](#solution-9-universality-classes-and-model-symmetries)
   - [Solution 10: Empirical Power Laws and the SOC Attribution Problem](#solution-10-empirical-power-laws-and-the-soc-attribution-problem)
3. [Implementation Sketches](#implementation-sketches)
   - [Solution 11: BTW Sandpile Simulation](#solution-11-btw-sandpile-simulation)
   - [Solution 12: Dhar's Burning Algorithm](#solution-12-dhars-burning-algorithm)
   - [Solution 13: Scaling Exponent Estimation via Data Collapse](#solution-13-scaling-exponent-estimation-via-data-collapse)

---

## Derivation Problems

### Solution 1: Abelian Property for a Two-Site System

**(a)** The graph has vertices $\{1, 2, s\}$ with edges $\{1,2\}$, $\{1,s\}$, and $\{2,s\}$. Both vertices have $\deg(1) = \deg(2) = 2$. The toppling matrix is:

$$\Delta = \begin{pmatrix} \deg(1) & -[\text{edge between 1 and 2}] \\ -[\text{edge between 2 and 1}] & \deg(2) \end{pmatrix} = \begin{pmatrix} 2 & -1 \\ -1 & 2 \end{pmatrix}$$

**(b)** Start: $\eta = (2, 2)$. Both sites have $z_i = 2 = \deg(i)$, so both are unstable.

**Sequence A: topple site 1 first.**
- Topple site 1: $z_1 \to 2 - 2 = 0$; site 2 receives 1 grain: $z_2 \to 3$; sink receives 1 grain (lost). New state: $(0, 3)$.
- Now site 1 is stable ($z_1 = 0 < 2$). Site 2 is unstable ($z_2 = 3 \geq 2$). Topple site 2: $z_2 \to 3 - 2 = 1$; site 1 receives 1 grain: $z_1 \to 1$; sink receives 1 grain (lost). New state: $(1, 1)$.
- Both stable. Toppling vector: $\mathbf{n}^A = (1, 1)$.

**Sequence B: topple site 2 first.**
- Topple site 2: $z_2 \to 0$; site 1 receives 1 grain: $z_1 \to 3$; sink receives 1 grain. New state: $(3, 0)$.
- Site 2 stable, site 1 unstable. Topple site 1: $z_1 \to 1$; site 2 receives 1 grain: $z_2 \to 1$; sink receives 1 grain. New state: $(1,1)$.
- Both stable. Toppling vector: $\mathbf{n}^B = (1,1)$.

Both sequences end at $\eta' = (1,1)$ with $\mathbf{n} = (1,1)$.

**(c)** Algebraic verification: $\eta' = \eta - \Delta \mathbf{n}$:
$$\Delta \mathbf{n} = \begin{pmatrix}2&-1\\-1&2\end{pmatrix}\begin{pmatrix}1\\1\end{pmatrix} = \begin{pmatrix}1\\1\end{pmatrix}$$
$$\eta' = (2,2) - (1,1) = (1,1) \checkmark$$

$\det(\Delta) = 2 \cdot 2 - (-1)(-1) = 4 - 1 = 3 \neq 0$. Since $\Delta$ is invertible, $\Delta \mathbf{n} = \mathbf{c}$ has a unique solution $\mathbf{n} = \Delta^{-1}\mathbf{c}$ for any constant vector $\mathbf{c}$. In particular, if $\eta' - \eta = -\Delta \mathbf{n}^\alpha = -\Delta \mathbf{n}^\beta$, then $\Delta(\mathbf{n}^\alpha - \mathbf{n}^\beta) = \mathbf{0}$, and invertibility forces $\mathbf{n}^\alpha = \mathbf{n}^\beta$.

---

### Solution 2: Toppling Lemma via Monotonicity

**(a)** Let $S = (i_1, \ldots, i_N)$ be a legal toppling sequence for $\eta$: when site $i_k$ is toppled at step $k$, the current height $z_{i_k} \geq \deg(i_k)$. After steps $1, \ldots, k-1$, the height of $i_k$ in both $\eta$ and $\eta + \mathbf{e}_j$ is the same, except possibly at $j$ which has one extra grain. Therefore, if $z_{i_k} \geq \deg(i_k)$ under $\eta$, then $(z + e_j)_{i_k} \geq z_{i_k} \geq \deg(i_k)$ under $\eta + \mathbf{e}_j$ as well (since the extra grain at $j$ can only increase heights). The sequence $S$ remains legal for $\eta + \mathbf{e}_j$.

**(b)** Since the sequence $S$ is legal for $\eta + \mathbf{e}_j$, the system can perform at least $n_i(\eta)$ topplings at each site $i$ starting from $\eta + \mathbf{e}_j$. By the Abelian property, all stabilizing sequences produce the same toppling vector. Therefore $n_i(\eta + \mathbf{e}_j) \geq n_i(\eta)$ for all $i$.

**(c)** The analogous statement $n_i(\eta - \mathbf{e}_j) \leq n_i(\eta)$ does hold (by the symmetric argument: removing a grain can only decrease heights, so a sequence legal for $\eta - \mathbf{e}_j$ is also legal for $\eta$, giving $n_i(\eta) \geq n_i(\eta - \mathbf{e}_j)$). However, care is needed: $\eta - \mathbf{e}_j$ must still be a valid configuration (i.e., $\eta_j \geq 1$), and if $\eta - \mathbf{e}_j$ is already stable, then $n_i(\eta - \mathbf{e}_j) = 0$ while $n_i(\eta)$ may be positive.

**(d)** The Green's function argument. For a recurrent configuration $\eta$, adding $\mathbf{e}_j$ and stabilizing gives toppling vector $\mathbf{n}(\eta, j) := \mathbf{n}(\eta + \mathbf{e}_j)$. We claim $G_{ij} := n_i(\eta + \mathbf{e}_j) - n_i(\eta)$ is independent of $\eta$ in the recurrent class. From $\eta' = \eta - \Delta \mathbf{n}(\eta)$ and $(\eta + \mathbf{e}_j)' = \eta + \mathbf{e}_j - \Delta \mathbf{n}(\eta + \mathbf{e}_j)$, both final configurations lie in $\mathcal{S}$. Subtracting: $\mathbf{e}_j - \Delta(\mathbf{n}(\eta+\mathbf{e}_j) - \mathbf{n}(\eta)) = \eta'' - \eta'$ where $\eta'', \eta'$ are both recurrent stable configurations. In the recurrent class, $\eta'' - \eta' = \Delta \mathbf{g}$ for some integer vector $\mathbf{g}$ (this follows from the group structure of $\mathcal{R}$), giving $\Delta \mathbf{G}_j = \mathbf{e}_j - \Delta \mathbf{g}$, i.e., $\mathbf{G}_j = \Delta^{-1} \mathbf{e}_j - \mathbf{g}$. The integer part $\mathbf{g}$ adjusts for the particular recurrent configuration; the fractional part $\Delta^{-1}\mathbf{e}_j$ is universal. The key result is that $G_{ij} = (\Delta^{-1})_{ij}$ is well-defined and independent of $\eta$, and $\Delta^{-1}$ plays the role of a discrete Green's function (lattice inverse Laplacian).

---

### Solution 3: Positive Definiteness and Invertibility of $\Delta$

**(a)** Write the quadratic form using the definition $\Delta_{ij} = \deg(i)\delta_{ij} - A_{ij}$ where $A_{ij}$ is the adjacency between $i,j \in V$:
$$\mathbf{x}^\top \Delta \mathbf{x} = \sum_i x_i^2 \deg(i) - \sum_{\{i,j\} \in E, i,j \in V} 2x_i x_j$$

Decompose $\deg(i) = d_V(i) + d_s(i)$ where $d_V(i)$ = number of neighbors in $V$, $d_s(i)$ = number of edges to sink. Then:
$$\mathbf{x}^\top \Delta \mathbf{x} = \sum_i x_i^2 d_V(i) + \sum_i x_i^2 d_s(i) - \sum_{\{i,j\} \in E, i,j \in V} 2x_i x_j$$

Using $\sum_{\{i,j\} \in E, i,j\in V} (x_i - x_j)^2 = \sum_i x_i^2 d_V(i) - 2\sum_{\{i,j\}} x_i x_j$ (each edge $\{i,j\}$ contributes $x_i^2 + x_j^2 - 2x_ix_j$):

$$\mathbf{x}^\top \Delta \mathbf{x} = \sum_{\{i,j\} \in E,\ i,j \in V} (x_i - x_j)^2 + \sum_{i \in V} d_s(i) x_i^2$$

Since each edge $\{i,s\}$ contributes $x_i^2$ (the sink has $x_s \equiv 0$):
$$= \sum_{\{i,j\} \in E,\ i,j \in V} (x_i - x_j)^2 + \sum_{\substack{i \in V \\ \{i,s\} \in E}} x_i^2 \qquad \square$$

**(b)** If $\mathbf{x}^\top \Delta \mathbf{x} = 0$, then both sums are zero. From the second sum: $x_i = 0$ for all $i$ adjacent to $s$. From the first sum: $x_i = x_j$ for all edges $\{i,j\}$ within $V$. Since $G$ is connected, all vertices in $V$ are reachable from any vertex adjacent to $s$ via paths within $V$. Starting from $x_i = 0$ for a neighbor of $s$, connectedness propagates $x_i = 0$ to all of $V$. Hence $\mathbf{x} = \mathbf{0}$.

**(c)** Positive definiteness ($\mathbf{x}^\top \Delta \mathbf{x} > 0$ for $\mathbf{x} \neq \mathbf{0}$) implies $\ker(\Delta) = \{\mathbf{0}\}$, hence $\Delta$ is invertible over $\mathbb{R}$. For the Abelian property: two stabilizing sequences give the same toppling vector because $\Delta(\mathbf{n}^\alpha - \mathbf{n}^\beta) = \mathbf{0}$ and invertibility forces $\mathbf{n}^\alpha = \mathbf{n}^\beta$. Invertibility is therefore the algebraic foundation of the abelian property.

**(d)** The full graph Laplacian $L$ on $V \cup \{s\}$ satisfies $L \mathbf{1} = \mathbf{0}$ (rows sum to zero), so $\mathbf{1}$ is in the kernel — $L$ is only positive semidefinite. When we delete the row and column for $s$, we are effectively setting $x_s = 0$ (grounding at the sink). The quadratic form $\mathbf{x}^\top L \mathbf{x} = \sum_{\{i,j\}\in E} (x_i - x_j)^2$ (summing over all edges including those to $s$, where $x_s = 0$) then includes the term $x_i^2$ for each $i$ adjacent to $s$. This grounds the boundary values and eliminates the flat direction $\mathbf{x} \propto \mathbf{1}$, upgrading semidefiniteness to definiteness.

---

### Solution 4: Mean Avalanche Size from Finite-Size Scaling

**(a)** Compute $\langle s \rangle_L = \int_1^\infty s \cdot P(s,L)\,ds$. Substituting $u = s/L^D$, so $s = u L^D$ and $ds = L^D\,du$:

$$\langle s \rangle_L = \int_{L^{-D}}^\infty (uL^D)(uL^D)^{-\tau_s} g(u)\, L^D\, du = L^{D(2-\tau_s)} \int_{L^{-D}}^{\infty} u^{1-\tau_s} g(u)\,du$$

For large $L$, the lower limit $L^{-D} \to 0$, so the integral converges to $C := \int_0^\infty u^{1-\tau_s} g(u)\,du$, which is finite for $\tau_s < 2$ (the integrand $u^{1-\tau_s}$ is integrable near zero since $1 - \tau_s > -1$ when $\tau_s < 2$, and $g(u)$ decays rapidly for large $u$). Therefore:

$$\langle s \rangle_L \sim C \cdot L^{D(2-\tau_s)}$$

**(b)** For 2D BTW: $\tau_s \approx 1.20$, $D \approx 2.75$:
$$D(2 - \tau_s) = 2.75 \times (2 - 1.20) = 2.75 \times 0.80 = 2.20$$
The mean avalanche size grows as $\langle s \rangle_L \sim L^{2.20}$, faster than the system area $L^2$. This seems surprising but is consistent: the distribution has a very heavy tail ($\tau_s < 2$), so rare large avalanches dominate the mean.

**(c)** For the second moment: $\langle s^2 \rangle_L \sim L^{D(3-\tau_s)}$ by the same substitution (integrand $u^{2-\tau_s}$, finite for $\tau_s < 3$). Since $\tau_s < 2 < 3$, this converges, and the variance $\langle s^2 \rangle_L - \langle s \rangle_L^2 \sim L^{D(3-\tau_s)}$ (dominant term). For 2D BTW: $D(3 - \tau_s) = 2.75 \times 1.80 = 4.95$. The variance diverges as $L^{4.95}$, confirming no characteristic scale.

**(d)** $\langle s \rangle_L$ is finite as $L \to \infty$ iff the integral $\int_0^\infty u^{1-\tau_s} g(u)\,du$ is finite. Near $u = 0$: convergent iff $1 - \tau_s > -1$, i.e., $\tau_s < 2$. In the infinite system ($L = \infty$): $P(s) \sim s^{-\tau_s}$ for all $s$, and $\langle s \rangle = \int_1^\infty s \cdot s^{-\tau_s}\,ds \sim \int_1^\infty s^{1-\tau_s}\,ds$ is finite iff $1 - \tau_s < -1$, i.e., $\tau_s > 2$. The two conditions are complementary: when $\tau_s < 2$, the infinite-system mean diverges but finite-size scaling gives a finite (size-dependent) mean; when $\tau_s > 2$, the infinite-system mean is finite and $\langle s \rangle_L \to \text{const}$ as $L \to \infty$.

---

### Solution 5: Spectral Exponent from Avalanche Duration Distribution

**(a)** The Fourier transform of the rectangular pulse $\phi_T(t) = h_0 \cdot \mathbf{1}[0 \leq t \leq T]$ is:

$$\hat{\phi}_T(f) = h_0 \int_0^T e^{-2\pi i f t}\,dt = h_0 \frac{1 - e^{-2\pi i f T}}{2\pi i f} = h_0 T e^{-\pi i f T} \cdot \text{sinc}(\pi f T)$$

where $\text{sinc}(x) = \sin(x)/x$. The power is:

$$|\hat{\phi}_T(f)|^2 = h_0^2 T^2 \text{sinc}^2(\pi f T)$$

For $fT \ll 1$: $\text{sinc}(\pi fT) \approx 1$, so $|\hat{\phi}_T(f)|^2 \approx h_0^2 T^2$.

For $fT \gg 1$: $|\text{sinc}(\pi fT)| \leq 1/(\pi fT) \to 0$, so $|\hat{\phi}_T(f)|^2 \sim h_0^2/(\pi f)^2 \ll h_0^2 T^2$.

**(b)** Split the integral at $T^* = 1/f$:

$$S(f) \approx \int_1^{1/f} T^{-\tau_T} \cdot h_0^2 T^2\,dT + \int_{1/f}^{\infty} T^{-\tau_T} \cdot h_0^2 T^2 \cdot \text{sinc}^2(\pi f T)\,dT$$

The second term dominates for $f \ll 1$ because: in the first term, the integrand is $T^{2-\tau_T}$ but the range is short; in the second term, though $\text{sinc}^2(\pi fT) < 1$, when $T \gg 1/f$ (so $fT \gg 1$), $\text{sinc}^2 \sim 1/(\pi fT)^2$ and the contribution $\sim T^{-\tau_T}$ — summing these over the long tail $T > 1/f$ gives the dominant contribution. Approximating $\text{sinc}^2(\pi f T) \approx 1$ for $T$ not much larger than $1/f$ (the leading contribution), we use the zeroth-order approximation:

$$S(f) \sim h_0^2 \int_{1/f}^{\infty} T^{2-\tau_T}\,dT$$

**(c)** Evaluate:

$$S(f) \sim h_0^2 \int_{1/f}^{\infty} T^{2-\tau_T}\,dT = h_0^2 \left[\frac{T^{3-\tau_T}}{3-\tau_T}\right]_{1/f}^{\infty}$$

For $\tau_T < 3$, the integral converges because the upper limit is zero (the exponent $3 - \tau_T > 0$ so $T^{3-\tau_T} \to \infty$ as $T \to \infty$ — wait, this diverges). More carefully: the integral $\int_{1/f}^\infty T^{2-\tau_T}\,dT$ diverges unless $2 - \tau_T < -1$, i.e., $\tau_T > 3$. For $1 < \tau_T < 3$, the integral is actually cut off by the finite system size (the avalanche duration cannot exceed $L^z$). Keeping this cutoff:

$$S(f) \sim h_0^2 \int_{1/f}^{L^z} T^{2-\tau_T}\,dT = h_0^2 \frac{(L^z)^{3-\tau_T} - f^{-(3-\tau_T)}}{3-\tau_T}$$

For $f \gg L^{-z}$ (frequencies above the system cutoff), the dominant term is $\sim f^{-(3-\tau_T)}$:

$$\boxed{S(f) \sim f^{-(3-\tau_T)}}$$

**(d)** $\beta = 1$ requires $3 - \tau_T = 1$, i.e., $\tau_T = 2$. This is the mean-field duration exponent (branching process at criticality). For 2D BTW: $\tau_T \approx 1.37$, giving $\beta = 3 - 1.37 = 1.63$. This is "reddish" noise, steeper than $1/f$ but not Brownian.

**(e)** A key assumption is that avalanches are independent rectangular pulses, so $S(f)$ is given by the average of individual pulse spectra. In reality, if grains are added slowly but not infinitely slowly, successive avalanche durations may be correlated (a long avalanche depletes the system and is followed by a period of small avalanches). These temporal correlations would modify $S(f)$ beyond the single-pulse model. Additionally, the "rectangular pulse" shape is an idealization; real avalanche activity profiles may have complex temporal structure (e.g., a roughly triangular profile), altering $|\hat{\phi}_T(f)|^2$ at frequencies $f \sim 1/T$.

---

## Conceptual Questions

### Solution 6: SOC vs. Tuned Criticality

**(a)** In the Ising model, temperature $T$ is an external control parameter set by coupling the system to a heat bath at temperature $T$. At $T_c$, the free energy is non-analytic and $\xi \to \infty$. If $T = T_c + \epsilon$ for small $\epsilon \neq 0$: $\xi \sim |\epsilon|^{-\nu}$ is large but finite, critical power laws are cut off at length scale $\xi$, and the system is generically off-critical. Fine-tuning means: the external agent (experimentalist, heat bath) must set $T$ to the precise value $T_c$ within experimental resolution.

**(b)** The analogous control parameter in the BTW sandpile is the average height $\langle z \rangle$, or equivalently, the overall grain density. In the Ising analogy: $\langle z \rangle$ plays the role of $T$, and $\langle z \rangle_c$ is the critical density. The key difference: $\langle z \rangle$ is not set externally — it evolves under the dynamics. The feedback mechanism is the competition between input flux (slow drive: +1 grain/step) and output flux (boundary dissipation: proportional to avalanche activity, which increases with $\langle z \rangle$). This implements an implicit proportional control loop that cannot drift away from $\langle z \rangle_c$ because any deviation is self-correcting.

**(c)** The argument that SOC is "just" internally implemented tuning has merit at a formal level — any feedback control system can be described as either internal or external depending on where we draw the system boundary. However, the distinction matters physically: in SOC, the feedback is intrinsic to the microdynamics (toppling + boundary dissipation) and emerges from the local rules without a separate control layer. This is more parsimonious as an explanation for why complex systems in nature (earthquakes, neural activity) exhibit criticality — they do not require an external Maxwell's demon tuning parameters. Whether one calls this "genuinely different" from tuned criticality is partly philosophical, but the phenomenological implication (criticality without external fine-tuning) is a substantive and falsifiable claim.

**(d)** Example: a scale-free network generated by preferential attachment (Barabasi-Albert model) has a degree distribution $P(k) \sim k^{-3}$. This is a genuine power law, but it arises from a growth process (older nodes accumulate disproportionately more links), not from any phase transition. There is no correlation length divergence, no critical slowing down, and no associated scaling theory with universality classes. Power laws can also arise from: (i) log-normal distributions with large variance (mimic power laws over finite ranges), (ii) mixtures of exponential distributions with power-law-distributed rate parameters, (iii) Zipf's law from optimization or information-theoretic constraints.

---

### Solution 7: The Attractor Argument and Its Prerequisites

**(a)** Define:
- Input flux: $J_{\text{in}} = h$ (grains per unit time, constant under slow drive)
- Output flux: $J_{\text{out}}(\langle z \rangle)$ = average number of grains dissipated per unit time, a monotone increasing function of $\langle z \rangle$ (higher density → larger avalanches → more boundary dissipation)

Mass balance:
$$\frac{d\langle z \rangle}{dt} = \frac{J_{\text{in}} - J_{\text{out}}(\langle z \rangle)}{|\Lambda|}$$

At the fixed point: $J_{\text{out}}(\langle z \rangle_c) = J_{\text{in}} = h$. Since $J_{\text{out}}$ is increasing, this fixed point is stable: if $\langle z \rangle > \langle z \rangle_c$, then $J_{\text{out}} > J_{\text{in}}$ and $d\langle z \rangle/dt < 0$.

**(b)**
- (i) **Periodic boundaries, constant $h > 0$**: No sink, so $J_{\text{out}} = 0$ — grains can never leave. $\frac{d\langle z \rangle}{dt} = h/|\Lambda| > 0$ always. The average height increases without bound; the system is always supercritical and never reaches a stationary state. SOC requires a drain.
- (ii) **No drive, $h = 0$, open boundaries**: $J_{\text{in}} = 0$. Starting from an arbitrary configuration, any unstable sites topple and dissipate grains through the boundary. The system monotonically loses grains ($\frac{d\langle z \rangle}{dt} \leq 0$) and reaches a stable absorbing state with $\langle z \rangle < \langle z \rangle_c$. Without a drive to replenish grains, the system freezes in a subcritical absorbing state — it cannot sustain criticality.

**(c)** Having $\langle z \rangle = \langle z \rangle_c$ guarantees that the system is at the fixed point of the flux balance equation, but says nothing about the shape of the stationary distribution over configurations. In principle, the stationary measure could be concentrated on a few typical configurations with short-range correlations, producing exponentially decaying avalanche distributions. The power-law statistics arise from the specific structure of the stationary measure $\mu^*$ — specifically from the algebraic structure of the recurrent configurations (characterized by Dhar's burning algorithm) and the way adding a grain propagates through the recurrent set. The Abelian property ensures that the stationary measure is uniform over recurrent configurations (by ergodicity and the group structure of $\mathcal{R}$), and the power laws emerge from the fractal structure of the recurrent class under the addition operator. Thus, the attractor argument explains self-tuning but not the power laws; the latter require the full Abelian sandpile theory.

---

### Solution 8: Interpreting the Finite-Size Scaling Function

**(a)** The function $g(u)$ must satisfy $g(u) \to g_0$ for $u \ll 1$ (power-law regime) and $g(u) \to 0$ for $u \gg 1$ (cutoff regime). Physically reasonable forms:
- **Exponential cutoff**: $g(u) = g_0 \exp(-u/u_0)$ for some $u_0 = O(1)$. Straightforward and often used in numerical fits.
- **Stretched exponential**: $g(u) = g_0 \exp(-u^\delta)$ for $\delta > 0$. Allows for sharper or softer cutoffs.

Both are consistent with an exponential cutoff at $s \sim L^D$. The exponential form is preferred because: the cutoff in finite systems comes from the finite number of sites ($s$ cannot exceed $L^2$ since each site topples at most $O(1)$ times per avalanche on average), suggesting an exponential rather than algebraic suppression at the cutoff scale.

**(b)** Schematic log-log plot of $P(s,L)$ vs $s$:
- For $1 \ll s \ll L^D$ (regime (i)): $\log P = -\tau_s \log s + \text{const}$, a straight line of slope $-\tau_s$.
- At $s \approx s^* = L^D$ (regime (ii)): crossover — the line bends downward. The crossover sharpens as $L$ increases.
- For $s \gg L^D$ (regime (iii)): $P(s,L)$ drops to zero exponentially fast.
As $L \to \infty$: the crossover scale $s^* \to \infty$ and the distribution approaches a pure power law $P(s) \sim s^{-\tau_s}$ for all $s$.

**(c)** Data collapse procedure:
1. For each system size $L_k$, compute the empirical histogram $\hat{P}(s, L_k)$ in log-spaced bins.
2. Compute the rescaled quantity $Q_k(s) = s^{\tau_s} \hat{P}(s, L_k)$ for trial values $(\tau_s, D)$.
3. Plot $Q_k$ vs. $u_k(s) = s/L_k^D$ for all $k$ on the same axes.
4. A successful collapse means all curves $k = 1, \ldots, K$ lie on a single master curve $g(u)$, regardless of $k$.
The best $(\tau_s, D)$ are those for which the collapse is tightest (minimum spread). Visual collapse is qualitative; a quantitative measure is the sum of squared deviations between each curve and the mean.

**(d)** Other length scales that could complicate collapse:
- **Lattice spacing**: for small $s$, the discrete lattice structure is visible and deviates from the continuum ansatz. This means the ansatz applies only for $s \gg 1$, motivating the use of $s_{\min} > 1$.
- **Correlation length of the initial configuration**: if the system has not fully equilibrated (insufficient burn-in), the initial transient introduces an artificial length scale. The cure is sufficient burn-in.
- **Correction-to-scaling length**: finite-$L$ corrections of order $L^{-\omega}$ introduce a secondary length scale; these become apparent as systematic deviations from the master curve for small $L_k$.

---

### Solution 9: Universality Classes and Model Symmetries

**(a)** In the Manna model on $\mathbb{Z}^2$: when site $i$ topples (i.e., $z_i \geq 2$ for the 2-grain threshold version), it sends 2 grains to neighbors chosen uniformly at random (with replacement or without, depending on variant). The symmetry broken relative to BTW: **stochasticity of toppling direction**. In BTW, the toppling rule deterministically sends one grain to each of the 4 neighbors; in Manna, the two grains are sent to randomly chosen (possibly repeated) neighbors. This breaks the $\mathbb{Z}_4$ symmetry of the lattice (each neighbor is no longer guaranteed to receive exactly one grain) and introduces stochastic fluctuations in the propagation of activity. The resulting universality class (Manna class) has $\tau_s^{\text{Manna}} \approx 1.28$, distinct from BTW's $\tau_s \approx 1.20$ in 2D.

**(b)** Analogues of the three equilibrium determinants:
- **Spatial dimension $d$**: BTW in $d=2$ ($\tau_s \approx 1.20$) vs. mean-field ($d \geq 4$, $\tau_s = 3/2$).
- **Toppling rule symmetry (deterministic vs. stochastic)**: BTW (deterministic, $\tau_s \approx 1.20$) vs. Manna (stochastic, $\tau_s \approx 1.28$) — both in $d=2$.
- **Conservation law**: locally conserved BTW (grains conserved except at boundary) vs. Olami-Feder-Christensen model (OFC, non-conserved topplings) — OFC shows different and debated scaling.

**(c)** Logarithmic corrections arise when the system is at its upper critical dimension $d_c$: at $d < d_c$, fluctuations are relevant (Wilson-Fisher fixed point, non-mean-field exponents); at $d = d_c$, the RG fixed point is marginal — the leading power law is correct but receives logarithmic multiplicative corrections (e.g., $P(s) \sim s^{-\tau_s} (\log s)^{\hat{\tau}}$). For the BTW sandpile, the exact abelian structure and known results on special graphs suggest $d_c = 4$ (same as directed percolation or mean-field branching process); in $d = 2$, the logarithmic corrections are believed to arise from the proximity to this upper critical dimension modulated by the exact abelian symmetry, which constrains the fixed point structure in a way that ordinary stochastic sandpiles (Manna) are not constrained. The logarithmic corrections make numerical determination of BTW exponents particularly difficult.

**(d)** Gutenberg-Richter gives $\tau_E \approx 5/3 \approx 1.67$. BTW mean field gives $\tau_s = 3/2 = 1.50$; 2D BTW gives $\tau_s \approx 1.20$. Neither agrees with the earthquake data precisely. For mean-field BTW to apply, the earthquake fault network would need to be high-dimensional (effectively $d \geq 4$) or have a topology equivalent to a complete graph (all faults interacting with all others at equal strength). This is unrealistic geophysically. The discrepancy suggests that earthquake fault systems are not simply described by the BTW universality class — more detailed models incorporating realistic fault geometry, viscoelastic loading, and heterogeneous stress distributions are needed.

---

### Solution 10: Empirical Power Laws and the SOC Attribution Problem

**(a)** Three non-SOC mechanisms for power laws:
- **Preferential attachment (Barabasi-Albert)**: rich-get-richer network growth produces degree distribution $P(k) \sim k^{-3}$. Example: the World Wide Web link structure. No phase transition involved; the power law reflects growth dynamics.
- **Multiplicative noise (log-normal approximation)**: if a quantity evolves as $X_{t+1} = X_t \cdot \xi_t$ where $\xi_t$ is a multiplicative noise, the long-time distribution of $X$ is approximately log-normal, which mimics a power law over intermediate ranges. Example: firm size distribution (Gibrat's law).
- **Mixture of exponentials**: if the relaxation time $\tau$ of a process is itself distributed as $P(\tau) \sim \tau^{-1}$ (log-uniform), then the superposition of exponential relaxation processes with these rates produces $1/f$ noise. No critical dynamics is required. Example: certain electronic noise sources.

**(b)** Additional measurements beyond the power-law exponent:
- **Duration exponent and scaling relation**: measure $P(T)$ for neural avalanche durations. SOC predicts $\tau_T$ consistent with $\tau_T = 1 + \sigma(\tau_s - 1)$ (the scaling relation). An independent measurement of $\tau_T$ that satisfies the relation strengthens the SOC case.
- **Spatial fractal dimension**: measure the area $a$ of neural avalanches and test for $P(a) \sim a^{-\tau_a}$ with $\tau_a$ consistent with the scaling relation $\tau_a = 1 + (D/d_f)(\tau_s - 1)$. Additionally, direct imaging of the spatial footprint could reveal fractal (non-space-filling) geometry.

**(c)** Bak-Sneppen model analogies to BTW:
- **Grains**: fitness values of species. The slowest-driven analogy is that the species with the lowest fitness is selected for replacement.
- **Toppling**: when the least-fit species is replaced (drawn to a new random fitness), it "destabilizes" its neighbors in the ecological interaction network, who may also become the new minimum-fitness species, propagating the cascade (extinction event).
- **Slow drive**: the process of natural selection iteratively replacing the weakest species, analogous to the slow grain addition.
- **Dissipation**: extinction cascades eventually terminate (the cascade "dissipates") when all neighboring fitnesses are above threshold — analogous to grains leaving the boundary.

**(d)** Statistical pitfalls of log-log linear regression for power laws:
- **Binning artifacts**: the slope estimate depends on bin width and bin placement; log-spaced bins reduce bias but introduce their own artifacts (bins contain different numbers of samples).
- **Systematic bias near the cutoff**: including data near the upper cutoff (where the true distribution deviates from a power law) biases the slope estimate. Fitting on log-log scale treats all data points equally without proper likelihood weighting.

The standard alternative is the **maximum likelihood estimator (MLE)** for the continuous power law $P(x) = (\tau-1)x_{\min}^{\tau-1} x^{-\tau}$ for $x \geq x_{\min}$:
$$\hat{\tau} = 1 + n\left[\sum_{k=1}^n \ln \frac{x_k}{x_{\min}}\right]^{-1}$$
This has the minimum variance among unbiased estimators (Cramér-Rao bound) and does not require binning. The **Kolmogorov-Smirnov (KS) test** measures the maximum absolute difference between the empirical CDF and the theoretical CDF of the fitted power law: $D_{\text{KS}} = \sup_{x \geq x_{\min}} |F_{\text{emp}}(x) - F_{\text{theory}}(x)|$. A small $D_{\text{KS}}$ (and large $p$-value from bootstrapping) indicates the power-law hypothesis is not rejected by the data.

---

## Implementation Sketches

### Solution 11: BTW Sandpile Simulation

**(a)** Pseudocode for toppling at site $(i,j)$ on an $L \times L$ grid:

```
function topple(z, i, j):
    neighbors = [(i-1,j), (i+1,j), (i,j-1), (i,j+1)]
    valid_neighbors = [(r,c) for (r,c) in neighbors if 0 <= r < L and 0 <= c < L]
    grains_dissipated = 4 - len(valid_neighbors)  // lost at boundary
    z[i][j] -= 4
    for (r,c) in valid_neighbors:
        z[r][c] += 1
    return grains_dissipated
```

The function returns the number of grains sent to the sink (dissipated).

**(b)** Two relaxation strategies:

**Naive scan:**
```
function relax_naive(z):
    total_topplings = 0
    changed = True
    while changed:
        changed = False
        for i in 0..L-1:
            for j in 0..L-1:
                if z[i][j] >= 4:
                    topple(z, i, j)
                    total_topplings += 1
                    changed = True
    return total_topplings
```
Worst-case complexity: $O(L^2 \cdot s)$ per avalanche — each pass costs $O(L^2)$ and there may be $O(s)$ passes if topplings propagate slowly through the grid.

**Queue-based (optimal):**
```
function relax_queue(z):
    Q = Queue()
    for i,j: if z[i][j] >= 4: Q.enqueue((i,j))
    total_topplings = 0
    while Q not empty:
        (i,j) = Q.dequeue()
        if z[i][j] >= 4:  // recheck (may have changed)
            topple(z, i, j)
            total_topplings += 1
            for (r,c) in neighbors(i,j):
                if z[r][c] >= 4: Q.enqueue((r,c))
    return total_topplings
```
Complexity: $O(s)$ per avalanche (each toppling is processed exactly once). This is optimal since we must at least record each toppling.

**(c)** Full simulation:
```
// Initialize
z = L×L array of zeros
Q = Queue()

// Burn-in
for t = 1 to N_burn:
    (i,j) = uniform_random(0..L-1, 0..L-1)
    z[i][j] += 1
    if z[i][j] >= 4: Q.enqueue((i,j))
    relax_queue(z, Q)  // relax in place

// Measurement
sizes = []
for t = 1 to N_measure:
    (i,j) = uniform_random(0..L-1, 0..L-1)
    z[i][j] += 1
    if z[i][j] >= 4: Q.enqueue((i,j))
    s = relax_queue(z, Q)  // returns total topplings
    sizes.append(s)
```

**(d)** MLE for the power-law exponent on discrete data with lower cutoff $s_{\min}$:
```
function estimate_tau(sizes, s_min):
    filtered = [s for s in sizes if s >= s_min]
    n = len(filtered)
    log_sum = sum(log(s / (s_min - 0.5)) for s in filtered)
    tau_hat = 1 + n / log_sum
    return tau_hat
```
The factor $s_{\min} - 0.5$ is the standard correction for discrete distributions (the continuous MLE uses $s_{\min}$ exactly; for integer-valued data, subtracting $0.5$ gives a bias-corrected estimator). The choice of $s_{\min}$ involves a tradeoff: too small includes the non-power-law regime near the lattice scale (bias); too large reduces $n$ (variance). Standard practice: iterate over $s_{\min} \in \{1, 2, \ldots, s_{\max}/10\}$, compute the KS statistic for each, and select the $s_{\min}$ minimizing the KS distance.

---

### Solution 12: Dhar's Burning Algorithm

**(a)** Data structures:
```
z[i][j]       // height array, 0 <= z[i][j] <= 3
burned[i][j]  // boolean, initially False
// The sink is a virtual node adjacent to all boundary sites; treat as burned from the start.
```
The sink being "burned" encodes that grains can flow there freely. A boundary site $(i,j)$ (with $i \in \{0,L-1\}$ or $j \in \{0,L-1\}$) has fewer than 4 grid neighbors; the missing neighbors are implicitly the sink.

**(b)** Burning rule — site $(i,j)$ is burnable if:
```
unburned_neighbors = count of (r,c) in neighbors(i,j) with burned[r][c] == False
z[i][j] >= unburned_neighbors
```
Note: for boundary sites, `neighbors(i,j)` counts only grid neighbors (not the sink), which is already "burned" — so the effective unburned neighbor count is smaller, making boundary sites easier to burn. This is correct: a corner site with $z = 0$ and 2 unburned neighbors is not burnable, but if both neighbors are burned, it is trivially burnable.

**(c)** Queue-based burning algorithm:
```
function is_recurrent(z):
    burned = L×L array of False
    Q = Queue()

    // Initialize: boundary sites may be immediately burnable
    for each boundary site (i,j):
        if z[i][j] >= count_unburned_grid_neighbors(i,j, burned):
            Q.enqueue((i,j))

    while Q not empty:
        (i,j) = Q.dequeue()
        if burned[i][j]: continue  // already processed
        unburned_nbrs = count_unburned_grid_neighbors(i,j, burned)
        if z[i][j] >= unburned_nbrs:
            burned[i][j] = True
            for (r,c) in grid_neighbors(i,j):
                if not burned[r][c]:
                    Q.enqueue((r,c))  // recheck after neighbor was burned

    return all(burned[i][j] for all i,j)
```

**(d)** Complexity: each site is enqueued and dequeued at most $O(\deg_{\max})$ times (once per neighbor that burns). For the $L \times L$ grid with $\deg_{\max} = 4$: total work is $O(4 L^2) = O(L^2)$. This is optimal (we must check each site). The naive algorithm (re-scan entire grid after each burn event) has $O(L^4)$ worst-case complexity (up to $L^2$ burn events, each requiring an $O(L^2)$ scan), far worse.

**(e)** Applications:
- (i) **Verifying stationary support**: for a sample of configurations $\eta_1, \ldots, \eta_M$ from the simulation after burn-in, run `is_recurrent(eta_k)` on each. If the simulation has converged, all should return `True`. A non-recurrent configuration indicates insufficient burn-in or a bug in the toppling rule.
- (ii) **Counting recurrent configurations**: for small $L$ (e.g., $L \leq 5$), enumerate all stable configurations $\{0,1,2,3\}^{L^2}$ and count the recurrent ones. Compare to $\det(\Delta)$ (computable as the determinant of the $L^2 \times L^2$ integer matrix). For the $2 \times 2$ grid with one sink at a corner: $|\mathcal{R}| = \det(\Delta)$ should match. This provides a concrete numerical verification of Dhar's identity.

---

### Solution 13: Scaling Exponent Estimation via Data Collapse

**(a)** For each system size $L_k$, construct a histogram with log-spaced bins: divide the range $[s_{\min}, s_{\max}(L_k)]$ into $B$ bins $[b_\ell, b_{\ell+1})$ where $b_{\ell+1}/b_\ell = \text{const}$, and estimate:
$$\hat{P}(s_\ell, L_k) \approx \frac{\text{count}(s \in [b_\ell, b_{\ell+1}))}{N_k (b_{\ell+1} - b_\ell)}$$
Log-spaced bins are used because: (i) the distribution spans many decades in $s$; (ii) a power law is linear on log-log axes; (iii) log-spaced bins give roughly equal numbers of samples per bin across decades, ensuring comparable statistical precision at all scales (linearly-spaced bins would leave exponentially few samples in high-$s$ bins for a power-law distribution).

**(b)** Grid-search collapse algorithm:
```
for tau_s in linspace(1.0, 2.0, 100):
    for D in linspace(1.0, 4.0, 100):
        // Compute rescaled curves
        for each system size L_k:
            u_k = s_bins / L_k^D
            Q_k = s_bins^tau_s * P_hat(s_bins, L_k)
        // Interpolate all Q_k onto a common u grid
        u_grid = sorted union of all u_k values
        for each u in u_grid:
            g_bar(u) = mean(Q_k(u) for k with u in range of u_k)
        // Collapse quality
        Q_score[tau_s, D] = mean over k,u of (Q_k(u) - g_bar(u))^2
return argmin_{tau_s, D} Q_score
```
Grid resolution to detect $\pm 0.05$ differences: steps of $\Delta\tau_s \leq 0.025$, $\Delta D \leq 0.025$ (half the desired resolution to avoid missing the minimum). A $200 \times 200$ grid is typically sufficient.

**(c)** Collapse quality functional:
$$\mathcal{Q}(\tau_s, D) = \sum_{k=1}^K \sum_{\ell} w_{k\ell}\left[Q_k(u_\ell) - \bar{g}(u_\ell)\right]^2$$
where $w_{k\ell}$ are inverse-variance weights $w_{k\ell} \propto N_k / \hat{P}(s_\ell, L_k)$ (Poisson counting statistics). **Advantages of least-squares**: computationally cheap, interpretable, yields analytic formulas for $\bar{g}$. **Disadvantages**: implicitly assumes Gaussian deviations; least-squares is optimal for homoskedastic Gaussian errors but the Poisson counting errors in $\hat{P}$ are heteroskedastic (variance proportional to bin count). A maximum-likelihood approach with Poisson likelihoods is more principled but requires iterative optimization over the unknown scaling function $g$.

**(d)** Detection and bias from finite-size corrections. With corrections $P(s,L) = s^{-\tau_s}[g(u) + L^{-\omega} h(u)]$:
- **Detection**: plot the collapse residuals $r_{k\ell} = Q_k(u_\ell) - \bar{g}(u_\ell)$ as a function of $L_k$ for fixed $u_\ell$. If $r_{k\ell} \propto L_k^{-\omega}$, corrections are present. A systematic trend of $r$ with $L$ (not consistent with noise) is the signature.
- **Bias in $\tau_s$**: the uncorrected collapse minimizes the scatter including the correction term. For small $L$ (large $L^{-\omega}$), the extra term $L^{-\omega}h(u)$ shifts the apparent location of the crossover and distorts the effective slope on log-log axes. This biases $\hat{\tau}_s$ away from the true value, typically toward a larger exponent (shallower cutoff) for small $L$.
- **Bias in $D$**: the correction term shifts the crossover scale $s^* \sim L^D$ by a $L^{-\omega}$-dependent amount, biasing the estimated $D$. Including only the largest system sizes $L_k$ (where corrections are smallest) reduces bias at the cost of fewer data points.
