# Self-Organized Criticality: Solutions

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

**Key insight:** Because $\Delta$ is invertible ($\det(\Delta) = 3 \neq 0$), the equation $\eta' = \eta - \Delta\mathbf{n}$ has a unique solution $\mathbf{n}$, so any two stabilizing sequences must produce the same toppling vector and hence the same final configuration.

**Sketch:**

(a) $\Delta = \begin{pmatrix} 2 & -1 \\ -1 & 2 \end{pmatrix}$.

(b) Sequence A (topple 1 first): $(2,2) \to (0,3) \to (1,1)$, $\mathbf{n}^A = (1,1)$. Sequence B (topple 2 first): $(2,2) \to (3,0) \to (1,1)$, $\mathbf{n}^B = (1,1)$. Both terminate at $\eta' = (1,1)$.

(c) Verify: $\Delta(1,1)^\top = (1,1)^\top$, so $\eta - \Delta\mathbf{n} = (2,2)-(1,1) = (1,1) = \eta'$. $\det(\Delta) = 3 \neq 0$ implies $\ker(\Delta) = \{\mathbf{0}\}$; if $\Delta\mathbf{n}^\alpha = \Delta\mathbf{n}^\beta$ then $\mathbf{n}^\alpha = \mathbf{n}^\beta$.

---

### Problem 2: Toppling Lemma via Monotonicity

**Key insight:** Adding a grain never makes a previously legal toppling illegal — it can only raise heights — so every stabilizing sequence for $\eta$ is also valid for $\eta + \mathbf{e}_j$, and the Abelian property then forces the toppling vector to be at least as large componentwise.

**Sketch:**

(a) Sequence $S$ topples $i_k$ only when $z_{i_k} \geq \deg(i_k)$ under the evolving configuration. The configuration for $\eta + \mathbf{e}_j$ after each step of $S$ has heights $\geq$ those under $\eta$ (the extra grain propagates monotonically). So each toppling in $S$ remains legal for $\eta + \mathbf{e}_j$.

(b) Since $S$ is legal for $\eta + \mathbf{e}_j$ and performs $n_i(\eta)$ topplings at each site $i$, and the Abelian property gives a unique (maximal) toppling vector, we conclude $\mathbf{n}(\eta + \mathbf{e}_j) \geq \mathbf{n}(\eta)$ componentwise.

(c) The symmetric argument holds for removal: any sequence legal for $\eta - \mathbf{e}_j$ is also legal for $\eta$, giving $\mathbf{n}(\eta) \geq \mathbf{n}(\eta - \mathbf{e}_j)$. So the statement does hold.

(d) From the toppling relation: $\Delta\mathbf{n}(\eta + \mathbf{e}_j) = \eta + \mathbf{e}_j - \eta^{(j)\prime}$ and $\Delta\mathbf{n}(\eta) = \eta - \eta'$. Subtracting and using $\eta^{(j)\prime} - \eta' = \mathbf{e}_j - \Delta G_j$ (group action on $\mathcal{R}$): $\Delta G_j = \mathbf{e}_j$, so $G_{ij} = (\Delta^{-1})_{ij}$, independent of $\eta$.

---

### Problem 3: Positive Definiteness and Invertibility of the Toppling Matrix

**Key insight:** The quadratic form $\mathbf{x}^\top \Delta \mathbf{x}$ splits into non-negative squared differences on interior edges plus squared values at sink-adjacent vertices; the sink-adjacent term forces strict positivity, unlike the full Laplacian whose constant eigenvector survives because it has no boundary ground.

**Sketch:**

(a) Decompose $\deg(i) = d_V(i) + d_s(i)$. Then $\mathbf{x}^\top \Delta \mathbf{x} = \sum_i \deg(i)x_i^2 - 2\sum_{\{i,j\}\subset V} x_ix_j = \sum_{\{i,j\}\subset V}(x_i-x_j)^2 + \sum_{i:\{i,s\}\in E} x_i^2$.

(b) If the form equals zero: $x_i = 0$ for all $i$ adjacent to $s$; $(x_i - x_j)^2 = 0$ for all interior edges implies $x$ is constant on $V$. By connectivity, constant propagates from the sink-adjacent vertex where $x_i = 0$, giving $\mathbf{x} = \mathbf{0}$.

(c) Positive definite $\Rightarrow$ $\ker(\Delta) = \{\mathbf{0}\}$ $\Rightarrow$ invertible. If two stabilizing sequences give toppling vectors $\mathbf{n}^\alpha, \mathbf{n}^\beta$, then $\Delta(\mathbf{n}^\alpha - \mathbf{n}^\beta) = \mathbf{0}$, so $\mathbf{n}^\alpha = \mathbf{n}^\beta$ — uniqueness of stabilization (Abelian property).

(d) Full Laplacian has $L\mathbf{1} = \mathbf{0}$ (rows sum to zero). Setting $x_s = 0$ eliminates this eigenvector from the allowed test vectors, and the sink-adjacent boundary terms $\sum x_i^2$ destroy the remaining flat direction, promoting the form to strict positivity.

---

### Problem 4: Mean Avalanche Size from Finite-Size Scaling

**Key insight:** The substitution $u = s/L^D$ pulls the $L$-dependence entirely into the prefactor $L^{D(2-\tau_s)}$, leaving a finite dimensionless integral; the divergence of $\langle s \rangle_L$ with $L$ is the precise quantitative statement that there is no characteristic avalanche scale.

**Sketch:**

(a) Let $u = s/L^D$: $\langle s\rangle_L = L^{D(2-\tau_s)}\int_{L^{-D}}^\infty u^{1-\tau_s}g(u)\,du \to C\cdot L^{D(2-\tau_s)}$ as $L\to\infty$, where $C = \int_0^\infty u^{1-\tau_s}g(u)\,du < \infty$ for $\tau_s < 2$.

(b) $D(2-\tau_s) \approx 2.75 \times 0.80 = 2.20$. Mean size grows as $L^{2.20}$, faster than area $L^2$, driven by the heavy tail of the distribution.

(c) $\langle s^2\rangle_L \sim L^{D(3-\tau_s)}$; variance $\sim L^{D(3-\tau_s)}$ dominates (since $D(3-\tau_s) > 2D(2-\tau_s)$ iff $\tau_s > 1$). For 2D BTW: exponent $\approx 2.75 \times 1.80 \approx 4.95$.

(d) $\langle s\rangle_\infty < \infty$ requires the integral $\int_1^\infty s^{1-\tau_s}\,ds < \infty$, i.e., $\tau_s > 2$. Same condition for finite mean of the pure power law $P(s) \sim s^{-\tau_s}$ on $[1,\infty)$.

---

### Problem 5: Spectral Exponent from Avalanche Duration Distribution

**Key insight:** The power spectral density is dominated by pulses of duration $T \sim 1/f$; integrating $T^{-\tau_T} \cdot T^2$ over $T > 1/f$ converts the duration distribution directly into a frequency power law $f^{-(3-\tau_T)}$, so true pink noise requires the mean-field exponent $\tau_T = 2$.

**Sketch:**

(a) $\hat\phi_T(f) = h_0 T e^{-\pi i fT}\operatorname{sinc}(\pi fT)$. For $fT \ll 1$: $\operatorname{sinc}\approx 1$, $|\hat\phi_T|^2 \approx h_0^2 T^2$. For $fT \gg 1$: $|\hat\phi_T|^2 \sim h_0^2/(\pi f)^2 \ll h_0^2 T^2$.

(b) Contribution from $T < 1/f$ is $O(f^{-(3-\tau_T)})$ but with a smaller coefficient; the dominant term comes from $T > 1/f$ where $|\hat\phi_T|^2 \approx h_0^2 T^2$, giving $S(f) \sim h_0^2\int_{1/f}^\infty T^{2-\tau_T}\,dT$.

(c) Evaluate with upper cutoff $L^z$: $S(f) \sim f^{-(3-\tau_T)}$ for $f \gg L^{-z}$.

(d) $\beta = 1$ requires $\tau_T = 2$ (mean-field). 2D BTW: $\tau_T \approx 1.37$, $\beta \approx 1.63$ — redder than pink noise.

(e) Pulses are not independent in practice (a large avalanche depletes heights, suppressing subsequent avalanches); inter-avalanche correlations modify $S(f)$ at low frequencies and can shift $\beta$.

---

### Problem 6: SOC vs. Tuned Criticality

**Key insight:** In tuned criticality an external agent holds the control parameter at $T_c$ against fluctuations; in SOC the drive-dissipation feedback implements an internal proportional controller that stabilizes the critical density without external intervention — the distinction is whether the restoring mechanism resides inside or outside the system's microdynamics.

**Sketch:**

(a) In Ising: external thermostat sets $T$. Perturbation $T \to T_c + \epsilon$: $\xi \sim |\epsilon|^{-\nu}$ finite; exponential cutoff in avalanche-size distribution at $s^* \sim \xi^D$; power law destroyed.

(b) In BTW: control parameter is $\langle z\rangle$. Schematic ODE: $d\langle z\rangle/dt = h - J_\text{out}(\langle z\rangle)$ with $dJ_\text{out}/d\langle z\rangle > 0$; fixed point $\langle z\rangle_c$ is stable. No external agent holds $\langle z\rangle_c$ — it emerges from balance.

(c) Formally equivalent to external control if you add a hypothetical observer, but the SOC mechanism is intrinsic: the microdynamics implement the feedback. Physical consequence: SOC systems self-correct after perturbation without intervention; tuned systems drift off-critical.

(d) Preferential attachment (Barabasi-Albert) generates $P(k) \sim k^{-3}$ through cumulative advantage in network growth, with no phase transition, no correlation length, and no universality class structure.

---

### Problem 7: The Attractor Argument and Its Prerequisites

**Key insight:** The attractor argument controls only the mean height via a flux balance, not the full measure; power-law tails require the additional Dhar result that the stationary measure is uniform on the abelian group $\mathcal{R}$, whose geometric structure near criticality produces the power laws.

**Sketch:**

(a) $J_\text{in} = h$ (constant); $J_\text{out}(\langle z\rangle)$ increasing. $d\langle z\rangle/dt = (h - J_\text{out}(\langle z\rangle))/|\Lambda|$. Fixed point $\langle z\rangle_c$: $J_\text{out}(\langle z\rangle_c) = h$; stable since $dJ_\text{out}/d\langle z\rangle > 0$.

(b) (i) Periodic boundaries: $J_\text{out} = 0$, so $d\langle z\rangle/dt = h > 0$ always; no attractor, $\langle z\rangle \to \infty$. (ii) $h = 0$: $d\langle z\rangle/dt = -J_\text{out} \leq 0$; system monotonically loses grains, falls into absorbing state below criticality.

(c) Two configurations can share the same mean height with very different fluctuation structure (e.g., one could be concentrated on a single stable configuration). Power laws follow from the Abelian property (stabilization is unique) plus Dhar's uniform measure theorem on $\mathcal{R}$ plus the fractal geometry of the recurrent class — none of which is implied by $\langle z\rangle = \langle z\rangle_c$ alone.

---

### Problem 8: Interpreting the Finite-Size Scaling Function

**Key insight:** The scaling function $g$ is the universal shape of the finite-size cutoff; data collapse is a two-parameter search where the correct $(\tau_s, D)$ collapses all empirical curves $s^{\tau_s}\hat P(s,L_k)$ vs. $s/L_k^D$ onto the single universal curve $g(u)$.

**Sketch:**

(a) $g(u) \approx g_0$ for $u \ll 1$; $g(u) \to 0$ rapidly for $u \gg 1$. Exponential $g(u) \sim e^{-u}$ gives a sharp cutoff consistent with the finite-volume exponential suppression; Gaussian $g(u) \sim e^{-u^2}$ gives a softer shoulder. Exponential is standard in sandpile models.

(b) Log-log plot: slope $-\tau_s$ for $s \ll L^D$; downward break at $s^* \sim L^D$; rapid fall-off beyond. As $L\to\infty$, break moves rightward; approaches pure power law for all $s$.

(c) Collapse: plot $s^{\tau_s}\hat P(s,L_k)$ vs. $s/L_k^D$ for each $k$. Successful collapse: all $K$ curves overlap on a single master curve. Optimize $(\tau_s, D)$ to minimize inter-curve spread.

(d) The microscopic lattice spacing $a$ introduces a UV cutoff; for $s \sim O(1)$, lattice corrections spoil the ansatz. This appears as a systematic offset of small-$L$ curves at the left end of the collapse plot.

---

### Problem 9: Universality Classes and Model Symmetries

**Key insight:** Universality class membership is determined by discrete symmetries and conservation laws of the toppling rule; the Manna model breaks the deterministic symmetry of BTW, landing at a different RG fixed point with observably different exponents, analogous to Ising vs. Potts in equilibrium.

**Sketch:**

(a) Manna: each toppling sends both grains to neighbors chosen uniformly at random (independently). Broken symmetry: determinism — in BTW each neighbor receives exactly one grain; in Manna grain distribution is stochastic. This stochasticity shifts the universality class; Manna 2D has $\tau_s \approx 1.28 \neq 1.20$.

(b) Three determinants: (i) $d$ (BTW 2D vs. BTW $d \geq 4$); (ii) deterministic vs. stochastic toppling (BTW vs. Manna, both 2D); (iii) conservation (BTW conserved vs. OFC non-conserved).

(c) At $d = d_c$, the dangerous coupling is marginal and the RG flow produces logarithmic corrections rather than power-law deviations. BTW 2D may be at $d_c = 4$ (in an effective sense due to its abelian symmetry), explaining the anomalously small $\tau_s \approx 1.20 < 3/2$ and observed logarithmic corrections to scaling.

(d) $\tau_E = 5/3 \approx 1.67$. BTW MF: $1.50$; BTW 2D: $1.20$; Manna 2D: $1.28$. None matches. MF applies if fault networks have effectively $d \geq 4$ (long-range elastic interactions coupling distant segments), which is physically implausible for shallow-crust seismicity.

---

### Problem 10: Empirical Power Laws and the SOC Attribution Problem

**Key insight:** A power-law histogram is consistent with at least a dozen distinct generative mechanisms; establishing SOC requires matching multiple exponent ratios simultaneously via the scaling relations, not just fitting one exponent.

**Sketch:**

(a) (i) Preferential attachment: $P(k) \sim k^{-3}$ for Web links — cumulative advantage, no phase transition. (ii) Multiplicative noise: firm size via Gibrat's law — log-normal mimics power law over finite range. (iii) Mixture of exponentials: $P(\tau) \sim \tau^{-1}$ gives $1/f$ noise without any criticality.

(b) Additional measurements: (i) Check the duration exponent $\tau_T$ and verify the scaling relation $\tau_T = 1 + \sigma(\tau_s - 1)$ with $\sigma = D/z$. (ii) Measure spatial fractal dimension $d_f < d$ of cascade footprints.

(c) Bak-Sneppen: grains $\leftrightarrow$ fitness values; height $\leftrightarrow$ species fitness; toppling $\leftrightarrow$ extinction/replacement of minimum-fitness species plus its neighbors; slow drive $\leftrightarrow$ iterative selection of global minimum; dissipation $\leftrightarrow$ cascade termination when all neighbors exceed threshold.

(d) Pitfalls: binning artifacts (bin width changes slope); finite-range bias (exponential tail mimics power law on limited log-log range). Standard alternative: MLE $\hat\tau = 1 + n[\sum\ln(s_k/s_\text{min})]^{-1}$. KS statistic measures $\sup_s|F_\text{emp}(s) - F_\text{fit}(s)|$; small KS means model is not rejected.

---

### Problem 11: Recurrent vs. Transient Configurations and det of Delta

**Key insight:** The bijection between recurrent configurations and spanning trees rooted at $s$ — mediated by the burning algorithm — transforms the combinatorial question about $|\mathcal{R}|$ into the algebraic matrix-tree theorem, immediately giving $|\mathcal{R}| = \det(\Delta)$.

**Sketch:**

(a) $\mu^*(\eta) > 0$ iff $\eta \in \mathcal{R}$; $\mu^*(\eta) = 0$ iff $\eta$ transient. By the ergodic theorem for positive-recurrent finite Markov chains, transient states are visited at most finitely often a.s.

(b) Matrix-tree theorem: $\det(\Delta) = $ number of spanning trees of $G$ rooted at $s$. Two-site graph from Problem 1: spanning trees rooted at $s$ are $\{1\to s, 2\to s\}$ (both go directly to $s$), $\{1\to 2\to s\}$, $\{2\to 1\to s\}$ — total 3. $\det(\Delta) = 3$. ✓

(c) Bijection sketch: given $\eta \in \mathcal{R}$, the burning algorithm burns vertices in an order determined by $\eta$; this order encodes a spanning tree rooted at $s$ (the tree edge from $i$ to its "burning neighbor"). Different $\eta$ produce different trees; the map is a bijection.

(d) For the $2\times 2$ grid with a corner sink, compute $\Delta$ explicitly (a $4\times 4$ matrix with degrees 2,3,3,4 at corner, edge, edge, interior sites) and evaluate $\det(\Delta)$; then enumerate all stable configurations applying the burning test, count those that pass, and verify the count equals $\det(\Delta)$.

---

### Problem 12: The Sandpile Group is Abelian

**Key insight:** Commutativity of $\oplus$ is immediate from the commutativity of ordinary addition; associativity uses the key identity $\text{Stab}(\text{Stab}(\alpha)+\beta) = \text{Stab}(\alpha+\beta)$; closure uses the invariance of $\mathcal{R}$ under the Markov chain dynamics.

**Sketch:**

(a) $\eta_1 \oplus \eta_2 = \text{Stab}(\eta_1+\eta_2) = \text{Stab}(\eta_2+\eta_1) = \eta_2\oplus\eta_1$. The Abelian property of topplings ensures $\text{Stab}$ is well-defined (order-independent).

(b) Key identity: $\text{Stab}(\text{Stab}(\alpha)+\beta) = \text{Stab}(\alpha+\beta)$ (topplings that occurred during the inner stabilization can be "replayed" or deferred; by the Abelian property the result is the same). Apply to $(\eta_1\oplus\eta_2)\oplus\eta_3 = \text{Stab}(\eta_1+\eta_2+\eta_3) = \eta_1\oplus(\eta_2\oplus\eta_3)$.

(c) Adding $\eta_2 \in \mathcal{R}$ grain-by-grain to $\eta_1 \in \mathcal{R}$ and stabilizing after each step: each intermediate configuration is reachable from a recurrent state by the Markov chain, so the final result is in the support of $\mu^*$, i.e., in $\mathcal{R}$.

(d) The identity element $e$ satisfies $e\oplus\eta = \eta$ for all $\eta\in\mathcal{R}$. It can be found by computing $\eta_\text{max}\oplus\eta_\text{max}\oplus\cdots$ (repeated doubling) until a fixed point of $x\mapsto x\oplus\eta_\text{max}$ is reached, or equivalently as the stabilization of the all-$(\deg(i)-1)$ configuration.

---

### Problem 13: Green's Function Symmetry

**Key insight:** The toppling relation $\Delta\mathbf{n} = \eta - \eta'$ directly gives $G_{ij} = (\Delta^{-1})_{ij}$; since $\Delta$ is a real symmetric matrix (the Laplacian), $\Delta^{-1}$ is also symmetric, giving $G_{ij} = G_{ji}$ without any further computation.

**Sketch:**

(a) For $\eta\in\mathcal{R}$: after adding $\mathbf{e}_j$ and stabilizing, the change in toppling vector satisfies $\Delta(n_i(\eta+\mathbf{e}_j) - n_i(\eta)) = (\mathbf{e}_j)_i + (\eta - \eta^{(j)\prime})_i$. In the group: $\eta + \mathbf{e}_j \oplus \eta = \eta^{(j)\prime}$, and the correction $\eta - \eta^{(j)\prime}$ lies in the image of $\Delta$; so $\Delta \mathbf{G}_j = \mathbf{e}_j$ modulo this correction.

(b) Linearity of $\Delta^{-1}$: $G_{ij} = (\Delta^{-1}\mathbf{e}_j)_i = (\Delta^{-1})_{ij}$.

(c) $\Delta_{ij} = \Delta_{ji}$ by definition. $(\Delta^{-1})_{ij} = (\Delta^{-1})_{ji}$. Hence $G_{ij} = G_{ji}$.

(d) Physical reciprocity: adding a grain at $j$ causes $i$ to topple exactly as often as adding a grain at $i$ causes $j$ to topple. This is a discrete Green's reciprocity theorem, analogous to the symmetry $G(\mathbf{x},\mathbf{y}) = G(\mathbf{y},\mathbf{x})$ of the electrostatic Green's function.

---

### Problem 14: Power-Law Tail Conditions and Moment Existence

**Key insight:** The $k$-th moment of $P(s) \sim s^{-\tau}$ is finite iff $\tau > k+1$; since $\tau_s \approx 1.20 < 2$, the mean diverges in infinite volume, which is the precise mathematical content of "no characteristic scale."

**Sketch:**

(a) $\int_{s_0}^\infty C s^{-\tau}\,ds = C s_0^{1-\tau}/(\tau-1) = 1$ gives $C = (\tau-1)s_0^{\tau-1}$, valid for $\tau > 1$.

(b) $\langle s\rangle = C\int_{s_0}^\infty s^{1-\tau}\,ds$; converges iff $1-\tau < -1$, i.e., $\tau > 2$.

(c) $\langle s^k\rangle < \infty$ iff $\tau > k+1$. Threshold $\tau_k^* = k+1$.

(d) $\tau_s \approx 1.20 < 2$: mean avalanche size is infinite in the thermodynamic limit. For any finite system size $L$, the mean is finite ($\sim L^{D(2-\tau_s)}$) but grows without bound as $L\to\infty$. This divergence is the quantitative definition of scale-free behavior.

---

### Problem 15: Upper Critical Dimension via Dimensional Analysis

**Key insight:** Mean-field theory fails when avalanche branches self-intersect with $O(1)$ probability; the probability of self-intersection in $d$ dimensions scales as $s^{2-d/2}$, which is $O(1)$ for $d \leq 4$, giving upper critical dimension $d_c = 4$.

**Sketch:**

(a) For a critical Galton-Watson process with offspring mean 1, the probability generating function $G(z)$ satisfies $G(z) = f(G(z))$ where $f$ is the offspring pgf. Near $z=1$, $G(z) \approx 1 - c\sqrt{1-z}$ (square-root singularity), which by singularity analysis of power series gives $P(s) \sim s^{-3/2}$ — the mean-field exponent.

(b) Compact avalanche: $r \sim s^{1/d}$, density $s/r^d \sim 1$. Self-intersection probability for a pair of branches: $O(r^{-d}) = O(s^{-1})$. Number of branch pairs in a tree of size $s$: $O(s^2)$. Expected self-intersections: $O(s^{2-d/2} \cdot s^{-d/2}) = O(s^{2-d/2})$... more carefully: $O(s^2 \cdot s^{-d/2}) = O(s^{2-d/2})$.

(c) Self-intersections become $O(1)$ when $2-d/2 = 0$, i.e., $d = 4$. For $d > 4$, self-intersections are negligible and mean-field is exact. Hence $d_c = 4$.

(d) Mean-field scaling relations check: $\sigma = D/z = 4/2 = 2$; $\tau_T = 1 + \sigma(\tau_s-1) = 1 + 2(1/2) = 2$. ✓ $\tau_a = 1 + (D/d_f)(\tau_s-1)$: with $d_f = d = 4$ (compact) and $D = 4$, gives $\tau_a = \tau_s = 3/2$. ✓

---

### Problem 16: Fractal Dimension of the Avalanche Boundary

**Key insight:** The relation $\tau_a = 1 + \frac{D}{d_f}(\tau_s-1)$ follows from the change-of-variables Jacobian when expressing $P(a)$ in terms of $P(s)$ via $a \sim s^{d_f/D}$; the "+1" arises from the Jacobian, not from the exponent alone.

**Sketch:**

(a) See definitions in exercises.

(b) Change variables $a = s^{d_f/D}$ in the distribution: $P(a)|da| = P(s)|ds|$. $|da/ds| = (d_f/D) s^{d_f/D - 1}$. So $a^{-\tau_a}(d_f/D)s^{d_f/D-1} \sim s^{-\tau_s}$, i.e., $s^{-\tau_a d_f/D}(d_f/D)s^{d_f/D-1} \sim s^{-\tau_s}$. Comparing exponents: $-\tau_a d_f/D + d_f/D - 1 = -\tau_s$, giving $\tau_a = 1 + (\tau_s-1)D/d_f$.

(c) Plug in: $1 + (1.20-1)(2.75/2.00) = 1 + 0.275 = 1.275$. Tabulated value $\tau_a \approx 1.14$ — mismatch reflects $d_f \approx 2$ being approximate and logarithmic corrections in 2D BTW.

(d) $d_f = D = 3$: $\tau_a = 1 + (\tau_s-1)\cdot 1 = \tau_s$. Area and size distributions have the same exponent, as expected when every toppled site is visited exactly once (compact avalanches with no multi-toppling).

---

### Problem 17: Eigenvalue Bounds on the Spectral Gap of Delta

**Key insight:** A slowly varying test vector achieving Rayleigh quotient $O(L^{-2})$ shows $\lambda_1 = O(L^{-2})$, so the spectral gap closes as $L^{-2}$ and the mixing time grows as $L^2$ — implying burn-in must scale at least quadratically in system size.

**Sketch:**

(a) Courant-Fischer: $\lambda_1 = \min_{\mathbf{x}\neq 0} \mathbf{x}^\top\Delta\mathbf{x}/\|\mathbf{x}\|^2 > 0$ (Problem 3). Lower bound via Cheeger: $\lambda_1 \geq 1/|V|$ in rough terms; for a single sink edge the exact bound depends on graph geometry.

(b) Gershgorin: all eigenvalues of $\Delta$ lie in $\bigcup_i [0, 2\deg(i)]$. Since $\deg(i) \leq \Delta_\text{max} \leq 4$ for the grid, $\lambda_{|V|} \leq 8$.

(c) Test vector $x_{ij} = \sin(\pi i/(L+1))\sin(\pi j/(L+1))$: Rayleigh quotient $\approx 2 - 2\cos(\pi/(L+1)) + 2 - 2\cos(\pi/(L+1)) \approx 2\pi^2/(L+1)^2 = O(L^{-2})$. Hence $\lambda_1 \leq O(L^{-2})$; combined with the lower bound, $\lambda_1 = \Theta(L^{-2})$.

(d) Mixing time $\sim 1/\lambda_1 = \Theta(L^2)$: the burn-in must exceed $\Omega(L^2)$ grain additions. For $L = 100$, this is $O(10^4)$ steps minimum — in practice, $O(10^5)$ to $10^6$ is used to be safe.

---

### Problem 18: Gutenberg-Richter Law as a Power-Law Derivation

**Key insight:** The Gutenberg-Richter relation is linear in $\log E$ (since $M \propto \log_{10}E$), so exponentiating it converts the linear relation to a power law for $N(\geq E)$; differentiation then gives the energy density exponent $\tau_E = 1 + 2b/3$.

**Sketch:**

(a) $M = \frac{2}{3}\log_{10}E + c_0 \Rightarrow E = E_0 \cdot 10^{(3/2)(M-c_0)} = E_0 \cdot 10^{(3/2)M}\cdot 10^{-(3/2)c_0}$.

(b) $N(\geq M) = 10^{a-bM}$. Substitute $M = \frac{2}{3}\log_{10}(E/E_0)$: $N(\geq E) = 10^{a}\cdot (E/E_0)^{-2b/3} \propto E^{-2b/3}$.

(c) $P(E) = -dN/dE \sim E^{-(1+2b/3)}$; $\tau_E = 1+2b/3$. For $b=1$: $\tau_E = 5/3 \approx 1.667$.

(d) BTW 2D ($\tau_s = 1.20$), MF ($\tau_s = 1.50$), Manna 2D ($\tau_s \approx 1.28$) — none matches $1.67$. For MF to apply, fault networks would need effective dimension $d \geq 4$ (long-range elastic stress coupling), which is geophysically unrealistic. The Gutenberg-Richter exponent likely reflects a distinct universality class or non-SOC mechanism.

---

## Algorithmic Applications

### Problem 19: BTW Sandpile Simulation

**Key insight:** The queue-based relaxation achieves $O(s)$ per avalanche because each toppling event is processed exactly once; the MLE estimator for the power-law exponent is unbiased and achieves the Cramer-Rao bound without requiring histogram binning.

**Sketch:**

```
function topple(z, i, j, L):
    z[i][j] -= 4
    dissipated = 0
    for (di, dj) in {(+-1,0),(0,+-1)}:
        ni, nj = i+di, j+dj
        if 0 <= ni < L and 0 <= nj < L:
            z[ni][nj] += 1
        else:
            dissipated += 1
    return dissipated

function relax(z, L):
    queue = deque of unstable sites
    total = 0
    toppled = set()
    while queue nonempty:
        (i,j) = queue.popleft()
        if z[i][j] >= 4:
            topple(z, i, j, L)
            total += 1; toppled.add((i,j))
            for neighbor (ni,nj): if z[ni][nj] >= 4: queue.append((ni,nj))
    return total, toppled

# Main
z = zeros(L,L)
for _ in range(N_burn):
    z[randint(L)][randint(L)] += 1; relax(z, L)
sizes = []
for _ in range(N_measure):
    z[randint(L)][randint(L)] += 1
    s, _ = relax(z, L); sizes.append(s)

# MLE
samples = [s for s in sizes if s >= s_min]
n = len(samples)
tau_hat = 1 + n / sum(log(s/(s_min-0.5)) for s in samples)
# Choose s_min by minimizing KS statistic over candidate values
```

Queue complexity: $O(s)$ per avalanche vs. $O(L^2 \cdot s)$ for naive scanning.

---

### Problem 20: Dhar's Burning Algorithm

**Key insight:** Tracking `unburned_neighbors` counts in an array and processing sites through a queue ensures each site enters the queue at most $O(\deg_\text{max})$ times, giving $O(|V|)$ total complexity vs. $O(|V|^2)$ for naive repeated full-grid passes.

**Sketch:**

```
function is_recurrent(z, L):
    burned = array of False, shape (L,L)
    unburned_nbrs[i][j] = number of grid neighbors of (i,j)
    queue = deque()

    # Initialize with boundary sites (adjacent to sink)
    for (i,j) in boundary_sites:
        if z[i][j] >= unburned_nbrs[i][j]:
            queue.append((i,j))

    while queue nonempty:
        (i,j) = queue.popleft()
        if burned[i][j]: continue
        if z[i][j] >= unburned_nbrs[i][j]:
            burned[i][j] = True
            for (ni,nj) in grid_neighbors(i,j):
                if not burned[ni][nj]:
                    unburned_nbrs[ni][nj] -= 1
                    if z[ni][nj] >= unburned_nbrs[ni][nj]:
                        queue.append((ni,nj))

    return all(burned[i][j] for all i,j)
```

Complexity: $O(|V|)$. Naive: $O(|V|^2)$.

Application (i): run `is_recurrent` on sampled configurations after burn-in; all should return `True`. Application (ii): iterate over all $\prod_i (\deg(i))$ stable configurations for small $L$, count passing configurations, verify against $\det(\Delta)$.

---

### Problem 21: Scaling Exponent Estimation via Data Collapse

**Key insight:** Data collapse is a two-parameter optimization over $(\tau_s, D)$; the correct pair makes $s^{\tau_s}\hat P(s, L_k)$ vs. $s/L_k^D$ collapse to a single curve $g(u)$ across all system sizes, with the collapse quality functional providing a smooth objective for grid search.

**Sketch:**

```
# Data collection: log-spaced bins for each L_k
for k in range(K):
    bins = logspace(log10(1), log10(s_max), N_bins+1)
    P_hat[k] = histogram(sizes[k], bins, density=True)

# Grid search
best = (inf, None, None)
for tau in linspace(1.0, 2.0, 200):
    for D in linspace(1.0, 4.0, 200):
        # Rescale
        for k: u[k] = bin_centers / L[k]**D
                Q[k] = bin_centers**tau * P_hat[k]
        # Common u grid via interpolation
        u_common = sorted union of all u[k]
        g_bar = mean(interp(Q[k], u[k], u_common) for k)
        score = mean over k of mean((interp(Q[k],u[k],u_common) - g_bar)**2)
        if score < best[0]: best = (score, tau, D)

# Detect finite-size corrections:
# Plot residuals r_k(u) = Q_k(u) - g_bar(u) vs L_k^{-omega} for various omega
# Systematic linear trend in residuals indicates corrections are present
```

Log-spaced bins: power laws span many decades; log-spacing gives equal numbers of samples per decade. Grid resolution 0.01 suffices to resolve $\pm 0.05$ differences in $\tau_s$ and $D$.

---

### Problem 22: Sandpile Group Enumeration on Small Graphs

**Key insight:** The enumeration reduces to applying Dhar's burning test to all stable configurations and then computing the Cayley table via the `relax` operation; verifying $|\mathcal{R}| = \det(\Delta)$ provides a concrete numerical check of Dhar's identity.

**Sketch:**

```
# Enumerate recurrent configurations
recurrent = []
for eta in all_stable_configs(G):   # iterate {0,...,deg(i)-1}^|V|
    if is_recurrent(eta, G):
        recurrent.append(eta)

# Build Cayley table
table = {}
for eta1 in recurrent:
    for eta2 in recurrent:
        combined = elementwise_add(eta1, eta2)
        table[(eta1,eta2)] = stabilize(combined, G)

# Verify commutativity
assert all(table[(e1,e2)] == table[(e2,e1)] for e1,e2 in recurrent x recurrent)

# Verify |R| = det(Delta)
assert len(recurrent) == round(det(Delta_matrix(G)))

# Find identity: unique e with table[(e,eta)] == eta for all eta
identity = first e in recurrent such that all(table[(e,eta)]==eta for eta in recurrent)
```

For the $2\times 2$ grid with corner sink: compute $\Delta$ (4x4 matrix), evaluate $\det(\Delta)$, run the above to enumerate $\mathcal{R}$ and construct the group table. The identity element is the configuration obtained by stabilizing the all-$(\deg(i)-1)$ configuration.

---

### Problem 23: Branching Process Simulation for Mean-Field SOC

**Key insight:** A critical Galton-Watson process is the exact mean-field limit of SOC and produces $P(s) \sim s^{-3/2}$ analytically; comparing its empirical distribution with BTW data at $d=2$ and $d\geq 4$ directly tests the $d_c = 4$ prediction.

**Sketch:**

```
function simulate_avalanche(s_max):
    pop = 1; total = 0
    while pop > 0 and total < s_max:
        total += pop
        # Poisson(1) offspring per individual (critical)
        pop = sum(Poisson(1.0) for _ in range(pop))
    if total >= s_max: return None  # discard; size diverges
    return total

# Generate N finite avalanches
sizes = []
attempts = 0
while len(sizes) < N:
    s = simulate_avalanche(s_max=10000)
    attempts += 1
    if s is not None: sizes.append(s)

# MLE estimate
tau_hat = mle_powerlaw(sizes, s_min=10)  # from Problem 19

# Comparison
# Run BTW at L large in d=2 and d=4 (hypercubic lattice)
# Plot CCDF (complementary CDF) for all three datasets on same axes
# Expect: d=4 BTW overlaps branching process; d=2 BTW departs (tau_s ~ 1.20 vs 1.50)
```

Sample size: to resolve $\Delta\tau = 0.30$ at $3\sigma$ confidence, the standard error of MLE $\approx (\tau-1)/\sqrt{n}$ requires $n \geq 9(\tau-1)^2/(\Delta\tau)^2 \approx 25$ in theory, but effective $n$ after applying $s_\text{min}$ cutoff means $N \sim 10^4$ total samples is a practical minimum. The finite-size cutoff $s_\text{max}$ should be chosen large enough ($s_\text{max} \gg n^{1/(\tau-1)}$) that truncation probability is negligible.
