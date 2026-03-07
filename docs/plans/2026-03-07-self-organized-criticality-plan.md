# Self-Organized Criticality Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a mathematically rigorous concept note on self-organized criticality anchored on the BTW sandpile and Abelian sandpile, with matching exercises and full solutions.

**Architecture:** Three parallel files under `concepts/self-organized-criticality/`. The note builds from intuition to formalism: motivation → BTW model → attractor argument → Abelian sandpile → scaling theory → references. Exercises follow the repo-mandated order: derivations → conceptual → implementation sketches.

**Tech Stack:** Markdown with LaTeX math. No code execution. Source material: Bak et al. (1987), Dhar (1990), Bak (1996) *How Nature Works*, Jensen (1998) *Self-Organized Criticality*.

---

### Task 1: Write note — Section 1 (What Is SOC?) and Section 2 (BTW Sandpile)

**Files:**
- Create: `concepts/self-organized-criticality/note.md`

**Step 1: Create the file with title and TOC placeholder**

Start the file with `# Self-Organized Criticality`, then a Table of Contents linking to all 6 sections and their subsections. Leave `<!-- TODO: sections 3-6 -->` at the end.

**Step 2: Write Section 1 — What Is SOC?**

Cover:
- The central claim: complex systems with many interacting components spontaneously evolve to a critical state — one exhibiting scale-free behavior — without any external fine-tuning of parameters.
- Contrast with ordinary criticality: in equilibrium statistical mechanics (e.g., the Ising model), a critical point requires tuning a control parameter (temperature $T$) to the critical value $T_c$. SOC reaches criticality automatically via its own dynamics.
- The three empirical hallmarks:
  1. **Power-law event-size distributions**: $P(s) \sim s^{-\tau}$ for avalanche sizes $s$
  2. **1/f noise**: the power spectrum of the system's output $S(f) \sim f^{-\beta}$ (pink noise), ubiquitous in nature and often used as an empirical signature of SOC
  3. **Fractal spatial geometry**: the spatial extent of events has no characteristic length scale
- Natural examples: earthquake magnitude-frequency (Gutenberg-Richter law $\log N \sim -b M$), forest fires, biological extinction events, neural avalanches in cortex. Note: the relevance of SOC to each of these is debated — state this honestly.
- Connection to neural scaling laws: Bahri et al. (2021) argue that the power-law scaling of neural network loss with scale arises from SOC-like dynamics in the loss landscape; the scaling exponent $\alpha \propto 1/d$ (intrinsic dimension) echoes how SOC exponents depend on lattice geometry.

**Step 3: Write Section 2 — The BTW Sandpile Model**

Subsections:

**2.1 Formal Definition**

Define on $\mathbb{Z}^2$ (square lattice) with $L \times L$ sites and open (absorbing) boundaries. Height variable $z_i \in \mathbb{Z}_{\geq 0}$ at each site $i$. Threshold $z_c = 4$ (degree of the lattice interior).

Toppling rule: if $z_i \geq z_c = 4$, site $i$ topples:
$$z_i \to z_i - 4, \qquad z_j \to z_j + 1 \quad \forall j \sim i$$
where $j \sim i$ denotes nearest neighbors. Boundary sites have fewer neighbors; grains falling off the boundary are lost (this is the dissipation mechanism).

**2.2 Time-Scale Separation**

The dynamics has two distinct time scales:
- **Slow drive**: add one grain to a randomly chosen site: $z_i \to z_i + 1$
- **Fast relaxation**: run all topplings to completion (the avalanche) before adding the next grain

This separation is essential. If grains were added at the same rate as topplings, the system would not reach SOC. The drive rate $h \to 0$ limit is the SOC regime.

**2.3 Avalanche Observables**

After each grain addition, measure:
- $s$ = total number of topplings (avalanche size)
- $t$ = number of time steps until no unstable sites remain (duration)
- $a$ = number of distinct sites that toppled at least once (area)

Empirically (and confirmed numerically for large $L$):
$$P(s) \sim s^{-\tau_s}, \quad P(t) \sim t^{-\tau_t}, \quad P(a) \sim a^{-\tau_a}$$
with $\tau_s \approx 1.2$, $\tau_t \approx 1.4$ for 2D BTW.

**2.4 The 1/f Noise Connection**

If grains are added at a slow constant rate and one records the output flux $\phi(t)$ (grains leaving through the boundary per unit time), the power spectrum $S(f) = |\hat{\phi}(f)|^2$ satisfies:
$$S(f) \sim f^{-\beta}, \qquad \beta \approx 1$$
This is "pink noise" or "1/f noise." Bak et al. (1987) argued this is the generic signature of SOC, and that 1/f noise in natural systems (electronic devices, music, heart rate variability) is evidence of SOC. Note: this claim is contested; many mechanisms produce 1/f noise.

**Step 4: Commit**

```bash
cd /Users/calvinwoo/Documents/notes
git add concepts/self-organized-criticality/note.md
git commit -m "feat: add SOC note sections 1-2 (motivation and BTW sandpile)"
```

---

### Task 2: Write note — Section 3 (Why SOC?) and Section 4 (Abelian Sandpile)

**Files:**
- Modify: `concepts/self-organized-criticality/note.md`

Append sections 3 and 4 before the `<!-- TODO: sections 3-6 -->` comment, replacing it with `<!-- TODO: sections 5-6 -->`.

**Step 1: Write Section 3 — Why Self-Organized Criticality?**

**3.1 The Attractor Argument**

Argue informally that the critical state is the unique stationary state of the BTW dynamics. Consider two extreme regimes:

- **Subcritical** (average height $\langle z \rangle \ll z_c$): most sites have height $< z_c$, avalanches are small and die quickly. Adding grains increases $\langle z \rangle$. The system is not stationary — it drifts upward.
- **Supercritical** (average height $\langle z \rangle \approx z_c$): avalanches are large and reach the boundary frequently, dissipating many grains. The output flux exceeds the input flux, so $\langle z \rangle$ decreases. The system drifts downward.

The critical state $\langle z \rangle = \langle z \rangle_c$ is the unique fixed point where input flux = output flux. The slow drive and boundary dissipation implement an implicit feedback loop that stabilizes this fixed point without external control.

**3.2 Relation to Absorbing State Phase Transitions**

More formally (Dickman et al. 2000): SOC can be understood as an absorbing state phase transition where the order parameter (activity density) is self-tuned to the critical point. The drive rate $h$ and dissipation rate $\epsilon$ play the role of control parameters; SOC corresponds to the limit $h, \epsilon \to 0$ with $h/\epsilon \to 0$. In this limit the system sits exactly at the phase boundary between an active phase and an absorbing (quiescent) phase.

This connects SOC to the broader theory of non-equilibrium phase transitions and directed percolation (though BTW is not in the DP universality class).

**Step 2: Write Section 4 — The Abelian Sandpile (Dhar 1990)**

**4.1 Setup on a General Graph**

Let $G = (V, E)$ be a finite connected graph with a distinguished sink vertex $s \notin V$. Each non-sink vertex $i \in V$ has a height $z_i \in \{0, 1, \ldots, \deg(i) - 1\}$ (stable) or $z_i \geq \deg(i)$ (unstable). Toppling rule for unstable vertex $i$:
$$z_i \to z_i - \deg(i), \qquad z_j \to z_j + 1 \quad \forall j \sim i$$
Grains sent to the sink are lost. This is the **toppling matrix** $\Delta$:
$$\Delta_{ij} = \begin{cases} \deg(i) & i = j \\ -1 & i \sim j \\ 0 & \text{otherwise} \end{cases}$$
so a toppling at $i$ changes heights by $-\Delta_{i\cdot}$ (the $i$-th row of $\Delta$ with sign flipped).

**4.2 The Abelian Property**

**Theorem (Dhar 1990):** Let $\eta$ be an unstable configuration. If $\alpha$ and $\beta$ are two legal toppling sequences (sequences of topplings of unstable sites) both leading to stable configurations, then $\alpha$ and $\beta$ lead to the same stable configuration, and each site $i$ topples the same number of times $n_i$ in both sequences.

**Proof sketch:**
1. Define $n_i^\alpha$ = number of times site $i$ topples in sequence $\alpha$. The final configuration is $\eta'_i = \eta_i - \deg(i) \cdot n_i^\alpha + \sum_{j \sim i} n_j^\alpha$ (grains received from neighbors minus grains lost).
2. Equivalently in matrix form: $\eta' = \eta - \Delta^\top \mathbf{n}^\alpha$.
3. If two legal sequences $\alpha, \beta$ both stabilize $\eta$, then $\Delta^\top \mathbf{n}^\alpha = \Delta^\top \mathbf{n}^\beta$. Since $\Delta$ is an M-matrix (positive definite on $V$, with the sink providing the necessary boundary condition), it is invertible, so $\mathbf{n}^\alpha = \mathbf{n}^\beta$.
4. Therefore $\eta'^\alpha = \eta'^\beta$. $\square$

The abelian property is what makes the sandpile analytically tractable: the final state after an avalanche is well-defined regardless of implementation details.

**4.3 The Toppling Lemma**

**Lemma:** If a legal toppling sequence starting from $\eta$ visits site $i$ at least once, then every legal toppling sequence from $\eta$ (leading to a stable state) also visits site $i$ at least once.

**Proof:** By the abelian property, all legal stabilizing sequences produce the same $\mathbf{n}$. If $n_i \geq 1$ in one sequence, then $n_i \geq 1$ in all. $\square$

This justifies speaking of "the" stabilization of a configuration without reference to a specific toppling order.

**4.4 Recurrent Configurations and the Sandpile Group**

A stable configuration $\eta$ is **recurrent** if it appears with positive probability in the stationary distribution of the Markov chain defined by the BTW dynamics. Equivalently (Dhar's burning algorithm): $\eta$ is recurrent iff it passes the "script" test — when grains are added one at a time to each vertex in turn, no vertex topples more than once before all vertices have fired.

The set $\mathcal{R}$ of recurrent configurations with the addition operator (add a grain at $i$, then stabilize) forms an **abelian group** — the **sandpile group** of $G$. For the $L \times L$ grid, $|\mathcal{R}| = \det(\Delta)$, which grows as $e^{cL^2}$.

**Step 3: Commit**

```bash
git add concepts/self-organized-criticality/note.md
git commit -m "feat: add SOC note sections 3-4 (attractor argument and Abelian sandpile)"
```

---

### Task 3: Write note — Section 5 (Scaling Theory) and Section 6 (References)

**Files:**
- Modify: `concepts/self-organized-criticality/note.md`

Append sections 5 and 6, replacing `<!-- TODO: sections 5-6 -->`. No TODO comment at the end.

**Step 1: Write Section 5 — Scaling Theory and Exponents**

**5.1 Finite-Size Scaling**

For a system of linear size $L$, the avalanche size distribution takes the finite-size scaling form:
$$P(s, L) = s^{-\tau_s} \, g\!\left(\frac{s}{L^D}\right)$$
where $D$ is the **fractal dimension** of avalanches (the exponent relating typical avalanche size to system size: $\langle s \rangle \sim L^D$), and $g(x)$ is a universal scaling function satisfying:
$$g(x) \approx \text{const} \quad x \ll 1, \qquad g(x) \to 0 \text{ rapidly} \quad x \gg 1$$
In the $L \to \infty$ limit, the cutoff $L^D \to \infty$ and $P(s) \sim s^{-\tau_s}$ for all $s$.

**5.2 Scaling Relations**

The exponents are not all independent. Define:
- $\tau_s$: size exponent ($P(s) \sim s^{-\tau_s}$)
- $\tau_t$: duration exponent ($P(t) \sim t^{-\tau_t}$)
- $D$: avalanche fractal dimension ($s \sim a^{D/d_f}$ where $d_f$ is the spatial fractal dimension)
- $z$: dynamical exponent relating duration to size ($t \sim s^z$)

From $t \sim s^z$ and $s \sim L^D$:
$$\tau_t = 1 + z(\tau_s - 1)$$
This scaling relation constrains the exponents. Additional relations arise from conservation laws (BTW is locally conservative, grains only lost at boundary).

**5.3 Known Values for 2D BTW**

Numerically established values (exact analytical values are an open problem):
- $\tau_s \approx 1.20 \pm 0.01$
- $\tau_t \approx 1.37 \pm 0.02$
- $D \approx 2.75$ (avalanche dimension)
- $z \approx 1.5$ (dynamical exponent)

Note: the 2D BTW sandpile is believed to have logarithmic corrections to pure power-law scaling, making accurate numerical determination of exponents difficult. Some exact results are known for the Abelian sandpile on specific graphs (e.g., the complete graph), but the 2D square lattice remains analytically open.

**5.4 Universality**

Different SOC models (BTW, Manna, forest-fire) belong to different universality classes with different exponents. The BTW sandpile (deterministic toppling) and the Manna model (stochastic toppling) are believed to be in different universality classes despite both being "sandpile" models. The universality class is determined by symmetries (conservation, abelian vs. non-abelian toppling), not just by the qualitative SOC behavior.

**Step 2: Write Section 6 — References**

Add a references table with columns "Reference Name", "Brief Summary", "Link to Reference":

| Bak, Tang, Wiesenfeld (1987), "Self-organized criticality: An explanation of the 1/f noise" | Original paper introducing the BTW sandpile model and the SOC concept; establishes power-law avalanche statistics and the 1/f noise connection | https://doi.org/10.1103/PhysRevLett.59.381 |
| Dhar (1990), "Self-organized critical state of sandpile automaton models" | Introduces the Abelian sandpile, proves the abelian property and toppling lemma, establishes the sandpile group; foundational mathematical treatment | https://doi.org/10.1103/PhysRevLett.64.1613 |
| Bak (1996), "How Nature Works: The Science of Self-Organized Criticality" | Accessible book-length treatment of SOC by one of its creators; covers motivation, applications, and conceptual framework with minimal formalism | https://link.springer.com/book/10.1007/978-1-4757-5426-1 |
| Jensen (1998), "Self-Organized Criticality: Emergent Complex Behavior in Physical and Biological Systems" | Graduate-level textbook; covers BTW, Abelian sandpile, scaling theory, and applications; good first reference for mathematical treatment | https://doi.org/10.1017/CBO9780511622717 |
| Dickman, Muñoz, Vespignani, Zapperi (2000), "Paths to self-organized criticality" | Reviews the connection between SOC and absorbing state phase transitions; clarifies the role of drive rate and dissipation in self-tuning | https://doi.org/10.1590/S0103-97332000000100030 |
| Pruessner (2012), "Self-Organised Criticality: Theory, Models and Characterisation" | Comprehensive modern reference; covers field-theoretic approaches, numerical methods, and a wide range of SOC models; the most complete mathematical treatment available | https://doi.org/10.1017/CBO9780511977671 |
| Christensen and Moloney (2005), "Complexity and Criticality" | Textbook with detailed BTW and Abelian sandpile chapters; good for self-study with worked examples | https://doi.org/10.1142/p365 |

**Step 3: Commit**

```bash
git add concepts/self-organized-criticality/note.md
git commit -m "feat: add SOC note sections 5-6 (scaling theory and references)"
```

---

### Task 4: Write exercises file

**Files:**
- Create: `concepts/self-organized-criticality/exercises.md`

**Step 1: Write Derivation Problems (section 1)**

1. **Abelian property for a 2-site system.** Consider a graph with two non-sink vertices $\{1, 2\}$ connected to each other and each connected to a sink $s$. The toppling matrix is $\Delta = \begin{pmatrix} 2 & -1 \\ -1 & 2 \end{pmatrix}$. Suppose $\eta = (2, 2)$ (both sites unstable with threshold 2). Show explicitly that toppling site 1 first then site 2, versus site 2 first then site 1, leads to the same final configuration. Identify $\mathbf{n} = (n_1, n_2)$.

2. **Toppling lemma from the abelian property.** Use the abelian property (all legal stabilizing sequences produce the same toppling vector $\mathbf{n}$) to prove the toppling lemma: if site $i$ topples at least once in some legal sequence, then $n_i \geq 1$, i.e., it topples at least once in every legal sequence.

3. **Invertibility of the toppling matrix.** Show that the toppling matrix $\Delta$ defined in Section 4.1 is positive definite on $V$ (the non-sink vertices), hence invertible. Use this to complete the proof of the abelian property: if $\Delta^\top \mathbf{n}^\alpha = \Delta^\top \mathbf{n}^\beta$, then $\mathbf{n}^\alpha = \mathbf{n}^\beta$.

4. **Finite-size scaling self-consistency.** Assume $P(s, L) = s^{-\tau_s} g(s/L^D)$ and that $g(x) \sim x^{-c}$ for $x \gg 1$ (exponential or faster cutoff). Derive the scaling of the mean avalanche size $\langle s \rangle_L = \int_0^\infty s \, P(s, L) \, ds$ as a function of $L$. Express the answer as a power law in $L$ and identify the exponent in terms of $\tau_s$ and $D$.

5. **The 1/f noise connection.** Suppose avalanche durations are power-law distributed: $P(t) \sim t^{-\tau_t}$ for $1 \leq t \leq t_{\max}$, and each avalanche contributes a rectangular pulse of unit amplitude and duration $t$ to the output signal $\phi(t)$. Show heuristically (using the Wiener-Khinchin theorem and a superposition argument) that the power spectrum of $\phi$ scales as $S(f) \sim f^{-(3-\tau_t)}$ for $\tau_t \in (1, 3)$. For $\tau_t \approx 1.4$, what value of $\beta$ does this predict?

**Step 2: Write Conceptual Questions (section 2)**

1. Explain the difference between SOC and ordinary criticality (e.g., the Ising model at $T_c$). In particular: what is "fine-tuned" in each case, and why does SOC not require fine-tuning?

2. Describe the attractor argument for why the BTW sandpile reaches the critical state. What is the role of (a) the slow drive and (b) the open boundary in this argument? What would happen if the boundary were closed (periodic)?

3. The abelian property says the final configuration after an avalanche is independent of toppling order. Why is this property nontrivial? Give an example of a cellular automaton rule for which the analogous property would fail.

4. What distinguishes a recurrent configuration from a transient one in the Abelian sandpile? State Dhar's burning algorithm criterion for recurrence and explain intuitively why it works.

5. The 2D BTW sandpile has exponent $\tau_s \approx 1.2$, meaning $P(s) \sim s^{-1.2}$. Since $\tau_s < 2$, the mean avalanche size $\langle s \rangle$ diverges as $L \to \infty$. Why is this not a contradiction with the system being stationary? How does the finite-size cutoff resolve this?

**Step 3: Write Implementation Sketches (section 3)**

1. **BTW simulation.** Sketch an algorithm for simulating the BTW sandpile on an $L \times L$ grid. Specify: (a) how to represent the height array; (b) the grain addition step; (c) the avalanche relaxation loop (what data structure to use for the active set of unstable sites, and why); (d) how to record avalanche statistics $s$, $t$, $a$. What is the computational complexity per grain addition in the worst case?

2. **Dhar's burning algorithm.** Sketch an algorithm for determining whether a given stable configuration $\eta$ on the $L \times L$ grid is recurrent, using Dhar's burning algorithm: iteratively "burn" (remove) vertices that can topple given the current configuration augmented by the burns already performed, until no more burns are possible. If all non-sink vertices burn, the configuration is recurrent. Specify the loop structure and termination condition.

3. **Estimating the scaling exponent.** Given a list of $N$ avalanche sizes $\{s_1, \ldots, s_N\}$ from a large-$L$ simulation, sketch a procedure for estimating $\tau_s$. Include: (a) why naive histogram binning in linear space is unreliable for heavy-tailed distributions; (b) how to use log-binning to construct the empirical $P(s)$; (c) the maximum likelihood estimator (MLE) for $\tau_s$ for a pure power law $P(s) \propto s^{-\tau_s}$ for $s \geq s_{\min}$.

**Step 4: Commit**

```bash
git add concepts/self-organized-criticality/exercises.md
git commit -m "feat: add exercises for self-organized criticality"
```

---

### Task 5: Write solutions file

**Files:**
- Create: `concepts/self-organized-criticality/solutions.md`

Read the exercises file first. Then provide complete worked solutions for all 13 problems:

**Derivation 1 (2-site abelian):** Work out the full toppling sequences explicitly. Starting from $\eta = (2,2)$ with $\Delta = \begin{pmatrix}2&-1\\-1&2\end{pmatrix}$: topple site 1 → $\eta = (-1+1, 2+1) = ...$. Wait — threshold is $\deg(i) = 2$ (two edges: one to the other site, one to sink). Topple 1: $z_1 \to z_1-2$, $z_2 \to z_2+1$ (neighbor), sink gets 1. Continue until stable. Do both orders and show same final state and same $\mathbf{n}$.

**Derivation 2 (toppling lemma):** Since all stabilizing sequences have the same $\mathbf{n}$ (by abelian property), if $n_i \geq 1$ in one sequence then $n_i \geq 1$ in all.

**Derivation 3 (invertibility):** $\Delta$ is diagonally dominant with positive diagonal and negative off-diagonal (it is a graph Laplacian restricted to non-sink vertices). The sink provides the "grounding" that makes it strictly diagonally dominant: for any vertex $i$ adjacent to the sink, $\Delta_{ii} > \sum_{j \neq i} |\Delta_{ij}|$. Strict diagonal dominance implies positive definiteness. Then invertibility follows. The abelian uniqueness follows from $\Delta^\top(\mathbf{n}^\alpha - \mathbf{n}^\beta) = 0$ and $\ker(\Delta^\top) = \{0\}$.

**Derivation 4 (mean avalanche size):** $\langle s \rangle_L = \int_0^\infty s \cdot s^{-\tau_s} g(s/L^D) \, ds$. Substitute $u = s/L^D$: $= L^{D(2-\tau_s)} \int_0^\infty u^{1-\tau_s} g(u)\, du$. The integral converges (since $\tau_s < 2$ makes $u^{1-\tau_s}$ integrable near 0, and $g$ decays fast). So $\langle s \rangle_L \sim L^{D(2-\tau_s)}$.

**Derivation 5 (1/f noise):** Use Wiener-Khinchin: $S(f) = \int R(\delta t) e^{-2\pi i f \delta t} d(\delta t)$. For a superposition of pulses with duration distribution $P(t)$, $R(\delta t) \sim \int_{\delta t}^\infty P(t) \, dt \sim (\delta t)^{1-\tau_t}$ (tail of duration distribution). Fourier transform: $S(f) \sim f^{-(2-\tau_t)} \cdot f^{-1} = f^{-(3-\tau_t)}$. For $\tau_t \approx 1.4$: $\beta = 3 - 1.4 = 1.6$... actually rework this carefully. The standard result for a Poisson process of pulses of duration $t$ with $P(t) \sim t^{-\tau_t}$ gives $S(f) \sim f^{\tau_t - 3}$ for $\tau_t \in (1,3)$, i.e., $\beta = 3 - \tau_t \approx 1.6$. Note this differs from $\beta = 1$ (pure 1/f); empirical values depend on details.

**Conceptual 1-5:** Provide thorough 3-5 sentence answers referencing specific results from the note.

**Implementation 1-3:** Provide complete pseudocode with variable names, complexity analysis, and justification of design choices.

**Step 4: Commit**

```bash
git add concepts/self-organized-criticality/solutions.md
git commit -m "feat: add complete solutions for self-organized criticality exercises"
```

---

### Task 6: Final review and cross-check

**Files:**
- Review: `concepts/self-organized-criticality/note.md`
- Review: `concepts/self-organized-criticality/exercises.md`
- Review: `concepts/self-organized-criticality/solutions.md`

**Step 1: Check**
- TOC links in note.md are correct (GFM anchor format)
- Notation consistent across all three files
- Every exercise has a solution
- References table present with all 7 entries
- No TODO comments remaining

**Step 2: Commit if any fixes needed**

```bash
git add concepts/self-organized-criticality/
git commit -m "fix: cross-check corrections to SOC note, exercises, solutions"
```
