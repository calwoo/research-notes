# Design: Self-Organized Criticality Concept Note

**Date:** 2026-03-07
**Topic slug:** `self-organized-criticality`
**Category:** `concepts`

## Scope

A unified concept note covering self-organized criticality from first principles, anchored on:
- Bak, Tang, Wiesenfeld (1987) — the BTW sandpile model and the SOC concept
- Dhar (1990) — the Abelian sandpile, exact mathematical framework

Style: intuition-first, then full derivations. Curated references table included.

## Files to Create

| File | Purpose |
|------|---------|
| `concepts/self-organized-criticality/note.md` | Main research note |
| `concepts/self-organized-criticality/exercises.md` | Problem set |
| `concepts/self-organized-criticality/solutions.md` | Full answer key |

## Note Structure

### 1. What Is SOC?
- The central claim: complex systems self-organize to a critical state without external parameter tuning
- Contrast with ordinary criticality: phase transitions (e.g., Ising model at T_c) require fine-tuning of a control parameter; SOC reaches criticality automatically
- The three empirical hallmarks: power-law event-size distributions, 1/f noise in time series, fractal spatial geometry
- Why this matters: SOC provides a candidate explanation for ubiquitous power laws in nature (earthquakes, forest fires, biological evolution, neural activity)

### 2. The BTW Sandpile Model
- Formal definition: integer height variable z_i on lattice Z^2, open boundary (sink), threshold z_c = 4
- Toppling rule: if z_i >= z_c, then z_i -> z_i - 4, and z_j -> z_j + 1 for each of 4 neighbors
- Time-scale separation: slow driving (add one grain at random site) vs. fast relaxation (avalanche runs to completion before next grain added)
- Avalanche observables: size s (total topplings), duration t, area a (distinct sites toppled)
- Empirical power laws: P(s) ~ s^{-tau}, P(t) ~ t^{-alpha}
- 1/f noise: if grains are added at rate r, the power spectrum of the output flux S(f) ~ f^{-beta}

### 3. Why Self-Organized Criticality?
- The attractor argument: in the subcritical regime the system quickly reaches an absorbing state (no more topplings); in the supercritical regime grains leave the system faster than they enter; the critical boundary is the only stable stationary state
- Analogy to SOC as a fixed point of the dynamics in the (drive rate, dissipation) plane
- Why no fine-tuning is needed: the slow drive and boundary dissipation together implement an implicit feedback loop that drives the system to criticality

### 4. The Abelian Sandpile (Dhar 1990)
- Setup: finite graph G with a distinguished sink vertex s; height variables z_i for i != s
- Toppling rule: same as BTW but on general graph with degree-based thresholds
- **Abelian property**: if two unstable sites i and j are both toppled, the order does not matter — the final configuration is the same. Formal statement and proof sketch via the toppling lemma.
- **Toppling lemma**: if a legal toppling sequence starting from configuration eta leads to a stable configuration eta', then every legal toppling sequence from eta also leads to eta', and every site topples the same number of times.
- Recurrent vs. transient configurations: a stable configuration is recurrent iff it appears with positive probability in the stationary distribution; equivalently, iff it can be reached from any configuration by adding grains
- Brief mention of the sandpile group: the set of recurrent configurations forms an abelian group under the addition operator

### 5. Scaling Theory and Exponents
- Finite-size scaling ansatz: P(s, L) = s^{-tau} g(s / L^D) where D is the fractal dimension of avalanches and g is a universal scaling function
- In the scaling limit, g(x) ~ const for x << 1 and g(x) -> 0 rapidly for x >> 1
- Scaling relations: tau, alpha (duration exponent), D (fractal/avalanche dimension), z (dynamical exponent z = D * nu_perp)
- Known values for 2D BTW sandpile: tau ~ 1.2, D ~ 2.75 (numerical); note that exact analytical values remain open
- Connection to standard critical phenomena: compare to percolation and directed percolation universality classes

### 6. References
Curated table covering:
- Bak, Tang, Wiesenfeld (1987) — original BTW paper
- Dhar (1990) — Abelian sandpile and exact results
- Bak (1996) — "How Nature Works" (accessible book)
- Jensen (1998) — "Self-Organized Criticality" (textbook)
- Dickman et al. (2000) — paths to SOC, absorbing state perspective
- Pruessner (2012) — "Self-Organised Criticality: Theory, Models and Characterisation" (comprehensive reference)
- Good lecture notes (e.g., Christensen & Moloney)

## Exercise Structure (exercises.md)

1. **Derivation problems** — prove the abelian property for a simple 2-site system; derive the toppling lemma; show the 1/f noise connection via Fourier analysis of avalanche durations
2. **Conceptual questions** — distinguish SOC from tuned criticality; explain the attractor argument; interpret the scaling function g(x)
3. **Implementation sketches** — pseudocode for BTW simulation on L x L grid; algorithm for identifying recurrent configurations
