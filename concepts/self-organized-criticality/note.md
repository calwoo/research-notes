# Self-Organized Criticality

## Table of Contents

1. [What Is Self-Organized Criticality?](#what-is-self-organized-criticality)
   - [Ordinary Criticality vs. SOC](#ordinary-criticality-vs-soc)
   - [The Three Empirical Hallmarks](#the-three-empirical-hallmarks)
   - [Natural Examples and Connection to Scaling Laws](#natural-examples-and-connection-to-scaling-laws)
2. [The BTW Sandpile Model](#the-btw-sandpile-model)
   - [2.1 Formal Definition](#21-formal-definition)
   - [2.2 Time-Scale Separation](#22-time-scale-separation)
   - [2.3 Avalanche Observables](#23-avalanche-observables)
   - [2.4 The 1/f Noise Connection](#24-the-1f-noise-connection)
3. [Why Self-Organized Criticality?](#why-self-organized-criticality)
   - [3.1 The Attractor Argument](#31-the-attractor-argument)
   - [3.2 Connection to Absorbing State Phase Transitions](#32-connection-to-absorbing-state-phase-transitions)
4. [The Abelian Sandpile (Dhar 1990)](#the-abelian-sandpile-dhar-1990)
   - [4.1 Setup on a General Graph](#41-setup-on-a-general-graph)
   - [4.2 The Abelian Property](#42-the-abelian-property)
   - [4.3 The Toppling Lemma](#43-the-toppling-lemma)
   - [4.4 Recurrent Configurations and the Sandpile Group](#44-recurrent-configurations-and-the-sandpile-group)
5. [Scaling Theory and Exponents](#scaling-theory-and-exponents)
   - [5.1 Finite-Size Scaling](#51-finite-size-scaling)
   - [5.2 Scaling Relations](#52-scaling-relations)
   - [5.3 Known Values for 2D BTW](#53-known-values-for-2d-btw)
   - [5.4 Universality](#54-universality)
6. [References](#references)

---

## What Is Self-Organized Criticality?

### Ordinary Criticality vs. SOC

In equilibrium statistical mechanics, a critical point — for instance the Ising model at temperature $T = T_c$ — requires fine-tuning a control parameter to a special value. The free energy landscape has a non-analytic point precisely at the critical temperature, and generically, the system is off-critical. Away from $T_c$, the correlation length $\xi$ is finite:

$$\xi \sim |T - T_c|^{-\nu}$$

and fluctuations are characterized by an intrinsic length scale $\xi$. The divergence of $\xi$ at $T_c$ is what produces scale-free behavior (power-law correlations, critical opalescence, etc.), but this behavior is a set of measure zero in parameter space.

**Self-organized criticality (SOC)**, introduced by Bak, Tang, and Wiesenfeld (1987), asserts that a broad class of complex systems — those with many slowly-driven, weakly-dissipative components — can spontaneously evolve to a critical state exhibiting scale-free behavior without any external fine-tuning of parameters.

The "self-organized" aspect is the key conceptual departure: the dynamics itself tunes the system to the critical point. The critical state is an **attractor** of the dynamics, not a fine-tuned boundary condition imposed by an external agent. The control parameter (analogous to $T$) is not fixed externally but is instead an emergent property of the stationary state reached by the dynamics under slow drive and weak dissipation.

### The Three Empirical Hallmarks

A system exhibiting SOC is characterized by three signatures, all of which reflect the absence of a characteristic scale:

1. **Power-law event-size distributions**: the probability of an event (avalanche, earthquake, neural cascade) of size $s$ satisfies

   $$P(s) \sim s^{-\tau}, \quad s \to \infty$$

   with no exponential cutoff. Compare to an exponential distribution $P(s) \sim e^{-s/s_0}$, which has a characteristic scale $s_0$ and arises generically away from criticality. The power law implies that large events are not exponentially rare — they are merely power-law suppressed.

2. **$1/f$ noise**: the power spectral density of the system's output signal $\phi(t)$ satisfies

   $$S(f) = |\hat{\phi}(f)|^2 \sim f^{-\beta}, \quad \beta \approx 1$$

   This is "pink noise," intermediate between white noise ($\beta = 0$, flat spectrum, no temporal correlations) and Brownian motion ($\beta = 2$, corresponding to a random walk). Pink noise implies temporal correlations at all scales — no characteristic time.

3. **Fractal spatial geometry**: the spatial footprint of events has no characteristic length scale and is statistically self-similar across scales. Formally, if $M(r)$ is the mass (number of active sites) within radius $r$ of the event origin, then

   $$M(r) \sim r^{D_f}$$

   for some fractal dimension $D_f < d$ (the embedding dimension). This is to be contrasted with compact events ($D_f = d$) or point-like events ($D_f = 0$).

All three hallmarks are consequences of the same underlying criticality: the system sits at a second-order phase transition point where the correlation length diverges, but the system arrives there dynamically rather than by external tuning.

### Natural Examples and Connection to Scaling Laws

**Earthquakes.** The Gutenberg-Richter law states

$$\log_{10} N(M) = a - bM, \quad b \approx 1$$

where $N(M)$ is the number of earthquakes with magnitude $\geq M$ and $M = \frac{2}{3}\log_{10} E + \text{const}$ is the moment magnitude. Substituting, this implies a power law for seismic energy release:

$$P(E) \sim E^{-\tau_E}, \quad \tau_E = 1 + \frac{2b}{3} \approx \frac{5}{3}$$

**Forest fires.** In certain ecosystems, the distribution of fire sizes is approximately power-law, consistent with the forest-fire model of Drossel and Schwabl (1992), itself an SOC model.

**Biological evolution.** Bak and Sneppen (1993) proposed SOC as a mechanistic explanation for punctuated equilibrium in the fossil record — long periods of stasis interrupted by rapid bursts of evolutionary change. The Bak-Sneppen model produces power-law distributions of extinction cascade sizes.

**Neural avalanches.** Beggs and Plenz (2003) recorded local field potentials in cortical slices and observed that spontaneous bursts of neural activity (neural avalanches) have power-law size distributions with exponent $\tau \approx 3/2$, consistent with a mean-field branching process at criticality. This suggests that neural networks may operate near an SOC critical point, potentially to maximize dynamic range and information transmission.

**Caveat.** The causal attribution of empirical power laws to SOC is debated. Many mechanisms can produce heavy-tailed or power-law distributions without SOC: preferential attachment (Barabasi-Albert), multiplicative noise, mixtures of exponentials, and simple self-similar geometry. The presence of a power law is necessary but not sufficient for SOC.

**Connection to neural scaling laws.** Bahri et al. (2021) provided a statistical mechanics derivation of neural network scaling laws (of the form $L(N) \sim N^{-\alpha}$ for loss $L$ as a function of model size $N$), arguing that the scaling exponent $\alpha$ is inversely proportional to the intrinsic dimension $d$ of the data manifold:

$$\alpha \propto \frac{1}{d}$$

This mirrors how SOC exponents depend on the geometry of the underlying system — in both cases, a power law emerges from an underlying geometric or statistical structure, and the exponent encodes dimensionality information. This is a deep structural analogy: in SOC, the critical exponents are determined by the dimension of the lattice and the universality class of the phase transition; in neural scaling, the loss exponents are determined by the intrinsic dimension of the data distribution.

---

## The BTW Sandpile Model

The Bak-Tang-Wiesenfeld (BTW) sandpile, introduced in Bak et al. (1987), is the canonical model of SOC. We give a precise mathematical definition.

### 2.1 Formal Definition

**Lattice.** Let $\Lambda = \{1, \ldots, L\}^2 \subset \mathbb{Z}^2$ be an $L \times L$ square lattice with open (absorbing) boundaries. Denote the set of boundary sites as $\partial\Lambda$ (those with fewer than 4 neighbors in $\Lambda$) and interior sites as $\Lambda^\circ = \Lambda \setminus \partial\Lambda$.

**State space.** A configuration is a function $z : \Lambda \to \mathbb{Z}_{\geq 0}$, where $z_i$ is the "height" (number of sand grains) at site $i \in \Lambda$. The state space is $\Omega = \mathbb{Z}_{\geq 0}^{|\Lambda|}$.

**Threshold.** The critical threshold is $z_c = 4$ (the degree of an interior site on $\mathbb{Z}^2$).

**Toppling rule.** A site $i$ is **unstable** if $z_i \geq z_c = 4$. When an unstable site $i$ topples:

$$z_i \to z_i - 4, \qquad z_j \to z_j + 1 \quad \forall j \in \mathcal{N}(i)$$

where $\mathcal{N}(i)$ is the set of nearest neighbors of $i$ in $\Lambda$ (with $|\mathcal{N}(i)| \leq 4$). If $i \in \partial\Lambda$, it has $|\mathcal{N}(i)| < 4$ neighbors; the $4 - |\mathcal{N}(i)|$ "missing" grains are lost to the boundary. This loss constitutes the **dissipation mechanism** — the only way grains leave the system.

**Stable configurations.** A configuration $z$ is **stable** if $z_i < 4$ for all $i \in \Lambda$. The set of stable configurations is $\mathcal{S} = \{0,1,2,3\}^{|\Lambda|}$.

**Relaxation operator.** Given any configuration $z \in \Omega$, the relaxation $\mathcal{R}(z)$ is the unique stable configuration reached by repeatedly toppling any unstable site. (Uniqueness is guaranteed by the Abelian property; see Section 4.2.)

### 2.2 Time-Scale Separation

The BTW dynamics consists of two processes operating on separated time scales:

**Slow drive.** At discrete time $t = 1, 2, 3, \ldots$, add one grain to a uniformly random site $i_t \sim \text{Uniform}(\Lambda)$:

$$z \to z + \mathbf{e}_{i_t}$$

where $\mathbf{e}_i$ is the unit vector at site $i$.

**Fast relaxation.** Immediately after each addition, apply the relaxation operator:

$$z(t) = \mathcal{R}(z(t-1) + \mathbf{e}_{i_t})$$

The relaxation (avalanche) runs to completion before the next grain is added.

**Time-scale separation condition.** The drive rate $h$ (grains per unit time) is taken to zero, $h \to 0$, relative to the relaxation rate. This ensures:
1. At most one avalanche is active at any instant.
2. The system is always in a stable configuration between additions.
3. The avalanche statistics are not contaminated by overlap between successive perturbations.

This separation is not merely a technical convenience — it is physically essential. If $h$ is too large (the "fast-drive" limit), avalanches overlap and the critical behavior is destroyed.

**Stationary distribution.** Under this dynamics, the system reaches a unique stationary distribution $\mu^*$ over stable configurations. The SOC phenomenology (power-law avalanche statistics) is a property of this stationary distribution. Computing $\mu^*$ exactly is the content of Dhar's abelian sandpile theory (Section 4).

### 2.3 Avalanche Observables

After adding a grain at time $t$, the resulting avalanche is characterized by three observables:

**Size** $s$: the total number of topplings summed over all sites and all time steps of the avalanche,

$$s = \sum_{i \in \Lambda} n_i$$

where $n_i$ is the number of times site $i$ topples during the avalanche.

**Duration** $T$: the number of parallel update steps (synchronous updates) until stability. In each step, all currently unstable sites topple simultaneously.

**Area** $a$: the number of distinct sites that topple at least once,

$$a = |\{i \in \Lambda : n_i \geq 1\}|$$

Note the inequalities $a \leq s$ (since multiple topplings at a single site count once in $a$ but multiple times in $s$) and $T \leq s$ (since each step involves at least one toppling).

**Empirical distributions in the stationary regime.** For large $L$, the distributions of $s$, $T$, and $a$ in the stationary state follow power laws with exponential cutoffs imposed by the finite system size:

$$P(s) \sim s^{-\tau_s} \, g_s(s / L^{D_s}), \qquad \tau_s \approx 1.20, \quad D_s \approx 2.75$$

$$P(T) \sim T^{-\tau_T} \, g_T(T / L^{z}), \qquad \tau_T \approx 1.37, \quad z \approx 1.57$$

$$P(a) \sim a^{-\tau_a} \, g_a(a / L^{2}), \qquad \tau_a \approx 1.14$$

where $g_s, g_T, g_a$ are scaling functions that decay rapidly for argument $\gg 1$ and approach constants for argument $\ll 1$. The exponent $D_s$ is the **avalanche fractal dimension** (relating avalanche mass to its spatial extent), and $z$ is the **dynamical exponent** (relating duration to spatial extent via $T \sim r^z$).

**Remark on exact values.** The exponents $\tau_s, \tau_T, D_s, z$ for the 2D BTW sandpile are not known exactly and are determined numerically. This is in contrast to the 1D sandpile (trivially solved) and the mean-field case ($d \geq 4$, where exact exponents are known via branching process arguments: $\tau_s = 3/2$, $\tau_T = 2$).

### 2.4 The 1/f Noise Connection

Consider the time series $\{\phi(t)\}_{t \geq 1}$ where $\phi(t)$ is the number of grains dissipated through the boundary during the avalanche triggered at time $t$. Bak, Tang, and Wiesenfeld argued that this signal exhibits $1/f$ noise.

**Heuristic argument.** Model each avalanche as a rectangular pulse of height $h_0$ and duration $T$. The Fourier transform of a single pulse of duration $T$ has power concentrated at frequencies $f \lesssim 1/T$:

$$|\hat{\phi}_T(f)|^2 \approx h_0^2 T^2 \cdot \mathbf{1}[f \lesssim 1/T]$$

The aggregate power spectrum is the average over avalanche durations drawn from $P(T) \sim T^{-\tau_T}$:

$$S(f) \sim \int_0^\infty P(T) \, |\hat{\phi}_T(f)|^2 \, dT \sim \int_{1/f}^\infty T^{-\tau_T} \cdot T^2 \, dT \sim f^{-(3 - \tau_T)}$$

where the lower limit of integration is $1/f$ because pulses shorter than $1/f$ contribute negligibly at frequency $f$. Therefore:

$$\boxed{S(f) \sim f^{-(3-\tau_T)}}$$

For $\tau_T \approx 1.37$: $S(f) \sim f^{-1.63}$, which is close to but not exactly $1/f$. The exact $\beta = 1$ (pink noise) would require $\tau_T = 2$, which is the mean-field value.

**Caveat.** The $1/f$ label is used loosely. More precisely, BTW predicts $S(f) \sim f^{-\beta}$ with $\beta = 3 - \tau_T \in (1, 2)$ depending on the model. Furthermore, $1/f$ noise is not unique to SOC: any superposition of relaxation processes (Lorentzians) with a power-law distribution of relaxation times $P(\tau) \sim \tau^{-1}$ produces $S(f) \sim f^{-1}$ without any SOC dynamics. The presence of $1/f$ noise is consistent with SOC but does not establish it.

<!-- TODO: sections 3-6 -->
