# Design: Lyapunov Functions Concept Note

**Date:** 2026-04-29
**Topic slug:** `lyapunov-functions`
**Category:** `concepts`
**Multi-note:** no

## Scope

This note develops the theory of Lyapunov functions from first principles — rigorous definitions, Lyapunov's direct method (stability/asymptotic stability theorems), and LaSalle's invariance principle — and then applies it to prove the central result: that the KL divergence $D_{\mathrm{KL}}(\hat{x} \| x(t))$ is a Lyapunov function for the replicator dynamic when $\hat{x}$ is an evolutionarily stable strategy (ESS).

The treatment targets a reader with a math/ML background: it assumes comfort with ODEs and information theory but builds the game-theory prerequisites (simplex, replicator ODE, ESS) from scratch. The goal is that a reader finishes understanding exactly *why* $\dot{D}_{\mathrm{KL}} \leq 0$ along replicator trajectories, not just *that* it holds. A secondary thread connects this to the information geometry of the simplex (Shahshahani metric = Fisher information metric) and mirror descent.

## Files to Create

| File | Purpose |
|------|---------|
| `concepts/lyapunov-functions/lyapunov-stability.md` | Main note: Lyapunov theory, replicator dynamic, KL Lyapunov proof, geometric interpretation |

## Note Structure

1. **Introduction** — statement of the goal; two-sentence teaser of why KL divergence is a natural Lyapunov function
2. **Lyapunov Stability Theory**
   - Equilibria and stability notions (Lyapunov stable, asymptotically stable, globally asymptotically stable)
   - Lyapunov functions: definition and Lyapunov's direct method (theorem + proof sketch)
   - LaSalle's invariance principle
   - Exercises: verify a Lyapunov function for a concrete ODE
3. **The Replicator Dynamic**
   - Setup: finite symmetric game, mixed strategy simplex $\Delta^n$
   - Replicator ODE: $\dot{x}_i = x_i(f_i(x) - \bar{f}(x))$
   - Fixed points and their classification
   - Evolutionarily stable strategies (ESS): definition and relation to Nash equilibria
   - Exercises: fixed points of rock-paper-scissors replicator
4. **KL Divergence as a Lyapunov Function**
   - Statement: $V(x) = D_{\mathrm{KL}}(\hat{x} \| x)$ is a Lyapunov function at an ESS $\hat{x}$
   - Computation of $\dot{V}$ along replicator trajectories (the key derivation)
   - Interpreting positivity/decrease conditions via the ESS inequality
   - Exercises: verify $\dot{V} \leq 0$ for a concrete 2-strategy game
5. **Geometric Interpretation** (callout-style aside)
   - Shahshahani metric on $\Delta^n$ equals Fisher information metric
   - Replicator dynamic as gradient flow of fitness under this metric
   - Connection to mirror descent / multiplicative weights
6. **References**

## References

- Khalil (2002), *Nonlinear Systems* — Lyapunov direct method
- Perko (2001), *Differential Equations and Dynamical Systems* — ODE/stability theory
- Hofbauer & Sigmund (1998), *Evolutionary Games and Population Dynamics* — replicator + KL Lyapunov
- Hofbauer & Sigmund (2003), AMS Bulletin survey — concise self-contained entry point
- Weibull (1995), *Evolutionary Game Theory* — ESS, stability hierarchy
- Sandholm (2010), *Population Games and Evolutionary Dynamics* — generalized dynamics
- Harper (2009), arXiv:0911.1383 — KL Lyapunov iff ESS + information geometry
- Fryer (2012), arXiv:1207.0036 — KL Lyapunov for incentive dynamics (generalization)
- Raskutti & Mukherjee (2015), arXiv:1310.7780 — mirror descent / KL / replicator connection
