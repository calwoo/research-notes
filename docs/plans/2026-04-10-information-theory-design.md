# Design: Information Theory Concept Note

**Date:** 2026-04-10
**Topic slug:** `information-theory`
**Category:** `concepts`
**Multi-note:** yes

## Scope

This note cluster covers classical and modern information theory from a rigorous mathematical perspective. The scope spans Shannon's foundational results (entropy, mutual information, channel capacity), the Kullback–Leibler divergence and its cousins, the asymptotic equipartition property, rate-distortion theory, and maximum-entropy methods. It extends into more modern territory: information-geometric structures on families of distributions, information cohomology (Baudot–Bennequin), and applications to quantization (e.g., the TurboQuant paper on learned quantization). The Jaynes maximum-entropy principle paper is treated as a primary source alongside Shannon's original works.

The cluster is motivated by a reading program spanning practical ML papers (quantization, compression) and mathematically deep fields (information cohomology, information geometry). Notes should be mathematically rigorous — formal definitions, entropy inequalities with proofs, derivations of the data-processing inequality, and the like.

## Files to Create

| File | Purpose |
|------|---------|
| `concepts/information-theory/overview.md` | Topic index, subtopic map, dependency graph, master references |
| `concepts/information-theory/entropy-and-divergences.md` | Entropy, KL divergence, mutual information — definitions, properties, inequalities |
| `concepts/information-theory/aep-and-typicality.md` | AEP, typical sets, source coding theorem |
| `concepts/information-theory/channel-capacity.md` | Channel capacity, Fano's inequality, noisy channel coding theorem |
| `concepts/information-theory/rate-distortion.md` | Rate-distortion theory, Blahut–Arimoto algorithm |
| `concepts/information-theory/maximum-entropy.md` | Jaynes max-entropy principle, exponential families, variational characterization |
| `concepts/information-theory/information-geometry.md` | Fisher information metric, exponential/mixture geodesics, α-connections |
| `concepts/information-theory/information-cohomology.md` | Baudot–Bennequin construction, simplicial cohomology of information, higher-order dependencies |

## Note Structure (first note: entropy-and-divergences.md)

1. **Shannon Entropy** — axiomatic characterization, binary entropy, chain rule
2. **Rényi and Tsallis Entropies** — one-parameter families, limit recovery of Shannon
3. **Kullback–Leibler Divergence** — definition, non-symmetry, non-negativity (Gibbs' inequality proof)
4. **f-Divergences** — general family, Pinsker's inequality, variational representations
5. **Mutual Information** — definition via KL, chain rule, data-processing inequality
6. **Entropy Inequalities** — subadditivity, strong subadditivity (Lieb–Ruskai for quantum analogue noted)
7. **Exercises** — Mathematical Development + Algorithmic Applications (inline)

## Planned Subtopics

| File | Description |
|------|-------------|
| `entropy-and-divergences.md` | Core definitions and inequalities — the foundation for all subsequent notes |
| `aep-and-typicality.md` | Weak and strong AEP; source coding; typical sequences |
| `channel-capacity.md` | Operational definitions; Shannon's channel coding theorem; Fano's inequality |
| `rate-distortion.md` | Distortion-rate function; Blahut–Arimoto; applications to quantization |
| `maximum-entropy.md` | Jaynes' principle; exponential family characterization; connections to statistical mechanics |
| `information-geometry.md` | Riemannian structure on distributions; Fisher metric; α-connections; Amari's work |
| `information-cohomology.md` | Baudot–Bennequin construction; topos-theoretic perspective; higher mutual information |

## References

- Shannon, C.E. (1948). *A Mathematical Theory of Communication*
- Jaynes, E.T. (1957). *Information Theory and Statistical Mechanics* — https://bayes.wustl.edu/etj/articles/theory.1.pdf
- Cover & Thomas. *Elements of Information Theory* (2nd ed.)
- Baudot & Bennequin (2015). *The Homological Nature of Entropy*
- TurboQuant paper (learned vector quantization — to be looked up)
- Amari, S. (2016). *Information Geometry and Its Applications*
- Csiszár & Körner. *Information Theory: Coding Theorems for Discrete Memoryless Systems*
