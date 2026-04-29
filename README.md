# Research Notes

A personal knowledge repository of agent-created notes, walkthroughs, and summaries of papers and research topics. Content spans machine learning, modern deep learning, and related fields.

Topics are approached with a mathematical bent — favoring rigorous definitions, formal notation, and derivations over high-level summaries.

## Structure

```
concepts/       ← explanations of ML/math concepts
papers/         ← summaries and analyses of specific papers
walkthroughs/   ← step-by-step derivations or implementations
docs/           ← documentation and design docs
```

Each topic directory contains:
- `note.md` — research note with formal derivations and references
- `exercises.md` — problem set (mathematical development + algorithmic applications)
- `solutions.md` — full answer key

## Topics

### Concepts

| Topic | Description |
|-------|-------------|
| [Mixture of Experts](concepts/mixture-of-experts/note.md) | Sparse MoE architectures, routing mechanisms, and scaling |
| [Power-Law Scaling Laws](concepts/ml-theory/power-law-scaling.md) | Power-law scaling of loss with compute, data, and parameters |
| [Lazy/Rich Dichotomy](concepts/ml-theory/lazy-rich-dichotomy.md) | Infinite-width limits, NTK, mean-field theory, and μP |
| [Self-Organized Criticality](concepts/self-organized-criticality/note.md) | Emergent critical behavior in complex systems |

### Papers

| Topic | Description |
|-------|-------------|
| [DHEN Ranking](papers/dhen-ranking/note.md) | Deep Hierarchical Ensemble Network for ranking |
| [Sampling-Bias-Corrected Retrieval](papers/sampling-bias-corrected-retrieval/note.md) | Two-tower retrieval with correction for selection bias |
