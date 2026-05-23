# Design: Sparsity and Pruning in Deep Learning

**Date:** 2026-05-23
**Topic slug:** `sparsity-pruning`
**Category:** `concepts`
**Multi-note:** yes

## Scope

This note cluster surveys sparsity and pruning in deep learning from first principles to the modern LLM era. The mathematical backbone is the Taylor-expansion view of weight saliency — both the diagonal-Hessian approximation of OBD and the full inverse-Hessian treatment of OBS — and its modern reincarnation in SparseGPT. Alongside the theory, each note includes PyTorch implementations of the core algorithms to reinforce the material from an engineering perspective.

The cluster spans four eras: (1) classical Hessian-based methods treating pruning as constrained optimization on the loss surface; (2) the empirical/hardware era treating it as a compression pipeline (prune → quantize → encode) targeting inference on custom ASICs; (3) the sparse-training era (Lottery Ticket Hypothesis, dynamic sparse training) questioning whether we need a dense model at all; and (4) the LLM-compression era where the OBS math is made tractable at 175B-parameter scale.

Two companion notes are planned in sibling concept folders: `concepts/optimization-theory/second-order-methods.md` (covering Newton, Gauss-Newton, Fisher information, natural gradient — mathematical prerequisites for understanding OBD/OBS rigorously) and `concepts/deep-learning-engineering/knowledge-distillation/knowledge-distillation.md` (covering the main competing compression paradigm).

## Files to Create

| File | Purpose |
|------|---------|
| `concepts/deep-learning-engineering/sparsity-pruning/overview.md` | Topic index, subtopic map, dependency graph, master references |
| `concepts/deep-learning-engineering/sparsity-pruning/classical-pruning.md` | OBD → OBS → magnitude pruning → iterative magnitude pruning; full Hessian-based saliency derivations + PyTorch implementations |
| `concepts/deep-learning-engineering/sparsity-pruning/compression-pipelines.md` | Deep Compression pipeline (prune + quantize + Huffman encode) and EIE hardware accelerator; complexity analysis + PyTorch implementations |
| `concepts/deep-learning-engineering/sparsity-pruning/structured-pruning.md` | Filter/channel pruning (Li et al., Liu et al. BN-scaling), attention head pruning (Michel et al., Voita et al.); structured vs. unstructured hardware implications + PyTorch implementations |
| `concepts/deep-learning-engineering/sparsity-pruning/sparse-training.md` | Lottery Ticket Hypothesis, SNIP, SET, SNFS, RigL — training sparse networks from scratch or online topology update; PyTorch implementations |
| `concepts/deep-learning-engineering/sparsity-pruning/llm-pruning.md` | Movement Pruning, SparseGPT, Wanda — LLM-scale compression; layerwise OBS + activation-weighted saliency; PyTorch/transformers implementations |
| `concepts/optimization-theory/second-order-methods.md` | Newton, Gauss-Newton, Fisher information matrix, natural gradient — prerequisite for OBD/OBS |
| `concepts/deep-learning-engineering/knowledge-distillation/knowledge-distillation.md` | Teacher-student distillation (Hinton et al. 2015), response-based vs. feature-based methods; contrast with pruning |

## Note Structure: classical-pruning.md (first note)

1. **Motivation** — why over-parameterized networks are sparse at convergence; parameter counts vs. task complexity
2. **The Taylor Expansion View** — second-order Taylor approximation of the loss change from deleting a weight; saliency definition
3. **Optimal Brain Damage (OBD)** — diagonal Hessian approximation; saliency criterion derivation; computational cost; backprop for second derivatives
4. **Optimal Brain Surgeon (OBS)** — full inverse Hessian; KKT-based weight update formula derivation; when OBD and OBS disagree
5. **Magnitude-based Pruning** — zeroth-order approximation; why it often works despite theoretical weakness; iterative magnitude pruning (IMP) pipeline
6. **PyTorch Implementations** — OBD saliency scorer, magnitude pruner, IMP training loop using `torch.nn.utils.prune`

Exercises inline after each major section.

## Planned PyTorch Coverage

| Note | Key Implementations |
|------|---------------------|
| classical-pruning.md | `DiagonalHessianPruner`, `MagnitudePruner`, `iterative_magnitude_prune()` training loop |
| compression-pipelines.md | k-means weight quantization, Huffman coding sketch, `torch.nn.utils.prune` + custom sparse weight storage |
| structured-pruning.md | `FilterPruner` (ℓ₁ filter norms), BN γ-based channel pruning, attention head masking |
| sparse-training.md | `SNIP` saliency at init, `RigL` mask update step, sparse optimizer wrapper |
| llm-pruning.md | `SparseGPT` layerwise weight update (approximate inverse Hessian), `Wanda` activation-weighted pruning |

## References

See master references in `overview.md`. Key anchors: LeCun et al. 1990 (OBD), Hassibi & Stork 1993 (OBS), Han et al. 2015 (IMP), Han et al. 2016 (Deep Compression, EIE), Frankle & Carlin 2019 (LTH), Evci et al. 2020 (RigL), Frantar & Alistarh 2023 (SparseGPT), Sun et al. 2023 (Wanda).
