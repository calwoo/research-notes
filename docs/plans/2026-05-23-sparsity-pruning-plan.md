# Plan: Sparsity and Pruning in Deep Learning

**Date:** 2026-05-23
**Design doc:** `2026-05-23-sparsity-pruning-design.md`

## Tasks

1. [x] Create design doc
2. [ ] Create `concepts/sparsity-pruning/` directory
3. [ ] Write `concepts/sparsity-pruning/overview.md` (topic index, subtopic map, dependency graph, master references)
4. [ ] Write `concepts/sparsity-pruning/classical-pruning.md` (OBD, OBS, magnitude pruning + PyTorch)
5. [ ] Write `concepts/sparsity-pruning/compression-pipelines.md` (Deep Compression, EIE + PyTorch)
6. [ ] Write `concepts/sparsity-pruning/structured-pruning.md` (filter/channel/head pruning + PyTorch)
7. [ ] Write `concepts/sparsity-pruning/sparse-training.md` (LTH, SNIP, RigL + PyTorch)
8. [ ] Write `concepts/sparsity-pruning/llm-pruning.md` (SparseGPT, Wanda, Movement Pruning + PyTorch)
9. [ ] Write `concepts/optimization-theory/second-order-methods.md` (Newton, Gauss-Newton, Fisher — OBD/OBS prerequisite)
10. [ ] Write `concepts/knowledge-distillation/knowledge-distillation.md` (Hinton et al. teacher-student)
11. [ ] Final cross-check: TOC anchors, notation consistency, every exercise has inline solution, PyTorch code runs
12. [ ] Commit all files

## Notes

- PyTorch implementations are woven into each note alongside the math — not appended separately
- Classical pruning note written first (most mathematical prerequisites); LLM pruning last (assumes all prior notes)
- Second-order methods companion note should be written before or alongside classical-pruning.md
