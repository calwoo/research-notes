# Plan: PyTorch Internals Concept Cluster

**Date:** 2026-05-23
**Design doc:** `docs/plans/2026-05-23-pytorch-internals-design.md`

## Tasks

1. [ ] Research references via `reference-finder` agent; compile master references table
2. [ ] Write `concepts/pytorch-internals/overview.md` — notes table, subtopic map, dependency graph, master references
3. [ ] Write `concepts/pytorch-internals/tensor-storage.md` — Tensor/Storage/TensorImpl, strides, views, aliasing
4. [ ] Write `concepts/pytorch-internals/dispatcher.md` — C10 dispatcher, dispatch keys, boxed/unboxed, op registration
5. [ ] Write `concepts/pytorch-internals/autograd-engine.md` — tape construction, Node/Edge DAG, backward scheduling
6. [ ] Write `concepts/pytorch-internals/torch-compile.md` — TorchDynamo, TorchInductor, guard logic, symbolic shapes
7. [ ] Write `concepts/pytorch-internals/memory-management.md` — CachingAllocator, CUDA pools, fragmentation
8. [ ] Write `concepts/pytorch-internals/custom-ops.md` — TORCH_LIBRARY, schemas, autograd formulas
9. [ ] Cross-check all notes: TOC anchors valid, Mermaid diagrams render, wikilinks consistent
10. [ ] Update `overview.md` status column as notes are completed
11. [ ] Commit all files

## Notes

- Start with `tensor-storage.md` and `dispatcher.md` — all other notes depend on understanding these primitives
- `torch-compile.md` is the most complex; leave for after autograd is written
- Exercises should appear inline after each section, not batched at the end
