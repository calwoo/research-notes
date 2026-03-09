# Implementation Plan: Attention Mechanisms Concept Note

**Date:** 2026-03-08
**Design doc:** `docs/plans/2026-03-08-attention-mechanisms-design.md`

## Tasks

### 1. Write `note.md` — Standard Softmax Attention
- Sections 1–5: Introduction, Q/K/V projections, attention score computation, causal masking,
  softmax normalization, output projection, matrix form, multi-head attention, KV caching.
- Include formal definitions, derivations, and the matrix-form equivalence.
- Typographic style: first-use italics, bold for key conclusions, Obsidian wikilink TOC.

### 2. Write `linear-attention.md` — Linear Attention
- Sections 1–7: Motivation, linear attention formulation, recurrence relation, training
  challenges, chunkwise-parallel form, decay/gating, neural memory / fast-weight perspective.
- Derive the state update recurrence from first principles.
- Cover scalar, vector, and matrix-valued gating with model examples.
- Derive the online linear regression connection and the delta rule.

### 3. Review both notes
- Check mathematical correctness (notation consistency, derivation steps).
- Verify TOC wikilinks resolve correctly.
- Confirm references table is complete.

### 4. Write `exercises.md`
- 16–18 Mathematical Development problems.
- 5–7 Algorithmic Applications problems.
- Each problem has italic preamble and prerequisites wikilink.
- Continuous numbering 1–N.

### 5. Write `solutions.md`
- Key insight + Sketch format for every exercise.
- No full worked derivations — concise, sharp.

### 6. Final cross-check
- Every exercise has a solution.
- TOC anchors match exact heading text.
- No LaTeX or em-dashes in headings.
- Notation consistent across both note files.

### 7. Commit
```bash
git add docs/plans/2026-03-08-attention-mechanisms-*.md concepts/attention-mechanisms/
git commit -m "feat: add attention-mechanisms concept note (standard + linear attention)"
```
