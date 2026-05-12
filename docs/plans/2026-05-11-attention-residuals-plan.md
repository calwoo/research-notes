# Plan: Attention Residuals Paper Note

**Date:** 2026-05-11
**Design doc:** `docs/plans/2026-05-11-attention-residuals-design.md`

## Tasks

1. [x] Write design doc and plan
2. [ ] Write `papers/attention-residuals.md` covering all 11 sections
3. [ ] Launch image-extractor agent to fetch Figs 1, 5, 8, 9 from arxiv source
4. [ ] Commit: `feat: add attention-residuals paper note — AttnRes, PreNorm dilution, gradient analysis`

## Notes

- Source paper: https://arxiv.org/abs/2603.15031
- Expositional inspiration: https://kindxiaoming.github.io/blog/2026/attention-residual/
- Key figures to embed: Fig 1 (architecture overview), Fig 5 (training dynamics), Fig 8 (learned weight heatmaps), Fig 9 (structured matrix view)
- The gradient Jacobian analysis (Section 3) is the core novel content not found in most presentations
