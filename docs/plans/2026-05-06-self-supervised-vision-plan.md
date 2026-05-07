# Plan: Self-Supervised Vision Concept Note

**Date:** 2026-05-06
**Design doc:** `docs/plans/2026-05-06-self-supervised-vision-design.md`
**Output:** `concepts/self-supervised-vision/ssl-vision.md`

## Tasks

1. **Create topic directory** — `mkdir -p concepts/self-supervised-vision`
2. **Write §1: Background** — augmentation setup, joint-embedding framework, formal notation (views, encoders, projectors)
3. **Write §2: Collapse problem** — formal definition, trivial solutions, necessary conditions
4. **Write §3: Contrastive baselines** — NT-Xent loss derivation, SimCLR/MoCo overview, batch-size dependency
   - Inline exercise after §3
5. **Write §4: BYOL** — architecture diagram, EMA update rule, predictor asymmetry, stop-gradient analysis
   - Inline exercise after §4
6. **Write §5: Barlow Twins** — cross-correlation objective derivation, invariance + redundancy-reduction terms, whitening connection
   - Inline exercise after §5
7. **Write §6: VICReg** — three-term loss derivation, per-branch application, architectural flexibility
   - Inline exercise after §6
8. **Write §7: Unified perspective** — taxonomy table, conceptual comparison
9. **Review** — verify correctness of all math, Obsidian TOC links, notation consistency, every exercise has a solution
10. **Commit** — `git add` + `git commit`
