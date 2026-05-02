# Plan: Deep Learning Engineering — Initial Note (weight-tying.md)

**Date:** 2026-05-02
**Design doc:** `docs/plans/2026-05-02-deep-learning-engineering-design.md`

## Tasks

1. **Create topic directory and overview.md** — scaffold `concepts/deep-learning-engineering/` with the multi-note index, subtopic map, dependency graph, and master references table
2. **Research weight tying** — gather sources: Press & Wolf (2017), Vaswani et al. (2017), ALBERT (factored embeddings), any theoretical analysis of tied vs. untied gradient flow
3. **Write `weight-tying.md`** — sections: Motivation → The Technique → Why It Works → Formal Analysis → When It Hurts → Variations; exercises and solutions inline after each section
4. **Review** — check TOC Obsidian wikilinks, notation consistency ($W_{\text{emb}}$, $W_{\text{out}}$), every exercise has an inline `[!TIP]-` solution
5. **Cross-check** — verify all section heading anchors match TOC entries exactly; confirm mermaid diagrams (if any) follow repo conventions
6. **Commit** — `git add` + descriptive commit message
