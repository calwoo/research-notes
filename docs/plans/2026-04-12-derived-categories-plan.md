# Implementation Plan: Derived Categories Cluster

**Date:** 2026-04-12
**Design doc:** `docs/plans/2026-04-12-derived-categories-design.md`

## Tasks

1. **Scaffold** — Create `concepts/category-theory/derived-categories/` directory and `overview.md` index (cluster map, dependency graph, master references)
2. **Note 1: triangulated-categories.md** — Write the first note following the section outline in the design doc; exercises inline after each section
3. **Note 2: construction.md** — Chain complexes, homotopy category K(A), localization at quasi-isomorphisms, D±/Db
4. **Note 3: derived-functors.md** — RF/LF via resolutions, spectral sequences, examples
5. **Note 4: geometric.md** — D(X), six functors, geometric applications
6. **Cross-check** — Verify TOC wikilink anchors resolve, notation consistent across all notes, every exercise has an inline solution, no LaTeX in headings
7. **Commit** — `feat: add derived-categories cluster — overview and triangulated categories note`

## Note-Writing Checklist (per note)

- [ ] TOC with exact wikilink anchors (`[[#Exact Heading|Display]]`)
- [ ] No LaTeX or em-dashes in headings
- [ ] Exercises inline after each section in `[!QUESTION]` callouts
- [ ] Solutions inline in `[!TIP]-` collapsible callouts
- [ ] At least one Obsidian callout per major section
- [ ] References table at end
- [ ] Figures fetched and embedded via image-extractor agent after writing
