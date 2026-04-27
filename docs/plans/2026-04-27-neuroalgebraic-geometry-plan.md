# Plan: Neuroalgebraic Geometry — Implementation

**Date:** 2026-04-27
**Design doc:** `docs/plans/2026-04-27-neuroalgebraic-geometry-design.md`

## Tasks

- [ ] 1. Create topic directory `concepts/algebraic-geometry/neuroalgebraic-geometry/`
- [ ] 2. Fetch and parse all seed documents (2501.18915, 2211.10049, 2010.11560, 2406.10234, IPAM, Watanabe homepage)
- [ ] 3. Run `reference-finder` subagent to build master reference list
- [ ] 4. Write `overview.md` — topic index, subtopic map, dependency graph, master references
- [ ] 5. Write `singular-learning-theory.md` via `note-writer` subagent
- [ ] 6. Review note for correctness: RLCT derivation, Bayes free energy expansion, notation consistency
- [ ] 7. Final cross-check: TOC anchors, inline exercises/solutions, Mermaid diagrams, wikilinks
- [ ] 8. Commit all files

## Notes

- First note: `singular-learning-theory.md` (Watanabe's framework — most mathematically developed thread)
- Subsequent notes (separate sessions): `expressivity-and-complexity.md`, `loss-landscape-geometry.md`, `algebraic-statistics-connections.md`
- Seed documents should inform overview.md scope; Watanabe homepage has RLCT table for specific models
