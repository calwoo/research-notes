# Design: Robotics Concept Cluster (Modern Robotics Foundations)

**Date:** 2026-08-02
**Topic slug:** `robotics`
**Category:** `concepts`
**Multi-note:** yes

## Scope

The math backbone that the [[curricula/robotics/curriculum|SO-101 Robotics Curriculum]] assumes but doesn't itself teach: rigid-body configuration, motion representations, kinematics, and dynamics, following Lynch & Park's *Modern Robotics* (the curriculum's primary textbook for Modules 0, 1, and 3). Each note in this cluster corresponds to one or two textbook chapters and is written with the repo's usual mathematical rigor — formal definitions, derivations, worked exercises — rather than a hand-wavy paraphrase of the book.

The first note, `configuration-space.md`, covers Ch. 1–2 (configuration space, degrees of freedom, Grubler's formula, task space vs. configuration space, holonomic/nonholonomic constraints) — the Module 0 orientation reading. It's being written now, while the physical SO-101 arm is in transit, so the curriculum's reading can proceed independent of hardware arrival.

## Files to Create

| File | Purpose |
|------|---------|
| `concepts/robotics/overview.md` | Cluster index: all planned chapter notes, subtopic map, dependency graph, master references |
| `concepts/robotics/configuration-space.md` | First note — Ch. 1–2, configuration space and DOF counting |

## Note Structure (configuration-space.md)

1. **Configuration space** — formal definition, examples (point, rigid body, linkage), configuration space as a manifold
2. **Degrees of freedom** — DOF of a rigid body in 2D/3D, DOF of a system of linked rigid bodies
3. **Grubler's formula** — derivation, worked examples on planar and spatial mechanisms
4. **Constraints** — holonomic vs. nonholonomic, how each affects configuration space dimension
5. **Task space vs. configuration space** — the distinction that matters for inverse kinematics later (Ch. 6 / Module 3)

Exercises are placed inline after each section per repo convention (e.g. a DOF-counting exercise after Grubler's formula, a holonomic/nonholonomic classification exercise after §4).

## Planned Subtopics (multi-note only)

Mirrors the curriculum's Module 0/1/3 reading sequence:

| File | Modern Robotics Ch. | Curriculum Module |
|------|---|---|
| `configuration-space.md` | 1–2 | Module 0 |
| `rigid-body-motions.md` | 3 | Module 1 |
| `forward-kinematics.md` | 4 | Module 1 |
| `velocity-kinematics.md` | 5 | Module 3 |
| `inverse-kinematics.md` | 6 | Module 3 |
| `dynamics.md` | 8 | Module 3 |

Only `configuration-space.md` is written now; the rest are placeholders in `overview.md`, to be written as the curriculum reaches each module.

## References

- **Modern Robotics** (Lynch & Park) — [free PDF](http://hades.mech.northwestern.edu/images/7/7f/MR.pdf), Ch. 1–2 primary source
- [modernrobotics.org](http://modernrobotics.org) — companion exercises/videos
- 16-811 Math Fundamentals for Robotics (Matt Mason, CMU) — supplementary rigor per curriculum Module 1, not directly needed for Ch. 1–2 but relevant to later notes in this cluster
