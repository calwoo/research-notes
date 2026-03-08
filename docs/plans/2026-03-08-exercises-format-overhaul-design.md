# Exercises Format Overhaul — Design Document

**Date:** 2026-03-08
**Scope:** All `exercises.md` and `solutions.md` files across the repo

---

## Background

The current exercises format has served as a starting point but has several gaps for deep problem-solving sessions:

- Three sections (Derivation / Conceptual / Implementation) split what is better understood as a math/applications divide
- No preamble to motivate each problem
- No prerequisite pointers linking back to the note or to prior problems
- Solutions are full worked derivations — thorough but slow to review
- Problem count (~13) is on the light side for a serious study session
- Conceptual questions are sometimes informal ("explain in your own words") rather than mathematically sharp

---

## Design Goals

1. **Deeper mathematical orientation** — more derivation-heavy problems, informal conceptual questions converted to mathematical ones
2. **Better navigation** — preambles + prerequisite pointers make it clear what each problem tests and where the theory lives
3. **Denser coverage** — 18–25 problems per topic (up from ~13)
4. **Faster solution review** — sketch-style solutions with a key insight line rather than full worked derivations

---

## Section Structure

### Two Sections (replacing three)

| Old | New |
|-----|-----|
| Derivation Problems | **Mathematical Development** |
| Conceptual Questions | (merged into Mathematical Development) |
| Implementation Sketches | **Algorithmic Applications** |

**Mathematical Development** (~16–18 problems): All derivations, proofs, limit arguments, variance calculations, fixed-point analyses, and mathematical conceptual results. Former "conceptual questions" are absorbed here but must be tightened to have mathematical content — informal "explain in your own words" questions become "prove that..." or "derive the condition under which...".

**Algorithmic Applications** (~5–7 problems): Pseudocode sketches, numerical implementation, shape annotations, gradient flow analysis. Format is unchanged from the current Implementation Sketches section.

Problems are numbered **continuously** across both sections (Problem 1 through Problem N, no restart).

---

## Problem Format

### exercises.md

Each problem follows this template:

```markdown
### Problem N: [Title]

*[1–2 sentence preamble: what this problem establishes and why it matters.
Written in italics. States the mathematical goal, not just the topic.]*

> **Prerequisites:** cf. note §X.Y — [Section title]; requires Problem M (optional)

(a) [First sub-part]

(b) [Second sub-part]

(c) [Third sub-part]
```

**Preamble rules:**
- Italicized, placed immediately after the problem heading
- States what the problem *establishes* (a result, a structure, a bound), not just what it's about
- 1–2 sentences maximum

**Prerequisite pointer rules:**
- Rendered as a blockquote for visual separation
- `cf. note §X.Y` links to the relevant section in `note.md` using Obsidian wikilink syntax: `[[note#Section Title|§X.Y — Section Title]]`
- `requires Problem M` used when the problem builds directly on a prior result
- Omit the `requires Problem M` part if there's no dependency

**Implementation problems** additionally use bold sub-part labels:

```markdown
(a) **Inputs and data structures**: ...
(b) **Forward pass**: ...
```

### solutions.md

Each solution follows this template:

```markdown
### Problem N: [Title]

**Key insight:** [1 sentence — the core mathematical observation, the pivotal trick, or the structural fact that makes the proof work]

**Sketch:**
[3–8 lines of the critical derivation steps. Skip routine algebra.
Show the pivotal move and the conclusion. Use displayed math where it clarifies.]
```

**Solutions rules:**
- No full worked derivations — approximately 30–40% of current solution length
- The "Key insight" line is mandatory and should be standalone-readable
- The sketch shows *which steps matter*, not every step
- The reader should be able to reconstruct the full derivation from the sketch if they understood the key insight

---

## TOC Format

```markdown
## Table of Contents

- [[#Mathematical Development|Mathematical Development]]
  - [[#Problem 1 Title|Problem 1: Title]]
  - [[#Problem 2 Title|Problem 2: Title]]
  - ...
- [[#Algorithmic Applications|Algorithmic Applications]]
  - [[#Problem 17 Title|Problem 17: Title]]
  - ...
```

Standard Obsidian wikilink format. No LaTeX or em-dashes in headings (per repo style rules).

---

## Problem Count Targets

| Section | Target |
|---------|--------|
| Mathematical Development | 16–18 |
| Algorithmic Applications | 5–7 |
| **Total** | **21–25** |

The extra problems relative to the current format come from:
- Converting informal conceptual questions into mathematically sharp derivations
- Adding limit/asymptotic analysis problems (e.g., behavior as k → E, as T → ∞)
- Adding generalization problems (e.g., extend a result from the special case in the note to the general case)
- Splitting overly long multi-part problems when they span two genuinely separate results

---

## What Does NOT Change

- Multi-part (a/b/c) structure is preserved
- No difficulty ratings
- Implementation sketches stay pseudocode-level (not math-level algorithm analysis)
- Balanced scope: roughly half problems track note derivations, half extend beyond them

---

## Migration Strategy for Existing Files

When retrofitting existing exercises files:

1. Audit all existing "Conceptual Questions" — convert mathematical ones into Mathematical Development, drop or convert informal ones
2. Add preambles to all existing problems (derivation + implementation)
3. Add prerequisite pointers, identifying the relevant note section for each problem
4. Add 5–10 new Mathematical Development problems to reach target count
5. Rewrite solutions.md with Key insight + Sketch format (discard full worked derivations)
6. Update TOC to reflect new section names and problem set

---

## Example

### Before

```markdown
### Problem 8: Expert Collapse and the Positive Feedback Loop

(a) Formalize the positive feedback loop: suppose expert i is initialized...

(b) The router z-loss is L_z = ... Compute ∂L_z/∂h_i(x)...

(c) Entropy regularization adds -λH(g(x))...
```

### After

```markdown
### Problem 8: Expert Collapse and the Positive Feedback Loop

*This problem formalizes the positive feedback mechanism behind routing collapse and
derives the gradient structure of two regularizers designed to counteract it.*

> **Prerequisites:** cf. note [[note#5.2 Load Balancing|§5.2 — Load Balancing]]; requires Problem 3

(a) Formalize the positive feedback loop: suppose expert i is initialized...

(b) The router z-loss is L_z = ... Compute ∂L_z/∂h_i(x)...

(c) Entropy regularization adds -λH(g(x))...
```

And in solutions.md:

```markdown
### Problem 8: Expert Collapse and the Positive Feedback Loop

**Key insight:** The z-loss gradient is proportional to g_i(x) · log Z(x), so it penalizes dominant experts (large g_i) most strongly while leaving near-zero experts alone — exactly the right direction to counteract collapse.

**Sketch:**
Differentiate L_z = (1/T) Σ_t (log Z(x_t))² with respect to h_i(x_t):
  ∂L_z/∂h_i = (2/T) · log Z · ∂(log Z)/∂h_i = (2/T) · log Z · g_i
For entropy: ∂(-H)/∂h_i = g_i(log g_i + H). Sign is positive (penalty) when g_i > e^{-H},
i.e., for above-average-weight experts. Contrast: z-loss operates on pre-softmax logits h,
entropy operates on post-softmax g; z-loss reuses log Z already computed during routing.
```
