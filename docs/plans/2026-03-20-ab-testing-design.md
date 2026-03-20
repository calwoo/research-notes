# Design Doc: A/B Testing Concept Notes

**Date:** 2026-03-20
**Status:** Approved

## Motivation

A/B testing is foundational to modern data-driven product development. A rigorous mathematical treatment requires three interlocking frameworks: (1) the frequentist hypothesis-testing formalism, (2) Bayesian posterior inference, and (3) sequential/adaptive methods including multi-armed bandits. Each has distinct assumptions, decision rules, and tradeoffs that practitioners must understand deeply.

## Scope

This topic lives at `concepts/ab-testing/` and covers:

- **Foundations** — the causal and statistical framework underlying all A/B tests
- **Frequentist testing** — z-tests, t-tests, sample size, confidence intervals
- **Bayesian testing** — Beta-Binomial conjugate model, expected loss, posterior inference
- **Experimental design** — CUPED variance reduction, multiple testing correction
- **Sequential and adaptive methods** — peeking problem, group sequential designs, bandits, Thompson sampling

## Design Principles

### Math-first style
All notes follow the repo's math-first style: rigorous definitions, formal notation (ATE, potential outcomes, conjugate posteriors), and derivations over high-level summaries.

### Obsidian compatibility
- TOC uses `[[#Heading Text|Display Text]]` wikilink syntax throughout
- No LaTeX in headings; no em-dashes in headings
- References table at end of each note

### Multi-file layout
The topic is broad enough to warrant five distinct note files, each with a focused scope. A single `exercises.md` + `solutions.md` pair covers all five notes.

## File Map

| File | Scope |
|------|-------|
| `foundations.md` | Causal estimand, hypothesis framework, errors, p-values, power, MDE |
| `frequentist-testing.md` | Z-test, t-test, CIs, sample size, ANOVA |
| `bayesian-testing.md` | Beta-Binomial model, posterior inference, expected loss |
| `experimental-design.md` | Randomization, CUPED, Bonferroni, Benjamini-Hochberg |
| `sequential-and-adaptive.md` | Peeking, GSD, alpha-spending, bandits, UCB, Thompson sampling |
| `exercises.md` | 16–18 mathematical + 5–7 algorithmic problems |
| `solutions.md` | Key insight + Sketch format |

## Cross-Cutting Concerns

- **CUPED** (in `experimental-design.md`) depends on understanding the variance of the estimator from `frequentist-testing.md`
- **Sequential validity of Bayesian tests** (in `bayesian-testing.md`) motivates `sequential-and-adaptive.md`
- **Expected loss** is the Bayesian decision criterion; its frequentist analog is power/MDE in `foundations.md`
