# Implementation Plan: A/B Testing Concept Notes

**Date:** 2026-03-20
**Design doc:** `docs/plans/2026-03-20-ab-testing-design.md`

## Files to Create

| File | Purpose |
|------|---------|
| `concepts/ab-testing/foundations.md` | Statistical hypothesis framework, errors, p-values, power, effect size |
| `concepts/ab-testing/frequentist-testing.md` | Z-tests, t-tests, sample size, confidence intervals |
| `concepts/ab-testing/bayesian-testing.md` | Beta-Binomial model, conjugate updates, expected loss, stopping |
| `concepts/ab-testing/experimental-design.md` | Randomization, CUPED variance reduction, multiple testing (BH, Bonferroni) |
| `concepts/ab-testing/sequential-and-adaptive.md` | Peeking problem, GSD, alpha-spending, bandits, Thompson sampling, regret |
| `concepts/ab-testing/exercises.md` | Problem set (Mathematical Development + Algorithmic Applications) |
| `concepts/ab-testing/solutions.md` | Full answer key |

## Execution Steps

1. [x] Create `docs/plans/2026-03-20-ab-testing-design.md`
2. [x] Create `docs/plans/2026-03-20-ab-testing-plan.md` (this file)
3. [ ] Write `foundations.md` via note-writer agent
4. [ ] Write `frequentist-testing.md` via note-writer agent
5. [ ] Write `bayesian-testing.md` via note-writer agent
6. [ ] Write `experimental-design.md` via note-writer agent
7. [ ] Write `sequential-and-adaptive.md` via note-writer agent
8. [ ] Write `exercises.md` + `solutions.md` via exercise-writer agent
9. [ ] Git commit: `feat: add ab-testing concept notes — 5 notes, exercises, solutions`

## Note Structure

### foundations.md
1. Motivation — why randomized experiments are the gold standard for causal claims
2. The Causal Estimand — ATE, potential outcomes framing
3. Statistical Hypotheses — null/alternative, one-tailed vs. two-tailed
4. Type I and Type II Errors — formal definitions, error rates α and β
5. P-values — definition via null distribution, common misconceptions
6. Statistical Power — definition, the power function, factors affecting power
7. Effect Size and MDE — Cohen's d, relative lift, MDE derivation
8. References

### frequentist-testing.md
1. The Test Statistic Framework — sufficient statistics, pivotal quantities
2. Proportions: The Z-test — normal approximation, pooled vs. unpooled variance
3. Continuous Metrics: The T-test — Student's t, Welch's t, assumptions
4. Confidence Intervals — duality with hypothesis tests, construction
5. Sample Size Formulas — derivation from power constraints
6. Multiple Variants — one-way ANOVA, F-statistic
7. References

### bayesian-testing.md
1. The Bayesian Framework — prior, likelihood, posterior; Bayes' theorem
2. Beta-Binomial Model — Beta prior, Binomial likelihood, Beta posterior conjugate update
3. Posterior Inference — credible intervals, posterior predictive
4. Decision Rules — P(p_B > p_A), expected loss
5. Prior Selection — uninformative (Uniform/Jeffreys), informative priors
6. Bayesian vs. Frequentist — interpretability, stopping rules, sequential validity
7. References

### experimental-design.md
1. Randomization Mechanisms — Bernoulli vs. complete randomization, hash-based assignment
2. Stratification and Blocking — variance reduction via pre-stratification
3. CUPED — covariate adjustment, regression identity, variance reduction formula
4. Multiple Testing: FWER — Bonferroni, Holm-Bonferroni
5. Multiple Testing: FDR — Benjamini-Hochberg, when to use FDR vs. FWER
6. Metric Selection — sensitivity, directionality, guardrail vs. primary metrics
7. References

### sequential-and-adaptive.md
1. The Peeking Problem — why naive interim looks inflate Type I error
2. Group Sequential Designs — information fraction, Pocock and O'Brien-Fleming boundaries
3. Alpha-Spending Functions — Lan-DeMets framework
4. The Multi-Armed Bandit Problem — formal setup, regret, exploration-exploitation
5. Epsilon-Greedy and UCB — algorithms, regret bounds O(√T log T)
6. Thompson Sampling — beta-Bernoulli Thompson sampling, Bayesian regret
7. Bandits vs. Fixed A/B Tests — when each is appropriate
8. References

## Verification Checklist

- [ ] All TOC links use `[[#Heading Text|Display Text]]` Obsidian wikilink syntax
- [ ] No LaTeX or em-dashes in headings
- [ ] `exercises.md` has two sections: Mathematical Development (16–18 problems) + Algorithmic Applications (5–7 problems), numbered continuously
- [ ] Each problem has italic preamble + Prerequisites blockquote
- [ ] `solutions.md` uses Key insight + Sketch format
- [ ] All files committed in a single git commit
