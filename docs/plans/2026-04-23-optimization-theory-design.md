# Design: Convex Optimization and Lagrangian Duality Concept Note

**Date:** 2026-04-23
**Topic slug:** `optimization-theory`
**Category:** `concepts`
**Multi-note:** no

## Scope

This note covers the mathematical foundations of convex optimization and Lagrangian duality, providing the reader with the tools needed to understand constrained optimization problems that arise in machine learning and information retrieval. The exposition proceeds from first principles: convex sets and functions, standard problem forms, Lagrangian relaxation, weak and strong duality, and KKT optimality conditions.

The motivating application is the paper "Personalized click shaping through lagrangian duality for online recommendation" (Agarwal et al., SIGIR 2012), which casts personalized recommendation with competing objectives as a constrained convex optimization problem and exploits strong duality for efficient primal recovery from dual solutions. The note is designed so that a reader who has completed it can follow the paper's optimization arguments without difficulty.

## Files to Create

| File | Purpose |
|------|---------|
| `concepts/optimization-theory/note.md` | Single note covering convex sets/functions, standard problem forms, Lagrangian duality, KKT conditions, and the duality gap |

## Note Structure

1. **Convex Sets and Convex Functions** — definitions, examples, operations that preserve convexity, first/second-order characterizations
2. **Convex Optimization Problems** — standard form, LP/QP/SOCP/SDP as instances, feasibility and optimality
3. **The Lagrangian** — definition, Lagrangian relaxation, the dual function as a pointwise infimum
4. **Lagrangian Duality** — the dual problem, weak duality theorem, strong duality and Slater's condition
5. **KKT Conditions** — stationarity, primal/dual feasibility, complementary slackness; necessity and sufficiency
6. **Recovering Primal Solutions from the Dual** — how strong convexity enables primal recovery; relevance to click-shaping paper
7. **Exercises** (Mathematical Development + Algorithmic Applications, inline throughout)

## References

- Boyd & Vandenberghe, *Convex Optimization* (Cambridge, 2004) — canonical textbook
- Agarwal et al., "Personalized click shaping through lagrangian duality for online recommendation" (SIGIR 2012) — motivating paper
- Rockafellar, *Convex Analysis* (Princeton, 1970) — foundational theory
