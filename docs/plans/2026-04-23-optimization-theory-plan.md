# Implementation Plan: Convex Optimization and Lagrangian Duality

**Date:** 2026-04-23
**Slug:** `optimization-theory`
**Output:** `concepts/optimization-theory/note.md`

## Tasks

1. **Write §1: Convex Sets and Convex Functions**
   - Definitions, examples (halfspaces, balls, polyhedra, epigraphs)
   - Operations preserving convexity
   - First-order condition: $f(y) \ge f(x) + \nabla f(x)^\top(y-x)$
   - Second-order condition: $\nabla^2 f(x) \succeq 0$
   - Inline exercises: identifying convex functions, proving log-sum-exp convexity

2. **Write §2: Convex Optimization Problems**
   - Standard form: minimize $f_0(x)$ subject to $f_i(x) \le 0$, $h_j(x) = 0$
   - LP, QP, SOCP as instances
   - Optimality criterion for unconstrained problems
   - Inline exercises: casting a QP in standard form

3. **Write §3: The Lagrangian**
   - Definition: $L(x, \lambda, \nu) = f_0(x) + \sum_i \lambda_i f_i(x) + \sum_j \nu_j h_j(x)$
   - Lagrangian relaxation: lower bound for any $\lambda \ge 0$
   - The dual function $g(\lambda, \nu) = \inf_x L(x, \lambda, \nu)$
   - Inline exercises: computing the dual function for a QP

4. **Write §4: Lagrangian Duality**
   - The dual problem: maximize $g(\lambda, \nu)$ subject to $\lambda \ge 0$
   - Weak duality: $g(\lambda, \nu) \le p^*$ always
   - Duality gap $p^* - d^*$
   - Strong duality and Slater's condition (constraint qualification)
   - Inline exercises: weak duality proof, Slater's condition verification

5. **Write §5: KKT Conditions**
   - Stationarity: $\nabla_x L = 0$
   - Primal feasibility: $f_i(x^*) \le 0$, $h_j(x^*) = 0$
   - Dual feasibility: $\lambda_i^* \ge 0$
   - Complementary slackness: $\lambda_i^* f_i(x^*) = 0$
   - Necessity (strong duality + differentiability) and sufficiency (convex problem)
   - Inline exercises: applying KKT to a constrained QP, complementary slackness interpretation

6. **Write §6: Recovering Primal Solutions from the Dual**
   - Strong convexity: unique minimizer of $L(\cdot, \lambda^*, \nu^*)$ is $x^*$
   - Dual decomposition and distributed optimization sketch
   - Connection to click-shaping paper: computing personalized recommendations via dual variables
   - Inline exercises: primal recovery from dual for a strongly convex problem

7. **Review and cross-check**
   - TOC anchors match heading text exactly
   - Notation consistent throughout ($\lambda$ for inequality multipliers, $\nu$ for equality)
   - Every exercise has an inline solution
   - References table complete

## Notes

- Follow CLAUDE.md conventions: Obsidian wikilink TOC, no LaTeX in headings, Mermaid over ASCII, emojis at section headings
- Exercises split: ~16–18 Mathematical Development + ~5–7 Algorithmic Applications, numbered continuously
- Solutions use [!TIP]- collapsible callouts
