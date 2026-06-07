# Design: RL for Value Modeling in Ranking

**Date:** 2026-06-07
**Topic slug:** `rl-value-modeling-ranking`
**Category:** `concepts/reinforcement-learning/`
**Multi-note:** no

## Scope

This note surveys reinforcement learning techniques for building *personalized value models* in search and recommendation ranking systems. The central problem is: given a deep network that produces per-task predictions (e.g., CTR, like probability, watch-time, shares), how do we learn to combine them into a single ranking score that optimizes long-term user value rather than short-term proxy metrics?

The survey covers the RL framing of this problem (MDP formulation, state/action/reward design), the main algorithmic families (contextual bandits, policy gradient, actor-critic, constrained MDPs), multi-objective reward scalarization theory, and practical considerations (off-policy correction, exploration, reward shaping) drawn from large-scale industrial systems at YouTube, ByteDance, Alibaba, and related venues.

## File to Create

| File | Purpose |
|------|---------|
| `concepts/reinforcement-learning/rl-value-modeling-ranking.md` | Single survey note covering the full topic |

## Note Structure

1. **The Value Modeling Problem** — formal setup: per-task predictions as features, long-term value as optimization target, MDP formulation
2. **Multi-Objective Reward Scalarization** — linear vs. nonlinear combination; when linear suffices (convex Pareto front); constrained MDP approach
3. **Contextual Bandits for Ranking** — LinUCB, NeuralUCB, off-policy correction; exploration-exploitation in industrial systems
4. **Policy Gradient Methods** — REINFORCE for recommendation (Chen et al. 2019), top-K off-policy correction, state-space formulations
5. **Constrained Actor-Critic** — CMDP framing, Lagrangian relaxation, two-stage constrained AC (Cai et al. 2022/2023)
6. **Learned Ranking Functions** — Wu et al. 2024 (YouTube LRF), multi-task fusion via RL (Zhang et al. 2022), long-term value prediction (Chen et al. 2026)
7. **Practical Considerations** — reward delay, position bias, feedback loops, simulator vs. live evaluation
8. **Summary and Taxonomy** — algorithm comparison table; when to use each approach

## References

See `docs/plans/2026-06-07-rl-value-modeling-ranking-plan.md` for the full reference list from the reference-finder agent.
