# Plan: RL for Value Modeling in Ranking — Concept Note

**Date:** 2026-06-07
**Target file:** `concepts/reinforcement-learning/rl-value-modeling-ranking.md`

## Tasks

1. Write §1: The Value Modeling Problem — MDP formulation, per-task predictions as state features
2. Write §2: Multi-Objective Reward Scalarization — linear combination, convex Pareto front theorem, constrained MDP formulation
3. Write §3: Contextual Bandits — LinUCB, NeuralUCB, off-policy correction
4. Write §4: Policy Gradient Methods — REINFORCE for rec (Chen 2019), top-K correction
5. Write §5: Constrained Actor-Critic — CMDP, Lagrangian relaxation, Cai et al. 2022/2023
6. Write §6: Learned Ranking Functions — YouTube LRF (Wu 2024), multi-task fusion RL (Zhang 2022), LTV prediction (Chen 2026)
7. Write §7: Practical Considerations — reward delay, position bias, feedback loops
8. Write §8: Summary taxonomy table
9. Review for notation consistency, TOC anchors, inline exercises after each section
10. Commit

## References (from reference-finder)

| Reference Name | Brief Summary | Link |
|---|---|---|
| Li et al. (2010), "A Contextual-Bandit Approach to Personalized News Recommendation" | Introduces LinUCB — foundational contextual bandit for large-scale recommendation; UCB exploration from ridge regression | https://arxiv.org/abs/1003.0146 |
| Zheng et al. (2018), "DRN: A Deep Reinforcement Learning Framework for News Recommendation" | First deep Q-learning for personalized news rec; MDP formulation optimizing click reward and long-term user return | https://dl.acm.org/doi/10.1145/3178876.3185994 |
| Chen et al. (2019), "Top-K Off-Policy Correction for a REINFORCE Recommender System" | Scales REINFORCE to YouTube production; top-K off-policy correction for logged data. Canonical industrial REINFORCE-for-rec paper | https://arxiv.org/abs/1812.02353 |
| Ie et al. (2019), "SlateQ: Reinforcement Learning for Slate-based Recommender Systems" | Decomposes Q-value of recommendation slate into tractable per-item long-term values; validated on YouTube | https://arxiv.org/abs/1905.12767 |
| Zhou et al. (2020), "Neural Contextual Bandits with UCB-based Exploration" | NeuralUCB — first neural contextual bandit with provable near-optimal regret O(√T) | https://arxiv.org/abs/1911.04462 |
| Roijers et al. (2013), "A Survey of Multi-Objective Sequential Decision-Making" | Foundational MORL survey: taxonomy by scalarization type and solution concept; when linear scalarization fails | https://arxiv.org/abs/1402.0590 |
| Christiano et al. (2017), "Deep Reinforcement Learning from Human Preferences" | RLHF: reward model from human pairwise comparisons via Bradley-Terry; seed of modern RLHF pipeline | https://arxiv.org/abs/1706.03741 |
| Afsar et al. (2022), "Reinforcement Learning Based Recommender Systems: A Survey" | Comprehensive taxonomy of RL-for-recommendation: state representation, policy optimization, reward formulation | https://arxiv.org/abs/2101.06286 |
| Hayes et al. (2022), "A Practical Guide to Multi-Objective Reinforcement Learning and Planning" | Structured guide to MORL: scalarization-based vs. Pareto-based; when linear combination fails | https://arxiv.org/abs/2103.09568 |
| Zhang et al. (2022), "Multi-Task Fusion via RL for Long-Term User Satisfaction in Recommender Systems" | KDD 2022: fuses per-task predictions (clicks, likes, shares, play time) into ranking score via batch RL targeting long-term user stickiness | https://arxiv.org/abs/2208.04560 |
| Zhu & Van Roy (2021), "Deep Exploration for Recommendation Systems" | Sequential recommendation as MDP; deep exploration substantially outperforms bandit exploration | https://arxiv.org/abs/2109.12509 |
| Cai et al. (2022), "Constrained RL for Short Video Recommendation" | Constrained MDP for short video: auxiliary-signal policies constrain main watch-time policy; avoids explicit scalarization weights | https://arxiv.org/abs/2205.13248 |
| Cai et al. (2023), "Two-Stage Constrained Actor-Critic for Short Video Recommendation" | WWW 2023: two-stage AC optimizing watch time under interaction constraints; deployed in production | https://arxiv.org/abs/2302.01680 |
| Wu et al. (2024), "Learned Ranking Function: From Short-term Behavior Predictions to Long-term User Satisfaction" | RecSys 2024 (YouTube LRF): per-task behavioral predictions as inputs; learns combination targeting long-term user satisfaction | https://arxiv.org/abs/2408.06512 |
| Zhang et al. (2023), "Unified Off-Policy Learning to Rank: A Reinforcement Learning Perspective" | NeurIPS 2023: unifies ranking under stochastic click models as MDP; offline RL learns debiased rankers | https://arxiv.org/abs/2306.07528 |
| Jeunen et al. (2024), "Multi-Objective Recommendation via Multivariate Policy Learning" | RecSys 2024: scalarization weights as continuous actions in policy learning; maximizes pessimistic lower bound on north-star reward | https://arxiv.org/abs/2405.02141 |
| Chen et al. (2026), "A Long-term Value Prediction Framework in Video Ranking" | WWW 2026 (Alibaba): multi-task augmentation jointly predicting short-term engagement and LTV with position-aware debias | https://arxiv.org/abs/2602.17058 |
| Xiao & Wang (2024), "Towards Off-Policy RL for Ranking Policies with Human Feedback" | Off-policy value ranking algorithm unifying long-term reward and NDCG in a single EM framework | https://arxiv.org/abs/2401.08959 |
