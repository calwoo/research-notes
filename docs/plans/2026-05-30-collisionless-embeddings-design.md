# Design: Collisionless Embedding Tables Cluster

**Date:** 2026-05-30
**Topic slug:** `collisionless-embeddings`
**Category:** `papers`
**Multi-note:** yes (2-paper cluster)

## Scope

Two ByteDance papers on collisionless embedding tables for large-scale recommendation systems:
- **Monolith** (arXiv 2209.07663, 2022): Foundational paper introducing cuckoo-hash-based collisionless embedding tables alongside a full online training architecture for TikTok's recommendation system.
- **MPZCH** (arXiv 2602.17050, 2026): Successor technique using linear probing with active eviction and optimizer state reset; achieves zero collisions at production scale (3B MAU).

## Files to Create

| File | Purpose |
|------|---------|
| `papers/collisionless-embeddings/monolith.md` | Monolith paper note |
| `papers/collisionless-embeddings/mpzch.md` | MPZCH paper note |

## References

- Monolith: https://arxiv.org/abs/2209.07663
- MPZCH: https://arxiv.org/abs/2602.17050
