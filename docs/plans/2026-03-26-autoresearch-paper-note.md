# Plan: Autoresearch Paper Note

**Date:** 2026-03-26
**Status:** Completed

## Objective

Write a paper-style concept note at `papers/autoresearch.md` synthesizing five practitioner sources on autoresearch — the practice of using LLMs in agentic loops to autonomously conduct research.

## Sources Consulted

1. https://ykumar.me/blog/eclip-autoresearch/ — 403 (blocked)
2. https://www.datacamp.com/tutorial/guide-to-autoresearch — 403 (blocked)
3. https://www.philschmid.de/autoresearch — fetched successfully
4. https://sidsaladi.substack.com/p/autoresearch-101-builders-playbook — fetched successfully
5. https://thecreatorsai.com/p/autoresearch-the-loop-that-improves — fetched successfully

Supplemented with:
- github.com/karpathy/autoresearch (primary technical source)
- arxiv.org/html/2508.12752v1 (Deep Research survey)
- arxiv.org/html/2505.18705v1 (AI-Researcher paper)
- latent.space/p/ainews-autoresearch-sparks-of-recursive
- kingy.ai autoresearch summary

## Structure Implemented

- Paper-style header: title → authors/venue → TL;DR table → Relations → TOC
- 7 body sections + references table
- Information-theoretic motivation for iterative retrieval (§1.3)
- Formal pseudocode for the core loop (§2.2)
- Taxonomy of failure modes with mitigations (§5)
- Coverage of 5 major open-source implementations (§6)
- Formal evaluation dimensions with equations (§7.3)

## Output

`papers/autoresearch.md` — ~1000 lines
