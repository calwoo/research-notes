# Design: Efficient Dense Retrieval at Scale — Survey Note

**Date:** 2026-05-30
**Topic slug:** `efficient-dense-retrieval-survey`
**Category:** `papers`
**Multi-note:** no (single flat file)

## Scope

A survey of the efficiency-quality trade-off in dense retrieval — the task of encoding queries and documents into dense vectors for fast approximate nearest-neighbor search. The survey covers the full stack: ANN infrastructure, model architecture choices (encoder vs. pruned decoder), knowledge distillation, LLM-based data augmentation, and vector quantization. The anchor paper is DRAMA (arXiv 2502.18460), which sits at the intersection of multiple efficiency axes simultaneously: a pruned LLM backbone (<1B params) trained with diverse LLM-generated data.

The motivating tension: LLM-based retrievers (Llama-7B+ backbones) achieve state-of-the-art quality on BEIR/MTEB but are 40× more expensive at inference than BERT-sized encoders. Practical retrieval systems need <1B-parameter models. The survey organizes approaches by which cost axis they target — inference-time model size, training-data cost, or index-side compression.

## Files to Create

| File | Purpose |
|------|---------|
| `docs/plans/2026-05-30-efficient-dense-retrieval-survey-design.md` | This design doc |
| `docs/plans/2026-05-30-efficient-dense-retrieval-survey-plan.md` | Implementation plan |
| `papers/efficient-dense-retrieval-survey.md` | Survey note |

## Note Structure

1. **Introduction** — the efficiency-quality tension; why LLMs at inference are expensive; benchmarks (BEIR, MTEB, MIRACL)
2. **Background: Dense Retrieval Fundamentals** — bi-encoder, InfoNCE loss, FAISS/ANN infrastructure; evaluation metrics
3. **The Quality Frontier: LLM-Based Dense Retrievers** — RepLlama, E5-mistral, NV-Embed, Gecko, SFR-Embedding; what makes them strong; why they're expensive
4. **The Efficiency Baseline: Encoder-Based Retrievers** — Contriever, SimLM, E5, BGE-M3; multi-stage training; multilingual/long-context trade-offs
5. **Knowledge Distillation for Retrieval** — TAS-B; cross-encoder→bi-encoder KD; teacher selection effects
6. **LLM Data Augmentation: Shifting Inference Cost to Training** — InPars, Promptagator, Gecko's two-stage approach; DRAMA's systematic comparison of augmentation strategies
7. **Model Compression: Pruning Decoder-Only LLMs into Small Retrievers** — ShearedLlama structured pruning; DRAMA's pruning of Llama3.2-1B → 0.1B/0.3B; bidirectional attention for encoders
8. **Matryoshka Representation Learning: Flexible Dimension Reduction** — MRL formulation; nested losses; DRAMA's adoption of MRL for deployment flexibility
9. **Quantization and Index Efficiency** — JPQ joint encoder+PQ training; FAISS product quantization; binary/scalar embeddings
10. **DRAMA: Bringing It All Together** — how DRAMA combines pruned backbone + multi-strategy augmentation; ablation results; key numbers
11. **Open Problems and Frontiers** — speculative directions
12. **References**

## References

Anchor: DRAMA (arXiv 2502.18460). See reference-finder output for full 19-paper list.
