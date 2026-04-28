# Phase III — Parameter Efficiency & Golf

*Weeks 15–22 · ~16 hrs*

> **Goal:** Understand what makes parameters "count" — the scaling laws, architectural choices, and efficiency techniques that let you do more with fewer parameters. By the end, you will have made at least five tracked submissions to the OpenAI Parameter Golf benchmark, with a documented record of what each change did and why.

**Weeks 15–16 primary:** Fleuret, [*The Little Book of Deep Learning*](https://fleuret.org/public/lbdl.pdf), Ch 5 (architectures) + Hoffmann et al., [Chinchilla paper](https://arxiv.org/abs/2203.15556) §1–3 (key results only).

**Weeks 17–18 primary:** Karpathy, [nanoGPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) (1h 56m).

**Weeks 19–20 primary:** OpenAI, [Parameter Golf](https://openai.com/index/parameter-golf/) (blog) — first submission and iteration.

**Weeks 21–22:** Open-ended experimentation — no new resources; iterate on the benchmark.

> **Why this structure:** Parameter counting and scaling laws (Weeks 15–16) give you the mental model for what a "budget" of parameters buys. nanoGPT (Weeks 17–18) is a clean transformer implementation that introduces the key efficiency techniques (weight tying, attention structure) in a concrete, readable codebase. The last four weeks are pure applied science: form a hypothesis, implement it, measure the result.

---

## Weeks 15–16 — Counting Parameters and Scaling Laws

**Primary resources:**
- Fleuret, [*The Little Book of Deep Learning*](https://fleuret.org/public/lbdl.pdf), Ch 5 (~1 hr)
- Hoffmann et al., ["Training Compute-Optimal LLMs" (Chinchilla)](https://arxiv.org/abs/2203.15556), §1 Introduction + §3 Results only (~30 min — skip the methodology appendix)

---

### Week 15 — Parameter Counting by Hand

**Concepts to understand:**

- [ ] Linear layer `Linear(d_in, d_out, bias=True)`: `d_in × d_out + d_out` parameters — the `+d_out` for the bias vector
- [ ] Embedding table `Embedding(vocab_size, d_model)`: `vocab_size × d_model` parameters — one vector per token
- [ ] Multi-head attention `MultiheadAttention(d_model, n_heads)`: four weight matrices (Q, K, V, output projection), each `d_model × d_model` — `4 × d_model²` parameters total (ignoring bias)
- [ ] LayerNorm `LayerNorm(d_model)`: `2 × d_model` parameters (scale `γ` and shift `β`)
- [ ] MLP block (two linear layers with expansion factor 4): `d_model × 4d_model + 4d_model × d_model = 8d_model²` parameters
- [ ] A single transformer block: attention (`4d²`) + MLP (`8d²`) + two LayerNorms (`4d`) ≈ `12d²` parameters for large `d`

**Coding tasks:**

- [ ] For a GPT-2 small (d=768, n_heads=12, n_layers=12, vocab=50257): estimate total parameter count by hand; verify with `sum(p.numel() for p in model.parameters())` on a loaded model
- [ ] Count parameters in your makemore MLP; identify which component dominates (embedding vs. hidden layers)

> [!NOTE] Milestone
> GPT-2 small has 117M parameters. Manual estimate: embedding table = 50257 × 768 ≈ 38.6M; 12 transformer blocks × (4 × 768² + 8 × 768²) ≈ 12 × 8.85M ≈ 106M; output head shares embedding (weight tying) so costs nothing extra; total ≈ 38.6 + 106 ≈ 125M (the ≈8M discrepancy comes from biases and positional embeddings). The key takeaway: the embedding table is a large fraction of parameters in small models — weight tying (sharing input and output embedding) removes it from the parameter count at zero cost in expressivity.

---

### Week 16 — Scaling Laws and the Chinchilla Result

**Concepts to understand:**

- [ ] The empirical scaling law: `L(N, D) ≈ A/N^α + B/D^β + L∞` where `N` = parameters, `D` = training tokens; loss decreases as a power law in both
- [ ] The Kaplan et al. (GPT-3 era) finding: scale parameters, keep data fixed; optimal model for a compute budget is roughly `N ∝ C^(0.73)`
- [ ] The Chinchilla correction: Kaplan et al. were wrong. Optimal training uses `N ∝ C^(0.5)` and `D ∝ C^(0.5)` — the optimal ratio is approximately 20 training tokens per parameter
- [ ] Implication: GPT-3 (175B params, ~300B tokens) was significantly undertrained; a 70B model trained on 1.4T tokens achieves the same loss at half the parameters — Chinchilla/Llama result
- [ ] Width vs. depth at fixed parameter count: wider models (more d_model, fewer layers) tend to converge faster per step; deeper models (more layers, smaller d_model) can represent more compositional functions but train slower

**Coding tasks:**

- [ ] Build two makemore MLPs with identical parameter counts: one wide+shallow (large hidden dim, 2 layers) and one narrow+deep (small hidden dim, 8 layers). Train both and compare convergence speed and final val loss.
- [ ] Predict which will converge faster before running; verify your prediction

> [!NOTE] Milestone
> Expected result: the wide+shallow model (2 layers) converges significantly faster per step and often to a similar or slightly worse final val loss than the narrow+deep model (8 layers). This is consistent with the Chinchilla intuition — depth gives expressivity, but width gives learning speed. For makemore (a simple distribution over character sequences), two layers is already sufficient capacity; the narrow 8-layer model will be slower to train because early layers receive weak gradient signal even with residuals. For harder tasks (e.g., arithmetic, code), deeper networks may win on final loss at the cost of training time.

---

## Weeks 17–18 — Efficient Architectures: nanoGPT

**Primary resource:** Karpathy, [nanoGPT video](https://www.youtube.com/watch?v=kCc8FmEb1nY) (1h 56m) + [nanoGPT source](https://github.com/karpathy/nanoGPT)

Code along with the video. Read the full `model.py` file after watching — it is ~300 lines and is the clearest clean-room transformer implementation available.

---

### Week 17 — The Transformer Architecture and Weight Tying

**Concepts to understand:**

- [ ] Self-attention: each token attends to all previous tokens; Q, K, V matrices each `d_model × d_head`; attention output `= softmax(QKᵀ / √d_head) V`
- [ ] Causal masking: the upper triangle of the attention matrix is set to -∞ before softmax, so token `i` only sees tokens `0, …, i-1`
- [ ] Weight tying: the input embedding matrix `wte` (shape `vocab × d_model`) is reused as the output projection (the "language model head"); instead of a separate `Linear(d_model, vocab)`, the logits are `h @ wte.T`
- [ ] Why weight tying works: the embedding matrix already encodes semantic similarity between tokens in its rows; the output projection benefits from the same prior; and it saves `vocab × d_model` parameters (≈38M for GPT-2 small) at no loss in expressivity
- [ ] Parameter savings from weight tying in small models: for a model with `vocab=10000` and `d_model=256`, weight tying saves `10000 × 256 = 2.56M` parameters — potentially 30–50% of total model size

**Coding tasks:**

- [ ] Clone nanoGPT; read `model.py` end to end; annotate each module with its parameter count and role
- [ ] Verify weight tying: confirm `model.transformer.wte.weight is model.lm_head.weight` in the nanoGPT codebase (it is tied in the constructor)
- [ ] Untie the weights (add a separate `lm_head` linear layer); retrain on Shakespeare; compare final val loss and total parameter count

> [!NOTE] Milestone
> Untying weights in a small nanoGPT (d_model=64, n_layers=4, n_heads=4) adds `vocab × d_model` parameters. For GPT's default vocab of 65 (Shakespeare character-level): untying adds `65 × 64 = 4160` parameters — a negligible fraction of total. The val loss difference will also be negligible. The lesson: weight tying matters most for large vocabulary models (BPE tokenizers with vocab≈50K); at small vocab sizes, it is nearly irrelevant. Parameter Golf tasks likely use small-vocab settings where other efficiency techniques dominate.

---

### Week 18 — Efficiency Techniques: Low-Rank, Grouped Attention, Shared Layers

**Concepts to understand:**

- [ ] Low-rank factorization: replace a weight matrix `W ∈ ℝ^{m×n}` with `W ≈ UV` where `U ∈ ℝ^{m×r}`, `V ∈ ℝ^{r×n}`, `r ≪ min(m, n)`; parameter count drops from `mn` to `r(m+n)`. Breakeven: `r < mn/(m+n)`
- [ ] Grouped-query attention (GQA): use `n_kv_heads < n_heads` — the K and V matrices are shared across multiple Q heads; reduces KV parameters by `n_heads / n_kv_heads`. Multi-query attention (MQA) is the extreme case: `n_kv_heads = 1`
- [ ] Shared transformer blocks: reuse the same block weights at multiple depths (ALBERT-style); `L` layers share `1` block's worth of parameters; expressivity is reduced but parameter count is `1/L` of an unshared model
- [ ] Depthwise separable convolution (for CNN-based approaches): replace a `k×k` convolution with `C_in` input channels by a depthwise conv (`C_in × k × k` params) followed by a `1×1` pointwise conv (`C_in × C_out` params); total `C_in(k² + C_out)` vs. `C_in × C_out × k²` — typically 8–9× fewer params for `k=3`

**Coding tasks:**

- [ ] Implement multi-query attention (MQA) in nanoGPT: modify `CausalSelfAttention` to use `n_kv_heads=1`; verify the output shape is unchanged
- [ ] Implement layer sharing: make all `n_layer` transformer blocks point to the same `nn.Module` instance; count total parameters before and after
- [ ] Table: for each technique (weight tying, MQA, layer sharing, low-rank with r=d/4), compute the parameter reduction percentage for a model with d=256, n_heads=4, n_layers=4, vocab=65

> [!NOTE] Milestone
> Expected table for d=256, n_layers=4, n_heads=4, vocab=65:
>
> | Technique | Params before | Params after | Reduction |
> |-----------|--------------|--------------|-----------|
> | Baseline | ~800K | — | — |
> | Weight tying | ~800K | ~783K | ~2% (tiny vocab) |
> | MQA (1 KV head) | ~800K | ~650K | ~19% |
> | Layer sharing | ~800K | ~230K | ~71% |
> | Low-rank (r=64) | ~800K | ~650K | ~19% |
>
> Layer sharing gives the largest reduction but is the most aggressive — the same weights must learn to be useful at every depth simultaneously, which constrains expressivity significantly. This is the key engineering trade-off in Parameter Golf: you need to find techniques that reduce parameter count without proportionally reducing task performance.

---

## Weeks 19–20 — First Submission and Iteration

**Primary resource:** OpenAI, [Parameter Golf](https://openai.com/index/parameter-golf/) (blog) — read in full (30 min).

Re-read the blog post now that you have Phases I–III. Every paragraph should connect to something you have observed.

---

### Week 19 — Understanding the Benchmark

**Concepts to understand:**

- [ ] What Parameter Golf measures: minimize the number of parameters in a model while keeping task performance above a threshold — the score is the parameter count (lower is better)
- [ ] Benchmark-specific inductive biases: the right architectural prior depends on the task; a conv-based model is very parameter-efficient for image tasks; a small MLP may be optimal for low-dimensional regression tasks
- [ ] The compression vs. accuracy trade-off: there is always a Pareto frontier of (parameter count, accuracy); Parameter Golf asks you to move along this frontier as efficiently as possible
- [ ] One change at a time: in a competition setting, changing multiple things simultaneously makes it impossible to attribute a score change to any specific technique

**Coding tasks:**

- [ ] Read the Parameter Golf rules and scoring carefully
- [ ] Run the baseline model provided; record the score and parameter count
- [ ] Identify the three easiest wins: techniques that are low-implementation-effort and likely to reduce parameters without hurting accuracy

> [!NOTE] Milestone
> Before implementing anything, write down: (1) the baseline parameter count, (2) what fraction of parameters are in the embedding vs. the trunk (attention + MLP), (3) which of the techniques from Week 18 are applicable to this task's architecture. This pre-analysis takes 15 minutes and saves hours of misguided implementation. The biggest gains in parameter efficiency almost always come from the largest parameter blocks — if 60% of parameters are in the embedding, weight tying or a smaller vocabulary will dominate all other optimizations.

---

### Week 20 — Three Tracked Iterations

**No new concepts — this week is applied scientific method.**

For each iteration:

- [ ] State the hypothesis: "Applying X should reduce parameter count by Y% with at most Z% accuracy degradation because..."
- [ ] Implement exactly one change
- [ ] Record: parameter count before/after, score before/after, and one sentence on what you observed
- [ ] If the change improves score: keep it as the new baseline before the next iteration
- [ ] If the change hurts score: revert and document why it didn't work

**Iteration log template:**

| # | Technique | Params before | Params after | Score before | Score after | Notes |
|---|-----------|--------------|--------------|-------------|-------------|-------|
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |

> [!NOTE] Milestone
> After three iterations, the most important thing to have is a clear log, not a great score. If all three changes hurt the score, that is still valuable — you now know which techniques do not apply to this task. The most common mistake at this stage is changing something, seeing a worse score, reverting, and moving on without recording *why* it got worse. Writing down the reason forces you to reason about the mechanism, which is where the learning happens.

---

## Weeks 21–22 — Experiment Cycle

*No new resources — iterate on the benchmark using any techniques from the curriculum.*

**Efficiency technique reference:**

| Technique | What it does | Best when |
|-----------|-------------|-----------|
| Weight tying | Share input/output embedding | Large vocabulary |
| MQA / GQA | Fewer KV heads | Attention-heavy architectures |
| Layer sharing | Same block weights at every depth | Deep models where per-layer specialization is low |
| Low-rank factorization | Rank-r approximation of weight matrices | Matrices where singular values decay quickly |
| Depthwise separable conv | Factorize spatial + channel mixing | CNN-based approaches |
| Smaller hidden dim | Reduce d_model, add more layers | When width is the bottleneck |
| Smaller vocabulary | BPE or character-level | When vocab is a large parameter fraction |

**Capstone:** Document your final five iterations in the iteration log. For each, write one sentence explaining the mechanism behind the change and whether the result matched your prediction. Mispredictions are more informative than correct predictions — explain why reality differed from the hypothesis.

> [!NOTE] Milestone
> A complete capstone log has: (1) a baseline entry, (2) at least 5 further iterations, (3) at least one technique that helped and one that hurt, (4) a clear final score vs. baseline score, and (5) a one-paragraph summary of what you learned about *this specific task's* structure from your experiments. The paragraph should say something specific — not "weight tying helped" but "weight tying reduced parameters by 23% with no accuracy loss because the vocabulary is large (50K tokens) and the task does not require fine-grained token discrimination in the output projection."
