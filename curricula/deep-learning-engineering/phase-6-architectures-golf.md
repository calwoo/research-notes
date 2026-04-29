# Phase VI — Novel Architectures & Parameter Golf

*Weeks 26–30 · ~25 hrs*

> **Goal:** Understand and implement the non-standard architectures that appear in the top Parameter Golf submissions. Mamba / SSMs, Test-Time Training (TTT) layers, depth recurrence, and the Muon optimizer's relationship to orthogonal gradient updates are the key ideas. By the end of Week 30, you will have run a systematic Parameter Golf campaign with a documented experiment log.

**Week 26 primary:** Gu & Dao, [Mamba paper](https://arxiv.org/abs/2312.00752) §1–3 + [mamba-minimal](https://github.com/johnma2006/mamba-minimal)

**Week 27 primary:** Sun et al., [TTT paper](https://arxiv.org/abs/2407.04620) + TTT-linear implementation

**Week 28 primary:** Dehghani et al., [Universal Transformers](https://arxiv.org/abs/1807.03819) + depth recurrence variants in Parameter Golf submissions

**Week 29 primary:** KellerJordan, [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt) — deep read of the full training pipeline; the Muon optimizer + architecture choices are the top-scoring open-source entry

**Week 30:** Parameter Golf campaign — no new primary resources; systematic experimentation

---

## Week 26 — State Space Models: Mamba

**Primary resources:**
- Gu & Dao, [Mamba paper](https://arxiv.org/abs/2312.00752) — §1 (motivation), §3 (selective SSM mechanism), §4 (hardware-efficient algorithm); skip §2 (prior SSM background unless you want deep theory)
- [mamba-minimal](https://github.com/johnma2006/mamba-minimal) — ~200 lines of clean PyTorch; read end to end

**Concepts to understand:**

- [ ] The core motivation: attention is `O(N²)` in sequence length; SSMs are `O(N)` in training (parallel convolutional form) and `O(1)` per step in inference (recurrent form); this enables long context at lower FLOPs
- [ ] The state space model: a latent state `h ∈ ℝ^d` evolves as `h_t = Ah_{t-1} + Bx_t`, `y_t = Ch_t`; the matrices `A, B, C` are fixed for classical SSMs; the key insight is that this is a linear recurrence, which can be parallelized as a scan during training
- [ ] Selective SSM (S6): the core Mamba innovation: make `B, C` and the time step `∆` input-dependent: `B_t = f_B(x_t)`, `C_t = f_C(x_t)`, `∆_t = f_∆(x_t)`; this breaks the linear recurrence (since `A, B, C` vary per step), but the discretized form can be efficiently computed with a parallel associative scan
- [ ] The hardware-efficient algorithm: Mamba's selective scan is memory-bandwidth-bound in the naive implementation; the paper's CUDA kernel fuses the discretization, scan, and output computation into a single kernel with careful recomputation (similar to FlashAttention's strategy for avoiding large HBM writes)
- [ ] Parameter efficiency: a Mamba block replaces the attention mechanism; for the same model dimension `d_model`, a Mamba block has approximately `6d²` parameters (similar to a transformer MLP block) while replacing both attention and MLP; this can halve model size at the same model dimension vs. a transformer block

**Coding tasks:**

- [ ] Read mamba-minimal end to end; run it and verify generation; trace through a single forward pass by hand, computing the shapes at each step
- [ ] Implement a Mamba block in your nanoGPT codebase: replace one transformer block (attention + MLP) with a Mamba block; train on Shakespeare; compare perplexity per parameter to the transformer baseline
- [ ] Compare inference speed: measure tokens/sec for Mamba vs. transformer at the same parameter count and at `seq_len ∈ {256, 1024, 4096}`; observe that transformer slows down quadratically while Mamba stays linear

> [!NOTE] Milestone
> At `seq_len=256` and `seq_len=512`: transformer and Mamba should have similar generation speed (the quadratic factor is small at short contexts). At `seq_len=4096`: transformer generation is ~16× slower than at `seq_len=1024` (quadratic scaling); Mamba is ~4× slower (linear scaling). The memory difference is more dramatic: at `seq_len=4096` with batch=8, the transformer attention matrix requires ~2GB; Mamba's state requires only `d_state × d_model × 2 bytes × batch ≈ 16 × 256 × 2 × 8 = 65KB` — four orders of magnitude smaller. For Parameter Golf tasks that evaluate on long sequences, Mamba-based architectures are parameter-efficient in a way transformers fundamentally cannot be.

---

## Week 27 — Test-Time Training (TTT) Layers

**Primary resource:** Sun et al., [TTT paper](https://arxiv.org/abs/2407.04620) — §1–4; focus on the TTT-Linear and TTT-MLP variants

**Concepts to understand:**

- [ ] The TTT motivation: standard RNNs compress all context into a fixed-size hidden state — expressivity is limited by state size; attention has unlimited expressivity but quadratic cost; TTT proposes a middle ground: the hidden state is a set of model *weights*, updated via gradient steps at test time
- [ ] The TTT layer: maintain a small "inner model" `f_θ` (e.g., a linear layer); the hidden state is `θ` itself; for each input token `x_t`, compute a self-supervised update rule on `θ` using `x_t`; output `y_t = f_{θ_t}(x_t)`; the update and the output can both be computed in a single pass using the TTT gradient formula
- [ ] TTT-Linear: `f_θ(x) = Wx` where `W ∈ ℝ^{d×d}`; the inner model is a single linear layer; the hidden state is the weight matrix `W_t`; the update rule is gradient descent on a self-supervised loss; after training, the outer (main) network learns what self-supervised task to apply at test time
- [ ] Why TTT is parameter-efficient: the inner model's weights are not model parameters — they are a dynamic hidden state; the only parameters are the outer network weights that define the update rule (`Q, K, V`-like projections); the effective context capacity scales with the inner model size, not parameter count
- [ ] TTT in Parameter Golf context: top entries combine TTT layers with depth recurrence; a single TTT block's weights can be shared across all depths (the dynamic state is different at each depth, but the static parameters are shared)
- [ ] `torch.autograd.grad` vs `.backward()`: `grads = torch.autograd.grad(loss, params)` returns gradients as a tuple **without accumulating into `.grad`**; `create_graph=False` (the default) is correct for the TTT inner update — the outer backward does not need second-order info through the inner step; `create_graph=True` is needed only for MAML-style meta-learning where the outer loss differentiates through the inner optimization itself; unlike `.backward()`, `autograd.grad` does not zero existing `.grad` accumulation between calls — if you mix both styles you will double-count gradients silently
- [ ] `torch.func.functional_call(module, params_dict, args)`: executes a stateless forward pass using `params_dict` instead of the module's own `.weight` / `.bias`; the original module is never mutated; enables per-token or per-sequence weight variants via `torch.func.vmap` over a batched dict; used in TTT to apply the dynamically-updated inner model weights without writing back to the `nn.Module` in-place

**Coding tasks:**

- [ ] **`autograd.grad` identity-grad drill:** Create `x = torch.randn(4, requires_grad=True)`, `loss = (x ** 2).sum()`. Compute `g1 = torch.autograd.grad(loss, x)[0]`. Verify `torch.allclose(g1, 2 * x)`. Then call `loss.backward()` and verify `x.grad == g1`. *Expected:* both return `2x`; `autograd.grad` does not touch `x.grad` at all. **The no-accumulation guarantee matters in TTT: calling `autograd.grad` in a decode loop never corrupts `.grad` buffers you may be using elsewhere.**
- [ ] **Inner TTT gradient drill:** Implement one step of the TTT inner update: create `inner = nn.Linear(16, 16, bias=False)`, construct a self-supervised reconstruction loss on a single token vector, then compute the weight gradient with `torch.autograd.grad(loss, inner.weight, create_graph=False)[0]` and apply the update manually: `new_W = inner.weight - lr * grad`. Use `torch.func.functional_call(inner, {"weight": new_W}, x)` to run the updated forward without touching `inner.weight`. *Expected:* the updated output differs from the original output; the graph is not retained (no memory leak across loop iterations).
- [ ] **Stateless forward with vmap drill:** Write a function `batched_forward(module, weight_batch, x)` that calls `torch.func.vmap(lambda w: functional_call(module, {"weight": w}, x))(weight_batch)`. Run with `weight_batch` of shape `[B, d, d]` and `x` of shape `[d]`. *Expected:* output shape is `[B, d]`; the module's own `.weight` is never modified; each element of the batch uses a different weight matrix.
- [ ] Implement TTT-Linear: replace a transformer attention layer with a TTT-Linear layer; implement the inner gradient update using `torch.autograd.grad` with `create_graph=False`; train on Shakespeare
- [ ] Compare perplexity per parameter for: transformer, Mamba, TTT-Linear at the same parameter count
- [ ] Measure training speed of TTT vs. transformer: the inner gradient computation adds overhead; measure the ratio of step times

> [!NOTE] Milestone
> TTT-Linear should match or slightly beat transformer perplexity at the same parameter count on short-to-medium contexts (up to ~512 tokens), because the dynamic hidden state provides more effective context compression than the fixed-size state of a standard RNN while using fewer parameters than full attention. On longer contexts, TTT should scale better than transformer (linear in sequence length) but may not match Mamba's efficiency. Training step time for TTT is typically 1.5–2.5× slower than transformer due to the inner gradient computation — this is a real cost and should be included when comparing parameter efficiency on a compute budget.

---

## Week 28 — Depth Recurrence and Shared-Weight Transformers

**Primary resource:** Dehghani et al., [Universal Transformers](https://arxiv.org/abs/1807.03819) — §1–3

**Concepts to understand:**

- [ ] Standard transformer: `L` layers each with their own weights; total parameters = `L × (parameters per layer)`; depth is "free" in parameter count once the first layer's weights are defined... except that each layer has independent weights
- [ ] Shared-weight transformer (Universal Transformer): use the same weight matrix for all `L` layers; run the same transformer block `L` times; total parameters = `1 × (parameters per layer)` regardless of depth; effective parameter count is reduced by `L`
- [ ] Why it works: each application of the same block refines the representation; the network learns a block that, when applied repeatedly, converges to a good representation; this is analogous to iterative algorithms (gradient descent, message passing) converging to a fixed point
- [ ] Depth recurrence with residuals: `h_t = h_{t-1} + f_θ(h_{t-1})` where `f_θ` is the shared block; the residual ensures stability (the output can always equal the input by setting `f_θ = 0`); this is the design used in top Parameter Golf entries combining depth recurrence with TTT
- [ ] Mini-depth recurrence: a hybrid — share weights within groups of layers; e.g., share the first 4 layers' weights, share the next 4 layers' weights separately, etc.; reduces parameters by a factor of the group size while allowing different representations at different depths

**Coding tasks:**

- [ ] Implement a shared-weight transformer: modify nanoGPT to run one transformer block `n_layer` times (all using the same `nn.Module` instance); train and compare perplexity per parameter to the unshared baseline
- [ ] Implement mini-depth recurrence: share weights in groups of 2 (e.g., layers 0+1 share, layers 2+3 share, etc.); find the sharing factor that minimizes perplexity per parameter
- [ ] Combine TTT layers with depth recurrence: implement a model with one shared TTT-Linear block run 8 times; the dynamic state evolves across the 8 applications while the static weights are shared

> [!NOTE] Milestone
> Expected result: a fully shared-weight model with `L=8` applications of a single block achieves significantly worse perplexity than an 8-layer model with independent weights at the same parameter count — but only if the shared block is the same size. The fair comparison is: shared-weight model with 1 large block run 8 times vs. 8 small blocks with independent weights at the same total parameter count. In this comparison, the shared-weight model often wins at shorter contexts because the large single block has more capacity per-pass. The parameter Golf insight: shared weights allow you to get "depth for free" — a model counted as having 1 block's parameters but achieving 8-block expressivity through iteration.

---

## Week 29 — The modded-nanoGPT Reference Implementation

**Primary resource:** KellerJordan, [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt) — read every file; this is the most engineering-dense reference implementation for Parameter Golf techniques

**Concepts to understand:**

- [ ] The modded-nanoGPT architecture: no biases, ReLU² activations (ReLU(x)² instead of GELU — faster, similar quality), learnable logit soft-cap (Gemma-style: `tanh(logits / cap) × cap` to bound logits and improve stability), no positional embeddings (relies on causal mask alone)
- [ ] ReLU² vs. GELU: `ReLU²(x) = max(0, x)²` is computationally simpler than GELU (`x × Φ(x)`) and matches its quality in practice; in a bandwidth-bound setting, simpler activation functions reduce memory operations and improve throughput
- [ ] No positional embeddings: for language modeling, removing positional embeddings forces the model to learn position information from the causal mask alone; reduces parameters by `seq_len × d_model` (e.g., 1024 × 768 = ~800K parameters for GPT-2); works well for long training runs where the model can internalize positional structure
- [ ] Muon + AdamW hybrid: apply Muon to all 2D weight matrices (projections); apply AdamW to 1D parameters (biases, embeddings, LayerNorm) and the embedding matrix; the justification: Muon's orthogonalization works best for weight matrices with full-rank gradient structure; embeddings have sparse gradients (only a subset of rows are updated per step) which violates Muon's assumptions
- [ ] The QK norm: apply LayerNorm to Q and K before computing attention scores; prevents attention logit explosion at long training runs; cheaper than attention soft-capping and more stable than without any normalization

**Coding tasks:**

- [ ] Clone modded-nanoGPT; run the default training script; record the baseline perplexity per parameter on the OpenWebText validation set
- [ ] Ablate each modification: remove (a) ReLU², (b) no-bias, (c) logit soft-cap, (d) QK norm one at a time; record perplexity change per modification
- [ ] Run modded-nanoGPT's Muon optimizer against standard AdamW on the same model; verify the Muon improvement is real (not noise) by running 3 seeds each

> [!NOTE] Milestone
> The modded-nanoGPT architecture choices compound: each individual modification provides a small improvement (typically 0.01–0.05 perplexity), but together they produce a model that is 5–10% more efficient per parameter than standard nanoGPT. The ablation study will show that Muon > QK norm > ReLU² > logit soft-cap in terms of individual contribution. If the Muon improvement is not reproducible (variance across seeds is larger than the signal), your learning rate for Muon is likely wrong — Muon requires a different LR than AdamW (typically `lr_muon ≈ 0.1 × lr_adamw` because Muon's updates are orthonormalized and thus larger in spectral norm).

---

## Week 30 — Parameter Golf Campaign

*No new primary resources — systematic experimentation using all techniques from the curriculum.*

**Pre-campaign analysis:**

- [ ] Read the [Parameter Golf rules](https://openai.com/index/parameter-golf/) carefully; understand how parameters are counted (FP32-equivalent bits / 32)
- [ ] Run the provided baseline; record: parameter count, loss (bits-per-byte), and the breakdown of parameters by component (embedding, attention, MLP, output head)
- [ ] Identify which component dominates the parameter count; this determines which technique gives the largest first-order improvement

**Experiment log (fill in during experimentation):**

| # | Technique | Params before | Params after | BPB before | BPB after | Δ params | Δ BPB | Notes |
|---|-----------|--------------|--------------|-----------|-----------|---------|-------|-------|
| 0 | Baseline | | | | | — | — | |
| 1 | | | | | | | | |
| 2 | | | | | | | | |
| 3 | | | | | | | | |
| 4 | | | | | | | | |
| 5 | | | | | | | | |

**Suggested experimentation sequence (one change at a time):**

1. **Weight tying**: tie `wte` and `lm_head`; free if vocabulary is large
2. **Vocab optimization**: train a custom BPE tokenizer at 2048–4096 tokens; measure the embedding parameter savings vs. the perplexity cost from shorter context compression
3. **Muon optimizer**: replace AdamW with Muon + AdamW hybrid; train a fresh model
4. **Architecture**: replace transformer blocks with shared-weight Mamba or TTT-Linear blocks
5. **GPTQ quantization**: apply 4-bit or 6-bit GPTQ to the trained model; measure bits-per-parameter after compression
6. **Brotli compression**: apply Brotli to the quantized weight byte stream; measure final effective parameter count

**Compatibility sanity checks (run before submitting):**

- [ ] **`torch.compile` × custom autograd check:** Wrap your full model in `torch.compile(model, fullgraph=False)`. Run one forward+backward pass. Then run `torch._dynamo.explain(model)(sample_input)` and count `graph_break_count`. *Expected:* `graph_break_count <= 2` (one break is acceptable at the TTT autograd boundary; more indicates an unintended `.item()` or data-dependent branch). If your `STERound` or TTT gradient function causes breaks, decorate the custom `Function` with `@torch._dynamo.allow_in_graph` to tell the compiler to treat it as a leaf operation. **Never use `fullgraph=True` with custom `autograd.Function` subclasses unless you have verified they are graph-break-free — the error message will not point at the right line.**
- [ ] **DDP × `autograd.grad` check:** If you run the TTT inner update inside a DDP-wrapped model, verify that `autograd.grad` on inner model weights (which are not DDP-synchronized parameters) does not trigger an unexpected all-reduce. The inner state weights must be `register_buffer` or plain tensors, not `nn.Parameter`, to stay outside DDP's gradient sync scope.

**Capstone requirement:**

After completing the experiment log, write a one-paragraph analysis of the task's structure: which component dominated the parameter count, which technique provided the best accuracy-per-parameter trade-off for *this specific task*, and whether the result matched your prediction before running the experiment. The paragraph should contain at least one specific number (e.g., "weight tying saved 23% of parameters at zero cost in bits-per-byte because the vocabulary size was 16K").

> [!NOTE] Milestone
> A strong Parameter Golf run will combine at least three techniques from this curriculum. The highest-scoring open-source entry (modded-nanoGPT + GPTQ) achieves its score by: (1) using modded-nanoGPT's architecture for maximum parameter efficiency during training, (2) applying aggressive GPTQ quantization (INT4 with group-size 32) for maximum compression during submission, (3) using Brotli to compress the quantized weight stream. Each step independently reduces the effective parameter count; the composition is multiplicative. A model that is 2× more efficient architecturally and 4× more efficient via quantization achieves 8× the parameter efficiency of the baseline.
