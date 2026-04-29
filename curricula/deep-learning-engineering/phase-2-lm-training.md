# Phase II — Modern LM Training Engineering

*Weeks 5–10 · ~30 hrs*

> **Goal:** Train language models efficiently on a single GPU. By the end, you will understand every knob that affects training speed and stability, know how to profile a training run and identify its bottleneck, and be able to implement the Muon optimizer — the most effective optimizer used in top Parameter Golf submissions.

**Weeks 5–6 primary:** PyTorch [AMP tutorial](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html) + [torch.utils.checkpoint docs](https://pytorch.org/docs/stable/checkpoint.html) + [torch.profiler tutorial](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

**Weeks 7–8 primary:** HuggingFace [datasets library](https://huggingface.co/docs/datasets) + [WebDataset](https://github.com/webdataset/webdataset)

**Weeks 9–10 primary:** Loshchilov & Hutter [AdamW paper](https://arxiv.org/abs/1711.05101) + KellerJordan [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt) (Muon optimizer reference implementation)

---

## Week 5 — Mixed Precision: BF16, FP16, and AMP

**Concepts to understand:**

- [ ] FP32 vs FP16 vs BF16: FP32 has 8 exponent bits + 23 mantissa bits; FP16 has 5 exponent + 10 mantissa (dynamic range limited — overflow/underflow common); BF16 has 8 exponent + 7 mantissa (same range as FP32, fewer mantissa bits — ideal for DL)
- [ ] Why BF16 is preferred over FP16 for training: BF16 cannot overflow because it has the same exponent range as FP32; FP16 overflows on activations/gradients above ~65504, requiring loss scaling; on A100+ hardware, BF16 matmuls are as fast as FP16
- [ ] Loss scaling (FP16 only): multiply the loss by a large constant (`scale=2^15`) before `.backward()`; divide gradients by the same constant before `optimizer.step()`; this prevents gradient underflow in FP16 storage. `GradScaler` automates this
- [ ] `torch.autocast`: a context manager that casts eligible operations to a lower precision; matmuls, convolutions, and attention are cast; reductions, normalization layers, and loss computations stay in FP32; does not affect the model's stored weights (still FP32 master copy)
- [ ] Master weights in FP32: even in BF16 training, optimizer states (momentum, variance in Adam) are stored in FP32; only the forward/backward computation uses BF16; this is essential for stability

**Coding tasks:**

- [ ] Add `torch.autocast(device_type='cuda', dtype=torch.bfloat16)` to your nanoGPT training loop; measure step time before and after; measure GPU memory before and after
- [ ] Verify correctness: run 1000 steps with and without AMP; val loss should differ by less than 0.02
- [ ] Add `GradScaler` for comparison; verify it is unnecessary with BF16 (no overflow occurs); confirm it is necessary with FP16 by observing gradient NaN without it

> [!NOTE] Milestone
> Expected speedup from BF16 AMP on an A100: 1.5–2.5× step time reduction. Memory reduction: approximately 40% (activations stored in BF16 rather than FP32; weights unchanged in master copy). If you see less than 1.2× speedup, check that you wrapped the entire forward pass (including attention) in the autocast context — if the matmuls are not being cast, the autocast is a no-op. Verify by inserting `print(q.dtype)` inside the attention forward to confirm Q/K/V tensors are BF16.

---

## Week 6 — Memory Management: Gradient Checkpointing and Profiling

**Concepts to understand:**

- [ ] The memory breakdown of a training step: activations (largest for long sequences), gradients (same size as parameters), optimizer states (2× parameters for Adam), parameters (1×); for a 100M-parameter model with seq_len=1024 and batch_size=8, activations alone require ~4–8 GB
- [ ] Gradient checkpointing (`torch.utils.checkpoint.checkpoint`): instead of storing all intermediate activations for the backward pass, recompute them on-the-fly during the backward; trades memory for compute (typically 30–40% more compute, but 8–10× memory reduction for very deep models)
- [ ] How to apply checkpointing: wrap individual transformer blocks: `output = checkpoint(block, x)` instead of `output = block(x)`; requires the wrapped function to be re-entrant (no global state mutations)
- [ ] `torch.profiler`: wraps a training loop and records kernel-level timing; output includes a timeline of CPU and GPU operations, a table of the top-N slowest kernels, and memory usage over time
- [ ] Compute-bound vs. memory-bandwidth-bound: a matmul is compute-bound (limited by FLOP throughput); an elementwise operation (e.g., GELU, softmax) is memory-bandwidth-bound (limited by how fast data can be read/written); FlashAttention (Phase IV) is the standard solution for the attention operation's memory bottleneck

**Coding tasks:**

- [ ] Add gradient checkpointing to every transformer block in your nanoGPT; measure memory before/after; verify the loss curve is unchanged
- [ ] Profile 10 training steps with `torch.profiler`; export the trace and open it in Chrome's `chrome://tracing`; identify the top 3 GPU kernels by time
- [ ] Find the operator that dominates GPU time: is it the attention computation, the MLP matmuls, or the embedding lookup?

> [!NOTE] Milestone
> In a standard nanoGPT training step at `d_model=768, n_layers=12, seq_len=1024, batch_size=8`: the two MLP linear layers (each `d_model × 4d_model`) should dominate GPU time — they account for roughly 60–70% of the compute. The attention computation is often 20–30%. The embedding lookup is negligible in time (it is a memory operation, very fast). If attention dominates at long sequence lengths (>2048), this is the motivation for FlashAttention in Phase IV.

---

## Week 7 — Data Pipeline Engineering

**Concepts to understand:**

- [ ] The data loading bottleneck: if `step_time` with `num_workers=0` ≫ `step_time` with `num_workers=4`, the CPU is the bottleneck; the GPU is idle waiting for data
- [ ] `DataLoader` tuning: `num_workers` (parallel data loading processes), `pin_memory=True` (pages CPU memory to be non-swappable, speeds host-to-device transfer), `prefetch_factor` (number of batches to prefetch per worker)
- [ ] Memory-mapped datasets: `np.memmap` for pre-tokenized binary files; `mmap` mode does not load the file into RAM — the OS pages in only the accessed bytes; essential for datasets larger than RAM
- [ ] WebDataset: a streaming dataset format based on `.tar` files; enables training on data stored in object storage (S3, GCS) without downloading locally; reads are sequential (no random access), which is often faster than random reads from disk
- [ ] Dataset deduplication: training on duplicate text inflates effective epoch count and hurts generalization; MinHash + LSH is the standard near-dedup algorithm for large text corpora (used in RefinedWeb, Dolma, FineWeb)
- [ ] Data mixing: training on a weighted mixture of sources (web text, code, math, books); the mixing ratios are a hyperparameter with large effects on downstream capability — Llama-3 spent significant effort on this

**Coding tasks:**

- [ ] Profile your data loader: time `get_batch()` in isolation vs. inside the training loop; verify the GPU is not waiting (data loading time < forward+backward time)
- [ ] Implement a streaming WebDataset loader for a text corpus; verify you can train on data without loading it entirely into memory
- [ ] Implement a multi-source data mixer: combine two token datasets with configurable weights; verify the sampling ratio matches the specified weights over a 10,000-step run

> [!NOTE] Milestone
> A well-configured data pipeline on a modern GPU should have near-zero data loading time relative to GPU compute time — the prefetched batch should always be ready before the GPU finishes the previous step. The test: log `data_time_ms` and `gpu_time_ms` separately in your training loop. If `data_time_ms > 0.1 × gpu_time_ms`, your pipeline is the bottleneck. The usual fix: increase `num_workers`, add `pin_memory=True`, or switch to pre-tokenized memory-mapped files.

---

## Week 8 — Debugging at Scale

**Concepts to understand:**

- [ ] Loss NaN tracking: NaN propagates through all subsequent operations; detect with `torch.isnan(loss).any()` and `torch.isnan(grads).any()` before calling `optimizer.step()`; the cause is almost always a gradient explosion — detect with `grad_norm = nn.utils.clip_grad_norm_(model.parameters(), float('inf'))` printed before clipping
- [ ] The gradient norm as a leading indicator: `grad_norm` spikes 1–2 steps before a loss spike; monitoring it in wandb is the earliest warning of training instability
- [ ] Loss spike recovery: when a spike occurs, roll back to the last checkpoint; reduce LR by 2× and resume — a spike indicates the LR is near the edge of stability for the current loss landscape
- [ ] Silent data corruption: a training loop that runs without error but produces degraded models; check for: off-by-one in the data loader (`X` and `Y` not shifted by exactly 1), incorrect attention masking (non-causal masking leaks future information), label smoothing applied to padding tokens
- [ ] The causal mask verification test: generate from a model trained with a suspected masking bug; the output should be incoherent if future tokens were visible during training (the model learns to "look ahead" and produces text with unusually high confidence on early tokens)
- [ ] Checkpoint diff: compare `model.state_dict()` between two checkpoints 100 steps apart; parameters that have not changed at all indicate dead neurons or a layer that is not receiving gradient signal

**Coding tasks:**

- [ ] Add NaN detection to your training loop; deliberately introduce a NaN by setting one weight to `inf`; verify detection fires before `optimizer.step()`
- [ ] Deliberately introduce an off-by-one error in the data loader (`Y = X` instead of `Y = X[:, 1:]`); observe how the loss curve changes; diagnose from the curve alone before reading the code
- [ ] Implement a "gradient flow check": after a backward pass, print the mean absolute gradient for each layer; verify no layer has zero gradient

> [!NOTE] Milestone
> The off-by-one target bug (`Y = X` instead of `Y = X[:, 1:]`) produces a model that is predicting the current token from itself — a trivially easy task. Loss will drop very quickly to near zero on training data and stay near zero on validation data. This is the only case where near-zero training AND validation loss is suspicious — it signals the model is memorizing an identity mapping, not learning language statistics. Any other near-zero training / near-zero validation pattern is just successful learning.

---

## Week 9 — Modern Optimizers: AdamW and Muon

**Primary resources:**
- Loshchilov & Hutter, [AdamW paper](https://arxiv.org/abs/1711.05101) (~30 min read — focus on §3)
- KellerJordan, [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt) — read `train_gpt2.py` end to end; the Muon optimizer implementation is ~50 lines

**Concepts to understand:**

- [ ] AdamW vs. Adam: standard Adam applies weight decay as `L2 loss += λ||θ||²`, which is equivalent to `θ ← θ - η(g + λθ)` — but Adam normalizes gradients by the second moment estimate, so the effective weight decay is `λ / √(v̂ + ε)`, not `λ`; AdamW applies weight decay directly to weights *after* the adaptive step: `θ ← (1 - ηλ)θ - η·m̂/√(v̂+ε)`, decoupling it from the adaptive scaling
- [ ] Why AdamW generalizes better: the decoupled weight decay is more predictable and acts as a proper L2 regularizer independent of the gradient magnitude; models trained with AdamW have lower weight norms for the same WD coefficient vs. Adam
- [ ] Muon optimizer (Modular Dual): applies Newton-Schulz iteration to approximate the matrix square root of the gradient second moment; effectively steepest descent in spectral norm rather than L2 norm; works best for weight matrices (not embeddings or biases)
- [ ] Newton-Schulz iteration: the update `X ← (3X - X³) / 2` converges to the matrix sign of the input in ~5 iterations; Muon uses this to orthogonalize the gradient direction; result: gradient updates are nearly orthogonal matrices, which have good conditioning properties for deep networks
- [ ] Muon in practice: apply Muon to all `weight` tensors in linear/attention layers; apply AdamW to embeddings, biases, and LayerNorm parameters; this hybrid is the configuration used in modded-nanoGPT

**Coding tasks:**

- [ ] Implement AdamW from scratch (without `torch.optim.AdamW`); verify it matches the PyTorch implementation on a small model by comparing parameter values after 100 steps
- [ ] Add the Muon optimizer to your nanoGPT training loop following the modded-nanoGPT reference; train on Shakespeare; compare final val loss and convergence speed vs. AdamW
- [ ] Profile the Muon optimizer step vs. AdamW: measure the additional time from the Newton-Schulz iterations; typical overhead is 5–15% of step time

> [!NOTE] Milestone
> Expected results: Muon should reach the same final validation loss as AdamW in fewer steps (~20–30% fewer on small language modeling tasks). The reason: Muon's orthogonalized updates are better conditioned than Adam's, especially in the early training phase where the gradient covariance structure is poorly estimated. If you see no improvement, check that you are applying Muon only to weight matrices and not to embeddings — Muon performs poorly on embedding tables because the rows are not fully connected to all outputs (sparse gradient structure).

---

## Week 10 — Learning Rate Schedules and Stability

**Concepts to understand:**

- [ ] Warmup: train at near-zero LR for the first `T_warm` steps (typically 1–5% of total steps), then ramp to the target LR; prevents early instability because at step 0, Adam's second moment estimate `v̂` is near zero, making `m̂/√(v̂+ε)` arbitrarily large without warmup
- [ ] Cosine annealing with warmup: the standard for LLM pretraining; LR rises linearly for `T_warm` steps, then follows `η(t) = η_min + (η_max - η_min)/2 × (1 + cos(π(t - T_warm)/(T_max - T_warm)))`
- [ ] Warmdown (cooldown): reduce LR to near-zero in the final 10–20% of training; empirically improves final loss by allowing the model to "settle" into a flat region of the loss landscape; used in modded-nanoGPT
- [ ] LR as a function of model size: the optimal LR scales roughly as `η ∝ 1/√d_model` (empirical); a 256-dim model trains well at lr=3e-3 while a 1024-dim model needs lr≈1e-3; this is one reason hyperparameter transfer across scales is non-trivial
- [ ] Gradient clipping threshold: clip to `max_norm=1.0` for most LM training; set it to `float('inf')` during the first experiment to measure the natural gradient norm — then set the clip threshold to 2× the typical gradient norm

**Coding tasks:**

- [ ] Implement the full warmup + cosine + warmdown schedule in a `get_lr(step)` function; plot the schedule for `T_warm=100, T_max=5000, T_down=500`
- [ ] Run an ablation: compare (a) constant LR, (b) cosine without warmup, (c) cosine + warmup, (d) cosine + warmup + warmdown; plot all four val loss curves on the same axes

> [!NOTE] Milestone
> Expected ranking of schedules by final val loss (best to worst): (d) > (c) > (b) > (a). The warmdown improvement is often 0.02–0.05 in val loss for a small LM trained for 5000 steps — surprisingly large for a modification to only the final 10% of training. The intuition: cosine decay without warmdown leaves the model in a moderate-LR regime at the end of training, where it is still making relatively large steps that prevent full convergence. Warmdown drives the LR to near-zero, forcing convergence to a lower-loss solution within the current loss basin.
