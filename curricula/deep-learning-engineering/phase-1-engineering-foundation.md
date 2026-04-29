# Phase I — Engineering Foundation

*Weeks 1–4 · ~20 hrs*

> **Goal:** Bridge the gap between knowing the theory and being able to run it. By the end, you will have a working LM training pipeline with proper checkpointing, a tokenizer you understand from the inside out, and experiment tracking that makes every subsequent phase's work reproducible. Theory is assumed — this phase is about building engineering muscle memory.

**Week 1 primary:** Karpathy, [nanoGPT video](https://www.youtube.com/watch?v=kCc8FmEb1nY) (1h56m) + full `model.py` / `train.py` read

**Weeks 2–3 primary:** Sennrich et al., [BPE paper](https://arxiv.org/abs/1508.07909) + OpenAI tiktoken source + Karpathy, [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) (2h13m)

**Week 4 primary:** wandb quickstart + Hugging Face `accelerate` config basics + learning rate finder implementation

---

## Week 1 — nanoGPT: Your Production Baseline

**Primary resource:** [Karpathy — nanoGPT video](https://www.youtube.com/watch?v=kCc8FmEb1nY) (1h56m) + [nanoGPT source](https://github.com/karpathy/nanoGPT)

Watch the video, then read `model.py` and `train.py` end to end. Every line. Annotate anything you would not have written yourself. The goal is not to understand the architecture (you already do) but to understand the *engineering decisions*: why is the training loop structured this way, where does checkpointing happen, how is the eval loop different from the training loop.

**Concepts to understand:**

- [ ] `torch.compile` and why it matters: fuses elementwise ops, reduces kernel launch overhead; wrap the model in `torch.compile(model)` and observe wallclock speedup
- [ ] The `model.eval()` / `model.train()` distinction: affects Dropout and BatchNorm; always call `model.eval()` before any generation or validation loop
- [ ] Gradient accumulation: when your batch size exceeds GPU memory, accumulate gradients over `k` forward/backward passes before calling `optimizer.step()`; effective batch size = `batch_size × grad_accum_steps`
- [ ] Checkpointing: save `model.state_dict()`, `optimizer.state_dict()`, `iter_num`, and the random seed together — resuming without the optimizer state causes a bad LR restart
- [ ] `ctx = torch.autocast(device_type, dtype=torch.bfloat16)`: context manager that casts eligible ops to BF16; understand which ops are and are not cast (matmuls yes, layer norms no)
- [ ] `scaler = torch.cuda.amp.GradScaler()`: used with FP16 (not BF16) to prevent gradient underflow; `scaler.scale(loss).backward()`, `scaler.step(optimizer)`, `scaler.update()`
- [ ] `register_buffer` vs `register_parameter` vs raw attribute: buffers (e.g., the causal mask) move with the model across devices and appear in `state_dict()` by default but receive no gradients; raw Python attributes (`self.mask = ...`) do not move on `.cuda()` and do not appear in `state_dict()`, causing silent device-mismatch bugs in Phase V (FSDP) and Phase IV (KV cache); `persistent=False` buffers move across devices but are excluded from `state_dict()`
- [ ] `torch.compile` graph breaks: `torch.compile` only fuses operations it can trace statically; calling `.item()`, branching on a tensor value, or using `print` inside `forward` creates a graph break that silently eliminates most of the speedup; diagnose with `TORCH_LOGS=graph_breaks` or `torch._dynamo.explain(model)(x)`

**Coding tasks:**

- [ ] Clone nanoGPT; annotate `train.py` line by line; write a comment above each block explaining what engineering problem it solves (not what it does)
- [ ] Add a gradient accumulation loop: modify `train.py` to support `gradient_accumulation_steps=4` and verify loss is identical to running with `batch_size × 4`
- [ ] Implement a save/load checkpoint cycle: save at step 100, deliberately corrupt a weight, reload from checkpoint, verify weights are restored
- [ ] **`register_buffer` drill:** Build an `nn.Module` with a causal mask stored three ways: as a raw attribute `self.m1`, as `self.register_buffer('m2', ...)`, and as `self.register_buffer('m3', ..., persistent=False)`. Call `model.cuda()`. Print `m.device` for each and `list(model.state_dict().keys())`. *Expected:* m1 stays on CPU and is absent from `state_dict`; m2 moves to CUDA and appears in `state_dict`; m3 moves to CUDA but is absent from `state_dict`.
- [ ] **Graph break drill:** Wrap nanoGPT with `torch.compile(model, fullgraph=True)`. Add `if loss.item() > 10: pass` inside the forward. Run one step. *Expected:* `torch._dynamo.exc.Unsupported` error. Switch to `fullgraph=False`, run `torch._dynamo.explain(model)(x)`, and confirm `graph_break_count >= 1`. Then remove the break and re-run with `fullgraph=True` to confirm it compiles cleanly.

> [!NOTE] Milestone
> After adding gradient accumulation: run with `batch_size=4, grad_accum=4` and with `batch_size=16, grad_accum=1`. Loss curves should be nearly identical (they are computing the same gradient in expectation). If they diverge significantly, check that you are dividing the loss by `grad_accum_steps` before each backward call — otherwise you are scaling the gradient by `grad_accum_steps` and the effective learning rate is too high.

---

## Week 2 — Tokenization Engineering

**Primary resource:** Karpathy, [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) (2h13m) + [minbpe source](https://github.com/karpathy/minbpe)

Tokenization is invisible in most tutorials but is one of the largest levers in Parameter Golf: a 4096-token vocabulary uses 4× fewer embedding parameters than a 16384-token vocabulary for the same `d_model`. The top Parameter Golf entries explicitly tune vocabulary size.

**Concepts to understand:**

- [ ] Byte-level BPE: start with 256 byte tokens; iteratively merge the most frequent adjacent pair; the merge table is the "tokenizer" — `encode` applies merges greedily, `decode` reverses them
- [ ] Vocabulary size vs. sequence length trade-off: larger vocab → shorter sequences (fewer tokens per character) → less context needed, but larger embedding table; smaller vocab → longer sequences → more context required, but much smaller embedding table
- [ ] Vocabulary size and parameter budget: `vocab_size × d_model` parameters in the embedding table; for `vocab=50257, d_model=768` (GPT-2 small), this is 38.6M of 117M total — 33% of the model
- [ ] Weight tying: `lm_head.weight = wte.weight` — the output projection is the transposed embedding matrix; saves `vocab × d_model` parameters with no loss in expressivity (confirmed empirically in most LM settings)
- [ ] `tiktoken` vs. `sentencepiece` vs. custom BPE: `tiktoken` (OpenAI's) is fastest for inference; `sentencepiece` is standard for T5/LLaMA-style models; custom BPE is used in Parameter Golf to minimize vocab for the specific evaluation corpus

**Coding tasks:**

- [ ] Implement BPE from scratch following minbpe: `train(text, vocab_size)`, `encode(text)`, `decode(ids)`; verify round-trip fidelity
- [ ] Train BPE tokenizers at vocab sizes 256, 512, 1024, 4096, 16384 on a small text corpus; plot the average tokens-per-character for each; record the embedding parameter cost for `d_model=256` at each vocab size
- [ ] Replace nanoGPT's default tiktoken tokenizer with your 1024-vocab BPE; train on Shakespeare; compare perplexity and total parameter count

> [!NOTE] Milestone
> Expected observations: at `vocab=256` (pure byte-level), average ~1.0 tokens/character; at `vocab=4096`, ~0.35 tokens/character; at `vocab=50257`, ~0.25 tokens/character. Smaller vocab → longer sequences → the model needs more context capacity to achieve the same perplexity. The Parameter Golf trade-off is exact: reducing vocab from 50257 to 4096 saves `(50257 - 4096) × d_model` embedding parameters, but requires `~1.4×` more context length to achieve the same loss. Whether it is worth it depends on how context length affects your architecture's parameter count.

---

## Week 3 — The Full Training Stack

**Primary resource:** [Karpathy — llm.c](https://github.com/karpathy/llm.c) — read the Python training script and the C implementation's main loop; don't worry about the CUDA kernels yet

This week is about understanding every engineering component of a complete training run: data loading, tokenization, batching, forward/backward, optimizer step, logging, and evaluation. The goal is to be able to write the entire stack from memory.

**Concepts to understand:**

- [ ] Pre-tokenizing and caching: pre-tokenize the entire dataset to a `.bin` file of token IDs; this avoids re-tokenizing every epoch and is essential for large datasets. Memory-mapped files (`np.memmap`) let you read a 100GB token array without loading it into RAM
- [ ] The data loader: a random-offset iterator over the memory-mapped token array; `get_batch(split)` returns `(X, Y)` where `Y = X[:, 1:]` shifted right — the autoregressive target
- [ ] Evaluation loop: `model.eval()`, `torch.no_grad()`, run `eval_iters` batches, average the loss; `model.train()` immediately after
- [ ] Logging: log `train_loss`, `val_loss`, `lr`, `grad_norm`, `step_time_ms`, and `mfu` (model FLOPs utilization) every N steps; these six numbers diagnose 90% of training problems
- [ ] MFU (Model FLOPs Utilization): `mfu = actual_flops_per_second / theoretical_peak_flops`; a well-tuned run on an A100 achieves 40–60% MFU; below 20% signals a serious bottleneck

**Coding tasks:**

- [ ] Pre-tokenize OpenWebText (or a subset) to a `.bin` file; implement a memory-mapped `get_batch` that reads random offsets
- [ ] Add MFU logging to your nanoGPT training loop: compute `flops_per_token = 6N` (where `N` is total parameters), multiply by `tokens_per_step / step_time_s`, divide by GPU peak FLOPS
- [ ] Run a 1000-step training run; verify MFU is above 20%; if not, identify the bottleneck by timing the data loader vs. forward pass vs. backward pass separately

> [!NOTE] Milestone
> A training step at `batch_size=32, seq_len=256, d_model=256, n_layers=4` on an A100 should take approximately 15–30ms. If your step time is 200ms+, the bottleneck is almost certainly the data loader (CPU-bound tokenization or disk I/O) rather than the GPU. The fix: pre-tokenize to a binary file and use `np.memmap`. After the fix, the GPU should be the bottleneck, and you should see MFU increase from <5% to >30%.

---

## Week 4 — Experiment Infrastructure

**Primary resources:** [wandb quickstart](https://docs.wandb.ai/quickstart) + [hydra config tutorial](https://hydra.cc/docs/intro/) (or use simple `argparse` + YAML)

Every experiment from here on should be fully reproducible. This week is about building the scaffolding that makes that possible.

**Concepts to understand:**

- [ ] `wandb.init(config=config)`: log hyperparameters at run start; use `wandb.log({"train_loss": loss, "lr": lr, ...})` every step; group runs by experiment name
- [ ] Reproducibility: set `torch.manual_seed`, `np.random.seed`, `random.seed`, and `torch.backends.cudnn.deterministic = True` at the start of every training run; log the seed to wandb
- [ ] Config management: every hyperparameter lives in a config dict/dataclass — never hardcode values in training loops; a run is defined by its config + seed
- [ ] The learning rate finder: sweep LR from 1e-7 to 1e-1 over 100 steps, log `(lr, loss)` to wandb; the optimal LR is at the steepest descent before the loss diverges
- [ ] Experiment naming: `{model_size}_{optimizer}_{vocab_size}_{date}` — a run name that encodes the key decisions; makes wandb dashboards navigable

**Coding tasks:**

- [ ] Instrument your nanoGPT training loop with wandb: log loss, LR, grad norm, MFU, and step time per step
- [ ] Implement a learning rate finder sweep; plot the LR-vs-loss curve in wandb; identify the recommended LR
- [ ] Run the same training config with 5 different random seeds; use wandb to plot the mean ± std of val loss across seeds; this is your first reproducibility check

> [!NOTE] Milestone
> After adding wandb logging: open the run page during training. You should see `val_loss` decreasing, `grad_norm` fluctuating around a roughly constant level (spikes indicate LR is near the edge of stability), and `mfu` stable after the first few steps. If `mfu` is high in the first 10 steps and then drops, you have a data loader issue that is hidden by the GPU warmup. The five-seed reproducibility check should show very small variance between runs — if val loss varies by more than 0.1 between seeds, your initialization or data ordering is non-deterministic despite your seed settings.

---

## Phase I Consolidation

**Engineering checklist — you should be able to do all of these from memory:**

- [ ] Write a complete LM training loop: data loading, forward, loss, backward, optimizer step, gradient clipping, LR scheduling
- [ ] Add gradient accumulation to any training loop without changing the effective learning rate
- [ ] Save and restore a complete training checkpoint (model + optimizer + step + seed)
- [ ] Pre-tokenize a text dataset to a memory-mapped `.bin` file and write a random-offset batch sampler
- [ ] Train a BPE tokenizer at a custom vocabulary size from scratch
- [ ] Log MFU, grad norm, train loss, and val loss to wandb; interpret the resulting curves
- [ ] Run a learning rate finder sweep and identify the recommended LR from the curve

> [!NOTE] Milestone
> You should now be able to reproduce nanoGPT's Shakespeare training run from memory — not by reading the source, but by writing a training loop that produces the same loss curve. If you can do this, Phase I is complete. The test: open a blank editor, write the full training loop without reference, run it, and get val loss below 1.6 on the character-level Shakespeare dataset in under 10 minutes of training.
