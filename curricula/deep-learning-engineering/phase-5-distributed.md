# Phase V — Distributed Training

*Weeks 21–25 · ~25 hrs*

> **Goal:** Train models across multiple GPUs. By the end, you will have implemented DDP from first principles, understand ZeRO's three stages of optimizer/gradient/parameter sharding, know when to use tensor parallelism vs. pipeline parallelism, and be able to design a parallelism strategy for a given model and hardware configuration.

**Week 21 primary:** PyTorch [DDP tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) + implement allreduce from scratch

**Week 22 primary:** Rajbhandari et al., [ZeRO paper](https://arxiv.org/abs/1910.02054) §1–3 + DeepSpeed ZeRO stage 1/2/3

**Week 23 primary:** Shoeybi et al., [Megatron-LM paper](https://arxiv.org/abs/1909.08053) §3 (tensor parallelism)

**Weeks 24–25 primary:** [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM) pipeline parallelism docs + practical multi-GPU recipe

---

## Week 21 — DDP: Data Parallelism from First Principles

**Concepts to understand:**

- [ ] Data parallelism: replicate the model on `N` GPUs; each GPU processes a different mini-batch; average gradients across GPUs before the optimizer step; effective batch size = `N × local_batch_size`
- [ ] Allreduce: a collective operation that sums (or averages) a tensor across all `N` processes and distributes the result to all processes; the bandwidth-optimal allreduce algorithm is ring-allreduce: each GPU sends `1/N` of its gradients to the next GPU in a ring, N-1 times; total communication = `2(N-1)/N × gradient_size ≈ 2 × gradient_size` for large N
- [ ] Gradient synchronization: after `loss.backward()`, each GPU has local gradients; call `allreduce(grad)` for each parameter; then `optimizer.step()` using the averaged gradient; all replicas stay in sync because they start from the same weights and use the same averaged gradient
- [ ] `torch.nn.parallel.DistributedDataParallel` (DDP): wraps a model and handles gradient synchronization automatically; `ddp_model = DDP(model, device_ids=[rank])`; gradients are averaged across all `world_size` processes in the backward pass
- [ ] Gradient bucketing: DDP groups small parameters into buckets and overlaps gradient communication with computation; parameters in the last bucket (computed first during backward) are communicated while the rest of the backward pass continues; this overlap is why DDP adds minimal overhead compared to single-GPU training
- [ ] Synchronizing batch statistics: if using BatchNorm, its running statistics must be synchronized across GPUs — use `nn.SyncBatchNorm`; LayerNorm has no inter-sample statistics and requires no synchronization

**Coding tasks:**

- [ ] Implement a bare-bones DDP training loop using `torch.distributed.init_process_group` and manual `allreduce` calls (before DDP module); launch with `torchrun --nproc_per_node=2`
- [ ] Verify correctness: run the same training configuration on 1 GPU (batch=16) and on 2 GPUs (batch=8 per GPU); confirm that loss curves are identical (same effective batch size, same gradients after averaging)
- [ ] Profile the DDP overhead: time the backward pass with and without gradient synchronization; measure the fraction of step time spent in communication vs. computation

> [!NOTE] Milestone
> DDP overhead at 2 GPUs connected via PCIe: communication adds approximately 20–40% to step time for a small model (communication dominates because model gradient size is large relative to compute time). At 2 GPUs connected via NVLink: communication overhead drops to 5–10%. The key metric: "communication efficiency" = `compute_time / (compute_time + communication_time)`; high efficiency means computation and communication can be overlapped effectively. If your 2-GPU run is *slower* than 1-GPU, the model is too small — at small scales, communication overhead exceeds the parallelism benefit; DDP pays off when compute time >> communication time.

---

## Week 22 — ZeRO: Sharding Optimizer States, Gradients, and Parameters

**Primary resource:** Rajbhandari et al., [ZeRO paper](https://arxiv.org/abs/1910.02054) — §1–4; focus on Table 1 (memory reduction per stage)

**Concepts to understand:**

- [ ] Memory breakdown for training: for a model with `Ψ` parameters, mixed precision training requires: `2Ψ` bytes (FP16 parameters) + `2Ψ` bytes (FP16 gradients) + `4Ψ` bytes (FP32 master copy) + `8Ψ` bytes (Adam optimizer states: first + second moment, both FP32) = `16Ψ` bytes total; a 1B-parameter model needs ~16GB for the model state alone
- [ ] ZeRO-Stage 1 (optimizer state sharding): partition the optimizer states across `N` GPUs; each GPU stores only `1/N` of the FP32 master copy and Adam moments; memory for optimizer states reduced from `12Ψ` to `12Ψ/N`; no communication overhead beyond standard DDP gradient allreduce
- [ ] ZeRO-Stage 2 (+ gradient sharding): additionally shard the gradients; each GPU only stores the gradients for the parameters it owns; memory for gradients reduced from `2Ψ` to `2Ψ/N`; requires a `reduce-scatter` collective instead of `allreduce`
- [ ] ZeRO-Stage 3 (+ parameter sharding, a.k.a. FSDP): additionally shard the parameters; each GPU stores only `1/N` of the FP16 parameters; before each layer's forward pass, gather the full parameters via `allgather`; after the backward pass, scatter the gradients; total memory per GPU: `16Ψ/N`; requires additional all-gather communication during forward
- [ ] `torch.distributed.fsdp.FullyShardedDataParallel` (FSDP): PyTorch's implementation of ZeRO-3; wraps a model and handles parameter gathering, gradient scattering, and optimizer state sharding automatically

**Coding tasks:**

- [ ] Apply DeepSpeed ZeRO Stage 2 to your nanoGPT; measure GPU memory before and after at `N=2 GPUs`; verify the `2×` memory reduction on optimizer states + gradients
- [ ] Apply PyTorch FSDP (ZeRO-3) to nanoGPT; measure peak memory at `N=2` and verify near-`2×` reduction in total model state memory; confirm training loss is unchanged
- [ ] Measure throughput (tokens/sec) for: (a) DDP only, (b) ZeRO-2, (c) FSDP/ZeRO-3 at 2 GPUs; note the throughput cost of parameter gathering in FSDP

> [!NOTE] Milestone
> For a 100M-parameter model at 2 GPUs: DDP uses ~1.6GB per GPU for model state; ZeRO-2 uses ~0.9GB (nearly half, because gradients and optimizer states are sharded); FSDP uses ~0.8GB (additionally parameters are sharded). The throughput penalty for FSDP vs. DDP is typically 5–15% due to all-gather communication overhead during the forward pass. FSDP pays off when model state memory is the binding constraint — for a 10B-parameter model, DDP requires ~160GB per GPU (infeasible), while FSDP with 8 GPUs requires ~20GB per GPU (feasible on A100s).

---

## Week 23 — Tensor Parallelism

**Primary resource:** Shoeybi et al., [Megatron-LM paper](https://arxiv.org/abs/1909.08053) — §3 (tensor model parallelism)

**Concepts to understand:**

- [ ] Why tensor parallelism: DDP requires each GPU to hold the full model; for models with layers too large to fit on a single GPU (e.g., `d_model=4096` attention with 12B parameters), tensor parallelism is required
- [ ] Column-parallel linear: partition a weight matrix `W ∈ ℝ^{m×n}` column-wise across `N` GPUs; GPU `i` holds columns `[i×n/N, (i+1)×n/N]`; compute `Y_i = X W_i` locally; all-gather `Y_i` to get full `Y`; splits the computation but requires one all-gather communication
- [ ] Row-parallel linear: partition `W` row-wise; GPU `i` holds rows `[i×m/N, (i+1)×m/N]` and a slice of the input `X_i`; compute `Y_i = X_i W_i` locally; allreduce `Y_i` to sum partial results; requires one allreduce communication
- [ ] The Megatron MLP: the first linear (`fc1`) is column-parallel; the second linear (`fc2`) is row-parallel; this allows GeLU to be applied locally after `fc1` (before the all-gather); total communication: one all-gather in the middle, one allreduce at the end — same as the non-tensor-parallel version for inference
- [ ] The Megatron attention: Q, K, V projections are column-parallel across heads; each GPU handles `n_heads / N` full attention heads; the output projection is row-parallel; similarly requires one allreduce per attention block
- [ ] Sequence parallelism: the Layer Norm and dropout operations between attention/MLP blocks are also parallelized along the sequence dimension to fully eliminate replicated computation

**Coding tasks:**

- [ ] Implement column-parallel linear and row-parallel linear as `nn.Module` subclasses using `torch.distributed.all_gather` and `torch.distributed.all_reduce`; verify output correctness vs. single-GPU baseline
- [ ] Replace the MLP in nanoGPT with Megatron-style column-parallel + row-parallel; run at tensor parallelism `N=2`; verify loss curves match single-GPU

> [!NOTE] Milestone
> Tensor parallelism `N=2` on a 12-layer, `d_model=512` model: each GPU holds half the weight matrices; the model state memory per GPU is reduced by approximately 2×. Communication overhead for one forward pass: two all-gather or allreduce operations per layer = 24 communications total; each communication moves `batch × seq_len × d_model × 2 bytes` of data. For `batch=8, seq_len=512, d_model=512`: each communication is ~4MB; 24 communications = ~96MB moved per forward pass. At PCIe bandwidth (~32 GB/s), this takes ~3ms; if the forward pass itself takes 50ms, the overhead is 6% — acceptable. At larger `d_model`, the communication cost scales quadratically while compute scales cubically, so tensor parallelism becomes more efficient at larger scales.

---

## Week 24 — Pipeline Parallelism

**Concepts to understand:**

- [ ] Pipeline parallelism: partition the model layers across `N` GPUs; GPU 0 holds layers 0 to `L/N - 1`, GPU 1 holds layers `L/N` to `2L/N - 1`, etc.; the output of each stage is sent to the next GPU via peer-to-peer communication (no collective needed)
- [ ] The pipeline bubble: in a naive pipeline, only one GPU is active at a time (the GPU processing the current micro-batch); this wastes `(N-1)/N × 100%` of GPU-time — a 4-GPU pipeline is only 25% utilized
- [ ] Micro-batch scheduling (GPipe): split the mini-batch into `m` micro-batches; pipeline them through the stages; GPU utilization improves to `m / (m + N - 1)`; at `m = 4N`, efficiency is 80%; at `m = ∞`, efficiency → 100%
- [ ] 1F1B schedule (PipeDream): interleave one forward pass with one backward pass for each micro-batch; reduces the number of in-flight activations from `O(Nm)` to `O(N + m)`, dramatically reducing memory usage while maintaining high utilization
- [ ] Tensor parallelism vs. pipeline parallelism: tensor parallelism requires fast all-to-all communication (NVLink); pipeline parallelism requires only peer-to-peer (PCIe is sufficient); pipeline parallelism is preferred when GPUs are spread across nodes (slow interconnect)

**Coding tasks:**

- [ ] Implement a 2-stage pipeline using `torch.distributed` send/recv; split nanoGPT's 12 layers across 2 GPUs (6 each); implement a 4-micro-batch GPipe schedule; verify loss matches single-GPU
- [ ] Measure the bubble fraction: for `N=2, m ∈ {1, 2, 4, 8}`, plot GPU utilization (fraction of time actually computing vs. waiting)

> [!NOTE] Milestone
> At `m=1` (no micro-batching): one GPU is active at a time, utilization is 50% for a 2-stage pipeline. At `m=4`: utilization is `4/(4+1) = 80%`. At `m=8`: `8/9 ≈ 89%`. The 20% loss at `m=4` (the bubble) translates directly to a 20% throughput loss vs. a single GPU at the same total batch size. This is why pipeline parallelism is used only when necessary (when layers cannot fit on a single GPU) and why the 1F1B schedule is preferred over GPipe in practice — it achieves similar efficiency with much lower activation memory.

---

## Week 25 — Combining Parallelism Strategies

**Concepts to understand:**

- [ ] 3D parallelism: combine data parallelism (DP) + tensor parallelism (TP) + pipeline parallelism (PP); the Megatron-LM GPT-3 training used `DP=64, TP=8, PP=8` across 512 A100s; total GPUs = `DP × TP × PP = 64 × 8 × 8 = 4096`
- [ ] The parallelism hierarchy: within a node, use TP (fast NVLink); across nodes within a cluster, use PP or DP (slower InfiniBand); across clusters, use only DP
- [ ] Communication-computation overlap: DDP overlaps gradient allreduce with the backward pass (bucketed communication); FSDP overlaps all-gather with the forward pass; pipeline parallelism overlaps forward and backward across stages; the goal is to keep all GPUs busy computing while communication happens in the background
- [ ] Effective batch size and learning rate scaling: when using DDP with `N` GPUs, effective batch size = `N × local_batch`; scale LR as `η_eff = η × N` (linear scaling rule, Goyal et al. 2017); add warmup proportional to the LR increase to prevent instability

**Coding tasks:**

- [ ] Set up a 2×2 parallelism grid (DP=2, TP=2) on 4 GPUs: assign GPUs to process groups for each parallelism dimension; verify that (a) gradients are synchronized across DP replicas, (b) tensor shards are gathered within TP groups, (c) loss matches single-GPU baseline
- [ ] Measure throughput scaling efficiency: compare (1 GPU, batch=32) vs. (4 GPUs, DP=4, batch=8 per GPU); compute scaling efficiency = `(4-GPU tokens/sec) / (4 × 1-GPU tokens/sec)`

> [!NOTE] Milestone
> Ideal scaling efficiency is 100%; realistic efficiency at 4 GPUs over PCIe is typically 70–85% due to communication overhead. If efficiency is below 60%, check: (1) are communication and computation overlapped (DDP bucketing should handle this automatically)? (2) is the model large enough that communication overhead is small relative to compute? A model that takes 10ms to compute one step and 8ms to communicate gradients has only 55% efficiency — at this point, either the model needs to be larger (more compute per step) or the communication needs to be compressed (gradient compression, which introduces bias but reduces bandwidth).
