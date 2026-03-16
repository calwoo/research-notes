# ML Performance Engineering: A Comprehensive Field Map

## Table of Contents

- [[#1. What is ML Performance Engineering|1. What is ML Performance Engineering]]
- [[#2. Core Prerequisites|2. Core Prerequisites]]
  - [[#2.1 Mathematics|2.1 Mathematics]]
  - [[#2.2 Systems Programming|2.2 Systems Programming]]
  - [[#2.3 ML Foundations|2.3 ML Foundations]]
- [[#3. GPU and Hardware Fundamentals|3. GPU and Hardware Fundamentals]]
  - [[#3.1 GPU Architecture|3.1 GPU Architecture]]
  - [[#3.2 Memory Hierarchy|3.2 Memory Hierarchy]]
  - [[#3.3 The Roofline Model|3.3 The Roofline Model]]
  - [[#3.4 CUDA Programming|3.4 CUDA Programming]]
  - [[#3.5 Triton|3.5 Triton]]
- [[#4. Distributed Training|4. Distributed Training]]
  - [[#4.1 Data Parallelism|4.1 Data Parallelism]]
  - [[#4.2 Tensor Parallelism|4.2 Tensor Parallelism]]
  - [[#4.3 Pipeline Parallelism|4.3 Pipeline Parallelism]]
  - [[#4.4 ZeRO and FSDP|4.4 ZeRO and FSDP]]
  - [[#4.5 Sequence Parallelism and Expert Parallelism|4.5 Sequence Parallelism and Expert Parallelism]]
- [[#5. Training Efficiency|5. Training Efficiency]]
  - [[#5.1 Mixed Precision Training|5.1 Mixed Precision Training]]
  - [[#5.2 Gradient Checkpointing|5.2 Gradient Checkpointing]]
  - [[#5.3 Compiler Optimizations|5.3 Compiler Optimizations]]
- [[#6. Attention Mechanisms and Efficient Attention|6. Attention Mechanisms and Efficient Attention]]
  - [[#6.1 Standard Multi-Head Attention|6.1 Standard Multi-Head Attention]]
  - [[#6.2 Multi-Query and Grouped-Query Attention|6.2 Multi-Query and Grouped-Query Attention]]
  - [[#6.3 Multi-Head Latent Attention|6.3 Multi-Head Latent Attention]]
  - [[#6.4 FlashAttention|6.4 FlashAttention]]
  - [[#6.5 Linear Attention|6.5 Linear Attention]]
- [[#7. Inference Optimization|7. Inference Optimization]]
  - [[#7.1 KV Cache|7.1 KV Cache]]
  - [[#7.2 Quantization|7.2 Quantization]]
  - [[#7.3 Speculative Decoding|7.3 Speculative Decoding]]
  - [[#7.4 Continuous Batching and PagedAttention|7.4 Continuous Batching and PagedAttention]]
- [[#8. Serving Systems|8. Serving Systems]]
- [[#9. Diffusion Model Efficiency|9. Diffusion Model Efficiency]]
- [[#10. Canonical Papers by Topic|10. Canonical Papers by Topic]]
- [[#11. Textbooks and Courses|11. Textbooks and Courses]]
- [[#12. Key Blogs and Technical Write-Ups|12. Key Blogs and Technical Write-Ups]]
- [[#References|References]]

---

## 1. What is ML Performance Engineering

*ML performance engineering* (also called ML systems engineering or AI infrastructure engineering) is the discipline of making training and inference of large neural models fast, memory-efficient, and cost-effective at scale. It sits at the intersection of numerical computing, computer architecture, distributed systems, and machine learning research. A practitioner must fluently reason about floating-point arithmetic, GPU microarchitecture, communication topology, and model structure simultaneously.

For *generative models* specifically — autoregressive LLMs and diffusion models — the constraints are extreme: models with hundreds of billions of parameters, context windows of hundreds of thousands of tokens, and latency requirements measured in milliseconds per token. Every technique in this map is ultimately a response to one of three bottlenecks:

- **Compute bound**: the GPU's arithmetic units are the limiting factor (FLOP/s ceiling).
- **Memory-bandwidth bound**: data movement between DRAM and compute units is the limiting factor (HBM bandwidth ceiling).
- **Overhead / latency bound**: Python interpreter overhead, kernel launch latency, or communication stalls dominate.

---

## 2. Core Prerequisites

### 2.1 Mathematics

| Topic | What to Know | Difficulty |
|---|---|---|
| Linear algebra | Matrix multiplication, SVD, low-rank approximations, Kronecker products | Beginner |
| Probability and statistics | Distributions, KL divergence, score functions, ELBO | Beginner |
| Numerical analysis | Floating-point representation (FP32/BF16/FP8), rounding error, numerical stability | Intermediate |
| Information theory | Entropy, cross-entropy, mutual information, rate-distortion | Intermediate |
| Optimization | SGD, Adam, learning rate schedules, loss landscapes | Beginner |

**Resources:**

| Resource | Type | URL |
|---|---|---|
| *Mathematics for Machine Learning* (Deisenroth, Faisal, Ong) | Textbook | https://mml-book.github.io |
| *Deep Learning* (Goodfellow, Bengio, Courville) — Chapters 2–4 | Textbook | https://www.deeplearningbook.org |
| *Numerical Methods for Engineers* (Chapra & Canale) — floating-point chapter | Textbook | — |

### 2.2 Systems Programming

| Topic | What to Know | Difficulty |
|---|---|---|
| C/C++ | Pointers, memory management, templates, RAII | Intermediate |
| Python | CPython internals, GIL, ctypes/cffi, Cython | Intermediate |
| Operating systems | Processes, threads, memory paging, virtual memory | Intermediate |
| Computer architecture | Cache hierarchy, SIMD, out-of-order execution, branch prediction | Intermediate |
| Profiling tools | `perf`, `Nsight Systems`, `Nsight Compute`, `py-spy` | Advanced |

**Resources:**

| Resource | Type | URL |
|---|---|---|
| *Computer Systems: A Programmer's Perspective* (Bryant & O'Hallaron) | Textbook | — |
| *Computer Organization and Design: ARM Edition* (Patterson & Hennessy) | Textbook | — |
| NVIDIA Nsight Systems documentation | Docs | https://developer.nvidia.com/nsight-systems |

### 2.3 ML Foundations

| Topic | What to Know | Difficulty |
|---|---|---|
| Transformer architecture | Attention, positional encoding, layer norm, residual connections | Beginner |
| Autoregressive decoding | Token-by-token generation, greedy vs. sampling, beam search | Beginner |
| Diffusion models | DDPM, DDIM, score matching, noise schedules | Intermediate |
| Backpropagation | Computational graph, chain rule, gradient accumulation | Beginner |

**Resources:**

| Resource | Type | URL |
|---|---|---|
| *Attention is All You Need* (Vaswani et al., 2017) | Paper | https://arxiv.org/abs/1706.03762 |
| Andrej Karpathy's *Neural Networks: Zero to Hero* video series | Course | https://karpathy.ai/zero-to-hero.html |
| *Denoising Diffusion Probabilistic Models* (Ho et al., 2020) | Paper | https://arxiv.org/abs/2006.11239 |

---

## 3. GPU and Hardware Fundamentals

### 3.1 GPU Architecture

A modern NVIDIA GPU (e.g. H100 SXM) consists of:

- **Streaming Multiprocessors (SMs)**: each SM contains multiple CUDA cores (FP32 units), Tensor Cores (for FP16/BF16/FP8 matrix multiply-accumulate), L1 cache, and shared memory.
- **Tensor Cores**: specialized matrix-multiply units that compute a $4 \times 4$ (Hopper: $8 \times 8$) multiply-accumulate in a single clock cycle, enabling throughput of ~1 PFLOP/s (BF16) per H100.
- **Warp**: a group of 32 threads that execute in lockstep (SIMT model). Divergent branches cause serialization.
- **Thread hierarchy**: threads → warps → thread blocks → grids. Shared memory is per-block; L2 cache and HBM are shared across SMs.

Key specs to memorize per-generation:

| GPU | HBM BW | Peak TFLOP/s (BF16) | HBM Capacity | NVLink BW |
|---|---|---|---|---|
| A100 80GB SXM | 2 TB/s | ~312 TFLOP/s | 80 GB | 600 GB/s |
| H100 SXM | 3.35 TB/s | ~989 TFLOP/s | 80 GB | 900 GB/s |
| H200 SXM | 4.8 TB/s | ~989 TFLOP/s | 141 GB | 900 GB/s |

**Resources:**

| Resource | Type | URL | Difficulty |
|---|---|---|---|
| NVIDIA H100 Whitepaper | Docs | https://resources.nvidia.com/en-us-tensor-core | Intermediate |
| *CUDA C Programming Guide* (NVIDIA) | Docs | https://docs.nvidia.com/cuda/cuda-c-programming-guide | Intermediate |
| Chip Huyen's *AI Engineering* book — hardware chapter | Textbook | — | Beginner |

### 3.2 Memory Hierarchy

Understanding the memory hierarchy is central to all performance work. From fastest to slowest (and smallest to largest):

1. **Registers** (~100s of KB per SM, sub-cycle latency)
2. **Shared memory / L1 cache** (~228 KB per SM on H100, ~5 ns latency) — explicitly managed in CUDA
3. **L2 cache** (~50 MB on H100, ~50 ns latency) — GPU-wide, transparent
4. **HBM (High Bandwidth Memory)** (80–141 GB, ~300 ns latency, ~3–5 TB/s) — "global memory" in CUDA
5. **NVLink / PCIe** (~900 GB/s NVLink, ~64 GB/s PCIe) — inter-GPU communication
6. **CPU DRAM / NFS / storage** (GB/s range) — host memory

*The fundamental performance principle*: any kernel that reads or writes HBM is memory-bandwidth bound unless it performs enough arithmetic per byte to hide the latency. This ratio is *arithmetic intensity*, measured in FLOP/byte.

### 3.3 The Roofline Model

The *roofline model* is a visual framework for determining whether a kernel is compute-bound or memory-bandwidth bound. Define:

- $I = \text{FLOP} / \text{Bytes}$ (arithmetic intensity, in FLOP/byte)
- $P_{\text{peak}}$ = peak compute throughput (TFLOP/s)
- $B_{\text{peak}}$ = peak memory bandwidth (TB/s)
- *Ridge point*: $I^* = P_{\text{peak}} / B_{\text{peak}}$

The achievable performance is:

$$\text{Perf} = \min(P_{\text{peak}},\ B_{\text{peak}} \cdot I)$$

For the H100, $I^* \approx 989 / 3.35 \approx 295$ FLOP/byte. A matrix multiply with large matrices has $I \propto N$ and is solidly compute-bound; an elementwise softmax has $I \approx 4$ FLOP/byte and is deeply memory-bandwidth bound.

**Key implication for attention**: naive attention is memory-bandwidth bound because it reads/writes the full $N \times N$ attention matrix from HBM. FlashAttention (Section 6.4) restructures the computation to remain in SRAM, pushing the kernel into the compute-bound regime.

**Resources:**

| Resource | Type | URL | Difficulty |
|---|---|---|---|
| "All About Rooflines" — JAX Scaling Book | Blog | https://jax-ml.github.io/scaling-book/roofline/ | Intermediate |
| Modal GPU Glossary: Roofline Model | Blog | https://modal.com/gpu-glossary/perf/roofline-model | Beginner |
| Original roofline paper (Williams et al., 2009) | Paper | https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf | Intermediate |

### 3.4 CUDA Programming

Core concepts in CUDA C/C++:

- **Kernel launch**: `<<<gridDim, blockDim>>>` syntax; each SM runs one or more blocks concurrently.
- **Memory management**: `cudaMalloc`, `cudaMemcpy`, unified memory (`cudaMallocManaged`).
- **Shared memory**: declared with `__shared__`; enables tiling — loading a tile of a matrix into shared memory to reuse across threads in the same block.
- **Synchronization**: `__syncthreads()` within a block; `cudaDeviceSynchronize()` across host/device.
- **Warp-level primitives**: `__shfl_down_sync`, `__ballot_sync` for fast warp-wide reductions without shared memory.
- **CUDA streams**: multiple independent streams can run concurrently on a device, enabling kernel-kernel overlap and compute-copy overlap.
- **Occupancy**: the fraction of maximum warps resident on an SM. Low occupancy (due to high register usage or shared memory usage per block) limits latency hiding.

**Resources:**

| Resource | Type | URL | Difficulty |
|---|---|---|---|
| *Programming Massively Parallel Processors*, 4th ed. (Hwu, Kirk, El Hajj) — "PMPP" | Textbook | https://www.oreilly.com/library/view/programming-massively-parallel/9780323984638/ | Intermediate |
| ECE408/CS483 — Applied Parallel Programming (UIUC) | Course | https://developer.nvidia.com/educators/existing-courses | Intermediate |
| NVIDIA CUDA C Programming Guide | Docs | https://docs.nvidia.com/cuda/cuda-c-programming-guide | Intermediate |

### 3.5 Triton

*Triton* is a Python-embedded DSL from OpenAI that compiles to PTX/SASS, allowing researchers to write high-performance GPU kernels without raw CUDA. It abstracts away warp-level management and auto-tunes tile sizes. `torch.compile` (TorchInductor backend) generates Triton kernels automatically for most ops.

Key concepts:

- **Program (Triton's abstraction for a thread block)**: each program handles a tile of output.
- **`tl.load` / `tl.store`**: masked memory access with automatic vectorization.
- **`tl.dot`**: Tensor Core matrix multiplication.
- **Auto-tuning**: `@triton.autotune` decorator sweeps tile shape and number of warps.

**Resources:**

| Resource | Type | URL | Difficulty |
|---|---|---|---|
| Triton official tutorials | Docs | https://triton-lang.org/main/getting-started/tutorials/ | Intermediate |
| "Unleashing the Power of Triton" (Chaim Rand, TDS) | Blog | https://towardsdatascience.com/unleashing-the-power-of-triton-mastering-gpu-kernel-optimization-in-python-160a3f52701e/ | Intermediate |
| PyTorch docs: Using User-Defined Triton Kernels with torch.compile | Docs | https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html | Advanced |

---

## 4. Distributed Training

### 4.1 Data Parallelism

In *data parallelism* (DP), each device holds a full copy of the model and processes a different micro-batch. After the backward pass, gradients are all-reduced across devices (ring all-reduce is $O(2(N-1)/N \cdot \text{data})$ communication). DDP (PyTorch DistributedDataParallel) overlaps gradient communication with backward computation by bucketing gradients.

- **Gradient accumulation**: run multiple forward/backward passes before the optimizer step to simulate a large effective batch without inter-node communication on every step.
- **Limitation**: model must fit on a single device.

### 4.2 Tensor Parallelism

*Tensor parallelism* (TP), introduced in Megatron-LM, shards individual weight matrices across devices. For a linear layer $Y = XA$ where $A \in \mathbb{R}^{d \times d}$, column-wise sharding computes $Y_i = X A_i$ on device $i$; row-wise sharding requires an all-reduce of partial sums. Attention heads are naturally parallelizable across devices.

- Communication: one all-reduce per transformer layer (two for the feedforward sub-layer).
- Requires fast intra-node interconnect (NVLink); typically limited to 8 devices per tensor-parallel group due to communication overhead.

### 4.3 Pipeline Parallelism

*Pipeline parallelism* (PP) partitions the model's layers across devices. GPipe divides a mini-batch into micro-batches and pipelines them — while device $k$ runs the forward pass for micro-batch $m+1$, device $k+1$ runs it for micro-batch $m$. The *pipeline bubble* (idle time at startup and teardown) has relative size $\approx (p-1)/(p + m - 1)$ where $p$ is the number of pipeline stages and $m$ is the number of micro-batches. Interleaved schedules (1F1B) reduce the bubble.

### 4.4 ZeRO and FSDP

*ZeRO* (Zero Redundancy Optimizer, DeepSpeed) eliminates the model state redundancy inherent in data parallelism:

- **ZeRO-1**: shards optimizer states across DP ranks ($4\times$ memory reduction for Adam, which stores first and second moments).
- **ZeRO-2**: shards optimizer states + gradients ($8\times$ reduction).
- **ZeRO-3**: shards optimizer states + gradients + parameters ($N\times$ reduction where $N$ is the DP rank count). Parameters are gathered on-demand before each forward/backward computation.

**PyTorch FSDP** (Fully Sharded Data Parallel) implements ZeRO-3 natively in PyTorch. It supports *mixed-precision sharding* (store in BF16, gather in BF16, compute in BF16) and *activation checkpointing*. FSDP2 (PyTorch 2.x) adds DTensor-based sharding for simpler composition with TP.

**Resources:**

| Resource | Type | URL | Difficulty |
|---|---|---|---|
| "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (Rajbhandari et al., 2020) | Paper | https://arxiv.org/abs/1910.02054 | Intermediate |
| "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM" (Narayanan et al., 2021) | Paper | https://arxiv.org/abs/2104.04473 | Advanced |
| Sumanth's "Everything about Distributed Training and Efficient Finetuning" | Blog | https://sumanthrh.com/post/distributed-and-efficient-finetuning/ | Intermediate |
| PyTorch FSDP Blog Post | Blog | https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/ | Intermediate |

### 4.5 Sequence Parallelism and Expert Parallelism

*Sequence parallelism* (Megatron-LM) shards along the sequence dimension for the non-attention parts of the transformer (layer norm, dropout) that tensor parallelism cannot parallelize. It requires two additional all-gather and reduce-scatter operations per layer but eliminates the sequence-length bottleneck on single-device memory.

*Expert parallelism* (used in MoE models like Mixtral, DeepSeek-V3) routes tokens to different experts residing on different devices. Each expert processes only the tokens routed to it; an all-to-all collective moves tokens between devices. The challenge is load imbalance: auxiliary load-balancing losses penalize routing all tokens to a subset of experts.

---

## 5. Training Efficiency

### 5.1 Mixed Precision Training

In *mixed precision training*, forward and backward passes use FP16 or BF16, while the master copy of weights and optimizer states are kept in FP32. This reduces memory by ~2× and doubles throughput on Tensor Cores.

**BF16 vs FP16**:
- FP16: 5-bit exponent, 10-bit mantissa. Prone to overflow for large gradients; requires loss scaling.
- BF16: 8-bit exponent, 7-bit mantissa (same exponent range as FP32). Overflow-safe; preferred for training.

**FP8 (Hopper/Ada)**: 4-bit exponent or 4-bit mantissa variants (E4M3, E5M2). NVIDIA's Transformer Engine automatically selects FP8 for GEMM operations and handles per-tensor scaling. Can double throughput over BF16 with careful calibration.

**Resources:**

| Resource | Type | URL | Difficulty |
|---|---|---|---|
| NVIDIA mixed precision training guide | Docs | https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html | Intermediate |
| "Transformer Math 101" (EleutherAI Blog) | Blog | https://blog.eleuther.ai/transformer-math/ | Intermediate |

### 5.2 Gradient Checkpointing

Naively, training stores all intermediate activations for the backward pass, requiring $O(n)$ memory for a depth-$n$ network. *Gradient checkpointing* (activation recomputation) stores only a subset of activations and recomputes the rest during the backward pass. Choosing which layers to checkpoint optimally reduces peak activation memory to $O(\sqrt{n})$ at the cost of one additional forward pass. In practice, most frameworks checkpoint at transformer layer boundaries.

PyTorch API: `torch.utils.checkpoint.checkpoint(fn, *inputs)`.

### 5.3 Compiler Optimizations

**`torch.compile`** (PyTorch 2.x): uses TorchDynamo (Python bytecode capture) + TorchInductor (code generation backend) to JIT-compile Python/PyTorch graphs into optimized Triton kernels. Key benefits:
- *Kernel fusion*: eliminates redundant HBM reads/writes for chains of elementwise ops (e.g., fused LayerNorm, fused attention).
- *Graph-level optimizations*: constant folding, dead code elimination, horizontal fusion.
- Modes: `torch.compile(model, mode="reduce-overhead")` for latency, `mode="max-autotune"` for throughput (slower compilation).

**XLA / JAX**: JAX uses XLA (Accelerated Linear Algebra compiler) which performs whole-program optimization via HLO (High Level Operations) IR. Enables aggressive fusion and SPMD-style sharding for TPU and GPU.

**Resources:**

| Resource | Type | URL | Difficulty |
|---|---|---|---|
| PyTorch 2.0 Introduction Blog | Blog | https://pytorch.org/blog/pytorch-2.0-release/ | Intermediate |
| *AI Systems Performance Engineering* (O'Reilly) — Chapter 14: PyTorch Compiler, Triton, and XLA | Book | https://www.oreilly.com/library/view/ai-systems-performance/9798341627772/ch14.html | Advanced |

---

## 6. Attention Mechanisms and Efficient Attention

### 6.1 Standard Multi-Head Attention

For query, key, value matrices $Q, K, V \in \mathbb{R}^{N \times d_k}$, multi-head attention (MHA) computes:

$$\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

where $\text{head}_i = \text{softmax}\!\left(\frac{Q_i K_i^\top}{\sqrt{d_k}}\right) V_i$.

Complexity: $O(N^2 d)$ compute, $O(N^2)$ memory for the attention matrix. The $O(N^2)$ memory is the primary obstacle to long-context modeling.

During autoregressive inference, each new token attends to all previous tokens. The *KV cache* stores previous $K$ and $V$ projections to avoid recomputation, at the cost of $O(N \cdot d \cdot \text{num\_layers})$ memory per sequence.

### 6.2 Multi-Query and Grouped-Query Attention

**Multi-Query Attention (MQA)** (Shazeer 2019): a single set of $K, V$ heads is shared across all query heads. Reduces KV cache size by a factor of $h$ (number of heads) at a small quality cost.

**Grouped-Query Attention (GQA)** (Ainslie et al. 2023): $G$ groups of query heads share one $K, V$ pair per group. GQA with $G = 1$ recovers MQA; $G = h$ recovers MHA. Used in LLaMA 3, Mistral, Gemma. Reduces KV cache by $h/G$ with near-MHA quality.

**Resources:**

| Resource | Type | URL | Difficulty |
|---|---|---|---|
| "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (Ainslie et al., 2023) | Paper | https://arxiv.org/abs/2305.13245 | Intermediate |
| Fast Self-Attention Mechanisms: MQA, GQA, Flash and Page Attentions (Medium) | Blog | https://medium.com/@lmpo/the-race-for-faster-transformers-innovations-in-self-attention-e602fb1b5f20 | Beginner |

### 6.3 Multi-Head Latent Attention

*Multi-Head Latent Attention* (MLA), introduced in DeepSeek-V2, further compresses the KV cache by projecting keys and values into a shared low-dimensional *latent space*. Concretely, rather than caching $K$ and $V$ separately, MLA caches a single compressed representation $c^{KV}_t \in \mathbb{R}^{d_c}$ with $d_c \ll d_h \cdot n_h$, and recovers full $K, V$ via learned up-projection matrices at inference time.

The KV cache size per token drops from $2 \cdot d_h \cdot n_h$ (MHA) to $d_c$ (MLA), achieving 5–13× compression with competitive or superior quality. Because MLA's latent representation is shared across heads, it increases the arithmetic intensity of the attention kernel, making it more compute-bound.

**Resources:**

| Resource | Type | URL | Difficulty |
|---|---|---|---|
| "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model" (DeepSeek-AI, 2024) | Paper | https://arxiv.org/abs/2405.04434 | Advanced |

### 6.4 FlashAttention

*FlashAttention* (Dao et al. 2022) is an IO-aware exact attention algorithm. The key insight is that the bottleneck in standard attention is the HBM reads/writes of the $N \times N$ attention matrix, not the FLOPs. FlashAttention tiles the computation into blocks that fit in SRAM, uses the *online softmax* trick (incrementally updating the running max and normalizer), and never materializes the full attention matrix in HBM.

Memory: $O(N)$ instead of $O(N^2)$.
Speedup: 2–4× wall-clock on A100 for typical sequence lengths.

**FlashAttention-2** (Dao 2023): better parallelism across sequence dimension, fewer non-matmul FLOPs, improved performance on A100/H100.

**FlashAttention-3** (Shah et al. 2024): exploits Hopper-specific features — WGMMA (warpgroup matrix multiply-accumulate) and TMA (tensor memory accelerator) for asynchronous data movement, achieving ~75% of H100 peak FLOP/s.

**Resources:**

| Resource | Type | URL | Difficulty |
|---|---|---|---|
| "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (Dao et al., 2022) | Paper | https://arxiv.org/abs/2205.14135 | Intermediate |
| "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (Dao, 2023) | Paper | https://arxiv.org/abs/2307.08691 | Advanced |
| "FlashAttention-3" (Shah et al., 2024) | Paper | https://arxiv.org/abs/2407.08608 | Advanced |
| FlashInfer: Efficient and Customizable Attention Engine | Paper | https://arxiv.org/abs/2501.01005 | Advanced |

### 6.5 Linear Attention

*Linear attention* approximates the softmax kernel with a feature map $\phi$ such that $\text{softmax}(QK^\top) \approx \phi(Q)\phi(K)^\top$, enabling the KV interaction to be computed as a running outer product sum $S_t = \sum_{i \leq t} \phi(k_i) v_i^\top$. This reduces complexity to $O(N d^2)$ (linear in sequence length) but at a non-trivial quality cost for large models.

Notable variants: RWKV (linear recurrence), Mamba (selective state space), RetNet (retention mechanism). These are increasingly practical for long-context settings where quadratic attention is prohibitive.

**Resources:**

| Resource | Type | URL | Difficulty |
|---|---|---|---|
| "Transformers are RNNs" (Katharopoulos et al., 2020) | Paper | https://arxiv.org/abs/2006.16236 | Advanced |
| "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023) | Paper | https://arxiv.org/abs/2312.00752 | Advanced |

---

## 7. Inference Optimization

### 7.1 KV Cache

During autoregressive generation, the key and value projections of all previous tokens are cached to avoid recomputation. For a model with $L$ layers, $h$ heads, head dimension $d_h$, precision $b$ bytes per element, and sequence length $N$:

$$\text{KV cache size} = 2 \cdot L \cdot h \cdot d_h \cdot N \cdot b \text{ bytes}$$

For a 70B LLaMA model in BF16 with $L=80$, $h=64$, $d_h=128$, $N=8192$: $\approx 160$ GB. This is the primary memory bottleneck for inference.

Techniques to reduce KV cache:
- GQA/MQA (architectural reduction by $h/G$)
- MLA (architectural latent compression)
- KV cache quantization (e.g., FP8 or INT4 KV)
- Eviction-based compression (H2O, StreamingLLM)

### 7.2 Quantization

*Post-training quantization* (PTQ) reduces weight or activation precision without retraining. Key schemes:

| Method | Target | Precision | Notes |
|---|---|---|---|
| LLM.int8() (Dettmers et al., 2022) | Weights + activations | INT8 | Mixed-precision: FP16 for outliers, INT8 for rest |
| GPTQ (Frantar et al., 2022) | Weights | INT4 / INT3 | Second-order Hessian-based post-training quantization |
| AWQ (Lin et al., 2023) | Weights | INT4 | Activation-aware; skips salient weights |
| SmoothQuant (Xiao et al., 2022) | Weights + activations | INT8 | Migrates quantization difficulty from activations to weights |
| GGUF / llama.cpp | Weights | 2–8 bit | CPU-friendly quantization, multiple schemes |

**Resources:**

| Resource | Type | URL | Difficulty |
|---|---|---|---|
| "LLM.int8()" (Dettmers et al., 2022) | Paper | https://arxiv.org/abs/2208.07339 | Intermediate |
| "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (Frantar et al., 2022) | Paper | https://arxiv.org/abs/2210.17323 | Advanced |
| "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" (Lin et al., 2023) | Paper | https://arxiv.org/abs/2306.00978 | Advanced |
| Hugging Face Quantization docs | Docs | https://huggingface.co/docs/transformers/en/main_classes/quantization | Beginner |

### 7.3 Speculative Decoding

*Speculative decoding* (Leviathan et al. 2023; Chen et al. 2023) decouples the draft and verification phases. A small, fast *draft model* proposes $k$ tokens autoregressively; the large *target model* verifies all $k$ tokens in a single forward pass (using parallel scoring). Accepted tokens are kept via a modified rejection sampling scheme that preserves the exact target distribution. Typical speedups: 2–3× for $k = 4$–8.

Variants:
- *Self-speculative decoding*: the target model itself generates drafts (e.g., using early exit or a subset of layers).
- *Medusa*: trains multiple draft heads on top of the target model's hidden states.
- *EAGLE*: draft model conditions on target model's features, achieving better acceptance rate.

**Resources:**

| Resource | Type | URL | Difficulty |
|---|---|---|---|
| "Fast Inference from Transformers via Speculative Decoding" (Leviathan et al., 2023) | Paper | https://arxiv.org/abs/2211.17192 | Intermediate |
| "Accelerating Large Language Model Decoding with Speculative Sampling" (Chen et al., 2023) | Paper | https://arxiv.org/abs/2302.01318 | Intermediate |

### 7.4 Continuous Batching and PagedAttention

**Continuous batching** (Orca, 2022; vLLM, 2023): traditional static batching waits for all sequences in a batch to finish before starting new ones. Continuous batching replaces completed sequences with new ones at each *iteration* (each decode step), dramatically improving GPU utilization when request lengths are heterogeneous. Throughput gains of 10–23× over static batching in production settings.

**PagedAttention** (Kwon et al., 2023 — vLLM): KV cache memory is allocated in fixed-size *pages* (analogous to OS virtual memory pages) rather than contiguous buffers. A *block table* maps logical KV positions to physical pages, enabling:
- Near-zero memory waste (only the last page of each sequence is partially filled).
- Cross-request KV sharing (prompt caching, parallel sampling sharing prefill pages).
- Memory fragmentation $< 4\%$.

**Resources:**

| Resource | Type | URL | Difficulty |
|---|---|---|---|
| "Efficient Memory Management for Large Language Model Serving with PagedAttention" (Kwon et al., 2023) | Paper | https://arxiv.org/abs/2309.06180 | Intermediate |
| vLLM blog: "Easy, Fast, and Cheap LLM Serving with PagedAttention" | Blog | https://blog.vllm.ai/2023/06/20/vllm.html | Beginner |
| Anyscale: "Achieve 23x LLM Inference Throughput with Continuous Batching" | Blog | https://www.anyscale.com/blog/continuous-batching-llm-inference | Beginner |

---

## 8. Serving Systems

| System | Organization | Key Features | Best For | Difficulty |
|---|---|---|---|---|
| **vLLM** | UC Berkeley / vLLM team | PagedAttention, continuous batching, chunked prefill, multi-LoRA, OpenAI-compatible API | General production serving | Intermediate |
| **SGLang** | Stanford / LMSys | RadixAttention (KV prefix sharing), structured generation, low-latency multi-turn | Agentic workloads, heavy KV reuse | Advanced |
| **TensorRT-LLM** | NVIDIA | Fused CUDA kernels, CUDA graphs, INT8/FP8, best on NVIDIA hardware | Maximum throughput on NVIDIA GPUs | Advanced |
| **TGI (Text Generation Inference)** | Hugging Face | Production-ready, OpenTelemetry, sharded serving, LoRA | Rapid deployment of HF models | Beginner |
| **llama.cpp** | Georgi Gerganov | CPU inference, GGUF quantization, cross-platform | Edge/on-device, CPU-only | Beginner |
| **Ollama** | Ollama | Developer-friendly wrapper over llama.cpp, local serving | Local development, MacOS | Beginner |

**Selection heuristics:**

- For maximum raw throughput on NVIDIA hardware: **TensorRT-LLM**.
- For a flexible, well-maintained open-source server: **vLLM**.
- For agentic / RAG pipelines with heavy prompt reuse: **SGLang** (RadixAttention reuses shared prefix pages).
- For CPU / edge: **llama.cpp** / **Ollama**.

**Resources:**

| Resource | Type | URL | Difficulty |
|---|---|---|---|
| vLLM documentation | Docs | https://docs.vllm.ai/en/latest/ | Intermediate |
| SGLang paper | Paper | https://arxiv.org/abs/2312.07104 | Advanced |
| "Comparing the Top 6 Inference Runtimes for LLM Serving in 2025" (MarkTechPost) | Blog | https://www.marktechpost.com/2025/11/07/comparing-the-top-6-inference-runtimes-for-llm-serving-in-2025/ | Beginner |

---

## 9. Diffusion Model Efficiency

Diffusion models (DDPM, DDIM, Stable Diffusion, FLUX, Sora) present different performance challenges from LLMs:

- **Iterative denoising**: inference requires $T$ forward passes (50–1000 for DDPM, 4–50 for DDIM), making inference $T$x more expensive than a single-step model.
- **Model architecture**: U-Net (older, SD 1.x/2.x) or Diffusion Transformer / DiT (newer, SD3, FLUX, Sora). DiTs scale more predictably with compute.
- **Parallelism**: DistriFusion (CVPR 2024) and PipeFusion parallelize the denoising U-Net / DiT across GPUs using asynchronous activation reuse.

Key optimization techniques:

| Technique | Description | Speedup |
|---|---|---|
| DDIM / DPM-Solver | Fewer denoising steps via ODE solvers | 10–50× over DDPM |
| Consistency Models | One or few-step generation via distillation | Up to 50× |
| DeepCache | Cache and reuse high-level U-Net features across timesteps | 2–5× |
| Quantization (PTQD, Q-Diffusion) | INT8/INT4 weights and activations | 2–4× |
| Flash Attention in DiT | Replaces naive attention in transformer blocks | 2–3× |
| Distillation (ADD, LCM) | Student learns from teacher in fewer steps | 4–50× |

**Resources:**

| Resource | Type | URL | Difficulty |
|---|---|---|---|
| "Efficient Diffusion Models: A Survey" (TMLR 2025) | Paper | https://arxiv.org/abs/2502.06805 | Intermediate |
| "DeepCache: Accelerating Diffusion Models for Free" (CVPR 2024) | Paper | https://arxiv.org/abs/2312.00858 | Intermediate |
| "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023) — DiT | Paper | https://arxiv.org/abs/2212.09748 | Intermediate |

---

## 10. Canonical Papers by Topic

### GPU Architecture and Roofline

| Paper | Year | Key Contribution |
|---|---|---|
| "Roofline: An Insightful Visual Performance Model" (Williams et al.) | 2009 | Foundational roofline framework |
| NVIDIA H100 Tensor Core GPU Architecture Whitepaper | 2022 | Hopper architecture: Tensor Memory Accelerator, FP8 |

### Efficient Attention

| Paper | Year | Key Contribution |
|---|---|---|
| "Attention is All You Need" (Vaswani et al.) | 2017 | Original Transformer |
| "Fast Transformer Decoding: One Write-Head is All You Need" (Shazeer) | 2019 | Multi-Query Attention |
| "FlashAttention" (Dao et al.) | 2022 | IO-aware tiled attention |
| "GQA: Training Generalized Multi-Query Transformer Models" (Ainslie et al.) | 2023 | Grouped-Query Attention |
| "FlashAttention-2" (Dao) | 2023 | Improved parallelism and FLOPs |
| "DeepSeek-V2" (DeepSeek-AI) | 2024 | Multi-Head Latent Attention |
| "FlashAttention-3" (Shah et al.) | 2024 | Hopper-specific async attention |

### Distributed Training

| Paper | Year | Key Contribution |
|---|---|---|
| "Megatron-LM: Training Multi-Billion Parameter Language Models" (Shoeybi et al.) | 2019 | Tensor parallelism for transformers |
| "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (Rajbhandari et al.) | 2020 | Sharded optimizer states, gradients, params |
| "Efficient Large-Scale LM Training Using Megatron-LM" (Narayanan et al.) | 2021 | 3D parallelism (TP + PP + DP) |
| "Reducing Activation Recomputation in Large Transformer Models" (Korthikanti et al.) | 2022 | Sequence parallelism |

### Inference Optimization

| Paper | Year | Key Contribution |
|---|---|---|
| "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale" (Dettmers et al.) | 2022 | First INT8 inference at scale |
| "GPTQ: Accurate Post-Training Quantization for GPTs" (Frantar et al.) | 2022 | 4-bit PTQ via second-order Hessian |
| "Efficient Memory Management for LLM Serving with PagedAttention" (Kwon et al.) | 2023 | Virtual-memory-style KV cache paging |
| "Fast Inference from Transformers via Speculative Decoding" (Leviathan et al.) | 2023 | Draft-then-verify decoding |
| "AWQ: Activation-aware Weight Quantization" (Lin et al.) | 2023 | Weight-only INT4 with activation scaling |
| "Sarathi-Serve: Taming Throughput-Latency Tradeoff" (Agrawal et al.) | 2024 | Chunked prefill for mixed SLOs |

---

## 11. Textbooks and Courses

| Resource | Type | Topic | Difficulty | URL |
|---|---|---|---|---|
| *Programming Massively Parallel Processors*, 4th ed. (Hwu, Kirk, El Hajj) | Textbook | CUDA, GPU architecture, parallel patterns | Intermediate | https://www.oreilly.com/library/view/programming-massively-parallel/9780323984638/ |
| *Computer Systems: A Programmer's Perspective* (Bryant & O'Hallaron) | Textbook | Systems programming, memory hierarchy, caching | Beginner | — |
| *Deep Learning* (Goodfellow, Bengio, Courville) | Textbook | ML theory, optimization, regularization | Beginner | https://www.deeplearningbook.org |
| ECE408 / CS483 Applied Parallel Programming (UIUC) | Course | CUDA, parallel algorithms | Intermediate | https://developer.nvidia.com/educators/existing-courses |
| Columbia COMSE6998-013 High-Performance Machine Learning (Fall 2024) | Course | PyTorch profiling, CUDA, quantization | Intermediate | https://www.cs.columbia.edu/~aa4870/high-performance-machine-learning/ |
| *AI Systems Performance Engineering* (O'Reilly) | Textbook | torch.compile, Triton, XLA | Advanced | https://www.oreilly.com/library/view/ai-systems-performance/9798341627772/ |
| Andrej Karpathy's *Neural Networks: Zero to Hero* | Course | Transformer implementation from scratch | Beginner | https://karpathy.ai/zero-to-hero.html |
| JAX Scaling Book | Online book | Distributed training, roofline, sharding | Intermediate | https://jax-ml.github.io/scaling-book/ |
| fast.ai *Practical Deep Learning for Coders* (Part 2) | Course | PyTorch internals, custom CUDA | Intermediate | https://course.fast.ai |

---

## 12. Key Blogs and Technical Write-Ups

| Title | Author / Org | Topic | Difficulty | URL |
|---|---|---|---|---|
| "Making Deep Learning Go Brrrr From First Principles" | Horace He | Memory-bandwidth vs compute-bound, overhead | Beginner | https://horace.io/brrr_intro.html |
| "Transformer Math 101" | EleutherAI | Parameter count, memory arithmetic, FLOP budgets | Beginner | https://blog.eleuther.ai/transformer-math/ |
| Lil'Log — "Attention? Attention!" | Lilian Weng | Attention mechanism history and variants | Beginner | https://lilianweng.github.io/posts/2018-06-24-attention/ |
| "The Illustrated Transformer" | Jay Alammar | Visual walkthrough of attention | Beginner | https://jalammar.github.io/illustrated-transformer/ |
| "Efficient Training on Multiple GPUs" | Hugging Face | DP, TP, PP, ZeRO practical guide | Intermediate | https://huggingface.co/docs/transformers/perf_train_gpu_many |
| "vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention" | vLLM Blog | PagedAttention, continuous batching | Intermediate | https://blog.vllm.ai/2023/06/20/vllm.html |
| "Achieve 23x LLM Inference Throughput with Continuous Batching" | Anyscale | Continuous batching mechanics | Intermediate | https://www.anyscale.com/blog/continuous-batching-llm-inference |
| "GPUs Go Brrr" | Hazy Research (Stanford) | ThunderKittens / custom kernel design | Advanced | https://hazyresearch.stanford.edu/blog/2024-05-12-tk |
| "Everything about Distributed Training and Efficient Finetuning" | Sumanth Rao | ZeRO, FSDP, practical recipes | Intermediate | https://sumanthrh.com/post/distributed-and-efficient-finetuning/ |
| "A Gentle Introduction to 8-bit Matrix Multiplication" | Hugging Face | bitsandbytes integration, LLM.int8() | Beginner | https://huggingface.co/blog/hf-bitsandbytes-integration |
| "Inside vLLM: Anatomy of a High-Throughput LLM Inference System" | vLLM Blog | vLLM internals, scheduler, engine | Advanced | https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html |
| Awesome-LLM-Inference (GitHub) | xlite-dev | Curated list of inference papers and code | Reference | https://github.com/xlite-dev/Awesome-LLM-Inference |

---

## References

| Reference Name | Brief Summary | Link |
|---|---|---|
| "Attention is All You Need" (Vaswani et al., 2017) | Original Transformer paper | https://arxiv.org/abs/1706.03762 |
| "FlashAttention" (Dao et al., 2022) | IO-aware, memory-efficient exact attention | https://arxiv.org/abs/2205.14135 |
| "FlashAttention-2" (Dao, 2023) | Improved parallelism, better FLOPs efficiency | https://arxiv.org/abs/2307.08691 |
| "FlashAttention-3" (Shah et al., 2024) | Hopper-native asynchronous attention | https://arxiv.org/abs/2407.08608 |
| "GQA" (Ainslie et al., 2023) | Grouped-query attention for KV cache reduction | https://arxiv.org/abs/2305.13245 |
| "DeepSeek-V2" (DeepSeek-AI, 2024) | Multi-Head Latent Attention, MoE architecture | https://arxiv.org/abs/2405.04434 |
| "ZeRO" (Rajbhandari et al., 2020) | Sharded optimizer for large model training | https://arxiv.org/abs/1910.02054 |
| "Megatron-LM" (Narayanan et al., 2021) | 3D parallelism: TP + PP + DP | https://arxiv.org/abs/2104.04473 |
| "PagedAttention / vLLM" (Kwon et al., 2023) | Virtual-memory KV cache paging for serving | https://arxiv.org/abs/2309.06180 |
| "Speculative Decoding" (Leviathan et al., 2023) | Draft-then-verify for 2–3× decode speedup | https://arxiv.org/abs/2211.17192 |
| "Accelerating LLM Decoding" (Chen et al., 2023) | Speculative sampling with Chinchilla | https://arxiv.org/abs/2302.01318 |
| "LLM.int8()" (Dettmers et al., 2022) | INT8 quantization with mixed-precision outlier handling | https://arxiv.org/abs/2208.07339 |
| "GPTQ" (Frantar et al., 2022) | One-shot 4-bit weight quantization | https://arxiv.org/abs/2210.17323 |
| "AWQ" (Lin et al., 2023) | Activation-aware 4-bit weight quantization | https://arxiv.org/abs/2306.00978 |
| "Roofline model" (Williams et al., 2009) | Visual performance model for memory-bound vs compute-bound | https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf |
| "Efficient Diffusion Models: A Survey" (TMLR 2025) | Taxonomy of diffusion model acceleration techniques | https://arxiv.org/abs/2502.06805 |
| "DiT" (Peebles & Xie, 2023) | Diffusion Transformer replacing U-Net backbone | https://arxiv.org/abs/2212.09748 |
| "Mamba" (Gu & Dao, 2023) | Selective state space model as linear attention alternative | https://arxiv.org/abs/2312.00752 |
| "Transformers are RNNs" (Katharopoulos et al., 2020) | Linear attention via kernel feature map | https://arxiv.org/abs/2006.16236 |
| Transformer Math 101 (EleutherAI Blog) | Memory, FLOP, and throughput arithmetic for transformers | https://blog.eleuther.ai/transformer-math/ |
| "Making Deep Learning Go Brrrr" (Horace He) | First-principles GPU performance bottleneck analysis | https://horace.io/brrr_intro.html |
| FlashInfer (Ye et al., 2025) | Customizable attention engine for diverse serving scenarios | https://arxiv.org/abs/2501.01005 |
| SGLang paper (Zheng et al., 2023) | RadixAttention and structured generation serving system | https://arxiv.org/abs/2312.07104 |
