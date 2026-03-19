# ML Performance Engineering: Exercises

---

## Section 1: Mathematical Development

*Problems 1–17 develop the formal machinery underlying GPU performance analysis, attention complexity, distributed memory accounting, and quantization.*

---

**Problem 1.** *This problem establishes the roofline bound as a formal inequality and derives the ridge point that separates compute-bound from memory-bound regimes.*

Let $P$ (FLOP/s) denote peak compute throughput, $B$ (bytes/s) peak memory bandwidth, and $I = F/M$ arithmetic intensity (FLOP per byte moved), where $F$ is total floating-point operations and $M$ is total bytes read/written from/to HBM. Show that the achievable performance $\hat{P}$ satisfies $\hat{P} \leq \min(P, B \cdot I)$, and define the *ridge point* $I^*$ such that a kernel is memory-bandwidth bound iff $I < I^*$.

> **Prerequisites:** [[#3.3 The Roofline Model|§3.3 The Roofline Model]]

---

**Problem 2.** *This problem derives the arithmetic intensity of a matrix multiplication and explains why large GEMMs are compute-bound on modern GPUs.*

Consider the multiplication $C = AB$ where $A \in \mathbb{R}^{M \times K}$ and $B \in \mathbb{R}^{K \times N}$, with all matrices stored in BF16 (2 bytes per element). Count the total FLOPs and total bytes read/written from HBM, and express the arithmetic intensity $I$ as a function of $M$, $K$, $N$. Under what condition on $M$, $K$, $N$ does the kernel become compute-bound on an H100 (ridge point $I^* \approx 295$ FLOP/byte)?

> **Prerequisites:** [[#3.2 Memory Hierarchy|§3.2 Memory Hierarchy]], [[#3.3 The Roofline Model|§3.3 The Roofline Model]]

---

**Problem 3.** *This problem quantifies the $O(N^2)$ memory cost of standard attention and shows why long-context attention is memory-bandwidth bound.*

For multi-head attention with $H$ heads, head dimension $d_h$, and sequence length $N$, stored in BF16:
- (a) Compute the number of bytes required to store the full attention matrix $A = \text{softmax}(QK^\top / \sqrt{d_h}) \in \mathbb{R}^{H \times N \times N}$.
- (b) Compute the arithmetic intensity of a standard attention forward pass (FLOPs divided by HBM bytes), keeping only leading-order terms.
- (c) Argue that for $N \gg d_h$, standard attention is strongly memory-bandwidth bound.

> **Prerequisites:** [[#6.1 Standard Multi-Head Attention|§6.1 Standard Multi-Head Attention]], [[#3.3 The Roofline Model|§3.3 The Roofline Model]]

---

**Problem 4.** *This problem formalizes the online softmax trick that enables FlashAttention to avoid materializing the attention matrix in HBM.*

Let $x_1, \ldots, x_n \in \mathbb{R}$ be a sequence of scalars. Define:
$$m_i = \max(m_{i-1}, x_i), \quad d_i = d_{i-1} e^{m_{i-1} - m_i} + e^{x_i - m_i}, \quad \text{with } m_0 = -\infty,\ d_0 = 0.$$
Show that at step $n$: $m_n = \max_i x_i$ and $d_n = \sum_{i=1}^n e^{x_i - m_n}$, so that $\text{softmax}(x_i) = e^{x_i - m_n} / d_n$ can be recovered from $(m_n, d_n)$ alone. Explain how this recurrence allows processing the sequence in tiles without storing all $x_i$ simultaneously.

> **Prerequisites:** [[#6.4 FlashAttention|§6.4 FlashAttention]]

---

**Problem 5.** *This problem derives the memory savings of FlashAttention relative to standard attention and establishes its $O(N)$ peak SRAM requirement.*

Let $B_c$ and $B_r$ denote the column-block and row-block sizes used in FlashAttention. Assume SRAM capacity is $M_{\text{SRAM}}$ bytes.
- (a) Express the SRAM required per tile iteration as a function of $B_c$, $B_r$, $d$, and bytes-per-element $b$.
- (b) Show that FlashAttention does not materialize the $N \times N$ attention matrix, replacing $O(N^2)$ HBM storage with $O(N d)$.
- (c) Count the number of HBM reads/writes in standard attention vs. FlashAttention, and express the ratio to leading order.

> **Prerequisites:** [[#6.4 FlashAttention|§6.4 FlashAttention]], [[#3.2 Memory Hierarchy|§3.2 Memory Hierarchy]]

---

**Problem 6.** *This problem establishes that GQA reduces KV cache memory by exactly $H/G$ and derives the tradeoff between cache size and model expressiveness.*

In grouped-query attention with $H$ query heads, $G$ KV head groups ($G \leq H$), head dimension $d_h$, sequence length $N$, $L$ layers, and $b$ bytes per element:
- (a) Write down the KV cache size for MHA ($G = H$), GQA (general $G$), and MQA ($G = 1$).
- (b) For a 70B model with $H = 64$, $d_h = 128$, $L = 80$, $b = 2$, $N = 8192$, compute the KV cache in GB for MHA and GQA with $G = 8$.
- (c) Argue that for a fixed memory budget $M_{\text{KV}}$, GQA allows serving $H/G$ times more concurrent requests than MHA.

> **Prerequisites:** [[#6.2 Multi-Query and Grouped-Query Attention|§6.2 Multi-Query and Grouped-Query Attention]], [[#7.1 KV Cache|§7.1 KV Cache]]

---

**Problem 7.** *This problem analyzes the memory savings of Multi-Head Latent Attention (MLA) by comparing its compressed KV representation to MHA.*

In MLA, instead of caching $K, V \in \mathbb{R}^{N \times d_h \cdot H}$, a single compressed latent $c^{KV} \in \mathbb{R}^{N \times d_c}$ is cached, with $d_c \ll d_h \cdot H$.
- (a) Express the compression ratio $\rho = \text{MHA KV size} / \text{MLA KV size}$ in terms of $d_c$, $d_h$, $H$.
- (b) For DeepSeek-V2 values $d_c = 512$, $d_h = 128$, $H = 128$, compute $\rho$.
- (c) At inference time, MLA must project $c^{KV}$ up to full $K, V$ for the attention computation. Show that this adds $O(N \cdot d_c \cdot d_h \cdot H)$ FLOPs but does not increase HBM traffic relative to caching the full $K, V$.

> **Prerequisites:** [[#6.3 Multi-Head Latent Attention|§6.3 Multi-Head Latent Attention]]

---

**Problem 8.** *This problem derives the communication volume of ring all-reduce and shows how it scales with the number of workers.*

In ring all-reduce with $N_w$ workers, each holding a gradient tensor of $p$ elements (floats), the reduce-scatter phase sends and receives $p (N_w - 1)/N_w$ elements per worker, and the all-gather phase sends and receives the same. Show that the total communication volume per worker is $2 p (N_w - 1)/N_w$ elements, and that this approaches $2p$ as $N_w \to \infty$ regardless of $N_w$. Contrast with the naive all-reduce (send to a single parameter server), which has communication volume $O(N_w \cdot p)$.

> **Prerequisites:** [[#4.1 Data Parallelism|§4.1 Data Parallelism]]

---

**Problem 9.** *This problem formalizes ZeRO memory partitioning and computes the per-rank memory reduction as a function of ZeRO stage.*

Consider training a model with $\Psi$ parameters in mixed precision (BF16 parameters, FP32 master weights and Adam optimizer states). With $N_w$ DP workers:
- (a) Baseline: compute total memory per rank (parameters, gradients, optimizer states) without sharding.
- (b) ZeRO-1: only optimizer states are sharded. Compute per-rank memory.
- (c) ZeRO-2: optimizer states and gradients are sharded. Compute per-rank memory.
- (d) ZeRO-3: optimizer states, gradients, and parameters are sharded. Compute per-rank memory.
- (e) For a 7B parameter model ($\Psi = 7 \times 10^9$) with $N_w = 64$, give the per-rank memory in GB for each stage.

> **Prerequisites:** [[#4.4 ZeRO and FSDP|§4.4 ZeRO and FSDP]]

---

**Problem 10.** *This problem analyzes the pipeline bubble in GPipe and shows how interleaved scheduling reduces idle time.*

In GPipe with $p$ pipeline stages and $m$ micro-batches:
- (a) Derive the fraction of total time spent in the pipeline bubble (idle stages) as $f_{\text{bubble}} = (p - 1)/(p + m - 1)$.
- (b) Show that $f_{\text{bubble}} \to 0$ as $m \to \infty$ for fixed $p$.
- (c) In the interleaved 1F1B schedule, the bubble fraction becomes $(p - 1)/(m \cdot v)$ where $v$ is the number of pipeline chunks per stage. For $p = 8$, $m = 32$, $v = 2$, compare the bubble fractions of GPipe and 1F1B.

> **Prerequisites:** [[#4.3 Pipeline Parallelism|§4.3 Pipeline Parallelism]]

---

**Problem 11.** *This problem derives the gradient checkpointing memory-compute tradeoff and identifies the optimal checkpoint placement for a linear chain of $n$ layers.*

Without checkpointing, the peak activation memory during backpropagation of a depth-$n$ network is $O(n a)$ where $a$ is the activation memory per layer. With checkpointing every $k$ layers, the peak activation memory is $O(k a + n/k \cdot a)$.
- (a) Minimize over $k$ to show the optimal checkpoint interval is $k^* = \sqrt{n}$, giving $O(\sqrt{n} \cdot a)$ peak memory.
- (b) Express the additional forward-pass cost of checkpointing as a fraction of total forward+backward FLOPs. What is the asymptotic overhead?
- (c) In PyTorch, `torch.utils.checkpoint.checkpoint` reruns the forward pass in the backward. Identify the conditions under which this incurs more than the expected $1\times$ forward-pass overhead.

> **Prerequisites:** [[#5.2 Gradient Checkpointing|§5.2 Gradient Checkpointing]]

---

**Problem 12.** *This problem analyzes the quantization error introduced by uniform INT8 quantization and derives the signal-to-noise ratio as a function of dynamic range.*

Let $x \in [-R, R]$ be a scalar weight. Uniform INT8 quantization maps $x$ to $\hat{x} = \text{round}(x / s) \cdot s$ where $s = R / 127$. The quantization error is $\epsilon = x - \hat{x}$.
- (a) Under the assumption that $\epsilon$ is uniformly distributed on $[-s/2, s/2]$, compute $\mathbb{E}[\epsilon^2]$ (quantization noise power).
- (b) If $x \sim \mathcal{N}(0, \sigma^2)$, compute the signal power $\mathbb{E}[x^2]$ and the SQNR (signal-to-quantization-noise ratio) in dB as a function of $R/\sigma$.
- (c) Explain why outlier activations (large $R$) degrade SQNR for INT8, motivating the mixed-precision approach of LLM.int8().

> **Prerequisites:** [[#7.2 Quantization|§7.2 Quantization]]

---

**Problem 13.** *This problem formalizes the speculative decoding acceptance probability and derives the expected number of accepted tokens per iteration.*

Let $q(x)$ denote the draft model's distribution and $p(x)$ the target model's distribution over the vocabulary $\mathcal{V}$. In speculative decoding, a draft token $x$ is accepted with probability $\alpha(x) = \min(1, p(x)/q(x))$, and on rejection a corrected token is sampled from the residual $p'(x) = \text{normalize}(\max(0, p(x) - q(x)))$.
- (a) Show that the token distribution after accept-or-correct exactly matches $p(x)$.
- (b) Compute the expected acceptance probability $\bar{\alpha} = \sum_x q(x) \min(1, p(x)/q(x))$ and express it in terms of the total variation distance $d_{\text{TV}}(p, q) = \frac{1}{2}\sum_x |p(x) - q(x)|$.
- (c) For a draft length of $k$ tokens (each accepted independently with probability $\bar{\alpha}$), derive the expected number of accepted tokens $\mathbb{E}[\tau]$.

> **Prerequisites:** [[#7.3 Speculative Decoding|§7.3 Speculative Decoding]]

---

**Problem 14.** *This problem derives the arithmetic intensity of autoregressive decoding and explains why it is deeply memory-bandwidth bound for small batch sizes.*

During the decode phase of an LLM with $L$ layers, hidden dimension $d$, and batch size $B$:
- (a) Count the FLOPs for a single linear layer $y = xW$ where $x \in \mathbb{R}^{B \times d}$ and $W \in \mathbb{R}^{d \times d}$.
- (b) Count the bytes read from HBM (weight matrix $W$ and input $x$, ignoring output).
- (c) Show that as $B \to 1$ (single-request decoding), the arithmetic intensity approaches $I \approx 1$ FLOP/byte, making decoding maximally memory-bandwidth bound.
- (d) Derive the minimum batch size $B_{\min}$ for which the attention projection layers become compute-bound on H100.

> **Prerequisites:** [[#3.3 The Roofline Model|§3.3 The Roofline Model]], [[#7.1 KV Cache|§7.1 KV Cache]]

---

**Problem 15.** *This problem formalizes the communication-computation overlap in tensor parallelism and derives the conditions under which communication can be fully hidden.*

In tensor parallelism with $t$ devices, each forward pass through an MLP block requires one all-reduce of size $B \cdot d$ elements (where $B$ is batch size, $d$ is hidden dim). Let $T_{\text{comm}}$ and $T_{\text{comp}}$ denote communication and computation times.
- (a) Express $T_{\text{comm}}$ in terms of $B$, $d$, $b$ (bytes/element), and inter-device bandwidth $\Lambda$.
- (b) Express $T_{\text{comp}}$ for the local GEMM (each device holds $d/t$ columns of the weight matrix).
- (c) Derive the condition on $t$, $B$, $d$, $\Lambda$, and device FLOP/s $P$ such that communication is fully hidden by computation.
- (d) For H100 NVLink ($\Lambda = 900$ GB/s), $P = 989$ TFLOP/s (BF16), $d = 8192$, $B = 1024$, what is the maximum $t$ for which communication remains hidden?

> **Prerequisites:** [[#4.2 Tensor Parallelism|§4.2 Tensor Parallelism]]

---

**Problem 16.** *This problem analyzes the memory waste in PagedAttention and proves the near-optimal bound of under 4% fragmentation.*

In PagedAttention, the KV cache is divided into pages of size $P_{\text{size}}$ tokens each. Each sequence is allocated pages on demand; the last page of each sequence is at most $P_{\text{size}} - 1$ tokens underutilized.
- (a) Let $N_{\text{seq}}$ be the number of concurrent sequences and $L_i$ the length of sequence $i$. Express total allocated KV cache as a function of $P_{\text{size}}$ and $\{L_i\}$.
- (b) Express the memory waste (allocated but unused) as a fraction of total allocated memory. Show this is at most $(P_{\text{size}} - 1) / \mathbb{E}[L_i]$.
- (c) For $P_{\text{size}} = 16$ tokens and $\mathbb{E}[L_i] = 512$ tokens, compute the maximum fragmentation fraction.

> **Prerequisites:** [[#7.4 Continuous Batching and PagedAttention|§7.4 Continuous Batching and PagedAttention]]

---

**Problem 17.** *This problem derives the FLOP and memory complexity of diffusion model inference and shows how DDIM reduces inference cost relative to DDPM.*

A diffusion model with a DiT backbone processes images of resolution $H \times W$ with latent spatial dimension $h \times w = (H/8) \times (W/8)$ and $C$ channels.
- (a) Express the FLOPs per denoising step as $F_{\text{step}}$ (treat the DiT as a transformer with sequence length $n = h \cdot w / p^2$ for patch size $p$).
- (b) For DDPM with $T = 1000$ steps vs. DDIM with $T' = 50$ steps, express the total inference FLOPs in terms of $F_{\text{step}}$, $T$, $T'$.
- (c) Consistency models reduce inference to $T'' = 1$–$4$ steps. What is the FLOP reduction relative to DDPM and DDIM?

> **Prerequisites:** [[#9. Diffusion Model Efficiency|§9. Diffusion Model Efficiency]]

---

## Section 2: Algorithmic Applications

*Problems 18–22 focus on implementing and analyzing the algorithms in pseudocode, profiling, complexity, and practical engineering trade-offs.*

---

**Problem 18.** *This problem asks the student to write a tiled matrix multiplication in pseudocode that achieves compute-bound performance by maximizing SRAM reuse.*

> **Prerequisites:** [[#3.4 CUDA Programming|§3.4 CUDA Programming]], [[#3.2 Memory Hierarchy|§3.2 Memory Hierarchy]]

Write a pseudocode CUDA kernel for tiled matrix multiplication $C = AB$ with $A \in \mathbb{R}^{M \times K}$, $B \in \mathbb{R}^{K \times N}$. Each thread block of size $T \times T$ should:
1. Load a $T \times T$ tile of $A$ and $B$ into shared memory.
2. Synchronize, compute the partial dot products, and accumulate into registers.
3. Iterate over the $K$ dimension in tiles of size $T$.

Analyze: (a) the arithmetic intensity of this kernel as a function of $T$; (b) the minimum tile size $T_{\min}$ required to reach the ridge point on H100.

---

**Problem 19.** *This problem develops a profiling methodology for identifying whether a transformer training run is compute-bound, memory-bandwidth bound, or overhead-bound.*

> **Prerequisites:** [[#3.3 The Roofline Model|§3.3 The Roofline Model]], [[#5.3 Compiler Optimizations|§5.3 Compiler Optimizations]]

Given a transformer model being trained with `torch.compile`, describe:
1. How to use `torch.profiler` and NVIDIA Nsight Compute to measure achieved FLOP/s and HBM bandwidth utilization per kernel.
2. How to construct a roofline plot for each kernel category (GEMM, attention, elementwise, communication).
3. A decision tree for diagnosing the bottleneck: if GEMM kernels are below ridge point, what are the likely causes? If elementwise kernels have low HBM BW utilization, what does that indicate?
4. How `torch.compile` (kernel fusion) changes the roofline position of elementwise op chains.

---

**Problem 20.** *This problem asks the student to implement FlashAttention-style blocked attention in Triton pseudocode and analyze the HBM IO complexity.*

> **Prerequisites:** [[#6.4 FlashAttention|§6.4 FlashAttention]], [[#3.5 Triton|§3.5 Triton]]

Write Triton pseudocode for a single-headed causal FlashAttention forward kernel. The kernel should:
1. Loop over row-blocks of $Q$ (outer loop) and column-blocks of $K$, $V$ (inner loop).
2. Maintain running $(m_i, \ell_i, O_i)$ statistics (online softmax max, normalizer, and output accumulator) and update them at each inner step.
3. Apply the causal mask by skipping column blocks entirely past the current row-block diagonal.

Count the number of HBM reads and writes as a function of $N$, $d$, $B_r$, $B_c$, and verify it is $O(N d)$.

---

**Problem 21.** *This problem develops the ZeRO-3 communication schedule and analyzes the memory and communication overhead for a two-layer transformer.*

> **Prerequisites:** [[#4.4 ZeRO and FSDP|§4.4 ZeRO and FSDP]]

Consider a two-layer transformer (attention + FFN) trained with ZeRO-3 across $N_w = 8$ workers. For each of the following operations, specify:
(a) What collective is performed, which parameters are gathered, and in what order.
(b) The peak per-worker memory during that operation.
(c) The communication volume.

Operations: (1) forward pass layer 1 (attention), (2) forward pass layer 2 (FFN), (3) backward pass layer 2, (4) backward pass layer 1, (5) optimizer step.

---

**Problem 22.** *This problem asks the student to sketch a continuous batching scheduler and analyze its throughput advantage over static batching under a Poisson arrival model.*

> **Prerequisites:** [[#7.4 Continuous Batching and PagedAttention|§7.4 Continuous Batching and PagedAttention]], [[#8. Serving Systems|§8. Serving Systems]]

Consider an LLM serving system with a single GPU and a Poisson request arrival process with rate $\lambda$ requests/second. Each request generates a random number of tokens $L \sim \text{Geom}(1/\mu)$ with mean $\mu$.
1. In static batching (batch size $B$, wait until $B$ requests complete before starting new batch), what is the expected GPU idle time per batch if $L$ is heterogeneous?
2. In continuous batching, a request slot is freed immediately upon completion. Sketch the scheduler loop in pseudocode (iteration-level scheduling).
3. Under what condition (relationship between $\lambda$, $\mu$, and GPU decode throughput $\tau$ tokens/s) does continuous batching achieve near-100% GPU utilization?
