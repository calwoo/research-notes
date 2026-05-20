# Memory-Bound Inference

## Table of Contents

- [[#1. Why Inference is Memory-Bound|1. Why Inference is Memory-Bound]]
  - [[#1.1 Autoregressive Decoding as a Stream of Matrix-Vector Products|1.1 Autoregressive Decoding as a Stream of Matrix-Vector Products]]
  - [[#1.2 Formal Derivation of Arithmetic Intensity for a Linear Layer|1.2 Formal Derivation of Arithmetic Intensity for a Linear Layer]]
  - [[#1.3 Contrast with Training and Prefill|1.3 Contrast with Training and Prefill]]
- [[#2. The Bandwidth-Scaling Gap|2. The Bandwidth-Scaling Gap]]
  - [[#2.1 Ridge Point Trend Across GPU Generations|2.1 Ridge Point Trend Across GPU Generations]]
  - [[#2.2 Consequences for Inference Workloads|2.2 Consequences for Inference Workloads]]
- [[#3. KV-Cache Bandwidth Math|3. KV-Cache Bandwidth Math]]
  - [[#3.1 KV-Cache Size|3.1 KV-Cache Size]]
  - [[#3.2 Per-Step Bandwidth Cost|3.2 Per-Step Bandwidth Cost]]
  - [[#3.3 Combined Arithmetic Intensity with KV Cache|3.3 Combined Arithmetic Intensity with KV Cache]]
- [[#4. Diagnosing the Regime: Practical Heuristics|4. Diagnosing the Regime: Practical Heuristics]]
  - [[#4.1 Theoretical Classification|4.1 Theoretical Classification]]
  - [[#4.2 Empirical Signals Without Profiling|4.2 Empirical Signals Without Profiling]]
  - [[#4.3 Profiling Tools|4.3 Profiling Tools]]
  - [[#4.4 Operation-Level Rules of Thumb|4.4 Operation-Level Rules of Thumb]]
- [[#5. Implications for Hardware Choice|5. Implications for Hardware Choice]]
- [[#References|References]]

---

> [!INFO] Companion note
> This note builds directly on the roofline model formalism — arithmetic intensity $I = F/B$, ridge point $I^* = \Pi/\beta$, and per-GPU $I^*$ values — established in [[concepts/deep-learning-engineering/nvidia-gpu-hardware#7. The Roofline Model|Roofline Model]]. Those fundamentals are not re-derived here. The roofline section of that note also introduces the *bandwidth-scaling gap* macro-trend; this note develops that thread in full.

---

## 1. Why Inference is Memory-Bound 🧠

### 1.1 Autoregressive Decoding as a Stream of Matrix-Vector Products

*Autoregressive decoding* generates tokens one at a time: at each step $t$, the model takes a single new token embedding $x_t \in \mathbb{R}^d$ and produces the next hidden state by passing $x_t$ through all transformer layers. Crucially, at each layer, every weight matrix is multiplied against a single vector — this is a *matrix-vector product* (GEMV), not a matrix-matrix product (GEMM).

The asymmetry is decisive. A GEMM of two $d \times d$ matrices requires $2d^3$ FLOPs and transfers $O(d^2)$ bytes, yielding arithmetic intensity $I \sim d/3$, which scales to hundreds of FLOP/byte for large $d$. A GEMV of a $d \times d$ matrix against a vector requires $2d^2$ FLOPs and transfers $2d^2 \cdot 2$ bytes (loading the matrix in BF16), yielding:

$$I_{\text{GEMV}} = \frac{2d^2}{2 \cdot 2d^2} = \frac{1}{2} \approx 1 \text{ FLOP/byte}$$

This is the core observation: **at batch size 1, every weight matrix is loaded from HBM exactly once per token, and only one FLOP of compute is performed per byte transferred.** The GPU's Tensor Cores sit idle; the bottleneck is the HBM bus.

### 1.2 Formal Derivation of Arithmetic Intensity for a Linear Layer

**Definition (Linear Layer Arithmetic Intensity).** Consider a single linear layer $y = Wx$ with $W \in \mathbb{R}^{M \times K}$ and a batch of $B$ tokens, so $x \in \mathbb{R}^{K \times B}$ and $y \in \mathbb{R}^{M \times B}$. In BF16 (2 bytes per element), the data-movement and compute costs are:

$$F(B) = 2MKB \quad \text{(FLOPs: one mul + one add per inner product entry)}$$

$$\mathcal{B}(B) = 2 \underbrace{MK}_{\text{weights}} \cdot 2 + 2 \underbrace{KB}_{\text{input}} \cdot 2 + 2 \underbrace{MB}_{\text{output}} \cdot 2 \quad \text{(bytes, BF16)}$$

$$= 4MK + 4KB + 4MB$$

The *arithmetic intensity* as a function of batch size is:

$$I(B) = \frac{F(B)}{\mathcal{B}(B)} = \frac{2MKB}{4MK + 4KB + 4MB} = \frac{MKB}{2MK + 2KB + 2MB}$$

For large $M, K \gg B$ (the typical regime: dimension $d \sim 4096$–$8192$, batch $B \leq 32$), the weight term $2MK$ dominates the denominator:

$$I(B) \approx \frac{MKB}{2MK} = \frac{B}{2} \quad \text{FLOP/byte}$$

> [!NOTE] The $B/2$ approximation
> The approximation $I(B) \approx B/2$ holds when the weight matrix bytes dominate activation bytes, i.e., when $2MK \gg 2KB + 2MB$, which simplifies to $M \gg B$ and $K \gg B$. For $M = K = 4096$ and $B \leq 512$, the error is less than 20%. For the purposes of order-of-magnitude reasoning, write $I(B) \approx B$.

**Consequence.** The layer is *memory-bound* whenever $I(B) < I^*$, i.e., whenever $B \lesssim I^* / 2$. For the H100 SXM ($I^* = 591$ FLOP/byte), this means batch sizes up to $B \approx 300$ keep the layer memory-bound. For the A100 ($I^* = 156$), the crossover is around $B \approx 78$.

### 1.3 Contrast with Training and Prefill

During *training* and *prefill* (processing a full input prompt of length $S$), the input is a matrix $x \in \mathbb{R}^{K \times S}$ rather than a vector. The GEMM $Wx$ with $W \in \mathbb{R}^{M \times K}$ has arithmetic intensity:

$$I_{\text{prefill}}(S) \approx \frac{S}{2} \quad \text{FLOP/byte (same formula, with } S \text{ replacing } B\text{)}$$

A prompt of length $S = 2048$ yields $I \approx 1024$ FLOP/byte — above the ridge point of any current GPU. **Prefill is compute-bound; decode is memory-bound.** This is why optimizing decode throughput requires maximally batching requests, while prefill optimization focuses on kernel efficiency (FlashAttention, etc.).

> [!EXAMPLE] A 7B-parameter model at batch size 1
> A LLaMA-2-7B model has roughly 7 billion parameters. In BF16, total weight bytes = $7 \times 10^9 \times 2 = 14$ GB. At each decode step, all 14 GB must be loaded from HBM. FLOPs per step: $\approx 14 \times 10^9$ (each weight is used in one multiply-add). Arithmetic intensity: $I \approx 14\text{ GFLOPs} / 14\text{ GB} = 1$ FLOP/byte. On an H100 SXM (3.35 TB/s), the minimum time per token is $14\text{ GB} / 3350\text{ GB/s} \approx 4.2\ \text{ms}$, corresponding to $\approx 238$ tokens/second. The H100's 1,979 TFLOPS of FP16 compute is almost entirely unused.

---

> [!QUESTION] Exercise 1: Batch-Size Crossover on A100 vs H100
> *This problem derives the batch sizes at which autoregressive decode crosses from memory-bound to compute-bound on two different GPUs, illustrating how the bandwidth-scaling gap raises the crossover threshold.*
>
> > **Prerequisites:** [[#1.2 Formal Derivation of Arithmetic Intensity for a Linear Layer|1.2 Formal Derivation of Arithmetic Intensity for a Linear Layer]]
>
> Consider a linear layer $W \in \mathbb{R}^{4096 \times 4096}$ in BF16. Using the approximation $I(B) \approx B/2$:
>
> (a) At what batch size $B_{\text{A100}}$ does the layer transition from memory-bound to compute-bound on an A100 SXM ($\Pi = 312$ TFLOPS, $\beta = 2.0$ TB/s)?
>
> (b) At what batch size $B_{\text{H100}}$ does the same layer transition on an H100 SXM ($\Pi = 1{,}979$ TFLOPS, $\beta = 3.35$ TB/s)?
>
> (c) Both GPUs doubled their FLOPS-per-dollar versus the previous generation, but the H100 requires roughly $4\times$ larger batches to become compute-bound. What does this imply for a latency-sensitive deployment serving requests one at a time?

> [!TIP]- Solution to Exercise 1
> **Key insight:** The H100's 6× FLOP increase outpaced its 1.7× bandwidth increase, so its ridge point is ~3.8× higher. A deployment constrained to $B = 1$ (latency-sensitive) gains almost nothing in throughput-per-request from the H100's extra Tensor Cores.
>
> **Sketch:**
>
> (a) $I^*_{\text{A100}} = 312 \times 10^{12} / 2 \times 10^{12} = 156$ FLOP/byte. Crossover: $B/2 = 156 \Rightarrow B_{\text{A100}} \approx 312$.
>
> (b) $I^*_{\text{H100}} = 1979 \times 10^{12} / 3.35 \times 10^{12} \approx 591$ FLOP/byte. Crossover: $B/2 = 591 \Rightarrow B_{\text{H100}} \approx 1182$.
>
> (c) At $B = 1$, the attainable throughput is $\beta \cdot I = \beta \cdot 0.5$ on both GPUs — limited entirely by bandwidth. The H100 has 1.675× more bandwidth than the A100, so decode speed at $B = 1$ improves by only $\approx 1.675\times$, not the $\sim 6\times$ TFLOPS ratio. An H100 is ~$3\times$ more expensive than an A100 per GPU-hour; at $B=1$, you pay a 3× premium for a 1.7× speedup — bandwidth per dollar is the right metric for this workload.

---

## 2. The Bandwidth-Scaling Gap 📈

### 2.1 Ridge Point Trend Across GPU Generations

The *ridge point* $I^* = \Pi / \beta$ has risen monotonically across every data-center GPU generation because compute throughput (TFLOPS) has grown faster than HBM bandwidth (TB/s). Quantitatively:

- **Compute (FP16 TFLOPS):** roughly 3–4× per generation (2 years).
- **HBM bandwidth (TB/s):** roughly 1.5–1.8× per generation.

The ratio $I^* = \Pi / \beta$ therefore grows by a factor of roughly $3/1.6 \approx 2$ per generation.

The table below tracks this trend from Pascal (the first HBM data-center GPU) through Blackwell:

| GPU | Architecture | FP16 TFLOPS ($\Pi$) | HBM BW ($\beta$, TB/s) | $I^*$ (FLOP/byte) |
|-----|-------------|---------------------|------------------------|-------------------|
| P100 SXM | Pascal (2016) | 21.2 | 0.72 | 29 |
| V100 SXM2 | Volta (2017) | 125 | 0.90 | 139 |
| A100 SXM | Ampere (2020) | 312 | 2.00 | 156 |
| H100 SXM | Hopper (2022) | 1,979 | 3.35 | 591 |
| H200 SXM | Hopper (2024) | 1,979 | 4.80 | 412 |
| B200 SXM | Blackwell (2024) | 4,500 | 8.00 | 563 |

> [!NOTE] H200 anomaly
> The H200 *lowers* $I^*$ relative to the H100 by pairing the same GH100 die with 141 GB of faster HBM3e (4.8 TB/s vs. 3.35 TB/s) while leaving compute unchanged. This was a deliberate product decision: at the time, inference-at-scale was bandwidth-starved, and the incremental cost of HBM3e was lower than a full die respin. The H200 is fundamentally an inference-optimized variant of the H100.

> [!NOTE] B200 FP16 figure
> The B200 figure of 4,500 TFLOPS FP16 uses the dense (non-sparsity) rating. With 2:4 structured sparsity the figure doubles. $I^*$ is computed using the dense value, which is the conservative baseline for kernels that have not been sparsified.

### 2.2 Consequences for Inference Workloads

The rising $I^*$ means **the fraction of inference workloads that are memory-bound increases with each GPU generation.** A workload that was compute-bound on V100 may be memory-bound on H100.

Concretely: a batch of $B = 64$ tokens has $I(64) \approx 32$ FLOP/byte, which is:
- Compute-bound on P100 ($I^* = 29$): barely above ridge point.
- Memory-bound on V100 ($I^* = 139$): comfortably below.
- Deeply memory-bound on H100 ($I^* = 591$): 18× below ridge point.

*Surprisingly,* GPUs are becoming arithmetically richer relative to their bandwidth — a property that accelerates training (which exploits TFLOPS at large batch) but actively hurts inference efficiency (which is already bandwidth-starved and benefits only from bandwidth improvements). **For inference at low batch size, the figure of merit is GB/s, not TFLOPS.**

---

> [!QUESTION] Exercise 2: Effective Utilization at Batch Size 1
> *This problem quantifies how wasteful high-TFLOPS GPUs are for single-request inference.*
>
> > **Prerequisites:** [[#2.1 Ridge Point Trend Across GPU Generations|2.1 Ridge Point Trend Across GPU Generations]]
>
> For each GPU in the table above, compute the fraction of peak TFLOPS actually utilized during decode at $B = 1$, where $I(1) \approx 0.5$ FLOP/byte. Express this as a percentage. What does the trend across generations imply?

> [!TIP]- Solution to Exercise 2
> **Key insight:** As $I^*$ rises, the fraction of compute utilized at fixed $I$ decreases proportionally. H100 utilizes less than 0.1% of its peak TFLOPS during single-request decode.
>
> **Sketch:**
>
> The attainable throughput when memory-bound is $P = \beta \cdot I$. The fraction of peak compute utilized is:
>
> $$\frac{P}{\Pi} = \frac{\beta \cdot I}{\Pi} = \frac{I}{I^*}$$
>
> At $I = 0.5$ FLOP/byte:
>
> | GPU | $I^*$ | Compute utilization at $I = 0.5$ |
> |-----|-------|----------------------------------|
> | P100 SXM | 29 | $0.5/29 \approx 1.7\%$ |
> | V100 SXM2 | 139 | $0.5/139 \approx 0.36\%$ |
> | A100 SXM | 156 | $0.5/156 \approx 0.32\%$ |
> | H100 SXM | 591 | $0.5/591 \approx 0.08\%$ |
> | H200 SXM | 412 | $0.5/412 \approx 0.12\%$ |
>
> The trend is unambiguous: as GPU generations optimize for compute, single-request inference utilizes an ever-smaller fraction of that compute. An H100 serving one request at a time is operating at <0.1% compute efficiency. It is effectively a very expensive memory-bandwidth device.

---

## 3. KV-Cache Bandwidth Math 🗄️

### 3.1 KV-Cache Size

The *KV cache* stores the key and value tensors for all past tokens so that attention at position $t$ does not recompute them. For a transformer with:
- $L$ layers
- $h$ attention heads
- Head dimension $d_h$ (so $d_{\text{model}} = h \cdot d_h$)
- Current context length $S$ tokens
- Stored in BF16 (2 bytes per element)

the total KV-cache memory is:

$$\text{KV}_{\text{bytes}}(S) = 2 \cdot 2 \cdot L \cdot h \cdot d_h \cdot S = 4 L h d_h S$$

The leading $2$ accounts for both keys and values; the second $2$ is bytes per BF16 element.

**Example.** For a LLaMA-3-70B model ($L = 80$, $h = 64$, $d_h = 128$, context $S = 4096$):

$$\text{KV}_{\text{bytes}}(4096) = 4 \times 80 \times 64 \times 128 \times 4096 = 4 \times 80 \times 33{,}554{,}432 \approx 10.7 \text{ GB}$$

At $S = 32768$ tokens the KV cache alone is $\approx 85.9$ GB — exceeding the H100's total 80 GB VRAM.

> [!WARNING] Multi-query and grouped-query attention
> With *grouped-query attention* (GQA) using $g$ KV groups ($g < h$), the KV cache size reduces by a factor of $h/g$. LLaMA-3-70B uses GQA with $g = 8$ groups, reducing the KV cache by $64/8 = 8\times$ to $\approx 1.3$ GB at $S = 4096$. The formula above holds; substitute $g$ for $h$ when GQA is used.

### 3.2 Per-Step Bandwidth Cost

At each decode step, the full KV cache must be loaded from HBM to compute attention for the current query. The bandwidth cost per decode step for attention alone is:

$$\mathcal{B}_{\text{kv}}(S) = 4 L h d_h S \quad \text{bytes}$$

This grows *linearly with context length*. At long contexts, KV-cache bandwidth can match or exceed weight-loading bandwidth.

**Example.** For the same 70B model with full-head attention ($h=64$) at $S = 4096$:
- Weight bytes: $70 \times 10^9 \times 2 = 140$ GB
- KV-cache bytes per step: $10.7$ GB (computed above)
- KV cache is $\approx 7.6\%$ of weight bandwidth at this context length

At $S = 32768$:
- KV-cache bytes: $\approx 85.9$ GB — now $\approx 61\%$ of weight bandwidth

### 3.3 Combined Arithmetic Intensity with KV Cache

Let $P$ be the total parameter count (so weight bytes $= 2P$ in BF16). At batch size $B$ and context length $S$, the combined arithmetic intensity for a decode step is:

$$I(B, S) = \frac{2P \cdot B}{2P \cdot 2 + \mathcal{B}_{\text{kv}}(S)} = \frac{2PB}{4P + 4LhdS}$$

where we write $4P$ for the weight bytes ($2P$ parameters $\times$ 2 bytes, loaded $\times$ 2 for the factor-of-2 I/O convention) and have simplified $4Lhd_hS$ as $4LhdS$.

For small $S$ (short context), the weight term dominates: $I(B, S) \approx B/2$ as before. For large $S$, the KV-cache term dominates:

$$I(B, S) \xrightarrow{S \to \infty} \frac{2PB}{4LhdS} = \frac{PB}{2LhdS}$$

This decreases with $S$: **long-context decoding becomes even more memory-bandwidth-limited than short-context decoding**, even as the compute cost (FLOPs for attention QK products) grows with $S$.

> [!NOTE] FLOPs from attention computation
> The attention QK product at decode time for a single query against $S$ cached keys requires $2 \cdot h \cdot d_h \cdot S$ FLOPs per layer, total $2LhdS$ FLOPs. Compared to weight FLOPs of $\sim 2P$ for the linear projections (at $B=1$), attention FLOPs become dominant only when $LhdS \gg P/2$. For a 70B model with $L=80$, $h=64$, $d_h=128$: the threshold is $S \gg P / (2Lhd_h) = 70 \times 10^9 / (2 \times 80 \times 64 \times 128) \approx 53{,}600$ tokens. Below this context length, weight-compute FLOPs dominate; above it, attention FLOPs dominate. In either case, $I$ remains far below current ridge points.

---

> [!QUESTION] Exercise 3: KV-Cache Arithmetic Intensity at Long Context
> *This problem shows how increasing context length degrades arithmetic intensity even further below the ridge point.*
>
> > **Prerequisites:** [[#3.3 Combined Arithmetic Intensity with KV Cache|3.3 Combined Arithmetic Intensity with KV Cache]]
>
> Consider a LLaMA-3-8B model: $P = 8 \times 10^9$ parameters, $L = 32$ layers, $h = 8$ KV heads (GQA), $d_h = 128$. The model runs in BF16.
>
> (a) Compute $I(B=1, S)$ for $S \in \{512, 4096, 32768\}$ using the formula $I(B,S) = 2PB / (4P + 4LhdS)$.
>
> (b) At what context length $S^*$ does the KV-cache bandwidth cost equal the weight-loading bandwidth cost?
>
> (c) How does each value of $I$ from (a) compare to $I^*_{\text{H100}} = 591$ FLOP/byte?

> [!TIP]- Solution to Exercise 3
> **Key insight:** For this 8B model with GQA ($h=8$), weight bytes dominate even at $S = 32768$. The crossover context length is far beyond practical deployment. Nevertheless, $I$ remains well below 1 FLOP/byte in all cases.
>
> **Sketch:**
>
> (a) $4P = 4 \times 8 \times 10^9 \times 1 = 32 \times 10^9$ bytes. $4Lhd_h = 4 \times 32 \times 8 \times 128 = 131{,}072$ bytes per token.
>
> - $S = 512$: $\mathcal{B}_{\text{kv}} = 131{,}072 \times 512 = 67.1\text{ MB}$; denominator $= 32\text{ GB} + 0.067\text{ GB} \approx 32.07\text{ GB}$; $I = 2 \times 8 \times 10^9 / 32.07 \times 10^9 \approx 0.499$.
> - $S = 4096$: $\mathcal{B}_{\text{kv}} = 131{,}072 \times 4096 = 537\text{ MB}$; denominator $\approx 32.54\text{ GB}$; $I \approx 0.491$.
> - $S = 32768$: $\mathcal{B}_{\text{kv}} = 131{,}072 \times 32768 = 4.29\text{ GB}$; denominator $\approx 36.29\text{ GB}$; $I \approx 0.441$.
>
> (b) Crossover at $4P = 4LhdS$: $S^* = P / (Lhd_h) = 8 \times 10^9 / (32 \times 8 \times 128) = 8 \times 10^9 / 32{,}768 \approx 244{,}140$ tokens. GQA dramatically extends the crossover beyond practical context lengths.
>
> (c) All values ($\approx 0.44$–$0.50$) are over 1,000× below $I^*_{\text{H100}} = 591$. An 8B model is $\sim 1,200\times$ more bandwidth-bound than compute-bound on H100 for any realistic decode scenario.

---

## 4. Diagnosing the Regime: Practical Heuristics 🔬

### 4.1 Theoretical Classification

Given the roofline model from [[concepts/deep-learning-engineering/nvidia-gpu-hardware#7. The Roofline Model|Roofline Model]], classifying a workload is straightforward in principle:

1. Compute the arithmetic intensity $I = F / B$ for the kernel or end-to-end step.
2. Compare to the GPU's ridge point $I^*$.

**If $I < I^*$:** memory-bound. Attainable throughput is $P \approx \beta \cdot I$; Tensor Cores are underutilized.

**If $I > I^*$:** compute-bound. Attainable throughput is $P \approx \Pi$; bandwidth is underutilized.

The quick rule: **autoregressive decode with $B < 100$ on any GPU from V100 onward is almost certainly memory-bound.** For H100, the threshold rises to $B \approx 300$–$600$ (Section 1).

> [!WARNING] The $I \approx B/2$ estimate ignores KV-cache and activation bandwidth
> The weight-only estimate $I \approx B/2$ understates total data movement. KV-cache loading (Section 3) and activation transfers across layers add additional bandwidth not captured by $I \approx B/2$. The true $I$ is lower. Thus $I \approx B/2$ is an *upper bound* on the true arithmetic intensity; the workload is at least as memory-bound as this estimate implies, and often more so.

### 4.2 Empirical Signals Without Profiling

When computing $I$ analytically is cumbersome, three empirical signals reveal the regime:

**📐 Signal 1: Batch-size scaling.** Double the batch size $B \to 2B$ and measure throughput change:
- *If throughput roughly doubles:* the layer was compute-bound (FLOPs doubled, bandwidth unchanged). *This is rare at low batch.*
- *If throughput barely increases:* memory-bound. The bandwidth was already saturated; doubling $B$ doubled FLOPs but the bottleneck is bytes/second, not FLOPs/second.
- *In practice:* a linear throughput increase with $B$ until a knee, then saturation, is the signature of memory-to-compute crossover.

**📐 Signal 2: nvidia-smi bandwidth vs. SM utilization.** The command `nvidia-smi dmon -s u` (polling interval 100 ms) reports two key columns: `sm` (SM compute utilization, %) and `mem` (HBM bandwidth utilization, %). The diagnostic:
- High `mem`, low `sm` → memory-bound. The HBM bus is the bottleneck.
- High `sm`, low `mem` → compute-bound. Tensor Cores are the bottleneck.
- Both high → well-balanced kernel near the ridge point (rare in practice).
- Both low → kernel is too short (latency-bound by kernel launch overhead or small problem size).

**📐 Signal 3: Manual bandwidth saturation check.** Estimate whether you are saturating bandwidth as follows. Let $T$ be the observed token generation time (seconds/token) and $P_{\text{bytes}}$ be the model size in bytes. Then:
$$\text{Effective memory bandwidth used} \approx \frac{P_{\text{bytes}}}{T}$$
If this value is close to the GPU's peak $\beta$ (within 20–30%), the kernel is memory-bandwidth-saturated. If it is far below $\beta$, the bottleneck may be compute, kernel launch latency, or CPU-GPU synchronization.

> [!EXAMPLE] Sanity-check on H100 for LLaMA-2-7B
> Model size: $7 \times 10^9 \times 2 = 14$ GB. H100 SXM bandwidth: $3.35$ TB/s = $3350$ GB/s. Predicted minimum decode latency at $B=1$: $14 / 3350 \approx 4.2$ ms/token = ~238 tokens/s. If you observe 200 tokens/s, efficiency = $200/238 \approx 84\%$ of bandwidth — memory-bandwidth-saturated. If you observe only 50 tokens/s, something else (quantization overhead, CPU scheduling, attention kernel inefficiency) is reducing throughput below the bandwidth limit.

### 4.3 Profiling Tools 🛠️

**NVIDIA Nsight Compute (`ncu`).** This is the authoritative kernel-level profiler. Run with:

```bash
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_active,\
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.pct_of_peak_sustained_elapsed \
python inference_script.py
```

The key output section is *GPU Speed of Light* (or *Memory Throughput* and *Compute (SM) Throughput* in newer versions). These metrics express the fraction of peak sustained throughput for memory and compute respectively:

- **Memory Throughput near 100%, Compute Throughput low:** memory-bound. The kernel is streaming data as fast as possible; Tensor Cores are idle.
- **Compute Throughput near 100%, Memory Throughput low:** compute-bound. Tensor Cores are saturated; bandwidth is available.
- **Both near 100%:** roofline-optimal (uncommon).

*Nsight Compute adds overhead (typically 10–100× slowdown per kernel) and should be used on a representative single batch, not in a production loop.*

**PyTorch Profiler.** For end-to-end attribution (identifying which ops dominate), use:

```python
import torch
from torch.profiler import profile, ProfilerActivity, schedule

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=2, active=5),
    with_stack=True,
) as prof:
    for _ in range(8):
        output = model.generate(inputs, max_new_tokens=1)
        prof.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

Look at the `cuda_time_total` column — linear layer kernels (e.g., `ampere_sgemm` or `cutlass_gemm_*`) dominate for weight-loading bottlenecks. A high ratio of memory-copy time to kernel execution time is another signal of bandwidth-bound behavior.

**Simple Python micro-benchmark.** For quick order-of-magnitude checks:

```python
import torch, time

# Warm-up
for _ in range(5):
    y = model(x)
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(N):
    y = model(x)
    torch.cuda.synchronize()
elapsed = (time.perf_counter() - t0) / N  # seconds per step

tflops_observed = flops_per_step / elapsed / 1e12
bw_observed     = bytes_per_step / elapsed / 1e12  # TB/s

print(f"Observed: {tflops_observed:.1f} TFLOPS, {bw_observed:.2f} TB/s")
print(f"Peak:     {peak_tflops:.1f} TFLOPS, {peak_bw:.2f} TB/s")
print(f"Compute util: {tflops_observed/peak_tflops*100:.1f}%")
print(f"Bandwidth util: {bw_observed/peak_bw*100:.1f}%")
```

If bandwidth utilization is high (>50%) and compute utilization is low (<5%), the workload is memory-bound.

### 4.4 Operation-Level Rules of Thumb 📋

The following table summarizes typical arithmetic intensities for operations encountered in transformer inference. Values assume BF16, large-$d$ regime, short sequences unless noted.

| Operation | Typical $I$ (FLOP/byte) | Regime (H100, $I^* = 591$) | Notes |
|-----------|------------------------|---------------------------|-------|
| Autoregressive decode, $B=1$ | $\approx 0.5$ | Deeply memory-bound | All weight matrices = GEMV |
| Autoregressive decode, $B=32$ | $\approx 16$ | Memory-bound ($36\times$ below $I^*$) | — |
| Element-wise (ReLU, GELU, add) | 1–3 | Always memory-bound | 1 FLOP per loaded byte |
| LayerNorm / RMSNorm | 5–15 | Memory-bound | Reduction + broadcast |
| Softmax (short seq) | $\approx 1.25$ | Memory-bound | 5 FLOPs / 4 bytes |
| KV-cache attention, $S=512$ | $\approx 1$ | Memory-bound | QK product is GEMV in decode |
| Prefill attention, $S=2048$ | $\approx 1024$ | Compute-bound | GEMM scales with $S^2$ |
| Large GEMM ($B \geq 512$, $d \geq 4096$) | 100–1000+ | Compute-bound | Training-like regime |
| Weight-only quantization load (INT4) | $\approx 0.25$ | More memory-bound | Fewer bytes, same FLOPs |

> [!NOTE] Quantization and memory-bound inference
> Weight-only quantization (INT4, INT8) reduces bytes loaded from HBM by 4× or 2× respectively, while FLOPs are unchanged (dequantize-then-multiply at BF16). This reduces the denominator in $I = F/B$, so $I$ increases — the kernel moves toward the ridge point. However, since $I$ starts near 0.5 and $I^*$ is near 591 on H100, even INT4 quantization yields $I \approx 2$, which remains deeply memory-bound. The speedup from quantization in this regime is from reduced bytes loaded, not from a regime change. *Quantization does not make inference compute-bound.*

---

> [!QUESTION] Exercise 4: Batch-Size Scaling as Regime Diagnostic
> *This problem formalizes the batch-size-doubling experiment as a regime discriminant.*
>
> > **Prerequisites:** [[#4.2 Empirical Signals Without Profiling|4.2 Empirical Signals Without Profiling]], [[#1.2 Formal Derivation of Arithmetic Intensity for a Linear Layer|1.2 Formal Derivation of Arithmetic Intensity for a Linear Layer]]
>
> Let $T(B)$ be the wall-clock time per token at batch size $B$ for a weight-dominant workload. From the roofline model:
>
> $$T(B) = \max\!\left(\frac{2P \cdot B}{\Pi},\ \frac{2P}{\beta}\right) / B$$
>
> (a) Simplify $T(B)$ in the memory-bound regime ($B \ll I^*$) and compute the throughput $\text{tok/s}(B) = B/T(B)$.
>
> (b) Simplify $T(B)$ in the compute-bound regime ($B \gg I^*$) and compute throughput.
>
> (c) Show that $\text{tok/s}$ is linear in $B$ for memory-bound and constant in $B$ for compute-bound. Use this to explain the batch-doubling test.

> [!TIP]- Solution to Exercise 4
> **Key insight:** In the memory-bound regime the model reads $2P$ bytes regardless of $B$ — so throughput scales linearly with $B$. In the compute-bound regime every additional token adds FLOPs that are now the constraint, so throughput saturates at $\Pi / (2P \text{ FLOP/token})$.
>
> **Sketch:**
>
> (a) Memory-bound: $I = B/2 < I^*$, so $\beta \cdot I < \Pi$ and the memory roof binds. Time to process $B$ tokens: $2P / \beta$ (load all weights once). Time per token: $2P / (B \beta)$. Throughput: $B / T(B) = B \cdot B\beta / 2P = B\beta/2P$... wait — step back. The time to process $B$ tokens at once is $\max(2PB/\Pi, 2P/\beta)$. Memory-bound: time = $2P/\beta$. Throughput $= B / (2P/\beta) = B\beta/(2P)$, which is linear in $B$. ✓
>
> (b) Compute-bound: time = $2PB/\Pi$. Throughput $= B / (2PB/\Pi) = \Pi/(2P)$, independent of $B$. ✓
>
> (c) Memory-bound $\Rightarrow$ throughput $\propto B$: doubling $B$ doubles tokens/second. Compute-bound $\Rightarrow$ throughput = const: doubling $B$ leaves tokens/second unchanged. The batch-doubling test exploits exactly this: observe whether throughput doubles or is constant.

---

## 5. Implications for Hardware Choice 💡

The memory-bound nature of low-batch inference has concrete implications for GPU selection, deployment strategy, and cost.

**📐 At low batch size ($B \leq 32$): bandwidth is the only figure of merit.**

An H100 doing inference at $B = 1$ operates at $< 0.1\%$ of its peak TFLOPS. It is running as a 3.35 TB/s memory bandwidth device. An H200, with 4.8 TB/s HBM3e, delivers $\approx 43\%$ more tokens/second at $B = 1$ — not because of its identical GH100 compute, but purely from the bandwidth upgrade. A B200 at 8.0 TB/s delivers $\approx 2.4\times$ the H100's throughput per GPU for single-request latency.

| GPU | HBM Bandwidth | Bandwidth gain vs. H100 | Relative throughput at $B=1$ |
|-----|--------------|------------------------|------------------------------|
| A100 SXM | 2.00 TB/s | 0.60× | 0.60× |
| H100 SXM | 3.35 TB/s | 1.00× (baseline) | 1.00× |
| H200 SXM | 4.80 TB/s | 1.43× | 1.43× |
| B200 SXM | 8.00 TB/s | 2.39× | 2.39× |

*TFLOPS advantages between these GPUs are irrelevant for $B = 1$ decode.*

**📐 At high batch size ($B \geq I^*$): compute matters.**

*Continuous batching* systems like vLLM aggregate requests across users, effectively building batch size from incoming request streams. At sufficient load, $B$ can approach or exceed the ridge point, transitioning the workload toward compute-bound. In this regime, higher TFLOPS (H100 vs. A100, for instance) yield proportional throughput gains.

The implication: for a latency-sensitive deployment (e.g., interactive chat with p95 TTFT < 500 ms), deploy fewer H200s or B200s and prioritize bandwidth. For a throughput-optimized deployment (offline batch inference, serving many users simultaneously), increase batch size until compute-bound and prioritize TFLOPS.

**📐 Blackwell and the convergence of regimes.**

The B200 targets both regimes simultaneously: 8.0 TB/s HBM3e addresses latency-critical memory-bound workloads, while 4,500 FP16 TFLOPS (9,000 with sparsity) addresses throughput-critical compute-bound workloads. Its ridge point of $I^* \approx 563$ FLOP/byte (dense FP16) is similar to the H100's, meaning the same batch-size crossover thresholds apply. The B200 advantage for inference is primarily bandwidth, not a structural change in the compute-vs-memory balance.

**📐 TPU note.** Google TPUs optimize for high-throughput, compute-bound training workloads: they have high TFLOPS but HBM bandwidth that is competitive with, not superior to, NVIDIA GPUs. TPU inference typically requires large batches or *speculative decoding* (which increases effective FLOPs per step) to achieve high utilization. Single-request TPU inference is similarly memory-bound.

> [!WARNING] The <1% utilization number is not a failure mode
> An H100 running at 0.08% compute utilization during inference is not misconfigured — it is doing exactly what the roofline model predicts. The GPU is bandwidth-saturated; all 3.35 TB/s of HBM is in use. Compute utilization is irrelevant when bandwidth is the bottleneck. Comparing GPUs by compute utilization alone ("the H100 is only 0.08% utilized!") is a category error. The correct metric is **tokens per second per dollar**, which decomposes to *bandwidth utilization efficiency* at low batch.

---

> [!QUESTION] Exercise 5: H200 vs. H100 Cost-Effectiveness for Inference
> *This problem derives a cost-effectiveness ratio for two GPUs across regimes.*
>
> > **Prerequisites:** [[#5. Implications for Hardware Choice|5. Implications for Hardware Choice]], [[#4.1 Theoretical Classification|4.1 Theoretical Classification]]
>
> Suppose an H100 SXM costs \$2.50/hr on a cloud provider and an H200 SXM costs \$3.50/hr (representative 2025 figures). The H100 has $\beta = 3.35$ TB/s and $\Pi = 1{,}979$ TFLOPS; the H200 has $\beta = 4.80$ TB/s and the same $\Pi = 1{,}979$ TFLOPS.
>
> (a) For a latency-sensitive deployment at $B = 1$, compute tokens-per-second for each GPU (use $I = 0.5$ FLOP/byte and a 7B model with $P_{\text{bytes}} = 14$ GB). Then compute tokens per dollar (tokens/second divided by \$/second).
>
> (b) For a throughput deployment at $B = 1000$ (compute-bound regime), compute tokens-per-second for each GPU and tokens per dollar. Which GPU is more cost-effective here?
>
> (c) What is the crossover batch size $B^{\dagger}$ above which the H100 is more cost-effective than the H200?

> [!TIP]- Solution to Exercise 5
> **Key insight:** The H200 is more cost-effective at low batch (bandwidth-limited regime) but the H100 becomes more cost-effective at high batch where compute is the bottleneck and both GPUs have the same TFLOPS — the H100's lower price wins.
>
> **Sketch:**
>
> (a) Memory-bound throughput $= B \cdot \beta / (2P_{\text{bytes}})$. At $B=1$: H100: $1 \times 3.35 \times 10^{12} / (2 \times 14 \times 10^9) \approx 119.6$ tok/s. H200: $1 \times 4.80 \times 10^{12} / (2 \times 14 \times 10^9) \approx 171.4$ tok/s. Cost: H100 = \$2.50/3600 = $6.94 \times 10^{-4}$/s; H200 = $9.72 \times 10^{-4}$/s. Tokens/dollar: H100 = $119.6 / 6.94 \times 10^{-4} \approx 172{,}000$ tok/\$; H200 = $171.4 / 9.72 \times 10^{-4} \approx 176{,}000$ tok/\$. H200 wins, but narrowly.
>
> (b) Compute-bound throughput $= B \cdot \Pi / (2PB) = \Pi / (2P)$ where $P = 7 \times 10^9$. Both GPUs: $1979 \times 10^{12} / (2 \times 7 \times 10^9) = 1979/14 \approx 141{,}000$ tok/s. Tokens/dollar: H100 $\approx 141{,}000 / 6.94 \times 10^{-4} \approx 2.03 \times 10^8$ tok/\$; H200 $\approx 141{,}000 / 9.72 \times 10^{-4} \approx 1.45 \times 10^8$ tok/\$. H100 wins by $\approx 40\%$ in cost-effectiveness (same TFLOPS, lower price).
>
> (c) Crossover when cost-effectiveness is equal. For memory-bound regime: throughput ratio H200/H100 = $4.80/3.35 \approx 1.433$; cost ratio H200/H100 = $3.50/2.50 = 1.40$. Since $1.433 > 1.40$, H200 is slightly better in memory-bound regime. As $B$ grows, both enter compute-bound where throughput is equal; H100 wins on cost. The crossover is near the ridge point $B \approx I^*_{\text{H100}} \approx 1182$, above which compute-bound behavior equalizes throughput and cost dominates. For this specific model and pricing, the H200 advantage is marginal even at low batch.

---

## References

| Reference Name | Brief Summary | Link to Reference |
|----------------|---------------|-------------------|
| Yuan et al., "LLM Inference Unveiled: Survey and Roofline Model Insights" (2024) | Survey applying the roofline model to LLM inference stages; derives arithmetic intensities for prefill and decode, shows quantization effectiveness depends on regime | [arXiv:2402.16363](https://arxiv.org/abs/2402.16363) |
| "Mind the Memory Gap: GPU Bottlenecks in Large-Batch LLM Inference" (2025) | Analyzes memory bottlenecks at large batch sizes; profiling methodology for identifying bandwidth vs. compute limits | [arXiv:2503.08311](https://arxiv.org/abs/2503.08311) |
| NVIDIA Nsight Compute Profiling Guide | Official documentation for `ncu` metrics including `Memory Throughput` and `Compute (SM) Throughput` percentages used in Section 4.3 | [NVIDIA Docs](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) |
| NVIDIA H100 Datasheet | Official specifications: 3.35 TB/s HBM3 bandwidth (SXM), 1,979 FP16 TFLOPS | [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/) |
| NVIDIA H200 Datasheet | Official specifications: 4.8 TB/s HBM3e, 141 GB memory, same GH100 compute as H100 | [NVIDIA H200 Datasheet](https://resources.nvidia.com/en-us-gpu-resources/hpc-datasheet-sc23) |
| Lienhart, "LLM Inference Series: 5. Dissecting Model Performance" | Derives compute-vs-bandwidth bound threshold for decode vs. prefill; H200's favorable bandwidth ratio for inference | [Medium](https://medium.com/@plienhar/llm-inference-series-5-dissecting-model-performance-6144aa93168f) |
| "AI's Memory Wall Problem: Why More GPUs Don't Fix Inference Latency" (Spheron, 2026) | Accessible treatment of the memory wall as barrier to inference scaling; bandwidth saturation at batch size 1 | [Spheron Blog](https://www.spheron.network/blog/ai-memory-wall-inference-latency-guide-2026/) |
| NVIDIA Tesla P100 Datasheet | Pascal-era specifications: 720 GB/s HBM2 bandwidth, 21.2 FP16 TFLOPS | [NVIDIA P100 Datasheet](https://images.nvidia.com/content/tesla/pdf/nvidia-tesla-p100-PCIe-datasheet.pdf) |
| "Evolution of NVIDIA Data Center GPUs: Pascal to Grace Blackwell" | Historical survey of bandwidth and compute specs across generations | [ServersImply Blog](https://www.serversimply.com/blog/evolution-of-nvidia-data-center-gpus) |
| Microbenchmarking NVIDIA Blackwell (arXiv:2512.02189) | Detailed microarchitectural analysis of Blackwell memory and compute performance | [arXiv:2512.02189](https://arxiv.org/abs/2512.02189) |
