# ML Performance Engineering: Solutions

---

## Section 1: Mathematical Development

---

**Problem 1.**

**Key insight:** The two physical ceilings — compute and memory bandwidth — give independent upper bounds on performance; the achievable rate is limited by whichever ceiling is lower.

**Sketch:** For any kernel, execution time $T$ satisfies $T \geq F/P$ (compute-limited) and $T \geq M/B$ (memory-limited). Performance $\hat{P} = F/T \leq F \cdot \min(P/F, B/M) = \min(P, B \cdot I)$. The ridge point is $I^* = P/B$: for $I < I^*$, $B \cdot I < P$ so the memory term is tighter; for $I \geq I^*$, $P$ is tighter. On H100: $I^* = 989\text{ TFLOP/s} / 3.35\text{ TB/s} \approx 295$ FLOP/byte.

---

**Problem 2.**

**Key insight:** GEMM arithmetic intensity grows as $O(\min(M, K, N))$, so large square matrices are always compute-bound.

**Sketch:** FLOPs: $2MKN$ (multiply-add). HBM bytes: $2(MK + KN + MN)$ (reading $A$, $B$, writing $C$ in BF16). Intensity $I = 2MKN / [2(MK + KN + MN)]$. For $M = K = N = n$: $I = n/3$. Compute-bound iff $n/3 \geq 295$, i.e., $n \geq 885$. For typical transformer GEMMs with $n \sim$ thousands, this is comfortably satisfied at large batch sizes. At batch size 1 (decode), $M = 1$ so $I = KN/(K + N + KN) \approx 1$, which is deeply memory-bandwidth bound.

---

**Problem 3.**

**Key insight:** The $O(N^2)$ attention matrix both dominates memory and drives arithmetic intensity to $O(d_h/N)$, making long-context attention strongly memory-bandwidth bound.

**Sketch:**
- (a) Attention matrix size: $H \cdot N \cdot N \cdot 2$ bytes $= 2HN^2$ bytes. For $H = 32$, $N = 8192$: $\approx 4$ GB.
- (b) FLOPs: $\sim 4HN^2 d_h$ (two GEMMs of shape $N \times d_h$ by $d_h \times N$). HBM bytes: $6HN^2$ (reading $Q$, $K$, writing attention, reading it back, reading $V$, writing output) plus lower-order $O(HNd_h)$ terms. Intensity $\approx 4HN^2 d_h / 6HN^2 = 2d_h/3$.
- (c) $I \approx 2d_h/3$. For $d_h = 128$: $I \approx 85$ FLOP/byte $< I^* = 295$, so memory-bandwidth bound. As $N$ grows, the $O(N^2)$ HBM traffic grows while FLOP intensity stays constant at $O(d_h)$.

---

**Problem 4.**

**Key insight:** The online softmax recurrence maintains sufficient statistics $(m_n, d_n)$ so that the true softmax can be recovered at the end of any sequential scan, without storing intermediate values.

**Sketch:** By induction: after step $i$, $m_i = \max_{j \leq i} x_j$ (trivially). For $d_i$: $d_i = d_{i-1} e^{m_{i-1} - m_i} + e^{x_i - m_i}$. Unrolling, $d_n = \sum_{j=1}^n e^{x_j - m_n}$ (all correction factors telescope to $m_n$). Therefore $\text{softmax}(x_i) = e^{x_i - m_n}/d_n$ is exact. In tiled attention: process tile $t$ of keys by updating $(m, d)$ incrementally, then finalize the output accumulator using the full $(m_n, d_n)$ after scanning all tiles — no storage of all $x_i$ needed.

---

**Problem 5.**

**Key insight:** FlashAttention replaces one $O(N^2)$ HBM write (the attention matrix) with $O(N)$ HBM writes (the output), reducing the dominant memory traffic term.

**Sketch:**
- (a) SRAM per tile: $B_r \cdot d + B_c \cdot d + B_r \cdot B_c$ elements for tiles of $Q$, $K$/$V$, and the temporary attention score tile. In bytes: $b(B_r d + B_c d + B_r B_c)$.
- (b) Standard attention materializes $A \in \mathbb{R}^{N \times N}$: $2N^2$ bytes. FlashAttention only writes output $O \in \mathbb{R}^{N \times d}$: $2Nd$ bytes. For $N = 8192$, $d = 128$: $4\times$ reduction in peak HBM storage alone.
- (c) Standard attention: read $Q, K, V$ ($6Nd$ bytes), write $A$ ($2N^2$), read $A$ ($2N^2$), write $O$ ($2Nd$) — total $\sim 4N^2 + 8Nd \approx 4N^2$. FlashAttention: read $Q$ once, $K$/$V$ each $N/B_c$ times; write $O$ once — total $\sim 4Nd + 2Nd \cdot N/B_r \approx 4N^2 d / B_r$. The ratio FlashAttention/Standard $\approx d/B_r$; for $d = 128$ and $B_r = 64$ this is $\sim 2\times$ less HBM IO.

---

**Problem 6.**

**Key insight:** GQA reduces the KV cache by exactly $H/G$ while preserving most of MHA's modeling quality, enabling a direct throughput scaling.

**Sketch:**
- (a) MHA: $2LHd_h N b$ bytes. GQA: $2L G d_h N b$ bytes. MQA ($G=1$): $2L d_h N b$ bytes.
- (b) MHA: $2 \times 80 \times 64 \times 128 \times 8192 \times 2 \approx 21$ GB. GQA ($G=8$): $21 \times 8/64 \approx 2.6$ GB.
- (c) If memory budget is $M_{\text{KV}}$, MHA supports $M_{\text{KV}} / (2LHd_h b)$ tokens; GQA supports $M_{\text{KV}} / (2LGd_h b)$ tokens — a factor of $H/G$ more. More tokens in flight = higher concurrent requests or longer context.

---

**Problem 7.**

**Key insight:** MLA's per-token KV footprint is $d_c$ floats regardless of $H$, achieving compression proportional to $d_h H / d_c$.

**Sketch:**
- (a) $\rho = (2 d_h H) / d_c$ (factor of 2 for K and V vs. single latent).
- (b) $\rho = 2 \times 128 \times 128 / 512 = 64$.
- (c) During the attention computation for one new token, MLA projects $c^{KV}$ via learned matrices $W^K \in \mathbb{R}^{d_c \times d_h H}$ and $W^V \in \mathbb{R}^{d_c \times d_h H}$. For $N$ cached tokens: FLOPs $= 2 \times N \times d_c \times d_h H$. However, these projections read $c^{KV}$ (already in HBM as the cached representation) and read the fixed weight matrices (which can be kept in L2/SRAM for small enough $d_c$). The HBM traffic for the KV cache itself is $N \times d_c \times b$ bytes — unchanged relative to standard MHA on an equal-cache-size basis. The key savings is that $d_c \ll d_h H$, so the same memory budget now supports $\rho$ times more tokens.

---

**Problem 8.**

**Key insight:** Ring all-reduce achieves $2p(N_w-1)/N_w \to 2p$ communication per worker as $N_w \to \infty$, making it bandwidth-optimal and independent of the number of workers at large scale.

**Sketch:** Reduce-scatter: $N_w$ workers each send one chunk of size $p/N_w$ to the right neighbor in each of $N_w - 1$ rounds, totaling $(N_w - 1) \cdot p/N_w$ elements sent. All-gather: symmetric, another $(N_w - 1) \cdot p/N_w$ elements. Total $= 2p(N_w - 1)/N_w$. As $N_w \to \infty$: $\to 2p$. Parameter server approach: each worker sends $p$ parameters to the server and receives $p$ back: $2p$ per worker but the server handles $N_w \cdot p$ total — $N_w$-times more pressure on the server link.

---

**Problem 9.**

**Key insight:** ZeRO stages progressively shard the three memory components of training state, each yielding additive memory reduction that compounds to $O(N_w)$ overall at ZeRO-3.

**Sketch:** Adam: FP32 param ($4\Psi$) + FP32 first moment ($4\Psi$) + FP32 second moment ($4\Psi$) + BF16 param ($2\Psi$) + BF16 grad ($2\Psi$) = $16\Psi$ bytes baseline per rank.
- ZeRO-1: optimizer states ($12\Psi$) sharded $\to$ per rank: $2\Psi + 2\Psi + 12\Psi/N_w$.
- ZeRO-2: optimizer + grads sharded: $2\Psi + 2\Psi/N_w + 12\Psi/N_w$.
- ZeRO-3: all sharded: $2\Psi/N_w + 2\Psi/N_w + 12\Psi/N_w = 16\Psi/N_w$.
- For $\Psi = 7\text{B}$, $N_w = 64$: Baseline $= 112$ GB; ZeRO-1 $\approx 15.75$ GB; ZeRO-2 $\approx 1.96$ GB; ZeRO-3 $\approx 1.75$ GB per rank.

---

**Problem 10.**

**Key insight:** The pipeline bubble shrinks as $m/p \to \infty$; interleaved schedules multiply the effective $m$ by $v$, allowing a larger number of stages without proportionally more bubble.

**Sketch:**
- (a) In GPipe, startup fills $p$ stages ($p-1$ idle bubbles at start), teardown empties them ($p-1$ idle at end). Total stages run: $p + m - 1$. Bubble stages: $p - 1$. Fraction: $(p-1)/(p+m-1)$.
- (b) $\lim_{m \to \infty} (p-1)/(p+m-1) = 0$.
- (c) GPipe: $(8-1)/(8+32-1) = 7/39 \approx 18\%$. 1F1B with $v=2$: $(8-1)/(32 \cdot 2) = 7/64 \approx 11\%$. Interleaved schedule halves the bubble for this configuration.

---

**Problem 11.**

**Key insight:** Gradient checkpointing converts an $O(n)$ memory problem to $O(\sqrt{n})$ by accepting one extra forward pass, a 33% compute overhead.

**Sketch:**
- (a) With checkpoints every $k$ layers: during backward, at most $k$ layers of activations are held simultaneously (from last checkpoint to current position), plus $n/k$ checkpoint tensors. Peak $= (k + n/k) a$. Minimize: $d/dk [(k + n/k)a] = a(1 - n/k^2) = 0 \Rightarrow k^* = \sqrt{n}$. Peak $= 2\sqrt{n} \cdot a$.
- (b) Extra forward passes: $n/k^* = \sqrt{n}$ re-computed segments. Each re-computation costs one forward pass over $k^*$ layers. Total extra forward FLOPs $= n \cdot F_{\text{layer}}$ (same as one full forward). Since backward $\approx 2\times$ forward, total FLOPs $\approx 3F_{\text{fwd}}$ instead of $2F_{\text{fwd}}$: 50% overhead in backward-normalized terms, 33% total overhead.
- (c) PyTorch reruns the function in `torch.no_grad()` during backward. Overhead exceeds $1\times$ forward if: (i) the checkpointed function has data-dependent branches; (ii) random state diverges (dropout seeds differ unless `preserve_rng_state=True`); (iii) the inputs were modified in-place between forward and backward.

---

**Problem 12.**

**Key insight:** SQNR degrades as $R/\sigma$ grows because clipping range must cover outliers at the expense of resolution, and outlier activations in large LLMs create pathologically large $R$.

**Sketch:**
- (a) $\epsilon \sim \text{Uniform}(-s/2, s/2)$: $\mathbb{E}[\epsilon^2] = s^2/12 = R^2/(127^2 \cdot 12)$.
- (b) $\mathbb{E}[x^2] = \sigma^2$. SQNR $= \sigma^2 / (s^2/12) = 12 \cdot 127^2 \cdot \sigma^2 / R^2$. In dB: $\text{SQNR}_{\text{dB}} = 10\log_{10}(12 \cdot 127^2) - 20\log_{10}(R/\sigma)$. For Gaussian $x$, optimal $R \approx 3\sigma$: SQNR $\approx 48$ dB (standard INT8 claim).
- (c) Outliers (activations with $|x_i| \gg \sigma$) force large $R$ to avoid clipping. As $R \nearrow$, $s = R/127$ grows, quantization noise $\propto s^2$ grows, and SQNR falls. LLM.int8() responds by decomposing: treat outlier dimensions in FP16, quantize the rest with small $R$.

---

**Problem 13.**

**Key insight:** The modified rejection sampling preserves the target distribution exactly, and expected accepted tokens equals $k\bar{\alpha}$ where $\bar{\alpha} = 1 - d_{\text{TV}}(p, q)$.

**Sketch:**
- (a) Marginal: $P(\text{output} = x) = q(x) \cdot \min(1, p(x)/q(x)) + (1 - q(x)\min(1,p(x)/q(x))) \cdot p'(x)$. Working through: $= \min(q(x), p(x)) + \text{normalize}(\max(0, p(x) - q(x))) \cdot \sum_y \max(0, p(y) - q(y))$. Since $\sum_y \max(0, p - q) = \sum_y \max(0, q - p)$ (they're equal as both distributions integrate to 1), this simplifies to $p(x)$. $\checkmark$
- (b) $\bar{\alpha} = \sum_x q(x)\min(1, p(x)/q(x)) = \sum_x \min(q(x), p(x)) = 1 - d_{\text{TV}}(p, q)$.
- (c) Each position is accepted independently with $\bar{\alpha}$; the process stops at the first rejection or after $k$ tokens. $\mathbb{E}[\tau] = \sum_{j=0}^{k-1} \bar{\alpha}^j \cdot j \cdot (1-\bar{\alpha}) + k \cdot \bar{\alpha}^k = (1 - \bar{\alpha}^{k+1})/(1 - \bar{\alpha}) - 1$.

---

**Problem 14.**

**Key insight:** At batch size 1, the weight matrix must be loaded from HBM for just 1 output token, making the arithmetic intensity $\approx 1$ FLOP/byte — GPU utilization is near zero, and throughput is bottlenecked purely by HBM bandwidth.

**Sketch:**
- (a) FLOPs for $y = xW$: $2 B d^2$ (multiply-accumulate over inner dimension $d$).
- (b) HBM bytes: $2d^2$ (weight $W$ in BF16) + $2Bd$ (input $x$) $\approx 2d^2$ for small $B$.
- (c) $I = 2Bd^2 / 2d^2 = B$. As $B \to 1$: $I \to 1$ FLOP/byte, far below $I^* = 295$.
- (d) Compute-bound when $B \geq I^* = 295$. So $B_{\min} = 295$ for the projection layers alone. In practice, KV attention also needs to be roofline-analyzed; the KV read dominates at small $B$.

---

**Problem 15.**

**Key insight:** Communication can be hidden by computation only when the GEMM is large enough to keep Tensor Cores busy during the in-flight all-reduce; this constrains the maximum useful tensor-parallelism degree.

**Sketch:**
- (a) $T_{\text{comm}} = 2 \cdot B \cdot d \cdot b / \Lambda$ (factor of 2: reduce-scatter then all-gather).
- (b) Each device runs GEMM of shape $B \times d$ by $d \times (d/t)$: FLOPs $= 2Bd^2/t$. $T_{\text{comp}} = 2Bd^2 / (t \cdot P)$.
- (c) Hide communication iff $T_{\text{comp}} \geq T_{\text{comm}}$: $2Bd^2/(tP) \geq 2Bdb/\Lambda \Rightarrow t \leq Pd/(Pb) = \Lambda d / (P b)$.
- (d) $\Lambda = 900 \text{ GB/s}$, $P = 989 \text{ TFLOP/s}$, $d = 8192$, $b = 2$ bytes: $t_{\max} = (900 \times 10^{12}) \times 8192 / (989 \times 10^{12} \times 2) \approx 3726$. This greatly exceeds available NVLink-connected GPUs (typically 8), so communication is easily hidden at $t = 8$.

---

**Problem 16.**

**Key insight:** Because only the last page is partially utilized, memory waste is bounded by $(P_{\text{size}} - 1)$ tokens per sequence — a constant independent of sequence length — so waste fraction vanishes as sequences grow.

**Sketch:**
- (a) Pages per sequence $i$: $\lceil L_i / P_{\text{size}} \rceil$. Allocated tokens: $\sum_i \lceil L_i / P_{\text{size}} \rceil \cdot P_{\text{size}}$.
- (b) Waste per sequence: $\lceil L_i / P_{\text{size}} \rceil \cdot P_{\text{size}} - L_i \leq P_{\text{size}} - 1$. Waste fraction $\leq N_{\text{seq}}(P_{\text{size}} - 1) / \sum_i L_i = (P_{\text{size}} - 1)/\mathbb{E}[L_i]$.
- (c) $(16 - 1)/512 \approx 2.9\% < 4\%$. This matches the vLLM paper's claimed bound.

---

**Problem 17.**

**Key insight:** DDIM's step reduction gives a linear speedup in inference FLOPs; consistency models reduce the multiplier to near-1, turning inference cost into a single forward pass.

**Sketch:**
- (a) DiT with $n = hw/p^2$ patches, $L$ transformer layers, hidden dim $d$: $F_{\text{step}} \approx 4Ln^2d + 8Lnd^2$ (attention + FFN). For FLUX at $1024\times1024$, $p = 2$, $h = w = 128$: $n = 4096$.
- (b) DDPM: $T \cdot F_{\text{step}} = 1000 F_{\text{step}}$. DDIM: $T' \cdot F_{\text{step}} = 50 F_{\text{step}}$. Speedup factor: $T/T' = 20$.
- (c) Consistency models ($T'' = 1$–$4$): FLOP reduction vs. DDPM: $1000/T''$. vs. DDIM: $50/T''$. At $T''=1$: $1000\times$ over DDPM, $50\times$ over DDIM.

---

## Section 2: Algorithmic Applications

---

**Problem 18.**

**Key insight:** Tiling loads a $T \times T$ block from HBM once and reuses it $T$ times, increasing arithmetic intensity from $\sim 1$ FLOP/byte to $\sim T/2$ FLOP/byte.

**Sketch:**

```
// Thread block: (T, T). Grid: (ceil(M/T), ceil(N/T))
__shared__ float As[T][T], Bs[T][T];
int row = blockIdx.y * T + threadIdx.y;
int col = blockIdx.x * T + threadIdx.x;
float acc = 0.0f;
for (int tile = 0; tile < ceil(K/T); ++tile) {
    // Load tile from A and B into shared memory
    As[threadIdx.y][threadIdx.x] = A[row][tile*T + threadIdx.x];
    Bs[threadIdx.y][threadIdx.x] = B[tile*T + threadIdx.y][col];
    __syncthreads();
    // Compute partial dot product
    for (int k = 0; k < T; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();
}
C[row][col] = acc;
```

Arithmetic intensity: $T$ FLOPs per element loaded $\Rightarrow I = T/2$ FLOP/byte (BF16). For $I \geq 295$: $T \geq 590$. In practice, $T = 128$ achieves $I \approx 64$, still compute-bound for large GEMMs because register reuse (not just shared memory) adds another factor; real GEMM implementations (cuBLAS) achieve $>85\%$ peak using multi-level tiling.

---

**Problem 19.**

**Key insight:** Each kernel category sits in a different roofline regime; fused elementwise kernels shift from memory-bound to compute-bound after fusion because bandwidth cost is amortized over more FLOPs.

**Sketch:**
1. Use `torch.profiler` with `activities=[ProfilerActivity.CUDA]` + `with_flops=True`; record trace. In Nsight Compute, launch with `--metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum` to get FLOP% and BW utilization.
2. Plot each kernel as a point $(I_k, \hat{P}_k)$. Draw roofline: horizontal line at $P_{\text{peak}}$, diagonal $B \cdot I$.
3. Decision tree: GEMM below ridge → low batch size (batch-size limited), or poor tile utilization (check occupancy). Elementwise below bandwidth ceiling → kernel launch overhead or insufficient parallelism.
4. `torch.compile` fuses a chain of $n$ elementwise ops (each with $I \approx 1$ FLOP/byte separately) into one kernel with $I \approx n$ FLOP/byte, shifting the point right on the roofline into or toward the compute-bound region.

---

**Problem 20.**

**Key insight:** The two-loop structure of FlashAttention processes the $N \times N$ attention score matrix in $O(N/B_r \times N/B_c)$ tile iterations, each of which reads only $O(B_r d + B_c d)$ bytes from HBM — totaling $O(N^2 d / B)$ HBM reads, which equals $O(N d)$ up to the constant $N/B$.

**Sketch:**

```python
@triton.jit
def flash_attn_fwd(Q, K, V, O, N, d, Br: tl.constexpr, Bc: tl.constexpr):
    row_block = tl.program_id(0)                      # outer: row blocks of Q
    # Load Q tile [Br, d]
    q = tl.load(Q + row_block * Br * d + ...)
    m = tl.full([Br], -float('inf'), dtype=tl.float32)
    l = tl.zeros([Br], dtype=tl.float32)
    o = tl.zeros([Br, d], dtype=tl.float32)

    for col_block in range(0, tl.cdiv(N, Bc)):
        # Causal mask: skip if col_block > row_block
        if col_block * Bc > (row_block + 1) * Br:
            break
        k = tl.load(K + col_block * Bc * d + ...)   # [Bc, d]
        v = tl.load(V + col_block * Bc * d + ...)   # [Bc, d]
        s = tl.dot(q, tl.trans(k))                   # [Br, Bc] scores
        # Causal mask within tile
        s = tl.where(causal_mask(row_block, col_block, Br, Bc), s, -1e9)
        m_new = tl.maximum(m, tl.max(s, axis=1))
        l_new = l * tl.exp(m - m_new) + tl.sum(tl.exp(s - m_new[:, None]), axis=1)
        o = o * (l * tl.exp(m - m_new))[:, None] / l_new[:, None] \
            + tl.dot(tl.exp(s - m_new[:, None]), v) / l_new[:, None]
        m, l = m_new, l_new

    tl.store(O + row_block * Br * d + ..., o)
```

HBM reads: $Q$ once ($N d b$), $K$ and $V$ each $N/B_r$ times ($N d b \cdot N/B_r$ each). Total $\approx N d b (1 + 2N/B_r)$. HBM writes: $O$ once ($N d b$). For $B_r = B_c = B$: total IO $= O(N^2 d / B + N d)$.

---

**Problem 21.**

**Key insight:** ZeRO-3 requires a parameter all-gather before each layer's forward and backward, and a reduce-scatter of gradients after each backward, making the communication schedule symmetric to the computation.

**Sketch:** For each operation:

| Operation | Collective | Parameters gathered | Peak per-worker memory | Comm. volume |
|---|---|---|---|---|
| Fwd layer 1 (attn) | All-gather attn params | $W_Q, W_K, W_V, W_O$ (all 4) | Shard + gathered attn params | $4d^2/N_w \times N_w = 4d^2$ |
| Fwd layer 2 (FFN) | All-gather FFN params | $W_1, W_2$ | Shard + gathered FFN params | $2 \cdot 4d^2 = 8d^2$ |
| Bwd layer 2 | Reduce-scatter grad FFN, free params | $W_1, W_2$ grads | Shard + grad shards | $8d^2$ |
| Bwd layer 1 | Reduce-scatter grad attn, free params | $W_Q, \ldots$ grads | Shard + grad shards | $4d^2$ |
| Optimizer | Local update on shards | None | $16d^2/N_w$ (all state) | None |

After each all-gather, non-owner shards are freed immediately to recover memory.

---

**Problem 22.**

**Key insight:** Continuous batching maintains $O(1)$ request slots perpetually occupied, approaching 100% GPU utilization whenever $\lambda \mu < \tau$ (arrival rate $\times$ mean length $<$ decode throughput).

**Sketch:**
1. Static batching: all $B$ requests must complete before next batch starts. If lengths are heterogeneous ($\text{Var}[L] > 0$), the GPU idles for $\mathbb{E}[\max(L_1,\ldots,L_B) - \min(L_1,\ldots,L_B)] \cdot T_{\text{step}}$ per batch due to padding or waiting for the slowest sequence.
2. Continuous batching pseudocode:
```
while True:
    run one decode step for all active_requests
    for req in active_requests:
        if req.is_finished():
            active_requests.remove(req)
            output_queue.put(req)
            if waiting_queue:
                new_req = waiting_queue.pop()
                active_requests.add(new_req)
                run prefill for new_req (chunked or full)
```
3. Near-100% utilization when offered load $\rho = \lambda \mu / \tau < 1$ (Little's Law: mean queue occupancy $= \lambda \times \text{sojourn time}$; as $\rho \to 1$, all $B$ slots are perpetually occupied and GPU never idles waiting for requests).
