# Phase IV — Inference Systems

*Weeks 16–20 · ~25 hrs*

> **Goal:** Understand how LMs are served efficiently at scale. This is the engineering that determines whether a deployed model is fast enough to be usable, and it is increasingly important for Parameter Golf submissions that require low-latency generation. By the end, you will understand the KV cache from first principles, be able to explain FlashAttention's tiling algorithm, and know how speculative decoding and continuous batching work.

**Week 16 primary:** Implement KV caching from scratch; read GQA paper (Ainslie et al., 2023)

**Week 17 primary:** Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135) paper — §1–3 (algorithm + memory analysis)

**Week 18 primary:** Leviathan et al., [Speculative decoding paper](https://arxiv.org/abs/2211.17192)

**Weeks 19–20 primary:** Kwon et al., [vLLM / PagedAttention paper](https://arxiv.org/abs/2309.06180) + quantized inference with llama.cpp/GGUF

---

## Week 16 — The KV Cache

**Concepts to understand:**

- [ ] Autoregressive generation without caching: to generate token `t`, you run the full forward pass on all `t` tokens; this requires recomputing the K and V projections for tokens `0, …, t-1` at every step; total compute for generating `T` tokens: `O(T² d)` — quadratic in sequence length
- [ ] The KV cache: store the K and V tensors for all previously generated tokens; at step `t`, compute only the Q, K, V for the new token; retrieve cached K, V for all previous tokens; perform attention between new Q and all (cached + new) K, V; total compute: `O(T d)` — linear in sequence length
- [ ] KV cache memory footprint: for each layer, K and V are each `[batch, n_heads, seq_len, d_head]`; total bytes = `2 × n_layers × batch × seq_len × n_heads × d_head × 2` (FP16); for GPT-2 small: `2 × 12 × 1 × 2048 × 12 × 64 × 2 = 75MB` — often the memory bottleneck at long context
- [ ] Multi-Query Attention (MQA): use a single K and V head shared across all Q heads; KV cache memory is reduced by `n_heads` (e.g., 12× for 12-head attention); LLaMA-2/3 and Mistral use this
- [ ] Grouped-Query Attention (GQA): group Q heads into `g` groups, each sharing a K/V head; `n_kv_heads = n_heads / g`; balances MQA's memory efficiency against the quality of MHA; GQA-8 (8 KV heads for 32 Q heads) is the LLaMA-3 default
- [ ] CUDA graphs for decode: the single-token decode step is ideal for CUDA graphs — the computation is identical at every step (static shapes), so capturing one graph and replaying it eliminates kernel-launch overhead (~50–100 launches × ~5μs each = 250–500μs of pure overhead per token); `torch.cuda.CUDAGraph()`, `g.capture_begin()` / `g.capture_end()`, `g.replay()`; inputs must be static tensors, updated in-place with `static_input.copy_(real_input)` before each replay; one graph per KV cache bucket size (e.g., 16, 32, 64, 128, 256)

**Coding tasks:**

- [ ] Add KV caching to the nanoGPT attention module: implement a `past_key_values` cache argument; verify that the generated tokens are identical with and without caching (by running both and comparing outputs token by token)
- [ ] Measure generation latency: generate 500 tokens with and without KV cache; plot tokens/sec vs. sequence length for both; observe the quadratic vs. linear scaling
- [ ] Implement MQA: modify the attention module to use `n_kv_heads=1`; measure KV cache memory reduction vs. full MHA; compare perplexity after training from scratch
- [ ] **CUDA graph drill:** Allocate `static_input = torch.zeros(1, 1, dtype=torch.long, device='cuda')` and `static_out`. Warm up with one non-graph forward. Capture: `with torch.cuda.graph(g): static_out = model(static_input)`. In a decode loop: `static_input.copy_(next_token); g.replay(); collect(static_out)`. *Expected:* output identical to non-graph forward; 2–4× wall-clock speedup at `batch=1`. **The static-shape trap: if you replay the graph with a different `seq_len` than at capture time, output is silently wrong (the captured CUDA ops use hardcoded pointer offsets from capture time). Always capture one graph per decode bucket.**

> [!NOTE] Milestone
> Expected speedup from KV cache: at `seq_len=256`, caching provides ~40× speedup per token; at `seq_len=64`, ~15×. The quadratic vs. linear scaling is stark: without caching, generating the 256th token requires attending over 255 previous tokens (the same full forward pass as prefill); with caching, it requires only one attention operation between the new token and the 255 cached K/V pairs. If your cached and uncached outputs are not identical, the most likely bug is that you are applying the causal mask incorrectly during cached decoding — the causal mask shape changes from `[T, T]` during prefill to `[1, T_cached + 1]` during cached generation.

---

## Week 17 — FlashAttention: IO-Optimal Attention

**Primary resource:** Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135) — §1 (motivation), §2 (GPU memory hierarchy), §3 (algorithm); skip §4+ for now

**Concepts to understand:**

- [ ] The memory hierarchy: an A100 has 40–80 GB of HBM (high-bandwidth memory, 2 TB/s) and 192 KB of SRAM (on-chip, ~20 TB/s); the bottleneck for most operations is the HBM bandwidth, not compute
- [ ] Standard attention is HBM-bound: the naive attention algorithm writes `O(N²)` intermediate values (the attention score matrix `S = QK^T`) to HBM; for `N=8192`, this is 64M FP32 values = 256MB; reading and writing this to HBM is the bottleneck, not the matmul
- [ ] FlashAttention's insight: reorder the computation to never materialize the full `N×N` attention matrix in HBM; compute attention in tiles that fit in SRAM; fuse the softmax, score computation, and weighted sum into a single kernel pass
- [ ] Tiling: partition Q, K, V into blocks of size `B_r × d` and `B_c × d`; for each tile of Q, iterate over all tiles of K and V; maintain running statistics (max and sum of softmax denominators) to apply the online softmax correctly
- [ ] Memory complexity: standard attention requires `O(N²)` HBM memory for the attention matrix; FlashAttention requires only `O(N)` (the output) plus the `O(N²)` attention matrix is never written to HBM
- [ ] FlashAttention-2 improvements: better work partitioning across thread blocks; reduces non-matmul operations; achieves ~50–73% of A100 theoretical FLOP throughput for attention
- [ ] Triton programming model: a Triton kernel is a Python function decorated with `@triton.jit` that runs on the GPU; each kernel launch spawns a grid of "programs" (analogous to CUDA thread blocks); `tl.program_id(axis=0)` identifies the current program; `tl.load(ptr + offsets, mask=mask)` reads a tile from HBM into registers; `tl.store(ptr + offsets, vals, mask=mask)` writes back; the SRAM tile lives in registers — explicitly managed by the programmer, unlike in PyTorch where the framework handles memory
- [ ] Tensor strides and contiguity: a tensor's memory layout is determined by its strides — `x.stride()` returns a tuple where `stride[i]` is the number of elements to skip in memory to move one step along dimension `i`; `x.transpose(1, 2)` swaps strides but does not move data, producing a non-contiguous tensor; `view` requires contiguous memory and raises `RuntimeError` if strides are non-standard; `reshape` calls `contiguous()` internally if needed; the attention Q/K/V layout `[B, N, H, D]` → `[B, H, N, D]` requires a transpose + contiguous that matters for FlashAttention's memory-access pattern

**Coding tasks:**

- [ ] **Triton drill — hello kernel:** Write a Triton kernel `add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr)` that adds two vectors elementwise. Launch with `add_kernel[(n // BLOCK_SIZE,)](x, y, out, n, BLOCK_SIZE=1024)`. *Expected:* output equals `x + y`; benchmark vs. `torch.add` — within 2× at this size.
- [ ] **Triton drill — fused softmax:** Write a Triton kernel that fuses "find row max, subtract, exp, sum, divide" into a single HBM-read pass over the input. Verify correctness against `torch.softmax`. *Expected:* for wide matrices (rows of size 4096+), the fused kernel is 2–4× faster than unfused because the unfused version reads the input twice — once for the max, once for the exp.
- [ ] Implement a CPU-only "tiled" attention in PyTorch that processes Q/K/V in blocks, maintaining running `(row_max, row_sum)` statistics for online softmax; verify it produces the same output as standard attention; then benchmark `torch.nn.functional.scaled_dot_product_attention` vs. the manual attention at `seq_len ∈ {256, 512, 1024, 2048, 4096}`
- [ ] **Stride drill:** For `x = torch.zeros(2, 3, 4)`, print `x.stride()`. Transpose: `y = x.transpose(1, 2)`. Print `y.is_contiguous()`. Try `y.view(2, 12)` — *Expected:* `RuntimeError`. Try `y.reshape(2, 12)` — *Expected:* succeeds (calls `contiguous()` internally). Understand: in the attention module, `q = q.transpose(1, 2).contiguous()` is not optional ceremony — the `.contiguous()` is load-bearing for any downstream `view` that flattens `[H, N]`.

> [!NOTE] Milestone
> At `seq_len=4096, batch_size=8, d_model=512, n_heads=8`: standard attention requires materializing an `[8, 8, 4096, 4096]` attention score tensor = 8 × 8 × 4096² × 4 bytes ≈ 4.3GB of HBM — exceeding most consumer GPU memory. FlashAttention uses <100MB for the same configuration because it never materializes the full `N×N` matrix. For `seq_len=1024`, the difference is less dramatic (70MB vs. 7MB), but the speedup is still 2–4× because FlashAttention makes fewer HBM reads/writes. If `torch.nn.functional.scaled_dot_product_attention` is not faster than manual attention on your hardware, check CUDA and PyTorch versions — FlashAttention requires PyTorch 2.0+ and CUDA 11.6+.

---

## Week 18 — Speculative Decoding

**Primary resource:** Leviathan et al., [Speculative decoding](https://arxiv.org/abs/2211.17192) — §1–3 (algorithm + acceptance analysis)

**Concepts to understand:**

- [ ] The inference bottleneck: for a single request, autoregressive generation is memory-bandwidth-bound, not compute-bound; the GPU spends most of its time reading model weights from HBM (not doing matmuls); this means using a smaller model uses less bandwidth per token but may need more tokens to generate the same output
- [ ] Speculative decoding: use a small "draft" model to generate `K` draft tokens quickly (K is typically 4–8); verify these draft tokens with the target model in parallel (all K tokens in one forward pass); accept or reject each draft token based on the acceptance probability
- [ ] The acceptance criterion: accept draft token `x_t` with probability `min(1, p_target(x_t) / p_draft(x_t))`; if rejected, sample from a corrected distribution `p_corrected = normalize(max(0, p_target - p_draft))`; the resulting output distribution is identical to sampling from the target model — no accuracy loss
- [ ] Latency analysis: if the draft model is `γ` times faster than the target model and the average acceptance rate is `α`, the expected tokens generated per target model call is `(1 - α^{K+1}) / (1 - α)`; at `α=0.7, K=4`, this is ~3.6 tokens per target model call
- [ ] Draft model choice: the draft model must be fast (small) and have high acceptance rate against the target model (similar distribution); typical choice: distilled version of the target model, or a much smaller model in the same family

**Coding tasks:**

- [ ] Implement speculative decoding in PyTorch with a small draft model (e.g., nanoGPT with 4 layers as draft, 12 layers as target); measure acceptance rate at `K=4, 6, 8`; measure wall-clock tokens/sec vs. standard autoregressive decoding
- [ ] Measure acceptance rate vs. temperature: at `T=1.0`, draft acceptance is high; at `T=0.0` (greedy), how does acceptance rate change?

> [!NOTE] Milestone
> Expected acceptance rate at `T=1.0` with a draft model trained to minimize KL divergence from the target: ~70–80% per token. With random initialization of the draft model (worst case): ~1/vocab_size ≈ 0 acceptance. The acceptance rate is the dominant factor in speculative decoding speedup — a 70% acceptance rate with K=4 gives ~3.6 tokens per target model call; a 50% acceptance rate gives only ~2.1. Measure acceptance rate first before measuring wall-clock speedup — if acceptance is low, the draft model needs to be improved or retrained, not the decoding algorithm.

---

## Week 19 — Continuous Batching and PagedAttention

**Primary resource:** Kwon et al., [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180) — §1–4

**Concepts to understand:**

- [ ] Batched inference: process multiple requests simultaneously to improve GPU utilization; naively, all requests in a batch must have the same sequence length (pad shorter sequences, wasting compute and memory)
- [ ] Static vs. continuous batching: static batching processes all requests in a batch to completion before starting new ones; continuous batching (a.k.a. iteration-level scheduling) inserts new requests into the batch whenever a slot opens up (when one request finishes its generation); dramatically improves throughput by eliminating padding
- [ ] Memory fragmentation in KV cache: with static allocation, each request pre-reserves a fixed maximum KV cache block; if a request generates fewer tokens than the maximum, memory is wasted; requests with unpredictable lengths cause severe fragmentation
- [ ] PagedAttention: inspired by virtual memory in OS; divide the KV cache into fixed-size pages (e.g., 16 tokens per page); allocate pages to requests on demand; store a page table mapping logical token positions to physical page locations; allows non-contiguous physical memory for a single sequence
- [ ] Throughput vs. latency trade-off: larger batch sizes → higher throughput (tokens/sec across all users) but higher per-request latency (waiting for other requests to finish); continuous batching improves throughput while controlling per-request latency

**Coding tasks:**

- [ ] Implement a simple continuous batching scheduler: maintain a pool of in-progress requests; at each step, select a batch of up to `max_batch_size` active requests; retire finished requests and admit new ones
- [ ] Measure throughput (total tokens/sec) vs. number of concurrent requests for: (a) static batching with padding, (b) your continuous batching implementation; plot the curves

> [!NOTE] Milestone
> Static batching with padding at 8 concurrent requests of varying lengths (e.g., half finish in 50 tokens, half in 200 tokens): the short requests are padded to 200 tokens and waste 75% of their computation. Continuous batching in the same scenario: short requests leave the batch at step 50, freeing slots for new requests; GPU utilization stays high throughout. Expected throughput improvement: 2–3× at 8 concurrent requests of highly variable length. At uniform request lengths, continuous batching provides negligible benefit — all requests finish at the same time anyway.

---

## Week 20 — Quantized Inference with GGUF and llama.cpp

**Concepts to understand:**

- [ ] GGUF format: the file format used by llama.cpp for quantized LLM inference; stores model weights in a structured binary format with per-layer quantization metadata; supports Q2_K through Q8_0 quantization types
- [ ] llama.cpp quantization types: Q4_K_M (4-bit, k-quants with mixed precision on critical layers), Q6_K (6-bit, higher quality), Q8_0 (8-bit, near-lossless); the `_K` suffix indicates "k-quants" — a technique that uses higher precision for the first and last layers (which are more sensitive to quantization)
- [ ] CPU inference with quantized models: GGUF enables LLM inference on consumer CPUs without a GPU; llama.cpp uses SIMD intrinsics (AVX2, ARM NEON) to perform fast integer matrix multiplication; a Q4_K_M model at 7B parameters fits in ~4.1GB of RAM and runs at ~10 tok/s on a modern laptop CPU
- [ ] Measurement: when reporting parameter counts in Parameter Golf, quantized models report the number of FP32-equivalent parameters (actual_bits / 32); an INT4 model with 10M INT4 weights is reported as 1.25M parameters

**Coding tasks:**

- [ ] Convert your trained nanoGPT to GGUF format using llama.cpp's conversion scripts; quantize to Q4_K_M and Q6_K; run inference and measure: perplexity vs. FP32, memory footprint, tokens/sec on CPU
- [ ] Compare GGUF Q4 vs. AutoGPTQ INT4 on the same model: which achieves lower perplexity at the same bit budget? Why might they differ despite both being 4-bit?

> [!NOTE] Milestone
> GGUF Q4_K_M and AutoGPTQ INT4 may produce different perplexity despite the same nominal bit depth. GGUF K-quants use mixed precision (higher bits on the first/last layers and on specific attention layers that are more quantization-sensitive); AutoGPTQ uses uniform 4-bit across all layers by default. The per-layer sensitivity insight — that some layers can tolerate aggressive quantization while others cannot — is the most important practical heuristic in model compression. Measuring per-layer quantization error and allocating more bits to sensitive layers is the approach used in the highest-scoring Parameter Golf quantization entries.
