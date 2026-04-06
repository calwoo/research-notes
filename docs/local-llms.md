# Running LLMs Locally: A Practical Guide

*Last updated: 2026-04-05*

## Table of Contents

1. [[#1. Why Run Locally|1. Why Run Locally]]
2. [[#2. Memory Requirements|2. Memory Requirements]]
   - [[#2.1 Model Weights|2.1 Model Weights]]
   - [[#2.2 KV Cache|2.2 KV Cache]]
   - [[#2.3 Activations and Overhead|2.3 Activations and Overhead]]
   - [[#2.4 Quick Reference Table|2.4 Quick Reference Table]]
3. [[#3. Quantization Schemes|3. Quantization Schemes]]
   - [[#3.1 Full Precision Baselines|3.1 Full Precision Baselines]]
   - [[#3.2 Post-Training Quantization Formats|3.2 Post-Training Quantization Formats]]
   - [[#3.3 Format Comparison|3.3 Format Comparison]]
4. [[#4. Hardware Considerations|4. Hardware Considerations]]
   - [[#4.1 GPU vs CPU Inference|4.1 GPU vs CPU Inference]]
   - [[#4.2 GPU Tiers|4.2 GPU Tiers]]
   - [[#4.3 CPU-Only Inference|4.3 CPU-Only Inference]]
   - [[#4.4 Apple Silicon|4.4 Apple Silicon]]
5. [[#5. Inference Runtimes|5. Inference Runtimes]]
6. [[#6. Agentic Coding Considerations|6. Agentic Coding Considerations]]
   - [[#6.1 Context Length Requirements|6.1 Context Length Requirements]]
   - [[#6.2 Model Selection for Coding|6.2 Model Selection for Coding]]
   - [[#6.3 Opencode Setup|6.3 Opencode Setup]]
7. [[#7. Practical Recipes|7. Practical Recipes]]
8. [[#8. Advanced Inference Techniques|8. Advanced Inference Techniques]]
   - [[#8.1 Speculative Decoding|8.1 Speculative Decoding]]
   - [[#8.2 KV Cache Compression Beyond Quantization|8.2 KV Cache Compression Beyond Quantization]]
9. [[#9. Gemma 4|9. Gemma 4]]
   - [[#9.1 Model Variants|9.1 Model Variants]]
   - [[#9.2 Architecture Highlights|9.2 Architecture Highlights]]
   - [[#9.3 Memory Requirements|9.3 Memory Requirements]]
   - [[#9.4 Running Locally|9.4 Running Locally]]
   - [[#9.5 Multimodal Capabilities|9.5 Multimodal Capabilities]]
   - [[#9.6 Suitability for Agentic Coding|9.6 Suitability for Agentic Coding]]

---

## 1. Why Run Locally

Running LLMs locally offers privacy, zero API costs, offline availability, and full control over context window and rate limits. For agentic coding workflows (e.g., opencode), local models eliminate per-token costs on long multi-turn sessions and let you run multiple agents in parallel without hitting provider rate limits.

The tradeoff is hardware cost and the inference gap — frontier cloud models (GPT-4o, Claude Sonnet/Opus) still outperform all open-weight alternatives on complex reasoning, though the gap has narrowed significantly as of 2025–2026.

---

## 2. Memory Requirements

### 2.1 Model Weights

The dominant cost at load time. The formula is simple:

$$\text{VRAM}_{\text{weights}} = N_{\text{params}} \times B_{\text{bytes}}$$

where $N_{\text{params}}$ is the total parameter count and $B_{\text{bytes}}$ is bytes per parameter under your chosen dtype:

| Dtype | Bits/param | Bytes/param | Notes |
|-------|-----------|-------------|-------|
| FP32 | 32 | 4 | Training default; rarely used at inference |
| BF16 | 16 | 2 | Inference standard on modern GPUs |
| FP16 | 16 | 2 | Same footprint as BF16; worse dynamic range |
| INT8 | 8 | 1 | ~2× compression vs BF16; minor quality loss |
| INT4 | 4 | 0.5 | ~4× compression vs BF16; some quality loss |
| Q4_K_M (GGUF) | ~4.5 | ~0.56 | Mixed precision; best quality-per-bit |
| Q2_K (GGUF) | ~2.6 | ~0.33 | Aggressive; noticeable degradation |

**Example:** A 70B parameter model in BF16 requires $70 \times 10^9 \times 2 = 140\,\text{GB}$ VRAM — requiring two A100-80GB GPUs or equivalent. At Q4_K_M it drops to ~$70 \times 10^9 \times 0.56 \approx 39\,\text{GB}$, fitting on a single A100-80GB with room for KV cache.

> [!TIP] Rule of Thumb
> A rough field estimate: **1 GB per billion parameters at INT4**, **2 GB/B at INT8**, **4 GB/B at FP32**. These hold well for dense transformers; MoE models (e.g., Mixtral, DeepSeek-V3) use only a fraction of params per forward pass — scale by the *active* parameter count for activation memory, but you still load all weights.

### 2.2 KV Cache

The KV cache stores key and value tensors for all prior tokens in the context window. This is the dominant memory cost for long agentic sessions.

$$\text{VRAM}_{\text{KV}} = 2 \times L \times H \times D_{\text{head}} \times T_{\text{ctx}} \times B_{\text{bytes}}$$

where:
- $L$ = number of layers
- $H$ = number of KV heads (may differ from query heads with GQA/MQA)
- $D_{\text{head}}$ = head dimension
- $T_{\text{ctx}}$ = context length in tokens
- Factor of 2 is for K and V separately

> [!NOTE] GQA Reduces KV Cache Dramatically
> *Grouped Query Attention* (GQA) reduces $H$ to a small number of KV head groups (e.g., 8 instead of 64). Most modern open-weight models (Llama 3, Qwen 2.5, Mistral) use GQA. A model like Llama 3.1 70B uses 8 KV heads vs 64 query heads — an 8× reduction in KV cache size vs standard MHA.

**Example — Llama 3.1 70B at 128k context:**

| Parameter | Value |
|-----------|-------|
| Layers $L$ | 80 |
| KV heads $H$ | 8 |
| Head dim $D$ | 128 |
| Context $T$ | 131,072 |
| Dtype | BF16 (2 bytes) |

$$2 \times 80 \times 8 \times 128 \times 131{,}072 \times 2 \approx 43.7\,\text{GB}$$

This means even after fitting the weights (~39 GB at Q4_K_M), a 128k context window adds another 44 GB of KV cache. In practice, quantizing the KV cache (e.g., to INT8) halves this.

### 2.3 Activations and Overhead

At inference (batch size 1), activation memory is small and proportional to a single layer's hidden dimension — typically 1–4 GB for 7–70B models. The runtime framework (llama.cpp, vLLM, Ollama) also consumes a few hundred MB to a few GB for buffers, CUDA context, and paged attention structures.

**Total budget (conservative):**

$$\text{VRAM}_{\text{total}} \approx \text{VRAM}_{\text{weights}} + \text{VRAM}_{\text{KV}} + \text{VRAM}_{\text{overhead}}$$

Add ~10–15% overhead above weights + KV to be safe.

### 2.4 Quick Reference Table

Estimates for popular models at common quantization levels, assuming 8k context (KV cache ~1–5 GB depending on architecture):

| Model | Params | BF16 | Q8_0 | Q4_K_M | Q2_K |
|-------|--------|------|------|--------|------|
| Qwen2.5 Coder 7B | 7B | 15 GB | 8 GB | 5 GB | 3 GB |
| Llama 3.1 8B | 8B | 17 GB | 9 GB | 5.5 GB | 3.5 GB |
| Mistral Nemo 12B | 12B | 25 GB | 13 GB | 8 GB | 5 GB |
| Qwen2.5 Coder 32B | 32B | 65 GB | 33 GB | 20 GB | 11 GB |
| Llama 3.3 70B | 70B | 140 GB | 70 GB | 43 GB | 24 GB |
| DeepSeek-R1 671B | 671B (~37B active) | 1.34 TB | 670 GB | 400 GB | — |

*Add KV cache on top. For 128k contexts, multiply KV estimates by ~16×.*

---

## 3. Quantization Schemes

### 3.1 Full Precision Baselines

- **FP32** (32-bit float): Standard training precision. 4 bytes/param. Almost never used for inference — no quality benefit over BF16.
- **BF16** (brain float 16): 2 bytes/param. Same exponent range as FP32 (8-bit exponent) but reduced mantissa. The de facto inference standard for GPU inference when VRAM allows.
- **FP16**: 2 bytes/param. Narrower exponent range than BF16 — numerically less stable but widely supported. Common on older GPUs (pre-Ampere).

### 3.2 Post-Training Quantization Formats

#### GGUF / llama.cpp quantization

GGUF is the file format used by llama.cpp and Ollama. It supports mixed-precision quantization where different layer types (attention, FFN, embeddings) can use different bit widths.

| Format | Avg bits/weight | Description |
|--------|----------------|-------------|
| Q8_0 | 8 | Nearly lossless; 2× compression vs BF16 |
| Q6_K | 6.6 | Excellent quality; good balance |
| Q5_K_M | 5.7 | Very good quality; recommended if VRAM allows |
| Q4_K_M | 4.8 | ✅ Best default for memory-constrained setups |
| Q4_K_S | 4.6 | Slightly smaller than Q4_K_M, slightly worse |
| Q3_K_M | 3.9 | Noticeable degradation on complex tasks |
| Q2_K | 2.6 | Aggressive; only for extreme memory constraints |
| IQ4_XS | 4.3 | iMatrix-based; often outperforms Q4_K_M |

> [!TIP] Recommended Default
> **Q4_K_M** is the community consensus for the best quality-per-byte tradeoff. If you have slightly more VRAM, **Q5_K_M** or **Q6_K** give near-BF16 quality. For coding tasks, prefer Q5_K_M or higher since code generation is more sensitive to precision than general chat.

The `_K` suffix uses *k-quants* — a block-wise quantization scheme that preserves scale and min values per block, recovering quality vs naive quantization. The `M` (medium) and `S` (small) variants refer to which layers get slightly higher precision.

> [!NOTE] Perplexity Cost of Quantization
> Measured on 7B-class models, the perplexity penalty relative to BF16 is roughly: **Q8_0** ~0.0 (indistinguishable), **Q6_K** ~+0.01, **Q5_K_M** ~+0.02, **Q4_K_M** ~+0.05, **Q3_K_M** ~+0.2, **Q2_K** ~+0.5+. Perplexity degrades non-linearly — the jump from Q4 to Q3 is larger than Q8 to Q4. For coding tasks, the non-linearity matters: Q3 and below tend to introduce subtle logic errors that perplexity alone doesn't capture.

#### Unsloth Dynamic GGUFs

[Unsloth Dynamic 2.0](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs) is a mixed-precision GGUF format that applies different quantization levels to each individual layer rather than a uniform bit width across the whole model. It also calibrates using a purpose-built 1.5M-token chat dataset (rather than text-only corpora), which matters for instruct models whose activation patterns during instruction following differ from those during plain text prediction.

Key differences from standard GGUF:
- **Per-layer quantization:** Sensitive layers (early attention, final layers) get higher precision; less critical FFN layers are compressed more aggressively
- **KL Divergence calibration:** Optimizes for answer consistency rather than raw perplexity, catching "answer flips" that perplexity misses
- **Works on all architectures:** Dense and MoE alike (earlier versions were MoE-only)
- **2-bit feasibility:** Gemma 3 12B Q2_K_XL under Dynamic 2.0 achieves ~7.5% lower KL divergence than standard Q2_K at the same bit budget

Unsloth Dynamic GGUFs are a drop-in replacement — any runtime that reads standard GGUF (llama.cpp, Ollama, LM Studio) accepts them. Look for models tagged `unsloth/…-GGUF` or `UD-Q4_K_XL`, `UD-Q2_K_XL` on HuggingFace.

#### GPTQ (GPU Post-Training Quantization)

Weight-only quantization using second-order gradient information (Hessian-based) to minimize quantization error. Operates on GPU tensors in native PyTorch/HuggingFace format.

- INT4 and INT8 variants
- Better quality than naïve round-to-nearest at INT4
- Requires GPU for inference (no CPU fallback)
- Used via AutoGPTQ or HuggingFace `bitsandbytes`

#### AWQ (Activation-aware Weight Quantization)

Like GPTQ but identifies and protects *salient channels* (those with high activation magnitude) during quantization. Empirically outperforms GPTQ at INT4 with same footprint.

- INT4 standard, some INT3 support
- Requires GPU; fast inference via AWQ kernels
- Available via `autoawq` or HuggingFace integration

#### bitsandbytes (BnB)

HuggingFace's quantization library for 8-bit and 4-bit inference with `load_in_8bit=True` / `load_in_4bit=True`. Uses NF4 (Normal Float 4) datatype for 4-bit, which is optimized for normally-distributed weights.

- Easiest to use from HuggingFace Transformers
- QLoRA fine-tuning uses NF4 + double quantization
- Inference is slower than AWQ/GPTQ due to dequantization overhead

### 3.3 Format Comparison

| Format | Platform | Quality at INT4 | Speed | CPU support |
|--------|----------|----------------|-------|-------------|
| GGUF Q4_K_M | llama.cpp / Ollama | ⭐⭐⭐⭐ | Fast | ✅ Yes |
| AWQ | GPU / vLLM | ⭐⭐⭐⭐ | Very fast | ❌ No |
| GPTQ | GPU / vLLM | ⭐⭐⭐ | Fast | ❌ No |
| BnB NF4 | HuggingFace | ⭐⭐⭐ | Moderate | ❌ No |
| EXL2 | ExLlamaV2 | ⭐⭐⭐⭐⭐ | Very fast | ❌ No |

> [!NOTE] EXL2
> EXL2 (ExLlamaV2 format) uses mixed-precision quantization with per-tensor bit allocation and is arguably the best quality/speed tradeoff on GPU. Less widely supported than GGUF but worth considering for dedicated GPU setups.

---

## 4. Hardware Considerations

### 4.1 GPU vs CPU Inference

| Aspect | GPU | CPU |
|--------|-----|-----|
| Speed (tokens/sec) | 10–100+ t/s | 1–15 t/s |
| Memory bandwidth | Very high (HBM) | DDR5 bandwidth limited |
| VRAM ceiling | 24–80 GB per card | System RAM (hundreds of GB possible) |
| Cost | High upfront | Already present |
| Best for | Responsive chat, agentic loops | Large models that don't fit in VRAM |

💡 Token generation is *memory-bandwidth-bound*, not compute-bound, at batch size 1. This means a fast CPU with high-bandwidth DDR5 can be surprisingly competitive for small models vs a GPU with limited memory bandwidth.

### 4.2 GPU Tiers

| GPU | VRAM | Memory BW | Target model size |
|-----|------|-----------|-------------------|
| RTX 3090 / 4090 | 24 GB | 936 GB/s | Up to 13B BF16, 34B Q4 |
| RTX 4090 | 24 GB | 1,008 GB/s | Same; slightly faster |
| A6000 Ada | 48 GB | 960 GB/s | Up to 34B BF16, 70B Q4 |
| A100 80GB | 80 GB | 2,000 GB/s | Up to 70B BF16 |
| H100 80GB | 80 GB | 3,350 GB/s | 70B BF16 comfortably |
| 2× RTX 4090 | 48 GB (tensor parallel) | 2,016 GB/s | 34B BF16, 70B Q4 |

> [!WARNING] Multi-GPU Tensor Parallelism
> Multi-GPU inference with llama.cpp or Ollama splits layers across GPUs (pipeline or tensor parallel). PCIe bandwidth between GPUs becomes a bottleneck — NVLink or NVSwitch (A100/H100) gives 600 GB/s vs PCIe 4.0's ~64 GB/s. Expect ~30–40% efficiency loss on PCIe multi-GPU vs single-GPU.

### 4.3 CPU-Only Inference

For machines without a discrete GPU or when the model is too large to fit in VRAM, CPU inference via llama.cpp is viable.

Key factors:
- **RAM:** Must hold the full quantized model + KV cache. 64–128 GB DDR5 is practical for 70B Q4 models.
- **Memory bandwidth:** Determines tokens/sec. DDR5-6400 quad-channel (~200 GB/s) gives better throughput than DDR4.
- **Cores:** llama.cpp parallelizes the prompt processing phase (prefill) across cores. More cores → faster time-to-first-token. Generation speed is memory-bandwidth-bound, not core-count-bound.
- **AVX-512:** Enables better SIMD quantization kernels in llama.cpp. Intel Sapphire Rapids / AMD Zen 4 support this.

**Practical expectation:** A 70B Q4_K_M model on a modern desktop CPU (Ryzen 9 7950X, DDR5-6000) yields ~5–8 tokens/sec generation — tolerable for agentic use, frustrating for interactive chat.

### 4.4 Apple Silicon

Apple Silicon (M1 Pro through M4 Max/Ultra) has a *unified memory architecture* — the same DRAM is shared between CPU and GPU with no PCIe transfer overhead. The GPU cores access system RAM at full memory bandwidth (~400–800 GB/s on M4 Max/Ultra).

This makes Apple Silicon uniquely excellent for local LLM inference:

| Chip | Unified RAM (max) | GPU BW | Practical model |
|------|------------------|--------|----------------|
| M3 / M4 (base) | 24 GB | ~100 GB/s | 13B Q4, 7B BF16 |
| M3 Pro / M4 Pro | 48 GB | ~273 GB/s | 34B Q4, 13B BF16 |
| M3 Max / M4 Max | 128 GB | ~400 GB/s | 70B Q4, 32B BF16 |
| M2 Ultra / M3 Ultra | 192 GB | ~800 GB/s | 70B BF16 |

> [!TIP] Metal Backend in llama.cpp
> llama.cpp's Metal backend offloads layers to the GPU cores on Apple Silicon. Set `-ngl 99` (number of GPU layers) to offload all layers. Ollama does this by default. At full GPU offload on an M4 Max (128 GB), you can run Llama 3.3 70B at Q4_K_M at ~15–20 tokens/sec.

#### Framework benchmark: M4 Pro (single-user vs high-concurrency)

A [2026 benchmark on M4 Pro with 64 GB](https://macgpu.com/en/blog/2026-mac-inference-framework-vllm-mlx-ollama-llamacpp-benchmark.html) running DeepSeek V3 (Q4_K_M) at 32 concurrent agent requests shows meaningfully different tradeoffs:

| Framework | Single-user t/s | 32-user total t/s | Time-to-first-token |
|-----------|----------------|-------------------|---------------------|
| **Ollama v0.8+** | **58** | 720 | **~45 ms** |
| **llama.cpp (Metal)** | 52 | 890 | ~85 ms |
| **vllm-mlx** | 42 | **1,150** | ~120 ms |

*Note: base M4 (16 GB, 120 GB/s bandwidth) yields roughly half these throughput numbers.*

The practical takeaway: **Ollama is fastest for interactive single-user use** (lowest latency); **vllm-mlx wins at high concurrency** (running parallel agentic requests). For solo agentic coding, Ollama's latency advantage makes it the right default. vllm-mlx becomes worthwhile if you are running multiple agents in parallel simultaneously.

---

## 5. Inference Runtimes

| Runtime | Best for | Backend support | Notes |
|---------|----------|----------------|-------|
| **llama.cpp** | CPU + Metal + CUDA | CPU, Metal, CUDA, Vulkan | Low-level; most flexible; GGUF format |
| **Ollama** | Easy local setup | CPU, Metal, CUDA (via llama.cpp) | Wraps llama.cpp; OpenAI-compatible API |
| **vLLM** | GPU serving, batching | CUDA + ROCm | PagedAttention; best for throughput |
| **LM Studio** | Desktop GUI | Metal, CUDA (via llama.cpp) | User-friendly; good for exploration |
| **Jan** | Desktop GUI | CPU, Metal, CUDA | Open-source LM Studio alternative |
| **ExLlamaV2** | GPU, quality | CUDA | Best quality/speed on GPU; EXL2 format |
| **MLX** | Apple Silicon | Metal only | Apple's ML framework; excellent M-chip perf |

For agentic coding use cases, **Ollama** is typically the right choice — it exposes an OpenAI-compatible REST API that most coding tools (opencode, Continue.dev, Cursor local mode) speak natively.

---

## 6. Agentic Coding Considerations

### 6.1 Context Length Requirements

Agentic coding sessions are context-hungry. A typical multi-file refactor or debugging session with opencode can burn through 20k–100k tokens of context, including:

- System prompt with tool definitions
- File contents loaded into context
- Multi-turn tool call / observation pairs
- Accumulated conversation history

**Minimum recommendation:** 32k context. **Target:** 128k.

The KV cache cost at 128k is significant (see §2.2). To manage this:

1. Use a model with *GQA* — almost all modern open-weight models have it
2. Quantize the KV cache (llama.cpp supports `--cache-type-k q8_0` and `--cache-type-v q8_0`)
3. Use KV cache compression at the application layer if the runtime supports it

### 6.2 Model Selection for Coding

For agentic coding specifically, model quality on coding benchmarks (HumanEval, MBPP, SWE-bench) matters more than general chat quality. As of early 2026:

| Model | Size | Coding quality | Context | Notes |
|-------|------|---------------|---------|-------|
| **Qwen2.5-Coder-32B-Instruct** | 32B | ⭐⭐⭐⭐⭐ | 128k | Best open-weight coding model; strong on agentic tasks |
| **DeepSeek-Coder-V2-Lite** | 16B (2.4B active) | ⭐⭐⭐⭐ | 128k | MoE; fast inference, modest active params |
| **Llama 3.3 70B Instruct** | 70B | ⭐⭐⭐⭐ | 128k | Strong general + coding; large VRAM requirement |
| **Qwen2.5-Coder-7B-Instruct** | 7B | ⭐⭐⭐ | 128k | Practical for <8 GB VRAM; decent at focused tasks |
| **Mistral Nemo 12B** | 12B | ⭐⭐⭐ | 128k | Good balance; fits in 8 GB at Q4 |
| **DeepSeek-R1 distills** | 7B–70B | ⭐⭐⭐⭐ | varies | Strong reasoning; slower (chain-of-thought) |

> [!WARNING] Instruction Following Matters Most for Agentic Use
> Raw coding benchmark performance doesn't fully predict agentic behavior. A model that scores well on HumanEval but poorly follows structured tool-call formats will fail in agentic loops. Prefer *-Instruct* variants explicitly fine-tuned for instruction following and prefer models known to handle tool-calling schemas well (Qwen2.5-Coder, Llama 3.x Instruct).

> [!DANGER] 7B Models Are Too Small for Real Agentic Work
> Real-world experiments ([Jethro Carr, 2025](https://www.jethrocarr.com/2025/08/17/experiments-with-local-llms-for-agentic-coding/)) confirm that 7B models produce "code-shaped text" — syntactically plausible output that halluccinates library names and cannot function autonomously across multi-file tasks without manual correction at each step. **Treat ~14B as the practical minimum for agentic coding.** Qwen2.5-Coder 14B (Q4_K_M, ~9 GB) was the smallest model that produced genuinely autonomous (if imperfect) output in these tests.
>
> A second finding worth noting: **context expansion doubles your actual memory footprint**. A 14B model at 9 GB base grew to ~19 GB total once a large coding context was loaded. Plan your VRAM budget against peak context usage, not base model size.

### 6.3 Opencode Setup

[opencode](https://github.com/sst/opencode) is a terminal-based AI coding assistant. To configure it for a local Ollama model:

```json
// ~/.config/opencode/config.json
{
  "provider": {
    "ollama": {
      "api": "http://localhost:11434/v1"
    }
  },
  "model": "ollama/qwen2.5-coder:32b-instruct-q4_K_M"
}
```

Pull the model first:

```bash
ollama pull qwen2.5-coder:32b-instruct-q4_K_M
```

**Performance tips for agentic sessions:**

1. **Set a high context window:** `ollama run qwen2.5-coder:32b --ctx 65536` — Ollama's default context is often 2048; override explicitly.
2. **Quantize KV cache:** Add `OLLAMA_KV_CACHE_TYPE=q8_0` env var (Ollama 0.4+) to halve KV cache memory.
3. **Flash Attention:** Enabled by default in recent Ollama builds — critical for long-context efficiency.
4. **Disable mmap on macOS if running from external SSD** to avoid I/O bottlenecks on model load.

---

## 7. Practical Recipes

### Fitting a 70B model on consumer hardware

**Target:** Llama 3.3 70B or Qwen2.5-Coder 32B on a single RTX 4090 (24 GB VRAM) + 64 GB system RAM

```bash
# Pull a split GGUF for 70B — too large for single-file
ollama pull llama3.3:70b-instruct-q4_K_M

# Or with llama.cpp directly, split VRAM/CPU:
./llama-server \
  -m llama-3.3-70b-instruct-q4_k_m.gguf \
  --n-gpu-layers 40 \      # offload 40/80 layers to GPU (~12 GB VRAM)
  --ctx-size 32768 \
  --cache-type-k q8_0 \    # quantize KV cache
  --threads 16
```

This runs ~40 layers on GPU and the rest on CPU — a hybrid approach that sacrifices some speed (~3–5 t/s) but lets a 39 GB model run on 24 GB VRAM.

### Maximizing throughput on Apple Silicon (M4 Max)

```bash
# Ollama with full GPU offload
OLLAMA_NUM_GPU=99 ollama serve

ollama run qwen2.5-coder:32b-instruct-q4_K_M \
  --ctx 65536 \
  --num-gpu 99
```

Expected: ~18–25 tokens/sec on M4 Max (128 GB) at 32B Q4_K_M.

### Checking memory before loading

```python
import math

def estimate_vram(
    params_b: float,        # billions
    dtype_bytes: float,     # e.g. 0.5 for INT4, 2 for BF16
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    ctx_len: int,
    kv_bytes: float = 2.0,  # BF16 KV cache
) -> dict:
    weights_gb = params_b * 1e9 * dtype_bytes / 1e9
    kv_gb = (2 * n_layers * n_kv_heads * head_dim * ctx_len * kv_bytes) / 1e9
    overhead_gb = 2.0  # conservative runtime overhead
    total_gb = weights_gb + kv_gb + overhead_gb
    return {
        "weights_gb": round(weights_gb, 1),
        "kv_cache_gb": round(kv_gb, 1),
        "total_gb": round(total_gb, 1),
    }

# Qwen2.5-Coder 32B at Q4_K_M, 64k context
print(estimate_vram(
    params_b=32, dtype_bytes=0.56,
    n_layers=64, n_kv_heads=8, head_dim=128,
    ctx_len=65536
))
# → {'weights_gb': 17.9, 'kv_cache_gb': 8.6, 'total_gb': 28.5}
```

---

## 8. Advanced Inference Techniques

Beyond choosing the right runtime and quantization, two additional techniques can substantially improve throughput or extend what fits in a given memory budget.

### 8.1 Speculative Decoding

*Speculative decoding* pairs a small, fast *draft model* with the large *target model*. The draft generates $k$ candidate tokens autoregressively (cheaply), then the target model verifies all $k$ in a single parallel forward pass. If the target accepts a draft token, it costs essentially nothing extra; if it rejects, it falls back to sampling from the target's distribution at that position.

The key insight is that transformer inference is *memory-bandwidth-bound* at batch size 1 — the GPU is idle most of the time waiting for weights to load. Verifying $k$ tokens in one forward pass uses the same memory read as verifying 1, so accepted draft tokens are nearly free.

**Speedup depends on acceptance rate $\alpha$:**

$$\text{speedup} \approx \frac{1 + k\alpha}{1 + \alpha \cdot \text{cost\_ratio}}$$

where $k$ is draft tokens per step and cost\_ratio is the relative cost of the draft vs target. At $\alpha = 0.7$–$0.8$, typical real-world speedups are **2–2.3×**. Below $\alpha \approx 0.5$, speculative decoding hurts throughput.

> [!WARNING] Diminishing Returns at High Concurrency
> Speculative decoding helps most at **batch size 1–4**. At batch size 8+, the GPU becomes more compute-saturated and the "idle bandwidth" argument weakens — speedups drop toward 1×. It is most valuable for interactive single-user agentic sessions.

**Runtime support:**

| Runtime | Speculative decoding support |
|---------|------------------------------|
| **vLLM v0.8.5+** | ✅ EAGLE3 (best), external draft models, n-gram |
| **SGLang** | ✅ EAGLE3 native |
| **TensorRT-LLM** | ✅ EAGLE1/2/3 |
| **llama.cpp** | ⚠️ Draft model support via `--draft-model`; no EAGLE yet |
| **Ollama** | ❌ Not yet supported |
| **MLX** | ❌ Not supported |

**Enabling with vLLM (EAGLE3, recommended):**

```bash
VLLM_USE_V1=1 vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --speculative-config '{
    "method": "eagle3",
    "model": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
    "num_speculative_tokens": 3
  }'
```

Real-world benchmark on Llama 3.3-70B with EAGLE3 on H100: **~2.3× speedup at batch size 1**, declining to ~1.4× at batch size 8. For 70B models in agentic use, this is significant.

**Enabling with llama.cpp:**

```bash
llama-server \
  -m llama-3.3-70b-instruct-q4_k_m.gguf \
  --draft-model llama-3.2-1b-instruct-q8_0.gguf \
  --draft-max 8 \
  -ngl 99
```

The draft model should ideally share the same tokenizer vocabulary as the target. Llama 3.2-1B and 3.3-70B are a natural pair.

### 8.2 KV Cache Compression Beyond Quantization

§2.2 covers simple KV cache quantization (INT8/INT4). The research frontier has pushed further, and some of these techniques are beginning to appear in production runtimes:

**Token eviction (sparse KV):** Not all tokens contribute equally to attention. Methods like *H2O* and *SnapKV* identify and evict low-importance keys/values during generation, keeping only the top-$k$ by attention score. This trades recall of evicted tokens for memory — effective for tasks where the full history is not needed.

**Low-rank KV decomposition (PALU):** Decomposes the K and V matrices into low-rank factors before caching. Achieves up to **7.59× compression** on Mistral-7B with accuracy comparable to INT4 quantization, and can be combined with quantization for further reduction.

**Semantic chunk compression (ChunkKV):** Compresses KV states at the semantic chunk level (sentences, clauses) rather than individual tokens, preserving linguistic coherence. Particularly effective for long-document tasks where individual token eviction breaks cross-sentence references.

**Coupled Quantization (CQ):** Exploits statistical dependencies between adjacent KV channels to push quantization down to **1 bit per channel** while maintaining reasonable accuracy — still research-stage but demonstrating that the information-theoretic floor is far lower than naive INT4.

> [!NOTE] Runtime Availability
> As of early 2026, most of these techniques are research implementations. What is widely available in production runtimes: **INT8/INT4 KV quantization** (llama.cpp `--cache-type-k`, vLLM `--kv-cache-dtype`), **prefix caching** (vLLM, SGLang — reuses KV states across identical prompt prefixes), and **chunked prefill** (vLLM — overlaps prefill and decode for lower latency). The rest require custom kernels or experimental branches.

**Prefix caching** is worth highlighting specifically for agentic coding: if your system prompt + tool definitions are constant across requests, vLLM and SGLang will reuse their KV states, effectively making them free after the first request. In long agentic sessions this can save 2–5 GB of KV recomputation per turn.

```bash
# vLLM prefix caching (enabled by default in v0.4+)
vllm serve qwen2.5-coder-32b-instruct \
  --enable-prefix-caching \
  --max-model-len 65536
```

---

## 9. Gemma 4

*Released April 2026 by Google DeepMind. Apache 2.0 licensed.*

> [!INFO] References
> - [Simon Willison's notes on Gemma 4](https://simonwillison.net/2026/Apr/2/gemma-4/) — practical local-running observations
> - [HuggingFace Gemma 4 blog post](https://huggingface.co/blog/gemma4) — architecture details, benchmarks, and deployment recipes

Gemma 4 is a family of four open-weight multimodal models spanning from a compact 2B-effective to a 31B dense model. The standout variant for local inference is the **26B-A4B MoE**, which has the weight footprint of a 26B model but the per-token compute and KV cache cost of a 4B model.

### 9.1 Model Variants

| Model | Total params | Active params | Context | Modalities |
|-------|-------------|--------------|---------|-----------|
| **gemma-4-E2B-it** | 5.1B (2.3B effective) | All (dense) | 128k | Image, video+audio, text |
| **gemma-4-E4B-it** | 8B (4.5B effective) | All (dense) | 128k | Image, video+audio, text |
| **gemma-4-26B-A4B-it** | 26B | ~4B (MoE) | 256k | Image, video, text |
| **gemma-4-31B-it** | 31B | All (dense) | 256k | Image, video, text |

The `E` prefix on the small models stands for *Effective* — their total parameter counts include a second embedding table (*Per-Layer Embeddings*, see §8.2) that doesn't contribute to the usual "active" compute path.

> [!NOTE] Audio Support
> Only E2B and E4B support audio input (speech recognition, audio QA). The 26B-A4B and 31B models handle video frames but without audio. The audio encoder is USM-style (same base as Gemma-3n).

### 9.2 Architecture Highlights

Gemma 4 introduces several architectural novelties worth understanding for memory/performance estimation:

**Per-Layer Embeddings (PLE):** Each decoder layer receives a small residual signal from its own dedicated embedding table. This is why E2B has 5.1B total parameters but only 2.3B *effective* parameters for the purposes of compute scaling — the embedding tables add parameter count cheaply. PLE improves per-layer specialization without adding depth.

**Alternating local/global attention:** Layers alternate between *sliding-window* local attention (512-token window in small models, 1024 in large) and full *global* attention. This reduces quadratic attention cost for long sequences — most layers see only a local window, with periodic global layers for long-range coherence.

**Dual RoPE:** Standard RoPE for local-attention layers and *proportional RoPE* for global-attention layers, enabling efficient position encoding at up to 256k tokens.

**Shared KV Cache:** The last $N$ layers reuse K/V states computed by earlier layers, reducing both memory and compute at inference. This makes the effective KV cache smaller than the formula in §2.2 would suggest for a full-attention model.

**MoE (26B-A4B):** Standard Mixture-of-Experts gating — only ~4B parameters are activated per forward pass, so KV cache and activation memory scale with 4B, not 26B. You still load all 26B weights into RAM/VRAM.

### 9.3 Memory Requirements

GGUF quantized weights (llama.cpp format):

| Model | GGUF Q4_K_M | BF16 (full) | Notes |
|-------|------------|-------------|-------|
| E2B | ~3 GB | ~10 GB | Fits in 8 GB unified RAM |
| E4B | ~5 GB | ~16 GB | Comfortable on 8 GB GPU |
| 26B-A4B | ~18 GB | ~52 GB | **Sweet spot** — 26B capacity at 4B KV cost |
| 31B | ~20 GB | ~62 GB | Needs 24 GB VRAM at Q4; shared KV cache helps |

> [!TIP] 26B-A4B KV Cache
> Despite loading 26B weights, the 26B-A4B model's KV cache scales with its ~4B *active* parameter path — roughly equivalent to a 4B dense model. At 256k context, this is a significant advantage over the 31B dense model.

Simon Willison reports the 31B variant showed instability in LM Studio (looping output), while E2B/E4B ran reliably. This may be a runtime/quantization issue rather than a model bug — prefer GGUF via llama.cpp/Ollama over LM Studio for the 31B.

### 9.4 Running Locally

#### Ollama / llama.cpp (GGUF)

```bash
# E4B — multimodal, fits in ~6 GB VRAM
ollama run gemma4:e4b-it-q4_K_M

# 26B-A4B — best quality/VRAM tradeoff for agentic use
ollama run gemma4:26b-a4b-it-q4_K_M

# llama.cpp server with explicit context
llama-server \
  -hf ggml-org/gemma-4-E4B-it-GGUF \
  --ctx-size 65536 \
  --cache-type-k q8_0 \
  -ngl 99
```

#### MLX (Apple Silicon)

MLX has native Gemma 4 support and supports TurboQuant for aggressive KV cache compression:

```bash
pip install -U mlx-vlm

# Standard inference
python -m mlx_vlm.generate \
  --model mlx-community/gemma-4-E4B-it \
  --prompt "Explain attention mechanisms"

# With TurboQuant KV cache (3.5-bit, ~4x memory reduction)
python -m mlx_vlm.generate \
  --model mlx-community/gemma-4-26B-A4B-it \
  --prompt "Review this code" \
  --kv-bits 3.5 \
  --kv-quant-scheme turboquant
```

TurboQuant is particularly valuable here — it compresses the KV cache to 3.5 bits, making even 256k context tractable on an M4 Max.

#### HuggingFace Transformers

```python
from transformers import pipeline

pipe = pipeline(
    "any-to-any",
    model="google/gemma-4-e4b-it",
    device_map="auto",
)

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": "https://..."},
        {"type": "text", "text": "What does this code do?"},
    ],
}]
output = pipe(messages, max_new_tokens=512)
```

#### Opencode Integration

The HuggingFace blog explicitly lists opencode as a supported local agent. Configure via:

```json
// ~/.config/opencode/opencode.json
{
  "model": "ollama/gemma4:26b-a4b-it-q4_K_M",
  "provider": {
    "ollama": { "api": "http://localhost:11434/v1" }
  }
}
```

### 9.5 Multimodal Capabilities

| Capability | E2B | E4B | 26B-A4B | 31B |
|-----------|-----|-----|---------|-----|
| Image understanding (OCR, charts) | ✅ | ✅ | ✅ | ✅ |
| Video (frames) | ✅ | ✅ | ✅ | ✅ |
| Audio input | ✅ | ✅ | ❌ | ❌ |
| Function calling | ✅ | ✅ | ✅ | ✅ |
| Extended thinking | ✅ | ✅ | ✅ | ✅ |

Variable image token budgets (70, 140, 280, 560, 1120 tokens) let you trade image detail for context/memory — useful in long agentic sessions where image context is secondary.

### 9.6 Suitability for Agentic Coding

**Recommended: gemma-4-26B-A4B-it at Q4_K_M (~18 GB)**

The 26B-A4B variant is the most compelling Gemma 4 option for local agentic coding:

| Benchmark | 26B-A4B | 31B | Notes |
|-----------|---------|-----|-------|
| MMLU Pro | 82.6% | 85.2% | Strong reasoning |
| LiveCodeBench v6 | 77.1% | 80.0% | Competitive coding |
| Codeforces ELO | 1718 | 2150 | 31B is clearly stronger on hard problems |
| LMArena text | ~1441 | ~1452 | Near-identical human preference |
| Context | 256k | 256k | Both support very long context |

The LMArena scores are striking — the 26B-A4B scores nearly identically to 31B in human preference evals, while using only 4B active parameters per token. For agentic coding tasks (which stress instruction following and tool-call format adherence more than raw reasoning), this tradeoff strongly favors 26B-A4B.

> [!WARNING] Coding vs Reasoning Tasks
> Codeforces ELO (1718 vs 2150) shows the 31B is substantially better at *competitive* programming — algorithmic problem solving under constraints. For typical agentic coding (reading/editing files, writing idiomatic code, following tool schemas), the 26B-A4B is likely sufficient and runs in half the VRAM.

> [!QUESTION] 31B Stability
> Simon Willison observed looping output from the 31B in LM Studio. As of April 2026, verify this is resolved in your chosen runtime before relying on 31B for production agentic use.

---

*See also: [Llama.cpp GitHub](https://github.com/ggml-org/llama.cpp), [Ollama docs](https://ollama.com/docs), [HuggingFace quantization docs](https://huggingface.co/docs/transformers/quantization)*
