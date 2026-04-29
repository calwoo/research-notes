# Phase III — Quantization & Compression

*Weeks 11–15 · ~25 hrs*

> **Goal:** Understand and implement the quantization techniques that dominate the Parameter Golf leaderboard. GPTQ, FP8, and INT6 QAT are responsible for the majority of top-50 entries. By the end, you will be able to apply post-training quantization (PTQ) and quantization-aware training (QAT) to any transformer, and understand exactly what the accuracy-vs-compression trade-off looks like in practice.

**Week 11 primary:** Dettmers et al., [LLM.int8() paper](https://arxiv.org/abs/2208.07339) + quantization fundamentals from scratch

**Weeks 12–13 primary:** Frantar et al., [GPTQ paper](https://arxiv.org/abs/2210.17323) + AutoGPTQ library

**Week 14 primary:** [PyTorch FX quantization tutorial](https://pytorch.org/docs/stable/quantization.html) (QAT section)

**Week 15 primary:** Dettmers et al., [QLoRA paper](https://arxiv.org/abs/2305.14314) + SVD-based low-rank compression

---

## Week 11 — Quantization Fundamentals

**Concepts to understand:**

- [ ] Quantization: map floating-point values to a smaller discrete set; a weight stored in INT8 uses 1 byte instead of 4 bytes (FP32) — 4× storage reduction; matrix multiplication in INT8 is also 2–4× faster than FP32 on hardware with INT8 tensor cores (e.g., A100, H100)
- [ ] Symmetric quantization: `q = round(x / s)` where `s = max(|x|) / 127`; zero maps to zero; range is `[-127s, 127s]`; dequantize: `x̂ = q × s`
- [ ] Asymmetric (zero-point) quantization: `q = round(x/s + z)` where `z` is a zero-point offset; allows the quantization range to be shifted to cover asymmetric distributions (e.g., ReLU outputs are all non-negative)
- [ ] Per-tensor vs. per-channel quantization: per-tensor uses one `(s, z)` pair for the entire weight matrix; per-channel uses one `(s, z)` per output channel (row of the weight matrix); per-channel has lower quantization error because each channel can have a different scale
- [ ] Quantization error: `|x̂ - x| ≤ s/2`; the maximum error scales with `s = range / num_bins`; larger range or fewer bits → larger maximum error
- [ ] Calibration: choosing the quantization range based on representative data; max-calibration uses the observed maximum; percentile calibration clips extreme outliers (e.g., use the 99.9th percentile instead of max) — often improves accuracy by avoiding wasteful range allocation to rare outliers

**Coding tasks:**

- [ ] Implement symmetric INT8 quantization and dequantization for a weight matrix; compute quantization error (mean absolute error between original and dequantized weights)
- [ ] Quantize a 100M nanoGPT model's weights to INT8 (post-training, no fine-tuning); measure the model size reduction; run inference and compare perplexity to the FP32 baseline

> [!NOTE] Milestone
> Expected results: INT8 post-training quantization (PTQ) of weights only (keeping activations in FP16) on a small LM should reduce model file size by ~3.5× (from 4 bytes/param to ~1.15 bytes/param including per-channel scales). Perplexity degradation should be less than 1% on a well-calibrated INT8 quantization. If perplexity degrades by more than 5%, check for outlier channels — a single channel with weights 100× larger than the median will cause per-tensor quantization to allocate nearly all its range to that channel, wasting 99% of the representational capacity.

---

## Week 12 — GPTQ: Post-Training Quantization with Second-Order Information

**Primary resource:** Frantar et al., [GPTQ paper](https://arxiv.org/abs/2210.17323) — read §1–3 carefully (the algorithm); §4 (results) for calibration

**Concepts to understand:**

- [ ] The Optimal Brain Surgeon (OBS) framework: find the quantization that minimizes `δL = (1/2) δW^T H δW` where `H` is the Hessian of the loss w.r.t. the weights; the OBS solution optimally redistributes quantization error to minimize loss impact
- [ ] GPTQ simplification: approximate the Hessian as `H ≈ 2 X X^T` where `X` is the input activation matrix (second-order statistics of the activations); this avoids computing second derivatives of the full network
- [ ] Column-wise quantization: GPTQ quantizes one column of `W` at a time; after quantizing column `j`, it updates the remaining columns to compensate for the introduced error using the inverse Hessian: `W[:, j+1:] -= (W[:, j] - Q[:, j]) × H⁻¹[j, j+1:] / H⁻¹[j, j]`
- [ ] Why GPTQ outperforms naive rounding: naive rounding treats each weight independently; GPTQ accounts for the fact that quantizing one weight changes the optimal values of others; the compensation step propagates this information
- [ ] Calibration data for GPTQ: ~128 random sequences from the training distribution; GPTQ only needs to observe the activation statistics (not gradients), so calibration is fast (~minutes for a 1B-parameter model)

**Coding tasks:**

- [ ] Read the GPTQ paper algorithm (Algorithm 1) and implement it for a single linear layer; verify it reduces quantization error vs. naive rounding on a random weight matrix
- [ ] Use the AutoGPTQ library to apply 4-bit GPTQ to your nanoGPT model; measure: model size, inference speed (tokens/sec), and perplexity vs. FP32 baseline
- [ ] Compare 4-bit GPTQ vs. 8-bit PTQ (from Week 11): record perplexity and model size for both; plot the accuracy-vs-size Pareto curve

> [!NOTE] Milestone
> Expected results for a 100M-parameter model: 4-bit GPTQ should achieve ~4× size reduction from FP16 (from ~200MB to ~50MB) with less than 2% perplexity degradation when calibrated on 128 sequences from the training distribution. Naive 4-bit rounding (round-to-nearest without compensation) will show 20–50% perplexity degradation on the same model — the improvement from GPTQ's second-order compensation is the entire story. For models below 100M parameters, GPTQ's advantage is smaller because the models are already well-conditioned; it becomes critical at 1B+ parameters.

---

## Week 13 — Quantization-Aware Training (QAT)

**Concepts to understand:**

- [ ] The QAT idea: simulate quantization *during* training by inserting "fake quantization" nodes that round weights to the nearest quantized value and then dequantize before the rest of the forward pass; gradients still flow through the continuous dequantized values, but the model learns weights that are robust to quantization noise
- [ ] `torch.autograd.Function`: the mechanism for defining custom differentiable operations; requires a class with `@staticmethod forward(ctx, *inputs)` and `@staticmethod backward(ctx, *grad_outputs)`; `ctx.save_for_backward(*tensors)` saves tensors needed by backward; backward must return one gradient per input, with `None` for non-tensor inputs (e.g., Python floats); verify any custom Function with `torch.autograd.gradcheck(fn, inputs)` — this numerically differentiates the forward and checks it matches the analytical backward
- [ ] Straight-through estimator (STE): the gradient of the round function is zero almost everywhere (it is a staircase); the STE approximates it as 1 — `d/dx round(x) ≈ 1`; this is a biased gradient estimator but works well in practice for quantization
- [ ] FP8 training: train with weights and activations in FP8 (E4M3 or E5M2 format); accumulate gradients in FP32; native hardware support on H100; reduces memory by 4× vs. FP32 and 2× vs. BF16
- [ ] E4M3 vs. E5M2: E4M3 (4 exponent bits, 3 mantissa) has higher precision and is used for weights/activations; E5M2 (5 exponent bits, 2 mantissa) has larger dynamic range and is preferred for gradients
- [ ] Per-tensor loss scaling for FP8: because FP8 has limited dynamic range (~448 max for E4M3), activations must be scaled to fit within this range; maintain a per-tensor scale factor `s` such that `x/s` fits in FP8; update `s` dynamically based on the current activation max

**Coding tasks:**

- [ ] **`autograd.Function` drill — STE from scratch:** Implement `class STERound(torch.autograd.Function)` with `forward` returning `x.round()` and `backward` returning `grad_output` unchanged. Test: `x = torch.tensor([0.3, 0.7], requires_grad=True); y = STERound.apply(x); y.sum().backward(); assert x.grad.tolist() == [1.0, 1.0]`. *Expected:* passes. **If `x.grad is None`, you called `STERound(x)` instead of `STERound.apply(x)`. If `x.grad` is all zeros, you forgot to override `backward` and PyTorch computed the true derivative of `round` (which is zero almost everywhere).**
- [ ] **Multi-arg STE with non-tensor input:** Extend to `QuantizedRound.apply(x, scale)` where `scale` is a Python float; `forward` returns `(x / scale).round() * scale`; `backward` must return `(grad_output, None)` — the `None` is for the `scale` argument. Verify that `scale` can be changed between forward calls without recompiling.
- [ ] Implement fake INT8 quantization as a `torch.nn.Module` that wraps a `Linear` layer using your `STERound`; insert it into your nanoGPT and train from scratch; compare final val loss to FP32 training
- [ ] Using `torchao` or `transformers` with FP8: run inference in FP8 on your trained model; measure the speedup on hardware that supports FP8 (H100/A100 with recent CUDA)

> [!NOTE] Milestone
> QAT trained to INT8 should match FP32 training loss within 0.5% and often within 0.1%. The key difference from PTQ: the model has learned to "work around" quantization noise during training, whereas PTQ applies quantization to a model that was never exposed to it. If QAT val loss is worse than FP32 by more than 1%, check: (1) are you applying fake quantization to all weight matrices including the embedding and output projection? (2) is the STE implemented correctly — gradients should not be zero? (3) is your quantization scale per-channel or per-tensor? Per-channel QAT is significantly more stable.

---

## Week 14 — INT6 and Extreme Quantization

**Concepts to understand:**

- [ ] Below 4 bits: INT3 and INT2 quantization cause severe accuracy degradation with standard methods; top Parameter Golf entries use INT6 because it offers a better accuracy/bit trade-off than INT4 for models below ~500M parameters
- [ ] GPTQ group quantization: instead of per-channel scale, use per-group scale — partition the `fan_in` dimension into groups of 128 (or 64, 32); each group has its own scale; provides finer-grained adaptation than per-channel without the overhead of per-element scales
- [ ] NF4 (Normal Float 4): a 4-bit data type designed for normally distributed weights; unlike INT4 (uniform grid), NF4 places quantization levels at the quantiles of a standard normal distribution; this minimizes quantization error for normally distributed weights (which most neural network weights approximately are, post-training)
- [ ] Brotli/LZMA compression of model weights: after quantization, the weight tensor is stored as a byte array; applying a general-purpose compressor (Brotli, LZMA) can reduce the stored size by an additional 10–30% if weights have exploitable statistical structure; used in top Parameter Golf entries to squeeze the final few kilobytes
- [ ] The bits-per-parameter metric: in Parameter Golf, the actual "parameter count" often means bit count / 32 (equivalently, FP32-parameter-equivalent count); a 4-bit model with 10M INT4 parameters counts as 1.25M FP32-equivalent parameters

> [!TIP]- What `torchao` does under the hood: `torch.fx`
> When you call `torchao.quantize_(model, int4_weight_only())`, it uses `torch.fx` to trace the model's computation graph, identify every `nn.Linear` call-site, and replace it with a quantized variant — without you writing any module surgery. `torch.fx.symbolic_trace(model)` produces a `GraphModule` whose `.graph` attribute is a list of `Node` objects with `node.op` ∈ `{'call_module', 'call_function', 'get_attr', 'placeholder', 'output'}`. If torchao fails to quantize a layer, it is usually because symbolic tracing hit a data-dependent branch — inspect with `torch.fx.symbolic_trace(model)` and look for `TraceError`. Fix by providing `concrete_args` or using `torch.fx.wrap` on the offending function.

**Coding tasks:**

- [ ] Apply GPTQ with group size 128 at 4-bit and 3-bit precision to your model; measure perplexity degradation at each bit level; find the minimum bits-per-weight where perplexity stays within 5% of FP32
- [ ] Implement NF4 quantization: compute the NF4 grid (the quantiles of N(0,1) at 16 evenly-spaced probabilities); implement lookup-based NF4 quantization and compare perplexity to INT4 GPTQ on the same model
- [ ] Compress your GPTQ-quantized model weights with Brotli (Python `brotli` library); measure final byte count vs. raw INT4 storage; record the effective bits-per-parameter after compression

> [!NOTE] Milestone
> Expected Pareto curve for a 50M-parameter LM: FP32 baseline at 4 bits/param (by definition); INT8 PTQ at 2 bits/param equivalent with <1% perplexity loss; INT4 GPTQ at 1 bit/param equivalent with ~2–5% perplexity loss; INT3 GPTQ with group-128 at ~0.75 bits/param with 5–15% perplexity loss. NF4 should consistently outperform INT4 by 1–2% perplexity at the same bit budget because neural network weights are approximately normally distributed — the NF4 grid is specifically designed for this distribution.

---

## Week 15 — Low-Rank Compression and the Full Compression Pipeline

**Concepts to understand:**

- [ ] SVD-based compression: any weight matrix `W ∈ ℝ^{m×n}` can be decomposed as `W = UΣV^T`; keeping only the top-`r` singular values gives a rank-`r` approximation `Ŵ = U_r Σ_r V_r^T`; parameter count reduces from `mn` to `r(m+n)`; the best `r` for a given accuracy target is where the singular values drop off sharply
- [ ] LoRA (Low-Rank Adaptation): instead of replacing `W` with a low-rank approximation, keep the pretrained `W` frozen and add a low-rank trainable update `∆W = BA` where `B ∈ ℝ^{m×r}` and `A ∈ ℝ^{r×n}`; during inference, merge: `W' = W + BA`; parameter count for fine-tuning: `r(m+n)` instead of `mn`
- [ ] Choosing the rank `r`: the intrinsic rank of the weight update during fine-tuning is much lower than the full rank; empirically `r = 4` to `r = 64` covers most use cases; `r = d_model / 4` is a reasonable default for SFT tasks
- [ ] Knowledge distillation: train a small "student" model to mimic the output distribution of a large "teacher" model by minimizing `KL(p_teacher || p_student)` rather than cross-entropy against hard labels; the teacher's soft probabilities carry more information than one-hot labels (they encode the relative likelihood of all tokens, not just the most likely one)
- [ ] Combining compression techniques: the full Parameter Golf pipeline is: (1) train a compact architecture, (2) apply QAT or GPTQ, (3) optionally apply SVD to remaining large matrices, (4) apply lossless compression (Brotli/LZMA) to the quantized byte stream
- [ ] Module surgery with `named_modules()` and `setattr`: iterate `[(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]`; to replace a module at path `"transformer.h.2.attn.c_proj"`, split on the last `.` to get the parent path and child name, retrieve the parent with `model.get_submodule(parent_path)`, then `setattr(parent, child_name, new_module)`; collect the full list before mutating — modifying a module's children during iteration over `named_modules()` causes a `RuntimeError`

**Coding tasks:**

- [ ] Apply SVD compression to the attention projection matrices in nanoGPT at various ranks; find the rank at which perplexity stays within 2% of baseline; compute the parameter savings
- [ ] **Module surgery drill:** Write `apply_lora(model, rank, target_substrings)` that walks `named_modules()`, identifies any `nn.Linear` whose name contains one of `target_substrings`, and replaces it with a `LoRALinear(original_linear, rank=rank)` that freezes the original weight and adds trainable `B ∈ ℝ^{out×r}` and `A ∈ ℝ^{r×in}`. After applying, verify: (a) `sum(p.numel() for p in model.parameters() if p.requires_grad)` equals `rank × (in + out) × n_replaced`; (b) `model.transformer.h[0].attn.c_proj` is the new `LoRALinear` instance. *Expected failure mode:* if you call `setattr` on the module returned by `named_modules()` rather than on its parent, the replacement does not register in the model's module hierarchy — the old linear is still used in the forward pass.
- [ ] Implement LoRA merge: `lora_linear.merge()` returns a plain `nn.Linear` with weight `W + B @ A`; verify that the merged output is identical to the un-merged LoRA forward to within `1e-5`; build the full compression pipeline: train → apply LoRA → fine-tune → merge → GPTQ INT4 → Brotli
- [ ] Build the full compression pipeline: train → GPTQ INT4 → SVD on attention projections → Brotli compress; record the final bit count and perplexity at each stage

> [!NOTE] Milestone
> The SVD singular value spectrum of a well-trained weight matrix has a characteristic "elbow" shape: a few large singular values followed by a long tail of small ones. If the spectrum is flat (all singular values similar in magnitude), the matrix is full-rank and SVD compression will not work well — you will need to accept large perplexity degradation for any compression. Matrices that benefit most from SVD: the output projection of the MLP (often low effective rank after training), and the Q/K projection matrices in attention (the attention mechanism often learns a low-dimensional subspace). Matrices that benefit least: the value projection and the MLP input projection (tend to be higher-rank).
