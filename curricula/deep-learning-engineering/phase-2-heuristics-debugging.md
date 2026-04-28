# Phase II — Training Heuristics & Debugging

*Weeks 7–14 · ~16 hrs*

> **Goal:** Develop the training intuitions that practitioners use daily — the ones that separate people who can reliably train models from those who cannot. By the end, you should be able to diagnose a broken training run from its loss curve alone, implement key regularization and normalization techniques from scratch, and understand why residual connections are the most important architectural idea in modern deep learning.

**Weeks 7–8 primary:** fast.ai, [Practical Deep Learning for Coders](https://course.fast.ai/), Lessons 1–2

**Weeks 9–10 primary:** Karpathy, [makemore Part 3](https://www.youtube.com/watch?v=P6sfmUTpUmc) (BatchNorm and activation statistics)

**Weeks 11–12 primary:** Karpathy, [makemore Part 4](https://www.youtube.com/watch?v=q8SA3rM6ckI) (initialization and gradient flow)

**Weeks 13–14 primary:** Karpathy, ["A Recipe for Training Neural Networks"](http://karpathy.github.io/2019/04/25/recipe/) (deep re-read) + fast.ai [Lesson 8](https://course.fast.ai/Lessons/lesson8.html)

> **Why this structure:** fast.ai Lessons 1–2 introduce the LR finder and practical recipes at the level of real image classifiers — this is where theory meets practice at scale. makemore Parts 3 and 4 then drill into the mechanisms (normalization, initialization, residuals) that make training stable. Weeks 13–14 are debugging-focused: you have enough vocabulary to re-read Karpathy's recipe and absorb it fully.

---

## Weeks 7–8 — Learning Rate: The Most Important Hyperparameter

**Primary resource:** fast.ai [Lesson 1](https://course.fast.ai/Lessons/lesson1.html) (2 hrs) and [Lesson 2](https://course.fast.ai/Lessons/lesson2.html) (2 hrs)

These lessons train image classifiers on real data. The LR finder is the central technique: a principled method for finding the right learning rate without guessing.

---

### Week 7 — The Learning Rate Finder and Training at Scale

**Concepts to understand:**

- [ ] The LR finder: train for one step at each LR from 1e-7 to 10 (log-spaced), record the loss; the right LR is where the loss decreases fastest (just before it diverges)
- [ ] Why LR matters more than any other hyperparameter: too high → loss spikes and diverges; too low → convergence is so slow it never finishes in practice
- [ ] Transfer learning: start from a pretrained model (e.g., ResNet trained on ImageNet), fine-tune only the last few layers; why this works even when the new task differs substantially
- [ ] `fit_one_cycle`: a 1-cycle learning rate schedule — LR rises from `lr/div` to `lr`, then decays to `lr/(div * 1e4)`; empirically faster convergence than a fixed LR

**Coding tasks:**

- [ ] Follow fast.ai Lesson 1: train a cat/dog classifier using a pretrained ResNet; achieve >90% accuracy in <5 minutes of compute
- [ ] Run the LR finder; plot the loss-vs-LR curve; identify the recommended learning rate visually

> [!NOTE] Milestone
> The LR finder plot has a characteristic shape: loss is flat at very low LR (too slow to learn), falls steeply in the good region, then spikes as LR becomes too large. The recommended LR is one order of magnitude below the minimum of the curve (the "elbow" going down, not the spike going up). If you pick the LR at the minimum itself, loss will often spike — you want the steepest descent, not the point of maximum curvature. Write down the recommended LR and then verify: training at 10× that LR should cause the loss to diverge within the first few steps.

---

### Week 8 — Learning Rate Schedules and the 1-Cycle Policy

**Concepts to understand:**

- [ ] Constant LR: simple but suboptimal — a fixed LR is too large for the final convergence phase
- [ ] Step decay: divide LR by 10 every N epochs; creates sudden jumps that can destabilize training
- [ ] Cosine annealing: `η(t) = η_min + (1/2)(η_max - η_min)(1 + cos(πt/T))`; smooth decay from max to min
- [ ] Warmup: start at LR=0, ramp to target over the first 1–5% of training steps; prevents early instability from large gradients on untrained weights
- [ ] Why warmup matters: at initialization, gradients point in random directions; a large LR at step 0 causes the parameters to jump wildly before the gradient signal is meaningful

**Coding tasks:**

- [ ] Re-train your makemore MLP with a cosine annealing schedule (use `torch.optim.lr_scheduler.CosineAnnealingLR`) and 100-step warmup; compare final loss to constant LR
- [ ] Plot the LR vs step alongside the loss vs step on the same x-axis

> [!NOTE] Milestone
> With cosine decay + warmup, the loss curve should be smoother and converge to a slightly lower final value than with constant LR. The warmup phase (flat LR at near-zero for the first 100 steps) will show slow loss decrease — this is expected. The key observation: without warmup at a high LR, you may see a spike in loss in the first 10–20 steps as the model overshoots from random initialization. Warmup eliminates this. If you see no spike either way, your LR may be too low overall.

---

## Weeks 9–10 — Normalization and Regularization

**Primary resource:** Karpathy, [makemore Part 3](https://www.youtube.com/watch?v=P6sfmUTpUmc) (1h 55m) — Karpathy builds BatchNorm from scratch and plots activation statistics throughout training to show why it matters.

---

### Week 9 — BatchNorm: What It Does and Why It Helps

**Concepts to understand:**

- [ ] Without normalization: activations drift toward zero or saturate at ±1 (for tanh) as depth increases; gradients vanish through the saturated regions
- [ ] BatchNorm: normalize each feature across the batch to mean 0, variance 1, then apply learnable scale `γ` and shift `β`; `x̂ = (x - μ_B) / √(σ²_B + ε)`, `y = γx̂ + β`
- [ ] Why BatchNorm stabilizes training: it decouples the scale of weight matrices from the scale of activations — a 2× scaling of weights does not change the output
- [ ] LayerNorm: normalize across the feature dimension (not the batch dimension); preferred for transformers because it does not depend on batch size
- [ ] Running statistics: at inference, BatchNorm uses a running mean/variance (accumulated during training), not the batch statistics — this is a subtle bug source if you forget `model.eval()`

**Coding tasks:**

- [ ] Following makemore Part 3: implement `BatchNorm1d` from scratch (`__init__`, `forward` with batch stats in train mode, running stats in eval mode)
- [ ] Plot pre-activation histograms at each layer, with and without BatchNorm, after 1000 training steps; observe the saturation difference

> [!NOTE] Milestone
> Without BatchNorm in a 4-layer MLP with tanh activations: the histogram of pre-activations at layer 4 is tightly clustered near ±1 (saturated). Gradient magnitudes at layer 1 are ~100× smaller than at layer 4 — vanishing gradient. With BatchNorm inserted before each tanh: pre-activation histograms stay roughly Gaussian at every layer throughout training. Gradient magnitudes are similar at all layers. Loss converges ~2× faster. If you do not see the saturation effect without BatchNorm, your network may be too shallow — use 5+ layers to make the effect visible.

---

### Week 10 — Dropout and Weight Decay

**Concepts to understand:**

- [ ] Dropout: during training, zero each activation with probability `p` independently; scale the surviving activations by `1/(1-p)` to preserve expected value; at test time, no dropout
- [ ] Why dropout works (intuition): forces the network to not rely on any single activation — each neuron must be useful independently; equivalent to training an exponential ensemble
- [ ] Weight decay (L2 regularization): add `(λ/2)||θ||²` to the loss; the gradient update becomes `θ ← θ - η(∇L + λθ)` — a constant fraction of each weight is subtracted each step
- [ ] Why weight decay helps: penalizes large weights, which tend to overfit; equivalent to placing a Gaussian prior on parameters
- [ ] Typical values: dropout `p ∈ {0.1, 0.2, 0.5}`; weight decay `λ ∈ {1e-4, 1e-3, 1e-2}` depending on model size

**Coding tasks:**

- [ ] Ablation study: train your makemore MLP in 8 configurations (on/off for each of: LayerNorm, dropout p=0.1, weight decay 1e-4). Record final validation loss in a table.
- [ ] Identify which single regularization technique gives the largest improvement; reason about why

> [!NOTE] Milestone
> Expected result on makemore MLP (small model, moderate data): weight decay gives a consistent ~0.05 improvement in val loss; LayerNorm gives the largest improvement (~0.1–0.2) because small MLPs without normalization are unstable; dropout at p=0.1 gives modest or no improvement (dropout is most useful for larger models on smaller datasets). The combination of all three should beat any single regularizer. If weight decay at 1e-2 makes things worse, it is too large — the model is being over-regularized and underfitting.

---

## Weeks 11–12 — Initialization and Gradient Flow

**Primary resource:** Karpathy, [makemore Part 4](https://www.youtube.com/watch?v=q8SA3rM6ckI) (1h 55m) — focuses on initialization pathologies and residual connections.

---

### Week 11 — Xavier/He Initialization and Vanishing Gradients

**Concepts to understand:**

- [ ] Default random init (e.g., `torch.randn(fan_in, fan_out)`): variance of activations grows with `fan_in`; after many layers, activations either explode or vanish
- [ ] Xavier (Glorot) init: scale weights by `√(2 / (fan_in + fan_out))`; designed for sigmoid/tanh — preserves activation variance through layers
- [ ] He (Kaiming) init: scale weights by `√(2 / fan_in)`; designed for ReLU — accounts for the fact that ReLU zeros half the activations
- [ ] Vanishing gradient: if activations are saturated (tanh near ±1), `∂tanh(x)/∂x ≈ 0` — gradients cannot flow backward; early layers learn nothing
- [ ] Exploding gradient: if weights are too large, gradients compound multiplicatively through layers; `loss.backward()` produces NaN; fix with gradient clipping (`torch.nn.utils.clip_grad_norm_`)

**Coding tasks:**

- [ ] Build a 10-layer MLP with tanh, no normalization, standard normal init (`torch.randn * 1.0`); plot gradient norms at each layer after one backward pass
- [ ] Switch to Kaiming init (`torch.randn * (2/fan_in)**0.5`); repeat the gradient norm plot; compare

> [!NOTE] Milestone
> With standard normal init in a 10-layer tanh MLP: gradient norms at layer 1 are often smaller than 1e-6, while at layer 10 they are O(1). The ratio between first and last layer gradient norms is often 1e4 or more. Layer 1 will not learn in any reasonable number of steps. With Kaiming init: gradient norms at all layers are within 1 order of magnitude of each other. Both converge, but Kaiming init converges ~10× faster because early layers receive meaningful gradient signal from step 1.

---

### Week 12 — Residual Connections

**Concepts to understand:**

- [ ] A residual block: `output = x + F(x)` where `F` is a small subnetwork (e.g., linear → norm → activation → linear)
- [ ] Why residuals fix the vanishing gradient problem: `∂L/∂x = ∂L/∂output + ∂L/∂output · ∂F/∂x` — there is always a direct gradient path (`∂L/∂output`) that bypasses `F` entirely
- [ ] ResNet intuition: at initialization, `F(x) ≈ 0` (if weights are small), so `output ≈ x` — the identity mapping; the network learns residuals (corrections) rather than full transformations
- [ ] Depth becomes free with residuals: you can stack 100+ layers and they train; without residuals, training degrades past ~10 layers

**Coding tasks:**

- [ ] Build a 12-layer MLP — first without residuals, then with a residual connection every 2 layers
- [ ] Train both on makemore; plot loss curves side by side
- [ ] Plot gradient norms at each layer for both, after 1000 steps

> [!NOTE] Milestone
> Without residuals (12-layer tanh MLP, Kaiming init): training is slow and may plateau — gradient norms at layers 1–3 are consistently 5–10× smaller than at layers 10–12. With residuals: gradient norms are nearly uniform across all layers; training loss reaches the same value as the non-residual version ~3× faster. This is not because residuals add capacity — both models have similar parameter counts. Residuals help because they create a highway for gradients to reach early layers directly.

---

## Weeks 13–14 — Debugging Training Runs

**Primary resources:**
- Karpathy, ["A Recipe for Training Neural Networks"](http://karpathy.github.io/2019/04/25/recipe/) — re-read carefully now that you have the vocabulary (30 min)
- fast.ai [Lesson 8](https://course.fast.ai/Lessons/lesson8.html) (2 hrs)

Read the Recipe again from scratch. The first time (Week 5–6) many of the references were unfamiliar. Now every heuristic should map to something you have observed.

---

### Week 13 — Reading Loss Curves

**Concepts to understand:**

- [ ] The single-batch overfit test: before any hyperparameter tuning, verify the model can overfit 1 batch to near-zero loss; if it can't, the model or loss function is broken
- [ ] Loss at step 0 as a sanity check: for N-class classification, initial loss should be `-log(1/N)` ± a small amount; anything wildly different signals a bug
- [ ] The gradient norm as a diagnostic: plot `||∇L||` alongside loss; a gradient norm that suddenly spikes predicts a loss spike 1–2 steps later
- [ ] Overfitting signature: train loss decreases while val loss plateaus or increases; solution = more regularization or more data
- [ ] Underfitting signature: both losses plateau at a high value; solution = larger model, lower LR, more training steps (in that order)
- [ ] The loss spike: usually caused by a corrupt batch, exploding gradient, or LR that is slightly too high; gradient clipping resolves the last case

**Coding tasks:**

- [ ] Implement gradient norm logging alongside loss logging in your training loop; plot both together
- [ ] Set LR deliberately 5× too high; observe the loss spike; verify the gradient norm spikes 1 step before loss does

> [!NOTE] Milestone
> With LR=5× the recommended value in your makemore MLP: the loss may decrease for the first 50–100 steps (the model is learning) then suddenly spike by a factor of 2–5× before recovering or diverging. The gradient norm plot will show the spike 1 step earlier. This is the "loss spike" pattern. Adding gradient clipping (`nn.utils.clip_grad_norm_(model.parameters(), 1.0)`) usually delays or eliminates the spike, allowing a slightly higher LR. The tradeoff: clipping slows early convergence because you are discarding large gradient updates.

---

### Week 14 — Systematic Debugging Protocol

**Concepts to understand:**

- [ ] Change one thing at a time: when debugging, never change LR and architecture simultaneously — you will not know which change fixed the issue
- [ ] Ablation study protocol: start from a working baseline; turn off one component at a time; record val loss; identify which component matters most
- [ ] The "random label" test: if you replace all labels with random labels, the model should still overfit the training set (loss → 0) — if it cannot, the architecture is broken
- [ ] Weight and activation statistics: after training stabilizes, weights should not be concentrated near 0 or ±1; activations (post-nonlinearity) should not be all-zero or all-saturated
- [ ] Karpathy's hierarchy of bugs: loss not decreasing → data pipeline bug or wrong loss; loss decreasing but val much higher → overfitting; loss NaN → exploding gradients or division by zero in loss

**Coding tasks:**

- [ ] Introduce three bugs into your makemore training loop: (a) forget `optimizer.zero_grad()`, (b) use the wrong loss function (MSE instead of cross-entropy), (c) accidentally pass targets as floats instead of longs to `F.cross_entropy`. Diagnose each from the loss curve alone — then verify by reading the code.
- [ ] For bug (a), describe what you observe in the gradient norm plot vs. the loss curve

> [!NOTE] Milestone
> Bug (a) — forgetting `zero_grad()`: gradients accumulate across steps, effectively increasing the learning rate. The loss decreases initially faster than expected, then spikes or oscillates. The gradient norm will show a linearly increasing trend rather than the usual noisily constant behavior. Bug (b) — MSE loss: loss starts near 0.5–1.0 (typical MSE range) rather than ~3.3, and converges more slowly; generation quality is poor. Bug (c) — float targets: PyTorch's `F.cross_entropy` will throw a runtime error immediately (expects `torch.long`). This is a useful pattern: bugs that crash immediately are easier to fix than bugs that silently degrade performance.

---

## Phase II Consolidation

**Debugging checklist** (Karpathy's recipe, distilled):

- [ ] Verify initial loss matches the theoretical baseline for the task
- [ ] Overfit a single batch before running on full data
- [ ] Run with a tiny dataset first; verify val loss converges before scaling
- [ ] Plot gradient norms alongside loss; watch for spikes
- [ ] Ablate regularization components one at a time
- [ ] Add complexity one piece at a time; never debug two unknowns simultaneously

> [!NOTE] Milestone
> At the end of Phase II, you should be able to: (1) implement a training loop with LR scheduling, BatchNorm/LayerNorm, dropout, and weight decay from memory; (2) read a loss curve and identify overfit, underfit, or a training spike within 10 seconds; (3) follow the single-batch overfit test as a reflex before any hyperparameter tuning. If you can also explain *why* residual connections prevent vanishing gradients using the chain rule, Phase II is complete.
