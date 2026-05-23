# 🏗️ Structured Pruning: Filters, Channels, and Attention Heads

## Table of Contents

- [[#1. 💡 Unstructured vs. Structured Sparsity|1. Unstructured vs. Structured Sparsity]]
- [[#2. 📐 Filter Pruning via ℓ₁ Norms|2. Filter Pruning via ℓ₁ Norms]]
  - [[#2.1 Saliency of a Filter|2.1 Saliency of a Filter]]
  - [[#2.2 Dependency Chains and Layer Coupling|2.2 Dependency Chains and Layer Coupling]]
  - [[#2.3 💻 PyTorch: ℓ₁ Filter Pruner|2.3 PyTorch: ℓ₁ Filter Pruner]]
- [[#3. 🎚️ Channel Pruning via Batch Norm Scaling|3. Channel Pruning via Batch Norm Scaling]]
  - [[#3.1 The γ-Sparsity Trick|3.1 The γ-Sparsity Trick]]
  - [[#3.2 💻 PyTorch: BN γ Channel Pruner|3.2 PyTorch: BN γ Channel Pruner]]
- [[#4. 🤖 Attention Head Pruning in Transformers|4. Attention Head Pruning in Transformers]]
  - [[#4.1 Head Importance Scoring|4.1 Head Importance Scoring]]
  - [[#4.2 L0 Gate Pruning|4.2 L0 Gate Pruning]]
  - [[#4.3 💻 PyTorch: Attention Head Masking|4.3 PyTorch: Attention Head Masking]]
- [[#5. 🔄 Rethinking the Value of Network Pruning|5. Rethinking the Value of Network Pruning]]
- [[#6. 📚 References|6. References]]

---

## 1. 💡 Unstructured vs. Structured Sparsity

[[concepts/sparsity-pruning/classical-pruning|Classical pruning methods]] produce *unstructured* sparsity: individual weights are zeroed at arbitrary positions. While this achieves high compression ratios, the resulting sparse matrices are *dense in memory layout* — the zero entries still occupy storage and memory bandwidth unless a custom sparse format (CSC, CSR) is used.

On modern hardware (GPU, TPU), dense matrix operations dominate:
- `cuBLAS` `SGEMM` operates on $m \times n$ tiles; any zero within a tile still costs a multiply-accumulate.
- Conv2d on GPU uses Winograd or FFT-based algorithms that assume dense filter tensors.

**Structured pruning** removes entire *groups* of parameters — whole convolutional filters, output channels, or attention heads — so that the resulting network is *smaller but still dense*. The pruned network requires no custom sparse kernels and runs at full speed on standard hardware.

| Property | Unstructured | Structured |
|----------|-------------|------------|
| Granularity | Individual weights | Filters, channels, heads |
| Hardware compatibility | Needs sparse format / custom ASIC | Native BLAS/cuDNN |
| Achievable sparsity | 80–99% | 20–60% (without accuracy loss) |
| Accuracy–compression tradeoff | Better (more fine-grained) | Worse (coarser) |
| Deployment complexity | High | Low |

---

## 2. 📐 Filter Pruning via ℓ₁ Norms

*Li, Kadav, Durdanovic, Samet, Graf (2016). "Pruning Filters for Efficient ConvNets." ICLR 2017.*

### 2.1 Saliency of a Filter

For a convolutional layer with weight tensor $W \in \mathbb{R}^{C_{out} \times C_{in} \times k_H \times k_W}$, each *filter* $W_i \in \mathbb{R}^{C_{in} \times k_H \times k_W}$ produces one output channel. Pruning filter $i$ removes the $i$-th output channel entirely and the corresponding input channel in the *next* layer.

**ℓ₁ filter saliency:**

$$s_i = \|W_i\|_1 = \sum_{c, h, w} |W_{i,c,h,w}|$$

Filters with small ℓ₁ norm produce activations close to zero on average — they contribute little to the representation. Prune the $k$ filters with lowest ℓ₁ norm.

> [!NOTE] Why ℓ₁ and not ℓ₂?
> The ℓ₁ norm sums *absolute values* of all weights in the filter, making it a natural proxy for the filter's average contribution to the output magnitude. The ℓ₂ norm is dominated by a few large weights and can overestimate a filter's importance when it has one large entry and many near-zero entries. Li et al. empirically found ℓ₁ more reliable than ℓ₂.

**Reconstruction error interpretation.** Removing filter $i$ zeroes its output feature map $\hat{a}_i = 0$. The change in the next layer's pre-activations is:

$$\delta z_{i'}^{(l+1)} \approx W^{(l+1)}_{i', i} * 0 = 0 \quad \forall i'$$

The reconstruction error in the next layer's feature maps depends on how much subsequent filters relied on the now-deleted channel. Filters with small activations (small $\|W_i\|_1$) contribute less to this error.

### 2.2 Dependency Chains and Layer Coupling

Pruning filter $i$ from layer $l$ forces us to also remove the $i$-th *input channel* from layer $l+1$. This propagates through skip connections and concatenations:

- **Sequential layers:** Remove filter $i$ from layer $l$ and input channel $i$ from layer $l+1$. Straightforward.
- **ResNet skip connections:** Layer $l$ and the skip branch must produce the same number of channels. If we prune from the residual branch, we must also prune from the identity branch (or use projection shortcuts).
- **Concatenation (DenseNet):** Pruning channels in a dense block requires coordinating which channels to remove from all concatenated inputs.

In practice, structured pruning tools handle these dependencies by building a *layer dependency graph* and propagating pruning decisions through it.

> [!QUESTION] Exercise 1: Filter pruning compression ratio
> *This exercise derives the FLOPs reduction from filter pruning.*
>
> > **Prerequisites:** [[#2.1 Saliency of a Filter|2.1 Saliency of a Filter]]
>
> A convolutional layer has $C_{in} = 64$, $C_{out} = 128$, kernel size $3 \times 3$, and spatial output $H \times W = 56 \times 56$. We prune $30\%$ of the output filters.
>
> (a) Compute the FLOPs (multiply-accumulates) before and after pruning. (FLOPs for a conv layer: $C_{out} \cdot C_{in} \cdot k_H \cdot k_W \cdot H \cdot W$.)
>
> (b) The next layer has $C_{in} = 128$ (must match), $C_{out} = 256$, kernel $3 \times 3$, spatial $28 \times 28$. What is the FLOPs reduction in the next layer from the same pruning decision?
>
> (c) Compare the total FLOPs reduction to the equivalent unstructured pruning at 30% weight sparsity. Which saves more computation on GPU? Why?

> [!TIP]- Solution to Exercise 1
> **Key insight:** Structured pruning saves FLOPs in two consecutive layers; unstructured pruning saves parameter count but not necessarily GPU throughput.
>
> **(a)** Pre-pruning FLOPs: $128 \cdot 64 \cdot 9 \cdot 56 \cdot 56 = 2{,}293{,}760$ MACs. After pruning 30% of output filters: $C_{out}' = 90$. Post-pruning FLOPs: $90 \cdot 64 \cdot 9 \cdot 56 \cdot 56 = 1{,}612{,}800$ MACs. Reduction: $30\%$.
>
> **(b)** The next layer now has $C_{in}' = 90$ (matching the pruned output). Pre-pruning: $256 \cdot 128 \cdot 9 \cdot 28 \cdot 28 = 2{,}293{,}760$ MACs. Post-pruning: $256 \cdot 90 \cdot 9 \cdot 28 \cdot 28 = 1{,}612{,}800$ MACs. Also $30\%$ reduction.
>
> **(c)** Unstructured 30% sparsity: $0$ GPU speedup on standard cuDNN (dense operations, zeros still computed). Structured 30%: direct $30\%$ FLOPs reduction executed at full cuDNN efficiency. Structured pruning is strictly superior for GPU throughput even at the same compression ratio.

### 2.3 💻 PyTorch: ℓ₁ Filter Pruner

```python
import torch
import torch.nn as nn
from typing import Optional


def l1_filter_saliency(conv: nn.Conv2d) -> torch.Tensor:
    """
    Compute ℓ₁ saliency for each output filter of a Conv2d layer.
    Returns a (C_out,) tensor of filter saliencies.
    """
    # Sum |W| over C_in, k_H, k_W dimensions
    return conv.weight.data.abs().sum(dim=[1, 2, 3])


def prune_conv_filters(
    conv: nn.Conv2d,
    next_conv: Optional[nn.Conv2d],
    next_bn: Optional[nn.BatchNorm2d],
    n_keep: int,
) -> tuple[nn.Conv2d, Optional[nn.Conv2d], Optional[nn.BatchNorm2d]]:
    """
    Prune a Conv2d layer to keep `n_keep` filters (output channels),
    and update the next Conv2d layer's input channels accordingly.

    Returns new (conv, next_conv, next_bn) modules with pruned weights.
    """
    saliency = l1_filter_saliency(conv)
    keep_idx = saliency.topk(n_keep).indices.sort().values

    # Prune current conv: keep only selected output filters
    new_weight = conv.weight.data[keep_idx]
    new_bias = conv.bias.data[keep_idx] if conv.bias is not None else None
    new_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=n_keep,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )
    new_conv.weight.data = new_weight
    if new_bias is not None:
        new_conv.bias.data = new_bias

    # Update BN layer if present (prune running stats and γ/β)
    new_bn = None
    if next_bn is not None:
        new_bn = nn.BatchNorm2d(n_keep)
        new_bn.weight.data = next_bn.weight.data[keep_idx]
        new_bn.bias.data = next_bn.bias.data[keep_idx]
        new_bn.running_mean = next_bn.running_mean[keep_idx]
        new_bn.running_var = next_bn.running_var[keep_idx]

    # Update next conv's input channels
    new_next_conv = None
    if next_conv is not None:
        new_next_conv = nn.Conv2d(
            in_channels=n_keep,
            out_channels=next_conv.out_channels,
            kernel_size=next_conv.kernel_size,
            stride=next_conv.stride,
            padding=next_conv.padding,
            bias=next_conv.bias is not None,
        )
        new_next_conv.weight.data = next_conv.weight.data[:, keep_idx]
        if next_conv.bias is not None:
            new_next_conv.bias.data = next_conv.bias.data

    return new_conv, new_next_conv, new_bn
```

---

## 3. 🎚️ Channel Pruning via Batch Norm Scaling

*Liu, Li, Shen, Huang, Yan, Zhang (2017). "Learning Efficient Convolutional Networks through Network Slimming." ICCV 2017.*

### 3.1 The γ-Sparsity Trick

Batch normalization scales each channel's activations by a learned parameter $\gamma_i$:

$$\hat{a}_i = \gamma_i \cdot \frac{a_i - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}} + \beta_i$$

If $\gamma_i \approx 0$, channel $i$'s contribution to subsequent layers is negligible — it can be removed with little accuracy impact. Network Slimming adds an ℓ₁ sparsity regularizer on all BN $\gamma$ parameters during training:

$$L_{total} = L_{task} + \lambda \sum_{i \in BN} |\gamma_i|$$

This is the *lasso* penalty applied to the BN scaling factors. After training, channels with $|\gamma_i| < \tau$ (threshold) are pruned globally.

**Why BN γ instead of weights directly?** BN γ is a single scalar per channel, making it an ideal "gate." The ℓ₁ regularizer drives small-γ channels to exactly zero (as in LASSO), creating a natural threshold. Pruning a channel requires no architectural assumptions about how weights were organized — any channel with $\gamma \approx 0$ can be removed, regardless of its position.

**The training objective.** Let $\Theta$ be all network parameters and $\Gamma = \{\gamma_i\}$ the BN scaling parameters:

$$\min_{\Theta, \Gamma} L_{task}(\Theta, \Gamma) + \lambda \|\Gamma\|_1$$

The ℓ₁ penalty on $\Gamma$ induces a *sparse* $\hat{\Gamma}$ at convergence. After training, threshold at a global $\tau$ determined by the desired sparsity budget.

> [!NOTE] ℓ₁ on BN γ is a convex relaxation of ℓ₀
> Directly penalizing the number of non-zero channels ($\|\Gamma\|_0$) is NP-hard. The ℓ₁ penalty is the tightest convex relaxation: it promotes sparsity while remaining differentiable everywhere except at zero (where subgradients exist). This is the same argument as LASSO vs. best-subset selection in statistics.

> [!QUESTION] Exercise 2: Effect of λ on channel sparsity
> *This exercise analyzes the tradeoff between regularization strength and sparsity.*
>
> > **Prerequisites:** [[#3.1 The γ-Sparsity Trick|3.1 The γ-Sparsity Trick]]
>
> Consider a BN layer with a single $\gamma$ parameter and task loss $L_{task}(\gamma) = (\gamma - \gamma^*)^2$ (a quadratic proxy centered at optimal value $\gamma^*$).
>
> (a) Write out the full regularized loss $L_{total}(\gamma) = (\gamma - \gamma^*)^2 + \lambda|\gamma|$.
>
> (b) Find the optimal $\hat{\gamma}$ as a function of $\gamma^*$ and $\lambda$. (This is the LASSO soft-thresholding operator.)
>
> (c) For what values of $\lambda$ is $\hat{\gamma} = 0$ (channel pruned)?

> [!TIP]- Solution to Exercise 2
> **Key insight:** The ℓ₁ penalty applies *soft thresholding* — it shifts the optimum toward zero by $\lambda/2$, and sets it exactly to zero when $|\gamma^*| < \lambda/2$.
>
> **(a)** $L_{total}(\gamma) = (\gamma - \gamma^*)^2 + \lambda|\gamma|$
>
> **(b)** Taking the subdifferential and setting it to zero:
> - For $\gamma > 0$: $2(\gamma - \gamma^*) + \lambda = 0 \implies \gamma = \gamma^* - \lambda/2$. Valid when $\gamma^* > \lambda/2$.
> - For $\gamma < 0$: $2(\gamma - \gamma^*) - \lambda = 0 \implies \gamma = \gamma^* + \lambda/2$. Valid when $\gamma^* < -\lambda/2$.
> - For $\gamma = 0$: valid when $|\gamma^*| \leq \lambda/2$.
>
> Soft-thresholding: $\hat{\gamma} = \text{sign}(\gamma^*) \max(|\gamma^*| - \lambda/2, 0)$.
>
> **(c)** $\hat{\gamma} = 0$ when $|\gamma^*| \leq \lambda/2$, i.e., $\lambda \geq 2|\gamma^*|$.

### 3.2 💻 PyTorch: BN γ Channel Pruner

```python
import torch
import torch.nn as nn


def sparsity_regularized_loss(
    model: nn.Module,
    task_loss: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """
    Add ℓ₁ penalty on all BN γ (weight) parameters to the task loss.
    Used as the training loss to drive BN scales toward zero.
    """
    bn_l1 = sum(
        module.weight.abs().sum()
        for module in model.modules()
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d))
    )
    return task_loss + lam * bn_l1


def global_bn_threshold(model: nn.Module, sparsity: float) -> float:
    """
    Find the global γ threshold such that `sparsity` fraction of BN channels
    fall below it. Uses the global distribution of |γ| values.
    """
    all_gamma = torch.cat([
        module.weight.data.abs()
        for module in model.modules()
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d))
    ])
    return float(torch.quantile(all_gamma, sparsity))


def prune_bn_channels(
    model: nn.Module,
    threshold: float,
) -> dict[str, torch.Tensor]:
    """
    Return a dict of channel masks for each BN layer.
    mask[name][i] = False means channel i is pruned (|γ_i| < threshold).
    Does NOT physically remove channels — call rebuild_pruned_model for that.
    """
    masks = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            masks[name] = module.weight.data.abs().ge(threshold)
    return masks


def zero_out_pruned_channels(model: nn.Module, masks: dict[str, torch.Tensor]) -> None:
    """
    Zero out γ and β for pruned channels in-place (soft mask application).
    Zeroed channels produce zero output — effectively removed at inference.
    """
    for name, module in model.named_modules():
        if name in masks:
            mask = masks[name].float()
            module.weight.data.mul_(mask)
            module.bias.data.mul_(mask)
```

---

## 4. 🤖 Attention Head Pruning in Transformers

*Michel, Levy, Neubig (2019). "Are Sixteen Heads Really Better than One?" NeurIPS 2019.*
*Voita, Talbot, Moiseev, Sennrich, Titov (2019). "Analyzing Multi-Head Self-Attention: Specialized Heads do the Heavy Lifting, the Rest can be Pruned." ACL 2019.*

### 4.1 Head Importance Scoring

A multi-head attention layer computes:

$$\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) W^O$$

$$\text{head}_h = \text{softmax}\!\left(\frac{Q W^Q_h (K W^K_h)^\top}{\sqrt{d_k}}\right) V W^V_h$$

**Can we remove heads?** Michel et al. introduce a head importance score based on the sensitivity of the loss to masking head $h$:

$$I_h = \left|\mathbb{E}_{x \sim \mathcal{D}}\left[\frac{\partial L}{\partial \xi_h} \cdot \xi_h\right]\right|$$

where $\xi_h$ is a scalar gate multiplying head $h$'s output. This is the magnitude of the *expected gradient times gate value* — equivalent to the first-order Taylor term for removing the head.

**Key empirical finding:** On WMT En-De, most of BERT's 144 heads can be removed with minimal BLEU drop. The distribution of importance scores is heavy-tailed: a few "specialized" heads carry most of the load.

### 4.2 L0 Gate Pruning

Voita et al. train a differentiable binary gate $z_h \in \{0, 1\}$ for each head using the *Hard Concrete* distribution (a continuous relaxation of discrete gates):

$$z_h \sim \text{HardConcrete}(\alpha_h) \quad \in [0, 1]$$

During training, the attention output is:

$$\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1 \cdot z_1, \ldots, \text{head}_H \cdot z_H) W^O$$

with an ℓ₀ regularizer on the expected number of active heads:

$$L_{total} = L_{task} + \lambda \sum_h \mathbb{E}[z_h]$$

At inference, gates are rounded to $\{0, 1\}$ — any head with $z_h = 0$ is permanently removed.

**Results on WMT En-Ru (6-layer Transformer):** 38 out of 48 encoder heads pruned with only $-0.15$ BLEU. The surviving specialized heads encode *positional*, *syntactic*, and *rare token* information.

> [!QUESTION] Exercise 3: Importance vs. magnitude for heads
> *This exercise contrasts head importance with weight magnitude for Transformer attention.*
>
> > **Prerequisites:** [[#4.1 Head Importance Scoring|4.1 Head Importance Scoring]]
>
> A 4-head attention layer has head outputs $\{a_h\}$ with norms $\|a_1\| = 2.0$, $\|a_2\| = 0.1$, $\|a_3\| = 1.5$, $\|a_4\| = 0.2$. The gradient-gate products (importance scores) are $I_1 = 0.01$, $I_2 = 0.8$, $I_3 = 0.2$, $I_4 = 0.05$.
>
> (a) Which head does magnitude-based pruning remove first? Which does importance-based pruning remove first?
>
> (b) Head 2 has high importance but small output norm. What does this tell you about the information it encodes?

> [!TIP]- Solution to Exercise 3
> **Key insight:** A head with small output norm but high importance encodes critical, difficult-to-redistribute information — it contributes little to the raw activation magnitude but a lot to the loss.
>
> **(a)** Magnitude: remove head 2 (smallest norm $\|a_2\| = 0.1$). Importance: remove head 1 (lowest $I_1 = 0.01$).
>
> **(b)** Head 2 has $\|a_2\| = 0.1$ (small activation) but $I_2 = 0.8$ (high loss sensitivity). This means: even though head 2's output is small in magnitude, the loss is *very sensitive* to removing it. The head likely encodes rare but critical patterns (syntactic relationships, long-range dependencies) that other heads don't. Removing it despite its small magnitude would cause a large accuracy drop. This is the same insight as OBD vs. magnitude pruning: a small-magnitude but high-curvature direction is more important than a large-magnitude flat direction.

### 4.3 💻 PyTorch: Attention Head Masking

```python
import torch
import torch.nn as nn
import math
from typing import Optional


class MaskedMultiheadAttention(nn.Module):
    """
    Multi-head attention with per-head binary masks.
    Masked heads (mask[h] = 0) produce zero output.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Learnable head gates (used for importance-based pruning training)
        self.head_mask = nn.Parameter(torch.ones(n_heads), requires_grad=False)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        H, d_k = self.n_heads, self.d_k

        Q = self.W_q(x).view(B, T, H, d_k).transpose(1, 2)  # (B, H, T, d_k)
        K = self.W_k(x).view(B, T, H, d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, H, d_k).transpose(1, 2)

        # Scaled dot-product attention per head
        scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # (B, H, T, T)
        if attn_mask is not None:
            scores = scores + attn_mask
        attn = scores.softmax(dim=-1)
        out = attn @ V  # (B, H, T, d_k)

        # Apply head mask: zero out pruned heads
        # head_mask: (H,) → broadcast to (1, H, 1, 1)
        out = out * self.head_mask.view(1, H, 1, 1)

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(out)

    def compute_head_importance(
        self,
        loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        Estimate head importance I_h = |E[dL/d(gate_h) * gate_h]|
        by temporarily making head_mask a differentiable gate.
        """
        self.head_mask.requires_grad_(True)
        importance = torch.zeros(self.n_heads, device=device)
        n_batches = 0

        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            out = self(inputs)  # forward with current mask
            loss = criterion(out, targets)
            loss.backward()

            if self.head_mask.grad is not None:
                # I_h ≈ |grad_h * mask_h|
                importance += (self.head_mask.grad * self.head_mask).abs().detach()
                self.head_mask.grad.zero_()
            n_batches += 1

        self.head_mask.requires_grad_(False)
        return importance / n_batches

    def prune_heads(self, n_prune: int) -> list[int]:
        """Prune the `n_prune` least important heads (set mask to 0)."""
        importance = self.compute_head_importance(...)
        _, prune_idx = importance.topk(n_prune, largest=False)
        with torch.no_grad():
            self.head_mask[prune_idx] = 0.0
        return prune_idx.tolist()
```

---

## 5. 🔄 Rethinking the Value of Network Pruning

*Liu, Sun, Zhou, Huang, Darrell (2019). "Rethinking the Value of Network Pruning." ICLR 2019.*

This paper challenges the conventional wisdom that the *inherited weights* of a pruned network are the source of its value.

**Standard view:** Train a large network → prune → fine-tune the *surviving weights*. The value of pruning is that the surviving weights are "better" (they were selected by the training process and survived the saliency filter).

**Liu et al.'s finding:** For structured pruning methods (filter pruning, channel pruning, Network Slimming), fine-tuning the pruned model with its inherited weights offers *no significant advantage* over:
- Training the pruned *architecture* (same structure, same parameter count) from *random initialization*.
- The pruned architecture's performance is determined by its *structure*, not the inherited weights.

**Implication:** Structured pruning is equivalent to *Neural Architecture Search* — it discovers a compact architecture. The training of the unpruned model is just a expensive architecture search procedure, not a source of good weights.

> [!WARNING] This result does NOT apply to unstructured pruning
> The Lottery Ticket Hypothesis (see [[concepts/sparsity-pruning/sparse-training|Sparse Training]]) shows that for *unstructured* pruning, the original initialization matters: rewinding to the initial weights (not random reinit) is key to the lottery ticket's performance. Liu et al.'s result is specific to structured pruning, where the surviving subgraph is a complete standard architecture.

> [!INFO] Contrast with the Lottery Ticket Hypothesis
> The two results are compatible: structured pruning produces a *dense but smaller* architecture where random init works fine. Unstructured pruning produces a *sparse architecture* where the specific initial weights matter (the "lottery ticket"). The key difference is whether the pruning mask respects the computational graph structure (structured) or creates irregular sparsity (unstructured).

---

## 6. 📚 References

| Reference Name | Brief Summary | Link |
|----------------|---------------|------|
| Li et al. (2016). "Pruning Filters for Efficient ConvNets" | ℓ₁-norm filter pruning; genuine wall-clock speedup via dense thinner networks | [arXiv:1608.08710](https://arxiv.org/abs/1608.08710) |
| Liu et al. (2017). "Learning Efficient ConvNets through Network Slimming" | BN γ sparsity regularization; global threshold across all layers | [arXiv:1708.06519](https://arxiv.org/abs/1708.06519) |
| Michel, Levy, Neubig (2019). "Are Sixteen Heads Really Better than One?" | Head importance scoring; empirical head redundancy in BERT and WMT models | [arXiv:1905.10650](https://arxiv.org/abs/1905.10650) |
| Voita et al. (2019). "Analyzing Multi-Head Self-Attention" | L0 Hard Concrete head pruning; 38/48 encoder heads pruned with -0.15 BLEU | [arXiv:1905.09418](https://arxiv.org/abs/1905.09418) |
| Liu et al. (2019). "Rethinking the Value of Network Pruning" | Architecture > weights for structured pruning; fine-tuning ≈ random reinit | [arXiv:1810.05270](https://arxiv.org/abs/1810.05270) |
