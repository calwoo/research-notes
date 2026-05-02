# Gradient Checkpointing ♻️

## Table of Contents

- [[#0. Intuition: The Hiker's Dilemma|0. Intuition: The Hiker's Dilemma]]
- [[#1. The Memory Problem|1. The Memory Problem]]
  - [[#1.1 Where Memory Goes During Training|1.1 Where Memory Goes During Training]]
  - [[#1.2 Activation Memory at GPT-2 Scale|1.2 Activation Memory at GPT-2 Scale]]
- [[#2. The Technique|2. The Technique]]
  - [[#2.1 Formal Setup|2.1 Formal Setup]]
  - [[#2.2 Algorithm|2.2 Algorithm]]
- [[#3. Optimal Checkpoint Placement|3. Optimal Checkpoint Placement]]
  - [[#3.1 Cost Model|3.1 Cost Model]]
  - [[#3.2 The sqrt(n) Bound|3.2 The sqrt(n) Bound]]
  - [[#3.3 Pareto Trade-off and the Recursive Extension|3.3 Pareto Trade-off and the Recursive Extension]]
- [[#4. Selective Recomputation|4. Selective Recomputation]]
  - [[#4.1 Activation Cost Asymmetry in Transformers|4.1 Activation Cost Asymmetry in Transformers]]
  - [[#4.2 The Korthikanti et al. Strategy|4.2 The Korthikanti et al. Strategy]]
  - [[#4.3 Memory Breakdown|4.3 Memory Breakdown]]
  - [[#4.4 Results Comparison|4.4 Results Comparison]]
- [[#5. PyTorch Implementation|5. PyTorch Implementation]]
  - [[#5.1 torch.utils.checkpoint API|5.1 torch.utils.checkpoint API]]
  - [[#5.2 use_reentrant: True vs. False|5.2 use_reentrant: True vs. False]]
  - [[#5.3 Wrapping Transformer Blocks|5.3 Wrapping Transformer Blocks]]
- [[#6. Interactions and Gotchas|6. Interactions and Gotchas]]
  - [[#6.1 Dropout and RNG State|6.1 Dropout and RNG State]]
  - [[#6.2 Distributed Data Parallel|6.2 Distributed Data Parallel]]
  - [[#6.3 Mixed Precision and Autocast|6.3 Mixed Precision and Autocast]]
  - [[#6.4 Double Backward|6.4 Double Backward]]
- [[#References|References]]

---

## 0. Intuition: The Hiker's Dilemma 🧭

Imagine a hiker climbing a mountain who needs to retrace their path exactly on the way back down. They have two options: photograph every single step (expensive, heavy backpack), or photograph only a handful of landmarks and trust that they can re-walk each segment between landmarks when needed.

Gradient checkpointing is the second strategy applied to neural network training.

*Backpropagation* needs the activations from every layer of the network to compute gradients — the chain rule requires multiplying local Jacobians together, and each Jacobian depends on the activation at that layer. Naively, one stores all activations from the forward pass and reads them back during the backward pass.

The key observation: if you have the input to a segment of computation, you can always rerun that segment to regenerate its internal activations. You don't need to store them — you just need to store enough *checkpoints* (snapshots at strategic positions) that no segment is too expensive to re-traverse. This trades some extra computation for a large reduction in peak memory.

> [!NOTE] Terminology
> The technique goes by several names in the literature: *gradient checkpointing*, *activation recomputation*, *rematerialization*, and *trading compute for memory*. The terms are largely synonymous. PyTorch calls the utility `torch.utils.checkpoint`; the Chen et al. (2016) paper uses "sublinear memory cost."

---

## 1. The Memory Problem 💾

### 1.1 Where Memory Goes During Training

During training, GPU memory holds four distinct categories of tensors:

1. **Parameters** $\theta$: the model weights, size $\approx P$ bytes (where $P$ counts parameters times bytes-per-parameter).
2. **Gradients** $\nabla_\theta \mathcal{L}$: same shape as parameters, so another $\approx P$ bytes.
3. **Optimizer state**: e.g. Adam maintains first and second moment estimates, adding $2P$ more bytes for a total of $3P$ bytes of optimizer state.
4. **Activations**: the intermediate tensors produced during the forward pass and needed for backpropagation.

For a transformer with $L$ layers, batch size $B$, sequence length $T$, and hidden dimension $d$, activation memory per layer consists of:

- Attention queries, keys, values: $3 \times B \times T \times d$ elements
- Attention weight matrix: $B \times H \times T \times T$ elements (where $H$ = number of heads)
- MLP intermediate activations: $\sim 4 \times B \times T \times d$ elements (for a $4d$ expansion)
- LayerNorm inputs and outputs: $\sim 2 \times B \times T \times d$ elements

Summing over $L$ layers, total activation memory scales as $O(BLTd)$, with an additional $O(BLHT^2)$ term from attention weights.

> [!WARNING]
> *The $O(BLHT^2)$ attention weight term grows quadratically in sequence length. For long-context models ($T \gg d/H$), this term dominates activation memory and checkpointing attention weights becomes especially valuable.*

### 1.2 Activation Memory at GPT-2 Scale

For concreteness, take GPT-2 Small ($L = 12$, $d = 768$, $H = 12$, $T = 1024$, $B = 8$) with fp32 (4 bytes/element).

**Parameter and optimizer memory (independent of batch/sequence):**

| Component | Count | Memory (fp32) |
|---|---|---|
| Embedding weights | $50257 \times 768$ | 154 MB |
| All transformer weights | $\approx 117$M params | 468 MB |
| Gradients | same as params | 468 MB |
| Adam moments ($\times 2$) | same as params | 936 MB |
| **Parameter + optimizer total** | | **1,872 MB** |

**Activation memory per layer (fp32, $B=8$, $T=1024$):**

| Activation Tensor | Shape | Memory |
|---|---|---|
| Q, K, V (each) | $8 \times 1024 \times 768$ | 25.2 MB |
| Attention weights | $8 \times 12 \times 1024 \times 1024$ | 402 MB |
| MLP intermediate | $8 \times 1024 \times 3072$ | 100.7 MB |
| LayerNorm inputs ($\times 2$) | $2 \times 8 \times 1024 \times 768$ | 50.3 MB |
| **Per-layer total** | | **$\approx 603$ MB** |

**Total activation memory across all 12 layers: $\approx 7.2$ GB.**

**Total parameter + optimizer memory: $\approx 1.9$ GB.**

**Activations dominate by roughly 4:1 at training time.** This ratio grows linearly with $B$ and $T$, making large-batch or long-sequence training infeasible without intervention. At inference time, no activations need to be stored across layers, so parameters dominate and the ratio inverts entirely.

> [!EXAMPLE] The Scaling Pressure
> Increasing batch size from 8 to 64 multiplies activation memory by $8\times$ to $\approx 58$ GB — far exceeding a single A100's 80 GB, and entirely consumed by activations. Parameters remain at 1.9 GB. This asymmetry is what makes gradient checkpointing indispensable for large-batch training.

> [!QUESTION] Exercise 1: Peak Activation Memory
> *This exercise makes the memory dominance of activations concrete at a larger scale.*
>
> > **Prerequisites:** [[#1.2 Activation Memory at GPT-2 Scale|1.2 Activation Memory at GPT-2 Scale]]
>
> Consider GPT-2 Medium ($L = 24$, $d = 1024$, $H = 16$, $T = 1024$, $B = 16$) in bf16 (2 bytes/element).
>
> (a) Compute total activation memory without checkpointing.
>
> (b) Compute total activation memory with full checkpointing (one checkpoint per layer, so each layer's internal activations are discarded and only the layer output is stored).
>
> (c) What fraction of peak memory does (b) save, and what does the memory now consist of?

> [!TIP]- Solution to Exercise 1
> **Key insight:** Full checkpointing reduces stored activations to one tensor per layer boundary — the layer output — discarding all intermediate tensors within a layer.
>
> **Sketch:**
>
> **(a) Without checkpointing:**
>
> Per-layer activations (bf16, $B=16$, $T=1024$, $d=1024$, $H=16$):
> - Q, K, V each: $16 \times 1024 \times 1024 \times 2$ bytes $= 33.6$ MB each, so $100.7$ MB total
> - Attention weights: $16 \times 16 \times 1024 \times 1024 \times 2 = 536.9$ MB
> - MLP intermediate ($4d = 4096$): $16 \times 1024 \times 4096 \times 2 = 134.2$ MB
> - LayerNorm inputs ($\times 2$): $67.1$ MB
>
> Per-layer total $\approx 839$ MB. Across 24 layers: $\approx \mathbf{20.1}$ **GB**.
>
> **(b) With full checkpointing:**
>
> Only the layer output is stored: $16 \times 1024 \times 1024 \times 2 \approx 33.6$ MB per layer.
> Total: $24 \times 33.6 \approx \mathbf{806}$ **MB**.
>
> **(c) Savings:**
>
> Reduction from 20.1 GB to 0.8 GB — roughly $25\times$ reduction. The remaining memory is almost entirely parameters, gradients, and optimizer state (roughly 1.2B params $\times$ 10 bytes/param for Adam bf16 $\approx 12$ GB), with activations now negligible.

---

## 2. The Technique 🔬

### 2.1 Formal Setup

**Definition (Computation Graph).** A forward pass defines a directed acyclic graph $G = (V, E)$ where each node $v \in V$ is an intermediate tensor and each edge $(u, v) \in E$ means $v$ depends on $u$. Backpropagation traverses $G$ in reverse topological order, computing $\partial \mathcal{L}/\partial v$ for each $v$ using stored values of the tensors on which $v$ depends.

**Definition (Checkpoint).** A *checkpoint* is a designated node $c \in V$ whose value is retained in memory across the full forward-backward pass. All non-checkpoint nodes are discarded after they are produced in the forward pass.

**Definition (Segment).** Given a sequence of checkpoints $c_0, c_1, \ldots, c_K$ (where $c_0$ is the input and $c_K$ is the loss), the $i$-th *segment* $S_i$ is the subgraph of $G$ between consecutive checkpoints $c_{i-1}$ and $c_i$. The segment $S_i$ has a well-defined recomputation: starting from $c_{i-1}$, one can re-execute the forward pass of $S_i$ to regenerate all internal tensors.

### 2.2 Algorithm

The full gradient checkpointing procedure is:

```python
import torch
from typing import Callable, List

def checkpointed_forward_backward(
    segments: List[Callable],
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Forward pass with gradient checkpointing.
    
    Args:
        segments: list of K callables, each representing one segment S_i.
                  Segment i maps checkpoint c_{i-1} to checkpoint c_i.
        x:        initial input (checkpoint c_0).
    
    Returns:
        Output of the final segment (checkpoint c_K).
    """
    checkpoints = [x]  # store c_0

    # --- Forward pass: run each segment, store only the output checkpoint ---
    with torch.no_grad():
        for segment in segments:
            x = segment(x)
            checkpoints.append(x)  # store c_i; internal tensors of S_i are discarded

    # --- Backward pass: for each segment, recompute internals then differentiate ---
    loss = checkpoints[-1]
    loss.requires_grad_(True)

    for i in reversed(range(len(segments))):
        c_prev = checkpoints[i]
        c_prev.requires_grad_(True)

        # Rerun segment i under autograd to rebuild the computation graph
        with torch.enable_grad():
            c_next_recomputed = segments[i](c_prev)

        # Backpropagate through the recomputed segment
        c_next_recomputed.backward(checkpoints[i + 1].grad)

    return checkpoints[-1]
```

> [!NOTE] Simplified Presentation
> The pseudocode above strips away PyTorch's saved-tensor hooks, RNG state management, and autocast contexts (all handled by `torch.utils.checkpoint` internally) to expose the core logic. In practice, one calls `torch.utils.checkpoint.checkpoint(segment, c_prev)` rather than implementing this manually.

The key structural property: at any point during the backward pass, memory holds only the $K$ checkpoint tensors plus the activations of the single segment currently being recomputed. Peak memory during backward is therefore:

$$M_{\text{backward}} = \underbrace{K \cdot a_c}_{\text{checkpoint tensors}} + \underbrace{\max_i M(S_i)}_{\text{largest segment recomputation}}$$

where $a_c$ is the (uniform) size of a checkpoint tensor and $M(S_i)$ is the peak activation memory needed to recompute segment $S_i$.

---

## 3. Optimal Checkpoint Placement 📐

### 3.1 Cost Model

Consider a sequential network of $n$ layers, each producing an activation of uniform size $a$ (bytes or elements). We place $K - 1$ checkpoints, dividing the network into $K$ equal segments of $n/K$ layers each.

**Memory cost** has two components:

1. *Checkpoint storage*: $K$ checkpoints (including input and output), so $K \cdot a$.
2. *Segment recomputation*: within any segment of $n/K$ layers, activations are produced one by one; peak activation count within a segment is $n/K$, so the recomputation buffer costs $(n/K) \cdot a$.

Total peak memory:

$$M(K) = K \cdot a + \frac{n}{K} \cdot a = a\left(K + \frac{n}{K}\right)$$

**Recomputation cost**: each of the $K$ segments is re-executed once during backward, and each segment costs $n/K$ layer evaluations, so total extra forward-pass work is:

$$C(K) = K \cdot \frac{n}{K} = n \quad \text{layer evaluations}$$

*Recompute cost is always $O(n)$ regardless of $K$, i.e., one extra forward pass through the whole network.* Only the memory cost depends on $K$.

> [!INFO] Why Recompute Cost is Always O(n)
> Each of the $K$ segments is exactly $n/K$ layers long, and each is recomputed exactly once during the backward pass. Total work $= K \times (n/K) = n$, independent of $K$. This is why gradient checkpointing is described as "one extra forward pass per mini-batch."

### 3.2 The sqrt(n) Bound

To minimize $M(K) = a(K + n/K)$ over $K > 0$, differentiate:

$$\frac{dM}{dK} = a\left(1 - \frac{n}{K^2}\right) = 0 \implies K^* = \sqrt{n}$$

The second derivative $2an/K^3 > 0$ confirms this is a minimum. Substituting $K^* = \sqrt{n}$:

$$M(K^*) = a\left(\sqrt{n} + \frac{n}{\sqrt{n}}\right) = a \cdot 2\sqrt{n} = O(\sqrt{n} \cdot a)$$

**With $\sqrt{n}$ uniformly spaced checkpoints, peak activation memory is $O(\sqrt{n} \cdot a)$, reduced from $O(n \cdot a)$, at a recompute cost of exactly one extra forward pass.**

> [!EXAMPLE] Concrete Numbers for a 96-Layer Transformer
> $n = 96$ layers, activation size $a = 600$ MB per layer.
>
> | Strategy | Checkpoints $K$ | Memory | Extra FLOPs |
> |---|---|---|---|
> | Store everything | 96 | $96 \times 600 = 57.6$ GB | 0 |
> | Optimal $K = \sqrt{96} \approx 10$ | 10 | $2\sqrt{96} \times 600 \approx 11.8$ GB | $1 \times$ forward |
> | No storage | 0 | $600$ MB | $96 \times$ forward |
>
> The $\sqrt{n}$ strategy achieves an $\approx 5\times$ memory reduction at the cost of one extra forward pass — the recomputation is ~30% runtime overhead for a typical transformer.

#### Optimality of Uniform Spacing

For sequential networks with *equal* activation sizes, uniform checkpoint spacing is optimal. The argument: given fixed $K$, memory $M(K)$ is minimized when the maximum segment length is minimized, which occurs when segments are equal. Any non-uniform partition increases the maximum segment length and hence increases the recomputation buffer term $\max_i M(S_i)$.

> [!WARNING]
> *Uniform spacing is only optimal under the equal-activation-size assumption. For transformers, attention blocks produce $O(T^2)$ activations (the attention weight matrix) while MLP blocks produce $O(Td)$ activations. In this setting, the optimal checkpoint placement skews toward denser checkpoints around attention layers. See Exercise 4.*

> [!QUESTION] Exercise 2: Optimal K for a Weighted Trade-off
> *This exercise generalizes the $\sqrt{n}$ result to a parametric cost objective.*
>
> > **Prerequisites:** [[#3.2 The sqrt(n) Bound|3.2 The sqrt(n) Bound]]
>
> Suppose you are willing to trade memory and compute with a parameter $\lambda > 0$. Define the combined cost:
>
> $$\text{Cost}(K) = M(K) + \lambda \cdot C(K) = a\left(K + \frac{n}{K}\right) + \lambda \cdot n$$
>
> (a) Find the optimal $K^*(\lambda)$ that minimizes $\text{Cost}(K)$.
>
> (b) Notice that $\lambda \cdot n$ is a constant in $K$. What does this imply about the optimal $K$?
>
> (c) Now suppose instead the recompute cost is $C(K) = K^2$ (as would be the case if each segment were recomputed $K$ times, not once). Find the optimal $K^*$ in this new setting and interpret the result.

> [!TIP]- Solution to Exercise 2
> **Key insight:** When recomputation cost is $O(n)$ regardless of $K$, the optimal checkpoint count is unaffected by $\lambda$. The trade-off is between memory regimes ($K$ small vs. large), but the memory-minimizing $K$ is always $\sqrt{n}$.
>
> **Sketch:**
>
> **(a)** $\text{Cost}(K) = a(K + n/K) + \lambda n$. Differentiating in $K$: $a(1 - n/K^2) = 0 \implies K^* = \sqrt{n}$, independent of $\lambda$.
>
> **(b)** The $\lambda n$ term is constant in $K$, so it shifts the objective up but does not change the minimizer. This confirms that when recompute cost is $O(n)$ for all $K$, the memory-minimizing checkpoint count $K^* = \sqrt{n}$ is also the cost-minimizing count for any $\lambda$.
>
> **(c)** With $C(K) = K^2$: $\text{Cost}(K) = a(K + n/K) + \lambda K^2$. Differentiate: $a(1 - n/K^2) + 2\lambda K = 0$. This is a cubic in $K$ with no closed-form solution in general. For large $\lambda$, the $2\lambda K$ term dominates and pushes $K^*$ smaller (fewer checkpoints, less recompute), at the expense of more memory per segment. For $\lambda \to 0$, we recover $K^* \to \sqrt{n}$.

> [!QUESTION] Exercise 3: Proof of Uniform Spacing Optimality
> *This exercise establishes rigorously that equal-length segments minimize peak memory for fixed K.*
>
> > **Prerequisites:** [[#3.2 The sqrt(n) Bound|3.2 The sqrt(n) Bound]]
>
> Let a sequential network have $n$ layers with uniform activation size $a$. Fix the number of checkpoints at $K$. Let the segment lengths be $\ell_1, \ell_2, \ldots, \ell_K$ with $\sum_{i=1}^K \ell_i = n$ and $\ell_i \geq 1$.
>
> Peak memory during backward pass is:
>
> $$M = K \cdot a + a \cdot \max_i \ell_i$$
>
> Prove that $M$ is minimized when all $\ell_i = n/K$ (assume $n/K$ is an integer), and compute the minimum.

> [!TIP]- Solution to Exercise 3
> **Key insight:** $M = Ka + a \cdot \max_i \ell_i$. The term $Ka$ is fixed. To minimize $M$, minimize $\max_i \ell_i$ subject to $\sum_i \ell_i = n$, $\ell_i \geq 1$.
>
> **Sketch:**
>
> Suppose the partition is not uniform: then some segment $j$ has $\ell_j > n/K$. By definition $\max_i \ell_i \geq \ell_j > n/K$.
>
> For the uniform partition $\ell_i = n/K$ for all $i$, we have $\max_i \ell_i = n/K$.
>
> Any non-uniform partition has $\max_i \ell_i > n/K$ (since if all segments were $\leq n/K$ and at least one were strictly less, the sum $\sum_i \ell_i < n$, contradicting the constraint). Therefore the uniform partition is the unique minimizer.
>
> Minimum memory: $M^* = Ka + a(n/K) = a(K + n/K)$, as in the main derivation.

### 3.3 Pareto Trade-off and the Recursive Extension

The $K$-checkpoint family traces a Pareto curve in (memory, recompute) space:

```python
import numpy as np
import matplotlib.pyplot as plt

def memory_cost(K: int, n: int, a: float = 1.0) -> float:
    """Peak activation memory with K checkpoints (in units of a)."""
    return K + n / K  # = a*(K + n/K) / a

def recompute_cost(K: int, n: int) -> float:
    """Extra forward evaluations with K checkpoints."""
    return n  # always O(n) for this model

n = 96  # e.g. 96 transformer layers
K_values = np.arange(1, n + 1)
memories = [memory_cost(K, n) for K in K_values]
recomputes = [recompute_cost(K, n) for K in K_values]

# Minimum is at K = sqrt(n)
K_opt = int(np.sqrt(n))
print(f"Optimal K = {K_opt}, memory = {memory_cost(K_opt, n):.1f}a, extra compute = {n} layers")
```

Since recompute cost is flat at $O(n)$ for all $K$, the Pareto curve degenerates: the $\sqrt{n}$ checkpoint placement is simultaneously memory-optimal and recompute-optimal among all $K > 0$.

**Recursive extension.** Chen et al. (2016) also derive a recursive variant: apply checkpointing hierarchically within each segment. This reduces memory to $O(\log n \cdot a)$ at the cost of $O(n \log n)$ extra compute. Concretely, with $k = 1$ checkpoint per segment and $\lceil \log_2 n \rceil$ levels of recursion, memory is $O(\log n)$.

> [!INFO] When to Use the Recursive Extension
> The $O(\log n)$ scheme is rarely used in practice because $O(n \log n)$ recompute overhead is prohibitive for large models ($\sim 33\%$ overhead from the $O(n)$ scheme is already significant). It is included in Chen et al. (2016) as a theoretical curiosity and for very memory-constrained scenarios.

---

## 4. Selective Recomputation 🎯

### 4.1 Activation Cost Asymmetry in Transformers

The uniform-cost assumption of Section 3 breaks down for transformers because different activation tensors have very different size-to-recompute-cost ratios.

**Definition (Recomputation Efficiency).** For an activation tensor of size $s$ bytes, let $f$ be the FLOPs required to recompute it from the nearest preceding checkpoint. The *recomputation efficiency* is $s/f$: bytes saved per FLOP of recomputation. Higher values mean it is cheaper (per byte) to recompute.

In a transformer block with sequence length $T$, hidden dimension $d$, and $H$ attention heads:

| Activation | Size | FLOPs to Recompute | Recompute Efficiency |
|---|---|---|---|
| Attention weight matrix $\text{softmax}(QK^T/\sqrt{d_h})$ | $O(BHT^2)$ | $O(BHT^2 d_h)$ | $O(1/d_h)$ — cheap! |
| Dropout mask (attention) | $O(BHT^2)$ bits | requires full attention forward | — |
| QKV projection inputs | $O(BTd)$ | requires LayerNorm + projection | moderate |
| LayerNorm output | $O(BTd)$ | cheap (elementwise) | very cheap |
| MLP input | $O(BTd)$ | requires full attention block | expensive |

*Attention weights* (the $T \times T$ softmax output) are large but cheap to recompute: the compute cost is $O(T^2 d_h)$ while the memory cost is $O(T^2)$, so efficiency scales as $O(1/d_h)$. *MLP inputs*, by contrast, require rerunning the entire preceding attention block to regenerate.

> [!WARNING]
> *The asymmetry is especially pronounced for long sequences. At $T = 8192$, the attention weight matrix is $8192^2 \times \text{bytes} \approx 256$ MB per head per batch element in fp16. For GPT-3 scale ($H = 96$ heads), this is $\approx 24$ GB per layer — overwhelmingly the dominant activation.*

### 4.2 The Korthikanti et al. Strategy

Korthikanti et al. (2022) introduce *selective activation recomputation*: store only the activations that are expensive to recompute; recompute the ones that are cheap (large but cheap).

**Definition (Selective Recomputation Policy).**

- **Recompute** (do not store): $QK^T$ matrix product, softmax output, softmax dropout output, weighted value aggregation $\text{softmax}(\cdot) V$.
- **Store** (retain in memory): LayerNorm inputs before self-attention, QKV projection inputs, MLP layer inputs, dropout masks for non-attention paths.

The rationale for recomputing attention activations: they account for the $O(T^2)$ memory but can be regenerated from the stored Q, K, V projections with only $O(T^2 d_h)$ work — which is the same work done during the original forward pass of the attention sublayer.

> [!NOTE] Sequence Parallelism Synergy
> Korthikanti et al. (2022) pair selective recomputation with *sequence parallelism*: partitioning activations along the sequence dimension for operations outside tensor-parallel regions (LayerNorms, dropouts). Sequence parallelism alone reduces activation memory by $\approx 50\%$; combined with selective recomputation, the total reduction is $\approx 5\times$. The two techniques are largely orthogonal.

### 4.3 Memory Breakdown

For a single transformer layer without any parallelism ($s =$ sequence length, $b =$ batch size, $h =$ hidden dimension, $a =$ number of attention heads), activation memory (in bytes, bf16) is:

| Region | Formula | Notes |
|---|---|---|
| Attention block | $11sbh + 5as^2b$ bytes | $11sbh$ for QKV, projections; $5as^2b$ for attention weights |
| MLP block | $19sbh$ bytes | Linear layers + activation function |
| LayerNorms ($\times 2$) | $4sbh$ bytes | Input tensors for both LNs |
| **Total per layer** | $sbh(34 + 5as/h)$ bytes | |

The $5as^2b / h$ term — from attention weights — is the $O(T^2)$ component that selective recomputation targets.

With selective recomputation (discarding attention weight tensors and related intermediates), the $5as^2b$ term is eliminated from stored memory. The remaining stored activations cost $sbh \cdot 34$ bytes per layer, independent of sequence length.

### 4.4 Results Comparison

The three regimes, at large scale (Megatron-LM 22B–530B parameter range):

| Strategy | Memory Saved vs. Baseline | Extra Compute | Notes |
|---|---|---|---|
| No checkpointing | 0% (baseline) | 0% | Training infeasible at large scale |
| Full checkpointing | $\sim 80\%$ | $\sim 30$–$40\%$ | One extra forward pass per segment |
| Selective recomputation + seq. parallel | $\sim 80\%$ ($\approx 5\times$ reduction) | $4$–$7\%$ (22B), $\sim 2\%$ (530B–1T) | Recomputes only cheap attention activations |

**Selective recomputation achieves comparable memory savings to full checkpointing at less than 10% compute overhead, compared to 30–40% for full checkpointing.** This is the central quantitative result of Korthikanti et al. (2022).

> [!QUESTION] Exercise 4: Non-Uniform Activation Sizes
> *This exercise analyzes how the $O(\sqrt{n})$ bound degrades when activation sizes are heterogeneous.*
>
> > **Prerequisites:** [[#3.2 The sqrt(n) Bound|3.2 The sqrt(n) Bound]], [[#4.3 Memory Breakdown|4.3 Memory Breakdown]]
>
> Suppose a transformer has $n$ layers total: $n/2$ attention layers each with activation size $a_{\text{attn}} = c \cdot T^2$ and $n/2$ MLP layers each with activation size $a_{\text{mlp}} = T \cdot d$, where $c > 0$ is a constant and $T \gg d$.
>
> (a) If you apply uniform checkpointing with $K = \sqrt{n}$ checkpoints equally spaced (so checkpoints alternate between attention and MLP layers), what is the peak activation memory of the worst-case segment?
>
> (b) Compare this to the optimal placement for this non-uniform case. What is the optimal checkpoint density ratio between attention and MLP layers, and what memory does it achieve?
>
> (c) At what sequence length $T$ does the non-uniform optimal placement become essential (i.e., when does the uniform-spacing memory exceed the non-uniform-spacing memory by more than a factor of 2)?

> [!TIP]- Solution to Exercise 4
> **Key insight:** Non-uniform activation sizes break the optimality of uniform spacing. The worst-case segment memory is dominated by the large attention activations, which can be reduced by placing checkpoints more densely around attention layers.
>
> **(a) Uniform spacing with $K = \sqrt{n}$:**
>
> Each segment contains $n/K = \sqrt{n}$ layers. Since layers alternate, each segment has $\sqrt{n}/2$ attention layers and $\sqrt{n}/2$ MLP layers. The worst-case segment activation memory is roughly the sum of all activations within one segment:
>
> $$M_{\text{seg}} \approx \frac{\sqrt{n}}{2} \cdot cT^2 + \frac{\sqrt{n}}{2} \cdot Td = \frac{\sqrt{n}}{2}(cT^2 + Td)$$
>
> For $T \gg d/c$, this is $\approx \frac{\sqrt{n}}{2} \cdot cT^2$.
>
> Total peak memory: $K \cdot a_{\text{checkpoint}} + M_{\text{seg}}$, where checkpoint tensors have size $\max(cT^2, Td)$. Dominant term: $O(\sqrt{n} \cdot cT^2)$.
>
> **(b) Optimal non-uniform placement:**
>
> Equalize segment memory by placing one checkpoint after every attention layer (since those are expensive) and one checkpoint after every $\approx cT^2 / Td = cT/d$ MLP layers. This requires $K_{\text{attn}} = n/2$ checkpoints for attention and $K_{\text{mlp}} = (n/2)/(cT/d) = nd/(2cT)$ for MLP — but $K_{\text{attn}}$ is just full storage of attention activations! In practice, one applies selective recomputation (Section 4.2) instead.
>
> The memory ratio between uniform and optimal grows as $O(T)$ for large $T$ — confirming that uniform spacing becomes increasingly suboptimal at long contexts.
>
> **(c) Break-even:** The uniform strategy exceeds the optimal by factor 2 when $\sqrt{n}/2 \cdot cT^2 > 2 \cdot \sqrt{n}/2 \cdot Td$, i.e., $cT > 2d$, i.e., $T > 2d/c$. For typical $d = 4096$ and $c \sim 1$, this is $T > 8192$.

---

## 5. PyTorch Implementation 💻

### 5.1 torch.utils.checkpoint API

The primary entry point is `torch.utils.checkpoint.checkpoint`:

```python
torch.utils.checkpoint.checkpoint(
    function,           # callable: the segment to checkpoint
    *args,              # positional inputs to function
    use_reentrant=None, # bool: controls implementation variant (see below)
    context_fn=noop_context_fn,  # callable returning a pair of context managers
    determinism_check='default', # 'default' | 'none'
    debug=False,
    early_stop=True,    # non-reentrant only: stop recompute early when possible
    **kwargs,           # keyword arguments (non-reentrant only)
)
```

The function returns the same output as `function(*args, **kwargs)` but with lower peak activation memory.

### 5.2 use_reentrant: True vs. False

The `use_reentrant` flag selects between two fundamentally different implementations.

**`use_reentrant=True` (legacy, reentrant variant):**

The forward pass is run under `torch.no_grad()`, so no autograd graph is recorded for the checkpointed region. During backward, PyTorch re-executes the function in a nested ("reentrant") backward context to reconstruct the computation graph on the fly.

Limitations:
- Requires at least one input and one output to have `requires_grad=True`.
- Cannot handle `detach()` or `torch.no_grad()` inside the checkpointed function — these confuse the reentrant backward.
- Does not support `torch.autograd.grad()` (only `.backward()`).
- Incompatible with double backward (second-order gradients).

**`use_reentrant=False` (recommended, non-reentrant variant):**

The forward pass records the autograd graph normally via *saved-tensor hooks*: instead of saving intermediate tensors, hooks are registered that trigger recomputation on demand when a saved tensor is accessed during backward.

Advantages:
- No `requires_grad` constraint on inputs.
- Handles detached tensors, nested checkpoints, keyword arguments.
- Supports `torch.autograd.grad()`.
- Compatible with double backward (with `preserve_rng_state=True`).
- `early_stop=True` (default): recomputation halts as soon as all needed tensors have been regenerated — avoids recomputing the full segment when only early activations are needed.

*Starting in PyTorch 2.9, omitting `use_reentrant` raises an error — always pass it explicitly.*

### 5.3 Wrapping Transformer Blocks

**Pattern 1: Checkpoint at the transformer block level (most common)**

```python
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention sublayer
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + attn_out)
        # FFN sublayer
        x = self.ln2(x + self.ffn(x))
        return x


class Transformer(nn.Module):
    def __init__(self, n_layers: int, d_model: int, n_heads: int, ffn_dim: int,
                 use_checkpointing: bool = True):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_dim) for _ in range(n_layers)
        ])
        self.use_checkpointing = use_checkpointing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if self.use_checkpointing:
                # Checkpoint the entire transformer block.
                # use_reentrant=False is recommended for modern PyTorch.
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x
```

**Pattern 2: Checkpoint every other block (partial checkpointing)**

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    for i, layer in enumerate(self.layers):
        # Checkpoint even-indexed layers only: sqrt(n) memory with n/2 recomputes
        if self.use_checkpointing and i % 2 == 0:
            x = checkpoint(layer, x, use_reentrant=False)
        else:
            x = layer(x)
    return x
```

**Pattern 3: torch.compile compatibility**

`torch.utils.checkpoint` is compatible with `torch.compile`. The compiler understands recomputation semantics and can fuse operations across segment boundaries:

```python
model = Transformer(n_layers=24, d_model=1024, n_heads=16, ffn_dim=4096,
                    use_checkpointing=True)

# torch.compile works with checkpointing; no special flags needed.
compiled_model = torch.compile(model)
```

> [!NOTE] torch.compile and RNG State
> Under `torch.compile`, the `preserve_rng_state` flag in `checkpoint()` is ignored — the compiler always preserves RNG state automatically. This means dropout behavior is always deterministic across the recomputed forward pass, even if you pass `preserve_rng_state=False`.

---

## 6. Interactions and Gotchas ⚠️

### 6.1 Dropout and RNG State

*Dropout* is stochastic: on the original forward pass, a random mask is sampled. On the recomputed forward pass during backward, the same mask must be applied — otherwise gradients are computed with respect to a different function than was used to produce the output, silently corrupting gradient estimates.

PyTorch's checkpointing handles this via RNG state saving:

```python
# Pseudocode of what checkpoint() does internally (simplified):
import torch

def _checkpointed_segment(fn, *args):
    # Before original forward: save RNG state
    rng_state_cpu = torch.get_rng_state()
    rng_state_cuda = torch.cuda.get_rng_state()  # if on CUDA

    # Original forward (under no_grad for reentrant, or normal for non-reentrant)
    output = fn(*args)

    # ... store rng_state_cpu, rng_state_cuda alongside the checkpoint ...

    # During recompute (backward):
    # Restore RNG state so dropout samples the same mask
    torch.set_rng_state(rng_state_cpu)
    torch.cuda.set_rng_state(rng_state_cuda)
    output_recomputed = fn(*args)
    # Restore RNG state to current (post-recompute) state
    # so subsequent random ops are unaffected
```

This is controlled by the `preserve_rng_state` argument (default `True`). Setting `preserve_rng_state=False` disables RNG saving and restoring — *only* do this if the checkpointed segment contains no stochastic operations or if determinism is not required.

> [!DANGER] Forgetting About Dropout in Custom Checkpointing
> If you implement gradient checkpointing manually (e.g. by saving only certain tensors and rerunning segments yourself), you must replicate the RNG save/restore logic. Forgetting this causes incorrect gradients with no error message — PyTorch will compute valid-looking but wrong gradients silently.

### 6.2 Distributed Data Parallel

Gradient checkpointing is safe with *Distributed Data Parallel* (DDP). Each rank independently performs its forward and backward pass, and checkpointing operates entirely within each rank's local computation — there is no interaction with the `AllReduce` communication primitives that DDP uses for gradient synchronization.

```python
model = Transformer(n_layers=24, d_model=1024, n_heads=16, ffn_dim=4096,
                    use_checkpointing=True)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
# DDP + checkpointing: safe, no special configuration needed.
```

> [!WARNING]
> *The reentrant variant (`use_reentrant=True`) has known issues with DDP and FSDP in some configurations, related to the nested backward pass interfering with DDP's gradient hooks. Prefer `use_reentrant=False` in distributed settings.*

### 6.3 Mixed Precision and Autocast

PyTorch's `torch.autocast` context manager casts operations to lower precision (e.g. bf16 or fp16) based on a dtype dispatch table. When a checkpointed segment is recomputed during backward, it must execute within the same autocast context as the original forward pass — otherwise operations that were bf16 in the forward become fp32 in the recompute, causing shape or dtype mismatches in the gradient computation.

`torch.utils.checkpoint` handles this automatically: it captures the autocast state (enabled/disabled, dtype) at the start of the original forward pass and restores it during recomputation.

```python
# Correct: autocast wraps the entire forward, including checkpointed regions.
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(x)
    loss = criterion(output, target)
loss.backward()
# The recomputed segments during backward() also run under bfloat16 autocast.
```

> [!WARNING]
> *If you manually manage autocast (e.g. enter/exit the context inside a custom `forward` method), ensure the autocast context is active when `checkpoint()` is called. A common mistake: wrapping only the loss computation in autocast but not the model forward — this causes the checkpoint to save a non-autocast state and recompute in fp32.*

### 6.4 Double Backward

*Double backward* (computing gradients of gradients, as needed for MAML, higher-order optimization, or Hessian-vector products) requires that the backward pass itself be differentiable — i.e. that the computation graph produced during backward can itself be differentiated.

The reentrant variant (`use_reentrant=True`) is fundamentally incompatible with double backward: the nested backward pass runs under `no_grad`, so no graph is recorded for the recomputed activations.

The non-reentrant variant (`use_reentrant=False`) supports double backward when `preserve_rng_state=True` (the default), because it records the autograd graph during recomputation via saved-tensor hooks, which are themselves differentiable.

```python
# Double backward: requires use_reentrant=False
x = checkpoint(layer, x, use_reentrant=False, preserve_rng_state=True)

# Compute first-order gradient
grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

# Compute second-order gradient (Hessian-vector product, etc.)
grad_norm = sum(g.pow(2).sum() for g in grads)
grad_norm.backward()  # This works with use_reentrant=False
```

> [!DANGER]
> Using `use_reentrant=True` with double backward will either raise an error or silently produce incorrect second-order gradients depending on the PyTorch version. Always use `use_reentrant=False` for any higher-order differentiation.

---

## References

| Reference Name | Brief Summary | Link to Reference |
|---|---|---|
| Chen et al. (2016) — Training Deep Nets with Sublinear Memory Cost | Introduces the $O(\sqrt{n})$ checkpoint placement result and general algorithm for trading memory for compute in deep network training | [arXiv:1604.06174](https://arxiv.org/abs/1604.06174) |
| Korthikanti et al. (2022) — Reducing Activation Recomputation in Large Transformer Models | Introduces selective activation recomputation and sequence parallelism for transformers; achieves 5x memory reduction at <10% compute overhead | [arXiv:2205.05198](https://arxiv.org/abs/2205.05198) |
| PyTorch — torch.utils.checkpoint documentation | Official API reference for `torch.utils.checkpoint.checkpoint`, covering `use_reentrant`, RNG state, and compatibility notes | [docs.pytorch.org](https://docs.pytorch.org/docs/2.11/checkpoint.html) |
| PyTorch Blog — How Activation Checkpointing Enables Scaling Up Training | Practical walkthrough of PyTorch's checkpointing implementation and `use_reentrant` semantics | [medium.com/pytorch](https://medium.com/pytorch/how-activation-checkpointing-enables-scaling-up-training-deep-learning-models-7a93ae01ff2d) |
| ar5iv — Chen et al. HTML rendering | HTML version of arXiv:1604.06174 for reading derivations inline | [ar5iv.labs.arxiv.org](https://ar5iv.labs.arxiv.org/html/1604.06174) |
