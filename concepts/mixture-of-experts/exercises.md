# Mixture of Experts: Exercises

## Table of Contents

- [[#Derivation Problems|Derivation Problems]]
  - [[#Problem 1 Soft Gating Gradient and the Expert Weighting Update|Problem 1: Soft Gating Gradient and the Expert Weighting Update]]
  - [[#Problem 2 Top-k Gating as a Constrained Argmax|Problem 2: Top-k Gating as a Constrained Argmax]]
  - [[#Problem 3 Load Balancing Loss Derivation|Problem 3: Load Balancing Loss Derivation]]
  - [[#Problem 4 Expert Capacity and Token Drop Probability|Problem 4: Expert Capacity and Token Drop Probability]]
  - [[#Problem 5 Expert-Choice Routing as Bipartite Matching|Problem 5: Expert-Choice Routing as Bipartite Matching]]
- [[#Conceptual Questions|Conceptual Questions]]
  - [[#Problem 6 MoE vs Ensemble vs Product of Experts|Problem 6: MoE vs. Ensemble vs. Product of Experts]]
  - [[#Problem 7 Why Top-1 Routing Works|Problem 7: Why Top-1 Routing Works]]
  - [[#Problem 8 Expert Collapse and the Positive Feedback Loop|Problem 8: Expert Collapse and the Positive Feedback Loop]]
  - [[#Problem 9 MoE Scaling Laws vs Dense Scaling Laws|Problem 9: MoE Scaling Laws vs. Dense Scaling Laws]]
  - [[#Problem 10 Distributed Expert Parallelism|Problem 10: Distributed Expert Parallelism]]
- [[#Implementation Sketches|Implementation Sketches]]
  - [[#Problem 11 Noisy Top-k Gating|Problem 11: Noisy Top-k Gating]]
  - [[#Problem 12 Expert Dispatch with Capacity Enforcement|Problem 12: Expert Dispatch with Capacity Enforcement]]
  - [[#Problem 13 Router Z-Loss Numerically Stable Implementation|Problem 13: Router Z-Loss Numerically Stable Implementation]]

---

## Derivation Problems

### Problem 1: Soft Gating Gradient and the Expert Weighting Update

The soft MoE output is $y = \sum_{i=1}^E g_i(x) f_i(x)$ where $g_i(x) = \operatorname{softmax}(W_g^\top x)_i$ and $f_i(x) = W_i x$ (linear experts for simplicity). Let $\mathcal{L}$ be a scalar loss with $\partial \mathcal{L}/\partial y$ given.

(a) Compute $\partial \mathcal{L}/\partial W_g$ via the chain rule. Show the gradient has the form of a weighted sum over experts, where the weights depend on the difference between each expert's output $f_i(x)$ and the ensemble output $y$.

(b) Show that at a stationary point (gradient with respect to $W_g$ equal to zero), the gating weights $g_i(x)$ are characterized by a fixed-point condition involving $f_i(x)$ and $y$. In particular, identify which experts receive zero gradient pressure from this term.

(c) The soft gating gradient is dense: all $E$ experts receive a gradient signal on every forward pass. Contrast with hard top-$k$ gating, where only the $k$ selected experts receive a gradient through the softmax weights. Assuming expert outputs are i.i.d. random variables with mean $\mu$ and variance $\sigma^2$, derive the variance of the Monte Carlo gradient estimator for the gating weights in each case as a function of $E$, $k$, and $\sigma^2$.

---

### Problem 2: Top-k Gating as a Constrained Argmax

Define $\operatorname{KeepTopK}(h, k)_i = h_i$ if $h_i$ is among the $k$ largest values in $h$, else $-\infty$. The hard gating probabilities are $g = \operatorname{softmax}(\operatorname{KeepTopK}(W_g^\top x, k))$.

(a) Show that as $k \to E$ (all experts selected), $g \to \operatorname{softmax}(W_g^\top x)$, recovering soft gating. Show that as $k \to 1$, $g \to \operatorname{one-hot}(\arg\max_i (W_g^\top x)_i)$.

(b) The KeepTopK operation is not differentiable at inputs where two logits are equal (a boundary set of measure zero). For inputs where all logits are distinct, show that the gradient of the selected logits with respect to $W_g$ is well-defined and has the same form as the soft gating gradient restricted to the selected top-$k$ experts.

(c) Show that the straight-through estimator — treating the selected set $\mathcal{T}(x)$ as fixed during backpropagation and passing gradients only through the selected experts' softmax logits — is equivalent to computing the exact gradient of a modified objective where unselected experts have their logits clamped to $-\infty$ before the softmax.

---

### Problem 3: Load Balancing Loss Derivation

Let $\mathcal{B}$ be a batch of $T$ tokens. Define:

$$\operatorname{Importance}_i(\mathcal{B}) = \sum_{x \in \mathcal{B}} g_i(x)$$

$$\operatorname{CV}(v)^2 = \frac{\operatorname{Var}(v)}{\operatorname{Mean}(v)^2}$$

The importance loss is $\mathcal{L}_{\operatorname{imp}} = w_{\operatorname{imp}} \cdot \operatorname{CV}(\operatorname{Importance}(\mathcal{B}))^2$.

(a) Show that $\mathcal{L}_{\operatorname{imp}} = 0$ if and only if all experts have equal total gating weight across the batch. Show that $\mathcal{L}_{\operatorname{imp}} \geq 0$ always and is unbounded above.

(b) The Switch Transformer replaces this with:

$$\mathcal{L}_{\text{aux}} = \alpha E \sum_{i=1}^{E} f_i \cdot P_i$$

where $f_i = \frac{1}{T}\sum_{x \in \mathcal{B}} \mathbf{1}[i^*(x) = i]$ and $P_i = \frac{1}{T}\sum_{x \in \mathcal{B}} g_i(x)$. Show that this loss is minimized when $f_i = P_i = 1/E$ for all $i$ (uniform routing), and compute the value of $\mathcal{L}_{\text{aux}}$ at this minimum.

(c) Compute the gradient $\partial \mathcal{L}_{\text{aux}} / \partial g_i(x)$ treating $f_i$ as a stop-gradient constant. Show that this gradient upweights soft probabilities for over-loaded experts and downweights them for under-loaded experts — directly counteracting the positive feedback loop of routing collapse.

---

### Problem 4: Expert Capacity and Token Drop Probability

With $E$ experts, batch size $T$, top-$k$ routing, and capacity factor $\phi$, the capacity of each expert is $C = \lfloor (kT/E) \cdot \phi \rfloor$ tokens.

(a) Under perfectly uniform routing (each of the $kT$ expert-token selections is distributed uniformly over experts), show that the expected number of tokens sent to each expert is exactly $kT/E$. For $\phi < 1$, compute the expected overflow fraction: the expected fraction of tokens that would be dropped per expert under uniform load.

(b) Suppose token routing is i.i.d. with each of the $T$ tokens independently assigned to expert $i$ with probability $p = k/E$ (as in top-1 uniform routing). The number of tokens routed to expert $i$ follows $\operatorname{Binomial}(T, k/E)$. Using the normal approximation to the binomial, derive an expression for the probability that expert $i$ overflows (receives more than $C$ tokens) as a function of $T$, $E$, $k$, and $\phi$.

(c) Using the CLT-based approximation from (b), analyze the behavior of the overflow probability as $T \to \infty$ with $E$, $k$, $\phi$ fixed. Show that:
  - For $\phi > 1$: overflow probability $\to 0$.
  - For $\phi = 1$: overflow probability $\to 1/2$.

  Conclude that $\phi > 1$ is necessary and sufficient for reliable zero-overflow behavior at large batch size under uniform routing.

---

### Problem 5: Expert-Choice Routing as Bipartite Matching

In expert-choice routing, each expert $i$ selects its top-$m$ preferred tokens (where $m = \phi T / E$) rather than each token selecting its top-$k$ experts.

(a) Formulate expert-choice routing as a bipartite $b$-matching problem: tokens on one side, experts on the other; edge weights $s_{t,i} = \operatorname{softmax}(W_g^\top x_t)_i$; each expert matches exactly $m$ tokens; each token can be matched to at most $k'$ experts. Write the integer linear program (ILP) for this problem.

(b) Show that the expert-choice selection (each expert independently takes its top-$m$ tokens by score) solves the LP relaxation of the ILP from (a) when the per-token constraint is $k' = \infty$ (tokens can be matched to any number of experts). Hence expert-choice is LP-optimal under this relaxed constraint.

(c) Define the coverage of a batch as $\operatorname{Coverage} = \frac{1}{T}|\{t : \exists i,\ t \in \operatorname{TopM}(i)\}|$. Show that $\operatorname{Coverage} \leq 1$ always, and that it can be strictly less than 1 (some tokens may not be selected by any expert). Under a simple uniform-random model where each expert selects each token independently with probability $m/T$, derive the expected coverage as a function of $T$, $E$, and $m$.

---

## Conceptual Questions

### Problem 6: MoE vs. Ensemble vs. Product of Experts

(a) A standard committee ensemble averages predictions from $E$ independently trained models: $y = \frac{1}{E}\sum_i f_i(x)$. MoE uses a learned input-dependent gating: $y = \sum_i g_i(x) f_i(x)$ with $\sum_i g_i(x) = 1$. Identify the key structural difference. Under what condition on $g$ does MoE reduce exactly to a committee ensemble?

(b) The Product of Experts (PoE) model defines $p(y|x) \propto \prod_i p_i(y|x)^{g_i(x)}$. Contrast PoE with MoE in terms of: (i) how disagreement between experts is handled (multiplication vs. averaging), (ii) what each model learns to do when experts disagree strongly on a particular input, and (iii) the computational cost of inference in each model.

(c) In MoE with top-1 routing, only one expert processes each token. This induces a partition of the input space: expert $i$ "owns" the region $\mathcal{R}_i = \{x : i = \arg\max_j (W_g^\top x)_j\}$. Show that these regions form a Voronoi-like partition with respect to the linear classifier $W_g^\top x$. What does this say about the geometric structure of expert specialization under top-1 routing?

---

### Problem 7: Why Top-1 Routing Works

The Switch Transformer paper demonstrates that top-1 routing ($k=1$) matches or exceeds top-2 ($k=2$) routing on many benchmarks, despite using half the expert FLOPs per token. Give three distinct explanations:

(a) **Specialization argument**: With $k=1$, each expert receives a more homogeneous distribution of inputs (it only receives tokens for which it is the strict top-1 choice). Argue that more homogeneous inputs allow each expert to specialize more deeply. What would "sharper" expert weights look like, and why does homogeneous input distribution encourage them?

(b) **Gradient purity argument**: Under top-1 routing, the gradient for expert $i$'s parameters comes only from tokens routed exclusively to it (with the full gate weight $g_i \approx 1$). Under top-2 routing, expert $i$ also receives gradient from tokens where it is the second-choice expert (with lower gate weight $g_i$). Argue why this mixing of high-confidence and low-confidence routing decisions makes the top-2 gradient noisier.

(c) **Load balancing argument**: Show that for fixed total inference FLOPs, top-1 routing with $E$ experts provides twice the routing resolution of top-2 routing with $E/2$ experts (each token chooses among more experts). Explain why higher routing resolution is a benefit even when the same total expert computation is performed.

---

### Problem 8: Expert Collapse and the Positive Feedback Loop

Expert collapse is the failure mode where one or a few experts receive nearly all tokens while the rest are starved of gradient signal.

(a) Formalize the positive feedback loop: suppose expert $i$ is initialized with a slightly higher logit for most inputs. Under top-1 routing, trace through the chain of events — from initial logit advantage, to more gradient updates, to improved parameters, to further increased logits — that leads to collapse. At which step does the feedback become self-reinforcing?

(b) The router z-loss is $\mathcal{L}_z = \frac{1}{T}\sum_x (\log \sum_i e^{h_i(x)})^2$. Compute $\partial \mathcal{L}_z / \partial h_i(x)$ and show it equals $\frac{2}{T} \cdot \log Z(x) \cdot g_i(x)$ where $Z(x) = \sum_j e^{h_j(x)}$. Interpret: for which expert (dominant or underdog) is the gradient largest in magnitude, and why is this the correct direction?

(c) Entropy regularization adds $-\lambda H(g(x))$ to the loss (penalizing low entropy to encourage uniform routing). Compute $\partial(-H(g))/\partial h_i(x)$ and show it equals $g_i(x)(\log g_i(x) + H(g(x)))$. Identify the sign of this gradient for an expert with above-average weight. Contrast entropy regularization with z-loss: which operates on the pre-softmax logits $h$ and which on the post-softmax distribution $g$? Explain why z-loss is computationally cheaper to implement alongside the routing computation.

---

### Problem 9: MoE Scaling Laws vs. Dense Scaling Laws

(a) A dense model with $N$ active parameters costs $C \propto N$ FLOPs per token. An MoE model with $EN$ total parameters and top-$k$ routing activates $kN$ parameters per token (same FLOPs as a dense model with $kN$ parameters), but has access to $EN$ total parameters. At matched FLOPs per token, MoE has $E/k$ times more parameters than the dense baseline. Argue that MoE achieves lower loss than this FLOP-matched dense baseline if and only if experts genuinely specialize (i.e., the extra parameters carry non-redundant information). What would redundant experts look like?

(b) Empirically, the MoE advantage over FLOP-matched dense models narrows at very large scale. Propose and justify a hypothesis for why the benefit diminishes as the dense model scale grows: consider the role of the data distribution, the definition of the "data-limited regime," and whether additional expert parameters have useful signal to specialize on.

(c) Define the "effective parameter count" of an MoE model as the number of parameters in a dense model that would achieve the same validation loss. Based on the scaling arguments in (a) and (b), argue that this effective count lies strictly between $kN$ (FLOP-matched dense) and $EN$ (total MoE parameters). What does this bound say about the routing efficiency — do experts fully utilize the capacity of their parameter budget?

---

### Problem 10: Distributed Expert Parallelism

In expert parallelism, each of the $E$ experts is placed on a different device. For a batch of $T$ tokens processed with hidden dimension $d$:

(a) Describe the all-to-all communication pattern required at a single MoE layer. Before the MoE layer: each device holds all tokens for its local portion of the batch, but must dispatch each token to the device holding its assigned expert. After the MoE layer: expert outputs must be returned to the originating device. Show that this requires two all-to-all collectives per MoE layer, and bound the bytes communicated per device per collective.

(b) Compare the communication cost of expert parallelism to tensor parallelism, where each device holds a row- or column-shard of the weight matrix and communicates activations via all-reduce. For a two-device tensor-parallel split of a linear layer with weight $W \in \mathbb{R}^{d \times d}$ and batch $T$, derive the bytes communicated per device. Identify the regime (in terms of $T$ and $d$) where expert parallelism dominates over tensor parallelism in communication cost, and vice versa.

(c) Token dropping introduces a subtle gradient bias in distributed training: dropped tokens still have labels and are counted in the loss denominator, but their hidden states do not pass through any expert — so they contribute zero gradient to expert parameters. Explain why this systematically under-trains experts that are frequently at capacity. Propose one practical mitigation strategy and argue whether it fully resolves the bias.

---

## Implementation Sketches

### Problem 11: Noisy Top-k Gating

Sketch the full forward pass of noisy top-$k$ gating as used in Shazeer et al. (2017).

(a) **Inputs and data structures**: Specify the inputs ($x$, $W_g$, $W_n$, $k$) with their shapes, and list the intermediate tensors that must be stored for the backward pass.

(b) **Noisy logit computation**: Write pseudocode that: (i) computes clean logits $z = W_g^\top x$; (ii) computes per-expert learned noise scales $s = \operatorname{Softplus}(W_n^\top x)$; (iii) samples noise $\epsilon \sim \mathcal{N}(0, I_E)$; (iv) forms noisy logits $H = z + \epsilon \odot s$.

(c) **Top-k selection and sparse softmax**: Write pseudocode that applies KeepTopK to $H$, then computes the softmax over non-$-\infty$ entries only. Note the numerical handling required: entries set to $-\infty$ must be masked before exponentiation to avoid NaN propagation.

(d) **Expert dispatch and aggregation**: Write pseudocode for dispatching $x$ to the $k$ selected experts, collecting their outputs, and computing the weighted sum $y = \sum_{i \in \mathcal{T}(x)} g_i(x) f_i(x)$. Indicate what must be saved for the backward pass.

---

### Problem 12: Expert Dispatch with Capacity Enforcement

Sketch the token dispatch algorithm for a single MoE layer with $E$ experts, per-expert capacity $C$, and top-1 routing on a batch of $T$ tokens.

(a) **Routing**: Describe how to compute routing scores $S \in \mathbb{R}^{T \times E}$ and select the top-1 expert assignment for each token. State the shape of the assignment index tensor.

(b) **Capacity enforcement**: For each expert $i$, describe how to sort its assigned tokens by routing score (descending), retain the first $C$, and flag the remainder as dropped. Write pseudocode for this step.

(c) **Dispatch and scatter**: Describe how to gather the kept tokens for each expert into a packed tensor of shape $(E, C, d)$ (zero-padded for experts with fewer than $C$ tokens), apply each expert's FFN, and scatter outputs back to the original token positions (writing zeros for dropped tokens). Write pseudocode with shape annotations.

(d) Identify explicitly where gradients flow and where they are blocked in this algorithm. In particular: do dropped tokens contribute gradient to the router weights? Do they contribute gradient to the expert FFN weights?

---

### Problem 13: Router Z-Loss Numerically Stable Implementation

The router z-loss is $\mathcal{L}_z = \frac{1}{T}\sum_{t=1}^T (\log \sum_{i=1}^E e^{h_i(x_t)})^2$ where $h_i(x_t) = (W_g^\top x_t)_i$.

(a) **Numerically stable forward pass**: The log-sum-exp is computed stably as $\operatorname{LSE}(h) = m + \log\sum_i e^{h_i - m}$ where $m = \max_i h_i$. Write complete pseudocode for the forward pass that computes $\mathcal{L}_z$ over a batch of $T$ tokens without numerical overflow.

(b) **Efficient backward pass**: Show that the gradient $\partial \mathcal{L}_z / \partial h_i(x_t) = \frac{2}{T} \cdot \operatorname{LSE}(h(x_t)) \cdot g_i(x_t)$ where $g_i(x_t) = \operatorname{softmax}(h(x_t))_i$. Write pseudocode for the backward pass that reuses the cached LSE values and softmax probabilities from the forward pass (which are also needed for routing), avoiding redundant computation.

(c) **Combined loss**: Write pseudocode for the full combined training objective $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \alpha \mathcal{L}_{\text{aux}} + \beta \mathcal{L}_z$. Identify the hyperparameters $\alpha$ and $\beta$, state their typical values from the Switch Transformer and ST-MoE papers, and discuss the trade-off each controls.
