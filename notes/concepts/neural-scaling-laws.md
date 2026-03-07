# Neural Scaling Laws

## 1. Motivation

One of the most striking empirical findings in modern deep learning is deceptively simple: if you train language models of varying sizes on varying amounts of data and plot the resulting loss on a log-log scale, you get a straight line. Not approximately straight — straight enough to extrapolate across many orders of magnitude with quantitative accuracy. This is power-law behavior, and it is the central empirical fact that scaling laws are built to explain.

### The Empirical Observation

Let $L$ denote cross-entropy loss (in nats or bits per token) on a held-out test set. Fix everything else — architecture family, optimizer, learning rate schedule — and vary either the number of non-embedding model parameters $N$, the number of training tokens $D$, or the total compute budget $C$. Empirically:

$$L(N) \sim N^{-\alpha_N}, \qquad L(D) \sim D^{-\alpha_D}, \qquad L(C) \sim C^{-\alpha_C}$$

where $\alpha_N, \alpha_D, \alpha_C$ are small positive exponents (e.g., $\alpha_N \approx 0.076$, $\alpha_D \approx 0.095$ in the original Kaplan et al. (2020) work on autoregressive language models). On a log-log plot, these become linear relationships:

$$\log L = -\alpha \log X + \text{const}$$

The fact that this holds across six or more orders of magnitude in $N$ and $D$ — from millions to hundreds of billions of parameters — is not obvious. Neural networks are complicated, nonlinear, high-dimensional systems. There is no a priori reason their generalization loss should follow such a clean functional form.

### Power Laws Are Not Unique to Neural Networks

Power-law distributions appear throughout nature and human systems, which should give us some confidence that we are looking at a genuine structural phenomenon rather than an artifact of our particular experimental setup.

**Zipf's law** (1935) states that in a natural language corpus, the frequency of the $r$-th most common word scales as:

$$f(r) \propto r^{-s}, \qquad s \approx 1$$

The most common word appears roughly twice as often as the second most common, three times as often as the third, and so on. This is a power law in rank.

**Pareto distributions** describe wealth, city populations, earthquake magnitudes (Gutenberg-Richter law), and internet traffic. They arise whenever a multiplicative or preferential-attachment process is at work — systems where "the rich get richer." Formally, the Pareto distribution has survival function $P(X > x) = (x_m/x)^k$ for $x \geq x_m$, $k > 0$, where $x_m > 0$ is the minimum value and $k$ is the shape (tail) exponent.

**Self-organized criticality** (Bak, Tang, Wiesenfeld 1987) provides a physics-grounded explanation: complex systems with many interacting components often evolve toward critical states at which power-law correlations emerge naturally, without fine-tuning.

The recurrence of power laws across such different domains suggests they reflect something fundamental about how complexity accumulates in high-dimensional systems — a property that neural language models, trained on the outputs of human cognition, plausibly share.

### Why This Matters Practically

The practical payoff of scaling laws is enormous: **you can predict loss at scale without running the full experiment.**

If you have established that $L(N) \approx A \cdot N^{-\alpha}$ from experiments at $N = 10^7$ through $N = 10^9$, you can extrapolate to $N = 10^{11}$ with quantifiable uncertainty. This transforms compute budgeting from guesswork into engineering. Before scaling laws, deciding how large to make a model — given a fixed compute budget — required expensive ablations or intuition. With scaling laws, you can:

1. Fit the power-law coefficients from small-scale runs.
2. Project what loss a larger model would achieve.
3. Optimize the allocation of compute between model size and data size (the Chinchilla problem).

This is why scaling laws have become central to frontier model development: they convert empirical observations into actionable predictions, allowing organizations to make principled decisions about where to invest compute before committing to a multi-million-dollar training run.

---

## 2. Mathematical Setup

### The Loss Decomposition Model

Kaplan et al. (2020) propose modeling the loss as a sum of three independent contributions:

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

where:

- $N$ — number of **non-embedding parameters** (attention and MLP weights, excluding token/positional embedding matrices, which scale with vocabulary size rather than model capacity in the relevant sense)
- $D$ — number of **training tokens** seen during optimization
- $\alpha, \beta > 0$ — positive scaling exponents
- $A, B, E > 0$ — positive fitted constants

Each term has a distinct interpretation.

#### The Irreducible Loss $E = L_\infty$

$$E = L_\infty$$

This is the **Bayes error floor** of the task — the entropy of the data-generating process itself. For language modeling, $E$ is approximately the per-token entropy of natural language: the irreducible randomness in what word a human would choose to write next, given perfect knowledge of all preceding context.

No model, regardless of size or training data, can push loss below $E$. A model with $N \to \infty$ parameters trained on $D \to \infty$ tokens would converge to the true conditional distribution $p(x_t \mid x_{<t})$ and achieve exactly $E$ nats per token. The quantity $L - E$ measures **excess loss** — how far a given model is from this theoretical optimum.

For natural English text, estimates of $E$ are in the range of 0.8–1.2 bits per character (or roughly 2–4 nats per token depending on tokenization), reflecting the genuine unpredictability of human writing.

#### The Capacity-Limited Term $A / N^\alpha$

$$\frac{A}{N^\alpha}$$

This term captures the **model capacity bottleneck**. Even with infinite data, a finite model cannot represent the true distribution exactly — it lacks the representational capacity to memorize all relevant statistical regularities. Larger $N$ reduces this term; the exponent $\alpha > 0$ governs how quickly capacity constraints relax as we scale.

Empirically, $\alpha \approx 0.076$ for autoregressive transformers on text (Kaplan et al. 2020). This is a small exponent, meaning you need to scale $N$ by a factor of $10^{1/\alpha} \approx 10^{13}$ to reduce this term by a factor of 10 — scaling is powerful but not magic. Explicitly: if $L(N) \propto N^{-0.076}$, then requiring $L(N')/L(N) = 1/10$ gives $(N'/N)^{0.076} = 10$, so $N'/N = 10^{1/0.076} \approx 10^{13.2}$.

#### The Data-Limited Term $B / D^\beta$

$$\frac{B}{D^\beta}$$

This term captures the **data bottleneck**. A model with finite training data $D$ cannot generalize perfectly — it overfits, or equivalently, it has not seen enough examples to estimate the true distribution well. More training tokens reduce this term; $\beta > 0$ governs the rate.

Empirically, $\beta \approx 0.095$ for the same class of models. The data exponent is somewhat larger than the model exponent, which (in this particular regime) means data is slightly more efficient to scale than parameters — though the Chinchilla work (Hoffmann et al. 2022) significantly revised the practical implications of this.

#### Why the Additive Structure Is Plausible

The additive form $L = E + A/N^\alpha + B/D^\beta$ deserves scrutiny. Why should the capacity-limited and data-limited contributions simply add?

The justification is that each term represents an **independent source of excess loss above the irreducible floor**:

- **Model error** ($A/N^\alpha$): even with a perfect optimizer and infinite data, a finite model makes systematic errors due to limited expressivity. This is an approximation error in the function class.
- **Estimation error** ($B/D^\beta$): even with a perfect model class, finite data means the learned parameters are imperfect estimates of the population optimum. This is a statistical estimation error.

When both sources of error are small (which is the relevant regime when $N$ and $D$ are large), the total excess loss is approximately the sum of the two independent contributions. This is the standard bias-variance-like decomposition from statistical learning theory, adapted to the power-law regime.

More precisely, in the large-$N$, large-$D$ regime where $A/N^\alpha \ll E$ and $B/D^\beta \ll E$, cross-coupling between the capacity and data terms is negligible, and the decomposition holds as a first-order approximation. More formally, if the per-parameter and per-token contributions to excess loss are statistically independent — in the sense that gradient updates from parameter scaling and data scaling affect orthogonal directions in function space — then the joint excess loss decomposes additively.

**Why power laws specifically?** Power-law decay is the simplest **scale-free** functional form. A function $f(x) = Cx^{-\gamma}$ satisfies:

$$f(\lambda x) = \lambda^{-\gamma} f(x)$$

for all $\lambda > 0$ — it has no characteristic scale. Scale-free behavior is the expected outcome when no single length scale dominates the problem. In the context of neural networks, this can be motivated by the observation that the loss surface and the relevant statistical regularities in natural language span many scales simultaneously (from character-level patterns to long-range discourse structure), so no single scale should set the decay rate.

Alternatively, power laws arise naturally as the limit of a sum of exponentials with a distribution of decay rates: if the model learns features at many different "difficulty levels," each contributing an exponentially decaying correction, and if these difficulty levels are distributed as a power law (consistent with Zipf's law for language), then the aggregate improvement from adding more capacity follows a power law. Formally, if $f(x) = \int_0^\infty e^{-\lambda x} \rho(\lambda)\,d\lambda$ and $\rho(\lambda) \sim C_0 \lambda^{\gamma-1}$ as $\lambda \to 0^+$, then by the Tauberian theorem, $f(x) \sim C_0 \Gamma(\gamma)\, x^{-\gamma}$ as $x \to \infty$. Power-law scaling emerges from the distribution of time scales in the system, not from any single mechanism.

### The Compute Approximation $C \approx 6ND$

Training requires not just a model and data, but **compute** — floating-point operations (FLOPs). To reason about the compute-optimal frontier, we need to express $C$ in terms of $N$ and $D$.

#### Deriving the Factor of 6

Consider a single linear layer computing $y = Wx$ where $W \in \mathbb{R}^{m \times n}$ and $x \in \mathbb{R}^n$. Each output element $y_i = \sum_{j=1}^n W_{ij} x_j$ requires $n$ multiplications and $n-1 \approx n$ additions, for a total of $2n$ FLOPs. Across all $m$ output elements:

$$\text{FLOPs}(y = Wx) = 2mn$$

This is $2$ FLOPs per parameter (since $W$ has $mn$ parameters). This factor of 2 — one multiply, one add — is the fundamental unit.

Now consider a full training step on a single token through a transformer with $N$ non-embedding parameters:

**Forward pass:** Each parameter participates in one multiply-add operation per token. Total: $2N$ FLOPs.

**Backward pass — gradient with respect to inputs:** To backpropagate through $y = Wx$, we compute $\frac{\partial \mathcal{L}}{\partial x} = W^\top \frac{\partial \mathcal{L}}{\partial y}$, which is another matrix-vector product of the same size: $2mn$ FLOPs, i.e., $2N$ FLOPs across the full model.

**Backward pass — gradient with respect to weights:** To compute $\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial y} \cdot x^\top$, we perform an outer product of the upstream gradient $\frac{\partial \mathcal{L}}{\partial y} \in \mathbb{R}^m$ with $x \in \mathbb{R}^n$: again $2mn$ FLOPs, i.e., $2N$ FLOPs across the full model.

Summing over the three contributions:

$$C_{\text{per token}} = \underbrace{2N}_{\text{forward}} + \underbrace{2N}_{\nabla_x} + \underbrace{2N}_{\nabla_W} = 6N \text{ FLOPs per token}$$

Over $D$ training tokens, the total compute is:

$$\boxed{C \approx 6ND}$$

#### Caveats and Scope

This approximation:

- **Ignores embedding layers**, which contribute $\mathcal{O}(V \cdot d_{\text{model}})$ FLOPs but are excluded from $N$ by convention, as their cost is dominated by vocabulary size $V$ rather than model depth/width in the scaling sense.
- **Ignores attention's quadratic term** in sequence length: the attention score computation contributes $\mathcal{O}(T^2 d)$ FLOPs per layer (where $T$ is sequence length), which becomes non-negligible for very long contexts but is subdominant for typical training configurations where $T \ll \sqrt{N/L}$ with $L$ layers.
- **Assumes standard dense transformers** — mixture-of-experts, sparse attention, and other architectural variants require separate accounting.
- **Ignores optimizer state** — AdamW maintains first and second moment estimates (2 additional copies of parameters), which affect memory but not FLOPs.

Within these caveats, $C \approx 6ND$ is the standard approximation used throughout the scaling laws literature and is accurate to within a factor of 2 for the transformer architectures and context lengths typical of large-scale language model training.

---

## 3. Kaplan et al. (2020): Fitting the Laws

The landmark Kaplan et al. (2020) paper ("Scaling Laws for Neural Language Models") established the empirical foundation for everything that followed. The core methodology is careful isolation: to measure how loss depends on one variable, you design experiments that make the other variables non-binding constraints.

### 3.1 The Univariate Scaling Laws

The strategy for measuring each univariate law is worth stating precisely, because the experimental design encodes important assumptions.

**Measuring $L(N)$: vary parameters, never run out of data.**

Train models of varying sizes $N$ to full convergence, always on a dataset large enough that the data term $B/D^\beta$ is negligible compared to the model term $A/N^\alpha$. If $D$ is large enough that $B/D^\beta \ll A/N^\alpha$, then $L \approx E + A/N^\alpha$ and the measured loss is essentially a function of $N$ alone. The result is:

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \qquad N_c \approx 8.8 \times 10^{13}, \quad \alpha_N \approx 0.076$$

Here $N_c$ is a fitted constant encoding the overall scale of model capacity; the ratio $(N_c/N)^{\alpha_N}$ is the excess loss attributable to finite model size.

**Measuring $L(D)$: vary data, train for exactly one epoch.**

To isolate the data term, train each model for exactly one epoch — meaning each token is seen exactly once, so $D$ is directly controlled and compute scales linearly with $D$ (not $N$). This prevents the model from overfitting and ensures the data term dominates. Fitting gives:

$$L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \qquad D_c \approx 5.4 \times 10^{13}, \quad \alpha_D \approx 0.095$$

**Measuring $L(C_{\min})$: compute-optimal runs.**

For the compute law, one runs experiments along the compute-efficient frontier — at each compute budget $C$, using the optimal $(N^*, D^*)$ allocation (derived in Section 3.2). The result is:

$$L(C_{\min}) = \left(\frac{C_c}{C_{\min}}\right)^{\alpha_C}, \qquad \alpha_C \approx 0.050$$

**Fitting via log-linear regression.** All three laws are fit by taking logarithms of both sides. For the model-size law:

$$\log L = \alpha_N \log N_c - \alpha_N \log N = \text{const} - \alpha_N \log N$$

This is a linear relationship between $\log L$ and $\log N$, so ordinary least-squares regression in log-log space yields $\alpha_N$ as the (negative) slope and $\log N_c$ from the intercept. The same applies to $L(D)$ and $L(C_{\min})$.

**Interpreting $\alpha_D > \alpha_N$.**

The data exponent $\alpha_D \approx 0.095$ exceeds the parameter exponent $\alpha_N \approx 0.076$. On a log-log plot, a steeper slope means faster loss reduction per unit log-scale investment. Concretely: to halve the excess loss from the data term alone, you need $D' = D \cdot 2^{1/\alpha_D} = D \cdot 2^{10.5} \approx 1480 D$; to halve the excess loss from the model term alone, you need $N' = N \cdot 2^{1/\alpha_N} = N \cdot 2^{13.2} \approx 9600 N$. Data delivers more loss reduction per order-of-magnitude scaling than parameters — at least within the regime Kaplan et al. studied. This observation, however, was later revised by Chinchilla (Hoffmann et al. 2022), which found the two exponents to be much closer to equal when the fitting procedure is tightened.

### 3.2 The Compute-Efficient Frontier

The most practically important result of Kaplan et al. is an answer to the question: **given a fixed compute budget $C$, how should we split between model size $N$ and training tokens $D$?**

The constraint is $C \approx 6ND$, so $D = C/(6N)$. The problem is to minimize loss subject to this constraint.

**Setting up the optimization.** Drop the irreducible floor $E$ (since it does not depend on $N$ or $D$, it does not affect the optimum) and use the two-term approximation:

$$L(N, D) \approx \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

Substitute the constraint $D = C/(6N)$:

$$L(N) = \frac{A}{N^\alpha} + \frac{B \cdot (6N)^\beta}{C^\beta} = \frac{A}{N^\alpha} + \frac{B \cdot 6^\beta}{C^\beta} \cdot N^\beta$$

Now $L$ is a function of $N$ alone, with $C$ entering as a parameter. The first term decreases in $N$ (more parameters reduces model error) and the second term increases in $N$ (for fixed $C$, more parameters means less data, which increases estimation error). There is a unique interior minimum.

**Minimizing over $N$.** Take the derivative and set it to zero:

$$\frac{dL}{dN} = -\frac{\alpha A}{N^{\alpha+1}} + \frac{6^\beta \beta B}{C^\beta} \cdot N^{\beta - 1} = 0$$

Rearrange to isolate the $N$ dependence:

$$\frac{\alpha A}{N^{\alpha+1}} = \frac{6^\beta \beta B}{C^\beta} \cdot N^{\beta-1}$$

$$\alpha A \cdot C^\beta = 6^\beta \beta B \cdot N^{\alpha+1} \cdot N^{\beta-1} = 6^\beta \beta B \cdot N^{\alpha + \beta}$$

$$N^{\alpha + \beta} = \frac{\alpha A \cdot C^\beta}{6^\beta \beta B}$$

Taking both sides to the power $1/(\alpha + \beta)$:

$$\boxed{N^* \propto C^{\,\beta/(\alpha+\beta)}}$$

Since $D^* = C/(6N^*)$:

$$\boxed{D^* = \frac{C}{6N^*} \propto C^{\,\alpha/(\alpha+\beta)}}$$

**Plugging in Kaplan's values.** With $\alpha \approx 0.076$ and $\beta \approx 0.095$:

$$\frac{\beta}{\alpha + \beta} = \frac{0.095}{0.076 + 0.095} = \frac{0.095}{0.171} \approx 0.556$$

$$\frac{\alpha}{\alpha + \beta} = \frac{0.076}{0.171} \approx 0.444$$

**Note:** The two-term substitution above yields exponents $0.556$ and $0.444$. The values $N^* \propto C^{0.73}$ and $D^* \propto C^{0.27}$ are the exponents Kaplan et al. report from numerical fits to the full three-term model — they are taken from the paper, not derived from the algebra above. The discrepancy arises because dropping $E$ in the two-term approximation overstates the marginal value of data (since $D$ must also compensate for $E$), shifting the optimal allocation toward parameters.

Using the paper's reported values:

$$N^* \propto C^{0.73}, \qquad D^* \propto C^{0.27}$$

**Interpretation.** When compute doubles ($C \to 2C$), the optimal model size scales as $2^{0.73} \approx 1.66\times$ while the optimal token count scales only as $2^{0.27} \approx 1.21\times$. Parameters should scale roughly $2.7\times$ faster than data per compute doubling. This prescription says: **scale models aggressively, and data relatively modestly.** GPT-3 (Brown et al. 2020) follows this prescription closely: 175B parameters trained on approximately 300B tokens, a ratio consistent with the Kaplan compute-efficient frontier.

This conclusion would later be contested by Hoffmann et al. (2022), who argued that Kaplan et al.'s fitting procedure — particularly the use of single-epoch data runs to estimate $\beta$ — systematically underestimated the data exponent. The Chinchilla result, with $\alpha \approx \beta$, implies equal scaling of parameters and tokens per compute doubling, overturning the "scale models faster" prescription.

<!-- TODO: sections 4-6 -->
