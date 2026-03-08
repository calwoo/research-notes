# Neural Scaling Laws

## Table of Contents

1. [Motivation](#1-motivation)
   - [The Empirical Observation](#the-empirical-observation)
   - [Power Laws Are Not Unique to Neural Networks](#power-laws-are-not-unique-to-neural-networks)
   - [Why This Matters Practically](#why-this-matters-practically)
2. [Mathematical Setup](#2-mathematical-setup)
   - [The Loss Decomposition Model](#the-loss-decomposition-model)
   - [The Compute Approximation C ≈ 6ND](#the-compute-approximation-c-approx-6nd)
3. [Kaplan et al. (2020): Fitting the Laws](#3-kaplan-et-al-2020-fitting-the-laws)
   - [3.1 The Univariate Scaling Laws](#31-the-univariate-scaling-laws)
   - [3.2 The Compute-Efficient Frontier](#32-the-compute-efficient-frontier)
4. [Chinchilla (2022): The IsoFLOP Revolution](#4-chinchilla-2022-the-isoflop-revolution)
   - [4.1 The IsoFLOP Methodology](#41-the-isoflop-methodology)
   - [4.2 Analytical Derivation via Lagrangian Optimization](#42-analytical-derivation-via-lagrangian-optimization)
   - [4.3 The 20 Tokens Per Parameter Rule](#43-the-20-tokens-per-parameter-rule)
   - [4.4 Why Kaplan's Exponents Were Biased](#44-why-kaplans-exponents-were-biased)
5. [Fitting Methodology](#5-fitting-methodology)
   - [5.1 Approach 1 — IsoFLOP Minimum Fitting](#51-approach-1--isoflop-minimum-fitting)
   - [5.2 Approach 2 — Parametric Global Fit](#52-approach-2--parametric-global-fit)
   - [5.3 Approach 3 — Per-Model-Size Estimation](#53-approach-3--per-model-size-estimation)
   - [5.4 Cross-Validation of Approaches](#54-cross-validation-of-approaches)
   - [5.5 Alternative Estimators](#55-alternative-estimators)
6. [Scope and Limitations](#6-scope-and-limitations)
   - [Upstream vs. Downstream Performance](#upstream-vs-downstream-performance)
   - [Non-Power-Law Scaling](#non-power-law-scaling)
   - [Trend Breaks and Extrapolation Risk](#trend-breaks-and-extrapolation-risk)
   - [Modality and Architecture Dependence](#modality-and-architecture-dependence)
   - [Scaling for Recommendation Systems and Ranking Models](#scaling-for-recommendation-systems-and-ranking-models)
7. [References](#references)
   - [Further Reading: Scaling for Recommendation Systems and Ranking](#further-reading-scaling-for-recommendation-systems-and-ranking)

---

## 1. Motivation

One of the most striking empirical findings in modern deep learning is deceptively simple: if you train language models of varying sizes on varying amounts of data and plot the resulting loss on a log-log scale, you get a straight line. Not approximately straight — straight enough to extrapolate across many orders of magnitude with quantitative accuracy. This is power-law behavior, and it is the central empirical fact that scaling laws are built to explain.

### The Empirical Observation

This observation did not originate with language models. Hestness et al. (2017) established, across machine translation, language modeling, image classification, and speech recognition, that generalization error follows a power law in training set size:

$$\varepsilon(m) \propto \alpha_0 \, m^{\beta_g}$$

with domain-specific exponents $\beta_g$ ranging from $-0.07$ (word language models) to $-0.35$ (image classification, top-5 error). Crucially, they found that model architecture improvements shift the loss curve downward — reducing the constant $\alpha_0$ — but do not alter the power-law exponent $\beta_g$. The slope in log-log space is a property of the task and data distribution, not of the model family. This observation provides an early empirical argument that scaling exponents are universal characteristics of the data manifold, not architectural artifacts.

The Kaplan et al. (2020) and Hoffmann et al. (2022) results then brought this phenomenon into the large-scale language model regime and provided a quantitative framework for compute-optimal training. Let $L$ denote cross-entropy loss (in nats or bits per token) on a held-out test set. Fix everything else — architecture family, optimizer, learning rate schedule — and vary either the number of non-embedding model parameters $N$, the number of training tokens $D$, or the total compute budget $C$. Empirically:

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

#### The Three-Regime Structure

The power-law model $L = E + A/N^\alpha + B/D^\beta$ is an approximation that is accurate only in a middle regime. Training loss across scale exhibits three qualitatively distinct regions, first explicitly identified by Hestness et al. (2017):

1. **Random-guessing plateau** — at very small $N$ or $D$, the model is near chance performance. Loss is roughly constant and high. The power-law terms have not yet engaged.
2. **Power-law region** — the regime where $L = E + A/N^\alpha + B/D^\beta$ is accurate. Loss decreases smoothly as a power law in scale. This is the regime studied by Kaplan and Chinchilla.
3. **Convergence plateau** — at very large $N$ or $D$, loss approaches $E$ asymptotically. The power-law terms are negligible and the model is effectively at the Bayes floor.

The simple decomposition model is not designed to capture transitions between these regimes. This matters for extrapolation: fitting the model in the power-law region and extrapolating into the convergence plateau will overestimate the benefit of further scaling. More complex functional forms (see Section 5.5) are needed to model the full curve.

#### The Intrinsic Dimension of the Data Manifold

The scaling exponents $\alpha$ and $\beta$ are not universal constants — they depend on the structure of the data distribution. Sharma and Kaplan (2020) and Bahri et al. (2021) showed that the scaling exponent is inversely proportional to the intrinsic dimension $d$ of the data manifold:

$$\alpha \propto \frac{1}{d}$$

The intuition: if the data lies on a low-dimensional manifold (e.g., natural images, which have strong local structure), then adding model capacity quickly captures the dominant directions of variation, and loss decreases fast (small $d$, large $\alpha$). If the data manifold is high-dimensional (e.g., diverse text spanning many domains), each additional unit of capacity captures only a small slice of the variation, and loss decreases slowly (large $d$, small $\alpha$).

This provides theoretical grounding for why $\alpha \approx 0.076$ is small for language: natural language is extremely high-dimensional, with structure ranging from phonetics to long-range discourse. It also predicts that scaling exponents should differ across modalities — a prediction confirmed empirically (vision transformers, acoustic models, and RL agents all exhibit different exponents).

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

---

## 4. Chinchilla (2022): The IsoFLOP Revolution

Hoffmann et al. (2022) — colloquially "Chinchilla" — overturned the Kaplan prescription for compute-optimal training. The central finding: **given a fixed compute budget, parameters and tokens should scale equally.** The practical corollary: most large models at the time of publication (including GPT-3, Gopher, and MT-NLG) were significantly undertrained. This section develops the methodology and the mathematics behind that conclusion.

### 4.1 The IsoFLOP Methodology

Chinchilla's key innovation is experimental design. Rather than varying $N$ while fixing $D$ (as Kaplan did to measure $L(N)$) or vice versa, they **fix the total compute $C$ and sweep $N$ and $D$ jointly along the constraint curve $6ND = C$.**

The procedure is:

1. Choose a set of compute budgets $\{C_1, C_2, \ldots\}$ spanning several orders of magnitude.
2. For each $C_i$, train many models at different $(N, D)$ pairs satisfying $6ND = C_i$. These are **IsoFLOP curves** — each curve holds compute constant and trades parameters for tokens.
3. Each run uses all $D$ tokens in a single pass (no multi-epoch training), so the comparison is clean: every model on the curve $6ND = C_i$ uses exactly $C_i$ FLOPs.
4. Record the final loss for each run and find $(N^*, D^*)$ that minimizes it on each IsoFLOP curve.
5. Fit the relationship between $C_i$ and $(N^*_i, D^*_i)$ across curves.

**Why this eliminates undertrained-model bias.** Kaplan's univariate $L(N)$ experiments train to convergence on a fixed dataset; their univariate $L(D)$ experiments train for exactly one epoch with a fixed model size. Neither condition corresponds to the regime of a real large-scale training run, where both $N$ and $D$ are chosen simultaneously under a compute budget. Specifically:

- When Kaplan trains to convergence to measure $L(N)$, larger models require more data to converge — but $D$ is held fixed, so large models never see enough data. The loss attributable to model size is inflated by data starvation, causing $\alpha_N$ to be underestimated.
- When Kaplan trains for one epoch to measure $L(D)$, small models are used to keep compute manageable. The data exponent is measured in a regime where model capacity is the bottleneck, not data, distorting $\beta$.

By contrast, Chinchilla's IsoFLOP curves compare every model on a **compute-equal footing**: a 7B-parameter model trained on $\sim$143B tokens competes directly against a 70B model trained on $\sim$14B tokens, both using the same total FLOPs. The optimal point emerges naturally from the loss surface rather than from extrapolation of univariate fits.

### 4.2 Analytical Derivation via Lagrangian Optimization

The formal setup is to minimize $L(N, D) = E + A/N^\alpha + B/D^\beta$ subject to the compute constraint $6ND = C$. This is a constrained optimization problem with one equality constraint, handled by the method of Lagrange multipliers.

**Forming the Lagrangian.**

$$\mathcal{L}(N, D, \lambda) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta} + \lambda(6ND - C)$$

The term $\lambda(6ND - C)$ enforces the constraint at the optimum: at any critical point, $6ND - C = 0$ automatically.

**First-order conditions.** Differentiate with respect to $N$ and set to zero:

$$\frac{\partial \mathcal{L}}{\partial N} = -\frac{\alpha A}{N^{\alpha+1}} + 6\lambda D = 0 \implies \lambda = \frac{\alpha A}{6 D N^{\alpha+1}}$$

Differentiate with respect to $D$ and set to zero:

$$\frac{\partial \mathcal{L}}{\partial D} = -\frac{\beta B}{D^{\beta+1}} + 6\lambda N = 0 \implies \lambda = \frac{\beta B}{6 N D^{\beta+1}}$$

**Eliminating $\lambda$.** Set the two expressions for $\lambda$ equal:

$$\frac{\alpha A}{6 D N^{\alpha+1}} = \frac{\beta B}{6 N D^{\beta+1}}$$

Multiply both sides by $6 D N^{\alpha+1} \cdot N D^{\beta+1}$:

$$\alpha A \cdot N D^{\beta+1} = \beta B \cdot D N^{\alpha+1}$$

Cancel one factor of $N$ from the right and one factor of $D$ from the left:

$$\alpha A \cdot D^{\beta} = \beta B \cdot N^{\alpha}$$

This is the **optimality condition**: at the compute-optimal point, the marginal loss reduction from adding one more parameter exactly equals the marginal loss reduction from seeing one more token (appropriately weighted by their respective constants).

Rearranging:

$$\frac{D^\beta}{N^\alpha} = \frac{\beta B}{\alpha A} \qquad \text{(constant, independent of } C\text{)}$$

**Solving for the scaling exponents.** From the optimality condition, express $D$ as a function of $N$:

$$D^\beta = \frac{\beta B}{\alpha A} \cdot N^\alpha \implies D = \left(\frac{\beta B}{\alpha A}\right)^{1/\beta} N^{\alpha/\beta}$$

Substitute into the compute constraint $6ND = C$:

$$6N \cdot \left(\frac{\beta B}{\alpha A}\right)^{1/\beta} \cdot N^{\alpha/\beta} = C$$

$$6 \left(\frac{\beta B}{\alpha A}\right)^{1/\beta} \cdot N^{1 + \alpha/\beta} = C$$

$$N^{(\alpha + \beta)/\beta} \propto C$$

Solve for $N^*$:

$$\boxed{N^* \propto C^{\,\beta/(\alpha+\beta)}}$$

And since $D^* = C / (6N^*)$:

$$\boxed{D^* \propto C^{\,\alpha/(\alpha+\beta)}}$$

This is structurally identical to the result derived in Section 3.2 from the Kaplan approach — the functional form $N^* \propto C^{\beta/(\alpha+\beta)}$ is a consequence of the model $L = E + A/N^\alpha + B/D^\beta$ and the constraint $6ND = C$, not of any particular fitting procedure. **The difference lies entirely in the estimated values of $\alpha$ and $\beta$.**

Chinchilla's Approach 2 (fitting the parametric loss model directly to IsoFLOP curve data) yields:

$$\alpha \approx 0.34, \quad \beta \approx 0.28 \quad \text{(Hoffmann et al. 2022, Table A3)}$$

With these values:

$$\frac{\beta}{\alpha + \beta} = \frac{0.28}{0.34 + 0.28} = \frac{0.28}{0.62} \approx 0.45, \qquad \frac{\alpha}{\alpha + \beta} \approx 0.55$$

More saliently, when $\alpha \approx \beta$ (as all three of Chinchilla's fitting approaches roughly agree):

$$N^* \propto C^{1/2}, \qquad D^* \propto C^{1/2}$$

**Both parameters and tokens should scale as $\sqrt{C}$.** This is the central quantitative claim: equal scaling in parameters and data per compute doubling, in stark contrast to Kaplan's $N^* \propto C^{0.73}$, $D^* \propto C^{0.27}$.

### 4.3 The 20 Tokens Per Parameter Rule

The optimality condition $\alpha A \cdot D^\beta = \beta B \cdot N^\alpha$ is a relationship between $D^*$ and $N^*$ that holds at every compute budget. Using Chinchilla's fitted constants from Approach 2 ($A \approx 406.4$, $B \approx 410.7$, $\alpha \approx 0.34$, $\beta \approx 0.28$), one can evaluate the ratio $D^*/N^*$ numerically at any given scale. Across the compute budgets considered (roughly $10^{19}$ to $10^{24}$ FLOPs), this ratio is approximately constant and close to 20:

$$D^* \approx 20 \cdot N^*$$

This is the **"20 tokens per parameter" rule**. It is not exact — the precise ratio varies weakly with $C$ because $\alpha \neq \beta$ exactly — but it is a robust approximation within the regimes studied.

**Implications for prior models.** At the time of Chinchilla's publication:

- GPT-3: $N = 175\text{B}$ parameters, $D = 300\text{B}$ tokens $\Rightarrow$ $D/N \approx 1.7$ tokens per parameter
- Gopher: $N = 280\text{B}$ parameters, $D = 300\text{B}$ tokens $\Rightarrow$ $D/N \approx 1.1$ tokens per parameter

Both models used roughly 10–20$\times$ fewer tokens than the compute-optimal prescription. The Chinchilla model itself — 70B parameters trained on 1.4T tokens — achieved lower loss than Gopher (280B parameters, $4\times$ larger) at the same compute budget, directly validating the prescription. The name "Chinchilla" refers to this 70B model.

### 4.4 Why Kaplan's Exponents Were Biased

The discrepancy between Kaplan's $N^* \propto C^{0.73}$ and Chinchilla's $N^* \propto C^{0.5}$ traces directly to the bias introduced by Kaplan's univariate fitting procedure.

**The data-saturation bias in $\hat{\alpha}_N$.** Kaplan's estimate of $\alpha_N \approx 0.076$ comes from training models to convergence on a fixed (large but finite) dataset. As $N$ grows with $D$ fixed, eventually the model has seen the training set many times — it has effectively memorized the data and further increases in $N$ yield diminishing returns. Formally, if $D$ is fixed and $N \to \infty$, then $L \to E + B/D^\beta$: the model-size term vanishes but the data-limited floor remains. The measured marginal benefit of adding parameters flattens as the model approaches the data-limited regime, causing $\hat{\alpha}_N$ to underestimate the true capacity exponent.

In the language of statistical estimation: Kaplan's $L(N)$ curve is not measured in the compute-optimal regime — it is measured in the **data-saturated regime**, where $B/D^\beta \gg A/N^\alpha$ for the largest models. The slope of $\log L$ vs. $\log N$ in this regime reflects the approach to a data-limited floor, not the intrinsic capacity scaling. The result is an artificially small $\hat{\alpha}_N$.

**The consequence for optimal allocation.** In the formula $N^* \propto C^{\beta/(\alpha+\beta)}$, underestimating $\alpha$ (relative to the true value) makes the exponent $\beta/(\alpha + \beta)$ appear larger than it truly is, biasing the conclusion toward **scaling $N$ faster than warranted**. Kaplan's prescription to scale parameters 2.7$\times$ faster than data per compute doubling is thus an artifact of measurement in the wrong regime.

**Chinchilla's correction.** By holding compute fixed and sweeping the IsoFLOP curve, Chinchilla always trains in the compute-optimal regime: as $C$ increases, both $N$ and $D$ increase together. The model is never data-saturated at the scale being measured, so the estimated $\hat{\alpha}$ reflects genuine capacity scaling rather than a data floor artifact. The resulting $\alpha \approx 0.34$ — more than $4\times$ larger than Kaplan's $0.076$ — gives the balanced scaling result $N^* \propto C^{1/2}$.

## 5. Fitting Methodology

The five parameters $(E, A, B, \alpha, \beta)$ in $L(N, D) = E + A/N^\alpha + B/D^\beta$ cannot be read off directly from data — they must be estimated by fitting the model to a collection of training runs. Chinchilla cross-validates three distinct approaches, each with different assumptions and statistical properties.

### 5.1 Approach 1 — IsoFLOP Minimum Fitting

For each compute budget $C_i$, run models at several $(N, D)$ points along the IsoFLOP curve $6ND = C_i$. Fit a parabola to $\log L$ vs. $\log N$ to find the minimizing $N^*(C_i)$. Then fit:

$$\log N^*(C_i) = a \log C_i + b$$

via ordinary least squares (OLS). The slope $a$ gives $\beta/(\alpha+\beta)$.

This approach estimates the scaling exponents only (not $A$, $B$, $E$), and is the most robust because it does not require a global model fit. The parabolic approximation to the loss surface near the minimum is justified by the local smoothness of $L(N)$ under the IsoFLOP constraint, and the OLS step requires only that the optimal $N^*$ follows a power law in $C$ — a consequence of the model that can be checked for self-consistency.

### 5.2 Approach 2 — Parametric Global Fit

Minimize the sum of squared residuals over all runs simultaneously:

$$\min_{E, A, B, \alpha, \beta} \sum_{\text{runs}} \left( L_{\text{obs}} - E - \frac{A}{N^\alpha} - \frac{B}{D^\beta} \right)^2$$

Practical notes:

- **Fit in log-space to enforce positivity:** reparametrize $E = \exp(e)$, $A = \exp(a)$, $B = \exp(b)$ — this also makes the loss surface more symmetric and reduces the condition number of the Hessian, improving convergence.
- **Use L-BFGS** (a quasi-Newton method) to optimize — gradient descent would be too slow for a 5-parameter nonlinear fit, and L-BFGS exploits curvature information to converge in far fewer iterations.
- **Initialization:** use Approach 1's slope estimate to warm-start $\alpha$ and $\beta$; initialize $E$ slightly below the minimum observed loss to avoid the optimizer placing $E$ above the data.
- **Multiple restarts** are recommended to avoid local minima in the $(\alpha, \beta, E)$ subspace, where the loss surface can be non-convex.

This approach yields all five parameters and is used to compute the headline result $D^* \approx 20 N^*$.

### 5.3 Approach 3 — Per-Model-Size Estimation

For each fixed $N_i$, collect runs with varying $D$ and fit:

$$L(D; N_i) = E_{N_i} + \frac{B_{N_i}}{D^{\beta}}$$

via OLS in log-space to obtain estimates $\hat{E}_{N_i}$ and $\hat{\beta}_{N_i}$.

Then treat $\hat{E}_{N_i}$ as the effective irreducible loss at model size $N_i$ and fit:

$$\hat{E}_{N_i} = E + \frac{A}{N_i^\alpha}$$

as a function of $N_i$ to recover $E$, $A$, and $\alpha$.

This two-stage approach is more stable than the global fit — each stage is a lower-dimensional regression — but less data-efficient: it requires many $D$ values per $N$ level to accurately estimate $\hat{E}_{N_i}$, whereas Approach 2 can pool information across all $(N, D)$ pairs simultaneously.

### 5.4 Cross-Validation of Approaches

All three approaches agree on the key qualitative finding: **$\alpha \approx \beta$** (both in the range $0.28$–$0.50$ across approaches), implying $N^* \propto D^* \propto C^{1/2}$. The specific fitted constants differ across approaches, giving $D^*/N^*$ ratios ranging from approximately 5 to 30, with the headline "20 tokens per parameter" coming from Approach 2.

The agreement across methods — despite their substantially different statistical assumptions and data requirements — strengthens confidence in the equal-scaling conclusion even if the exact constants are uncertain. The qualitative result (equal $N$ and $D$ scaling) is robust; the quantitative constant (20) should be treated as an order-of-magnitude estimate.

### 5.5 Alternative Estimators

The simple parametric model $L = E + A/N^\alpha + B/D^\beta$ works well in the power-law region but breaks down near the regime boundaries (Section 2). Two alternative estimators have been proposed to handle transitions:

**M4 estimator** (Alabdulmohsin et al. 2022): fits a transformed version of the loss that saturates correctly at both ends:

$$\frac{L - E}{I - L}^{\,\alpha} = \frac{A}{N^a} + \frac{B}{D^b}$$

where $I$ is the random-guessing loss (the upper plateau). The ratio $(L-E)/(I-L)$ maps the loss into $(0, \infty)$, compressing the lower plateau and expanding the power-law region, making the power-law fit more robust across regimes.

**BNSL estimator** (Caballero et al. 2022 — "Broken Neural Scaling Laws"): models the loss as a product of sigmoid-like transitions:

$$L(D) = E + b \cdot D^{-c_0} \prod_{i} \left(1 + \left(\frac{D}{d_i}\right)^{1/f_i}\right)^{-c_i f_i}$$

Each factor in the product captures one transition (e.g., the onset of power-law scaling, or the approach to the Bayes floor). This is more expressive but has many more parameters and is harder to fit reliably.

Both estimators are strictly more general than the Kaplan/Chinchilla model — they reduce to the simple power law in the middle regime. The practical tradeoff: the simple model is sufficient for compute-optimal analysis (which operates in the power-law region by design), while the richer models are needed for predicting behavior at extreme scales or for detecting trend breaks.

---

## 6. Scope and Limitations

The scaling law framework described in this note is powerful but not universal. Several important caveats apply.

### Upstream vs. Downstream Performance

Scaling laws for pre-training loss (upstream) do not straightforwardly transfer to task-specific (downstream) performance. Pre-training loss decreases smoothly as a power law, but downstream metrics (e.g., few-shot accuracy on specific benchmarks) can exhibit **sharp transitions** — sudden jumps in capability at particular scales — that are not predictable from the smooth upstream curve. Tay et al. (2022) showed that downstream performance depends critically on task choice, architecture, and hyperparameters, not just compute.

This means compute-optimal training (minimizing pre-training loss) is not necessarily optimal for downstream deployment. The Chinchilla prescription optimizes upstream loss; whether it also optimizes downstream capability is an empirical question that varies by task.

### Non-Power-Law Scaling

Sorscher et al. (2022) showed that for some tasks, loss scales **exponentially** with dataset size rather than as a power law:

$$L(D) \sim e^{-\gamma D}$$

This occurs when the data distribution is well-structured and the model can effectively interpolate — for example, in settings with a small effective vocabulary of patterns. Exponential scaling is much more favorable than power-law scaling, but it means the standard Kaplan/Chinchilla framework does not apply. Identifying which regime applies requires empirical probing at multiple scales.

### Trend Breaks and Extrapolation Risk

Caballero et al. (2022) demonstrated that **trend breaks** — abrupt deviations from the fitted power law — can occur at scales not anticipated from smaller-scale observations. A scaling curve that appears cleanly power-law over three orders of magnitude may break at the fourth. This limits the reliability of extrapolation, particularly for predicting whether a much larger model will follow the projected loss curve.

### Modality and Architecture Dependence

Scaling exponents vary substantially across domains. Hestness et al. (2017) measured generalization error exponents $\beta_g$ across four domains with a single epoch of data scaling:

| Domain | Model | $\beta_g$ (data exponent) |
|--------|-------|--------------------------|
| Word language modeling | LSTM | $-0.066$ |
| Character language modeling | RHN | $-0.094$ |
| Machine translation | LSTM seq2seq | $-0.128$ |
| Image classification (top-5) | ResNet | $-0.488$ |
| Speech recognition | Deep Speech 2 | $-0.299$ |

Vision transformers (Zhai et al. 2022), acoustic models (Droppo and Elibol 2021), reinforcement learning agents (Neumann and Gros 2022), and recommendation systems (Ardalani et al. 2022) continue this pattern. There is no single universal $(\alpha, \beta)$ pair. The intrinsic dimension argument (Section 2) provides a theoretical explanation for this variation, but predicting exponents from first principles remains an open problem.

Hestness et al. also established that architecture improvements — replacing LSTMs with better recurrent architectures, adding depth, tuning hyperparameters — consistently shift the loss curve downward (reducing the multiplicative constant $\alpha_0$) without changing $\beta_g$. The exponent is a property of the task, not the model. This implies a ceiling on what architectural innovation alone can achieve: better architecture buys a one-time constant factor, not a change in the rate at which more data or more parameters help.

### Scaling for Recommendation Systems and Ranking Models

Recommendation and ranking systems present a distinct scaling regime from language models, shaped by several structural differences: inputs are sparse categorical features (user and item IDs) rather than dense token embeddings; daily interaction volumes are several orders of magnitude larger than the token corpora used for LLM training; and the downstream metric (CTR, NDCG, Recall@K) is bounded and nonlinear in loss, complicating direct comparison of scaling curves.

**Traditional DLRMs scale weakly.** Ardalani et al. (2022) studied DLRM-style models for click-through rate (CTR) prediction and found that model quality scales as a power law with a constant floor — the same functional form as the Kaplan decomposition:

$$L(N) \approx c + \frac{A}{N^\alpha}$$

But the effective scaling exponent for model size is small, and the dominant bottleneck is **data**: data scaling is the most viable path to improvement for this architecture class. The model-size term saturates quickly — beyond a moderate parameter count, DLRMs cannot usefully absorb additional capacity without more data. This is qualitatively different from transformers, where model size scaling (at fixed data) yields consistent gains.

**Generative recommenders unlock transformer-style scaling.** Zhai et al. (2024) reformulated recommendation as a sequential transduction problem using the HSTU (Hierarchical Sequential Transduction Units) architecture and showed that model quality scales as a **power law in training compute** across three orders of magnitude — directly analogous to the $L(C) \sim C^{-\alpha_C}$ behavior observed in language models. A trillion-parameter HSTU deployed in production achieved 12.4% metric improvements in live A/B testing. The key architectural changes enabling this:

- **SiLU replaces Softmax** in attention scoring: Softmax bounds maximum attention scores, limiting the model's ability to concentrate attention as sequences lengthen. SiLU removes this ceiling and is essential for scaling.
- **Relative temporal bias replaces positional encoding**: recommendation sequences carry timestamps, and relative time differences are more informative than absolute positions. Without temporal bias, scaling stalls.
- **Explicit feature interaction**: a pointwise transformation layer coupling user and item features is required; removing it causes ~7% degradation in HR@10.

**Scaling dimensions in recommendation: depth, width, and sequence length.** Guo et al. (2024) systematically studied scaling across model depth $L$ (transformer blocks), embedding dimension $D$, and sequence length $T$ for HSTU. A key structural finding analogous to the Chinchilla compute-optimal trade-off: **the product $L \times D$ is approximately constant at the optimum** for a fixed parameter budget. Scaling the model vertically (more layers) versus horizontally (wider embeddings) involves a trade-off, and the optimal balance is determined by the dataset size. Larger datasets require larger models, but oversized models on small datasets degrade performance — the same model-data alignment principle as Chinchilla, now applied to the depth-width-data triangle.

**Log-linear vs. power-law scaling.** Yan et al. (2025, WSDM 2026) trained the Large User Model (LUM) — a generative model pre-trained on 4 billion interactions, then fine-tuned discriminatively — and reported:

$$\text{Recall@10}(P) \approx 0.0068 \cdot \log P + 0.1741, \qquad P \in [19\text{M},\ 7\text{B}]$$

$$\text{Recall@10}(L) \approx 0.0147 \cdot \log L + 0.2326, \qquad L \in [256,\ 8192]$$

where $P$ is the number of parameters and $L$ is the sequence length. Note that these are log-linear relationships in the **metric** (Recall@10), not power-law relationships in **loss**. These are compatible: if cross-entropy loss scales as $\ell \sim P^{-\alpha}$ and Recall@10 is a saturating function of $-\ell$, then Recall@10 $\sim 1 - e^{-\ell^{-1}} \approx \beta \log P + \text{const}$ in the regime where $\ell$ is small. The log-linear metric curve is thus consistent with an underlying power-law loss curve in the regime where the metric is still far from saturation. However, log-linear metric extrapolation saturates at 1 and cannot continue indefinitely; the loss formulation is the correct substrate for long-range extrapolation.

The LUM paradigm — generative pre-training to capture joint $p(x, y)$, then discriminative fine-tuning to estimate $p(y|x)$ — is presented as the key to unlocking scaling. Traditional discriminative models trained end-to-end on $p(y|x)$ exhibit weak scaling because conditional distributions are simpler and offer less structure for the model to exploit at scale. This mirrors the observation that self-supervised pre-training (a generative objective) produces representations that scale better than purely supervised objectives in vision and NLP.

**Summary: differences from language model scaling.**

| Dimension | LLMs | Generative Recommenders |
|-----------|------|------------------------|
| Input representation | Dense token embeddings | Sparse categorical + temporal features |
| Data volume | Trillions of tokens (months) | Trillions of interactions (days) |
| Scaling variable | $N$, $D$, $C$ simultaneously | $L$, $D$, $T$ with $L \times D$ trade-off |
| Scaling metric | Cross-entropy loss (power law) | NDCG/Recall (log-linear in regime) |
| Traditional baseline | N/A — transformers from the start | DLRMs with weak scaling |
| Key enabling change | Scale itself | Generative objective + temporal features |
| Compute-optimal prescription | $N^* \propto D^* \propto C^{1/2}$ (Chinchilla) | $L^* \times D^* \approx \text{const}$ (Guo et al.) |

---

## References

| Reference Name | Brief Summary | Link to Reference |
|---|---|---|
| Kaplan et al. (2020), "Scaling Laws for Neural Language Models" | Establishes power-law scaling of LM loss with $N$, $D$, $C$; introduces the loss decomposition model; derives compute-efficient frontier favoring parameter-heavy scaling | https://arxiv.org/abs/2001.08361 |
| Hoffmann et al. (2022), "Training Compute-Optimal Large Language Models" (Chinchilla) | Revises Kaplan via IsoFLOP analysis; introduces three fitting approaches; shows equal $N$ and $D$ scaling is optimal; establishes the 20-tokens-per-parameter rule | https://arxiv.org/abs/2203.15556 |
| Henighan et al. (2020), "Scaling Laws for Autoregressive Generative Modeling" | Extends power-law scaling beyond language to images, video, and mathematical problems; shows universality of the scaling law framework | https://arxiv.org/abs/2010.14701 |
| Bahri et al. (2021), "Explaining Neural Scaling Laws" | Provides a theoretical derivation of power-law scaling from statistical mechanics; connects neural scaling to solvable models with broken power-law structure in the data | https://arxiv.org/abs/2102.06701 |
| Villalobos (2023), "Scaling Laws Literature Review" (Epoch AI) | Broad survey of scaling law results across modalities; discusses alternative estimators (M4, BNSL), non-power-law regimes, trend breaks, and the limits of upstream-to-downstream transfer | https://epoch.ai/blog/scaling-laws-literature-review |
| Caballero et al. (2022), "Broken Neural Scaling Laws" | Introduces the BNSL functional form to model transitions between scaling regimes; demonstrates that trend breaks cannot be reliably predicted from smaller-scale observations | https://arxiv.org/abs/2210.14891 |
| Sorscher et al. (2022), "Beyond neural scaling laws: beating power law scaling via data pruning" | Shows that with structured data pruning, loss can scale exponentially rather than as a power law, offering a route beyond the power-law frontier | https://arxiv.org/abs/2206.14486 |
| Hestness et al. (2017), "Deep Learning Scaling is Predictable, Empirically" | Establishes cross-domain power-law scaling of generalization error with training set size across MT, language modeling, image classification, and speech recognition; identifies the three learning curve regimes; shows architecture improvements shift the curve but not the exponent | https://arxiv.org/abs/1712.00409 |

### Further Reading: Scaling for Recommendation Systems and Ranking

| Reference Name | Brief Summary | Link to Reference |
|---|---|---|
| Ardalani et al. (2022), "Understanding Scaling Laws for Recommendation Models" | Empirical scaling study of DLRM-style CTR models; finds power-law-plus-constant scaling in model size, data, and compute; concludes data scaling is the dominant lever for this architecture class | https://arxiv.org/abs/2208.08489 |
| Zhai et al. (2024), "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations" (ICML 2024) | Introduces HSTU; demonstrates power-law scaling in training compute across three orders of magnitude for generative recommendation; deploys a trillion-parameter model with 12.4% live metric gains; identifies SiLU attention and temporal bias as prerequisites for scaling | https://arxiv.org/abs/2402.17152 |
| Guo et al. (2024), "Scaling New Frontiers: Insights into Large Recommendation Models" | Systematic study of scaling across depth, embedding dimension, and sequence length for HSTU; establishes the $L \times D = \text{const}$ optimality condition; shows temporal information dominates positional encoding and model-data alignment governs scaling efficiency | https://arxiv.org/abs/2412.00714 |
| Yan et al. (2025), "Unlocking Scaling Law in Industrial Recommendation Systems with a Three-step Paradigm based Large User Model" (WSDM 2026) | Proposes the LUM paradigm (generative pre-training then discriminative fine-tuning); reports log-linear Recall@10 scaling with model size (19M–7B) and sequence length (256–8192); validates with 2.9% CTR and 1.2% revenue gains in production at Taobao | https://arxiv.org/abs/2502.08309 |
