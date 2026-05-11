# Rotary Position Embeddings (RoPE) and Context Extension

## Table of Contents

- [[#1. Introduction: Why Positional Encodings Need Rethinking|1. Introduction: Why Positional Encodings Need Rethinking]]
- [[#2. RoPE Derivation|2. RoPE Derivation]]
  - [[#2.1 Setup and Desiderata|2.1 Setup and Desiderata]]
  - [[#2.2 The Rotation Matrix Construction|2.2 The Rotation Matrix Construction]]
  - [[#2.3 Complex-Number Shorthand|2.3 Complex-Number Shorthand]]
  - [[#2.4 Frequency Schedule|2.4 Frequency Schedule]]
- [[#3. Key Properties|3. Key Properties]]
  - [[#3.1 Relative Position in the Dot Product|3.1 Relative Position in the Dot Product]]
  - [[#3.2 Long-Term Decay|3.2 Long-Term Decay]]
  - [[#3.3 Equivariance to Sequence Shifts|3.3 Equivariance to Sequence Shifts]]
- [[#4. The Context Extension Problem|4. The Context Extension Problem]]
- [[#5. Positional Interpolation (PI)|5. Positional Interpolation (PI)]]
- [[#6. NTK-Aware Scaling|6. NTK-Aware Scaling]]
  - [[#6.1 The NTK Lens on RoPE|6.1 The NTK Lens on RoPE]]
  - [[#6.2 Base-Frequency Rescaling|6.2 Base-Frequency Rescaling]]
  - [[#6.3 What NTK Scaling Leaves Uncalibrated|6.3 What NTK Scaling Leaves Uncalibrated]]
- [[#7. YaRN: Yet Another RoPE ExtensioN|7. YaRN: Yet Another RoPE ExtensioN]]
  - [[#7.1 Dimension-Wise Wavelength Analysis|7.1 Dimension-Wise Wavelength Analysis]]
  - [[#7.2 The Ramp Function and NTK-by-Parts Interpolation|7.2 The Ramp Function and NTK-by-Parts Interpolation]]
  - [[#7.3 Attention Temperature Scaling|7.3 Attention Temperature Scaling]]
  - [[#7.4 Fine-Tuning Recipe|7.4 Fine-Tuning Recipe]]
- [[#8. Comparison and Practical Guidance|8. Comparison and Practical Guidance]]
- [[#References|References]]

---

## 1. Introduction: Why Positional Encodings Need Rethinking 💡

A transformer's self-attention mechanism is *permutation-equivariant* by design: shuffle the input tokens and the output shuffles identically. This is useful for sets, but language is a sequence — the meaning of a sentence depends critically on word order. Positional encodings break this symmetry by injecting order information into the token representations.

The earliest approaches, due to Vaswani et al. (2017), add a fixed sinusoidal signal to the token embeddings before the first layer. This is called *absolute positional encoding* (APE): each position $m$ maps to a deterministic vector, and the model learns to decode order from the additive perturbation. APE has two well-known weaknesses:

1. **Poor generalization to unseen lengths.** The model must have seen position indices up to the maximum training length. Beyond that boundary, position vectors are simply not present in the embedding table (for learned APE) or lie in an extrapolation regime of the sinusoidal function.
2. **No explicit relative bias.** Two tokens that are three positions apart convey the same relational signal regardless of whether they occur near the start or near the end of a sequence. Yet the model must infer this relative structure from absolute signals through learned weights.

*Relative positional encodings* address both issues by encoding the offset $m - n$ between positions directly into the attention computation. RoPE, introduced by [Su et al. (2021)](https://arxiv.org/abs/2104.09864), is now the dominant approach: it achieves relative encoding without extra parameters by rotating query and key vectors by a position-dependent angle before computing their dot product. The rotation is structured so that the inner product $\langle \mathbf{q}_m, \mathbf{k}_n \rangle$ is a function of $m - n$ alone — not of $m$ or $n$ individually. Crucially, RoPE is applied at every layer inside attention, not as a one-time additive offset.

RoPE underpins LLaMA, Mistral, Falcon, Qwen, and most modern open-weight LLMs. Its simplicity (no trainable parameters, purely geometric) and its natural long-term decay property make it attractive. However, as models are deployed on contexts longer than those seen during training, the rotation angles fall outside the trained distribution, leading to *perplexity explosion*. Sections 5–7 cover the three main remedies: Positional Interpolation (PI), NTK-Aware Scaling, and YaRN.

---

## 2. RoPE Derivation 📐

### 2.1 Setup and Desiderata

Let $d$ be the head dimension (assumed even throughout). For a token at position $m$ with embedding vector $\mathbf{x}_m \in \mathbb{R}^d$, the query and key vectors are

$$\mathbf{q}_m = W_Q\,\mathbf{x}_m \in \mathbb{R}^d, \qquad \mathbf{k}_n = W_K\,\mathbf{x}_n \in \mathbb{R}^d.$$

We want a function $f(\cdot, m): \mathbb{R}^d \to \mathbb{R}^d$ that encodes the absolute position $m$ into the query/key, such that

$$\langle f(\mathbf{q}, m),\; f(\mathbf{k}, n) \rangle = g(\mathbf{q}, \mathbf{k},\; m - n)$$

for some function $g$ that depends on the *difference* $m - n$ only. This is the relative-position desideratum.

### 2.2 The Rotation Matrix Construction

**Definition (RoPE Operation).** For a vector $\mathbf{v} \in \mathbb{R}^d$ at position $m$, define the RoPE encoding as

$$f(\mathbf{v}, m) = R_m\,\mathbf{v},$$

where $R_m \in \mathbb{R}^{d \times d}$ is the *block-diagonal rotation matrix*

$$R_m = \begin{pmatrix}
\cos(m\theta_1) & -\sin(m\theta_1) & & & \\
\sin(m\theta_1) & \phantom{-}\cos(m\theta_1) & & & \\
& & \ddots & & \\
& & & \cos(m\theta_{d/2}) & -\sin(m\theta_{d/2}) \\
& & & \sin(m\theta_{d/2}) & \phantom{-}\cos(m\theta_{d/2})
\end{pmatrix}.$$

Each $2 \times 2$ diagonal block is a planar rotation by angle $m\theta_i$ in the $i$-th frequency plane. All off-diagonal blocks are zero. The matrix is therefore *orthogonal*: $R_m^\top R_m = I$.

**Proof that the inner product depends only on relative position.** Since $R_m$ is orthogonal,

$$\langle f(\mathbf{q}, m),\; f(\mathbf{k}, n) \rangle = (R_m\,\mathbf{q})^\top (R_n\,\mathbf{k}) = \mathbf{q}^\top R_m^\top R_n\,\mathbf{k} = \mathbf{q}^\top R_{n-m}\,\mathbf{k}.$$

The last equality uses the group property $R_m^\top R_n = R_{n-m}$, which follows because each $2 \times 2$ rotation block satisfies $R(-m\theta_i)\,R(n\theta_i) = R((n-m)\theta_i)$. The result is a function of $\mathbf{q}$, $\mathbf{k}$, and $n - m$ only. $\square$

> [!NOTE] Efficient Implementation
> Because $R_m$ is block-diagonal, multiplying $R_m \mathbf{v}$ does not require a full $d \times d$ matrix-vector product. Only $d/2$ independent rotations are applied. In practice, using the identity
>
> $$\begin{pmatrix}\cos\phi & -\sin\phi\\\sin\phi & \phantom{-}\cos\phi\end{pmatrix}\begin{pmatrix}v_{2i}\\v_{2i+1}\end{pmatrix} = \begin{pmatrix}v_{2i}\cos\phi - v_{2i+1}\sin\phi\\ v_{2i}\sin\phi + v_{2i+1}\cos\phi\end{pmatrix}$$
>
> one applies the rotation elementwise to interleaved pairs, which maps to a simple vectorized operation requiring no explicit matrix construction.

### 2.3 Complex-Number Shorthand

Working in $\mathbb{C}^{d/2}$ instead of $\mathbb{R}^d$ gives a cleaner presentation. Define the *complex embedding*

$$\tilde{v}_i = v_{2i} + i\,v_{2i+1}, \qquad i = 1, \ldots, d/2,$$

so that $\tilde{\mathbf{v}} = (\tilde{v}_1, \ldots, \tilde{v}_{d/2}) \in \mathbb{C}^{d/2}$.

**Definition (RoPE in Complex Form).** The RoPE encoding at position $m$ acts on the $i$-th complex component as

$$f(\tilde{v}_i, m) = \tilde{v}_i \cdot e^{im\theta_i}.$$

This is multiplication by the unit complex number $e^{im\theta_i}$, i.e., a rotation by angle $m\theta_i$ in the complex plane. The inner product in real space corresponds to $\operatorname{Re}(\tilde{\mathbf{q}}^* \cdot \tilde{\mathbf{k}})$ after applying position-dependent phase factors:

$$\operatorname{Re}\!\left(\sum_{i=1}^{d/2} \overline{f(\tilde{q}_i, m)} \cdot f(\tilde{k}_i, n)\right) = \operatorname{Re}\!\left(\sum_{i=1}^{d/2} \tilde{q}_i^*\tilde{k}_i \cdot e^{i(n-m)\theta_i}\right),$$

which manifestly depends only on $n - m$.

### 2.4 Frequency Schedule

**Definition (Frequency Schedule).** The angles $\{\theta_i\}_{i=1}^{d/2}$ follow a geometric progression

$$\theta_i = b^{-2(i-1)/d}, \qquad i = 1, \ldots, d/2,$$

where $b = 10{,}000$ is the *base* (following the convention of the original transformer sinusoidal encoding). In the zero-indexed convention used in code, this is `inv_freq[i] = base^{-(2i/d)}` for `i in range(0, d, 2)`.

The *wavelength* of dimension $i$ is the number of token positions required for the rotation to complete one full cycle:

$$\lambda_i = \frac{2\pi}{\theta_i} = 2\pi \cdot b^{2(i-1)/d}.$$

Low-indexed dimensions ($i \approx 1$) have $\theta_i \approx 1$ (fast rotation, short wavelength $\approx 2\pi$) and encode fine-grained local position. High-indexed dimensions ($i \approx d/2$) have $\theta_i \approx b^{-1}$ (slow rotation, long wavelength $\approx 2\pi b$) and encode coarse global position. **The dynamic range of wavelengths spans $b$ orders of magnitude** — for $b = 10{,}000$, from $\approx 6$ tokens up to $\approx 60{,}000$ tokens at $d = 128$.

> [!QUESTION] Exercise 1: Wavelength Bounds
> *This exercise establishes the extremes of the frequency schedule, which are essential for understanding which dimensions RoPE can and cannot handle at long contexts.*
>
> > **Prerequisites:** [[#2.4 Frequency Schedule|2.4 Frequency Schedule]]
>
> For base $b = 10{,}000$ and head dimension $d = 64$:
>
> (a) Compute the wavelength $\lambda_1$ of the fastest-rotating dimension and $\lambda_{32}$ of the slowest.
> (b) If a model is trained on sequences of length $L = 4096$, which dimension indices $i$ have $\lambda_i > L$? Express your answer as an inequality on $i$.

> [!TIP]- Solution to Exercise 1
> **Key insight:** Dimensions with $\lambda_i > L$ complete fewer than one full rotation during training — they encode only coarse positional relationships and are the hardest to generalize beyond $L$.
>
> **Sketch:**
>
> (a) $\lambda_1 = 2\pi \cdot b^{0} = 2\pi \approx 6.28$ tokens. $\lambda_{32} = 2\pi \cdot b^{2 \cdot 31/64} = 2\pi \cdot 10000^{31/32} \approx 2\pi \cdot 6310 \approx 39{,}650$ tokens.
>
> (b) Require $2\pi \cdot b^{2(i-1)/d} > L$. Taking logs: $\frac{2(i-1)}{d}\ln b > \ln(L/2\pi)$. With $d = 64$, $b = 10^4$, $L = 4096$: $\frac{i-1}{32} > \frac{\ln(651.5)}{9.21} \approx 0.703$, so $i - 1 > 22.5$, giving $i \geq 24$. Dimensions 24–32 have wavelengths exceeding the training length.

---

## 3. Key Properties 🔑

### 3.1 Relative Position in the Dot Product

The central result was already established in §2.2: for any query at position $m$ and key at position $n$,

$$\langle f(\mathbf{q}, m),\; f(\mathbf{k}, n) \rangle = \mathbf{q}^\top R_{n-m}\,\mathbf{k} = \sum_{i=1}^{d/2}\left[q_{2i-1}k_{2i-1} + q_{2i}k_{2i}\right]\cos((n-m)\theta_i) + \left[q_{2i-1}k_{2i} - q_{2i}k_{2i-1}\right]\sin((n-m)\theta_i).$$

This can be written compactly as

$$\langle f(\mathbf{q}, m),\; f(\mathbf{k}, n) \rangle = \operatorname{Re}\!\left(\tilde{\mathbf{q}}^* \odot \tilde{\mathbf{k}} \cdot e^{i(n-m)\boldsymbol{\theta}}\right)_{\text{sum}},$$

where $\odot$ is elementwise product and $\boldsymbol{\theta} = (\theta_1, \ldots, \theta_{d/2})$. **The attention score is a cosine-weighted sum of query-key correlations at frequency $\theta_i$, with phase determined by the relative position $n - m$.**

### 3.2 Long-Term Decay

**Proposition (Long-Term Decay).** For random unit-norm vectors $\mathbf{q}, \mathbf{k} \sim \mathcal{S}^{d-1}$, the expected inner product $\mathbb{E}[\langle f(\mathbf{q}, m), f(\mathbf{k}, n)\rangle]$ decays toward zero as $|n - m| \to \infty$.

*Sketch.* The inner product is a sum of $d/2$ terms of the form $A_i\cos((n-m)\theta_i) + B_i\sin((n-m)\theta_i)$. For generic (incommensurate) frequencies $\{\theta_i\}$, the Weyl equidistribution theorem implies that as $|n-m|$ grows, the phases $(n-m)\theta_i \bmod 2\pi$ become equidistributed, so each term averages to zero. The sum therefore decays in expectation. In practice, the geometric frequency schedule ensures the $\theta_i$ are incommensurate, giving the decay behavior observed empirically.

*This result is heuristic in the sense that the decay rate depends on the specific $\mathbf{q}, \mathbf{k}$ values and is not a pointwise guarantee — but it captures the right intuition:* distant tokens are harder to attend to, all else equal.

### 3.3 Equivariance to Sequence Shifts

If the entire sequence is shifted by $\delta$ positions (i.e., every position $m \mapsto m + \delta$), the inner product becomes $\mathbf{q}^\top R_{n-m}\mathbf{k}$ — unchanged. RoPE is therefore *translation-equivariant*: shifting a window of tokens within a longer context does not alter pairwise attention scores between tokens within that window.

> [!NOTE] Contrast with APE
> With absolute positional encoding, shifting a window changes all absolute position vectors, disrupting every pairwise score. RoPE's equivariance is exactly why sliding-window attention and chunked inference work naturally with RoPE but are awkward under APE.

> [!QUESTION] Exercise 2: Relative Encoding via Inner Product
> *This exercise reinforces the algebra showing that the RoPE dot product depends only on relative position.*
>
> > **Prerequisites:** [[#2.2 The Rotation Matrix Construction|2.2 The Rotation Matrix Construction]], [[#2.3 Complex-Number Shorthand|2.3 Complex-Number Shorthand]]
>
> Working in the complex shorthand, verify that
> $$\operatorname{Re}\!\left(\overline{f(\tilde{q}_i, m)} \cdot f(\tilde{k}_i, n)\right) = \operatorname{Re}\!\left(\tilde{q}_i^*\tilde{k}_i\, e^{i(n-m)\theta_i}\right).$$
> Then expand $\tilde{q}_i = q_{2i-1} + iq_{2i}$ and $\tilde{k}_i = k_{2i-1} + ik_{2i}$ to show this equals
> $$\left(q_{2i-1}k_{2i-1} + q_{2i}k_{2i}\right)\cos((n-m)\theta_i) + \left(q_{2i}k_{2i-1} - q_{2i-1}k_{2i}\right)\sin((n-m)\theta_i).$$

> [!TIP]- Solution to Exercise 2
> **Key insight:** Conjugation of the query factor cancels the absolute position $m$, leaving only the relative phase.
>
> **Sketch:** $\overline{f(\tilde{q}_i, m)} = \overline{\tilde{q}_i e^{im\theta_i}} = \tilde{q}_i^* e^{-im\theta_i}$. Multiplying:
>
> $\overline{f(\tilde{q}_i,m)} \cdot f(\tilde{k}_i,n) = \tilde{q}_i^* e^{-im\theta_i} \cdot \tilde{k}_i e^{in\theta_i} = \tilde{q}_i^*\tilde{k}_i\, e^{i(n-m)\theta_i}.$
>
> Taking the real part: $\operatorname{Re}(\tilde{q}_i^*\tilde{k}_i\,e^{i(n-m)\theta_i})$. Write $\tilde{q}_i^*\tilde{k}_i = (q_{2i-1} - iq_{2i})(k_{2i-1} + ik_{2i}) = (q_{2i-1}k_{2i-1} + q_{2i}k_{2i}) + i(q_{2i-1}k_{2i} - q_{2i}k_{2i-1})$ and $e^{i\phi} = \cos\phi + i\sin\phi$. Multiplying and extracting the real part yields the stated formula. Note the sign on the sine term differs from §3.1 by the sign convention for $q$ and $k$ — both are correct, corresponding to different orderings of $q$ vs. $k$ factors.

---

## 4. The Context Extension Problem ⚠️

Let $L$ be the maximum sequence length during training. At inference time, if a sequence of length $L' > L$ is presented, positions $m \in \{L, L+1, \ldots, L'-1\}$ are encountered for the first time. The rotation angles $m\theta_i$ lie outside the range $[0, L\theta_i]$ used during training.

**Why this breaks attention.** The model's weight matrices $W_Q$ and $W_K$ have learned to produce query-key pairs whose inner products $\mathbf{q}^\top R_{n-m}\mathbf{k}$ are informative for offsets $n - m$ seen during training. For large $|n - m|$, especially offsets approaching or exceeding $L$, the rotation matrix $R_{n-m}$ has never been seen. The inner products can take arbitrary values, producing *out-of-distribution attention scores* that disrupt the softmax.

**The out-of-distribution frequencies intuition.** Consider dimension $i$ with wavelength $\lambda_i$. During training the angle $m\theta_i$ sweeps through the range $[0, 2\pi L/\lambda_i]$ radians. Dimensions with $\lambda_i \gg L$ (slow, high-index dimensions) complete much less than one full rotation during training — their angle range is tiny, so any angle much larger than $2\pi L/\lambda_i$ is entirely unseen. These *low-frequency dimensions* are most vulnerable.

Dimensions with $\lambda_i \ll L$ (fast, low-index dimensions) complete many full rotations during training and are effectively well-calibrated for any relative position, since their angle distribution is approximately uniform on $[0, 2\pi]$ even at training length.

**Perplexity explosion.** In practice, models trained on $L = 4096$ tokens exhibit near-baseline perplexity when evaluated at $L' \lesssim 1.5L$, but perplexity grows sharply for $L' \gtrsim 2L$ and diverges for $L' \gg L$. The effective usable context is strictly less than $L$.

> [!WARNING]
> *The perplexity explosion is not gradual — it is a phase transition. A model may seem to function at $L' = 1.1L$ but completely fail at $L' = 2L$. Do not assume a safety margin of "a bit more than $L$" without empirical validation.*

---

## 5. Positional Interpolation (PI) 📐

[Chen et al. (2023)](https://arxiv.org/abs/2306.15595) propose the simplest fix: instead of letting position indices exceed $L$, *compress* them back into the training range by a scale factor $s = L'/L$.

**Definition (Positional Interpolation).** Given a scaling factor $s > 1$, the interpolated RoPE replaces position $m$ with $m/s$:

$$f'(\mathbf{v}, m) = R_{m/s}\,\mathbf{v}.$$

Equivalently, the attention computation uses the inner product $\mathbf{q}^\top R_{(n-m)/s}\,\mathbf{k}$. Every pairwise offset is compressed: a relative offset of $s$ tokens now produces the same rotation as an offset of $1$ token did during training.

**Why interpolation beats extrapolation.** Chen et al. prove that the upper bound on the change in attention logits from interpolation is at least $600\times$ smaller than from extrapolation to the same context length $L'$. Heuristically: extrapolation requires the model to generalize rotation angles it has never seen at all, while interpolation reuses angles within $[0, L\theta_i]$ that were always seen during training — just with finer granularity.

**Limitation.** PI compresses *all* frequency dimensions equally by $1/s$. Fast-rotating (high-frequency) dimensions, which already complete many cycles during training, now rotate even faster relative to their role. The high-frequency components that distinguish nearby tokens (positions 1, 2, 3 apart) are squeezed together. *After PI fine-tuning, models show degraded ability to distinguish fine-grained local order.* This motivates the frequency-discriminating approaches below.

> [!QUESTION] Exercise 3: Interpolation Compression
> *This exercise makes precise how PI changes the effective resolution of nearby-position discrimination.*
>
> > **Prerequisites:** [[#2.4 Frequency Schedule|2.4 Frequency Schedule]], [[#5. Positional Interpolation (PI)|5. Positional Interpolation (PI)]]
>
> Under PI with scale $s$, the rotation angle between positions $m$ and $m+1$ changes from $\theta_i$ to $\theta_i/s$ for all dimensions $i$.
>
> (a) For dimension $i = 1$ (fastest), what fraction of a full cycle does an offset of $1$ token subtend after PI with $s = 8$?
> (b) Argue why this is problematic for local relative-position discrimination but not for global structure.

> [!TIP]- Solution to Exercise 3
> **Key insight:** PI crushes the angular step for nearby tokens, making adjacent and near-adjacent positions nearly indistinguishable in the high-frequency dimensions that were specifically designed to distinguish them.
>
> **Sketch:**
>
> (a) Before PI: $\theta_1 = b^0 = 1$ rad per token. Fraction of full cycle = $1/(2\pi) \approx 15.9\%$. After PI with $s=8$: angle per token = $1/8$ rad. Fraction = $1/(16\pi) \approx 2.0\%$. The angular step shrinks 8-fold.
>
> (b) Locally (small offsets), $\cos((n-m)\theta_1/s)$ is nearly 1 for $|n-m|$ up to several tokens, so the model's inner product loses sensitivity to small offsets. Globally (large offsets), the low-frequency dimensions (large $i$) naturally complete less than one cycle and are unaffected in practice — the compression simply slows their rotation slightly, which they can tolerate because they were already operating over the full $[0, L]$ range.

---

## 6. NTK-Aware Scaling 📐

### 6.1 The NTK Lens on RoPE

In the Neural Tangent Kernel (NTK) theory of neural network generalization, the *effective bandwidth* of the position encoding governs which frequencies of positional signal the network can represent and generalize. The key observation, due to [bloc97 (2023)](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/), is:

- **High-frequency dimensions** (small $i$, small $\lambda_i$): rotation angles cycle many times in $[0, L]$. The model is effectively interpolating between familiar angle values, even at slightly longer sequences.
- **Low-frequency dimensions** (large $i$, large $\lambda_i$): rotation angles barely complete one cycle in $[0, L]$. Any sequence longer than $\lambda_i$ requires the model to extrapolate into an entirely unseen angular regime.

PI's uniform compression helps but at the cost of degrading high-frequency resolution. The NTK insight is: *rather than compressing positions, change the base $b$ so that all wavelengths scale up proportionally.* This spreads the "extension budget" across all frequencies instead of concentrating it in position compression.

### 6.2 Base-Frequency Rescaling

**Definition (NTK-Aware Scaled RoPE).** For a context scaling factor $s = L'/L > 1$, replace the frequency base $b$ with a rescaled base

$$b' = b \cdot s^{d/(d-2)}.$$

The new frequencies are $\theta_i' = (b')^{-2(i-1)/d}$, and the new wavelengths are

$$\lambda_i' = 2\pi \cdot (b')^{2(i-1)/d} = 2\pi \cdot b^{2(i-1)/d} \cdot s^{\frac{d}{d-2}\cdot\frac{2(i-1)}{d}} = \lambda_i \cdot s^{\frac{2(i-1)}{d-2}}.$$

**Derivation of the exponent.** We want the slowest dimension ($i = d/2$) to have its wavelength scaled by exactly $s$, since that dimension spans the longest range and benefits most from uniform stretching. Set $\lambda_{d/2}' = s\,\lambda_{d/2}$:

$$s^{\frac{d}{d-2}\cdot\frac{2(d/2-1)}{d}} = s^{\frac{d-2}{d-2}} = s^1 = s. \quad \checkmark$$

So the exponent $d/(d-2)$ is chosen so that the *lowest-frequency dimension stretches by exactly $s$*, while faster dimensions stretch by a smaller factor. The fastest dimension ($i = 1$) stretches by $s^0 = 1$ — it is not changed at all. This is the sense in which NTK scaling is "frequency-aware": it applies proportionally more stretching to the low-frequency (high-index) dimensions that most need it, and leaves the high-frequency (low-index) dimensions untouched.

> [!EXAMPLE] NTK Rescaling at s = 4, d = 64, b = 10000
>
> The new base is $b' = 10000 \cdot 4^{64/62} \approx 10000 \cdot 4.13 = 41{,}300$.
>
> - Dimension $i=1$: $\lambda_1' = 2\pi \cdot (b')^0 = 2\pi$. Unchanged (fast dim).
> - Dimension $i=32$ (slowest): $\lambda_{32}' = 2\pi \cdot (b')^{31/32} \approx 4 \times \lambda_{32}$. Stretched by $s = 4$.
> - Dimension $i=16$: stretched by $s^{15/31} \approx 4^{0.48} \approx 1.94$. Intermediate.

### 6.3 What NTK Scaling Leaves Uncalibrated

NTK-aware scaling correctly stretches low-frequency dimensions but does not constrain the high-frequency dimensions to stay within the trained regime — they were already fine, so touching them introduces no benefit and no harm. However, NTK scaling applies the *same uniform base change* to all dimensions, which means even the fast-rotating dimensions receive a slight modification. *Empirically, NTK scaling without fine-tuning achieves good perplexity at modest extensions ($s \leq 4$) but degrades at larger extensions ($s \geq 8$).* The YaRN method addresses this by treating each frequency dimension individually.

> [!QUESTION] Exercise 4: NTK Base Rescaling Derivation
> *This exercise works through the algebra connecting the desired wavelength scaling to the base rescaling formula.*
>
> > **Prerequisites:** [[#2.4 Frequency Schedule|2.4 Frequency Schedule]], [[#6.2 Base-Frequency Rescaling|6.2 Base-Frequency Rescaling]]
>
> We want a new base $b'$ such that $\theta_i' = (b')^{-2(i-1)/d}$ satisfies $\lambda_i' = \lambda_i \cdot s^{2(i-1)/(d-2)}$ for all $i$.
>
> (a) Write $b' = b \cdot c$ for some multiplier $c$ to be determined. Express $\lambda_i'$ in terms of $\lambda_i$, $c$, $i$, and $d$.
> (b) Require $\lambda_{d/2}' = s \cdot \lambda_{d/2}$ and solve for $c$. Verify that $c = s^{d/(d-2)}$.

> [!TIP]- Solution to Exercise 4
> **Key insight:** Setting $c$ by the constraint on the slowest dimension uniquely determines the base multiplier.
>
> **Sketch:**
>
> (a) $\theta_i' = (bc)^{-2(i-1)/d} = \theta_i \cdot c^{-2(i-1)/d}$, so $\lambda_i' = \lambda_i \cdot c^{2(i-1)/d}$.
>
> (b) At $i = d/2$: $\lambda_{d/2}' = \lambda_{d/2} \cdot c^{2(d/2-1)/d} = \lambda_{d/2} \cdot c^{(d-2)/d}$. Setting this equal to $s \cdot \lambda_{d/2}$: $c^{(d-2)/d} = s$, so $c = s^{d/(d-2)}$. Thus $b' = b \cdot s^{d/(d-2)}$. $\square$

---

## 7. YaRN: Yet Another RoPE ExtensioN 📐

[Peng et al. (2023)](https://arxiv.org/abs/2309.00071) unify and improve upon PI and NTK scaling. The key observation is that no single global strategy is optimal across the full frequency spectrum:

- **Fast dimensions** ($\lambda_i \ll L$): complete many cycles during training. Interpolating them (PI) compresses their already-fine resolution. Leaving them alone (NTK-inspired) is better.
- **Slow dimensions** ($\lambda_i \gg L'$): never complete even a partial cycle even at extended length $L'$. Standard interpolation (PI) is appropriate — just compress their position indices.
- **Intermediate dimensions**: need a blend.

YaRN implements this dimension-wise interpolation strategy via a smooth ramp function.

### 7.1 Dimension-Wise Wavelength Analysis

**Definition (Dimension Wavelength).** For dimension index $i$ (zero-indexed, $i = 0, \ldots, d/2 - 1$) with training length $L$ and base $b$, define

$$\lambda_i = 2\pi \cdot b^{2i/d}, \qquad r_i = \frac{L}{\lambda_i} = \frac{L}{2\pi \cdot b^{2i/d}}.$$

$r_i$ measures how many complete rotations dimension $i$ undergoes during training on sequences of length $L$. Large $r_i$ means many cycles — high frequency, well-calibrated. Small $r_i$ means few cycles — low frequency, extrapolation risk.

### 7.2 The Ramp Function and NTK-by-Parts Interpolation

**Definition (YaRN Ramp Function).** Let $\alpha$ and $\beta$ be threshold parameters with $0 < \alpha < \beta$. The *ramp function* is

$$\gamma(r) = \begin{cases}
0 & \text{if } r < \alpha \\
1 & \text{if } r > \beta \\
\dfrac{r - \alpha}{\beta - \alpha} & \text{otherwise.}
\end{cases}$$

For LLaMA-family models, $\alpha = 1$ and $\beta = 32$ are recommended.

**Definition (NTK-by-Parts Interpolation).** YaRN modifies each frequency $\theta_i$ independently using the ramp:

$$\theta_i^{\text{YaRN}} = \left(1 - \gamma(r_i)\right)\,\frac{\theta_i}{s} + \gamma(r_i)\,\theta_i.$$

Equivalently, define the *effective scaling factor* for dimension $i$ as

$$\alpha_i = \left(1 - \gamma(r_i)\right)\,\frac{1}{s} + \gamma(r_i) \cdot 1,$$

so $\theta_i^{\text{YaRN}} = \alpha_i\,\theta_i$. The three regimes are:

| Regime | Condition | $\gamma$ | Action | Rationale |
|--------|-----------|---------|--------|-----------|
| Pure interpolation | $r_i < 1$ | 0 | $\theta_i \leftarrow \theta_i / s$ | Wavelength $> L$; must interpolate |
| Pure extrapolation | $r_i > 32$ | 1 | $\theta_i \leftarrow \theta_i$ | Fast dim; no modification needed |
| Blend | $1 \leq r_i \leq 32$ | linear | weighted average | Intermediate; smooth transition |

**Relationship to NTK.** Setting $\gamma \equiv 1$ everywhere recovers vanilla (unscaled) RoPE. Setting $\gamma \equiv 0$ recovers standard PI. The NTK base scaling of §6.2 corresponds to a different approach: it applies a single base change that implicitly scales each $\theta_i$ by a different amount, but without precise per-dimension control. NTK-by-parts gives explicit per-dimension control via $\gamma(r_i)$.

> [!EXAMPLE] YaRN Frequency Assignment at s = 8, d = 64, b = 10000, L = 4096
>
> Compute $r_i = L / (2\pi b^{2i/d})$:
>
> - $i = 0$: $\lambda_0 = 2\pi \approx 6.3$, $r_0 = 4096/6.3 \approx 650 \gg \beta = 32$. Full extrapolation ($\gamma = 1$, $\theta$ unchanged).
> - $i = 20$: $\lambda_{20} = 2\pi \cdot 10000^{40/64} \approx 2\pi \cdot 794 \approx 4990$, $r_{20} = 4096/4990 \approx 0.82 < \alpha = 1$. Full interpolation ($\gamma = 0$, $\theta \leftarrow \theta/8$).
> - $i = 15$: $\lambda_{15} \approx 2\pi \cdot 178 \approx 1118$, $r_{15} \approx 3.67$. Blend: $\gamma = (3.67 - 1)/(32 - 1) \approx 0.086$, mostly interpolated.

### 7.3 Attention Temperature Scaling

A subtle effect of extending context is that the distribution of attention logits shifts. With more tokens in scope, the unnormalized pre-softmax scores $q_m^\top k_n / \sqrt{d}$ have higher variance, leading to softer (higher-entropy) attention distributions than intended. This is sometimes called the *attention saturation* effect: extended models attend more uniformly, losing sharpness.

**Definition (YaRN Temperature Scaling).** YaRN modifies the softmax by introducing a *temperature* parameter $t$:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{t\sqrt{d}}\right)V, \qquad \sqrt{\frac{1}{t}} = 0.1\ln(s) + 1.$$

Here $s = L'/L$ is the context scaling factor. Equivalently, the inverse temperature $\sqrt{1/t}$ increases logarithmically with the scale: at $s = 1$ (no extension), $\sqrt{1/t} = 1$, recovering the standard scaling. At $s = e^{10}$, the temperature correction doubles. The empirical formula was derived by fitting optimal temperature values across LLaMA 7B–65B without fine-tuning.

*The correction acts like sharpening: by dividing by a larger effective scaling factor $t\sqrt{d}$, the model's attention distributions concentrate more, restoring the sharpness calibration expected from training.*

> [!NOTE] Why t and Not the Standard 1/sqrt(d)?
> The standard $1/\sqrt{d}$ scaling controls the variance of $QK^\top$ for random unit-norm vectors. Temperature scaling is an additional correction for the *changed geometry* induced by operating at extended context: YaRN's interpolated frequencies produce query-key inner products with different variance statistics than the trained regime. The $\sqrt{1/t}$ factor re-calibrates this.

### 7.4 Fine-Tuning Recipe

YaRN's theoretical improvements are amplified by a short fine-tuning phase:

- Fine-tune on $\approx 0.1\%$ of the original pretraining data volume on sequences of length $L'$.
- Use the NTK-by-parts frequencies and temperature correction throughout fine-tuning.
- **400 gradient steps** at LLaMA scale suffice to reach state-of-the-art perplexity at $L'$, compared to $\sim 1000$ steps for PI.

**YaRN requires $10\times$ fewer tokens and $2.5\times$ fewer training steps than prior methods** to achieve equivalent perplexity at extended context.

> [!QUESTION] Exercise 5: Ramp Function Design
> *This exercise explores sensitivity of YaRN to the ramp parameters alpha and beta.*
>
> > **Prerequisites:** [[#7.2 The Ramp Function and NTK-by-Parts Interpolation|7.2 The Ramp Function and NTK-by-Parts Interpolation]]
>
> (a) What happens to YaRN in the limit $\alpha \to 0$, $\beta \to \infty$? What method does it reduce to?
> (b) Suppose you set $\alpha = \beta = r^*$ for some $r^*$. Describe the behavior of $\gamma(r_i)$ as a function of dimension. What does the resulting method look like?
> (c) The recommended values are $\alpha = 1$, $\beta = 32$. Give an intuitive justification for why $\alpha = 1$ is a natural lower threshold.

> [!TIP]- Solution to Exercise 5
> **Key insight:** The ramp parameters $\alpha$ and $\beta$ define which dimensions are treated as "well-calibrated" vs. "under-calibrated" — $\alpha$ and $\beta$ correspond to $r_i$ cutoffs for pure interpolation and pure extrapolation respectively.
>
> **Sketch:**
>
> (a) As $\alpha \to 0$ and $\beta \to \infty$, $\gamma(r_i)$ lies in the blending regime for all dimensions. In the limit $\gamma(r_i) = r_i / (\beta - \alpha) \to 0$ for all finite $r_i$, so $\theta_i^{\text{YaRN}} \to \theta_i/s$ for all $i$. This is pure PI.
>
> (b) With $\alpha = \beta = r^*$, the ramp becomes a step function: $\gamma(r_i) = 0$ for $r_i < r^*$ and $\gamma(r_i) = 1$ for $r_i \geq r^*$. The method becomes a hard frequency cutoff: dimensions with $r_i < r^*$ are fully interpolated, those with $r_i \geq r^*$ are untouched. This is a simplified "NTK-by-parts" with a hard threshold.
>
> (c) $r_i < 1$ means $\lambda_i > L$ — the dimension's wavelength exceeds the training context length, so it has not completed even one full rotation during training. Such dimensions are genuinely under-calibrated and benefit from interpolation. $r_i \geq 1$ means $\lambda_i \leq L$ — the dimension has completed at least one full cycle, so it has been well-trained over its full angular range. Interpolation would harm these dimensions.

---

## 8. Comparison and Practical Guidance 🔑

The four methods form a progression of increasing sophistication:

| Method | Position Modification | Frequency Modification | Fine-tuning Needed | Strengths | Weaknesses |
|--------|----------------------|----------------------|--------------------|-----------|------------|
| **Vanilla RoPE** | None | None | N/A | Exact trained behavior | Fails beyond $L$ |
| **PI** (Chen et al. 2023) | $m \mapsto m/s$ (all dims) | None (equivalent to $\theta_i \mapsto \theta_i/s$) | ~1000 steps | Stable, simple | Destroys high-freq resolution |
| **NTK Scaling** (bloc97 2023) | None | $b \mapsto b \cdot s^{d/(d-2)}$ (global base change) | Optional | No fine-tuning needed; preserves high-freq | Leaves high-freq slightly disturbed; degrades at large $s$ |
| **YaRN** (Peng et al. 2023) | None | Per-dim: $\theta_i \mapsto \alpha_i \theta_i$ via ramp; + temperature scaling | ~400 steps | Best perplexity; frequency-discriminating | Slightly more complex; requires $\alpha, \beta$ tuning |

**Practical guidance:**

- *If no fine-tuning is possible*, use NTK-Aware Scaling with $b' = b \cdot s^{d/(d-2)}$. It requires only changing the base constant in the RoPE computation and provides good perplexity at moderate extensions ($s \leq 4$).
- *If a small fine-tuning budget is available* ($\sim 400$ steps on $0.1\%$ of pretrain data), YaRN is the best choice and consistently outperforms all alternatives at all measured context lengths.
- *Avoid pure PI* unless the downstream task is insensitive to local positional order (e.g., long-document retrieval where paragraph-level structure dominates). Its compression of high-frequency dimensions degrades near-neighbor sensitivity.
- *Dynamic NTK* (a variant where the base $b'$ is recomputed on-the-fly based on the actual inference context length) can be used for zero-shot extension when the target length is not known at deployment time.

> [!WARNING]
> *The perplexity tables in the YaRN paper (Table 5, reproduced below) compare methods at LLaMA 7B with 400 fine-tuning steps on 32k context. Without fine-tuning, NTK-aware scaling outperforms YaRN's NTK-by-parts on some intermediate lengths. The margin of YaRN is largest at the extended length $L' = 32768$ and requires fine-tuning to materialize.*

**YaRN Table 5 Ablation (LLaMA 7B, 400 steps, target $L' = 32768$, perplexity at each eval length):**

| Method | 2048 | 4096 | 8192 | 16384 | 32768 |
|--------|------|------|------|-------|-------|
| PI | 5.70 | 4.95 | 4.64 | 3.97 | 3.57 |
| NTK-aware | 4.39 | 3.92 | 3.73 | 3.21 | 8.49 |
| NTK-by-parts | 4.14 | 3.75 | 3.62 | 3.12 | 2.81 |
| YaRN | 4.19 | 3.77 | 3.30 | 3.09 | **2.77** |

*Baseline (no extension) at 2048 tokens: 4.05. Lower perplexity is better.*

**Key takeaway:** **NTK-aware scaling collapses at 32768 (perplexity 8.49 $\gg$ 2.77 for YaRN), revealing that global base rescaling without per-dimension discrimination is insufficient at large extension ratios.** YaRN's dimension-wise frequency assignment, combined with temperature correction and brief fine-tuning, is the robust solution.

> [!QUESTION] Exercise 6: Combining the Methods
> *This exercise synthesizes the design principles of all three methods.*
>
> > **Prerequisites:** [[#5. Positional Interpolation (PI)|5. Positional Interpolation (PI)]], [[#6.2 Base-Frequency Rescaling|6.2 Base-Frequency Rescaling]], [[#7.2 The Ramp Function and NTK-by-Parts Interpolation|7.2 The Ramp Function and NTK-by-Parts Interpolation]]
>
> Suppose you want to extend a model with $d = 128$, $b = 10{,}000$, $L = 8192$ to $L' = 65536$ (scaling $s = 8$).
>
> (a) Compute the NTK-aware new base $b'$.
> (b) At what dimension index $i$ does the YaRN ramp function $\gamma(r_i) = 0.5$ (i.e., equal blend of interpolation and extrapolation)? Use $\alpha = 1$, $\beta = 32$.
> (c) Compute the YaRN temperature correction $\sqrt{1/t}$ for $s = 8$.

> [!TIP]- Solution to Exercise 6
> **Key insight:** The three quantities — NTK base, YaRN blend crossover, temperature — can all be computed analytically from $s$, $d$, $b$, $L$.
>
> **Sketch:**
>
> (a) $b' = 10000 \cdot 8^{128/126} = 10000 \cdot 8^{1.0159} \approx 10000 \cdot 8.11 \approx 81{,}100$.
>
> (b) $\gamma(r_i) = 0.5$ requires $r_i = \alpha + 0.5(\beta - \alpha) = 1 + 15.5 = 16.5$. So $L/(2\pi b^{2i/d}) = 16.5$, giving $b^{2i/d} = L/(2\pi \cdot 16.5) = 8192/(2\pi \cdot 16.5) \approx 79.0$. Taking $\log_b$: $2i/128 = \log_{10000}(79.0) = \ln(79)/\ln(10000) \approx 4.37/9.21 \approx 0.474$, so $i \approx 0.474 \cdot 64 \approx 30$.
>
> (c) $\sqrt{1/t} = 0.1 \ln(8) + 1 = 0.1 \cdot 2.079 + 1 \approx 1.208$. The effective denominator becomes $1.208\sqrt{128} \approx 13.66$ instead of $\sqrt{128} \approx 11.31$ — roughly a 21% tightening of the attention logit scale.

---

## References

| Reference Name | Brief Summary | Link to Reference |
|----------------|---------------|-------------------|
| Su et al. (2021) — RoFormer | Introduces Rotary Position Embedding (RoPE): block-diagonal rotation matrices encoding relative position in the attention dot product. Demonstrates favorable properties including long-term decay and equivariance. | [arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864) |
| Chen et al. (2023) — Positional Interpolation | Proposes extending context via linear compression of position indices ($m \mapsto m/s$). Proves interpolation upper bound is 600× smaller than extrapolation. Enables LLaMA extension to 32k with <1000 fine-tuning steps. | [arxiv.org/abs/2306.15595](https://arxiv.org/abs/2306.15595) |
| bloc97 (2023) — NTK-Aware Scaled RoPE | Reddit/GitHub post introducing base-frequency rescaling $b' = b \cdot s^{d/(d-2)}$ via NTK theory. No fine-tuning required; good at moderate scaling factors. | [Reddit: LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/) |
| Peng et al. (2023) — YaRN | Introduces dimension-wise frequency interpolation via ramp function $\gamma(r_i)$, plus attention temperature scaling. 10× fewer tokens and 2.5× fewer steps than PI. State-of-the-art at long-context extension. | [arxiv.org/abs/2309.00071](https://arxiv.org/abs/2309.00071) |
| EleutherAI Blog — Rotary Embeddings | Accessible exposition of RoPE mathematics, including complex-number representation and comparison with sinusoidal encodings. | [blog.eleuther.ai/rotary-embeddings](https://blog.eleuther.ai/rotary-embeddings/) |
| EleutherAI Blog — YaRN | Technical blog companion to the YaRN paper, with additional derivations of the NTK base rescaling formula and ramp function design. | [blog.eleuther.ai/yarn](https://blog.eleuther.ai/yarn/) |
| Amara et al. (2025) — How LLMs Scaled from 512 to 2M Context | Survey blog post tracing context extension methods from RoPE through PI, NTK, and YaRN with mathematical detail and code. | [amaarora.github.io](https://amaarora.github.io/posts/2025-09-21-rope-context-extension.html) |
