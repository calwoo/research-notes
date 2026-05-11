# Rotary Position Embeddings (RoPE) and Context Extension

## Table of Contents

- [[#1. Introduction: Why Positional Encodings Need Rethinking|1. Introduction: Why Positional Encodings Need Rethinking]]
- [[#2. RoPE Derivation|2. RoPE Derivation]]
  - [[#2.1 Setup and Desiderata|2.1 Setup and Desiderata]]
  - [[#2.2 The Rotation Matrix Construction|2.2 The Rotation Matrix Construction]]
  - [[#2.3 Complex-Number Shorthand|2.3 Complex-Number Shorthand]]
  - [[#2.4 Frequency Schedule|2.4 Frequency Schedule]]
  - [[#2.5 Why Rotation? Uniqueness of the Construction|2.5 Why Rotation? Uniqueness of the Construction]]
- [[#3. Key Properties|3. Key Properties]]
  - [[#3.1 Relative Position in the Dot Product|3.1 Relative Position in the Dot Product]]
  - [[#3.2 Long-Term Decay|3.2 Long-Term Decay]]
  - [[#3.3 Equivariance to Sequence Shifts|3.3 Equivariance to Sequence Shifts]]
- [[#4. The Context Extension Problem|4. The Context Extension Problem]]
- [[#5. Positional Interpolation (PI)|5. Positional Interpolation (PI)]]
- [[#6. NTK-Aware Scaling|6. NTK-Aware Scaling]]
  - [[#6.1 The NTK Argument — and Its Limits|6.1 The NTK Argument — and Its Limits]]
  - [[#6.2 Base-Frequency Rescaling|6.2 Base-Frequency Rescaling]]
  - [[#6.3 What NTK Scaling Leaves Uncalibrated|6.3 What NTK Scaling Leaves Uncalibrated]]
  - [[#6.4 A Cleaner Lens: Nyquist Sampling Theory|6.4 A Cleaner Lens: Nyquist Sampling Theory]]
  - [[#6.5 Dynamic NTK: Zero-Shot Length Generalization|6.5 Dynamic NTK: Zero-Shot Length Generalization]]
- [[#7. YaRN: Yet Another RoPE ExtensioN|7. YaRN: Yet Another RoPE ExtensioN]]
  - [[#7.1 Dimension-Wise Wavelength Analysis|7.1 Dimension-Wise Wavelength Analysis]]
  - [[#7.2 The Ramp Function and NTK-by-Parts Interpolation|7.2 The Ramp Function and NTK-by-Parts Interpolation]]
  - [[#7.3 Attention Temperature Scaling|7.3 Attention Temperature Scaling]]
  - [[#7.4 Fine-Tuning Recipe|7.4 Fine-Tuning Recipe]]
- [[#8. Large-Base RoPE|8. Large-Base RoPE]]
- [[#9. Comparison and Practical Guidance|9. Comparison and Practical Guidance]]
- [[#References|References]]

---

## 1. Introduction: Why Positional Encodings Need Rethinking 💡

A transformer's self-attention mechanism is *permutation-equivariant* by design: shuffle the input tokens and the output shuffles identically. This is useful for sets, but language is a sequence — the meaning of a sentence depends critically on word order. Positional encodings break this symmetry by injecting order information into the token representations.

The earliest approaches, due to Vaswani et al. (2017), add a fixed sinusoidal signal to the token embeddings before the first layer. This is called *absolute positional encoding* (APE): each position $m$ maps to a deterministic vector, and the model learns to decode order from the additive perturbation. APE has two well-known weaknesses:

1. **Poor generalization to unseen lengths.** The model must have seen position indices up to the maximum training length. Beyond that boundary, position vectors are simply not present in the embedding table (for learned APE) or lie in an extrapolation regime of the sinusoidal function.
2. **No explicit relative bias.** Two tokens that are three positions apart convey the same relational signal regardless of whether they occur near the start or near the end of a sequence. Yet the model must infer this relative structure from absolute signals through learned weights.

*Relative positional encodings* address both issues by encoding the offset $m - n$ between positions directly into the attention computation. RoPE, introduced by [Su et al. (2021)](https://arxiv.org/abs/2104.09864), is now the dominant approach: it achieves relative encoding without extra parameters by rotating query and key vectors by a position-dependent angle before computing their dot product. The rotation is structured so that the inner product $\langle \mathbf{q}_m, \mathbf{k}_n \rangle$ is a function of $m - n$ alone — not of $m$ or $n$ individually. Crucially, RoPE is applied at every layer inside attention, not as a one-time additive offset.

RoPE underpins LLaMA, Mistral, Falcon, Qwen, and most modern open-weight LLMs. Its simplicity (no trainable parameters, purely geometric) and its natural long-term decay property make it attractive. However, as models are deployed on contexts longer than those seen during training, the rotation angles fall outside the trained distribution, leading to *perplexity explosion*. Sections 5–8 cover the main remedies: Positional Interpolation (PI), NTK-Aware Scaling, YaRN, and Large-Base RoPE.

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

### 2.5 Why Rotation? Uniqueness of the Construction 📐

The construction in §2.2 may look like a clever guess: *why should the encoding $f(\mathbf{v}, m)$ be a rotation?* The answer is that, under mild regularity conditions, rotation is essentially the *only* linear encoding satisfying the relative-position desideratum.

**Setting up the uniqueness argument.** Suppose $f(\mathbf{v}, m) = g_m\,\mathbf{v}$ for some matrix $g_m \in \mathbb{R}^{2\times2}$ (restricting to $d = 2$ first). The desideratum requires

$$\langle g_m\,\mathbf{q},\; g_n\,\mathbf{k}\rangle = \mathbf{q}^\top g_m^\top g_n\,\mathbf{k}$$

to depend only on $n - m$ for all $\mathbf{q}, \mathbf{k} \in \mathbb{R}^2$. Since this must hold for *all* $\mathbf{q}$ and $\mathbf{k}$, the matrix $g_m^\top g_n$ itself can depend only on $n - m$:

$$g_m^\top g_n = F(n - m) \qquad \text{for some } F : \mathbb{Z} \to \mathbb{R}^{2\times 2}.$$

**Deriving the rotation structure.** Setting $m = n$: $g_m^\top g_m = F(0)$, constant in $m$. Setting $n = 0$: $F(-m) = g_m^\top g_0$. Applying the functional equation twice:

$$g_m^\top g_n = F(n-m) = g_0^\top g_{n-m}.$$

This is a *cocycle condition*: the product $g_m^\top g_n$ factors through the difference $n - m$. Normalising so that $g_0 = I$ (no rotation at the reference position), the condition becomes $g_m^\top g_n = g_{n-m}$. Setting $n = 0$: $g_m^\top = g_{-m}$, so each $g_m$ is an *orthogonal* matrix. The map $m \mapsto g_m$ is then a group homomorphism from $(\mathbb{Z}, +)$ into $O(2)$.

**Proposition.** *Any group homomorphism $\varphi : (\mathbb{Z}, +) \to SO(2)$ has the form $\varphi(m) = R(m\theta)$ for some $\theta \in \mathbb{R}$, where $R(\phi)$ denotes the $2\times 2$ rotation by angle $\phi$.*

*Proof sketch.* $SO(2) \cong U(1) \cong \mathbb{R}/2\pi\mathbb{Z}$ as groups. A homomorphism from $\mathbb{Z}$ into $\mathbb{R}/2\pi\mathbb{Z}$ is determined by the image of the generator $1 \mapsto \theta \pmod{2\pi}$, which gives $m \mapsto m\theta \pmod{2\pi}$. $\square$

Restricting to orientation-preserving maps ($\det g_m = 1$, i.e., $g_m \in SO(2)$) eliminates reflections. **In 2D, rotation by $m\theta$ is the unique linear encoding satisfying the desideratum with $g_0 = I$ and $\det g_m = 1$.**

**Extension to $\mathbb{R}^d$.** For $d > 2$, the inner product decomposes across pairs of coordinates:

$$\langle f(\mathbf{q}, m),\; f(\mathbf{k}, n) \rangle = \sum_{i=1}^{d/2} \langle g_m^{(i)}\,\tilde{\mathbf{q}}_i,\; g_n^{(i)}\,\tilde{\mathbf{k}}_i\rangle,$$

where $\tilde{\mathbf{v}}_i = (v_{2i-1}, v_{2i})^\top \in \mathbb{R}^2$ is the $i$-th pair. The 2D uniqueness result applies to each block independently: each $g_m^{(i)}$ must be a rotation $R(m\theta_i)$ for some angle $\theta_i$. The block-diagonal structure of $R_m$ follows. **The only freedom remaining is the choice of angles $\{\theta_i\}_{i=1}^{d/2}$ — which is precisely the frequency schedule of §2.4.**

> [!NOTE] What the Uniqueness Argument Does Not Fix
> The proof shows rotation is forced given the desideratum and linearity. It does *not* determine the frequency schedule $\theta_i = b^{-2(i-1)/d}$ — that is a design choice, motivated by spectral coverage (§2.4) and continuity with the sinusoidal APE convention of Vaswani et al. (2017). Different valid schedules yield different RoPE variants; the geometric spectrum is one particular choice.

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

### 6.1 The NTK Argument — and Its Limits

> [!WARNING]
> *The "NTK" label is somewhat misleading. The connection to Neural Tangent Kernel theory is primarily by analogy, not by formal theorem. Understanding what the argument actually claims — and where it breaks down — is essential for interpreting both NTK scaling's successes and its failures at large $s$.*

#### The Position-Encoding Kernel

In NTK theory, a network's function class is determined by the spectral structure of its *kernel*:

$$K(x, x') = \mathbb{E}_\theta\!\left[\langle \nabla_\theta f(x;\theta),\, \nabla_\theta f(x';\theta) \rangle\right].$$

For a RoPE-based attention head, the attention score between tokens at positions $m$ and $n$ is

$$\text{score}(m, n) = \mathbf{q}^\top R_{n-m}\,\mathbf{k} = \sum_{i=1}^{d/2} A_i \cos((n-m)\theta_i) + B_i \sin((n-m)\theta_i),$$

where $A_i, B_i$ depend on $\mathbf{q}$ and $\mathbf{k}$ but not on position. The attention kernel is therefore a *trigonometric polynomial* in relative offset $n - m$, with Fourier modes $\{\cos(\cdot\,\theta_i), \sin(\cdot\,\theta_i)\}_{i=1}^{d/2}$. **The model's ability to represent arbitrary functions of relative position is entirely determined by this frequency set $\{\theta_i\}$.**

#### PI's Problem Through the Kernel Lens

When PI substitutes $m \mapsto m/s$, the effective kernel becomes

$$\text{score}^{\text{PI}}(m, n) = \sum_{i=1}^{d/2} A_i \cos\!\left(\frac{n-m}{s}\theta_i\right) + B_i \sin\!\left(\frac{n-m}{s}\theta_i\right).$$

The kernel now has $1/s$ the original frequency bandwidth. Concretely, the innermost dimension (largest $\theta_i \approx 1$) previously changed by $\approx 1$ radian per token; after PI it changes by $\approx 1/s$ radians per token. For nearby positions ($|n-m| \ll s$), $\cos((n-m)/s) \approx 1$ — the model can no longer distinguish them. **PI destroys high-frequency resolution in exchange for in-distribution angles at long range.**

The NTK framing names this precisely: PI changes what the kernel *sees*, effectively replacing the trained frequency basis $\{\theta_i\}$ with a compressed basis $\{\theta_i/s\}$. The model's function class changes discontinuously with $s$.

#### The NTK Scaling Insight

Rather than compressing positions (changing the kernel's input), change the frequencies $\theta_i$ directly so that the kernel's spectral structure expands to cover the longer range. The goal: at new context length $L'$, the kernel should "look the same" as at training length $L$ — the same frequency basis, just with longer wavelengths. This requires stretching $\{\theta_i\}$ rather than squashing $\{m\}$.

Concretely:
- **High-frequency dimensions** ($\lambda_i \ll L$): angles cycle many times in $[0, L]$. The model interpolates between familiar values even slightly beyond $L$. Leave them untouched.
- **Low-frequency dimensions** ($\lambda_i \gg L$): angles barely complete one cycle. Any $m > \lambda_i$ is entirely unseen. These need the most stretching.

Changing the base $b$ to a larger value $b'$ uniformly stretches all wavelengths $\lambda_i \propto b^{2(i-1)/d}$, with the slowest dimensions stretching proportionally more (because their exponent $2(i-1)/d$ is larger). This is *frequency-aware* in a way that PI is not.

#### Where the Formal Argument Breaks Down

Three issues prevent this from being a rigorous theorem:

1. **NTK requires infinite width.** The NTK theorem governs infinitely-wide networks trained under gradient flow from random initialization. Real transformer heads have finite $d$ and are trained with Adam — the actual kernel evolves throughout training and is not frozen at initialization.

2. **Fine-tuning invalidates the fixed-kernel assumption.** Even the 400 gradient steps YaRN uses re-enters a regime where the NTK is not the relevant object. The argument is about initialization-time function class, not post-training behavior.

3. **The base-rescaling formula encodes an arbitrary design choice.** The exponent $d/(d-2)$ pins the *slowest* dimension's stretch to exactly $s$. There is no NTK theorem that specifies which dimension should be the anchor. The YaRN ablation table (§8) shows that NTK scaling collapses at large $s$ precisely because this single constraint leaves intermediate dimensions simultaneously over-stretched and under-stretched.

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

NTK-aware scaling correctly stretches low-frequency dimensions but applies the *same uniform base change* to all dimensions, including the fast-rotating ones that don't need it. At moderate extensions ($s \leq 4$) the perturbation to fast dimensions is negligible; at large extensions ($s \geq 8$) the accumulated error becomes significant. *Empirically, NTK scaling without fine-tuning achieves good perplexity at $s \leq 4$ but collapses at $s = 8$* (see the perplexity table in §9). The YaRN method addresses this by moving from a single base change to explicit per-dimension control.

### 6.4 A Cleaner Lens: Nyquist Sampling Theory 💡

The NTK framing is useful for motivating *why frequencies matter*, but the *Nyquist-Shannon sampling theorem* provides a cleaner and more honest mathematical basis for both NTK scaling and its successor, YaRN.

#### The Sampling Analogy

Recall the rotation count from §7.1:

$$r_i = \frac{L}{\lambda_i} = \frac{L}{2\pi\,b^{2(i-1)/d}}.$$

$r_i$ is the number of complete revolutions dimension $i$ makes during training on sequences of length $L$. Think of each dimension as a periodic clock with period $\lambda_i$:

| Regime | Condition | Interpretation |
|--------|-----------|----------------|
| Well-sampled | $r_i \gg 1$ | Clock cycles many times. Model has seen all angles $[0, 2\pi]$ repeatedly. Reliable interpolation at any offset. |
| Barely sampled | $r_i \approx 1$ | Exactly one full revolution. Model has seen all angles, but sparsely. |
| Undersampled | $r_i < 1$ | Clock hasn't completed a full revolution. Angles beyond $2\pi r_i$ are entirely unseen. Any relative position $|n-m| > \lambda_i$ is out-of-distribution. |

By the Nyquist-Shannon theorem, a signal of frequency $f$ must be sampled at rate $\geq 2f$ to be recoverable without aliasing. The analog here is that dimension $i$ can only reliably encode relative positions up to $\lambda_i/2$ — beyond that, the rotation angle wraps around in a way the model has never seen. The context extension problem is precisely an *undersampling* problem for the low-frequency (large $\lambda_i$, small $r_i$) dimensions.

#### The Undersampling Condition

The out-of-distribution failure at inference time has a dimension-by-dimension characterization. At training length $L$:

$$r_i < 1 \iff \lambda_i > L \iff i > i^* := \frac{d}{2}\left(1 + \frac{\log(L/2\pi)}{\log b}\right).$$

Dimensions $i > i^*$ are undersampled — they have never completed a full revolution. For the standard setting $b = 10{,}000$, $d = 128$, $L = 4096$, this gives $i^* \approx 43$, so the top $\approx 21$ dimensions are undersampled.

The correct fix is to *slow down* these clocks so that each completes at most one revolution within the new context length $L'$. The ideal intervention: stretch $\lambda_i$ for dimension $i$ such that

$$r_i' = \frac{L'}{\lambda_i'} \lesssim 1 \quad \text{for all formerly undersampled dims.}$$

For well-sampled dimensions ($r_i \gg 1$), no modification is needed.

#### NTK Scaling as an Approximation to the Nyquist Fix

Base rescaling changes $\lambda_i \mapsto \lambda_i \cdot s^{2(i-1)/(d-2)}$. At dimension $i = d/2$ (the slowest clock), the stretch factor is $s^1 = s$, so:

$$r_{d/2}' = \frac{L'}{s\,\lambda_{d/2}} = \frac{sL}{s\,\lambda_{d/2}} = \frac{L}{\lambda_{d/2}} = r_{d/2}.$$

The slowest clock's rotation count is preserved — it now completes the same number of revolutions over $L' = sL$ as it did over $L$. This is exactly the Nyquist prescription applied to the single most critical dimension.

But the base change applies a *continuous gradient* of stretching to all other dimensions, even those with $r_i \gg 1$ that need no modification. At $s = 8$, intermediate dimensions receive a stretch factor of $s^{(d-4)/(d-2)} \approx 6\text{–}7$, perturbing clocks that were already well-calibrated. This accounts for the empirical collapse at large $s$.

#### YaRN as the Nyquist-Optimal Solution

YaRN's ramp function $\gamma(r_i)$ directly implements the Nyquist prescription dimension-by-dimension:

$$\gamma(r_i) = \begin{cases}
0 & r_i < \alpha \quad \text{(undersampled: apply PI)} \\
1 & r_i > \beta \quad \text{(well-sampled: leave alone)} \\
\frac{r_i - \alpha}{\beta - \alpha} & \text{otherwise (blend)}
\end{cases}$$

The threshold $\alpha = 1$ is not a tunable hyperparameter in disguise — it is the **Nyquist boundary**: dimensions with $r_i < 1$ have $\lambda_i > L$, meaning the training data covered less than one full revolution. Any relative position beyond $\lambda_i$ is extrapolation in the strictest sense. Applying pure PI to these dimensions places their angles back within $[0, 2\pi r_i]$, the range the model has seen.

The threshold $\beta = 32$ is empirical ($r_i > 32$ means 32 complete cycles; the model has dense coverage of all angles), but it corresponds to a well-motivated criterion: *sufficiently sampled that perturbation is harmless*.

> [!NOTE]
> The Nyquist analogy is structural, not literal. Nyquist-Shannon is about recovering bandlimited signals from discrete samples; RoPE dimensions are not signals being recovered. The analogy holds at the level of "how many times has the model seen the full angular range of this dimension," not at the level of information-theoretic bit rates. The value is in making the threshold $\alpha = 1$ and the asymmetric treatment of fast vs. slow dimensions mathematically principled rather than heuristic.

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

### 6.5 Dynamic NTK: Zero-Shot Length Generalization 💡

Standard NTK scaling requires committing to a fixed target length $L'$ at deployment — the base $b'$ is computed once and baked into all position computations. *Dynamic NTK* removes this constraint by recomputing the effective base on-the-fly as the sequence grows.

**Definition (Dynamic NTK Scaling).** At inference time, when the current sequence has length $\ell$ tokens, compute the effective base as

$$b'(\ell) = b \cdot \left(\frac{\ell}{L}\right)^{d/(d-2)},$$

where $L$ is the training context length and $b$ is the original base. If $\ell \leq L$, set $b'(\ell) = b$ (no modification). Otherwise, $b'(\ell)$ grows with $\ell$ so that the slowest dimension is always stretched to match the current length.

**The consistency problem.** At step $\ell$, tokens at all positions $0, 1, \ldots, \ell$ are encoded with base $b'(\ell)$. But in a KV-cache setting, tokens $0, \ldots, \ell-1$ were previously encoded with base $b'(\ell-1) \neq b'(\ell)$. The cached keys and values are *inconsistent* with the new base: token $j < \ell$ was rotated by angle $j\theta_i^{(\ell-1)}$ when cached, but the current query expects it to have been rotated by $j\theta_i^{(\ell)}$.

**Why it works anyway.** Despite this inconsistency, dynamic NTK performs well in practice for two reasons:

1. **Fast dimensions are nearly unaffected.** For small $i$ (large $\theta_i$), $\lambda_i \ll L$, so $r_i \gg 1$ at any sequence length. The base rescaling formula stretches $\lambda_i$ by $(\ell/L)^{2(i-1)/(d-2)} \approx 1$ for small $i$ — the angular change $\Delta\theta_i = \theta_i^{(\ell)} - \theta_i^{(\ell-1)}$ is negligible. High-frequency dimensions dominate the QK inner product for nearby tokens, so the inconsistency has little effect on local attention.

2. **Slow dimensions have small absolute angles.** For large $i$ (small $\theta_i$), the rotation angle $m\theta_i$ is small even for large $m$, so the error $m\,|\Delta\theta_i|$ remains bounded relative to $2\pi$.

> [!WARNING]
> *Dynamic NTK is a heuristic — the KV-cache inconsistency is real and accumulates for very long sequences. It is appropriate for serving when the target length is unknown, but YaRN fine-tuning is preferred when the target length is fixed and latency allows a fine-tuning phase. Many inference engines (vLLM, Ollama) implement dynamic NTK as the default zero-shot extension strategy.*

**Relationship to §6.2.** Dynamic NTK is not a different formula — it is the same formula $b' = b \cdot s^{d/(d-2)}$ evaluated at $s = \ell/L$ for each new token. The novelty is in applying this as an online update rather than a one-time configuration.

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

## 8. Large-Base RoPE 🔑

NTK scaling, YaRN, and PI are all *post-hoc* methods: they take a model trained with base $b$ on context length $L$ and extend it at inference or fine-tuning time. An alternative is to simply train with a **much larger base from the start**, so that the frequency grid is pre-stretched to cover the intended extended context.

### 8.1 The Large-Base Intuition

Recall the slowest dimension's wavelength: $\lambda_{d/2} = 2\pi\,b^{(d-2)/d} \approx 2\pi\,b$ for large $d$. For the standard $b = 10{,}000$, $\lambda_{d/2} \approx 62{,}800$ tokens. A model trained on $L = 4096$ has $r_{d/2} \approx 4096/62800 \approx 0.065$ — the slowest clock completes only 6.5% of a revolution during pretraining, yet the model must use it to encode coarse global position.

With $b = 500{,}000$ (as used in LLaMA 3), $\lambda_{d/2} \approx 3.14\text{M}$ tokens. Now:
- At pretraining length $L = 8192$: $r_{d/2} \approx 0.003$ — essentially a static bias, not a periodic signal.
- At extended length $L' = 128{,}000$: $r_{d/2} \approx 0.04$ — still well below 1, so no out-of-distribution angles appear.

**The large-base approach works by ensuring that $r_i < 1$ for slow dimensions even at the intended extended length** — the model is trained in an "always-undersampled" regime for those dimensions, and fine-tuning at $L'$ teaches them their role.

### 8.2 Effect on the Frequency Spectrum

The Nyquist condition from §6.4 becomes: dimension $i$ is undersampled ($r_i < 1$) iff $\lambda_i > L$, i.e.,

$$i > i^*(b, L, d) = \frac{d}{2}\left(1 + \frac{\log(L/2\pi)}{\log b}\right).$$

With $b = 500{,}000$, $d = 128$, $L = 128{,}000$:

$$i^* = 64\left(1 + \frac{\log(128000/2\pi)}{\log(500000)}\right) = 64\left(1 + \frac{\ln(20373)}{13.12}\right) \approx 64\left(1 + 0.748\right) \approx 112.$$

All but 16 of the 64 frequency dimensions are undersampled — the vast majority of the spectrum encodes only coarse-grained positional information. This is acceptable because the model never needs to extrapolate: any sequence up to 128k tokens uses only angles in the range that training covered.

Contrast with the standard base $b = 10{,}000$ at $L = 128{,}000$:

$$i^* = 64\left(1 + \frac{\ln(20373)}{9.21}\right) \approx 64 \cdot 2.065 \approx 132 > 64 = d/2.$$

$i^*$ exceeds $d/2$ — *every single dimension* would be undersampled, meaning the model is extrapolating everywhere. Standard RoPE with $b = 10{,}000$ cannot handle 128k contexts at all.

### 8.3 Tradeoffs vs. Post-Hoc Methods

| Dimension | Large-Base (train-time) | YaRN / NTK (post-hoc) |
|-----------|------------------------|----------------------|
| When applied | Before pretraining | After pretraining |
| Fine-tuning required | Yes (extended pretraining at $L'$) | ~400 steps |
| Perplexity at short contexts | Slightly degraded (slow dims carry less info at short range) | No change from base model |
| Maximum achievable $L'$ | Essentially unlimited (just fine-tune longer) | Bounded by stability of the interpolation formula |
| Flexibility | Must know target $L'$ before pretraining | Can be applied post-hoc to any checkpoint |

> [!WARNING]
> *Large-base RoPE has a subtle cost at short contexts: with $b = 500{,}000$, slow dimensions contribute almost nothing to pairwise scores ($r_i \approx 0$, so $\cos((n-m)\theta_i) \approx 1$ for all offsets). The model loses positional resolution in those dimensions at training time. In practice this is offset by fine-tuning at the extended length, which re-teaches the slow dimensions, but the pretraining perplexity at $L = 8192$ may be slightly higher than with the standard base.*

---

## 9. Comparison and Practical Guidance 🔑

The methods form a progression of increasing sophistication:

| Method | When Applied | Frequency Modification | Fine-tuning Needed | Strengths | Weaknesses |
|--------|-------------|----------------------|--------------------|-----------|------------|
| **Vanilla RoPE** | — | None | N/A | Exact trained behavior | Fails beyond $L$ |
| **PI** (Chen et al. 2023) | Post-hoc | $\theta_i \mapsto \theta_i/s$ (uniform) | ~1000 steps | Stable, simple | Destroys high-freq resolution |
| **NTK Scaling** (bloc97 2023) | Post-hoc | $b \mapsto b \cdot s^{d/(d-2)}$ (global) | Optional | No fine-tuning needed | Degrades at large $s$ |
| **Dynamic NTK** | Online (per-token) | $b'(\ell) = b\cdot(\ell/L)^{d/(d-2)}$ | None | Zero-shot; length-agnostic | KV-cache inconsistency |
| **YaRN** (Peng et al. 2023) | Post-hoc | Per-dim ramp $\alpha_i$; + temperature | ~400 steps | Best perplexity; Nyquist-principled | Requires $\alpha, \beta$ tuning |
| **Large-Base RoPE** (LLaMA 3) | Train-time | $b \gg 10{,}000$ (pre-stretched grid) | Extended pretraining | Unlimited $L'$; no post-hoc hacks | Must know target $L'$ before training; short-context cost |

**Practical guidance:**

- *If serving at variable lengths with no fine-tuning*, use Dynamic NTK. It requires only a one-line change to the base computation per forward pass and gracefully handles unknown target lengths.
- *If the target length is fixed and a small fine-tuning budget is available* ($\sim 400$ steps on $0.1\%$ of pretrain data), YaRN is the best choice and consistently outperforms all alternatives.
- *If training a new model intended for very long contexts*, use a large base ($b \geq 500{,}000$) from the start, then extended-pretrain at $L'$. This is the approach of LLaMA 3 / 3.1.
- *Avoid pure PI* unless the downstream task is insensitive to local positional order (e.g., long-document retrieval). Its uniform frequency compression degrades near-neighbor sensitivity.

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
| Dubey et al. (2024) — LLaMA 3 | Introduces $b = 500{,}000$ large-base RoPE as the training-time alternative to post-hoc extension; extended pretraining to 128k context. | [arXiv:2407.21783](https://arxiv.org/abs/2407.21783) |
| Amara et al. (2025) — How LLMs Scaled from 512 to 2M Context | Survey blog post tracing context extension methods from RoPE through PI, NTK, and YaRN with mathematical detail and code. | [amaarora.github.io](https://amaarora.github.io/posts/2025-09-21-rope-context-extension.html) |
