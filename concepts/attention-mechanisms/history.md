# A History of Attention Mechanisms

## Table of Contents

- [[#1. Origins: Sequence-to-Sequence Models and Alignment (1990s–2014)|1. Origins: Sequence-to-Sequence Models and Alignment (1990s–2014)]]
  - [[#1.1 The Fixed-Vector Bottleneck|1.1 The Fixed-Vector Bottleneck]]
  - [[#1.2 Bahdanau Attention (2014): Additive Alignment|1.2 Bahdanau Attention (2014): Additive Alignment]]
  - [[#1.3 Luong Attention (2015): Multiplicative Variants|1.3 Luong Attention (2015): Multiplicative Variants]]
- [[#2. Scaled Dot-Product Attention and the Transformer (2017)|2. Scaled Dot-Product Attention and the Transformer (2017)]]
  - [[#2.1 The Core Mechanism|2.1 The Core Mechanism]]
  - [[#2.2 Multi-Head Attention|2.2 Multi-Head Attention]]
  - [[#2.3 Positional Encoding|2.3 Positional Encoding]]
  - [[#2.4 Full Self-Attention: Architectural Implications|2.4 Full Self-Attention: Architectural Implications]]
- [[#3. BERT, GPT, and the Rise of Pretraining (2018–2020)|3. BERT, GPT, and the Rise of Pretraining (2018–2020)]]
  - [[#3.1 GPT-1 (2018): Decoder-Only Causal Language Modeling|3.1 GPT-1 (2018): Decoder-Only Causal Language Modeling]]
  - [[#3.2 BERT (2018): Bidirectional Encoder Representations|3.2 BERT (2018): Bidirectional Encoder Representations]]
  - [[#3.3 GPT-2 and GPT-3: Scaling and Emergence (2019–2020)|3.3 GPT-2 and GPT-3: Scaling and Emergence (2019–2020)]]
  - [[#3.4 T5 (2020): Encoder-Decoder with Relative Biases|3.4 T5 (2020): Encoder-Decoder with Relative Biases]]
- [[#4. Efficient Attention: Overcoming the Quadratic Bottleneck (2020–2022)|4. Efficient Attention: Overcoming the Quadratic Bottleneck (2020–2022)]]
  - [[#4.1 Sparse Attention: Longformer and BigBird|4.1 Sparse Attention: Longformer and BigBird]]
  - [[#4.2 FlashAttention: IO-Aware Exact Attention (2022)|4.2 FlashAttention: IO-Aware Exact Attention (2022)]]
  - [[#4.3 Multi-Query and Grouped-Query Attention|4.3 Multi-Query and Grouped-Query Attention]]
- [[#5. Linear Attention and Recurrent Reformulations (2020–2023)|5. Linear Attention and Recurrent Reformulations (2020–2023)]]
  - [[#5.1 Katharopoulos et al. (2020): Transformers are RNNs|5.1 Katharopoulos et al. (2020): Transformers are RNNs]]
  - [[#5.2 RWKV (2023): Attention-Free Transformer|5.2 RWKV (2023): Attention-Free Transformer]]
  - [[#5.3 RetNet (2023): Constant Decay and Three Computation Modes|5.3 RetNet (2023): Constant Decay and Three Computation Modes]]
- [[#6. State Space Models and the Mamba Line (2022–2024)|6. State Space Models and the Mamba Line (2022–2024)]]
  - [[#6.1 S4 (2022): Structured State Spaces|6.1 S4 (2022): Structured State Spaces]]
  - [[#6.2 Mamba (2023): Selective State Space Models|6.2 Mamba (2023): Selective State Space Models]]
  - [[#6.3 Mamba-2 / SSD (2024): Explicit Duality with Linear Attention|6.3 Mamba-2 / SSD (2024): Explicit Duality with Linear Attention]]
- [[#7. Gating and Hybrid Architectures (2024–present)|7. Gating and Hybrid Architectures (2024–present)]]
  - [[#7.1 GLA (2024): Vector-Valued Data-Dependent Gating|7.1 GLA (2024): Vector-Valued Data-Dependent Gating]]
  - [[#7.2 RWKV-6 and Eagle/Finch: Matrix-Valued Gating|7.2 RWKV-6 and Eagle/Finch: Matrix-Valued Gating]]
  - [[#7.3 Hybrid Models: Interleaving Attention and Recurrence|7.3 Hybrid Models: Interleaving Attention and Recurrence]]
  - [[#7.4 The Neural Memory Perspective|7.4 The Neural Memory Perspective]]
- [[#8. Open Problems and Future Directions|8. Open Problems and Future Directions]]
- [[#9. References|9. References]]

---

## 1. Origins: Sequence-to-Sequence Models and Alignment (1990s–2014)

### 1.1 The Fixed-Vector Bottleneck

The dominant paradigm for sequence transduction before 2014 was the *encoder-decoder* architecture built on *recurrent neural networks* (RNNs) or, more successfully, *Long Short-Term Memory* (LSTM) networks. The encoder reads an input sequence $\mathbf{x} = (x_1, \ldots, x_{T_x})$ one token at a time, updating a hidden state:

$$\mathbf{h}_t = f(\mathbf{h}_{t-1}, x_t)$$

After reading all $T_x$ tokens the final hidden state $\mathbf{h}_{T_x}$ — a single fixed-dimensional vector — is passed to the decoder as the entire summary of the input. The decoder then generates output tokens autoregressively, conditioned only on this single context vector.

This design suffers from a clear information bottleneck: the entire variable-length input must be compressed into a vector of constant dimension. Empirically, performance on machine translation degrades sharply as source sentence length grows. This observation motivated the first attention mechanisms.

### 1.2 Bahdanau Attention (2014): Additive Alignment

Bahdanau, Cho, and Bengio (2015, published from a 2014 preprint) proposed resolving the fixed-vector bottleneck by allowing the decoder, at each output step $i$, to form a *weighted combination* of all encoder hidden states rather than relying on a single summary vector.

**Setup.** The encoder is a *bidirectional RNN*. For each source position $j$, the forward and backward hidden states are concatenated to form an *annotation*:

$$\mathbf{h}_j = \left[\overrightarrow{\mathbf{h}}_j\,;\,\overleftarrow{\mathbf{h}}_j\right] \in \mathbb{R}^{2n}$$

This gives each annotation access to context from both directions.

**Alignment score.** At decoder step $i$ with previous hidden state $\mathbf{s}_{i-1}$, the *alignment score* between the decoder state and encoder annotation $j$ is computed by a small feed-forward network parameterized by $\mathbf{W}_a \in \mathbb{R}^{n \times n}$, $\mathbf{U}_a \in \mathbb{R}^{n \times 2n}$, and $\mathbf{v}_a \in \mathbb{R}^n$:

$$e_{ij} = \mathbf{v}_a^\top \tanh\!\left(\mathbf{W}_a \mathbf{s}_{i-1} + \mathbf{U}_a \mathbf{h}_j\right)$$

This is called *additive attention* because it combines the decoder state and encoder annotation by addition inside the nonlinearity.

**Attention weights.** The scores are normalized into a probability distribution over source positions:

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}$$

**Context vector.** The context for output step $i$ is the convex combination of annotations:

$$\mathbf{c}_i = \sum_{j=1}^{T_x} \alpha_{ij}\, \mathbf{h}_j$$

**Decoder.** The decoder hidden state and output distribution are conditioned on the dynamic context:

$$\mathbf{s}_i = f(\mathbf{s}_{i-1},\, y_{i-1},\, \mathbf{c}_i), \qquad p(y_i \mid y_{<i}, \mathbf{x}) = g(y_{i-1},\, \mathbf{s}_i,\, \mathbf{c}_i)$$

The key insight: attention acts as a *soft alignment*, allowing the model to automatically discover which source words are relevant for each output word without requiring a hard monotonic alignment. The alignment matrix $\{\alpha_{ij}\}$ can be visualized and often recovers linguistically meaningful word correspondences.

### 1.3 Luong Attention (2015): Multiplicative Variants

Luong, Pham, and Manning (2015) simplified and generalized the alignment function. Their *global attention* model evaluates the alignment between the current decoder state $\mathbf{h}_t$ and each encoder state $\bar{\mathbf{h}}_s$ using one of three score functions:

$$\text{score}(\mathbf{h}_t, \bar{\mathbf{h}}_s) = \begin{cases} \mathbf{h}_t^\top \bar{\mathbf{h}}_s & \text{(dot)} \\ \mathbf{h}_t^\top \mathbf{W}_a\, \bar{\mathbf{h}}_s & \text{(general)} \\ \mathbf{v}_a^\top \tanh(\mathbf{W}_a[\mathbf{h}_t\,;\,\bar{\mathbf{h}}_s]) & \text{(concat)} \end{cases}$$

The *dot* and *general* variants are *multiplicative attention*: they replace the sum of two linearly transformed vectors with a bilinear product. This is computationally cheaper than additive attention (no tanh) and, as Vaswani et al. would later show, scales more naturally to high-dimensional settings.

Luong et al. also introduced *local attention*, which restricts the context window to a neighborhood $[p_t - D,\, p_t + D]$ around a predicted alignment position $p_t$, reducing computation for long sequences. This idea of attending over a local window reappears in Longformer's sliding-window attention (Section 4.1).

---

## 2. Scaled Dot-Product Attention and the Transformer (2017)

Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, and Polosukhin (2017) — "Attention Is All You Need" — eliminated recurrence entirely. Rather than using attention as an auxiliary mechanism to augment an RNN encoder-decoder, they built an architecture composed *solely* of attention and feed-forward layers.

### 2.1 The Core Mechanism

**Definition (Scaled Dot-Product Attention).** Given query matrix $Q \in \mathbb{R}^{T \times d_k}$, key matrix $K \in \mathbb{R}^{S \times d_k}$, and value matrix $V \in \mathbb{R}^{S \times d_v}$, the attention output is:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

The division by $\sqrt{d_k}$ is not cosmetic. Assume query and key components are i.i.d. with mean $0$ and variance $1$. The dot product $\mathbf{q}^\top \mathbf{k} = \sum_{i=1}^{d_k} q_i k_i$ then has variance $d_k$. For large $d_k$, these large-magnitude logits drive the softmax into a near-one-hot regime where gradients vanish. Scaling by $1/\sqrt{d_k}$ restores unit variance to the logits regardless of dimension, stabilizing training. See [[note#2. Single-Head Scaled Dot-Product Attention|§2 of note.md]] for a full derivation.

Compared to Luong's general (bilinear) score, scaled dot-product attention is a special case where the weight matrix $\mathbf{W}_a$ is constrained to the identity scaled by $1/\sqrt{d_k}$. The key difference from Bahdanau's additive score is computational: no tanh evaluation, and the full $QK^\top$ matrix can be computed as a single matrix multiply.

### 2.2 Multi-Head Attention

Rather than computing one attention function over the full dimension $D$, the Transformer projects queries, keys, and values into $H$ lower-dimensional subspaces and runs attention in parallel:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H)\, W^O$$

$$\text{head}_i = \text{Attention}(Q W_i^Q,\, K W_i^K,\, V W_i^V)$$

where $W_i^Q, W_i^K \in \mathbb{R}^{D \times d_k}$, $W_i^V \in \mathbb{R}^{D \times d_v}$, $W^O \in \mathbb{R}^{H d_v \times D}$, and $d_k = d_v = D/H$ in the original architecture. Each head can specialize in a different type of relationship — syntactic, semantic, positional — without interference from the other heads.

### 2.3 Positional Encoding

Because attention has no inherent notion of order (it is a set operation over tokens), position information must be injected explicitly. Vaswani et al. add fixed sinusoidal *positional encodings* to the input embeddings:

$$\text{PE}_{(\text{pos},\, 2i)} = \sin\!\left(\frac{\text{pos}}{10000^{2i/D}}\right), \qquad \text{PE}_{(\text{pos},\, 2i+1)} = \cos\!\left(\frac{\text{pos}}{10000^{2i/D}}\right)$$

Different frequencies encode different scales of relative position. Nearby positions produce similar encoding vectors; the inner product between two position encodings depends only on their offset. This observation was later generalized by T5's learned relative biases (Section 3.4) and by the Rotary Position Embedding (RoPE) used in LLaMA and subsequent work.

### 2.4 Full Self-Attention: Architectural Implications

The decisive departure from prior work is using attention for *self-attention*: tokens attend to each other within the same sequence, not merely from decoder to encoder. This removes the sequential dependency of RNNs: every attention output can be computed in parallel from the input sequence, reducing training-time sequential depth from $O(T)$ to $O(1)$ at the cost of $O(T^2)$ compute and memory per layer.

The Transformer encoder uses full bidirectional self-attention (each token attends to all others). The decoder uses *causal self-attention* (each token attends only to itself and prior tokens), plus cross-attention to the encoder outputs.

**The path-length argument.** In an RNN, the number of sequential operations required to relate token $i$ to token $j$ is $|i - j|$, creating long gradient paths for distant tokens. In a Transformer, this path length is $O(1)$ — any two tokens interact directly in the attention matrix.

---

## 3. BERT, GPT, and the Rise of Pretraining (2018–2020)

The Transformer architecture was immediately recognized as a powerful general-purpose sequence model. Within months of "Attention Is All You Need," the NLP community had reorganized around a new paradigm: *pretraining* a large Transformer on unlabeled text, then *fine-tuning* on downstream tasks.

### 3.1 GPT-1 (2018): Decoder-Only Causal Language Modeling

Radford et al. (OpenAI, 2018) introduced GPT-1 (*Generative Pre-trained Transformer*), which takes the decoder half of the original Transformer and drops the cross-attention to the encoder. The result is a 12-layer decoder-only model with causal (left-to-right) self-attention, pretrained on the BooksCorpus dataset with a standard *next-token prediction* objective:

$$\mathcal{L}_{\text{LM}} = -\sum_t \log p(x_t \mid x_{t-k}, \ldots, x_{t-1};\, \Theta)$$

Causal masking ensures that the attention at position $t$ cannot see tokens at positions $> t$, enforcing the autoregressive constraint. GPT-1 demonstrated that a single pretrained model with a small task-specific head could match or exceed task-specific models on several benchmarks, establishing the fine-tuning paradigm.

### 3.2 BERT (2018): Bidirectional Encoder Representations

Devlin, Chang, Lee, and Toutanova (BERT, 2018) argued that GPT-1's unidirectional attention was suboptimal for understanding tasks, where context from both sides of a token is informative. BERT uses the *encoder* half of the Transformer with *full bidirectional self-attention* — every token can attend to every other token.

Training bidirectionally on a language modeling objective is impossible in the standard next-token form, since the model could trivially copy the current token. BERT instead uses two pretraining tasks:

1. *Masked Language Modeling* (MLM): 15% of tokens are replaced with a `[MASK]` token; the model must predict the original token. This forces the model to use bidirectional context.
2. *Next Sentence Prediction* (NSP): The model is given two sentence segments and must predict whether the second follows the first in the original document.

BERT-Large (24 layers, 1024 hidden, 16 heads, 340M parameters) significantly outperformed GPT-1 on the GLUE benchmark and became the standard pretrained backbone for NLP tasks through 2019.

The architectural contrast is clean: GPT uses causal attention and is suited to generation; BERT uses bidirectional attention and is suited to classification and extraction. Later work (XLNet, RoBERTa, ALBERT) would refine both pretraining objectives and architectural details, but the encoder/decoder split established by GPT and BERT remained the dominant taxonomy.

### 3.3 GPT-2 and GPT-3: Scaling and Emergence (2019–2020)

GPT-2 (Radford et al., 2019) scaled GPT-1 from 117M to 1.5B parameters, applied on a higher-quality web-scraped dataset (WebText), and made minor architectural changes: layer normalization moved to the input of each sublayer, and an additional layer norm added after the final attention block. The central finding was that *scale itself* was a competitive approach — GPT-2 achieved strong zero-shot performance on multiple tasks without any fine-tuning.

GPT-3 (Brown et al., 2020) scaled further to 175B parameters. At this scale, the model exhibited *few-shot learning*: given a natural-language task description and a handful of input-output examples in the prompt, GPT-3 could perform translation, question answering, code generation, and arithmetic without any gradient updates. **GPT-3 showed that scaling decoder-only causal attention models produces qualitatively new capabilities unavailable at smaller scales.** This observation drove subsequent work on scaling laws and large-scale pretraining.

### 3.4 T5 (2020): Encoder-Decoder with Relative Biases

Raffel et al. (2020) introduced T5 (*Text-to-Text Transfer Transformer*), which reformulated every NLP task as text-to-text: the input is a string, the output is a string. T5 uses the full encoder-decoder Transformer architecture and made one notable architectural contribution: replacing the absolute sinusoidal positional encodings of Vaswani et al. with *relative position biases*. A scalar bias $b(i - j)$ is added to each attention logit based only on the offset between query position $i$ and key position $j$:

$$\tilde{e}_{ij} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_k}} + b(i - j)$$

The bias values are shared across layers but learned per head, with a small number of learned buckets covering different offset ranges. Relative biases extrapolate more gracefully to sequence lengths longer than those seen during training, a desirable property for long-context tasks.

---

## 4. Efficient Attention: Overcoming the Quadratic Bottleneck (2020–2022)

By 2020, the quadratic $O(T^2)$ cost of full self-attention was the primary constraint on context length. A long context is necessary for document-level understanding, multi-turn dialogue, and code reasoning. Several directions attacked this bottleneck simultaneously.

### 4.1 Sparse Attention: Longformer and BigBird

The *Longformer* (Beltagy et al., 2020) and *BigBird* (Zaheer et al., 2020) replace the full $T \times T$ attention matrix with a sparse pattern that achieves $O(T)$ total attention operations while preserving the ability to capture global context.

**Longformer's pattern** combines:
- A *sliding-window* component: each token attends to a window of $w$ neighbors, contributing $O(T \cdot w)$ operations total.
- *Global tokens* (e.g., the `[CLS]` token in classification): a small set of designated tokens attend to and are attended by every other token, at $O(T \cdot g)$ cost for $g$ global tokens.

**BigBird** extends this with a third component: *random attention*, where each token attends to $r$ uniformly sampled tokens. BigBird's theoretical analysis shows that the combination of local, global, and random attention is a *universal approximator* of full attention and can simulate any function computable by a Turing machine, providing a theoretical basis for why sparse attention suffices.

*Both models achieve $O(T)$ complexity*, but at a cost: they require custom CUDA kernels and the sparse pattern must be fixed at model design time, limiting flexibility.

### 4.2 FlashAttention: IO-Aware Exact Attention (2022)

Dao, Fu, Ermon, Rudra, and Ré (2022) introduced *FlashAttention*, which is not an approximation to attention — it computes the exact same output as standard attention — but restructures the computation to minimize data movement between GPU memory tiers.

The key observation is that modern GPUs have a two-tier memory hierarchy:
- *High Bandwidth Memory* (HBM): large (40–80 GB), bandwidth ~1.5–2 TB/s.
- *On-chip SRAM*: small (~20 MB), bandwidth ~19 TB/s — roughly 10× faster.

Standard attention materializes the full $T \times T$ attention weight matrix $P = \text{softmax}(QK^\top / \sqrt{d_k})$ in HBM, then reads it back to compute $PV$. For $T = 4096$ and $d_k = 64$, this matrix is $4096^2 \times 4 \approx 67$ MB — large enough that HBM bandwidth, not arithmetic throughput, limits wall-clock speed.

FlashAttention avoids materializing $P$ by *tiling*: it partitions $Q$, $K$, $V$ into blocks that fit in SRAM, uses the *online softmax* trick to accumulate partial outputs block by block, and writes only the final output to HBM. The online softmax trick maintains running statistics $(m, \ell)$ — the running maximum and the sum of exponentials — that allow rescaling accumulated partial sums when the global maximum becomes known:

$$m(x_1, x_2) = \max(m(x_1), m(x_2)), \qquad \ell(x_1, x_2) = e^{m(x_1) - m} \ell(x_1) + e^{m(x_2) - m} \ell(x_2)$$

**FlashAttention reduces HBM reads/writes from $O(T^2)$ to $O(T^2 / M)$** (where $M$ is the SRAM size), achieving 2–4× wall-clock speedup and reducing memory from $O(T^2)$ to $O(T)$ for the intermediate attention matrices. Arithmetic work is slightly increased, but the bottleneck is IO, not compute.

FlashAttention-2 (2023) and FlashAttention-3 (2024) refined the tiling strategy and improved occupancy on H100 GPUs. The IO-aware design principle influenced virtually all subsequent efficient attention implementations. See [[note#5. KV Caching|§5 of note.md]] for the KV cache context.

### 4.3 Multi-Query and Grouped-Query Attention

A distinct bottleneck is the *KV cache* during autoregressive inference: for $H$ heads, each of dimension $d_k + d_v$, and sequence length $T$, the KV cache holds $2 \cdot H \cdot T \cdot d_k$ floats per layer. At $T = 32768$ and $H = 32$, $d_k = 128$, this is over 8 GB per layer pair — severely limiting batch size and throughput.

*Multi-query attention* (MQA; Shazeer, 2019) addresses this by sharing a single key head and a single value head across all query heads. Each query head has its own projection $W_i^Q$, but all heads share the same $W^K$ and $W^V$. This reduces the KV cache by a factor of $H$ at the cost of some quality.

*Grouped-query attention* (GQA; Ainslie et al., 2023) interpolates between MHA and MQA: query heads are partitioned into $G$ groups, each group sharing one KV head. With $G = H$ this recovers MHA; with $G = 1$ it recovers MQA. GQA achieves near-MHA quality with near-MQA memory cost and was adopted in LLaMA-2, Mistral, and Gemini.

---

## 5. Linear Attention and Recurrent Reformulations (2020–2023)

FlashAttention and sparse attention reduce the practical cost of full attention but do not change its asymptotic character: at inference, the KV cache still grows with sequence length, and each generation step requires reading the full cache. A more radical alternative is to replace softmax attention with a computation that maintains a *fixed-size state* regardless of $T$.

### 5.1 Katharopoulos et al. (2020): Transformers are RNNs

Katharopoulos, Vyas, Pappas, and Fleuret (2020) observed that the quadratic cost of softmax attention comes entirely from the normalization: the softmax couples all positions $j$ through the denominator, preventing the sum over $j$ from being factored out of the query term.

Their fix: replace the softmax kernel $\exp(\mathbf{q}\mathbf{k}^\top / \sqrt{d_k})$ with a *factored* similarity $\phi(\mathbf{q}_t)\phi(\mathbf{k}_j)^\top$, where $\phi$ is a feature map. For any $\phi$, the output at position $t$ becomes:

$$\mathbf{o}_t = \frac{\phi(\mathbf{q}_t) \left(\displaystyle\sum_{j=1}^{t} \phi(\mathbf{k}_j)^\top \mathbf{v}_j\right)}{\phi(\mathbf{q}_t) \left(\displaystyle\sum_{j=1}^{t} \phi(\mathbf{k}_j)^\top\right)}$$

Since $\phi(\mathbf{q}_t)$ does not depend on $j$, it factors out, reducing the summation to $O(T \cdot d_k d_v)$ rather than $O(T^2)$.

The running sum $S_t = \sum_{j=1}^{t} \phi(\mathbf{k}_j)^\top \mathbf{v}_j \in \mathbb{R}^{d_k \times d_v}$ satisfies an exact recurrence:

$$S_t = S_{t-1} + \phi(\mathbf{k}_t)^\top \mathbf{v}_t, \qquad \mathbf{o}_t = \frac{\phi(\mathbf{q}_t) S_t}{\phi(\mathbf{q}_t) z_t}$$

where $z_t = z_{t-1} + \phi(\mathbf{k}_t)^\top \in \mathbb{R}^{d_k}$ is the normalizer accumulator. This is an RNN with state $S_t$ of dimension $d_k \times d_v$ — *independent of $T$*. The paper proposed $\phi(\mathbf{x}) = \text{elu}(\mathbf{x}) + 1$ to ensure positive similarities, which they showed was necessary for convergence.

The paper titled their work "Transformers are RNNs," making explicit that causal linear attention and RNN inference are two views of the same computation. The full derivation and connection to the *state matrix* are developed in [[linear-attention#2. Linear Attention: Removing Softmax|§2 of linear-attention.md]].

### 5.2 RWKV (2023): Attention-Free Transformer

Peng et al. (2023) introduced *RWKV* (Receptance Weighted Key Value), which aims to achieve the parallelism of a Transformer during training while having RNN-like $O(1)$ inference cost.

The core component is the *WKV* operator. At position $t$ in the time-mixing block, receptance $r_t$, key $k_t$, and value $v_t$ are computed via token-shifted linear projections:

$$r_t = W_r \cdot (\mu_r \odot x_t + (1 - \mu_r) \odot x_{t-1})$$
$$k_t = W_k \cdot (\mu_k \odot x_t + (1 - \mu_k) \odot x_{t-1})$$
$$v_t = W_v \cdot (\mu_v \odot x_t + (1 - \mu_v) \odot x_{t-1})$$

The WKV computation, with a position-independent exponential decay $w \in \mathbb{R}^d$ (channel-wise) and a learnable first-position bias $u$, is:

$$\text{wkv}_t = \frac{\displaystyle\sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i} \odot v_i \;+\; e^{u + k_t} \odot v_t}{\displaystyle\sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i} \;+\; e^{u + k_t}}$$

This can be computed recurrently in $O(d)$ per step. The final output of the time-mixing block is $\sigma(r_t) \odot \text{wkv}_t$.

RWKV is notable for scaling to 14B parameters while maintaining purely sequential recurrence at inference — the first RNN-lineage model to operate at this scale. However, its decay $w$ is *input-independent*, meaning the model cannot selectively forget or retain information based on content. This is the key limitation addressed by Mamba and GLA.

### 5.3 RetNet (2023): Constant Decay and Three Computation Modes

Sun et al. (2023) introduced *RetNet* (Retention Network), which formalized the *retention* mechanism as a principled replacement for attention. The retention mechanism with scalar per-head decay $\gamma \in (0, 1)$ has a dual representation.

**Recurrent form.** Let $S_n \in \mathbb{R}^{d_k \times d_v}$ be the state matrix:

$$S_n = \gamma S_{n-1} + \mathbf{k}_n^\top \mathbf{v}_n, \qquad \mathbf{o}_n = \mathbf{q}_n S_n$$

**Parallel form.** After applying complex exponentials to queries and keys for positional encoding (using angle $\theta$), the output can be written as a masked matrix product:

$$\mathbf{o}_n = \sum_{m=1}^{n} \gamma^{n-m} (\mathbf{q}_n e^{in\theta}) (\mathbf{k}_m e^{im\theta})^\dagger \mathbf{v}_m$$

**Chunkwise form.** The sequence is split into chunks of size $C$. Within each chunk the parallel (quadratic) form is used; state matrices are passed recurrently between chunks. This provides a hardware-efficient middle ground.

RetNet uses multi-scale decay with different $\gamma$ values per head: $\gamma_h = 1 - 2^{-5-h}$ for $h = 0, \ldots, H-1$. Different heads thus operate with different effective context windows.

**The limitation of constant decay is that $\gamma$ is fixed for all positions and all inputs.** The model cannot learn to retain some information indefinitely while forgetting other information based on content — a fundamental expressiveness constraint addressed by later gated variants (see [[linear-attention#5. Decay and Gating Mechanisms|§5 of linear-attention.md]]).

---

## 6. State Space Models and the Mamba Line (2022–2024)

State space models developed in parallel with the linear attention literature, sharing the goal of $O(T)$ training and $O(1)$ inference, but approaching the problem from continuous-time dynamical systems rather than attention.

### 6.1 S4 (2022): Structured State Spaces

Gu, Goel, and Ré (2022) introduced *S4* (Structured State Space for Sequences), grounding sequence modeling in continuous-time linear dynamical systems. The *state space model* (SSM) maps a 1-D input signal $u(t)$ to output $y(t)$ via an $N$-dimensional latent state $x(t)$:

$$x'(t) = A x(t) + B u(t), \qquad y(t) = C x(t) + D u(t)$$

where $A \in \mathbb{R}^{N \times N}$, $B \in \mathbb{R}^{N \times 1}$, $C \in \mathbb{R}^{1 \times N}$, $D \in \mathbb{R}$. To apply this to discrete sequences, the continuous system is discretized via the bilinear (Tustin) method with step size $\Delta$:

$$\bar{A} = (I - \tfrac{\Delta}{2} A)^{-1}(I + \tfrac{\Delta}{2} A), \qquad \bar{B} = (I - \tfrac{\Delta}{2} A)^{-1} \Delta B$$

The discretized system is a linear recurrence $x_t = \bar{A} x_{t-1} + \bar{B} u_t$, and can equivalently be written as a global convolution: $y = \bar{C} \cdot (k * u)$ where the *SSM kernel* $k_t = \bar{C} \bar{A}^t \bar{B}$ can be precomputed for all $t$ simultaneously.

The critical challenge is computing powers of $A$ efficiently. S4's key innovation is initializing $A$ with the *HiPPO* matrix, which has a closed-form Normal Plus Low-Rank (NPLR) decomposition:

$$A_{nk} = -\begin{cases} \sqrt{(2n+1)(2k+1)} & n > k \\ n + 1 & n = k \\ 0 & n < k \end{cases}$$

This structure allows the SSM convolution kernel to be computed in $O(N + L)$ operations (where $L$ is sequence length) via frequency-domain techniques and the Cauchy kernel reduction, instead of the naïve $O(N^2 L)$. **S4 achieved near-perfect accuracy on sequential MNIST and strong results on the Long Range Arena benchmark, demonstrating that structured SSMs could match or exceed Transformers on long-context tasks.**

### 6.2 Mamba (2023): Selective State Space Models

Gu and Dao (2023) identified a fundamental limitation of S4 and all prior SSMs: the matrices $B$, $C$, and $\Delta$ are *input-independent* — they are fixed functions of the position, not the token content. This means the SSM performs the same filtering operation on every sequence, regardless of what it contains. In particular, it cannot selectively copy or ignore specific tokens based on their content — a capability that is critical for language modeling.

*Mamba* introduces *selective SSMs*: the matrices $B_t$, $C_t$, and $\Delta_t$ are now functions of the input $x_t$ at each position:

$$B_t = \text{Linear}(x_t), \quad C_t = \text{Linear}(x_t), \quad \Delta_t = \text{softplus}(\text{Linear}(x_t))$$

The discretized state transitions thus depend on the input:

$$\bar{A}_t = \exp(\Delta_t \odot A), \qquad \bar{B}_t = \Delta_t \odot B_t$$

$$h_t = \bar{A}_t \odot h_{t-1} + \bar{B}_t x_t, \qquad y_t = C_t^\top h_t$$

(Here $A$ is diagonal so $\exp(\Delta_t \odot A)$ is elementwise; the per-channel structure keeps computation tractable.)

The price of input-dependence is that the recurrence can no longer be unrolled as a global convolution (since the kernel changes at every step), so S4's convolutional training strategy is unavailable. Mamba instead uses a *parallel scan* (prefix-sum) algorithm: the $T$ recurrence steps are computed with $O(\log T)$ sequential depth using the associativity of the state update, materialized in a single pass over the sequence in parallel.

To avoid materializing the full state sequence in slow HBM memory, Mamba uses a *hardware-aware scan*: all intermediate states are kept in SRAM and fused into a single kernel, analogous to FlashAttention's tiling strategy.

### 6.3 Mamba-2 / SSD (2024): Explicit Duality with Linear Attention

Dao and Gu (2024) introduced *Mamba-2*, whose theoretical contribution is the *State Space Duality* (SSD) framework: a formal proof that a specific class of selective SSMs and a specific class of masked attention are two different algorithms for the *same sequence transformation*.

The key restriction enabling the duality is that $A$ must be a *scalar times the identity* at each timestep (rather than a general diagonal matrix as in Mamba-1). Under this constraint, the SSM state update is:

$$h_t = a_t h_{t-1} + B_t x_t, \qquad y_t = C_t^\top h_t$$

where $a_t \in \mathbb{R}$ is a scalar. The output can then be written as:

$$y_t = \sum_{j=1}^{t} \left(\prod_{i=j+1}^{t} a_i\right) C_t^\top B_j x_j$$

The $T \times T$ matrix $M$ with $M_{tj} = \left(\prod_{i=j+1}^{t} a_i\right) C_t^\top B_j$ is a *1-semiseparable* matrix: every submatrix contained within the lower triangular part has rank at most 1. The SSD attention form is then:

$$Y = M \cdot X, \qquad M = L \circ (CB^\top)$$

where $L$ is the scalar cumulative-product mask and $\circ$ denotes elementwise multiplication. This is exactly masked attention with a structured (semiseparable) mask — the same computation as linear attention with a scalar decay, expressed as a matrix product.

The chunkwise algorithm for SSD partitions $M$ into $Q \times Q$ blocks. Diagonal blocks are computed using the quadratic (attention-like) form. Off-diagonal blocks factor as low-rank products using the semiseparable property and are accumulated via the recurrence. **This unified view enabled Mamba-2 to be 2–8× faster than Mamba-1 with simpler code, while making explicit that Mamba-2, RetNet, and linear attention are all instances of the same structural family.** See [[linear-attention#5. Decay and Gating Mechanisms|§5 of linear-attention.md]] for the gating perspective.

---

## 7. Gating and Hybrid Architectures (2024–present)

### 7.1 GLA (2024): Vector-Valued Data-Dependent Gating

Yang et al. (2024) introduced *GLA* (Gated Linear Attention), which extends the scalar decay of RetNet/Mamba-2 to a *vector-valued* data-dependent gate. The state update is:

$$S_t = G_t \odot S_{t-1} + \mathbf{k}_t^\top \mathbf{v}_t, \qquad \mathbf{o}_t = \mathbf{q}_t S_t$$

where $G_t \in \mathbb{R}^{d_k \times d_v}$ is a gate matrix that is computed from the input at position $t$ via a low-rank factorization:

$$G_t = \alpha_t^\top \beta_t, \quad \alpha_t = \sigma(\text{Linear}_{d_k}(x_t)), \quad \beta_t = \sigma(\text{Linear}_{d_v}(x_t))$$

Here $\sigma$ is the sigmoid function and $\odot$ denotes elementwise multiplication. Compared to RetNet's fixed scalar $\gamma$ or Mamba-2's scalar $a_t$, GLA allows each entry of the state matrix to decay at a different, input-dependent rate — a strictly more expressive gating structure.

However, because $G_t$ varies per position, the chunkwise training algorithm must be modified. Within a chunk, the quadratic form cannot use a simple constant mask; instead the mask entries involve products of the gates across positions. Yang et al. develop a *hardware-efficient chunkwise algorithm* implemented in Triton that achieves competitive throughput on modern GPUs despite this added complexity. See [[linear-attention#5. Decay and Gating Mechanisms|§5 of linear-attention.md]] for the full GLA derivation.

### 7.2 RWKV-6 and Eagle/Finch: Matrix-Valued Gating

RWKV-6 (the Eagle and Finch variants, Peng et al., 2024) extends the original RWKV architecture by replacing the channel-wise scalar decay with a *matrix-valued gate* computed as an outer product:

$$G_t = \mathbf{a}_t \otimes \mathbf{b}_t, \quad \mathbf{a}_t, \mathbf{b}_t \in \mathbb{R}^d$$

This gives each state matrix entry $(i, j)$ a gate that depends on both the $i$-th component of the key projection and the $j$-th component of the value projection, allowing finer-grained control over which memories are retained. RWKV-6 also incorporates low-rank adaptations of the time-mixing weights, improving parameter efficiency. The motivation parallels GLA: the original RWKV's fixed decay was a computational convenience that hurt expressiveness on tasks requiring selective retention.

### 7.3 Hybrid Models: Interleaving Attention and Recurrence

Despite the expressiveness improvements of selective SSMs and gated linear attention, empirical evidence shows that full softmax attention still outperforms linear attention variants on tasks requiring sharp, content-based retrieval from long contexts — particularly *needle-in-a-haystack* retrieval benchmarks. This has motivated a class of *hybrid architectures* that interleave a small number of full attention layers with a majority of linear/SSM layers.

*Jamba* (AI21 Labs, 2024) interleaves Mamba layers with Transformer attention layers and MoE feed-forward blocks, achieving competitive quality at much lower inference memory than a pure Transformer of similar capacity.

*Zamba* (Zyphra, 2024) takes a more extreme ratio: a single shared Transformer attention layer is prepended to each block, with Mamba layers providing the bulk of the computation. The shared attention layer is reused across blocks (via weight-sharing), dramatically reducing the parameter overhead of attention.

The hybrid approach reflects a pragmatic consensus: linear recurrences are excellent for *in-context accumulation* (building up a compressed representation over long inputs), while full attention is better for *sharp retrieval* (looking up a specific prior token). A model that needs both benefits from having both.

### 7.4 The Neural Memory Perspective

The *neural memory* or *fast weight programmer* perspective (Schmidhuber, 1992; Schlag et al., 2021) provides a unified conceptual frame for all the architectures in Sections 5–7. In this view, the state matrix $S_t \in \mathbb{R}^{d_k \times d_v}$ is a *fast-weight matrix* that stores an associative memory. Writing to this memory is the outer product update $S_t \leftarrow S_t + \mathbf{k}_t^\top \mathbf{v}_t$; reading from it is $\mathbf{o}_t = \mathbf{q}_t S_t$.

Gating — whether scalar ($\gamma$), per-channel ($\mathbf{g}_t$), or full-matrix ($G_t$) — corresponds to *forgetting*: the memory is weighted down before the new write, preventing unbounded accumulation and implementing a form of recency bias. The more expressive the gate, the more the model can selectively retain some memories while discarding others.

The *delta rule* (Widrow and Hoff, 1960) provides a motivated write policy beyond the pure outer-product: instead of blindly accumulating every key-value pair, the delta rule writes only the *prediction error* $(\mathbf{v}_t - \mathbf{q}_t S_{t-1})$, subtracting the current memory's prediction from the new value. This was formalized for Transformers by Schlag et al. (2021) and forms the basis for DeltaNet and Gated DeltaNet (2024). See [[linear-attention#6. Neural Memory Perspective|§6 of linear-attention.md]] for the full derivation.

---

## 8. Open Problems and Future Directions

The trajectory from Bahdanau's additive scores to SSD's semiseparable matrices spans a decade of sustained progress, but several fundamental questions remain open.

**Can linear attention match softmax quality?** On perplexity benchmarks at scale, the best gated linear attention models (GLA, RWKV-6) still lag behind comparably-sized Transformers by a non-trivial margin. It is unclear whether this is a fundamental limitation of the fixed-state-size bottleneck, a consequence of insufficient architectural tuning, or a training dynamics issue. The hybrid evidence (Section 7.3) suggests that at minimum, a small amount of full attention is necessary for tasks requiring precise retrieval.

**Expressive gating versus hardware efficiency.** The step from scalar decay (RetNet) to vector-valued gating (GLA) to matrix-valued gating (RWKV-6) improves expressiveness but increases implementation complexity and reduces GPU utilization. The outer-product gate in GLA and RWKV-6 can be computed efficiently via chunkwise tiling (Yang et al., 2024), but the custom kernels required are non-trivial. A general theory of which gating structures admit hardware-efficient chunkwise algorithms remains incomplete.

**Long-context and multi-modal attention.** As context lengths extend to millions of tokens (as in Gemini 1.5 and recent Claude releases), even FlashAttention's $O(T^2/M)$ IO cost becomes prohibitive. Retrieval-augmented architectures, hierarchical attention, and hybrid SSM-Transformer models are all active approaches. Multi-modal attention — where tokens come from images, audio, and video alongside text — introduces additional challenges: patch tokens from vision encoders have very different statistical properties than text tokens, and the optimal attention pattern may differ substantially across modalities.

**What does the state matrix memorize?** The state matrix $S_t$ at any position $t$ is an $d_k \times d_v$ matrix that must summarize all context seen so far. Characterizing what information is preserved and what is lost — as a function of the gating mechanism, the feature map $\phi$, and the sequence statistics — is an open theoretical problem. Some progress has been made by analyzing linear attention as *online linear regression* (Irie et al., 2021) and by connecting specific gating patterns to ridge regression estimators, but a general theory of linear attention's in-context learning capabilities is lacking.

---

## 9. References

| Reference Name | Brief Summary | Link to Reference |
|---|---|---|
| Bahdanau, Cho, Bengio (2015) | Introduced additive attention for neural machine translation; the original "soft alignment" paper | [https://arxiv.org/abs/1409.0473](https://arxiv.org/abs/1409.0473) |
| Luong, Pham, Manning (2015) | Multiplicative (dot and general) attention variants; global vs local attention for NMT | [https://arxiv.org/abs/1508.04025](https://arxiv.org/abs/1508.04025) |
| Vaswani et al. (2017) | "Attention Is All You Need": the Transformer, scaled dot-product attention, multi-head attention | [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762) |
| Radford et al. GPT-1 (2018) | Decoder-only Transformer pretrained with causal language modeling; established fine-tuning paradigm | [https://openai.com/research/language-unsupervised](https://openai.com/research/language-unsupervised) |
| Devlin et al. BERT (2018) | Encoder-only bidirectional Transformer with masked LM pretraining; dominant NLU backbone 2018–2020 | [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805) |
| Radford et al. GPT-2 (2019) | 1.5B parameter causal Transformer; demonstrated zero-shot generalization at scale | [https://openai.com/research/better-language-models](https://openai.com/research/better-language-models) |
| Brown et al. GPT-3 (2020) | 175B parameter causal Transformer; few-shot in-context learning as emergent capability | [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165) |
| Raffel et al. T5 (2020) | Text-to-text encoder-decoder Transformer with relative position biases | [https://arxiv.org/abs/1910.10683](https://arxiv.org/abs/1910.10683) |
| Beltagy et al. Longformer (2020) | Sparse attention: sliding window + global tokens for O(T) complexity | [https://arxiv.org/abs/2004.05150](https://arxiv.org/abs/2004.05150) |
| Zaheer et al. BigBird (2020) | Sparse attention: local + global + random; universal approximation proof | [https://arxiv.org/abs/2007.14062](https://arxiv.org/abs/2007.14062) |
| Katharopoulos et al. (2020) | "Transformers are RNNs": linear attention via kernel feature maps, state matrix recurrence | [https://arxiv.org/abs/2006.16236](https://arxiv.org/abs/2006.16236) |
| Shazeer (2019) | Multi-query attention: single KV head shared across query heads for KV cache reduction | [https://arxiv.org/abs/1911.02150](https://arxiv.org/abs/1911.02150) |
| Dao et al. FlashAttention (2022) | IO-aware exact attention via tiling; O(N) memory, 2–4× wall-clock speedup | [https://arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135) |
| Ainslie et al. GQA (2023) | Grouped-query attention: interpolates between MHA and MQA; adopted in LLaMA-2, Mistral | [https://arxiv.org/abs/2305.13245](https://arxiv.org/abs/2305.13245) |
| Gu, Goel, Ré S4 (2022) | Structured state spaces with HiPPO initialization; O(N+L) SSM training via convolution | [https://arxiv.org/abs/2111.00396](https://arxiv.org/abs/2111.00396) |
| Peng et al. RWKV (2023) | RNN with attention-like training parallelism; WKV operator with channel-wise exponential decay | [https://arxiv.org/abs/2305.13048](https://arxiv.org/abs/2305.13048) |
| Sun et al. RetNet (2023) | Retention mechanism with constant scalar decay; parallel / recurrent / chunkwise training modes | [https://arxiv.org/abs/2307.08621](https://arxiv.org/abs/2307.08621) |
| Gu, Dao Mamba (2023) | Selective SSMs: input-dependent B, C, Delta; hardware-aware parallel scan | [https://arxiv.org/abs/2312.00752](https://arxiv.org/abs/2312.00752) |
| Dao, Gu Mamba-2 / SSD (2024) | State Space Duality: selective SSMs equivalent to masked attention via 1-semiseparable matrices | [https://arxiv.org/abs/2405.21060](https://arxiv.org/abs/2405.21060) |
| Yang et al. GLA (2024) | Gated Linear Attention: vector-valued data-dependent gate via low-rank outer product; hardware-efficient chunkwise training | [https://arxiv.org/abs/2312.06635](https://arxiv.org/abs/2312.06635) |
| Schlag, Irie, Schmidhuber (2021) | Fast weight programmers as attention; delta rule as error-correcting memory write | [https://arxiv.org/abs/2102.11174](https://arxiv.org/abs/2102.11174) |
