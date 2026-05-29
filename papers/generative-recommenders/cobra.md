# Sparse Meets Dense: Unified Generative Recommendations with Cascaded Sparse-Dense Representations

*Yuhao Yang, Zhi Ji, Zhaopeng Li, Yi Li, Zhonglin Mo, Yue Ding, Kai Chen, Zijian Zhang, Jie Li, Shuanglong Li, Lin Liu — Baidu Inc. ArXiv 2503.02453, 2025.*

| Dimension | Prior State | This Paper | Key Result |
|---|---|---|---|
| Information completeness | RQ-VAE semantic IDs lose fine-grained detail via quantization; final residual discarded | Cascaded sparse→dense generation jointly captures both categorical structure and intra-cluster detail | Removing dense branch: −27.1% R@500 on industrial dataset |
| Retrieval diversity | Beam search over code tree confined to a single representation type | BeamFusion combines beam probability and ANN cosine similarity for diversity–precision control | Removing BeamFusion: −22.1% R@500 |
| Dense representation training | Prior work (LIGER) used fixed pre-trained dense embeddings not adapted to recommendation signal | Dense encoder trained end-to-end with contrastive loss over interaction data | Removing sparse ID branch: −33.6% R@500 |
| Industrial deployment | TIGER-style discrete generation | Coarse-to-fine sparse→dense pipeline | +3.60% conversion, +4.15% ARPU at Baidu (200M DAU, Jan 2025) |

## Relations

**Builds on:** [[papers/generative-recommenders/hstu|HSTU (Zhai et al., 2024)]], TIGER (Rajput et al., NeurIPS 2023) *(no note yet)*, LIGER (Yang et al., 2024) *(no note yet)*
**Concepts used:** [[concepts/generative-recommenders/note|Generative Recommender Systems]] — §4 (Generative Retrieval, RQ-VAE, COBRA)

## Table of Contents

- [[#1. Motivation: The Information Loss Problem|1. Motivation: The Information Loss Problem]]
- [[#2. Sparse and Dense Representations|2. Sparse and Dense Representations]]
  - [[#2.1 Sparse IDs via RQ-VAE|2.1 Sparse IDs via RQ-VAE]]
  - [[#2.2 End-to-End Learnable Dense Vectors|2.2 End-to-End Learnable Dense Vectors]]
- [[#3. Cascaded Sequential Modeling|3. Cascaded Sequential Modeling]]
  - [[#3.1 Probabilistic Factorization|3.1 Probabilistic Factorization]]
  - [[#3.2 Input Construction and Forward Pass|3.2 Input Construction and Forward Pass]]
- [[#4. Training Objectives|4. Training Objectives]]
  - [[#4.1 Sparse Cross-Entropy Loss|4.1 Sparse Cross-Entropy Loss]]
  - [[#4.2 Dense Contrastive Loss|4.2 Dense Contrastive Loss]]
- [[#5. Coarse-to-Fine Inference: BeamFusion|5. Coarse-to-Fine Inference: BeamFusion]]
  - [[#5.1 Sparse ID Beam Search|5.1 Sparse ID Beam Search]]
  - [[#5.2 Conditional Dense Generation|5.2 Conditional Dense Generation]]
  - [[#5.3 BeamFusion Ranking Score|5.3 BeamFusion Ranking Score]]
- [[#6. Experiments|6. Experiments]]
  - [[#6.1 Public Benchmarks|6.1 Public Benchmarks]]
  - [[#6.2 Industrial Ablation|6.2 Industrial Ablation]]
  - [[#6.3 Recall–Diversity Trade-off|6.3 Recall–Diversity Trade-off]]
- [[#7. References|7. References]]

---

## 1. Motivation: The Information Loss Problem 📐

*Generative retrieval* via [[concepts/generative-recommenders/note#4.1 Semantic IDs via RQ-VAE: TIGER|RQ-VAE semantic IDs]] encodes each item $x$ as a tuple $(c_1, c_2, \ldots, c_K)$ of discrete codeword indices. The encoder compresses the item's dense embedding $\mathbf{z} \in \mathbb{R}^d$ into $K$ successive quantization residuals; the decoder reconstructs $\hat{\mathbf{z}} = \sum_{k=1}^K \mathbf{e}_{c_k}$. By construction, the final residual $\mathbf{r}_K = \mathbf{z} - \hat{\mathbf{z}}$ is discarded.

**The information loss is irreducible.** For finite $K$ and codebook sizes, $\|\mathbf{r}_K\| > 0$ almost surely. Two items can share an identical semantic ID yet differ in their true embeddings — a *collision* that makes them indistinguishable to the retrieval model. Increasing $K$ or codebook size reduces the residual but never eliminates it, and increases inference cost.

LIGER (Yang et al., 2024) addressed this by appending a separately pre-trained dense embedding alongside the sparse ID at each sequence position. But using a frozen, separately-trained dense encoder introduces a *representation mismatch*: the dense vector is not adapted to the generative pretraining objective and carries no collaborative signal from interaction sequences.

COBRA's solution is a *cascade*: generate the sparse ID first, then condition the generation of the dense vector on the produced sparse ID. This ordering allows the dense vector to refine the coarse categorical prediction — recovering intra-cluster detail that the sparse ID discards — while avoiding the mismatch of pre-trained fixed embeddings.

> [!QUESTION] Exercise 1: Why cascade sparse → dense, not dense → sparse?
> *This question establishes why the ordering of the cascade matters for information-theoretic reasons.*
>
> > **Prerequisites:** [[#1. Motivation: The Information Loss Problem|§1]]
>
> The COBRA factorization generates the sparse ID first and conditions the dense vector on it. Suppose instead you generate the dense vector $\mathbf{v}_{t+1}$ first, then the sparse ID $\text{ID}_{t+1}$ conditioned on $\mathbf{v}_{t+1}$. Write out the corresponding factorization of $P(\text{ID}_{t+1}, \mathbf{v}_{t+1} \mid S_{1:t})$ and argue whether this ordering is preferable, equivalent, or worse than COBRA's.

> [!TIP]- Solution to Exercise 1
> **Key insight:** The conditional $P(\text{ID}_{t+1} \mid \mathbf{v}_{t+1}, S_{1:t})$ is computing a quantization (nearest-neighbor lookup in code space) given an already-predicted dense vector. But the dense vector prediction $P(\mathbf{v}_{t+1} \mid S_{1:t})$ is an unconstrained regression over $\mathbb{R}^d$ — it has no guarantee of landing near any real item in the catalog. The downstream ID assignment then nearest-neighbor projects an arbitrary point, which may not correspond to the right cluster.
>
> **Sketch:** In the sparse→dense order, beam search over the sparse ID acts as a *constraint* that confines the dense prediction to items within the beam's cluster. The dense head then answers a much easier question: "given that the next item is in cluster $c_1, c_2, \ldots$, where exactly within that cluster is it?" In dense→sparse order, the first step solves a harder, unconstrained problem; the second step is a deterministic mapping from dense to discrete that adds no capacity. The cascade loses its refinement structure. Empirically, COBRA's ablation confirms that the sparse branch contributes +33.6% recall — it provides the structural prior that makes the dense prediction tractable.

---

## 2. Sparse and Dense Representations

### 2.1 Sparse IDs via RQ-VAE

The sparse representation follows the standard RQ-VAE construction (see [[concepts/generative-recommenders/note#4.1 Semantic IDs via RQ-VAE: TIGER|§4.1 of the concept note]] for full derivation). Item $x$ is encoded from its textual attribute description via a Transformer text encoder, then quantized hierarchically into a $K$-tuple $(c_1, \ldots, c_K)$.

COBRA uses a 3-level RQ-VAE (codebook size 32 per level) for public datasets and a 2-level $32 \times 32$ scheme for the industrial dataset. The codebook is trained with EMA updates and random last-level assignment to achieve $\geq$99% code uniqueness (see [[concepts/generative-recommenders/note#4.1 Semantic IDs via RQ-VAE: TIGER|codebook collapse mitigation]]).

### 2.2 End-to-End Learnable Dense Vectors

**Definition (COBRA Dense Encoder).** A Transformer encoder processes the flattened item attribute tokens prefixed with a [CLS] token. The dense vector for item $x$ is the [CLS] hidden state:

$$\mathbf{v}_x = \text{TransformerEncoder}([\texttt{[CLS]}, w_1, w_2, \ldots, w_L])_{[0]}$$

where $w_1, \ldots, w_L$ are the item's attribute tokens and the subscript $[0]$ selects the [CLS] position.

*Critically, this encoder is trained end-to-end with the contrastive loss of §4.2 — not pre-trained and frozen.* The dense vector therefore adapts to the recommendation objective: items that co-occur in user interaction sequences are pulled together in dense space, regardless of their attribute similarity.

---

## 3. Cascaded Sequential Modeling 🔑

### 3.1 Probabilistic Factorization

**Definition (COBRA Factorization).** Let $S_{1:t} = ((x_1, \mathbf{v}_1), \ldots, (x_t, \mathbf{v}_t))$ be the user's interaction history where $x_i$ has sparse ID $\text{ID}_i$ and dense vector $\mathbf{v}_i$. COBRA factorizes the joint predictive distribution as:

$$P(\text{ID}_{t+1}, \mathbf{v}_{t+1} \mid S_{1:t}) = \underbrace{P(\text{ID}_{t+1} \mid S_{1:t})}_{\text{sparse prediction}} \cdot \underbrace{P(\mathbf{v}_{t+1} \mid \text{ID}_{t+1}, S_{1:t})}_{\text{dense refinement}}$$

The factorization is the product rule applied with $\text{ID}_{t+1}$ as the first variable. This is not an independence assumption — the dense prediction is explicitly conditioned on the sparse ID.

**Intuition.** The sparse ID answers "which cluster does the next item belong to?" The dense vector, conditioned on the sparse ID, answers "where within that cluster is the next item?" The cascade decomposes a single hard prediction into two simpler sequential steps.

### 3.2 Input Construction and Forward Pass

Each sequence position $t$ is represented by concatenating the sparse ID embedding with the dense vector:

$$\mathbf{h}_t = [\mathbf{e}(\text{ID}_t);\ \mathbf{v}_t] \in \mathbb{R}^{2d}$$

where $\mathbf{e}(\text{ID}_t) = \sum_{k=1}^K \mathbf{e}_{c_k^{(t)}}$ is the sum of codeword embeddings. The full sequence history is thus:

$$S_{1:t} = [\mathbf{h}_1, \mathbf{h}_2, \ldots, \mathbf{h}_t] \in \mathbb{R}^{t \times 2d}$$

**Sparse ID forward pass.** A causal Transformer decoder processes $S_{1:t}$ and a *SparseHead* projects the output at position $t$ to logits over the codebook:

$$\mathbf{z}_{t+1} = \text{SparseHead}(\text{TransformerDecoder}(S_{1:t}))$$

> [!NOTE] What TransformerDecoder outputs
> $\text{TransformerDecoder}(S_{1:t})$ is a stack of causal self-attention layers (GPT-style, decoder-only) that maps the input sequence $S_{1:t} \in \mathbb{R}^{t \times 2d}$ to a sequence of hidden states $\mathbf{O}_{1:t} \in \mathbb{R}^{t \times d_{\text{model}}}$. Position $i$ attends only to positions $\leq i$ (causal mask). The *last* hidden state $\mathbf{o}_t \in \mathbb{R}^{d_{\text{model}}}$ summarizes the full causal context of the sequence — it is the only output that SparseHead reads. The intermediate hidden states $\mathbf{o}_1, \ldots, \mathbf{o}_{t-1}$ are by-products of the computation and are not used directly by either head.

> [!NOTE] What SparseHead is
> SparseHead is a linear projection $W_s \in \mathbb{R}^{V \times d_{\text{model}}}$ applied to the last decoder hidden state $\mathbf{o}_t$:
> $$\mathbf{z}_{t+1} = W_s\, \mathbf{o}_t \in \mathbb{R}^V$$
> The output $\mathbf{z}_{t+1}$ is a vector of $V$ unnormalized logits — one per codebook entry at the current quantization level. Since the RQ-VAE semantic ID is a $K$-tuple $(c_1, \ldots, c_K)$, the sparse ID is generated *autoregressively over levels*: SparseHead is applied at each decode step $k \in [K]$, with the code token $c_k$ selected (by argmax during training, or by beam search at inference) and appended to the sequence before the next step. At the $k$-th decode step the decoder attends to both the history $S_{1:t}$ and the previously generated code tokens $c_1, \ldots, c_{k-1}$.

> [!NOTE] What $\widehat{\text{ID}}_{t+1}$ is
> $\widehat{\text{ID}}_{t+1} = (\hat{c}_1, \hat{c}_2, \ldots, \hat{c}_K)$ is the *predicted* sparse ID for the next item — the model's best guess at which code cluster the next item belongs to. It is obtained by running $K$ autoregressive decode steps through SparseHead:
> - During **training**: greedy argmax at each level (or teacher forcing in some implementations), giving a single predicted $K$-tuple.
> - During **inference**: beam search over the code tree, giving $M$ candidate $K$-tuples each with an associated log-probability $\phi^k$ (see §5.1).
>
> Crucially, $\widehat{\text{ID}}_{t+1}$ is a *discrete index* — not a continuous vector. The embedding $\mathbf{e}(\widehat{\text{ID}}_{t+1}) = \sum_{k=1}^K \mathbf{e}_{\hat{c}_k}$ converts it back into a continuous vector so it can be appended to the sequence for the dense pass.

**Dense vector forward pass.** The predicted sparse ID embedding $\mathbf{e}(\widehat{\text{ID}}_{t+1})$ is appended to the history, and a *DenseHead* projects the decoder output at the new final position:

$$\bar{S}_{1:t} = [S_{1:t};\ \mathbf{e}(\widehat{\text{ID}}_{t+1})], \qquad \hat{\mathbf{v}}_{t+1} = \text{DenseHead}(\text{TransformerDecoder}(\bar{S}_{1:t}))$$

> [!NOTE] What DenseHead is
> DenseHead is a separate linear projection $W_d \in \mathbb{R}^{d \times d_{\text{model}}}$ applied to the decoder's last hidden state *after* the extended sequence $\bar{S}_{1:t}$:
> $$\hat{\mathbf{v}}_{t+1} = W_d\, \bar{\mathbf{o}}_{t+1} \in \mathbb{R}^d$$
> where $\bar{\mathbf{o}}_{t+1}$ is the final hidden state of $\text{TransformerDecoder}(\bar{S}_{1:t})$ — i.e., the hidden state at the *newly appended position* $\mathbf{e}(\widehat{\text{ID}}_{t+1})$. This position attends causally to all of $S_{1:t}$ plus the sparse ID embedding, so its hidden state encodes both the full interaction history *and* the coarse cluster constraint imposed by $\widehat{\text{ID}}_{t+1}$. The output $\hat{\mathbf{v}}_{t+1} \in \mathbb{R}^d$ lives in the same space as the pre-computed item dense vectors, enabling ANN retrieval.
>
> Why is a second decoder pass needed? Because $\bar{\mathbf{o}}_{t+1}$ (the hidden state at the appended position) does not exist in the first forward pass — appending $\mathbf{e}(\widehat{\text{ID}}_{t+1})$ to the sequence creates a new final position whose hidden state is not computed until the decoder processes $\bar{S}_{1:t}$. The KV cache of the first pass can be reused for the prefix $S_{1:t}$; only one additional attention step is needed for the new position.

Note that the dense head sees the predicted sparse ID — not the ground-truth — at training time. The model is therefore trained to produce dense vectors *conditioned on the predicted cluster*, which matches inference behavior.

---

## 4. Training Objectives

### 4.1 Sparse Cross-Entropy Loss

The sparse ID loss is standard cross-entropy over codebook tokens at each level $k \in [K]$. For a sequence of length $T$:

$$\mathcal{L}_{\text{sparse}} = -\sum_{t=1}^{T-1} \log \frac{\exp\!\left(z_{t+1}^{\text{ID}_{t+1}}\right)}{\sum_j \exp\!\left(z_{t+1}^j\right)}$$

where $z_{t+1}^j$ is the $j$-th logit of $\mathbf{z}_{t+1}$.

### 4.2 Dense Contrastive Loss

The dense loss aligns the predicted vector $\hat{\mathbf{v}}_{t+1}$ with the ground-truth item dense vector $\mathbf{v}_{t+1}$ via an InfoNCE objective with in-batch negatives:

$$\mathcal{L}_{\text{dense}} = -\sum_{t=1}^{T-1} \log \frac{\exp\!\left(\cos(\hat{\mathbf{v}}_{t+1}, \mathbf{v}_{t+1}) / \tau\right)}{\sum_{j \in \mathcal{B}} \exp\!\left(\cos(\hat{\mathbf{v}}_{t+1}, \mathbf{v}_j) / \tau\right)}$$

where $\mathcal{B}$ is the in-batch item set and $\tau$ is a temperature hyperparameter. The cosine similarity metric is invariant to embedding scale, which stabilizes training when the encoder and decoder operate at different magnitudes.

**Combined objective:**

$$\mathcal{L} = \mathcal{L}_{\text{sparse}} + \mathcal{L}_{\text{dense}}$$

The two terms are summed without weighting hyperparameters — the ablation shows both are necessary and they operate on disjoint output heads.

---

## 5. Coarse-to-Fine Inference: BeamFusion 💡

### 5.1 Sparse ID Beam Search

At inference, beam search with width $M$ generates the top-$M$ sparse ID candidates from the model's distribution:

$$\{\widehat{\text{ID}}_{T+1}^k\}_{k=1}^M = \text{BeamSearch}\!\left(\text{TransformerDecoder}(S_{1:T}),\ M\right)$$

Each beam $k$ carries an associated log-probability $\phi^k = \log P(\widehat{\text{ID}}_{T+1}^k \mid S_{1:T})$.

### 5.2 Conditional Dense Generation

For each beam $k$, the sparse ID embedding is appended to the history and the DenseHead generates a conditional dense vector:

$$\hat{\mathbf{v}}_{T+1}^k = \text{DenseHead}\!\left(\text{TransformerDecoder}\!\left([S_{1:T};\ \mathbf{e}(\widehat{\text{ID}}_{T+1}^k)]\right)\right)$$

An ANN index over pre-computed item dense vectors is then queried for each beam:

$$\mathcal{A}_k = \text{ANN}\!\left(\hat{\mathbf{v}}_{T+1}^k,\ \mathcal{C}(\widehat{\text{ID}}_{T+1}^k),\ N\right)$$

where $\mathcal{C}(\widehat{\text{ID}}_{T+1}^k)$ restricts the search to items in the beam's code cluster, and $N$ is the number of retrieved neighbors per beam.

### 5.3 BeamFusion Ranking Score

Each candidate item $a \in \bigcup_{k=1}^M \mathcal{A}_k$ is scored by combining the sparse beam probability and the dense cosine similarity:

$$\Phi(\hat{\mathbf{v}}_{T+1}^k, \widehat{\text{ID}}_{T+1}^k, a) = \underbrace{\text{Softmax}(\tau \cdot \phi^k)}_{\text{sparse beam weight}} \cdot \underbrace{\text{Softmax}(\psi \cdot \cos(\hat{\mathbf{v}}_{T+1}^k, \mathbf{a}))}_{\text{dense similarity weight}}$$

The hyperparameters $\tau$ and $\psi$ control the precision–diversity trade-off: higher $\tau$ concentrates weight on high-probability beams (lower diversity); higher $\psi$ concentrates weight on the dense nearest neighbors (lower recall from lower-probability beams). Empirically, the optimal balance is $\tau = 0.9$, $\psi = 16$.

**Final retrieval:**

$$\mathcal{R} = \text{Top-}K\!\left(\bigcup_{k=1}^M \mathcal{A}_k,\ \Phi\right)$$

> [!QUESTION] Exercise 2: BeamFusion as a product of experts
> *This question connects BeamFusion to probabilistic model combination.*
>
> > **Prerequisites:** [[#5.3 BeamFusion Ranking Score|§5.3]]
>
> The BeamFusion score $\Phi$ can be interpreted as a product of two softmax distributions over items. Show that $\log \Phi \propto \tau \phi^k + \psi \cos(\hat{\mathbf{v}}_{T+1}^k, \mathbf{a}) + \text{const}$, and identify the conditions under which this is equivalent to the log-probability of a *product of experts* model combining the sparse and dense predictions.

> [!TIP]- Solution to Exercise 2
> **Key insight:** $\text{Softmax}(\tau \cdot \phi^k)_i = \exp(\tau \phi^k_i) / Z_\tau$ and $\text{Softmax}(\psi \cdot \cos(\cdot))_j = \exp(\psi \cos_j) / Z_\psi$. The product is $\exp(\tau \phi^k_i + \psi \cos_j) / (Z_\tau Z_\psi)$, which is proportional to $\exp(\tau \phi + \psi \cos)$ — exactly a log-linear combination (product of experts) where the two experts are the sparse beam probability and the dense cosine similarity.
>
> **Sketch:** A product of experts model defines $P_{\text{PoE}}(a) \propto \prod_m P_m(a)^{\alpha_m}$, or equivalently $\log P_{\text{PoE}}(a) = \sum_m \alpha_m \log P_m(a) + \text{const}$. BeamFusion matches this with $\alpha_{\text{sparse}} = \tau$ and $\alpha_{\text{dense}} = \psi$. It is a PoE under the assumption that the softmax-normalized scores are valid probabilities — they are not, strictly, since $\tau$ and $\psi$ rescale the log-probabilities before normalization. But the form has the same flavor: it rewards items that both beams agree on, with calibration controlled by the temperature parameters.

---

## 6. Experiments 📊

### 6.1 Public Benchmarks

COBRA is evaluated on three Amazon product review datasets (Beauty, Sports and Outdoors, Toys and Games) against TIGER and dense sequential baselines (SASRec, BERT4Rec, RecFormer, LIGER).

| Dataset | Metric | TIGER | LIGER | COBRA | Δ vs. TIGER |
|---|---|---|---|---|---|
| Beauty | Recall@5 | 0.0454 | 0.0471 | **0.0537** | +18.3% |
| Beauty | NDCG@5 | 0.0321 | 0.0341 | **0.0395** | +23.1% |
| Sports | Recall@5 | 0.0264 | 0.0281 | **0.0305** | +15.5% |
| Sports | NDCG@10 | 0.0225 | 0.0236 | **0.0257** | +14.2% |
| Toys | Recall@10 | 0.0712 | 0.0733 | **0.0781** | +9.7% |
| Toys | NDCG@10 | 0.0432 | 0.0449 | **0.0515** | +19.2% |

COBRA improves over TIGER by 9–23% across all metrics, with larger gains on NDCG (which penalizes lower-ranked correct items more heavily) than Recall — consistent with BeamFusion's improvement in ranking quality, not just top-$K$ coverage.

### 6.2 Industrial Ablation

Evaluated on Baidu's advertising dataset (200M DAU). The ablation isolates the contribution of each component:

| Variant | R@50 | R@100 | R@500 | R@800 |
|---|---|---|---|---|
| Full COBRA | **0.1180** | **0.1737** | **0.3716** | **0.4466** |
| w/o sparse ID branch | 0.0611 | 0.0964 | 0.2466 | 0.3111 |
| w/o dense branch | 0.0690 | 0.1032 | 0.2709 | 0.3273 |
| w/o BeamFusion | 0.0856 | 0.1254 | 0.2455 | 0.2855 |

**Both branches are necessary.** Removing the sparse branch hurts more at low recall cutoffs (R@50: −48.2%) because beam search over semantic IDs is the primary structuring mechanism. Removing the dense branch hurts more at high recall cutoffs (R@800: −26.7%) because the dense vector is what distinguishes items within a cluster once the cluster is identified.

Online A/B test (January 2025, 10% traffic): **+3.60% conversion rate, +4.15% ARPU**.

### 6.3 Recall–Diversity Trade-off

The BeamFusion parameters $\tau$ and $\psi$ trade recall (broad candidate coverage) against diversity (distributing candidates across semantic clusters). Optimal performance at $\tau = 0.9$, $\psi = 16$; increasing $\tau$ beyond 1.0 collapses the retrieval toward the single highest-probability beam, reducing diversity with diminishing recall gains.

---

## 7. References

| Reference Name | Brief Summary | Link |
|---|---|---|
| [Sparse Meets Dense: COBRA (Yang et al., 2025)](https://arxiv.org/abs/2503.02453) | This paper | https://arxiv.org/abs/2503.02453 |
| [TIGER: Recommender Systems with Generative Retrieval (Rajput et al., NeurIPS 2023)](https://arxiv.org/abs/2305.05065) | RQ-VAE semantic IDs + seq2seq beam search; baseline COBRA improves over | https://arxiv.org/abs/2305.05065 |
| [HSTU: Actions Speak Louder than Words (Zhai et al., 2024)](https://arxiv.org/abs/2402.17152) | Foundational generative ranking architecture; motivates the industrial context | https://arxiv.org/abs/2402.17152 |
| [LIGER (Yang et al., 2024)](https://arxiv.org/abs/2406.06485) | Simultaneous sparse + fixed dense generation; COBRA's direct predecessor and key baseline | https://arxiv.org/abs/2406.06485 |
| [Generating Diverse High-Fidelity Images with VQ-VAE-2 (Razavi et al., 2019)](https://arxiv.org/abs/1906.00446) | Foundational RQ-VAE architecture | https://arxiv.org/abs/1906.00446 |
| [Representation Learning with Contrastive Predictive Coding (Oord et al., 2018)](https://arxiv.org/abs/1807.03748) | InfoNCE objective used in the dense contrastive loss | https://arxiv.org/abs/1807.03748 |
