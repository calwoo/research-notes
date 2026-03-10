# Structured Attention Networks

## Table of Contents

- [[#1. Motivation and Setup|1. Motivation and Setup]]
- [[#2. Standard Attention as Marginal Inference|2. Standard Attention as Marginal Inference]]
- [[#3. Linear-Chain CRF Attention|3. Linear-Chain CRF Attention]]
  - [[#3.1 Model Definition|3.1 Model Definition]]
  - [[#3.2 Forward-Backward Algorithm|3.2 Forward-Backward Algorithm]]
  - [[#3.3 Complexity|3.3 Complexity]]
- [[#4. Graph-Based Parser Attention|4. Graph-Based Parser Attention]]
  - [[#4.1 Projective Dependency Trees|4.1 Projective Dependency Trees]]
  - [[#4.2 Non-Projective Trees and the Matrix-Tree Theorem|4.2 Non-Projective Trees and the Matrix-Tree Theorem]]
- [[#5. Differentiable Dynamic Programming|5. Differentiable Dynamic Programming]]
  - [[#5.1 Log-Space Computation|5.1 Log-Space Computation]]
  - [[#5.2 Signed Log-Space Backward Pass|5.2 Signed Log-Space Backward Pass]]
  - [[#5.3 Gradient Structure|5.3 Gradient Structure]]
- [[#6. Relation to Standard Attention|6. Relation to Standard Attention]]
- [[#7. Experiments and Results|7. Experiments and Results]]
- [[#References|References]]

---

## 1. Motivation and Setup

Standard softmax attention produces a *categorical* distribution over $n$ input positions. Given a query $\mathbf{q} \in \mathbb{R}^d$ and a sequence of input representations $\mathbf{x} = (\mathbf{x}_1, \ldots, \mathbf{x}_n)$ with $\mathbf{x}_i \in \mathbb{R}^d$, the output context vector selects a single soft-weighted combination of inputs. This restricts the inductive bias to that of a *mixture model*: the network may implicitly attend to multiple positions, but the latent structure is entirely unordered and independent across positions.

Many natural language phenomena are instead governed by *combinatorial* structures: contiguous spans, constituency parses, dependency trees. Structured Attention Networks (Kim, Denton, Hoang, and Rush, ICLR 2017) replace the categorical latent variable in standard attention with a *structured* latent variable whose distribution is defined by a *graphical model*, and compute the context vector as an expectation under that distribution via exact marginal inference.

The key insight is that marginal inference in many tractable graphical models — linear-chain *conditional random fields*, projective dependency parsers — can be expressed as differentiable dynamic programs, making the entire attention mechanism end-to-end differentiable without requiring discrete sampling.

---

## 2. Standard Attention as Marginal Inference

**Definition (Attention as Posterior Expectation).** Let $z \in \{1, \ldots, n\}$ be a categorical latent variable selecting one input position. Define a distribution over $z$ given inputs and query:

$$p(z = i \mid \mathbf{x}, \mathbf{q}) = \frac{\exp(\theta_i)}{\sum_{j=1}^{n} \exp(\theta_j)}, \qquad \theta_i = \frac{\mathbf{q}^\top \mathbf{x}_i}{\sqrt{d}}$$

The context vector is the posterior mean of the selected representation:

$$\mathbf{c} = \mathbb{E}_{z \sim p(z \mid \mathbf{x}, \mathbf{q})}[f(\mathbf{x}, z)] = \sum_{i=1}^{n} p(z = i \mid \mathbf{x}, \mathbf{q}) \cdot \mathbf{x}_i$$

where $f(\mathbf{x}, z) = \mathbf{x}_z$ retrieves the $z$-th representation.

This framing exposes the generalization seam: the choice of graphical model governing $z$ is not fixed. If we replace the categorical $z$ with a *structured* latent variable and replace the softmax normalization with the corresponding *partition function*, we obtain a family of structured attention mechanisms parameterized by the graphical model.

**Remark.** The potentials $\theta_i$ play the role of *log-unnormalized probabilities* in the graphical model. Inference in the categorical model reduces to computing a single normalizing constant — the softmax denominator — which is trivially $O(n)$. Structured models require more sophisticated inference algorithms but retain the same interpretive skeleton.

---

## 3. Linear-Chain CRF Attention

### 3.1 Model Definition

**Definition (Linear-Chain CRF Attention).** Let $z = (z_1, \ldots, z_n)$ with each $z_i \in \{0, 1\}$ be a binary sequence indicating which positions are selected. The joint distribution over $z$ is a *linear-chain conditional random field*:

$$p(z \mid \mathbf{x}, \mathbf{q}) \propto \exp\!\left(\sum_{i=1}^{n} \theta_i z_i + \sum_{i=1}^{n-1} \psi_{i,i+1}(z_i, z_{i+1})\right)$$

where:
- $\theta_i = \mathbf{q}^\top \mathbf{x}_i / \sqrt{d}$ are *unary potentials* (attention scores for position $i$),
- $\psi_{i,i+1} : \{0,1\}^2 \to \mathbb{R}$ are *pairwise potentials* encoding dependencies between adjacent binary indicators. In the simplest parameterization, $\psi_{i,i+1}$ is a learned $2 \times 2$ parameter matrix shared across positions.

The context vector is the marginal expectation:

$$\mathbf{c} = \sum_{i=1}^{n} \mu_i \cdot \mathbf{x}_i, \qquad \mu_i = p(z_i = 1 \mid \mathbf{x}, \mathbf{q})$$

where $\mu_i$ is the *unary marginal* for position $i$, obtained via the forward-backward algorithm.

### 3.2 Forward-Backward Algorithm

Define *forward messages* $\alpha_i(z_i)$ and *backward messages* $\beta_i(z_i)$ for each $z_i \in \{0, 1\}$.

**Initialization:**

$$\alpha_1(z_1) = \exp(\theta_1 z_1)$$

$$\beta_n(z_n) = 1$$

**Recursions:**

$$\alpha_i(z_i) = \exp(\theta_i z_i) \sum_{z_{i-1} \in \{0,1\}} \exp\!\left(\psi_{i-1,i}(z_{i-1}, z_i)\right) \alpha_{i-1}(z_{i-1})$$

$$\beta_i(z_i) = \sum_{z_{i+1} \in \{0,1\}} \exp\!\left(\psi_{i,i+1}(z_i, z_{i+1})\right) \exp(\theta_{i+1} z_{i+1}) \,\beta_{i+1}(z_{i+1})$$

**Marginal computation.** The unnormalized joint at position $i$ with $z_i = v$ is $\alpha_i(v) \beta_i(v)$. The unary marginal is therefore:

$$\mu_i = p(z_i = 1 \mid \mathbf{x}, \mathbf{q}) = \frac{\alpha_i(1)\,\beta_i(1)}{\alpha_i(0)\,\beta_i(0) + \alpha_i(1)\,\beta_i(1)}$$

**Remark.** The forward-backward algorithm is exact for chain-structured graphical models by the elimination principle: the chain topology ensures each variable is eliminated exactly once, and no fill-in is created.

### 3.3 Complexity

Each forward or backward step involves a sum over $|\{0,1\}|^2 = 4$ terms, so each step is $O(1)$ in the binary alphabet. Performing $n$ forward steps and $n$ backward steps gives:

$$\text{Total complexity} = O(n \cdot |\mathcal{Z}|^2) = O(4n) = O(n)$$

**The linear-chain CRF attention mechanism has the same asymptotic complexity as softmax attention but encodes sequential contiguity structure through the pairwise potentials.**

---

## 4. Graph-Based Parser Attention

### 4.1 Projective Dependency Trees

**Definition (Dependency Tree Attention).** Let $z_{ij} \in \{0, 1\}$ indicate whether token $j$ is the *head* (parent) of token $i$ in a dependency tree over $n$ tokens. A valid *projective dependency tree* satisfies:

1. Each non-root node $i$ has exactly one parent: $\sum_{j \neq i} z_{ij} = 1$.
2. The directed graph is acyclic.
3. The tree is *projective*: for all arcs $(i, j)$, every token $k$ between $i$ and $j$ is a descendant of $j$ (no crossing arcs).

The distribution over valid projective trees is:

$$p(z \mid \mathbf{x}, \mathbf{q}) \propto \exp\!\left(\sum_{i \neq j} z_{ij}\, \theta_{ij}\right) \cdot \mathbf{1}[z \text{ is a valid projective tree}]$$

where $\theta_{ij} = \mathbf{q}^\top \mathbf{x}_j / \sqrt{d}$ is the *arc score* for the edge from $j$ to $i$. The context vector aggregates over all arc marginals:

$$\mathbf{c} = \sum_{i \neq j} \mu_{ij}\, \mathbf{x}_j, \qquad \mu_{ij} = p(z_{ij} = 1 \mid \mathbf{x}, \mathbf{q})$$

*Edge marginals* $\mu_{ij}$ are computed via the *inside-outside algorithm* for projective dependency parsing, known as the Eisner algorithm. The Eisner algorithm builds complete and incomplete spans bottom-up in $O(n^3)$ time and $O(n^2)$ space.

**Proposition (Eisner Complexity).** The inside-outside algorithm for projective dependency tree marginals runs in $O(n^3)$ time, matching the complexity of CKY parsing for constituency grammars.

*This cubic cost is the primary practical overhead of parser attention relative to softmax or CRF attention.*

### 4.2 Non-Projective Trees and the Matrix-Tree Theorem

For *non-projective* dependency trees (where crossing arcs are permitted), the set of valid structures is the set of all *spanning trees* of the complete directed graph on $n$ nodes. The partition function over spanning trees and the edge marginals admit a closed-form via the *Matrix-Tree Theorem* (Kirchhoff, 1847; extended to directed graphs by Tutte, 1948).

**Definition (Weighted Laplacian).** Let $A \in \mathbb{R}^{n \times n}$ be the arc-weight matrix with $A_{ij} = \exp(\theta_{ij})$ for $i \neq j$ and $A_{ii} = 0$. The *weighted Laplacian* $\mathbf{L} \in \mathbb{R}^{n \times n}$ is:

$$L_{ij} = \begin{cases} \sum_{k \neq i} A_{ki} & i = j \\ -A_{ji} & i \neq j \end{cases}$$

**Theorem (Matrix-Tree).** The sum of weights of all spanning trees rooted at node $r$ equals any cofactor of $\mathbf{L}$ obtained by deleting row $r$ and column $r$. In the unrooted setting, the partition function is $\det(\mathbf{L}^{(r)})$ where $\mathbf{L}^{(r)}$ is the $(n{-}1) \times (n{-}1)$ reduced Laplacian.

**Corollary (Edge Marginals).** The marginal probability of arc $(j \to i)$ in the non-projective spanning-tree distribution is:

$$\mu_{ij} = A_{ij}\, [\mathbf{L}^{-1}]_{ii}$$

where $[\mathbf{L}^{-1}]_{ii}$ is the $(i,i)$ entry of the inverse of the reduced Laplacian. This requires a single matrix inversion, costing $O(n^3)$ in general but amenable to GPU parallelism via batched LAPACK routines.

---

## 5. Differentiable Dynamic Programming

For structured attention to be trained end-to-end, gradients must flow from the loss through the context vector $\mathbf{c}$, through the marginals $\mu$, and back to the arc potentials $\theta$ and the model parameters that produce them. The forward-backward and inside-outside algorithms are deterministic functions of the potentials; backpropagation requires differentiating through the dynamic programming recurrences.

### 5.1 Log-Space Computation

Numerical stability demands that forward and backward passes operate in *log-space*. Define the *log-sum-exp semiring* (also called the log-probability semifield):

$$a \oplus b = \log(\exp(a) + \exp(b)), \qquad a \otimes b = a + b$$

In this semiring, the forward recurrence becomes:

$$\log \alpha_i(z_i) = \theta_i z_i \oplus \bigoplus_{z_{i-1}} \left(\psi_{i-1,i}(z_{i-1}, z_i) \otimes \log \alpha_{i-1}(z_{i-1})\right)$$

which numerically is a sequence of `logsumexp` and addition operations — stable under overflow for large $n$.

### 5.2 Signed Log-Space Backward Pass

**The critical difficulty in backpropagating through DP recurrences is that gradient expressions involve differences of marginals, which can be negative.** The log-sum-exp semiring cannot represent negative quantities.

The backward pass therefore uses a *signed log-space semifield* that tracks each value as a pair $(\text{sign}, \log|\text{value}|) \in \{+1, -1\} \times \mathbb{R}$, with:

$$(\sigma_a, a') \oplus (\sigma_b, b') = \begin{cases} (\sigma_a, \,a' \oplus b') & \sigma_a = \sigma_b \\ (\operatorname{sgn}(e^{a'} - e^{b'}),\, |a' - b'|) & \text{otherwise (approximately)} \end{cases}$$

Multiplication is sign-multiplicative: $(\sigma_a \cdot \sigma_b,\; a' + b')$. This representation allows reverse-mode autodiff to propagate through DP recurrences that produce negative intermediate gradient values, without losing numerical stability.

### 5.3 Gradient Structure

**Proposition (Gradient of Context Vector).** Let $\mathbf{c} = \sum_i \mu_i \mathbf{x}_i$ be the CRF attention context vector. The gradient of a scalar loss $\ell$ with respect to the unary potential $\theta_i$ is:

$$\frac{\partial \ell}{\partial \theta_i} = \frac{\partial \ell}{\partial \mathbf{c}} \cdot \mathbf{x}_i \cdot \frac{\partial \mu_i}{\partial \theta_i} + \sum_{j \neq i} \frac{\partial \ell}{\partial \mathbf{c}} \cdot \mathbf{x}_j \cdot \frac{\partial \mu_j}{\partial \theta_i}$$

The terms $\partial \mu_j / \partial \theta_i$ are entries of the *Jacobian of the marginal map*, which encodes how changing the potential at one position affects marginals at all others — a quantity that is nonzero in the CRF model (through pairwise couplings) but zero in the softmax model (independent positions). **The full Jacobian has the same dynamic programming structure as the forward pass, with recurrences running in reverse:** the backward recurrences for $\partial \mu / \partial \theta$ have exactly the same topology as the forward-backward algorithm.

---

## 6. Relation to Standard Attention

The following table places standard softmax attention, linear-chain CRF attention, and parser attention within the unified marginal-inference framework.

| Dimension | Standard Softmax Attention | Linear-Chain CRF Attention | Parser Attention |
|---|---|---|---|
| Latent variable $z$ | Categorical (one-hot) | Binary sequence | Projective dependency tree |
| Marginal inference | Softmax (closed-form) | Forward-backward, $O(n)$ | Inside-outside, $O(n^3)$ |
| Context vector | $\sum_i \operatorname{softmax}(\theta)_i \mathbf{x}_i$ | $\sum_i \mu_i \mathbf{x}_i$ | $\sum_{ij} \mu_{ij} \mathbf{x}_j$ |
| Structural bias | None | Sequential contiguity | Tree structure |

**Theorem (Softmax as Degenerate CRF).** Standard softmax attention is the special case of linear-chain CRF attention where all pairwise potentials are zero: $\psi_{i,i+1} \equiv 0$ for all $i$. In this case the joint distribution factorizes as a product of independent Bernoullis:

$$p(z \mid \mathbf{x}, \mathbf{q}) \propto \prod_{i=1}^{n} \exp(\theta_i z_i)$$

and the marginals are $\mu_i = \sigma(\theta_i)$ (sigmoid), which under the constraint $\sum_i \mu_i = 1$ — enforced by an additional normalization — reduces to the softmax distribution.

*More precisely,* the categorical softmax model and the binary CRF model with $\psi = 0$ are not literally identical (categorical $z$ selects exactly one index; binary $z$ can select any subset), but both recover the same context vector formula $\mathbf{c} = \sum_i \operatorname{softmax}(\theta)_i \mathbf{x}_i$ in the limit where the model is constrained to select exactly one position. The CRF with nonzero $\psi$ is therefore a strict generalization, adding sequential dependencies without changing the form of the context vector computation.

**The key theoretical unification: structured attention subsumes softmax attention as the zero-coupling degenerate case, and extends it to any tractable graphical model by substituting the appropriate marginal inference algorithm.**

---

## 7. Experiments and Results

Kim et al. (2017) evaluate structured attention on four tasks: tree transduction, neural machine translation (NMT), question answering, and natural language inference (SNLI).

**Tree transduction.** Inputs are trees of depth up to 3; the task requires the model to transduce the tree structure from the input. Softmax attention achieves 49.6% accuracy; linear-chain CRF attention achieves 87.0%. The structured model learns to segment inputs along tree boundaries without any explicit structural supervision.

**NMT (Japanese-English, character-to-word).** CRF attention achieves 14.6 BLEU versus 12.6 BLEU for softmax attention, a gain of 2.0 BLEU points. The improvement is attributed to the model's ability to attend to contiguous character spans corresponding to morphological units.

**SNLI.** Structured attention achieves 86.8% accuracy, matching the performance of intra-sentence attention models while additionally inducing interpretable parse-like structure over premise-hypothesis pairs without supervision.

*The parser attention model requires $O(n^3)$ inference per attention head, making it more expensive than CRF attention at longer sequence lengths. In practice, the cubic cost is acceptable for sentence-level tasks (typical $n \leq 50$) but prohibitive for document-level or large-sequence settings.*

---

## References

| Reference Name | Brief Summary | Link to Reference |
|---|---|---|
| Kim et al. (2017), "Structured Attention Networks" | Introduces structured attention by replacing softmax with CRF and parser-based distributions; shows gains on NMT, QA, SNLI | [arXiv:1702.00887](https://arxiv.org/abs/1702.00887) |
| Lafferty et al. (2001), "Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data" | Foundational paper defining linear-chain CRFs and the forward-backward algorithm for exact marginal inference | [ACM DL](https://dl.acm.org/doi/10.5555/645530.655813) |
| Eisner (1996), "Three New Probabilistic Models for Dependency Parsing" | Introduces the Eisner algorithm for projective dependency parsing; basis for $O(n^3)$ inside-outside computation used in parser attention | [ACL Anthology](https://aclanthology.org/C96-1058/) |
| Kirchhoff (1847) / Tutte (1948), Matrix-Tree Theorem | Classical result giving the partition function and edge marginals of spanning-tree distributions via the determinant of the weighted graph Laplacian | — |
| Vaswani et al. (2017), "Attention Is All You Need" | Introduces the Transformer and scaled dot-product softmax attention; the baseline mechanism that structured attention generalizes | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| Bahdanau et al. (2015), "Neural Machine Translation by Jointly Learning to Align and Translate" | Original additive attention mechanism for NMT; precursor to the attention-as-marginal-inference framing | [arXiv:1409.0473](https://arxiv.org/abs/1409.0473) |
