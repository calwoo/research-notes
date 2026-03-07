# Neural Scaling Laws Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a unified, mathematically rigorous concept note on neural scaling laws (Kaplan 2020 + Chinchilla 2022), with a matching exercise set and full solutions.

**Architecture:** Three parallel files under `notes/concepts/`, `exercises/concepts/`, and `solutions/concepts/`, all sharing the slug `neural-scaling-laws`. The note builds intuition first, then full derivations. Exercises follow the repo-mandated order: derivations → conceptual → implementation sketches.

**Tech Stack:** Markdown with LaTeX math notation. No code execution. Source papers: Kaplan et al. (2020) "Scaling Laws for Neural Language Models", Hoffmann et al. (2022) "Training Compute-Optimal Large Language Models".

---

### Task 1: Write the main concept note — Motivation + Mathematical Setup

**Files:**
- Create: `notes/concepts/neural-scaling-laws.md`

**Step 1: Write the Motivation section**

Cover:
- The empirical observation: when you plot loss vs. model size, data size, or compute on a log-log scale, you get a straight line
- Power laws in nature: Zipf's law, Pareto distributions — scaling laws are not unique to neural networks
- Why this matters: predicting loss at scale without running the experiments

**Step 2: Write the Mathematical Setup section**

Cover the loss decomposition model:

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

Define each term:
- $E = L_\infty$: irreducible loss (Bayes error floor) — the entropy of natural language itself; no model can go below this
- $A/N^\alpha$: capacity-limited term — larger models reduce this; $\alpha > 0$ governs how fast
- $B/D^\beta$: data-limited term — more data reduces this; $\beta > 0$ governs how fast
- $N$: number of non-embedding parameters
- $D$: number of training tokens

Explain why the additive structure is plausible:
- Each term represents an independent source of "excess loss" above the irreducible floor
- Power-law decay is the simplest scale-free functional form consistent with the empirical data

Also define the compute approximation:
$$C \approx 6ND \quad \text{(FLOPs for a transformer forward+backward pass)}$$

Derive this: each parameter is touched once per token per forward pass (~2 FLOPs multiply-add), and the backward pass costs ~2x forward, giving $C \approx 6ND$.

**Step 3: Commit**

```bash
git add notes/concepts/neural-scaling-laws.md
git commit -m "feat: add motivation and mathematical setup to scaling laws note"
```

---

### Task 2: Write the Kaplan et al. (2020) section

**Files:**
- Modify: `notes/concepts/neural-scaling-laws.md`

**Step 1: Write the univariate scaling laws**

Kaplan et al. fit three separate power laws by varying one variable while fixing the others:

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad N_c \approx 8.8 \times 10^{13}, \quad \alpha_N \approx 0.076$$

$$L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad D_c \approx 5.4 \times 10^{13}, \quad \alpha_D \approx 0.095$$

$$L(C_{\min}) = \left(\frac{C_c}{C_{\min}}\right)^{\alpha_C}, \quad \alpha_C \approx 0.050$$

Explain:
- These are fitted empirically via log-linear regression: $\log L = \text{const} - \alpha \log N$
- "While fixing others" means: for $L(N)$, train until convergence (data not a bottleneck); for $L(D)$, use a single epoch (compute not a bottleneck)

**Step 2: Derive the compute-efficient frontier**

Setup: given a compute budget $C = 6ND$, how should we split it between $N$ and $D$?

Kaplan assumes both terms in the loss decomposition are active:
$$L(N, D) \approx \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

Subject to: $6ND = C$, so $D = C/(6N)$.

Substitute:
$$L(N) = \frac{A}{N^\alpha} + \frac{B \cdot (6N)^\beta}{C^\beta}$$

Minimize over $N$ by taking $dL/dN = 0$:
$$-\frac{\alpha A}{N^{\alpha+1}} + \frac{6^\beta \beta B}{C^\beta} N^{\beta - 1} = 0$$

Solve for $N^*$:
$$N^{*\,\alpha+\beta} = \frac{\alpha A C^\beta}{6^\beta \beta B}$$

$$N^* \propto C^{\,\beta/(\alpha+\beta)}$$

And since $D^* = C/(6N^*)$:
$$D^* \propto C^{\,\alpha/(\alpha+\beta)}$$

Kaplan's fitted values give $\alpha \approx 0.076$, $\beta \approx 0.095$, so:
$$\frac{\beta}{\alpha+\beta} \approx \frac{0.095}{0.171} \approx 0.56 \quad \Rightarrow \quad N^* \propto C^{0.73}, \quad D^* \propto C^{0.27}$$

Interpretation: parameters should scale ~3x faster than data per unit compute.

**Step 3: Commit**

```bash
git add notes/concepts/neural-scaling-laws.md
git commit -m "feat: add Kaplan 2020 scaling laws and compute frontier derivation"
```

---

### Task 3: Write the Chinchilla (2022) section

**Files:**
- Modify: `notes/concepts/neural-scaling-laws.md`

**Step 1: Write the IsoFLOP analysis**

Chinchilla's key methodological innovation: instead of varying $N$ while fixing $D$ (or vice versa), they fix $C$ and jointly sweep $N$ and $D$ subject to $C = 6ND$.

For each fixed compute level $C$, they train many models along the IsoFLOP curve and record which $(N, D)$ pair achieves the lowest loss. Empirically: the optimal split is approximately $N^* \approx D^*/20$ — or equivalently, $D^* \approx 20 N^*$.

**Step 2: Analytical derivation via Lagrangian**

Minimize:
$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

Subject to constraint $g(N, D) = 6ND - C = 0$.

Form the Lagrangian:
$$\mathcal{L}(N, D, \lambda) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta} + \lambda(6ND - C)$$

First-order conditions:
$$\frac{\partial \mathcal{L}}{\partial N} = -\frac{\alpha A}{N^{\alpha+1}} + 6\lambda D = 0 \quad \Rightarrow \quad \lambda = \frac{\alpha A}{6 D N^{\alpha+1}}$$

$$\frac{\partial \mathcal{L}}{\partial D} = -\frac{\beta B}{D^{\beta+1}} + 6\lambda N = 0 \quad \Rightarrow \quad \lambda = \frac{\beta B}{6 N D^{\beta+1}}$$

Set equal:
$$\frac{\alpha A}{D N^{\alpha+1}} = \frac{\beta B}{N D^{\beta+1}}$$

$$\alpha A \cdot D^{\beta} = \beta B \cdot N^{\alpha}$$

$$\frac{D^\beta}{N^\alpha} = \frac{\beta B}{\alpha A}$$

Combined with $6ND = C$, this gives (after substitution):
$$N^* \propto C^{1/2}, \quad D^* \propto C^{1/2}$$

Both scale as $C^{0.5}$ — equal scaling. Chinchilla's fitted constants give approximately:
$$N^* \approx \frac{1}{20} D^* \quad \Rightarrow \quad \text{"20 tokens per parameter"}$$

**Step 3: Explain why Kaplan was wrong**

Kaplan's exponents $\alpha \approx 0.076, \beta \approx 0.095$ were estimated from models trained with fixed data budgets — meaning the models were undertrained relative to their size. This biases $\alpha$ downward (parameters look less powerful than they are) and makes the optimal $N/D$ ratio appear to favor parameters more heavily than it should.

Chinchilla's IsoFLOP methodology eliminates this bias by always comparing models at the same compute level.

**Step 4: Commit**

```bash
git add notes/concepts/neural-scaling-laws.md
git commit -m "feat: add Chinchilla Lagrangian derivation and Kaplan comparison"
```

---

### Task 4: Write the Fitting Methodology section + References

**Files:**
- Modify: `notes/concepts/neural-scaling-laws.md`

**Step 1: Write the Fitting Methodology section**

Cover three approaches Chinchilla uses to estimate $E, A, B, \alpha, \beta$:

**Approach 1 — IsoFLOP minimum fitting:**
For each fixed $C$, find the $(N^*, D^*)$ that minimizes loss empirically. Fit the relationship $N^*(C) \propto C^a$ in log-log space.

**Approach 2 — Parametric fit:**
Fit all five parameters $(E, A, B, \alpha, \beta)$ jointly by minimizing:
$$\min_{E, A, B, \alpha, \beta} \sum_{\text{runs}} \left( L_{\text{obs}} - E - \frac{A}{N^\alpha} - \frac{B}{D^\beta} \right)^2$$
using L-BFGS in log-space (to enforce positivity and handle the multiplicative structure).

**Approach 3 — Per-model exponent estimation:**
For each model size $N$, fit $L(D) = E_N + B_N/D^\beta$ separately, then fit $E_N$ as a function of $N$.

All three approaches agree on $N^* \propto C^{0.5}$, giving confidence in the result.

**Step 2: Write the References table**

| Reference Name | Brief Summary | Link |
|---|---|---|
| Kaplan et al. (2020) — "Scaling Laws for Neural Language Models" | Establishes power-law scaling of LM loss with N, D, C; derives compute-efficient frontier favoring parameter scaling | https://arxiv.org/abs/2001.08361 |
| Hoffmann et al. (2022) — "Training Compute-Optimal Large Language Models" (Chinchilla) | Revises Kaplan via IsoFLOP analysis; shows equal N and D scaling is optimal; introduces 20-tokens-per-parameter rule | https://arxiv.org/abs/2203.15556 |
| Henighan et al. (2020) — "Scaling Laws for Autoregressive Generative Modeling" | Extends scaling laws beyond language to images, video, math | https://arxiv.org/abs/2010.14701 |
| Bahri et al. (2021) — "Explaining Neural Scaling Laws" | Provides a theoretical explanation for power-law scaling via statistical mechanics | https://arxiv.org/abs/2102.06701 |

**Step 3: Commit**

```bash
git add notes/concepts/neural-scaling-laws.md
git commit -m "feat: add fitting methodology and references to scaling laws note"
```

---

### Task 5: Write exercises file

**Files:**
- Create: `exercises/concepts/neural-scaling-laws.md`

**Step 1: Write Derivation Problems (section 1)**

Problems should require the student to re-derive key results:

1. Starting from $L(N, D) = E + A/N^\alpha + B/D^\beta$ with constraint $C = 6ND$, derive the compute-optimal scaling $N^* \propto C^{\beta/(\alpha+\beta)}$ using substitution (Kaplan's approach). Show all steps.

2. Re-derive the same result using the Lagrangian method. Show that both methods give identical answers.

3. Under what condition on $\alpha$ and $\beta$ do parameters and data scale equally (i.e., $N^* \propto D^*$)? What does this imply about the loss surface?

4. The compute approximation $C \approx 6ND$ comes from counting FLOPs in a transformer. Derive this from first principles: consider a single linear layer $y = Wx$ with $W \in \mathbb{R}^{d \times d}$, count FLOPs for forward and backward passes, then generalize.

5. If $\alpha = \beta$ (equal exponents), show that the Kaplan and Chinchilla conclusions coincide. What does $\alpha = \beta$ say about the relative information capacity of parameters vs. data?

**Step 2: Write Conceptual Questions (section 2)**

1. What is the irreducible loss $E = L_\infty$? Why can no model, regardless of size or data, achieve loss below $E$? Give a concrete example of what $E$ represents in language modeling.

2. Kaplan et al. found $\alpha_N \approx 0.076$ and $\alpha_D \approx 0.095$ with $\alpha_D > \alpha_N$. What does this inequality imply about the relative effectiveness of adding parameters vs. adding data, holding compute fixed?

3. Explain in your own words why the IsoFLOP methodology eliminates the undertrained-model bias present in Kaplan's approach. Why does fixing $C$ while sweeping $N$ and $D$ give more reliable exponent estimates?

4. The "20 tokens per parameter" rule comes from Chinchilla's fitted constants, not from the exponents alone. Why can't the optimal $N/D$ ratio be determined from $\alpha$ and $\beta$ alone? What additional information is needed?

5. If you have a compute budget $C$ and want to minimize loss, but you are also constrained by a maximum model size $N_{\max}$ (e.g., due to memory), how would you adjust the compute-optimal strategy? Describe qualitatively.

**Step 3: Write Implementation Sketches (section 3)**

1. Sketch an algorithm for the IsoFLOP sweep used in Chinchilla. Given a set of compute budgets $\{C_1, \ldots, C_k\}$, describe how you would design the experiment to find $N^*(C_i)$ for each budget. Include: how you choose $(N, D)$ pairs, what you measure, and how you find the minimum.

2. Sketch a log-linear regression procedure for estimating the scaling exponent $\alpha$ from a set of observations $\{(N_i, L_i)\}$. Write out the transformed regression equation and describe how to compute $\alpha$ and its confidence interval.

3. Sketch the L-BFGS fitting procedure for Approach 2 (parametric fit). Define the objective function, explain why fitting in log-space is preferable, and describe what initialization strategy you would use to avoid local minima.

**Step 4: Commit**

```bash
git add exercises/concepts/neural-scaling-laws.md
git commit -m "feat: add exercises for neural scaling laws (derivations, conceptual, implementation)"
```

---

### Task 6: Write solutions file

**Files:**
- Create: `solutions/concepts/neural-scaling-laws.md`

**Step 1: Write solutions to all Derivation Problems**

Provide complete, worked mathematical solutions with all steps shown for each of the 5 derivation problems.

**Step 2: Write solutions to all Conceptual Questions**

Provide thorough answers to each of the 5 conceptual questions, referencing specific equations and results from the note.

**Step 3: Write solutions to all Implementation Sketches**

Provide complete pseudocode and descriptions for each of the 3 implementation sketches.

**Step 4: Commit**

```bash
git add solutions/concepts/neural-scaling-laws.md
git commit -m "feat: add solutions for neural scaling laws exercises"
```

---

### Task 7: Final review and cross-check

**Files:**
- Review: `notes/concepts/neural-scaling-laws.md`
- Review: `exercises/concepts/neural-scaling-laws.md`
- Review: `solutions/concepts/neural-scaling-laws.md`

**Step 1: Check internal consistency**

- All notation is consistent across note, exercises, and solutions
- Every exercise has a corresponding solution
- The references table in the note is complete
- Math renders correctly (balanced delimiters, correct LaTeX)

**Step 2: Final commit**

```bash
git add notes/concepts/neural-scaling-laws.md exercises/concepts/neural-scaling-laws.md solutions/concepts/neural-scaling-laws.md
git commit -m "feat: complete neural scaling laws note, exercises, and solutions"
```
