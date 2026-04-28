# Phase I — Backprop & Training Basics

*Weeks 1–6 · ~12 hrs*

> **Goal:** Understand what a neural network actually does — gradients, loss, the training loop — by building one from scratch. By the end, you should be able to implement a training loop in PyTorch from memory, explain why `loss.backward()` works, and diagnose the difference between overfitting and underfitting from a loss curve.

**Weeks 1–2 primary:** Karpathy, [micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) — builds a scalar-valued autograd engine from nothing (Python only, no NumPy).

**Weeks 3–4 primary:** Karpathy, [makemore Part 1](https://www.youtube.com/watch?v=PaCmpygFfXo) + [PyTorch in 25 minutes](https://www.youtube.com/watch?v=ic55579V8ag) — first real training loop in PyTorch.

**Weeks 5–6 primary:** Fleuret, [*The Little Book of Deep Learning*](https://fleuret.org/public/lbdl.pdf), Chapters 1–3 + Karpathy, ["A Recipe for Training Neural Networks"](http://karpathy.github.io/2019/04/25/recipe/).

> **Why this structure:** Micrograd is the single most effective entry point for a hands-on learner. You build the backward pass yourself, so `loss.backward()` in PyTorch is never magic — it is literally the chain rule applied node by node on a computation graph. Weeks 3–4 transfer that understanding to real PyTorch. Weeks 5–6 put names to the things you have already observed.

---

## Weeks 1–2 — micrograd: Backpropagation from Scratch

**Primary resource:** [Karpathy — micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) (2h 25m)

Code along. Do not watch passively. Pause after each operation is implemented, run it, and verify the gradient by finite differences (perturb the input by `h = 1e-4` and check `(f(x+h) - f(x)) / h ≈ backward()`).

---

### Week 1 — Computation Graphs and the Chain Rule

**Concepts to understand:**

- [ ] A scalar `Value` object: wraps a number and tracks how it was created (`_op`, `_children`)
- [ ] Forward pass: evaluate an expression and record the computation graph
- [ ] Backward pass: traverse the graph in reverse topological order, accumulating `.grad` at each node
- [ ] Chain rule: `∂L/∂x = (∂L/∂z)(∂z/∂x)` — every `.backward()` method is just this, applied locally
- [ ] `+`, `*`, `tanh` operations: implement `_backward` for each, verifying the gradient formula

**Coding tasks:**

- [ ] Implement `Value.__add__`, `Value.__mul__`, `Value.tanh` with correct `_backward` hooks
- [ ] Build the expression `L = ((a * b) + c).tanh()` and call `L.backward()`. Print `.grad` on each leaf.
- [ ] Verify `a.grad` by finite differences: check `(L(a + 1e-4) - L(a)) / 1e-4 ≈ a.grad`

> [!NOTE] Milestone
> After implementing micrograd `+` and `*`, evaluate `f = a * b + c` with `a=2.0, b=-3.0, c=10.0`. The forward pass gives `f = 4.0`. Call `f.backward()`. By the chain rule: `∂f/∂a = b = -3.0`, `∂f/∂b = a = 2.0`, `∂f/∂c = 1.0`. If your `.grad` values match these, the backward pass is correct. The fact that this is all just arithmetic — no magic — is the point.

---

### Week 2 — Building a Neural Network on Top of micrograd

**Concepts to understand:**

- [ ] A `Neuron`: a linear combination of inputs followed by `tanh` — `out = tanh(w · x + b)`
- [ ] A `Layer`: a list of neurons applied in parallel to the same input
- [ ] A `MLP`: a sequence of layers; the final layer has a single output (for regression/binary classification)
- [ ] Loss: mean squared error `L = (1/N) Σ (y_pred - y_true)²`; why squaring makes sense
- [ ] Gradient descent update: `p.data -= lr * p.grad` for every parameter `p`; why `p.grad` must be zeroed between steps

**Coding tasks:**

- [ ] Implement `Neuron`, `Layer`, `MLP` on top of your `Value` engine
- [ ] Train on the XOR dataset: `[(0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0]`
- [ ] Print loss every 10 steps for 100 steps; verify loss decreases monotonically with a small enough learning rate

> [!NOTE] Milestone
> XOR is the canonical "two layers required" problem. A single neuron (one `tanh` of a linear combination) cannot solve it — the decision boundary is a single line in 2D, but XOR is not linearly separable. An MLP with one hidden layer of ≥2 neurons can solve it. If your loss stalls above 0.1 with a single neuron, that is correct behavior. If it stalls with a 2-2-1 MLP and lr=0.1, check that you are zeroing gradients before each step — forgetting `p.grad = 0` is the single most common bug.

---

## Weeks 3–4 — The Real Training Loop in PyTorch

**Primary resource:** [Karpathy — makemore Part 1](https://www.youtube.com/watch?v=PaCmpygFfXo) (1h 57m) + [PyTorch in 25 minutes](https://www.youtube.com/watch?v=ic55579V8ag) (25m)

makemore is a character-level language model. Karpathy builds it step by step. Switch to PyTorch here — micrograd is pedagogically complete; PyTorch is what you will use for everything else.

---

### Week 3 — PyTorch Fundamentals and the Training Loop

**Concepts to understand:**

- [ ] `torch.Tensor`: multi-dimensional array with `.grad` tracking; `requires_grad=True` enables autograd
- [ ] `nn.Module` and `nn.Parameter`: how PyTorch structures learnable parameters
- [ ] The four-step training loop: `optimizer.zero_grad()` → forward → `loss.backward()` → `optimizer.step()`
- [ ] Why `zero_grad()` first: PyTorch accumulates gradients by default — not zeroing between steps sums gradients over batches
- [ ] Mini-batching: sampling a random subset of `B` examples each step; the gradient is an unbiased estimate of the full-dataset gradient
- [ ] `F.cross_entropy`: takes raw logits (unnormalized scores), applies log-softmax internally — never manually softmax before cross-entropy

**Coding tasks:**

- [ ] Implement makemore's bigram model: an embedding table `C` of shape `(27, 27)` + cross-entropy loss
- [ ] Write the training loop manually: no `Trainer`, no `fit()` — every line explicit
- [ ] Sample from the trained model: write the generation loop from scratch

> [!NOTE] Milestone
> At initialization (random weights), the loss for a bigram character model over a 27-character alphabet should be approximately `-log(1/27) ≈ 3.3`. If your initial loss is wildly different (e.g., 10+), weights are not initialized correctly. If it is exactly 3.3, the model has no useful information yet — it assigns uniform probability to all next characters. After 10,000 steps with lr=10, you should see loss around 2.45. If you are above 2.6, check that `F.cross_entropy` receives logits, not probabilities.

---

### Week 4 — Embeddings, Train/Val Split, and Overfitting

**Concepts to understand:**

- [ ] Embedding lookup: `C[x]` retrieves rows of `C` for each index in `x` — equivalent to one-hot times `C`, but O(1)
- [ ] Context window: using the last `k` characters to predict the next; k=1 is bigram, k=3 is trigram
- [ ] Train/val/test split: 80/10/10 — why the test set must never be seen during development
- [ ] Overfitting signal: training loss ≪ validation loss; the model memorizes rather than generalizes
- [ ] Underfitting signal: both losses plateau high; the model lacks capacity or has too large a learning rate

**Coding tasks:**

- [ ] Upgrade makemore to a trigram MLP: embed the 3 context characters, concatenate, pass through one hidden layer, predict next character
- [ ] Deliberately overfit: train on only 50 names and watch validation loss diverge from training loss
- [ ] Plot both curves with matplotlib; label the point where overfitting visibly begins

> [!NOTE] Milestone
> With 50 training examples and 1000+ training steps, training loss should reach near 0 while validation loss stays around 2.0–3.0. This gap is overfitting. The correct interpretation: the model memorized the 50 names rather than learning the structure of names. When you train on the full dataset (~32,000 names), the gap should shrink substantially — both losses converge to ~2.1, meaning the model is learning genuine patterns in the data.

---

## Weeks 5–6 — Loss Functions, Optimizers, and the Recipe

**Primary resources:**
- Fleuret, [*The Little Book of Deep Learning*](https://fleuret.org/public/lbdl.pdf), Ch 1–3 (~1.5 hrs)
- Karpathy, ["A Recipe for Training Neural Networks"](http://karpathy.github.io/2019/04/25/recipe/) (~30 min)

Read Ch 1–3 of the Little Book now to put names to things already observed. Do not attempt to read cover to cover — return to later chapters as needed throughout Phases II and III.

---

### Week 5 — Loss Functions and What They Optimize

**Concepts to understand:**

- [ ] Cross-entropy loss `H(p, q) = -Σ p(x) log q(x)`: the right loss for classification; minimizing it is equivalent to maximizing log-likelihood of the correct class
- [ ] Mean squared error `L = (1/N)||y - ŷ||²`: the right loss for regression; minimizing it is equivalent to MLE under a Gaussian noise model
- [ ] Logits vs. probabilities: never apply softmax before cross-entropy — numerical instability from `log(softmax(x))` vs. the numerically stable `log_softmax`
- [ ] The baseline loss sanity check: at initialization, loss should match `-log(1/num_classes)` for classification; if it doesn't, the initialization is wrong

**Coding tasks:**

- [ ] Verify the baseline loss on your makemore MLP at step 0; confirm it is ≈3.3
- [ ] Swap cross-entropy for MSE (treating one-hot targets as regression targets); observe the training dynamics difference

> [!NOTE] Milestone
> Training the same character MLP with MSE instead of cross-entropy: MSE will converge more slowly and to a worse final loss. The reason is that MSE does not match the distributional structure of the problem — character prediction is a 27-way classification, and MSE penalizes all wrong predictions equally regardless of how wrong they are. Cross-entropy penalizes confident wrong predictions exponentially more, which is the correct inductive bias for a probability model.

---

### Week 6 — Optimizers: SGD vs Adam, and the Hyperparameter Hierarchy

**Concepts to understand:**

- [ ] SGD: `θ ← θ - η∇L`; momentum SGD: `v ← βv + ∇L`, `θ ← θ - ηv` — accumulates a running velocity
- [ ] Adam: maintains per-parameter estimates of first moment (momentum) and second moment (adaptive scale); `η_eff = η / (√v̂ + ε)` shrinks the step size for high-variance parameters
- [ ] Why Adam usually converges faster: it normalizes gradient scale per parameter, so a single learning rate works across parameters with very different gradient magnitudes
- [ ] Why SGD sometimes generalizes better: the noise of SGD acts as a regularizer; Adam can converge to sharp minima that generalize worse
- [ ] The hyperparameter hierarchy: learning rate matters most; batch size second; most other things are secondary

**Coding tasks:**

- [ ] Train your makemore MLP with three configurations: SGD (lr=0.1), SGD (lr=0.01), Adam (lr=3e-4)
- [ ] Plot all three loss curves on the same axes; record the final validation loss for each

> [!NOTE] Milestone
> Expected observations: Adam at lr=3e-4 should converge in ~10,000 steps to loss ≈2.1. SGD at lr=0.1 will converge in ~20,000–50,000 steps to a similar loss. SGD at lr=0.01 will converge very slowly — loss still around 2.5–3.0 after 10,000 steps. The takeaway is not that Adam is universally better, but that lr=3e-4 is a reliable default for Adam while SGD requires tuning per problem. Karpathy's recipe: "start with Adam, switch to SGD+momentum later if you need the generalization boost."
