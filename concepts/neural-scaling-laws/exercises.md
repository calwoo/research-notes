# Exercises: Neural Scaling Laws

---

## 1. Derivation Problems

**Problem 1.** Starting from the loss decomposition $L(N, D) = E + A/N^\alpha + B/D^\beta$ with compute constraint $C = 6ND$, derive the compute-optimal scaling $N^* \propto C^{\beta/(\alpha+\beta)}$ using direct substitution (Kaplan's approach). Begin by eliminating $D$, then minimize $dL/dN = 0$. Show all steps.

**Problem 2.** Re-derive the same compute-optimal scaling using Lagrangian optimization. Form the Lagrangian

$$\mathcal{L}(N, D, \lambda) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta} + \lambda(6ND - C),$$

derive both first-order conditions, eliminate $\lambda$, and combine with the constraint. Verify your answer matches Problem 1.

**Problem 3.** The optimality condition from the Lagrangian is $\alpha A \cdot D^\beta = \beta B \cdot N^\alpha$. Under what condition on $\alpha$ and $\beta$ does $N^* \propto D^*$ (equal scaling of parameters and data)? Prove your answer algebraically, and give a geometric interpretation in terms of the loss surface.

**Problem 4.** The compute approximation $C \approx 6ND$ comes from counting FLOPs in a transformer. Derive it from first principles:

(a) Count FLOPs for a single linear layer $y = Wx$, $W \in \mathbb{R}^{m \times n}$, in the forward pass.

(b) Count FLOPs for the backward pass (gradient w.r.t. input and gradient w.r.t. weight separately).

(c) Argue why the total is $\approx 6ND$ for a transformer.

**Problem 5.** Suppose $\alpha = \beta$ (equal exponents). Show that in this case the Kaplan and Chinchilla derivations give the same scaling exponents $N^* \propto D^* \propto C^{1/2}$, regardless of the values of $A$, $B$, and $E$. What does $\alpha = \beta$ say about the relative "efficiency" of parameters vs. data at reducing excess loss?

---

## 2. Conceptual Questions

**Question 1.** What is the irreducible loss $E = L_\infty$? Explain why no model — regardless of how large or how much data it is trained on — can achieve loss below $E$. Give a concrete example of what $E$ represents in the context of language modeling. Is $E$ in principle computable?

**Question 2.** Kaplan et al. find $\alpha_D \approx 0.095 > \alpha_N \approx 0.076$. What does the inequality $\alpha_D > \alpha_N$ imply about the relative effectiveness of adding data vs. adding parameters to reduce loss, holding compute fixed? Be precise: which is more efficient, by how much, and why does this inequality reverse the Kaplan compute-optimal conclusion (parameters scale faster)?

**Question 3.** Explain in your own words why the IsoFLOP experimental design eliminates the undertrained-model bias present in Kaplan's univariate approach. Specifically:

(a) What is the bias mechanism in Kaplan's $L(N)$ measurement?

(b) Why does fixing $C$ while sweeping $(N, D)$ remove this bias?

**Question 4.** The "20 tokens per parameter" rule comes from Chinchilla's fitted constants, not from the exponents $\alpha$ and $\beta$ alone. Why can't the optimal token-to-parameter ratio $D^*/N^*$ be determined from $\alpha$ and $\beta$ without knowing $A$ and $B$? What additional information is needed, and why?

**Question 5.** You have a compute budget $C$ and want to minimize loss, but your hardware limits you to a maximum model size $N_{\max}$ (e.g., due to GPU memory). Describe qualitatively how you would adjust the compute-optimal strategy. How does the optimal $D^*$ change relative to the unconstrained case? What happens to the achievable loss?

---

## 3. Implementation Sketches

**Sketch 1 — IsoFLOP sweep.** Write pseudocode for the IsoFLOP experimental design. Given a set of compute budgets $\{C_1, \ldots, C_k\}$, describe how you would:

(a) Generate a set of $(N_j, D_j)$ pairs for each budget $C_i$ satisfying $6N_j D_j = C_i$.

(b) Decide how many models to train per budget.

(c) Measure the loss for each run.

(d) Estimate $N^*(C_i)$ from the results.

What does the final output look like, and how would you fit the scaling exponent from it?

**Sketch 2 — Log-linear regression for scaling exponents.** Write the mathematical procedure for estimating the exponent $\alpha$ from observations $\{(N_i, L_i)\}_{i=1}^n$ (univariate case, $D$ fixed). Define the transformed variables, write the OLS normal equations, and describe how to compute a confidence interval for $\hat{\alpha}$. What assumption about residuals is required for the confidence interval to be valid?

**Sketch 3 — Parametric fit via L-BFGS.** Define the objective function for Chinchilla's Approach 2 parametric fit. Explain:

(a) Why fitting in log-space (reparametrizing $E \to \exp(e)$, etc.) is preferable to fitting directly.

(b) What gradient $\nabla_{e,\, a,\, b,\, \alpha,\, \beta}$ of the objective looks like.

(c) What initialization strategy you would use.

(d) How you would detect and handle local minima.
