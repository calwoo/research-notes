# Exercises: Neural Scaling Laws

## Table of Contents

- [[#Mathematical Development|Mathematical Development]]
  - [[#Problem 1 Compute-Optimal Scaling via Direct Substitution|Problem 1: Compute-Optimal Scaling via Direct Substitution]]
  - [[#Problem 2 Compute-Optimal Scaling via Lagrangian Optimization|Problem 2: Compute-Optimal Scaling via Lagrangian Optimization]]
  - [[#Problem 3 The Equal-Scaling Condition|Problem 3: The Equal-Scaling Condition]]
  - [[#Problem 4 The Factor-of-Six Approximation|Problem 4: The Factor-of-Six Approximation]]
  - [[#Problem 5 The Symmetric Exponent Limit|Problem 5: The Symmetric Exponent Limit]]
  - [[#Problem 6 Second-Order Optimality Condition|Problem 6: Second-Order Optimality Condition]]
  - [[#Problem 7 The IsoFLOP Curve as a Hyperbola|Problem 7: The IsoFLOP Curve as a Hyperbola]]
  - [[#Problem 8 Compute-Optimal Loss Exponent|Problem 8: Compute-Optimal Loss Exponent]]
  - [[#Problem 9 Sensitivity of the Optimal Ratio to Fitted Constants|Problem 9: Sensitivity of the Optimal Ratio to Fitted Constants]]
  - [[#Problem 10 Chinchilla Optimum via the Envelope Theorem|Problem 10: Chinchilla Optimum via the Envelope Theorem]]
  - [[#Problem 11 Excess Loss Suboptimality Bound|Problem 11: Excess Loss Suboptimality Bound]]
  - [[#Problem 12 When Parameters Are More Efficient Than Data|Problem 12: When Parameters Are More Efficient Than Data]]
  - [[#Problem 13 Irreducible Loss and the Bayes Floor|Problem 13: Irreducible Loss and the Bayes Floor]]
  - [[#Problem 14 Bias in the Kaplan Exponent Estimate|Problem 14: Bias in the Kaplan Exponent Estimate]]
  - [[#Problem 15 The 20-Tokens-Per-Parameter Rule|Problem 15: The 20-Tokens-Per-Parameter Rule]]
  - [[#Problem 16 Power Laws from a Mixture of Exponentials|Problem 16: Power Laws from a Mixture of Exponentials]]
  - [[#Problem 17 Scaling Exponents and Intrinsic Dimension|Problem 17: Scaling Exponents and Intrinsic Dimension]]
  - [[#Problem 18 Exponential vs. Power-Law Scaling Regimes|Problem 18: Exponential vs. Power-Law Scaling Regimes]]
- [[#Algorithmic Applications|Algorithmic Applications]]
  - [[#Problem 19 IsoFLOP Sweep Design|Problem 19: IsoFLOP Sweep Design]]
  - [[#Problem 20 Log-Linear Regression for Scaling Exponents|Problem 20: Log-Linear Regression for Scaling Exponents]]
  - [[#Problem 21 Parametric Fit via L-BFGS|Problem 21: Parametric Fit via L-BFGS]]
  - [[#Problem 22 Compute-Budget Allocation Algorithm|Problem 22: Compute-Budget Allocation Algorithm]]
  - [[#Problem 23 Per-Model-Size Two-Stage Estimator|Problem 23: Per-Model-Size Two-Stage Estimator]]

---

## Mathematical Development

### Problem 1: Compute-Optimal Scaling via Direct Substitution

*Starting from the two-term loss approximation and the compute constraint, this problem derives the compute-optimal parameter count $N^* \propto C^{\beta/(\alpha+\beta)}$ by reducing the constrained problem to a single-variable calculus exercise. The goal is to see exactly how the constraint geometry forces the two power-law terms to balance.*

> **Prerequisites:** cf. note [[note#3.2 The Compute-Efficient Frontier|§3.2 — The Compute-Efficient Frontier]]

(a) Write the loss as $L(N, D) \approx A/N^\alpha + B/D^\beta$ (dropping $E$). Using $C = 6ND$, eliminate $D$ to obtain $L$ as a function of $N$ and $C$ alone. Identify which term is increasing in $N$ and which is decreasing.

(b) Compute $dL/dN$ and set it to zero. Show that the critical point satisfies $\alpha A / N^{\alpha+1} = \beta B \cdot 6^\beta \cdot N^{\beta-1} / C^\beta$.

(c) Solve for $N^*$ and show $N^* \propto C^{\beta/(\alpha+\beta)}$. Then derive $D^* \propto C^{\alpha/(\alpha+\beta)}$ from the constraint.

(d) Plug in Kaplan's values $\alpha = 0.076$, $\beta = 0.095$ and Chinchilla's values $\alpha = 0.34$, $\beta = 0.28$. Compute the exponent $\beta/(\alpha+\beta)$ in each case and comment on the difference.

---

### Problem 2: Compute-Optimal Scaling via Lagrangian Optimization

*This problem re-derives the same $N^* \propto C^{\beta/(\alpha+\beta)}$ result using Lagrange multipliers, making explicit the economic interpretation of $\lambda$ as the shadow price of compute. Comparing with Problem 1 reveals what the two derivation routes share and where they differ.*

> **Prerequisites:** cf. note [[note#4.2 Analytical Derivation via Lagrangian Optimization|§4.2 — Analytical Derivation via Lagrangian Optimization]]; requires Problem 1

(a) Form the Lagrangian $\mathcal{L}(N, D, \lambda) = E + A/N^\alpha + B/D^\beta + \lambda(6ND - C)$. Write the three first-order conditions: $\partial\mathcal{L}/\partial N = 0$, $\partial\mathcal{L}/\partial D = 0$, and $\partial\mathcal{L}/\partial \lambda = 0$.

(b) From the first two conditions, derive two expressions for $\lambda$ and show that equating them yields the optimality condition $\alpha A \cdot D^\beta = \beta B \cdot N^\alpha$.

(c) Interpret $\lambda$ at the optimum: show that $\lambda = -\partial L^*/\partial C$ (the shadow price of compute), and that the optimality condition states that the marginal loss per unit of relaxed compute is equal whether the relaxation is used to increase $N$ or $D$.

(d) Combine the optimality condition with the constraint to recover $N^* \propto C^{\beta/(\alpha+\beta)}$. Verify consistency with Problem 1.

---

### Problem 3: The Equal-Scaling Condition

*The optimality condition $\alpha A \cdot D^\beta = \beta B \cdot N^\alpha$ is a constraint on the ratio $D^*/N^*$. This problem characterizes exactly when that ratio is scale-invariant (independent of $C$) and when $N^*$ and $D^*$ grow at the same rate.*

> **Prerequisites:** cf. note [[note#4.2 Analytical Derivation via Lagrangian Optimization|§4.2 — Analytical Derivation via Lagrangian Optimization]]; requires Problem 2

(a) From the optimality condition, express $D^*$ as a function of $N^*$ alone. Show that $D^* = G \cdot (N^*)^{\alpha/\beta}$ where $G = (\beta B / \alpha A)^{1/\beta}$.

(b) Hence show that $D^*/N^* = G \cdot (N^*)^{\alpha/\beta - 1}$. Under what condition on $\alpha$ and $\beta$ is this ratio constant (independent of $N^*$, and thus of $C$)?

(c) When $\alpha = \beta$, prove that $N^* \propto D^* \propto C^{1/2}$ exactly. What does this say about the loss surface geometry: along the optimal path, how does $A/N^\alpha$ compare to $B/D^\beta$?

(d) Give a geometric interpretation: in the $(N, D)$ plane, the constraint curve $6ND = C$ is a hyperbola. Sketch how the optimal point moves along it as $C$ increases for (i) $\alpha < \beta$ and (ii) $\alpha = \beta$.

---

### Problem 4: The Factor-of-Six Approximation

*The approximation $C \approx 6ND$ is the linchpin connecting the abstract loss model to the physical cost of compute. This problem derives it from first principles by counting FLOPs in each phase of a training step.*

> **Prerequisites:** cf. note [[note#The Compute Approximation|§2 — The Compute Approximation]]

(a) Count the FLOPs for the forward pass through a single linear layer $y = Wx$ with $W \in \mathbb{R}^{m \times n}$. Express the answer in terms of the number of parameters $mn$.

(b) Derive the FLOP count for the backward pass: compute $\partial\mathcal{L}/\partial x = W^\top (\partial\mathcal{L}/\partial y)$ and $\partial\mathcal{L}/\partial W = (\partial\mathcal{L}/\partial y) x^\top$ separately, expressing each in terms of $mn$.

(c) Sum the three contributions (forward, $\nabla_x$, $\nabla_W$) and argue that for a transformer with $N$ non-embedding parameters, the total FLOPs per training token is $6N$. Over $D$ tokens, conclude $C \approx 6ND$.

(d) Identify two regimes where the $6ND$ approximation breaks down (attention's quadratic sequence-length term; embedding-layer costs). Bound the relative error introduced by ignoring the attention term when sequence length $T \ll \sqrt{N/L}$ for a model with $L$ layers.

---

### Problem 5: The Symmetric Exponent Limit

*Setting $\alpha = \beta$ is a special case that recovers clean $C^{1/2}$ scaling regardless of $A$, $B$, and $E$. This problem proves the result and extracts its physical meaning.*

> **Prerequisites:** cf. note [[note#4.2 Analytical Derivation via Lagrangian Optimization|§4.2 — Analytical Derivation via Lagrangian Optimization]]; requires Problem 3

(a) Show that when $\alpha = \beta$, the exponents in $N^* \propto C^{\beta/(\alpha+\beta)}$ and $D^* \propto C^{\alpha/(\alpha+\beta)}$ both equal $1/2$, independent of the common value of $\alpha = \beta$.

(b) Show that when $\alpha = \beta$, the optimality condition $\alpha A \cdot D^\beta = \beta B \cdot N^\alpha$ reduces to $A/N^\alpha = B/D^\beta$ — i.e., the two excess-loss terms are exactly equal at the optimum. Interpret this as a statement about how compute budget should be balanced.

(c) Show that the compute-optimal loss in the two-term model simplifies to $L^*(C) - E = 2(AB)^{1/2} \cdot (6C)^{-\alpha/2}$ when $\alpha = \beta$. What is the effective compute scaling exponent $\alpha_C$ in this case?

(d) Compare $\alpha_C$ to $\alpha_N$ and $\alpha_D$ (all equal to $\alpha$). Is $\alpha_C$ larger or smaller? Give an intuition for why the compute-optimal loss decays faster or slower than either univariate law.

---

### Problem 6: Second-Order Optimality Condition

*The first-order condition $dL/dN = 0$ identifies a critical point, but does not confirm it is a minimum. This problem verifies that the critical point from Problems 1–2 is indeed a minimum by computing the second derivative and signing it.*

> **Prerequisites:** cf. note [[note#3.2 The Compute-Efficient Frontier|§3.2 — The Compute-Efficient Frontier]]; requires Problem 1

(a) Using the substituted form $L(N) = A/N^\alpha + B \cdot 6^\beta \cdot N^\beta / C^\beta$, compute $d^2L/dN^2$ at the critical point $N^*$.

(b) Show that $d^2L/dN^2 \big|_{N^*} > 0$ for all $\alpha, \beta > 0$. Conclude that $N^*$ is a local minimum of $L$ along the IsoFLOP curve.

(c) Show that $L(N) \to \infty$ as $N \to 0^+$ and as $N \to \infty$ (for fixed $C > 0$). Conclude that the local minimum is also a global minimum on $(0, C/6)$.

---

### Problem 7: The IsoFLOP Curve as a Hyperbola

*The constraint $6ND = C$ defines a family of rectangular hyperbolas in the $(N, D)$ plane. This problem establishes the geometry precisely and uses it to characterize the curvature of the constraint near the optimal point.*

> **Prerequisites:** cf. note [[note#4.1 The IsoFLOP Methodology|§4.1 — The IsoFLOP Methodology]]

(a) Show that $6ND = C$ is a rectangular hyperbola in the $(N, D)$ plane with the coordinate axes as asymptotes. Write the equation in standard form using the substitution $u = \log N$, $v = \log D$.

(b) Show that on the log-log scale, the IsoFLOP constraint becomes a straight line with slope $-1$. What is its intercept in terms of $C$?

(c) The IsoFLOP parabola fitting used in Approach 1 fits $\log L$ vs. $\log N$ locally around $N^*(C)$. Argue that the curvature of $L$ in the $\log N$ direction is $(\alpha(\alpha+1)A/(N^*)^{\alpha+2} + \beta(\beta+1)B \cdot 6^\beta \cdot (N^*)^{\beta-2}/C^\beta)$, and show this is positive.

---

### Problem 8: Compute-Optimal Loss Exponent

*Once $N^*$ and $D^*$ are expressed as functions of $C$, one can substitute back into the loss model to obtain $L^*(C)$ as a pure function of compute. This problem derives the resulting exponent $\alpha_C$ in terms of $\alpha$ and $\beta$.*

> **Prerequisites:** cf. note [[note#3.2 The Compute-Efficient Frontier|§3.2 — The Compute-Efficient Frontier]]; requires Problems 1 and 5

(a) Substitute $N^* \propto C^{\beta/(\alpha+\beta)}$ and $D^* \propto C^{\alpha/(\alpha+\beta)}$ into $L = E + A/N^\alpha + B/D^\beta$ (keeping $E$). Show that both the model-capacity and data terms scale as $C^{-\alpha\beta/(\alpha+\beta)}$.

(b) Conclude that the compute-optimal excess loss satisfies $L^*(C) - E \propto C^{-\gamma}$ where $\gamma = \alpha\beta/(\alpha+\beta)$. Identify this as the harmonic mean of $\alpha$ and $\beta$, divided by 2: $\gamma = (\alpha^{-1} + \beta^{-1})^{-1}$.

(c) Compute $\gamma$ for Kaplan's values ($\alpha = 0.076$, $\beta = 0.095$) and for Chinchilla's values ($\alpha = 0.34$, $\beta = 0.28$). Compare with the reported $\alpha_C \approx 0.050$ from Kaplan's paper. Does the formula reproduce this?

(d) Show that $\gamma < \min(\alpha, \beta)$ always. Interpret: the compute-optimal loss decays slower than either univariate law. Why is this geometrically expected from the shape of the IsoFLOP curve?

---

### Problem 9: Sensitivity of the Optimal Ratio to Fitted Constants

*The "20 tokens per parameter" ratio $D^*/N^*$ depends not only on the exponents $\alpha$, $\beta$ but also on the fitted constants $A$, $B$. This problem quantifies how sensitive the ratio is to perturbations in each, guiding intuition about which parameters matter most for the practical prescription.*

> **Prerequisites:** cf. note [[note#4.3 The 20 Tokens Per Parameter Rule|§4.3 — The 20 Tokens Per Parameter Rule]]; requires Problem 3

(a) From the optimality condition $\alpha A \cdot D^\beta = \beta B \cdot N^\alpha$ and the equal-exponent case $\alpha = \beta$, show that $D^*/N^* = (B/A)^{1/\alpha}$.

(b) Compute $\partial \log(D^*/N^*) / \partial \log(B/A)$ and $\partial \log(D^*/N^*) / \partial \log \alpha$. Which sensitivity is larger when $\alpha$ is small?

(c) Suppose $A$ and $B$ are both perturbed by a multiplicative factor $1 + \epsilon$ (i.e., $A \to (1+\epsilon)A$, $B \to (1+\epsilon)B$). Show that the ratio $D^*/N^*$ is unchanged. Interpret: the ratio depends only on $B/A$, not the overall scale.

(d) Using Chinchilla's Approach 2 constants ($A \approx 406.4$, $B \approx 410.7$, $\alpha \approx 0.34$, $\beta \approx 0.28$), compute $D^*/N^*$ from the optimality condition numerically. How sensitive is this to a $10\%$ change in $\alpha$?

---

### Problem 10: Chinchilla Optimum via the Envelope Theorem

*The envelope theorem provides an alternative route to $\partial L^*/\partial C$ without fully re-solving the optimization. This problem applies it to recover the compute-optimal loss scaling and verify consistency with Problem 8.*

> **Prerequisites:** cf. note [[note#4.2 Analytical Derivation via Lagrangian Optimization|§4.2 — Analytical Derivation via Lagrangian Optimization]]; requires Problems 2 and 8

(a) State the envelope theorem for constrained optimization: if $L^*(C) = \min_{6ND=C} L(N,D)$, then $dL^*/dC = \partial \mathcal{L}/\partial C \big|_{N^*, D^*, \lambda^*}$. Compute $\partial \mathcal{L}/\partial C$ and show it equals $-\lambda^*$.

(b) From Problem 2, express $\lambda^*$ in terms of $N^*$, $D^*$, $\alpha$, $A$. Substitute the power-law expressions $N^* \propto C^{\beta/(\alpha+\beta)}$ to obtain $\lambda^* \propto C^{-(\alpha\beta/(\alpha+\beta)+1)}$.

(c) Integrate $dL^*/dC = -\lambda^*$ to recover $L^*(C) - E \propto C^{-\gamma}$ with $\gamma = \alpha\beta/(\alpha+\beta)$. Verify this matches Problem 8.

(d) What is the economic interpretation of $\lambda^*$ as compute $C$ increases? Does the shadow price increase or decrease with $C$, and why?

---

### Problem 11: Excess Loss Suboptimality Bound

*In practice, one cannot always train at the exact compute-optimal $(N^*, D^*)$. This problem bounds the excess loss incurred by a suboptimal allocation $(N, D)$ on the IsoFLOP curve $6ND = C$, quantifying the cost of miscalibration.*

> **Prerequisites:** cf. note [[note#3.2 The Compute-Efficient Frontier|§3.2 — The Compute-Efficient Frontier]]; requires Problems 1 and 8

(a) For fixed $C$, let $N = r \cdot N^*$ for some $r > 0$ (so $D = D^*/r$). Write $\Delta L(r) = L(N, D) - L^*(C)$ as a function of $r$, $\alpha$, $\beta$, $A$, $B$, and $N^*$, $D^*$.

(b) Show that $\Delta L(r) \geq 0$ for all $r > 0$ (i.e., the optimal allocation indeed minimizes loss). Hint: use the AM-GM inequality or the convexity of the function.

(c) Expand $\Delta L(r)$ around $r = 1$ to second order. Show that $\Delta L(r) \approx \frac{1}{2}L''(N^*) \cdot (N^*)^2 \cdot (\log r)^2$ for $r$ near 1, where $L''(N^*)$ is the second derivative from Problem 6. Interpret the curvature as a measure of how "sharp" the IsoFLOP minimum is.

(d) Under Chinchilla values ($\alpha \approx 0.34$, $\beta \approx 0.28$), by what multiplicative factor can $N$ deviate from $N^*$ before incurring $5\%$ excess loss relative to $L^*(C) - E$?

---

### Problem 12: When Parameters Are More Efficient Than Data

*The Kaplan vs. Chinchilla disagreement can be formalized as a condition on the ratio of marginal loss reductions. This problem derives the condition under which adding a parameter reduces loss more than adding a token, and explains how it relates to the exponent bias.*

> **Prerequisites:** cf. note [[note#4.4 Why Kaplan's Exponents Were Biased|§4.4 — Why Kaplan's Exponents Were Biased]]; requires Problem 2

(a) Define the marginal loss reduction per FLOP from increasing $N$ (holding $D$ fixed) and from increasing $D$ (holding $N$ fixed). Compute each using $C = 6ND$ and the loss model.

(b) Show that parameters are more efficient than data (at a given $(N, D)$) if and only if $\alpha A / N^{\alpha+1} > \beta B / D^{\beta+1}$, equivalently $\alpha A \cdot D^{\beta+1} > \beta B \cdot N^{\alpha+1}$.

(c) At the compute-optimal point, show that this inequality becomes an equality. For $N < N^*$ (holding the IsoFLOP curve), determine whether parameters or data are more efficient.

(d) Kaplan's $\hat{\alpha}_N \approx 0.076$ implies $N^*$ grows much faster than $D^*$. Show that if the true exponents satisfy $\alpha_{\rm true} = \beta_{\rm true} = 0.34$, then Kaplan's biased estimate produces a predicted $N^*/D^*$ ratio that is $\approx 10\times$ larger than the true ratio.

---

### Problem 13: Irreducible Loss and the Bayes Floor

*The term $E$ in $L(N, D) = E + A/N^\alpha + B/D^\beta$ is the irreducible loss — the entropy floor of the data distribution. This problem establishes its role formally and shows that no finite-resource training can achieve loss below $E$.*

> **Prerequisites:** cf. note [[note#The Loss Decomposition Model|§2 — The Loss Decomposition Model]]

(a) Show that $L(N, D) > E$ for all finite $N, D > 0$ under the model. What is $\lim_{N \to \infty, D \to \infty} L(N, D)$?

(b) Suppose we observe two models with losses $L_1$ and $L_2$ and want to estimate $E$ from them. If we assume the model $L = E + A/N^\alpha$ with known $\alpha$, write a closed-form expression for $\hat{E}$ given observations $(N_1, L_1)$ and $(N_2, L_2)$.

(c) Show that if the true loss curve is $L = E + A/N^\alpha$ but we mistakenly assume $E = 0$ (and fit $L = \tilde{A}/N^{\tilde{\alpha}}$), the fitted exponent $\tilde{\alpha}$ will be biased downward relative to $\alpha$. (Hint: consider the log-log slope $d \log L / d \log N$ vs. $d \log(L - E) / d \log N$.)

(d) Relate this to the practical concern that scaling-law extrapolations may systematically overestimate loss reductions at very large $N$ once the true $E$ is not negligible.

---

### Problem 14: Bias in the Kaplan Exponent Estimate

*Kaplan's $\hat{\alpha}_N \approx 0.076$ was measured in a data-saturated regime, producing a systematic downward bias. This problem makes the bias mechanism precise by analyzing the log-log slope of $L(N)$ when $D$ is fixed and finite.*

> **Prerequisites:** cf. note [[note#4.4 Why Kaplan's Exponents Were Biased|§4.4 — Why Kaplan's Exponents Were Biased]]; requires Problem 13

(a) For fixed $D = D_0$, write $L(N) = E + A/N^\alpha + B/D_0^\beta$ (a constant). Compute the log-log slope $s(N) = d\log L / d\log N$ as a function of $N$.

(b) Show that $s(N) \to -\alpha$ as $N \to 0^+$ and $s(N) \to 0$ as $N \to \infty$. Hence the measured slope always underestimates $\alpha$ for large (but finite) $N$.

(c) In the Kaplan experiment, $D = D_0$ is fixed and $N$ varies. Suppose the true $\alpha = 0.34$ and $B/D_0^\beta = 0.5 \cdot A/N_{\max}^\alpha$ (the data term is half the model term at the largest $N$). Compute the log-log slope $s(N_{\max})$ and show it is substantially less than $0.34$.

(d) Propose a correction procedure: given a set of observations $\{(N_i, L_i)\}$ at fixed $D = D_0$ and an independent estimate $\hat{E}$, write an estimator for $\alpha$ that accounts for the additive bias from $E + B/D_0^\beta$.

---

### Problem 15: The 20-Tokens-Per-Parameter Rule

*The "20 tokens per parameter" rule is not a consequence of the exponents alone but depends critically on the magnitude of the fitted constants $A$ and $B$. This problem derives the rule from first principles and characterizes its dependence on the full parameter set.*

> **Prerequisites:** cf. note [[note#4.3 The 20 Tokens Per Parameter Rule|§4.3 — The 20 Tokens Per Parameter Rule]]; requires Problem 3

(a) From the general optimality condition $\alpha A \cdot D^\beta = \beta B \cdot N^\alpha$, solve for $D^*/N^*$ as a function of $\alpha$, $\beta$, $A$, $B$, and $N^*$ (which itself depends on $C$).

(b) When $\alpha \neq \beta$, show that the ratio $D^*/N^*$ is not constant: it depends weakly on $C$ via $N^* \propto C^{\beta/(\alpha+\beta)}$. Express this $C$-dependence explicitly.

(c) Using Chinchilla's Approach 2 fitted values ($A = 406.4$, $B = 410.7$, $\alpha = 0.34$, $\beta = 0.28$), compute $D^*/N^*$ at $C = 10^{21}$ FLOPs (a typical large-model training budget). How close is the result to 20?

(d) Show that if $A = B$ and $\alpha = \beta$, then $D^*/N^* = 1$ exactly. What does this say about the "20" in the Chinchilla rule — is it a consequence of symmetric exponents, or of the asymmetry in $A$ vs. $B$?

---

### Problem 16: Power Laws from a Mixture of Exponentials

*The note mentions a Tauberian theorem argument that power-law scaling emerges from a mixture of exponentials with a power-law distribution of rates. This problem makes that argument precise.*

> **Prerequisites:** cf. note [[note#The Loss Decomposition Model|§2 — The Loss Decomposition Model]]

(a) Let $f(x) = \int_0^\infty e^{-\lambda x} \rho(\lambda) \, d\lambda$ where $\rho(\lambda) \sim C_0 \lambda^{\gamma - 1}$ as $\lambda \to 0^+$ for some $\gamma > 0$. Use the substitution $u = \lambda x$ to rewrite $f(x)$ in a form that separates the $x$-dependence.

(b) Show that as $x \to \infty$, the dominant contribution to $f(x)$ comes from $\lambda \sim 1/x \to 0$. Use the assumed form of $\rho$ to conclude $f(x) \sim C_0 \Gamma(\gamma) x^{-\gamma}$.

(c) Interpret this in the neural scaling context: if "features" of language are learned in order of decreasing frequency $\lambda$ (a feature learned at rate $\lambda$ contributes $e^{-\lambda N}$ to the capacity error when a model has $N$ parameters), and if feature frequencies are distributed as $\rho(\lambda) \propto \lambda^{\gamma-1}$, what scaling exponent $\alpha$ results?

(d) Under what condition on $\rho(\lambda)$ does the integral fail to produce a power law? (Consider $\rho(\lambda) = \delta(\lambda - \lambda_0)$, a single exponential rate.) What does this say about the universality of power-law scaling?

---

### Problem 17: Scaling Exponents and Intrinsic Dimension

*Sharma and Kaplan (2020) showed that the scaling exponent satisfies $\alpha \propto 1/d$ where $d$ is the intrinsic dimension of the data manifold. This problem derives the relationship heuristically and traces its implications.*

> **Prerequisites:** cf. note [[note#The Loss Decomposition Model|§2 — The Loss Decomposition Model]]

(a) Suppose the data lies on a $d$-dimensional manifold and a model with $N$ parameters can represent $\sim N^{1/d}$ "directions" in function space. Argue heuristically that the residual approximation error scales as $N^{-1/d}$, giving $\alpha = 1/d$.

(b) From the table in Section 6, read off the empirical data exponents for word language modeling ($\beta_g \approx 0.066$) and image classification ($\beta_g \approx 0.488$). What effective intrinsic dimensions $d$ do these imply?

(c) Language data is argued to have $d$ much larger than image data. Is this consistent with the exponents above? Give an intuitive explanation grounded in the structure of language vs. images.

(d) If $\alpha = 1/d$ and $\beta = 1/d'$ for two (possibly different) intrinsic dimensions $d$ and $d'$, derive the compute-optimal exponent $\gamma = \alpha\beta/(\alpha+\beta)$ and show that $1/\gamma = 1/\alpha + 1/\beta = d + d'$. Interpret.

---

### Problem 18: Exponential vs. Power-Law Scaling Regimes

*Sorscher et al. (2022) showed that well-structured data can yield exponential rather than power-law loss scaling. This problem formalizes the distinction and derives the crossover condition.*

> **Prerequisites:** cf. note [[note#Non-Power-Law Scaling|§6 — Non-Power-Law Scaling]]

(a) Suppose $L(D) = E + K e^{-\gamma D}$ for constants $K, \gamma > 0$ (exponential scaling). Compute the log-log slope $d\log(L - E)/d\log D$ and show it is not constant — exponential scaling does not appear as a straight line on a log-log plot.

(b) Compare the sample efficiency: how many tokens $D_{\exp}$ does exponential scaling require to achieve $L - E = \epsilon$, vs. how many tokens $D_{\rm pow}$ does power-law scaling $L(D) = E + B/D^\beta$ require? Show $D_{\exp} \propto \log(1/\epsilon)$ while $D_{\rm pow} \propto \epsilon^{-1/\beta}$.

(c) For small $\epsilon$, which regime is more sample-efficient? At what $\epsilon$ do the two expressions cross (for some fixed $K$, $B$, $\gamma$, $\beta$)?

(d) The BNSL model (Caballero et al. 2022) approximates the full loss curve with a product of smooth transitions. Explain qualitatively why such a product form can interpolate between an exponential-like early phase and a power-law middle regime.

---

## Algorithmic Applications

### Problem 19: IsoFLOP Sweep Design

*The IsoFLOP experimental design generates data at fixed compute budgets to identify the compute-optimal $(N^*, D^*)$ at each budget level. This problem specifies the full procedure from budget selection to exponent extraction.*

> **Prerequisites:** cf. note [[note#4.1 The IsoFLOP Methodology|§4.1 — The IsoFLOP Methodology]]

(a) **Inputs and data structures**: Define the inputs to the IsoFLOP sweep procedure. What data structure holds the $(C_i, N_j, D_j, L_{ij})$ observations? How should compute budgets $\{C_i\}$ be spaced (linear, log, or other)?

(b) **Model generation**: Write pseudocode for generating a grid of $(N_j, D_j)$ pairs for a fixed budget $C_i$, sampling $N_j$ log-uniformly in $[N_{\min}, C_i / (6 N_{\min})]$ and setting $D_j = C_i / (6 N_j)$.

(c) **Minimum estimation**: For each $C_i$, fit a degree-2 polynomial to $\{\log N_j, L_{ij}\}$ via OLS to locate $N^*(C_i)$. Write the OLS normal equations for this local fit.

(d) **Exponent extraction**: Given the set of pairs $\{(C_i, N^*(C_i))\}$, write pseudocode for fitting $\log N^* = a \log C + b$ via OLS to estimate $a = \beta/(\alpha+\beta)$. State the complexity of the full procedure.

---

### Problem 20: Log-Linear Regression for Scaling Exponents

*Fitting a scaling exponent from observations $\{(N_i, L_i)\}$ reduces to ordinary least squares in log-log space. This problem specifies the full estimation procedure including confidence intervals.*

> **Prerequisites:** cf. note [[note#3.1 The Univariate Scaling Laws|§3.1 — The Univariate Scaling Laws]]

(a) **Transformed variables**: Define $u_i = \log N_i$ and $v_i = \log(L_i - \hat{E})$ where $\hat{E}$ is a pre-estimated irreducible loss. Write the linear model $v_i = -\alpha u_i + c + \varepsilon_i$ and identify the design matrix $X \in \mathbb{R}^{n \times 2}$.

(b) **Normal equations**: Write the OLS estimator $(\hat{\alpha}, \hat{c}) = (X^\top X)^{-1} X^\top v$ in closed form for the univariate case. Simplify using $\bar{u} = n^{-1}\sum u_i$ and $\bar{v} = n^{-1} \sum v_i$.

(c) **Confidence interval**: Under the assumption $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$, write the $95\%$ confidence interval for $\hat{\alpha}$. What is the standard error of $\hat{\alpha}$ in terms of $\sigma^2$, $n$, and $\text{Var}(u)$?

(d) **Sensitivity to $\hat{E}$**: Describe how errors in $\hat{E}$ propagate into $\hat{\alpha}$. If $\hat{E}$ is too large by $\delta > 0$, does $\hat{\alpha}$ increase or decrease? Sketch a pseudocode procedure that jointly estimates $E$ and $\alpha$ via a 2D grid search.

---

### Problem 21: Parametric Fit via L-BFGS

*Chinchilla's Approach 2 fits all five parameters $(E, A, B, \alpha, \beta)$ simultaneously to all training runs. This problem designs the full optimization procedure.*

> **Prerequisites:** cf. note [[note#5.2 Approach 2: Parametric Global Fit|§5.2 — Approach 2: Parametric Global Fit]]

(a) **Objective function**: Write the sum-of-squared-residuals objective in log-space: reparametrize $E = e^e$, $A = e^a$, $B = e^b$ and define $\theta = (e, a, b, \alpha, \beta)$. Write $\mathcal{F}(\theta) = \sum_i (L_i - e^\epsilon - e^a / N_i^\alpha - e^b / D_i^\beta)^2$.

(b) **Gradient**: Derive $\partial \mathcal{F} / \partial \alpha$ and $\partial \mathcal{F} / \partial e$ symbolically. Write pseudocode for computing the full gradient $\nabla_\theta \mathcal{F}$ over all runs.

(c) **Initialization and restarts**: Write a pseudocode initialization strategy: warm-start $\alpha$ and $\beta$ from an Approach 1 slope estimate; initialize $e$ to $\log(\min_i L_i - \delta)$ for small $\delta > 0$; sample $(a, b)$ from a grid. Describe a multi-restart strategy for escaping local minima.

(d) **Convergence check**: Define a stopping criterion for the L-BFGS loop. How would you detect that the optimizer has converged to a local minimum rather than the global minimum? Sketch a pseudocode validation step using a held-out set of runs.

---

### Problem 22: Compute-Budget Allocation Algorithm

*Given a total compute budget $C$ and fitted scaling law parameters, one can compute the optimal $(N^*, D^*)$ allocation analytically. This problem designs an algorithm that also handles the practically important constrained case where $N \leq N_{\max}$.*

> **Prerequisites:** cf. note [[note#3.2 The Compute-Efficient Frontier|§3.2 — The Compute-Efficient Frontier]]; requires Problems 1 and 6

(a) **Unconstrained allocation**: Write pseudocode for computing $N^*(C)$ and $D^*(C)$ given $(\alpha, \beta, A, B, C)$ using the closed-form expressions from Problem 1. Include the prefactor, not just the exponent.

(b) **Constrained allocation**: Suppose hardware limits $N \leq N_{\max}$. Show that if $N^*(C) \leq N_{\max}$, the unconstrained solution is feasible. If $N^*(C) > N_{\max}$, the constrained optimum is $N = N_{\max}$, $D = C/(6 N_{\max})$. Prove this by showing $L(N)$ is decreasing on $(0, N^*(C))$.

(c) **Loss prediction**: Extend the pseudocode to output not just $(N^*, D^*)$ but also the predicted loss $L^*(C)$ (and its uncertainty, given uncertainty in the fitted parameters).

(d) **Batch allocation**: Write pseudocode for a function that, given a list of compute budgets $[C_1, \ldots, C_k]$ and a constraint $N_{\max}$, returns the optimal allocations and predicted losses for all budgets. State the total complexity.

---

### Problem 23: Per-Model-Size Two-Stage Estimator

*Chinchilla's Approach 3 estimates the scaling parameters in two sequential regressions, trading global optimality for numerical stability. This problem designs the two-stage procedure and characterizes its statistical properties.*

> **Prerequisites:** cf. note [[note#5.3 Approach 3: Per-Model-Size Estimation|§5.3 — Approach 3: Per-Model-Size Estimation]]; requires Problem 20

(a) **Stage 1 inputs and model**: For each fixed model size $N_i$ in a grid $\{N_1, \ldots, N_K\}$, collect observations $\{(D_{ij}, L_{ij})\}_{j=1}^{m_i}$. Write the log-linear model for Stage 1: $\log(L_{ij} - E_{N_i}) = \log B_{N_i} - \beta \log D_{ij} + \varepsilon_{ij}$. Identify the unknowns per model size.

(b) **Stage 1 pseudocode**: Write pseudocode for fitting $(\hat{E}_{N_i}, \hat{\beta}_i)$ for each $N_i$ via a 1D grid search over $E_{N_i} \in [0, \min_j L_{ij})$ followed by OLS on the residuals. How should the grid for $E_{N_i}$ be chosen?

(c) **Stage 2**: Given $\{(N_i, \hat{E}_{N_i})\}_{i=1}^K$, fit $\hat{E}_{N_i} = E + A / N_i^\alpha$ by log-linear regression (treat $\hat{E}_{N_i} - \hat{E}$ as the response, iterate over candidate $\hat{E}$). Write the pseudocode for this stage.

(d) **Comparison with Approach 2**: Identify one statistical advantage and one disadvantage of the two-stage approach vs. the joint L-BFGS fit of Problem 21. Under what data collection design (number of model sizes vs. number of $D$ values per size) does Approach 3 have lower variance than Approach 2?
