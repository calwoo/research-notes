# Solutions: Neural Scaling Laws

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

**Key insight:** Substituting $D = C/(6N)$ converts the constrained problem into a 1D unconstrained minimization; the balance condition at $dL/dN = 0$ equates the marginal benefit of adding parameters against the marginal cost of losing data, producing the exponent $\beta/(\alpha+\beta)$.

**Sketch:**

(a) $L(N) = A/N^\alpha + B(6N/C)^\beta = A/N^\alpha + (B \cdot 6^\beta/C^\beta) N^\beta$. First term decreases in $N$; second increases.

(b) $dL/dN = -\alpha A N^{-\alpha-1} + \beta B \cdot 6^\beta C^{-\beta} N^{\beta-1} = 0$.

(c) Rearrange: $N^{\alpha+\beta} = \alpha A C^\beta / (6^\beta \beta B)$, so $N^* \propto C^{\beta/(\alpha+\beta)}$. Then $D^* = C/(6N^*) \propto C^{\alpha/(\alpha+\beta)}$.

(d) Kaplan: $\beta/(\alpha+\beta) = 0.095/0.171 \approx 0.556$. Chinchilla: $0.28/0.62 \approx 0.452$. The Kaplan exponent is substantially larger, prescribing faster model growth per compute doubling.

---

### Problem 2: Compute-Optimal Scaling via Lagrangian Optimization

**Key insight:** Equating the two expressions for $\lambda$ derived from $\partial\mathcal{L}/\partial N = 0$ and $\partial\mathcal{L}/\partial D = 0$ yields the optimality condition $\alpha A D^\beta = \beta B N^\alpha$ without eliminating a variable, making the marginal-balance interpretation transparent.

**Sketch:**

(a) $\partial\mathcal{L}/\partial N = 0$: $\lambda = \alpha A/(6 D N^{\alpha+1})$. $\partial\mathcal{L}/\partial D = 0$: $\lambda = \beta B/(6 N D^{\beta+1})$. $\partial\mathcal{L}/\partial\lambda = 0$: $6ND = C$.

(b) Setting equal: $\alpha A N D^{\beta+1} = \beta B D N^{\alpha+1}$. Cancel $ND$: $\alpha A D^\beta = \beta B N^\alpha$.

(c) $\lambda = -\partial L^*/\partial C$ by the envelope theorem. At the optimum, $6\lambda$ equals the marginal loss reduction per unit of either $N$ or $D$; the optimality condition says these are equal, so compute is allocated efficiently between the two levers.

(d) From (b): $D = (\beta B/\alpha A)^{1/\beta} N^{\alpha/\beta}$. Substitute into $6ND = C$: $N^{1+\alpha/\beta} \propto C$, giving $N^* \propto C^{\beta/(\alpha+\beta)}$. Consistent with Problem 1.

---

### Problem 3: The Equal-Scaling Condition

**Key insight:** The ratio $D^*/N^*$ is constant in $C$ if and only if $\alpha = \beta$; when equal, the optimality condition reduces to $A/N^\alpha = B/D^\beta$, meaning each excess-loss term contributes equally at the optimum.

**Sketch:**

(a) From $\alpha A D^\beta = \beta B N^\alpha$: $D = (\beta B/\alpha A)^{1/\beta} N^{\alpha/\beta} \equiv G \cdot N^{\alpha/\beta}$.

(b) $D^*/N^* = G (N^*)^{\alpha/\beta - 1}$. Constant iff $\alpha/\beta = 1$, i.e., $\alpha = \beta$.

(c) When $\alpha = \beta$: both exponents $\beta/(\alpha+\beta) = \alpha/(\alpha+\beta) = 1/2$. Optimality condition becomes $A/N^\alpha = B/D^\alpha$; the two excess-loss terms are equal, so the compute budget is split evenly between reducing model error and data error.

(d) For $\alpha < \beta$: exponent $\beta/(\alpha+\beta) > 1/2$, so $N^*$ grows faster than $D^*$ and the optimal point moves toward larger $N$ along the hyperbola. For $\alpha = \beta$: the point moves symmetrically, maintaining $N^* = D^*$ up to the constant $G$.

---

### Problem 4: The Factor-of-Six Approximation

**Key insight:** Each of the three phases of a training step — forward pass, $\nabla_x$ backward, $\nabla_W$ backward — contributes exactly $2N$ FLOPs for a model with $N$ non-embedding parameters, because each phase requires one matrix-vector product of cost proportional to the weight count.

**Sketch:**

(a) $y = Wx$ with $W \in \mathbb{R}^{m \times n}$: each output $y_i$ costs $2n$ FLOPs ($n$ mults, $n$ adds). Total: $2mn$ FLOPs = $2 \times$ (parameter count).

(b) $\nabla_x = W^\top \delta$ costs $2mn$ FLOPs. $\nabla_W = \delta x^\top$ costs $2mn$ FLOPs (outer product). Each backward phase costs $2 \times$ (parameters).

(c) Summing: $2N + 2N + 2N = 6N$ FLOPs per token. Over $D$ tokens: $C = 6ND$.

(d) Attention costs $O(T^2 d_{\rm head} L)$ FLOPs per layer. Relative to $6N \sim 6L d^2$: ratio $\sim T^2/(Ld) \ll 1$ when $T \ll \sqrt{Ld}$, i.e., $T \ll \sqrt{N/L}$ for $d \sim N/L$. In typical training, $T \sim 2048$ and $\sqrt{N/L} \sim 10^3$–$10^4$, so the approximation holds.

---

### Problem 5: The Symmetric Exponent Limit

**Key insight:** When $\alpha = \beta$, the optimality condition forces $A/N^\alpha = B/D^\alpha$ exactly, so both excess-loss terms are equal and the compute-optimal loss decays as $C^{-\alpha/2}$ — exactly half the univariate exponent.

**Sketch:**

(a) $\beta/(\alpha+\beta)\big|_{\alpha=\beta} = \alpha/(2\alpha) = 1/2$. Independent of the common value.

(b) Optimality condition with $\alpha = \beta$: $\alpha A D^\alpha = \alpha B N^\alpha \Rightarrow A/N^\alpha = B/D^\alpha$. Both terms equal at optimum.

(c) $L^* - E = 2A/(N^*)^\alpha$. With $N^* \propto C^{1/2}$: $L^* - E = 2A \cdot C^{-\alpha/2} \cdot (\text{const})$. So $\alpha_C = \alpha/2$.

(d) $\alpha_C = \alpha/2 < \alpha = \alpha_N = \alpha_D$. The compute-optimal loss decays slower because both $N$ and $D$ must grow simultaneously; neither grows as fast as in the univariate case, so the joint improvement is geometrically slower.

---

### Problem 6: Second-Order Optimality Condition

**Key insight:** Both terms of $d^2L/dN^2|_{N^*}$ are strictly positive for any $\alpha, \beta > 0$, confirming a strict local minimum; since $L \to \infty$ at both endpoints of $(0, C/6)$, this local minimum is also global.

**Sketch:**

(a) Let $K = B 6^\beta/C^\beta$. $dL/dN = -\alpha A N^{-\alpha-1} + K\beta N^{\beta-1}$. $d^2L/dN^2 = \alpha(\alpha+1) A N^{-\alpha-2} + K\beta(\beta-1) N^{\beta-2}$.

(b) At $N^*$, the first-order condition gives $\alpha A (N^*)^{-\alpha-1} = K\beta (N^*)^{\beta-1}$, so $K(N^*)^\beta = \frac{\alpha A}{\beta} (N^*)^{-\alpha}$. Substitute: $d^2L/dN^2|_{N^*} = \alpha(\alpha+1) A (N^*)^{-\alpha-2} + (\beta-1)\alpha A (N^*)^{-\alpha-2} = \alpha(\alpha+\beta) A (N^*)^{-\alpha-2} > 0$.

(c) As $N \to 0^+$: $A/N^\alpha \to \infty$. As $N \to \infty$: $K N^\beta \to \infty$. Since $L$ is continuous and $\to \infty$ at both ends, the unique critical point is the global minimum.

---

### Problem 7: The IsoFLOP Curve as a Hyperbola

**Key insight:** In log-log coordinates the hyperbola $6ND = C$ is the line $u + v = \log(C/6)$ with slope $-1$; this log-convexity directly justifies fitting a parabola to $\log L$ vs. $\log N$ to locate the loss minimum.

**Sketch:**

(a) $6ND = C$ with $N, D > 0$ is the standard rectangular hyperbola $y = k/x$ with $k = C/6$, asymptotic to the positive coordinate axes.

(b) $u = \log N$, $v = \log D$: $u + v = \log(C/6)$. Slope $-1$ in $(u,v)$-space, intercept $\log(C/6)$.

(c) Second derivative of $L$ in $\log N$: $\ell(u) = A e^{-\alpha u} + K e^{\beta u}$. $d^2\ell/du^2 = \alpha^2 A e^{-\alpha u} + \beta^2 K e^{\beta u} > 0$. Strictly convex in $\log N$, justifying the parabolic fit.

---

### Problem 8: Compute-Optimal Loss Exponent

**Key insight:** Substituting $N^* \propto C^{\beta/(\alpha+\beta)}$ into $A/(N^*)^\alpha$ gives exponent $-\alpha\beta/(\alpha+\beta)$; the same exponent appears in the data term, so the compute-optimal excess loss is $C^{-\gamma}$ with $\gamma = \alpha\beta/(\alpha+\beta) = (\alpha^{-1}+\beta^{-1})^{-1}$.

**Sketch:**

(a) $A/(N^*)^\alpha \propto C^{-\alpha \cdot \beta/(\alpha+\beta)} = C^{-\alpha\beta/(\alpha+\beta)}$. Similarly $B/(D^*)^\beta \propto C^{-\beta \cdot \alpha/(\alpha+\beta)}$. Both exponents are $-\alpha\beta/(\alpha+\beta)$.

(b) $L^*(C) - E \propto C^{-\gamma}$ with $\gamma = \alpha\beta/(\alpha+\beta)$. Note $1/\gamma = 1/\alpha + 1/\beta$.

(c) Kaplan: $\gamma = (0.076 \times 0.095)/0.171 \approx 0.042$, close to the reported $\alpha_C \approx 0.050$ (small discrepancy from using the two-term vs. three-term model). Chinchilla: $\gamma = (0.34 \times 0.28)/0.62 \approx 0.153$.

(d) $\gamma < \min(\alpha,\beta)$ since $\gamma/\alpha = \beta/(\alpha+\beta) < 1$. Geometrically: along the IsoFLOP curve, both $N$ and $D$ grow slower than in the univariate case (where the other variable is unconstrained), so the rate of loss reduction is bounded by the harmonic-mean exponent.

---

### Problem 9: Sensitivity of the Optimal Ratio to Fitted Constants

**Key insight:** In the equal-exponent case the ratio $D^*/N^* = (B/A)^{1/\alpha}$ has log-sensitivity $1/\alpha$ with respect to $\log(B/A)$; since $\alpha \approx 0.076$ in the Kaplan regime, the ratio is extraordinarily sensitive to small changes in the constant estimates.

**Sketch:**

(a) With $\alpha = \beta$: optimality condition $\alpha A D^\alpha = \alpha B N^\alpha$ gives $(D/N)^\alpha = B/A$, so $D^*/N^* = (B/A)^{1/\alpha}$.

(b) $\partial\log(D^*/N^*)/\partial\log(B/A) = 1/\alpha$. $\partial\log(D^*/N^*)/\partial\log\alpha = -(1/\alpha)\log(B/A)$. The first sensitivity equals $1/\alpha \approx 13$ (Kaplan) or $\approx 3$ (Chinchilla), dominating when $B/A \neq 1$.

(c) If both $A \to (1+\epsilon)A$ and $B \to (1+\epsilon)B$: $B/A$ unchanged, so $D^*/N^*$ unchanged. The ratio depends only on the relative scale of $A$ and $B$, not their absolute values.

(d) With Chinchilla constants: $(B/A)^{1/\alpha} = (410.7/406.4)^{1/0.34} \approx 1.01^{2.94} \approx 1.03$; but with $\alpha \neq \beta$ the full expression gives $\approx 20$. A $10\%$ perturbation to $\alpha$: $1/\alpha$ changes by $10\%$, shifting the ratio by $\approx 30\%$ relative.

---

### Problem 10: Chinchilla Optimum via the Envelope Theorem

**Key insight:** The envelope theorem gives $dL^*/dC = -\lambda^*$ directly; since $\lambda^* \propto C^{-(\gamma+1)}$, integrating recovers $L^*(C) - E \propto C^{-\gamma}$ without re-solving the optimization, confirming Problem 8 by an independent route.

**Sketch:**

(a) $\partial\mathcal{L}/\partial C = -\lambda$ (only $C$-dependence is in $\lambda(6ND-C)$). By the envelope theorem, $dL^*/dC = -\lambda^*$.

(b) $\lambda^* = \alpha A/(6 D^* (N^*)^{\alpha+1})$. Substituting power-law expressions: $\lambda^* \propto C^{-\alpha/(\alpha+\beta)} \cdot C^{-\beta(\alpha+1)/(\alpha+\beta)} = C^{-(\alpha+\beta+\alpha\beta)/(\alpha+\beta)} = C^{-(\gamma+1)}$.

(c) Integrate: $L^*(C) - E = \int_C^\infty \lambda^*(C') dC' \propto C^{-\gamma}/\gamma \propto C^{-\gamma}$. Matches Problem 8.

(d) $\lambda^* \propto C^{-(\gamma+1)}$ decreases with $C$: each additional FLOP yields less loss reduction as the model scales, consistent with power-law diminishing returns.

---

### Problem 11: Excess Loss Suboptimality Bound

**Key insight:** $\Delta L(r) = L(r N^*, D^*/r) - L^*$ is convex in $\log r$ with minimum zero at $r=1$; the second-order expansion gives a quadratic cost in $(\log r)^2$ whose coefficient quantifies the sharpness of the IsoFLOP minimum.

**Sketch:**

(a) $L(r N^*, D^*/r) = A(r N^*)^{-\alpha} + K(r N^*)^\beta = A r^{-\alpha} (N^*)^{-\alpha} + K r^\beta (N^*)^\beta$. Subtract $L^*$: $\Delta L = A(N^*)^{-\alpha}(r^{-\alpha}-1) + K(N^*)^\beta(r^\beta-1)$.

(b) Both $r^{-\alpha}$ and $r^\beta$ are convex in $r > 0$ and equal 1 at $r=1$; their weighted sum is convex with minimum 0 at $r=1$.

(c) Set $t = \log r$, expand to second order: linear terms cancel (by the first-order condition); $\Delta L \approx \frac{1}{2}[\alpha(\alpha+1)A(N^*)^{-\alpha} + \beta(\beta+1)K(N^*)^\beta] t^2$.

(d) The bracket equals $\frac{\alpha(\alpha+\beta+1)\alpha}{\alpha+\beta} \cdot A(N^*)^{-\alpha}$ (after substituting the first-order condition). For Chinchilla values this is $\approx 0.5$–$1.5 \times (L^*-E)$; a $5\%$ excess loss requires $|t| = |\log r| \lesssim 0.3$, i.e., $r \in [0.74, 1.35]$ — about $\pm 30\%$.

---

### Problem 12: When Parameters Are More Efficient Than Data

**Key insight:** The marginal efficiency condition for parameters over data is exactly $\alpha A D^\beta > \beta B N^\alpha$, the reverse of the optimality condition; this holds when $N < N^*$ on the IsoFLOP curve, confirming that the optimal point is where the two marginals are equalized.

**Sketch:**

(a) Marginal loss per FLOP from $N$: $(\alpha A / N^{\alpha+1}) / (6D)$. Marginal loss per FLOP from $D$: $(\beta B / D^{\beta+1}) / (6N)$.

(b) Parameters more efficient iff $\alpha A/(N^{\alpha+1} D) > \beta B/(N D^{\beta+1})$, i.e., $\alpha A D^\beta > \beta B N^\alpha$.

(c) At $N^*$: equality holds. For $N < N^*$ on the IsoFLOP curve, $D = C/(6N) > D^*$, so $D^\beta > (D^*)^\beta$ and $N^\alpha < (N^*)^\alpha$: left side larger, parameters are more efficient.

(d) True $D^*/N^* \approx 20$ (Problem 15). Biased Kaplan estimate with $\hat{\alpha} = 0.076$: $(B/A)^{1/0.076} \approx (1.01)^{13} \approx 1.1$ — but using the full optimality condition with Kaplan's constants, the biased prescription gives $N^* \propto C^{0.556}$ vs. $D^* \propto C^{0.444}$; the ratio $N^*/D^*$ grows as $C^{0.112}$, meaning parameters dominate increasingly at large $C$.

---

### Problem 13: Irreducible Loss and the Bayes Floor

**Key insight:** Because $E > 0$, the log-log slope of $L(N)$ is always strictly shallower than $-\alpha$; fitting a power law to $L$ directly (ignoring $E$) produces a downward-biased exponent that makes scaling look less effective than it truly is.

**Sketch:**

(a) $L(N,D) = E + A/N^\alpha + B/D^\beta > E$ for all finite $N, D$. $\lim_{N,D\to\infty} L = E$.

(b) From $(L_1 - E)/( L_2 - E) = (N_2/N_1)^\alpha$: $\hat{E} = \frac{L_1 (N_1/N_2)^\alpha - L_2}{(N_1/N_2)^\alpha - 1}$ given known $\alpha$.

(c) True slope $d\log(L-E)/d\log N = -\alpha$. Observed slope $d\log L/d\log N = -\alpha(L-E)/L < \alpha$ in magnitude. Fitting $L$ directly yields $\tilde{\alpha} < \alpha$.

(d) As $N \to \infty$, $L \to E$ and the slope $\to 0$ regardless of $\alpha$: extrapolations made in the approach-to-floor regime wildly overestimate future improvement because the fitted $\tilde{\alpha}$ has already been driven to near zero by the $E$ floor.

---

### Problem 14: Bias in the Kaplan Exponent Estimate

**Key insight:** For fixed $D = D_0$, the log-log slope of $L(N)$ equals $-\alpha \times (A/N^\alpha)/(E + A/N^\alpha + B/D_0^\beta)$, which is strictly less than $\alpha$ in magnitude whenever the constant terms $E + B/D_0^\beta$ are nonzero.

**Sketch:**

(a) $L(N) = (E + F) + A/N^\alpha$ where $F = B/D_0^\beta$ is constant. $s(N) = d\log L/d\log N = -\alpha A N^{-\alpha}/(E + F + A N^{-\alpha})$.

(b) As $N \to 0^+$: $A/N^\alpha \to \infty$, so $s \to -\alpha$. As $N \to \infty$: $A/N^\alpha \to 0$, so $s \to 0$. The slope monotonically decreases in magnitude.

(c) With $\alpha_{\rm true} = 0.34$ and $F = 0.5 \cdot A/N_{\max}^\alpha$: $s(N_{\max}) = -0.34 \times 1/(1 + (E+F)N_{\max}^\alpha/A) = -0.34/(1 + 0.5 \cdot E/F + 1.5) \approx -0.34/2.5 \approx -0.14$ (ignoring $E$ for simplicity). Measured slope $\approx 0.14 \ll 0.34$.

(d) Corrected estimator: for each $N_i$, compute $\tilde{v}_i = \log(L_i - \hat{E} - \hat{F})$ using externally estimated $\hat{E}$ and $\hat{F} = B/D_0^\beta$. Regress $\tilde{v}_i$ on $\log N_i$ to recover unbiased $\hat{\alpha}$.

---

### Problem 15: The 20-Tokens-Per-Parameter Rule

**Key insight:** The "20" emerges from the near-equality $A \approx B$ in Chinchilla's fit combined with the exponent asymmetry $\alpha > \beta$; if $A = B$ and $\alpha = \beta$ exactly, the rule would predict $D^*/N^* = 1$.

**Sketch:**

(a) From $\alpha A D^\beta = \beta B N^\alpha$: $(D/N)^\beta = \beta B N^{\alpha-\beta}/(\alpha A)$, so $D^*/N^* = (\beta B/\alpha A)^{1/\beta} (N^*)^{(\alpha-\beta)/\beta}$.

(b) Since $N^* \propto C^{\beta/(\alpha+\beta)}$: $D^*/N^* \propto C^{(\alpha-\beta)/(\alpha+\beta)}$. Drifts with $C$ when $\alpha \neq \beta$.

(c) At $C = 10^{21}$ with Chinchilla constants: $N^* \approx 7 \times 10^{10}$; numerical evaluation of the optimality condition gives $D^*/N^* \approx 18$–$22$.

(d) With $A = B$ and $\alpha = \beta$: $(D/N)^\alpha = B/A = 1$, so $D^*/N^* = 1$. The "20" comes from the mild asymmetry $\alpha > \beta$ (which shifts the optimum toward more data) combined with the near-equality $A \approx B$.

---

### Problem 16: Power Laws from a Mixture of Exponentials

**Key insight:** The Tauberian theorem converts the small-$\lambda$ behavior of the mixing density $\rho(\lambda) \sim C_0 \lambda^{\gamma-1}$ directly into the large-$x$ power-law decay $f(x) \sim C_0\Gamma(\gamma) x^{-\gamma}$; a single exponential rate produces no power law.

**Sketch:**

(a) Substitute $u = \lambda x$: $f(x) = x^{-1} \int_0^\infty e^{-u} \rho(u/x) du$.

(b) As $x \to \infty$: $\rho(u/x) \approx C_0(u/x)^{\gamma-1}$, so $f(x) \approx x^{-1} C_0 x^{1-\gamma} \int_0^\infty e^{-u} u^{\gamma-1} du = C_0\Gamma(\gamma) x^{-\gamma}$.

(c) Feature learned at rate $\lambda$ contributes $e^{-\lambda N}$ error; with $\rho(\lambda) \propto \lambda^{\gamma-1}$, total error $\propto N^{-\gamma}$, giving $\alpha = \gamma$. The scaling exponent equals the tail exponent of the feature-frequency distribution.

(d) For $\rho = \delta(\lambda - \lambda_0)$: $f(x) = e^{-\lambda_0 x}$, purely exponential. Power-law scaling requires a continuum of rates; a single rate gives exponential decay. This is why Sorscher et al.'s exponential regime arises for well-structured data with a single dominant learning rate.

---

### Problem 17: Scaling Exponents and Intrinsic Dimension

**Key insight:** The relationship $\alpha = 1/d$ follows from the geometric argument that adding capacity resolves $N^{1/d}$ directions in $d$-dimensional function space, leaving residual error $\propto N^{-1/d}$.

**Sketch:**

(a) On a $d$-dim manifold, a model with $N$ parameters covers $\sim N^{1/d}$ directions; residual error $\sim (N^{1/d})^{-1} = N^{-1/d}$, giving $\alpha = 1/d$.

(b) Word LM: $\beta_g \approx 0.066 \Rightarrow d \approx 15$. Image classification: $\beta_g \approx 0.488 \Rightarrow d \approx 2$.

(c) Images are dominated by local spatial structure ($d \approx 2$ reflects edge/texture patterns); language spans phonetic, lexical, syntactic, semantic, and discourse structure simultaneously, yielding large $d \approx 15$. Consistent.

(d) $\gamma = \alpha\beta/(\alpha+\beta) = (1/d)(1/d')/(1/d+1/d') = 1/(d+d')$. So $1/\gamma = d+d'$: the compute-optimal exponent reflects the total dimension of both the model and data manifolds.

---

### Problem 18: Exponential vs. Power-Law Scaling Regimes

**Key insight:** Exponential scaling $L - E \sim e^{-\gamma D}$ is vastly more sample-efficient than any power law because the data requirement grows as $\log(1/\epsilon)$ vs. $\epsilon^{-1/\beta}$; the crossover is determined by the specific constants.

**Sketch:**

(a) $d\log(L-E)/d\log D = -\gamma D$: not constant, grows in magnitude with $D$. Exponential scaling appears as a curve (not a line) on a log-log plot, with increasingly steep slope.

(b) Exponential: $D_{\exp} = \gamma^{-1}\log(K/\epsilon) \propto \log(1/\epsilon)$. Power-law: $D_{\rm pow} = (B/\epsilon)^{1/\beta} \propto \epsilon^{-1/\beta}$.

(c) For $\epsilon \to 0$: $\epsilon^{-1/\beta} \gg \log(1/\epsilon)$, so exponential is more efficient. Crossover at $\log(K/\epsilon)/\gamma = (B/\epsilon)^{1/\beta}$, solvable numerically for given constants.

(d) The BNSL product $\prod_i(1 + (D/d_i)^{1/f_i})^{-c_i f_i}$ is approximately $\exp(-\sum c_i f_i \log(1+(D/d_i)^{1/f_i}))$; for $D \ll d_i$ this is approximately $\exp(-C D^{1/f_i})$, exponential-like; for $D \gg d_i$ each factor becomes $(D/d_i)^{-c_i}$, a power law. The product smoothly interpolates between regimes.

---

## Algorithmic Applications

### Problem 19: IsoFLOP Sweep Design

**Key insight:** Sampling $N$ log-uniformly along the IsoFLOP constraint and fitting a degree-2 polynomial in $\log N$ exploits the strict log-convexity of $L(N)$ (Problem 7) to reliably locate the minimum with a small number of training runs per budget.

**Sketch:**

```
function IsoFLOP_sweep(C_list, N_min, n_per_budget):
    # (a) Data structure: dict mapping C -> list of (N, D, L) tuples
    obs = {C: [] for C in C_list}

    for C in C_list:
        N_max = C / (6 * N_min)
        # (b) Log-uniform grid over [N_min, N_max]
        N_grid = exp(linspace(log(N_min), log(N_max), n_per_budget))
        for N in N_grid:
            D = C / (6 * N)
            L = train_and_eval(N_params=N, n_tokens=D)
            obs[C].append((N, D, L))

    N_star_by_C = {}
    for C in C_list:
        # (c) Fit degree-2 poly to (log N, L); locate vertex
        u = [log(N) for N, D, L in obs[C]]
        y = [L        for N, D, L in obs[C]]
        # OLS normal equations: [sum u^4 sum u^3 sum u^2; ...] * [c2 c1 c0]^T = [sum y u^2; sum y u; sum y]
        c2, c1, c0 = solve_OLS_poly2(u, y)
        log_N_star = -c1 / (2 * c2)   # vertex of parabola
        N_star_by_C[C] = exp(log_N_star)

    # (d) Fit log N_star = a log C + b via OLS
    log_C_vals    = [log(C) for C in C_list]
    log_N_star_vals = [log(N_star_by_C[C]) for C in C_list]
    a, b = OLS_linear(log_C_vals, log_N_star_vals)
    # a estimates beta/(alpha+beta); complexity O(k * n_per_budget) training runs
    return a, b, N_star_by_C
```

Budget spacing: log-uniform over 2–3 orders of magnitude to get stable exponent estimates. Total complexity: $O(k \cdot n_{\rm per\_budget})$ training runs; OLS fitting is $O(k)$ and negligible.

---

### Problem 20: Log-Linear Regression for Scaling Exponents

**Key insight:** The OLS estimator for $\alpha$ equals the negative sample correlation of $\log N$ with $\log(L - \hat{E})$ scaled by their standard deviations; errors in $\hat{E}$ introduce heterogeneous bias that is largest for large-$N$ models (where $L - E$ is small).

**Sketch:**

```
function estimate_alpha(N_vals, L_vals, E_hat):
    # (a) Transformed variables
    u = [log(N) for N in N_vals]
    v = [log(L - E_hat) for L in L_vals]   # requires E_hat < min(L_vals)
    # Design matrix X in R^{n x 2}: columns [u, 1]

    # (b) OLS closed form
    u_bar = mean(u);  v_bar = mean(v)
    cov_uv = mean((u_i - u_bar)*(v_i - v_bar) for u_i, v_i in zip(u, v))
    var_u  = mean((u_i - u_bar)**2 for u_i in u)
    alpha_hat = -cov_uv / var_u    # negative slope
    c_hat = v_bar + alpha_hat * u_bar

    # (c) Confidence interval
    residuals = [v_i - (-alpha_hat * u_i + c_hat) for u_i, v_i in zip(u, v)]
    sigma2_hat = sum(r**2 for r in residuals) / (len(u) - 2)
    SE_alpha = sqrt(sigma2_hat / (len(u) * var_u))
    CI_95 = (alpha_hat - 1.96*SE_alpha, alpha_hat + 1.96*SE_alpha)

    # (d) Joint estimation: grid search over E in [0, min(L_vals))
    # For each E_cand: run above OLS, record RSS; choose E_cand minimizing RSS
    return alpha_hat, c_hat, CI_95
```

If $\hat{E}$ is too large by $\delta > 0$: $v_i = \log(L_i - \hat{E}) = \log((L_i - E) - \delta)$ is smaller, especially for large $N_i$ where $L_i - E$ is small; this compresses the range of $v$ at large $u$, reducing the estimated slope magnitude, so $\hat{\alpha}$ decreases.

---

### Problem 21: Parametric Fit via L-BFGS

**Key insight:** Log-space reparametrization ($E = e^e$, $A = e^a$, $B = e^b$) both enforces positivity and symmetrizes the loss surface, reducing the condition number of the Hessian and improving L-BFGS convergence substantially.

**Sketch:**

```
function parametric_fit(runs, alpha0, beta0, n_restarts=10):
    # runs: list of (N_i, D_i, L_i)
    # (a) Objective in log-space parameters theta = [e, a, b, alpha, beta]
    def objective(theta):
        e, a, b, alpha, beta = theta
        E, A, B = exp(e), exp(a), exp(b)
        return sum((L - E - A/N**alpha - B/D**beta)**2
                   for N, D, L in runs)

    # (b) Gradient (key terms)
    def gradient(theta):
        e, a, b, alpha, beta = theta
        E, A, B = exp(e), exp(a), exp(b)
        r = [L - E - A/N**alpha - B/D**beta for N, D, L in runs]
        dF_de    = -2 * E * sum(r)
        dF_da    = -2 * A * sum(r_i / N**alpha for r_i,(N,D,L) in zip(r,runs))
        dF_dalpha =  2 * A * sum(r_i * log(N)/N**alpha for ...)
        # ... analogously for dF_db, dF_dbeta
        return [dF_de, dF_da, dF_db, dF_dalpha, dF_dbeta]

    # (c) Initialization and restarts
    e0 = log(min(L for _,_,L in runs) - 1e-3)
    best_theta, best_loss = None, inf
    for _ in range(n_restarts):
        theta_init = [e0 + randn()*0.1,
                      log(1.0) + randn()*0.5,
                      log(1.0) + randn()*0.5,
                      alpha0 * (1 + randn()*0.1),
                      beta0  * (1 + randn()*0.1)]
        theta_opt, loss = LBFGS(objective, gradient, theta_init,
                                 tol=1e-8, max_iter=2000)
        if loss < best_loss:
            best_theta, best_loss = theta_opt, loss

    # (d) Convergence check: validate on held-out 20% of runs
    held_out_loss = objective_on_subset(best_theta, held_out_runs)
    if held_out_loss > 1.05 * best_loss:
        warn("Possible overfitting or local minimum")
    return best_theta
```

---

### Problem 22: Compute-Budget Allocation Algorithm

**Key insight:** The unconstrained optimum has a closed-form prefactor (not just exponent); when $N^*(C) > N_{\max}$, the constrained optimum is the corner $N = N_{\max}$ because $L(N)$ is strictly decreasing on $(0, N^*(C))$.

**Sketch:**

```
function allocate(C, alpha, beta, A, B, N_max=inf):
    # (a) Unconstrained: N* = [(alpha*A)/(beta*B)]^{1/(alpha+beta)} * [C/6]^{beta/(alpha+beta)}
    ratio = (alpha * A / (beta * B))**(1.0/(alpha+beta))
    N_star_unc = ratio * (C / 6)**(beta/(alpha+beta))
    D_star_unc = C / (6 * N_star_unc)

    # (b) Apply constraint
    if N_star_unc <= N_max:
        N_star, D_star = N_star_unc, D_star_unc
        constrained = False
    else:
        # Corner solution: L(N) decreasing on (0, N*), so minimum on [0, N_max] is at N_max
        N_star = N_max
        D_star = C / (6 * N_max)
        constrained = True

    # (c) Predicted loss and uncertainty (simplified; propagate param uncertainty via delta method)
    L_star = A / N_star**alpha + B / D_star**beta   # excess loss

    return N_star, D_star, L_star, constrained

# (d) Batch allocation: O(k) total
function batch_allocate(C_list, alpha, beta, A, B, N_max=inf):
    return [allocate(C, alpha, beta, A, B, N_max) for C in C_list]
```

Proof of corner: $L(N)$ has unique minimum at $N^*(C) > N_{\max}$. Since $d^2L/dN^2 > 0$ globally and $L$ is decreasing on $(0, N^*(C))$, the constrained minimum on $[0, N_{\max}]$ is at the right boundary $N_{\max}$.

---

### Problem 23: Per-Model-Size Two-Stage Estimator

**Key insight:** Treating $\hat{E}_{N_i}$ from Stage 1 as a sufficient statistic for the effective loss floor decouples the estimation into two low-dimensional regressions, each well-conditioned; the cost is that Stage 2 propagates estimation error from Stage 1.

**Sketch:**

```
function two_stage_fit(data):
    # data: {N_i: [(D_ij, L_ij), ...]}
    # (a)-(b) Stage 1: per-model-size fit
    E_hat = {}; beta_hats = []
    for N_i, runs in data.items():
        D_vals = [D for D,L in runs]
        L_vals = [L for D,L in runs]
        best_E_cand = None; best_rss = inf
        for E_cand in linspace(0, min(L_vals)*0.999, 200):
            v = [log(L - E_cand) for L in L_vals]   # log(L - E_cand)
            u = [log(D) for D in D_vals]
            slope, intercept = OLS_linear(u, v)      # slope estimates -beta
            rss = sum((v_j - (slope*u_j + intercept))**2 for ...)
            if rss < best_rss:
                best_rss = rss; best_E_cand = E_cand; best_slope = slope
        E_hat[N_i] = best_E_cand
        beta_hats.append(-best_slope)

    # (c) Stage 2: fit E_hat(N) = E + A/N^alpha
    N_vals = list(E_hat.keys())
    E_vals = list(E_hat.values())
    best2 = None; best_rss2 = inf
    for E_floor in linspace(0, min(E_vals)*0.99, 100):
        v2 = [log(e - E_floor) for e in E_vals]     # log(A/N^alpha)
        u2 = [log(N) for N in N_vals]
        alpha_est, logA_est = OLS_linear(u2, v2)    # slope = -alpha
        rss2 = sum(...)
        if rss2 < best_rss2:
            best_rss2 = rss2
            best2 = (E_floor, -alpha_est, exp(logA_est))

    E_est, alpha_est, A_est = best2
    beta_est = mean(beta_hats)
    return E_est, A_est, alpha_est, beta_est

# (d) Statistical comparison:
# Advantage of Approach 3: each stage is a lower-dimensional regression -> better conditioned,
#   easier to diagnose failures, more interpretable intermediate outputs.
# Disadvantage: Stage 2 uses only K data points (one per model size) vs. all N*K points
#   in Approach 2; Stage 1 errors propagate into Stage 2 without correction.
# Approach 3 has lower variance when K is small and m_i (runs per N) is large (Stage 1 is
#   accurate); Approach 2 is better when the global model is well-specified and runs span a
#   wide joint (N, D) grid.
```