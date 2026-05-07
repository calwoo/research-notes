# 📐 Theoretical Foundations of Self-Supervised Vision

*Why collapse is avoided, what representations SSL learns, and the geometry of the hypersphere*

## Table of Contents

- [[#🧮 The Geometry of Collapse|🧮 The Geometry of Collapse]]
- [[#🔂 Stop-Gradient: A Formal Analysis|🔂 Stop-Gradient: A Formal Analysis]]
- [[#🔁 SimSiam as an EM Algorithm|🔁 SimSiam as an EM Algorithm]]
- [[#📦 Batch Normalization as Implicit Contrastive Signal|📦 Batch Normalization as Implicit Contrastive Signal]]
- [[#⚖️ Alignment and Uniformity on the Hypersphere|⚖️ Alignment and Uniformity on the Hypersphere]]
- [[#📉 Dimensional Collapse and Spectral Analysis|📉 Dimensional Collapse and Spectral Analysis]]
- [[#🎯 What Do SSL Representations Converge To?|🎯 What Do SSL Representations Converge To?]]
- [[#📚 References|📚 References]]

> [!INFO] Companion note
> This note provides the theoretical underpinnings for [[concepts/self-supervised-vision/ssl-vision|Self-Supervised Vision: Contrastive Learning and Beyond]], which covers BYOL, SimSiam, Barlow Twins, and VICReg at an applied level. The present note assumes familiarity with those methods.

---

## 🧮 The Geometry of Collapse

### 📐 Eigenvalue Characterization

Let $f_\theta : \mathcal{X} \to \mathbb{R}^d$ be the representation function (encoder + projector). Given a dataset of $N$ images, let $Z \in \mathbb{R}^{N \times d}$ be the empirical embedding matrix. The central object is the *empirical embedding covariance*:

$$\Sigma = \frac{1}{N} Z^\top Z - \bar{z}\bar{z}^\top \in \mathbb{R}^{d \times d}, \quad \bar{z} = \frac{1}{N}\sum_b z_b.$$

**Definition (Collapse).** The representation has *collapsed* if $\mathrm{rank}(\Sigma) < d$. Full collapse has $\mathrm{rank}(\Sigma) = 1$ (all embeddings identical up to a mean shift); *dimensional collapse* has $1 < \mathrm{rank}(\Sigma) \ll d$.

**Proposition 1 (Collapse characterization).** Let $0 \leq \lambda_d \leq \cdots \leq \lambda_1$ be the eigenvalues of $\Sigma$. The embedding has not collapsed if and only if $\lambda_d > 0$, i.e. $\Sigma \succ 0$.

*Proof.* $\mathrm{rank}(\Sigma) = d \Leftrightarrow$ all eigenvalues positive $\Leftrightarrow \lambda_d > 0$. $\square$

### 📏 Effective Rank

The binary rank is too coarse — representations can be *nearly* collapsed without having exact zero eigenvalues. Two useful scalar proxies:

**Stable rank:**
$$\mathrm{srank}(\Sigma) = \frac{\bigl(\sum_i \lambda_i\bigr)^2}{\sum_i \lambda_i^2} = \frac{(\mathrm{tr}\,\Sigma)^2}{\|\Sigma\|_F^2} \in [1, d].$$

At $\mathrm{srank} = d$, all eigenvalues are equal (maximally spread). At $\mathrm{srank} = 1$, only one eigenvalue is nonzero. The stable rank equals $d$ iff $\Sigma = \sigma^2 I$ for some $\sigma > 0$.

**Entropy rank:**
$$\mathrm{erank}(\Sigma) = \exp\!\Bigl(-\sum_i \sigma_i \log \sigma_i\Bigr), \quad \sigma_i = \frac{\lambda_i}{\sum_j \lambda_j}.$$

This is the exponentiated Shannon entropy of the normalized eigenvalue distribution. $\mathrm{erank} = d$ iff all eigenvalues are equal; $\mathrm{erank} = 1$ iff only one eigenvalue is nonzero.

> [!NOTE] Why stable rank?
> The stable rank is more numerically robust than the spectral rank (which counts nonzero eigenvalues) and more interpretable than entropy rank. It can be computed from the Frobenius and trace norms without eigendecomposition. For an SSL researcher, tracking $\mathrm{srank}$ during training reveals whether dimensional collapse is occurring.

### 🔑 Each SSL Method Maintains $\lambda_d > 0$ via a Different Mechanism

| Method | Mechanism that keeps $\lambda_d > 0$ |
|--------|--------------------------------------|
| SimCLR | Negative repulsion explicitly separates embeddings; denominator of NT-Xent creates a pressure on all eigenvalues |
| BYOL | BN forces unit diagonal of $\Sigma$ (preventing scale collapse); EMA + stop-gradient prevent constant solution |
| SimSiam | Predictor creates a non-trivial optimization landscape; constant solution becomes a saddle |
| Barlow Twins | Redundancy-reduction term directly penalizes when any off-diagonal $C_{ij}$ is nonzero — equivalent to penalizing when $\Sigma$ is low-rank |
| VICReg | Variance term directly penalizes when any $\lambda_j < \gamma^2$ |

---

> [!QUESTION] Exercise 1: Stable Rank Under Scaling
> *The stable rank is invariant to isotropic scaling of the embedding.*
>
> > **Prerequisites:** [[#🧮 The Geometry of Collapse|The Geometry of Collapse]]
>
> Let $\Sigma' = \alpha \Sigma$ for scalar $\alpha > 0$. Show that $\mathrm{srank}(\Sigma') = \mathrm{srank}(\Sigma)$. Now suppose $\Sigma = \mathrm{diag}(\lambda_1, \ldots, \lambda_d)$ with $\lambda_1 \gg \lambda_2 = \cdots = \lambda_d = \varepsilon$. Compute $\mathrm{srank}(\Sigma)$ in the limit $\varepsilon \to 0$ and interpret.

> [!TIP]- Solution to Exercise 1
> **Key insight:** Stable rank is scale-invariant and detects effective low-dimensionality even when no eigenvalue is exactly zero.
>
> **Sketch:** $\mathrm{srank}(\alpha\Sigma) = (\mathrm{tr}(\alpha\Sigma))^2 / \|\alpha\Sigma\|_F^2 = \alpha^2 (\mathrm{tr}\,\Sigma)^2 / (\alpha^2 \|\Sigma\|_F^2) = \mathrm{srank}(\Sigma)$. For the diagonal case: $\mathrm{tr}\,\Sigma = \lambda_1 + (d-1)\varepsilon$, $\|\Sigma\|_F^2 = \lambda_1^2 + (d-1)\varepsilon^2$. As $\varepsilon \to 0$: $\mathrm{srank} \to \lambda_1^2/\lambda_1^2 = 1$. *Interpretation:* A representation dominated by one eigenvalue — even with $d-1$ small but nonzero eigenvalues — has stable rank approaching 1, correctly flagging near-full collapse.

---

## 🔂 Stop-Gradient: A Formal Analysis

### 🎯 What Stop-Gradient Does to the Loss Surface

Consider the symmetric naive loss $\mathcal{L}_{\mathrm{sym}} = \mathcal{D}(z_1, z_2)$ where $\mathcal{D}$ is negative cosine similarity and both $z_1, z_2$ are functions of $\theta$. For unit-normalized embeddings, $\mathcal{D}(z_1, z_2) = -z_1^\top z_2 \in [-1, 1]$, minimized at $-1$ when $z_1 = z_2$.

**Every constant** $z_1 = z_2 = \mathbf{c}$ (with $\|\mathbf{c}\|=1$) achieves $\mathcal{L}_{\mathrm{sym}} = -1$ and is a global minimum. The loss surface has a degenerate family of collapsed minima.

Now introduce the stop-gradient on one branch and a predictor $h_\phi$:

$$\mathcal{L}_{\mathrm{sg}} = \frac{1}{2}\mathcal{D}(h_\phi(z_1), \mathrm{sg}(z_2)) + \frac{1}{2}\mathcal{D}(h_\phi(z_2), \mathrm{sg}(z_1)).$$

**Does the constant solution remain a critical point?** At $z_1 = z_2 = \mathbf{c}$ and $h_\phi(\mathbf{c}) = \mathbf{c}$:

$$\nabla_\theta \mathcal{L}_{\mathrm{sg}} = \frac{1}{2}\nabla_{z_1} \mathcal{D}(h_\phi(z_1), \mathbf{c})\Big|_{z_1=\mathbf{c}} \cdot \nabla_\theta z_1 + \frac{1}{2}\nabla_{z_2}\mathcal{D}(h_\phi(z_2), \mathbf{c})\Big|_{z_2=\mathbf{c}} \cdot \nabla_\theta z_2.$$

Since $\nabla_z \mathcal{D}(h_\phi(z), \mathbf{c})|_{z=\mathbf{c},\, h_\phi(\mathbf{c})=\mathbf{c}} = 0$ (gradient of cosine similarity at maximum), **the constant solution is still a critical point**.

**But is it a minimum or a saddle?** The Hessian of $\mathcal{L}_{\mathrm{sg}}$ at collapse has a different structure than without stop-gradient. The key difference: the stop-gradient *breaks symmetry* in the Hessian. Without stop-gradient, all directions in weight space are degenerate (flat manifold of minima). With stop-gradient, only certain directions remain flat — others gain positive curvature, making the collapsed point a *saddle* rather than a minimum.

> [!WARNING] This argument is informal
> *A rigorous proof that the collapsed fixed point is a saddle for SimSiam remains open.* Chen & He (2021) provide the EM interpretation as a plausible mechanism, and empirically demonstrate that collapse does not occur with reasonable initialization. The theoretical question of whether stop-gradient + predictor provably prevents collapse — independently of BN — is still active research.

### 📊 The Predictor as a Bottleneck

Without a predictor, the stop-gradient loss becomes:

$$\frac{1}{2}\mathcal{D}(\mathrm{sg}(z_1), z_2) + \frac{1}{2}\mathcal{D}(\mathrm{sg}(z_2), z_1).$$

The gradient w.r.t. $\theta$ through $z_1$ (from the second term only) is:

$$\frac{1}{2}\nabla_{z_1}\mathcal{D}(\mathrm{sg}(z_2), z_1) \cdot \nabla_\theta z_1.$$

At collapse $z_1 = z_2 = \mathbf{c}$: $\nabla_{z_1}\mathcal{D}(\mathbf{c}, z_1)|_{z_1 = \mathbf{c}} = 0$. **Zero gradient — collapse is stable.**

*Without the predictor, the stop-gradient does not rescue the optimization from collapse.* The predictor introduces an additional parameter $\phi$ that changes the gradient landscape — specifically, the E-step finds the optimal $h_\phi^*$ which is non-trivial when $z_1 \neq z_2$, and the M-step gradient is then non-zero at the initialization before collapse.

---

> [!QUESTION] Exercise 2: Hessian Asymmetry Under Stop-Gradient
> *Stop-gradient changes the curvature structure at the collapsed fixed point.*
>
> > **Prerequisites:** [[#🔂 Stop-Gradient: A Formal Analysis|Stop-Gradient: A Formal Analysis]]
>
> Consider a 1-dimensional toy model: $z_1 = \theta$ and $z_2 = \theta + \varepsilon$ (two branches parameterized by the same scalar $\theta$, with $\varepsilon$ a fixed perturbation). Compare the second derivative $\partial^2 \mathcal{L} / \partial\theta^2$ at $\varepsilon = 0$ for (a) the symmetric loss $\mathcal{L}_\text{sym} = (z_1 - z_2)^2$ and (b) the stop-gradient loss $\mathcal{L}_\text{sg} = (z_1 - \mathrm{sg}(z_2))^2$. Which has positive curvature at the collapsed point $\theta$ such that $z_1 = z_2$?

> [!TIP]- Solution to Exercise 2
> **Key insight:** The symmetric loss has zero curvature at collapse (every constant is a minimum); the stop-gradient loss has positive curvature, making collapse an unstable critical point.
>
> **Sketch:** **(a) Symmetric:** $\mathcal{L}_\text{sym} = (z_1 - z_2)^2 = \varepsilon^2$, a constant in $\theta$. So $\partial^2 \mathcal{L}_\text{sym}/\partial\theta^2 = 0$ — flat landscape, no gradient signal, collapse is stable. **(b) Stop-gradient:** $\mathcal{L}_\text{sg} = (z_1 - \mathrm{sg}(z_2))^2 = (\theta - (\theta + \varepsilon))^2 = \varepsilon^2$ — also a constant in $\theta$! This shows that even with stop-gradient, if $z_1$ and $z_2$ are both linear in $\theta$ (same branch), the gradient is zero. The predictor changes this: $\mathcal{L} = (h(\theta) - \mathrm{sg}(\theta + \varepsilon))^2$ where $h$ is a learnable function. When $h \neq \mathrm{id}$ (non-trivial predictor), $\partial^2 \mathcal{L}/\partial\theta^2 = 2(h'(\theta))^2 > 0$, creating positive curvature.

---

## 🔁 SimSiam as an EM Algorithm

### 📐 Formal Derivation

Consider the SimSiam objective:

$$\mathcal{L}(\theta, \phi) = \frac{1}{2}\mathbb{E}\!\bigl[\mathcal{D}(h_\phi(z_1),\, \mathrm{sg}(z_2))\bigr] + \frac{1}{2}\mathbb{E}\!\bigl[\mathcal{D}(h_\phi(z_2),\, \mathrm{sg}(z_1))\bigr]$$

where $z_i = g_\theta(f_\theta(v_i))$ and $\phi$ parameterizes the predictor independently of $\theta$.

**Claim:** Gradient descent on $(\theta, \phi)$ with stop-gradient is equivalent to the following alternating optimization on $(\theta, h)$ (where $h$ ranges over all measurable functions, not just the parameterized family):

$$h^{(t+1)} = \operatorname{argmin}_h\; \mathbb{E}\!\bigl[\mathcal{D}(h(z_1^{(t)}),\, z_2^{(t)})\bigr] \tag{E-step}$$

$$\theta^{(t+1)} = \theta^{(t)} - \eta\, \nabla_\theta\;\mathbb{E}\!\bigl[\mathcal{D}(h^{(t+1)}(z_1),\, z_2)\bigr]\Big|_{\theta = \theta^{(t)}} \tag{M-step}$$

**E-step solution.** Replacing $\mathcal{D}(p, z) = -p^\top z / (\|p\|\|z\|)$ with its MSE analogue (equivalent for normalized vectors with fixed $\|z\|$):

$$h^*(z_1) = \underset{h}{\mathrm{argmin}}\; \mathbb{E}\!\bigl[\|h(z_1) - z_2\|^2\bigr] = \mathbb{E}[z_2 \mid z_1].$$

The optimal predictor is the *conditional expectation* of the target embedding given the online embedding. For sufficiently augmented images (where $z_1 \neq z_2$), this is a non-trivial function that captures the invariances learned by the encoder.

**M-step interpretation.** With $h^*$ fixed:

$$\nabla_\theta\; \mathbb{E}\!\bigl[\mathcal{D}(h^*(z_1),\, z_2)\bigr] = \nabla_\theta\;\mathbb{E}\!\bigl[-h^*(z_1)^\top z_2\bigr]$$

$= -\nabla_\theta\; \mathbb{E}[\mathbb{E}[z_2 \mid z_1]^\top z_2]$ (substituting $h^*$).

By the tower property: $\mathbb{E}[\mathbb{E}[z_2 \mid z_1]^\top z_2] = \mathbb{E}[\|z_2\|^2 - \mathrm{Var}(z_2 \mid z_1)]$. The M-step therefore *minimizes the conditional variance* $\mathrm{Var}(z_2 \mid z_1)$ with respect to $\theta$. In other words: update the encoder to make the two augmented views of the same image produce more mutually predictable embeddings.

**Why EM?** The stop-gradient in $\mathcal{L}_{\mathrm{SimSiam}}$ prevents $\nabla_\theta z_2$ from entering the gradient — treating $z_2$ as fixed during the encoder update, exactly as the M-step treats $h^*$ as fixed. The predictor gradient update approximates the E-step (finding optimal $h$ given fixed $\theta$). The decoupling between E and M steps is what prevents the trivial collapsed solution from being a stable attractor.

> [!EXAMPLE] Concrete EM trajectory for Gaussian embeddings
> Suppose $z_1, z_2$ are jointly Gaussian with $\mathbb{E}[z] = 0$, $\mathrm{Cov}(z_1) = \mathrm{Cov}(z_2) = I$, and cross-covariance $\mathrm{Cov}(z_1, z_2) = \rho I$ for $\rho \in [0, 1]$ (controlled by augmentation strength).
>
> E-step: $h^*(z_1) = \mathbb{E}[z_2 \mid z_1] = \rho z_1$ (linear, by Gaussian conditioning formula).
>
> M-step loss: $\mathbb{E}[-\rho z_1^\top z_2] = -\rho \cdot \mathbb{E}[z_1^\top z_2] = -\rho^2 d$.
>
> The M-step gradient pushes $\theta$ to increase $\rho$ (make views more correlated). At $\rho = 1$ (zero augmentation), $h^* = \mathrm{id}$ and the system is at a fixed point — but augmented training ($\rho < 1$) creates a non-trivial E-step and productive M-step gradient.

---

> [!QUESTION] Exercise 3: E-Step for Linear Predictors
> *The optimal linear predictor has a closed form via the Gauss–Markov theorem.*
>
> > **Prerequisites:** [[#🔁 SimSiam as an EM Algorithm|SimSiam as an EM Algorithm]]
>
> Suppose $z_1, z_2 \in \mathbb{R}^d$ are jointly distributed with $\mathbb{E}[z_1] = \mathbb{E}[z_2] = 0$, $\mathbb{E}[z_1 z_1^\top] = \Sigma_{11}$, and $\mathbb{E}[z_2 z_1^\top] = \Sigma_{21}$. Find the optimal linear predictor $h^*(z_1) = A z_1$ by minimizing $\mathbb{E}[\|Az_1 - z_2\|^2]$ over $A \in \mathbb{R}^{d \times d}$. Show that $A^* = \Sigma_{21}\Sigma_{11}^{-1}$ and interpret $A^*$ when $z_1 = z_2$ (perfect augmentation invariance) and when $z_1 \perp z_2$ (completely independent views).

> [!TIP]- Solution to Exercise 3
> **Key insight:** The optimal linear predictor is the least-squares regression matrix; it equals the identity at perfect invariance and zero at complete independence.
>
> **Sketch:** Expand $\mathbb{E}[\|Az_1 - z_2\|^2] = \mathrm{tr}(A\Sigma_{11}A^\top - 2A\Sigma_{12} + \Sigma_{22})$. Taking the matrix gradient and setting to zero: $2A\Sigma_{11} - 2\Sigma_{21} = 0$, giving $A^* = \Sigma_{21}\Sigma_{11}^{-1}$. **Perfect invariance** ($z_1 = z_2$): $\Sigma_{21} = \Sigma_{11}$, so $A^* = I$ — the predictor is the identity, the E-step is trivial, and the M-step has zero gradient (nothing to learn). **Independent views** ($\Sigma_{21} = 0$): $A^* = 0$ — the predictor is the zero map, meaning $h^*(z_1) = 0$, and the M-step tries to make $z_2 = 0$ (collapse). *This shows SSL fails when augmentations are so strong that views share no information.*

---

## 📦 Batch Normalization as Implicit Contrastive Signal

### 🔑 What BN Actually Does

Batch normalization applied to embedding dimension $j$ across a batch $\{z_{b,j}\}_{b=1}^N$:

$$\hat{z}_{b,j} = \gamma_j \cdot \frac{z_{b,j} - \mu_j}{\sigma_j} + \beta_j, \quad \mu_j = \frac{1}{N}\sum_b z_{b,j}, \quad \sigma_j = \sqrt{\frac{1}{N}\sum_b (z_{b,j} - \mu_j)^2}.$$

Three immediate consequences:

1. **Forced variance:** $\mathrm{Var}_b(\hat{z}^j) = \gamma_j^2$ exactly. Each dimension is forced to have nonzero variance — all $N$ batch samples cannot map to the same value in dimension $j$.

2. **Cross-sample coupling:** $\partial \hat{z}_{b,j}/\partial z_{b',j} \neq 0$ for $b \neq b'$ (because $\mu_j$ and $\sigma_j$ depend on all samples). The gradient of the loss for sample $b$ depends on the embeddings of all other samples — an *implicit* cross-sample interaction.

3. **Degenerate behavior at collapse:** If $z_{b,j} = c_j$ for all $b$ (dimension $j$ collapses), then $\sigma_j = 0$ and BN is undefined. In practice, the $\varepsilon$ stabilizer gives $\hat{z}_{b,j} = \gamma_j (c_j - c_j)/\sqrt{\varepsilon} + \beta_j = \beta_j$ — all samples get output $\beta_j$, a constant independent of the input. The BN gradient w.r.t. $z_{b,j}$ at this point is $\gamma_j / \sqrt{\varepsilon} \cdot (1 - 1/N)$, which is very large (since $\varepsilon \ll 1$). This large gradient signal strongly pushes $z_{b,j}$ away from the constant solution.

**Formal statement:** BN makes the collapsed solution $z_{b,j} = c_j$ an *unstable critical point* by creating a large gradient pointing away from it. The magnitude of this gradient is $O(1/\sqrt{\varepsilon})$, which is typically $O(10^2)$ for standard $\varepsilon = 10^{-5}$.

### 📉 BN's Role in BYOL

Richemond et al. (2020) showed that BYOL can avoid collapse without BN if certain alternative normalizations are used. This suggests BN is not uniquely required — rather, it instantiates a *general principle*:

**The diversity constraint:** Any normalization that prevents all $N$ batch samples from mapping to the same embedding will suffice. BN enforces this exactly (via forced variance). Other options include:
- Layer normalization (normalizes over features, not batch — weaker diversity constraint)
- Group normalization (intermediate)
- Spectral normalization (bounds Lipschitz constant, prevents explosive collapse but not subtle dimensional collapse)
- Explicit diversity loss (VICReg's variance term does this without BN)

The insight that BN = implicit diversity constraint explains why VICReg succeeds without BN: its explicit variance regularizer $v(Z)$ enforces the same property directly.

---

> [!QUESTION] Exercise 4: BN Gradient at Near-Collapse
> *Batch normalization creates a strong restoring force when a dimension is near-collapse.*
>
> > **Prerequisites:** [[#📦 Batch Normalization as Implicit Contrastive Signal|Batch Normalization as Implicit Contrastive Signal]]
>
> Let $z_{b,j} = c + \delta_b$ where $\delta_b$ is a small zero-mean perturbation with $\mathrm{Var}(\delta_b) = \sigma^2 \ll 1$. Compute $\partial \hat{z}_{b,j}/\partial z_{b,j}$ to leading order in $\sigma$, and show it scales as $O(1/\sigma)$. Interpret: what does a very large BN gradient in the near-collapse regime imply about the implicit regularization effect of BN?

> [!TIP]- Solution to Exercise 4
> **Key insight:** BN acts as a spring with stiffness $O(1/\sigma)$ — near collapse, the restoring force grows without bound, making collapse dynamically unreachable.
>
> **Sketch:** With $z_{b,j} = c + \delta_b$: $\mu_j = c$, $\sigma_j = \sqrt{(1/N)\sum_b \delta_b^2} \approx \sigma$. So $\hat{z}_{b,j} = \gamma_j \delta_b / \sigma + \beta_j$. Then $\partial \hat{z}_{b,j}/\partial z_{b,j} = \gamma_j(1 - 1/N)/\sigma \approx \gamma_j/\sigma$ for large $N$. As $\sigma \to 0$ (approaching collapse), this derivative diverges as $O(1/\sigma)$. The chain rule then gives $\partial \mathcal{L}/\partial z_{b,j} \sim (\gamma_j/\sigma) \cdot \partial\mathcal{L}/\partial\hat{z}_{b,j}$, creating an increasingly large gradient pointing away from $\sigma = 0$. This is the dynamic origin of BN's collapse prevention: it becomes infinitely stiff near the collapsed manifold.

---

## ⚖️ Alignment and Uniformity on the Hypersphere

Wang & Isola (2020) provide a geometric decomposition of contrastive representation learning that unifies all SSL methods.

### 📐 The Decomposition

Restrict embeddings to the unit hypersphere: $z \in \mathbb{S}^{d-1}$. Define two geometric properties:

**Alignment** (closeness of positive pairs):
$$\mathcal{L}_{\mathrm{align}} = \mathbb{E}_{x,\, t \sim \mathcal{T}}\!\bigl[\|f(t(x)) - f(t'(x))\|^2\bigr].$$

**Uniformity** (spread of representations on $\mathbb{S}^{d-1}$):
$$\mathcal{L}_{\mathrm{uniform}} = \log\, \mathbb{E}_{x, y \sim p_{\mathrm{data}}}\!\bigl[e^{-2\|f(x) - f(y)\|^2}\bigr].$$

The uniformity loss is the log of the average Gaussian kernel between all pairs — minimizing it (making it more negative) means the distribution of $f(x)$ on $\mathbb{S}^{d-1}$ becomes more uniform.

**Theorem (Wang & Isola, 2020).** In the large-batch limit ($N \to \infty$), the NT-Xent loss decomposes as:

$$\mathcal{L}_{\mathrm{NT\text{-}Xent}} = \mathcal{L}_{\mathrm{align}} - \mathcal{L}_{\mathrm{uniform}} + C(\tau)$$

where $C(\tau)$ is a temperature-dependent constant. *Minimizing NT-Xent simultaneously minimizes alignment and maximizes uniformity.*

**Corollary (Collapse prevention).** The uniformity loss $\mathcal{L}_{\mathrm{uniform}} = -\infty$ if and only if all embeddings are identical (the Gaussian kernel $e^{-2\|f(x)-f(y)\|^2}$ equals $1$ for all pairs, so the expectation is $1$ and the log is $0$ — actually, $\mathcal{L}_{\mathrm{uniform}} = 0$ at collapse, which is the maximum, not minimum). Minimizing $\mathcal{L}_{\mathrm{uniform}}$ (making it more negative) pushes embeddings apart — exactly preventing collapse.

> [!WARNING] Sign convention
> *Wang & Isola define the uniformity loss to be minimized, so $\mathcal{L}_{\mathrm{uniform}}$ is maximally negative when embeddings are perfectly uniform on the sphere. At collapse, $\mathcal{L}_{\mathrm{uniform}} = 0$ (all pairs have distance 0, kernel = 1, log = 0). Thus collapse corresponds to the* maximum *of $\mathcal{L}_{\mathrm{uniform}}$ — minimizing it is exactly collapse prevention.*

### 🗺️ All SSL Methods as Alignment + Uniformity

The Wang–Isola lens reveals that every SSL method optimizes the same two objectives, via different mechanisms:

| Method | Alignment mechanism | Uniformity mechanism |
|--------|---------------------|----------------------|
| SimCLR | Numerator of NT-Xent: positive pairs close | Denominator: negatives repelled |
| BYOL | Prediction loss: $\|h(z) - z'\|^2$ | BN (implicit) |
| SimSiam | Prediction loss: $\mathcal{D}(h(z_1), z_2)$ | BN + predictor dynamics |
| Barlow Twins | Invariance term: diagonal $C_{ii} \to 1$ | Redundancy-reduction: off-diagonal $C_{ij} \to 0$ |
| VICReg | Invariance term: $s(Z, Z')$ | Variance + covariance terms |

*The contrastive vs. non-contrastive divide is not about objectives — both camps optimize alignment and uniformity. The divide is about implementation: contrastive methods enforce uniformity explicitly (via negatives); non-contrastive methods enforce it implicitly or via distributional constraints.*

---

> [!QUESTION] Exercise 5: Uniformity and Entropy
> *Maximizing uniformity on $\mathbb{S}^{d-1}$ is related to maximizing differential entropy.*
>
> > **Prerequisites:** [[#⚖️ Alignment and Uniformity on the Hypersphere|Alignment and Uniformity on the Hypersphere]]
>
> The maximum-entropy distribution on $\mathbb{S}^{d-1}$ is the uniform distribution. Show that the uniformity loss $\mathcal{L}_{\mathrm{uniform}} = \log \mathbb{E}_{x,y}[e^{-2\|f(x)-f(y)\|^2}]$ is minimized (made most negative) by the uniform distribution on $\mathbb{S}^{d-1}$, in the sense that among distributions with a fixed covariance trace, the uniform distribution minimizes $\mathbb{E}[e^{-2\|f(x)-f(y)\|^2}]$.

> [!TIP]- Solution to Exercise 5
> **Key insight:** The Gaussian kernel $e^{-2\|u-v\|^2}$ is maximized when $u = v$, so its expectation is minimized when mass is as spread out as possible — i.e. when the distribution is uniform on the sphere.
>
> **Sketch:** For $u, v \in \mathbb{S}^{d-1}$: $\|u-v\|^2 = 2 - 2u^\top v$, so $e^{-2\|u-v\|^2} = e^{-4}e^{4u^\top v}$. Thus $\mathbb{E}[e^{-2\|u-v\|^2}] = e^{-4}\mathbb{E}[e^{4u^\top v}]$. The function $u \mapsto e^{4u^\top v}$ is convex and maximized at $u = v$. By Jensen's inequality applied to the distribution of $u$, $\mathbb{E}_u[e^{4u^\top v}]$ is minimized when the distribution of $u$ has maximum spread — which is the uniform distribution on $\mathbb{S}^{d-1}$. Hence the uniformity loss is minimized by the uniform distribution, confirming that minimizing $\mathcal{L}_{\mathrm{uniform}}$ drives embeddings toward uniform coverage of the hypersphere.

---

## 📉 Dimensional Collapse and Spectral Analysis

Full collapse (all embeddings identical) is the most visible failure mode, but *dimensional collapse* — where the representation spans only a low-dimensional subspace of $\mathbb{R}^d$ — is more insidious and harder to detect.

### 📐 Formal Definition

**Definition (Dimensional collapse).** The representation has $k$-dimensional collapse if $\mathrm{rank}(\Sigma) = k \ll d$. The effective rank $\mathrm{srank}(\Sigma) \approx k$.

### 🔑 Why Dimensional Collapse Happens

**In contrastive learning (SimCLR, MoCo):** The NT-Xent loss with batch size $N$ requires only enough embedding dimensions to separate $N$ negative pairs. Asymptotically, only $O(\log N)$ dimensions are needed to achieve near-zero NT-Xent loss — the remaining $d - O(\log N)$ dimensions provide no gradient signal and shrink toward zero. This is the *dimensional collapse* induced by finite batches.

**Formal argument (Jing et al., 2022):** Consider a rank-$k$ encoder $f_\theta(x) = U_k g(x)$ where $U_k \in \mathbb{R}^{d \times k}$ has orthonormal columns and $g : \mathcal{X} \to \mathbb{R}^k$. If $k \geq O(\log N)$, this encoder achieves NT-Xent loss close to the global minimum. The loss landscape provides no gradient signal to increase $k$, so gradient descent may converge to a low-rank solution.

**In non-contrastive learning:** Dimensional collapse can occur when the invariance term dominates the regularization terms. If $\lambda$ is too large (VICReg) or $\lambda$ too small (Barlow Twins), the encoder is pushed to make $z \approx z'$ but the anti-collapse mechanism is too weak to maintain full rank.

### 📊 Measuring Dimensional Collapse in Practice

Three metrics are commonly used:

**1. Log of eigenvalue ratio:** $\log(\lambda_1 / \lambda_d)$ — large value indicates collapse.

**2. Stable rank:** $\mathrm{srank}(\Sigma) = (\mathrm{tr}\,\Sigma)^2 / \|\Sigma\|_F^2$ — tracks effective dimensionality during training.

**3. Cumulative explained variance:** Plot $\sum_{i=1}^k \lambda_i / \sum_{i=1}^d \lambda_i$ vs. $k$ — rapid saturation (90% of variance in $k \ll d$ dimensions) signals dimensional collapse.

> [!NOTE] The redundancy-reduction methods explicitly prevent dimensional collapse
> Barlow Twins and VICReg both have terms that directly penalize low-dimensional representations. The off-diagonal terms of Barlow Twins ($\sum_{i \neq j} C_{ij}^2$) are zero only when all dimensions are mutually uncorrelated — impossible in a low-rank representation (some dimensions would be linear combinations of others, hence correlated). Similarly, VICReg's covariance term $c(Z)$ penalizes correlated dimensions. *These are direct anti-dimensional-collapse regularizers.*

---

> [!QUESTION] Exercise 6: Dimensional Collapse and the NT-Xent Loss
> *The NT-Xent loss admits low-rank solutions when the batch size is small.*
>
> > **Prerequisites:** [[#📉 Dimensional Collapse and Spectral Analysis|Dimensional Collapse and Spectral Analysis]]
>
> Let $z_1, \ldots, z_N \in \mathbb{S}^{d-1}$ be $N$ unit-norm embeddings. Show that if these $N$ points are at maximum pairwise angular distance (i.e. form an equiangular tight frame or simplex ETF), the NT-Xent loss is minimized. Now argue that such a configuration can be achieved with embeddings lying in a $(N-1)$-dimensional affine subspace of $\mathbb{R}^d$, regardless of $d$. What does this imply about dimensional collapse when $d \gg N$?

> [!TIP]- Solution to Exercise 6
> **Key insight:** The NT-Xent loss is minimized when all pairwise similarities are as negative as possible — achievable in just $N-1$ dimensions, making dimensional collapse to rank $N-1$ a global minimizer.
>
> **Sketch:** NT-Xent for anchor $z_k$ is minimized when $\mathrm{sim}(z_k, z_k')$ is maximized and all $\mathrm{sim}(z_k, z_m)$ for $m \neq k$ are minimized. The maximum entropy arrangement of $N$ points on $\mathbb{S}^{d-1}$ that maximizes minimum pairwise distance is the *regular simplex* (simplex ETF): $N$ points with equal pairwise inner products $z_i^\top z_j = -1/(N-1)$ for $i \neq j$. A regular $N$-simplex in $\mathbb{R}^d$ lies on an $(N-1)$-dimensional affine subspace (the convex hull of $N$ vertices is at most $(N-1)$-dimensional). So for $d \gg N$, the global NT-Xent minimizer uses only $N-1 \ll d$ embedding dimensions — *dimensional collapse to rank $N-1$ is built into the loss landscape*. Increasing batch size is not just about more negatives — it directly increases the minimum rank of the solution.

---

## 🎯 What Do SSL Representations Converge To?

### 📐 The Linear Regime Analysis (Tian et al., 2021)

Tian et al. analyze a linearized version of BYOL/SimSiam where both the encoder and predictor are linear: $z = Wx$ and $h(z) = Az$.

**Setup.** Data $x \in \mathbb{R}^n$; linear encoder $W \in \mathbb{R}^{d \times n}$; linear predictor $A \in \mathbb{R}^{d \times d}$. Two augmented views are generated by additive noise: $x_1 = x + \eta_1$, $x_2 = x + \eta_2$ where $\eta_i \sim \mathcal{N}(0, \sigma^2 I)$.

**Key object.** Define the *augmentation-averaged covariance*:

$$F = \mathbb{E}_{t, t'}[f(t(x)) f(t'(x))^\top] \in \mathbb{R}^{n \times n}$$

For the additive-noise model: $F = \mathbb{E}[x_1 x_2^\top] = \Sigma_x$ (the data covariance, since the noise is uncorrelated across views).

**Theorem (Tian et al., informal).** In the linear regime, gradient descent on the SimSiam/BYOL objective converges (under mild conditions) to a point where the row space of $W$ spans the top-$d$ eigenvectors of $F$. That is, the encoder learns the *principal components* of $F$.

**Interpretation.** The representation $z = Wx$ converges to the projection onto the subspace of maximum variance in $F$ — the same representation learned by PCA on $F$. For the additive-noise model, $F = \Sigma_x$ so this is PCA of the data covariance. With stronger data augmentations, $F$ encodes only the information that is *consistent across augmented views*, filtering out augmentation-specific nuisance. **SSL thus learns the PCA of augmentation-invariant information.**

> [!NOTE] Nonlinear regime
> The linear analysis is a proxy — real encoders are deep nonlinear networks. The theorem does not directly extend, but it provides a useful conceptual model: SSL finds the low-dimensional subspace of $\mathcal{X}$ that captures the most information consistent across augmented views. The choice of augmentation distribution $\mathcal{T}$ determines which information is "invariant" and hence preserved.

### 🔑 Connection to the InfoMax Principle

The Tian et al. result connects SSL to the *infomax principle* (Bell & Sejnowski, 1995): a good representation maximizes the mutual information between the representation and the data, subject to an information bottleneck constraint imposed by the augmentations.

In the linear Gaussian regime:
$$I(z_1; z_2) = \frac{1}{2}\log \frac{\det \Sigma_{z_1}}{\det(\Sigma_{z_1} - \Sigma_{z_1,z_2}\Sigma_{z_2}^{-1}\Sigma_{z_2,z_1})}.$$

Maximizing $I(z_1; z_2)$ over linear encoders with fixed output dimension $d$ yields exactly the top-$d$ eigenvectors of $F$ — the same solution as BYOL/SimSiam in the linear regime. **SSL is (approximately) mutual information maximization between augmented views.**

---

## 📚 References

| Reference Name | Brief Summary | Link |
|---|---|---|
| [Wang & Isola 2020] "Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere" | Decomposes contrastive loss into alignment + uniformity; geometric framework unifying all SSL methods | [arXiv:2005.10242](https://arxiv.org/abs/2005.10242) |
| [Tian et al. 2021] "Understanding Self-supervised Learning Dynamics without Contrastive Pairs" | Linear-regime analysis proving BYOL/SimSiam converge to top eigenvectors of augmentation-averaged covariance | [arXiv:2102.06810](https://arxiv.org/abs/2102.06810) |
| [Chen & He 2021] "Exploring Simple Siamese Representation Learning" | Introduces SimSiam; EM interpretation of stop-gradient + predictor; proves EMA is not required | [arXiv:2011.10566](https://arxiv.org/abs/2011.10566) |
| [Jing et al. 2022] "Understanding Dimensional Collapse in Contrastive Self-supervised Learning" | Formal analysis of why contrastive methods collapse to low-rank representations; spectral mitigation strategies | [arXiv:2110.09348](https://arxiv.org/abs/2110.09348) |
| [Richemond et al. 2020] "BYOL works even without batch statistics" | Shows BYOL avoids collapse without BN when other normalizations are used; identifies diversity constraint as the key | [arXiv:2010.10241](https://arxiv.org/abs/2010.10241) |
| [Grill et al. 2020] "Bootstrap Your Own Latent" | Original BYOL paper; EMA + stop-gradient + predictor; ablation showing BN is load-bearing | [arXiv:2006.07733](https://arxiv.org/abs/2006.07733) |
| [Bell & Sejnowski 1995] "An Information-Maximization Approach to Blind Separation and Blind Deconvolution" | Original infomax principle; maximum mutual information as a representation learning objective | [Neural Computation](https://doi.org/10.1162/neco.1995.7.6.1129) |
