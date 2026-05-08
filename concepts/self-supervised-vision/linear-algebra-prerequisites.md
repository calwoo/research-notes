# 📐 Linear Algebra Prerequisites for SSL Theory

*A problem set covering the exact linear algebra needed to read the SSL theory note — rank, eigenvalues, covariance matrices, SVD, and matrix calculus*

> [!INFO] How to use this note
> Each section gives a brief definition block, then one or more exercises. Work through the exercises before expanding the solution. The goal is not completeness — it is to build fluency with the specific objects that appear in [[concepts/self-supervised-vision/ssl-theory|SSL Theory]] and [[concepts/self-supervised-vision/ssl-vision|SSL Vision]].

## Table of Contents

- [[#🔢 Rank and the Column Space|🔢 Rank and the Column Space]]
- [[#🔬 Eigenvalues and the Spectral Theorem|🔬 Eigenvalues and the Spectral Theorem]]
- [[#✅ Positive Semidefinite Matrices|✅ Positive Semidefinite Matrices]]
- [[#📊 Covariance Matrices|📊 Covariance Matrices]]
- [[#✂️ Singular Value Decomposition|✂️ Singular Value Decomposition]]
- [[#📏 Trace, Frobenius Norm, and Stable Rank|📏 Trace, Frobenius Norm, and Stable Rank]]
- [[#🔄 Orthogonal Matrices and the Group O(d)|🔄 Orthogonal Matrices and the Group O(d)]]
- [[#🧮 Matrix Calculus|🧮 Matrix Calculus]]
- [[#📚 References|📚 References]]

---

## 🔢 Rank and the Column Space

**Definition.** The *column space* of $A \in \mathbb{R}^{m \times n}$ is $\mathrm{col}(A) = \{Ax : x \in \mathbb{R}^n\} \subseteq \mathbb{R}^m$. The *rank* is $\mathrm{rank}(A) = \dim \mathrm{col}(A)$.

**Key facts.**
- $\mathrm{rank}(A) \leq \min(m, n)$. If equality holds, $A$ is *full rank*.
- $\mathrm{rank}(A) = \mathrm{rank}(A^\top)$ — row rank equals column rank.
- $\mathrm{rank}(AB) \leq \min(\mathrm{rank}(A), \mathrm{rank}(B))$.
- $A \in \mathbb{R}^{n \times n}$ is invertible if and only if $\mathrm{rank}(A) = n$.

**Why it matters for SSL.** In SSL, we form an embedding matrix $Z \in \mathbb{R}^{N \times d}$ (rows = embeddings for $N$ images, columns = $d$ embedding dimensions). *Representational collapse* is exactly $\mathrm{rank}(Z) \ll d$ — the embeddings lie in a low-dimensional subspace and carry limited information.

---

> [!QUESTION] Exercise 1: Rank of an Outer Product
> *The outer product of two vectors is the simplest rank-1 matrix — and the building block of low-rank structure.*
>
> > **Prerequisites:** [[#🔢 Rank and the Column Space|Rank and the Column Space]]
>
> Let $u \in \mathbb{R}^m$, $v \in \mathbb{R}^n$ be nonzero vectors. Show that $A = uv^\top \in \mathbb{R}^{m \times n}$ has rank exactly 1. Then show that if all $N$ embedding vectors are identical — $z_1 = z_2 = \cdots = z_N = c$ — the centered embedding matrix $\tilde{Z} = Z - \bar{z}\mathbf{1}^\top$ has rank 0. What is the rank of the uncentered $Z$ in this case?

> [!TIP]- Solution to Exercise 1
> **Key insight:** Any matrix whose rows all point in the same direction has rank 1; subtracting the mean when all rows are equal gives the zero matrix (rank 0).
>
> **Sketch:** $A = uv^\top$: every column of $A$ is $u \cdot v_i$ (a scalar multiple of $u$), so $\mathrm{col}(A) = \mathrm{span}\{u\}$, which has dimension 1. For the collapse case: $Z = \mathbf{1} c^\top$ (all rows equal $c^\top$). Then $\bar{z} = c$, so $\tilde{Z} = Z - \mathbf{1}c^\top = 0$ — rank 0. The uncentered $Z = \mathbf{1}c^\top$ has rank 1 (since $\mathbf{1}$ and $c$ are nonzero). This is why the SSL literature says "full collapse has $\mathrm{rank}(Z) = 1$": technically the uncentered matrix has rank 1, but there is zero variance in the embeddings — no information is encoded.

---

> [!QUESTION] Exercise 2: Rank and Information Capacity
> *Low rank directly limits how many linearly separable classes a representation can support.*
>
> > **Prerequisites:** [[#🔢 Rank and the Column Space|Rank and the Column Space]]
>
> Suppose a linear classifier uses the embedding $z \in \mathbb{R}^d$ to predict one of $K$ classes: $\hat{y} = \mathrm{argmax}_k\, w_k^\top z$. If the embedding matrix $Z \in \mathbb{R}^{N \times d}$ has rank $r$, what is the maximum number of classes $K$ that can be *linearly separated* by any classifier of this form? Relate this to dimensional collapse in SSL (where $r \ll d$).

> [!TIP]- Solution to Exercise 2
> **Key insight:** A rank-$r$ embedding lives in an $r$-dimensional subspace, so at most $r+1$ classes can be linearly separated — regardless of $d$.
>
> **Sketch:** All embeddings lie in a subspace $V \subseteq \mathbb{R}^d$ with $\dim V = r$. The linear classifier $w_k^\top z$ restricted to $V$ is equivalent to a linear classifier in $\mathbb{R}^r$. By the VC dimension of linear classifiers in $\mathbb{R}^r$, at most $r + 1$ points in general position can be shattered. So at most $r + 1$ classes can be linearly separated. If $r \ll d$, then dimensional collapse wastes most of the embedding capacity — a 2048-dimensional embedding with rank 10 cannot linearly separate more classes than a 10-dimensional embedding.

---

## 🔬 Eigenvalues and the Spectral Theorem

**Definition.** A scalar $\lambda \in \mathbb{R}$ is an *eigenvalue* of $A \in \mathbb{R}^{n \times n}$ with *eigenvector* $v \neq 0$ if $Av = \lambda v$.

**The spectral theorem.** If $A \in \mathbb{R}^{n \times n}$ is *symmetric* ($A = A^\top$), then:
1. All eigenvalues of $A$ are real.
2. Eigenvectors for distinct eigenvalues are orthogonal.
3. $A$ has an *eigendecomposition*: $A = Q \Lambda Q^\top$ where $Q \in \mathbb{R}^{n \times n}$ has orthonormal columns (the eigenvectors) and $\Lambda = \mathrm{diag}(\lambda_1, \ldots, \lambda_n)$.

**Why it matters.** The covariance matrix $\Sigma = \mathrm{Cov}(z)$ is always symmetric. Its eigenvectors are the *principal directions* of the data distribution; its eigenvalues measure the variance in each direction. $\mathrm{rank}(\Sigma) < d$ exactly when some eigenvalue is zero — the embedding has collapsed in that direction.

---

> [!QUESTION] Exercise 3: Eigenvalues and Matrix Functions
> *Functions of a symmetric matrix act on eigenvalues, not on the matrix entries.*
>
> > **Prerequisites:** [[#🔬 Eigenvalues and the Spectral Theorem|Eigenvalues and the Spectral Theorem]]
>
> Let $A = Q\Lambda Q^\top$ be the eigendecomposition of a symmetric matrix $A$ with eigenvalues $\lambda_1, \ldots, \lambda_n > 0$. Show that $A^{-1} = Q\Lambda^{-1}Q^\top$ and $A^{1/2} = Q\Lambda^{1/2}Q^\top$ (where $\Lambda^{-1} = \mathrm{diag}(1/\lambda_i)$ and $\Lambda^{1/2} = \mathrm{diag}(\sqrt{\lambda_i})$). Use this to compute $\mathrm{tr}(A^{-1})$ in terms of the eigenvalues.

> [!TIP]- Solution to Exercise 3
> **Key insight:** Any reasonable function $f$ of a symmetric matrix is computed by applying $f$ to the diagonal entries of $\Lambda$: $f(A) = Q f(\Lambda) Q^\top$. This makes matrix functions tractable whenever the eigendecomposition is known.
>
> **Sketch:** $A^{-1} = (Q\Lambda Q^\top)^{-1} = (Q^\top)^{-1}\Lambda^{-1}Q^{-1} = Q\Lambda^{-1}Q^\top$ (using $Q^{-1} = Q^\top$ for orthogonal $Q$). Similarly $A^{1/2} \cdot A^{1/2} = Q\Lambda^{1/2}Q^\top Q\Lambda^{1/2}Q^\top = Q\Lambda Q^\top = A$ ✓. For the trace: $\mathrm{tr}(A^{-1}) = \mathrm{tr}(Q\Lambda^{-1}Q^\top) = \mathrm{tr}(\Lambda^{-1}Q^\top Q) = \mathrm{tr}(\Lambda^{-1}) = \sum_i 1/\lambda_i$ (using the cyclic property $\mathrm{tr}(ABC) = \mathrm{tr}(BCA)$).

---

> [!QUESTION] Exercise 4: Spectral Characterization of Rank
> *The rank of a symmetric matrix equals the number of nonzero eigenvalues.*
>
> > **Prerequisites:** [[#🔬 Eigenvalues and the Spectral Theorem|Eigenvalues and the Spectral Theorem]]
>
> Let $A = Q\Lambda Q^\top$ be symmetric with eigenvalues $\lambda_1 \geq \cdots \geq \lambda_r > \lambda_{r+1} = \cdots = \lambda_n = 0$. Show that $\mathrm{rank}(A) = r$. (Hint: use $A = \sum_{i=1}^n \lambda_i q_i q_i^\top$.) Why does this mean $\Sigma \succ 0$ (all eigenvalues positive) is equivalent to $\mathrm{rank}(\Sigma) = d$?

> [!TIP]- Solution to Exercise 4
> **Key insight:** The spectral decomposition writes $A$ as a sum of rank-1 outer products — the number of nonzero terms is exactly the rank.
>
> **Sketch:** From $A = Q\Lambda Q^\top$: expand as $A = \sum_{i=1}^n \lambda_i q_i q_i^\top$. The nonzero terms are $i = 1, \ldots, r$ (since $\lambda_i = 0$ for $i > r$). Each $\lambda_i q_i q_i^\top$ has rank 1 and column space $\mathrm{span}\{q_i\}$. Since $q_1, \ldots, q_r$ are orthonormal (hence linearly independent), $\mathrm{col}(A) = \mathrm{span}\{q_1, \ldots, q_r\}$ has dimension $r$. So $\mathrm{rank}(A) = r$. The equivalence $\Sigma \succ 0 \Leftrightarrow \mathrm{rank}(\Sigma) = d$ follows immediately: $\Sigma \succ 0$ means all $\lambda_i > 0$ (by definition), which means all $n$ eigenvalues are nonzero, so $r = n = d$.

---

## ✅ Positive Semidefinite Matrices

**Definition.** $A \in \mathbb{R}^{n \times n}$ is *positive semidefinite* (PSD), written $A \succeq 0$, if $x^\top A x \geq 0$ for all $x \in \mathbb{R}^n$. It is *positive definite* (PD), written $A \succ 0$, if $x^\top Ax > 0$ for all $x \neq 0$.

**Equivalent characterizations (for symmetric $A$):**
- $A \succeq 0 \Leftrightarrow$ all eigenvalues $\lambda_i \geq 0$.
- $A \succ 0 \Leftrightarrow$ all eigenvalues $\lambda_i > 0 \Leftrightarrow A$ is invertible.
- $A \succeq 0 \Leftrightarrow A = B^\top B$ for some $B$ (the *square root factorization*).

**SSL connection.** The anti-collapse condition in [[concepts/self-supervised-vision/ssl-theory|SSL Theory]] is $\Sigma \succ 0$ — the embedding covariance must be positive definite. VICReg's variance term explicitly enforces $\lambda_j \geq \gamma^2 > 0$ for all $j$, making $\Sigma \succ 0$ a hard constraint.

---

> [!QUESTION] Exercise 5: Every Covariance Matrix Is PSD
> *This is why eigenvalue analysis is always valid for covariance matrices — their eigenvalues are guaranteed non-negative.*
>
> > **Prerequisites:** [[#✅ Positive Semidefinite Matrices|Positive Semidefinite Matrices]]
>
> Let $Z \in \mathbb{R}^{N \times d}$ be any matrix and $\hat\Sigma = Z^\top Z / N$. Show that $\hat\Sigma \succeq 0$ directly from the definition (i.e. show $x^\top \hat\Sigma x \geq 0$ for all $x \in \mathbb{R}^d$). Under what condition on $Z$ is $\hat\Sigma \succ 0$ (positive definite)?

> [!TIP]- Solution to Exercise 5
> **Key insight:** $x^\top (Z^\top Z) x = \|Zx\|^2 \geq 0$ always — PSD follows from the norm being nonneg. PD requires $Zx \neq 0$ for all $x \neq 0$, i.e. $Z$ has full column rank.
>
> **Sketch:** $x^\top \hat\Sigma x = x^\top (Z^\top Z/N) x = \|Zx\|^2/N \geq 0$ for all $x$ since norms are nonnegative. So $\hat\Sigma \succeq 0$. For $\hat\Sigma \succ 0$: need $\|Zx\|^2 > 0$ for all $x \neq 0$, i.e. $Zx \neq 0$ for all $x \neq 0$, i.e. the null space of $Z$ is $\{0\}$, i.e. $Z$ has full column rank ($\mathrm{rank}(Z) = d$). In SSL terms: $\hat\Sigma \succ 0 \Leftrightarrow$ no dimensional collapse in $Z$.

---

## 📊 Covariance Matrices

**Definition.** For a random vector $z \in \mathbb{R}^d$ with mean $\mu = \mathbb{E}[z]$, the *covariance matrix* is:
$$\Sigma = \mathbb{E}[(z - \mu)(z - \mu)^\top] \in \mathbb{R}^{d \times d}.$$

The $(i,j)$ entry is $\Sigma_{ij} = \mathrm{Cov}(z_i, z_j)$. The diagonal $\Sigma_{jj} = \mathrm{Var}(z_j)$.

**Eigendecomposition of $\Sigma$.** Since $\Sigma \succeq 0$ and symmetric (by Exercise 5), $\Sigma = Q\Lambda Q^\top$ with $\Lambda = \mathrm{diag}(\lambda_1, \ldots, \lambda_d)$, $\lambda_i \geq 0$. The eigenvectors $q_i$ (columns of $Q$) are the *principal directions* — the directions of maximum/minimum variance. The eigenvalue $\lambda_i = \mathrm{Var}(q_i^\top z)$ is the variance of $z$ projected onto $q_i$.

**SSL connection.** The alignment and uniformity framework ([[concepts/self-supervised-vision/ssl-theory|SSL Theory §Alignment and Uniformity]]) requires the embedding covariance to be "spread" — all eigenvalues significant. The *effective rank* measures how spread the eigenvalue distribution is.

---

> [!QUESTION] Exercise 6: Variance in a Rotated Basis
> *Eigenvalues are the variances in the eigenvector directions — the most natural coordinate system for a covariance matrix.*
>
> > **Prerequisites:** [[#📊 Covariance Matrices|Covariance Matrices]]
>
> Let $\Sigma = Q\Lambda Q^\top$ be the covariance of $z$. Define the *whitened* variable $\tilde{z} = \Lambda^{-1/2}Q^\top z$ (assuming $\Sigma \succ 0$). Compute $\mathrm{Cov}(\tilde{z})$ and show it equals $I_d$. Why is whitening useful for the Barlow Twins objective (which targets $C = I_d$)?

> [!TIP]- Solution to Exercise 6
> **Key insight:** Whitening simultaneously normalizes all variances to 1 and removes all correlations — it maps any full-rank covariance to the identity. Barlow Twins' $C = I_d$ target is exactly the whitened condition.
>
> **Sketch:** $\mathrm{Cov}(\tilde{z}) = \Lambda^{-1/2}Q^\top \mathrm{Cov}(z) Q\Lambda^{-1/2} = \Lambda^{-1/2}Q^\top Q\Lambda Q^\top Q\Lambda^{-1/2} = \Lambda^{-1/2}\Lambda\Lambda^{-1/2} = I_d$. The whitening transform makes all dimensions unit-variance and pairwise uncorrelated. Barlow Twins drives the *cross-correlation* matrix $C = (Z^A)^\top Z^B / N$ toward $I_d$ — this is cross-whitening: requiring the joint embedding $(z^A, z^B)$ to satisfy $\mathrm{Cov}(z^A_i, z^B_j) = \delta_{ij}$, i.e. each augmented view's embeddings are whitened and the two views are maximally aligned dimension-by-dimension.

---

## ✂️ Singular Value Decomposition

**Theorem (SVD).** Every matrix $A \in \mathbb{R}^{m \times n}$ can be written as:
$$A = U \Sigma V^\top$$
where $U \in \mathbb{R}^{m \times m}$ and $V \in \mathbb{R}^{n \times n}$ are orthogonal, and $\Sigma = \mathrm{diag}(\sigma_1, \ldots, \sigma_r, 0, \ldots, 0)$ with $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$. The $\sigma_i$ are the *singular values*, $r = \mathrm{rank}(A)$.

**Connection to eigendecomposition.** $A^\top A = V\Sigma^\top U^\top U\Sigma V^\top = V\mathrm{diag}(\sigma_i^2)V^\top$ — so the right singular vectors of $A$ are the eigenvectors of $A^\top A$, with eigenvalues $\sigma_i^2$.

**Eckart–Young theorem (best low-rank approximation).** The best rank-$k$ approximation to $A$ in Frobenius norm is:
$$A_k = \sum_{i=1}^k \sigma_i u_i v_i^\top, \quad \|A - A_k\|_F^2 = \sum_{i=k+1}^r \sigma_i^2.$$

**SSL connection.** Tian et al.'s theorem says that SSL converges to the top-$d$ eigenvectors of the augmentation-averaged covariance $F$. Since $F = \mathbb{E}[z_1 z_2^\top]$ is symmetric PSD, its eigenvectors = its right singular vectors. SSL converges to the rank-$d$ SVD truncation of $F$ — the *PCA* of augmentation-invariant covariance.

---

> [!QUESTION] Exercise 7: SVD and PCA
> *PCA is the SVD of the centered data matrix — the connection that underlies Tian et al.'s convergence theorem.*
>
> > **Prerequisites:** [[#✂️ Singular Value Decomposition|Singular Value Decomposition]]
>
> Let $X \in \mathbb{R}^{N \times n}$ be a centered data matrix (zero mean columns). Show that the top-$d$ principal components of $X$ (the directions of maximum variance) are exactly the top-$d$ right singular vectors of $X$. Then argue why minimizing the prediction error $\mathbb{E}[\|h^*(z_1) - z_2\|^2]$ over linear encoders $z = Wx$ with $W \in \mathbb{R}^{d \times n}$ is equivalent to computing the top-$d$ PCA directions of $\mathbb{E}[x_1 x_2^\top]$.

> [!TIP]- Solution to Exercise 7
> **Key insight:** PCA = truncated SVD of the data matrix. SSL in the linear regime finds the PCA directions of the cross-view covariance — the maximally augmentation-consistent subspace.
>
> **Sketch:** The sample covariance is $\hat\Sigma_X = X^\top X / N$. PCA finds the $d$ eigenvectors of $\hat\Sigma_X$ with largest eigenvalues. Since $X = U\Sigma V^\top$: $\hat\Sigma_X = V(\Sigma^\top\Sigma/N)V^\top$ — the right singular vectors $V$ are the eigenvectors of $\hat\Sigma_X$, so the top-$d$ PC directions = top-$d$ right singular vectors of $X$. ✓ For the SSL claim: the optimal linear predictor is $h^*(z_1) = \Sigma_{21}\Sigma_{11}^{-1}z_1$ (Exercise 3 in ssl-theory). Substituting $z_i = Wx_i$ and minimizing over $W$: the problem reduces to finding $W$ that maximizes $\mathrm{tr}(W F W^\top)$ subject to $WW^\top = I$ (orthonormality), where $F = \mathbb{E}[x_1 x_2^\top]$. By the variational characterization of eigenvalues, the optimum is $W = $ top-$d$ eigenvectors of $F$ — i.e., the PCA of the cross-view covariance.

---

## 📏 Trace, Frobenius Norm, and Stable Rank

**Definitions.**

| Quantity | Formula | Properties |
|---|---|---|
| Trace | $\mathrm{tr}(A) = \sum_i A_{ii}$ | $\mathrm{tr}(AB) = \mathrm{tr}(BA)$; $\mathrm{tr}(A) = \sum_i \lambda_i$ |
| Frobenius norm | $\|A\|_F = \sqrt{\sum_{i,j}A_{ij}^2} = \sqrt{\mathrm{tr}(A^\top A)}$ | $\|A\|_F^2 = \sum_i \sigma_i^2$ |
| Stable rank | $\mathrm{srank}(A) = \|A\|_F^2 / \|A\|_2^2 = \sum_i \sigma_i^2 / \sigma_1^2$ | $\in [1, \mathrm{rank}(A)]$ |

**The cyclic trace property** $\mathrm{tr}(ABC) = \mathrm{tr}(BCA) = \mathrm{tr}(CAB)$ is used constantly in matrix calculus derivations.

**Stable rank for covariance matrices.** For $\Sigma$ with eigenvalues $\lambda_i$: $\mathrm{tr}(\Sigma) = \sum_i \lambda_i$, $\|\Sigma\|_F^2 = \sum_i \lambda_i^2$, and $\mathrm{srank}(\Sigma) = (\sum_i \lambda_i)^2 / \sum_i \lambda_i^2$. This equals $d$ when all $\lambda_i$ are equal, and 1 when only one $\lambda_i$ is nonzero.

---

> [!QUESTION] Exercise 8: Trace and the Barlow Twins Loss
> *The Barlow Twins loss has a natural expression in terms of the Frobenius norm of the cross-correlation matrix.*
>
> > **Prerequisites:** [[#📏 Trace, Frobenius Norm, and Stable Rank|Trace, Frobenius Norm, and Stable Rank]]
>
> Let $C \in \mathbb{R}^{d \times d}$ be the cross-correlation matrix from Barlow Twins. Show that:
> $$\mathcal{L}_{\mathrm{BT}} = \|C - I_d\|_F^2 + (\lambda - 1)\sum_i (1 - C_{ii})^2.$$
> For the special case $\lambda = 1$, the loss simplifies to $\|C - I_d\|_F^2$. Interpret: what does minimizing the Frobenius distance $\|C - I_d\|_F$ mean geometrically?

> [!TIP]- Solution to Exercise 8
> **Key insight:** The Barlow Twins loss (at $\lambda=1$) is exactly the squared Frobenius distance between the cross-correlation matrix and the identity — it directly minimizes the "distance to whitened" condition.
>
> **Sketch:** $\|C - I_d\|_F^2 = \sum_{i,j}(C_{ij} - \delta_{ij})^2 = \sum_i(C_{ii}-1)^2 + \sum_{i\neq j}C_{ij}^2$. The Barlow Twins loss is $\mathcal{L}_\text{BT} = \sum_i(1-C_{ii})^2 + \lambda\sum_{i\neq j}C_{ij}^2$. Comparing: $\mathcal{L}_\text{BT} = \|C-I\|_F^2 + (\lambda-1)\sum_{i\neq j}C_{ij}^2$. At $\lambda=1$: $\mathcal{L}_\text{BT} = \|C-I_d\|_F^2$. Geometrically, this is the squared distance (in the space of $d\times d$ matrices with the Frobenius inner product) between $C$ and the identity. Minimizing it finds the $C$ closest to $I_d$ — i.e., drives the embedding toward the whitened cross-covariance condition.

---

> [!QUESTION] Exercise 9: Stable Rank Bounds
> *The stable rank lies between 1 and the true rank — it detects near-collapse even when no eigenvalue is exactly zero.*
>
> > **Prerequisites:** [[#📏 Trace, Frobenius Norm, and Stable Rank|Trace, Frobenius Norm, and Stable Rank]]
>
> Let $\Sigma$ have eigenvalues $\lambda_1 \geq \cdots \geq \lambda_d \geq 0$. Prove the bounds $1 \leq \mathrm{srank}(\Sigma) \leq \mathrm{rank}(\Sigma)$ using the Cauchy–Schwarz inequality. Then compute $\mathrm{srank}(\Sigma)$ for: (a) $\Sigma = \sigma^2 I_d$, (b) $\Sigma = \mathrm{diag}(1, 0, \ldots, 0)$, (c) $\Sigma = \mathrm{diag}(1, \varepsilon, \ldots, \varepsilon)$ with small $\varepsilon > 0$.

> [!TIP]- Solution to Exercise 9
> **Key insight:** Cauchy–Schwarz gives both bounds simultaneously; the three cases illustrate that stable rank correctly identifies full rank, collapse, and near-collapse.
>
> **Sketch:** Let $r = \mathrm{rank}(\Sigma)$ and $\lambda_1, \ldots, \lambda_r > 0$ (nonzero eigenvalues). **Upper bound:** Cauchy–Schwarz gives $(\sum_{i=1}^r \lambda_i)^2 = (\sum_{i=1}^r \lambda_i \cdot 1)^2 \leq (\sum_{i=1}^r \lambda_i^2)(\sum_{i=1}^r 1^2) = r\sum\lambda_i^2$. So $\mathrm{srank} = (\sum\lambda_i)^2/\sum\lambda_i^2 \leq r = \mathrm{rank}(\Sigma)$. **Lower bound:** $(\sum\lambda_i)^2 \geq \max_i\lambda_i \cdot \sum\lambda_i \geq \sum\lambda_i^2$ (since $\max_i\lambda_i \geq \lambda_i$ for each $i$), so $\mathrm{srank} \geq 1$. **Cases:** (a) $\Sigma = \sigma^2 I$: $\mathrm{srank} = (d\sigma^2)^2/(d\sigma^4) = d$ ✓ full. (b) $\Sigma = \mathrm{diag}(1,0,\ldots,0)$: $\mathrm{srank} = 1^2/1^2 = 1$ ✓ full collapse. (c) $\Sigma = \mathrm{diag}(1,\varepsilon,\ldots,\varepsilon)$: $\mathrm{srank} = (1+(d-1)\varepsilon)^2/(1+(d-1)\varepsilon^2)$. As $\varepsilon\to 0$: $\to 1$; as $\varepsilon \to 1$: $\to d$. For small $\varepsilon$, $\mathrm{srank} \approx 1 + 2(d-1)\varepsilon$ — near-collapse is detected even though all eigenvalues are strictly positive.

---

## 🔄 Orthogonal Matrices and the Group O(d)

**Definition.** $R \in \mathbb{R}^{d \times d}$ is *orthogonal* if $R^\top R = RR^\top = I_d$. Equivalently, $R^{-1} = R^\top$, and $R$ preserves inner products: $\langle Ru, Rv\rangle = u^\top R^\top R v = u^\top v$.

**The orthogonal group.** $O(d) = \{R \in \mathbb{R}^{d \times d} : R^\top R = I\}$ is a group under matrix multiplication. It has two connected components: rotations ($\det R = +1$, forming $SO(d)$) and reflections ($\det R = -1$).

**SSL connection.** In [[concepts/self-supervised-vision/ssl-vision|SSL Vision §Exercise 6 (Barlow Twins)]], the solution set of $\mathcal{L}_\text{BT} = 0$ forms an *orbit* under $O(d)$: if $Z$ is a solution, so is $ZR$ for any $R \in O(d)$. This means the SSL objective does not prefer any particular orientation of the embedding space — all rotations are equivalent. This is the *rotational symmetry* of SSL.

---

> [!QUESTION] Exercise 10: Orthogonal Invariance of the Covariance
> *Rotating the embedding space does not change the spectrum of the covariance — only the eigenvectors rotate.*
>
> > **Prerequisites:** [[#🔄 Orthogonal Matrices and the Group O(d)|Orthogonal Matrices and the Group O(d)]]
>
> Let $Z \in \mathbb{R}^{N \times d}$ have covariance $\hat\Sigma = Z^\top Z / N$ (assuming zero mean). Let $\tilde Z = ZR$ for $R \in O(d)$. Compute the covariance $\tilde\Sigma$ of $\tilde Z$ and show it has the same eigenvalues as $\hat\Sigma$ (but generally different eigenvectors). What does this imply about $\mathrm{srank}(\tilde\Sigma)$ and $\mathrm{rank}(\tilde\Sigma)$?

> [!TIP]- Solution to Exercise 10
> **Key insight:** Orthogonal transformations rotate the eigenvectors but leave eigenvalues unchanged — so rank, stable rank, and all spectral properties are $O(d)$-invariant.
>
> **Sketch:** $\tilde\Sigma = \tilde Z^\top \tilde Z / N = (ZR)^\top(ZR)/N = R^\top(Z^\top Z/N)R = R^\top\hat\Sigma R$. If $\hat\Sigma = Q\Lambda Q^\top$, then $\tilde\Sigma = R^\top Q\Lambda Q^\top R = (Q^\top R)^\top \Lambda (Q^\top R)$ — so $\tilde\Sigma$ has the same eigenvalues $\Lambda$ but eigenvectors $Q^\top R$ (a rotation of $Q$). Therefore $\mathrm{srank}(\tilde\Sigma) = \mathrm{srank}(\hat\Sigma)$ and $\mathrm{rank}(\tilde\Sigma) = \mathrm{rank}(\hat\Sigma)$. The rotational ambiguity in SSL ($ZR$ is equally valid as $Z$) does not affect any spectral diagnostic — tracking $\mathrm{srank}$ during training is invariant to the arbitrary rotation of the learned basis.

---

## 🧮 Matrix Calculus

Two gradient rules appear repeatedly in SSL loss derivations.

**Gradient of trace.** For differentiable $f : \mathbb{R}^{d \times d} \to \mathbb{R}$:
$$\frac{\partial}{\partial A}\,\mathrm{tr}(AB) = B^\top, \qquad \frac{\partial}{\partial A}\,\mathrm{tr}(A^\top A) = 2A.$$

**Gradient of the Frobenius norm.**
$$\frac{\partial}{\partial A}\,\|A - B\|_F^2 = 2(A - B).$$

**Gradient of a quadratic form.** For symmetric $\Sigma$:
$$\frac{\partial}{\partial x}\,(x^\top \Sigma x) = 2\Sigma x.$$

**Jacobian of matrix products.** When $A$ depends on parameter $\theta$ via $A(\theta) = f(\theta)g(\theta)^\top$, the chain rule applies componentwise: $\partial \ell / \partial \theta = (\partial \ell / \partial A) \cdot (\partial A / \partial \theta)$, using the Frobenius inner product $\langle \cdot, \cdot \rangle_F$.

---

> [!QUESTION] Exercise 11: Gradient of the Barlow Twins Loss
> *Computing the gradient of $\mathcal{L}_\text{BT}$ with respect to the embeddings reveals the update rule that drives $C \to I$.*
>
> > **Prerequisites:** [[#🧮 Matrix Calculus|Matrix Calculus]]
>
> For simplicity, take $\lambda = 1$ so $\mathcal{L}_\text{BT} = \|C - I\|_F^2$ where $C = (Z^A)^\top Z^B / N$ and $Z^A, Z^B$ are batch-normalized (treat them as free variables). Compute $\partial \mathcal{L}_\text{BT} / \partial Z^A$ and show it equals $2(C - I)(Z^B)^\top / N$. Interpret: when $C = I$ (target reached), the gradient vanishes — confirming $C = I$ is a fixed point.

> [!TIP]- Solution to Exercise 11
> **Key insight:** The gradient of the Barlow Twins loss w.r.t. $Z^A$ is proportional to the "error matrix" $(C - I)$ projected back through $Z^B$ — it pushes $Z^A$ to reduce the deviation of each cross-correlation from its target.
>
> **Sketch:** Write $\mathcal{L}_\text{BT} = \mathrm{tr}((C-I)^\top(C-I))$ and $C = (Z^A)^\top Z^B/N$. Then:
> $$\frac{\partial\mathcal{L}}{\partial C} = 2(C - I).$$
> By the chain rule, $\partial \mathcal{L}/\partial Z^A = (\partial \mathcal{L}/\partial C)(\partial C/\partial Z^A)$. Since $C = (Z^A)^\top Z^B/N$, we have $\partial C_{ij}/\partial Z^A_{bk} = Z^B_{bj}\delta_{ik}/N$, giving:
> $$\frac{\partial\mathcal{L}}{\partial Z^A} = \frac{2}{N}(C-I)(Z^B)^\top \cdot \frac{1}{N} \cdot N = \frac{2}{N} Z^B (C - I)^\top.$$
> At $C = I$: gradient $= 0$ ✓. When $C_{ij} > \delta_{ij}$ (dimensions over-correlated), the gradient pushes $Z^A$ in the direction that reduces $C_{ij}$ toward $\delta_{ij}$.

---

## 📚 References

| Reference Name | Brief Summary | Link |
|---|---|---|
| Strang, *Introduction to Linear Algebra* (5th ed.) | Standard reference for rank, eigenvalues, SVD, and PCA with geometric intuition | [MIT OpenCourseWare](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) |
| Horn & Johnson, *Matrix Analysis* | Rigorous treatment of spectral theory, PSD matrices, and matrix norms | [Cambridge University Press](https://doi.org/10.1017/CBO9780511810817) |
| Petersen & Pedersen, *The Matrix Cookbook* | Dense reference for matrix calculus identities (trace gradients, Frobenius norm derivatives) | [matrixcookbook.com](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) |
| [[concepts/self-supervised-vision/ssl-theory\|SSL Theory]] | Uses all concepts above: stable rank, PSD covariance, SVD/PCA convergence, matrix calculus for loss gradients | local |
