# Prerequisites for Multi-Task Bayesian Optimization (arXiv:1904.01049)

**Target paper:** [Practical Multi-Task Scalable Bayesian Optimization](https://arxiv.org/abs/1904.01049)
**Goal:** Read the paper deeply — follow derivations, kernel construction, and theoretical results
**Profile:** Solid multivariable calculus + linear algebra; introductory probability; surface-level GP and BO familiarity
**Estimated pace:** ~10 hrs/week → **4–5 weeks** to paper readiness

---

## Dependency Graph

```
Multivariate Gaussian conditioning
        ↓
Bayesian inference (formally)
        ↓
GP regression (derive predictive posterior)
        ↓                    ↓
Kernel theory           Marginal likelihood + hyperparameter learning
        ↓                    ↓
        └─────────┬──────────┘
                  ↓
      Bayesian optimization (EI / UCB derivations)
                  ↓
      Multi-task GP kernels (LMC / ICM)
                  ↓
             📄 The Paper
```

---

## Week 1 — Multivariate Gaussians & Bayesian Foundations (~10 hrs)

**🎯 Goal:** Derive the Gaussian conditional and marginal from scratch; understand Bayesian inference as posterior computation.

The GP posterior *is* a Gaussian conditional. Everything downstream depends on internalizing this one formula.

### 📚 Resources

| Resource | Sections | Format | Time |
|----------|----------|--------|------|
| [Bishop — PRML](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) | §2.3 (Gaussian Distribution) | Textbook | 3 hrs |
| [Murphy — Probabilistic ML: An Introduction](https://probml.github.io/pml-book/book1.html) | §2.3 (Gaussians), §4.2 (Bayesian inference for Gaussians) | Textbook | 3 hrs |
| [Petersen & Pedersen — The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) | §8 (block matrix inversion, Schur complement) | Reference | 1 hr |

### ✏️ Exercises

**Exercise 1.1 — Derive the Gaussian conditional.**
Given a joint Gaussian $\begin{pmatrix} \mathbf{x} \\ \mathbf{y} \end{pmatrix} \sim \mathcal{N}\!\left(\begin{pmatrix} \boldsymbol{\mu}_x \\ \boldsymbol{\mu}_y \end{pmatrix}, \begin{pmatrix} \Sigma_{xx} & \Sigma_{xy} \\ \Sigma_{yx} & \Sigma_{yy} \end{pmatrix}\right)$, derive $p(\mathbf{x} \mid \mathbf{y})$ by completing the square. Identify the Schur complement $\Sigma_{xx} - \Sigma_{xy}\Sigma_{yy}^{-1}\Sigma_{yx}$.

**Exercise 1.2 — Bayesian linear regression.**
Place a Gaussian prior $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \alpha^{-1}I)$ on weights and a Gaussian likelihood $y \mid \mathbf{w} \sim \mathcal{N}(\mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}), \beta^{-1})$. Derive the posterior $p(\mathbf{w} \mid \mathcal{D})$ in closed form. This is the finite-dimensional analogue of GP regression — understanding this derivation cold makes Week 2 immediate.

**Exercise 1.3 — Marginal likelihood.**
From Exercise 1.2, compute the marginal likelihood $p(\mathbf{y} \mid \mathbf{X})$ by integrating out $\mathbf{w}$. Identify which terms penalize model complexity.

### ✅ Checkpoint
You should be able to: (a) state the Gaussian conditioning formula without reference, (b) explain why the conditional mean is a *linear* function of the conditioning variable, and (c) derive the posterior in Bayesian linear regression in under 10 minutes.

---

## Week 2 — Gaussian Process Regression (~12 hrs)

**🎯 Goal:** Understand GPs as distributions over functions. Derive the GP predictive posterior and connect it to Week 1's conditioning formula.

### 📚 Resources

| Resource | Sections | Format | Time |
|----------|----------|--------|------|
| [Rasmussen & Williams — Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/) | Ch. 1 (intro), Ch. 2 (GP regression — derive everything) | Textbook (free) | 5 hrs |
| [R&W GPML](http://www.gaussianprocess.org/gpml/) | Ch. 4 §4.1–4.3 (covariance functions) | Textbook | 3 hrs |
| [Neil Lawrence — GP Summer School lectures](http://gpss.cc/) | Intro GP lecture + lab | Video + notebook | 2 hrs |

### ✏️ Exercises

**Exercise 2.1 — GP predictive posterior from scratch.**
Suppose you observe noisy data $\mathbf{y} = f(\mathbf{X}) + \boldsymbol{\varepsilon}$ where $f \sim \mathcal{GP}(0, k)$ and $\varepsilon_i \sim \mathcal{N}(0, \sigma_n^2)$. Using only the Gaussian conditioning formula from Week 1, derive the posterior predictive $p(f_* \mid \mathbf{X}_*, \mathbf{X}, \mathbf{y})$. Write out mean and variance explicitly. Verify this matches R&W eq. (2.23–2.24).

**Exercise 2.2 — Implement GP regression from scratch.**
In Python (NumPy only — no GPy/GPflow), implement:
- SE kernel: $k(x, x') = \sigma_f^2 \exp\!\left(-\frac{(x-x')^2}{2\ell^2}\right)$
- GP posterior mean and variance
- Posterior sample paths via Cholesky

Plot the posterior with 3, 10, and 50 training points on a 1D function of your choice. Verify variance collapses to zero at observed points.

**Exercise 2.3 — Kernel composition.**
Prove: if $k_1$ and $k_2$ are PSD kernels, then $k_1 + k_2$ and $k_1 \cdot k_2$ are PSD. Use this to construct a kernel modeling a trend plus periodic component.

**Exercise 2.4 — Mercer's theorem (conceptual).**
State Mercer's theorem and explain in one paragraph why it licenses thinking of $k(x, x')$ as a dot product in a (possibly infinite-dimensional) feature space. Why does this matter for GP regression?

### ✅ Checkpoint
You should be able to: (a) derive the GP posterior mean and covariance in 5 minutes given the conditioning formula, (b) explain what "distribution over functions" means precisely (not just intuitively), and (c) construct a novel kernel for a given function class and justify its PSD property.

---

## Week 3 — Bayesian Optimization (~10 hrs)

**🎯 Goal:** Understand the BO loop, derive the main acquisition functions, and see how GP uncertainty drives exploration.

### 📚 Resources

| Resource | Sections | Format | Time |
|----------|----------|--------|------|
| [Frazier — A Tutorial on Bayesian Optimization](https://arxiv.org/abs/1807.02811) | Full (short, 22pp) | Paper | 2 hrs |
| [Garnett — Bayesian Optimization](https://bayesoptbook.com/) | Ch. 1–5 (foundations through acquisition functions) | Textbook (free) | 4 hrs |
| [Shahriari et al. — Taking the Human Out of the Loop](https://ieeexplore.ieee.org/document/7352306) | §III (acquisition functions) only | Survey | 1.5 hrs |

### ✏️ Exercises

**Exercise 3.1 — Derive Expected Improvement.**
Given a GP surrogate $f \sim \mathcal{GP}$ with posterior mean $\mu(\mathbf{x})$ and variance $\sigma^2(\mathbf{x})$, derive the EI acquisition function:
$$\text{EI}(\mathbf{x}) = \mathbb{E}[\max(f(\mathbf{x}) - f^+, 0)]$$
in closed form. You will need the formula $\mathbb{E}[\max(Z - \tau, 0)]$ for $Z \sim \mathcal{N}(\mu, \sigma^2)$ — derive this too using integration by parts.

**Exercise 3.2 — BO from scratch.**
Implement a minimal BO loop:
1. GP surrogate with SE kernel
2. EI acquisition maximized by grid search
3. Sequential evaluation loop

Test on $f(x) = -x^2 + \sin(3x)$ on $[-3, 3]$. Plot the surrogate, EI, and observed points at each iteration.

**Exercise 3.3 — Regret analysis (conceptual).**
Read Garnett Ch. 5 on convergence. In your own words: what is *cumulative regret* and what does a sublinear regret bound imply about BO's performance as $T \to \infty$?

### ✅ Checkpoint
You should be able to: (a) derive EI closed form without reference, (b) explain the exploration-exploitation tradeoff in terms of the GP posterior variance, and (c) identify one assumption in the standard BO convergence analysis that the paper under study relaxes.

---

## Week 4 — Multi-Task GPs & Kernel Coregionalization (~10 hrs)

**🎯 Goal:** Understand how GP regression extends to vector-valued outputs (tasks), then read the paper.

This is the direct prerequisite for the paper's kernel construction.

### 📚 Resources

| Resource | Sections | Format | Time |
|----------|----------|--------|------|
| [R&W GPML](http://www.gaussianprocess.org/gpml/) | §5.4–5.5 (multi-output GPs, coregionalization) | Textbook | 2 hrs |
| [Bonilla et al. — Multi-Task GP Prediction](https://proceedings.neurips.cc/paper/2007/hash/66368270ffd51418ec58bd793f2d9b1b-Abstract.html) | Full | Paper | 2 hrs |
| [Álvarez et al. — Kernels for Vector-Valued Functions](https://jmlr.org/papers/v13/alvarez12a.html) | §2–3 (LMC and ICM) | Survey | 2 hrs |

### ✏️ Exercises

**Exercise 4.1 — Linear Model of Coregionalization.**
In the LMC, each output is modeled as $f_d(\mathbf{x}) = \sum_{q=1}^Q a_{d,q} u_q(\mathbf{x})$ for independent GPs $u_q \sim \mathcal{GP}(0, k_q)$. Derive the cross-covariance $\text{Cov}(f_d(\mathbf{x}), f_{d'}(\mathbf{x}'))$ and show the multi-task covariance matrix factorizes as a Kronecker product in the ICM special case.

**Exercise 4.2 — Bias-variance in multi-task settings.**
Suppose task 2 provides a *biased* but low-noise version of task 1. Sketch (analytically or geometrically) how the optimal weight on task-2 data in the multi-task posterior trades off between the bias it introduces and the variance it reduces.

**Exercise 4.3 — Read the paper.**
After completing 4.1–4.2, read arXiv:1904.01049 with these questions in mind:
- How does the paper's kernel for the biased offline/online task pair differ from standard ICM?
- What theoretical result characterizes the optimal weight on biased data?
- What assumption about the offline simulator does the kernel encode?

### ✅ Checkpoint
You can reconstruct the paper's multi-task kernel from the LMC framework and explain (in the language of Section 3) what the kernel assumes about the relationship between the online and offline tasks.

---

## Week 5 — Deep Reading of the Paper (~6–8 hrs)

**🎯 Goal:** Work through every derivation in the paper, annotate assumptions, and identify what's novel vs. standard.

### Strategy

1. **Pass 1 (2 hrs):** Read straight through, noting every symbol and equation you don't immediately recognize. Flag them.
2. **Pass 2 (3 hrs):** For each flagged item, trace it back to the prerequisite concept. Derive or look up.
3. **Pass 3 (1.5 hrs):** Re-read the theory section (§3–4) and reproduce the main result on paper without looking at the proof.

### Questions to answer during deep read

- [ ] What does the paper assume about the offline simulator bias structure?
- [ ] How is the multi-task acquisition function derived — is it standard EI applied to the GP posterior?
- [ ] What is the theoretical guarantee, and what does it require of the kernel?
- [ ] Where does the paper depart from the Bonilla et al. setup?
- [ ] What are the limits of the convergence result?

---

## Reference Table

| Reference | Brief Summary | Link |
|-----------|---------------|------|
| Bishop — PRML Ch. 2 | Gaussians, conjugate priors, Bayesian inference | [PDF](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) |
| Murphy — PML Book 1 | Modern probabilistic ML foundations | [Free online](https://probml.github.io/pml-book/book1.html) |
| Rasmussen & Williams — GPML | The GP bible; derivations of everything | [Free online](http://www.gaussianprocess.org/gpml/) |
| Frazier — BO Tutorial | Best short intro to BO; EI derivation | [arXiv:1807.02811](https://arxiv.org/abs/1807.02811) |
| Garnett — Bayesian Optimization | Full rigorous textbook; convergence theory | [Free online](https://bayesoptbook.com/) |
| Shahriari et al. 2016 | Comprehensive BO survey | [IEEE](https://ieeexplore.ieee.org/document/7352306) |
| Bonilla et al. 2007 | Multi-task GP prediction; LMC derivation | [NeurIPS](https://proceedings.neurips.cc/paper/2007/hash/66368270ffd51418ec58bd793f2d9b1b-Abstract.html) |
| Álvarez et al. — Vector-Valued Kernels | Survey of LMC, ICM, convolution kernels | [JMLR](https://jmlr.org/papers/v13/alvarez12a.html) |
| Petersen & Pedersen — Matrix Cookbook | Quick reference for matrix identities | [PDF](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) |
