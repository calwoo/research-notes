# Convex Optimization and Lagrangian Duality

## Table of Contents

- [[#1. Convex Sets and Convex Functions|1. Convex Sets and Convex Functions]]
  - [[#1.1 Convex Sets|1.1 Convex Sets]]
  - [[#1.2 Operations Preserving Convexity of Sets|1.2 Operations Preserving Convexity of Sets]]
  - [[#1.3 Convex Functions|1.3 Convex Functions]]
  - [[#1.4 First-Order and Second-Order Conditions|1.4 First-Order and Second-Order Conditions]]
  - [[#1.5 Operations Preserving Convexity of Functions|1.5 Operations Preserving Convexity of Functions]]
- [[#2. Convex Optimization Problems|2. Convex Optimization Problems]]
  - [[#2.1 Standard Form|2.1 Standard Form]]
  - [[#2.2 Special Cases|2.2 Special Cases]]
  - [[#2.3 Local Equals Global|2.3 Local Equals Global]]
- [[#3. The Lagrangian|3. The Lagrangian]]
  - [[#3.1 Definition and Lower-Bound Interpretation|3.1 Definition and Lower-Bound Interpretation]]
  - [[#3.2 The Dual Function|3.2 The Dual Function]]
  - [[#3.3 Concavity of the Dual Function|3.3 Concavity of the Dual Function]]
- [[#4. Lagrangian Duality|4. Lagrangian Duality]]
  - [[#4.1 The Dual Problem|4.1 The Dual Problem]]
  - [[#4.2 Weak Duality|4.2 Weak Duality]]
  - [[#4.3 The Duality Gap|4.3 The Duality Gap]]
  - [[#4.4 Strong Duality and Slater's Condition|4.4 Strong Duality and Slater's Condition]]
- [[#5. KKT Conditions|5. KKT Conditions]]
  - [[#5.1 Statement of the Conditions|5.1 Statement of the Conditions]]
  - [[#5.2 Necessity|5.2 Necessity]]
  - [[#5.3 Sufficiency|5.3 Sufficiency]]
  - [[#5.4 Geometric Interpretation|5.4 Geometric Interpretation]]
- [[#6. Recovering Primal Solutions from the Dual|6. Recovering Primal Solutions from the Dual]]
  - [[#6.1 Strong Convexity and Uniqueness|6.1 Strong Convexity and Uniqueness]]
  - [[#6.2 Dual Decomposition Sketch|6.2 Dual Decomposition Sketch]]
  - [[#6.3 Connection to Click Shaping|6.3 Connection to Click Shaping]]
- [[#References|References]]

---

## 1. Convex Sets and Convex Functions 📐

The study of convex optimization begins with the geometry of convex sets and functions. These objects possess structural properties that make optimization problems over them tractable.

### 1.1 Convex Sets

**Definition (Convex Set).** A set $C \subseteq \mathbb{R}^n$ is *convex* if for every $x, y \in C$ and every $\theta \in [0, 1]$, the point $\theta x + (1 - \theta) y$ belongs to $C$.

Geometrically, a set is convex if the line segment connecting any two of its points stays entirely within the set. The following are the canonical examples.

**Example (Halfspace).** Fix $a \in \mathbb{R}^n \setminus \{0\}$ and $b \in \mathbb{R}$. The set $H = \{x \in \mathbb{R}^n : a^\top x \le b\}$ is a *halfspace*. It is convex: if $a^\top x \le b$ and $a^\top y \le b$, then $a^\top(\theta x + (1-\theta)y) = \theta a^\top x + (1-\theta)a^\top y \le \theta b + (1-\theta)b = b$.

**Example (Norm Ball).** For any norm $\|\cdot\|$ on $\mathbb{R}^n$, the ball $B(x_c, r) = \{x : \|x - x_c\| \le r\}$ is convex by the triangle inequality.

**Example (Polyhedron).** A *polyhedron* is a finite intersection of halfspaces and hyperplanes: $\mathcal{P} = \{x : Ax \preceq b,\; Cx = d\}$ where $\preceq$ denotes componentwise inequality. Since each halfspace is convex and intersections of convex sets are convex (see below), polyhedra are convex.

**Example (Epigraph).** Given $f : \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$, its *epigraph* is $\operatorname{epi}(f) = \{(x, t) \in \mathbb{R}^n \times \mathbb{R} : f(x) \le t\}$. As we will see, $f$ is a convex function if and only if $\operatorname{epi}(f)$ is a convex set.

![[figures/bv2004-fig2.2-convex-nonconvex-sets.png]]
*Figure 2.2 (Boyd & Vandenberghe, 2004): Convex and non-convex sets. Left: the hexagon (including boundary) is convex. Middle: the kidney-shaped set is not convex — the line segment between the two marked points leaves the set. Right: the square with some boundary points missing is not convex. The convex hulls of two sets are shown below.*

> [!EXAMPLE] Convex versus Non-Convex Sets
> The unit disk $\{(x_1, x_2) : x_1^2 + x_2^2 \le 1\}$ is convex. The unit circle (boundary only) is not, since the midpoint of two boundary points is typically interior. The set $\{x : |x_1| + |x_2| \le 1\}$ (the $\ell^1$ ball) is convex; the set $\{x : x_1^2 + x_2^2 = 1\} \cup \{0\}$ is not.

### 1.2 Operations Preserving Convexity of Sets

Several operations preserve convexity, making it easy to verify that compound sets are convex without returning to the definition each time.

**Proposition (Intersection).** An arbitrary intersection of convex sets is convex. If $C_\alpha$ is convex for every $\alpha \in \mathcal{A}$, then $\bigcap_{\alpha \in \mathcal{A}} C_\alpha$ is convex.

*Proof.* Take $x, y \in \bigcap_\alpha C_\alpha$. Then $x, y \in C_\alpha$ for every $\alpha$, so $\theta x + (1-\theta)y \in C_\alpha$ for every $\alpha$, hence $\theta x + (1-\theta)y \in \bigcap_\alpha C_\alpha$. $\square$

**Proposition (Affine Image).** If $C$ is convex and $f(x) = Ax + b$ is an affine map, then $f(C) = \{Ax + b : x \in C\}$ is convex.

**Proposition (Perspective Map).** The *perspective function* $P : \mathbb{R}^{n+1} \to \mathbb{R}^n$ defined by $P(x, t) = x/t$ (with $t > 0$) maps convex sets to convex sets.

> [!NOTE] Why Intersections Matter
> Feasible sets in optimization are typically intersections of halfspaces (linear inequality constraints), hyperplanes (equality constraints), and level sets of convex functions (convex inequality constraints). The intersection property ensures that any such feasible set is convex — a crucial structural fact exploited throughout this note.

> [!QUESTION] Exercise 1: Convexity of Feasible Sets
> *This problem establishes that the feasible set of a convex optimization problem is always convex, which underlies the local-equals-global property proved in Section 2.*
>
> > **Prerequisites:** [[#1.1 Convex Sets|1.1 Convex Sets]], [[#1.2 Operations Preserving Convexity of Sets|1.2 Operations Preserving Convexity of Sets]]
>
> Let $f_1, \ldots, f_m : \mathbb{R}^n \to \mathbb{R}$ be convex functions and $A \in \mathbb{R}^{p \times n}$, $b \in \mathbb{R}^p$. Define $\mathcal{F} = \{x \in \mathbb{R}^n : f_i(x) \le 0 \text{ for all } i,\; Ax = b\}$. Prove that $\mathcal{F}$ is convex.

> [!TIP]- Solution to Exercise 1
> **Key insight:** Every sublevel set of a convex function is convex, and the feasible set is an intersection of such sublevel sets.
>
> **Sketch:** Each constraint $\{x : f_i(x) \le 0\}$ is the zero-sublevel set of a convex function $f_i$. For any $x, y$ in this set, $f_i(\theta x + (1-\theta)y) \le \theta f_i(x) + (1-\theta)f_i(y) \le 0$ by convexity of $f_i$. So each zero-sublevel set is convex. The equality constraint set $\{x : Ax = b\}$ is an affine subspace, hence convex. The feasible set $\mathcal{F}$ is the intersection of all these convex sets, so it is convex by the Intersection Proposition.

### 1.3 Convex Functions

**Definition (Convex Function).** A function $f : \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$ with convex domain $\operatorname{dom}(f) \subseteq \mathbb{R}^n$ is *convex* if for every $x, y \in \operatorname{dom}(f)$ and $\theta \in [0,1]$:
$$f(\theta x + (1-\theta)y) \le \theta f(x) + (1-\theta) f(y).$$

The function is *strictly convex* if the inequality is strict for $x \ne y$ and $\theta \in (0,1)$.

**Epigraph characterization.** $f$ is convex if and only if $\operatorname{epi}(f)$ is a convex set. This is immediate from the definitions: the inequality above says exactly that the midpoint on the graph lies at or below the chord, which is the set-theoretic statement that $\operatorname{epi}(f)$ is convex.

![[figures/bv2004-fig3.1-convex-function-chord.png]]
*Figure 3.1 (Boyd & Vandenberghe, 2004): Graph of a convex function. The chord (line segment) between any two points $(x, f(x))$ and $(y, f(y))$ on the graph lies above the graph — this is the geometric content of the convexity inequality $f(\theta x + (1-\theta)y) \le \theta f(x) + (1-\theta)f(y)$.*

![[figures/bv2004-fig3.5-epigraph.png]]
*Figure 3.5 (Boyd & Vandenberghe, 2004): The epigraph of a function $f$, shown shaded. The lower boundary (shown darker) is the graph of $f$. The set $\operatorname{epi}(f)$ is convex if and only if $f$ is convex.*

**Definition (Sublevel Set).** The *$\alpha$-sublevel set* of $f$ is $C_\alpha = \{x \in \operatorname{dom}(f) : f(x) \le \alpha\}$. Every sublevel set of a convex function is convex. (*The converse is false: a function with convex sublevel sets need not be convex.*) Functions with convex sublevel sets are called *quasiconvex*.

> [!EXAMPLE] Standard Convex Functions
> - **Affine:** $f(x) = a^\top x + b$ is both convex and concave.
> - **Squared norm:** $f(x) = \|x\|_2^2$ is convex (Hessian $= 2I \succeq 0$).
> - **Negative entropy:** $f(x) = \sum_i x_i \log x_i$ on $\mathbb{R}_{++}^n$ is convex.
> - **Log-sum-exp:** $f(x) = \log \sum_i e^{x_i}$ is convex (a smooth approximation to $\max_i x_i$).
> - **Maximum:** $f(x) = \max(x_1, \ldots, x_n)$ is convex (pointwise max of linear functions).

### 1.4 First-Order and Second-Order Conditions

These conditions give practical tools for checking convexity without appealing to the definition directly.

**Theorem (First-Order Condition).** Let $f$ be differentiable on its (open, convex) domain. Then $f$ is convex if and only if for all $x, y \in \operatorname{dom}(f)$:
$$f(y) \ge f(x) + \nabla f(x)^\top (y - x).$$

This says the graph of $f$ lies above every tangent hyperplane — convex functions have globally supporting hyperplanes at every point.

![[figures/bv2004-fig3.2-first-order-condition.png]]
*Figure 3.2 (Boyd & Vandenberghe, 2004): Geometric content of the first-order condition. The tangent line (hyperplane) at any point $(x, f(x))$ is a global underestimator of $f$: the graph of $f$ lies everywhere above the affine function $f(x) + \nabla f(x)^\top(y - x)$.*

*Proof sketch* ($\Rightarrow$). Fix $x, y$ and let $\theta \in (0,1)$. By convexity, $f(\theta y + (1-\theta)x) \le \theta f(y) + (1-\theta)f(x)$, i.e., $f(x + \theta(y-x)) \le f(x) + \theta(f(y) - f(x))$. Rearranging and dividing by $\theta$:
$$\frac{f(x + \theta(y-x)) - f(x)}{\theta} \le f(y) - f(x).$$
Taking $\theta \to 0^+$, the left side converges to $\nabla f(x)^\top(y-x)$. $\square$

*Proof sketch* ($\Leftarrow$). For $x, y \in \operatorname{dom}(f)$ and $z = \theta x + (1-\theta)y$, apply the tangent inequality at $z$ to both $x$ and $y$, weight by $\theta$ and $(1-\theta)$ respectively, and add.

**Corollary.** If $f$ is convex and $\nabla f(x) = 0$ at some $x$, then $x$ is a global minimum of $f$.

**Theorem (Second-Order Condition).** Let $f$ be twice differentiable on its open, convex domain. Then $f$ is convex if and only if $\nabla^2 f(x) \succeq 0$ for all $x \in \operatorname{dom}(f)$, i.e., the Hessian is positive semidefinite everywhere. If $\nabla^2 f(x) \succ 0$ everywhere (positive definite), then $f$ is strictly convex.

*Proof sketch.* The second-order Taylor expansion $f(y) = f(x) + \nabla f(x)^\top(y-x) + \frac{1}{2}(y-x)^\top \nabla^2 f(z)(y-x)$ for some $z$ on the segment $[x,y]$ shows that $\nabla^2 f \succeq 0$ is exactly the condition that the remainder term is nonnegative, which is equivalent to the first-order condition.

> [!WARNING] Strong Convexity
> A function $f$ is *$\mu$-strongly convex* ($\mu > 0$) if $f(y) \ge f(x) + \nabla f(x)^\top(y-x) + \frac{\mu}{2}\|y-x\|^2$ for all $x, y$. Equivalently, $\nabla^2 f(x) \succeq \mu I$. Strong convexity implies a unique global minimizer — a fact exploited heavily in Section 6. Do not confuse strong convexity with strict convexity: strict convexity does not guarantee a unique minimum in general (though it does guarantee at most one).

> [!QUESTION] Exercise 2: First-Order Condition Implies Optimality
> *This problem sharpens the corollary above: for differentiable convex $f$, vanishing gradient is both necessary and sufficient for global optimality.*
>
> > **Prerequisites:** [[#1.4 First-Order and Second-Order Conditions|1.4 First-Order and Second-Order Conditions]]
>
> Let $f : \mathbb{R}^n \to \mathbb{R}$ be convex and differentiable. Prove rigorously that $x^*$ minimizes $f$ over $\mathbb{R}^n$ if and only if $\nabla f(x^*) = 0$.

> [!TIP]- Solution to Exercise 2
> **Key insight:** The first-order condition turns the global inequality $f(y) \ge f(x^*)$ into a statement about the gradient at $x^*$.
>
> **Sketch:** ($\Rightarrow$) If $x^*$ is a minimizer, then for any direction $d$ and $t > 0$, $f(x^* + td) \ge f(x^*)$. Dividing by $t$ and taking $t \to 0^+$ gives $\nabla f(x^*)^\top d \ge 0$ for all $d$. Applying the same to $-d$ gives $\nabla f(x^*)^\top d = 0$ for all $d$, hence $\nabla f(x^*) = 0$.
>
> ($\Leftarrow$) If $\nabla f(x^*) = 0$, then the first-order condition gives $f(y) \ge f(x^*) + \nabla f(x^*)^\top(y - x^*) = f(x^*)$ for all $y$, so $x^*$ is a global minimizer.

> [!QUESTION] Exercise 3: Convexity via Second-Order Condition
> *This problem gives practice applying the second-order condition and introduces log-sum-exp, a function appearing in softmax-based recommendation models.*
>
> > **Prerequisites:** [[#1.4 First-Order and Second-Order Conditions|1.4 First-Order and Second-Order Conditions]]
>
> Prove that $f(x) = \log\bigl(\sum_{i=1}^n e^{x_i}\bigr)$ is convex by computing its Hessian and showing it is positive semidefinite. Let $p_i = e^{x_i} / \sum_j e^{x_j}$; express the Hessian in terms of $p$.

> [!TIP]- Solution to Exercise 3
> **Key insight:** The Hessian equals $\operatorname{diag}(p) - pp^\top$, which is a covariance matrix of a discrete distribution — hence PSD.
>
> **Sketch:** Compute $\partial f/\partial x_i = p_i$ and $\partial^2 f/\partial x_i^2 = p_i(1-p_i)$, $\partial^2 f/\partial x_i \partial x_j = -p_i p_j$ for $i \ne j$. So $\nabla^2 f = \operatorname{diag}(p) - pp^\top$. For any $v \in \mathbb{R}^n$: $v^\top(\operatorname{diag}(p) - pp^\top)v = \sum_i p_i v_i^2 - (\sum_i p_i v_i)^2 \ge 0$ by Cauchy-Schwarz (or Jensen's inequality applied to $g(v) = v^2$ under the distribution $p$). Hence $\nabla^2 f \succeq 0$.

### 1.5 Operations Preserving Convexity of Functions

**Non-negative weighted sum.** If $f_1, \ldots, f_k$ are convex and $w_i \ge 0$, then $\sum_i w_i f_i$ is convex.

**Composition with affine map.** If $f$ is convex and $g(x) = Ax + b$, then $f \circ g$ is convex.

**Pointwise maximum.** If $\{f_\alpha\}_{\alpha \in \mathcal{A}}$ are all convex, then $f(x) = \sup_\alpha f_\alpha(x)$ is convex (with appropriate domain). This is a consequence of the epigraph being the intersection $\bigcap_\alpha \operatorname{epi}(f_\alpha)$.

**Pointwise infimum.** *Crucially for duality*: $g(y) = \inf_x f(x, y)$ is convex in $y$ if $f(x, y)$ is jointly convex in $(x, y)$. We will use this in Section 3.3 when showing the dual function is concave.

---

## 2. Convex Optimization Problems 🔑

### 2.1 Standard Form

**Definition (Convex Optimization Problem, Standard Form).** A *convex optimization problem* in standard form is:
$$\begin{aligned}
\text{minimize} \quad & f_0(x) \\
\text{subject to} \quad & f_i(x) \le 0, \quad i = 1, \ldots, m \\
& h_j(x) = 0, \quad j = 1, \ldots, p
\end{aligned}$$
where $x \in \mathbb{R}^n$ is the *optimization variable*, $f_0, f_1, \ldots, f_m : \mathbb{R}^n \to \mathbb{R}$ are convex functions, and $h_1, \ldots, h_p : \mathbb{R}^n \to \mathbb{R}$ are affine functions $h_j(x) = a_j^\top x - b_j$.

The requirement that equality constraints are affine is not merely a convention: a convex equality constraint $h(x) = 0$ with $h$ nonlinear forces $x$ to a point (the constraint set $\{h = 0\} \cap \{h \ge 0\} = \{h = 0, h \le 0\}$ collapses) and is never convex unless $h$ is affine.

The *feasible set* is $\mathcal{F} = \{x : f_i(x) \le 0,\; h_j(x) = 0\;\forall\,i,j\}$. By Exercise 1, $\mathcal{F}$ is convex. The *optimal value* is $p^* = \inf_{x \in \mathcal{F}} f_0(x)$.

> [!NOTE] Notation for Optimal Value
> We allow $p^* = +\infty$ (infeasible problem) and $p^* = -\infty$ (unbounded below). A point $x^*$ is *optimal* if it is feasible and $f_0(x^*) = p^*$. A point $x$ is *$\varepsilon$-suboptimal* if it is feasible and $f_0(x) \le p^* + \varepsilon$.

### 2.2 Special Cases

Many problems encountered in machine learning and information retrieval are convex problems in disguise. 💡

**Linear Program (LP).** $f_0(x) = c^\top x$, $f_i(x) = a_i^\top x - b_i$, $h_j(x) = e_j^\top x - d_j$. All functions are affine, and the feasible set is a polyhedron.

**Quadratic Program (QP).** $f_0(x) = \frac{1}{2}x^\top P x + q^\top x + r$ with $P \succeq 0$, and affine inequality and equality constraints.

**Second-Order Cone Program (SOCP).** The constraint $\|Ax + b\|_2 \le c^\top x + d$ defines a *second-order cone* (also called an ice-cream cone or Lorentz cone). SOCPs include LPs and QPs as special cases.

**Recommendation as a Constrained QP.** In the motivating paper by [Agarwal et al. (SIGIR 2012)](https://dl.acm.org/doi/10.1145/2348283.2348350), the click-shaping problem has the form: maximize an expected click objective (a linear or quadratic function of an allocation vector) subject to constraints enforcing minimum engagement on certain content categories. When the click model uses a regularized log-likelihood or Gaussian prior, the objective becomes a convex quadratic, and the whole problem is a QP admitting strong duality.

### 2.3 Local Equals Global

The most important property of convex optimization is that every local minimum is a global minimum.

**Theorem.** Let $f_0$ be convex and $\mathcal{F}$ be the (convex) feasible set of a convex optimization problem. If $x^*$ is a *local minimum* — there exists $\delta > 0$ such that $f_0(x^*) \le f_0(y)$ for all feasible $y$ with $\|y - x^*\| \le \delta$ — then $x^*$ is a *global minimum*.

*Proof.* Suppose for contradiction that some feasible $z$ satisfies $f_0(z) < f_0(x^*)$. Consider the point $y_\theta = \theta z + (1-\theta)x^*$ for $\theta \in (0,1)$. Since $\mathcal{F}$ is convex, $y_\theta$ is feasible. By convexity of $f_0$:
$$f_0(y_\theta) \le \theta f_0(z) + (1-\theta)f_0(x^*) < \theta f_0(x^*) + (1-\theta)f_0(x^*) = f_0(x^*).$$
For sufficiently small $\theta$, $\|y_\theta - x^*\| = \theta\|z - x^*\| < \delta$, contradicting local optimality of $x^*$. $\square$

**Corollary.** If $f_0$ is strictly convex, the global minimum is unique.

> [!QUESTION] Exercise 4: The Optimality Condition for Constrained Convex Problems
> *This problem establishes the constrained analogue of "vanishing gradient implies optimality" — a result used in the KKT stationarity condition.*
>
> > **Prerequisites:** [[#1.4 First-Order and Second-Order Conditions|1.4 First-Order and Second-Order Conditions]], [[#2.3 Local Equals Global|2.3 Local Equals Global]]
>
> Let $f_0$ be convex and differentiable, and let $\mathcal{F}$ be a convex feasible set. Show that $x^*$ minimizes $f_0$ over $\mathcal{F}$ if and only if for every feasible $y \in \mathcal{F}$: $\nabla f_0(x^*)^\top(y - x^*) \ge 0$.

> [!TIP]- Solution to Exercise 4
> **Key insight:** This is the variational inequality characterizing constrained optimality. It says every feasible direction from $x^*$ is a non-descent direction.
>
> **Sketch:** ($\Leftarrow$) The first-order condition gives $f_0(y) \ge f_0(x^*) + \nabla f_0(x^*)^\top(y-x^*) \ge f_0(x^*)$ for all feasible $y$, so $x^*$ is optimal.
>
> ($\Rightarrow$) If $x^*$ is optimal and $y$ is feasible, consider $y_t = (1-t)x^* + ty$ feasible for $t \in [0,1]$. Since $x^*$ is optimal, $f_0(y_t) \ge f_0(x^*)$. The difference quotient $(f_0(y_t) - f_0(x^*))/t \ge 0$, and taking $t \to 0^+$ gives $\nabla f_0(x^*)^\top(y - x^*) \ge 0$.

---

## 3. The Lagrangian 📐

The Lagrangian is the central object of convex duality. It relaxes the constrained problem into an unconstrained one by "pricing" constraint violations.

### 3.1 Definition and Lower-Bound Interpretation

Consider the standard form problem from Section 2.1. Introduce *dual variables* (also called *Lagrange multipliers*): $\lambda = (\lambda_1, \ldots, \lambda_m) \in \mathbb{R}^m$ for the inequality constraints and $\nu = (\nu_1, \ldots, \nu_p) \in \mathbb{R}^p$ for the equality constraints.

**Definition (Lagrangian).** The *Lagrangian* $L : \mathbb{R}^n \times \mathbb{R}^m \times \mathbb{R}^p \to \mathbb{R}$ is:
$$L(x, \lambda, \nu) = f_0(x) + \sum_{i=1}^m \lambda_i f_i(x) + \sum_{j=1}^p \nu_j h_j(x).$$

The variables $(\lambda, \nu)$ are called *dual variables*; the vector $\lambda$ must satisfy $\lambda \succeq 0$ (componentwise) to retain the lower-bound property. The variable $x$ is the *primal variable*.

**Key observation: lower bound for feasible $x$.** Fix any $\lambda \succeq 0$ and any $\nu$. For any feasible point $\tilde{x}$ (i.e., $f_i(\tilde{x}) \le 0$ and $h_j(\tilde{x}) = 0$):
$$L(\tilde{x}, \lambda, \nu) = f_0(\tilde{x}) + \underbrace{\sum_i \lambda_i f_i(\tilde{x})}_{\le 0} + \underbrace{\sum_j \nu_j h_j(\tilde{x})}_{= 0} \le f_0(\tilde{x}).$$
So $L(\tilde{x}, \lambda, \nu) \le f_0(\tilde{x})$ for any feasible $\tilde{x}$ and any $\lambda \succeq 0$. This is the lower-bound property that drives the entire duality theory.

> [!INFO] Interpretation as Penalized Objective
> The Lagrangian can be read as: start with the objective $f_0(x)$ and add a *penalty* $\sum_i \lambda_i f_i(x)$ for constraint violations. When $f_i(x) > 0$ (constraint violated), the term $\lambda_i f_i(x) > 0$ increases the Lagrangian. When $f_i(x) \le 0$ (feasible), the term $\lambda_i f_i(x) \le 0$ decreases it. The dual variable $\lambda_i$ is the *price* per unit of constraint violation — a Pigouvian tax on infeasibility.

### 3.2 The Dual Function

**Definition (Lagrange Dual Function).** The *Lagrange dual function* $g : \mathbb{R}^m \times \mathbb{R}^p \to \mathbb{R} \cup \{-\infty\}$ is:
$$g(\lambda, \nu) = \inf_{x \in \mathcal{D}} L(x, \lambda, \nu) = \inf_{x \in \mathcal{D}} \left\{ f_0(x) + \sum_{i=1}^m \lambda_i f_i(x) + \sum_{j=1}^p \nu_j h_j(x) \right\}$$
where $\mathcal{D} = \operatorname{dom}(f_0) \cap \bigcap_i \operatorname{dom}(f_i)$ is the natural domain of the problem.

Note that the infimum is taken over all $x \in \mathcal{D}$, not just feasible $x$. This is the *Lagrangian relaxation*: we drop the constraints and instead penalize violations.

**Claim.** For any $\lambda \succeq 0$ and any $\nu$: $g(\lambda, \nu) \le p^*$.

*Proof.* Let $x^*$ be any optimal point (if none exists, $p^* = \inf_{x\in\mathcal{F}} f_0(x)$ and the argument applies to any sequence approaching it). Then:
$$g(\lambda, \nu) = \inf_{x \in \mathcal{D}} L(x, \lambda, \nu) \le L(x^*, \lambda, \nu) \le f_0(x^*) = p^*$$
where the second inequality uses the lower-bound property from Section 3.1. $\square$

So the dual function provides a *certificate* that $p^*$ is at least $g(\lambda, \nu)$.

> [!QUESTION] Exercise 5: Dual Function of an LP
> *This problem computes the dual function of a linear program, revealing the characteristic structure (finite or $-\infty$) and motivating the LP dual.*
>
> > **Prerequisites:** [[#3.2 The Dual Function|3.2 The Dual Function]]
>
> Consider the LP: minimize $c^\top x$ subject to $Ax \preceq b$, $x \succeq 0$. Compute $g(\lambda)$ explicitly and show that $g(\lambda) = -\infty$ unless $A^\top \lambda + c \succeq 0$.

> [!TIP]- Solution to Exercise 5
> **Key insight:** The infimum of an affine function over all of $\mathbb{R}^n$ is $-\infty$ unless the linear coefficient is zero.
>
> **Sketch:** The Lagrangian is $L(x, \lambda, \mu) = c^\top x + \lambda^\top(Ax - b) - \mu^\top x = (c + A^\top\lambda - \mu)^\top x - \lambda^\top b$ where $\mu \succeq 0$ handles $x \succeq 0$. The infimum over $x \in \mathbb{R}^n$ is $-\infty$ unless $c + A^\top\lambda - \mu = 0$, i.e., $\mu = c + A^\top\lambda$. For feasibility $\mu \succeq 0$ requires $A^\top\lambda + c \succeq 0$. When this holds, $g(\lambda) = -\lambda^\top b$. The dual LP is thus: maximize $-b^\top\lambda$ subject to $A^\top\lambda + c \succeq 0$, $\lambda \succeq 0$.

### 3.3 Concavity of the Dual Function

The dual function is always concave in $(\lambda, \nu)$, regardless of whether the original problem is convex.

**Theorem.** $g(\lambda, \nu) = \inf_x L(x, \lambda, \nu)$ is a concave function of $(\lambda, \nu)$ on $\{(\lambda,\nu) : g(\lambda,\nu) > -\infty\}$.

*Proof.* Fix $x \in \mathcal{D}$. The function $(\lambda, \nu) \mapsto L(x, \lambda, \nu) = f_0(x) + \sum_i \lambda_i f_i(x) + \sum_j \nu_j h_j(x)$ is *affine* (hence concave) in $(\lambda, \nu)$. The dual function $g(\lambda, \nu) = \inf_x L(x, \lambda, \nu)$ is a pointwise infimum of affine (hence concave) functions, and pointwise infima of concave functions are concave. $\square$

This is a fundamental structural fact: **the dual function is always concave and the dual problem (Section 4.1) is always convex, even when the primal is not.** This means we can apply all the machinery of convex optimization to the dual.

> [!QUESTION] Exercise 6: Dual Function of a QP
> *This problem computes the dual function of a quadratic program, a calculation central to support vector machines and regularized recommendation.*
>
> > **Prerequisites:** [[#3.2 The Dual Function|3.2 The Dual Function]], [[#3.3 Concavity of the Dual Function|3.3 Concavity of the Dual Function]]
>
> Let $P \succ 0$, $q \in \mathbb{R}^n$, $A \in \mathbb{R}^{m \times n}$, $b \in \mathbb{R}^m$. Consider: minimize $\frac{1}{2}x^\top Px + q^\top x$ subject to $Ax \preceq b$. Form the Lagrangian, minimize over $x$ in closed form, and express $g(\lambda)$ as a quadratic in $\lambda$.

> [!TIP]- Solution to Exercise 6
> **Key insight:** The Lagrangian is strongly convex in $x$ when $P \succ 0$, so it has a unique minimizer expressible in closed form via $\nabla_x L = 0$.
>
> **Sketch:** $L(x,\lambda) = \frac{1}{2}x^\top Px + q^\top x + \lambda^\top(Ax - b)$. Setting $\nabla_x L = Px + q + A^\top\lambda = 0$ gives $x^*(\lambda) = -P^{-1}(q + A^\top\lambda)$. Substituting back: $g(\lambda) = -\frac{1}{2}(q + A^\top\lambda)^\top P^{-1}(q + A^\top\lambda) - b^\top\lambda$. This is a concave quadratic in $\lambda$, confirming the concavity theorem.

---

## 4. Lagrangian Duality 🔑

### 4.1 The Dual Problem

The dual function provides a lower bound on $p^*$ for every $\lambda \succeq 0$ and $\nu$. We naturally want the *best* lower bound, i.e., the largest value of $g$.

**Definition (Lagrange Dual Problem).** The *Lagrange dual problem* is:
$$\begin{aligned}
\text{maximize} \quad & g(\lambda, \nu) \\
\text{subject to} \quad & \lambda \succeq 0.
\end{aligned}$$

The optimal value of the dual problem is $d^* = \sup_{\lambda \succeq 0, \nu} g(\lambda, \nu)$.

Since $g$ is concave and $\lambda \succeq 0$ defines a convex constraint, the dual problem is a *convex optimization problem* — even when the primal is not. The dual variables $(\lambda, \nu)$ are called *dual feasible* if $\lambda \succeq 0$ (and $g(\lambda,\nu) > -\infty$). A pair $(\lambda^*, \nu^*)$ is *dual optimal* if it achieves $d^*$.

> [!NOTE] Asymmetry Between Primal and Dual Constraints
> The primal has both inequality constraints ($f_i \le 0$) and equality constraints ($h_j = 0$). The dual has only the non-negativity constraint $\lambda \succeq 0$ on the multipliers for inequality constraints; the $\nu_j$ for equality constraints are unconstrained. This asymmetry reflects the asymmetry of the original problem.

### 4.2 Weak Duality

**Theorem (Weak Duality).** Always $d^* \le p^*$.

*Proof.* For any $\lambda \succeq 0$, any $\nu$, and any feasible $\tilde{x}$:
$$g(\lambda, \nu) \le L(\tilde{x}, \lambda, \nu) \le f_0(\tilde{x}).$$
The first inequality holds because $g$ is an infimum. The second holds because $f_i(\tilde{x}) \le 0$, $h_j(\tilde{x}) = 0$, and $\lambda \succeq 0$, so the extra terms in $L$ are non-positive. Taking the infimum over feasible $\tilde{x}$ on the right gives $g(\lambda, \nu) \le p^*$. Since this holds for all dual-feasible $(\lambda, \nu)$, taking the supremum over dual-feasible points on the left gives $d^* \le p^*$. $\square$

![[figures/bv2004-fig5.3-weak-duality-lower-bound.png]]
*Figure 5.3 (Boyd & Vandenberghe, 2004): Geometric interpretation of weak duality. The set $\mathcal{G} = \{(f_1(x), f_0(x)) : x \in \mathcal{D}\}$ is shown. The dual function value $g(\lambda)$ equals the $t$-intercept of the supporting hyperplane $\lambda u + t = g(\lambda)$ with slope $-\lambda$. The hyperplane always lies at or below $p^*$ (the infimum of $t$ over $u \le 0$), establishing $g(\lambda) \le p^*$.*

**Corollary (Certificate of Suboptimality).** If $\tilde{x}$ is primal feasible and $(\tilde{\lambda}, \tilde{\nu})$ is dual feasible, then:
$$f_0(\tilde{x}) - p^* \le f_0(\tilde{x}) - g(\tilde{\lambda}, \tilde{\nu}).$$
The right-hand side is computable without knowing $p^*$ and provides an upper bound on how suboptimal $\tilde{x}$ is. In practice, optimization algorithms use this *duality gap* as a stopping criterion.

> [!QUESTION] Exercise 7: Weak Duality for Nonconvex Problems
> *Weak duality holds universally — this problem makes that explicit by applying it to a nonconvex integer program.*
>
> > **Prerequisites:** [[#4.2 Weak Duality|4.2 Weak Duality]]
>
> Consider the integer program: minimize $x_1^2 + x_2^2$ subject to $x_1 + x_2 \ge 3$, $x_1, x_2 \in \mathbb{Z}_{\ge 0}$. (a) Find $p^*$. (b) Form the Lagrangian with multiplier $\lambda \ge 0$ for the constraint $3 - x_1 - x_2 \le 0$. Compute $g(\lambda)$ by minimizing over $x \in \mathbb{Z}_{\ge 0}^2$. (c) Find $d^*$ and confirm $d^* \le p^*$.

> [!TIP]- Solution to Exercise 7
> **Key insight:** The duality gap $p^* - d^*$ for integer programs can be strictly positive — this is the "integrality gap."
>
> **Sketch:** (a) The feasible points with $x_1 + x_2 = 3$ and minimal $x_1^2 + x_2^2$ are $(1,2)$ or $(2,1)$ both giving $f_0 = 5$, and $(0,3)$ or $(3,0)$ giving $f_0 = 9$. So $p^* = 5$.
>
> (b) $L(x,\lambda) = x_1^2 + x_2^2 + \lambda(3 - x_1 - x_2)$. Minimizing over $x_i \in \mathbb{Z}_{\ge 0}$ independently: the minimum of $x_i^2 - \lambda x_i$ over $x_i \in \mathbb{Z}_{\ge 0}$ is achieved at $x_i = \lfloor \lambda/2 \rceil$ (nearest integer to $\lambda/2$). For $\lambda = 2$: $g(2) = 4 - 4 + 6 = 6 > 5$. Wait, $g(\lambda) = 3\lambda + 2\min_{x_i \ge 0, x_i \in \mathbb{Z}}(x_i^2 - \lambda x_i)$. At $\lambda = 2$, the minimizer $x_i = 1$ gives $1 - 2 = -1$, so $g(2) = 6 + 2(-1) = 4 \le 5 = p^*$. Optimizing over $\lambda$ gives $d^* \le 4.5$. Confirm $d^* \le p^* = 5$. ✓

### 4.3 The Duality Gap

**Definition (Duality Gap).** The *duality gap* is $p^* - d^* \ge 0$.

When the duality gap is zero ($p^* = d^*$), the dual provides exactly as tight a lower bound as possible. This is *strong duality*, and it is the condition that makes the dual problem useful for computing primal solutions.

The duality gap has a beautiful geometric interpretation. Consider the set:
$$\mathcal{G} = \{(u, t) \in \mathbb{R}^m \times \mathbb{R} : \exists x \in \mathcal{D},\; f_i(x) \le u_i\;\forall i,\; f_0(x) \le t\}$$
(ignoring equality constraints for simplicity). The optimal value $p^*$ is the infimum of $t$ over points in $\mathcal{G}$ with $u \preceq 0$. The dual function value $g(\lambda)$ is $-\lambda^\top u^* + (p^* \text{ contribution})$, corresponding to a supporting hyperplane to $\mathcal{G}$ at the point $(0, p^*)$. Strong duality holds when there is a *non-vertical* supporting hyperplane at $(0, p^*)$, which is guaranteed by Slater's condition.

![[figures/bv2004-fig5.4-duality-gap.png]]
*Figure 5.4 (Boyd & Vandenberghe, 2004): The duality gap when strong duality fails. Three supporting hyperplanes (for dual values $\lambda_1, \lambda^*, \lambda_2$) are shown tangent to $\mathcal{G}$. The best dual lower bound $d^*$ is achieved at $\lambda^*$, but $d^* < p^*$ — the hyperplane at $\lambda^*$ cannot be made to pass through $(0, p^*)$ without becoming vertical. This illustrates why strong duality requires Slater's condition.*

### 4.4 Strong Duality and Slater's Condition

**Theorem (Slater's Condition implies Strong Duality).** Suppose the primal problem is convex ($f_0, f_1, \ldots, f_m$ are convex, $h_j$ are affine). If there exists a point $\tilde{x} \in \operatorname{relint}(\mathcal{D})$ such that:
$$f_i(\tilde{x}) < 0 \text{ for all } i = 1, \ldots, m, \quad \text{and} \quad h_j(\tilde{x}) = 0 \text{ for all } j = 1, \ldots, p,$$
then strong duality holds: $d^* = p^*$, and the dual optimal value is attained (there exists $(\lambda^*, \nu^*)$ with $g(\lambda^*, \nu^*) = d^*$).

Such a point $\tilde{x}$ is called *Slater feasible* or *strictly feasible*.

> [!NOTE] Refined Slater's Condition
> For constraints that are already affine (linear), strict feasibility is not required — they may hold with equality at $\tilde{x}$. Only the nonlinear convex constraints need to be strictly satisfied. This refinement matters in practice: mixed linear/nonlinear constraint sets appear frequently in recommendation problems.

*Proof sketch (via supporting hyperplane theorem).* The proof proceeds by showing the set $\mathcal{G}$ (defined in Section 4.3) is a convex set, and $(0, p^*)$ is a boundary point. The supporting hyperplane theorem guarantees a supporting hyperplane at $(0, p^*)$. Slater's condition rules out a vertical supporting hyperplane (which would give no useful dual), so the hyperplane has the form $\{(u, t) : \lambda^\top u + t \ge p^*\}$ for some $\lambda \succeq 0$, $\mu > 0$. Normalizing $\mu = 1$ and unpacking gives $g(\lambda, \nu) = p^*$ for some $\nu$. $\square$

![[figures/bv2004-fig5.6-slater-strong-duality.png]]
*Figure 5.6 (Boyd & Vandenberghe, 2004): Geometric proof of strong duality under Slater's condition. The convex set $\mathcal{A}$ (shaded) and the vertical segment $\mathcal{B}$ (below $(0, p^*)$) are disjoint and can be separated by a hyperplane. Slater's condition guarantees this separating hyperplane is non-vertical — it must pass to the left of the strictly feasible point $(\tilde{u}, \tilde{t}) = (f_1(\tilde{x}), f_0(\tilde{x}))$ with $\tilde{u} < 0$ — yielding a dual certificate with $g(\lambda^*, \nu^*) = p^*$.*

> [!WARNING] When Slater's Condition Fails
> Slater's condition is sufficient but not necessary for strong duality. It can fail for feasible problems — e.g., if all constraints are equality constraints (no inequality constraints to satisfy strictly). When it fails, the duality gap may be strictly positive even for convex problems. Always verify Slater's condition before assuming $d^* = p^*$.

> [!QUESTION] Exercise 8: Strong Duality for Equality-Constrained QP
> *This problem verifies strong duality directly for a QP with only equality constraints, where Slater's condition is vacuously satisfied.*
>
> > **Prerequisites:** [[#4.4 Strong Duality and Slater's Condition|4.4 Strong Duality and Slater's Condition]], [[#3.2 The Dual Function|3.2 The Dual Function]]
>
> Consider: minimize $\frac{1}{2}\|x\|^2$ subject to $Ax = b$ where $A \in \mathbb{R}^{p \times n}$ has full row rank. (a) Solve the primal directly using the KKT conditions (derived in Section 5). (b) Compute the dual function $g(\nu)$ and find $d^*$. (c) Verify $p^* = d^*$.

> [!TIP]- Solution to Exercise 8
> **Key insight:** For an equality-constrained QP, strong duality is equivalent to the Lagrangian having a unique minimizer, which exists whenever the objective is strongly convex.
>
> **Sketch:** (a) $\nabla_x(\frac{1}{2}\|x\|^2 + \nu^\top(Ax-b)) = x + A^\top\nu = 0$ gives $x^* = -A^\top\nu^*$. From $Ax^* = b$: $-AA^\top\nu^* = b$, so $\nu^* = -(AA^\top)^{-1}b$ (full row rank ensures invertibility). Then $x^* = A^\top(AA^\top)^{-1}b$, the pseudoinverse. $p^* = \frac{1}{2}b^\top(AA^\top)^{-1}b$.
>
> (b) $g(\nu) = \inf_x(\frac{1}{2}\|x\|^2 + \nu^\top(Ax-b)) = -\frac{1}{2}\|A^\top\nu\|^2 - b^\top\nu$. Maximizing over $\nu$: $\nabla_\nu g = -AA^\top\nu - b = 0$ gives $\nu^* = -(AA^\top)^{-1}b$ and $d^* = -\frac{1}{2}b^\top(AA^\top)^{-1}b = p^*$. ✓

> [!QUESTION] Exercise 9: Verifying Slater's Condition
> *This problem develops intuition for when Slater's condition holds and when it fails.*
>
> > **Prerequisites:** [[#4.4 Strong Duality and Slater's Condition|4.4 Strong Duality and Slater's Condition]]
>
> For each of the following problems, determine whether Slater's condition holds, and state whether strong duality is guaranteed: (a) minimize $e^x$ subject to $x^2 \le 1$; (b) minimize $x$ subject to $x^2 \le 0$; (c) minimize $x_1 + x_2$ subject to $x_1^2 + x_2^2 \le 1$, $x_1 = 0$.

> [!TIP]- Solution to Exercise 9
> **Key insight:** Slater's condition requires a strictly feasible point — all inequality constraints strictly satisfied, all equality constraints exactly satisfied.
>
> **Sketch:** (a) $\tilde{x} = 0$ satisfies $0^2 = 0 < 1$. Slater holds; strong duality guaranteed. (b) The only feasible point is $x = 0$ ($x^2 \le 0$ and $x \in \mathbb{R}$ forces $x = 0$). No strictly feasible point. However, the constraint is active everywhere on the feasible set, which is a single point. Strong duality holds trivially ($p^* = d^* = 0$) but Slater fails. (c) The equality $x_1 = 0$ and $x_1^2 + x_2^2 \le 1$ allows the strictly feasible point $(0, 0)$ for the inequality (since $0 < 1$), and it satisfies the equality. Slater (refined version) holds; strong duality guaranteed.

---

## 5. KKT Conditions 🔑

The *Karush-Kuhn-Tucker* (KKT) conditions are the central optimality conditions for constrained optimization. For convex problems satisfying Slater's condition, they are both necessary and sufficient.

### 5.1 Statement of the Conditions

**Definition (KKT Conditions).** Let $x^*$ be a primal point and $(\lambda^*, \nu^*)$ a dual point. The four KKT conditions are:

1. **Primal feasibility:** $f_i(x^*) \le 0$ for all $i$; $h_j(x^*) = 0$ for all $j$.

2. **Dual feasibility:** $\lambda_i^* \ge 0$ for all $i$.

3. **Complementary slackness:** $\lambda_i^* f_i(x^*) = 0$ for all $i$.

4. **Stationarity:** $\nabla f_0(x^*) + \sum_{i=1}^m \lambda_i^* \nabla f_i(x^*) + \sum_{j=1}^p \nu_j^* \nabla h_j(x^*) = 0$.

Complementary slackness says: either constraint $i$ is active ($f_i(x^*) = 0$) or its multiplier is zero ($\lambda_i^* = 0$). Both can be zero; what is excluded is $\lambda_i^* > 0$ with $f_i(x^*) < 0$.

### 5.2 Necessity

**Theorem (Necessity of KKT).** Suppose strong duality holds (e.g., the problem is convex and Slater's condition is satisfied), and suppose $x^*$ is primal optimal and $(\lambda^*, \nu^*)$ is dual optimal, and $f_0, f_1, \ldots, f_m, h_1, \ldots, h_p$ are all differentiable. Then $x^*$ and $(\lambda^*, \nu^*)$ satisfy the four KKT conditions.

*Proof.* Primal and dual feasibility hold by assumption. It remains to derive stationarity and complementary slackness from $p^* = d^*$.

Since $x^*$ is optimal and $(\lambda^*, \nu^*)$ is dual optimal:
$$f_0(x^*) = p^* = d^* = g(\lambda^*, \nu^*) = \inf_x L(x, \lambda^*, \nu^*) \le L(x^*, \lambda^*, \nu^*).$$
But also by primal feasibility and $\lambda^* \succeq 0$:
$$L(x^*, \lambda^*, \nu^*) = f_0(x^*) + \underbrace{\sum_i \lambda_i^* f_i(x^*)}_{\le 0} + \underbrace{\sum_j \nu_j^* h_j(x^*)}_{= 0} \le f_0(x^*).$$
Combining: $f_0(x^*) \le L(x^*, \lambda^*, \nu^*) \le f_0(x^*)$, so $L(x^*, \lambda^*, \nu^*) = f_0(x^*)$. This forces:
$$\sum_i \lambda_i^* f_i(x^*) = 0.$$
Since $\lambda_i^* \ge 0$ and $f_i(x^*) \le 0$ for each $i$, each term is non-positive; the sum is zero only if each term is zero. Hence $\lambda_i^* f_i(x^*) = 0$ for all $i$ — complementary slackness.

Moreover, the chain of equalities shows $x^*$ minimizes $L(\cdot, \lambda^*, \nu^*)$ over $\mathcal{D}$. Since $L(\cdot, \lambda^*, \nu^*)$ is differentiable, its minimizer satisfies $\nabla_x L(x^*, \lambda^*, \nu^*) = 0$ — stationarity. $\square$

### 5.3 Sufficiency

**Theorem (Sufficiency of KKT for Convex Problems).** Suppose $f_0, f_1, \ldots, f_m$ are convex and $h_1, \ldots, h_p$ are affine. If $\tilde{x}$ and $(\tilde{\lambda}, \tilde{\nu})$ satisfy all four KKT conditions, then $\tilde{x}$ is primal optimal and $(\tilde{\lambda}, \tilde{\nu})$ is dual optimal, with zero duality gap.

*Proof.* By stationarity, $\tilde{x}$ minimizes $L(\cdot, \tilde{\lambda}, \tilde{\nu})$, so $g(\tilde{\lambda}, \tilde{\nu}) = L(\tilde{x}, \tilde{\lambda}, \tilde{\nu})$. By complementary slackness and primal feasibility:
$$g(\tilde{\lambda}, \tilde{\nu}) = L(\tilde{x}, \tilde{\lambda}, \tilde{\nu}) = f_0(\tilde{x}) + \underbrace{\sum_i \tilde{\lambda}_i f_i(\tilde{x})}_{=0} + \underbrace{\sum_j \tilde{\nu}_j h_j(\tilde{x})}_{=0} = f_0(\tilde{x}).$$
Thus $f_0(\tilde{x}) = g(\tilde{\lambda}, \tilde{\nu}) \le d^* \le p^* \le f_0(\tilde{x})$, where the last inequality uses primal feasibility of $\tilde{x}$. All inequalities are equalities, so $\tilde{x}$ is primal optimal, $(\tilde{\lambda}, \tilde{\nu})$ is dual optimal, and $p^* = d^*$. $\square$

> [!INFO] Summary: KKT and Slater Together
> For a convex problem satisfying Slater's condition:
> - $x^*$ is primal optimal and $(\lambda^*, \nu^*)$ is dual optimal with zero gap $\iff$ all four KKT conditions hold.
> This bidirectional equivalence makes KKT the central computational tool: solve the four-condition system to find both the primal and dual optimal solutions simultaneously.

### 5.4 Geometric Interpretation

The stationarity condition $\nabla f_0(x^*) = -\sum_i \lambda_i^* \nabla f_i(x^*) - \sum_j \nu_j^* \nabla h_j(x^*)$ has a clear geometric meaning: the negative gradient of the objective (the direction of steepest descent) must lie in the cone spanned by the gradients of the active constraints.

Intuitively: at the optimum, every direction that decreases $f_0$ also increases some active constraint $f_i$ (i.e., moves outside the feasible set). The multipliers $\lambda_i^*$ quantify how much the active constraints resist the descent direction.

Complementary slackness encodes which constraints are "binding": inactive constraints ($f_i(x^*) < 0$) have zero multiplier, meaning they are not contributing to the balance of forces.

> [!EXAMPLE] KKT in a Simple QP
> Consider: minimize $\frac{1}{2}(x_1^2 + x_2^2)$ subject to $x_1 + x_2 \ge 1$ (rewritten as $1 - x_1 - x_2 \le 0$).
>
> KKT conditions with multiplier $\lambda \ge 0$ for the constraint $1 - x_1 - x_2 \le 0$:
> - Stationarity: $x_1 - \lambda = 0$, $x_2 - \lambda = 0$, so $x_1 = x_2 = \lambda$.
> - Complementary slackness: $\lambda(1 - x_1 - x_2) = 0$.
> - If $\lambda > 0$: constraint is active, $x_1 + x_2 = 1$, so $2\lambda = 1$, $\lambda = 1/2$, $x_1 = x_2 = 1/2$.
> - If $\lambda = 0$: $x_1 = x_2 = 0$, but $0 + 0 = 0 < 1$ violates feasibility.
>
> So $x^* = (1/2, 1/2)$, $\lambda^* = 1/2$, $p^* = 1/4$.

> [!QUESTION] Exercise 10: KKT for a Constrained QP with Two Constraints
> *This problem practices applying all four KKT conditions and reading off complementary slackness to determine which constraints are active.*
>
> > **Prerequisites:** [[#5.1 Statement of the Conditions|5.1 Statement of the Conditions]], [[#5.3 Sufficiency|5.3 Sufficiency]]
>
> Solve via KKT: minimize $x_1^2 + 2x_2^2$ subject to $x_1 + x_2 \le 1$ and $x_1 \ge 0$, $x_2 \ge 0$. Find all KKT points and identify which is the global minimum.

> [!TIP]- Solution to Exercise 10
> **Key insight:** Check all combinations of which constraints are active (active set enumeration), then verify primal and dual feasibility.
>
> **Sketch:** Let $\lambda_1 \ge 0$ for $x_1 + x_2 - 1 \le 0$, $\mu_1 \ge 0$ for $-x_1 \le 0$, $\mu_2 \ge 0$ for $-x_2 \le 0$. Stationarity: $2x_1 + \lambda_1 - \mu_1 = 0$, $4x_2 + \lambda_1 - \mu_2 = 0$.
>
> Case 1: $x_1 > 0$, $x_2 > 0$, constraint inactive ($\lambda_1 = 0$, $\mu_1 = \mu_2 = 0$). Then $x_1 = x_2 = 0$ — contradicts $x_1, x_2 > 0$.
>
> Case 2: Constraint active, $x_1 + x_2 = 1$, $x_1, x_2 > 0$. Then $\mu_1 = \mu_2 = 0$, $2x_1 + \lambda_1 = 0$ — forces $\lambda_1 < 0$, violating dual feasibility unless $x_1 = 0$.
>
> Case 3: $x_1 = 0$, constraint active. $x_2 = 1$, $\lambda_1 \ge 0$: stationarity gives $\lambda_1 = \mu_1 \ge 0$ and $4 + \lambda_1 = \mu_2 \ge 0$. Both satisfied with $\lambda_1 = 0$, $\mu_1 = 0$, $\mu_2 = 4$. KKT point: $x^* = (0, 1)$, $f_0 = 2$.
>
> Case 4: $x_2 = 0$, constraint inactive. $x_1 = 0$ from stationarity. $f_0 = 0$. But check: $x_1 = x_2 = 0$ is feasible ($0 \le 1$), $\lambda_1 = 0$ (constraint inactive), $\mu_1 = 0$, $\mu_2 = 0$. All KKT conditions satisfied. Global minimum: $x^* = (0,0)$, $f_0 = 0$.

> [!QUESTION] Exercise 11: Complementary Slackness and Primal Recovery
> *This problem establishes how complementary slackness can be used to determine the active set and thereby recover a primal solution from dual multipliers.*
>
> > **Prerequisites:** [[#5.1 Statement of the Conditions|5.1 Statement of the Conditions]], [[#5.2 Necessity|5.2 Necessity]]
>
> Suppose strong duality holds and $(\lambda^*, \nu^*)$ is dual optimal with $\lambda_i^* > 0$ for some index $i$. What can you conclude about $f_i(x^*)$ for any primal optimal $x^*$? Now suppose instead $f_i(x^*) < 0$. What does this imply about $\lambda_i^*$? State both as formal propositions and prove them from complementary slackness.

> [!TIP]- Solution to Exercise 11
> **Key insight:** Complementary slackness is a product condition: $\lambda_i^* f_i(x^*) = 0$ forces at least one factor to be zero.
>
> **Sketch:** By necessity (Theorem in §5.2), KKT conditions hold at any pair of primal/dual optima when strong duality holds. The complementary slackness condition is $\lambda_i^* f_i(x^*) = 0$.
>
> **Proposition 1:** If $\lambda_i^* > 0$, then $f_i(x^*) = 0$ (constraint $i$ is active). *Proof:* $\lambda_i^* f_i(x^*) = 0$ and $\lambda_i^* \ne 0$ implies $f_i(x^*) = 0$. $\square$
>
> **Proposition 2:** If $f_i(x^*) < 0$ (constraint $i$ is strictly inactive), then $\lambda_i^* = 0$. *Proof:* $\lambda_i^* f_i(x^*) = 0$ and $f_i(x^*) \ne 0$ implies $\lambda_i^* = 0$. $\square$
>
> Practical consequence: knowing $\lambda^*$ reveals which constraints are active at optimality (those with $\lambda_i^* > 0$), enabling primal recovery.

> [!QUESTION] Exercise 12: KKT Necessity Fails Without Regularity
> *This problem provides a counterexample showing that KKT conditions need not hold at a constrained minimum when constraint qualifications fail.*
>
> > **Prerequisites:** [[#5.2 Necessity|5.2 Necessity]], [[#4.4 Strong Duality and Slater's Condition|4.4 Strong Duality and Slater's Condition]]
>
> Consider: minimize $x_2$ subject to $x_2 \ge x_1^2$, or equivalently $f_1(x) = x_1^2 - x_2 \le 0$. (a) Find the optimal solution. (b) Write the KKT stationarity condition. (c) Check whether KKT conditions hold at the optimum and identify why this example is not a counterexample to our theorem.

> [!TIP]- Solution to Exercise 12
> **Key insight:** KKT conditions do hold here — but this example illustrates why Slater's condition must be checked: the optimal point $x^* = (0,0)$ is on the boundary of the feasible set, but Slater's condition still holds.
>
> **Sketch:** (a) $x^* = (0, 0)$, $p^* = 0$. (b) Stationarity: $\nabla f_0 + \lambda \nabla f_1 = (0, 1) + \lambda(-2x_1^*, 2x_2^*, -1)$... correcting: $f_1(x) = x_1^2 - x_2$, $\nabla f_1 = (2x_1, -1)$. At $x^* = (0,0)$: stationarity $(0,1) + \lambda(0,-1) = 0$ gives $\lambda = 1 \ge 0$. All four conditions hold. Slater's condition: $\tilde{x} = (0, 1)$ gives $f_1 = -1 < 0$. Strong duality holds. KKT necessity theorem applies, and KKT conditions do hold.

---

## 6. Recovering Primal Solutions from the Dual 💡

This section addresses the operational question: given a dual optimal $(\lambda^*, \nu^*)$, how do we find the corresponding primal optimal $x^*$? This is the key step in the click-shaping algorithm of Agarwal et al.

### 6.1 Strong Convexity and Uniqueness

If we know $(\lambda^*, \nu^*)$, the stationarity condition tells us $x^*$ minimizes $L(\cdot, \lambda^*, \nu^*)$. But what if this Lagrangian has multiple minimizers? The following result resolves this.

**Theorem (Primal Recovery via Strong Convexity).** Suppose strong duality holds and $(\lambda^*, \nu^*)$ is dual optimal. If $L(\cdot, \lambda^*, \nu^*)$ is *$\mu$-strongly convex* in $x$ for some $\mu > 0$, then $L(\cdot, \lambda^*, \nu^*)$ has a unique minimizer $x^*$, and this $x^*$ is the unique primal optimal solution.

*Proof.* Since $L(\cdot, \lambda^*, \nu^*)$ is $\mu$-strongly convex, it has a unique minimizer $\hat{x}$ (this follows from strong convexity: any two minimizers $\hat{x}_1, \hat{x}_2$ satisfy $L(\frac{\hat{x}_1 + \hat{x}_2}{2}, \lambda^*, \nu^*) < \frac{1}{2}L(\hat{x}_1, \lambda^*, \nu^*) + \frac{1}{2}L(\hat{x}_2, \lambda^*, \nu^*)$, contradicting both being minimizers). We have:
$$f_0(\hat{x}) + \underbrace{\sum_i \lambda_i^* f_i(\hat{x})}_{\ge 0 \text{ (sign unclear)}} + \underbrace{\sum_j \nu_j^* h_j(\hat{x})}_{= 0\text{ (if feasible)}} = g(\lambda^*, \nu^*) = p^*.$$
The necessity proof in Section 5.2 established that any primal optimal $x^*$ must satisfy $L(x^*, \lambda^*, \nu^*) = p^* = g(\lambda^*, \nu^*)$, i.e., $x^*$ also minimizes $L(\cdot, \lambda^*, \nu^*)$. By uniqueness of the minimizer, $x^* = \hat{x}$. $\square$

> [!NOTE] Sufficient Condition for Strong Convexity of L
> If $f_0$ is $\mu$-strongly convex and $f_i, h_j$ are convex (not necessarily strongly convex), then $L(\cdot, \lambda, \nu) = f_0 + \sum_i \lambda_i f_i + \sum_j \nu_j h_j$ inherits at least $\mu$-strong convexity in $x$ (since $\lambda_i \ge 0$ and adding non-negative combinations of convex functions preserves strong convexity modulus). So strong convexity of the objective is sufficient for unique primal recovery from the dual.

> [!QUESTION] Exercise 13: Sensitivity of Optimal Value to Constraints
> *This problem derives the shadow price interpretation of dual variables: $\lambda_i^*$ equals the rate of change of the optimal value when the $i$-th constraint is tightened.*
>
> > **Prerequisites:** [[#6.1 Strong Convexity and Uniqueness|6.1 Strong Convexity and Uniqueness]], [[#5.1 Statement of the Conditions|5.1 Statement of the Conditions]]
>
> Consider the perturbed problem: minimize $f_0(x)$ subject to $f_i(x) \le u_i$, $h_j(x) = v_j$, with optimal value $p^*(u, v)$. Assume strong duality and that the optimal dual $(\lambda^*, \nu^*)$ is unique. Show that $\nabla_u p^*(0, 0) = -\lambda^*$ and $\nabla_v p^*(0, 0) = -\nu^*$ (i.e., $\partial p^*/\partial u_i |_{u=0} = -\lambda_i^*$).

> [!TIP]- Solution to Exercise 13
> **Key insight:** The dual optimal value provides a linear lower bound on $p^*(u, v)$ that is tight at $(u,v) = (0,0)$, giving sensitivity as a byproduct.
>
> **Sketch:** By weak duality applied to the perturbed problem: $p^*(u,v) \ge g(\lambda, \nu) - \lambda^\top u - \nu^\top v$ for any dual feasible $(\lambda, \nu)$. At $(\lambda^*, \nu^*)$ dual optimal for the original problem: $p^*(u,v) \ge p^*(0,0) - (\lambda^*)^\top u - (\nu^*)^\top v$. This shows $p^*(u,v) - p^*(0,0) \ge -(\lambda^*)^\top u - (\nu^*)^\top v$, bounding the directional derivative from below. Under regularity (continuous differentiability of $p^*$), the bound is tight at the origin, giving $\nabla_u p^* = -\lambda^*$ and $\nabla_v p^* = -\nu^*$.

> [!QUESTION] Exercise 14: Dual Ascent Step Size
> *This problem derives the gradient of the dual function, which is needed for implementing gradient ascent on the dual.*
>
> > **Prerequisites:** [[#6.1 Strong Convexity and Uniqueness|6.1 Strong Convexity and Uniqueness]], [[#3.2 The Dual Function|3.2 The Dual Function]]
>
> Suppose $L(\cdot, \lambda, \nu)$ has a unique minimizer $x(\lambda, \nu)$ for each dual-feasible $(\lambda, \nu)$. Show that the dual function $g$ is differentiable at $(\lambda, \nu)$ and $\nabla_\lambda g(\lambda, \nu) = f(x(\lambda, \nu))$ (the vector of constraint function values at $x(\lambda,\nu)$), i.e., $\partial g/\partial \lambda_i = f_i(x(\lambda, \nu))$.

> [!TIP]- Solution to Exercise 14
> **Key insight:** The gradient of $g$ with respect to $\lambda_i$ is precisely the $i$-th constraint residual $f_i(x(\lambda,\nu))$, because the envelope theorem applies when the minimizer is unique.
>
> **Sketch:** $g(\lambda, \nu) = L(x(\lambda,\nu), \lambda, \nu) = f_0(x(\lambda,\nu)) + \sum_i \lambda_i f_i(x(\lambda,\nu)) + \sum_j \nu_j h_j(x(\lambda,\nu))$. Differentiating with respect to $\lambda_i$: $\partial g/\partial \lambda_i = \nabla_{x} L \cdot \partial x/\partial \lambda_i + f_i(x(\lambda,\nu))$. Since $x(\lambda,\nu)$ minimizes $L$, $\nabla_x L = 0$ at $x(\lambda,\nu)$. The first term vanishes (envelope theorem), leaving $\partial g/\partial \lambda_i = f_i(x(\lambda,\nu))$. This shows that dual gradient ascent has a natural interpretation: the gradient tells you how much constraint $i$ is violated at the current primal point.

### 6.2 Dual Decomposition Sketch

Dual decomposition is a method for solving the dual problem that simultaneously produces a primal solution. It is particularly useful when the primal problem *decomposes* across a sum structure.

Suppose the objective and constraints separate across user groups: $f_0(x) = \sum_k f_0^{(k)}(x^{(k)})$, $f_i(x) = \sum_k f_i^{(k)}(x^{(k)})$, where $x = (x^{(1)}, \ldots, x^{(K)})$ are disjoint blocks. Then:
$$L(x, \lambda, \nu) = \sum_k \left[ f_0^{(k)}(x^{(k)}) + \sum_i \lambda_i f_i^{(k)}(x^{(k)}) + \sum_j \nu_j h_j^{(k)}(x^{(k)}) \right] = \sum_k L^{(k)}(x^{(k)}, \lambda, \nu).$$

The infimum over $x$ decomposes into independent sub-problems:
$$g(\lambda, \nu) = \sum_k \inf_{x^{(k)}} L^{(k)}(x^{(k)}, \lambda, \nu).$$

The dual decomposition algorithm alternates between:
1. **Primal step:** for each $k$ in parallel, compute $x^{(k)}(\lambda, \nu) = \arg\min_{x^{(k)}} L^{(k)}(x^{(k)}, \lambda, \nu)$.
2. **Dual step:** update $\lambda$ by gradient ascent on $g$: $\lambda \leftarrow [\lambda + \alpha \nabla_\lambda g(\lambda, \nu)]_+$, where $[\cdot]_+$ denotes projection onto $\lambda \succeq 0$ and $\alpha > 0$ is a step size.

By Exercise 14, $\nabla_{\lambda_i} g = \sum_k f_i^{(k)}(x^{(k)}(\lambda, \nu)) = f_i(x(\lambda, \nu))$ — the total constraint violation.

```python
def dual_decomposition(subproblem_solvers, f_constraints, lambda_init, nu_init,
                       step_size, n_iterations):
    """
    Dual decomposition for a separable convex problem.

    Args:
        subproblem_solvers: list of callables, each taking (lambda_, nu) and
                            returning x_k minimizing L^(k)(x_k, lambda_, nu)
        f_constraints: callable taking x (concatenated) and returning
                       vector of constraint function values f_i(x)
        lambda_init: initial dual vector (shape: [m])
        nu_init: initial equality dual vector (shape: [p])
        step_size: positive float, gradient ascent step size
        n_iterations: number of dual ascent iterations
    Returns:
        lambda_opt, nu_opt, x_opt
    """
    import numpy as np
    lambda_ = lambda_init.copy()
    nu_ = nu_init.copy()

    for t in range(n_iterations):
        # Primal step: solve each sub-problem independently
        x_blocks = [solver(lambda_, nu_) for solver in subproblem_solvers]
        x = np.concatenate(x_blocks)

        # Dual gradient: constraint residuals
        constraint_vals = f_constraints(x)         # shape: [m]

        # Dual step: gradient ascent on g, project to lambda >= 0
        lambda_ = np.maximum(lambda_ + step_size * constraint_vals, 0.0)
        # Note: nu_ update for equality constraints (no projection needed)
        # nu_ += step_size * h_constraints(x)

    return lambda_, nu_, x
```

> [!WARNING] Step Size and Convergence
> *Constant step-size dual ascent converges only if the dual function is smooth (differentiable with Lipschitz gradient), which holds when the primal problem is strongly convex. Otherwise, subgradient methods with diminishing step sizes ($\alpha_t = c/\sqrt{t}$) are needed, at the cost of slower convergence.*

### 6.3 Connection to Click Shaping

We now make the connection to [Agarwal et al. (SIGIR 2012)](https://dl.acm.org/doi/10.1145/2348283.2348350) explicit. The paper considers a personalized recommendation problem with the following structure.

**Problem setup.** There is a set of items (news articles, ads) to be recommended to users. Each user $u$ can be assigned a recommendation allocation $x_u \in \Delta$ (a probability distribution over items, or a scoring vector). The objectives are:
- Maximize total expected clicks: $\sum_u f_0^{(u)}(x_u)$ (concave, since click models are log-concave in item scores).
- Subject to minimum engagement constraints on content categories: $\sum_u g_k^{(u)}(x_u) \ge b_k$ for each category $k$.

This is a *concave maximization problem* (equivalently, a convex minimization problem with $-f_0$) with convex constraints. The Lagrangian is:
$$L(x, \lambda) = -\sum_u f_0^{(u)}(x_u) + \sum_k \lambda_k \left(b_k - \sum_u g_k^{(u)}(x_u)\right)$$
where $\lambda_k \ge 0$ is the dual variable for category $k$'s engagement constraint.

**Decomposition.** The Lagrangian decomposes over users:
$$L(x, \lambda) = \sum_u \left[-f_0^{(u)}(x_u) - \sum_k \lambda_k g_k^{(u)}(x_u)\right] + \sum_k \lambda_k b_k.$$

For fixed $\lambda$, each user's sub-problem is independent:
$$x_u^*(\lambda) = \arg\max_{x_u \in \Delta} \left[f_0^{(u)}(x_u) + \sum_k \lambda_k g_k^{(u)}(x_u)\right].$$

**Interpretation of dual variables.** Each $\lambda_k$ is the *price* of category $k$ engagement. A higher $\lambda_k$ means the algorithm is willing to sacrifice more total clicks to meet category $k$'s engagement quota. The personalized recommendation for user $u$ at optimality is the item allocation that maximizes the *$\lambda$-weighted* combination of click likelihood and category engagement.

**Primal recovery.** The paper exploits strong convexity (from regularization of the click model, e.g., a Gaussian prior on item scores) to guarantee that the Lagrangian has a unique minimizer for each $\lambda$. The algorithm is:
1. Solve the dual problem $\max_{\lambda \succeq 0} g(\lambda)$ using gradient ascent on $\lambda$.
2. At each gradient step, solve each user's sub-problem to get $x_u(\lambda)$ — this is efficient since it decomposes by user.
3. At convergence to $\lambda^*$, set $x_u^* = x_u(\lambda^*)$ — this is the primal optimal allocation.

**The KKT conditions say:** at optimality, $\lambda_k^* > 0$ only if category $k$'s constraint is active (binding), and $\lambda_k^* = 0$ if the system naturally meets category $k$'s quota without any price incentive.

**Key result from Agarwal et al.** The strong convexity of the per-user click models (from regularization) ensures the dual function is smooth, gradient ascent converges at a linear rate, and the mapping $\lambda \mapsto x_u(\lambda)$ is continuous — enabling efficient online serving: the optimal personalized recommendation is computed by solving the small per-user argmax at the stored $\lambda^*$, without re-solving the full global problem.

> [!INFO] Segment-Based vs. Personalized Duality
> Earlier work (pre-Agarwal et al.) used a *segment-based* approach: partition users into a few non-overlapping groups (segments) and assign one allocation per segment. This is equivalent to solving the optimization problem with segment-level variables rather than user-level variables. The Lagrangian duality approach in Agarwal et al. generalizes this: the dual variables $\lambda^*$ provide a global price vector, and each user independently solves for their optimal allocation at those prices. This is precisely personalization — same prices, different per-user optima.

> [!QUESTION] Exercise 15: Per-User Sub-Problem Structure
> *This problem works through the per-user optimization that arises when dual decomposition is applied to the click-shaping problem.*
>
> > **Prerequisites:** [[#6.3 Connection to Click Shaping|6.3 Connection to Click Shaping]], [[#6.2 Dual Decomposition Sketch|6.2 Dual Decomposition Sketch]]
>
> Suppose the click model for user $u$ on item $a$ is logistic: $\text{CTR}_{u,a}(\theta) = \sigma(\theta_{u,a})$ where $\sigma$ is the sigmoid and $\theta_{u,a} \in \mathbb{R}$ is a score. The per-user objective is $f_0^{(u)}(\theta_u) = \sum_a \sigma(\theta_{u,a})$ and the engagement for category $k$ is $g_k^{(u)}(\theta_u) = \sum_{a \in \mathcal{A}_k} \sigma(\theta_{u,a})$. Write the per-user Lagrangian sub-problem and show it decomposes further into independent per-item problems.

> [!TIP]- Solution to Exercise 15
> **Key insight:** When items enter both the click objective and the engagement constraints through independent terms, the per-item sub-problems are fully decoupled.
>
> **Sketch:** The per-user Lagrangian is:
> $$L^{(u)}(\theta_u, \lambda) = -\sum_a \sigma(\theta_{u,a}) - \sum_k \lambda_k \sum_{a \in \mathcal{A}_k} \sigma(\theta_{u,a}) + \text{const}.$$
> Define $w_{u,a} = 1 + \sum_{k: a \in \mathcal{A}_k} \lambda_k$ (the total weight on item $a$ for user $u$). Then:
> $$L^{(u)}(\theta_u, \lambda) = -\sum_a w_{u,a} \sigma(\theta_{u,a}) + \text{const}.$$
> This decomposes over items: maximize $\theta_{u,a} \mapsto w_{u,a}\sigma(\theta_{u,a})$ independently for each $a$. Since $\sigma$ is monotone increasing, the optimal score $\theta_{u,a}^*$ is determined by any budget constraint on $\theta_{u,a}$; without constraints on $\theta$, the per-item problem is unbounded unless a regularizer is added. With an L2 regularizer $\frac{\rho}{2}\theta_{u,a}^2$, the per-item problem $\max_{\theta_{u,a}} w_{u,a}\sigma(\theta_{u,a}) - \frac{\rho}{2}\theta_{u,a}^2$ has a unique solution by strong convexity.

> [!QUESTION] Exercise 16: Dual Gradient Ascent as a Price Adjustment
> *This problem interprets dual gradient ascent on the click-shaping dual in economic terms, building intuition for why the algorithm converges to a personalized equilibrium.*
>
> > **Prerequisites:** [[#6.3 Connection to Click Shaping|6.3 Connection to Click Shaping]], [[#6.2 Dual Decomposition Sketch|6.2 Dual Decomposition Sketch]]
>
> In the click-shaping dual, the gradient ascent update for $\lambda_k$ is $\lambda_k \leftarrow \max(0, \lambda_k + \alpha (b_k - \sum_u g_k^{(u)}(x_u(\lambda))))$. Interpret this update economically: when does $\lambda_k$ increase, and what does that do to user recommendations in the next iteration?

> [!TIP]- Solution to Exercise 16
> **Key insight:** The dual update is a market clearing mechanism — prices rise when demand (engagement) falls below supply (quota), driving users toward the under-served category.
>
> **Sketch:** The gradient is $b_k - \sum_u g_k^{(u)}(x_u(\lambda))$ = (required engagement) - (actual engagement) for category $k$.
>
> If $\sum_u g_k^{(u)} < b_k$ (category $k$ under-served), the gradient is positive, so $\lambda_k$ increases. A higher $\lambda_k$ increases the weight $w_{u,a}$ for items $a \in \mathcal{A}_k$ in every user's per-item problem (Exercise 15), making category-$k$ items more attractive. In the next primal step, users are pushed toward category $k$, increasing $\sum_u g_k^{(u)}$.
>
> Conversely, if $\sum_u g_k^{(u)} > b_k$, the constraint is over-satisfied; $\lambda_k$ decreases (subject to $\lambda_k \ge 0$), reducing the incentive toward category $k$ and freeing the system to optimize clicks more freely. At convergence, $\lambda_k^* > 0$ implies $\sum_u g_k^{(u)} = b_k$ (binding); $\lambda_k^* = 0$ implies the system naturally over-satisfies the constraint without price incentive.

> [!QUESTION] Exercise 17: Primal Infeasibility Diagnostic
> *This problem uses the duality gap as a diagnostic tool — a practical skill for anyone implementing dual-based optimization.*
>
> > **Prerequisites:** [[#4.2 Weak Duality|4.2 Weak Duality]], [[#6.2 Dual Decomposition Sketch|6.2 Dual Decomposition Sketch]]
>
> Suppose after $T$ iterations of dual gradient ascent, you have a dual feasible point $\lambda^{(T)}$ with $g(\lambda^{(T)}) = -0.83$ and a primal candidate $x^{(T)}$ (produced by the primal step) with $f_0(x^{(T)}) = -0.80$. (a) What is the duality gap bound? (b) Is $x^{(T)}$ necessarily primal feasible? (c) If $x^{(T)}$ is primal infeasible, what does the gap bound certify?

> [!TIP]- Solution to Exercise 17
> **Key insight:** The weak duality guarantee $g \le p^*$ provides a lower bound on $p^*$ regardless of whether the primal candidate is feasible.
>
> **Sketch:** (a) $d^* \ge g(\lambda^{(T)}) = -0.83$, and the primal candidate has $f_0(x^{(T)}) = -0.80$. If $x^{(T)}$ were feasible, the gap would be at most $-0.80 - (-0.83) = 0.03$.
>
> (b) Not necessarily. Primal iterates produced by argmin-L are not guaranteed to be feasible — the Lagrangian relaxes the constraints. Feasibility must be checked separately.
>
> (c) Even if $x^{(T)}$ is infeasible, we know $p^* \ge -0.83$. If we also find any feasible primal point $\tilde{x}$ with $f_0(\tilde{x}) = -0.80$, then the gap $p^* - d^* \le -0.80 - (-0.83) = 0.03$ certifies $\tilde{x}$ is within 0.03 of optimal. The gap bounds suboptimality once a feasible point is in hand.

> [!QUESTION] Exercise 18: Strong Convexity Bound on Primal Suboptimality
> *This problem derives a key bound: when the Lagrangian is strongly convex, the primal iterate at each dual step is close to the true primal optimum, with a gap controlled by the dual optimality gap.*
>
> > **Prerequisites:** [[#6.1 Strong Convexity and Uniqueness|6.1 Strong Convexity and Uniqueness]], [[#4.3 The Duality Gap|4.3 The Duality Gap]]
>
> Suppose the Lagrangian $L(\cdot, \lambda^*, \nu^*)$ is $\mu$-strongly convex, and let $x(\lambda)$ denote the primal iterate (minimizer of $L(\cdot, \lambda, \nu)$) at a dual-feasible point $(\lambda, \nu)$. Let $x^* = x(\lambda^*)$ be the primal optimum. Show that $\|x(\lambda) - x^*\|^2 \le \frac{2}{\mu}[g(\lambda^*, \nu^*) - g(\lambda, \nu)]$, bounding the primal distance by the dual optimality gap.

> [!TIP]- Solution to Exercise 18
> **Key insight:** Strong convexity of the Lagrangian provides a quadratic lower bound on how much $L$ increases as we move away from its minimizer.
>
> **Sketch:** By $\mu$-strong convexity of $L(\cdot, \lambda^*, \nu^*)$:
> $$L(x(\lambda), \lambda^*, \nu^*) \ge L(x^*, \lambda^*, \nu^*) + \frac{\mu}{2}\|x(\lambda) - x^*\|^2 = g(\lambda^*, \nu^*) + \frac{\mu}{2}\|x(\lambda) - x^*\|^2.$$
> Also, $L(x(\lambda), \lambda^*, \nu^*) \le f_0(x(\lambda)) + \text{constraint terms} \le $ some bound. More directly: since $x(\lambda)$ minimizes $L(\cdot,\lambda,\nu)$, and $x^*$ minimizes $L(\cdot,\lambda^*,\nu^*)$, the dual function values give:
> $$g(\lambda, \nu) = L(x(\lambda), \lambda, \nu) \ge g(\lambda^*, \nu^*) - [g(\lambda^*, \nu^*) - L(x(\lambda), \lambda^*, \nu^*)].$$
> Combining with the strong convexity bound: $g(\lambda^*, \nu^*) - g(\lambda,\nu) \ge \frac{\mu}{2}\|x(\lambda) - x^*\|^2$, so $\|x(\lambda) - x^*\|^2 \le \frac{2}{\mu}[g(\lambda^*, \nu^*) - g(\lambda, \nu)]$.

---

> [!QUESTION] Exercise 19: Python Implementation of Dual Ascent for a QP
> *This problem implements dual gradient ascent for a small constrained QP and verifies convergence of both primal and dual objectives.*
>
> > **Prerequisites:** [[#6.2 Dual Decomposition Sketch|6.2 Dual Decomposition Sketch]], [[#3.2 The Dual Function|3.2 The Dual Function]]
>
> Implement dual gradient ascent for the QP: minimize $\frac{1}{2}\|x\|^2 - c^\top x$ subject to $Ax \preceq b$, with $n = 10$, $m = 5$, $A \in \mathbb{R}^{5\times 10}$ random, $b = A\mathbf{1}$, $c$ random. At each iteration: (a) solve the primal sub-problem in closed form given $\lambda$; (b) update $\lambda$ by projected gradient ascent. Plot the duality gap over 200 iterations.

> [!TIP]- Solution to Exercise 19
> **Key insight:** The primal sub-problem is an unconstrained QP with closed-form solution $x(\lambda) = c - A^\top\lambda$, making each iteration $O(mn)$.
>
> **Sketch:**
> ```python
> import numpy as np
> import matplotlib.pyplot as plt
>
> rng = np.random.default_rng(42)
> n, m = 10, 5
> A = rng.standard_normal((m, n))
> b = A @ np.ones(n)
> c = rng.standard_normal(n)
>
> def primal_solve(lam):
>     return c - A.T @ lam
>
> def primal_obj(x):
>     return 0.5 * x @ x - c @ x
>
> def dual_fn(lam):
>     x = primal_solve(lam)
>     return primal_obj(x) - lam @ (A @ x - b)
>
> lam = np.zeros(m)
> alpha = 0.01
> gaps = []
>
> for _ in range(200):
>     x = primal_solve(lam)
>     grad = A @ x - b          # constraint residuals
>     lam = np.maximum(lam - alpha * grad, 0.0)  # projected ascent: max lam
>
>     # duality gap = primal_obj(x_feas) - dual_fn(lam)
>     # use x from argmin L, which may be infeasible; gap is approximate
>     gaps.append(primal_obj(x) - dual_fn(lam))
>
> plt.plot(gaps)
> plt.xlabel("Iteration")
> plt.ylabel("Duality gap (approx)")
> plt.title("Dual gradient ascent convergence")
> plt.show()
> ```
> The gap converges to near zero under a suitable step size $\alpha < 2/\|AA^\top\|$ (twice the inverse of the largest eigenvalue of $AA^\top$).

> [!QUESTION] Exercise 20: Complexity of Dual Decomposition vs. Direct Solve
> *This problem analyzes why dual decomposition is preferred over direct interior-point methods for large-scale recommendation problems.*
>
> > **Prerequisites:** [[#6.2 Dual Decomposition Sketch|6.2 Dual Decomposition Sketch]]
>
> Suppose the click-shaping problem has $U$ users, $A$ items, and $K$ category constraints. (a) What is the size of the primal variable? (b) What is the per-iteration cost of dual gradient ascent (with per-user closed-form primal solves)? (c) What is the cost of solving the full problem directly using an interior-point method? Under what regime ($U$, $A$, $K$) does dual decomposition dominate?

> [!TIP]- Solution to Exercise 20
> **Key insight:** Interior-point methods scale as $O(n^3)$ in the primal dimension, while dual decomposition scales linearly in $U$ with a cost proportional to $K$ (the small dual dimension).
>
> **Sketch:** (a) Primal variable $x \in \mathbb{R}^{U \times A}$, dimension $n = UA$.
>
> (b) Per-user primal solve: $O(AK)$ (weight $w_{u,a}$ for each item, $K$ categories). Total primal step: $O(UAK)$. Dual update: $O(UK)$ for gradient computation plus $O(K)$ for the update. Per-iteration: $O(UAK)$.
>
> (c) Interior-point: $O((UA)^3) = O(U^3 A^3)$ per iteration, typically $O(\sqrt{K}\log(1/\varepsilon))$ outer iterations.
>
> Dual decomposition dominates when $U^2 A^2 \gg K$, which is almost always true in production recommendation systems ($U \sim 10^6$, $A \sim 10^3$, $K \sim 10$–$100$). The dual dimension $K$ (number of category constraints) is small and fixed; the primal dimension $UA$ is enormous. Dual decomposition exploits this mismatch.

> [!QUESTION] Exercise 21: Nondifferentiability of the Dual When the Primal Lacks Strong Convexity
> *This problem shows that without strong convexity, the dual function may be nondifferentiable, motivating the use of subgradient methods.*
>
> > **Prerequisites:** [[#3.3 Concavity of the Dual Function|3.3 Concavity of the Dual Function]], [[#6.1 Strong Convexity and Uniqueness|6.1 Strong Convexity and Uniqueness]]
>
> Consider the LP: minimize $c^\top x$ subject to $Ax \le b$, $x \ge 0$. The dual function $g(\lambda) = -b^\top\lambda + \inf_{x \ge 0}(c + A^\top\lambda)^\top x$. (a) Show $g$ is concave but generally not differentiable. (b) Define the subgradient of a concave function and show the constraint residual $b - Ax(\lambda)$ is a subgradient of $g$ at $\lambda$ whenever $x(\lambda)$ is any minimizer of $L(\cdot,\lambda)$.

> [!TIP]- Solution to Exercise 21
> **Key insight:** The dual function of an LP is a piecewise linear concave function (being the infimum of linear functions of $\lambda$), which is nondifferentiable at kinks — i.e., at dual points where multiple primal vertices are tied.
>
> **Sketch:** (a) $g(\lambda)$ is a pointwise infimum of linear (affine) functions of $\lambda$ (one per vertex of the LP polytope), hence concave and piecewise linear. At a kink, multiple vertices achieve the minimum simultaneously, so $\partial g$ is a set rather than a singleton — $g$ is not differentiable.
>
> (b) A vector $s$ is a *subgradient* of a concave $g$ at $\lambda$ if $g(\mu) \le g(\lambda) + s^\top(\mu - \lambda)$ for all $\mu$. For any $\mu$: $g(\mu) = \inf_x L(x,\mu) \le L(x(\lambda), \mu) = L(x(\lambda), \lambda) + (b - Ax(\lambda))^\top(\mu - \lambda) = g(\lambda) + (b - Ax(\lambda))^\top(\mu - \lambda)$. So $s = b - Ax(\lambda)$ satisfies the subgradient inequality, confirming it is a subgradient. When $x(\lambda)$ is unique, $s$ equals the gradient; when multiple minimizers exist, different choices of $x(\lambda)$ give different subgradients.

---

## References

| Reference Name | Brief Summary | Link to Reference |
|---|---|---|
| [Boyd & Vandenberghe, *Convex Optimization* (2004)](https://web.stanford.edu/~boyd/cvxbook/) | The canonical textbook on convex optimization; Chapters 2–5 cover convex sets/functions, standard form, Lagrangian duality, and KKT conditions. Primary source for this note. | [https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf) |
| [Agarwal, Chen, Elango, Wang — "Personalized Click Shaping through Lagrangian Duality for Online Recommendation" (SIGIR 2012)](https://dl.acm.org/doi/10.1145/2348283.2348350) | Motivating paper; casts personalized recommendation as a constrained convex optimization problem and uses Lagrangian duality + strong convexity for efficient primal recovery. | [https://dl.acm.org/doi/10.1145/2348283.2348350](https://dl.acm.org/doi/10.1145/2348283.2348350) |
| [Rockafellar, *Convex Analysis* (Princeton, 1970)](https://press.princeton.edu/books/paperback/9780691015866/convex-analysis) | Foundational treatise on convex analysis; rigorous treatment of dual functions, conjugates, subdifferentials, and separation theorems underlying duality theory. | [https://press.princeton.edu/books/paperback/9780691015866/convex-analysis](https://press.princeton.edu/books/paperback/9780691015866/convex-analysis) |
| [Rosenberg, "Extreme Abridgment of Boyd & Vandenberghe" (NYU, 2017)](https://davidrosenberg.github.io/mlcourse/Notes/convex-optimization.pdf) | Concise lecture notes summarizing key results on Lagrangian duality, KKT conditions, and dual decomposition, tailored for an ML audience. | [https://davidrosenberg.github.io/mlcourse/Notes/convex-optimization.pdf](https://davidrosenberg.github.io/mlcourse/Notes/convex-optimization.pdf) |
| [Rosenberg, "Lagrangian Duality and Convex Optimization" (NYU, 2017)](https://davidrosenberg.github.io/mlcourse/Archive/2017/Lectures/4a.convex-optimization.pdf) | Lecture slides covering the full duality pipeline from Lagrangian to KKT to primal recovery, with ML examples. | [https://davidrosenberg.github.io/mlcourse/Archive/2017/Lectures/4a.convex-optimization.pdf](https://davidrosenberg.github.io/mlcourse/Archive/2017/Lectures/4a.convex-optimization.pdf) |
| [Statistical Odds & Ends — "Lagrange dual, weak duality and strong duality"](https://statisticaloddsandends.wordpress.com/2022/11/12/lagrange-dual-weak-duality-and-strong-duality/) | Blog post with clean derivations of weak/strong duality and Slater's condition; useful secondary reference. | [https://statisticaloddsandends.wordpress.com/2022/11/12/lagrange-dual-weak-duality-and-strong-duality/](https://statisticaloddsandends.wordpress.com/2022/11/12/lagrange-dual-weak-duality-and-strong-duality/) |
| [Tibs, "10-725 Convex Optimization: Duality" (CMU, 2016)](https://www.stat.cmu.edu/~ryantibs/convexopt-F16/scribes/dual-gen-scribed.pdf) | Scribed lecture notes on Lagrangian duality and KKT conditions; includes coverage of dual decomposition and examples. | [https://www.stat.cmu.edu/~ryantibs/convexopt-F16/scribes/dual-gen-scribed.pdf](https://www.stat.cmu.edu/~ryantibs/convexopt-F16/scribes/dual-gen-scribed.pdf) |
| [Tang — "Introduction to Convex Optimization: Primal to Dual" (2023)](https://www.tangliyan.com/blog/posts/convex2/) | Accessible blog post deriving the dual problem step-by-step with clean notation; good supplementary reading for the Section 3–4 material. | [https://www.tangliyan.com/blog/posts/convex2/](https://www.tangliyan.com/blog/posts/convex2/) |
| [Hui — "Machine Learning: Lagrange Multiplier & Dual Decomposition"](https://jonathan-hui.medium.com/machine-learning-lagrange-multiplier-dual-decomposition-4afe66158c9) | Practical overview of dual decomposition with the price/market interpretation of Lagrange multipliers. | [https://jonathan-hui.medium.com/machine-learning-lagrange-multiplier-dual-decomposition-4afe66158c9](https://jonathan-hui.medium.com/machine-learning-lagrange-multiplier-dual-decomposition-4afe66158c9) |
| [Wikipedia — Duality (optimization)](https://en.wikipedia.org/wiki/Duality_(optimization)) | Reference for definitions and the full generality of Lagrangian duality beyond convex problems, including integer programming duality gaps. | [https://en.wikipedia.org/wiki/Duality_(optimization)](https://en.wikipedia.org/wiki/Duality_(optimization)) |
