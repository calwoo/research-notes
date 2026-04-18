# Shafarevich Ch. II: Local Properties

## Table of Contents

- [[#1. Singular and Nonsingular Points|1. Singular and Nonsingular Points]]
  - [[#1.1 The Local Ring at a Point|1.1 The Local Ring at a Point]]
  - [[#1.2 The Cotangent Space and Zariski Tangent Space|1.2 The Cotangent Space and Zariski Tangent Space]]
  - [[#1.3 The Jacobian Criterion|1.3 The Jacobian Criterion]]
  - [[#1.4 Regular Local Rings|1.4 Regular Local Rings]]
  - [[#1.5 The Smooth Locus is Open and Dense|1.5 The Smooth Locus is Open and Dense]]
  - [[#1.6 Examples: Nodal and Cuspidal Cubics|1.6 Examples: Nodal and Cuspidal Cubics]]
- [[#2. Power Series Expansion|2. Power Series Expansion]]
  - [[#2.1 Local Parameters|2.1 Local Parameters]]
  - [[#2.2 The Formal Completion|2.2 The Formal Completion]]
  - [[#2.3 Power Series Expansion of Regular Functions|2.3 Power Series Expansion of Regular Functions]]
- [[#3. Properties of Nonsingular Points|3. Properties of Nonsingular Points]]
  - [[#3.1 Analytical Irreducibility|3.1 Analytical Irreducibility]]
  - [[#3.2 Smooth Subvarieties and the Normal Bundle|3.2 Smooth Subvarieties and the Normal Bundle]]
  - [[#3.3 The Algebraic Implicit Function Theorem|3.3 The Algebraic Implicit Function Theorem]]
  - [[#3.4 Local Factoriality|3.4 Local Factoriality]]
- [[#4. The Tangent Cone|4. The Tangent Cone]]
  - [[#4.1 Definition via Initial Forms|4.1 Definition via Initial Forms]]
  - [[#4.2 The Associated Graded Ring|4.2 The Associated Graded Ring]]
  - [[#4.3 Multiplicity|4.3 Multiplicity]]
  - [[#4.4 Examples|4.4 Examples]]
- [[#5. Discrete Valuation Rings and Smooth Curves|5. Discrete Valuation Rings and Smooth Curves]]
  - [[#5.1 DVRs: Definitions and Characterizations|5.1 DVRs: Definitions and Characterizations]]
  - [[#5.2 Smooth Points of Curves are DVRs|5.2 Smooth Points of Curves are DVRs]]
  - [[#5.3 The Valuation and Order of Vanishing|5.3 The Valuation and Order of Vanishing]]
- [[#References|References]]

---

Throughout, $k$ denotes an algebraically closed field (e.g. $k = \mathbb{C}$) and all varieties are over $k$, unless otherwise stated. We write $\mathbb{A}^n = \mathbb{A}^n_k$ for affine $n$-space. For background on affine varieties, coordinate rings, rational functions, local rings, DVRs, and intersection numbers, see [[concepts/algebraic-geometry/note|Classical Algebraic Geometry]].

---

## 1. Singular and Nonsingular Points

### 1.1 The Local Ring at a Point

📐 Let $X \subset \mathbb{A}^n$ be an irreducible affine variety with coordinate ring $A(X) = k[x_1,\ldots,x_n]/I(X)$. For a point $P \in X$, the *local ring* at $P$ is the localization

$$\mathcal{O}_{X,P} = A(X)_{\mathfrak{m}_P},$$

where $\mathfrak{m}_P = \{ f \in A(X) : f(P) = 0 \}$ is the maximal ideal of $P$ in $A(X)$. Concretely, $\mathcal{O}_{X,P}$ consists of fractions $f/g$ with $f, g \in A(X)$ and $g(P) \neq 0$, representing *germs of regular functions* near $P$.

**Definition (Local Ring and Maximal Ideal).** The *local ring of $X$ at $P$* is the pair $(\mathcal{O}_{X,P}, \mathfrak{m}_{X,P})$, where

$$\mathfrak{m}_{X,P} = \{ \phi \in \mathcal{O}_{X,P} : \phi(P) = 0 \}$$

is the unique maximal ideal of $\mathcal{O}_{X,P}$. The residue field is $\mathcal{O}_{X,P}/\mathfrak{m}_{X,P} \cong k$.

> [!INFO] Invariance under open embeddings
> The local ring $\mathcal{O}_{X,P}$ depends only on the variety $X$ and the point $P$, not on any ambient affine embedding. If $U \subset X$ is any open affine neighborhood of $P$, then $\mathcal{O}_{X,P} \cong \mathcal{O}_{U,P}$. This makes local ring data intrinsic to the abstract variety.

### 1.2 The Cotangent Space and Zariski Tangent Space

🔑 The *cotangent space* of $X$ at $P$ is the $k$-vector space $\mathfrak{m}_{X,P}/\mathfrak{m}_{X,P}^2$. Its dual encodes the first-order linear geometry of $X$ at $P$.

**Definition (Zariski Tangent Space).** The *Zariski tangent space* of $X$ at $P$ is

$$T_{X,P} = \bigl(\mathfrak{m}_{X,P}/\mathfrak{m}_{X,P}^2\bigr)^\vee = \mathrm{Hom}_k\!\bigl(\mathfrak{m}_{X,P}/\mathfrak{m}_{X,P}^2,\, k\bigr).$$

The space $\mathfrak{m}_{X,P}/\mathfrak{m}_{X,P}^2$ is the *Zariski cotangent space*.

**Why quotient by $\mathfrak{m}^2$?** A function $f \in \mathfrak{m}_{X,P}$ vanishes at $P$; in a Taylor expansion, $f = \sum_i (\partial f/\partial x_i)(P)(x_i - P_i) + \text{higher order}$. The class $[f] \in \mathfrak{m}/\mathfrak{m}^2$ retains only the linear part. Indeed, $\mathfrak{m}^2 = \{fg : f,g \in \mathfrak{m}\}$ corresponds exactly to functions whose first-order Taylor coefficients all vanish at $P$. Thus $\mathfrak{m}/\mathfrak{m}^2$ is the space of linear forms on $X$ at $P$, and its dual is the space of tangent directions.

**Derivation description.** Equivalently, $T_{X,P} \cong \mathrm{Der}_k(\mathcal{O}_{X,P}, k)$, the set of $k$-linear maps $\delta : \mathcal{O}_{X,P} \to k$ satisfying the Leibniz rule $\delta(fg) = f(P)\delta(g) + g(P)\delta(f)$. Any such derivation is determined by its values on $\mathfrak{m}_{X,P}$, and since $\delta(\mathfrak{m}^2) = 0$ (Leibniz with $f(P) = g(P) = 0$), it factors through $\mathfrak{m}/\mathfrak{m}^2$.

> [!EXAMPLE] Affine space
> For $X = \mathbb{A}^n$ and $P = 0$, we have $\mathcal{O}_{\mathbb{A}^n,0} = k[x_1,\ldots,x_n]_{(x_1,\ldots,x_n)}$ and $\mathfrak{m} = (x_1,\ldots,x_n)$. Then $\mathfrak{m}/\mathfrak{m}^2$ is freely spanned by the classes $[x_1],\ldots,[x_n]$, so $T_{\mathbb{A}^n, 0} \cong k^n$. This matches the naive geometric notion: affine space is its own tangent space at every point.

### 1.3 The Jacobian Criterion

📐 For a concrete hypersurface or complete intersection, the tangent space is computed by linearization. We derive the Jacobian criterion from the cotangent-space definition.

Let $X = V(f_1,\ldots,f_m) \subset \mathbb{A}^n$ be a variety defined by polynomials $f_1,\ldots,f_m \in k[x_1,\ldots,x_n]$, with $P \in X$.

**Derivation of the Jacobian Criterion.** The coordinate ring of $\mathbb{A}^n$ maps surjectively onto $A(X)$, hence $\mathcal{O}_{\mathbb{A}^n, P} \to \mathcal{O}_{X,P}$ is surjective with kernel generated by the images of $f_1,\ldots,f_m$. Writing $\mathfrak{m} = \mathfrak{m}_{\mathbb{A}^n,P}$ and $\mathfrak{n} = \mathfrak{m}_{X,P}$, we get a surjection

$$\mathfrak{m}/\mathfrak{m}^2 \twoheadrightarrow \mathfrak{n}/\mathfrak{n}^2,$$

whose kernel is spanned by the images of $f_1,\ldots,f_m$ modulo $\mathfrak{m}^2$. Since each $f_i$ vanishes at $P$, we write $f_i = \sum_j (\partial f_i/\partial x_j)(P)(x_j - P_j) + O(|\mathbf{x}-P|^2)$, so $[f_i]$ maps to the linear form $\sum_j (\partial f_i / \partial x_j)(P) \cdot [x_j - P_j]$ in $\mathfrak{m}/\mathfrak{m}^2 \cong k^n$. Therefore

$$\mathfrak{n}/\mathfrak{n}^2 \cong k^n \Big/ \mathrm{span}\!\left\{ \left(\tfrac{\partial f_i}{\partial x_j}(P)\right)_{j=1}^n : i = 1,\ldots,m \right\}.$$

Taking duals:

**Theorem (Jacobian Criterion).** With notation as above,

$$T_{X,P} = \ker\!\left(\frac{\partial f_i}{\partial x_j}(P)\right)_{i,j} \subset k^n,$$

where the Jacobian matrix $J(P) = \bigl((\partial f_i/\partial x_j)(P)\bigr) \in \mathrm{Mat}_{m \times n}(k)$ acts on $k^n = T_{\mathbb{A}^n, P}$.

*Remark.* The formula shows $\dim T_{X,P} = n - \mathrm{rank}\, J(P)$. Since $\dim X \geq n - m$ (by Krull's principal ideal theorem), we always have $\dim T_{X,P} \geq \dim X$.

> [!WARNING] The criterion depends on the choice of equations
> The Jacobian matrix $J(P)$ depends on the chosen generators $f_1,\ldots,f_m$. However, the kernel of $J(P)$ — i.e. $T_{X,P}$ — is intrinsic to $X$ and $P$. Different choices of generators give the same subspace of $k^n$.

### 1.4 Regular Local Rings

**Definition (Smooth and Singular Points).** A point $P \in X$ is *smooth* (or *nonsingular*, or *regular*) if

$$\dim_k T_{X,P} = \dim X.$$

$P$ is *singular* if $\dim_k T_{X,P} > \dim X$. The *singular locus* of $X$ is $\mathrm{Sing}(X) = \{ P \in X : P \text{ is singular}\}$.

**Definition (Regular Local Ring).** A Noetherian local ring $(R, \mathfrak{m}, k)$ is *regular* if

$$\dim_k \mathfrak{m}/\mathfrak{m}^2 = \dim R,$$

where $\dim R$ is the Krull dimension. The left side is at least $\dim R$ for any Noetherian local ring (by Nakayama's lemma and the theory of Hilbert–Samuel polynomials), so regularity is the extremal equality.

**Proposition.** $P \in X$ is smooth if and only if $\mathcal{O}_{X,P}$ is a regular local ring.

*Proof sketch.* We have $\dim_k \mathfrak{m}_{X,P}/\mathfrak{m}_{X,P}^2 = \dim_k T_{X,P}$. By definition of smoothness, this equals $\dim X = \dim \mathcal{O}_{X,P}$, which is exactly the regularity condition. $\square$

> [!TIP] Regularity implies integrality
> A regular local ring is an integral domain (Serre's theorem). This has geometric content: at a smooth point, the local ring has no zero divisors, meaning the variety is "locally irreducible." (At a singular point, the variety may look locally like two crossing branches, introducing zero divisors in the local ring.)

### 1.5 The Smooth Locus is Open and Dense

🔑 Singular points are isolated in a precise topological sense.

**Theorem (Smooth Locus).** Let $X$ be an irreducible variety over an algebraically closed field $k$ of characteristic zero. Then the smooth locus $X_{\mathrm{sm}} = X \setminus \mathrm{Sing}(X)$ is open and dense in $X$.

*Proof sketch.* **Openness:** By the Jacobian criterion, smoothness at $P$ is the condition that the Jacobian matrix $J(P)$ has rank $\geq \mathrm{codim}(X, \mathbb{A}^n)$. This is an open condition — the locus where a matrix has rank less than a given value is Zariski closed (defined by vanishing of minors). Hence $\mathrm{Sing}(X)$ is closed, and $X_{\mathrm{sm}}$ is open.

**Density (nonemptiness of $X_{\mathrm{sm}}$):** It suffices to show $X$ has at least one smooth point. By Noether normalization, the function field $k(X)$ is a separable extension of a pure transcendental subfield (in characteristic zero, all finite extensions are separable). One may show the generic point of $X$ is smooth — i.e., the local ring at the generic point is regular — and since smoothness is an open condition, there is a dense open subset of smooth points.

**Remark.** In positive characteristic, one must additionally assume $X$ is geometrically reduced (which holds e.g. for varieties defined over a perfect field). In characteristic zero over an algebraically closed field, irreducibility suffices.

> [!NOTE] Singular locus has lower dimension
> Since $\mathrm{Sing}(X)$ is a proper closed subvariety of the irreducible variety $X$, it satisfies $\dim \mathrm{Sing}(X) < \dim X$. Singularities are always "of lower dimension" than the variety itself.

### 1.6 Examples: Nodal and Cuspidal Cubics

**Example (Nodal Cubic).** Let $X = V(y^2 - x^2(x+1)) \subset \mathbb{A}^2$. Setting $f = y^2 - x^2(x+1) = y^2 - x^3 - x^2$:

$$\frac{\partial f}{\partial x} = -3x^2 - 2x, \qquad \frac{\partial f}{\partial y} = 2y.$$

At $P = (0,0)$: both partials vanish, so $J(P) = (0, 0)$ has rank $0 < 1 = \mathrm{codim}(X)$. Thus $(0,0)$ is singular. For any other point, at least one partial is nonzero, so $X$ has only one singular point: the *node* at the origin.

**Example (Cuspidal Cubic).** Let $X = V(y^2 - x^3) \subset \mathbb{A}^2$. Setting $f = y^2 - x^3$:

$$\frac{\partial f}{\partial x} = -3x^2, \qquad \frac{\partial f}{\partial y} = 2y.$$

At $P = (0,0)$: both vanish, so the origin is singular — a *cusp*.

**Example (Whitney Umbrella).** The surface $X = V(x^2 - y^2 z) \subset \mathbb{A}^3$ is singular along the $z$-axis. Setting $f = x^2 - y^2z$:

$$\frac{\partial f}{\partial x} = 2x, \quad \frac{\partial f}{\partial y} = -2yz, \quad \frac{\partial f}{\partial z} = -y^2.$$

The Jacobian vanishes when $x = 0$, $yz = 0$, $y = 0$, i.e. when $x = 0$ and $y = 0$. So $\mathrm{Sing}(X) = V(x,y) \cap X$ is the $z$-axis $\{x = y = 0\}$, which has dimension $1 < 2 = \dim X$, confirming the density theorem.

> [!QUESTION] Exercise 1: Singular Locus of a Surface
> *This problem asks you to compute the singular locus of a specific surface and verify the dimension bound.*
>
> > **Prerequisites:** [[#1.3 The Jacobian Criterion|1.3 The Jacobian Criterion]]
>
> Let $X = V(x^2y + y^2z + z^2x) \subset \mathbb{A}^3$. Compute $\mathrm{Sing}(X)$ using the Jacobian criterion, and verify that $\dim \mathrm{Sing}(X) < 2 = \dim X$.

> [!TIP]- Solution to Exercise 1
> **Key insight:** The singular locus of a surface in $\mathbb{A}^3$ defined by $f$ is $V(f, \partial f/\partial x, \partial f/\partial y, \partial f/\partial z)$.
>
> **Sketch:** Let $f = x^2y + y^2z + z^2x$. Compute $\partial f/\partial x = 2xy + z^2$, $\partial f/\partial y = x^2 + 2yz$, $\partial f/\partial z = y^2 + 2zx$. Setting all three to zero along with $f = 0$: from $\partial f/\partial x = 0$ we get $z^2 = -2xy$; from $\partial f/\partial y = 0$ we get $x^2 = -2yz$; from $\partial f/\partial z = 0$ we get $y^2 = -2xz$. Multiplying: $(xyz)^2 = -8(xyz)^2$, so $9(xyz)^2 = 0$, hence $xyz = 0$. Examining cases ($x=0$, $y=0$, $z=0$) and substituting back shows the only solution is $P = (0,0,0)$. Thus $\mathrm{Sing}(X) = \{(0,0,0)\}$, which has dimension $0 < 2$.

> [!QUESTION] Exercise 2: Tangent Space Computation
> *This problem asks you to compute the Zariski tangent space at a smooth point and verify the dimension formula.*
>
> > **Prerequisites:** [[#1.2 The Cotangent Space and Zariski Tangent Space|1.2 The Cotangent Space and Zariski Tangent Space]], [[#1.3 The Jacobian Criterion|1.3 The Jacobian Criterion]]
>
> Let $X = V(x^2 + y^2 + z^2 - 1) \subset \mathbb{A}^3$ (the unit sphere, valid over $\mathbb{C}$). Compute $T_{X,P}$ for $P = (1,0,0)$ using both the cotangent-space definition and the Jacobian criterion, and verify they agree.

> [!TIP]- Solution to Exercise 2
> **Key insight:** Both definitions give the kernel of $J(P)$, which is the hyperplane of vectors perpendicular (in the linearized sense) to $\nabla f(P)$.
>
> **Sketch:** With $f = x^2 + y^2 + z^2 - 1$, we have $J(P) = (2x, 2y, 2z)|_P = (2, 0, 0)$. The Jacobian criterion gives $T_{X,P} = \ker(2, 0, 0) = \{(a,b,c) : 2a = 0\} = \{0\} \times k^2$, a 2-dimensional space. Since $\dim X = 2$, $P$ is smooth. For the cotangent-space approach: $\mathfrak{m}_P = (x-1, y, z)/I(X)$ and $\mathfrak{m}_P^2$ kills all degree-$\geq 2$ terms in the shifted variables; modding out by $I(X)$ and $\mathfrak{m}^2$ uses $x^2 - 1 \equiv 2(x-1) \pmod{\mathfrak{m}^2}$ to eliminate $x-1$ in favor of $y^2, z^2$ (both in $\mathfrak{m}^2$). Hence $\mathfrak{m}_P/\mathfrak{m}_P^2$ is spanned by $[y], [z]$, confirming $T_{X,P} \cong k^2$.

---

## 2. Power Series Expansion

### 2.1 Local Parameters

🗺️ At a smooth point of an $n$-dimensional variety, local algebra admits a canonical set of "coordinates."

**Definition (Local Parameters).** Let $X$ be smooth of dimension $n$ at a point $P$. A *system of local parameters* (or *regular system of parameters*) at $P$ is a set $\{t_1,\ldots,t_n\} \subset \mathfrak{m}_{X,P}$ whose images in $\mathfrak{m}_{X,P}/\mathfrak{m}_{X,P}^2$ form a $k$-basis.

By Nakayama's lemma (since $\mathfrak{m}_{X,P}/\mathfrak{m}_{X,P}^2 \cong k^n$ at a smooth point), the elements $t_1,\ldots,t_n$ generate $\mathfrak{m}_{X,P}$ as an ideal. They are the algebraic analogue of local coordinates in differential geometry.

**Existence.** Choose any $n$ regular functions on an affine neighborhood $U$ of $P$ such that $t_i(P) = 0$ and the differentials $dt_i(P) = [t_i] \in \mathfrak{m}/\mathfrak{m}^2$ form a basis. For example, if $U \hookrightarrow \mathbb{A}^n$ is an open immersion (valid near a smooth point), the shifted coordinate functions $x_i - P_i$ restrict to local parameters.

**Remark.** *Different choices of local parameters give isomorphic completions* — see below. The set of local parameters is far from unique, but the local ring structure is canonical.

> [!EXAMPLE] Local parameters on a curve
> For the curve $X = V(y - x^2) \subset \mathbb{A}^2$ and the smooth point $P = (0,0)$, the coordinate ring is $k[x,y]/(y-x^2) \cong k[x]$ and $\mathfrak{m}_P = (x)$. The element $t = x$ (a single element, since $\dim X = 1$) is a local parameter. The element $y = x^2$ is *not* a local parameter since $[y] = [x]^2 \in \mathfrak{m}^2$.

### 2.2 The Formal Completion

📐 The most powerful tool for local analysis is the *formal completion*.

**Definition (Completion).** The *formal completion* of $\mathcal{O}_{X,P}$ with respect to $\mathfrak{m}_{X,P}$ is the inverse limit

$$\hat{\mathcal{O}}_{X,P} = \varprojlim_{n \geq 0} \mathcal{O}_{X,P}/\mathfrak{m}_{X,P}^n.$$

Elements of $\hat{\mathcal{O}}_{X,P}$ are coherent sequences $(\phi_n)_{n \geq 0}$ with $\phi_n \in \mathcal{O}_{X,P}/\mathfrak{m}^n$ and $\phi_{n+1} \mapsto \phi_n$ under reduction. There is a canonical ring homomorphism $\mathcal{O}_{X,P} \to \hat{\mathcal{O}}_{X,P}$, which is injective when $\mathcal{O}_{X,P}$ is Noetherian (Krull's intersection theorem: $\bigcap_n \mathfrak{m}^n = 0$).

**Theorem (Cohen Structure Theorem, smooth case).** If $P \in X$ is a smooth point of dimension $n$ with local parameters $t_1,\ldots,t_n$, then

$$\hat{\mathcal{O}}_{X,P} \cong k\llbracket t_1,\ldots,t_n\rrbracket,$$

the ring of formal power series in $n$ variables over $k$.

*Proof sketch.* Since $P$ is smooth, $\mathcal{O}_{X,P}$ is a regular local ring of dimension $n$. The general Cohen structure theorem states that every complete equicharacteristic Noetherian local ring $(R, \mathfrak{m}, k)$ is a quotient $k\llbracket x_1,\ldots,x_N\rrbracket/I$ for some $N$ and ideal $I$. For a *regular* local ring, the minimal number of generators of $\mathfrak{m}$ equals $n = \dim R$, so $I = 0$ and $\hat{\mathcal{O}}_{X,P} \cong k\llbracket t_1,\ldots,t_n\rrbracket$. $\square$

> [!NOTE] Universal property
> The completion $\hat{\mathcal{O}}_{X,P}$ is the universal ring to which $\mathcal{O}_{X,P}$ maps and where the $\mathfrak{m}$-adic topology is complete (every Cauchy sequence converges). It captures all local data "to all orders."

### 2.3 Power Series Expansion of Regular Functions

**Proposition (Unique Power Series Expansion).** Let $P \in X$ be smooth with local parameters $t_1,\ldots,t_n$, and let $f \in \mathcal{O}_{X,P}$. Then $f$ admits a unique expansion

$$f = \sum_{\alpha \in \mathbb{Z}_{\geq 0}^n} c_\alpha t^\alpha \in k\llbracket t_1,\ldots,t_n\rrbracket, \qquad c_\alpha \in k,$$

where $t^\alpha = t_1^{\alpha_1}\cdots t_n^{\alpha_n}$, via the injective map $\mathcal{O}_{X,P} \hookrightarrow \hat{\mathcal{O}}_{X,P} \cong k\llbracket t_1,\ldots,t_n\rrbracket$.

*Derivation.* Since $t_1,\ldots,t_n$ generate $\mathfrak{m}_{X,P}$ and $\mathcal{O}_{X,P}/\mathfrak{m}^{m+1}$ is spanned by monomials $t^\alpha$ with $|\alpha| \leq m$, any $f \in \mathcal{O}_{X,P}$ determines, for each $m$, a polynomial of degree $\leq m$ that is the truncation of the expansion. These truncations are compatible, giving the formal series. Injectivity of $\mathcal{O}_{X,P} \to k\llbracket t_1,\ldots,t_n\rrbracket$ follows from $\bigcap_m \mathfrak{m}^m = 0$.

**Geometric interpretation.** The local parameters $t_1,\ldots,t_n$ play the role of "local coordinates" centered at $P$. The power series expansion of $f$ is the analogue of the Taylor series expansion in differential geometry. Unlike the complex-analytic setting, this series need not converge in any metric sense — it lives in the *formal* power series ring — but it contains complete local information about $f$.

> [!TIP] Comparing coefficients
> The coefficient $c_\alpha$ in the expansion of $f$ can be computed as: $c_\alpha = \frac{1}{\alpha!} \cdot [\text{image of } f \text{ in } \mathfrak{m}^{|\alpha|}/\mathfrak{m}^{|\alpha|+1}]$ in the associated graded. More concretely, $c_0 = f(P)$ and the linear coefficients $c_{e_i}$ encode the "partial derivatives" of $f$ in the directions $t_i$.

> [!QUESTION] Exercise 3: Local Parameters and Expansion
> *This problem establishes that different choices of local parameters yield the same formal germ of a function.*
>
> > **Prerequisites:** [[#2.1 Local Parameters|2.1 Local Parameters]], [[#2.2 The Formal Completion|2.2 The Formal Completion]]
>
> Let $X = \mathbb{A}^2$, $P = (0,0)$, and consider two systems of local parameters: $(t_1,t_2) = (x,y)$ and $(s_1,s_2) = (x+y, x-y)$. Express $f = e^x \sin(y)$ (formally) in both systems and verify that the two completions $k\llbracket t_1,t_2\rrbracket$ and $k\llbracket s_1,s_2\rrbracket$ are isomorphic via the change of variables $t_1 = (s_1+s_2)/2$, $t_2 = (s_1-s_2)/2$.

> [!TIP]- Solution to Exercise 3
> **Key insight:** The completion isomorphism $k\llbracket t_1,t_2\rrbracket \cong k\llbracket s_1,s_2\rrbracket$ is given by any automorphism of formal power series rings induced by a linear change of variables with invertible Jacobian at the origin.
>
> **Sketch:** The change $(t_1,t_2) = ((s_1+s_2)/2, (s_1-s_2)/2)$ has Jacobian matrix $\begin{pmatrix}1/2 & 1/2 \\ 1/2 & -1/2\end{pmatrix}$ with determinant $-1/2 \neq 0$. By the formal implicit function theorem (valid over any field), any such invertible linear substitution extends to an isomorphism of formal power series rings. For the expansion of $f = xy + \cdots$ (the formal version): in $(t_1,t_2)$ coordinates, $f = t_1 t_2 + \cdots$; substituting $t_1 = (s_1+s_2)/2$, $t_2 = (s_1-s_2)/2$ gives $f = (s_1^2 - s_2^2)/4 + \cdots$, a valid formal series in $s_1, s_2$.

> [!QUESTION] Exercise 4: Detecting Smoothness via Completion
> *This problem shows that regularity of the completion characterizes smooth points.*
>
> > **Prerequisites:** [[#1.4 Regular Local Rings|1.4 Regular Local Rings]], [[#2.2 The Formal Completion|2.2 The Formal Completion]]
>
> Let $R = k[x,y]/(y^2 - x^3)$ localized at the maximal ideal $(x,y)$. Show that $\hat{R} \not\cong k\llbracket t\rrbracket$ (i.e., the completion is not a power series ring in one variable), hence $(0,0)$ is not a smooth point of $V(y^2 - x^3)$.

> [!TIP]- Solution to Exercise 4
> **Key insight:** If $(0,0)$ were smooth on the curve $V(y^2-x^3)$, the completion would be $k\llbracket t\rrbracket$. But $k\llbracket t\rrbracket$ is a domain, while $\hat{R}$ has a specific structure that can be analyzed.
>
> **Sketch:** In $R$, we have $y^2 = x^3$. In the completion $\hat{R}$, suppose $\hat{R} \cong k\llbracket t\rrbracket$; then $x = u(t) t^2$ and $y = v(t) t^3$ for units $u, v$ (matching valuations $v(x) = 2, v(y) = 3$ implied by $y^2 = x^3$). But these valuations show $\mathfrak{m}_R = (x,y)$ requires two generators with $v(x) = 2, v(y) = 3$, so $\mathfrak{m}_R \not\cong (t)$ — the minimal generator of $\mathfrak{m}$ in $k\llbracket t\rrbracket$ has valuation 1. More formally, $\dim_k \mathfrak{m}/\mathfrak{m}^2 = 2$ (since $[x]$ and $[y]$ are linearly independent in $\mathfrak{m}/\mathfrak{m}^2$) while $\dim R = 1$, so $R$ is not regular.

---

## 3. Properties of Nonsingular Points

### 3.1 Analytical Irreducibility

🔑 At a smooth point, the variety cannot "branch."

**Definition (Analytically Irreducible).** A local ring $(R, \mathfrak{m})$ is *analytically irreducible* if its completion $\hat{R}$ is an integral domain.

**Theorem (Smooth points are analytically irreducible).** If $P \in X$ is a smooth point of an irreducible variety $X$, then $\mathcal{O}_{X,P}$ is analytically irreducible.

*Proof sketch.* By the Cohen structure theorem, $\hat{\mathcal{O}}_{X,P} \cong k\llbracket t_1,\ldots,t_n\rrbracket$, which is an integral domain (formal power series rings over a field are domains: if $fg = 0$ in $k\llbracket t_1,\ldots,t_n\rrbracket$, compare leading terms). $\square$

**Geometric meaning.** Analytically irreducible means the formal neighborhood of $P$ in $X$ is connected: the variety does not locally look like two (or more) branches crossing at $P$. This fails at singular points like nodes. For example, the nodal cubic $V(y^2 - x^2(x+1))$ at the origin: the completion of its local ring factors as $k\llbracket t_1\rrbracket \times k\llbracket t_2\rrbracket$ (one factor per branch), which is not a domain.

> [!INFO] Zariski's theorem
> Zariski (1948) proved more: if $\mathcal{O}_{X,P}$ is a *normal* local ring (i.e., integrally closed in its fraction field), then it is analytically irreducible. Since regular local rings are normal (a theorem of Serre), this recovers the above result. The node is not normal at the singular point, which is the algebraic reason for its analytic reducibility.

### 3.2 Smooth Subvarieties and the Normal Bundle

📐 When $Y \subset X$ is a smooth closed subvariety of a smooth ambient variety, the first-order geometry of the pair $(X,Y)$ is captured by the *normal bundle*.

**Setup.** Let $X$ be a smooth variety of dimension $n$, and let $Y \subset X$ be a closed smooth subvariety of dimension $r$. Denote by $\mathcal{I}_Y \subset \mathcal{O}_X$ the ideal sheaf of $Y$ (the sheaf whose sections on an open $U$ are the regular functions on $U$ vanishing on $U \cap Y$).

**Definition (Conormal Sheaf and Normal Bundle).** The *conormal sheaf* of $Y$ in $X$ is

$$\mathcal{N}^\vee_{Y/X} = \mathcal{I}_Y / \mathcal{I}_Y^2,$$

viewed as an $\mathcal{O}_Y$-module (via the surjection $\mathcal{O}_X \to \mathcal{O}_Y$). The *normal sheaf* (or *normal bundle*) is its $\mathcal{O}_Y$-dual:

$$\mathcal{N}_{Y/X} = \mathcal{H}om_{\mathcal{O}_Y}(\mathcal{I}_Y/\mathcal{I}_Y^2,\, \mathcal{O}_Y).$$

**Theorem.** If $Y$ and $X$ are both smooth, then $\mathcal{I}_Y/\mathcal{I}_Y^2$ is a locally free $\mathcal{O}_Y$-module of rank $n - r = \mathrm{codim}(Y,X)$.

*Proof sketch.* Locally at a point $P \in Y$, choose local parameters $t_1,\ldots,t_n$ for $X$ at $P$ such that $t_{r+1},\ldots,t_n$ are a system of local parameters for the regular ideal $\mathcal{I}_{Y,P}$. (This is possible because $Y$ is smooth and $\mathcal{I}_{Y,P}$ is generated by a regular sequence in $\mathcal{O}_{X,P}$.) Then $\mathcal{I}_{Y,P}/\mathcal{I}_{Y,P}^2$ is freely generated over $\mathcal{O}_{Y,P}$ by $[t_{r+1}],\ldots,[t_n]$, giving rank $n - r$. $\square$

**Algebraic derivation.** The conormal sheaf fits into the *conormal sequence*:

$$0 \to \mathcal{I}_Y/\mathcal{I}_Y^2 \to \Omega_{X/k}|_Y \to \Omega_{Y/k} \to 0,$$

where $\Omega_{X/k}$ is the sheaf of Kähler differentials. When $X$ and $Y$ are smooth, all three terms are locally free, and the sequence is the algebraic version of the split short exact sequence of tangent/cotangent bundles.

**Fiber description.** At a point $P \in Y$, the fiber of $\mathcal{N}_{Y/X}$ is

$$(\mathcal{N}_{Y/X})_P \otimes k(P) = T_{X,P} / T_{Y,P},$$

the quotient of tangent spaces, which is a $k$-vector space of dimension $n - r$.

> [!EXAMPLE] Normal bundle of a curve in a surface
> If $Y$ is a smooth curve on a smooth surface $X$ (so $n=2, r=1$), then $\mathcal{N}_{Y/X}$ is a line bundle on $Y$. Its degree (as a divisor on $Y$) equals the self-intersection number $Y \cdot Y$ in the surface $X$. This is the starting point of intersection theory on surfaces.

### 3.3 The Algebraic Implicit Function Theorem

**Theorem (Algebraic Implicit Function Theorem).** Let $X$ be an $n$-dimensional variety. Then $P \in X$ is smooth if and only if there exists a Zariski open neighborhood $U \ni P$ in $X$ and a closed immersion $U \hookrightarrow \mathbb{A}^n$ that is an isomorphism onto its image (i.e., $U$ is isomorphic to an open subset of $\mathbb{A}^n$).

*Proof sketch (one direction).* If $U \cong V \subset \mathbb{A}^n$ for some open $V$, then $\mathcal{O}_{X,P} \cong \mathcal{O}_{\mathbb{A}^n,P}$, which is a regular local ring (it is a localization of the polynomial ring $k[x_1,\ldots,x_n]$, hence regular). So $P$ is smooth.

*Other direction.* Suppose $P$ is smooth. If $X \subset \mathbb{A}^N$ and $X$ has codimension $c = N - n$, the Jacobian matrix $J(P)$ has rank $c$. By an implicit function theorem argument in the local ring: $c$ of the defining equations $f_1,\ldots,f_c$ form a regular sequence in $\mathcal{O}_{\mathbb{A}^N,P}$, and the quotient $\mathcal{O}_{\mathbb{A}^N,P}/(f_1,\ldots,f_c) = \mathcal{O}_{X,P}$ is a regular local ring of dimension $n$ generated by the remaining $n$ coordinates. These $n$ coordinates, restricted to $X$, form an étale (locally biholomorphic in the algebraic sense) map $X \to \mathbb{A}^n$ on some Zariski open neighborhood of $P$. $\square$

> [!WARNING] Étale vs. isomorphism
> The correct algebraic analogue is that $X$ is *étale-locally* isomorphic to $\mathbb{A}^n$ at a smooth point — not literally Zariski-locally isomorphic in general (the Zariski topology is coarser than the complex topology). However, the local ring $\mathcal{O}_{X,P}$ is *formally isomorphic* to $k\llbracket t_1,\ldots,t_n\rrbracket$, which is the algebraic substitute for the classical chart map.

### 3.4 Local Factoriality

**Theorem (Regular local rings are UFDs).** If $P \in X$ is a smooth point, then $\mathcal{O}_{X,P}$ is a unique factorization domain (UFD).

*Proof sketch.* A regular local ring is a UFD by a theorem of Auslander–Buchsbaum (1959), which proves that regular local rings have finite global dimension, and Serre's criterion then gives the UFD property. $\square$

**Geometric consequence.** Every *codimension-1 subvariety* (Weil divisor) through a smooth point $P \in X$ is *locally a principal divisor* — it is the zero locus of a single regular function in $\mathcal{O}_{X,P}$. This fails at singular points (e.g. the Weil divisors on a cone need not be Cartier).

> [!QUESTION] Exercise 5: Normal Bundle of a Hyperplane
> *This problem computes the normal bundle in a concrete case and connects it to intersection numbers.*
>
> > **Prerequisites:** [[#3.2 Smooth Subvarieties and the Normal Bundle|3.2 Smooth Subvarieties and the Normal Bundle]]
>
> Let $Y = V(x_0) \subset \mathbb{P}^n$ be a hyperplane (a smooth hypersurface). Compute $\mathcal{I}_Y/\mathcal{I}_Y^2$ on $Y \cong \mathbb{P}^{n-1}$ and identify it as a line bundle. What is its degree?

> [!TIP]- Solution to Exercise 5
> **Key insight:** The ideal sheaf of a hyperplane in projective space is a line bundle, and $\mathcal{I}_Y/\mathcal{I}_Y^2 \cong \mathcal{O}_Y(-1)$ twisted to the correct degree.
>
> **Sketch:** On $\mathbb{P}^n$, the ideal sheaf of $Y = \{x_0 = 0\}$ is $\mathcal{I}_Y = \mathcal{O}_{\mathbb{P}^n}(-1)$ (the tautological line bundle). The conormal sheaf is $\mathcal{I}_Y/\mathcal{I}_Y^2 = \mathcal{I}_Y \otimes \mathcal{O}_Y = \mathcal{O}_Y(-1)$, which is a line bundle of degree $-1$ on $Y \cong \mathbb{P}^{n-1}$. Thus $\mathcal{N}_{Y/\mathbb{P}^n} = \mathcal{O}_Y(1)$, of degree $+1$. This matches the self-intersection: $Y \cdot Y = 1$ in the Chow ring of $\mathbb{P}^n$ (the hyperplane meets itself in one point up to rational equivalence).

> [!QUESTION] Exercise 6: Factoriality Failure at a Node
> *This problem shows that the local ring at a node is not a UFD, illustrating the failure of local factoriality at singular points.*
>
> > **Prerequisites:** [[#3.4 Local Factoriality|3.4 Local Factoriality]], [[#1.6 Examples: Nodal and Cuspidal Cubics|1.6 Examples: Nodal and Cuspidal Cubics]]
>
> Let $R = k[x,y]/(y^2 - x^2(x+1))$ localized at $(x,y)$. Show that $R$ is not a UFD by exhibiting two distinct factorizations of $y^2$ in $R$.

> [!TIP]- Solution to Exercise 6
> **Key insight:** At a node, the two branches of the curve give two distinct elements that multiply to $y^2$, breaking unique factorization.
>
> **Sketch:** In $R$, we have $y^2 = x^2(x+1)$. Near the origin, $x+1$ is a unit (since $(x+1)(0) = 1 \neq 0$), so $x^2(x+1) = x \cdot x(x+1)^{1/2} \cdot (x+1)^{1/2}$ formally. More concretely: the two branches of the node are $y = x\sqrt{x+1}$ and $y = -x\sqrt{x+1}$, corresponding to elements $\alpha = y - x\sqrt{x+1}$ and $\beta = y + x\sqrt{x+1}$ in an étale cover. Then $\alpha \beta = y^2 - x^2(x+1) = 0$ in $R$, so $\alpha$ and $\beta$ are zero divisors in the normalization — but in $R$ itself, the factorization $y^2 = y \cdot y$ and $y^2 = x \cdot x(x+1)$ are two genuinely different factorizations not related by units in $R$, since $y$ and $x$ are not associate in $R$.

---

## 4. The Tangent Cone

### 4.1 Definition via Initial Forms

📐 At a singular point, the Zariski tangent space is "too large" — it does not distinguish the local geometry of the singularity. The *tangent cone* provides a finer first-order invariant.

**Setup.** Let $P \in X$ be a (possibly singular) point. We may assume $P = 0 \in \mathbb{A}^n$ after a translation. Every $f \in I(X) \subset k[x_1,\ldots,x_n]$ vanishing at $P$ can be written as $f = f_m + f_{m+1} + \cdots$, where $f_d$ is the homogeneous component of degree $d$ and $m = \mathrm{ord}_P(f) \geq 1$ is the order of vanishing.

**Definition (Initial Form).** For $f \in \mathfrak{m}_{\mathbb{A}^n,P}$, the *initial form* (or *leading form*) of $f$ at $P$ is the lowest-degree homogeneous component $f_m$ where $m = \mathrm{ord}_P(f)$.

**Definition (Tangent Cone).** The *tangent cone* $C_{X,P}$ to $X$ at $P$ is the algebraic set

$$C_{X,P} = V\!\left(\mathrm{in}(I(X))\right) \subset \mathbb{A}^n,$$

where $\mathrm{in}(I(X)) = \{ \mathrm{in}(f) : f \in I(X) \}$ denotes the *initial ideal*, i.e. the ideal generated by all initial forms of elements of $I(X)$.

*Remark.* The tangent cone is a *cone* in the sense that it is stable under the scaling $\mathbf{x} \mapsto \lambda \mathbf{x}$ for $\lambda \in k^*$, because $\mathrm{in}(f)$ is homogeneous.

> [!WARNING] Initial ideal vs. span of initial forms
> The initial ideal $\mathrm{in}(I)$ is the ideal generated by $\{\mathrm{in}(f) : f \in I\}$, which is generally larger than $\mathrm{span}\{\mathrm{in}(f_i)\}$ for a finite set of generators $f_i$ of $I$. Computing $\mathrm{in}(I)$ correctly requires Gröbner basis techniques or careful analysis of the ideal.

### 4.2 The Associated Graded Ring

🔑 The algebraic description of the tangent cone is via the *associated graded ring*.

**Definition (Associated Graded Ring).** For a local ring $(\mathcal{O}_{X,P}, \mathfrak{m})$, the *associated graded ring* is

$$\mathrm{gr}_\mathfrak{m}\, \mathcal{O}_{X,P} = \bigoplus_{d \geq 0} \mathfrak{m}^d/\mathfrak{m}^{d+1},$$

with multiplication $[\phi][\psi] = [\phi\psi]$ for $\phi \in \mathfrak{m}^a/\mathfrak{m}^{a+1}$, $\psi \in \mathfrak{m}^b/\mathfrak{m}^{b+1}$. This is a graded $k$-algebra (with degree-0 part $\mathcal{O}_{X,P}/\mathfrak{m} \cong k$).

**Theorem (Tangent cone as Spec of associated graded).** There is a canonical isomorphism

$$C_{X,P} \cong \mathrm{Spec}\!\left(\mathrm{gr}_\mathfrak{m}\, \mathcal{O}_{X,P}\right)$$

as algebraic varieties (or schemes). Precisely: if $X \subset \mathbb{A}^n$ and $P = 0$, then

$$\mathrm{gr}_\mathfrak{m}\, \mathcal{O}_{X,P} \cong k[x_1,\ldots,x_n]/\mathrm{in}(I(X)).$$

*Proof sketch.* The associated graded of the local ring of $\mathbb{A}^n$ at the origin is $k[x_1,\ldots,x_n]$ itself (since the polynomial ring is already graded). The ideal $I(X)$ acts on successive quotients $\mathfrak{m}^d/\mathfrak{m}^{d+1}$ via its initial forms, giving exactly $k[x_1,\ldots,x_n]/\mathrm{in}(I(X))$ as the associated graded of the quotient. $\square$

**Relationship to the tangent space.** At a smooth point $P$, $\mathrm{gr}_\mathfrak{m}\, \mathcal{O}_{X,P} \cong k[t_1,\ldots,t_n]$ (the polynomial ring in the local parameters), since the associated graded of a regular local ring is a polynomial ring. This means $C_{X,P} = T_{X,P}$ at smooth points — the tangent cone equals the tangent space.

### 4.3 Multiplicity

**Definition (Multiplicity).** The *multiplicity* of $X$ at $P$, denoted $\mathrm{mult}_P(X)$, is the degree of the Hilbert polynomial of the associated graded ring $\mathrm{gr}_\mathfrak{m}\,\mathcal{O}_{X,P}$. Equivalently, it is the *degree* of the tangent cone $C_{X,P}$ as a projective variety (via the projectivization of the leading ideal).

For a hypersurface $X = V(f) \subset \mathbb{A}^n$ with $P = 0$:

$$\mathrm{mult}_P(X) = \mathrm{ord}_P(f) = \min\{ d : f_d \neq 0 \},$$

the order of vanishing of the defining polynomial at $P$.

**Theorem (Multiplicity and smoothness).** $\mathrm{mult}_P(X) = 1$ if and only if $P$ is a smooth point of $X$.

*Proof sketch.* $\mathrm{mult}_P(X) = 1$ means the leading form $f_1$ is linear, i.e. $\nabla f(P) \neq 0$, which is exactly the Jacobian criterion for smoothness of the hypersurface. For the general case, $\mathrm{mult}_P(X) = 1$ iff $\mathrm{gr}_\mathfrak{m}\,\mathcal{O}_{X,P}$ is generated in degree 1, iff $\dim_k(\mathfrak{m}/\mathfrak{m}^2) = \dim X$, i.e. regularity. $\square$

**Remark (semicontinuity of multiplicity).** *Multiplicity is upper semicontinuous in families:* in a flat family $\mathcal{X} \to T$, the function $P \mapsto \mathrm{mult}_P(\mathcal{X}_t)$ can only jump up at special values of $t$. This is the algebraic version of "singularities can be created but not destroyed by specialization."

### 4.4 Examples

**Example (Node, revisited).** For $X = V(y^2 - x^2(x+1))$ at $P = (0,0)$:

The defining polynomial is $f = y^2 - x^2(x+1) = y^2 - x^2 - x^3$. The leading form is $f_2 = y^2 - x^2 = (y-x)(y+x)$. So

$$C_{X,(0,0)} = V(y^2 - x^2) = V(y-x) \cup V(y+x),$$

two distinct lines through the origin. This reflects the two smooth branches of the node: $y \approx x$ and $y \approx -x$ near $(0,0)$. The multiplicity is $\mathrm{mult}_{(0,0)}(X) = 2$.

**Example (Cusp).** For $X = V(y^2 - x^3)$ at $P = (0,0)$:

The leading form is $f_2 = y^2$, so $C_{X,(0,0)} = V(y^2)$, a *double line* along the $x$-axis. The multiplicity is $2$. The tangent cone is a non-reduced scheme, reflecting the fact that the cusp has only one branch but it is "tangent to itself."

**Example (Whitney Umbrella).** For $X = V(x^2 - y^2z)$ at $P = (0,0,0)$:

The polynomial $f = x^2 - y^2z$ has lowest-degree term $f_2 = x^2$ (since $y^2 z$ has degree 3). So $C_{X,(0,0,0)} = V(x^2)$, a double plane (the $yz$-plane with multiplicity 2). The multiplicity is $2$.

> [!NOTE] Tangent space vs. tangent cone
> For the cusp: $T_{X,(0,0)} = k^2$ (the full plane, since $J = (0,0)$ at the origin), while $C_{X,(0,0)} = V(y^2)$ is a (double) line. The tangent space sees dimension but misses the directional structure; the tangent cone sees both.

> [!QUESTION] Exercise 7: Tangent Cone of a Quartic
> *This problem asks for the tangent cone at a singular point and identification of its components.*
>
> > **Prerequisites:** [[#4.1 Definition via Initial Forms|4.1 Definition via Initial Forms]], [[#4.3 Multiplicity|4.3 Multiplicity]]
>
> Let $X = V(x^4 + y^4 - x^2y^2) \subset \mathbb{A}^2$ at $P = (0,0)$. Compute the tangent cone $C_{X,P}$ and the multiplicity $\mathrm{mult}_P(X)$. How many branches does $X$ have at $P$?

> [!TIP]- Solution to Exercise 7
> **Key insight:** The initial form determines both the tangent cone and the multiplicity; its irreducible factors over $k$ correspond to the branches.
>
> **Sketch:** The polynomial is $f = x^4 + y^4 - x^2y^2$. The terms of degree 2 are: $-x^2y^2$ (degree 4 — wait, $-x^2y^2$ has degree 4, not 2). In fact, $f_2 = 0$, $f_3 = 0$, and $f_4 = x^4 - x^2y^2 + y^4$. So the initial form is $f_4 = x^4 - x^2y^2 + y^4$ and $\mathrm{mult}_P(X) = 4$. Factoring over $\mathbb{C}$: $f_4 = x^4 - x^2y^2 + y^4$. Setting $u = x/y$: $u^4 - u^2 + 1 = 0$, so $u^2 = (1 \pm \sqrt{-3})/2 = e^{\pm i\pi/3}$ or $e^{\pm 2i\pi/3}$ (complex roots of unity). Thus $f_4$ factors into 4 distinct linear factors over $\mathbb{C}$, giving $C_{X,P}$ as a union of 4 lines. The variety $X$ has 4 branches at $P$.

> [!QUESTION] Exercise 8: Associated Graded of a Regular Ring
> *This problem computes the associated graded ring at a smooth point and confirms it is a polynomial ring.*
>
> > **Prerequisites:** [[#4.2 The Associated Graded Ring|4.2 The Associated Graded Ring]], [[#2.1 Local Parameters|2.1 Local Parameters]]
>
> Let $R = k[x,y]_{(x,y)}$ (the local ring of $\mathbb{A}^2$ at the origin). Compute $\mathrm{gr}_\mathfrak{m} R$ where $\mathfrak{m} = (x,y)$, and show it is isomorphic to $k[s,t]$ (a polynomial ring in two variables).

> [!TIP]- Solution to Exercise 8
> **Key insight:** For a regular local ring, the associated graded is always a polynomial ring, confirming that the tangent cone at a smooth point is the tangent space.
>
> **Sketch:** We have $\mathfrak{m}^d/\mathfrak{m}^{d+1} \cong k^{\binom{d+1}{1}} = $ the space of homogeneous polynomials of degree $d$ in $x, y$. The graded pieces are $\mathfrak{m}^0/\mathfrak{m}^1 = k$ (degree 0), $\mathfrak{m}/\mathfrak{m}^2 = k \cdot [x] \oplus k \cdot [y]$ (degree 1), $\mathfrak{m}^2/\mathfrak{m}^3 = k[x]^2 \oplus k[x][y] \oplus k[y]^2$ (degree 2), etc. Thus $\mathrm{gr}_\mathfrak{m} R = \bigoplus_d \mathfrak{m}^d/\mathfrak{m}^{d+1} \cong k[x,y]$ (the polynomial ring), where $[x], [y]$ are the degree-1 generators. No relations arise because $k[x,y]_{(x,y)}$ is a domain and the monomials $x^a y^b$ remain linearly independent in each graded piece.

---

## 5. Discrete Valuation Rings and Smooth Curves

### 5.1 DVRs: Definitions and Characterizations

🔑 Discrete valuation rings are the algebraic avatars of smooth points on curves.

**Definition (Discrete Valuation).** A *discrete valuation* on a field $K$ is a surjective homomorphism $v : K^\times \to \mathbb{Z}$ satisfying the ultrametric inequality:

$$v(a + b) \geq \min(v(a), v(b)) \quad \text{for all } a, b \in K^\times.$$

(Extend to $K$ by $v(0) = +\infty$.)

**Definition (Discrete Valuation Ring).** The *ring of integers* (or *valuation ring*) of $v$ is

$$R = \{ a \in K : v(a) \geq 0 \} \cup \{0\}.$$

A ring $R$ is a *discrete valuation ring* (DVR) if it arises this way for some discrete valuation on $K = \mathrm{Frac}(R)$.

**Theorem (Equivalent Characterizations).** For a Noetherian local domain $(R, \mathfrak{m})$, the following are equivalent:

1. $R$ is a DVR.
2. $R$ is a principal ideal domain with a unique nonzero prime ideal.
3. $R$ is a regular local ring of Krull dimension 1.
4. $\mathfrak{m}$ is principal (generated by a single element $t$, the *uniformizer* or *local parameter*).
5. Every nonzero element $f \in R$ can be written uniquely as $f = u \cdot t^n$ for $u \in R^\times$ and $n \geq 0$.

*Proof sketch (3 $\Rightarrow$ 1).* Regularity of dimension 1 means $\mathfrak{m} = (t)$ for a single generator. By the unique factorization property of regular local rings (which are UFDs), every nonzero $f \in R$ writes as $f = u t^{v(f)}$ for a unit $u$ and $v(f) \geq 0$. Extending to $\mathrm{Frac}(R)$ by $v(f/g) = v(f) - v(g)$ gives a discrete valuation. $\square$

> [!NOTE] DVRs and completions
> The completion of a DVR $R$ with uniformizer $t$ and residue field $k$ is $\hat{R} \cong k\llbracket t\rrbracket$ (formal power series), by the Cohen structure theorem. This is the source of the connection between DVRs and formal power series.

### 5.2 Smooth Points of Curves are DVRs

**Theorem.** Let $C$ be an irreducible curve and $P \in C$ a smooth point. Then $\mathcal{O}_{C,P}$ is a DVR.

*Proof.* Since $\dim C = 1$ and $P$ is smooth, $\mathcal{O}_{C,P}$ is a regular local ring of Krull dimension 1. By characterization 3 in the theorem above, $\mathcal{O}_{C,P}$ is a DVR. The uniformizer is any local parameter $t \in \mathfrak{m}_{C,P} \setminus \mathfrak{m}_{C,P}^2$. $\square$

**Explicit construction.** In an affine model $C \subset \mathbb{A}^2$ near $P$: a local parameter $t$ is any function in $A(C)$ that vanishes to exactly first order at $P$. For example, if $C = V(f)$ with $(\partial f/\partial y)(P) \neq 0$, one may take $t = x - P_x$ (restricted to $C$).

The *formal completion* satisfies $\hat{\mathcal{O}}_{C,P} \cong k\llbracket t\rrbracket$.

> [!EXAMPLE] DVR on the parabola
> Let $C = V(y - x^2) \subset \mathbb{A}^2$ and $P = (0,0)$. The coordinate ring is $k[x,y]/(y-x^2) \cong k[x]$, so $\mathcal{O}_{C,P} = k[x]_{(x)}$, with maximal ideal $(x)$. The uniformizer is $t = x$. Every function $f \in \mathcal{O}_{C,P}$ can be written as $f = x^n \cdot u$ for a unit $u$ and $n = v_P(f) \geq 0$.

### 5.3 The Valuation and Order of Vanishing

**Definition (Order of Vanishing).** For $P$ a smooth point of a curve $C$ and $f \in \mathcal{O}_{C,P}$, the *order of vanishing* of $f$ at $P$ is

$$v_P(f) = \max\{ n \geq 0 : f \in \mathfrak{m}_{C,P}^n \},$$

i.e. the unique integer $n$ such that $f = u t^n$ for a unit $u \in \mathcal{O}_{C,P}^\times$ and uniformizer $t$. Extend to rational functions $\phi = f/g \in k(C)$ by $v_P(\phi) = v_P(f) - v_P(g)$.

**Properties of the valuation:**

1. $v_P(\phi \psi) = v_P(\phi) + v_P(\psi)$
2. $v_P(\phi + \psi) \geq \min(v_P(\phi), v_P(\psi))$, with equality when $v_P(\phi) \neq v_P(\psi)$
3. $v_P(\phi) > 0$ iff $\phi$ has a zero at $P$; $v_P(\phi) < 0$ iff $\phi$ has a pole at $P$

**Connection to intersection numbers.** The intersection number of a curve $D$ (defined by $f = 0$) with $C$ at $P$ is $I_P(C,D) = v_P(f|_C)$, the order of vanishing of $f$ on $C$ at $P$. This connects DVR valuations to intersection theory from Chapter I of Shafarevich (see [[concepts/algebraic-geometry/note|Classical Algebraic Geometry]], §6 on intersection numbers).

> [!INFO] Global picture: Weil divisors on a smooth curve
> For a smooth projective curve $C$, summing the local data gives a global *Weil divisor* associated to a nonzero rational function $\phi \in k(C)^\times$: the *principal divisor* $(\phi) = \sum_{P \in C} v_P(\phi) \cdot [P]$. Since $\phi$ has only finitely many zeros and poles, this is a well-defined divisor. **The product formula $\sum_P v_P(\phi) = 0$ is the algebraic version of the fact that a meromorphic function on a compact Riemann surface has as many zeros as poles** (counting multiplicities). This is the foundational result of divisor theory on smooth curves.

> [!QUESTION] Exercise 9: Uniformizer on a Twisted Cubic
> *This problem computes the local parameter and the DVR structure at a smooth point of a space curve.*
>
> > **Prerequisites:** [[#5.2 Smooth Points of Curves are DVRs|5.2 Smooth Points of Curves are DVRs]], [[#5.3 The Valuation and Order of Vanishing|5.3 The Valuation and Order of Vanishing]]
>
> Let $C = \{(t, t^2, t^3) : t \in k\} \subset \mathbb{A}^3$ be the twisted cubic, parametrized by $t$. The coordinate ring is $A(C) = k[x,y,z]/(y-x^2, z-x^3) \cong k[x]$. Fix the smooth point $P = (0,0,0)$. Show that $\mathcal{O}_{C,P} \cong k[x]_{(x)}$ is a DVR with uniformizer $x$, and compute $v_P(y)$ and $v_P(z)$.

> [!TIP]- Solution to Exercise 9
> **Key insight:** The parametrization $t \mapsto (t, t^2, t^3)$ shows that $x$ generates the maximal ideal at $P$, while $y = x^2$ and $z = x^3$ have higher-order vanishing.
>
> **Sketch:** Since $A(C) \cong k[x]$ (an isomorphism of rings), we have $\mathcal{O}_{C,P} = k[x]_{(x)}$. The maximal ideal is $(x)$, so $t = x$ is the uniformizer. Then $v_P(x) = 1$, and since $y = x^2$ in $A(C)$, we get $v_P(y) = v_P(x^2) = 2$. Similarly, $z = x^3$, so $v_P(z) = 3$. This illustrates the general principle: for a rational normal curve parametrized by $t$, the valuation at the origin assigns $v_P(x_i) = i+1$ for coordinate $x_i = t^{i+1}$.

> [!QUESTION] Exercise 10: Valuation and Intersection
> *This problem connects the DVR valuation to intersection numbers from Chapter I.*
>
> > **Prerequisites:** [[#5.3 The Valuation and Order of Vanishing|5.3 The Valuation and Order of Vanishing]]
>
> Let $C = V(y - x^2) \subset \mathbb{A}^2$ and $L = V(y) \subset \mathbb{A}^2$. Compute $v_P(y|_C)$ at $P = (0,0)$ using the DVR structure on $\mathcal{O}_{C,P}$, and interpret the result as the intersection number $I_P(C,L)$.

> [!TIP]- Solution to Exercise 10
> **Key insight:** Restricting the equation of $L$ to $C$ and computing its order of vanishing gives the intersection multiplicity directly.
>
> **Sketch:** On $C$, we have $y = x^2$ (the defining relation). The uniformizer at $P$ is $t = x$. So $y|_C = x^2 = t^2$ in $\mathcal{O}_{C,P} = k[x]_{(x)}$. Thus $v_P(y|_C) = 2$. This gives $I_P(C,L) = 2$, meaning $C$ and $L$ meet at the origin with intersection multiplicity 2 — geometrically, the parabola is tangent to the $x$-axis, and they meet to second order.

---

## Mathematical Development Exercises

> [!QUESTION] Exercise 11: Derivation Description of Tangent Space
> *This problem establishes the equivalence between the dual-of-cotangent-space and the derivation descriptions of the tangent space.*
>
> > **Prerequisites:** [[#1.2 The Cotangent Space and Zariski Tangent Space|1.2 The Cotangent Space and Zariski Tangent Space]]
>
> Let $(R, \mathfrak{m}, k)$ be a local $k$-algebra. Show that there is a natural isomorphism
> $$(\mathfrak{m}/\mathfrak{m}^2)^\vee \cong \mathrm{Der}_k(R, k),$$
> where $k$ is viewed as an $R$-module via the quotient $R \to R/\mathfrak{m} \cong k$.

> [!TIP]- Solution to Exercise 11
> **Key insight:** A derivation $\delta : R \to k$ must kill constants ($\delta(k) = 0$ by linearity and $\delta(1) = 0$ from Leibniz) and kills $\mathfrak{m}^2$ by Leibniz, so it factors through $\mathfrak{m}/\mathfrak{m}^2$.
>
> **Sketch:** Given $\delta \in \mathrm{Der}_k(R,k)$: for $c \in k \subset R$, $\delta(c) = \delta(c \cdot 1) = c\delta(1)$ and $\delta(1) = \delta(1 \cdot 1) = 1 \cdot \delta(1) + 1 \cdot \delta(1) = 2\delta(1)$, forcing $\delta(1) = 0$. For $f, g \in \mathfrak{m}$: $\delta(fg) = f(P)\delta(g) + g(P)\delta(f) = 0 + 0 = 0$ (since $f(P) = g(P) = 0$). Thus $\delta|_{\mathfrak{m}^2} = 0$, and $\delta$ factors through $\mathfrak{m}/\mathfrak{m}^2$, giving a $k$-linear functional. Conversely, any linear functional $\lambda : \mathfrak{m}/\mathfrak{m}^2 \to k$ extends to a derivation via $\delta(f) = \lambda(f - f(P) \bmod \mathfrak{m}^2)$. These maps are mutually inverse, establishing the isomorphism.

> [!QUESTION] Exercise 12: Invariance of the Tangent Space
> *This problem shows that the tangent space is intrinsic to the variety, independent of the embedding.*
>
> > **Prerequisites:** [[#1.2 The Cotangent Space and Zariski Tangent Space|1.2 The Cotangent Space and Zariski Tangent Space]]
>
> Let $\phi : X \to Y$ be an isomorphism of varieties and $P \in X$, $Q = \phi(P) \in Y$. Show that $\phi$ induces a $k$-linear isomorphism $d\phi_P : T_{X,P} \xrightarrow{\sim} T_{Y,Q}$.

> [!TIP]- Solution to Exercise 12
> **Key insight:** An isomorphism of varieties induces an isomorphism on local rings, hence on maximal ideals and their squares, hence on cotangent and tangent spaces.
>
> **Sketch:** The isomorphism $\phi$ induces an isomorphism on sheaves of regular functions, hence an isomorphism of local rings $\phi^* : \mathcal{O}_{Y,Q} \xrightarrow{\sim} \mathcal{O}_{X,P}$ with $\phi^*(\mathfrak{m}_{Y,Q}) = \mathfrak{m}_{X,P}$. This descends to an isomorphism $\mathfrak{m}_{Y,Q}/\mathfrak{m}_{Y,Q}^2 \xrightarrow{\sim} \mathfrak{m}_{X,P}/\mathfrak{m}_{X,P}^2$ of cotangent spaces. Taking duals gives the tangent space isomorphism $d\phi_P : T_{X,P} \xrightarrow{\sim} T_{Y,Q}$.

> [!QUESTION] Exercise 13: The Singular Locus is Closed
> *This problem gives a direct algebraic proof that the singular locus is Zariski closed.*
>
> > **Prerequisites:** [[#1.3 The Jacobian Criterion|1.3 The Jacobian Criterion]], [[#1.4 Regular Local Rings|1.4 Regular Local Rings]]
>
> Let $X = V(f_1,\ldots,f_m) \subset \mathbb{A}^n$ (not necessarily irreducible). Show that $\mathrm{Sing}(X) = V(f_1,\ldots,f_m,\, \text{all } (n-\dim X) \times (n-\dim X) \text{ minors of } J)$, where $J = (\partial f_i/\partial x_j)$, and conclude that $\mathrm{Sing}(X)$ is Zariski closed in $X$.

> [!TIP]- Solution to Exercise 13
> **Key insight:** Smoothness is characterized by the non-vanishing of at least one maximal minor of the Jacobian — so singularity is the vanishing of *all* maximal minors, a closed condition.
>
> **Sketch:** A point $P \in X$ is smooth iff $\mathrm{rank}(J(P)) \geq c = n - \dim X$ (the codimension). Equivalently, $P$ is singular iff every $(c \times c)$-minor of $J$ vanishes at $P$. Each minor is a polynomial function of the coordinates of $P$, so the locus where all such minors vanish is a closed algebraic set. The singular locus equals this closed set intersected with $X$, hence is closed in $X$.

> [!QUESTION] Exercise 14: Completion of a DVR
> *This problem verifies the Cohen structure theorem in the concrete DVR case.*
>
> > **Prerequisites:** [[#5.1 DVRs: Definitions and Characterizations|5.1 DVRs: Definitions and Characterizations]], [[#2.2 The Formal Completion|2.2 The Formal Completion]]
>
> Let $R = k[t]_{(t)}$ (the local ring of $\mathbb{A}^1$ at the origin). Show directly that $\hat{R} \cong k\llbracket t\rrbracket$ by constructing a ring isomorphism.

> [!TIP]- Solution to Exercise 14
> **Key insight:** Elements of the completion are coherent sequences of polynomials modulo $t^n$, which precisely encode the coefficients of a formal power series.
>
> **Sketch:** By definition, $\hat{R} = \varprojlim_n R/(t^n) = \varprojlim_n k[t]/(t^n)$. An element of $\hat{R}$ is a compatible sequence $(a_0, a_0 + a_1 t, a_0 + a_1 t + a_2 t^2, \ldots)$ of polynomials modulo successive powers of $t$. This is precisely a formal power series $\sum_{n \geq 0} a_n t^n \in k\llbracket t\rrbracket$. The map $\hat{R} \to k\llbracket t\rrbracket$ sending such a sequence to $\sum a_n t^n$ is clearly a ring isomorphism (adding and multiplying sequences corresponds to adding and multiplying formal power series).

> [!QUESTION] Exercise 15: Multiplicity from the Hilbert Function
> *This problem derives the multiplicity from the Hilbert function of the associated graded ring.*
>
> > **Prerequisites:** [[#4.2 The Associated Graded Ring|4.2 The Associated Graded Ring]], [[#4.3 Multiplicity|4.3 Multiplicity]]
>
> For the node $X = V(y^2 - x^2(x+1))$ at $P = (0,0)$, compute $\dim_k(\mathfrak{m}^n/\mathfrak{m}^{n+1})$ for $n = 0, 1, 2$, where $\mathfrak{m} = \mathfrak{m}_{X,P}$. Verify that the Hilbert function stabilizes with value $\mathrm{mult}_P(X) = 2$.

> [!TIP]- Solution to Exercise 15
> **Key insight:** The associated graded ring of the node is $k[x,y]/(y^2-x^2)$, and its graded pieces have Hilbert function eventually equal to the multiplicity.
>
> **Sketch:** The tangent cone is $C = V(y^2-x^2)$ and $\mathrm{gr}_\mathfrak{m}\,\mathcal{O}_{X,P} \cong k[x,y]/(y^2-x^2)$. In degree 0: $k[x,y]/(y^2-x^2)$ in degree 0 is $k$, so $\dim = 1$. In degree 1: basis $\{x, y\}$ (no relations in degree 1), $\dim = 2$. In degree 2: basis $\{x^2, xy, y^2\} = \{x^2, xy, x^2\}$ — but $y^2 = x^2$ in the quotient, so $\{x^2, xy\}$, giving $\dim = 2$. For all $n \geq 1$, $\dim_k(\mathfrak{m}^n/\mathfrak{m}^{n+1}) = 2$. The Hilbert polynomial is the constant $2 = \mathrm{mult}_P(X)$.

> [!QUESTION] Exercise 16: Cotangent Sequence at a Smooth Subvariety
> *This problem derives the conormal sequence from first principles and uses it to compute the normal bundle.*
>
> > **Prerequisites:** [[#3.2 Smooth Subvarieties and the Normal Bundle|3.2 Smooth Subvarieties and the Normal Bundle]]
>
> Let $X = \mathbb{A}^n$ and $Y = V(x_1,\ldots,x_r) \cong \mathbb{A}^{n-r}$ (a coordinate linear subspace). Write down the conormal sequence $0 \to \mathcal{I}_Y/\mathcal{I}_Y^2 \to \Omega_{X/k}|_Y \to \Omega_{Y/k} \to 0$ explicitly in terms of coordinates, and verify that $\mathcal{I}_Y/\mathcal{I}_Y^2$ is free of rank $r$.

> [!TIP]- Solution to Exercise 16
> **Key insight:** For a linear subspace, all sheaves in the conormal sequence are trivial vector bundles, and the sequence splits.
>
> **Sketch:** $\mathcal{I}_Y = (x_1,\ldots,x_r)$ and $\mathcal{I}_Y^2 = (x_ix_j : 1 \leq i,j \leq r)$. So $\mathcal{I}_Y/\mathcal{I}_Y^2$ is freely generated over $\mathcal{O}_Y$ by $[x_1],\ldots,[x_r]$ (no relation since $x_ix_j \in \mathcal{I}_Y^2$ for all $i,j$). This is free of rank $r$. The sheaf $\Omega_{X/k}|_Y$ is free of rank $n$ on $Y$ with basis $\{dx_1,\ldots,dx_n\}$, and $\Omega_{Y/k}$ is free of rank $n-r$ with basis $\{dx_{r+1},\ldots,dx_n\}$. The conormal sequence is $0 \to \mathcal{O}_Y^{\oplus r} \to \mathcal{O}_Y^{\oplus n} \to \mathcal{O}_Y^{\oplus (n-r)} \to 0$ (split exact, via the direct sum decomposition of $\Omega_{X/k}$).

> [!QUESTION] Exercise 17: Tangent Cone and Blowing Up
> *This problem connects the tangent cone to the exceptional divisor of the blow-up, linking the local and global theories.*
>
> > **Prerequisites:** [[#4.2 The Associated Graded Ring|4.2 The Associated Graded Ring]]
>
> Let $\widetilde{\mathbb{A}^2}$ be the blow-up of $\mathbb{A}^2$ at the origin, with exceptional divisor $E \cong \mathbb{P}^1$. Show that the strict transform of a curve $C \subset \mathbb{A}^2$ passing through $P = (0,0)$ meets $E$ at the points corresponding to the tangent directions of the tangent cone $C_{C,P}$.

> [!TIP]- Solution to Exercise 17
> **Key insight:** The blow-up replaces a point by the projectivization of its tangent space; the strict transform of a curve approaches the blown-up point along the tangent directions of its initial form.
>
> **Sketch:** Recall $\widetilde{\mathbb{A}^2} = \{(x,y; [u:v]) \in \mathbb{A}^2 \times \mathbb{P}^1 : xv = yu\}$ with $E = \{0\} \times \mathbb{P}^1$. In the chart $v = 1$: coordinates $(u, y)$ with $x = uy$, and $E \cap U_v = \{y = 0\}$. If $C = V(f)$ with $f = f_m + \text{h.o.t.}$, then in the chart $v=1$: $f(uy, y) = y^m(f_m(u,1) + y(\cdots))$. The factor $y^m$ gives $E^m$ (the exceptional divisor with multiplicity $m$), and the strict transform is $V(f_m(u,1) + y(\cdots))$. At $y = 0$, this meets $E$ at the zeros of $f_m(u,1) = 0$, i.e. the slopes $u = x/y$ solving the leading form — exactly the directions in the tangent cone.

> [!QUESTION] Exercise 18: Regularity vs. Normality
> *This problem illustrates that regular implies normal, but not conversely, with an explicit example.*
>
> > **Prerequisites:** [[#1.4 Regular Local Rings|1.4 Regular Local Rings]], [[#3.1 Analytical Irreducibility|3.1 Analytical Irreducibility]]
>
> Let $R = k[x,y]/(y^2 - x^3 - x^2)$ localized at $(x,y)$ (local ring of the nodal cubic at the origin), and let $\tilde{R}$ be its integral closure in $\mathrm{Frac}(R)$. Show $R \subsetneq \tilde{R}$, so $R$ is not normal. Then let $S = k[x,y]/(y^2 - x^3)$ localized at $(x,y)$ (cuspidal cubic). Show $S$ is also not normal, but that the normalization of $S$ is a DVR (hence regular).

> [!TIP]- Solution to Exercise 18
> **Key insight:** Normalization of a singular curve replaces the singular point by smooth points on the branches; the node's normalization separates two branches, while the cusp's normalization smooths the single branch.
>
> **Sketch:** For the node $R$: in $\mathrm{Frac}(R)$, we have $y^2 = x^2(x+1)$, so $t = y/x$ satisfies $t^2 = x+1$, hence $t^2 - 1 = x$ and $t^3 - t = y$. So $t \in \mathrm{Frac}(R)$ is integral over $R$ ($t^2 = x + 1 \in R$) but $t \notin R$ (since $R$ does not contain $y/x$). The normalization is $k[t]_{(t-1)} \times k[t]_{(t+1)}$ (two DVRs). For the cusp $S$: $y^2 = x^3$, so $s = y/x$ satisfies $s^2 = x$, giving $s^3 = y$. Thus $s \in \mathrm{Frac}(S)$ is integral over $S$ with $k[s] \cong$ the normalization; localized at $(s)$ this gives $k[s]_{(s)}$, a DVR (regular), so the normalization of the cusp is smooth.

---

## Algorithmic Applications Exercises

> [!QUESTION] Exercise 19: Computing the Smooth Locus
> *This problem implements the Jacobian criterion to compute the smooth locus of a given variety.*
>
> > **Prerequisites:** [[#1.3 The Jacobian Criterion|1.3 The Jacobian Criterion]], [[#1.5 The Smooth Locus is Open and Dense|1.5 The Smooth Locus is Open and Dense]]
>
> Design a Python pseudocode algorithm that, given a polynomial $f \in \mathbb{Z}[x_1,\ldots,x_n]$ defining a hypersurface $X = V(f) \subset \mathbb{A}^n$, outputs the ideal $I(\mathrm{Sing}(X)) = (f, \partial f/\partial x_1, \ldots, \partial f/\partial x_n)$ and uses a Gröbner basis computation to determine whether $\mathrm{Sing}(X)$ is empty (i.e., $X$ is smooth).

> [!TIP]- Solution to Exercise 19
> **Key insight:** The singular locus of a hypersurface is cut out by $f$ together with all its partials; a Gröbner basis computation checks if the resulting ideal is the unit ideal (empty variety).
>
> **Sketch:**
> ```python
> from sympy import symbols, diff, groebner, Poly, S
>
> def singular_locus_hypersurface(f, variables):
>     """
>     Input: f - polynomial in variables (as sympy expression)
>            variables - list of sympy symbols [x1, ..., xn]
>     Output: (generators of Sing(X) ideal, is_smooth bool)
>     """
>     # Build the ideal generators: f and all partial derivatives
>     generators = [f] + [diff(f, xi) for xi in variables]
>
>     # Compute Gröbner basis over QQ
>     G = groebner(generators, *variables, domain='QQ')
>
>     # X is smooth iff the ideal is the unit ideal, i.e. 1 is in the basis
>     is_smooth = (S.One in G)
>
>     return generators, is_smooth
>
> # Example: nodal cubic y^2 - x^2*(x+1)
> x, y = symbols('x y')
> f = y**2 - x**2 * (x + 1)
> gens, smooth = singular_locus_hypersurface(f, [x, y])
> # gens = [y^2 - x^3 - x^2, -3x^2 - 2x, 2y]
> # Groebner basis will contain {x, y}, not 1, so is_smooth = False
> ```
> The Gröbner basis of $(f, -3x^2-2x, 2y)$ over $\mathbb{Q}$ contains $\{y, x\}$ (or $\{y, x(3x+2)\}$), not $\{1\}$, so $\mathrm{Sing}(X) = \{(0,0)\}$ is nonempty and $X$ is singular.

> [!QUESTION] Exercise 20: Power Series Truncation
> *This problem implements the power series expansion of a regular function up to a given order.*
>
> > **Prerequisites:** [[#2.3 Power Series Expansion of Regular Functions|2.3 Power Series Expansion of Regular Functions]], [[#2.1 Local Parameters|2.1 Local Parameters]]
>
> Design a Python pseudocode algorithm that, given a rational function $f = p/q$ with $q(P) \neq 0$, a smooth point $P$, and local parameters $t_1,\ldots,t_n$, computes the Taylor expansion of $f$ in $t_1,\ldots,t_n$ up to total degree $d$.

> [!TIP]- Solution to Exercise 20
> **Key insight:** A rational function $f = p/q$ with $q(P) \neq 0$ can be expanded by substituting $t_i = x_i - P_i$ (or more general local parameters) and using polynomial division modulo $\mathfrak{m}^{d+1}$.
>
> **Sketch:**
> ```python
> from sympy import symbols, series, Poly, expand, factor
>
> def power_series_expansion(p, q, point, local_params, ambient_vars, degree):
>     """
>     Input: p, q - polynomials (f = p/q), q(point) != 0
>            point - dict {xi: P_i} giving coordinates of P
>            local_params - list of expressions for t_i in terms of ambient_vars
>            ambient_vars - list of sympy symbols [x1, ..., xn]
>            degree - truncation degree d
>     Output: polynomial in local_params representing expansion up to degree d
>     """
>     # Substitute ambient coords in terms of local params centered at P
>     # e.g. x_i = P_i + t_i if local_params = [x1-P1, ..., xn-Pn]
>     subs_map = {x: p_val + t
>                 for x, p_val, t in zip(ambient_vars,
>                                        [point[x] for x in ambient_vars],
>                                        local_params)}
>     p_shifted = p.subs(subs_map).expand()
>     q_shifted = q.subs(subs_map).expand()
>
>     # q_shifted = q(P) * (1 + r) where r is in m, expand (1+r)^{-1} to degree d
>     q0 = q_shifted.subs({t: 0 for t in local_params})  # = q(P) != 0
>     r = (q_shifted / q0 - 1).expand()  # r in m
>
>     # Geometric series: 1/(1+r) = sum_{k=0}^{d} (-r)^k mod m^{d+1}
>     inv_q_series = sum(
>         ((-r)**k).expand() for k in range(degree + 1)
>     )
>     # Truncate to total degree <= d
>     result = (p_shifted * inv_q_series / q0).expand()
>     # Drop monomials of degree > d
>     result_poly = Poly(result, *local_params)
>     truncated = sum(
>         coeff * monom
>         for coeff, monom in zip(result_poly.coeffs(), result_poly.monoms())
>         if sum(monom) <= degree
>     )
>     return truncated
> ```

> [!QUESTION] Exercise 21: Multiplicity via Dimension Counting
> *This problem implements multiplicity computation using the dimension of graded pieces.*
>
> > **Prerequisites:** [[#4.3 Multiplicity|4.3 Multiplicity]], [[#4.2 The Associated Graded Ring|4.2 The Associated Graded Ring]]
>
> Design a Python algorithm that, given a polynomial $f \in k[x,y]$ vanishing at the origin, computes $\mathrm{mult}_{(0,0)}(V(f))$ by finding the smallest degree $d$ such that $f$ has a nonzero degree-$d$ homogeneous component.

> [!TIP]- Solution to Exercise 21
> **Key insight:** The multiplicity equals the order of vanishing, which is the smallest degree appearing in the Taylor expansion of $f$ at $P$.
>
> **Sketch:**
> ```python
> from sympy import symbols, Poly, total_degree
>
> def multiplicity_at_origin(f, variables):
>     """
>     Input: f - polynomial vanishing at origin (f(0,...,0) = 0)
>            variables - list of sympy symbols
>     Output: mult_{(0,0)}(V(f)) = order of vanishing of f at origin
>     """
>     poly = Poly(f, *variables, domain='QQ')
>     max_deg = poly.total_degree()
>
>     for d in range(1, max_deg + 1):
>         # Check if f has any monomial of degree exactly d
>         homog_part = sum(
>             coeff * Poly.from_dict({monom: 1}, *variables).as_expr()
>             for coeff, monom in zip(poly.coeffs(), poly.monoms())
>             if sum(monom) == d and coeff != 0
>         )
>         if homog_part != 0:
>             return d  # smallest degree present = multiplicity
>
>     return max_deg  # fallback (f is homogeneous)
>
> # Example: cuspidal cubic y^2 - x^3
> x, y = symbols('x y')
> f = y**2 - x**3
> print(multiplicity_at_origin(f, [x, y]))  # Output: 2
>
> # Example: x^4 + y^4 - x^2*y^2
> g = x**4 + y**4 - x**2*y**2
> print(multiplicity_at_origin(g, [x, y]))  # Output: 4
> ```

> [!QUESTION] Exercise 22: DVR Valuation Algorithm
> *This problem implements the valuation $v_P$ on a smooth curve given its parametrization.*
>
> > **Prerequisites:** [[#5.3 The Valuation and Order of Vanishing|5.3 The Valuation and Order of Vanishing]], [[#5.2 Smooth Points of Curves are DVRs|5.2 Smooth Points of Curves are DVRs]]
>
> Let $C$ be a smooth affine curve with a parametrization $\gamma : \mathbb{A}^1 \to C$, $\gamma(0) = P$, such that $\mathcal{O}_{C,P} \cong k[s]_{(s)}$ where $s$ is the parameter. Design an algorithm that, given a rational function $f \in k(C)$ (presented as a ratio of polynomials in the ambient coordinates), computes $v_P(f)$.

> [!TIP]- Solution to Exercise 22
> **Key insight:** Composing $f$ with the parametrization $\gamma$ converts a rational function on $C$ to a rational function in $s$, and the valuation at $s = 0$ is the standard order of vanishing.
>
> **Sketch:**
> ```python
> from sympy import symbols, Rational, cancel, series, oo
>
> def curve_valuation(f_num, f_den, param_map, param, point_val=0):
>     """
>     Input: f_num, f_den - numerator and denominator of f in ambient coords
>            param_map - dict {ambient_var: expr_in_param}, the parametrization
>            param - sympy symbol for the curve parameter s
>            point_val - value of param at P (default 0)
>     Output: v_P(f) = order of vanishing of f at P
>     """
>     # Compose f with parametrization: compute f(gamma(s))
>     f_num_param = f_num.subs(param_map)
>     f_den_param = f_den.subs(param_map)
>
>     # Shift parameter so P corresponds to s = 0
>     t = param - point_val
>     f_num_shifted = f_num_param.subs(param, t + point_val).expand()
>     f_den_shifted = f_den_param.subs(param, t + point_val).expand()
>
>     # Find order of vanishing of numerator and denominator at t = 0
>     def order_at_zero(poly, var):
>         """Return smallest degree of var in poly."""
>         from sympy import Poly
>         p = Poly(poly, var)
>         nonzero = [i for i, c in enumerate(reversed(p.all_coeffs()))
>                    if c != 0]
>         return min(nonzero) if nonzero else oo
>
>     v_num = order_at_zero(f_num_shifted, t)
>     v_den = order_at_zero(f_den_shifted, t)
>     return v_num - v_den
>
> # Example: C = V(y - x^2), param: s -> (s, s^2), f = y/x = s
> s, x, y = symbols('s x y')
> param_map = {x: s, y: s**2}
> print(curve_valuation(y, x, param_map, s))  # v_P(y/x) = 2 - 1 = 1
> print(curve_valuation(y, 1, param_map, s))  # v_P(y) = 2
> ```

> [!QUESTION] Exercise 23: Gröbner Basis for Tangent Cone
> *This problem implements the computation of the tangent cone via the initial ideal.*
>
> > **Prerequisites:** [[#4.1 Definition via Initial Forms|4.1 Definition via Initial Forms]], [[#4.2 The Associated Graded Ring|4.2 The Associated Graded Ring]]
>
> Design a Python pseudocode algorithm that, given an ideal $I = (f_1,\ldots,f_m) \subset k[x_1,\ldots,x_n]$ with $P = 0 \in V(I)$, computes the initial ideal $\mathrm{in}(I)$ (hence the tangent cone) using a degree-reverse-lexicographic Gröbner basis.

> [!TIP]- Solution to Exercise 23
> **Key insight:** The initial ideal with respect to a graded monomial order (such as the graded reverse lexicographic order) is computed directly from a Gröbner basis: for each basis element $g$, take its leading monomial in the graded sense.
>
> **Sketch:**
> ```python
> from sympy import symbols, groebner, Poly, degree
>
> def tangent_cone_ideal(generators, variables):
>     """
>     Input: generators - list of polynomials generating I
>            variables - list of sympy symbols
>     Output: generators of the initial ideal in(I) (tangent cone ideal)
>
>     Uses: degree-reverse-lex (grlex) Gröbner basis; leading forms
>     give generators of in(I).
>     """
>     # Compute Gröbner basis with grlex order (graded reverse lex)
>     G = groebner(generators, *variables, order='grlex', domain='QQ')
>
>     # For each basis element, extract its lowest-degree homogeneous part
>     def leading_form(f, vars):
>         poly = Poly(f, *vars, domain='QQ')
>         min_deg = min(sum(m) for m in poly.monoms())
>         # Sum all monomials of degree min_deg
>         leading = sum(
>             c * Poly.from_dict({m: 1}, *vars, domain='QQ').as_expr()
>             for c, m in zip(poly.coeffs(), poly.monoms())
>             if sum(m) == min_deg
>         )
>         return leading
>
>     initial_gens = [leading_form(g, variables) for g in G]
>
>     # Remove redundant generators
>     initial_gens = list(set(initial_gens))
>     return initial_gens
>
> # Example: nodal cubic
> x, y = symbols('x y')
> I_gens = [y**2 - x**2*(x + 1)]  # y^2 - x^3 - x^2
> tc_gens = tangent_cone_ideal(I_gens, [x, y])
> # Expected: [y^2 - x^2] (the leading form y^2 - x^2)
> print(tc_gens)
> ```

---

## References

| Reference Name | Brief Summary | Link to Reference |
|---|---|---|
| [Shafarevich, *Basic Algebraic Geometry 1* (3rd ed.)](https://link.springer.com/book/10.1007/978-3-642-37956-7) | Primary source. Chapter II covers tangent spaces, singular/nonsingular points, power series, tangent cones. | https://link.springer.com/book/10.1007/978-3-642-37956-7 |
| [Hartshorne, *Algebraic Geometry*](https://link.springer.com/book/10.1007/978-1-4757-3849-0) | §I.5 on tangent spaces and singular points; Appendix B on regular local rings. | https://link.springer.com/book/10.1007/978-1-4757-3849-0 |
| [Atiyah–Macdonald, *Introduction to Commutative Algebra*](https://www.routledge.com/Introduction-To-Commutative-Algebra/Atiyah-MacDonald/p/book/9780201407518) | Background on local rings, completions, regular local rings, and Nakayama's lemma. | https://www.routledge.com/Introduction-To-Commutative-Algebra/Atiyah-MacDonald/p/book/9780201407518 |
| [Gathmann, *Algebraic Geometry* (RPTU lecture notes, 2014)](https://agag-gathmann.math.rptu.de/class/alggeom-2014/alggeom-2014-c10.pdf) | Chapter 10 on smooth varieties, Jacobian criterion, and density of smooth locus. | https://agag-gathmann.math.rptu.de/class/alggeom-2014/alggeom-2014-c10.pdf |
| [Wikipedia: Zariski Tangent Space](https://en.wikipedia.org/wiki/Zariski_tangent_space) | Definition via $\mathfrak{m}/\mathfrak{m}^2$, dual numbers interpretation, Jacobian connection. | https://en.wikipedia.org/wiki/Zariski_tangent_space |
| [Wikipedia: Tangent Cone](https://en.wikipedia.org/wiki/Tangent_cone) | Associated graded ring definition, initial ideal, multiplicity, examples. | https://en.wikipedia.org/wiki/Tangent_cone |
| [Wikipedia: Discrete Valuation Ring](https://en.wikipedia.org/wiki/Discrete_valuation_ring) | Ten equivalent characterizations of DVRs; uniformizer; smooth curves. | https://en.wikipedia.org/wiki/Discrete_valuation_ring |
| [Wikipedia: Cohen Structure Theorem](https://en.wikipedia.org/wiki/Cohen_structure_theorem) | Classification of complete Noetherian local rings; equicharacteristic case gives $k\llbracket x_1,\ldots,x_n\rrbracket$. | https://en.wikipedia.org/wiki/Cohen_structure_theorem |
| [Stacks Project: Tangent Spaces (Tag 0B28)](https://stacks.math.columbia.edu/tag/0B28) | Scheme-theoretic treatment of Zariski tangent spaces. | https://stacks.math.columbia.edu/tag/0B28 |
| [Stacks Project: Cohen Structure Theorem (Tag 0323)](https://stacks.math.columbia.edu/tag/0323) | Formal statement and proof of the Cohen structure theorem. | https://stacks.math.columbia.edu/tag/0323 |
| [Vakil, *Foundations of Algebraic Geometry* (Stanford notes)](https://math.stanford.edu/~vakil/0708-216/216class21.pdf) | Class 21: Zariski tangent space, regularity, Jacobian criterion derivation. | https://math.stanford.edu/~vakil/0708-216/216class21.pdf |
| [Milne, *Algebraic Geometry* (v5.10)](https://www.jmilne.org/math/CourseNotes/AG510.pdf) | Comprehensive notes covering smooth points, local properties, and DVRs. | https://www.jmilne.org/math/CourseNotes/AG510.pdf |
