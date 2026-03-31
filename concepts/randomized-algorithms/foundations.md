# Polynomial Identity Testing

*CS 265/CME309: Randomized Algorithms — Lecture 1*
*Gregory Valiant, updated by Mary Wootters (Stanford, Fall 2020)*

## Table of Contents

- [[#1. Introduction and Motivation|1. Introduction and Motivation]]
  - [[#1.1 The Core Problem|1.1 The Core Problem]]
  - [[#1.2 Why Naive Expansion is Expensive|1.2 Why Naive Expansion is Expensive]]
  - [[#1.3 Applications|1.3 Applications]]
- [[#2. Computational Model for Randomized Algorithms|2. Computational Model for Randomized Algorithms]]
  - [[#2.1 Binary-Tree View of Computation|2.1 Binary-Tree View of Computation]]
  - [[#2.2 Las Vegas vs Monte Carlo Algorithms|2.2 Las Vegas vs Monte Carlo Algorithms]]
  - [[#2.3 Error Amplification|2.3 Error Amplification]]
- [[#3. The Schwartz-Zippel Polynomial Identity Test|3. The Schwartz-Zippel Polynomial Identity Test]]
  - [[#3.1 Algorithm|3.1 Algorithm]]
  - [[#3.2 Schwartz-Zippel Theorem|3.2 Schwartz-Zippel Theorem]]
  - [[#3.3 Proof|3.3 Proof]]
- [[#4. Discussion and Open Problems|4. Discussion and Open Problems]]
- [[#References|References]]

---

## 1. Introduction and Motivation

### 1.1 The Core Problem

**Definition (Polynomial Identity Testing).** Given a multivariate polynomial $P(x_1, x_2, \ldots, x_n)$ presented in some *implicit* form (e.g., as a product of factors, or as the determinant of a matrix whose entries are polynomials), determine whether $P$ is the *zero polynomial*, i.e., whether all its coefficients are zero.

> [!NOTE] Identically zero vs. has a root
> "$P = 0$" (as a polynomial) is a much stronger statement than "$P(\mathbf{a}) = 0$ for some $\mathbf{a}$". The former means every evaluation is zero; the latter just means $\mathbf{a}$ is a root. Schwartz-Zippel tests the former.

A concrete example of a polynomial that *is* identically zero despite looking complicated:

$$P(x_1, x_2, x_3) = -x_3^2(x_1+x_2)(x_1-x_2) + (x_1^2 + (1+x_2)(1-x_2))x_3^2 - x_3^2 - x_3^2(x_1+x_2)(x_1-x_2).$$

Expanding this fully would reveal all terms cancel — but doing so naively is very expensive.

---

> [!EXAMPLE] Warm-up: Which is identically zero?
> Consider the two degree-4 polynomials:
> $$f(x) = (x+1)^2(x-2)^2 + x^2(3+2x-x^2) - 4(x+1)$$
> $$g(x) = (2x+1)^2(x-3)^2 - x^2(1+x+2x^2) - (5x+9)$$
> Which is identically zero? Try plugging in $x = 0$ and $x = 1$ before expanding.

> [!TIP]- Solution
> **Plug in $x = 0$:** $f(0) = (1)(4) + 0 - 4 = 0$. $g(0) = (1)(9) - 0 - 9 = 0$. Both vanish — not conclusive.
>
> **Plug in $x = 1$:** $f(1) = (4)(1) + 1(4) - 8 = 0$. $g(1) = (9)(4) - 1(4) - 14 = 36 - 4 - 14 = 18 \neq 0$.
>
> Since $g(1) \neq 0$, $g$ is **not** identically zero. By elimination (and verification), $f$ **is** identically zero. Expanding $f$ fully confirms all coefficients are 0.

---

### 1.2 Why Naive Expansion is Expensive

📐 The naive approach — expand every monomial and collect like terms — has exponential worst-case cost for multivariate polynomials.

**Claim.** A degree-$d$ polynomial in $n$ variables can have up to $\binom{n+d}{d} \approx n^d$ monomials (for fixed $d$).

More dramatically, for the product form $f(x) = \prod_{i=1}^{n}(x + a_i)$, computing the coefficient of $x^{n/2}$ requires summing over all $\binom{n}{n/2}$ subsets of size $n/2$:

$$[x^{n/2}]f = \sum_{S \subseteq [n],\, |S|=n/2} \prod_{i \in S} a_i.$$

Since $\binom{n}{n/2} \geq 2^{\Omega(n)}$, the naive algorithm takes time $2^{\Omega(n)}$ in the worst case.

---

> **Exercise 1.** *This problem establishes that naive expansion of a product of linear factors is superpolynomially expensive.*
>
> > **Prerequisites:** [[#1.2 Why Naive Expansion is Expensive]]
>
> Let $f(x) = \prod_{i=1}^{n}(x + i)$. You want to compute $f$'s coefficient on $x^{n/2}$ by first expanding the product and collecting like terms.
>
> (a) Write an exact formula for this coefficient as a sum over subsets.
>
> (b) Show the number of terms in this sum is at least $2^{n/4}$ for all large $n$.
>
> (c) Argue why this implies no algorithm that uses only ring operations and works by explicit monomial expansion can do better than $2^{\Omega(n)}$ work in the worst case for this family.

> [!TIP]- Solution to Exercise 1
> **(a)** By the Vieta / elementary symmetric polynomial formula:
> $$[x^{n/2}]f = e_{n/2}(1, 2, \ldots, n) = \sum_{S \subseteq [n],\, |S|=n/2} \prod_{i \in S} i.$$
>
> **(b)** The number of terms is $\binom{n}{n/2}$. By Stirling, $\binom{n}{n/2} \sim \frac{2^n}{\sqrt{\pi n/2}} \geq 2^{n/4}$ for all large $n$ (much stronger than asked).
>
> **(c)** Any algorithm computing this sum by iterating over contributing subsets must visit at least $\binom{n}{n/2}$ distinct subsets. Since each visit requires $\Omega(1)$ work, the total is $2^{\Omega(n)}$. This lower bound applies to the "explicit expansion" approach — other algorithms (FFT-based, DFT over rings) can do better for *specific* structures, but for the general multivariate case the problem remains hard.

---

### 1.3 Applications

💡 PIT is a useful primitive precisely because polynomials appear throughout combinatorics and number theory.

**Perfect Matching in Graphs.** Given a bipartite graph $G = (U \cup V, E)$ with $|U| = |V| = n$, construct the *Schwartz matrix* $M$ where $M_{ij} = x_{ij}$ if $(i,j) \in E$, else $0$. Then $\det(M)$ is a polynomial in the $x_{ij}$ variables, and:
$$G \text{ has a perfect matching} \iff \det(M) \not\equiv 0.$$
Testing $\det(M) \not\equiv 0$ reduces to PIT on the polynomial $\det(M)$.

**Primality Testing.** The polynomial $P(z) := (1+z)^n - 1 - z^n \pmod{n}$ satisfies:
$$P(z) \equiv 0 \pmod{n} \iff n \text{ is prime.}$$
Testing $P \equiv 0$ is again a PIT instance (mod $n$).

---

## 2. Computational Model for Randomized Algorithms

**Definition (Randomized Algorithm).** A *randomized algorithm* is an algorithm computed by a Turing machine (or RAM) with access to an infinite string of uniformly random bits, equivalently one that can execute a special `flip-coin` instruction returning an independent fair coin flip.

The key consequence: the output and runtime of a randomized algorithm are **random variables**. We typically analyze correctness and runtime for a *fixed input*, averaging only over the algorithm's internal coin flips.

### 2.1 Binary-Tree View of Computation

Every randomized algorithm with at most $k$ coin flips traces a path through a complete binary tree of depth $k$. Leaves are labeled with outputs. This immediately gives:

> [!WARNING] Probability constraint
> If the algorithm terminates after at most $k$ coin flips, then every output probability is an **integer multiple of $1/2^k$**. In particular, you cannot achieve probability exactly $1/3$ with a fixed number of flips — but you *can* simulate a $1/3$-biased coin using a *geometrically distributed* number of fair flips.

---

> **Exercise 2.** *This problem establishes that a $p$-biased coin for irrational $p$ cannot be simulated with a fixed number of fair coin flips, but can be simulated in expected finite time.*
>
> > **Prerequisites:** [[#2.1 Binary-Tree View of Computation]]
>
> (a) Prove that no algorithm using exactly $k$ fair coin flips can output `Heads` with probability $1/3$.
>
> (b) Describe an algorithm that outputs `Heads` with probability exactly $1/3$ using a *geometrically distributed* (unbounded) number of fair coin flips. What is the expected number of flips?
>
> (c) Generalize: for any rational $p = a/b$ with $b = 2^m$ for some $m$, give a fixed-$m$-flip procedure. What if $b$ is odd?

> [!TIP]- Solution to Exercise 2
> **(a)** With exactly $k$ flips, each leaf has probability $1/2^k$. Any output event is a union of leaves, so its probability is $j/2^k$ for some non-negative integer $j$. Since $1/3$ is not of this form for any $k$ (as $2^k$ is never divisible by $3$), no such algorithm exists.
>
> **(b)** Interpret two fair flips as a digit in $\{00, 01, 10, 11\}$, each with probability $1/4$. Map: $00 \mapsto$ Heads, $01 \mapsto$ Tails, $10$ or $11 \mapsto$ retry. This outputs Heads with probability $\frac{1/4}{1/4 + 1/4} = 1/2$. For $1/3$: use the von Neumann trick over blocks of flips. More directly, flip coins in sequence; on seeing `HT` output H, on `TH` output T, on `HH` or `TT` retry. This gives probability $1/2$, not $1/3$. A correct $1/3$ simulation: interpret three-flip blocks $\{H, T\}^3$ as 8 outcomes, each probability $1/8$. Assign `HHH` $\to$ category A (prob $1/8$), `HHT`, `HTH`, `THH` $\to$ category B (prob $3/8$). *Actually* the cleanest approach: generate a uniform integer in $\{1, 2, 3\}$ by rejection sampling from $\{1, \ldots, 4\}$ (reject on $4$). Each successful trial uses 2 flips; expected total flips = $2 \cdot 4/3 = 8/3$.
>
> **(c)** If $b = 2^m$, flip $m$ coins, interpret as a binary integer $r \in \{0, \ldots, 2^m - 1\}$; output Heads iff $r < a$. If $b$ is odd, use rejection sampling from the smallest $2^m \geq b$, rejecting values $\geq b$.

---

### 2.2 Las Vegas vs Monte Carlo Algorithms

The two major classes of randomized algorithms trade correctness against runtime:

| Property | Las Vegas | Monte Carlo |
|---|---|---|
| Correctness | Always correct | May err |
| Runtime | Random (finite expectation) | Deterministically bounded |
| Complexity class (poly-time) | ZPP | RP (1-sided), BPP (2-sided) |

**Monte Carlo subtypes for decision problems (yes/no):**

- **1-Sided Error:** If the true answer is "Yes," the algorithm always outputs "Yes." If the true answer is "No," it outputs "No" with probability $\geq p > 0$. *(Equivalently with Yes/No swapped.)* The error only occurs on one side.

- **2-Sided Error:** The algorithm is correct with probability $\geq 1/2 + \epsilon$ for some $\epsilon > 0$.

> [!INFO] Complexity classes
> *RP* = problems with poly-time 1-sided error Monte Carlo algorithms. *BPP* = problems with poly-time 2-sided error Monte Carlo algorithms. *ZPP* = $\text{RP} \cap \text{co-RP}$ = Las Vegas poly-time. It is known that $\text{P} \subseteq \text{ZPP} \subseteq \text{RP} \subseteq \text{BPP}$, but whether any of these containments is strict is open.

---

> **Exercise 3.** *This problem establishes the relationship between Las Vegas and Monte Carlo algorithms.*
>
> > **Prerequisites:** [[#2.2 Las Vegas vs Monte Carlo Algorithms]]
>
> (a) Show that every Las Vegas algorithm (with polynomial expected runtime) can be converted into a 1-sided error Monte Carlo algorithm with polynomial *worst-case* runtime.
>
> (b) Show that a 1-sided error Monte Carlo algorithm can be converted into a Las Vegas algorithm with polynomial *expected* runtime (assuming the problem is a decision problem and you can verify a "Yes" certificate).
>
> (c) Conclude: for decision problems with efficiently verifiable certificates, $\text{ZPP} = \text{RP}$.

> [!TIP]- Solution to Exercise 3
> **(a)** Let $A$ be Las Vegas with expected runtime $T(n)$. By Markov's inequality, $\Pr[\text{runtime} > 2T(n)] \leq 1/2$. So: run $A$ for at most $2T(n)$ steps; if it terminates, output its answer; if it times out, output "No" (or some fixed default). The resulting algorithm has worst-case runtime $2T(n) = O(T(n))$. It errs only if it times out *and* the correct answer was "Yes", which happens with probability $\leq 1/2$. This gives 1-sided error (it never wrongly says "Yes").
>
> **(b)** Run the Monte Carlo algorithm; if it says "Yes," verify the certificate. If certificate is valid, output "Yes" (correct). If it says "No" or certificate is invalid, repeat. This terminates in expected $O(1/p)$ repetitions where $p > 0$ is the probability of a correct "No" answer. Expected runtime = $O(T(n)/p)$, polynomial if $p$ is polynomially bounded below.
>
> **(c)** Part (a) shows $\text{ZPP} \subseteq \text{RP}$; part (b) gives $\text{RP} \subseteq \text{ZPP}$ for problems with efficiently verifiable certificates. For general RP problems the converse requires more care, but the containment $\text{ZPP} = \text{RP} \cap \text{co-RP}$ is standard.

---

### 2.3 Error Amplification

Running a randomized algorithm multiple times reduces the error probability exponentially.

**1-Sided Error Amplification.** Run the algorithm $t$ times independently. If it ever outputs "No," output "No"; otherwise output "Yes." Since a false "Yes" requires all $t$ runs to output "Yes," the error probability is at most $(1-p)^t$.

**2-Sided Error Amplification.** Run the algorithm $t$ times and take the majority vote.

**Lemma (Majority Vote).** *If an algorithm is correct with probability $1/2 + \epsilon$, and we run it $t$ times and take the majority, the probability of error is at most $e^{-2\epsilon^2 t}$.*

*Proof.* Let $X_i = \mathbf{1}[\text{run } i \text{ is wrong}]$, so $\Pr[X_i = 1] \leq 1/2 - \epsilon$. The majority is wrong iff $\sum_i X_i \geq t/2$. We bound:

$$\Pr\!\left[\text{majority wrong}\right] \leq \sum_{i=0}^{\lfloor t/2 \rfloor} \binom{t}{i}\!\left(\tfrac{1}{2}+\epsilon\right)^i\!\left(\tfrac{1}{2}-\epsilon\right)^{t-i}.$$

Each term is at most $\binom{t}{i}(1/4 - \epsilon^2)^{t/2}$, and summing over $i \leq t/2$ gives at most $2^t \cdot (1-4\epsilon^2)^{t/2}$. Using $1-x \leq e^{-x}$:

$$\Pr[\text{error}] \leq 2^t \cdot e^{-2\epsilon^2 t} \cdot 2^{-t} = e^{-2\epsilon^2 t}. \qquad \square$$

**Key conclusion:** $O\!\left(\frac{\log(1/\delta)}{\epsilon^2}\right)$ repetitions suffice to drive error below $\delta$.

---

> **Exercise 4.** *This problem sharpens the error amplification analysis using Chernoff bounds.*
>
> > **Prerequisites:** [[#2.3 Error Amplification]]
>
> Let $X = \sum_{i=1}^t X_i$ where each $X_i \in \{0,1\}$ independently with $\Pr[X_i = 1] = 1/2 - \epsilon$.
>
> (a) Compute $\mathbb{E}[X]$ and show that majority-wrong requires $X \geq t/2 = \mathbb{E}[X] + \epsilon t$.
>
> (b) Apply the Chernoff bound $\Pr[X \geq \mu + \delta] \leq e^{-2\delta^2/t}$ (Hoeffding's inequality for bounded variables) to re-derive the $e^{-2\epsilon^2 t}$ bound cleanly.
>
> (c) How many repetitions are needed to achieve error probability $\leq 10^{-6}$ for $\epsilon = 0.01$?

> [!TIP]- Solution to Exercise 4
> **(a)** $\mathbb{E}[X] = t(1/2 - \epsilon)$. Majority is wrong iff $X \geq t/2$, i.e., $X \geq \mathbb{E}[X] + \epsilon t$.
>
> **(b)** By Hoeffding with $\delta = \epsilon t$:
> $$\Pr[X \geq \mathbb{E}[X] + \epsilon t] \leq e^{-2(\epsilon t)^2/t} = e^{-2\epsilon^2 t}.$$
>
> **(c)** Set $e^{-2\epsilon^2 t} \leq 10^{-6}$, so $t \geq \frac{6\ln 10}{2\epsilon^2} = \frac{6 \times 2.303}{2 \times 0.0001} \approx 69{,}100$ repetitions. In practice the constant $\epsilon = 0.01$ (barely above $1/2$ success probability) is very small; a more typical $\epsilon = 0.1$ requires only $\approx 691$ repetitions.

---

## 3. The Schwartz-Zippel Polynomial Identity Test

### 3.1 Algorithm

The algorithm is almost absurdly simple:

**Algorithm (Schwartz-Zippel PIT).**
Given $P(x_1, \ldots, x_n)$ of total degree $\leq d$:
1. Fix a finite set $S \subseteq \mathbb{F}$ (or $\mathbb{Z}$).
2. Sample $r_1, \ldots, r_n \overset{\text{i.i.d.}}{\sim} \mathrm{Uniform}(S)$.
3. Evaluate $P(r_1, \ldots, r_n)$.
   - If $P(r_1, \ldots, r_n) = 0$: output "$P = 0$."
   - Else: output "$P \neq 0$."

```python
import random

def schwartz_zippel(P_eval, n, d, S):
    """
    P_eval: callable (r_1, ..., r_n) -> field element
    n: number of variables
    d: degree bound
    S: finite set to sample from (list)
    Returns: "zero" or "nonzero"
    """
    r = [random.choice(S) for _ in range(n)]
    val = P_eval(*r)
    return "zero" if val == 0 else "nonzero"
```

> [!NOTE] One-sided error
> The algorithm has **1-sided error**: if $P = 0$, it always correctly outputs "$P = 0$" (since $P(\mathbf{r}) = 0$ for all $\mathbf{r}$). It can only err when $P \neq 0$ — outputting "$P = 0$" when $P(\mathbf{r}) = 0$ by coincidence. This places PIT in **RP**.

---

### 3.2 Schwartz-Zippel Theorem

**Theorem (Schwartz-Zippel, 1979; DeMillo-Lipton, 1978).** *Let $P(x_1, \ldots, x_n)$ be a nonzero polynomial of total degree $d$ over an integral domain $\mathbb{F}$. Let $S \subseteq \mathbb{F}$ be a finite set and $r_1, \ldots, r_n \overset{\text{i.i.d.}}{\sim} \mathrm{Uniform}(S)$. Then:*

$$\boxed{\Pr[P(r_1, \ldots, r_n) = 0] \leq \frac{d}{|S|}.}$$

*Consequently, the Schwartz-Zippel algorithm is correct with probability at least $1 - d/|S|$ on nonzero inputs. Taking $|S| = 2d$ gives error probability $\leq 1/2$.*

---

> **Exercise 5.** *This exercise develops intuition for the bound before the formal proof.*
>
> > **Prerequisites:** [[#3.2 Schwartz-Zippel Theorem]]
>
> (a) Verify the bound in the univariate case ($n=1$) directly using the fundamental theorem of algebra.
>
> (b) Show the bound is **tight**: for any $d$ and $|S| = d+1$, construct a polynomial $P(x_1)$ of degree $d$ such that $\Pr[P(r_1) = 0] = d/(d+1) = d/|S|$.
>
> (c) The bound $d/|S|$ depends only on the *degree*, not on $n$. Does this seem surprising? Explain intuitively why the number of variables doesn't appear.

> [!TIP]- Solution to Exercise 5
> **(a)** A nonzero degree-$d$ univariate polynomial has at most $d$ roots (fundamental theorem of algebra). Exactly $d$ of the $|S|$ elements can be roots, so $\Pr[P(r_1)=0] \leq d/|S|$.
>
> **(b)** Let $S = \{a_1, \ldots, a_{d+1}\}$ and $P(x_1) = \prod_{i=1}^{d}(x_1 - a_i)$. This is degree $d$, nonzero, and has exactly $d$ roots in $S$, so $\Pr[P(r_1) = 0] = d/(d+1)$.
>
> **(c)** *Surprisingly*, more variables don't worsen the bound. The intuition: adding variables only "spreads out" the zero set across a higher-dimensional space — but the *fraction* of the evaluation space that is zero is still controlled by the degree, because at each induction step we fix all but one variable and reduce to the univariate case. The polynomial can't "use" extra variables to create more zeros per random sample without also increasing its degree.

---

### 3.3 Proof

The proof is by **induction on $n$**, the number of variables, reducing each step to the univariate case.

**Fact (used in base case).** A nonzero degree-$d$ univariate polynomial has at most $d$ roots.

---

**Base case ($n = 1$).** $P(x_1)$ is a nonzero univariate polynomial of degree $d$. At most $d$ elements of $S$ are roots, so:
$$\Pr[P(r_1) = 0] \leq \frac{d}{|S|}.$$

---

**Inductive step.** Assume the theorem holds for all nonzero polynomials in $\leq n-1$ variables. Let $P(x_1, \ldots, x_n)$ be nonzero of degree $d$.

Since $P \neq 0$, at least one variable actually appears; WLOG let $x_1$ appear. Write $P$ by grouping terms according to the power of $x_1$. Let $k$ be the highest power of $x_1$ that appears:

$$P(x_1, \ldots, x_n) = x_1^k \cdot Q(x_2, \ldots, x_n) + T(x_1, \ldots, x_n),$$

where:
- $Q(x_2, \ldots, x_n)$ collects all coefficients of $x_1^k$ — a nonzero polynomial in $n-1$ variables of degree $\leq d - k$.
- $T(x_1, \ldots, x_n)$ contains all terms with $x_1$-degree $< k$.

Now condition on the event "$Q(r_2, \ldots, r_n) = 0$":

$$\Pr[P(\mathbf{r}) = 0] = \Pr[P(\mathbf{r}) = 0 \mid Q(\mathbf{r}_{-1}) = 0]\,\Pr[Q(\mathbf{r}_{-1}) = 0] + \Pr[P(\mathbf{r}) = 0 \mid Q(\mathbf{r}_{-1}) \neq 0]\,\Pr[Q(\mathbf{r}_{-1}) \neq 0].$$

🔑 Bounding each piece:

1. **$\Pr[Q(r_2,\ldots,r_n) = 0]$:** By the inductive hypothesis applied to the nonzero polynomial $Q$ of degree $\leq d-k$ in $n-1$ variables:
$$\Pr[Q(r_2, \ldots, r_n) = 0] \leq \frac{d-k}{|S|}.$$

2. **$\Pr[P(r_1, \ldots, r_n) = 0 \mid Q(r_2,\ldots,r_n) \neq 0]$:** Condition on fixed values $r_2, \ldots, r_n$ with $Q(r_2, \ldots, r_n) \neq 0$. Then $P(x_1, r_2, \ldots, r_n)$ is a nonzero univariate polynomial of degree exactly $k$ in $x_1$ (the leading coefficient $Q(r_2, \ldots, r_n)$ is nonzero). By the base case:
$$\Pr[P(r_1, r_2, \ldots, r_n) = 0 \mid Q(r_2,\ldots,r_n) \neq 0] \leq \frac{k}{|S|}.$$

Combining via the union bound (or total probability, noting the conditional on the second event and using $\Pr[A] \leq 1$):

$$\Pr[P(\mathbf{r}) = 0] \leq \frac{d-k}{|S|} \cdot 1 + \frac{k}{|S|} \cdot 1 = \frac{d}{|S|}. \qquad \square$$

> [!NOTE] Why the union bound suffices
> More carefully: $\Pr[P(\mathbf{r}) = 0] \leq \Pr[Q(\mathbf{r}_{-1}) = 0] + \Pr[P(\mathbf{r}) = 0 \mid Q(\mathbf{r}_{-1}) \neq 0]$. The first term is $\leq (d-k)/|S|$; the second is $\leq k/|S|$.

---

> **Exercise 6.** *This problem reconstructs the key inductive step with explicit algebraic detail.*
>
> > **Prerequisites:** [[#3.3 Proof]]
>
> Let $P(x_1, x_2) = x_1^2 x_2 + x_1 x_2^2 - x_1 x_2$ and $S = \{0, 1, 2, 3, 4\}$.
>
> (a) Identify $k$, $Q(x_2)$, and $T(x_1, x_2)$ in the decomposition $P = x_1^k Q + T$.
>
> (b) Compute the upper bound on $\Pr[P(r_1, r_2) = 0]$ given by Schwartz-Zippel.
>
> (c) Numerically verify the bound by computing $|\{(r_1, r_2) \in S^2 : P(r_1, r_2) = 0\}| / |S|^2$.

> [!TIP]- Solution to Exercise 6
> **(a)** The highest power of $x_1$ is $k = 2$, appearing in $x_1^2 x_2$. So $Q(x_2) = x_2$ and $T(x_1, x_2) = x_1 x_2^2 - x_1 x_2 = x_1 x_2(x_2 - 1)$.
>
> **(b)** $P$ has degree $d = 3$ (total degree of $x_1^2 x_2$ or $x_1 x_2^2$ is $3$). $|S| = 5$. Schwartz-Zippel gives $\Pr[P(r_1,r_2) = 0] \leq 3/5 = 0.6$.
>
> **(c)** $P(r_1, r_2) = r_1 r_2(r_1 + r_2 - 1)$. This is zero iff $r_1 = 0$, $r_2 = 0$, or $r_1 + r_2 = 1$.
> - $r_1 = 0$: 5 pairs $(0, 0), (0,1), \ldots, (0,4)$.
> - $r_2 = 0$ (and $r_1 \neq 0$): 4 additional pairs.
> - $r_1 + r_2 = 1$ (and $r_1, r_2 \neq 0$): only $(1,0)$ already counted; no new ones in $S$ since we need $r_1, r_2 \in \{1,\ldots,4\}$ with $r_1 + r_2 = 1$, impossible.
>
> Total zeros: $5 + 4 = 9$ out of $25$. Fraction $= 9/25 = 0.36 \leq 0.6$. Bound holds. ✓

---

> **Exercise 7.** *This problem extends the proof technique to a field of characteristic $p$.*
>
> > **Prerequisites:** [[#3.3 Proof]]
>
> The Schwartz-Zippel proof uses the fact that a nonzero degree-$d$ univariate polynomial has at most $d$ roots. This holds over any *integral domain* (a commutative ring with no zero divisors).
>
> (a) Is $\mathbb{Z}/p\mathbb{Z}$ for prime $p$ an integral domain? What about $\mathbb{Z}/6\mathbb{Z}$?
>
> (b) Does Schwartz-Zippel apply to polynomials over $\mathbb{Z}/6\mathbb{Z}$? Exhibit a nonzero polynomial $P(x) \in (\mathbb{Z}/6\mathbb{Z})[x]$ of degree 1 with more than 1 root.
>
> (c) Conclude: over what algebraic structures does Schwartz-Zippel hold, and what is the appropriate version of the bound when working modulo a prime $p$ (and $S = \mathbb{Z}/p\mathbb{Z}$)?

> [!TIP]- Solution to Exercise 7
> **(a)** $\mathbb{Z}/p\mathbb{Z}$ is a field (hence an integral domain) for prime $p$: if $ab \equiv 0 \pmod{p}$ then $p \mid ab$, so $p \mid a$ or $p \mid b$, i.e., $a = 0$ or $b = 0$ in $\mathbb{Z}/p\mathbb{Z}$.
> $\mathbb{Z}/6\mathbb{Z}$ is **not** an integral domain: $2 \cdot 3 = 6 \equiv 0$ but $2 \neq 0$ and $3 \neq 0$.
>
> **(b)** $P(x) = 2x \in (\mathbb{Z}/6\mathbb{Z})[x]$ is degree 1 and nonzero (the coefficient 2 is nonzero mod 6). Its roots: $2x \equiv 0 \pmod{6}$ iff $6 \mid 2x$ iff $3 \mid x$, so $x \in \{0, 3\}$ — two roots for a degree-1 polynomial. Schwartz-Zippel fails.
>
> **(c)** Schwartz-Zippel holds over any integral domain (in particular any field). Over $\mathbb{Z}/p\mathbb{Z}$ with $S = \mathbb{Z}/p\mathbb{Z}$, the bound gives $\Pr[P(\mathbf{r}) = 0] \leq d/p$. For the primality application, $S = \mathbb{Z}/p\mathbb{Z}$ and the polynomial has degree $\leq n$, giving error $\leq n/p$.

---

> **Exercise 8.** *This problem applies Schwartz-Zippel to perfect matching detection.*
>
> > **Prerequisites:** [[#1.3 Applications]], [[#3.2 Schwartz-Zippel Theorem]]
>
> Let $G = (U \cup V, E)$ be a bipartite graph with $|U| = |V| = n$. Define the *Schwartz matrix* $M$ by $M_{ij} = x_{ij}$ if $(i,j) \in E$, else $M_{ij} = 0$. Let $P = \det(M)$.
>
> (a) Show $P$ has total degree $\leq n$.
>
> (b) Show $P \not\equiv 0$ iff $G$ has a perfect matching. (*Hint*: expand the determinant as a sum over permutations.)
>
> (c) Give a randomized algorithm to test for a perfect matching with error probability $\leq 1/2$, and state its runtime. What is the error probability after $k$ independent repetitions?

> [!TIP]- Solution to Exercise 8
> **(a)** By Leibniz formula, $\det(M) = \sum_{\sigma \in S_n} \text{sgn}(\sigma)\prod_{i=1}^n M_{i,\sigma(i)}$. Each term $\prod_i M_{i,\sigma(i)}$ is a product of at most $n$ distinct variables, so has degree $\leq n$. Hence $\deg(P) \leq n$.
>
> **(b)** If $G$ has a perfect matching $\sigma^*$, then the term $\text{sgn}(\sigma^*)\prod_i x_{i,\sigma^*(i)}$ is a nonzero monomial in $P$ (all variables distinct, coefficient $\pm 1$). No other term can cancel it (they use different sets of variables). So $P \not\equiv 0$. Conversely, if $G$ has no perfect matching, every permutation $\sigma$ has at least one edge $(i, \sigma(i)) \notin E$, so $M_{i,\sigma(i)} = 0$ and every term in the Leibniz expansion is zero; thus $P \equiv 0$.
>
> **(c)** Choose $S = \{1, \ldots, 2n\}$ ($|S| = 2n$). Sample $r_{ij} \sim \text{Uniform}(S)$ for each $(i,j) \in E$, evaluate $\det(M(r))$ in $O(n^3)$ time (Gaussian elimination). By Schwartz-Zippel with $d = n$, $|S| = 2n$: error probability $\leq n/(2n) = 1/2$. After $k$ repetitions, error $\leq (1/2)^k$.

---

## 4. Discussion and Open Problems

💡 The Schwartz-Zippel algorithm is essentially the simplest possible randomized algorithm: pick a random point, evaluate, check. Yet no efficient deterministic algorithm is known.

**State of deterministic PIT:** No sub-exponential time deterministic algorithm for general PIT is known. For special *structured* polynomial families (e.g., noncommutative polynomials computed by bounded-width algebraic branching programs), deterministic algorithms exist — see Raz-Shpilka [2].

**The Kabanets-Impagliazzo connection.** A breakthrough result by Kabanets and Impagliazzo [1] shows:

> [!WARNING] Derandomization is hard
> **If** a deterministic polynomial-time PIT algorithm exists, **then** either $\text{NEXP} \not\subset \text{poly-circuits}$ or the permanent does not have polynomial-size arithmetic circuits. Both conclusions are major open problems in complexity theory.

In other words: *derandomizing PIT would imply* that we can prove hard circuit lower bounds — something we currently cannot do. This suggests that either PIT is genuinely hard to derandomize, or proving it isn't requires fundamentally new techniques.

**Connection to $P$ vs $BPP$.** More broadly, the question of whether randomness is essential for efficiency — whether $P = BPP$ — remains open. PIT is one of the flagship examples where:
1. A clean, efficient randomized algorithm exists.
2. We believe an efficient deterministic algorithm exists.
3. We cannot prove (2) unconditionally.

> [!QUESTION] Open problem
> Does PIT have a deterministic polynomial-time algorithm? The current best result is a *quasi-polynomial* time algorithm for restricted circuit classes (due to a series of works culminating in recent breakthroughs around 2020-2023). The general case remains open.

---

> **Exercise 9.** *This problem explores why derandomization of PIT is connected to lower bounds.*
>
> > **Prerequisites:** [[#4. Discussion and Open Problems]]
>
> A polynomial $P$ is *hard* if the smallest arithmetic circuit computing it has size $\geq 2^{n^\epsilon}$ for some $\epsilon > 0$.
>
> (a) Explain informally why an efficient PIT algorithm would let us "find" hard polynomials.
>
> (b) The Kabanets-Impagliazzo result says: a poly-time deterministic PIT algorithm implies $\text{NEXP} \not\subset P/\text{poly}$. This implication goes via the "hardness vs randomness" paradigm. State (without proof) the key idea: what does a hard function give you that helps derandomize?
>
> (c) Why is the existence of a hard polynomial *not* obvious, even though most polynomials are hard by a counting argument?

> [!TIP]- Solution to Exercise 9
> **(a)** A deterministic PIT algorithm can test whether a given arithmetic circuit computes the zero polynomial. By searching over circuits of increasing size, one could find the minimal circuit for any polynomial — if no small circuit works, we've proven hardness. So a fast PIT algorithm gives a fast method to certify circuit lower bounds.
>
> **(b)** The hardness-vs-randomness paradigm: a function $f$ that is hard for small circuits can be used to build a *pseudorandom generator* (PRG) — a deterministic function that stretches a short seed into a long string that "looks random" to any efficient algorithm. If a PRG exists that fools all poly-time algorithms, then $BPP = P$, because we can replace random coins with the PRG output and enumerate all seeds deterministically. The key insight: a PRG requires a hard function to "mix" the seed, and circuit lower bounds provide that hardness.
>
> **(c)** A counting argument shows that *almost all* polynomials of degree $d$ in $n$ variables require large circuits — there are vastly more polynomials than circuits. But "most" doesn't give an *explicit* construction: we need to write down a *specific* polynomial and prove it's hard. Proving lower bounds for explicit functions is the central challenge of complexity theory, and we currently have only weak (e.g., $\Omega(n\log n)$) lower bounds for explicit functions in the arithmetic circuit model.

---

## References

| Reference | Brief Summary | Link |
|---|---|---|
| CS 265 Lecture 1 (Valiant/Wootters) | Primary source: computational model, Schwartz-Zippel theorem and proof | [PDF](https://web.stanford.edu/class/archive/cs/cs265/cs265.1212/Lectures/Lecture1/l1.pdf) |
| CS 265 Class 1 Solutions | In-class exercises and solutions on PIT | [PDF](https://web.stanford.edu/class/archive/cs/cs265/cs265.1212/Lectures/Lecture1/class1-sols.pdf) |
| [Kabanets & Impagliazzo (2004)](https://link.springer.com/article/10.1007/s00037-004-0182-6) | Derandomizing PIT implies circuit lower bounds (NEXP ⊄ poly-circuits) | Computational Complexity 13(1-2):1–46 |
| [Raz & Shpilka (2005)](https://link.springer.com/article/10.1007/s00037-005-0188-8) | Deterministic PIT for noncommutative algebraic branching programs | Computational Complexity 14(1):1–19 |
