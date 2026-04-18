# Design: Differential Forms, Riemann-Roch, and Hurwitz

**Date:** 2026-04-18
**Topic slug:** `shafarevich-ch3-differentials-riemann-roch`
**Category:** `concepts/algebraic-geometry`
**Multi-note:** no

## Scope

This note covers Shafarevich §III.3–4 plus the Hurwitz formula (§III.4), corresponding to Weeks 13–15 of the Phase I curriculum. It is the direct sequel to `shafarevich-ch3-divisors-picard.md`: differential forms on a smooth projective curve define a canonical divisor class $K_X \in \mathrm{Pic}(X)$, and Riemann-Roch computes $h^0(\mathcal{O}(D))$ in terms of $\deg D$, $g$, and $h^0(\mathcal{O}(K-D))$.

The note develops three interlocking threads. First, Kähler differentials $\Omega_{X/k}$: defined via the universal derivation, concretely computed on affine curves, and shown to be a line bundle (the *canonical bundle* $\omega_X$) on a smooth curve. Second, the Riemann-Roch theorem: statement, proof sketch via Serre duality, and systematic worked applications — genus-0 curves are $\mathbb{P}^1$, elliptic curves in Weierstrass form from RR, the genus formula for smooth plane curves. Third, the Hurwitz formula: ramification of a finite separable morphism $f: X \to Y$ of smooth projective curves, the ramification divisor $R$, and the formula $2g(X)-2 = n(2g(Y)-2) + \deg R$; applications to hyperelliptic curves and the genus formula for plane curves.

## Files to Create

| File | Purpose |
|------|---------|
| `concepts/algebraic-geometry/shafarevich-ch3-differentials-riemann-roch.md` | Single note on Kähler differentials, Riemann-Roch, and Hurwitz |

## Note Structure

1. **Introduction** — the canonical bundle bridges divisor theory and global geometry; RR is the master tool for computing $h^0$
2. **Kähler Differentials**
   - Module of Kähler differentials $\Omega_{A/k}$: universal derivation $d: A \to \Omega_{A/k}$, presentation via generators $da$ and relations $d(ab) = a\,db + b\,da$, $dk = 0$
   - For $A = k[x_1,\ldots,x_n]/(f_1,\ldots,f_m)$: $\Omega_{A/k} = \bigoplus_i A\,dx_i / \langle \sum_j \partial f_i/\partial x_j \, dx_j \rangle$
   - Sheafification: $\Omega_{X/k}$; for a smooth curve this is a line bundle (rank-1 locally free)
   - Local description: on an affine chart with local parameter $t$, $\Omega_{X/k}|_U \cong \mathcal{O}_U \cdot dt$
   - The first Chern class: $c_1(\Omega_{X/k}) = K_X$ (the canonical class)
3. **The Canonical Divisor**
   - For $\omega \in H^0(X, \Omega_{X/k}) \setminus \{0\}$: $\mathrm{div}(\omega) = \sum_P v_P(\omega)[P]$ using the DVR at each point
   - Any two nonzero forms differ by a rational function, so $\mathrm{div}(\omega) \sim \mathrm{div}(\omega')$: well-defined class $K_X \in \mathrm{Pic}(X)$
   - On $\mathbb{P}^1$: $\mathrm{div}(dx) = -2[\infty]$, so $\deg K_{\mathbb{P}^1} = -2$; generally $\deg K_X = 2g-2$
   - Geometric genus: $g = h^0(\omega_X) = \dim H^0(X, \Omega_{X/k})$
4. **The Riemann-Roch Theorem**
   - Setup: $\ell(D) = h^0(\mathcal{O}(D))$; Serre duality: $H^1(X, \mathcal{O}(D)) \cong H^0(X, \mathcal{O}(K-D))^\vee$, so $h^1(\mathcal{O}(D)) = \ell(K-D)$
   - **Theorem (Riemann-Roch):** $\ell(D) - \ell(K-D) = \deg D + 1 - g$
   - Proof sketch: use the long exact sequence for $0 \to \mathcal{O}(D-P) \to \mathcal{O}(D) \to k_P \to 0$, induction on $\deg D$, and Serre duality at the base case
   - **Corollaries:**
     - $\ell(0) = 1$, $\ell(K) = g$, $\deg K = 2g-2$
     - For $\deg D > 2g-2$: $\ell(K-D) = 0$, so $\ell(D) = \deg D + 1 - g$
     - For $\deg D < 0$: $\ell(D) = 0$
5. **Applications of Riemann-Roch**
   - **Genus 0:** $g=0$ implies $\ell(-K) = \deg(-K)+1-0 = 3$, and $-K$ very ample implies $X \cong \mathbb{P}^1$
   - **Elliptic curves (genus 1):** $\ell(nP)$ for $n=0,1,2,3$; the functions $x$ (double pole) and $y$ (triple pole); the relation $y^2 = x^3+ax+b$ from $\ell(6P)=6$ and 7 monomials
   - **Genus formula for smooth plane curves:** $C \subset \mathbb{P}^2$ of degree $d$; $g = \binom{d-1}{2}$; proof via adjunction formula $K_C = (K_{\mathbb{P}^2} + C)|_C = \mathcal{O}(d-3)|_C$
6. **Hurwitz's Formula**
   - Setup: $f: X \to Y$ finite separable morphism of smooth projective curves, degree $n$
   - Ramification index $e_P$ at $P \in X$: order of vanishing of $f^* t_{f(P)}$ at $P$, where $t_{f(P)}$ is a local parameter at $f(P)$
   - Unramified at $P$ iff $e_P = 1$; branch points: images of ramified points
   - Ramification divisor $R = \sum_{P \in X} (e_P - 1)[P]$
   - **Theorem (Hurwitz):** $2g(X) - 2 = n(2g(Y)-2) + \deg R$
   - Proof: pull back a nonzero form $\omega$ on $Y$; compute $\mathrm{div}(f^*\omega)$
   - **Applications:**
     - Degree-$n$ map $X \to \mathbb{P}^1$: $\deg R = 2g(X) + 2n - 2$ branch points (over $\mathbb{C}$)
     - Hyperelliptic curves: $f: C \to \mathbb{P}^1$ degree 2, $\deg R = 2g+2$
     - Genus formula for plane curves via projection: project a smooth degree-$d$ curve from a general point to $\mathbb{P}^1$, use Hurwitz to recover $g = \binom{d-1}{2}$
     - Purely inseparable maps in char $p$: $g(X) = g(Y)$
7. **Exercises** (inline: ~16–18 mathematical development + 5–7 algorithmic)

## References

- Shafarevich, *Basic Algebraic Geometry* Vol 1, §III.3–4 (primary)
- Fulton, *Algebraic Curves*, Ch 7–8 (Hurwitz, differentials, RR)
- Hartshorne, *Algebraic Geometry*, §II.8 (differentials), §IV.1 (RR for curves)
- Reid, *Undergraduate Algebraic Geometry*, Ch 9 §9.4–9.7
- Silverman, *Arithmetic of Elliptic Curves*, Ch II §5 (differentials on elliptic curves)
- Miranda, *Algebraic Curves and Riemann Surfaces* (Ch V, very readable RR proof)
