# Algebraic and Arithmetic Geometry Curriculum
*82 weeks · ~6.5 hrs/wk · ~530 hrs total*
*Profile: undergraduate algebra background, goal of Berkeley/Harvard PhD qualifying exam level*
*Focus: algebraic constructions and their geometric avatars — every definition paired with a geometric picture*

---

## Overview

| Phase | Weeks | Theme | File |
|-------|-------|-------|------|
| I | 1–20 | Classical Varieties (Shafarevich) | [[curricula/algebraic-geometry/phase-1-classical-varieties\|Phase I]] |
| II | 21–42 | Scheme Theory (Hartshorne I-II + Mumford) | [[curricula/algebraic-geometry/phase-2-scheme-theory\|Phase II]] |
| III | 43–64 | Cohomology and Curves (Hartshorne III-IV) | [[curricula/algebraic-geometry/phase-3-cohomology-curves\|Phase III]] |
| IV | 65–82 | Arithmetic Geometry (Silverman + Milne) | [[curricula/algebraic-geometry/phase-4-arithmetic-geometry\|Phase IV]] |

**Commutative algebra** is not a separate phase. Study the relevant Atiyah-Macdonald (A&M) sections at the point of first need — each week flags exactly which chapters are prerequisites. Eisenbud's *Commutative Algebra with a View Toward Algebraic Geometry* is the secondary CA reference: its geometric commentary on standard theorems is invaluable.

**Qualifying exam problem sets** used as benchmarks:
- Ritvik Ramkumar (Berkeley, 2017): scheme theory + commutative algebra + algebraic topology
- Will Fisher (Berkeley, 2024): scheme theory + category theory + algebraic topology
- Harvard AG qualifying problem collection: curves, divisors, cohomology, elliptic curves

---

## Dependency Map

```mermaid
flowchart TD
    subgraph P1["Phase I: Classical Varieties (Wks 1-20)"]
        affine["Affine Varieties<br/>Nullstellensatz"]
        projV["Projective Varieties<br/>Homogeneous Coords"]
        maps["Morphisms<br/>Rational Maps"]
        local["Local Geometry<br/>Smoothness, DVRs"]
        curvesCl["Classical Curves<br/>Divisors, Genus, RR"]
    end

    subgraph P2["Phase II: Scheme Theory (Wks 21-42)"]
        spec["Spec A<br/>Structure Sheaf"]
        shvs["Sheaves<br/>Locally Ringed Spaces"]
        schm["Schemes<br/>Morphisms"]
        proj["Proj<br/>Fiber Products"]
        coh["Coherent Sheaves<br/>Picard Group"]
        diff["Differentials<br/>Blowing Up"]
    end

    subgraph P3["Phase III: Cohomology and Curves (Wks 43-64)"]
        derived["Derived Functors<br/>Sheaf Cohomology"]
        cech["Cech Cohomology<br/>H^i of P^n"]
        sd["Serre Duality<br/>Vanishing Theorems"]
        rr["Riemann-Roch<br/>Linear Systems"]
        ellG["Elliptic Curves<br/>j-invariant, Group Law"]
    end

    subgraph P4["Phase IV: Arithmetic Geometry (Wks 65-82)"]
        ellA["EC over Fields<br/>Isogenies, Torsion"]
        mw["Mordell-Weil<br/>over Q"]
        fq["EC over Fq<br/>Frobenius, Hasse"]
        weil["Weil Conjectures<br/>Zeta Functions"]
    end

    affine --> projV
    affine --> maps
    projV --> maps
    maps --> local
    local --> curvesCl

    affine --> spec
    projV --> proj
    local --> shvs
    curvesCl --> coh

    spec --> shvs
    shvs --> schm
    schm --> proj
    proj --> coh
    coh --> diff

    shvs --> derived
    coh --> derived
    coh --> cech
    derived --> cech
    cech --> sd
    sd --> rr
    rr --> ellG
    curvesCl --> rr

    ellG --> ellA
    ellA --> mw
    ellA --> fq
    fq --> weil
    rr --> weil
```

---

## References

| Text | Role |
|---|---|
| Shafarevich, *Basic Algebraic Geometry* Vol 1 | Phase I primary |
| Reid, *Undergraduate Algebraic Geometry* | Phase I companion |
| Gathmann, *Algebraic Geometry* lecture notes (free) | Phase I–II problem supplement |
| Atiyah-Macdonald, *Introduction to Commutative Algebra* | CA reference (woven throughout) |
| Eisenbud, *Commutative Algebra with a View Toward AG* | CA supplement (geometric commentary) |
| Hartshorne, *Algebraic Geometry* | Phase II–III primary |
| Mumford, *The Red Book of Varieties and Schemes* | Phase II geometric companion |
| Vakil, *The Rising Sea* (free) | Phase II reference and exercises |
| Serre, *Faisceaux Algébriques Cohérents* (FAC) | Phase III historical source |
| Miranda, *Algebraic Curves and Riemann Surfaces* | Phase III curves supplement |
| Weibel, *Introduction to Homological Algebra* | Phase III derived functor reference |
| Silverman, *The Arithmetic of Elliptic Curves* | Phase IV primary |
| Ireland-Rosen, *A Classical Introduction to Modern Number Theory* | Phase IV finite fields |
| Milne, *Lectures on Étale Cohomology* (free) | Phase IV Weil conjectures |
| Fulton, *Algebraic Curves* (free) | Optional: deeper classical curves reference |

## Qualifying Exam Sources

| Source | Notes |
|---|---|
| [Ritvik Ramkumar, Berkeley 2017](https://math.berkeley.edu/~ritvik/Qualifying_Exam_Syllabus_and_Transcript.pdf) | Major: AG + Comm. Algebra; Minor: Alg. Topology |
| [Will Fisher, Berkeley 2024](https://math.berkeley.edu/~willfisher/papers/Qual_Transcript.pdf) | Major: AG + Category Theory; Minor: Alg. Topology |
| [Harvard AG Qualifying Problems](https://www.math.harvard.edu/media/alggeom.pdf) | Broad problem set; use as weekly diagnostic |
