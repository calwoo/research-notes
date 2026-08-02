# Configuration Space and Degrees of Freedom

## Table of Contents

- [[#1. Configuration and Configuration Space|1. Configuration and Configuration Space]]
  - [[#1.1 Examples of Configuration Spaces|1.1 Examples of Configuration Spaces]]
- [[#2. Degrees of Freedom|2. Degrees of Freedom]]
  - [[#2.1 Degrees of Freedom of a Free Rigid Body|2.1 Degrees of Freedom of a Free Rigid Body]]
  - [[#2.2 Degrees of Freedom of a System of Rigid Bodies Before Constraints|2.2 Degrees of Freedom of a System of Rigid Bodies Before Constraints]]
- [[#3. Grubler's Formula|3. Grubler's Formula]]
  - [[#3.1 Derivation|3.1 Derivation]]
  - [[#3.2 Worked Example: The Planar Four-Bar Linkage|3.2 Worked Example: The Planar Four-Bar Linkage]]
  - [[#3.3 Worked Example: The SO-101 Serial Chain|3.3 Worked Example: The SO-101 Serial Chain]]
- [[#4. Holonomic and Nonholonomic Constraints|4. Holonomic and Nonholonomic Constraints]]
  - [[#4.1 Holonomic Constraints|4.1 Holonomic Constraints]]
  - [[#4.2 Nonholonomic Constraints and the Rolling Wheel|4.2 Nonholonomic Constraints and the Rolling Wheel]]
  - [[#4.3 Effect on the Configuration Space|4.3 Effect on the Configuration Space]]
- [[#5. Task Space versus Configuration Space|5. Task Space versus Configuration Space]]
  - [[#5.1 Redundancy and Underactuation|5.1 Redundancy and Underactuation]]
- [[#References|References]]

> [!NOTE] Scope and place in the cluster
> This note covers Chapters 1–2 of Lynch & Park's *Modern Robotics* — the Module 0 orientation reading of the [[curricula/robotics/curriculum|SO-101 Robotics Curriculum]] — and is the first entry in the [[concepts/robotics/overview|Robotics concept cluster]]. It builds the vocabulary (configuration, C-space, degrees of freedom, constraints) that [[concepts/robotics/overview|later notes in this cluster]] (rigid-body motions, forward/inverse kinematics, dynamics) all assume.

## 1. Configuration and Configuration Space 🧭

**Definition (Configuration).** Let a mechanical system consist of one or more rigid bodies, possibly connected to one another by joints. A *configuration* of the system is a specification of the position of every point comprising the system, relative to a fixed reference frame.

**Definition (Degrees of Freedom, Configuration Space).** The *degrees of freedom* (dof) of the system, denoted $n$, is the minimum number of real-valued coordinates $q = (q_1, \dots, q_n)$ required to uniquely determine a configuration. The set $\mathcal{C}$ of all valid configurations is the *configuration space*, or *C-space*, of the system.

*This is stronger than it looks.* Formally, $\mathcal{C}$ is an $n$-dimensional smooth *manifold*: a topological space that is locally diffeomorphic to $\mathbb{R}^n$ — every configuration has a neighborhood in which it can be described by $n$ real coordinates — but which need **not** be globally diffeomorphic to $\mathbb{R}^n$. Orientation variables are the standard source of this subtlety: rotating a body by $2\pi$ returns it to the same physical configuration, so an orientation coordinate cannot range freely over $\mathbb{R}$ without redundancy; it must live on a circle. Confusing "$n$ real coordinates locally" with "$\mathcal{C} = \mathbb{R}^n$ globally" is the single most common source of error in reasoning about configuration spaces, so the examples below are chosen specifically to make the distinction concrete.

### 1.1 Examples of Configuration Spaces

**(a) A point in the plane.** No orientation, no rigidity constraint — the configuration is simply $q = (x,y)$, and $\mathcal{C} = \mathbb{R}^2$ is globally Euclidean. $n = 2$.

**(b) A rigid body in the plane (planar rigid body).** A configuration is fixed by the position $(x,y) \in \mathbb{R}^2$ of a chosen reference point on the body together with an orientation angle $\theta$. Since $\theta$ and $\theta + 2\pi$ describe the identical physical orientation, $\theta$ is a coordinate on the *circle* $S^1 := \mathbb{R}/2\pi\mathbb{Z}$, not on $\mathbb{R}$. Hence
$$ \mathcal{C} = \mathbb{R}^2 \times S^1, \qquad n = 3. $$
This space (as a manifold) is exactly the group $SE(2)$ of planar rigid transformations.

**(c) A rigid body in three-dimensional space (spatial rigid body).** A configuration is a position $p \in \mathbb{R}^3$ together with an orientation $R \in SO(3)$, where $SO(3)$ denotes the group of $3\times 3$ rotation matrices (orthogonal matrices with determinant $+1$) — itself a 3-dimensional manifold, though not one diffeomorphic to $\mathbb{R}^3$ globally (it is diffeomorphic to real projective 3-space $\mathbb{RP}^3$). As a manifold,
$$ \mathcal{C} = \mathbb{R}^3 \times SO(3), \qquad n = 6. $$
*(This is the underlying manifold of $SE(3)$, the spatial rigid-transformation group; the group's multiplication law is a semidirect product $\mathbb{R}^3 \rtimes SO(3)$, not a direct product, but only the manifold — hence dimension — matters for C-space purposes here.)*

**(d) A rigid body with an attached rod (pinned linkage).** Take the planar rigid body of (b), with configuration $(x,y,\theta) \in \mathbb{R}^2 \times S^1$, and pin a rigid rod to it at a fixed point via a revolute (single-axis rotational) joint, free to swing through an angle $\phi \in S^1$ relative to the body. Before any further constraint closes a loop, the combined configuration is
$$ (x,y,\theta,\phi) \in \mathbb{R}^2 \times S^1 \times S^1 = \mathbb{R}^2 \times T^2, \qquad n=4, $$
where $T^2 := S^1 \times S^1$ is the 2-torus. This is the general pattern for linked rigid bodies before constraints: the *unconstrained* configuration space of a linkage is a product of the individual bodies' spaces.

**(e) A simple planar arm.** An $n$-link, all-revolute, serial planar arm with a fixed base is described entirely by its joint angles $(\theta_1, \dots, \theta_n)$, each living on $S^1$. Hence
$$ \mathcal{C} = T^n := \underbrace{S^1 \times \cdots \times S^1}_{n}, $$
the $n$-torus. *This is the example to keep in mind for the [[curricula/robotics/curriculum|SO-101 Robotics Curriculum]]:* the SO-101's six revolute joints give, in the idealized case of unlimited joint travel, $\mathcal{C} = T^6$. In practice the STS3215 servos have bounded travel (not a full $2\pi$), so the arm's true reachable C-space is a bounded coordinate box $\prod_i [\theta_i^{\min}, \theta_i^{\max}] \subset T^6$ rather than the full torus — a physical caveat worth flagging even though it doesn't change the *dimension* count of §3.3 below.

## 2. Degrees of Freedom 🔑

**Definition (Degrees of Freedom, restated).** $\mathrm{dof} := \dim \mathcal{C}$.

### 2.1 Degrees of Freedom of a Free Rigid Body

**Proposition.** A free (unconstrained) rigid body in the plane has 3 degrees of freedom.

*Proof (constructive, by counting freedoms point by point).* A rigid body is, by definition, a set of points maintaining fixed pairwise distances. Fix a labeled point $A$ on the body: it has $2$ free coordinates $(x_A, y_A) \in \mathbb{R}^2$, unconstrained. Fix a second, distinct point $B$: its position must satisfy $\|B - A\| = d_{AB}$ (a fixed, known distance), which confines $B$ to a circle of radius $d_{AB}$ about $A$ — a $1$-dimensional set, so $B$ contributes $1$ freedom (an angle). Any further point $C$ on the body (not collinear with $A,B$) is fixed by two known distances to $A$ and $B$, which pin down its position up to a two-fold reflection ambiguity; disallowing reflections (rigid-body motions are orientation-preserving) removes this ambiguity, so $C$ contributes $0$ additional freedom. Total: $2 + 1 + 0 = 3$. $\blacksquare$

This matches the direct parametrization of §1.1(b): $2$ for the position of a reference point, $1$ for orientation $\theta \in S^1$.

**Proposition.** A free rigid body in three-dimensional space has 6 degrees of freedom.

*Proof (same method).* Point $A$: $3$ free coordinates, $0$ constraints. Point $B$ (distinct from $A$): fixed distance to $A$ confines it to a 2-sphere, contributing $2$ freedoms. Point $C$ (not collinear with $A,B$): fixed distances to both $A$ and $B$ confine it to the intersection of two spheres — generically a circle — contributing $1$ freedom. Any further point $D$ (not coplanar with $A,B,C$) is pinned by three fixed distances up to a reflection, disallowed as before, contributing $0$. Total: $3+2+1+0 = 6$. $\blacksquare$

Equivalently: $3$ for the position of a reference point, $3$ for orientation $R \in SO(3)$.

> [!NOTE] The general point-counting pattern
> Writing $m$ for the total degrees of freedom of a single free rigid body ($m=3$ planar, $m=6$ spatial), the pattern above is $m = \binom{m}{1}$-many "new" freedoms contributed by the first non-redundant point, decreasing by one per additional constraining point until exhausted. This same counting logic — total freedoms minus independent constraints — is exactly what generalizes to Grubler's formula in §3.

### 2.2 Degrees of Freedom of a System of Rigid Bodies Before Constraints

Consider a mechanism built from $N$ rigid bodies in total, one of which is a fixed *ground link* serving as the reference frame (and hence contributing $0$ freedoms once we work in its frame). Before any joints are introduced to connect the remaining $N-1$ movable bodies, each of them is an independent free rigid body contributing $m$ freedoms ($m=3$ planar, $m=6$ spatial, as in §2.1). The total unconstrained freedom of the system is therefore
$$ \text{freedom before joint constraints} = m(N-1). $$
This quantity is exactly the starting point for Grubler's formula below.

## 3. Grubler's Formula 📐

**Proposition (Grubler's Formula, also written the *Grübler–Kutzbach criterion*).** For a mechanism of $N$ links (including one fixed ground link) connected by $J$ joints, where $m = 3$ for a planar mechanism and $m=6$ for a spatial mechanism, and joint $i$ permits $f_i$ independent relative degrees of freedom between the two links it connects, the degrees of freedom of the mechanism is
$$ \mathrm{dof} = m(N-1-J) + \sum_{i=1}^{J} f_i, $$
**provided all joint constraints are independent** at the configuration in question (see the caveat in §3.1).

### 3.1 Derivation

By §2.2, the $N-1$ movable links have $m(N-1)$ freedoms before any joints are imposed. A joint between two links does not remove all $m$ relative degrees of freedom between them — it removes only $m - f_i$ of them, leaving $f_i$ relative freedoms. Write $c_i := m - f_i$ for the number of independent constraint equations contributed by joint $i$ (e.g. for $m=6$, a revolute joint has $f_i = 1$ and hence $c_i = 5$: it eliminates all relative motion except rotation about one fixed axis). Summing over all $J$ joints and subtracting from the unconstrained total:
$$
\mathrm{dof} = m(N-1) - \sum_{i=1}^J c_i = m(N-1) - \sum_{i=1}^J (m - f_i) = m(N-1) - Jm + \sum_{i=1}^J f_i = m(N-1-J) + \sum_{i=1}^J f_i.
$$

> [!WARNING] The independence caveat
> This derivation silently assumes the $\sum c_i$ constraint equations are independent, i.e. that the stacked constraint Jacobian has full row rank at the configuration considered. When constraints are redundant — which happens in certain symmetric or specially-dimensioned mechanisms — Grubler's formula can *undercount* the true mobility. This is a genuine failure mode of the formula, not a footnote: several well-known "paradoxical" linkages (e.g. some symmetric parallel mechanisms) are mobile despite Grubler's formula predicting $\mathrm{dof} \le 0$.

### 3.2 Worked Example: The Planar Four-Bar Linkage

A planar four-bar linkage has four links total (one of them ground), connected in a single closed loop by four revolute joints. So $m=3$, $N=4$, $J=4$, and $f_i = 1$ for each of the four revolute joints:
$$ \mathrm{dof} = 3(4 - 1 - 4) + 4(1) = 3(-1) + 4 = 1. $$
One degree of freedom — consistent with the familiar fact that turning a single crank of a four-bar linkage fully determines the configuration of every other link (generically).

### 3.3 Worked Example: The SO-101 Serial Chain

The SO-101 arm is an *open* (serial) kinematic chain: a fixed base plus five intermediate links plus a gripper/end-effector link, connected in series by six revolute joints, with no closed loops. Counting the ground/base as one of the links, $N = 7$ (base $+$ 6 moving links, one appended per joint), $J = 6$ revolute joints, $m=6$ (spatial), and $f_i = 1$ for each joint:
$$ \mathrm{dof} = 6(7 - 1 - 6) + 6(1) = 6(0) + 6 = 6. $$

> [!TIP] 💡 Why open chains always give the "obvious" answer
> For *any* open serial chain, each joint introduces exactly one new link, so $N - 1 = J$ always — the $m(N-1-J)$ term vanishes identically, and Grubler's formula collapses to $\mathrm{dof} = \sum_i f_i$: just the sum of joint freedoms, with no interesting cancellation. Grubler's formula earns its keep on *closed*-loop (parallel) mechanisms like the four-bar linkage of §3.2, where $N - 1 \ne J$ and the formula's cancellation is doing real work.

This $6$ matches the dimension of the C-space $T^6$ computed directly in §1.1(e) — a useful cross-check that the two counting methods (product-manifold construction vs. Grubler's formula) agree.

> [!QUESTION] Exercise 1: Degrees of Freedom of a Planar Five-Bar Linkage
> *This exercise checks fluency with Grubler's formula on a closed-loop mechanism distinct from the worked four-bar example, and connects the result to a common class of 2-DOF planar manipulators.*
>
> > **Prerequisites:** [[#3. Grubler's Formula|3. Grubler's Formula]]
>
> A planar five-bar linkage consists of five rigid links (including the fixed ground link) connected in a single closed loop by five revolute joints. Using Grubler's formula, compute its degrees of freedom, and briefly interpret the result relative to the four-bar linkage of §3.2 (in particular: how many independently actuated joints does each mechanism need to fully determine its configuration?).

> [!TIP]- Solution to Exercise 1
> **Key insight:** the five-bar linkage differs from the four-bar only in having one more link and one more joint, both closed in a single loop, so $N=5, J=5$ (vs. $N=4,J=4$ for the four-bar) while $m=3$ and $f_i=1$ throughout.
>
> **Sketch:** $\mathrm{dof} = 3(5-1-5) + 5(1) = 3(-1) + 5 = 2.$ Two degrees of freedom, versus the four-bar's one. Physically: a four-bar linkage needs only a single actuated crank to fully determine its configuration, while a five-bar linkage needs *two* independently actuated joints — this is exactly the structure behind common 2-DOF planar parallel manipulators (e.g. five-bar SCARA-like mechanisms), where two motors at the base drive a closed five-bar loop to position an end-effector in the plane.

## 4. Holonomic and Nonholonomic Constraints ⚠️

### 4.1 Holonomic Constraints

**Definition (Holonomic Constraint).** Given configuration coordinates $q \in \mathbb{R}^N$ for some ambient (possibly over-parametrized) description of a system, a constraint expressible purely as an equation on configuration,
$$ g(q) = 0, \qquad g : \mathbb{R}^N \to \mathbb{R}^k, $$
is a *holonomic constraint*. If $g$ is smooth and its Jacobian $Dg(q)$ has full rank $k$ at every solution (a *regular value*), the implicit function theorem guarantees that the solution set $\{q : g(q) = 0\}$ is itself a smooth manifold of dimension $N-k$ — a codimension-$k$ submanifold of the ambient space. **Imposing a holonomic constraint directly reduces the dimension of the configuration space, generically by one dimension per independent scalar constraint.**

**Definition (Pfaffian Velocity Constraint).** A constraint linear in the generalized velocities $\dot q$, of the form
$$ A(q)\,\dot q = 0, \qquad A(q) \in \mathbb{R}^{k \times N}, $$
with configuration-dependent coefficients $A(q)$, is a *Pfaffian constraint*.

Differentiating a holonomic constraint $g(q) = 0$ with respect to time gives $Dg(q)\,\dot q = 0$ by the chain rule — so every holonomic constraint induces a Pfaffian constraint with $A(q) = Dg(q)$. Holonomic constraints are thus the special case of Pfaffian constraints whose $A(q)$ is (locally, up to an invertible left factor) the Jacobian of some function $g$ — i.e. *integrable*.

**Definition (Nonholonomic Constraint).** A Pfaffian constraint $A(q)\dot q = 0$ for which no function $g : \mathbb{R}^N \to \mathbb{R}^k$ exists, even locally, with $A(q)$ proportional to $Dg(q)$, is called *nonholonomic*, or *non-integrable*.

### 4.2 Nonholonomic Constraints and the Rolling Wheel

Consider a single wheel of radius $r$ rolling upright on a plane (the standard "unicycle" idealization). Its configuration is $q = (x,y,\phi,\theta) \in \mathbb{R}^2 \times S^1 \times S^1$, where $(x,y)$ is the contact point, $\phi$ is the heading angle (direction the wheel points), and $\theta$ is the rolling angle (rotation of the wheel about its own axle). So $N=4$.

Rolling without slipping requires the contact point's instantaneous velocity to point purely along the heading direction $(\cos\phi, \sin\phi)$ — no lateral sliding. Equivalently, the velocity $(\dot x, \dot y)$ must have zero component along the perpendicular direction $(-\sin\phi, \cos\phi)$:
$$ \dot x \sin\phi - \dot y \cos\phi = 0. $$
This is a Pfaffian constraint with $A(q) = [\sin\phi, \; -\cos\phi, \; 0, \; 0] \in \mathbb{R}^{1\times 4}$, $k=1$; note $A(q)$ places **no** restriction whatsoever on $\dot\phi$ or $\dot\theta$.

> [!WARNING] The following argument is heuristic
> A fully rigorous non-integrability proof uses the Frobenius theorem (checking that the constraint distribution's annihilator fails to be involutive, or equivalently exhibiting a nonzero Lie bracket of vector fields spanning the feasible-velocity distribution). That machinery is outside the scope of this note; the argument below is the standard *informal* justification given in introductory treatments, and is flagged as such per this repo's no-hand-waving convention.

*Heuristic argument for non-integrability.* Suppose, for contradiction, the constraint were holonomic — i.e. some $g(x,y,\phi,\theta) = 0$ (independent of $\theta$, since $A$ has no $\theta$-component) held along every feasible trajectory, restricting $(x,y,\phi)$ to a fixed 2-dimensional subset of its ambient 3 dimensions for all time. But because $A(q)$ imposes no restriction on $\dot\phi$ at all, the heading can be changed *freely* at any instant, independent of position. By alternating "roll forward along the current heading" and "reorient to a new heading" segments — precisely the maneuver used to parallel-park a car — one can return $\phi$ (and $\theta$) to their exact starting values while net-translating the contact point $(x,y)$ to a different nearby location. This directly contradicts the existence of a single conserved relation $g(x,y,\phi) = 0$: the reachable set of $(x,y,\phi)$ is fully 3-dimensional, even though the instantaneous velocity is confined to a 1-dimensional line at every point. Hence no such $g$ exists, and the constraint is nonholonomic. $\blacksquare$ *(heuristic)*

### 4.3 Effect on the Configuration Space

**A holonomic constraint reduces the dimension of the configuration space itself — the reachable set becomes a genuinely lower-dimensional submanifold. A nonholonomic constraint restricts only the admissible velocity directions at each configuration (a $k$-codimensional distribution on the tangent bundle) without reducing the dimension of the reachable configuration space.**

> [!NOTE] 💡 Pointer to controllability (heuristic aside)
> Under mild conditions — informally, that the admissible velocity directions and their iterated Lie brackets eventually span the full tangent space at every point (the *Lie Algebra Rank Condition*), formalized by the Chow–Rashevskii theorem — a nonholonomically-constrained system remains fully controllable across the *entire* original configuration space, despite instantaneous velocity being confined to a lower-dimensional subspace everywhere. This is why wheeled vehicles can reach any planar position and heading despite never being able to slide sideways. The full proof is outside the scope of this note; it becomes directly relevant at Stage 2 of the curriculum ([[curricula/robotics/curriculum|Beyond This Curriculum]] — mobile robotics), which this note's cluster does not yet cover.

*This matters directly for the current curriculum stage:* **a fixed-base serial arm such as the SO-101 has purely holonomic constraints.** Each revolute joint restricts the relative configuration of two adjacent links to a one-parameter subgroup of rotations about a fixed axis — a constraint expressible directly at the configuration level (it is exactly the $f_i = 1$ freedom counted in §3.3), and there is no rolling-contact or nonslip constraint anywhere in the chain. Consequently, the SO-101's full C-space is precisely the product manifold of §1.1(e) ($T^6$, modulo joint-range limits), with no velocity-only restrictions beyond those already implied by the joint structure. This is why the arm's kinematics can be treated entirely at the configuration level — as maps between manifolds, forward and inverse — without the extra nonholonomic-velocity machinery ([[curricula/robotics/curriculum|Module 3: Kinematics and Dynamics]]) that a wheeled mobile robot would require.

> [!QUESTION] Exercise 2: Classifying Two Constraints
> *This exercise tests the integrability criterion of §4.1–4.2 directly, contrasting a configuration-level constraint with a velocity-level one on the same coordinate system.*
>
> > **Prerequisites:** [[#4. Holonomic and Nonholonomic Constraints|4. Holonomic and Nonholonomic Constraints]]
>
> For a system with configuration $q = (x,y,\phi) \in \mathbb{R}^2 \times S^1$, classify each of the following constraints as holonomic or nonholonomic, and justify briefly:
>
> (a) $x^2 + y^2 - r^2 = 0$ (the reference point is confined to a circle of radius $r$).
>
> (b) $\dot x \sin\phi - \dot y \cos\phi = 0$ (the no-lateral-sliding constraint of §4.2).

> [!TIP]- Solution to Exercise 2
> **Key insight:** (a) is already written purely as a function of configuration with no velocity terms, so integrability is not in question — it *is* the integrated form. (b) is exactly the rolling constraint whose non-integrability was argued in §4.2.
>
> **Sketch:** (a) Holonomic: $g(q) = x^2+y^2-r^2$ involves only $q$, not $\dot q$; since $Dg = (2x, 2y, 0)$ has rank $1$ away from the origin, the constraint set $\{x^2+y^2=r^2\}$ is a codimension-$1$ submanifold of $\mathbb{R}^2\times S^1$ (a cylinder $S^1_{\text{radius }r} \times S^1_\phi$). (b) Nonholonomic, by the heuristic parallel-parking argument of §4.2: $A(q)$ constrains only the direction of $(\dot x,\dot y)$ relative to $\phi$, leaves $\dot\phi$ completely free, and no conserved relation $g(x,y,\phi)=0$ can hold along all feasible trajectories.

## 5. Task Space versus Configuration Space 🎯

**Definition (Task Space).** A *task space* $\mathcal{T}$ is a space in which a robot's task is naturally and directly expressed, chosen according to the task at hand rather than the robot's internal joint structure. For a pick-and-place task the relevant task space might be $SE(3)$ (end-effector position and orientation) or simply $\mathbb{R}^3$ (position only, if orientation is unconstrained by the task); for planar navigation it might be $\mathbb{R}^2$.

**Definition (Workspace).** The *workspace* of a robot is the set of end-effector poses reachable over all valid configurations $q \in \mathcal{C}$ — the image of the *forward kinematics map* $f : \mathcal{C} \to \mathcal{T}$ (or into whatever ambient space contains the task/workspace). Unlike task space, the workspace is determined by the robot's mechanical structure, not by the task it is being asked to perform.

Because $f$ need not be a bijection, $\dim \mathcal{C}$ need not equal $\dim \mathcal{T}$ — this asymmetry is exactly what makes inverse kinematics (recovering $q$ from a desired $f(q)$) a nontrivial problem, taken up in depth in [[curricula/robotics/curriculum|Module 3: Kinematics and Dynamics]] of the curriculum.

### 5.1 Redundancy and Underactuation

- **Kinematic redundancy:** $\dim \mathcal{C} > \dim \mathcal{T}$. The manipulator has more independently controllable degrees of freedom than strictly necessary to achieve any given task-space target, so the preimage $f^{-1}(x)$ of a reachable target $x \in \mathcal{T}$ is generically a positive-dimensional submanifold of $\mathcal{C}$ — a continuum of joint configurations all achieving the same end-effector pose. This extra freedom is exactly what redundancy-resolution / null-space control techniques exploit.
- **Underactuation:** a distinct notion — fewer independently *actuated* inputs than $\dim \mathcal{C}$, meaning some directions in the configuration space cannot be commanded instantaneously and must be reached indirectly through the system's dynamics. Underactuation is a statement about actuation, not about the $\mathcal{C}$-versus-$\mathcal{T}$ dimension mismatch, though the two often co-occur in practice; a full treatment belongs to the dynamics note later in this cluster.
- **The SO-101 concretely:** $\dim \mathcal{C} = 6$ (§3.3). If the task is full end-effector pose ($\dim \mathcal{T} = \dim SE(3) = 6$), the arm is exactly matched — non-redundant, generically admitting only an isolated, finite set of joint solutions for a given reachable pose. If the task is position only ($\dim \mathcal{T} = 3$), the arm is redundant by $3$ degrees of freedom for that reduced task, and inverse kinematics admits a $3$-parameter family of solutions at each reachable target position — exactly the machinery [[curricula/robotics/curriculum|Module 3]] develops via the Jacobian and Newton–Raphson iterative IK.

> [!QUESTION] Exercise 3: Redundancy of a Planar Arm
> *This exercise applies the dimension-counting argument for redundancy directly, using a planar analogue of the SO-101 example above, and previews why 3-revolute planar arms are the minimal "fully posed" planar manipulator.*
>
> > **Prerequisites:** [[#5. Task Space versus Configuration Space|5. Task Space versus Configuration Space]], [[#1. Configuration and Configuration Space|1. Configuration and Configuration Space]]
>
> A planar serial arm has $n$ revolute joints (coplanar, base fixed), so by §1.1(e) its configuration space is the $n$-torus $T^n$. Suppose its task is to place its end-effector at a specified point $(x,y)$ in the plane, ignoring orientation, so $\mathcal{T} = \mathbb{R}^2$.
>
> (a) For which values of $n$ is the arm kinematically redundant for this task?
>
> (b) Now suppose the task additionally specifies end-effector orientation $\phi \in S^1$, so $\mathcal{T} = \mathbb{R}^2 \times S^1$ ($\dim \mathcal{T} = 3$). Is a 3-link planar arm ($n=3$) redundant for this augmented task?

> [!TIP]- Solution to Exercise 3
> **Key insight:** redundancy is purely a dimension comparison, $\dim\mathcal{C} = n$ versus $\dim\mathcal{T}$.
>
> **Sketch:** (a) $\dim\mathcal{C} = n$, $\dim\mathcal{T}=2$, so the arm is redundant whenever $n > 2$, i.e. $n \ge 3$. At $n=2$ the dimensions match exactly (the classic 2-link planar arm, with a locally finite set of IK solutions — "elbow up / elbow down" — for a reachable position target); at $n=1$ the arm is under-dimensioned and generically cannot reach arbitrary points of $\mathbb{R}^2$ at all (its reachable set is only 1-dimensional, a circle). (b) For $n=3$ and $\dim\mathcal{T} = 3$, $\dim\mathcal{C} = \dim\mathcal{T}$ exactly — the arm is *not* redundant for the full-pose task. This is exactly why the 3-revolute (RRR) planar arm is the standard minimal manipulator capable of achieving arbitrary reachable planar poses (position and orientation together), generically with a finite number of IK solutions rather than a continuum.

## References

| Reference Name | Brief Summary | Link to Reference |
|---|---|---|
| Modern Robotics: Mechanics, Planning, and Control (Lynch & Park, 2017) | Primary source for this note — Ch. 1–2 cover configuration space, degrees of freedom, Grubler's formula, holonomic/nonholonomic constraints, and task space vs. workspace | [Free PDF](http://hades.mech.northwestern.edu/images/7/7f/MR.pdf) |
| 2.1 Degrees of Freedom of a Rigid Body (Modern Robotics companion site) | Companion video/text resource for §2 — point-counting derivation of rigid-body DOF in 2D and 3D | [modernrobotics.northwestern.edu](https://modernrobotics.northwestern.edu/nu-gm-book-resource/2-1-degrees-of-freedom-of-a-rigid-body/) |
| 2.2 Degrees of Freedom of a Robot (Modern Robotics companion site) | Companion video/text resource for §3 — states and derives Grubler's formula, with the four-bar-linkage and Stewart-platform worked examples used to cross-check §3 | [modernrobotics.northwestern.edu](https://modernrobotics.northwestern.edu/nu-gm-book-resource/2-2-degrees-of-freedom-of-a-robot/) |
| Chapter 2 Overview: Foundations of Robot Motion (Modern Robotics companion site) | Section-by-section outline of Ch. 2, used to structure this note's coverage of C-space topology, constraints, and task space/workspace | [modernrobotics.northwestern.edu](https://modernrobotics.northwestern.edu/chapters/chapter2/) |
| 13.3.1 Modeling of Nonholonomic Wheeled Mobile Robots (Modern Robotics companion site) | Source for the rolling-wheel nonslip constraint equation used in §4.2 | [modernrobotics.northwestern.edu](https://modernrobotics.northwestern.edu/nu-gm-book-resource/13-3-1-modeling-of-nonholonomic-wheeled-mobile-robots/) |
| Park & Lynch, Introduction to Robotics (early edition text) | Cross-checked exact wording of the configuration/C-space, Grubler's formula, task space/workspace, and holonomic/nonholonomic definitions against the published book | [archive.org](https://archive.org/stream/parklynchintroductiontorobotics/Park-lynch%20introduction%20to%20robotics_djvu.txt) |
| SO-101 Robotics Curriculum | The curriculum this note's Module 0 reading supports; source of the SO-101 six-revolute-joint worked example in §3.3 | [[curricula/robotics/curriculum\|SO-101 Robotics Curriculum]] |
