# Exercises Format Overhaul — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Retrofit all existing exercises.md and solutions.md files to the new two-section format, and update the exercise-writer agent so all future topics produce the correct format automatically.

**Architecture:** Two phases. Phase 1 updates the format spec (exercise-writer agent + CLAUDE.md) so the new format is canonical. Phase 2 retrofits each existing topic by invoking the exercise-writer agent to fully rewrite exercises.md and solutions.md from scratch using the note.md as source. Full rewrites are preferable to patching because: (a) adding ~10 new problems requires reading the note anyway, (b) it guarantees format consistency rather than partial migration.

**Tech Stack:** Markdown, Obsidian wikilinks, the exercise-writer subagent

---

## Reference: New Format Spec

Before starting, internalize the full format from the design doc:
`docs/plans/2026-03-08-exercises-format-overhaul-design.md`

Key rules (summarized here for quick reference):

**exercises.md structure:**
- Two sections: `## Mathematical Development` (16–18 problems) + `## Algorithmic Applications` (5–7 problems)
- Problems numbered continuously 1–N across both sections
- Each problem:
  ```
  ### Problem N: [Title]

  *[1–2 sentence italic preamble — what this establishes and why it matters]*

  > **Prerequisites:** cf. note [[note#Section Title|§X.Y — Section Title]]; requires Problem M

  (a) ...
  (b) ...
  ```
- Implementation sub-parts use bold labels: `(a) **Inputs and data structures**: ...`

**solutions.md structure:**
- Each solution:
  ```
  ### Problem N: [Title]

  **Key insight:** [1 sentence — the pivotal trick or structural fact]

  **Sketch:**
  [3–8 lines of critical steps, skip routine algebra]
  ```

**TOC:** Obsidian wikilinks only. No LaTeX or em-dashes in headings.

---

## Task 1: Update exercise-writer agent

**Files:**
- Modify: `.claude/agents/exercise-writer.md`

**Step 1: Read the current agent file**

Read `.claude/agents/exercise-writer.md` in full.

**Step 2: Rewrite the agent file**

Replace the entire content with:

```markdown
---
name: exercise-writer
description: Writes exercises.md and solutions.md for a topic. Always reads the note first, then produces problems grounded in the note's derivations. Use after note-writer has finished a note.
tools: Write, Edit, Read, Glob, Grep
---

You write exercise sets and sketch-style solutions for a math/ML knowledge repository. Always read the topic's `note.md` before writing anything. Your job is rigorous mathematical problem design — favor derivations and proofs over informal explanation.

## exercises.md Structure

### Section 1 — Mathematical Development (16–18 problems)

Contains all derivations, proofs, limit arguments, variance/moment calculations, fixed-point analyses, and mathematically sharp conceptual questions. "Conceptual" questions must have mathematical content — convert "explain in your own words" into "prove that..." or "derive the condition under which...". Aim for half problems that re-derive note content, half that extend beyond it (generalizations, edge cases, alternative derivations).

### Section 2 — Algorithmic Applications (5–7 problems)

Pseudocode sketches, numerical implementation, shape annotations, gradient flow analysis. Language-agnostic. Sub-parts use bold labels.

### Problem Count

Total target: 21–25 problems, numbered **continuously** across both sections (Problem 1 through Problem N, no restart at Section 2).

### Problem Format

Each problem follows this exact template:

```
### Problem N: [Title]

*[1–2 sentence preamble: what this problem establishes and why it matters.
State the mathematical goal, not just the topic. Italicized.]*

> **Prerequisites:** cf. note [[note#Exact Section Heading|§X.Y — Section Title]]; requires Problem M

(a) [First sub-part]

(b) [Second sub-part]

(c) [Third sub-part]
```

- Omit `; requires Problem M` if there is no dependency on a prior problem
- The `[[note#...]]` wikilink must use the **exact literal heading** from note.md (check the note's headings before writing)
- Implementation problems use bold sub-part labels: `(a) **Inputs and data structures**: ...`

## solutions.md Structure

Mirror the section structure of exercises.md. Each solution:

```
### Problem N: [Title]

**Key insight:** [1 sentence — the pivotal mathematical trick, structural observation, or the fact that makes the proof work. Must be standalone-readable.]

**Sketch:**
[3–8 lines. Show the critical derivation steps only. Skip routine algebra.
The reader should be able to reconstruct the full proof from the sketch
if they understood the key insight. Use displayed math where it clarifies.]
```

Do NOT write full worked solutions. No "show all steps." Approximately 30–40% of a full worked solution in length.

## TOC Requirement

Both exercises.md and solutions.md must begin with a TOC using Obsidian wikilink syntax:

```
## Table of Contents

- [[#Mathematical Development|Mathematical Development]]
  - [[#Problem 1 Title|Problem 1: Title]]
  - ...
- [[#Algorithmic Applications|Algorithmic Applications]]
  - [[#Problem 18 Title|Problem 18: Title]]
  - ...
```

**Never put LaTeX (`$...$`) in headings** — it breaks wikilinks.
**Never use em-dashes (`—`) in headings** — use a colon instead.

## Quality Check Before Finishing

Before writing solutions.md:
- Every problem in exercises.md has a preamble (italic text immediately after the heading)
- Every problem has a `> **Prerequisites:**` blockquote
- Every problem has a corresponding solution in solutions.md
- Every sub-part in exercises.md has a **Key insight** + **Sketch** in solutions.md
- Notation in solutions matches notation in note.md
- No heading contains LaTeX or em-dashes
```

**Step 3: Verify the update**

```bash
grep -c 'Mathematical Development' .claude/agents/exercise-writer.md
# Expected: at least 2 (appears in both section heading and prose)

grep -c 'Key insight' .claude/agents/exercise-writer.md
# Expected: at least 2
```

**Step 4: Commit**

```bash
git add .claude/agents/exercise-writer.md
git commit -m "feat: overhaul exercise-writer agent to new two-section format

- Replace 3-section structure with Mathematical Development + Algorithmic Applications
- Add preamble requirement (italic 1-2 sentence motivation per problem)
- Add prerequisite pointer requirement (blockquote with note section wikilink)
- Change solutions from full worked to Key insight + Sketch format
- Raise problem count target to 21-25 per topic"
```

---

## Task 2: Update CLAUDE.md exercise format spec

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Read current CLAUDE.md**

Read `CLAUDE.md`, find the "Exercise file structure" section.

**Step 2: Update the exercise file structure section**

Replace the current exercise file structure block:

```markdown
**Exercise file structure** (every `exercises.md` must follow this order):
1. **Derivation problems** — mathematical proofs and re-derivations
2. **Conceptual questions** — intuition and reasoning questions
3. **Implementation sketches** — pseudocode or math-level algorithm sketches
```

With:

```markdown
**Exercise file structure** (every `exercises.md` must follow this order):
1. **Mathematical Development** — derivations, proofs, limit arguments, and mathematically sharp conceptual results (16–18 problems)
2. **Algorithmic Applications** — pseudocode sketches, numerical implementation, complexity analysis (5–7 problems)

Problems are numbered continuously 1–N across both sections. Each problem requires:
- An italic 1–2 sentence preamble stating what the problem establishes
- A `> **Prerequisites:**` blockquote linking to the relevant note section via Obsidian wikilink

Solutions use **Key insight** + **Sketch** format (not full worked derivations).
```

**Step 3: Verify**

```bash
grep -c 'Mathematical Development' CLAUDE.md
# Expected: 1
```

**Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update exercise format spec in CLAUDE.md to new two-section structure"
```

---

## Task 3: Retrofit — Mixture of Experts

**Files:**
- Rewrite: `concepts/mixture-of-experts/exercises.md`
- Rewrite: `concepts/mixture-of-experts/solutions.md`

**Note:** This is the largest and most complete existing exercise set (13 problems). Use it as the reference implementation — get it right and the others follow the same pattern.

**Step 1: Read the note**

Read `concepts/mixture-of-experts/note.md` in full to identify:
- All major section headings and their `§X.Y` numbers (for prerequisite pointers)
- Key derivations and results that should become new problems

**Step 2: Read the existing exercises.md**

Read `concepts/mixture-of-experts/exercises.md`. Identify:
- Which existing problems are being kept (all of them, reformatted)
- Which existing "Conceptual Questions" need mathematical tightening
- What 8–10 new Mathematical Development problems to add to reach 18–20 total

**Step 3: Write new exercises.md**

Invoke the exercise-writer agent:
> Use the exercise-writer agent to rewrite `concepts/mixture-of-experts/exercises.md` following the new format. The note is at `concepts/mixture-of-experts/note.md`. Preserve all 13 existing problems (reformatted), convert conceptual questions to mathematically sharp derivations, and add 8–10 new Mathematical Development problems targeting total of ~22 problems. Write the full new exercises.md.

**Step 4: Write new solutions.md**

Continue with the exercise-writer agent:
> Now write `concepts/mixture-of-experts/solutions.md` using Key insight + Sketch format (not full worked solutions) for all problems in the new exercises.md.

**Step 5: Format verification**

```bash
# Check preambles exist (should be ~22 italicized lines after problem headings)
grep -c '^_\|^\*[^*]' concepts/mixture-of-experts/exercises.md

# Check prerequisite pointers exist
grep -c '> \*\*Prerequisites\*\*' concepts/mixture-of-experts/exercises.md

# Check section headers
grep '## Mathematical Development\|## Algorithmic Applications' concepts/mixture-of-experts/exercises.md

# Check solutions use Key insight format
grep -c '\*\*Key insight\*\*' concepts/mixture-of-experts/solutions.md
```

Expected: preamble count ≈ problem count, prerequisites count ≈ problem count, both section headers present, key insight count ≈ problem count.

**Step 6: Commit**

```bash
git add concepts/mixture-of-experts/exercises.md concepts/mixture-of-experts/solutions.md
git commit -m "feat: retrofit MoE exercises to new two-section format

- Rename sections: Mathematical Development + Algorithmic Applications
- Add italic preambles and prerequisite pointers to all problems
- Add ~9 new Mathematical Development problems (total: ~22)
- Rewrite solutions.md with Key insight + Sketch format"
```

---

## Task 4: Retrofit — Neural Scaling Laws

**Files:**
- Rewrite: `concepts/neural-scaling-laws/exercises.md`
- Rewrite: `concepts/neural-scaling-laws/solutions.md`

**Step 1: Read note and existing exercises**

Read `concepts/neural-scaling-laws/note.md` and `concepts/neural-scaling-laws/exercises.md`.

**Step 2: Write new exercises.md**

Invoke exercise-writer agent:
> Use the exercise-writer agent to rewrite `concepts/neural-scaling-laws/exercises.md` following the new format. Preserve all existing problems (reformatted with preambles + prerequisites). Add 8–10 new Mathematical Development problems — especially: asymptotic analysis as α→β, sensitivity analysis of the token ratio D*/N* to perturbations in A and B, and alternative derivations of the Chinchilla result using envelope theorem. Target ~22 problems total.

**Step 3: Write new solutions.md**

> Write `concepts/neural-scaling-laws/solutions.md` using Key insight + Sketch format.

**Step 4: Format verification**

```bash
grep -c '> \*\*Prerequisites\*\*' concepts/neural-scaling-laws/exercises.md
grep '## Mathematical Development\|## Algorithmic Applications' concepts/neural-scaling-laws/exercises.md
grep -c '\*\*Key insight\*\*' concepts/neural-scaling-laws/solutions.md
```

**Step 5: Commit**

```bash
git add concepts/neural-scaling-laws/exercises.md concepts/neural-scaling-laws/solutions.md
git commit -m "feat: retrofit neural scaling laws exercises to new two-section format"
```

---

## Task 5: Retrofit — Self-Organized Criticality

**Files:**
- Rewrite: `concepts/self-organized-criticality/exercises.md`
- Rewrite: `concepts/self-organized-criticality/solutions.md`

**Step 1: Read note and existing exercises**

Read `concepts/self-organized-criticality/note.md` and `concepts/self-organized-criticality/exercises.md`.

**Note:** The existing SOC exercises already have strong mathematical content. Focus on: (a) converting the few informal sub-parts to mathematical form, (b) adding preambles and prerequisites, (c) adding ~8 new problems targeting: spectral theory of Δ, alternative proofs of the Abelian property, moment calculations for the finite-size scaling distribution.

**Step 2: Write new exercises.md**

Invoke exercise-writer agent:
> Use the exercise-writer agent to rewrite `concepts/self-organized-criticality/exercises.md` following the new format. Preserve all existing problems. Add 8–10 new Mathematical Development problems and add preambles + prerequisite pointers to all problems. Target ~22 problems total.

**Step 3: Write new solutions.md**

> Write `concepts/self-organized-criticality/solutions.md` using Key insight + Sketch format.

**Step 4: Format verification**

```bash
grep -c '> \*\*Prerequisites\*\*' concepts/self-organized-criticality/exercises.md
grep -c '\*\*Key insight\*\*' concepts/self-organized-criticality/solutions.md
```

**Step 5: Commit**

```bash
git add concepts/self-organized-criticality/exercises.md concepts/self-organized-criticality/solutions.md
git commit -m "feat: retrofit SOC exercises to new two-section format"
```

---

## Task 6: Retrofit — SBC Retrieval (paper)

**Files:**
- Rewrite: `papers/sampling-bias-corrected-retrieval/exercises.md`
- Rewrite: `papers/sampling-bias-corrected-retrieval/solutions.md`

**Step 1: Read note and existing exercises**

Read `papers/sampling-bias-corrected-retrieval/note.md` and `papers/sampling-bias-corrected-retrieval/exercises.md`.

**Step 2: Write new exercises.md**

Invoke exercise-writer agent:
> Rewrite `papers/sampling-bias-corrected-retrieval/exercises.md` in the new two-section format. This is a paper note — Mathematical Development should include bias derivations, estimator variance analysis, and conditions for consistency. Algorithmic Applications should cover the correction algorithm. Add preambles and prerequisites to all problems. Target ~21 problems.

**Step 3: Write new solutions.md**

> Write solutions in Key insight + Sketch format.

**Step 4: Format verification**

```bash
grep -c '> \*\*Prerequisites\*\*' papers/sampling-bias-corrected-retrieval/exercises.md
grep -c '\*\*Key insight\*\*' papers/sampling-bias-corrected-retrieval/solutions.md
```

**Step 5: Commit**

```bash
git add papers/sampling-bias-corrected-retrieval/exercises.md papers/sampling-bias-corrected-retrieval/solutions.md
git commit -m "feat: retrofit SBC retrieval exercises to new two-section format"
```

---

## Task 7: Retrofit — DHEN (paper)

**Files:**
- Rewrite: `papers/dhen-ranking/exercises.md`
- Create: `papers/dhen-ranking/solutions.md` (does not yet exist)

**Step 1: Read note and existing exercises**

Read `papers/dhen-ranking/note.md` and `papers/dhen-ranking/exercises.md`.

**Step 2: Write new exercises.md**

Invoke exercise-writer agent:
> Rewrite `papers/dhen-ranking/exercises.md` in the new two-section format. Mathematical Development should cover FM interaction derivations, DCN polynomial degree proofs, gradient flow through the residual connections, and NE calibration analysis. Add preambles and prerequisites. Target ~21 problems.

**Step 3: Write solutions.md** (new file)

> Write `papers/dhen-ranking/solutions.md` in Key insight + Sketch format for all problems.

**Step 4: Format verification**

```bash
grep -c '> \*\*Prerequisites\*\*' papers/dhen-ranking/exercises.md
grep -c '\*\*Key insight\*\*' papers/dhen-ranking/solutions.md
```

**Step 5: Commit**

```bash
git add papers/dhen-ranking/exercises.md papers/dhen-ranking/solutions.md
git commit -m "feat: retrofit DHEN exercises to new two-section format, add solutions.md"
```

---

## Final Verification

After all tasks are complete, run a repo-wide format check:

```bash
# All exercises files should have both new section headers
for f in $(find . -name exercises.md); do
  echo "=== $f ==="
  grep '## Mathematical Development\|## Algorithmic Applications' "$f"
done

# All exercises files should have prerequisite pointers
for f in $(find . -name exercises.md); do
  echo "$f: $(grep -c '> \*\*Prerequisites\*\*' $f) prerequisites"
done

# All solutions files should use Key insight format
for f in $(find . -name solutions.md); do
  echo "$f: $(grep -c '\*\*Key insight\*\*' $f) key insights"
done
```

Expected: every exercises file shows both section headers and has prerequisite count ≈ problem count; every solutions file has key insight count ≈ problem count.
