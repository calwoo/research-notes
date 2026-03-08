# Typographic Style Rules Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Apply consistent bold/italic typographic rules to all existing note.md files and codify the rules in CLAUDE.md and the note-writer agent.

**Architecture:** Each note gets an editing pass by a general-purpose agent that applies three rules: (1) italicize first-use technical terms in prose, (2) ensure formal definition/proposition/remark labels are bold, (3) bold key conclusions and italicize warnings/caveats/counterintuitive results. Rules are then codified in CLAUDE.md and note-writer.md.

**Tech Stack:** Edit tool, general-purpose agents, git.

---

### Task 1: Codify style rules in CLAUDE.md and note-writer agent

**Files:**
- Modify: `CLAUDE.md`
- Modify: `.claude/agents/note-writer.md`

**Step 1: Add typographic style section to CLAUDE.md**

Add the following section after the "Notes Format" section in `/Users/calvinwoo/Documents/notes/CLAUDE.md`:

```markdown
### Typographic Style Rules

Apply these consistently in all `note.md` files:

| Element | Style | Example |
|---------|-------|---------|
| First use of a technical term in prose | *italics* | the *capacity factor* controls overflow |
| Formal definition / proposition / remark label | **bold** | `**Definition (Soft MoE Output).**` |
| Key conclusion — main quantitative takeaway of a derivation | **bold** | `**Both N and D should scale as √C.**` |
| Counterintuitive result | *italics* with inline signal | *Surprisingly,* linear alone outperforms DCN... |
| Warning or caveat | *italics* | *This bound only holds for $T \to \infty$.* |

Do NOT italicize terms after their first use. Do NOT bold entire sentences except for genuine key conclusions.
```

**Step 2: Add the same rules to note-writer agent**

Add a `## Typographic Style` section to `/Users/calvinwoo/Documents/notes/.claude/agents/note-writer.md` with the same table.

**Step 3: Commit**

```bash
git add CLAUDE.md .claude/agents/note-writer.md
git commit -m "docs: add typographic style rules to CLAUDE.md and note-writer agent"
```

---

### Task 2: Apply style rules to MoE note

**Files:**
- Modify: `concepts/mixture-of-experts/note.md`

**Step 1: Apply rules**

Read the file. Apply:
- *Italics* on first use of: *mixture of experts*, *gating network*, *sparse activation*, *conditional computation*, *top-k gating*, *noisy top-k gating*, *load balancing*, *capacity factor*, *token dropping*, *expert collapse*, *router z-loss*, *expert parallelism*, *expert-choice routing*, *token-choice routing*, and any other technical term introduced without markup on first use in prose.
- **Bold** on definition/remark labels already present — verify consistency.
- **Bold** on key conclusions (e.g. the sentence stating MoE decouples FLOPs from parameters).
- *Italics* on warnings/caveats (e.g. notes about approximations or conditions).

**Step 2: Commit**

```bash
git add concepts/mixture-of-experts/note.md
git commit -m "style: apply typographic rules to MoE note"
```

---

### Task 3: Apply style rules to neural scaling laws note

**Files:**
- Modify: `concepts/neural-scaling-laws/note.md`

**Step 1: Apply rules**

First-use terms to italicize (among others): *neural scaling law*, *power law*, *IsoFLOP curve*, *compute-optimal*, *token budget*, *irreducible loss*, *data-limited regime*, *Chinchilla scaling*, *loss exponent*, *effective compute*.

Key conclusions to bold: the Kaplan exponents ($N^* \propto C^{0.73}$), the Chinchilla result ($N^* \propto C^{0.5}$, "20 tokens per parameter"), the three-regime structure.

Warnings/caveats to italicize: notes about approximation validity, dataset-specific exponent variation.

**Step 2: Commit**

```bash
git add concepts/neural-scaling-laws/note.md
git commit -m "style: apply typographic rules to neural scaling laws note"
```

---

### Task 4: Apply style rules to SOC note

**Files:**
- Modify: `concepts/self-organized-criticality/note.md`

**Step 1: Apply rules**

First-use terms to italicize (among others): *self-organized criticality*, *sandpile model*, *avalanche*, *toppling rule*, *abelian property*, *attractor*, *universality class*, *power spectrum*, *finite-size scaling*, *BTW model*, *critical exponent*, *1/f noise*.

Bold key conclusions (e.g. the statement that SOC systems self-tune to criticality without parameter adjustment).

Italicize caveats about empirical power-law attribution.

**Step 2: Commit**

```bash
git add concepts/self-organized-criticality/note.md
git commit -m "style: apply typographic rules to SOC note"
```

---

### Task 5: Apply style rules to SBC retrieval note

**Files:**
- Modify: `papers/sampling-bias-corrected-retrieval/note.md`

**Step 1: Apply rules**

First-use terms to italicize (among others): *two-tower model*, *batch softmax*, *sampling bias*, *logQ correction*, *importance sampling*, *streaming frequency estimation*, *inter-arrival time*, *capacity factor* (if used), *approximate nearest neighbor*, *maximum inner product search*.

Bold key conclusions: the statement that logQ correction is the sole training-time change with zero inference cost.

Italicize any approximation caveats.

**Step 2: Commit**

```bash
git add papers/sampling-bias-corrected-retrieval/note.md
git commit -m "style: apply typographic rules to SBC retrieval note"
```

---

### Task 6: Apply style rules to DHEN note

**Files:**
- Modify: `papers/dhen-ranking/note.md`

**Step 1: Apply rules**

First-use terms to italicize (among others): *click-through rate*, *feature interaction*, *cross network*, *factorization machine*, *self-attention*, *hierarchical ensemble*, *non-overlapping information hypothesis*, *data parallel*, *model parallel*, *fully sharded data parallel*, *hybrid sharded data parallel*, *normalized entropy*.

Bold key conclusions: NE improvement numbers, the combinatorial $k^N$ argument statement.

Italicize caveats about the HSDP model-size ceiling.

**Step 2: Commit**

```bash
git add papers/dhen-ranking/note.md
git commit -m "style: apply typographic rules to DHEN note"
```

---

### Task 7: Final cross-check

**Step 1: Verify no double-italicized terms**

For each note, check that no term is italicized more than once (only first use). A quick grep for repeated italic patterns is sufficient.

**Step 2: Verify bold conclusions are not over-applied**

Skim each note — bold conclusions should appear roughly 2–5 times per note, not on every paragraph.

**Step 3: Commit any fixes**

```bash
git add concepts/ papers/
git commit -m "style: fix over/under-application of typographic rules"
```
