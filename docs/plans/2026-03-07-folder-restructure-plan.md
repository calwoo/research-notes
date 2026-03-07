# Folder Restructure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure the repo so notes, exercises, and solutions are colocated under topic folders inside category directories.

**Architecture:** Replace the three-tree layout (`notes/`, `exercises/`, `solutions/`) with category-first directories (`concepts/`, `papers/`, `walkthroughs/`), each containing per-topic subdirectories with `note.md`, `exercises.md`, and `solutions.md`. Use `git mv` to preserve file history.

**Tech Stack:** Git, Markdown.

---

### Task 1: Move files into new colocated structure

**Files:**
- Move: `notes/concepts/neural-scaling-laws.md` → `concepts/neural-scaling-laws/note.md`
- Move: `exercises/concepts/neural-scaling-laws.md` → `concepts/neural-scaling-laws/exercises.md`
- Move: `solutions/concepts/neural-scaling-laws.md` → `concepts/neural-scaling-laws/solutions.md`

**Step 1: Create the new topic directory**

```bash
mkdir -p /Users/calvinwoo/Documents/notes/concepts/neural-scaling-laws
```

**Step 2: Move the files with git mv**

```bash
cd /Users/calvinwoo/Documents/notes
git mv notes/concepts/neural-scaling-laws.md concepts/neural-scaling-laws/note.md
git mv exercises/concepts/neural-scaling-laws.md concepts/neural-scaling-laws/exercises.md
git mv solutions/concepts/neural-scaling-laws.md concepts/neural-scaling-laws/solutions.md
```

**Step 3: Verify the moves**

```bash
git status
```

Expected: three renames shown (e.g. `renamed: notes/concepts/neural-scaling-laws.md -> concepts/neural-scaling-laws/note.md`), no untracked files.

**Step 4: Commit**

```bash
git commit -m "refactor: colocate notes, exercises, solutions under concepts/neural-scaling-laws/"
```

---

### Task 2: Remove old empty directories

**Step 1: Verify directories are empty**

```bash
find /Users/calvinwoo/Documents/notes/notes -type f
find /Users/calvinwoo/Documents/notes/exercises -type f
find /Users/calvinwoo/Documents/notes/solutions -type f
```

Expected: no output (all files have been moved).

**Step 2: Remove the old trees**

```bash
cd /Users/calvinwoo/Documents/notes
rm -rf notes exercises solutions
```

**Step 3: Stage the deletions and commit**

```bash
git add -A
git commit -m "refactor: remove old notes/, exercises/, solutions/ top-level trees"
```

---

### Task 3: Update CLAUDE.md with new structure

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Replace the directory structure section**

Update the `## Directory Structure` section to reflect the new layout:

```markdown
## Directory Structure

The repository uses a category-first layout. Each topic gets its own subdirectory containing all three related files:

\```
concepts/       ← explanations of ML/math concepts
  <topic>/
    note.md         ← the research note/summary
    exercises.md    ← problem set
    solutions.md    ← full answer key
papers/         ← summaries/analyses of specific papers
  <topic>/
    note.md
    exercises.md
    solutions.md
walkthroughs/   ← step-by-step derivations or implementations
  <topic>/
    note.md
    exercises.md
    solutions.md
docs/           ← documentation and design docs
  plans/        ← implementation plans before execution
\```
```

**Step 2: Update the naming convention description**

Replace the old naming convention block with:

```markdown
**Naming convention:** The topic slug is the folder name. For a topic `attention-transformer` under `concepts`:
- `concepts/attention-transformer/note.md` — the research note
- `concepts/attention-transformer/exercises.md` — problem set
- `concepts/attention-transformer/solutions.md` — full answer key
```

**Step 3: Verify the exercise file structure rule still reads correctly**

The exercise file structure rule (derivation → conceptual → implementation sketches) describes internal file organization and does not reference paths — it should need no change. Confirm it reads:

```markdown
**Exercise file structure** (every `exercises.md` must follow this order):
1. **Derivation problems** — mathematical proofs and re-derivations
2. **Conceptual questions** — intuition and reasoning questions
3. **Implementation sketches** — pseudocode or math-level algorithm sketches
```

Update the parenthetical if it still says "every exercise file" to say "every `exercises.md`".

**Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for new colocated folder structure"
```
