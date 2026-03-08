---
name: new-topic
description: Scaffold and research a new topic in the knowledge repo. Creates the design doc, implementation plan, and begins writing note.md using the note-writer agent.
---

# New Topic Skill

Invocation: `/new-topic`

When invoked, ask the user for:
1. **Topic name** — e.g., "attention mechanisms", "variational autoencoders"
2. **Category** — one of `concepts`, `papers`, `walkthroughs`
3. **Anchor papers or sources** — key references to ground the note (optional; will search if not provided)

Then execute the following steps in order:

## Step 1: Derive the Slug

Convert the topic name to a slug: lowercase, spaces→hyphens, strip punctuation.
Example: "Variational Autoencoders" → `variational-autoencoders`

## Step 2: Create the Design Doc

Create `docs/plans/YYYY-MM-DD-{slug}-design.md` (use today's date) following this template:

```markdown
# Design: {Topic Name} Concept Note

**Date:** YYYY-MM-DD
**Topic slug:** `{slug}`
**Category:** `{category}`

## Scope

[1–2 paragraphs describing what this note will cover and why]

## Files to Create

| File | Purpose |
|------|---------|
| `{category}/{slug}/note.md` | Main research note |
| `{category}/{slug}/exercises.md` | Problem set |
| `{category}/{slug}/solutions.md` | Full answer key |

## Note Structure

[Outline the planned sections with brief descriptions of content]

## Exercise Structure

1. **Derivation problems** — [list planned derivation problems]
2. **Conceptual questions** — [list planned conceptual questions]
3. **Implementation sketches** — [list planned implementation sketches]

## References

[List anchor papers and sources]
```

## Step 3: Create the Implementation Plan

Create `docs/plans/YYYY-MM-DD-{slug}-plan.md` following the standard plan format with tasks:
1. Write note.md sections 1–N
2. Review note for correctness and completeness
3. Write exercises.md
4. Write solutions.md
5. Final cross-check (TOC anchors, notation consistency, every exercise has a solution)

## Step 4: Create the Topic Directory

```bash
mkdir -p {category}/{slug}
```

## Step 5: Research and Write the Note

Use the `note-writer` subagent to research the topic and write `{category}/{slug}/note.md`. Pass it the design doc and the list of anchor sources.

## Step 6: Commit

After the note is written and reviewed:

```bash
git add docs/plans/ {category}/{slug}/
git commit -m "feat: scaffold {slug} — design doc, plan, and initial note"
```
