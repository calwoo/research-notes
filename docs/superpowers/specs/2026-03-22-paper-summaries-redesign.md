# Paper Summaries Redesign — Design Spec

**Date:** 2026-03-22
**Status:** Approved

---

## 1. Problem Statement

Paper summary notes currently have no cross-paper linking. Notes feel isolated from one another and from the broader concept graph. The two concrete gaps are:

1. No explicit "builds on / extended by" relationships between papers
2. No links from paper notes to the concept notes that explain the underlying math

Additionally, the TL;DR table lacks a dedicated column for the primary quantitative result, which is often the most important thing to surface quickly.

Finally, the folder-per-paper convention creates unnecessary nesting for single papers — a flat file is simpler and less cluttered.

---

## 2. Scope

- Update `CLAUDE.md` with revised conventions
- Retrofit all existing paper notes to match the new format
- No changes to `concepts/`, `walkthroughs/`, or `curricula/` conventions

---

## 3. File Organization

### Current

```
papers/
  <topic>/
    note.md
    exercises.md   (optional)
    solutions.md   (optional)
```

### New

| Scenario | Structure |
|---|---|
| Single paper | `papers/<paper-slug>.md` |
| Multi-paper topic cluster | `papers/<topic>/` folder, one named `.md` file per paper |
| Exercises | `papers/<paper-slug>-exercises.md` (flat) or alongside in folder |
| Solutions | `papers/<paper-slug>-solutions.md` (flat) or alongside in folder |

**Rule:** A folder is created only when multiple papers belong to the same topic cluster (e.g. `papers/generative-recommenders/` containing `hstu.md` and `wukong.md`). A single standalone paper is always a flat file at the `papers/` root.

**Exercises and solutions** remain strictly opt-in — created only when explicitly requested.

### Existing papers to migrate

| Current path | New path |
|---|---|
| `papers/dhen-ranking/note.md` | `papers/dhen-ranking.md` |
| `papers/dhen-ranking/exercises.md` | `papers/dhen-ranking-exercises.md` |
| `papers/dhen-ranking/solutions.md` | `papers/dhen-ranking-solutions.md` |
| `papers/sampling-bias-corrected-retrieval/note.md` | `papers/sampling-bias-corrected-retrieval.md` |
| `papers/sampling-bias-corrected-retrieval/exercises.md` | `papers/sampling-bias-corrected-retrieval-exercises.md` |
| `papers/sampling-bias-corrected-retrieval/solutions.md` | `papers/sampling-bias-corrected-retrieval-solutions.md` |
| `papers/generative-recommenders/hstu.md` | `papers/generative-recommenders/hstu.md` (keep — multi-paper cluster) |
| `papers/generative-recommenders/wukong.md` | `papers/generative-recommenders/wukong.md` (keep) |
| `papers/generative-recommenders/exercises.md` | `papers/generative-recommenders/exercises.md` (keep) |
| `papers/generative-recommenders/solutions.md` | `papers/generative-recommenders/solutions.md` (keep) |

---

## 4. Note Header Format

The header of every paper note follows this order:

```markdown
# Paper Title

Authors. Venue. Year.

| Dimension | Prior State | This Paper | Key Result |
|---|---|---|---|
| Row 1 | ... | ... | ... |

## Relations

**Builds on:** [[papers/foo|Foo]], [[papers/bar|Bar]]
**Extended by:** [[papers/baz|Baz]] *(no note yet)*
**Concepts used:** [[concepts/attention-mechanisms/standard-attention|Standard Attention]]

## Table of Contents

- [[#Section 1|Section 1]]
  - [[#Subsection|Subsection]]
```

### TL;DR Table

- **Columns:** `Dimension | Prior State | This Paper | Key Result`
- `Key Result` contains the primary quantitative takeaway for that dimension row (e.g. "+12.4% engagement", "5.3x faster than FlashAttention2")
- Rows should reflect the most important dimensions of novelty — not every detail

### Relations Section

- Appears after TL;DR, before TOC
- Three optional sub-fields (omit any that are empty):
  - `**Builds on:**` — papers this work directly extends or critiques
  - `**Extended by:**` — known follow-up papers
  - `**Concepts used:**` — links to `concepts/` entries explaining the underlying math
- Placeholder links (no note yet) are marked with `*(no note yet)*`
- Use Obsidian wikilink syntax: `[[path/to/file|Display Name]]`

---

## 5. Inline Linking

Throughout the body of a paper note, wikilink to:
- Related paper notes when a specific paper is discussed
- Concept notes when an underlying method or definition is referenced

The Relations block is a quick-scan index. Inline links are contextual — they appear where the intellectual connection is most relevant to the reader, not just at the top.

---

## 6. Rollout Steps

1. Update `CLAUDE.md` — revise the Papers section with new file organization rules and note header format
2. Flatten single-paper folders: move `note.md` to `papers/<slug>.md`, move exercises/solutions to flat equivalents
3. Retrofit each paper note:
   - Add `Key Result` column to TL;DR table
   - Add `## Relations` section (with accurate links where known, placeholders otherwise)
   - Add inline wikilinks to related papers and concepts throughout the body
4. Verify Obsidian graph picks up all new wikilinks

---

## 7. CLAUDE.md Changes Required

The `papers/` section of `CLAUDE.md` needs to:
- Describe the flat-vs-folder rule (single paper = flat file, multi-paper cluster = folder)
- Specify the four-column TL;DR table format
- Define the `## Relations` section format, including placeholder syntax
- Clarify that inline wikilinks to related papers and concepts should be used throughout the body
