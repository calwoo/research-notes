# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository

This is a personal knowledge repository for agent-created notes, walkthroughs, and summaries of papers and research topics. Content spans machine learning, modern deep learning, and related fields.

**Style preference:** Always approach topics with a mathematical bent — favor rigorous definitions, formal notation, and derivations over high-level hand-waving, even for applied ML/DL topics.

## Directory Structure

The repository uses a category-first layout. Each topic gets its own subdirectory containing all three related files:

```
concepts/       ← explanations of ML/math concepts
  <topic>/
    note.md         ← the research note/summary
    exercises.md    ← problem set
    solutions.md    ← full answer key
    figures/        ← images downloaded from cited papers (optional)
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
```

**Naming convention:** The topic slug is the folder name. For a topic `attention-transformer` under `concepts`:
- `concepts/attention-transformer/note.md` — the research note
- `concepts/attention-transformer/exercises.md` — problem set
- `concepts/attention-transformer/solutions.md` — full answer key

**Exercise file structure** (every `exercises.md` must follow this order):
1. **Derivation problems** — mathematical proofs and re-derivations
2. **Conceptual questions** — intuition and reasoning questions
3. **Implementation sketches** — pseudocode or math-level algorithm sketches

## Notes Format

- Each note must begin with a table of contents listing all top-level sections and their subsections, immediately after the title.
- **For paper notes:** include a TL;DR table immediately after the author line and before the TOC. Columns: `| Dimension | Prior State | This Paper |`.
- When researching a topic, always include a references table at the end of the note with columns: "Reference Name", "Brief Summary", "Link to Reference".

### Obsidian TOC Anchor Rules (CRITICAL)

Notes are viewed in Obsidian. Broken anchors are silent — links just don't navigate. Follow these rules exactly:
- **Never put LaTeX (`$...$`) in a heading** — Obsidian renders math before generating anchors, stripping Unicode symbols unpredictably.
- **Never use em-dashes (`—`) in headings** — use a colon instead. GFM produces `--`; Obsidian collapses to `-`, breaking the link.
- Anchor generation: lowercase, strip non-alphanumeric/non-hyphen chars, replace spaces with hyphens, collapse consecutive hyphens to one.
- Example: `### 4.2 The Abelian Property` → anchor `#42-the-abelian-property`

## Project Agents

Specialized subagents are defined in `.claude/agents/`. Available agents:
- `note-writer` — researches and writes `note.md` following repo format
- `exercise-writer` — writes `exercises.md` + `solutions.md` from a finished note
- `image-extractor` — fetches figures from arXiv HTML (`ar5iv.org/html/{id}`) and embeds them
- `reference-finder` — finds high-quality references for a topic via web search

## Workflow

- Commit notes with descriptive messages explaining what was added or changed.
- Store plans in a `plans/` directory before executing them.
- Store any documentation in a `docs/` directory.
