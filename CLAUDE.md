# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository

This is a personal knowledge repository for agent-created notes, walkthroughs, and summaries of papers and research topics. Content spans machine learning, modern deep learning, and related fields.

**Style preference:** Always approach topics with a mathematical bent — favor rigorous definitions, formal notation, and derivations over high-level hand-waving, even for applied ML/DL topics.

## Directory Structure

The repository uses a category-first layout. Each topic gets its own subdirectory:

```
concepts/       ← explanations of ML/math concepts
  <topic>/
    <descriptive-name>.md   ← one or more note files, named meaningfully
    exercises.md            ← problem set (optional, generate only when explicitly asked)
    solutions.md            ← full answer key (optional, generate only when explicitly asked)
    figures/                ← images downloaded from cited papers (optional)
papers/         ← summaries/analyses of specific papers
  <paper-slug>.md                    ← single standalone paper (flat file)
  <topic>/                           ← multi-paper topic cluster (folder only when ≥2 papers)
    <paper-slug>.md                  ← one named file per paper in the cluster
    exercises.md                     ← optional, generate only when explicitly asked
    solutions.md                     ← optional, generate only when explicitly asked
    figures/                         ← figures for papers in this cluster
  figures/
    <paper-slug>/                    ← figures for a flat single-paper file
walkthroughs/   ← step-by-step derivations or implementations
  <topic>/
    note.md
    exercises.md    ← optional, generate only when explicitly asked
    solutions.md    ← optional, generate only when explicitly asked
curricula/      ← structured multi-week learning curricula for a field or subfield
  <topic>/
    curriculum.md   ← week-by-week checklist of materials, concepts, learning goals, and milestones
    exercises.md    ← optional, generate only when explicitly asked
    solutions.md    ← optional, generate only when explicitly asked
docs/           ← documentation and design docs
  plans/        ← implementation plans before execution
```

**Naming convention for `concepts/`:** The topic slug is the folder name. Note files inside a concept folder should be named to reflect their content — use `note.md` only when a single file suffices; split into multiple descriptively-named files when a topic is broad enough to warrant it. If `exercises.md` and `solutions.md` exist, they must use those exact names — but only create them when explicitly requested.

Example for a multi-file concept topic `attention-mechanisms`:
- `concepts/attention-mechanisms/standard-attention.md` — softmax attention
- `concepts/attention-mechanisms/linear-attention.md` — linear attention variants
- `concepts/attention-mechanisms/history.md` — historical development
- `concepts/attention-mechanisms/exercises.md` — problem set
- `concepts/attention-mechanisms/solutions.md` — full answer key

**Naming convention for `papers/`:** A folder under `papers/` is created only when two or more papers belong to the same topic cluster. A single standalone paper is always a flat `.md` file at the `papers/` root.
- If a cluster later shrinks to one paper, leave the folder in place to avoid breaking wikilinks. A one-paper folder is an accepted exception.
- Exercises and solutions for flat paper files use the names `<paper-slug>-exercises.md` and `<paper-slug>-solutions.md`. Inside cluster folders, use the exact names `exercises.md` and `solutions.md`.

**Naming convention for `walkthroughs/`:** Single `note.md` is the default. Split only if the topic genuinely has distinct subtopics.

**Exercise file structure** (when `exercises.md` is generated, it must follow this order):
1. **Mathematical Development** — derivations, proofs, limit arguments, and mathematically sharp conceptual results (16–18 problems)
2. **Algorithmic Applications** — pseudocode sketches, numerical implementation, complexity analysis (5–7 problems)

Problems are numbered continuously 1–N across both sections. Each problem requires:
- An italic 1–2 sentence preamble stating what the problem establishes
- A `> **Prerequisites:**` blockquote linking to the relevant note section via Obsidian wikilink

Solutions use **Key insight** + **Sketch** format (not full worked derivations).

## Notes Format

- Each note must begin with a table of contents listing all top-level sections and their subsections, immediately after the title.
- **For paper notes:** include a TL;DR table immediately after the author line and before the TOC. Columns: `| Dimension | Prior State | This Paper | Key Result |`. The `Key Result` column holds the primary quantitative takeaway for that dimension (e.g. "+12.4% engagement", "5.3x faster than FlashAttention2"). Rows should reflect the most important dimensions of novelty.
- **For paper notes:** include a `## Relations` section immediately after the TL;DR table and before the TOC. The TOC rule ("immediately after the title") is superseded for paper notes by this header order: Title → Authors/venue → TL;DR → Relations → TOC. The `## Relations` heading itself is omitted from the TOC. If all sub-fields are empty, omit the heading entirely.

  The Relations section has three optional sub-fields (omit any that have no entries):
  - `**Builds on:**` — papers this work directly extends or critiques
  - `**Extended by:**` — known follow-up papers
  - `**Concepts used:**` — links to `concepts/` entries explaining the underlying math

  Example:

  ````
  ## Relations

  **Builds on:** [[papers/paper-slug|Display Name]]
  **Extended by:** [[papers/follow-up|Display Name]] *(no note yet)*
  **Concepts used:** [[concepts/topic/filename|Display Name]]
  ````

  Placeholder links for papers or concepts without notes yet are marked with `*(no note yet)*`.
- When researching a topic, always include a references table at the end of the note with columns: "Reference Name", "Brief Summary", "Link to Reference".
- After writing a note, fetch figures and diagrams from the cited references and embed them inline at relevant locations to improve exposition. Use the `image-extractor` agent for this.
- **Paper note header order:** For paper notes specifically, the header order is: Title → Authors/venue line → TL;DR table → Relations section → Table of Contents. This supersedes the general "TOC immediately after the title" rule for paper notes only.
- **Inline wikilinks** to related paper notes and concept notes should appear throughout the body — link on the first mention of a related paper or concept within each major section (`##` heading level). Do not re-link the same target within the same section.

### Typographic Style Rules

Apply these consistently in all note files:

| Element | Style | Example |
|---------|-------|---------|
| First use of a technical term in prose | *italics* | the *capacity factor* controls overflow |
| Formal definition / proposition / remark label | **bold** | `**Definition (Soft MoE Output).**` |
| Key conclusion — main quantitative takeaway of a derivation | **bold** | `**Both N and D should scale as √C.**` |
| Counterintuitive result | *italics* with inline signal | *Surprisingly,* linear alone outperforms DCN... |
| Warning or caveat | *italics* | *This bound only holds for $T \to \infty$.* |

Do NOT italicize terms after their first use. Do NOT bold entire sentences except for genuine key conclusions.

### Emoji Usage

Use emojis throughout notes to add visual color to the exposition. Sprinkle them at section headings, before key definitions, and to signal tone (e.g. ⚠️ for warnings, 💡 for insights, 📐 for derivations, 🔑 for key results). Do not overuse — one per paragraph at most.

### References and Hyperlinks

When citing a reference inline or in the references table, hyperlink the text wherever a URL is available. Use `[Reference Name](url)` format. Prefer hyperlinking the paper title or author name over bare URLs.

### Obsidian TOC Link Rules (CRITICAL)

Notes are viewed in Obsidian. Use Obsidian's wikilink syntax for all TOC links — standard markdown `[text](#slug)` does NOT navigate in Obsidian.

**Correct format:** `[[#Exact Heading Text|Display Text]]`
- The text after `#` must be the **exact literal heading text** as it appears in the document (strip the leading `##`/`###` only)
- Example: heading `### 4.2 The Abelian Property` → TOC entry `[[#4.2 The Abelian Property|4.2 The Abelian Property]]`
- Subsections are indented with spaces in the TOC list

**Never put LaTeX (`$...$`) in headings** — it makes the literal heading text awkward in wikilinks and renders unpredictably.
**Never use em-dashes (`—`) in headings** — use a colon instead.

**Cross-file wikilinks** (for Relations blocks and inline body links) use vault-relative paths without the `.md` extension:
- Paper links: `[[papers/paper-slug|Display Name]]` (flat) or `[[papers/topic/paper-slug|Display Name]]` (cluster)
- Concept links: `[[concepts/topic/filename|Display Name]]`

This differs from intra-document TOC links (`[[#Exact Heading Text|Display Text]]`). Do not mix the two forms.

## Project Agents

Specialized subagents are defined in `.claude/agents/`. Available agents:
- `note-writer` — researches and writes note files following repo format
- `exercise-writer` — writes `exercises.md` + `solutions.md` from a finished note
- `image-extractor` — fetches figures from arXiv HTML (`ar5iv.org/html/{id}`) and embeds them
- `reference-finder` — finds high-quality references for a topic via web search

**Skills** (invoked via `/skill-name`):
- `new-topic` — scaffolds and researches a new topic; creates design doc, implementation plan, and starts writing

## Communication Preferences

**ALWAYS use the `AskUserQuestion` tool when asking the user anything** — whether clarifying intent, choosing between approaches, or gathering requirements. Never ask questions as plain text. Every question must use the interactive multiple-choice interface. This is a hard requirement with no exceptions.

## Workflow

- Commit notes with descriptive messages explaining what was added or changed.
  - Format: `feat: add <topic> concept note`, `docs: update <topic> references`, `fix: correct <section> in <topic>`
- Store plans in `docs/plans/` before executing them.
- Store any documentation in a `docs/` directory.
