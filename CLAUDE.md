# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository

This is a personal knowledge repository for agent-created notes, summaries of papers, and active research notes. Content spans machine learning, modern deep learning, and related fields.

**Style preference:** Always approach topics with a mathematical bent — favor rigorous definitions, formal notation, and derivations over high-level hand-waving, even for applied ML/DL topics.

## Directory Structure

The repository uses a category-first layout. Each topic gets its own subdirectory:

```
concepts/       ← explanations of ML/math concepts
  <topic>/
    <descriptive-name>.md   ← one or more note files, named meaningfully
    figures/                ← images downloaded from cited papers (optional)
papers/         ← summaries/analyses of specific papers
  <paper-slug>.md                    ← single standalone paper (flat file)
  <topic>/                           ← multi-paper topic cluster (folder only when ≥2 papers)
    <paper-slug>.md                  ← one named file per paper in the cluster
    figures/                         ← figures for papers in this cluster
  figures/
    <paper-slug>/                    ← figures for a flat single-paper file
notes/          ← unstructured quick notes, scratchpad entries, or draft ideas
  <slug>.md     ← flat files, no subdirectory required
curricula/      ← structured multi-week learning curricula for a field or subfield
  <topic>/
    curriculum.md   ← week-by-week checklist of materials, concepts, learning goals, and milestones
research/       ← active research notes: synthesis-in-progress across sources
  <thread-slug>.md   ← one flat file per research thread
  figures/           ← figures pulled from source papers (optional)
docs/           ← documentation and design docs
  plans/        ← implementation plans before execution
```

**Naming convention for `concepts/`:** The topic slug is the folder name. Note files inside a concept folder must always be descriptively named — never use `note.md`. Even a single-file topic should have a filename that reflects its content.

Example:
- `concepts/attention-mechanisms/standard-attention.md` — softmax attention
- `concepts/attention-mechanisms/linear-attention.md` — linear attention variants
- `concepts/attention-mechanisms/history.md` — historical development

**Naming convention for `papers/`:** A folder under `papers/` is created only when two or more papers belong to the same topic cluster. A single standalone paper is always a flat `.md` file at the `papers/` root.
- If a cluster later shrinks to one paper, leave the folder in place to avoid breaking wikilinks. A one-paper folder is an accepted exception.

**Naming convention for `notes/`:** Flat directory — one `.md` file per quick note directly under `notes/`. No subdirectories, no required structure. Use for drafts, scratchpad entries, or anything that doesn't yet fit a structured category.

**Naming convention for `research/`:** Flat directory — one `.md` file per research thread directly under `research/`, named by thread slug (e.g. `research/diffusion-posterior-collapse.md`). No subdirectories. Figures go in the shared `research/figures/` folder.

**Exercises and solutions are inline.** Do not create separate `exercises.md` or `solutions.md` files. Exercises are distributed throughout the note — place them immediately after the section whose content they test, so each exercise appears after all its prerequisites. Do not batch all exercises at the end of the note.

Each exercise is wrapped in a `[!QUESTION]` callout (yellow, non-collapsible). Inside the callout:
- An italic 1–2 sentence preamble stating what the problem establishes
- A `> **Prerequisites:**` blockquote linking to the relevant note section via Obsidian wikilink
- The problem statement

The solution immediately follows as a `[!TIP]-` collapsible callout (see Obsidian Callouts section). Example:

```
> [!QUESTION] Exercise N: Short Title
> *Preamble sentence describing what this problem establishes.*
>
> > **Prerequisites:** [[#Section Title|Section Title]]
>
> Problem statement here.

> [!TIP]- Solution to Exercise N
> **Key insight:** ...
>
> **Sketch:** ...
```

Problems are numbered continuously 1–N throughout the note. Solutions use **Key insight** + **Sketch** format (not full worked derivations).

## Notes Format

- For notes with more than a few sections, include a table of contents listing all top-level sections and their subsections, immediately after the title.
- **For paper notes:** include a TL;DR table immediately after the author line. Columns: `| Dimension | Prior State | This Paper | Key Result |`. The `Key Result` column holds the primary quantitative takeaway for that dimension (e.g. "+12.4% engagement", "5.3x faster than FlashAttention2"). Rows should reflect the most important dimensions of novelty.
- **For paper notes:** include a `## Relations` section immediately after the TL;DR table and before the TOC. See **Paper note header order** below for the full header sequence. The `## Relations` heading itself is omitted from the TOC. If all sub-fields are empty, omit the heading entirely.

  The Relations section has three optional sub-fields (omit any that have no entries):
  - `**Builds on:**` — papers this work directly extends or critiques
  - `**Extended by:**` — known follow-up papers
  - `**Concepts used:**` — links to `concepts/` entries explaining the underlying math

  Example:

  ````
  ## Relations

  **Builds on:** [[papers/paper-slug|Display Name]], [[papers/topic/paper-slug|Display Name]]
  **Extended by:** [[papers/follow-up|Display Name]] *(no note yet)*
  **Concepts used:** [[concepts/topic/filename|Display Name]]
  ````

  Placeholder links for papers or concepts without notes yet are marked with `*(no note yet)*`.
- **For research notes:** see the dedicated **Research Notes Format** section below for structure and workflow rules.
- When researching a topic, always include a references table at the end of the note with columns: "Reference Name", "Brief Summary", "Link to Reference".
- When a note cites sources with figures worth embedding, use the `image-extractor` agent to fetch and embed them inline at relevant locations.
- **Image sizing:** Never embed images at full native resolution. Use `<img>` tags with an explicit `width` attribute so figures render at a readable size in Obsidian. Recommended widths: single plots or diagrams → `width="420"–"500"`; two-panel figures → `width="550"–"600"`; three-or-more-panel figures → `width="650"–"700"`. Place the italic caption on its own line immediately below the `<img>` tag. Example:
  ```markdown
  <img src="figures/author2025-fig1-description.png" width="500" alt="brief alt text">

  *Figure 1 (Author et al., 2025): Caption describing the figure in context.*
  ```
- **Paper note header order:** For paper notes specifically, the header order is: Title → Authors/venue line → TL;DR table → Relations section → Table of Contents. This supersedes the general "TOC immediately after the title" rule for paper notes only.
- **Inline wikilinks** to related paper notes and concept notes should appear throughout the body — link on the first mention of a related paper or concept within each major section (`##` heading level). Do not re-link the same target within the same section.

## Research Notes Format

Research notes live in `research/<thread-slug>.md`. They are synthesis-in-progress documents: the goal is to understand and connect results from multiple sources toward new insight, not to teach settled material.

### Structure

Research notes have **one fixed section** and otherwise evolve freely:

```
# Title

## Sources

| Source | Type | Key Contribution | Link |
|--------|------|-----------------|------|
| ...    | paper/textbook/talk | ... | ... |

## <Section — user-defined, free-form>

...
```

The `## Sources` table is always present and always first after the title. It is a live registry: add rows as new sources are incorporated. Columns: **Source** (author + short title), **Type** (paper / textbook / blog / talk), **Key Contribution** (one-line summary of what this source adds to the thread), **Link**.

All other sections are free-form and driven by the research thread. Common patterns (use as needed, not as a template):
- `## Context and Motivation` — what problem or gap is this thread addressing
- `## Key Results` — theorem/result summaries drawn from sources, with attribution
- `## Synthesis` — cross-source connections, tensions, and emerging insights
- `## Open Questions` — unresolved points, conjectures, things to investigate
- `## Scratchpad` — rough working space, back-of-envelope arguments, fragments

Use `[!QUESTION]` callouts (non-collapsible) for conjectures and open questions inline. Use `[!WARNING]` for tensions or contradictions between sources. Do not add exercises.

### Workflow: injecting new content from a source

**Before writing up material from a new source, always ask the user probing questions about it.** Use `AskUserQuestion` to surface 3–5 targeted questions that clarify:
- Which specific results or ideas from the source are most relevant to this thread
- What the user finds surprising, unclear, or most worth dwelling on
- How the source connects to or tensions with material already in the note

Only after the user answers should you draft new content. This is a hard rule — never silently ingest a source and write it up without interrogating the user first.

### Promotion

Research notes are never automatically promoted to concept notes. The user decides when a thread has crystallized enough to become a `concepts/` entry. Do not suggest promotion unless asked.

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

### Obsidian Callouts

Use Obsidian callouts often to inject supplementary information without breaking the main narrative flow. Callouts are ideal for asides, worked examples, caveats, and historical context that would disrupt a derivation if placed inline.

Syntax:
```
> [!TYPE] Optional custom title
> Content here.

> [!TYPE]- Collapsible callout (collapsed by default)
> Content here.
```

Use these types consistently:

| Type | Color | When to use |
|------|-------|-------------|
| `NOTE` | blue | Clarifications, definitions that don't fit inline |
| `INFO` | blue | Background context, historical notes |
| `TIP` | green | Practical advice, implementation hints; **always use for exercise solutions** |
| `EXAMPLE` | purple | Worked examples, concrete instantiations |
| `WARNING` | orange | Caveats, conditions where a result breaks down |
| `DANGER` | red | Common misconceptions, hard failure modes |
| `QUESTION` | yellow | Open problems, unresolved debates |
| `QUOTE` | grey | Verbatim excerpts worth preserving |

**Guidance:** Prefer callouts over parenthetical asides or footnotes. Use collapsible callouts (`-`) for lengthy digressions or full worked examples that only some readers will want.

**Exercise solutions** must always use `[!TIP]-` (collapsible, green). Title format: `Solution to Exercise N`. Example:
```
> [!TIP]- Solution to Exercise 3
> Solution content here.
```

### Emoji Usage

Use emojis throughout notes to add visual color to the exposition. Sprinkle them at section headings, before key definitions, and to signal tone (e.g. ⚠️ for warnings, 💡 for insights, 📐 for derivations, 🔑 for key results).

### References and Hyperlinks

When citing a reference inline or in the references table, hyperlink the text wherever a URL is available. Use `[Reference Name](url)` format. Prefer hyperlinking the paper title or author name over bare URLs.

### Obsidian TOC Link Rules (CRITICAL)

Notes are viewed in Obsidian. Use Obsidian's wikilink syntax for all TOC links — standard markdown `[text](#slug)` does NOT navigate in Obsidian.

**Correct format:** `[[#Exact Heading Text|Display Text]]`
- The text after `#` must be the **exact literal heading text** as it appears in the document (strip the leading `##`/`###` only)
- Example: heading `### 4.2 The Abelian Property` → TOC entry `[[#4.2 The Abelian Property|4.2 The Abelian Property]]`
- Subsections are indented with spaces in the TOC list

**Cross-file wikilinks** (for Relations blocks and inline body links) use vault-relative paths without the `.md` extension:
- Paper links: `[[papers/paper-slug|Display Name]]` (flat) or `[[papers/topic/paper-slug|Display Name]]` (cluster)
- Concept links: `[[concepts/topic/filename|Display Name]]`
- Research links: `[[research/thread-slug|Display Name]]`

This differs from intra-document TOC links (`[[#Exact Heading Text|Display Text]]`). Do not mix the two forms.

### Mermaid Diagram Conventions

- **Always use Python for pseudocode.** Never use generic pseudocode syntax or algorithm-block notation — write executable (or near-executable) Python instead.
- **Always prefer Mermaid diagrams over ASCII art.** Never use ASCII diagrams (e.g., box-and-arrow art made with `─`, `│`, `→`, etc.) — use a `mermaid` fenced code block instead.
- **Line breaks in node labels:** use `<br/>`, not `\n`. Obsidian's Mermaid renderer does not interpret `\n` as a newline inside node label strings.
- **No `&` multi-edge shorthand:** `A & B --> C` is not supported in Obsidian's bundled Mermaid — write individual edges instead (`A --> C` and `B --> C` on separate lines).
- **No em-dashes in edge labels:** `-->|yes — label|` causes parse errors — use plain text instead (`-->|yes|`). Avoid colons in edge labels too.
- **No unicode in diamond labels:** special characters like `≠`, `∅`, `→` inside `{...}` diamond nodes cause parse errors — use plain ASCII words instead (e.g., `{"gaps remain?"}` not `{"G ≠ ∅?"}`). Unicode is safe inside regular `["..."]` node labels.
  - Correct: `kI["k_s^I ∈ ℝ^d<br/>indexer key"]`
  - Wrong: `kI["k_s^I ∈ ℝ^d\nindexer key"]`

## Reading PDFs

When a URL points to a PDF and `WebFetch` returns binary content, download and convert with pandoc first:

```bash
curl -sL <url> -o /tmp/paper.pdf
pandoc /tmp/paper.pdf -t markdown -o /tmp/paper.md
```

Then read `/tmp/paper.md`. This applies to arXiv PDFs, textbooks, and any other PDF source.

## Project Agents

Specialized subagents are defined in `.claude/agents/`. Available agents:
- `note-writer` — researches and writes note files following repo format
- `image-extractor` — extracts figures from arXiv source tarballs (`arxiv.org/src/{id}`) and embeds them at scaled sizes
- `reference-finder` — finds high-quality references for a topic via web search

**Skills** (invoked via `/skill-name`):
- `new-topic` — scaffolds and researches a new topic; creates design doc, implementation plan, and starts writing

## Communication Preferences

**ALWAYS use the `AskUserQuestion` tool when asking the user anything** — whether clarifying intent, choosing between approaches, or gathering requirements. Never ask questions as plain text. Every question must use the interactive multiple-choice interface. This is a hard requirement with no exceptions.

## Workflow

- **After any document is updated, immediately create a commit and push it to the remote repo.** Do not wait to be asked. This applies to every note, plan, or doc change — no exceptions.
- Commit notes with descriptive messages explaining what was added or changed.
  - Format: `feat: add <topic> concept note`, `docs: update <topic> references`, `fix: correct <section> in <topic>`
- Store plans in `docs/plans/` before executing them.
- Store any documentation in a `docs/` directory.
