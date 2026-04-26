---
name: todo-check
description: Use when the user invokes /todo-check to find TODO: markers in notes and resolve each one — either by editing the main text inline or by inserting an Obsidian callout, whichever fits best.
---

# todo-check

Scan all Markdown notes in the current repo for `TODO:` markers and resolve each one. **You choose the resolution strategy** — inline edit or callout — based on what serves the note best.

## Workflow

### 1. Discover all TODOs

```bash
grep -rn "TODO:" --include="*.md" .
```

Collect every match: file path, line number, and the full TODO text.

### 2. For each TODO — understand context

Read the surrounding section (at minimum the enclosing `##` block) so you understand:
- What topic the note covers at that point
- What the TODO is specifically asking for (explain, cite, add example, clarify, etc.)

### 3. Resolve the TODO

Use whatever tools are needed to produce a complete, correct answer:
- `WebSearch` / `WebFetch` for facts, citations, or missing derivations
- Direct reasoning for explanations or clarifications
- The note itself for continuity of notation and style

The answer must match the repo's mathematical style: rigorous definitions, formal notation, no hand-waving.

### 4. Choose the resolution strategy

For each TODO, decide between two approaches:

**A. Inline edit** — rewrite or extend the surrounding prose directly. Use this when:
- The TODO points out an error, ambiguity, or gap that belongs in the main text (e.g. a wrong calculation, a missing word, a confusing sentence)
- The answer is short and reads naturally as part of the flow
- Adding a callout would interrupt rather than supplement

**B. Callout (infobox)** — insert a supplementary block without touching the main narrative. Use this when:
- The answer is a digression, elaboration, or side explanation that would bloat the prose
- The TODO asks for an example, caveat, or background context that sits beside the text
- The content is long enough to warrant its own visual block

Map callout-style TODOs to the repo's established taxonomy:

| TODO intent | Callout type |
|---|---|
| Explain a concept or term | `[!NOTE]` |
| Background / historical context | `[!INFO]` |
| Implementation hint or practical advice | `[!TIP]` |
| Worked example or concrete instantiation | `[!EXAMPLE]` |
| Caveat, edge case, or condition where something breaks | `[!WARNING]` |
| Common misconception or hard failure mode | `[!DANGER]` |
| Open problem or unresolved question | `[!QUESTION]` |

For long digressions use a collapsible callout (`[!TYPE]-`).

### 5. Replace in the file

Remove the `TODO: ...` text, then apply the chosen strategy:

- **Inline edit:** modify the surrounding sentence(s) directly, preserving the note's voice and notation
- **Callout:** insert the block immediately below the paragraph or list item where the TODO appeared:

```
> [!TYPE] Optional descriptive title
> Content here. Use LaTeX for math. Write in the note's own voice and notation.
```

Do **not** leave a blank `TODO:` stub or a placeholder — every replacement must contain the actual resolved content.

### 6. Report

After all replacements, output a compact table:

| File | Line | TODO summary | Strategy | Callout type / edit description |
|---|---|---|---|---|

List every TODO processed, including any you skipped (and why).

## Rules

- **Never delete a TODO without replacing it with resolved content.** If you cannot resolve a TODO (insufficient information, out-of-scope), leave it in place and explain why in your report.
- **Preserve surrounding prose.** Only touch the TODO itself and the immediately adjacent whitespace.
- **Match the note's notation exactly** — same variable names, same LaTeX macros, same section voice.
- **One callout per TODO.** Do not split one TODO into multiple callouts.
- **Do not re-open resolved items.** If a TODO is actually already answered nearby, remove the marker and note it as "already resolved" in your report.
