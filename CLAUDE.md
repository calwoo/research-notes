# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository

This is a personal knowledge repository for agent-created notes, walkthroughs, and summaries of papers and research topics. Content spans machine learning, modern deep learning, and related fields.

**Style preference:** Always approach topics with a mathematical bent — favor rigorous definitions, formal notation, and derivations over high-level hand-waving, even for applied ML/DL topics.

## Directory Structure

The repository uses three parallel directory trees, each with the same subdirectory types:

```
notes/
  papers/         ← summaries/analyses of specific papers
  concepts/       ← explanations of ML/math concepts
  walkthroughs/   ← step-by-step derivations or implementations
exercises/
  papers/
  concepts/
  walkthroughs/
solutions/
  papers/
  concepts/
  walkthroughs/
plans/            ← implementation plans before execution
docs/             ← documentation for new functionality
```

**Naming convention:** topic slugs are shared across all three trees. For a topic `attention-transformer`:
- `notes/papers/attention-transformer.md` — the note/summary
- `exercises/papers/attention-transformer.md` — problem set
- `solutions/papers/attention-transformer.md` — full answer key

**Exercise file structure** (every exercise file must follow this order):
1. **Derivation problems** — mathematical proofs and re-derivations
2. **Conceptual questions** — intuition and reasoning questions
3. **Implementation sketches** — pseudocode or math-level algorithm sketches

## Notes Format

- When researching a topic, always include a references table at the end of the note with columns: "Reference Name", "Brief Summary", "Link to Reference".

## Workflow

- Commit notes with descriptive messages explaining what was added or changed.
- Store plans in a `plans/` directory before executing them.
- Store any documentation in a `docs/` directory.
