# Folder Restructure Design

**Date:** 2026-03-07

## Problem

The current three-tree structure separates related files across top-level directories:

```
notes/concepts/neural-scaling-laws.md
exercises/concepts/neural-scaling-laws.md
solutions/concepts/neural-scaling-laws.md
```

This makes topic-level navigation awkward — working on one topic requires jumping between three distant paths.

## Goal

Colocate notes, exercises, and solutions by topic so all files for a given topic live together.

## New Structure

```
concepts/
  neural-scaling-laws/
    note.md
    exercises.md
    solutions.md
papers/
  <topic>/
    note.md
    exercises.md
    solutions.md
walkthroughs/
  <topic>/
    note.md
    exercises.md
    solutions.md
docs/
  plans/
CLAUDE.md
```

- Top-level `notes/`, `exercises/`, `solutions/` trees are removed.
- Categories (`concepts/`, `papers/`, `walkthroughs/`) become top-level.
- Each topic gets a subdirectory named by its slug.
- Inside each topic dir: `note.md`, `exercises.md`, `solutions.md`.

## File Naming Convention

| Role | Filename |
|------|----------|
| Main research note | `note.md` |
| Exercise set | `exercises.md` |
| Full solutions | `solutions.md` |

The folder name carries the topic identity. Files use generic role-based names.

## Migration

For the existing `neural-scaling-laws` topic:

```bash
mkdir -p concepts/neural-scaling-laws
git mv notes/concepts/neural-scaling-laws.md concepts/neural-scaling-laws/note.md
git mv exercises/concepts/neural-scaling-laws.md concepts/neural-scaling-laws/exercises.md
git mv solutions/concepts/neural-scaling-laws.md concepts/neural-scaling-laws/solutions.md
```

Remove now-empty directories and update CLAUDE.md.

## CLAUDE.md Changes

- Update directory structure diagram
- Update naming convention description (topic slug = folder name, not filename prefix)
- Update exercise file structure rule (same section order, file is now `exercises.md`)
- Update references to the three-tree structure throughout
