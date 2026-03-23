# Paper Summaries Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure paper summary notes to add cross-paper linking (Relations section), a four-column TL;DR table, and flat-file organization for single-paper topics.

**Architecture:** Update CLAUDE.md first to establish new conventions, then migrate existing folder-per-paper notes to flat files (with figures moved accordingly), then retrofit every paper note with the Relations block, Key Result column, and inline wikilinks.

**Spec:** `docs/superpowers/specs/2026-03-22-paper-summaries-redesign.md`

---

## Files Touched

| Action | Path |
|---|---|
| Modify | `CLAUDE.md` |
| Move + modify | `papers/dhen-ranking/note.md` → `papers/dhen-ranking.md` |
| Move | `papers/dhen-ranking/exercises.md` → `papers/dhen-ranking-exercises.md` |
| Move | `papers/dhen-ranking/solutions.md` → `papers/dhen-ranking-solutions.md` |
| Move (dir) | `papers/dhen-ranking/figures/` → `papers/figures/dhen-ranking/` |
| Delete | `papers/dhen-ranking/` (empty after moves) |
| Move + modify | `papers/sampling-bias-corrected-retrieval/note.md` → `papers/sampling-bias-corrected-retrieval.md` |
| Move | `papers/sampling-bias-corrected-retrieval/exercises.md` → `papers/sampling-bias-corrected-retrieval-exercises.md` |
| Move | `papers/sampling-bias-corrected-retrieval/solutions.md` → `papers/sampling-bias-corrected-retrieval-solutions.md` |
| Delete | `papers/sampling-bias-corrected-retrieval/` (empty after moves) |
| Modify | `papers/generative-recommenders/hstu.md` |
| Modify | `papers/generative-recommenders/wukong.md` |
| No action | `papers/generative-recommenders/exercises.md` (keep in place) |
| No action | `papers/generative-recommenders/solutions.md` (keep in place) |

---

## Task 1: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

The goal is to update the `papers/` section so agents follow the new conventions from the start. This must happen before any note work.

- [ ] **Step 1: Read the current papers section of CLAUDE.md**

  Open `CLAUDE.md` and locate the `papers/` block, the TL;DR description, and the Notes Format section. Understand exactly what will change.

- [ ] **Step 2: Update the papers/ directory entry**

  Replace the current papers block:
  ```
  papers/         ← summaries/analyses of specific papers
    <topic>/
      note.md         ← main paper summary (single file is fine here)
      exercises.md    ← optional, generate only when explicitly asked
      solutions.md    ← optional, generate only when explicitly asked
  ```

  With:
  ```
  papers/         ← summaries/analyses of specific papers
    <paper-slug>.md                    ← single standalone paper (flat file)
    <topic>/                           ← multi-paper topic cluster (folder only when ≥2 papers)
      <paper-slug>.md                  ← one named file per paper in the cluster
      exercises.md                     ← optional, generate only when explicitly asked
      solutions.md                     ← optional, generate only when explicitly asked
      figures/                         ← figures for papers in this cluster
    figures/
      <paper-slug>/                    ← figures for a flat single-paper file
  ```

  Add the following rules after the block:
  - A folder under `papers/` is created only when two or more papers belong to the same topic cluster. A single standalone paper is always a flat `.md` file at the `papers/` root.
  - If a cluster later shrinks to one paper, leave the folder in place to avoid breaking wikilinks. A one-paper folder is an accepted exception.
  - Exercises and solutions for flat paper files use the names `<paper-slug>-exercises.md` and `<paper-slug>-solutions.md`. Inside cluster folders, use the exact names `exercises.md` and `solutions.md`.

- [ ] **Step 3: Update the TL;DR table description**

  Find the line:
  ```
  **For paper notes:** include a TL;DR table immediately after the author line and before the TOC. Columns: `| Dimension | Prior State | This Paper |`.
  ```

  Replace with:
  ```
  **For paper notes:** include a TL;DR table immediately after the author line and before the TOC. Columns: `| Dimension | Prior State | This Paper | Key Result |`. The `Key Result` column holds the primary quantitative takeaway for that dimension (e.g. "+12.4% engagement", "5.3x faster than FlashAttention2"). Rows should reflect the most important dimensions of novelty.
  ```

- [ ] **Step 4: Add the Relations section rule**

  After the TL;DR description, add:

  ```markdown
  **For paper notes:** include a `## Relations` section immediately after the TL;DR table and before the TOC. The TOC rule ("immediately after the title") is superseded for paper notes by this header order: Title → Authors/venue → TL;DR → Relations → TOC. The `## Relations` heading itself is omitted from the TOC. If all sub-fields are empty, omit the heading entirely.

  The Relations section has three optional sub-fields (omit any that have no entries):

  ```markdown
  ## Relations

  **Builds on:** [[papers/paper-slug|Display Name]], [[papers/topic/paper-slug|Display Name]]
  **Extended by:** [[papers/follow-up|Display Name]] *(no note yet)*
  **Concepts used:** [[concepts/topic/filename|Display Name]]
  ```

  Placeholder links for papers or concepts without notes yet are marked with `*(no note yet)*`.
  ```

- [ ] **Step 5: Add the cross-file wikilink format rule**

  In the Obsidian TOC Link Rules section (or immediately after it), add:

  ```markdown
  **Cross-file wikilinks** (for Relations blocks and inline body links) use vault-relative paths without the `.md` extension:
  - Paper links: `[[papers/paper-slug|Display Name]]` (flat) or `[[papers/topic/paper-slug|Display Name]]` (cluster)
  - Concept links: `[[concepts/topic/filename|Display Name]]`

  This differs from intra-document TOC links (`[[#Exact Heading Text|Display Text]]`). Do not mix the two forms.
  ```

- [ ] **Step 6: Add paper note header order rule**

  In the Notes Format section, add a standalone rule (separate from the Relations description):

  ```markdown
  **Paper note header order:** For paper notes specifically, the header order is: Title → Authors/venue line → TL;DR table → Relations section → Table of Contents. This supersedes the general "TOC immediately after the title" rule for paper notes only.
  ```

- [ ] **Step 7: Add inline linking density rule**

  In the Notes Format section, add:

  ```markdown
  **Inline wikilinks** to related paper notes and concept notes should appear throughout the body — link on the first mention of a related paper or concept within each major section (`##` heading level). Do not re-link the same target within the same section.
  ```

- [ ] **Step 8: Commit**

  ```bash
  git add CLAUDE.md
  git commit -m "docs: update CLAUDE.md with new paper summary conventions"
  ```


---

## Task 2: Flatten dhen-ranking folder

**Files:**
- Move+modify: `papers/dhen-ranking/note.md` → `papers/dhen-ranking.md`
- Move: `papers/dhen-ranking/exercises.md` → `papers/dhen-ranking-exercises.md`
- Move: `papers/dhen-ranking/solutions.md` → `papers/dhen-ranking-solutions.md`
- Move (dir): `papers/dhen-ranking/figures/` → `papers/figures/dhen-ranking/`
- Delete: `papers/dhen-ranking/` (now empty)

The note embeds 6 figures with relative paths like `figures/dhen2022-fig3.png`. After flattening, those paths become `figures/dhen-ranking/dhen2022-fig3.png` (relative from `papers/`).

- [ ] **Step 1: Move files and figures directory**

  Use `git mv` for the tracked text files so git records the rename. Move the figures directory with `mv` then stage with `git add`:

  ```bash
  git mv papers/dhen-ranking/note.md papers/dhen-ranking.md
  git mv papers/dhen-ranking/exercises.md papers/dhen-ranking-exercises.md
  git mv papers/dhen-ranking/solutions.md papers/dhen-ranking-solutions.md
  mkdir -p papers/figures/dhen-ranking
  mv papers/dhen-ranking/figures/* papers/figures/dhen-ranking/
  rmdir papers/dhen-ranking/figures
  rmdir papers/dhen-ranking
  ```

- [ ] **Step 2: Update all figure references in papers/dhen-ranking.md**

  First verify all figure filenames in the moved directory — they should all begin with `dhen2022-`:
  ```bash
  ls papers/figures/dhen-ranking/
  ```

  The note has 6 figure embeds each using the pattern `![...](figures/<filename>)`. After flattening, the relative path from `papers/dhen-ranking.md` to the figures directory is `figures/dhen-ranking/<filename>`. Use a broad sed replacement that rewrites any `figures/` reference to `figures/dhen-ranking/` — safe because after flattening, all figures refs in this note point to this paper's figures:

  ```bash
  sed -i '' 's|(figures/|(figures/dhen-ranking/|g' papers/dhen-ranking.md
  ```

  Verify all 6 references are updated and no plain `(figures/` remains:
  ```bash
  grep -n "figures/" papers/dhen-ranking.md
  ```

  Expected: 6 lines, all containing `figures/dhen-ranking/`.

- [ ] **Step 3: Commit**

  ```bash
  git add papers/dhen-ranking.md papers/figures/dhen-ranking/
  git commit -m "refactor: flatten dhen-ranking folder to flat files"
  ```

  Note: `git mv` in Step 1 already staged the three file renames. `git add papers/dhen-ranking.md` re-stages it after the sed edits in Step 2. `git add papers/figures/dhen-ranking/` stages the moved (untracked) figures.

---

## Task 3: Flatten sampling-bias-corrected-retrieval folder

**Files:**
- Move: `papers/sampling-bias-corrected-retrieval/note.md` → `papers/sampling-bias-corrected-retrieval.md`
- Move: `papers/sampling-bias-corrected-retrieval/exercises.md` → `papers/sampling-bias-corrected-retrieval-exercises.md`
- Move: `papers/sampling-bias-corrected-retrieval/solutions.md` → `papers/sampling-bias-corrected-retrieval-solutions.md`
- Delete: `papers/sampling-bias-corrected-retrieval/` (empty after moves)

This folder has no figures directory — no image path updates needed.

- [ ] **Step 1: Move files**

  ```bash
  git mv papers/sampling-bias-corrected-retrieval/note.md papers/sampling-bias-corrected-retrieval.md
  git mv papers/sampling-bias-corrected-retrieval/exercises.md papers/sampling-bias-corrected-retrieval-exercises.md
  git mv papers/sampling-bias-corrected-retrieval/solutions.md papers/sampling-bias-corrected-retrieval-solutions.md
  rmdir papers/sampling-bias-corrected-retrieval
  ```

- [ ] **Step 2: Verify**

  ```bash
  ls papers/sampling-bias-corrected-retrieval*
  ```

  Expected:
  ```
  papers/sampling-bias-corrected-retrieval-exercises.md
  papers/sampling-bias-corrected-retrieval-solutions.md
  papers/sampling-bias-corrected-retrieval.md
  ```

- [ ] **Step 3: Commit**

  `git mv` in Step 1 already staged all three renames:

  ```bash
  git commit -m "refactor: flatten sampling-bias-corrected-retrieval folder to flat files"
  ```

---

## Task 4: Retrofit papers/dhen-ranking.md

**Files:**
- Modify: `papers/dhen-ranking.md`

Changes: (1) add `Key Result` column to TL;DR; (2) add `## Relations` section; (3) add inline wikilinks in body.

Read the full note before making any changes to understand what relations and key results to add.

- [ ] **Step 1: Read papers/dhen-ranking.md**

  Read the full file. Identify:
  - The TL;DR rows and what the key quantitative result is for each dimension
  - Which papers it builds on (look at the Relation to Prior Work section and references)
  - Which concept notes could be wikilinked (e.g., attention mechanisms, DCN, FM)

- [ ] **Step 2: Add Key Result column to TL;DR table**

  The current TL;DR has 5 rows with 3 columns. The 4th column `Key Result` should contain the most important quantitative result per row. Based on the paper:

  | Dimension | Key Result |
  |---|---|
  | Model architecture | +0.27% NE improvement over AdvancedDLRM |
  | Interaction modeling | Combinatorial expressivity: $k^N$ distinct interaction compositions with $k$ modules and $N$ layers |
  | Ensemble approach | Heterogeneous stacking outperforms homogeneous at every ablated depth |
  | Training | 1.2x training throughput over FSDP on a 256-GPU cluster |
  | Empirical gains | +0.27% NE; 1.2x throughput (already captured above) — leave empty or summarize |

  Edit the TL;DR table to add the fourth column header and fill in Key Result per row.

- [ ] **Step 3: Add ## Relations section**

  Insert after the TL;DR table and before `## Table of Contents`. Base the entries on what you found in Step 1 (especially Section 5 "Relation to Prior Work" and the References). Accurate example based on the paper's content:

  ```markdown
  ## Relations

  **Extended by:** [[papers/generative-recommenders/wukong|Wukong]]
  **Concepts used:** [[concepts/ab-testing/foundations|A/B Testing Foundations]] *(for NE metric interpretation)*
  ```

  Rules:
  - Only include `**Builds on:**` entries for papers DHEN explicitly extends in an intellectual or architectural sense (not just same domain)
  - HSTU belongs under `**Extended by:**` (HSTU post-dates DHEN and extends its ideas), not `**Builds on:**`
  - Do not link sampling-bias-corrected-retrieval — it has no direct lineage relationship to DHEN
  - If a concept note exists in the repo, omit `*(no note yet)*`

- [ ] **Step 4: Add inline wikilinks in body**

  Go through each major `##` section. On the first mention of:
  - DCN, xDeepFM, AutoInt, FM, GIN → these don't have concept notes yet; skip or mark *(no note yet)* only in Relations
  - HSTU → `[[papers/generative-recommenders/hstu|HSTU]]`
  - Wukong → `[[papers/generative-recommenders/wukong|Wukong]]`
  - Any concept that has a note in `concepts/`

  Use `[[path|Display Name]]` syntax. Link only the first mention per section.

- [ ] **Step 5: Commit**

  ```bash
  git add papers/dhen-ranking.md
  git commit -m "feat: retrofit dhen-ranking note — Relations section, Key Result column, inline links"
  ```

---

## Task 5: Retrofit papers/sampling-bias-corrected-retrieval.md

**Files:**
- Modify: `papers/sampling-bias-corrected-retrieval.md`

Changes: (1) add `Key Result` column to TL;DR; (2) add `## Relations` section; (3) add inline wikilinks in body.

Note: The existing TL;DR already has a `Key Result`-like final column in some rows (e.g., "+8% Recall@1 on Wikipedia link prediction") — but the header is only 3 columns. The fourth column needs to be added explicitly as a header and populated consistently across all rows.

- [ ] **Step 1: Read papers/sampling-bias-corrected-retrieval.md**

  Read the full file. Identify:
  - The TL;DR rows — note that the last column of some rows already contains quantitative results; these move cleanly into `Key Result`
  - Which papers it builds on (two-tower models, softmax cross-entropy literature)
  - Which concept notes could be wikilinked

- [ ] **Step 2: Add Key Result column to TL;DR table**

  The current TL;DR is 3-column with an em-dash `—` in the Key Result position for rows without results. Add the `Key Result` header and populate:

  | Dimension | Key Result |
  |---|---|
  | Training objective | Corrects bias from frequency-reweighted to true full-softmax partition function |
  | Bias source | Formal proof that uncorrected batch softmax is an unbiased estimator of the wrong objective |
  | Frequency estimation | Online estimator; no recomputation needed as vocabulary changes |
  | Architecture | Zero inference cost — correction is training-only |
  | Deployment | Same serving infrastructure |
  | Empirical gains | +8% Recall@1 on Wikipedia; live metric gains on YouTube retrieval |

- [ ] **Step 3: Add ## Relations section**

  Insert after TL;DR, before `## Table of Contents`. Read the References section of the note to identify what this paper directly builds on. This paper predates HSTU — do not link HSTU here.

  The paper explicitly builds on YouTube DNN (Covington et al. 2016) and the batch softmax / sampled softmax literature. Those papers do not have notes in this repo yet. A correct Relations block:

  ```markdown
  ## Relations

  **Builds on:** YouTube DNN *(no note yet)*, Sampled Softmax *(no note yet)*
  ```

  If Step 1 identifies additional papers with notes in this repo, add them. If no papers in the repo have a direct lineage relationship, it is acceptable to have only a `**Builds on:**` with `*(no note yet)*` entries, or to omit the heading if all sub-fields would be empty.

- [ ] **Step 4: Add inline wikilinks in body**

  On first mention per `##` section of any paper or concept that has a note in this repo, add a wikilink. This note likely references the two-tower model, softmax, and retrieval concepts — add links where concept notes exist.

- [ ] **Step 5: Commit**

  ```bash
  git add papers/sampling-bias-corrected-retrieval.md
  git commit -m "feat: retrofit sampling-bias-corrected-retrieval note — Relations, Key Result column, inline links"
  ```

---

## Task 6: Retrofit papers/generative-recommenders/hstu.md

**Files:**
- Modify: `papers/generative-recommenders/hstu.md`

Changes: (1) add `Key Result` column to TL;DR; (2) add `## Relations` section; (3) add inline wikilinks in body.

- [ ] **Step 1: Read papers/generative-recommenders/hstu.md**

  Read the full file (especially the TL;DR, Relation to Wukong section, and References). Identify:
  - Key quantitative results per TL;DR row (e.g. "+12.4% engagement", "+6.2% retrieval HR@100", "5.3–15.2x faster than FlashAttention2")
  - Papers it builds on (Zhao et al. 2023 user-centric, Flash Attention, Transformers, etc.)
  - Papers that extend it (Wukong is explicitly related)
  - Concept notes it could link to

- [ ] **Step 2: Add Key Result column to TL;DR table**

  The existing TL;DR has 6 rows. Fill the `Key Result` column with the most important quantitative result per row:

  | Dimension | Key Result |
  |---|---|
  | Recommendation paradigm | +12.4% engagement (ranking), +6.2% retrieval HR@100 |
  | Architecture | Stable training at 1.5 trillion parameters |
  | Scale | 1.5T parameters; scaling law holds across 3 orders of magnitude |
  | Inference cost | M-FALCON: O(n²d) amortized across $b_m$ candidates |
  | Training efficiency | 5.3–15.2x faster than FlashAttention2 |
  | Online performance | +12.4% engagement, +6.2% HR@100 (deployed to billions of users) |

- [ ] **Step 3: Add ## Relations section**

  Insert after TL;DR, before `## Table of Contents`:

  ```markdown
  ## Relations

  **Builds on:** [[papers/dhen-ranking|DHEN]] *(ensemble/interaction modeling ideas)*
  **Extended by:** [[papers/generative-recommenders/wukong|Wukong]]
  **Concepts used:** [[concepts/ab-testing/foundations|A/B Testing Foundations]]
  ```

  Note: `concepts/ab-testing/foundations.md` exists in the repo — do not add `*(no note yet)*`. Adjust other entries based on what you find in Step 1.

- [ ] **Step 4: Add inline wikilinks in body**

  In the body, on first mention per `##` section of:
  - DHEN → `[[papers/dhen-ranking|DHEN]]`
  - Wukong → `[[papers/generative-recommenders/wukong|Wukong]]`
  - Flash Attention → no note yet; skip
  - Any concept with an existing note

- [ ] **Step 5: Commit**

  ```bash
  git add papers/generative-recommenders/hstu.md
  git commit -m "feat: retrofit hstu note — Relations section, Key Result column, inline links"
  ```

---

## Task 7: Retrofit papers/generative-recommenders/wukong.md

**Files:**
- Modify: `papers/generative-recommenders/wukong.md`

Changes: (1) add `Key Result` column to TL;DR; (2) add `## Relations` section; (3) add inline wikilinks in body.

- [ ] **Step 1: Read papers/generative-recommenders/wukong.md**

  Read the full file. Identify:
  - Key quantitative results per TL;DR row
  - Papers it builds on (DHEN, HSTU, FM, DCNv2, etc.)
  - Concept notes it could link to

- [ ] **Step 2: Add Key Result column to TL;DR table**

  The existing TL;DR has 5 rows. Fill the `Key Result` column based on what you find in Step 1:

  | Dimension | Key Result |
  |---|---|
  | Scaling behavior | Power-law scaling up to 100+ GFLOP/example, ~637B parameters |
  | Interaction order | Layer $l$ captures interactions up to order $2^l$ |
  | Architecture | State-of-the-art on all 6 public datasets tested |
  | Compute efficiency | DCNv2 requires 40× more compute to match Wukong quality |
  | Public benchmarks | SOTA on KuaiVideo, TaobaoAds, and 4 others |

- [ ] **Step 3: Add ## Relations section**

  Insert after TL;DR, before `## Table of Contents`:

  ```markdown
  ## Relations

  **Builds on:** [[papers/dhen-ranking|DHEN]], [[papers/generative-recommenders/hstu|HSTU]]
  **Concepts used:** [[concepts/ab-testing/foundations|A/B Testing Foundations]]
  ```

  Note: `concepts/ab-testing/foundations.md` exists — no `*(no note yet)*`. Adjust based on what you find in Step 1 (Section 1.3 "Research Lineage: DHEN to Wukong" is especially relevant).

- [ ] **Step 4: Add inline wikilinks in body**

  On first mention per `##` section of:
  - DHEN → `[[papers/dhen-ranking|DHEN]]`
  - HSTU → `[[papers/generative-recommenders/hstu|HSTU]]`
  - FM, DCN, xDeepFM → no notes yet; skip
  - Any concept with an existing note

- [ ] **Step 5: Commit**

  ```bash
  git add papers/generative-recommenders/wukong.md
  git commit -m "feat: retrofit wukong note — Relations section, Key Result column, inline links"
  ```

---

## Task 8: Human checkpoint

- [ ] Open Obsidian and verify:
  - The Obsidian graph shows connections between `dhen-ranking`, `hstu`, and `wukong`
  - Image embeds in `papers/dhen-ranking.md` resolve correctly (6 figures should display)
  - All `## Relations` blocks render correctly — no broken wikilinks for notes that exist
  - TOC links within each note still navigate correctly

- [ ] If any images are broken: check that `papers/figures/dhen-ranking/` exists and the filenames match the embeds in `papers/dhen-ranking.md`

- [ ] Commit a final cleanup commit if any minor fixes were needed:

  ```bash
  git add -A
  git commit -m "fix: resolve any broken links or paths found in Obsidian review"
  ```
