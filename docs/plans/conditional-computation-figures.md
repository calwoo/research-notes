# Plan: Copy Figures for conditional-computation-attention.md

## Status: Pending execution

## Figures Required

Five figures need to be copied from the WebFetch tool cache to:
`concepts/attention-mechanisms/figures/`

## Source

The figures were downloaded during the note-writing session. They are available in the tool cache at:

```
CACHE=/Users/calvinwoo/.claude/projects/-Users-calvinwoo-Documents-notes/35df5b44-f4a6-49a1-8f33-1a251a2bb9c1/tool-results
```

## Copy Commands

```python
import shutil, os

CACHE = "/Users/calvinwoo/.claude/projects/-Users-calvinwoo-Documents-notes/35df5b44-f4a6-49a1-8f33-1a251a2bb9c1/tool-results"
DEST = "/Users/calvinwoo/Documents/notes/concepts/attention-mechanisms/figures"

copies = [
    ("webfetch-1774619363271-0atxz6.png", "raposo2024-fig1-mod-architecture.png"),
    ("webfetch-1774619376073-qzl6ar.png", "raposo2024-fig2-routing-schemes.png"),
    ("webfetch-1774619395092-38dj6r.png", "raposo2024-fig4-isoflop.png"),
    ("webfetch-1774619393921-s30cax.png", "zhu2025-fig1-moda-architecture.png"),
    ("webfetch-1774619375254-ge8yj2.png", "zhu2025-fig3-depth-stream-taxonomy.png"),
]

for src_name, dst_name in copies:
    shutil.copy2(os.path.join(CACHE, src_name), os.path.join(DEST, dst_name))
    print(f"Copied {dst_name}")
```

## Figure Descriptions

| Filename | Paper | Caption |
|---|---|---|
| `raposo2024-fig1-mod-architecture.png` | Raposo et al. 2024 | MoD architecture diagram (two adjacent blocks with router gates) + routing decisions heatmap |
| `raposo2024-fig2-routing-schemes.png` | Raposo et al. 2024 | Token-choice vs. expert-choice vs. expert-choice MoD routing comparison |
| `raposo2024-fig4-isoflop.png` | Raposo et al. 2024 | isoFLOP analysis at 3 FLOP budgets: loss vs. parameters and normalized FLOPs vs. normalized loss |
| `zhu2025-fig1-moda-architecture.png` | Zhu et al. 2025 | MoDA decoder block with depth KV cache + visible relationships matrix |
| `zhu2025-fig3-depth-stream-taxonomy.png` | Zhu et al. 2025 | Four depth-stream mechanisms: (a) Depth Residual (b) Depth Dense (c) Depth Attention (d) MoDA |

## Note

These figures are already embedded in the note with correct `![[...]]` wikilinks.
Once copied, they will render inline in Obsidian.

Alternatively, re-run the `image-extractor` agent pointing to:
- https://arxiv.org/html/2404.02258 (Figures 1, 2, 4)
- https://arxiv.org/html/2603.15619 (Figures 1, 3)
