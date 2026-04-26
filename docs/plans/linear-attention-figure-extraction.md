# Plan: Extract and Embed Figures into linear-attention.md

## Target Note
`/Users/calvinwoo/Documents/notes/concepts/attention-mechanisms/linear-attention.md`

## Figures to Extract

### 1. Katharopoulos et al. (2020) — ar5iv: 2006.16236
- Target: Figure showing linear attention as RNN (recurrent form)
- Embed near: Section 2.3 (Linear RNN Interpretation) / end of Section 2

### 2. Sun et al. (2023) RetNet — ar5iv: 2307.08621
- Target: Retention mechanism diagram (parallel / recurrent / chunkwise)
- Embed near: Section 4 intro or Section 5.2 (RetNet)

### 3. Dao & Gu (2024) Mamba-2/SSD — ar5iv: 2405.21060
- Target A: SSD chunkwise algorithm figure
- Target B: Duality diagram between SSMs and attention
- Embed near: Section 4.5 (IO-Aware) and Section 5.3 (Mamba-2)

### 4. Yang et al. (2024) GLA — ar5iv: 2312.06635
- Target: GLA gating mechanism figure
- Embed near: Section 5.4 (GLA)

## Steps
1. Fetch HTML from ar5iv.org for each paper
2. Identify <img> tags for target figures
3. Download images with curl to figures/ subdirectory
4. Verify non-zero file sizes
5. Embed markdown image syntax at appropriate note locations
6. Commit changes

## Output Directory
`/Users/calvinwoo/Documents/notes/concepts/attention-mechanisms/figures/`

## Naming Convention
`{author-year}-fig{N}-{description}.png`
