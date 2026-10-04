# w17/plots — FINDINGS

Branch `w17/plots` (base `base/iter8 = 148f3b7`) · machine `batmac` (Apple M4) · **zero compute,
zero scored runs** · **no w15 or w16 artefact read** (AMENDMENT 1).

## Deliverables — in `~/clu/outputs/w17-paper/figures/` (owned by `plots`; `figures/rendered/` untouched)

| figure | PDF + PNG | status | source artefacts (under `~/clu/evidence/outputs/`) |
|---|---|---|---|
| 1 crossover | `fig_crossover` | **PLACEHOLDER** (awaits w15 crossover-seeds) | `w12-single-design-crossover/CROSSOVER-val.json` |
| 2 ladder + margin panel | `fig_ladder` | **PLACEHOLDER** (w18 regenerates) | `w12b-nr6-and-kstar/KSTAR-LADDER.json`, `w12-composition-ladder/LADDER.csv` |
| 3 deletion + adaptation cost | `fig_deletion` | **PLACEHOLDER** (awaits w15 single-design-seeds) | `w12-single-design-crossover/deletion/test_seed0_K32/deletion.json`, `w12-deletion-baselines/comparators/seed_0/*__retrain_without__value-object_hue-0.rival.json` |
| 4 encoder-adapt margins | `fig_encoder_adapt` | **final** | `w13-encoder-adapt/CROSS-ARM.json` |
| 5a maze: read verb | `fig_maze_readverb` | **PLACEHOLDER** (awaits w16 maze-wm) | `w14-read-verb/runs/scored{1,2,3}_seed{0,1,2}_test.json` |
| 5b maze: request-layer fix | `fig_maze_requestlayer` | **PLACEHOLDER**, appendix | `w14-request-layer-fix/artefacts/measure_{before_0,after_0,after_1,after_2}.json` |

Plus `FIGURES.md` (figure → artefact map, arm/colour table, byte convention, placeholder list,
LaTeX include block with `%` source comments), `PROVENANCE.json` (machine-readable, written by
the build), `make_figures.py` + `build.sh` (so wave 18 regenerates in place), and the two
SHA-256 manifests.

## ⭐ Rebuild determinism — PROVEN

`build.sh` runs `make_figures.py` twice, in two separate Python processes, into two scratch dirs,
then diffs `shasum -a 256` of every output. It refuses to install on any difference.
Pins: Agg backend forced; DejaVu Sans (bundled); `pdf.fonttype 42`; PDF metadata
CreationDate/ModDate/Producer/Creator = None; PNG `Software` = None; `SOURCE_DATE_EPOCH=0`;
no randomness; every input list sorted. Python 3.12.14 · matplotlib 3.10.8 · numpy 2.4.1.

Output of the installing run: `REBUILD: BYTE-IDENTICAL`. Build 1 = build 2:

```
4f0f42c1d725bd94bf4319c7b6fee134523b9a7bc413e9c91b3a3a1e048af1bd  fig_crossover.pdf
02cffe646268a96fddade1ec73762bc81eff6521a0301f9f0f6d89723175be3f  fig_crossover.png
2a5520fa47f5128a4dc1f9a8935226c736642611214a50e0cd09477ffc63c7f0  fig_deletion.pdf
6f913caafc5591cb0f2779aec00ea3bf91c3845ea2cde409bde3c6bc08e1f67f  fig_deletion.png
65754756c7a3f964f10e10a7ef8b03bcc402d8c031a68d6fce39b85d02291da4  fig_encoder_adapt.pdf
78c2eabf829370b1d629972425f6aededbf8d1c8ff551c3ba93c837dd8c16eba  fig_encoder_adapt.png
9e212bcd142735f2445a41f8f8f929f63e7f79c3d923730eb60fb19829ff51bd  fig_ladder.pdf
103ffa81276b75247c8816acb95500d443d19ab288dc1ee109fc05772123dc75  fig_ladder.png
14ebc4d41c6fcb84f829966d327cc4e7a3ccb140fac4abda2181a8d8fc4bedd4  fig_maze_readverb.pdf
57123908907145e829a667f955cb944286ea8c5fe94fb11d996a5dff4a21259c  fig_maze_readverb.png
55ddc8f055250ecf7a9ec75b9b440d74156a60e7b994d99e49dbcab9515b2423  fig_maze_requestlayer.pdf
6c3a11f29dbf13706d4e3949a39afd9b85505be761af13ed8952e3c0eac850c0  fig_maze_requestlayer.png
bfe6a47309bb5f500457328c4b7e89bbbde39f12e3c48ae280a79a2c50e76cb5  PROVENANCE.json
```
(`diff SHA256-build1.txt SHA256-build2.txt` → empty; both files are in `figures/` and here.)

## Choices taken (pre-authorised; closest to an ICLR main-track result)

1. **ONE byte convention everywhere: the four-part honest ledger** (crossover hub's), summed never
   netted, 524,288 B as a reference line only. The K = 5 full-square-priced 1× opponent is not drawn.
2. **Ladder = the K\* = 32 replication** (main text carries one K). The K = 5 ladder (appendix
   endpoint) is not drawn — see NOT-DONE.
3. **Placeholder marking** = red suptitle tag "PLACEHOLDER · w≤14 evidence · wave 18 regenerates"
   (a status mark, not a caption) + the list in FIGURES.md + `placeholder: true` in PROVENANCE.json.
   Amendment 1 names crossover, ladder, deletion and every maze figure; all five carry it. The
   encoder-adapt figure needs nothing from w15/w16 and is final.
4. **No in-plot captions.** The only in-plot text: panel titles, axis labels (with units), legends,
   value labels on the rows-read bars, the "declared +1.0" reference label and the placeholder tag.
5. **Arm names**: "CLU" everywhere (the maze CLU and the Shapes3D single design are both the CLU
   arm of their table; one name, one colour).

## ⛔ Things found while building (for the Hub / draft-v2 / referee-3)

- ⛔ **"Margin vs best rival" at K\* has no artefact.** Learned 1× is NOT the strongest learned arm
  at n = 3 on seeds 0 and 2 (a 524,288 B LSTM / Transformer scores above it, per LADDER.csv vs
  KSTAR-LADDER.json). The build detected this and titled panel (b) "Margin vs learned 1×".
  `\TODO{no artefact for the K* margin against the best of ALL learned arms at n = 3}`.
- The crossover's "3-seed band" exists only at K = 5 and K = 32; K = 8…28 are n = 1. Printed in
  the legend.
- `CROSSOVER-val.json`'s first key is `SMOKE_NOT_A_CLAIM: true`; the honest four-part columns are
  plotted (the w13 advisor report's correction), not the parts-1-2-4 column.
- Rival post-deletion **composition** on the scored corners was never measured (w≤14);
  `\TODO{no artefact for rival post-deletion composition on the scored corners}`.
- Deletion cost pools differ (CLU 131,072 rows vs rivals 16,384 rows) — printed on the axis.
- fig 5b puts the CMAZE-2 and CMAZE-2b request layers in one figure because it *is* the fix; no
  score crosses environments. Flagged as a judgement call in FIGURES.md.

## Verification

- `pdflatex` (TeX Live 2026) against `iclr2027/iclr2027_conference.sty`: all six PDFs included at
  `\linewidth` (encoder-adapt at 0.55\linewidth): **0 errors, 0 overfull boxes**; fonts embedded
  (TrueType subsets).
- Every figure was rendered and looked at (PNG) for collisions/overflow; two layout passes fixed
  tag/title overlap, clipped legends and colliding tick labels.

## DECLARED NOT-DONE

| item | reason |
|---|---|
| running the dataviz palette validator (`validate_palette.js`) | **node is not installed on this machine** (nor deno/bun). The palette is the skill's reference palette in its documented validated order (slots 1–6 adjacent-pair pass, light mode); every series also has a marker/hatch as secondary encoding. Slots 7–8 beside slot 1 (fig 4) are not validated as a pair — markers and x-position carry identity there. |
| dark-mode variants | paper figures for a printed PDF; light only. |
| an appendix figure for the K = 5 ladder | its 1× opponent is priced in the full-square convention (the "inflated" row); drawing it would break the one-convention rule, and drawing K = 5 against only the 524,288 B rivals would hide the 1× loss. Left to the draft's appendix table. |
| an appendix figure for the four-part-convention K = 5 "bridge" | n = 1 only (seed 0); not requested. |
| figures for the composition-only / item-store ablations (T11), the positioning count, the refusal displays | not in this spoke's figure list; data exists (w13-seeds-and-tables TABLES.json) if the Hub wants them. |
| 3-seed crossover band at every K, 3-seed deletion, any maze-wm figure | **w15/w16 data — forbidden by AMENDMENT 1.** Placeholders stand in. |
| a hover/tooltip layer (dataviz step 5) | static vector figures for a LaTeX paper; not applicable. |
