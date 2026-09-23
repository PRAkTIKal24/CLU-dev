# w18 spoke/seeds-and-tables: FINDINGS (THIRD PASS; extends pass 2, which is kept below)

**Branch:** `w18/seeds-and-tables`. **Machine:** batmac. **Scored runs: 0. Compute: 0**, apart from LaTeX compiles of the drafts in scratch build directories.
**Pass 3 EDITED BOTH DRAFTS**, `outputs/w17-paper/DRAFT-v2.tex` and `DRAFT-v2-style.tex`, as the Head directed. **No `\PLACEHOLDER{}` was touched**: the ordered, byte-identical sequence was checked by the edit script and by the style-diff tool. **No w19 number entered either draft.**
Every number in `TABLES.md` is a build-verified cell. A number in this file is quoted from `TABLES.md`, with the section named, or it carries its own `path:line`.

## 0. What shipped (pass 3)

| file | what |
|---|---|
| `TABLES.md` §9 | ⭐⭐ **w19 `maze-t-symmetric` folded IN FULL**, in its own tables W19-1 … W19-8 (CMAZE-T only, never a w16 row). The tables are: the condition table with every condition and the OBS-X10 diagnostic; BOTH TEST per arm and per seed; the other rows; per-step and rollout; the ledger, with the convention in the table; deletion; extractor accuracy; and the budget |
| `TABLES.md` §10 | ⛔⛔ **The consequence for w16**, written where the two tables are presented, and the list of where it now sits in both drafts |
| `TABLES.md` §11 | w20 `shapes-data-matched`: **awaiting, a fourth pass** |
| `TABLES.md` W16-T1, §4b, §5, §6.2e, §8 | ⛔ PASS-3 lines. The w16 store row is qualified. §4b is closed by the Hub. MF2 and `tab:k5` are closed |
| `draft_pass3/apply_pass3_edits.py` | The guarded, idempotent draft edit: 9 anchors per file. It refuses to write if an anchor is missing or ambiguous, if the `\PLACEHOLDER` sequence changes, or if 23.5117 survives |
| `draft_pass3/DRAFT-v2{,-style}.pass3.diff`, `STYLE-DIFF-CHECK-pass3{,-baseline}.md`, `COMPILE.md`, `compile.sh` | The evidence for the edit |
| `draft_pass3/DRAFT-v2.pass2-e7745f2c.tex`, `pass2-manifest.json` | The frozen pass-2 inputs (§4) |
| `cells_pass3.py`, `gen_cells_pass3.py`, `build_tables.py` | 146 new cells. The build now also resolves `@here/` sources and checks HISTORICAL cells against the pass-2 manifest. **653 cells verified, 583 used, 29 historical.** Two builds are byte-identical. The negative test fails with exit 1 on a mutated w19 row, a wrong draft line, a bare number, an orphan value-only token and a tampered historical cell |

## 1. ⭐⭐ w19: every condition LOST, and every condition is a printed row (`TABLES.md` §9)

- **W19-1** prints all six rows of the record's condition table (`w19-maze-t-symmetric/TABLES.md:11–16`): WORDS-MAIN, WORDS-FULL, OBS, BOTH-VAL, BOTH-TEST, and OBS-X10. Each row has its seeds (3), its scored runs and its verdict. **The verdicts are WORDS-MAIN LOSS, WORDS-FULL LOSS, OBS LOSS, and BOTH LOSS** (VAL PASS, then TEST fails SB1). **OBS-X10 is labelled DIAGNOSTIC ONLY (AMENDMENT 1), never a bar.** It would read LOSS, and it cannot pass a condition or earn a run (`PREREG-w19-maze-t-symmetric.md:70–71`).
- **Budget** (W19-8): 3 of 9 scored runs spent (BOTH, TEST) and 6 declared not-run (`w19-maze-t-symmetric/FINDINGS.md:200–201`). Config C-512 was not re-run. It is declared not-run, not a loss.
- ⚠ **The launch brief's BOTH figure is the TEST margin.** The brief reads "val PASS knife-edge (+0.099 …)". The record's VAL margin is +0.108, a PASS (`TABLES.md:14`). The +0.099 is TEST, the SB1 FAIL (`TABLES.md:15`). The record's own §0 row sets "knife-edge" beside the TEST margin (`FINDINGS.md:24`). `TABLES.md` §9 prints the record.
- ⚠ **"786 objects".** The record says 384 objects (`FINDINGS.md:224`). Summing the entries of the 18 `MANIFEST.json` files gives 384. Adding each object's `.sha256` file and the 18 manifests gives 786 files: **786 = 384 + 384 + 18.** ⚠ **Only the 18 manifests are staged on batmac** (168K in `persist/`). The objects themselves are not here, and nothing on this box re-verifies them.
- ⚠ Two discrepancies that the record itself declares are printed rather than resolved. The observation is 634 wide in the PREREG and 636 as run (`PREREG…:21` against `FINDINGS.md:73`). Each result JSON's `persist_dir` names the path before the objects were moved (`FINDINGS.md:236–238`).

## 2. ⛔⛔ The consequence for w16 (`TABLES.md` §10, W16-T1)

**w16's CMAZE-T A1 n-way row was measured with the CLU reading a 318-wide `[bias|main|pair]` design that the rivals did not receive** (`w19-maze-t-symmetric/FINDINGS.md:29–30`). w19's WORDS-FULL condition gave every arm that design. In the record's own words, *"the pair block, not the store, carries the w16 margin"*: a matched transformer given the same design reads 0.742 against the CLU's 0.725 (`FINDINGS.md:22`). The PREREG had named exactly this outcome, in advance, as the proof of an input artefact (`PREREG…:57–58`).
**What stands:** the store gate and exact store-scoped deletion (`FINDINGS.md:36`, `:194`). **What does not stand:** the claim of a margin over matched rivals.
**Where it is stated:**
- In `TABLES.md`, at W16-T1, where the w16 row is printed, and in §10.
- In **both drafts**, in seven places, each replacing a "symmetric-input test is pending": the abstract, the contributions item, `sec:maze-wm`, the section summary, the limitations item, the `tab:cmazet` caption, and the `tab:cmazet` n-way row. The n-way row is now labelled "STORE row under asymmetric inputs only".
The two tables stay separate. `tab:cmazet-sym`'s rows remain placeholder-fill's.

## 3. The two open findings from pass 2: CLOSED in both drafts

- **(a) MF2.** 23.5117 → **23.5121**, with `% SRC: w15-table-fixes/TABLE-FIXES.md`, at `DRAFT-v2.tex:1738` and `DRAFT-v2-style.tex:1725`.
  - ⚠ The pass-2 locations (L1317, L2262) no longer exist. The draft moved after pass 2.
  - The L2262 appendix had already been moved **verbatim** into `CORRECTIONS-ABSORBED.tex` by w20 referee3-fixes. That file is `\iffalse`, outside the submission, and **still reads 23.5117** (`CORRECTIONS-ABSORBED.tex:29`). It was not edited. It is a verbatim archive, and it is not one of the two drafts.
  - **There is no `DRAFT-v1.tex` in `w17-paper/`.** "Both .tex files" was read as the two live drafts.
- **(b) `tab:k5`.** The caption now carries the corner counts and the inference note ("for $n\ge3$ the rungs differ in fit-pool coverage, not in composition depth"), with `% SRC: w12-composition-ladder/FINDINGS.md`, the K = 5 ladder's own record (`DRAFT-v2.tex:1475–1477`).
**Checks** (`draft_pass3/COMPILE.md`, `STYLE-DIFF-CHECK-pass3.md`):
- Both drafts compile before and after (pdflatex → bibtex with `BIBINPUTS`/`BSTINPUTS` = `w17-paper/` → pdflatex ×3). Each gives exit 0, **0 undefined citations** (32 `\cite` each), 0 undefined references and 0 overfull boxes, at 47 and 48 pages.
- **All four style diffs are EMPTY** (exit 0).
- The edit ran on scratch copies first. A second run was a no-op, and the live files are byte-identical to the tested copies.
- md5: `DRAFT-v2.tex` fab8ddcb… → bb489831…; `DRAFT-v2-style.tex` a77cdae5… → 70b7d89c….
**Ownership check before editing:** there was no live session of placeholder-fill, referee3-fixes, citation-prune, referee4-fixes or draft-figure-slots (checked with `ps` and `tmux ls`), and each of those spokes' output directories has an `END_UTC.txt`. The only concurrent session is `w18-registry-fold`, which edits registry files only.

## 4. ⚠ Why the pass-2 register broke, and how it was pinned (no number moved)

1. **The Hub corrected `w16-advisor-report.md` after pass 2, using pass 2's §4b.** Its l.41 reads "CORRECTED 2026-09-23 against the primary record … by `w18/seeds-and-tables` pass 2". The report's old sentences are gone, so 29 cells quoting them could not be re-checked on the live file. They are now checked against the committed pass-2 manifest (`af7ec86`, copied to `draft_pass3/pass2-manifest.json`): same value, same line, `verbatim_on_line: true`. They print as **HISTORICAL**. §4b is marked **CLOSED BY THE HUB**.
2. **The draft moved.** Placeholder-fill, referee3/4-fixes and figure-slots all wrote to it after pass 2, and then this pass edited it. The pass-2 draft cells describe the pass-2 text, which is **reconstructed exactly**: `caption_conventions/DRAFT-v2.captions.tex` + `patch -R CAPTIONS.diff` gives md5 `e7745f2c…` and sha256 `8e45fa98…`, the same as the pass-2 register. It is kept as `draft_pass3/DRAFT-v2.pass2-e7745f2c.tex`. Post-pass-3 facts about the drafts are separate cells against the live files.
Rendered with the pass-3 build, the pass-2 template is **identical in content** to pass 2's `TABLES.md` once pointers are stripped.

## 5. DECLARED NOT-DONE / what did NOT run (pass 3)

- **w20 `shapes-data-matched`: STILL RUNNING and NOT FOLDED.** Its M-131k, M-16k and matched-deletion rows are a **fourth pass**. The only trace on batmac is the draft's `\PLACEHOLDER` (`DRAFT-v2.tex:512`). Nothing is guessed.
- **`tab:cmazet-sym`'s seven `\PLACEHOLDER`s were not filled.** They belong to placeholder-fill, and §9 is their fill specification. ⚠ Until they are filled, that table's caption still says "pending", while `tab:cmazet`'s caption now says "no condition wins". The inconsistency is declared.
- **Not edited:** `CORRECTIONS-ABSORBED.tex` (still 23.5117; archive); MF14(i) (the BLIND columns in `tab:cmaze2`); the pass-2 convention caption patch (§8, still unapplied); the `tab:cmaze-del` label collision; any figure.
- **Not verified here:** the 384 persisted w19 objects, which are not on batmac.
- **Not re-derived:** no w19 number was recomputed. Every cell is copied from the record.
- **Background jobs:** one, the live-draft re-compile (PID 61442). It finished inside the turn, and `PIDS.txt` reads None. **Sub-agents: none.** No process was killed, and `tmux kill-server` was not used.
- **The three w15 spokes that pass 2 declared AWAITING** (`bar-d-score`, `coverage-control`, `lace-poe-rows`, pass-2 §6.1) **have since landed.** Their artefacts are now staged at `evidence/outputs/w15-{bar-d-score,coverage-control,lace-poe-rows}/`. Per `DRAFT-v2.tex:17`, placeholder-fill pass 2 has already written them into the draft. ⚠ **This pass did NOT fold them into `TABLES.md`**, because they were outside this pass's brief. §6.1 still reads "awaiting" and is **stale**; for those three rows, placeholder-fill's record supersedes it. Folding them (and cross-checking the draft's cells against the artefacts) is left for the fourth pass.

---

# PASS 2 FINDINGS (retained verbatim; where pass 3 above says otherwise, pass 3 wins)

**Branch:** `w18/seeds-and-tables`. **Machine:** batmac. **Scored runs: 0. Compute: 0** (apart from two LaTeX compiles of a draft *copy* in the scratchpad, §4).
**No `.tex` file was edited and no `\PLACEHOLDER{}` was touched.** The live `outputs/w17-paper/DRAFT-v2.tex` md5 is `e7745f2c9ea99d932635e4b062755ae7` before and after this pass.
Every number in `TABLES.md` is a build-verified cell: the value is verbatim on its cited `[file:line]`. A number in this file is either quoted from `TABLES.md`, with the section named, or carries its own `path:line`.

## 0. What shipped

| file | what |
|---|---|
| `TABLES.md` | The fold specification, rebuilt. w16 is re-folded from the primary record (§2 CMAZE-T, §3 CMAZE-O, §4 budget). **§4b is the report-vs-artefact reconciliation.** The six landed w15 spokes are folded (§6.2). The three awaiting spokes are declared again (§6.1). The 1× bracket (§7) and the convention audit with its closures (§8) complete it |
| `TABLES-w18.json` | Machine-readable: 507 cells, each with its value, source, line, `verbatim_on_line: true`, and the sha256 of every source file |
| `build_tables.py`, `cells_pass2.py`, `TABLES.template.md` | The sources. Pass 2 adds row pointers `{{@ALIAS:LINE}}` and value-only cells `{{id:v}}`. A value-only cell **fails the build** unless it shares its template line with its own row pointer. It also adds an unrendered-token check, an id-shape check, and scientific notation in the bare-number guard. The first two closed a hole found during this pass: capitalised ids slipped past the token regex. The third closed another: `7.7e-16` slipped past the guard. A negative test (a mutated cell, an orphan value-only token, a bare number, a bad id) exits 1 with all four caught. The register is static. It was generated once from `gen_cells_spec.py` (kept for provenance; the build does not run it) and is re-verified on every build |
| `caption_conventions/` | `apply_caption_conventions.py` (five guarded, idempotent caption / label edits), `CAPTIONS.diff`, and `DRAFT-v2.captions.tex` (the patched **copy**) |

## 1. ⛔ Wave 16 re-folded from `w16-maze-wm/`. The Hub's report disagrees with the primary artefact in eight places, and THE ARTEFACT WINS in every one

The Hub re-derived nothing, but it paraphrased. Pass 1 folded that paraphrase. It was the only w16 source on batmac at the time. Pass 2 re-reads every cell from `w16-maze-wm/{FINDINGS.md, TABLES.md, MAZE-WM.json, PREREG-*-T2.md, t2/C2r_val_seed*.json}`. The full table with pointers is `TABLES.md` §4b. In short:

1. **The CMAZE-O "converged rivals" triples belong to Config C2 (latent), not to C-field.** The report (`w16-advisor-report.md:42`) puts "CLU n-way 0.088 / 0.150 / 0.173 vs best rival 0.315 / 0.413 / 0.463" under the C-field withdrawal. In the record they are the C2 VAL leg (`w16-maze-wm/FINDINGS.md:219–220`). C-field against its own 30k-step rivals is the MLP at 0.988 / 0.990 / 0.990 (`w16-maze-wm/FINDINGS.md:212`) against the CLU in the header of `w16-maze-wm/TABLES.md:1292`. ⛔ **Pass 1 printed the C2 triples as "the same comparison" as C-field. That row is withdrawn.**
2. **"CMAZE-O yields no store row at any config"** (`w16-advisor-report.md:49`) contradicts the record. The C-field **store gate holds**, and what is withdrawn is the composition-over-rivals claim (`w16-maze-wm/MAZE-WM.json:369`). It is still not a win, and `TABLES.md` never prints it as one.
3. **CMAZE-O has three failing links, not one.** The record gives Config A content/read, C-field vs converged rivals interface, and C2 read (`w16-maze-wm/MAZE-WM.json:377–379`).
4. **The 80,000-step re-test is Config A1 on VAL** (`w16-maze-wm/TABLES.md:1281`), a diagnostic and not a TEST re-score. The scored leg against 80k-step rivals is C-512 on TEST (`TABLES.md` W16-T6).
5. **"Relative ΔS = 0.0 on every seed" does not describe the CMAZE-T winner.** For A1 the statistic residual is 7.7e-16 … 1.2e-15 and only the atoms Δ is 0.0 (`w16-maze-wm/FINDINGS.md:110`, `TABLES.md:144–149`). The report's wording is the CMAZE-O C-field deletion's (`w16-maze-wm/FINDINGS.md:205`).
6. **Rollout, "every learned arm".** This holds for the CLU and the matched rivals. The 10×-float *diagnostic* transformer reads a non-zero validity on seed 2 (`w16-maze-wm/TABLES.md:138`). The per-step comparison is CLU 0.925 against transformer 0.818 (`w16-maze-wm/FINDINGS.md:107–108`). ⛔ **Rollout validity 0.000 for every learned arm that is the CLU or a matched rival stays printed as a declared negative, link READ** (`TABLES.md` W16-T3).
7. **The C-512 seed-1 caveat names both numbers**: random_atoms 0.464 against the CLU's 0.567 (`w16-maze-wm/FINDINGS.md:169`).
8. **CMAZE-O deletion RAN**, on C-field TEST, both scopes, every seed (`w16-maze-wm/TABLES.md:877–888`). Pass 1 declared it not-run "by construction" from the report's silence. The declared not-runs are only B, C2 and C2r (`w16-maze-wm/FINDINGS.md:268–269`).

**Also closed from the record:** the within-maze lookup (`TABLES.md` W16-T1), the seen, 2-way and value rows with their claim labels (W16-T2), both deletion scopes with composition and cost (W16-T4, W16-O3), the ledgers with the convention named (W16-T5, W16-O4), and seed ids rather than run indices. **Pass-1 flag 1** (A1 and C-512 agreeing to three places) is explained by the record: VAL n-way plateaus at one per-seed triple across about 15 configs, a ceiling set by the request design (`w16-maze-wm/FINDINGS.md:137–140`).
**The CMAZE-O margin stays WITHDRAWN BY THE HUB** (`w16-maze-wm/MAZE-WM.json:22`). Runs 4–6 are attached to a claim that does not stand (`TABLES.md` W16-O1, §4).

## 2. The six landed w15 spokes, folded from their own artefacts (`TABLES.md` §6.2)

- **`crossover-seeds` → `tab:kcurve`.** The seven middle rows are now at n = 3, with the convention in the table (`w15-crossover-seeds/CROSSOVER-3SEED.md:11–15, 21–29`). Three things must travel with the table:
  - ⛔ **The frozen rule is not resolvable at n = 3.** The seed sd exceeds the tolerance at six of nine K (`CROSSOVER-3SEED.md:37–47`). This confirms, from the artefact, the one secondary quote pass 1 took from `w17-advisor-report.md:199`.
  - ⛔ **K = 20 fails the strict control gate** (`CROSSOVER-3SEED.md:112, 123, 134`).
  - ⛔ **Two draft sentences become false at n = 3.** The draft says "the largest drop anywhere is +0.012731 at K = 10" (`DRAFT-v2.tex:1247`), while the artefact says every drop is ≤ 0 (`w15-crossover-seeds/FINDINGS.md:130–131`). The draft prints the w12 n = 1 rows (`DRAFT-v2.tex:1230`), which the n = 3 rows supersede.
  - The U-shape is confirmed, with its minimum at K = 24 on every seed (`CROSSOVER-3SEED.md:97–101`).
- **`single-design-seeds` → `tab:del`, `tab:rivals-del`, `tab:hole`.**
  - `tab:del`: seeds 1 and 2 on the value, red-sphere and item rows, with cos_min from each seed's `deletion.json`.
  - `tab:rivals-del`: the seconds column goes to 3 seeds (`T7-3SEEDS.md:135–146`).
  - `tab:hole`: our row goes to 3 seeds (`w15-single-design-seeds/FINDINGS.md:273–274`).
  - T7 reproduces the draft: all 36 per-seed cells of the six arm and comparator rows (`T7-3SEEDS.md:15–34` against `DRAFT-v2.tex:358–366`) agree at the artefact's three decimals. This was checked by script and is not a registered cell.
  - ⛔ **What travels:** the deleted red corners **still compose red at all three seeds** (`FINDINGS.md:290–292`). The deletion claim is store-scoped exactness plus a data-free adaptation cost, not capability erasure. ⚠ The draft's "every row rewrites 2,081,024 bytes" (`DRAFT-v2.tex:555`) is seed 0 only; the bytes differ per seed (`T7-3SEEDS.md:86–87`).
- **`maze-deletion-2b` → new `tab:maze-del-2b`, CMAZE-2b only.**
  - Exactness, adaptation cost and the rival's retrain-without are folded, with the convention in the table (`w15-maze-deletion-2b/FINDINGS.md:73–75`).
  - ⛔ Deletion does **not** induce refusal (`FINDINGS.md:126–128`).
  - ⚠ The artefact is internally inconsistent: TABLE.md prints "—" for the carry-path seconds, and FINDINGS §4 prints them (`FINDINGS.md:111`).
- **`cmaze-t-harness`, `cmaze-o-harness` → P12.** The replay, impossible-example and VAL-smoke audits are folded (`TABLES.md` §6.2d).
  - ⚠ The CMAZE-T harness shipped **three** warnings (`w15-cmaze-t-harness/HARNESS-READY.json:280, 289, 298`). The w16 report named one.
  - ⚠ The w16 record's AMENDMENT 1 notes that this harness's across-wells control has a partial-identity risk at K = 4 and K = 8 (`w16-maze-wm/FINDINGS.md:65`). The w15 VAL smoke should carry that note.
- **`table-fixes` → MF2, MF14(i).** Checked against `DRAFT-v2.tex`, not DRAFT-v1:
  - ✘ **MF2:** the draft still carries the referee's rounding slip, 23.5117 (`DRAFT-v2.tex:1317, 2262`). w15 corrected it to 23.5121 in DRAFT-v1 (`w15-table-fixes/TABLE-FIXES.md:83–84`).
  - ✘ **MF14(i):** the BLIND columns are in the prose after `tab:cmaze2` (`DRAFT-v2.tex:1591`), not in the table (`DRAFT-v2.tex:1573`).
  - Both are edits for the Head or placeholder-fill. Neither touches a placeholder.

## 3. ⚠ The 1× opponent: the BRACKET stays printed. The pass-1 judgement stands

The ladder (K = 5) sized its 1× opponent from the full-square Σ d_k² ledger. The K\* replication used the symmetric-triangle float64 four-part total. **The bracket 938,413–1,710,324 (`w13-advisor-report.md:85`) is printed wherever the K = 5 opponent appears. The K\* opponent (536,270–536,972, `w12b-nr6-and-kstar/FINDINGS.md:33`) is labelled symmetric-triangle float64.** The three reasons are unchanged:
1. **Re-pricing is re-deriving.** Moving the ladder's opponent onto the triangle convention would mean recomputing ledgers from `well_sizes`, which this wave forbids.
2. **No same-convention opponent exists at 3 seeds.** The only in-bracket opponent (1,029,776 params, `w13-advisor-report.md:73`) is **n = 1**, a replication, not scored. Making it the convention would promote a 1-seed row to the table's standard.
3. **The draft already prints the bracket** (`app:k5`) and drops the out-of-bracket rows. Keeping it stays consistent with the draft and with referee-2's N1/N2.

Nothing in the w15 or w16 artefacts moves the bracket. w15 `single-design-seeds` restates the same limit: the 1× column is a declared not-run (`w15-single-design-seeds/FINDINGS.md:97–100`). `tab:1x`'s caption is now closed (§4).

## 4. The four byte-bearing tables without a convention in the table: CLOSED

Each convention is named from its own source (`TABLES.md` §8), as caption text in a guarded patch:
- `tab:1x`: float32 opponent parameters matched to the object's four-part total, with a symmetric-triangle float64 statistic.
- `tab:hole`: ours is one byte per hole in the coupling table; the comparator's is its whole parameter blob.
- `tab:rivals-del`: the whole parameter blob.
- `tab:verb`: the lookup floor is matched to the store's read floats; the match does not bind, and the floor is the whole training set.

The partial `tab:maze-del` also gets (float32) on atoms and address book (`w13-seeds-and-tables/TABLES.md:162, 164`).
**Verified on a copy:**
- All five edits apply, and a second run is a no-op with byte-identical output.
- The `\PLACEHOLDER` count is unchanged: 32 tokens, on 33 lines that mention it.
- Base and patched copies both compile with pdflatex and bibtex: exit 0, 35 pages, the same warning count (102), 0 overfull boxes.

The diff is 10 insertions and 5 deletions, all in caption or label text (`caption_conventions/CAPTIONS.diff`).
✔ The ladder's n ≥ 3 TRAINING-DATA note travels with `tab:ladder`. ⚠ It still should also travel to `tab:k5`, which has the same rungs. That is prose, not a convention, so it is on the Head's list and not in the patch.

## 5. Why the caption patch was NOT applied to the live draft

1. `outputs/w17-paper/DRAFT-v2.tex` is not in this spoke's tree. It is the paper's working file, and `placeholder-fill` writes into the same file. A concurrent write from here could collide.
2. The patch adds only banked conventions and no w15/w16 number, so AMENDMENT-1 does not block it. Applying it is still the owner's single command: `python w18_seeds_and_tables/caption_conventions/apply_caption_conventions.py outputs/w17-paper/DRAFT-v2.tex`. The script refuses to write if any anchor moved.

## 6. New flags from the primary records. These are readings, not corrections

- ⚠ **CMAZE-T "matched" rivals are up to +11.9 % (transformer) and −6.4 % (LSTM) off the CLU's read floats** (`w16-maze-wm/TABLES.md:155`). The record's "within 2 %" is stated for CMAZE-O only (`w16-maze-wm/FINDINGS.md:238`). The CMAZE-T caption should print the residuals.
- ⚠ **CMAZE-O C-field has no AE in its read path** (`w16-maze-wm/FINDINGS.md:195`), but the draft's `tab:cmazeo` caption says "the same frozen encoder" (`DRAFT-v2.tex:2050–2051`).
- ⚠ **The CMAZE-O atom-row `landscape_shuffled` equals the CLU on every row.** It is a declared identity (`w16-maze-wm/PREREG-w16-maze-wm-T2.md:21–23`) and must be labelled that way, never read as a control that failed to collapse.
- The C2 per-seed values are 0.0875 / 0.15 / 0.1725 in the JSON (`t2/C2r_val_seed{0,1,2}.json:281`). FINDINGS rounds half-up and TABLES rounds half-to-even. Neither is wrong.
- Pass-1 flag 5 (the `tab:cmaze-del` label collision) is **still open**.

## 7. DECLARED NOT-DONE / what did NOT run

- **w15, STILL AWAITING, NOT FOLDED** (running on the other box; `TABLES.md` §6.1):
  - `bar-d-score` → a **new `tab:bard`**, never `tab:encloop`: the full-decoder GPU BAR D row for `alone` / `unfrozen` / `loop_swap`. **P11, P31.**
  - `coverage-control` → **one new row in `tab:ladder`**, in the n = 6 column: store, n = 6, fit pool restricted to n = 3's train index. **P4.**
  - `lace-poe-rows` → `tab:t7`'s "LACE, PoE" row split into separate LACE and PoE rows (row 3, row 4, paired Δ); LACE and PoE rows on **every rung of `tab:ladder`**; and generation-table **run 3**. **P3, P7.**
- **Not done:** any `.tex` edit on the live draft; any `\PLACEHOLDER{}`; applying the caption patch to the live draft (§5); any figure (including OPEN-W15-1, the `fig1_cmaze2_main_scoreboard.png` mismatch); any rebuild of w13's `build_tables.py` / `verify_tables.py`; the `tab:k5` note; the MF2 and MF14(i) draft edits (§2); the `tab:cmaze-del` label fix.
- **Not re-derived:** no w15 or w16 number was recomputed. Where a table printed a rounded value, the fold either used it as printed or cited the JSON line with the raw value.
- **Pass-1 cells no longer used in `TABLES.md`**, kept in the register for audit: 69 report-sourced, plan and spare cells, listed under `cells_unused_in_template` in `TABLES-w18.json`.
- **Background jobs: none. Sub-agents: none.** No process was killed.
- ⛔ **AMENDMENT-1 still binds.** No w15 or w16 number enters a draft until the Head and the Advisor have discussed them. `TABLES.md` is the fold spec, not an edit.
