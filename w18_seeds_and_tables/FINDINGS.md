# w18 spoke/seeds-and-tables: FINDINGS (SECOND PASS; supersedes the first pass)

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
