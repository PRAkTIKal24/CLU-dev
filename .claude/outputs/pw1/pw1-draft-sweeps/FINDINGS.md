# FINDINGS — pw1-draft-sweeps (PW-1 plan items 2, 3, 5 and "the engine's judge")

Session `pw1-draft-sweeps` · branch `pw1/draft-sweeps` · batmac · started 2026-09-25T10:56:41Z.
Tooling (committed): `pw1_draft_sweeps/sweep.py` (the sweep), `scan.py` (residual scan), `make_tables.py`
(evidence tables), with the reused checkers `style_diff_check.py`, `check_provenance.py`, `compile_check.sh`, `texnum.py`.

## Status per assigned item

| item | status | note |
|---|---|---|
| 2 — nomenclature, banned words, economic metaphors, scored run / spend, physics renames, store → n-CLU, naive-CLU (n-CLU) at first mention, naive-CLU in the title, plain control names | **done** in both `.tex` files (prose, captions, table headers, table cells, appendices, abstract except its `\TODO`) | Figure images and figure scripts are not mine (brief); their in-image text still carries old words (see residual list B). |
| 3 — held-out (factor) combination, capsule, ledger → memory footprint, never "held-out composition" | **done** | "oblong" and "held-out composition" had 0 occurrences before; still 0. |
| 5 — cross-reference style | **done** | 0 `\S`, `Section~`, `Figure~`, `Table~`, `Appendix~` before a `\ref` except inside the Head's abstract `\TODO`, which the brief says to keep byte-identical (1 `\S\ref` + 1 `Table~\ref`, listed below). |
| pending resolved — "exact judge" → "the engine's judge"; state once where the judge is defined | **done** | Stated once in the Rule-validity definition (Sec. 1.2): "on the maze, the engine's judge, which uses integer arithmetic with no tolerance". The Shapes3D judge is a classifier, not the engine, so there the word "exact" was dropped ("the judge is the hash-pinned external classifier") and "the engine's" was **not** used. |

## Acceptance 1 — replacement table per rule, counts per draft

Full table (all 300+ rules, hits per draft, execution order) and every in-context replacement with the sentence it sat in: **`REPLACEMENTS.md`** (1,792 lines).
Totals per class:

| class | DRAFT-v2 | DRAFT-v2-style |
|---|---|---|
| store → n-CLU (generic) | 324 | 327 |
| rival → baseline (generic) | 190 | 190 |
| cross-references | 168 | 169 |
| corner → held-out (factor) combination | 104 | 104 |
| special (named objects, content sense, controls, artefact → record, title, first mention) | 94 | 94 |
| budget (by sense) | 57 | 57 |
| pre-registration / scored run | 52 | 52 |
| configuration labels (A1, A10, C-512, C-field, B, A, C2) | 48 | 48 |
| economic metaphors (one by one) | 43 | 44 |
| judge ("exact judge" → the engine's judge) | 33 | 33 |
| ledger → memory footprint | 24 | 25 |
| gate / charter | 17 | 17 |
| pointer comments added (numbers/labels that left the text) | 4 | 4 |

**Why the two drafts differ where they do** (every difference is a wording difference of the style paraphrase, not a missed hit — the residual scan below is 0 in both): store 324/327 (the style twin says "store" three more times in paraphrased sentences); xref 168/169 (the style twin has one more `\S\ref{sec:shapes}`: 3 against 2 in DRAFT-v2); metaphors 43/44 (the style twin says "what the naive unit does not buy" in the follow-up section, DRAFT-v2 has no such sentence); ledger 24/25 (style: "its byte ledger" in the Sec. 2 lead-in). Per-rule inequalities in REPLACEMENTS.md §1 are rules written for one draft's wording (e.g. `met-price-alternative` vs `met-price-alternative-style`); each pair sums to the same hit.

### Cross-references, replacements per kind (item 5)

| kind | DRAFT-v2 before | replaced | DRAFT-v2-style before | replaced | left (reason) |
|---|---|---|---|---|---|
| `\S\ref` → `Sec.~\ref` | 43 | 40 + 1 range (`\S\ref{a}--\S\ref{b}` → `Secs.~\ref{a}--\ref{b}`, 2 tokens) | 44 | 41 + 1 range | 1 each: abstract `\TODO` (Head-held, byte-identical by brief) |
| `Figure~\ref` → `Fig.~\ref` | 7 | 7 | 7 | 7 | 0 |
| `Figures~\ref` → `Figs.~\ref` | 3 | 3 | 3 | 3 | 0 |
| `Table~\ref` → `Tab.~\ref` | 80 | 79 | 80 | 79 | 1 each: abstract `\TODO` |
| `Tables~\ref` → `Tabs.~\ref` | 7 | 7 | 7 | 7 | 0 |
| `Appendix~\ref` → `App.~\ref` | 29 | 29 | 29 | 29 | 0 |
| `Appendices~\ref` → `Apps.~\ref` | 2 | 2 | 2 | 2 | 0 |
| `Section~` / `Eq.` / `Alg.` | 0 | — | 0 | — | none occur |

The brief's baseline (`\S\ref` 43, `Figure~\ref` 7, `Table~\ref` 80, `Appendix~\ref` 29 in DRAFT-v2) reproduces exactly; the plural forms (`Figures~` 3, `Tables~` 7, `Appendices~` 2) are extra and also swept. Lists keep their existing `and~\ref` / `, \ref` tails (`Figs.~\ref{a}, \ref{b} and~\ref{c}`).

### Economic metaphors, one by one (DRAFT-v2; the style twin's are in REPLACEMENTS.md §3)

| before | after | where |
|---|---|---|
| Covariance priced on the symmetric-triangle … | Covariance counted under the … | Tab. 2 caption |
| (the statistic) is priced on the symmetric-triangle … / One object is priced on the full-square … | is counted under … (×2) | App. A |
| The comparator's job is to price the alternative to a hole, not to lose. | … to measure what the alternative to a hole requires, not to lose. | Sec. 3.5 |
| which is the price it does not pay and the reason it has no exact deletion | which is the requirement it does not meet and … | Sec. 3.6 |
| 96% to 99.8% of the price arrives through the channel … | … of the loss arrives through … | App. requirement 9 |
| **Priced channel.** … the price law holds | **Dissipative channel.** … the retention law holds | App. prior derivations |
| $t_{\text{pay}}(50\%)$ = 7.992 / 5.994 / 7.992 half-lives | $t_{1/2}$ = 7.992 / 5.994 / 7.992 half-lives (pointer comment added) | App. retention law |
| The naive unit buys exact deletion … / It does not buy composition … | … provides … / … does not provide … | Leaving note |
| the store's read budget buys more latents | the n-CLU's read size holds more latents | Sec. 4.3 |
| a factor probe … buys nothing | … changes nothing | App. encoder |
| this is what it buys. | this is what it yields. | App. encoder |
| 1,001,188 B buys 200,238 macro-actions | 1,001,188 B provides 200,238 macro-actions | App. requirements |
| "the statistic buys deletion, not composition" | "the statistic provides deletion, not composition" | App. not-runs |
| while paying 81,312 to 84,216 bytes … the static read does not | while incurring … that the static read does not | Sec. 5.2 |
| while paying 81,312 B more read-side | while incurring 81,312 B more read-side | App. retention law |
| it pays for exactness in bytes and in the time of the book's refit | it incurs the memory and time cost of exactness, in bytes and in the book's refit | Leaving note |
| is paid before refusal arrives / of the cost is paid through | is incurred before … / of the loss is incurred through | App. retention law |
| its composition cost differs from eviction's | its composition loss differs | App. retention law |
| Exact is expensive in bytes. | Exact requires a large number of bytes. | Limitation 4 |
| where the ledger is cheapest / is not the cheapest point | where the memory footprint is smallest / is not the point of smallest footprint | Sec. 3.4, Limitation 4 |
| (evicting red) costs −2.0899 / −0.9101 / −6.7416 composition | changes composition by −2.0899 / … | Sec. 3.5 |
| (punching a hole) costs exactly +0.0000 | changes it by exactly +0.0000 | Sec. 3.5 |
| never cost-free erasure | never erasure without loss | Sec. 3.5 |
| a lower bound on what an item store costs | … on what an item memory loses | Sec. 3.6 |
| Switching the dynamics on costs composition / the dynamics cost composition / The same dynamics cost composition. | … lowers / lower composition (×3) | Sec. 5.2, leaving note, App. |
| deleting a conjunction costs it what the book knew | deleting a conjunction removes what the book knew | Leaving note |
| the only place the exactness claim costs content | … removes content | App. implementations |
| its costs are declared / Two costs are declared | its side effects are declared / Two side effects are declared | Sec. 4.2, App. read verb |
| at the cost printed in Tab. | at the memory cost printed in Tab. | Sec. 3.6 |
| two of the three scored runs were returned unspent | two of the three declared runs were not used | App. retention law |
| nine were spent / scored seeds were spent / spends (spending) no scored run | nine were used / seeds were used / uses (using) no test run | Sec. 4.1, App. maze |

`cost` stays (8 residuals, all technical: seconds, bytes, "memory and time cost", "cost line" of a byte/row display); "not cheap" stays (allowed). Each was read in its sentence; no blind regex was run over a metaphor (every metaphor rule is a literal phrase).

### Names introduced (item 2), defined once
- Title: `naive-CLU: Exactly Deletable Compositional Generation on a Frozen Encoder` ("Exactly" is item 13's, not touched).
- First mention: abstract "the *naive-CLU* (n-CLU)", intro "The naive-CLU (n-CLU) is a naive AI unit".
- Controls, defined once where introduced (Sec. 2, "Baselines: naive against naive"): *landscape removed* (atoms zeroed), *landscape shuffled* (atom values permuted within each well), *shuffled across wells* (well blocks dealt to other wells), *random atoms* (random atoms in place of the fitted ones) — definitions from the w12b/w13 records and the draft's own harness paragraph; used in plain words everywhere after (tables included; `lscape_*` headers too). `store_removed` → "n-CLU removed"; `rival_transformer`/`rival_lstm` → "transformer and LSTM".
- Configurations: Config A1 → "the base design" / "base n-CLU"; CLU-REL-A10 → n-CLU-REL; C-512 → "a wider design (latent 512, 8 atoms)"; C-field → "the field design"; Config B → "a/the match-bit design"; CMAZE-O Config A → "the autoencoder-latent design"; C2 → "a/the second latent design"; the pre-amendment CMAZE-T Config A → "the first design". Pointer comments give the old names.
- Non-n-CLU "stores": item store → **item memory**; learned store → **learned memory**; designed store → **designed memory**; "a store cannot have all three" (trilemma) → a memory; "these laws govern the store" (prior derivations) → the memory; the LDM definition "a store whose content is read by relaxing" → a memory. The physics "field store" (this study's n-CLU with its landscape switched on) → **field-mode n-CLU**. Content sense (plan: "the n-CLU's content"): the deletion measurement (a), the Tab. 4/Fig. 5 captions, "store mode deviation" → content mode deviation, "store content and the address book", CMAZE-O "Store and head", "own store and head", and the energy symbol $E_{\text{store},k}$ → $E_{\text{content},k}$.
- Gate: control gate / store gate → **control check** (strict gate → strict check; "left 25/53 store-gate eligible" → "eligible under the control check"; "a mandatory gate" → check). charter-C controls → landscape controls.
- artefact → record; pre-registered → declared (criteria, runs, bars, rules); scored run → test run; "static corner" (a limiting case, not a held-out combination) → static limit.
- Judge: the two-senses-of-exact sentence in Sec. 2 was **removed** (with the judge no longer called exact, only one sense remains; the brief forbids new "exact" text). "per-step exactness" (rollouts) → per-step accuracy; "is exact / no exact $n$-way row" in the CMAZE-T panel captions → judged valid / no valid.

### Meaning checks that changed wording beyond a token
- The quoted inference note ("We beat the naive fair rivals …; the CLU's own …") now reads baselines / n-CLU, so "The inference note, verbatim, travels with" → "The inference note travels with".
- "its artefact reads ``no scored run and no claim''" → "its record states that it is neither a test run nor a claim".
- "a $\mu^{-2}$ budget law" (shorts' retention triad) → "a $\mu^{-2}$ scaling law": the only budget replacement that renames a law; flagged for the Head.
- LACE/PoE "own budget(s)" → "own settings"; "a different budget" (CMAZE-1 harness tier, rivals at ~60,000 params) → "a different parameter scale"; "(retired-budget selection, declared)" → "(selected under a retired byte limit, declared)".

## Acceptance 2 — residual scan of rendered text

Command: `python3 pw1_draft_sweeps/scan.py <DRAFT>.tex --lines` (comments and `\label`/`\ref`/`\cite` keys excluded). Full before/after table and every residual line: **`RESIDUALS.md`**. After the sweep, both drafts:

| token | v2 | style | reason it stays |
|---|---|---|---|
| rival, store, gate, artefact, pre-regist*, A1, A10, C-512, improved, spoke, hub, wave, charter, ledger, bar-k labels, scored run, budget, spend/spent, price, buy, pay, expensive, $t_{pay}$, corner, held-out composition, oblong, exact judge, 10x, `landscape_*`/`random_atoms`, "Naive CLU", `Section~`/`Figure~`/`Appendix~` | 0 | 0 | — |
| `\S` + `Table~` before `\ref` | 1 + 1 | 1 + 1 | inside the abstract's CMAZE-T `\TODO` (Head-held; brief: byte-identical) |
| config* | 26 | 26 | all are the English word "configuration(s)" (and the table header "configurations"); the abbreviation "config"/"Config X" is 0 |
| cost | 8 | 8 | technical sense only (seconds, bytes, memory and time cost, the refusal "cost line") |
| cheap | 1 | 1 | "not cheap in memory or time" (allowed by the plan) |
| exact* | 114 | 114 | deletion sense and technical sense ("exact root", "exactly constant", "exact-match request"); item 13 (not mine) owns the deletion sense; 0 in the judge sense |
| bare CLU | 3 | 3 | the CLU class / the physics (settling) CLU of the follow-up, not the naive unit |
| bare `\ref` | 7 | 7 | second and later members of `Figs.~`/`Tabs.~`/`Secs.~` lists and ranges, plus the `\TODO` |
| "bar" (threshold) | kept | kept | numeric thresholds ("bar 0.10"); not in the banned list; the "bar k"-type label count is 0 |

**PDF text scan** (pypdf over the built PDFs, `DRAFT-v2.pdf.txt`, `DRAFT-v2-style.pdf.txt`): no banned token in body text or bibliography. Residual hits are all (B) inside figure images — `fig_ladder` ("rivals", "corner(s)", "CLU"), `fig_deletion` ("store = refit", "rival", "corner(s)"), `fig_crossover` ("Four-part byte ledger vs K"), `fig_task_shapes` ("pair corner"), `fig_encoder_adapt` ("tuples / corner"), `fig_maze_conditions` ("store gate", "rival", "A1", "CLU-REL-A10", "exact judge"), `fig_maze_rel_tuples` ("rival", "artefact order", "exact judge"), `fig_maze_readverb` ("exact judge"), `fig_cmazet_*` ("matched rivals", "rival"), `fig_task_cmazet` ("key spent", "event spend" — engine event name) — owned by the figure spokes (items 1, 10–12, 16); plus the LaTeX-generated "Table N:"/"Figure N:" caption labels (not cross-references) and the `\TODO`.

## Acceptance 3 — cross-reference residuals
0 occurrences of `\S`, `Section~`, `Figure~`, `Table~`, `Appendix~` before a `\ref` in either draft outside the Head's abstract `\TODO` (1 `\S\ref`, 1 `Table~\ref` there, kept byte-identical by the brief). Counts per kind above.

## Acceptance 4 — builds and checks (both drafts, built from the w17-paper layout: `iclr2027/`, `references.bib`, `figures/`)

| check | before | after |
|---|---|---|
| exit codes (pdflatex, bibtex, pdflatex ×3) | 0 | 0 |
| undefined citations / references | 0 / 0 | 0 / 0 |
| LaTeX errors / overfull boxes | 0 / 0 | 0 / 0 |
| bibliography items | 55 | 55 |
| pages DRAFT-v2 / DRAFT-v2-style | 50 / 50 | 50 / 50 |
| style diff-check (numbers, placeholders, TODOs, artefact paths, provenance comments and order, headings, labels, floats, preamble) | ALL EMPTY | **ALL EMPTY** (`stage-final-111006/STYLE-DIFF-CHECK.md`) |
| provenance checker findings | v2: 7 unsourced lines / 8 not-found; style: 7 / 9 | **identical** lists (same lines, same tokens); checked tokens 3,236 → 3,233 per draft |
| `\PLACEHOLDER` count and order | 3, in order | 3, same order; contents swept for vocabulary (store → n-CLU, rival → baseline, exact-judge → the engine's judge's, artefacts → records, scored → test run) identically in both drafts |
| `\TODO` | 4 | 4; the abstract CMAZE-T `\TODO`, the AI-use `\TODO` and the code-archive `\TODO` byte-identical; the composition-only byte-ratio `\TODO` swept (artefact → record, store → n-CLU, Table → Tab.) |

The 3 number tokens that left the text: "512" twice (the label "Config C-512"; "latent 512" stays) and "50" once (the "(50\%)" of $t_{\text{pay}}(50\%)$). Each line now carries a `% pw1-draft-sweeps (item 2)` pointer comment naming the old label and its record; a fourth pointer comment on the CMAZE-T base-design line records Config A1 / CLU-REL-A10.

## Acceptance 5 — install
Lock guard clean before install (live sessions: pw1-plot-maze, pw1-render-shapes; no ABORT pattern). Live drafts and PDFs were cmp-equal to the snapshot. Installed byte-identically (`installed/SHA256-install.txt`):

| file | before (snap-before/SHA256SUMS) | after (installed, cmp-equal to staged/built) |
|---|---|---|
| DRAFT-v2.tex | `93e796ce1f1f29fb68cf63e0b4f3b8ed3a911134bf00abbcdd6eaa89fd464948` | `a277235db5ad2d434fbca1efe438a8620dde6461fc0bcb6b38d4a2512ac3fb12` |
| DRAFT-v2-style.tex | `533ae4d3cf991fe828e3f149f39531fb1b8c8b1f2d1ca0ebce937f6b7646c40e` | `53063ef5818cc26f2b53dccd083b678435ab81d1a3f30a5b06911ee21f4e8095` |
| DRAFT-v2.pdf | `8357ef94536854d73daa3c47064252070a8172f4ea40b6b756e58ddaab945ddc` | `58004b79442c0c2365478ac1b9af69767ccf36b00f34ce348942a8040e2d548e` |
| DRAFT-v2-style.pdf | `b949211684893d428946385c2709b2baa5bc8842f816f59ccce21dcb538f9051` | `dd4092108468272e97ad63732ecb301ddc43668df5f01a7c97abf6e28b588f71` |

Only the two `.tex` and the two PDFs were installed; the stale `.aux/.bbl/.log/.out` in `w17-paper/` are from the previous build and were not replaced.

## For the Head / next spokes
- Figure images still say rival / store / corner / ledger / A1 / exact judge (list B above). Captions are swept; the images belong to items 1, 10, 11, 12, 16.
- Item 13 owns "exact" in the deletion sense (title "Exactly Deletable …", "Exact deletion", "n-CLU-scoped exactness" — the plan's own interim phrase).
- Choices taken (logged): field store → "field-mode n-CLU"; non-n-CLU stores → "memory"; LACE/PoE budgets → "settings"; control check for gate; "per-step accuracy"; $\mu^{-2}$ "scaling law".

## What did NOT run
- No figure image or figure script was touched (not mine by the brief); in-image vocabulary is unchanged.
- No bibliography entry was edited (the PDF bibliography carried no banned token).
- `\label` keys (e.g. `tab:ledger-kstar`, `tab:rivals-del`) were deliberately not renamed: they are not rendered, and renaming them would touch every other spoke's cross-references.
- The abstract's CMAZE-T `\TODO` (Head-held) was not edited, so 1 `\S\ref` and 1 `Table~\ref` remain inside it.
- No wording outside items 2, 3, 5 and the judge ruling (items 7, 8, 13, 14, 15, 17 untouched; maze material swept, not deleted).
- The stale `.aux/.bbl/.log` in `w17-paper/` were not rebuilt in place.
- pdftotext is not installed on batmac; the PDF text scan used `uv run --with pypdf` (an ephemeral environment), not a system tool.
- No measurement, fit or GPU run of any kind.

## Branch and bundle
Branch `pw1/draft-sweeps` commit `d80809f` (tooling in `pw1_draft_sweeps/`; no outputs committed). Bundle `pw1-draft-sweeps.bundle`, `git bundle verify` OK, head `d80809fa9bf7`.
