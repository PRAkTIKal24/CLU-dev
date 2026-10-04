# scan-report — anonymity scan of `nclu-iclr-code/`

Scanner: `tooling/scan.py` (outside the package; it holds the banned list). Target of the real scan: `nclu-iclr-code` (196 files). Command: `python3 tooling/scan.py nclu-iclr-code --git <source clone>`.

## 1. Result on the real package

**Hits: 0.** The report is EMPTY.


## 2. What is scanned, and how

* every path component (file and directory names);
* every file that decodes as text (UTF-8), whole content, line by line (code, JSON, Markdown, TOML, the licence stub);
* every binary file: printable runs of >= 4 bytes (as `strings -n 4`), plus
  - PNG `tEXt`, `zTXt` (zlib-decompressed) and `iTXt` chunks, decoded;
  - PDF: every stream decompressed, the Info dictionary (incl. UTF-16 hex strings) and XMP;
  - ZIP / NPZ: member names, archive comment, every member decompressed and scanned recursively; gzip decompressed;
  - array files (`.npy`, and the equinox `.eqx` weight files, which are concatenated `.npy` blobs): every header is scanned as text; string/object arrays are scanned in full; the numeric payload is checked to be exactly the declared arrays (shape x itemsize), and any byte not accounted for is flagged (`UNACCOUNTED-BYTES`, itself a banned token) and scanned in full. Raw float payloads are not text: before this rule, `strings` on the six weight files produced random 2-4 character runs matching the short patterns (e.g. `W6/`, `M:/`, `ERc`); the positive control below shows that bytes appended to a weight file are caught.
* matching is case-insensitive, except the role word `Head` (below).

## 3. The banned list (seed list of the brief, extended; never shrunk)

**scrub scars**: `\(\s*[,;:]`, `,\s*,`, `\.\s*:\s`, `\bon -[0-9]+\b`, `(?<![\w\])}'\"`\\])'s\b`, `\b(spent\|run\|fitted\|landed\|used\|declared\|fixed) (in\|at\|by\|from\|per\|under)[;,)]`, `PROTOCOL-§`, `[A-Z]{3}-\)[case-sensitive]`, `^\s*init\s*\.`, `\binit  `, `\(\s+[ivx]{1,4}\)`, `\b(the\|a\|an\|of\|to) \1\b`, `\bthe (that\|which\|who\|of\|and)\b`, `\(\s+\)`, `[A-Za-z0-9] [,;](?=\s)`, `\(\s*-\s*\)`, `--\s*\)`

**people/org**: `Pratik`, `Jawahar`, `Pierini`, `forgis`, `forgisprova`, `@forgis\.com`, `x10719pj`, `ORCID`, `PRAkTIKal24`, `55033105`, `batmac`, `Copilot`, `198982749`, `\bgrant(s\|ed)?\b`, `\bfund(ed\|ing\|er)\b`, `acknowledg`, `\bsponsor`, `\bEPSRC\b`, `\bUKRI\b`, `\bNSF\b`, `\bERC\b`, `\bDFG\b`, `\bARIA\b`, `\bSNSF\b`, `Royal Society`, `Leverhulme`

**machines/paths**: `FORGIS_14`, `forgis-14`, `batmac`, `thinkpad`, `sshkey`, `tailscale`, `\bmsys`, `/Users/`, `/home/`, `C:\\`, `\b[A-Za-z]:/`, `/c/chlu`, `farside`, `clu-dev`, `tailfb0a84`, `\.ts\.net`, `/private/`, `/var/folders`, `~/clu`, `Desktop/CHLU`, `\bWindows-1`, `PowerShell`, `\bhost\"?\s*:\s*\"`

**tooling**: `Claude`, `Anthropic`, `Co-Authored-By`, `CLAUDE\.md`, `\.claude`, `noreply@`, `\bOpus\b`, `\bFable\b`, `\bChatGPT\b`, `\bGPT-?[0-9]`, `\bCodex\b`, `copilot-swe`, `\bLLM\b`, `experiment-engineer`

**programme**: `spoke`, `(?<![a-z])hub(?![a-z])`, `\bHead\b[case-sensitive]`, `Advisor`, `PREREG`, `pre-?regist`, `ledger`, `charter`, `wave`, `\bw[0-9]{1,2}[_\-/]`, `(?<![a-z0-9])w[0-9]{1,2}[a-z]?(?=[_\-/.:]\|\b)`, `base/iter`, `iter[0-9]`, `CHLU`, `chlu`, `referee`, `placeholder`, `TODO`, `scored run`, `amendment`, `\bruling\b`, `registry`, `\bN[0-9]{3}\b[case-sensitive]`, `\bMF[0-9]{1,2}\b`, `\bc[0-9]w[0-9]`, `\bpw[0-9]\b`, `bprime`, `SCORED-`, `\bcampaign\b`, `Montero pool`, `PW-?[0-9]`, `(?<![a-z])pw[0-9]`, `\bitem [0-9]+(\.[0-9]+)*\b`, `\bruled\b`, `(?<![A-Za-z])-ruled`, `20[0-9]{2}-[01][0-9]-[0-3][0-9]`, `retired_by`, `\bR[0-9]{1,3}(\.[0-9]+)?\b[case-sensitive]`, `AMENDMENT-?[0-9]`, `\bnot drawn [0-9]`, `\bretired\b`

**people/org, extracted mechanically from `git log --all` of the source clone** (author and committer names and e-mails, co-author trailers; every full address kept, its local part and domain added as separate tokens; the ordinary words `info`, `local`, `noreply`, `github`, `claude` are not kept as separate tokens but `noreply@` and `Claude` are on the seed list): `198982749+Copilot`, `198982749+Copilot@users.noreply.github.com`, `55033105+PRAkTIKal24`, `55033105+PRAkTIKal24@users.noreply.github.com`, `Claude Fable 5`, `Claude Opus 4.8`, `Claude Opus 4.8 (1M context)`, `Claude Opus 5`, `Claude Opus 5 (1M context)`, `Claude Opus 5.5`, `PRAkTIKal24`, `Pratik Jawahar`, `anthropic.com`, `batmac`, `batmac.tailfb0a84.ts.net`, `batmac@batmac`, `batmac@batmac.tailfb0a84.ts.net`, `chlu-w8-downdate`, `chlu-w9-harness`, `chlu.local`, `copilot-swe-agent[bot]`, `experiment-engineer`, `experiment-engineer@users.noreply.github.com`, `forgis.com`, `hub-erasure`, `hub-factor-wells`, `hub-retention`, `hub-single-design`, `hub-w6-potentials`, `hub-w9-maze`, `hub/field-store`, `hub/placement`, `hub/read`, `hub@chlu.local`, `hub@local`, `info@forgis.com`, `noreply@anthropic.com`, `noreply@github.com`, `w5 hub/delete`, `w8-encoder-loop`, `w8-env-scout`

Word boundaries: `\bhub\b` is widened to `(?<![a-z])hub(?![a-z])` so that `hub_x` and `NCLU_HUB` are caught while `GitHub` is not; `\bw[0-9]{1,2}[_\-/]` is kept and widened by `(?<![a-z0-9])w[0-9]{1,2}[a-z]?(?=[_\-/.:]|\b)` so that `exp_w8_`, `W16`, `w21C` are caught; `\bN[0-9]{3}\b` (record codes) is matched case-sensitively, because a newline escape before a number (`\n614,400` in a tick label) otherwise matched.

## 4. Whitelist (the only one)

* lowercase 'head' (the model's request-indexed head, e.g. 'closed-form head', 'refuse-token head') is legitimate: the role word is matched case-sensitively as \bHead\b, so 'head', 'heads', 'HeadSpec' never match and the capitalised role word always does.

No other token is whitelisted. Legitimate collisions were rephrased instead: the licence stub reads "Licence to be added on publication."; identifiers containing `ledger` became `footprint`, `registry` became `index`, `placeholder` became `stand_in`; the paper label `ledger-kstar` is written "the K* footprint table" in the README.

## 5. Positive control

The package was copied to a scratch directory and one token of every category was planted in a file body, a file name, a PDF metadata field (Author) and a PNG metadata field (tEXt Author); additionally a compressed PNG zTXt chunk, bytes appended to a weight file, and an NPZ member name. For AMENDMENT-3 four body plants were added: `PW-1` in a comment, `PW1_X` as an identifier, the scar `(, 2026-09-25)` in a docstring, and a bare scar `(, )`. The scanner was run on the planted copy.

| placement | category | file | token | caught | rule(s) |
|---|---|---|---|---|---|
| body | people/org | `nclu/plant_body_people.py` | `Jawahar` | YES | `Jawahar` |
| body | machines/paths | `nclu/plant_body_machines.py` | `thinkpad` | YES | `thinkpad` |
| body | tooling | `nclu/plant_body_tooling.py` | `Anthropic` | YES | `Anthropic` |
| body | programme | `nclu/plant_body_programme.py` | `PREREG` | YES | `PREREG` |
| body (AMENDMENT-3) | programme | `figures/plant_pwdash.py` | `PW-1` | YES | `PW-?[0-9]` |
| body (AMENDMENT-3) | programme | `figures/plant_pwident.py` | `PW1` | YES | `(?<![a-z])pw[0-9]`, `PW-?[0-9]` |
| body (AMENDMENT-3) | scrub scars | `figures/plant_scar.py` | `(, 2026-09-25)` | YES | `20[0-9]{2}-[01][0-9]-[0-3][0-9]`, `\(\s*[,;:]` |
| body (AMENDMENT-3) | scrub scars | `figures/plant_scaronly.py` | `(,` | YES | `\(\s*[,;:]` |
| name | people/org | `figures/forgis_notes.txt` | `forgis` | YES | `forgis` |
| name | machines/paths | `figures/batmac_run.sh` | `batmac` | YES | `batmac` |
| name | tooling | `figures/claude_helper.py` | `claude` | YES | `Claude` |
| name | programme | `figures/spoke_outputs` | `spoke` | YES | `spoke` |
| pdf-metadata | people/org | `figures/plant_meta_people.pdf` | `Jawahar` | YES | `Jawahar` |
| png-metadata | people/org | `figures/plant_meta_people.png` | `Jawahar` | YES | `Jawahar` |
| pdf-metadata | machines/paths | `figures/plant_meta_machines.pdf` | `thinkpad` | YES | `thinkpad` |
| png-metadata | machines/paths | `figures/plant_meta_machines.png` | `thinkpad` | YES | `thinkpad` |
| pdf-metadata | tooling | `figures/plant_meta_tooling.pdf` | `Anthropic` | YES | `Anthropic` |
| png-metadata | tooling | `figures/plant_meta_tooling.png` | `Anthropic` | YES | `Anthropic` |
| pdf-metadata | programme | `figures/plant_meta_programme.pdf` | `PREREG` | YES | `PREREG` |
| png-metadata | programme | `figures/plant_meta_programme.png` | `PREREG` | YES | `PREREG` |
| png-zTXt (compressed) | tooling | `figures/plant_ztxt.png` | `Co-Authored-By` | YES | `Co-Authored-By` |
| bytes appended to .eqx | people/org | `weights/classifier/seed_0/external_classifier.eqx` | `forgisprova` | YES | `forgis`, `forgisprova` |
| npz member name | machines/paths | `weights/plant.npz` | `tailscale` | YES | `tailscale` |

**23 of 23 plants caught; 0 missed.**

## 6. Hub re-scan findings and fixes (AMENDMENT-3)

The Hub's independent re-scan of the harvested package found what the scanner of sections 1-5 missed: review-item IDs (`PW-1` x18, `PW1` x4, e.g. `PW1_BANNED`), deletion scars (`(, 2026-09-25)` x4, `(, 2026-09-14)`, `-ruled 2026-09-25`), history fields (`retired`, `retired_by`, `rival_sets_retired`) and ISO development dates (21). The old rule `\bpw[0-9]\b` could not match `PW-1` (hyphen) or `PW1_` (underscore is a word character).

**Rules added** (all listed in section 3; nothing was removed): programme: `PW-?[0-9]`, `(?<![a-z])pw[0-9]`, `\bitem [0-9]+(\.[0-9]+)*\b`, `\bruled\b`, `(?<![A-Za-z])-ruled`, ISO dates `20[0-9]{2}-[01][0-9]-[0-3][0-9]`, `retired_by`, `\bretired\b`, review-round references `\bR[0-9]{1,3}(\.[0-9]+)?\b` (case-sensitive), `AMENDMENT-?[0-9]`; a new category **scrub scars** for the stubs a deletion leaves (a punctuation run where a word was): `\(\s*[,;:]`, `,\s*,`, `\(\s+\)`, a space before `,`/`;`, `(-)`, `--)`, `.:`, `on -<n>`, a doubled article (`the the`), an orphan possessive (`'s` with no word before it), `( ii)`, `PROTOCOL-§`, `ABC-)`, the stub docstring `init  .` and a dangling preposition (`spent in;`). Two of these new rules were narrowed while they were written, before any result was taken from them, because they matched working code: `[A-Za-z]-\)` matched a regex look-behind `(?<!n-)` and became `[A-Z]{3}-\)`; a bare `(in|at|per)[;,)]` matched variables named `at`/`per` and became the verb-plus-preposition form.

**Before / after.** The final scanner run on the proof-C copy of the harvested package (195 files): **216 hits**. On the rebuilt package (196 files incl. MANIFEST.md): **0 hits**.

| category | rule | hits before | hits after |
|---|---|---|---|
| programme | `\bR[0-9]{1,3}(\.[0-9]+)?\b[case-sensitive]` | 77 | 0 |
| programme | `PW-?[0-9]` | 28 | 0 |
| scrub scars | `(?<![\w\])}'\"`\\])'s\b` | 17 | 0 |
| programme | `20[0-9]{2}-[01][0-9]-[0-3][0-9]` | 16 | 0 |
| programme | `\bitem [0-9]+(\.[0-9]+)*\b` | 14 | 0 |
| programme | `\bretired\b` | 13 | 0 |
| programme | `(?<![a-z])pw[0-9]` | 11 | 0 |
| scrub scars | `\binit  ` | 10 | 0 |
| scrub scars | `\(\s*[,;:]` | 9 | 0 |
| programme | `retired_by` | 6 | 0 |
| scrub scars | `[A-Za-z0-9] [,;](?=\s)` | 6 | 0 |
| scrub scars | `\(\s+[ivx]{1,4}\)` | 2 | 0 |
| scrub scars | `\b(the\|a\|an\|of\|to) \1\b` | 2 | 0 |
| scrub scars | `\bon -[0-9]+\b` | 1 | 0 |
| programme | `\bruled\b` | 1 | 0 |
| programme | `(?<![A-Za-z])-ruled` | 1 | 0 |
| scrub scars | `PROTOCOL-§` | 1 | 0 |
| scrub scars | `[A-Z]{3}-\)[case-sensitive]` | 1 | 0 |

The 77 review-round hits include 4 functional labels that were not review references (`"R0"` a rung label, `"R2"`/`"R3"` unused router names in a config default) and the module alias `gen_deletion as R7` (42 hits); the other 31 are review-round references in prose; they were renamed (`rung_0`, `router_2`, `router_3`, `GD`) so that the rule can stay unconditional.

**Fixes (all in the build transform `tooling/transform.py` / `tooling/names.py`, then a full rebuild; nothing edited by hand):**

* review-item and review-round references dropped with the preposition that introduced them ("per AMENDMENT-1", "(R26.3)", "PW-1 item 10:"); a possessive reference becomes "the" ("R8's declared embedding" -> "the declared embedding");
* identifiers renamed: `PW1_BANNED` -> `FIGURE_TEXT_BANNED`, `PW1_TEXT` -> `FIGURE_TEXT_SEEN`, `check_vocab_pw1` -> `check_figure_vocabulary`, figure stems `rc_pw1_shapes_*` -> `shapes_*`, `retired_byte_budget` -> `prior_byte_budget` (a functional quantity, 688,128 B, read by the rival-budget notes: kept, renamed);
* figure data files (`figures/data/*.json`): `retired`, `retired_by` and `retired_rows` fields removed (`retired_rows` is read by no code: dropped), `_doc` fields removed (read by no code), the status value `retired` -> `not_drawn` and the output key `rival_sets_retired` -> `rival_sets_not_drawn` (read by `results_figures_b.py`, kept functional under the neutral name);
* ISO development dates removed from strings, docstrings and data (a whole-string timestamp becomes "not recorded");
* scar repair after every deletion (`descar`): empty or comma-led parentheses, doubled articles/spaces, `.:`, orphan possessives, while never touching the edges of an f-string fragment (an earlier draft of this step glued `{WHITENING_D_LIMIT}` to the next word; caught in review of the diff and fixed before the proof);
* words that contain a banned token are replaced whole instead of cut (`spoken` -> `expressed`, `scored runs` -> `runs`, `pre-registered` -> `declared in advance`), because cutting left stubs such as `n`.

A structural check compared every Python file of the proof-C copy and the rebuilt package with string contents masked: the ASTs are identical apart from the renames above and the two AMENDMENT-2 report fixes.

