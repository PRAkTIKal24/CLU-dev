# W24 — repro-package (batmac; experiment-engineer; ZERO scored runs; NO pre-registration)

## The one job
Build the **minimal anonymous reproducibility package** for the ICLR paper: one folder containing the minimum code that reproduces what the paper reports, and nothing else — anonymous, standalone, proven to run.

## Paths (batmac)
| what | path |
|---|---|
| the paper (READ-ONLY copy) | `/Users/batmac/clu/tasks/w24/draft/DRAFT-v2.tex` (+ `figures/`, `DRAFT-v2.pdf`) |
| source repo (standalone clone, all wave/base branches) | `/Users/batmac/clu/worktrees/CHLU-w24-repro-package` (branch `w24/repro-package`; batmac-only branches under `refs/remotes/bm/*`, e.g. `bm/w16/maze-wm`) |
| figure/analysis scripts that never reached git | `/Users/batmac/clu/evidence/outputs/*` and `/Users/batmac/clu/outputs/w*/` (READ-ONLY) |
| data | Shapes3D `/Users/batmac/clu/data/gen/shapes3d/` (sha256 `0a0f6ed98baff276a50f3a081a7434d788da63cb135a98189b2a5b5769be1785`) |
| frozen instrument | `/Users/batmac/clu/stage/outputs/{gen-decoder-lift/winner, gen-external-classifier, spoke-a-rivals/cache_generation2 (seed_0 only)}` |
| **the only place you write** | `/Users/batmac/clu/outputs/w24-repro-package/` |
| **the package** | `/Users/batmac/clu/outputs/w24-repro-package/nclu-iclr-code/` — ⛔ **no `.git` inside, ever** |

The Head's final destination is a different machine; the Hub copies the folder there on harvest. Build it here.

## Steps
1. **MAP first** (`OUT/MAP.md`). Every table (20) and figure in `DRAFT-v2.tex`, main text AND appendices → the script(s) + config(s) that produced it → the data/weights it needs → the branch:path it comes from. Cells that are `\TODO`/placeholders or "declared not-run" are listed as such and need no code. Only what is on the map ships.
   Scope per the Head: Shapes3D pipeline (frozen encoder/decoder loading; the n-CLU — wells partition, content memory, address book, closed-form head, downdate/deletion, refusal display; judge classifier; hold-outs; K sweep; ladder; the four landscape controls; the four matched baselines with the read-path matching rule and training protocol; LACE/PoE ONLY if a table the draft prints needs them) · CMAZE-T (engine, request words, banks and hold-out families, the engine's judge, n-CLU and n-CLU-REL designs, baselines, controls, blind arms, lookup floors, deletion) · physics-section code ONLY for results the paper prints (the retention-law criteria table, the zero-dynamics-limit check). Everything else stays out.
2. **Assemble** into `nclu-iclr-code/`: a clean Python package (**rename the import package `chlu` → `nclu`**, env vars `CHLU_*` → `NCLU_*`), task-descriptive script names, neutral comments/docstrings, one entry point per paper table/figure. Data/weight locations come from CLI flags or env vars with a **relative** documented default — no absolute path anywhere. `pyproject.toml` with **pinned** versions (jax 0.9.0 / numpy 2.4.1 / equinox 0.13.4 and the rest as currently resolved), **no `authors` field**. `LICENSE` = placeholder text only (Head chooses). Do not ship `__pycache__`, `.DS_Store`, egg-info, results, run outputs, or trained baseline/store checkpoints.
3. **Weights:** measure every frozen encoder / decoder / judge-classifier file. ≤ 50 MB → ship under `weights/`. 50–100 MB → do NOT ship; list it for the Head's decision. > 100 MB → do NOT ship; ship its SHA-256 and a loader expecting it at a documented relative path.
4. **Anonymity scanner** (tool lives in `OUT/tooling/`, ⛔ never inside the package — it contains the banned list). Scan every file: contents (text), file and directory names, and binary metadata (PDF/PNG/npz/zip/ipynb metadata, `strings` on anything binary). Case-insensitive; word-boundary where a token is also an ordinary word. Seed list — extend it, never shrink it:
   - people/org: every author/committer name and e-mail in `git log --all` of the clone (extract them mechanically), `Pratik`, `Jawahar`, `forgis`, `forgisprova`, `@forgis.com`, `x10719pj`, `ORCID`, grant/funding words
   - machines/paths: `FORGIS_14`, `forgis-14`, `batmac`, `thinkpad`, `sshkey`, `tailscale`, `msys`, `/Users/`, `/home/`, `C:\`, `/c/chlu`, `farside`, `clu-dev`
   - tooling: `Claude`, `Anthropic`, `Co-Authored-By`, `CLAUDE.md`, `.claude`, `noreply@`
   - programme vocabulary: `spoke`, `\bhub\b` (not `GitHub`), `\bHead\b` capitalised as a role (the model's "head" in lowercase is legitimate — state how you separate them), `Advisor`, `PREREG`, `pre-regist`, `ledger`, `charter`, `wave`, `\bw[0-9]{1,2}[_\-/]`, `base/iter`, `iter[0-9]`, `CHLU`, `chlu`, `referee`, `placeholder`, `TODO`, `scored run`
   ⭐ **Positive control, mandatory:** copy the package to a temp dir, plant one token from each category (in a file body, a file name, and a PDF/PNG metadata field), and show the scanner catches every one. A scanner that has not been shown to fire proves nothing — this programme has already shipped tamper tests that could not fail.
   Rephrase legitimate collisions rather than whitelisting them (e.g. the LICENSE stub must not say "placeholder" or "TODO" — use "Licence to be added on publication."); the only whitelist allowed is the model's lowercase "head", and every whitelist rule is printed in the report.
   **Acceptance = an EMPTY report on the real package.** Print the full list in the report so the Head can extend it.
5. **Standalone proof** (in `OUT/proof/`, outside the package): copy the package to a temp dir, create a **fresh** venv (`uv venv` + install from the package's own `pyproject.toml`), with `PYTHONPATH` unset and ⛔ **without** sourcing `env_bm.sh`. Assert `nclu.__file__` resolves inside the temp copy. Run every entry point's documented smoke (tiny data, seconds). Then **one full reproduction per task to a printed number that matches a paper cell**: a Shapes3D headline cell and one CMAZE-T cell (pick cells reproducible from what is staged here — the feature cache is seed_0 only). Log command, runtime, printed number, and the paper cell it matches (tex line). A mismatch is reported as a mismatch with both numbers — ⛔ never tuned until it matches.
6. **README.md** in the package: install; data (Shapes3D download instructions + expected sha256; CMAZE-T banks generated by the shipped engine from seeds); one command per paper table/figure with expected runtime (measured where you ran it, estimated and marked as such where not); licence placeholder.
7. **MANIFEST.md** in the package: every file, size, sha256; total size. Generate it last, after the scan passes; re-scan after writing it.

## Deliverables in `OUT/`
`nclu-iclr-code/` · `MAP.md` · `scan-report.md` (banned list + positive-control proof + the EMPTY real-package result) · `proof/smoke.log` · `proof/repro.log` · `LOG.md` · `START_UTC.txt` / `END_UTC.txt` · **`w24-advisor-report.md`**: every paper table/figure with its command (or "placeholder / declared not-run"); everything deliberately left out and why; every weight file with its size and decision; the reproduction results; **decisions the Head must make** (50–100 MB files; licence text; anything ambiguous you resolved, with your choice).

## What would prove this wrong / not done
- A paper table/figure with printed numbers and no command → the package does not reproduce the paper. List it; do not hide it.
- A full reproduction that does not match its cell → report it plainly with both numbers.
- Any scanner hit on the real package, or a positive control that does not fire → not done.
- A `.git`, a result file, a checkpoint of a trained store/baseline, or an absolute path inside the package → not done.

## Hard limits
⛔ Zero scored runs; no ledger row (reproductions are re-executions of existing configs, not new scored runs). ⛔ No pre-registration of any kind — no bars, labels or predictions before a number. ⛔ No writes to any paper folder, `w17-paper/`, `figures/`, `paper-visuals/`, `TABLES.md`, the registry or the matrix — the draft copy is read-only. ⛔ No sub-agents. ⛔ Nothing needs committing to the repo; the clone is your read source. Tooling lives in `OUT/tooling/`.
