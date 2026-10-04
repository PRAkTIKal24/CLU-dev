# `hub/refusal-2` — RUNNING LOG
⭐ Dated bullets, appended AS I GO. ⛔ Never written at the end, never retrospectively rewritten.
Corrections are NEW bullets.

- **Agent:** `autonomous-hub` · **Branch:** `hub/refusal-2` @ `945a3e9` (= `base/iter2`) ·
  **Worktree:** `/c/Users/PratikJawahar/chlu/CHLU-hub-refusal-2`
- **Machine:** FORGIS_14, Windows AMD64, CPU only, **4 threads**, ≤2 sub-agents.
- **START_UTC:** `2026-09-17T10:49:57Z` → **deadline `2026-09-17T22:49:57Z`** (12 h TRUE wall clock,
  COMMON §1b). Measured only by `START_UTC.txt` + the real clock via `elapsed.sh`.

---

## 2026-09-17T10:49Z — BOOT

- Read in order: `LAUNCH-COMMON.md` → `LAUNCH-hub-refusal-2.md` → `.claude/autonomous-hub-charter.md`
  → `.claude/AGENT_PROTOCOL.md` → `.claude/handover_context.md` §7 → `.claude/tasks/hub-refusal.md`,
  as re-posed by the addendum. ⛔ Nothing else from the campaign archive, except the three
  PREDECESSOR hub reports, which COMMON §3 makes **inherited fact** and explicitly permits.

- **⭐ COMMON §0 — THE OVERRIDE, quoted as required.** `WORKSPACE-READY.json` reads
  `"status": "NOT-READY"` and `verify_hub_workspace.sh` prints *"⛔ Do not launch a hub."*
  A human has overridden this in writing:

  > **Override recorded (Hub, 2026-09-16): launch `hub/refusal`, `hub/composition` and
  > `hub/generation-table` on v2 despite
  > `test_documents_are_decoded_as_utf8_regardless_of_the_platform_locale`.**
  > The failure is in the test's `_WindowsFlavourPath` mock, not the bundle. The mock's `read_text()`
  > calls `read_bytes().decode()`, which does **not** fold universal newlines, while real
  > `Path.read_text()` opens in text mode and does. On Windows the fixture is written by a real
  > `write_text()`, which expands `\n → \r\n`, so the mock counts two characters per newline and
  > `chars` exceeds `len(text)` by exactly the newline count. **Nothing to do with encoding.** It
  > passes on macOS only because macOS does not expand newlines. The mock is fixed on the Mac and
  > ships in the next cut; the UTF-8 behaviour it guards is unaffected and still enforced by the
  > shipped code's explicit `encoding="utf-8"`.

  ⛔ I record that this override covers **exactly that one test** and is **not** a licence to wave
  through any other red check. ⛔ `SKIP_LOAD=1` / `SKIP_TEST=1` are not a pass and I will not use them.

- **⭐ WRITE PATH — a deliberate deviation from the slug, and why.** COMMON §5 says write only to
  `$CHLU_HUB_OUTPUTS/hub-<slug>/**`, and my slug is `hub-refusal`. But
  `$CHLU_HUB_OUTPUTS/hub-refusal/` **is the predecessor's directory** and COMMON §3 makes its
  contents *inherited fact*; §0.1 says **I NEVER DELETE**. Writing my `LOG.md` there would destroy
  an inherited fact. ⭐ I therefore write to **`$CHLU_HUB_OUTPUTS/hub-refusal-2/`** — the same `-2`
  suffix, for the same stated reason, that COMMON §1a already applies to my worktree and branch.
  ⭐ Verified `--return-pack hub-refusal-2` is legal: `bundle_hub_workspace.sh:697` requires only a
  bare name. ⛔ Nothing of the predecessor's is touched, moved or deleted.

- **Environment, verified (COMMON §2):** `CHLU_ROOT=/c/Users/PratikJawahar/chlu/chlu-bundle-v2`
  (pin `4125fce7…`), every `CHLU_*` path contains `chlu-bundle-v2`, `PYTHONUTF8=1` exported
  (load-bearing — without it the first encode dies at `gen_bridge.py:915` on `⭐` under cp1252),
  interpreter `$CHLU_ROOT/.venv/Scripts/python.exe` = **Python 3.11.16, jax/jaxlib 0.9.0,
  numpy 2.4.1, CPU**. ⛔ The v1 bundle at `…/chlu/chlu-hub-bundle` is not mine and I will not read it.
  Threads pinned `OMP=MKL=OPENBLAS=4` in `_env.sh`, sourced by every unit.

- **Base confirmed by measurement, not by assertion.** `git diff --stat 8e36bc9 945a3e9` =
  **952 insertions, 2 deletions** over 6 files: the five reference-read files
  (`gen_address.py`, `gen_factored_read.py`, `gen_partition.py`, `gen_sensitivity.py`,
  `tests/test_gen_factored_read.py` = 940 insertions, 0 modified) and the §2.2(b) gate patch
  (`exp_gen_cell.py`, +14/−2). ⭐ Matches COMMON §1a exactly. `exp_gen_store.py` is **not** in the
  diff ⇒ `ULP_BOUND`, `FRAC_BOUND` and `_assert_cache_reencodes` are untouched, as promised.

- **The clock, the budget and the gates I am under, restated so I cannot drift off them:**
  - **≤ 1 scored run this iteration, SEED 0 ONLY** (COMMON §1c). 0/3 spent on this target on any
    machine. A seed-0 number is reported as *"scored 1/3 — seed 0 only, 3-seed pending full caches"*.
  - ⛔ **Seeds 1 and 2 are not attemptable here** — `scope="used"` caches; covering the cell needs
    ~32,200 new rows = a re-encode = forbidden. ⛔ I will not attempt them.
  - ⛔ **A shortfall > 0 means STOP and report. It never means encode.** If I find myself wanting to
    raise `ULP_BOUND`, widen `FRAC_BOUND`, or pass `will_extend=False` with a shortfall > 0, that is
    the exact loosening the ruling forbids and it is my signal to stop.
  - ⛔ **The GO/NO-GO comes first:** `S` on the predecessor's synthetic analogue at rungs A and B,
    seed 0. **If `min(AUROC_oor, AUROC_dup) < 0.6` there, I DO NOT SPEND** — I report the number and
    stop the refusal line for this iteration. That is a complete and acceptable outcome.
  - ⛔ **"Truthfully" is the operative word.** Stopping at 12 h with *"not solved — here is exactly
    how far I got and what blocked me"* is a SUCCESS. Manufacturing a result to fill the window is
    the one unrecoverable failure. I name the deadline pressure here, at hour 0, so I recognise it
    at hour 11.

- **⛔ The old criterion is DEAD and I will not re-derive it.** Inherited fact from
  `hub-refusal/REPORT-hub-refusal.md`: settled energy of the argmax/MALA read is **structurally
  blind** — the address book is a simplex over stored atoms and *a simplex has no outside*;
  `out_of_range` has the **sign backwards monotonically over 13× extrapolation** (`r_t` −16.2 →
  −87.1, frac-satisfied 0.688 → 0.891); and the store is **decorative** for it
  (`store_removed` 0.7809 ≥ full-store 0.7511 at the launch; 0.5888 ≥ 0.5825 settled).

- **NEXT:** read the predecessor's `scorer/refusal_lib.py` and `term_experts`; establish what the
  **linear read** is in the shipped code; then build and run the GO/NO-GO. ⛔ No scored spend
  before that number exists.

## 2026-09-17T10:58Z — PREREG-1 FILED, BEFORE ANY MEASUREMENT

- **`PREREG-hub-refusal-2-1.md` filed.** ⛔ Filed before any statistic was computed under this
  criterion and before any invalid row of any kind was scored by it. It carries the **mandatory
  disclosure verbatim** (§0), the dial declaration, the criterion, the threshold rule, the four
  companions **rejected in advance**, the three mandatory controls, the bar, the go/no-go, the seven
  falsifiers and the flag-provenance table.

- **⭐⭐ THE FINDING THAT MAKES THIS CRITERION IMPLEMENTABLE WITHOUT TOUCHING THE PIN.**
  The "linear read" is **already shipped**: `gen_landscape.launch_state` has
  `launch_mode="posterior_mean"` (`gen_landscape.py:1028`), one of four `LAUNCH_MODES`
  (`gen_landscape.py:123`), reachable through the declared config knob
  `GenDynamicalReadConfig.launch_mode` (`config.py:6196`, default `"clamp_atom"`).
  ⭐ **I flip one declared config value. ⛔ I edit no manifest-pinned file.**

- **⭐ And `store_energy` is exactly the right shape for the claim.** Read at
  `gen_landscape.py:895`: `E_store(x) = Σ_k −σ_k²·logsumexp_i(−‖x_k − a_ki‖²/2σ_k²)`. That is a
  **soft squared distance to the nearest stored atom**, per well. Two consequences, both stated
  before measuring:
  - under the **argmax** read `x_k` **is** an atom ⇒ some `‖x_k − a_ki‖ = 0` ⇒ `E_store` is pinned at
    its floor and carries **zero** validity information. ⭐ That is why the predecessor measured
    distance `0.0000`, and why this criterion is impossible under the shipped read.
  - under the **linear** read `x_k = Σ_i P_i a_ki` is a **combination**, which lands in the void
    between basins exactly when the posterior is spread over atoms that are **far apart in feature
    space** — i.e. mutually inconsistent. ⭐ **The read change is what makes refusal possible at all.**

- **⛔ The risk I am NOT hiding from, named at hour 0.** `E_store` at a blend depends on BOTH the
  posterior's spread (address book — survives shuffling) AND the blended atoms' separation (content —
  destroyed by shuffling). ⚠ **If the signal is mostly the former, `store_removed` will NOT collapse
  and PREREG-1 F1 fires — exactly as it did for my predecessor (0.7809 ≥ 0.7511).** That is why §4.2
  adds the **physics-free readout margin** (`−mean_k max_i P`, `−mean_k (P₁−P₂)`): it isolates the
  address-book-only component so the two cannot be confused. ⛔ I am not assuming the control passes.

- **Scorer built, under `hub-refusal-2/scorer/`:**
  - `inherited/` — the predecessor's nine scorer files copied **byte-for-byte**; `sha256sum` diff
    against the originals is **empty** and `SHA256SUMS.txt` records every digest.
    ⭐ Copied rather than imported in place so I cannot write into an inherited artefact.
  - `linear_read.py` — the linear read, the criterion, the companions, the controls and the
    R1 paired-bootstrap. ⭐ `store_energy`, `launch_state`, `threshold_from_valid`, `refusal_rates`
    and `binary_auroc` are **called**, ⛔ never re-implemented. `term_experts` is inherited verbatim.
  - `go_no_go.py` — PREREG-1 §6, resumable per rung, writing a `runs/LEDGER.jsonl` line per unit.
  - ⭐ `assert_is_shipped_linear_read` proves my per-term read equals the shipped
    `launch_state(..., "posterior_mean")` on valid row-derived requests — the proof that this is the
    shipped arm and not a different one.
  - ⭐ `assert_quantile_is_shipped` reads `q` **from the config** and aborts if it ever moves, so `q`
    can never quietly become a number I chose.

- **NEXT:** run the go/no-go at rungs A and B, seed 0. ⛔ No scored spend until that number exists.
