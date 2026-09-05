# PREREG — `harness-debt` (C2W4, experiment-engineer)

**Written and committed BEFORE any re-score was computed.** Base `main @ d4f56c8`, branch
`agent/experiment-engineer/harness-debt`, worktree `../CHLU-debt`.

**Prior knowledge declared (so nothing here is retro-fitted):** I have read `bprime-theory.md` T1
(which states the corrected law and reports 28/28 vs 24/28 and the `52.00 / 43.33 / 2.40 / 2.00`
numbers), `doctrine-repairs.md` §2.3 (which gives the theorist's `eps_acq` spec and the two named
predicted recoveries with `slope_acq = +7.8e-4`), and `bprime-fb4-gate.md` §C3 (which measures
`overload/base@s0`: `slope_loss = −4.0293e-2`, `slope_acq = +7.8431e-4`, state `clear`). I have also
inspected **one** cell of the recorded re-score artifact (`overload/ref3@s0`, 4 readings, all
`pre=post=0`) while establishing its schema. **I have computed no counts, no diffs and no
aggregates before writing this file.**

---

## A. D1/D2 — the byte law

| # | registered prediction | how derived |
|---|---|---|
| A1 | The corrected law `ratio = [A·(D+2) + d] / (d+m)` with `A = atoms_per_live_item`, `D = d + m + n_spec`, `d = addr_dim = 4`, `m = payload_dim = 1` reproduces the **measured** `byte_ledger.ratio` in **28/28** C2W1 cells to **0 ulp in exact rational arithmetic** | theorist T1.2; it is an accounting identity over the store's parameter leaves, not a fit |
| A2 | The **shipped** law reproduces it in **24/28**; the failures are exactly the **4 `manifold` cells** (`base@s0/s1/s2`, `ridge@s0`), the only cells with `n_spectator = 1` | the shipped denominator is `D`, the correct one is `(d+m)`; the two coincide iff `n_spec = 0` |
| A3 | On those 4 cells the published `closed_form_ratio` **43.33** becomes **52.00** — a shift of **+8.6667 = +20.0 %**, *identical on all four cells* (the shift is `A·(D+2)·[1/(d+m) − 1/D] + d·[1/(d+m) − 1/D]`, and `A` is the same for all four only if their atom budget is the same; if `A` differs the shift differs and **A3 is registered as "+8.6667 on the manifold/base cells"** with `ridge` free) | theorist T1.2 |
| A4 | `floor_note` on those 4 cells prints **2.00×** now and **2.40×** after; on the other 24 it prints **2.20×** before and after (**unchanged**) | `byte_ratio_law(1.0, 4, 1, n_spec)`: shipped `1·(D+2)/D + 4/D` = `8/6 + 4/6 = 2.00` at `n_spec=1`; corrected `(8+4)/5 = 2.40` |
| A5 | ⭐ **Bit-identity gate:** on all **24** `n_spectator = 0` cells the corrected law returns a value **bit-identical** to the shipped law (`==`, not `approx`) | algebraic: `D = d+m` ⇒ the two expressions are the same float expression up to association; registered as **bitwise equal**, and if only `1e-16`-equal that is a **reported deviation**, not a pass |
| A6 | The measured **min ratio 2.28×** over the 28 cells is **unchanged** (it is a measurement, not a closed form) | it comes from `ByteAccount`, which I do not touch |

**Falsifier (task §5).** If the corrected law fails to reproduce the measured `manifold` ratio to
0 ulp in rational arithmetic, the *theorem's statement* is wrong ⇒ first-10-lines report, same day.

## B. D3/D4 — monitor #6's `+eps_acq` half

**What I am landing** (task §2, verbatim): `slope_acq <= +eps_acq` with
`eps_acq = max(eps_acq_rel · scale_acq, eps_acq_floor if eps_acq_rel > 0 else 0)`,
`scale_acq = max|acq|` over the window, `eps_acq_rel` defaulting to **`1e-9`** — i.e. built the *same
relative way* as the shipped loss band, which is a **roundoff** band, not the theorist's
**resolution** band.

| # | registered prediction | how derived |
|---|---|---|
| B1 | ⭐ **Blocking:** `eps_acq_rel = 0.0` reproduces the current shipped predicate **bit-for-bit** on all **112** recorded readings (and `eps_rel = eps_acq_rel = 0` reproduces the pre-repair predicate bit-for-bit) | `eps_acq = 0` ⇒ `slope_acq <= +0.0` ≡ `slope_acq <= 0.0` |
| B2 | ⭐⭐ **At the shipped default `eps_acq_rel = 1e-9` I predict `0` cells flip and the post-repair monitor-#6 count stays `27`.** | `acq` is a **proportion in [0,1]** ⇒ `scale_acq ≤ 1` ⇒ `eps_acq ≤ 1e-9` **exactly**, for every reading. A flip needs a reading with the loss leg passing *and* `0 < slope_acq ≤ 1e-9`, i.e. an acquisition slope at the float64 roundoff floor while the write loss falls genuinely. The C2W2 artefact population has **both** slopes at `~1e-17` together (they are the same converged-window artefact), so the mixed case is expected to be empty. |
| B3 | ⛔ **The theorist's two predicted recoveries (`overload/base@s0`, `overload/reach_free@s0`) will NOT materialise at this band, and this is a prediction, not an excuse.** Their `slope_acq = +7.84e-4` is **~6 orders above** `eps_acq ≤ 1e-9`. The prediction of 2 recoveries was made for the theorist's `eps_acq = max(8u·Ya, 1/(n_probed·W)) ≈ 1/24 = 4.2e-2` — a **resolution** floor — and the shipped repair (both legs) implements a **roundoff** floor only. | `doctrine-repairs.md` §2.3 vs the shipped `ObjectiveDivergenceMonitor.__init__` |
| B4 | **Declared, labelled SENSITIVITY (not the shipped default, not tuning):** at an `eps_acq` in the theorist's resolution band (`~4.2e-2`) I predict the **2 named cells DO flip `no-trip → TRIP`**, and I register that **I do not know whether other cells also flip** — the number will be reported as measured, in both directions. This sweep is published as a diagnostic that *locates* the theorist's prediction; **it does not change the shipped default.** | §2.3's own re-score |
| B5 | **No `TRIP → no-trip` flip anywhere, at any `eps_acq ≥ 0`.** A dead band on the acq leg only *widens* the trip condition (`<= 0` → `<= +eps`, `eps ≥ 0`), so trips can only be added. Any observed removal ⇒ **falsifier fired, stop and report** (task §5). | monotonicity of the predicate in `eps_acq` |
| B6 | **No other monitor moves** — the diff is computed offline from *recorded* readings; the store is never re-run, so monitors #1–#5, #7–#13 are bit-identical **by construction**, and I will state it that way rather than claim it as a measurement. | task §2 D4, C2W2 D4 pattern |

## C. D5 (monitor #2's domain guard)
Registered as **conditional and likely NOT-RUN**: it is P1, it collides with `bprime-c6`'s live
re-location of `B = 0.33`, and the `max(λ, 1e-9)` clamp lives in `clu_system.py`, which I do not own.
If skipped it is reported as a **declared NOT-RUN with the reason**, never as a null.

## D. Verification bar registered in advance
`uv run pytest -q` green, delta vs **1061** accounted for reading-by-reading; `ruff check` clean;
both bit-identity gates green; a trip-state diff table with a stated reason per changed cell.
