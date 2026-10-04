# Task: camels-pur-tripwire — the matched-byte tripwire on CAMELS-US PUR (ungauged *regions*, spatially blocked)

- **Agent:** `results-analyst` · **Output:** `.claude/outputs/camels-pur-tripwire.md` + artefacts under `.claude/outputs/camels-pur-tripwire/`
- **Read first:** protocol **§5** and **§7** · `.claude/outputs/camels-pub-tripwire.md` (**the sibling run — its harness, its controls and its mechanism section are the basis of this one**) · `.claude/outputs/camels-caravan-refs.md` **including §8** · your predecessor's code at `.claude/scratch/camels-pub-tripwire/code/`
- **Repo:** read-only on `chlu/`. **Zero worktrees, zero branches, zero tracked-code edits.** CPU-only, local-only.
- **Precondition:** none — everything it needs is already on disk. Spawnable now.

## ⭐ DIAL DECLARATION (protocol §7)
- **Dial:** *none — venue admissibility / instrument.* ⛔ No CLU cell, store, byte dividend or performance
  number is computed or implied. This measures the **classical** competition only.
- **Laundering control:** shuffled-order null + the companion rows, exactly as the PUB run carried them.
- **Falsifies (fires):** best in-budget, non-leaky store arm within 0.02 abs / 2% rel of **any** PUR reference.
- **Does NOT falsify:** the store losing to the LSTM references — expected, the metric-native ceiling, not news.
  ⛔ A fired tripwire falsifies the **venue**, not the CLU.

## The one job
Run the matched-byte tripwire on **Li et al. 2025's PUR protocol** (prediction in ungauged **regions**,
7 rounds, spatially blocked) and return a fired / did-not-fire verdict with the margin against **every**
PUR reference.

## Why — and the pre-registered question
CAMELS-US **PUB died**: a 1.41 MB untrained donor-basin store hit median NSE **0.6369** and fired against
7 of 20 references. The measured cause was that PUB holds basins out at **random**, leaving 71% of
"ungauged" basins a gauged neighbour inside 50 km; flow-copy NSE ran 0.822 under 10 km and **−0.157**
beyond 100 km. **PUR spatially blocks the folds, which is exactly what removes that neighbour.**

⚠ **But PUR is NOT automatically safer, and the plan should not assume it is.** The PUR *references are
also much weaker* — the bar drops with the store:

| | PUB | PUR |
|---|---|---|
| frontier (tri-forcing LSTM+δHBV) | 0.785833 | **0.700814** |
| Daymet LSTM+δHBV (forcing-matched primary) | 0.742270 | **0.644398** |
| Daymet LSTM | 0.702636 | **0.578365** |

⭐ **The question this spoke answers is therefore not "does the store get worse" — it is: DOES THE STORE
DROP FASTER THAN THE REFERENCES DROP?** If the donor arm collapses toward the >100 km bin it clears
comfortably. If it only sags to ~0.55 it fires again, because 0.578365 − 0.02 = 0.558 is the new bar
against the Daymet LSTM alone. **Pre-register a point prediction for exactly this ratio.**

## Order of work — binding

### 1. Pin the region assignment. ⛔ THIS IS THE BIGGEST RISK IN THE TASK — do it first and report it honestly.
PUB's folds were random, so a reconstruction was exchangeable with the published one and the spread across
three reconstructions was only 0.0167. **Spatial blocks are NOT exchangeable.** A different 7-way partition
is a *materially different task*, and a partition we happen to draw could be easier or harder than Li's by
much more than the margins we are measuring.
- Establish how Li/Feng define the 7 regions. Look in Li et al. 2025 and in Feng et al. 2023 (which
  originated the protocol). If the definition is published, **use it**.
- If it is **not** published: build the most defensible reconstruction you can, state the rule verbatim,
  and run **≥ 3 independent partitions**. Report the spread. ⛔ If the spread is comparable to the
  smallest margin, say so plainly — the verdict is then **INCONCLUSIVE**, which is a legitimate result and
  is far better than a confident wrong one.
- Report the reconstruction's realised geography: per-round held-out basin count, and the
  nearest-*train*-neighbour distance distribution. That distribution is the direct test of whether your
  blocking actually removed the neighbour. **If it did not, your partition is wrong, not the venue.**

### 2. Loader positive control — before any tripwire number is quoted
The Zenodo record already used for PUB also ships PUR, **confirmed by the Hub**:
`py999_PUR_simulations.npy`, **530,583,824 B**, md5 **`1e908d47d25522f883300be159713b72`** (record
10.5281/zenodo.16895228). Same shape as the PUB array. `py999_OBS.npy` and `py999_ids.npy` are already
local from the sibling spoke — reuse, do not re-download.

Reproduce, using **our** discharge, **our** alignment, **our** NSE code against **their** published PUR
simulations, under the metric convention the PUB run pinned (average members within a seed → per-seed
median over 531 → mean over the 3 seeds):

| target | published |
|---|---|
| LSTM + δHBV, Daymet | **0.644398** |
| δHBV, Daymet | **0.622278** |
| LSTM, Daymet | **0.578365** |
| LSTM + δHBV, All (frontier) | **0.700814** |

Source: Li et al. 2025 Table D3, publisher's machine-readable `hess-29-6829-2025-t07.xlsx`, Hub-parsed,
sha256 `1b592ed0560cbbd93c59f18840c11257648f27cee2a1b49c7b2f8d95edcfb939`, at
`.claude/scratch/hub-refs-verify/li_t07.xlsx`. Pre-register the tolerance. **Miss it and the loader is
wrong — stop and report.** ⚠ Re-verify the 6-config axis ordering rather than assuming PUB's holds.

### 3. Arms — carry the PUB run's lesson, do not re-derive it
- ⭐ **Build the DONOR form (one exemplar per training-fold basin, concurrent donor discharge payload).**
  The PUB run proved the literal "kNN over static attributes" DAY form is **degenerate (−0.0254)** while the
  donor form reached 0.6435. ⛔ Running only the DAY form would hobble the dangerous arm and clear the
  venue wrongly. That is the single easiest way to get this task wrong.
- Keysets, payloads, k, weighting, normalisers, selection and dtypes: reuse the PUB grid.
- ⛔ **Non-leaky normalisers only in the admissible set.** `oracle` (query basin's own flow statistics) stays
  quarantined and reported as an upper bound, exactly as before.
- **≥ 3 seeds on anything seeded.** Assert no test-region basin enters an exemplar set, and report the
  assertion count.

### 4. Budget — Head ruling, 2026-09-08
⭐ **The byte budget is PER FOLD: B = 1,966,080 B per round**, matching the reference, which trains a
separate same-size model per round. ⛔ Not 7× aggregate. Report the parameter-matched budget as the
secondary ladder point as before.

### 5. Companions, null, verdict
Day-of-year climatology · spatial-nearest-neighbour donor · regional mean flow · shuffled-order null.
Write `VERDICT.json` with `S`, `margin_abs`, `margin_rel`, `fires` for **every** PUR reference, plus `dies`.

## KILL RULE — pre-registered
> **The venue DIES if the best in-budget store arm comes within 0.02 absolute or 2% relative of the best
> strong reference.** Evaluated and reported against **EVERY** PUR reference.

If it fires: **stop, write up, escalate to the Head and the Main Advisor.** ⛔ Do not soften it and do not
hunt for a framing that saves the venue. If PUR falls too, the CAMELS family is out — that is a clean,
useful answer and it is the one the program needs.

## Acceptance
- Region assignment pinned or reconstructed with ≥3 partitions and its spread reported; the realised
  nearest-train-neighbour distribution reported as the check that blocking worked.
- Loader control passed within pre-registered tolerance against the four PUR targets above.
- Donor form built and run; leak assertions executed and counted.
- `VERDICT.json` covering every PUR reference. Every number traces to a file under
  `.claude/outputs/camels-pur-tripwire/`.
- **Anything not run is reported as NOT RUN — never a zero, never a failure.** An INCONCLUSIVE verdict is
  acceptable and must be declared as such rather than resolved by picking a favourable partition.

## What would prove this spoke wrong
- **Your blocking did not actually remove the neighbour** — check the distance distribution before you
  believe any margin.
- **The store drops but the references drop as fast**, so PUR fires too. That is a real outcome, not a
  failure; report it.
- **The partition spread swamps the margins**, making the verdict an artefact of your reconstruction.

## File ownership
- **Owns:** `.claude/outputs/camels-pur-tripwire/**`, `.claude/scratch/camels-pur-tripwire/**`.
- **Read-only:** `.claude/data/c3-camels/**`, `.claude/scratch/camels-pub-tripwire/**` (copy code into your
  own scratch; ⛔ do not edit the sibling's files in place), `.claude/scratch/hub-refs-verify/**`.
- ⛔ **Do not reuse `camels_tripwire.py` / `camels_local.py`** — they carry a known PUB leak (they
  un-standardise with the query basin's own discharge statistics, which biases toward a **false fire**).
  Build on `pub_donor.py` / `pub_day.py`, which are clean.
- ⛔ No sub-agents. ⛔ No cluster. ⛔ No CLU numbers.
