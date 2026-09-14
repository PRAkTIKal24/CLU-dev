# c3-venue-loaders — production loaders for the two ADOPTED Track-B venues (CAMELS-US, N-CMAPSS DS02)

**Campaign 3. Agent:** experiment-engineer. **ONE worktree.** ⏳ **QUEUED — ⛔ do NOT spawn until the Track-A ladder has started** (Head+Advisor sequencing, 2026-08-13). Written now so it is ready; the Head spawns it when the ladder is under way.
Branch **`agent/experiment-engineer/c3-venue-loaders`** off `main` at the then-current C3 head (the Hub names it at spawn).
Writes `.claude/outputs/c3-venue-loaders.md`. **Budget:** ≈ 2 days. **Two venues, §1 first.**

**Binding documents:** ⭐ `.claude/outputs/c3-trackb-tripwire.md` **IN FULL** — it adopted these venues and found the data defects below · `.claude/outputs/c3-benchmark-scout/trackB-scorecard.json` rows 1–2 (`csf3` blocks) · `chlu/data/enwik8.py`'s docstring (the staging contract you copy) · `chlu/data/industrial/base.py::download_file`.

---

## ⭐ DIAL DECLARATION (protocol §7)

- **Dial:** **none — data plumbing.** ⛔ No claim, no venue result, no CLU number.
- **Falsifies the task:** a loader that races under array tasks; a split that leaks; a defective forcing product reaching a model; an N-CMAPSS number produced without its mandatory reference row.
- **Does NOT falsify:** either venue turning out to be hard. Both are **adopted**; that is settled.

⭐ **You write these FRESH against `chlu/data` conventions.** ⛔ **The tripwire spoke's throwaway loader is NOT promoted** (explicit Head ruling) — read it for the data's shape and its landmines, then write production code. Same staging contract as `enwik8.py`: **download-once, serial, concurrency-safe atomic-rename**, a **sha256 contract** on the frozen artifact, consolidation into memmap-able arrays, and a `NotStagedError` that prints the exact serial staging command rather than downloading from inside an array task.

## 1. CAMELS-US — Track B **PRIMARY** (adopted; survived criterion 4)

**Licence: `cc-by-4.0`** (Zenodo 15529996 = DOI 10.5065/D6MW2F4D, v1.2) — **resolved, record it in the module docstring.** ~**14.57 GB measured** (⛔ the "130 GB uncompressed" figure is unverified — do not repeat it).

**⛔⛔ TWO DEFECTS THE TRIPWIRE MEASURED — both must be structurally impossible to hit:**
1. ⛔ **The shipped `maurer` forcing product is DEFECTIVE** — `tmax == tmin` on **>99 % of days in 20/20 basins sampled**, and **3 files have a malformed header**. **Use `daymet` ONLY.** Do not expose `maurer` as a selectable option; if someone needs `maurer_extended` later they source it separately and prove it. **A validation check should assert `tmax > tmin` on a sample at stage time** so a future product swap cannot silently reintroduce this.
2. ⛔ **The shipped SAC-SMA model output is NOT an out-of-sample benchmark** and **must never be quoted against the LSTM.** It scores **0.708** on the same basins/window where Kratzert's split-sample series scores **0.603** ⇒ it is calibration-period fit. Carry this caveat **in the module docstring and in any comparison surface**, so the number cannot be picked up innocently.

**Build:** the 531-basin ML-benchmark subset with the standard split; `daymet` forcings + discharge + the 27 static attributes; **causal standardisation from training-period statistics only** (a full-record z-score is a test-set leak — the tripwire was explicit about this); NSE as the venue metric, with **median** reported as primary (⭐ `S = 0.758` median, **not** the 0.72 mean — the median is the venue's currency and the scout's row was corrected on exactly this point). Register the stream in the corpus/dataset registry the harness built.

## 2. N-CMAPSS DS02 — an **APPLICATION** venue, ⛔ **NEVER a primary-claim venue**

**Head ruling 2026-08-13:** adopted for company-research relevance (HEPA-adjacent). **~15.76 GB** (3× the scout's relayed figure). NASA PCoE original files, HDF5, sha256-frozen.

**⛔⛔ THREE GUARDRAILS THAT TRAVEL WITH THIS VENUE FOREVER — encode what you can, and state all three in the module docstring so they cannot be lost with a task file:**

**(a) THE REFERENCE ROW IS PRINTED IN EVERY N-CMAPSS TABLE WE EVER PRODUCE.** The matched-bytes exemplar store scores **7.988 RMSE** and **ties the weakest published deep model** (FNN 7.89 ± 0.12, ratio 1.0124 — which *does* trip the 2 % rule against that one reference). **Disclosure is the defence.** ⇒ Make the table emitter **structurally incapable** of producing an N-CMAPSS table without that row — the same posture as the unledgered-arm error, not a docstring plea. ⛔ Claims take **capability / application** form only; **never** *"beats SOTA on RUL"*.

**(b) The CLU-native questions this venue is for** (record them; do not build them here): the **persistent-store-across-units** question — C2W10's re-homed VALUE question, and a fleet is its natural home · the **anytime curve**, which is operationally meaningful in prognostics · **interpretable-store probes** (degradation-mode structure, the CAMELS-style analysis).

**(c) ⛔ THE CAFE EMBARGO IS ABSOLUTE.** Original NASA files only. **No CAFE-preprocessed number in any direction**, for or against. HEPA comparisons are **internal-only** unless re-derived on corrected labels (the banked label bug: test labels under-estimate by exactly `RUL_unit`). Put a **hard check** on the staging path that refuses a CAFE-derived artifact, and record file checksums.

⛔ **Classic C-MAPSS is not a claim venue** and is out of scope — do not build it, do not price it.

## 3. Ownership, stops, acceptance

**Yours:** new modules under `chlu/data/` for the two venues + their registry entries + tests. ⚠ **Check the registry's owner at spawn** — the C3 harness line owns `chlu/data/__init__.py` and the corpus registry; if that work is still open, coordinate or take the seam it left. ⛔ **NOT yours:** `chlu/core/**` · the Track-A corpora · the byte-ledger/exemption surface · any modelling code.

**Stops:** a licence or checksum that does not match what is recorded → **STOP and report**, do not mirror · a defect of the §1 class in any *other* product → report it as a finding, the tripwire's precedent is that these are real and worth naming.

**Acceptance (one line):** both venues stage serially, once, concurrency-safely with sha256 contracts and refuse to download from inside an array task; CAMELS is `daymet`-only with the `tmax > tmin` validation and both caveats in the docstring; N-CMAPSS carries all three guardrails with (a) and (c) enforced **in code, not prose**; splits are leak-free with causal standardisation; tests cover staging, split boundaries, the defect guards and the mandatory reference row; suite green with counts against a **named, re-verified HEAD**.

**Report:** protocol §5 + flag-provenance + dial declaration + git footprint.
