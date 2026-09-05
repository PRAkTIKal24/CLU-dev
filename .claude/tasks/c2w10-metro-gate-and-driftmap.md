# c2w10-metro-gate-and-driftmap — test the fallback before spending a wave on it

**Campaign 2, C2W10 ("The persistent store"). Agent:** results-analyst. **ZERO worktrees.**
No tracked-code edits: everything under `.claude/` and a scratch venv.
Writes `.claude/outputs/c2w10-metro-gate.md` + artifacts to `.claude/outputs/c2w10-metro-gate/`.
**Budget:** ≈ 1 day. **Runs before any Metro cell of any kind.**

## Why this task exists
The INSECTS tripwire **fired** (`criterion4_cleared = false`, SAM-kNN at 0.634 MiB reaching 76.92 %
against ARF's 78.81 %, and firing against *every* ARF reference tried). It cost ≈4 h of CPU and it
saved the wave from spending a lifecycle build on an inadmissible venue. **The Metro fallback is
pre-authorized (Head ruling 1) — but it is not yet argued.** The scout flagged criterion 4 on Metro as
⚠, not ✅: *"regression over an 8-D input; nearest-neighbour-over-past-windows is a real threat and
must be laundered the same way."* Firing a fallback without testing it would repeat, at full price,
the exact mistake we just avoided at 1/50th of it.
**Two jobs, one spoke: (A) the criterion-4 tripwire on Metro; (B) the drift map that PREREG-C2W10 §2
makes a mandatory precondition of any retention claim on this stream.**

**Binding documents, read first:**
- `.claude/outputs/c2w10-persistent-store/PREREG-C2W10.md` — **§2 (fallback paragraph), §2.0, §2.1,
  §2.2, §2.2b, §3, and AMENDMENT 3 IN FULL.**
- `.claude/outputs/c2w10-benchmark-gate.md` + `BENCHMARK-GATE.json` — **your template.** Reuse its
  protocol, its scratch-venv discipline, its reproduction gate, and its artifact shape. ⭐ Its §-final
  recommendation 3 is this task.
- `.claude/outputs/c2w10-benchmark-scout.md` §3 C2 + §4 (the hidden-clock protocol and its honest cost).

## ⭐ DIAL DECLARATION (protocol §7) — echo before your first result
- **Dial:** none — **benchmark admissibility instrument + a drift map.** ⛔ No CLU cell, no store, no
  performance claim about the CLU.
- **Laundering control:** N/A — **you ARE the laundering control**, exactly as the INSECTS gate was.
- **Falsifies:** M1 failing ⇒ the harness/loader is defective and nothing on this stream is quotable.
- **Does NOT falsify:** an exemplar/NN store beating the forecasting baselines falsifies **Metro's
  admissibility as a VALUE venue**, not the CLU.
- ⛔ NOT-RUNs declared, never nulls.

---

## (A) The tripwire — same shape as the INSECTS gate, adapted to regression

**Protocol:** Metro Interstate Traffic Volume (UCI, CC BY 4.0, 48 204 hourly records, 2012–2018),
**hidden-clock, 24-h-ahead horizon**: `date_time` **withheld from the features** so calendar
regression cannot win and persistence cannot win. Freeze the stream to one local file with a recorded
**sha256**, exactly as the INSECTS gate did — that file is the contract for every later arm.

| leg | label | statement | rule |
|---|---|---|---|
| **M1** loader/protocol control | MECHANICS | persistence (last-observed) and a seasonal-naive (t−24 h) baseline under prequential MAE/RMSE | both computed and reported; if persistence is at ceiling under the 24-h horizon, the hidden-clock protocol has failed and **the venue is dead** — say so |
| **M2 criterion-4 tripwire** ⭐ | MECHANICS | **k-NN over past windows** (the input-metric attack the scout named) at the same **state-byte budgets** as INSECTS' arms — 0.634 MiB and 133 kB — against the tuned strong baselines (GBDT + at least one sequence model: GRU or a small sliding-window attention) | **exemplar/NN store within 2 % relative MAE of the best strong baseline (or better) ⇒ criterion 4 FIRES on Metro too.** Report the margin and its robustness across every baseline reference, as the INSECTS gate did |
| **M3** headroom | MECHANICS | best strong baseline vs persistence and seasonal-naive | no headroom (nothing beats the naive baselines materially) ⇒ criterion 2 fails and the venue is dead |
| **M4** byte ledger | MECHANICS | state bytes of every arm, computed from the data's own shape | reported; note which arms are **not** byte-matched, as ARF was not on INSECTS (14.35×) |

⛔⛔ **If M2 fires, STOP and report — do not improvise a third venue.** Two fired tripwires in one wave
is a **finding about the fallback track**, not a scheduling problem, and it is the Hub's call (and
above it, the Head's) what a persistent-store VALUE venue can be. ⭐ A fired M2 is *also* the sixth
confirmation of the criterion-4 theorem and is written up as one.

## (B) The drift map — the mandatory precondition PREREG §2 attaches to this stream
⚠ **Metro's regime structure is undocumented in the stream-learning literature** (the scout verified
this). Under Souza §4.1's warning about informal drift assumptions, **the annotation would be ours** —
so it must be *published as a measurement*, before any retention claim, and labelled as ours forever.
1. **Webb-style drift magnitude between windows** across the stream; identify and **date** the
   candidate regimes (daily / weekly / holiday / seasonal / weather).
2. **Emit the revisit schedule as an explicit index table** — the Metro analogue of INSECTS' published
   change points — with, for each regime, its first visit and every revisit.
3. ⛔⛔ **Apply the b = 4 lesson from the INSECTS gate BEFORE the map is used:** for every candidate
   band, report **class/label entropy, the naive-baseline score, and the best-arm score**, and
   **exclude any band that is persistence-trivial or at ceiling**. INSECTS' terminal band was
   uninterpretable and we found it only after registering it. Do not hand the wave a contaminated map.
4. **Reconstruct the drift-free NULL** that `out-of-control` was going to provide and cannot (no data
   source exists): a stationarized control on the same features — e.g. a time-shuffled stream where
   the regime structure is destroyed but the marginal distribution is preserved. ⛔ **The control is
   re-constructed, never quietly dropped**; state its construction and its limits.

## Environment, reproduction gate, decimation
Identical to the INSECTS gate and non-negotiable: ⛔ **nothing enters `pyproject.toml` / `uv.lock`**
(the recorded `pandas 3.0.3 → 2.3.3` project-wide precedent); **scratch venv** with its **exact
versions and full frozen package list** recorded; **the sha256'd stream file is the contract**; and
⭐ **the reproduction gate — the harness's own loader must reproduce that sha256 before any number
here is consumed downstream.** Reproduce first, compare second.
At 48 204 records Metro is 0.6× INSECTS, so **decimation is likely unnecessary** — but report the
ladder structurally anyway (`m ∈ {1, 2, 5, 10}`: instance counts, surviving regime boundaries, per-band
counts) so the Hub can file `m` if the CLU harness needs it. ⚠ Any adaptation-like quantity is
**per-instance-since-change, never per-stream-position** (decimation compresses the drift timeline).

## Deliverable (exact path — later spokes gate on this string verbatim)
`.claude/outputs/c2w10-metro-gate/METRO-GATE.json` with **at minimum**:
```
m1_persistence, m1_seasonal_naive, m1_pass,
m2_arms: {knn_windows_634kib, knn_windows_133kb, gbdt, <sequence_model>} -> {mae, rmse, state_bytes},
m2_margin_rel, criterion4_cleared_metro (bool, computed arithmetically), m2_robustness,
m3_headroom, byte_ledger,
drift_map: {regimes[], revisit_schedule[], band_diagnostics[], excluded_bands[], method},
drift_free_null: {construction, path, sha256, limits},
stream: {path, sha256, n_records, features_used, horizon_hours},
scratch_venv, river_or_lib_versions, reproduction_gate, decimation_ladder, notes
```

## FILE OWNERSHIP (declared)
**You own:** `.claude/outputs/c2w10-metro-gate/**` · `.claude/scratch/c2w10-metro/**` · your declared
frozen-stream cache path. ⛔ **You touch NO tracked code** — not `chlu/`, not `tests/`, not
`pyproject.toml`, not `uv.lock`.

## Acceptance (mechanical)
1. `METRO-GATE.json` at the exact path, with `criterion4_cleared_metro` computed **arithmetically**.
2. M1 reported **before** M2; the robustness table present (every baseline reference, as INSECTS did).
3. The drift map published **with its band diagnostics and its exclusions**, and labelled **ours, not
   the literature's**, in the artifact itself.
4. The drift-free null re-constructed, with its construction and limits stated.
5. Scratch venv frozen list + sha256 contract + reproduction gate all recorded; **zero** tracked-file changes.
6. Reconciliation list in the **first 10 lines**; NOT-RUNs declared as NOT-RUNs.
⛔ You do not choose the wave's venue — you compute the boolean and publish the map; the Hub rules.
