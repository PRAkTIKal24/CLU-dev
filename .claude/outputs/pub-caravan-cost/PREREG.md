# PREREG — pub-caravan-cost (filed BEFORE any measurement in this spoke)

Filed 2026-09-07 ~17:45 CEST. Repo HEAD at filing: `7fcef50`. `git status --porcelain`: clean
(see `env.txt`). Agent: results-analyst. ⛔ No CLU number, no model training, no cluster contact
(Head ruling C1). Hardware for every measured row: Apple M1 Pro, 8 cores (6P+2E), 16 GB unified
RAM, 82 GB free disk, macOS 25.6.0 arm64.

## §0 Disclosure — what I had already read before filing
- `.claude/outputs/camels-caravan-refs.md` (frozen reference set; Li 2025 published V100 timings 5 h /
  21 h; Kratzert wall-clock NOT-FOUND).
- The Aug-13 gauged-run logs `.claude/scratch/c3-trackb-tripwire/{run_regional,run_phase2,build_camels,
  fetch_camels}.log`. **These are pre-existing measurements, not mine**; they are the basis of the
  predictions below and are quoted as *inherited measurements*, clearly labelled, in the report.
  From them: full gauged store-arm sweep = 10:19→12:07:22 BST (regional) + 12:07:48→12:25:13 (phase 2);
  per-invocation "built N stores in Xs / done in Ys" lines; total build ≈ 801 s, total eval ≈ 6,724 s.
- `build_camels.py` (157 lines) and `camels_tripwire.py` source.
- No number in this spoke's own measurement set existed at filing time.

## Why a prereg is required here
The acceptance criterion contains a **measured ratio**: "the extrapolation factor from the calibrated
run to the full run is stated explicitly." P5 below is that ratio. P1–P4, P6 are the measurement
predictions it rests on.

## Point predictions (committed before running)

| # | quantity | prediction | interval | derivation |
|---|---|---|---|---|
| P1 | sha256 throughput on the 3.406 GB CAMELS source zip | 1.0 GB/s | 0.5–2.0 | M1 has HW SHA-256; disk read from APFS at ~2 GB/s |
| P2a | full re-stage `build_camels.py` wall-clock (531 basins, Daymet + discharge + attrs, incl. sha256 of the 3.4 GB zip) | 240 s | 120–600 s | 1,062 text files parsed by pandas out of a zip; ~0.2 s/file + checksum |
| P2b | peak RSS of the re-stage | 1.5 GB | 0.8–3.0 GB | X is 531×12,784×5 float32 = 136 MB; Q float64 = 54 MB; pandas churn dominates |
| P2c | re-staged sha256s vs `staged/MANIFEST.json` | **identical for all 3 npz** | — | pipeline is deterministic; a mismatch would be a finding |
| P3 | `camels_tripwire.py --window 30 --scaling perbasin --Ls 250 --seeds 0 --kmeans 0 --sub 1` wall-clock | 10 s | 5–25 s | eval cost ∝ L: Aug-13 log gives 601 s eval for ΣL = 26,044 ⇒ 0.0231 s per L-unit ⇒ 5.8 s + ~3 s build/load |
| P3b | peak RSS of that arm | 1.2 GB | 0.6–2.5 GB | forcing array 136 MB + per-basin query block |
| P4 | same at `--Ls 2761` (the at-budget L) | 70 s | 40–150 s | 2761 × 0.0231 = 64 s + build |
| P5 ⭐ | **extrapolation factor: gauged temporal sweep → the same sweep on the Li-2025 10-fold PUB split, at the SAME test period** | **1.9×** | 1.5–2.5× | each basin is still queried exactly once across the 10 rounds ⇒ eval cost unchanged; store build runs 10× ⇒ (6,724 + 10×801)/7,525 = 1.96 |
| P6 | Zenodo throughput for the Li 2025 bundle (measured by a small ranged GET; ⛔ the full download is the tripwire spoke's, not mine) | 10 MB/s | 2–50 MB/s | typical EU CDN from this link |

**What would surprise me:** P2c failing (non-deterministic staging); the arm cost being super-linear in
L (it should be exactly linear — brute-force kNN); P5 > 3× (would mean store build, not query, dominates).

**What falsifies this spoke** (task DIAL): a measured cost differing from my extrapolation by > 2×.
⛔ Cancelled by Head ruling C3: the reference-LSTM local benchmark. GPU rows are **PROJECTIONS from
Li et al. 2025's published V100 timings**, never measurements, and are labelled as such.
