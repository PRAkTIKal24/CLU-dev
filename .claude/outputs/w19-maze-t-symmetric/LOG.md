# w19 hub/maze-t-symmetric — LOG

* 2026-09-23T16:24Z START. Read COMMON.md (w15) + brief w19/maze-t-symmetric.md.
* Worktree /c/chlu/worktrees/CHLU-w19-maze-t-symmetric, branch w19/maze-t-symmetric at 9f80f59 (w16/maze-wm tip). Clean.
* env: `source /c/chlu/env_w3.sh`; assert OK: chlu.__file__ = C:\chlu\worktrees\CHLU-w19-maze-t-symmetric\chlu\__init__.py. (PYTHONPATH ':' joined imports equinox fine here; no ';' fix needed.)
* Box shared: 4 cores, 8 foreign python.exe processes already running. Never killed; only my PIDs go to PIDS.txt.

## Reading of the instrument (what is asymmetric in w16)
* w16 CLU: request = the 5 words -> `request_design` [bias 1 | main 27 | pair 290] = 318 wide.
* w16 rivals: w9 observation = 5x5 conv-AE patch (300) + pooled global 5x5 (300) + 29 dyn scalars + action one-hot 5 = 634 wide. NOT the words.
* w16 lookup WORDS floor: keyed on the words.
=> w19 changes ONLY the input, per condition, identically for every arm.

## Choices taken where the brief leaves them open (closest to an ICLR main-track result)
* C1 Rival margin is claimed against the STRONGER of the 8,000-step row and the 80,000-step (10x steps) column (both printed). Conservative; consistent with w16 PREREG-T1 AMENDMENT 4.
* C2 cont-MLP (w4 ContMLP convention: five words as min-max-normalised numbers + a requested flag each, 10 wide) is a rival in WORDS-MAIN and WORDS-FULL (information-identical, encoding differs -- declared) and counts toward VB1/SB1 (conservative). In OBS/BOTH it is DECLARED N/A (the observation is already continuous; the MLP row is the continuous-input MLP).
* C3 WORDS-MAIN: the CLU's design is [bias|main] (pair_block=False) == w16's learned BLIND-SEQUENTIAL arm. VB3/SB3 on blind-seq is vacuous there (it IS the CLU) -> evaluated on BLIND-COLOUR only, declared; the row is printed as "the learned blind-sequential reading". Blind-colour and controls use the condition's design (main only).
* C4 OBS extractor: MLP 634 -> 32 (relu) -> 27 logits (5 softmax heads), trained on TRAIN(seen) rows only, 6,000 Adam steps. The CLU head + store are fitted on the EXTRACTED train words and read on extracted words (pure pipeline obs -> extractor -> request -> store). Extractor floats are ADDED to the matched-rival target (target = CLU read floats + extractor floats) -- conservative. Rivals: raw obs row AND obs ⊕ extractor one-hots row, both in VB1. WORDS lookup floor keyed on extracted words.
* C5 BOTH: rivals get obs ⊕ the 318-wide design; CLU reads the true words (request-only by signature, it cannot consume the obs). VAL for all arms; scored only if runs remain.
* C6 State encoder is trained only in OBS/BOTH (no arm consumes it in WORDS-*); declared.
* C7 Deletion rows produced on every SCORED condition's TEST run (so the winning condition is covered); rival retrain-without at the declared 8,000 steps.

## Build
* 16:28Z PREREG-w19-maze-t-symmetric.md written + sha256 BEFORE any number.
* scripts/w19_symmetric.py (runner), scripts/w19_bars.py (bars). w15/w16 modules imported, not edited.
* Smoke (VAL seed 0, 60 steps, 150 rows/class, NOT reported — plumbing only): WORDS-FULL ok (1124 s), OBS ok (608 s).
* ⚠ Measured: the w9 observation's dynamics block is dyn_dim(4) = 31 wide (w9 docstring / brief say "29"); obs = 600 conv-AE + 31 dyn + 5 action = 636. The instrument's obs is used unchanged; the brief's "29 dynamics scalars" = this block.
* User request mid-wave (applied before any real run): PERSIST every trained object per condition+seed to persist/<cond>_seed<N>/<leg>/ — codec.eqx, state_encoder.eqx, clu.pkl (+controls, blind arms), every rival (.eqx; 8k, 80k, 10x-floats), word_extractor.pkl, deletion pair (clu_after_*.pkl, clu_fitted_without_*.pkl, rival_mlp_retrain_without_*.eqx, rival_mlp_as_trained_*.eqx), each with <file>.sha256 (param hash + file hash) and MANIFEST.json with rebuild args.
* 2026-09-23T17:27:31Z smokes of all 4 conditions pass (persist reload verified: rival + clu param hashes match). LAUNCHED 12 VAL runs (4 cond x seeds 0/1/2), single-threaded XLA each; PIDs in PIDS.txt; outputs val/, logs logs/val_*.log
* 2026-09-23T17:37:54Z OBS extractor TRAIN tuple-acc 0.404/0.683/0.767 (under-fit). PREREG AMENDMENT 1: OBS-X10 extractor diagnostic (60k steps), printed only; OBS verdict stays on the declared extractor.
* 2026-09-23T18:28:02Z VAL WORDS-MAIN seeds 0,2 done: nway CLU 0.242/0.446 vs best matched rival 8k 0.872/0.803, 80k 0.923/0.871 (log lines).
* 2026-09-23T18:43:46Z VAL BARS (val/BARS-*-val.json): WORDS-MAIN LOSS (VB1 F, VB2 F: CLU 0.334 vs R* 0.747); WORDS-FULL LOSS (VB1 F: CLU 0.7247 vs R* transformer-8k 0.742 — matched rivals given the CLU's identical 318 design tie/beat it; the pair block, not the store, carries the w16 margin); BOTH PASS (margin +0.108; seed-1 lead 0.0015 — knife edge; 10x-read-floats transformer diag 0.828 > CLU 0.802 on seed 0, printed, not a bar). Seed 1 caps every arm at 0.5689 (data ceiling, not the store).
* Decision (PREREG spending rule): WORDS-MAIN, WORDS-FULL DECLARED, 0 runs. Runs remain whatever OBS does (9-0-0-<=3 >= 3), so BOTH earns 3 scored runs NOW. Ledger rows maze-t-symmetric 1/9..3/9 written 2026-09-23T18:43:46Z, BEFORE launch.
* 2026-09-23T20:08:13Z OBS-X10 diagnostic bars (val/BARS-OBS-X10-val.json): CLU 0.447 vs R* 0.461 (transformer+extractor 80k) -> would read LOSS even with the 10x extractor; extractor whole-tuple accuracy on held-out n-way rows 0.15/0.30/0.33 (it does not compose). Printed only.
* 2026-09-23T20:08:13Z OBS VAL bars (val/BARS-OBS-val.json): LOSS — VB1 F (CLU 0.351 vs R* transformer-80k raw obs 0.457), VB4 F (extracted-words floor 0.407), VB2/VB3 pass. DECLARED, 0 runs.
* Budget now: BOTH 3 spent (1–3/9, running); WORDS-MAIN, WORDS-FULL, OBS failed VAL -> 0 runs each; the remaining 6 runs are DECLARED NOT-RUN (no other condition passes VAL; 3 seeds all-or-none, BOTH already has its 3).
* BOTH TEST nway from logs (final after JSON): seed0 CLU 0.800 vs best 0.627; seed1 CLU 0.5660 vs rival-8k 0.5674 (CLU BEHIND on seed 1 -> SB1 every-seed clause fails); seed2 0.808 vs 0.784.
* 2026-09-23T20:21:47Z BOTH TEST done (test/BOTH_test_seed*.json; ~5,000 s each). test/BARS-BOTH-test.json: SB1 FAIL (margin +0.0988 < 0.10; seed 1 CLU 0.5660 < lstm-8k 0.5674), SB2–SB4 PASS -> LOSS by the rule. CLU + controls on TEST reproduce w16 A1 to every printed digit (instrument unchanged).
* Deletion (BOTH TEST): store rel dS <= 1.1e-15; downdate == fitted-without on 100% of surviving n-way rows; CLU adapt 0.09–0.39 s data-free; rival retrain-without 100–140 s / 8,000 steps.
* Fixed w19_bars.py (OBS-X10 extractor-accuracy key: cond.startswith("OBS")); all BARS re-generated; TABLES.md + MAZE-T-SYMMETRIC.json generated.
* Persisted objects were written to <leg>/persist/... by the runner and MOVED (mv, nothing deleted) to persist/<cond>_seed<N>/<leg>/ as requested; 384 objects, file hashes re-verified 0 mismatches. JSON 'persist_dir' fields name the pre-move path.
* Final: NO condition wins. Budget 3/9 spent (BOTH, LOSS), 6 DECLARED NOT-RUN (WORDS-MAIN, WORDS-FULL, OBS failed VAL). FINDINGS.md written.
* All my background PIDs (PIDS.txt) have exited; no kills issued.
