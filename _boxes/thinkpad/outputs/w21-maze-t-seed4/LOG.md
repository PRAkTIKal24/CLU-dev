# LOG — w21 spoke/maze-t-seed4 (FORGIS_14)
- 2026-09-24T07:28:40Z START. Base w19/maze-t-symmetric 0455961. Read COMMON.md + brief.
- 2026-09-24T07:28:40Z HEAD AMENDMENT-1 read (69e758c26d434f60). It REPLACES Step 1: no screening; seeds 3,4,5 ALL run and kept.
  Scored runs (3) = words-plus-observation TEST, CLU A1 + all rivals, one per seed 3/4/5. Free VAL leg = OBS on VAL, CLU A1 + all rivals, seeds 3/4/5.
  words-full gets no new seeds (3-seed row). Run 3 improved-CLU leg superseded -> optional FREE VAL leg if peer config exists at the end.
  Easy/hard classified by per-seed CEILING (best any arm), never by CLU score. Zero-run diagnosis for seed 1 + every hard seed. No PREREG.
- 2026-09-24T07:30:17Z ENV OK: chlu=C:\chlu\worktrees\CHLU-w21-maze-t-seed4\chlu\__init__.py; jax 0.9.0 np 2.4.1 eqx 0.13.4; PYTHONPATH re-exported with ';' (env.sh).
- 2026-09-24T07:31:52Z SMOKES launched (val leg, --smoke 1, seed 3; BOTH and OBS) via unmodified scripts/w19_symmetric.py; PIDs 29060, 35492 in PIDS.txt. Box: 15.7 GB free / 63.4, 13 foreign python procs.
- 2026-09-24T07:50:09Z scripts/w21_symmetric.py = w19_symmetric.py + additive PER_TUPLE_NWAY printout (diff = _sc wrapper; W16.sc numbers unchanged, per-tuple sum asserted equal). scripts/w21_seed_diagnosis.py = zero-fit hold-out support diagnosis. Probe seed1 TEST: 10/20 n-way tuples (403/742 rows) have a colour pair with ZERO train rows, each blocked by a held-out 2-way pair; 4 tuples (rule_a) have no eval rows. Full diagnosis seeds 0-5 x test,val launched PID 35100.
- 2026-09-24T08:01:03Z SMOKE BOTH val seed3 (w19 runner) EXIT 0 [1284s]; derangement [1,0] asserted. OBS smoke finishing.
- 2026-09-24T08:01:03Z Launched: w21-runner smoke BOTH val seed3 (PID 24676) and FREE VAL leg OBS val seed3 (PID 29464, w21 runner, A1 latent128/K2/6 atoms/ridge1, rival_steps 8000/80000, diag_mult 10).
- Choice C1: VAL legs and scored legs use scripts/w21_symmetric.py (additive per-tuple print only) so every arm's per-tuple n-way score exists for the per-tuple ceiling. Smoked before any scored launch.
- Choice C2: seed order 3,4,5; at most 3 heavy procs of mine; hold for >=5 GB free.
- NOTE: peer config exists (outputs/w21-maze-t-clu-improve prereg/CLU-REL-A10.json; used on its own maze-t-symmetric 7-9/9). Per AMENDMENT-1 it is at most a FREE VAL leg at the end.
- 2026-09-24T08:06:43Z REPLAY (zero-fit) w19 BOTH test seed1: 26 arms, ALL param hashes match MANIFEST, ALL n-way numbers reproduce bit-exact -> replay/BOTH_test_seed1_PERTUPLE.json.
- Choice C3 (CEILING definition, fixed before reading any new-seed number): per-seed ceiling = max n-way exact over every arm FITTED TO THE TRAIN DATA (CLU, every rival at 8k and 80k, 10x-read-floats diagnostic, learned blind arms, controls, lookup floors). The planner columns (ORACLE/BLIND-* planners) are GIVEN the engine rules (ORACLE = 1.000 by construction) and are excluded from the ceiling; they are printed beside it.
- Diagnosis TEST (zero-fit): unidentified (class C/D) tuples/rows: s0 3/118 of 761; s1 10/403 of 742; s2 9/215 of 686; s3 9/339 of 675; s4 8/305 of 723; s5 4/40 of 660. Every class-D tuple is blocked by a held-out 2-WAY pair (pocket x door / pocket x tint) -> zero train rows by construction.
- 2026-09-24T08:24:21Z LEDGER rows maze-t-seed4 1/3 (seed 3) and 2/3 (seed 4) written BEFORE launch; launched BOTH TEST seeds 3,4 (w21 runner @1c4e304, sha256 68497f65…). Heavy procs of mine: OBS val s3, BOTH test s3, BOTH test s4 (=3). 12.2 GB free at launch.
- 2026-09-24T08:28:50Z Peer kept config FOUND: CLU-REL-A10.json sha256 0d4ec5b9fba55bd6… (= peer PREREG record). Vendored peer runner+w21_clu.py verbatim from e7fdb9b into peer_w21_clu_improve_e7fdb9b/ (sha256 5876162e…/e969b380…). Per AMENDMENT-1 run as FREE VAL leg only: BOTH val seeds 3,4,5, --rel_diag 1 (same flags as the peer's runs). Queue2 starts after queue 1.
- 2026-09-24T08:29:42Z diagnosis v1 (SEED-DIAGNOSIS.json) complete; SUPERSEDED by v2 (adds colour-relation reachability + blocking 2-way pair); v1 kept, not deleted. v2 launched. Peer finding logged: CLU-REL-A10 VAL seed1 = 0.9141 vs A1-family ceiling 0.5689 -> seed 1 hard-ness is REPRESENTATIONAL for colour-identity designs; must be printed beside the 'hard seed' framing.
- DIAGNOSIS v2 (zero-fit; diagnosis/SEED-DIAGNOSIS-v2.json, logs/diagnosis_v2.log). TEST n-way, unidentified-for-colour-identity rows / eval rows (…also unreachable by colour relation):
  s0 118/761 (0) | s1 403/742 (79 rows, 4 interaction tuples) | s2 215/686 (0) | s3 339/675 (4 rows, 2 tuples) | s4 305/723 (0) | s5 40/660 (40 rows, 4 interaction tuples).
  Mechanism: every class-D tuple has a colour pair (pocket x door or pocket x tint) that IS a held-out 2-way pair -> 0 train rows by construction. The interaction family (c,c,c) with signature pocket==door==tint has NO train row of that relation under ANY colour on seeds 1 and 5 -> unreachable even relationally.
- 2026-09-24T09:59:28Z LEDGER row maze-t-seed4 3/3 (seed 5) written BEFORE launch.
- 2026-09-24T09:59:36Z QUEUE launched BOTH test seed 5 (free 26GB, mine before launch 2)
- 2026-09-24T10:01:35Z SCORED 1/3 BOTH TEST seed 3 DONE EXIT 0 [5718s] -> test/BOTH_test_seed3.json. nway CLU 0.6741 best-rival8k 0.4119 best-rival80k 0.4074 worst-ctl 0.1348 bseq 0.4519 bcol 0.0000 (full arm table in TABLES.md). NOTE queue saw mine=2 at 09:59:28 while seed-3 was writing its JSON; seed 5 launched then (brief overlap at most seconds).
- 2026-09-24T10:02:58Z QUEUE launched OBS val seed 4 (free 29GB, mine before launch 2)
- 2026-09-24T10:11:20Z SCORED 2/3 BOTH TEST seed 4 DONE EXIT 0 -> test/BOTH_test_seed4.json. nway CLU 0.5325 best-rival8k 0.5201 best-rival80k 0.5311 worst-ctl 0.2656 bseq 0.3195 bcol 0.0000.
- 2026-09-24T10:14:41Z CORRECTION to the 'brief overlap' note: seed 4 finished FIRST (~09:57, 5556s), seed 3 at ~10:00 (5718s). Queue counts mine=2 at 09:59:28 and 10:02:51 were CORRECT; never more than 3 heavy procs of mine.
- 2026-09-24T10:14:41Z REPLAY CHECK of MY persisted objects (zero-fit): BOTH TEST seeds 3,4 -> all 26 param hashes match MANIFEST, all n-way numbers AND per-tuple counts reproduce exactly (replay/check_BOTH_test_seed{3,4}_PERTUPLE.json).
- Seed classification so far (largest-gap rule on BOTH TEST ceilings, 5 seeds): easy 0,2,3; hard 1,4. Seed 3 ceiling 0.714 = diagnostic@10x-read-floats transformer (CLU 0.674); seed 4 ceiling 0.535 = 10x-floats transformer (CLU 0.533). FINAL classification waits for seed 5.
- 2026-09-24T10:27:56Z QUEUE launched OBS val seed 5 (free 31GB, mine before launch 2)
- 2026-09-24T10:33:49Z FREE VAL OBS seed 3 DONE EXIT 0 -> val/OBS_val_seed3.json: row nway  n=720   CLU 0.2792 best-rival8k 0.4347 best-rival80k 0.4583 worst-ctl 0.2319 bseq 0.1361 bcol 0.0236  ext-tuple-acc 0.1403
- 2026-09-24T10:44:12Z SCORED 3/3 BOTH TEST seed 5 DONE -> test/BOTH_test_seed5.json [2525s]: row nway  n=660   CLU 0.6818 best-rival8k 0.8394 best-rival80k 0.8667 worst-ctl 0.4955 bseq 0.1030 bcol 0.0000. All 3 scored runs of maze-t-seed4 spent.
- 2026-09-24T10:44:28Z QUEUE2 launched FREE VAL CLU-REL-A10 BOTH val seed 3 (free 26GB, mine 2)
- 2026-09-24T10:47:46Z REPLAY CHECK seed 5 persisted objects: hashes ok, reproduce-all True. Seed 5 per-tuple: CLU 0.000 on class-A 'free' tuples (2,0,1,2) n=90 and (2,0,3,2) n=80 where 8 trained arms reach 1.000 -> CLU-specific miss on REACHABLE data (printed). Seed 4 hard tuples = class D blocked by held-out pair (pocket 2 x tint 2), all reachable by colour relation.
- 2026-09-24T11:57:43Z QUEUE2 launched FREE VAL CLU-REL-A10 BOTH val seed 4 (free 31GB, mine 2)
- 2026-09-24T11:58:37Z FREE VAL OBS seed 4 DONE EXIT 0 -> val/OBS_val_seed4.json: row nway  n=696   CLU 0.1236 best-rival8k 0.3822 best-rival80k 0.4167 worst-ctl 0.2888 bseq 0.1394 bcol 0.0144  ext-tuple-acc 0.1207
- 2026-09-24T12:18:17Z NOTE (printed, not hidden): seed 3 matched mlp (width 4, 954-dim input) did NOT converge at either budget (final loss 0.180 @8k, 0.205 @80k; n-way 0.003/0.107; seen 0.483/0.846). Seed 4 matched lstm 80k final loss 0.137 (n-way 0.107). Rivals ran their full declared budgets; the non-convergence is a capacity/optimisation pathology of the 4-unit matched rival, printed in the training table.
- 2026-09-24T12:56:03Z QUEUE2 launched FREE VAL CLU-REL-A10 BOTH val seed 5 (free 31GB, mine 2)
- 2026-09-24T12:58:07Z FREE VAL OBS seed 5 DONE EXIT 0 -> val/OBS_val_seed5.json: row nway  n=717   CLU 0.2036 best-rival8k 0.4338 best-rival80k 0.4854 worst-ctl 0.2399 bseq 0.2538 bcol 0.0181  ext-tuple-acc 0.2120
- 2026-09-24T13:08:35Z FREE VAL CLU-REL-A10 BOTH val seed 3 DONE EXIT 0 -> val_cluimprove/BOTH_val_seed3.json: row nway  diag+rel best 0.9236 (never a bar)
row nway  n=720   CLU 0.9181 best-rival8k 0.7500 best-rival80k 0.7111 worst-ctl 0.6889 bseq 0.4403 bcol 0.0000
- 2026-09-24T13:10:56Z Replays done: w19 OBS VAL seeds 0,1,2 (34 arms each) hashes ok + reproduce-all. FINDINGS.md drafted (CLU-REL-A10 section pending seeds 4,5).
- 2026-09-24T13:30:57Z FREE VAL CLU-REL-A10 BOTH val seed 4 DONE EXIT 0 -> val_cluimprove/BOTH_val_seed4.json: row nway  n=696   CLU 0.8994 best-rival8k 0.6566 best-rival80k 0.5632 worst-ctl 0.8994 bseq 0.3218 bcol 0.0000
- 2026-09-24T13:37:44Z NOTE: CLU-REL-A10 seed 4 VAL n-way: a charter-C control equals the CLU (0.8994) -> on this seed the store is NOT load-bearing for CLU-REL-A10 (printed).
- 2026-09-24T13:46:12Z FREE VAL CLU-REL-A10 BOTH val seed 5 DONE EXIT 0 -> val_cluimprove/BOTH_val_seed5.json: row nway  n=717   CLU 0.9052 best-rival8k 0.7936 best-rival80k 0.9052 worst-ctl 0.8020 bseq 0.1158 bcol 0.0000. ALL COMPUTE DONE.
- 2026-09-24T13:47:10Z LEDGER: factual OUTCOME row appended (no label). TABLES regenerated (final).

## FINAL SUMMARY (all numbers in TABLES.md / MAZE-T-SEED4.json; prose in FINDINGS.md)
- Scored (maze-t-seed4 1-3/3, BOTH TEST, CLU A1): n-way CLU s3 0.6741 / s4 0.5325 / s5 0.6818; best matched rival 0.4119 (transformer-8k; mlp non-converged) / 0.5311 (mlp-80k) / 0.8667 (mlp-80k); ceiling 0.7141 / 0.5353 / 0.9394 (10x-floats transformer).
- 6-seed classification by BOTH TEST ceiling, largest-gap cut 0.6407: easy 0,2,3,5; hard 1,4. Matched-only ceiling -> hard 1,3,4 (printed as sensitivity).
- OBS VAL (free) n-way CLU s3 0.2792 / s4 0.1236 / s5 0.2036 vs best matched rival 0.4583 / 0.4167 / 0.4854.
- FREE VAL CLU-REL-A10 BOTH: s3 0.9181 / s4 0.8994 / s5 0.9052 (seeds 0-2 peer: 0.9029/0.9141/0.9943). s4: random_atoms control = CLU (0.8994); s5: matched transformer-80k ties (0.9052).
- Zero-fit: CLU-REL-A10 VAL = 1 - (relation-unreachable rows)/n exactly on seeds 1 (58/675) and 5 (68/717).
- Choices: C1 w21 runner (additive per-tuple print); C2 seed order 3,4,5 / <=3 heavy / >=5GB; C3 ceiling = trained arms, planner columns excluded; C4 easy/hard cut = largest gap in sorted BOTH TEST ceilings (+ full sensitivity + matched-only variant); C5 peer config run as FREE VAL on BOTH with its own runner verbatim (--rel_diag 1), per AMENDMENT-1.
- NOT RUN: brief screening (cancelled); words-full new seeds (AMENDMENT-1); scored improved-CLU run (superseded -> free VAL); A1 BOTH VAL seeds 3-5 (not requested); OBS TEST; per-tuple tables for OBS VAL / CLU-REL-A10 not tabulated (OBS per-tuple data exists).
- No box-wide kill ever issued; all my python processes exited on their own (count 0 at 13:46Z).
- 2026-09-24T13:47:29Z BUNDLE w21-maze-t-seed4.bundle (branch w21/maze-t-seed4 @ bd2468f) created; git bundle verify OK; sha256 c34e5872869b4ec2…. Worktree clean. END.
