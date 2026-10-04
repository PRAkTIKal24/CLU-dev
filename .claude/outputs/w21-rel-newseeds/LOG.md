# LOG — w21 spoke/rel-newseeds (FORGIS_14)
- 2026-09-24T15:24:11Z START. Base w21/maze-t-seed4 bd2468f. Read COMMON.md + brief rel-newseeds.md. NO PREREG (banned 2026-09-24).
- ENV OK: chlu = worktree copy; jax 0.9.0 / numpy 2.4.1 / eqx 0.13.4; PYTHONPATH re-exported with ';' (env.sh), + scripts/ for w15_cmaze_t_run / w16_cmaze_t.
- TRAP 3 (new, printed): the worktree copy of peer_w21_clu_improve_e7fdb9b/CLU-REL-A10.json is CRLF (core.autocrlf=true) -> sha256 83db4d04…, NOT 0d4ec5b9…. The runner hashes raw bytes, so it must not read that copy. config/CLU-REL-A10.json = git blob bytes (git show bd2468f:…), sha256 0d4ec5b9fba55bd6f943a91fc8afa20027d79c5edea3531a47117c86f08ee5c6 ASSERTED, byte-identical (cmp) to outputs/w21-maze-t-clu-improve/prereg/CLU-REL-A10.json. Vendored runner/w21_clu.py = e7fdb9b verbatim modulo CRLF (diff after tr -d '\r' empty); git-blob sha256 w19_symmetric.py 5876162e359c6ed1… w21_clu.py e969b380c3035049….
- Choice D1 (like-for-like): run the PEER's vendored runner exactly as the peer ran seeds 0-2 TEST and seed4-agent ran seeds 3-5 VAL: `w19_symmetric.py <s> <out>/test --cond BOTH --leg test --rel_diag 1 --clu_cfg config/CLU-REL-A10.json` (defaults: rival_steps 8000 [+10x=80k], diag_mult 10, deletion on). Not scripts/w21_symmetric.py: that runner has no --clu_cfg; the per-tuple n-way print is recovered afterwards by zero-fit replay of the persisted objects if the replay supports the CLU-REL-A10 factory.
- Choice D2: no smoke — the identical runner/flags ran to EXIT 0 on this box three times on the VAL leg (outputs/w21-maze-t-seed4/val_cluimprove); only --leg differs. All 3 seeds launched together (3 heavy procs of mine = cap). Box: 46 GB free; foreign python = 3 (ladder-rivals-32k).
- 2026-09-24T15:25:25Z LEDGER rows maze-t-rel-newseeds 1/3 (seed 3), 2/3 (seed 4), 3/3 (seed 5) appended (>>) BEFORE launch.
- 2026-09-24T15:25:56Z LAUNCHED scored BOTH TEST seeds 3,4,5 (PIDs in PIDS.txt).
- 2026-09-24T15:30:51Z Runs 1-3/3 alive (PIDs 10656/8888/33220); log headers identical to the peer's seed-0 TEST header; cfg sha256 0d4ec5b9… printed by the runner; derangement [1,0] asserted on all three.
- scripts/w21_relnew_replay.py (new; = w21_pertuple_replay.py adapted for w21_clu objects + '+rel' rivals via the vendored runner's own Inputs/REL_FN; also checks file_sha256). Validated on the PEER's seed-0 TEST objects: 24 objects hash-match (param+file), all 32 n-way arms reproduce bit-exact -> replay/BOTH_test_seed0_PERTUPLE.json.
- scripts/w21_relnew_tables.py (new; imports w21_tables UNCHANGED for arm/training/deletion/rollout tables and the ceiling definition). Choice D3: easy/hard column split = maze-t-seed4's A1-ceiling classification carried over unchanged; REL's own largest-gap split printed beside. Choice D4: ceiling = w21_tables.CEIL_PREFIX exactly (same as A1; excludes '+rel' arms, which A1 runs do not have); a second REL ceiling INCLUDING '+rel' printed as its own column. Choice D5: convergence printing convention = WORSE@80k (80k final loss >= 8k's) and HIGH (final loss > 1e-2); raw losses + seen-row score always printed.
- Observed on existing data (seeds 0-2, peer): REL matched mlp on seed 2 is HIGH + WORSE@80k (0.0128 -> 0.0670; n-way 0.001/0.000). Printed.
- 2026-09-24T15:33:34Z Replays peer seeds 1,2 TEST: 24 objects each hash-match, 32/32 arms reproduce bit-exact (replay/BOTH_test_seed{1,2}_PERTUPLE.json).
- 2026-09-24T16:11:13Z poll: runs in rival/+rel training or scoring; no exit yet.
- 2026-09-24T16:11:39Z SCORED 1/3 BOTH TEST seed 3 DONE EXIT 0 -> test/BOTH_test_seed3.json [json [2717.3s]]:   row nway  n=675   CLU 0.9274 best-rival8k 0.7230 best-rival80k 0.6933 worst-ctl 0.6978 bseq 0.4519 bcol 0.0000 |  row seen  diag+rel best 1.0000 (never a bar)
  row pair  diag+rel best 0.7834 (never a bar)
  row nway  diag+rel best 0.9215 (never a bar)
  row value diag+rel best 0.9018 (never a bar)
- 2026-09-24T16:11:39Z SCORED 2/3 BOTH TEST seed 4 DONE EXIT 0 -> test/BOTH_test_seed4.json [json [2653.6s]]:   row nway  n=723   CLU 0.9004 best-rival8k 0.6238 best-rival80k 0.5325 worst-ctl 0.9004 bseq 0.3195 bcol 0.0000 |  row seen  diag+rel best 0.9992 (never a bar)
  row pair  diag+rel best 0.8266 (never a bar)
  row nway  diag+rel best 0.9004 (never a bar)
  row value diag+rel best 0.7422 (never a bar)
- 2026-09-24T16:11:39Z SCORED 3/3 BOTH TEST seed 5 DONE EXIT 0 -> test/BOTH_test_seed5.json [json [2717.4s]]:   row nway  n=660   CLU 0.9394 best-rival8k 0.8212 best-rival80k 0.9394 worst-ctl 0.8273 bseq 0.1030 bcol 0.0000 |  row seen  diag+rel best 1.0000 (never a bar)
  row pair  diag+rel best 0.8373 (never a bar)
  row nway  diag+rel best 0.9394 (never a bar)
  row value diag+rel best 0.9032 (never a bar)
- NOTE (printed, not hidden): seed 4 TEST n-way: max charter-C control = CLU = 0.9004 (the VAL-leg tie at 0.8994 REPEATS on TEST). Which control: see TABLES §3.
- 2026-09-24T16:15:53Z Replays seeds 3,4,5 (zero-fit): 24 objects each hash-match (param+file), 32/32 n-way arms reproduce bit-exact; all 32 MANIFEST objects per seed file-hash-match (replay/PERSIST-HASHES.json).
- 2026-09-24T16:15:53Z TABLES.md + REL-NEWSEEDS.json generated (scripts/w21_relnew_tables.py); added §9 (REL misses vs zero-fit relation-unreachable rows). FINDINGS.md written. Ledger OUTCOME row appended (factual, no label).
## FINAL SUMMARY
- Scored maze-t-rel-newseeds 1-3/3 (BOTH TEST, CLU-REL-A10): n-way CLU s3 0.927 / s4 0.900 / s5 0.939; best matched rival 0.723 / 0.624 / 0.939; best +rel 0.921 / 0.900 / 0.939; max control 0.698 / 0.900 (random_atoms TIES, as on VAL) / 0.827.
- Six seeds, same rows/arms/ceiling: CLU-REL-A10 mean 0.925 vs A1 0.677; REL CLU reaches the trained-arm ceiling on all 6 seeds (shared with random_atoms on s4, with matched transformer-80k on s5).
- Convergence: no WORSE@80k on seeds 3-5; best matched rival (transformer) unflagged on 3-5. Peer seed-2 mlp HIGH+WORSE@80k printed.
- NOT RUN: smoke; other conditions with CLU-REL-A10; re-scoring of seen/pair/value/deletion in replay (hash-checked only); no A1 +rel exists.
- No box-wide kill ever issued; only my own PIDs (PIDS.txt), all exited on their own.
- 2026-09-24T16:16:03Z COMMIT 31eeadd on w21/rel-newseeds; BUNDLE w21-rel-newseeds.bundle (branch only) git bundle verify OK; sha256 d1514372…. Worktree clean. END.
