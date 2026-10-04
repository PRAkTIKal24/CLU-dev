# w15 maze-deletion-2b — LOG

- 2026-09-23T07:07:18Z START. Worktree /c/chlu/worktrees/CHLU-w15-maze-deletion-2b, branch w15/maze-deletion-2b @ 148f3b7. Scored runs: ZERO.
- ENV: chlu.__file__ = C:\chlu\worktrees\CHLU-w15-maze-deletion-2b\chlu\__init__.py (assert PASS). jax 0.9.0 / numpy 2.4.1 / equinox 0.13.4.
  ⚠ env_w3.sh's ':'-separated PYTHONPATH is collapsed by MSYS into ONE entry under Git Bash → equinox not importable.
  Fix: re-export the SAME two entries in the SAME order with the Windows ';' separator. No venv, no install.
- FOUND: banked maze deletion = w9/maze scored legs on CMAZE-2 (outputs/w9-maze/runs/scored{1,2,3}_seed{0,1,2}_test.txt:66):
  7.522e-16 / 8.826e-16 / **7.818e-16** (the "8.2e-16" quoted in w9 LOG.md:220 and downstream is not what the artefact says), projector 0.000e+00 x3.
- CHOICE (ICLR-closest): CMAZE-2b = the configuration of the printed CMAZE-2b rows = w14_read_verb.leg defaults:
  banks bank_after_seed{s} (w14-request-layer-fix), GRID_REBASED, CMAZE-2b pool 270 sites (w14-read-verb/pool), K=8, n_atoms=31, ridge=0.01, ae_steps=12000.
  PRIMARY run loads the very frozen encoders those printed rows used (w14-read-verb/runs/ae/ae_seed{s}_rebased_12000_270.eqx, read-only).
  CLEAN-CHECKOUT replication retrains the encoder from scratch (12000 steps) in a fresh clone of this branch.
- CHOICE: the exactness rows read TRAIN items only (asserted identical for val and test splits) → no held-out row touched.
  Adaptation-cost composition reads VAL rows only; test-leg composition DECLARED NOT-RUN (would be a scored act).
- CHOICE: beyond w9's statistic + projector, also measure the WHOLE deletable object (head Gram A, Mx and the read itself) vs refit-without
  (same fixed well partition), and supplementary deletions (every target word, maze=0, agent=0, random 10%). Primary row = target=green as banked.
- 2026-09-23T07:13:42Z FIRST PRIMARY PASS (cache encoders), seeds 0/1/2: statistic 7.212e-16 / 9.093e-16 / 8.937e-16, projector 0.000e+00 x3,
  head Gram A/Mx and read residual 0.000e+00 x3 (runs_old/deletion_primary_seed*.json). Re-running with sanity fields (deletion must MOVE the object) + rival retrain-without cost → runs/.
- 2026-09-23T07:13:42Z committed f9f5246; bundle clean_src.bundle; CLEAN CHECKOUT = git clone of that bundle at /c/chlu/outputs/w15-maze-deletion-2b/clean_checkout (0 dirty files).
  Clean replication rebuilds EVERYTHING from the clean checkout: 3 CMAZE-2b banks (w14_request_layer.build_bank after s 96), 3 pool shards
  (w14_read_verb.pool s 90), then encoders retrained from scratch (12000 steps), then the deletion rows. Inputs compared by sha256 to the live ones.
- 2026-09-23T07:21:47Z clean banks+pool built; sha256 vs live in clean/INPUTS_SHA.txt. Launched clean encoder-train+deletion runs seeds 0/1/2 (logs/clean_run_seed*.log).
- 2026-09-23T07:27:43Z primary+rival pass done (runs/). Clean encoders at step ~1000/12000 (~5 min/1000 steps). assemble.py written; TABLE.md draft built.
- 2026-09-23T07:37:27Z clean encoders step 2500/12000.
- 2026-09-23T07:47:23Z clean encoders step 4000/12000.
- 2026-09-23T07:57:29Z clean encoders step 5000/12000 (box contention; ~70 min left).
- 2026-09-23T08:07:34Z clean encoders step 6000/12000.
- 2026-09-23T08:17:49Z clean encoders step 6500-7000/12000.
- 2026-09-23T08:27:54Z clean encoders step 7500-8000/12000.
- 2026-09-23T08:38:01Z clean encoders step 8500/12000.
- 2026-09-23T08:48:08Z clean encoders step 9500/12000.
- 2026-09-23T08:58:16Z clean encoders step 10000-10500/12000.
- 2026-09-23T09:29:48Z CLEAN RUNS DONE. Encoders retrained from scratch are sha256-IDENTICAL to w14-read-verb's printed encoders; Ztrain identical; all 27 residuals identical.
  Banks+pool rebuilt in clean checkout: 6/6 sha256-identical to live inputs.
- 2026-09-23T09:29:49Z assembled MAZE-DELETION-2B.json + TABLE.md; wrote FINDINGS.md (fixed one wrong probe count before commit: 2038/2015/2055). Mirrored to worktree w15_maze_deletion_out/. Committing + bundling.
- 2026-09-23T09:30:07Z END. Branch 7511d68 bundled + verified (w15-maze-deletion-2b.bundle).
