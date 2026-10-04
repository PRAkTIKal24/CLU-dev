# w12 spoke/clue-theorist — LOG

- START_UTC: 2026-09-21T11:09:27Z
- Worktree: C:/chlu/worktrees/CHLU-w12-clue-theorist, branch w12/clue-theorist
- Output dir: /c/chlu/outputs/w12-clue-theorist/

## 11:09 — read task file /c/chlu/tasks/w12-spoke-clue-theorist.md
One job: tightest honest statement that the subspace store is the CLU with settling off
(zero-dynamics limit) + what CRR predicts when settling is on. Deliverable CLUE.md.
Scored runs: none. Numerical sanity checks only.

## 11:12–11:25 — evidence read (no writes outside my dir)
Read, in order:
- /c/chlu/evidence/outputs/w9-advisor-report.md (239 l) + w10-advisor-report.md (139 l) — current state.
- /c/chlu/evidence/outputs/advisor-wave-plan-w12-w14.md — confirms this spoke's scope (§1 scouts).
- /c/chlu/evidence/outputs/w6-advisor-report.md — the wave-6 physics wave.
- /c/chlu/evidence/outputs/_thinkpad/w6/w6-compose-physics/{LOG.md,PREREG-w6-compose-physics-1.md,SCORED-TABLE.md}
- /c/chlu/evidence/outputs/_thinkpad/w6/w6-compose-physics/scored/test_seed{0,1,2}/arms/pca_settle_zerostep/arm.json
- /c/chlu/evidence/outputs/_thinkpad/w6/w61-compose-physics/LOG.md
- /c/chlu/evidence/outputs/_thinkpad/w6/w6-potentials/RESULTS-LEDGER.md
- /c/chlu/evidence/outputs/w10-physics-section-evidence/EVIDENCE.md (§2, §4.2, §4.3, §4.4)
DID NOT read $CHLU_CLAUDE_ROOT (frozen stale snapshot), per standing rules.

Key anchors located (file : value):
- zerostep == pca_tensor on all six scored columns: SCORED-TABLE.md
  r4 63.612/64.327/67.653, r3 29.717/31.617/31.450 — identical to pca_tensor's row.
- `zerostep_bit_identical_to_pca_tensor: true`, `settle_steps: 0` in
  scored/test_seed{0,1,2}/arms/pca_settle_zerostep/arm.json (3/3 seeds).
- np.array_equal over 256 requests: w6-compose-physics/LOG.md 08:32Z (lines 163-167).
- The algebraic-identity argument: w6-compose-physics/LOG.md 06:56Z (lines ~85-97).
- Closed-form settled state z* = (beta I - lambda W)^-1 beta z_req: LOG.md 07:24Z.
- Off-span fractions (w6.1): w61-compose-physics/LOG.md 08:30Z table.
- CRR bars B1/B5/B6: w6-potentials/RESULTS-LEDGER.md.
- +1.1385 provenance FINDING 1: w10-physics-section-evidence/EVIDENCE.md §4.2.

## 11:25–11:50 — checks written and run (2 threads, OMP_NUM_THREADS=2, no background jobs)
All three run in-turn, foreground, to completion. No PIDs detached; no kills issued.

- checks/check_00_env.py  -> PASS. chlu.__file__ =
  C:\chlu\worktrees\CHLU-w12-clue-theorist\chlu\__init__.py  (INSIDE my worktree).
  python 3.11.9, numpy 2.4.1, jax 0.9.0.
- checks/check_01_zero_dynamics.py -> ALL PASS (A..F). Writes check_01_zero_dynamics.json.
  A off-span max 9.950e-08 (wave 6 real store: 1.96e-07)
  B span-coord settle steps 0/1/10/1000 all np.array_equal=True, max|diff| 0.0
  B2 channel-coord settle: array_equal False, rel max|diff| 7.447e-08 (rounding, not motion)
  C address pushed off-span by 0.10 -> displacement 0.050000 vs closed form 0.050000 (rel err 2.2e-07)
  D atoms-as-POINTS energy -> rel displacement median 1.7244 (the settle MOVES)
  E lambda=0 -> rel displacement exactly 0.0 and array_equal(z*,z_req)=True;
    lambda 1e-4/1e-2/0.05/0.2 -> 7.875e-05 / 7.882e-03 / 3.971e-02 / 1.720e-01
  F certificate from first principles: lambda* = 0.25000000 (arm.json 0.24999999969559716),
    min eig = -1.50000000 (arm.json -1.5000000024352236), used 0.225000 (LOG 08:33Z 0.225000)
- checks/check_02_crr.py -> ALL PASS (G..N). Writes check_02_crr.json.
  G production velocity_verlet_step Jacobian vs tr=(2-g)(1-x/2), det=1-g: max|diff| 4.387e-08
  H D17 = exact log-derivative at small x: max|diff| 7.84e-04 over the x=1e-6 rows
  I the fixed-h/h* sign trap reproduces: -1.0260 at gamma=0.05 (EVIDENCE.md 4.1 says -1.026)
  J 1+g/(2-g)=1.1385 <=> gamma=0.243303; NOT on pins.gammas
  K underdamped |lambda|=sqrt(1-gamma) EXACTLY (max err 0.000e+00, 7 cells); D18 exact to 1.43e-09;
    D18=-1.0050 at gamma 0.009917, D18=-1.1090 at gamma 0.184104
  L 10**0.0433 = 1.1048; instrument floor 10**0.0349 = 1.0837
  M d14_confinement max_abs_residual = 0.0 on n=60, all 3 seeds; hessian_fd_check
    3.519263e-11 / 1.202332e-10 / 2.699430e-11 at eps_fd 1e-05, n=10
  N gamma_crit(x) table; reaching gamma=0.1841 on the underdamped branch needs
    x >= 0.010305, i.e. mu >= 2.030 at pins.eta = 0.05
- checks/check_03_provenance.py -> ALL PASS. Writes check_03_provenance.json.
  B1 re-derived from results.json estimand: 0.043256 / 0.043516 / 0.043102 vs ledger
    0.0433 / 0.0435 / 0.0431 (agree to 4.4e-05 / 1.6e-05 / 1.8e-06)
  B2 re-derived: -0.596831 / -0.579639 / -0.598172 vs ledger -0.5968 / -0.5796 / -0.5982;
    49/49 corners better than the FITTED rival on all three seeds
  scan of 56 files (53 .json, 3 .md) under w6-potentials: "1.1368" and "1.1367" appear in
    LOG.md and RESULTS-LEDGER.md ONLY; "1.1385" in LOG.md ONLY; zero .json hits
  results.json blob contains no "slope", no "d17", no "d18", no "fit_window"

## 11:50–12:05 — extra artefact check + deliverable written
- checks/check_04_byte_ledger.py added and run -> exit 0.
  FINDING (mine, reported not fixed): within each scored seed all 7 settle legs are
  byte-identical (address/coupling/standardisation/store), so the parameter-identical dial
  holds exactly; store pin 524,288 B exact on every leg and seed. BUT SCORED-TABLE.md prints
  ONE read-side column and it is the seed-0 value: address 189,344 (s0) vs 195,064 (s1,s2);
  read-side added 81,312 (s0) vs 84,216 (s1,s2). Understates the settle arm's cost on 2 of 3
  seeds -> makes the losing arm worse, never better. dev/val_seed0 carries a third ledger
  (82,368 B) and must not be mixed with scored.
- Registry check: grep of /c/chlu/evidence/claims_matrix.md and negative_results.md returns
  0 hits for pca_settle, zerostep, Curvature-Retention, 1.1385. Confirms the w10 advisor's
  "zero w5/w6 entries" disclosure. HAZARD recorded: "w6" in claims_matrix.md (CM-1/CM-5/CM-7)
  is CAMPAIGN-1's wave 6, not this programme's.
- No pkl/npz in the evidence base (find returns nothing) -> DECLARED NOT-RUN: cannot re-emit
  the deployed store's read; check_01 uses the production code path on a synthetic store and
  says so on every number.
- CLUE.md written (369 lines): the carriable paragraph + the CRR paragraph + a 10-item
  "may NOT say" list + the four conditions C1-C4 + the C6 provenance verdict + the D18
  one-row diagnosis + the byte-ledger disclosure + declared not-runs.
- checks/README.md written.
- Exit codes re-verified for all five checks: 0 / 0 / 0 / 0 / 0.

## 12:05 — commit and bundle
- Committed 5b47ab9 on branch w12/clue-theorist (only w12_clue_theorist/ added; no
  production chlu/ code touched; no .claude/ dir created).
- Bundle: git bundle create /c/chlu/outputs/w12-clue-theorist/w12-clue-theorist.bundle
  w12/clue-theorist   (that branch only, never --all). 6,036,278 B.
  verify -> "is okay", "The bundle records a complete history."
  list-heads -> exactly 1 ref: 5b47ab99e15929e04b9a783dedcb0362081aa897
                refs/heads/w12/clue-theorist
- REGENERATION CHECK: cloned the bundle into the scratchpad and re-ran all four
  substantive checks from the clone -> exit 0 each, and all four JSON outputs
  diff-identical to the ones in checks/.
- Re-bundled after the second commit. Final bundle head bfedaf4f5398c464d986d21e875a6b3f6c31af9e
  refs/heads/w12/clue-theorist; verify "is okay" + "complete history"; list-heads = 1 ref.
  Re-cloned once more to confirm; scratchpad clones deleted.
- No background job was ever launched; no PID was killed; no box-wide kill issued.
- Nothing written outside /c/chlu/outputs/w12-clue-theorist/ and the worktree's own
  w12_clue_theorist/ directory (plus the session scratchpad, now cleaned).
- END_UTC 2026-09-21T11:18:29Z (START 2026-09-21T11:09:27Z; ~9 min wall).
- NOTE ON THIS FILE: the authoritative LOG is this one, in the outputs directory. The copy
  committed at w12_clue_theorist/LOG.md is a snapshot and necessarily ends one line before
  the final bundle head it would otherwise have to quote; the two differ only in that tail.
