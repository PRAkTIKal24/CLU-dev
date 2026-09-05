# psi-payload-residual — close the read-out's payload gap (§A20.3(a); GATES CSF3 RUN 2)

**Campaign 2, wave C2W5 (Head-ordered, first experimental TODO). Agent:** experiment-engineer.
**Worktree 1 of ≤3.** Branch `psi-payload-residual`. ~1-day build. Writes
`.claude/outputs/psi-payload-residual.md` + artifacts.

**The conviction (two independent measurements):** the dynamics DELIVER the payload — `q*[payload]`
moves 30–50 % of the way to the true value when the store is live and exactly 0.000 when blank —
and **ψ then compresses the between-item spread 7–25×** (probe §6: `q*` spread 0.053–0.114 →
decoded 0.0065–0.0117), leaving the decode near-constant and acquisition exactly at chance. The
cat test independently showed the store POOR-not-INERT (0.45–0.58 at 4×tol). The read-out is the
convicted component.

**Read first:** `.claude/outputs/pilot-placement-probe.md` **§6 in full (the mechanism + the two
diagnostic artifacts `decode_dispersion.json`/`qstar_payload.json`) + §10 (the run-1 config)**;
charter **§A20.3(a)** (your acceptance criterion verbatim) + **§A20.4** (run-2 discipline);
`chlu/core/psi_readout.py` (⚠ AttentionPsi stays QUARANTINED — do not route through it);
`chlu/core/blocks.py` (`DeepSetsPsi` usage in `CluStoreCell`); the `[C2W5]` second-review §10 entry.

## ⭐ DIAL DECLARATION (protocol §7, C2 form) — echo before your first result
- **Dial:** compute-adaptive reads (the read-out's fidelity) — tier-iii instrument + tier-ii design
  input. ⛔ No paper number; run-2 config evidence.
- **Laundering control:** live vs blank vs memory-deleted, paired seeds (the probe's arms); the
  residual must NOT create a query bypass — ⛔ **blank-store decode must stay at chance** (a payload
  residual that reads the query is N68's leak in a new coat; blocking check).
- **Falsifies:** the acceptance bar (below) not met AND the compression mechanism not localized
  further ⇒ report exactly where the payload dies (per-stage spread ledger).
- **Does NOT falsify:** acquisition still at chance even with spread restored (that would isolate
  the NEXT bottleneck — the assignment rule, §A20.3(b)'s territory, not yours); anything tier-iii
  at scale (monitor #13/N94 caveats travel).

## The build
1. **A payload-carrying residual path in ψ:** the settled/trajectory read's payload coordinates
   reach the decode without passing through the spread-collapsing pooling — e.g. a gated residual
   from `q*[payload]` (and strided trajectory payload slots) into the read-out head, gate learned,
   init to pass-through (measure, then let it train). Design freedom is yours; the constraint is
   N46 (never hand the write the value) and the blank-leak check above.
2. **Measure the spread ledger per stage** (the probe's §6 instrument, promoted): true payload
   spread → `q*` spread → post-residual spread → decoded spread; per cell (baseline · h1b_r0.3 ·
   h1b_m1.0), 3 paired seeds.
3. **The trained tier** (200 steps, per probe §7): does the residual survive outer training (R3's
   well-destruction is upstream of you — report depth beside your spread numbers; placement+margin
   config is your default per run-1).
4. **Acceptance (§A20.3(a) verbatim):** **decoded spread reaches `q*` spread** (ratio ≥ 0.5 across
   cells, vs the current 0.04–0.15), with the blank-store decode at chance — or the gap's location
   measured exactly (which stage, which magnitude) if not.
5. ⭐ **The run-2 flag block (gates CSF3 run 2):** your report ends with the decoder-fixed
   submission block — run-1's exact config (probe §10 + `plan_workers=8` + `-c 12`) **plus only
   your ψ flags** (§A20.4: same config otherwise; the pre-registered designed ablation). All knobs
   as flags via the `--set/--mem/--store` path; zero module edits.

## Rider (Head item 3, ~1 minute)
Correct the two stale rows in `.claude/outputs/orgdiv-cat-test/FROZEN-interfaces.md` **in place**
(the capacity-ledger line → `a = 32`, 1024 atoms, 57 344 B, ratio 9.67×; reader params → 72/92),
leaving the Hub's dated erratum banner beneath as the record of the correction. The curator records
the erratum registry-side.

## Acceptance + hygiene
Spread ledger delivered · acceptance bar adjudicated mechanically · blank-leak check green · run-2
flag block ready · tests green on your branch (new tests for the residual path + the leak check) ·
declared NOT-RUNs never nulls. **Ownership:** `psi_readout.py` (quarantine untouched) + additive
hunks in `blocks.py` + your experiment/test files + the FROZEN-interfaces rider. ⛔ Not
`train_cluformer.py`'s lane-parallel code, not the factored-store/null-arms files. **Git:** branch
+ scoped worktree; never push `origin`; `clu-dev` only. Report → Hub, spawn nothing.
