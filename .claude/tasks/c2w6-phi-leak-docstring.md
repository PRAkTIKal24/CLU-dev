# c2w6-phi-leak-docstring — correct `write_sign`'s false claim (charter §A23.3, engineer half)

**Campaign 2, wave C2W6 close-out. Agent:** experiment-engineer. **Branch
`c2w6-phi-leak-docstring`** from `main @ d1149a4`. **No worktree needed** (docs-in-code only; if you
prefer one, wt1 is free). **~20 minutes.** Writes `.claude/outputs/c2w6-phi-leak-docstring.md`
(short — a diff, a test note, and the verification lines; no dial declaration, no prereg: this
task measures nothing).

⛔ **Mechanical precondition (standing doctrine, §A23 process lesson): none — this task has no
gate and may run immediately.**

## The finding you are recording (already measured, do NOT re-measure)

`StreamMemoryConfig.write_sign`'s docstring argues that `jnp.sign`'s zero derivative severs
`d(store state)/d(phi)`, making the trajectory read the only channel to φ *"by construction as well
as by theorem"*. **That is true of the inner-loop path and FALSE whenever `atom_place_radius > 0`**
— which is the run-1/2/3 config. H1b's localized placement
(`centers[:, :addr] = z[:addr] + jig`, `blocks.py` ~l.1292–1304) is a plain differentiable
assignment of φ's output into the store's state, outside the sign-gated loop.
**Measured and Advisor-verified (charter §A22): φ's layer-0 gradient 0.0908 → 0.0659 under P1 =
27 % of it flowing through the write.** `erosion_partition=True` is what closes it.

## The job
1. **Correct the docstring** in `chlu/core/blocks.py` (`write_sign`, and any adjacent comment making
   the same claim): state the severance as **conditional on `atom_place_radius == 0`**, name the
   placement path as the live exception in the shipped run-1/2/3 config, cite the 27 % measurement
   and `erosion_partition` as the closure. Keep it to the repo's comment idiom — a constraint the
   code cannot show, not a change log.
2. **Grep the codebase for the same claim elsewhere** (`grep -rn "sever\|by construction" chlu/`
   around φ/write/sign) and correct any other instance; **list every hit you found and what you did
   with it**, including hits you judged fine — the curator needs that list to propagate the
   qualifier program-wide (§A23.3's other half).
3. **One regression test** pinning the corrected behaviour so the docstring cannot silently drift
   back: assert the φ gradient is **non-zero** with `atom_place_radius > 0` and `erosion_partition
   False`, and **exactly 0.0 through that path** with the partition on. (The in-system probe already
   exists — `k1_insystem.py` — reuse its construction rather than writing a new rig.)

## Constraints
⛔ **Do not touch behaviour** — this is a comment fix plus a test; any `.py` change that moves a
number is out of scope and must be flagged to the Hub instead. ⛔ Do not touch
`chlu/core/multiplicity_read.py`, `monitors.py`, `experiment_cmd.py` (**C2W7's**), or
`psi_readout.py` (quarantine). ✅ Run `tests/test_anti_erosion.py` + `tests/test_blocks.py` + your
new test; ruff green. ⛔ **K1's exact-zero probe and K2's fingerprint test must stay green — they
are the Head's ship condition for `erosion_partition` (§A23.2 (i)+(iv)); if either reds, stop and
report, do not adjust the fingerprint.** ⛔ Never push `origin`; do not push `clu-dev` (the Hub
pushes after merge).
