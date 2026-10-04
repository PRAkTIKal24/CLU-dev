# PREREG addendum 1 — `gen-wells-route-b`

**Filed 2026-09-15, during the BUILD, before any scored measurement.**
⛔ `PREREG.md` is **not edited**; this is additive, in the `PREREG-addendum-N.md` form the
`gen-dynamical-read` prereg established.

---

## A1. ⛔⭐ A null that was manufacturable by a learning rate — caught in the build, not in the report

**What happened.** The first end-to-end verification run of this harness completed the
label-free placement arm and hardened it with:

```
  [learned_label_free] hardened to the reference histogram; 0/2048 coordinates (0.0 %) moved well
```

**Zero coordinates moved.** The learned partition was *bit-identical* to the inherited one,
so its margin over the D-matched random floor was exactly the inherited partition's margin.

⚠ **Read the falsifier and notice the trap.** `PREREG.md` §6b's named falsifier is
*"learned placement does not beat D-matched random placement ⇒ placement was never the
lever"*. A frozen relaxation produces **precisely that observation** — and it would have
been reported as the pre-registered null, in the pre-registered words, with a straight
face. ⛔ **That would have been a wrong finding, not a conservative one.**

**The cause, measured not guessed.** The relaxation carries `K·M·D = 5·64·2048 = 655,360`
mode floats against `D·K = 10,240` placement logits. Under one shared Adam rate the modes
absorb essentially all of the gradient and the logits do not move far enough to cross the
warm-start gap before `harden_to_histogram` takes the argmax.

**The change.** The placement logits get their own Adam rate:
`placement_logit_lr_mult: float = 30.0` (a multiple of `placement_lr`), implemented with
`optax.multi_transform` over a two-label pytree. ⛔ Nothing else about the objective, the
arms, the floor, the metric or the byte ledger changes.

**Why this is not a prereg violation.** It is an **optimisation** hyper-parameter fixed
during the build, before any scored number existed, and it is declared here in advance of
the scored run. ⛔ It was **not** chosen by looking at a sharpness margin — it was chosen by
the mechanical criterion *"the placement must be able to move at all"*, which is a property
of the optimiser and not of the outcome. The predictions in `PREREG.md` §6a/§6b stand
**unchanged**.

**The guard, so it cannot come back.**
`tests/test_gen_wells.py::test_the_placement_logits_actually_MOVE_under_the_objective`
fails if the relaxation is frozen, and asserts the `mult = 30` run moves strictly more
coordinates than the `mult = 1` run. ⭐ **A null must be earned, not manufactured by a
learning rate.**

⚠ **Standing caveat for whoever reads the scored run.** A "no margin" result is only a
finding when `placement.fraction_moved` is **non-zero**; that field is written into every
learned arm's record for exactly this reason. ⛔ A null at `fraction_moved == 0` is an
optimisation artefact and must be reported as one.

---

## A2. Two build-time defects fixed, recorded because both were silent

1. ⛔ **`--quick` was not quick.** `select_store_rows` reads
   `config.gen_store.n_store_train`, **not** this block's field, so a `--quick` smoke
   gathered the full **32,768** rows while printing a perfectly normal
   `rows: store_train=32768`. Fixed by forwarding the value into the module actually
   reused. ⚠ A quick run that is not quick is a quick run nobody re-runs.
2. ⛔ **The weakly-supervised arm could not survive `jit`.** `weak_supervision_term`
   inferred its one-hot width with `y.max()`, which reads a **traced** array inside the
   jitted step. The failure mode was nasty: the *un-jitted* label-free gate passes and logs
   `label-free check FIRED as designed` immediately before the arm dies. Fixed by passing
   static `cardinalities`; guarded by
   `test_gen_wells.py::test_the_weak_sup_term_survives_JIT`.

---

## A3. ⚠ A cluster-leg blocker found while writing the jobscript — NOT mine to fix

`gen-cluster-stage-artefacts`' `STAGED.json` stages **44 files / 1.95 GiB**, and under
`data/gen/` it stages **exactly one**: `shapes3d/c_invalid.json`.
⛔ **`3dshapes.h5` is not in the staged set**, and neither is `factor_index.npz`.

This run needs it twice: `build_split`/`factor_labels` read the label array from it, and
`_assert_cache_reencodes` streams images through the frozen encoder from it.

⛔ **I have not improvised a staging path** — the task file forbids it and a peer spoke owns
that file set; two staging scripts would diverge silently. What I have done instead is make
the jobscript **name the missing file at job start**, so the failure is instant and free
rather than arriving inside the loader after the scheduler has allocated a GPU.
⇒ **Reconciliation item, owner Hub → `gen-cluster-stage-artefacts`.**
