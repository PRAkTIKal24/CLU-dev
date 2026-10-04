# PREREG-01 — `hub/deletion-2`, the DEV design
**Written 2026-09-17 ~11:05Z, BEFORE any full-scale number exists.** START_UTC 2026-09-17T11:03:31Z.
Seed 0 only. Machine: FORGIS_14 (Windows 11 AMD64), 4 threads, no CUDA. Model: Opus.
⛔ This file is not edited after the first number it governs is produced; corrections are appended
as dated addenda with the reason, never as silent rewrites.

---
## 0. What is being claimed, and what is explicitly NOT being claimed
**Claimed:** on the generative store, `delete(v)` implemented as *canonical rebuild from the live row
set minus `rows(v)`* is (i) **exact** — byte-identical to a store that never held `v` — and (ii) has a
**measurable collateral cost on the rest of the space**, which is compared against `knn_proto` at
matched bytes.

**⛔ NOT claimed:** certified (ε, δ) unlearning. **⛔ NOT claimed:** anything about the frozen encoder,
which is a separate channel and is *not* rebuilt (caveat carried verbatim from *Retention You Can
Predict, Scope and Delete*, App. G.8). **⛔ NOT claimed:** that "the rival has no delete operator" —
`knn_proto` is given the identical predicate and does have one.

⚠ **The trap, named up front (task file):** exactness is **true by construction** — if the layout is a
canonical function of the live set then `delete` *is* `rebuild` and the hash equality is a tautology
about my own code. ⭐ **So exactness is a METHOD-SECTION fact in this report, never a headline.**
The headline is §4, COLLATERAL, which is not true by construction in either direction.

---
## 1. ⭐ THE BLOCKING GATE (running as this is written): determinism at real scale
Object: the shipped cell — `K=5` wells, `M=64` modes, `D=2048`, `n_live=32768`, `train_steps=1200`,
`train_batch=128`, `grouping="correlation"`, seed 0, features from the frozen Mac-written cache
(`scope="full"`, 480,000 rows, `features_sha256 d6728f01...`, verified by recompute in `det_A`).

Two **independent OS processes** `det_A`, `det_B`, both `spec={"kind":"full"}`, both OMP=MKL=OPENBLAS=4.
**PASS** := equality of all of `store_digest`, `param_hash`, `book_hash`, `labels_sha256`,
`live_sha256`, `well_sizes`. **FAIL** => ⛔ stop, report both digests + the batching, run nothing
downstream (falsifier §5.5). ⛔ "byte-identical" is never softened to "close".

⚠ The predecessor's smoke-scale pass is **not** cited as cover: it predates the current digest (no
`book_hash`), see LOG. Smoke scale is not the claim.

## 2. The deletions — the rule is fixed HERE, before any of them is run
⭐ **One rule, no per-factor choices, so nothing can be cherry-picked:** for each of the 6 factors,
delete the **middle value `v = card // 2`**.

| f | factor | card | deleted value index |
|---|---|---|---|
| 0 | floor_hue | 10 | 5 |
| 1 | wall_hue | 10 | 5 |
| 2 | object_hue | 10 | 5 |
| 3 | scale | 8 | 4 |
| 4 | shape | 4 | 2 |
| 5 | orientation | 15 | 7 |

Why the middle and not index 0: index 0 is the extreme of every one of these factors (darkest hue,
smallest scale, most extreme rotation), so a value-0 rule would confound "deleted" with "boundary".

⚠ **Declared consequence, noticed before running and reported because it is convenient to me:** the
held-out B set is `fh,wh,oh >= 0.8 AND scale >= 1.1 AND shape = cylinder(raw 1) AND orientation > 20`,
so **none of the six deleted values occurs in B**. => every B request is a *non-`v`* request =>
**B is a pure collateral probe under this rule.** That is a property of the rule, not a choice made
after seeing B, and it is stated here so it cannot later look like one.

**Conjunction (the gs task file's item 3, DEV only):** `object_hue = 5 AND shape = 2`.

## 3. ⭐ THE CHURN CONTROL — without it §4 is uninterpretable
A CLU `delete` is a **fresh 1200-step optimisation**, so the rebuilt store differs from the original
even when the deletion is scientifically empty. => for every deletion tag `del_f{f}v{v}` there is a
**placebo** `plc_f{f}v{v}` that removes the **same number of rows chosen uniformly at random** from the
same pool, rebuilt identically. ⛔ A raw pre-vs-post delta is **not** reported as collateral.

**Collateral := Delta_delete − Delta_placebo**, and the placebo spread over repeats is the noise band.

⚠ `knn_proto` has no optimiser, so its placebo delta is expected ~ 0 — that asymmetry is *the point*
and is reported, not hidden: it is the price the CLU pays for being trained rather than tabulated.

## 4. ⭐⭐ THE HEADLINE — COLLATERAL, CLU vs `knn_proto`, matched bytes
For each deleted value `v` of factor `f`, on the request set restricted to rows **not** naming `v`:

- `acc_pre(f')` = per-factor probe accuracy from the undeleted build (`det_A`);
- `acc_post(f')` = same from the `del_f{f}v{v}` build;
- `D_arm(f,f') = acc_post − acc_pre`, paired per row, for `arm` in {`clu`, `knn_proto`};
- **collateral** `C_arm(f) = mean over f' != f of ( D_arm(f,f') − D_placebo_arm(f,f') )`;
- **the reported statistic** `Delta(f) = C_clu(f) − C_knn(f)`, and its mean over the 6 factors.

**Decision rule = Head R1, set now, before any number:** *tie* := paired 95 % CI on Delta includes 0
**or** |Delta| <= m = 0.020; *win* := Delta > 0 with CI excluding 0; *loss* := Delta < −m with CI
excluding 0.

⭐ Direction of interest: `Delta >= 0` means the CLU's deletion costs the surviving space **no more**
than a lookup table's does. ⛔ A loss is reported as a loss; it is a real and publishable finding that
a trained store pays a rebuild tax a table does not.

**Bytes** are reported exactly, not asserted matched: CLU store = `64 x 2048 x 4 = 524,288 B`;
`knn_proto` = its own `byte_ledger()` (58 prototypes x 2048 x 4 = 475,136 B expected — ⚠ the rival is
*smaller*, which is conservative **against** the CLU, and is reported as such).

**Cost line:** rebuild wall-clock per delete, measured, `O(n)` per the ported mechanism.

## 5. The four charter conditions — where each is measured
1. **Unrecoverable.** On `dev` + `a` corners, requests that DO name `v`: external-probe accuracy on
   factor `f` at value `v`. ⭐ **The pre-deletion control is mandatory** (falsifier 2): if `acc_pre` on
   `v` is already at chance (⚠ the codebook is known to cost 90.5 % of `scale` and 100 % of
   `orientation`), then "unrecoverable after" is **vacuous for that factor and is reported as vacuous**,
   ⛔ not as a success. Chance = `1/card`.
2. **Preserved.** §4's `acc_post` on non-`v` requests, against the placebo band.
3. **Provenance.** `store_digest(delete(pool, v)) == store_digest(build(pool minus rows(v)))`, asserted
   in code. ⭐ Reported as method-section, per §0.
4. **Rivals.** `knn_proto` gets the identical predicate (`rival_unit.py` reads the live vector off the
   CLU build's own artefact and re-asserts `live_sha256`). `lace` / `ebm_compose`: **NOT RUN** unless
   time permits, with a method-section line either way.

⭐ Reported beside every read: the **laundering floor** `store_removed_read(mode="mean")` at identical
codes and bytes (⛔ *not* my delete operator — it is `gen_store.delete_store`'s stated purpose), and
the **feature-addressed** companion, to separate "the STORE lost `v`" from "the ADDRESSING lost `v`".

## 6. The scored run — 1 max, seed 0, and its bar is stated NOW
⛔ Not fired until (i) the gate in §1 passes and (ii) the §4 statistic on **dev/A** (non-held-out)
clears the R1 bar with the §3 placebo control present. Then **one** run adds `--with-b` (the 72
held-out rows) to the already-built stores — ⛔ no rebuild, no encode, coverage shortfall asserted 0.

**Pre-registered bar for the scored cell:** `Delta` over the 6 deletions on B is a **win or tie under
R1** (`Delta > −0.020` or CI includes 0). Reported as *"scored 1/3 — seed 0 only, 3-seed pending full
caches"*. ⛔ If it is a loss, it is reported as a loss.

## 7. Falsifiers I accept in advance
- §1 fails => ⛔ everything downstream is void and I say so.
- `acc_pre` on `v` at chance => condition 1 vacuous **for that factor**.
- The conjunction re-composes after deletion => ⭐ *"a compositional memory cannot forget a conjunction
  whose parts it still holds"* — an honest negative and a real result, reported as the finding.
- `Delta < −m` with CI excluding 0 => the CLU's deletion damages the surviving space more than the
  table's, and that is what the report says.
