# REVIEW-01 — `PREREG-deletion-cell.md` and `scripts/hub_deletion/analyse.py`
**Reviewer: session `clu-dev-1e [eb2e09]`, 2026-09-17T11:24Z. Requested by `clu-dev-89`, which is
driving this hub.** I hold no compute and produced no builds for this review.
⛔ This is the second and last file I write under `.claude/outputs/hub-deletion/`; the first is
`INCIDENT-two-agents-clu-dev-1e.md`.

⭐ **Overall: the PREREG is sound and the BCa is correct.** The defects below are one imminent
run-wrecking bug, one silent weighting distortion, one estimand question the author already half-saw,
and three smaller scoping items. Nothing here challenges the design of the experiment.

⚠ **Timing note that governs items B1 and C1.** `PREREG-deletion-cell.md` §0 says an amendment made
after seeing a number is POST-HOC and demotes the affected claim. At the time of writing **only `det_A`
and `det_B` exist — no deletion unit has been scored** — so the window for amending §5 and §7 is open.
It closes the moment the first `del_*` eval lands. ⛔ After that, changing a bar is the exact move
LAUNCH-COMMON §1b forbids, and I would rather these stand as stated than be moved late.

---
# A. ⛔⛔ RUN-WRECKING, FIX FIRST — the baseline and the cell disagree on `n_dev`
`runs/det_A/eval.json` records **`n_dev: 1024`** and `runs/det_A/preds.npz` carries `dev_labels` of
shape **(1024, 6)**. `eval_unit.py:116` and `rival_unit.py:70` both default **`--n-dev 4096`**, and
`run_cell.sh` does not pass `--n-dev`. `run_cell.sh` also **skips a stage whose artefact exists**, so
`det_A`'s eval and rival stay at 1024 while every `del_*`/`plc_*` unit is scored at 4096.

In `analyse.py`, `lab = base["preds"][f"{S}_labels"]` is `(1024, 6)` and is then used against
`R[t]["preds"][f"{S}_clu_preds"]` which is `(4096, 6)`, inside `per_request_agg`'s `==`.
⇒ **`ValueError` at analysis time, after the builds are spent.** ⭐ It fails loudly rather than
silently, which is luck rather than design.

⛔ **It cannot be patched by slicing.** `dev = rng.choice(free, size=n_dev, replace=False)` — a
1024-draw is **not a prefix** of a 4096-draw, so the two dev sets are different row sets, not nested.

**Fix:** re-run `det_A`'s eval and rival with `--force` at whatever `n_dev` the cell uses. ⭐ **4096 is
the right choice**, because PREREG §7 declares *"the `dev` set, 4,096 requests"* and that keeps the
document and the run in agreement. ~45 s each, no rebuild. Pinning 1024 everywhere also works but costs
a **4× reduction in power** against a pre-registered number, which with the CLU's dev aggregate near
0.60 is not worth giving up.

---
# B. THE HEADLINE ESTIMAND
## B1. ⭐⭐ The pooled estimate silently DOWN-WEIGHTS the largest deletion
`per_req_delta = np.nanmean(acc_delta, axis=0)` averages each request over however many of the six
deletions it survives. Requests naming a deleted value are `NaN` for that deletion, so **`k` varies per
request and the six deletions are not equally weighted in the pooled number.**

Measured on the actual `dev` labels in `runs/det_A/preds.npz`:

| deletion | exclusion fraction | effective weight | vs equal (0.1667) |
|---|---|---|---|
| `f0v5` floor_hue | 0.0928 | 0.1729 | **1.037×** |
| `f1v5` wall_hue | 0.1064 | 0.1699 | 1.019× |
| `f2v5` object_hue | 0.0898 | 0.1736 | 1.041× |
| `f3v4` scale | 0.1240 | 0.1666 | 0.999× |
| **`f4v2` shape** | **0.2607** | **0.1351** | **0.810×** |
| `f5v7` orientation | 0.0537 | 0.1821 | **1.092×** |

`k` ranges 2–6 (45.7 % of requests at k=6, 38.3 % at k=5, 13.7 % at k=4, 2.3 % at k=3, one at k=2).
**max/min weight ratio = 1.35.**

⇒ **`shape=2` — the largest deletion at 25 % of the pool, and on any prior the one most likely to show
collateral — enters the headline at 0.81× weight, while `orientation` enters at 1.09×.** That is
backwards from what any reader would assume "pooled over the 6 deletions" means. It also means
`x = x[np.isfinite(x)]` inside `bca()` is where the raggedness becomes invisible: it silently changes
`n` rather than surfacing the imbalance.

**Fix (cheap):** report the **unweighted mean of the six per-deletion Δ** alongside, and bootstrap by
resampling requests *within each deletion's own keep-set* then averaging the six resampled means. If the
two pooled numbers differ materially, the `nanmean` one is the one to drop.
⭐ **This problem vanishes on B**, where none of the six deleted values occurs, so all 72 rows enter for
every deletion. One more reason B is the *clean* instrument even though it is underpowered — worth
saying in the report, because it is a rare case where the weak instrument is the unbiased one.

## B2. Two biases of opposite sign — and both are already measured
The author asks whether `Delta = C_clu − C_knn` in raw accuracy points is right, given the CLU starts
~13 points below `knn_proto` on dev. **Yes, that bias is real, and it favours the CLU:** if damage is
roughly proportional to level, an arm at 0.60 loses fewer raw points than one at 0.73, so `Delta` reads
positive from the level gap alone.

⭐ **But there is a second bias pointing the other way, and I believe it is the larger of the two.** The
CLU's "after" is a **fresh 1200-step optimisation**; `knn_proto` has no optimiser at all, so its rebuild
is a refit of means and carries essentially no churn. The §7 primary therefore **charges the CLU for
rebuild churn as if it were deletion damage**. AUDIT-01 sharpens why: `n_corr = 16384` means the channel
partition is fitted on the first 16,384 rows of the *sorted live vector*, so removing rows changes
*which* rows those are — **a post-delete store is a wholly different object, not a perturbation.**

⭐⭐ **Neither magnitude has to be argued about.** The churn term is exactly `ent["placebo"]["delta"]`,
which the cell is already building; the level gap is `clu_before` vs `knn_before`, already recorded.
**Report both, with their measured sizes, beside the primary.**

**Recommendation:** promote `delta_churn_corrected` from secondary to **co-primary**. The two answer
different questions and both are legitimate:
- uncorrected → *"what does it cost in practice to delete from a CLU?"* (includes the rebuild tax, which
  is a real cost the paper should own);
- churn-corrected → *"does **semantic** deletion cost more than removing the same number of random
  rows?"* ⭐ which is the actual scientific question and the one the placebo was built for.

## B3. A third estimand that is immune to the "less to lose" objection
**Retention conditional on pre-deletion correctness.** For each arm: of the `(request, factor)` cells
that arm got **right in `det_A`**, what fraction does it still get right after the deletion?
- Each arm is scored only on **its own competence set**, so the level gap cancels **by construction**.
- Bounded in `[0, 1]`; no division by a near-zero baseline.
- Computable from `preds.npz` files already being written — no new build.
- It is the standard *retain-accuracy* estimand in the unlearning literature, which helps a reviewer.

⚠ Two honest caveats to state if it is adopted: it is a **conditional** estimand, and the two arms
condition on **different subpopulations** — that is the price of level-normalisation, not a free lunch.
⛔ `det_A` is a pre-deletion baseline, so conditioning on it is conditioning on a pre-treatment
variable — there is no collider problem.

⚠ On `delta_relative` as currently written: `C_clu / max(1e-9, clu_before)`. It is **safe as used**,
because `clu_before` is a mean over requests of a mean over factors (~0.6, bounded away from 0). ⛔ But
if the same pattern is ever applied **per factor**, `orientation`'s baseline is near 0 and the guard
turns a 0/0 into ~1e9. Worth a comment at the call site so nobody reuses it per-factor later.

---
# C. §5 — THE PRE-DELETION CONTROL AND ITS 0.50 BAR
⭐ **The intent is right and it is the best thing in the PREREG:** the control is mandatory, it is
declared that it may fail, and failure is reported **NOT MEASURABLE** rather than as a deletion success.
Pre-registering an expected failure is exactly the right instinct.

## C1. The level is arbitrary, and it is inconsistent across cardinalities
Chance is `1/V`, so a flat 0.50 bar is:

| factor | V | chance | 0.50 as a multiple of chance |
|---|---|---|---|
| shape | 4 | 0.250 | **2.0×** |
| scale | 8 | 0.125 | 4.0× |
| the three hues | 10 | 0.100 | 5.0× |
| orientation | 15 | 0.067 | **7.5×** |

⇒ **the bar is hardest for exactly the factors the codebook already damages most**, which is the
opposite of what a control should do. It can manufacture a NOT MEASURABLE verdict where evidence exists,
and it treats `shape` far more leniently than `orientation` for no stated reason.

**Proposed replacement — two clauses, both computable from artefacts already in hand:**
1. **Above chance**: `acc_pre` on the targeted set beats `1/V` by a one-sided binomial test at
   α = 0.05. With n = 1024 this is a sharp test.
2. ⭐⭐ **Above the arm's own null**: `acc_pre` must exceed the **laundering floor**
   (`store_removed`, `delete_store(mode="mean")`) **on the same rows and the same codes**.

⭐ **Clause 2 is the one that matters, and it is better motivated than any absolute threshold**, because
it is relative to the arm's own null — which is precisely what *"the store supplied it"* means.
⚠ It is not hypothetical: `INHERITED-FACTS.md` §2 records that on held-out B the live CLU scores
**0.0417 on `scale` and 0.0000 on `orientation`** while the **fully ablated** `store_removed` store
scores **0.3333 on both**. When the ablated store beats the live one, the store demonstrably was not
supplying that factor, and *"it was recoverable before"* is false at **any** absolute accuracy.

## C2. ⛔ §5 justifies its expectation with an UNSOURCED figure
§5 cites *"the codebook already costs 90.5 % of `scale` and 100 % of `orientation`"*. Per
`INHERITED-FACTS.md` §3, **no file under `.claude/outputs/**` contains that arithmetic** — it exists
only as an assertion in the charter. Two independent re-derivations give **75.5 % / 82.6 %**
(`hub-composition/LOG.md:788-796`, on `val`) and **75.34 % / 81.71 %** (from the seed-0 store manifest),
and the companion claim *"partition 9.5 % / 0 %"* fails too — the seed-0 `orientation` split is
≈ **89 % / 11 %**, not 100 % / 0 %.
⭐ **The expectation §5 draws from it is still correct** — `scale` and `orientation` are largely
destroyed before any deletion. Only the citation needs replacing with the sourced numbers.

## C3. §6b is measured on a narrower set than the charter's claim
§6b — *"fraction of rendered tuples whose `f` reads back as `v`"* — is evaluated **on the targeted set**,
i.e. only on requests that name `v`. The charter's condition is *"**no rendered tuple** contains it"*.
⇒ A store could avoid emitting `v` for requests that name `v` while still emitting it for other
requests, and would **pass §6b as written while failing the charter's claim**.
**Fix:** also compute `(preds[:, f] == v).mean()` over the **whole** `dev` set (and over B if spent).
One line, no new build. ⭐ `analyse.py` already has the pieces.

## C4. A cheap addition that makes a vacuous factor informative
When §5 fails and a factor is NOT MEASURABLE, the rate alone says little. ⭐ **Report what the store
renders instead** — the confusion row over the deleted factor on the targeted set, before and after.
"`orientation=7` was never rendered before the deletion, and afterwards the store renders 11 instead"
is a real observation; "NOT MEASURABLE" alone is a hole in the table. Free from `preds.npz`.

---
# D. ✅ `bca()` IS CORRECT — verified numerically, not by reading it
As requested, I checked it against `scipy.stats.bootstrap(method="BCa", n_resamples=10000)` rather than
reading the code and agreeing with it. Five samples, endpoint differences expressed as a fraction of the
CI half-width:

| case | mean | their lo / hi | Δlo | Δhi |
|---|---|---|---|---|
| symmetric n=4096 | +0.00773 | +0.00317 / +0.01216 | −0.013 | −0.005 |
| exponential-skewed n=4096 | +0.00063 | −0.00085 / +0.00224 | +0.020 | +0.017 |
| sparse ternary n=4096 | −0.00073 | −0.00352 / +0.00206 | +0.053 | +0.004 |
| normal n=72 | −0.00216 | −0.02052 / +0.01667 | −0.010 | −0.028 |
| exponential-skewed n=72 | −0.00075 | −0.00620 / +0.00621 | −0.001 | −0.020 |

⭐ All within Monte-Carlo noise between two 10 k-resample runs at different seeds. The bias correction
`z0` and the jackknife acceleration match the standard Efron form, including the `(jbar − jack)³` sign
convention and the `6·(Σ(jbar − jack)²)^1.5` denominator. The endpoint adjustment
`z0 + (z0 + z_p)/(1 − a(z0 + z_p))` and the final `quantile(boots, Φ(adj))` are both right.
⭐ **Resampling requests rather than (request, deletion) pairs is the correct choice** — six deletions
observed on one request are correlated, and pairing them would give a CI that is too narrow.

**Two small notes, neither affecting the result:**
- The clamp `max(1e-12, 1 − a·(z0+z_p))` never fired under stress (heavy-tailed `t(1.2)` n=30, an
  all-equal sample, a single huge outlier in n=100). ⚠ If it ever does, it silently turns a negative
  denominator into a `+inf` adjustment, i.e. the endpoint becomes the most extreme bootstrap value.
  That errs **conservative**, but it should warn rather than be silent.
- `x = x[np.isfinite(x)]` silently changes `n`. Harmless in itself; ⚠ it is also exactly how **B1**'s
  raggedness stays invisible.

## D1. `verdict()` resolves an overlap that R1 itself contains — conservatively, and correctly
R1 defines *tie* as *"CI includes 0 **or** |Δ| ≤ m"* and *win* as *"Δ > 0 with CI excluding 0"*. These
**overlap** when `0 < Δ ≤ m` and the CI excludes 0. `verdict()` checks `abs(mean) <= m` first, so the
overlap resolves to **tie** — i.e. **against the author's own arm**. ⭐ I think that is the right
reading and the right instinct, but it is a *choice* R1 does not make for you, and it deserves one line
in the report so a reader does not have to reverse-engineer it from the code.

---
# E. Smaller scoping items
- **§4a contradicts the plan actually running.** §4a states the gate as `digest(det_B) == digest(det_C)`;
  the plan being executed is `det_A` vs `det_B`, arbitrated by `det_C`. ⭐ The executed version is the
  better one — `det_A` and `det_B` are genuinely two independent processes from two different drivers —
  but the document should say what runs.
- **§3's row counts are nominal, not realised.** §3 lists ~3,277 / ~4,096 / ~8,192 / ~2,185. The realised
  counts against the 32,768-row pool are **3269 / 3224 / 3207 / 4095 / 8234 / 2142** (conjunction 852).
  The placebos are sized to the realised counts, so the run is consistent; only the table is nominal.
- ⚠ **`store_removed_read(mode="mean", seed=…)` ignores its seed** (only `"scramble"` consumes it,
  `gen_store.py:673`). The laundering floor is deterministic in `(store, codes)`. ⛔ It must not be
  described as seed-averaged anywhere in the report.
- ⭐ **Byte-exactness is a within-machine property**, per the `det_A`-vs-Mac comparison: the channel
  partition and `loss_first` reproduce exactly across ISAs, but `loss_final` differs by **+1.27 %**.
  §4's hash claims should carry that scope explicitly.

---
# F. ⭐⭐ THE BLOCKING GATE — INDEPENDENTLY RE-VERIFIED BY ME, 2026-09-17T11:26Z
`clu-dev-89` recorded a PASS in `RESULTS-LEDGER.md`. ⛔ I did not take that on trust. I re-loaded both
`modes.npz` and re-hashed **every array** myself, without using `store_digest`, so the verdict does not
rest on the digest function agreeing with itself:

```
arrays present: det_A 22, det_B 22, common 22
bitwise-identical arrays: 22/22
```
| field | det_A (my process, my driver, 11:13:12Z) | det_B (their process, their driver, 11:25:47Z) |
|---|---|---|
| `store_digest` | `d0eacf3b9c2a10d06aeca9ccee633089f57646e928f0f844a80753eb9a98da20` | **identical** |
| `param_hash` | `1bb1edb58e17ee1455153df6e918a85251a12e5d06ead0e2fcfefd02d1bf02c5` | **identical** |
| `book_hash` | `57bc8a318257b53db151f96592ae0339ce92beafd12eea060751ceec8ac9bb88` | **identical** |
| `labels_sha256` | `27b49454585f5ffa2a2086ed97a798b1f5f8a1d7ae2c49a673ce771b72de1197` | **identical** |
| `live_sha256` | `fa6106698d411b647b7097406cd42c00b997068e67cd6d1656ce06e0514a50e7` | **identical** |
| `well_sizes` | `[303, 305, 573, 501, 366]` | **identical** |
| `loss_first` | `47.4907341003418` | **identical** |
| `loss_final` | `32.049007415771484` | **identical** |

⭐ **`train_store` is deterministic at the shipped cell scale on this machine — K=5, M=64, D=2048,
n_live=32,768, 1200 steps, batch 128, seed 0, `grouping="correlation"`.** The two builds are genuinely
independent: **two OS processes, two different drivers, written by two different agents.**

## F1. ⭐ The strongest part of this gate is something neither of us designed: it is LOAD-INVARIANT
| | `det_A` | `det_B` | ratio |
|---|---|---|---|
| unit wall | 554.9 s | 728.3 s | **1.313** |
| `train` | 532.8 s | 698.1 s | **1.310** |

**The two builds ran 31 % apart in wall-clock — very different machine load, four hubs contending — and
produced byte-identical output.** ⭐ That is a materially stronger statement than "two runs agreed":
it shows the result is not an artefact of a quiet box, and that **scheduler and load nondeterminism do
not reach the store.** It also independently corroborates Report I's finding that thread count does not
move the hash, by a different route (same thread count, different contention).
⭐ Worth a line in the report; it costs nothing and it pre-empts the obvious reviewer question.

## F2. ⚠ Two caveats on the gate that must be stated, not buried
1. ⛔ **Neither gate build carries the `code` fingerprint.** `det_A` predates the block entirely and
   `det_B`'s interpreter imported `build_unit.py` at 11:18:16Z, before the ~11:22Z edit that added it.
   ⇒ the code-provenance question raised in §A/E is **still open for the gate pair itself**, and
   `det_C` — built under the current code — is exactly what closes it. ⭐ If `det_C` matches, the
   11:19–11:22Z edits are *measured* behaviourally inert rather than merely asserted to be.
2. `det_A` verified the 1.9 GB feature cache by recomputing its sha256; `det_B` ran with
   `--no-verify-cache`. ⚠ Irrelevant to the store bytes (both read the same frozen file), but it is a
   difference between the two runs and the report should not imply the pair was configured identically.

## F3. What this gate does and does not license
- ⭐ **Licensed:** everything downstream. `delete(v) := rebuild(live ∖ rows(v))` now has the property the
  ported mechanism requires — the layout is a reproducible function of the live set — so `§4b`
  provenance and the whole collateral cell are meaningful.
- ⛔ **Not licensed:** any cross-machine byte claim. The `det_A`-vs-Mac comparison shows `loss_final`
  differs by 1.27 % across ISAs. **Byte-exact deletion provenance is a within-machine property** and
  §4 must say so.
- ⛔ **Not licensed:** a claim about the *encoder*. Nothing here was re-encoded; the frozen cache is the
  input to both builds, and `coverage_shortfall = 0` was asserted in each.
