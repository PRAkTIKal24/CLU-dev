# w12 hub/single-design-crossover — where does composition break as the partition gets finer?

- **Agent:** autonomous hub. **Targets:** `single-design` (**3/3 intact**) + `generation-table` (**run 3 unspent**).
- **Output:** `/c/chlu/outputs/w12-single-design-crossover/`. **Branch:** `w12/single-design-crossover`.
- **GATED — mechanical:** ⛔ do not start the SCORED leg until
  `/c/chlu/outputs/w12-deletion-baselines/BASELINES-READY.json` exists. ⭐ The **val sweep is free**
  and may start immediately.
- ⭐ **5 seeds IF** `/c/chlu/outputs/w12-cache-gen2-seeds-3-4/CACHE-READY.json` exists when your scored
  leg starts; ⛔ otherwise **3 seeds, DECLARED**. Never mix 3-seed and 5-seed rows in one plot
  without a printed `n`.

## ⭐⭐ THE BYTE PIN IS GONE. THE SINGLE DESIGN IS 1×.
⛔ **Do not re-run wave 8's experiment.** w8 already answered it: at the 524,288 B pin the covariance
forces K ≥ 36 and composition dies above K ≈ 5 — **the two properties are incompatible AT THAT PIN.**
⭐ **This wave changes the pin, not the object.** Learned baselines are matched to the single design's
**TOTAL bytes (1×)**; the old 524,288 B matching is printed as a **reference column**.
⭐ **The four-part ledger — atoms · covariance/count ledger · address book · coupling/hole table — is
printed on EVERY row and ⛔ NEVER netted.**

## 1. VAL SWEEP (free, no scored run, start immediately)
Label-free correlation partition (**ward, balanced**) at **K ∈ {5, 8, 10, 12, 16, 20, 24, 28, 32}**,
**rank 64 per well, capped STRICTLY below well width**. ⭐ **Print rank/width on every row.**
Per K report: val composition (the existing 24-corner val read) · four-part ledger bytes ·
exactness (downdate == refit-without) at 3 seeds.

⛔⛔ **DECLARE THE SELECTION RULE BEFORE THE SWEEP RUNS, IN A FILE, DATED:**
> **K\* = the finest partition whose val composition is within 0.02 of K=5.**

⭐ Also report **the K where composition first drops > 0.10 — the crossover.** That is a paper figure:
composition and ledger bytes vs K, two y-axes.

## 2. ⛔⛔ THE CONTROL THAT WAVE 9 PROVED YOU CANNOT OMIT — read this twice
⚠ **Wave 9's maze arm LOOKED like it worked and was decorative.** At `rank/d_k = 0.938` any full-rank
atom block spans the same subspace, so the content carried nothing and the address book did all the
work. ⛔ **It was caught ONLY because `landscape_shuffled` and `random_atoms` were run.**
⭐⭐ **THEREFORE, ON EVERY SCORED ROW, RUN AND PRINT:**
- **`landscape_shuffled`** — atoms shuffled across wells. **MUST collapse.**
- **`random_atoms`** — atoms replaced by random directions of the same rank. **MUST collapse.**
- **`landscape_removed`** — atoms zeroed. **MUST collapse to floor.**
⛔ **If shuffled or random does NOT collapse, the store is DECORATIVE at that K and NO ROW MAY BE
CLAIMED for it** — report that as the result. ⭐ This is the saturation trap, and it is why you print
rank/width everywhere.

## 3. SCORED at K\* (3 seeds, 5 if caches exist)
Rows 3 and 4 vs **MLP · LSTM · transformer · continuous MLP** at **1×**, with 524,288 B printed as
reference; **table floor** printed; **composition-only store** ("composes, cannot forget") and the
**wave-4 item store** ("forgets, loses composition") printed as **ABLATIONS**.
⭐ **`generation-table` run 3** on the same object vs the full rival set.

## 4. DELETION — RESULT rows, same seeds
- **(a) VALUE:** delete every row carrying **red** ⇒ green/blue composition on rows 3–4, vs baselines
  **retrained-without-red** (cost printed) **and** vs baselines **as-trained**.
- **(b) CONJUNCTION:** delete every row carrying **red ∧ cube** ⇒ green/blue cubes and red
  spheres/cylinders on rows 3–4, same comparators.
- **(c) ITEM:** delete a random **1 %** of fit rows ⇒ composition unchanged vs refit-without
  (exactness printed).
⭐ **EVERY row prints adaptation cost for EVERY arm: seconds · bytes rewritten · data rows read**
(**0 for us**).

## 5. DELETION — DISPLAY rows (⛔ reported, never claimed)
Post-deletion **refusal**: red requests after (a), red-cube requests after (b) — our refusal via the
**hole** vs every baseline's **render rate**. Comparator = the refuse-token transformer from
`spoke/deletion-baselines`, ⛔ the only permitted one.

## ⭐ TWO EXACT MECHANISMS, ON ONE OBJECT, STATED AS SUCH
(i) **data forgetting = covariance/count DOWNDATE**; (ii) **refusal after deletion = the request
count/coupling table with HOLES.**
⭐⭐ **A compositional store can still compose "red cube" from "red" and "cube" after the red-cube rows
are gone. THAT IS COMPOSITION WORKING, NOT DELETION FAILING** — the refusal row is what the hole is
for. ⛔ Report both, separately, on the same object.

## ⛔ LANGUAGE — the Head's ruling, and a referee's
- ⛔ **NEVER "no changes after deletion."** Say **post-deletion adaptation is LIGHT and DATA-FREE**
  (re-eigendecompose the ledger, carry the address book, **no row of data read**) versus baselines
  that can only delete by **retraining from scratch without the deleted rows.**
- ⛔ **DEFINE "EXACT."** The referee rejected the draft partly because it never was. The measurements
  support a **STORE-SCOPED** definition — ⭐ the frozen encoder and decoder are **not refit** — so say
  that, in those words, wherever the word appears.
- ⛔ **Never "9/9 seeds."** Report the actual seed count and the actual configuration count separately.

## ⛔ WHAT WOULD PROVE THIS HUB WRONG
- **No K satisfies the declared rule** ⇒ report that; ⛔ do not move the rule. Charter A.3.
- **The controls in §2 do not collapse at K\*** ⇒ ⭐ the store is decorative there and **no row is
  claimed** — that is the wave's result.
- **Composition at K\* is not within 0.02 of K=5 on TEST** though it was on val ⇒ ⭐ report the val/test
  gap as the finding.
- ⭐ Budget: `single-design` 3 runs, `generation-table` 1. ⛔ **Write the ledger row in
  `/c/chlu/SCORED-LEDGER.md` BEFORE each scored run** — your OWN row; another hub's does not authorise you.
