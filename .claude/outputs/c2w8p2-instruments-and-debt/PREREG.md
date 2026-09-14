# PREREG — `c2w8p2-instruments-and-debt` (wt3, riders 2 + 3a)

**This file is a copy of the registration block filed as `ERRATA-C2W8-PASS2.md` §2 (the canonical
site), reproduced here because protocol §5's pre-registration rule wants a `PREREG.md` in the output
directory.** Both were written **before** `chlu/core/soft_certificate.py` gained a single line of the
K9 predicate and **before** any refusal rate was measured. Commit order proves it: the ERRATA block
was appended before commit `c13f953`.

## Scope of what is pre-registered

⛔ **No dial claim, no performance number, no cell.** The only quantities this spoke measures are
(i) the re-registered criterion's verdicts on **designed** pairs and on the **banked** census pairs,
and (ii) test counts. There is no ratio/exponent/slope/law in the acceptance criterion, so the rule
applies here only in its weak form — but the three predictions below were still committed to first.

## The registered operating point (before the predicate existed)

* geometry leg: `center_sep <= rho_geom * key_spacing`, **`rho_geom = 1.0`**, `key_spacing` = the
  **measured** per-seed `geometry.median_nn_task1`. INAPPLICABLE ⇒ REFUSE if the spacing is
  missing/non-finite/non-positive.
* payload leg: `payload_dist <= tau_payload * payload_scale`, **`tau_payload = 0.25`**,
  `payload_scale` = median pairwise `||a_i − a_j||` over the **whole** live-well pair population.
  **`payload_scale <= 1e-9` ⇒ INAPPLICABLE ⇒ REFUSE** (the anti-vacuity clause).

## Predictions (falsifiable, committed before measurement)

| # | prediction | basis |
|---|---|---|
| **R1** | the frozen census's **28 / 29 / 29** pass-1-admitted pairs go to **0 admitted**, refusal rate **1.000**, all refused on the **geometry** leg | banked minima `center_sep / key_spacing` = **2.0599 / 1.4942 / 1.1589**, all > `rho_geom = 1.0` |
| **R2** | `vacuous_gate` **still trips on the frozen census, at the OPPOSITE end** (f = 1.000, not pass 1's 0.000) — registered **as expected, not as a failure**: pass 1 measured `capture_radius` 0.000 on 47/48 wells, so there is nothing there to merge | pass-1 finding (a) |
| **R3** | the smallest `rho_geom` that would admit **any** banked pair is **2.0599 / 1.4942 / 1.1589** per seed, so any future loosening is visibly a decision | same minima |

## What would falsify the deliverable

The criterion cannot be made to refuse on **either** leg (i.e. no designed pair is declined) ⇒ that
outcome is the finding and is reported as such.

*Filed 2026-08-06 by `c2w8p2-instruments-and-debt`.*
