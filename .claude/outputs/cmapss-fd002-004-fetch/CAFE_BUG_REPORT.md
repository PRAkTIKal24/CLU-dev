# CAFE bug report — C-MAPSS (forwardable verbatim to the CAFE team)

Found while registering an external encoder against `cafe-bench` on the Event
Prediction track. Verified against `cafe_bench` @ `dc3dbd0` and the canonical
NASA C-MAPSS files. Reported in priority order. Happy to send patches.

---

## BUG 1 (critical, correctness) — C-MAPSS **test** labels are cycles-remaining-in-recording, not RUL; the `RUL_FD00x.txt` files are never read

**Where:** `cafe_bench/datasets/event/cmapss.py`, `_load_split`, lines 67–72:

```python
for i in range(self._window, max_cycle + 1):
    window = sensors[i - self._window:i]
    rul    = max_cycle - i     # <-- remaining cycles IN THE RECORDING
    X_list.append(window)
    t_list.append(rul)
    e_list.append(1)           # <-- all windows marked "event observed"
```

`_load_split` is called identically for train and test (lines 82–83), and the
`RUL_FD00x.txt` files shipped with the dataset are never opened.

**Why it is wrong.** C-MAPSS *train* units are run to failure, so for train
`max_cycle − i` **is** the true RUL and the code is correct. But the *test*
units are deliberately **truncated some time before failure**, and the residual
RUL at the last recorded cycle is given in `RUL_FD00x.txt`. So for test:

```
t_true(window ending at cycle i) = RUL_unit + (max_cycle − i)
```

⇒ **every test label is under-estimated by exactly `RUL_unit`** — a constant
offset per unit, i.e. a systematic per-unit shift, not noise. And `e=1` asserts
the failure was observed within the recording, which for test units is false
(they are right-censored).

**Concrete example** (FD001 test unit 1): 31 recorded cycles, `RUL_FD001.txt`
line 1 = **112**. CAFE labels its final window `t = 0` — "this engine is failing
right now" — when it in fact has 112 cycles of life left.

**Measured damage** (window=30, the 125 horizons CAFE evaluates):

| set | test windows | mean label error (cycles) | max | windows mislabelled | windows whose binary label flips for ≥1 horizon | P(event by h=125): CAFE vs true |
|---|---|---|---|---|---|---|
| FD001 | 10196 | **+62.5** | 145 | **100%** | 89.1% | 0.891 vs **0.497** |
| FD002 | 26505 | **+58.9** | 194 | **100%** | 85.0% | 0.850 vs **0.498** |
| FD003 | 13696 | **+62.7** | 145 | **100%** | 72.0% | 0.720 vs **0.376** |
| FD004 | 34081 | **+71.7** | 195 | **100%** | 69.9% | 0.699 vs **0.343** |

(Train-split label error is exactly **0.00** in all four sets, confirming the
bug is test-split-only.)

The last column is the practical headline: at the longest evaluated horizon
CAFE believes **~89%** of FD001 test windows have failed, when the truth is
**~50%**. The evaluation is scoring a substantially different classification
problem from the one it reports.

**Consequence.** Numbers remain *internally* comparable (all models see the same
wrong labels), but they are **not comparable to any externally-published C-MAPSS
result**, including the numbers CAFE's own leaderboard compares against.

**Suggested fix.** Read `RUL_FD00x.txt` for the test split:

```python
def _load_split(self, path, rul_path=None):
    unit_rul = np.loadtxt(rul_path) if rul_path else None
    ...
    residual = 0.0 if unit_rul is None else float(unit_rul[unit_idx])
    t_list.append(max_cycle - i + residual)
    e_list.append(1 if unit_rul is None else 0)   # test units are censored
```

Whether test rows should be `e=0` (censored at the recording end) or `e=1` with
the exact known failure time is a benchmark-design choice — but the current
combination (`t` truncated **and** `e=1`) is the one option that is
unambiguously inconsistent.

---

## BUG 2 (metadata) — `FD004` train unit count is off by one

`cafe_bench/datasets/event/cmapss.py`, `_CMAPSS_INFO`:

```python
"FD004": (248,  248,  6, 2, "6 op conditions, 2 fault modes"),
#         ^^^ n_train
```

Canonical FD004 has **249** train units and 248 test units. Verified:

| set | train units | test units | train rows | test rows | `RUL` lines |
|---|---|---|---|---|---|
| FD001 | 100 | 100 | 20631 | 13096 | 100 |
| FD002 | 260 | 259 | 53759 | 33991 | 259 |
| FD003 | 100 | 100 | 24720 | 16596 | 100 |
| FD004 | **249** | 248 | 61249 | 41214 | 248 |

FD001–FD003 entries are correct. Cosmetic (`DatasetInfo` is metadata only, and
the loader counts units itself), but it is user-facing.

---

## BUG 3 (blocking for new users) — the C-MAPSS download path is dead (404)

`scripts/download_all.py`'s C-MAPSS route fails: both

- `https://ti.arc.nasa.gov/c/6/` → **404**
- `https://data.nasa.gov/download/ff5v-kuh6/...` → **404**

NASA retired both endpoints. Network to NASA/HF/GitHub is otherwise fine from
the same machine, so this is not a local issue. A fresh `cafe-bench` checkout
therefore cannot obtain C-MAPSS at all.

**Workaround we used:** the HuggingFace mirror
(`LucasThil/nasa_turbofan_degradation_FD001`) converted to the expected txt
layout — cross-checked as numerically identical to a canonical FD001 from an
independent source. Worth pinning a mirror in `download_all.py`.

---

## Note (not a bug, but a documentation gap worth a line in the README)

CAFE's default Event probe is **CoxPH on the frozen embedding**. A
proportional-hazards risk score `β·z` induces the **same sample ranking at every
horizon** — we measured rank correlation **1.0000** and a bit-identical
permutation between h=1 and h=125. So per-horizon AUROC varies *only* because
the labels change, and **h-AUROC cannot discriminate models on
horizon-specific behaviour** under the default probe. Anyone reading the metric
as evidence of "graceful long-horizon degradation" will be misled; that claim
needs a horizon/Δt-conditioned head. Worth stating explicitly next to the
metric definition.
