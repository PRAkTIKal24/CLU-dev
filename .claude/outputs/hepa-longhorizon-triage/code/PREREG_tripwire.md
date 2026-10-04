# PREREG — hepa-lh-tripwire (spoke-level, subordinate to the Hub's frozen PREREG.md)

**Written 2026-09-07 by the experiment-engineer spoke, AFTER the harness was coded and the loader
smoke-tested, and BEFORE any arm was scored.** No h-AUROC of any kind had been computed at the time
of writing: the only numbers observed so far are shapes and window counts (FD001 @512: 60/10/30
entities, 1167/205/2354 windows, mean effective context 178, all 150 horizons non-degenerate).

The Hub's `../PREREG.md` is frozen and governs the dataset-level KEEP/DROP predictions. It does not
predict the tripwire's own numbers, which is what this file commits to. Protocol §5 requires it
because §8 of my task states a falsification condition in terms of a measured quantity.

## Reference values I am predicting against (not measured by me)
- HEPA FD001 @512, paper Table 1: **0.81 ± 0.03** (5 seeds).
- HEPA FD001 @512, `REPRODUCIBILITY.md` clone-actual: **0.918 ± 0.008**.
- These differ by the finetune operating point (early-stopped vs fixed-epoch), scout §2.

## Predictions (point + interval), FD001, context 512, seeds {42,123,456}
| Arm | Predicted h-AUROC | Derivation |
|---|---|---|
| `prevalence` | **0.500 exactly** | scores are constant within a horizon, so every non-degenerate per-horizon `roc_auc_score` is exactly 0.5 by construction. Mechanical, not empirical. A value != 0.500 means the harness is mis-wired. |
| `seasonal` | **NOT_APPLICABLE** | C-MAPSS native rate is 1/flight-cycle on a run-to-failure stream; no calendar or diurnal period exists. Declared in `tw_config.SEASONAL_PERIOD`. |
| `last_event_age` | **0.80** (interval 0.72–0.88) | On C-MAPSS this reduces to the cycle index + 1, because each engine has exactly one event (its last cycle) and no event ever precedes a cut point. Age predicts RUL only through the lifetime distribution: FD001 train engines run 128–362 cycles (median 195), so a fixed age maps to very different RULs across engines. Informative, far from perfect. |
| `knn` exemplar store, best descriptor, `residual_stream` budget (98 304 B) | **0.85** (interval 0.78–0.92) | Nearest-neighbour on absolute sensor level is the classical C-MAPSS similarity-based RUL method and C-MAPSS is loaded with RevIN OFF precisely so the absolute degradation level survives. I expect it to beat age but not to reach a trained encoder's fixed-epoch operating point. |
| `knn` seed spread (max − min over 3 seeds) | **0.02** (interval 0.005–0.06) | The store subsamples: 431 of 1167 train windows at `chan_stats`/`residual_stream`, i.e. 37 % kept, so seeds genuinely differ. |

## The falsification named in task §8
> "You falsify [the idea that HEPA's encoder does something a same-byte lookup cannot] by building a
> tripwire that matches or beats HEPA at 512 on FD001."

**I predict the tripwire does NOT beat clone-actual 0.918, and DOES beat the paper's 0.81.**
That combination would mean the *published* FD001 margin (+0.01 over PatchTST) is not evidence of
representation learning, while the *repo's* operating point still is. I pre-register that split
explicitly because it is the awkward outcome and I do not want to be able to claim either half after
the fact.

- If **max(tripwire) >= 0.918**: HEPA's own best reported FD001 number is matched by a same-byte
  lookup. Reported in the first line of my report, per task §8.
- If **max(tripwire) < 0.81**: the reference is weak; I must say so and not present it as a
  calibrated kill instrument for wave 2 without strengthening it.
- If `last_event_age` alone >= 0.81: a baseline with **no model and no window** matches the published
  number. This is the referee-magnet outcome and gets the same first-line treatment.

## Ledger prediction (arithmetic, committed before it is reviewed)
At L=512, P=16, d_model=256, n_layers=2, fp32: n_tokens = 32;
`token_only` = 32·256·4 = **32 768 B**; `residual_stream` = 3·32·256·4 = **98 304 B**;
`kv_cache` = 5·32·256·4 + 1 024 = **164 864 B**. Primary = `residual_stream`.
