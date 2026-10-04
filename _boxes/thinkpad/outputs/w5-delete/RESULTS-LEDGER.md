# w5 hub/delete — RESULTS LEDGER

Generated 2026-09-20T02:29:16Z from the artifacts, not typed. Target `deletion`, budget **3 scored runs**, banked since wave 2.

| run | PREREG | what it was | status | outcome |
|---|---|---|---|---|
| **1/3** | `PREREG-w5-delete-1.md` | Row 7 on the field store: delete `object_hue=0.0` by lifetime/eviction, and delete `red ∧ cylinder` by the 1-byte coupling hole. BOTH halves — the refusal and the surviving composition — on 109 held-out corners (20 deleted / 89 survivor), seeds 0/1/2, cache generation 2. | COMPLETE | see below |
| 2/3 | — | **NOT SPENT** | banked | — |
| 3/3 | — | **NOT SPENT** | banked | — |

⛔ A declared NOT-RUN is never a null. Runs 2 and 3 are returned unspent.

## Per-bar outcome of run 1/3

| bar | what it required (declared in advance) | result |
|---|---|---|
| **BAR 5** — byte property | 0 bytes changed OUTSIDE the permitted ranges, exact, no tolerance; `assert_no_refit` passes; the hole is 1 byte | **PASS on seeds 0, 1, 2.** evict 17,569 / 16,764 / 19,114 B changed, **outside 0 / 0 / 0**; hole **1 B / outside 0** on every seed; `ALL_BYTE_PROPERTIES_HOLD` = **True**; negative control fires (5,735 / 5,726 / 5,783 B) |
| **BAR 4(b)** — the trap test | every conjunction the pool never held, other than the hole, keeps `hole` flag 0.0000 and refusal drift ≤ 0.01 | **PASS on seeds 0, 1, 2.** 102 such conjunctions; the hole flags 1.0000; of the other 101, **0 / 0 / 0 light up** and the worst refusal drift is 0.0000 |
| **The dash, earned** | a row a rival cannot enter gets a dash and a footnote, never a win for us | **5 learner families each refuse 0.0000** of the 720 deleted-conjunction requests and answer them anyway; `lace` and `poe_ebm` are not callable at all. Population-caveated and descriptive, not a contest. |
| **§C.4** — nearest atom | an arm whose settled state sits on its nearest atom is a lookup table | **NOT a lookup table**: relative distance 0.3840 / 0.4447 / 0.4583 (min 0.2225 / 0.1817 / 0.2673) |
| **BAR 1** — eviction refuses | refusal ≥ 0.99 on the deleted set and on red's other shapes | **PASS** per seed: True / True / True |
| **BAR 2** — composition survives | abs(delta) <= 1.0 novel tuple/corner on the 89 survivor corners | **MIXED/FAIL** per seed: False / True / False |
| **BAR 3** — the hole refuses | hole flag = 1.0000 and refusal ≥ 0.99 on the deleted conjunction | **PASS** per seed: True / True / True |
| **BAR 4** — the hole is surgical | hole flag 0.0000 on every survivor probe, drift ≤ 0.01, paired Δ exactly 0 | **PASS** per seed: True / True / True |

## Replication (charter §A.5)

**REPLICATED = True** — clean checkout of `w5/delete`, its own process tree, output root and partitions. EXACT tolerance on byte properties and refusal rates; 1e-6 on the two judge-mediated composition deltas.

| quantity | class | run | replication | max gap | agrees |
|---|---|---|---|---|---|
| `evict_composition_mean_delta` | TIGHT | -2.0899 / -0.9101 / -6.7416 | -2.0899 / -0.9101 / -6.7416 | 0.00e+00 | **True** |
| `evict_refuse_on_deleted_set` | EXACT | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 0.00e+00 | **True** |
| `field_hole_refuse_on_survivor_set` | EXACT | 0.0054 / 0.0000 / 0.0000 | 0.0054 / 0.0000 / 0.0000 | 0.00e+00 | **True** |
| `gate8_conjunction_refuse_on_deleted_set` | EXACT | 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 | 0.00e+00 | **True** |
| `hole_composition_mean_delta` | TIGHT | 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 | 0.00e+00 | **True** |
| `hole_flag_on_deleted_set` | EXACT | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 0.00e+00 | **True** |
| `joint_backoff_refuse_on_deleted_set` | EXACT | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 0.00e+00 | **True** |
| `joint_backoff_refuse_on_survivor_set` | EXACT | 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 | 0.00e+00 | **True** |
| `joint_presence_gate_refuse_on_survivor_set` | EXACT | 0.2899 / 0.2899 / 0.2899 | 0.2899 / 0.2899 / 0.2899 | 0.00e+00 | **True** |

## Cost line

- table **retrain from scratch (GOLD STANDARD)**: **3.79 s**, **120,028 rows refitted**, 11,044 dropped.
- table **row removal**: 0.0000 s, **0 rows refitted**.
- **field store, evict**: an in-place zero of 6–7 items. **0 rows refitted, 0 bytes outside the permitted ranges.**
- **field store, the hole**: **1 byte. 0 rows refitted.**
- **`mlp` retrained without the value**: **88.2 s**, 120,028 rows — and afterwards it still answers the deleted requests at refusal **0.0000**.