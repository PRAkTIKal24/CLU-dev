# w5 hub/refuse — results

Condition **test** · seeds queries [0, 1, 2], deletion [0, 1, 2] · cache **generation 2** · 109 scored corners (60 row-3 + 49 row-4).

⭐ **Labels:** used ONCE at fit time to align wells and build the address book, and at the interface. ⛔ Never in the landscape, the read or the write.

⛔ **The bar is the 8-byte rule, not the table.** Every number below has `bitmap8` (8 B) printed beside it.

## Rows 5 and 6 — the pair (reject_invalid, accept_unseen_valid), mean ± seed spread

| row | refuser | bytes | reject_invalid | accept_unseen_valid | gmean |
|---|---|---|---|---|---|
| `r5_comp_refusal` | `field_geom` | 524,288 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| `r5_comp_refusal` | `field_geom_gap` | 524,288 | 1.0000 ± 0.0000 | 0.9948 ± 0.0020 | 0.9974 ± 0.0010 |
| `r5_comp_refusal` | `field_landscape_removed` | 524,288 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| `r5_comp_refusal` | `field_landscape_shuffled` | 524,288 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| `r5_comp_refusal` | `bitmap8` | 8 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| `r5_comp_refusal` | `pair_presence` | 166 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| `r5_comp_refusal` | `pair_deletion` | 174 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| `r5_comp_refusal` | `refuse_everything` | 0 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| `r5_comp_refusal` | `accept_everything` | 0 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| `r6_invalid_refusal` | `field_geom` | 524,288 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| `r6_invalid_refusal` | `field_geom_gap` | 524,288 | 1.0000 ± 0.0000 | 0.9948 ± 0.0020 | 0.9974 ± 0.0010 |
| `r6_invalid_refusal` | `field_landscape_removed` | 524,288 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| `r6_invalid_refusal` | `field_landscape_shuffled` | 524,288 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| `r6_invalid_refusal` | `bitmap8` | 8 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| `r6_invalid_refusal` | `pair_presence` | 166 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| `r6_invalid_refusal` | `pair_deletion` | 174 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| `r6_invalid_refusal` | `refuse_everything` | 0 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| `r6_invalid_refusal` | `accept_everything` | 0 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 |

## Refuse-after-deletion

### A. CONJUNCTION deletion (`punch_hole`) — ⭐ THE row

| refuser | bytes | reject_deleted | Δreject | accept unseen-conj | accept row1 | survivors: distinct valid novel tuples |
|---|---|---|---|---|---|---|
| `field_geom` | 524,288 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 2317.3333 ± 312.0000 |
| `field_geom_gap` | 524,288 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.9948 ± 0.0020 | 2312.6667 ± 305.0000 |
| `bitmap8` | 8 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 2317.3333 ± 312.0000 |
| `pair_presence` | 166 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1458.0000 ± 222.0000 |
| `pair_deletion` | 174 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 2317.3333 ± 312.0000 |

### B. VALUE deletion (`evict_term`)

| refuser | bytes | reject_deleted | Δreject | accept survivors | accept row1 | survivors: distinct valid novel tuples |
|---|---|---|---|---|---|---|
| `field_geom` | 524,288 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.9453 ± 0.0000 | 1964.6667 ± 362.0000 |
| `field_geom_gap` | 524,288 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.9916 ± 0.0068 | 0.9401 ± 0.0020 | 1961.6667 ± 359.0000 |
| `bitmap8` | 8 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.9453 ± 0.0000 | 1964.6667 ± 362.0000 |
| `pair_presence` | 166 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.7101 ± 0.0000 | 0.9453 ± 0.0000 | 1433.3333 ± 322.0000 |
| `pair_deletion` | 174 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.9453 ± 0.0000 | 1964.6667 ± 362.0000 |

### The joint table's cost, per family — ⚠ the target file's framing, checked

The target file says the joint table that can express a conjunction deletion "refuses every row-3 and row-4 request". On **this** scored split that is true of row 3 and ⛔ **false of row 4**: the 49 n-way corners name `(object_hue, shape)` conjunctions the pool HOLDS, so a pair-presence table answers them. Measured, after the hole:

| refuser | bytes | accept row-3 (60 corners) | accept row-4 (49 corners) | distinct valid novel tuples |
|---|---|---|---|---|
| `field_geom` | 524,288 | 1.0000 ± 0.0000 | 0.9082 ± 0.0000 | 2317.3333 ± 312.0000 |
| `field_geom_gap` | 524,288 | 1.0000 ± 0.0000 | 0.8964 ± 0.0096 | 2312.6667 ± 305.0000 |
| `bitmap8` | 8 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 2317.3333 ± 312.0000 |
| `pair_presence` | 166 | 0.0000 ± 0.0000 | 0.9082 ± 0.0000 | 1458.0000 ± 222.0000 |
| `pair_deletion` | 174 | 1.0000 ± 0.0000 | 0.9082 ± 0.0000 | 2317.3333 ± 312.0000 |

## Verdicts

- **r5_comp_refusal** — ⛔ THE 8-BYTE RULE MATCHES MY PAIR ON THIS ROW ⇒ NO CLAIM. The dial declaration is falsified here: what was measured is the factor grid, not the store.
- **r6_invalid_refusal** — ⛔ THE 8-BYTE RULE MATCHES MY PAIR ON THIS ROW ⇒ NO CLAIM. The dial declaration is falsified here: what was measured is the factor grid, not the store.
- **r7_hole** — ⭐ The contrast is EXACT on one side and MEASURED on the other: the per-factor gate's delta_reject is 0.0000 because ZERO of its bits change under a conjunction deletion, and the joint pair-PRESENCE table that can express the deletion refuses the never-held conjunctions. ⚠ The declared residual -- a SEPARATE pair-deletion table -- does follow, and is priced on the table.

⚠ **Population footnote (binding):** Every rival in RIVALS-READY.json is fitted on the montero_combination pool, which CONTAINS both held-out values (3,157 rows at object_hue 0.5; 7,972 at shape 0.0). On rows 5 and 7 a rival saw the value the field store never did. ⛔ Mechanism can enter the row; the fit cannot. Not counted in either direction.