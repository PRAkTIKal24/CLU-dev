# PREREG — hub/factor-wells — scored run 2 (the hub's second; `composition` 3/3)

**Written 2026-09-19, BEFORE any held-out number for `ward_balanced`.**
**Budget:** the `composition` target. 1 spent in wave 2, 1 spent by this hub (run 2/3). **This is 3/3 —
the last one for this target.**
**Charter §A.3:** this PREREG does not change any bar in `PREREG-hub-factor-wells-1.md`. Run 1's numbers
stand exactly as reported; this run answers a *different* question and its own bar is set here, first.

---

## 1. The question, and why it is worth the last run

Run 1 measured `factor_aligned_balanced` beating `basis_head` (the shipped **ward** cut) on row 4 by
**+11.612** novel tuples per corner over 49 corners, CI95 [+7.612, +15.408], p = 9.26e-06.

Those two partitions differ in **two** ways at once:
1. one is aligned to the factors, the other to channel correlation; and
2. mine has **equal** wells [410, 410, 410, 409, 409], ward's are **unequal** [332, 364, 421, 411, 520].

⛔ A referee reads that and asks whether the win is factor-alignment or merely equal well sizes. On
`val` (24 corners) I measured the separating arm and **could not resolve it**: `ward_balanced` −
`basis_head` = +1.333 (CI95 [−0.958, +3.583], p = 0.286, a tie) and `factor_aligned_balanced` −
`ward_balanced` = +3.958 (CI95 [+0.250, +7.667]) but **14 W / 0 T / 10 L, sign-test p = 0.541**. 24
corners lack the power.

**This run decides it on the 49 + 60 corners, where run 1 had the power to find effects val missed.**

## 2. The arm, and what makes it the right separator

`ward_balanced` — the **ward linkage's dendrogram leaf order**, cut into 5 equal contiguous blocks. This
is the variant the target file itself names: *"a balanced variant cutting the dendrogram leaf order gives
equal wells at any K. Both keep the correlation structure and neither sees a label."*

| | `factor_aligned_balanced` | `ward_balanced` |
|---|---|---|
| grouped by | `I(bin(channel); factor)` on the fit pool | ward linkage on `\|corr\|`, leaf order |
| sees a label | at the address book only | **never** |
| well sizes | 410 · 410 · 410 · 409 · 409 | **410 · 410 · 410 · 409 · 409** |
| K / M / D / store bytes | 5 / 64 / 2048 / 524,288 | identical |
| bits per read | 30.0 | 30.0 |
| reachable read rank | 320 | 320 |
| read / head / ridge / fit pool / decoder / judge | — | **identical** |

Every knob in §5 of PREREG 1 is unchanged and carries over verbatim. **The equal-well histogram is now
held fixed too**, so the only thing that moves between the two arms is *which* equal-sized wells the
channels fall into — factor-aligned or correlation-aligned.

## 3. ⭐ THE MARGIN, DECLARED IN ADVANCE

**Estimand:** the mean paired per-corner difference in `distinct_valid_novel_tuples`,
`factor_aligned_balanced` minus `ward_balanced`, over the **same 49 n-way corners (row 4)** and **60
pair-tiled sub-corners (row 3)**, in the same fixed order, with the same 10,000-sample paired bootstrap
and the same two-sided sign test.

**Margin: `+1.0` distinct valid novel tuple per corner** — identical to run 1's, so the two results are
read on one scale.

**Factor-alignment is shown to carry the win only if BOTH hold on row 4:**
1. `mean_paired_delta > 1.0`, **and**
2. the paired-bootstrap 95 % lower bound is also `> 1.0`.

## 4. ⛔ What each outcome means, written down before the numbers exist

| outcome on row 4 | what I will report |
|---|---|
| clears the margin | the win survives holding the well-size histogram fixed ⇒ **it is factor-alignment**, and run 1's +11.612 over ward decomposes into a factor-alignment part and a balancing part, both reported |
| **ties** (CI contains the margin or zero) | ⛔ **the partition being factor-aligned is decoration relative to being balanced, and I report it in those words.** Run 1's headline then survives only as "equal wells beat the shipped unequal ward cut", the factor-alignment claim is **withdrawn**, and this hub's target is **not met** |
| `ward_balanced` **wins** | the same withdrawal, stated more sharply, and I say the relevance grouping was the worse of the two label-free partitions |

⛔ I do not get to choose afterwards which of these I am in. The third column of run 1's table already
shows `ward_balanced` **above** `basis_head` on val, so a tie here is a live outcome and I have written
its sentence in advance.

## 5. Also reported from this run, claimed as nothing

`ward_balanced` is scored on **all seven rows** with both controls, exactly like every other arm, and its
row goes into the same `scoreboard.json`. Its own laundering floor is `random_dmatched_balanced`, which
already carries the identical equal-well histogram and is already on the table.

## 6. Nothing else moves

⛔ No arm from run 1 is refitted, re-run or re-scored. ⛔ No margin from PREREG 1 is touched. ⛔ The store
fit, the head, the ridge, the pool, the corners, the decoder and the judge are the same objects, and the
run is executed by the same committed code (`de1d692`) with one additional label vector.
