# WAVE 3 — REPORT FOR THE MAIN ADVISOR
**Compiled by the ICLR-long Hub, 2026-09-19. Four autonomous hubs, FORGIS_14, seed 0, `base/iter3 = afd37cf`
(= pin `4125fce` + 4 additive files, 3,566 insertions, 0 deletions, no pinned file edited).**
Raw record: `.claude/outputs/_thinkpad/w3/hub-<target>/LOG.md` + PREREGs + `run/*.json`.
⚠ Every number below is quoted from a hub's own report. ⛔ **The Hub has re-derived none of the statistics.**

---

## 0. The scoreboard

| hub | verdict | scored |
|---|---|---|
| ⭐ `factor-wells` | **WIN on row 4 (compositional n-way)**; row 3 ties and is decoration | **2/3 + 3/3** |
| ⛔ `store-content` | **FALSIFIED** — decisively beaten, a clean negative | 2/3 |
| ⛔ `unseen-valid` | **row 2 is NOT SCORABLE with the frozen judge** — stopped before spending | **0/3** |
| ⚠ `erasure` | gate-vs-erasure established; CLU-vs-table **not** | **0/3** |

**Budget after wave 3:** composition **3/3 exhausted** (⚠ see §5.1) · generation-table **2/3, one left** ·
deletion **0/3 intact** · unseen-valid **0/3 intact** · admission lapsed.

---

## 1. ⭐ `factor-wells` — the first outright win the programme has produced

Partition the store's channels by the factor they carry (`R[d,f] = I(bin(channel_d); factor_f)`, argmax to
factor, merge to the pinned K = 5). Labels touch the address book only.

**Row 4, compositional n-way, vs the fitted basis + linear head at matched bytes:**
**Δ = +11.612 novel tuples/corner · 49 paired held-out corners · CI95 [+7.612, +15.408] · 40 W / 0 T / 9 L ·
sign-test p = 9.26e-06**, against a **+1.0 margin declared in advance**. Absolute: **41.551** vs basis+head
**29.939** vs `knn_proto` **15.408**. Against `knn_proto`: **+26.143**, p = 1.01e-07.

**Row 3, compositional 2-way: Δ = +1.200, CI95 [−0.850, +3.150], p = 0.322 — a TIE.** The hub writes the
required sentence against itself: on that row the mechanism is decoration.

**The confound leg (scored run 3/3) is what makes the win interpretable.** `ward_balanced` holds the equal
well-size histogram exactly and sees no label:
- mine − `ward_balanced`, row 4: **+25.571** [+21.041, +30.143], 47/0/2, **p = 4.36e-12**
- `ward_balanced` − `basis_head`, row 4: **−13.959**, 5/0/44, p = 7.60e-09
⇒ **Equal wells do not explain the win — for a correlation partition they cost 14 tuples/corner. The effect
is factor-alignment, and the arm wins while carrying a handicap.**

**Floors:** all §C pass; `landscape_removed` and `landscape_shuffled` both collapse to **0.000** on rows 3–4.
Replicated exactly on a clean checkout. Cost 2 h 36 m, local CPU, no GPU.

**Limits the hub states itself:** seed 0 · row 3 ties, so half the target row set is decoration and it does
not know why · its `val` proxy **misled twice** (called the balance contrast a tie at p = 0.68 where the
scored corners give p = 3.6e-07) though no val number changed a choice · the head is **3.37× the store** ·
⭐ **the encoder limits the idea**: only **12 of 1984** live channels are top-ranked by `shape` and **4** by
`scale`, three of five wells go to hue · one declared, non-bit-identical fitter deviation (traced `τ`),
applied identically to every arm.

⚠ **Row 7 finding the Advisor should note:** this architecture **gates a value perfectly and cannot erase
one** — deletion here is an operation on the interface, not on atoms, so ⛔ **it does not satisfy charter
§B.4.** Structural, not a tuning failure: under a shared-basis read no atom belongs to one value.

## 2. ⛔ `store-content` — falsified, and it retires a line of attack

Earlier-layer content at matched bytes does not recover `scale` or `orientation`; **it makes them worse.**
All four pre-registered bars fail, three significantly in the wrong direction, on 49 paired corners.
Against the like-for-like: **aggregate −0.3039 [−0.3313, −0.2764], 0 corners won of 49, p = 1.1e-09.**
⛔ Not a tie — decisively beaten, so the decoration clause does not apply.

**It is a real negative, not a broken arm:** the content is live (beats its row-shuffled control by +0.0715,
p = 1.3e-05), the store is genuinely joint (mixing 0.935, every well holds both blocks), all §C floors pass,
the earlier-layer cache re-encodes bit-exactly from pixels, and it replicated at max abs diff **0.0**.

⭐⭐ **The positive finding inside the negative, and it is the most important line in this report.** From the
**same** penultimate content, at or below the same budget:

| reader of identical content | `scale` | `orientation` |
|---|---|---|
| `knn_proto` (per-value means) | 0.2188 | **0.9320** |
| `basis_head` | 0.2066 | 0.5142 |
| `P` — the addressed read | 0.1865 | **0.1828** |

⇒ **The loss of `scale` and `orientation` is a property of the ADDRESSING AND THE READ, not of the layer.**
A table of per-value means recovers orientation at 0.9320 where the addressed store gets 0.1828.

⚠ **A structural constraint nobody had stated:** at a 2^17-byte budget with a fixed 2048-dim render
interface, `N·D = 2^17` forces content to be paid for in atoms, and the frozen decoder's (128,4,4) input
forces the whole penultimate slice to survive. **Anyone reopening the content dial must first move the
render interface.**

## 3. ⛔ `unseen-valid` — the instrument cannot measure the row, and it proved that before spending

The external judge is a **10-way categorical `object_hue` head**; a row-2 request asks for 0.05.
The hub built each request's exact ground-truth in-between image from its two real bracketing rows — the
score a **perfect** arm would get, with no arm in the room:

| readout | perfect arm | perfect lookup | usable span |
|---|---|---|---|
| ⛔ **argmax — the pin's OWN metric** | 0.5000 Δ | 0.5000 Δ | ⛔ **0.0000 Δ** |
| best constructible (bracket-restricted) | 0.3090 Δ | 0.5000 Δ | 0.1911 Δ |

⭐⭐ **Under the harness's own top-1 scoring a perfect arm and a perfect lookup are literally
indistinguishable on row 2.** The declared margin was 0.2 Δ; the widest usable span over every readout is
0.1911 Δ. **No arm could ever have won this row under any metric the instrument supports.**

⛔⛔ **And the two lookup diagnostics the row-2 task specifies are produced by the JUDGE on a perfect
render:** `SNAPFRAC` null **0.6680** (registered thresholds 0.5 — *below* the null) and `d_near` 0.187 Δ <
`d_req` 0.313 Δ. **A hub that ran the arm first would have reported a lookup that was not there.**
⭐ Remedy, measured not proposed: an oracle-referenced readout gives the full 0.5 Δ span, 2.6× the
registered metric's range.
⚠ Also: the frozen `row2_unseen.json` realises only **48** corners and cannot support the ≥49 estimand.
⭐ On DEV, `clu_span` **ties a same-size PCA dictionary at +0.0000 Δ** on hue and loses on the whole tuple.

## 4. ⚠ `erasure` — a real distinction, and a framing correction against itself

**Claim 1, `object_hue = 0.5`, 3 store seeds, all at ≤ budget + 8 B:** recovery by an adversary holding the
weights — `knn_proto_gated` **1.000**, `basis_head_gated` **0.642 ± 0.021**, refitted CLU **0.088 ± 0.078**;
gap **+0.912**. Sensitivity and specificity controls both present (`clu_full` 0.539, `clu_shuffled` 0.020).

⛔⛔ **The hub's own correction, in its first paragraph:** refitted `basis_head` and `knn_proto` score
**0.000** — *better* erasure than the CLU's 0.088. **Erasure is a property of REFITTING, not of the CLU.
The defensible claim is `gate` vs `erasure`; ⛔ it is NOT CLU vs table.**
⚠ `shape = cube` is a **NULL** — the adversary failed calibration on 2 of 3 seeds; the cube is the shape a
degraded render degrades towards on this decoder. Eligible deletion targets: `object_hue` yes; `shape=cube`,
`scale`, `orientation` no.

**Claim 2, churn:** null (identical live set) **0.0000** ×3; after deleting 8,234 of 32,768, churn
**0.9954 ± 0.0012**, 2.87 of 6 factors changed. ⚠ **This does not reproduce wave 2's 18.8 %** — it is near
total — and the hub notes 99.5 % churn is also *instability*, not self-evidently a virtue.

**Claim 3, conjunction:** post/placebo — CLU **0.321** vs `basis_head` 0.557, `knn_proto` 0.446 ⇒ the CLU
forgets ~1.7× more. ⛔ **Nothing like wave 2's 2.9 % vs 60.4 %, and it is not quoted as if it were**; the CLU
holds these conjunctions weakly to begin with (0.382 vs 0.83–0.86).

⭐⭐ **The strongest thing in the hub is exact rather than measured:** after a conjunction deletion both
values still occur separately, so the 8-byte per-factor bitmap changes **ZERO bits** — it cannot express the
deletion and keeps serving the conjunction at 0.833–0.858 against the refitted CLU's 0.104. The only gate
that *can* express it is the full joint table (**60,000 B**), which then refuses **every row-3 and row-4
request**. ⇒ ⭐ **A gate can have the deletion or compositional generation, not both.**

---

## 5. What the Advisor is asked to rule on

1. ⛔⛔ **Does a §A.5 replication count as a scored run?** `factor-wells` flagged this against itself: it
   counted its replication inside run 2/3 (same PREREG, same configuration, no new selection). **If a
   replication counts separately, its 3/3 confound leg is over budget and must be struck** — and that leg is
   the only evidence that the win is alignment rather than balance. ⭐ **Cheap to close and it changes what
   the paper may claim.**
2. **May row 4 carry a primary claim while row 3 ties?** Half the hub's target row set is decoration, by its
   own report.
3. **What may `erasure` claim?** `gate` vs `erasure` is supported; CLU vs table is refuted by the hub's own
   refit controls. The conjunction contrast (§4, exact) looks stronger than the recovery numbers.
4. ⚠ **Churn: 99.5 % vs wave 2's 18.8 %.** Wave 2's framing was "the CLU churns 13.4× more than a table."
   At 99.5 % that framing inverts into an instability finding. **Which is it, and does the wave-2 result
   survive?**
5. **Row 2 is unmeasurable with the frozen judge.** Dropping it narrows the taxonomy; changing the
   instrument un-freezes the judge and breaks comparability with every wave-2 number. ⛔ Head/Advisor call.
6. **Seeds.** Everything is seed 0 and buys a verdict, not a paper number. The three-seed rule now has one
   result actually worth paying for.

## 6. ⭐ Cross-cutting — the thing two hubs found independently

**The read and the addressing are the bottleneck; the content and the bytes are not.**
`store-content` measured a per-value-means table recovering `orientation` at **0.9320** from exactly the
content the addressed read gets **0.1828** from. `unseen-valid` measured `clu_span` tying a same-size **PCA
dictionary** at +0.0000 Δ — the landscape is behaving as a basis. Wave 2 said the same from two other
directions (the read verb was the largest effect; above ~2× budget the head binds).
⇒ **Three waves have now pointed at the read. Nothing has yet changed it.**

## 7. Limits bounding every number above
1. ⛔ Seed 0 only, every hub. 2. ⚠ The Hub re-derived no statistic. 3. ⚠ `factor-wells` deviates from the
shipped fitter (declared, not bit-identical, identical across arms). 4. ⚠ `knn_proto` was selected at the
retired 688,128 B budget and deploys 475,136 B here — conservative direction. 5. ⛔ `unseen-valid` and
`store-content` returned **no git bundle** — their code lives in the outputs tree, so a reviewer can read
their numbers but cannot re-run them.
