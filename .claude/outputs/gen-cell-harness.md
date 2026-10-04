# gen-cell-harness — experiment-engineer report

**Task + acceptance criterion:** build `request → launch read → codes → ONE frozen decoder → image →
ONE external classifier` on the held-out combinations, across **CLU · store-removed · knn_proto · LACE ·
energy-based composition**, ≥3 seeds, and produce **the table** — every accuracy beside its
distinct-output count, byte ledger in-table.
**Status: done.** 5 arms (7 rows incl. LACE's two companion samplers) × 3 seeds, **0 NOT RUN**.
All three preconditions verified on disk before any other work.

> ## ⛔ DOWNSTREAM RECONCILIATION LIST — 4 items, needs a Hub owner (protocol §5 corollary)
> 1. ⭐⭐ **The image-space number does NOT re-order the arms at the top or the bottom, but it DOES
>    flip one adjacent pair.** `knn_proto` first and `store_removed` last on both instruments; but
>    `clu` (0.6568 ext / 0.6759 probe) and `lace/ld_corrected` (0.6605 / 0.6420) **swap**. ⛔ Any
>    sentence of the form *"CLU beats LACE's best Langevin leg"* is **instrument-dependent** and must
>    not be written. (§4)
> 2. ⭐ **LACE's per-sampler composition number carries a CLASSIFIER-REFIT sd of ≈ 0.023 that has
>    never been reported.** `gen-arms-a7-wiring`'s `ode` 0.6797 ± 0.0207 is a **seed** spread computed
>    from **one** classifier fit per seed. Four independent refits at a fixed (seed 0, selected
>    configuration, decoder, instrument) give `ode` **0.6476 ± 0.0227**, and the published 0.7077 sits
>    **2.65 sample-sd above** it, at **identical classifier quality** (own-val 0.9867 vs 0.9870).
>    ⇒ ⛔ **Do not quote a single LACE per-sampler number without this second spread.** (§6)
> 3. ⚠ **`gen_cell.n_fit/n_val` had to be set to `GenBaselinesConfig`'s 32,256/512, and this is a
>    general trap:** at 1,024 fit rows `knn_proto` fell 0.8763 → **0.8308** and LACE's whitening went
>    rank-deficient (|diag−1| 0.2382 → 0.6329). **Any future spoke that re-fits a rival on its own row
>    budget silently weakens it.** (§8)
> 4. ⚠ **Three of my reference constants were transcribed from RENDERED 4-dp tables, not from disk,
>    and a bit-identity claim was impossible until they were corrected.** Same class as
>    `gen-external-classifier` §11.4 — **third strike on this file family**. With the corrected
>    constants the CLU and `knn_proto` arms reproduce their published probe numbers **bit-identically,
>    3/3 seeds each, Δ = 0.0 exactly**. (§5)

---

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dial:** *composition, end to end, scored on IMAGES.*
- **Laundering control:** ⭐ the **`store_removed`** arm, a **full row on every table below**, never a
  footnote. It reads **0.5000 ± 0.0000** with **1** distinct output of 72 on all three seeds.
- **Falsifies:** the table itself. ⭐ **Composition was RELEASED as a claim in R29 — kill rule (ii)
  already fired.** ⛔ Nothing here tries to win it back and ⛔ no kill-rule verdict is rendered.
- **Does NOT falsify:** losing to an unbounded-channel arithmetic baseline (R27.1's already-rendered
  verdict and the metric-native-ceiling theorem), or absolute levels below the decoder ceiling.

---

## 0. ⛔ THE PRECONDITIONS — all three, verified on disk, shown

```
⭐ PRECONDITION 1/2: classifier_gate.json gate=PASS (floor 0.95, 3 seeds)
   .claude/outputs/gen-external-classifier/classifier_gate.json   "gate": "PASS"
   per_seed {0: 1.0, 1: 1.0, 2: 1.0}, statistic = in-corner shape balacc on REAL held-out images
⭐ PRECONDITION 2/2: refusal_scores.json status=complete (not_run=[])
   .claude/outputs/gen-arms-a7-wiring/refusal_scores.json         "status": "complete"
```
Both are **parsed and compared as values**, ⛔ never string-matched — a test proves a file that merely
*mentions* `"PASS"` does not satisfy the gate, and three tests show the gate **refusing** (`FAIL`
verdict / `partial` status / missing file). They also run **inside** every production invocation and
are written into each seed manifest.

**Precondition 3 (R27.3), mechanical, from `gen-arms-a7-wiring`'s own manifests**
(`.claude/outputs/gen-cell-harness/precondition3.txt`):

| seed | gate fired | selected shrinkage / \|diag−1\| | pre-registered leg (best comp.) | `_wexact` leg (best comp.) | stronger |
|---|---|---|---|---|---|
| 0 | **yes** | `1e-3` / 0.2382, not exact | **0.7418** (`ld_published`) | 0.6913 (`ld_published`) | **PRE-REGISTERED** |
| 1 | no | `none` / 0.0567, exact | 0.6732 (`ode`) | — (no fallback needed) | — |
| 2 | **yes** | `ledoit_wolf` / 0.1438, not exact | **0.6582** (`ode`) | 0.6563 (`ld_corrected`) | **PRE-REGISTERED** |

⭐ **Both legs ran on BOTH gated seeds and the stronger is quoted per axis.** On the **composition**
axis — the axis this harness measures — the **pre-registered** leg is stronger on both, so quoting it
is *correct and required*, and LACE's published composition row `ode` = **0.6797 ± 0.0207**
(re-derived from disk here: 0.7077 / 0.6732 / 0.6582 → 0.6797 ± 0.0207, **match**) **stands
unchanged**. ⛔ I did not use R27.3 to swap it; this harness **inherits** that exact configuration.

⚠ **Residue carried, unresolved, as instructed:** on seeds 0 and 2 the pre-registered leg still ran
with the prior-score omission **unlicensed**, and R27.3's second route — the general form **with** the
prior score in the ODE drift — is **NOT RUN**. It needs a new drift term in `sample_ode`, a sampler
change nobody has made, and ⛔ I did not make it here.

---

## ⭐ Flag provenance (mandatory, critique M4)

| item | value |
|---|---|
| branch / base | `agent/experiment-engineer/gen-cell-harness`, **rebased onto `main @ 22f9dc5`** (spawn base was `be27713`; `main` advanced by the `gen-navigation` + `push_repo` merges mid-run — §10) |
| commits | `198d8c0` · `55e8359` · `56e3eb3` · `1616f9e` |
| env | **main venv reused, ⛔ no `uv sync` in the worktree** — jax **0.9.0**, equinox 0.13.4, numpy 2.4.1, h5py 3.16.0, Python 3.11.13, `JAX_PLATFORMS=cpu`, float32 |
| dataset · pretext · condition | `shapes3d` · `recon` · `recomb2element` (Montero) |
| seeds | **0, 1, 2** — ⛔ none NOT RUN |
| **decoder** | ⭐ `decoder_source="winner"`, `dec_recipe = v1_capacity_schedule` selected **explicitly** (⛔ never the `"baseline"` default, which FAILED the gate). Hashes `01ef91f75eea…` / `001011bc603a…` / `2d7eea75add7…` — **hash AND recipe asserted per ARM per SEED**, and the run aborts if either moves between arms |
| **external classifier** | `gen-external-classifier`'s gated checkpoints, **loaded not fitted**, hash-asserted against each seed's own manifest: `c8741ac417d8…` / `aa69b436989b…` / `ad16c4e53273…` (978,873 params, 1,500 steps) — **one instrument for all 7 rows in a seed** |
| store | ⭐ **loaded from `gen-read-compose`'s checkpoints**, ⛔ not re-trained (that is what makes the CLU arm's probe number bit-comparable). `K=5`, `n_modes=64`, `D=2048`, **30 bits/read** (R20's design constant), **524,288 B asserted in-run** |
| grouping | `grouping_source="per_seed"`; re-derivation reproduces `store_plan.json` **bit-identically** on its own seed |
| frozen encoder / probe | asserted against `bridge_manifest.json`; cache re-encode check **bit-identical on 64 rows/seed** |
| rows | B **72** · MS **198 / 199 / 199** · rivals' fit **32,256** · val **512** (⭐ `GenBaselinesConfig`'s own values — §8) |
| feature caches | seed 0 `scope="full"` (480,000 rows); seeds 1–2 `scope="used"` (9,288) with coverage extended through the **frozen** encoder for 32,199 rows |
| **LACE** | configuration **INHERITED** from `gen-arms-a7-wiring`: (σ=0.1, `both`, `1e-3`) / (0.03, `both`, `none`) / (0.03, `both`, `ledoit_wolf`); whitening reproduced to 4 dp (0.2382 / 0.0567 / 0.1438); classifier re-fitted at the shipped **`n_epochs=50`**, batch 128, lr 1e-4, 1000-iter warmup, 12,600 steps; own-val 0.9870 / 0.9495 / 0.9678 |
| **`ebm_compose`** | 6 **independent** experts (2×256 SiLU), 20 epochs, Adam 1e-3, **14,226,660 B**; own-val **0.9925 / 0.9961 / 0.9967**; 36-trial grid, selected (lr 0.03, σ_prior 2.0, 200/50/200 steps, clip=False) |
| tuning budget | **36 trials for every arm**; realised **0** (clu, store_removed, knn_proto — no tunable axis), **36** (LACE, inherited), **36** (EBM, run here) |
| non-default flags | ⛔ **none.** Every number is at `GenCellConfig`'s defaults; the block is new, so its defaults **are** the production configuration |
| wall clock | 622 / 650 / 707 s per seed + 1,248 s for the LACE refit-noise control + 375 s for the tip re-score; **CPU only** |
| ⛔ CSF3 | unreachable (2FA). **Every cluster item NOT RUN** — §9 |

---

## 1. ⭐⭐ THE TABLE — in-corner `shape` balanced accuracy, on IMAGES

Positives = each arm's renders of the **72 held-out (B) requests**; negatives = its renders of the
**matched-seen** requests in the same colour corner. Chance **0.5**. AUROC is the mandatory companion
and ⛔ never the gate; TPR is *detection* and ⛔ never the gate.

| arm | **deployed bytes** | tuning trials | **IMAGE balacc (external)** | per seed | AUROC | TPR / TNR | distinct **vectors** of 72 | ⭐ distinct **factor-tuples** of 72 | probe companion |
|---|---|---|---|---|---|---|---|---|---|
| **`knn_proto`** | **475,136** ⚠ | 0 of 36 | **0.9119 ± 0.0222** | 0.9040 / 0.9422 / 0.8894 | 0.9875 | 1.000 / 0.824 | 72.0 | **54.7 ± 8.1** | 0.8884 ± 0.0088 |
| `lace/ld_corrected` | n/a (not byte-matched) | 36 of 36 | 0.6605 ± 0.0197 | 0.6338 / 0.6669 / 0.6808 | 0.7299 | 0.435 / 0.886 | 72.0 | 71.3 ± 0.5 | 0.6420 ± 0.0120 |
| ⭐ **`clu`** (30-bit read) | **524,288** + 74,240 head | 0 of 36 | **0.6568 ± 0.0226** | 0.6540 / 0.6307 / 0.6858 | 0.7653 | 0.736 / 0.578 | **9.0 ± 0.8** | ⭐ **9.0 ± 0.8** | 0.6759 ± 0.0107 |
| **`lace`** (`ode`, the published row) | n/a | 36 of 36 | 0.6347 ± 0.0177 | 0.6143 / 0.6324 / 0.6575 | 0.7186 | 0.394 / 0.876 | 72.0 | 71.3 ± 0.9 | 0.6389 ± 0.0112 |
| `lace/ld_published` | n/a | 36 of 36 | 0.5580 ± 0.0564 | 0.6288 / 0.5544 / 0.4908 | 0.7599 | 0.190 / 0.926 | 72.0 | 56.3 ± 4.1 | 0.5924 ± 0.1010 |
| **`ebm_compose`** | **14,226,660** ⚠ | 36 of 36 | 0.5408 ± 0.0087 | 0.5455 / 0.5287 / 0.5483 | 0.6077 | 0.199 / 0.883 | 72.0 | 71.7 ± 0.5 | 0.5318 ± 0.0090 |
| ⭐ **`store_removed`** (laundering) | 524,288 + 74,240 head | 0 of 36 | **0.5000 ± 0.0000** | 0.5000 / 0.5000 / 0.5000 | 0.5008 | 0.000 / 1.000 | **1.0** | ⭐ **1.0** | 0.5000 ± 0.0000 |
| *decoder ceiling (real features)* | — | — | *0.9357 ± 0.0193* | 0.9085 / 0.9472 / 0.9514 | — | — | *72* | — | *0.9414 ± 0.0179* |

### ⛔ THE BYTE LEDGER, IN THE TABLE AND IN WORDS

- **CLU store = 524,288 B** (`n_modes·D·4`), **asserted in-run**, the run aborts on any other value.
  The **163,840 B** inter-well coupling is **EXCLUDED** (R26.3 — dreaming left the critical path);
  ⛔ the retired **688,128 B** target is **not inherited**. The **74,240 B** addressing head is
  reported *beside* the store by `store_bytes.json`'s own convention.
- **`knn_proto` = 475,136 B**, in budget. ⚠ **DECLARED, NOT SILENT:** the shipped `knn_proto`
  *configuration* was selected at the retired **688,128 B** budget — a **~31 % byte advantage it
  should not have**. ⭐ This is the **conservative** direction: an over-resourced competitor that wins
  is a stronger result for it. ⭐ And it is a **measured no-op** — `gen-arms-a7-wiring` P5 found
  `knn_proto` **bit-identical at both budgets** (Δ = 0.000000, 3/3), which this harness independently
  re-confirms by reproducing the 524,288-budget numbers bit-identically (§5).
- **LACE and `ebm_compose` are NOT byte-matched and are not claimed to be.** LACE has no enumerable
  store to match. ⚠ `ebm_compose` deploys **14,226,660 B — 27× the CLU store** and is scored anyway:
  an over-resourced rival that loses is the honest direction, and it is stated rather than trimmed.
- ⚠ **Supervision asymmetry, declared (R9.3):** LACE and `ebm_compose` both get **dense factor labels
  over the whole train split**; the store gets **none**. Their own validation accuracies (0.9495–0.9870
  and 0.9925–0.9967) show they are strong, ⛔ not hobbled.

---

## 2. ⭐⭐ THE THREE THINGS NO PREVIOUS TABLE COULD SHOW

**(1) The image-space number, from an instrument that is not the decoder's own family.** Every row
above is read by `gen-external-classifier`'s CNN, fitted on **real train-split images disjoint from the
decoder's and the probe's fit rows**, which per-item disagrees with the probe on 2.94 % of *renders*
and on **0 of 3,884 reals**. ⛔ No previous phase-1 composition table had one.

**(2) The distinct-output count beside every accuracy — and it needs THREE columns, not one.**

| count | `clu` | `store_removed` | `knn_proto` | `lace` | `ebm_compose` |
|---|---|---|---|---|---|
| distinct **feature vectors** on B (of 72) | **9.0 ± 0.8** | **1.0** | 72.0 | 72.0 | 72.0 |
| distinct **factor-tuples** read off B (of 72) | **9.0 ± 0.8** | **1.0** | **54.7 ± 8.1** | 71.3 ± 0.9 | 71.7 ± 0.5 |
| distinct **factor-tuples** on MS (of ~199) | 24.3 ± 5.4 | 1.0 | 148.7 ± 12.7 | 195.3 ± 0.9 | 198.0 |

⚠⭐ **The vector count is TRIVIALLY 72 for any continuous sampler** — a float draw cannot repeat —
so `lace`'s and `ebm_compose`'s 72/72 is **not** evidence of diversity and must never be put in the
same column as the read's 9. ⭐ **The cross-arm comparable column is the factor-tuple count**, and on
it the ranking is `ebm ≈ lace (71) > knn_proto (55) > clu (9) > store_removed (1)`.
⭐ The read's **9 / 8 / 10** distinct code configurations on B reproduce R30/R31's per-seed counts
**exactly**, and MS's **35 / 24 / 24** likewise — so the commitment collapse is reproduced here on a
third instrument and is ⛔ **not** an artefact of the probe.
⭐ ⛔ **And it is a property of the CORNER, not of held-out-ness** (R31, binding): MS — *seen* data in
the same corner — collapses at the same rate.

**(3) The store-removed row.** **0.5000 ± 0.0000** in-corner (TPR 0.000, TNR 1.000), **0.0000 ± 0.0000**
B `shape`, **one** distinct output of 72, on all three seeds. ⇒ **the ablation is provably dead, and
every non-chance number in the CLU rows is a number the store supplied.**

---

## 3. The other half of the table — composition on B, on IMAGES

| arm | B aggregate (ext) | B `shape` (ext) |
|---|---|---|
| `knn_proto` | **0.9005 ± 0.0182** | **1.0000 ± 0.0000** |
| `clu` | 0.5949 ± 0.0082 | 0.7361 ± 0.2051 ⚠ |
| `lace` (`ode`) | 0.3056 ± 0.0208 | 0.3935 ± 0.0524 |
| `lace/ld_corrected` | 0.2855 ± 0.0159 | 0.4352 ± 0.0346 |
| `lace/ld_published` | 0.2276 ± 0.0165 | 0.1898 ± 0.1029 |
| `ebm_compose` | 0.2130 ± 0.0132 | 0.1991 ± 0.0327 |
| `store_removed` | 0.1204 ± 0.0131 | **0.0000 ± 0.0000** |

⚠ **B `shape` is a one-class DETECTION rate** (R19's carry-forward, R20's whole reason for moving the
gate): the held-out cell realises exactly one `shape` value, so this column is **TPR** and ⛔ must not
be read as a composition score. §1's in-corner balanced accuracy is the like-for-like number.
⚠ `clu`'s ±0.2051 is that degeneracy showing: seed 1 detects 1.000 at TNR 0.261.
⚠ **The B-aggregate column is dominated by the four non-compositional factors** (four of six Shapes3D
factors sit at ≈1.0 for a good arm); `ebm_compose` and `lace` lose it mostly on hue, ⛔ not on shape.

---

## 4. ⭐⭐ Does swapping the instrument change the verdict? — **mostly no, and exactly where it does**

| arm | external | probe | Δ = ext − probe |
|---|---|---|---|
| `knn_proto` | 0.9119 | 0.8884 | **+0.0235** |
| `lace/ld_corrected` | 0.6605 | 0.6420 | +0.0185 |
| `clu` | 0.6568 | 0.6759 | **−0.0191** |
| `lace` (`ode`) | 0.6347 | 0.6389 | −0.0041 |
| `lace/ld_published` | 0.5580 | 0.5924 | −0.0344 |
| `ebm_compose` | 0.5408 | 0.5318 | +0.0090 |
| `store_removed` | 0.5000 | 0.5000 | +0.0000 |
| *decoder ceiling* | *0.9357* | *0.9414* | *−0.0057* |

```
ranking by external: knn_proto > lace/ld_corrected > clu > lace > lace/ld_published > ebm > store_removed
ranking by probe   : knn_proto > clu > lace/ld_corrected > lace > lace/ld_published > ebm > store_removed
identical: False
```
⭐ **Top and bottom are instrument-invariant**, and every headline statement survives: `knn_proto`
first by **+0.25**, `store_removed` at chance, the read **+0.157 above its own ablation**.
⛔ **But ONE adjacent pair flips**: `clu` and `lace/ld_corrected` are separated by **0.0037** on the
external instrument and **0.0339** on the probe, with **opposite signs**. Both gaps are at or below the
sampler-noise floor (§6). ⇒ ⛔ **"the read beats LACE's best Langevin leg" is not a statement this
table supports.** It is reported as a **tie inside the noise**, on both instruments.
⭐ The ceiling's own Δ (−0.0057) reproduces `gen-external-classifier`'s leg-3 number **to four
decimals**, which is the independent check that the two instruments here are the two it gated.

---

## 5. ⭐⭐ Nothing moved that I did not move — the plumbing proof

**Probe-vs-probe, per seed, against each arm's OWN published number** (⛔ never external-vs-probe: that
would confound the instrument with the arm, which is the one thing this table exists not to do):

| arm | seed 0 | seed 1 | seed 2 | verdict |
|---|---|---|---|---|
| **`clu`** | 0.6609848484848484 | 0.6809045226130653 | 0.6858249581239531 | ⭐ **Δ = 0.0 EXACTLY, 3/3 — bit-identical to `gen-read-compose`** |
| **`knn_proto`** | 0.8762626262626263 | 0.8919597989949748 | 0.8969849246231156 | ⭐ **Δ = 0.0 EXACTLY, 3/3 — bit-identical to `gen-arms-a7-wiring`** |
| `lace` (`ode`) | 0.6237 vs 0.7077 (**−0.0840**) | 0.6423 vs 0.6732 (−0.0309) | 0.6505 vs 0.6582 (−0.0077) | see §6 |

⭐⭐ **Six of six deterministic comparisons are bit-identical**, from a store checkpoint, a prototype
table and a decoder that three different spokes wrote. That is the strongest available evidence that
*"an arm moved and you cannot name what moved it"* did not happen here: for `clu` and `knn_proto`
**nothing moved at all**, so the entire external-vs-probe Δ in §4 is **the instrument**, by
construction.
⭐ **Tip re-score:** seed 0 re-run at the branch tip into a separate directory reproduces **every**
deterministic arm bit-identically — `clu` 0.6540, `store_removed` 0.5000, `knn_proto` 0.9040 — and
also `ebm_compose` **0.5455** with the identical selected trial and a `val_aggregate` equal to 16
digits, so that arm is fully seed-deterministic too.

---

## 6. ⭐⭐ How much of the spread is the sampler, and did I weaken LACE?

**(a) Sampler noise, measured.** Three independent **sampler** keys at a fixed (seed 0, configuration,
classifier, decoder, instrument): `lace/ode` image-space balacc = **0.6023 / 0.6383 / 0.5903**,
**sd = 0.0204**.

**(b) Classifier-refit noise, measured** (`lace_refit_noise.json`, 3 refits × 423/403/422 s). ⛔
`gen-arms-a7-wiring` did **not persist its LACE classifier**, so re-running the rival *must* re-fit it.
At a fixed (seed 0, **inherited** configuration, decoder, instrument), changing only the classifier's
PRNG stream:

| leg | values (probe) | mean ± sd |
|---|---|---|
| `ode` | 0.6326 / 0.6679 / 0.6660 | 0.6555 ± 0.0162 |
| `ld_published` | 0.6313 / 0.6288 / 0.6995 | 0.6532 ± 0.0328 |
| `ld_corrected` | 0.6086 / 0.6414 / 0.5846 | 0.6115 ± 0.0233 |

⭐ Pooling the main run's fit with these three: seed 0 `ode` over **four independent classifier fits**
= **0.6476 ± 0.0227** (sample sd). The published **0.7077** sits **2.65 sample-sd above** it — at
**identical classifier quality** (published own-val **0.9867**, mine **0.9870**).
⇒ ⭐ **The −0.0840 is a different draw, not a weakening.** Nothing about LACE's method, constants or
configuration changed: the configuration is *inherited byte-for-byte from the rival's own manifest*,
`n_epochs = 50` is the shipped one (⛔ not 100, ⛔ not 200), the whitening diagnostic reproduces to 4
decimals, and the OR/NOT constants are never exercised (every request is a pure conjunction).
⚠ **Stated honestly and not more strongly:** with n = 4 I cannot *exclude* that `gen-arms-a7-wiring`'s
single fit was genuinely better, and ⛔ I do not claim mine is better. What I claim is that the delta
is **fully inside the refit distribution I measured**, and that the published number is a **high** draw
from it.

**(c) The fairer never-weaken check — LACE's BEST OF FAMILY** (R9.4: *report LACE's best*):

| seed | published best | measured best | Δ |
|---|---|---|---|
| 0 | 0.7418 (`ld_published`) | 0.7348 (`ld_published`) | −0.0069 |
| 1 | 0.6732 (`ode`) | 0.6423 (`ode`) | −0.0309 |
| 2 | 0.6582 (`ode`) | 0.6531 (`ld_corrected`) | −0.0052 |

⭐ **published 0.6911 ± 0.0364 · measured 0.6767 ± 0.0413 · Δ = −0.0143** — **inside the 0.0204
sampler-noise floor and well inside the 0.0227 refit floor.** ⇒ **LACE was not weakened at the level
R9.4 says to report it at.** Only the per-sampler *assignment* moved.

**(d) ⚠ How much of the spread is sampler noise versus arm difference — plainly, on 72 items.**

| quantity | value | read against the 0.0204 noise floor |
|---|---|---|
| `knn_proto` − `clu` | **+0.2551** | **12.5×** — an arm difference |
| `clu` − `store_removed` | **+0.1568** | **7.7×** — an arm difference |
| `clu` − `lace`(`ode`) | +0.0221 | **1.1×** — ⛔ **noise; report as a tie** |
| `clu` − `lace/ld_corrected` | −0.0037 | **0.2×** — ⛔ **noise; report as a tie** |
| `lace` − `ebm_compose` | +0.0939 | 4.6× — an arm difference |
| across-seed sd of `clu` | 0.0226 | 1.1× the noise floor |

⇒ ⭐ **Only two contrasts in this table are larger than the noise: `knn_proto` over everything, and the
read over its own ablation. ⛔ Everything in the middle band (0.53–0.66) is a four-way tie on 72 items
and must be written as one.** ⚠ 72 held-out combinations is a **small** set; the seed spread on most
rows is the same size as the sampler spread, and I am ⛔ not going to let it carry more than that.

---

## 7. ⭐ The pre-registration scorecard
`PREREG.md` filed **21:30:24**; the first harness output (`smoke.log`) is **21:33:56** — mtime-verifiable.
⚠ **The margin is 3.5 minutes, not hours:** the prereg was written before the harness produced a single
number, but in the same session. ⛔ Read it as *"committed before the first measurement"*, ⛔ not as an
overnight gap. (Same disclosure `gen-external-classifier` made at 37 seconds.)

| # | prediction | measured | verdict |
|---|---|---|---|
| **P0** | `clu` probe-space number bit-identical to `gen-read-compose` | **Δ = 0.0 exactly, 3/3** — and `knn_proto` too, which I had not predicted | ⭐ **HELD, and stronger than pre-registered** |
| P1 | `clu` external ≈ 0.67 [0.61, 0.73] | **0.6568 ± 0.0226** | **HELD** |
| P2 | `store_removed` = 0.5000 exactly, 1 distinct image | **0.5000 ± 0.0000**, TPR 0 / TNR 1, **1** of 72 | ⭐ **HELD, exactly** |
| P3 | `knn_proto` external ≈ 0.88 [0.83, 0.93] | **0.9119 ± 0.0222** | **HELD** (top of the band) |
| P4 | `lace` external ≈ 0.68 [0.56, 0.80] | **0.6347 ± 0.0177** | **HELD** |
| **P5** | distinct vectors on B: `clu` **9 / 8 / 10** exactly; others 72/1 | **9 / 8 / 10**, `store_removed` 1, the rest 72 | ⭐ **HELD, exactly, 3/3** |
| P6 | factor-tuples the comparable column; `clu` ≤ 12, `knn_proto` ≥ 55 | `clu` **9.0**, `knn_proto` **54.7 ± 8.1** | **SPLIT** — the mechanism held; ⛔ `knn_proto` missed its bound by 0.3, and the bound was mine, not the arm's |
| P7 | `ebm_compose` between `lace` and `knn_proto`, point 0.70 [0.50, 0.90] | **0.5408 ± 0.0087** — **below LACE**, at the bottom of the band | ⛔ **FALSIFIED on the point, in-band on the interval** — §8 |
| P8 | sampler-noise sd ≤ 0.05 | **0.0204** | ⭐ **HELD** |
| **P9** | the instrument does not re-order the arms | top and bottom invariant, ⛔ **one adjacent pair flips** (`clu` ↔ `lace/ld_corrected`) | ⛔ **SPLIT — and it is the finding of §4** |

**Why P7 failed, and it is a result rather than a miss.** `ebm_compose`'s experts are *excellent*
classifiers (own-val **0.9925–0.9967**, better than LACE's) and its 36-trial grid selected the
**largest** step size and the **widest** prior on all three seeds — i.e. the selection ran to the edge
of the grid, which is the signature of a sampler that cannot reach the data manifold at any setting the
grid contains. ⭐ **The lesson is LACE's own R9.1 in reverse:** LACE's whitening is what makes a
Gaussian prior *exact* in its sampling coordinates; my arm's isotropic prior over standardised raw
post-ReLU features is **not**, and the composition degrades accordingly. ⛔ I am **not** presenting this
as evidence that energy-based composition is weak — it is evidence that **an un-whitened one is**, and
the honest caveat is that a stronger version of this rival exists and I did not build it (§11.2).

---

## 8. How I verified — commands and observed output

```
# ⚠ smoke first (plumbing only, ⛔ never a number): 60 s, whole path, every mechanism fired
python -u -m chlu.experiments.exp_gen_cell --stage all --quick --out .claude/scratch/.../smoke

# production, 3 seeds, ~35 min
python -u -m chlu.experiments.exp_gen_cell --dataset shapes3d --pretext recon \
    --stage all --seed 0 --seed 1 --seed 2 --out .claude/outputs/gen-cell-harness/runs/shapes3d_recon
  ⭐ PRECONDITION 1/2: classifier_gate.json gate=PASS   ⭐ PRECONDITION 2/2: status=complete
  ⭐ decoder: winner recipe=v1_capacity_schedule hash=01ef91f75eeaae7b… (hash + recipe asserted)
  ⭐ external classifier c8741ac417d88ebf… — THE instrument, identical across every arm in this seed
  ⭐ store loaded from gen-read-compose/.../store_s0.pkl (⛔ not re-trained)
  ⭐ byte ledger: store 524288 B + addressing head 74240 B (⛔ coupling 163840 B EXCLUDED; ⛔ not 688128)
  ⭐ no-bypass: pure codebook gather, 30 bits/read · request-only path asserted on all four legs
  ⭐ [clu] IMAGE balacc=0.6540 · distinct on B: 9 vectors / 9 factor-tuples of 72 · probe 0.6610
  ⭐ [store_removed] IMAGE balacc=0.5000 (TPR 0.000 / TNR 1.000) · distinct on B: 1 / 1 of 72
  ⭐ [knn_proto] IMAGE balacc=0.9040 · distinct 72 / 50 · probe 0.8763
  ⭐ [lace] inheriting the SELECTED configuration from disk (36 trials, ⛔ not re-selected)
  ⭐ sampler-noise floor [lace]: sd=0.0204 over 3 independent keys
  ⛔ REGRESSION on lace: measured 0.6237 vs published 0.7077 (Δ -0.0840) — reported, ⛔ not absorbed
```

**The four structural assertions, all passing on every seed** (`request_only_path` in each manifest):
`signature_ok` ✓ · `no_feature_argument_on_the_path` ✓ · `posterior_re_derived_from_requests_bit_identical`
✓ · `soft_read_not_imported` ✓ — plus `assert_no_bypass`: pure codebook gather, **30.0 bits/read**.
⭐ **The CLU path touches no feature tensor**, checked by name and by numeric decoy substitution.

**Tests** (⛔ no full suite started — R28/R30 policy, and a 16 GB machine with a live run):
```
pytest tests/test_gen_cell.py                                  -> 24 passed, 0 skipped, 0 failed
pytest test_gen_cell test_config test_gen_compose test_gen_classifier test_gen_baselines
       test_gen_certificate test_gen_store test_gen_data_splits -> 245 passed, 0 skipped, 0 failed (3:20)
# after rebasing onto the NEW main (gen-navigation merged in mid-run):
  ... + tests/test_gen_navigate.py                              -> 279 passed, 0 skipped, 0 failed (3:26)
ruff check chlu/config.py chlu/experiments/exp_gen_cell.py tests/test_gen_cell.py -> All checks passed!
```
⭐ **Skip count read and named, per R31's fifth-strike rule: ZERO skips** — `CHLU_GEN_DATA_ROOT` was set
on every invocation, so the real-data path **executed**.
⛔ `ruff format` would reformat my files *and* several shipped ones (`gen_bridge.py`,
`exp_gen_compose.py`): the repo is not `ruff format`-clean, so I did not reformat.

**⚠ Four defects found on the way, each fixed and each reported** (commit `1616f9e`):
1. ⛔ **LACE was reported NOT RUN on seed 1** — `fit_whitening` takes the **strings**
   `"none"`/`"ledoit_wolf"`; I mapped `"none"` to Python `None` → `TypeError`. Fixed to be identical to
   `exp_gen_baselines._parse_shrinkage`; the re-run reproduces the published |diag−1| = 0.0567 exactly.
   ⭐ The incremental-write design meant the repair cost **one arm on one seed**, not a run.
2. ⛔ **A repair invocation (`--seed 1 --arm lace --stage all`) silently overwrote the 3-seed aggregate
   with a 1-seed one.** Caught because the file was read immediately after. The aggregate now unions
   the requested seeds with every `seed_*` on disk.
3. ⛔ **Three reference constants were transcribed from rendered 4-dp tables, not from disk** — see
   reconciliation item 4. Corrected, and pinned by a test that **refuses a constant that looks like a
   4-dp figure**.
4. ⛔ **`gen_cell.claude_root` was a DEAD knob** (`exp_gen_compose._claude_root` reads
   `gen_compose.claude_root`), and the preconditions test **did not fire** until `_cell_root` existed —
   a config knob that does nothing is how a test stops testing anything.
   ⚠ **Provenance disclosure:** that edit landed *after* the seed-0/2 run began. It is **provably
   inert** for the reported run (`CHLU_CLAUDE_ROOT` was set and `gen_cell.claude_root` is `None`, so
   both resolvers return the same path) **and** the seed-0 tip re-score (§5) reproduces every
   deterministic arm bit-identically.
5. ⚠ **Two of my own tests were wrong first and were fixed as tests, ⛔ not as code:** the stand-in
   reader was not traceable under `eqx.filter_jit(jax.vmap(…))`, and the continuous-sampler fixture
   jittered across an integer bin boundary. ⭐ Recorded because a test that has never failed has never
   been tested.

**⚠ The row-budget trap, measured** (reconciliation item 3): my first defaults were `n_fit = 8192`.
In the smoke at 1,024 rows `knn_proto`'s probe number fell **0.8763 → 0.8308** and LACE's whitening went
rank-deficient (**|diag−1| 0.2382 → 0.6329**, `n_fit < D = 2048`). ⇒ ⛔ **A rival re-fitted on a spoke's
own row budget is a weakened rival.** `GenCellConfig.n_fit/n_val` now default to
`GenBaselinesConfig`'s **32,256 / 512**, which is *why* §5's bit-identity is achievable at all.

---

## 9. ⛔ CSF3 — every cluster item **NOT RUN** (unreachable, 2FA is Head-only)

No compute was committed. The launcher's `STAGE=cell` already routes to
`chlu.experiments.exp_gen_cell` (`DEF_MODULE`, line ~263) with `UPSTREAM="store"` (line ~392), so **the
stage that has been aborting since `gen-csf3-launcher` now has its module.**

```bash
export GEN_GIT_COMMIT=$(git rev-parse HEAD)      # 1616f9e on this branch
cd ~/scratch/CHLU && mkdir -p logs

# (i) launcher-native, ONE ARM PER ARRAY TASK (see the caveat below)
sbatch --export="ALL,DATASET=shapes3d,PRETEXT=recon,STAGE=cell,\
ARMS=clu store_removed knn_proto lace ebm_compose,SEEDS=0 1 2,GEN_GIT_COMMIT=$GEN_GIT_COMMIT" \
       -a 0-14%3 -t 04:00:00 scripts/csf3/job_gpu_gen.sh

# (ii) one task per SEED, whole table in one manifest — ⛔ needs no launcher edit
sbatch -p gpuA -G 1 -n 1 -c 8 -t 04:00:00 -J clu-gen-cell -a 0-2%2 \
  -o logs/%x-%j.out -e logs/%x-%j.err --wrap \
  "source scripts/csf3/env.sh && python -u -m chlu.experiments.exp_gen_cell \
     --dataset shapes3d --pretext recon --stage cell --seed \$SLURM_ARRAY_TASK_ID \
     --out \$SCRATCH/CHLU-gen/workspaces/shapes3d_recon/cell"
```
**Status of every line above: NOT RUN.**
⚠ **Interface caveat, flagged not fixed** (`scripts/csf3/` is ⛔ outside my ownership): the launcher
runs **one (arm, seed) per array task into its own `cell/<arm>_s<seed>/` directory**, but this module
writes **one manifest per seed containing all arms**, and `--stage aggregate` reads `seed_*/` under a
single root. Form (i) therefore produces 15 single-arm manifests that need a gather step; **form (ii)
is the faithful one**. ⭐ A cluster run would change nothing about the verdicts — the whole table is
35 minutes of laptop CPU. Its value is **more seeds** (the honest lever on §6's noise floor) and the
wide condition.
⛔ The **cluster smoke gate (R21/R23.6) is unrelated to this spoke and still shut**; nothing here is a
pre-registered cluster cell.

---

## 10. Git footprint

| | |
|---|---|
| branch | `agent/experiment-engineer/gen-cell-harness` |
| worktree | `/Users/user/Desktop/CHLU-gen-harness` (fresh; ⛔ the shared checkout was on another agent's branch `pilot-ttt-nan-and-d5-wiring` and was **never touched**) |
| base | spawn base `main @ be27713`; ⚠ **`main` advanced to `22f9dc5` mid-run** (`gen-navigation` + two `push_repo` merges) and the branch is **rebased onto it** (⛔ never onto the stale `origin/main`, §7.21) |
| commits | `198d8c0` config block · `55e8359` the module · `56e3eb3` the tests · `1616f9e` the four defects |
| files | `chlu/config.py` **+226 / −0** · `chlu/experiments/exp_gen_cell.py` (new, 1,441) · `tests/test_gen_cell.py` (new, 430) |
| ⛔ not touched | `chlu/core/gen_compose.py` (read-only, as instructed), `gen_store.py`, `gen_bridge.py`, `gen_certificate.py`, `gen_request.py`, `gen_dream.py`, `chlu/eval/rivals/**`, `chlu/eval/gen_classifier.py`, `chlu/data/**`, `scripts/csf3/**`, `chlu/experiments/exp_gen_baselines.py` |
| ⚠ **the predicted same-anchor conflict FIRED** | `gen-navigation` appended at the same `chlu/config.py` tail. Resolved by the **whole-file rule** (R30 housekeeping): took the peer's entire file, re-applied my delimited block + three wiring sites with a script, and **asserted `git diff --numstat main -- chlu/config.py` = `226 0`** — **zero deletions**, `ast.parse` clean. ⛔ No hunk-by-hunk "keep both". |
| ⛔ no CLI hook | `chlu/cli/experiment_cmd.py` is **not** in my task's file-ownership list and is a shared file; the module ships its own argparse `main()` (`python -m chlu.experiments.exp_gen_cell`), like `exp_gen_classifier`/`exp_gen_compose`. **Same call `gen-external-classifier` made; the ~35-line hook is a Hub decision.** |
| unresolved conflicts | none |
| ⛔ not pushed, no PR | branch left for Hub/Head review |

**Verified from the MAIN repo** (protocol §3.2):
`git -C /Users/user/Desktop/CHLU log --oneline main..agent/experiment-engineer/gen-cell-harness` lists
all four hashes; `git diff --numstat main …` = **2,097 insertions, 0 deletions, 3 files**.
⛔ The worktree is **left in place** for review.

**Artefacts** (all under `.claude/outputs/gen-cell-harness/`):
`runs/shapes3d_recon/seed_{0,1,2}/cell_manifest.json` (every number, every assertion, every hash) ·
`runs/shapes3d_recon/cell_results.json` (the 3-seed aggregate) · `cell_table.png` ·
`tables.md` + `table.py` (regenerates every table above from the manifests) · `PREREG.md` ·
`precondition3.txt` · `lace_refit_noise.json` · `runs_s0_recheck/` (the tip re-score) ·
`run_main.log` / `run_s1_lace.log` / `run_lace_refit.log` / `run_s0_recheck.log`.
Scratch: `.claude/scratch/gen-cell-harness/{smoke/,smoke.log,lace_refit_noise.py,recompute_regression.py}`.

---

## 11. Open questions / follow-ups / risks

1. ⭐⭐ **The middle of the table is a four-way tie and must be written as one.** `clu` 0.657,
   `lace/ld_corrected` 0.661, `lace` 0.635, `ebm_compose` 0.541 against a **0.020** sampler floor and a
   **0.023** refit floor. ⛔ Only two contrasts clear it: `knn_proto` over everything (+0.26) and the
   read over its own ablation (+0.16). **72 held-out combinations cannot support more.** The lever is
   **more seeds**, which is what the cluster is for.
2. ⚠ **`ebm_compose` is MY re-implementation and its weakness may be mine, not the method's.**
   Du et al.'s composition in an **un-whitened** feature space has no exact prior identity, and the
   selection ran to the edge of the grid on all three seeds. ⛔ Do not write *"energy-based composition
   fails"*; write *"our un-whitened energy-based composition arm reaches 0.54"*. A whitened variant
   (LACE's own `fit_whitening` with independent experts) is the obvious stronger version and is
   **NOT RUN**.
3. ⚠ **The `clu` and `store_removed` rows are deterministic; the LACE and EBM rows are not.** The
   table mixes an exact number with three sampled ones, and the ± means different things in different
   rows. It is labelled here; it needs labelling wherever it is quoted.
4. ⚠ **`clu`'s B-`shape` ±0.2051** is R19's one-class degeneracy, not variance in the mechanism.
5. ⚠ **One dataset, one condition, one pretext, one decoder recipe.** Nothing here speaks to the wide
   condition or dSprites. The external classifier is Shapes3D-only by construction, and ⛔ the wide
   comparison must not be run on the shipped classifiers (`gen-external-classifier` §8b: leak-
   contaminated TNR).
6. ⚠ **`ebm_compose`'s 14.2 MB is 27× the store** and its experts see dense labels. It is a
   deliberately over-resourced rival; ⛔ do not let it be read as a byte-matched comparison.
7. ⛔ **R27.3's second route remains NOT RUN** (the general form with the prior score in `sample_ode`).

---

## Proposed handover updates (for the Hub)

- **§7 NEW (config, standing trap — the row-budget one).** A spoke that re-fits a rival must use
  **`GenBaselinesConfig.n_fit/n_val` = 32,256 / 512**, the rows the rival published from. Measured
  cost of getting it wrong: `knn_proto` 0.8763 → **0.8308** and LACE's whitening rank-deficient at
  `n_fit = 1024 < D = 2048`. `GenCellConfig` now defaults to the correct values.
- **§7 NEW (instrument, standing).** `gen-arms-a7-wiring` **did not persist the LACE classifier**, so
  every re-run of that rival re-fits it, and the re-fit alone carries **sd ≈ 0.023** on the per-sampler
  in-corner number — **larger than the published ± 0.0207 seed spread.** ⛔ A single LACE per-sampler
  number is not reproducible to better than that. **Owner: Hub** (either persist the classifier, or
  carry both spreads in every LACE quote).
- **§7 NEW (reporting, fourth strike on this family).** **Reference constants must come from the
  peer's manifest, never from a rendered 4-dp table.** Third occurrence (`gen-external-classifier`
  §11.4, `gen-arms-a7-wiring`'s `ddof`, and this). `tests/test_gen_cell.py` now **refuses a constant
  that looks like a 4-dp figure**; worth folding into the task template.
- **§7 NEW (launcher, ⛔ NOT fixed here).** `STAGE=cell` now has its module, but the launcher's
  **one-(arm, seed)-per-task** layout does not match this module's **one-manifest-per-seed** output.
  Either `ARMS=all` semantics for this stage, or form (ii) of §9's submit block. **Owner: the launcher
  spoke already scheduled for the first engineer slot after the smoke (R33).**
- **§3 config default (⛔ unchanged, still a trap).** `GenBridgeConfig.dec_recipe` **still defaults to
  `"baseline"`** — the recipe that FAILED the gate. This is the **fourth** module to carry a private
  guard against it. Either flip the default (one line + one test) or record it as permanent.
- **Result to record.** Shapes3D `recon`, Montero condition, seeds {0,1,2}, `v1_capacity_schedule`
  decoder, `gen-external-classifier`'s gated instrument — **in-corner `shape` balanced accuracy on
  IMAGES: `knn_proto` 0.9119 ± 0.0222 (475,136 B, ⚠ configuration selected at the retired 688,128 B
  budget, declared) · `lace/ld_corrected` 0.6605 ± 0.0197 · `clu` 0.6568 ± 0.0226 (524,288 B + 74,240 B
  head, 30-bit read) · `lace`(`ode`) 0.6347 ± 0.0177 · `ebm_compose` 0.5408 ± 0.0087 (14,226,660 B) ·
  `store_removed` 0.5000 ± 0.0000**; decoder ceiling **0.9357 ± 0.0193**. **Distinct factor-tuples read
  off the 72 B requests: 9.0 ± 0.8 (`clu`) · 1.0 (`store_removed`) · 54.7 ± 8.1 (`knn_proto`) ·
  71.3 (`lace`) · 71.7 (`ebm_compose`).** ⭐ The `clu` and `knn_proto` **probe** numbers are
  **bit-identical** to `gen-read-compose`'s and `gen-arms-a7-wiring`'s, 3/3 seeds each, so the
  external-vs-probe difference is **the instrument by construction**. ⭐ Sampler-noise floor **0.0204**;
  ⛔ only `knn_proto` − everything (+0.26) and `clu` − `store_removed` (+0.16) clear it.
