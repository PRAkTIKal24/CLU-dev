# w12 hub/composition-ladder — how far does composition stretch as k grows?

- **Agent:** autonomous hub. **Target:** replications (**free**), but ⭐ the numbers enter the paper.
- **Output:** `/c/chlu/outputs/w12-composition-ladder/`. **Branch:** `w12/composition-ladder`.
- **GATED — mechanical:** ⛔ do not start until
  `/c/chlu/outputs/w12-kway-ladder-harness/LADDER-READY.json` exists. ⛔ Do not build your own
  k-way families — if the harness is wrong, **report that**; do not route around it.
- ⭐ **5 seeds IF** `/c/chlu/outputs/w12-cache-gen2-seeds-3-4/CACHE-READY.json` exists when you start;
  ⛔ otherwise **3 seeds, DECLARED**. ⛔ Never mix 3-seed and 5-seed rows in one plot without a
  printed `n`.

## THE ONE QUESTION
**Do all arms deteriorate from their 2-way result as k grows, and is there a breaking point on
Shapes?** ⭐ Report the breaking point **per arm** if one exists.

## THE ARMS — all of them, every k
- **single design at K\*** (from `hub/single-design-crossover`; if K\* is not published when you
  start, ⭐ **use K=5 and DECLARE it**, then say so on every row)
- ⭐ **the wave-5 subspace store (`pca_tensor`) — ALWAYS an arm.** This is the Head's *"re-run with the
  old w5 store if n-way fails"*, built in from the start rather than added after a disappointment.
- **MLP · LSTM · transformer · continuous MLP · table floor**
- **k = 2, 3, 4, 5, 6.** ⭐ **Both byte matchings** (1× single-design total, and 524,288 B reference).

## ⛔ THE CONTROL WAVE 9 PROVED YOU CANNOT OMIT
On the store arms, at every k, also run **`landscape_shuffled`** and **`random_atoms`**.
⛔ **If they do not collapse, the store is DECORATIVE at that k and no row is claimed for it** —
report it as the result. ⚠ Wave 9's maze arm looked fine and was decorative; only these controls
caught it. ⭐ Print **rank/width** on every store row.

## THE FIGURE (follow the dataviz rules)
⭐ **x = number of jointly composed factors · y = score · one line per arm · seed band.**
⭐ **Second panel = paired margin vs the BEST learned baseline** — ⛔ not vs the mean, and ⛔ not vs a
baseline chosen after seeing the result.

## ⛔ NON-NEGOTIABLE
- ⭐ **k=2 must reproduce row 3 and k=6 must reproduce row 4.** ⛔ If they do not, the ladder is
  off-instrument — **stop and report that**, do not plot it.
- ⭐ The paired-corner estimand and the bar are **the harness's**, pre-declared. ⛔ Do not restate them.
- ⭐ **All seeds or none.** ⛔ The best is never reported alone.
- ⛔ **Never "9/9 seeds"** — report seed count and configuration count separately.

## ⛔ WHAT WOULD PROVE THIS HUB WRONG
- ⭐ **Every arm collapses at the same k** ⇒ that is a property of **Shapes3D**, ⛔ not of any arm, and
  it must be reported in those words. It is a finding about the dataset's composition ceiling.
- ⭐ **`pca_tensor` matches or beats the single design at every k** ⇒ report it plainly. The single
  design's justification is deletion, not composition, and this would say so.
- **No arm deteriorates at all** ⇒ the ladder does not discriminate on this dataset; ⭐ say so rather
  than reporting a flat line as a result.
