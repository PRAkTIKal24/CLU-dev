# Task: gen-montero-splits-spec — pin the Montero recombination-to-element splits from primary

- **Agent:** `web-scout` · **Output:** `.claude/outputs/gen-montero-splits-spec.md` + **`.claude/outputs/gen-montero-splits-spec/montero_splits.json`**
- **Read first:** `.claude/AGENT_PROTOCOL.md` **§5 and §6** only. ⛔ Do NOT read the campaign ledgers, charters or handovers — they are a different programme and will mislead you.
- **Repo:** read-only. **Zero worktrees, zero branches, zero tracked-code edits.**
- **Precondition:** none. Spawnable now.

## ⭐ DIAL DECLARATION (protocol §7)
- **Dial:** *none — instrument / spec recon.* ⛔ No CLU number, no architecture opinion, no performance claim.
- **Laundering control:** N/A (no measurement).
- **Falsifies:** N/A. The failure mode here is a *silently reconstructed* split, not a wrong number.

## The one job
Pin, **from the primary source**, the exact **recombination-to-element** split definition used by Montero et al.,
*The Role of Disentanglement in Generalisation* (ICLR 2021), for **Shapes3D** and **dSprites** — in enough detail
that an engineer can generate the identical train/test partition without reading anything else.

## Why this exists
The whole phase's B cell ("generate a combination never seen in training") is only meaningful if the held-out
set is *the published one*. A reconstructed-but-different split is a different task and silently voids every
comparison to the literature. ⭐ **A split we invented is worth less than an honest NOT OBTAINED.**

## What the JSON must contain (this file is the mechanical precondition for the data spoke)
Per dataset (`shapes3d`, `dsprites`):
- `factors`: ordered list of factor names with their **cardinalities and value grids**, exactly as the dataset ships them.
- `condition`: the recombination-to-element rule, transcribed **verbatim** from the paper (quote + section/table number),
  plus a machine-readable form: which factor(s) are held out, at which value(s), and whether the hold-out is over
  **factor values** or over **factor-value combinations**.
- `expected_counts`: `n_train`, `n_test` implied by the rule, computed by you from the factor grid — and, if the paper
  reports counts, **their** numbers alongside yours with the discrepancy stated.
- `randomised`: whether any seed/sampling enters the split, and if so its published value.
- `source`: paper id, version, section/table, and — if an official implementation exists — repo URL, **commit hash**, and
  the path to the file that actually constructs the split. ⭐ **If the official split code exists, it is the primary source,
  not the prose.** Record its sha256.
- `status`: `PINNED` | `RECONSTRUCTED` | `NOT OBTAINED` — per dataset, per field.

Also record (secondary, for later phases, not the acceptance criterion): the **recombination-to-range** and
**extrapolation** conditions, same shape, marked as not-yet-needed.

## Dataset provenance (needed by the data spoke, cheap to collect while you are here)
For Shapes3D and dSprites: canonical download URL, file name, **byte size, and published checksum if one exists**,
licence, and the canonical citation. ⛔ Do not download anything — record the pointers only.

## Acceptance
- `montero_splits.json` exists and validates against the fields above, with a `status` on **every** field.
- The recombination-to-element rule appears as a **verbatim quote** with a locator, not a paraphrase.
- Your computed `expected_counts` are shown; if they disagree with any published count, the disagreement is stated
  plainly and **not resolved by adjusting your reading to match**.
- Any field you could not obtain from primary reads `NOT OBTAINED` with one line on what you tried.

## What would prove this spoke wrong
- You paraphrased a rule that the paper only defines in code, and the code says something else. **Check for the repo
  before you trust the prose.**
- You assumed dSprites and Shapes3D share a hold-out rule. Verify per dataset; do not carry one across.
