# Task: camels-caravan-refs — freeze the reference set for the ungauged (PUB) venue, and decide whether Caravan is a venue at all

- **Agent:** `web-scout` · **Output:** `.claude/outputs/camels-caravan-refs.md`
- **Read first:** protocol **§5 (pre-registration + flag-provenance)**. Nothing else from the archive.
- **Repo:** read-only everywhere. **Zero worktrees, zero branches, zero tracked-code edits.** No runs, no training, no tripwire numbers.
- **Wave:** ICLR-long W1, spoke 1 of 3. **This spoke gates the other two.** Both of them key their precondition on the frozen section you write.

## The one job
Produce the **frozen reference set** the CAMELS-US ungauged/PUB tripwire will be judged against, and answer one binary question about Caravan: **does a recognised benchmark protocol with published baseline numbers exist for it, yes or no.**

⛔ **No CLU arm, no CLU number, no store number of any kind.** You are pricing the *competition*, not us.

## Why this is not a formality
CAMELS-US **gauged** already cleared the tripwire by a wide margin — the best in-budget classical store hit median NSE 0.256 against an LSTM-ensemble 0.758. **That verdict does not transfer.** The ungauged task's classical competition is **donor-basin regionalization** — a similarity lookup over static catchment attributes. That is structurally the same object as our kNN-over-static-attributes store arm, i.e. **the arm most likely to fire**. Assume the tripwire can fire until the numbers say otherwise.

## Items

### A. CAMELS-US PUB — the protocol
1. The **canonical k-fold ungauged protocol**: fold count, how basins are assigned to folds, the exact basin set (is it the 531 set, the 447 set, or another?), the train/test **period**, and the **metric convention** (NSE definition, per-basin then median vs pooled, how NSE ≤ 0 basins are handled, any clipping).
2. Name the **paper of record** for that protocol and quote it, do not paraphrase it.
3. State whether the fold assignment itself is **published as a file**. If it is not, say so plainly — an unpublished fold assignment means no like-for-like comparison is possible and the venue is not auditable.

### B. CAMELS-US PUB — the CURRENT frontier
4. Search **2019 → 2026** for the **best published median NSE on the PUB protocol**. ⚠ **Do not stop at the 2019 number.** Actively hunt: multi-forcing LSTMs, mass-conserving / hybrid-conceptual variants, ensembles, transformer and foundation-model hydrology work, regional and continental-scale models.
5. Report **every** credible reference as a row, not just the winner — the kill rule is evaluated against **every** reference, so a weaker-but-credible reference that our store gets close to still fires the tripwire.
6. For each row: median NSE, basin set, fold protocol, period, metric convention, citation, and **whether per-basin predictions are published and downloadable**.

### C. Reproducibility and re-train pricing
7. For each reference: **can we reproduce it without training** (per-basin predictions or metrics shipped), or must it be **re-trained**?
8. Already on disk and relevant: `.claude/scratch/c3-trackb-tripwire/ref/all_metrics.p` (per-basin NSE for the **gauged** HESS-2019 benchmark suite) and `ref/basin_list_531.txt`. Determine whether the PUB references use that same 531 basin set, and whether an equivalent per-basin artifact exists for the **ungauged** experiment.
9. If a re-train is required: **price it** from the paper's own reported setup — parameter count, epochs, seeds in the published ensemble, reported hardware and wall-clock. This number is consumed directly by the cost spoke.

### D. Caravan — the binary question
10. What it **actually is**: basin count, regions covered, forcing product, static attribute set, period, licence, download size.
11. ⭐ **The decisive item: is there a recognised benchmark protocol — a fixed split, a defined metric, and published leaderboard numbers?** If there is no established protocol with published baselines, **say so plainly and prominently.** That single fact decides whether Caravan can carry a SOTA claim at all, and a hedged answer here is worse than useless.
12. If a protocol does exist: same treatment as items A–C for it.

## What you must write, in this exact form
The output **must** contain a section headed literally:

```
## REFERENCE SET (FROZEN)
```

containing one machine-readable table row per reference, with columns:
`venue | reference_name | median_NSE | basin_set | folds | period | metric_convention | per_basin_available (yes/no + path or URL) | reproducible_without_training (yes/no) | citation`

and immediately after it a line reading literally one of:

```
caravan_protocol: USABLE
caravan_protocol: NONE
```

Downstream spokes key their precondition on that heading and that line. Get the strings exactly right.

## Acceptance
- Every number traces to a **primary source** — the paper's own table, ideally its machine-readable supplement — downloaded to `.claude/scratch/camels-caravan-refs/` with its URL and a checksum recorded. ⛔ No number sourced only from a blog post, a secondary summary, or a leaderboard site without the underlying paper.
- The `REFERENCE SET (FROZEN)` section and the `caravan_protocol:` line are present and exactly formatted.
- The 2019→2026 frontier search is documented as a search — queries tried, what was found, **what was looked for and not found**. A frontier claim with no visible search behind it is not accepted.
- **NOT-FOUND is reported as NOT-FOUND, never as absent and never as a zero.**

## What would prove this spoke's answer wrong
- **A PUB reference exists above the number you report as the frontier.** This is the failure mode that matters: an under-reported frontier makes the tripwire look safe when it is not. Bias your search toward finding a *higher* number.
- **The fold assignment is not published**, in which case every margin computed downstream is against a reconstructed protocol, not the published one — and you must say so, because it changes what the paper is allowed to claim.
- **Caravan turns out to have a protocol you missed**, or conversely a "leaderboard" you report turns out to be a dataset-paper table with no held-out convention.

## Constraints
- ⛔ No sub-agents. No runs. No training. No CLU numbers.
- ≥ 3 independent retrievals before any number is called confirmed; a single retrieval is marked provisional.
- Under ~4 pages. Tables over prose.
