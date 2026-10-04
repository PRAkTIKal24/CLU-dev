# Task: w3-value-holdout-amendment — re-pair row 5's accept-leg to row 1 (Head ruling, option 1)

- **Agent:** `experiment-engineer` — ⭐ **the same one that built the harness**, same branch
  `agent/experiment-engineer/w3-value-holdout`. ⛔ Not a fresh spoke.
- **Output:** append a dated `## AMENDMENT 1` section to `.claude/outputs/w3-value-holdout-harness.md`.
  ⛔ Do not rewrite the existing report — your §0 finding is correct and stands as written.
- **Tests:** the Head's standing instruction for this spoke is **skip tests**, ⭐ **except the three
  named in §4**, which are the acceptance itself and are not optional.

## DIAL DECLARATION
**Dial: the instrument.** No arm, no scored run. **Falsified if** a no-model rule still wins row 5's
pair under the new pairing — then option 1 does not work and you report that instead of patching it.

---

## 1. Why — the ruling, in one paragraph

Your §0 is right and the Hub re-derived it independently (1.0000 / 1.0000 / 1.0000, zero bytes).
⭐ **The cause is the pairing, not the query sets.** Row 5's accept-leg was row 2, and row 2 is
**off-grid by definition** while row 5 is **on-grid by definition**, so a grid look-up separates them
perfectly. Worse, on the hue axis the two rows are the *same request*: row 2 asks for 0.35 (unstored,
bracketed by stored 0.3 and 0.4) and says accept; row 5 asks for 0.5 (unstored, bracketed by stored 0.4
and 0.6) and says refuse. ⛔ **No store can satisfy both**, which is exactly why only the grid separates
them.

**The Head has ruled option 1: row 5's accept-leg becomes ROW 1 — a stored, on-grid value.** Then both
sides of the pair are well-formed on-grid requests that look identical to any syntactic rule, and differ
only in **whether the store holds the value**. That is the claim, and it is coherent on both axes:
render what you hold, refuse what you do not.

## 2. What changes — four things, and nothing else

1. ⭐ **New query set `row1_seen.json`, 512 records**, the accept-leg for row 5. On-grid requests whose
   **every** value is stored — drawn from **train** rows. ⭐ **Structurally matched to `row5_held_out`
   as closely as the content allows**: same term count, same factor order convention, same id format,
   **the same record-key set**. ⚠ Row 5 carries answer keys (`violated_factor`, `row`); give row 1 the
   analogous keys so `record_key_count` does not become a new giveaway. Add a `validate_row1` that
   asserts each request is on-grid, denotes its own grid row, and that **the row is in train** — with a
   positive control that injects a test row and requires it to raise.
2. ⭐ **`score_refusal_row`: row 5's `accept_unseen_valid` is measured on ROW 1's requests**, with the
   same threshold / setting fingerprint / deployed bytes check you already enforce. ⛔ Row 2 is no
   longer row 5's accept-leg.
3. ⭐ **Row 2 becomes an accept-only row** — `hub/unseen-valid`'s target. It keeps its own scoring and
   is no longer consumed by any refusal row.
4. ⭐ **Row 6 is demoted to a declared sanity check.** Out-of-range is decidable from the grid alone —
   the pin already says so — so it can **never** be non-free. Keep its pair as the degenerate-arm guard,
   keep `grid_syntax_rule` beside it, and mark it in the artefact with a machine-readable
   **`evidential: false`** plus the reason. ⛔ `build_scoreboard` must refuse to emit a row-6 verdict
   phrased as evidence for the store.

⛔ **Everything else stands:** the split index, the cells, row 2, row 5's records, the scope statement,
the byte ledger, row 7, and `grid_syntax_rule` as a mandatory row on all seven.

## 3. ⭐ Row 7 is now the programme's strongest instrument — say so in the artefact
Deleting a value from the store **does not move the factor grid**, so `grid_syntax_rule`'s
`delta_reject` is exactly 0.0. An arm that changes its refusal after a deletion, with no retraining, has
done something no syntactic rule can do. ⭐ Record that ranking in the artefact so no hub has to
rediscover it: **row 7 evidential, row 5 evidential under the new pairing, row 6 sanity only.**

## 4. Acceptance — three checks, and they are the point of the task
1. ⭐⭐ **`grid_syntax_rule` must now FAIL on row 5.** Re-score it under the new pairing and report the
   pair. Expected ≈ **0** (it rejects all-on-grid row 1, so its accept-leg collapses). ⛔ If it still
   scores well, **option 1 has not worked — report that and stop.**
2. ⭐⭐ **Re-run `assert_no_model_detector_at_chance` with `row1_seen` as the control for both row-5
   kinds.** Report the full per-feature structural table as before. It must pass at the pin's tolerance.
3. ⭐⭐ **Make an honest attempt to break it yourself.** You found the first leak by trying; try again
   against the new pairing and **report what you tried and what it scored, whether or not it worked.**
   ⚠ A "passes" with no attempt behind it is worth much less than the first finding was.

⚠ Also re-run the four pre-existing split/detector test files, since the registry gains a query set
again — that inheritance mechanism is what caught nothing last time only because there was nothing to
catch.

## 5. Files
- **You own:** `chlu/data/gen_value_holdout.py` · `chlu/experiments/gen_scoreboard.py` and their two
  test files — **all four already yours on this branch.**
- ⛔ **Still read-only, still voids every hub's numbers:** `exp_gen_cell.py` · `gen_splits.py` ·
  `gen_rowsplit.py` · `gen_store.py` · `gen_landscape.py` · `gen_bridge.py` · `config.py`.
- ⛔ Re-emit the condition deterministically; the existing six artefact hashes must be **unchanged**
  except where a file genuinely gains content. **Report which hashes moved and why.**

## 6. Precondition — mechanical
`git rev-parse --verify agent/experiment-engineer/w3-value-holdout` resolves, and
`.claude/data/gen/shapes3d/value_holdout/row5_held_out.json` exists with sha256
`e0b6ea401fb675c550b4442da05467b2a3958870e65c2ca23b78e0e975304d82`.

## 7. ⛔ What would prove this amendment wrong
- A no-model rule wins row 5's pair under the new pairing ⇒ option 1 fails. **Report and stop.**
- Row 1 cannot be structurally matched to row 5 without contrivance ⇒ say so; a contrived control is a
  new leak.
- Making row 1 the accept-leg turns row 5 into a test the **basis+head control also passes trivially**
  (it never saw the held-out value either) ⇒ ⚠ that is not a defect, it is the comparison — but say it
  plainly in the artefact so no hub mistakes a shared property for a CLU win.
