# w9 spoke/maze-harness — the shared instrument the maze hub scores through. ⛔ Build only.

- **Agent:** `experiment-engineer`. **Base `base/iter8 = fa60b15`.**
- **Worktree:** `/c/chlu/worktrees/CHLU-w9-harness` · **Output:** `/c/chlu/outputs/w9-harness/`
- ⭐⭐ **Critical path: `hub/maze` cannot score until you publish `HARNESS-READY.json`.** Publish it
  **partial** the moment the judge and one baseline work. ⛔ No scored run, no claim.

## DIAL DECLARATION
**Dial: the instrument.** **Falsified if** the judge needs a tolerance anywhere — an approximate
validity check makes every downstream refusal and deletion claim approximate.

## The job
`env-scout` shipped **CMAZE-1** (branch `w8/env-scout`, `w8_envscout/maze_colour.py`, `screen.py`,
`sensitivity.py`). ⛔ **Do not rebuild the environment.** Build the instrument around it:
1. ⭐ **One shared encoder**, reconstruction on maze images — **every arm consumes its features.**
2. ⭐ **One shared action head.**
3. ⭐⭐ **One EXACT judge**: legal moves · reaches target · within budget. ⛔ **Zero tolerance.**
   `env-scout` achieved this with `isclose`-free integer dynamics — keep that.
4. **Baselines at matched parameters, counts printed:** MLP · LSTM · transformer, **all on the same
   features**, plus the **table floor**.
5. ⭐ The **fragment-lookup** and **composition-blind** columns, wired as **reported columns**.
6. ⭐ The **supervised refuser** — a transformer with a refuse token trained on impossible examples.
   ⛔ It is the ONLY comparator for DISPLAY rows, and our line is *"no impossible example ever seen."*

## The readiness file
`/c/chlu/outputs/w9-harness/HARNESS-READY.json`:
```json
{"status":"ready|partial","judge_exact":false,"tolerance_used":null,
 "encoder":{"trained":false,"shared":true},"action_head":{"shared":true},
 "baselines":{"mlp":false,"lstm":false,"transformer":false,"table":false,"refuser":false},
 "param_counts":{}, "columns":{"fragment_lookup":false,"composition_blind":false},
 "seeds":[0,1,2],"utc":""}
```

## ⛔ What would prove this spoke wrong
- The judge cannot be exact in the maze ⇒ ⛔ **stop and report.** Everything downstream inherits it.
- Baselines cannot be brought to matched parameters ⇒ report the achievable counts, do not ship a
  mismatched baseline quietly.
- The shared encoder advantages one arm ⇒ that breaks rule 1; say so rather than proceeding.
