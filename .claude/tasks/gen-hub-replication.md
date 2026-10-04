# Task: gen-hub-replication — ⭐ unreplicated is NOT a result. Re-run a hub's winner from its PREREG, independently.

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-hub-replication-<hub-slug>.md`
- ⭐ **INVOKED ONCE PER HUB.** The Head names the hub; `<hub-slug>` ∈ `hub-generation-table` · `hub-composition` · `hub-refusal` · `hub-deletion`.
- **Read first:** `.claude/AGENT_PROTOCOL.md` · `.claude/autonomous-hub-charter.md` **§A.5 (this task IS §A.5)** · the hub's `PREREG-<hub>-<n>.md` for its winning run
- ⛔ **DO NOT READ THE HUB'S REPORT OR ITS `LOG.md` BEFORE YOU RUN.** See "The blindness rule" below.
- **Base:** ⭐ a **clean checkout of the HUB's branch** `hub/<target>` · **Branch:** `agent/experiment-engineer/replication-<hub-slug>` · **Worktree:** `../CHLU-repl-<hub-slug>`
- **Env:** main venv. ⚠ **Budget ≈ 0.5 day per hub.** ⛔ No sub-agents. ⛔ No number of your own enters any table.

## ⭐ DIAL DECLARATION
- **Dial:** ⛔ **none — independent replication.** ⛔ No new arm, no new mechanism, no verdict on any claim.
- **Laundering control:** ⭐ **the blindness rule** — you reproduce from the PREREG, ⛔ never toward a known target.
- **Falsifies:** ⭐ **the hub's headline number.** That is the entire point of your existence.

## ⛔⛔ THE BLINDNESS RULE — read before anything else
⛔ **You do not read the hub's report, its `LOG.md`, or its result files until your own run has finished and its
number is written to disk.** ⭐ **You reproduce from the PREREG file alone.**
⚠ **Why:** a replicator who knows the answer reproduces the answer. If you know the hub got 0.87, every ambiguity
in the PREREG resolves toward 0.87 without you noticing. ⛔ **That is not replication, it is confirmation, and it
would make §A.5 worthless.**
⭐ **Write your number down FIRST. Then open the hub's report and compare.** Both numbers go in your report,
in that order, with the timestamps that prove the order.

## The one job
**Re-run the hub's winning configuration from its PREREG file, on a clean checkout, and say whether it reproduces.**

## Scope
1. ⭐ **Clean checkout of the hub's branch.** ⛔ A fresh worktree, ⛔ never the hub's own — its worktree carries
   uncommitted state, caches and half-finished runs, and reproducing inside it proves nothing.
2. ⭐⭐ **RECONSTRUCT FROM THE PREREG, NOT FROM THE CODE.** Configure the run using only what the PREREG states.
   ⛔⛔ **If the PREREG is insufficient to reproduce the run, STOP AND REPORT THAT — it is a finding, and a serious
   one.** ⭐ A result whose own pre-registration cannot regenerate it is not reproducible, and no amount of digging
   in the hub's scratch files repairs that. **Name every field you had to guess.**
3. **Same seeds, same split, same frozen harness at the pin** `ae907466ba2b5fbec124e8995cf4d7c32a1ab30e`.
   ⛔ You do not edit the classifier, the splits, the rival rows, the byte ledger or the bypass assertions.
4. ⭐ **Assert the byte ledger and the decoder** yourself — ⛔ do not inherit the hub's word for either.
   ⚠ `dec_recipe` defaults to `"baseline"`, the recipe that FAILED the gate; the winner is `v1_capacity_schedule`
   and its hash must be asserted. ⛔ A baseline run **looks normal in the logs and is void.**
5. **Report the comparison:** your number, the hub's number, Δ, and whether Δ sits within the seed spread the
   PREREG declared. ⭐ **State REPLICATED or NOT REPLICATED in those words.**

## ⛔ What NOT to do
- ⛔ **Do not fix the hub's code**, even if you find a bug. ⭐ **Report it.** A replication that silently repairs the
  thing it is testing has measured nothing.
- ⛔ **Do not tune, re-seed, extend, or "give it a fair chance."** You run what the PREREG says, once.
- ⛔ **Do not report a partial re-run as a replication.** Incomplete = **NOT RUN**, ⛔ never a pass and ⛔ never a fail.
- ⛔ **No GPU** unless the hub's winning run itself required one and the Head has approved it — ⭐ CPU by default,
  GCloud CPU or CSF3 `multicore` under the **`fatpou01`** account.

## Files you own — and nothing else
- `.claude/outputs/gen-hub-replication-<hub-slug>/**` and your report.
⛔ **You own NO source file.** ⭐ You run the hub's code as committed. ⛔ Not `chlu/**`, ⛔ not `chlu/config.py`,
⛔ not the hub's branch (you check it out read-only into your own worktree; you do not commit to it).

## Acceptance
1. ⭐ **Your number on disk, timestamped, BEFORE you opened the hub's report.** Show both timestamps.
2. The run reconstructed **from the PREREG alone**, with every guessed field named — or a stop-and-report that the
   PREREG was insufficient.
3. Byte ledger and decoder hash asserted independently.
4. ⭐ **REPLICATED / NOT REPLICATED**, with Δ against the hub's declared seed spread.
5. ⛔ Every bug found is **reported, not fixed**.
6. A **cost line**. ⛔ No full suite.

## What would prove this spoke wrong
- ⭐⭐ **You read the hub's result first.** Then you confirmed, you did not replicate, and §A.5 bought nothing.
- ⭐ **You reconstructed from the hub's code or scratch files** rather than its PREREG — so an unreproducible
  pre-registration passes as reproducible.
- ⭐ **You fixed something to make it run.** The question is whether it reproduces as written, not whether it can.
- You reported a partial run as a verdict in either direction.
