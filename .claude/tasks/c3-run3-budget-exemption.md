# c3-run3-budget-exemption — the pre-registered-continuation exemption that unblocks run 3

**Campaign 3, wave 1. Agent:** experiment-engineer. **ONE worktree (wt1).** ⛔ **SMALL AND SHARPLY BOUNDED** — this is a ~half-day task on the critical path, not a refactor.
Branch **`agent/experiment-engineer/c3-run3-budget-exemption`** off **`agent/experiment-engineer/c3-csf3-harness` @ `f98f939`** (that branch is accepted, 1781 green, HEAD-stable, and **not yet merged to `main`** — you build on it directly so run 3 is not gated on a merge; if the Hub has merged it by the time you start, branch off `main` instead and say so).
⚠ The shared checkout sits on `pilot-ttt-nan-and-d5-wiring`; **take a worktree**, cwd = the worktree.
⚠ **Reuse the main venv:** `PYTHONPATH=<worktree> /Users/user/Desktop/CHLU/.venv/bin/python …`.
Writes `.claude/outputs/c3-run3-budget-exemption.md`.

**Binding documents, read first:**
1. `.claude/outputs/c3-csf3-harness.md` **§5 IN FULL** — the ledger you are extending (`chlu/eval/byte_ledger.py`, `MATCHED_STATE_BYTE_BUDGET`, `StateByteBudgetError`), and **§5.1**, the finding that motivated this task.
2. `.claude/outputs/c2w11/PILOT-TTT-RULINGS.md` **IN FULL** — run 3's config rides these three rulings.
3. `.claude/outputs/c2w6-anti-erosion/PREREG-LeakAblation.md` — run 3 **is** this prereg: run 2 + `erosion_partition=True`, one flag.
4. `.claude/AGENT_PROTOCOL.md` §3, §4, §5, §7.

---

## ⭐ WHY THIS EXISTS — read it before you design anything

The harness correctly **refuses to train the pilot config** (CLU 5,523,456 B = 2.63× the interim 2 MiB). **That refusal is right and stays.** But **run 3 is a pre-registered one-flag ablation of run 2** whose geometry **must not change** — the budget was never meant to govern it (Head+Advisor, 2026-08-13). Changing run 3's geometry to satisfy a budget would destroy the very thing it measures.

⇒ You are building **a narrow, auditable exemption**, and the whole engineering problem is that **it must not become a loophole.** Design for the hostile reader: someone later who wants to skip a budget check will find this flag. Make that impossible rather than merely discouraged.

## ⭐ DIAL DECLARATION (protocol §7) — echo before your first result

- **Dial:** **none — instrument/plumbing.** No claim, no number a paper could quote.
- **Laundering control:** n/a. ⛔ But note the *inverse* obligation: this mechanism **weakens a guard**, so the deliverable is the **set of things it still refuses**, evidenced by tests that try to break it.
- **Falsifies the task:** any config that is not exactly run-2-plus-the-registered-flag being accepted; the exemption suppressing the ledger rather than annotating it; the unledgered-arm check being reachable through this path.
- **Does NOT falsify:** the pilot geometry still being over the interim budget. **It is, deliberately, and that is not this task's problem.**

## 1. The mechanism

Add `preregistered_continuation` to the run config. When set, it names **(a)** the run-2 journal and **(b)** the single registered flag that is permitted to differ.

**It must:**
1. **Verify via the existing `load_journal` fingerprint machinery** that the current config is **bit-identical to the run-2 journal except the registered flag.** ⛔ **Reuse that code path — do not reimplement a config comparison.** The pilot merge repaired that machinery precisely so there is one source of truth about "is this the same leg"; a second, parallel comparison is how the two drift apart.
2. **Refuse anything else, loudly**, with the differing keys printed. Specifically refuse: a second differing key · a differing key that is not the registered one · a missing/unreadable journal · a journal whose own fingerprint does not validate.
3. ⛔ **Accept exactly ONE registered flag.** No list, no wildcard, no glob, no "allow these N keys". If a future ablation needs two flags, that is a new prereg and a code change — which is the point.
4. **Annotate, never suppress.** The byte ledger is still **computed in full** and still **printed**, with the exemption stamped into it: the journal path, its **sha256**, the registered flag and its old→new value, and the arms' actual bytes and occupancy. An auditor reading the artifact alone must see *that* an exemption was taken, *why*, and *what the bytes were anyway*.
5. **Exempt the BUDGET check only.** ⛔ The unledgered-arm check (`UnledgeredArmError`), the φ-accounting assertion and the shared-shell identity assertion **remain live and must not be reachable through this path**.

## 2. The anti-loophole tests (the actual deliverable)

Tests that **try to break it** and assert it holds. At minimum:
- exemption + exactly the registered flag differing ⇒ **accepted**, ledger printed, exemption stamped;
- exemption + **a second key** differing ⇒ **refused**, both keys named;
- exemption + **a different single key** than the registered one ⇒ **refused**;
- exemption naming a **missing / corrupt** journal ⇒ **refused**;
- exemption set, arms **over budget** ⇒ accepted **but the ledger still reports true bytes and occupancy** (assert the numbers are present and correct, not zeroed or omitted);
- exemption set, an arm **unledgered** ⇒ still **refused** (the exemption does not reach that guard);
- **no** exemption + over budget ⇒ still **refused** (the original behaviour is intact).

## 3. Two rulings to record in code while you are here (one-line each, no design freedom)

- ⭐ **NO dtype normalisation** (Head+Advisor 2026-08-13). The convention is **total state bytes AS DEPLOYED**: an fp32 store pays its real 2× cost against a bf16 rival. **Harder for us is the defensible direction.** Record this in the ledger's docstring and in `BUDGET_PROVENANCE` so nobody later "fixes" it as a bug.
- ⭐ **`MATCHED_STATE_BYTE_BUDGET = 2_097_152` is INTERIM and binds nothing yet** (Head+Advisor 2026-08-13). The ceiling digit is set **in the rival-ladder prereg**, when the C3 CLU arm's store geometry is frozen — **pilot geometry is not presumed to be C3 geometry.** Rename/annotate it so its interim status is unmissable at the point of use, and ⛔ **add a guard: no rival-ladder arm may train while the budget is interim** (the ladder entry point refuses with a message naming the missing prereg). Run 3 is not a ladder arm and is unaffected.

## 4. Ownership, kills, acceptance

**Yours:** `chlu/eval/byte_ledger.py` · the run-config surface for `preregistered_continuation` · the ladder entry-point guard (§3) · new tests. **⛔ NOT yours:** `chlu/core/blocks.py` · the resume/fingerprint internals themselves (you *call* them; if they need changing, **STOP and report**) · anything in the tripwire or Track-B surface · `PilotConfig.addr_dim`'s default (**stays 8**, Hub-ruled — expose, do not flip).

**Stop conditions:** the fingerprint machinery cannot express "identical except key K" without modification → **STOP and report** with the specific obstacle; it is a Hub call whether to change that code. Any temptation to widen the exemption to make a test pass → **STOP**; the narrowness is the feature.

**Acceptance (one line):** run 3's config (run 2 + `erosion_partition=True`, geometry unchanged) trains under the exemption with a full, stamped byte ledger on disk; **every §2 break-attempt is refused as specified**; the interim-budget guard blocks ladder arms; the no-normalisation ruling is recorded in code; the full suite is green with counts reported against a **HEAD you name and re-verify at the end**; branch ref verified from the main repo before the worktree is removed.

**Report:** protocol §5 + flag-provenance + dial declaration + git footprint, and a short **"what this still refuses"** table — that table is what the Advisor will read.
