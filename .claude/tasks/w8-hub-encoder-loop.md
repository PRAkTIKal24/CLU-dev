# w8 hub/encoder-loop — TARGET: does the store force better features?

**Boot:** charter → `w8-hub-common.md` → this file. **Base `base/iter7 = ec2ef7c`.**
**Worktree:** `/c/chlu/worktrees/CHLU-w8-encoder-loop` · **LOG:** `/c/chlu/outputs/w8-encoder-loop/LOG.md`
**Budget: `encoder-loop`, 3 runs.**

## DIAL DECLARATION
**Dial: the encoder** — trained alone vs trained with the store in the loop, at **equal compute**.
**Falsified if** store-in-the-loop features do not beat encoder-alone features when **every arm is
given the same features** — then the store does not shape the representation and the claim dies.

## ⛔⛔ READ THIS FIRST — the GPU path, and what you may and may not do
**`gcloud` is authenticated on the Head's Mac (project `forgisprova`), and is NOT installed on this
box.** ⛔ **You cannot launch, poll or fetch a cloud job yourself.** The programme's runbook marks
account-level cloud actions as needing the Head's account.

⭐ **So the split is:** you build and verify everything, and **emit a ready-to-run launch spec**; the
Hub executes the authenticated step from the Mac and returns the results to
`/c/chlu/outputs/w8-encoder-loop/cloud/`.

**Measured quota (runbook, 2026-09-15, project `forgisprova`): A100-40GB SPOT limit 16, in use 0 —
⭐ free. A100-40GB on-demand is 0 free. L4 spot 8 free** as the cheap interruption-test card.
⭐ **The repo already has the tooling: `scripts/cloud/` — `launch.sh`, `ckpt.py`, `fetch_results.sh`,
`cost.sh`, `config.sh`, `build_image.sh`, a pinned `IMAGE_DIGEST`. ⛔ Read its README before writing
anything new; do not reimplement it.**

⭐⭐ **CHECKPOINT-FIRST IS NOT OPTIONAL — spot VMs are taken away.** The job checkpoints to Cloud
Storage on a wall-clock timer, verifies a config fingerprint before resuming, and syncs outputs while
it runs. ⭐⭐ **Prove resumability with `kill -9` before you emit the spec** — a standing Head rule.
⚠ **Verify the whole pipeline end-to-end on CPU at small scale first.** A spec that has only ever been
reasoned about is not ready to spend money.

## The experiment
On **BOTH objects** — the w8 single design, and **w5 `pca_tensor` as-is**:
**encoder trained ALONE vs encoder trained WITH THE STORE IN THE LOOP**, at **equal compute**, 3 seeds.
The loop's objective is the **swap loss** — swap one well between two images, decode, let the judge
score validity — **plus reconstruction**.

**Scoring.** Rows 3–4 vs the learned baselines, ⭐ **each baseline given the SAME encoder-loop
features** (w8 rule 1: same encoder for every arm; when unfrozen, each arm trains its own at equal
compute). Plus a **factor probe on the features**, reported ⭐ **as a margin over the random-init
floor, never as an absolute.**

## ⛔ What would prove this hub wrong
- Store-in-the-loop features do not beat encoder-alone at equal compute ⇒ the store does not shape the
  representation. **Report it; it is a clean answer to a claim the programme has carried a long time.**
- The gain appears only because the loop got more compute ⇒ ⛔ not a result. **Equal compute or nothing.**
- Baselines given the same features close the gap ⇒ the features are better for **everyone**, which is
  a finding about the encoder, ⛔ **not about the store**. Say which.
- The probe improves in absolute terms but not over random-init ⇒ ⛔ no claim; random init is the floor.
- ⚠ Unfreezing alone made things worse in an earlier wave. ⭐ The swap loss is the new ingredient —
  **isolate it**: report unfrozen-without-swap-loss as its own column.
