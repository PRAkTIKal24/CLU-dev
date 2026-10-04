# VENDORED, UNMODIFIED — spoke A's wave-4 rival API

`w4_api.py`, `w4_cache2.py`, `w4_common.py`, `w4_deletion.py`, `w4_learners.py`,
`w4_population.py`, `w4_report.py`, `w4_rows.py`, `w4_specialists.py` in this directory are
**copied byte-for-byte from `hub/spoke-a-rivals`' worktree** and are not this hub's work.

They are here because the full rival set (w4-common §1) must be run through **this hub's**
corners by **this hub's** judge, and `chlu.eval.rivals.w4_api` cannot be imported from another
worktree without shadowing this worktree's `chlu` package. ⛔ No line of them is edited by
hub/read; a hub that edits a rival voids its numbers (charter §A.4).

Provenance, recorded at copy time and re-recorded before every scored run:
* source worktree: `/c/chlu/worktrees/CHLU-w4-spoke-a` (branch HEAD `9dfed5e`, files uncommitted there)
* sha256 of every copied file: `/c/chlu/outputs/hub-read/vendor/SHA256SUMS.txt`
* the fitted rival parameters themselves are NOT copied: they are read live from
  `/c/chlu/outputs/spoke-a-rivals/rivals/seed_<s>/<name>.rival.json`, spoke A's own artifacts.
