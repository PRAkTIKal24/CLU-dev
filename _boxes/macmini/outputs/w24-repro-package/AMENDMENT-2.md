# AMENDMENT 2 — you were stranded for about 35 minutes; two smokes fail (2026-09-26 ~00:45Z)
1. **Your two completion waiters could never exit.** `until ! pgrep -f "proof_final.sh"` matches the waiter's own command line, because that line contains the string. `proof_final.sh` had ended at 00:09Z. The Hub killed the two waiter shells (PIDs 79198 and 80089) to wake you. Nothing else was touched.
   - From now on, wait on a PID captured at launch (`while kill -0 $PID; do sleep 20; done`) or on a marker file the script writes last.
2. **Proof C smoke has two non-zero exits: `sweep_k_report rc=1` (0.1 s) and `coverage_control_report rc=1` (7.1 s).** Fix them and re-smoke those two in the fresh venv, or declare them in the report with the reason. ⛔ An entry point that exits 1 is not "smoked".
3. The Shapes and CMAZE-T numbers in `proof/repro.log` look complete. Finish the scan, the MANIFEST, the report and END_UTC.
