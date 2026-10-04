# w5 spoke/gym-1 — INSTRUMENT ONLY. ⛔ No claim, no scored run.

- **Agent:** `experiment-engineer`. **Base `base/iter5 = 6675904`.**
- **Worktree:** `/c/chlu/worktrees/CHLU-w5-gym1` · **Output:** `/c/chlu/outputs/w5-gym1/`
- ⛔ **You build a measuring device for wave 7 and report its properties. You make no claim about the CLU.**

## DIAL DECLARATION
**Dial: the instrument.** **Falsified if** the trajectory-lookup tripwire can solve the held-out pairs —
then the gym does not test what it is for, and the Head needs that **before** wave 7 is scoped on it.

## The one job
**One agent on a line with a speed limit. Held-out start–goal pairs. An obstacle no valid path can cross.**

Deliver, and measure each:
1. the environment + a held-out split where **paths must be new**, not recombined from seen ones;
2. ⭐⭐ **a matched-bytes TRAJECTORY-LOOKUP TRIPWIRE** — if a table of stored trajectories at the same
   budget solves the held-out pairs, the task is not testing composition. **Report its score first.**
3. ⭐ **the persistence-at-one-step rule** — a model that only ever predicts one step and is rolled out
   must still obey the speed limit and the obstacle; state how you enforce and measure it;
4. **LSTM and transformer baselines trained with the rules VISIBLE**, matched parameters, count printed;
5. the validity metrics: speed-limit violations, wall crossings, teleports, and goal reach.

## ⛔ What would prove this spoke wrong
- The tripwire solves the held-out pairs ⇒ **the gym is not an instrument. Report and stop.**
- The LSTM/transformer trained with rules visible reach ~100 % ⇒ there is no headroom to measure and
  the task needs to be harder **before** wave 7, not after.
- Validity cannot be checked exactly (a rule needs a tolerance) ⇒ say so; an approximate validity
  check makes every downstream refusal claim approximate.
