"""Sensitivity of the CMAZE-1 screening numbers to the generator knobs.

The point of this file is to show that the composition-blind score is not a
knife-edge artefact of one parameter setting.  It reports, for each setting, the
STRONGEST blind planner (the max over the blind family), which is the number that
matters -- a candidate is only as good as its best blind attacker.
"""

from __future__ import annotations

import time

import numpy as np

from w8_envscout.maze_colour import generate, solve
from w8_envscout.screen import PLANNERS, plan, score_route

BLIND = [k for k in PLANNERS if k.startswith("BLIND")]


def make_set_bounded(n, seed0, slack, require_bend, genkw, max_seeds):
    """Like screen.make_set, but gives up instead of looping forever.

    A setting can be *structurally* unable to satisfy the acceptance contract --
    braid=0.00 is the case that motivated this: a perfect maze has exactly one
    route between any two cells, so rule B can never BEND a route, only make the
    instance infeasible, which condition 2 then rejects.  That must be reported as
    an acceptance rate of 0, not hung on.
    """
    out, s, tried = [], seed0, 0
    while len(out) < n and tried < max_seeds:
        inst = generate(s, slack=slack, require_bend=require_bend, max_tries=60, **genkw)
        s += 1
        tried += 1
        if inst is not None:
            out.append(inst)
    return out, tried


def row(n_eval, seed0, slack, require_bend, genkw):
    ev, tried = make_set_bounded(n_eval, seed0, slack, require_bend, genkw, 12 * n_eval)
    if not ev:
        return None
    opt = np.array([i.opt_len for i in ev], dtype=float)
    optA = np.array([i.opt_len_a_only for i in ev], dtype=float)
    best_name, best = None, -1.0
    per = {}
    for name in BLIND:
        v = np.mean([score_route(i, plan(i, PLANNERS[name]))[1] for i in ev])
        per[name.split()[0]] = v
        if v > best:
            best, best_name = v, name.split()[0]
    return dict(accept=len(ev) / tried, opt=opt.mean(), detour=(opt - optA).mean(),
                bends=(opt > optA).mean(), best_blind=best, best_name=best_name, per=per)


def main(n_eval=150):
    t0 = time.time()
    print("CMAZE-1 sensitivity at the ADOPTED DEFAULTS "
          "(k=4, braid=0.40, adversarial=2, require_bend=False)")
    print("'blind' is the MAX over the blind planner family -- the same quantity as the headline.")
    print("%-36s %7s %7s %7s %7s %9s  %s" %
          ("setting", "accept", "opt", "detour", "bends", "blind", "which"))
    seed = 7_000_000
    settings = []                                   # (label, genkw, slack, require_bend)
    for k in (2, 3, 4, 5):
        settings.append(("n_colours=%d" % k, dict(n_colours=k), 0.10, False))
    for r in (0, 1, 2, 3):
        settings.append(("adversarial=%d%s" % (r, " (uniform)" if r == 0 else ""),
                         dict(adversarial=r), 0.10, False))
    for b in (0.12, 0.25, 0.40, 0.55):
        settings.append(("braid=%.2f" % b, dict(braid=b), 0.10, False))
    for p in (0.12, 0.22, 0.32):
        settings.append(("p_tint=%.2f" % p, dict(p_tint=p), 0.10, False))
    for sl in (0.00, 0.10, 0.25, 0.50):
        settings.append(("slack=%.2f" % sl, dict(), sl, False))
    for h in (5, 7, 9):
        settings.append(("maze=%dx%d cells" % (h, h), dict(h=h, w=h), 0.10, False))
    for mp in (2, 4, 8, 16):
        settings.append(("max_poison=%d" % mp, dict(max_poison=mp), 0.10, False))
    settings.append(("require_bend=True (filter ON)", dict(), 0.10, True))
    for label, gk, slack, rb in settings:
        seed += 100_000
        try:
            r = row(n_eval, seed, slack, rb, gk)
        except Exception as exc:                     # noqa: BLE001
            print("%-36s  FAILED: %s" % (label, exc))
            continue
        if r is None:
            print("%-36s %7.3f %7s %7s %7s %9s  %s"
                  % (label, 0.0, "-", "-", "-", "-", "NO INSTANCE SATISFIES THE CONTRACT"))
            continue
        print("%-36s %7.3f %7.1f %7.1f %7.3f %9.4f  %s"
              % (label, r["accept"], r["opt"], r["detour"], r["bends"], r["best_blind"],
                 r["best_name"].split("(")[0].strip()))
    print("total %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    import sys
    kw = {}
    for a in sys.argv[1:]:
        k, v = a.split("=")
        kw[k] = float(v) if "." in v else int(v)
    main(**kw)
