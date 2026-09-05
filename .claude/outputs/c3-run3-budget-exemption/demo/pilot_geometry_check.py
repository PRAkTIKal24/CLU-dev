"""Demo 3 — the REAL run-3 case: the REAL banked run-2 journal, PILOT geometry.

⛔ No training. This is the config arithmetic the actual run-3 submission will hit:
verify the exemption against the real `csf3_outs/run2/pilot_pilot_seed<N>_PARTIAL.json`,
then build the byte ledger at the unchanged pilot geometry and show it is
computed in full, stamped, and NOT suppressed.
"""
import json
import sys
from pathlib import Path

import jax

import chlu.experiments.exp_cluformer_pilot as EXP
from chlu.eval.byte_ledger import build_byte_ledger, format_ledger_summary
from chlu.training.train_cluformer import solve_arms

RUN2 = Path("/Users/user/Desktop/CHLU/.claude/outputs/cluformer-pilot/csf3_outs/run2")
PREREG = ".claude/outputs/c2w6-anti-erosion/PREREG-LeakAblation.md"
OUT = Path("/Users/user/Desktop/CHLU/.claude/outputs/c3-run3-budget-exemption")
OUT.mkdir(parents=True, exist_ok=True)


def cfg_from_journal(jflags, **extra_memory):
    """Rebuild the leg's config from its own flag table (this is what the run-3
    launch line must reproduce)."""
    ov = dict(jflags["pilot"])
    ov["memory"] = dict(ov.get("memory", {}), **extra_memory)
    ov["store"] = dict(ov.get("store", {}))
    return EXP.make_config("pilot", int(jflags.get("seed", 0)), ov)


report = {}
for seed in (0, 1, 2):
    jpath = RUN2 / f"pilot_pilot_seed{seed}_PARTIAL.json"
    jflags = json.loads(jpath.read_text())["flags"]
    p3 = cfg_from_journal(jflags, erosion_partition=True)
    p3.seed = seed
    ex = EXP.verify_preregistered_continuation(
        {"journal": str(jpath), "flag": "memory.erosion_partition",
         "prereg": PREREG}, p3)
    print(f"\n=== seed {seed}: ACCEPTED ===")
    print(json.dumps(ex.as_stamp(), indent=1)[:1200])
    report[f"seed{seed}"] = ex.as_stamp()
    if seed == 0:
        _, led = solve_arms(p3, jax.random.PRNGKey(0))
        art = build_byte_ledger(p3, led, sorted(p3.arms),
                                budget=int(p3.state_byte_budget),
                                enforce=True, exemption=ex)
        print(format_ledger_summary(art))
        report["ledger_seed0"] = {
            "budget_bytes": art["budget_bytes"],
            "budget_is_interim": art["budget_is_interim"],
            "budget_ceiling_prereg": art["budget_ceiling_prereg"],
            "dtype_normalisation": art["dtype_normalisation"],
            "enforced": art["enforced"], "budget_exempted": art["budget_exempted"],
            "over_budget": art["over_budget"],
            "arms": {a: {k: r[k] for k in ("total_state_bytes", "occupancy",
                                           "within_budget", "phi_accounted",
                                           "cell_state_bytes_per_layer")}
                     for a, r in art["arms"].items()},
            "preregistered_continuation": art["preregistered_continuation"],
        }

# ⛔ and the refusals, against the SAME REAL journal
jpath = RUN2 / "pilot_pilot_seed0_PARTIAL.json"
jflags = json.loads(jpath.read_text())["flags"]
refusals = {}
cases = {
    "second_key": cfg_from_journal(jflags, erosion_partition=True,
                                   refresh_monotonic=True),
    "no_registered_change": cfg_from_journal(jflags),
    "geometry_shrunk_to_fit": None,
}
shrunk = cfg_from_journal(jflags, erosion_partition=True)
shrunk.atoms_per_item = 64          # "just shrink the store until it fits"
cases["geometry_shrunk_to_fit"] = shrunk
for name, cfg in cases.items():
    try:
        EXP.verify_preregistered_continuation(
            {"journal": str(jpath), "flag": "memory.erosion_partition",
             "prereg": PREREG}, cfg)
        refusals[name] = "⛔⛔ ACCEPTED — THIS IS A BUG"
    except Exception as e:                      # noqa: BLE001
        refusals[name] = str(e)
    print(f"\n=== refusal case {name} ===\n{refusals[name][:900]}")
report["refusals_against_the_real_journal"] = refusals

(OUT / "pilot-geometry-exemption.json").write_text(json.dumps(report, indent=1))
print("\nwrote", OUT / "pilot-geometry-exemption.json")
sys.exit(0)
