#!/usr/bin/env python
"""w13 patch 5/5 -- NR7 cited `/c/chlu/outputs/spoke-a-rivals` as a bare directory, which is
not a registerable source, so verify_tables.py rejected it with MD PATH NOT IN REGISTER.
Register the spoke's own readiness manifest instead and cite the FIELD that records which
seeds have fitted rivals.  Exact-string patch."""
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "build_tables.py"
t = SRC.read_text(encoding="utf-8")


def sub(anchor, replacement):
    global t
    n = t.count(anchor)
    assert n == 1, "anchor occurs %d times, expected 1: %r" % (n, anchor[:110])
    t = t.replace(anchor, replacement)


sub('''    # the plan this fold is executed against
    "wave_plan_12": "/c/chlu/evidence/outputs/advisor-wave-plan-w12-w14.md",''',
    '''    # which seeds actually have FITTED RIVALS - the field NR7 turns on
    "rivals_ready": "/c/chlu/outputs/spoke-a-rivals/RIVALS-READY.json",
    # the plan this fold is executed against
    "wave_plan_12": "/c/chlu/evidence/outputs/advisor-wave-plan-w12-w14.md",''')

sub('''                "does not exist, and `/c/chlu/outputs/spoke-a-rivals` holds `seed_0`, `seed_1` and "
                "`seed_2` only, so the rivals are missing too. `Judge(3)` raises FileNotFoundError. "''',
    '''                "does not exist, and spoke A's own readiness manifest records "
                "`rivals_seeds_present` = [0, 1, 2] for every one of its seven rival families, so the "
                "rivals are missing for seeds 3 and 4 too. `Judge(3)` raises FileNotFoundError. "''')

sub('''                     {"source": "x_findings", "path": SOURCES["x_findings"], "locator": "FINDINGS.md section 4"}],
        "launch_line": (
            "# the missing pieces are the JUDGE's, not the cache's: for SEED in 3 4\\n"''',
    '''                     {"source": "x_findings", "path": SOURCES["x_findings"], "locator": "FINDINGS.md section 4"},
                     {"source": "rivals_ready", "path": SOURCES["rivals_ready"],
                      "pointer": "/rivals_seeds_present/mlp"},
                     {"source": "rivals_ready", "path": SOURCES["rivals_ready"], "pointer": "/seeds_complete"}],
        "launch_line": (
            "# the missing pieces are the JUDGE's, not the cache's: for SEED in 3 4\\n"''')

sub('''    "why_5_seed_rows_are_impossible_anyway": (
        "no frozen decoder (`decoder_v1_s3.pkl` absent), no external classifier "
        "(`fw.classifier_dir(3)` and `(4)` absent) and no fitted rivals "
        "(`/c/chlu/outputs/spoke-a-rivals` holds seed_0, seed_1 and seed_2 only) exist for either seed, "''',
    '''    "why_5_seed_rows_are_impossible_anyway": (
        "no frozen decoder (`decoder_v1_s3.pkl` absent), no external classifier "
        "(`fw.classifier_dir(3)` and `(4)` absent) and no fitted rivals "
        "(`/c/chlu/outputs/spoke-a-rivals/RIVALS-READY.json` records `rivals_seeds_present` = [0, 1, 2] "
        "for all seven families) exist for either seed, "''')

# and a completeness gate on the file we now depend on
sub('''        {"artefact": "`BASELINES-READY.json`", "check": "records present on every seed",''',
    '''        {"artefact": "`RIVALS-READY.json`", "check": "which seeds have FITTED rivals",
         "expected": "status ready, a seed list per family",
         "found": "status %r, seeds_complete %r, rivals_seeds_present[mlp] %r" % (
             DOCS["rivals_ready"]["status"], DOCS["rivals_ready"]["seeds_complete"],
             DOCS["rivals_ready"]["rivals_seeds_present"]["mlp"]),
         "verdict": "COMPLETE - and it names 3 seeds, not 5"},
        {"artefact": "`BASELINES-READY.json`", "check": "records present on every seed",''')

SRC.write_text(t, encoding="utf-8")
print("p5 OK ->", SRC)
