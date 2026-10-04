#!/usr/bin/env python
"""w13 patch 6/6 -- three rendering defects found by reading the rendered TABLES.md:
 (1) a 1-seed cell printed as `mean (value)`, e.g. `516,096 (516,096)`;
 (2) the byte-ledger header hard-coded `part (seed 0 / seed 1 / seed 2)` even on a
     1-seed table;
 (3) a duplicated `seeds (n)` column, because the row renderer prints the seed count AND
     several tables also carried a traced cell of the same name.
Exact-string patch; no output file is edited by hand."""
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "build_tables.py"
t = SRC.read_text(encoding="utf-8")


def sub(anchor, replacement, count=1):
    global t
    n = t.count(anchor)
    assert n == count, "anchor occurs %d times, expected %d: %r" % (n, count, anchor[:110])
    t = t.replace(anchor, replacement)


# ---- (1) a single-value cell carries no mean ------------------------------------------
sub('''    nums = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if stats and nums and len(nums) == len(vals):
        out["mean"] = sum(nums) / len(nums)
        out["min"] = min(nums)
        out["max"] = max(nums)
    return out''',
    '''    nums = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if stats and nums and len(nums) == len(vals) and len(vals) > 1:
        out["mean"] = sum(nums) / len(nums)
        out["min"] = min(nums)
        out["max"] = max(nums)
    return out''')

# ---- (2) the ledger header says which seeds it is actually printing --------------------
sub('''    out.append(md_table(["part (seed 0 / seed 1 / seed 2)", "bytes", "how it is obtained"], rows))''',
    '''    out.append(md_table([led.get("seed_label", "part (seed 0 / seed 1 / seed 2)"),
                         "bytes", "how it is obtained"], rows))''')

sub('''        "rule": "FOUR PARTS, PRINTED, NEVER NETTED (w12r/MF3). Seed 0, K* = 32, TEST.",''',
    '''        "rule": "FOUR PARTS, PRINTED, NEVER NETTED (w12r/MF3). Seed 0, K* = 32, TEST.",
        "seed_label": "part (ONE seed, id 0)",''')

sub('''        "rule": ("FOUR PARTS, NEVER NETTED. The ladder's ledger file carries parts 1-3 measured on all 18 "''',
    '''        "seed_label": "part (seed 0 / seed 1 / seed 2 unless the row says otherwise)",
        "rule": ("FOUR PARTS, NEVER NETTED. The ladder's ledger file carries parts 1-3 measured on all 18 "''')

sub('''            "rule": "FOUR PARTS, NEVER NETTED. Measured on the object this deletion ran on.",''',
    '''            "rule": "FOUR PARTS, NEVER NETTED. Measured on the object this deletion ran on.",
            "seed_label": "part (ONE seed, id %d)" % seed,''')

# ---- (3) no table carries a cell whose name collides with the renderer's own column ----
sub('''        "seeds (n)": w1("lad_json", "/n_seeds_declared", "{:.0f}"),
        "corners": w3("lad_json", "/rungs/" + _rg + "/n_corners_per_seed/{s}", "{:.0f}"),''',
    '''        "seeds (n), traced": w1("lad_json", "/n_seeds_declared", "{:.0f}"),
        "corners": w3("lad_json", "/rungs/" + _rg + "/n_corners_per_seed/{s}", "{:.0f}"),''')

sub('''            "seeds (n)": derived([1], [dsrc(sid, "/seed")], "one seed, id " + str(seed), fmt="{:.0f}"),''',
    '''            "seeds (n), traced": derived([1], [dsrc(sid, "/seed")], "one seed, id " + str(seed),
                                         fmt="{:.0f}"),''')

sub('''         "seeds (n)": derived([1], [dsrc("x_del_test", "/seed")], "one seed, id 0", fmt="{:.0f}"),''',
    '''         "seeds (n), traced": derived([1], [dsrc("x_del_test", "/seed")], "one seed, id 0", fmt="{:.0f}"),''')

sub('''         "seeds (n)": derived([1], [dsrc("x_del_val", "/seed")], "one seed, id 0", fmt="{:.0f}"),''',
    '''         "seeds (n), traced": derived([1], [dsrc("x_del_val", "/seed")], "one seed, id 0", fmt="{:.0f}"),''')

sub('''            "seeds (n)": derived([3], [dsrc("base_ready", "/seeds_present")], "stored field", fmt="{:.0f}"),''',
    '''            "seeds (n), traced": derived([3], [dsrc("base_ready", "/seeds_present")], "stored field",
                                         fmt="{:.0f}"),''')

# ---- (3b) T12's rows vary in seed count, so the renderer's column must carry it ---------
sub('''        "label": "**K = %d**%s" % (_K, "  (= K\\\\*)" if _K == 32 else ""),
        "row_class": "SMOKE (val sweep) + exact byte arithmetic",
        "seeds_printed": None, "n_seeds_printed": None,''',
    '''        "label": "**K = %d**%s" % (_K, "  (= K\\\\*)" if _K == 32 else ""),
        "row_class": "SMOKE (val sweep) + exact byte arithmetic",
        "seeds_printed": ptr(DOCS["x_cross_val"], _P + "/seeds"),
        "n_seeds_printed": ptr(DOCS["x_cross_val"], _P + "/n_seeds"),''')

SRC.write_text(t, encoding="utf-8")
print("p6 OK ->", SRC)
