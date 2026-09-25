#!/usr/bin/env python3
"""The README's top section (PW-1 item 4: the one rebuild command) and its patch into an existing
README. make_index.py (pw1_draft_figures) imports REBUILD_MD / RETIRED_HEADLINE / OTHER_CODE_MD
from here, so a regenerated README says the same thing.
usage: python readme_block.py <README.md in> <README.md out>
"""
import re
import sys

ENGINE_COMMIT = "a01bea7"

REBUILD_MD = """\
## Rebuild: the one command

```
cd ~/clu/outputs/paper-visuals && bash build.sh
```

This builds every figure this folder owns (below) twice from the evidence, stops unless the two
builds are byte-identical, then installs them into `~/clu/outputs/w17-paper/figures/` and
`~/clu/outputs/paper-visuals/` and re-hashes them there. The same command works in
`~/clu/outputs/w17-paper/figures/` (same code), in a fresh copy of either folder anywhere (e.g.
under `/tmp`), and from a CHLU checkout as `bash w21_figures/build.sh`. No `source` step is needed:
`build.sh` sets its own `PYTHONPATH`. It needs `python` from the project venv on `PATH`
(`~/clu/workspace/.venv/bin`), or `PYTHON=<that python> bash build.sh`.

- **Try an edit without touching the installed files:** `bash build.sh <scratch dir 1> <scratch dir 2>`.
- **Plot-edit loop:** edit `make_figures_w22.py`, `make_figures_w21.py`, `cmazet_panels.py` or a
  `data/*.json` file here, then run `bash build.sh`. The build copies its code only into a
  destination that is not this folder. To keep an edit, copy the file back to `w21_figures/` (or
  `w19_task_figures/`) in the CHLU repo and commit it there.
- **Built here** (10 stems: 20 files, plus `PROVENANCE.json`, `GLYPH-IDENTITY.json` and the
  manifests `SHA256-w22-build{1,2}.txt`): `fig_task_cmazet`, `fig_ladder`, `fig_crossover`,
  `fig_deletion`, `fig_encoder_adapt`, `fig_maze_conditions`, `fig_cmazet_families`,
  `fig_cmazet_per_arm`, `fig_cmazet_rel_per_arm` and `fig_cmazet_door_then_tint`.
  `fig_headline_steps` is no longer built. It has no slot, and its last build is in `retired-pw1/`.
- **Inputs it reads:**
  - the evidence tree `$CHLU_EVIDENCE` (default `~/clu/evidence/outputs`);
  - `data/*.json` in this folder;
  - the maze-engine snapshot `engine/`: 61 files copied from a CHLU checkout by
    `tools/vendor_engine.py`, each hash-pinned in `engine/ENGINE.json` with the source commit
    (the files are byte-identical at `f571b2b` and `%s`).

  Set `CHLU_FIG_CODE=<a CHLU checkout>` to use a checkout's engine instead. If no engine is found,
  the build stops before drawing and names exactly this. Every engine module is checked to load
  from the engine root, never from another checkout on `sys.path`.

### Mirrored figures (not built by `build.sh`): one rebuild line each

- `fig_task_shapes.{pdf,png}` (w19 task figures), rebuilt from the frozen w19 assets. The line
  below reproduces the installed bytes (`c83b24d5…` / `78a8d7ed…`, two builds, checked 2026-09-25):
  `cd ~/clu/outputs/paper-visuals && PYTHONPATH=.:engine/scripts:engine python make_task_figures.py --only all --assets ~/clu/outputs/w19-task-figures/assets --out /tmp/task-shapes && cp /tmp/task-shapes/fig_task_shapes.* .`
  The assets come from two upstream stages, run from a CHLU checkout:
  - `shapes_images.npz` (`9f9d71bc…`): `w19_task_figures/extract_images.py --h5 ~/clu/data/gen/shapes3d/3dshapes.h5`,
    using the Shapes3D dataset (sha256 `0a0f6ed9…`, the runbook pin) in the h5py venv
    `~/clu/outputs/w19-task-figures/h5venv`.
  - `task_assets.npz` (`bc6df90e…`): `w19_task_figures/extract_assets.py`, in the project venv (JAX)
    with the frozen decoder `~/clu/stage/outputs/gen-decoder-lift/winner/decoder_v1_s0.pkl`
    (param hash `01ef91f7…`) and the judge classifier (`c8741ac4…`). Note that `extract_assets.py`
    reads `w17-paper/figures/rendered/REQUESTS.json`, which is now under `rendered/retired-pw1/`.
- `rendered/rc_pw1_shapes_{per_model,baseline_wins}.{pdf,png}`, with `rendered/REQUESTS-pw1.json`
  and `rendered/PROVENANCE-render.json`, rendered through the one frozen decoder:
  `cd <CHLU checkout at %s or later> && bash pw1_render_shapes/build_twice.sh`
  This builds twice into `~/clu/outputs/pw1-render-shapes/build/{a,b}` (earlier builds are moved
  aside, not deleted). Then copy `build/a/rc_pw1_*`, `REQUESTS-pw1.json`, `PROVENANCE-render.json`
  and `SHA256-build{1,2}.txt` (as `SHA256-pw1-render-build{1,2}.txt`) into `rendered/`. The inputs
  are the project venv (JAX) and `pw1_render_shapes/run_env.sh`'s stage roots:
  - the frozen decoder `stage/outputs/gen-decoder-lift/winner` (`01ef91f7…`);
  - the judge classifier `stage/outputs/gen-external-classifier` (`c8741ac4…`);
  - the Shapes3D value hold-out split `stage/outputs/data/gen`;
  - the w20 baseline objects `stage/w20-shapes-data-matched`, hash-checked against `SHA256-w20-test-objects.txt`;
  - the n-CLU object `stage/outputs/w12-single-design-crossover/scored/run1_seed0_K32/arm.pkl`.
""" % (ENGINE_COMMIT, ENGINE_COMMIT)

RETIRED_HEADLINE = ("Retired (no slot since w22 Ruling 2; no longer built, PW-1 item 4): distinct valid "
                    "novel tuples per held-out combination against the baselines' optimiser steps (log), "
                    "3 seeds.")

OTHER_CODE_MD = ("- `build.sh`, `figroot.py`, `make_figures*.py`, `maze_six.py`, `cmazet_*.py`, "
                 "`make_task_figures.py`, `verify_glyph.py`, `validate_palette.py`, `tools/`, `data/*.json`, "
                 "`engine/`: the build code, its data and its maze-engine snapshot (see Rebuild). "
                 "`cmazet_extract_rel.py` is not part of the build: it re-reads persisted models and "
                 "needs a full CHLU checkout (`CHLU_FIG_CODE`).")

OLD_HEADLINE = ("Retired (no slot since w22 Ruling 2; still produced by build.sh): distinct valid novel "
                "tuples per held-out combination against the baselines' optimiser steps (log), 3 seeds.")


def patch(text):
    lines = text.split("\n")
    assert lines[0].startswith("# ") and lines[1] == "", lines[:2]
    body = "\n".join(lines[2:])
    # drop the old Rebuild section (up to the next "## ")
    body, n = re.subn(r"## Rebuild\n.*?(?=\n## )", "", body, count=1, flags=re.S)
    assert n == 1, "old Rebuild section not found"
    body = body.replace("\n\n\n## ", "\n\n## ")
    assert body.count(OLD_HEADLINE) == 2, body.count(OLD_HEADLINE)
    body = body.replace(OLD_HEADLINE, RETIRED_HEADLINE)
    old_other = re.search(r"^- `build.sh`, `make_figures\*\.py`.*$", body, flags=re.M)
    assert old_other, "old code bullet not found"
    body = body[:old_other.start()] + OTHER_CODE_MD + body[old_other.end():]
    return "\n".join(lines[:2]) + "\n" + REBUILD_MD + "\n" + body


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src = open(sys.argv[1], encoding="utf-8").read()
    open(sys.argv[2], "w", encoding="utf-8").write(patch(src))
