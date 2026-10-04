#!/usr/bin/env python
"""A HUB-SHAPED BOOT, run from the bundle root — exactly what an autonomous-hub
does in its first five minutes, and nothing it is forbidden to do.

  1. read AGENT_PROTOCOL.md          (agent profile line 1)
  2. read handover_context.md §7     (line 2)
  3. read the charter + the pin      (§A.4)
  4. read its OWN task file          (line 3)
  5. write to the ONE allow-listed path  (§0.3)   <- the thing a push cannot give it
  6. import chlu FROM THE BUNDLE and touch a real gen loader
"""
import json, os, sys
from pathlib import Path

root = Path(os.environ["CHLU_ROOT"]).resolve()
claude = Path(os.environ["CHLU_CLAUDE_ROOT"])
hub = sys.argv[1]
out = {"chlu_root": str(root), "hub": hub, "steps": {}}

proto = (claude / "AGENT_PROTOCOL.md").read_text()
out["steps"]["1_protocol"] = {"chars": len(proto), "has_dial_declaration": "DIAL DECLARATION" in proto}

hand = (claude / "handover_context.md").read_text()
i = hand.index("## 7. Known Issues")
out["steps"]["2_handover_s7"] = {"chars": len(hand), "s7_first_line": hand[i:i + 60]}

charter = (claude / "autonomous-hub-charter.md").read_text()
PIN = "ae907466ba2b5fbec124e8995cf4d7c32a1ab30e"
assert PIN in charter, "the charter does not carry the pin"
out["steps"]["3_charter"] = {"chars": len(charter), "pin": PIN,
                             "floor": "MATCHED-BYTES LOOKUP" in charter}

task = (claude / "tasks" / f"{hub}.md").read_text()
out["steps"]["4_task_file"] = {"chars": len(task), "first_line": task.splitlines()[0]}

# ⭐ §0.3: the ONE path a hub may write. ⛔ Absolute, and derived from CHLU_ROOT.
d = Path(os.environ["CHLU_HUB_OUTPUTS"]) / hub
d.mkdir(parents=True, exist_ok=True)
log = d / "LOG.md"
with open(log, "a") as fh:
    fh.write("- boot smoke: charter+task read, chlu imported, writing here works\n")
assert log.is_file() and log.resolve().is_relative_to(root), log  # /tmp -> /private/tmp on macOS
out["steps"]["5_write"] = {"path": str(log.resolve()), "bytes": log.stat().st_size,
                           "inside_chlu_root": True}

import chlu
from chlu.core.gen_store import load_store_plan
from chlu.data.gen_splits import default_gen_data_root

assert Path(chlu.__file__).resolve().is_relative_to(root), chlu.__file__
plan = load_store_plan(Path(os.environ["CHLU_GEN_STORE_RUNS"]) / "shapes3d_recon" / "store_plan.json")
out["steps"]["6_import"] = {
    "chlu_file": str(Path(chlu.__file__).resolve()),
    "chlu_version": chlu.__version__,
    "default_gen_data_root": str(default_gen_data_root()),
    "store_plan": {"K": int(plan["K"]), "n_modes": int(plan["n_modes"]), "D": int(plan["D"])},
}
out["verdict"] = "HUB BOOT OK"
print(json.dumps(out, indent=2))
