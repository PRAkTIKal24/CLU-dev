"""Compare this box's regenerated GPU-leg caches with the Mac's, block by block (sha256)."""
import json, sys
from pathlib import Path
R = Path("C:/chlu/outputs/w15-bar-d-score/gpu-run")
M = Path("C:/chlu/evidence/outputs/w8-encoder-loop/cloud/manifests")
out = {}
for s in (0, 1, 2):
    for a in ("alone", "loop_nosw", "loop_swap"):
        box_p = R / f"seed_{s}" / a / "cache.json"
        if not box_p.exists():
            out[f"s{s}:{a}"] = "box cache not built yet"; continue
        box = json.loads(box_p.read_text())["manifest"]["block_sha256"]
        mac = json.loads((M / f"seed_{s}" / a / "cache.json").read_text())["manifest"]["block_sha256"]
        same = [k for k in mac if box.get(k) == mac[k]]
        out[f"s{s}:{a}"] = {"n_blocks_mac": len(mac), "n_blocks_box": len(box),
                            "n_blocks_identical": len(same),
                            "all_identical": len(same) == len(mac) == len(box)}
Path("C:/chlu/outputs/w15-bar-d-score/CACHE-XMACHINE.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
