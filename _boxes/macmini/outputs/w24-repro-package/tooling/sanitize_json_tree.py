"""Apply the transform's string sanitiser to every JSON file under a directory (tooling only)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import transform
for p in Path(sys.argv[1]).rglob('*.json'):
    if p.name == 'weights.json':
        continue
    def jmap(o):
        if isinstance(o, dict):
            return {transform.sanitize(k): jmap(v) for k, v in o.items()}
        if isinstance(o, list):
            return [jmap(v) for v in o]
        if isinstance(o, str):
            return transform.sanitize(o)
        return o
    p.write_text(json.dumps(jmap(json.load(open(p))), indent=1) + '\n')
