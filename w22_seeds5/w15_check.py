"""CLI for the w15 three-way check. usage: w15_check.py DRAFT.tex OUT.json"""
import json, sys
from w15_check_lib import run
r = run(sys.argv[1])
json.dump(r, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
rows = r["rows"]
bad_f = [x for x in rows if x["findings_line"] is None]; bad_j = [x for x in rows if x["artefact_ok"] is False]; noj = [x for x in rows if x["artefact_ok"] is None]
print(f"tokens scanned {r['n_tokens']}; mapped {len(rows)}; problems {len(r['problems'])}; not verbatim in cited FINDINGS {len(bad_f)}; JSON disagrees {len(bad_j)}; no JSON {len(noj)}; extras {len(r['extras'])}")
for p in r["problems"]: print("  PROBLEM", p)
for x in bad_f: print("  FINDINGS-MISS", x["draft_line"], x["token"], x["findings"])
for x in bad_j: print("  JSON-DISAGREE", x["draft_line"], x["token"], x.get("artefact"), x.get("pointer"), x.get("json_value"))
for x in noj: print("  NOJSON", x["draft_line"], x["token"], x["nojson"])
for e in r["extras"]: print("  EXTRA", e["draft_line"], e["draft_verbatim"], e["on_line"], e["values"])
