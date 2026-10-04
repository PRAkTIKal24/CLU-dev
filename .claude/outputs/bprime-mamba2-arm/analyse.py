"""Aggregation + PREREG scoring for the mamba2 row. Reads the shipped artifact."""
import json, sys
import numpy as np

P = sys.argv[1] if len(sys.argv) > 1 else \
    '/Users/user/Desktop/CHLU/.claude/outputs/bprime-mamba2-arm/run_agg_n9/exp_bprime_rivals_metrics.json'
d = json.load(open(P))

def show(tab, title):
    print(f"\n===== {title} =====")
    rows = tab['aggregate']['rivals']
    hdr = ("arm","n","d","full","+0B","raw","lnd","div","null","blank","lift","RESC","p5-raw")
    print("{:11s}{:>3s}{:>4s}".format(*hdr[:3]) + "".join(f"{h:>18s}" for h in hdr[3:]))
    for k, v in rows.items():
        f = lambda a, b=None: f"{v[a]:+.4f}" + (f" ± {v[b]:.4f}" if b else "")
        print(f"{k:11s}{v['n_seeds']:>3d}{v['d_head']:>4d}"
              f"{f('full','full_se'):>18s}{f('zero_byte_margin','zero_byte_margin_se'):>18s}"
              f"{f('raw_table_margin','raw_table_margin_se'):>18s}{f('launder','launder_se'):>18s}"
              f"{f('dividend_vs_own_table','dividend_se'):>18s}{f('same_keys_null'):>18s}"
              f"{f('blank'):>18s}{f('lift_over_own_blank','lift_se'):>18s}"
              f"{str(v['RESCUED_above_own_blank_2se']):>18s}"
              f"{f('p5_vs_raw_gap','p5_vs_raw_gap_se'):>18s}")

for lbl, tab in d.get('audit_table_by_selection', {}).items():
    show(tab, f"selection = {lbl}  (n = {len(d['cells'])} seeds)")

# ---- reproduction of the banked f3 n=3 column (seeds 0,1,2 only) ------------
def mean_se(v):
    a = np.asarray([x for x in v if np.isfinite(x)])
    return float(a.mean()), float(a.std(ddof=1)/np.sqrt(a.size)) if a.size > 1 else 0.0

sub = [c for c in d['cells'] if c['seed'] in (0, 1, 2)]
print("\n===== f3 selection, seeds 0-2 only (vs the banked bprime-rivals-f3 n=3 column) =====")
BANKED = {"ttt_linear": (-0.6332, -0.4251), "ttt_mlp": (-0.5052, -0.2971),
          "deltanet": (-0.4478, -0.2396), "gdn": (-0.4104, -0.2022),
          "gdn2": (-0.4350, -0.2269)}
for arm in [a for a in d['rivals']]:
    per = [c['rivals_by_selection']['f3'][arm] for c in sub if arm in c['rivals_by_selection']['f3']]
    if not per: continue
    fm, _ = mean_se([p['arms']['full'] for p in per])
    rm, _ = mean_se([p['raw_table_control']['signed_margin_full_minus_raw'] for p in per])
    b = BANKED.get(arm)
    tag = "" if b is None else f"   banked {b[0]:+.4f} / {b[1]:+.4f}   Δ {fm-b[0]:+.5f} / {rm-b[1]:+.5f}"
    print(f"{arm:11s} full {fm:+.4f}  raw {rm:+.4f}{tag}")

# ---- selected configs + fit surface for mamba2 ------------------------------
print("\n===== mamba2 selected config per seed (f3 / fit-split selection) =====")
for c in d['cells']:
    b = c['rivals_by_selection']['f3']['mamba2']['fit']['best']
    v = c['rivals_by_selection'].get('f3_val', {}).get('mamba2', {}).get('fit', {}).get('best', {})
    print(f"s{c['seed']}  lr={b['lr']:<8g} wd={b['wd']:<4g} b={b['mini_batch']:<3d} fit={b['final']:.4f} "
          f"val={b.get('val_final', float('nan')):.4f}   |  held-out pick: lr={v.get('lr','-')} wd={v.get('wd','-')} val={v.get('val_final',float('nan')):.4f}")

print("\n===== low-lr / wd selection counts (the F3-grid-is-decorative question) =====")
for lbl in ('f3', 'f3_val'):
    n_low = n_wd = n_tot = 0
    for c in d['cells']:
        for arm, r in c['rivals_by_selection'].get(lbl, {}).items():
            b = r['fit']['best']; n_tot += 1
            n_low += b['lr'] < 1e-3; n_wd += b['wd'] > 0
    print(f"{lbl}: lr<1e-3 in {n_low}/{n_tot} cells; wd=0.1 in {n_wd}/{n_tot}")
    n_low = n_wd = n_tot = 0
    for c in d['cells']:
        r = c['rivals_by_selection'].get(lbl, {}).get('mamba2')
        if r: b = r['fit']['best']; n_tot += 1; n_low += b['lr'] < 1e-3; n_wd += b['wd'] > 0
    print(f"    mamba2 only: lr<1e-3 in {n_low}/{n_tot}; wd=0.1 in {n_wd}/{n_tot}")

# ---- ledger + coverage -----------------------------------------------------
print("\n===== ledger / coverage (mamba2) =====")
c0 = d['cells'][0]['rivals']['mamba2']
bl = c0['byte_ledger']
print("state floats declared", bl['state_floats_declared'], "measured moved", bl['state_floats_measured_moved'])
print("param B", bl['rival']['param_bytes'], "state B", bl['rival']['state_bytes'],
      "table B", bl['matched_table']['state_bytes'], "rows", bl['table_rows_affordable'],
      "lossless", bl['table_is_lossless'], "state/table", round(bl['state_over_own_table_bytes'], 4))
print("param breakdown", bl['rival']['param_breakdown'])
print("phi_id", d['cells'][0]['phi_id'])
print("raw readers", {k: round(v,4) for k,v in c0['raw_table_control']['all_raw_readers'].items()})
print("best raw reader", c0['raw_table_control']['best_reader'], "rows", c0['raw_table_control']['rows_affordable'],
      "lossless", c0['raw_table_control']['table_is_lossless'])
print("\nCLU reproduction (fidelity, per seed):",
      [round(c['clu_reproduction']['full'], 6) for c in d['cells'][:3]])
print("CLU ledger identity:", d['cells'][0]['clu_byte_ledger']['identity_T1']['ok'],
      d['cells'][0]['clu_byte_ledger']['full_bytes'], d['cells'][0]['clu_byte_ledger']['launder_bytes'])
