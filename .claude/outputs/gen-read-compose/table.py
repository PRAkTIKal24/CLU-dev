"""Read the per-seed compose manifests and print the report tables (pure stdlib)."""
import json, sys, glob, statistics as st

paths = sorted(glob.glob('/Users/user/Desktop/CHLU/.claude/outputs/gen-read-compose/runs/shapes3d_recon/seed_*/compose_manifest.json'))
mans = [json.load(open(p)) for p in paths]
print(f"{len(mans)} seeds:", [m['seed'] for m in mans])

def ms(vals):
    v=[x for x in vals if x is not None]
    m=sum(v)/len(v); s=(sum((x-m)**2 for x in v)/len(v))**0.5
    return m, s, v

def row(label, getter):
    m,s,v = ms([getter(x) for x in mans])
    return f"| {label} | **{m:.4f} ± {s:.4f}** | " + " / ".join(f"{x:.4f}" for x in v) + " |"

print("\n### gate statistic: in-corner shape balanced accuracy")
print("| arm | mean ± sd | per seed |\n|---|---|---|")
print(row("decoder ceiling", lambda m: m['ceilings']['decoder']['in_corner_discrimination']['balanced_accuracy']))
print(row("STORE ceiling (measured)", lambda m: m['ceilings']['store']['in_corner_discrimination']['balanced_accuracy']))
for a in ('read','clu_wide','store_removed','proto'):
    print(row(a, lambda m,a=a: m['arms'][a]['in_corner_discrimination']['balanced_accuracy']))

print("\n### AUROC companion")
print(row("decoder ceiling", lambda m: m['ceilings']['decoder']['in_corner_discrimination']['auroc']))
print(row("STORE ceiling", lambda m: m['ceilings']['store']['in_corner_discrimination']['auroc']))
for a in ('read','clu_wide','store_removed','proto'):
    print(row(a, lambda m,a=a: m['arms'][a]['in_corner_discrimination']['auroc']))

print("\n### TPR / TNR")
for a in ('read','clu_wide','store_removed','proto'):
    t=ms([m['arms'][a]['in_corner_discrimination']['tpr'] for m in mans])
    n=ms([m['arms'][a]['in_corner_discrimination']['tnr'] for m in mans])
    print(f"| {a} | TPR {t[0]:.4f} ± {t[1]:.4f} | TNR {n[0]:.4f} ± {n[1]:.4f} |")

for split in ('A','B'):
    key = 'composition_'+split
    print(f"\n### {split}: aggregate and shape (separate, always)")
    print("| arm | aggregate | shape |\n|---|---|---|")
    for lbl, get in (("decoder ceiling", lambda m: m['ceilings']['decoder'][key]),
                     ("STORE ceiling", lambda m: m['ceilings']['store'][key])):
        a=ms([get(m)['aggregate'] for m in mans]); s=ms([get(m)['per_factor']['shape'] for m in mans])
        print(f"| {lbl} | {a[0]:.4f} ± {a[1]:.4f} | {s[0]:.4f} ± {s[1]:.4f} |")
    for arm in ('read','clu_wide','store_removed','proto'):
        a=ms([m['arms'][arm][key]['aggregate'] for m in mans]); s=ms([m['arms'][arm][key]['per_factor']['shape'] for m in mans])
        print(f"| {arm} | {a[0]:.4f} ± {a[1]:.4f} | {s[0]:.4f} ± {s[1]:.4f} |")

print("\n### falsifier: read − store_removed (in-corner shape balacc)")
d = ms([m['falsifier_read_vs_store_removed']['delta_in_corner_balacc'] for m in mans])
spread = max(ms([m['arms']['read']['in_corner_discrimination']['balanced_accuracy'] for m in mans])[1],
             ms([m['arms']['store_removed']['in_corner_discrimination']['balanced_accuracy'] for m in mans])[1])
print(f"delta = {d[0]:+.4f} ± {d[1]:.4f}  per-seed " + " / ".join(f"{x:+.4f}" for x in d[2]))
print(f"largest arm spread = {spread:.4f} -> delta/spread = {d[0]/spread if spread else float('inf'):.1f}x; seeds positive = {sum(1 for x in d[2] if x>0)}/{len(d[2])}")
dB = ms([m['falsifier_read_vs_store_removed']['delta_B_shape_accuracy'] for m in mans])
print(f"delta B shape = {dB[0]:+.4f} ± {dB[1]:.4f}")

print("\n### fractions of ceilings")
for a in ('read','clu_wide','store_removed','proto'):
    fd=ms([m['arms'][a]['fraction_of_decoder_ceiling'] for m in mans]); fs=ms([m['arms'][a]['fraction_of_store_ceiling'] for m in mans])
    print(f"| {a} | {fd[0]:.4f} ± {fd[1]:.4f} of decoder | {fs[0]:.4f} ± {fs[1]:.4f} of store |")

print("\n### diagnostics")
for k in ('A','B','MS'):
    e=ms([m['request_addressing']['code_agreement_vs_feature_addressing'][k]['exact_set'] for m in mans])
    w=ms([m['request_addressing']['code_agreement_vs_feature_addressing'][k]['mean_well_agreement'] for m in mans])
    print(f"code agreement {k}: exact-set {e[0]:.4f} ± {e[1]:.4f}, per-well {w[0]:.4f} ± {w[1]:.4f}")
c=ms([m['request_addressing']['mean_confidence']['B'] for m in mans])
print(f"request posterior confidence on B: {c[0]:.4f} ± {c[1]:.4f}")
for k in ('A','B'):
    n=ms([float(m['request_addressing']['n_distinct_code_configs'][k]) for m in mans])
    print(f"distinct code configs {k}: {n[0]:.1f} ± {n[1]:.1f} (of {mans[0]['rows'][k]})")
for a in ('read','clu_wide','store_removed','proto'):
    n=ms([float(m['arms'][a]['n_distinct_images_proxy']['B']) for m in mans])
    print(f"distinct rendered vectors on B, {a}: {n[0]:.1f} ± {n[1]:.1f} (of 72)")
print("\nbytes:", mans[0]['byte_ledger']['deployed_bytes'], "excluded coupling:", mans[0]['byte_ledger']['coupling_bytes_excluded'],
      "addressing head:", mans[0]['byte_ledger']['addressing_head']['bytes'], "proto table:", mans[0]['prototype_ledger']['bytes'])
print("channel:", {k: (v.get('bits') or v.get('n_real_values')) for k,v in mans[0]['channel_ledger'].items() if k!='reference_rows'})
print("posterior entropy bits:", [round(m['channel_ledger']['clu_wide']['mean_posterior_entropy_bits'],3) for m in mans])
print("invariance:", [(m['metric_invariance_check']['invariant'], round(m['metric_invariance_check']['delta_raw_accuracy'],4)) for m in mans])
print("wall clock s:", [round(m['wall_clock_s']) for m in mans])
print("decoder recipes:", {m['seed']: (m['decoder']['decoder_recipe'], m['decoder']['decoder_param_hash'][:12]) for m in mans})
print("store training final loss:", [round(m['store_training']['loss_final'],3) if m['store_training']['loss_final'] else 'ckpt' for m in mans])
print("rows:", mans[0]['rows'], [m['rows'] for m in mans])
