"""SPOKE I §1 — do the FROZEN artefacts LOAD to bit-identical arrays, and does a
forward pass through them reproduce values recorded in the shipped manifests?
READ-ONLY: opens files, writes nothing."""
import os, sys, json, hashlib
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, os.environ["CHLU_ROOT"])
R = os.environ["CHLU_ROOT"]
from chlu.utils.checkpoints import load_model
from chlu.core.gen_bridge import param_hash, param_count

def J(p): return json.load(open(os.path.join(R, p), encoding="utf-8"))
def fsha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()

ok = fail = 0
def chk(label, got, exp):
    global ok, fail
    m = got == exp
    ok, fail = ok + m, fail + (not m)
    print(f"  [{'MATCH' if m else 'DIFFER':6s}] {label}")
    print(f"           got={got}")
    if not m: print(f"           exp={exp}")

print("#" * 78)
print("# A. WINNER DECODERS  (.claude/outputs/gen-decoder-lift/winner/)")
print("#    checked against winner_hashes.json AND classifier_gate.json")
print("#" * 78)
W = J(".claude/outputs/gen-decoder-lift/winner/winner_hashes.json")
G = J(".claude/outputs/gen-external-classifier/classifier_gate.json")
dec = {}
for s in (0, 1, 2):
    p = os.path.join(R, f".claude/outputs/gen-decoder-lift/winner/decoder_v1_s{s}.pkl")
    print(f"\n-- seed {s}: {os.path.basename(p)} ({os.path.getsize(p):,} B)")
    chk(f"raw file sha256 vs winner_hashes['seed_{s}'].pkl_sha256", fsha(p), W[f"seed_{s}"]["pkl_sha256"])
    d = load_model(p); dec[s] = d
    chk(f"param_hash(loaded) vs winner_hashes['seed_{s}'].param_hash", param_hash(d), W[f"seed_{s}"]["param_hash"])
    chk(f"param_count(loaded) vs winner_hashes['seed_{s}'].param_count", param_count(d), W[f"seed_{s}"]["param_count"])
    chk(f"param_hash(loaded) vs classifier_gate.decoder.param_hash_per_seed['{s}']",
        param_hash(d), G["decoder"]["param_hash_per_seed"][str(s)])
    lv = jax.tree_util.tree_leaves(__import__("equinox").partition(d, __import__("equinox").is_array)[0])
    print(f"           n leaves={len(lv)}  leaf-bytes sha="
          f"{hashlib.sha256(b''.join(np.ascontiguousarray(np.asarray(x)).tobytes() for x in lv)).hexdigest()}")

print()
print("#" * 78)
print("# B. FROZEN ENCODER / DECODER / PROBE  (gen-feature-bridge, bridge_manifest.json)")
print("#" * 78)
enc = {}
for s in (0, 1, 2):
    b = f".claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_{s}"
    M = J(b + "/bridge_manifest.json")
    print(f"\n-- seed {s}")
    e = load_model(os.path.join(R, b, "encoder.pkl")); enc[s] = e
    chk(f"encoder param_hash vs bridge_manifest.encoder_param_hash", param_hash(e), M["encoder_param_hash"])
    chk(f"encoder param_hash vs feature_cache.encoder_param_hash", param_hash(e), M["feature_cache"]["encoder_param_hash"])
    chk(f"encoder feat_shape vs feature_cache.feat_shape", [int(x) for x in e.feat_shape], M["feature_cache"]["feat_shape"])
    d0 = load_model(os.path.join(R, b, "decoder.pkl"))
    chk(f"decoder param_hash vs bridge_manifest.decoder_param_hash", param_hash(d0), M["decoder_param_hash"])
    chk(f"decoder param_count vs bridge_manifest.decoder_param_count", param_count(d0), M["decoder_param_count"])
    pr = load_model(os.path.join(R, b, "probe.pkl"))
    chk(f"probe param_hash vs bridge_manifest.probe_param_hash", param_hash(pr), M["probe_param_hash"])
    chk(f"probe param_count vs bridge_manifest.probe_param_count", param_count(pr), M["probe_param_count"])
    chk(f"winner_hashes['seed_{s}'].encoder_param_hash_it_was_fit_against",
        param_hash(e), W[f"seed_{s}"]["encoder_param_hash_it_was_fit_against"])

print()
print("#" * 78)
print("# C. EXTERNAL CLASSIFIERS  (classifier_manifest.json)")
print("#" * 78)
for s in (0, 1, 2):
    b = f".claude/outputs/gen-external-classifier/seed_{s}"
    M = J(b + "/classifier_manifest.json")
    c = load_model(os.path.join(R, b, "external_classifier.pkl"))
    print(f"\n-- seed {s}  (manifest keys: {sorted(M)[:12]})")
    for k in M:
        if "param_hash" in k: chk(f"classifier {k}", param_hash(c), M[k])
        if "param_count" in k: chk(f"classifier {k}", param_count(c), M[k])

print()
print("#" * 78)
print("# D. FROZEN STORES  (gen-read-compose/store_s*.pkl vs store_manifest.json)")
print("#" * 78)
for s in (0, 1, 2):
    p = os.path.join(R, f".claude/outputs/gen-read-compose/runs/shapes3d_recon/seed_{s}/store_s{s}.pkl")
    M = J(f".claude/outputs/gen-store-grouping/runs/shapes3d_recon/seed_{s}/store_manifest.json")
    st = load_model(p)
    print(f"\n-- seed {s} ({os.path.getsize(p):,} B)")
    print(f"           store param_hash = {param_hash(st)}")
    print(f"           n_wells={st.n_wells} modes_per_well={st.n_modes_per_well} dim={st.dim}")
    for k in sorted(M):
        if isinstance(M[k], (str, int, float)) and ("hash" in k or k in ("D", "K")):
            print(f"           manifest[{k}] = {M[k]!r}")

print()
print("#" * 78)
print("# E. FORWARD PASS through a loaded frozen encoder on the shipped features.npy")
print("#" * 78)
for s in (0,):
    b = os.path.join(R, f".claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_{s}")
    fp = os.path.join(b, "features.npy")
    M = J(f".claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_{s}/bridge_manifest.json")
    fc = M["feature_cache"]
    print(f"-- seed {s} features.npy present={os.path.exists(fp)} "
          f"size={os.path.getsize(fp) if os.path.exists(fp) else 0:,} B "
          f"(manifest says {fc['features_bytes']:,} B)")
    if os.path.exists(fp) and os.path.getsize(fp) == fc["features_bytes"]:
        got = fsha(fp)
        chk("features.npy raw sha256 vs feature_cache.features_sha256", got, fc["features_sha256"])
        F = np.load(fp, mmap_mode="r")
        print(f"           features dtype={F.dtype} shape={F.shape} (manifest dtype={fc['dtype']} D={fc['D']})")
        rows = np.asarray(F[:64], np.float32)
        print(f"           sha256(first 64 rows, float32) = {hashlib.sha256(rows.tobytes()).hexdigest()}")
        d = dec[s]
        # decode 8 rows through the frozen winner decoder -> a real inference number
        fs = tuple(int(x) for x in fc["feat_shape"])
        X = jnp.asarray(rows[:8].reshape((8,) + fs))
        Y = np.asarray(jax.vmap(d)(X))
        print(f"           decoder(8 cached rows): shape={Y.shape} dtype={Y.dtype}")
        print(f"           sha256(render bytes)  = {hashlib.sha256(np.ascontiguousarray(Y).tobytes()).hexdigest()}")
        print(f"           mean={float(Y.mean()):.17g}  std={float(Y.std()):.17g}")
        print(f"           Y.ravel()[:6] = {Y.ravel()[:6].tolist()}")
    else:
        rf = os.path.join(b, "feature_rows.npy")
        print(f"           feature_rows.npy present={os.path.exists(rf)}")
        print("           ** features.npy absent or truncated - cache check NOT RUN **")

print()
print("#" * 78)
print(f"# SUMMARY: {ok} MATCH, {fail} DIFFER")
print("#" * 78)
