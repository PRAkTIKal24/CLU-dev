import os, sys, hashlib, zipfile, json, tempfile
import numpy as np
sys.path.insert(0, os.environ["CHLU_ROOT"])
import chlu.data.shapes3d as s3d
from chlu.data.gen_splits import save_index

def _synthetic_shapes3d_labels():
    vals = [np.linspace(0.0,0.9,10), np.linspace(0.0,0.9,10), np.linspace(0.0,0.9,10),
            np.linspace(0.75,1.25,8), np.arange(4,dtype=float), np.linspace(-30.0,30.0,15)]
    return np.stack([g.ravel() for g in np.meshgrid(*vals, indexing="ij")], axis=-1)

REF = os.path.join(os.environ["CHLU_ROOT"], ".claude/data/gen/shapes3d/factor_index.npz")
labels = _synthetic_shapes3d_labels()
split, _ = s3d.build_split(labels=labels)
d = tempfile.mkdtemp()
p, sha = save_index(split, os.path.join(d, "factor_index.npz"))
print("generated sha :", sha)
print("shipped   sha :", hashlib.sha256(open(REF,"rb").read()).hexdigest())
print("sizes         : generated", os.path.getsize(p), " shipped", os.path.getsize(REF))
print()

za, zb = zipfile.ZipFile(p), zipfile.ZipFile(REF)
na, nb = za.namelist(), zb.namelist()
print("member order generated:", na)
print("member order shipped  :", nb)
print()
A = np.load(p, allow_pickle=False); B = np.load(REF, allow_pickle=False)
for k in na:
    key = k[:-4]
    if key not in B.files:
        print(f"{key:22s} MISSING IN SHIPPED"); continue
    a, b = A[key], B[key]
    same_bytes = (a.shape == b.shape and a.dtype == b.dtype
                  and np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes())
    print(f"{key:22s} gen[{str(a.dtype):9s} {str(a.shape):12s}]  ship[{str(b.dtype):9s} {str(b.shape):12s}]  identical={same_bytes}")
    if not same_bytes:
        try:
            eq = np.array_equal(a, b)
            print(f"{'':22s}   values equal (ignoring dtype): {eq}")
        except Exception as e:
            print(f"{'':22s}   compare error: {e}")
        if key == "meta":
            ja, jb = json.loads(str(a)), json.loads(str(b))
            allk = sorted(set(ja) | set(jb))
            for kk in allk:
                if ja.get(kk, "<absent>") != jb.get(kk, "<absent>"):
                    print(f"{'':22s}   meta['{kk}']: gen={ja.get(kk,'<absent>')!r}  ship={jb.get(kk,'<absent>')!r}")
# raw compressed-member compare
print("\nraw member bytes:")
for k in na:
    ba, bb = za.read(k), zb.read(k)
    print(f"  {k:26s} gen_len={len(ba):9d} ship_len={len(bb):9d} equal={ba==bb}")
