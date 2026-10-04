import os, sys, zipfile, hashlib, tempfile
import numpy as np
sys.path.insert(0, os.environ["CHLU_ROOT"])
import chlu.data.shapes3d as s3d
from chlu.data.gen_splits import save_index
def _lab():
    vals=[np.linspace(0.,.9,10),np.linspace(0.,.9,10),np.linspace(0.,.9,10),
          np.linspace(.75,1.25,8),np.arange(4,dtype=float),np.linspace(-30.,30.,15)]
    return np.stack([g.ravel() for g in np.meshgrid(*vals,indexing="ij")],axis=-1)
REF=os.path.join(os.environ["CHLU_ROOT"],".claude/data/gen/shapes3d/factor_index.npz")
split,_=s3d.build_split(labels=_lab())
d=tempfile.mkdtemp(); p,_=save_index(split,os.path.join(d,"f.npz"))
A,B=open(p,"rb").read(),open(REF,"rb").read()
print("len equal:",len(A)==len(B))
diffs=[i for i in range(len(A)) if A[i]!=B[i]]
print("n differing bytes:",len(diffs),"of",len(A))
print("offsets:",diffs[:40])
for i in diffs[:20]:
    print(f"  off {i:8d} (0x{i:x}): gen=0x{A[i]:02x} ship=0x{B[i]:02x}   ctx_gen={A[max(0,i-6):i+6]!r}")
print()
print("--- ZipInfo fields (generated vs shipped) ---")
za,zb=zipfile.ZipFile(p),zipfile.ZipFile(REF)
for ia,ib in zip(za.infolist(),zb.infolist()):
    if (ia.create_system,ia.create_version,ia.extract_version,ia.external_attr,ia.flag_bits,ia.date_time,ia.compress_type) != \
       (ib.create_system,ib.create_version,ib.extract_version,ib.external_attr,ib.flag_bits,ib.date_time,ib.compress_type):
        print(f"  {ia.filename}:")
        for f in ("create_system","create_version","extract_version","external_attr","flag_bits","date_time","compress_type","CRC"):
            va,vb=getattr(ia,f),getattr(ib,f)
            print(f"     {f:17s} gen={va!r:22s} ship={vb!r}   {'DIFFERS' if va!=vb else ''}")
    else:
        print(f"  {ia.filename}: all header fields equal")
print()
print("sys.platform =", sys.platform, "-> ZipInfo default create_system =", zipfile.ZipInfo("x").create_system)
print("(CPython zipfile.ZipInfo.__init__: create_system = 0 if sys.platform=='win32' else 3)")
