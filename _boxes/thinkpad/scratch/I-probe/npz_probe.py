import os, sys, time, hashlib, zipfile, tempfile
import numpy as np
sys.path.insert(0, os.environ["CHLU_ROOT"])
def sha(p):
    return hashlib.sha256(open(p,"rb").read()).hexdigest()
d = tempfile.mkdtemp()
# 1) is np.savez itself byte-stable across calls?
a = {"x": np.arange(10, dtype=np.int64)}
p1, p2 = os.path.join(d,"a.npz"), os.path.join(d,"b.npz")
with open(p1,"wb") as f: np.savez(f, **a)
time.sleep(0)  # same second
with open(p2,"wb") as f: np.savez(f, **a)
print("np.savez same-second identical :", sha(p1) == sha(p2), sha(p1)[:16], sha(p2)[:16])
zi = zipfile.ZipFile(p1).infolist()[0]
print("np.savez ZipInfo.date_time     :", zi.date_time, " <- localtime embedded?" )
print("now localtime                  :", time.localtime()[:6])
