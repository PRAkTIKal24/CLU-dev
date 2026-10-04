"""Stage A (h5py venv): read the Shapes3D thumbnails the task figure shows.

Reads ONLY the rows listed below from the pinned 3dshapes.h5 (sha256 checked against
chlu.data.shapes3d.H5_SHA256 by the caller's ASSETS record) and writes them, with their
row ids and label vectors, to <out>/shapes_images.npz.  No model, no randomness.

Rows: the request row (REQUESTS.json set1[0], dataset row 424182) and, for each of the six
factors, the request's tuple with that one factor set to each STRIP value.
usage: python extract_images.py --h5 <3dshapes.h5> --out <dir>
"""
import argparse
import json
import h5py
import numpy as np
CARDS = (10, 10, 10, 8, 4, 15)
NAMES = ("floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation")
REQUEST_ROW = 424182
STRIP = {
    "floor_hue": (0, 2, 4, 6, 8),
    "wall_hue": (0, 2, 4, 6, 8),
    "object_hue": (0, 2, 4, 6, 8),
    "scale": (0, 2, 4, 5, 7),
    "shape": (0, 1, 2, 3),
    "orientation": (0, 4, 7, 10, 14),
}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base = np.array(np.unravel_index(REQUEST_ROW, CARDS))
    rows = [REQUEST_ROW]
    strip = {}
    for f, name in enumerate(NAMES):
        rr = []
        for v in STRIP[name]:
            t = base.copy()
            t[f] = v
            rr.append(int(np.ravel_multi_index(tuple(t), CARDS)))
        strip[name] = rr
        rows += rr
    uniq = sorted(set(rows))
    with h5py.File(a.h5, "r") as h:
        imgs = np.stack([np.asarray(h["images"][r]) for r in uniq]).astype(np.uint8)
        labs = np.stack([np.asarray(h["labels"][r]) for r in uniq]).astype(np.float64)
    np.savez(f"{a.out}/shapes_images.npz", rows=np.array(uniq, np.int64), images=imgs,
             labels=labs, request_row=np.int64(REQUEST_ROW),
             strip=json.dumps({"strip_indices": STRIP, "strip_rows": strip}, sort_keys=True))
    print("wrote", len(uniq), "images", imgs.shape)


if __name__ == "__main__":
    main()
