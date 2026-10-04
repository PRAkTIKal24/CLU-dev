"""w21 spoke/words-full-test -- the INSTRUMENT CHECK (run before any TEST number).

    python scripts/w21_val_repro.py <seed> <out_dir>

Re-fits the CLU (config A1) and the four charter-C controls on WORDS-FULL, VAL leg, through
the w19 runner's own objects (``w19_symmetric.Inputs``, W16.banks / cap_split, A.train_codec,
A.TransitionClu, W16.make_control, W16.sc) and scores the unseen n-way row.  It is the
first ~110 lines of ``w19_symmetric.run`` with the rival / floor / rollout blocks omitted
(none of them touches the CLU path: the runner's ``rng`` is consumed only by the rival
trajectory sampler).  Nothing is re-implemented; every call is the runner's call.

The w19 printed triple it must reproduce: CLU n-way 0.8019 / 0.5689 / 0.8034 (seeds 0/1/2),
``/c/chlu/outputs/w19-maze-t-symmetric/val/WORDS-FULL_val_seed<N>.json``.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import w19_symmetric as S                                            # noqa: E402

T, A, R, W16, K = S.T, S.A, S.R, S.W16, S.K_COLOURS


def main(seed_index, out_dir, cond="WORDS-FULL", leg="val"):
    t0 = time.time()
    IN = S.Inputs(cond)
    cfg = dict(latent=128, K=2, n_atoms=6, ridge=1.0)
    base, ev_base, (tr_insts, tr_rows, tr_nat, tr_meta), \
        (va_insts, va_rows, va_nat, va_meta) = W16.banks(seed_index, leg)
    ho = T.make_holdout(K, seed=seed_index)
    tr_split = T.split_rows(tr_rows, ho, K)
    va_split = W16.cap_split(T.split_rows(va_rows, ho, K), seed_index)
    train_rows = tr_split.get("seen", [])
    Ctr = A.build_codes(train_rows, K)
    Ccodec = np.concatenate([Ctr, T.enumerate_code_alphabet(K)], axis=0)
    codec, crep = A.train_codec(Ccodec, K, latent=cfg["latent"], steps=2500, seed=seed_index)
    Ztr = A.codec_encode(codec, Ctr)
    Wtr = A.build_words(train_rows)
    pb = IN.clu_pair_block
    clu = A.TransitionClu(Ztr, Wtr, IN.grids, K=cfg["K"], n_atoms=cfg["n_atoms"],
                          ridge=cfg["ridge"], pair_block=pb, seed=seed_index)
    controls = {c: W16.make_control(Ztr, Wtr, IN.grids, cfg["K"], cfg["n_atoms"],
                                    cfg["ridge"], c, seed_index, pair_block=pb)
                for c in A.TransitionClu.CONTROLS}
    der = [int(x) for x in controls["landscape_shuffled_across_wells"].derangement]
    assert not np.any(np.asarray(der) == np.arange(len(der))), der

    def dec(obj, Wx):
        return T.decode_codes(K, A.codec_decode(codec, obj.read(Wx)))

    out = {"seed_index": seed_index, "leg": leg, "condition": cond, "config": cfg,
           "clu_read_floats": int(clu.ledger()["read_floats"]),
           "across_wells_derangement": der, "across_wells_is_derangement": True,
           "rows": {}}
    for name in ("seen", "pair", "nway", "value"):
        rs = va_split.get(name, [])
        Wr = A.build_words(rs)
        b = {"n": len(rs), "CLU": W16.sc(va_insts, rs, dec(clu, Wr))}
        for c, obj in controls.items():
            b["control: " + c] = W16.sc(va_insts, rs, dec(obj, Wr))
        out["rows"][name] = b
        print("[repro %s %s seed %d] %-5s n=%-5d CLU %.4f  ctl %s" % (
            cond, leg, seed_index, name, len(rs), b["CLU"],
            " ".join("%.4f" % b["control: " + c] for c in A.TransitionClu.CONTROLS)),
            flush=True)
    print("[repro] across-wells derangement %s (no fixed point: asserted)" % der, flush=True)
    out["seconds"] = time.time() - t0
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "REPRO_%s_%s_seed%d.json" % (cond, leg, seed_index))
    with open(p, "w") as fh:
        json.dump(out, fh, indent=1, default=R._json_default)
    print("[repro] wrote %s [%.1fs]" % (p, out["seconds"]), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2])
