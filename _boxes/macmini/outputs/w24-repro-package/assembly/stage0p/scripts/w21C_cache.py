"""w21 hub/maze-t-clu-improve -- cache the FROZEN VAL instrument once per seed.

    python scripts/w21_cache.py <seed> <out_pkl>

Everything here is the w19 runner's own path, byte-for-byte: W16.banks, make_holdout,
split_rows, cap_split, build_codes, the frozen codec recipe (A.train_codec, latent 128,
2,500 steps, seed = seed_index).  It is cached so CLU candidates (closed form) can be swept
on VAL without re-training the codec each time.  The w9 dynamics scalars (``w9feat.dyn``,
31 wide at k = 4; the observation's dyn block, NO learned encoder involved) are cached per
row so a candidate may consume them.  Nothing else of the observation is cached.
"""
import os, sys, pickle, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import chlu.exp_w15_cmaze_t as T
import chlu.exp_w15_cmaze_t_arms as A
import w16_cmaze_t as W16
from w9_harness import features as w9feat

K_COLOURS = 4


def main(seed, out, leg="val", latent=128, codec_steps=2500):
    t0 = time.time()
    base, ev_base, (tr_insts, tr_rows, tr_nat, tr_meta), (va_insts, va_rows, va_nat, va_meta) = \
        W16.banks(seed, leg)
    ho = T.make_holdout(K_COLOURS, seed=seed)
    train_rows = T.split_rows(tr_rows, ho, K_COLOURS).get("seen", [])
    va_split = W16.cap_split(T.split_rows(va_rows, ho, K_COLOURS), seed)
    Ctr = A.build_codes(train_rows, K_COLOURS)
    Ccodec = np.concatenate([Ctr, T.enumerate_code_alphabet(K_COLOURS)], axis=0)
    codec, crep = A.train_codec(Ccodec, K_COLOURS, latent=latent, steps=codec_steps, seed=seed)
    Ztr = A.codec_encode(codec, Ctr)
    dyn = lambda rows, insts: np.stack([w9feat.dyn(insts[r.maze], r.state()) for r in rows]).astype(np.float32)
    d = {"seed": seed, "leg": leg, "Ztr": Ztr, "Ctr": Ctr, "Wtr": A.build_words(train_rows),
         "Dtr": dyn(train_rows, tr_insts), "codec": codec, "va_insts": va_insts,
         "va_split": va_split, "Wva": {n: A.build_words(rs) for n, rs in va_split.items() if rs},
         "Dva": {n: dyn(rs, va_insts) for n, rs in va_split.items() if rs},
         "ho": ho, "seconds": time.time() - t0}
    with open(out, "wb") as f:
        pickle.dump(d, f, protocol=4)
    print("cached seed %d leg %s: train %d rows, val %s, dyn %d wide [%.1fs]"
          % (seed, leg, len(train_rows), {n: len(r) for n, r in va_split.items()},
             d["Dtr"].shape[1], time.time() - t0), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], *(sys.argv[3:4] or []))
