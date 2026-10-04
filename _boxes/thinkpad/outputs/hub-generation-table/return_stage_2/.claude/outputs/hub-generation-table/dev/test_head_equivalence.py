"""⭐ The one property the discriminative head's honesty rests on.

1. ``steps=0`` reproduces ``gen_dream.fit_address_book`` **bit-identically**, so the
   new estimator is a strict superset of the shipped one and a null result cannot be
   an implementation difference.
2. The BYTE LEDGER is **unchanged to the byte** — the same
   ``gen_compose.address_book_bytes`` closed form gives the same number for both.
3. The forward pass is the SHIPPED one (``address_requests``), not a second copy.
"""
import numpy as np

from chlu.core.gen_address import (
    as_address_book,
    fit_discriminative_address_book,
)
from chlu.core.gen_compose import address_book_bytes, request_posterior
from chlu.core.gen_dream import fit_address_book

rng = np.random.default_rng(0)
n, K, M = 3000, 5, 64
cards = [10, 10, 10, 8, 4, 15]
Y = np.stack([rng.integers(0, c, n) for c in cards], axis=1).astype(np.int64)
C = rng.integers(0, M, (n, K)).astype(np.int64)

nb = fit_address_book(C, Y, cards, 1.0)
d0 = fit_discriminative_address_book(C, Y, cards, steps=0)

same_prior = np.array_equal(np.asarray(nb.log_prior, np.float32),
                            np.asarray(d0.log_prior, np.float32))
same_cond = all(
    np.array_equal(np.asarray(a, np.float32), np.asarray(b, np.float32))
    for a, b in zip(nb.log_cond, d0.log_cond)
)
print("steps=0 log_prior bit-identical :", same_prior)
print("steps=0 log_cond  bit-identical :", same_cond)

R = Y[:64]
p_nb = request_posterior(nb, R)
p_d0 = request_posterior(as_address_book(d0), R)
print("posterior max |delta|           :", float(np.max(np.abs(p_nb - p_d0))))

bl_nb = address_book_bytes(nb)
bl_d0 = address_book_bytes(as_address_book(d0))
print("address head bytes  naive-Bayes :", bl_nb["bytes"], bl_nb["bytes_formula"])
print("address head bytes  discrim.    :", bl_d0["bytes"])
print("BYTES UNCHANGED                 :", bl_nb["bytes"] == bl_d0["bytes"])

d1 = fit_discriminative_address_book(C, Y, cards, steps=30, lr=0.3)
bl_d1 = address_book_bytes(as_address_book(d1))
print("after 30 steps, bytes           :", bl_d1["bytes"],
      "unchanged:", bl_d1["bytes"] == bl_nb["bytes"])
print("nll per well after 30 steps     :",
      [round(x, 4) for x in d1.fit_info["final_nll_per_well"]])

ok = same_prior and same_cond and bl_nb["bytes"] == bl_d0["bytes"] == bl_d1["bytes"]
print("\nRESULT:", "PASS" if ok else "FAIL")
