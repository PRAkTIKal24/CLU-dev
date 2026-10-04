"""⭐ Synthetic plumbing smoke for the `x0` arm. ⛔ No number here is ever reported."""
import numpy as np, jax
from chlu.core.gen_landscape import (
    AtomLandscape, AtomAddressBook, build_atom_store, atom_bandwidths,
    fit_atom_address_book, fit_low_rank_coupling, launch_state,
)
import chlu.experiments.exp_gen_cell as M

rng = np.random.default_rng(0)
K, dk, N, n_train = 5, 6, 8, 400
D = K * dk
wc = [tuple(range(k * dk, (k + 1) * dk)) for k in range(K)]
Z = rng.normal(size=(n_train, D)).astype(np.float32)
cards = [3, 4]
Y = np.stack([rng.integers(0, c, n_train) for c in cards], axis=1)
store = build_atom_store(wc, (D,), Z, N, "kmeanspp", 0)
sigma = atom_bandwidths(store, 1.0, "nn_median")
book = fit_atom_address_book(store, Z, Y, cards, sigma, 1.0)
coup = fit_low_rank_coupling(store, Z[:100], 2, True)
land = AtomLandscape(store, sigma, coup, 1.0, 0.05)

R = np.stack([rng.integers(-1, c, 20) for c in cards], axis=1).astype(np.int64)

hb = M.x0_address_book_bytes(book)
assert hb["bytes"] == (K * N + sum(cards) * K * N) * 4, hb
print("address-book bytes OK:", hb["bytes"], "formula", hb["bytes_formula"])
print("store n_bytes:", store.n_bytes(), "= N*D*4 =", N * D * 4)
assert store.n_bytes() == N * D * 4

proof = M.x0_coupling_is_not_needed({"land": land, "store": store, "book": book}, R)
print("coupling-not-needed:", proof["bit_identical"], "coupling_needed =", proof["coupling_needed"])

xs = {}
for row in M.X0_ROWS:
    g = M.x0_gather(store, book, R, row, land=land, seed=3)
    xs[row] = g["x"]
    print(f"{row:18s} shape={g['x'].shape} distinct={np.unique(g['x'],axis=0).shape[0]} "
          f"mode={g['launch_mode']}")

# `full` returns one stored atom per well
A = [np.asarray(a, np.float32) for a in store.atoms]
for i in range(R.shape[0]):
    for k, w in enumerate(wc):
        assert np.any(np.all(A[k] == xs["full"][i, list(w)], axis=1)), (i, k)
print("⭐ `full` returns a STORED ATOM in every well: OK")

# `row_lookup` returns a WHOLE stored train row
rows = np.asarray(store.atom_rows, np.int64)
for i in range(R.shape[0]):
    assert np.any(np.all(Z[rows] == xs["row_lookup"][i], axis=1)), i
print("⭐ `row_lookup` returns a WHOLE stored train row: OK")

# atom_mean_lookup == the pinned posterior_mean launch
ref = np.asarray(launch_state(land, book, R, "posterior_mean", 0.5)["x0"], np.float32)
assert np.array_equal(ref, xs["atom_mean_lookup"])
print("⭐ `atom_mean_lookup` IS the pinned posterior_mean launch, bitwise: OK")

# atoms_shuffled uses the SAME indices as full
gf = M.x0_gather(store, book, R, "full", land=land, seed=3)
gs = M.x0_gather(store, book, R, "atoms_shuffled", land=land, seed=3)
assert np.array_equal(gf["atom_index_per_well"], gs["atom_index_per_well"])
assert not np.array_equal(gf["x"], gs["x"])
print("⭐ `atoms_shuffled`: SAME indices, DIFFERENT content: OK "
      f"(frac moved {gs['frac_indices_moved']:.2f})")

z = M.x0_zero_step_proof(xs["full"], xs["full"].reshape((R.shape[0], D)), (D,))
print("zero-step proof:", z["settling_steps"], z["decoder_input_is_the_launch_bitwise"])

ours = {0: 0.50, 1: 0.52, 2: 0.51}
print("_x0_pair vs weak:", M._x0_pair(ours, {0: .1, 1: .1, 2: .1}, 0.02)["verdict"])
print("_x0_pair vs clu-like:", M._x0_pair(ours, {0: .59, 1: .60, 2: .58}, 0.02)["verdict"])
print("_x0_pair tie-as-ceiling:", M._x0_pair(ours, {0: .505, 1: .515, 2: .512}, 0.02, True)["verdict"])
print("\n⭐ SMOKE PASSED — ⛔ no number above is a result.")
