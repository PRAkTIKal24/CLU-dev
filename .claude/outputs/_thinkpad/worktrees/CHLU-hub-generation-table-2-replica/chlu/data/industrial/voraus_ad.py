"""voraus-AD — anomaly detection on a 6-axis industrial robot (pick & place).

The F3 headline anomaly benchmark: rigid-body robot dynamics are literally
Hamiltonian, and the 12 anomaly categories (collisions, load changes,
actuation faults, ...) are energy/dynamics excursions. ~130 machine signals
(motor currents/torques/positions per axis + robot-level electrical values).

Labelling is **per episode** (column ``sample`` indexes one pick-and-place
cycle; ``anomaly`` is one boolean per episode) — there is no per-timestep
ground truth, so the evaluation convention is episode-level AUROC (as in the
dataset paper), not VUS.

Canonical split (matches the reference implementation
``vorausrobotik/voraus-ad-dataset``): train = episodes recorded under setting
``PRE_A`` (all normal); test = every other episode.

Files (single parquet per variant):
- 100 Hz: ~1.1 GB disk / ~2.5 GB RAM fully loaded,
- 500 Hz: ~5.3 GB disk / ~12.5 GB RAM (use CSF3, not a laptop).
Use ``columns=`` to project a channel subset — memory scales accordingly.

LICENSE NOTE: the datasets are **CC BY-NC-SA 4.0** (non-commercial, share
alike; fine for papers with attribution). The reference repo's code is MIT.

Citation: Brockmann, Rudolph, Rosenhahn, Wandt, "The voraus-AD Dataset for
Anomaly Detection in Robot Applications", IEEE T-RO 2024 (arXiv:2311.04765).
"""

import numpy as np

from chlu.data.industrial.base import (
    IndustrialDataset,
    UnitRecord,
    download_file,
    require_pandas,
)

URLS = {
    "100hz": "https://media.vorausrobotik.com/voraus-ad-dataset-100hz.parquet",
    "500hz": "https://media.vorausrobotik.com/voraus-ad-dataset-500hz.parquet",
}
#: sha256 of the 100 Hz parquet as downloaded/verified 2026-07-06.
SHA256 = {
    "100hz": "c90ab1c78af52651b954d41787f7e89d750f0a128b57600b0e5ceec22621f704",
    "500hz": None,  # record on first CSF3 download
}

#: Time-invariant per-episode meta columns (reference loader `meta_constant`).
META_CONSTANT = ("sample", "anomaly", "category", "setting")
#: Time-varying meta columns (excluded from machine data).
META_VARIABLE = ("time", "action", "active")
META_COLUMNS = META_CONSTANT + META_VARIABLE

#: ``setting`` value marking the normal-only training episodes (Variant.PRE_A).
TRAIN_SETTING_PRE_A = 72

#: Anomaly categories (paper Table; 12 = normal operation).
CATEGORY_NAMES = {
    0: "AXIS_FRICTION",
    1: "AXIS_WEIGHT",
    2: "COLLISION_FOAM",
    3: "COLLISION_CABLE",
    4: "COLLISION_CARTON",
    5: "MISS_CAN",
    6: "LOSE_CAN",
    7: "CAN_WEIGHT",
    8: "ENTANGLED",
    9: "INVALID_POSITION",
    10: "MOTOR_COMMUTATION",
    11: "WOBBLING_STATION",
    12: "NORMAL_OPERATION",
}

SAMPLING_RATE_HZ = {"100hz": 100.0, "500hz": 500.0}


class VorausAD(IndustrialDataset):
    """voraus-AD loader (unit = one pick-and-place episode).

    Args:
        root: Directory containing ``voraus-ad-dataset-<variant>.parquet``.
        download: Fetch the parquet from the official host if missing.
        variant: "100hz" (default) or "500hz".
        columns: Optional machine-signal subset to load (memory control).
    """

    name = "voraus_ad"
    label_kind = "episode"
    protocol = "cross_unit"
    license_note = (
        "voraus-AD data is CC BY-NC-SA 4.0 (non-commercial, share-alike): "
        "academic use with attribution is fine; do not redistribute in-repo."
    )
    citation = "Brockmann et al., IEEE T-RO 2024, arXiv:2311.04765"

    def __init__(self, root=None, download=False, variant="100hz", columns=None):
        if variant not in URLS:
            raise ValueError(f"variant must be one of {sorted(URLS)}, got {variant}")
        self.variant = variant
        self.columns = tuple(columns) if columns is not None else None
        self._index = None  # per-episode meta table
        self._frames = None  # episode -> (T, C) ndarray cache
        self._channels = None
        super().__init__(root=root, download=download)

    @property
    def parquet_path(self):
        return self.root / f"voraus-ad-dataset-{self.variant}.parquet"

    def is_available(self) -> bool:
        return self.parquet_path.exists()

    def fetch(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        download_file(
            URLS[self.variant], self.parquet_path, sha256=SHA256.get(self.variant)
        )

    # -- lazy episode index (reads only 4 tiny meta columns) -----------------
    def _load_index(self):
        if self._index is None:
            pd = require_pandas("voraus-AD loader")
            meta = pd.read_parquet(self.parquet_path, columns=list(META_CONSTANT))
            self._index = (
                meta.drop_duplicates(subset="sample").set_index("sample").sort_index()
            )
        return self._index

    def _load_frames(self):
        """Load machine-signal frames grouped per episode (one parquet scan)."""
        if self._frames is None:
            pd = require_pandas("voraus-AD loader")
            if self.columns is not None:
                cols = list(dict.fromkeys(list(self.columns) + ["sample"]))
                df = pd.read_parquet(self.parquet_path, columns=cols)
            else:
                df = pd.read_parquet(self.parquet_path)
                df = df.drop(
                    columns=[
                        c for c in META_COLUMNS if c != "sample" and c in df.columns
                    ]
                )
            self._channels = tuple(c for c in df.columns if c != "sample")
            self._frames = {
                int(sid): g.drop(columns="sample").to_numpy(dtype=np.float32)
                for sid, g in df.groupby("sample", sort=True)
            }
            del df
        return self._frames

    def unit_ids(self) -> tuple:
        return tuple(str(s) for s in self._load_index().index)

    def train_ids(self) -> tuple:
        idx = self._load_index()
        return tuple(str(s) for s in idx.index[idx["setting"] == TRAIN_SETTING_PRE_A])

    def test_ids(self) -> tuple:
        idx = self._load_index()
        return tuple(str(s) for s in idx.index[idx["setting"] != TRAIN_SETTING_PRE_A])

    def load_unit(self, unit_id: str) -> UnitRecord:
        sid = int(unit_id)
        row = self._load_index().loc[sid]
        frames = self._load_frames()
        if sid not in frames:
            raise FileNotFoundError(f"unknown voraus-AD episode {unit_id!r}")
        category = int(row["category"])
        return UnitRecord(
            unit_id=unit_id,
            data=frames[sid],
            channels=self._channels,
            point_labels=None,  # episode-labelled dataset
            episode_label=int(bool(row["anomaly"])),
            fault_class=CATEGORY_NAMES.get(category, f"category_{category}"),
            condition=f"setting_{int(row['setting'])}",
            sampling_rate_hz=SAMPLING_RATE_HZ[self.variant],
            meta={"setting": int(row["setting"]), "category": category},
        )


#: The 6 robot-axis JOINT ANGLES θ_j (the literal T^n = U(1)^n coordinates).
#: These are the physical joint positions (not the gear-reduced ``motor_position``
#: nor the commanded ``target_position``); each is one U(1) register.
JOINT_ANGLE_COLUMNS = tuple(f"joint_position_{j}" for j in range(1, 7))


def embed_joint_angles(data: np.ndarray, channels) -> tuple:
    """Embed the joint-angle channels of ``data`` as ``(cos θ, sin θ)`` pairs.

    Returns ``(new_data, new_channels, n_so2)``: the ``(cos, sin)`` pairs for
    every present ``joint_position_j`` come FIRST (axis order), then all other
    channels raw. ``n_so2`` = number of joint-angle columns found. Pure/JAX-free
    so it is unit-testable without loading the parquet.
    """
    chans = list(channels)
    angle_idx = [chans.index(c) for c in JOINT_ANGLE_COLUMNS if c in chans]
    aux_idx = [i for i in range(len(chans)) if i not in set(angle_idx)]
    cols, names = [], []
    for k, ci in enumerate(angle_idx, start=1):
        theta = data[:, ci]
        cols.append(np.cos(theta))
        cols.append(np.sin(theta))
        names.extend([f"cos_joint_{k}", f"sin_joint_{k}"])
    for ci in aux_idx:
        cols.append(data[:, ci])
        names.append(chans[ci])
    new_data = np.stack(cols, axis=1).astype(np.float32)
    return new_data, tuple(names), len(angle_idx)


class VorausTorusAD(VorausAD):
    """voraus-AD with the LITERAL joint-angle -> torus-coset embedding (G7b).

    The theory's falsifiable prediction: match the CLU's coset to the data's own
    topology. voraus joint space is ``T^6 = U(1)^6``; this loader embeds each
    joint angle ``θ_j`` on its ring as ``(cos θ_j, sin θ_j)`` and lays the six
    ``(cos, sin)`` pairs FIRST, so each pair feeds one dim-2 ``so2_invariant``
    unit whose ``T^1`` coset *is* that joint's ``U(1)`` (see
    ``CLULatticeConfig(layout="literal")`` + ``_build_literal_lattice``). Every
    other machine signal (velocities, torques, currents, ...) is kept raw and
    appended as auxiliary channels — "angles-on-the-torus, everything else
    auxiliary" (the clean first pass; Head 2026-07-19).

    ``n_so2_units`` (= number of joint-angle columns actually present, ≤ 6) is
    what the CLI forwards into the lattice config so the model routes the first
    ``2 * n_so2_units`` channels to coset units. Angles are assumed to be in
    radians (robotics convention); ``(cos, sin)`` lives on the unit circle for
    any real θ regardless, so the ring embedding is well-defined even for the
    limited-range sweeps of a pick-and-place cycle.

    NOTE (scaling caveat, honest): the harness fits a per-channel
    ``StandardScaler`` on the flattened windows, so the CLU (like the
    baselines, on the IDENTICAL scaled data — the fair protocol) sees an affine
    map of each ``(cos, sin)`` circle to an off-origin ellipse; the per-unit
    SO(2) invariance is therefore only approximate on scaled data. The
    LATTICE-level U(1) coupling structure and the pre-registered topology-match
    control (``shuffle_angles``) are scaling-invariant and carry the
    falsification lever.
    """

    name = "voraus_ad"  # same underlying data/split/labels as VorausAD

    @property
    def n_so2_units(self) -> int:
        """Number of joint-angle (so2 coset) units this loader will emit."""
        chans = self._resolve_channels()
        return sum(1 for c in JOINT_ANGLE_COLUMNS if c in chans)

    def _resolve_channels(self) -> tuple:
        """Machine-signal channel names in load order (triggers one lazy scan)."""
        if self._channels is None:
            self._load_frames()
        return self._channels

    def load_unit(self, unit_id: str):
        rec = super().load_unit(unit_id)
        new_data, names, n_so2 = embed_joint_angles(rec.data, rec.channels)

        from dataclasses import replace

        return replace(
            rec,
            data=new_data,
            channels=names,
            meta={**rec.meta, "n_so2_units": n_so2},
        )
