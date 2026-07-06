"""SKAB — Skoltech Anomaly Benchmark (water-circulation testbed).

Real rotating-machinery rig (pump, valves, shaft); labelled faults are
physical energy-signature excursions (cavitation, shaft imbalance, reduced
motor power, partial valve closure). Tiny — the iteration/ablation benchmark.

Structure (v0.9, pinned commit below): 8 sensor channels at ~1 Hz;
``anomaly-free/anomaly-free.csv`` is the normal-only training run; 34 labelled
experiment runs live in ``valve1/ valve2/ other/`` with per-timestep
``anomaly`` and ``changepoint`` columns. Each CSV file is one physical
experiment run = one **unit** (the split key).

LICENSE NOTE: the SKAB repository (code AND data) is **GPL-3.0**. Data files
are therefore never vendored into this repo; ``fetch()`` downloads the pinned
upstream archive to the user's data root at the user's request, or point
``root`` at an existing user-downloaded copy. Keep SKAB data out of tracked
files.

Citation: Katser & Kozitsin, "Skoltech Anomaly Benchmark (SKAB)", 2020,
https://github.com/waico/SKAB.
"""

import numpy as np

from chlu.data.industrial.base import (
    IndustrialDataset,
    UnitRecord,
    download_file,
    extract_zip,
    require_pandas,
)

#: Pinned upstream commit (master @ 2024-08-11) and its archive checksum.
SKAB_COMMIT = "b2c0d46c2971dcbfe71e26087b6d231998bb91c2"
SKAB_ZIP_URL = f"https://github.com/waico/SKAB/archive/{SKAB_COMMIT}.zip"
SKAB_ZIP_SHA256 = "45ac11b460e495ba2c1301c3f8e871b688b5ecfa6cd0770b4225594fe45efc80"

CHANNELS = (
    "Accelerometer1RMS",
    "Accelerometer2RMS",
    "Current",
    "Pressure",
    "Temperature",
    "Thermocouple",
    "Voltage",
    "Volume Flow RateRMS",
)
GROUPS = ("anomaly-free", "valve1", "valve2", "other")
SAMPLING_RATE_HZ = 1.0  # signals recorded each second (SKAB README)


class SKAB(IndustrialDataset):
    """SKAB loader (unit = one experiment-run CSV)."""

    name = "skab"
    label_kind = "point"
    protocol = "cross_unit"
    license_note = (
        "SKAB is GPL-3.0 (code and data): load from a user-downloaded copy; "
        "never vendor/redistribute the files in this repository."
    )
    citation = "Katser & Kozitsin 2020, https://github.com/waico/SKAB"

    def is_available(self) -> bool:
        return (self.root / "anomaly-free" / "anomaly-free.csv").exists()

    def fetch(self) -> None:
        """Download the pinned SKAB archive and lay out ``<root>/<group>/*.csv``."""
        self.root.mkdir(parents=True, exist_ok=True)
        zip_path = download_file(
            SKAB_ZIP_URL, self.root / "skab_src.zip", sha256=SKAB_ZIP_SHA256
        )
        tmp = extract_zip(zip_path, self.root / "_extract", member_filter="/data/")
        data_dir = tmp / f"SKAB-{SKAB_COMMIT}" / "data"
        for group in GROUPS:
            src = data_dir / group
            dst = self.root / group
            if src.exists() and not dst.exists():
                src.rename(dst)

    def unit_ids(self) -> tuple:
        ids = []
        for group in GROUPS:
            for csv in sorted((self.root / group).glob("*.csv")):
                ids.append(f"{group}/{csv.stem}")
        return tuple(sorted(ids))

    def train_ids(self) -> tuple:
        """Canonical train = the anomaly-free run (normal-only protocol)."""
        return ("anomaly-free/anomaly-free",)

    def test_ids(self) -> tuple:
        """Canonical test = the 34 labelled experiment runs."""
        return tuple(u for u in self.unit_ids() if not u.startswith("anomaly-free"))

    def load_unit(self, unit_id: str) -> UnitRecord:
        pd = require_pandas("SKAB loader")
        group, stem = unit_id.split("/", 1)
        path = self.root / group / f"{stem}.csv"
        if not path.exists():
            raise FileNotFoundError(f"unknown SKAB unit {unit_id!r} ({path})")
        df = pd.read_csv(path, sep=";")
        missing = [c for c in CHANNELS if c not in df.columns]
        if missing:
            raise ValueError(f"{path} lacks channels {missing}")
        data = df[list(CHANNELS)].to_numpy(dtype=np.float32)
        labels = (
            df["anomaly"].to_numpy(dtype=np.int8) if "anomaly" in df.columns else None
        )
        meta = {"path": str(path)}
        if "changepoint" in df.columns:
            meta["changepoint"] = df["changepoint"].to_numpy(dtype=np.int8)
        return UnitRecord(
            unit_id=unit_id,
            data=data,
            channels=CHANNELS,
            point_labels=labels,
            episode_label=int(labels.any()) if labels is not None else 0,
            fault_class=None if group == "anomaly-free" else group,
            condition=group,
            sampling_rate_hz=SAMPLING_RATE_HZ,
            meta=meta,
        )


def _records_from_dataframe(unit_id: str, df, group: str) -> UnitRecord:
    """Build a UnitRecord from an in-memory dataframe (test seam)."""
    data = df[list(CHANNELS)].to_numpy(dtype=np.float32)
    labels = df["anomaly"].to_numpy(dtype=np.int8) if "anomaly" in df.columns else None
    return UnitRecord(
        unit_id=unit_id,
        data=data,
        channels=CHANNELS,
        point_labels=labels,
        episode_label=int(labels.any()) if labels is not None else 0,
        condition=group,
        sampling_rate_hz=SAMPLING_RATE_HZ,
    )
