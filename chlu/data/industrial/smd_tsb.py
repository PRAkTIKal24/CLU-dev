"""SMD (Server Machine Dataset) via TSB-AD's curated version — negative control.

Server telemetry (CPU/memory/network) has **no mechanical conservation
structure**: if CLU's energy-anomaly signal "wins" here, we are measuring
benchmark pathology, not physics benefit (program principle P1). Used strictly
as the falsifiability control, scored with the same VUS-PR pipeline.

We take SMD **as curated inside TSB-AD-M** (Liu & Paparrizos, NeurIPS 2024) to
dodge the raw benchmark's documented triviality/label criticisms: 22 series,
filename convention ``<idx>_SMD_id_<k>_Facility_tr_<n>_1st_<m>.csv`` where the
first ``n`` rows are the normal training prefix; columns are the feature
dimensions plus a final ``Label`` column.

Protocol: **per_unit_prefix** (TSB-AD's own): each series is fit on its own
normal prefix and scored on the remainder — a temporal, unit-internal split
(never window-random). Units (machines) remain separate; metrics are reported
per unit and aggregated per dataset.

Download: ``fetch()`` pulls the official TSB-AD-M archive
(https://www.thedatum.org/datasets/TSB-AD-M.zip, 540 MB, sha256 pinned below)
and extracts only the SMD members. Point ``root`` at an existing extraction to
skip the download.

LICENSE NOTE: TSB-AD (curation + harness) is Apache-2.0; the original SMD was
released with the OmniAnomaly code (MIT) by Su et al., KDD 2019.
"""

import re

import numpy as np

from chlu.data.industrial.base import (
    IndustrialDataset,
    UnitRecord,
    download_file,
    extract_zip,
    require_pandas,
)

TSB_AD_M_URL = "https://www.thedatum.org/datasets/TSB-AD-M.zip"
#: sha256 of TSB-AD-M.zip as downloaded/verified 2026-07-06 (540,383,983 bytes).
TSB_AD_M_SHA256 = "7de86ac27f30eeb48d833bb061055670e3f3de07defd995cf2bd5db10ccc9a0d"

FILENAME_RE = re.compile(
    r"(?P<idx>\d+)_SMD_id_(?P<mid>\d+)_(?P<domain>[A-Za-z]+)"
    r"_tr_(?P<tr>\d+)_1st_(?P<first>\d+)\.csv$"
)


class SMDTSB(IndustrialDataset):
    """SMD-as-curated-by-TSB-AD loader (unit = one server machine series)."""

    name = "smd_tsb"
    label_kind = "point"
    protocol = "per_unit_prefix"
    license_note = (
        "TSB-AD curation: Apache-2.0; original SMD (OmniAnomaly, KDD'19): MIT."
    )
    citation = "Su et al. KDD 2019 (SMD); Liu & Paparrizos NeurIPS 2024 (TSB-AD)"

    def _files(self) -> dict:
        files = {}
        for path in sorted(self.root.rglob("*_SMD_*.csv")):
            m = FILENAME_RE.search(path.name)
            if m:
                files[f"SMD_id_{int(m.group('mid')):02d}"] = (path, int(m.group("tr")))
        return files

    def is_available(self) -> bool:
        return bool(self._files())

    def fetch(self) -> None:
        """Download TSB-AD-M.zip and extract only the SMD members."""
        self.root.mkdir(parents=True, exist_ok=True)
        zip_path = download_file(
            TSB_AD_M_URL, self.root / "TSB-AD-M.zip", sha256=TSB_AD_M_SHA256
        )
        extract_zip(zip_path, self.root, member_filter="_SMD_")

    def unit_ids(self) -> tuple:
        return tuple(self._files())

    def train_ids(self) -> tuple:
        """per_unit_prefix: training data is each unit's own prefix."""
        return ()

    def test_ids(self) -> tuple:
        return self.unit_ids()

    def load_unit(self, unit_id: str) -> UnitRecord:
        pd = require_pandas("SMD/TSB-AD loader")
        files = self._files()
        if unit_id not in files:
            raise FileNotFoundError(f"unknown SMD unit {unit_id!r}")
        path, train_len = files[unit_id]
        df = pd.read_csv(path).dropna()
        if "Label" not in df.columns:
            raise ValueError(f"{path} lacks the TSB-AD 'Label' column")
        feature_cols = [c for c in df.columns if c != "Label"]
        return UnitRecord(
            unit_id=unit_id,
            data=df[feature_cols].to_numpy(dtype=np.float32),
            channels=tuple(feature_cols),
            point_labels=df["Label"].to_numpy(dtype=np.int8),
            episode_label=int(df["Label"].any()),
            fault_class=None,
            condition=None,
            sampling_rate_hz=None,  # 1-minute telemetry, not stated in curation
            meta={"path": str(path), "train_len": train_len},
        )
