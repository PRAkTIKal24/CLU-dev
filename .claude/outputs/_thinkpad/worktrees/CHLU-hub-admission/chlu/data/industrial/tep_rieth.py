"""Tennessee Eastman Process — Rieth et al. 2017 edition (CC0, label-perfect).

Simulated chemical plant, 52 process variables (xmeas_1..41 measurements +
xmv_1..11 manipulated), 20 fault classes + fault-free, 500 independently
seeded simulation runs per class. Labels are perfect by construction
(simulation), which makes this the review-proof scale benchmark.

Units and splits: one **unit = one simulation run** (a ``simulationRun`` seed
within a fault class within a file) — the binding "split by simulation seed"
rule. Canonical AD protocol: train on **fault-free training** runs
(normal-only); test on the **testing** runs (fault-free + faulty).

Fault onset (Rieth et al. dataset description; 3-minute sampling):
- training runs: 25 h = 500 samples, faults introduced 1 h in  -> sample > 20
  is anomalous;
- testing runs: 48 h = 960 samples, faults introduced 8 h in   -> sample > 160
  is anomalous.
Fault-free runs are all-normal (degenerate for per-run metrics — the harness
reports NaN for them and aggregates over valid runs).

Files (Harvard Dataverse, DOI 10.7910/DVN/6C3JR1, CC0; RData format, needs
``pyreadr`` — install the eval extra):

===================================  ==========  ======  =========
file                                 dataverse    size    md5
===================================  ==========  ======  =========
TEP_FaultFree_Training.RData         3031241     23.5MB  ec126484534331f85001d8c4ebce6d17
TEP_FaultFree_Testing.RData          3031240     45.1MB  38ad9810fc871026157086ae2c2f0ee9
TEP_Faulty_Training.RData            3031242     471MB   c5f594d54c47e620ff877feb58407fda
TEP_Faulty_Testing.RData             3031243     798MB   556bdb64c83021bc0c5f92e427753565
===================================  ==========  ======  =========

MEMORY NOTE: pyreadr materialises a whole RData file; the faulty testing file
expands to ~4 GB in RAM. Laptop smoke-tests should stick to the fault-free
files (defaults of ``fetch()``); full faulty runs belong on CSF3.

Citation: Rieth, Amsel, Tran, Cook, "Additional Tennessee Eastman Process
Simulation Data for Anomaly Detection Evaluation", Harvard Dataverse 2017,
doi:10.7910/DVN/6C3JR1.
"""

import numpy as np

from chlu.data.industrial.base import IndustrialDataset, UnitRecord, download_file

DATAVERSE_ACCESS = "https://dataverse.harvard.edu/api/access/datafile/{id}"

#: file-key -> (filename, dataverse id, md5, bytes) — from the Dataverse API
#: listing for doi:10.7910/DVN/6C3JR1 (retrieved 2026-07-06).
FILES = {
    "fault_free_training": (
        "TEP_FaultFree_Training.RData",
        3031241,
        "ec126484534331f85001d8c4ebce6d17",
        24_678_017,
    ),
    "fault_free_testing": (
        "TEP_FaultFree_Testing.RData",
        3031240,
        "38ad9810fc871026157086ae2c2f0ee9",
        47_327_663,
    ),
    "faulty_training": (
        "TEP_Faulty_Training.RData",
        3031242,
        "c5f594d54c47e620ff877feb58407fda",
        494_063_194,
    ),
    "faulty_testing": (
        "TEP_Faulty_Testing.RData",
        3031243,
        "556bdb64c83021bc0c5f92e427753565",
        836_882_037,
    ),
}

#: Fault-onset sample indices (1-based ``sample`` column; see module docstring).
TRAIN_ONSET_SAMPLE = 20
TEST_ONSET_SAMPLE = 160

CHANNELS = tuple(f"xmeas_{i}" for i in range(1, 42)) + tuple(
    f"xmv_{i}" for i in range(1, 12)
)
SAMPLING_RATE_HZ = 1.0 / 180.0  # one sample every 3 minutes


class TEPRieth(IndustrialDataset):
    """TEP-Rieth loader (unit = one seeded simulation run).

    Args:
        root: Directory holding the ``TEP_*.RData`` files.
        download: Fetch the two fault-free files (laptop-safe defaults) if
            nothing is present; faulty files via ``fetch(keys=...)``.
        faults: Optional subset of fault numbers to expose (0 = fault-free),
            e.g. ``(0, 1, 4)`` — trims memory and runtime.
        runs_per_fault: Optional cap on simulation runs exposed per
            (file, fault) pair — smoke-test control, applied deterministically
            to the lowest run indices.
    """

    name = "tep_rieth"
    label_kind = "point"
    protocol = "cross_unit"
    license_note = "CC0 (Harvard Dataverse doi:10.7910/DVN/6C3JR1)."
    citation = "Rieth et al. 2017, doi:10.7910/DVN/6C3JR1"

    def __init__(self, root=None, download=False, faults=None, runs_per_fault=None):
        self.faults = tuple(faults) if faults is not None else None
        self.runs_per_fault = runs_per_fault
        self._frames = {}  # file_key -> pandas dataframe
        self._units = None
        super().__init__(root=root, download=download)

    def is_available(self) -> bool:
        return any((self.root / fn).exists() for fn, _, _, _ in FILES.values())

    def available_keys(self) -> tuple:
        return tuple(
            k for k, (fn, _, _, _) in FILES.items() if (self.root / fn).exists()
        )

    def fetch(self, keys=("fault_free_training", "fault_free_testing")) -> None:
        """Download the requested files (defaults are the laptop-safe pair)."""
        self.root.mkdir(parents=True, exist_ok=True)
        for key in keys:
            fn, file_id, md5, _ = FILES[key]
            download_file(DATAVERSE_ACCESS.format(id=file_id), self.root / fn, md5=md5)

    # -- RData loading --------------------------------------------------------
    def _frame(self, key):
        if key not in self._frames:
            try:
                import pyreadr  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "TEP-Rieth loader needs pyreadr (uv sync --extra eval)"
                ) from exc
            fn = FILES[key][0]
            result = pyreadr.read_r(str(self.root / fn))
            self._frames[key] = next(iter(result.values()))
        return self._frames[key]

    def _unit_table(self):
        """(unit_id -> (file_key, fault, run)) for all available files."""
        if self._units is None:
            units = {}
            for key in self.available_keys():
                df = self._frame(key)
                pairs = df[["faultNumber", "simulationRun"]].drop_duplicates()
                for fault, run in pairs.itertuples(index=False):
                    fault, run = int(fault), int(run)
                    if self.faults is not None and fault not in self.faults:
                        continue
                    if self.runs_per_fault is not None and run > self.runs_per_fault:
                        continue
                    units[f"{key}:fault{fault:02d}:run{run:03d}"] = (key, fault, run)
            self._units = dict(sorted(units.items()))
        return self._units

    def unit_ids(self) -> tuple:
        return tuple(self._unit_table())

    def train_ids(self) -> tuple:
        """Canonical train = fault-free TRAINING runs (normal-only)."""
        return tuple(
            u
            for u, (k, _, _) in self._unit_table().items()
            if k == "fault_free_training"
        )

    def test_ids(self) -> tuple:
        """Canonical test = all TESTING runs (fault-free + faulty)."""
        return tuple(
            u for u, (k, _, _) in self._unit_table().items() if k.endswith("_testing")
        )

    def load_unit(self, unit_id: str) -> UnitRecord:
        table = self._unit_table()
        if unit_id not in table:
            raise FileNotFoundError(f"unknown TEP unit {unit_id!r}")
        key, fault, run = table[unit_id]
        df = self._frame(key)
        sub = df[(df["faultNumber"] == fault) & (df["simulationRun"] == run)]
        sub = sub.sort_values("sample")
        return _record_from_run(unit_id, sub, key, fault, run)


def _record_from_run(unit_id, run_df, file_key, fault, run) -> UnitRecord:
    """Build the UnitRecord for one simulation run (test seam)."""
    data = run_df[list(CHANNELS)].to_numpy(dtype=np.float32)
    onset = TRAIN_ONSET_SAMPLE if file_key.endswith("_training") else TEST_ONSET_SAMPLE
    samples = run_df["sample"].to_numpy()
    labels = (
        ((samples > onset) & (fault > 0)).astype(np.int8)
        if len(samples)
        else np.zeros(0, np.int8)
    )
    return UnitRecord(
        unit_id=unit_id,
        data=data,
        channels=CHANNELS,
        point_labels=labels,
        episode_label=int(fault > 0),
        fault_class=f"fault{fault:02d}",
        condition=f"fault{fault:02d}",
        sampling_rate_hz=SAMPLING_RATE_HZ,
        meta={
            "file": file_key,
            "fault": fault,
            "simulation_run": run,
            "onset_sample": onset if fault > 0 else None,
        },
    )
