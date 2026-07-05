"""MIMII acoustic dataset — loader SKELETON (stretch item; full runs on CSF3).

Malfunctioning IndustrialMachine Investigation and Inspection: 8-channel
microphone recordings (16 kHz) of fans/pumps/valves/slide rails at SNR
-6/0/+6 dB, with real induced damage. Episode-labelled (normal/abnormal wav
segments) — DCASE 2020 Task 2 lineage.

SIZE/LICENSE: ~100 GB total on Zenodo (record 3384388), **CC BY-SA 4.0**.
No automatic download here — this laptop-side skeleton only walks an existing
partial local copy (e.g. one machine type / one SNR) laid out as extracted
from the official zips::

    <root>/<snr>/<machine>/<machine_id>/{normal,abnormal}/*.wav
    e.g.  6dB/fan/id_00/normal/00000000.wav

Full-dataset fetching + featurization (log-mel etc.) is deliberately deferred
to the CSF3 runbook; wav decoding uses scipy (already a core dependency).

Citation: Purohit et al., "MIMII Dataset: Sound Dataset for Malfunctioning
Industrial Machine Investigation and Inspection", DCASE 2019 Workshop
(Zenodo record 3384388).
"""

import numpy as np

from chlu.data.industrial.base import IndustrialDataset, UnitRecord

ZENODO_RECORD_URL = "https://zenodo.org/records/3384388"
MACHINE_TYPES = ("fan", "pump", "slider", "valve")
SNRS = ("min6dB", "0dB", "6dB")


class MIMII(IndustrialDataset):
    """MIMII skeleton loader (unit = one wav segment; episode-labelled)."""

    name = "mimii"
    label_kind = "episode"
    protocol = "cross_unit"
    license_note = (
        f"CC BY-SA 4.0, ~100 GB on Zenodo ({ZENODO_RECORD_URL}); download "
        "manually / on CSF3 — no automatic fetch from this skeleton."
    )
    citation = "Purohit et al., DCASE 2019 Workshop; Zenodo 3384388"

    def is_available(self) -> bool:
        return self.root.exists() and any(self.root.rglob("*.wav"))

    def fetch(self) -> None:
        raise NotImplementedError(
            "MIMII is ~100 GB (CC BY-SA 4.0); fetch it on CSF3 from "
            f"{ZENODO_RECORD_URL} and point root= at the extraction. "
            "This skeleton only reads existing local subsets."
        )

    def unit_ids(self) -> tuple:
        return tuple(
            sorted(
                str(p.relative_to(self.root))[: -len(".wav")]
                for p in self.root.rglob("*.wav")
            )
        )

    def train_ids(self) -> tuple:
        """Convention: normal wavs train (unsupervised protocol)."""
        return tuple(u for u in self.unit_ids() if "/normal/" in u)

    def test_ids(self) -> tuple:
        return tuple(u for u in self.unit_ids() if "/abnormal/" in u)

    def load_unit(self, unit_id: str) -> UnitRecord:
        from scipy.io import wavfile

        path = self.root / f"{unit_id}.wav"
        if not path.exists():
            raise FileNotFoundError(f"unknown MIMII unit {unit_id!r}")
        rate, wav = wavfile.read(path)
        if wav.ndim == 1:
            wav = wav[:, None]
        was_int = np.issubdtype(wav.dtype, np.integer)
        wav = wav.astype(np.float32)
        if was_int:
            wav /= 32768.0  # 16-bit PCM to [-1, 1]
        parts = unit_id.split("/")
        return UnitRecord(
            unit_id=unit_id,
            data=wav,
            channels=tuple(f"mic_{i}" for i in range(wav.shape[1])),
            point_labels=None,
            episode_label=int("abnormal" in parts),
            fault_class="abnormal" if "abnormal" in parts else "normal",
            condition="/".join(parts[:-2]) if len(parts) > 2 else None,
            sampling_rate_hz=float(rate),
            meta={"path": str(path)},
        )
