"""Loader tests on synthetic fixtures (no network, no real downloads).

Real-data smoke runs are performed out-of-band (see the f2-eval-harness
report); these tests pin the parsing/schema/split logic.
"""

import numpy as np
import pytest

pd = pytest.importorskip("pandas", reason="loaders need the eval extra")

from chlu.data.industrial import DATASET_REGISTRY, get_dataset  # noqa: E402
from chlu.data.industrial.skab import CHANNELS as SKAB_CHANNELS  # noqa: E402
from chlu.data.industrial.skab import SKAB  # noqa: E402
from chlu.data.industrial.smd_tsb import SMDTSB  # noqa: E402
from chlu.data.industrial.tep_rieth import (  # noqa: E402
    CHANNELS as TEP_CHANNELS,
    TEST_ONSET_SAMPLE,
    _record_from_run,
)
from chlu.eval.splits import assert_no_unit_leakage  # noqa: E402


# ---------------------------------------------------------------------------
# SKAB
# ---------------------------------------------------------------------------


def _write_skab_csv(path, n, labelled):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.normal(size=(n, len(SKAB_CHANNELS))), columns=list(SKAB_CHANNELS)
    )
    df.insert(0, "datetime", pd.date_range("2020-01-01", periods=n, freq="s"))
    if labelled:
        anomaly = np.zeros(n)
        anomaly[n // 2 :] = 1.0
        df["anomaly"] = anomaly
        df["changepoint"] = 0.0
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep=";", index=False)


@pytest.fixture
def skab_root(tmp_path):
    _write_skab_csv(tmp_path / "anomaly-free" / "anomaly-free.csv", 120, False)
    _write_skab_csv(tmp_path / "valve1" / "0.csv", 80, True)
    _write_skab_csv(tmp_path / "valve2" / "0.csv", 60, True)
    return tmp_path


def test_skab_loader(skab_root):
    ds = SKAB(root=skab_root)
    assert ds.unit_ids() == ("anomaly-free/anomaly-free", "valve1/0", "valve2/0")
    assert ds.train_ids() == ("anomaly-free/anomaly-free",)
    assert ds.test_ids() == ("valve1/0", "valve2/0")
    assert_no_unit_leakage(ds.train_ids(), ds.test_ids())

    train = ds.load_unit("anomaly-free/anomaly-free")
    assert train.data.shape == (120, 8)
    assert train.point_labels is None
    assert train.data.dtype == np.float32

    test = ds.load_unit("valve1/0")
    assert test.point_labels.sum() == 40
    assert test.condition == "valve1"
    assert test.episode_label == 1
    assert "changepoint" in test.meta


def test_skab_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="GPL"):
        SKAB(root=tmp_path / "nowhere")


# ---------------------------------------------------------------------------
# SMD via TSB-AD curation
# ---------------------------------------------------------------------------


@pytest.fixture
def smd_root(tmp_path):
    rng = np.random.default_rng(1)
    n, d, tr = 200, 5, 120
    df = pd.DataFrame(rng.normal(size=(n, d)), columns=[f"f{i}" for i in range(d)])
    label = np.zeros(n, int)
    label[150:160] = 1
    df["Label"] = label
    dest = tmp_path / "TSB-AD-M"
    dest.mkdir()
    df.to_csv(dest / f"060_SMD_id_4_Facility_tr_{tr}_1st_150.csv", index=False)
    return tmp_path


def test_smd_tsb_loader(smd_root):
    ds = SMDTSB(root=smd_root)
    assert ds.protocol == "per_unit_prefix"
    assert ds.unit_ids() == ("SMD_id_04",)
    assert ds.test_ids() == ds.unit_ids()
    rec = ds.load_unit("SMD_id_04")
    assert rec.data.shape == (200, 5)
    assert rec.meta["train_len"] == 120
    assert rec.point_labels.sum() == 10


# ---------------------------------------------------------------------------
# TEP-Rieth (record construction from an in-memory frame — no pyreadr needed)
# ---------------------------------------------------------------------------


def _tep_frame(fault, run, n_samples):
    rng = np.random.default_rng(2)
    df = pd.DataFrame(rng.normal(size=(n_samples, 52)), columns=list(TEP_CHANNELS))
    df.insert(0, "faultNumber", fault)
    df.insert(1, "simulationRun", run)
    df.insert(2, "sample", np.arange(1, n_samples + 1))
    return df


def test_tep_record_labels_faulty_testing_run():
    rec = _record_from_run(
        "faulty_testing:fault04:run001",
        _tep_frame(4, 1, 960),
        "faulty_testing",
        4,
        1,
    )
    assert rec.data.shape == (960, 52)
    # onset AFTER 8h: samples 1..160 normal, 161..960 anomalous
    assert rec.point_labels[:TEST_ONSET_SAMPLE].sum() == 0
    assert rec.point_labels.sum() == 960 - TEST_ONSET_SAMPLE
    assert rec.fault_class == "fault04"
    assert rec.episode_label == 1


def test_tep_record_fault_free_is_all_normal():
    rec = _record_from_run(
        "fault_free_testing:fault00:run001",
        _tep_frame(0, 1, 960),
        "fault_free_testing",
        0,
        1,
    )
    assert rec.point_labels.sum() == 0
    assert rec.episode_label == 0


# ---------------------------------------------------------------------------
# voraus-AD (synthetic parquet — needs pyarrow)
# ---------------------------------------------------------------------------


def test_voraus_ad_loader(tmp_path):
    pytest.importorskip("pyarrow", reason="voraus loader needs pyarrow")
    from chlu.data.industrial.voraus_ad import TRAIN_SETTING_PRE_A, VorausAD

    rows = []
    for sid, (setting, anomaly, category) in enumerate(
        [
            (TRAIN_SETTING_PRE_A, False, 12),
            (TRAIN_SETTING_PRE_A, False, 12),
            (3, True, 4),
        ]
    ):
        for t in range(50):
            rows.append(
                {
                    "sample": sid,
                    "anomaly": anomaly,
                    "category": category,
                    "setting": setting,
                    "time": float(t),
                    "action": 1,
                    "active": 1,
                    "motor_current_1": float(np.sin(0.1 * t) + sid),
                    "joint_velocity_1": float(np.cos(0.1 * t)),
                }
            )
    root = tmp_path / "voraus_ad"
    root.mkdir()
    pd.DataFrame(rows).to_parquet(root / "voraus-ad-dataset-100hz.parquet")

    ds = VorausAD(root=root)
    assert ds.label_kind == "episode"
    assert ds.train_ids() == ("0", "1")
    assert ds.test_ids() == ("2",)
    rec = ds.load_unit("2")
    assert rec.episode_label == 1
    assert rec.fault_class == "COLLISION_CARTON"
    assert rec.data.shape == (50, 2)  # meta columns excluded
    assert set(rec.channels) == {"motor_current_1", "joint_velocity_1"}
    normal = ds.load_unit("0")
    assert normal.episode_label == 0
    assert normal.fault_class == "NORMAL_OPERATION"


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def test_registry_contents_and_unknown_key():
    assert set(DATASET_REGISTRY) == {
        "skab",
        "voraus_ad",
        "tep_rieth",
        "smd_tsb",
        "mimii",
    }
    with pytest.raises(KeyError, match="unknown dataset"):
        get_dataset("does_not_exist")
