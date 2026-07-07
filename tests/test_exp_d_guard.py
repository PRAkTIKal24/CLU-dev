"""exp-d sleep-erosion guard (fix-pack-4 item 4).

A designed SO(2) vacuum trained with an active sleep phase for >300 epochs and
NO V(data) anchor is the exact erosion regime (handover §7.14 /
anchor-robustness P11). run_experiment_d must warn loudly (UX only — no
behavior change) in that regime and stay silent otherwise.

Implementation note: the guard fires BEFORE train_chlu is called, so these
tests stub train_chlu to abort immediately — they exercise the guard predicate
without paying for training or the measurement harness.
"""

import warnings

import pytest

import chlu.experiments.exp_d_goldstone as ed
from chlu.config import get_default_config


class _Abort(RuntimeError):
    pass


def _stub_train(*args, **kwargs):
    raise _Abort("stop-after-guard")


def _run(config, tmp_path, **kw):
    return ed.run_experiment_d(config=config, save_dir=str(tmp_path), **kw)


def test_guard_warns_in_erosion_regime(monkeypatch, tmp_path):
    monkeypatch.setattr(ed, "train_chlu", _stub_train)
    config = get_default_config()
    config.experiment_d.train_epochs = 400  # > 300
    # sleep_mode "on", potential so2_invariant (designed), anchor 0.0 = defaults
    with pytest.warns(RuntimeWarning, match="sleep-erosion"):
        with pytest.raises(_Abort):
            _run(config, tmp_path, sleep_mode="on")


def test_guard_silent_with_anchor(monkeypatch, tmp_path):
    monkeypatch.setattr(ed, "train_chlu", _stub_train)
    config = get_default_config()
    config.experiment_d.train_epochs = 400
    config.training.anchor_data_energy_lambda = 10.0  # the cure => no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(_Abort):
            _run(config, tmp_path, sleep_mode="on")


def test_guard_silent_when_sleep_off(monkeypatch, tmp_path):
    monkeypatch.setattr(ed, "train_chlu", _stub_train)
    config = get_default_config()
    config.experiment_d.train_epochs = 400
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(_Abort):
            _run(config, tmp_path, sleep_mode="off")


def test_guard_silent_below_epoch_threshold(monkeypatch, tmp_path):
    monkeypatch.setattr(ed, "train_chlu", _stub_train)
    config = get_default_config()
    config.experiment_d.train_epochs = 150  # default intact regime
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(_Abort):
            _run(config, tmp_path, sleep_mode="on")
