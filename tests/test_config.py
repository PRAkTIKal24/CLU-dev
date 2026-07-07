"""Config integrity tests.

Regression context: the wave-2 union merge dropped the @dataclass decorator
from ExperimentV1GateConfig (list-typed defaults then evaluated to bare
``dataclasses.Field`` objects at runtime and save/load_config crashed on the
group). These tests pin every sub-config as a real dataclass and exercise the
YAML round trip.
"""

import dataclasses

from chlu.config import CHLUConfig, get_default_config, load_config, save_config


def test_all_subconfigs_are_dataclasses():
    cfg = get_default_config()
    assert dataclasses.is_dataclass(CHLUConfig)
    for f in dataclasses.fields(CHLUConfig):
        sub = getattr(cfg, f.name)
        assert dataclasses.is_dataclass(sub), (
            f"CHLUConfig.{f.name} is not a dataclass instance "
            "(missing @dataclass decorator?)"
        )


def test_list_defaults_materialize_as_lists():
    cfg = get_default_config()
    # These were dataclasses.Field objects while the decorator was missing.
    assert isinstance(cfg.experiment_v1_gate.zeta_grid, list)
    assert isinstance(cfg.experiment_v1_gate.difficulty_levels, list)
    assert len(cfg.experiment_v1_gate.zeta_grid) > 0
    # Mutable defaults must not be shared across instances.
    other = get_default_config()
    other.experiment_v1_gate.zeta_grid.append(99.0)
    assert 99.0 not in cfg.experiment_v1_gate.zeta_grid


def test_yaml_round_trip(tmp_path):
    cfg = get_default_config()
    cfg.experiment_v1_gate.hopfield_beta = 12.5
    cfg.experiment_v1_gate.difficulty_levels = [[32, 4]]
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded.experiment_v1_gate.hopfield_beta == 12.5
    assert loaded.experiment_v1_gate.difficulty_levels == [[32, 4]]
    assert loaded.experiment_v1_gate.zeta_grid == cfg.experiment_v1_gate.zeta_grid


def test_default_config_full_round_trip(tmp_path):
    """Every field group must survive save->load unchanged (the w2/w3 merge
    artifact killer: a group dropped from save_config/load_config, or one that
    lost its @dataclass decorator, breaks this)."""
    cfg = get_default_config()
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    loaded = load_config(path)
    assert dataclasses.asdict(loaded) == dataclasses.asdict(cfg)


def _mutate_group(group):
    """Flip every bool and bump every numeric field to a non-default value
    (bool is an int subclass, so it is checked first)."""
    for f in dataclasses.fields(group):
        v = getattr(group, f.name)
        if isinstance(v, bool):
            setattr(group, f.name, not v)
        elif isinstance(v, int):
            setattr(group, f.name, v + 7)
        elif isinstance(v, float):
            setattr(group, f.name, v * 2.0 + 1.0)
    return group


def test_every_group_round_trips_mutated(tmp_path):
    """Mutate a numeric/bool field in EVERY group, round-trip, assert full
    equality. A group missing from load_config's reconstruction would silently
    revert its mutated values to defaults and fail this comparison."""
    cfg = get_default_config()
    for f in dataclasses.fields(cfg):
        _mutate_group(getattr(cfg, f.name))
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    loaded = load_config(path)
    assert dataclasses.asdict(loaded) == dataclasses.asdict(cfg)


def test_mass_lr_mult_default_and_round_trip(tmp_path):
    """training.mass_lr_mult (critique P5/G4) defaults to 1.0 (bit-compatible)
    and survives the YAML round trip at a non-default value."""
    cfg = get_default_config()
    assert cfg.training.mass_lr_mult == 1.0
    cfg.training.mass_lr_mult = 100.0
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded.training.mass_lr_mult == 100.0
