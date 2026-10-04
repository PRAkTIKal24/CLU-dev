"""Industrial datasets for the F2/F3 evaluation harness.

Datasets (see each module's docstring for license + download notes):

=============  ==========================================  ========  ==========
key            what                                        labels    role
=============  ==========================================  ========  ==========
voraus_ad      6-axis robot pick&place (CC BY-NC-SA 4.0)   episode   headline
skab           water-circulation rig (GPL-3.0 — no vendor) point     iteration
tep_rieth      Tennessee Eastman, Rieth ed. (CC0)          point     scale
smd_tsb        SMD via TSB-AD curation (Apache/MIT)        point     negative
                                                                     control
mimii          acoustic machines (CC BY-SA, ~100 GB)       episode   stretch
                                                                     skeleton
=============  ==========================================  ========  ==========

Loader modules import pandas/pyarrow/pyreadr lazily — install the eval extra
(``uv sync --extra eval``) before touching voraus_ad/skab/tep_rieth/smd_tsb.
"""

from importlib import import_module

from chlu.data.industrial.base import (
    IndustrialDataset,
    UnitRecord,
    default_data_root,
)

#: dataset key -> (module, class); imported lazily via :func:`get_dataset`.
DATASET_REGISTRY = {
    "skab": ("chlu.data.industrial.skab", "SKAB"),
    "voraus_ad": ("chlu.data.industrial.voraus_ad", "VorausAD"),
    "tep_rieth": ("chlu.data.industrial.tep_rieth", "TEPRieth"),
    "smd_tsb": ("chlu.data.industrial.smd_tsb", "SMDTSB"),
    "mimii": ("chlu.data.industrial.mimii", "MIMII"),
}


def get_dataset(name: str, **kwargs) -> IndustrialDataset:
    """Instantiate a registered dataset by key (lazy import)."""
    try:
        module_name, cls_name = DATASET_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown dataset {name!r}; known: {sorted(DATASET_REGISTRY)}"
        ) from exc
    return getattr(import_module(module_name), cls_name)(**kwargs)


__all__ = [
    "DATASET_REGISTRY",
    "IndustrialDataset",
    "UnitRecord",
    "default_data_root",
    "get_dataset",
]
