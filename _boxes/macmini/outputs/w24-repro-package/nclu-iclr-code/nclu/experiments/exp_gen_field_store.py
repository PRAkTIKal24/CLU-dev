"""exp gen field store."""

from __future__ import annotations
from nclu.experiments.factor_wells import PINS as W3PINS

PINS = dict(W3PINS)
PINS.update(
    {
        "condition": "val",
        "bridge_seed": 0,
        "seeds": (0, 1, 2),
        "seed_meaning": "store-fit seed at bridge seed 0",
        "pool_seed_stride": 7919,
        "atom_seed_stride": 1000003,
        "val_pairs": ((3, 1), (4, 2), (6, 3)),
        "n_row1_requests": 512,
        "row1_rng_seed": 20260919,
        "field_n_atoms": 64,
        "field_eta": 1.0,
        "field_ridge_frac": 0.001,
        "field_sigma_mult": 1.0,
        "field_atom_select": "span",
        "field_term_map": "marginal",
        "field_coupling_prior_strength": 32.0,
        "field_coupling_rank": 8,
        "field_decontaminate": 0.5,
        "field_coupling_shrink": 2.0,
        "field_coupling_eta": 0.0,
        "field_gap_tau": 4.0,
        "field_basin_live_tau": 2.0,
        "field_atom_seed": 20260919,
        "delete_factor": 2,
        "delete_value": 0,
        "hole_pair": ((2, 0), (4, 1)),
    }
)


def _host_factors(part_info: dict, K: int) -> list:
    prov = part_info["factor_aligned"]["provenance"]
    return [list(map(int, prov[str(k)]["source_factor_indices"])) for k in range(K)]


def field_spec(store_seed: int = 0) -> "object":
    from nclu.core.gen_field import FieldSpec

    return FieldSpec(
        n_atoms=int(PINS["field_n_atoms"]),
        eta=float(PINS["field_eta"]),
        ridge_frac=float(PINS["field_ridge_frac"]),
        sigma_mult=float(PINS["field_sigma_mult"]),
        coupling_prior_strength=float(PINS["field_coupling_prior_strength"]),
        coupling_rank=int(PINS["field_coupling_rank"]),
        coupling_shrink=float(PINS["field_coupling_shrink"]),
        coupling_eta=float(PINS["field_coupling_eta"]),
        atom_select=str(PINS["field_atom_select"]),
        decontaminate=float(PINS["field_decontaminate"]),
        term_map=str(PINS["field_term_map"]),
        gap_tau=float(PINS["field_gap_tau"]),
        basin_live_tau=float(PINS["field_basin_live_tau"]),
        atom_seed=int(PINS["field_atom_seed"]) + int(PINS["atom_seed_stride"]) * int(store_seed),
    )
