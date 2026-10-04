# tables-r1 — the w10 seeds-and-tables artefact, REVISED by w12r spoke/artefact-mustfix

This directory is a **copy** of `/c/chlu/outputs/w10-seeds-and-tables/` with the referee's
artefact-level must-fixes applied to the **generator**, `build_tables.py`, and the outputs
regenerated from it. ⛔ The w10 directory itself and `/c/chlu/evidence/` were NOT written.

- `build_tables.py` — the generator. Edited. Deterministic: rebuilding reproduces `TABLES.md`
  byte-for-byte (except the `**Built**` timestamp line), `TABLES.json` field-for-field, and all
  five figure sha256 hashes.
- `TABLES.json`, `TABLES.md`, `figures/` — regenerated.
- `verify_tables.py` — UNCHANGED from w10. It does not import the generator; it re-reads every
  source from the path recorded in the manifest.
- `W10-ORIGINAL-LOG.md`, `W10-ORIGINAL-FINDINGS.md` — the w10 spoke's own records, carried
  verbatim and renamed so they are not mistaken for this pass's. w12r's records are
  `../LOG.md` and `../FINDINGS.md`.

## Acceptance

    cd tables-r1 && python verify_tables.py .

| | before (w10, pristine) | after (w12r) |
|---|---|---|
| verdict | **PASS** | **PASS** |
| sources registered / re-hashed | 27 / 27 | **32 / 32** |
| cells in TABLES.json | 240 | **286** |
| per-seed values re-resolved | 720 (3 reductions) | **854** (3 reductions) |
| figures (hashes re-verified) | 5 (5) | 5 (5) |
| declared NOT-RUN rows | 5 | 5 |
| corrections | 4 | **14** |

## Rebuild

    cd tables-r1 && python build_tables.py .
