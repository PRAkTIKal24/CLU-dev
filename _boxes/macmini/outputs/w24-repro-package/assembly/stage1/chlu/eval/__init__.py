"""Dataset-agnostic industrial anomaly-detection evaluation harness (F2).

Binding evaluation rules (scout-industrial-datasets, 2026-07-05):
- VUS-PR is the primary metric; range-aware/event metrics and AUROC/AUPR are
  secondary. Metrics are computed by TSB-AD's reference implementation
  (vendored, pinned — see ``chlu/eval/_tsb_vendor/README.md``).
- Point-adjust F1 is FORBIDDEN and deliberately absent from this package.
- The four statistical baselines (PCA-recon, IsolationForest, LOF, KNN) are
  wired into every evaluation run by default.
- Splits are leakage-safe: by physical unit / run / simulation seed, never by
  window.
- Results are reported per dataset, never as a single grand mean.

No CLU modeling lives here — this is the measurement backbone.
"""
