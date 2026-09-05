#!/bin/zsh
# Retry-cell replication on FRESH seeds (PREREG §5: a CLU-gated win over the
# kNN-in-φ floor must be re-run before belief). Entry + retry only, no baselines.
cd /Users/user/Desktop/CHLU-cl-entry || exit 1
for s in 3 4 5; do
  PYTHONPATH=. /Users/user/Desktop/CHLU/.venv/bin/python -m chlu.experiments.exp_cl_entry \
      --seed $s --items entry,retry --baselines none > /tmp/cl_mnist_seed$s.log 2>&1
  mv results/exp_cl_entry_mnist_metrics.json /tmp/mnist_replic_seed$s.json
done
echo REPLICATION_DONE
