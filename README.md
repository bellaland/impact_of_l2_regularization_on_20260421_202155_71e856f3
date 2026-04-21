# Impact of L2 Regularization on Small Dataset Generalization

This workspace runs a controlled study of weakly regularized versus validation-tuned L2 logistic regression on four small tabular classification datasets: `sklearn_wine`, `sklearn_breast_cancer`, `uci_glass_identification`, and `uci_connectionist_bench_sonar`. The goal is to test whether L2 regularization reduces overfitting and improves held-out performance under `<1000` samples.

Key findings:
- Mean test accuracy improved from `0.8292` to `0.8385`, but the paired improvement was not statistically significant across the 12 dataset-seed runs (`p = 0.345`).
- Mean train-validation gap dropped from `0.1297` to `0.0257`, and that reduction was statistically significant (`p = 0.011`, paired Cohen's `d = -0.878`).
- L2 reduced the overfitting gap in `11/12` runs, but improved test accuracy in only `5/12` runs.
- The clearest accuracy gain appeared on `sklearn_breast_cancer`; `sonar` and `glass` showed weaker or negative accuracy effects despite better shrinkage.

Reproduce:
```bash
source .venv/bin/activate
python -m research_workspace.run_experiments
```

Outputs:
- Full report: `REPORT.md`
- Plan / preregistration: `planning.md`
- Metrics JSON: `results/metrics.json`
- Per-run results: `results/per_run_results.csv`
- Statistical tests: `results/statistical_tests.json`
- Figures: `figures/regularization_summary.png`, `figures/train_val_gap_boxplot.png`

File structure:
- `src/research_workspace/run_experiments.py`: end-to-end experiment runner
- `datasets/`: local benchmark CSVs and sample rows
- `results/`: metrics, statistical tests, environment, and run manifest
- `figures/`: generated PNG visualizations

The full interpretation, limitations, and links to the supporting literature are in `REPORT.md`.
