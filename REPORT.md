# Report: Impact of L2 Regularization on Small Dataset Generalization

## 1. Executive Summary
This study tested whether L2 regularization improves generalization for small tabular classification datasets with fewer than 1000 samples. Across four datasets and three random seeds per dataset, validation-tuned L2 logistic regression increased mean test accuracy from `0.8267` to `0.8385`, but that accuracy gain was not statistically significant in the paired aggregate analysis (`p = 0.205`). In contrast, L2 dramatically reduced the mean train-validation gap from `0.1297` to `0.0257`, and that reduction was statistically significant (`p = 0.011`, paired Cohen's `d = -0.878`).

The practical implication is narrower than the initial hypothesis: on these small datasets, L2 was reliable as an anti-overfitting mechanism but not reliable as a universal accuracy booster. It reduced the generalization gap in `11/12` runs, while test accuracy improved in only `5/12` runs, tied in `5/12`, and worsened in `2/12`.

## 2. Research Question & Motivation
### Research question
Does L2 regularization improve held-out generalization on small classification datasets compared with an otherwise matched logistic-regression baseline?

### Motivation
The literature review in [literature_review.md](./literature_review.md) showed a gap between strong theoretical and large-scale evidence for weight decay and the practical small-data tabular setting. Most prior work focuses on vision or synthetic tasks, and relatively few reports quantify how much L2 changes both predictive performance and overfitting indicators under repeated random splits on tiny real datasets.

### Literature summary
- Krogh and Hertz (1991) provide the classical smaller-norm argument for weight decay.
- Loshchilov and Hutter (2019) show that optimizer semantics matter when comparing L2 and weight decay.
- Zhang et al. (2017) caution against claiming that explicit regularization alone explains generalization.
- Power et al. (2022) and D'Angelo et al. (2024) support the view that regularization affects optimization dynamics and data efficiency, especially in small-data or overtraining-prone regimes.

## 3. Data Construction
### Dataset description
The experiment used four local CSV datasets gathered in `datasets/`:

| Dataset | Samples | Features | Classes | Task |
|---|---:|---:|---:|---|
| `sklearn_wine` | 178 | 13 | 3 | multiclass classification |
| `sklearn_breast_cancer` | 569 | 30 | 2 | binary classification |
| `uci_glass_identification` | 214 | 9 | 6 | multiclass classification |
| `uci_connectionist_bench_sonar` | 208 | 60 | 2 | binary classification |

### Example samples
Representative raw examples are stored in each dataset directory as `samples.json`, for example:
- `datasets/sklearn_wine/samples.json`
- `datasets/sklearn_breast_cancer/samples.json`
- `datasets/uci_connectionist_bench_sonar/samples.json`

### Data quality
The EDA summary in `results/eda_summary.csv` found:
- No missing values in the four primary datasets.
- No duplicate rows in the four primary datasets.
- Moderate class imbalance in `breast_cancer`, `sonar`, and `glass`, so weighted F1 was reported alongside accuracy.

### Preprocessing
- Target labels were encoded with `LabelEncoder`.
- Features were standardized with `StandardScaler`.
- The scaler was fit on the training split only and applied to validation/test splits to avoid leakage.

### Train/Val/Test split
Each run used a stratified `70/15/15` split with seeds `{7, 21, 42}`. Hyperparameter selection used only the validation split; the test split was held out until final evaluation.

## 4. Methodology
### Approach
The study compared:
- A near-unregularized logistic regression baseline, implemented as a very weak L2 penalty (`C = 1e12`) because the current `scikit-learn 1.8.0` API deprecates the older explicit penalty controls.
- A validation-tuned L2 logistic regression model searched over a logarithmic grid of `C` values.
- A `DummyClassifier(strategy="stratified")` sanity baseline.

### Hyperparameters

| Parameter | Value | Selection Method |
|---|---|---|
| Seeds | `7, 21, 42` | fixed preregistration |
| Split ratio | `70/15/15` | fixed preregistration |
| Solver | `lbfgs` | constant across conditions |
| Max iterations | `5000` | fixed for convergence stability |
| Baseline regularization | `C=1e12` | implementation workaround for near-unregularized fit |
| L2 search grid | `1e-4` to `1e3` | fixed log grid |
| Selection metric | validation weighted F1 | fixed preregistration |

### Metrics
- Test accuracy
- Weighted F1
- Train-validation accuracy gap
- Train-test accuracy gap
- Coefficient L2 norm

### Statistical analysis
Paired `ttest_rel` compared L2 and baseline metrics across the 12 dataset-seed pairs. I also report paired Cohen's `d`, 95% confidence intervals for the mean paired difference, and Shapiro-Wilk tests on the paired differences as a rough assumption check. Because each per-dataset analysis has only `n=3`, those tests should be treated as descriptive rather than definitive.

### Environment and compute
- CPU only; GPU detection returned `NO_GPU`.
- Python `3.12.8`
- NumPy `2.4.4`
- pandas `3.0.2`
- scikit-learn `1.8.0`
- SciPy `1.17.1`
- Matplotlib `3.10.8`
- seaborn `0.13.2`
- Latest validation execution time: about `8s`

## 5. Results
### Requested metrics output
The required JSON file `results/metrics.json` contains:

```json
{
  "baseline_test_acc": 0.8266855815911048,
  "l2_test_acc": 0.8384884455015268,
  "baseline_train_val_gap": 0.12967718538627435,
  "l2_train_val_gap": 0.025704230868268108
}
```

### Aggregate comparison

| Metric | Baseline Mean | L2 Mean | Mean Difference (L2 - Baseline) | p-value |
|---|---:|---:|---:|---:|
| Test accuracy | 0.8267 | 0.8385 | +0.0118 | 0.205 |
| Weighted F1 | 0.8257 | 0.8357 | +0.0100 | 0.302 |
| Train-validation gap | 0.1297 | 0.0257 | -0.1040 | 0.011 |

Additional aggregate statistics from `results/statistical_tests.json`:
- Accuracy: 95% CI `[-0.0075, 0.0311]`, paired Cohen's `d = 0.389`
- Weighted F1: 95% CI `[-0.0103, 0.0303]`, paired Cohen's `d = 0.312`
- Train-validation gap: 95% CI `[-0.1792, -0.0287]`, paired Cohen's `d = -0.878`

### Per-dataset summary

| Dataset | Dummy Acc | Baseline Acc | L2 Acc | Baseline Gap | L2 Gap | Median Best C |
|---|---:|---:|---:|---:|---:|---:|
| `sklearn_breast_cancer` | 0.558 | 0.946 | 0.977 | 0.027 | -0.011 | 0.10 |
| `sklearn_wine` | 0.321 | 0.963 | 1.000 | 0.049 | 0.019 | 0.01 |
| `uci_connectionist_bench_sonar` | 0.448 | 0.802 | 0.781 | 0.301 | 0.049 | 0.10 |
| `uci_glass_identification` | 0.273 | 0.596 | 0.596 | 0.141 | 0.046 | 100.00 |

Observed run-level pattern:
- L2 improved test accuracy in `5/12` runs.
- L2 matched the baseline in `5/12` runs.
- L2 reduced the train-validation gap in `11/12` runs.

### Figures
- `figures/regularization_summary.png`: three-panel figure showing wine train/validation curves, wine coefficient shrinkage path, and validation weighted F1 versus `C` across datasets.
- `figures/train_val_gap_boxplot.png`: baseline vs L2 train-validation gap distribution by dataset.

## 6. Analysis & Discussion
### What the results show
The strongest supported claim is that L2 regularization reduces overfitting on small datasets in this setup. The mean train-validation gap shrank by about `0.104`, and the paired test was significant at the preregistered `alpha = 0.05`. This is consistent with the classical role of L2 as a shrinkage-based variance control.

The accuracy story is weaker. Mean test accuracy increased by only `0.0118`, with a 95% confidence interval crossing zero and a non-significant paired test. That means the experiment does not support the stronger claim that L2 reliably improves held-out predictive performance across small datasets in aggregate.

### Dataset-specific behavior
- `sklearn_breast_cancer` showed the clearest practical win for L2, with both accuracy and weighted F1 improving substantially. The paired p-values are below `0.05`, but this rests on only three seeds and should be interpreted cautiously.
- `sklearn_wine` also favored L2, reaching perfect mean test accuracy under the selected regularization settings.
- `sonar` and `glass` showed the opposite pattern: substantial shrinkage and smaller gaps, but no clear predictive gain. This suggests the penalty can reduce variance while adding enough bias to offset the benefit on harder or noisier small datasets.

### Error analysis and model complexity
The coefficient norms confirm the expected shrinkage effect, especially on the highest-variance datasets. `breast_cancer` and `wine` benefited from both shrinkage and accuracy gains, while `sonar` and `glass` mostly exhibited shrinkage without improved prediction. That is consistent with the literature's qualified view that regularization helps conditionally, not universally.

### Comparison to the literature
These findings align better with the cautious interpretation from the literature review than with a blanket pro-regularization claim. The results support the idea that L2 is reliable for controlling overfitting, but they do not show that this automatically translates into statistically significant accuracy gains across all small real tabular datasets.

## 7. Limitations
- Only logistic regression was studied; results may differ for MLPs or tree-based models.
- The aggregate analysis uses only 12 paired observations and per-dataset analyses use only 3 seeds.
- The near-unregularized baseline is approximated with `C=1e12` for compatibility with `scikit-learn 1.8.0`, not a mathematically exact zero-penalty solver.
- A single train/validation/test split per seed was used instead of nested cross-validation, so hyperparameter variance may still be understated.
- Shapiro-Wilk rejected normality for the aggregate train-gap differences, so the paired t-test on that metric should be read with some caution even though the effect size is large.

## 8. Conclusions & Next Steps
The hypothesis is only partially supported. L2 regularization clearly reduced overfitting in this small-data benchmark, but the aggregate improvement in held-out accuracy was small and not statistically significant. A more defensible conclusion is that L2 is a dependable shrinkage tool for small tabular datasets, while its predictive gains are dataset-dependent.

Recommended follow-up work:
- Extend the same protocol to small MLPs, where overfitting pressure is stronger.
- Replace single validation splits with repeated nested cross-validation.
- Add calibration metrics and confusion-matrix-based error analysis.
- Test whether the same pattern holds on more small tabular datasets or stratified subsamples of larger datasets.

## References
- Krogh, A., & Hertz, J. A. (1991). *A Simple Weight Decay Can Improve Generalization*.
- Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization*.
- Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2017). *Understanding Deep Learning Requires Rethinking Generalization*.
- Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*.
- D'Angelo, F., Andriushchenko, M., Varre, A., & Flammarion, N. (2024). *Why Do We Need Weight Decay in Modern Deep Learning?*
- Golatkar, A., Achille, A., & Soatto, S. (2019). *Time Matters in Regularizing Deep Networks*.
