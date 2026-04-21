# Research Plan: Impact of L2 Regularization on Small Dataset Generalization

## Motivation & Novelty Assessment

### Why This Research Matters
L2 regularization is one of the most common defenses against overfitting, but practitioners working with small tabular datasets rarely get a quantitative estimate of how much it changes the train-validation gap or held-out accuracy in a controlled setting. A careful small-data benchmark is useful because model selection noise is high below 1000 samples, and seemingly standard choices can produce unstable conclusions.

### Gap in Existing Work
The literature review shows that most weight-decay evidence comes from vision models or synthetic tasks, while direct repeated-seed evidence on tiny real tabular datasets is relatively thin. Existing work also warns that regularization gains are conditional rather than universal, so a practical controlled study on small real datasets fills a concrete measurement gap.

### Our Novel Contribution
This project contributes a reproducible, repeated-seed comparison of unregularized versus L2-regularized logistic regression on four small tabular classification datasets, with fixed preprocessing, validation-based alpha selection, held-out test evaluation, and explicit reporting of overfitting gaps, coefficient norms, and statistical significance. The contribution is empirical and practical: quantify the size and consistency of the L2 effect in a regime where split variance is large.

### Experiment Justification
- Experiment 1: Run repeated stratified hold-out evaluation on `sklearn_wine` because it is the prompt-aligned primary dataset and a direct small multiclass benchmark.
- Experiment 2: Repeat the same protocol on `sklearn_breast_cancer`, `uci_glass_identification`, and `uci_connectionist_bench_sonar` to test whether any observed benefit generalizes beyond one dataset.
- Experiment 3: Sweep L2 strengths on the validation split and aggregate performance-versus-regularization curves to measure whether improvements come with lower train-validation gaps and smaller coefficient norms.
- Experiment 4: Perform paired statistical testing across random seeds to determine whether any apparent accuracy or F1 gains are larger than split noise.

## Research Question
Does L2 regularization improve held-out generalization on small classification datasets with fewer than 1000 samples, compared with an otherwise identical unregularized logistic regression baseline?

## Background and Motivation
Regularization is intended to reduce variance by discouraging large weights, but on small datasets it can also increase bias enough to erase any benefit. The gathered literature supports a qualified version of the hypothesis: L2 can help when sample size is small relative to feature count or class complexity, but the effect depends on dataset structure and on evaluation variance. This motivates a repeated-seed benchmark that reports both predictive performance and direct overfitting indicators.

## Hypothesis Decomposition
- Sub-hypothesis 1: L2-regularized logistic regression will have a smaller train-validation gap than the unregularized baseline on average.
- Sub-hypothesis 2: Validation-tuned L2 regularization will improve held-out test accuracy and weighted F1 on at least the primary dataset and, ideally, on average across datasets.
- Sub-hypothesis 3: L2 regularization will reduce model complexity as measured by the L2 norm of the fitted coefficients.
- Independent variables: dataset identity, random seed, and regularization strength `C` / equivalent alpha.
- Dependent variables: train accuracy, validation accuracy, test accuracy, weighted F1, train-validation gap, train-test gap, and coefficient norm.
- Alternative explanation: any observed gain may be caused mostly by split variance rather than by regularization; repeated-seed paired testing is required to separate these.

## Proposed Methodology

### Approach
Use a controlled comparison study with identical preprocessing and model families. For each dataset and seed, create a stratified 70/15/15 split, fit a preprocessing-plus-logistic-regression pipeline, treat a nearly unregularized model as the baseline, select the best L2 strength from a logarithmic grid using validation weighted F1, then compare the final baseline and tuned L2 models on the untouched test split.

### Experimental Steps
1. Verify the isolated environment, install required dependencies with `uv`, and record hardware and package versions.
2. Load the four primary small-data datasets from `datasets/`, identify the label column, and run data quality checks for missing values, duplicates, class counts, and feature summary statistics.
3. Save representative samples and EDA plots so dataset properties are documented and reproducible.
4. For each dataset and each seed in `{7, 21, 42}`, create a stratified 70/15/15 split and fit a `StandardScaler` on the training subset only to prevent leakage.
5. Train the baseline logistic regression with effectively no penalty using a very large `C`, plus a stratified random dummy baseline for sanity checking.
6. Train L2 logistic regression models across a logarithmic grid of `C` values, choose the best value using validation weighted F1, and refit the selected model on the training subset for evaluation.
7. Record train, validation, and test metrics, the train-validation gap, the train-test gap, selected `C`, and coefficient norms for both baseline and tuned L2 models.
8. Aggregate results across seeds and datasets, compute paired t-tests and effect sizes, then generate figures for performance-versus-regularization, coefficient paths, and train/validation comparisons.
9. Export the requested metrics JSON, full per-run CSV/JSON artifacts, and a final report with interpretation, limitations, and reproducibility instructions.

### Baselines
- `DummyClassifier(strategy="stratified")`: lower-bound sanity check.
- Logistic regression with effectively no regularization (`C=1e6`): primary baseline matched to the regularized model except for the penalty strength.

### Evaluation Metrics
- Test accuracy: intuitive primary predictive metric.
- Weighted F1: robust to moderate class imbalance and multiclass settings.
- Train-validation gap: direct overfitting indicator required by the task.
- Train-test gap: additional overfitting indicator for final evaluation.
- Coefficient L2 norm: proxy for model complexity and expected shrinkage effect.

### Statistical Analysis Plan
- Main null hypothesis: mean paired difference between tuned L2 and baseline test accuracy is zero.
- Secondary null hypothesis: mean paired difference in train-validation gap is zero.
- Use paired `ttest_rel` across the three required random seeds for each dataset and for the macro aggregate across dataset-seed pairs.
- Check approximate normality of paired differences with Shapiro-Wilk where sample size permits, but treat low power as a limitation.
- Report p-values, 95% confidence intervals for paired mean differences, and Cohen's d for paired samples.
- Significance threshold: `alpha = 0.05`.

## Expected Outcomes
Results support the hypothesis if the tuned L2 model reduces the train-validation gap and shows equal or better mean test performance than the unregularized baseline, with statistically significant paired improvement in at least one core outcome. Results weaken the hypothesis if L2 mostly shrinks coefficients without improving held-out performance or if gains are inconsistent relative to seed variance.

## Timeline and Milestones
1. Planning and preregistration in `planning.md`: 10 minutes.
2. Environment verification and dependency installation: 5 minutes.
3. Data validation and EDA: 10 minutes.
4. Pipeline implementation and local checks: 20 minutes.
5. Full experiment runs and artifact generation: 15 minutes.
6. Statistical analysis, reporting, and validation rerun: 15 minutes.

## Potential Challenges
- Tiny validation splits can make hyperparameter selection noisy; mitigate with a modest log-grid and repeated seeds.
- Some datasets may be too easy or too unstable, which can hide the effect size; interpret per-dataset results rather than only an overall average.
- Multiclass and binary datasets differ in difficulty; use weighted F1 and dataset-specific reporting to avoid misleading aggregation.
- Logistic regression in scikit-learn always includes a penalty parameterization, so the “unregularized” baseline will be approximated with a very large `C`; document this explicitly.

## Success Criteria
- The code runs from the project-local environment without errors.
- `results/metrics.json` contains `baseline_test_acc`, `l2_test_acc`, `baseline_train_val_gap`, and `l2_train_val_gap`.
- Required figures are produced as PNG files.
- The report includes paired significance testing, effect sizes, limitations, and reproducibility details.
- Results are reproducible across the three specified random seeds.
