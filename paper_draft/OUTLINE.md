# Outline

## Title
- Emphasize the main finding: L2 clearly reduces overfitting on small tabular datasets, but does not yield a statistically significant aggregate accuracy gain.

## Abstract
- State the practical question for sub-1000-sample tabular classification.
- Describe the repeated-seed logistic-regression comparison on four datasets.
- Report the main quantitative outcomes: accuracy `0.8292 -> 0.8385` (`p=0.345`), train-validation gap `0.1297 -> 0.0257` (`p=0.011`).
- Close with the practical implication: L2 is a dependable shrinkage tool, not a universal accuracy booster.

## Introduction
- Hook: small tabular datasets make model selection noisy and overfitting easy.
- Gap: most weight-decay evidence comes from deep vision or synthetic settings.
- Approach: controlled repeated-seed comparison of near-unregularized vs validation-tuned L2 logistic regression.
- Preview: strong gap reduction, weak aggregate accuracy effect.
- Contributions:
  - quantify the L2 effect under fixed preprocessing and held-out testing;
  - separate predictive gains from overfitting control;
  - report repeated-seed paired statistics and confidence intervals;
  - provide reproducible figures and tables for four datasets.

## Related Work
- Classical weight decay and smaller-norm solutions.
- Optimizer semantics and the L2 vs weight-decay distinction.
- Cautionary generalization literature arguing explicit regularization is not a complete explanation.
- Small-data and modern dynamics views that motivate conditional expectations instead of blanket claims.

## Methodology
- Experimental question and paired comparison setup.
- Datasets, preprocessing, and `70/15/15` stratified splits over seeds `{7,21,42}`.
- Baselines: dummy, near-unregularized logistic regression, tuned L2 logistic regression.
- Metrics, hyperparameter search, and statistical tests.
- Reproducibility details from environment artifact.

## Results
- Aggregate table: accuracy, weighted F1, train-validation gap, confidence intervals, and p-values.
- Dataset table: dummy, baseline, and L2 means plus median best `C`.
- Figure references to regularization summary and gap boxplot.
- Sensitivity/ablation framing for the `C` sweep and coefficient shrinkage.

## Discussion
- Interpret why L2 helps gap reduction more consistently than test accuracy.
- Explain dataset-specific wins (`breast_cancer`, `wine`) and mixed cases (`sonar`, `glass`).
- State limitations: small `n`, approximate unregularized baseline, split protocol, normality issue.
- Broader implication: regularization control is more stable than accuracy uplift.

## Conclusion
- Summarize the empirical contribution and practical takeaway.
- Suggest extensions: MLPs, nested CV, more datasets, calibration/error analysis.
