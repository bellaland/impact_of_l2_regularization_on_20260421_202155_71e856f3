# Resources Catalog

## Summary

This document catalogs the papers, datasets, and code repositories gathered for the project on L2 regularization and small-dataset generalization.

## Papers

Total papers downloaded: 6

| Title | Authors | Year | File | Key Info |
|------|------|------|------|------|
| A Simple Weight Decay Can Improve Generalization | Krogh, Hertz | 1991 | `papers/krogh_hertz_1991_simple_weight_decay.pdf` | Classical theory and empirical support for weight decay |
| Decoupled Weight Decay Regularization | Loshchilov, Hutter | 2019 | `papers/loshchilov_hutter_2019_decoupled_weight_decay.pdf` | AdamW distinction matters for optimizer comparisons |
| Understanding Deep Learning Requires Rethinking Generalization | Zhang et al. | 2017 | `papers/zhang_et_al_2017_rethinking_generalization.pdf` | Counterpoint against simplistic regularization narratives |
| Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets | Power et al. | 2022 | `papers/power_et_al_2022_grokking.pdf` | Small-data relevance; weight decay improves data efficiency |
| Why Do We Need Weight Decay in Modern Deep Learning? | D'Angelo et al. | 2024 | `papers/xie_et_al_2024_why_weight_decay.pdf` | Modern interpretation via optimization dynamics |
| Time Matters in Regularizing Deep Networks | Golatkar, Achille, Soatto | 2019 | `papers/golatkar_achille_soatto_2019_time_matters_regularizing.pdf` | Early regularization is more important than late regularization |

See `papers/README.md` for short descriptions.

## Datasets

Total datasets downloaded: 6

| Name | Source | Size | Task | Location | Notes |
|------|------|------|------|------|------|
| sklearn Wine | scikit-learn | 178 rows | classification | `datasets/sklearn_wine/` | Best direct match to prompt background |
| sklearn Breast Cancer | scikit-learn | 569 rows | classification | `datasets/sklearn_breast_cancer/` | Strong small-data binary benchmark |
| sklearn Iris | scikit-learn | 150 rows | classification | `datasets/sklearn_iris/` | Good sanity check, likely too easy |
| UCI Glass Identification | UCI / ucimlrepo | 214 rows | classification | `datasets/uci_glass_identification/` | Small multiclass benchmark |
| UCI Sonar | UCI / ucimlrepo | 208 rows | classification | `datasets/uci_connectionist_bench_sonar/` | Small high-dimensional binary benchmark |
| UCI Wine Quality | UCI direct CSV | 6,497 rows | regression or classification | `datasets/uci_wine_quality/` | Optional larger reference; not primary for `<1000` sample claim |

See `datasets/README.md` for loading and download instructions.

## Code Repositories

Total repositories cloned: 2

| Name | URL | Purpose | Location | Notes |
|------|------|------|------|------|
| AdamW-and-SGDW | https://github.com/loshchil/AdamW-and-SGDW | Reference implementation for decoupled weight decay | `code/adamw-and-sgdw/` | Torch7/Lua, historical reference |
| why-weight-decay | https://github.com/tml-epfl/why-weight-decay | Official code for NeurIPS 2024 paper | `code/why-weight-decay/` | Heavy GPU-oriented repo; useful as reference |

See `code/README.md` for more detail.

## Resource Gathering Notes

### Search Strategy

- Started with the local paper-finder helper script.
- The helper reported that the local service at `localhost:8000` was unavailable, so manual search was used.
- Manual search focused on accessible primary paper sources: arXiv, NeurIPS proceedings, and OpenReview-linked materials.
- Priority was given to papers that were either foundational, directly relevant to small-data generalization, or important for correct optimizer implementation.
- Revalidated on 2026-05-15: the local paper-finder helper still fell back because the service was unavailable, and web search against primary sources did not identify a more directly suitable small-tabular L2 paper than the included set.

### Selection Criteria

- Direct relevance to L2 regularization or weight decay.
- Usefulness for designing reproducible experiments on small datasets.
- Availability of full PDFs and, when possible, official code.
- Balance between foundational and modern papers.

### Challenges Encountered

- The local paper-finder service was not running.
- One initial PDF download pointed to the wrong NeurIPS paper and was removed during validation.
- The Wine Quality artifact available from the direct CSV pull is larger than the target small-data regime, so it is kept only as an optional reference dataset.
- Fresh environment setup was rerun on 2026-05-15 with `uv venv --clear` and `uv sync`; the workspace `.venv` contains the dependencies needed by the experiment runner.

### Gaps and Workarounds

- Most weight-decay literature uses vision or synthetic tasks rather than tiny real tabular datasets.
- To address that mismatch, the workspace includes several small tabular benchmarks that are more appropriate for the hypothesis test.
- No lightweight code repo was found that directly matches the planned small-tabular experiment design, so the cloned repos are methodological references rather than runnable baselines.

## Recommendations for Experiment Design

1. **Primary dataset(s)**: `sklearn_wine`, `sklearn_breast_cancer`, `uci_glass_identification`, and `uci_connectionist_bench_sonar`.
2. **Baseline methods**: unregularized logistic regression, ridge/logistic baseline with tuned L2, unregularized MLP, MLP with tuned L2, and optionally Adam versus AdamW variants.
3. **Evaluation metrics**: accuracy, macro-F1 when needed, and train-test gap over repeated stratified splits.
4. **Code to adapt/reuse**: borrow optimizer and ablation ideas from `code/why-weight-decay/`; use `code/adamw-and-sgdw/` only as a reference for correct decoupled weight decay semantics.
