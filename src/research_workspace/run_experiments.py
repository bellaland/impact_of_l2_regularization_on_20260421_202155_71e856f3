"""Run a controlled L2-regularization study on small tabular datasets."""

from __future__ import annotations

import json
import math
import platform
import random
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import sklearn
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore", message=r".*'penalty' was deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=r".*Setting penalty=None will ignore the C and l1_ratio parameters.*", category=UserWarning)


ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = ROOT / "datasets"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
LOGS_DIR = ROOT / "logs"

SEEDS = [7, 21, 42]
C_GRID = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10, 30, 100, 300, 1000])
PRIMARY_DATASETS = {
    "sklearn_wine": DATASETS_DIR / "sklearn_wine" / "sklearn_wine.csv",
    "sklearn_breast_cancer": DATASETS_DIR / "sklearn_breast_cancer" / "sklearn_breast_cancer.csv",
    "uci_glass_identification": DATASETS_DIR / "uci_glass_identification" / "uci_glass_identification.csv",
    "uci_connectionist_bench_sonar": DATASETS_DIR / "uci_connectionist_bench_sonar" / "uci_connectionist_bench_sonar.csv",
}
TARGET_COLUMNS = {
    "sklearn_wine": "target",
    "sklearn_breast_cancer": "target",
    "uci_glass_identification": "Type_of_glass",
    "uci_connectionist_bench_sonar": "class",
}


@dataclass
class SplitBundle:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_directories() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)


def save_environment_info() -> dict[str, Any]:
    info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "scipy_version": scipy.__version__,
        "gpu": "NO_GPU",
        "seeds": SEEDS,
        "c_grid": C_GRID.tolist(),
    }
    with (RESULTS_DIR / "environment.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return info


def load_dataset(name: str, path: Path) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(path)
    target_col = TARGET_COLUMNS[name]
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {path}")
    return df, target_col


def encode_target(y: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(y.astype(str))
    return encoded, encoder


def stratified_split(X: pd.DataFrame, y: np.ndarray, seed: int) -> SplitBundle:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=seed,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=seed,
        stratify=y_temp,
    )
    return SplitBundle(X_train, X_val, X_test, y_train, y_val, y_test)


def build_model(*, penalty: str | None, C: float | None, seed: int) -> Pipeline:
    numeric_features = slice(None)
    preprocessor = ColumnTransformer(
        transformers=[("scale", StandardScaler(), numeric_features)],
        remainder="drop",
    )
    if penalty is None:
        clf = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            C=1e12,
            max_iter=5000,
            random_state=seed,
        )
    else:
        clf = LogisticRegression(
            C=float(C),
            solver="lbfgs",
            max_iter=5000,
            random_state=seed,
        )
    return Pipeline([("preprocess", preprocessor), ("model", clf)])


def evaluate_classifier(model: Pipeline, X: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
    pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y, pred),
        "f1_weighted": f1_score(y, pred, average="weighted", zero_division=0),
    }


def coefficient_norm(model: Pipeline) -> float:
    coef = model.named_steps["model"].coef_
    return float(np.linalg.norm(coef))


def run_single_dataset(name: str, path: Path) -> dict[str, Any]:
    dataset_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    eda_rows: list[dict[str, Any]] = []
    df, target_col = load_dataset(name, path)
    y_encoded, label_encoder = encode_target(df[target_col])
    X = df.drop(columns=[target_col])

    eda_rows.append(
        {
            "dataset": name,
            "samples": int(df.shape[0]),
            "features": int(X.shape[1]),
            "target_column": target_col,
            "n_classes": int(len(label_encoder.classes_)),
            "duplicates": int(df.duplicated().sum()),
            "missing_values": int(df.isna().sum().sum()),
        }
    )

    for seed in SEEDS:
        set_seed(seed)
        split = stratified_split(X, y_encoded, seed)

        dummy = DummyClassifier(strategy="stratified", random_state=seed)
        dummy.fit(split.X_train, split.y_train)
        dummy_test_acc = accuracy_score(split.y_test, dummy.predict(split.X_test))

        baseline_model = build_model(penalty=None, C=None, seed=seed)
        baseline_model.fit(split.X_train, split.y_train)

        baseline_train = evaluate_classifier(baseline_model, split.X_train, split.y_train)
        baseline_val = evaluate_classifier(baseline_model, split.X_val, split.y_val)
        baseline_test = evaluate_classifier(baseline_model, split.X_test, split.y_test)

        best_record: dict[str, Any] | None = None
        for C in C_GRID:
            l2_model = build_model(penalty="l2", C=float(C), seed=seed)
            l2_model.fit(split.X_train, split.y_train)

            train_metrics = evaluate_classifier(l2_model, split.X_train, split.y_train)
            val_metrics = evaluate_classifier(l2_model, split.X_val, split.y_val)
            test_metrics = evaluate_classifier(l2_model, split.X_test, split.y_test)
            record = {
                "dataset": name,
                "seed": seed,
                "C": float(C),
                "alpha": float(1.0 / C),
                "train_accuracy": train_metrics["accuracy"],
                "val_accuracy": val_metrics["accuracy"],
                "test_accuracy": test_metrics["accuracy"],
                "train_f1_weighted": train_metrics["f1_weighted"],
                "val_f1_weighted": val_metrics["f1_weighted"],
                "test_f1_weighted": test_metrics["f1_weighted"],
                "train_val_gap": train_metrics["accuracy"] - val_metrics["accuracy"],
                "train_test_gap": train_metrics["accuracy"] - test_metrics["accuracy"],
                "coef_l2_norm": coefficient_norm(l2_model),
            }
            tuning_rows.append(record)
            if best_record is None:
                best_record = record
                continue
            if record["val_f1_weighted"] > best_record["val_f1_weighted"]:
                best_record = record
            elif math.isclose(record["val_f1_weighted"], best_record["val_f1_weighted"], rel_tol=0.0, abs_tol=1e-12):
                if record["val_accuracy"] > best_record["val_accuracy"]:
                    best_record = record

        assert best_record is not None
        dataset_rows.append(
            {
                "dataset": name,
                "seed": seed,
                "n_samples": int(df.shape[0]),
                "n_features": int(X.shape[1]),
                "n_classes": int(len(label_encoder.classes_)),
                "dummy_test_acc": dummy_test_acc,
                "baseline_train_acc": baseline_train["accuracy"],
                "baseline_val_acc": baseline_val["accuracy"],
                "baseline_test_acc": baseline_test["accuracy"],
                "baseline_train_f1": baseline_train["f1_weighted"],
                "baseline_val_f1": baseline_val["f1_weighted"],
                "baseline_test_f1": baseline_test["f1_weighted"],
                "baseline_train_val_gap": baseline_train["accuracy"] - baseline_val["accuracy"],
                "baseline_train_test_gap": baseline_train["accuracy"] - baseline_test["accuracy"],
                "baseline_coef_norm": coefficient_norm(baseline_model),
                "best_C": best_record["C"],
                "best_alpha": best_record["alpha"],
                "l2_train_acc": best_record["train_accuracy"],
                "l2_val_acc": best_record["val_accuracy"],
                "l2_test_acc": best_record["test_accuracy"],
                "l2_train_f1": best_record["train_f1_weighted"],
                "l2_val_f1": best_record["val_f1_weighted"],
                "l2_test_f1": best_record["test_f1_weighted"],
                "l2_train_val_gap": best_record["train_val_gap"],
                "l2_train_test_gap": best_record["train_test_gap"],
                "l2_coef_norm": best_record["coef_l2_norm"],
            }
        )

    return {
        "results": dataset_rows,
        "tuning": tuning_rows,
        "eda": eda_rows,
    }


def paired_cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    std = np.std(diff, ddof=1)
    if np.isclose(std, 0.0):
        return float("inf") if not np.isclose(np.mean(diff), 0.0) else 0.0
    return float(np.mean(diff) / std)


def ci_95(diff: np.ndarray) -> tuple[float, float]:
    n = len(diff)
    if n < 2:
        mean_diff = float(np.mean(diff))
        return mean_diff, mean_diff
    sem = stats.sem(diff)
    margin = stats.t.ppf(0.975, df=n - 1) * sem
    mean_diff = float(np.mean(diff))
    return mean_diff - float(margin), mean_diff + float(margin)


def paired_summary(df: pd.DataFrame, left: str, right: str) -> dict[str, Any]:
    left_values = df[left].to_numpy(dtype=float)
    right_values = df[right].to_numpy(dtype=float)
    diffs = left_values - right_values
    if len(diffs) >= 2:
        t_stat, p_value = stats.ttest_rel(left_values, right_values)
    else:
        t_stat, p_value = float("nan"), float("nan")
    if len(diffs) >= 3:
        shapiro_stat, shapiro_p = stats.shapiro(diffs)
    else:
        shapiro_stat, shapiro_p = float("nan"), float("nan")
    ci_low, ci_high = ci_95(diffs)
    return {
        "mean_left": float(np.mean(left_values)),
        "mean_right": float(np.mean(right_values)),
        "mean_diff": float(np.mean(diffs)),
        "std_diff": float(np.std(diffs, ddof=1)) if len(diffs) >= 2 else 0.0,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d_paired": paired_cohens_d(left_values, right_values),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "shapiro_statistic": float(shapiro_stat),
        "shapiro_p_value": float(shapiro_p),
        "n_pairs": int(len(diffs)),
    }


def aggregate_results(results_df: pd.DataFrame) -> dict[str, Any]:
    metrics = {
        "baseline_test_acc": float(results_df["baseline_test_acc"].mean()),
        "l2_test_acc": float(results_df["l2_test_acc"].mean()),
        "baseline_train_val_gap": float(results_df["baseline_train_val_gap"].mean()),
        "l2_train_val_gap": float(results_df["l2_train_val_gap"].mean()),
    }

    by_dataset = []
    for dataset_name, group in results_df.groupby("dataset", sort=True):
        by_dataset.append(
            {
                "dataset": dataset_name,
                "n_runs": int(len(group)),
                "dummy_test_acc_mean": float(group["dummy_test_acc"].mean()),
                "baseline_test_acc_mean": float(group["baseline_test_acc"].mean()),
                "l2_test_acc_mean": float(group["l2_test_acc"].mean()),
                "baseline_test_f1_mean": float(group["baseline_test_f1"].mean()),
                "l2_test_f1_mean": float(group["l2_test_f1"].mean()),
                "baseline_gap_mean": float(group["baseline_train_val_gap"].mean()),
                "l2_gap_mean": float(group["l2_train_val_gap"].mean()),
                "best_C_median": float(group["best_C"].median()),
            }
        )

    stats_payload = {
        "overall_accuracy": paired_summary(results_df, "l2_test_acc", "baseline_test_acc"),
        "overall_weighted_f1": paired_summary(results_df, "l2_test_f1", "baseline_test_f1"),
        "overall_train_val_gap": paired_summary(results_df, "l2_train_val_gap", "baseline_train_val_gap"),
        "per_dataset": {
            dataset_name: {
                "accuracy": paired_summary(group, "l2_test_acc", "baseline_test_acc"),
                "weighted_f1": paired_summary(group, "l2_test_f1", "baseline_test_f1"),
                "train_val_gap": paired_summary(group, "l2_train_val_gap", "baseline_train_val_gap"),
            }
            for dataset_name, group in results_df.groupby("dataset", sort=True)
        },
    }

    return {
        "metrics": metrics,
        "dataset_summary": by_dataset,
        "statistical_tests": stats_payload,
    }


def generate_plots(results_df: pd.DataFrame, tuning_df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")

    wine_tuning = tuning_df[tuning_df["dataset"] == "sklearn_wine"].copy()
    if not wine_tuning.empty:
        mean_curve = (
            wine_tuning.groupby("C", as_index=False)[["train_accuracy", "val_accuracy", "test_accuracy", "coef_l2_norm"]]
            .mean()
            .sort_values("C")
        )
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))

        axes[0].plot(mean_curve["C"], mean_curve["train_accuracy"], marker="o", label="Train Accuracy")
        axes[0].plot(mean_curve["C"], mean_curve["val_accuracy"], marker="o", label="Validation Accuracy")
        axes[0].set_xscale("log")
        axes[0].set_xlabel("C (inverse regularization strength)")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Wine Train/Validation Curves")
        axes[0].legend()

        axes[1].plot(1.0 / mean_curve["C"], mean_curve["coef_l2_norm"], marker="o", color="#C44E52")
        axes[1].set_xscale("log")
        axes[1].set_xlabel("Alpha = 1 / C")
        axes[1].set_ylabel("Coefficient L2 Norm")
        axes[1].set_title("Wine Regularization Path")

        for dataset_name, group in tuning_df.groupby("dataset", sort=True):
            performance_curve = group.groupby("C", as_index=False)["val_f1_weighted"].mean().sort_values("C")
            axes[2].plot(performance_curve["C"], performance_curve["val_f1_weighted"], marker="o", label=dataset_name)
        axes[2].set_xscale("log")
        axes[2].set_xlabel("C (inverse regularization strength)")
        axes[2].set_ylabel("Mean Validation Weighted F1")
        axes[2].set_title("Performance vs Regularization Strength")
        axes[2].legend(fontsize=10)

        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "regularization_summary.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    gap_df = results_df.melt(
        id_vars=["dataset", "seed"],
        value_vars=["baseline_train_val_gap", "l2_train_val_gap"],
        var_name="model",
        value_name="train_val_gap",
    )
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(data=gap_df, x="dataset", y="train_val_gap", hue="model", ax=ax)
    ax.set_title("Train-Validation Gap by Dataset")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Accuracy Gap")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "train_val_gap_boxplot.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    environment: dict[str, Any],
    eda_df: pd.DataFrame,
    results_df: pd.DataFrame,
    tuning_df: pd.DataFrame,
    aggregate: dict[str, Any],
    elapsed_seconds: float,
) -> None:
    eda_df.to_csv(RESULTS_DIR / "eda_summary.csv", index=False)
    results_df.to_csv(RESULTS_DIR / "per_run_results.csv", index=False)
    tuning_df.to_csv(RESULTS_DIR / "regularization_grid_results.csv", index=False)
    pd.DataFrame(aggregate["dataset_summary"]).to_csv(RESULTS_DIR / "dataset_summary.csv", index=False)

    with (RESULTS_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate["metrics"], f, indent=2)
    with (RESULTS_DIR / "statistical_tests.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate["statistical_tests"], f, indent=2)

    run_manifest = {
        "datasets": list(PRIMARY_DATASETS.keys()),
        "execution_time_seconds": elapsed_seconds,
        "environment": environment,
        "output_files": {
            "metrics_json": str(RESULTS_DIR / "metrics.json"),
            "per_run_results_csv": str(RESULTS_DIR / "per_run_results.csv"),
            "grid_results_csv": str(RESULTS_DIR / "regularization_grid_results.csv"),
            "statistical_tests_json": str(RESULTS_DIR / "statistical_tests.json"),
            "figure_regularization_summary": str(FIGURES_DIR / "regularization_summary.png"),
            "figure_gap_boxplot": str(FIGURES_DIR / "train_val_gap_boxplot.png"),
        },
    }
    with (RESULTS_DIR / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)


def main() -> None:
    start = time.time()
    ensure_directories()
    environment = save_environment_info()

    all_results: list[dict[str, Any]] = []
    all_tuning: list[dict[str, Any]] = []
    all_eda: list[dict[str, Any]] = []

    for dataset_name, path in PRIMARY_DATASETS.items():
        outcome = run_single_dataset(dataset_name, path)
        all_results.extend(outcome["results"])
        all_tuning.extend(outcome["tuning"])
        all_eda.extend(outcome["eda"])

    results_df = pd.DataFrame(all_results)
    tuning_df = pd.DataFrame(all_tuning)
    eda_df = pd.DataFrame(all_eda)

    aggregate = aggregate_results(results_df)
    generate_plots(results_df, tuning_df)
    elapsed_seconds = time.time() - start
    write_outputs(environment, eda_df, results_df, tuning_df, aggregate, elapsed_seconds)

    summary = {
        "metrics": aggregate["metrics"],
        "overall_accuracy_test": aggregate["statistical_tests"]["overall_accuracy"],
        "overall_train_val_gap_test": aggregate["statistical_tests"]["overall_train_val_gap"],
        "elapsed_seconds": elapsed_seconds,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
