"""Train and honestly evaluate the Stage 2 hazard surrogate.

**What this model is.**  The cellular automaton is expensive — a hazardous
128 px patch needs ~30,000 iterations to settle.  This model predicts what the
automaton *would* produce, from terrain statistics that cost one pass.  That is
a surrogate (an emulator of a simulation), not a landslide predictor validated
against observed events.  No observed lunar mass-wasting event appears anywhere
in this pipeline.

**Why the previous evaluation was meaningless.**  It reported macro F1 = 1.00 on
53 samples that were largely synthetic terrain generated from the label being
predicted, split randomly, with a target defined by a threshold on a feature the
model could see, and no baseline to beat.  Every one of those is addressed here:

* **Geographic split.**  Train on the nearside, test on the farside.  The farside
  is older and more heavily cratered, so this is a real distribution shift, not a
  random split wearing a geographic label.
* **Baselines.**  A mean predictor and a single-feature linear model.  A
  surrogate that cannot beat "look at the fraction of cells already above the
  angle of repose" has learned nothing worth reporting.
* **Regression first.**  The underlying quantity is continuous; bucketing it into
  three classes discards information and flatters the score.
* **Data-driven thresholds.**  Class cut-points come from training-set quantiles,
  not from constants tuned against synthetic craters.

    python3 -m scripts.train_hazard_model
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, f1_score, mean_absolute_error, r2_score

DATASET = Path("data/stage2/hazard_dataset.csv")
FIGURES = Path("figures")
TARGET = "toppled_fraction"

# Terrain statistics only. Every one is computable in a single pass without
# running the automaton -- which is the entire point of a surrogate. Nothing
# derived from the simulation may appear here.
FEATURES = [
    "relief_m", "elevation_std_m",
    "slope_mean", "slope_std", "slope_p90", "slope_p99", "slope_max",
    "unstable_fraction", "excess_total_m", "excess_max_m",
]


def evaluate_regression(name: str, y_true, y_pred) -> dict:
    finite = np.isfinite(y_pred)
    y_true, y_pred = np.asarray(y_true)[finite], np.asarray(y_pred)[finite]
    rho = spearmanr(y_true, y_pred).statistic if len(np.unique(y_true)) > 1 else np.nan
    return {
        "model": name,
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Spearman": float(rho),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    args = parser.parse_args()

    if not args.dataset.is_file():
        print(f"No dataset at {args.dataset}.\nRun: python3 -m scripts.build_hazard_dataset")
        return 1
    frame = pd.read_csv(args.dataset)

    train = frame[frame.hemisphere == "nearside"]
    test = frame[frame.hemisphere == "farside"]
    print(f"Dataset  {len(frame)} patches from real LOLA+Kaguya topography")
    print(f"Split    train = nearside ({len(train)}), test = farside ({len(test)})")
    print(f"Budget   {frame.converged.mean() * 100:.1f}% of runs converged within budget")
    if len(train) < 20 or len(test) < 20:
        print("\nToo few patches per hemisphere for a meaningful split.")
        return 1

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    results = [evaluate_regression("baseline: train mean", y_test,
                                   np.full(len(y_test), y_train.mean()))]

    single = LinearRegression().fit(train[["unstable_fraction"]], y_train)
    results.append(evaluate_regression(
        "baseline: unstable_fraction only", y_test,
        single.predict(test[["unstable_fraction"]])))

    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("\nxgboost is not installed; run: pip install -r requirements.txt")
        return 1

    model = XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, random_state=42,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    results.append(evaluate_regression("XGBoost (all terrain features)", y_test, predictions))

    print("\n--- Regression on held-out farside ---")
    table = pd.DataFrame(results).set_index("model")
    print(table.to_string(float_format=lambda v: f"{v: .4f}"))

    best_baseline = max(row["R2"] for row in results[:2])
    gain = results[-1]["R2"] - best_baseline
    verdict = (
        f"XGBoost improves R2 by {gain:+.4f} over the best baseline."
        if gain > 0.01 else
        f"XGBoost does NOT meaningfully beat the baseline ({gain:+.4f} R2). "
        "The cheap terrain statistics already carry the signal."
    )
    print(f"\n{verdict}")

    # --- Classification, with cut-points taken from the training distribution ---
    low, high = float(y_train.quantile(0.60)), float(y_train.quantile(0.90))
    print(f"\n--- Classification (thresholds from nearside quantiles: "
          f"{low:.6f}, {high:.6f}) ---")
    def grade(values):
        return np.digitize(values, [low, high])
    truth, predicted = grade(y_test), grade(predictions)
    labels = ["0 stable", "1 marginal", "2 active"]
    present = sorted(set(truth) | set(predicted))
    print(classification_report(
        truth, predicted, labels=present,
        target_names=[labels[i] for i in present], zero_division=0))
    macro = f1_score(truth, predicted, average="macro")
    print(f"Macro F1 (held-out farside): {macro:.3f}")

    # --- Figure ---
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    fig.suptitle(
        "Stage 2 hazard surrogate — trained on nearside, evaluated on farside",
        fontsize=14, fontweight="bold")

    axes[0].scatter(y_test, predictions, s=22, alpha=0.65, color="#1f77b4",
                    edgecolor="none")
    span = [0, float(max(y_test.max(), predictions.max())) * 1.05]
    axes[0].plot(span, span, "k--", linewidth=1.2, label="perfect")
    axes[0].set_xlabel("simulated fraction of cells shedding material")
    axes[0].set_ylabel("surrogate prediction")
    axes[0].set_title(f"R² = {results[-1]['R2']:.3f}   ρ = {results[-1]['Spearman']:.3f}",
                      fontsize=11)
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    names = [row["model"].replace("baseline: ", "") for row in results]
    scores = [row["R2"] for row in results]
    bars = axes[1].barh(names, scores,
                        color=["#999999", "#999999", "#c1440e"])
    axes[1].bar_label(bars, fmt="%.3f", fontsize=9, padding=3)
    axes[1].set_xlabel("R² on held-out farside")
    axes[1].set_title("Against baselines", fontsize=11)
    axes[1].grid(axis="x", alpha=0.3)

    importance = pd.Series(model.feature_importances_, index=FEATURES).nlargest(8)
    axes[2].barh(importance.index[::-1], importance.values[::-1], color="#1f77b4")
    axes[2].set_xlabel("gain importance")
    axes[2].set_title("What the surrogate uses", fontsize=11)
    axes[2].grid(axis="x", alpha=0.3)

    fig.text(0.005, 0.01,
             f"Surrogate for the cellular automaton, not a landslide predictor: the target is "
             f"simulator output, and no observed mass-wasting event is used as ground truth.\n"
             f"{len(train)} nearside training patches, {len(test)} farside test patches, "
             f"128 px at 59 m/px from LOLA+Kaguya.  {verdict}\n"
             f"generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}",
             fontsize=7.5, family="monospace", color="#333333")
    fig.subplots_adjust(bottom=0.24, top=0.87, wspace=0.32)
    target_path = FIGURES / "hazard_surrogate_evaluation.png"
    fig.savefig(target_path, dpi=140)
    plt.close(fig)
    print(f"\nWrote {target_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
