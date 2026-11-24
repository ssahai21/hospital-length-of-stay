"""Exploratory paired tests on the previously inspected length-of-stay holdout."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap, permutation_test
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, mean_pinball_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from evaluate import DEFAULT_DATA, FEATURES, SEED, TARGET

REFERENCE = "Original unweighted random forest"
BOOSTING_CLASSIFIER = "Histogram gradient boosting classifier"
MLP_CLASSIFIER = "MLP classifier (32, 16)"
P90 = "Histogram quantile boosting P90 (upper estimate)"
PERMUTATIONS = 49_999
BOOTSTRAPS = 9_999


def forest():
    return RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)


def mlp(estimator):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        estimator(hidden_layer_sizes=(32, 16), early_stopping=True, random_state=SEED),
    )


def predict_models(X_train, X_test, y_train):
    long_train = y_train > 7
    models = {
        REFERENCE: forest(),
        "Training-median baseline": DummyRegressor(strategy="median"),
        "Random forest with median imputation": make_pipeline(SimpleImputer(strategy="median"), forest()),
        "Random forest, labs and vitals": forest(),
        "Random forest, lifestyle": forest(),
        "Weighted random forest": forest(),
        "Vanilla histogram boosting": HistGradientBoostingRegressor(random_state=SEED),
        "Histogram quantile boosting P50": HistGradientBoostingRegressor(
            loss="quantile", quantile=0.5, random_state=SEED
        ),
        P90: HistGradientBoostingRegressor(loss="quantile", quantile=0.9, random_state=SEED),
        "Linear ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0)),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
        "Poisson histogram boosting": HistGradientBoostingRegressor(loss="poisson", random_state=SEED),
        "Spline ridge": make_pipeline(
            SimpleImputer(strategy="median"),
            SplineTransformer(n_knots=5, degree=3, knots="quantile", include_bias=False),
            StandardScaler(),
            Ridge(alpha=1.0),
        ),
        "Direct MLP": mlp(MLPRegressor),
    }
    feature_sets = {
        "Random forest, labs and vitals": FEATURES[:4],
        "Random forest, lifestyle": FEATURES[4:],
    }
    predictions = {}
    for name, model in models.items():
        columns = feature_sets.get(name, FEATURES)
        fit_params = {"sample_weight": np.where(long_train, 2.0, 1.0)} if name == "Weighted random forest" else {}
        predictions[name] = model.fit(X_train[columns], y_train, **fit_params).predict(X_test[columns])

    classifiers = {
        BOOSTING_CLASSIFIER: HistGradientBoostingClassifier(random_state=SEED),
        MLP_CLASSIFIER: mlp(MLPClassifier),
    }
    scores = {
        name: model.fit(X_train, long_train).predict_proba(X_test)[:, 1]
        for name, model in classifiers.items()
    }

    short_expert = HistGradientBoostingRegressor(random_state=SEED)
    long_expert = HistGradientBoostingRegressor(random_state=SEED)
    short_days = short_expert.fit(X_train.loc[~long_train], y_train.loc[~long_train]).predict(X_test)
    long_days = long_expert.fit(X_train.loc[long_train], y_train.loc[long_train]).predict(X_test)
    for classifier, label in [(BOOSTING_CLASSIFIER, "boosting"), (MLP_CLASSIFIER, "MLP")]:
        probability = scores[classifier]
        predictions[f"Two-stage with {label} classifier"] = (1 - probability) * short_days + probability * long_days
    return predictions, scores


def interval(values):
    return np.quantile(values, [0.025, 0.975]).tolist()


def compare_errors(actual, predictions):
    long_stays = actual > 7
    reference_absolute_error = np.abs(predictions[REFERENCE][long_stays] - actual[long_stays])
    results = {}
    for name, estimates in predictions.items():
        errors = estimates - actual
        long_errors = errors[long_stays]
        differences = np.abs(long_errors) - reference_absolute_error
        result = {
            "overall_mae_days": float(np.abs(errors).mean()),
            "long_stay_mae_days": float(np.abs(long_errors).mean()),
            "long_stay_signed_error_days": float(long_errors.mean()),
        }
        if name != REFERENCE:
            # Sign-flipping a paired difference is equivalent to swapping the two losses for that row.
            test = permutation_test(
                (differences,), np.mean, permutation_type="samples", alternative="two-sided",
                n_resamples=PERMUTATIONS, batch=512, rng=np.random.default_rng(SEED),
            )
            ci = bootstrap(
                (differences,), np.mean, method="percentile", n_resamples=BOOTSTRAPS,
                batch=512, rng=np.random.default_rng(SEED),
            ).confidence_interval
            result.update({
                "long_stay_mae_difference_vs_forest_days": float(differences.mean()),
                "paired_bootstrap_95_ci_days": [float(ci.low), float(ci.high)],
                "p_value_raw": float(test.pvalue),
            })
        results[name] = result

    comparisons = sorted(
        (result for name, result in results.items() if name != REFERENCE),
        key=lambda result: result["p_value_raw"],
    )
    adjusted = 0.0
    for remaining, result in zip(range(len(comparisons), 0, -1), comparisons):
        adjusted = min(1.0, max(adjusted, remaining * result["p_value_raw"]))
        result["p_value_holm"] = adjusted
    return results


def compare_classifiers(labels, scores):
    rng = np.random.default_rng(SEED)
    score_matrix = np.column_stack(list(scores.values()))
    boot = np.empty((BOOTSTRAPS, score_matrix.shape[1]))
    for replicate in range(BOOTSTRAPS):
        rows = rng.integers(len(labels), size=len(labels))
        boot[replicate] = [
            average_precision_score(labels[rows], column[rows]) for column in score_matrix.T
        ]
    estimates = {name: float(average_precision_score(labels, values)) for name, values in scores.items()}
    return {
        "held_out_prevalence": float(labels.mean()),
        "models": {
            name: {"average_precision": estimates[name], "paired_bootstrap_95_ci": interval(boot[:, i])}
            for i, name in enumerate(scores)
        },
        "mlp_minus_boosting": {
            "average_precision_difference": estimates[MLP_CLASSIFIER] - estimates[BOOSTING_CLASSIFIER],
            "paired_bootstrap_95_ci": interval(boot[:, 1] - boot[:, 0]),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().with_name("exploratory_significance.json")
    )
    args = parser.parse_args()

    data = pd.read_csv(args.data, usecols=[*FEATURES, TARGET])
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES], data[TARGET], test_size=0.2, random_state=SEED
    )
    actual = y_test.to_numpy()
    long_stays = actual > 7
    predictions, scores = predict_models(X_train, X_test, y_train)
    results = compare_errors(actual, predictions)
    upper = predictions[P90]
    report = {
        "analysis_status": "exploratory",
        "reason": "The held-out split was previously inspected.",
        "reference_model": REFERENCE,
        "methods": {
            "seed": SEED,
            "test_size": 0.2,
            "permutation": "Two-sided paired permutation of absolute losses on actual stays over seven days",
            "permutation_resamples": PERMUTATIONS,
            "multiple_testing": "Holm across all length-of-stay comparisons against the reference",
            "holm_comparisons": len(results) - 1,
            "bootstrap": "Paired row resampling with replacement, 95% percentile intervals, conditional on fitted models",
            "bootstrap_resamples": BOOTSTRAPS,
            "difference_direction": "Model MAE minus forest MAE; negative favors the model",
            "signed_error_direction": "Prediction minus actual",
        },
        "subgroup_counts": {
            "training_total": len(y_train),
            "held_out_total": len(actual),
            "held_out_up_to_3_days": int((actual <= 3).sum()),
            "held_out_over_3_through_7_days": int(((actual > 3) & (actual <= 7)).sum()),
            "held_out_over_7_days": int(long_stays.sum()),
        },
        "length_of_stay_models": results,
        "upper_quantile_diagnostics": {
            "model": P90,
            "overall_exceedance_rate": float((actual > upper).mean()),
            "long_stay_exceedance_rate": float((actual[long_stays] > upper[long_stays]).mean()),
            "pinball_loss_days": float(mean_pinball_loss(actual, upper, alpha=0.9)),
            "interpretation": "P90 is an upper estimate; judge its quantile performance using exceedance and pinball loss.",
        },
        "long_stay_classifiers": compare_classifiers(long_stays, scores),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Exploratory held-out results written to {args.output}")


if __name__ == "__main__":
    main()
