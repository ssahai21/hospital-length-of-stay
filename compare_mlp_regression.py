"""Compare direct MLP regression and two-stage models on identical training folds."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import DEFAULT_DATA, FEATURES, SEED, TARGET


def mlp_pipeline(estimator):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        estimator(hidden_layer_sizes=(32, 16), early_stopping=True, random_state=SEED),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    data = pd.read_csv(args.data, usecols=[*FEATURES, TARGET])
    X, _, y, _ = train_test_split(data[FEATURES], data[TARGET], test_size=0.2, random_state=SEED)
    folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
    direct = np.empty(len(y))
    classifiers = {
        "Two-stage with boosting classifier": HistGradientBoostingClassifier(random_state=SEED),
        "Two-stage with MLP classifier": mlp_pipeline(MLPClassifier),
    }
    combined = {name: np.empty(len(y)) for name in classifiers}
    for train, validation in folds.split(X):
        X_fit, y_fit = X.iloc[train], y.iloc[train]
        X_validation = X.iloc[validation]
        long_fit = y_fit > 7
        direct[validation] = mlp_pipeline(MLPRegressor).fit(X_fit, y_fit).predict(X_validation)
        short_model = HistGradientBoostingRegressor(random_state=SEED).fit(
            X_fit.loc[~long_fit], y_fit.loc[~long_fit]
        )
        long_model = HistGradientBoostingRegressor(random_state=SEED).fit(
            X_fit.loc[long_fit], y_fit.loc[long_fit]
        )
        short_predictions = short_model.predict(X_validation)
        long_predictions = long_model.predict(X_validation)
        for name, classifier in classifiers.items():
            p_long = classifier.fit(X_fit, long_fit).predict_proba(X_validation)[:, 1]
            combined[name][validation] = (1 - p_long) * short_predictions + p_long * long_predictions

    long_stays = y > 7
    short_stays = y <= 3
    print(f"Training rows: {len(y):,} | Stays over 7 days: {long_stays.sum():,}")
    print("Model | Overall MAE | Over-7-day MAE | Over-7-day signed error | Up-to-3-day signed error")
    for name, predictions in {"Direct MLP": direct, **combined}.items():
        errors = predictions - y
        print(
            f"{name} | {errors.abs().mean():.6f} | {errors[long_stays].abs().mean():.6f} | "
            f"{errors[long_stays].mean():.6f} | {errors[short_stays].mean():.6f}"
        )


if __name__ == "__main__":
    main()
