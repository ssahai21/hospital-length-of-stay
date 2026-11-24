"""Evaluate median and upper-quantile boosting predictions on the training split."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_pinball_loss
from sklearn.model_selection import KFold, cross_val_predict, train_test_split

from evaluate import DEFAULT_DATA, FEATURES, SEED, TARGET, summarize_errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()

    data = pd.read_csv(args.data, usecols=[*FEATURES, TARGET])
    X, _, y, _ = train_test_split(data[FEATURES], data[TARGET], test_size=0.2, random_state=SEED)
    folds = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(X))
    median, upper = (
        cross_val_predict(
            HistGradientBoostingRegressor(loss="quantile", quantile=q, random_state=SEED),
            X, y, cv=folds,
        )
        for q in (0.5, 0.9)
    )
    # The fold-specific empirical quantile is a feature-free upper-estimate baseline.
    upper_baseline = np.empty(len(y))
    for train, validation in folds:
        upper_baseline[validation] = y.iloc[train].quantile(0.9)

    long_stays = y > 7
    print(f"Training rows: {len(y):,} | Median prediction MAE: {mean_absolute_error(y, median):.3f} days")
    print(summarize_errors(y, median).to_string(float_format="{:.3f}".format))
    for name, predictions in [("P90 boosting", upper), ("Training-fold P90", upper_baseline)]:
        exceeds = y > predictions
        print(
            f"{name}: {exceeds.mean():.2%} exceed overall, "
            f"{exceeds[long_stays].mean():.2%} exceed among stays over 7 days, "
            f"P90 pinball loss {mean_pinball_loss(y, predictions, alpha=0.9):.3f} days"
        )
    print(f"P90 below predicted median: {(upper < median).mean():.2%} of rows")


if __name__ == "__main__":
    main()
