"""Compare all-feature models using out-of-fold predictions on the training split."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_predict, train_test_split

from evaluate import DEFAULT_DATA, FEATURES, SEED, TARGET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    X_train, _, y_train, _ = train_test_split(
        data[FEATURES], data[TARGET], test_size=0.2, random_state=SEED
    )
    folds = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(X_train))
    long_stays = y_train > 7
    models = {
        "Random forest": RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
        "Histogram gradient boosting": HistGradientBoostingRegressor(random_state=SEED),
    }

    print(f"Training rows: {len(y_train):,}; stays over 7 days: {long_stays.sum():,}")
    print("Model | Overall MAE | Long-stay MAE | Long-stay signed error")
    for name, model in models.items():
        predictions = cross_val_predict(model, X_train, y_train, cv=folds)
        long_errors = predictions[long_stays] - y_train[long_stays]
        print(
            f"{name} | {mean_absolute_error(y_train, predictions):.6f} | "
            f"{long_errors.abs().mean():.6f} | {long_errors.mean():.6f}"
        )


if __name__ == "__main__":
    main()
