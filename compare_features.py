"""Compare feature sets using out-of-fold predictions on the training split."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
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
    forest = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
    experiments = [
        ("Median baseline", FEATURES, DummyRegressor(strategy="median")),
        ("Labs and vitals", FEATURES[:4], forest),
        ("Lifestyle", FEATURES[4:], forest),
        ("All features", FEATURES, forest),
    ]
    long_stays = y_train > 7
    print(f"Training rows: {len(y_train):,}; stays over 7 days: {long_stays.sum():,}")
    results = []
    for name, columns, model in experiments:
        predictions = cross_val_predict(model, X_train[columns], y_train, cv=folds)
        long_errors = predictions[long_stays] - y_train[long_stays]
        results.append({
            "Model": name,
            "Overall MAE": mean_absolute_error(y_train, predictions),
            "Long-stay MAE": long_errors.abs().mean(),
            "Long-stay bias": long_errors.mean(),
        })
    print(pd.DataFrame(results).to_string(index=False, float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
