"""Compare long-stay weighting with unweighted models on the training split."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict, train_test_split

from evaluate import DEFAULT_DATA, FEATURES, SEED, TARGET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()

    data = pd.read_csv(args.data, usecols=[*FEATURES, TARGET])
    X_train, _, y_train, _ = train_test_split(
        data[FEATURES], data[TARGET], test_size=0.2, random_state=SEED
    )
    folds = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(X_train))
    long_stays = y_train > 7
    short_stays = y_train <= 3
    sample_weight = long_stays.map({False: 1.0, True: 2.0}).to_numpy()
    forest = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
    experiments = [
        ("Unweighted random forest", forest, None),
        ("Weighted random forest", forest, {"sample_weight": sample_weight}),
        ("Histogram gradient boosting", HistGradientBoostingRegressor(random_state=SEED), None),
    ]

    results = []
    for name, model, fit_params in experiments:
        # cross_val_predict subsets sample weights to each fold's training rows.
        predictions = cross_val_predict(model, X_train, y_train, cv=folds, params=fit_params)
        errors = predictions - y_train
        results.append({
            "Model": name,
            "Overall MAE": errors.abs().mean(),
            "Over-7-day MAE": errors[long_stays].abs().mean(),
            "Over-7-day signed error": errors[long_stays].mean(),
            "Up-to-3-day signed error": errors[short_stays].mean(),
        })

    print(f"Training rows: {len(y_train):,}; stays over 7 days: {long_stays.sum():,}; "
          f"stays up to 3 days: {short_stays.sum():,}")
    print(pd.DataFrame(results).to_string(index=False, float_format="{:.6f}".format))


if __name__ == "__main__":
    main()
