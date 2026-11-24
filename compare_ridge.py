"""Evaluate ridge regression with fold-local imputation and scaling."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import DEFAULT_DATA, FEATURES, SEED, TARGET, summarize_errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()

    data = pd.read_csv(args.data, usecols=[*FEATURES, TARGET])
    X, _, y, _ = train_test_split(data[FEATURES], data[TARGET], test_size=0.2, random_state=SEED)
    folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    predictions = cross_val_predict(model, X, y, cv=folds)
    print(f"Training rows: {len(y):,} | Ridge MAE: {mean_absolute_error(y, predictions):.3f} days")
    print(summarize_errors(y, predictions).to_string(float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
