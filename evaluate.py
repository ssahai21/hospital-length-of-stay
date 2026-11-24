import argparse
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

FEATURES = [
    "Glucose", "Oxygen Saturation", "Cholesterol", "HbA1c",
    "Physical Activity", "Stress Level", "Diet Score", "Sleep Hours",
]
TARGET = "LengthOfStay"
SEED = 42
DEFAULT_DATA = Path(__file__).resolve().parent / "data" / "dirty_v3_path.csv"


def main():
    parser = argparse.ArgumentParser(description="Evaluate hospital length-of-stay models.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"CSV not found at {args.data}. See README.md for the download link.")

    data = pd.read_csv(args.data)
    missing_columns = sorted(set([*FEATURES, TARGET]) - set(data.columns))
    if missing_columns:
        parser.error(f"CSV is missing required columns: {', '.join(missing_columns)}")
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES], data[TARGET], test_size=0.2, random_state=SEED
    )
    models = {
        "Median baseline": DummyRegressor(strategy="median"),
        "Random forest": RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
        "Random forest with median imputation": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
        ),
    }
    print(f"Training rows: {len(y_train):,} | Test rows: {len(y_test):,}")
    print(f"Training median: {y_train.median():g} days")
    for name, model in models.items():
        model.fit(X_train, y_train)
        mae = mean_absolute_error(y_test, model.predict(X_test))
        print(f"{name}: {mae:.3f} days MAE")


if __name__ == "__main__":
    main()
