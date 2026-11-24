"""Compare out-of-fold classifiers for stays over seven days."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import DEFAULT_DATA, FEATURES, SEED, TARGET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()

    data = pd.read_csv(args.data, usecols=[*FEATURES, TARGET])
    X, _, y, _ = train_test_split(data[FEATURES], data[TARGET], test_size=0.2, random_state=SEED)
    long_stays = y > 7
    folds = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(X))
    models = {
        "MLP (32, 16)": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(32, 16), early_stopping=True, random_state=SEED),
        ),
        "Histogram gradient boosting": HistGradientBoostingClassifier(random_state=SEED),
    }

    target_recall = 0.5
    print(f"Training rows: {len(y):,} | Long-stay prevalence: {long_stays.mean():.2%}")
    print(f"Model | Average precision | Precision at {target_recall:.0%} recall | Achieved recall")
    for name, model in models.items():
        probabilities = cross_val_predict(model, X, long_stays, cv=folds, method="predict_proba")[:, 1]
        precision, recall, _ = precision_recall_curve(long_stays, probabilities)
        # Thresholds are ascending. Use the highest threshold that attains the target recall.
        operating_point = (recall >= target_recall).nonzero()[0][-1]
        print(
            f"{name} | {average_precision_score(long_stays, probabilities):.4f} | "
            f"{precision[operating_point]:.2%} | {recall[operating_point]:.2%}"
        )


if __name__ == "__main__":
    main()
