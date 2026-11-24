"""Evaluate a probability-weighted mixture of short- and long-stay models."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error, precision_recall_curve
from sklearn.model_selection import KFold, train_test_split

from evaluate import DEFAULT_DATA, FEATURES, SEED, TARGET, summarize_errors


def save_precision_recall(path, actual, probabilities):
    precision, recall, _ = precision_recall_curve(actual, probabilities)
    average_precision = average_precision_score(actual, probabilities)
    prevalence = actual.mean()
    points = " ".join(f"{60 + 500 * x:.2f},{360 - 300 * y:.2f}" for x, y in zip(recall, precision))
    baseline_y = 360 - 300 * prevalence
    ticks = "".join(
        f'<text x="{60 + 100 * i}" y="382" text-anchor="middle">{i / 5:.1f}</text>'
        f'<text x="50" y="{364 - 60 * i}" text-anchor="end">{i / 5:.1f}</text>'
        for i in range(6)
    )
    path.write_text(
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="600" height="430" viewBox="0 0 600 430"
role="img" aria-labelledby="title">
<title id="title">Precision–recall for stays over 7 days</title>
<rect width="600" height="430" fill="white"/>
<g font-family="sans-serif" font-size="13" fill="#17263a">
<text x="60" y="24" font-size="17">Precision–recall for stays over 7 days</text>
<text x="60" y="44">Average precision {average_precision:.3f} · Dashed baseline {prevalence:.3f}</text>
<path d="M60 60 V360 H560" fill="none" stroke="#17263a"/>
<path d="M60 {baseline_y} H560" stroke="#596779" stroke-dasharray="5 4"/>
<polyline points="{points}" fill="none" stroke="#1265a8" stroke-width="1.5"/>
{ticks}
<text x="310" y="410" text-anchor="middle">Recall</text>
<text transform="translate(17 210) rotate(-90)" text-anchor="middle">Precision</text>
</g></svg>''',
        encoding="utf-8",
    )
    print(f"Average precision: {average_precision:.3f} | Positive-class prevalence: {prevalence:.3f}")
    print(f"Precision–recall curve: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--curve", type=Path, default=Path("data/long_stay_precision_recall.svg"))
    args = parser.parse_args()

    data = pd.read_csv(args.data, usecols=[*FEATURES, TARGET])
    X, _, y, _ = train_test_split(data[FEATURES], data[TARGET], test_size=0.2, random_state=SEED)
    long_stays = y > 7
    probabilities = np.empty(len(y))
    combined = np.empty(len(y))
    baseline = np.empty(len(y))
    folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for train, validation in folds.split(X):
        X_fit, y_fit = X.iloc[train], y.iloc[train]
        X_validation = X.iloc[validation]
        long_fit = y_fit > 7
        classifier = HistGradientBoostingClassifier(random_state=SEED).fit(X_fit, long_fit)
        p_long = classifier.predict_proba(X_validation)[:, 1]
        short_model = HistGradientBoostingRegressor(random_state=SEED).fit(
            X_fit.loc[~long_fit], y_fit.loc[~long_fit]
        )
        long_model = HistGradientBoostingRegressor(random_state=SEED).fit(
            X_fit.loc[long_fit], y_fit.loc[long_fit]
        )
        probabilities[validation] = p_long
        combined[validation] = (
            (1 - p_long) * short_model.predict(X_validation) + p_long * long_model.predict(X_validation)
        )
        baseline[validation] = (
            HistGradientBoostingRegressor(random_state=SEED).fit(X_fit, y_fit).predict(X_validation)
        )

    print(f"Training rows: {len(y):,}")
    for name, predictions in [("Two-stage boosting", combined), ("Vanilla boosting", baseline)]:
        print(f"{name}: {mean_absolute_error(y, predictions):.3f} days MAE")
        print(summarize_errors(y, predictions).to_string(float_format="{:.3f}".format))
    args.curve.parent.mkdir(parents=True, exist_ok=True)
    save_precision_recall(args.curve, long_stays, probabilities)


if __name__ == "__main__":
    main()
