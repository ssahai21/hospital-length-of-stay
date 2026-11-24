# Hospital Length-of-Stay Prediction

I built a machine learning pipeline to predict hospital length of stay from eight features in a synthetic healthcare dataset. A random forest beat a training-median baseline, but its overall score hid a consistent failure: it underestimated stays longer than seven days by about **5.19 days on average**. I tested whether feature selection, sample weighting, gradient boosting, and neural networks could reduce that error.

## Key findings

On the 6,000-row holdout, the original random forest reduced overall mean absolute error (MAE) from **1.858 to 1.639 days**, about a **12% improvement** over predicting the training median for every row. On the 518 stays longer than seven days, its MAE was **5.210 days**.

| Model | Overall MAE | MAE for stays >7 days | Mean signed error for stays >7 days |
| --- | ---: | ---: | ---: |
| Training-median baseline | 1.858 | 6.938 | −6.938 |
| Random forest | 1.639 | 5.210 | −5.187 |
| Weighted random forest | 1.629 | 5.211 | −5.179 |
| Histogram gradient boosting | 1.603 | 5.391 | −5.383 |
| Quantile boosting P50 | 1.560 | 5.982 | −5.982 |
| Direct neural network (MLP) | 1.624 | 5.178 | −5.169 |
| Two-stage model with an MLP classifier | 1.603 | 5.232 | −5.204 |

All errors are in days. Signed error is prediction minus actual, so a negative value means underprediction.

- **Lower overall MAE came with larger long-stay errors.** Histogram gradient boosting increased long-stay MAE by 0.182 days versus the original forest (paired 95% bootstrap CI: 0.122 to 0.245 days). The P50 model had the lowest overall MAE among the point forecasts in the table and the highest long-stay MAE among the learned models shown.
- **Doubling the weight of long stays barely changed their error.** The weighted forest changed long-stay MAE by +0.002 days (paired 95% bootstrap CI: −0.054 to +0.056 days).
- **Better classification didn’t clearly improve duration estimates.** The MLP classifier achieved 0.263 average precision versus 0.240 for the boosting classifier, with a long-stay prevalence of 0.086. The two-stage MLP model’s long-stay MAE difference versus the original forest was +0.022 days (paired 95% bootstrap CI: −0.048 to +0.093 days).

I also evaluated a **P90 upper forecast** separately. Actual stays exceeded it 11.2% of the time overall and 79.5% of the time among stays that actually lasted over seven days. The subgroup result shows that an upper forecast can have close-to-target coverage across the full dataset while still falling short on many of the longest stays.

## Evaluation design

- **Target:** `LengthOfStay`, measured in days.
- **Inputs:** Glucose, oxygen saturation, cholesterol, HbA1c, physical activity, stress level, diet score, and sleep hours.
- **Data split:** 24,000 training rows and 6,000 held-out rows using a fixed 80/20 split with seed `42`. Development comparisons used the same five shuffled folds within the training split.
- **Statistical comparisons:** Each candidate was compared with the original random forest on the same held-out rows. The evaluation reports paired bootstrap intervals for differences in long-stay MAE and two-sided paired permutation tests with Holm-adjusted p-values. The holdout had already been inspected during error analysis, so these comparisons are exploratory.

See [the full results JSON](exploratory_significance.json) for the other model variants, subgroup counts, confidence intervals, adjusted p-values, and classifier metrics.

## Reproduce the results

Use Python 3.12. From the repository root on macOS or Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download the [Healthcare Risk Factors Dataset](https://www.kaggle.com/datasets/abdallaahmed77/healthcare-risk-factors-dataset) and place `dirty_v3_path.csv` in `data/`.

```bash
python evaluate.py --by-stay
python compare_features.py
python compare_boosting.py
python compare_weighted_forest.py
python evaluate_significance.py --output exploratory_significance.json
```

Each script accepts `--data /path/to/dirty_v3_path.csv` if your CSV is stored elsewhere. `length_of_stay.ipynb` is the original Colab prototype and uses a separate Excel copy.

## Limitations

The dataset is synthetic and contains missing glucose values and measurements outside plausible ranges. I did not verify whether all features would be available at hospital admission. The results describe performance on this dataset, and performance on real hospital records remains untested.
