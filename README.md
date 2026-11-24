# Hospital Length-of-Stay Prediction

A reproducible comparison of a random forest and a median baseline on synthetic hospital data.

## Run in VS Code

Open the cloned repository as a folder. In the integrated terminal, create and activate a Python 3.12 environment (macOS/Linux).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download the [Healthcare Risk Factors Dataset](https://www.kaggle.com/datasets/abdallaahmed77/healthcare-risk-factors-dataset) and extract `dirty_v3_path.csv` into `data/`. Then run:

```bash
python evaluate.py
```

You can also pass `--data /path/to/dirty_v3_path.csv`. The historical Colab notebook used a separate Excel copy.

## Results

The fixed 80/20 split (seed 42) gives 24,000 training rows and 6,000 test rows.

| Model | Test MAE (days) |
| --- | ---: |
| Training median (4 days) | 1.858 |
| Random forest | 1.639 |
| Random forest with training-median imputation | 1.641 |

The raw synthetic CSV has missing glucose values and out-of-range measurements. These experiments measure performance on this dataset. Feature availability at admission and the preparation of the original Excel copy remain unverified.
