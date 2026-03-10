"""
DrivenData competition template
Competition: <competition-name>
"""

import os
import pandas as pd
import numpy as np
import wandb
from pathlib import Path

COMP_DIR = Path(__file__).parent
DATA_DIR = COMP_DIR / "data"
SUBMISSION_FILE = COMP_DIR / "submission.csv"

PROJECT_NAME = "drivendata-<competition-slug>"
RUN_MEMO = os.environ.get("RUN_MEMO", "local")


def load_data():
    train = pd.read_csv(DATA_DIR / "train_values.csv", index_col=0)
    train_labels = pd.read_csv(DATA_DIR / "train_labels.csv", index_col=0)
    test = pd.read_csv(DATA_DIR / "test_values.csv", index_col=0)
    return train, train_labels, test


def train_model(X_train, y_train):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def main():
    run = wandb.init(
        project=PROJECT_NAME,
        name=RUN_MEMO,
        config={"model": "GradientBoosting", "n_estimators": 100},
    )

    train, train_labels, test = load_data()

    # --- Feature engineering ---
    X_train = train.select_dtypes(include=[np.number]).fillna(0)
    y_train = train_labels.iloc[:, 0]
    X_test = test.select_dtypes(include=[np.number]).fillna(0)

    # --- Train ---
    model = train_model(X_train, y_train)

    # --- Predict ---
    preds = model.predict(X_test)

    # --- Save submission ---
    submission = pd.DataFrame({"id": test.index, "label": preds})
    submission.to_csv(SUBMISSION_FILE, index=False)
    print(f"Saved: {SUBMISSION_FILE} ({len(submission)} rows)")

    wandb.log({"submission_rows": len(submission)})
    run.finish()


if __name__ == "__main__":
    main()
