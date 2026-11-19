"""Compare logistic regression and XGBoost, evaluate on later months, explain the winner with SHAP.

    python -m ldp.train
"""

import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from ldp.features import CATEGORICAL, FEATURES, NUMERIC, build_dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TRAIN_FROM, VALIDATION_FROM, TEST_FROM, TEST_UNTIL = "2017-01-01", "2018-02-01", "2018-05-01", "2018-08-01"
FLAG_SHARE = 0.10


def make_pipeline(model) -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100, sparse_output=False)
    pre = ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)],
                            verbose_feature_names_out=False)
    return Pipeline([("pre", pre), ("model", model)])


def candidates(pos_weight: float) -> dict:
    return {
        "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=3000),
        "xgboost": XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, min_child_weight=5, scale_pos_weight=pos_weight,
                                 tree_method="hist", eval_metric="aucpr", n_jobs=-1, random_state=42),
    }


def split(df: pd.DataFrame):
    df = df.sort_values("purchased_at")
    train = df[(df["purchased_at"] >= TRAIN_FROM) & (df["purchased_at"] < TEST_FROM)]
    test = df[(df["purchased_at"] >= TEST_FROM) & (df["purchased_at"] < TEST_UNTIL)]
    return train, test


def flag_threshold(model, train: pd.DataFrame) -> float:
    """Score cut-off that flags the riskiest 10%, set on months the model was not trained on."""
    early = train[train["purchased_at"] < VALIDATION_FROM]
    validation = train[train["purchased_at"] >= VALIDATION_FROM]
    pipeline = make_pipeline(clone(model)).fit(early[FEATURES], early["late"])
    return float(np.quantile(pipeline.predict_proba(validation[FEATURES])[:, 1], 1 - FLAG_SHARE))


def flagged(threshold, scores, y) -> dict:
    flag = scores >= threshold
    top = scores >= np.quantile(scores, 1 - FLAG_SHARE)
    return {"threshold": threshold, "share_flagged": float(flag.mean()), "precision": float(y[flag].mean()),
            "recall": float(flag[y == 1].mean()), "top_10_percent_precision": float(y[top].mean()),
            "top_10_percent_recall": float(top[y == 1].mean())}


def main():
    df = build_dataset(ROOT / "data" / "raw")
    train, test = split(df)
    x_train, y_train = train[FEATURES], train["late"].to_numpy()
    x_test, y_test = test[FEATURES], test["late"].to_numpy()
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    report = {"orders": len(df), "late_rate": float(df["late"].mean()),
              "train": {"from": TRAIN_FROM, "until": TEST_FROM, "orders": len(train), "late_rate": float(y_train.mean())},
              "test": {"from": TEST_FROM, "until": TEST_UNTIL, "orders": len(test), "late_rate": float(y_test.mean())},
              "models": {}}
    fitted = {}
    for name, model in candidates(pos_weight).items():
        pipeline = make_pipeline(model)
        cv = cross_validate(pipeline, x_train, y_train, cv=TimeSeriesSplit(n_splits=5),
                            scoring=["roc_auc", "average_precision"], n_jobs=1)
        pipeline.fit(x_train, y_train)
        scores = pipeline.predict_proba(x_test)[:, 1]
        report["models"][name] = {
            "cv_roc_auc_mean": float(cv["test_roc_auc"].mean()), "cv_roc_auc_std": float(cv["test_roc_auc"].std()),
            "cv_pr_auc_mean": float(cv["test_average_precision"].mean()),
            "test_roc_auc": float(roc_auc_score(y_test, scores)),
            "test_pr_auc": float(average_precision_score(y_test, scores)),
            "flag_top_10_percent": flagged(flag_threshold(model, train), scores, y_test),
        }
        fitted[name] = (pipeline, scores)
        print(name, json.dumps(report["models"][name], indent=2))

    best = max(report["models"], key=lambda n: report["models"][n]["cv_pr_auc_mean"])
    report["chosen"] = best
    pipeline, scores = fitted[best]
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipeline, "threshold": report["models"][best]["flag_top_10_percent"]["threshold"]},
                ROOT / "models" / "model.joblib")

    figures = ROOT / "reports" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4.5))
    for name, (_, s) in fitted.items():
        precision, recall, _ = precision_recall_curve(y_test, s)
        plt.plot(recall, precision, label=f"{name} (PR-AUC {average_precision_score(y_test, s):.2f})")
    plt.axhline(y_test.mean(), color="grey", linestyle="--", label=f"base rate {y_test.mean():.2f}")
    plt.xlabel("Recall (share of late orders caught)")
    plt.ylabel("Precision (share of flags that were late)")
    plt.title("Held-out months: May to July 2018")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "precision_recall.png", dpi=150)
    plt.close()

    xgb_pipeline = fitted["xgboost"][0]
    sample = x_test.sample(n=min(3000, len(x_test)), random_state=42)
    transformed = pd.DataFrame(xgb_pipeline["pre"].transform(sample),
                               columns=xgb_pipeline["pre"].get_feature_names_out())
    values = shap.TreeExplainer(xgb_pipeline["model"]).shap_values(transformed)
    importance = pd.Series(np.abs(values).mean(axis=0), index=transformed.columns).sort_values(ascending=False)
    report["shap_mean_abs_top"] = {k: round(float(v), 4) for k, v in importance.head(12).items()}
    shap.summary_plot(values, transformed, max_display=12, show=False)
    plt.title("What pushes an order towards late (XGBoost, SHAP)")
    plt.tight_layout()
    plt.savefig(figures / "shap_summary.png", dpi=150)
    plt.close()

    (ROOT / "reports" / "metrics.json").write_text(json.dumps(report, indent=2))
    print("chosen:", best)


if __name__ == "__main__":
    main()
