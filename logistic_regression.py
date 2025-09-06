# -*- coding: utf-8 -*-
"""
HBV vs Non-Viral classifier — Logistic Regression version
"""

import os
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, accuracy_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

import matplotlib.pyplot as plt
import joblib

# ---------------- CONFIG VARIABLES ----------------
HBV_FILE = r"C:\Users\Admin\Desktop\IGEM Hackathon\data\HBV.tsv"
NON_VIRAL_FILE = r"C:\Users\Admin\Desktop\IGEM Hackathon\data\Non-Viral.tsv"
OUT_DIR  = "results_lr"

TEST_SIZE = 0.25  # Portion of data to use as test set
VAL_SIZE  = 0.20  # Portion of training set to use as validation set
SEED = 42  # Random seed for reproducibility
TOPK = 20  # Number of top features to save
CALIBRATE = True  # Whether to calibrate predicted probabilities
# ---------------------------------------------------

# Ensure output directory exists
def ensure_dir(d):
    if not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

# Load TSV file and check for 'Case' column
def load_tsv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(path, sep="\t", header=0, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    if "Case" not in df.columns:
        raise ValueError(f"'Case' column not found in {path}")
    df["Case"] = df["Case"].astype(str).str.strip()
    return df

# Convert all features to numeric, ignoring specified columns
def to_numeric_df(df, skip_cols):
    cols = [c for c in df.columns if c not in skip_cols]
    return df[cols].apply(pd.to_numeric, errors="coerce")

# Compute balanced sample weights to account for class imbalance
def compute_balanced_sample_weights(y):
    y = pd.Series(y)
    n = len(y)
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return np.ones(n, dtype=float)
    w_pos = 0.5 / n_pos  # Weight for HBV class
    w_neg = 0.5 / n_neg  # Weight for Non-Viral class
    weights = np.where(y.values == 1, w_pos, w_neg).astype(float)
    return weights * (n / weights.sum())  # Normalize sum of weights to n

# Choose best threshold based on F1 score (or Youden if specified)
def choose_best_threshold(y_true, y_proba, mode="f1"):
    fpr, tpr, roc_th = roc_curve(y_true, y_proba)
    prec, rec, pr_th = precision_recall_curve(y_true, y_proba)
    best_thr = 0.5
    best_score = -1.0

    if mode.lower() == "youden":
        youden = tpr - fpr
        idx = int(np.nanargmax(youden))
        best_thr = float(roc_th[idx])
    else:
        # Loop over all thresholds and pick the one maximizing F1
        for i, thr in enumerate(pr_th):
            y_pred = (y_proba >= thr).astype(int)
            score = f1_score(y_true, y_pred)
            if score > best_score:
                best_score = score
                best_thr = float(thr)
    return best_thr, best_score

def main():
    ensure_dir(OUT_DIR)

    print("Loading the data...")
    hbv_df = load_tsv(HBV_FILE)
    nonviral_df = load_tsv(NON_VIRAL_FILE)

    # Combine both datasets
    all_df = pd.concat([hbv_df, nonviral_df], axis=0, ignore_index=True)
    hbv_cases = set(hbv_df["Case"].tolist())
    all_df["is_HBV"] = all_df["Case"].apply(lambda x: 1 if x in hbv_cases else 0)

    # Convert features to numeric
    feature_cols = [c for c in all_df.columns if c not in ("Case", "is_HBV")]
    X = to_numeric_df(all_df, skip_cols=["Case", "is_HBV"])
    y = all_df["is_HBV"].astype(int)
    cases = all_df["Case"].astype(str)

    # Drop samples with >50% missing features and fill remaining with median
    keep_mask = X.isna().mean(axis=1) <= 0.5
    if keep_mask.sum() < len(X):
        print(f"Dropping {len(X)-keep_mask.sum()} samples with >50% missing features")
    X = X.loc[keep_mask].fillna(X.median()).reset_index(drop=True)
    y = y.loc[keep_mask].reset_index(drop=True)
    cases = cases.loc[keep_mask].reset_index(drop=True)

    # Standardize features for Logistic Regression
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Split into train, validation, test sets (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, stratify=y_train, random_state=SEED
    )
    print(f"Split into Train: {X_tr.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Compute balanced sample weights for training
    w_tr = compute_balanced_sample_weights(y_tr)

    # Train Logistic Regression
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=SEED
    )
    print("Training Logistic Regression...")
    clf.fit(X_tr, y_tr, sample_weight=w_tr)

    # Calibrate probabilities using sigmoid method if desired
    if CALIBRATE:
        print("Calibrating probabilities on validation set (sigmoid)…")
        calibrator = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
        calibrator.fit(X_val, y_val)
        model_for_pred = calibrator
        is_calibrated = True
    else:
        model_for_pred = clf
        is_calibrated = False

    # Determine best threshold from validation set
    val_proba = model_for_pred.predict_proba(X_val)[:, 1]
    best_thr, _ = choose_best_threshold(y_val, val_proba)
    test_proba = model_for_pred.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= best_thr).astype(int)

    # Compute metrics
    auc = float(roc_auc_score(y_test, test_proba))
    ap  = float(average_precision_score(y_test, test_proba))
    f1  = float(f1_score(y_test, test_pred))
    acc = float(accuracy_score(y_test, test_pred))
    metrics = {
        "ROC-AUC": auc,
        "PR-AUC": ap,
        "F1": f1,
        "Accuracy": acc,
        "BestThreshold(F1)": float(best_thr),
        "Calibrated": is_calibrated
    }
    print("Test metrics:", metrics)

    # Save metrics and confusion matrix
    pd.DataFrame([metrics]).to_csv(os.path.join(OUT_DIR, "metrics.csv"), index=False)
    cm = confusion_matrix(y_test, test_pred)
    pd.DataFrame(cm, index=["true_0","true_1"], columns=["pred_0","pred_1"]).to_csv(
        os.path.join(OUT_DIR, "confusion_matrix.csv")
    )

    # Save ROC and PR curves
    fpr, tpr, _ = roc_curve(y_test, test_proba)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "roc_curve.png"))
    plt.close()

    prec, rec, _ = precision_recall_curve(y_test, test_proba)
    plt.figure(figsize=(6,5))
    plt.plot(rec, prec, lw=2, label=f"PR-AUC={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (Test)")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pr_curve.png"))
    plt.close()

    # Save model
    joblib.dump(clf, os.path.join(OUT_DIR, "lr_model.joblib"))
    if CALIBRATE:
        joblib.dump(model_for_pred, os.path.join(OUT_DIR, "calibrated_model.joblib"))

    # Save processed features + labels for reference
    proc_df = pd.concat(
        [pd.Series(cases, name="Case"), X.add_prefix("feat_"), pd.Series(y.values, name="is_HBV")],
        axis=1
    )
    proc_path = os.path.join(OUT_DIR, "processed_features_and_labels.parquet")
    proc_df.to_parquet(proc_path)
    print(f"Saved processed matrix to {proc_path}")

    # Save top features based on absolute LR coefficients
    coef_importance = pd.DataFrame({
        "feature": X.columns,
        "abs_coeff": np.abs(clf.coef_[0])
    }).sort_values("abs_coeff", ascending=False)
    coef_importance.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"), index=False)
    coef_importance.head(TOPK).to_csv(os.path.join(OUT_DIR, f"top{TOPK}_features.csv"), index=False)

    # Save summary JSON
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump({
            "metrics": metrics,
            "class_counts_total": all_df["is_HBV"].value_counts().to_dict(),
            "class_counts_after_filters": y.value_counts().to_dict(),
            "top_features": coef_importance.head(TOPK).to_dict(orient="records")
        }, f, indent=2)

    print("All done. Outputs in:", os.path.abspath(OUT_DIR))


if __name__ == "__main__":
    main()
