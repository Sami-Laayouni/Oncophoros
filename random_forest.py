
"""
HBV vs Non-Viral classifier with Random Forest version
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ---------------- CONFIG VARIABLES ----------------
HBV_FILE = r"C:\Users\Admin\Desktop\IGEM Hackathon\data\HBV.tsv"   
NON_VIRAL_FILE = r"C:\Users\Admin\Desktop\IGEM Hackathon\data\Non-Viral.tsv"    
OUT_DIR  = "results_rf"

TEST_SIZE = 0.25 # Portion of data to use as test set
VAL_SIZE  = 0.20 # Portion of training set to use as validation set
SEED = 42  # Random seed for reproducibility
TOPK = 20 # Number of top features to save based on importance
CALIBRATE = True
# ---------------------------------------------------

# ---------------- IMPORTANT FUNCTIONS (Just Copied and Pasted Between Files) ----------------


# This function ensures the output directory exists or creates it if it doesn't.
def ensure_dir(d):
    if not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

# Load TSV file and ensure 'Case' column exists
def load_tsv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(path, sep="\t", header=0, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    if "Case" not in df.columns:
        raise ValueError(f"'Case' column not found in {path}. Columns include: {df.columns.tolist()[:10]}")
    df["Case"] = df["Case"].astype(str).str.strip()
    return df


def to_numeric_df(df, skip_cols):
    cols = [c for c in df.columns if c not in skip_cols]
    numeric = df[cols].apply(pd.to_numeric, errors="coerce")
    return numeric

# Balanced sample weights (same logic as LightGBM file)
def compute_balanced_sample_weights(y):
    y = pd.Series(y)
    n = len(y)
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return np.ones(n, dtype=float)
    w_pos = 0.5 / n_pos
    w_neg = 0.5 / n_neg
    weights = np.where(y.values == 1, w_pos, w_neg).astype(float)
    return weights * (n / weights.sum())

# Threshold selection (same simple F1 search used in LightGBM file)
def threshold(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    best_thr = 0.5
    best_score = -1.0

    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        score = f1_score(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_thr = thr
    return best_thr, best_score

def ensure_patient_separation(all_df, test_size, val_size, seed):
    """
    Ensure patients are unique to train/validation or test - never both.
    Prevents class leakage by splitting on patients, not individual samples.
    """
    # Get unique patients and their labels
    patient_labels = all_df.groupby('Case')['is_HBV'].first()
    unique_cases = patient_labels.index.tolist()
    unique_labels = patient_labels.values
    
    # Split patients (not samples) into train/test
    cases_train, cases_test, _, _ = train_test_split(
        unique_cases, unique_labels, 
        test_size=test_size, 
        stratify=unique_labels, 
        random_state=seed
    )
    
    # Further split training patients into train/val
    labels_train = [patient_labels[case] for case in cases_train]
    cases_tr, cases_val, _, _ = train_test_split(
        cases_train, labels_train,
        test_size=val_size,
        stratify=labels_train,
        random_state=seed
    )
    
    # Create masks for the original dataframe
    train_mask = all_df['Case'].isin(cases_tr)
    val_mask = all_df['Case'].isin(cases_val)
    test_mask = all_df['Case'].isin(cases_test)
    
    return train_mask, val_mask, test_mask

def plot_confusion_matrix(cm, out_dir):
    """Plot and save confusion matrix heatmap"""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Viral', 'HBV'], 
                yticklabels=['Non-Viral', 'HBV'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix_plot.png"))
    plt.close()

def plot_calibration_curve(y_true, y_proba, out_dir):
    """Plot calibration curve to assess probability calibration"""
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_proba, n_bins=10, strategy='uniform'
    )
    
    plt.figure(figsize=(6, 5))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", 
             label="Calibrated Model", color='blue', markersize=8)
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated", linewidth=2)
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "calibration_curve.png"))
    plt.close()

def create_all_plots(y_test, test_proba, test_pred, out_dir):
    """Create all required plots"""
    # ROC Curve
    auc = roc_auc_score(y_test, test_proba)
    fpr, tpr, _ = roc_curve(y_test, test_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Random Forest - Test)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curve.png"))
    plt.close()

    # Precision-Recall Curve
    ap = average_precision_score(y_test, test_proba)
    prec, rec, _ = precision_recall_curve(y_test, test_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, lw=2, label=f"PR-AUC={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (Random Forest - Test)")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pr_curve.png"))
    plt.close()

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, test_pred)
    plot_confusion_matrix(cm, out_dir)
    
    # Calibration Curve
    plot_calibration_curve(y_test, test_proba, out_dir)

def main():
    ensure_dir(OUT_DIR)

    print("Loading the data...")
    hbv_df = load_tsv(HBV_FILE)
    nonviral_df = load_tsv(NON_VIRAL_FILE)

    # Combine both files (we will label HBV=1 if in HBV file, 0 if Non-Viral)
    all_df = pd.concat([hbv_df, nonviral_df], axis=0, ignore_index=True)
    hbv_cases = set(hbv_df["Case"].tolist())
    all_df["is_HBV"] = all_df["Case"].apply(lambda x: 1 if x in hbv_cases else 0)

    # Preprocessing comments exactly as in the LightGBM file
    feature_cols = [c for c in all_df.columns if c not in ("Case", "is_HBV")]
    if not feature_cols:
        raise ValueError("There are no feature columns in the data.")

    # Convert features to numeric (mistakes -> NaN)
    X = to_numeric_df(all_df, skip_cols=["Case", "is_HBV"])
    y = all_df["is_HBV"].astype(int)
    cases = all_df["Case"].astype(str)

    # Drop samples with too many missing features
    row_nan_frac = X.isna().mean(axis=1)
    keep_mask = row_nan_frac <= 0.5
    if keep_mask.sum() < len(X):
        print(f"Dropping {len(X) - keep_mask.sum()} samples with >{0.5*100:.0f}% missing features")
    X = X.loc[keep_mask].reset_index(drop=True)
    y = y.loc[keep_mask].reset_index(drop=True)
    cases = cases.loc[keep_mask].reset_index(drop=True)
    all_df = all_df.loc[keep_mask].reset_index(drop=True)  # Update all_df too for patient separation

    # Impute remaining missing values with median of each feature
    X = X.fillna(X.median())

    # Apply log2(x+1) transform (kept identical to LightGBM file)
    print("Applying log2(x+1) transform.")
    X = np.log2(X + 1)

    # Save processed features and labels
    proc_df = pd.concat(
        [pd.Series(cases, name="Case"), X.add_prefix("feat_"), pd.Series(y.values, name="is_HBV")],
        axis=1
    )
    proc_path = os.path.join(OUT_DIR, "processed_features_and_labels.parquet")
    proc_df.to_parquet(proc_path)
    print(f"Saved processed matrix to {proc_path}")

    if y.nunique() < 2:
        raise ValueError("Need both classes present after processing.")

    # Ensure patients are unique to train/val or test - prevent class leakage
    print("Splitting data while ensuring patient separation...")
    train_mask, val_mask, test_mask = ensure_patient_separation(
        all_df, TEST_SIZE, VAL_SIZE, SEED
    )

    # Split data using patient separation to prevent class leakage
    X_train = X.loc[train_mask].reset_index(drop=True)
    y_train = y.loc[train_mask].reset_index(drop=True)
    
    X_val = X.loc[val_mask].reset_index(drop=True)
    y_val = y.loc[val_mask].reset_index(drop=True)
    
    X_test = X.loc[test_mask].reset_index(drop=True)
    y_test = y.loc[test_mask].reset_index(drop=True)
    
    print(f"Patient separation - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Verify no patient overlap
    train_patients = set(all_df.loc[train_mask, 'Case'].values)
    val_patients = set(all_df.loc[val_mask, 'Case'].values)
    test_patients = set(all_df.loc[test_mask, 'Case'].values)
    
    assert len(train_patients & test_patients) == 0, "Patient leakage detected: train/test overlap!"
    assert len(val_patients & test_patients) == 0, "Patient leakage detected: val/test overlap!"
    print("✓ No patient leakage detected")

    # Balanced sample weights
    w_tr = compute_balanced_sample_weights(y_train)

    # Random Forest classifier
    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1
    )

    print("Training Random Forest...")
    # fit supports sample_weight to handle imbalance
    clf.fit(X_train, y_train, sample_weight=w_tr)

    # Calibrate probabilities on validation set (sigmoid / Platt scaling)
    if CALIBRATE:
        print("Calibrating probabilities on validation set (sigmoid)…")
        calibrator = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
        calibrator.fit(X_val, y_val)
        model_for_pred = calibrator
        is_calibrated = True
    else:
        model_for_pred = clf
        is_calibrated = False

    val_proba = model_for_pred.predict_proba(X_val)[:, 1]
    thr_f1, _ = threshold(y_val, val_proba)
    best_thr = float(thr_f1)

    # Predict on test set
    test_proba = model_for_pred.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= best_thr).astype(int)

    # Metrics
    auc = float(roc_auc_score(y_test, test_proba))
    ap  = float(average_precision_score(y_test, test_proba))
    f1  = float(f1_score(y_test, test_pred))
    acc = float(accuracy_score(y_test, test_pred))
    cm  = confusion_matrix(y_test, test_pred)

    metrics = {
        "ROC-AUC": auc,
        "PR-AUC": ap,
        "F1": f1,
        "Accuracy": acc,
        "BestThreshold(F1)": best_thr,
        "Calibrated": is_calibrated,
        "Model": "RandomForest"
    }
    print("Test metrics:", metrics)

    # Save outputs
    pd.DataFrame([metrics]).to_csv(os.path.join(OUT_DIR, "metrics.csv"), index=False)
    pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
        os.path.join(OUT_DIR, "confusion_matrix.csv")
    )

    joblib.dump(clf, os.path.join(OUT_DIR, "rf_model.joblib"))
    if is_calibrated:
        joblib.dump(model_for_pred, os.path.join(OUT_DIR, "calibrated_rf_model.joblib"))

    # Feature importance for Random Forest: built-in importances
    fi = pd.DataFrame({
        'feature': X.columns,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    fi.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"), index=False)
    fi.head(TOPK).to_csv(os.path.join(OUT_DIR, f"top{TOPK}_features.csv"), index=False)

    # ---- Plots ----
    print("Creating plots...")
    create_all_plots(y_test, test_proba, test_pred, OUT_DIR)

    # Save per-sample predictions
    test_cases = cases.loc[test_mask].reset_index(drop=True)
    out_pred = pd.DataFrame({
        "Case": test_cases.values,
        "y_true": y_test.values,
        "y_proba": test_proba,
        "y_pred": test_pred
    })
    out_pred.to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)

    # Summary JSON
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump({
            "metrics": metrics,
            "class_counts_total": all_df["is_HBV"].value_counts().to_dict(),
            "class_counts_after_filters": y.value_counts().to_dict(),
            "top_features": fi.head(TOPK).to_dict(orient="records"),
            "patient_splits": {
                "train_patients": len(train_patients),
                "val_patients": len(val_patients), 
                "test_patients": len(test_patients)
            }
        }, f, indent=2)


if __name__ == "__main__":
    main()