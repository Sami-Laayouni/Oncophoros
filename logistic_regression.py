
"""
HBV vs Non-Viral classifier with Logistic Regression version
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
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

import matplotlib.pyplot as plt
import seaborn as sns
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

# ---------------- IMPORTANT FUNCTIONS (Just Copied and Pasted Between Files) ----------------

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

def clean_data_for_ml(X_train, X_val, X_test):
    """
    Comprehensive data cleaning to handle NaN values properly
    """
    print("Cleaning data and handling missing values...")
    
    # Step 1: Remove columns that are entirely NaN in training set
    all_nan_cols = X_train.columns[X_train.isna().all()].tolist()
    if all_nan_cols:
        print(f"Removing {len(all_nan_cols)} columns that are entirely NaN in training set")
        X_train = X_train.drop(columns=all_nan_cols)
        X_val = X_val.drop(columns=all_nan_cols)
        X_test = X_test.drop(columns=all_nan_cols)
    
    # Step 2: Remove columns with >95% missing values in training set
    high_missing_cols = X_train.columns[X_train.isna().mean() > 0.95].tolist()
    if high_missing_cols:
        print(f"Removing {len(high_missing_cols)} columns with >95% missing values")
        X_train = X_train.drop(columns=high_missing_cols)
        X_val = X_val.drop(columns=high_missing_cols)
        X_test = X_test.drop(columns=high_missing_cols)
    
    # Step 3: Fill remaining missing values with training median
    # For any remaining NaN medians, use 0 as fallback
    train_medians = X_train.median()
    train_medians = train_medians.fillna(0)  # Fill NaN medians with 0
    
    X_train = X_train.fillna(train_medians)
    X_val = X_val.fillna(train_medians)
    X_test = X_test.fillna(train_medians)
    
    # Step 4: Check for any remaining NaN values
    if X_train.isna().any().any():
        print("WARNING: Still have NaN values after cleaning. Filling with 0...")
        X_train = X_train.fillna(0)
        X_val = X_val.fillna(0)
        X_test = X_test.fillna(0)
    
    # Step 5: Handle infinite values that might arise from log transform
    print("Checking for infinite values...")
    for df_name, df in [("train", X_train), ("val", X_val), ("test", X_test)]:
        inf_mask = np.isinf(df.values)
        if inf_mask.any():
            print(f"Found {inf_mask.sum()} infinite values in {df_name} set. Replacing with 0.")
            df.replace([np.inf, -np.inf], 0, inplace=True)
    
    return X_train, X_val, X_test

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
    plt.title("ROC Curve (Test)")
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
    plt.title("Precision–Recall Curve (Test)")
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

    # Combine both datasets
    all_df = pd.concat([hbv_df, nonviral_df], axis=0, ignore_index=True)
    hbv_cases = set(hbv_df["Case"].tolist())
    all_df["is_HBV"] = all_df["Case"].apply(lambda x: 1 if x in hbv_cases else 0)

    # Convert features to numeric
    feature_cols = [c for c in all_df.columns if c not in ("Case", "is_HBV")]
    if not feature_cols:
        raise ValueError("There are no feature columns in the data.")
        
    X = to_numeric_df(all_df, skip_cols=["Case", "is_HBV"])
    y = all_df["is_HBV"].astype(int)
    cases = all_df["Case"].astype(str)

    # Drop samples with >50% missing features
    keep_mask = X.isna().mean(axis=1) <= 0.5
    if keep_mask.sum() < len(X):
        print(f"Dropping {len(X)-keep_mask.sum()} samples with >50% missing features")
    X = X.loc[keep_mask].reset_index(drop=True)
    y = y.loc[keep_mask].reset_index(drop=True)
    cases = cases.loc[keep_mask].reset_index(drop=True)
    all_df = all_df.loc[keep_mask].reset_index(drop=True)  # Update all_df for patient separation

    # Ensure patients are unique to train/val or test - prevent class leakage
    print("Splitting data while ensuring patient separation...")
    train_mask, val_mask, test_mask = ensure_patient_separation(all_df, TEST_SIZE, VAL_SIZE, SEED)

    # Split features and labels
    X_train = X.loc[train_mask].reset_index(drop=True)
    y_train = y.loc[train_mask].reset_index(drop=True)

    X_val = X.loc[val_mask].reset_index(drop=True)
    y_val = y.loc[val_mask].reset_index(drop=True)

    X_test = X.loc[test_mask].reset_index(drop=True)
    y_test = y.loc[test_mask].reset_index(drop=True)

    # Clean data and handle missing values properly
    X_train, X_val, X_test = clean_data_for_ml(X_train, X_val, X_test)

    # Apply log2(x+1) transform as in LightGBM version
    print("Applying log2(x+1) transform.")
    # Ensure no negative values before log transform
    X_train = X_train.clip(lower=0)
    X_val = X_val.clip(lower=0)
    X_test = X_test.clip(lower=0)
    
    X_train = np.log2(X_train + 1)
    X_val   = np.log2(X_val + 1)
    X_test  = np.log2(X_test + 1)

    # Final check for NaN/inf values after log transform
    for df_name, df in [("train", X_train), ("val", X_val), ("test", X_test)]:
        if df.isna().any().any():
            print(f"Found NaN values in {df_name} after log transform. Filling with 0.")
            df.fillna(0, inplace=True)
        inf_mask = np.isinf(df.values)
        if inf_mask.any():
            print(f"Found infinite values in {df_name} after log transform. Replacing with 0.")
            df.replace([np.inf, -np.inf], 0, inplace=True)

    # Standardize features for Logistic Regression
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_val_scaled   = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Final verification - no NaN values should remain
    print(f"Final data check - Train NaN: {X_train_scaled.isna().sum().sum()}")
    print(f"Final data check - Val NaN: {X_val_scaled.isna().sum().sum()}")  
    print(f"Final data check - Test NaN: {X_test_scaled.isna().sum().sum()}")

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

    print(f"Patient separation - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Verify no patient overlap
    train_patients = set(all_df.loc[train_mask, 'Case'].values)
    val_patients = set(all_df.loc[val_mask, 'Case'].values)
    test_patients = set(all_df.loc[test_mask, 'Case'].values)
    
    assert len(train_patients & test_patients) == 0, "Patient leakage detected: train/test overlap!"
    assert len(val_patients & test_patients) == 0, "Patient leakage detected: val/test overlap!"
    print("✓ No patient leakage detected")

    # Compute balanced sample weights for training
    w_tr = compute_balanced_sample_weights(y_train)

    # Train Logistic Regression
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=SEED
    )
    print("Training Logistic Regression...")
    clf.fit(X_train_scaled, y_train, sample_weight=w_tr)

    # Calibrate probabilities using sigmoid method if desired
    if CALIBRATE:
        print("Calibrating probabilities on validation set (sigmoid)…")
        calibrator = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
        calibrator.fit(X_val_scaled, y_val)
        model_for_pred = calibrator
        is_calibrated = True
    else:
        model_for_pred = clf
        is_calibrated = False

    # Determine best threshold from validation set
    val_proba = model_for_pred.predict_proba(X_val_scaled)[:, 1]
    best_thr, _ = threshold(y_val, val_proba)
    test_proba = model_for_pred.predict_proba(X_test_scaled)[:, 1]
    test_pred = (test_proba >= best_thr).astype(int)

    # Compute metrics
    auc = float(roc_auc_score(y_test, test_proba))
    ap  = float(average_precision_score(y_test, test_proba))
    f1  = float(f1_score(y_test, test_pred))
    acc = float(accuracy_score(y_test, test_pred))
    cm = confusion_matrix(y_test, test_pred)
    
    metrics = {
        "ROC-AUC": auc,
        "PR-AUC": ap,
        "F1": f1,
        "Accuracy": acc,
        "BestThreshold(F1)": float(best_thr),
        "Calibrated": is_calibrated,
        "Model": "LogisticRegression"
    }
    print("Test metrics:", metrics)

    # Save metrics and confusion matrix
    pd.DataFrame([metrics]).to_csv(os.path.join(OUT_DIR, "metrics.csv"), index=False)
    pd.DataFrame(cm, index=["true_0","true_1"], columns=["pred_0","pred_1"]).to_csv(
        os.path.join(OUT_DIR, "confusion_matrix.csv")
    )

    # Save models
    joblib.dump(clf, os.path.join(OUT_DIR, "lr_model.joblib"))
    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))  # Save scaler for inference
    if CALIBRATE:
        joblib.dump(model_for_pred, os.path.join(OUT_DIR, "calibrated_model.joblib"))

    # Save top features based on absolute LR coefficients
    coef_importance = pd.DataFrame({
        "feature": X_train.columns,  # Use the cleaned column names
        "abs_coeff": np.abs(clf.coef_[0])
    }).sort_values("abs_coeff", ascending=False)
    coef_importance.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"), index=False)
    coef_importance.head(TOPK).to_csv(os.path.join(OUT_DIR, f"top{TOPK}_features.csv"), index=False)

    # Create all plots
    print("Creating plots...")
    create_all_plots(y_test, test_proba, test_pred, OUT_DIR)

    # Save predictions
    test_cases = cases.loc[test_mask].reset_index(drop=True)
    out_pred = pd.DataFrame({
        "Case": test_cases.values,
        "y_true": y_test.values,
        "y_proba": test_proba,
        "y_pred": test_pred
    })
    out_pred.to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)

    # Save summary JSON
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump({
            "metrics": metrics,
            "class_counts_total": all_df["is_HBV"].value_counts().to_dict(),
            "class_counts_after_filters": y.value_counts().to_dict(),
            "top_features": coef_importance.head(TOPK).to_dict(orient="records"),
            "patient_splits": {
                "train_patients": len(train_patients),
                "val_patients": len(val_patients), 
                "test_patients": len(test_patients)
            }
        }, f, indent=2)


if __name__ == "__main__":
    main()
