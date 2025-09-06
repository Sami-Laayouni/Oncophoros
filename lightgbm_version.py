
"""
HBV vs Non-Viral classifier using the LightGBM model
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
from sklearn.calibration import CalibratedClassifierCV


import matplotlib.pyplot as plt
import joblib
import lightgbm as lgb

# ---------------- CONFIG VARIABLES ----------------
HBV_FILE = r"C:\Users\Admin\Desktop\IGEM Hackathon\data\HBV.tsv"   
NON_VIRAL_FILE = r"C:\Users\Admin\Desktop\IGEM Hackathon\data\Non-Viral.tsv"    
OUT_DIR  = "results_lightgbm"

TEST_SIZE = 0.25 # Portion of data to use as test set (set as a variable for easy modification)
VAL_SIZE  = 0.20 # Portion of training set to use as validation set (set as a variable for easy modification)
SEED = 42  # Random seed for reproducibility
TOPK = 20 # Number of top features to save based on importance
# ---------------------------------------------------

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

"""
Because we don't have the same number of samples for each class: 
    - HBV: 105 cases
    - Non-Viral: 149 cases
we will compute balanced sample weights to give more importance to the minority class during training.
"""

def compute_balanced_sample_weights(y):
    y = pd.Series(y)
    n = len(y)
    n_pos = (y == 1).sum() 
    n_neg = (y == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return np.ones(n, dtype=float)
    w_pos = 0.5 / n_pos # Weight for HBV class (0.006493506493506494 with current dataset)
    w_neg = 0.5 / n_neg # Weight for Non-Viral class (0.006944444444444444 with current dataset > w_pos because it has less samples)
    weights = np.where(y.values == 1, w_pos, w_neg).astype(float)
    return weights * (n / weights.sum()) # Normalization to keep the sum of weights equal to n

"""
Function to choose the best threshold based on F1 score from precision-recall curve.
Try all possible thresholds for turning probabilities into 0/1 predictions. 
Pick the one that makes the F1 score the highest
"""
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


def main():
    ensure_dir(OUT_DIR)

    print("Loading the data...")
    hbv_df = load_tsv(HBV_FILE)
    nonviral_df = load_tsv(NON_VIRAL_FILE)

    # Combine both files (we will label HBV=1 if in HBV file, 0 if Non-Viral) 
    all_df = pd.concat([hbv_df, nonviral_df], axis=0, ignore_index=True)
    hbv_cases = set(hbv_df["Case"].tolist())
    all_df["is_HBV"] = all_df["Case"].apply(lambda x: 1 if x in hbv_cases else 0)


    """
    After combing the data files, we have to preprocess the data before feeding it into the model.
    This includes handling missing values, dropping samples with too many missing features,
    applying log transformation if necessary, and optionally applying PCA for dimensionality reduction.
    """

    # Just to be sure we don't include non-numeric columns
    feature_cols = [c for c in all_df.columns if c not in ("Case", "is_HBV")]
    if not feature_cols:
        raise ValueError("There are no feature columns in the data.")
    

    # Convert features to numeric, ( mistakes get converted to NaN )
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

    # Impute remaining missing values with median of each feature (We are using median as supposed to mean in order to account for outliers)
    X = X.fillna(X.median())


    """
    The document suggested as step 3 to apply log2(x+1) transformation to normalize the data. The data from 
    TCGA‑LIHC already seems to be normalized, but we will still apply the log2(x+1) transformation as suggested.
    """

    print("Applying log2(x+1) transform.") 
    X = np.log2(X + 1)

    # Many papers use PCA for dimensionality reduction is not necessary for this dataset it, so we will skip it.


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


    """ 
    Now we will split the data into training, validation, and test sets. 
    We will use stratified splitting to ensure that the class distribution is maintained 
    in each set. 
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, stratify=y_train, random_state=SEED
    )
    # Final shapes (reference and debugging)
    print(f"Split into Train: {X_tr.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Compute balanced sample weights for training set
    w_tr = compute_balanced_sample_weights(y_tr)

    """
    This file is using the last method (LightGBM). The benefits to using this model include: 

        - Efficiency: LightGBM is designed to be efficient in both memory usage and training speed, making it suitable for large datasets.
        - Accuracy: LightGBM often provides high accuracy and can handle complex relationships in the data
    
    Credits: LightBGM is a free and open-source library that can be found at https://lightgbm.readthedocs.io/en/latest/    
    
    """

    # I changed the values several times in order to get the best results (however this is not guaranteed to be the optimal values)
    
    clf = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=2000,         # large number of trees, rely on early stopping
        learning_rate=0.02,        # slower learning rate but more stable learning
        num_leaves=48,             # complexity (smaller than 64 to reduce overfit)
        max_depth=12,              # cap depth to bound complexity
        min_child_samples=20,      # require >=20 samples in leaf -> regularizes
        min_split_gain=0.01,       # require small gain to split -> avoid tiny splits
        subsample=0.8,             # row sampling (bagging_fraction)
        subsample_freq=5,          # do bagging every 5 iterations
        colsample_bytree=0.7,      # feature sampling (feature_fraction)
        reg_alpha=0.1,             # L1 regularization to reduce overfitting
        reg_lambda=2.0,            # stronger L2 regularization
        max_bin=255,               # histogram bins (default/safe)
        verbose=-1,
        n_jobs=-1,
        random_state=SEED
    )

    """
    We will train the model using early stopping based on validation AUC to prevent overfitting. This tells the model to stop 
    training if the validation AUC does not improve for 200 consecutive rounds.
    """
    clf.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False)]
    )

    """
    We will calibrate the model's predicted probabilities using Platt scaling (sigmoid method) on the validation set.
    This step improves the reliability of the predicted probabilities.
    """
    print("Calibrating probabilities on validation set (sigmoid)…")
    calibrator = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
    calibrator.fit(X_val, y_val)
    model_for_pred = calibrator
    

    """ 
    We will evaluate the model on the test set using various metrics including ROC-AUC, PR-AUC, F1 score, Confusion Matrix 
    We will also determine the best threshold for classification based on F1 score from the validation set.
    """
    val_proba = model_for_pred.predict_proba(X_val)[:, 1]
    thr_f1, _ = threshold(y_val, val_proba)
    best_thr = float(thr_f1)

    test_proba = model_for_pred.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= best_thr).astype(int)


    # Calculate metrics
    auc = float(roc_auc_score(y_test, test_proba))
    ap  = float(average_precision_score(y_test, test_proba))
    f1  = float(f1_score(y_test, test_pred))
    acc = float(accuracy_score(y_test, test_pred))
    cm  = confusion_matrix(y_test, test_pred) # 2 * 2 matrix 

    metrics = {
        "ROC-AUC": auc,
        "PR-AUC": ap,
        "F1": f1,
        "Accuracy": acc,
        "BestThreshold(F1)": best_thr,
        "UsedEstimators": int(getattr(clf, "best_iteration_", clf.n_estimators))
    }
    print("Test metrics:", metrics)

    # Save outputs
    pd.DataFrame([metrics]).to_csv(os.path.join(OUT_DIR, "metrics.csv"), index=False)
    pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
        os.path.join(OUT_DIR, "confusion_matrix.csv")
    )

    joblib.dump(clf, os.path.join(OUT_DIR, "lgbm_model.joblib"))
    joblib.dump(model_for_pred, os.path.join(OUT_DIR, "calibrated_model.joblib"))

    try:
        booster = clf.booster_
        gain = booster.feature_importance(importance_type="gain")
        split = booster.feature_importance(importance_type="split")
        fi = pd.DataFrame({
            "feature": X.columns,
            "gain_importance": gain,
            "split_importance": split
        }).sort_values("gain_importance", ascending=False)
    except Exception:
        fi = pd.DataFrame({"feature": X.columns, "gain_importance": 0.0, "split_importance": 0.0})

    fi.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"), index=False)
    # Save only top K features
    fi.head(TOPK).to_csv(os.path.join(OUT_DIR, f"top{TOPK}_features.csv"), index=False)

    # ---- Plots ----
    fpr, tpr, _ = roc_curve(y_test, test_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "roc_curve.png"))
    plt.close()

    prec, rec, _ = precision_recall_curve(y_test, test_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, lw=2, label=f"PR-AUC={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (LightGBM Version)")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pr_curve.png"))
    plt.close()

    out_pred = pd.DataFrame({
        "Case": cases.loc[X_test.index].values,
        "y_true": y_test.values,
        "y_proba": test_proba,
        "y_pred": test_pred
    })
    out_pred.to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump({
            "metrics": metrics,
            "class_counts_total": all_df["is_HBV"].value_counts().to_dict(),
            "class_counts_after_filters": y.value_counts().to_dict(),
            "top_features": fi.head(TOPK).to_dict(orient="records")
        }, f, indent=2)

    print("All done. Outputs in:", os.path.abspath(OUT_DIR))

if __name__ == "__main__":
    main()
