#!/usr/bin/env python3
"""
ISRO CME Prediction — Dual Model Training Script (v2)

Trains THREE models from halo_classification_dataset.csv:
  1. Halo CME Class Classifier (1-4) with SMOTE balancing
  2. CME Velocity Regressor (km/s) — GradientBoostingRegressor
  3. Earth-Directed Binary Classifier (0/1) — RandomForestClassifier

All models saved with their scalers and feature names for safe inference.
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.impute import SimpleImputer

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    print("⚠️  imbalanced-learn not found. Installing...")
    import subprocess, sys
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "imbalanced-learn", "--break-system-packages", "-q"],
        check=True,
    )
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True

DATASET_PATH = "datasets/ml_ready/halo_classification_dataset.csv"
MODELS_DIR   = "models"

# ── Features used by all three models ────────────────────────────────────────
# These are the 57 numeric input columns from halo_classification_dataset.csv
# (excluding the 3 target columns: target_halo_class, cme_velocity, earth_directed)
FEATURE_COLS = [
    "cme_width", "pa",
    "pre_cme_12h_proton_density_mean", "pre_cme_12h_proton_density_std",
    "pre_cme_12h_proton_density_max", "pre_cme_12h_proton_density_min",
    "pre_cme_12h_proton_density_trend",
    "pre_cme_12h_proton_bulk_speed_mean", "pre_cme_12h_proton_bulk_speed_std",
    "pre_cme_12h_proton_bulk_speed_max", "pre_cme_12h_proton_bulk_speed_min",
    "pre_cme_12h_proton_bulk_speed_trend",
    "pre_cme_12h_proton_temperature_mean", "pre_cme_12h_proton_temperature_std",
    "pre_cme_12h_proton_temperature_max", "pre_cme_12h_proton_temperature_min",
    "pre_cme_12h_proton_temperature_trend",
    "pre_cme_12h_alpha_density_mean", "pre_cme_12h_alpha_density_std",
    "pre_cme_12h_alpha_density_max", "pre_cme_12h_alpha_density_min",
    "pre_cme_12h_alpha_density_trend",
    "pre_cme_12h_alpha_proton_ratio_mean", "pre_cme_12h_alpha_proton_ratio_std",
    "pre_cme_12h_alpha_proton_ratio_max", "pre_cme_12h_alpha_proton_ratio_min",
    "pre_cme_12h_alpha_proton_ratio_trend",
    "post_cme_12h_proton_density_mean", "post_cme_12h_proton_density_std",
    "post_cme_12h_proton_density_max", "post_cme_12h_proton_density_min",
    "post_cme_12h_proton_density_trend",
    "post_cme_12h_proton_bulk_speed_mean", "post_cme_12h_proton_bulk_speed_std",
    "post_cme_12h_proton_bulk_speed_max", "post_cme_12h_proton_bulk_speed_min",
    "post_cme_12h_proton_bulk_speed_trend",
    "post_cme_12h_proton_temperature_mean", "post_cme_12h_proton_temperature_std",
    "post_cme_12h_proton_temperature_max", "post_cme_12h_proton_temperature_min",
    "post_cme_12h_proton_temperature_trend",
    "post_cme_12h_alpha_density_mean", "post_cme_12h_alpha_density_std",
    "post_cme_12h_alpha_density_max", "post_cme_12h_alpha_density_min",
    "post_cme_12h_alpha_density_trend",
    "post_cme_12h_alpha_proton_ratio_mean", "post_cme_12h_alpha_proton_ratio_std",
    "post_cme_12h_alpha_proton_ratio_max", "post_cme_12h_alpha_proton_ratio_min",
    "post_cme_12h_alpha_proton_ratio_trend",
    "cme_kinetic_energy", "cme_momentum",
    "cme_hour", "cme_day_of_year", "cme_month",
]


def load_data():
    print("📊 Loading dataset:", DATASET_PATH)
    df = pd.read_csv(DATASET_PATH)
    print(f"   Shape: {df.shape}")

    # Impute any NaNs
    imputer = SimpleImputer(strategy="median")
    df[FEATURE_COLS] = imputer.fit_transform(df[FEATURE_COLS])

    print("   Missing values after imputation:", df[FEATURE_COLS].isnull().sum().sum())
    return df


def split_and_scale(X, y_cls):
    """Split into train/test and scale X; returns split datasets + scaler."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)
    return X_tr_s, X_te_s, y_tr, y_te, scaler


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1 — Halo CME Class Classifier (1–4) with SMOTE
# ─────────────────────────────────────────────────────────────────────────────
def train_halo_classifier(df):
    print("\n" + "=" * 55)
    print("🏷️  MODEL 1 — Halo CME Class Classifier (1–4)")
    print("=" * 55)

    X = df[FEATURE_COLS].values
    y = df["target_halo_class"].values

    print("   Class distribution BEFORE SMOTE:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"     Class {cls}: {cnt} samples")

    # Split first so we don't leak test data into SMOTE
    X_tr, X_te, y_tr, y_te, scaler = split_and_scale(X, y)

    # Apply SMOTE only on training data
    # k_neighbors must be < minority class count; Class 3 has ~4 training samples
    minority_count = min(np.bincount(y_tr - 1)[1:])  # exclude class 1 majority
    k = max(1, min(3, minority_count - 1))
    sm = SMOTE(random_state=42, k_neighbors=k)
    X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)

    print("\n   Class distribution AFTER SMOTE (training only):")
    unique_r, counts_r = np.unique(y_tr_res, return_counts=True)
    for cls, cnt in zip(unique_r, counts_r):
        print(f"     Class {cls}: {cnt} samples")

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.08,
        max_depth=5,
        subsample=0.8,
        min_samples_leaf=3,
        random_state=42,
    )
    print("\n   Training GradientBoostingClassifier...")
    model.fit(X_tr_res, y_tr_res)

    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    print(f"\n   Test Accuracy : {acc:.4f}")
    print("\n   Classification Report:")
    print(classification_report(y_te, y_pred, zero_division=0))
    print("   Confusion Matrix:")
    print(confusion_matrix(y_te, y_pred))

    model_pkg = {
        "model": model,
        "scaler": scaler,
        "feature_names": FEATURE_COLS,
        "classes": model.classes_.tolist(),
        "test_accuracy": acc,
    }
    path = f"{MODELS_DIR}/halo_class_model.pkl"
    joblib.dump(model_pkg, path)
    print(f"\n   ✅ Saved → {path}")
    return model_pkg


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2a — CME Velocity Regressor
# ─────────────────────────────────────────────────────────────────────────────
def train_velocity_regressor(df):
    print("\n" + "=" * 55)
    print("🚀  MODEL 2a — CME Velocity Regressor (km/s)")
    print("=" * 55)

    X = df[FEATURE_COLS].values
    y = df["cme_velocity"].values.astype(float)

    print(f"   Velocity range: {y.min():.0f} – {y.max():.0f} km/s  |  mean: {y.mean():.0f} km/s")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.08,
        max_depth=5,
        subsample=0.8,
        min_samples_leaf=3,
        random_state=42,
    )
    print("   Training GradientBoostingRegressor...")
    model.fit(X_tr_s, y_tr)

    y_pred = model.predict(X_te_s)
    mae = mean_absolute_error(y_te, y_pred)
    r2  = r2_score(y_te, y_pred)
    print(f"\n   MAE : {mae:.1f} km/s")
    print(f"   R²  : {r2:.4f}")

    # Show a few sample predictions
    print("\n   Sample predictions vs actual:")
    for actual, pred in zip(y_te[:6], y_pred[:6]):
        print(f"     Actual: {actual:.0f}  Predicted: {pred:.0f}")

    model_pkg = {
        "model": model,
        "scaler": scaler,
        "feature_names": FEATURE_COLS,
        "velocity_min": float(y.min()),
        "velocity_max": float(y.max()),
        "mae_kms": float(mae),
        "r2": float(r2),
    }
    path = f"{MODELS_DIR}/velocity_regressor.pkl"
    joblib.dump(model_pkg, path)
    print(f"\n   ✅ Saved → {path}")
    return model_pkg


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2b — Earth-Directed Binary Classifier
# ─────────────────────────────────────────────────────────────────────────────
def train_earth_directed_classifier(df):
    print("\n" + "=" * 55)
    print("🌍  MODEL 2b — Earth-Directed Binary Classifier (0/1)")
    print("=" * 55)

    X = df[FEATURE_COLS].values
    y = df["earth_directed"].values.astype(int)

    print("   Class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    labels = ["Not Earth-Directed", "Earth-Directed"]
    for cls, cnt in zip(unique, counts):
        print(f"     {labels[cls]} (class {cls}): {cnt} samples")

    X_tr, X_te, y_tr, y_te, scaler = split_and_scale(X, y)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    print("\n   Training RandomForestClassifier...")
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    print(f"\n   Test Accuracy : {acc:.4f}")
    print("\n   Classification Report:")
    print(classification_report(y_te, y_pred, target_names=labels, zero_division=0))

    model_pkg = {
        "model": model,
        "scaler": scaler,
        "feature_names": FEATURE_COLS,
        "classes": model.classes_.tolist(),
        "test_accuracy": acc,
    }
    path = f"{MODELS_DIR}/earth_directed_model.pkl"
    joblib.dump(model_pkg, path)
    print(f"\n   ✅ Saved → {path}")
    return model_pkg


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("🚀 ISRO CME Prediction — Dual Model Training (v2)")
    print("=" * 55)

    df = load_data()

    halo_pkg   = train_halo_classifier(df)
    vel_pkg    = train_velocity_regressor(df)
    earth_pkg  = train_earth_directed_classifier(df)

    print("\n" + "=" * 55)
    print("🎉 All models trained and saved!")
    print(f"   Model 1 — Halo Class    accuracy : {halo_pkg['test_accuracy']:.2%}")
    print(f"   Model 2a — Velocity      MAE      : {vel_pkg['mae_kms']:.1f} km/s  |  R²: {vel_pkg['r2']:.4f}")
    print(f"   Model 2b — Earth-Direct  accuracy : {earth_pkg['test_accuracy']:.2%}")
    print("\n✅ Update the Streamlit app and restart to use these models.")


if __name__ == "__main__":
    main()
