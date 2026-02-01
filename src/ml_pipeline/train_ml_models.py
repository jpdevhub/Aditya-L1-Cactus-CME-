#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Setting up ML models for CME prediction...")

model_dir = '/Volumes/T7/ISRO(HALO_CME)/ml_models'
os.makedirs(model_dir, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv('/Volumes/T7/ISRO(HALO_CME)/finalized_cme_dataset_Aug2024_Jun2025.csv')

df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])

print(f"Dataset loaded: {len(df)} records")
print("\nClass distribution:")
print(df['halo_class'].value_counts())

print("\nCME vs non-CME:")
print(df['has_cme'].value_counts())

print("\nChecking for null values in features...")
null_counts = df.isnull().sum()
print(null_counts[null_counts > 0])

# =============================================================================
# DATA PREPROCESSING
# =============================================================================
print("\nPreprocessing data...")

df['hour'] = df['datetime'].dt.hour
df['day_of_year'] = df['datetime'].dt.dayofyear
df['month'] = df['datetime'].dt.month

for col in ['cme_velocity', 'cme_width', 'pa']:
    median_val = df.loc[df['has_cme'] == 1, col].median()
    df[col] = df[col].fillna(df['has_cme'].map({1: median_val, 0: 0}))

feature_cols_binary = [
    'proton_density_mean', 'proton_temperature_mean', 'proton_bulk_speed_mean',
    'he_h_ratio', 'temp_ratio', 'delta_velocity', 'delta_density',
    'delta_he_h_ratio', 'delta_temp_ratio', 'hour', 'day_of_year', 'month'
]

feature_cols_halo = [
    'proton_density_mean', 'proton_temperature_mean', 'proton_bulk_speed_mean',
    'he_h_ratio', 'temp_ratio', 'delta_velocity', 'delta_density',
    'delta_he_h_ratio', 'delta_temp_ratio', 'hour', 'day_of_year', 'month',
    'cme_velocity', 'cme_width', 'pa'  # Adding CME-specific features
]

feature_cols_width = [
    'proton_density_mean', 'proton_temperature_mean', 'proton_bulk_speed_mean',
    'he_h_ratio', 'temp_ratio', 'delta_velocity', 'delta_density',
    'delta_he_h_ratio', 'delta_temp_ratio', 'hour', 'day_of_year', 'month',
    'cme_velocity'  # Using CME velocity to help predict width
]

print("\nChecking for missing values in features...")
for col_set, name in [(feature_cols_binary, "Binary classification"), 
                       (feature_cols_halo, "Halo classification"),
                       (feature_cols_width, "Width prediction")]:
    missing = df[col_set].isnull().sum().sum()
    print(f"{name} features - Missing values: {missing}")

all_features = list(set(feature_cols_binary + feature_cols_halo + feature_cols_width))
null_counts = df[all_features].isnull().sum()
print("\nColumns with missing values:")
print(null_counts[null_counts > 0])

# =============================================================================
# MODEL 1: BINARY CLASSIFICATION (HAS_CME PREDICTION)
# =============================================================================
print("\n" + "="*80)
print("TRAINING BINARY CLASSIFICATION MODEL (HAS_CME PREDICTION)")
print("="*80)

# Prepare data for binary classification
X_binary = df[feature_cols_binary]
y_binary = df['has_cme']

# Split data
X_train_binary, X_test_binary, y_train_binary, y_test_binary = train_test_split(
    X_binary, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

print(f"Training set: {X_train_binary.shape[0]} samples")
print(f"Test set: {X_test_binary.shape[0]} samples")

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, feature_cols_binary)
    ])

binary_clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

print("Training binary classification model...")
binary_clf.fit(X_train_binary, y_train_binary)

# Evaluate binary model
y_pred_binary = binary_clf.predict(X_test_binary)
binary_accuracy = accuracy_score(y_test_binary, y_pred_binary)
print(f"\nBinary Classification Accuracy: {binary_accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test_binary, y_pred_binary))

print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test_binary, y_pred_binary)
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No CME', 'CME'],
            yticklabels=['No CME', 'CME'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Binary CME Classification')
plt.tight_layout()
plt.savefig(f'{model_dir}/binary_confusion_matrix.png', dpi=300)

# Feature importance for binary model
binary_importances = binary_clf.named_steps['classifier'].feature_importances_
binary_feat_imp = pd.Series(
    binary_importances, 
    index=feature_cols_binary
).sort_values(ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x=binary_feat_imp.values, y=binary_feat_imp.index)
plt.title('Feature Importance - Binary CME Classification')
plt.tight_layout()
plt.savefig(f'{model_dir}/binary_feature_importance.png', dpi=300)

# Save binary model
joblib.dump(binary_clf, f'{model_dir}/binary_cme_classifier.pkl')
print(f"Binary classification model saved to {model_dir}/binary_cme_classifier.pkl")

# =============================================================================
# MODEL 2: MULTI-CLASS CLASSIFICATION (HALO CLASS PREDICTION)
# =============================================================================
print("\n" + "="*80)
print("TRAINING MULTI-CLASS CLASSIFICATION MODEL (HALO CLASS PREDICTION)")
print("="*80)

# Filter only CME events for halo class prediction
# For halo class prediction, we only use actual CME events (where we know the class)
cme_df = df[df['has_cme'] == 1].copy()

X_halo = cme_df[feature_cols_halo]
y_halo = cme_df['halo_class']

# Split data for halo classification
X_train_halo, X_test_halo, y_train_halo, y_test_halo = train_test_split(
    X_halo, y_halo, test_size=0.2, random_state=42, stratify=y_halo
)

print(f"Training set: {X_train_halo.shape[0]} samples")
print(f"Test set: {X_test_halo.shape[0]} samples")
print(f"Class distribution in training set:\n{y_train_halo.value_counts()}")

# Create preprocessing pipeline for halo classification
halo_preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, feature_cols_halo)
    ])

# Create and train the halo classification model
halo_clf = Pipeline(steps=[
    ('preprocessor', halo_preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, 
                                         class_weight='balanced', 
                                         random_state=42))
])

print("Training halo class prediction model...")
halo_clf.fit(X_train_halo, y_train_halo)

# Evaluate halo classification model
y_pred_halo = halo_clf.predict(X_test_halo)
halo_accuracy = accuracy_score(y_test_halo, y_pred_halo)
print(f"\nHalo Classification Accuracy: {halo_accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test_halo, y_pred_halo))

print("\nConfusion Matrix:")
conf_matrix_halo = confusion_matrix(y_test_halo, y_pred_halo)
print(conf_matrix_halo)

# Plot confusion matrix for halo classification
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_halo, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Halo Class')
plt.ylabel('Actual Halo Class')
plt.title('Confusion Matrix - Halo Class Classification')
plt.tight_layout()
plt.savefig(f'{model_dir}/halo_confusion_matrix.png', dpi=300)

# Feature importance for halo classification
halo_importances = halo_clf.named_steps['classifier'].feature_importances_
halo_feat_imp = pd.Series(
    halo_importances, 
    index=feature_cols_halo
).sort_values(ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x=halo_feat_imp.values, y=halo_feat_imp.index)
plt.title('Feature Importance - Halo Class Classification')
plt.tight_layout()
plt.savefig(f'{model_dir}/halo_feature_importance.png', dpi=300)

# Save halo model
joblib.dump(halo_clf, f'{model_dir}/halo_class_classifier.pkl')
print(f"Halo classification model saved to {model_dir}/halo_class_classifier.pkl")

# =============================================================================
# MODEL 3: REGRESSION (CME WIDTH PREDICTION)
# =============================================================================
print("\n" + "="*80)
print("TRAINING REGRESSION MODEL (CME WIDTH PREDICTION)")
print("="*80)

# For width prediction, we also only use CME events
X_width = cme_df[feature_cols_width]
y_width = cme_df['cme_width']

# Split data for width regression
X_train_width, X_test_width, y_train_width, y_test_width = train_test_split(
    X_width, y_width, test_size=0.2, random_state=42
)

print(f"Training set: {X_train_width.shape[0]} samples")
print(f"Test set: {X_test_width.shape[0]} samples")

# Create preprocessing pipeline for width regression
width_preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, feature_cols_width)
    ])

# Create and train the width regression model
width_reg = Pipeline(steps=[
    ('preprocessor', width_preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

print("Training CME width prediction model...")
width_reg.fit(X_train_width, y_train_width)

# Evaluate width regression model
y_pred_width = width_reg.predict(X_test_width)
rmse = np.sqrt(mean_squared_error(y_test_width, y_pred_width))
mae = mean_absolute_error(y_test_width, y_pred_width)
r2 = r2_score(y_test_width, y_pred_width)

print(f"\nWidth Prediction Results:")
print(f"Root Mean Squared Error: {rmse:.2f} degrees")
print(f"Mean Absolute Error: {mae:.2f} degrees")
print(f"R² Score: {r2:.4f}")

# Plot actual vs. predicted width
plt.figure(figsize=(10, 6))
plt.scatter(y_test_width, y_pred_width, alpha=0.5)
plt.plot([0, 360], [0, 360], 'r--')  # Perfect prediction line
plt.xlabel('Actual CME Width (degrees)')
plt.ylabel('Predicted CME Width (degrees)')
plt.title('CME Width: Actual vs. Predicted')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{model_dir}/width_prediction_scatter.png', dpi=300)

# Feature importance for width prediction
width_importances = width_reg.named_steps['regressor'].feature_importances_
width_feat_imp = pd.Series(
    width_importances, 
    index=feature_cols_width
).sort_values(ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x=width_feat_imp.values, y=width_feat_imp.index)
plt.title('Feature Importance - CME Width Prediction')
plt.tight_layout()
plt.savefig(f'{model_dir}/width_feature_importance.png', dpi=300)

# Save width model
joblib.dump(width_reg, f'{model_dir}/cme_width_regressor.pkl')
print(f"Width regression model saved to {model_dir}/cme_width_regressor.pkl")

print("\nAll models trained and saved successfully!")

# =============================================================================
# MODEL SUMMARY
# =============================================================================
print("\n" + "="*80)
print("MODEL SUMMARY")
print("="*80)

print("\n1. Binary CME Prediction Model:")
print(f"   - Accuracy: {binary_accuracy:.4f}")
print(f"   - Top features: {', '.join(binary_feat_imp.index[:3])}")

print("\n2. Halo Class Prediction Model:")
print(f"   - Accuracy: {halo_accuracy:.4f}")
print(f"   - Top features: {', '.join(halo_feat_imp.index[:3])}")

print("\n3. CME Width Prediction Model:")
print(f"   - RMSE: {rmse:.2f} degrees")
print(f"   - R²: {r2:.4f}")
print(f"   - Top features: {', '.join(width_feat_imp.index[:3])}")

# Plot all feature importances in one figure
plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
sns.barplot(x=binary_feat_imp.values[:10], y=binary_feat_imp.index[:10])
plt.title('Top 10 Features - Binary CME Classification')

plt.subplot(3, 1, 2)
sns.barplot(x=halo_feat_imp.values[:10], y=halo_feat_imp.index[:10])
plt.title('Top 10 Features - Halo Class Classification')

plt.subplot(3, 1, 3)
sns.barplot(x=width_feat_imp.values[:10], y=width_feat_imp.index[:10])
plt.title('Top 10 Features - CME Width Prediction')

plt.tight_layout()
plt.savefig(f'{model_dir}/all_feature_importance.png', dpi=300)
print(f"\nFeature importance comparison plot saved to {model_dir}/all_feature_importance.png")
