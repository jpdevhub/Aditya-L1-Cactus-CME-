# ISRO CME Prediction - Dataset Information

## Dataset Files Overview

### Raw Datasets (`datasets/raw/`)
- `CACTUS_CME_Combined_Aug2024_Jun2025.csv` - Original CACTUS CME observations
- `FINAL_10K_REAL_WORLD_CME_DATASET.csv` - 10K balanced real-world dataset

### Processed Datasets (`datasets/processed/`)
- `balanced_cme_prediction_dataset_final.csv` - Final balanced training dataset
- `mag_ml_integrated_dataset.csv` - MAG-integrated ML dataset
- `ace_wind_cme_predictions.csv` - ACE/WIND validation predictions

### ML-Ready Datasets (`datasets/ml_ready/`)
- `cme_prediction_ml_dataset.csv` - Complete ML dataset with all features
- `cme_12hour_time_strip_features.csv` - 12-hour time window features
- `transit_time_prediction_dataset.csv` - CME arrival time prediction dataset
- `halo_classification_dataset.csv` - Halo CME classification dataset
- `cactus_cme_enhanced_with_halo_classification.csv` - Enhanced CACTUS data

## Feature Categories

### Solar Wind Features
- Proton density (mean, std, max, min, trend)
- Proton bulk speed (mean, std, max, min, trend)
- Proton temperature (mean, std, max, min, trend)
- Alpha particle density (mean, std, max, min, trend)
- Alpha-proton ratio (mean, std, max, min, trend)

### CME Geometry Features
- CME velocity (km/s)
- Angular width (degrees)
- Position angle (degrees)
- Earth-directed classification

### Time Features
- UNIX timestamp
- Hour of day
- Day of year
- Solar cycle phase

### Statistical Features
- Pre-CME 12h measurements
- Post-CME 12h measurements
- Delta (difference) features
- Trend analysis (slope calculations)

## Target Variables

### Halo CME Classification (1-4 Scale)
1. **Normal CME**: Angular width < 120°
2. **Partial Halo**: 120° ≤ angular width < 300°
3. **Full Halo**: Angular width ≥ 300°
4. **Complex Halo**: Multiple/complex structure

### Binary Classification
- `has_cme`: 0 (No CME) or 1 (CME detected)
- `earth_directed`: 0 (Not Earth-directed) or 1 (Earth-directed)

### Regression Targets
- `transit_time`: CME arrival time (hours)
- `cme_intensity`: Estimated impact intensity

## Data Quality Notes

### Dataset Sizes
- **CACTUS CME Observations**: 1,744 CME events (Aug 2024 - Jun 2025)
- **Large Real-World Dataset**: 10,000 samples (balanced negative samples)
- **Final Training Dataset**: 4,248 samples (balanced CME/non-CME)
- **ML-Ready Datasets**: 1,062 samples (feature-engineered)

### Data Processing
- Missing value handling: Forward fill and interpolation
- Outlier detection: IQR-based filtering applied
- Feature scaling: StandardScaler used for ML models
- Temporal validation: No data leakage across time boundaries

## Usage Instructions

```python
import pandas as pd

# Load main ML dataset
df = pd.read_csv('datasets/ml_ready/cme_prediction_ml_dataset.csv')

# Load specific task datasets
transit_df = pd.read_csv('datasets/ml_ready/transit_time_prediction_dataset.csv')
halo_df = pd.read_csv('datasets/ml_ready/halo_classification_dataset.csv')
```