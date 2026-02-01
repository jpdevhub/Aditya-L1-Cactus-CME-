#!/usr/bin/env python3
"""
Merge ML integrated dataset (positive CME samples) with 10K dataset (negative samples)
to create a balanced dataset for CME arrival prediction.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def normalize_column_names(df):
    """Normalize column names to handle variations"""
    # Create a mapping of common variations
    column_mapping = {}
    
    for col in df.columns:
        # Convert to lowercase and handle common variations
        normalized = col.lower().strip()
        
        # Handle time columns
        if 'time' in normalized and 'unix' not in normalized:
            if 'cme' in normalized or 'datetime' in normalized:
                column_mapping[col] = 'cme_time'
        elif 'unix' in normalized or 'epoch' in normalized:
            column_mapping[col] = 'cme_time_unix'
        
        # Handle CME properties
        elif normalized in ['velocity', 'cme_velocity', 'v']:
            column_mapping[col] = 'cme_velocity'
        elif normalized in ['width', 'cme_width', 'da']:
            column_mapping[col] = 'cme_width'
        elif normalized in ['pa', 'position_angle']:
            column_mapping[col] = 'pa'
        
        # Handle flags
        elif normalized in ['has_cme', 'cme_detected']:
            column_mapping[col] = 'has_cme'
        elif normalized in ['halo_class', 'halo']:
            column_mapping[col] = 'halo_class'
        elif normalized in ['earth_directed', 'earth_direction']:
            column_mapping[col] = 'earth_directed'
        
        # Handle transit time
        elif 'transit' in normalized and 'theoretical' in normalized:
            column_mapping[col] = 'theoretical_transit_hours'
        elif 'transit' in normalized and 'actual' in normalized:
            column_mapping[col] = 'actual_transit_hours'
    
    return column_mapping

def convert_to_unix_timestamp(datetime_str):
    """Convert datetime string to UNIX timestamp"""
    try:
        if pd.isna(datetime_str):
            return np.nan
        # Handle timezone aware datetime
        dt = pd.to_datetime(datetime_str)
        return int(dt.timestamp())
    except:
        return np.nan

def standardize_dataset(df, dataset_name):
    """Standardize a dataset to common format"""
    print(f"\nStandardizing {dataset_name}...")
    
    # Normalize column names
    column_mapping = normalize_column_names(df)
    df_renamed = df.rename(columns=column_mapping)
    
    print(f"Column mappings applied: {len(column_mapping)}")
    for old, new in column_mapping.items():
        print(f"  {old} -> {new}")
    
    # Ensure we have essential columns
    essential_cols = ['has_cme']
    
    # Add missing essential columns with default values
    for col in essential_cols:
        if col not in df_renamed.columns:
            print(f"Warning: {col} not found in {dataset_name}")
    
    # Handle time columns
    if 'cme_time' in df_renamed.columns and 'cme_time_unix' not in df_renamed.columns:
        print("Converting cme_time to UNIX timestamp...")
        df_renamed['cme_time_unix'] = df_renamed['cme_time'].apply(convert_to_unix_timestamp)
    
    return df_renamed

def extract_common_features(df, dataset_name):
    """Extract common features across both datasets"""
    
    # Define target feature groups
    target_features = {
        'time': ['cme_time', 'cme_time_unix'],
        'cme_properties': ['cme_velocity', 'cme_width', 'pa'],
        'labels': ['has_cme', 'halo_class', 'theoretical_transit_hours', 'actual_transit_hours', 'earth_directed'],
        'solar_wind': [
            'proton_density_mean', 'alpha_density_mean', 'proton_temperature_mean',
            'proton_bulk_speed_mean', 'alpha_proton_ratio_mean'
        ],
        'identifiers': ['cme_number']
    }
    
    # Find available features
    available_features = []
    missing_features = []
    
    for group, features in target_features.items():
        for feature in features:
            if feature in df.columns:
                available_features.append(feature)
            else:
                missing_features.append(feature)
    
    print(f"\n{dataset_name} - Available features: {len(available_features)}")
    print(f"{dataset_name} - Missing features: {len(missing_features)}")
    
    if missing_features:
        print("Missing features:")
        for feat in missing_features:
            print(f"  - {feat}")
    
    # Extract available features
    if available_features:
        extracted_df = df[available_features].copy()
    else:
        print(f"No common features found in {dataset_name}")
        return pd.DataFrame()
    
    # Fill missing target features with appropriate defaults
    for group, features in target_features.items():
        for feature in features:
            if feature not in extracted_df.columns:
                if feature == 'has_cme':
                    extracted_df[feature] = 0  # Default to no CME
                elif feature in ['halo_class', 'earth_directed']:
                    extracted_df[feature] = 0  # Default to no halo/not earth directed
                elif 'time' in feature:
                    continue  # Skip time features if missing
                else:
                    extracted_df[feature] = np.nan  # Default to NaN for others
    
    return extracted_df

def main():
    print("=== CME DATASET MERGER ===")
    print("Loading datasets...")
    
    # Load datasets
    try:
        ml_df = pd.read_csv('mag_ml_integrated_dataset.csv')
        print(f"ML integrated dataset loaded: {ml_df.shape}")
    except Exception as e:
        print(f"Error loading ML dataset: {e}")
        return
    
    try:
        full_df = pd.read_csv('FINAL_10K_REAL_WORLD_CME_DATASET.csv')
        print(f"10K real world dataset loaded: {full_df.shape}")
    except Exception as e:
        print(f"Error loading 10K dataset: {e}")
        return
    
    # Check has_cme distribution
    if 'has_cme' in ml_df.columns:
        ml_cme_dist = ml_df['has_cme'].value_counts()
        print(f"\nML dataset has_cme distribution:\n{ml_cme_dist}")
    
    if 'has_cme' in full_df.columns:
        full_cme_dist = full_df['has_cme'].value_counts()
        print(f"\n10K dataset has_cme distribution:\n{full_cme_dist}")
    
    # Standardize both datasets
    ml_std = standardize_dataset(ml_df, "ML Integrated Dataset")
    full_std = standardize_dataset(full_df, "10K Real World Dataset")
    
    # Extract common features
    ml_features = extract_common_features(ml_std, "ML Dataset")
    full_features = extract_common_features(full_std, "10K Dataset")
    
    if ml_features.empty or full_features.empty:
        print("Cannot proceed - one or both datasets have no extractable features")
        return
    
    # Find common columns
    ml_cols = set(ml_features.columns)
    full_cols = set(full_features.columns)
    common_cols = ml_cols.intersection(full_cols)
    
    print(f"\nCommon columns found: {len(common_cols)}")
    print("Common columns:", sorted(list(common_cols)))
    
    if not common_cols:
        print("No common columns found - cannot merge datasets")
        return
    
    # Filter to common columns
    ml_common = ml_features[list(common_cols)].copy()
    full_common = full_features[list(common_cols)].copy()
    
    # Filter negative samples from 10K dataset
    if 'has_cme' in full_common.columns:
        negative_samples = full_common[full_common['has_cme'] == 0].copy()
        print(f"\nNegative samples extracted: {len(negative_samples)}")
    else:
        print("Warning: has_cme not found, using all 10K samples")
        negative_samples = full_common.copy()
    
    # Combine datasets
    print("\nCombining datasets...")
    combined_df = pd.concat([ml_common, negative_samples], ignore_index=True)
    
    print(f"Combined dataset shape: {combined_df.shape}")
    
    # Check final has_cme distribution
    if 'has_cme' in combined_df.columns:
        final_dist = combined_df['has_cme'].value_counts()
        print(f"\nFinal has_cme distribution:\n{final_dist}")
    
    # Sort by time if available
    if 'cme_time_unix' in combined_df.columns:
        print("Sorting by timestamp...")
        combined_df = combined_df.sort_values('cme_time_unix').reset_index(drop=True)
    elif 'cme_time' in combined_df.columns:
        print("Sorting by datetime...")
        combined_df = combined_df.sort_values('cme_time').reset_index(drop=True)
    
    # Arrange columns in logical order
    column_order = []
    priority_order = [
        'cme_time', 'cme_time_unix', 'has_cme', 'cme_number',
        'cme_velocity', 'cme_width', 'pa', 'halo_class', 'earth_directed',
        'theoretical_transit_hours', 'actual_transit_hours',
        'proton_density_mean', 'alpha_density_mean', 'proton_temperature_mean',
        'proton_bulk_speed_mean', 'alpha_proton_ratio_mean'
    ]
    
    # Add priority columns first
    for col in priority_order:
        if col in combined_df.columns:
            column_order.append(col)
    
    # Add remaining columns
    for col in combined_df.columns:
        if col not in column_order:
            column_order.append(col)
    
    # Reorder columns
    combined_df = combined_df[column_order]
    
    # Clean data - remove rows with too many missing values
    print("\nCleaning data...")
    initial_rows = len(combined_df)
    
    # Remove rows where essential columns are missing
    essential_cols = ['has_cme']
    for col in essential_cols:
        if col in combined_df.columns:
            combined_df = combined_df.dropna(subset=[col])
    
    final_rows = len(combined_df)
    print(f"Rows removed due to missing essential data: {initial_rows - final_rows}")
    
    # Save the combined dataset
    output_file = 'balanced_cme_prediction_dataset.csv'
    combined_df.to_csv(output_file, index=False)
    print(f"\n✅ Combined dataset saved as: {output_file}")
    
    # Final summary
    print(f"\n🎯 FINAL SUMMARY:")
    print(f"   - Total rows: {len(combined_df)}")
    print(f"   - Total columns: {len(combined_df.columns)}")
    if 'has_cme' in combined_df.columns:
        pos_samples = (combined_df['has_cme'] == 1).sum()
        neg_samples = (combined_df['has_cme'] == 0).sum()
        print(f"   - Positive samples (has_cme=1): {pos_samples}")
        print(f"   - Negative samples (has_cme=0): {neg_samples}")
        if neg_samples > 0:
            ratio = pos_samples / neg_samples
            print(f"   - Positive/Negative ratio: {ratio:.3f}")
    
    print(f"\n📋 Column order:")
    for i, col in enumerate(combined_df.columns, 1):
        print(f"   {i:2d}. {col}")
    
    return combined_df

if __name__ == "__main__":
    result = main()
