#!/usr/bin/env python3
"""
Create final dataset with core feature groups:
- Time features (with UNIX timestamp)
- Solar Wind features  
- Statistical features (with delta values)
- MAG features (using existing Bx, By, Bz turbulence as components)
- CME Geometry features
- CME Class Labels
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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

def calculate_delta_features(df):
    """Calculate delta (difference) between pre and post CME measurements"""
    delta_features = {}
    
    # Solar wind delta features
    for param in ['proton_density', 'proton_bulk_speed', 'proton_temperature', 'alpha_density', 'alpha_proton_ratio']:
        pre_col = f'pre_cme_12h_{param}_mean'
        post_col = f'post_cme_12h_{param}_mean'
        
        if pre_col in df.columns and post_col in df.columns:
            delta_features[f'delta_{param}_mean'] = df[post_col] - df[pre_col]
    
    # MAG delta features
    for param in ['B_mag_gse', 'B_mag_gsm']:
        pre_col = f'pre_cme_12h_{param}_mean'
        post_col = f'post_cme_12h_{param}_mean'
        
        if pre_col in df.columns and post_col in df.columns:
            delta_features[f'delta_{param}_mean'] = df[post_col] - df[pre_col]
    
    return pd.DataFrame(delta_features)

def main():
    print("Loading dataset...")
    df = pd.read_csv('mag_ml_integrated_dataset.csv')
    
    print(f"Original dataset shape: {df.shape}")
    
    # Create final dataset with core feature groups
    final_dataset = pd.DataFrame()
    
    # ===== TIME FEATURES =====
    print("Processing Time features...")
    final_dataset['cme_time'] = df['cme_time']
    final_dataset['cme_time_unix'] = df['cme_time'].apply(convert_to_unix_timestamp)
    
    # ===== SOLAR WIND FEATURES =====
    print("Processing Solar Wind features...")
    # Main solar wind parameters
    solar_wind_cols = [
        'proton_density_mean', 'alpha_density_mean', 'proton_temperature_mean',
        'proton_bulk_speed_mean', 'alpha_proton_ratio_mean'
    ]
    for col in solar_wind_cols:
        if col in df.columns:
            final_dataset[col] = df[col]
    
    # ===== STATISTICAL FEATURES (12h Pre/Post + Delta) =====
    print("Processing Statistical features...")
    # Pre-CME 12h features
    pre_stats_cols = [
        'pre_cme_12h_proton_density_mean', 'pre_cme_12h_proton_bulk_speed_mean',
        'pre_cme_12h_proton_temperature_mean', 'pre_cme_12h_alpha_density_mean',
        'pre_cme_12h_alpha_proton_ratio_mean'
    ]
    for col in pre_stats_cols:
        if col in df.columns:
            final_dataset[col] = df[col]
    
    # Post-CME 12h features  
    post_stats_cols = [
        'post_cme_12h_proton_density_mean', 'post_cme_12h_proton_bulk_speed_mean',
        'post_cme_12h_proton_temperature_mean', 'post_cme_12h_alpha_density_mean',
        'post_cme_12h_alpha_proton_ratio_mean'
    ]
    for col in post_stats_cols:
        if col in df.columns:
            final_dataset[col] = df[col]
    
    # Calculate Delta features
    print("Calculating Delta features...")
    delta_df = calculate_delta_features(df)
    final_dataset = pd.concat([final_dataset, delta_df], axis=1)
    
    # ===== MAG FEATURES =====
    print("Processing MAG features...")
    # B_mag features
    mag_cols = [
        'pre_cme_12h_B_mag_gse_mean', 'pre_cme_12h_B_mag_gsm_mean',
        'post_cme_12h_B_mag_gse_mean', 'post_cme_12h_B_mag_gsm_mean'
    ]
    for col in mag_cols:
        if col in df.columns:
            final_dataset[col] = df[col]
    
    # Rolling STD features
    rolling_std_cols = [
        'pre_cme_12h_B_mag_rolling_std_6min', 'pre_cme_12h_B_mag_rolling_std_30min',
        'post_cme_12h_B_mag_rolling_std_6min', 'post_cme_12h_B_mag_rolling_std_30min'
    ]
    for col in rolling_std_cols:
        if col in df.columns:
            final_dataset[col] = df[col]
    
    # Variability Index features
    variability_cols = [
        'pre_cme_12h_B_mag_variability_cv', 'post_cme_12h_B_mag_variability_cv'
    ]
    for col in variability_cols:
        if col in df.columns:
            final_dataset[col] = df[col]
    
    # Turbulence CV features
    turbulence_cols = [
        'pre_cme_12h_B_mag_turbulence_cv', 'post_cme_12h_B_mag_turbulence_cv'
    ]
    for col in turbulence_cols:
        if col in df.columns:
            final_dataset[col] = df[col]
    
    # Bx, By, Bz components (using turbulence CV as proxy)
    bxyz_cols = [
        'pre_cme_12h_Bx_gse_turbulence_cv', 'pre_cme_12h_By_gse_turbulence_cv', 'pre_cme_12h_Bz_gse_turbulence_cv',
        'pre_cme_12h_Bx_gsm_turbulence_cv', 'pre_cme_12h_By_gsm_turbulence_cv', 'pre_cme_12h_Bz_gsm_turbulence_cv',
        'post_cme_12h_Bx_gse_turbulence_cv', 'post_cme_12h_By_gse_turbulence_cv', 'post_cme_12h_Bz_gse_turbulence_cv',
        'post_cme_12h_Bx_gsm_turbulence_cv', 'post_cme_12h_By_gsm_turbulence_cv', 'post_cme_12h_Bz_gsm_turbulence_cv'
    ]
    for col in bxyz_cols:
        if col in df.columns:
            # Rename for clarity
            new_name = col.replace('_turbulence_cv', '')
            final_dataset[new_name] = df[col]
    
    # ===== CME GEOMETRY FEATURES =====
    print("Processing CME Geometry features...")
    geometry_cols = ['cme_width', 'cme_velocity', 'pa']
    for col in geometry_cols:
        if col in df.columns:
            final_dataset[col] = df[col]
    
    # ===== CME CLASS LABELS =====
    print("Processing CME Class Labels...")
    label_cols = [
        'halo_class', 'theoretical_transit_hours', 'actual_transit_hours', 
        'has_cme', 'earth_directed'
    ]
    for col in label_cols:
        if col in df.columns:
            final_dataset[col] = df[col]
    
    # Add CME number for reference
    if 'cme_number' in df.columns:
        final_dataset['cme_number'] = df['cme_number']
    
    print(f"Final dataset shape: {final_dataset.shape}")
    print(f"Total features: {len(final_dataset.columns)}")
    
    # Save the final dataset
    output_file = 'final_cme_feature_dataset.csv'
    final_dataset.to_csv(output_file, index=False)
    print(f"Final dataset saved as: {output_file}")
    
    # Print feature summary
    print("\n=== FEATURE SUMMARY ===")
    print("🕒 Time Features:")
    time_features = [col for col in final_dataset.columns if 'time' in col.lower()]
    for feat in time_features:
        print(f"   - {feat}")
    
    print("\n⚙️ Solar Wind Features:")
    sw_features = [col for col in final_dataset.columns if any(param in col for param in ['proton_density_mean', 'alpha_density_mean', 'proton_temperature_mean', 'proton_bulk_speed_mean', 'alpha_proton_ratio_mean'])]
    for feat in sw_features:
        print(f"   - {feat}")
    
    print("\n📉 Statistical Features (12h Pre/Post/Delta):")
    stat_features = [col for col in final_dataset.columns if 'pre_cme_12h' in col or 'post_cme_12h' in col or 'delta_' in col]
    stat_features = [f for f in stat_features if 'B_mag' not in f and 'Bx' not in f and 'By' not in f and 'Bz' not in f]
    for feat in stat_features:
        print(f"   - {feat}")
    
    print("\n📈 MAG Features:")
    mag_features = [col for col in final_dataset.columns if 'B_mag' in col or 'Bx' in col or 'By' in col or 'Bz' in col]
    for feat in mag_features:
        print(f"   - {feat}")
    
    print("\n📐 CME Geometry:")
    geom_features = [col for col in final_dataset.columns if col in ['cme_width', 'cme_velocity', 'pa']]
    for feat in geom_features:
        print(f"   - {feat}")
    
    print("\n🎯 CME Class Labels:")
    label_features = [col for col in final_dataset.columns if col in ['halo_class', 'theoretical_transit_hours', 'actual_transit_hours', 'has_cme', 'earth_directed']]
    for feat in label_features:
        print(f"   - {feat}")
    
    return final_dataset

if __name__ == "__main__":
    final_df = main()
