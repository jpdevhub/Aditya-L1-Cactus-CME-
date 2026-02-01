#!/usr/bin/env python3
"""
Clean up the balanced dataset by removing duplicate columns and ensuring proper structure
"""

import pandas as pd
import numpy as np

def main():
    print("Loading balanced dataset...")
    df = pd.read_csv('balanced_cme_prediction_dataset.csv')
    
    print(f"Original shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Remove duplicate columns
    df_clean = df.loc[:, ~df.columns.duplicated()]
    
    print(f"After removing duplicate columns: {df_clean.shape}")
    print(f"Cleaned columns: {list(df_clean.columns)}")
    
    # Define proper column order
    desired_order = [
        'cme_time',
        'cme_time_unix', 
        'has_cme',
        'cme_number',
        'cme_velocity',
        'cme_width',
        'pa',
        'halo_class',
        'earth_directed',
        'theoretical_transit_hours',
        'actual_transit_hours',
        'proton_density_mean',
        'alpha_density_mean', 
        'proton_temperature_mean',
        'proton_bulk_speed_mean',
        'alpha_proton_ratio_mean'
    ]
    
    # Reorder columns
    available_cols = [col for col in desired_order if col in df_clean.columns]
    remaining_cols = [col for col in df_clean.columns if col not in available_cols]
    
    final_order = available_cols + remaining_cols
    df_final = df_clean[final_order]
    
    print(f"Final shape: {df_final.shape}")
    
    # Check has_cme distribution
    if 'has_cme' in df_final.columns:
        dist = df_final['has_cme'].value_counts().sort_index()
        print(f"\nhas_cme distribution:")
        print(dist)
        
        pos_samples = (df_final['has_cme'] == 1).sum()
        neg_samples = (df_final['has_cme'] == 0).sum()
        ratio = pos_samples / neg_samples if neg_samples > 0 else 0
        print(f"\nPositive/Negative ratio: {ratio:.3f}")
    
    # Save cleaned dataset
    output_file = 'balanced_cme_prediction_dataset_clean.csv'
    df_final.to_csv(output_file, index=False)
    print(f"\n✅ Cleaned dataset saved as: {output_file}")
    
    # Show final structure
    print(f"\n📋 Final columns ({len(df_final.columns)}):")
    for i, col in enumerate(df_final.columns, 1):
        print(f"   {i:2d}. {col}")
    
    # Show sample data
    print(f"\n📊 Sample data (first 3 rows):")
    print(df_final.head(3).to_string())
    
    return df_final

if __name__ == "__main__":
    result = main()
