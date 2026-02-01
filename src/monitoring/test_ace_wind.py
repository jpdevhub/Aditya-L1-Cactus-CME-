#!/usr/bin/env python3
"""
Test the trained CME model on ACE/WIND data
"""

import pandas as pd
import pickle
import numpy as np

def test_ace_wind_data():
    print("🔬 TESTING CME MODEL ON ACE/WIND DATA")
    print("=" * 50)
    
    # Load the trained model
    print("Loading trained model...")
    with open('best_cme_model.pkl', 'rb') as f:
        model_package = pickle.load(f)
    
    model = model_package['model']
    scaler = model_package['scaler']
    feature_names = model_package['feature_names']
    model_name = model_package['model_name']
    
    print(f"Loaded model: {model_name}")
    print(f"Features: {len(feature_names)}")
    
    # Load ACE/WIND sample data
    print(f"\nLoading ACE/WIND data...")
    ace_data = pd.read_csv('sample_ace_wind_data.csv')
    print(f"ACE/WIND data shape: {ace_data.shape}")
    print(f"ACE/WIND columns: {list(ace_data.columns)}")
    
    # Map ACE/WIND features to model format
    print(f"\nMapping features...")
    feature_mapping = {
        'time_unix': 'cme_time_unix',
        'proton_density': 'proton_density_mean',
        'alpha_density': 'alpha_density_mean', 
        'proton_temperature': 'proton_temperature_mean',
        'bulk_speed': 'proton_bulk_speed_mean',
        'alpha_proton_ratio': 'alpha_proton_ratio_mean'
    }
    
    # Create feature matrix
    ace_features = pd.DataFrame()
    
    for ace_col, model_col in feature_mapping.items():
        if ace_col in ace_data.columns and model_col in feature_names:
            ace_features[model_col] = ace_data[ace_col]
            print(f"  {ace_col} -> {model_col}")
    
    # Fill missing features with defaults (no CME properties in real-time data)
    for feature in feature_names:
        if feature not in ace_features.columns:
            if any(cme_prop in feature for cme_prop in ['cme_velocity', 'cme_width', 'pa', 'theoretical_transit']):
                ace_features[feature] = -999  # No CME detected yet
            elif feature in ['halo_class', 'earth_directed']:
                ace_features[feature] = 0  # Default values
            else:
                ace_features[feature] = ace_features.iloc[:, 0].median() if len(ace_features.columns) > 0 else 0
    
    # Ensure correct column order
    ace_features = ace_features[feature_names]
    
    print(f"Prepared features shape: {ace_features.shape}")
    
    # Make predictions
    print(f"\nMaking predictions...")
    
    # Use unscaled features for RandomForest (tree-based model)
    if model_name in ['RandomForest', 'GradientBoosting']:
        predictions_proba = model.predict_proba(ace_features)[:, 1]
    else:
        # Scale features for linear models
        ace_features_scaled = scaler.transform(ace_features)
        predictions_proba = model.predict_proba(ace_features_scaled)[:, 1]
    
    predictions_binary = (predictions_proba > 0.5).astype(int)
    
    # Add predictions to original data
    results = ace_data.copy()
    results['cme_arrival_probability'] = predictions_proba
    results['cme_prediction'] = predictions_binary
    results['prediction_confidence'] = np.where(
        predictions_proba > 0.5, 
        predictions_proba, 
        1 - predictions_proba
    )
    
    # Analysis
    print(f"\n📊 PREDICTION RESULTS:")
    print(f"Total data points: {len(results)}")
    print(f"Predicted CME arrivals: {predictions_binary.sum()}")
    print(f"Mean CME probability: {predictions_proba.mean():.4f}")
    print(f"Max CME probability: {predictions_proba.max():.4f}")
    print(f"Min CME probability: {predictions_proba.min():.4f}")
    
    # Show detailed results
    print(f"\n📋 DETAILED PREDICTIONS:")
    display_cols = ['time_unix', 'proton_density', 'bulk_speed', 'cme_arrival_probability', 'cme_prediction']
    available_cols = [col for col in display_cols if col in results.columns]
    print(results[available_cols].to_string())
    
    # Identify high-risk periods
    high_risk = results[results['cme_arrival_probability'] > 0.3]
    if len(high_risk) > 0:
        print(f"\n⚠️  HIGH-RISK PERIODS (probability > 30%):")
        print(high_risk[available_cols].to_string())
    else:
        print(f"\n✅ No high-risk periods detected")
    
    # Save results
    output_file = 'ace_wind_cme_predictions.csv'
    results.to_csv(output_file, index=False)
    print(f"\n💾 Results saved as: {output_file}")
    
    # Feature importance analysis
    if hasattr(model, 'feature_importances_'):
        print(f"\n🔍 FEATURE IMPORTANCE ANALYSIS:")
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("Top 5 most important features:")
        print(importance_df.head().to_string(index=False))
    
    return results, predictions_proba

if __name__ == "__main__":
    results, predictions = test_ace_wind_data()
