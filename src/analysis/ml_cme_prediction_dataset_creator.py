#!/usr/bin/env python3
"""
APEX SWISS - CACTUS CME Machine Learning Dataset Creator
========================================================
Creates a comprehensive dataset for machine learning models to predict
CME arrival times with 12-hour time strips, halo CME classification (1-4),
and transit time visualization.

This script combines APEX SWISS in-situ measurements with CACTUS CME observations
to create features for:
1. CME arrival time prediction
2. Halo CME classification (1-4 scale)
3. Transit time modeling
4. 12-hour time strip analysis

Author: Space Weather Analysis Team
Date: July 6, 2025
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, classification_report
import cdflib
from cdflib import cdfepoch

warnings.filterwarnings('ignore')

class MLCMEPredictionDatasetCreator:
    def __init__(self):
        """Initialize the ML dataset creator"""
        self.apex_data_path = "/Volumes/T7/ISRO/APEX_SWISS_BLK_FILES/analysis/data/apex_swiss_temporal_sample_aug2024_jun2025.csv"
        self.cactus_data_path = "/Volumes/T7/ISRO/CACTUS_Data_Analysis/CACTUS_CME_Corrected_Aug2024_Jun2025.csv"
        self.correlation_data_path = "/Volumes/T7/ISRO/APEX_SWISS_BLK_FILES/analysis/data/apex_cactus_cme_correlations.csv"
        
        self.output_dir = "/Volumes/T7/ISRO/APEX_SWISS_BLK_FILES/analysis"
        self.data_output_dir = os.path.join(self.output_dir, 'data')
        self.graphs_dir = os.path.join(self.output_dir, 'graphs')
        self.ml_output_dir = os.path.join(self.data_output_dir, 'ml_datasets')
        
        # Create directories
        for dir_path in [self.ml_output_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Set plotting style
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (20, 12)
        plt.rcParams['font.size'] = 10
        
        print("🚀 APEX SWISS - CACTUS ML DATASET CREATOR")
        print("=" * 60)
        print("📁 Output Directory:", self.ml_output_dir)
        
    def load_data(self):
        """Load all required datasets"""
        print("\n📥 Loading datasets...")
        
        # Load APEX SWISS data
        print("   📊 Loading APEX SWISS temporal data...")
        self.apex_df = pd.read_csv(self.apex_data_path)
        self.apex_df['timestamp'] = pd.to_datetime(self.apex_df['timestamp'])
        print(f"      ✅ APEX SWISS: {len(self.apex_df):,} data points")
        
        # Load CACTUS CME data
        print("   🌞 Loading CACTUS CME data...")
        self.cactus_df = pd.read_csv(self.cactus_data_path)
        # Try different date formats
        self.cactus_df['t0'] = pd.to_datetime(self.cactus_df['t0'], errors='coerce')
        self.cactus_df = self.cactus_df.dropna(subset=['t0'])
        print(f"      ✅ CACTUS CME: {len(self.cactus_df)} events")
        
        # Load existing correlation data
        print("   🔗 Loading correlation data...")
        self.correlation_df = pd.read_csv(self.correlation_data_path)
        self.correlation_df['cme_time'] = pd.to_datetime(self.correlation_df['cme_time'])
        self.correlation_df['expected_arrival'] = pd.to_datetime(self.correlation_df['expected_arrival'])
        print(f"      ✅ Correlations: {len(self.correlation_df)} CME-APEX pairs")
        
    def classify_halo_cmes(self):
        """Classify halo CMEs on a scale of 1-4"""
        print("\n🎯 Classifying halo CMEs (1-4 scale)...")
        
        # Create halo classification for CACTUS data
        def classify_halo(row):
            """
            Halo CME Classification (1-4):
            1 = No halo (normal CME, angular width < 120°)
            2 = Partial halo (120° ≤ angular width < 300°)
            3 = Full halo (angular width ≥ 300°)
            4 = Complex/Multiple halo (type IV or multiple events)
            """
            if pd.isna(row['da']):
                return 1  # Default to normal
            
            angular_width = row['da']
            halo_type = str(row.get('halo?', '')).strip()
            
            # Complex/Multiple halo
            if halo_type == 'IV':
                return 4
            
            # Full halo
            if angular_width >= 300 or halo_type in ['III']:
                return 3
            
            # Partial halo
            if angular_width >= 120 or halo_type in ['II']:
                return 2
            
            # Normal CME
            return 1
        
        self.cactus_df['halo_class'] = self.cactus_df.apply(classify_halo, axis=1)
        
        # Also classify velocity classes
        def classify_velocity(velocity):
            """Velocity classification for CMEs"""
            if pd.isna(velocity):
                return 'Unknown'
            elif velocity < 300:
                return 'Slow'
            elif velocity < 500:
                return 'Medium'
            elif velocity < 800:
                return 'Fast'
            else:
                return 'Very Fast'
        
        self.cactus_df['velocity_class'] = self.cactus_df['v'].apply(classify_velocity)
        
        # Print classification summary
        halo_counts = self.cactus_df['halo_class'].value_counts().sort_index()
        print(f"   Halo CME Classification Summary:")
        print(f"      Class 1 (Normal): {halo_counts.get(1, 0)} events")
        print(f"      Class 2 (Partial Halo): {halo_counts.get(2, 0)} events")
        print(f"      Class 3 (Full Halo): {halo_counts.get(3, 0)} events")
        print(f"      Class 4 (Complex Halo): {halo_counts.get(4, 0)} events")
        
        velocity_counts = self.cactus_df['velocity_class'].value_counts()
        print(f"   Velocity Classification Summary:")
        for vel_class, count in velocity_counts.items():
            print(f"      {vel_class}: {count} events")
    
    def calculate_transit_times(self):
        """Calculate actual transit times for CMEs with APEX signatures"""
        print("\n⏱️  Calculating CME transit times...")
        
        # Calculate theoretical transit time (Sun to L1)
        def calculate_theoretical_transit(velocity_km_s):
            """Calculate theoretical transit time from Sun to L1"""
            if pd.isna(velocity_km_s) or velocity_km_s <= 0:
                return np.nan
            
            # Distance Sun to L1 ≈ 1.5 million km
            distance_km = 1.5e6
            transit_time_seconds = distance_km / velocity_km_s
            transit_time_hours = transit_time_seconds / 3600
            return transit_time_hours
        
        self.cactus_df['theoretical_transit_hours'] = self.cactus_df['v'].apply(calculate_theoretical_transit)
        
        # For correlation data, calculate observed transit times
        transit_data = []
        
        for idx, row in self.correlation_df.iterrows():
            cme_time = row['cme_time']
            expected_arrival = row['expected_arrival']
            
            # Find actual solar wind enhancement in APEX data
            search_start = expected_arrival - timedelta(hours=24)
            search_end = expected_arrival + timedelta(hours=24)
            
            apex_window = self.apex_df[
                (self.apex_df['timestamp'] >= search_start) & 
                (self.apex_df['timestamp'] <= search_end)
            ].copy()
            
            if len(apex_window) > 0:
                # Look for density enhancement as CME signature
                density_threshold = apex_window['proton_density'].quantile(0.8)
                enhanced_periods = apex_window[apex_window['proton_density'] > density_threshold]
                
                if len(enhanced_periods) > 0:
                    # Take first enhancement as arrival time
                    actual_arrival = enhanced_periods['timestamp'].min()
                    actual_transit_hours = (actual_arrival - cme_time).total_seconds() / 3600
                    
                    transit_data.append({
                        'cme_number': row['cme_number'],
                        'cme_time': cme_time,
                        'expected_arrival': expected_arrival,
                        'actual_arrival': actual_arrival,
                        'theoretical_transit_hours': (expected_arrival - cme_time).total_seconds() / 3600,
                        'actual_transit_hours': actual_transit_hours,
                        'transit_error_hours': actual_transit_hours - ((expected_arrival - cme_time).total_seconds() / 3600),
                        'cme_velocity': row['cme_velocity'],
                        'cme_width': row['cme_width']
                    })
        
        self.transit_df = pd.DataFrame(transit_data)
        print(f"   ✅ Calculated transit times for {len(self.transit_df)} CME events")
        
        if len(self.transit_df) > 0:
            mean_error = self.transit_df['transit_error_hours'].abs().mean()
            print(f"   📊 Mean transit time prediction error: {mean_error:.1f} hours")
    
    def create_12hour_time_strips(self):
        """Create 12-hour time strip features for ML models"""
        print("\n📊 Creating 12-hour time strip features...")
        
        # Create time strip features for each CME correlation
        strip_features = []
        
        for idx, row in self.correlation_df.iterrows():
            cme_time = row['cme_time']
            expected_arrival = row['expected_arrival']
            
            # Create multiple time strips around the event
            strips = [
                ('pre_cme_12h', cme_time - timedelta(hours=12), cme_time),
                ('post_cme_12h', cme_time, cme_time + timedelta(hours=12)),
                ('pre_arrival_12h', expected_arrival - timedelta(hours=12), expected_arrival),
                ('post_arrival_12h', expected_arrival, expected_arrival + timedelta(hours=12))
            ]
            
            strip_data = {'cme_number': row['cme_number']}
            
            for strip_name, start_time, end_time in strips:
                # Get APEX data for this strip
                strip_apex = self.apex_df[
                    (self.apex_df['timestamp'] >= start_time) & 
                    (self.apex_df['timestamp'] <= end_time)
                ].copy()
                
                if len(strip_apex) > 0:
                    # Calculate statistical features for each parameter
                    for param in ['proton_density', 'proton_bulk_speed', 'proton_temperature', 
                                 'alpha_density', 'alpha_proton_ratio']:
                        if param in strip_apex.columns:
                            valid_data = strip_apex[param].dropna()
                            if len(valid_data) > 0:
                                strip_data[f'{strip_name}_{param}_mean'] = valid_data.mean()
                                strip_data[f'{strip_name}_{param}_std'] = valid_data.std()
                                strip_data[f'{strip_name}_{param}_max'] = valid_data.max()
                                strip_data[f'{strip_name}_{param}_min'] = valid_data.min()
                                strip_data[f'{strip_name}_{param}_trend'] = self.calculate_trend(valid_data)
                            else:
                                strip_data[f'{strip_name}_{param}_mean'] = np.nan
                                strip_data[f'{strip_name}_{param}_std'] = np.nan
                                strip_data[f'{strip_name}_{param}_max'] = np.nan
                                strip_data[f'{strip_name}_{param}_min'] = np.nan
                                strip_data[f'{strip_name}_{param}_trend'] = np.nan
                else:
                    # No data available for this strip
                    for param in ['proton_density', 'proton_bulk_speed', 'proton_temperature', 
                                 'alpha_density', 'alpha_proton_ratio']:
                        strip_data[f'{strip_name}_{param}_mean'] = np.nan
                        strip_data[f'{strip_name}_{param}_std'] = np.nan
                        strip_data[f'{strip_name}_{param}_max'] = np.nan
                        strip_data[f'{strip_name}_{param}_min'] = np.nan
                        strip_data[f'{strip_name}_{param}_trend'] = np.nan
            
            strip_features.append(strip_data)
        
        self.time_strip_features = pd.DataFrame(strip_features)
        print(f"   ✅ Created 12-hour time strip features for {len(self.time_strip_features)} CME events")
        print(f"   📊 Feature dimensions: {self.time_strip_features.shape}")
    
    def calculate_trend(self, data):
        """Calculate trend (slope) of time series data"""
        if len(data) < 2:
            return 0
        
        x = np.arange(len(data))
        try:
            slope = np.polyfit(x, data, 1)[0]
            return slope
        except:
            return 0
    
    def create_ml_features(self):
        """Create comprehensive ML feature set"""
        print("\n🔧 Creating ML feature set...")
        
        # Merge all data sources
        ml_dataset = self.correlation_df.copy()
        
        # Add CACTUS CME features
        cactus_features = self.cactus_df[['CME', 'v', 'da', 'pa', 'halo_class', 'velocity_class', 'theoretical_transit_hours']].copy()
        cactus_features.rename(columns={'CME': 'cme_number'}, inplace=True)
        
        ml_dataset = ml_dataset.merge(cactus_features, on='cme_number', how='left', suffixes=('', '_cactus'))
        
        # Add time strip features
        ml_dataset = ml_dataset.merge(self.time_strip_features, on='cme_number', how='left')
        
        # Add transit time data if available
        if hasattr(self, 'transit_df') and len(self.transit_df) > 0:
            transit_features = self.transit_df[['cme_number', 'actual_transit_hours', 'transit_error_hours']].copy()
            ml_dataset = ml_dataset.merge(transit_features, on='cme_number', how='left')
        
        # Create derived features
        ml_dataset['cme_kinetic_energy'] = 0.5 * ml_dataset['cme_velocity']**2  # Proportional to kinetic energy
        ml_dataset['cme_momentum'] = ml_dataset['cme_velocity'] * ml_dataset['cme_width']  # Momentum proxy
        
        # Time-based features
        ml_dataset['cme_hour'] = ml_dataset['cme_time'].dt.hour
        ml_dataset['cme_day_of_year'] = ml_dataset['cme_time'].dt.dayofyear
        ml_dataset['cme_month'] = ml_dataset['cme_time'].dt.month
        
        # Earth-directed classification
        ml_dataset['earth_directed'] = (
            ((ml_dataset['pa'] >= 270) & (ml_dataset['pa'] <= 360)) | 
            ((ml_dataset['pa'] >= 0) & (ml_dataset['pa'] <= 90))
        ).astype(int)
        
        self.ml_dataset = ml_dataset
        print(f"   ✅ Created ML dataset with {len(ml_dataset)} samples and {ml_dataset.shape[1]} features")
        
        # Print feature summary
        print(f"\n📋 Feature Categories:")
        print(f"   🌞 CACTUS CME features: velocity, width, position angle, halo class")
        print(f"   🛰️  APEX SWISS features: solar wind parameters (mean, std, max, min, trend)")
        print(f"   ⏱️  Time strip features: 12-hour pre/post CME and arrival windows")
        print(f"   🎯 Target variables: actual transit time, halo classification")
    
    def prepare_ml_datasets(self):
        """Prepare training/test datasets for different ML tasks"""
        print("\n🎯 Preparing ML datasets for different tasks...")
        
        # Task 1: CME Arrival Time Prediction (Regression)
        print("   📊 Task 1: CME Arrival Time Prediction")
        
        # Features for arrival time prediction
        feature_cols = [col for col in self.ml_dataset.columns if any(x in col for x in [
            'cme_velocity', 'cme_width', 'pa', 'cme_kinetic_energy', 'cme_momentum',
            'cme_hour', 'cme_day_of_year', 'cme_month', 'earth_directed',
            'pre_cme_12h', 'post_cme_12h'
        ]) and col not in ['cme_time', 'expected_arrival', 'actual_arrival']]
        
        # Target: actual transit time (if available) or expected transit time
        if 'actual_transit_hours' in self.ml_dataset.columns:
            target_col = 'actual_transit_hours'
            available_data = self.ml_dataset.dropna(subset=[target_col])
        else:
            # Use theoretical transit time as proxy
            self.ml_dataset['target_transit_hours'] = (
                self.ml_dataset['expected_arrival'] - self.ml_dataset['cme_time']
            ).dt.total_seconds() / 3600
            target_col = 'target_transit_hours'
            available_data = self.ml_dataset.dropna(subset=[target_col])
        
        X_transit = available_data[feature_cols].select_dtypes(include=[np.number])
        y_transit = available_data[target_col]
        
        # Handle missing values more robustly
        # First, remove columns that are all NaN
        X_transit = X_transit.dropna(axis=1, how='all')
        # Then fill remaining NaN values with median
        X_transit = X_transit.fillna(X_transit.median())
        # If there are still NaN values, fill with 0
        X_transit = X_transit.fillna(0)
        
        self.transit_prediction_data = {
            'X': X_transit,
            'y': y_transit,
            'feature_names': X_transit.columns.tolist(),
            'sample_count': len(X_transit)
        }
        
        print(f"      ✅ Transit time prediction: {len(X_transit)} samples, {X_transit.shape[1]} features")
        
        # Task 2: Halo CME Classification
        print("   🎯 Task 2: Halo CME Classification")
        
        halo_data = self.ml_dataset.dropna(subset=['halo_class'])
        X_halo = halo_data[feature_cols].select_dtypes(include=[np.number])
        y_halo = halo_data['halo_class']
        
        # Handle missing values more robustly
        X_halo = X_halo.dropna(axis=1, how='all')
        X_halo = X_halo.fillna(X_halo.median())
        X_halo = X_halo.fillna(0)
        
        self.halo_classification_data = {
            'X': X_halo,
            'y': y_halo,
            'feature_names': X_halo.columns.tolist(),
            'sample_count': len(X_halo),
            'class_distribution': y_halo.value_counts().to_dict()
        }
        
        print(f"      ✅ Halo classification: {len(X_halo)} samples, {X_halo.shape[1]} features")
        print(f"         Class distribution: {y_halo.value_counts().to_dict()}")
        
        # Task 3: Combined Prediction (Multi-output)
        print("   🔄 Task 3: Combined Multi-output Prediction")
        
        # Combined dataset with both targets
        combined_data = available_data.dropna(subset=['halo_class'])
        X_combined = combined_data[feature_cols].select_dtypes(include=[np.number])
        X_combined = X_combined.dropna(axis=1, how='all')
        X_combined = X_combined.fillna(X_combined.median())
        X_combined = X_combined.fillna(0)
        
        y_combined_transit = combined_data[target_col]
        y_combined_halo = combined_data['halo_class']
        
        self.combined_prediction_data = {
            'X': X_combined,
            'y_transit': y_combined_transit,
            'y_halo': y_combined_halo,
            'feature_names': X_combined.columns.tolist(),
            'sample_count': len(X_combined)
        }
        
        print(f"      ✅ Combined prediction: {len(X_combined)} samples, {X_combined.shape[1]} features")
    
    def train_baseline_models(self):
        """Train baseline ML models for evaluation"""
        print("\n🤖 Training baseline ML models...")
        
        # Model 1: Transit Time Prediction
        print("   📊 Training transit time prediction model...")
        
        X = self.transit_prediction_data['X']
        y = self.transit_prediction_data['y']
        
        if len(X) > 10:  # Need minimum samples
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler_transit = StandardScaler()
            X_train_scaled = scaler_transit.fit_transform(X_train)
            X_test_scaled = scaler_transit.transform(X_test)
            
            # Train Random Forest
            rf_transit = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_transit.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = rf_transit.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            
            self.transit_model = {
                'model': rf_transit,
                'scaler': scaler_transit,
                'mae': mae,
                'feature_importance': dict(zip(X.columns, rf_transit.feature_importances_))
            }
            
            print(f"      ✅ Transit time model MAE: {mae:.2f} hours")
        
        # Model 2: Halo Classification
        print("   🎯 Training halo classification model...")
        
        X = self.halo_classification_data['X']
        y = self.halo_classification_data['y']
        
        if len(X) > 10:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler_halo = StandardScaler()
            X_train_scaled = scaler_halo.fit_transform(X_train)
            X_test_scaled = scaler_halo.transform(X_test)
            
            # Train Random Forest
            rf_halo = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_halo.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = rf_halo.predict(X_test_scaled)
            accuracy = rf_halo.score(X_test_scaled, y_test)
            
            self.halo_model = {
                'model': rf_halo,
                'scaler': scaler_halo,
                'accuracy': accuracy,
                'feature_importance': dict(zip(X.columns, rf_halo.feature_importances_))
            }
            
            print(f"      ✅ Halo classification accuracy: {accuracy:.3f}")
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n📊 Creating ML dataset visualizations...")
        
        # Create main dashboard
        fig = plt.figure(figsize=(24, 20))
        
        # 1. Halo CME Distribution
        ax1 = plt.subplot(3, 4, 1)
        halo_counts = self.cactus_df['halo_class'].value_counts().sort_index()
        bars = ax1.bar(halo_counts.index, halo_counts.values, 
                      color=['skyblue', 'orange', 'red', 'darkred'])
        ax1.set_title('Halo CME Classification Distribution', fontweight='bold')
        ax1.set_xlabel('Halo Class')
        ax1.set_ylabel('Number of CMEs')
        ax1.set_xticks([1, 2, 3, 4])
        ax1.set_xticklabels(['Normal', 'Partial\nHalo', 'Full\nHalo', 'Complex\nHalo'])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # 2. Velocity vs Angular Width
        ax2 = plt.subplot(3, 4, 2)
        scatter = ax2.scatter(self.cactus_df['v'], self.cactus_df['da'], 
                            c=self.cactus_df['halo_class'], cmap='viridis', alpha=0.6)
        ax2.set_title('CME Velocity vs Angular Width', fontweight='bold')
        ax2.set_xlabel('Velocity (km/s)')
        ax2.set_ylabel('Angular Width (degrees)')
        plt.colorbar(scatter, ax=ax2, label='Halo Class')
        
        # 3. Transit Time Distribution
        if hasattr(self, 'transit_df') and len(self.transit_df) > 0:
            ax3 = plt.subplot(3, 4, 3)
            ax3.hist(self.transit_df['actual_transit_hours'], bins=20, alpha=0.7, color='lightgreen')
            ax3.set_title('Actual Transit Time Distribution', fontweight='bold')
            ax3.set_xlabel('Transit Time (hours)')
            ax3.set_ylabel('Frequency')
            ax3.axvline(self.transit_df['actual_transit_hours'].mean(), color='red', 
                       linestyle='--', label=f'Mean: {self.transit_df["actual_transit_hours"].mean():.1f}h')
            ax3.legend()
        
        # 4. Velocity Distribution by Halo Class
        ax4 = plt.subplot(3, 4, 4)
        for halo_class in sorted(self.cactus_df['halo_class'].unique()):
            data = self.cactus_df[self.cactus_df['halo_class'] == halo_class]['v'].dropna()
            ax4.hist(data, alpha=0.5, label=f'Class {halo_class}', bins=15)
        ax4.set_title('Velocity Distribution by Halo Class', fontweight='bold')
        ax4.set_xlabel('Velocity (km/s)')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        
        # 5. Transit Time Prediction Accuracy
        if hasattr(self, 'transit_df') and len(self.transit_df) > 0:
            ax5 = plt.subplot(3, 4, 5)
            ax5.scatter(self.transit_df['theoretical_transit_hours'], 
                       self.transit_df['actual_transit_hours'], alpha=0.6)
            ax5.plot([0, 200], [0, 200], 'r--', label='Perfect Prediction')
            ax5.set_title('Predicted vs Actual Transit Times', fontweight='bold')
            ax5.set_xlabel('Theoretical Transit Time (hours)')
            ax5.set_ylabel('Actual Transit Time (hours)')
            ax5.legend()
        
        # 6. Feature Importance (if models trained)
        if hasattr(self, 'transit_model'):
            ax6 = plt.subplot(3, 4, 6)
            importance = self.transit_model['feature_importance']
            # Get top 10 features
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            features, values = zip(*top_features)
            ax6.barh(range(len(features)), values)
            ax6.set_yticks(range(len(features)))
            ax6.set_yticklabels([f.replace('_', ' ').title() for f in features])
            ax6.set_title('Top Features for Transit Time Prediction', fontweight='bold')
            ax6.set_xlabel('Feature Importance')
        
        # 7. Solar Wind Parameter Correlation
        ax7 = plt.subplot(3, 4, 7)
        if len(self.correlation_df) > 0:
            corr_params = ['proton_density_mean', 'proton_bulk_speed_mean', 
                          'proton_temperature_mean', 'alpha_density_mean']
            available_params = [p for p in corr_params if p in self.correlation_df.columns]
            if len(available_params) > 1:
                corr_matrix = self.correlation_df[available_params].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax7)
                ax7.set_title('Solar Wind Parameter Correlations', fontweight='bold')
        
        # 8. CME Frequency by Month
        ax8 = plt.subplot(3, 4, 8)
        monthly_cmes = self.cactus_df.groupby(self.cactus_df['t0'].dt.month).size()
        ax8.bar(monthly_cmes.index, monthly_cmes.values, color='lightcoral')
        ax8.set_title('CME Frequency by Month', fontweight='bold')
        ax8.set_xlabel('Month')
        ax8.set_ylabel('Number of CMEs')
        ax8.set_xticks(range(1, 13))
        
        # 9. Earth-Directed CME Analysis
        ax9 = plt.subplot(3, 4, 9)
        earth_directed = self.ml_dataset['earth_directed'].value_counts()
        ax9.pie(earth_directed.values, labels=['Not Earth-Directed', 'Earth-Directed'], 
               autopct='%1.1f%%', startangle=90)
        ax9.set_title('Earth-Directed CME Distribution', fontweight='bold')
        
        # 10. 12-Hour Time Strip Sample
        ax10 = plt.subplot(3, 4, 10)
        if len(self.time_strip_features) > 0:
            sample_features = [col for col in self.time_strip_features.columns 
                              if 'proton_density_mean' in col]
            if sample_features:
                data = self.time_strip_features[sample_features].mean()
                ax10.bar(range(len(data)), data.values)
                ax10.set_title('12-Hour Strip: Mean Proton Density', fontweight='bold')
                ax10.set_ylabel('Density (cm⁻³)')
                ax10.set_xticks(range(len(data)))
                ax10.set_xticklabels([s.replace('_proton_density_mean', '') for s in sample_features], 
                                   rotation=45)
        
        # 11. Model Performance Summary
        ax11 = plt.subplot(3, 4, 11)
        if hasattr(self, 'transit_model') and hasattr(self, 'halo_model'):
            models = ['Transit Time\nPrediction', 'Halo\nClassification']
            scores = [1 - self.transit_model['mae']/100, self.halo_model['accuracy']]  # Normalized
            bars = ax11.bar(models, scores, color=['lightblue', 'lightgreen'])
            ax11.set_title('Model Performance Summary', fontweight='bold')
            ax11.set_ylabel('Performance Score')
            ax11.set_ylim(0, 1)
            
            # Add score labels
            for bar, score in zip(bars, scores):
                ax11.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                         f'{score:.3f}', ha='center', va='bottom')
        
        # 12. Dataset Size Summary
        ax12 = plt.subplot(3, 4, 12)
        dataset_sizes = {
            'APEX Data': len(self.apex_df),
            'CACTUS CMEs': len(self.cactus_df),
            'Correlations': len(self.correlation_df),
            'ML Features': len(self.ml_dataset)
        }
        ax12.bar(dataset_sizes.keys(), dataset_sizes.values(), color='gold')
        ax12.set_title('Dataset Size Summary', fontweight='bold')
        ax12.set_ylabel('Number of Records')
        ax12.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save visualization
        plot_path = os.path.join(self.graphs_dir, 'ml_cme_prediction_dataset_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ ML dataset visualization saved: ml_cme_prediction_dataset_analysis.png")
    
    def export_ml_datasets(self):
        """Export all ML-ready datasets"""
        print("\n💾 Exporting ML-ready datasets...")
        
        # Export main ML dataset
        main_output = os.path.join(self.ml_output_dir, 'cme_prediction_ml_dataset.csv')
        self.ml_dataset.to_csv(main_output, index=False, float_format='%.6f')
        print(f"   ✅ Main ML dataset: cme_prediction_ml_dataset.csv ({self.ml_dataset.shape})")
        
        # Export time strip features
        strip_output = os.path.join(self.ml_output_dir, 'cme_12hour_time_strip_features.csv')
        self.time_strip_features.to_csv(strip_output, index=False, float_format='%.6f')
        print(f"   ✅ Time strip features: cme_12hour_time_strip_features.csv ({self.time_strip_features.shape})")
        
        # Export enhanced CACTUS data
        cactus_output = os.path.join(self.ml_output_dir, 'cactus_cme_enhanced_with_halo_classification.csv')
        self.cactus_df.to_csv(cactus_output, index=False, float_format='%.6f')
        print(f"   ✅ Enhanced CACTUS data: cactus_cme_enhanced_with_halo_classification.csv ({self.cactus_df.shape})")
        
        # Export transit time data
        if hasattr(self, 'transit_df') and len(self.transit_df) > 0:
            transit_output = os.path.join(self.ml_output_dir, 'cme_transit_time_analysis.csv')
            self.transit_df.to_csv(transit_output, index=False, float_format='%.6f')
            print(f"   ✅ Transit time data: cme_transit_time_analysis.csv ({self.transit_df.shape})")
        
        # Export feature-target splits for specific ML tasks
        if hasattr(self, 'transit_prediction_data'):
            # Transit time prediction dataset
            X_transit = self.transit_prediction_data['X']
            y_transit = self.transit_prediction_data['y']
            
            transit_ml = X_transit.copy()
            transit_ml['target_transit_hours'] = y_transit
            
            transit_ml_output = os.path.join(self.ml_output_dir, 'transit_time_prediction_dataset.csv')
            transit_ml.to_csv(transit_ml_output, index=False, float_format='%.6f')
            print(f"   ✅ Transit prediction dataset: transit_time_prediction_dataset.csv ({transit_ml.shape})")
        
        if hasattr(self, 'halo_classification_data'):
            # Halo classification dataset
            X_halo = self.halo_classification_data['X']
            y_halo = self.halo_classification_data['y']
            
            halo_ml = X_halo.copy()
            halo_ml['target_halo_class'] = y_halo
            
            halo_ml_output = os.path.join(self.ml_output_dir, 'halo_classification_dataset.csv')
            halo_ml.to_csv(halo_ml_output, index=False, float_format='%.6f')
            print(f"   ✅ Halo classification dataset: halo_classification_dataset.csv ({halo_ml.shape})")
    
    def generate_ml_documentation(self):
        """Generate comprehensive documentation for ML datasets"""
        print("\n📄 Generating ML dataset documentation...")
        
        doc_path = os.path.join(self.ml_output_dir, 'ML_Dataset_Documentation.md')
        
        with open(doc_path, 'w') as f:
            f.write("# APEX SWISS - CACTUS CME Machine Learning Dataset Documentation\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Period:** August 2024 - June 2025\n")
            f.write(f"**Purpose:** CME arrival time prediction and halo classification\n\n")
            
            f.write("## Dataset Overview\n\n")
            f.write("This comprehensive dataset combines APEX SWISS in-situ solar wind measurements\n")
            f.write("with CACTUS Coronal Mass Ejection observations to enable machine learning\n")
            f.write("models for space weather prediction.\n\n")
            
            f.write("## Halo CME Classification Scale (1-4)\n\n")
            f.write("| Class | Description | Criteria |\n")
            f.write("|-------|-------------|----------|\n")
            f.write("| 1 | Normal CME | Angular width < 120° |\n")
            f.write("| 2 | Partial Halo | 120° ≤ angular width < 300° |\n")
            f.write("| 3 | Full Halo | Angular width ≥ 300° |\n")
            f.write("| 4 | Complex Halo | Multiple/complex structure |\n\n")
            
            f.write("## Dataset Files\n\n")
            f.write("### Primary Datasets\n")
            f.write("1. **`cme_prediction_ml_dataset.csv`** - Complete ML dataset with all features\n")
            f.write("2. **`cme_12hour_time_strip_features.csv`** - 12-hour time strip features\n")
            f.write("3. **`cactus_cme_enhanced_with_halo_classification.csv`** - Enhanced CACTUS data\n\n")
            
            f.write("### Task-Specific Datasets\n")
            f.write("4. **`transit_time_prediction_dataset.csv`** - CME arrival time prediction\n")
            f.write("5. **`halo_classification_dataset.csv`** - Halo CME classification\n")
            f.write("6. **`cme_transit_time_analysis.csv`** - Transit time analysis\n\n")
            
            f.write("## Feature Categories\n\n")
            f.write("### CACTUS CME Features\n")
            f.write("- `cme_velocity`: CME propagation velocity (km/s)\n")
            f.write("- `cme_width`: Angular width (degrees)\n")
            f.write("- `pa`: Position angle (degrees)\n")
            f.write("- `halo_class`: Halo classification (1-4)\n")
            f.write("- `velocity_class`: Velocity classification (Slow/Medium/Fast/Very Fast)\n")
            f.write("- `earth_directed`: Binary flag for Earth-directed CMEs\n\n")
            
            f.write("### APEX SWISS Solar Wind Features\n")
            f.write("- `proton_density`: Proton number density (cm⁻³)\n")
            f.write("- `proton_bulk_speed`: Solar wind speed (km/s)\n")
            f.write("- `proton_temperature`: Proton temperature (K)\n")
            f.write("- `alpha_density`: Alpha particle density (cm⁻³)\n")
            f.write("- `alpha_proton_ratio`: Alpha to proton ratio\n\n")
            
            f.write("### 12-Hour Time Strip Features\n")
            f.write("For each parameter, statistical measures (mean, std, max, min, trend) are calculated\n")
            f.write("for the following time windows:\n")
            f.write("- `pre_cme_12h`: 12 hours before CME\n")
            f.write("- `post_cme_12h`: 12 hours after CME\n")
            f.write("- `pre_arrival_12h`: 12 hours before expected arrival\n")
            f.write("- `post_arrival_12h`: 12 hours after expected arrival\n\n")
            
            f.write("### Derived Features\n")
            f.write("- `cme_kinetic_energy`: Proportional to v²\n")
            f.write("- `cme_momentum`: Velocity × angular width\n")
            f.write("- `cme_hour`: Hour of CME detection\n")
            f.write("- `cme_day_of_year`: Day of year\n")
            f.write("- `cme_month`: Month of detection\n\n")
            
            f.write("## Target Variables\n\n")
            f.write("### Regression Tasks\n")
            f.write("- `actual_transit_hours`: Observed CME transit time (hours)\n")
            f.write("- `target_transit_hours`: Target for transit time prediction\n\n")
            
            f.write("### Classification Tasks\n")
            f.write("- `target_halo_class`: Halo CME classification (1-4)\n\n")
            
            f.write("## Machine Learning Applications\n\n")
            f.write("### 1. CME Arrival Time Prediction\n")
            f.write("- **Type:** Regression\n")
            f.write("- **Target:** Transit time (hours)\n")
            f.write("- **Features:** CME properties + 12-hour pre-CME conditions\n")
            f.write("- **Use Case:** Space weather forecasting\n\n")
            
            f.write("### 2. Halo CME Classification\n")
            f.write("- **Type:** Multi-class classification\n")
            f.write("- **Target:** Halo class (1-4)\n")
            f.write("- **Features:** CME observational properties\n")
            f.write("- **Use Case:** Event severity assessment\n\n")
            
            f.write("### 3. Real-time CME Detection\n")
            f.write("- **Type:** Binary classification\n")
            f.write("- **Target:** CME presence in solar wind\n")
            f.write("- **Features:** 12-hour time strip statistics\n")
            f.write("- **Use Case:** Operational space weather monitoring\n\n")
            
            f.write("## Data Quality Notes\n\n")
            f.write("- Missing values are handled using median imputation for numerical features\n")
            f.write("- All timestamps are in UTC\n")
            f.write("- Features are normalized using StandardScaler for ML models\n")
            f.write("- Outliers are retained for physical realism\n\n")
            
            f.write("## Usage Examples\n\n")
            f.write("```python\n")
            f.write("import pandas as pd\n")
            f.write("from sklearn.ensemble import RandomForestRegressor\n")
            f.write("from sklearn.preprocessing import StandardScaler\n\n")
            f.write("# Load transit time prediction dataset\n")
            f.write("data = pd.read_csv('transit_time_prediction_dataset.csv')\n\n")
            f.write("# Separate features and target\n")
            f.write("X = data.drop('target_transit_hours', axis=1)\n")
            f.write("y = data['target_transit_hours']\n\n")
            f.write("# Train model\n")
            f.write("scaler = StandardScaler()\n")
            f.write("X_scaled = scaler.fit_transform(X)\n")
            f.write("model = RandomForestRegressor()\n")
            f.write("model.fit(X_scaled, y)\n")
            f.write("```\n\n")
            
            if hasattr(self, 'transit_model') and hasattr(self, 'halo_model'):
                f.write("## Baseline Model Performance\n\n")
                f.write(f"- **Transit Time Prediction MAE:** {self.transit_model['mae']:.2f} hours\n")
                f.write(f"- **Halo Classification Accuracy:** {self.halo_model['accuracy']:.3f}\n\n")
            
            f.write("## Citation\n\n")
            f.write("If you use this dataset, please cite:\n")
            f.write("- Aditya-L1 APEX SWISS instrument team\n")
            f.write("- CACTUS CME catalog (CACTus.oma.be)\n")
            f.write("- This analysis framework\n\n")
            
            f.write("---\n")
            f.write(f"*Documentation generated by ML Dataset Creator*\n")
            f.write(f"*Aditya-L1 Mission - Space Weather Analysis*\n")
        
        print(f"   ✅ Documentation saved: ML_Dataset_Documentation.md")
    
    def run_complete_analysis(self):
        """Run the complete ML dataset creation pipeline"""
        print("🚀 Starting APEX SWISS - CACTUS ML Dataset Creation...")
        
        # Load data
        self.load_data()
        
        # Classify halo CMEs
        self.classify_halo_cmes()
        
        # Calculate transit times
        self.calculate_transit_times()
        
        # Create time strip features
        self.create_12hour_time_strips()
        
        # Create ML features
        self.create_ml_features()
        
        # Prepare ML datasets
        self.prepare_ml_datasets()
        
        # Train baseline models
        self.train_baseline_models()
        
        # Create visualizations
        self.create_visualizations()
        
        # Export datasets
        self.export_ml_datasets()
        
        # Generate documentation
        self.generate_ml_documentation()
        
        print(f"\n🎉 ML DATASET CREATION COMPLETE!")
        print("=" * 70)
        print(f"📊 Main dataset: {self.ml_dataset.shape}")
        print(f"🎯 Halo classes: {sorted(self.cactus_df['halo_class'].unique())}")
        
        if hasattr(self, 'transit_df') and len(self.transit_df) > 0:
            print(f"⏱️  Transit events: {len(self.transit_df)}")
        
        if hasattr(self, 'transit_model'):
            print(f"🤖 Transit prediction MAE: {self.transit_model['mae']:.2f} hours")
        
        if hasattr(self, 'halo_model'):
            print(f"🎯 Halo classification accuracy: {self.halo_model['accuracy']:.3f}")
        
        print(f"📁 Outputs: {self.ml_output_dir}")
        print("=" * 70)

def main():
    """Main execution function"""
    creator = MLCMEPredictionDatasetCreator()
    creator.run_complete_analysis()

if __name__ == "__main__":
    main()
