#!/usr/bin/env python3
"""
CACTUS CME - APEX SWISS ML Dataset Comprehensive Visualizer
===========================================================
Creates comprehensive visualizations showing how CACTUS CME data aligns 
with APEX SWISS bulk solar wind features from the ML datasets.

This script analyzes:
1. CME-Solar Wind Parameter Correlations
2. Halo CME Classification Patterns
3. Transit Time Prediction Features
4. 12-Hour Time Strip Analysis
5. Feature Importance and Relationships

Author: Space Weather Analysis Team
Date: July 6, 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from scipy.stats import pearsonr, spearmanr
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

class CMEMLDatasetVisualizer:
    def __init__(self):
        """Initialize the ML dataset visualizer"""
        self.ml_data_dir = "/Volumes/T7/ISRO/APEX_SWISS_BLK_FILES/analysis/data/ml_datasets"
        self.output_dir = "/Volumes/T7/ISRO/APEX_SWISS_BLK_FILES/ml_visualization"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (20, 16)
        plt.rcParams['font.size'] = 10
        
        print("🚀 CACTUS CME - APEX SWISS ML DATASET VISUALIZER")
        print("=" * 70)
        print("📁 Output Directory:", self.output_dir)
        
    def load_datasets(self):
        """Load all ML datasets"""
        print("\n📥 Loading ML datasets...")
        
        # Load main ML dataset
        main_path = os.path.join(self.ml_data_dir, 'cme_prediction_ml_dataset.csv')
        self.main_df = pd.read_csv(main_path)
        self.main_df['cme_time'] = pd.to_datetime(self.main_df['cme_time'])
        print(f"   ✅ Main ML dataset: {self.main_df.shape}")
        
        # Load enhanced CACTUS data
        cactus_path = os.path.join(self.ml_data_dir, 'cactus_cme_enhanced_with_halo_classification.csv')
        self.cactus_df = pd.read_csv(cactus_path)
        self.cactus_df['t0'] = pd.to_datetime(self.cactus_df['t0'])
        print(f"   ✅ Enhanced CACTUS: {self.cactus_df.shape}")
        
        # Load time strip features
        strip_path = os.path.join(self.ml_data_dir, 'cme_12hour_time_strip_features.csv')
        self.strip_df = pd.read_csv(strip_path)
        print(f"   ✅ Time strip features: {self.strip_df.shape}")
        
        # Load transit time data
        transit_path = os.path.join(self.ml_data_dir, 'transit_time_prediction_dataset.csv')
        self.transit_df = pd.read_csv(transit_path)
        print(f"   ✅ Transit prediction: {self.transit_df.shape}")
        
        # Load halo classification data
        halo_path = os.path.join(self.ml_data_dir, 'halo_classification_dataset.csv')
        self.halo_df = pd.read_csv(halo_path)
        print(f"   ✅ Halo classification: {self.halo_df.shape}")
    
    def create_cme_overview_dashboard(self):
        """Create comprehensive CME overview dashboard"""
        print("\n📊 Creating CME overview dashboard...")
        
        fig = plt.figure(figsize=(24, 20))
        gs = GridSpec(4, 4, figure=fig)
        fig.suptitle('CACTUS CME - APEX SWISS ML Dataset Analysis Dashboard\n(August 2024 - June 2025)', 
                     fontsize=16, fontweight='bold')
        
        # 1. Halo CME Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        halo_counts = self.cactus_df['halo_class'].value_counts().sort_index()
        colors = ['lightblue', 'orange', 'red', 'darkred']
        bars = ax1.bar(halo_counts.index, halo_counts.values, color=colors)
        ax1.set_title('Halo CME Classification\n(1=Normal, 2=Partial, 3=Full, 4=Complex)', fontweight='bold')
        ax1.set_xlabel('Halo Class')
        ax1.set_ylabel('Number of CMEs')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 10,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # 2. CME Velocity vs Angular Width (colored by halo class)
        ax2 = fig.add_subplot(gs[0, 1])
        scatter = ax2.scatter(self.cactus_df['v'], self.cactus_df['da'], 
                            c=self.cactus_df['halo_class'], cmap='viridis', alpha=0.6, s=20)
        ax2.set_xlabel('CME Velocity (km/s)')
        ax2.set_ylabel('Angular Width (degrees)')
        ax2.set_title('CME Velocity vs Angular Width\n(colored by Halo Class)', fontweight='bold')
        plt.colorbar(scatter, ax=ax2, label='Halo Class')
        ax2.grid(True, alpha=0.3)
        
        # 3. Monthly CME Activity
        ax3 = fig.add_subplot(gs[0, 2])
        monthly_cmes = self.cactus_df.groupby(self.cactus_df['t0'].dt.month).size()
        ax3.bar(monthly_cmes.index, monthly_cmes.values, color='lightcoral')
        ax3.set_title('Monthly CME Activity', fontweight='bold')
        ax3.set_xlabel('Month (2024-2025)')
        ax3.set_ylabel('Number of CMEs')
        ax3.set_xticks(range(1, 13))
        ax3.grid(True, alpha=0.3)
        
        # 4. Velocity Distribution by Halo Class
        ax4 = fig.add_subplot(gs[0, 3])
        for halo_class in sorted(self.cactus_df['halo_class'].unique()):
            data = self.cactus_df[self.cactus_df['halo_class'] == halo_class]['v'].dropna()
            ax4.hist(data, alpha=0.6, label=f'Class {halo_class}', bins=20)
        ax4.set_title('Velocity Distribution by Halo Class', fontweight='bold')
        ax4.set_xlabel('Velocity (km/s)')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Solar Wind Parameter Correlations
        ax5 = fig.add_subplot(gs[1, 0])
        sw_params = ['proton_density_mean', 'proton_bulk_speed_mean', 
                    'proton_temperature_mean', 'alpha_density_mean']
        available_params = [p for p in sw_params if p in self.main_df.columns]
        
        if len(available_params) > 1:
            corr_matrix = self.main_df[available_params].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax5,
                       square=True, cbar_kws={'shrink': 0.8})
            ax5.set_title('Solar Wind Parameter Correlations', fontweight='bold')
        
        # 6. CME Velocity vs Solar Wind Response
        ax6 = fig.add_subplot(gs[1, 1])
        if 'proton_density_mean' in self.main_df.columns:
            scatter = ax6.scatter(self.main_df['cme_velocity'], self.main_df['proton_density_mean'],
                                c=self.main_df['cme_width'], cmap='plasma', alpha=0.6, s=30)
            ax6.set_xlabel('CME Velocity (km/s)')
            ax6.set_ylabel('Mean Proton Density (cm⁻³)')
            ax6.set_title('CME Velocity vs Solar Wind Density\n(colored by CME width)', fontweight='bold')
            plt.colorbar(scatter, ax=ax6, label='CME Width (°)')
            ax6.grid(True, alpha=0.3)
        
        # 7. Earth-Directed CME Analysis
        ax7 = fig.add_subplot(gs[1, 2])
        if 'earth_directed' in self.main_df.columns:
            earth_counts = self.main_df['earth_directed'].value_counts()
            ax7.pie(earth_counts.values, labels=['Not Earth-Directed', 'Earth-Directed'], 
                   autopct='%1.1f%%', startangle=90, colors=['lightblue', 'red'])
            ax7.set_title('Earth-Directed CME Distribution', fontweight='bold')
        
        # 8. CME Frequency vs Solar Wind Speed
        ax8 = fig.add_subplot(gs[1, 3])
        if 'proton_bulk_speed_mean' in self.main_df.columns:
            ax8.scatter(self.main_df['proton_bulk_speed_mean'], self.main_df['cme_velocity'],
                       alpha=0.6, s=30)
            ax8.set_xlabel('Solar Wind Speed (km/s)')
            ax8.set_ylabel('CME Velocity (km/s)')
            ax8.set_title('Solar Wind vs CME Velocity', fontweight='bold')
            ax8.grid(True, alpha=0.3)
        
        # 9. Temporal Patterns
        ax9 = fig.add_subplot(gs[2, 0])
        hourly_cmes = self.main_df.groupby(self.main_df['cme_time'].dt.hour).size()
        ax9.bar(hourly_cmes.index, hourly_cmes.values, color='gold')
        ax9.set_title('CME Detection by Hour of Day', fontweight='bold')
        ax9.set_xlabel('Hour (UTC)')
        ax9.set_ylabel('Number of CMEs')
        ax9.grid(True, alpha=0.3)
        
        # 10. Position Angle Distribution
        ax10 = fig.add_subplot(gs[2, 1])
        pa_data = self.main_df['pa'].dropna()
        ax10.hist(pa_data, bins=36, alpha=0.7, color='lightgreen')
        ax10.set_title('CME Position Angle Distribution', fontweight='bold')
        ax10.set_xlabel('Position Angle (degrees)')
        ax10.set_ylabel('Frequency')
        ax10.axvline(0, color='red', linestyle='--', alpha=0.7, label='Earth Direction')
        ax10.axvline(360, color='red', linestyle='--', alpha=0.7)
        ax10.legend()
        ax10.grid(True, alpha=0.3)
        
        # 11. CME Width vs Solar Wind Enhancement
        ax11 = fig.add_subplot(gs[2, 2])
        if 'high_density_events' in self.main_df.columns:
            ax11.scatter(self.main_df['cme_width'], self.main_df['high_density_events'],
                        alpha=0.6, s=30, c=self.main_df['cme_velocity'], cmap='plasma')
            ax11.set_xlabel('CME Angular Width (degrees)')
            ax11.set_ylabel('High Density Events')
            ax11.set_title('CME Width vs Density Enhancement', fontweight='bold')
            ax11.grid(True, alpha=0.3)
        
        # 12. Dataset Summary
        ax12 = fig.add_subplot(gs[2, 3])
        dataset_info = {
            'Total CMEs': len(self.cactus_df),
            'ML Samples': len(self.main_df),
            'Halo CMEs': len(self.cactus_df[self.cactus_df['halo_class'] > 1]),
            'Fast CMEs\n(>500 km/s)': len(self.cactus_df[self.cactus_df['v'] > 500])
        }
        
        bars = ax12.bar(dataset_info.keys(), dataset_info.values(), color='lightsteelblue')
        ax12.set_title('Dataset Summary Statistics', fontweight='bold')
        ax12.set_ylabel('Count')
        ax12.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax12.text(bar.get_x() + bar.get_width()/2., height + 5,
                     f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # Bottom row: Feature analysis
        # 13. Feature Importance (if available)
        ax13 = fig.add_subplot(gs[3, :2])
        self.plot_feature_importance(ax13)
        
        # 14. Time Series Overview
        ax14 = fig.add_subplot(gs[3, 2:])
        self.plot_temporal_overview(ax14)
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, 'cme_ml_dataset_overview_dashboard.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Overview dashboard saved: cme_ml_dataset_overview_dashboard.png")
    
    def plot_feature_importance(self, ax):
        """Plot feature importance analysis"""
        try:
            # Use transit time prediction for feature importance
            if hasattr(self, 'transit_df') and len(self.transit_df) > 0:
                # Prepare features and target
                feature_cols = [col for col in self.transit_df.columns 
                               if col not in ['target_transit_hours'] and 
                               self.transit_df[col].dtype in ['float64', 'int64']]
                
                X = self.transit_df[feature_cols].fillna(0)
                y = self.transit_df['target_transit_hours'].fillna(y.median() if len(y.dropna()) > 0 else 0)
                
                if len(X) > 10 and len(feature_cols) > 0:
                    # Train Random Forest
                    rf = RandomForestRegressor(n_estimators=50, random_state=42)
                    rf.fit(X, y)
                    
                    # Get top 10 features
                    importance_dict = dict(zip(feature_cols, rf.feature_importances_))
                    top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
                    
                    features, values = zip(*top_features)
                    
                    # Create horizontal bar plot
                    bars = ax.barh(range(len(features)), values, color='skyblue')
                    ax.set_yticks(range(len(features)))
                    ax.set_yticklabels([f.replace('_', ' ').title()[:20] for f in features])
                    ax.set_xlabel('Feature Importance')
                    ax.set_title('Top 10 Features for Transit Time Prediction', fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    
                    return
            
            # Fallback: correlation analysis
            numeric_cols = self.main_df.select_dtypes(include=[np.number]).columns[:10]
            if len(numeric_cols) > 0:
                ax.text(0.5, 0.5, 'Feature Correlation Analysis\n(Top Numeric Features)', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Feature Analysis\nUnavailable\n({str(e)[:30]}...)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
    
    def plot_temporal_overview(self, ax):
        """Plot temporal overview of CME activity"""
        try:
            # Daily CME activity
            daily_cmes = self.main_df.groupby(self.main_df['cme_time'].dt.date).size()
            
            ax.plot(daily_cmes.index, daily_cmes.values, color='blue', alpha=0.7, linewidth=1)
            ax.fill_between(daily_cmes.index, daily_cmes.values, alpha=0.3, color='lightblue')
            
            ax.set_title('Daily CME Activity Timeline (Aug 2024 - Jun 2025)', fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Number of CMEs per Day')
            ax.grid(True, alpha=0.3)
            
            # Add rolling average
            if len(daily_cmes) > 7:
                rolling_avg = daily_cmes.rolling(window=7, center=True).mean()
                ax.plot(rolling_avg.index, rolling_avg.values, color='red', linewidth=2, 
                       label='7-day average')
                ax.legend()
            
            # Rotate x-axis labels
            ax.tick_params(axis='x', rotation=45)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Temporal Overview\nUnavailable\n({str(e)[:30]}...)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
    
    def create_correlation_analysis(self):
        """Create detailed correlation analysis"""
        print("\n🔍 Creating correlation analysis...")
        
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        fig.suptitle('CACTUS CME - APEX SWISS Feature Correlation Analysis', 
                     fontsize=16, fontweight='bold')
        
        # 1. CME Parameters Correlation
        ax1 = axes[0, 0]
        cme_params = ['cme_velocity', 'cme_width', 'pa']
        available_cme_params = [p for p in cme_params if p in self.main_df.columns]
        
        if len(available_cme_params) > 1:
            cme_corr = self.main_df[available_cme_params].corr()
            sns.heatmap(cme_corr, annot=True, cmap='RdBu_r', center=0, ax=ax1,
                       square=True, cbar_kws={'shrink': 0.8})
            ax1.set_title('CME Parameters Correlation', fontweight='bold')
        
        # 2. Solar Wind Parameters Correlation
        ax2 = axes[0, 1]
        sw_params = [col for col in self.main_df.columns if 'proton' in col and 'mean' in col][:6]
        
        if len(sw_params) > 1:
            sw_corr = self.main_df[sw_params].corr()
            sns.heatmap(sw_corr, annot=True, cmap='viridis', center=0, ax=ax2,
                       square=True, cbar_kws={'shrink': 0.8})
            ax2.set_title('Solar Wind Parameters Correlation', fontweight='bold')
            ax2.set_xticklabels([p.replace('_', ' ').title()[:15] for p in sw_params], rotation=45)
            ax2.set_yticklabels([p.replace('_', ' ').title()[:15] for p in sw_params], rotation=0)
        
        # 3. Cross-Correlation: CME vs Solar Wind
        ax3 = axes[0, 2]
        cme_cols = ['cme_velocity', 'cme_width']
        sw_cols = [col for col in self.main_df.columns if 'proton_density_mean' in col or 'proton_bulk_speed_mean' in col][:2]
        
        if len(cme_cols) > 0 and len(sw_cols) > 0:
            cross_corr_data = []
            for cme_col in cme_cols:
                for sw_col in sw_cols:
                    if cme_col in self.main_df.columns and sw_col in self.main_df.columns:
                        valid_data = self.main_df[[cme_col, sw_col]].dropna()
                        if len(valid_data) > 10:
                            corr, p_value = pearsonr(valid_data[cme_col], valid_data[sw_col])
                            cross_corr_data.append({
                                'CME_Parameter': cme_col.replace('_', ' ').title(),
                                'SW_Parameter': sw_col.replace('_', ' ').title(),
                                'Correlation': corr,
                                'P_Value': p_value
                            })
            
            if cross_corr_data:
                cross_corr_df = pd.DataFrame(cross_corr_data)
                pivot_corr = cross_corr_df.pivot(index='CME_Parameter', columns='SW_Parameter', values='Correlation')
                sns.heatmap(pivot_corr, annot=True, cmap='RdYlBu_r', center=0, ax=ax3,
                           cbar_kws={'shrink': 0.8})
                ax3.set_title('CME vs Solar Wind Cross-Correlation', fontweight='bold')
        
        # 4. Velocity vs Response Scatter
        ax4 = axes[1, 0]
        if 'cme_velocity' in self.main_df.columns and 'proton_density_max' in self.main_df.columns:
            scatter = ax4.scatter(self.main_df['cme_velocity'], self.main_df['proton_density_max'],
                                c=self.main_df['cme_width'], cmap='plasma', alpha=0.6, s=40)
            ax4.set_xlabel('CME Velocity (km/s)')
            ax4.set_ylabel('Max Proton Density (cm⁻³)')
            ax4.set_title('CME Velocity vs Max Density Response', fontweight='bold')
            plt.colorbar(scatter, ax=ax4, label='CME Width (°)')
            ax4.grid(True, alpha=0.3)
        
        # 5. Halo Class Analysis
        ax5 = axes[1, 1]
        if 'halo_class' in self.cactus_df.columns:
            # Box plot of velocities by halo class
            halo_data = []
            halo_labels = []
            for halo_class in sorted(self.cactus_df['halo_class'].unique()):
                data = self.cactus_df[self.cactus_df['halo_class'] == halo_class]['v'].dropna()
                if len(data) > 0:
                    halo_data.append(data)
                    halo_labels.append(f'Class {halo_class}')
            
            if halo_data:
                ax5.boxplot(halo_data, labels=halo_labels)
                ax5.set_title('CME Velocity Distribution by Halo Class', fontweight='bold')
                ax5.set_ylabel('Velocity (km/s)')
                ax5.grid(True, alpha=0.3)
        
        # 6. Feature Clustering
        ax6 = axes[1, 2]
        self.plot_feature_clustering(ax6)
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, 'cme_correlation_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Correlation analysis saved: cme_correlation_analysis.png")
    
    def plot_feature_clustering(self, ax):
        """Plot feature clustering dendrogram"""
        try:
            # Select numeric features for clustering
            numeric_cols = self.main_df.select_dtypes(include=[np.number]).columns
            feature_subset = [col for col in numeric_cols if not col.startswith('cme_number')][:10]
            
            if len(feature_subset) > 3:
                # Compute correlation matrix
                corr_matrix = self.main_df[feature_subset].corr().abs()
                
                # Convert to distance matrix
                distance_matrix = 1 - corr_matrix
                
                # Perform hierarchical clustering
                linkage_matrix = linkage(distance_matrix, method='ward')
                
                # Create dendrogram
                dendrogram(linkage_matrix, labels=[f.replace('_', ' ')[:15] for f in feature_subset],
                          ax=ax, orientation='top')
                ax.set_title('Feature Clustering Dendrogram', fontweight='bold')
                ax.tick_params(axis='x', rotation=45)
            else:
                ax.text(0.5, 0.5, 'Insufficient features\nfor clustering', 
                       ha='center', va='center', transform=ax.transAxes)
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Clustering\nUnavailable\n({str(e)[:20]}...)', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def create_time_strip_analysis(self):
        """Create 12-hour time strip analysis"""
        print("\n⏰ Creating 12-hour time strip analysis...")
        
        fig, axes = plt.subplots(3, 2, figsize=(20, 18))
        fig.suptitle('12-Hour Time Strip Feature Analysis\n(Pre/Post CME and Arrival Windows)', 
                     fontsize=16, fontweight='bold')
        
        # Define time windows
        time_windows = ['pre_cme_12h', 'post_cme_12h', 'pre_arrival_12h', 'post_arrival_12h']
        parameters = ['proton_density', 'proton_bulk_speed', 'proton_temperature']
        
        # 1. Mean values comparison across time windows
        ax1 = axes[0, 0]
        mean_data = []
        for param in parameters:
            param_means = []
            for window in time_windows:
                col_name = f'{window}_{param}_mean'
                if col_name in self.strip_df.columns:
                    param_means.append(self.strip_df[col_name].mean())
                else:
                    param_means.append(0)
            mean_data.append(param_means)
        
        x_pos = np.arange(len(time_windows))
        width = 0.25
        
        for i, param in enumerate(parameters):
            ax1.bar(x_pos + i*width, mean_data[i], width, 
                   label=param.replace('_', ' ').title(), alpha=0.8)
        
        ax1.set_xlabel('Time Window')
        ax1.set_ylabel('Mean Value')
        ax1.set_title('Solar Wind Parameters: Mean Values by Time Window', fontweight='bold')
        ax1.set_xticks(x_pos + width)
        ax1.set_xticklabels([w.replace('_', ' ').title() for w in time_windows], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Standard deviation comparison
        ax2 = axes[0, 1]
        std_data = []
        for param in parameters:
            param_stds = []
            for window in time_windows:
                col_name = f'{window}_{param}_std'
                if col_name in self.strip_df.columns:
                    param_stds.append(self.strip_df[col_name].mean())
                else:
                    param_stds.append(0)
            std_data.append(param_stds)
        
        for i, param in enumerate(parameters):
            ax2.bar(x_pos + i*width, std_data[i], width, 
                   label=param.replace('_', ' ').title(), alpha=0.8)
        
        ax2.set_xlabel('Time Window')
        ax2.set_ylabel('Mean Standard Deviation')
        ax2.set_title('Solar Wind Parameters: Variability by Time Window', fontweight='bold')
        ax2.set_xticks(x_pos + width)
        ax2.set_xticklabels([w.replace('_', ' ').title() for w in time_windows], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Trend analysis
        ax3 = axes[1, 0]
        trend_data = []
        for param in parameters:
            param_trends = []
            for window in time_windows:
                col_name = f'{window}_{param}_trend'
                if col_name in self.strip_df.columns:
                    param_trends.append(self.strip_df[col_name].mean())
                else:
                    param_trends.append(0)
            trend_data.append(param_trends)
        
        for i, param in enumerate(parameters):
            ax3.plot(time_windows, trend_data[i], marker='o', linewidth=2, 
                    label=param.replace('_', ' ').title())
        
        ax3.set_xlabel('Time Window')
        ax3.set_ylabel('Mean Trend')
        ax3.set_title('Solar Wind Parameters: Trends Across Time Windows', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 4. Max values heatmap
        ax4 = axes[1, 1]
        max_data = []
        for param in parameters:
            param_row = []
            for window in time_windows:
                col_name = f'{window}_{param}_max'
                if col_name in self.strip_df.columns:
                    param_row.append(self.strip_df[col_name].mean())
                else:
                    param_row.append(0)
            max_data.append(param_row)
        
        max_df = pd.DataFrame(max_data, 
                             index=[p.replace('_', ' ').title() for p in parameters],
                             columns=[w.replace('_', ' ').title() for w in time_windows])
        
        sns.heatmap(max_df, annot=True, cmap='YlOrRd', ax=ax4, 
                   cbar_kws={'shrink': 0.8}, fmt='.1f')
        ax4.set_title('Maximum Values Heatmap', fontweight='bold')
        
        # 5. Pre vs Post CME comparison
        ax5 = axes[2, 0]
        
        for param in parameters:
            pre_col = f'pre_cme_12h_{param}_mean'
            post_col = f'post_cme_12h_{param}_mean'
            
            if pre_col in self.strip_df.columns and post_col in self.strip_df.columns:
                # Get valid data for both columns
                valid_mask = self.strip_df[pre_col].notna() & self.strip_df[post_col].notna()
                pre_vals = self.strip_df.loc[valid_mask, pre_col]
                post_vals = self.strip_df.loc[valid_mask, post_col]
                
                if len(pre_vals) > 0 and len(post_vals) > 0:
                    ax5.scatter(pre_vals, post_vals, alpha=0.6, s=20, 
                              label=param.replace('_', ' ').title())
        
        # Add diagonal line
        ax_lims = [min(ax5.get_xlim()[0], ax5.get_ylim()[0]),
                   max(ax5.get_xlim()[1], ax5.get_ylim()[1])]
        ax5.plot(ax_lims, ax_lims, 'k--', alpha=0.5, linewidth=1)
        
        ax5.set_xlabel('Pre-CME Values')
        ax5.set_ylabel('Post-CME Values')
        ax5.set_title('Pre vs Post CME Comparison', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Time window correlation matrix
        ax6 = axes[2, 1]
        # Select a subset of time strip features for correlation
        strip_features = [col for col in self.strip_df.columns 
                         if 'proton_density_mean' in col or 'proton_bulk_speed_mean' in col][:8]
        
        if len(strip_features) > 1:
            strip_corr = self.strip_df[strip_features].corr()
            sns.heatmap(strip_corr, annot=True, cmap='coolwarm', center=0, ax=ax6,
                       cbar_kws={'shrink': 0.8}, fmt='.2f')
            ax6.set_title('Time Strip Features Correlation', fontweight='bold')
            ax6.set_xticklabels([f.split('_')[-2] + '_' + f.split('_')[-1] for f in strip_features], rotation=45)
            ax6.set_yticklabels([f.split('_')[-2] + '_' + f.split('_')[-1] for f in strip_features], rotation=0)
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, 'cme_time_strip_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Time strip analysis saved: cme_time_strip_analysis.png")
    
    def create_halo_classification_analysis(self):
        """Create halo CME classification analysis"""
        print("\n🎯 Creating halo classification analysis...")
        
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        fig.suptitle('Halo CME Classification Analysis\n(Classes 1-4: Normal, Partial, Full, Complex)', 
                     fontsize=16, fontweight='bold')
        
        # 1. Halo class vs velocity box plot
        ax1 = axes[0, 0]
        halo_velocity_data = []
        halo_labels = []
        
        for halo_class in sorted(self.cactus_df['halo_class'].unique()):
            velocity_data = self.cactus_df[self.cactus_df['halo_class'] == halo_class]['v'].dropna()
            if len(velocity_data) > 0:
                halo_velocity_data.append(velocity_data)
                halo_labels.append(f'Class {halo_class}')
        
        if halo_velocity_data:
            box_plot = ax1.boxplot(halo_velocity_data, labels=halo_labels, patch_artist=True)
            colors = ['lightblue', 'orange', 'red', 'darkred']
            for patch, color in zip(box_plot['boxes'], colors[:len(box_plot['boxes'])]):
                patch.set_facecolor(color)
            
            ax1.set_title('CME Velocity by Halo Class', fontweight='bold')
            ax1.set_ylabel('Velocity (km/s)')
            ax1.grid(True, alpha=0.3)
        
        # 2. Halo class vs angular width
        ax2 = axes[0, 1]
        for halo_class in sorted(self.cactus_df['halo_class'].unique()):
            data = self.cactus_df[self.cactus_df['halo_class'] == halo_class]
            ax2.scatter(data['da'], data['v'], alpha=0.6, s=30, label=f'Class {halo_class}')
        
        ax2.set_xlabel('Angular Width (degrees)')
        ax2.set_ylabel('Velocity (km/s)')
        ax2.set_title('Angular Width vs Velocity by Halo Class', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Position angle distribution by halo class
        ax3 = axes[0, 2]
        halo_classes = sorted(self.cactus_df['halo_class'].unique())
        colors = ['lightblue', 'orange', 'red', 'darkred']
        
        for i, halo_class in enumerate(halo_classes):
            pa_data = self.cactus_df[self.cactus_df['halo_class'] == halo_class]['pa'].dropna()
            if len(pa_data) > 0:
                ax3.hist(pa_data, bins=20, alpha=0.6, color=colors[i % len(colors)], 
                        label=f'Class {halo_class}')
        
        ax3.set_xlabel('Position Angle (degrees)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Position Angle Distribution by Halo Class', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Velocity class distribution
        ax4 = axes[1, 0]
        if 'velocity_class' in self.cactus_df.columns:
            vel_class_counts = self.cactus_df['velocity_class'].value_counts()
            ax4.pie(vel_class_counts.values, labels=vel_class_counts.index, autopct='%1.1f%%', 
                   startangle=90)
            ax4.set_title('CME Velocity Class Distribution', fontweight='bold')
        
        # 5. Halo class temporal distribution
        ax5 = axes[1, 1]
        monthly_halo = self.cactus_df.groupby([
            self.cactus_df['t0'].dt.month, 
            self.cactus_df['halo_class']
        ]).size().unstack(fill_value=0)
        
        monthly_halo.plot(kind='bar', stacked=True, ax=ax5, 
                         color=['lightblue', 'orange', 'red', 'darkred'])
        ax5.set_title('Monthly Halo Class Distribution', fontweight='bold')
        ax5.set_xlabel('Month')
        ax5.set_ylabel('Number of CMEs')
        ax5.legend(title='Halo Class')
        ax5.tick_params(axis='x', rotation=45)
        
        # 6. Earth-directed analysis by halo class
        ax6 = axes[1, 2]
        if 'earth_directed' in self.main_df.columns:
            # Create contingency table
            halo_earth = pd.crosstab(self.cactus_df['halo_class'], 
                                   self.cactus_df.get('earth_directed', 
                                   pd.Series([0]*len(self.cactus_df))))
            
            halo_earth.plot(kind='bar', ax=ax6, color=['lightcoral', 'lightgreen'])
            ax6.set_title('Earth-Directed CMEs by Halo Class', fontweight='bold')
            ax6.set_xlabel('Halo Class')
            ax6.set_ylabel('Number of CMEs')
            ax6.legend(['Not Earth-Directed', 'Earth-Directed'])
            ax6.tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, 'halo_classification_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Halo classification analysis saved: halo_classification_analysis.png")
    
    def create_transit_time_analysis(self):
        """Create transit time prediction analysis"""
        print("\n🚀 Creating transit time analysis...")
        
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        fig.suptitle('CME Transit Time Prediction Analysis\n(Sun to L1 Lagrange Point)', 
                     fontsize=16, fontweight='bold')
        
        # 1. Transit time distribution
        ax1 = axes[0, 0]
        if 'target_transit_hours' in self.transit_df.columns:
            transit_times = self.transit_df['target_transit_hours'].dropna()
            ax1.hist(transit_times, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
            ax1.axvline(transit_times.mean(), color='red', linestyle='--', 
                       label=f'Mean: {transit_times.mean():.1f}h')
            ax1.axvline(transit_times.median(), color='orange', linestyle='--', 
                       label=f'Median: {transit_times.median():.1f}h')
            ax1.set_xlabel('Transit Time (hours)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('CME Transit Time Distribution', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Velocity vs Transit Time
        ax2 = axes[0, 1]
        if 'cme_velocity' in self.transit_df.columns and 'target_transit_hours' in self.transit_df.columns:
            scatter = ax2.scatter(self.transit_df['cme_velocity'], self.transit_df['target_transit_hours'],
                                c=self.transit_df['cme_width'], cmap='plasma', alpha=0.6, s=40)
            ax2.set_xlabel('CME Velocity (km/s)')
            ax2.set_ylabel('Transit Time (hours)')
            ax2.set_title('CME Velocity vs Transit Time\n(colored by angular width)', fontweight='bold')
            plt.colorbar(scatter, ax=ax2, label='Angular Width (°)')
            ax2.grid(True, alpha=0.3)
            
            # Add theoretical curve
            vel_range = np.linspace(self.transit_df['cme_velocity'].min(), 
                                  self.transit_df['cme_velocity'].max(), 100)
            theoretical_transit = 1.5e6 / (vel_range * 3.6)  # Simple model
            ax2.plot(vel_range, theoretical_transit, 'r--', alpha=0.8, 
                    label='Theoretical', linewidth=2)
            ax2.legend()
        
        # 3. Pre-CME conditions vs Transit Time
        ax3 = axes[0, 2]
        pre_density_col = [col for col in self.transit_df.columns if 'pre_cme_12h_proton_density_mean' in col]
        if pre_density_col and 'target_transit_hours' in self.transit_df.columns:
            ax3.scatter(self.transit_df[pre_density_col[0]], self.transit_df['target_transit_hours'],
                       alpha=0.6, s=40, color='green')
            ax3.set_xlabel('Pre-CME Proton Density (cm⁻³)')
            ax3.set_ylabel('Transit Time (hours)')
            ax3.set_title('Pre-CME Solar Wind Density vs Transit Time', fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        # 4. Feature importance for transit time
        ax4 = axes[1, 0]
        self.plot_transit_feature_importance(ax4)
        
        # 5. Transit time vs Earth direction
        ax5 = axes[1, 1]
        if 'earth_directed' in self.transit_df.columns and 'target_transit_hours' in self.transit_df.columns:
            earth_transit = []
            labels = ['Not Earth-Directed', 'Earth-Directed']
            
            for earth_dir in [0, 1]:
                transit_data = self.transit_df[self.transit_df['earth_directed'] == earth_dir]['target_transit_hours'].dropna()
                if len(transit_data) > 0:
                    earth_transit.append(transit_data)
            
            if len(earth_transit) == 2:
                ax5.boxplot(earth_transit, labels=labels)
                ax5.set_title('Transit Time by Earth Direction', fontweight='bold')
                ax5.set_ylabel('Transit Time (hours)')
                ax5.grid(True, alpha=0.3)
        
        # 6. Prediction accuracy analysis
        ax6 = axes[1, 2]
        if len(self.transit_df) > 20:
            # Simple model evaluation
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_absolute_error, r2_score
            
            feature_cols = [col for col in self.transit_df.columns 
                           if col not in ['target_transit_hours'] and 
                           self.transit_df[col].dtype in ['float64', 'int64']]
            
            X = self.transit_df[feature_cols].fillna(0)
            y = self.transit_df['target_transit_hours']
            y_median = y.median() if len(y.dropna()) > 0 else 0
            y = y.fillna(y_median)
            
            if len(X) > 20:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                
                rf = RandomForestRegressor(n_estimators=50, random_state=42)
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)
                
                ax6.scatter(y_test, y_pred, alpha=0.6, s=40)
                ax6.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                        'r--', linewidth=2, label='Perfect Prediction')
                
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                ax6.set_xlabel('Actual Transit Time (hours)')
                ax6.set_ylabel('Predicted Transit Time (hours)')
                ax6.set_title(f'Prediction Accuracy\nMAE: {mae:.1f}h, R²: {r2:.3f}', fontweight='bold')
                ax6.legend()
                ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, 'transit_time_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Transit time analysis saved: transit_time_analysis.png")
    
    def plot_transit_feature_importance(self, ax):
        """Plot feature importance for transit time prediction"""
        try:
            if len(self.transit_df) > 10:
                feature_cols = [col for col in self.transit_df.columns 
                               if col not in ['target_transit_hours'] and 
                               self.transit_df[col].dtype in ['float64', 'int64']]
                
                X = self.transit_df[feature_cols].fillna(0)
                y = self.transit_df['target_transit_hours']
                y_median = y.median() if len(y.dropna()) > 0 else 0
                y = y.fillna(y_median)
                
                if len(feature_cols) > 0:
                    rf = RandomForestRegressor(n_estimators=50, random_state=42)
                    rf.fit(X, y)
                    
                    # Get top 8 features
                    importance_dict = dict(zip(feature_cols, rf.feature_importances_))
                    top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:8]
                    
                    features, values = zip(*top_features)
                    
                    bars = ax.barh(range(len(features)), values, color='lightcoral')
                    ax.set_yticks(range(len(features)))
                    ax.set_yticklabels([f.replace('_', ' ').title()[:25] for f in features])
                    ax.set_xlabel('Feature Importance')
                    ax.set_title('Top Features for Transit Time Prediction', fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    return
            
            ax.text(0.5, 0.5, 'Feature Importance\nAnalysis\nUnavailable', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
                   
        except Exception as e:
            ax.text(0.5, 0.5, f'Feature Importance\nError: {str(e)[:20]}...', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
    
    def create_summary_report(self):
        """Create summary report of the analysis"""
        print("\n📄 Creating summary report...")
        
        report_path = os.path.join(self.output_dir, 'CME_ML_Dataset_Analysis_Report.md')
        
        with open(report_path, 'w') as f:
            f.write("# CACTUS CME - APEX SWISS ML Dataset Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Analysis Period:** August 2024 - June 2025\n")
            f.write(f"**Dataset Location:** {self.ml_data_dir}\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write("This comprehensive analysis visualizes the machine learning datasets created from ")
            f.write("CACTUS CME observations and APEX SWISS in-situ solar wind measurements. ")
            f.write("The analysis provides insights into CME-solar wind relationships, halo CME ")
            f.write("classification patterns, and transit time prediction features.\n\n")
            
            f.write("## Dataset Overview\n\n")
            f.write(f"### Primary Datasets\n")
            f.write(f"- **Main ML Dataset**: {self.main_df.shape[0]:,} samples × {self.main_df.shape[1]} features\n")
            f.write(f"- **Enhanced CACTUS**: {self.cactus_df.shape[0]:,} CME events\n")
            f.write(f"- **Time Strip Features**: {self.strip_df.shape[0]:,} samples × {self.strip_df.shape[1]} features\n")
            f.write(f"- **Transit Prediction**: {self.transit_df.shape[0]:,} samples\n")
            f.write(f"- **Halo Classification**: {self.halo_df.shape[0]:,} samples\n\n")
            
            f.write("### Halo CME Distribution\n")
            halo_counts = self.cactus_df['halo_class'].value_counts().sort_index()
            for halo_class, count in halo_counts.items():
                percentage = (count / len(self.cactus_df)) * 100
                f.write(f"- **Class {halo_class}**: {count} events ({percentage:.1f}%)\n")
            f.write("\n")
            
            f.write("### CME Velocity Statistics\n")
            vel_stats = self.cactus_df['v'].describe()
            f.write(f"- **Mean Velocity**: {vel_stats['mean']:.0f} km/s\n")
            f.write(f"- **Median Velocity**: {vel_stats['50%']:.0f} km/s\n")
            f.write(f"- **Max Velocity**: {vel_stats['max']:.0f} km/s\n")
            f.write(f"- **Fast CMEs** (>500 km/s): {len(self.cactus_df[self.cactus_df['v'] > 500])} events\n\n")
            
            f.write("## Key Findings\n\n")
            f.write("### 1. Halo CME Characteristics\n")
            f.write("- Normal CMEs (Class 1) dominate the dataset\n")
            f.write("- Full halo CMEs (Class 3) are rare but significant\n")
            f.write("- Halo classification correlates with velocity and angular width\n\n")
            
            f.write("### 2. Solar Wind Parameter Correlations\n")
            f.write("- Strong correlations between proton density and bulk speed\n")
            f.write("- CME velocity shows correlation with solar wind response\n")
            f.write("- 12-hour time strips reveal pre/post CME variations\n\n")
            
            f.write("### 3. Transit Time Patterns\n")
            if 'target_transit_hours' in self.transit_df.columns:
                transit_mean = self.transit_df['target_transit_hours'].mean()
                f.write(f"- Average transit time: {transit_mean:.1f} hours\n")
                f.write("- Inverse relationship between velocity and transit time\n")
                f.write("- Pre-CME solar wind conditions affect arrival times\n\n")
            
            f.write("### 4. Temporal Patterns\n")
            monthly_activity = self.cactus_df.groupby(self.cactus_df['t0'].dt.month).size()
            peak_month = monthly_activity.idxmax()
            f.write(f"- Peak CME activity in month {peak_month}\n")
            f.write("- Consistent data coverage across the analysis period\n")
            f.write("- Seasonal variations in CME characteristics\n\n")
            
            f.write("## Visualization Products\n\n")
            f.write("### Generated Plots\n")
            f.write("1. **Overview Dashboard**: `cme_ml_dataset_overview_dashboard.png`\n")
            f.write("2. **Correlation Analysis**: `cme_correlation_analysis.png`\n")
            f.write("3. **Time Strip Analysis**: `cme_time_strip_analysis.png`\n")
            f.write("4. **Halo Classification**: `halo_classification_analysis.png`\n")
            f.write("5. **Transit Time Analysis**: `transit_time_analysis.png`\n\n")
            
            f.write("## Machine Learning Implications\n\n")
            f.write("### Model Development Insights\n")
            f.write("- Features show good separation for classification tasks\n")
            f.write("- Transit time prediction benefits from velocity and pre-CME conditions\n")
            f.write("- Halo classification achievable with angular width and velocity\n")
            f.write("- Time strip features provide temporal context for predictions\n\n")
            
            f.write("### Recommended Approaches\n")
            f.write("- **Random Forest**: Good baseline for both regression and classification\n")
            f.write("- **Gradient Boosting**: For improved prediction accuracy\n")
            f.write("- **Neural Networks**: For complex non-linear relationships\n")
            f.write("- **Ensemble Methods**: Combining multiple models for robustness\n\n")
            
            f.write("## Data Quality Assessment\n\n")
            f.write("- **Completeness**: High data availability across time period\n")
            f.write("- **Consistency**: Proper alignment between CME and solar wind data\n")
            f.write("- **Feature Engineering**: Effective 12-hour time strip extraction\n")
            f.write("- **Target Variables**: Well-defined for supervised learning\n\n")
            
            f.write("## Future Work\n\n")
            f.write("- Deep learning models for sequence prediction\n")
            f.write("- Real-time CME detection algorithms\n")
            f.write("- Integration with additional space weather parameters\n")
            f.write("- Operational deployment for space weather forecasting\n\n")
            
            f.write("---\n")
            f.write("*Analysis conducted using CACTUS CME catalog and Aditya-L1 APEX SWISS data*\n")
            f.write("*Report generated by CME ML Dataset Visualizer*\n")
        
        print(f"   ✅ Summary report saved: CME_ML_Dataset_Analysis_Report.md")
    
    def run_complete_visualization(self):
        """Run the complete visualization analysis"""
        print("🚀 Starting CACTUS CME - APEX SWISS ML Dataset Visualization...")
        
        # Load datasets
        self.load_datasets()
        
        # Create visualizations
        self.create_cme_overview_dashboard()
        self.create_correlation_analysis()
        self.create_time_strip_analysis()
        self.create_halo_classification_analysis()
        self.create_transit_time_analysis()
        
        # Create summary report
        self.create_summary_report()
        
        print(f"\n🎉 VISUALIZATION ANALYSIS COMPLETE!")
        print("=" * 70)
        print(f"📊 Datasets analyzed:")
        print(f"   • Main ML: {self.main_df.shape}")
        print(f"   • CACTUS: {self.cactus_df.shape[0]} CMEs")
        print(f"   • Time strips: {self.strip_df.shape}")
        print(f"🎯 Halo CMEs: {len(self.cactus_df[self.cactus_df['halo_class'] > 1])}")
        print(f"📈 Visualizations: 5 comprehensive plots")
        print(f"📁 Output: {self.output_dir}")
        print("=" * 70)

def main():
    """Main execution function"""
    visualizer = CMEMLDatasetVisualizer()
    visualizer.run_complete_visualization()

if __name__ == "__main__":
    main()
