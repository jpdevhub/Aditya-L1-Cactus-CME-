#!/usr/bin/env python3
"""
Comprehensive Multi-Dataset Integration System for ISRO Data Analysis
Integrates APEX SWISS solar wind data, CACTUS CME data, and MAG data for advanced ML analysis

Author: AI Analysis System
Date: 2025-01-07
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import xarray as xr
import os
import warnings
warnings.filterwarnings('ignore')

class ISRODataIntegrator:
    """
    Comprehensive data integration system for ISRO data analysis
    """
    
    def __init__(self, base_dir="/Volumes/T7/ISRO"):
        self.base_dir = base_dir
        self.datasets = {}
        self.integrated_data = None
        
        # Data paths
        self.paths = {
            'swiss_data': f"{base_dir}/APEX_SWISS_BLK_FILES/cme_feature_alignment_analysis/processed_data/alpha_proton_cme_merged_data.csv",
            'cme_data': f"{base_dir}/CME(CACTUS)/CACTUS_CME_Combined_Aug2024_Jun2025.csv",
            'mag_data_dir': f"{base_dir}/APEX_SWISS_MAG_FILES/data",
            'output_dir': f"{base_dir}/APEX_SWISS_BLK_FILES/integrated_analysis"
        }
        
        # Create output directory
        os.makedirs(self.paths['output_dir'], exist_ok=True)
        os.makedirs(f"{self.paths['output_dir']}/plots", exist_ok=True)
        os.makedirs(f"{self.paths['output_dir']}/data", exist_ok=True)
        os.makedirs(f"{self.paths['output_dir']}/reports", exist_ok=True)
        
        print("🚀 ISRO Data Integration System Initialized")
        print(f"📂 Output directory: {self.paths['output_dir']}")
    
    def load_solar_wind_data(self):
        """Load and process APEX SWISS solar wind data"""
        try:
            print("\n📊 Loading APEX SWISS Solar Wind Data...")
            
            if os.path.exists(self.paths['swiss_data']):
                df = pd.read_csv(self.paths['swiss_data'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Extract key features
                solar_wind_features = [
                    'proton_density', 'proton_bulk_speed', 'alpha_density', 
                    'proton_temperature', 'alpha_proton_ratio'
                ]
                
                # Clean and process data
                for feature in solar_wind_features:
                    if feature in df.columns:
                        df[feature] = pd.to_numeric(df[feature], errors='coerce')
                
                self.datasets['solar_wind'] = df
                print(f"   ✅ Loaded {len(df)} solar wind data points")
                print(f"   📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                return True
                
            else:
                print(f"   ❌ Solar wind data file not found: {self.paths['swiss_data']}")
                return False
                
        except Exception as e:
            print(f"   ❌ Error loading solar wind data: {e}")
            return False
    
    def load_cme_data(self):
        """Load and process CACTUS CME data"""
        try:
            print("\n🌪️ Loading CACTUS CME Data...")
            
            if os.path.exists(self.paths['cme_data']):
                df = pd.read_csv(self.paths['cme_data'])
                
                # Clean timestamp
                df['timestamp'] = pd.to_datetime(df['t0'], errors='coerce')
                df = df.dropna(subset=['timestamp'])
                
                # Extract key CME features
                cme_features = ['width', 'v', 'v_error', 'accel', 'accel_error']
                for feature in cme_features:
                    if feature in df.columns:
                        df[feature] = pd.to_numeric(df[feature], errors='coerce')
                
                # Add CME classification
                df['cme_speed_class'] = df['v'].apply(self._classify_cme_speed)
                
                self.datasets['cme'] = df
                print(f"   ✅ Loaded {len(df)} CME events")
                print(f"   📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                return True
                
            else:
                print(f"   ❌ CME data file not found: {self.paths['cme_data']}")
                return False
                
        except Exception as e:
            print(f"   ❌ Error loading CME data: {e}")
            return False
    
    def scan_mag_data(self):
        """Scan for available MAG data files"""
        try:
            print("\n🧲 Scanning MAG Data Files...")
            
            mag_files = []
            mag_folders = [
                f"{self.base_dir}/APEX_SWISS_MAG_FILES/APEX_SWISS_MAG_BLK_FILES/data",
                f"{self.base_dir}/APEX_SWISS_MAG_FILES/APEX_SWISS_MAG_TH1_FILES/data",
                f"{self.base_dir}/APEX_SWISS_MAG_FILES/APEX_SWISS_MAG_TH2_FILES/data"
            ]
            
            for folder in mag_folders:
                if os.path.exists(folder):
                    for root, dirs, files in os.walk(folder):
                        for file in files:
                            if file.endswith('.nc') and 'MAG' in file:
                                mag_files.append(os.path.join(root, file))
            
            self.datasets['mag_files'] = mag_files
            print(f"   📁 Found {len(mag_files)} MAG data files")
            
            if len(mag_files) > 0:
                print("   📂 Sample files:")
                for i, file in enumerate(mag_files[:3]):
                    print(f"      {i+1}. {os.path.basename(file)}")
                if len(mag_files) > 3:
                    print(f"      ... and {len(mag_files)-3} more files")
            
            return len(mag_files) > 0
            
        except Exception as e:
            print(f"   ❌ Error scanning MAG data: {e}")
            return False
    
    def load_sample_mag_data(self, max_files=5):
        """Load a sample of MAG data for analysis"""
        try:
            print(f"\n🔬 Loading Sample MAG Data (max {max_files} files)...")
            
            if 'mag_files' not in self.datasets or len(self.datasets['mag_files']) == 0:
                print("   ⚠️ No MAG files available")
                return False
            
            mag_data_list = []
            loaded_count = 0
            
            for file_path in self.datasets['mag_files'][:max_files]:
                try:
                    # Load NetCDF file
                    ds = xr.open_dataset(file_path)
                    
                    # Extract timestamp and magnetic field components
                    if 'time' in ds.coords:
                        timestamps = pd.to_datetime(ds.time.values)
                        
                        # Extract magnetic field components (typical names)
                        mag_components = {}
                        possible_names = {
                            'Bx': ['Bx', 'B_x', 'mag_x', 'bx'],
                            'By': ['By', 'B_y', 'mag_y', 'by'],
                            'Bz': ['Bz', 'B_z', 'mag_z', 'bz'],
                            'Btotal': ['Btotal', 'B_total', 'mag_total', 'btotal']
                        }
                        
                        for component, names in possible_names.items():
                            for name in names:
                                if name in ds.data_vars:
                                    mag_components[component] = ds[name].values
                                    break
                        
                        if len(mag_components) > 0:
                            # Create DataFrame for this file
                            df = pd.DataFrame({'timestamp': timestamps})
                            for comp, values in mag_components.items():
                                df[f'mag_{comp.lower()}'] = values
                            
                            df['file_source'] = os.path.basename(file_path)
                            mag_data_list.append(df)
                            loaded_count += 1
                    
                    ds.close()
                    
                except Exception as file_error:
                    print(f"      ⚠️ Error loading {os.path.basename(file_path)}: {file_error}")
                    continue
            
            if mag_data_list:
                combined_mag = pd.concat(mag_data_list, ignore_index=True)
                combined_mag = combined_mag.sort_values('timestamp').reset_index(drop=True)
                
                self.datasets['mag'] = combined_mag
                print(f"   ✅ Loaded {loaded_count} MAG files")
                print(f"   📊 Combined data points: {len(combined_mag)}")
                print(f"   📅 Date range: {combined_mag['timestamp'].min()} to {combined_mag['timestamp'].max()}")
                
                # Display available magnetic field components
                mag_cols = [col for col in combined_mag.columns if col.startswith('mag_')]
                print(f"   🧲 Available components: {', '.join(mag_cols)}")
                
                return True
            else:
                print("   ❌ No valid MAG data could be loaded")
                return False
                
        except Exception as e:
            print(f"   ❌ Error loading MAG data: {e}")
            return False
    
    def create_time_aligned_dataset(self, time_window_minutes=60):
        """Create time-aligned integrated dataset"""
        try:
            print(f"\n🎯 Creating Time-Aligned Integrated Dataset (±{time_window_minutes} min window)...")
            
            available_datasets = [name for name in ['solar_wind', 'cme', 'mag'] if name in self.datasets]
            print(f"   📊 Available datasets: {', '.join(available_datasets)}")
            
            if 'solar_wind' not in self.datasets:
                print("   ❌ Solar wind data is required as the base dataset")
                return False
            
            # Start with solar wind data as base
            base_data = self.datasets['solar_wind'].copy()
            print(f"   📈 Base dataset (solar wind): {len(base_data)} points")
            
            # Add CME features if available
            if 'cme' in self.datasets:
                print("   🌪️ Adding CME features...")
                base_data = self._add_cme_features(base_data, time_window_minutes)
            
            # Add MAG features if available
            if 'mag' in self.datasets:
                print("   🧲 Adding MAG features...")
                base_data = self._add_mag_features(base_data, time_window_minutes)
            
            # Clean and prepare final dataset
            base_data = base_data.sort_values('timestamp').reset_index(drop=True)
            
            # Remove rows with all NaN values in feature columns
            feature_cols = [col for col in base_data.columns if col not in ['timestamp', 'file_source']]
            base_data = base_data.dropna(subset=feature_cols, how='all')
            
            self.integrated_data = base_data
            
            print(f"   ✅ Integrated dataset created: {len(base_data)} data points")
            print(f"   📊 Features: {len(feature_cols)} columns")
            print(f"   📅 Date range: {base_data['timestamp'].min()} to {base_data['timestamp'].max()}")
            
            # Save integrated dataset
            output_file = f"{self.paths['output_dir']}/data/integrated_dataset.csv"
            base_data.to_csv(output_file, index=False)
            print(f"   💾 Saved to: {output_file}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error creating integrated dataset: {e}")
            return False
    
    def _add_cme_features(self, base_data, time_window_minutes):
        """Add CME features to base dataset using time windows"""
        cme_data = self.datasets['cme']
        time_delta = timedelta(minutes=time_window_minutes)
        
        # Initialize CME feature columns
        cme_features = ['cme_count', 'cme_max_speed', 'cme_avg_speed', 'cme_max_width', 
                       'cme_speed_class_fast', 'cme_speed_class_extreme']
        for feature in cme_features:
            base_data[feature] = 0
        
        total_matches = 0
        
        for idx, row in base_data.iterrows():
            timestamp = row['timestamp']
            
            # Find CMEs within time window
            time_mask = (cme_data['timestamp'] >= timestamp - time_delta) & \
                       (cme_data['timestamp'] <= timestamp + time_delta)
            nearby_cmes = cme_data[time_mask]
            
            if len(nearby_cmes) > 0:
                total_matches += 1
                
                # CME count
                base_data.at[idx, 'cme_count'] = len(nearby_cmes)
                
                # Speed features
                valid_speeds = nearby_cmes['v'].dropna()
                if len(valid_speeds) > 0:
                    base_data.at[idx, 'cme_max_speed'] = valid_speeds.max()
                    base_data.at[idx, 'cme_avg_speed'] = valid_speeds.mean()
                
                # Width features
                valid_widths = nearby_cmes['width'].dropna()
                if len(valid_widths) > 0:
                    base_data.at[idx, 'cme_max_width'] = valid_widths.max()
                
                # Speed class features
                speed_classes = nearby_cmes['cme_speed_class'].value_counts()
                base_data.at[idx, 'cme_speed_class_fast'] = speed_classes.get('Fast', 0) + speed_classes.get('Very Fast', 0)
                base_data.at[idx, 'cme_speed_class_extreme'] = speed_classes.get('Extreme', 0)
        
        print(f"      📈 CME features added to {total_matches} data points")
        return base_data
    
    def _add_mag_features(self, base_data, time_window_minutes):
        """Add MAG features to base dataset using time windows"""
        mag_data = self.datasets['mag']
        time_delta = timedelta(minutes=time_window_minutes)
        
        # Get available MAG components
        mag_cols = [col for col in mag_data.columns if col.startswith('mag_')]
        
        # Initialize MAG feature columns
        for col in mag_cols:
            base_data[f'{col}_mean'] = np.nan
            base_data[f'{col}_std'] = np.nan
            base_data[f'{col}_min'] = np.nan
            base_data[f'{col}_max'] = np.nan
        
        total_matches = 0
        
        for idx, row in base_data.iterrows():
            timestamp = row['timestamp']
            
            # Find MAG data within time window
            time_mask = (mag_data['timestamp'] >= timestamp - time_delta) & \
                       (mag_data['timestamp'] <= timestamp + time_delta)
            nearby_mag = mag_data[time_mask]
            
            if len(nearby_mag) > 0:
                total_matches += 1
                
                for col in mag_cols:
                    if col in nearby_mag.columns:
                        values = nearby_mag[col].dropna()
                        if len(values) > 0:
                            base_data.at[idx, f'{col}_mean'] = values.mean()
                            base_data.at[idx, f'{col}_std'] = values.std()
                            base_data.at[idx, f'{col}_min'] = values.min()
                            base_data.at[idx, f'{col}_max'] = values.max()
        
        print(f"      🧲 MAG features added to {total_matches} data points")
        return base_data
    
    def _classify_cme_speed(self, speed):
        """Classify CME speed into categories"""
        if pd.isna(speed):
            return 'Unknown'
        elif speed < 500:
            return 'Slow'
        elif speed < 800:
            return 'Fast'
        elif speed < 1200:
            return 'Very Fast'
        else:
            return 'Extreme'
    
    def generate_integration_summary(self):
        """Generate comprehensive summary report"""
        try:
            print("\n📋 Generating Integration Summary Report...")
            
            report_lines = []
            report_lines.append("# ISRO Multi-Dataset Integration Summary Report")
            report_lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")
            
            # Dataset summaries
            report_lines.append("## Dataset Summary")
            report_lines.append("")
            
            for dataset_name, data in self.datasets.items():
                if dataset_name == 'mag_files':
                    report_lines.append(f"### {dataset_name.upper()} Dataset")
                    report_lines.append(f"- **Available files**: {len(data)}")
                    if len(data) > 0:
                        report_lines.append(f"- **Sample files**: {', '.join([os.path.basename(f) for f in data[:3]])}")
                elif isinstance(data, pd.DataFrame):
                    report_lines.append(f"### {dataset_name.upper()} Dataset")
                    report_lines.append(f"- **Data points**: {len(data):,}")
                    report_lines.append(f"- **Features**: {len(data.columns)}")
                    if 'timestamp' in data.columns:
                        report_lines.append(f"- **Date range**: {data['timestamp'].min()} to {data['timestamp'].max()}")
                        report_lines.append(f"- **Duration**: {(data['timestamp'].max() - data['timestamp'].min()).days} days")
                report_lines.append("")
            
            # Integrated dataset summary
            if self.integrated_data is not None:
                report_lines.append("## Integrated Dataset Summary")
                report_lines.append("")
                report_lines.append(f"- **Total data points**: {len(self.integrated_data):,}")
                report_lines.append(f"- **Total features**: {len(self.integrated_data.columns)}")
                report_lines.append(f"- **Date range**: {self.integrated_data['timestamp'].min()} to {self.integrated_data['timestamp'].max()}")
                
                # Feature categories
                feature_categories = {
                    'Solar Wind': [col for col in self.integrated_data.columns if any(x in col.lower() for x in ['proton', 'alpha', 'temperature'])],
                    'CME': [col for col in self.integrated_data.columns if 'cme' in col.lower()],
                    'Magnetic Field': [col for col in self.integrated_data.columns if 'mag_' in col]
                }
                
                report_lines.append("\n### Feature Categories")
                for category, features in feature_categories.items():
                    if features:
                        report_lines.append(f"- **{category}**: {len(features)} features")
                        report_lines.append(f"  - {', '.join(features[:5])}")
                        if len(features) > 5:
                            report_lines.append(f"  - ... and {len(features)-5} more")
                
                # Data quality assessment
                report_lines.append("\n### Data Quality Assessment")
                total_points = len(self.integrated_data)
                feature_cols = [col for col in self.integrated_data.columns if col != 'timestamp']
                
                for col in feature_cols[:10]:  # Show first 10 features
                    valid_count = self.integrated_data[col].notna().sum()
                    completeness = (valid_count / total_points) * 100
                    report_lines.append(f"- **{col}**: {completeness:.1f}% complete ({valid_count:,}/{total_points:,})")
            
            # Save report
            report_content = "\n".join(report_lines)
            report_file = f"{self.paths['output_dir']}/reports/integration_summary_report.md"
            
            with open(report_file, 'w') as f:
                f.write(report_content)
            
            print(f"   ✅ Integration summary saved to: {report_file}")
            return True
            
        except Exception as e:
            print(f"   ❌ Error generating summary: {e}")
            return False
    
    def create_correlation_analysis(self):
        """Create correlation analysis between different data types"""
        try:
            print("\n🔗 Creating Correlation Analysis...")
            
            if self.integrated_data is None:
                print("   ❌ No integrated dataset available")
                return False
            
            # Select numeric columns for correlation
            numeric_cols = self.integrated_data.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != 'timestamp']
            
            if len(numeric_cols) < 2:
                print("   ⚠️ Not enough numeric features for correlation analysis")
                return False
            
            # Calculate correlation matrix
            corr_matrix = self.integrated_data[numeric_cols].corr()
            
            # Create correlation heatmap
            plt.figure(figsize=(20, 16))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                       center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8},
                       fmt='.2f', annot_kws={'size': 8})
            
            plt.title('Multi-Dataset Feature Correlation Matrix\n(Solar Wind + CME + MAG)', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            
            plot_file = f"{self.paths['output_dir']}/plots/correlation_matrix.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ Correlation heatmap saved to: {plot_file}")
            
            # Find strongest correlations
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if not np.isnan(corr_val) and abs(corr_val) > 0.3:  # Strong correlation threshold
                        corr_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': corr_val
                        })
            
            # Sort by absolute correlation
            corr_pairs = sorted(corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)
            
            print(f"   🔍 Found {len(corr_pairs)} strong correlations (|r| > 0.3)")
            if corr_pairs:
                print("   📊 Top 5 correlations:")
                for pair in corr_pairs[:5]:
                    print(f"      {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error creating correlation analysis: {e}")
            return False
    
    def run_full_integration(self):
        """Run complete integration workflow"""
        print("\n" + "="*80)
        print("🚀 STARTING COMPREHENSIVE ISRO DATA INTEGRATION")
        print("="*80)
        
        success_steps = []
        
        # Step 1: Load all available datasets
        if self.load_solar_wind_data():
            success_steps.append("Solar Wind Data")
        
        if self.load_cme_data():
            success_steps.append("CME Data")
        
        if self.scan_mag_data():
            success_steps.append("MAG File Scan")
            if self.load_sample_mag_data():
                success_steps.append("MAG Data Loading")
        
        # Step 2: Create integrated dataset
        if len(success_steps) >= 2:  # At least 2 datasets needed
            if self.create_time_aligned_dataset():
                success_steps.append("Data Integration")
        
        # Step 3: Generate analysis
        if self.generate_integration_summary():
            success_steps.append("Summary Report")
        
        if self.create_correlation_analysis():
            success_steps.append("Correlation Analysis")
        
        # Final summary
        print("\n" + "="*80)
        print("🎯 INTEGRATION WORKFLOW COMPLETE")
        print("="*80)
        print(f"✅ Completed steps: {', '.join(success_steps)}")
        print(f"📁 Output directory: {self.paths['output_dir']}")
        
        if self.integrated_data is not None:
            print(f"📊 Final integrated dataset: {len(self.integrated_data):,} data points")
            print(f"🔧 Available for ML analysis: ✅")
        
        return len(success_steps) >= 4

if __name__ == "__main__":
    # Initialize and run integration
    integrator = ISRODataIntegrator()
    success = integrator.run_full_integration()
    
    if success:
        print("\n🎉 Integration completed successfully!")
        print("📋 Next steps: Use integrated dataset for ML model development")
    else:
        print("\n⚠️ Integration completed with some limitations")
        print("📋 Check individual dataset availability and try again")
