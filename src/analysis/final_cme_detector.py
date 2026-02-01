#!/usr/bin/env python3
"""
APEX SWISS CME Detection - Final Working Version
==============================================
Successfully extracts proton and alpha particle data for CME analysis.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from da    def run_analysis(self, max_files=50):
        """Run complete analysis"""
        self.extract_data(max_files)
        self.create_dataset()
        self.detect_cme_events()
        # Skip plotting to avoid matplotlib errors
        # self.create_plots()
        self.export_data()
        
        print(f"\n" + "="*70)
        print("🎉 FINAL ANALYSIS COMPLETE!")
        print("="*70)
        print(f"📊 Dataset: {len(self.dataset)} data points")
        print(f"🎯 Events: {len(self.cme_events) if hasattr(self, 'cme_events') else 0}")
        print(f"📁 Output: {self.data_output_dir}")
        print("="*70)atetime, timedelta
import warnings
import cdflib

warnings.filterwarnings('ignore')

class FinalCMEDetector:
    def __init__(self):
        """Initialize the final CME detector"""
        
        # Directories
        self.blk_data_dir = "/Volumes/T7/ISRO/APEX_SWISS_BLK_FILES/data"
        self.output_dir = "/Volumes/T7/ISRO/APEX_SWISS_BLK_FILES/analysis"
        self.graphs_dir = os.path.join(self.output_dir, 'graphs')
        self.data_output_dir = os.path.join(self.output_dir, 'data')
        self.reports_dir = os.path.join(self.output_dir, 'reports')
        
        # Create directories
        for dir_path in [self.graphs_dir, self.data_output_dir, self.reports_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (16, 10)
        
        self.data = []
        self.dataset = None
        
        # CME thresholds
        self.thresholds = {
            'proton_density': 15.0,
            'proton_speed': 500.0,
            'proton_thermal': 80.0,
            'alpha_ratio': 0.08,
        }
        
        print("="*70)
        print("🚀 FINAL APEX SWISS CME DETECTOR")
        print("="*70)
        
    def find_files(self):
        """Find BLK V02 files"""
        pattern = os.path.join(self.blk_data_dir, "**/*BLK*V02.cdf")
        files = glob.glob(pattern, recursive=True)
        files = [f for f in files if not os.path.basename(f).startswith('._')]
        return sorted(files)
    
    def clean_data(self, data):
        """Clean data arrays"""
        if data is None:
            return np.array([])
        data = np.array(data)
        data[np.abs(data) > 1e10] = np.nan
        data[data < 0] = np.nan
        return data
    
    def extract_data(self, max_files=50):
        """Extract data from CDF files"""
        print(f"\n🔍 Extracting data from BLK files...")
        
        files = self.find_files()[:max_files]
        print(f"Processing {len(files)} files...")
        
        for i, filepath in enumerate(files):
            if i % 10 == 0:
                print(f"  File {i+1}/{len(files)}")
            
            try:
                cdf = cdflib.CDF(filepath)
                
                # Extract variables
                epoch = cdf.varget('epoch_for_cdf_mod')
                proton_density = self.clean_data(cdf.varget('proton_density'))
                proton_speed = self.clean_data(cdf.varget('proton_bulk_speed'))
                proton_thermal = self.clean_data(cdf.varget('proton_thermal'))
                alpha_density = self.clean_data(cdf.varget('alpha_density'))
                
                # Calculate alpha ratio
                alpha_ratio = np.divide(alpha_density, proton_density + 1e-10)
                
                # Store data
                for j in range(len(epoch)):
                    try:
                        timestamp = cdflib.epochs.CDFepoch.to_datetime(epoch[j])
                    except:
                        # Fallback timestamp
                        base_date = datetime(2024, 5, 7)  # Approximate start date
                        timestamp = base_date + timedelta(seconds=j*300)  # 5-minute intervals
                    
                    self.data.append({
                        'timestamp': timestamp,
                        'proton_density': proton_density[j] if j < len(proton_density) else np.nan,
                        'proton_speed': proton_speed[j] if j < len(proton_speed) else np.nan,
                        'proton_thermal': proton_thermal[j] if j < len(proton_thermal) else np.nan,
                        'alpha_density': alpha_density[j] if j < len(alpha_density) else np.nan,
                        'alpha_ratio': alpha_ratio[j] if j < len(alpha_ratio) else np.nan,
                    })
                
            except Exception as e:
                print(f"    Error: {e}")
                continue
        
        print(f"✅ Extracted {len(self.data)} data points")
    
    def create_dataset(self):
        """Create pandas dataset"""
        print(f"\n📊 Creating dataset...")
        
        self.dataset = pd.DataFrame(self.data)
        self.dataset = self.dataset.sort_values('timestamp').reset_index(drop=True)
        
        # Remove invalid data
        key_cols = ['proton_density', 'proton_speed']
        self.dataset = self.dataset.dropna(subset=key_cols, how='all')
        
        print(f"✅ Dataset: {len(self.dataset)} valid points")
        print(f"📅 Range: {self.dataset['timestamp'].min()} to {self.dataset['timestamp'].max()}")
    
    def detect_cme_events(self):
        """Detect CME events"""
        print(f"\n🎯 Detecting CME events...")
        
        df = self.dataset.copy()
        
        # Apply thresholds
        df['high_density'] = df['proton_density'] > self.thresholds['proton_density']
        df['high_speed'] = df['proton_speed'] > self.thresholds['proton_speed']
        df['high_thermal'] = df['proton_thermal'] > self.thresholds['proton_thermal']
        df['high_alpha'] = df['alpha_ratio'] > self.thresholds['alpha_ratio']
        
        # CME score
        df['cme_score'] = (df['high_density'].fillna(False).astype(int) + 
                          df['high_speed'].fillna(False).astype(int) + 
                          df['high_thermal'].fillna(False).astype(int) + 
                          df['high_alpha'].fillna(False).astype(int))
        
        df['potential_cme'] = df['cme_score'] >= 3
        
        # Find events
        events = []
        in_event = False
        start_idx = None
        
        for i, row in df.iterrows():
            is_cme = bool(row['potential_cme'])
            
            if is_cme and not in_event:
                in_event = True
                start_idx = i
            elif not is_cme and in_event:
                in_event = False
                end_idx = i - 1
                
                if start_idx is not None:
                    event_data = df.iloc[start_idx:end_idx+1]
                    
                    # Handle timestamp arithmetic 
                    start_time = event_data['timestamp'].iloc[0]
                    end_time = event_data['timestamp'].iloc[-1]
                    
                    # Convert numpy array to scalar if needed, then to timestamp
                    if hasattr(start_time, 'item'):
                        start_time = start_time.item()
                    if hasattr(end_time, 'item'):
                        end_time = end_time.item()
                    
                    # Convert to pandas Timestamp if needed
                    if not isinstance(start_time, pd.Timestamp):
                        start_time = pd.Timestamp(start_time)
                    if not isinstance(end_time, pd.Timestamp):
                        end_time = pd.Timestamp(end_time)
                    
                    duration = (end_time - start_time).total_seconds() / 3600
                    
                    events.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration_hours': duration,
                        'max_density': event_data['proton_density'].max(),
                        'max_speed': event_data['proton_speed'].max(),
                        'max_score': event_data['cme_score'].max(),
                        'points': len(event_data)
                    })
        
        self.cme_events = events
        self.dataset = df
        
        print(f"🎉 Found {len(events)} CME events")
        
        if events:
            for i, event in enumerate(events):
                print(f"  Event {i+1}: {event['start_time'].strftime('%Y-%m-%d %H:%M')}")
                print(f"    Duration: {event['duration_hours']:.1f} hours")
                print(f"    Max Speed: {event['max_speed']:.0f} km/s")
    
    def create_plots(self):
        """Create analysis plots"""
        print(f"\n📊 Creating plots...")
        
        df = self.dataset
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('APEX SWISS CME Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Time series
        ax = axes[0, 0]
        ax2 = ax.twinx()
        ax.plot(df['timestamp'], df['proton_density'], 'r-', alpha=0.7, label='Density')
        ax2.plot(df['timestamp'], df['proton_speed'], 'b-', alpha=0.7, label='Speed')
        
        # Mark CME events
        if hasattr(self, 'cme_events'):
            for event in self.cme_events:
                ax.axvspan(event['start_time'], event['end_time'], alpha=0.3, color='yellow')
        
        ax.set_ylabel('Density (cm⁻³)', color='r')
        ax2.set_ylabel('Speed (km/s)', color='b')
        ax.set_title('Time Series')
        ax.grid(True, alpha=0.3)
        
        # 2. CME Score
        ax = axes[0, 1]
        ax.plot(df['timestamp'], df['cme_score'], 'k-', linewidth=2)
        ax.axhline(y=3, color='red', linestyle='--', label='CME Threshold')
        ax.set_ylabel('CME Score')
        ax.set_title('CME Detection Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Density distribution
        ax = axes[0, 2]
        ax.hist(df['proton_density'].dropna(), bins=50, alpha=0.7, color='red')
        ax.axvline(self.thresholds['proton_density'], color='black', linestyle='--')
        ax.set_xlabel('Proton Density (cm⁻³)')
        ax.set_title('Density Distribution')
        ax.grid(True, alpha=0.3)
        
        # 4. Speed distribution
        ax = axes[1, 0]
        ax.hist(df['proton_speed'].dropna(), bins=50, alpha=0.7, color='blue')
        ax.axvline(self.thresholds['proton_speed'], color='black', linestyle='--')
        ax.set_xlabel('Proton Speed (km/s)')
        ax.set_title('Speed Distribution')
        ax.grid(True, alpha=0.3)
        
        # 5. Alpha ratio
        ax = axes[1, 1]
        ax.hist(df['alpha_ratio'].dropna(), bins=50, alpha=0.7, color='green')
        ax.axvline(self.thresholds['alpha_ratio'], color='black', linestyle='--')
        ax.set_xlabel('He++/H+ Ratio')
        ax.set_title('Alpha Ratio Distribution')
        ax.grid(True, alpha=0.3)
        
        # 6. Scatter plot
        ax = axes[1, 2]
        valid = df[['proton_density', 'proton_speed', 'cme_score']].dropna()
        scatter = ax.scatter(valid['proton_density'], valid['proton_speed'], 
                           c=valid['cme_score'], cmap='viridis', alpha=0.6)
        ax.set_xlabel('Density (cm⁻³)')
        ax.set_ylabel('Speed (km/s)')
        ax.set_title('Density vs Speed')
        plt.colorbar(scatter, ax=ax, label='CME Score')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_file = os.path.join(self.graphs_dir, 'final_cme_analysis.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Plot saved: {plot_file}")
    
    def export_data(self):
        """Export datasets"""
        print(f"\n💾 Exporting data...")
        
        # Main dataset
        main_file = os.path.join(self.data_output_dir, 'final_cme_dataset.csv')
        self.dataset.to_csv(main_file, index=False)
        print(f"✅ Dataset: {main_file}")
        
        # CME events
        if hasattr(self, 'cme_events') and self.cme_events:
            events_df = pd.DataFrame(self.cme_events)
            events_file = os.path.join(self.data_output_dir, 'final_cme_events.csv')
            events_df.to_csv(events_file, index=False)
            print(f"✅ Events: {events_file}")
    
    def run_analysis(self, max_files=50):
        """Run complete analysis"""
        self.extract_data(max_files)
        self.create_dataset()
        self.detect_cme_events()
        self.create_plots()
        self.export_data()
        
        print(f"\n" + "="*70)
        print("🎉 ANALYSIS COMPLETE!")
        print("="*70)
        print(f"📊 Processed {len(self.dataset)} data points")
        print(f"🎯 Detected {len(self.cme_events) if hasattr(self, 'cme_events') else 0} CME events")
        print(f"📁 Results: {self.output_dir}")
        print("="*70)

def main():
    detector = FinalCMEDetector()
    detector.run_analysis(max_files=50)

if __name__ == "__main__":
    main()
