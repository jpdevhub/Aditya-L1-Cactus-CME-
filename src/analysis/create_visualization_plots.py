#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import os

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

# Create directory for plots
plots_dir = '/Volumes/T7/ISRO(HALO_CME)/plots'
os.makedirs(plots_dir, exist_ok=True)

print("Loading dataset...")
# Load the finalized dataset
df = pd.read_csv('/Volumes/T7/ISRO(HALO_CME)/finalized_cme_dataset_Aug2024_Jun2025.csv')

# Convert date and time columns to datetime
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])

# Print basic stats
print(f"Dataset loaded: {len(df)} records")
print("\nClass distribution:")
print(df['halo_class'].value_counts())
print("\nCME vs non-CME:")
print(df['has_cme'].value_counts())

# =============================================================================
# 1. DATASET CHARACTERIZATION
# =============================================================================
print("\nGenerating dataset characterization plots...")

# 1.1 Histogram/KDE plots for CME properties
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Filter only CME events for some plots
cme_df = df[df['has_cme'] == 1]

# CME Velocity Distribution
sns.histplot(cme_df['cme_velocity'].dropna(), kde=True, ax=axes[0], color='blue')
axes[0].set_title('Distribution of CME Velocities')
axes[0].set_xlabel('CME Velocity (km/s)')
axes[0].set_ylabel('Count')

# CME Width Distribution
sns.histplot(cme_df['cme_width'].dropna(), kde=True, ax=axes[1], color='green')
axes[1].set_title('Distribution of CME Widths')
axes[1].set_xlabel('Angular Width (degrees)')
axes[1].set_ylabel('Count')

# Earth-directed Probability Distribution
sns.histplot(cme_df['earth_directed'].dropna(), kde=True, ax=axes[2], color='red')
axes[2].set_title('Distribution of Earth-directed Probability')
axes[2].set_xlabel('Earth-directed Probability')
axes[2].set_ylabel('Count')

plt.tight_layout()
plt.savefig(f'{plots_dir}/cme_property_distributions.png', dpi=300, bbox_inches='tight')

# 1.2 Class Balance Plot
plt.figure(figsize=(10, 6))
class_counts = df['halo_class'].value_counts().sort_index()
ax = sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
plt.title('CME Class Distribution')
plt.xlabel('Halo Class')
plt.ylabel('Count')
plt.xticks(range(5), ['No CME (0)', 'Class 1', 'Class 2', 'Class 3', 'Class 4'])

# Add count labels on top of bars
for i, v in enumerate(class_counts.values):
    ax.text(i, v + 50, f"{v}", ha='center', fontweight='bold')

# Use log scale for better visualization
plt.yscale('log')
plt.savefig(f'{plots_dir}/cme_class_distribution.png', dpi=300, bbox_inches='tight')

# 1.3 CME vs Non-CME Balance
plt.figure(figsize=(8, 6))
cme_balance = df['has_cme'].value_counts().sort_index()
ax = sns.barplot(x=['No CME', 'CME'], y=cme_balance.values, palette=['lightblue', 'darkblue'])
plt.title('CME vs Non-CME Distribution')
plt.xlabel('Event Type')
plt.ylabel('Count')

# Add count labels on top of bars
for i, v in enumerate(cme_balance.values):
    ax.text(i, v + 50, f"{v}", ha='center', fontweight='bold')

# Use log scale for better visualization
plt.yscale('log')
plt.savefig(f'{plots_dir}/cme_vs_noncme_distribution.png', dpi=300, bbox_inches='tight')

# 1.4 Time Series Plot: CME events over time
# Group by date and count CMEs
df['date'] = pd.to_datetime(df['date'])
cme_daily_counts = df[df['has_cme'] == 1].groupby(df['date'].dt.date).size()
cme_daily_counts = cme_daily_counts.reindex(pd.date_range(start=cme_daily_counts.index.min(), 
                                                      end=cme_daily_counts.index.max()), 
                                        fill_value=0)

# Plot time series
plt.figure(figsize=(18, 6))
plt.plot(cme_daily_counts.index, cme_daily_counts.values, marker='o', linestyle='-', markersize=3)
plt.title('Number of CME Events per Day')
plt.xlabel('Date')
plt.ylabel('Number of CME Events')
plt.grid(True, alpha=0.3)

# Format x-axis to show months
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{plots_dir}/cme_events_time_series.png', dpi=300, bbox_inches='tight')

# Monthly CME counts
df['month'] = df['date'].dt.strftime('%Y-%m')
cme_monthly_counts = df[df['has_cme'] == 1].groupby('month').size()

plt.figure(figsize=(18, 6))
sns.barplot(x=cme_monthly_counts.index, y=cme_monthly_counts.values, color='darkblue')
plt.title('Number of CME Events per Month')
plt.xlabel('Month')
plt.ylabel('Number of CME Events')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{plots_dir}/cme_events_monthly.png', dpi=300, bbox_inches='tight')

# =============================================================================
# 2. CORRELATION & RELATIONSHIPS
# =============================================================================
print("\nGenerating correlation and relationship plots...")

# 2.1 Correlation Heatmap
# Select numerical features for correlation analysis
corr_columns = [
    'has_cme', 'cme_velocity', 'cme_width', 'pa', 'halo_class', 'earth_directed',
    'he_h_ratio', 'proton_density_mean', 'proton_temperature_mean', 'proton_bulk_speed_mean', 
    'temp_ratio'
]

# Replace NaN values with 0 for correlation calculation in the case of non-CME events
corr_df = df[corr_columns].copy()
for col in ['cme_velocity', 'cme_width', 'pa', 'earth_directed']:
    corr_df[col] = corr_df[col].fillna(0)

# Calculate correlation matrix
corr_matrix = corr_df.corr(method='spearman')

# Plot correlation heatmap
plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
            mask=mask, vmin=-1, vmax=1, linewidths=0.5, annot_kws={"size": 10})
plt.title('Spearman Correlation Heatmap')
plt.tight_layout()
plt.savefig(f'{plots_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')

# 2.2 Pairplot for CME vs Non-CME
# Select a subset of data for pairplot (can be computationally intensive)
pairplot_sample = df.sample(min(2000, len(df)))

# Convert has_cme to categorical for better visualization
pairplot_sample['CME Event'] = pairplot_sample['has_cme'].map({0: 'No CME', 1: 'CME'})

# Select features for pairplot
pairplot_features = [
    'proton_density_mean', 'proton_temperature_mean', 'proton_bulk_speed_mean',
    'he_h_ratio', 'temp_ratio', 'CME Event'
]

# Create pairplot
g = sns.pairplot(pairplot_sample[pairplot_features], 
                 hue='CME Event', 
                 palette={'No CME': 'blue', 'CME': 'red'},
                 plot_kws={'alpha': 0.5, 's': 20}, 
                 diag_kind='kde')
g.fig.suptitle('Feature Relationships: CME vs Non-CME', y=1.02, fontsize=16)
plt.tight_layout()
plt.savefig(f'{plots_dir}/cme_feature_pairplot.png', dpi=300, bbox_inches='tight')

# 2.3 Boxplots for Outlier Visualization
# CME Velocity Boxplot by Class
plt.figure(figsize=(12, 6))
sns.boxplot(x='halo_class', y='cme_velocity', data=cme_df, palette='viridis')
plt.title('CME Velocity Distribution by Halo Class')
plt.xlabel('Halo Class')
plt.ylabel('CME Velocity (km/s)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{plots_dir}/cme_velocity_boxplot.png', dpi=300, bbox_inches='tight')

# CME Width Boxplot by Class
plt.figure(figsize=(12, 6))
sns.boxplot(x='halo_class', y='cme_width', data=cme_df, palette='viridis')
plt.title('CME Width Distribution by Halo Class')
plt.xlabel('Halo Class')
plt.ylabel('CME Width (degrees)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{plots_dir}/cme_width_boxplot.png', dpi=300, bbox_inches='tight')

# 2.4 Scatterplot: CME Velocity vs Width with Halo Class
plt.figure(figsize=(14, 10))
scatter = sns.scatterplot(x='cme_velocity', y='cme_width', hue='halo_class', 
                         palette='viridis', data=cme_df, s=100, alpha=0.7)
plt.title('CME Velocity vs Width by Halo Class')
plt.xlabel('CME Velocity (km/s)')
plt.ylabel('CME Width (degrees)')
plt.grid(True, alpha=0.3)
plt.legend(title='Halo Class')
plt.tight_layout()
plt.savefig(f'{plots_dir}/cme_velocity_vs_width_scatter.png', dpi=300, bbox_inches='tight')

# 2.5 Additional plot: Solar wind parameters during CME vs non-CME
# Compare solar wind parameters for CME vs non-CME events
solar_params = [
    'proton_density_mean', 'proton_temperature_mean', 'proton_bulk_speed_mean', 
    'he_h_ratio', 'temp_ratio'
]

fig, axes = plt.subplots(len(solar_params), 1, figsize=(12, 3*len(solar_params)))

for i, param in enumerate(solar_params):
    sns.violinplot(x='has_cme', y=param, data=df, ax=axes[i], palette=['lightblue', 'darkred'])
    axes[i].set_title(f'{param} Distribution: CME vs No CME')
    axes[i].set_xlabel('')
    axes[i].set_ylabel(param)
    axes[i].set_xticklabels(['No CME', 'CME'])

plt.tight_layout()
plt.savefig(f'{plots_dir}/solar_params_cme_vs_noncme.png', dpi=300, bbox_inches='tight')

print(f"All plots saved to {plots_dir}/")
