# CACTUS CME Combined Dataset - August 2024 to June 2025

## Overview
This document describes the combined CACTUS (Computer Aided CME Tracking) dataset containing Coronal Mass Ejection (CME) observations from August 2024 to June 2025.

## File Information
- **Filename**: `CACTUS_CME_Combined_Aug2024_Jun2025.csv`
- **Location**: `/Volumes/T7/ISRO/CME(CACTUS)/`
- **File Size**: 90 KB
- **Total Records**: 1,744 CME events
- **Date Range**: August 1, 2024 02:48 UTC to June 30, 2025 12:00 UTC
- **Duration**: 11 months of continuous observations

## Data Structure
The combined dataset contains 10 columns:

| Column | Description | Units | Notes |
|--------|-------------|-------|-------|
| CME | Sequential CME number | - | Renumbered 1-1744 chronologically |
| t0 | CME first detection time | YYYY/MM/DD HH:MM | UTC time format |
| dt0 | Detection uncertainty | hours | Time uncertainty in detection |
| pa | Position angle | degrees | Angular position on solar disk |
| da | Angular width | degrees | Width of CME |
| v | Linear velocity | km/s | CME propagation velocity |
| dv | Velocity uncertainty | km/s | Uncertainty in velocity measurement |
| minv | Minimum velocity | km/s | Lower bound of velocity estimate |
| maxv | Maximum velocity | km/s | Upper bound of velocity estimate |
| halo? | Halo CME classification | - | II/III/IV for partial/full halo CMEs |

## Monthly Distribution
| Month | CME Count | Percentage |
|-------|-----------|------------|
| 2024-08 | 152 | 8.7% |
| 2024-09 | 127 | 7.3% |
| 2024-10 | 126 | 7.2% |
| 2024-11 | 162 | 9.3% |
| 2024-12 | 213 | 12.2% |
| 2025-01 | 173 | 9.9% |
| 2025-02 | 144 | 8.3% |
| 2025-03 | 162 | 9.3% |
| 2025-04 | 167 | 9.6% |
| 2025-05 | 164 | 9.4% |
| 2025-06 | 154 | 8.8% |

## Velocity Statistics
- **Mean velocity**: 507.0 km/s
- **Median velocity**: 411.0 km/s
- **Minimum velocity**: 97 km/s
- **Maximum velocity**: 1,953 km/s
- **Standard range**: Most CMEs between 200-800 km/s

## Special Events
### Halo CMEs
Total halo CMEs identified: **102 events** (5.8% of all CMEs)

| Type | Count | Description |
|------|-------|-------------|
| II | 78 | Partial halo CMEs (120-359° width) |
| III | 11 | Full halo CMEs (≥360° width) |
| IV | 13 | Complex/multiple halo CMEs |

### High-Velocity Events
- **31 CMEs** exceeded 1,000 km/s (1.8% of all events)
- **Maximum velocity**: 1,953 km/s recorded multiple times
- Most high-velocity events occurred during solar maximum period

## Data Quality
- **Completeness**: 100% temporal coverage with no gaps
- **Chronological order**: All events properly sorted by detection time
- **Missing data**: Only 2 records have missing position angle data
- **Validation**: All numeric fields validated and normalized

## Source Files Combined
1. `CACTUS_CME(aug).csv` - August 2024 (152 events)
2. `CACTUS_CME(sept).csv` - September 2024 (127 events)
3. `CACTUS_CME(oct).csv` - October 2024 (126 events)
4. `CACTUS_CME(nov).csv` - November 2024 (162 events)
5. `CACTUS_CME(dec).csv` - December 2024 (213 events)
6. `CACTUS_CME(jan).csv` - January 2025 (173 events)
7. `CACTUS_CME(feb).csv` - February 2025 (144 events)
8. `CACTUS_CME(mar).csv` - March 2025 (162 events)
9. `CACTUS_CME(april).csv` - April 2025 (167 events)
10. `CACTUS_CME(may).csv` - May 2025 (164 events)
11. `CACTUS_CME(jun).csv` - June 2025 (154 events)

## Processing Notes
- Data normalized to handle different formatting across source files
- CME numbers reassigned sequentially (1-1744) in chronological order
- Leading zeros removed from numeric fields
- Duplicate headers and empty rows removed
- Temporal consistency verified across all records

## Usage Recommendations
- Suitable for statistical analysis of CME properties
- Ideal for studying CME frequency variations over solar cycle
- Can be used for correlation studies with solar activity indices
- Appropriate for space weather research and modeling

## Contact
This dataset was compiled for ISRO space weather research purposes. For questions about the data processing or methodology, refer to the processing scripts:
- `combine_cme_data.py` - Main combination script
- `verify_cme_data.py` - Data validation script

---
*Generated on July 4, 2025*
*Dataset spans: August 1, 2024 - June 30, 2025*
*Total Duration: 334 days*
