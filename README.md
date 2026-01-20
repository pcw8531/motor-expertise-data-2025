# Fractal Architectures in Motor Expertise

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Data repository and analysis code for the study: **"Fractal architectures in motor expertise: bridging deterministic and stochastic control"**

## Overview

This repository contains raw experimental data and reproducible analysis code examining how motor expertise manifests as fractal organization of movement. The study demonstrates that expert performers exhibit higher fractal dimensions, broader frequency utilization, and stronger scale-invariant properties compared to novices.

## Repository Structure

```
motor-expertise-data-2025/
├── README.md                     # This file
├── LICENSE                       # MIT License
├── requirements.txt              # Python dependencies
├── data/
│   ├── raw/                      # Original experimental measurements
│   │   ├── trajectory_e_.txt     # Expert trajectory data
│   │   ├── trajectory_n_.txt     # Novice trajectory data
│   │   ├── start_angle_e_.txt    # Expert start angles
│   │   ├── start_angle_n_.txt    # Novice start angles
│   │   ├── max_angle_e_.txt      # Expert max angles
│   │   ├── max_angle_n_.txt      # Novice max angles
│   │   ├── max_velocity_e_.txt   # Expert max velocity
│   │   ├── max_velocity_n_.txt   # Novice max velocity
│   │   ├── mean_velocity_e_.txt  # Expert mean velocity
│   │   └── mean_velocity_n_.txt  # Novice mean velocity
│   └── processed/                # Analysis-ready datasets
│       ├── performance_data.csv
│       ├── trajectory_metrics.csv
│       └── participant_summary.csv
├── src/
│   ├── __init__.py
│   ├── fractal_analysis.py       # Box-counting, DFA functions
│   ├── frequency_analysis.py     # FFT, PSD calculations
│   ├── statistical_analysis.py   # Statistical tests
│   └── visualization.py          # Figure generation
├── notebooks/
│   └── analysis_walkthrough.ipynb  # Step-by-step demonstration
└── results/
    └── figures/                  # Generated figures (optional)
```

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/pcw8531/motor-expertise-data-2025.git
cd motor-expertise-data-2025
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the analysis
```python
from src.fractal_analysis import calculate_fractal_dimension, detrended_fluctuation_analysis
from src.frequency_analysis import compute_fft, compute_psd

# Example: Calculate fractal dimension
import pandas as pd
data = pd.read_csv('data/processed/trajectory_metrics.csv')
```

Or explore the interactive notebook:
```bash
jupyter notebook notebooks/analysis_walkthrough.ipynb
```

## Data Description

### Participants
- **Experts**: 10 table tennis players (Korea National League level)
- **Novices**: 10 participants with no formal training
- **Total trials**: 200 (10 participants × 10 trials × 2 groups)

### Measurements
| Variable | Description | Unit |
|----------|-------------|------|
| Trajectory | 3D racket movement path (X, Y, Z) | meters |
| Absolute Error | Distance from target contact point | cm |
| Frequency Bandwidth | Range containing 90% spectral power | Hz |
| Fractal Dimension | Box-counting dimension | D (1.0-2.0) |

### File Formats
- `.txt`: Tab-delimited raw measurements
- `.csv`: Comma-separated processed data

## Key Analyses

### Fractal Analysis
- **Box-counting method**: Calculates fractal dimension across 20 logarithmically-spaced scales
- **Detrended Fluctuation Analysis (DFA)**: Assesses long-range temporal correlations
- **Multifractal spectrum**: Characterizes scaling heterogeneity

### Frequency Analysis
- **Fast Fourier Transform (FFT)**: Converts temporal data to frequency domain
- **Power Spectral Density (PSD)**: Reveals frequency component distribution

### Statistical Analysis
- Independent t-tests with Cohen's d effect sizes
- Bootstrap resampling (n=1000) for confidence intervals
- Bonferroni correction for multiple comparisons

## Citation

If you use this data or code, please cite:

```bibtex
@article{park2025fractal,
  title={Fractal architectures in motor expertise: bridging deterministic and stochastic control},
  author={Park, Chulwook},
  year={2025}
}
```

## Author

**Chulwook Park**
- BK Professor, Seoul National University
- Okinawa Institute of Science and Technology (OIST)
- International Institute for Applied Systems Analysis (IIASA)
- Email: pcw8531@snu.ac.kr

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work was supported by the Basic Science Research Program through the National Research Foundation of Korea (NRF), funded by the Ministry of Education (Grant No. 2020R11A1A01056967).
