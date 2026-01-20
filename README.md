# Fractal Motor Expertise Data Repository

## Study Title

**Fractal architectures in motor expertise: bridging deterministic and stochastic control**

## Authors

**Chulwook Park**
- BK Professor, Department of Physical Education, Seoul National University
- Okinawa Institute of Science and Technology (OIST)
- International Institute for Applied Systems Analysis (IIASA)
- Email: pcw8531@snu.ac.kr

## Description

This repository contains the raw data and analysis code for our study examining motor expertise through fractal architectures, comparing movement trajectories in expert vs novice performers.

## Repository Contents

| File | Description |
|------|-------------|
| `trajectory_data.pkl` | 3D movement trajectory data (pickle format) |
| `trajectory_data_full.pkl` | Complete trajectory dataset with all trials |
| `trajectory_metrics.csv` | Computed trajectory metrics (fractal dimensions, etc.) |
| `trajectory_summary.csv` | Summary statistics for trajectory data |
| `performance_data.csv` | Performance metrics (accuracy, errors) |
| `performance_data_empirical.csv` | Empirical performance measurements |
| `participant_summary.csv` | Participant demographics and group assignments |
| `analysis_code.ipynb` | Jupyter notebook with FFT, PSD, and statistical analyses |

## Data Format

- `.pkl` files: Python pickle format (use `pandas.read_pickle()` or `pickle.load()`)
- `.csv` files: Comma-separated values, UTF-8 encoded
- `.ipynb` files: Jupyter Notebook format

## Usage

```python
import pandas as pd
import pickle

# Load trajectory data
trajectory_data = pd.read_pickle('trajectory_data.pkl')

# Load CSV files
performance = pd.read_csv('performance_data.csv')
metrics = pd.read_csv('trajectory_metrics.csv')
```

## License

- **Code**: MIT License
- **Data**: CC BY 4.0 - freely available with attribution

## Citation

If you use this data, please cite:

> Park, C. (2025). Fractal architectures in motor expertise: bridging deterministic and stochastic control. *Manuscript submitted for publication*.

## Contact

For questions about the data or methodology, contact: pcw8531@snu.ac.kr

## Ethics

This study was approved by Seoul National University Institutional Review Board (IRB). All participants provided informed consent.

## Funding

This work was supported by the National Research Foundation of Korea.
