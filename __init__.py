"""
Fractal Architectures in Motor Expertise
Analysis modules for fractal, frequency, and statistical analysis
"""

from .fractal_analysis import (
    calculate_fractal_dimension,
    detrended_fluctuation_analysis,
    multifractal_spectrum,
    self_similarity_correlation
)

from .frequency_analysis import (
    compute_fft,
    compute_psd,
    calculate_bandwidth,
    shannon_entropy
)

from .statistical_analysis import (
    independent_ttest,
    cohens_d,
    bootstrap_ci
)

__version__ = '1.0.0'
__author__ = 'Chulwook Park'
