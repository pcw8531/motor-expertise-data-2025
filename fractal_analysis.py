"""
Fractal Analysis Module
=======================
Functions for calculating fractal dimensions and related measures.

Methods implemented:
- Box-counting fractal dimension
- Detrended Fluctuation Analysis (DFA)
- Multifractal spectrum analysis
- Self-similarity correlation
"""

import numpy as np
from scipy import stats
from typing import Tuple, List, Optional


def calculate_fractal_dimension(
    trajectory: np.ndarray,
    n_boxes: int = 20,
    min_scale: float = 0.01,
    max_scale: float = 1.0
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Calculate fractal dimension using box-counting method.
    
    Parameters
    ----------
    trajectory : np.ndarray
        2D or 3D trajectory data, shape (n_points, n_dimensions)
    n_boxes : int, optional
        Number of box sizes to use (default: 20)
    min_scale : float, optional
        Minimum box size as fraction of data range (default: 0.01)
    max_scale : float, optional
        Maximum box size as fraction of data range (default: 1.0)
    
    Returns
    -------
    D : float
        Fractal dimension
    r_squared : float
        R² value of the linear fit
    log_scales : np.ndarray
        Log of box sizes used
    log_counts : np.ndarray
        Log of box counts at each scale
    
    Example
    -------
    >>> trajectory = np.random.randn(1000, 3)
    >>> D, r2, scales, counts = calculate_fractal_dimension(trajectory)
    >>> print(f"Fractal dimension: {D:.3f}, R² = {r2:.3f}")
    """
    # Normalize trajectory to [0, 1] range
    trajectory = np.asarray(trajectory)
    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)
    
    # Normalize each dimension
    mins = trajectory.min(axis=0)
    maxs = trajectory.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1  # Avoid division by zero
    normalized = (trajectory - mins) / ranges
    
    # Generate logarithmically spaced box sizes
    scales = np.logspace(np.log10(min_scale), np.log10(max_scale), n_boxes)
    counts = []
    
    for scale in scales:
        # Count boxes needed to cover the trajectory
        bins = [np.arange(0, 1 + scale, scale) for _ in range(normalized.shape[1])]
        
        # Digitize points into boxes
        box_indices = np.zeros((len(normalized), normalized.shape[1]), dtype=int)
        for dim in range(normalized.shape[1]):
            box_indices[:, dim] = np.digitize(normalized[:, dim], bins[dim])
        
        # Count unique boxes
        unique_boxes = len(set(map(tuple, box_indices)))
        counts.append(unique_boxes)
    
    counts = np.array(counts)
    
    # Linear regression on log-log plot
    log_scales = np.log(scales)
    log_counts = np.log(counts)
    
    # Remove any invalid values
    valid = np.isfinite(log_scales) & np.isfinite(log_counts)
    log_scales_valid = log_scales[valid]
    log_counts_valid = log_counts[valid]
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        log_scales_valid, log_counts_valid
    )
    
    # Fractal dimension is negative of slope
    D = -slope
    r_squared = r_value ** 2
    
    return D, r_squared, log_scales, log_counts


def detrended_fluctuation_analysis(
    time_series: np.ndarray,
    min_window: int = 4,
    max_window: Optional[int] = None,
    n_windows: int = 20,
    order: int = 2
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Perform Detrended Fluctuation Analysis (DFA).
    
    Parameters
    ----------
    time_series : np.ndarray
        1D time series data
    min_window : int, optional
        Minimum window size (default: 4)
    max_window : int, optional
        Maximum window size (default: N/4)
    n_windows : int, optional
        Number of window sizes to use (default: 20)
    order : int, optional
        Polynomial order for detrending (default: 2)
    
    Returns
    -------
    alpha : float
        DFA scaling exponent
    r_squared : float
        R² value of the linear fit
    window_sizes : np.ndarray
        Window sizes used
    fluctuations : np.ndarray
        RMS fluctuations at each window size
    
    Notes
    -----
    - alpha = 0.5: uncorrelated white noise
    - alpha > 0.5: persistent long-range correlations (fractal)
    - alpha < 0.5: anti-persistent behavior
    - alpha ≈ 0.75: critical boundary for optimal adaptability
    
    Example
    -------
    >>> ts = np.cumsum(np.random.randn(1000))
    >>> alpha, r2, windows, fluct = detrended_fluctuation_analysis(ts)
    >>> print(f"DFA exponent: {alpha:.3f}")
    """
    time_series = np.asarray(time_series).flatten()
    N = len(time_series)
    
    if max_window is None:
        max_window = N // 4
    
    # Integrate the time series
    mean_ts = np.mean(time_series)
    cumsum = np.cumsum(time_series - mean_ts)
    
    # Generate window sizes
    window_sizes = np.unique(
        np.logspace(np.log10(min_window), np.log10(max_window), n_windows).astype(int)
    )
    
    fluctuations = []
    
    for window in window_sizes:
        # Number of non-overlapping windows
        n_segments = N // window
        
        if n_segments < 1:
            continue
            
        rms_values = []
        
        for i in range(n_segments):
            start = i * window
            end = start + window
            segment = cumsum[start:end]
            
            # Fit polynomial trend
            x = np.arange(window)
            coeffs = np.polyfit(x, segment, order)
            trend = np.polyval(coeffs, x)
            
            # Calculate RMS of residuals
            residuals = segment - trend
            rms = np.sqrt(np.mean(residuals ** 2))
            rms_values.append(rms)
        
        fluctuations.append(np.mean(rms_values))
    
    window_sizes = window_sizes[:len(fluctuations)]
    fluctuations = np.array(fluctuations)
    
    # Linear regression on log-log plot
    log_windows = np.log(window_sizes)
    log_fluct = np.log(fluctuations)
    
    valid = np.isfinite(log_windows) & np.isfinite(log_fluct)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        log_windows[valid], log_fluct[valid]
    )
    
    alpha = slope
    r_squared = r_value ** 2
    
    return alpha, r_squared, window_sizes, fluctuations


def multifractal_spectrum(
    time_series: np.ndarray,
    q_values: Optional[np.ndarray] = None,
    n_scales: int = 20
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate multifractal spectrum using partition function method.
    
    Parameters
    ----------
    time_series : np.ndarray
        1D time series data
    q_values : np.ndarray, optional
        Moment orders (default: -5 to 5)
    n_scales : int, optional
        Number of scales to analyze (default: 20)
    
    Returns
    -------
    alpha : np.ndarray
        Singularity strengths
    f_alpha : np.ndarray
        Multifractal spectrum f(α)
    delta_alpha : float
        Spectrum width (α_max - α_min)
    
    Example
    -------
    >>> ts = np.random.randn(1000)
    >>> alpha, f_alpha, width = multifractal_spectrum(ts)
    >>> print(f"Spectrum width: {width:.3f}")
    """
    time_series = np.asarray(time_series).flatten()
    N = len(time_series)
    
    if q_values is None:
        q_values = np.linspace(-5, 5, 41)
    
    # Generate scales
    scales = np.unique(np.logspace(1, np.log10(N // 4), n_scales).astype(int))
    
    tau_q = []
    
    for q in q_values:
        chi_q = []
        
        for scale in scales:
            n_segments = N // scale
            
            if n_segments < 1:
                continue
            
            # Calculate measure for each segment
            measures = []
            for i in range(n_segments):
                segment = time_series[i * scale:(i + 1) * scale]
                measure = np.sum(np.abs(segment)) / N
                if measure > 0:
                    measures.append(measure)
            
            if len(measures) > 0:
                # Partition function
                if q == 1:
                    chi = np.exp(np.mean(np.log(measures)))
                else:
                    chi = np.mean(np.array(measures) ** q)
                chi_q.append(chi)
        
        if len(chi_q) > 2:
            # Calculate tau(q) from scaling
            log_scales = np.log(scales[:len(chi_q)])
            log_chi = np.log(np.array(chi_q))
            
            valid = np.isfinite(log_scales) & np.isfinite(log_chi)
            if np.sum(valid) > 2:
                slope, _, _, _, _ = stats.linregress(log_scales[valid], log_chi[valid])
                tau_q.append(slope)
            else:
                tau_q.append(np.nan)
        else:
            tau_q.append(np.nan)
    
    tau_q = np.array(tau_q)
    
    # Calculate alpha and f(alpha) using Legendre transform
    valid_tau = np.isfinite(tau_q)
    q_valid = q_values[valid_tau]
    tau_valid = tau_q[valid_tau]
    
    if len(q_valid) > 2:
        # Numerical differentiation
        alpha = np.gradient(tau_valid, q_valid)
        f_alpha = q_valid * alpha - tau_valid
        
        delta_alpha = np.max(alpha) - np.min(alpha)
    else:
        alpha = np.array([np.nan])
        f_alpha = np.array([np.nan])
        delta_alpha = np.nan
    
    return alpha, f_alpha, delta_alpha


def self_similarity_correlation(
    trajectory: np.ndarray,
    scales: List[int] = [2, 4, 8, 16, 32]
) -> Tuple[float, np.ndarray]:
    """
    Quantify self-similarity across temporal scales.
    
    Parameters
    ----------
    trajectory : np.ndarray
        Time series or trajectory data
    scales : list of int, optional
        Scale factors to analyze (default: [2, 4, 8, 16, 32])
    
    Returns
    -------
    mean_correlation : float
        Mean correlation across scales
    correlations : np.ndarray
        Correlation at each scale
    
    Example
    -------
    >>> traj = np.sin(np.linspace(0, 10*np.pi, 1000))
    >>> mean_corr, corrs = self_similarity_correlation(traj)
    >>> print(f"Mean self-similarity: {mean_corr:.3f}")
    """
    trajectory = np.asarray(trajectory).flatten()
    N = len(trajectory)
    
    correlations = []
    
    for scale in scales:
        if N // scale < 10:  # Need sufficient points
            continue
            
        # Downsample by taking every scale-th point
        downsampled = trajectory[::scale]
        
        # Interpolate back to original length for comparison
        x_original = np.arange(N)
        x_downsampled = np.arange(0, N, scale)
        
        # Use linear interpolation
        interpolated = np.interp(x_original, x_downsampled, downsampled)
        
        # Calculate correlation
        corr = np.corrcoef(trajectory, interpolated)[0, 1]
        correlations.append(corr)
    
    correlations = np.array(correlations)
    mean_correlation = np.mean(correlations[np.isfinite(correlations)])
    
    return mean_correlation, correlations
