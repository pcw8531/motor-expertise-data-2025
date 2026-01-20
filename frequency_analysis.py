"""
Frequency Analysis Module
=========================
Functions for spectral analysis of movement data.

Methods implemented:
- Fast Fourier Transform (FFT)
- Power Spectral Density (PSD)
- Frequency bandwidth calculation
- Shannon entropy
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional


def compute_fft(
    time_series: np.ndarray,
    sampling_rate: float = 120.0,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Fast Fourier Transform of time series.
    
    Parameters
    ----------
    time_series : np.ndarray
        1D time series data
    sampling_rate : float, optional
        Sampling frequency in Hz (default: 120.0)
    normalize : bool, optional
        Whether to normalize amplitude (default: True)
    
    Returns
    -------
    frequencies : np.ndarray
        Frequency values in Hz
    amplitudes : np.ndarray
        FFT amplitudes (normalized if requested)
    
    Example
    -------
    >>> ts = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 120))  # 5 Hz sine
    >>> freqs, amps = compute_fft(ts, sampling_rate=120)
    >>> dominant_freq = freqs[np.argmax(amps)]
    >>> print(f"Dominant frequency: {dominant_freq:.1f} Hz")
    """
    time_series = np.asarray(time_series).flatten()
    N = len(time_series)
    
    # Remove mean (DC component)
    time_series = time_series - np.mean(time_series)
    
    # Compute FFT
    fft_values = np.fft.fft(time_series)
    
    # Get positive frequencies only
    frequencies = np.fft.fftfreq(N, d=1/sampling_rate)
    positive_mask = frequencies >= 0
    
    frequencies = frequencies[positive_mask]
    amplitudes = np.abs(fft_values[positive_mask])
    
    if normalize:
        amplitudes = amplitudes / N * 2  # Scale for single-sided spectrum
    
    return frequencies, amplitudes


def compute_psd(
    time_series: np.ndarray,
    sampling_rate: float = 120.0,
    method: str = 'welch',
    nperseg: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Power Spectral Density.
    
    Parameters
    ----------
    time_series : np.ndarray
        1D time series data
    sampling_rate : float, optional
        Sampling frequency in Hz (default: 120.0)
    method : str, optional
        PSD estimation method: 'welch' or 'periodogram' (default: 'welch')
    nperseg : int, optional
        Length of each segment for Welch method (default: N/4)
    
    Returns
    -------
    frequencies : np.ndarray
        Frequency values in Hz
    psd : np.ndarray
        Power spectral density values
    
    Example
    -------
    >>> ts = np.random.randn(1000)
    >>> freqs, psd = compute_psd(ts, sampling_rate=120)
    """
    time_series = np.asarray(time_series).flatten()
    N = len(time_series)
    
    if nperseg is None:
        nperseg = min(256, N // 4)
    
    if method == 'welch':
        frequencies, psd = signal.welch(
            time_series,
            fs=sampling_rate,
            nperseg=nperseg,
            noverlap=nperseg // 2
        )
    elif method == 'periodogram':
        frequencies, psd = signal.periodogram(
            time_series,
            fs=sampling_rate
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'welch' or 'periodogram'")
    
    return frequencies, psd


def calculate_bandwidth(
    frequencies: np.ndarray,
    psd: np.ndarray,
    power_threshold: float = 0.90
) -> Tuple[float, float, float]:
    """
    Calculate frequency bandwidth containing specified power.
    
    Parameters
    ----------
    frequencies : np.ndarray
        Frequency values in Hz
    psd : np.ndarray
        Power spectral density values
    power_threshold : float, optional
        Fraction of total power to capture (default: 0.90)
    
    Returns
    -------
    bandwidth : float
        Frequency bandwidth in Hz
    low_freq : float
        Lower frequency bound
    high_freq : float
        Upper frequency bound
    
    Example
    -------
    >>> freqs, psd = compute_psd(time_series)
    >>> bw, f_low, f_high = calculate_bandwidth(freqs, psd)
    >>> print(f"90% power bandwidth: {bw:.2f} Hz ({f_low:.2f} - {f_high:.2f})")
    """
    frequencies = np.asarray(frequencies)
    psd = np.asarray(psd)
    
    # Calculate cumulative power
    total_power = np.sum(psd)
    cumulative_power = np.cumsum(psd) / total_power
    
    # Find frequency bounds
    low_idx = np.searchsorted(cumulative_power, (1 - power_threshold) / 2)
    high_idx = np.searchsorted(cumulative_power, (1 + power_threshold) / 2)
    
    # Ensure valid indices
    low_idx = max(0, low_idx)
    high_idx = min(len(frequencies) - 1, high_idx)
    
    low_freq = frequencies[low_idx]
    high_freq = frequencies[high_idx]
    bandwidth = high_freq - low_freq
    
    return bandwidth, low_freq, high_freq


def shannon_entropy(
    time_series: np.ndarray,
    n_bins: int = 50,
    normalize: bool = True
) -> float:
    """
    Calculate Shannon entropy of time series.
    
    Parameters
    ----------
    time_series : np.ndarray
        1D time series data
    n_bins : int, optional
        Number of bins for histogram (default: 50)
    normalize : bool, optional
        Normalize by maximum entropy (default: True)
    
    Returns
    -------
    entropy : float
        Shannon entropy value
        If normalized, returns value between 0 and 1
    
    Notes
    -----
    Higher entropy indicates greater movement unpredictability.
    Experts typically show higher entropy despite better accuracy.
    
    Example
    -------
    >>> ts = np.random.randn(1000)
    >>> H = shannon_entropy(ts)
    >>> print(f"Shannon entropy: {H:.3f}")
    """
    time_series = np.asarray(time_series).flatten()
    
    # Create histogram
    counts, _ = np.histogram(time_series, bins=n_bins)
    
    # Calculate probabilities
    probabilities = counts / np.sum(counts)
    
    # Remove zeros to avoid log(0)
    probabilities = probabilities[probabilities > 0]
    
    # Calculate entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    if normalize:
        max_entropy = np.log2(n_bins)
        entropy = entropy / max_entropy
    
    return entropy


def dominant_frequencies(
    frequencies: np.ndarray,
    amplitudes: np.ndarray,
    n_peaks: int = 5,
    min_prominence: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find dominant frequencies in spectrum.
    
    Parameters
    ----------
    frequencies : np.ndarray
        Frequency values in Hz
    amplitudes : np.ndarray
        FFT amplitudes or PSD values
    n_peaks : int, optional
        Maximum number of peaks to return (default: 5)
    min_prominence : float, optional
        Minimum prominence as fraction of max (default: 0.1)
    
    Returns
    -------
    peak_frequencies : np.ndarray
        Frequencies of dominant peaks
    peak_amplitudes : np.ndarray
        Amplitudes at peak frequencies
    
    Example
    -------
    >>> freqs, amps = compute_fft(time_series)
    >>> peak_f, peak_a = dominant_frequencies(freqs, amps)
    """
    frequencies = np.asarray(frequencies)
    amplitudes = np.asarray(amplitudes)
    
    # Find peaks
    prominence_threshold = min_prominence * np.max(amplitudes)
    peaks, properties = signal.find_peaks(
        amplitudes,
        prominence=prominence_threshold
    )
    
    if len(peaks) == 0:
        # Return maximum if no peaks found
        max_idx = np.argmax(amplitudes)
        return np.array([frequencies[max_idx]]), np.array([amplitudes[max_idx]])
    
    # Sort by amplitude
    sorted_indices = np.argsort(amplitudes[peaks])[::-1]
    top_peaks = peaks[sorted_indices[:n_peaks]]
    
    peak_frequencies = frequencies[top_peaks]
    peak_amplitudes = amplitudes[top_peaks]
    
    return peak_frequencies, peak_amplitudes
