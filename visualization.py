"""
Visualization Module
====================
Functions for generating publication-quality figures.

Includes:
- Trajectory plots
- Fractal dimension visualizations
- Phase-space plots
- Statistical comparison figures
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional, Tuple, List
import warnings


# Set default style for publication
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Color scheme
EXPERT_COLOR = '#E74C3C'  # Red
NOVICE_COLOR = '#3498DB'  # Blue
EXPERT_LIGHT = '#FADBD8'
NOVICE_LIGHT = '#D4E6F1'


def plot_trajectories(
    expert_traj: np.ndarray,
    novice_traj: np.ndarray,
    time: Optional[np.ndarray] = None,
    title: str = 'Movement Trajectories',
    xlabel: str = 'Time (s)',
    ylabel: str = 'Amplitude',
    figsize: Tuple[float, float] = (8, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot expert and novice trajectories for comparison.
    
    Parameters
    ----------
    expert_traj : np.ndarray
        Expert trajectory data
    novice_traj : np.ndarray
        Novice trajectory data
    time : np.ndarray, optional
        Time vector (default: auto-generate)
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size (width, height)
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if time is None:
        time_e = np.arange(len(expert_traj))
        time_n = np.arange(len(novice_traj))
    else:
        time_e = time_n = time
    
    ax.plot(time_e, expert_traj, color=EXPERT_COLOR, label='Expert', linewidth=1.5)
    ax.plot(time_n, novice_traj, color=NOVICE_COLOR, label='Novice', linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_fractal_dimension(
    log_scales: np.ndarray,
    log_counts: np.ndarray,
    D: float,
    r_squared: float,
    group_name: str = 'Group',
    color: str = EXPERT_COLOR,
    figsize: Tuple[float, float] = (5, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot log-log relationship for fractal dimension calculation.
    
    Parameters
    ----------
    log_scales : np.ndarray
        Log of box sizes
    log_counts : np.ndarray
        Log of box counts
    D : float
        Calculated fractal dimension
    r_squared : float
        R² value of fit
    group_name : str
        Name for legend
    color : str
        Plot color
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Data points
    ax.scatter(log_scales, log_counts, color=color, s=50, alpha=0.7, label='Data')
    
    # Fit line
    slope = -D
    intercept = np.mean(log_counts) - slope * np.mean(log_scales)
    fit_line = slope * log_scales + intercept
    ax.plot(log_scales, fit_line, color=color, linestyle='--', linewidth=2,
            label=f'Fit: D = {D:.2f}, R² = {r_squared:.3f}')
    
    ax.set_xlabel('log(Box Size)')
    ax.set_ylabel('log(Box Count)')
    ax.set_title(f'Fractal Dimension Analysis - {group_name}')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_phase_space(
    expert_error: np.ndarray,
    expert_complexity: np.ndarray,
    novice_error: np.ndarray,
    novice_complexity: np.ndarray,
    figsize: Tuple[float, float] = (6, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot phase-space with expert and novice attractors.
    
    Parameters
    ----------
    expert_error : np.ndarray
        Deterministic error measure for experts
    expert_complexity : np.ndarray
        Stochastic complexity measure for experts
    novice_error : np.ndarray
        Deterministic error for novices
    novice_complexity : np.ndarray
        Stochastic complexity for novices
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    ax.scatter(expert_error, expert_complexity, color=EXPERT_COLOR, 
               s=80, alpha=0.7, label='Expert', marker='o')
    ax.scatter(novice_error, novice_complexity, color=NOVICE_COLOR, 
               s=80, alpha=0.7, label='Novice', marker='s')
    
    # Add attractor centers
    expert_center = (np.mean(expert_error), np.mean(expert_complexity))
    novice_center = (np.mean(novice_error), np.mean(novice_complexity))
    
    ax.scatter(*expert_center, color=EXPERT_COLOR, s=200, marker='*', 
               edgecolor='black', linewidth=1.5, zorder=5)
    ax.scatter(*novice_center, color=NOVICE_COLOR, s=200, marker='*', 
               edgecolor='black', linewidth=1.5, zorder=5)
    
    # Add threshold lines
    ax.axvline(x=0.62, color='gray', linestyle='--', alpha=0.5, label='Error threshold')
    ax.axhline(y=0.6, color='gray', linestyle=':', alpha=0.5, label='Complexity threshold')
    
    ax.set_xlabel('Deterministic Error (normalized)')
    ax.set_ylabel('Stochastic Complexity (normalized)')
    ax.set_title('Phase Space: Expert vs Novice Attractors')
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_comparison_bars(
    expert_values: np.ndarray,
    novice_values: np.ndarray,
    metric_name: str,
    ylabel: str,
    figsize: Tuple[float, float] = (4, 5),
    show_individual: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create bar plot comparing expert and novice groups.
    
    Parameters
    ----------
    expert_values : np.ndarray
        Values for expert group
    novice_values : np.ndarray
        Values for novice group
    metric_name : str
        Name of the metric being compared
    ylabel : str
        Y-axis label
    show_individual : bool
        Whether to show individual data points
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    means = [np.mean(novice_values), np.mean(expert_values)]
    stds = [np.std(novice_values), np.std(expert_values)]
    
    x_pos = [0, 1]
    colors = [NOVICE_COLOR, EXPERT_COLOR]
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, 
                  alpha=0.7, edgecolor='black', linewidth=1.5)
    
    if show_individual:
        # Add individual data points
        jitter = 0.1
        for i, (values, color) in enumerate([(novice_values, NOVICE_COLOR), 
                                              (expert_values, EXPERT_COLOR)]):
            x_jittered = np.random.normal(i, jitter, len(values))
            ax.scatter(x_jittered, values, color='black', s=20, alpha=0.5, zorder=3)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Novice', 'Expert'])
    ax.set_ylabel(ylabel)
    ax.set_title(metric_name)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add significance marker if applicable
    from .statistical_analysis import independent_ttest
    result = independent_ttest(expert_values, novice_values)
    if result['p_value'] < 0.001:
        sig_text = '***'
    elif result['p_value'] < 0.01:
        sig_text = '**'
    elif result['p_value'] < 0.05:
        sig_text = '*'
    else:
        sig_text = 'n.s.'
    
    y_max = max(means) + max(stds) * 1.5
    ax.plot([0, 1], [y_max, y_max], color='black', linewidth=1)
    ax.text(0.5, y_max * 1.02, sig_text, ha='center', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_psd_comparison(
    expert_freqs: np.ndarray,
    expert_psd: np.ndarray,
    novice_freqs: np.ndarray,
    novice_psd: np.ndarray,
    max_freq: float = 30.0,
    figsize: Tuple[float, float] = (7, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot power spectral density comparison.
    
    Parameters
    ----------
    expert_freqs : np.ndarray
        Frequency values for expert
    expert_psd : np.ndarray
        PSD values for expert
    novice_freqs : np.ndarray
        Frequency values for novice
    novice_psd : np.ndarray
        PSD values for novice
    max_freq : float
        Maximum frequency to display
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter to max frequency
    expert_mask = expert_freqs <= max_freq
    novice_mask = novice_freqs <= max_freq
    
    ax.semilogy(expert_freqs[expert_mask], expert_psd[expert_mask], 
                color=EXPERT_COLOR, label='Expert', linewidth=1.5)
    ax.semilogy(novice_freqs[novice_mask], novice_psd[novice_mask], 
                color=NOVICE_COLOR, label='Novice', linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title('Frequency Domain Analysis')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, max_freq)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_self_similarity(
    scales: List[int],
    expert_corr: np.ndarray,
    novice_corr: np.ndarray,
    figsize: Tuple[float, float] = (5, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot self-similarity preservation across scales.
    
    Parameters
    ----------
    scales : list
        Scale factors used
    expert_corr : np.ndarray
        Correlations for expert
    novice_corr : np.ndarray
        Correlations for novice
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(scales[:len(expert_corr)], expert_corr, 'o-', color=EXPERT_COLOR, 
            label='Expert', linewidth=2, markersize=8)
    ax.plot(scales[:len(novice_corr)], novice_corr, 's-', color=NOVICE_COLOR, 
            label='Novice', linewidth=2, markersize=8)
    
    ax.set_xlabel('Scale Factor')
    ax.set_ylabel('Self-Similarity (Correlation)')
    ax.set_title('Self-Similarity Across Temporal Scales')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def create_summary_figure(
    results_dict: dict,
    figsize: Tuple[float, float] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create multi-panel summary figure for publication.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary containing all analysis results
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Panel labels
    panels = ['A', 'B', 'C', 'D', 'E', 'F']
    
    # Create placeholder panels
    for i, (panel_label, gs_pos) in enumerate(zip(panels, 
        [gs[0, 0], gs[0, 1], gs[0, 2], gs[1, 0], gs[1, 1], gs[1, 2]])):
        ax = fig.add_subplot(gs_pos)
        ax.text(-0.1, 1.1, panel_label, transform=ax.transAxes, 
                fontsize=14, fontweight='bold', va='top')
        ax.set_title(f'Panel {panel_label}')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig
