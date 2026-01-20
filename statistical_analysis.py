"""
Statistical Analysis Module
===========================
Functions for statistical testing and validation.

Methods implemented:
- Independent t-tests with effect sizes
- Cohen's d calculation
- Bootstrap confidence intervals
- Multiple comparison correction
"""

import numpy as np
from scipy import stats
from typing import Tuple, Dict, Optional, List


def independent_ttest(
    group1: np.ndarray,
    group2: np.ndarray,
    alpha: float = 0.05,
    alternative: str = 'two-sided'
) -> Dict:
    """
    Perform independent samples t-test with comprehensive output.
    
    Parameters
    ----------
    group1 : np.ndarray
        Data for first group
    group2 : np.ndarray
        Data for second group
    alpha : float, optional
        Significance level (default: 0.05)
    alternative : str, optional
        'two-sided', 'less', or 'greater' (default: 'two-sided')
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - t_statistic: t-value
        - p_value: p-value
        - df: degrees of freedom
        - cohens_d: effect size
        - significant: boolean
        - mean_diff: difference in means
        - ci_lower, ci_upper: confidence interval for difference
    
    Example
    -------
    >>> experts = np.array([1.67, 1.72, 1.65, 1.70, 1.68])
    >>> novices = np.array([1.23, 1.28, 1.20, 1.25, 1.22])
    >>> results = independent_ttest(experts, novices)
    >>> print(f"t({results['df']}) = {results['t_statistic']:.2f}, p < {results['p_value']:.3f}")
    """
    group1 = np.asarray(group1).flatten()
    group2 = np.asarray(group2).flatten()
    
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Perform t-test
    t_stat, p_val = stats.ttest_ind(group1, group2, alternative=alternative)
    
    # Degrees of freedom (Welch's approximation)
    df = (var1/n1 + var2/n2)**2 / (
        (var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1)
    )
    
    # Effect size
    d = cohens_d(group1, group2)
    
    # Confidence interval for mean difference
    mean_diff = mean1 - mean2
    se_diff = np.sqrt(var1/n1 + var2/n2)
    t_crit = stats.t.ppf(1 - alpha/2, df)
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff
    
    return {
        't_statistic': t_stat,
        'p_value': p_val,
        'df': df,
        'cohens_d': d,
        'significant': p_val < alpha,
        'mean_diff': mean_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'mean_group1': mean1,
        'mean_group2': mean2,
        'sd_group1': np.sqrt(var1),
        'sd_group2': np.sqrt(var2),
        'n_group1': n1,
        'n_group2': n2
    }


def cohens_d(
    group1: np.ndarray,
    group2: np.ndarray,
    pooled: bool = True
) -> float:
    """
    Calculate Cohen's d effect size.
    
    Parameters
    ----------
    group1 : np.ndarray
        Data for first group
    group2 : np.ndarray
        Data for second group
    pooled : bool, optional
        Use pooled standard deviation (default: True)
    
    Returns
    -------
    d : float
        Cohen's d effect size
    
    Notes
    -----
    Interpretation (Cohen, 1988):
    - |d| < 0.2: negligible
    - 0.2 ≤ |d| < 0.5: small
    - 0.5 ≤ |d| < 0.8: medium
    - |d| ≥ 0.8: large
    
    Example
    -------
    >>> experts = np.array([1.67, 1.72, 1.65])
    >>> novices = np.array([1.23, 1.28, 1.20])
    >>> d = cohens_d(experts, novices)
    >>> print(f"Cohen's d = {d:.2f}")
    """
    group1 = np.asarray(group1).flatten()
    group2 = np.asarray(group2).flatten()
    
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    if pooled:
        # Pooled standard deviation
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        sd_pooled = np.sqrt(pooled_var)
    else:
        # Use control group (group2) SD
        sd_pooled = np.sqrt(var2)
    
    d = (mean1 - mean2) / sd_pooled
    
    return d


def bootstrap_ci(
    data: np.ndarray,
    statistic: callable = np.mean,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_state: Optional[int] = None
) -> Tuple[float, float, float, np.ndarray]:
    """
    Calculate bootstrap confidence intervals.
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    statistic : callable, optional
        Function to compute statistic (default: np.mean)
    n_bootstrap : int, optional
        Number of bootstrap samples (default: 1000)
    ci_level : float, optional
        Confidence level (default: 0.95)
    random_state : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    point_estimate : float
        Point estimate of statistic
    ci_lower : float
        Lower confidence bound
    ci_upper : float
        Upper confidence bound
    bootstrap_distribution : np.ndarray
        Bootstrap sampling distribution
    
    Example
    -------
    >>> data = np.array([1.67, 1.72, 1.65, 1.70, 1.68, 1.63, 1.75, 1.69, 1.66, 1.71])
    >>> mean, ci_low, ci_high, dist = bootstrap_ci(data, n_bootstrap=1000)
    >>> print(f"Mean: {mean:.3f} [{ci_low:.3f}, {ci_high:.3f}]")
    """
    data = np.asarray(data).flatten()
    n = len(data)
    
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate bootstrap samples
    bootstrap_stats = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        # Resample with replacement
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats[i] = statistic(sample)
    
    # Calculate confidence interval (percentile method)
    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    point_estimate = statistic(data)
    
    return point_estimate, ci_lower, ci_upper, bootstrap_stats


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05
) -> Tuple[float, List[bool]]:
    """
    Apply Bonferroni correction for multiple comparisons.
    
    Parameters
    ----------
    p_values : list of float
        List of p-values from multiple tests
    alpha : float, optional
        Family-wise error rate (default: 0.05)
    
    Returns
    -------
    adjusted_alpha : float
        Bonferroni-adjusted significance threshold
    significant : list of bool
        Whether each test is significant after correction
    
    Example
    -------
    >>> p_vals = [0.01, 0.03, 0.04, 0.08]
    >>> adj_alpha, sig = bonferroni_correction(p_vals)
    >>> print(f"Adjusted alpha: {adj_alpha:.4f}")
    """
    n_tests = len(p_values)
    adjusted_alpha = alpha / n_tests
    
    significant = [p < adjusted_alpha for p in p_values]
    
    return adjusted_alpha, significant


def effect_size_interpretation(d: float) -> str:
    """
    Interpret Cohen's d effect size.
    
    Parameters
    ----------
    d : float
        Cohen's d value
    
    Returns
    -------
    interpretation : str
        Text description of effect size
    """
    d_abs = abs(d)
    
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def summary_statistics(
    data: np.ndarray,
    group_name: str = "Group"
) -> Dict:
    """
    Calculate comprehensive summary statistics.
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    group_name : str, optional
        Name for the group (default: "Group")
    
    Returns
    -------
    stats : dict
        Dictionary of summary statistics
    
    Example
    -------
    >>> data = np.array([1.67, 1.72, 1.65, 1.70, 1.68])
    >>> stats = summary_statistics(data, "Experts")
    """
    data = np.asarray(data).flatten()
    
    return {
        'group': group_name,
        'n': len(data),
        'mean': np.mean(data),
        'std': np.std(data, ddof=1),
        'sem': stats.sem(data),
        'median': np.median(data),
        'min': np.min(data),
        'max': np.max(data),
        'range': np.max(data) - np.min(data),
        'q1': np.percentile(data, 25),
        'q3': np.percentile(data, 75),
        'iqr': np.percentile(data, 75) - np.percentile(data, 25),
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data)
    }
