"""
Statistical utilities for rigorous experiment analysis.

Provides:
- Bootstrap confidence intervals
- Effect size calculations (Cohen's d)
- Cross-validation wrappers
"""

import numpy as np
from typing import Callable, Tuple, Dict, Any, List


def bootstrap_ci(
    data: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    statistic: Callable = np.mean,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        data: Array of observations
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level (0.95 = 95% CI)
        statistic: Function to compute (default: mean)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    np.random.seed(seed)
    data = np.asarray(data)
    n = len(data)

    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic(sample))

    alpha = (1 - ci) / 2
    lower = np.percentile(bootstrap_stats, alpha * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha) * 100)

    return (lower, upper)


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size.

    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large

    Args:
        group1: First group observations
        group2: Second group observations

    Returns:
        Cohen's d effect size
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std < 1e-10:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def run_with_cv(
    experiment_fn: Callable,
    n_folds: int = 10,
    n_bootstrap: int = 1000,
    seed_offset: int = 42,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run experiment with cross-validation and bootstrap CI.

    Args:
        experiment_fn: Function that takes seed and returns a numeric result
        n_folds: Number of folds (seeds)
        n_bootstrap: Bootstrap samples for CI
        seed_offset: Starting seed
        **kwargs: Additional arguments for experiment_fn

    Returns:
        Dict with mean, std, ci_95, raw_results, n_folds
    """
    results = []
    for fold in range(n_folds):
        seed = seed_offset + fold
        result = experiment_fn(seed=seed, **kwargs)
        results.append(result)

    results_array = np.array(results)

    return {
        "mean": np.mean(results_array),
        "std": np.std(results_array),
        "ci_95": bootstrap_ci(results_array, n_bootstrap=n_bootstrap),
        "raw_results": results_array,
        "n_folds": n_folds,
    }


def format_ci(mean: float, ci: Tuple[float, float], decimals: int = 2) -> str:
    """Format mean with CI for printing."""
    return f"{mean:.{decimals}f} (95% CI: [{ci[0]:.{decimals}f}, {ci[1]:.{decimals}f}])"


def compare_methods(
    results1: List[float],
    results2: List[float],
    method1_name: str = "Method 1",
    method2_name: str = "Method 2",
) -> Dict[str, Any]:
    """
    Compare two methods with statistical tests.

    Returns comprehensive comparison including:
    - Means and CIs
    - t-test results
    - Effect size
    """
    from scipy import stats

    results1 = np.asarray(results1)
    results2 = np.asarray(results2)

    t_stat, p_value = stats.ttest_ind(results1, results2)
    d = cohens_d(results1, results2)

    winner = method1_name if np.mean(results1) < np.mean(results2) else method2_name
    improvement = abs(np.mean(results1) - np.mean(results2)) / max(np.mean(results1), np.mean(results2)) * 100

    return {
        method1_name: {
            "mean": np.mean(results1),
            "std": np.std(results1),
            "ci_95": bootstrap_ci(results1),
        },
        method2_name: {
            "mean": np.mean(results2),
            "std": np.std(results2),
            "ci_95": bootstrap_ci(results2),
        },
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": d,
        "effect_size": interpret_cohens_d(d),
        "winner": winner,
        "improvement_pct": improvement,
        "significant": p_value < 0.05,
    }


def print_comparison(comparison: Dict[str, Any]) -> None:
    """Pretty print a method comparison."""
    methods = [k for k in comparison.keys() if isinstance(comparison[k], dict)]

    print("\n" + "=" * 60)
    print("STATISTICAL COMPARISON")
    print("=" * 60)

    for method in methods:
        stats = comparison[method]
        ci_str = format_ci(stats["mean"], stats["ci_95"])
        print(f"{method}: {ci_str}")

    print("-" * 60)
    print(f"t-statistic: {comparison['t_statistic']:.3f}")
    print(f"p-value: {comparison['p_value']:.6f}")
    print(f"Cohen's d: {comparison['cohens_d']:.3f} ({comparison['effect_size']})")
    print(f"Winner: {comparison['winner']} ({comparison['improvement_pct']:.1f}% better)")
    sig = "YES (p < 0.05)" if comparison["significant"] else "NO (p >= 0.05)"
    print(f"Significant: {sig}")
    print("=" * 60)
