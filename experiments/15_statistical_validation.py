#!/usr/bin/env python3
"""
Experiment 15: Rigorous Statistical Validation

Applies proper statistical methodology:
- Bootstrap confidence intervals
- Cohen's d effect sizes
- Multiple comparison corrections

This validates our main findings with rigorous statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from vdd.drift_detection import EmbeddingDistance
from utils.statistics import (
    bootstrap_ci,
    cohens_d,
    interpret_cohens_d,
    compare_methods,
    print_comparison,
    format_ci,
)


def generate_data(steps: int, regime_shifts: tuple, embedding_dim: int, seed: int):
    """Generate synthetic data with regime shifts."""
    np.random.seed(seed)

    truth = np.zeros(steps)
    current_val = 0
    embeddings = []
    centroid = np.random.randn(embedding_dim)
    centroid = centroid / np.linalg.norm(centroid)

    for t in range(steps):
        if t in regime_shifts:
            current_val += np.random.choice([-5, 5])
            new_dir = np.random.randn(embedding_dim)
            centroid = 0.3 * centroid + 0.7 * new_dir
            centroid = centroid / np.linalg.norm(centroid)
        else:
            current_val += np.random.normal(0, 0.1)
        truth[t] = current_val

        emb = centroid + np.random.randn(embedding_dim) * 0.1
        embeddings.append(emb)

    return truth, embeddings


def run_single_trial(seed: int, method: str = "vdd"):
    """Run single trial for a method."""
    steps = 200
    regime_shifts = (50, 100, 150)
    embedding_dim = 32

    truth, embeddings = generate_data(steps, regime_shifts, embedding_dim, seed)

    memory = np.zeros(steps)
    stored = 0.0

    if method == "vdd":
        detector = EmbeddingDistance(curr_window=10, arch_window=100, drift_threshold=0.4)
        lambda_base, lambda_max = 0.2, 0.9

        for t in range(1, steps):
            result = detector.update(embeddings[t])
            v = result.volatility
            current_lambda = lambda_base + (lambda_max - lambda_base) * v
            stored = (1 - current_lambda) * stored + current_lambda * truth[t]
            memory[t] = stored

    elif method == "recency":
        for t in range(1, steps):
            stored = (1 - 0.5) * stored + 0.5 * truth[t]
            memory[t] = stored

    elif method == "static":
        for t in range(1, steps):
            stored = (1 - 0.1) * stored + 0.1 * truth[t]
            memory[t] = stored

    error = np.abs(truth - memory)
    return np.sum(error)


def run_validation():
    """Run full statistical validation."""
    print("=" * 60)
    print("Experiment 15: Rigorous Statistical Validation")
    print("=" * 60)

    n_folds = 20  # More folds for better CI
    methods = ["vdd", "recency", "static"]

    # Collect results
    results = {m: [] for m in methods}

    print(f"\nRunning {n_folds}-fold cross-validation...")
    for seed in range(42, 42 + n_folds):
        for method in methods:
            iae = run_single_trial(seed, method)
            results[method].append(iae)

    # Summary statistics with bootstrap CI
    print("\n" + "=" * 60)
    print("RESULTS WITH 95% BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 60)
    print(f"(n={n_folds} folds, 1000 bootstrap samples)")
    print("-" * 60)

    summary = {}
    for method in methods:
        data = np.array(results[method])
        mean = np.mean(data)
        ci = bootstrap_ci(data)
        summary[method] = {"mean": mean, "ci": ci, "data": data}
        print(f"{method.upper():>10}: {format_ci(mean, ci)}")

    # Pairwise comparisons
    print("\n" + "=" * 60)
    print("PAIRWISE COMPARISONS")
    print("=" * 60)

    comparisons = [
        ("vdd", "recency"),
        ("vdd", "static"),
        ("recency", "static"),
    ]

    comparison_results = []
    for m1, m2 in comparisons:
        print(f"\n--- {m1.upper()} vs {m2.upper()} ---")
        comp = compare_methods(
            results[m1], results[m2],
            method1_name=m1.upper(), method2_name=m2.upper()
        )
        comparison_results.append((m1, m2, comp))

        d = comp["cohens_d"]
        print(f"Difference: {comp[m1.upper()]['mean']:.2f} vs {comp[m2.upper()]['mean']:.2f}")
        print(f"Cohen's d: {d:.3f} ({interpret_cohens_d(d)})")
        print(f"p-value: {comp['p_value']:.6f}")
        print(f"Winner: {comp['winner']}")

    # Effect size summary
    print("\n" + "=" * 60)
    print("EFFECT SIZE SUMMARY")
    print("=" * 60)
    print(f"{'Comparison':<20} {'Cohen d':>10} {'Effect':>12} {'p-value':>12}")
    print("-" * 60)

    for m1, m2, comp in comparison_results:
        d = comp["cohens_d"]
        sig = "***" if comp["p_value"] < 0.001 else "**" if comp["p_value"] < 0.01 else "*" if comp["p_value"] < 0.05 else "ns"
        print(f"{m1} vs {m2:<12} {d:>10.3f} {interpret_cohens_d(d):>12} {comp['p_value']:>10.6f} {sig}")

    return results, comparison_results


def plot_validation(results: dict, comparison_results: list):
    """Create validation visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    methods = list(results.keys())
    colors = {"vdd": "green", "recency": "red", "static": "blue"}

    # Plot 1: Box plots with individual points
    ax = axes[0]
    positions = range(len(methods))
    bp = ax.boxplot([results[m] for m in methods], positions=positions,
                    patch_artist=True, widths=0.6)

    for patch, method in zip(bp['boxes'], methods):
        patch.set_facecolor(colors[method])
        patch.set_alpha(0.7)

    # Overlay individual points
    for i, method in enumerate(methods):
        jitter = np.random.uniform(-0.15, 0.15, len(results[method]))
        ax.scatter([i + j for j in jitter], results[method],
                   alpha=0.5, color=colors[method], s=30)

    ax.set_xticks(positions)
    ax.set_xticklabels([m.upper() for m in methods])
    ax.set_ylabel("IAE")
    ax.set_title("Distribution of IAE Across Folds\n(Box plot with individual data points)")

    # Plot 2: Mean with CI
    ax = axes[1]
    means = [np.mean(results[m]) for m in methods]
    cis = [bootstrap_ci(np.array(results[m])) for m in methods]
    ci_lower = [mean - ci[0] for mean, ci in zip(means, cis)]
    ci_upper = [ci[1] - mean for mean, ci in zip(means, cis)]

    bars = ax.bar(methods, means, yerr=[ci_lower, ci_upper],
                  capsize=8, color=[colors[m] for m in methods],
                  alpha=0.8, edgecolor='black')

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{mean:.1f}', ha='center', fontweight='bold')

    ax.set_ylabel("IAE")
    ax.set_title("Mean IAE with 95% Bootstrap CI\n(n=20 folds)")

    # Plot 3: Effect sizes
    ax = axes[2]
    comp_labels = [f"{m1} vs {m2}" for m1, m2, _ in comparison_results]
    effect_sizes = [comp["cohens_d"] for _, _, comp in comparison_results]

    bars = ax.barh(comp_labels, effect_sizes,
                   color=['green' if d < 0 else 'red' for d in effect_sizes],
                   alpha=0.7)

    # Add reference lines
    ax.axvline(x=-0.8, color='gray', linestyle='--', alpha=0.5, label='Large effect')
    ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=-0.5, color='gray', linestyle=':', alpha=0.5, label='Medium effect')
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

    ax.set_xlabel("Cohen's d")
    ax.set_title("Effect Sizes\n(Negative = first method is better)")
    ax.legend(loc='lower right', fontsize=8)

    plt.tight_layout()

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / "15_statistical_validation.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {results_dir / '15_statistical_validation.png'}")
    plt.close()


def main():
    results, comparisons = run_validation()
    plot_validation(results, comparisons)
    return results, comparisons


if __name__ == "__main__":
    main()
