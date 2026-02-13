#!/usr/bin/env python3
"""
Experiment 08: Baseline Comparison

Compare VDD against multiple baselines with statistical significance testing.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vdd.drift_detection import EmbeddingDistance


def run_method(method: str, truth: np.ndarray, embeddings: list, regime_shifts: tuple):
    """Run a specific method and return error."""
    steps = len(truth)
    memory = np.zeros(steps)
    stored = 0

    if method == "vdd":
        detector = EmbeddingDistance(curr_window=5, arch_window=50, drift_threshold=0.4)
        lambda_base, lambda_max = 0.2, 0.9
    elif method.startswith("static_"):
        lam = float(method.split("_")[1])
        detector = None
    elif method == "recency":
        detector = None
    elif method == "no_decay":
        detector = None
    else:
        raise ValueError(f"Unknown method: {method}")

    for t in range(1, steps):
        if method == "vdd":
            result = detector.update(embeddings[t])
            v = result.volatility
            current_lambda = lambda_base + (lambda_max - lambda_base) * v
        elif method.startswith("static_"):
            current_lambda = float(method.split("_")[1])
        elif method == "recency":
            current_lambda = 0.5  # Heavy recency bias
        elif method == "no_decay":
            current_lambda = 0.1  # Very slow decay (almost no forgetting)

        stored = (1 - current_lambda) * stored + current_lambda * truth[t]
        memory[t] = stored

    error = np.abs(truth - memory)
    iae = np.sum(error)
    mae = np.mean(error)

    return {"iae": iae, "mae": mae, "error": error}


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


def run_comparison():
    """Run full baseline comparison."""
    print("=" * 60)
    print("Experiment 08: Baseline Comparison")
    print("=" * 60)

    steps = 200
    regime_shifts = (50, 100, 150)
    embedding_dim = 32
    n_runs = 10

    methods = [
        "vdd",
        "static_0.05",
        "static_0.10",
        "static_0.20",
        "static_0.30",
        "recency",
        "no_decay",
    ]

    # Collect results across multiple runs
    results = {m: {"iae": [], "mae": []} for m in methods}

    print(f"\nRunning {n_runs} trials for each method...")
    for seed in range(42, 42 + n_runs):
        truth, embeddings = generate_data(steps, regime_shifts, embedding_dim, seed)

        for method in methods:
            res = run_method(method, truth, embeddings, regime_shifts)
            results[method]["iae"].append(res["iae"])
            results[method]["mae"].append(res["mae"])

    # Compute statistics
    print(f"\n{'='*60}")
    print("RESULTS (mean ± std over 10 runs)")
    print(f"{'='*60}")
    print(f"{'Method':<15} {'IAE':<20} {'MAE':<20}")
    print("-" * 55)

    for method in methods:
        iae_mean = np.mean(results[method]["iae"])
        iae_std = np.std(results[method]["iae"])
        mae_mean = np.mean(results[method]["mae"])
        mae_std = np.std(results[method]["mae"])
        print(f"{method:<15} {iae_mean:>8.2f} ± {iae_std:<8.2f} {mae_mean:>8.4f} ± {mae_std:<8.4f}")

    # Statistical significance tests (VDD vs each baseline)
    print(f"\n{'='*60}")
    print("STATISTICAL SIGNIFICANCE (t-test, VDD vs baseline)")
    print(f"{'='*60}")
    print(f"{'Comparison':<25} {'t-stat':<12} {'p-value':<12} {'Significant?':<12}")
    print("-" * 60)

    vdd_iae = results["vdd"]["iae"]
    for method in methods[1:]:  # Skip VDD
        baseline_iae = results[method]["iae"]
        t_stat, p_value = stats.ttest_ind(vdd_iae, baseline_iae)
        significant = "YES" if p_value < 0.05 and np.mean(vdd_iae) < np.mean(baseline_iae) else "NO"
        print(f"VDD vs {method:<18} {t_stat:>8.3f}    {p_value:>8.4f}     {significant}")

    # Best baseline
    baseline_means = {m: np.mean(results[m]["iae"]) for m in methods[1:]}
    best_baseline = min(baseline_means, key=baseline_means.get)
    vdd_mean = np.mean(results["vdd"]["iae"])
    improvement = (baseline_means[best_baseline] - vdd_mean) / baseline_means[best_baseline] * 100

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Best baseline: {best_baseline} (IAE={baseline_means[best_baseline]:.2f})")
    print(f"VDD IAE: {vdd_mean:.2f}")
    print(f"Improvement over best baseline: {improvement:.1f}%")

    return results


def plot_results(results: dict, save_path: str = None):
    """Plot comparison results with 95% confidence intervals."""
    methods = list(results.keys())
    iae_means = [np.mean(results[m]["iae"]) for m in methods]
    iae_stds = [np.std(results[m]["iae"]) for m in methods]
    n = len(results[methods[0]]["iae"])
    iae_cis = [1.96 * s / np.sqrt(n) for s in iae_stds]  # 95% CI

    fig, ax = plt.subplots(figsize=(12, 6))

    # Color scheme
    color_map = {
        'vdd': '#2ecc71',  # Green
        'recency': '#e74c3c',  # Red
        'no_decay': '#9b59b6',  # Purple
    }
    colors = [color_map.get(m, '#95a5a6') for m in methods]

    bars = ax.bar(range(len(methods)), iae_means, yerr=iae_cis, capsize=6,
                  color=colors, alpha=0.85, edgecolor='black', linewidth=1)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=45, ha="right")
    ax.set_ylabel("Integrated Absolute Error (IAE)")
    ax.set_title("VDD vs Baselines: Error Comparison\n(Error bars = 95% CI, n=10 runs, Lower is Better)")

    # Add value labels
    for bar, mean, ci in zip(bars, iae_means, iae_cis):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ci + 2,
                f"{mean:.1f}", ha="center", va="bottom", fontsize=9, fontweight='bold')

    # Add horizontal line at VDD level
    vdd_idx = methods.index('vdd')
    ax.axhline(y=iae_means[vdd_idx], color='green', linestyle='--', alpha=0.5, label='VDD baseline')

    ax.set_ylim(0, max(iae_means) * 1.15)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.close()


def main():
    """Main entry point."""
    results = run_comparison()

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    plot_results(results, str(results_dir / "08_baseline_comparison.png"))

    return results


if __name__ == "__main__":
    main()
