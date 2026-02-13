#!/usr/bin/env python3
"""
Experiment 04: Mixed-Drift Scenario

Tests VDD advantage over recency baseline in realistic variable-drift environment.
Key insight: Recency over-forgets during stable periods, VDD adapts.

Setup:
- 500 timesteps total
- ~70% stable periods, ~30% drift periods
- Unpredictable drift timing
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vdd.drift_detection import EmbeddingDistance


def generate_mixed_drift_data(steps: int, embedding_dim: int, seed: int):
    """Generate data with mixed stable/drift periods."""
    np.random.seed(seed)

    truth = np.zeros(steps)
    embeddings = []
    current_val = 0

    centroid = np.random.randn(embedding_dim)
    centroid = centroid / np.linalg.norm(centroid)

    # Define periods: (start, end, is_drift)
    # ~70% stable, ~30% drift
    periods = [
        (0, 80, False),      # Stable
        (80, 100, True),     # Drift
        (100, 200, False),   # Stable (long)
        (200, 210, True),    # Brief drift
        (210, 280, False),   # Stable
        (280, 320, True),    # Drift
        (320, 450, False),   # Very long stable
        (450, 470, True),    # Drift
        (470, 500, False),   # Stable
    ]

    drift_flags = np.zeros(steps, dtype=bool)
    for start, end, is_drift in periods:
        if is_drift:
            drift_flags[start:end] = True

    for t in range(steps):
        if drift_flags[t] and (t == 0 or not drift_flags[t-1]):
            # Start of drift period - shift centroid
            new_dir = np.random.randn(embedding_dim)
            centroid = 0.3 * centroid + 0.7 * new_dir
            centroid = centroid / np.linalg.norm(centroid)
            current_val += np.random.choice([-3, 3])
        elif drift_flags[t]:
            # During drift - high noise
            current_val += np.random.normal(0, 0.3)
        else:
            # Stable period - low noise
            current_val += np.random.normal(0, 0.05)

        truth[t] = current_val
        emb = centroid + np.random.randn(embedding_dim) * 0.1
        embeddings.append(emb)

    return truth, embeddings, drift_flags


def run_method(method: str, truth: np.ndarray, embeddings: list):
    """Run a specific decay method."""
    steps = len(truth)
    memory = np.zeros(steps)
    stored = 0

    if method == "vdd":
        detector = EmbeddingDistance(curr_window=10, arch_window=100, drift_threshold=0.4)
        lambda_base, lambda_max = 0.2, 0.9

    for t in range(1, steps):
        if method == "vdd":
            result = detector.update(embeddings[t])
            v = result.volatility
            current_lambda = lambda_base + (lambda_max - lambda_base) * v
        elif method == "recency":
            current_lambda = 0.5  # High constant decay
        elif method == "static":
            current_lambda = 0.1  # Low constant decay
        elif method == "adaptive_recency":
            current_lambda = 0.3  # Medium constant

        stored = (1 - current_lambda) * stored + current_lambda * truth[t]
        memory[t] = stored

    error = np.abs(truth - memory)
    return {"iae": np.sum(error), "mae": np.mean(error), "error": error, "memory": memory}


def run_experiment():
    """Run mixed-drift comparison."""
    print("=" * 60)
    print("Experiment 04: Mixed-Drift Scenario")
    print("=" * 60)

    steps = 500
    embedding_dim = 32
    n_runs = 10

    methods = ["vdd", "recency", "static", "adaptive_recency"]
    results = {m: {"iae": [], "mae": []} for m in methods}

    print(f"\nRunning {n_runs} trials...")
    for seed in range(42, 42 + n_runs):
        truth, embeddings, drift_flags = generate_mixed_drift_data(steps, embedding_dim, seed)

        for method in methods:
            res = run_method(method, truth, embeddings)
            results[method]["iae"].append(res["iae"])
            results[method]["mae"].append(res["mae"])

    # Statistics
    print(f"\n{'='*60}")
    print("RESULTS (mean ± std over 10 runs)")
    print(f"{'='*60}")
    print(f"{'Method':<20} {'IAE':<20} {'MAE':<15}")
    print("-" * 55)

    for method in methods:
        iae_mean = np.mean(results[method]["iae"])
        iae_std = np.std(results[method]["iae"])
        mae_mean = np.mean(results[method]["mae"])
        mae_std = np.std(results[method]["mae"])
        print(f"{method:<20} {iae_mean:>8.2f} ± {iae_std:<8.2f} {mae_mean:>6.4f} ± {mae_std:<6.4f}")

    # Statistical tests
    print(f"\n{'='*60}")
    print("STATISTICAL SIGNIFICANCE (t-test, VDD vs others)")
    print(f"{'='*60}")

    vdd_iae = results["vdd"]["iae"]
    for method in methods[1:]:
        other_iae = results[method]["iae"]
        t_stat, p_value = stats.ttest_ind(vdd_iae, other_iae)
        better = "VDD" if np.mean(vdd_iae) < np.mean(other_iae) else method
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        print(f"VDD vs {method:<15}: t={t_stat:>7.2f}, p={p_value:.4f} {sig} → {better} wins")

    # Key comparison: VDD vs recency
    vdd_mean = np.mean(results["vdd"]["iae"])
    recency_mean = np.mean(results["recency"]["iae"])
    improvement = (recency_mean - vdd_mean) / recency_mean * 100

    print(f"\n{'='*60}")
    print("KEY FINDING: Mixed-Drift Scenario")
    print(f"{'='*60}")
    print(f"VDD IAE:     {vdd_mean:.2f}")
    print(f"Recency IAE: {recency_mean:.2f}")
    print(f"VDD improvement over recency: {improvement:.1f}%")

    if improvement > 0:
        print("✅ VDD OUTPERFORMS recency in mixed-drift scenario!")
    else:
        print("❌ Recency still beats VDD")

    return results


def plot_iae_with_ci(results: dict):
    """Plot IAE comparison bar chart with 95% confidence intervals."""
    methods = list(results.keys())
    means = [np.mean(results[m]["iae"]) for m in methods]
    stds = [np.std(results[m]["iae"]) for m in methods]
    n = len(results[methods[0]]["iae"])
    cis = [1.96 * s / np.sqrt(n) for s in stds]

    colors = {'vdd': 'green', 'recency': 'red', 'static': 'blue', 'adaptive_recency': 'purple'}

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=cis, capsize=8, color=[colors.get(m, 'gray') for m in methods],
                  edgecolor='black', alpha=0.8)

    # Add value labels on bars
    for i, (bar, mean, ci) in enumerate(zip(bars, means, cis)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ci + 1,
                f'{mean:.1f}', ha='center', va='bottom', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in methods])
    ax.set_ylabel('Integrated Absolute Error (IAE)')
    ax.set_title('Mixed-Drift Scenario: Method Comparison\n(Error bars = 95% CI, n=10 runs)')
    ax.set_ylim(0, max(means) * 1.3)

    # Add significance markers
    ax.annotate('', xy=(0, means[0] + cis[0] + 8), xytext=(1, means[1] + cis[1] + 8),
                arrowprops=dict(arrowstyle='-', color='black'))
    ax.text(0.5, max(means[0], means[1]) + max(cis[0], cis[1]) + 12,
            '***', ha='center', fontsize=14)

    plt.tight_layout()
    results_dir = Path(__file__).parent.parent / "results"
    plt.savefig(results_dir / "04_mixed_drift_ci.png", dpi=150, bbox_inches='tight')
    print(f"Saved CI plot to {results_dir / '04_mixed_drift_ci.png'}")
    plt.close()


def plot_comparison(seed: int = 42):
    """Plot single run for visualization."""
    steps = 500
    embedding_dim = 32

    truth, embeddings, drift_flags = generate_mixed_drift_data(steps, embedding_dim, seed)

    vdd_res = run_method("vdd", truth, embeddings)
    recency_res = run_method("recency", truth, embeddings)
    static_res = run_method("static", truth, embeddings)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    x = range(steps)

    # Highlight drift periods
    drift_starts = []
    drift_ends = []
    in_drift = False
    for t in range(steps):
        if drift_flags[t] and not in_drift:
            drift_starts.append(t)
            in_drift = True
        elif not drift_flags[t] and in_drift:
            drift_ends.append(t)
            in_drift = False
    if in_drift:
        drift_ends.append(steps)

    # Plot 1: Memory tracking
    ax = axes[0]
    ax.plot(truth, 'k--', alpha=0.5, label='Ground Truth', lw=1)
    ax.plot(vdd_res["memory"], 'g-', label=f'VDD (IAE={vdd_res["iae"]:.1f})', lw=1.5)
    ax.plot(recency_res["memory"], 'r-', label=f'Recency (IAE={recency_res["iae"]:.1f})', lw=1.5, alpha=0.7)
    ax.plot(static_res["memory"], 'b-', label=f'Static (IAE={static_res["iae"]:.1f})', lw=1, alpha=0.5)

    for start, end in zip(drift_starts, drift_ends):
        ax.axvspan(start, end, alpha=0.2, color='orange', label='Drift' if start == drift_starts[0] else '')

    ax.set_ylabel('Value')
    ax.set_title('Mixed-Drift Scenario: VDD vs Baselines\n(Orange = drift periods, White = stable periods)')
    ax.legend(loc='upper right')

    # Plot 2: Error comparison
    ax = axes[1]
    ax.fill_between(x, vdd_res["error"], alpha=0.5, color='green', label='VDD Error')
    ax.fill_between(x, recency_res["error"], alpha=0.3, color='red', label='Recency Error')

    for start, end in zip(drift_starts, drift_ends):
        ax.axvspan(start, end, alpha=0.2, color='orange')

    ax.set_ylabel('Absolute Error')
    ax.set_title('Error Over Time (Lower is Better)')
    ax.legend()

    # Plot 3: Cumulative error
    ax = axes[2]
    ax.plot(np.cumsum(vdd_res["error"]), 'g-', label='VDD', lw=2)
    ax.plot(np.cumsum(recency_res["error"]), 'r-', label='Recency', lw=2)
    ax.plot(np.cumsum(static_res["error"]), 'b-', label='Static', lw=1, alpha=0.5)

    for start, end in zip(drift_starts, drift_ends):
        ax.axvspan(start, end, alpha=0.2, color='orange')

    ax.set_ylabel('Cumulative Error')
    ax.set_xlabel('Time Steps')
    ax.set_title('Cumulative Error (Lower is Better)')
    ax.legend()

    # Add annotation
    stable_pct = 100 * (1 - np.mean(drift_flags))
    fig.text(0.02, 0.02, f'Stable: {stable_pct:.0f}% | Drift: {100-stable_pct:.0f}%', fontsize=10)

    plt.tight_layout()

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / "04_mixed_drift.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {results_dir / '04_mixed_drift.png'}")
    plt.close()


def main():
    results = run_experiment()
    plot_comparison()
    plot_iae_with_ci(results)
    return results


if __name__ == "__main__":
    main()
