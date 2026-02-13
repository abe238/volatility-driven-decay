#!/usr/bin/env python3
"""
Experiment 12: Gradual Drift Patterns

Tests VDD on gradual transitions instead of sharp regime changes.
Hypothesis: VDD's sigmoid activation should handle gradual changes smoothly.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vdd.drift_detection import EmbeddingDistance


def generate_gradual_drift_data(steps: int, embedding_dim: int, transition_length: int, seed: int):
    """Generate data with gradual regime transitions."""
    np.random.seed(seed)

    truth = np.zeros(steps)
    embeddings = []

    # Define regime centers
    regime_A = np.random.randn(embedding_dim)
    regime_A = regime_A / np.linalg.norm(regime_A)
    regime_B = np.random.randn(embedding_dim)
    regime_B = regime_B / np.linalg.norm(regime_B)

    current_val = 0.0
    target_val = 0.0

    for t in range(steps):
        # Regime schedule: A -> gradual -> B -> stable -> gradual -> A
        if t < 100:
            # Regime A (stable)
            centroid = regime_A
            target_val = 0
        elif t < 100 + transition_length:
            # Gradual transition A → B
            alpha = (t - 100) / transition_length
            centroid = (1 - alpha) * regime_A + alpha * regime_B
            centroid = centroid / np.linalg.norm(centroid)
            target_val = alpha * 5
        elif t < 300:
            # Regime B (stable)
            centroid = regime_B
            target_val = 5
        elif t < 300 + transition_length:
            # Gradual transition B → A
            alpha = (t - 300) / transition_length
            centroid = (1 - alpha) * regime_B + alpha * regime_A
            centroid = centroid / np.linalg.norm(centroid)
            target_val = 5 - alpha * 5
        else:
            # Back to A (stable)
            centroid = regime_A
            target_val = 0

        # Smooth value transition
        current_val = 0.9 * current_val + 0.1 * target_val
        truth[t] = current_val + np.random.normal(0, 0.1)

        # Add noise to embedding
        emb = centroid + np.random.randn(embedding_dim) * 0.1
        embeddings.append(emb)

    return truth, embeddings


def run_method(method: str, truth: np.ndarray, embeddings: list):
    """Run a specific decay method."""
    steps = len(truth)
    memory = np.zeros(steps)
    stored = 0.0
    lambdas = []

    if method == "vdd":
        detector = EmbeddingDistance(curr_window=10, arch_window=100, drift_threshold=0.4)
        lambda_base, lambda_max = 0.2, 0.9

    for t in range(1, steps):
        if method == "vdd":
            result = detector.update(embeddings[t])
            v = result.volatility
            current_lambda = lambda_base + (lambda_max - lambda_base) * v
        elif method == "recency":
            current_lambda = 0.5
        elif method == "static":
            current_lambda = 0.1

        lambdas.append(current_lambda)
        stored = (1 - current_lambda) * stored + current_lambda * truth[t]
        memory[t] = stored

    error = np.abs(truth - memory)
    return {
        "iae": np.sum(error),
        "mae": np.mean(error),
        "error": error,
        "memory": memory,
        "lambdas": lambdas if method == "vdd" else None,
    }


def run_experiment():
    """Run gradual drift comparison."""
    print("=" * 60)
    print("Experiment 12: Gradual Drift Patterns")
    print("=" * 60)

    steps = 500
    embedding_dim = 32
    n_runs = 10
    transition_lengths = [10, 25, 50, 100]

    methods = ["vdd", "recency", "static"]
    all_results = {tl: {m: {"iae": [], "mae": []} for m in methods} for tl in transition_lengths}

    print(f"\nRunning {n_runs} trials for each transition length...")

    for tl in transition_lengths:
        print(f"\n  Transition length: {tl} steps")
        for seed in range(42, 42 + n_runs):
            truth, embeddings = generate_gradual_drift_data(steps, embedding_dim, tl, seed)

            for method in methods:
                res = run_method(method, truth, embeddings)
                all_results[tl][method]["iae"].append(res["iae"])
                all_results[tl][method]["mae"].append(res["mae"])

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS BY TRANSITION LENGTH")
    print(f"{'='*60}")

    for tl in transition_lengths:
        print(f"\nTransition length: {tl} steps")
        print(f"{'Method':<15} {'IAE':<20} {'MAE':<15}")
        print("-" * 50)
        for method in methods:
            iae_mean = np.mean(all_results[tl][method]["iae"])
            iae_std = np.std(all_results[tl][method]["iae"])
            mae_mean = np.mean(all_results[tl][method]["mae"])
            print(f"{method:<15} {iae_mean:>8.2f} ± {iae_std:<8.2f} {mae_mean:>8.4f}")

    # Statistical comparison: VDD vs recency by transition length
    print(f"\n{'='*60}")
    print("VDD vs RECENCY BY TRANSITION LENGTH")
    print(f"{'='*60}")

    for tl in transition_lengths:
        vdd_iae = all_results[tl]["vdd"]["iae"]
        rec_iae = all_results[tl]["recency"]["iae"]
        t_stat, p_value = stats.ttest_ind(vdd_iae, rec_iae)
        winner = "VDD" if np.mean(vdd_iae) < np.mean(rec_iae) else "Recency"
        improvement = (np.mean(rec_iae) - np.mean(vdd_iae)) / np.mean(rec_iae) * 100
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        print(f"TL={tl:>3}: VDD={np.mean(vdd_iae):.1f} vs Rec={np.mean(rec_iae):.1f} "
              f"| {winner} wins ({improvement:+.1f}%) {sig}")

    return all_results


def plot_results(all_results: dict, transition_lengths: list):
    """Visualize gradual drift results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    methods = ["vdd", "recency", "static"]
    colors = {"vdd": "green", "recency": "red", "static": "blue"}
    n = 10

    # Plot 1: IAE by transition length
    ax = axes[0, 0]
    x = np.arange(len(transition_lengths))
    width = 0.25

    for i, method in enumerate(methods):
        means = [np.mean(all_results[tl][method]["iae"]) for tl in transition_lengths]
        stds = [np.std(all_results[tl][method]["iae"]) for tl in transition_lengths]
        cis = [1.96 * s / np.sqrt(n) for s in stds]
        ax.bar(x + i * width, means, width, yerr=cis, label=method.upper(),
               color=colors[method], alpha=0.8, capsize=4)

    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{tl} steps" for tl in transition_lengths])
    ax.set_ylabel("IAE")
    ax.set_xlabel("Transition Length")
    ax.set_title("IAE by Transition Length\n(Error bars = 95% CI)")
    ax.legend()

    # Plot 2: VDD improvement over recency
    ax = axes[0, 1]
    improvements = []
    for tl in transition_lengths:
        vdd_mean = np.mean(all_results[tl]["vdd"]["iae"])
        rec_mean = np.mean(all_results[tl]["recency"]["iae"])
        improvements.append((rec_mean - vdd_mean) / rec_mean * 100)

    bars = ax.bar(transition_lengths, improvements, color=['green' if x > 0 else 'red' for x in improvements])
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel("Transition Length (steps)")
    ax.set_ylabel("VDD Improvement over Recency (%)")
    ax.set_title("VDD vs Recency by Transition Length\n(Positive = VDD wins)")

    # Add value labels
    for bar, imp in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{imp:+.1f}%', ha='center', va='bottom', fontsize=10)

    # Plot 3: Single run trajectory comparison (TL=50)
    ax = axes[1, 0]
    seed = 42
    steps = 500
    tl = 50
    truth, embeddings = generate_gradual_drift_data(steps, 32, tl, seed)

    vdd_res = run_method("vdd", truth, embeddings)
    rec_res = run_method("recency", truth, embeddings)

    ax.plot(truth, 'k--', alpha=0.5, label='Ground Truth', lw=1)
    ax.plot(vdd_res["memory"], 'g-', label=f'VDD (IAE={vdd_res["iae"]:.1f})', lw=1.5)
    ax.plot(rec_res["memory"], 'r-', label=f'Recency (IAE={rec_res["iae"]:.1f})', lw=1.5, alpha=0.7)

    # Shade transition periods
    ax.axvspan(100, 100 + tl, alpha=0.2, color='orange', label='Transition')
    ax.axvspan(300, 300 + tl, alpha=0.2, color='orange')

    ax.set_ylabel("Value")
    ax.set_xlabel("Time Steps")
    ax.set_title(f"Memory Tracking (Transition Length = {tl})\nOrange = gradual transition periods")
    ax.legend()

    # Plot 4: Lambda trajectory during gradual drift
    ax = axes[1, 1]
    ax.plot(vdd_res["lambdas"], 'g-', lw=1)
    ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='λ_base')
    ax.axvspan(100, 100 + tl, alpha=0.2, color='orange', label='Transition')
    ax.axvspan(300, 300 + tl, alpha=0.2, color='orange')

    ax.set_ylabel("λ(t)")
    ax.set_xlabel("Time Steps")
    ax.set_title("VDD Decay Rate During Gradual Transitions\n(Should rise smoothly, not spike)")
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / "12_gradual_drift.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {results_dir / '12_gradual_drift.png'}")
    plt.close()


def main():
    transition_lengths = [10, 25, 50, 100]
    results = run_experiment()
    plot_results(results, transition_lengths)
    return results


if __name__ == "__main__":
    main()
