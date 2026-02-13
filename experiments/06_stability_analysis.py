#!/usr/bin/env python3
"""
Experiment 06: Stability Analysis

Run 10K step simulation to verify no oscillation or pathological behavior.
Tests VDD under various conditions including high noise and rapid regime changes.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vdd.drift_detection import EmbeddingDistance


def run_stability_test(
    steps: int = 10000,
    noise_level: float = 0.1,
    seed: int = 42,
):
    """Run long-term stability simulation."""
    np.random.seed(seed)

    # Complex regime pattern
    # Steps 0-1000: Stable A
    # Steps 1001-1010: Drift to B
    # Steps 1011-3000: Stable B
    # Steps 3001-3500: Gradual drift to C
    # Steps 3501-5000: Stable C
    # Steps 5001-5010: Drift back to A
    # Steps 5011-7000: Stable A
    # Steps 7001-10000: High noise (no clear regime)

    embedding_dim = 32
    truth = np.zeros(steps)
    current_val = 0

    centroids = {
        "A": np.random.randn(embedding_dim),
        "B": np.random.randn(embedding_dim),
        "C": np.random.randn(embedding_dim),
    }
    for k in centroids:
        centroids[k] = centroids[k] / np.linalg.norm(centroids[k])

    embeddings = []
    regimes = []
    current_centroid = centroids["A"]

    for t in range(steps):
        # Determine regime
        if t <= 1000:
            regime = "A"
            current_centroid = centroids["A"]
        elif t <= 1010:
            regime = "A→B"
            alpha = (t - 1000) / 10
            current_centroid = (1 - alpha) * centroids["A"] + alpha * centroids["B"]
        elif t <= 3000:
            regime = "B"
            current_centroid = centroids["B"]
        elif t <= 3500:
            regime = "B→C"
            alpha = (t - 3000) / 500  # Gradual
            current_centroid = (1 - alpha) * centroids["B"] + alpha * centroids["C"]
        elif t <= 5000:
            regime = "C"
            current_centroid = centroids["C"]
        elif t <= 5010:
            regime = "C→A"
            alpha = (t - 5000) / 10
            current_centroid = (1 - alpha) * centroids["C"] + alpha * centroids["A"]
        elif t <= 7000:
            regime = "A"
            current_centroid = centroids["A"]
        else:
            regime = "NOISE"
            # Random centroid each step
            current_centroid = np.random.randn(embedding_dim)

        current_centroid = current_centroid / np.linalg.norm(current_centroid)
        regimes.append(regime)

        # Generate scalar truth
        if regime == "NOISE":
            current_val += np.random.normal(0, 0.5)  # High noise
        elif "→" in regime:
            current_val += np.random.choice([-0.5, 0.5])
        else:
            current_val += np.random.normal(0, noise_level)
        truth[t] = current_val

        # Generate embedding
        noise = np.random.randn(embedding_dim) * 0.1
        emb = current_centroid + noise
        embeddings.append(emb)

    # Run VDD
    detector = EmbeddingDistance(curr_window=10, arch_window=200, drift_threshold=0.4)

    lambda_base = 0.2
    lambda_max = 0.9

    memory = np.zeros(steps)
    lambdas = np.zeros(steps)
    volatilities = np.zeros(steps)
    stored = 0

    for t in range(1, steps):
        result = detector.update(embeddings[t])
        v = result.volatility
        volatilities[t] = v
        current_lambda = lambda_base + (lambda_max - lambda_base) * v
        lambdas[t] = current_lambda
        stored = (1 - current_lambda) * stored + current_lambda * truth[t]
        memory[t] = stored

    error = np.abs(truth - memory)

    return {
        "steps": steps,
        "truth": truth,
        "memory": memory,
        "error": error,
        "lambdas": lambdas,
        "volatilities": volatilities,
        "regimes": regimes,
    }


def analyze_stability(results: dict):
    """Analyze stability metrics."""
    lambdas = results["lambdas"]
    volatilities = results["volatilities"]

    # Check for oscillation: count rapid λ changes
    lambda_diff = np.abs(np.diff(lambdas))
    high_freq_changes = np.sum(lambda_diff > 0.3)
    oscillation_rate = high_freq_changes / len(lambdas)

    # Compute windowed statistics
    window = 100
    lambda_std = []
    for i in range(0, len(lambdas) - window, window):
        lambda_std.append(np.std(lambdas[i:i+window]))

    metrics = {
        "oscillation_rate": oscillation_rate,
        "mean_lambda": np.mean(lambdas),
        "std_lambda": np.std(lambdas),
        "max_lambda_std_window": np.max(lambda_std) if lambda_std else 0,
        "mean_volatility": np.mean(volatilities),
        "total_error": np.sum(results["error"]),
        "mean_error": np.mean(results["error"]),
    }

    return metrics


def plot_results(results: dict, metrics: dict, save_path: str = None):
    """Plot stability analysis results."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    steps = results["steps"]
    x = range(steps)

    # Regime markers
    regime_changes = [1000, 1010, 3000, 3500, 5000, 5010, 7000]

    # Plot 1: Truth vs Memory
    ax = axes[0]
    ax.plot(results["truth"], "k-", alpha=0.5, label="Ground Truth", lw=0.5)
    ax.plot(results["memory"], "g-", label="VDD Memory", lw=0.5)
    ax.set_ylabel("Value")
    ax.set_title("10K Step Stability Test: VDD Memory Tracking")
    ax.legend(loc="upper right")
    for rc in regime_changes:
        ax.axvline(x=rc, color="red", ls=":", alpha=0.3)

    # Plot 2: Lambda over time
    ax = axes[1]
    ax.plot(results["lambdas"], "b-", lw=0.5)
    ax.fill_between(x, results["lambdas"], alpha=0.3)
    ax.set_ylabel("λ(t)")
    ax.set_title("Decay Rate Over Time (Check for Oscillation)")
    ax.set_ylim(0, 1)
    for rc in regime_changes:
        ax.axvline(x=rc, color="red", ls=":", alpha=0.3)

    # Plot 3: Volatility
    ax = axes[2]
    ax.plot(results["volatilities"], "orange", lw=0.5)
    ax.fill_between(x, results["volatilities"], alpha=0.3, color="orange")
    ax.set_ylabel("Volatility")
    ax.set_title("Detected Volatility")
    ax.set_ylim(0, 1)
    for rc in regime_changes:
        ax.axvline(x=rc, color="red", ls=":", alpha=0.3)

    # Plot 4: Rolling error
    ax = axes[3]
    window = 100
    rolling_error = np.convolve(results["error"], np.ones(window)/window, mode='valid')
    ax.plot(range(window-1, steps), rolling_error, "r-", lw=1)
    ax.fill_between(range(window-1, steps), rolling_error, alpha=0.3, color="red")
    ax.set_ylabel("Rolling MAE")
    ax.set_xlabel("Time Steps")
    ax.set_title(f"Rolling Mean Absolute Error (window={window})")
    for rc in regime_changes:
        ax.axvline(x=rc, color="gray", ls=":", alpha=0.3)

    # Add regime labels
    regime_labels = ["A", "→B", "B", "→C", "C", "→A", "A", "NOISE"]
    regime_positions = [500, 1005, 2000, 3250, 4250, 5005, 6000, 8500]
    for pos, label in zip(regime_positions, regime_labels):
        axes[0].text(pos, axes[0].get_ylim()[1], label, ha='center', fontsize=8,
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    # Add metrics text
    metrics_text = f"Oscillation Rate: {metrics['oscillation_rate']:.4f}\n"
    metrics_text += f"Mean λ: {metrics['mean_lambda']:.3f}\n"
    metrics_text += f"Total Error: {metrics['total_error']:.1f}\n"
    metrics_text += f"Mean Error: {metrics['mean_error']:.3f}"
    fig.text(0.02, 0.98, metrics_text, transform=fig.transFigure, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.close()


def main():
    """Main entry point."""
    print("=" * 60)
    print("Experiment 06: Stability Analysis (10K steps)")
    print("=" * 60)

    print("\nRunning 10,000 step simulation...")
    results = run_stability_test(steps=10000)

    print("Analyzing stability metrics...")
    metrics = analyze_stability(results)

    print(f"\n{'='*60}")
    print("STABILITY METRICS")
    print(f"{'='*60}")
    print(f"Oscillation Rate: {metrics['oscillation_rate']:.4f} (lower is better)")
    print(f"Mean λ: {metrics['mean_lambda']:.3f}")
    print(f"Std λ: {metrics['std_lambda']:.3f}")
    print(f"Max Window Std: {metrics['max_lambda_std_window']:.3f}")
    print(f"Total Error: {metrics['total_error']:.1f}")
    print(f"Mean Error: {metrics['mean_error']:.3f}")

    # Stability assessment
    print(f"\n{'='*60}")
    print("STABILITY ASSESSMENT")
    print(f"{'='*60}")
    if metrics['oscillation_rate'] < 0.01:
        print("✅ No oscillation detected (rate < 1%)")
    else:
        print(f"⚠️ Some oscillation detected (rate = {metrics['oscillation_rate']*100:.1f}%)")

    if metrics['max_lambda_std_window'] < 0.4:
        print("✅ Lambda is stable within windows")
    else:
        print("⚠️ High lambda variance in some windows")

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    plot_results(results, metrics, str(results_dir / "06_stability_10k.png"))

    return results, metrics


if __name__ == "__main__":
    main()
