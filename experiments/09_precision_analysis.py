#!/usr/bin/env python3
"""
Experiment 09: Precision Cost Analysis

Quantify the cost of low precision (99% false positives) in drift detection.
Measures: recovery time, over-forgetting cost, actual λ distribution.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vdd.drift_detection import EmbeddingDistance


def analyze_precision_cost():
    """Analyze the cost of low precision drift detection."""
    print("=" * 60)
    print("Experiment 09: Precision Cost Analysis")
    print("=" * 60)

    # Generate stable data (no real drift)
    np.random.seed(42)
    steps = 500
    embedding_dim = 32

    # Create stable embeddings with minor noise
    base_centroid = np.random.randn(embedding_dim)
    base_centroid = base_centroid / np.linalg.norm(base_centroid)

    embeddings = []
    for t in range(steps):
        emb = base_centroid + np.random.randn(embedding_dim) * 0.1
        embeddings.append(emb)

    # Run detector
    detector = EmbeddingDistance(curr_window=10, arch_window=100, drift_threshold=0.4)

    volatilities = []
    lambdas = []
    drift_detected = []
    lambda_base, lambda_max = 0.2, 0.9

    for t, emb in enumerate(embeddings):
        result = detector.update(emb)
        v = result.volatility
        volatilities.append(v)

        current_lambda = lambda_base + (lambda_max - lambda_base) * v
        lambdas.append(current_lambda)
        drift_detected.append(result.detected)

    # Analysis
    lambdas = np.array(lambdas)
    volatilities = np.array(volatilities)

    print(f"\n{'='*60}")
    print("FINDINGS: No Real Drift, But What Does VDD Do?")
    print(f"{'='*60}")

    # Lambda distribution
    print(f"\nλ Distribution (should be ~{lambda_base} if no drift):")
    print(f"  Mean λ:   {np.mean(lambdas):.3f} (target: {lambda_base})")
    print(f"  Std λ:    {np.std(lambdas):.3f}")
    print(f"  Min λ:    {np.min(lambdas):.3f}")
    print(f"  Max λ:    {np.max(lambdas):.3f}")

    # How often is λ elevated?
    high_lambda_threshold = lambda_base + 0.1  # Anything above 0.3
    panic_mode_pct = 100 * np.mean(lambdas > high_lambda_threshold)
    print(f"\n  Time in 'elevated mode' (λ > {high_lambda_threshold}): {panic_mode_pct:.1f}%")

    # Volatility analysis
    print(f"\nVolatility Distribution (no real drift):")
    print(f"  Mean V:   {np.mean(volatilities):.3f}")
    print(f"  Std V:    {np.std(volatilities):.3f}")
    print(f"  Max V:    {np.max(volatilities):.3f}")

    # False positive analysis
    false_positives = sum(drift_detected)
    print(f"\nFalse Positive Analysis:")
    print(f"  Drift detections: {false_positives} / {steps} = {100*false_positives/steps:.1f}%")
    print(f"  (All are false positives since no real drift)")

    # Over-forgetting cost calculation
    # Compare VDD memory vs what it would be with constant λ_base
    truth = np.cumsum(np.random.randn(steps) * 0.1)  # Random walk

    vdd_memory = np.zeros(steps)
    static_memory = np.zeros(steps)
    vdd_stored = 0
    static_stored = 0

    for t in range(1, steps):
        vdd_stored = (1 - lambdas[t]) * vdd_stored + lambdas[t] * truth[t]
        static_stored = (1 - lambda_base) * static_stored + lambda_base * truth[t]
        vdd_memory[t] = vdd_stored
        static_memory[t] = static_stored

    vdd_error = np.sum(np.abs(truth - vdd_memory))
    static_error = np.sum(np.abs(truth - static_memory))
    over_forgetting_cost = 100 * (vdd_error - static_error) / static_error

    print(f"\nOver-Forgetting Cost (stable data):")
    print(f"  VDD IAE:    {vdd_error:.2f}")
    print(f"  Static IAE: {static_error:.2f}")
    print(f"  Extra error from false positives: {over_forgetting_cost:.1f}%")

    return {
        "mean_lambda": np.mean(lambdas),
        "panic_mode_pct": panic_mode_pct,
        "false_positive_rate": false_positives / steps,
        "over_forgetting_cost_pct": over_forgetting_cost,
        "lambdas": lambdas,
        "volatilities": volatilities,
    }


def plot_precision_analysis(analysis: dict):
    """Plot precision analysis results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Lambda distribution histogram
    ax = axes[0, 0]
    ax.hist(analysis["lambdas"], bins=30, color='orange', alpha=0.7, edgecolor='black')
    ax.axvline(x=0.2, color='green', linestyle='--', label='Target λ_base=0.2')
    ax.axvline(x=np.mean(analysis["lambdas"]), color='red', linestyle='-', label=f'Actual mean={np.mean(analysis["lambdas"]):.2f}')
    ax.set_xlabel('λ(t)')
    ax.set_ylabel('Frequency')
    ax.set_title('λ Distribution in Stable Environment\n(Should cluster at 0.2 if precision were perfect)')
    ax.legend()

    # Plot 2: Lambda over time
    ax = axes[0, 1]
    ax.plot(analysis["lambdas"], color='orange', alpha=0.7, lw=0.5)
    ax.axhline(y=0.2, color='green', linestyle='--', label='Target λ_base')
    ax.fill_between(range(len(analysis["lambdas"])), analysis["lambdas"], 0.2,
                    where=np.array(analysis["lambdas"]) > 0.2, alpha=0.3, color='red',
                    label='Over-forgetting')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('λ(t)')
    ax.set_title('λ Over Time (No Real Drift)\nRed area = unnecessary high decay')
    ax.legend()
    ax.set_ylim(0, 1)

    # Plot 3: Volatility over time
    ax = axes[1, 0]
    ax.plot(analysis["volatilities"], color='blue', alpha=0.7, lw=0.5)
    ax.axhline(y=0.5, color='red', linestyle='--', label='Drift threshold (V_0)')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Volatility V_t')
    ax.set_title('Detected Volatility in Stable Environment\n(Should be consistently low)')
    ax.legend()

    # Plot 4: Summary metrics
    ax = axes[1, 1]
    ax.axis('off')

    summary = f"""
    PRECISION COST SUMMARY
    ══════════════════════════════════════

    In a STABLE environment (no real drift):

    λ Distribution:
    • Target λ_base:         0.20
    • Actual mean λ:         {np.mean(analysis["lambdas"]):.2f}
    • Time in elevated mode: {analysis["panic_mode_pct"]:.1f}%

    False Positive Rate:
    • Drift detections:      {analysis["false_positive_rate"]*100:.1f}%
    • (All are false positives)

    Over-Forgetting Cost:
    • Extra error:           {analysis["over_forgetting_cost_pct"]:.1f}%
    • (vs static λ=0.2)

    ══════════════════════════════════════
    CONCLUSION: Low precision causes VDD to
    operate at elevated λ even in stable
    periods, explaining why recency (λ=0.5)
    can compete with VDD.
    """

    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / "09_precision_cost.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {results_dir / '09_precision_cost.png'}")
    plt.close()


def main():
    analysis = analyze_precision_cost()
    plot_precision_analysis(analysis)
    return analysis


if __name__ == "__main__":
    main()
