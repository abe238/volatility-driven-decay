#!/usr/bin/env python3
"""
Experiment 02: Fixed Scalar Simulation

Fixed version of the original simulation that uses REAL drift
detection instead of oracle volatility signal.

This addresses the critical flaw in the original paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vdd.drift_detection import ADWIN, EmbeddingDistance


def run_simulation(
    steps: int = 100,
    regime_shifts: tuple = (20, 50, 80),
    noise_level: float = 0.1,
    lambda_base: float = 0.1,
    lambda_max: float = 0.9,
    use_oracle: bool = False,
    embedding_dim: int = 32,
    seed: int = 42,
):
    """Run VDD simulation with real or oracle drift detection."""
    np.random.seed(seed)

    # Generate ground truth (scalar for plotting)
    truth = np.zeros(steps)
    current_val = 0

    # Generate embeddings that shift with regimes
    embeddings = []
    centroid = np.random.randn(embedding_dim)
    centroid = centroid / np.linalg.norm(centroid)

    for t in range(steps):
        if t in regime_shifts:
            current_val += np.random.choice([-5, 5])
            # Shift centroid
            new_dir = np.random.randn(embedding_dim)
            centroid = 0.3 * centroid + 0.7 * new_dir
            centroid = centroid / np.linalg.norm(centroid)
        else:
            current_val += np.random.normal(0, noise_level)
        truth[t] = current_val

        # Generate embedding with noise
        emb = centroid + np.random.randn(embedding_dim) * 0.1
        embeddings.append(emb)

    # Initialize detector - use EmbeddingDistance for semantic drift
    detector = EmbeddingDistance(curr_window=5, arch_window=50, drift_threshold=0.4)

    # Run agents
    def run_agent(adaptive: bool):
        memory = np.zeros(steps)
        stored = 0
        lambdas = np.zeros(steps)
        volatilities = np.zeros(steps)

        for t in range(1, steps):
            # Use embedding for drift detection
            result = detector.update(embeddings[t]) if adaptive else None

            if adaptive:
                if use_oracle and t in regime_shifts:
                    current_lambda = lambda_max
                    v = 1.0
                else:
                    v = result.volatility if result else 0
                    current_lambda = lambda_base + (lambda_max - lambda_base) * v
                volatilities[t] = v
            else:
                current_lambda = lambda_base

            lambdas[t] = current_lambda
            stored = (1 - current_lambda) * stored + current_lambda * truth[t]
            memory[t] = stored

        detector.reset()
        return memory, lambdas, volatilities

    static_memory, static_lambdas, _ = run_agent(adaptive=False)
    adaptive_memory, adaptive_lambdas, adaptive_volatilities = run_agent(adaptive=True)

    return {
        "truth": truth,
        "static_memory": static_memory,
        "adaptive_memory": adaptive_memory,
        "static_lambdas": static_lambdas,
        "adaptive_lambdas": adaptive_lambdas,
        "adaptive_volatilities": adaptive_volatilities,
        "regime_shifts": regime_shifts,
    }


def plot_results(results: dict, title_suffix: str = "", save_path: str = None):
    """Plot simulation results."""
    truth = results["truth"]
    static = results["static_memory"]
    adaptive = results["adaptive_memory"]
    shifts = results["regime_shifts"]

    static_error = np.abs(truth - static)
    adaptive_error = np.abs(truth - adaptive)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(truth, "k--", lw=2, label="Ground Truth")
    ax1.plot(static, "r", alpha=0.6, label="Static Decay")
    ax1.plot(adaptive, "g", lw=2.5, label="VDD (Real Detection)")
    ax1.set_title(f"Memory Adaptation During Regime Changes {title_suffix}")
    ax1.set_ylabel("Context Value")
    ax1.legend(loc="upper left")
    for s in shifts:
        ax1.axvline(x=s, color="gray", ls=":", alpha=0.5)

    ax2.fill_between(range(len(truth)), static_error, color="red", alpha=0.3, label="Static Error")
    ax2.fill_between(range(len(truth)), adaptive_error, color="green", alpha=0.3, label="Adaptive Error")
    ax2.set_title("Retrieval Error (Lower is Better)")
    ax2.set_ylabel("Absolute Error")
    ax2.set_xlabel("Time Steps")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.close()

    # Print metrics
    print(f"\nIntegrated Absolute Error:")
    print(f"  Static:   {np.sum(static_error):.2f}")
    print(f"  Adaptive: {np.sum(adaptive_error):.2f}")
    print(f"  Improvement: {(1 - np.sum(adaptive_error)/np.sum(static_error))*100:.1f}%")


if __name__ == "__main__":
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    print("Running with REAL drift detection (no oracle)...")
    results = run_simulation(use_oracle=False)
    plot_results(results, "(Real Detection)", str(results_dir / "02_simulation_real.png"))
