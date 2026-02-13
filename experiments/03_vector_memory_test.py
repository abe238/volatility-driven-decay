#!/usr/bin/env python3
"""
Experiment 03: Vector Memory Test

Test VDDMemoryBank with real vector embeddings to verify
the extension from scalar to vector memory works correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vdd.drift_detection import EmbeddingDistance
from vdd.memory import VDDMemoryBank, StaticDecayMemory
from vdd.utils.embeddings import DummyEmbedder


def run_experiment(
    steps: int = 200,
    regime_shifts: tuple = (50, 100, 150),
    embedding_dim: int = 64,
    memories_per_step: int = 1,
    seed: int = 42,
):
    """Run vector memory experiment."""
    np.random.seed(seed)
    embedder = DummyEmbedder(dim=embedding_dim, seed=seed)

    # Initialize memory banks
    detector = EmbeddingDistance(curr_window=10, arch_window=100, drift_threshold=0.3)
    vdd_bank = VDDMemoryBank(
        drift_detector=detector,
        lambda_base=0.05,
        lambda_max=0.9,
        k=10.0,
        v_threshold=0.5,
    )
    static_bank = StaticDecayMemory(lambda_static=0.1)

    # Generate regime-based content
    regimes = ["API_v1", "API_v2", "API_v3", "API_v4"]
    current_regime = 0
    regime_history = []

    # Track metrics
    vdd_sizes = []
    static_sizes = []
    vdd_lambdas = []
    volatilities = []
    retrieval_accuracy_vdd = []
    retrieval_accuracy_static = []

    for t in range(steps):
        # Check for regime shift
        if t in regime_shifts:
            current_regime += 1

        regime_history.append(current_regime)

        # Generate content for current regime
        content = f"{regimes[current_regime]}_doc_{t}"
        # Embeddings cluster by regime
        base_emb = embedder(regimes[current_regime])
        noise = np.random.randn(embedding_dim) * 0.1
        embedding = base_emb + noise
        embedding = embedding / np.linalg.norm(embedding)

        # Add to both banks
        vdd_bank.add(embedding, content, {"regime": current_regime, "time": t})
        static_bank.add(embedding, content, {"regime": current_regime, "time": t})

        # Step (apply decay)
        vdd_bank.step()
        static_bank.step()

        # Track metrics
        vdd_sizes.append(vdd_bank.size)
        static_sizes.append(static_bank.size)
        vdd_lambdas.append(vdd_bank.get_current_lambda())
        volatilities.append(detector.get_volatility())

        # Test retrieval - query for current regime
        if t > 10:
            query_emb = embedder(regimes[current_regime])
            query_emb = query_emb + np.random.randn(embedding_dim) * 0.05
            query_emb = query_emb / np.linalg.norm(query_emb)

            # Retrieve top-5
            vdd_results = vdd_bank.retrieve(query_emb, k=5)
            static_results = static_bank.retrieve(query_emb, k=5)

            # Check how many are from current regime
            vdd_correct = sum(1 for r in vdd_results if r.memory.metadata.get("regime") == current_regime)
            static_correct = sum(1 for r in static_results if r.memory.metadata.get("regime") == current_regime)

            retrieval_accuracy_vdd.append(vdd_correct / 5)
            retrieval_accuracy_static.append(static_correct / 5)

    return {
        "steps": steps,
        "regime_shifts": regime_shifts,
        "regime_history": regime_history,
        "vdd_sizes": vdd_sizes,
        "static_sizes": static_sizes,
        "vdd_lambdas": vdd_lambdas,
        "volatilities": volatilities,
        "retrieval_accuracy_vdd": retrieval_accuracy_vdd,
        "retrieval_accuracy_static": retrieval_accuracy_static,
    }


def plot_results(results: dict, save_path: str = None):
    """Plot experiment results."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    steps = results["steps"]
    shifts = results["regime_shifts"]

    # Plot 1: Memory sizes
    ax = axes[0]
    ax.plot(results["vdd_sizes"], "g-", label="VDD Memory Size", lw=2)
    ax.plot(results["static_sizes"], "r-", label="Static Memory Size", alpha=0.7)
    ax.set_ylabel("Memory Size")
    ax.set_title("Vector Memory Bank Comparison")
    ax.legend(loc="upper left")
    for s in shifts:
        ax.axvline(x=s, color="gray", ls=":", alpha=0.5)

    # Plot 2: Lambda and Volatility
    ax = axes[1]
    ax.plot(results["vdd_lambdas"], "b-", label="λ(t) Decay Rate", lw=2)
    ax.plot(results["volatilities"], "orange", label="Volatility", alpha=0.7)
    ax.set_ylabel("Value")
    ax.set_title("VDD Decay Rate and Volatility")
    ax.legend(loc="upper right")
    for s in shifts:
        ax.axvline(x=s, color="gray", ls=":", alpha=0.5)

    # Plot 3: Retrieval Accuracy
    ax = axes[2]
    x_acc = range(11, steps)  # Offset for warmup
    ax.plot(x_acc, results["retrieval_accuracy_vdd"], "g-", label="VDD Accuracy", lw=2)
    ax.plot(x_acc, results["retrieval_accuracy_static"], "r-", label="Static Accuracy", alpha=0.7)
    ax.set_ylabel("Accuracy")
    ax.set_title("Retrieval Accuracy (Current Regime Documents)")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.1)
    for s in shifts:
        ax.axvline(x=s, color="gray", ls=":", alpha=0.5)

    # Plot 4: Rolling average accuracy
    ax = axes[3]
    window = 10
    vdd_rolling = np.convolve(results["retrieval_accuracy_vdd"], np.ones(window)/window, mode='valid')
    static_rolling = np.convolve(results["retrieval_accuracy_static"], np.ones(window)/window, mode='valid')
    x_roll = range(11 + window - 1, steps)
    ax.plot(x_roll, vdd_rolling, "g-", label=f"VDD (rolling avg)", lw=2)
    ax.plot(x_roll, static_rolling, "r-", label=f"Static (rolling avg)", alpha=0.7)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Time Steps")
    ax.set_title(f"Rolling Average Accuracy (window={window})")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.1)
    for s in shifts:
        ax.axvline(x=s, color="gray", ls=":", alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.close()


def main():
    """Main entry point."""
    print("=" * 60)
    print("Experiment 03: Vector Memory Test")
    print("=" * 60)

    results = run_experiment()

    # Calculate summary metrics
    vdd_acc = np.mean(results["retrieval_accuracy_vdd"])
    static_acc = np.mean(results["retrieval_accuracy_static"])

    print(f"\nResults:")
    print(f"  VDD Mean Retrieval Accuracy:    {vdd_acc:.3f}")
    print(f"  Static Mean Retrieval Accuracy: {static_acc:.3f}")
    print(f"  Improvement: {(vdd_acc - static_acc) / static_acc * 100:.1f}%")

    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    plot_results(results, str(results_dir / "03_vector_memory.png"))

    return results


if __name__ == "__main__":
    main()
