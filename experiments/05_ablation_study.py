#!/usr/bin/env python3
"""
Experiment 05: Ablation Study

Grid search λ_base, λ_max to find optimal settings and
understand parameter sensitivity.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vdd.drift_detection import EmbeddingDistance


def run_single_config(
    lambda_base: float,
    lambda_max: float,
    steps: int = 100,
    regime_shifts: tuple = (25, 50, 75),
    seed: int = 42,
):
    """Run simulation with specific config."""
    np.random.seed(seed)

    embedding_dim = 32
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

    detector = EmbeddingDistance(curr_window=5, arch_window=50, drift_threshold=0.4)

    memory = np.zeros(steps)
    stored = 0

    for t in range(1, steps):
        result = detector.update(embeddings[t])
        v = result.volatility
        current_lambda = lambda_base + (lambda_max - lambda_base) * v
        stored = (1 - current_lambda) * stored + current_lambda * truth[t]
        memory[t] = stored

    error = np.abs(truth - memory)
    iae = np.sum(error)

    return iae


def run_ablation():
    """Run full ablation study."""
    print("=" * 60)
    print("Experiment 05: Ablation Study")
    print("=" * 60)

    # Parameter ranges
    lambda_base_values = [0.01, 0.05, 0.1, 0.2, 0.3]
    lambda_max_values = [0.5, 0.7, 0.8, 0.9, 0.95]

    # Results matrix
    results = np.zeros((len(lambda_base_values), len(lambda_max_values)))

    print(f"\nGrid: {len(lambda_base_values)} x {len(lambda_max_values)} = {len(lambda_base_values) * len(lambda_max_values)} configs")
    print("Running...")

    for i, lb in enumerate(lambda_base_values):
        for j, lm in enumerate(lambda_max_values):
            # Run multiple seeds and average
            errors = []
            for seed in [42, 43, 44]:
                iae = run_single_config(lb, lm, seed=seed)
                errors.append(iae)
            results[i, j] = np.mean(errors)
            print(f"  λ_base={lb:.2f}, λ_max={lm:.2f}: IAE={results[i,j]:.2f}")

    # Find best config
    best_idx = np.unravel_index(np.argmin(results), results.shape)
    best_base = lambda_base_values[best_idx[0]]
    best_max = lambda_max_values[best_idx[1]]
    best_iae = results[best_idx]

    print(f"\n{'='*60}")
    print(f"BEST CONFIG: λ_base={best_base}, λ_max={best_max}")
    print(f"BEST IAE: {best_iae:.2f}")
    print(f"{'='*60}")

    # Compare with static baseline
    static_iae = run_single_config(0.1, 0.1)  # Static = same base and max
    print(f"\nStatic baseline (λ=0.1): IAE={static_iae:.2f}")
    print(f"Best VDD improvement: {(1 - best_iae/static_iae)*100:.1f}%")

    return {
        "lambda_base_values": lambda_base_values,
        "lambda_max_values": lambda_max_values,
        "results": results,
        "best_base": best_base,
        "best_max": best_max,
        "best_iae": best_iae,
        "static_iae": static_iae,
    }


def plot_results(data: dict, save_path: str = None):
    """Plot heatmap of results."""
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(data["results"], cmap="RdYlGn_r", aspect="auto")

    # Labels
    ax.set_xticks(range(len(data["lambda_max_values"])))
    ax.set_xticklabels([f"{v:.2f}" for v in data["lambda_max_values"]])
    ax.set_yticks(range(len(data["lambda_base_values"])))
    ax.set_yticklabels([f"{v:.2f}" for v in data["lambda_base_values"]])

    ax.set_xlabel("λ_max (Panic Rate)", fontsize=12)
    ax.set_ylabel("λ_base (Resting Rate)", fontsize=12)
    ax.set_title("VDD Ablation Study: Integrated Absolute Error\n(Lower is Better)", fontsize=14)

    # Add values to cells
    for i in range(len(data["lambda_base_values"])):
        for j in range(len(data["lambda_max_values"])):
            val = data["results"][i, j]
            color = "white" if val > np.median(data["results"]) else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=color, fontsize=10)

    # Mark best
    best_idx = np.unravel_index(np.argmin(data["results"]), data["results"].shape)
    ax.add_patch(plt.Rectangle((best_idx[1]-0.5, best_idx[0]-0.5), 1, 1,
                                fill=False, edgecolor="blue", linewidth=3))

    plt.colorbar(im, ax=ax, label="IAE")

    # Add text annotation
    text = f"Best: λ_base={data['best_base']}, λ_max={data['best_max']}\n"
    text += f"IAE: {data['best_iae']:.1f} (vs Static: {data['static_iae']:.1f})"
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.close()


def main():
    """Main entry point."""
    data = run_ablation()

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    plot_results(data, str(results_dir / "05_ablation_heatmap.png"))

    return data


if __name__ == "__main__":
    main()
