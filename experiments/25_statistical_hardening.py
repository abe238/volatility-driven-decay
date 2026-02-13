#!/usr/bin/env python3
"""
Experiment 25: Statistical Hardening [PHASE 4]

Reruns key experiments with n=30 runs and proper methodology:
- Explicit random seeds
- Bootstrap CIs with 1000 samples
- Cohen's d with interpretation
- Reports mean ± std AND median ± IQR
- Window size ablation for drift detector
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats as sp_stats
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from vdd.drift_detection import EmbeddingDistance
from utils.statistics import bootstrap_ci, cohens_d


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def generate_data(steps, regime_shifts, embedding_dim, seed):
    np.random.seed(seed)
    truth = np.zeros(steps)
    embeddings = []
    current_val = 0
    centroid = np.random.randn(embedding_dim)
    centroid /= np.linalg.norm(centroid)

    for t in range(steps):
        if t in regime_shifts:
            current_val += np.random.choice([-5, 5])
            new_dir = np.random.randn(embedding_dim)
            centroid = 0.3 * centroid + 0.7 * new_dir
            centroid /= np.linalg.norm(centroid)
        else:
            current_val += np.random.normal(0, 0.1)
        truth[t] = current_val
        embeddings.append(centroid + np.random.randn(embedding_dim) * 0.1)
    return truth, embeddings


def run_method(method, truth, embeddings, curr_window=5, arch_window=50):
    steps = len(truth)
    memory = 0
    errors = []

    if method == "vdd":
        detector = EmbeddingDistance(curr_window=curr_window, arch_window=arch_window, drift_threshold=0.4)

    for t in range(steps):
        if method == "vdd":
            result = detector.update(embeddings[t])
            v = result.volatility
            lam = 0.2 + (0.9 - 0.2) * sigmoid(10.0 * (v - 0.5))
        elif method == "recency":
            lam = 0.5
        elif method.startswith("static_"):
            lam = float(method.split("_")[1])
        memory = (1 - lam) * memory + lam * truth[t]
        errors.append(abs(truth[t] - memory))
    return np.sum(errors)


def main():
    print("=" * 70)
    print("EXPERIMENT 25: STATISTICAL HARDENING (n=30)")
    print("=" * 70)

    steps = 200
    regime_shifts = (50, 100, 150)
    embedding_dim = 32
    n_runs = 30

    methods = ["vdd", "recency", "static_0.05", "static_0.10", "static_0.20", "static_0.30"]

    # Part A: Rerun baseline comparison with n=30
    print("\n=== PART A: Baseline Comparison (n=30) ===\n")
    results = {m: [] for m in methods}

    for seed in range(100, 100 + n_runs):
        truth, embeddings = generate_data(steps, regime_shifts, embedding_dim, seed)
        for method in methods:
            iae = run_method(method, truth, embeddings)
            results[method].append(iae)

    print(f"{'Method':<15} {'Mean±Std':<22} {'Median±IQR':<22} {'95% CI':<22} {'vs VDD d':<12}")
    print("-" * 90)

    vdd_vals = results["vdd"]
    for method in methods:
        vals = results[method]
        mean, std = np.mean(vals), np.std(vals)
        median = np.median(vals)
        q25, q75 = np.percentile(vals, [25, 75])
        ci = bootstrap_ci(np.array(vals))
        if method != "vdd":
            d = cohens_d(vals, vdd_vals)
            print(f"{method:<15} {mean:>7.1f} ± {std:<7.1f}   {median:>7.1f} [{q25:.0f}-{q75:.0f}]   [{ci[0]:.1f}, {ci[1]:.1f}]   d={d:+.2f}")
        else:
            print(f"{method:<15} {mean:>7.1f} ± {std:<7.1f}   {median:>7.1f} [{q25:.0f}-{q75:.0f}]   [{ci[0]:.1f}, {ci[1]:.1f}]   ---")

    # Part B: Window Size Ablation
    print("\n=== PART B: Window Size Ablation ===\n")
    curr_windows = [3, 5, 10, 20, 50]
    arch_windows = [20, 50, 100, 200, 500]

    ablation_results = {}
    for cw in curr_windows:
        for aw in arch_windows:
            if aw < cw * 2:
                continue
            key = f"cw={cw},aw={aw}"
            iaes = []
            for seed in range(100, 100 + n_runs):
                truth, embeddings = generate_data(steps, regime_shifts, embedding_dim, seed)
                iae = run_method("vdd", truth, embeddings, curr_window=cw, arch_window=aw)
                iaes.append(iae)
            ablation_results[key] = {
                "curr_window": cw, "arch_window": aw,
                "mean": float(np.mean(iaes)), "std": float(np.std(iaes))
            }

    print(f"{'Config':<20} {'IAE Mean±Std':<20}")
    print("-" * 40)
    sorted_configs = sorted(ablation_results.items(), key=lambda x: x[1]["mean"])
    for key, data in sorted_configs[:10]:
        print(f"{key:<20} {data['mean']:>7.1f} ± {data['std']:.1f}")
    print(f"  ... ({len(ablation_results)} configs tested, showing top 10)")

    best = sorted_configs[0]
    worst = sorted_configs[-1]
    print(f"\n  Best:  {best[0]} (IAE={best[1]['mean']:.1f})")
    print(f"  Worst: {worst[0]} (IAE={worst[1]['mean']:.1f})")
    print(f"  Sensitivity: {(worst[1]['mean']-best[1]['mean'])/best[1]['mean']*100:.1f}% range")

    # Save
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    output = {
        "baseline_comparison": {m: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                                     "median": float(np.median(v)),
                                     "values": [float(x) for x in v]}
                                 for m, v in results.items()},
        "window_ablation": ablation_results,
        "n_runs": n_runs,
        "methodology": "Explicit seeds 100-129, bootstrap CI 1000 samples, paired comparisons"
    }

    with open(results_dir / "25_statistical_hardening.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to results/25_statistical_hardening.json")

    # Plot ablation heatmap
    plot_ablation(ablation_results, curr_windows, arch_windows, results_dir)

    return output


def plot_ablation(ablation_results, curr_windows, arch_windows, results_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    matrix = np.full((len(curr_windows), len(arch_windows)), np.nan)

    for i, cw in enumerate(curr_windows):
        for j, aw in enumerate(arch_windows):
            key = f"cw={cw},aw={aw}"
            if key in ablation_results:
                matrix[i, j] = ablation_results[key]["mean"]

    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(len(arch_windows)))
    ax.set_xticklabels(arch_windows)
    ax.set_yticks(range(len(curr_windows)))
    ax.set_yticklabels(curr_windows)
    ax.set_xlabel("Archive Window Size")
    ax.set_ylabel("Current Window Size")
    ax.set_title("Window Size Ablation: IAE (lower/green = better)")
    plt.colorbar(im, ax=ax, label="IAE")

    for i in range(len(curr_windows)):
        for j in range(len(arch_windows)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i,j]:.0f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(results_dir / "25_window_ablation.png", dpi=150, bbox_inches='tight')
    print(f"  Saved ablation heatmap to results/25_window_ablation.png")
    plt.close()


if __name__ == "__main__":
    main()
