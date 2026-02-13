#!/usr/bin/env python3
"""
Experiment 22: Extended Baselines Comparison [PHASE 1]

Adds two practitioner-relevant baselines that reviewers expect:
1. Time-Weighted Retrieval: score = similarity * exp(-alpha * age)
   (What Pinecone/Weaviate/Chroma offer natively)
2. Sliding Window: Keep only last N memories, discard rest

Tests all methods across 4 scenarios: regime shifts, mixed drift,
bursty drift, and reversion.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats as sp_stats
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from vdd.drift_detection import EmbeddingDistance


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def generate_scenario(name, steps, embedding_dim, seed):
    """Generate test scenarios matching Exp 21."""
    np.random.seed(seed)
    truth = np.zeros(steps)
    embeddings = []
    current_val = 0
    centroid = np.random.randn(embedding_dim)
    centroid /= np.linalg.norm(centroid)

    if name == "regime_shifts":
        shifts = (int(steps*0.25), int(steps*0.5), int(steps*0.75))
        for t in range(steps):
            if t in shifts:
                current_val += np.random.choice([-5, 5])
                new_dir = np.random.randn(embedding_dim)
                centroid = 0.3 * centroid + 0.7 * new_dir
                centroid /= np.linalg.norm(centroid)
            current_val += np.random.normal(0, 0.1)
            truth[t] = current_val
            embeddings.append(centroid + np.random.randn(embedding_dim) * 0.1)

    elif name == "mixed_drift":
        drift_ranges = [(int(steps*0.15), int(steps*0.25)),
                        (int(steps*0.50), int(steps*0.60)),
                        (int(steps*0.80), int(steps*0.87))]
        for t in range(steps):
            in_drift = any(s <= t < e for s, e in drift_ranges)
            if in_drift and t > 0:
                if not any(s <= t-1 < e for s, e in drift_ranges):
                    current_val += np.random.choice([-5, 5])
                    new_dir = np.random.randn(embedding_dim)
                    centroid = 0.3 * centroid + 0.7 * new_dir
                    centroid /= np.linalg.norm(centroid)
                noise = 0.3
            else:
                noise = 0.05
            current_val += np.random.normal(0, noise)
            truth[t] = current_val
            embeddings.append(centroid + np.random.randn(embedding_dim) * noise)

    elif name == "bursty":
        burst_at = int(steps * 0.4)
        for t in range(steps):
            if t == burst_at:
                current_val += 15
                new_dir = np.random.randn(embedding_dim)
                centroid = 0.1 * centroid + 0.9 * new_dir
                centroid /= np.linalg.norm(centroid)
            current_val += np.random.normal(0, 0.1)
            truth[t] = current_val
            embeddings.append(centroid + np.random.randn(embedding_dim) * 0.1)

    elif name == "reversion":
        original_centroid = centroid.copy()
        original_val = current_val
        for t in range(steps):
            if t == int(steps * 0.25):
                current_val += 5
                new_dir = np.random.randn(embedding_dim)
                centroid = 0.2 * centroid + 0.8 * new_dir
                centroid /= np.linalg.norm(centroid)
            elif t == int(steps * 0.50):
                centroid = original_centroid + np.random.randn(embedding_dim) * 0.05
                centroid /= np.linalg.norm(centroid)
                current_val = original_val + np.random.normal(0, 0.5)
            elif t == int(steps * 0.75):
                current_val += 5
                new_dir = np.random.randn(embedding_dim)
                centroid = 0.2 * centroid + 0.8 * new_dir
                centroid /= np.linalg.norm(centroid)
            current_val += np.random.normal(0, 0.1)
            truth[t] = current_val
            embeddings.append(centroid + np.random.randn(embedding_dim) * 0.1)

    return truth, embeddings


def run_vdd(truth, embeddings, lambda_base=0.2, lambda_max=0.9):
    steps = len(truth)
    detector = EmbeddingDistance(curr_window=5, arch_window=50, drift_threshold=0.4)
    memory = 0
    errors = []
    for t in range(steps):
        result = detector.update(embeddings[t])
        v = result.volatility
        lam = lambda_base + (lambda_max - lambda_base) * sigmoid(10.0 * (v - 0.5))
        memory = (1 - lam) * memory + lam * truth[t]
        errors.append(abs(truth[t] - memory))
    return np.sum(errors)


def run_static(truth, lam):
    memory = 0
    errors = []
    for t in range(len(truth)):
        memory = (1 - lam) * memory + lam * truth[t]
        errors.append(abs(truth[t] - memory))
    return np.sum(errors)


def run_time_weighted(truth, alpha=0.02):
    """Time-weighted retrieval baseline.
    Keeps all memories, scores by recency: weight = exp(-alpha * age).
    At each step, retrieves weighted average of all past values.
    """
    memories = []  # (timestep, value)
    errors = []
    for t in range(len(truth)):
        memories.append((t, truth[t]))
        if len(memories) == 1:
            estimate = truth[t]
        else:
            weights = np.array([np.exp(-alpha * (t - ts)) for ts, _ in memories])
            values = np.array([v for _, v in memories])
            weights /= weights.sum()
            estimate = np.dot(weights, values)
        errors.append(abs(truth[t] - estimate))
    return np.sum(errors)


def run_sliding_window(truth, window_size=50):
    """Sliding window baseline.
    Keep only last N observations, estimate = mean of window.
    """
    errors = []
    for t in range(len(truth)):
        start = max(0, t - window_size + 1)
        window = truth[start:t+1]
        estimate = np.mean(window)
        errors.append(abs(truth[t] - estimate))
    return np.sum(errors)


def main():
    print("=" * 70)
    print("EXPERIMENT 22: EXTENDED BASELINES COMPARISON")
    print("Adding time-weighted retrieval and sliding window baselines")
    print("=" * 70)

    steps = 500
    embedding_dim = 32
    n_runs = 30
    scenarios = ["regime_shifts", "mixed_drift", "bursty", "reversion"]

    methods = {
        "VDD": lambda t, e: run_vdd(t, e),
        "Recency (λ=0.5)": lambda t, e: run_static(t, 0.5),
        "Static (λ=0.1)": lambda t, e: run_static(t, 0.1),
        "Static (λ=0.2)": lambda t, e: run_static(t, 0.2),
        "TimeWeighted (α=0.01)": lambda t, e: run_time_weighted(t, 0.01),
        "TimeWeighted (α=0.02)": lambda t, e: run_time_weighted(t, 0.02),
        "TimeWeighted (α=0.05)": lambda t, e: run_time_weighted(t, 0.05),
        "SlidingWindow (N=20)": lambda t, e: run_sliding_window(t, 20),
        "SlidingWindow (N=50)": lambda t, e: run_sliding_window(t, 50),
        "SlidingWindow (N=100)": lambda t, e: run_sliding_window(t, 100),
    }

    all_results = {}

    for scenario in scenarios:
        print(f"\n{'─'*60}")
        print(f"Scenario: {scenario}")
        print(f"{'─'*60}")

        scenario_results = {name: [] for name in methods}

        for seed in range(42, 42 + n_runs):
            truth, embeddings = generate_scenario(scenario, steps, embedding_dim, seed)
            for name, fn in methods.items():
                iae = fn(truth, embeddings)
                scenario_results[name].append(iae)

        # Print results sorted by IAE
        print(f"\n  {'Method':<28} {'IAE (mean ± std)':<25} {'vs VDD p-value'}")
        print(f"  {'─'*75}")

        vdd_iaes = scenario_results["VDD"]
        sorted_methods = sorted(scenario_results.keys(),
                                key=lambda m: np.mean(scenario_results[m]))

        for name in sorted_methods:
            iaes = scenario_results[name]
            mean = np.mean(iaes)
            std = np.std(iaes)
            if name != "VDD":
                t_stat, p_val = sp_stats.ttest_rel(vdd_iaes, iaes)
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                marker = "←BEST" if name == sorted_methods[0] else ""
                print(f"  {name:<28} {mean:>8.1f} ± {std:<8.1f}     p={p_val:.4f} {sig} {marker}")
            else:
                print(f"  {name:<28} {mean:>8.1f} ± {std:<8.1f}     ---")

        all_results[scenario] = {
            name: {"mean": float(np.mean(iaes)), "std": float(np.std(iaes)),
                   "values": [float(x) for x in iaes]}
            for name, iaes in scenario_results.items()
        }

    # Summary: Where does VDD rank?
    print(f"\n{'='*70}")
    print("VDD RANKING ACROSS SCENARIOS")
    print(f"{'='*70}")

    for scenario in scenarios:
        ranked = sorted(all_results[scenario].items(), key=lambda x: x[1]["mean"])
        vdd_rank = next(i+1 for i, (name, _) in enumerate(ranked) if name == "VDD")
        best_name = ranked[0][0]
        print(f"  {scenario:<20} VDD rank: {vdd_rank}/{len(methods)} | Best: {best_name}")

    # Save
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / "22_extended_baselines.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to results/22_extended_baselines.json")

    # Plot
    plot_results(all_results, scenarios, results_dir)

    return all_results


def plot_results(all_results, scenarios, results_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for i, scenario in enumerate(scenarios):
        ax = axes[i // 2][i % 2]
        data = all_results[scenario]
        sorted_methods = sorted(data.keys(), key=lambda m: data[m]["mean"])

        names = [m.replace("(", "\n(") for m in sorted_methods]
        means = [data[m]["mean"] for m in sorted_methods]
        stds = [data[m]["std"] for m in sorted_methods]

        colors = []
        for m in sorted_methods:
            if "VDD" in m:
                colors.append('#2ecc71')
            elif "Recency" in m:
                colors.append('#e74c3c')
            elif "TimeWeighted" in m:
                colors.append('#3498db')
            elif "SlidingWindow" in m:
                colors.append('#f39c12')
            else:
                colors.append('#95a5a6')

        bars = ax.barh(range(len(names)), means, xerr=stds, capsize=3,
                       color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("IAE (lower is better)")
        ax.set_title(f"{scenario}", fontweight='bold')
        ax.invert_yaxis()

    plt.suptitle("Experiment 22: VDD vs Extended Baselines\n(Including Time-Weighted & Sliding Window)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "22_extended_baselines.png", dpi=150, bbox_inches='tight')
    print(f"  Saved plot to results/22_extended_baselines.png")
    plt.close()


if __name__ == "__main__":
    main()
