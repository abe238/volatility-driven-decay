#!/usr/bin/env python3
"""
Experiment 31: Adaptive Baselines Comparison

Compares VDD against adaptive/ensemble baselines missing from the paper:
1. DWM-lite (Dynamic Weighted Majority): 5 experts with static lambdas, weighted voting
2. EMA-lambda: Lambda adapts based on exponentially weighted error signal
3. Holt-Winters-inspired: Tracks level + trend of error rate to adjust lambda

Tests across 4 scenarios with n=30 runs, bootstrap CIs, Cohen's d, paired t-tests.
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


def generate_scenario(name, steps, embedding_dim, seed):
    np.random.seed(seed)
    truth = np.zeros(steps)
    embeddings = []
    current_val = 0
    centroid = np.random.randn(embedding_dim)
    centroid /= np.linalg.norm(centroid)

    if name == "regime_shifts":
        shifts = (int(steps * 0.25), int(steps * 0.5), int(steps * 0.75))
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
        drift_ranges = [(int(steps * 0.15), int(steps * 0.25)),
                        (int(steps * 0.50), int(steps * 0.60)),
                        (int(steps * 0.80), int(steps * 0.87))]
        for t in range(steps):
            in_drift = any(s <= t < e for s, e in drift_ranges)
            if in_drift and t > 0:
                if not any(s <= t - 1 < e for s, e in drift_ranges):
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


def run_vdd(truth, embeddings):
    detector = EmbeddingDistance(curr_window=5, arch_window=50, drift_threshold=0.4)
    memory = 0
    errors = []
    for t in range(len(truth)):
        result = detector.update(embeddings[t])
        v = result.volatility
        lam = 0.2 + (0.9 - 0.2) * sigmoid(10.0 * (v - 0.5))
        memory = (1 - lam) * memory + lam * truth[t]
        errors.append(abs(truth[t] - memory))
    return np.sum(errors)


def run_static(truth, embeddings, lam=0.1):
    memory = 0
    errors = []
    for t in range(len(truth)):
        memory = (1 - lam) * memory + lam * truth[t]
        errors.append(abs(truth[t] - memory))
    return np.sum(errors)


def run_recency(truth, embeddings):
    return run_static(truth, embeddings, lam=0.5)


def run_dwm_lite(truth, embeddings):
    expert_lambdas = [0.05, 0.1, 0.2, 0.3, 0.5]
    expert_memories = [0.0] * 5
    expert_weights = [1.0] * 5
    errors = []
    beta = 0.5
    for t in range(len(truth)):
        total_w = sum(expert_weights)
        prediction = sum(w * m for w, m in zip(expert_weights, expert_memories)) / total_w
        errors.append(abs(truth[t] - prediction))
        expert_errors = [abs(truth[t] - m) for m in expert_memories]
        mean_error = np.mean(expert_errors)
        for i in range(5):
            if expert_errors[i] > mean_error:
                expert_weights[i] *= beta
            expert_memories[i] = (1 - expert_lambdas[i]) * expert_memories[i] + expert_lambdas[i] * truth[t]
        max_w = max(expert_weights)
        if max_w > 0:
            expert_weights = [w / max_w for w in expert_weights]
    return np.sum(errors)


def run_ema_lambda(truth, embeddings, alpha=0.2):
    memory = 0
    lam = 0.2
    errors = []
    for t in range(len(truth)):
        error = abs(truth[t] - memory)
        lam = np.clip(lam + alpha * (error - lam), 0.05, 0.9)
        memory = (1 - lam) * memory + lam * truth[t]
        errors.append(abs(truth[t] - memory))
    return np.sum(errors)


def run_holt_winters(truth, embeddings, alpha_level=0.3, alpha_trend=0.1):
    memory = 0
    lam = 0.2
    level = 0
    trend = 0
    errors = []
    for t in range(len(truth)):
        error = abs(truth[t] - memory)
        new_level = alpha_level * error + (1 - alpha_level) * (level + trend)
        trend = alpha_trend * (new_level - level) + (1 - alpha_trend) * trend
        level = new_level
        lam = np.clip(0.2 + 0.7 * sigmoid(5 * (level + trend - 1.0)), 0.05, 0.9)
        memory = (1 - lam) * memory + lam * truth[t]
        errors.append(abs(truth[t] - memory))
    return np.sum(errors)


def effect_label(d):
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    return "large"


def main():
    print("=" * 70)
    print("EXPERIMENT 31: ADAPTIVE BASELINES COMPARISON")
    print("VDD vs DWM-lite, EMA-lambda, Holt-Winters-inspired")
    print("=" * 70)

    steps = 500
    embedding_dim = 32
    n_runs = 30
    scenarios = ["regime_shifts", "mixed_drift", "bursty", "reversion"]

    methods = {
        "VDD": lambda t, e: run_vdd(t, e),
        "DWM-lite": lambda t, e: run_dwm_lite(t, e),
        "EMA-lambda (a=0.1)": lambda t, e: run_ema_lambda(t, e, alpha=0.1),
        "EMA-lambda (a=0.2)": lambda t, e: run_ema_lambda(t, e, alpha=0.2),
        "Holt-Winters": lambda t, e: run_holt_winters(t, e),
        "Recency (0.5)": lambda t, e: run_recency(t, e),
        "Static (0.1)": lambda t, e: run_static(t, e, lam=0.1),
    }

    all_results = {}
    vdd_wins = 0

    for scenario in scenarios:
        print(f"\n{'─' * 60}")
        print(f"Scenario: {scenario}")
        print(f"{'─' * 60}")

        scenario_results = {name: [] for name in methods}

        for seed in range(100, 100 + n_runs):
            truth, embeddings = generate_scenario(scenario, steps, embedding_dim, seed)
            for name, fn in methods.items():
                iae = fn(truth, embeddings)
                scenario_results[name].append(iae)

        vdd_iaes = np.array(scenario_results["VDD"])
        sorted_methods = sorted(scenario_results.keys(),
                                key=lambda m: np.mean(scenario_results[m]))

        print(f"\n  {'Rank':<5} {'Method':<22} {'IAE Mean+-Std':<22} {'95% CI':<24} {'vs VDD p':<14} {'Cohen d'}")
        print(f"  {'─' * 100}")

        for rank, name in enumerate(sorted_methods, 1):
            iaes = np.array(scenario_results[name])
            mean_val = np.mean(iaes)
            std_val = np.std(iaes)
            ci = bootstrap_ci(iaes, n_bootstrap=1000)

            if name != "VDD":
                t_stat, p_val = sp_stats.ttest_rel(vdd_iaes, iaes)
                d = cohens_d(vdd_iaes, iaes)
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                label = effect_label(d)
                print(f"  {rank:<5} {name:<22} {mean_val:>8.1f} +- {std_val:<8.1f} [{ci[0]:>8.1f}, {ci[1]:>8.1f}]   p={p_val:<10.4f} d={d:+.3f} ({label}) {sig}")
            else:
                print(f"  {rank:<5} {name:<22} {mean_val:>8.1f} +- {std_val:<8.1f} [{ci[0]:>8.1f}, {ci[1]:>8.1f}]   {'---':<14} {'---'}")

        vdd_rank = next(i + 1 for i, name in enumerate(sorted_methods) if name == "VDD")
        if vdd_rank == 1:
            vdd_wins += 1

        all_results[scenario] = {}
        for name, iaes in scenario_results.items():
            iaes_arr = np.array(iaes)
            ci = bootstrap_ci(iaes_arr, n_bootstrap=1000)
            entry = {
                "mean": float(np.mean(iaes_arr)),
                "std": float(np.std(iaes_arr)),
                "ci_lower": float(ci[0]),
                "ci_upper": float(ci[1]),
                "values": [float(x) for x in iaes_arr],
                "rank": next(i + 1 for i, m in enumerate(sorted_methods) if m == name),
            }
            if name != "VDD":
                t_stat, p_val = sp_stats.ttest_rel(vdd_iaes, iaes_arr)
                d = cohens_d(vdd_iaes, iaes_arr)
                entry["vs_vdd_p"] = float(p_val)
                entry["vs_vdd_d"] = float(d)
                entry["vs_vdd_d_label"] = effect_label(d)
            all_results[scenario][name] = entry

    print(f"\n{'=' * 70}")
    print("SUMMARY: VDD RANKING ACROSS SCENARIOS")
    print(f"{'=' * 70}")

    adaptive_names = ["DWM-lite", "EMA-lambda (a=0.1)", "EMA-lambda (a=0.2)", "Holt-Winters"]

    for scenario in scenarios:
        data = all_results[scenario]
        ranked = sorted(data.items(), key=lambda x: x[1]["mean"])
        vdd_rank = next(i + 1 for i, (n, _) in enumerate(ranked) if n == "VDD")
        best_name = ranked[0][0]
        print(f"  {scenario:<20} VDD rank: {vdd_rank}/{len(methods)} | Best: {best_name} (IAE={ranked[0][1]['mean']:.1f})")

    print(f"\n  VDD wins (rank 1): {vdd_wins}/{len(scenarios)} scenarios")

    print(f"\n{'─' * 70}")
    print("COHEN'S d: VDD vs EACH ADAPTIVE BASELINE")
    print(f"{'─' * 70}")
    print(f"  {'Baseline':<22}", end="")
    for s in scenarios:
        print(f" {s:<18}", end="")
    print()
    print(f"  {'─' * 94}")

    for baseline in adaptive_names:
        print(f"  {baseline:<22}", end="")
        for scenario in scenarios:
            d = all_results[scenario][baseline]["vs_vdd_d"]
            label = all_results[scenario][baseline]["vs_vdd_d_label"]
            direction = "VDD better" if d < 0 else "baseline better" if d > 0 else "equal"
            print(f" d={d:+.2f} ({label[:3]})", end="  ")
        print()

    vdd_better_count = 0
    total_comparisons = len(adaptive_names) * len(scenarios)
    for scenario in scenarios:
        for baseline in adaptive_names:
            if all_results[scenario]["VDD"]["mean"] <= all_results[scenario][baseline]["mean"]:
                vdd_better_count += 1

    print(f"\n  VDD outperforms/matches adaptive baselines in {vdd_better_count}/{total_comparisons} comparisons")

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    output = {
        "experiment": "31_adaptive_baselines",
        "methodology": {
            "n_runs": n_runs,
            "seeds": "100-129",
            "steps": steps,
            "embedding_dim": embedding_dim,
            "bootstrap_samples": 1000,
            "statistical_tests": "paired t-test, Cohen's d, bootstrap 95% CI",
        },
        "methods": list(methods.keys()),
        "adaptive_baselines": adaptive_names,
        "scenarios": scenarios,
        "results": all_results,
        "summary": {
            "vdd_rank1_count": vdd_wins,
            "vdd_outperforms_adaptive": vdd_better_count,
            "total_adaptive_comparisons": total_comparisons,
        },
    }

    with open(results_dir / "31_adaptive_baselines.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to results/31_adaptive_baselines.json")

    plot_results(all_results, scenarios, methods, results_dir)

    return output


def plot_results(all_results, scenarios, methods, results_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    method_colors = {
        "VDD": "#2ecc71",
        "DWM-lite": "#9b59b6",
        "EMA-lambda (a=0.1)": "#3498db",
        "EMA-lambda (a=0.2)": "#2980b9",
        "Holt-Winters": "#e67e22",
        "Recency (0.5)": "#e74c3c",
        "Static (0.1)": "#95a5a6",
    }

    for i, scenario in enumerate(scenarios):
        ax = axes[i // 2][i % 2]
        data = all_results[scenario]
        sorted_methods = sorted(data.keys(), key=lambda m: data[m]["mean"])

        names = []
        means = []
        ci_lowers = []
        ci_uppers = []
        colors = []

        for m in sorted_methods:
            names.append(m)
            means.append(data[m]["mean"])
            ci_lowers.append(data[m]["mean"] - data[m]["ci_lower"])
            ci_uppers.append(data[m]["ci_upper"] - data[m]["mean"])
            colors.append(method_colors.get(m, "#7f8c8d"))

        y_pos = range(len(names))
        xerr = [ci_lowers, ci_uppers]

        bars = ax.barh(y_pos, means, xerr=xerr, capsize=3,
                       color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)

        for j, m in enumerate(sorted_methods):
            if m == "VDD":
                bars[j].set_edgecolor("#1a8a4a")
                bars[j].set_linewidth(2.0)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("IAE (lower is better)")
        ax.set_title(scenario, fontweight="bold", fontsize=11)
        ax.invert_yaxis()

        vdd_mean = data["VDD"]["mean"]
        ax.axvline(x=vdd_mean, color="#2ecc71", linestyle="--", alpha=0.4, linewidth=1)

    plt.suptitle("Experiment 31: VDD vs Adaptive Baselines\n"
                 "(DWM-lite, EMA-lambda, Holt-Winters) | n=30, 95% CI",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(results_dir / "31_adaptive_baselines.png", dpi=150, bbox_inches="tight")
    print(f"  Saved plot to results/31_adaptive_baselines.png")
    plt.close()


if __name__ == "__main__":
    main()
