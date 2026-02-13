#!/usr/bin/env python3
"""
Experiment 29: Sigmoid Hyperparameter Sensitivity Analysis

Ablates the sigmoid steepness (k) and threshold (V_0) parameters that control
how volatility maps to decay rate. Default: k=10, V_0=0.5. Tests whether
performance is robust across a 6x6 grid of configurations.

Grid: k in {1, 2, 5, 10, 20, 50} x V_0 in {0.1, 0.2, 0.3, 0.5, 0.7, 0.9}
Scenarios: regime_shifts, mixed_drift, bursty, reversion
Stats: n=30 runs (seeds 100-129), bootstrap CI
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


def run_vdd(truth, embeddings, k=10.0, v0=0.5, lambda_base=0.2, lambda_max=0.9):
    detector = EmbeddingDistance(curr_window=5, arch_window=50, drift_threshold=0.4)
    memory = 0
    errors = []
    for t in range(len(truth)):
        result = detector.update(embeddings[t])
        v = result.volatility
        lam = lambda_base + (lambda_max - lambda_base) * sigmoid(k * (v - v0))
        memory = (1 - lam) * memory + lam * truth[t]
        errors.append(abs(truth[t] - memory))
    return np.sum(errors)


def main():
    print("=" * 70)
    print("EXPERIMENT 29: SIGMOID HYPERPARAMETER SENSITIVITY")
    print("Ablating k (steepness) and V_0 (threshold) in VDD sigmoid")
    print("=" * 70)

    steps = 500
    embedding_dim = 32
    n_runs = 30
    scenarios = ["regime_shifts", "mixed_drift", "bursty", "reversion"]
    k_values = [1, 2, 5, 10, 20, 50]
    v0_values = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

    grid_results = {}
    default_key = "k=10,V0=0.5"

    total_configs = len(k_values) * len(v0_values)
    print(f"\nGrid: {len(k_values)} k values x {len(v0_values)} V_0 values = {total_configs} configs")
    print(f"Scenarios: {len(scenarios)} | Runs per config: {n_runs} | Seeds: 100-129")
    print(f"Total VDD evaluations: {total_configs * len(scenarios) * n_runs}")

    config_idx = 0
    for k in k_values:
        for v0 in v0_values:
            config_idx += 1
            key = f"k={k},V0={v0}"
            grid_results[key] = {}

            print(f"\r  Config {config_idx}/{total_configs}: {key}", end="", flush=True)

            for scenario in scenarios:
                iaes = []
                for seed in range(100, 100 + n_runs):
                    truth, embeddings = generate_scenario(scenario, steps, embedding_dim, seed)
                    iae = run_vdd(truth, embeddings, k=float(k), v0=float(v0))
                    iaes.append(iae)

                iaes_arr = np.array(iaes)
                ci = bootstrap_ci(iaes_arr)
                grid_results[key][scenario] = {
                    "mean": float(np.mean(iaes_arr)),
                    "std": float(np.std(iaes_arr)),
                    "ci_95": [float(ci[0]), float(ci[1])],
                    "values": [float(x) for x in iaes_arr],
                }

    print(f"\r  All {total_configs} configs complete.{' ' * 40}")

    default_results = grid_results[default_key]

    scenario_analysis = {}
    for scenario in scenarios:
        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario}")
        print(f"{'='*70}")

        ranked = sorted(grid_results.keys(),
                        key=lambda cfg: grid_results[cfg][scenario]["mean"])
        best_key = ranked[0]
        worst_key = ranked[-1]
        default_rank = ranked.index(default_key) + 1

        best_mean = grid_results[best_key][scenario]["mean"]
        worst_mean = grid_results[worst_key][scenario]["mean"]
        default_mean = default_results[scenario]["mean"]

        iae_range_pct = (worst_mean - best_mean) / best_mean * 100
        top_25_cutoff = int(total_configs * 0.25)
        in_top_25 = default_rank <= top_25_cutoff

        print(f"\n  {'Config':<18} {'IAE Mean':>10} {'IAE Std':>10} {'95% CI':<24}")
        print(f"  {'-'*64}")

        for rank_idx, cfg in enumerate(ranked[:5]):
            d = grid_results[cfg][scenario]
            marker = " <-- DEFAULT" if cfg == default_key else ""
            print(f"  {cfg:<18} {d['mean']:>10.2f} {d['std']:>10.2f} [{d['ci_95'][0]:.1f}, {d['ci_95'][1]:.1f}]{marker}")

        if default_rank > 5:
            print(f"  {'...'}")
            d = default_results[scenario]
            print(f"  {default_key:<18} {d['mean']:>10.2f} {d['std']:>10.2f} [{d['ci_95'][0]:.1f}, {d['ci_95'][1]:.1f}] <-- DEFAULT (rank {default_rank})")

        print(f"\n  {'...'}")
        for cfg in ranked[-3:]:
            d = grid_results[cfg][scenario]
            marker = " <-- DEFAULT" if cfg == default_key else ""
            print(f"  {cfg:<18} {d['mean']:>10.2f} {d['std']:>10.2f} [{d['ci_95'][0]:.1f}, {d['ci_95'][1]:.1f}]{marker}")

        print(f"\n  Best config:    {best_key} (IAE={best_mean:.2f})")
        print(f"  Worst config:   {worst_key} (IAE={worst_mean:.2f})")
        print(f"  Default rank:   {default_rank}/{total_configs} ({'TOP 25%' if in_top_25 else 'outside top 25%'})")
        print(f"  IAE range:      {iae_range_pct:.1f}% (worst/best spread)")

        best_vals = np.array(grid_results[best_key][scenario]["values"])
        default_vals = np.array(default_results[scenario]["values"])
        d_effect = cohens_d(default_vals, best_vals)
        t_stat, p_val = sp_stats.ttest_rel(default_vals, best_vals)
        print(f"  Default vs best: d={d_effect:+.3f}, p={p_val:.4f}")

        scenario_analysis[scenario] = {
            "best_config": best_key,
            "best_iae": float(best_mean),
            "worst_config": worst_key,
            "worst_iae": float(worst_mean),
            "default_rank": default_rank,
            "default_iae": float(default_mean),
            "in_top_25": in_top_25,
            "iae_range_pct": float(iae_range_pct),
            "default_vs_best_d": float(d_effect),
            "default_vs_best_p": float(p_val),
            "ranking": ranked,
        }

    print(f"\n{'='*70}")
    print("SUMMARY: SIGMOID SENSITIVITY")
    print(f"{'='*70}")

    top25_count = sum(1 for s in scenario_analysis.values() if s["in_top_25"])
    avg_range = np.mean([s["iae_range_pct"] for s in scenario_analysis.values()])
    avg_rank = np.mean([s["default_rank"] for s in scenario_analysis.values()])

    print(f"\n  Default (k=10, V_0=0.5) summary:")
    for scenario in scenarios:
        sa = scenario_analysis[scenario]
        status = "TOP 25%" if sa["in_top_25"] else f"rank {sa['default_rank']}/{total_configs}"
        print(f"    {scenario:<18} rank {sa['default_rank']:>2}/{total_configs} ({status})  IAE range: {sa['iae_range_pct']:.1f}%")

    print(f"\n  In top 25%: {top25_count}/{len(scenarios)} scenarios")
    print(f"  Average rank: {avg_rank:.1f}/{total_configs}")
    print(f"  Average IAE range across configs: {avg_range:.1f}%")

    if avg_range < 20:
        verdict = "PLATEAU: VDD is robust to sigmoid hyperparameters (<20% IAE range)"
    elif top25_count >= 3:
        verdict = "NEAR-PLATEAU: Default is consistently near-optimal despite moderate sensitivity"
    else:
        verdict = "SENSITIVE: VDD performance depends meaningfully on k and V_0 choices"

    print(f"\n  VERDICT: {verdict}")

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    grid_for_json = {}
    for cfg, scenarios_data in grid_results.items():
        grid_for_json[cfg] = {}
        for sc, data in scenarios_data.items():
            grid_for_json[cfg][sc] = {
                "mean": data["mean"],
                "std": data["std"],
                "ci_95": data["ci_95"],
            }

    sensitivity_for_json = {}
    for sc, data in scenario_analysis.items():
        sensitivity_for_json[sc] = {k: v for k, v in data.items() if k != "ranking"}
        sensitivity_for_json[sc]["ranking"] = data["ranking"][:5] + ["..."] + data["ranking"][-3:]

    output = {
        "grid_results": grid_for_json,
        "best_config": {sc: data["best_config"] for sc, data in scenario_analysis.items()},
        "default_config": {
            "k": 10,
            "V0": 0.5,
            "key": default_key,
            "avg_rank": float(avg_rank),
            "in_top_25_count": top25_count,
        },
        "sensitivity_analysis": sensitivity_for_json,
        "verdict": verdict,
        "methodology": {
            "n_runs": n_runs,
            "seeds": "100-129",
            "steps": steps,
            "embedding_dim": embedding_dim,
            "k_values": k_values,
            "v0_values": v0_values,
            "bootstrap_samples": 1000,
            "lambda_base": 0.2,
            "lambda_max": 0.9,
        },
    }

    with open(results_dir / "29_sigmoid_sensitivity.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to results/29_sigmoid_sensitivity.json")

    plot_heatmaps(grid_results, scenarios, k_values, v0_values, default_key, results_dir)

    return output


def plot_heatmaps(grid_results, scenarios, k_values, v0_values, default_key, results_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    for idx, scenario in enumerate(scenarios):
        ax = axes[idx // 2][idx % 2]
        matrix = np.zeros((len(k_values), len(v0_values)))

        for i, k in enumerate(k_values):
            for j, v0 in enumerate(v0_values):
                key = f"k={k},V0={v0}"
                matrix[i, j] = grid_results[key][scenario]["mean"]

        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', interpolation='nearest')
        ax.set_xticks(range(len(v0_values)))
        ax.set_xticklabels([str(v) for v in v0_values])
        ax.set_yticks(range(len(k_values)))
        ax.set_yticklabels([str(k) for k in k_values])
        ax.set_xlabel("V_0 (threshold)")
        ax.set_ylabel("k (steepness)")

        for i in range(len(k_values)):
            for j in range(len(v0_values)):
                val = matrix[i, j]
                color = 'white' if val > (matrix.max() + matrix.min()) / 2 else 'black'
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7, color=color)

        dk = k_values.index(10)
        dv = v0_values.index(0.5)
        rect = plt.Rectangle((dv - 0.5, dk - 0.5), 1, 1, linewidth=3,
                              edgecolor='blue', facecolor='none', linestyle='--')
        ax.add_patch(rect)

        best_iae = matrix.min()
        best_pos = np.unravel_index(matrix.argmin(), matrix.shape)
        rect_best = plt.Rectangle((best_pos[1] - 0.5, best_pos[0] - 0.5), 1, 1,
                                  linewidth=3, edgecolor='lime', facecolor='none')
        ax.add_patch(rect_best)

        iae_range = (matrix.max() - matrix.min()) / matrix.min() * 100
        ax.set_title(f"{scenario}\n(range: {iae_range:.0f}%  blue=default, green=best)",
                     fontsize=10, fontweight='bold')

        plt.colorbar(im, ax=ax, label="IAE", shrink=0.8)

    plt.suptitle("Experiment 29: Sigmoid Hyperparameter Sensitivity (k vs V_0)\n"
                 "IAE across 4 scenarios, n=30, lower (green) is better",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "29_sigmoid_heatmap.png", dpi=150, bbox_inches='tight')
    print(f"  Saved heatmap to results/29_sigmoid_heatmap.png")
    plt.close()


if __name__ == "__main__":
    main()
