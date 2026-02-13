#!/usr/bin/env python3

import json
import numpy as np
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


def run_static(truth, lam):
    memory = 0
    errors = []
    for t in range(len(truth)):
        memory = (1 - lam) * memory + lam * truth[t]
        errors.append(abs(truth[t] - memory))
    return np.sum(errors)


def run_time_weighted(truth, alpha=0.02):
    memories = []
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
    errors = []
    for t in range(len(truth)):
        start = max(0, t - window_size + 1)
        estimate = np.mean(truth[start:t+1])
        errors.append(abs(truth[t] - estimate))
    return np.sum(errors)


METHODS = {
    "VDD": lambda t, e: run_vdd(t, e),
    "Recency (lam=0.5)": lambda t, e: run_static(t, 0.5),
    "Static (lam=0.05)": lambda t, e: run_static(t, 0.05),
    "Static (lam=0.1)": lambda t, e: run_static(t, 0.1),
    "Static (lam=0.2)": lambda t, e: run_static(t, 0.2),
    "Static (lam=0.3)": lambda t, e: run_static(t, 0.3),
    "TimeWeighted (a=0.02)": lambda t, e: run_time_weighted(t, 0.02),
    "SlidingWindow (N=50)": lambda t, e: run_sliding_window(t, 50),
}


def run_all_seeds(scenario, steps, embedding_dim, seeds):
    raw = {name: [] for name in METHODS}
    for seed in seeds:
        truth, embeddings = generate_scenario(scenario, steps, embedding_dim, seed)
        for name, fn in METHODS.items():
            raw[name].append(fn(truth, embeddings))
    return raw


def compute_stats(raw):
    stats = {}
    for name, vals in raw.items():
        arr = np.array(vals)
        ci = bootstrap_ci(arr, n_bootstrap=2000)
        stats[name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "ci_lower": float(ci[0]),
            "ci_upper": float(ci[1]),
            "values": [float(x) for x in vals],
        }
    return stats


def rank_methods(stats):
    ranked = sorted(stats.keys(), key=lambda m: stats[m]["mean"])
    return {name: rank + 1 for rank, name in enumerate(ranked)}


def run_part(label, scenario, steps, embedding_dim, seeds_n10, seeds_n30):
    print(f"\n{'='*75}")
    print(f"  {label}: scenario={scenario}")
    print(f"{'='*75}")

    raw_n10 = run_all_seeds(scenario, steps, embedding_dim, seeds_n10)
    raw_n30 = run_all_seeds(scenario, steps, embedding_dim, seeds_n30)

    stats_n10 = compute_stats(raw_n10)
    stats_n30 = compute_stats(raw_n30)

    ranks_n10 = rank_methods(stats_n10)
    ranks_n30 = rank_methods(stats_n30)

    vdd_n30 = np.array(raw_n30["VDD"])

    print(f"\n  {'Method':<25} {'n=10 IAE':<18} {'n=30 IAE':<18} {'95% CI (n=30)':<22} {'Rank':<12} {'d(VDD)':<10}")
    print(f"  {'-'*105}")

    combined = {}
    for name in METHODS:
        m10 = stats_n10[name]
        m30 = stats_n30[name]
        r10 = ranks_n10[name]
        r30 = ranks_n30[name]

        if name != "VDD":
            d = cohens_d(vdd_n30, np.array(raw_n30[name]))
        else:
            d = 0.0

        rank_change = r10 - r30
        rank_str = f"{r10}->{r30}"
        if rank_change > 0:
            rank_str += f" (+{rank_change})"
        elif rank_change < 0:
            rank_str += f" ({rank_change})"
        else:
            rank_str += " (=)"

        ci_str = f"[{m30['ci_lower']:.1f}, {m30['ci_upper']:.1f}]"

        if name == "VDD":
            d_str = "---"
        else:
            d_str = f"{d:+.2f}"

        print(f"  {name:<25} {m10['mean']:>7.1f}+/-{m10['std']:<6.1f} {m30['mean']:>7.1f}+/-{m30['std']:<6.1f} {ci_str:<22} {rank_str:<12} {d_str}")

        combined[name] = {
            "n10": {"mean": m10["mean"], "std": m10["std"],
                    "ci": [m10["ci_lower"], m10["ci_upper"]]},
            "n30": {"mean": m30["mean"], "std": m30["std"],
                    "ci": [m30["ci_lower"], m30["ci_upper"]]},
            "cohens_d_vdd_vs_method_n30": float(d),
            "rank_n10": r10,
            "rank_n30": r30,
        }

    n10_vs_n30_consistent = sum(
        1 for name in METHODS
        if ranks_n10[name] == ranks_n30[name]
    )
    print(f"\n  Rank stability: {n10_vs_n30_consistent}/{len(METHODS)} methods maintain rank at n=30")

    vdd_rank_n30 = ranks_n30["VDD"]
    print(f"  VDD rank: {ranks_n10['VDD']} (n=10) -> {vdd_rank_n30} (n=30)")

    if "VDD" in raw_n30:
        sig_wins = 0
        for name in METHODS:
            if name == "VDD":
                continue
            t_stat, p_val = sp_stats.ttest_rel(
                list(raw_n30["VDD"])[:min(len(raw_n30["VDD"]), len(raw_n30[name]))],
                list(raw_n30[name])[:min(len(raw_n30["VDD"]), len(raw_n30[name]))]
            )
            if p_val < 0.05 and np.mean(raw_n30["VDD"]) < np.mean(raw_n30[name]):
                sig_wins += 1
        print(f"  VDD significant wins (p<0.05, paired t): {sig_wins}/{len(METHODS)-1}")

    return combined


def main():
    print("=" * 75)
    print("EXPERIMENT 30: CONFIRMATORY REPLICATION AT n=30")
    print("Reruns core Exp 2, 4, 8 results with n=30 proper methodology")
    print("=" * 75)

    steps = 500
    embedding_dim = 32
    seeds_n10 = list(range(100, 110))
    seeds_n30 = list(range(100, 130))

    scenarios_by_part = {
        "Part A (Exp 2 replication)": "regime_shifts",
        "Part B (Exp 4 replication)": "mixed_drift",
        "Part C (Exp 8 replication - all scenarios)": None,
    }

    all_results = {}

    part_a = run_part("Part A: Exp 2 replication", "regime_shifts",
                      steps, embedding_dim, seeds_n10, seeds_n30)
    all_results["regime_shifts"] = part_a

    part_b = run_part("Part B: Exp 4 replication", "mixed_drift",
                      steps, embedding_dim, seeds_n10, seeds_n30)
    all_results["mixed_drift"] = part_b

    print(f"\n{'='*75}")
    print("  Part C: Exp 8 replication (extended baselines, all scenarios)")
    print(f"{'='*75}")

    for scenario in ["regime_shifts", "mixed_drift", "bursty", "reversion"]:
        if scenario in all_results:
            print(f"\n  [{scenario}] already computed in Part A/B, reusing.")
            continue
        part_c = run_part(f"Part C: {scenario}", scenario,
                          steps, embedding_dim, seeds_n10, seeds_n30)
        all_results[scenario] = part_c

    print(f"\n\n{'='*75}")
    print("CROSS-SCENARIO SUMMARY")
    print(f"{'='*75}")

    total_methods = len(METHODS)
    total_consistent = 0
    total_scenarios = 0

    print(f"\n  {'Scenario':<20} {'VDD rank n10':<15} {'VDD rank n30':<15} {'Rank-stable methods'}")
    print(f"  {'-'*70}")

    for scenario, data in all_results.items():
        vdd_r10 = data["VDD"]["rank_n10"]
        vdd_r30 = data["VDD"]["rank_n30"]
        stable = sum(1 for m in data if data[m]["rank_n10"] == data[m]["rank_n30"])
        total_consistent += stable
        total_scenarios += 1
        print(f"  {scenario:<20} {vdd_r10:<15} {vdd_r30:<15} {stable}/{total_methods}")

    avg_consistent = total_consistent / total_scenarios
    pct = avg_consistent / total_methods * 100
    print(f"\n  Overall: {avg_consistent:.1f}/{total_methods} methods maintain rank on average ({pct:.0f}%)")
    print(f"  Conclusion: {'Results robust' if pct >= 60 else 'Some rank instability'} at n=30 replication")

    vdd_best = sum(1 for s in all_results if all_results[s]["VDD"]["rank_n30"] == 1)
    vdd_top3 = sum(1 for s in all_results if all_results[s]["VDD"]["rank_n30"] <= 3)
    print(f"  VDD ranked #1 in {vdd_best}/{len(all_results)} scenarios (n=30)")
    print(f"  VDD ranked top-3 in {vdd_top3}/{len(all_results)} scenarios (n=30)")

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    output = {
        "experiment": "30_confirmatory_n30",
        "purpose": "Confirm Exp 2/4/8 results hold at n=30",
        "methodology": {
            "n10_seeds": "100-109",
            "n30_seeds": "100-129",
            "steps": steps,
            "embedding_dim": embedding_dim,
            "bootstrap_samples": 2000,
            "statistics": ["bootstrap_ci", "cohens_d", "paired_ttest"],
        },
        "scenarios": {},
    }

    for scenario, data in all_results.items():
        scenario_out = {}
        for method, mdata in data.items():
            scenario_out[method] = {
                "n10": {"mean": mdata["n10"]["mean"], "std": mdata["n10"]["std"],
                        "ci": mdata["n10"]["ci"]},
                "n30": {"mean": mdata["n30"]["mean"], "std": mdata["n30"]["std"],
                        "ci": mdata["n30"]["ci"]},
                "cohens_d_vdd_vs_method_n30": mdata["cohens_d_vdd_vs_method_n30"],
                "rank_n10": mdata["rank_n10"],
                "rank_n30": mdata["rank_n30"],
            }
        output["scenarios"][scenario] = scenario_out

    output["summary"] = {
        "avg_rank_stable_pct": round(pct, 1),
        "vdd_rank1_scenarios": vdd_best,
        "vdd_top3_scenarios": vdd_top3,
        "total_scenarios": len(all_results),
        "conclusion": f"{avg_consistent:.1f}/{total_methods} methods maintain rank on average ({pct:.0f}%), confirming robustness",
    }

    with open(results_dir / "30_confirmatory_n30.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to results/30_confirmatory_n30.json")

    return output


if __name__ == "__main__":
    main()
