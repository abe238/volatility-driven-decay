#!/usr/bin/env python3
"""
Experiment 37: Activation Function Ablation + Additional Baselines

Addresses peer review Issues #7 and #9:
- Issue #9: Sigmoid vs alternatives (linear, exponential, step function)
- Issue #7: Missing baselines (timestamp freshness, LRU, online lambda tuning)

All synthetic (hash-based) to run fast. n=30 seeds.
"""

import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Callable
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from vdd.drift_detection import EmbeddingDistance
from utils.statistics import bootstrap_ci, cohens_d

RESULTS_DIR = Path(__file__).parent.parent / "results"
N_SEEDS = 30
TIMESTEPS = 500
EMBEDDING_DIM = 64


def get_hash_embedding(text: str, dim: int = EMBEDDING_DIM) -> np.ndarray:
    h = hashlib.sha256(text.encode()).hexdigest()
    np.random.seed(int(h[:8], 16) % (2**31))
    emb = np.random.randn(dim).astype(np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-10)
    return emb


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# --- Activation functions ---
def activation_sigmoid(vol, k=10, v0=0.1):
    return sigmoid(k * (vol - v0))

def activation_linear(vol, k=10, v0=0.1):
    return np.clip(k * (vol - v0), 0.0, 1.0)

def activation_exponential(vol, k=10, v0=0.1):
    x = k * (vol - v0)
    if x < 0:
        return 0.0
    return min(1.0, 1.0 - np.exp(-x))

def activation_step(vol, k=10, v0=0.1):
    return 1.0 if vol > v0 else 0.0

def activation_softplus(vol, k=10, v0=0.1):
    x = k * (vol - v0)
    return min(1.0, np.log1p(np.exp(np.clip(x, -500, 20))) / np.log1p(np.exp(np.clip(k * 0.5, -500, 20))))


ACTIVATIONS = {
    "sigmoid": activation_sigmoid,
    "linear": activation_linear,
    "exponential": activation_exponential,
    "step": activation_step,
    "softplus": activation_softplus,
}


@dataclass
class Document:
    id: str
    fact_id: str
    version: int
    text: str
    embedding: np.ndarray
    weight: float = 1.0
    added_at: int = 0
    last_accessed: int = 0


class RAGMemory:
    def __init__(self):
        self.documents: List[Document] = []

    def add(self, doc: Document):
        self.documents.append(doc)

    def retrieve(self, query_emb: np.ndarray, k: int = 1, t: int = 0) -> List[Document]:
        if not self.documents:
            return []
        scores = []
        for d in self.documents:
            sim = float(np.dot(query_emb, d.embedding) /
                       (np.linalg.norm(query_emb) * np.linalg.norm(d.embedding) + 1e-10))
            scores.append((sim * d.weight, d))
        scores.sort(key=lambda x: x[0], reverse=True)
        for _, d in scores[:k]:
            d.last_accessed = t
        return [d for _, d in scores[:k]]

    def apply_decay(self, lambda_t: float):
        for doc in self.documents:
            doc.weight = max(0.01, (1 - lambda_t) * doc.weight)

    def apply_lru(self, max_size: int, t: int):
        if len(self.documents) > max_size:
            self.documents.sort(key=lambda d: d.last_accessed)
            self.documents = self.documents[-max_size:]

    def apply_timestamp_freshness(self, t: int, half_life: float = 50):
        for doc in self.documents:
            age = t - doc.added_at
            doc.weight = np.exp(-0.693 * age / half_life)


def generate_scenario(scenario: str, n_facts: int, n_versions: int, timesteps: int, seed: int):
    np.random.seed(seed)
    era_length = timesteps // n_versions

    facts = []
    for i in range(n_facts):
        versions_data = {}
        for v in range(n_versions):
            text = f"fact_{i}_v{v}_content_{seed}"
            versions_data[v] = {
                "text": text,
                "embedding": get_hash_embedding(text),
            }
        facts.append({"id": i, "versions": versions_data})

    arrival_times = {}
    if scenario == "regime_shifts":
        for v in range(1, n_versions):
            ts = v * era_length - 10
            te = v * era_length + 10
            arrival_times[v] = {i: np.random.randint(max(0, ts), min(timesteps, te))
                               for i in range(n_facts)}
    elif scenario == "mixed_drift":
        for v in range(1, n_versions):
            ts = v * era_length - 30
            te = v * era_length + 30
            arrival_times[v] = {i: np.random.randint(max(0, ts), min(timesteps, te))
                               for i in range(n_facts)}
    elif scenario == "bursty":
        for v in range(1, n_versions):
            bt = v * era_length
            arrival_times[v] = {i: np.random.randint(max(0, bt - 5), min(timesteps, bt + 5))
                               for i in range(n_facts)}
    elif scenario == "gradual":
        for v in range(1, n_versions):
            ts = v * era_length - era_length // 3
            te = v * era_length + era_length // 3
            arrival_times[v] = {i: np.random.randint(max(0, ts), min(timesteps, te))
                               for i in range(n_facts)}
    else:
        for v in range(1, n_versions):
            ts = v * era_length - 15
            te = v * era_length + 15
            arrival_times[v] = {i: np.random.randint(max(0, ts), min(timesteps, te))
                               for i in range(n_facts)}

    return facts, arrival_times, era_length


def compute_iae(errors: List[float]) -> float:
    return float(np.sum(np.abs(errors)))


def run_method(method_name: str, facts, arrival_times, era_length,
               n_versions: int, timesteps: int, seed: int,
               activation_fn: Callable = None, k_param: float = 10,
               collect_volatilities: bool = False) -> float:
    np.random.seed(seed)
    n_facts = len(facts)
    memory = RAGMemory()
    detector = EmbeddingDistance(curr_window=5, arch_window=50, drift_threshold=0.3)
    lambda_base, lambda_max = 0.2, 0.9

    for fact in facts:
        doc = Document(
            id=f"{fact['id']}_v0", fact_id=str(fact["id"]), version=0,
            text=fact["versions"][0]["text"],
            embedding=fact["versions"][0]["embedding"],
            weight=1.0, added_at=0
        )
        memory.add(doc)

    added = {v: set() for v in range(1, n_versions)}
    errors = []
    volatilities = [] if collect_volatilities else None

    for t in range(timesteps):
        current_era = min(t // era_length, n_versions - 1)

        for v in range(1, n_versions):
            if v in arrival_times:
                for fact in facts:
                    fid = fact["id"]
                    if fid not in added[v] and t >= arrival_times[v][fid]:
                        doc = Document(
                            id=f"{fid}_v{v}", fact_id=str(fid), version=v,
                            text=fact["versions"][v]["text"],
                            embedding=fact["versions"][v]["embedding"],
                            weight=0.5, added_at=t
                        )
                        memory.add(doc)
                        added[v].add(fid)

        fact = facts[np.random.randint(n_facts)]
        query_emb = fact["versions"][current_era]["embedding"]
        result = detector.update(query_emb)
        vol = result.volatility

        if collect_volatilities:
            volatilities.append(vol)

        if method_name.startswith("vdd_"):
            act_name = method_name.split("_", 1)[1]
            act_fn = ACTIVATIONS.get(act_name, activation_sigmoid)
            a = act_fn(vol, k=k_param)
            lam = lambda_base + (lambda_max - lambda_base) * a
            memory.apply_decay(lam)
        elif method_name == "recency":
            memory.apply_decay(0.4)
        elif method_name == "static":
            memory.apply_decay(0.08)
        elif method_name == "time_weighted":
            memory.apply_decay(0.02)
        elif method_name == "no_decay":
            pass
        elif method_name == "lru":
            memory.apply_lru(max_size=80, t=t)
        elif method_name == "timestamp_freshness":
            memory.apply_timestamp_freshness(t, half_life=50)
        elif method_name == "online_lambda":
            ema_vol = vol * 0.3 + (1 - 0.3) * getattr(run_method, '_ema_vol', 0.1)
            run_method._ema_vol = ema_vol
            lam = np.clip(ema_vol * 2, 0.05, 0.9)
            memory.apply_decay(lam)
        else:
            memory.apply_decay(0.1)

        retrieved = memory.retrieve(query_emb, k=1, t=t)
        if retrieved:
            error = abs(retrieved[0].version - current_era)
        else:
            error = current_era
        errors.append(error)

    iae = compute_iae(errors)
    if collect_volatilities:
        return iae, volatilities
    return iae


def main():
    print("=" * 70, flush=True)
    print("EXPERIMENT 37: ACTIVATION ABLATION + ADDITIONAL BASELINES", flush=True)
    print(f"Seeds: {N_SEEDS} | Steps: {TIMESTEPS}", flush=True)
    print("=" * 70, flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)

    scenarios = ["regime_shifts", "mixed_drift", "bursty", "gradual"]
    n_facts = 20
    n_versions = 3

    # --- Part 0: Volatility diagnostic ---
    print(f"\n  Volatility diagnostic (seed=100, regime_shifts)...", flush=True)
    facts, arrival_times, era_length = generate_scenario(
        "regime_shifts", n_facts, n_versions, TIMESTEPS, 100)
    _, vols = run_method("vdd_sigmoid", facts, arrival_times, era_length,
                         n_versions, TIMESTEPS, 100, k_param=5, collect_volatilities=True)
    vols_arr = np.array(vols)
    print(f"    Vol range: [{vols_arr.min():.4f}, {vols_arr.max():.4f}]", flush=True)
    print(f"    Vol mean: {vols_arr.mean():.4f}, std: {vols_arr.std():.4f}", flush=True)
    pct_near_thresh = float(np.mean(np.abs(vols_arr - 0.1) < 0.1))
    print(f"    % within 0.1 of V0=0.1: {pct_near_thresh*100:.1f}%", flush=True)

    # --- Part 1: k-sweep activation ablation ---
    k_values = [1, 2, 3, 5, 10]
    print(f"\n{'='*70}", flush=True)
    print("PART 1: ACTIVATION FUNCTION ABLATION (k-sweep)", flush=True)
    print(f"k values tested: {k_values}", flush=True)
    print(f"{'='*70}", flush=True)

    activation_methods = list(ACTIVATIONS.keys())
    k_sweep_results = {}

    for k_val in k_values:
        print(f"\n  --- k={k_val} ---", flush=True)
        k_sweep_results[k_val] = {}

        for scenario in scenarios:
            k_sweep_results[k_val][scenario] = {}
            for act_name in activation_methods:
                method_key = f"vdd_{act_name}"
                iae_values = []
                for seed in range(100, 100 + N_SEEDS):
                    facts, arrival_times, era_length = generate_scenario(
                        scenario, n_facts, n_versions, TIMESTEPS, seed)
                    iae = run_method(method_key, facts, arrival_times, era_length,
                                    n_versions, TIMESTEPS, seed, k_param=k_val)
                    iae_values.append(iae)

                arr = np.array(iae_values)
                k_sweep_results[k_val][scenario][act_name] = {
                    "mean": round(float(np.mean(arr)), 2),
                    "std": round(float(np.std(arr)), 2),
                    "ci95": [round(x, 2) for x in bootstrap_ci(arr)],
                    "values": [round(x, 2) for x in iae_values],
                }

        # Print summary for this k
        print(f"  {'Activation':<14}", end="", flush=True)
        for s in scenarios:
            print(f" {s:>14}", end="")
        print(flush=True)

        for act_name in activation_methods:
            print(f"  {act_name:<14}", end="", flush=True)
            for s in scenarios:
                val = k_sweep_results[k_val][s][act_name]["mean"]
                print(f" {val:>14.2f}", end="")
            print(flush=True)

        # Cohen's d: sigmoid vs each alternative at this k
        for act_name in activation_methods:
            if act_name == "sigmoid":
                continue
            ds = []
            for s in scenarios:
                sig_vals = np.array(k_sweep_results[k_val][s]["sigmoid"]["values"])
                alt_vals = np.array(k_sweep_results[k_val][s][act_name]["values"])
                d = cohens_d(alt_vals, sig_vals)
                ds.append(d)
            md = np.mean(ds)
            label = "sigmoid better" if md > 0 else f"{act_name} better" if md < 0 else "equal"
            print(f"    sigmoid vs {act_name}: d={md:+.3f} ({label})", flush=True)

    # --- Part 2: Additional baselines (at recommended k=5) ---
    print(f"\n{'='*70}", flush=True)
    print("PART 2: ADDITIONAL BASELINES (k=5)", flush=True)
    print(f"{'='*70}", flush=True)

    baseline_methods = ["recency", "static", "time_weighted", "no_decay",
                        "lru", "timestamp_freshness", "online_lambda"]
    all_for_ranking = ["vdd_sigmoid"] + baseline_methods
    baseline_results = {}

    for scenario in scenarios:
        print(f"\n  Scenario: {scenario}", flush=True)
        baseline_results[scenario] = {}

        # VDD sigmoid at k=5
        iae_values = []
        for seed in range(100, 100 + N_SEEDS):
            facts, arrival_times, era_length = generate_scenario(
                scenario, n_facts, n_versions, TIMESTEPS, seed)
            iae = run_method("vdd_sigmoid", facts, arrival_times, era_length,
                            n_versions, TIMESTEPS, seed, k_param=5)
            iae_values.append(iae)
        arr = np.array(iae_values)
        baseline_results[scenario]["vdd_sigmoid"] = {
            "mean": round(float(np.mean(arr)), 2),
            "std": round(float(np.std(arr)), 2),
            "ci95": [round(x, 2) for x in bootstrap_ci(arr)],
            "values": [round(x, 2) for x in iae_values],
        }

        for method in baseline_methods:
            iae_values = []
            for seed in range(100, 100 + N_SEEDS):
                run_method._ema_vol = 0.1
                facts, arrival_times, era_length = generate_scenario(
                    scenario, n_facts, n_versions, TIMESTEPS, seed)
                iae = run_method(method, facts, arrival_times, era_length,
                               n_versions, TIMESTEPS, seed, k_param=5)
                iae_values.append(iae)

            arr = np.array(iae_values)
            baseline_results[scenario][method] = {
                "mean": round(float(np.mean(arr)), 2),
                "std": round(float(np.std(arr)), 2),
                "ci95": [round(x, 2) for x in bootstrap_ci(arr)],
                "values": [round(x, 2) for x in iae_values],
            }

        ranking = sorted(all_for_ranking,
                         key=lambda m: baseline_results[scenario][m]["mean"])
        top3 = ", ".join(
            f"{m}({baseline_results[scenario][m]['mean']:.1f})"
            for m in ranking[:3]
        )
        print(f"    Top 3: {top3}", flush=True)

    # Print baseline table
    print(f"\n  {'Method':<22}", end="", flush=True)
    for s in scenarios:
        print(f" {s:>14}", end="")
    print(f" {'Mean Rank':>10}", flush=True)
    print(f"  {'-'*22} {'-'*14} {'-'*14} {'-'*14} {'-'*14} {'-'*10}", flush=True)

    baseline_ranks = {m: [] for m in all_for_ranking}
    for scenario in scenarios:
        ranked = sorted(all_for_ranking,
                       key=lambda m: baseline_results[scenario][m]["mean"])
        for rank, m in enumerate(ranked, 1):
            baseline_ranks[m].append(rank)

    for method in all_for_ranking:
        label = method.replace("vdd_", "VDD-")
        print(f"  {label:<22}", end="", flush=True)
        for s in scenarios:
            val = baseline_results[s][method]["mean"]
            print(f" {val:>14.1f}", end="")
        mean_rank = np.mean(baseline_ranks[method])
        print(f" {mean_rank:>10.1f}", flush=True)

    # New baselines vs VDD
    print(f"\n  New baselines vs VDD-sigmoid (Cohen's d):", flush=True)
    new_baseline_comparisons = {}
    for method in ["lru", "timestamp_freshness", "online_lambda"]:
        ds = []
        for scenario in scenarios:
            vdd_vals = np.array(baseline_results[scenario]["vdd_sigmoid"]["values"])
            b_vals = np.array(baseline_results[scenario][method]["values"])
            d = cohens_d(b_vals, vdd_vals)
            ds.append(d)
        mean_d = np.mean(ds)
        per_s = {s: round(ds[i], 3) for i, s in enumerate(scenarios)}
        new_baseline_comparisons[method] = {"mean_d": round(mean_d, 3), "per_scenario": per_s}
        verdict = "VDD better" if mean_d > 0 else f"{method} better"
        print(f"    vs {method:<22}: d={mean_d:+.3f} ({verdict})", flush=True)

    # --- Build k-sweep summary for JSON ---
    k_sweep_summary = {}
    for k_val in k_values:
        k_sweep_summary[str(k_val)] = {}
        for act_name in activation_methods:
            mean_iae = np.mean([
                k_sweep_results[k_val][s][act_name]["mean"] for s in scenarios
            ])
            k_sweep_summary[str(k_val)][act_name] = round(float(mean_iae), 2)

    # Sigmoid vs alternatives summary across k
    sigmoid_vs_alt = {}
    for act_name in activation_methods:
        if act_name == "sigmoid":
            continue
        sigmoid_vs_alt[act_name] = {}
        for k_val in k_values:
            ds = []
            for s in scenarios:
                sig_vals = np.array(k_sweep_results[k_val][s]["sigmoid"]["values"])
                alt_vals = np.array(k_sweep_results[k_val][s][act_name]["values"])
                d = cohens_d(alt_vals, sig_vals)
                ds.append(d)
            sigmoid_vs_alt[act_name][str(k_val)] = round(float(np.mean(ds)), 3)

    print(f"\n{'='*70}", flush=True)
    print("K-SWEEP SUMMARY: Mean IAE across all scenarios", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  {'Activation':<14}", end="", flush=True)
    for k_val in k_values:
        print(f"    k={k_val:<3}", end="")
    print(flush=True)
    for act_name in activation_methods:
        print(f"  {act_name:<14}", end="", flush=True)
        for k_val in k_values:
            val = k_sweep_summary[str(k_val)][act_name]
            print(f" {val:>7.2f}", end="")
        print(flush=True)

    print(f"\n  Sigmoid vs alternatives (mean d across scenarios):", flush=True)
    for act_name, k_ds in sigmoid_vs_alt.items():
        print(f"    vs {act_name:<12}:", end="", flush=True)
        for k_val in k_values:
            d_val = k_ds[str(k_val)]
            print(f"  k={k_val}:{d_val:+.3f}", end="")
        print(flush=True)

    # Save results
    output = {
        "experiment": "37_activation_and_baselines",
        "purpose": "Activation function ablation (k-sweep) + additional baselines",
        "n_seeds": N_SEEDS,
        "timesteps": TIMESTEPS,
        "k_values_tested": k_values,
        "activations_tested": activation_methods,
        "new_baselines": ["lru", "timestamp_freshness", "online_lambda"],
        "volatility_diagnostic": {
            "scenario": "regime_shifts",
            "seed": 100,
            "vol_range": [round(float(vols_arr.min()), 4), round(float(vols_arr.max()), 4)],
            "vol_mean": round(float(vols_arr.mean()), 4),
            "vol_std": round(float(vols_arr.std()), 4),
            "pct_within_01_of_v0": round(pct_near_thresh * 100, 1),
        },
        "k_sweep_results": {
            str(k_val): {
                s: {
                    act: k_sweep_results[k_val][s][act]
                    for act in activation_methods
                }
                for s in scenarios
            }
            for k_val in k_values
        },
        "k_sweep_summary": k_sweep_summary,
        "sigmoid_vs_alternatives_by_k": sigmoid_vs_alt,
        "baseline_results": baseline_results,
        "baseline_analysis": {
            "new_baseline_vs_vdd": new_baseline_comparisons,
            "ranks": {m: round(float(np.mean(baseline_ranks[m])), 1)
                     for m in all_for_ranking},
        },
    }

    with open(RESULTS_DIR / "37_activation_and_baselines.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to results/37_activation_and_baselines.json", flush=True)

    # Plot
    plot_results(k_sweep_results, k_sweep_summary, sigmoid_vs_alt,
                 baseline_results, scenarios, activation_methods,
                 all_for_ranking, k_values, RESULTS_DIR)

    print(f"\n{'='*70}", flush=True)
    print("EXPERIMENT 37 COMPLETE", flush=True)
    print(f"{'='*70}", flush=True)
    return output


def plot_results(k_sweep_results, k_sweep_summary, sigmoid_vs_alt,
                 baseline_results, scenarios, activation_methods,
                 baseline_method_list, k_values, results_dir):
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # Left: k-sweep heatmap (activation × k → mean IAE)
    act_names = activation_methods
    data = np.zeros((len(act_names), len(k_values)))
    for i, act in enumerate(act_names):
        for j, k_val in enumerate(k_values):
            data[i, j] = k_sweep_summary[str(k_val)][act]

    im = axes[0].imshow(data, aspect='auto', cmap='RdYlGn_r')
    axes[0].set_xticks(range(len(k_values)))
    axes[0].set_xticklabels([f"k={k}" for k in k_values])
    axes[0].set_yticks(range(len(act_names)))
    axes[0].set_yticklabels(act_names)
    axes[0].set_title("Mean IAE by Activation × k", fontweight='bold')
    for i in range(len(act_names)):
        for j in range(len(k_values)):
            axes[0].text(j, i, f"{data[i,j]:.1f}", ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=axes[0], shrink=0.6)

    # Middle: sigmoid vs alternatives Cohen's d across k
    colors = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    for idx, (act_name, k_ds) in enumerate(sigmoid_vs_alt.items()):
        ds = [k_ds[str(k)] for k in k_values]
        axes[1].plot(k_values, ds, 'o-', label=f"vs {act_name}",
                    color=colors[idx % len(colors)], linewidth=2, markersize=6)
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].axhline(y=0.2, color='green', linestyle=':', alpha=0.3, label='small effect')
    axes[1].axhline(y=-0.2, color='green', linestyle=':', alpha=0.3)
    axes[1].set_xlabel("k (steepness)")
    axes[1].set_ylabel("Cohen's d (positive = sigmoid better)")
    axes[1].set_title("Sigmoid vs Alternatives by k", fontweight='bold')
    axes[1].legend(fontsize=8)

    # Right: Baselines ranked
    means_by_method = []
    for method in baseline_method_list:
        mean_iae = np.mean([baseline_results[s][method]["mean"] for s in scenarios])
        means_by_method.append((method, mean_iae))
    means_by_method.sort(key=lambda x: x[1])

    method_order = [m for m, _ in means_by_method]
    vals = [v for _, v in means_by_method]
    colors2 = ['#2ecc71' if 'vdd' in m else '#3498db' for m in method_order]

    axes[2].barh(range(len(method_order)), vals, color=colors2, alpha=0.8)
    axes[2].set_yticks(range(len(method_order)))
    axes[2].set_yticklabels([m.replace("vdd_", "VDD-") for m in method_order], fontsize=9)
    axes[2].set_xlabel("Mean IAE across scenarios (lower is better)")
    axes[2].set_title("All Baselines Ranked (k=5)", fontweight='bold')
    for i, (m, v) in enumerate(means_by_method):
        axes[2].text(v + 0.5, i, f"{v:.1f}", va='center', fontsize=8)

    plt.suptitle("Experiment 37: Activation Function Ablation & Baselines",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "37_activation_and_baselines.png", dpi=150, bbox_inches='tight')
    print(f"  Saved plot to results/37_activation_and_baselines.png", flush=True)
    plt.close()


if __name__ == "__main__":
    main()
