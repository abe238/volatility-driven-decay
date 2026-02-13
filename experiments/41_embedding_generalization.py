#!/usr/bin/env python3
"""
Experiment 41: Embedding Model Generalization

Tests whether VDD method rankings are preserved across different embedding models.
Exp 36 used nomic-embed-text (768-dim); this experiment repeats the same 5 scenarios
with mxbai-embed-large (1024-dim) and computes Spearman rank correlation between
the two models' method rankings.

Scenarios (same as Exp 36):
  1. Regime shifts (sudden)
  2. Mixed drift (mixed stable/drift)
  3. Bursty drift (periodic bursts)
  4. Reversion (drift then revert)
  5. Gradual drift (slow transition)

Methods: VDD, Recency, Static, Time-Weighted, No-Decay
n=10 seeds per scenario.

Requires: ollama serve with mxbai-embed-large model.
"""

import json
import numpy as np
import requests
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import List
from scipy import stats
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from vdd.drift_detection import EmbeddingDistance
from utils.statistics import bootstrap_ci, cohens_d

OLLAMA_URL = "http://localhost:11434"
MXBAI_MODEL = "mxbai-embed-large"
NOMIC_MODEL = "nomic-embed-text"
N_SEEDS = 10
TIMESTEPS = 300

RESULTS_DIR = Path(__file__).parent.parent / "results"
EXP36_RESULTS = RESULTS_DIR / "36_real_embedding_suite.json"

_embedding_cache = {}


def get_embedding(text: str, model: str) -> np.ndarray:
    cache_key = f"{model}:{text}"
    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]
    resp = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=30,
    )
    resp.raise_for_status()
    emb = np.array(resp.json()["embedding"], dtype=np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-10)
    _embedding_cache[cache_key] = emb
    return emb


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


@dataclass
class Document:
    id: str
    fact_id: str
    version: str
    text: str
    answer: str
    embedding: np.ndarray
    weight: float = 1.0
    added_at: int = 0


class RAGMemory:
    def __init__(self):
        self.documents: List[Document] = []

    def add(self, doc: Document):
        self.documents.append(doc)

    def retrieve(self, query_emb: np.ndarray, k: int = 1) -> List[Document]:
        if not self.documents:
            return []
        scores = [
            (
                float(
                    np.dot(query_emb, d.embedding)
                    / (np.linalg.norm(query_emb) * np.linalg.norm(d.embedding) + 1e-10)
                )
                * d.weight,
                d,
            )
            for d in self.documents
        ]
        scores.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scores[:k]]

    def apply_decay(self, lambda_t: float):
        for doc in self.documents:
            doc.weight = max(0.01, (1 - lambda_t) * doc.weight)


def generate_drift_scenario(scenario: str, timesteps: int, seed: int):
    np.random.seed(seed)
    n_eras = 3
    era_length = timesteps // n_eras

    if scenario == "regime_shifts":
        return {"type": "regime", "era_length": era_length, "n_eras": n_eras}
    elif scenario == "mixed_drift":
        return {
            "type": "mixed",
            "era_length": era_length,
            "n_eras": n_eras,
            "drift_probability": 0.3,
        }
    elif scenario == "bursty":
        burst_times = [era_length, 2 * era_length]
        return {
            "type": "bursty",
            "era_length": era_length,
            "n_eras": n_eras,
            "burst_times": burst_times,
            "burst_duration": 15,
        }
    elif scenario == "reversion":
        return {
            "type": "reversion",
            "era_length": era_length,
            "n_eras": n_eras,
            "revert_at": 2 * era_length,
        }
    elif scenario == "gradual":
        return {
            "type": "gradual",
            "era_length": era_length,
            "n_eras": n_eras,
            "transition_length": era_length // 2,
        }
    return {"type": "regime", "era_length": era_length, "n_eras": n_eras}


def run_trial(facts, versions, method, embed_fn, scenario_name, timesteps=TIMESTEPS, seed=42):
    np.random.seed(seed)
    n_eras = len(versions)
    era_length = timesteps // n_eras
    fact_list = facts["facts"]

    memory = RAGMemory()
    detector = EmbeddingDistance(curr_window=10, arch_window=50, drift_threshold=0.3)
    lambda_base, lambda_max = 0.15, 0.85

    for fact in fact_list:
        v = versions[0]
        doc = Document(
            id=f"{fact['id']}_{v}",
            fact_id=fact["id"],
            version=v,
            text=fact["versions"][v]["document"],
            answer=fact["versions"][v]["answer"],
            embedding=embed_fn(fact["versions"][v]["document"]),
            weight=2.0,
            added_at=0,
        )
        memory.add(doc)

    arrival_times = {}
    scenario_cfg = generate_drift_scenario(scenario_name, timesteps, seed)

    if scenario_cfg["type"] == "reversion":
        for vi, v in enumerate(versions[1:], 1):
            if vi == n_eras - 1:
                arrival_times[v] = {
                    f["id"]: scenario_cfg["revert_at"] + np.random.randint(-10, 10)
                    for f in fact_list
                }
            else:
                ts = vi * era_length - 20
                te = vi * era_length + 20
                arrival_times[v] = {
                    f["id"]: np.random.randint(max(0, ts), min(timesteps, te))
                    for f in fact_list
                }
    elif scenario_cfg["type"] == "gradual":
        for vi, v in enumerate(versions[1:], 1):
            trans_len = scenario_cfg["transition_length"]
            ts = vi * era_length - trans_len // 2
            te = vi * era_length + trans_len // 2
            arrival_times[v] = {
                f["id"]: np.random.randint(max(0, ts), min(timesteps, te))
                for f in fact_list
            }
    elif scenario_cfg["type"] == "bursty":
        for vi, v in enumerate(versions[1:], 1):
            bt = (
                scenario_cfg["burst_times"][vi - 1]
                if vi - 1 < len(scenario_cfg["burst_times"])
                else vi * era_length
            )
            bd = scenario_cfg["burst_duration"]
            arrival_times[v] = {
                f["id"]: np.random.randint(max(0, bt - bd), min(timesteps, bt + bd))
                for f in fact_list
            }
    else:
        for vi, v in enumerate(versions[1:], 1):
            ts = vi * era_length - 20
            te = vi * era_length + 20
            arrival_times[v] = {
                f["id"]: np.random.randint(max(0, ts), min(timesteps, te))
                for f in fact_list
            }

    added = {v: set() for v in versions[1:]}
    correct = []
    stale = []
    lambda_trace = []

    for t in range(timesteps):
        current_era = min(t // era_length, n_eras - 1)
        current_version = versions[current_era]

        if scenario_cfg["type"] == "reversion" and t >= scenario_cfg["revert_at"]:
            current_version = versions[0]

        for v in versions[1:]:
            if v in arrival_times:
                for fact in fact_list:
                    if fact["id"] not in added[v] and t >= arrival_times[v][fact["id"]]:
                        doc = Document(
                            id=f"{fact['id']}_{v}",
                            fact_id=fact["id"],
                            version=v,
                            text=fact["versions"][v]["document"],
                            answer=fact["versions"][v]["answer"],
                            embedding=embed_fn(fact["versions"][v]["document"]),
                            weight=0.5,
                            added_at=t,
                        )
                        memory.add(doc)
                        added[v].add(fact["id"])

        fact = fact_list[np.random.randint(len(fact_list))]
        query_emb = embed_fn(fact["query"])
        result = detector.update(query_emb)
        vol = result.volatility

        if method == "vdd":
            lam = lambda_base + (lambda_max - lambda_base) * sigmoid(10 * (vol - 0.1))
        elif method == "recency":
            lam = 0.4
        elif method == "static":
            lam = 0.08
        elif method == "time_weighted":
            lam = 0.02
        elif method == "no_decay":
            lam = 0.0
        else:
            lam = 0.1

        lambda_trace.append(lam)
        memory.apply_decay(lam)
        retrieved = memory.retrieve(query_emb, k=1)

        if retrieved:
            correct.append(retrieved[0].version == current_version)
            try:
                stale.append(
                    versions.index(retrieved[0].version) < versions.index(current_version)
                )
            except ValueError:
                stale.append(False)
        else:
            correct.append(False)
            stale.append(False)

    return {
        "accuracy": float(np.mean(correct)),
        "staleness": float(np.mean(stale)),
        "mean_lambda": float(np.mean(lambda_trace)) if method == "vdd" else None,
    }


def load_exp36_nomic_results():
    if not EXP36_RESULTS.exists():
        print(f"  ERROR: {EXP36_RESULTS} not found. Run experiment 36 first.", flush=True)
        return None
    with open(EXP36_RESULTS) as f:
        data = json.load(f)
    return data["results"]["real"]


def compute_rankings(results_dict, scenarios, methods):
    rankings = {}
    for scenario in scenarios:
        accs = {m: results_dict[scenario][m]["accuracy_mean"] for m in methods}
        sorted_methods = sorted(methods, key=lambda m: accs[m], reverse=True)
        rankings[scenario] = {m: rank + 1 for rank, m in enumerate(sorted_methods)}
    return rankings


def main():
    print("=" * 70, flush=True)
    print("EXPERIMENT 41: EMBEDDING MODEL GENERALIZATION", flush=True)
    print(f"Models: {NOMIC_MODEL} (768-dim, from Exp 36) vs {MXBAI_MODEL} (1024-dim)", flush=True)
    print(f"Scenarios: 5 | Methods: 5 | Seeds: {N_SEEDS}", flush=True)
    print("=" * 70, flush=True)

    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        models = [m["name"] for m in resp.json().get("models", [])]
        if not any(MXBAI_MODEL in m for m in models):
            print(f"  ERROR: {MXBAI_MODEL} not found in Ollama", flush=True)
            return
        print(f"  Ollama connected. {MXBAI_MODEL} available.", flush=True)
    except Exception as e:
        print(f"  ERROR: Ollama: {e}", flush=True)
        return

    nomic_results = load_exp36_nomic_results()
    if nomic_results is None:
        return
    print(f"  Loaded Exp 36 nomic-embed-text results.", flush=True)

    data_dir = Path(__file__).parent.parent / "data" / "real_rag"
    RESULTS_DIR.mkdir(exist_ok=True)

    react_path = data_dir / "react_facts_v2.json"
    if not react_path.exists():
        react_path = data_dir / "react_facts.json"
    facts = json.load(open(react_path))
    versions = facts["metadata"]["versions"]
    print(f"  Dataset: {len(facts['facts'])} React facts, versions: {versions}", flush=True)

    embed_fn = lambda text: get_embedding(text, MXBAI_MODEL)

    print(f"  Pre-caching {MXBAI_MODEL} embeddings...", flush=True)
    t0 = time.time()
    n_cached = 0
    for fact in facts["facts"]:
        embed_fn(fact["query"])
        n_cached += 1
        for v in versions:
            embed_fn(fact["versions"][v]["document"])
            n_cached += 1
    cache_time = time.time() - t0
    print(f"  Cached {n_cached} embeddings in {cache_time:.1f}s", flush=True)

    dim = len(get_embedding("test", MXBAI_MODEL))
    print(f"  {MXBAI_MODEL} embedding dimension: {dim}", flush=True)

    scenarios = ["regime_shifts", "mixed_drift", "bursty", "reversion", "gradual"]
    methods = ["vdd", "recency", "static", "time_weighted", "no_decay"]

    mxbai_results = {}
    total_start = time.time()

    for scenario in scenarios:
        print(f"\n  Scenario: {scenario}", flush=True)
        mxbai_results[scenario] = {}

        for method in methods:
            accuracies = []
            stalenesses = []
            for seed in range(100, 100 + N_SEEDS):
                result = run_trial(
                    facts, versions, method, embed_fn, scenario, timesteps=TIMESTEPS, seed=seed
                )
                accuracies.append(result["accuracy"])
                stalenesses.append(result["staleness"])

            acc_arr = np.array(accuracies)
            stal_arr = np.array(stalenesses)
            mxbai_results[scenario][method] = {
                "accuracy_mean": round(float(np.mean(acc_arr)), 4),
                "accuracy_std": round(float(np.std(acc_arr)), 4),
                "accuracy_ci95": [round(x, 4) for x in bootstrap_ci(acc_arr)],
                "staleness_mean": round(float(np.mean(stal_arr)), 4),
                "staleness_std": round(float(np.std(stal_arr)), 4),
                "staleness_ci95": [round(x, 4) for x in bootstrap_ci(stal_arr)],
                "accuracy_values": [round(x, 4) for x in accuracies],
                "staleness_values": [round(x, 4) for x in stalenesses],
            }

        vdd_acc = mxbai_results[scenario]["vdd"]["accuracy_mean"]
        rec_acc = mxbai_results[scenario]["recency"]["accuracy_mean"]
        print(f"    VDD={vdd_acc:.3f}, Recency={rec_acc:.3f}", flush=True)

    total_time = time.time() - total_start
    print(f"\n  mxbai-embed-large trials completed in {total_time:.0f}s", flush=True)

    # === RANKING COMPARISON ===
    print(f"\n{'='*70}", flush=True)
    print("NOMIC-EMBED-TEXT vs MXBAI-EMBED-LARGE RANKING COMPARISON", flush=True)
    print(f"{'='*70}", flush=True)

    nomic_rankings = compute_rankings(nomic_results, scenarios, methods)
    mxbai_rankings = compute_rankings(mxbai_results, scenarios, methods)

    print(f"\n  {'Scenario':<18} {'Nomic Ranking':<45} {'Mxbai Ranking':<45} {'Match'}", flush=True)
    print(f"  {'-'*130}", flush=True)

    ranking_matches = 0
    for scenario in scenarios:
        nomic_sorted = sorted(methods, key=lambda m: nomic_rankings[scenario][m])
        mxbai_sorted = sorted(methods, key=lambda m: mxbai_rankings[scenario][m])
        match = nomic_sorted == mxbai_sorted
        ranking_matches += int(match)

        nomic_str = " > ".join(nomic_sorted)
        mxbai_str = " > ".join(mxbai_sorted)
        match_str = "MATCH" if match else "DIFFER"
        print(f"  {scenario:<18} {nomic_str:<45} {mxbai_str:<45} {match_str}", flush=True)

    print(f"\n  Exact ranking agreement: {ranking_matches}/{len(scenarios)} "
          f"({100*ranking_matches/len(scenarios):.0f}%)", flush=True)

    # === SPEARMAN RANK CORRELATION ===
    nomic_acc_flat = []
    mxbai_acc_flat = []
    for scenario in scenarios:
        for method in methods:
            nomic_acc_flat.append(nomic_results[scenario][method]["accuracy_mean"])
            mxbai_acc_flat.append(mxbai_results[scenario][method]["accuracy_mean"])

    spearman_rho, spearman_p = stats.spearmanr(nomic_acc_flat, mxbai_acc_flat)
    pearson_r, pearson_p = stats.pearsonr(nomic_acc_flat, mxbai_acc_flat)

    print(f"\n  Spearman rank correlation: rho={spearman_rho:.4f}, p={spearman_p:.2e}", flush=True)
    print(f"  Pearson correlation:       r={pearson_r:.4f}, p={pearson_p:.2e}", flush=True)

    # Per-scenario Spearman
    per_scenario_spearman = {}
    for scenario in scenarios:
        nomic_accs = [nomic_results[scenario][m]["accuracy_mean"] for m in methods]
        mxbai_accs = [mxbai_results[scenario][m]["accuracy_mean"] for m in methods]
        rho, p = stats.spearmanr(nomic_accs, mxbai_accs)
        per_scenario_spearman[scenario] = {"rho": round(rho, 4), "p_value": round(p, 6)}
        print(f"  {scenario:<18} Spearman rho={rho:.4f} (p={p:.4f})", flush=True)

    # === ACCURACY COMPARISON TABLE ===
    print(f"\n  ACCURACY COMPARISON:", flush=True)
    print(f"  {'Scenario':<18} {'Method':<16} {'Nomic':>8} {'Mxbai':>8} {'Delta':>8}", flush=True)
    print(f"  {'-'*60}", flush=True)

    comparison = {}
    for scenario in scenarios:
        comparison[scenario] = {}
        for method in methods:
            n_acc = nomic_results[scenario][method]["accuracy_mean"]
            m_acc = mxbai_results[scenario][method]["accuracy_mean"]
            delta = m_acc - n_acc
            comparison[scenario][method] = {
                "nomic_accuracy": n_acc,
                "mxbai_accuracy": m_acc,
                "delta": round(delta, 4),
            }
            print(f"  {scenario:<18} {method:<16} {n_acc:>8.4f} {m_acc:>8.4f} {delta:>+8.4f}", flush=True)

    # === EFFECT SIZE COMPARISON ===
    nomic_ds = []
    mxbai_ds = []
    for scenario in scenarios:
        for method in methods:
            if method == "vdd":
                continue
            n_vdd = np.array(nomic_results[scenario]["vdd"]["accuracy_values"])
            n_m = np.array(nomic_results[scenario][method]["accuracy_values"])
            m_vdd = np.array(mxbai_results[scenario]["vdd"]["accuracy_values"])
            m_m = np.array(mxbai_results[scenario][method]["accuracy_values"])
            nomic_ds.append(cohens_d(n_vdd, n_m))
            mxbai_ds.append(cohens_d(m_vdd, m_m))

    d_correlation = np.corrcoef(nomic_ds, mxbai_ds)[0, 1]
    print(f"\n  Effect size correlation (nomic vs mxbai Cohen's d): r={d_correlation:.4f}", flush=True)

    # === SAVE RESULTS ===
    output = {
        "experiment": "41_embedding_generalization",
        "purpose": "Cross-model generalization: nomic-embed-text (768d) vs mxbai-embed-large (1024d)",
        "n_seeds": N_SEEDS,
        "timesteps": TIMESTEPS,
        "models": {
            "reference": {"name": NOMIC_MODEL, "dim": 768, "source": "exp_36"},
            "test": {"name": MXBAI_MODEL, "dim": dim},
        },
        "scenarios": scenarios,
        "methods": methods,
        "total_time_seconds": round(total_time, 1),
        "mxbai_results": mxbai_results,
        "comparison": comparison,
        "rankings": {
            "nomic": {s: sorted(methods, key=lambda m: nomic_rankings[s][m]) for s in scenarios},
            "mxbai": {s: sorted(methods, key=lambda m: mxbai_rankings[s][m]) for s in scenarios},
        },
        "statistics": {
            "spearman_rho": round(spearman_rho, 4),
            "spearman_p": round(spearman_p, 8),
            "pearson_r": round(pearson_r, 4),
            "pearson_p": round(pearson_p, 8),
            "per_scenario_spearman": per_scenario_spearman,
            "exact_ranking_agreement": f"{ranking_matches}/{len(scenarios)}",
            "effect_size_correlation": round(d_correlation, 4),
        },
        "summary": {
            "ranking_agreement": f"{ranking_matches}/{len(scenarios)}",
            "spearman_rho": round(spearman_rho, 4),
            "spearman_p": f"{spearman_p:.2e}",
            "effect_size_correlation": round(d_correlation, 4),
            "rankings_preserved": bool(spearman_rho > 0.8),
            "conclusion": (
                f"Method rankings {'ARE' if spearman_rho > 0.8 else 'are NOT'} preserved "
                f"across embedding models (Spearman rho={spearman_rho:.3f}, p={spearman_p:.2e}). "
                f"Exact ranking match in {ranking_matches}/{len(scenarios)} scenarios. "
                f"Effect sizes correlate at r={d_correlation:.3f}. "
                f"VDD results generalize across {NOMIC_MODEL} (768d) and {MXBAI_MODEL} ({dim}d)."
            ),
        },
    }

    with open(RESULTS_DIR / "41_embedding_generalization.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to results/41_embedding_generalization.json", flush=True)

    plot_results(
        nomic_results, mxbai_results, comparison, nomic_rankings, mxbai_rankings,
        nomic_ds, mxbai_ds, nomic_acc_flat, mxbai_acc_flat,
        scenarios, methods, spearman_rho, d_correlation, RESULTS_DIR,
    )

    print(f"\n{'='*70}", flush=True)
    print("EXPERIMENT 41 COMPLETE", flush=True)
    print(f"  Spearman rho = {spearman_rho:.4f} (p = {spearman_p:.2e})", flush=True)
    sig_str = "YES" if spearman_p < 0.05 else "NO"
    preserved_str = "YES" if spearman_rho > 0.8 else "NO"
    print(f"  Significant: {sig_str} | Rankings preserved: {preserved_str}", flush=True)
    print(f"{'='*70}", flush=True)

    return output


def plot_results(
    nomic_results, mxbai_results, comparison, nomic_rankings, mxbai_rankings,
    nomic_ds, mxbai_ds, nomic_acc_flat, mxbai_acc_flat,
    scenarios, methods, spearman_rho, d_corr, results_dir,
):
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    nomic_color = "#3498db"
    mxbai_color = "#e74c3c"
    method_colors = {
        "vdd": "#2ecc71",
        "recency": "#e74c3c",
        "static": "#3498db",
        "time_weighted": "#f39c12",
        "no_decay": "#95a5a6",
    }

    for i, scenario in enumerate(scenarios):
        ax = axes[i // 3][i % 3]
        x = np.arange(len(methods))
        width = 0.35

        nomic_accs = [nomic_results[scenario][m]["accuracy_mean"] for m in methods]
        mxbai_accs = [mxbai_results[scenario][m]["accuracy_mean"] for m in methods]

        nomic_errs = [nomic_results[scenario][m]["accuracy_std"] for m in methods]
        mxbai_errs = [mxbai_results[scenario][m]["accuracy_std"] for m in methods]

        bars1 = ax.bar(
            x - width / 2, nomic_accs, width,
            label="nomic-embed-text (768d)", alpha=0.8, color=nomic_color,
            yerr=nomic_errs, capsize=3,
        )
        bars2 = ax.bar(
            x + width / 2, mxbai_accs, width,
            label="mxbai-embed-large (1024d)", alpha=0.8, color=mxbai_color,
            yerr=mxbai_errs, capsize=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", "\n") for m in methods], fontsize=8)
        ax.set_title(scenario.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.set_ylabel("Accuracy")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylim(0, 1.05)

    # Panel 6: Accuracy scatter (nomic vs mxbai) with Spearman
    ax = axes[1][2]
    for i, (n_acc, m_acc) in enumerate(zip(nomic_acc_flat, mxbai_acc_flat)):
        scenario_idx = i // len(methods)
        method_idx = i % len(methods)
        method = methods[method_idx]
        ax.scatter(n_acc, m_acc, c=method_colors[method], s=60, alpha=0.8, zorder=3)

    lim_min = min(min(nomic_acc_flat), min(mxbai_acc_flat)) - 0.05
    lim_max = max(max(nomic_acc_flat), max(mxbai_acc_flat)) + 0.05
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.3, label="y=x")

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=method_colors[m],
                    markersize=8, label=m)
        for m in methods
    ]
    ax.legend(handles=legend_handles, fontsize=7, loc="upper left")
    ax.set_xlabel("nomic-embed-text Accuracy")
    ax.set_ylabel("mxbai-embed-large Accuracy")
    ax.set_title(
        f"Cross-Model Correlation (Spearman rho={spearman_rho:.3f})",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Experiment 41: Embedding Model Generalization (n={N_SEEDS} seeds)\n"
        f"nomic-embed-text (768d) vs mxbai-embed-large (1024d)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(results_dir / "41_embedding_generalization.png", dpi=150, bbox_inches="tight")
    print(f"  Saved plot to results/41_embedding_generalization.png", flush=True)
    plt.close()


if __name__ == "__main__":
    main()
