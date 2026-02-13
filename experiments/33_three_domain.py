#!/usr/bin/env python3

import json
import numpy as np
import requests
import matplotlib.pyplot as plt
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from vdd.drift_detection import EmbeddingDistance
from utils.statistics import bootstrap_ci, cohens_d

OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
_embedding_cache = {}


def get_real_embedding(text):
    if text in _embedding_cache:
        return _embedding_cache[text]
    resp = requests.post(f"{OLLAMA_URL}/api/embeddings",
                         json={"model": EMBEDDING_MODEL, "prompt": text})
    resp.raise_for_status()
    emb = np.array(resp.json()["embedding"], dtype=np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-10)
    _embedding_cache[text] = emb
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
        scores = []
        for doc in self.documents:
            sim = float(np.dot(query_emb, doc.embedding) /
                       (np.linalg.norm(query_emb) * np.linalg.norm(doc.embedding) + 1e-10))
            scores.append((sim * doc.weight, doc))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scores[:k]]

    def apply_decay(self, lambda_t: float):
        for doc in self.documents:
            doc.weight = max(0.01, (1 - lambda_t) * doc.weight)


def run_trial(facts, versions, method, timesteps=300, seed=42):
    np.random.seed(seed)
    n_eras = len(versions)
    era_length = timesteps // n_eras
    fact_list = facts["facts"]
    memory = RAGMemory()
    detector = EmbeddingDistance(curr_window=10, arch_window=50, drift_threshold=0.3)
    lambda_base, lambda_max = 0.15, 0.85
    correct = []
    stale = []

    for fact in fact_list:
        v = versions[0]
        doc = Document(
            id=f"{fact['id']}_{v}", fact_id=fact["id"], version=v,
            text=fact["versions"][v]["document"], answer=fact["versions"][v]["answer"],
            embedding=get_real_embedding(fact["versions"][v]["document"]),
            weight=2.0, added_at=0
        )
        memory.add(doc)

    arrival_times = {}
    for vi, v in enumerate(versions[1:], 1):
        ts = vi * era_length - 20
        te = vi * era_length + 20
        arrival_times[v] = {
            f["id"]: np.random.randint(max(0, ts), min(timesteps, te))
            for f in fact_list
        }
    added = {v: set() for v in versions[1:]}

    for t in range(timesteps):
        current_era = min(t // era_length, n_eras - 1)
        current_version = versions[current_era]

        for v in versions[1:]:
            if v in arrival_times:
                for fact in fact_list:
                    if fact["id"] not in added[v] and t >= arrival_times[v][fact["id"]]:
                        doc = Document(
                            id=f"{fact['id']}_{v}", fact_id=fact["id"], version=v,
                            text=fact["versions"][v]["document"],
                            answer=fact["versions"][v]["answer"],
                            embedding=get_real_embedding(fact["versions"][v]["document"]),
                            weight=0.5, added_at=t
                        )
                        memory.add(doc)
                        added[v].add(fact["id"])

        fact = fact_list[np.random.randint(len(fact_list))]
        query_emb = get_real_embedding(fact["query"])
        result = detector.update(query_emb)
        v = result.volatility

        if method == "vdd":
            lam = lambda_base + (lambda_max - lambda_base) * sigmoid(10 * (v - 0.5))
        elif method == "recency":
            lam = 0.4
        elif method == "static":
            lam = 0.08
        elif method == "time_weighted":
            lam = 0.02
        elif method == "sliding_window":
            lam = 0.0
            if len(memory.documents) > 80:
                memory.documents = sorted(memory.documents, key=lambda d: d.added_at)[-80:]
        elif method == "no_decay":
            lam = 0.0
        else:
            lam = 0.1

        memory.apply_decay(lam)
        retrieved = memory.retrieve(query_emb, k=1)
        if retrieved:
            correct.append(retrieved[0].version == current_version)
            stale.append(
                versions.index(retrieved[0].version) < versions.index(current_version)
                if retrieved[0].version in versions else False
            )
        else:
            correct.append(False)
            stale.append(False)

    return {"accuracy": float(np.mean(correct)), "staleness": float(np.mean(stale))}


def precache_embeddings(facts, versions):
    count = 0
    for fact in facts["facts"]:
        get_real_embedding(fact["query"])
        count += 1
        for v in versions:
            get_real_embedding(fact["versions"][v]["document"])
            count += 1
    return count


def plot_results(domain_results, methods, results_dir):
    domains = list(domain_results.keys())
    n_domains = len(domains)
    n_methods = len(methods)
    x = np.arange(n_domains)
    width = 0.12
    offsets = np.arange(n_methods) - (n_methods - 1) / 2
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#95a5a6']
    method_labels = {
        'vdd': 'VDD', 'recency': 'Recency', 'static': 'Static',
        'time_weighted': 'Time-Weighted', 'sliding_window': 'Sliding Window',
        'no_decay': 'No Decay'
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for i, method in enumerate(methods):
        accs = [domain_results[d][method]["accuracy_mean"] for d in domains]
        errs = [domain_results[d][method]["accuracy_std"] for d in domains]
        axes[0].bar(x + offsets[i] * width, accs, width, yerr=errs,
                    label=method_labels.get(method, method), color=colors[i], alpha=0.85,
                    capsize=2)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels([d.replace("_", " ").title() for d in domains], fontsize=11)
    axes[0].set_ylabel("Accuracy", fontsize=12)
    axes[0].set_title("Accuracy by Domain and Method", fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].set_ylim(0, 1)

    for i, method in enumerate(methods):
        stales = [domain_results[d][method]["staleness_mean"] for d in domains]
        errs = [domain_results[d][method]["staleness_std"] for d in domains]
        axes[1].bar(x + offsets[i] * width, stales, width, yerr=errs,
                    label=method_labels.get(method, method), color=colors[i], alpha=0.85,
                    capsize=2)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels([d.replace("_", " ").title() for d in domains], fontsize=11)
    axes[1].set_ylabel("Staleness", fontsize=12)
    axes[1].set_title("Staleness by Domain and Method", fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=8, ncol=2)
    axes[1].set_ylim(0, 1)

    plt.suptitle("Experiment 33: Three-Domain Real-World RAG Evaluation (n=30, real embeddings)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "33_three_domain.png", dpi=150, bbox_inches='tight')
    print(f"  Saved plot: results/33_three_domain.png")
    plt.close()


def main():
    print("=" * 70)
    print("EXPERIMENT 33: THREE-DOMAIN REAL-WORLD RAG EVALUATION")
    print("Domains: React (50 facts), Python (30 facts), Node.js (30 facts)")
    print("Methods: vdd, recency, static, time_weighted, sliding_window, no_decay")
    print("Stats: n=30 seeds, real embeddings via nomic-embed-text")
    print("=" * 70)

    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags")
        models = [m["name"] for m in resp.json().get("models", [])]
        if not any(EMBEDDING_MODEL in m for m in models):
            print(f"  ERROR: {EMBEDDING_MODEL} not found. Available: {models}")
            return
        print(f"  Ollama connected. Model: {EMBEDDING_MODEL}")
    except Exception as e:
        print(f"  ERROR: Cannot connect to Ollama: {e}")
        return

    data_dir = Path(__file__).parent.parent / "data" / "real_rag"
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    react_path = data_dir / "react_facts_v2.json"
    if not react_path.exists():
        react_path = data_dir / "react_facts.json"
    python_path = data_dir / "python_facts.json"
    nodejs_path = data_dir / "nodejs_facts.json"

    for p in [react_path, python_path, nodejs_path]:
        if not p.exists():
            print(f"  ERROR: Missing {p}")
            return

    react_facts = json.load(open(react_path))
    python_facts = json.load(open(python_path))
    nodejs_facts = json.load(open(nodejs_path))

    domains = {
        "react": {"facts": react_facts, "versions": react_facts["metadata"]["versions"]},
        "python": {"facts": python_facts, "versions": python_facts["metadata"]["versions"]},
        "nodejs": {"facts": nodejs_facts, "versions": nodejs_facts["metadata"]["versions"]},
    }

    total_facts = sum(len(d["facts"]["facts"]) for d in domains.values())
    print(f"\n  Datasets loaded:")
    for name, d in domains.items():
        print(f"    {name}: {len(d['facts']['facts'])} facts, versions: {d['versions']}")
    print(f"    Total: {total_facts} facts across 3 domains")

    print(f"\n  Pre-caching all embeddings...")
    t0 = time.time()
    total_cached = 0
    for name, d in domains.items():
        n = precache_embeddings(d["facts"], d["versions"])
        total_cached += n
        print(f"    {name}: {n} embeddings cached")
    elapsed = time.time() - t0
    print(f"  Total: {total_cached} embeddings in {elapsed:.1f}s ({total_cached/max(elapsed,0.1):.0f}/s)")

    methods = ["vdd", "recency", "static", "time_weighted", "sliding_window", "no_decay"]
    n_runs = 30
    seeds = list(range(100, 100 + n_runs))

    domain_results = {}
    all_raw = {}

    for domain_name, domain_data in domains.items():
        print(f"\n  Running {domain_name} ({len(domain_data['facts']['facts'])} facts)...")
        raw = {m: {"accuracy": [], "staleness": []} for m in methods}

        for si, seed in enumerate(seeds):
            for method in methods:
                trial = run_trial(domain_data["facts"], domain_data["versions"],
                                  method, seed=seed)
                raw[method]["accuracy"].append(trial["accuracy"])
                raw[method]["staleness"].append(trial["staleness"])
            if (si + 1) % 10 == 0:
                print(f"    {si + 1}/{n_runs} seeds complete")

        domain_results[domain_name] = {}
        for method in methods:
            acc_arr = np.array(raw[method]["accuracy"])
            stale_arr = np.array(raw[method]["staleness"])
            acc_ci = bootstrap_ci(acc_arr)
            stale_ci = bootstrap_ci(stale_arr)
            domain_results[domain_name][method] = {
                "accuracy_mean": round(float(np.mean(acc_arr)), 4),
                "accuracy_std": round(float(np.std(acc_arr)), 4),
                "accuracy_ci95": [round(acc_ci[0], 4), round(acc_ci[1], 4)],
                "staleness_mean": round(float(np.mean(stale_arr)), 4),
                "staleness_std": round(float(np.std(stale_arr)), 4),
                "staleness_ci95": [round(stale_ci[0], 4), round(stale_ci[1], 4)],
            }
        all_raw[domain_name] = raw

    print(f"\n{'='*70}")
    print("PER-DOMAIN RESULTS (n=30)")
    print(f"{'='*70}")

    for domain_name in domains:
        print(f"\n  --- {domain_name.upper()} ---")
        print(f"  {'Method':<16} {'Accuracy':<20} {'Staleness':<20}")
        print(f"  {'-'*56}")
        ranked = sorted(methods, key=lambda m: domain_results[domain_name][m]["accuracy_mean"],
                        reverse=True)
        for method in ranked:
            r = domain_results[domain_name][method]
            print(f"  {method:<16} {r['accuracy_mean']:.3f} +/- {r['accuracy_std']:.3f}   "
                  f"{r['staleness_mean']:.3f} +/- {r['staleness_std']:.3f}")
        vdd_rank = ranked.index("vdd") + 1
        print(f"  VDD rank: {vdd_rank}/{len(methods)}")

    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS (across all domains)")
    print(f"{'='*70}")

    aggregate = {}
    for method in methods:
        all_acc = []
        all_stale = []
        for domain_name in domains:
            all_acc.extend(all_raw[domain_name][method]["accuracy"])
            all_stale.extend(all_raw[domain_name][method]["staleness"])
        acc_arr = np.array(all_acc)
        stale_arr = np.array(all_stale)
        acc_ci = bootstrap_ci(acc_arr)
        aggregate[method] = {
            "accuracy_mean": round(float(np.mean(acc_arr)), 4),
            "accuracy_std": round(float(np.std(acc_arr)), 4),
            "accuracy_ci95": [round(acc_ci[0], 4), round(acc_ci[1], 4)],
            "staleness_mean": round(float(np.mean(stale_arr)), 4),
            "staleness_std": round(float(np.std(stale_arr)), 4),
            "n_observations": len(all_acc),
        }

    agg_ranked = sorted(methods, key=lambda m: aggregate[m]["accuracy_mean"], reverse=True)
    print(f"\n  {'Method':<16} {'Accuracy':<20} {'Staleness':<20} {'N':<6}")
    print(f"  {'-'*62}")
    for method in agg_ranked:
        a = aggregate[method]
        print(f"  {method:<16} {a['accuracy_mean']:.3f} +/- {a['accuracy_std']:.3f}   "
              f"{a['staleness_mean']:.3f} +/- {a['staleness_std']:.3f}   {a['n_observations']}")

    vdd_agg_rank = agg_ranked.index("vdd") + 1
    print(f"\n  VDD aggregate rank: {vdd_agg_rank}/{len(methods)}")

    print(f"\n{'='*70}")
    print("NEVER-WORST ANALYSIS")
    print(f"{'='*70}")

    never_worst = True
    for domain_name in domains:
        ranked = sorted(methods, key=lambda m: domain_results[domain_name][m]["accuracy_mean"],
                        reverse=True)
        vdd_rank = ranked.index("vdd") + 1
        is_worst = vdd_rank == len(methods)
        status = "WORST" if is_worst else "OK"
        print(f"  {domain_name:<10} VDD rank: {vdd_rank}/{len(methods)} [{status}]")
        if is_worst:
            never_worst = False

    print(f"\n  NEVER-WORST claim: {'CONFIRMED' if never_worst else 'VIOLATED'}")

    print(f"\n{'='*70}")
    print("COHEN'S d: VDD vs EACH BASELINE (aggregate)")
    print(f"{'='*70}")

    vdd_all_acc = []
    for domain_name in domains:
        vdd_all_acc.extend(all_raw[domain_name]["vdd"]["accuracy"])

    effect_sizes = {}
    for method in methods:
        if method == "vdd":
            continue
        baseline_all_acc = []
        for domain_name in domains:
            baseline_all_acc.extend(all_raw[domain_name][method]["accuracy"])
        d = cohens_d(np.array(vdd_all_acc), np.array(baseline_all_acc))
        magnitude = "negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
        effect_sizes[method] = {"d": round(d, 4), "magnitude": magnitude}
        print(f"  VDD vs {method:<16} d={d:+.3f} ({magnitude})")

    output = {
        "per_domain": domain_results,
        "aggregate": aggregate,
        "effect_sizes": effect_sizes,
        "rankings": {
            "per_domain": {},
            "aggregate": agg_ranked,
        },
        "never_worst": never_worst,
        "_metadata": {
            "n_runs": n_runs,
            "seeds": "100-129",
            "methods": methods,
            "embedding_model": EMBEDDING_MODEL,
            "domains": {
                name: {
                    "n_facts": len(d["facts"]["facts"]),
                    "versions": d["versions"],
                }
                for name, d in domains.items()
            },
            "total_facts": total_facts,
            "total_observations_per_method": n_runs * len(domains),
        },
    }

    for domain_name in domains:
        ranked = sorted(methods, key=lambda m: domain_results[domain_name][m]["accuracy_mean"],
                        reverse=True)
        output["rankings"]["per_domain"][domain_name] = ranked

    with open(results_dir / "33_three_domain.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved results: results/33_three_domain.json")

    plot_results(domain_results, methods, results_dir)

    print(f"\n{'='*70}")
    print("EXPERIMENT 33 COMPLETE")
    print(f"  Domains: {len(domains)} | Total facts: {total_facts}")
    print(f"  Seeds: {n_runs} | Methods: {len(methods)}")
    print(f"  VDD aggregate rank: {vdd_agg_rank}/{len(methods)}")
    print(f"  Never-worst: {'CONFIRMED' if never_worst else 'VIOLATED'}")
    print(f"{'='*70}")

    return output


if __name__ == "__main__":
    main()
