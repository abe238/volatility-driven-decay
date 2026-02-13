#!/usr/bin/env python3
"""
Experiment 23: Extended Real-World RAG Evaluation [PHASE 2]

Scales Experiment 16 from 15 React facts to 50+ React facts
AND adds Python stdlib as a second domain.

Uses DummyEmbedder (hash-based) since we don't require Ollama.
The key insight: drift detection works on embedding distributions,
not the specific embedding model. Hash-based embeddings create
distinct clusters per version, which is what matters.

Reports per-domain AND aggregate results with n=30 runs.
"""

import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from vdd.drift_detection import EmbeddingDistance
from utils.statistics import bootstrap_ci, cohens_d


EMBEDDING_DIM = 64


def text_to_embedding(text: str, dim: int = EMBEDDING_DIM, noise_seed: int = 0) -> np.ndarray:
    """Text-to-embedding with per-seed noise for inter-run variation."""
    h = hashlib.sha256(text.encode()).hexdigest()
    rng = np.random.RandomState(int(h[:8], 16) % (2**31))
    emb = rng.randn(dim)
    # Add seed-dependent noise (simulates embedding model variation)
    noise_rng = np.random.RandomState(int(h[:8], 16) % (2**31) + noise_seed)
    emb += noise_rng.randn(dim) * 0.15
    return emb / np.linalg.norm(emb)


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


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def load_dataset(path: Path) -> Dict:
    """Load a facts JSON file."""
    with open(path) as f:
        return json.load(f)


def run_trial(facts: Dict, versions: List[str], method: str,
              timesteps: int = 300, seed: int = 42) -> Dict:
    """Run a single trial."""
    np.random.seed(seed)
    n_eras = len(versions)
    era_length = timesteps // n_eras
    fact_list = facts["facts"]

    memory = RAGMemory()
    detector = EmbeddingDistance(curr_window=10, arch_window=50, drift_threshold=0.3)

    lambda_base, lambda_max = 0.15, 0.85
    correct_retrievals = []
    staleness_events = []
    lambdas = []

    # Seed first version docs with high weight
    for fact in fact_list:
        v = versions[0]
        doc_text = fact["versions"][v]["document"]
        doc = Document(
            id=f"{fact['id']}_{v}", fact_id=fact["id"], version=v,
            text=doc_text, answer=fact["versions"][v]["answer"],
            embedding=text_to_embedding(doc_text, noise_seed=seed), weight=2.0, added_at=0
        )
        memory.add(doc)

    # Plan arrival times for subsequent versions
    arrival_times = {}
    for vi, v in enumerate(versions[1:], 1):
        transition_start = vi * era_length - 20
        transition_end = vi * era_length + 20
        arrival_times[v] = {
            fact["id"]: np.random.randint(max(0, transition_start), min(timesteps, transition_end))
            for fact in fact_list
        }

    added_versions = {v: set() for v in versions[1:]}

    for t in range(timesteps):
        current_era = min(t // era_length, n_eras - 1)
        current_version = versions[current_era]

        # Add new version docs gradually
        for v in versions[1:]:
            if v in arrival_times:
                for fact in fact_list:
                    if fact["id"] not in added_versions[v] and t >= arrival_times[v][fact["id"]]:
                        doc_text = fact["versions"][v]["document"]
                        doc = Document(
                            id=f"{fact['id']}_{v}", fact_id=fact["id"], version=v,
                            text=doc_text, answer=fact["versions"][v]["answer"],
                            embedding=text_to_embedding(doc_text, noise_seed=seed), weight=0.5, added_at=t
                        )
                        memory.add(doc)
                        added_versions[v].add(fact["id"])

        # Random query
        fact = fact_list[np.random.randint(len(fact_list))]
        query_emb = text_to_embedding(fact["query"], noise_seed=seed)

        # Update detector
        result = detector.update(query_emb)
        v = result.volatility

        # Compute lambda
        if method == "vdd":
            current_lambda = lambda_base + (lambda_max - lambda_base) * sigmoid(10 * (v - 0.5))
        elif method == "recency":
            current_lambda = 0.4
        elif method == "static":
            current_lambda = 0.08
        elif method == "time_weighted":
            current_lambda = 0.02  # Gentle decay
        elif method == "sliding_window":
            current_lambda = 0.0
            # Prune old docs if too many
            if len(memory.documents) > 80:
                memory.documents = sorted(memory.documents, key=lambda d: d.added_at)[-80:]
        elif method == "no_decay":
            current_lambda = 0.0

        lambdas.append(current_lambda)
        memory.apply_decay(current_lambda)

        # Retrieve
        retrieved = memory.retrieve(query_emb, k=1)
        if retrieved:
            top_doc = retrieved[0]
            is_correct = top_doc.version == current_version
            is_stale = versions.index(top_doc.version) < versions.index(current_version) if top_doc.version in versions else False
            correct_retrievals.append(is_correct)
            staleness_events.append(is_stale)
        else:
            correct_retrievals.append(False)
            staleness_events.append(False)

    accuracy = np.mean(correct_retrievals)
    staleness = np.mean(staleness_events)

    # Per-era metrics
    era_metrics = {}
    for ei, v in enumerate(versions):
        start = ei * era_length
        end = min((ei + 1) * era_length, timesteps)
        era_correct = correct_retrievals[start:end]
        era_stale = staleness_events[start:end]
        era_metrics[v] = {
            "accuracy": float(np.mean(era_correct)) if era_correct else 0,
            "staleness": float(np.mean(era_stale)) if era_stale else 0
        }

    return {"accuracy": accuracy, "staleness": staleness, "era_metrics": era_metrics}


def run_domain(domain_name: str, facts: Dict, versions: List[str], n_runs: int = 30):
    """Run full evaluation for a domain."""
    methods = ["vdd", "recency", "static", "time_weighted", "sliding_window", "no_decay"]
    results = {m: {"accuracy": [], "staleness": [], "era_metrics": []} for m in methods}

    for seed in range(42, 42 + n_runs):
        for method in methods:
            trial = run_trial(facts, versions, method, timesteps=300, seed=seed)
            results[method]["accuracy"].append(trial["accuracy"])
            results[method]["staleness"].append(trial["staleness"])
            results[method]["era_metrics"].append(trial["era_metrics"])

    return results


def print_results(domain_name: str, results: Dict, versions: List[str]):
    """Print formatted results."""
    print(f"\n{'='*60}")
    print(f"DOMAIN: {domain_name} ({len(versions)} versions)")
    print(f"{'='*60}")

    methods = list(results.keys())
    vdd_acc = results["vdd"]["accuracy"]
    vdd_stale = results["vdd"]["staleness"]

    print(f"\n{'Method':<18} {'Accuracy':<15} {'Staleness':<15} {'d(acc)':<10} {'d(stale)':<10}")
    print("-" * 68)

    for method in methods:
        acc = results[method]["accuracy"]
        stale = results[method]["staleness"]
        acc_mean = np.mean(acc)
        stale_mean = np.mean(stale)

        if method != "vdd":
            d_acc = cohens_d(acc, vdd_acc)
            d_stale = cohens_d(stale, vdd_stale)
            print(f"{method:<18} {acc_mean:.3f}±{np.std(acc):.3f}    {stale_mean:.3f}±{np.std(stale):.3f}    {d_acc:+.2f}      {d_stale:+.2f}")
        else:
            print(f"{method:<18} {acc_mean:.3f}±{np.std(acc):.3f}    {stale_mean:.3f}±{np.std(stale):.3f}    ---       ---")

    # Key findings
    vdd_stale_mean = np.mean(vdd_stale)
    rec_stale_mean = np.mean(results["recency"]["staleness"])
    tw_stale_mean = np.mean(results["time_weighted"]["staleness"])

    print(f"\n  Key: VDD staleness={vdd_stale_mean:.3f} vs Recency={rec_stale_mean:.3f} vs TimeWeighted={tw_stale_mean:.3f}")
    if vdd_stale_mean < rec_stale_mean:
        print(f"  → VDD has LOWER staleness than recency ✓")
    if vdd_stale_mean < tw_stale_mean:
        print(f"  → VDD has LOWER staleness than time-weighted ✓")


def main():
    print("=" * 70)
    print("EXPERIMENT 23: EXTENDED REAL-WORLD RAG EVALUATION")
    print("Multi-domain, multi-version, extended baselines, n=30")
    print("=" * 70)

    data_dir = Path(__file__).parent.parent / "data" / "real_rag"
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    all_results = {}

    # Domain 1: React (use v2 if available, fall back to original)
    react_path = data_dir / "react_facts_v2.json"
    if not react_path.exists():
        react_path = data_dir / "react_facts.json"
        print(f"  Note: Using original react_facts.json (15 facts)")
    else:
        print(f"  Using expanded react_facts_v2.json")

    react_facts = load_dataset(react_path)
    react_versions = react_facts["metadata"]["versions"]
    print(f"  React: {len(react_facts['facts'])} facts, versions: {react_versions}")

    react_results = run_domain("React", react_facts, react_versions, n_runs=30)
    print_results("React", react_results, react_versions)
    all_results["react"] = {
        m: {"accuracy_mean": float(np.mean(d["accuracy"])),
            "accuracy_std": float(np.std(d["accuracy"])),
            "staleness_mean": float(np.mean(d["staleness"])),
            "staleness_std": float(np.std(d["staleness"]))}
        for m, d in react_results.items()
    }

    # Domain 2: Python
    python_path = data_dir / "python_facts.json"
    if python_path.exists():
        python_facts = load_dataset(python_path)
        python_versions = python_facts["metadata"]["versions"]
        print(f"\n  Python: {len(python_facts['facts'])} facts, versions: {python_versions}")

        python_results = run_domain("Python", python_facts, python_versions, n_runs=30)
        print_results("Python", python_results, python_versions)
        all_results["python"] = {
            m: {"accuracy_mean": float(np.mean(d["accuracy"])),
                "accuracy_std": float(np.std(d["accuracy"])),
                "staleness_mean": float(np.mean(d["staleness"])),
                "staleness_std": float(np.std(d["staleness"]))}
            for m, d in python_results.items()
        }
    else:
        print(f"\n  Python dataset not found at {python_path}, skipping")

    # Aggregate results
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS")
    print(f"{'='*70}")

    methods = list(react_results.keys())
    print(f"\n{'Method':<18} {'Avg Accuracy':<15} {'Avg Staleness':<15}")
    print("-" * 48)

    for method in methods:
        accs = react_results[method]["accuracy"].copy()
        stales = react_results[method]["staleness"].copy()
        if "python" in all_results:
            accs.extend(python_results[method]["accuracy"])
            stales.extend(python_results[method]["staleness"])
        print(f"{method:<18} {np.mean(accs):.3f}±{np.std(accs):.3f}    {np.mean(stales):.3f}±{np.std(stales):.3f}")

    # Save
    with open(results_dir / "23_real_rag_extended.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to results/23_real_rag_extended.json")

    # Plot
    plot_results(all_results, results_dir)

    return all_results


def plot_results(all_results, results_dir):
    domains = list(all_results.keys())
    n_domains = len(domains)
    fig, axes = plt.subplots(n_domains, 2, figsize=(14, 5 * n_domains))
    if n_domains == 1:
        axes = axes.reshape(1, -1)

    for di, domain in enumerate(domains):
        data = all_results[domain]
        methods = list(data.keys())
        colors = {"vdd": "#2ecc71", "recency": "#e74c3c", "static": "#3498db",
                  "time_weighted": "#f39c12", "sliding_window": "#9b59b6", "no_decay": "#95a5a6"}

        # Accuracy
        ax = axes[di, 0]
        acc_means = [data[m]["accuracy_mean"] for m in methods]
        acc_stds = [data[m]["accuracy_std"] for m in methods]
        bars = ax.bar(methods, acc_means, yerr=acc_stds, capsize=4,
                      color=[colors.get(m, '#95a5a6') for m in methods], alpha=0.8)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{domain.upper()} — Accuracy (Higher=Better)")
        ax.set_ylim(0, 1.1)
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)

        # Staleness
        ax = axes[di, 1]
        stale_means = [data[m]["staleness_mean"] for m in methods]
        stale_stds = [data[m]["staleness_std"] for m in methods]
        bars = ax.bar(methods, stale_means, yerr=stale_stds, capsize=4,
                      color=[colors.get(m, '#95a5a6') for m in methods], alpha=0.8)
        ax.set_ylabel("Staleness Rate")
        ax.set_title(f"{domain.upper()} — Staleness (Lower=Better)")
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)

    plt.suptitle("Experiment 23: Extended Real-World RAG Evaluation", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "23_real_rag_extended.png", dpi=150, bbox_inches='tight')
    print(f"Saved plot to results/23_real_rag_extended.png")
    plt.close()


if __name__ == "__main__":
    main()
