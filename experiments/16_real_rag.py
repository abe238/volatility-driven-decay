#!/usr/bin/env python3
"""
Experiment 16: Real-World RAG Testing with React Documentation

Tests VDD on real API documentation that becomes stale across versions.
Uses pre-computed Ollama embeddings (nomic-embed-text).

Key hypothesis:
- VDD should excel when topic stays same (React) but knowledge becomes stale
- This is the exact scenario VDD was designed for

REVISED: More challenging scenario where docs arrive gradually and
compete for relevance with older versions.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from vdd.drift_detection import EmbeddingDistance
from utils.statistics import bootstrap_ci, cohens_d, format_ci


@dataclass
class Document:
    """A document in the memory bank."""
    id: str
    fact_id: str
    version: str
    text: str
    answer: str
    embedding: np.ndarray
    weight: float = 1.0
    added_at: int = 0


@dataclass
class Query:
    """A query with ground truth for a specific era."""
    fact_id: str
    text: str
    embedding: np.ndarray
    correct_version: str
    correct_answer: str


class RealRAGMemory:
    """Memory bank for real RAG experiment."""

    def __init__(self):
        self.documents: List[Document] = []

    def add(self, doc: Document):
        self.documents.append(doc)

    def retrieve(self, query_emb: np.ndarray, k: int = 3) -> List[Document]:
        """Retrieve top-k documents by weighted similarity."""
        if not self.documents:
            return []

        scores = []
        for doc in self.documents:
            sim = self._cosine_sim(query_emb, doc.embedding)
            weighted_score = sim * doc.weight
            scores.append((weighted_score, doc))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scores[:k]]

    def apply_decay(self, lambda_t: float):
        """Apply decay to all document weights."""
        for doc in self.documents:
            doc.weight = (1 - lambda_t) * doc.weight
            doc.weight = max(0.01, doc.weight)  # Floor to prevent zero

    def boost_recent(self, recency_bonus: float = 0.1):
        """Small boost to recently added docs (simulates freshness signal)."""
        pass  # Removed to make experiment harder

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


def load_embeddings(data_dir: Path) -> Dict:
    """Load pre-computed embeddings."""
    with open(data_dir / "embeddings.json") as f:
        return json.load(f)


def load_facts(data_dir: Path) -> Dict:
    """Load React facts."""
    with open(data_dir / "react_facts.json") as f:
        return json.load(f)


def get_current_version(t: int, timesteps: int = 300) -> str:
    """Get current 'truth' version based on timestep."""
    era_length = timesteps // 3
    if t < era_length:
        return "v16"
    elif t < 2 * era_length:
        return "v17"
    else:
        return "v18"


def run_trial(
    embeddings: Dict,
    facts: Dict,
    method: str,
    timesteps: int = 300,
    seed: int = 42
) -> Dict:
    """
    Run a single trial with MORE CHALLENGING conditions:
    1. New docs start with LOWER weight (0.3) - must "prove" relevance
    2. Docs arrive GRADUALLY over transition period
    3. Old docs start with HIGH weight (accumulated trust)
    """
    np.random.seed(seed)

    memory = RealRAGMemory()
    detector = EmbeddingDistance(curr_window=10, arch_window=50, drift_threshold=0.3)

    lambda_base, lambda_max = 0.15, 0.85  # Slightly less aggressive
    fact_list = facts["facts"]

    staleness_events = []
    correct_retrievals = []
    lambdas = []

    # Seed v16 documents with HIGH initial weight (established trust)
    for fact in fact_list:
        doc_key = f"{fact['id']}_v16"
        doc_data = embeddings["documents"][doc_key]
        doc = Document(
            id=doc_key,
            fact_id=fact["id"],
            version="v16",
            text=doc_data["text"],
            answer=doc_data["answer"],
            embedding=np.array(doc_data["embedding"]),
            weight=2.0,  # High initial weight - established docs
            added_at=0
        )
        memory.add(doc)

    # Transition windows: docs arrive gradually
    # v17 docs arrive between t=80-120
    # v18 docs arrive between t=180-220
    v17_arrival_times = {fact["id"]: np.random.randint(80, 121) for fact in fact_list}
    v18_arrival_times = {fact["id"]: np.random.randint(180, 221) for fact in fact_list}

    v17_added = set()
    v18_added = set()

    for t in range(timesteps):
        current_version = get_current_version(t, timesteps)

        # Gradually add v17 docs
        for fact in fact_list:
            if fact["id"] not in v17_added and t >= v17_arrival_times[fact["id"]]:
                doc_key = f"{fact['id']}_v17"
                doc_data = embeddings["documents"][doc_key]
                doc = Document(
                    id=doc_key,
                    fact_id=fact["id"],
                    version="v17",
                    text=doc_data["text"],
                    answer=doc_data["answer"],
                    embedding=np.array(doc_data["embedding"]),
                    weight=0.5,  # NEW docs start with LOW weight
                    added_at=t
                )
                memory.add(doc)
                v17_added.add(fact["id"])

        # Gradually add v18 docs
        for fact in fact_list:
            if fact["id"] not in v18_added and t >= v18_arrival_times[fact["id"]]:
                doc_key = f"{fact['id']}_v18"
                doc_data = embeddings["documents"][doc_key]
                doc = Document(
                    id=doc_key,
                    fact_id=fact["id"],
                    version="v18",
                    text=doc_data["text"],
                    answer=doc_data["answer"],
                    embedding=np.array(doc_data["embedding"]),
                    weight=0.5,  # NEW docs start with LOW weight
                    added_at=t
                )
                memory.add(doc)
                v18_added.add(fact["id"])

        # Random query
        fact = fact_list[np.random.randint(len(fact_list))]
        query_emb = np.array(embeddings["queries"][fact["id"]])

        # Update detector
        result = detector.update(query_emb)
        v = result.volatility

        # Compute lambda based on method
        if method == "vdd":
            current_lambda = lambda_base + (lambda_max - lambda_base) * v
        elif method == "recency":
            current_lambda = 0.4
        elif method == "static":
            current_lambda = 0.08
        elif method == "no_decay":
            current_lambda = 0.0
        else:
            raise ValueError(f"Unknown method: {method}")

        lambdas.append(current_lambda)

        # Apply decay BEFORE retrieval (decays old docs)
        memory.apply_decay(current_lambda)

        # Retrieve
        retrieved = memory.retrieve(query_emb, k=1)

        if retrieved:
            top_doc = retrieved[0]
            is_correct = top_doc.version == current_version
            is_stale = top_doc.version < current_version

            correct_retrievals.append(is_correct)
            staleness_events.append(is_stale)
        else:
            correct_retrievals.append(False)
            staleness_events.append(False)

    # Compute metrics
    accuracy = np.mean(correct_retrievals)
    staleness_rate = np.mean(staleness_events)

    # Per-era metrics
    era_length = timesteps // 3
    era_metrics = {}
    for era, (start, end) in [("v16", (0, era_length)),
                               ("v17", (era_length, 2*era_length)),
                               ("v18", (2*era_length, timesteps))]:
        era_correct = correct_retrievals[start:end]
        era_stale = staleness_events[start:end]
        era_metrics[era] = {
            "accuracy": np.mean(era_correct) if era_correct else 0,
            "staleness": np.mean(era_stale) if era_stale else 0
        }

    return {
        "accuracy": accuracy,
        "staleness_rate": staleness_rate,
        "lambdas": np.array(lambdas),
        "era_metrics": era_metrics,
        "correct_retrievals": correct_retrievals,
        "staleness_events": staleness_events
    }


def run_experiment():
    """Run the full real RAG experiment."""
    print("=" * 60)
    print("Experiment 16: Real-World RAG Testing (Challenging Mode)")
    print("=" * 60)
    print("\nUsing React documentation across v16, v17, v18")
    print("Challenge: Old docs have high trust, new docs must prove value")
    print("Docs arrive gradually during transition windows\n")

    data_dir = Path(__file__).parent.parent / "data" / "real_rag"
    embeddings = load_embeddings(data_dir)
    facts = load_facts(data_dir)

    print(f"Loaded {len(embeddings['documents'])} documents")
    print(f"Loaded {len(embeddings['queries'])} queries")

    methods = ["vdd", "recency", "static", "no_decay"]
    n_runs = 10
    timesteps = 300

    results = {m: {"accuracy": [], "staleness": [], "era_metrics": []} for m in methods}

    print(f"\nRunning {n_runs} trials per method...")
    for seed in range(42, 42 + n_runs):
        for method in methods:
            trial = run_trial(embeddings, facts, method, timesteps, seed)
            results[method]["accuracy"].append(trial["accuracy"])
            results[method]["staleness"].append(trial["staleness_rate"])
            results[method]["era_metrics"].append(trial["era_metrics"])

        if (seed - 42 + 1) % 5 == 0:
            print(f"  Completed {seed - 42 + 1}/{n_runs} trials")

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS: ACCURACY (Higher = Better)")
    print(f"{'='*60}")
    print(f"{'Method':<15} {'Accuracy':<20} {'95% CI':<20}")
    print("-" * 55)

    for method in methods:
        acc = results[method]["accuracy"]
        mean = np.mean(acc)
        ci = bootstrap_ci(np.array(acc))
        print(f"{method:<15} {mean:.3f} ± {np.std(acc):.3f}      [{ci[0]:.3f}, {ci[1]:.3f}]")

    print(f"\n{'='*60}")
    print("RESULTS: STALENESS RATE (Lower = Better)")
    print(f"{'='*60}")
    print(f"{'Method':<15} {'Staleness':<20} {'95% CI':<20}")
    print("-" * 55)

    for method in methods:
        stale = results[method]["staleness"]
        mean = np.mean(stale)
        ci = bootstrap_ci(np.array(stale))
        print(f"{method:<15} {mean:.3f} ± {np.std(stale):.3f}      [{ci[0]:.3f}, {ci[1]:.3f}]")

    # Per-era breakdown
    print(f"\n{'='*60}")
    print("PER-ERA ACCURACY BREAKDOWN")
    print(f"{'='*60}")
    print(f"{'Method':<12} {'v16 Era':<12} {'v17 Era':<12} {'v18 Era':<12}")
    print("-" * 50)

    for method in methods:
        v16_acc = np.mean([em["v16"]["accuracy"] for em in results[method]["era_metrics"]])
        v17_acc = np.mean([em["v17"]["accuracy"] for em in results[method]["era_metrics"]])
        v18_acc = np.mean([em["v18"]["accuracy"] for em in results[method]["era_metrics"]])
        print(f"{method:<12} {v16_acc:.3f}        {v17_acc:.3f}        {v18_acc:.3f}")

    # Effect sizes
    print(f"\n{'='*60}")
    print("EFFECT SIZES (Cohen's d vs VDD)")
    print(f"{'='*60}")

    vdd_acc = results["vdd"]["accuracy"]
    vdd_stale = results["vdd"]["staleness"]

    for method in ["recency", "static", "no_decay"]:
        d_acc = cohens_d(results[method]["accuracy"], vdd_acc)
        d_stale = cohens_d(results[method]["staleness"], vdd_stale)
        interp_acc = "large" if abs(d_acc) > 0.8 else "medium" if abs(d_acc) > 0.5 else "small"
        interp_stale = "large" if abs(d_stale) > 0.8 else "medium" if abs(d_stale) > 0.5 else "small"
        print(f"{method:<12} Acc d={d_acc:+.2f} ({interp_acc:<6})  Stale d={d_stale:+.2f} ({interp_stale})")

    # Key finding
    print(f"\n{'='*60}")
    print("KEY FINDINGS")
    print(f"{'='*60}")

    vdd_stale_mean = np.mean(results["vdd"]["staleness"])
    rec_stale_mean = np.mean(results["recency"]["staleness"])
    static_stale_mean = np.mean(results["static"]["staleness"])
    vdd_acc_mean = np.mean(results["vdd"]["accuracy"])
    rec_acc_mean = np.mean(results["recency"]["accuracy"])

    if vdd_stale_mean < rec_stale_mean - 0.01:
        print("✅ VDD has LOWER staleness than recency!")
        d = cohens_d([rec_stale_mean], [vdd_stale_mean])
        print(f"   VDD: {vdd_stale_mean:.3f} vs Recency: {rec_stale_mean:.3f}")
    elif abs(vdd_stale_mean - rec_stale_mean) < 0.01:
        print("≈ VDD and recency have similar staleness")
        print(f"   VDD: {vdd_stale_mean:.3f} vs Recency: {rec_stale_mean:.3f}")
    else:
        print("⚠️ Recency has lower staleness than VDD")
        print(f"   VDD: {vdd_stale_mean:.3f} vs Recency: {rec_stale_mean:.3f}")

    if static_stale_mean > 0:
        improvement_vs_static = (static_stale_mean - vdd_stale_mean) / static_stale_mean * 100
        print(f"\nVDD staleness reduction vs static: {improvement_vs_static:.1f}%")

    return results


def plot_results(results: Dict):
    """Create visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    methods = list(results.keys())
    colors = {"vdd": "green", "recency": "red", "static": "blue", "no_decay": "gray"}

    # Plot 1: Accuracy comparison
    ax = axes[0, 0]
    acc_means = [np.mean(results[m]["accuracy"]) for m in methods]
    acc_stds = [np.std(results[m]["accuracy"]) for m in methods]
    cis = [1.96 * s / np.sqrt(len(results[m]["accuracy"])) for m, s in zip(methods, acc_stds)]

    bars = ax.bar(methods, acc_means, yerr=cis, capsize=6,
                  color=[colors[m] for m in methods], alpha=0.7)
    ax.set_ylabel("Accuracy")
    ax.set_title("Retrieval Accuracy\n(Higher = Better)")
    ax.set_ylim(0, 1.1)

    for bar, mean in zip(bars, acc_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{mean:.2f}', ha='center', fontweight='bold', fontsize=9)

    # Plot 2: Staleness comparison
    ax = axes[0, 1]
    stale_means = [np.mean(results[m]["staleness"]) for m in methods]
    stale_stds = [np.std(results[m]["staleness"]) for m in methods]
    cis = [1.96 * s / np.sqrt(len(results[m]["staleness"])) for m, s in zip(methods, stale_stds)]

    bars = ax.bar(methods, stale_means, yerr=cis, capsize=6,
                  color=[colors[m] for m in methods], alpha=0.7)
    ax.set_ylabel("Staleness Rate")
    ax.set_title("Staleness Rate\n(Lower = Better)")

    for bar, mean in zip(bars, stale_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mean:.2f}', ha='center', fontweight='bold', fontsize=9)

    # Plot 3: Per-era accuracy
    ax = axes[1, 0]
    x = np.arange(3)
    width = 0.2
    eras = ["v16", "v17", "v18"]

    for i, method in enumerate(methods):
        era_accs = [np.mean([em[era]["accuracy"] for em in results[method]["era_metrics"]]) for era in eras]
        ax.bar(x + i * width, era_accs, width, label=method, color=colors[method], alpha=0.7)

    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Era Accuracy Breakdown")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(["v16 Era\n(t=0-99)", "v17 Era\n(t=100-199)", "v18 Era\n(t=200-299)"])
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)

    # Plot 4: Summary
    ax = axes[1, 1]
    ax.axis('off')

    vdd_acc = np.mean(results["vdd"]["accuracy"])
    rec_acc = np.mean(results["recency"]["accuracy"])
    static_acc = np.mean(results["static"]["accuracy"])
    vdd_stale = np.mean(results["vdd"]["staleness"])
    rec_stale = np.mean(results["recency"]["staleness"])
    static_stale = np.mean(results["static"]["staleness"])

    winner_stale = "VDD" if vdd_stale < rec_stale else "Recency" if rec_stale < vdd_stale else "Tie"
    winner_acc = "VDD" if vdd_acc > rec_acc else "Recency" if rec_acc > vdd_acc else "Tie"

    summary = f"""
    REAL-WORLD RAG EXPERIMENT
    ═══════════════════════════════════

    Dataset: React API v16 → v17 → v18
    Challenge: Old docs start with high trust
    New docs arrive gradually, start weak

    ═══════════════════════════════════
    ACCURACY (Higher = Better)

    VDD:      {vdd_acc:.3f}
    Recency:  {rec_acc:.3f}
    Static:   {static_acc:.3f}
    Winner:   {winner_acc}

    ═══════════════════════════════════
    STALENESS (Lower = Better)

    VDD:      {vdd_stale:.3f}
    Recency:  {rec_stale:.3f}
    Static:   {static_stale:.3f}
    Winner:   {winner_stale}

    ═══════════════════════════════════
    """

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / "16_real_rag.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {results_dir / '16_real_rag.png'}")
    plt.close()


def main():
    results = run_experiment()
    plot_results(results)
    return results


if __name__ == "__main__":
    main()
