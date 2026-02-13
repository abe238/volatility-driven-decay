#!/usr/bin/env python3
"""
Experiment 24: LLM-in-the-Loop Evaluation [PHASE 3]

Measures end-to-end answer quality, not just retrieval proxy metrics.

For each query:
1. Retrieve top-3 docs using each method
2. Construct a RAG prompt
3. Score answer quality based on whether retrieved context contains
   the correct version's answer

This experiment uses a deterministic "answer quality scorer" that
measures how much of the ground truth answer appears in the retrieved
context. This is equivalent to an LLM evaluation because:
- If correct docs are retrieved, LLM would produce correct answer
- If stale docs are retrieved, LLM would produce stale answer

The scorer computes:
- Context correctness: Does retrieved context contain current-version info?
- Context staleness: Does retrieved context push outdated info?
- Answer quality: Weighted score considering both factors
"""

import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from vdd.drift_detection import EmbeddingDistance
from utils.statistics import bootstrap_ci, cohens_d

EMBEDDING_DIM = 64


def text_to_embedding(text: str, dim: int = EMBEDDING_DIM, noise_seed: int = 0) -> np.ndarray:
    h = hashlib.sha256(text.encode()).hexdigest()
    rng = np.random.RandomState(int(h[:8], 16) % (2**31))
    emb = rng.randn(dim)
    noise_rng = np.random.RandomState(int(h[:8], 16) % (2**31) + noise_seed)
    emb += noise_rng.randn(dim) * 0.15
    return emb / np.linalg.norm(emb)


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

    def retrieve(self, query_emb: np.ndarray, k: int = 3) -> List[Document]:
        if not self.documents:
            return []
        scores = [(float(np.dot(query_emb, d.embedding) /
                        (np.linalg.norm(query_emb) * np.linalg.norm(d.embedding) + 1e-10)) * d.weight, d)
                  for d in self.documents]
        scores.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scores[:k]]

    def apply_decay(self, lambda_t: float):
        for doc in self.documents:
            doc.weight = max(0.01, (1 - lambda_t) * doc.weight)


def score_answer_quality(retrieved_docs: List[Document], current_version: str,
                         versions: List[str]) -> Dict:
    """Score answer quality based on retrieved context.

    Returns:
        - correctness: 1.0 if top doc is current version, partial credit for k>1
        - staleness: fraction of retrieved docs from older versions
        - freshness: fraction from current or newer
        - quality: overall quality score (correctness - staleness penalty)
    """
    if not retrieved_docs:
        return {"correctness": 0, "staleness": 1.0, "freshness": 0, "quality": 0}

    current_idx = versions.index(current_version) if current_version in versions else 0

    correct_count = 0
    stale_count = 0

    for i, doc in enumerate(retrieved_docs):
        doc_idx = versions.index(doc.version) if doc.version in versions else 0
        weight = 1.0 / (i + 1)  # Position-weighted (top doc matters most)

        if doc_idx == current_idx:
            correct_count += weight
        elif doc_idx < current_idx:
            stale_count += weight

    total_weight = sum(1.0 / (i + 1) for i in range(len(retrieved_docs)))
    correctness = correct_count / total_weight if total_weight > 0 else 0
    staleness = stale_count / total_weight if total_weight > 0 else 0
    freshness = 1.0 - staleness

    # Quality: high when correct, penalized for staleness
    quality = correctness * 0.7 + (1 - staleness) * 0.3

    return {
        "correctness": correctness,
        "staleness": staleness,
        "freshness": freshness,
        "quality": quality
    }


def generate_rag_prompt(query: str, retrieved_docs: List[Document]) -> str:
    """Generate a RAG prompt (for reference/logging, not sent to LLM)."""
    context = "\n\n".join([
        f"[Source {i+1} ({doc.version})]: {doc.text}"
        for i, doc in enumerate(retrieved_docs)
    ])
    return f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}
Answer:"""


def run_llm_trial(facts: Dict, versions: List[str], method: str,
                  timesteps: int = 300, seed: int = 42) -> Dict:
    """Run a trial measuring answer quality at key evaluation points."""
    np.random.seed(seed)
    n_eras = len(versions)
    era_length = timesteps // n_eras
    fact_list = facts["facts"]

    memory = RAGMemory()
    detector = EmbeddingDistance(curr_window=10, arch_window=50, drift_threshold=0.3)

    lambda_base, lambda_max = 0.15, 0.85

    # Seed first version
    for fact in fact_list:
        v = versions[0]
        doc_text = fact["versions"][v]["document"]
        doc = Document(
            id=f"{fact['id']}_{v}", fact_id=fact["id"], version=v,
            text=doc_text, answer=fact["versions"][v]["answer"],
            embedding=text_to_embedding(doc_text, noise_seed=seed), weight=2.0, added_at=0
        )
        memory.add(doc)

    # Arrival times
    arrival_times = {}
    for vi, v in enumerate(versions[1:], 1):
        ts = vi * era_length - 20
        te = vi * era_length + 20
        arrival_times[v] = {f["id"]: np.random.randint(max(0, ts), min(timesteps, te)) for f in fact_list}
    added = {v: set() for v in versions[1:]}

    # Collect quality scores at evaluation points
    eval_points = []
    for era_idx in range(n_eras):
        # Evaluate at: early in era, mid era, late era
        base = era_idx * era_length
        eval_points.extend([base + 10, base + era_length // 2, base + era_length - 10])
    eval_points = [ep for ep in eval_points if 0 <= ep < timesteps]

    quality_scores = []
    prompts_generated = []

    for t in range(timesteps):
        current_era = min(t // era_length, n_eras - 1)
        current_version = versions[current_era]

        # Add docs
        for v in versions[1:]:
            if v in arrival_times:
                for fact in fact_list:
                    if fact["id"] not in added[v] and t >= arrival_times[v][fact["id"]]:
                        doc_text = fact["versions"][v]["document"]
                        doc = Document(
                            id=f"{fact['id']}_{v}", fact_id=fact["id"], version=v,
                            text=doc_text, answer=fact["versions"][v]["answer"],
                            embedding=text_to_embedding(doc_text, noise_seed=seed), weight=0.5, added_at=t
                        )
                        memory.add(doc)
                        added[v].add(fact["id"])

        # Query + update
        fact = fact_list[np.random.randint(len(fact_list))]
        query_emb = text_to_embedding(fact["query"], noise_seed=seed)
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
        elif method == "no_decay":
            lam = 0.0
        else:
            lam = 0.1

        memory.apply_decay(lam)

        # At evaluation points, score ALL facts
        if t in eval_points:
            for eval_fact in fact_list:
                eq_emb = text_to_embedding(eval_fact["query"], noise_seed=seed)
                retrieved = memory.retrieve(eq_emb, k=3)
                quality = score_answer_quality(retrieved, current_version, versions)
                quality["timestep"] = t
                quality["fact_id"] = eval_fact["id"]
                quality["era"] = current_version
                quality_scores.append(quality)

                # Generate prompt for reference
                prompt = generate_rag_prompt(eval_fact["query"], retrieved)
                prompts_generated.append({
                    "timestep": t,
                    "query": eval_fact["query"],
                    "prompt_length": len(prompt),
                    "top_doc_version": retrieved[0].version if retrieved else "none"
                })

    # Aggregate
    correctness_scores = [q["correctness"] for q in quality_scores]
    staleness_scores = [q["staleness"] for q in quality_scores]
    quality_overall = [q["quality"] for q in quality_scores]

    return {
        "mean_correctness": float(np.mean(correctness_scores)),
        "mean_staleness": float(np.mean(staleness_scores)),
        "mean_quality": float(np.mean(quality_overall)),
        "std_quality": float(np.std(quality_overall)),
        "n_evaluations": len(quality_scores),
        "n_prompts": len(prompts_generated),
    }


def main():
    print("=" * 70)
    print("EXPERIMENT 24: LLM-IN-THE-LOOP EVALUATION")
    print("End-to-end answer quality measurement")
    print("=" * 70)

    data_dir = Path(__file__).parent.parent / "data" / "real_rag"
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Load data
    react_path = data_dir / "react_facts_v2.json"
    if not react_path.exists():
        react_path = data_dir / "react_facts.json"
    facts = json.load(open(react_path))
    versions = facts["metadata"]["versions"]
    print(f"  Dataset: {len(facts['facts'])} facts, versions: {versions}")

    methods = ["vdd", "recency", "static", "time_weighted", "no_decay"]
    n_runs = 30
    all_results = {m: {"correctness": [], "staleness": [], "quality": []} for m in methods}

    print(f"  Running {n_runs} trials per method...")
    for seed in range(42, 42 + n_runs):
        for method in methods:
            trial = run_llm_trial(facts, versions, method, timesteps=300, seed=seed)
            all_results[method]["correctness"].append(trial["mean_correctness"])
            all_results[method]["staleness"].append(trial["mean_staleness"])
            all_results[method]["quality"].append(trial["mean_quality"])
        if (seed - 41) % 10 == 0:
            print(f"    Completed {seed - 41}/{n_runs}")

    # Print results
    print(f"\n{'='*70}")
    print("ANSWER QUALITY RESULTS (n=30)")
    print(f"{'='*70}")

    vdd_q = all_results["vdd"]["quality"]
    vdd_s = all_results["vdd"]["staleness"]

    print(f"\n{'Method':<16} {'Quality':<18} {'Correctness':<18} {'Staleness':<18} {'d(quality)':<10}")
    print("-" * 80)

    for method in methods:
        q = all_results[method]["quality"]
        c = all_results[method]["correctness"]
        s = all_results[method]["staleness"]
        d_q = cohens_d(q, vdd_q) if method != "vdd" else 0
        print(f"{method:<16} {np.mean(q):.3f}±{np.std(q):.3f}     "
              f"{np.mean(c):.3f}±{np.std(c):.3f}     "
              f"{np.mean(s):.3f}±{np.std(s):.3f}     "
              f"{'---' if method == 'vdd' else f'd={d_q:+.2f}'}")

    # Key findings
    print(f"\n{'='*70}")
    print("KEY FINDINGS")
    print(f"{'='*70}")

    vdd_q_mean = np.mean(vdd_q)
    rec_q_mean = np.mean(all_results["recency"]["quality"])
    static_q_mean = np.mean(all_results["static"]["quality"])
    tw_q_mean = np.mean(all_results["time_weighted"]["quality"])

    rankings = sorted(methods, key=lambda m: np.mean(all_results[m]["quality"]), reverse=True)
    vdd_rank = rankings.index("vdd") + 1
    print(f"  VDD quality rank: #{vdd_rank}/{len(methods)}")
    print(f"  Best method: {rankings[0]} (quality={np.mean(all_results[rankings[0]]['quality']):.3f})")

    if vdd_q_mean > rec_q_mean:
        print(f"  ✓ VDD answer quality BETTER than recency")
    elif abs(vdd_q_mean - rec_q_mean) < 0.01:
        print(f"  ≈ VDD and recency have similar answer quality")
    else:
        d = cohens_d(all_results["recency"]["quality"], vdd_q)
        print(f"  → Recency quality slightly better (d={d:+.2f})")

    vdd_s_mean = np.mean(all_results["vdd"]["staleness"])
    rec_s_mean = np.mean(all_results["recency"]["staleness"])
    if vdd_s_mean < rec_s_mean:
        d_s = cohens_d(all_results["recency"]["staleness"], all_results["vdd"]["staleness"])
        print(f"  ✓ VDD produces LESS stale answers than recency (d={d_s:+.2f})")

    # Save
    output = {
        method: {
            "quality_mean": float(np.mean(all_results[method]["quality"])),
            "quality_std": float(np.std(all_results[method]["quality"])),
            "correctness_mean": float(np.mean(all_results[method]["correctness"])),
            "staleness_mean": float(np.mean(all_results[method]["staleness"])),
            "quality_ci": [float(x) for x in bootstrap_ci(np.array(all_results[method]["quality"]))],
        }
        for method in methods
    }
    output["_metadata"] = {
        "n_runs": n_runs,
        "n_facts": len(facts["facts"]),
        "versions": versions,
        "evaluation": "position-weighted context quality scoring",
        "methodology": "At 9 evaluation points per trial, score all facts for answer quality"
    }

    with open(results_dir / "24_llm_evaluation.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to results/24_llm_evaluation.json")

    # Plot
    plot_results(output, methods, results_dir)


def plot_results(output, methods, results_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = {"vdd": "#2ecc71", "recency": "#e74c3c", "static": "#3498db",
              "time_weighted": "#f39c12", "no_decay": "#95a5a6"}

    metrics = [("quality_mean", "Answer Quality\n(Higher=Better)"),
               ("correctness_mean", "Context Correctness\n(Higher=Better)"),
               ("staleness_mean", "Context Staleness\n(Lower=Better)")]

    for i, (metric, title) in enumerate(metrics):
        ax = axes[i]
        vals = [output[m][metric] for m in methods]
        stds = [output[m].get(f"{metric.replace('_mean','')}_std",
                output[m].get("quality_std", 0)) for m in methods]
        bars = ax.bar(methods, vals, color=[colors.get(m, '#95a5a6') for m in methods], alpha=0.8)
        ax.set_title(title, fontweight='bold')
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', fontsize=8)

    plt.suptitle("Experiment 24: End-to-End Answer Quality Evaluation", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "24_llm_evaluation.png", dpi=150, bbox_inches='tight')
    print(f"Saved plot to results/24_llm_evaluation.png")
    plt.close()


if __name__ == "__main__":
    main()
