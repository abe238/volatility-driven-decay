#!/usr/bin/env python3
"""
Experiment 28: Actual LLM-in-the-Loop Evaluation

Uses nomic-embed-text for retrieval and llama3.1:8b for generation.
LLM generates answers from retrieved context, then we judge quality
by checking if the answer contains correct-version key terms.

Optimized for speed: n=5 seeds, 10 eval facts, aggressive caching.
"""

import json
import hashlib
import numpy as np
import requests
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
import sys
import time
import re

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from vdd.drift_detection import EmbeddingDistance
from utils.statistics import bootstrap_ci, cohens_d

OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1:8b"

_embedding_cache = {}
_llm_cache = {}


def get_embedding(text: str) -> np.ndarray:
    if text in _embedding_cache:
        return _embedding_cache[text]
    resp = requests.post(f"{OLLAMA_URL}/api/embeddings",
                         json={"model": EMBEDDING_MODEL, "prompt": text},
                         timeout=30)
    resp.raise_for_status()
    emb = np.array(resp.json()["embedding"], dtype=np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-10)
    _embedding_cache[text] = emb
    return emb


def llm_generate(prompt: str, max_tokens: int = 100) -> str:
    if prompt in _llm_cache:
        return _llm_cache[prompt]
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/generate",
                             json={"model": LLM_MODEL, "prompt": prompt,
                                   "stream": False,
                                   "options": {"num_predict": max_tokens, "temperature": 0.0}},
                             timeout=60)
        resp.raise_for_status()
        result = resp.json()["response"].strip()
    except Exception:
        result = "[error]"
    _llm_cache[prompt] = result
    return result


def score_answer(llm_answer: str, ground_truth_answer: str, question: str) -> float:
    """Score by keyword overlap between LLM answer and ground truth."""
    gt_words = set(re.findall(r'\b\w+\b', ground_truth_answer.lower()))
    answer_words = set(re.findall(r'\b\w+\b', llm_answer.lower()))
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at',
                  'to', 'for', 'of', 'with', 'by', 'and', 'or', 'not', 'it', 'that',
                  'this', 'be', 'has', 'have', 'had', 'do', 'does', 'did', 'will',
                  'can', 'could', 'should', 'would', 'may', 'use', 'used', 'using'}
    gt_keywords = gt_words - stop_words
    if not gt_keywords:
        return 0.5
    overlap = len(gt_keywords & answer_words)
    return min(1.0, overlap / max(len(gt_keywords) * 0.5, 1))


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


def run_llm_trial(facts: Dict, versions: List[str], method: str,
                  timesteps: int = 300, seed: int = 42,
                  n_eval_facts: int = 10) -> Dict:
    np.random.seed(seed)
    n_eras = len(versions)
    era_length = timesteps // n_eras
    fact_list = facts["facts"]

    memory = RAGMemory()
    detector = EmbeddingDistance(curr_window=10, arch_window=50, drift_threshold=0.3)
    lambda_base, lambda_max = 0.15, 0.85

    for fact in fact_list:
        v = versions[0]
        doc_text = fact["versions"][v]["document"]
        doc = Document(
            id=f"{fact['id']}_{v}", fact_id=fact["id"], version=v,
            text=doc_text, answer=fact["versions"][v]["answer"],
            embedding=get_embedding(doc_text), weight=2.0, added_at=0
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

    eval_points = [era_idx * era_length + era_length // 2 for era_idx in range(n_eras)]
    eval_points = [ep for ep in eval_points if 0 <= ep < timesteps]

    llm_scores = []
    det_scores = []
    n_llm_calls = 0

    for t in range(timesteps):
        current_era = min(t // era_length, n_eras - 1)
        current_version = versions[current_era]

        for v in versions[1:]:
            if v in arrival_times:
                for fact in fact_list:
                    if fact["id"] not in added[v] and t >= arrival_times[v][fact["id"]]:
                        doc_text = fact["versions"][v]["document"]
                        doc = Document(
                            id=f"{fact['id']}_{v}", fact_id=fact["id"], version=v,
                            text=doc_text, answer=fact["versions"][v]["answer"],
                            embedding=get_embedding(doc_text), weight=0.5, added_at=t
                        )
                        memory.add(doc)
                        added[v].add(fact["id"])

        fact = fact_list[np.random.randint(len(fact_list))]
        query_emb = get_embedding(fact["query"])
        result = detector.update(query_emb)
        vol = result.volatility

        if method == "vdd":
            lam = lambda_base + (lambda_max - lambda_base) * sigmoid(10 * (vol - 0.5))
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

        if t in eval_points:
            for eval_fact in fact_list[:n_eval_facts]:
                eq_emb = get_embedding(eval_fact["query"])
                retrieved = memory.retrieve(eq_emb, k=3)

                if not retrieved:
                    llm_scores.append(0.0)
                    det_scores.append(0.0)
                    continue

                context = "\n".join([f"- {doc.text[:200]}" for doc in retrieved])
                rag_prompt = f"Answer concisely using only this context:\n{context}\n\nQuestion: {eval_fact['query']}\nAnswer:"

                llm_answer = llm_generate(rag_prompt)
                n_llm_calls += 1

                ground_truth = eval_fact["versions"][current_version]["answer"]
                score = score_answer(llm_answer, ground_truth, eval_fact["query"])
                llm_scores.append(score)

                top_version = retrieved[0].version
                det_scores.append(1.0 if top_version == current_version else 0.0)

    return {
        "llm_quality_mean": float(np.mean(llm_scores)) if llm_scores else 0,
        "llm_quality_std": float(np.std(llm_scores)) if llm_scores else 0,
        "det_quality_mean": float(np.mean(det_scores)) if det_scores else 0,
        "n_evaluations": len(llm_scores),
        "n_llm_calls": n_llm_calls,
    }


def main():
    print("=" * 70, flush=True)
    print("EXPERIMENT 28: ACTUAL LLM-IN-THE-LOOP EVALUATION", flush=True)
    print(f"Generation: {LLM_MODEL} | Embeddings: {EMBEDDING_MODEL}", flush=True)
    print("=" * 70, flush=True)

    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        models = [m["name"] for m in resp.json().get("models", [])]
        has_emb = any(EMBEDDING_MODEL in m for m in models)
        has_llm = any(LLM_MODEL in m for m in models)
        if not has_emb or not has_llm:
            print(f"  ERROR: Missing models", flush=True)
            return
        print(f"  Ollama connected. Models verified.", flush=True)
    except Exception as e:
        print(f"  ERROR: Cannot connect to Ollama: {e}", flush=True)
        return

    data_dir = Path(__file__).parent.parent / "data" / "real_rag"
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    react_path = data_dir / "react_facts_v2.json"
    if not react_path.exists():
        react_path = data_dir / "react_facts.json"
    facts = json.load(open(react_path))
    versions = facts["metadata"]["versions"]
    print(f"  Dataset: {len(facts['facts'])} facts, versions: {versions}", flush=True)

    print(f"  Pre-caching embeddings...", flush=True)
    t0 = time.time()
    for fact in facts["facts"]:
        get_embedding(fact["query"])
        for v in versions:
            get_embedding(fact["versions"][v]["document"])
    print(f"  Cached {len(_embedding_cache)} embeddings in {time.time()-t0:.1f}s", flush=True)

    methods = ["vdd", "recency", "static", "time_weighted", "no_decay"]
    n_runs = 5
    n_eval_facts = 10

    all_results = {m: {"llm_quality": [], "det_quality": []} for m in methods}

    print(f"\n  Running {n_runs} trials per method...", flush=True)
    print(f"  {n_eval_facts} facts × 3 eval points × {len(methods)} methods = "
          f"{n_eval_facts * 3 * len(methods)} LLM calls/seed", flush=True)

    total_start = time.time()
    total_llm_calls = 0
    for seed in range(100, 100 + n_runs):
        seed_start = time.time()
        for method in methods:
            trial = run_llm_trial(facts, versions, method, timesteps=300,
                                  seed=seed, n_eval_facts=n_eval_facts)
            all_results[method]["llm_quality"].append(trial["llm_quality_mean"])
            all_results[method]["det_quality"].append(trial["det_quality_mean"])
            total_llm_calls += trial["n_llm_calls"]
        elapsed = time.time() - seed_start
        print(f"    Seed {seed}: {elapsed:.1f}s ({total_llm_calls} LLM calls so far)", flush=True)

    total_time = time.time() - total_start
    print(f"  Total: {total_time:.0f}s ({total_time/60:.1f}min), {total_llm_calls} LLM calls", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f"ACTUAL LLM EVALUATION RESULTS (n={n_runs})", flush=True)
    print(f"{'='*70}", flush=True)

    vdd_llm = all_results["vdd"]["llm_quality"]

    print(f"\n{'Method':<16} {'LLM Quality':<18} {'Det Quality':<18} {'d(LLM)':<10}", flush=True)
    print("-" * 62, flush=True)

    output = {}
    for method in methods:
        llm_q = all_results[method]["llm_quality"]
        det_q = all_results[method]["det_quality"]
        d = cohens_d(llm_q, vdd_llm) if method != "vdd" else 0

        print(f"{method:<16} {np.mean(llm_q):.3f}±{np.std(llm_q):.3f}     "
              f"{np.mean(det_q):.3f}±{np.std(det_q):.3f}     "
              f"{'---' if method == 'vdd' else f'd={d:+.2f}'}", flush=True)

        output[method] = {
            "llm_quality_mean": round(float(np.mean(llm_q)), 4),
            "llm_quality_std": round(float(np.std(llm_q)), 4),
            "det_quality_mean": round(float(np.mean(det_q)), 4),
            "det_quality_std": round(float(np.std(det_q)), 4),
            "llm_quality_ci": [round(x, 4) for x in bootstrap_ci(np.array(llm_q))],
        }

    rankings_llm = sorted(methods, key=lambda m: np.mean(all_results[m]["llm_quality"]), reverse=True)
    rankings_det = sorted(methods, key=lambda m: np.mean(all_results[m]["det_quality"]), reverse=True)

    print(f"\n  LLM ranking: {' > '.join(rankings_llm)}", flush=True)
    print(f"  Det ranking: {' > '.join(rankings_det)}", flush=True)

    correlation = np.corrcoef(
        [np.mean(all_results[m]["llm_quality"]) for m in methods],
        [np.mean(all_results[m]["det_quality"]) for m in methods]
    )[0, 1]
    print(f"  LLM-Deterministic correlation: r={correlation:.3f}", flush=True)

    output["_metadata"] = {
        "n_runs": n_runs,
        "n_eval_facts": n_eval_facts,
        "versions": versions,
        "llm_model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "total_time_seconds": round(total_time, 1),
        "total_llm_calls": total_llm_calls,
        "seeds": f"100-{99+n_runs}",
        "correlation": round(correlation, 4),
        "scoring": "keyword_overlap",
    }

    with open(results_dir / "28_llm_actual_evaluation.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to results/28_llm_actual_evaluation.json", flush=True)

    plot_results(output, methods, results_dir)
    return output


def plot_results(output, methods, results_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {"vdd": "#2ecc71", "recency": "#e74c3c", "static": "#3498db",
              "time_weighted": "#f39c12", "no_decay": "#95a5a6"}

    llm_means = [output[m]["llm_quality_mean"] for m in methods]
    llm_stds = [output[m]["llm_quality_std"] for m in methods]
    det_means = [output[m]["det_quality_mean"] for m in methods]

    bars = axes[0].bar(methods, llm_means, yerr=llm_stds, capsize=4,
                       color=[colors.get(m, '#95a5a6') for m in methods], alpha=0.8)
    axes[0].set_ylabel("Answer Quality (LLM-generated, keyword-scored)")
    axes[0].set_title(f"LLM Answer Quality ({LLM_MODEL})")
    axes[0].set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    for bar, val in zip(bars, llm_means):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', fontsize=8)

    x = np.arange(len(methods))
    width = 0.35
    axes[1].bar(x - width/2, llm_means, width, label='LLM-generated', color='#2ecc71', alpha=0.8)
    axes[1].bar(x + width/2, det_means, width, label='Deterministic', color='#3498db', alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    axes[1].set_ylabel("Quality Score")
    axes[1].set_title("LLM vs Deterministic Scoring")
    axes[1].legend()

    plt.suptitle("Experiment 28: Actual LLM-in-the-Loop Evaluation",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "28_llm_actual_evaluation.png", dpi=150, bbox_inches='tight')
    print(f"Saved plot to results/28_llm_actual_evaluation.png", flush=True)
    plt.close()


if __name__ == "__main__":
    main()
