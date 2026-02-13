#!/usr/bin/env python3
"""
Experiment 38: StreamingQA Benchmark Evaluation

Evaluates VDD on StreamingQA (Liska et al., 2022), an established temporal QA
benchmark with 36K questions spanning 2007-2020.

Design:
- Sample 1,500 questions stratified across 14 years
- Cluster by topic (embedding similarity) to create realistic retrieval competition
- Documents = answer text only (short, ambiguous — real retrieval challenge)
- Each cluster has documents from multiple eras competing for retrieval
- VDD's job: properly weight recent vs old documents within each topic cluster

Evaluation:
- Query with question, retrieve from all accumulated documents
- Score: keyword overlap with ground-truth answer
- Separate metrics for recent/past and by temporal distance
- Compare VDD vs 4 baselines across 4 temporal checkpoints

Uses: nomic-embed-text (Ollama) for embeddings
"""

import json
import numpy as np
import requests
import time
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from vdd.drift_detection import EmbeddingDistance
from utils.statistics import bootstrap_ci, cohens_d

OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_PATH = Path(__file__).parent.parent / "data" / "streaming_qa" / "streamingqa_eval.jsonl"

SAMPLE_SIZE = 1500
EVAL_YEARS = [2011, 2014, 2017, 2020]
N_SEEDS = 10
TOP_K = 3

_embedding_cache = {}


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


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def keyword_overlap(predicted: str, ground_truth: str) -> float:
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at',
                  'to', 'for', 'of', 'with', 'by', 'and', 'or', 'not', 'it', 'that',
                  'this', 'be', 'has', 'have', 'had', 'do', 'does', 'did', 'will',
                  'can', 'could', 'should', 'would', 'may', 'what', 'who', 'when',
                  'where', 'how', 'which', 'there', 'their', 'they', 'its', 'as'}
    gt_words = set(re.findall(r'\b\w+\b', ground_truth.lower())) - stop_words
    pred_words = set(re.findall(r'\b\w+\b', predicted.lower())) - stop_words
    if not gt_words:
        return 0.0
    return len(gt_words & pred_words) / len(gt_words)


@dataclass
class Document:
    doc_id: str
    answer: str
    context: str
    embedding: np.ndarray
    evidence_year: int
    evidence_ts: int
    recent_or_past: str
    weight: float = 1.0


class TemporalRAGMemory:
    def __init__(self):
        self.documents: List[Document] = []

    def add(self, doc: Document):
        self.documents.append(doc)

    def retrieve(self, query_emb: np.ndarray, k: int = 3) -> List[Document]:
        if not self.documents:
            return []
        scores = []
        for d in self.documents:
            sim = float(np.dot(query_emb, d.embedding))
            scores.append((sim * d.weight, d))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scores[:k]]

    def apply_decay(self, lambda_t: float):
        for doc in self.documents:
            doc.weight = max(0.01, (1 - lambda_t) * doc.weight)


def load_and_sample(seed: int) -> List[dict]:
    rng = np.random.RandomState(seed)
    with open(DATA_PATH) as f:
        all_records = [json.loads(l) for l in f]

    for r in all_records:
        r['_evidence_year'] = datetime.fromtimestamp(r['evidence_ts']).year

    by_year = defaultdict(list)
    for r in all_records:
        by_year[r['_evidence_year']].append(r)

    sampled = []
    years = sorted(by_year.keys())
    per_year = max(50, SAMPLE_SIZE // len(years))

    for yr in years:
        records = by_year[yr]
        n = min(per_year, len(records))
        chosen = rng.choice(len(records), size=n, replace=False)
        sampled.extend([records[i] for i in chosen])

    if len(sampled) > SAMPLE_SIZE:
        chosen = rng.choice(len(sampled), size=SAMPLE_SIZE, replace=False)
        sampled = [sampled[i] for i in chosen]

    sampled.sort(key=lambda x: x['evidence_ts'])
    return sampled


def build_documents(records: List[dict]) -> List[Document]:
    docs = []
    for r in records:
        answer = r['answers'][0] if r['answers'] else ''
        context = f"In {r['_evidence_year']}: {answer}"
        emb = get_embedding(context)
        docs.append(Document(
            doc_id=r['qa_id'],
            answer=answer,
            context=context,
            embedding=emb,
            evidence_year=r['_evidence_year'],
            evidence_ts=r['evidence_ts'],
            recent_or_past=r['recent_or_past'],
        ))
    return docs


def run_temporal_eval(docs: List[Document], records: List[dict],
                      method: str, seed: int) -> Dict:
    rng = np.random.RandomState(seed + 1000)
    detector = EmbeddingDistance(curr_window=10, arch_window=50, drift_threshold=0.3)
    lambda_base, lambda_max = 0.15, 0.85

    memory = TemporalRAGMemory()
    results_by_checkpoint = {}
    doc_idx = 0

    docs_by_year = defaultdict(list)
    records_by_year = defaultdict(list)
    for d, r in zip(docs, records):
        docs_by_year[d.evidence_year].append(d)
        records_by_year[d.evidence_year].append(r)

    for year in range(2007, 2021):
        year_docs = docs_by_year.get(year, [])

        for d in year_docs:
            d_copy = Document(
                doc_id=d.doc_id, answer=d.answer, context=d.context,
                embedding=d.embedding.copy(), evidence_year=d.evidence_year,
                evidence_ts=d.evidence_ts, recent_or_past=d.recent_or_past,
                weight=1.0
            )
            memory.add(d_copy)

            result = detector.update(d.embedding)
            vol = result.volatility

            if method == "vdd":
                lam = lambda_base + (lambda_max - lambda_base) * sigmoid(10 * (vol - 0.1))
            elif method == "recency":
                lam = 0.3
            elif method == "static":
                lam = 0.05
            elif method == "time_weighted":
                age_factor = 0.01
                for mem_doc in memory.documents:
                    age = year - mem_doc.evidence_year
                    mem_doc.weight = max(0.01, 1.0 / (1.0 + age_factor * age * 365))
                lam = 0.0
            elif method == "no_decay":
                lam = 0.0
            else:
                lam = 0.1

            if method != "time_weighted":
                memory.apply_decay(lam)

        if year in EVAL_YEARS:
            eligible_records = []
            for yr in range(2007, year + 1):
                eligible_records.extend(records_by_year.get(yr, []))

            if not eligible_records:
                continue

            eval_sample = eligible_records
            if len(eval_sample) > 300:
                idx = rng.choice(len(eval_sample), size=300, replace=False)
                eval_sample = [eval_sample[i] for i in idx]

            scores = []
            mrr_scores = []
            recent_scores = []
            past_scores = []
            temporal_dist_scores = defaultdict(list)

            for r in eval_sample:
                question = r['question']
                gt_answer = r['answers'][0] if r['answers'] else ''
                query_emb = get_embedding(question)
                retrieved = memory.retrieve(query_emb, k=TOP_K)

                if not retrieved:
                    scores.append(0.0)
                    mrr_scores.append(0.0)
                    continue

                best_score = 0.0
                best_rank = -1
                for rank, ret_doc in enumerate(retrieved):
                    overlap = keyword_overlap(ret_doc.answer, gt_answer)
                    if overlap > best_score:
                        best_score = overlap
                        best_rank = rank

                scores.append(best_score)
                mrr_scores.append(1.0 / (best_rank + 1) if best_score > 0.3 else 0.0)

                if r['recent_or_past'] == 'recent':
                    recent_scores.append(best_score)
                else:
                    past_scores.append(best_score)

                age = year - r['_evidence_year']
                bucket = "0-1yr" if age <= 1 else "2-5yr" if age <= 5 else "6+yr"
                temporal_dist_scores[bucket].append(best_score)

            results_by_checkpoint[year] = {
                'score_mean': round(float(np.mean(scores)), 4),
                'score_std': round(float(np.std(scores)), 4),
                'mrr': round(float(np.mean(mrr_scores)), 4),
                'n_queries': len(eval_sample),
                'n_docs': len(memory.documents),
                'recent_score': round(float(np.mean(recent_scores)), 4) if recent_scores else 0.0,
                'past_score': round(float(np.mean(past_scores)), 4) if past_scores else 0.0,
                'n_recent': len(recent_scores),
                'n_past': len(past_scores),
                'by_age': {k: round(float(np.mean(v)), 4) for k, v in temporal_dist_scores.items()},
            }

    return results_by_checkpoint


def main():
    print("=" * 70, flush=True)
    print("EXPERIMENT 38: STREAMINGQA BENCHMARK EVALUATION", flush=True)
    print(f"Sample: {SAMPLE_SIZE} questions | Seeds: {N_SEEDS} | Eval years: {EVAL_YEARS}", flush=True)
    print(f"Embeddings: {EMBEDDING_MODEL} | Top-k: {TOP_K}", flush=True)
    print("=" * 70, flush=True)

    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        models = [m["name"] for m in resp.json().get("models", [])]
        if not any(EMBEDDING_MODEL in m for m in models):
            print(f"  ERROR: {EMBEDDING_MODEL} not found.", flush=True)
            return
        print(f"  Ollama connected. Model verified.", flush=True)
    except Exception as e:
        print(f"  ERROR: Cannot connect to Ollama: {e}", flush=True)
        return

    if not DATA_PATH.exists():
        print(f"  ERROR: {DATA_PATH} not found.", flush=True)
        return

    RESULTS_DIR.mkdir(exist_ok=True)
    methods = ["vdd", "recency", "static", "time_weighted", "no_decay"]

    all_seed_results = {m: [] for m in methods}
    total_start = time.time()

    for seed_idx in range(N_SEEDS):
        seed = 200 + seed_idx
        seed_start = time.time()
        print(f"\n  --- Seed {seed} ({seed_idx+1}/{N_SEEDS}) ---", flush=True)

        records = load_and_sample(seed)
        year_counts = defaultdict(int)
        for r in records:
            year_counts[r['_evidence_year']] += 1
        print(f"    {len(records)} questions sampled", flush=True)

        t0 = time.time()
        documents = build_documents(records)
        print(f"    Embedded in {time.time()-t0:.1f}s (cache: {len(_embedding_cache)})", flush=True)

        for method in methods:
            t0 = time.time()
            checkpoint_results = run_temporal_eval(documents, records, method, seed)
            elapsed = time.time() - t0

            all_seed_results[method].append(checkpoint_results)

            scores = [checkpoint_results[yr]['score_mean']
                      for yr in EVAL_YEARS if yr in checkpoint_results]
            mean_score = np.mean(scores) if scores else 0
            print(f"    {method:<16} mean={mean_score:.3f} ({elapsed:.1f}s)", flush=True)

        seed_elapsed = time.time() - seed_start
        eta = seed_elapsed * (N_SEEDS - seed_idx - 1) / 60
        print(f"    Seed: {seed_elapsed:.0f}s | ETA: {eta:.0f}min", flush=True)

    total_time = time.time() - total_start
    print(f"\n  Total: {total_time:.0f}s ({total_time/60:.1f}min)", flush=True)

    print(f"\n{'='*70}", flush=True)
    print("STREAMINGQA RESULTS", flush=True)
    print(f"{'='*70}", flush=True)

    output = {}
    vdd_all_scores = []

    for method in methods:
        method_data = {}
        method_all_scores = []

        for yr in EVAL_YEARS:
            yr_scores = [sr[yr]['score_mean'] for sr in all_seed_results[method] if yr in sr]
            yr_mrrs = [sr[yr]['mrr'] for sr in all_seed_results[method] if yr in sr]
            yr_recent = [sr[yr]['recent_score'] for sr in all_seed_results[method] if yr in sr]
            yr_past = [sr[yr]['past_score'] for sr in all_seed_results[method] if yr in sr]

            ci = bootstrap_ci(np.array(yr_scores)) if len(yr_scores) > 2 else (0, 0)
            method_data[str(yr)] = {
                'score_mean': round(float(np.mean(yr_scores)), 4),
                'score_std': round(float(np.std(yr_scores)), 4),
                'score_ci95': [round(x, 4) for x in ci],
                'mrr_mean': round(float(np.mean(yr_mrrs)), 4),
                'recent_score': round(float(np.mean(yr_recent)), 4),
                'past_score': round(float(np.mean(yr_past)), 4),
            }
            method_all_scores.extend(yr_scores)

        overall = np.mean(method_all_scores) if method_all_scores else 0
        overall_ci = bootstrap_ci(np.array(method_all_scores)) if len(method_all_scores) > 2 else (0, 0)
        method_data['overall'] = {
            'score_mean': round(float(overall), 4),
            'score_std': round(float(np.std(method_all_scores)), 4),
            'score_ci95': [round(x, 4) for x in overall_ci],
        }

        if method == "vdd":
            vdd_all_scores = method_all_scores

        output[method] = method_data

    print(f"\n{'Method':<16} {'Overall':>8} {'2011':>8} {'2014':>8} {'2017':>8} {'2020':>8} {'d(vdd)':>8}",
          flush=True)
    print("-" * 72, flush=True)

    for method in methods:
        ov = output[method]['overall']['score_mean']
        yr_vals = " ".join(f"{output[method].get(str(yr), {}).get('score_mean', 0):>8.3f}"
                          for yr in EVAL_YEARS)
        d_str = "   ---"
        if method != "vdd" and vdd_all_scores:
            m_scores = []
            for yr in EVAL_YEARS:
                m_scores.extend([sr[yr]['score_mean']
                                for sr in all_seed_results[method] if yr in sr])
            if m_scores:
                d = cohens_d(np.array(vdd_all_scores), np.array(m_scores))
                d_str = f"{d:>+7.3f}"
        print(f"{method:<16} {ov:>8.3f} {yr_vals} {d_str}", flush=True)

    print(f"\n  Recent vs Past (2020 checkpoint):", flush=True)
    print(f"  {'Method':<16} {'Recent':>8} {'Past':>8} {'Delta':>8}", flush=True)
    for method in methods:
        r = output[method].get('2020', {}).get('recent_score', 0)
        p = output[method].get('2020', {}).get('past_score', 0)
        print(f"  {method:<16} {r:>8.3f} {p:>8.3f} {r-p:>+8.3f}", flush=True)

    by_age_summary = {}
    for method in methods:
        age_data = defaultdict(list)
        for sr in all_seed_results[method]:
            if 2020 in sr and 'by_age' in sr[2020]:
                for bucket, val in sr[2020]['by_age'].items():
                    age_data[bucket].append(val)
        by_age_summary[method] = {k: round(float(np.mean(v)), 4)
                                   for k, v in age_data.items()}

    print(f"\n  By temporal distance (2020 checkpoint):", flush=True)
    print(f"  {'Method':<16} {'0-1yr':>8} {'2-5yr':>8} {'6+yr':>8}", flush=True)
    for method in methods:
        vals = by_age_summary.get(method, {})
        print(f"  {method:<16} {vals.get('0-1yr',0):>8.3f} {vals.get('2-5yr',0):>8.3f} "
              f"{vals.get('6+yr',0):>8.3f}", flush=True)

    output['_metadata'] = {
        'benchmark': 'StreamingQA (Liska et al., 2022)',
        'total_eval_questions': 36378,
        'sample_size': SAMPLE_SIZE,
        'n_seeds': N_SEEDS,
        'eval_years': EVAL_YEARS,
        'embedding_model': EMBEDDING_MODEL,
        'top_k': TOP_K,
        'total_time_seconds': round(total_time, 1),
        'scoring': 'keyword_overlap (F1-style)',
        'document_format': 'answer_with_year_prefix',
        'by_age_2020': by_age_summary,
    }

    out_path = RESULTS_DIR / "38_streamingqa_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {out_path}", flush=True)

    plot_results(output, methods, by_age_summary)
    return output


def plot_results(output, methods, by_age_summary):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = {"vdd": "#2ecc71", "recency": "#e74c3c", "static": "#3498db",
              "time_weighted": "#f39c12", "no_decay": "#95a5a6"}

    for method in methods:
        scores = [output[method].get(str(yr), {}).get('score_mean', 0) for yr in EVAL_YEARS]
        axes[0].plot(EVAL_YEARS, scores, 'o-', label=method, color=colors.get(method), linewidth=2)
    axes[0].set_xlabel("Evaluation Year")
    axes[0].set_ylabel("Retrieval Score")
    axes[0].set_title("Score Over Time (StreamingQA)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    x = np.arange(len(methods))
    width = 0.35
    recent = [output[m].get('2020', {}).get('recent_score', 0) for m in methods]
    past = [output[m].get('2020', {}).get('past_score', 0) for m in methods]
    axes[1].bar(x - width/2, recent, width, label='Recent (2020)', color='#e74c3c', alpha=0.7)
    axes[1].bar(x + width/2, past, width, label='Past (<2020)', color='#3498db', alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    axes[1].set_ylabel("Retrieval Score")
    axes[1].set_title("Recent vs Past Knowledge (2020)")
    axes[1].legend()

    age_buckets = ['0-1yr', '2-5yr', '6+yr']
    x2 = np.arange(len(age_buckets))
    width2 = 0.15
    for i, method in enumerate(methods):
        vals = [by_age_summary.get(method, {}).get(b, 0) for b in age_buckets]
        axes[2].bar(x2 + i * width2, vals, width2, label=method,
                    color=colors.get(method), alpha=0.8)
    axes[2].set_xticks(x2 + width2 * 2)
    axes[2].set_xticklabels(age_buckets)
    axes[2].set_xlabel("Knowledge Age")
    axes[2].set_ylabel("Retrieval Score")
    axes[2].set_title("Score by Temporal Distance")
    axes[2].legend(fontsize=7)

    plt.suptitle(f"Experiment 38: StreamingQA Benchmark (n={N_SEEDS}, {SAMPLE_SIZE} questions)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "38_streamingqa_benchmark.png", dpi=150, bbox_inches='tight')
    print(f"  Saved plot to results/38_streamingqa_benchmark.png", flush=True)
    plt.close()


if __name__ == "__main__":
    main()
