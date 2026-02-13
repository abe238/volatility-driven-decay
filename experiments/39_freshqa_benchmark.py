#!/usr/bin/env python3
"""
Experiment 39: FreshQA Benchmark Evaluation

Evaluates VDD on FreshQA (Tu et al., 2023), a temporal QA benchmark with 600
questions categorized by how fast their answers change:
  - fast-changing (154): answers change within a year (VDD's sweet spot)
  - slow-changing (222): answers change over years
  - never-changing (224): answers rarely/never change

Design:
- For each question, simulate a knowledge base with multiple temporal versions
  of answers (old + current), using real embeddings via nomic-embed-text
- VDD's job: properly weight current answers over outdated ones
- FreshQA tests knowledge REPLACEMENT, not accumulation (unlike StreamingQA)
- Hypothesis: VDD should outperform static baselines on fast-changing questions

Evaluation:
- Query with question, retrieve from accumulated answer versions
- Score: keyword overlap with ground-truth answer (same as Exp 38)
- Separate metrics by fact_type category
- Compare VDD vs 4 baselines

Uses: nomic-embed-text (Ollama) for embeddings
"""

import csv
import json
import numpy as np
import requests
import time
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from vdd.drift_detection import EmbeddingDistance
from utils.statistics import bootstrap_ci, cohens_d

OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_PATH = Path(__file__).parent.parent / "data" / "freshqa" / "freshqa.csv"

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
    overlap = gt_words & pred_words
    precision = len(overlap) / len(pred_words) if pred_words else 0.0
    recall = len(overlap) / len(gt_words)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@dataclass
class Document:
    doc_id: str
    text: str
    embedding: np.ndarray
    timestamp: int
    is_current: bool
    fact_type: str
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

    def reset_weights(self):
        for doc in self.documents:
            doc.weight = 1.0


def load_freshqa() -> List[dict]:
    with open(DATA_PATH, encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)

    header_idx = next(i for i, r in enumerate(rows) if len(r) > 0 and r[0] == 'id')
    header = rows[header_idx]
    data = []
    for r in rows[header_idx + 1:]:
        if not r[0].strip():
            continue
        entry = dict(zip(header, r))
        answers = []
        for i in range(10):
            key = f'answer_{i}'
            if key in entry and entry[key].strip():
                answers.append(entry[key].strip())
        entry['answers'] = answers
        if answers and entry.get('fact_type', '').strip():
            data.append(entry)
    return data


def generate_outdated_answer(question: str, current_answer: str, rng: np.random.RandomState) -> str:
    prefixes = [
        "Previously, ",
        "As of earlier records, ",
        "In the past, ",
        "Before the latest update, ",
        "According to older sources, ",
    ]
    suffixes = [
        " (this information may be outdated)",
        " (based on earlier data)",
        " (prior version)",
    ]
    prefix = rng.choice(prefixes)
    suffix = rng.choice(suffixes)
    words = current_answer.split()
    if len(words) > 5:
        n_swap = max(1, len(words) // 4)
        swap_positions = rng.choice(len(words), size=min(n_swap, len(words)), replace=False)
        replacement_words = ["different", "unknown", "earlier", "previous", "another",
                             "former", "old", "prior", "past", "alternate"]
        modified = words.copy()
        for pos in swap_positions:
            modified[pos] = rng.choice(replacement_words)
        return prefix + " ".join(modified) + suffix
    return prefix + current_answer + " " + rng.choice(["(outdated)", "(old version)", "(superseded)"])


def build_temporal_corpus(questions: List[dict], rng: np.random.RandomState) -> List[Document]:
    docs = []
    doc_id = 0

    n_versions_map = {
        'fast-changing': (3, 6),
        'slow-changing': (2, 4),
        'never-changing': (1, 2),
    }

    for q in questions:
        fact_type = q['fact_type'].strip()
        current_answer = q['answers'][0]
        lo, hi = n_versions_map.get(fact_type, (2, 4))
        n_old_versions = rng.randint(lo, hi + 1)

        for v in range(n_old_versions):
            outdated = generate_outdated_answer(q['question'], current_answer, rng)
            timestamp = 1000 + v * 100
            emb = get_embedding(outdated)
            docs.append(Document(
                doc_id=f"q{q['id']}_v{v}",
                text=outdated,
                embedding=emb,
                timestamp=timestamp,
                is_current=False,
                fact_type=fact_type,
            ))
            doc_id += 1

        current_ts = 1000 + n_old_versions * 100
        emb = get_embedding(current_answer)
        docs.append(Document(
            doc_id=f"q{q['id']}_current",
            text=current_answer,
            embedding=emb,
            timestamp=current_ts,
            is_current=True,
            fact_type=fact_type,
        ))
        doc_id += 1

    docs.sort(key=lambda d: d.timestamp)
    return docs


def run_freshqa_eval(docs: List[Document], questions: List[dict],
                     method: str, seed: int) -> Dict:
    rng = np.random.RandomState(seed + 2000)
    detector = EmbeddingDistance(curr_window=10, arch_window=50, drift_threshold=0.3)
    lambda_base, lambda_max = 0.15, 0.85

    memory = TemporalRAGMemory()

    for d in docs:
        d_copy = Document(
            doc_id=d.doc_id, text=d.text, embedding=d.embedding.copy(),
            timestamp=d.timestamp, is_current=d.is_current,
            fact_type=d.fact_type, weight=1.0
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
            max_ts = d.timestamp
            for mem_doc in memory.documents:
                age = max_ts - mem_doc.timestamp
                mem_doc.weight = max(0.01, 1.0 / (1.0 + 0.005 * age))
            lam = 0.0
        elif method == "no_decay":
            lam = 0.0
        else:
            lam = 0.1

        if method != "time_weighted":
            memory.apply_decay(lam)

    scores_by_type = defaultdict(list)
    mrr_by_type = defaultdict(list)
    current_retrieved_by_type = defaultdict(list)
    all_scores = []

    for q in questions:
        fact_type = q['fact_type'].strip()
        current_answer = q['answers'][0]
        query_emb = get_embedding(q['question'])
        retrieved = memory.retrieve(query_emb, k=TOP_K)

        if not retrieved:
            scores_by_type[fact_type].append(0.0)
            all_scores.append(0.0)
            current_retrieved_by_type[fact_type].append(0)
            continue

        best_score = 0.0
        best_rank = -1
        any_current = 0
        for rank, ret_doc in enumerate(retrieved):
            overlap = keyword_overlap(ret_doc.text, current_answer)
            if overlap > best_score:
                best_score = overlap
                best_rank = rank
            if ret_doc.is_current:
                any_current = 1

        scores_by_type[fact_type].append(best_score)
        mrr_by_type[fact_type].append(1.0 / (best_rank + 1) if best_score > 0.1 else 0.0)
        current_retrieved_by_type[fact_type].append(any_current)
        all_scores.append(best_score)

    results = {}
    for ft in ['fast-changing', 'slow-changing', 'never-changing']:
        scores = scores_by_type[ft]
        mrrs = mrr_by_type[ft]
        curr_rates = current_retrieved_by_type[ft]
        results[ft] = {
            'score_mean': round(float(np.mean(scores)), 4) if scores else 0.0,
            'score_std': round(float(np.std(scores)), 4) if scores else 0.0,
            'mrr_mean': round(float(np.mean(mrrs)), 4) if mrrs else 0.0,
            'current_retrieval_rate': round(float(np.mean(curr_rates)), 4) if curr_rates else 0.0,
            'n_questions': len(scores),
        }
    results['overall'] = {
        'score_mean': round(float(np.mean(all_scores)), 4),
        'score_std': round(float(np.std(all_scores)), 4),
        'n_questions': len(all_scores),
        'n_docs': len(memory.documents),
    }
    return results


def main():
    print("=" * 70, flush=True)
    print("EXPERIMENT 39: FRESHQA BENCHMARK EVALUATION", flush=True)
    print(f"Seeds: {N_SEEDS} | Top-k: {TOP_K}", flush=True)
    print(f"Embeddings: {EMBEDDING_MODEL}", flush=True)
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

    questions = load_freshqa()
    by_type = defaultdict(int)
    for q in questions:
        by_type[q['fact_type'].strip()] += 1
    print(f"  Loaded {len(questions)} questions: {dict(by_type)}", flush=True)

    questions_no_fp = [q for q in questions if q.get('false_premise', '').upper() != 'TRUE']
    by_type2 = defaultdict(int)
    for q in questions_no_fp:
        by_type2[q['fact_type'].strip()] += 1
    print(f"  After removing false-premise: {len(questions_no_fp)} questions: {dict(by_type2)}", flush=True)

    methods = ["vdd", "recency", "static", "time_weighted", "no_decay"]
    all_seed_results = {m: [] for m in methods}
    total_start = time.time()

    for seed_idx in range(N_SEEDS):
        seed = 300 + seed_idx
        seed_start = time.time()
        rng = np.random.RandomState(seed)
        print(f"\n  --- Seed {seed} ({seed_idx+1}/{N_SEEDS}) ---", flush=True)

        t0 = time.time()
        docs = build_temporal_corpus(questions_no_fp, rng)
        embed_time = time.time() - t0
        print(f"    Built {len(docs)} documents in {embed_time:.1f}s (cache: {len(_embedding_cache)})",
              flush=True)

        for method in methods:
            t0 = time.time()
            results = run_freshqa_eval(docs, questions_no_fp, method, seed)
            elapsed = time.time() - t0

            all_seed_results[method].append(results)

            overall = results['overall']['score_mean']
            fast = results.get('fast-changing', {}).get('score_mean', 0)
            print(f"    {method:<16} overall={overall:.3f} fast={fast:.3f} ({elapsed:.1f}s)",
                  flush=True)

        seed_elapsed = time.time() - seed_start
        eta = seed_elapsed * (N_SEEDS - seed_idx - 1) / 60
        print(f"    Seed: {seed_elapsed:.0f}s | ETA: {eta:.0f}min", flush=True)

    total_time = time.time() - total_start
    print(f"\n  Total: {total_time:.0f}s ({total_time/60:.1f}min)", flush=True)

    print(f"\n{'='*70}", flush=True)
    print("FRESHQA RESULTS", flush=True)
    print(f"{'='*70}", flush=True)

    output = {}
    vdd_overall_scores = []
    vdd_fast_scores = []

    for method in methods:
        method_data = {}
        method_overall = []
        method_fast = []

        for ft in ['fast-changing', 'slow-changing', 'never-changing', 'overall']:
            ft_scores = [sr[ft]['score_mean'] for sr in all_seed_results[method] if ft in sr]
            ft_mrrs = [sr[ft].get('mrr_mean', 0) for sr in all_seed_results[method] if ft in sr]
            ft_curr = [sr[ft].get('current_retrieval_rate', 0) for sr in all_seed_results[method] if ft in sr]

            ci = bootstrap_ci(np.array(ft_scores)) if len(ft_scores) > 2 else (0, 0)
            method_data[ft] = {
                'score_mean': round(float(np.mean(ft_scores)), 4),
                'score_std': round(float(np.std(ft_scores)), 4),
                'score_ci95': [round(x, 4) for x in ci],
            }
            if ft != 'overall':
                method_data[ft]['mrr_mean'] = round(float(np.mean(ft_mrrs)), 4)
                method_data[ft]['current_retrieval_rate'] = round(float(np.mean(ft_curr)), 4)

            if ft == 'overall':
                method_overall = ft_scores
            elif ft == 'fast-changing':
                method_fast = ft_scores

        if method == "vdd":
            vdd_overall_scores = method_overall
            vdd_fast_scores = method_fast

        output[method] = method_data

    fact_types = ['fast-changing', 'slow-changing', 'never-changing', 'overall']
    print(f"\n  {'Method':<16}", end="", flush=True)
    for ft in fact_types:
        label = ft[:12]
        print(f" {label:>12}", end="")
    print(f" {'d(fast)':>8} {'d(over)':>8}")
    print("  " + "-" * 80, flush=True)

    for method in methods:
        print(f"  {method:<16}", end="", flush=True)
        for ft in fact_types:
            val = output[method].get(ft, {}).get('score_mean', 0)
            print(f" {val:>12.4f}", end="")

        d_fast = "   ---"
        d_over = "   ---"
        if method != "vdd":
            m_fast = [sr.get('fast-changing', {}).get('score_mean', 0)
                      for sr in all_seed_results[method]]
            m_over = [sr.get('overall', {}).get('score_mean', 0)
                      for sr in all_seed_results[method]]
            if m_fast and vdd_fast_scores:
                d = cohens_d(np.array(vdd_fast_scores), np.array(m_fast))
                d_fast = f"{d:>+7.3f}"
            if m_over and vdd_overall_scores:
                d = cohens_d(np.array(vdd_overall_scores), np.array(m_over))
                d_over = f"{d:>+7.3f}"
        print(f" {d_fast} {d_over}", flush=True)

    print(f"\n  Current Answer Retrieval Rate:", flush=True)
    print(f"  {'Method':<16} {'Fast':>10} {'Slow':>10} {'Never':>10}", flush=True)
    for method in methods:
        fast_cr = output[method].get('fast-changing', {}).get('current_retrieval_rate', 0)
        slow_cr = output[method].get('slow-changing', {}).get('current_retrieval_rate', 0)
        never_cr = output[method].get('never-changing', {}).get('current_retrieval_rate', 0)
        print(f"  {method:<16} {fast_cr:>10.3f} {slow_cr:>10.3f} {never_cr:>10.3f}", flush=True)

    output['_metadata'] = {
        'benchmark': 'FreshQA (Tu et al., 2023)',
        'total_questions': len(questions),
        'evaluated_questions': len(questions_no_fp),
        'excluded_false_premise': len(questions) - len(questions_no_fp),
        'by_fact_type': dict(by_type2),
        'n_seeds': N_SEEDS,
        'embedding_model': EMBEDDING_MODEL,
        'top_k': TOP_K,
        'total_time_seconds': round(total_time, 1),
        'scoring': 'keyword_overlap (F1-style)',
        'design': 'temporal_corpus_with_outdated_versions',
    }

    out_path = RESULTS_DIR / "39_freshqa_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {out_path}", flush=True)

    plot_results(output, methods)
    return output


def plot_results(output, methods):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = {"vdd": "#2ecc71", "recency": "#e74c3c", "static": "#3498db",
              "time_weighted": "#f39c12", "no_decay": "#95a5a6"}

    fact_types = ['fast-changing', 'slow-changing', 'never-changing']
    x = np.arange(len(fact_types))
    width = 0.15
    for i, method in enumerate(methods):
        vals = [output[method].get(ft, {}).get('score_mean', 0) for ft in fact_types]
        axes[0].bar(x + i * width, vals, width, label=method,
                    color=colors.get(method), alpha=0.8)
    axes[0].set_xticks(x + width * 2)
    axes[0].set_xticklabels(fact_types, fontsize=9)
    axes[0].set_ylabel("Retrieval Score (F1)")
    axes[0].set_title("Score by Question Category")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3, axis='y')

    for i, method in enumerate(methods):
        vals = [output[method].get(ft, {}).get('current_retrieval_rate', 0) for ft in fact_types]
        axes[1].bar(x + i * width, vals, width, label=method,
                    color=colors.get(method), alpha=0.8)
    axes[1].set_xticks(x + width * 2)
    axes[1].set_xticklabels(fact_types, fontsize=9)
    axes[1].set_ylabel("Current Answer Retrieval Rate")
    axes[1].set_title("How Often Current Answer is Retrieved")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3, axis='y')

    overall_scores = [output[m]['overall']['score_mean'] for m in methods]
    overall_cis = [output[m]['overall'].get('score_ci95', [0, 0]) for m in methods]
    errs = [(s - ci[0], ci[1] - s) for s, ci in zip(overall_scores, overall_cis)]
    err_low = [e[0] for e in errs]
    err_high = [e[1] for e in errs]
    bar_colors = [colors.get(m, '#333') for m in methods]
    bars = axes[2].bar(methods, overall_scores, color=bar_colors, alpha=0.8,
                       yerr=[err_low, err_high], capsize=5)
    axes[2].set_ylabel("Overall Score (F1)")
    axes[2].set_title("Overall Performance")
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3, axis='y')

    for bar, score in zip(bars, overall_scores):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f'{score:.3f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle(f"Experiment 39: FreshQA Benchmark (n={N_SEEDS}, {len(load_freshqa())} questions)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "39_freshqa_benchmark.png", dpi=150, bbox_inches='tight')
    print(f"  Saved plot to results/39_freshqa_benchmark.png", flush=True)
    plt.close()


if __name__ == "__main__":
    main()
