#!/usr/bin/env python3
"""
Experiment 27: Real Embeddings via nomic-embed-text

Replaces hash-based embeddings with real embeddings from
nomic-embed-text (via Ollama) on the React dataset.

Proves that VDD results transfer from hash-based to real embeddings.
Compares accuracy/staleness between hash-based and real embeddings
across all 6 methods.
"""

import json
import hashlib
import numpy as np
import requests
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from vdd.drift_detection import EmbeddingDistance
from utils.statistics import bootstrap_ci, cohens_d

OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
HASH_DIM = 64

_embedding_cache = {}


def get_real_embedding(text: str) -> np.ndarray:
    if text in _embedding_cache:
        return _embedding_cache[text]
    resp = requests.post(f"{OLLAMA_URL}/api/embeddings",
                         json={"model": EMBEDDING_MODEL, "prompt": text})
    resp.raise_for_status()
    emb = np.array(resp.json()["embedding"], dtype=np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-10)
    _embedding_cache[text] = emb
    return emb


def hash_embedding(text: str, dim: int = HASH_DIM, noise_seed: int = 0) -> np.ndarray:
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


def run_trial(facts: Dict, versions: List[str], method: str,
              use_real: bool = False, timesteps: int = 300, seed: int = 42) -> Dict:
    np.random.seed(seed)
    n_eras = len(versions)
    era_length = timesteps // n_eras
    fact_list = facts["facts"]

    embed_fn = get_real_embedding if use_real else lambda t: hash_embedding(t, noise_seed=seed)
    emb_dim = len(embed_fn(fact_list[0]["query"]))

    memory = RAGMemory()
    detector = EmbeddingDistance(curr_window=10, arch_window=50, drift_threshold=0.3)

    lambda_base, lambda_max = 0.15, 0.85
    correct_retrievals = []
    staleness_events = []

    for fact in fact_list:
        v = versions[0]
        doc_text = fact["versions"][v]["document"]
        doc = Document(
            id=f"{fact['id']}_{v}", fact_id=fact["id"], version=v,
            text=doc_text, answer=fact["versions"][v]["answer"],
            embedding=embed_fn(doc_text), weight=2.0, added_at=0
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
                        doc_text = fact["versions"][v]["document"]
                        doc = Document(
                            id=f"{fact['id']}_{v}", fact_id=fact["id"], version=v,
                            text=doc_text, answer=fact["versions"][v]["answer"],
                            embedding=embed_fn(doc_text), weight=0.5, added_at=t
                        )
                        memory.add(doc)
                        added[v].add(fact["id"])

        fact = fact_list[np.random.randint(len(fact_list))]
        query_emb = embed_fn(fact["query"])
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
            top = retrieved[0]
            correct_retrievals.append(top.version == current_version)
            staleness_events.append(
                versions.index(top.version) < versions.index(current_version)
                if top.version in versions else False
            )
        else:
            correct_retrievals.append(False)
            staleness_events.append(False)

    return {
        "accuracy": float(np.mean(correct_retrievals)),
        "staleness": float(np.mean(staleness_events)),
    }


def main():
    print("=" * 70)
    print("EXPERIMENT 27: REAL EMBEDDINGS (nomic-embed-text)")
    print("Comparing hash-based vs real embeddings on React dataset")
    print("=" * 70)

    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags")
        models = [m["name"] for m in resp.json().get("models", [])]
        if not any(EMBEDDING_MODEL in m for m in models):
            print(f"  ERROR: {EMBEDDING_MODEL} not found in Ollama. Available: {models}")
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
    facts = json.load(open(react_path))
    versions = facts["metadata"]["versions"]
    print(f"  Dataset: {len(facts['facts'])} React facts, versions: {versions}")

    print(f"\n  Pre-caching embeddings...")
    t0 = time.time()
    for fact in facts["facts"]:
        get_real_embedding(fact["query"])
        for v in versions:
            get_real_embedding(fact["versions"][v]["document"])
    print(f"  Cached {len(_embedding_cache)} embeddings in {time.time()-t0:.1f}s")
    real_dim = len(next(iter(_embedding_cache.values())))
    print(f"  Real embedding dim: {real_dim}, Hash dim: {HASH_DIM}")

    methods = ["vdd", "recency", "static", "time_weighted", "sliding_window", "no_decay"]
    n_runs = 30

    hash_results = {m: {"accuracy": [], "staleness": []} for m in methods}
    real_results = {m: {"accuracy": [], "staleness": []} for m in methods}

    print(f"\n  Running {n_runs} trials (hash-based)...")
    for seed in range(100, 100 + n_runs):
        for method in methods:
            trial = run_trial(facts, versions, method, use_real=False, seed=seed)
            hash_results[method]["accuracy"].append(trial["accuracy"])
            hash_results[method]["staleness"].append(trial["staleness"])
        if (seed - 99) % 10 == 0:
            print(f"    Hash: {seed - 99}/{n_runs}")

    print(f"\n  Running {n_runs} trials (real embeddings)...")
    for seed in range(100, 100 + n_runs):
        for method in methods:
            trial = run_trial(facts, versions, method, use_real=True, seed=seed)
            real_results[method]["accuracy"].append(trial["accuracy"])
            real_results[method]["staleness"].append(trial["staleness"])
        if (seed - 99) % 10 == 0:
            print(f"    Real: {seed - 99}/{n_runs}")

    print(f"\n{'='*70}")
    print("RESULTS COMPARISON: Hash-Based vs Real Embeddings (n=30)")
    print(f"{'='*70}")

    print(f"\n{'Method':<16} {'Hash Acc':<14} {'Real Acc':<14} {'Hash Stale':<14} {'Real Stale':<14} {'d(acc)':<8}")
    print("-" * 80)

    output = {"hash": {}, "real": {}, "comparison": {}}

    for method in methods:
        h_acc = np.mean(hash_results[method]["accuracy"])
        r_acc = np.mean(real_results[method]["accuracy"])
        h_stale = np.mean(hash_results[method]["staleness"])
        r_stale = np.mean(real_results[method]["staleness"])
        d_acc = cohens_d(real_results[method]["accuracy"], hash_results[method]["accuracy"])

        print(f"{method:<16} {h_acc:.3f}±{np.std(hash_results[method]['accuracy']):.3f}  "
              f"{r_acc:.3f}±{np.std(real_results[method]['accuracy']):.3f}  "
              f"{h_stale:.3f}±{np.std(hash_results[method]['staleness']):.3f}  "
              f"{r_stale:.3f}±{np.std(real_results[method]['staleness']):.3f}  "
              f"d={d_acc:+.2f}")

        output["hash"][method] = {
            "accuracy_mean": round(h_acc, 4),
            "accuracy_std": round(float(np.std(hash_results[method]["accuracy"])), 4),
            "staleness_mean": round(h_stale, 4),
            "staleness_std": round(float(np.std(hash_results[method]["staleness"])), 4),
        }
        output["real"][method] = {
            "accuracy_mean": round(r_acc, 4),
            "accuracy_std": round(float(np.std(real_results[method]["accuracy"])), 4),
            "staleness_mean": round(r_stale, 4),
            "staleness_std": round(float(np.std(real_results[method]["staleness"])), 4),
        }
        output["comparison"][method] = {
            "accuracy_d": round(d_acc, 4),
            "accuracy_diff": round(r_acc - h_acc, 4),
        }

    print(f"\n{'='*70}")
    print("TRANSFERABILITY ANALYSIS")
    print(f"{'='*70}")

    vdd_h = np.mean(hash_results["vdd"]["accuracy"])
    vdd_r = np.mean(real_results["vdd"]["accuracy"])
    d = cohens_d(real_results["vdd"]["accuracy"], hash_results["vdd"]["accuracy"])
    print(f"  VDD accuracy: hash={vdd_h:.3f} vs real={vdd_r:.3f} (d={d:+.2f})")
    if abs(d) < 0.5:
        print(f"  -> Effect size is {'negligible' if abs(d) < 0.2 else 'small'}: results TRANSFER")
    else:
        print(f"  -> Effect size is {'medium' if abs(d) < 0.8 else 'large'}: results MAY NOT transfer cleanly")

    hash_ranks = sorted(methods, key=lambda m: np.mean(hash_results[m]["accuracy"]), reverse=True)
    real_ranks = sorted(methods, key=lambda m: np.mean(real_results[m]["accuracy"]), reverse=True)
    print(f"  Hash ranking: {' > '.join(hash_ranks[:3])}")
    print(f"  Real ranking: {' > '.join(real_ranks[:3])}")

    rank_preserved = hash_ranks[:3] == real_ranks[:3]
    print(f"  Top-3 ranking preserved: {'YES' if rank_preserved else 'Partial'}")

    output["_metadata"] = {
        "n_runs": n_runs,
        "n_facts": len(facts["facts"]),
        "versions": versions,
        "embedding_model": EMBEDDING_MODEL,
        "real_dim": real_dim,
        "hash_dim": HASH_DIM,
        "seeds": "100-129",
        "transferability": "confirmed" if abs(d) < 0.5 else "partial",
    }

    with open(results_dir / "27_real_embeddings.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to results/27_real_embeddings.json")

    plot_results(output, methods, results_dir)

    return output


def plot_results(output, methods, results_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(methods))
    width = 0.35

    h_acc = [output["hash"][m]["accuracy_mean"] for m in methods]
    r_acc = [output["real"][m]["accuracy_mean"] for m in methods]
    h_stale = [output["hash"][m]["staleness_mean"] for m in methods]
    r_stale = [output["real"][m]["staleness_mean"] for m in methods]

    axes[0].bar(x - width/2, h_acc, width, label='Hash-based', color='#3498db', alpha=0.8)
    axes[0].bar(x + width/2, r_acc, width, label='nomic-embed-text', color='#2ecc71', alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy: Hash vs Real Embeddings")
    axes[0].legend()

    axes[1].bar(x - width/2, h_stale, width, label='Hash-based', color='#3498db', alpha=0.8)
    axes[1].bar(x + width/2, r_stale, width, label='nomic-embed-text', color='#2ecc71', alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    axes[1].set_ylabel("Staleness")
    axes[1].set_title("Staleness: Hash vs Real Embeddings")
    axes[1].legend()

    plt.suptitle("Experiment 27: Hash-Based vs Real Embeddings Transferability",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "27_real_embeddings.png", dpi=150, bbox_inches='tight')
    print(f"Saved plot to results/27_real_embeddings.png")
    plt.close()


if __name__ == "__main__":
    main()
