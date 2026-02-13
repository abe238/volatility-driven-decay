# Volatility-Driven Decay (VDD)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Experiments: 42](https://img.shields.io/badge/experiments-42-green.svg)](#experiments)
[![Facts: 120](https://img.shields.io/badge/real--world%20facts-120-orange.svg)](#datasets)
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97-Interactive%20Demo-yellow.svg)](https://huggingface.co/spaces/abe238/vdd-demo)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18629018.svg)](https://doi.org/10.5281/zenodo.18629018)

> **Your RAG system is confidently serving wrong answers 67% of the time.** When React deprecates an API, Python removes a module, or a medical guideline changes, retrieval keeps surfacing the old version. The user gets a well-formatted, authoritative, *stale* response.

VDD makes memory decay a function of environmental change instead of a fixed timer. When knowledge is stable, forget slowly. When semantic drift is detected, forget fast.

**The honest pitch:** VDD is not the best method in any single scenario. Recency wins in high-drift. Time-weighted wins in gradual transitions. But VDD is **never the worst**---making it the safest default when you don't know what kind of drift you'll face. That's the real-world condition.

---

## The Core Idea

Every RAG system with a knowledge base faces a question: *how aggressively should old memories be discounted?*

- Too slow (static low decay) &rarr; stale answers accumulate (67% staleness rate)
- Too fast (aggressive recency) &rarr; valid knowledge is destroyed during reversions
- Any fixed rate &rarr; wrong for at least some drift pattern you'll encounter

VDD makes the decay rate adaptive:

```
lambda(t) = lambda_base + (lambda_max - lambda_base) * sigmoid(k * (V_t - V_0))
```

`V_t` is the detected semantic volatility (embedding centroid distance). When embeddings shift, lambda increases and old memories fade. When embeddings are stable, lambda stays low and knowledge is preserved.

| Parameter | Default | What it does |
|-----------|---------|-------------|
| `lambda_base` | 0.05 | Resting decay (stable periods) |
| `lambda_max` | 0.9 | Maximum decay (during drift) |
| `k` | 10 | Sigmoid steepness (at k>=5, activation choice is immaterial) |
| `V_0` | 0.1 | Volatility threshold (**critical** to tune, 254% performance range) |

---

## Results: What We Actually Found

42 experiments. 120 real-world facts. 3 domains. 13 methods compared. Here's the truth.

### Where VDD wins

| Claim | Evidence | Experiment |
|-------|----------|------------|
| 87.5% error reduction vs static baselines | IAE 15.81 vs 126.67 | Exp 2 |
| Better staleness handling than recency | d = +1.85 (large effect) | Exp 20 |
| Better reversion handling than recency | d = +1.22 | Exp 18 |
| Never the worst across all 4 drift patterns | Rank #2/13, 100% rank stability | Exp 8, 22, 31 |
| Negligible latency overhead | 2.75ms mean, 6.48ms P99 | Exp 7 |
| Beats LRU-based decay | d = +0.921 | Exp 37 |
| FreshQA: 100% current-answer retrieval | vs no_decay 70-80% | Exp 39 |
| Auto-calibration eliminates V_0 tuning | 22.7% improvement over hand-tuned | Exp 40 |
| Rankings hold across embedding models | Spearman rho = 0.978 | Exp 41 |

### Where VDD loses (and we're explicit about it)

| Scenario | Winner | How badly VDD loses |
|----------|--------|-------------------|
| Constant high drift | Recency (lambda=0.5) | d = -2.90 (very large) |
| Gradual transitions | Time-weighted (alpha=0.01) | 0.871 vs 0.584 accuracy |
| 3 of 4 synthetic scenarios | Holt-Winters, EMA-lambda | VDD ranks #5, they rank #2-3 |
| Real embeddings, absolute accuracy | Time-weighted, static | 0.875 and 0.736 vs VDD 0.628 |
| **StreamingQA (36K questions)** | **No decay** | **d = -7.68** (accumulation, not replacement) |

**The key insight**: VDD doesn't win individual races. It places consistently, which means it's the method you want when you can't predict the race.

### Real-World Multi-Domain Results (120 Facts)

Tested on React (v16-18), Python (v3.8-3.12), Node.js (v16-20) documentation:

| Method | Accuracy | Staleness | Verdict |
|--------|----------|-----------|---------|
| Time-weighted | **0.875** | 0.000 | Best for gradual transitions |
| Static (lambda=0.1) | 0.736 | 0.135 | Good if stable |
| **VDD** | **0.628** | **0.243** | Best staleness/robustness tradeoff |
| Recency (lambda=0.5) | 0.570 | 0.300 | Wins on constant churn |
| Sliding window | 0.421 | 0.398 | Catastrophic failure on 2/3 domains |
| No decay | 0.333 | 0.667 | Random chance |

### Statistical Rigor

- **85/88** significant results survive Benjamini-Hochberg FDR correction (0 lost)
- **Type M analysis**: at d=0.5 true effect, our experiments inflate to d=0.715 (1.43x). Production effects will be smaller than reported.
- **Hash-to-real embedding transfer**: rankings preserved in 4/5 scenarios (r=0.935), but effect sizes attenuate ~30%
- **LLM-in-the-loop validation**: n=50 seeds, 7,500 LLM calls via llama3.1:8b + nomic-embed-text

---

## When to Use VDD (Decision Guide)

### Use VDD when:
- You **don't know** what drift pattern you'll face (the common case)
- Knowledge **reversions** are possible (rollbacks, seasonal changes, corrections)
- **Avoiding stale answers** matters more than peak accuracy
- You're building a **coding assistant** (gradual deprecations + sudden breaking changes)
- You're building a **medical/legal knowledge base** (long stability + sudden guideline changes)

### Don't use VDD when:
- **Constant high churn** (news, social media) &rarr; use recency
- **Gradual, predictable evolution** &rarr; use time-weighted decay
- **Known stable domain** &rarr; use static decay with low lambda
- **You have labeled data** for drift &rarr; online_lambda matches VDD with labels (d = -0.056)

### Never use:
- **Sliding window (N>50)** on multi-topic domains: catastrophic 0.333 accuracy on 2/3 test domains

See the [practitioner decision tree](results/42_decision_tree.png) for a visual guide to method selection based on your use case.

---

## Quick Start

```bash
git clone https://github.com/abe238/volatility-driven-decay.git
cd volatility-driven-decay
python -m venv venv && source venv/bin/activate
pip install -e ".[all]"
```

```python
from vdd.drift_detection import EmbeddingDistanceDetector
from vdd.memory import VDDMemoryBank
from vdd.retrieval import VDDRetriever

detector = EmbeddingDistanceDetector(current_window=10, archive_window=200)
memory = VDDMemoryBank(lambda_base=0.05, lambda_max=0.9, k=10.0, v0=0.1)
retriever = VDDRetriever(memory, detector)

retriever.add("React 18 uses Suspense for data fetching", embedding)
results = retriever.retrieve(query_embedding, k=5)
```

### Framework Integration

**LangChain**: Custom `VDDRetriever` wrapping any vectorstore. Drift detection runs on the embedding history within the retriever's `_get_relevant_documents` method.

**LlamaIndex**: Custom `NodePostprocessor` that applies VDD's temporal weighting after initial retrieval.

**Haystack**: Custom `Ranker` component operating between retriever and reader stages.

---

## Experiments

42 experiments organized by what they test:

| Group | Experiments | What they validate | External deps |
|-------|-----------|-------------------|---------------|
| Core validation | 1-8 | Drift detection, VDD mechanism, baselines | None |
| Extended validation | 9-15 | Precision, scaling, statistical hardening | None |
| Real-world | 16-20 | 120 facts, 3 domains, staleness | Ollama (Exp 16) |
| Advanced analysis | 21-26, 29 | Effective lambda, sigmoid sensitivity | None |
| Ollama-required | 23, 27-28, 33 | Real embeddings, LLM evaluation, 3-domain | Ollama |
| Confirmatory | 30-32 | n=30 reruns, adaptive baselines, bimodality | None |
| Revision | 34-41 | FDR correction, n=50 LLM, real embedding suite, activation ablation, StreamingQA, FreshQA, auto-calibration, cross-model embedding | Ollama (35-36, 38-39, 41) |
| Practitioner | 42 | Decision tree visualization for method selection | None |

```bash
# Run all CPU experiments (~15 min)
python run_experiments.py --all

# Run a specific experiment
python experiments/02_scalar_simulation_fixed.py

# Experiments requiring Ollama
ollama pull nomic-embed-text && ollama pull llama3.1:8b
python experiments/33_three_domain.py
```

All results (plots + JSON) are saved to `results/`.

---

## The Research Process (Radical Transparency)

This paper started with a fatal flaw and was rebuilt from scratch. We're sharing the full journey because science should show its work.

**V1**: Built a VDD prototype with oracle drift detection. Looked amazing. The oracle *was* the result---we were measuring our own assumptions. Fatal flaw discovered, entire approach scrapped.

**V2-V3**: Replaced oracle with real detection. Discovered recency (the simplest baseline) beats VDD on IAE. This was demoralizing but honest. We kept going.

**V4**: Reframed from "VDD is best" to "VDD is never worst." Added staleness analysis. The weaker claim turned out to be the more useful one: practitioners need reliable defaults, not optimistic benchmarks.

**V5**: Proved VDD's lambda distribution is genuinely bimodal (Hartigan's Dip p < 0.001), not just noisy static decay. Expanded to 90 facts, n=30 seeds. Added bootstrap CIs.

**V6**: Three-domain validation (React + Python + Node.js, 120 facts). Added adaptive baselines (Holt-Winters, EMA-lambda, DWM-lite). Published with 31 experiments.

**V7-V8**: Peer review variations and condensed arxiv formatting.

**V9 (current)**: Major revision addressing 11 peer review issues head-on:
- Added Benjamini-Hochberg FDR correction (85/88 survive)
- Expanded LLM validation from n=5 to n=50 (7,500 LLM calls)
- Validated hash-to-real embedding transfer (4/5 scenarios, r=0.935)
- Added 12 missing citations (HippoRAG, MemoRAG, Mem0, ARM, T-GRAG, etc.)
- Proved activation function choice is immaterial at k>=5
- Added Type M error analysis (1.43x inflation at d=0.5)
- Acknowledged every limitation we could identify (14 total)
- Added theoretical analysis (regret framework formalizing "never worst" property)
- Added practitioner decision tree for method selection (Exp 42)

The paper is stronger because reviewers pushed back. Every objection made the science more honest.

---

## 14 Limitations (Yes, We're Listing All of Them)

1. Recency often wins on raw accuracy (d = -2.90 in high-drift)
2. Adaptive baselines (Holt-Winters, EMA-lambda) beat VDD in 3/4 synthetic scenarios
3. Detector false positive rate is 13.5%, not the ~1% initially reported
4. Time-weighted dominates gradual transitions (0.871 vs 0.584)
5. V_0 and window sizes require tuning (254-325% performance range), though auto-calibration (Exp 40) mitigates V_0 tuning with 22.7% improvement over hand-tuned defaults
6. Effect sizes inflate ~1.43x under controlled conditions; production effects will be smaller
7. O(n) scaling; >10K memories needs approximate nearest neighbor search
8. LLM validation shows VDD/recency have zero variance (deterministic retrieval, not statistical robustness)
9. Only 3 technical documentation domains tested; no news, medical, or legal validation
10. No production deployment study
11. StreamingQA shows decay is counterproductive on accumulation tasks (d = -7.68); need replacement-focused benchmarks
12. Effect sizes attenuate ~30% with real embeddings vs hash-based
13. online_lambda matches VDD (d = -0.056), weakening the unique mechanism claim
14. 120-fact dataset is purpose-built, not a community benchmark

We believe listing what doesn't work is as valuable as listing what does.

---

## Datasets

120 versioned facts with ground-truth answers across 3 API evolution timelines:

| Domain | Facts | Versions | Example drift |
|--------|-------|----------|--------------|
| React | 60 | v16, v17, v18 | Class components &rarr; Hooks &rarr; Suspense |
| Python | 30 | 3.8, 3.10, 3.12 | Assignment expressions, match statements |
| Node.js | 30 | v16, v18, v20 | Fetch API, test runner, permission model |

All data is in `data/real_rag/`. Released under Apache 2.0 for reproducibility.

---

## Repository Structure

```
volatility-driven-decay/
├── paper_v9.md              # Paper (markdown, 1,190+ lines)
├── paper_v9.pdf             # Paper (PDF, 42 pages)
├── paper_v9.tex             # Paper (LaTeX)
├── paper_v9.docx            # Paper (Word)
├── arxiv_submission/        # arXiv-ready package
│   ├── main.tex             # Self-contained LaTeX (1,800+ lines)
│   ├── main.pdf             # Compiled PDF
│   └── figures/             # All 19 figures
├── src/vdd/                 # Core library
│   ├── drift_detection/     # ADWIN, embedding distance, Page-Hinkley
│   ├── memory/              # VDD memory bank, static decay baseline
│   └── retrieval/           # VDD retriever
├── experiments/             # All 42 experiment scripts
├── results/                 # Generated plots and JSON data
├── data/real_rag/           # 120 versioned facts (React, Python, Node.js)
├── tests/                   # Unit tests
└── run_experiments.py       # CLI experiment runner
```

---

## Citation

```bibtex
@article{diaz2026vdd,
  title={Volatility-Driven Decay: Adaptive Memory Retention for
         RAG Systems Under Unknown Drift},
  author={Diaz, Abe},
  journal={arXiv preprint},
  year={2026}
}
```

---

## License

Apache License 2.0. See [LICENSE](LICENSE).

*Opinions are my own. This work does not relate to my position at Amazon.*
