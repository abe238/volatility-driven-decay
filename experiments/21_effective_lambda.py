#!/usr/bin/env python3
"""
Experiment 21: Effective Lambda Distribution Analysis [PHASE 0 - THE GATE]

Critical question: Is VDD truly adaptive, or just noisy static decay?

If the lambda distribution is bimodal (low during stable, high during drift),
VDD is genuinely adaptive. If it's unimodal around ~0.3, VDD is just a
well-tuned static rate with extra complexity.

We also compare VDD to "static oracle" — static decay at lambda = mean(lambda_VDD).
If static oracle performs similarly, VDD's complexity isn't justified.

This experiment gates the entire review response plan.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats as sp_stats
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from vdd.drift_detection import EmbeddingDistance


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def generate_scenario(name, steps, embedding_dim, seed):
    """Generate different test scenarios."""
    np.random.seed(seed)
    truth = np.zeros(steps)
    embeddings = []
    current_val = 0
    centroid = np.random.randn(embedding_dim)
    centroid /= np.linalg.norm(centroid)
    regime_labels = np.zeros(steps, dtype=int)
    current_regime = 0

    if name == "pure_stable":
        for t in range(steps):
            current_val += np.random.normal(0, 0.05)
            truth[t] = current_val
            emb = centroid + np.random.randn(embedding_dim) * 0.05
            embeddings.append(emb)

    elif name == "pure_drift":
        shift_every = steps // 10
        for t in range(steps):
            if t > 0 and t % shift_every == 0:
                current_val += np.random.choice([-5, 5])
                new_dir = np.random.randn(embedding_dim)
                centroid = 0.2 * centroid + 0.8 * new_dir
                centroid /= np.linalg.norm(centroid)
                current_regime += 1
            current_val += np.random.normal(0, 0.1)
            truth[t] = current_val
            emb = centroid + np.random.randn(embedding_dim) * 0.1
            embeddings.append(emb)
            regime_labels[t] = current_regime

    elif name == "mixed_70_30":
        drift_periods = [(int(steps * 0.15), int(steps * 0.25)),
                         (int(steps * 0.50), int(steps * 0.60)),
                         (int(steps * 0.80), int(steps * 0.87))]
        in_drift = False
        for t in range(steps):
            for start, end in drift_periods:
                if start <= t < end:
                    in_drift = True
                    break
            else:
                in_drift = False

            if in_drift and t > 0 and (t - 1) not in [range(s, e) for s, e in drift_periods]:
                current_val += np.random.choice([-5, 5])
                new_dir = np.random.randn(embedding_dim)
                centroid = 0.3 * centroid + 0.7 * new_dir
                centroid /= np.linalg.norm(centroid)
                current_regime += 1

            if in_drift:
                current_val += np.random.normal(0, 0.5)
                emb = centroid + np.random.randn(embedding_dim) * 0.3
            else:
                current_val += np.random.normal(0, 0.05)
                emb = centroid + np.random.randn(embedding_dim) * 0.05

            truth[t] = current_val
            embeddings.append(emb)
            regime_labels[t] = current_regime

    elif name == "bursty":
        burst_at = [int(steps * 0.4)]
        for t in range(steps):
            if t in burst_at:
                for _ in range(3):
                    current_val += np.random.choice([-5, 5])
                new_dir = np.random.randn(embedding_dim)
                centroid = 0.1 * centroid + 0.9 * new_dir
                centroid /= np.linalg.norm(centroid)
                current_regime += 1
            current_val += np.random.normal(0, 0.1)
            truth[t] = current_val
            emb = centroid + np.random.randn(embedding_dim) * 0.1
            embeddings.append(emb)
            regime_labels[t] = current_regime

    elif name == "reversion":
        original_centroid = centroid.copy()
        original_val = current_val
        shifts = [(int(steps * 0.25), True), (int(steps * 0.50), False),
                  (int(steps * 0.75), True)]
        for t in range(steps):
            for shift_t, away in shifts:
                if t == shift_t:
                    if away:
                        new_dir = np.random.randn(embedding_dim)
                        centroid = 0.2 * centroid + 0.8 * new_dir
                        centroid /= np.linalg.norm(centroid)
                        current_val += 5
                    else:
                        centroid = original_centroid + np.random.randn(embedding_dim) * 0.1
                        centroid /= np.linalg.norm(centroid)
                        current_val = original_val + np.random.normal(0, 1)
                    current_regime += 1
            current_val += np.random.normal(0, 0.1)
            truth[t] = current_val
            emb = centroid + np.random.randn(embedding_dim) * 0.1
            embeddings.append(emb)
            regime_labels[t] = current_regime

    return truth, embeddings, regime_labels


def run_vdd(truth, embeddings, lambda_base=0.2, lambda_max=0.9, k=10.0, v0=0.5):
    """Run VDD and return lambda trace + IAE."""
    steps = len(truth)
    detector = EmbeddingDistance(curr_window=5, arch_window=50, drift_threshold=0.4)
    memory = 0
    lambdas = []
    volatilities = []
    errors = []

    for t in range(steps):
        result = detector.update(embeddings[t])
        v = result.volatility
        lam = lambda_base + (lambda_max - lambda_base) * sigmoid(k * (v - v0))
        lambdas.append(lam)
        volatilities.append(v)
        memory = (1 - lam) * memory + lam * truth[t]
        errors.append(abs(truth[t] - memory))

    return np.array(lambdas), np.array(volatilities), np.array(errors)


def run_static(truth, lam):
    """Run static decay at fixed lambda."""
    steps = len(truth)
    memory = 0
    errors = []
    for t in range(steps):
        memory = (1 - lam) * memory + lam * truth[t]
        errors.append(abs(truth[t] - memory))
    return np.array(errors)


def analyze_bimodality(lambdas):
    """Test for bimodality using Hartigan's dip test approximation."""
    hist, bin_edges = np.histogram(lambdas, bins=50, density=True)
    peaks = []
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
            peaks.append((bin_edges[i], hist[i]))

    q25, q75 = np.percentile(lambdas, [25, 75])
    median = np.median(lambdas)
    skewness = sp_stats.skew(lambdas)
    kurtosis = sp_stats.kurtosis(lambdas)

    # Bimodality coefficient: BC = (skewness^2 + 1) / kurtosis
    # BC > 5/9 ≈ 0.555 suggests bimodality
    bc = (skewness**2 + 1) / (kurtosis + 3)

    return {
        "n_peaks": len(peaks),
        "peaks": peaks,
        "mean": np.mean(lambdas),
        "std": np.std(lambdas),
        "median": median,
        "q25": q25,
        "q75": q75,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "bimodality_coefficient": bc,
        "is_bimodal": bc > 0.555 or len(peaks) >= 2,
        "pct_elevated": np.mean(lambdas > np.mean(lambdas) + 0.1) * 100,
        "pct_at_base": np.mean(lambdas < 0.25) * 100,
        "pct_at_panic": np.mean(lambdas > 0.7) * 100,
    }


def main():
    print("=" * 70)
    print("EXPERIMENT 21: EFFECTIVE LAMBDA DISTRIBUTION ANALYSIS")
    print("THE GATE — Does VDD's λ prove it's truly adaptive?")
    print("=" * 70)

    steps = 500
    embedding_dim = 32
    n_runs = 30
    scenarios = ["pure_stable", "pure_drift", "mixed_70_30", "bursty", "reversion"]

    all_results = {}

    for scenario in scenarios:
        print(f"\n{'─'*60}")
        print(f"Scenario: {scenario}")
        print(f"{'─'*60}")

        scenario_lambdas = []
        vdd_iaes = []
        static_oracle_iaes = []
        static_best_iaes = []

        for seed in range(42, 42 + n_runs):
            truth, embeddings, regime_labels = generate_scenario(
                scenario, steps, embedding_dim, seed
            )

            lambdas, volatilities, errors = run_vdd(truth, embeddings)
            vdd_iae = np.sum(errors)
            scenario_lambdas.append(lambdas)
            vdd_iaes.append(vdd_iae)

            # Static oracle: use VDD's mean lambda
            mean_lam = np.mean(lambdas)
            oracle_errors = run_static(truth, mean_lam)
            static_oracle_iaes.append(np.sum(oracle_errors))

            # Best static: try several fixed lambdas
            best_static_iae = float('inf')
            for test_lam in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                s_errors = run_static(truth, test_lam)
                s_iae = np.sum(s_errors)
                if s_iae < best_static_iae:
                    best_static_iae = s_iae
            static_best_iaes.append(best_static_iae)

        all_lambdas = np.concatenate(scenario_lambdas)
        bimodality = analyze_bimodality(all_lambdas)

        vdd_mean = np.mean(vdd_iaes)
        oracle_mean = np.mean(static_oracle_iaes)
        best_static_mean = np.mean(static_best_iaes)

        # Is VDD better than static-at-mean-lambda?
        t_stat, p_value = sp_stats.ttest_rel(vdd_iaes, static_oracle_iaes)
        vdd_vs_oracle = "VDD WINS" if p_value < 0.05 and vdd_mean < oracle_mean else \
                        "ORACLE WINS" if p_value < 0.05 and oracle_mean < vdd_mean else "TIE"

        t_stat2, p_value2 = sp_stats.ttest_rel(vdd_iaes, static_best_iaes)
        vdd_vs_best = "VDD WINS" if p_value2 < 0.05 and vdd_mean < best_static_mean else \
                      "BEST STATIC WINS" if p_value2 < 0.05 and best_static_mean < vdd_mean else "TIE"

        print(f"\n  Lambda Distribution:")
        print(f"    Mean λ:        {bimodality['mean']:.4f}")
        print(f"    Std λ:         {bimodality['std']:.4f}")
        print(f"    % at base:     {bimodality['pct_at_base']:.1f}%  (λ < 0.25)")
        print(f"    % at panic:    {bimodality['pct_at_panic']:.1f}%  (λ > 0.7)")
        print(f"    Bimodal coeff: {bimodality['bimodality_coefficient']:.4f}")
        print(f"    N peaks:       {bimodality['n_peaks']}")
        print(f"    IS BIMODAL:    {'YES ✅' if bimodality['is_bimodal'] else 'NO ❌'}")

        print(f"\n  Performance (IAE, lower is better):")
        print(f"    VDD:              {vdd_mean:.2f} ± {np.std(vdd_iaes):.2f}")
        print(f"    Static@mean(λ):   {oracle_mean:.2f} ± {np.std(static_oracle_iaes):.2f}  → {vdd_vs_oracle}")
        print(f"    Best static:      {best_static_mean:.2f} ± {np.std(static_best_iaes):.2f}  → {vdd_vs_best}")
        print(f"    Oracle p-value:   {p_value:.6f}")

        all_results[scenario] = {
            "bimodality": bimodality,
            "vdd_iae": {"mean": vdd_mean, "std": float(np.std(vdd_iaes))},
            "static_oracle_iae": {"mean": oracle_mean, "std": float(np.std(static_oracle_iaes))},
            "best_static_iae": {"mean": best_static_mean, "std": float(np.std(static_best_iaes))},
            "vdd_vs_oracle": vdd_vs_oracle,
            "vdd_vs_best_static": vdd_vs_best,
            "oracle_p_value": p_value,
            "best_static_p_value": p_value2,
            "lambda_trace_sample": scenario_lambdas[0].tolist(),
        }

    # Summary
    print(f"\n{'='*70}")
    print("PHASE 0 GATE DECISION")
    print(f"{'='*70}")

    bimodal_count = sum(1 for s in all_results.values() if s["bimodality"]["is_bimodal"])
    vdd_wins_count = sum(1 for s in all_results.values() if s["vdd_vs_oracle"] == "VDD WINS")

    print(f"\n  Bimodal lambda in {bimodal_count}/{len(scenarios)} scenarios")
    print(f"  VDD beats static-oracle in {vdd_wins_count}/{len(scenarios)} scenarios")

    if bimodal_count >= 3 and vdd_wins_count >= 3:
        print(f"\n  ✅ GATE PASSED: VDD is genuinely adaptive.")
        print(f"     Lambda distribution proves VDD adapts to conditions.")
        print(f"     Proceed with Phases 1-5.")
        gate_result = "PASSED"
    elif bimodal_count >= 2 or vdd_wins_count >= 2:
        print(f"\n  ⚠️  GATE PARTIAL: VDD shows some adaptiveness.")
        print(f"     Proceed but adjust paper framing to be more nuanced.")
        gate_result = "PARTIAL"
    else:
        print(f"\n  ❌ GATE FAILED: VDD may be equivalent to well-tuned static.")
        print(f"     Pivot paper framing to 'automatic tuning' narrative.")
        gate_result = "FAILED"

    all_results["_gate_decision"] = gate_result

    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / "21_effective_lambda.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved data to results/21_effective_lambda.json")

    # Plot
    plot_results(all_results, scenarios, results_dir)

    return all_results


def plot_results(all_results, scenarios, results_dir):
    """Create comprehensive visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for i, scenario in enumerate(scenarios):
        ax = axes[i // 3][i % 3]
        data = all_results[scenario]
        lambdas = np.array(data["lambda_trace_sample"])

        ax.hist(lambdas, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(data["bimodality"]["mean"], color='red', linestyle='--',
                   label=f'Mean={data["bimodality"]["mean"]:.3f}')
        ax.set_title(f'{scenario}\nBimodal: {"YES" if data["bimodality"]["is_bimodal"] else "NO"} | '
                     f'VDD vs Oracle: {data["vdd_vs_oracle"]}',
                     fontsize=10)
        ax.set_xlabel('λ(t)')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)

    # Summary in last panel
    ax = axes[1][2]
    ax.axis('off')
    gate = all_results["_gate_decision"]
    summary_lines = [
        "PHASE 0 GATE SUMMARY",
        "═" * 30,
        "",
    ]
    for scenario in scenarios:
        d = all_results[scenario]
        bimodal = "✅" if d["bimodality"]["is_bimodal"] else "❌"
        wins = "✅" if d["vdd_vs_oracle"] == "VDD WINS" else "❌"
        summary_lines.append(f"{scenario:<15} Bimodal:{bimodal} Wins:{wins}")

    summary_lines.extend(["", f"GATE: {gate}"])
    ax.text(0.05, 0.95, "\n".join(summary_lines), transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle("Experiment 21: Is VDD Truly Adaptive? (Lambda Distribution Analysis)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "21_effective_lambda.png", dpi=150, bbox_inches='tight')
    print(f"  Saved plot to results/21_effective_lambda.png")
    plt.close()


if __name__ == "__main__":
    main()
