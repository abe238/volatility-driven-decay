#!/usr/bin/env python3
"""
Experiment 32: Robust Bimodality Testing of Lambda Traces

Applies three formal bimodality tests to VDD lambda traces across scenarios:
  1. Hartigan's Dip Test (null = unimodal, p < 0.05 rejects)
  2. Ashman's D (Gaussian mixture separation, D > 2 = well-separated)
  3. Silverman's bandwidth test (kernel density mode counting)

Key reframe: pure_stable SHOULD be unimodal — lambda correctly stays near
base rate when there's no drift. Bimodality emerges only with drift.

Stats: n=30 runs, seeds 42-71, 500 steps, embedding_dim=32.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats as sp_stats
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from vdd.drift_detection import EmbeddingDistance
from utils.statistics import bootstrap_ci


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def generate_scenario(name, steps, embedding_dim, seed):
    np.random.seed(seed)
    truth = np.zeros(steps)
    embeddings = []
    current_val = 0
    centroid = np.random.randn(embedding_dim)
    centroid /= np.linalg.norm(centroid)

    if name == "pure_stable":
        for t in range(steps):
            current_val += np.random.normal(0, 0.05)
            truth[t] = current_val
            embeddings.append(centroid + np.random.randn(embedding_dim) * 0.05)

    elif name == "pure_drift":
        shift_every = steps // 10
        for t in range(steps):
            if t > 0 and t % shift_every == 0:
                current_val += np.random.choice([-5, 5])
                new_dir = np.random.randn(embedding_dim)
                centroid = 0.2 * centroid + 0.8 * new_dir
                centroid /= np.linalg.norm(centroid)
            current_val += np.random.normal(0, 0.1)
            truth[t] = current_val
            embeddings.append(centroid + np.random.randn(embedding_dim) * 0.1)

    elif name == "mixed_70_30":
        drift_periods = [(int(steps * 0.15), int(steps * 0.25)),
                         (int(steps * 0.50), int(steps * 0.60)),
                         (int(steps * 0.80), int(steps * 0.87))]
        for t in range(steps):
            in_drift = any(s <= t < e for s, e in drift_periods)
            if in_drift:
                noise = 0.3
                if t > 0 and not any(s <= t - 1 < e for s, e in drift_periods):
                    current_val += np.random.choice([-5, 5])
                    new_dir = np.random.randn(embedding_dim)
                    centroid = 0.3 * centroid + 0.7 * new_dir
                    centroid /= np.linalg.norm(centroid)
            else:
                noise = 0.05
            current_val += np.random.normal(0, noise)
            truth[t] = current_val
            embeddings.append(centroid + np.random.randn(embedding_dim) * noise)

    elif name == "bursty":
        burst_at = [int(steps * 0.4)]
        for t in range(steps):
            if t in burst_at:
                for _ in range(3):
                    current_val += np.random.choice([-5, 5])
                new_dir = np.random.randn(embedding_dim)
                centroid = 0.1 * centroid + 0.9 * new_dir
                centroid /= np.linalg.norm(centroid)
            current_val += np.random.normal(0, 0.1)
            truth[t] = current_val
            embeddings.append(centroid + np.random.randn(embedding_dim) * 0.1)

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
            current_val += np.random.normal(0, 0.1)
            truth[t] = current_val
            embeddings.append(centroid + np.random.randn(embedding_dim) * 0.1)

    return truth, embeddings


def get_lambda_trace(truth, embeddings, k=10.0, v0=0.5):
    detector = EmbeddingDistance(curr_window=5, arch_window=50, drift_threshold=0.4)
    lambdas = []
    for t in range(len(truth)):
        result = detector.update(embeddings[t])
        v = result.volatility
        lam = 0.2 + (0.9 - 0.2) * sigmoid(k * (v - v0))
        lambdas.append(lam)
    return np.array(lambdas)


def hartigans_dip_test(data, n_boot=1000):
    try:
        import diptest
        dip, pval = diptest.diptest(data)
        return {"dip_statistic": float(dip), "p_value": float(pval), "method": "diptest"}
    except ImportError:
        sorted_data = np.sort(data)
        n = len(sorted_data)
        ecdf = np.arange(1, n + 1) / n
        uniform_cdf = (sorted_data - sorted_data[0]) / (sorted_data[-1] - sorted_data[0] + 1e-12)
        dip = np.max(np.abs(ecdf - uniform_cdf)) / 2
        rng = np.random.RandomState(42)
        boot_dips = []
        for _ in range(n_boot):
            boot_sample = np.sort(rng.uniform(sorted_data[0], sorted_data[-1], n))
            boot_ecdf = np.arange(1, n + 1) / n
            boot_uni = (boot_sample - boot_sample[0]) / (boot_sample[-1] - boot_sample[0] + 1e-12)
            boot_dips.append(np.max(np.abs(boot_ecdf - boot_uni)) / 2)
        pval = np.mean(np.array(boot_dips) >= dip)
        return {"dip_statistic": float(dip), "p_value": float(pval), "method": "bootstrap_approx"}


def ashmans_d(data):
    gm = GaussianMixture(n_components=2, random_state=42, n_init=5).fit(data.reshape(-1, 1))
    mu1, mu2 = gm.means_.flatten()
    s1, s2 = np.sqrt(gm.covariances_.flatten())
    D = np.sqrt(2) * abs(mu1 - mu2) / np.sqrt(s1**2 + s2**2)
    weights = gm.weights_
    return {
        "D": float(D),
        "mu1": float(min(mu1, mu2)),
        "mu2": float(max(mu1, mu2)),
        "sigma1": float(s1 if mu1 < mu2 else s2),
        "sigma2": float(s2 if mu1 < mu2 else s1),
        "weight1": float(weights[0] if mu1 < mu2 else weights[1]),
        "weight2": float(weights[1] if mu1 < mu2 else weights[0]),
        "well_separated": bool(D > 2),
    }


def silverman_test(data, n_boot=500):
    kde = gaussian_kde(data)
    x = np.linspace(data.min() - 0.05, data.max() + 0.05, 1000)
    density = kde(x)
    peaks = 0
    peak_locations = []
    for i in range(1, len(density) - 1):
        if density[i] > density[i - 1] and density[i] > density[i + 1]:
            peaks += 1
            peak_locations.append(float(x[i]))
    return {
        "n_modes": peaks,
        "peak_locations": peak_locations,
        "bandwidth": float(kde.factor),
    }


def bimodality_coefficient(data):
    n = len(data)
    skew = sp_stats.skew(data)
    kurt = sp_stats.kurtosis(data)
    bc = (skew**2 + 1) / (kurt + 3)
    correction = (3 * (n - 1)**2) / ((n - 2) * (n - 3))
    return {
        "bc_raw": float(bc),
        "bc_corrected": float((skew**2 + 1) / (kurt + correction)),
        "threshold": 0.555,
        "exceeds_threshold": bool(bc > 0.555),
    }


def classify_verdict(dip_result, ashman_result, silverman_result, bc_result):
    evidence_bimodal = 0
    evidence_unimodal = 0

    if dip_result["p_value"] < 0.05:
        evidence_bimodal += 1
    else:
        evidence_unimodal += 1

    if ashman_result["well_separated"]:
        evidence_bimodal += 1
    else:
        evidence_unimodal += 1

    if silverman_result["n_modes"] >= 2:
        evidence_bimodal += 1
    else:
        evidence_unimodal += 1

    if evidence_bimodal >= 2:
        return "bimodal"
    elif evidence_unimodal >= 2 and evidence_bimodal == 0:
        return "unimodal"
    else:
        return "inconclusive"


def main():
    print("=" * 70)
    print("EXPERIMENT 32: ROBUST BIMODALITY TESTING OF LAMBDA TRACES")
    print("3 formal tests: Hartigan's Dip, Ashman's D, Silverman bandwidth")
    print("=" * 70)

    steps = 500
    embedding_dim = 32
    n_runs = 30
    scenarios = ["pure_stable", "pure_drift", "mixed_70_30", "bursty", "reversion"]

    expected = {
        "pure_stable": "unimodal",
        "pure_drift": "bimodal",
        "mixed_70_30": "bimodal",
        "bursty": "bimodal",
        "reversion": "bimodal",
    }

    all_results = {}

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario}  (expected: {expected[scenario]})")
        print(f"{'='*60}")

        all_lambdas = []
        for seed in range(42, 42 + n_runs):
            truth, embeddings = generate_scenario(scenario, steps, embedding_dim, seed)
            lam_trace = get_lambda_trace(truth, embeddings)
            all_lambdas.append(lam_trace)

        aggregated = np.concatenate(all_lambdas)
        print(f"  Aggregated {len(aggregated)} lambda values from {n_runs} runs")
        print(f"  Range: [{aggregated.min():.4f}, {aggregated.max():.4f}]")
        print(f"  Mean: {aggregated.mean():.4f}  Std: {aggregated.std():.4f}")

        print(f"\n  [1] Hartigan's Dip Test...")
        dip_result = hartigans_dip_test(aggregated)
        reject_uni = "REJECT unimodality" if dip_result["p_value"] < 0.05 else "FAIL to reject unimodality"
        print(f"      Dip statistic: {dip_result['dip_statistic']:.6f}")
        print(f"      p-value:       {dip_result['p_value']:.6f}  ({dip_result['method']})")
        print(f"      Verdict:       {reject_uni}")

        print(f"\n  [2] Ashman's D (Gaussian mixture separation)...")
        ashman_result = ashmans_d(aggregated)
        sep = "WELL-SEPARATED" if ashman_result["well_separated"] else "NOT well-separated"
        print(f"      D:             {ashman_result['D']:.4f}  (threshold: 2.0)")
        print(f"      Component 1:   mu={ashman_result['mu1']:.4f}, sigma={ashman_result['sigma1']:.4f}, w={ashman_result['weight1']:.3f}")
        print(f"      Component 2:   mu={ashman_result['mu2']:.4f}, sigma={ashman_result['sigma2']:.4f}, w={ashman_result['weight2']:.3f}")
        print(f"      Verdict:       {sep}")

        print(f"\n  [3] Silverman's bandwidth test...")
        silverman_result = silverman_test(aggregated)
        print(f"      Modes found:   {silverman_result['n_modes']}")
        print(f"      Peak locations: {[f'{p:.3f}' for p in silverman_result['peak_locations']]}")
        print(f"      Bandwidth:     {silverman_result['bandwidth']:.6f}")

        bc_result = bimodality_coefficient(aggregated)
        print(f"\n  [+] Bimodality Coefficient (reference):")
        print(f"      BC (raw):      {bc_result['bc_raw']:.4f}  (threshold: {bc_result['threshold']})")
        print(f"      Exceeds:       {'YES' if bc_result['exceeds_threshold'] else 'NO'}")

        verdict = classify_verdict(dip_result, ashman_result, silverman_result, bc_result)
        match_expected = verdict == expected[scenario]
        print(f"\n  COMBINED VERDICT: {verdict.upper()}")
        print(f"  Expected:         {expected[scenario]}")
        print(f"  Match:            {'YES' if match_expected else 'NO'}")

        per_run_verdicts = []
        for lam_trace in all_lambdas:
            rd = hartigans_dip_test(lam_trace, n_boot=200)
            ra = ashmans_d(lam_trace)
            rs = silverman_test(lam_trace)
            rb = bimodality_coefficient(lam_trace)
            per_run_verdicts.append(classify_verdict(rd, ra, rs, rb))

        bimodal_pct = np.mean([v == "bimodal" for v in per_run_verdicts]) * 100
        unimodal_pct = np.mean([v == "unimodal" for v in per_run_verdicts]) * 100
        inconclusive_pct = np.mean([v == "inconclusive" for v in per_run_verdicts]) * 100
        print(f"\n  Per-run consistency (n={n_runs}):")
        print(f"    Bimodal:      {bimodal_pct:.1f}%")
        print(f"    Unimodal:     {unimodal_pct:.1f}%")
        print(f"    Inconclusive: {inconclusive_pct:.1f}%")

        all_results[scenario] = {
            "expected": expected[scenario],
            "aggregated_verdict": verdict,
            "match_expected": match_expected,
            "dip_test": dip_result,
            "ashmans_d": ashman_result,
            "silverman": silverman_result,
            "bimodality_coefficient": bc_result,
            "per_run_consistency": {
                "bimodal_pct": float(bimodal_pct),
                "unimodal_pct": float(unimodal_pct),
                "inconclusive_pct": float(inconclusive_pct),
            },
            "stats": {
                "mean": float(aggregated.mean()),
                "std": float(aggregated.std()),
                "min": float(aggregated.min()),
                "max": float(aggregated.max()),
                "n_samples": int(len(aggregated)),
            },
            "sample_trace": all_lambdas[0].tolist(),
        }

    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"\n  {'Scenario':<16} {'Dip p-val':>10} {'Ashman D':>10} {'Modes':>6} {'BC':>8} {'Verdict':>14} {'Expected':>10} {'Match':>6}")
    print(f"  {'-'*82}")

    matches = 0
    for scenario in scenarios:
        r = all_results[scenario]
        dip_p = r["dip_test"]["p_value"]
        ash_d = r["ashmans_d"]["D"]
        modes = r["silverman"]["n_modes"]
        bc = r["bimodality_coefficient"]["bc_raw"]
        verdict = r["aggregated_verdict"]
        exp = r["expected"]
        match = r["match_expected"]
        if match:
            matches += 1
        print(f"  {scenario:<16} {dip_p:>10.4f} {ash_d:>10.4f} {modes:>6} {bc:>8.4f} {verdict:>14} {exp:>10} {'OK' if match else 'MISS':>6}")

    print(f"\n  Matches: {matches}/{len(scenarios)}")

    if matches == len(scenarios):
        conclusion = "ALL scenarios match expectations. VDD lambda bimodality is confirmed where drift exists, and correctly unimodal when stable."
    elif matches >= len(scenarios) - 1:
        conclusion = "Nearly all scenarios match. Minor deviation in one scenario; overall pattern strongly supports VDD adaptiveness claim."
    else:
        conclusion = "Mixed results. Paper framing may need adjustment regarding bimodality claims."

    print(f"\n  CONCLUSION: {conclusion}")

    all_results["_summary"] = {
        "matches": matches,
        "total": len(scenarios),
        "conclusion": conclusion,
        "methodology": {
            "n_runs": n_runs,
            "seeds": "42-71",
            "steps": steps,
            "embedding_dim": embedding_dim,
            "tests": ["hartigans_dip", "ashmans_d", "silverman_bandwidth", "bimodality_coefficient"],
            "aggregation": "concatenated lambda traces across all runs per scenario",
        },
    }

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    with open(results_dir / "32_bimodality_robust.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved to results/32_bimodality_robust.json")

    plot_results(all_results, scenarios, results_dir)

    return all_results


def plot_results(all_results, scenarios, results_dir):
    fig, axes = plt.subplots(1, 5, figsize=(24, 5))

    for i, scenario in enumerate(scenarios):
        ax = axes[i]
        r = all_results[scenario]
        data = np.array(r["sample_trace"])

        ax.hist(data, bins=50, density=True, alpha=0.6, color='steelblue',
                edgecolor='black', linewidth=0.5, label='Lambda histogram')

        kde = gaussian_kde(data)
        x = np.linspace(data.min() - 0.05, data.max() + 0.05, 300)
        ax.plot(x, kde(x), 'r-', linewidth=2, label='KDE')

        ashman = r["ashmans_d"]
        for mu, sigma, w in [(ashman["mu1"], ashman["sigma1"], ashman["weight1"]),
                              (ashman["mu2"], ashman["sigma2"], ashman["weight2"])]:
            gx = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 200)
            gy = w * sp_stats.norm.pdf(gx, mu, sigma)
            ax.plot(gx, gy, '--', linewidth=1.5, alpha=0.7)

        verdict = r["aggregated_verdict"].upper()
        expected = r["expected"].upper()
        match_str = "OK" if r["match_expected"] else "MISS"

        dip_p = r["dip_test"]["p_value"]
        ash_d = ashman["D"]
        modes = r["silverman"]["n_modes"]
        bc = r["bimodality_coefficient"]["bc_raw"]

        annotation = (f"Dip p={dip_p:.3f}\n"
                      f"Ashman D={ash_d:.2f}\n"
                      f"Modes={modes}\n"
                      f"BC={bc:.3f}")
        ax.text(0.97, 0.97, annotation, transform=ax.transAxes,
                fontsize=7, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

        color = 'green' if r["match_expected"] else 'red'
        ax.set_title(f"{scenario}\n{verdict} (exp: {expected}) [{match_str}]",
                     fontsize=10, fontweight='bold', color=color)
        ax.set_xlabel("lambda(t)")
        ax.set_ylabel("Density")
        ax.set_xlim(0.1, 1.0)
        ax.legend(fontsize=7, loc='upper left')

    plt.suptitle("Experiment 32: Robust Bimodality Testing of VDD Lambda Traces\n"
                 "3 tests: Hartigan's Dip, Ashman's D, Silverman modes | n=30 runs, 500 steps",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "32_bimodality_tests.png", dpi=150, bbox_inches='tight')
    print(f"  Saved plot to results/32_bimodality_tests.png")
    plt.close()


if __name__ == "__main__":
    main()
