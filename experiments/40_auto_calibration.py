#!/usr/bin/env python3
"""
Experiment 40: Auto-Calibration of V_0 Parameter

V_0 (volatility threshold) is VDD's most sensitive parameter -- paper Limitation #5
shows a 254% performance range across V_0 values.

This experiment demonstrates a simple auto-calibration method:
  1. Run a burn-in period collecting raw volatilities
  2. Set V_0 = percentile(volatilities, 75) from burn-in
  3. Compare auto-calibrated V_0 vs hand-tuned (0.1) vs wrong values

Drift patterns tested: sudden, gradual, reversion, mixed
Seeds: 20 per condition for statistical reliability
No external dependencies (Ollama not required).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from pathlib import Path
from scipy import stats


# ---------------------------------------------------------------------------
# Inline drift detector (mirrors EmbeddingDistance from src/vdd)
# ---------------------------------------------------------------------------

class InlineEmbeddingDistanceDetector:
    """Self-contained embedding distance drift detector."""

    def __init__(self, curr_window=10, arch_window=100, drift_threshold=0.4, smoothing=0.1):
        self.curr_window = curr_window
        self.arch_window = arch_window
        self.drift_threshold = drift_threshold
        self.smoothing = smoothing
        self._curr_buffer = deque(maxlen=curr_window)
        self._arch_buffer = deque(maxlen=arch_window)
        self._volatility = 0.0

    def reset(self):
        self._curr_buffer.clear()
        self._arch_buffer.clear()
        self._volatility = 0.0

    def update(self, embedding):
        embedding = np.asarray(embedding).flatten()
        self._curr_buffer.append(embedding)
        self._arch_buffer.append(embedding)

        if len(self._curr_buffer) < self.curr_window // 2:
            return 0.0, 0.0
        if len(self._arch_buffer) < self.curr_window * 2:
            return 0.0, 0.0

        curr_centroid = np.mean(list(self._curr_buffer), axis=0)
        arch_list = list(self._arch_buffer)
        arch_centroid = np.mean(arch_list[:-self.curr_window], axis=0)

        norm_c = np.linalg.norm(curr_centroid)
        norm_a = np.linalg.norm(arch_centroid)
        if norm_c < 1e-10 or norm_a < 1e-10:
            distance = 0.0
        else:
            cos_sim = np.clip(np.dot(curr_centroid, arch_centroid) / (norm_c * norm_a), -1.0, 1.0)
            distance = 1.0 - cos_sim

        drift_detected = distance > self.drift_threshold
        raw_volatility = min(1.0, distance / self.drift_threshold)
        self._volatility = self.smoothing * raw_volatility + (1 - self.smoothing) * self._volatility
        if drift_detected:
            self._volatility = max(self._volatility, 0.8)

        return self._volatility, distance


# ---------------------------------------------------------------------------
# VDD core: lambda(t) = lambda_base + (lambda_max - lambda_base) * sigmoid(k * (V_t - V_0))
# ---------------------------------------------------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def vdd_lambda(v_t, v_0, lambda_base=0.05, lambda_max=0.9, k=10):
    return lambda_base + (lambda_max - lambda_base) * sigmoid(k * (v_t - v_0))


# ---------------------------------------------------------------------------
# Auto-calibration
# ---------------------------------------------------------------------------

def auto_calibrate_v0(volatility_history, percentile=75):
    """
    Calibrate V_0 from observed volatilities during burn-in.

    Sets V_0 to the given percentile of observed volatilities so that
    'normal' fluctuations stay below threshold and only genuine drift
    triggers fast decay.

    Args:
        volatility_history: array of raw volatility values from burn-in
        percentile: percentile to use (default 75 = top 25% triggers fast decay)

    Returns:
        Calibrated V_0 value
    """
    vols = np.array(volatility_history)
    vols = vols[vols > 0]
    if len(vols) == 0:
        return 0.1
    return float(np.percentile(vols, percentile))


# ---------------------------------------------------------------------------
# Drift pattern generators
# ---------------------------------------------------------------------------

def generate_sudden_drift(steps, embedding_dim, seed):
    """Sharp regime shifts at fixed intervals."""
    np.random.seed(seed)
    truth = np.zeros(steps)
    embeddings = []
    current_val = 0.0
    centroid = np.random.randn(embedding_dim)
    centroid /= np.linalg.norm(centroid)

    shift_points = [steps // 4, steps // 2, 3 * steps // 4]

    for t in range(steps):
        if t in shift_points:
            current_val += np.random.choice([-5, 5])
            new_dir = np.random.randn(embedding_dim)
            centroid = 0.3 * centroid + 0.7 * new_dir
            centroid /= np.linalg.norm(centroid)
        else:
            current_val += np.random.normal(0, 0.05)
        truth[t] = current_val
        emb = centroid + np.random.randn(embedding_dim) * 0.1
        embeddings.append(emb)

    return truth, embeddings, "sudden"


def generate_gradual_drift(steps, embedding_dim, seed):
    """Slow continuous drift -- centroid rotates over time."""
    np.random.seed(seed)
    truth = np.zeros(steps)
    embeddings = []
    current_val = 0.0
    centroid = np.random.randn(embedding_dim)
    centroid /= np.linalg.norm(centroid)
    drift_dir = np.random.randn(embedding_dim)
    drift_dir /= np.linalg.norm(drift_dir)

    for t in range(steps):
        alpha = 0.005
        centroid = centroid + alpha * drift_dir
        centroid /= np.linalg.norm(centroid)
        current_val += 0.02 + np.random.normal(0, 0.03)
        truth[t] = current_val
        emb = centroid + np.random.randn(embedding_dim) * 0.1
        embeddings.append(emb)

    return truth, embeddings, "gradual"


def generate_reversion_drift(steps, embedding_dim, seed):
    """Shift then revert to original -- tests memory retention."""
    np.random.seed(seed)
    truth = np.zeros(steps)
    embeddings = []
    current_val = 0.0
    original_centroid = np.random.randn(embedding_dim)
    original_centroid /= np.linalg.norm(original_centroid)
    centroid = original_centroid.copy()

    shift_t = steps // 3
    revert_t = 2 * steps // 3

    for t in range(steps):
        if t == shift_t:
            current_val += 5
            new_dir = np.random.randn(embedding_dim)
            centroid = 0.2 * centroid + 0.8 * new_dir
            centroid /= np.linalg.norm(centroid)
        elif t == revert_t:
            current_val -= 5
            centroid = 0.2 * centroid + 0.8 * original_centroid
            centroid /= np.linalg.norm(centroid)
        else:
            current_val += np.random.normal(0, 0.05)
        truth[t] = current_val
        emb = centroid + np.random.randn(embedding_dim) * 0.1
        embeddings.append(emb)

    return truth, embeddings, "reversion"


def generate_mixed_drift(steps, embedding_dim, seed):
    """Mixed: ~70% stable, ~30% drift periods (from Exp 04 pattern)."""
    np.random.seed(seed)
    truth = np.zeros(steps)
    embeddings = []
    current_val = 0.0
    centroid = np.random.randn(embedding_dim)
    centroid /= np.linalg.norm(centroid)

    periods = [
        (0, int(0.16 * steps), False),
        (int(0.16 * steps), int(0.20 * steps), True),
        (int(0.20 * steps), int(0.40 * steps), False),
        (int(0.40 * steps), int(0.44 * steps), True),
        (int(0.44 * steps), int(0.56 * steps), False),
        (int(0.56 * steps), int(0.64 * steps), True),
        (int(0.64 * steps), int(0.90 * steps), False),
        (int(0.90 * steps), int(0.94 * steps), True),
        (int(0.94 * steps), steps, False),
    ]

    drift_flags = np.zeros(steps, dtype=bool)
    for start, end, is_drift in periods:
        if is_drift:
            drift_flags[start:end] = True

    for t in range(steps):
        if drift_flags[t] and (t == 0 or not drift_flags[t - 1]):
            new_dir = np.random.randn(embedding_dim)
            centroid = 0.3 * centroid + 0.7 * new_dir
            centroid /= np.linalg.norm(centroid)
            current_val += np.random.choice([-3, 3])
        elif drift_flags[t]:
            current_val += np.random.normal(0, 0.3)
        else:
            current_val += np.random.normal(0, 0.05)
        truth[t] = current_val
        emb = centroid + np.random.randn(embedding_dim) * 0.1
        embeddings.append(emb)

    return truth, embeddings, "mixed"


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

DRIFT_GENERATORS = {
    "sudden": generate_sudden_drift,
    "gradual": generate_gradual_drift,
    "reversion": generate_reversion_drift,
    "mixed": generate_mixed_drift,
}


def run_vdd_with_v0(truth, embeddings, v_0, lambda_base=0.05, lambda_max=0.9, k=10):
    """Run VDD with a specific V_0 and return metrics."""
    steps = len(truth)
    detector = InlineEmbeddingDistanceDetector(curr_window=10, arch_window=100, drift_threshold=0.4)
    memory = np.zeros(steps)
    stored = 0.0
    lambdas = np.zeros(steps)

    for t in range(1, steps):
        v_t, _ = detector.update(embeddings[t])
        lam = vdd_lambda(v_t, v_0, lambda_base, lambda_max, k)
        lambdas[t] = lam
        stored = (1 - lam) * stored + lam * truth[t]
        memory[t] = stored

    error = np.abs(truth - memory)
    iae = float(np.sum(error))
    mae = float(np.mean(error))

    correct = np.abs(error) < np.std(truth) * 0.5
    accuracy = float(np.mean(correct[20:]))

    return {"iae": iae, "mae": mae, "accuracy": accuracy}


def collect_burn_in_volatilities(embeddings, burn_in_steps):
    """Collect raw volatility readings during burn-in."""
    detector = InlineEmbeddingDistanceDetector(curr_window=10, arch_window=100, drift_threshold=0.4)
    volatilities = []
    for t in range(burn_in_steps):
        v_t, _ = detector.update(embeddings[t])
        volatilities.append(v_t)
    return volatilities


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment():
    print("=" * 70)
    print("Experiment 40: Auto-Calibration of V_0 Parameter")
    print("=" * 70)

    steps = 500
    embedding_dim = 32
    n_seeds = 20
    burn_in_frac = 0.2
    burn_in_steps = int(steps * burn_in_frac)

    v0_hand_tuned = 0.1
    v0_wrong_values = [0.01, 0.05, 0.3, 0.5]
    all_v0_fixed = [v0_hand_tuned] + v0_wrong_values

    patterns = list(DRIFT_GENERATORS.keys())

    results = {}
    auto_v0_values = {p: [] for p in patterns}

    for pattern in patterns:
        print(f"\n--- Pattern: {pattern} ---")
        gen_fn = DRIFT_GENERATORS[pattern]

        pattern_results = {
            "hand_tuned": {"iae": [], "mae": [], "accuracy": []},
            "auto_calibrated": {"iae": [], "mae": [], "accuracy": []},
        }
        for v0_val in v0_wrong_values:
            label = f"v0={v0_val}"
            pattern_results[label] = {"iae": [], "mae": [], "accuracy": []}

        for seed in range(42, 42 + n_seeds):
            truth, embeddings, _ = gen_fn(steps, embedding_dim, seed)

            burn_in_vols = collect_burn_in_volatilities(embeddings, burn_in_steps)
            v0_auto = auto_calibrate_v0(burn_in_vols, percentile=75)
            auto_v0_values[pattern].append(v0_auto)

            res_hand = run_vdd_with_v0(truth, embeddings, v0_hand_tuned)
            for k_metric in ["iae", "mae", "accuracy"]:
                pattern_results["hand_tuned"][k_metric].append(res_hand[k_metric])

            res_auto = run_vdd_with_v0(truth, embeddings, v0_auto)
            for k_metric in ["iae", "mae", "accuracy"]:
                pattern_results["auto_calibrated"][k_metric].append(res_auto[k_metric])

            for v0_val in v0_wrong_values:
                label = f"v0={v0_val}"
                res_wrong = run_vdd_with_v0(truth, embeddings, v0_val)
                for k_metric in ["iae", "mae", "accuracy"]:
                    pattern_results[label][k_metric].append(res_wrong[k_metric])

        results[pattern] = pattern_results

        auto_mean = np.mean(auto_v0_values[pattern])
        auto_std = np.std(auto_v0_values[pattern])
        hand_iae = np.mean(pattern_results["hand_tuned"]["iae"])
        auto_iae = np.mean(pattern_results["auto_calibrated"]["iae"])
        pct_diff = (auto_iae - hand_iae) / hand_iae * 100

        print(f"  Auto-calibrated V_0: {auto_mean:.4f} +/- {auto_std:.4f}")
        print(f"  Hand-tuned IAE:      {hand_iae:.2f}")
        print(f"  Auto-calibrated IAE: {auto_iae:.2f} ({pct_diff:+.1f}%)")

    print_summary(results, auto_v0_values, v0_wrong_values)
    return results, auto_v0_values


def print_summary(results, auto_v0_values, v0_wrong_values):
    print(f"\n{'=' * 70}")
    print("SUMMARY: Auto-Calibrated V_0 Values")
    print(f"{'=' * 70}")
    print(f"{'Pattern':<12} {'Auto V_0':>10} {'Hand V_0':>10} {'Auto IAE':>10} {'Hand IAE':>10} {'Diff':>8}")
    print("-" * 62)

    for pattern in results:
        auto_mean = np.mean(auto_v0_values[pattern])
        hand_iae = np.mean(results[pattern]["hand_tuned"]["iae"])
        auto_iae = np.mean(results[pattern]["auto_calibrated"]["iae"])
        diff = (auto_iae - hand_iae) / hand_iae * 100
        print(f"{pattern:<12} {auto_mean:>10.4f} {0.1:>10.4f} {auto_iae:>10.2f} {hand_iae:>10.2f} {diff:>+7.1f}%")

    print(f"\n{'=' * 70}")
    print("IAE COMPARISON: All V_0 values (mean over 20 seeds)")
    print(f"{'=' * 70}")

    header = f"{'Pattern':<12} {'Hand(0.1)':>10} {'Auto':>10}"
    for v0_val in v0_wrong_values:
        header += f" {'V0=' + str(v0_val):>10}"
    print(header)
    print("-" * (32 + 11 * len(v0_wrong_values)))

    for pattern in results:
        hand_iae = np.mean(results[pattern]["hand_tuned"]["iae"])
        auto_iae = np.mean(results[pattern]["auto_calibrated"]["iae"])
        row = f"{pattern:<12} {hand_iae:>10.2f} {auto_iae:>10.2f}"
        for v0_val in v0_wrong_values:
            label = f"v0={v0_val}"
            wrong_iae = np.mean(results[pattern][label]["iae"])
            row += f" {wrong_iae:>10.2f}"
        print(row)

    print(f"\n{'=' * 70}")
    print("STATISTICAL SIGNIFICANCE (paired t-test, hand-tuned vs auto-calibrated)")
    print(f"{'=' * 70}")
    for pattern in results:
        hand_iaes = results[pattern]["hand_tuned"]["iae"]
        auto_iaes = results[pattern]["auto_calibrated"]["iae"]
        t_stat, p_value = stats.ttest_rel(hand_iaes, auto_iaes)
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        better = "AUTO" if np.mean(auto_iaes) < np.mean(hand_iaes) else "HAND"
        print(f"  {pattern:<12}: t={t_stat:>7.3f}, p={p_value:.4f} {sig:>4s} -> {better} wins")

    print(f"\n{'=' * 70}")
    print("PERFORMANCE RANGE ANALYSIS")
    print(f"{'=' * 70}")
    for pattern in results:
        all_iae_means = {}
        all_iae_means["hand_tuned"] = np.mean(results[pattern]["hand_tuned"]["iae"])
        all_iae_means["auto"] = np.mean(results[pattern]["auto_calibrated"]["iae"])
        for v0_val in v0_wrong_values:
            label = f"v0={v0_val}"
            all_iae_means[label] = np.mean(results[pattern][label]["iae"])

        worst = max(all_iae_means.values())
        best = min(all_iae_means.values())
        pct_range = (worst - best) / best * 100
        best_label = min(all_iae_means, key=all_iae_means.get)
        auto_rank = sorted(all_iae_means.values()).index(all_iae_means["auto"]) + 1
        hand_rank = sorted(all_iae_means.values()).index(all_iae_means["hand_tuned"]) + 1
        n_methods = len(all_iae_means)

        print(f"  {pattern:<12}: range={pct_range:.0f}% | best={best_label} | "
              f"auto rank={auto_rank}/{n_methods} | hand rank={hand_rank}/{n_methods}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results, auto_v0_values, v0_wrong_values, save_path):
    patterns = list(results.keys())
    n_patterns = len(patterns)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    method_colors = {
        "hand_tuned": "#2ecc71",
        "auto_calibrated": "#3498db",
    }
    wrong_colors = ["#e74c3c", "#e67e22", "#9b59b6", "#95a5a6"]

    for idx, pattern in enumerate(patterns):
        ax = axes[idx]
        pr = results[pattern]

        labels = []
        means = []
        cis = []
        colors = []
        n = len(pr["hand_tuned"]["iae"])

        for label, color in [("hand_tuned", method_colors["hand_tuned"]),
                             ("auto_calibrated", method_colors["auto_calibrated"])]:
            iae_vals = pr[label]["iae"]
            labels.append(label.replace("_", "\n"))
            means.append(np.mean(iae_vals))
            cis.append(1.96 * np.std(iae_vals) / np.sqrt(n))
            colors.append(color)

        for v0_val, color in zip(v0_wrong_values, wrong_colors):
            key = f"v0={v0_val}"
            iae_vals = pr[key]["iae"]
            labels.append(f"V0={v0_val}")
            means.append(np.mean(iae_vals))
            cis.append(1.96 * np.std(iae_vals) / np.sqrt(n))
            colors.append(color)

        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=cis, capsize=5, color=colors, edgecolor="black",
                      alpha=0.85, linewidth=0.8)

        for bar, mean_val, ci_val in zip(bars, means, cis):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ci_val + 0.5,
                    f"{mean_val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        auto_v0_mean = np.mean(auto_v0_values[pattern])
        ax.set_title(f"{pattern.title()} Drift\n(auto V_0={auto_v0_mean:.3f})", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("IAE (lower is better)")
        ax.set_ylim(0, max(means) * 1.25)
        ax.axhline(y=means[0], color=method_colors["hand_tuned"], linestyle="--", alpha=0.4, linewidth=1)

    fig.suptitle(
        "Experiment 40: Auto-Calibration of V_0\n"
        "Green=hand-tuned(0.1) | Blue=auto-calibrated(P75) | "
        "Red/Orange/Purple/Gray=wrong V_0\n"
        f"n={len(pr['hand_tuned']['iae'])} seeds, 95% CI error bars",
        fontsize=13, fontweight="bold", y=1.02,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {save_path}")
    plt.close()


def plot_auto_v0_distribution(auto_v0_values, save_path):
    """Plot distribution of auto-calibrated V_0 across seeds and patterns."""
    patterns = list(auto_v0_values.keys())
    fig, ax = plt.subplots(figsize=(10, 6))

    positions = np.arange(len(patterns))
    bp = ax.boxplot(
        [auto_v0_values[p] for p in patterns],
        positions=positions,
        widths=0.5,
        patch_artist=True,
        boxprops=dict(facecolor="#3498db", alpha=0.7),
        medianprops=dict(color="black", linewidth=2),
    )

    ax.axhline(y=0.1, color="#2ecc71", linestyle="--", linewidth=2, label="Hand-tuned V_0=0.1")
    ax.set_xticks(positions)
    ax.set_xticklabels([p.title() for p in patterns])
    ax.set_ylabel("Auto-Calibrated V_0")
    ax.set_title("Distribution of Auto-Calibrated V_0 Across Drift Patterns\n(n=20 seeds per pattern)")
    ax.legend()

    for i, pattern in enumerate(patterns):
        vals = auto_v0_values[pattern]
        mean_val = np.mean(vals)
        ax.annotate(f"{mean_val:.3f}", (i, mean_val), textcoords="offset points",
                    xytext=(30, 0), fontsize=9, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="gray"))

    plt.tight_layout()

    dist_path = str(save_path).replace(".png", "_v0_distribution.png")
    plt.savefig(dist_path, dpi=150, bbox_inches="tight")
    print(f"Saved V_0 distribution plot to {dist_path}")
    plt.close()


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------

def save_results(results, auto_v0_values, v0_wrong_values, json_path):
    out = {"experiment": "40_auto_calibration", "n_seeds": 20, "steps": 500,
           "burn_in_fraction": 0.2, "percentile": 75, "patterns": {}}

    for pattern in results:
        pr = results[pattern]
        p_out = {
            "auto_v0_mean": float(np.mean(auto_v0_values[pattern])),
            "auto_v0_std": float(np.std(auto_v0_values[pattern])),
            "auto_v0_values": [float(v) for v in auto_v0_values[pattern]],
            "methods": {},
        }

        for label in pr:
            m = pr[label]
            p_out["methods"][label] = {
                "iae_mean": float(np.mean(m["iae"])),
                "iae_std": float(np.std(m["iae"])),
                "mae_mean": float(np.mean(m["mae"])),
                "mae_std": float(np.std(m["mae"])),
                "accuracy_mean": float(np.mean(m["accuracy"])),
                "accuracy_std": float(np.std(m["accuracy"])),
            }

        hand_iae = np.mean(pr["hand_tuned"]["iae"])
        auto_iae = np.mean(pr["auto_calibrated"]["iae"])
        p_out["auto_vs_hand_pct"] = float((auto_iae - hand_iae) / hand_iae * 100)

        t_stat, p_value = stats.ttest_rel(pr["hand_tuned"]["iae"], pr["auto_calibrated"]["iae"])
        p_out["ttest_t"] = float(t_stat)
        p_out["ttest_p"] = float(p_value)

        all_iae = {}
        all_iae["hand_tuned"] = hand_iae
        all_iae["auto"] = auto_iae
        for v0_val in v0_wrong_values:
            key = f"v0={v0_val}"
            all_iae[key] = float(np.mean(pr[key]["iae"]))
        worst = max(all_iae.values())
        best = min(all_iae.values())
        p_out["performance_range_pct"] = float((worst - best) / best * 100)
        p_out["best_method"] = min(all_iae, key=all_iae.get)
        sorted_methods = sorted(all_iae.items(), key=lambda x: x[1])
        p_out["auto_rank"] = [k for k, _ in sorted_methods].index("auto") + 1
        p_out["hand_rank"] = [k for k, _ in sorted_methods].index("hand_tuned") + 1
        p_out["total_methods"] = len(all_iae)

        out["patterns"][pattern] = p_out

    overall_auto_gap = np.mean([out["patterns"][p]["auto_vs_hand_pct"] for p in out["patterns"]])
    out["overall_auto_vs_hand_pct"] = float(overall_auto_gap)

    auto_never_worst = all(
        out["patterns"][p]["auto_rank"] < out["patterns"][p]["total_methods"]
        for p in out["patterns"]
    )
    out["auto_never_worst"] = auto_never_worst

    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved JSON results to {json_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    results, auto_v0_values = run_experiment()

    v0_wrong_values = [0.01, 0.05, 0.3, 0.5]

    plot_path = results_dir / "40_auto_calibration.png"
    plot_results(results, auto_v0_values, v0_wrong_values, str(plot_path))

    plot_auto_v0_distribution(auto_v0_values, str(plot_path))

    json_path = results_dir / "40_auto_calibration.json"
    save_results(results, auto_v0_values, v0_wrong_values, str(json_path))

    print("\nDone.")


if __name__ == "__main__":
    main()
