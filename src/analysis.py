"""
Analyze representation drift between base, critique, and paraphrase conditions.
Produces metrics and visualizations for the research report.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import cosine as cosine_dist
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

RESULTS_DIR = "results"
DATA_DIR = "results/data"
PLOT_DIR = "results/plots"

os.makedirs(PLOT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.2)


def load_data():
    """Load all activation data."""
    base = np.load(f"{DATA_DIR}/base_activations.npy")
    critique = np.load(f"{DATA_DIR}/critique_activations.npy")
    paraphrase = np.load(f"{DATA_DIR}/paraphrase_activations.npy")
    labels = np.load(f"{DATA_DIR}/labels.npy")
    return base, critique, paraphrase, labels


# =============================================================================
# Metric 1: Per-layer cosine similarity (drift magnitude)
# =============================================================================

def compute_cosine_similarity_per_layer(base, critique, paraphrase):
    """
    Compute cosine similarity between base and critique/paraphrase at each layer.
    Returns: dict with 'critique' and 'paraphrase' arrays of shape (n_layers,)
    """
    n_samples, n_layers, d_model = base.shape
    cos_critique = np.zeros((n_samples, n_layers))
    cos_paraphrase = np.zeros((n_samples, n_layers))

    for i in range(n_samples):
        for l in range(n_layers):
            b = base[i, l]
            c = critique[i, l]
            p = paraphrase[i, l]

            # Avoid division by zero
            if np.linalg.norm(b) > 1e-8 and np.linalg.norm(c) > 1e-8:
                cos_critique[i, l] = 1 - cosine_dist(b, c)
            if np.linalg.norm(b) > 1e-8 and np.linalg.norm(p) > 1e-8:
                cos_paraphrase[i, l] = 1 - cosine_dist(b, p)

    return {
        "critique_mean": cos_critique.mean(axis=0),
        "critique_std": cos_critique.std(axis=0),
        "paraphrase_mean": cos_paraphrase.mean(axis=0),
        "paraphrase_std": cos_paraphrase.std(axis=0),
        "critique_raw": cos_critique,
        "paraphrase_raw": cos_paraphrase,
    }


def plot_cosine_similarity(cos_data, n_layers):
    """Plot per-layer cosine similarity for both conditions."""
    layers = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(layers, cos_data["critique_mean"], "o-", color="tab:red",
            label="Base → Critique", markersize=4)
    ax.fill_between(layers,
                    cos_data["critique_mean"] - cos_data["critique_std"],
                    cos_data["critique_mean"] + cos_data["critique_std"],
                    alpha=0.2, color="tab:red")

    ax.plot(layers, cos_data["paraphrase_mean"], "s-", color="tab:blue",
            label="Base → Paraphrase", markersize=4)
    ax.fill_between(layers,
                    cos_data["paraphrase_mean"] - cos_data["paraphrase_std"],
                    cos_data["paraphrase_mean"] + cos_data["paraphrase_std"],
                    alpha=0.2, color="tab:blue")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Representation Drift: Cosine Similarity by Layer\n(Lower = More Drift)")
    ax.legend()
    ax.set_xlim(0, n_layers - 1)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/cosine_similarity_per_layer.png", dpi=150)
    plt.close()
    print("Saved cosine_similarity_per_layer.png")


# =============================================================================
# Metric 2: L2 displacement magnitude
# =============================================================================

def compute_displacement(base, critique, paraphrase):
    """Compute L2 displacement magnitude per layer."""
    n_samples, n_layers, _ = base.shape

    disp_critique = np.zeros((n_samples, n_layers))
    disp_paraphrase = np.zeros((n_samples, n_layers))

    for i in range(n_samples):
        for l in range(n_layers):
            disp_critique[i, l] = np.linalg.norm(critique[i, l] - base[i, l])
            disp_paraphrase[i, l] = np.linalg.norm(paraphrase[i, l] - base[i, l])

    return {
        "critique_mean": disp_critique.mean(axis=0),
        "critique_std": disp_critique.std(axis=0),
        "paraphrase_mean": disp_paraphrase.mean(axis=0),
        "paraphrase_std": disp_paraphrase.std(axis=0),
        "critique_raw": disp_critique,
        "paraphrase_raw": disp_paraphrase,
    }


def plot_displacement(disp_data, n_layers):
    """Plot L2 displacement per layer."""
    layers = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(layers, disp_data["critique_mean"], "o-", color="tab:red",
            label="Base → Critique", markersize=4)
    ax.fill_between(layers,
                    disp_data["critique_mean"] - disp_data["critique_std"],
                    disp_data["critique_mean"] + disp_data["critique_std"],
                    alpha=0.2, color="tab:red")

    ax.plot(layers, disp_data["paraphrase_mean"], "s-", color="tab:blue",
            label="Base → Paraphrase", markersize=4)
    ax.fill_between(layers,
                    disp_data["paraphrase_mean"] - disp_data["paraphrase_std"],
                    disp_data["paraphrase_mean"] + disp_data["paraphrase_std"],
                    alpha=0.2, color="tab:blue")

    ax.set_xlabel("Layer")
    ax.set_ylabel("L2 Displacement")
    ax.set_title("Representation Displacement Magnitude by Layer\n(Higher = More Change)")
    ax.legend()
    ax.set_xlim(0, n_layers - 1)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/displacement_per_layer.png", dpi=150)
    plt.close()
    print("Saved displacement_per_layer.png")


# =============================================================================
# Metric 3: Statistical comparison (Wilcoxon per layer)
# =============================================================================

def statistical_tests(cos_data, disp_data, n_layers):
    """Run paired Wilcoxon signed-rank tests comparing critique vs paraphrase drift."""
    results = []
    for l in range(n_layers):
        # Compare cosine similarities (critique vs paraphrase)
        cos_stat, cos_p = stats.wilcoxon(
            cos_data["critique_raw"][:, l],
            cos_data["paraphrase_raw"][:, l],
            alternative="two-sided"
        )
        # Compare displacements
        disp_stat, disp_p = stats.wilcoxon(
            disp_data["critique_raw"][:, l],
            disp_data["paraphrase_raw"][:, l],
            alternative="two-sided"
        )
        # Effect size (rank-biserial correlation approximation)
        n = len(cos_data["critique_raw"][:, l])
        cos_effect = 1 - (2 * cos_stat) / (n * (n + 1) / 2)
        disp_effect = 1 - (2 * disp_stat) / (n * (n + 1) / 2)

        results.append({
            "layer": l,
            "cos_statistic": float(cos_stat),
            "cos_p_value": float(cos_p),
            "cos_effect_size": float(cos_effect),
            "disp_statistic": float(disp_stat),
            "disp_p_value": float(disp_p),
            "disp_effect_size": float(disp_effect),
            "cos_sig_bonferroni": cos_p < 0.05 / n_layers,
            "disp_sig_bonferroni": disp_p < 0.05 / n_layers,
        })

    return results


def plot_pvalues(stat_results, n_layers):
    """Plot p-values per layer."""
    layers = [r["layer"] for r in stat_results]
    cos_ps = [r["cos_p_value"] for r in stat_results]
    disp_ps = [r["disp_p_value"] for r in stat_results]
    threshold = 0.05 / n_layers  # Bonferroni

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.semilogy(layers, cos_ps, "o-", label="Cosine similarity p-value", color="tab:orange")
    ax.semilogy(layers, disp_ps, "s-", label="Displacement p-value", color="tab:green")
    ax.axhline(threshold, ls="--", color="gray", label=f"Bonferroni threshold ({threshold:.4f})")
    ax.axhline(0.05, ls=":", color="lightgray", label="p = 0.05")
    ax.set_xlabel("Layer")
    ax.set_ylabel("p-value (log scale)")
    ax.set_title("Statistical Significance: Critique vs Paraphrase Drift")
    ax.legend(fontsize=10)
    ax.set_xlim(0, n_layers - 1)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/pvalues_per_layer.png", dpi=150)
    plt.close()
    print("Saved pvalues_per_layer.png")


# =============================================================================
# Metric 4: Linear probe accuracy (correct vs incorrect)
# =============================================================================

def train_linear_probes(base, critique, paraphrase, labels, n_layers):
    """
    Train linear probes to classify correct vs incorrect answers
    on base, critique, and paraphrase activations separately.
    """
    if len(np.unique(labels)) < 2:
        print("WARNING: Only one class in labels, skipping probe training")
        return None

    results = {"base": [], "critique": [], "paraphrase": []}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for l in range(n_layers):
        for cond_name, acts in [("base", base), ("critique", critique), ("paraphrase", paraphrase)]:
            X = acts[:, l, :]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
            scores = cross_val_score(clf, X_scaled, labels, cv=cv, scoring="accuracy")
            results[cond_name].append({
                "layer": l,
                "accuracy_mean": float(scores.mean()),
                "accuracy_std": float(scores.std()),
            })

    return results


def plot_probe_accuracy(probe_results, n_layers):
    """Plot linear probe accuracy per layer for each condition."""
    if probe_results is None:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    for cond, color, marker in [("base", "tab:gray", "o"),
                                  ("critique", "tab:red", "^"),
                                  ("paraphrase", "tab:blue", "s")]:
        means = [r["accuracy_mean"] for r in probe_results[cond]]
        stds = [r["accuracy_std"] for r in probe_results[cond]]
        layers = [r["layer"] for r in probe_results[cond]]

        ax.plot(layers, means, f"{marker}-", color=color, label=f"{cond.capitalize()}", markersize=4)
        ax.fill_between(layers,
                       np.array(means) - np.array(stds),
                       np.array(means) + np.array(stds),
                       alpha=0.15, color=color)

    ax.axhline(0.5, ls="--", color="gray", alpha=0.5, label="Chance")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy (5-fold CV)")
    ax.set_title("Linear Probe Accuracy: Correct vs Incorrect Answer Classification")
    ax.legend()
    ax.set_xlim(0, n_layers - 1)
    ax.set_ylim(0.3, 1.0)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/probe_accuracy_per_layer.png", dpi=150)
    plt.close()
    print("Saved probe_accuracy_per_layer.png")


# =============================================================================
# Metric 5: PCA visualization
# =============================================================================

def pca_visualization(base, critique, paraphrase, labels, selected_layers):
    """PCA visualization of activations at selected layers."""
    n_selected = len(selected_layers)
    fig, axes = plt.subplots(1, n_selected, figsize=(6 * n_selected, 5))
    if n_selected == 1:
        axes = [axes]

    for idx, layer in enumerate(selected_layers):
        ax = axes[idx]
        # Combine all conditions
        all_acts = np.vstack([base[:, layer], critique[:, layer], paraphrase[:, layer]])
        n = len(base)
        condition_labels = (["Base"] * n + ["Critique"] * n + ["Paraphrase"] * n)
        correctness = np.concatenate([labels, labels, labels])

        pca = PCA(n_components=2)
        projected = pca.fit_transform(all_acts)

        colors = {"Base": "tab:gray", "Critique": "tab:red", "Paraphrase": "tab:blue"}
        markers_correct = {0: "x", 1: "o"}

        for cond in ["Base", "Critique", "Paraphrase"]:
            for correct in [0, 1]:
                mask = np.array([(cl == cond and c == correct)
                               for cl, c in zip(condition_labels, correctness)])
                if mask.sum() == 0:
                    continue
                ax.scatter(projected[mask, 0], projected[mask, 1],
                          c=colors[cond], marker=markers_correct[correct],
                          alpha=0.5, s=30,
                          label=f"{cond} ({'correct' if correct else 'wrong'})")

        ax.set_title(f"Layer {layer}\n(Var: {pca.explained_variance_ratio_[:2].sum():.1%})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        if idx == n_selected - 1:
            ax.legend(fontsize=7, loc="upper right")

    plt.suptitle("PCA of Residual Stream Activations", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/pca_visualization.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved pca_visualization.png")


# =============================================================================
# Metric 6: CKA (Centered Kernel Alignment)
# =============================================================================

def linear_CKA(X, Y):
    """Compute linear CKA between two activation matrices."""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    hsic_xy = np.linalg.norm(X.T @ Y, "fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, "fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, "fro") ** 2

    if hsic_xx * hsic_yy == 0:
        return 0.0
    return hsic_xy / np.sqrt(hsic_xx * hsic_yy)


def compute_cka_per_layer(base, critique, paraphrase, n_layers):
    """Compute CKA between base and critique/paraphrase at each layer."""
    cka_critique = []
    cka_paraphrase = []
    cka_crit_para = []

    for l in range(n_layers):
        cka_critique.append(linear_CKA(base[:, l], critique[:, l]))
        cka_paraphrase.append(linear_CKA(base[:, l], paraphrase[:, l]))
        cka_crit_para.append(linear_CKA(critique[:, l], paraphrase[:, l]))

    return {
        "critique": cka_critique,
        "paraphrase": cka_paraphrase,
        "crit_vs_para": cka_crit_para,
    }


def plot_cka(cka_data, n_layers):
    """Plot CKA per layer."""
    layers = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(layers, cka_data["critique"], "o-", color="tab:red",
            label="Base ↔ Critique", markersize=4)
    ax.plot(layers, cka_data["paraphrase"], "s-", color="tab:blue",
            label="Base ↔ Paraphrase", markersize=4)
    ax.plot(layers, cka_data["crit_vs_para"], "^-", color="tab:green",
            label="Critique ↔ Paraphrase", markersize=4)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Linear CKA")
    ax.set_title("Centered Kernel Alignment Between Conditions")
    ax.legend()
    ax.set_xlim(0, n_layers - 1)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/cka_per_layer.png", dpi=150)
    plt.close()
    print("Saved cka_per_layer.png")


# =============================================================================
# Metric 7: Drift direction consistency
# =============================================================================

def compute_drift_direction_consistency(base, critique, paraphrase, n_layers):
    """
    Measure consistency of drift directions across samples at each layer.
    Higher consistency = more structured (systematic) drift.
    """
    n_samples = base.shape[0]
    critique_consistency = []
    paraphrase_consistency = []

    for l in range(n_layers):
        # Compute drift vectors
        crit_drifts = critique[:, l] - base[:, l]  # (n_samples, d_model)
        para_drifts = paraphrase[:, l] - base[:, l]

        # Mean drift direction
        mean_crit = crit_drifts.mean(axis=0)
        mean_para = para_drifts.mean(axis=0)

        # Cosine similarity of each drift with mean drift
        crit_cos = []
        para_cos = []
        for i in range(n_samples):
            if np.linalg.norm(crit_drifts[i]) > 1e-8 and np.linalg.norm(mean_crit) > 1e-8:
                crit_cos.append(1 - cosine_dist(crit_drifts[i], mean_crit))
            if np.linalg.norm(para_drifts[i]) > 1e-8 and np.linalg.norm(mean_para) > 1e-8:
                para_cos.append(1 - cosine_dist(para_drifts[i], mean_para))

        critique_consistency.append(np.mean(crit_cos) if crit_cos else 0)
        paraphrase_consistency.append(np.mean(para_cos) if para_cos else 0)

    return {
        "critique": critique_consistency,
        "paraphrase": paraphrase_consistency,
    }


def plot_drift_consistency(consistency_data, n_layers):
    """Plot drift direction consistency."""
    layers = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(layers, consistency_data["critique"], "o-", color="tab:red",
            label="Critique drift consistency", markersize=4)
    ax.plot(layers, consistency_data["paraphrase"], "s-", color="tab:blue",
            label="Paraphrase drift consistency", markersize=4)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Cosine Similarity with Mean Drift")
    ax.set_title("Drift Direction Consistency Across Samples\n(Higher = More Systematic)")
    ax.legend()
    ax.set_xlim(0, n_layers - 1)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/drift_consistency.png", dpi=150)
    plt.close()
    print("Saved drift_consistency.png")


# =============================================================================
# Main analysis pipeline
# =============================================================================

def main():
    print("Loading activation data...")
    base, critique, paraphrase, labels = load_data()
    n_samples, n_layers, d_model = base.shape
    print(f"Shape: {n_samples} samples, {n_layers} layers, {d_model} dims")
    print(f"Labels: {labels.sum()} correct, {(1-labels).sum()} wrong")

    all_results = {
        "n_samples": int(n_samples),
        "n_layers": int(n_layers),
        "d_model": int(d_model),
        "n_correct": int(labels.sum()),
        "n_wrong": int((1-labels).sum()),
    }

    # 1. Cosine similarity
    print("\n=== Cosine Similarity Analysis ===")
    cos_data = compute_cosine_similarity_per_layer(base, critique, paraphrase)
    plot_cosine_similarity(cos_data, n_layers)
    all_results["cosine_similarity"] = {
        "critique_mean": cos_data["critique_mean"].tolist(),
        "paraphrase_mean": cos_data["paraphrase_mean"].tolist(),
    }

    # 2. L2 displacement
    print("\n=== Displacement Analysis ===")
    disp_data = compute_displacement(base, critique, paraphrase)
    plot_displacement(disp_data, n_layers)
    all_results["displacement"] = {
        "critique_mean": disp_data["critique_mean"].tolist(),
        "paraphrase_mean": disp_data["paraphrase_mean"].tolist(),
    }

    # 3. Statistical tests
    print("\n=== Statistical Tests ===")
    stat_results = statistical_tests(cos_data, disp_data, n_layers)
    plot_pvalues(stat_results, n_layers)
    n_sig_cos = sum(1 for r in stat_results if r["cos_sig_bonferroni"])
    n_sig_disp = sum(1 for r in stat_results if r["disp_sig_bonferroni"])
    print(f"Significant layers (Bonferroni): cosine={n_sig_cos}/{n_layers}, displacement={n_sig_disp}/{n_layers}")
    all_results["statistical_tests"] = stat_results

    # 4. Linear probes
    print("\n=== Linear Probe Analysis ===")
    probe_results = train_linear_probes(base, critique, paraphrase, labels, n_layers)
    plot_probe_accuracy(probe_results, n_layers)
    if probe_results:
        all_results["probe_accuracy"] = probe_results

    # 5. PCA visualization
    print("\n=== PCA Visualization ===")
    # Select early, middle, and late layers
    selected = [1, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    pca_visualization(base, critique, paraphrase, labels, selected)

    # 6. CKA
    print("\n=== CKA Analysis ===")
    cka_data = compute_cka_per_layer(base, critique, paraphrase, n_layers)
    plot_cka(cka_data, n_layers)
    all_results["cka"] = cka_data

    # 7. Drift direction consistency
    print("\n=== Drift Direction Consistency ===")
    consistency = compute_drift_direction_consistency(base, critique, paraphrase, n_layers)
    plot_drift_consistency(consistency, n_layers)
    all_results["drift_consistency"] = consistency

    # Save all results (convert numpy types for JSON)
    def convert(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    import copy
    def deep_convert(obj):
        if isinstance(obj, dict):
            return {k: deep_convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [deep_convert(v) for v in obj]
        return convert(obj)

    with open(f"{RESULTS_DIR}/analysis_results.json", "w") as f:
        json.dump(deep_convert(all_results), f, indent=2)
    print(f"\nAll results saved to {RESULTS_DIR}/analysis_results.json")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Average cosine similarity difference
    cos_crit_mean = np.mean(cos_data["critique_mean"])
    cos_para_mean = np.mean(cos_data["paraphrase_mean"])
    print(f"Avg cosine sim (base→critique): {cos_crit_mean:.4f}")
    print(f"Avg cosine sim (base→paraphrase): {cos_para_mean:.4f}")
    print(f"Difference: {cos_para_mean - cos_crit_mean:.4f} (positive = critique drifts more)")

    disp_crit_mean = np.mean(disp_data["critique_mean"])
    disp_para_mean = np.mean(disp_data["paraphrase_mean"])
    print(f"\nAvg displacement (base→critique): {disp_crit_mean:.2f}")
    print(f"Avg displacement (base→paraphrase): {disp_para_mean:.2f}")

    print(f"\nSignificant layers (Bonferroni-corrected):")
    print(f"  Cosine: {n_sig_cos}/{n_layers}")
    print(f"  Displacement: {n_sig_disp}/{n_layers}")

    if probe_results:
        best_base = max(r["accuracy_mean"] for r in probe_results["base"])
        best_crit = max(r["accuracy_mean"] for r in probe_results["critique"])
        best_para = max(r["accuracy_mean"] for r in probe_results["paraphrase"])
        print(f"\nBest probe accuracy:")
        print(f"  Base: {best_base:.3f}")
        print(f"  Critique: {best_crit:.3f}")
        print(f"  Paraphrase: {best_para:.3f}")


if __name__ == "__main__":
    main()
