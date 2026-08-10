"""E1: resolve response-defined boundaries within SQL and pandas domains.

This analysis is CPU-only and operates exclusively on cached gradient and
embedding features produced by the pilot experiments.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from boundary import fit_subspace, score, unit
from grad_features import stable_seed


ROOT = Path(__file__).resolve().parent.parent
PILOT = ROOT / "results" / "pilot"
FIGS = ROOT / "results" / "figs"
DOMAINS = ["gsm8k-code", "gsm8k-cot", "pandas", "sql", "alpaca"]
SPACES = ["grad", "emb-traj", "emb-query"]
PAIRS = [
    ("sql-join vs sql-plain", "sql-join", "sql-plain"),
    ("sql-plain vs sql-join", "sql-plain", "sql-join"),
    ("pandas-loc vs pandas-plain", "pandas-loc", "pandas-plain"),
    ("pandas-plain vs pandas-loc", "pandas-plain", "pandas-loc"),
]
COLORS = {
    "grad": "#2a78d6",
    "emb-traj": "#eb6834",
    "emb-query": "#1baf7a",
}
MUTED = "#6b6a63"
N_FIT, N_CAL, N_TEST, N_PROBE = 20, 10, 15, 30
ALPHA = 0.1
N_SEEDS = 5


def load_spaces():
    """Load and validate the three identically ordered cached feature spaces."""
    grad = np.load(PILOT / "features_grad.npz", allow_pickle=True)
    emb = np.load(PILOT / "features_emb.npz", allow_pickle=True)
    X = grad["X"].astype(np.float64)
    domain = grad["domain"].astype(str)
    Eq = emb["Eq"].astype(np.float64)
    Et = emb["Et"].astype(np.float64)

    expected_domain = np.repeat(DOMAINS, 120)
    if not np.array_equal(domain, expected_domain):
        raise ValueError("cached gradient rows do not follow the expected domain order")
    if X.shape[0] != 600 or Eq.shape != (600, 768) or Et.shape != (600, 768):
        raise ValueError(
            f"unexpected cache shapes: X={X.shape}, Eq={Eq.shape}, Et={Et.shape}"
        )

    return {
        "grad": unit(X),
        "emb-traj": unit(Et),
        "emb-query": unit(Eq),
    }, domain


def load_subgroups(domain):
    """Assign response-defined SQL and pandas subgroup labels in file order."""
    labels = np.full(len(domain), "", dtype=object)
    definitions = {
        "sql": ("join", "sql-join", "sql-plain"),
        "pandas": ("loc[", "pandas-loc", "pandas-plain"),
    }
    for domain_name, (needle, hit_label, plain_label) in definitions.items():
        path = ROOT / "data" / "pilot" / f"{domain_name}.jsonl"
        with path.open() as handle:
            rows = [json.loads(line) for line in handle]
        domain_ids = np.flatnonzero(domain == domain_name)
        if len(rows) != len(domain_ids):
            raise ValueError(
                f"{path} has {len(rows)} rows but the cache has {len(domain_ids)}"
            )
        labels[domain_ids] = [
            hit_label if needle in row["response"].lower() else plain_label
            for row in rows
        ]

    subgroup_names = ["sql-join", "sql-plain", "pandas-loc", "pandas-plain"]
    return labels, {name: int(np.sum(labels == name)) for name in subgroup_names}


def make_splits(labels):
    """Create shared per-pair splits so every feature space sees the same rows."""
    splits = {}
    required = N_FIT + N_CAL + N_TEST
    for pair_name, target, probe in PAIRS:
        target_ids = np.flatnonzero(labels == target)
        probe_ids = np.flatnonzero(labels == probe)
        if len(target_ids) < required:
            raise ValueError(f"{target} has {len(target_ids)} rows; need {required}")
        pair_splits = []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(
                1000 + stable_seed(pair_name) % 1000 + seed
            )
            shuffled = rng.permutation(target_ids)
            selected_probes = rng.choice(
                probe_ids, size=min(N_PROBE, len(probe_ids)), replace=False
            )
            pair_splits.append(
                {
                    "fit": shuffled[:N_FIT],
                    "cal": shuffled[N_FIT:N_FIT + N_CAL],
                    "test": shuffled[N_FIT + N_CAL:required],
                    "probe": selected_probes,
                }
            )
        splits[pair_name] = pair_splits
    return splits


def conformal_threshold(calibration_scores):
    """Lower split-conformal cutoff used by the pilot boundary analysis."""
    k = int(np.floor(ALPHA * (len(calibration_scores) + 1)))
    return np.sort(calibration_scores)[max(k - 1, 0)]


def evaluate(spaces, splits):
    results = {pair_name: {} for pair_name, _, _ in PAIRS}
    for pair_name, _, _ in PAIRS:
        for space_name in SPACES:
            X = spaces[space_name]
            aurocs, rejections = [], []
            for split in splits[pair_name]:
                U = fit_subspace(X[split["fit"]])
                threshold = conformal_threshold(score(U, X[split["cal"]]))
                in_scores = score(U, X[split["test"]])
                probe_scores = score(U, X[split["probe"]])
                y_true = np.r_[
                    np.ones(len(in_scores), dtype=int),
                    np.zeros(len(probe_scores), dtype=int),
                ]
                aurocs.append(
                    roc_auc_score(y_true, np.r_[in_scores, probe_scores])
                )
                rejections.append(float(np.mean(probe_scores < threshold)))
            results[pair_name][space_name] = {
                "auroc": np.asarray(aurocs),
                "rejection": np.asarray(rejections),
            }
    return results


def markdown_report(results):
    lines = [
        "# E1: Intra-domain boundary resolution",
        "",
        "Cells show AUROC mean±std over 5 seeds (mean conformal probe rejection rate).",
        "",
        "| directed pair (target vs probe) | grad | emb-traj | emb-query |",
        "|---|---:|---:|---:|",
    ]
    for pair_name, _, _ in PAIRS:
        cells = []
        for space_name in SPACES:
            metrics = results[pair_name][space_name]
            cells.append(
                f"{metrics['auroc'].mean():.3f}±{metrics['auroc'].std():.3f} "
                f"({metrics['rejection'].mean():.3f})"
            )
        lines.append(f"| {pair_name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def plot_results(results):
    pair_names = [pair_name for pair_name, _, _ in PAIRS]
    x = np.arange(len(pair_names))
    width = 0.23
    fig, ax = plt.subplots(figsize=(10.2, 4.8), facecolor="white")
    ax.set_facecolor("white")

    for offset, space_name in zip((-width, 0.0, width), SPACES):
        means = [results[pair][space_name]["auroc"].mean() for pair in pair_names]
        stds = [results[pair][space_name]["auroc"].std() for pair in pair_names]
        ax.bar(
            x + offset,
            means,
            width,
            color=COLORS[space_name],
            label=space_name,
            yerr=stds,
            error_kw={"ecolor": "black", "elinewidth": 0.8, "capsize": 2.5,
                      "capthick": 0.8},
        )

    ax.axhline(0.5, color=MUTED, linestyle="--", linewidth=1)
    ax.set_xticks(x, pair_names)
    ax.set_ylabel("AUROC", color=MUTED)
    ax.set_ylim(0.3, 1.02)
    ax.tick_params(axis="both", colors=MUTED, labelsize=9)
    for label in ax.get_xticklabels():
        label.set_rotation(12)
        label.set_ha("right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / "e1_intra.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def main():
    spaces, domain = load_spaces()
    labels, counts = load_subgroups(domain)
    print("Subgroup counts:")
    for name, count in counts.items():
        print(f"  {name}: {count}")

    splits = make_splits(labels)
    results = evaluate(spaces, splits)
    report = markdown_report(results)

    PILOT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    (PILOT / "e1_intra.md").write_text(report)
    plot_results(results)
    print()
    print(report, end="")
    print(f"\nWrote {PILOT / 'e1_intra.md'}")
    print(f"Wrote {FIGS / 'e1_intra.png'}")


if __name__ == "__main__":
    main()
