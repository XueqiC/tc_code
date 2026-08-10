"""E1: compare fine, coarse, and multi-scale intra-domain boundaries.

This analysis is CPU-only and operates exclusively on cached gradient features.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PILOT = ROOT / "results" / "pilot"
FIGS = ROOT / "results" / "figs"
STEPS_PATH = PILOT / "features_gradsteps.npz"
COARSE_PATH = PILOT / "features_grad.npz"

if __name__ == "__main__" and not STEPS_PATH.exists():
    print(
        f"Skipping fine-scale E1 analysis: missing {STEPS_PATH}. "
        "Generate it with src/grad_features_steps.py on a GPU."
    )
    raise SystemExit(0)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from boundary import fit_subspace, score, unit
from e1_intra import (
    DOMAINS,
    MUTED,
    PAIRS,
    conformal_threshold,
    load_subgroups,
    make_splits,
)


SCORERS = ["fine", "coarse", "multi"]
COLORS = {
    "fine": "#2a78d6",
    "coarse": "#eb6834",
    "multi": "#1baf7a",
}


def load_features():
    """Load and validate block and whole-trajectory gradient caches."""
    with np.load(STEPS_PATH, allow_pickle=True) as steps:
        X_steps = steps["X_steps"].astype(np.float32, copy=False)
        row_id = steps["row_id"].astype(np.int64, copy=False)
        block_idx = steps["block_idx"].astype(np.int64, copy=False)
        step_domain = steps["domain"].astype(str)
    with np.load(COARSE_PATH, allow_pickle=True) as coarse:
        X_coarse = coarse["X"].astype(np.float32, copy=False)
        domain = coarse["domain"].astype(str)

    expected_domain = np.repeat(DOMAINS, 120)
    if not np.array_equal(domain, expected_domain):
        raise ValueError("cached gradient rows do not follow the expected domain order")
    if not np.array_equal(step_domain, domain):
        raise ValueError("step and coarse caches have different per-row domain order")
    if X_coarse.shape[0] != 600 or step_domain.shape != (600,):
        raise ValueError(
            f"unexpected per-row cache shapes: X={X_coarse.shape}, "
            f"domain={step_domain.shape}"
        )
    if X_steps.ndim != 2 or X_steps.shape[1] != X_coarse.shape[1]:
        raise ValueError(
            f"incompatible feature shapes: X_steps={X_steps.shape}, "
            f"X={X_coarse.shape}"
        )
    if row_id.shape != (len(X_steps),) or block_idx.shape != (len(X_steps),):
        raise ValueError(
            f"unexpected step index shapes: row_id={row_id.shape}, "
            f"block_idx={block_idx.shape}, X_steps={X_steps.shape}"
        )
    if np.any((row_id < 0) | (row_id >= len(domain))):
        raise ValueError("step cache contains an out-of-range global row_id")

    blocks_by_row = [np.flatnonzero(row_id == i) for i in range(len(domain))]
    missing_rows = [i for i, ids in enumerate(blocks_by_row) if len(ids) == 0]
    if missing_rows:
        raise ValueError(
            "step cache has no retained blocks for rows: "
            + ", ".join(map(str, missing_rows[:10]))
        )

    return unit(X_steps), unit(X_coarse), blocks_by_row, domain


def pooled_block_ids(blocks_by_row, query_rows):
    """Collect every block index belonging to the supplied global row ids."""
    return np.concatenate([blocks_by_row[row] for row in query_rows])


def fine_scores(U, X_steps, blocks_by_row, query_rows):
    """Score each row by the mean subspace ratio over all of its blocks."""
    return np.asarray([
        score(U, X_steps[blocks_by_row[row]]).mean()
        for row in query_rows
    ])


def evaluate(X_steps, X_coarse, blocks_by_row, splits):
    results = {pair_name: {} for pair_name, _, _ in PAIRS}
    for pair_name, _, _ in PAIRS:
        metrics = {name: {"auroc": [], "rejection": []} for name in SCORERS}
        for split in splits[pair_name]:
            fine_fit_ids = pooled_block_ids(blocks_by_row, split["fit"])
            U_fine = fit_subspace(X_steps[fine_fit_ids])
            U_coarse = fit_subspace(X_coarse[split["fit"]])

            fine_cal = fine_scores(
                U_fine, X_steps, blocks_by_row, split["cal"]
            )
            fine_in = fine_scores(
                U_fine, X_steps, blocks_by_row, split["test"]
            )
            fine_probe = fine_scores(
                U_fine, X_steps, blocks_by_row, split["probe"]
            )
            coarse_cal = score(U_coarse, X_coarse[split["cal"]])
            coarse_in = score(U_coarse, X_coarse[split["test"]])
            coarse_probe = score(U_coarse, X_coarse[split["probe"]])

            split_scores = {
                "fine": (fine_cal, fine_in, fine_probe),
                "coarse": (coarse_cal, coarse_in, coarse_probe),
                "multi": (
                    0.5 * coarse_cal + 0.5 * fine_cal,
                    0.5 * coarse_in + 0.5 * fine_in,
                    0.5 * coarse_probe + 0.5 * fine_probe,
                ),
            }
            for scorer_name, scores_by_role in split_scores.items():
                cal_scores, in_scores, probe_scores = scores_by_role
                threshold = conformal_threshold(cal_scores)
                y_true = np.r_[
                    np.ones(len(in_scores), dtype=int),
                    np.zeros(len(probe_scores), dtype=int),
                ]
                metrics[scorer_name]["auroc"].append(
                    roc_auc_score(y_true, np.r_[in_scores, probe_scores])
                )
                metrics[scorer_name]["rejection"].append(
                    float(np.mean(probe_scores < threshold))
                )

        for scorer_name in SCORERS:
            results[pair_name][scorer_name] = {
                key: np.asarray(values)
                for key, values in metrics[scorer_name].items()
            }
    return results


def markdown_report(results):
    lines = [
        "# E1: Fine-scale intra-domain boundary resolution",
        "",
        "Cells show AUROC mean±std over 5 seeds (mean conformal probe rejection rate).",
        "",
        "| directed pair (target vs probe) | fine | coarse | multi |",
        "|---|---:|---:|---:|",
    ]
    for pair_name, _, _ in PAIRS:
        cells = []
        for scorer_name in SCORERS:
            metrics = results[pair_name][scorer_name]
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

    for offset, scorer_name in zip((-width, 0.0, width), SCORERS):
        means = [results[pair][scorer_name]["auroc"].mean()
                 for pair in pair_names]
        stds = [results[pair][scorer_name]["auroc"].std()
                for pair in pair_names]
        ax.bar(
            x + offset,
            means,
            width,
            color=COLORS[scorer_name],
            label=scorer_name,
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
    fig.savefig(
        FIGS / "e1_intra_fine.png", dpi=150, bbox_inches="tight",
        facecolor="white"
    )
    plt.close(fig)


def main():
    if not STEPS_PATH.exists():
        print(
            f"Skipping fine-scale E1 analysis: missing {STEPS_PATH}. "
            "Generate it with src/grad_features_steps.py on a GPU."
        )
        return

    X_steps, X_coarse, blocks_by_row, domain = load_features()
    labels, counts = load_subgroups(domain)
    print("Subgroup counts:")
    for name, count in counts.items():
        print(f"  {name}: {count}")

    splits = make_splits(labels)
    results = evaluate(X_steps, X_coarse, blocks_by_row, splits)
    report = markdown_report(results)

    PILOT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    (PILOT / "e1_intra_fine.md").write_text(report)
    plot_results(results)
    print()
    print(report, end="")
    print(f"\nWrote {PILOT / 'e1_intra_fine.md'}")
    print(f"Wrote {FIGS / 'e1_intra_fine.png'}")


if __name__ == "__main__":
    main()
