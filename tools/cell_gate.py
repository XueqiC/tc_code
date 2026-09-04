#!/usr/bin/env python3
"""Does the capability-signature space carry outcome information?

The generation policy we want targets regions of a *capability* space -- what a
query demands -- rather than the benchmark's own category labels, which are
arbitrary and do not transfer across benchmarks. That only works if the space
separates queries the student handles from queries it fails. Our own earlier
result (e1-intra-v1) says the opposite is possible: gradient features separate
coarse domains perfectly but sit near their resolution limit on fine
subdivision. So this is a gate, run before any generation budget is spent.

A query's signature is the LoRA-B gradient the base student receives from the
query paired with the TEACHER's verified trajectory -- what the task demands,
not what the student happened to do.

Two tests, both against a permutation null:

  distance-outcome  do queries close in signature space have similar phat?
                    Spearman between pairwise signature distance and pairwise
                    |phat difference|. No cluster count to choose.
  cell separation   for k cells, between-cell variance of phat over total
                    variance, reported across a range of k.

Neither passing means the space is unusable for targeting, and the honest move
is to improve the representation before buying data against it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_generated() -> tuple[list[dict], np.ndarray, list[str]]:
    """Outcome-labelled GENERATED queries -- the corpus the gate lacked.

    The first gate run had n=34 and could only detect a very strong effect.
    The generation loops since then left hundreds of items each carrying the
    student's own success count, which is exactly the label the test needs.
    Signature stays what the task demands: query + its verified answer.
    """
    import glob
    rows, outcomes, ids = [], [], []
    seen = set()
    for path in ("data/bfcl_sft/gen_pool.jsonl",
                 "data/bfcl_sft/gen_pool_v2.jsonl",
                 "data/bfcl_sft/gen_pool_v3.jsonl",
                 "data/bfcl_sft/gen_pool_v4.jsonl"):
        f = ROOT / path
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            attempts = item.get("student_attempts") or 0
            if not attempts or item["id"] in seen:
                continue
            seen.add(item["id"])
            target = item.get("ground_truth")
            target = json.dumps(target, ensure_ascii=False)[:4000]
            query = item["question"][0][0]["content"]
            rows.append({"prompt": query, "response": target})
            outcomes.append(item["student_good"] / attempts)
            ids.append(item["id"])
    return rows, np.asarray(outcomes), ids


def load_rows() -> tuple[list[dict], np.ndarray, list[str]]:
    from bfas.adapters.bfcl import BFCLAdapter

    adapter = BFCLAdapter()
    demos = json.loads((ROOT / "data/bfcl_sft/demos_ds.json").read_text())
    phat = json.loads((ROOT / "results/analysis/phat_v4.json").read_text())
    split = json.loads((ROOT / "configs/bfcl_support_split.json").read_text())

    rows, outcomes, ids = [], [], []
    for task_id in sorted(split["demand"]):
        demo = demos.get(task_id)
        if demo is None:
            continue  # no verified teacher trajectory -> no demand signature
        target = demo["teacher_result"]
        if not isinstance(target, str):
            target = json.dumps(target, ensure_ascii=False)
        try:
            prompt = adapter._render(
                adapter._messages(task_id), adapter._functions(task_id)
            )
        except Exception as exc:
            print(f"[cellgate] skip {task_id}: {type(exc).__name__}: {exc}")
            continue
        rows.append({"prompt": prompt, "response": target[:4000]})
        outcomes.append(float(phat.get(task_id, 0.0)))
        ids.append(task_id)
    return rows, np.asarray(outcomes), ids


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom else 0.0


def distance_outcome(feats: np.ndarray, phat: np.ndarray,
                     trials: int = 2000, seed: int = 0) -> dict:
    n = len(phat)
    iu = np.triu_indices(n, k=1)
    # features are L2-normalized, so cosine distance is the natural metric
    gram = feats @ feats.T
    dist = (1.0 - gram)[iu]
    gap = np.abs(phat[:, None] - phat[None, :])[iu]
    observed = spearman(dist, gap)
    rng = np.random.default_rng(seed)
    null = np.empty(trials)
    for t in range(trials):
        shuffled = phat[rng.permutation(n)]
        null[t] = spearman(dist, np.abs(
            shuffled[:, None] - shuffled[None, :])[iu])
    p = float((np.abs(null) >= abs(observed)).mean())
    return {"rho": observed, "p_permutation": p,
            "null_mean": float(null.mean()), "null_sd": float(null.std())}


def cell_separation(feats: np.ndarray, phat: np.ndarray, k: int,
                    trials: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    labels = kmeans(feats, k, rng)
    total = phat.var()
    if total == 0:
        return {"k": k, "explained": 0.0, "p_permutation": 1.0}

    def explained(values: np.ndarray) -> float:
        between = 0.0
        for cell in range(k):
            mask = labels == cell
            if mask.sum():
                between += mask.sum() * (values[mask].mean() - values.mean())**2
        return float(between / len(values) / total)

    observed = explained(phat)
    null = np.array([explained(phat[rng.permutation(len(phat))])
                     for _ in range(trials)])
    return {"k": k, "explained": observed,
            "p_permutation": float((null >= observed).mean()),
            "null_mean": float(null.mean())}


def kmeans(x: np.ndarray, k: int, rng, iters: int = 60) -> np.ndarray:
    centers = x[rng.choice(len(x), k, replace=False)]
    labels = np.zeros(len(x), dtype=int)
    for _ in range(iters):
        new = np.argmin(((x[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
        if (new == labels).all():
            break
        labels = new
        for cell in range(k):
            mask = labels == cell
            if mask.sum():
                centers[cell] = x[mask].mean(0)
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--proj-dim", type=int, default=64)
    # extract_features defaults to a 640-token prompt cap, which is the old
    # truncation setting: BFCL prompts carry full function schemas and run to
    # thousands of tokens, so at 640 the signature is computed on a fragment
    # that may not contain what distinguishes one capability demand from another
    parser.add_argument("--max-prompt-tok", type=int, default=4096)
    parser.add_argument("--generated", action="store_true",
                        help="test on the outcome-labelled generated corpus "
                             "instead of the 34 demand tasks")
    parser.add_argument("--out", default="results/analysis/cell_gate.json")
    args = parser.parse_args()

    rows, phat, ids = (load_generated() if args.generated else load_rows())
    print(f"[cellgate] {len(rows)} demand tasks with a verified teacher "
          f"trajectory; phat spread {phat.min():.2f}-{phat.max():.2f}")
    if len(rows) < 8:
        print("[cellgate] too few tasks to test")
        return 1

    from features import extract_features

    feats = extract_features(args.student, rows, proj_dim=args.proj_dim,
                             max_prompt_tok=args.max_prompt_tok)
    feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    print(f"[cellgate] signatures {feats.shape}")

    result = {
        "n_tasks": len(rows),
        "max_prompt_tok": args.max_prompt_tok,
        "task_ids": ids,
        "phat": phat.tolist(),
        "distance_outcome": distance_outcome(feats, phat),
        "cell_separation": [cell_separation(feats, phat, k)
                            for k in (3, 4, 5, 6, 8, 12, 16)],
    }
    d = result["distance_outcome"]
    print(f"\ndistance-outcome  rho={d['rho']:+.3f}  "
          f"p={d['p_permutation']:.4f}  (null {d['null_mean']:+.3f}"
          f"±{d['null_sd']:.3f})")
    print("\ncell separation")
    print(f"  {'k':>2s} {'explained':>10s} {'p':>8s} {'null':>8s}")
    for row in result["cell_separation"]:
        print(f"  {row['k']:2d} {row['explained']:10.3f} "
              f"{row['p_permutation']:8.4f} {row['null_mean']:8.3f}")

    passed = (d["p_permutation"] < 0.05
              or any(r["p_permutation"] < 0.05
                     for r in result["cell_separation"]))
    result["verdict"] = "PASS" if passed else "FAIL"
    print(f"\nVERDICT: {result['verdict']} — "
          + ("the signature space carries outcome information; targeting "
             "generation at its regions is justified."
             if passed else
             "no detectable link between signature geometry and success. "
             "Improve the representation before spending generation budget "
             "against these regions."))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"[cellgate] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
