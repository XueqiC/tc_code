#!/usr/bin/env python3
"""Assign every training-pool row to a capability cluster (P1).

Signatures are computed per distinct TASK -- the (query, verified answer) pair
that defines what the task demands -- and every row inherits its task's cluster.
Computing per task rather than per row keeps the geometry about capability
demand rather than about rendering (a multi-turn task's rows share one demand),
and costs ~a hundred backward passes instead of a thousand.

k is chosen by the same statistic the gate validated: the k in a small range
whose clusters explain the most outcome variance on the tasks that carry
outcomes, against a permutation p-value. No hand-set cluster count.

Output: results/analysis/pool_clusters.json
  {"k": ..., "assignments": {task_id: cluster}, "cluster_fail": {cluster: rate}}

Usage: PYTHONPATH=src CUDA_VISIBLE_DEVICES=<gpu> python tools/cluster_pool.py \
          [--pool data/bfcl_sft/pool_bfcl_v9.jsonl] [--max-prompt-tok 2048]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))


def gather_tasks(pool_path: Path) -> tuple[list[dict], list[str], np.ndarray]:
    """One (query, answer) exemplar per distinct task in the pool, plus the
    student outcome for tasks that carry one (generated items and phat'd
    demand tasks); NaN where no outcome exists."""
    from bfas.adapters.bfcl import BFCLAdapter

    adapter = BFCLAdapter()
    entries, _ = adapter._load_entries()
    phat = json.loads((ROOT / "results/analysis/phat_v4.json").read_text())

    gen_items: dict[str, dict] = {}
    for name in ("gen_pool.jsonl", "gen_pool_v2.jsonl", "gen_pool_v3.jsonl",
                 "gen_pool_v4.jsonl", "gen_stateful_v3.jsonl", "gen_oos.jsonl"):
        f = ROOT / "data/bfcl_sft" / name
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            gen_items.setdefault(item["id"], item)

    rows, ids, outcomes = [], [], []
    seen: set[str] = set()
    for line in pool_path.read_text().splitlines():
        if not line.strip():
            continue
        task_id = json.loads(line)["task_id"]
        if task_id in seen:
            continue
        seen.add(task_id)

        if task_id in gen_items:
            item = gen_items[task_id]
            query = " ".join(
                turn[0]["content"] for turn in item["question"] if turn)
            answer = json.dumps(item.get("ground_truth"),
                                ensure_ascii=False)[:4000]
            attempts = item.get("student_attempts") or 0
            outcome = (item["student_good"] / attempts) if attempts else np.nan
        elif task_id in entries:
            entry = entries[task_id]
            query = " ".join(
                turn[0]["content"] for turn in entry["question"] if turn)
            answer = json.dumps(entry.get("ground_truth", ""))[:4000] or query
            outcome = phat.get(task_id, np.nan)
        else:
            continue
        rows.append({"prompt": query, "response": answer})
        ids.append(task_id)
        outcomes.append(outcome)
    return rows, ids, np.asarray(outcomes, dtype=float)


def gather_generic(pool_path: Path) -> tuple[list[dict], list[str], np.ndarray]:
    """Benchmark-agnostic variant for pools whose rows already carry the
    student's own state (AppWorld/ALFWorld/tau2 pools): one exemplar per
    distinct task_id taken straight from the row (prompt/response), outcome
    = the row's _task_phat stamp when present, NaN otherwise."""
    rows, ids, outcomes = [], [], []
    seen: set[str] = set()
    for line in pool_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task_id = str(row.get("task_id"))
        if task_id in seen:
            continue
        seen.add(task_id)
        prompt = row.get("prompt") or " ".join(
            m.get("content", "") for m in row.get("messages", [])
            if isinstance(m, dict) and m.get("role") != "assistant")
        rows.append({"prompt": str(prompt)[:8000],
                     "response": str(row.get("response", ""))[:4000]})
        ids.append(task_id)
        phat = row.get("_task_phat")
        outcomes.append(float(phat) if phat is not None else np.nan)
    return rows, ids, np.asarray(outcomes, dtype=float)


def pick_k(feats: np.ndarray, outcomes: np.ndarray,
           candidates=(4, 6, 8, 12, 16), trials: int = 1000) -> tuple[int, dict]:
    from cell_gate import cell_separation

    labelled = ~np.isnan(outcomes)
    best_k, best = None, None
    report = {}
    for k in candidates:
        if k >= labelled.sum():
            continue
        result = cell_separation(feats[labelled], outcomes[labelled], k,
                                 trials=trials)
        report[k] = result
        # a partition is only usable if its outcome structure is unlikely
        # under permutation; among the significant ones take the most
        # explanatory, otherwise fall back to the smallest p
        significant = result["p_permutation"] < 0.05
        score = (significant, result["explained"] if significant
                 else -result["p_permutation"])
        if best is None or score > best:
            best, best_k = score, k
    return best_k, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default="data/bfcl_sft/pool_bfcl_v9.jsonl")
    parser.add_argument("--student", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--max-prompt-tok", type=int, default=2048)
    parser.add_argument("--out", default="results/analysis/pool_clusters.json")
    parser.add_argument("--generic", action="store_true",
                        help="read prompt/response/_task_phat straight from the "
                             "pool rows (AppWorld/ALFWorld/tau2) instead of the "
                             "BFCL entry tables")
    args = parser.parse_args()

    gather = gather_generic if args.generic else gather_tasks
    rows, ids, outcomes = gather(ROOT / args.pool)
    print(f"[cluster] {len(rows)} distinct tasks "
          f"({int((~np.isnan(outcomes)).sum())} with outcomes)")

    from features import extract_features
    from cell_gate import kmeans

    feats = extract_features(args.student, rows, proj_dim=64,
                             max_prompt_tok=args.max_prompt_tok)
    feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)

    k, report = pick_k(feats, outcomes)
    print(f"[cluster] chose k={k}")
    for kk, result in report.items():
        print(f"    k={kk:2d} explained={result['explained']:.3f} "
              f"p={result['p_permutation']:.4f}")

    rng = np.random.default_rng(0)
    labels = kmeans(feats, k, rng)
    cluster_fail: dict[int, float] = {}
    for cluster in range(k):
        mask = (labels == cluster) & ~np.isnan(outcomes)
        cluster_fail[cluster] = float(1 - outcomes[mask].mean()) if mask.any() \
            else float("nan")
        members = int((labels == cluster).sum())
        print(f"    cluster {cluster}: {members} tasks, "
              f"fail={cluster_fail[cluster]:.2f}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "k": int(k),
        "assignments": {task: int(label) for task, label in zip(ids, labels)},
        "cluster_fail": {str(c): v for c, v in cluster_fail.items()},
        "gate_report": {str(kk): r for kk, r in report.items()},
    }, indent=1))
    print(f"[cluster] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
