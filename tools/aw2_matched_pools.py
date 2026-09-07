#!/usr/bin/env python3
"""Build the AW-2 early/spread matched control pools (§8.2).

Only shared tasks with non-overflow events enter either arm. Each contributes
min(n_early, n_spread) rows, sampled without replacement with --seed (default 0).
Both arms receive the same final row permutation. No dU/composition filtering.

Defaults read aw1_base_k3.jsonl and aw1_spread_k3.jsonl in data/appworld_events
and write pool_aw2_{early,spread}_matched.jsonl plus aw2_matched_report.json
there. --halfdose also keeps every other row of the shuffled early matched
pool (ceil(n/2) rows). Reported steps assume one epoch and eight rows per step.
Response token counts use a locally cached tokenizer, with no special tokens,
chat template, or truncation; they are target lengths, not measured loss tokens.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re


EVENT_DIR = Path("data/appworld_events")
DEFAULT_TOKENIZER = "Qwen/Qwen3.5-4B"
ROWS_PER_STEP = 8
DOC_LOGIN_RE = re.compile(r"\b(?:api_docs|login)\b", re.IGNORECASE)
CODE_RE = re.compile(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE)


def read_events(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def group_events(rows):
    groups = defaultdict(list)
    for row in rows:
        if not row.get("_event_overflow"):
            groups[row["task_id"]].append(row)
    return groups


def match_pools(early_rows, spread_rows, seed=0):
    """Return matched arms and eligible per-task counts, preserving whole rows."""
    early, spread = group_events(early_rows), group_events(spread_rows)
    tasks = sorted(early.keys() & spread.keys())
    if not tasks:
        raise ValueError("No shared tasks with non-overflow events.")
    early_rng, spread_rng = random.Random(seed), random.Random(seed)
    matched_early, matched_spread, counts = [], [], {}
    for task in tasks:
        n = min(len(early[task]), len(spread[task]))
        counts[task] = {
            "n_early": len(early[task]), "n_spread": len(spread[task]), "n_matched": n,
        }
        matched_early.extend(early_rng.sample(early[task], n))
        matched_spread.extend(spread_rng.sample(spread[task], n))
    # Identical task order in both arms, including within each training batch.
    order = list(range(len(matched_early)))
    random.Random(seed).shuffle(order)
    return ([matched_early[i] for i in order],
            [matched_spread[i] for i in order], counts)


def is_doc_login(response):
    """Regex heuristic on teacher code only; raw code is accepted without fences."""
    blocks = CODE_RE.findall(response)
    code = "\n".join(blocks) if blocks else response
    return bool(DOC_LOGIN_RE.search(code))


def probe_position(row):
    """Older early events predate _probe_pos; derive the miner's same statistic."""
    if row.get("_probe_pos") is not None:
        return row["_probe_pos"], "recorded"
    turn = row.get("turn_index", row.get("_event_turn"))
    demo_len = row.get("_demo_len")
    if turn is not None and demo_len is not None and demo_len > 0:
        return (turn + 1) / demo_len, "derived"
    return None, "missing"


def summarize_pool(rows, tokenizer):
    n = len(rows)
    positions, position_sources = [], Counter()
    signs = dict.fromkeys(("positive", "negative", "zero", "missing"), 0)
    tokens = []
    for row in rows:
        pos, source = probe_position(row)
        position_sources[source] += 1
        if pos is not None:
            positions.append(pos)
        du = row.get("_event_dU")
        sign = ("missing" if du is None else "positive" if du > 0
                else "negative" if du < 0 else "zero")
        signs[sign] += 1
        tokens.append(len(tokenizer.encode(row["response"], add_special_tokens=False)))
    doc_rows = sum(is_doc_login(row["response"]) for row in rows)
    task_counts = Counter(row["task_id"] for row in rows)
    return {
        "tasks": sorted(task_counts),
        "per_task_counts": dict(sorted(task_counts.items())),
        "total_rows": n,
        "steps": (n + ROWS_PER_STEP - 1) // ROWS_PER_STEP,
        "mean_probe_pos": sum(positions) / len(positions) if positions else None,
        "probe_pos_sources": {key: position_sources[key]
                              for key in ("recorded", "derived", "missing")},
        "doc_login_rows": doc_rows,
        "doc_login_fraction": doc_rows / n if n else None,
        "mean_yT_tokens": sum(tokens) / n if n else None,
        "total_yT_tokens": sum(tokens),
        "dU_sign_counts": signs,
        "dU_sign_proportions": {key: count / n if n else None
                                for key, count in signs.items()},
    }


def load_tokenizer(name):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name, local_files_only=True, trust_remote_code=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("early", nargs="?", type=Path,
                        default=EVENT_DIR / "aw1_base_k3.jsonl")
    parser.add_argument("spread", nargs="?", type=Path,
                        default=EVENT_DIR / "aw1_spread_k3.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=EVENT_DIR)
    parser.add_argument("--report", type=Path,
                        help="Default: <out-dir>/aw2_matched_report.json")
    parser.add_argument("--halfdose", action="store_true",
                        help="Also keep rows 0, 2, ... of the seeded early matched pool")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                        help="Cached tokenizer name or local directory; no downloads")
    args = parser.parse_args(argv)

    inputs = {"early": read_events(args.early), "spread": read_events(args.spread)}
    try:
        early, spread, counts = match_pools(inputs["early"], inputs["spread"], args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    tokenizer = load_tokenizer(args.tokenizer)
    pools = {"early_matched": early, "spread_matched": spread}
    if args.halfdose:
        pools["early_halfdose"] = early[::2]
    report = {
        "seed": args.seed,
        "sources": {"early": str(args.early), "spread": str(args.spread)},
        "tasks": list(counts),
        "n_tasks": len(counts),
        "per_task_counts": counts,
        "input_counts": {
            arm: {
                "total_rows": len(rows),
                "overflow_rows": sum(bool(row.get("_event_overflow")) for row in rows),
                "eligible_rows": sum(not row.get("_event_overflow") for row in rows),
                "unmatched_rows": sum(not row.get("_event_overflow")
                                      and row["task_id"] not in counts for row in rows),
            } for arm, rows in inputs.items()
        },
        "rows_per_step": ROWS_PER_STEP,
        "step_rule": "ceil(total_rows / 8), one epoch",
        "token_lengths": {
            "tokenizer": args.tokenizer,
            "add_special_tokens": False,
            "scope": "Full response, including fences/prose; no template or truncation",
        },
        "doc_login_detection": {
            "regex": DOC_LOGIN_RE.pattern,
            "flags": "IGNORECASE",
            "scope": "Python/unlabelled fenced response code, or raw response if unfenced",
        },
        "probe_pos_fallback": "(turn_index + 1) / _demo_len; _event_turn if turn_index absent",
        "halfdose_rule": "early_matched[::2] after seeded shuffle; ceil(n/2) rows",
        "arms": {name: {"path": str(args.out_dir / f"pool_aw2_{name}.jsonl"),
                        **summarize_pool(rows, tokenizer)} for name, rows in pools.items()},
    }
    # Compute/serialize the complete report before creating any output artifacts.
    report_text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    report_path = args.report or args.out_dir / "aw2_matched_report.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    for name, rows in pools.items():
        path = Path(report["arms"][name]["path"])
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stats = report["arms"][name]
        print(f"{name}: {len(rows)} rows, {len(stats['tasks'])} tasks, "
              f"ceil({len(rows)}/8) = {stats['steps']} steps -> {path}")
    report_path.write_text(report_text, encoding="utf-8")
    print(f"report -> {report_path}")


if __name__ == "__main__":
    main()
