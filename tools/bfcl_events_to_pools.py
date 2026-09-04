#!/usr/bin/env python3
"""Turn mined intervention events into the CRCD B-ladder training pools.

Inputs: events_v1.jsonl (stateful, causal dU from K continuations) and
events_single_v1.jsonl (single-turn, dU = 1 - student pass rate).

Outputs (all rows are appworld_train.py rows: prompt / response / turn_index /
_task_phat / token_hint, plus _rejected for preference arms):

  pool_events_ce.jsonl    B2: corrected-span CE only (response = a+), consequential events
  pool_events_pref.jsonl  B3/B4: same rows with _rejected = a- (AW_DISTILL=ddpo consumes them)

Normalisation: the student's replies carry the empty "<think>\n\n</think>" block
of the non-thinking template while the ground-truth block does not; it is
stripped from a- so a preference pair contrasts the decision, not the template.
Weights: _task_phat := u_minus so the trainer's (1 - p_hat) equals the event's
dU for single-turn events and (1 - u_minus) for stateful ones; the causal dU is
kept in _event_dU for U x R gating later (B5).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THINK_RE = re.compile(r"^\s*<think>\s*</think>\s*", re.S)


def strip_think(text: str) -> str:
    return THINK_RE.sub("", text or "").strip()


def load(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stateful", default="data/bfcl_sft/events_v1.jsonl")
    parser.add_argument("--single", nargs="+", default=["data/bfcl_sft/events_single_v1.jsonl"],
                        help="one or more single-turn event files (round 2 adds the irrelevance/abstain sets)")
    parser.add_argument("--min-du", type=float, default=0.0, help="keep events with dU > this")
    parser.add_argument("--cap-per-category", type=int, default=80,
                        help="max events per seed category (gen_pool_v3 is 270 simple_python rows; "
                             "uncapped, one axis would dominate the ladder)")
    parser.add_argument("--out-ce", default="data/bfcl_sft/pool_events_ce.jsonl")
    parser.add_argument("--out-pref", default="data/bfcl_sft/pool_events_pref.jsonl")
    args = parser.parse_args()

    events = load(ROOT / args.stateful)
    for single in args.single:
        events += load(ROOT / single)
    kept, dropped, seen = [], 0, set()
    for e in events:
        if float(e.get("_event_dU", 0.0)) <= args.min_du:
            dropped += 1
            continue
        rejected = strip_think(e.get("_rejected", ""))
        response = e["response"].strip()
        if not response or rejected == response:
            dropped += 1
            continue
        key = (e["task_id"], e.get("_event_turn", 0), rejected[:200])
        if key in seen:  # identical (state, wrong action) from another rollout
            dropped += 1
            continue
        seen.add(key)
        kept.append({
            "task_id": e["task_id"],
            "teacher": e.get("teacher", "oracle_gt"),
            "turn_index": int(e.get("_event_turn", e.get("turn_index", 0))),
            "prompt": e["prompt"],
            "response": response,
            "token_hint": max(len(response) // 4, 1),
            "_task_phat": float(e.get("_event_u_minus", 0.0)),
            "_traj": e.get("_traj", e["task_id"]),
            "_seed_category": e.get("_seed_category", ""),
            "_event_dU": float(e.get("_event_dU", 0.0)),
            "_event_u_plus": float(e.get("_event_u_plus", 1.0)),
            "_event_u_minus": float(e.get("_event_u_minus", 0.0)),
            "_rejected": rejected,
        })
    if args.cap_per_category:
        from collections import defaultdict

        kept.sort(key=lambda r: -r["_event_dU"])  # keep the most consequential per category
        per: dict[str, int] = defaultdict(int)
        capped = []
        for r in kept:
            if per[r["_seed_category"]] < args.cap_per_category:
                per[r["_seed_category"]] += 1
                capped.append(r)
        dropped += len(kept) - len(capped)
        kept = capped
    ce_rows = [{k: v for k, v in r.items() if k != "_rejected"} for r in kept]
    for path, rows in ((ROOT / args.out_ce, ce_rows), (ROOT / args.out_pref, kept)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as h:
            for r in rows:
                h.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter

    cats = Counter(r["_seed_category"] for r in kept)
    print(f"[pools] events={len(events)} kept={len(kept)} dropped={dropped} "
          f"stateful={sum(1 for r in kept if r['turn_index'] > 0 or '#ev' in r['_traj'] and 'ev1' not in r['_traj'])}")
    print(f"[pools] categories: {cats.most_common()}")
    print(f"[pools] mean dU={sum(r['_event_dU'] for r in kept) / max(1, len(kept)):.3f} -> {args.out_ce}, {args.out_pref}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
