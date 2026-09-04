#!/usr/bin/env python3
"""Build BFCL training pools and a compact demo library from verified
teacher demonstrations.

Outputs
- data/bfcl_sft/pool_bfcl_<tag>_sft.jsonl: trainer rows for the
  distillation baselines, one row per verified single-turn demo. The
  training context replicates the exact prompting-handler messages the
  student sees at evaluation time (system prompt with function docs
  injected by the official helper, then the user turn).
- data/bfcl_sft/demos_<tag>.json: compact worked examples per verified
  demand task (question plus the ordered function calls the teacher
  issued), used as guidance context for guided rollouts. Multi-turn
  demos appear here even though they are excluded from the SFT rows.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BFCL = ROOT / "envs/bfcl/gorilla/berkeley-function-call-leaderboard"
sys.path.insert(0, str(BFCL))

MULTI_TURN_PREFIXES = ("multi_turn", "web_search", "memory")


def serialize(messages: list[dict]) -> str:
    return "\n".join(
        f"<|{m['role']}|>\n{m['content']}" for m in messages
    ) + "\n<|assistant|>\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result-dir", default="result_demos_deepseek_v4_pro_FC")
    ap.add_argument("--verified", default="data/bfcl_demos_ds_verified.json")
    ap.add_argument("--tag", default="ds")
    args = ap.parse_args()

    from bfcl_eval.utils import load_file  # noqa: F401 (env sanity)
    from bfcl_eval.model_handler.utils import (
        system_prompt_pre_processing_chat_model,
    )

    verified = set(json.load((ROOT / args.verified).open()))

    # entries come from the adapter: memory tasks only exist under their
    # per-backend ids once the group is expanded, so data-file membership
    # would drop them here without a word
    sys.path.insert(0, str(ROOT / "src"))
    from bfas.adapters.bfcl import BFCLAdapter

    all_entries = BFCLAdapter()._load_entries()[0]
    entries: dict[str, dict] = {
        task_id: row for task_id, row in all_entries.items() if task_id in verified
    }
    missing = verified - set(entries)
    if missing:
        print(f"[bfcldemopool] WARNING {len(missing)} verified ids have no "
              f"entry: {sorted(missing)[:5]}")

    results: dict[str, dict] = {}
    for f in (BFCL / args.result_dir).rglob("*_result.json"):
        for line in f.open():
            if line.strip():
                e = json.loads(line)
                if e["id"] in verified:
                    results[e["id"]] = e

    out_dir = ROOT / "data/bfcl_sft"
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_path = out_dir / f"pool_bfcl_{args.tag}_sft.jsonl"
    demo_path = out_dir / f"demos_{args.tag}.json"

    rows, demos, skipped_mt = [], {}, 0
    for tid in sorted(verified):
        entry = entries.get(tid)
        result = results.get(tid)
        if entry is None or result is None:
            continue
        is_multi = tid.startswith(MULTI_TURN_PREFIXES)
        # compact worked example for guidance context
        question = entry["question"][0]
        user_texts = [m["content"] for m in question if m["role"] == "user"]
        demos[tid] = {
            "question": user_texts[0] if user_texts else "",
            "teacher_result": result["result"],
        }
        if is_multi:
            skipped_mt += 1
            continue
        # replicate the evaluation-time prompt of the prompting handler
        e2 = copy.deepcopy(entry)
        e2["question"][0] = system_prompt_pre_processing_chat_model(
            e2["question"][0], e2["function"], tid
        )
        messages = list(e2["question"][0])
        answer = result["result"]
        if not isinstance(answer, str):
            answer = json.dumps(answer, ensure_ascii=False)
        rows.append({
            "task_id": tid,
            "teacher": args.tag,
            "turn_index": 0,
            "messages": messages,
            "prompt": serialize(messages),
            "response": answer,
            "token_hint": max(len(answer) // 4, 1),
        })

    with pool_path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(demos, demo_path.open("w"), ensure_ascii=False, indent=1)
    print(f"[bfclpool] sft rows={len(rows)} demos={len(demos)} "
          f"(multi-turn kept as demos only: {skipped_mt}) -> {pool_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
