#!/usr/bin/env python3
"""Turn verified out-of-scope items into trainer rows.

The correct behaviour on these is to answer without calling anything, so the
training target is what the assistant says when it declines. The wording is
written by the teacher rather than filled into a template: a fixed phrase would
teach the model one sentence instead of the judgement behind it, and BFCL's
irrelevance checker looks only at whether a call was made, so the text is free
to be natural.

Rows carry _task_phat = 0 -- the student currently calls on these, which is the
failure they exist to correct.

Usage: bfcl_oos_rows.py [--in data/bfcl_sft/gen_oos.jsonl]
                        [--out data/bfcl_sft/pool_gen_oos.jsonl]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BFCL = ROOT / "envs/bfcl/gorilla/berkeley-function-call-leaderboard"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(BFCL))

REPLY_PROMPT = """A user asked an assistant this:

{query}

The assistant has only these functions available:
{names}

None of them can do what was asked: {why}

Write what the assistant should say back. Tell the user plainly that this is
not something you can do here, say briefly why, and where it is natural offer
what the available functions COULD do instead. Two or three sentences, no
apology theatre, no invented capability.

Reply with the assistant's message only, no quotes, no preamble."""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="src", default="data/bfcl_sft/gen_oos.jsonl")
    parser.add_argument("--out", default="data/bfcl_sft/pool_gen_oos.jsonl")
    args = parser.parse_args()

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gen", str(ROOT / "tools/bfcl_generate.py"))
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)

    from bfas.adapters.bfcl import BFCLAdapter

    adapter = BFCLAdapter()
    key = gen.api_key()
    rows = []
    skipped = 0
    for line in (ROOT / args.src).read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if not item.get("verified"):
            continue
        query = item["question"][0][0]["content"]
        names = ", ".join(
            str(s.get("name")) for s in item["function"] if isinstance(s, dict))
        try:
            reply = gen.ask(REPLY_PROMPT.format(
                query=query, names=names, why=item.get("why", "")), 0.7, key)
        except RuntimeError as exc:
            print(f"[oosrows] teacher stopped: {exc}")
            break
        reply = reply.strip()
        # a decline that contains a call would train the opposite of the point
        if not reply or "<tool_call>" in reply:
            skipped += 1
            continue
        rows.append({
            "task_id": item["id"],
            "teacher": "gen",
            "turn_index": 0,
            "prompt": adapter._render(item["question"][0], item["function"]),
            "response": reply,
            "token_hint": max(len(reply) // 4, 1),
            "_task_phat": 0.0,
            "_traj": f"{item['id']}#oos",
            "_seed_task": item["seed_task"],
            "_seed_category": item.get("seed_category", ""),
            "_out_of_scope": True,
        })
        print(f"    {query[:56]}\n      -> {reply[:80]}")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n[oosrows] {len(rows)} decline rows ({skipped} skipped) "
          f"-> {out_path}")
    if not rows:
        print("[oosrows] WARNING: no decline rows produced -- the pool will "
              "again contain no example of not calling")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
