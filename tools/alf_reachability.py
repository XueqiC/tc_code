#!/usr/bin/env python3
"""Deployment reachability of ALFWorld intervention events (user 2026-09-03).

For every train game with mined events, run N free student rollouts (deployment ReAct scaffold, no
forcing) and record the command sequences. An event at depth t on demo prefix p[0:t] is reachable
by a rollout if the rollout's first t commands equal the prefix exactly (ALFWorld is deterministic,
so equal commands => same state). Writes per-event rows with
  _reach     = fraction of rollouts matching the prefix (0..1)
  _reach_soft = fraction whose first t commands match the prefix as a multiset (order-insensitive)
  _edit      = 1 - u_minus   (how far the student is from the corrected action: editability proxy)
  _value     = _reach * _edit * max(dU, 0)
and a summary by probe index / depth.

    PYTHONPATH=src python tools/alf_reachability.py --port 8983 --n 4 --events data/alf_sft/events_v1.jsonl \
        --out data/alf_sft/event_value_v1.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from bfas.adapters import alfworld as alf  # noqa: E402
from alf_event_mine import Runner  # noqa: E402


def free_rollout(runner: Runner, task_id: str, max_steps: int) -> list[str]:
    bridge, state, goal = runner.open(task_id)
    cmds: list[str] = []
    try:
        history: list[str] = []
        for _ in range(max_steps):
            _, _, command = runner.student_step(task_id, goal, state, history)
            cmds.append(command)
            state = bridge.step(command)
            history.append(f"> {command}\n{str(state['observation'])[:300]}")
            if state["done"]:
                break
    finally:
        bridge.close()
    return cmds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", default="data/alf_sft/events_v1.jsonl")
    parser.add_argument("--ledger", default="data/teacher_ledger/alfworld.jsonl")
    parser.add_argument("--port", type=int, default=8983)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--student", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--out", default="data/alf_sft/event_value_v1.jsonl")
    args = parser.parse_args()

    demos = {}
    for l in (ROOT / args.ledger).read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            if r.get("verified") and r["task_id"] not in demos:
                demos[r["task_id"]] = [t["target"].strip() for t in r["demo"]["turns"]]
    events = [json.loads(l) for l in (ROOT / args.events).read_text().splitlines() if l.strip()]
    games = sorted({e["task_id"] for e in events})
    print(f"[reach] {len(games)} games, {len(events)} events, N={args.n}", flush=True)
    adapter = alf.ALFWorldAdapter(seed=0, port=args.port)
    adapter.prepare_renderer(args.student)
    runner = Runner(adapter, args.temperature, args.max_steps)
    t0 = time.time()
    rollouts: dict[str, list[list[str]]] = {}
    for i, g in enumerate(games, 1):
        with ThreadPoolExecutor(max_workers=args.n) as pool:
            rollouts[g] = list(pool.map(lambda _: free_rollout(runner, g, args.max_steps), range(args.n)))
        if i % 10 == 0:
            print(f"[reach] {i}/{len(games)} games, calls={runner.calls}, elapsed={time.time() - t0:.0f}s", flush=True)
    out = ROOT / args.out
    by_probe: dict[str, list[float]] = defaultdict(list)
    by_depth: dict[int, list[float]] = defaultdict(list)
    with out.open("w", encoding="utf-8") as handle:
        for e in events:
            t = int(e["_prefix_len"])
            prefix = demos[e["task_id"]][:t]
            rs = rollouts[e["task_id"]]
            exact = sum(1 for r in rs if r[:t] == prefix) / len(rs)
            soft = sum(1 for r in rs if Counter(r[:t]) == Counter(prefix)) / len(rs)
            edit = 1.0 - float(e["_event_u_minus"])
            d_u = float(e["_event_dU"])
            q = dict(e)
            q.update({"_reach": exact, "_reach_soft": soft, "_edit": edit, "_value": exact * edit * max(d_u, 0.0)})
            handle.write(json.dumps(q, ensure_ascii=False) + "\n")
            by_probe[str(e.get("_traj", "")).split("#")[-1]].append(exact)
            by_depth[min(t, 5)].append(exact)
    print("[reach] mean exact reachability by probe: " + ", ".join(f"{k}={sum(v)/len(v):.2f} (n={len(v)})" for k, v in sorted(by_probe.items())), flush=True)
    print("[reach] mean exact reachability by depth: " + ", ".join(f"{k}={sum(v)/len(v):.2f} (n={len(v)})" for k, v in sorted(by_depth.items())), flush=True)
    cons = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    cons = [e for e in cons if float(e["_event_dU"]) > 0]
    print(f"[reach] consequential events: {len(cons)}, mean reach {sum(e['_reach'] for e in cons)/max(len(cons),1):.2f}, "
          f"with reach>0: {sum(1 for e in cons if e['_reach'] > 0)} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
