#!/usr/bin/env python3
"""Re-estimate dU with more continuations for ALFWorld events whose K=3 estimate was ~0
(user prompt 16.3 / the BFCL zero-bucket finding: the K=3 estimator under-detects long-horizon value).

For each selected event, replay the demo prefix, force a+ / a-, and run K_EXTRA more student
continuations per branch; pooled counts (old K + K_EXTRA) give the refined dU and P(dU>0).
Writes a new events file with _event_dU / _event_u_plus / _event_u_minus / _event_k updated and
_event_dU_k3 keeping the original estimate.

    PYTHONPATH=src python tools/alf_event_reestimate.py --port 8981 --k-extra 5 \
        --events data/alf_sft/events_v1.jsonl --select zero --out data/alf_sft/events_v1_k8.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from bfas.adapters import alfworld as alf  # noqa: E402
from alf_event_mine import Runner  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", default="data/alf_sft/events_v1.jsonl")
    parser.add_argument("--ledger", default="data/teacher_ledger/alfworld.jsonl")
    parser.add_argument("--port", type=int, default=8981)
    parser.add_argument("--k-extra", type=int, default=5)
    parser.add_argument("--select", choices=("zero", "all", "nonpos"), default="zero")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--student", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--out", default="data/alf_sft/events_v1_k8.jsonl")
    args = parser.parse_args()

    demos = {}
    for l in (ROOT / args.ledger).read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("verified") and r["task_id"] not in demos:
            demos[r["task_id"]] = [t["target"].strip() for t in r["demo"]["turns"]]
    rows = [json.loads(l) for l in (ROOT / args.events).read_text().splitlines() if l.strip()]
    sel = []
    for r in rows:
        d = float(r["_event_dU"])
        if args.select == "all" or (args.select == "zero" and d == 0) or (args.select == "nonpos" and d <= 0):
            sel.append(r)
    random.Random(0).shuffle(sel)
    if args.limit:
        sel = sel[: args.limit]
    print(f"[reest] {len(sel)} events selected ({args.select}), K_extra={args.k_extra}", flush=True)
    adapter = alf.ALFWorldAdapter(seed=0, port=args.port)
    adapter.prepare_renderer(args.student)
    runner = Runner(adapter, args.temperature, args.max_steps)
    out = ROOT / args.out
    handle = out.open("a", encoding="utf-8")
    t0 = time.time()
    flipped = 0
    for i, r in enumerate(sel, 1):
        task_id = r["task_id"]
        commands = demos.get(task_id)
        if commands is None:
            continue
        prefix = commands[: int(r["_prefix_len"])]
        good, bad = r["_event_good"], r["_event_bad"]
        k_old = int(r["_event_k"])
        wg_old = round(float(r["_event_u_plus"]) * (k_old + 2) - 1)
        wb_old = round(float(r["_event_u_minus"]) * (k_old + 2) - 1)
        try:
            u_plus_x, u_minus_x = runner.utilities(task_id, prefix, good, bad, args.k_extra)
        except Exception as exc:  # noqa: BLE001
            print(f"  {task_id[:50]} ERROR {type(exc).__name__}: {str(exc)[:100]}", flush=True)
            continue
        wg = wg_old + round(u_plus_x * (args.k_extra + 2) - 1)
        wb = wb_old + round(u_minus_x * (args.k_extra + 2) - 1)
        k = k_old + args.k_extra
        u_plus, u_minus = (wg + 1) / (k + 2), (wb + 1) / (k + 2)
        q = dict(r)
        q["_event_dU_k3"] = float(r["_event_dU"])
        q["_event_u_plus"], q["_event_u_minus"], q["_event_dU"], q["_event_k"] = u_plus, u_minus, u_plus - u_minus, k
        q["_event_wins"] = [wg, wb]
        handle.write(json.dumps(q, ensure_ascii=False) + "\n")
        handle.flush()
        if q["_event_dU"] > 0:
            flipped += 1
        if i % 10 == 0:
            print(f"[reest] {i}/{len(sel)} now-positive={flipped} calls={runner.calls} elapsed={time.time() - t0:.0f}s", flush=True)
    handle.close()
    print(f"[reest] FINAL {len(sel)} re-estimated, now-positive={flipped} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
