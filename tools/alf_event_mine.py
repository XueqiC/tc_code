#!/usr/bin/env python3
"""CRCD intervention-event mining for ALFWorld (no teacher API).

For every verified teacher demo (ledger, train split) replay the demo's command prefix in the
environment and, at each step, ask the STUDENT (base, served by vllm under the deployment ReAct
scaffold) what it would do. The first step where the student's command differs from the demo's
is the intervention point: h = replayed prefix, a+ = demo command, a- = student command.
Its utility dU = U(h, a+) - U(h, a-) is estimated with K student continuations from each branch
(ALFWorld is deterministic given the command sequence, so a branch is replayed from scratch).

Event rows use the student's own deployment prompt at h (so training and deployment match):
  response  = the student's reply with its ACTION line replaced by a+   (corrected span)
  _rejected = the student's original reply                              (a-)

    PYTHONPATH=src python tools/alf_event_mine.py --port 8981 --k 3 --out data/alf_sft/events_v1.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bfas.adapters import alfworld as alf  # noqa: E402

os.environ.setdefault("BFAS_ALFWORLD_STUDENT_REACT", "1")


def student_messages(task_id: str, goal: str, state: dict, history: list[str]) -> list[dict]:
    task_type = alf._category(task_id)
    text = alf.TEACHER_REACT_PROMPT.format(
        task_type=task_type,
        example=alf.TEACHER_REACT_EXAMPLES[task_type],
        obs=alf._obs_with_goal(goal, str(state["observation"])),
        cmds="\n".join(state["admissible"]),
        hist="\n".join(history[-8:]) or "(start)",
    )
    return [
        {"role": "system", "content": alf.TEACHER_REACT_INSTRUCTION},
        {"role": "user", "content": text},
    ]


def corrected_reply(student_reply: str, good_command: str) -> str:
    markers = list(alf._ACTION_MARKER_RE.finditer(student_reply))
    if markers:
        return student_reply[: markers[-1].start()] + "ACTION: " + good_command
    return "ACTION: " + good_command


class Runner:
    def __init__(self, adapter: alf.ALFWorldAdapter, temperature: float, max_steps: int):
        self.adapter = adapter
        self.temperature = temperature
        self.max_steps = max_steps
        self.calls = 0

    def open(self, task_id: str):
        bridge = alf._EnvBridge("train", task_id)
        state = bridge._read()
        goal = alf._goal_line(str(state["observation"]))
        return bridge, state, goal

    def replay(self, bridge, state, commands: list[str], history: list[str]):
        """Execute a fixed command prefix; returns (state, done, won)."""
        for command in commands:
            state = bridge.step(command)
            history.append(f"> {command}\n{str(state['observation'])[:300]}")
            if state["done"]:
                return state, True, state["won"] is True
        return state, False, False

    def student_step(self, task_id: str, goal: str, state: dict, history: list[str]):
        messages = student_messages(task_id, goal, state, history)
        prompt = self.adapter._render(messages)
        reply = self.adapter._server_reply(prompt, self.temperature)
        self.calls += 1
        command = self.adapter._teacher_command(reply, state["admissible"])
        return prompt, reply, command

    def continuation(self, task_id: str, prefix: list[str], forced: str) -> bool:
        """Success of one student rollout from (prefix, forced) -- fresh env each time."""
        bridge, state, goal = self.open(task_id)
        try:
            history: list[str] = []
            state, done, won = self.replay(bridge, state, prefix + [forced], history)
            if done:
                return won
            for _ in range(self.max_steps - len(prefix) - 1):
                _, _, command = self.student_step(task_id, goal, state, history)
                state = bridge.step(command)
                history.append(f"> {command}\n{str(state['observation'])[:300]}")
                if state["done"]:
                    return state["won"] is True
            return False
        finally:
            bridge.close()

    def utilities(self, task_id: str, prefix: list[str], good: str, bad: str, k: int) -> tuple[float, float]:
        """K student continuations per branch, all 2K in parallel (each owns an env worker)."""
        from concurrent.futures import ThreadPoolExecutor

        jobs = [good] * k + [bad] * k
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            wins = list(pool.map(lambda forced: self.continuation(task_id, prefix, forced), jobs))
        u_plus = (sum(wins[:k]) + 1) / (k + 2)   # Beta(1,1) posterior mean, as in the BFCL miners
        u_minus = (sum(wins[k:]) + 1) / (k + 2)
        return u_plus, u_minus


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", default="data/teacher_ledger/alfworld.jsonl")
    parser.add_argument("--port", type=int, default=8981)
    parser.add_argument("--k", type=int, default=3, help="continuations per branch")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--max-events-per-demo", type=int, default=2,
                        help="stop a demo after this many CONSEQUENTIAL (dU>0) events")
    parser.add_argument("--max-probes", type=int, default=4,
                        help="max divergences probed per demo (each costs 2K continuations)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--out", default="data/alf_sft/events_v1.jsonl")
    parser.add_argument("--student", default="Qwen/Qwen3.5-4B")
    args = parser.parse_args()

    rows = [json.loads(l) for l in (ROOT / args.ledger).read_text().splitlines() if l.strip()]
    demos = {}
    for r in rows:
        if r.get("verified") and r["task_id"] not in demos:
            demos[r["task_id"]] = [t["target"].strip() for t in r["demo"]["turns"]]
    items = sorted(demos.items())[args.start_index:]
    if args.limit:
        items = items[: args.limit]
    print(f"[alf-events] {len(items)} verified demos, K={args.k}", flush=True)

    adapter = alf.ALFWorldAdapter(seed=0, port=args.port)
    adapter.prepare_renderer(args.student)
    runner = Runner(adapter, args.temperature, args.max_steps)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    handle = out.open("a", encoding="utf-8")
    stats = {"demos": 0, "events": 0, "consequential": 0, "agree_all": 0, "errored": 0}
    t0 = time.time()
    for task_id, commands in items:
        stats["demos"] += 1
        try:
            bridge, state, goal = runner.open(task_id)
            history: list[str] = []
            prefix: list[str] = []
            events_here = 0
            probes = 0
            try:
                for t, good in enumerate(commands):
                    prompt, reply, student_cmd = runner.student_step(task_id, goal, state, history)
                    if student_cmd == good:
                        state, done, _ = runner.replay(bridge, state, [good], history)
                        prefix.append(good)
                        if done:
                            break
                        continue
                    # divergence: branch utilities
                    probes += 1
                    u_plus, u_minus = runner.utilities(task_id, prefix, good, student_cmd, args.k)
                    d_u = u_plus - u_minus
                    stats["events"] += 1
                    if d_u > 0:
                        stats["consequential"] += 1
                        events_here += 1
                    handle.write(json.dumps({
                        "task_id": task_id,
                        "teacher": "demo_replay",
                        "turn_index": t,
                        "prompt": prompt,
                        "response": corrected_reply(reply, good),
                        "_rejected": reply.strip(),
                        "token_hint": max(len(reply) // 4, 8),
                        "_task_phat": u_minus,
                        "_traj": f"{task_id}#p{probes}",
                        "_seed_category": alf._category(task_id),
                        "_event_turn": t,
                        "_event_good": good,
                        "_event_bad": student_cmd,
                        "_event_u_plus": u_plus,
                        "_event_u_minus": u_minus,
                        "_event_dU": d_u,
                        "_event_k": args.k,
                        "_prefix_len": len(prefix),
                    }, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(f"  {task_id[:50]} t={t} good={good!r} bad={student_cmd!r} "
                          f"u+={u_plus:.2f} u-={u_minus:.2f} dU={d_u:+.2f}", flush=True)
                    if events_here >= args.max_events_per_demo or probes >= args.max_probes:
                        break
                    # follow the demo and keep looking for the next divergence
                    state, done, _ = runner.replay(bridge, state, [good], history)
                    prefix.append(good)
                    if done:
                        break
                else:
                    if probes == 0:
                        stats["agree_all"] += 1
            finally:
                bridge.close()
        except Exception as exc:  # noqa: BLE001
            stats["errored"] += 1
            print(f"  {task_id[:50]} ERROR {type(exc).__name__}: {str(exc)[:120]}", flush=True)
        if stats["demos"] % 5 == 0:
            print(f"[alf-events] {stats} calls={runner.calls} elapsed={time.time() - t0:.0f}s", flush=True)
    handle.close()
    print(f"[alf-events] FINAL {stats} calls={runner.calls} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
