#!/usr/bin/env python3
"""Grow the distillation set until no query stands out as suspicious.

How much data to generate is decided by the data, not set by hand. Each seed
query carries a Beta posterior over the student's success on it, built from the
student's rollouts and from every variation generated off it. Suspicion is the
upper confidence bound on failure -- high when the student does badly AND when
too little has been observed to be sure. Each round spends on the single most
suspicious query, then re-measures.

Two reasons a query can look unremarkable, and they must not be confused: it is
genuinely average, or nobody has looked at it hard enough to tell. An earlier
version of this loop stopped as soon as the top query's interval covered the
pooled rate, which the LEAST observed points satisfy most easily -- it halted
exactly where the evidence was thinnest. So each round now asks:

  confidently worse than the pool   spend the teacher: generate variations here
  merely uncertain                  spend rollouts instead -- free, and what is
                                    missing is observation, not data
  neither                           resolved; stop considering it

Spending saturates on information rather than on outcome. A genuinely hard
query stays confidently worse however much data is bought for it, so a rule
that keeps buying until the student succeeds pours everything into one point:
in an earlier run 89% of the budget went to a single query while a confirmed
capability gap received eight items. The loop is here to COVER the boundary,
not to cure each point. So generation for a query stops once a fresh batch
moves its estimate by less than the estimate's own uncertainty -- at that point
the query is characterised, we know it is hard and we hold data for it, and the
budget moves to the next suspicious query.

"Merely uncertain" is judged against the other points' dispersion, not a fixed
number: a point whose posterior is wider than the typical point's has not been
looked at as hard as the rest. The loop ends when nothing is left unresolved,
so the number of generated items is an outcome of the run.

`--max-items` is a spend guard, not part of the rule; hitting it is reported as
an interruption rather than as convergence.

Needs a vLLM server already holding the student (see bfcl_gen_lane.sh).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BFCL = ROOT / "envs/bfcl/gorilla/berkeley-function-call-leaderboard"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(BFCL))

CHECKER_MODEL = os.environ.get("BFAS_BFCL_CHECKER_MODEL", "Qwen/Qwen3.5-4B-FC")
CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def student_calls(prompt: str, port: int, temperature: float,
                  max_tokens: int = 512) -> list[dict]:
    request = urllib.request.Request(
        f"http://localhost:{port}/v1/completions",
        data=json.dumps({
            "model": "Qwen/Qwen3.5-4B",
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        text = json.loads(response.read())["choices"][0]["text"]
    calls = []
    for blob in CALL_RE.findall(text):
        try:
            call = json.loads(blob)
        except json.JSONDecodeError:
            continue
        name = call.get("name")
        if name:
            calls.append({name: call.get("arguments", {})})
    return calls


def scores(item: dict, adapter, port: int, rounds: int,
           temperature: float) -> tuple[int, int]:
    """Roll the student on one generated item; return (successes, attempts)."""
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
    from bfcl_eval.constants.enums import Language

    prompt = adapter._render(item["question"][0], item["function"])
    category = item.get("seed_category", "")
    good = 0
    for _ in range(rounds):
        try:
            output = student_calls(prompt, port, temperature)
        except Exception:
            continue
        if not output:
            continue
        try:
            verdict = ast_checker(item["function"], output,
                                  item["ground_truth"], Language.PYTHON,
                                  category, CHECKER_MODEL)
        except Exception:
            continue
        good += bool(verdict.get("valid"))
    return good, rounds


class Point:
    """One seed query's posterior over the student's success on it."""

    def __init__(self, task_id: str, good: float, total: float):
        self.task_id = task_id
        self.good = good
        self.total = total
        self.generated = 0
        self.items: list[dict] = []
        self.exhausted = False   # the teacher cannot produce variations here
        self.characterised = False  # further generation here adds no information
        self.measured_out = False   # further observation here adds no information

    @property
    def alpha(self) -> float:
        return 1.0 + self.good

    @property
    def beta(self) -> float:
        return 1.0 + (self.total - self.good)

    @property
    def fail_mean(self) -> float:
        return self.beta / (self.alpha + self.beta)

    @property
    def fail_sd(self) -> float:
        a, b = self.alpha, self.beta
        return float(np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1))))

    @property
    def suspicion(self) -> float:
        return self.fail_mean + self.fail_sd

    def interval(self, z: float = 1.645) -> tuple[float, float]:
        return (self.fail_mean - z * self.fail_sd,
                self.fail_mean + z * self.fail_sd)

    def observe(self, good: int, total: int) -> None:
        self.good += good
        self.total += total


def seed_as_item(task_id: str, entries: dict, official: dict,
                 categories: dict) -> dict | None:
    entry = entries.get(task_id)
    truth = official.get(task_id)
    if entry is None or truth is None or "function" not in entry:
        return None
    return {
        "id": task_id,
        "question": entry["question"],
        "function": entry["function"],
        "ground_truth": truth,
        "seed_category": categories.get(task_id, ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8971)
    parser.add_argument("--per-round", type=int, default=6)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-items", type=int, default=400,
                        help="spend guard, not part of the stopping rule")
    parser.add_argument("--out", default="data/bfcl_sft/gen_pool.jsonl")
    args = parser.parse_args()

    from bfas.adapters.bfcl import BFCLAdapter

    adapter = BFCLAdapter()
    entries, categories = adapter._load_entries()
    official: dict[str, list] = {}
    for path in (BFCL / "bfcl_eval/data/possible_answer").glob("BFCL_v4_*.json"):
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    row = json.loads(line)
                    official[row["id"]] = row["ground_truth"]
                except (json.JSONDecodeError, KeyError):
                    continue
    phat = json.loads((ROOT / "results/analysis/phat_v4.json").read_text())
    rounds_seen = 4  # rollouts behind each phat measurement
    points = {t: Point(t, p * rounds_seen, rounds_seen) for t, p in phat.items()}

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    handle = out_path.open("a")
    produced = 0
    history: list[dict] = []

    def pooled_fail() -> float:
        """The typical POINT's failure rate, not the typical observation's.

        Averaging over observations lets the investigation move its own
        baseline: a hard point that earns many rollouts contributes most of the
        failures, the bar rises, and every other point is then declared no
        worse than the pool. Averaging over points keeps the comparison fixed
        while the loop spends unevenly, which is the whole point of spending
        unevenly.
        """
        live_points = list(points.values())
        return (float(np.mean([p.fail_mean for p in live_points]))
                if live_points else 0.0)

    resolved: list[str] = []
    while produced < args.max_items:
        live = [p for p in points.values() if p.task_id not in resolved]
        if not live:
            print("[loop] STOP — every query is resolved: each is either "
                  "no worse than the pool, or as well observed as the rest "
                  "and not confidently worse.")
            break
        ranked = sorted(live, key=lambda p: -p.suspicion)
        top = ranked[0]
        pool = pooled_fail()
        low, high = top.interval()

        # The action is read straight off the posterior, and observation is
        # bought only while it could still change that action:
        #   interval above the pool   -> confidently worse: generate here
        #   interval below the pool   -> no worse than the pool: resolved
        #   interval straddles it     -> the action is undetermined; one more
        #                                look is worth exactly as much as it
        #                                narrows this interval -- observe,
        #                                until the posterior stops moving
        # No fixed rollout count and no dispersion heuristic: measurement is
        # allocated by the same rule that allocates the teacher.
        if low > pool and not top.exhausted and not top.characterised:
            action = "generate"
        elif high < pool or top.measured_out:
            resolved.append(top.task_id)
            print(f"[loop] resolved {top.task_id} "
                  f"(fail {top.fail_mean:.2f}±{top.fail_sd:.2f}, pool {pool:.2f})")
            continue
        elif low > pool:
            # confidently worse but nothing left to buy or learn here
            resolved.append(top.task_id)
            print(f"[loop] resolved {top.task_id} (hard, characterised; "
                  f"fail {top.fail_mean:.2f}±{top.fail_sd:.2f})")
            continue
        else:
            action = "observe"

        print(f"\n[loop] round {len(history) + 1}: {action} on {top.task_id} "
              f"(fail {top.fail_mean:.2f}±{top.fail_sd:.2f}, pool {pool:.2f})")

        if action == "observe":
            # more rollouts on what this point already has: the gap is
            # observation, and observation costs no teacher quota
            if not top.items:
                seed_item = seed_as_item(top.task_id, entries, official, categories)
                if seed_item is None:
                    resolved.append(top.task_id)
                    print("[loop]   nothing rollable here; resolved")
                    continue
                top.items = [seed_item]
            good = attempts = 0
            before_mean, before_sd = top.fail_mean, top.fail_sd
            for item in top.items:
                g, a = scores(item, adapter, args.port, args.rollouts,
                              args.temperature)
                good += g
                attempts += a
            top.observe(good, attempts)
            if abs(top.fail_mean - before_mean) < before_sd:
                # the batch landed inside what the posterior already expected:
                # more looking will not move the estimate, so the straddle is
                # as resolved as observation can make it
                top.measured_out = True
            print(f"[loop]   student {good}/{attempts}; "
                  f"fail now {top.fail_mean:.2f}±{top.fail_sd:.2f}"
                  + ("; measured out" if top.measured_out else ""))
            history.append({"target": top.task_id, "action": "observe",
                            "student_good": good, "student_attempts": attempts})
            continue

        staged = ROOT / "data/bfcl_sft/_gen_round.jsonl"
        result = subprocess.run(
            [str(ROOT / "envs/bfcl/.venv/bin/python"),
             str(ROOT / "tools/bfcl_generate.py"),
             "--per-seed", str(args.per_round),
             "--only", top.task_id,
             "--out", str(staged.relative_to(ROOT))],
            cwd=ROOT, env={**os.environ, "PYTHONPATH": "src"},
            capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[loop] generation failed: {result.stderr[-300:]}")
            break

        items = [json.loads(l) for l in staged.read_text().splitlines()
                 if l.strip()]
        usable = [i for i in items if i.get("reproduced")]
        print(f"[loop]   drafted {len(items)}, reproduced {len(usable)}")
        if not usable:
            # the teacher cannot make variations here; keep the point in play
            # for observation rather than dropping what may be a real gap
            top.exhausted = True
            history.append({"target": top.task_id, "action": "generate",
                            "drafted": len(items), "usable": 0})
            print("[loop]   no reproducible variation; will observe instead")
            continue

        good_total = attempts_total = 0
        for item in usable:
            good, attempts = scores(item, adapter, args.port,
                                    args.rollouts, args.temperature)
            good_total += good
            attempts_total += attempts
            item["student_good"] = good
            item["student_attempts"] = attempts
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            produced += 1
        handle.flush()
        before_mean, before_sd = top.fail_mean, top.fail_sd
        top.items.extend(usable)
        top.observe(good_total, attempts_total)
        top.generated += len(usable)
        moved = abs(top.fail_mean - before_mean)
        if moved < before_sd:
            top.characterised = True
        print(f"[loop]   student {good_total}/{attempts_total} on the new "
              f"items; fail now {top.fail_mean:.2f}±{top.fail_sd:.2f} "
              f"(moved {moved:.3f} vs sd {before_sd:.3f}"
              + ("; characterised, moving on)" if top.characterised else ")"))
        history.append({"target": top.task_id, "action": "generate",
                        "drafted": len(items), "usable": len(usable),
                        "student_good": good_total,
                        "student_attempts": attempts_total})
    else:
        print(f"[loop] INTERRUPTED by the --max-items spend guard at "
              f"{produced} items; this is not convergence.")

    handle.close()
    summary = ROOT / "results/analysis/gen_loop.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps({
        "produced": produced,
        "rounds": history,
        "resolved": resolved,
        "characterised": [t for t, p in points.items() if p.characterised],
        "points": {t: {"fail_mean": p.fail_mean, "fail_sd": p.fail_sd,
                       "generated": p.generated, "observations": p.total}
                   for t, p in points.items()},
    }, indent=1))
    print(f"\n[loop] {produced} items -> {out_path}; trace -> {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
