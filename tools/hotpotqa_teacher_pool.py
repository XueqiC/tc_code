#!/usr/bin/env python3
"""Collect a resumable, capped HotpotQA support pool without a student or GPU."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bfas import hotpotqa as hp
from bfas.adapters.hotpotqa import HotpotQAAdapter
from bfas.hotpotqa_budget import Budget, Limits, TEACHER_MAX_TOKENS
from bfas.ledger import append_episode, append_record, read_records, _minimum_interval
from bfas.protocol import TEACHER_ATTEMPTS, SAMPLING_TEMPERATURE


class PoolAdapter(HotpotQAAdapter):
    def _render(self, messages):
        # Portable archive; _import_pool applies the deployment tokenizer later.
        return "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)


def collection_identity(teacher, limits):
    return {"schema_version": 1, "teacher": teacher,
            "service_tier": os.environ.get("BFAS_OPENAI_SERVICE_TIER", "flex") or None,
            "endpoint_overrides": {key: os.environ.get(key) for key in
                                   ("OPENAI_BASE_URL", "OPENROUTER_BASE_URL", "OLLAMA_BASE_URL", "AZURE_LLM_ENDPOINT")},
            "prompt_version": hp.PROMPT_VERSION, "wiki_version": hp.WIKI_VERSION,
            "max_steps": hp.MAX_STEPS, "max_completion_tokens": TEACHER_MAX_TOKENS,
            "support": hp.load_manifest("train"), "attempts": TEACHER_ATTEMPTS,
            "prices": [str(limits.usd_in), str(limits.usd_out), str(limits.usd_cached)],
            "rendering": "portable messages; deployment tokenizer applied on BFAS import",
            "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
                "tools/hotpotqa_teacher_pool.py", "src/bfas/hotpotqa.py",
                "src/bfas/hotpotqa_budget.py", "src/bfas/adapters/hotpotqa.py", "src/appworld_teacher.py")}}


def append_attempt(path, question, row, episode=None):
    """Archive all outcomes separately from the verified-only demo payloads."""
    trajectory = episode.raw if episode is not None else None
    if trajectory is None:
        # A durable purchase without an archive must never trigger a re-purchase.
        # Missing output cannot be reconstructed from token accounting.
        trajectory = {"question": question["question"], "gold": question["answer"],
                      "category": question["type"], "prediction": None, "em": None, "f1": None,
                      "steps": None, "history": [], "transcript": None, "finished": None,
                      "error": "Trajectory was not saved before interruption.",
                      "termination_reason": "interrupted_before_trajectory_saved"}
    record = dict(trajectory, **{key: row[key] for key in (
        "task_id", "attempt_index", "teacher", "temperature", "verified", "timestamp",
        "tokens_spent", "usage", "usage_status", "prompt_version")})
    record.update(schema_version=1, trajectory_available=episode is not None and episode.raw is not None,
                  response_texts=list(episode.response_texts) if episode is not None else [],
                  tokens=row["usage"]["prompt_tokens"] + row["usage"]["completion_tokens"])
    append_record(path, record)


def reconcile_attempts(path, questions, rows):
    """Validate unique audit keys and fill crash gaps without replaying requests."""
    purchased = {(r["task_id"], r["attempt_index"]): r for r in rows}
    archived = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = row["task_id"], row["attempt_index"]
            if key in archived or key not in purchased:
                raise ValueError("attempt archive contains duplicate or unpurchased attempts")
            if any(row[field] != purchased[key][field] for field in (
                    "teacher", "prompt_version", "verified", "usage", "usage_status", "tokens_spent")):
                raise ValueError("attempt archive and purchase ledger disagree")
            archived.add(key)
    for key, row in purchased.items():
        if key not in archived:
            append_attempt(path, questions[key[0]], row)


def append_result(path, teacher, task_id, attempt, budget, question, episode=None):
    usage = budget.usage(task_id, attempt)
    row = append_episode(path, task_id=task_id, teacher=teacher, attempt_index=attempt,
                         temperature=0.0 if attempt == 0 else SAMPLING_TEMPERATURE,
                         verified=episode.verified if episode else False,
                         demo=episode.demo if episode else None, tokens_spent=usage["completion_tokens"],
                         usage=usage, usage_status=budget.status(task_id, attempt),
                         prompt_version=hp.PROMPT_VERSION)
    append_attempt(path.with_name("attempts.jsonl"), question, row, episode)
    return row


def collect(out, *, workers=1, limits=None, offline=False, adapter_factory=PoolAdapter):
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    limits = limits or Limits()
    os.environ.setdefault("BFAS_OPENAI_SERVICE_TIER", "flex")
    teacher = HotpotQAAdapter.teacher_name()
    questions = {q["_id"]: q for q in hp.load_questions("train")}
    ids = hp.load_manifest("train")["ids"]
    if set(questions) != set(ids):
        raise ValueError("support question inventory mismatch")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with hp.file_lock(out / "collection.lock"):
        identity = collection_identity(teacher, limits)
        identity_path = out / "identity.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != identity:
                raise ValueError("resume teacher, support, code, or prices changed; use a new --out")
        elif any(out.glob("*.jsonl")):
            raise ValueError("refusing to adopt unbound collection accounting")
        else:
            hp.write_json(identity_path, identity)
        path = out / "teacher_ledger.jsonl"
        budget = Budget(out / "usage.jsonl", limits)
        previous = {}
        for row in read_records(path):
            key = row["task_id"], row["attempt_index"]
            if (key in previous or key[0] not in ids or not 0 <= key[1] < TEACHER_ATTEMPTS
                    or row["teacher"] != teacher or row.get("prompt_version") != hp.PROMPT_VERSION):
                raise ValueError("ledger contains duplicate, outside, or incompatible attempts")
            if (row["usage"] != budget.usage(*key) or row["usage_status"] != budget.status(*key)
                    or row["tokens_spent"] != row["usage"]["completion_tokens"]):
                raise ValueError("ledger and durable request accounting disagree")
            previous[key] = row
        attempts_path = out / "attempts.jsonl"
        reconcile_attempts(attempts_path, questions, previous.values())
        for call in budget.calls.values():
            key = call["task_id"], call["attempt_index"]
            if key[0] not in ids or not 0 <= key[1] < TEACHER_ATTEMPTS:
                raise ValueError("outside request reservation")
            if key not in previous:
                # Crash between response and episode append: charged failed attempt.
                previous[key] = append_result(path, teacher, *key, budget, questions[key[0]])
        interval = _minimum_interval()
        gate, next_start = threading.Lock(), [0.0]
        reasons = []

        def work(task_id):
            prior = [r for (tid, _), r in previous.items() if tid == task_id]
            if any(r["verified"] for r in prior) or budget.exhausted(task_id):
                return
            indices = sorted(r["attempt_index"] for r in prior)
            if indices != list(range(len(indices))):
                raise ValueError("teacher attempts are not contiguous")
            adapter = adapter_factory(offline=offline)
            adapter._questions["train"] = questions
            for attempt in range(len(prior), TEACHER_ATTEMPTS):
                if budget.stopped:
                    return
                with gate:
                    delay = max(0.0, next_start[0] - time.monotonic())
                    if delay:
                        time.sleep(delay)
                    next_start[0] = time.monotonic() + interval
                try:
                    episode = adapter.teacher_episode(task_id, attempt,
                                                      0.0 if attempt == 0 else SAMPLING_TEMPERATURE,
                                                      budget=budget)
                    if (episode.task_id != task_id or episode.teacher != teacher
                            or episode.verified != (episode.demo is not None)):
                        raise ValueError("invalid teacher episode identity/verdict")
                    if not budget.usage(task_id, attempt)["prompt_tokens"]:
                        with gate:
                            reasons.append("hard cap cannot reserve the next request")
                        return  # No request: do not consume an attempt.
                    append_result(path, teacher, task_id, attempt, budget, questions[task_id], episode)
                    if episode.verified or budget.exhausted(task_id):
                        return
                except BaseException as exc:
                    budget.cancel(f"collection stopped: {type(exc).__name__}: {exc}")
                    raise

        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="hotpotqa-pool")
        try:
            futures = [executor.submit(work, tid) for tid in ids]
            for future in as_completed(futures):
                future.result()
        except BaseException:
            budget.cancel("interrupted collection; in-flight requests will be accounted")
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            # Executor shutdown finishes in-flight calls before reconciliation.
            rows = read_records(path)
            seen = {(r["task_id"], r["attempt_index"]) for r in rows}
            for call in budget.calls.values():
                key = call["task_id"], call["attempt_index"]
                if key not in seen:
                    append_result(path, teacher, *key, budget, questions[key[0]])
                    seen.add(key)
            rows = read_records(path)
            reconcile_attempts(attempts_path, questions, rows)
            verified = {r["task_id"]: r["demo"] for r in rows if r["verified"]}
            hp.write_json(out / "demos.json", {"schema_version": 1, "prompt_version": hp.PROMPT_VERSION,
                                              "teacher": teacher, "demos": verified})
            usage = budget.usage()
            attempted = {r["task_id"] for r in rows}
            complete = all(tid in verified or budget.exhausted(tid)
                           or sum(r["task_id"] == tid for r in rows) == TEACHER_ATTEMPTS for tid in ids)
            summary = {"teacher": teacher, "tasks": len(ids), "tasks_attempted": len(attempted),
                       "verified": len(verified), "attempts": len(rows), **usage,
                       "tokens": usage["prompt_tokens"] + usage["completion_tokens"],
                       "estimated_usd": float(limits.cost(usage)),
                       "uncertain_calls": sum(r["status"] != "reported" for r in budget.calls.values()),
                       "complete": complete, "stop_reason": budget.stopped or (reasons[0] if reasons else "complete"),
                       "limits": {"max_tokens": limits.max_tokens, "max_usd": str(limits.max_usd)},
                       "ledger": str(path), "attempts_path": str(attempts_path)}
            hp.write_json(out / "summary.json", summary)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "envs/hotpotqa/teacher_pool")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=400000)
    parser.add_argument("--max-usd", type=Decimal, default=Decimal("5"))
    parser.add_argument("--usd-per-mtok-in", type=Decimal, default=Decimal("0.10"))
    parser.add_argument("--usd-per-mtok-out", type=Decimal, default=Decimal("0.60"))
    parser.add_argument("--usd-per-mtok-cached", type=Decimal, default=Decimal("0.01"))
    args = parser.parse_args(argv)
    try:
        limits = Limits(args.max_tokens, args.max_usd, args.usd_per_mtok_in,
                        args.usd_per_mtok_out, args.usd_per_mtok_cached)
        summary = collect(args.out, workers=args.workers, limits=limits, offline=args.offline)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, str(exc) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
