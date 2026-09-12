"""Durable request reservations for capped, concurrent HotpotQA acquisition."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import json
from pathlib import Path
import threading

from .ledger import append_record

TEACHER_MAX_TOKENS = 2048  # Includes hidden reasoning, unlike visible student tokens.


class BudgetStopped(RuntimeError):
    pass


def prompt_bound(messages):
    # Conservative text-token upper bound, including chat framing.
    return 256 + sum(64 + len(m["role"].encode()) + len(m["content"].encode()) for m in messages)


def reported_usage(usage):
    result = {}
    for name in ("prompt_tokens", "completion_tokens"):
        value = usage.get(name)
        if type(value) is not int or value < 0:
            raise ValueError("teacher did not report exact prompt/completion usage")
        result[name] = value
    details = usage.get("prompt_tokens_details") or {}
    if not isinstance(details, dict):
        raise ValueError("teacher reported invalid prompt-token details")
    cached = details.get("cached_tokens", 0)
    if type(cached) is not int or not 0 <= cached <= result["prompt_tokens"]:
        raise ValueError("teacher reported invalid cached-token usage")
    result["cached_tokens"] = cached
    return result


@dataclass(frozen=True)
class Limits:
    max_tokens: int = 400000
    max_usd: Decimal = Decimal("5")
    usd_in: Decimal = Decimal("0.10")
    usd_out: Decimal = Decimal("0.60")
    usd_cached: Decimal = Decimal("0.01")

    def __post_init__(self):
        if type(self.max_tokens) is not int or self.max_tokens < 0:
            raise ValueError("max_tokens must be a nonnegative integer")
        for name in ("max_usd", "usd_in", "usd_out", "usd_cached"):
            value = Decimal(str(getattr(self, name)))
            if not value.is_finite() or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        if self.usd_cached > self.usd_in:
            raise ValueError("cached input rate exceeds input rate")

    def cost(self, usage):
        return ((usage["prompt_tokens"] - usage["cached_tokens"]) * self.usd_in
                + usage["cached_tokens"] * self.usd_cached
                + usage["completion_tokens"] * self.usd_out) / Decimal(1000000)


class Budget:
    """An output-level collection lock excludes other processes for our lifetime.

    The journal is append-only. On replay, the latest row per call wins; an
    unfinished call retains its reservation and stops new purchases on resume.
    """
    def __init__(self, path, limits):
        self.path, self.limits = Path(path), limits
        self.condition = threading.Condition()
        self.calls, self.pending = {}, set()
        self.stopped = None
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                row = json.loads(line)
                self.calls[row["id"]] = row
        if any(r["status"] != "reported" for r in self.calls.values()):
            self.stopped = "unresolved request usage; reservations retained, no further purchases"

    def usage(self, task_id=None, attempt_index=None):
        with self.condition:
            rows = [r for r in self.calls.values() if task_id is None or
                    (r["task_id"], r["attempt_index"]) == (task_id, attempt_index)]
            return {key: sum(r["usage"][key] for r in rows)
                    for key in ("prompt_tokens", "completion_tokens", "cached_tokens")}

    def status(self, task_id, attempt_index):
        with self.condition:
            return "reported" if all(r["status"] == "reported" for r in self.calls.values()
                                     if (r["task_id"], r["attempt_index"]) == (task_id, attempt_index)) else "estimated"

    def cancel(self, reason):
        with self.condition:
            self.stopped = self.stopped or reason
            self.condition.notify_all()

    def reserve(self, task_id, attempt_index, messages):
        reserve = {"prompt_tokens": prompt_bound(messages), "completion_tokens": TEACHER_MAX_TOKENS,
                   "cached_tokens": 0}
        with self.condition:
            while True:
                if self.stopped:
                    raise BudgetStopped(self.stopped)
                total = self.usage()
                combined = {k: total[k] + reserve[k] for k in total}
                fits = (combined["prompt_tokens"] + combined["completion_tokens"] <= self.limits.max_tokens
                        and self.limits.cost(combined) <= self.limits.max_usd)
                if fits:
                    call = {"id": max(self.calls, default=-1) + 1, "task_id": task_id,
                            "attempt_index": attempt_index, "status": "reserved", "usage": reserve}
                    append_record(self.path, call)  # fsync before the external call.
                    self.calls[call["id"]] = call
                    self.pending.add(call["id"])
                    return call
                if self.pending:
                    self.condition.wait()
                    continue
                raise BudgetStopped("hard cap cannot reserve the next prompt and completion")

    def settle(self, call, usage):
        with self.condition:
            append_record(self.path, dict(call, status="reported", usage=usage))
            self.calls[call["id"]] = dict(call, status="reported", usage=usage)
            if any(usage[k] > call["usage"][k] for k in ("prompt_tokens", "completion_tokens")):
                self.cancel("provider exceeded reserved usage envelope")
                raise BudgetStopped(self.stopped)

    def finish(self, call):
        with self.condition:
            self.pending.discard(call["id"])
            if self.calls[call["id"]]["status"] != "reported":
                self.cancel("request usage is unknown; full reservation retained")
            self.condition.notify_all()


class TeacherSession:
    def __init__(self, config, *, budget=None, task_id=None, attempt_index=None):
        self.config, self.budget = config, budget
        self.task_id, self.attempt_index = task_id, attempt_index
        self.response_texts = []
        self.usage = dict(prompt_tokens=0, completion_tokens=0, cached_tokens=0)
        self.usage_status = "reported"

    @property
    def tokens_spent(self):
        return self.usage["completion_tokens"]

    def generate(self, messages, stop, temperature):
        import appworld_teacher
        call = self.budget.reserve(self.task_id, self.attempt_index, messages) if self.budget else None
        usage = None

        def charge(raw):
            nonlocal usage
            usage = reported_usage(raw)
            if self.budget:
                self.budget.settle(call, usage)

        try:
            # Reasoning models reject stop; truncate locally after accounting.
            supports_stop = not self.config.model.split("/")[-1].startswith(("gpt-5", "o1", "o3", "o4"))
            reply = appworld_teacher.generate_reply(
                self.config, messages, temperature=temperature, retries=0,
                max_completion_tokens=TEACHER_MAX_TOKENS,
                stop=stop if supports_stop else None, usage_callback=charge)
            self.response_texts.append(reply)
            if usage is None:
                raise ValueError("teacher returned no exact usage")
            for marker in stop:
                reply = reply.split(marker, 1)[0]
            return reply
        finally:
            if usage is None:
                self.usage_status = "estimated"
                usage = {"prompt_tokens": prompt_bound(messages), "completion_tokens": TEACHER_MAX_TOKENS,
                         "cached_tokens": 0}
            for key, value in usage.items():
                self.usage[key] += value
            if self.budget:
                self.budget.finish(call)
