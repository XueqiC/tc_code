"""Append-only, fail-closed accounting around the shared official API client."""
import json
import os
import time
import uuid

from .common import BUDGETS, TEACHER, append_row, digest, lock, read_rows, write_json


class UnknownUsage(RuntimeError):
    pass


def exact_usage(raw):
    if not isinstance(raw, dict):
        raise UnknownUsage("Provider did not return a usage object")
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if type(raw.get(key)) is not int or raw[key] < 0:
            raise UnknownUsage(f"Missing exact provider usage: {key}")
    if raw["total_tokens"] != raw["prompt_tokens"] + raw["completion_tokens"]:
        raise UnknownUsage("Provider usage does not reconcile")
    return raw


def ledger_summary(path):
    calls = {}
    for row in read_rows(path):
        calls.setdefault(row["call_id"], {}).update(row)
    summary = {b: dict(input_tokens=0, output_tokens=0, calls=0, unresolved=0) for b in BUDGETS}
    for row in calls.values():
        bucket = summary[row["bucket"]]
        bucket["calls"] += 1
        try:
            usage = exact_usage(row.get("usage"))
        except UnknownUsage:
            bucket["unresolved"] += 1
            continue
        bucket["input_tokens"] += usage["prompt_tokens"]
        bucket["output_tokens"] += usage["completion_tokens"]
    return summary


class Teacher:
    def __init__(self, directory, config=None, generate=None):
        import appworld_teacher as shared
        self.directory = directory
        self.path = directory / "teacher_ledger.jsonl"
        if config is None:
            os.environ.setdefault("BFAS_TEACHER", TEACHER)
            os.environ.setdefault("BFAS_OPENAI_SERVICE_TIER", "flex")
            if os.environ["BFAS_TEACHER"] != TEACHER or os.environ["BFAS_OPENAI_SERVICE_TIER"] != "flex":
                raise ValueError("Requires BFAS_TEACHER=openai/gpt-5.6-luna and BFAS_OPENAI_SERVICE_TIER=flex")
            config = shared.load_teacher_config(TEACHER)
        if config.backend != "openai_api" or config.endpoint != "https://api.openai.com/v1/chat/completions":
            raise ValueError("Teacher must use the official OpenAI endpoint")
        if config.model != "gpt-5.6-luna" or config.service_tier != "flex":
            raise ValueError("Teacher model and requested tier must be luna/flex")
        self.config = config
        self.generate = generate or shared.generate_reply

    def ask(self, bucket, key, messages, cap, *, output_budget=None, require_full_cap=False):
        """One attempt per stable key, no hidden retries or token estimates.

        A timeout can have been billed without returning usage. Such attempts
        cannot honestly be assigned exact zero cost: preserve their reservation
        and block all further purchases until external billing reconciliation.
        Arm generation supplies its frozen output budget and requires a full
        reservation. Cached attempts are returned even when no new call fits.
        """
        with lock(self.path.with_suffix(".lock")):
            summary = ledger_summary(self.path)
            if any(v["unresolved"] for v in summary.values()):
                raise UnknownUsage("Unresolved API attempt in ledger; reconcile exact usage before continuing")
            previous = [r for r in read_rows(self.path) if r.get("key") == key and r.get("event") == "final"]
            if previous:
                row = previous[-1]
                if row["prompt_hash"] != digest(messages):
                    raise ValueError("Attempt key reused with changed context")
                return row.get("parsed"), row["call_id"]
            budget = BUDGETS[bucket] if output_budget is None else output_budget
            remaining = budget - summary[bucket]["output_tokens"]
            if require_full_cap and cap > remaining:
                return None, None
            cap = min(cap, remaining)
            if cap <= 0:
                return None, None
            call_id = uuid.uuid4().hex
            record = dict(call_id=call_id, key=key, bucket=bucket, event="reserved",
                          requested_model=self.config.model, requested_service_tier="flex",
                          reasoning_effort="none",
                          cap=cap, output_budget=budget, timestamp=time.time(),
                          prompt_hash=digest(messages), usage=None)
            append_row(self.path, record)
            write_json(self.directory / "teacher_raw" / (call_id + ".request.json"),
                       dict(messages=messages, cap=cap, model=self.config.model, service_tier="flex", reasoning_effort="none"))
            envelope, usage, parsed, error = {}, None, None, None

            def receive(event):
                nonlocal envelope, usage
                envelope = event
                data = event.get("data")
                data = data if isinstance(data, dict) else {}
                usage = data.get("usage")
                write_json(self.directory / "teacher_raw" / (call_id + ".response.json"), event)
                append_row(self.path, dict(call_id=call_id, event="response", usage=usage,
                    returned_model=data.get("model"), system_fingerprint=data.get("system_fingerprint"),
                    response_id=data.get("id"), request_id=event.get("request_id"),
                    returned_service_tier=data.get("service_tier"), http_status=event.get("http_status")))

            try:
                text = self.generate(self.config, messages, retries=0,
                                     max_completion_tokens=cap, response_callback=receive, reasoning_effort="none")
                exact_usage(usage)
                data = envelope.get("data", {})
                if not data.get("model"):
                    raise ValueError("Missing returned model ID")
                choices = data.get("choices", [])
                if not choices or choices[0].get("finish_reason") != "stop":
                    raise ValueError("Discarded non-stop/truncated generation")
                parsed = json.loads(text)
                if not isinstance(parsed, dict):
                    raise ValueError("Teacher must return a JSON object")
            except Exception as exc:
                parsed = None
                error = f"{type(exc).__name__}: {exc}".replace(self.config.api_key, "[REDACTED]")
            final = dict(record, event="final", usage=usage, parsed=parsed, error=error,
                         disposition="candidate" if parsed is not None else "discarded",
                         wall_seconds=time.time() - record["timestamp"])
            append_row(self.path, final)
            exact_usage(usage)  # Unknown usage never becomes a zero-cost failure.
            if usage["completion_tokens"] > cap:
                raise RuntimeError("Provider exceeded the reserved output cap")
            return parsed, call_id
