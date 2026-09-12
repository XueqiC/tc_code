"""Run the official BFCL agent loop/checkers, capturing every native FC state."""
import copy
from contextvars import ContextVar
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

from .common import (HARNESS, REGISTRY, ROOT, STUDENT, append_row, digest,
                     read_rows, setup_harness, write_json)

_active = ContextVar("mech_bfcl_capture", default=None)


def pack(value):
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, Path):
        return {"__mech_path__": str(value)}
    if isinstance(value, tuple):
        return {"__mech_tuple__": [pack(v) for v in value]}
    if isinstance(value, set):
        return {"__mech_set__": [pack(v) for v in value]}
    if isinstance(value, list):
        return [pack(v) for v in value]
    if isinstance(value, dict):
        return {"__mech_dict__": [[pack(k), pack(v)] for k, v in value.items()]}
    raise TypeError(f"Unserializable executor state: {type(value).__name__}")


def unpack(value):
    if isinstance(value, list):
        return [unpack(v) for v in value]
    if isinstance(value, dict):
        if "__mech_path__" in value:
            return Path(value["__mech_path__"])
        if "__mech_tuple__" in value:
            return tuple(map(unpack, value["__mech_tuple__"]))
        if "__mech_set__" in value:
            return set(map(unpack, value["__mech_set__"]))
        if "__mech_dict__" in value:
            return {unpack(k): unpack(v) for k, v in value["__mech_dict__"]}
    return value


class NoThinking:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __getattr__(self, key):
        return getattr(self.tokenizer, key)

    def apply_chat_template(self, *args, **kwargs):
        kwargs["enable_thinking"] = False
        return self.tokenizer.apply_chat_template(*args, **kwargs)


class CompletionProxy:
    def __init__(self, client):
        self.client = client
        self.completions = self

    def create(self, **kwargs):
        kwargs["model"] = os.environ["MECH_SERVED_MODEL"]
        kwargs["temperature"] = 0.001
        kwargs["seed"] = int(os.environ.get("MECH_SEED", "0"))
        kwargs.setdefault("extra_body", {})["top_k"] = 1
        return self.client.with_options(max_retries=0).completions.create(**kwargs)


def register_capture():
    """Opt-in extension called by tools/bfcl_cli.py only for this pipeline."""
    # overrides resolves superclass names from globals, even for a locally
    # defined opt-in handler. Keep heavyweight harness imports lazy.
    global Gemma4FCHandler
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
    from bfcl_eval.model_handler import base_handler
    from bfcl_eval.model_handler.local_inference.gemma4_fc import Gemma4FCHandler
    from overrides import override

    original_execute = base_handler.execute_multi_turn_func_call

    def execute(*args, **kwargs):
        outputs, instances = original_execute(*args, **kwargs)
        active = _active.get()
        if active is not None:
            try:
                active["snapshot"] = {k: pack(vars(v)) for k, v in instances.items()}
                active["snapshot_error"] = None
            except TypeError as exc:
                active["snapshot"] = None
                active["snapshot_error"] = str(exc)
            if active.get("last_frame"):
                append_row(active["path"], dict(event="execution", frame_id=active["last_frame"],
                    outputs=outputs, snapshot_after=active["snapshot"], snapshot_error=active["snapshot_error"]))
        return outputs, instances

    base_handler.execute_multi_turn_func_call = execute

    class CapturedGemma(Gemma4FCHandler):
        @override(check_signature=False)
        def inference(self, test_entry, include_input_log, exclude_state_log):
            context = dict(task_id=test_entry["id"], entry=copy.deepcopy(test_entry),
                           snapshot={}, snapshot_error=None, index=0,
                           path=Path(os.environ["MECH_CAPTURE_DIR"]) / (test_entry["id"] + ".jsonl"))
            token = _active.set(context)
            try:
                return super().inference(test_entry, True, False)
            finally:
                _active.reset(token)

        @override(check_signature=False)
        def _query_prompting(self, inference_data):
            active = _active.get()
            frame_id = f"{active['task_id']}:{active['index']}"
            active["index"] += 1
            active["last_frame"] = frame_id
            frame = dict(event="query", frame_id=frame_id, task_id=active["task_id"],
                         messages=copy.deepcopy(inference_data["message"]),
                         functions=copy.deepcopy(inference_data["function"]),
                         snapshot=copy.deepcopy(active["snapshot"]),
                         snapshot_error=active["snapshot_error"],
                         initial_config=active["entry"].get("initial_config", {}),
                         involved_classes=active["entry"].get("involved_classes", []))
            proxy = SimpleNamespace(tokenizer=NoThinking(self.tokenizer),
                                    client=CompletionProxy(self.client), temperature=0.001,
                                    max_context_length=int(os.environ.get("MECH_MAX_CONTEXT", "32768")),
                                    model_path_or_id=self.model_path_or_id)
            proxy._format_prompt = lambda m, f: Gemma4FCHandler._format_prompt(proxy, m, f)
            append_row(active["path"], dict(frame, event="query_started"))
            response, latency = Gemma4FCHandler._query_prompting(proxy, inference_data)
            append_row(active["path"], dict(frame, response=response.choices[0].text,
                finish_reason=response.choices[0].finish_reason,
                usage=response.usage.model_dump(), prompt=inference_data["inference_input_log"]["formatted_prompt"]))
            return response, latency

    MODEL_CONFIG_MAPPING[REGISTRY] = replace(MODEL_CONFIG_MAPPING[REGISTRY], model_handler=CapturedGemma)


def frames(directory):
    result = []
    for path in sorted(Path(directory).glob("*.jsonl")):
        by_id = {}
        for row in read_rows(path):
            if row["event"] == "query":
                by_id[row["frame_id"]] = row
            elif row["event"] == "execution" and row["frame_id"] in by_id:
                by_id[row["frame_id"]].update(execution=row)
        result.extend(by_id.values())
    return result


def official_run(args, ids, destination, adapter, splits):
    from bfas.adapters.bfcl import extract_verdicts, read_score_summaries
    from bfas.bfcl_teacher import read_results
    destination = Path(destination).resolve()
    metadata = dict(ids=ids, split_hash=digest(splits), base_url=args.base_url,
                    served_model=args.served_model, temperature=0.001, top_k=1, seed=splits["seed"])
    if (destination / "items.json").exists():
        from .common import read_json
        if read_json(destination / "run.json") != metadata:
            raise ValueError("Cannot reuse a harness run with changed inputs")
        return read_json(destination / "items.json")
    if destination.exists():
        raise ValueError(f"Incomplete harness run preserved at {destination}; use a new run directory")
    destination.mkdir(parents=True)
    write_json(destination / "run.json", metadata)
    write_json(destination / "test_case_ids_to_generate.json", adapter._selective_file(ids))
    env = os.environ.copy()
    env.update(BFCL_PROJECT_ROOT=str(destination), REMOTE_OPENAI_BASE_URL=args.base_url,
               REMOTE_OPENAI_TOKENIZER_PATH=args.tokenizer, MECH_SERVED_MODEL=args.served_model,
               MECH_CAPTURE_DIR=str(destination / "trajectories"), MECH_BFCL_CAPTURE="1",
               MECH_SEED=str(splits["seed"]), PYTHONDONTWRITEBYTECODE="1")
    command = [args.bfcl_python, str(ROOT / "tools/bfcl_cli.py")]
    start = time.monotonic()
    with (destination / "generate.log").open("w") as log:
        subprocess.run(command + ["generate", "--model", REGISTRY, "--run-ids",
            "--skip-server-setup", "--backend", "vllm", "--temperature", "0.001",
            "--num-threads", "1", "--include-input-log", "--result-dir", "result"],
            env=env, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
    with (destination / "evaluate.log").open("w") as log:
        outcome = subprocess.run(command + ["evaluate", "--model", REGISTRY,
            "--result-dir", "result", "--score-dir", "score", "--partial-eval"],
            env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    expected = {t: splits["items"][t]["category"] for t in ids}
    summaries = read_score_summaries(destination / "score")
    if set(expected.values()) - summaries.keys():
        raise RuntimeError("Official checker omitted categories; inspect evaluate.log")
    verdicts = extract_verdicts(destination / "score", expected)
    results = {r["id"]: r for r in read_results(destination / "result")}
    if set(ids) - results.keys():
        raise RuntimeError("Harness omitted requested tasks")
    captured = frames(destination / "trajectories")
    items = []
    for tid in ids:
        own = [f for f in captured if f["task_id"] == tid]
        if not own:
            raise RuntimeError(f"No completed student trajectory for {tid}")
        row = dict(id=tid, **splits["items"][tid], correct=verdicts[tid], layer=3,
                   checker_result=results[tid], metrics=trajectory_metrics(own, verdicts[tid]))
        items.append(row)
    write_json(destination / "items.json", items)
    write_json(destination / "timing.json", dict(wall_seconds=time.monotonic()-start,
                                                 evaluator_returncode=outcome.returncode))
    return items


def trajectory_metrics(states, correct):
    from bfcl_eval.model_handler.local_inference.gemma4_fc import _parse_response
    calls, illegal, repeated, no_calls, truncated = 0, 0, 0, 0, 0
    seen = set()
    for f in states:
        try:
            parsed = _parse_response(f["response"])[0]
        except ValueError:
            parsed = []
            illegal += 1
        no_calls += int(not parsed)
        truncated += int(f.get("finish_reason") == "length")
        calls += len(parsed)
        for call in parsed:
            key = digest(call)
            repeated += int(key in seen)
            seen.add(key)
            illegal += int(call["name"] not in {d["name"] for d in f["functions"]})
        illegal += sum("error" in str(o).lower() for o in f.get("execution", {}).get("outputs", []))
    return dict(tool_calls=calls, illegal_actions=illegal, repeated_actions=repeated,
                no_call_turns=no_calls, truncated_generations=truncated,
                early_stops=int(not correct and bool(states) and not parsed),
                early_stop_definition="failed item whose final action contains no call (proxy)")
