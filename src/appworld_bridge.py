"""JSON-lines bridge between the model evaluator and an AppWorld process."""

from __future__ import annotations

import contextlib
import json
import os
import sys
import traceback
from collections.abc import Mapping, Sequence
from typing import Any


_world: Any | None = None
_task_ids_by_split: dict[str, list[str]] = {}


def _close_world() -> None:
    global _world
    if _world is None:
        return
    try:
        close = getattr(_world, "close", None)
        if callable(close):
            close()
        else:
            exit_method = getattr(_world, "__exit__", None)
            if callable(exit_method):
                exit_method(None, None, None)
    finally:
        _world = None


def _field(value: Any, *names: str) -> Any | None:
    for name in names:
        try:
            item = value.get(name) if isinstance(value, Mapping) else getattr(value, name)
        except (AttributeError, KeyError, TypeError):
            continue
        if item is not None and not callable(item):
            return item
    return None


def _supervisor_details(task: Any) -> dict[str, str | None]:
    supervisor = _field(task, "supervisor")
    if supervisor is None:
        return {"name": None, "email": None, "phone": None}

    name = _field(supervisor, "name", "full_name")
    if name is None:
        first = _field(supervisor, "first_name")
        last = _field(supervisor, "last_name")
        name = " ".join(str(part) for part in (first, last) if part) or None

    return {
        "name": str(name) if name is not None else None,
        "email": (
            str(email)
            if (email := _field(supervisor, "email", "email_address")) is not None
            else None
        ),
        "phone": (
            str(phone)
            if (phone := _field(supervisor, "phone", "phone_number")) is not None
            else None
        ),
    }


def _result_text(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    output = _field(result, "output", "stdout", "text")
    return str(output if output is not None else result)


def _tracker_count(tracker: Any, names: tuple[str, ...]) -> int:
    for name in names:
        value = _field(tracker, name)
        if value is None or isinstance(value, (str, bytes)):
            continue
        try:
            return len(value)
        except TypeError:
            continue

    for method_name in ("to_dict", "model_dump", "dict"):
        method = getattr(tracker, method_name, None)
        if not callable(method):
            continue
        try:
            data = method()
        except Exception:
            continue
        if isinstance(data, Mapping):
            for name in names:
                value = data.get(name)
                if isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes)):
                    return len(value)
    return 0


def _handle(request: dict[str, Any], AppWorld: Any, load_task_ids: Any) -> dict[str, Any]:
    global _world
    operation = request.get("op")

    if operation == "start":
        _close_world()
        split = str(request.get("split", "train"))
        if split not in _task_ids_by_split:
            _task_ids_by_split[split] = list(load_task_ids(split))
        task_ids = _task_ids_by_split[split]

        if "task_id" in request:
            task_id = str(request["task_id"])
        else:
            index = int(request.get("index", 0))
            if index < 0 or index >= len(task_ids):
                return {
                    "ok": False,
                    "error_code": "end_of_split",
                    "error": f"Task index {index} is outside split {split!r} ({len(task_ids)} tasks).",
                }
            task_id = task_ids[index]

        _world = AppWorld(
            task_id=task_id,
            experiment_name=str(request.get("experiment_name", "appworld_eval")),
            random_seed=int(request.get("seed", 42)),
        )
        task = _world.task
        return {
            "ok": True,
            "task_id": task_id,
            "instruction": str(_field(task, "instruction") or ""),
            "supervisor": _supervisor_details(task),
        }

    if operation == "execute":
        if _world is None:
            raise RuntimeError("No active AppWorld task; send a start request first.")
        code = request.get("code")
        if not isinstance(code, str):
            raise TypeError("execute requires a string 'code' field")
        return {"ok": True, "output": _result_text(_world.execute(code))}

    if operation == "evaluate":
        if _world is None:
            raise RuntimeError("No active AppWorld task; send a start request first.")
        tracker = _world.evaluate()
        success = _field(tracker, "success")
        return {
            "ok": True,
            "success": bool(success),
            "passed_tests": _tracker_count(
                tracker, ("passes", "passed_tests", "pass_tests", "passed")
            ),
            "failed_tests": _tracker_count(
                tracker, ("failures", "failed_tests", "fail_tests", "failed")
            ),
        }

    if operation == "stop":
        _close_world()
        return {"ok": True}

    raise ValueError(f"Unknown operation: {operation!r}")


def _emit(stream: Any, response: dict[str, Any]) -> None:
    stream.write(json.dumps(response, ensure_ascii=False, default=str) + "\n")
    stream.flush()


def main() -> int:
    protocol_stdout = sys.stdout
    if not os.environ.get("APPWORLD_ROOT"):
        _emit(protocol_stdout, {"ok": False, "error": "APPWORLD_ROOT is not set"})
        return 2

    # Keep import-time and AppWorld runtime logging away from the JSON protocol.
    with contextlib.redirect_stdout(sys.stderr):
        from appworld import AppWorld, load_task_ids

    try:
        for line in sys.stdin:
            request_id: Any = None
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise TypeError("Each request must be a JSON object")
                request_id = request.get("request_id")
                with contextlib.redirect_stdout(sys.stderr):
                    response = _handle(request, AppWorld, load_task_ids)
            except Exception as exc:
                response = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=8),
                }
            response["request_id"] = request_id
            _emit(protocol_stdout, response)
    finally:
        with contextlib.redirect_stdout(sys.stderr):
            _close_world()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
