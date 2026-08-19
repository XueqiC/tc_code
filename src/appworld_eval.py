# Usage: .venv/bin/python src/appworld_eval.py --model MODEL --tag RUN_NAME
# Optional LoRA: add --adapter PEFT_DIR (the adapter is merged before evaluation).
# Outputs: results/appworld/RUN_NAME/records.jsonl and metrics.json.
"""Minimal HuggingFace causal-LM evaluation harness for AppWorld."""

from __future__ import annotations

import argparse
import ast
import html
import json
import os
import re
import select
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


CODE_BLOCK_RE = re.compile(
    r"```[ \t]*(?:(?:python|py)[ \t]*)?\r?\n(.*?)```", re.IGNORECASE | re.DOTALL
)
CODE_TAG_RE = re.compile(r"<code(?:\s[^>]*)?>(.*?)</code>", re.IGNORECASE | re.DOTALL)
GENERIC_CHAT_TEMPLATE = """{% for message in messages %}{{ message['role'] | upper + ':\\n' + message['content'] + '\\n\\n' }}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:\\n' }}{% endif %}"""
BRIDGE_RESPONSE_TIMEOUT_SECONDS = 300.0


class BridgeError(RuntimeError):
    """The bridge process exited, timed out, or violated its protocol."""


class BridgeRemoteError(RuntimeError):
    """AppWorld reported an error while handling a valid bridge request."""

    def __init__(self, response: dict[str, Any]):
        super().__init__(str(response.get("error", "Unknown AppWorld bridge error")))
        self.response = response


class AppWorldBridge:
    def __init__(self, project_root: Path):
        bridge_python = project_root / "envs" / "appworld-venv" / "bin" / "python"
        bridge_script = project_root / "src" / "appworld_bridge.py"
        bridge_cwd = project_root / "envs"
        bridge_env = os.environ.copy()
        bridge_env["APPWORLD_ROOT"] = str(project_root / "envs" / "appworld-data")

        self._process = subprocess.Popen(
            [str(bridge_python), "-u", str(bridge_script)],
            cwd=str(bridge_cwd),
            env=bridge_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._next_request_id = 1

    @property
    def alive(self) -> bool:
        return self._process.poll() is None

    def request(self, operation: str, **payload: Any) -> dict[str, Any]:
        if not self.alive or self._process.stdin is None or self._process.stdout is None:
            raise BridgeError(f"AppWorld bridge is not running (exit={self._process.poll()}).")

        request_id = self._next_request_id
        self._next_request_id += 1
        request = {"op": operation, "request_id": request_id, **payload}
        try:
            self._process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise BridgeError(f"Could not write to AppWorld bridge: {exc}") from exc

        deadline = time.monotonic() + BRIDGE_RESPONSE_TIMEOUT_SECONDS
        ignored_lines = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BridgeError(f"AppWorld bridge timed out during {operation!r}.")
            ready, _, _ = select.select([self._process.stdout], [], [], remaining)
            if not ready:
                raise BridgeError(f"AppWorld bridge timed out during {operation!r}.")
            line = self._process.stdout.readline()
            if line == "":
                raise BridgeError(
                    f"AppWorld bridge exited during {operation!r} "
                    f"(exit={self._process.poll()})."
                )
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                # A dependency may bypass Python's redirected stdout. Ignore a small
                # amount of such logging while waiting for the actual response.
                ignored_lines += 1
                if ignored_lines > 100:
                    raise BridgeError("Too much non-JSON output from AppWorld bridge.")
                continue
            if not isinstance(response, dict):
                continue
            if response.get("request_id") != request_id:
                continue
            if not response.get("ok", False):
                raise BridgeRemoteError(response)
            return response

    def close(self) -> None:
        if self.alive:
            try:
                self.request("stop")
            except (BridgeError, BridgeRemoteError):
                pass
        if self._process.stdin is not None:
            try:
                self._process.stdin.close()
            except OSError:
                pass
        if self.alive:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        else:
            self._process.wait()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _tag(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) or value in {".", ".."}:
        raise argparse.ArgumentTypeError(
            "must contain only letters, digits, '.', '_', or '-' and start alphanumerically"
        )
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--adapter", help="Optional PEFT LoRA adapter directory")
    parser.add_argument("--split", choices=("train", "dev", "test_normal", "test_challenge"), default="train")
    parser.add_argument("--max-tasks", type=_positive_int, default=10)
    parser.add_argument("--max-steps", type=_positive_int, default=12)
    parser.add_argument("--max-new-tokens", type=_positive_int, default=512)
    parser.add_argument("--tag", required=True, type=_tag, help="Output directory name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=_nonnegative_float, default=0.0)
    return parser.parse_args(argv)


def load_model(args: argparse.Namespace) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if getattr(tokenizer, "chat_template", None) is None:
        tokenizer.chat_template = GENERIC_CHAT_TEMPLATE
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=False,
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def _model_input_device(model: Any) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("Model has no materialized parameters")


def generate_reply(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
) -> str:
    template_args = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
    }
    try:
        if os.environ.get("APPWORLD_THINK") == "1":
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, **template_args
            )
            suffix = "<think>\n\n</think>\n\n"
            if rendered.endswith(suffix):
                rendered = rendered[: -len(suffix)]
            encoded = tokenizer(rendered, return_tensors="pt")
        else:
            encoded = tokenizer.apply_chat_template(messages, return_dict=True, **template_args)
    except TypeError as exc:
        if "return_dict" not in str(exc):
            raise
        encoded = tokenizer.apply_chat_template(messages, **template_args)

    if isinstance(encoded, Mapping):
        inputs = dict(encoded)
    else:
        inputs = {"input_ids": encoded}
    device = _model_input_device(model)
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    input_length = inputs["input_ids"].shape[-1]

    generation_args: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generation_args["temperature"] = temperature
    if tokenizer.pad_token_id is not None:
        generation_args["pad_token_id"] = tokenizer.pad_token_id
    # Merged fine-tuned checkpoints can ship without an eos list in
    # their generation_config, in which case generation runs through
    # the turn boundary and the model hallucinates the following
    # user/execution turns into its own reply. Pin the stop tokens
    # explicitly so every model stops where the chat template ends a
    # turn.
    eos_ids = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))
    for token in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(token)
        if isinstance(tid, int) and tid >= 0:
            eos_ids.add(tid)
    if eos_ids:
        generation_args["eos_token_id"] = sorted(eos_ids)

    with torch.inference_mode():
        generated = model.generate(**inputs, **generation_args)
    reply_tokens = generated[0, input_length:]
    return tokenizer.decode(reply_tokens, skip_special_tokens=True).strip()


def extract_python_code(reply: str) -> str | None:
    if os.environ.get("APPWORLD_THINK") == "1" and "</think>" in reply:
        reply = reply.split("</think>", 1)[1]
    match = CODE_BLOCK_RE.search(reply)
    if match:
        code = match.group(1).strip()
        return code or None
    match = CODE_TAG_RE.search(reply)
    if match:
        code = html.unescape(match.group(1)).strip()
        return code or None
    return None


def calls_complete_task(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "complete_task":
            continue
        supervisor = node.func.value
        if (
            isinstance(supervisor, ast.Attribute)
            and supervisor.attr == "supervisor"
            and isinstance(supervisor.value, ast.Name)
            and supervisor.value.id == "apis"
        ):
            return True
    return False


def truncate_output(text: str, limit: int = 1500) -> str:
    if len(text) <= limit:
        return text
    marker = "\n...[output truncated]...\n"
    left = (limit - len(marker)) // 2
    right = limit - len(marker) - left
    return text[:left] + marker + text[-right:]


def make_system_prompt(task: dict[str, Any]) -> str:
    supervisor = task.get("supervisor") or {}
    details = ", ".join(
        f"{key}={value}"
        for key, value in (
            ("name", supervisor.get("name")),
            ("email", supervisor.get("email")),
            ("phone", supervisor.get("phone")),
        )
        if value
    ) or "not available"
    instruction = task.get("instruction", "")
    return (
        "You are an agent operating inside AppWorld. Write Python code that "
        "uses the preloaded `apis` object to complete the user's task.\n\n"
        "How the `apis` object works (important):\n"
        "- Apps are namespaces, APIs are methods: call "
        "`apis.<app_name>.<api_name>(...)`. Never call an app itself "
        "(`apis.api_docs()` is a TypeError).\n"
        "- Discover what exists, in this order:\n"
        "  1. `print(apis.api_docs.show_app_descriptions())`\n"
        "  2. `print(apis.api_docs.show_api_descriptions(app_name='<app>'))`\n"
        "  3. `print(apis.api_docs.show_api_doc(app_name='<app>', "
        "api_name='<api>'))`\n"
        "- Most apps require login. Get the supervisor's stored passwords "
        "with `print(apis.supervisor.show_account_passwords())`, then call "
        "the app's `login` API with the supervisor's email/username and that "
        "password; pass the returned access token to later calls of that app "
        "as its api_doc specifies.\n"
        "- Print API results so you can inspect them before deciding the "
        "next step. Do not invent APIs or argument names.\n"
        "- If the task asks a question, submit the answer explicitly: "
        "`apis.supervisor.complete_task(answer=<value>)`. If the task only "
        "requires actions, call `apis.supervisor.complete_task()` with no "
        "arguments.\n"
        "- List APIs are paginated (`page_index` starts at 0): loop over "
        "pages until an empty page when you need a complete list.\n"
        "- Stay with the app(s) the task points to; do not wander into "
        "unrelated apps.\n"
        "\n"
        "Return exactly one executable Python code block per turn, with no "
        "additional code blocks. You will receive the execution output and "
        "may then write the next block. State persists across turns. When "
        "the task is fully complete, call `apis.supervisor.complete_task()` "
        "in the final code block.\n\n"
        f"Task instruction:\n{instruction}\n\n"
        f"Supervisor details: {details}"
    )

def _append_error(existing: str | None, new_error: str) -> str:
    return new_error if existing is None else f"{existing}; {new_error}"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "results" / "appworld" / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    metrics_path = output_dir / "metrics.json"

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model, tokenizer = load_model(args)

    records: list[dict[str, Any]] = []
    bridge: AppWorldBridge | None = None
    with records_path.open("w", encoding="utf-8") as records_file:
        for task_index in range(args.max_tasks):
            started_at = time.perf_counter()
            task_id = f"{args.split}[{task_index}]"
            steps_used = 0
            success = False
            passed_tests = 0
            failed_tests = 0
            error: str | None = None
            task_started = False

            if bridge is None or not bridge.alive:
                if bridge is not None:
                    bridge.close()
                bridge = AppWorldBridge(project_root)

            try:
                task = bridge.request(
                    "start",
                    split=args.split,
                    index=task_index,
                    experiment_name=args.tag,
                    seed=args.seed,
                )
                task_started = True
                task_id = str(task["task_id"])
                # APPWORLD_TASK_FILTER=prefix1,prefix2 evaluates only
                # matching tasks (trajectory dissection reruns).
                _flt = os.environ.get("APPWORLD_TASK_FILTER")
                if _flt and not any(
                    task_id.startswith(p)
                    for p in _flt.split(",") if p
                ):
                    try:
                        bridge.request("stop")
                    except Exception:
                        pass
                    continue
            except BridgeRemoteError as exc:
                if exc.response.get("error_code") == "end_of_split":
                    break
                error = _append_error(error, f"start: {exc}")
                task = None
            except Exception as exc:
                error = _append_error(error, f"start: {type(exc).__name__}: {exc}")
                task = None

            if task is not None:
                messages = [
                    {"role": "system", "content": make_system_prompt(task)},
                    {"role": "user", "content": "Begin by consulting the API documentation."},
                ]
                try:
                    for _ in range(args.max_steps):
                        reply = generate_reply(
                            model,
                            tokenizer,
                            messages,
                            args.max_new_tokens,
                            args.temperature,
                        )
                        messages.append({"role": "assistant", "content": reply})
                        code = extract_python_code(reply)
                        if code is None:
                            break
                        steps_used += 1
                        execution = bridge.request("execute", code=code)
                        output = truncate_output(str(execution.get("output", "")))
                        messages.append(
                            {
                                "role": "user",
                                "content": f"Execution output:\n{output}\n\nContinue the task.",
                            }
                        )
                        if calls_complete_task(code):
                            break
                except Exception as exc:
                    error = _append_error(error, f"agent loop: {type(exc).__name__}: {exc}")

            if task_started:
                try:
                    evaluation = bridge.request("evaluate")
                    success = bool(evaluation.get("success", False))
                    passed_tests = int(evaluation.get("passed_tests", 0))
                    failed_tests = int(evaluation.get("failed_tests", 0))
                except Exception as exc:
                    error = _append_error(error, f"evaluate: {type(exc).__name__}: {exc}")
                try:
                    bridge.request("stop")
                except Exception as exc:
                    error = _append_error(error, f"stop: {type(exc).__name__}: {exc}")

            if os.environ.get("APPWORLD_DEBUG") == "1":
                with (output_dir / "transcripts.jsonl").open(
                    "a", encoding="utf-8"
                ) as dbg:
                    dbg.write(json.dumps(
                        {"task_id": task_id, "messages": messages},
                        ensure_ascii=False) + "\n")
            record: dict[str, Any] = {
                "task_id": task_id,
                "steps_used": steps_used,
                "success": success,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "wall_seconds": round(time.perf_counter() - started_at, 3),
            }
            if error is not None:
                record["error"] = error
            records.append(record)
            records_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            records_file.flush()

            if bridge is not None and not bridge.alive:
                bridge.close()
                bridge = None

    if bridge is not None:
        bridge.close()

    n_tasks = len(records)
    n_successes = sum(int(record["success"]) for record in records)
    # Official AppWorld aggregates: TGC is the per-task pass rate under
    # the official evaluator, and SGC counts a scenario as passed only
    # when every one of its task variants passes (scenario = the task-id
    # prefix before the underscore).
    scenario_tasks: dict[str, list[bool]] = {}
    for record in records:
        scen = str(record["task_id"]).rsplit("_", 1)[0]
        scenario_tasks.setdefault(scen, []).append(bool(record["success"]))
    n_scen = len(scenario_tasks)
    metrics = {
        "n_tasks": n_tasks,
        "success_rate": n_successes / n_tasks if n_tasks else 0.0,
        "tgc": n_successes / n_tasks if n_tasks else 0.0,
        "sgc": (
            sum(all(v) for v in scenario_tasks.values()) / n_scen
            if n_scen else 0.0
        ),
        "n_scenarios": n_scen,
        "mean_steps": (
            sum(int(record["steps_used"]) for record in records) / n_tasks if n_tasks else 0.0
        ),
        "config": {
            "model": args.model,
            "adapter": args.adapter,
            "split": args.split,
            "max_tasks": args.max_tasks,
            "max_steps": args.max_steps,
            "max_new_tokens": args.max_new_tokens,
            "tag": args.tag,
            "seed": args.seed,
            "temperature": args.temperature,
        },
    }
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2, ensure_ascii=False)
        metrics_file.write("\n")

    print(
        f"AppWorld: {n_successes}/{n_tasks} succeeded "
        f"({metrics['success_rate']:.1%}); mean_steps={metrics['mean_steps']:.2f}; "
        f"results={output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
