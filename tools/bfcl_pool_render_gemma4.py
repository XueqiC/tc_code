#!/usr/bin/env python3
"""Convert BFCL demo JSONL to Gemma 4's deployed native function-call format.

Usage: .venv/bin/python tools/bfcl_pool_render_gemma4.py --in POOL --out ROWS --verify
Loads only the cached google/gemma-4-12B-it tokenizer, never model weights or APIs.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BFCL = ROOT / "envs/bfcl/gorilla/berkeley-function-call-leaderboard"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(BFCL))

from bfcl_eval.model_handler.local_inference.gemma4_fc import Gemma4FCHandler

MODEL = "google/gemma-4-12B-it"
FORMAT = "bfcl-gemma4-native-fc-v1"
EMPTY_THOUGHT = "<|channel>thought\n<channel|>"
CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


class _TemplateMode:
    """Use the handler's tool conversion with the template's completion mode.

    The existing handler exposes only generation-prompt rendering. This small
    tokenizer proxy changes that one flag, keeping its schema conversion and
    dotted-name handling as the single source of truth.
    """

    def __init__(self, tokenizer, add_generation_prompt):
        self.tokenizer = tokenizer
        self.add_generation_prompt = add_generation_prompt

    def apply_chat_template(self, messages, **kwargs):
        kwargs["add_generation_prompt"] = self.add_generation_prompt
        return self.tokenizer.apply_chat_template(messages, **kwargs)


def template_render(tokenizer, messages, functions, *, generation=True):
    return Gemma4FCHandler._format_prompt(
        SimpleNamespace(tokenizer=_TemplateMode(tokenizer, generation)),
        copy.deepcopy(messages), copy.deepcopy(functions),
    )


def _call(call: dict, functions: list[dict]) -> dict:
    call = call.get("function", call)
    if "name" in call and "arguments" in call:
        name, arguments = call["name"], call["arguments"]
    elif len(call) == 1:
        name, arguments = next(iter(call.items()))
    else:
        raise ValueError(f"Invalid teacher tool call: {call!r}")
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(name, str) or not name or not isinstance(arguments, dict):
        raise ValueError("Tool calls require a name and a JSON argument object")
    # OpenAI substitutes underscores for dots; Gemma's deployed handler keeps
    # the original BFCL name. Resolve against this row's actual declarations.
    names = [function["name"] for function in functions]
    if name not in names:
        candidates = [n for n in names if n.replace(".", "_") == name]
        if len(candidates) != 1:
            raise ValueError(f"Unknown or ambiguous teacher function: {name!r}")
        name = candidates[0]
    return {"type": "function", "function": {"name": name, "arguments": arguments}}


def assistant_message(response: Any, functions: list[dict]) -> dict:
    """Accept BFCL decoded calls, OpenAI messages, or legacy Qwen FC blocks."""
    if isinstance(response, str):
        try:
            decoded = json.loads(response)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, list) and any(isinstance(item, list) for item in decoded):
            raise ValueError("Whole-episode targets require separate rows with each step's context")
        names = {function["name"] for function in functions}
        names |= {name.replace(".", "_") for name in names}
        if isinstance(decoded, list) and all(isinstance(item, dict) for item in decoded):
            response = decoded
        elif isinstance(decoded, dict) and (
            "tool_calls" in decoded or decoded.get("role") == "assistant"
            or {"name", "arguments"} <= decoded.keys()
            or (len(decoded) == 1 and next(iter(decoded)) in names)
        ):
            response = decoded
    if isinstance(response, str):
        if "<tool_call>" in response:
            matches = list(CALL_BLOCK.finditer(response))
            remainder = CALL_BLOCK.sub("", response)
            if not matches or remainder.strip():
                raise ValueError("Malformed or mixed-text Qwen tool-call target")
            response = [json.loads(match[1]) for match in matches]
    if isinstance(response, dict):
        if "tool_calls" in response or response.get("role") == "assistant":
            calls = response.get("tool_calls") or []
            content = response.get("content") or ""
        else:
            calls, content = [response], ""
    elif isinstance(response, list):
        calls, content = response, ""
    elif isinstance(response, str):
        calls, content = [], response
    else:
        raise ValueError("Teacher response must be text or structured tool calls")
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [_call(call, functions) for call in calls]
    return message


def render_pair(tokenizer, messages, functions, assistant):
    prompt = template_render(tokenizer, messages, functions)
    completed = template_render(
        tokenizer, [*messages, assistant], functions, generation=False
    )
    if completed.startswith(prompt):
        return prompt, completed[len(prompt):]
    # The cached template emits this empty channel ONLY with
    # add_generation_prompt=True. Align the completed assistant header to that
    # prefilled generation prefix; all target bytes still come from the template.
    if prompt.endswith(EMPTY_THOUGHT):
        header = prompt[:-len(EMPTY_THOUGHT)]
        if completed.startswith(header):
            return prompt, completed[len(header):]
    raise ValueError("Gemma assistant rendering does not extend the deployment prompt")


def _chatml_context(prompt: str) -> dict:
    """Recover context from pools saved with the Qwen handler's raw prompt."""
    suffix = "<|im_start|>assistant\n"
    if not prompt.endswith(suffix):
        raise ValueError("Row needs _render_context or a complete Qwen FC prompt")
    text = prompt[:-len(suffix)]
    matches = list(re.finditer(
        r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>\n?",
        text, re.DOTALL,
    ))
    if "".join(m[0] for m in matches) != text:
        raise ValueError("Cannot recover complete messages from Qwen prompt")
    messages = [{"role": m[1], "content": m[2]} for m in matches]
    if not messages or messages[0]["role"] != "system":
        raise ValueError("Qwen prompt has no tool declaration block")
    system = messages.pop(0)["content"]
    block = re.search(r"<tools>\n(.*?)\n</tools>", system, re.DOTALL)
    if block is None:
        raise ValueError("Qwen prompt has no recoverable tool schemas")
    functions = [json.loads(line) for line in block[1].splitlines() if line.strip()]
    prefix, marker, _ = system.partition("# Tools")
    if not marker:
        raise ValueError("Unrecognized Qwen tool system prompt")
    if prefix.strip():
        messages.insert(0, {"role": "system", "content": prefix.rstrip("\n")})
    history = []
    pending = []
    for message in messages:
        content = message["content"]
        if message["role"] == "assistant":
            # Qwen's empty thinking prefill is not an assistant answer.
            content = re.sub(r"^<think>.*?</think>\s*", "", content, flags=re.DOTALL)
            message = assistant_message(content, functions)
            pending = message.get("tool_calls", [])
            for index, call in enumerate(pending):
                call["id"] = f"call_{len(history)}_{index}"
        elif message["role"] == "user" and content.startswith("<tool_response>"):
            outputs = re.findall(r"<tool_response>\n?(.*?)\n?</tool_response>", content, re.DOTALL)
            if len(outputs) != len(pending):
                raise ValueError("Tool response count does not match preceding calls")
            for output, call in zip(outputs, pending):
                history.append({"role": "tool", "name": call["function"]["name"],
                                "tool_call_id": call["id"], "content": output})
            pending = []
            continue
        history.append(message)
    return {"messages": history, "functions": functions}


def row_context(row):
    context = row.get("_render_context")
    if context is not None:
        return copy.deepcopy(context)
    if "messages" in row and ("functions" in row or "tools" in row):
        functions = row.get("functions")
        if functions is None:
            functions = [tool.get("function", tool) for tool in row["tools"]]
        return {"messages": copy.deepcopy(row["messages"]), "functions": functions}
    if row.get("prompt", "").startswith("<|im_start|>"):
        return _chatml_context(row["prompt"])
    # Original bfcl_demo_pool rows contain prompting-mode instructions in
    # `messages`. Recover the native-FC question and declarations by task id,
    # never pass that obsolete instruction block to Gemma.
    task_id = row.get("task_id", "")
    if int(row.get("turn_index", 0)) != 0 or task_id.startswith(("multi_turn", "memory", "web_search")):
        raise ValueError("Stateful rows require explicit context or a Qwen FC prompt")
    category = task_id.rsplit("_", 1)[0]
    path = BFCL / "bfcl_eval/data" / f"BFCL_v4_{category}.json"
    if path.is_file():
        for line in path.read_text().splitlines():
            entry = json.loads(line)
            if entry.get("id") == task_id:
                return {"messages": entry["question"][0], "functions": entry["function"]}
    raise ValueError(f"Cannot recover native FC context for {task_id!r}")


def verify_row(row, tokenizer):
    context = row["_render_context"]
    prompt, target = render_pair(
        tokenizer, context["messages"], context["functions"], row["_gemma4_assistant"]
    )
    if (row["prompt"], row["response"]) != (prompt, target):
        raise AssertionError(f"Gemma template mismatch for {row.get('task_id')}")
    expected = row["_gemma4_assistant"].get("tool_calls", [])
    decoded = Gemma4FCHandler._extract_tool_calls(target)
    if decoded != [call["function"] for call in expected]:
        raise AssertionError("Gemma native tool calls do not round-trip")


def render_row(row, tokenizer, *, verify=False):
    if row.get("_row_format") == FORMAT:
        verify_row(row, tokenizer)
        return copy.deepcopy(row)
    context = row_context(row)
    assistant = assistant_message(row["response"], context["functions"])
    prompt, target = render_pair(tokenizer, context["messages"], context["functions"], assistant)
    result = copy.deepcopy(row)
    # The trainer retemplates `messages` without tools if present. Keep the
    # lossless context under metadata and train on the exact serialized prompt.
    result.pop("messages", None)
    result.update(prompt=prompt, response=target, _row_format=FORMAT,
                  _render_context=context, _gemma4_assistant=assistant,
                  _native_fc_empty_response=not assistant.get("tool_calls") and not assistant["content"],
                  token_hint=len(tokenizer.encode(target, add_special_tokens=False)))
    if "_rejected" in result:
        rejected = assistant_message(result["_rejected"], context["functions"])
        _, result["_rejected"] = render_pair(tokenizer, context["messages"], context["functions"], rejected)
    if verify:
        verify_row(result, tokenizer)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", "--input", dest="input", required=True, type=Path)
    parser.add_argument("--out", "--output", dest="output", required=True, type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    rows = []
    with args.input.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if line.strip():
                try:
                    rows.append(render_row(json.loads(line), tokenizer, verify=args.verify))
                except (ValueError, AssertionError, KeyError) as exc:
                    raise ValueError(f"{args.input}:{number}: {exc}") from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(f"Rendered {len(rows)} Gemma 4 rows to {args.output}" + (" (verified)" if args.verify else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
