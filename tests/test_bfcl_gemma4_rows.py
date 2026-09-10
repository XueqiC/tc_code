"""Fidelity against the cached tokenizer and the actual deployed Gemma handler."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import bfcl_pool_render_gemma4 as renderer


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    return AutoTokenizer.from_pretrained(renderer.MODEL, local_files_only=True)


@pytest.fixture
def context():
    return {"messages": [{"role": "system", "content": "Answer the user."},
                         {"role": "user", "content": "Find Zürich and 東京."}],
            "functions": [{"name": "places.find", "description": "Find a place",
                           "parameters": {"type": "dict", "properties": {
                               "query": {"type": "string", "description": "Search text"},
                               "limit": {"type": "integer", "description": "Result limit"},
                           }, "required": ["query"]}}]}


def row_for(context, response):
    return {"task_id": "parallel_0", "teacher": "azure/gpt-5.4-FC", "turn_index": 0,
            "messages": context["messages"], "prompt": "obsolete", "response": response,
            "token_hint": 1, "_render_context": context, "_custom": {"keep": True}}


@pytest.mark.parametrize("encoding", ["bfcl", "openai", "qwen"])
def test_parallel_calls_template_suffix_and_evaluation_request(tokenizer, context, encoding):
    arguments = [{"query": 'Zürich "quoted" \\path\n雪', "nested": {"a": True, "b": None},
                  "array": [False, 1, 2.5, {"key": "value"}]}, {"query": "東京", "limit": 3}]
    calls = [{"type": "function", "function": {"name": "places_find", "arguments": args}}
             for args in arguments]
    if encoding == "bfcl":
        response = json.dumps([{"places_find": json.dumps(args)} for args in arguments])
    elif encoding == "openai":
        response = {"role": "assistant", "tool_calls": calls}
    else:
        response = "\n".join("<tool_call>\n" + json.dumps(call["function"]) + "\n</tool_call>" for call in calls)
    original = row_for(context, response)
    snapshot = copy.deepcopy(original)
    row = renderer.render_row(original, tokenizer, verify=True)
    assert original == snapshot
    assert "messages" not in row  # trainer must use prompt bytes, including tools
    assert row["_custom"] == original["_custom"]
    assert row["response"].endswith("<tool_call|><|tool_response>")
    assert row["response"].count("<|tool_call>call:places.find{") == 2
    assert '<|"|>Zürich "quoted" \\path\n雪<|"|>' in row["response"]
    assert renderer.Gemma4FCHandler._extract_tool_calls(row["response"]) == [
        {"name": "places.find", "arguments": args} for args in arguments]

    # Independently obtain schemas from the real evaluation handler and render
    # a completed assistant directly with the real HF template.
    captured = {}

    class CaptureTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            captured.update(kwargs)
            return tokenizer.apply_chat_template(messages, **kwargs)

    h = object.__new__(renderer.Gemma4FCHandler)
    h.tokenizer = CaptureTokenizer()
    assert h._format_prompt(context["messages"], context["functions"]).encode() == row["prompt"].encode()
    expected_full = tokenizer.apply_chat_template(
        [*context["messages"], row["_gemma4_assistant"]], tools=captured["tools"],
        tokenize=False, add_generation_prompt=False,
    )
    assert row["prompt"].endswith(renderer.EMPTY_THOUGHT)
    assert row["prompt"][:-len(renderer.EMPTY_THOUGHT)] + row["response"] == expected_full

    # Exercise the actual raw-Completions evaluation request, not just _format_prompt.
    requests = []
    h.tokenizer = tokenizer
    h.max_context_length = 8192
    h.model_path_or_id = renderer.MODEL
    h.temperature = 0
    h.client = SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: requests.append(kwargs)))
    h._query_prompting({"message": context["messages"], "function": context["functions"]})
    assert requests[0]["prompt"].encode() == row["prompt"].encode()
    assert requests[0]["extra_body"]["add_special_tokens"] is False


@pytest.mark.parametrize("response", ["No suitable tool. 雪", '{"answer":"No tool"}', '["plain", "text"]', "", "[]", []])
def test_no_call_uses_text_and_template_terminator(tokenizer, context, response):
    row = renderer.render_row(row_for(context, response), tokenizer, verify=True)
    expected = "" if response in ("", "[]", []) else response
    assert row["response"] == expected + "<turn|>\n"
    assert row["_native_fc_empty_response"] is (expected == "")
    assert "<|tool_call>" not in row["response"]


def test_history_after_tool_result(tokenizer, context):
    context["messages"].extend([
        {"role": "assistant", "content": "", "tool_calls": [{"id": "old", "type": "function",
          "function": {"name": "places.find", "arguments": {"query": "Zürich"}}}]},
        {"role": "tool", "tool_call_id": "old", "name": "places.find", "content": "Found Zürich"},
    ])
    row = renderer.render_row(row_for(context, [{"places_find": '{"query":"東京"}'}]), tokenizer, verify=True)
    assert "Found Zürich" in row["prompt"]
    assert row["response"].startswith("<|tool_call>call:places.find{")


def test_verification_detects_changed_prompt_and_target(tokenizer, context):
    row = renderer.render_row(row_for(context, [{"places_find": '{"query":"Zürich"}'}]), tokenizer)
    assert renderer.render_row(row, tokenizer, verify=True) == row
    for field in ("prompt", "response"):
        damaged = copy.deepcopy(row)
        damaged[field] += " "
        with pytest.raises(AssertionError, match="template mismatch"):
            renderer.verify_row(damaged, tokenizer)


def test_legacy_qwen_context_and_cli(tokenizer, context, tmp_path):
    from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler

    row = row_for(context, '<tool_call>\n{"name":"places_find","arguments":{"query":"東京"}}\n</tool_call>')
    row.pop("_render_context")
    row.pop("messages")
    row["prompt"] = QwenFCHandler._format_prompt(None, context["messages"], context["functions"])
    inp, out = tmp_path / "pool.jsonl", tmp_path / "gemma.jsonl"
    inp.write_text(json.dumps(row, ensure_ascii=False) + "\n")
    assert renderer.main(["--in", str(inp), "--out", str(out), "--verify"]) == 0
    converted = json.loads(out.read_text())
    assert converted["_render_context"] == context
    renderer.verify_row(converted, tokenizer)


def test_malformed_and_unknown_calls_fail(tokenizer, context):
    for response in ('<tool_call>{broken}</tool_call>', [{"unknown": "{}"}],
                     [{"places_find": "[]"}], '<tool_call>{"name":"places_find"}'):
        with pytest.raises(ValueError):
            renderer.render_row(row_for(context, response), tokenizer, verify=True)


def test_whole_episode_requires_per_step_context(tokenizer, context):
    with pytest.raises(ValueError, match="Whole-episode"):
        renderer.render_row(row_for(context, '[[{"places_find":"{}"}]]'), tokenizer)
