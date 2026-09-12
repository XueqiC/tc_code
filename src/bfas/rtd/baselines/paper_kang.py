"""Inference-only Kang SAG hooks around the existing official handlers.

All candidates come from the local student. Probe outcomes never use reference
answers or success labels; only the chosen action reaches the live environment.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import re
import threading
import time
import uuid


def consistency_vote(candidates, execute_key):
    if not candidates:
        raise ValueError("at least one candidate required")
    keys = [execute_key(candidate) for candidate in candidates]
    counts = Counter(k for k in keys if k is not None)
    if not counts:
        return candidates[0], dict(keys=keys, selected=0, all_invalid=True)
    winner = max((i for i, key in enumerate(keys) if key is not None),
                 key=lambda i: (counts[keys[i]], -i))
    return candidates[winner], dict(keys=keys, selected=winner, all_invalid=False)


def observable_key(state):
    """Public action outcome, deliberately excluding won/reward/reference labels."""
    return json.dumps(dict(observation=state["observation"],
                           admissible=sorted(state["admissible"]), done=state["done"]), sort_keys=True)


def alfworld_probe(task_id, split, commands, current, reply, *, bridge_factory, parse):
    command = parse(reply, current["admissible"])
    if command not in current["admissible"]:
        return None
    bridge = bridge_factory(split, task_id)
    try:
        state = bridge._read()
        for prior in commands:
            state = bridge.step(prior)
        if observable_key(state) != observable_key(current):
            raise ValueError("ALFWorld SAG replay did not reproduce the live state")
        return observable_key(bridge.step(command))
    finally:
        bridge.close()


def install_alfworld_kang(adapter, audit_path, *, n=3):
    from ...adapters.alfworld import _EnvBridge
    import urllib.request
    lock = threading.Lock()

    def select(prompt, temperature, *, task_id, split, commands, state):
        seed = int(hashlib.sha256(f"0:{task_id}:{len(commands)}".encode()).hexdigest()[:8], 16)
        payload = dict(model="bfas-policy", prompt=prompt, max_tokens=256, temperature=.7,
                       n=n, seed=seed, add_special_tokens=False, skip_special_tokens=False)
        req = urllib.request.Request(f"http://127.0.0.1:{adapter.port}/v1/completions",
            data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as response:
            candidates = [c["text"] for c in json.loads(response.read())["choices"]]
        if len(candidates) != n:
            raise ValueError("incomplete Kang candidate group")
        reply, audit = consistency_vote(candidates, lambda text: alfworld_probe(
            task_id, split, commands, state, text, bridge_factory=_EnvBridge,
            parse=adapter._teacher_command))
        with lock, open(audit_path, "a") as stream:
            stream.write(json.dumps(dict(task_id=task_id, step=len(commands), seed=seed,
                candidates=candidates, **audit)) + "\n")
        return reply
    adapter.select_student_reply = select


def bfcl_execution_key(handler, entry, text):
    """Execute isolated copies of the current official multi-turn objects.

    Schema-only and external/file-backed tasks have no cloneable execution
    sandbox: vote on the official decoded AST there, explicitly labeled in
    each vote key. No fabricated function implementation or correctness vote.
    """
    try:
        calls = handler.decode_execute(text, has_tool_call_tag=False)
        ast = handler.decode_ast(text, language="Python", has_tool_call_tag=False)
    except (ValueError, SyntaxError, KeyError, TypeError):
        return None
    if not calls:
        return "final:" + text.strip()
    classes = entry.get("involved_classes", [])
    if not classes or entry["id"].startswith(("memory", "web_search")):
        return "decoded_ast:" + json.dumps(ast, sort_keys=True)
    from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils as execution
    model = handler.model_name_underline_replaced
    probe_model = "baseline_probe_" + uuid.uuid4().hex
    names = []
    try:
        for cls in classes:
            original = re.sub(r"[-./:]", "_", f"{model}_{entry['id']}_{cls}_instance")
            probe = re.sub(r"[-./:]", "_", f"{probe_model}_{entry['id']}_{cls}_instance")
            if original not in vars(execution):
                raise ValueError("missing current official execution state")
            setattr(execution, probe, deepcopy(getattr(execution, original)))
            names.append(probe)
        results, _ = execution.execute_multi_turn_func_call(calls, entry.get("initial_config", {}),
            classes, probe_model, entry["id"], long_context=("long_context" in entry["id"] or
                                                          "composite" in entry["id"]))
        if any(r.startswith("Error during execution:") for r in results):
            return None
        return "executed:" + json.dumps(results, sort_keys=True)
    finally:
        for name in names:
            delattr(execution, name)


def install_bfcl_kang(audit_path, *, n=3):
    """Register a local subclass; leave the installed harness files untouched."""
    # overrides resolves superclass names in module globals, not closure locals.
    # Keep this import lazy so CPU tests do not need the BFCL installation.
    global Gemma4FCHandler
    from dataclasses import replace
    from overrides import override
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
    from bfcl_eval.model_handler.local_inference.gemma4_fc import Gemma4FCHandler
    lock = threading.Lock()
    local = threading.local()

    class KangGemma4(Gemma4FCHandler):
        @override
        def inference(self, test_entry, include_input_log, exclude_state_log):
            local.entry = {k: deepcopy(test_entry[k]) for k in ("id", "initial_config", "involved_classes")
                           if k in test_entry}
            local.step = 0
            return super().inference(test_entry, include_input_log, exclude_state_log)

        @override
        def _query_prompting(self, inference_data):
            prompt = self._format_prompt(inference_data["message"], inference_data["function"])
            inference_data["inference_input_log"] = {"formatted_prompt": prompt}
            room = self.max_context_length-len(self.tokenizer.encode(prompt, add_special_tokens=False))-2
            if room < 1:
                raise ValueError("Kang prompt exceeds official context")
            seed = int(hashlib.sha256(f"0:{local.entry['id']}:{local.step}".encode()).hexdigest()[:8], 16)
            start = time.monotonic()
            response = self.client.completions.create(model=self.model_path_or_id,
                prompt=prompt, temperature=.7, n=n, seed=seed, max_tokens=min(4096, room),
                extra_body=dict(add_special_tokens=False, skip_special_tokens=False,
                    stop_token_ids=[self.tokenizer.convert_tokens_to_ids(t)
                                    for t in ("<eos>", "<turn|>", "<|tool_response>")]), timeout=72000)
            if len(response.choices) != n:
                raise ValueError("incomplete Kang candidate group")
            candidates = [c.text for c in response.choices]
            _, audit = consistency_vote(candidates, lambda text: bfcl_execution_key(self, local.entry, text))
            response.choices = [response.choices[audit["selected"]]]
            with lock, open(audit_path, "a") as stream:
                stream.write(json.dumps(dict(task_id=local.entry["id"], step=local.step,
                    seed=seed, candidates=candidates, **audit)) + "\n")
            local.step += 1
            return response, time.monotonic()-start

    name = "google/gemma-4-12B-it-FC"
    MODEL_CONFIG_MAPPING[name] = replace(MODEL_CONFIG_MAPPING[name], model_handler=KangGemma4)
