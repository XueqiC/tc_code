"""Native rendering, official AST checks, and isolated executor comparisons."""
import copy
import importlib
import re
import uuid

from .common import REGISTRY, digest, setup_harness
from .harness import NoThinking, unpack

FORBIDDEN = ("<eos>", "<turn|>", "<|im_end|>", "<|endoftext|>", "<|tool_response>")
VARIANTS = {"condition", "surface", "neighbour_correct"}


def exercise_fingerprint(exercise):
    """Identify the same supervision in the same state, ignoring audit labels."""
    return digest(dict(messages=exercise["messages"], functions=exercise["functions"],
                       snapshot=exercise.get("snapshot"),
                       involved_classes=exercise.get("involved_classes", []),
                       demonstration=demonstration(exercise["demo"])))


def demonstration(value):
    if not isinstance(value, dict) or value.get("kind") not in {"call", "abstain"}:
        raise ValueError("Demonstration requires kind=call|abstain")
    if any(token in str(value) for token in FORBIDDEN):
        raise ValueError("Demonstration contains an authored end-of-task/control token")
    if value["kind"] == "abstain":
        if not isinstance(value.get("text"), str) or not value["text"].strip() or value.get("calls"):
            raise ValueError("Abstention requires short natural text and no calls")
        return {"role": "assistant", "content": value["text"]}
    calls = value.get("calls")
    if not isinstance(calls, list) or not 1 <= len(calls) <= 4:
        raise ValueError("A short demonstration requires 1–4 calls")
    for c in calls:
        if set(c) != {"name", "arguments"} or not isinstance(c["arguments"], dict):
            raise ValueError("Calls require name and arguments object")
        if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", c["name"]):
            raise ValueError("Invalid function name")
    return dict(role="assistant", content="", tool_calls=[
        dict(id=f"call_{i}", type="function", function=c) for i, c in enumerate(calls)])


def native_pair(tokenizer, exercise):
    from tools.bfcl_pool_render_gemma4 import render_pair
    prompt, target = render_pair(NoThinking(tokenizer), exercise["messages"],
                                exercise["functions"], demonstration(exercise["demo"]))
    # A local supervision boundary is not an end-of-task target. Only remove
    # native trailing turn/handoff markers; never add EOS to a short exercise.
    target = target.rstrip()
    for token in ("<eos>", "<turn|>", "<|tool_response>"):
        if target.endswith(token):
            target = target[:-len(token)].rstrip()
    if any(t in target for t in FORBIDDEN):
        raise ValueError("Unexpected control marker inside the supervision span")
    if not target:
        raise ValueError("Empty supervised span")
    return prompt, target


def materialize(proposal, context, *, arm, group, index, call_id, layer=0):
    """Whitelist authored fields; never render diagnosis/validation metadata."""
    if not isinstance(proposal, dict):
        raise ValueError("Exercise must be an object")
    allowed = {"user", "demo", "variant", "evidence_frame", "possible_answer", "state_index"}
    if set(proposal) - allowed:
        raise ValueError("Unexpected teacher fields (history, tools, and state cannot be replaced)")
    messages = copy.deepcopy(context["messages"])
    if layer != 2:
        user = proposal.get("user")
        if not isinstance(user, str) or not user.strip():
            raise ValueError("Local exercise requires a student-visible user task")
        if any(t in user for t in FORBIDDEN) or re.search(r"\b(diagnosis|gap hypothesis|student error)\b", user, re.I):
            raise ValueError("Diagnosis or control tokens leaked into student task")
        # Keep the real prefix and start state. A variant may edit only the
        # final user request; it cannot manufacture tools or past observations.
        if not messages or messages[-1]["role"] != "user":
            messages.append(dict(role="user", content=user))
        else:
            messages[-1]["content"] = user
    elif "user" in proposal:
        raise ValueError("Natural continuation history must remain byte-identical")
    demonstration(proposal["demo"])
    return dict(id=f"{group}:{index}", parent_id=context["parent_id"],
                task_id=context["task_id"], category=context["category"],
                generation_group=group, arm=arm, layer=layer, cost_call_id=call_id,
                source_frame=context["frame_id"], source_hash=digest(context),
                messages=messages, functions=copy.deepcopy(context["functions"]),
                snapshot=copy.deepcopy(context.get("snapshot")),
                snapshot_error=context.get("snapshot_error"),
                involved_classes=context.get("involved_classes", []),
                demo=proposal["demo"], variant=proposal.get("variant", "ordinary"),
                evidence_frame=proposal.get("evidence_frame"),
                possible_answer=proposal.get("possible_answer"))


class OfficialValidator:
    def __init__(self):
        setup_harness()
        from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
        from bfcl_eval.constants.enums import Language
        from bfcl_eval.model_handler.local_inference.gemma4_fc import _parse_response
        self.ast_checker, self.Language, self.parse = ast_checker, Language, _parse_response

    def ast(self, exercise, calls, possible=None):
        if not calls:
            return {"valid": True, "method": "official Gemma decoder: no tool call"}
        names = {f["name"] for f in exercise["functions"]}
        if any(c["name"] not in names for c in calls):
            return {"valid": False, "error": "Undeclared tool"}
        possible = possible or [{c["name"]: {k: [v] for k, v in c["arguments"].items()}} for c in calls]
        # The official checker accepts Python representations of calls even
        # for Java/JS tools; use each source category's declared type language.
        lang = self.Language.JAVA if "java" in exercise["category"] and "javascript" not in exercise["category"] else (
            self.Language.JAVASCRIPT if "javascript" in exercise["category"] else self.Language.PYTHON)
        return self.ast_checker(copy.deepcopy(exercise["functions"]),
            [{c["name"]: c["arguments"]} for c in calls], possible,
            lang, "parallel_multiple" if len(possible) > 1 else "multiple", REGISTRY)

    def execute(self, exercise, calls):
        from bfcl_eval.constants.executable_backend_config import CLASS_FILE_PATH_MAPPING
        from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils as util
        from bfcl_eval.model_handler.utils import convert_to_function_call
        if not exercise.get("involved_classes"):
            return None
        if exercise.get("snapshot_error") or exercise.get("snapshot") is None:
            raise ValueError("Executor snapshot unavailable")
        if "WebSearchAPI" in exercise["involved_classes"]:
            raise ValueError("Live web executor excluded from local deterministic validation")
        model, tid = "mech_local", uuid.uuid4().hex
        keys = []
        try:
            for cls in exercise["involved_classes"]:
                impl = getattr(importlib.import_module(CLASS_FILE_PATH_MAPPING[cls]), cls)
                instance = impl.__new__(impl)
                instance.__dict__.update(unpack(copy.deepcopy(exercise["snapshot"][cls])))
                key = f"{model}_{tid}_{cls}_instance"
                keys.append(key)
                setattr(util, key, instance)
            code = convert_to_function_call([{c["name"]: c["arguments"]} for c in calls])
            # Only declared functions with JSON-literal arguments reach the
            # official executor, never arbitrary teacher-generated Python.
            outputs, instances = util.execute_multi_turn_func_call(
                code, {}, exercise["involved_classes"], model, tid)
            return outputs, copy.deepcopy(instances)
        finally:
            for key in keys:
                delattr(util, key)

    def validate(self, exercise):
        unavailable = ["Task semantics are teacher-authored; AST self-consistency is not independent semantic verification"]
        try:
            demonstration(exercise["demo"])
            calls = exercise["demo"].get("calls", [])
            ast_result = self.ast(exercise, calls, exercise.get("possible_answer"))
            if not ast_result["valid"]:
                return dict(valid=False, ast=ast_result, unvalidated=unavailable)
            execution = self.execute(exercise, calls)
            outputs = execution[0] if execution else None
            if outputs is not None and any("error" in o.lower() for o in outputs):
                return dict(valid=False, ast=ast_result, execution_outputs=outputs,
                            unvalidated=unavailable, reason="Official executor returned an error")
            if execution is None:
                unavailable.append("No official executable backend for this tool schema")
            if not calls:
                unavailable.append("Abstention text semantics; only no-call decision is scored")
            return dict(valid=True, ast=ast_result, execution_outputs=outputs,
                        executable_consistency=execution is not None, unvalidated=unavailable)
        except (ValueError, KeyError, TypeError, AssertionError) as exc:
            return dict(valid=False, reason=str(exc), unvalidated=unavailable)

    def score(self, exercise, response):
        try:
            calls, _, _, _ = self.parse(response)
            # The decoder otherwise treats arbitrary malformed text as a
            # no-call response, which must not pass abstention exercises.
            if "<|tool_call>" in response and not calls:
                raise ValueError("Malformed tool call")
            format_check = self.ast(exercise, calls)
            if not format_check["valid"]:
                return dict(correct=False, illegal_actions=1, tool_calls=len(calls), checker=format_check)
            target = exercise["demo"].get("calls", [])
            if not target or not calls:
                correct = not target and not calls and bool(response.strip())
                return dict(correct=correct, illegal_actions=0, tool_calls=len(calls),
                            early_stops=int(bool(target) and not calls), checker="no-call decision")
            actual, expected = self.execute(exercise, calls), self.execute(exercise, target)
            if actual is not None:
                from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import state_checker, response_checker
                illegal = int(any("error" in o.lower() for o in actual[0]))
                check = dict(state=state_checker(actual[1], expected[1]),
                             response=response_checker(actual[0], expected[0], 0))
                correct = not illegal and all(c["valid"] for c in check.values())
            else:
                possible = exercise.get("possible_answer") or [
                    {c["name"]: {k: [v] for k, v in c["arguments"].items()}} for c in target]
                check = self.ast(exercise, calls, possible)
                correct, illegal = check["valid"], 0
            return dict(correct=bool(correct), tool_calls=len(calls), illegal_actions=illegal,
                        early_stops=0, checker=check)
        except (ValueError, KeyError, TypeError, AssertionError) as exc:
            return dict(correct=False, illegal_actions=1, tool_calls=0, early_stops=0, checker=str(exc))
