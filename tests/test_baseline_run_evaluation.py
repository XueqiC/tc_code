"""Evaluation dispatch and execute-consistency with tiny environments."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace, ModuleType
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT)]
from bfas.rtd.baselines.paper_evaluation import bfcl_commands, protocol
from bfas.rtd.baselines.paper_kang import alfworld_probe, consistency_vote, bfcl_execution_key


def test_vote_uses_execution_equivalence_not_string_or_reward():
    outputs = dict(a="state1", b="state2", c="state2", invalid=None)
    selected, audit = consistency_vote(["a", "b", "c", "invalid"], outputs.get)
    assert selected == "b" and audit["selected"] == 1
    assert consistency_vote(["a", "b"], outputs.get)[0] == "a"
    assert consistency_vote(["bad", "invalid"], outputs.get)[0] == "bad"


def test_alfworld_shadow_replay_never_mutates_live_or_votes_on_success():
    current = dict(observation="step1", admissible=["a", "b"], done=False, won=False)
    saved = deepcopy(current)
    bridges = []
    class Bridge:
        def __init__(self, split, task):
            self.closed, self.commands = False, []
            bridges.append(self)
        def _read(self):
            return dict(current, observation="reset")
        def step(self, command):
            self.commands.append(command)
            return dict(current, observation="step1" if len(self.commands) == 1 else "outcome", won=command == "b")
        def close(self):
            self.closed = True
    def probe(reply):
        return alfworld_probe("task", "valid_seen", ["prefix"], current, reply,
                              bridge_factory=Bridge, parse=lambda text, _: text)
    assert probe("a") == probe("b")  # same public result, different hidden success
    assert probe("invalid") is None
    assert current == saved
    assert all(b.closed for b in bridges)
    assert [b.commands for b in bridges] == [["prefix", "a"], ["prefix", "b"]]


def test_bfcl_commands_pin_official_gemma_full_and_separate_sag(tmp_path):
    for kang in (False, True):
        generate, evaluate, env = bfcl_commands(ROOT, tmp_path/"model", tmp_path/"out", kang=kang)
        assert generate[generate.index("--model")+1] == "google/gemma-4-12B-it-FC"
        assert generate[generate.index("--temperature")+1] == "0.001"
        assert generate[generate.index("--test-category")+1] == "all"
        assert generate[generate.index("--backend")+1] == "vllm"
        assert ("BASELINE_KANG_AUDIT" in env) is kang
        assert evaluate[evaluate.index("--score-dir")+1] == str(tmp_path/"out/scoredir")
    assert protocol("bfcl")["tasks"] == 5217
    assert protocol("alfworld") == dict(benchmark="alfworld", split="valid_seen", tasks=140,
        temperature=0., max_steps=40, max_action_tokens=256, backend="vllm", evaluator="ALFWorldAdapter.evaluate")


def test_bfcl_uses_official_overall_csv_and_rejects_missing_generation(tmp_path, monkeypatch):
    from bfas.rtd import evaluation
    from bfas.rtd.baselines.paper_evaluation import bfcl_metrics
    ids = [f"simple_python_{i}" for i in range(5217)]
    expected = dict(generation={"simple_python": ids}, scoring={"simple_python": ids})
    monkeypatch.setattr(evaluation, "official_expectations", lambda root: expected)
    (tmp_path/"resultdir").mkdir()
    (tmp_path/"scoredir").mkdir()
    result = tmp_path/"resultdir/BFCL_v4_simple_python_result.json"
    result.write_text("\n".join(json.dumps(dict(id=tid)) for tid in ids))
    (tmp_path/"scoredir/BFCL_v4_simple_python_score.json").write_text(
        json.dumps(dict(correct_count=5216, total_count=5217))+"\n"+json.dumps(dict(id=ids[-1], valid=False)))
    (tmp_path/"scoredir/data_overall.csv").write_text("Model,Overall Acc\ngemma,41.23\n")
    metrics = bfcl_metrics(ROOT, tmp_path)
    assert metrics["overall_accuracy_percent"] == 41.23  # never invent an unweighted mean
    assert metrics["per_category"]["simple_python"]["total"] == 5217
    result.write_text("\n".join(json.dumps(dict(id=tid)) for tid in ids[:-1]))
    with pytest.raises(ValueError, match="incomplete evaluation generation"):
        bfcl_metrics(ROOT, tmp_path)


def test_bfcl_schema_vote_uses_official_ast_and_handles_invalid():
    class Handler:
        def decode_execute(self, text, **kwargs):
            if text == "bad":
                raise ValueError("malformed")
            return ["f(x=1)"]
        def decode_ast(self, text, **kwargs):
            return [{"f": {"x": 1}}]
    h = Handler()
    assert bfcl_execution_key(h, {"id": "simple_1"}, "different raw text").startswith("decoded_ast:")
    assert bfcl_execution_key(h, {"id": "simple_1"}, "bad") is None


def test_bfcl_execution_clones_and_cleans_official_objects(monkeypatch):
    package = "bfcl_eval.eval_checker.multi_turn_eval"
    module = ModuleType(package)
    execution = SimpleNamespace(model_task_Counter_instance=SimpleNamespace(value=2))
    def execute(calls, initial, classes, model, task, **kwargs):
        obj = getattr(execution, f"{model}_{task}_Counter_instance")
        obj.value += 1
        return [str(obj.value)], {"Counter": obj}
    execution.execute_multi_turn_func_call = execute
    module.multi_turn_utils = execution
    monkeypatch.setitem(sys.modules, package, module)
    h = SimpleNamespace(model_name_underline_replaced="model",
        decode_execute=lambda *a, **kw: ["increment()"], decode_ast=lambda *a, **kw: [{"increment": {}}])
    result = bfcl_execution_key(h, {"id": "task", "involved_classes": ["Counter"]}, "call")
    assert result == 'executed:["3"]'
    assert execution.model_task_Counter_instance.value == 2
    assert not any(k.startswith("baseline_probe") for k in vars(execution))


def test_kang_bfcl_hook_samples_three_and_returns_voted_raw_response(tmp_path, monkeypatch):
    from dataclasses import dataclass
    from bfas.rtd.baselines import paper_kang
    config_module = ModuleType("bfcl_eval.constants.model_config")
    handler_module = ModuleType("bfcl_eval.model_handler.local_inference.gemma4_fc")
    @dataclass
    class Config:
        model_handler: object
    requested = []
    def completions(**kwargs):
        requested.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(text=x) for x in ("a", "b", "b")])
    class Gemma:
        def inference(self, test_entry, include_input_log, exclude_state_log):
            return self._query_prompting(dict(message=[], function=[]))
        def _query_prompting(self, inference_data):
            raise AssertionError("Kang must install the candidate query")
        def _format_prompt(self, messages, functions):
            return "native"
        def decode_execute(self, text, **kwargs):
            return [text+"()"]
        def decode_ast(self, text, **kwargs):
            return [{text: {}}]
    name = "google/gemma-4-12B-it-FC"
    config_module.MODEL_CONFIG_MAPPING = {name: Config(Gemma)}
    handler_module.Gemma4FCHandler = Gemma
    monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
    monkeypatch.setitem(sys.modules, handler_module.__name__, handler_module)
    path = tmp_path/"votes.jsonl"
    paper_kang.install_bfcl_kang(path)
    handler = config_module.MODEL_CONFIG_MAPPING[name].model_handler()
    handler.max_context_length = 100
    handler.model_path_or_id = "local"
    handler.tokenizer = SimpleNamespace(encode=lambda *a, **kw: [1], convert_tokens_to_ids=lambda _: 2)
    handler.client = SimpleNamespace(completions=SimpleNamespace(create=completions))
    response, elapsed = handler.inference(dict(id="simple_0"), False, False)
    assert response.choices[0].text == "b"
    assert requested[0]["n"] == 3 and requested[0]["temperature"] == .7
    audit = json.loads(path.read_text())
    assert audit["selected"] == 1 and audit["keys"][1].startswith("decoded_ast:")


def test_official_alfworld_metrics_and_dispatch_use_existing_adapter(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from bfas.rtd.baselines import paper_evaluation
    from bfas.rtd.benchmarks import alfworld_identity
    from bfas.adapters import alfworld
    from bfas import run
    calls = []
    ids = [f"task{i}" for i in range(140)]
    class Adapter:
        def __init__(self, seed, port):
            assert seed == 0
        def prepare_renderer(self, model):
            calls.append("renderer")
        def evaluate(self, model, out):
            import os
            assert os.environ["BFAS_ALFWORLD_EVAL_SPLIT"] == "valid_seen"
            assert os.environ["BFAS_ALFWORLD_STUDENT_REACT"] == "1"
            (out/"records.jsonl").write_text("\n".join(json.dumps(dict(task_id=t, won=True, steps=3)) for t in ids))
            return dict(success_rate=1., per_category={"fake": 1.})
        def release_policy(self):
            calls.append("release")
    monkeypatch.setattr(alfworld, "ALFWorldAdapter", Adapter)
    monkeypatch.setattr(alfworld_identity, "official_expectations", lambda _: dict(task_ids=ids))
    monkeypatch.setattr(run, "serving_lane", lambda *a, **kw: nullcontext())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    result = paper_evaluation.run_alfworld(ROOT, tmp_path/"model", tmp_path/"eval", kang=False, port=1)
    assert result["complete"] and result["overall_accuracy_percent"] == 100
    assert result["per_category"]["fake"]["accuracy_percent"] == 100
    assert calls == ["renderer", "release"]
