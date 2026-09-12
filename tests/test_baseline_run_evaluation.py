"""Evaluation dispatch and execute-consistency with tiny environments."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace, ModuleType
import json
import shlex
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT)]
from bfas.rtd.baselines.paper_evaluation import bfcl_commands, protocol
from bfas.rtd.baselines.paper_kang import alfworld_probe, consistency_vote, bfcl_execution_key


def test_alfworld_vllm_command_uses_hub_id_and_absolute_lora_and_logs_before_launch(tmp_path, monkeypatch):
    """Exercise the real command construction; block Popen before any server/API work."""
    from contextlib import nullcontext, redirect_stdout
    from bfas import run
    from bfas.adapters import alfworld
    from bfas.rtd.baselines import paper_evaluation
    from bfas.rtd.benchmarks import alfworld_identity
    monkeypatch.chdir(tmp_path)
    lora = Path("run with spaces/checkpoint/lora")
    lora.mkdir(parents=True)
    model = "google/gemma-4-12B-it"
    events = []
    class Adapter:
        name = "alfworld"
        def __init__(self, **kwargs):
            pass
        def prepare_renderer(self, policy):
            assert policy == model
        def release_policy(self):
            events.append("release")
    log_path = tmp_path/"evaluate.log"
    def spawn(command, **kwargs):
        events.append("launch")
        assert command[:3] == [str(ROOT/"envs/vllm-serve/.venv/bin/vllm"), "serve", model]
        assert command[command.index("--served-model-name")+1] == model
        assert "--enable-lora" in command
        assert command[command.index("--max-lora-rank")+1] == "16"
        name, path = command[command.index("--lora-modules")+1].split("=", 1)
        assert name == alfworld.SERVER_MODEL_NAME == "bfas-policy"
        assert Path(path).is_absolute() and Path(path) == lora.resolve()
        assert command[command.index("--port")+1] == "12345"
        assert kwargs["env"]["HF_HUB_OFFLINE"] == kwargs["env"]["TRANSFORMERS_OFFLINE"] == "1"
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["start_new_session"] is True
        # Read the file at the launch boundary to verify the command was flushed.
        logged = log_path.read_text().removeprefix("vLLM server command: ").strip()
        assert shlex.split(logged) == command
        raise RuntimeError("CPU test blocked server launch")
    monkeypatch.setattr(alfworld, "ALFWorldAdapter", Adapter)
    monkeypatch.setattr(alfworld_identity, "official_expectations",
                        lambda _: dict(task_ids=[f"task{i}" for i in range(140)]))
    monkeypatch.setattr(run, "PortRegistry", lambda _: nullcontext())
    monkeypatch.setattr(run.subprocess, "Popen", spawn)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with log_path.open("w") as log, redirect_stdout(log), pytest.raises(RuntimeError, match="CPU test blocked"):
        paper_evaluation.run_alfworld(ROOT, lora, tmp_path/"eval", kang=False, port=12345, smoke=True)
    assert events == ["launch", "release"]
    assert (tmp_path/"eval/gpu_usage.json").exists()


@pytest.mark.parametrize("smoke", [False, True])
@pytest.mark.parametrize("method", ["smartad", "kang"])
def test_alfworld_dispatch_serves_checkpoint_lora_without_merge(tmp_path, monkeypatch, smoke, method):
    from bfas.rtd import evaluation, hardware
    from bfas.rtd.baselines import paper_evaluation
    from bfas.rtd.persistence import tree_hash
    model, lora = tmp_path/"snapshot", tmp_path/"checkpoint/lora"
    model.mkdir()
    lora.mkdir(parents=True)
    (model/"config.json").write_text("{}")
    (lora/"adapter_config.json").write_text('{"r": 16}')
    manifest = dict(config={}, benchmark="alfworld", method=method, smoke=smoke,
        checkpoint_sha256=tree_hash(lora.parent), base_checkpoint_hash=tree_hash(model),
        model_path=str(model), hardware={"hard": "cpu-stub"}, teacher_tokens_charged=33, B=40, port=8930)
    monkeypatch.setattr(hardware, "hardware_identity", lambda: manifest["hardware"])
    def no_export(*args, **kwargs):
        pytest.fail("ALFWorld LoRA serving must not flatten or merge the model")
    monkeypatch.setattr(evaluation, "_flatten_adapter", no_export)
    monkeypatch.setattr(paper_evaluation.subprocess, "run", no_export)
    calls = []
    def campaign(root, policy, out, **kwargs):
        assert policy == lora and policy.is_absolute()
        assert kwargs["smoke"] is smoke
        calls.append((out.name, kwargs["kang"]))
        return dict(tasks=3 if smoke else 140, overall_accuracy_percent=0.)
    monkeypatch.setattr(paper_evaluation, "run_alfworld", campaign)
    result = paper_evaluation.evaluate_run(ROOT, tmp_path, manifest)
    assert calls == [("smoke" if smoke else "official", False)] + ([("kang_sag", True)] if method == "kang" else [])
    assert result["checkpoint_sha256"] == manifest["checkpoint_sha256"]
    assert result["export_sha256"] == tree_hash(lora)
    assert not (tmp_path/"export").exists()
    assert json.loads((tmp_path/"metrics.json").read_text()) == result
    if method == "kang":
        assert "kang_self_consistency" not in result["kang_self_consistency"]["official_single_sample"]


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


def test_smoke_alfworld_runs_exactly_three_and_restores_environment(tmp_path, monkeypatch):
    import os
    from contextlib import nullcontext
    from bfas.rtd.baselines import paper_evaluation
    from bfas.rtd.benchmarks import alfworld_identity
    from bfas.adapters import alfworld
    from bfas import run
    ids = [f"task{i}" for i in range(140)]
    class Adapter:
        def __init__(self, **kwargs):
            pass
        def prepare_renderer(self, model):
            pass
        def evaluate(self, model, out):
            assert os.environ["BFAS_ALFWORLD_EVAL_GAMES"] == "3"
            (out/"records.jsonl").write_text("\n".join(json.dumps(dict(task_id=t, won=True, steps=1)) for t in ids[:3]))
            return dict(success_rate=1., per_category={"fake": 1.})
        def release_policy(self):
            pass
    monkeypatch.setattr(alfworld, "ALFWorldAdapter", Adapter)
    monkeypatch.setattr(alfworld_identity, "official_expectations", lambda _: dict(task_ids=ids))
    monkeypatch.setattr(run, "serving_lane", lambda *a, **kw: nullcontext())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("BFAS_ALFWORLD_EVAL_GAMES", "77")
    result = paper_evaluation.run_alfworld(ROOT, tmp_path/"model", tmp_path/"eval", kang=False, port=1, smoke=True)
    assert result["tasks"] == 3 and result["smoke"] and not result["official_full"]
    assert os.environ["BFAS_ALFWORLD_EVAL_GAMES"] == "77"
    assert protocol("alfworld", smoke=True)["tasks"] == 3


def test_smoke_bfcl_selection_and_partial_validation(tmp_path, monkeypatch):
    from bfas.rtd import evaluation
    from bfas.rtd.baselines.paper_evaluation import bfcl_metrics
    ids = [f"simple_python_{i}" for i in range(5217)]
    monkeypatch.setattr(evaluation, "official_expectations",
                        lambda root: dict(generation={"simple_python": ids}, scoring={"simple_python": ids}))
    generate, evaluate, env = bfcl_commands(ROOT, tmp_path/"model", tmp_path, smoke=True)
    assert "--skip-server-setup" in generate and "--run-ids" in generate
    assert "--partial-eval" in evaluate
    assert evaluate[evaluate.index("--test-category")+1] == "simple_python"
    (tmp_path/"resultdir").mkdir()
    (tmp_path/"scoredir").mkdir()
    result_path = tmp_path/"resultdir/BFCL_v4_simple_python_result.json"
    result_path.write_text("\n".join(json.dumps(dict(id=tid)) for tid in ids[:3]))
    (tmp_path/"scoredir/BFCL_v4_simple_python_score.json").write_text(
        json.dumps(dict(correct_count=2, total_count=3))+"\n"+json.dumps(dict(id=ids[0], valid=False)))
    result = bfcl_metrics(ROOT, tmp_path, smoke=True)
    assert result["tasks"] == 3 and not result["official_full"]
    assert result["overall_accuracy_percent"] == pytest.approx(200/3)
    result_path.write_text(json.dumps(dict(id=ids[0])))
    with pytest.raises(ValueError, match="incomplete evaluation generation"):
        bfcl_metrics(ROOT, tmp_path, smoke=True)


@pytest.mark.parametrize("leader_exits", [False, True])
def test_vllm_setsid_cleanup_kills_engine_descendants_without_server(tmp_path, monkeypatch, leader_exits):
    """Real harmless Python process tree, mocked readiness; no network or GPU."""
    import os
    import subprocess
    import time
    from contextlib import nullcontext
    from bfas import run
    original_popen = subprocess.Popen
    pid_file = tmp_path/"child.pid"
    child_code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); time.sleep(60)"
    leader_code = (
        "import subprocess,sys,pathlib,time\n"
        f"child = subprocess.Popen([sys.executable, '-c', {child_code!r}], stdout=subprocess.PIPE, text=True)\n"
        "child.stdout.readline()\n"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))\n"
        + ("sys.exit(0)\n" if leader_exits else "time.sleep(60)\n"))
    def spawn(command, **kwargs):
        assert kwargs["start_new_session"] is True  # Popen invokes setsid before exec.
        return original_popen([sys.executable, "-c", leader_code], **kwargs)
    monkeypatch.setattr(run.subprocess, "Popen", spawn)
    monkeypatch.setattr(run.urllib.request, "urlopen", lambda *a, **kw: nullcontext(SimpleNamespace(status=200)))
    server = run.VLLMServer("tiny", "", 12345, tmp_path/"server.log", "tiny")
    try:
        server.start()
        deadline = time.monotonic()+5
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert pid_file.exists()
        child_pid = int(pid_file.read_text())
        assert os.getpgid(child_pid) == os.getsid(child_pid) == server.process.pid
        if leader_exits:
            server.process.wait(timeout=5)
        server.close()
        deadline = time.monotonic()+5
        while time.monotonic() < deadline:
            stat = Path(f"/proc/{child_pid}/stat")
            if not stat.exists() or stat.read_text().split()[2] == "Z":
                break
            time.sleep(.01)
        else:
            pytest.fail("EngineCore-like descendant survived close()")
        assert server.process is None and server._log is None
        server.close()  # Idempotent after cleanup.
    finally:
        server.close()


def test_vllm_kills_group_on_term_timeout(monkeypatch):
    import signal
    import subprocess
    from bfas import processes
    sent, waits = [], []
    monkeypatch.setattr(processes.os, "killpg", lambda pid, sig: sent.append((pid, sig)))
    def wait(**kwargs):
        waits.append(kwargs)
        if kwargs:
            raise subprocess.TimeoutExpired("fake", kwargs["timeout"])
    processes.stop_process_group(SimpleNamespace(pid=123456, wait=wait), timeout=.01)
    assert sent == [(123456, signal.SIGTERM), (123456, signal.SIGKILL)]
    assert waits == [{"timeout": .01}, {}]


def test_runner_interrupt_gives_worker_time_to_close_server(tmp_path, monkeypatch):
    import signal
    from tools import baseline_run
    from bfas import processes
    sent, waits = [], []
    def wait(**kwargs):
        waits.append(kwargs)
        if len(waits) == 1:
            raise KeyboardInterrupt
        return 0
    process = SimpleNamespace(pid=123456, wait=wait)
    def spawn(command, **kwargs):
        assert kwargs["start_new_session"]
        return process
    monkeypatch.setattr(baseline_run.subprocess, "Popen", spawn)
    monkeypatch.setattr(processes.os, "killpg", lambda pid, sig: sent.append((pid, sig)))
    with (tmp_path/"worker.log").open("w") as log, pytest.raises(KeyboardInterrupt):
        baseline_run.run_worker(["fake"], log)
    assert waits == [{}, {"timeout": 45}, {}]
    assert sent == [(123456, signal.SIGTERM), (123456, signal.SIGKILL)]


@pytest.mark.parametrize("fail_generation", [False, True])
def test_bfcl_runner_owns_server_and_always_closes(tmp_path, monkeypatch, fail_generation):
    from contextlib import contextmanager
    from bfas import run
    from bfas.rtd import evaluation, evaluation_lock
    from bfas.rtd.baselines import paper_evaluation
    events = []
    class Server:
        def __init__(self, *args, **kwargs):
            assert kwargs["server_args"] == ["--dtype", "bfloat16", "--tensor-parallel-size", "1",
                "--gpu-memory-utilization", "0.85", "--trust-remote-code"]
        def start(self):
            events.append("start")
        def close(self):
            events.append("close")
    @contextmanager
    def reserve(*a, **kw):
        yield 12345, None
    monkeypatch.setattr(run, "VLLMServer", Server)
    monkeypatch.setattr(evaluation_lock, "reserve_port", reserve)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("BFCL_PROJECT_ROOT", str(tmp_path/"old"))
    ids = [f"simple_python_{i}" for i in range(3)]
    monkeypatch.setattr(evaluation, "official_expectations", lambda _: dict(generation={"simple_python": ids}))
    def execute(command, **kwargs):
        if "generate" in command:
            assert "--skip-server-setup" in command and "--run-ids" in command
            from pathlib import Path
            path = Path(kwargs["env"]["BFCL_PROJECT_ROOT"])/"test_case_ids_to_generate.json"
            assert json.loads(path.read_text()) == {"simple_python": ids}
            events.append("generate")
            if fail_generation:
                raise RuntimeError("fake generation failure")
        else:
            assert events[-1] == "close"
            assert "--partial-eval" in command
            events.append("score")
    monkeypatch.setattr(paper_evaluation.subprocess, "run", execute)
    monkeypatch.setattr(paper_evaluation, "bfcl_metrics", lambda *a, **kw: dict(tasks=3, smoke=kw["smoke"]))
    if fail_generation:
        with pytest.raises(RuntimeError, match="fake generation failure"):
            paper_evaluation.run_bfcl(ROOT, tmp_path/"model", tmp_path/"eval", kang=False, port=12345, smoke=True)
        assert events == ["start", "generate", "close"]
    else:
        result = paper_evaluation.run_bfcl(ROOT, tmp_path/"model", tmp_path/"eval", kang=False, port=12345, smoke=True)
        assert result == dict(tasks=3, smoke=True)
        assert events == ["start", "generate", "close", "score"]
    assert (tmp_path/"eval/gpu_usage.json").exists()
