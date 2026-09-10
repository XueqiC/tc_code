"""CPU contract tests: stub the environment, student and paid teacher."""

import builtins
from copy import deepcopy
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import appworld_teacher
from bfas import ledger, run as bfas_run
from bfas.adapter import CollectionArtifacts, Demo, Turn
from bfas.adapters import WebShopAdapter
from bfas.adapters import webshop
from bfas.arms import build_pool
from tools import webshop_eval


class StubTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        rendered = "".join(f"<{m['role']}>{m['content']}<END>" for m in messages)
        return rendered + ("<GEN>" if add_generation_prompt else "")


class StubBridge:
    def __init__(self, rewards=(1.0,), done_after=2, tuple_reset=False):
        self.categories = ["bottles" if i % 2 else "clothing" for i in range(6910)]
        self.rewards = list(rewards)
        self.done_after = done_after
        self.tuple_reset = tuple_reset
        self.sessions = []
        self.actions = []
        self.done = False
        self.closed = False

    def reset(self, session):
        assert type(session) is int
        self.sessions.append(session)
        self.steps = 0
        self.done = False
        self.reward = self.rewards[min(len(self.sessions) - 1, len(self.rewards) - 1)]
        self.observation = "Instruction: buy a blue bottle. " + "i" * 5000
        return (self.observation, {}) if self.tuple_reset else self.observation

    def step(self, action):
        self.actions.append(action)
        self.steps += 1
        self.done = self.steps == self.done_after
        self.observation = f"Page {self.steps} [SEP] Buy Now " + "p" * 5000
        # Allow artificial nonterminal rewards to test the stricter demo rule.
        reward = self.reward if self.done or self.done_after is None else 0.0
        return self.observation, reward, self.done, {}

    def close(self):
        self.closed = True


class StubClient:
    def __init__(self, responses=None):
        self.responses = iter(responses) if responses is not None else None
        self.calls = []
        self.closed = False
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        reply = next(self.responses) if self.responses is not None else "click[Buy Now]"
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])

    def __enter__(self):
        self.closed = False
        return self

    def __exit__(self, *args):
        self.closed = True


@pytest.fixture(autouse=True)
def no_network_or_real_environment(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU tests must not call an API or load a real teacher config")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(appworld_teacher, "load_teacher_config", forbidden)
    monkeypatch.setattr(appworld_teacher, "generate_reply", forbidden)
    monkeypatch.setattr(ledger, "LEDGER_ROOT", tmp_path / "teacher_ledger")
    monkeypatch.setenv("BFAS_TEACHER_MIN_INTERVAL_S", "0")
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in {"gym", "transformers"} or name.startswith("web_agent_site"):
            pytest.fail(f"CPU tests must not import {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


@pytest.fixture
def harness(monkeypatch):
    bridge = StubBridge()
    client = StubClient()
    monkeypatch.setattr(webshop, "_EnvBridge", lambda: bridge)
    adapter = WebShopAdapter(seed=19, port=9123)
    adapter._tokenizer = StubTokenizer()
    adapter._loaded_policy = "stub-policy"
    monkeypatch.setattr(adapter, "_client", lambda: client)
    yield adapter, bridge, client
    adapter.close()


@pytest.fixture
def teacher(monkeypatch):
    state = SimpleNamespace(configs=[], calls=[], replies=None, report_usage=True)

    def load(name):
        config = SimpleNamespace(name=name, attempt=len(state.configs))
        state.configs.append(config)
        return config

    def generate(config, messages, temperature, *, usage_callback):
        state.calls.append((config.attempt, deepcopy(messages), temperature))
        reply = next(state.replies) if state.replies is not None else "Thought: It matches.\nAction: click[Buy Now]"
        if isinstance(reply, Exception):
            raise reply
        if state.report_usage:
            usage_callback({"completion_tokens": 37 + config.attempt,
                            "prompt_tokens": 2000, "total_tokens": 2037 + config.attempt})
        return reply

    monkeypatch.setattr(appworld_teacher, "load_teacher_config", load)
    monkeypatch.setattr(appworld_teacher, "generate_reply", generate)
    return state


def ledger_records():
    return [json.loads(line) for line in (ledger.LEDGER_ROOT / "webshop.jsonl").read_text().splitlines()]


def test_support_pool_and_stratified_split(harness):
    adapter, bridge, _ = harness
    tasks = adapter.task_pool()
    assert len(tasks) == 6410
    assert {int(t.task_id) for t in tasks} == set(range(500, 6910))
    assert tasks[0].category == "clothing" and tasks[1].category == "bottles"
    split = adapter.support_split()
    assert (len(split.support), len(split.demand), len(split.calibration)) == (250, 200, 50)
    assert set(split.demand).isdisjoint(split.calibration)
    assert set(split.demand) | set(split.calibration) == set(split.support)
    assert all(int(task_id) >= 500 for task_id in split.support)
    categories = adapter.task_categories()
    assert sum(categories[t] == "bottles" for t in split.support) == 125
    assert sum(categories[t] == "bottles" for t in split.calibration) == 25
    adapter.seed = 77
    assert adapter.support_split() == split  # protocol seed is fixed to 50
    assert adapter.official_eval_split_disjoint() is True
    assert bridge.sessions == []  # metadata alone, no task rollouts


def test_category_fallback_is_deterministic_index_buckets(harness):
    adapter, bridge, _ = harness
    bridge.categories = []
    tasks = adapter.task_pool()
    assert tasks[0].category == "index_bucket_01"
    assert tasks[-1].category == "index_bucket_13"
    assert len(adapter.support_split().support) == 250


@pytest.mark.parametrize("tuple_reset", [True, False])
def test_rollout_prompts_and_decoding_equal_validated_evaluator(harness, tuple_reset):
    adapter, bridge, client = harness
    responses = ["invalid"] + ["Thought: Keep shopping.\nAction: search[blue bottle] " + " " * 400] * 13
    responses += ["Thought: All options match.\nclick[Buy Now]"]
    bridge.done_after = 14
    bridge.tuple_reset = tuple_reset
    client.responses = iter(responses)
    reference_env = StubBridge(done_after=14, tuple_reset=tuple_reset)
    reference_client = StubClient(responses)
    expected = webshop_eval.run_episode(reference_env, reference_client, 500, "bfas-policy", 15, webshop_eval.OBS_CHARS)
    actual = adapter.rollout("stub-policy", ["500"], 0.0)[0]
    assert client.calls == reference_client.calls
    assert actual.verified == expected["success"] is True
    for key, value in expected.items():
        if key != "elapsed":
            assert actual.raw[key] == value
    assert bridge.actions == reference_env.actions
    assert len(actual.turns) == 15
    for turn, call in zip(actual.turns, client.calls):
        assert turn.context == call["messages"]
        assert turn.prompt == adapter.rerender({"_render_context": turn.context})
        assert turn.prompt.endswith(adapter.generation_suffix())
        assert len(turn.context[1]["content"]) <= 60000
        assert turn.context[0]["content"] == webshop_eval.SYSTEM_PROMPT
    last_prompt = actual.turns[-1].context[1]["content"]
    assert "i" * 601 not in last_prompt  # first observation now uses the 600 cap
    assert client.closed


@pytest.mark.parametrize("reward,done_after,verified", [
    (1.0, 1, True), (0.999999, 1, False), (0.5, 1, False), (0.0, 1, False),
    (1.0, 15, True), (1.0, 16, False), (1.0, None, False),
])
def test_teacher_verification_requires_terminal_exact_reward_within_15_steps(
    harness, teacher, reward, done_after, verified,
):
    adapter, bridge, _ = harness
    bridge.rewards = [reward]
    bridge.done_after = done_after
    episode = adapter.teacher_episode("500", 0, 0.0)
    assert episode.verified is verified
    assert (episode.demo is not None) is verified
    assert len(teacher.calls) <= 15
    assert episode.tokens_spent == 37 * len(teacher.calls)
    assert len(episode.response_texts) == len(teacher.calls)
    assert [config.name for config in teacher.configs] == ["gpt-5.4"]


def test_failed_attempts_are_charged_once_and_success_is_cached(harness, teacher):
    adapter, bridge, _ = harness
    bridge.rewards = [0.0, 0.5, 1.0]
    bridge.done_after = 1
    demos = adapter.teacher_demo(["500"], attempts=3)
    assert set(demos) == {"500"}
    records = ledger_records()
    assert [r["attempt_index"] for r in records] == [0, 1, 2]
    assert [r["temperature"] for r in records] == [0.0, 0.7, 0.7]
    assert [r["verified"] for r in records] == [False, False, True]
    assert [r["tokens_spent"] for r in records] == [37, 38, 39]
    assert all(r["teacher"] == "gpt-5.4" and r["purpose"] == "teacher" and r["timestamp"] for r in records)
    assert ledger.load_ledger("webshop")["500"]["tokens_total"] == 114
    assert set(adapter.teacher_demo(["500"], attempts=3)) == {"500"}
    assert len(teacher.calls) == len(ledger_records()) == 3


def test_failed_attempt_cap_and_response_estimate_fallback(harness, teacher):
    adapter, bridge, _ = harness
    bridge.rewards = [0.0]
    bridge.done_after = 1
    teacher.report_usage = False
    assert adapter.teacher_demo(["500"], attempts=9) == {}
    expected = ledger.estimate_response_tokens(["Thought: It matches.\nAction: click[Buy Now]"])
    assert [r["tokens_spent"] for r in ledger_records()] == [expected] * 3
    assert adapter.teacher_demo(["500"], attempts=9) == {}
    assert len(teacher.calls) == 3


def test_partial_paid_episode_is_charged_after_later_api_failure(harness, teacher):
    adapter, bridge, _ = harness
    teacher.replies = iter(["search[bottle]", appworld_teacher.TeacherAPIError("HTTP 429")])
    assert adapter.teacher_demo(["500"], attempts=1) == {}
    record, = ledger_records()
    assert record["verified"] is False and record["tokens_spent"] == 37
    assert bridge.closed


def test_initial_quota_rejection_does_not_spend_attempt(harness, teacher):
    adapter, _, _ = harness
    teacher.replies = iter([appworld_teacher.TeacherAPIError("HTTP 429")])
    with pytest.raises(appworld_teacher.TeacherAPIError):
        adapter.teacher_demo(["500"], attempts=3)
    assert ledger.load_ledger("webshop") == {}


def test_demos_emit_action_rows_with_deployment_context_and_full_usage(harness, teacher):
    adapter, bridge, _ = harness
    teacher.replies = iter(["not an action", "Thought: Search.\nAction: search[bottle]", "click[Buy Now]"])
    demos = adapter.teacher_demo(["500"], 3)
    rows = build_pool("sft", adapter, teacher_demos=demos, unguided_rollouts=[], guided_rollouts=[], p_hats={})
    assert [r["response"] for r in rows] == ["search[bottle]", "click[Buy Now]"]
    for row, (_, messages, _) in zip(rows, teacher.calls[1:]):
        assert row["_render_context"] == messages
        assert row["prompt"] == adapter.rerender(row)
        assert {"task_id", "category", "teacher", "turn_index", "token_hint"} <= row.keys()
    assert "Worked example:" not in demos["500"].worked_example
    assert "Instruction: buy a blue bottle" in teacher.calls[0][1][1]["content"]
    assert ledger_records()[0]["tokens_spent"] == 111  # invalid turn is paid too


def test_guided_collection_keeps_deployment_prompts_for_training(harness):
    adapter, _, client = harness
    demo = Demo("500", [Turn("p", "click[Buy Now]")], "PRIVATE GUIDANCE")
    rollout = adapter.rollout("stub-policy", ["500"], 0.7, {"500": demo})[0]
    assert all(call["temperature"] == 0.7 for call in client.calls)
    assert all("PRIVATE GUIDANCE" in turn.prompt for turn in rollout.turns)
    assert all("PRIVATE GUIDANCE" not in turn.prompt for turn in rollout.raw["deployment_turns"])
    rollout.raw["mu_fields"] = [{"_mu_nll_sum": 2.0, "_mu_ntok": 3}] * len(rollout.turns)
    artifacts = CollectionArtifacts((), {}, {}, {"500": demo}, (rollout,), {"500": 1}, {})
    restored = CollectionArtifacts.from_dict(artifacts.as_dict())
    assert restored.guided_rollouts[0].raw["deployment_turns"] == rollout.raw["deployment_turns"]
    rows = build_pool("ours", adapter, teacher_demos={}, unguided_rollouts=[],
                      guided_rollouts=[rollout], p_hats={"500": 0.25})
    assert all("PRIVATE GUIDANCE" not in row["prompt"] for row in rows)
    assert all(row["_guided"] and row["_mu_ntok"] == 3 for row in rows)


def test_evaluation_always_uses_all_500_test_sessions_and_reward_metrics(harness, tmp_path, monkeypatch):
    adapter, bridge, client = harness
    bridge.rewards = [1.0, 0.5] * 250
    bridge.done_after = 1
    monkeypatch.setenv("BFAS_WEBSHOP_EVAL_GAMES", "2")  # no evaluation subset override
    metrics = adapter.evaluate("stub-policy", tmp_path)
    assert bridge.sessions == list(range(500))
    assert metrics["n"] == 500
    assert metrics["headline"] == metrics["success_rate"] == 0.5
    assert metrics["mean_score"] == metrics["score"] == 75.0
    assert metrics["config"]["max_steps"] == 15
    records = [json.loads(line) for line in (tmp_path / "records.jsonl").read_text().splitlines()]
    assert [r["session"] for r in records] == list(range(500))
    assert json.loads((tmp_path / "metrics.json").read_text()) == metrics
    assert bridge.closed and client.closed
    assert all(call["temperature"] == 0 and call["max_tokens"] == 128 for call in client.calls)


def test_registration_and_serving_lane(harness, tmp_path, monkeypatch):
    adapter, _, client = harness
    args = bfas_run.parse_args(["--benchmark", "webshop", "--arm", "base"])
    assert args.benchmark == "webshop"
    registered = bfas_run.make_adapter("webshop", 9, 9234)
    assert isinstance(registered, WebShopAdapter)
    assert (registered.seed, registered.port) == (9, 9234)
    events = []

    class Registry:
        def __init__(self, port):
            assert port == adapter.port

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class Server:
        def __init__(self, *args, served_model_name):
            assert served_model_name == "bfas-policy"

        def start(self):
            events.append("start")

        def close(self):
            events.append("close")

    monkeypatch.setattr(bfas_run, "PortRegistry", Registry)
    monkeypatch.setattr(bfas_run, "VLLMServer", Server)
    with bfas_run.serving_lane(adapter, "stub-policy", "0", adapter.port, tmp_path / "server.log"):
        events.append("body")
    assert events == ["start", "body", "close"]
    assert client.calls[0]["model"] == "bfas-policy"


def test_teacher_rejects_test_goals_without_api_calls(harness):
    adapter, _, _ = harness
    with pytest.raises(ValueError, match="train goals"):
        adapter.teacher_episode("499", 0, 0.0)


def test_teacher_usage_hook_reports_azure_output_tokens(monkeypatch):
    # Exercise the real helper with a stub HTTP transport, never an API call.
    import importlib.util
    spec = importlib.util.spec_from_file_location("teacher_usage_test", ROOT / "src/appworld_teacher.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    payload = {"usage": {"completion_tokens": 83},
               "choices": [{"message": {"content": "click[Buy Now]"}}]}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps(payload).encode()

    monkeypatch.setattr(module.urllib.request, "build_opener", lambda: SimpleNamespace(open=lambda *a, **k: Response()))
    monkeypatch.delenv("AZURE_USAGE_LOG", raising=False)
    config = module.TeacherConfig("gpt-5.4", "gpt-5.4", "https://stub.invalid/", "stub", "azure_openai")
    usages = []
    assert module.generate_reply(config, [], temperature=0.7, usage_callback=usages.append) == "click[Buy Now]"
    assert usages == [{"completion_tokens": 83}]


def test_python38_bridge_protocol_with_stub_environment(monkeypatch, tmp_path):
    # Run the actual worker code under WebShop's Python 3.8, replacing only its
    # lazy make_env import. No gym, catalog, search index or APIs are needed.
    python = webshop.ENV_PYTHON
    if not python.exists():
        pytest.skip("WebShop Python 3.8 is not installed")
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools/__init__.py").write_text("")
    (tmp_path / "tools/webshop_eval.py").write_text('''
import os
from types import SimpleNamespace
class Env:
    def __init__(self):
        self.unwrapped = self
        self.server = SimpleNamespace(goals=[{"category": "bottles"}] * 6910)
    def reset(self, session):
        assert type(session) is int
        print("non-protocol environment log")
        return "goal " + str(session), {}
    def step(self, action):
        if action == "bad":
            raise ValueError("bad action")
        return action, 1.0, True, {}
    def close(self):
        print("closed")
def make_env():
    os.write(1, b"native library log\\n")
    return Env()
''')
    (tmp_path / "envs/webshop/repo").mkdir(parents=True)
    monkeypatch.setattr(webshop, "ROOT", tmp_path)
    bridge = webshop._EnvBridge()
    try:
        assert bridge.categories == ["bottles"] * 6910
        assert bridge.reset(500) == "goal 500"
        assert bridge.done is False
        with pytest.raises(RuntimeError, match="bad action"):
            bridge.step("bad")
        assert bridge.step("click[Buy Now]") == ("click[Buy Now]", 1.0, True, {})
        assert bridge.reset(501) == "goal 501"  # process reused across sessions
        assert bridge.done is False
    finally:
        bridge.close()
    assert bridge.process.returncode == 0
