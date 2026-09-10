"""CPU-only WebShop evaluator tests; never import the real WebShop environment."""

import builtins
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/webshop_eval.py"


@pytest.fixture
def evaluator(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "web_agent_site" or name.startswith("web_agent_site."):
            pytest.fail("CPU tests must not import web_agent_site")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    spec = importlib.util.spec_from_file_location("webshop_eval", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeEnv:
    def __init__(self, reward=1.0, tuple_reset=True):
        self.reward = reward
        self.tuple_reset = tuple_reset
        self.sessions = []
        self.actions = []
        self.closed = False

    def reset(self, session):
        assert type(session) is int
        self.sessions.append(session)
        observation = "Instruction: buy a blue bottle " + "x" * 100
        return (observation, None) if self.tuple_reset else observation

    def step(self, action):
        self.actions.append(action)
        done = action == "click[Buy Now]"
        return "Bottle page " + "y" * 100, self.reward if done else 0.0, done, None

    def close(self):
        self.closed = True


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.closed = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response))])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True


@pytest.mark.parametrize("response,expected", [
    ("search[blue bottle]", "search[blue bottle]"),
    ("  click[Buy Now]  ", "click[Buy Now]"),
    ("Thought: I should search.\nsearch[blue bottle]", "search[blue bottle]"),
    ("Thought: The options match.\nAction: click[Buy Now]", "click[Buy Now]"),
    ("search[bottle]\nThought: Choose blue.\nclick[blue]", "click[blue]"),
    ("```\nclick[24 oz]\n```", "click[24 oz]"),
    ("click[blue]\nclick[unfinished", "click[blue]"),
    ("Thought: maybe click[Buy Now]", None),
    ("Please search[bottle]", None),
    ("search[bottle] trailing prose", None),
    ("click[blue] search[bottle]", None),
    ("click[]", None),
    ("search[  ]", None),
    ("click[Buy Now", None),
    ("garbage", None),
    ("", None),
    (None, None),
])
def test_action_parsing(evaluator, response, expected):
    assert evaluator.parse_action(response) == expected


def test_prompt_rebuilding_determinism(evaluator):
    history = [("FIRST12345hidden", "Thought: Find it.\nsearch[bottle]")]
    original = deepcopy(history)
    first = evaluator.build_messages(history, "SECOND1234hidden", 10)
    second = evaluator.build_messages(history, "SECOND1234hidden", 10)
    assert first == second
    assert history == original
    assert [message["role"] for message in first] == ["system", "user"]
    assert first[1]["content"].count(evaluator.WORKED_EXAMPLE.rstrip()) == 1
    assert first[1]["content"].endswith(
        "Observation: FIRST12345\nAction: Thought: Find it.\nsearch[bottle]\n"
        "Observation: SECOND1234\nAction:"
    )
    first[1]["content"] = "changed"
    assert evaluator.build_messages(history, "SECOND1234hidden", 10) == second


@pytest.mark.parametrize("tuple_reset", [True, False])
def test_episode_api_and_full_transcript(evaluator, tuple_reset):
    env = FakeEnv(tuple_reset=tuple_reset)
    client = FakeClient(["Thought: Search first.\nsearch[bottle]", "click[Buy Now]"])
    record = evaluator.run_episode(env, client, 7, "student", 15, 30)
    assert env.sessions == [7]
    assert env.actions == ["search[bottle]", "click[Buy Now]"]
    assert record == {
        "session": 7, "reward": 1.0, "success": True, "steps": 2,
        "format_failures": 0, "final_action": "click[Buy Now]", "elapsed": record["elapsed"],
    }
    assert record["elapsed"] >= 0
    for call in client.calls:
        assert {key: value for key, value in call.items() if key != "messages"} == {
            "model": "student", "temperature": 0, "max_tokens": 128,
            "stop": ["\nObservation", "Observation:"],
        }
        assert [message["role"] for message in call["messages"]] == ["system", "user"]
    before = client.calls[0]["messages"][1]["content"]
    after = client.calls[1]["messages"][1]["content"]
    assert after == before + (
        " Thought: Search first.\nsearch[bottle]\nObservation: "
        + ("Bottle page " + "y" * 100)[:30] + "\nAction:"
    )


@pytest.mark.parametrize("responses,max_steps,steps,failures,actions", [
    (["garbage"] * 4, 15, 3, 3, []),
    (["garbage", "garbage", "search[bottle]", "garbage", "garbage", "click[Buy Now]"],
     15, 6, 4, ["search[bottle]", "click[Buy Now]"]),
    (["search[bottle]"] * 4, 2, 2, 0, ["search[bottle]"] * 2),
    ([None] * 4, 2, 2, 2, []),
])
def test_noops_failure_streak_and_step_limit(evaluator, responses, max_steps, steps, failures, actions):
    env = FakeEnv()
    client = FakeClient(responses)
    record = evaluator.run_episode(env, client, 0, "student", max_steps, 30)
    assert record["steps"] == steps == len(client.calls)
    assert record["format_failures"] == failures
    assert env.actions == actions
    assert record["success"] == ("click[Buy Now]" in actions)
    if not actions:
        assert record["final_action"] is None
        assert record["reward"] == 0
        # No-op leaves the next observation unchanged.
        for call in client.calls:
            assert call["messages"][1]["content"].endswith(
                "Observation: Instruction: buy a blue bottle\nAction:"
            )


def cli_args(evaluator, out, *extra):
    return evaluator.parse_args([
        "--base-url", "http://localhost:8000/v1", "--model", "student",
        "--out", str(out), *extra,
    ])


def test_cli_defaults(evaluator, tmp_path):
    args = cli_args(evaluator, tmp_path)
    assert (args.start, args.n, args.max_steps, args.obs_chars) == (0, 500, 15, 2500)
    assert args.num_products is None
    assert args.seed == 0


def test_metrics(evaluator):
    config = {"seed": 73, "tag": "test"}
    # Exact equality determines success, including for nearly perfect rewards.
    records = [
        {"reward": 1.0, "steps": 3}, {"reward": 0.5, "steps": 5},
        {"reward": 0.0, "steps": 7}, {"reward": 0.999, "steps": 9},
    ]
    metrics = evaluator.compute_metrics(records, config)
    assert metrics["n"] == 4
    assert metrics["score"] == pytest.approx(62.475)
    assert metrics["success_rate"] == 0.25
    assert metrics["mean_steps"] == 6
    assert metrics["config"] == config
    assert evaluator.compute_metrics([], config) == {
        "n": 0, "score": 0.0, "success_rate": 0.0, "mean_steps": 0.0, "config": config,
    }


def test_resume_skip_incremental_writes_and_metrics(evaluator, tmp_path, monkeypatch, capsys):
    args = cli_args(evaluator, tmp_path, "--start", "4", "--n", "3", "--seed", "73", "--tag", "cpu")
    path = tmp_path / "records.jsonl"
    existing = [
        {"session": 99, "reward": 0.0, "success": False, "steps": 100},
        {"session": 4, "reward": 1.0, "success": True, "steps": 3},
    ]
    original = "".join(json.dumps(record) + "\n" for record in existing)
    path.write_text(original)
    env = FakeEnv(reward=0.5)
    client = FakeClient(["click[Buy Now]", RuntimeError("server interrupted")])
    monkeypatch.setattr(evaluator, "make_env", lambda num_products: env)
    constructors = []

    def fake_openai(**kwargs):
        constructors.append(kwargs)
        return client

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=fake_openai))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="server interrupted"):
        evaluator.evaluate(args)
    assert constructors == [{"base_url": args.base_url, "api_key": "EMPTY"}]
    assert env.sessions == [5, 6]
    assert env.closed and client.closed
    assert path.read_text().startswith(original)
    assert [json.loads(line)["session"] for line in path.read_text().splitlines()] == [99, 4, 5]

    env = FakeEnv()
    client = FakeClient(["click[Buy Now]"])
    metrics = evaluator.evaluate(args)
    assert env.sessions == [6]
    assert metrics["n"] == 3
    assert metrics["score"] == pytest.approx(100 * 2.5 / 3)
    assert metrics["success_rate"] == pytest.approx(2 / 3)
    assert metrics["mean_steps"] == pytest.approx(5 / 3)
    assert metrics["config"]["seed"] == 73
    assert metrics["config"]["tag"] == "cpu"
    assert json.loads((tmp_path / "metrics.json").read_text()) == metrics
    assert len(capsys.readouterr().out.splitlines()) == 1

    def forbidden(*args, **kwargs):
        pytest.fail("completed resume must not construct environment or client")

    monkeypatch.setattr(evaluator, "make_env", forbidden)
    monkeypatch.setattr(evaluator, "make_client", forbidden)
    saved = path.read_bytes()
    assert evaluator.evaluate(args) == metrics
    assert path.read_bytes() == saved


@pytest.mark.parametrize("tail", [b'{"session":', b'{"session":2,"reward":0}\n', b'{"session":2,"reward":0}'])
def test_resume_final_line_recovery(evaluator, tmp_path, tail):
    path = tmp_path / "records.jsonl"
    first = b'{"session":1,"reward":1}\n'
    path.write_bytes(first + tail)
    records = evaluator.load_records(path)
    assert records[1]["reward"] == 1
    if tail == b'{"session":':
        assert list(records) == [1]
        assert path.read_bytes() == first
    else:
        assert list(records) == [1, 2]
        assert path.read_bytes().endswith(b"\n")


def test_resume_rejects_corrupt_complete_line(evaluator, tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text("garbage\n")
    with pytest.raises(ValueError):
        evaluator.load_records(path)
