from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bfas.adapters import appworld_official as aw  # noqa: E402
from bfas.adapter import Demo  # noqa: E402

FIXTURE = ROOT / "tests/fixtures/appworld_official"


class FixtureTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt=False, **_):
        assert tokenize is False
        rendered = "".join(f"<{m['role']}>{m.get('content') or ''}<END>" for m in messages)
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered


def _fixture_run(tmp_path: Path, guidance_block: str | None = None) -> aw.OfficialRun:
    outputs = tmp_path / "outputs"
    shutil.copytree(FIXTURE / "tasks", outputs / "tasks")
    if guidance_block is not None:
        # inject the guidance into every user message that ends the instructions
        path = outputs / "tasks/50e1ac9_1/logs/lm_calls.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            first_user = next(m for m in row["input"]["messages"] if m["role"] == "user")
            first_user["content"] += guidance_block
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    evaluation = json.loads((FIXTURE / "evaluation.json").read_text())
    return aw.OfficialRun(
        experiment_name="simplified_react_code_agent/bfas/test",
        dataset_name="bfas_test",
        outputs_dir=outputs,
        evaluation=evaluation,
        guidance_block=guidance_block,
    )


def _adapter() -> aw.AppWorldOfficialAdapter:
    adapter = aw.AppWorldOfficialAdapter(seed=1, port=9555)
    adapter._tokenizer = FixtureTokenizer()
    adapter._loaded_policy = "fixture-policy"
    adapter.keep_outputs = True
    return adapter


def test_rollout_renders_official_lm_calls_and_uses_official_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _adapter()
    monkeypatch.setattr(adapter, "prepare_renderer", lambda policy: None)
    captured: dict[str, Any] = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return _fixture_run(tmp_path)

    monkeypatch.setattr(adapter, "_run_official", fake_run)
    rollouts = adapter.rollout("fixture-policy", ["50e1ac9_1", "50e1ac9_2"], temperature=0.7)

    assert captured["model_name"] == "bfas-policy"
    assert captured["base_url"] == "http://localhost:9555/v1"
    assert captured["temperature"] == 0.7
    assert [r.verified for r in rollouts] == [True, False]
    first = rollouts[0]
    assert len(first.turns) == 2
    assert first.turns[0].prompt.startswith("<user>I am your supervisor")
    assert first.turns[0].prompt.endswith("<assistant>")
    assert "```python" in first.turns[0].target
    assert first.raw["steps"] == 2 and first.raw["tokens"] == 2560
    assert adapter.rerender({"_render_context": first.turns[0].context}) == first.turns[0].prompt
    # a task with no log (never ran) yields an unverified, turn-less rollout
    assert rollouts[1].turns == []
    assert adapter.generation_suffix() == "<assistant>"


def test_guided_rollout_strips_the_worked_example_for_deployment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _adapter()
    monkeypatch.setattr(adapter, "prepare_renderer", lambda policy: None)
    demo = Demo("50e1ac9_1", [], "```python\nprint(apis.api_docs.show_app_descriptions())\n```")
    block = adapter._guidance_block(demo)
    assert block is not None and block.startswith(aw.GUIDANCE_HEADER)

    def fake_run(**kwargs):
        assert kwargs["guidance"] is demo and kwargs["num_processes"] == 1
        return _fixture_run(tmp_path, guidance_block=block)

    monkeypatch.setattr(adapter, "_run_official", fake_run)
    rollout = adapter.rollout("fixture-policy", ["50e1ac9_1"], 0.7, {"50e1ac9_1": demo})[0]

    assert rollout.raw["guided"] is True
    assert block in rollout.turns[0].prompt
    deployment = rollout.raw["deployment_turns"]
    assert len(deployment) == len(rollout.turns)
    assert block not in deployment[0].prompt
    assert deployment[0].target == rollout.turns[0].target


def test_teacher_episode_builds_demo_and_charges_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _adapter()
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://teacher.example/")
    monkeypatch.setenv("OLLAMA_API_KEY", "secret")
    captured: dict[str, Any] = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return _fixture_run(tmp_path)

    monkeypatch.setattr(adapter, "_run_official", fake_run)
    episode = adapter.teacher_episode("50e1ac9_1", attempt_index=1, temperature=0.7)

    assert captured["model_name"] == "deepseek-v4-pro"
    assert captured["base_url"] == "https://teacher.example/v1"
    assert captured["api_key"] == "secret"
    assert captured["client_name"] == "openai" and captured["extra_env"] == {}
    assert episode.verified is True and episode.tokens_spent == 2560
    assert episode.demo is not None
    assert episode.demo.raw["attempt"] == 2
    assert "```python" in episode.demo.worked_example
    assert len(episode.response_texts) == 2


def test_evaluate_reports_official_aggregate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter()
    monkeypatch.setenv("BFAS_APPWORLD_EVAL_SPLIT", "dev")
    monkeypatch.setenv("BFAS_APPWORLD_EVAL_TASKS", "2")
    monkeypatch.setattr(adapter, "_dataset_ids", staticmethod(lambda name: ["50e1ac9_1", "50e1ac9_2", "x_1"]))
    run = _fixture_run(tmp_path)
    (run.outputs_dir / "evaluations").mkdir()
    (run.outputs_dir / "evaluations" / "bfas_test.json").write_text(json.dumps(run.evaluation))
    monkeypatch.setattr(adapter, "_run_official", lambda **kwargs: run)

    out_dir = tmp_path / "eval"
    result = adapter.evaluate("fixture-policy", out_dir)

    assert result["headline"] == pytest.approx(0.5)
    assert result["per_category"] == {"50e1ac9": 0.5}
    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert metrics["n_tasks"] == 2 and metrics["scaffold"].startswith("official")
    records = [json.loads(l) for l in (out_dir / "records.jsonl").read_text().splitlines()]
    assert [r["success"] for r in records] == [True, False]


def test_config_text_is_valid_jsonnet_and_pins_the_prompt_file() -> None:
    text = aw.AppWorldOfficialAdapter._config_text(
        model_name="bfas-policy", temperature=0.3, seed=7,
        prompt_file=Path("/tmp/p.txt"), dataset_name="bfas_x", max_steps=50,
    )
    config = json.loads(text)
    agent = config["config"]["agent"]
    assert agent["model_config"]["name"] == "bfas-policy"
    assert agent["model_config"]["temperature"] == 0.3
    assert agent["prompt_file_path"] == "/tmp/p.txt"
    assert agent["skip_if_finished"] is False
    assert config["config"]["dataset"] == "bfas_x"


def test_azure_teacher_goes_through_litellm_with_env_only_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _adapter()
    monkeypatch.setenv("BFAS_TEACHER", "gpt-5.6-luna")
    monkeypatch.setenv("AZURE_LLM_ENDPOINT", "https://apim.example/")
    monkeypatch.setenv("AZURE_LLM_KEY", "azure-secret")
    captured: dict[str, Any] = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return _fixture_run(tmp_path)

    monkeypatch.setattr(adapter, "_run_official", fake_run)
    episode = adapter.teacher_episode("50e1ac9_1", attempt_index=0, temperature=0.0)

    assert captured["model_name"] == "azure/gpt-5.6-luna"
    assert captured["client_name"] == "litellm"
    assert captured["extra_env"]["AZURE_API_KEY"] == "azure-secret"
    assert captured["extra_env"]["AZURE_API_BASE"] == "https://apim.example"
    assert "azure-secret" not in aw.AppWorldOfficialAdapter._config_text(
        model_name="azure/gpt-5.6-luna", temperature=0.0, seed=1, prompt_file=Path("/p"),
        dataset_name="d", max_steps=5, client_name="litellm",
    )
    assert episode.verified is True
