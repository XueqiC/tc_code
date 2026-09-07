"""AW-2 matching and report checks with synthetic files and an offline tokenizer."""

from collections import Counter
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "aw2_matched_pools.py"
SPEC = importlib.util.spec_from_file_location("aw2_matched_pools", SCRIPT)
aw2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(aw2)


def event(task, index, arm="early", **fields):
    return {
        "task_id": task, "prompt": "api_docs login in the prompt must not count",
        "response": f"print('{arm} café {index}')", "_rejected": "apis.spotify.login()",
        "_event_dU": [-0.5, 0, 0.5][index % 3], "_probe_pos": (index + 1) / 20,
        "_event_overflow": False, "_test_id": f"{arm}:{task}:{index}",
        **fields,
    }


def write_events(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n" + "\n\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8")


@pytest.fixture
def tokenizer_dir(tmp_path):
    # A real, tiny tokenizer avoids dependence on network or the user's HF cache.
    path = tmp_path / "tokenizer"
    path.mkdir()
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "a": 1, "b": 2, "c": 3},
                                          unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "[UNK]",
    }))
    return path


@pytest.fixture
def tokenizer(tokenizer_dir):
    return aw2.load_tokenizer(str(tokenizer_dir))


def test_cli_matching_overflow_and_report(tmp_path, tokenizer_dir):
    early = [event("a", i) for i in range(6)] + [event("b", i) for i in range(3)]
    spread = ([event("b", i, "spread") for i in range(5)]
              + [event("a", i, "spread") for i in range(5)])
    # Overflow changes n_a from 5 to 4. Tasks with only overflow on either side
    # must disappear entirely, as must tasks present in only one source.
    spread[-1]["_event_overflow"] = True
    early += [event("early_only", 0), event("overflow_early", 0, _event_overflow=True),
              event("overflow_spread", 0)]
    spread += [event("spread_only", 0, "spread"), event("overflow_early", 0, "spread"),
               event("overflow_spread", 0, "spread", _event_overflow=True)]
    directory = tmp_path / "data" / "appworld_events"
    write_events(directory / "aw1_base_k3.jsonl", early)
    write_events(directory / "aw1_spread_k3.jsonl", spread)
    command = [sys.executable, str(SCRIPT), "--tokenizer", str(tokenizer_dir)]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=True)
    report = json.loads((directory / "aw2_matched_report.json").read_text())
    assert report["seed"] == 0
    assert report["tasks"] == ["a", "b"]
    assert report["n_tasks"] == 2
    assert report["per_task_counts"] == {
        "a": {"n_early": 6, "n_spread": 4, "n_matched": 4},
        "b": {"n_early": 3, "n_spread": 5, "n_matched": 3},
    }
    assert report["input_counts"] == {
        "early": {"total_rows": 12, "overflow_rows": 1, "eligible_rows": 11, "unmatched_rows": 2},
        "spread": {"total_rows": 13, "overflow_rows": 2, "eligible_rows": 11, "unmatched_rows": 2},
    }
    task_orders = []
    for arm, source in [("early_matched", early), ("spread_matched", spread)]:
        rows = aw2.read_events(directory / f"pool_aw2_{arm}.jsonl")
        assert Counter(r["task_id"] for r in rows) == {"a": 4, "b": 3}
        assert len({r["_test_id"] for r in rows}) == 7
        assert all(r in source and not r["_event_overflow"] for r in rows)
        assert report["arms"][arm]["total_rows"] == 7
        assert report["arms"][arm]["steps"] == 1
        assert report["arms"][arm]["doc_login_fraction"] == 0
        task_orders.append([r["task_id"] for r in rows])
        assert f"{arm}: 7 rows, 2 tasks, ceil(7/8) = 1 steps" in result.stdout
    assert task_orders[0] == task_orders[1]
    assert not (directory / "pool_aw2_early_halfdose.jsonl").exists()
    outputs = [directory / "aw2_matched_report.json", *directory.glob("pool_aw2_*.jsonl")]
    original_bytes = {p: p.read_bytes() for p in outputs}
    subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=True)
    assert {p: p.read_bytes() for p in outputs} == original_bytes


def test_sampling_seed_and_preservation(tmp_path):
    early = [event("t", i) for i in range(20)]
    spread = [event("t", i, "spread") for i in range(10)]
    write_events(tmp_path / "early.jsonl", early)
    write_events(tmp_path / "spread.jsonl", spread)
    sources = [aw2.read_events(tmp_path / f"{arm}.jsonl") for arm in ("early", "spread")]
    first = aw2.match_pools(*sources)
    assert first == aw2.match_pools(*sources, seed=0)
    changed = aw2.match_pools(*sources, seed=19)
    assert {r["_test_id"] for r in first[0]} != {r["_test_id"] for r in changed[0]}
    assert first[1] != changed[1]
    assert sources == [early, spread]


@pytest.mark.parametrize("n", [1, 8, 9, 16, 17])
def test_halfdose_and_step_counts(tmp_path, tokenizer, monkeypatch, capsys, n):
    monkeypatch.setattr(aw2, "load_tokenizer", lambda name: tokenizer)
    early_path, spread_path = tmp_path / "early.jsonl", tmp_path / "spread.jsonl"
    write_events(early_path, [event("a", i) for i in range(n)])
    write_events(spread_path, [event("a", i, "spread") for i in range(n)])
    output = tmp_path / "pools"
    report_path = tmp_path / "reports" / "control.json"
    arguments = [str(early_path), str(spread_path), "--out-dir", str(output),
                 "--report", str(report_path), "--halfdose", "--seed", "7"]
    aw2.main(arguments)
    matched = aw2.read_events(output / "pool_aw2_early_matched.jsonl")
    half_path = output / "pool_aw2_early_halfdose.jsonl"
    half = aw2.read_events(half_path)
    assert half == matched[::2]
    assert len(half) == (n + 1) // 2
    report = json.loads(report_path.read_text())
    assert report["seed"] == 7
    for arm, rows in [("early_matched", matched), ("early_halfdose", half)]:
        stats = report["arms"][arm]
        assert stats["total_rows"] == len(rows)
        assert stats["steps"] == (len(rows) + 7) // 8
    stdout = capsys.readouterr().out
    assert f"ceil({n}/8) = {(n + 7) // 8} steps" in stdout
    assert f"ceil({len(half)}/8) = {(len(half) + 7) // 8} steps" in stdout
    before = half_path.read_bytes()
    aw2.main(arguments)
    assert half_path.read_bytes() == before


@pytest.mark.parametrize("response,expected", [
    ("```python\nprint(apis.api_docs.show_app_descriptions())\n```", True),
    ("apis.spotify.login(username='u')", True),
    ("```PYTHON\napis.spotify.LOGIN()\n```", True),
    ("```\napis.api_docs.show_api_doc(api_name='login')\n```", True),
    ("```py\nx = 1\n```\n```python\napis.spotify.login()\n```", True),
    ("Login and api_docs are done.\n```python\napis.supervisor.complete_task()\n```", False),
    ("apis.spotify.logout()", False),
    ("login_count = 1; api_docs_cache = {}", False),
])
def test_doc_login_detection(response, expected):
    assert aw2.is_doc_login(response) is expected


def test_report_statistics(tokenizer):
    rows = [event("a", 0, response="apis.api_docs.show()", _probe_pos=0.1, _event_dU=0.2),
            event("a", 1, response="apis.spotify.login()", _probe_pos=0.3, _event_dU=-0.2),
            event("b", 2, response="a b c", _probe_pos=None, _event_dU=0,
                  turn_index=3, _demo_len=8),
            event("b", 3, response="a", _probe_pos=None, _event_dU=None)]
    report = aw2.summarize_pool(rows, tokenizer)
    assert report["doc_login_rows"] == 2
    assert report["doc_login_fraction"] == 0.5
    assert report["mean_yT_tokens"] == 1.5
    assert report["total_yT_tokens"] == 6
    assert report["mean_probe_pos"] == pytest.approx(0.3)
    assert report["probe_pos_sources"] == {"recorded": 2, "derived": 1, "missing": 1}
    assert report["dU_sign_counts"] == {"positive": 1, "negative": 1, "zero": 1, "missing": 1}
    assert report["dU_sign_proportions"] == dict.fromkeys(report["dU_sign_counts"], 0.25)
    assert aw2.probe_position({"_event_turn": 0, "_demo_len": 4}) == (0.25, "derived")
    assert aw2.probe_position({"turn_index": 0, "_demo_len": 0}) == (None, "missing")
    empty = aw2.summarize_pool([], tokenizer)
    assert empty["mean_probe_pos"] is None
    assert empty["mean_yT_tokens"] is None
    assert empty["doc_login_fraction"] is None
    assert empty["steps"] == 0


@pytest.mark.parametrize("early,spread", [
    ([], []), ([event("a", 0)], [event("b", 0, "spread")]),
    ([event("a", 0, _event_overflow=True)], [event("a", 0, "spread")]),
])
def test_no_eligible_shared_tasks(tmp_path, early, spread, capsys):
    paths = [tmp_path / "early.jsonl", tmp_path / "spread.jsonl"]
    for path, rows in zip(paths, [early, spread]):
        write_events(path, rows)
    output = tmp_path / "pools"
    with pytest.raises(SystemExit) as exc:
        aw2.main([*(str(p) for p in paths), "--out-dir", str(output)])
    assert exc.value.code == 2
    assert "No shared tasks with non-overflow events" in capsys.readouterr().err
    assert not output.exists()
