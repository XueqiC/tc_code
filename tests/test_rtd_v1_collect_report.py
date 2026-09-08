"""Synthetic archives: official provenance, accounting, missing work and read-only I/O."""
import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

from tools import rtd_v1_collect_report as report


def put_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def put_events(path, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text())


def put_csv(path, overall):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report.AXES.values()))
        writer.writeheader()
        writer.writerow(dict(zip(report.AXES.values(), [f"{overall}%", "80%", "77", "49.50%", "N/A", "83%", "75%", "12.5%"])))


def make_arm(root, machine, arm, *, rounds=2):
    directory = root / ("results/rtd_v1" if machine == "rai" else "results/rtd_v1_hpg") / (f"rai_{arm}" if machine == "rai" else arm)
    campaigns = root / ("results/bfcl_std" if machine == "rai" else "results/bfcl_std_hpg")
    internal_arm = "R0" if arm == "R0" else "R1"  # R1s is identified by run path.
    put_json(directory / "manifest.json", dict(arm=internal_arm, budget_ceilings=[100, 200, 300],
             recorded_bank_usage=55370, config=dict(mode="sealed_replay", benchmark="bfcl", training_seed=0,
             rounds=3, decision_steps_per_round=[1, 4, 7, 10], max_new_packages_per_decision=1),
             hardware={"gpu": f"GPU-{arm}"} if machine == "rai" else {"hard": {"gpu": "B200"}}))
    teacher = [dict(kind="reserve", query_id="a", cap=20),
               dict(kind="reveal", query_id="a", cost=10, confidence="exact"),
               dict(kind="reveal", query_id="b", cost=20, confidence="estimated"),
               dict(kind="release", query_id="unused")]
    if rounds >= 2:
        teacher += [dict(kind="authorize", budget=200), dict(kind="reveal", query_id="c", cost=40, confidence="estimated")]
    steps, checkpoints, compute = [], [], []
    for r in range(1, rounds + 1):
        owned = ["a", "b"] if r == 1 else ["a", "b", "c"]
        checkpoints.append(dict(round=r, owned=sorted(owned), actual_spend=30 if r == 1 else 70, authorized_budget=r * 100))
        for step in (1, 4, 7, 10):
            steps.append(dict(round=r, step=step, decision=True,
                              selection=dict(query_ids=["a", None, "b"], probabilities=[.25, .5, .25], selected=None),
                              truncation=dict(truncated_rollouts=int(step == 1), actual_feedback_reused=True),
                              malformed_feedback=dict(malformed_rollouts=int(step == 1), actual_feedback_reused=True)))
        for bad, truncated in ((True, True), (True, False), (False, False)):
            compute.append(dict(kind="feedback_rollout", round=r, step=1,
                                rollout=dict(malformed=bad, truncated=truncated, actions=[])))
        tag = f"archived_{arm}_r{r}"
        compute.append(dict(kind="evaluation_begin", round=r, tag=tag))
        # The hpg remote output_directory deliberately names bfcl_std, as real archives do.
        evaluation = dict(validation={"complete": True}, overall_accuracy_percent=46 + r + (machine == "hpg"),
                          output_directory=f"/remote/results/bfcl_std/{tag}", campaign_identity={"checkpoint": {"round": r}})
        put_json(directory / f"evaluation-{r}.json", evaluation)
        put_csv(campaigns / tag / "data_overall.csv", evaluation["overall_accuracy_percent"])
    put_json(directory / "trajectory.json", dict(checkpoints=checkpoints, steps=steps))
    put_events(directory / "teacher.jsonl", teacher)
    put_events(directory / "compute.jsonl", compute)
    return directory


@pytest.fixture
def archives(tmp_path):
    for machine in report.MACHINES:
        for arm in report.ARMS:
            make_arm(tmp_path, machine, arm)
    return tmp_path


def arm_rows(collected, machine="rai", arm="R0"):
    return next(a["rows"] for a in collected["arms"] if (a["machine"], a["arm"]) == (machine, arm))


def inventory(root):
    return {str(p.relative_to(root)): (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
            for p in root.rglob("*") if p.is_file()}


def test_six_arms_three_rounds_official_axes_and_costs(archives):
    before = inventory(archives / "results")
    collected = report.collect(archives, base_overall=46.74)
    assert inventory(archives / "results") == before
    assert len(collected["arms"]) == 6
    assert all(len(a["rows"]) == 3 for a in collected["arms"])
    first, second, third = arm_rows(collected)
    assert first["official"]["scores"] == {"Overall": 47., "Non-Live AST": 80., "Live": 77., "Multi-Turn": 49.5,
                                           "Memory": None, "Irrelevance": 83., "Relevance": 75., "Web Search": 12.5}
    assert first["packages_purchased"] == 2  # reserve/release are not purchases.
    assert first["actual_spend"] == 30
    assert first["spend_ceiling_ratio"] == .3
    assert first["actual_exact"] == 10 and first["actual_estimated"] == 20
    assert second["packages_purchased"] == 1 and second["cumulative_packages"] == 3
    assert second["actual_spend"] == 70 and second["cap_ceiling"] == 200
    assert second["spend_bank_percent"] == pytest.approx(70 / 55370 * 100)
    assert third["actual_spend"] is None and third["packages_purchased"] is None
    assert not third["complete"]
    assert arm_rows(collected, "hpg", "R1s")[0]["official"]["scores"]["Overall"] == 48.
    assert "/bfcl_std_hpg/" in arm_rows(collected, "hpg", "R1s")[0]["official"]["csv"]
    assert collected["scope"]["completed_rounds"] == 12
    assert collected["scope"]["seed_count"] == 1
    assert collected["scope"]["rai_gpu_model_count"] == 3
    assert all(a["planned_windows"] == 12 and a["max_packages_per_window"] == 1 for a in collected["arms"])


def test_empty_is_null_not_a_position_or_name():
    stats = report.selection_stats(dict(query_ids=["empty", None, "package"], probabilities=[.25, .5, .25]))
    assert stats["candidate_pool_size"] == 2 and stats["option_count"] == 3
    assert stats["empty_probability"] == .5
    assert stats["entropy_ratio"] == pytest.approx(1.5 * math.log(2) / math.log(3))
    assert stats["package_entropy_ratio"] == pytest.approx(1.)
    assert report.selection_stats(dict(query_ids=[None], probabilities=[1]))["entropy_ratio"] is None
    assert report.selection_stats(dict(query_ids=[None, "a", "b"], probabilities=[1, 0, 0]))["package_entropy_ratio"] is None
    assert report.selection_stats(dict(query_ids=[None, "a", "b"], probabilities=[0, 1, 0]))["package_entropy_ratio"] == 0


@pytest.mark.parametrize("ids,p", [([None, "a"], [.2, .2]), ([None, "a"], [1.1, -.1]),
                                  ([None, "a"], [math.nan, 0]), (["empty", "a"], [.5, .5]),
                                  ([None, "a", "a"], [.5, .25, .25])])
def test_invalid_distributions_are_not_silently_normalized(ids, p):
    with pytest.raises(ValueError):
        report.selection_stats(dict(query_ids=ids, probabilities=p))


def test_current_round_rollouts_and_reused_feedback(archives):
    first, second, _ = arm_rows(report.collect(archives, base_overall=46.74))
    for row in (first, second):
        assert row["rollouts"]["malformed"]["sampled"] == 2
        assert row["rollouts"]["malformed"]["committed"] == 1
        assert row["rollouts"]["truncated"]["sampled"] == 1
        assert row["rollouts"]["truncated"]["committed"] == 1


def test_report_fallback_uses_differences_and_preserves_unknowns(archives):
    directory = archives / "results/rtd_v1/rai_R0"
    (directory / "compute.jsonl").unlink()
    t = read_json(directory / "trajectory.json")
    for s in t["steps"]:
        s.pop("truncation")
        s.pop("malformed_feedback")
    put_json(directory / "trajectory.json", t)
    put_json(directory / "report/budget_curve.json", dict(rows=[
        dict(round=1, run="/original/rai_R0", sampled_malformed_rollouts=3, committed_malformed_rollouts=2,
             sampled_truncated_rollouts=5, committed_truncated_rollouts=4),
        dict(round=2, run="/original/rai_R0", sampled_malformed_rollouts=12, committed_malformed_rollouts=10,
             sampled_truncated_rollouts=7, committed_truncated_rollouts=6)]))
    first, second, _ = arm_rows(report.collect(archives, base_overall=46.74))
    assert first["rollouts"]["malformed"]["sampled"] == 3
    assert second["rollouts"]["malformed"]["sampled"] == 9
    assert second["rollouts"]["malformed"]["committed"] == 8
    assert second["rollouts"]["truncated"]["sampled"] == 2
    payload = read_json(directory / "report/budget_curve.json")
    payload["rows"].pop(0)
    put_json(directory / "report/budget_curve.json", payload)
    first, second, _ = arm_rows(report.collect(archives, base_overall=46.74))
    assert first["rollouts"]["malformed"]["sampled"] is None
    assert second["rollouts"]["malformed"]["sampled"] is None


def test_legacy_missing_flags_are_not_zero(archives):
    path = archives / "results/rtd_v1/rai_R0/compute.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    for e in events:
        if e["kind"] == "feedback_rollout":
            e["rollout"].pop("malformed")
    put_events(path, events)
    first = arm_rows(report.collect(archives, base_overall=46.74))[0]
    assert first["rollouts"]["malformed"]["sampled"] is None
    assert first["rollouts"]["truncated"]["sampled"] == 1


def test_partially_recorded_rollouts_retain_lower_bound(archives):
    path = archives / "results/rtd_v1/rai_R0/compute.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    events[0]["rollout"].pop("malformed")
    put_events(path, events)
    first = arm_rows(report.collect(archives, base_overall=46.74))[0]
    assert first["rollouts"]["malformed"]["sampled"] is None
    assert first["rollouts"]["malformed"]["flags_recorded"] == 2
    assert report.rollout_cell(first, "malformed", "sampled") == "≥1 (未完成)"


def test_missing_arms_and_rounds_are_retained(tmp_path):
    make_arm(tmp_path, "hpg", "R1", rounds=1)
    collected = report.collect(tmp_path, base_overall=46.74)
    assert collected["scope"]["available_arms"] == 1
    assert collected["scope"]["completed_rounds"] == 1
    assert len(arm_rows(collected, "rai", "R1s")) == 3
    text = report.render_markdown(collected, "curve.png")
    assert "| rai | R0 | (未完成) | (未完成) | (未完成) |" in text
    assert "| rai | R1s | 3 | (未完成)" in text
    assert "55,370" in text and "12 个决策窗口" in text and "每窗口至多 1 包" in text


def test_base_manifest_config_precedence_and_required_fallback(archives):
    with pytest.raises(ValueError, match="--base-overall"):
        report.collect(archives)
    path = archives / "results/rtd_v1/rai_R0/manifest.json"
    m = read_json(path)
    m["config"]["base_overall"] = "46.74%"
    put_json(path, m)
    collected = report.collect(archives, base_overall=12)
    assert collected["base"]["overall"] == 46.74
    assert collected["base"]["sources"][0].endswith("config.base_overall")
    m["base_overall"] = 47
    put_json(path, m)
    with pytest.raises(ValueError, match="conflicting"):
        report.collect(archives)


def test_campaign_identity_tag_and_event_fallback(archives):
    directory = archives / "results/rtd_v1/rai_R0"
    path = directory / "evaluation-1.json"
    e = read_json(path)
    e.pop("output_directory")
    e["campaign_identity"]["tag"] = "archived_R0_r1"
    put_json(path, e)
    first = arm_rows(report.collect(archives, base_overall=46.74))[0]
    assert first["official"]["tag_source"] == "campaign_identity.tag"
    assert first["complete"]
    path.unlink()
    first = arm_rows(report.collect(archives, base_overall=46.74))[0]
    assert first["official"]["tag_source"] == "compute event.tag"
    assert first["official"]["scores"]["Overall"] == 47
    assert report.score_cell(first) == "47.00 (未完成)"


def test_mismatched_score_and_ambiguous_campaign_fail(archives):
    path = archives / "results/bfcl_std/archived_R0_r1/data_overall.csv"
    put_csv(path, 1)
    with pytest.raises(ValueError, match="disagrees"):
        report.collect(archives, base_overall=46.74)
    put_csv(path, 47)
    text = path.read_text()
    path.write_text(text + text.splitlines()[1] + "\n")
    with pytest.raises(ValueError, match="one official model row"):
        report.collect(archives, base_overall=46.74)


def test_evaluation_bound_to_another_checkpoint_is_rejected(archives):
    path = archives / "results/rtd_v1/rai_R0/evaluation-1.json"
    evaluation = read_json(path)
    evaluation["identity"] = {"checkpoint": {"round": 2}}
    put_json(path, evaluation)
    with pytest.raises(ValueError, match="another checkpoint"):
        report.collect(archives, base_overall=46.74)


def test_machine_matching_is_ordered_cumulative_prefix(archives):
    collected = report.collect(archives, base_overall=46.74)
    assert collected["machine_effects"][0]["hpg_minus_rai"] == 1
    assert collected["machine_effects"][2]["identical_purchase_prefix"] is None
    directory = archives / "results/rtd_v1_hpg/R0"
    events = [json.loads(line) for line in (directory / "teacher.jsonl").read_text().splitlines()]
    events[1], events[2] = events[2], events[1]  # Same set, different purchase order.
    put_events(directory / "teacher.jsonl", events)
    effects = report.collect(archives, base_overall=46.74)["machine_effects"]
    assert effects[0]["identical_purchase_prefix"] is False
    assert effects[0]["hpg_minus_rai"] is None


def test_partial_next_round_spend_does_not_contaminate_checkpoint(archives):
    directory = archives / "results/rtd_v1/rai_R0"
    events = [json.loads(line) for line in (directory / "teacher.jsonl").read_text().splitlines()]
    events += [dict(kind="authorize", budget=300), dict(kind="reveal", query_id="pending", cost=7, confidence="exact")]
    put_events(directory / "teacher.jsonl", events)
    first, second, third = arm_rows(report.collect(archives, base_overall=46.74))
    assert first["actual_spend"] == 30 and second["actual_spend"] == 70
    assert third["actual_spend"] == 77 and third["packages_purchased"] == 1
    assert not third["complete"] and not third["training_complete"]


def test_torn_journal_is_read_only_and_recorded(archives):
    path = archives / "results/rtd_v1/rai_R0/teacher.jsonl"
    with path.open("ab") as stream:
        stream.write(b'{"kind":"reveal"')
    before = inventory(archives / "results")
    collected = report.collect(archives, base_overall=46.74)
    assert inventory(archives / "results") == before
    assert any("未结束的末行" in message for message in collected["warnings"])
    assert next(f for f in collected["inputs"] if f["path"] == str(path))["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="invalid journal record"):
        report.collect(archives, base_overall=46.74)


def test_missing_reveal_fails_reconciliation(archives):
    path = archives / "results/rtd_v1/rai_R0/teacher.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    put_events(path, [e for e in events if e.get("query_id") != "b"])
    with pytest.raises(ValueError, match="checkpoint ownership/spend"):
        report.collect(archives, base_overall=46.74)


def test_cli_creates_chinese_sidecar_and_matplotlib_figure_without_input_writes(archives, capsys):
    before = inventory(archives / "results")
    out = archives / "docs/summary_zh.md"
    assert report.main(["--root", str(archives), "--out", str(out), "--base-overall", "46.74"]) == 0
    assert inventory(archives / "results") == before
    payload = read_json(out.with_suffix(".json"))
    figure = out.with_name("rtd_v1_budget_curve.png")
    assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert figure.stat().st_size > 10000
    assert payload["scope"]["completed_rounds"] == 12
    assert "基座学生" in out.read_text() and "范围限制" in out.read_text()
    assert "(rtd_v1_budget_curve.png)" in out.read_text()
    assert "| rai | R0 | 47.00 | 48.00 | (未完成) |" in capsys.readouterr().out
    outputs = inventory(archives / "docs")
    with pytest.raises(FileExistsError):
        report.write_report(payload, out, figure)
    assert inventory(archives / "docs") == outputs
    with pytest.raises(ValueError, match="read-only"):
        report.write_report(payload, archives / "results/forbidden.md", archives / "docs/new.png")
