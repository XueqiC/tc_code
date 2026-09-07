from collections import Counter
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from bfas.rtd.baselines.config import BaselineConfig
from bfas.rtd.baselines.feedback import collect_groups
from bfas.rtd.baselines.journal import ResourceJournal
from bfas.rtd.baselines.runner import BaselineRunner
from bfas.rtd.experiment import RTDExperiment
from bfas.rtd.functional_step import lora_parameters
from bfas.rtd.identity import verified_checkpoint
from bfas.rtd.persistence import atomic_json, digest, file_hash
from test_rtd_baselines_helpers import prepared_fixture, runtime_config, tiny_backend


@pytest.fixture(autouse=True)
def single_cpu_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def run_tiny(tmp_path, recipe, *, view="slots", commits=36):
    evidence, evidence_path, support = prepared_fixture(tmp_path)
    cfg = BaselineConfig(recipe=recipe, view=view, optimizer="fixed" if recipe == "b4" else "adamw",
        lr=3e-5, commits=commits, runtime=runtime_config(), evidence_path=str(evidence_path), evidence_hash=file_hash(evidence_path),
        update_tokens=[400]*3 if view == "tokens" else [], pg_reserved_tokens=[144]*3)
    cfg.validate_ready()
    config = dict(cfg.runtime, baseline=cfg.to_dict())
    manifest = dict(config=config, config_hash=digest(config), arm=recipe, synthetic=True)
    directory = tmp_path/"baseline"
    directory.mkdir()
    atomic_json(directory/"manifest.json", manifest)
    journal = ResourceJournal(directory/"compute.jsonl")
    backend = tiny_backend()
    runner = BaselineRunner(cfg, evidence, manifest, directory, backend, support, journal)
    return runner.run(), runner, journal, directory, support


@pytest.mark.parametrize("recipe", ["b1", "b2", "b3_sft", "b3_mix", "b4"])
def test_real_update_loops_save_official_verified_checkpoint_layout_and_resource_totals(tmp_path, recipe):
    result, runner, journal, directory, support = run_tiny(tmp_path, recipe)
    assert result["passed"] and result["complete"] and result["exposure"]["matched"]
    assert result["optimizer_commits"] == 36
    assert result["resources"]["raw_slots"] == result["resources"]["weighted_slots"] == 288
    assert result["resources"]["teacher_tokens"] == 31
    assert result["resources"]["new_teacher_tokens"] == 0
    assert result["resources"]["gpu_seconds"] == 0
    assert result["resources"]["T_update"] == sum(runner.actual_tokens)
    assert runner.steps[0]["start_hash"] != runner.steps[-1]["actual_hash"]
    for r in (1, 2, 3):
        checkpoint = verified_checkpoint(directory, runner.manifest, r)
        assert checkpoint["optimizer_commits"] == r*12
        assert (directory/f"round-{r}/lora/adapter_model.safetensors").exists()
        saved = torch.load(directory/f"round-{r}/round_state.pt", weights_only=False)
        assert saved["optimizer_commits"] == r*12
        assert saved["optimizer"] is not None if recipe != "b4" else saved["optimizer"] is None
    if recipe.startswith("b3") or recipe == "b4":
        assert result["feedback_rollouts"] == len(support.calls) == 108
        assert result["resources"]["feedback_rollouts"] == 108
        rows = [e for e in journal.events if e["kind"] == "feedback_rollout"]
        for row in runner.steps:
            if row["decision"]:
                actual = [e for e in rows if (e["round"], e["step"]) == (row["round"], row["step"])]
                assert len({e["sampling_parameter_hash"] for e in actual}) == 1
                expected_policy = row["actual_hash"] if recipe == "b4" else row["start_hash"]
                assert row["sampling_parameter_hash"] == expected_policy
        if recipe.startswith("b3"):
            assert result["pg_nonzero_commits"] > 0
        else:
            updates = [e for e in journal.events if e["kind"] == "teacher_weight_update"]
            assert len(updates) == 12
            assert all(e["features_off"] and not e["transport_term"] for e in updates)
    else:
        assert not support.calls
    if recipe == "b2":
        assert result["resources"]["reference_input_tokens"] > 0
        assert all(source.frozen_snapshot_id == runner.backend.identity(runner.initial)
                   for samples in torch.load(directory/"round-1/round_state.pt", weights_only=False)["source_cache"].values()
                   for source in samples)


def test_b1_72_steps_repartitions_slots_without_inflating_exposure(tmp_path):
    result, runner, _, directory, _ = run_tiny(tmp_path, "b1", commits=72)
    assert result["optimizer_commits"] == 72
    assert all(e["raw_slots"] == 4 for e in runner.steps)
    assert [e["step"] for e in runner.steps[:24] if e["decision"]] == [1, 7, 13, 19]
    assert verified_checkpoint(directory, runner.manifest, 3)["optimizer_commits"] == 72


def test_vt_run_spends_exact_constant_cost_target_and_journals_distribution(tmp_path):
    result, runner, journal, _, _ = run_tiny(tmp_path, "b1", view="tokens")
    assert result["exposure"]["matched"]
    assert result["exposure"]["actual_tokens"] == [400]*3
    assert result["resources"]["raw_slots"] == 300  # token view is allowed a different number of slots
    distributions = [e for e in journal.events if e["kind"] == "exposure_schedule"]
    assert len(distributions) == 3
    assert all(e["c"] and e["q"] and e["rho"] and e["round_frozen"] for e in distributions)


def test_feedback_uses_exact_rtd_choose_tasks_limits_and_injected_parents(tmp_path):
    evidence, _, support = prepared_fixture(tmp_path)
    reference = RTDExperiment.__new__(RTDExperiment)
    reference.support, reference.rng = support, np.random.default_rng(0)
    for r in (1, 2, 3):
        reference.state = dict(smoke=False, feedback={p for p in support.states if int(p, 16)%2 != (r-1)%2})
        expected = dict(reference.choose_feedback_tasks())  # existing RTD public method, no runner/live IO
        for g in evidence["feedback"]:
            if g["round"] == r:
                assert {t["parent_hash"]: t["rollout_count"] for t in g["tasks"]} == expected
    groups = [g for g in evidence["feedback"] if (g["round"], g["step"]) == (1, 4)]
    b = tiny_backend()
    journal = ResourceJournal(tmp_path/"compute-test.jsonl")
    got = collect_groups(groups, support, b, lora_parameters(b.model), torch.Generator().manual_seed(6), None, journal)
    assert got.metadata["rollouts"] == 12 and got.metadata["groups"] == 2
    assert Counter(p for p, _ in support.calls) == Counter({f"{1:064x}": 8, f"{3:064x}": 4})
    assert len({policy for _, policy in support.calls}) == 1


def test_feedback_loo_is_per_group_zero_advantage_retains_all_quotas(tmp_path):
    evidence, _, support = prepared_fixture(tmp_path)
    original = support.feedback
    def group_constant(*args):
        rollout = original(*args)
        return replace(rollout, reward=float(len(support.calls) > 6))
    support.feedback = group_constant
    groups = [g for g in evidence["feedback"] if (g["round"], g["step"]) == (1, 4)]
    b = tiny_backend()
    result = collect_groups(groups, support, b, lora_parameters(b.model), torch.Generator().manual_seed(2), None,
                            ResourceJournal(tmp_path/"feedback.jsonl"))
    assert len(support.calls) == 12
    assert not result.metadata["identifiable"]
    assert all(g.eq(0).all() for g in result.gradient.values())


def test_journal_nested_gpu_time_not_double_counted(tmp_path):
    journal = ResourceJournal(tmp_path/"compute.jsonl")
    with journal.phase("outer"):
        with journal.phase("inner"):
            journal.usage("update", input_tokens=10, action_tokens=3)
    summary = journal.summary()
    ends = [e for e in journal.events if e["kind"] == "compute_end"]
    assert summary["wall_seconds"] == ends[-1]["wall_seconds"]
    assert summary["update_input_tokens"] == summary["update_backward_tokens"] == 10
    assert summary["T_update"] == 10
