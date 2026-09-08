"""CPU-only scoring inventory, frozen data/model and audited-chain boundaries."""
from copy import deepcopy
import json
import shutil
import subprocess

import pytest

from bfas.rtd.benchmarks import alfworld_identity as identity
from bfas.rtd.persistence import digest
from rtd_alfworld_evaluation_fixtures import campaign, audit, changed_harness, put, round_checkpoint


def test_identity_without_git_gpu_or_importing_environment(campaign, monkeypatch):
    c = campaign
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("identity must not launch a subprocess"))
    current, hashes = identity.guard_manifest(c.root, json.loads(json.dumps(c.manifest)), hardware=c.hardware)
    assert current == c.manifest
    assert hashes == [c.manifest["harness_hash"]]
    assert not (c.root / ".git").exists()
    expected = current["evaluation_harness"]["expected"]
    assert len(expected["task_ids"]) == expected["count"] == 140
    assert expected["task_ids_hash"] == digest(expected["task_ids"])
    assert len(expected["data_manifest"]) == 420


def test_portable_content_identity(campaign):
    c = campaign
    other = c.root.parent / "relocated"
    shutil.copytree(c.root, other)
    relative = c.data.relative_to(c.root)
    relocated = identity.make_manifest(other, c.config, data_root=other / relative,
                                       model_path=other / "model", hardware=c.hardware)
    assert relocated["evaluation_harness"] == c.manifest["evaluation_harness"]
    assert relocated["checkpoint"] == c.manifest["checkpoint"]


@pytest.mark.parametrize("field,value", [("alfworld_evaluation_split", "valid_unseen"),
    ("alfworld_expected_eval_tasks", 139), ("alfworld_max_episode_steps", 39),
    ("alfworld_student_react", 1), ("evaluation_temperature", True),
    ("evaluation_temperature", 1), ("max_context_tokens", 0),
    ("max_action_tokens_by_benchmark", {"alfworld": {"agent_action": 32}})])
def test_official_config_is_explicit(campaign, field, value):
    with pytest.raises(ValueError):
        identity.checked_config(dict(campaign.config, **{field: value}))


@pytest.mark.parametrize("name", ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"])
def test_actual_model_config_tokenizer_mutations_rejected(campaign, name):
    c = campaign
    path = c.model / name
    put(path, dict(changed=True, chat_template="changed") if name.endswith(".json") else "changed weights")
    with pytest.raises(ValueError):
        identity.guard_manifest(c.root, c.manifest, hardware=c.hardware)


@pytest.mark.parametrize("mutation", ["missing_task", "extra_task", "world", "traj", "initial_state"])
def test_data_population_and_bytes_are_frozen(campaign, mutation):
    c = campaign
    tid = c.manifest["evaluation_harness"]["expected"]["task_ids"][0]
    path = c.data / "valid_seen" / tid
    if mutation == "missing_task":
        (path / "game.tw-pddl").unlink()
    elif mutation == "extra_task":
        shutil.copytree(path, path.with_name("extra_trial"))
    else:
        name = {"world": "game.tw-pddl", "traj": "traj_data.json", "initial_state": "initial_state.pddl"}[mutation]
        put(path / name, (path / name).read_text() + " ")
    with pytest.raises(ValueError):
        identity.guard_manifest(c.root, c.manifest, hardware=c.hardware)


def test_training_and_operational_drift_outside_scoring_inventory(campaign):
    c = campaign
    put(c.root / "src/bfas/rtd/new_training_code.py", "training changes")
    path = c.root / "src/bfas/adapters/alfworld.py"
    path.write_text(path.read_text().replace('teacher_name = os.environ.get(', 'irrelevant_name = os.environ.get('))
    evaluate = c.root / "src/bfas/rtd/benchmarks/alfworld_evaluation.py"
    original = evaluate.read_text()
    changed = original.replace('tag=f"alfworld/{run_name}/{tag}"', 'tag=f"alfworld/log/{run_name}/{tag}"')
    assert changed != original
    evaluate.write_text(changed)
    assert identity.guard_manifest(c.root, c.manifest, hardware=c.hardware)[0]["harness_hash"] == c.manifest["harness_hash"]


@pytest.mark.parametrize("name,old,new", [
    ("src/bfas/adapters/alfworld.py", "history[-8:]", "history[-7:]"),
    ("src/bfas/rtd/benchmarks/alfworld_support.py", "lines[-8:]", "lines[-7:]"),
    ("src/alfworld_eval.py", 'return "look"', 'return "inventory"'),
    ("src/bfas/rtd/benchmarks/alfworld_evaluation.py", "range(40)", "range(39)"),
    ("src/bfas/rtd/benchmarks/alfworld_evaluation.py", 'split="valid_seen"', 'split="train"'),
    ("src/bfas/rtd/benchmarks/alfworld_evaluation.py", "100 * successes / 140", "successes / 140"),
])
def test_scientific_edits_change_identity(campaign, name, old, new):
    c = campaign
    # _episode itself is unused; its historical window is represented by the
    # exact C26-B prompt_messages selector instead.
    if name.endswith("adapters/alfworld.py"):
        old, new = 'observation[-2000:]', 'observation[-1000:]'
    path = c.root / name
    assert old in path.read_text()
    path.write_text(path.read_text().replace(old, new))
    with pytest.raises(ValueError, match="harness"):
        identity.guard_manifest(c.root, c.manifest, hardware=c.hardware)


def test_missing_scoring_class_and_import_alias_fail_closed(campaign):
    c = campaign
    path = "src/bfas/adapters/alfworld.py"
    content = (c.root / path).read_text()
    with pytest.raises(ValueError, match="class"):
        identity.scoring_projection(path, content.replace("class ALFWorldAdapter(", "class MissingAdapter("))
    projection = identity.scoring_projection(path, content)
    assert identity.scoring_projection(path, content.replace("import re\n", "import alternate_re as re\n")) != projection
    base = "src/bfas/adapter.py"
    with pytest.raises(ValueError, match="class"):
        identity.scoring_projection(base, (c.root / base).read_text().replace("class BenchmarkAdapter(", "class MissingBase("))


def test_device_instance_changes_reuse_same_class(campaign):
    c = campaign
    moved = deepcopy(c.hardware)
    moved["metadata"].update(uuid="different-device", hostname="different-node")
    assert identity.guard_manifest(c.root, c.manifest, hardware=moved)[0]["hardware_hash"] == c.manifest["hardware_hash"]
    moved["hard"]["memory"] += 1
    with pytest.raises(ValueError, match="class"):
        identity.guard_manifest(c.root, c.manifest, hardware=moved)


def test_rtd_round_lora_layout_binds_full_training_checkpoint(campaign):
    c = campaign
    directory = round_checkpoint(c)
    manifest = identity.make_manifest(c.root, c.config, data_root=c.data, model_path=c.model,
        hardware=c.hardware, run_directory=directory, round_number=1)
    assert manifest["checkpoint"]["loading"] == "adapter_overlay"
    assert manifest["round_checkpoint"]["round"] == 1
    assert identity.guard_manifest(c.root, manifest, hardware=c.hardware)
    put(directory / "round-1/lora/adapter_model.safetensors", "corrupt lora")
    with pytest.raises(ValueError, match="checkpoint"):
        identity.guard_manifest(c.root, manifest, hardware=c.hardware)


def test_multiple_audited_updates_preserve_original_manifest(campaign):
    c = campaign
    original = deepcopy(c.manifest)
    middle = changed_harness(c)
    supplement = audit(c, c.manifest["evaluation_harness"], middle)
    current = changed_harness(c)
    supplement = audit(c, middle, current, supplement=supplement)
    _, hashes = identity.guard_manifest(c.root, c.manifest, hardware=c.hardware, supplement=supplement)
    assert hashes == [digest(current), digest(middle), c.manifest["harness_hash"]]
    assert c.manifest == original


@pytest.mark.parametrize("fault", ["manifest", "link", "new_hash", "endpoint", "evidence", "bfcl", "model", "expected", "config", "tokenizer"])
def test_disconnected_or_foreign_audit_refused(campaign, fault):
    c = campaign
    current = changed_harness(c)
    supplement = audit(c, c.manifest["evaluation_harness"], current)
    note = supplement["identity_updates"][0]
    if fault == "manifest":
        supplement["manifest_hash"] = digest(dict(c.manifest, arm="another-run"))
    elif fault == "link":
        note["previous_identity"]["harness_hash"] = "disconnected"
    elif fault == "new_hash":
        note["new_identity"]["harness_hash"] = "corrupt"
    elif fault == "endpoint":
        supplement["harness_hash"] = "wrong"
    elif fault == "evidence":
        note["audit"] = {}
    elif fault == "bfcl":
        supplement["version"] = "rtd-c25j-audited-identity-v1"
    else:
        current = deepcopy(current)
        current[fault] = {"changed": True}
        supplement = audit(c, c.manifest["evaluation_harness"], current)
    with pytest.raises(ValueError):
        identity.audited_harness_hashes(c.manifest, current, supplement)
