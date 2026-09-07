from dataclasses import replace
import json
from pathlib import Path

import pytest
import yaml

from bfas.rtd.baselines.cli import choose_trials, grid_trials, main, selected_config, tune
from bfas.rtd.baselines.config import BaselineConfig, GRIDS, load_config
from bfas.rtd.persistence import atomic_json, file_hash
from test_rtd_baselines_helpers import TinyTokenizer, make_fixture, prepared_fixture, runtime_config


def test_tune_executes_all_twelve_fresh_seed_zero_trials_with_same_extra_support_schedule(tmp_path):
    evidence, evidence_path, _ = prepared_fixture(tmp_path)
    prepared = tmp_path/"prepared"
    prepared.mkdir()
    for recipe in ("b1", "b2"):
        cfg = BaselineConfig(recipe=recipe, runtime=runtime_config(), evidence_path=str(evidence_path), evidence_hash=file_hash(evidence_path))
        (prepared/f"A1_{recipe}_slots.yaml").write_text(yaml.safe_dump(cfg.to_dict()))
    spec = dict(version="c27-tuning-v1", recipes=["b1", "b2"], seed=0, pool="A1", view="slots",
                development_seed=20000, grids={r: GRIDS[r] for r in ("b1", "b2")})
    path = tmp_path/"tuning.yaml"
    path.write_text(yaml.safe_dump(spec))
    calls = []
    def trial_runner(config, directory, *, development_schedule):
        calls.append((config, directory, development_schedule))
        assert not directory.exists()  # each trial gets a separate theta0 run
        assert config.seed == 0 and config.slots == 288
        assert len(development_schedule) == 3
        assert all(g["role"] == "support_development" and not g["reused"] for g in development_schedule)
        return dict(passed=True, exposure=dict(matched=True),
            development_scores=[dict(score=.5)]*3,
            resources=dict(teacher_tokens=31, new_teacher_tokens=0, tuning_rollouts=18, T_update=1000))
    result = tune(path, prepared, tmp_path/"tuning", trial_runner=trial_runner)
    assert len(calls) == result["trial_count"] == 12
    assert [(c.recipe, c.lr, c.commits) for c, _, _ in calls] == [
        (recipe, hp["lr"], hp["commits"]) for recipe, hp in grid_trials(["b1", "b2"])]
    assert all(call[2] == calls[0][2] for call in calls)
    for recipe, lr in (("b1", 3e-6), ("b2", 1e-6)):
        winner = result["selected"][recipe]
        assert winner["hyperparameters"] == dict(lr=lr, commits=36)
        assert winner["tied_best"] == 6 and not winner["identifiable"]
    assert result["total_resources"]["tuning_rollouts"] == 216
    cfg = load_config(prepared/"A1_b1_slots.yaml")
    selected = selected_config(replace(cfg, pool="A0", view="tokens"), tmp_path/"tuning/selection.json")
    assert selected.pool == "A0" and selected.view == "tokens" and selected.lr == 3e-6
    assert selected.evidence_hash == cfg.evidence_hash


def test_full_grid_has_24_trials_and_both_b3_variants():
    trials = list(grid_trials(list(GRIDS)))
    assert len(trials) == 24
    assert sum(r == "b3_sft" for r, _ in trials) == sum(r == "b3_mix" for r, _ in trials) == 3


def test_cli_prepare_template_refuses_absent_final_sources_before_any_model_or_run_access(tmp_path):
    path = tmp_path/"suite.yaml"
    path.write_text(yaml.safe_dump(dict(version="c27-suite-v1", bank="no-live-bank",
        hardware_class_hash=None, pools={}, recipes=["b1"], views=["slots"], common="unused")))
    with pytest.raises(ValueError, match="finished source"):
        main(["prepare", "--config", str(path), "--out", str(tmp_path/"generated")])
    assert not (tmp_path/"generated").exists()


def test_cli_help_is_import_only(capsys):
    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0
    assert "prepare" in capsys.readouterr().out


def test_checked_in_recipe_templates_all_validate_without_live_source_reads():
    root = Path(__file__).resolve().parents[1]/"configs/rtd/baselines"
    for name in ("bfcl_b1_sft", "bfcl_b2_pairwise", "bfcl_b3_sft_pg", "bfcl_b3_mix_pg",
                 "bfcl_b4_meta_reweight", "exposure_slots", "exposure_tokens"):
        config = load_config(root/(name+".yaml"))
        with pytest.raises(ValueError, match="template"):
            config.validate_ready()


def test_prepare_suite_reads_recipe_paths_and_writes_self_contained_bound_configs(tmp_path, monkeypatch):
    from bfas.rtd.baselines import cli
    from bfas.rtd.baselines.evidence import prepare as real_prepare
    source, bank, support, manifest = make_fixture(tmp_path)
    common = tmp_path/"common.yaml"
    common.write_text(yaml.safe_dump(dict(runtime={}, seed=0, new_teacher_calls=False, evaluate_after_round=False)))
    recipe = tmp_path/"b1.yaml"
    recipe.write_text(yaml.safe_dump(dict(recipe="b1", lr=3e-6, commits=72)))
    suite = tmp_path/"suite.yaml"
    suite.write_text(yaml.safe_dump(dict(version="c27-suite-v1", bank=str(bank),
        hardware_class_hash=manifest["hardware_hash"], pools={"A1": str(source)},
        recipes={"b1": str(recipe)}, views=["slots", "tokens"], common=str(common))))
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "prepare", lambda *a, **k: real_prepare(*a, **k, tokenizer=TinyTokenizer(), support=support))
    cli.prepare_suite(suite, tmp_path/"generated")
    for view in ("slots", "tokens"):
        config = load_config(tmp_path/f"generated/A1_b1_{view}.yaml").validate_ready()
        assert config.view == view and config.lr == 3e-6 and config.commits == 72
        assert config.runtime == runtime_config()
        assert config.evidence_hash == file_hash(config.evidence_path)
        assert config.update_tokens == [672]*3
    assert (tmp_path/"generated/prepared.json").exists()
