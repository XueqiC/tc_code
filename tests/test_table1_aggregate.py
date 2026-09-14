"""CPU-only Table 1 aggregation tests; all result fixtures live in tmp_path."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import table1_aggregate as table1


def write_run(root, benchmark="alfworld", method="smartad", seed=0, **overrides):
    metrics = {
        "complete": True,
        "overall_accuracy_percent": {"alfworld": 60, "hotpotqa": 40, "bamboogle": 50, "musique": 30, "2wiki": 35}[benchmark] + 2 * seed,
        "teacher_tokens_charged": 28_000 + 500 * seed,
    }
    if benchmark in table1.QA_BENCHMARKS:
        metrics.update(em=metrics["overall_accuracy_percent"] / 100, f1=.5 + .1 * seed)
    if benchmark == "alfworld":
        metrics["per_category"] = {"pick_and_place": {"success_rate": .6}}
    metrics.update(overrides)
    path = root / f"table1_{benchmark}_{method}_s{seed}" / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics), encoding="utf-8")
    return path


@pytest.fixture
def full_campaign(tmp_path):
    root = tmp_path / "synthetic_results"
    for benchmark in table1.BENCHMARKS:
        for method in table1.METHODS:
            for seed in table1.SEEDS:
                write_run(root, benchmark, method, seed)
    return root


def test_all_seeds_sample_std_primary_metrics_and_spend(full_campaign):
    before = {p: p.read_bytes() for p in full_campaign.rglob("*") if p.is_file()}
    # Unrelated directories (including other seeds) must not affect the 45 cells.
    write_run(full_campaign, seed=3, overall_accuracy_percent=100)
    summary = table1.aggregate(full_campaign)
    assert summary["counts"] == dict(expected=45, complete=45, missing=0, incomplete=0, invalid=0)
    assert not summary["issues"]
    for benchmark, mean in (("alfworld", 62), ("hotpotqa", 42), ("bamboogle", 52), ("musique", 32), ("2wiki", 37)):
        for method in table1.METHODS:
            cell = summary["benchmarks"][benchmark][method]
            score = cell["overall_accuracy_percent"]
            assert score == dict(n=3, seeds=[0, 1, 2], per_seed={"0": mean - 2, "1": mean, "2": mean + 2},
                                 mean=mean, std=2.)
            if table1.INITIAL[benchmark] is None:
                assert cell["improvement_pp"]["mean"] is None
            else:
                assert cell["improvement_pp"]["mean"] == pytest.approx(mean - table1.INITIAL[benchmark])
                assert cell["improvement_pp"]["std"] == pytest.approx(2)
            spend = cell["teacher_tokens_charged"]
            assert spend["per_seed"] == {"0": 28_000, "1": 28_500, "2": 29_000}
            assert spend["max"] == 29_000 and spend["n"] == 3
            assert spend["std"] == 500 and spend["seeds_over_cap"] == []
    assert summary["teacher_tokens_charged_max"] == 29_000
    assert summary["teacher_spend_receipt_count"] == 45
    assert summary["methods"]["smartad"]["average_improvement_pp"] is None
    assert summary["benchmarks"]["alfworld"]["sad"]["runs"][0]["per_category"] == {
        "pick_and_place": {"success_rate": .6}}
    assert all(p.read_bytes() == contents for p, contents in before.items())


@pytest.mark.parametrize("benchmark", table1.BENCHMARKS)
def test_unsuffixed_seed_zero_is_the_same_logical_cell(full_campaign, benchmark):
    expected = table1.aggregate(full_campaign)
    path = full_campaign / f"table1_{benchmark}_smartad_s0"
    unsuffixed = path.with_name(path.name.removesuffix("_s0"))
    path.rename(unsuffixed)
    summary = table1.aggregate(full_campaign)
    # Only the source path changes; counts, per-seed metrics, spend and LaTeX stay identical.
    expected["benchmarks"][benchmark]["smartad"]["runs"][0]["metrics_path"] = str(unsuffixed / "metrics.json")
    assert summary == expected
    assert table1.latex_body(summary) == table1.latex_body(expected)


@pytest.mark.parametrize("metrics_in_unsuffixed", [False, True])
def test_seed_zero_prefers_the_directory_with_metrics(full_campaign, metrics_in_unsuffixed):
    expected = table1.aggregate(full_campaign)
    suffixed = full_campaign / "table1_alfworld_smartad_s0"
    unsuffixed = full_campaign / "table1_alfworld_smartad"
    unsuffixed.mkdir()
    if metrics_in_unsuffixed:
        (suffixed / "metrics.json").rename(unsuffixed / "metrics.json")
        expected["benchmarks"]["alfworld"]["smartad"]["runs"][0]["metrics_path"] = str(unsuffixed / "metrics.json")
    summary = table1.aggregate(full_campaign)
    assert summary == expected
    assert summary["counts"]["complete"] == 45
    assert summary["teacher_spend_receipt_count"] == 45


def test_unsuffixed_seed_zero_without_metrics_is_reported(tmp_path):
    directory = tmp_path / "table1_alfworld_smartad"
    directory.mkdir()
    summary = table1.aggregate(tmp_path)
    run = summary["benchmarks"]["alfworld"]["smartad"]["runs"][0]
    assert run["cell"] == "table1_alfworld_smartad_s0"
    assert run["metrics_path"] == str(directory / "metrics.json")
    assert run["status"] == "missing"
    assert run["issues"] == ["missing metrics.json"]


@pytest.mark.parametrize("duplicate_contents", ["identical", "different", "malformed"])
def test_duplicate_seed_zero_metrics_report_a_conflict(tmp_path, duplicate_contents):
    suffixed = write_run(tmp_path)
    unsuffixed = tmp_path / "table1_alfworld_smartad" / "metrics.json"
    unsuffixed.parent.mkdir()
    contents = suffixed.read_text()
    if duplicate_contents == "different":
        metrics = json.loads(contents)
        metrics["overall_accuracy_percent"] = 99
        contents = json.dumps(metrics)
    elif duplicate_contents == "malformed":
        contents = '{"complete":'
    unsuffixed.write_text(contents)
    with pytest.raises(ValueError, match="conflicting directories.*seed 0") as exc:
        table1.aggregate(tmp_path)
    assert str(suffixed) in str(exc.value)
    assert str(unsuffixed) in str(exc.value)


@pytest.mark.parametrize("valid_seed_exists", [False, True])
def test_doubled_suffix_directories_are_listed_and_never_counted(full_campaign, valid_seed_exists):
    if not valid_seed_exists:
        path = full_campaign / "table1_alfworld_smartad_s1" / "metrics.json"
        path.unlink()
        path.parent.rmdir()
    expected = table1.aggregate(full_campaign)
    ignored = []
    for suffix in ("_s0_s0", "_s1_s1", "_s2_s2", "_s1_s2", "_s12_s12", "_s1_s1_s1"):
        path = write_run(full_campaign, seed=9, overall_accuracy_percent=99,
                         teacher_tokens_charged=999_999)
        name = f"table1_alfworld_smartad{suffix}"
        path.parent.rename(full_campaign / name)
        ignored.append(name)
    summary = table1.aggregate(full_campaign)
    expected["ignored_directories"] = sorted(ignored)
    assert summary == expected
    assert table1.latex_body(summary) == table1.latex_body(expected)
    human = table1.human_table(summary)
    assert "Ignored directories (doubled seed suffix): " + ", ".join(sorted(ignored)) in human


def test_hotpotqa_f1_is_secondary_json_only(full_campaign):
    # A deliberately different em field proves the primary comes from overall_accuracy_percent.
    write_run(full_campaign, "hotpotqa", seed=0, em=.99)
    summary = table1.aggregate(full_campaign)
    cell = summary["benchmarks"]["hotpotqa"]["smartad"]
    assert cell["overall_accuracy_percent"]["mean"] == 42
    assert cell["f1"]["n"] == 3
    assert cell["f1"]["per_seed"] == {"0": .5, "1": .6, "2": .7}
    assert cell["f1"]["mean"] == pytest.approx(.6)
    assert cell["f1"]["std"] == pytest.approx(.1)
    assert cell["f1"]["unit"] == "fraction (0-1)"
    assert "f1" not in table1.human_table(summary).lower()
    assert "f1" not in table1.latex_body(summary).lower()
    assert all("f1" in summary["benchmarks"][b]["smartad"] for b in table1.QA_BENCHMARKS)


def test_missing_seed_is_reported_and_available_seeds_remain_visible(full_campaign):
    path = full_campaign / "table1_alfworld_smartad_s1" / "metrics.json"
    path.unlink()
    path.parent.rmdir()
    summary = table1.aggregate(full_campaign)
    assert summary["missing_cells"] == ["table1_alfworld_smartad_s1"]
    score = summary["benchmarks"]["alfworld"]["smartad"]["overall_accuracy_percent"]
    assert score["n"] == 2 and score["seeds"] == [0, 2]
    assert score["per_seed"] == {"0": 60, "2": 64}
    assert score["mean"] == 62 and score["std"] == pytest.approx(8 ** .5)
    assert summary["methods"]["smartad"]["seed_counts"] == {b: 2 if b == "alfworld" else 3 for b in table1.BENCHMARKS}
    assert summary["methods"]["smartad"]["average_improvement_pp"] is None
    human = table1.human_table(summary)
    assert "(2/3)" in human and "s0=60.0000,s2=64.0000" in human
    assert "table1_alfworld_smartad_s1: missing directory" in human


def test_incomplete_cell_is_excluded_but_spend_is_preserved(full_campaign):
    write_run(full_campaign, seed=2, complete=False, overall_accuracy_percent=99,
              teacher_tokens_charged=30_001)
    summary = table1.aggregate(full_campaign)
    assert summary["incomplete_cells"] == ["table1_alfworld_smartad_s2"]
    assert summary["counts"]["complete"] == 44
    cell = summary["benchmarks"]["alfworld"]["smartad"]
    assert cell["overall_accuracy_percent"]["mean"] == 61
    assert cell["overall_accuracy_percent"]["n"] == 2
    assert cell["overall_accuracy_percent"]["std"] == pytest.approx(2 ** .5)
    assert cell["runs"][2]["overall_accuracy_percent"] == 99  # Retained only as a receipt.
    assert cell["teacher_tokens_charged"]["per_seed"]["2"] == 30_001
    assert cell["teacher_tokens_charged"]["max"] == 30_001
    assert cell["teacher_tokens_charged"]["n"] == 3
    assert cell["teacher_tokens_charged"]["seeds_over_cap"] == [2]
    assert summary["teacher_tokens_charged_max"] == 30_001
    assert summary["over_cap_cells"] == ["table1_alfworld_smartad_s2"]
    assert "complete=false" in table1.human_table(summary)


def test_one_seed_std_is_undefined_and_average_requires_all_benchmarks_and_references(tmp_path):
    write_run(tmp_path, seed=0)
    write_run(tmp_path, "hotpotqa", seed=1)
    summary = table1.aggregate(tmp_path)
    score = summary["benchmarks"]["alfworld"]["smartad"]["overall_accuracy_percent"]
    assert score["std"] is None and score["n"] == 1 and score["mean"] == 60
    assert summary["methods"]["smartad"]["average_improvement_pp"] is None
    assert summary["methods"]["smartad"]["benchmark_count"] == 2
    for b in table1.OOD_BENCHMARKS:
        write_run(tmp_path, b, seed=2)
    summary = table1.aggregate(tmp_path, dict(alfworld=55, hotpotqa=35, bamboogle=45, musique=25, **{"2wiki": 30}))
    assert summary["methods"]["smartad"]["average_improvement_pp"] == 7.8  # (5 + 7 + 9 + 9 + 9) / 5
    assert summary["methods"]["sad"]["average_improvement_pp"] is None


def test_absent_root_is_an_empty_report(tmp_path):
    missing_root = tmp_path / "not_arrived"
    summary = table1.aggregate(missing_root)
    assert not missing_root.exists()
    assert summary["counts"] == dict(expected=45, complete=0, missing=45, incomplete=0, invalid=0)
    assert len(summary["missing_cells"]) == 45
    assert summary["teacher_tokens_charged_max"] is None
    score = summary["benchmarks"]["bamboogle"]["kang_action_list_summary"]["overall_accuracy_percent"]
    assert score == dict(n=0, seeds=[], per_seed={}, mean=None, std=None)
    assert summary["methods"]["kang_action_list_summary"]["average_improvement_pp"] is None
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("contents, expected_status", [
    (None, "missing"),
    ('{"complete":', "invalid"),
    ("[]", "invalid"),
    ('{"complete": false}', "incomplete"),
    ('{"complete": "false", "overall_accuracy_percent": 60, "teacher_tokens_charged": 12}', "invalid"),
    ('{"complete": true, "overall_accuracy_percent": null, "teacher_tokens_charged": 12}', "invalid"),
    ('{"complete": true, "overall_accuracy_percent": 60, "teacher_tokens_charged": -1}', "invalid"),
])
def test_pending_and_malformed_metrics_do_not_crash(tmp_path, contents, expected_status):
    path = write_run(tmp_path)
    if contents is None:
        path.unlink()
    else:
        path.write_text(contents)
    summary = table1.aggregate(tmp_path)
    run = summary["benchmarks"]["alfworld"]["smartad"]["runs"][0]
    assert run["status"] == expected_status
    assert run["issues"]
    assert summary["benchmarks"]["alfworld"]["smartad"]["overall_accuracy_percent"]["n"] == 0
    json.dumps(summary, allow_nan=False)


def test_missing_secondary_metric_has_its_own_seed_count(full_campaign):
    write_run(full_campaign, "hotpotqa", seed=1, f1=None)
    summary = table1.aggregate(full_campaign)
    cell = summary["benchmarks"]["hotpotqa"]["smartad"]
    assert cell["overall_accuracy_percent"]["n"] == 3
    assert cell["f1"]["n"] == 2 and cell["f1"]["seeds"] == [0, 2]
    assert "excluded from secondary statistics" in summary["issues"][0]["messages"][0]


def test_latex_matches_paper_rows_column_order_precision_and_counts(full_campaign):
    body = table1.latex_body(table1.aggregate(full_campaign))
    rows = [line for line in body.splitlines() if " & " in line]
    labels = [line.split(" & ")[0] for line in rows]
    assert labels == [r"\rowcolor{rowhead}Initial student (Gemma-4-12B)",
                      *[table1.LABELS[m] for m in table1.METHODS],
                      "GAD", r"\rowcolor{rowours}\textbf{RTD (ours)}"]
    assert all(line.count(" & ") == 6 and r"\\" in line for line in rows)
    assert rows[0].endswith(r"$56.4$ & $38.2$ & $\cdot$ & $\cdot$ & $\cdot$ & --\\ % Single reference evaluation where available; dots are pending.")
    assert rows[1] == (table1.LABELS["smartad"] + r" & $62.0\pm 2.0$ {\scriptsize ($n=3$)} & "
                       r"$42.0\pm 2.0$ {\scriptsize ($n=3$)} & "
                       r"$52.0\pm 2.0$ {\scriptsize ($n=3$)} & "
                       r"$32.0\pm 2.0$ {\scriptsize ($n=3$)} & "
                       r"$37.0\pm 2.0$ {\scriptsize ($n=3$)} & $\cdot$\\")
    assert r"\dagger" not in rows[1]
    assert "29000 / 30000 tokens; 45/45 receipts" in body


def test_latex_partial_and_empty_cells_use_daggers_and_no_fabricated_std(tmp_path):
    write_run(tmp_path, seed=0, overall_accuracy_percent=61.234)
    write_run(tmp_path, "hotpotqa", seed=0)
    write_run(tmp_path, "hotpotqa", seed=2)
    body = table1.latex_body(table1.aggregate(tmp_path))
    row = next(line for line in body.splitlines() if line.startswith(table1.LABELS["smartad"] + " &"))
    assert r"${61.2\pm \text{--}}^{\dagger}$ {\scriptsize ($n=1$)}" in row
    assert r"${42.0\pm 2.8}^{\dagger}$ {\scriptsize ($n=2$)}" in row
    assert r"${\cdot}^{\dagger}$ {\scriptsize ($n=0$)}" in row
    assert row.endswith(r" & $\cdot$\\")
    for b in table1.OOD_BENCHMARKS:
        write_run(tmp_path, b)
    row = next(line for line in table1.latex_body(table1.aggregate(tmp_path, {b: 30 for b in table1.OOD_BENCHMARKS})).splitlines()
               if line.startswith(table1.LABELS["smartad"] + " &"))
    assert row.endswith(r" & $+6.7^{\dagger}$\\")


def test_zero_scores_and_identical_seeds_are_valid_percentages(tmp_path):
    for seed in table1.SEEDS:
        write_run(tmp_path, seed=seed, overall_accuracy_percent=0, teacher_tokens_charged=0)
    summary = table1.aggregate(tmp_path)
    cell = summary["benchmarks"]["alfworld"]["smartad"]
    assert cell["overall_accuracy_percent"]["n"] == 3
    assert cell["overall_accuracy_percent"]["mean"] == 0
    assert cell["overall_accuracy_percent"]["std"] == 0
    assert cell["improvement_pp"]["mean"] == -56.4
    assert cell["teacher_tokens_charged"]["max"] == 0
    assert r"$0.0\pm 0.0$ {\scriptsize ($n=3$)}" in table1.latex_body(summary)


def test_cli_writes_both_outputs_and_accepts_all_initial_values(full_campaign, tmp_path):
    json_path, tex_path = tmp_path / "report/summary.json", tmp_path / "report/body.tex"
    proc = subprocess.run([
        sys.executable, str(ROOT / "tools/table1_aggregate.py"),
        "--results-root", str(full_campaign), "--json-out", str(json_path), "--latex-out", str(tex_path),
        "--initial-alfworld", "50", "--initial-hotpotqa", "30", "--initial-bamboogle", "40", "--initial-musique", "20", "--initial-2wiki", "25",
    ], check=True, capture_output=True, text=True, cwd=tmp_path,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"))
    summary = json.loads(json_path.read_text())
    assert summary["initial_student_percent"] == dict(alfworld=50, hotpotqa=30, bamboogle=40, musique=20, **{"2wiki": 25})
    assert all(row["average_improvement_pp"] == 12 for row in summary["methods"].values())
    assert "$50.0$ & $30.0$ & $40.0$ & $20.0$ & $25.0$" in tex_path.read_text()
    assert "complete=45, missing=0, incomplete=0, invalid=0" in proc.stdout
    assert "s0=60.0000,s1=62.0000,s2=64.0000" in proc.stdout
    assert "JSON:" in proc.stdout and "LaTeX:" in proc.stdout


def test_cli_rejects_output_inside_input_root(full_campaign, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        table1.main(["--results-root", str(full_campaign), "--json-out", str(full_campaign / "summary.json"),
                     "--latex-out", str(tmp_path / "body.tex")])
    assert exc.value.code == 2
    assert "outside the results root" in capsys.readouterr().err
    assert not (full_campaign / "summary.json").exists()


def test_cli_reports_duplicate_conflict_without_writing_outputs(full_campaign, tmp_path, capsys):
    path = full_campaign / "table1_alfworld_smartad" / "metrics.json"
    path.parent.mkdir()
    path.write_bytes((full_campaign / "table1_alfworld_smartad_s0" / "metrics.json").read_bytes())
    json_path, tex_path = tmp_path / "summary.json", tmp_path / "body.tex"
    with pytest.raises(SystemExit) as exc:
        table1.main(["--results-root", str(full_campaign), "--json-out", str(json_path),
                     "--latex-out", str(tex_path)])
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "conflicting directories" in error and "seed 0" in error
    assert "table1_alfworld_smartad/metrics.json" in error
    assert "table1_alfworld_smartad_s0/metrics.json" in error
    assert not json_path.exists() and not tex_path.exists()


@pytest.mark.parametrize("benchmark", table1.OOD_BENCHMARKS)
def test_standalone_ood_em_f1_enters_table_without_fabricating_training_spend(tmp_path, benchmark):
    path = tmp_path / f"table1_{benchmark}_smartad" / "metrics.json"
    path.parent.mkdir()
    path.write_text(json.dumps(dict(complete=True, em=.4, f1=.6, n=125,
                                   config=dict(dataset=benchmark))))
    summary = table1.aggregate(tmp_path)
    cell = summary["benchmarks"][benchmark]["smartad"]
    assert cell["overall_accuracy_percent"]["mean"] == 40
    assert cell["f1"]["mean"] == .6 and cell["primary_metric"] == "exact match"
    assert cell["teacher_tokens_charged"]["n"] == 0
    assert cell["runs"][0]["teacher_tokens_charged"] is None
    assert cell["improvement_pp"]["mean"] is None
    assert summary["methods"]["smartad"]["average_improvement_pp"] is None
    assert "BFCL" not in table1.latex_body(summary)


def test_bfcl_archives_do_not_enter_current_suite(tmp_path):
    directory = tmp_path / "table1_bfcl_smartad"
    directory.mkdir()
    (directory / "metrics.json").write_text(json.dumps(dict(complete=True,
        overall_accuracy_percent=100, teacher_tokens_charged=123)))
    before = (directory / "metrics.json").read_bytes()
    summary = table1.aggregate(tmp_path)
    assert summary["benchmark_order"] == ["alfworld", "hotpotqa", "bamboogle", "musique", "2wiki"]
    assert summary["counts"]["complete"] == 0 and summary["counts"]["expected"] == 45
    assert (directory / "metrics.json").read_bytes() == before


@pytest.mark.parametrize("config", [None, [], "bamboogle"])
def test_malformed_ood_config_is_reported_without_crashing(tmp_path, config):
    path = tmp_path / "table1_bamboogle_smartad" / "metrics.json"
    path.parent.mkdir()
    path.write_text(json.dumps(dict(complete=True, em=.4, f1=.6, config=config)))
    summary = table1.aggregate(tmp_path)
    assert summary["benchmarks"]["bamboogle"]["smartad"]["runs"][0]["status"] == "invalid"
