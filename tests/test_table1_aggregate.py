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
        "overall_accuracy_percent": {"alfworld": 60, "hotpotqa": 40, "bfcl": 50}[benchmark] + 2 * seed,
        "teacher_tokens_charged": 28_000 + 500 * seed,
    }
    if benchmark == "hotpotqa":
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
    # Unrelated directories (including other seeds) must not affect the 27 cells.
    write_run(full_campaign, seed=3, overall_accuracy_percent=100)
    summary = table1.aggregate(full_campaign)
    assert summary["counts"] == dict(expected=27, complete=27, missing=0, incomplete=0, invalid=0)
    assert not summary["issues"]
    for benchmark, mean in (("alfworld", 62), ("hotpotqa", 42), ("bfcl", 52)):
        for method in table1.METHODS:
            cell = summary["benchmarks"][benchmark][method]
            score = cell["overall_accuracy_percent"]
            assert score == dict(n=3, seeds=[0, 1, 2], per_seed={"0": mean - 2, "1": mean, "2": mean + 2},
                                 mean=mean, std=2.)
            assert cell["improvement_pp"]["mean"] == pytest.approx(mean - table1.INITIAL[benchmark])
            assert cell["improvement_pp"]["std"] == pytest.approx(2)
            spend = cell["teacher_tokens_charged"]
            assert spend["per_seed"] == {"0": 28_000, "1": 28_500, "2": 29_000}
            assert spend["max"] == 29_000 and spend["n"] == 3
            assert spend["std"] == 500 and spend["seeds_over_cap"] == []
    assert summary["teacher_tokens_charged_max"] == 29_000
    assert summary["teacher_spend_receipt_count"] == 27
    assert summary["methods"]["smartad"]["average_improvement_pp"] == pytest.approx((5.6 + 3.8 + 6.4) / 3)
    assert summary["benchmarks"]["alfworld"]["sad"]["runs"][0]["per_category"] == {
        "pick_and_place": {"success_rate": .6}}
    assert all(p.read_bytes() == contents for p, contents in before.items())


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
    assert "f1" not in summary["benchmarks"]["bfcl"]["smartad"]


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
    assert summary["methods"]["smartad"]["seed_counts"] == dict(alfworld=2, hotpotqa=3, bfcl=3)
    assert summary["methods"]["smartad"]["average_improvement_pp"] is not None
    human = table1.human_table(summary)
    assert "(2/3)" in human and "s0=60.0000,s2=64.0000" in human
    assert "table1_alfworld_smartad_s1: missing directory" in human


def test_incomplete_cell_is_excluded_but_spend_is_preserved(full_campaign):
    write_run(full_campaign, seed=2, complete=False, overall_accuracy_percent=99,
              teacher_tokens_charged=30_001)
    summary = table1.aggregate(full_campaign)
    assert summary["incomplete_cells"] == ["table1_alfworld_smartad_s2"]
    assert summary["counts"]["complete"] == 26
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


def test_one_seed_std_is_undefined_and_average_requires_three_benchmarks(tmp_path):
    write_run(tmp_path, seed=0)
    write_run(tmp_path, "hotpotqa", seed=1)
    summary = table1.aggregate(tmp_path)
    score = summary["benchmarks"]["alfworld"]["smartad"]["overall_accuracy_percent"]
    assert score["std"] is None and score["n"] == 1 and score["mean"] == 60
    assert summary["methods"]["smartad"]["average_improvement_pp"] is None
    assert summary["methods"]["smartad"]["benchmark_count"] == 2
    write_run(tmp_path, "bfcl", seed=2)
    summary = table1.aggregate(tmp_path, dict(alfworld=55, hotpotqa=35, bfcl=45))
    assert summary["methods"]["smartad"]["average_improvement_pp"] == 7  # (5 + 7 + 9) / 3
    assert summary["methods"]["sad"]["average_improvement_pp"] is None


def test_absent_root_is_an_empty_report(tmp_path):
    missing_root = tmp_path / "not_arrived"
    summary = table1.aggregate(missing_root)
    assert not missing_root.exists()
    assert summary["counts"] == dict(expected=27, complete=0, missing=27, incomplete=0, invalid=0)
    assert len(summary["missing_cells"]) == 27
    assert summary["teacher_tokens_charged_max"] is None
    score = summary["benchmarks"]["bfcl"]["kang"]["overall_accuracy_percent"]
    assert score == dict(n=0, seeds=[], per_seed={}, mean=None, std=None)
    assert summary["methods"]["kang"]["average_improvement_pp"] is None
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
    assert labels == [r"\rowcolor{rowhead}Initial student (Gemma-4-12B)", "SmartAD", "SAD",
                      "Agent Distillation", "GAD", r"\rowcolor{rowours}\textbf{RTD (ours)}"]
    assert all(line.count(" & ") == 4 and r"\\" in line for line in rows)
    assert rows[0].endswith(r"$56.4$ & $38.2$ & $45.6$ & --\\ % Single reference evaluation per benchmark (n=1).")
    assert rows[1] == (r"SmartAD & $62.0\pm 2.0$ {\scriptsize ($n=3$)} & "
                       r"$42.0\pm 2.0$ {\scriptsize ($n=3$)} & "
                       r"$52.0\pm 2.0$ {\scriptsize ($n=3$)} & $+5.3$\\")
    assert r"\dagger" not in rows[1]
    assert "29000 / 30000 tokens; 27/27 receipts" in body


def test_latex_partial_and_empty_cells_use_daggers_and_no_fabricated_std(tmp_path):
    write_run(tmp_path, seed=0, overall_accuracy_percent=61.234)
    write_run(tmp_path, "hotpotqa", seed=0)
    write_run(tmp_path, "hotpotqa", seed=2)
    body = table1.latex_body(table1.aggregate(tmp_path))
    row = next(line for line in body.splitlines() if line.startswith("SmartAD &"))
    assert r"${61.2\pm \text{--}}^{\dagger}$ {\scriptsize ($n=1$)}" in row
    assert r"${42.0\pm 2.8}^{\dagger}$ {\scriptsize ($n=2$)}" in row
    assert r"${\cdot}^{\dagger}$ {\scriptsize ($n=0$)}" in row
    assert row.endswith(r" & $\cdot$\\")
    write_run(tmp_path, "bfcl")
    row = next(line for line in table1.latex_body(table1.aggregate(tmp_path)).splitlines()
               if line.startswith("SmartAD &"))
    assert row.endswith(r" & $+4.3^{\dagger}$\\")


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
        "--initial-alfworld", "50", "--initial-hotpotqa", "30", "--initial-bfcl", "40",
    ], check=True, capture_output=True, text=True, cwd=tmp_path,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"))
    summary = json.loads(json_path.read_text())
    assert summary["initial_student_percent"] == dict(alfworld=50, hotpotqa=30, bfcl=40)
    assert all(row["average_improvement_pp"] == 12 for row in summary["methods"].values())
    assert "$50.0$ & $30.0$ & $40.0$" in tex_path.read_text()
    assert "complete=27, missing=0, incomplete=0, invalid=0" in proc.stdout
    assert "s0=60.0000,s1=62.0000,s2=64.0000" in proc.stdout
    assert "JSON:" in proc.stdout and "LaTeX:" in proc.stdout


def test_cli_rejects_output_inside_input_root(full_campaign, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        table1.main(["--results-root", str(full_campaign), "--json-out", str(full_campaign / "summary.json"),
                     "--latex-out", str(tmp_path / "body.tex")])
    assert exc.value.code == 2
    assert "outside the results root" in capsys.readouterr().err
    assert not (full_campaign / "summary.json").exists()
