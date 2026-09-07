"""C22 evaluation launcher contracts; no SLURM or inference is launched."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/cc_eval_arms_hpg.slurm"


def test_tag_loop_and_skip_logic():
    for pattern in (
        '${CC_EVAL_TAGS:-cc_s1_A_s0 cc_s1_B_s0 cc_s1_C_s0}',
        'for tag in "${tags[@]}"; do',
        'if [ -f "results/bfcl_std/$tag/data_overall.csv" ]; then',
        'already scored, skip',
        'bash tools/bfcl_std_campaign.sh 0 "$port" "$tag"',
        '#SBATCH --mem=240G',
        '#SBATCH --time=06:00:00',
        '#SBATCH --gres=gpu:b200:1',
    ):
        subprocess.run(["grep", "-Fq", "--", pattern, str(SCRIPT)], check=True)


@pytest.mark.parametrize("selection", [None, "custom_A custom_B custom_C"])
def test_launcher_skips_scores_and_reports_axes_from_submit_dir(tmp_path, selection):
    root = tmp_path / "checkout with spaces"
    (root / "tools").mkdir(parents=True)
    (root / ".venv/bin").mkdir(parents=True)
    (root / ".venv/bin/activate").touch()
    shutil.copy(ROOT / "tools/bfcl_print_scores.py", root / "tools")
    tags = (selection or "cc_s1_A_s0 cc_s1_B_s0 cc_s1_C_s0").split()
    score = ("Overall Acc,Non-Live AST Acc,Live Acc,Multi Turn Acc,Memory Acc,"
             "Irrelevance Detection\n46.06,79.67,77.65,51.12,24.95,82.07\n")
    saved = root / f"results/bfcl_std/{tags[1]}/data_overall.csv"
    saved.parent.mkdir(parents=True)
    saved.write_text(score)
    # Replace only the external cache paths in a spooled copy.
    script = tmp_path / "spooled.slurm"
    script.write_text(SCRIPT.read_text().replace(
        "/blue/fsu-compsci-dept/xc25.fsu/hq/tools", str(tmp_path / "cache")))
    stub = root / "tools/bfcl_std_campaign.sh"
    stub.write_text(
        '#!/bin/bash\nset -eu\n'
        'printf "%s %s %s\\n" "$1" "$2" "$3" >> calls.txt\n'
        'mkdir -p "results/bfcl_std/$3"\n'
        f"cat > \"results/bfcl_std/$3/data_overall.csv\" <<'CSV'\n{score}CSV\n")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("CC_", "SLURM_"))}
    env.update(SLURM_SUBMIT_DIR=str(root), SLURM_JOB_ID="41233505")
    if selection:
        env["CC_EVAL_TAGS"] = selection
    result = subprocess.run(["bash", str(script)], cwd=tmp_path, env=env,
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert (root / "calls.txt").read_text().splitlines() == [
        f"0 43505 {tags[0]}", f"0 43507 {tags[2]}"]
    assert f"{tags[1]} already scored, skip" in result.stdout
    assert saved.read_text() == score
    for tag in tags:
        assert f"{tag} OVERALL=46.06" in result.stdout
        assert f"| {tag} | 79.67 | 77.65 | 51.12 | 24.95 | 82.07 |" in result.stdout
    assert "Non-Live | Live | Multi-turn | Memory | Irrelevance" in result.stdout


def test_extracted_printer_preserves_proxy_output(tmp_path):
    score = tmp_path / "data_overall.csv"
    score.write_text("Overall Acc,Multi Turn Acc,Irrelevance Detection\n46.06,51.12,82.07\n")
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/bfcl_print_scores.py"), str(score),
         "base", "--prefix", "[fast]", "--proxy"],
        check=True, capture_output=True, text=True)
    assert result.stdout == (
        "[fast] base PROXY Overall=46.06  Non-Live AST=N/A  Live=N/A  "
        "Multi Turn=51.12  Memory=N/A  Web Search=N/A  Relevance=N/A  Irrelevance=82.07\n")
