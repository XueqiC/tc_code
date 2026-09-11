"""Official full evaluations, with separate Kang SAG receipts and GPU timing."""
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from ..persistence import atomic_json, digest, tree_hash
from .paper_data import STUDENT


def protocol(benchmark):
    if benchmark == "alfworld":
        return dict(benchmark=benchmark, split="valid_seen", tasks=140, temperature=0.,
                    max_steps=40, max_action_tokens=256, backend="vllm", evaluator="ALFWorldAdapter.evaluate")
    return dict(benchmark=benchmark, split="all", tasks=5217, temperature=.001,
                handler="gemma4_fc", model=STUDENT+"-FC", backend="vllm", evaluator="bfcl_eval CLI")


def bfcl_commands(root, merged, out, *, kang=False, port=8930):
    root, merged, out = Path(root), Path(merged), Path(out)
    python = root/"envs/bfcl/.venv/bin/python"
    if not python.exists():
        python = root/"envs/bfcl-venv/bin/python"
    cli = [str(python), str(root/"tools/baseline_bfcl.py")]
    common = ["--model", STUDENT+"-FC", "--test-category", "all", "--result-dir", str(out/"resultdir")]
    generate = cli + ["generate", *common, "--backend", "vllm", "--num-gpus", "1",
        "--local-model-path", str(merged), "--temperature", "0.001", "--num-threads", "8",
        "--gpu-memory-utilization", "0.85"]
    evaluate = cli + ["evaluate", *common, "--score-dir", str(out/"scoredir")]
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               LOCAL_SERVER_PORT=str(port), VLLM_USE_FLASHINFER_SAMPLER="0",
               BFCL_PROJECT_ROOT=str(out/"harness_runtime"),
               PATH=str(root/"envs/vllm-serve/.venv/bin")+os.pathsep+os.environ.get("PATH", ""))
    env.pop("LOCAL_SERVER_ENDPOINT", None)
    env.pop("BASELINE_KANG_AUDIT", None)
    if kang:
        env["BASELINE_KANG_AUDIT"] = str(out/"kang_votes.jsonl")
    return generate, evaluate, env


def bfcl_metrics(root, out):
    from ..evaluation import official_expectations, validate_evaluation
    from ...adapters.bfcl import read_score_summaries
    expected = official_expectations(root)
    count = sum(len(ids) for ids in expected["generation"].values())
    if count != 5217:
        raise ValueError(f"official BFCL inventory differs from frozen 5,217: {count}")
    validation = validate_evaluation(expected, out/"resultdir", out/"scoredir")
    csv_path = out/"scoredir/data_overall.csv"
    with csv_path.open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError("expected one official overall model score")
    score = float(rows[0]["Overall Acc"].strip().removesuffix("%"))
    if not math.isfinite(score) or not 0 <= score <= 100:
        raise ValueError("invalid official BFCL overall")
    categories = read_score_summaries(out/"scoredir")
    return dict(complete=True, overall_accuracy_percent=score, tasks=count,
        per_category={k: dict(**v, accuracy_percent=100*v["verified"]/v["total"] if v["total"] else None)
                      for k, v in categories.items()}, validation=validation, expected=expected)


def run_bfcl(root, merged, out, *, kang, port):
    from ..evaluation_lock import reserve_port
    out.mkdir(parents=True, exist_ok=False)
    (out/"harness_runtime").mkdir()
    # The native CLI owns serving; reuse RTD's port lease through generation.
    with reserve_port(root, tag="baseline_"+digest(str(out))[:20], port=port) as (port, _):
        generate, evaluate, env = bfcl_commands(root, merged, out, kang=kang, port=port)
        atomic_json(out/"commands.json", dict(generate=generate, evaluate=evaluate,
            protocol=protocol("bfcl"), kang=kang))
        start = time.monotonic()
        try:
            with (out/"generate.log").open("w") as log:
                subprocess.run(generate, cwd=root, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
        finally:
            atomic_json(out/"gpu_usage.json", dict(gpu_seconds=time.monotonic()-start,
                basis="one GPU reserved during official generation including server startup"))
    with (out/"evaluate.log").open("w") as log:
        subprocess.run(evaluate, cwd=root, env=dict(env, CUDA_VISIBLE_DEVICES=""),
                       check=True, stdout=log, stderr=subprocess.STDOUT)
    # official_expectations imports the harness in this process as well.
    os.environ["BFCL_PROJECT_ROOT"] = env["BFCL_PROJECT_ROOT"]
    return bfcl_metrics(root, out)


def run_alfworld(root, merged, out, *, kang, port):
    from ...adapters.alfworld import ALFWorldAdapter, DATA
    from ...run import serving_lane
    from ..benchmarks.alfworld_identity import official_expectations
    from ..benchmarks.webshop_evaluation import frozen_environment, validate_records
    out.mkdir(parents=True, exist_ok=False)
    expected = official_expectations(DATA)
    if len(expected["task_ids"]) != 140:
        raise ValueError("ALFWorld valid_seen must contain 140 tasks")
    adapter = ALFWorldAdapter(seed=0, port=port)
    if kang:
        from .paper_kang import install_alfworld_kang
        install_alfworld_kang(adapter, out/"kang_votes.jsonl")
    try:
        with frozen_environment("alfworld"):
            adapter.prepare_renderer(str(merged))
            start = time.monotonic()
            try:
                with serving_lane(adapter, str(merged), os.environ["CUDA_VISIBLE_DEVICES"], port, out/"vllm.log"):
                    metrics = adapter.evaluate(str(merged), out)
            finally:
                atomic_json(out/"gpu_usage.json", dict(gpu_seconds=time.monotonic()-start,
                    basis="one GPU reserved during vLLM adapter campaign including server startup"))
    finally:
        adapter.release_policy()
    validation = validate_records("alfworld", out, metrics, expected)
    return dict(complete=True, tasks=140, overall_accuracy_percent=100*metrics["success_rate"],
        per_category={k: dict(accuracy_percent=100*v) for k, v in metrics["per_category"].items()},
        validation=validation, expected=expected)


def evaluate_run(root, directory, manifest):
    from ..evaluation import _flatten_adapter
    from ..hardware import hardware_identity
    root, directory = Path(root), Path(directory)
    if tree_hash(directory/"checkpoint") != manifest["checkpoint_sha256"]:
        raise ValueError("trained checkpoint changed before evaluation")
    if tree_hash(manifest["model_path"]) != manifest["base_checkpoint_hash"]:
        raise ValueError("base checkpoint changed before evaluation")
    hardware = hardware_identity()
    if hardware["hard"] != manifest["hardware"]["hard"]:
        raise ValueError("evaluation hardware class differs from training")
    flat, merged = directory/"export/adapter", directory/"export/hub_merged"
    _flatten_adapter(manifest, directory/"checkpoint", flat)
    subprocess.run([sys.executable, str(root/"tools/bfcl_hub_merge_export.py"),
        "--adapter", str(flat), "--out", str(merged), "--model", manifest["model_path"], "--verify"],
        cwd=root, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), check=True)
    callback = run_alfworld if manifest["benchmark"] == "alfworld" else run_bfcl
    receipt = dict(protocol=protocol(manifest["benchmark"]), method=manifest["method"],
        teacher_tokens_charged=manifest["teacher_tokens_charged"], B=manifest["B"],
        checkpoint_sha256=manifest["checkpoint_sha256"], export_sha256=tree_hash(merged))
    receipt["hardware"] = hardware
    result = callback(root, merged, directory/"official", kang=False, port=manifest["port"])
    result.update(receipt, evaluation_mode="official_single_sample")
    atomic_json(directory/"official_metrics.json", result)
    if manifest["method"] == "kang":
        sag = callback(root, merged, directory/"kang_sag", kang=True, port=manifest["port"])
        sag.update(receipt, evaluation_mode="kang_sag", n=3, sampling_temperature=.7,
                   official_single_sample=result)
        # The method-specific score is separate and never labeled greedy.
        atomic_json(directory/"kang_metrics.json", sag)
        result["kang_self_consistency"] = sag
    atomic_json(directory/"metrics.json", result)
    return result
