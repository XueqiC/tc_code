#!/usr/bin/env python
"""AppWorld P0 audit (CRCD spec 2026-09-04, §7.1-7.2, §11 P0 items 1 and 3).

Sub-commands
------------
split   Split audit from train/dev metadata ONLY (never touches test tasks):
        scenario families x variants per split, required-app combinations,
        difficulty; writes results/analysis/appworld_audit_split.json and prints
        the markdown table used by docs/2026-09-04-appworld-split-audit.md.
smoke   Deterministic base-agent smoke test: runs the official scaffold through
        AppWorldOfficialAdapter._run_official (the exact code path used by
        rollout()/evaluate()) on N train tasks, R times at temperature 0 against
        an already-running vLLM server, converts every task into the §7.1
        interaction record, checks run-to-run identity, runs `appworld evaluate`
        offline on the kept outputs, and writes
        results/analysis/appworld_audit_smoke.json.

Only train/dev dataset files and train task directories are read. Nothing under
data/tasks/<test task> is opened, and no ground-truth solution/evaluation content
is copied anywhere: the only ground-truth files read are required_apps.json and
metadata.json (difficulty/num_apps/num_apis/num_api_calls), both of which §7.2
explicitly allows as train/dev-only grouping labels that are never shown to the
model.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bfas.adapters import appworld_official as aw  # noqa: E402

DATA = aw.DATA
ANALYSIS = ROOT / "results/analysis"
FULL_CODE_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)
PARTIAL_CODE_RE = re.compile(r".*```python\n(.*)", re.DOTALL)


# --------------------------------------------------------------------------- split
def _ids(split: str) -> list[str]:
    return [l.strip() for l in (DATA / "datasets" / f"{split}.txt").read_text().splitlines() if l.strip()]


def _scenario(task_id: str) -> str:
    return task_id.rsplit("_", 1)[0]


def train_dev_metadata() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for split in ("train", "dev"):
        for task_id in _ids(split):
            gt = DATA / "tasks" / task_id / "ground_truth"
            specs = json.loads((DATA / "tasks" / task_id / "specs.json").read_text())
            apps = sorted(json.loads((gt / "required_apps.json").read_text()))
            md = json.loads((gt / "metadata.json").read_text())
            rows[task_id] = {
                "split": split,
                "scenario": _scenario(task_id),
                "variant": int(task_id.rsplit("_", 1)[1]),
                "required_apps": apps,
                "difficulty": md["difficulty"],
                "num_apps": md["num_apps"],
                "num_apis": md["num_apis"],
                "num_api_calls": md["num_api_calls"],
                "instruction_chars": len(specs["instruction"]),
            }
    return rows


def split_audit(write: bool = True) -> dict[str, Any]:
    rows = train_dev_metadata()
    counts = {s: sum(1 for r in rows.values() if r["split"] == s) for s in ("train", "dev")}
    # test counts come from the dataset id lists only (no task directory is opened)
    counts["test_normal"] = len(_ids("test_normal"))
    counts["test_challenge"] = len(_ids("test_challenge"))
    scen: dict[str, dict[str, Any]] = {}
    for tid, r in rows.items():
        s = scen.setdefault(r["scenario"], {"train": [], "dev": [], "required_apps": r["required_apps"],
                                            "difficulty": set(), "num_api_calls": []})
        s[r["split"]].append(tid)
        s["difficulty"].add(r["difficulty"])
        s["num_api_calls"].append(r["num_api_calls"])
    for s in scen.values():
        s["difficulty"] = sorted(s["difficulty"])
        s["train"].sort()
        s["dev"].sort()
    spanning = [k for k, s in scen.items() if s["train"] and s["dev"]]
    combos: dict[str, dict[str, int]] = collections.defaultdict(lambda: {"train": 0, "dev": 0, "scenarios_train": 0, "scenarios_dev": 0})
    for k, s in scen.items():
        key = "+".join(s["required_apps"])
        combos[key]["train"] += len(s["train"])
        combos[key]["dev"] += len(s["dev"])
        combos[key]["scenarios_train"] += bool(s["train"])
        combos[key]["scenarios_dev"] += bool(s["dev"])
    per_app: dict[str, dict[str, int]] = collections.defaultdict(lambda: {"train": 0, "dev": 0})
    for r in rows.values():
        for a in r["required_apps"]:
            per_app[a][r["split"]] += 1
    difficulty = {s: dict(collections.Counter(r["difficulty"] for r in rows.values() if r["split"] == s)) for s in ("train", "dev")}
    out = {
        "data_version": (DATA / "version.txt").read_text().strip(),
        "task_counts": counts,
        "scenarios": {"train": sum(1 for s in scen.values() if s["train"]), "dev": sum(1 for s in scen.values() if s["dev"]),
                      "spanning_train_and_dev": len(spanning), "spanning_ids": spanning},
        "variants_per_scenario": {s: dict(collections.Counter(len(v[s]) for v in scen.values() if v[s])) for s in ("train", "dev")},
        "variants_differ_in_required_apps": sum(1 for k in scen if len({tuple(rows[t]["required_apps"]) for t in scen[k]["train"] + scen[k]["dev"]}) > 1),
        "app_combos": dict(sorted(combos.items(), key=lambda kv: -(kv[1]["train"] + kv[1]["dev"]))),
        "per_app": dict(sorted(per_app.items())),
        "difficulty": difficulty,
        "per_scenario": scen,
        "tasks": rows,
    }
    if write:
        ANALYSIS.mkdir(parents=True, exist_ok=True)
        (ANALYSIS / "appworld_audit_split.json").write_text(json.dumps(out, indent=1))
    return out


def split_markdown(a: dict[str, Any]) -> str:
    lines = ["| required-app combination | train tasks | train scenarios | dev tasks | dev scenarios | spans train+dev |",
             "|---|---|---|---|---|---|"]
    for k, c in a["app_combos"].items():
        lines.append(f"| {k} | {c['train']} | {c['scenarios_train']} | {c['dev']} | {c['scenarios_dev']} | {'yes' if c['train'] and c['dev'] else 'no'} |")
    lines += ["", "| scenario | split | variants | required apps | difficulty | gt api calls (min-max) |", "|---|---|---|---|---|---|"]
    for k, s in sorted(a["per_scenario"].items(), key=lambda kv: (kv[1]["dev"] != [], kv[0])):
        split = "train" if s["train"] else "dev"
        n = len(s["train"]) or len(s["dev"])
        lines.append(f"| {k} | {split} | {n} | {'+'.join(s['required_apps'])} | {','.join(map(str, s['difficulty']))} | {min(s['num_api_calls'])}-{max(s['num_api_calls'])} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- records
def extract_code(text: str) -> str:
    """Same rule as SimplifiedReActCodeAgent with ignore_multiple_calls=True."""
    m = FULL_CODE_RE.search(text)
    if m:
        return m.group(1).strip()
    m = PARTIAL_CODE_RE.match(text)
    return m.group(1).strip() if m else ""


def parse_environment_io(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    out = []
    for block in re.split(r"\n### Environment Interaction \d+\n-+\n", text)[1:]:
        m = re.match(r"```python\n(.*?)\n```\n\n```\n(.*?)```\n", block, re.DOTALL)
        if m:
            out.append({"input": m.group(1), "output": m.group(2)})
    return out


def task_record(outputs_dir: Path, task_id: str, split: str, evaluation: dict[str, Any] | None) -> dict[str, Any]:
    """The §7.1 per-interaction record assembled from what the official run writes."""
    tdir = outputs_dir / "tasks" / task_id
    calls = aw.read_lm_calls(outputs_dir, task_id)
    env_io = parse_environment_io(tdir / "logs/environment_io.md")
    api_rows = []
    p = tdir / "logs/api_calls.jsonl"
    if p.is_file():
        api_rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    urls = [str(r.get("url", "")) for r in api_rows]
    specs = json.loads((DATA / "tasks" / task_id / "specs.json").read_text())
    interactions = []
    for i, call in enumerate(calls):
        text = aw.call_output_text(call)
        usage = (call.get("output") or {}).get("usage") or {}
        interactions.append({
            "step": i + 1,
            "prompt_messages": aw.call_messages(call),           # full state s_i
            "response_text": text,
            "code_block": extract_code(text),                     # y_i as executed
            "execution_output": env_io[i]["output"] if i < len(env_io) else None,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "request_seed": (call.get("input") or {}).get("seed"),
            "request_temperature": (call.get("input") or {}).get("temperature"),
        })
    usage_file = tdir / "misc/usage.json"
    usage_total = json.loads(usage_file.read_text())["tokens"] if usage_file.is_file() else None
    ev = (evaluation or {}).get("individual", {}).get(task_id)
    return {
        "task_id": task_id,
        "split": split,
        "scenario": _scenario(task_id),
        "instruction": specs["instruction"],
        "supervisor": specs["supervisor"],
        "reconstruction": {  # deterministic replay id: task + env seed + executed code sequence
            "task_id": task_id, "random_seed": None, "code_sequence_len": len(env_io),
            "final_dbs_dir": str(tdir / "dbs"),
        },
        "interactions": interactions,
        "n_model_calls": len(calls),
        "n_env_interactions": len(env_io),
        "n_api_calls": len(api_rows),
        "api_call_urls": urls,
        # the supervisor.complete_task API posts to /supervisor/message, so detect it from the
        # executed code (environment_io) and accept any complete_task URL as a fallback
        "complete_task_called": any(re.search(r"apis\.supervisor\.complete_task\s*\(", e["input"]) for e in env_io)
        or any("complete_task" in u for u in urls),
        "tokens": {"prompt": sum(i["prompt_tokens"] or 0 for i in interactions),
                   "completion": sum(i["completion_tokens"] or 0 for i in interactions),
                   "usage_json": usage_total},
        "official_eval": ev,
        "success": bool(ev and ev.get("success") is True),
        "errors": [i["execution_output"][:300] for i in interactions
                   if i["execution_output"] and ("Traceback" in i["execution_output"] or "Error" in i["execution_output"])],
    }


def determinism_key(rec: dict[str, Any]) -> list[Any]:
    return [(i["prompt_messages"], i["response_text"], i["execution_output"]) for i in rec["interactions"]] + \
        [rec["api_call_urls"], rec["success"]]


# --------------------------------------------------------------------------- smoke
def pick_train_tasks(n: int) -> list[str]:
    """First variant of the first n scenarios in sorted train order (distinct families)."""
    seen: list[str] = []
    out: list[str] = []
    for tid in sorted(_ids("train")):
        if _scenario(tid) not in seen:
            seen.append(_scenario(tid))
            out.append(tid)
        if len(out) == n:
            break
    return out


def smoke(args: argparse.Namespace) -> int:
    os.environ.setdefault("BFAS_APPWORLD_PROCESSES", str(args.processes))
    os.environ["BFAS_APPWORLD_KEEP_RUNS"] = "1"
    task_ids = args.tasks.split(",") if args.tasks else pick_train_tasks(args.n_tasks)
    print(f"[smoke] tasks={task_ids} repeats={args.repeats} port={args.port} policy={args.policy}", flush=True)
    runs: list[dict[str, Any]] = []
    kept: list[aw.OfficialRun] = []
    for rep in range(args.repeats):
        # fresh adapter per repeat -> identical config seed (seed + run_serial) and env random_seed
        adapter = aw.AppWorldOfficialAdapter(seed=args.seed, port=args.port)
        adapter.prepare_renderer(args.policy)
        adapter.serving_probe()
        model, base_url, api_key = adapter._student_endpoint()
        t0 = time.time()
        run = adapter._run_official(task_ids=task_ids, model_name=model, base_url=base_url, api_key=api_key,
                                    temperature=0.0, label=f"smoke{rep + 1}")
        wall = time.time() - t0
        kept.append(run)
        rollouts = adapter._rollouts_from_run(task_ids, run, guided=False)
        records = {tid: task_record(run.outputs_dir, tid, "train", run.evaluation) for tid in task_ids}
        for tid in task_ids:
            records[tid]["reconstruction"]["random_seed"] = args.seed + adapter._run_serial
            records[tid]["adapter_turns"] = len(next(r for r in rollouts if r.task_id == tid).turns)
        # explicit OFFLINE evaluator call on the kept outputs (the run already evaluated once)
        ev_cmd = [str(aw.OFFICIAL_BIN), "evaluate", run.experiment_name, run.dataset_name, "--root", str(aw.REPO)]
        ev = subprocess.run(ev_cmd, cwd=aw.REPO, capture_output=True, text=True)
        ev_json = json.loads((run.outputs_dir / "evaluations" / f"{run.dataset_name}.json").read_text())
        runs.append({
            "repeat": rep + 1, "experiment_name": run.experiment_name, "outputs_dir": str(run.outputs_dir),
            "wall_seconds": round(wall, 1), "records": records,
            "offline_evaluate_rc": ev.returncode, "offline_evaluate_tail": ev.stdout[-600:] + ev.stderr[-300:],
            "aggregate": ev_json.get("aggregate"),
            "individual_keys": sorted(next(iter(ev_json.get("individual", {}).values()), {}).keys()),
        })
        print(f"[smoke] repeat {rep + 1}: {wall:.0f}s aggregate={ev_json.get('aggregate')} "
              f"offline-evaluate rc={ev.returncode}", flush=True)
        for tid in task_ids:
            r = records[tid]
            print(f"   {tid}: calls={r['n_model_calls']} env_io={r['n_env_interactions']} api={r['n_api_calls']} "
                  f"complete_task={r['complete_task_called']} tokens={r['tokens']['prompt']}+{r['tokens']['completion']} "
                  f"success={r['success']} errors={len(r['errors'])}", flush=True)
    determinism = {}
    for tid in task_ids:
        keys = [determinism_key(r["records"][tid]) for r in runs]
        same = all(k == keys[0] for k in keys)
        first_div = None
        if not same:
            a, b = runs[0]["records"][tid]["interactions"], runs[1]["records"][tid]["interactions"]
            for i, (x, y) in enumerate(zip(a, b)):
                if x["prompt_messages"] != y["prompt_messages"] or x["response_text"] != y["response_text"]:
                    first_div = {"step": i + 1, "prompt_same": x["prompt_messages"] == y["prompt_messages"],
                                 "response_a": x["response_text"][:400], "response_b": y["response_text"][:400]}
                    break
            if first_div is None and len(a) != len(b):
                first_div = {"step": min(len(a), len(b)) + 1, "reason": f"length {len(a)} vs {len(b)}"}
        determinism[tid] = {"identical": same, "first_divergence": first_div,
                            "calls": [r["records"][tid]["n_model_calls"] for r in runs],
                            "success": [r["records"][tid]["success"] for r in runs]}
    out = {
        "policy": args.policy, "port": args.port, "temperature": 0.0, "seed": args.seed,
        "processes": int(os.environ["BFAS_APPWORLD_PROCESSES"]), "tasks": task_ids,
        "vllm_version": args.vllm_version, "determinism": determinism,
        "all_identical": all(d["identical"] for d in determinism.values()), "runs": runs,
    }
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    (ANALYSIS / "appworld_audit_smoke.json").write_text(json.dumps(out, indent=1))
    print(f"[smoke] determinism: {json.dumps({k: v['identical'] for k, v in determinism.items()})} "
          f"all_identical={out['all_identical']}", flush=True)
    for run in kept:  # config/prompt/dataset scratch files go; outputs stay if --keep
        run.cleanup(keep_outputs=args.keep)
    return 0 if out["all_identical"] else 3


def rebuild_records(args: argparse.Namespace) -> int:
    """Rebuild the smoke json's records/determinism from kept official outputs dirs."""
    path = ANALYSIS / "appworld_audit_smoke.json"
    out = json.loads(path.read_text())
    for run in out["runs"]:
        odir = Path(run["outputs_dir"])
        ev_files = list((odir / "evaluations").glob("*.json"))
        ev = json.loads(ev_files[0].read_text()) if ev_files else None
        for tid, old in run["records"].items():
            rec = task_record(odir, tid, "train", ev)
            rec["reconstruction"]["random_seed"] = old["reconstruction"]["random_seed"]
            rec["adapter_turns"] = old.get("adapter_turns")
            run["records"][tid] = rec
    for tid in out["tasks"]:
        keys = [determinism_key(r["records"][tid]) for r in out["runs"]]
        out["determinism"][tid]["identical"] = all(k == keys[0] for k in keys)
        out["determinism"][tid]["success"] = [r["records"][tid]["success"] for r in out["runs"]]
        out["determinism"][tid]["complete_task"] = [r["records"][tid]["complete_task_called"] for r in out["runs"]]
    out["all_identical"] = all(d["identical"] for d in out["determinism"].values())
    path.write_text(json.dumps(out, indent=1))
    for tid, d in out["determinism"].items():
        print(tid, d)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("split")
    sub.add_parser("records", help="rebuild results/analysis/appworld_audit_smoke.json records from kept outputs")
    s = sub.add_parser("smoke")
    s.add_argument("--policy", default="Qwen/Qwen3.5-4B")
    s.add_argument("--port", type=int, default=8988)
    s.add_argument("--n-tasks", type=int, default=3)
    s.add_argument("--tasks", default="", help="comma-separated train task ids (overrides --n-tasks)")
    s.add_argument("--repeats", type=int, default=2)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--processes", type=int, default=1, help="official-runner processes (1 = no cross-task batching)")
    s.add_argument("--keep", action="store_true", help="keep experiments/outputs of the smoke runs")
    s.add_argument("--vllm-version", default="")
    args = ap.parse_args()
    if args.cmd == "records":
        return rebuild_records(args)
    if args.cmd == "split":
        a = split_audit()
        print(json.dumps({k: a[k] for k in ("data_version", "task_counts", "scenarios", "variants_per_scenario",
                                             "variants_differ_in_required_apps", "difficulty", "per_app")}, indent=1))
        print(split_markdown(a))
        return 0
    return smoke(args)


if __name__ == "__main__":
    raise SystemExit(main())
