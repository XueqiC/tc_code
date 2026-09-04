#!/usr/bin/env python3
"""CRCD P0-1 (single-turn half): mine intervention events on single-turn BFCL tasks.

For a single-turn task the whole response is the decision, so the event is
(h = rendered task prompt, a^- = a wrong student reply, a^+ = the ground-truth
call block) and the causal utility is the student's own pass rate:

    u^- = P(student reply passes the AST checker)   (K samples, Laplace)
    u^+ = 1 (the ground truth passes by construction)
    dU  = 1 - u^-

Tasks with u^- close to 1 produce no event (nothing to correct); tasks where
the student never passes are frontier events (dU ~ 1). Sources: the support
single-turn tasks (official entries) and the verified generated single-turn
tasks (gen_pool_v3 / gen_oos style rows). Irrelevance categories are skipped
here (the correction is an abstention, not a call; handled by the pool's
abstain rows).

Usage:
    PYTHONPATH=src:tools envs/bfcl/.venv/bin/python tools/bfcl_event_mine_single.py \
        --port 8975 --k 4 --out data/bfcl_sft/events_single_v1.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from onpolicy_repair_collect import as_call_strings, student_reply, target_block  # noqa: E402

CALL_RE_JSON = None
IRRELEVANCE = {"irrelevance", "live_irrelevance"}


def language_for(category: str):
    from bfcl_eval.constants.enums import Language

    if "java" in category and "javascript" not in category:
        return Language.JAVA
    if "javascript" in category:
        return Language.JAVASCRIPT
    return Language.PYTHON


def parsed_calls(text: str) -> list[dict]:
    """<tool_call> JSON blocks -> [{name: arguments}] for the AST checker."""
    import re

    out = []
    for blob in re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S):
        try:
            call = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(call, dict) and call.get("name"):
            out.append({call["name"]: call.get("arguments") or {}})
    return out


def truth_from_entry(entry: dict) -> list[dict]:
    truth = entry.get("ground_truth")
    if truth is None:
        truth = entry.get("possible_answer")
    if isinstance(truth, str):
        try:
            truth = json.loads(truth)
        except json.JSONDecodeError:
            import ast

            try:
                truth = ast.literal_eval(truth)  # generated rows store a python repr
            except (ValueError, SyntaxError):
                return []
    if isinstance(truth, dict):
        truth = truth.get("ground_truth", truth)
    return truth if isinstance(truth, list) else []


def gt_call_strings(truth: list[dict]) -> list[str]:
    """Checker-format truth [{name: {arg: [choices]}}] -> executable call strings (first choice)."""
    calls = []
    for item in truth:
        for name, args in item.items():
            rendered = ", ".join(
                f"{k}={(v[0] if isinstance(v, list) and v else v)!r}" for k, v in (args or {}).items()
            )
            calls.append(f"{name}({rendered})")
    return calls


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8975)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--gen", nargs="*", default=["data/bfcl_sft/gen_pool_v3.jsonl", "data/bfcl_sft/gen_pool_v4.jsonl"])
    parser.add_argument("--no-support", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--irrelevance", action="store_true",
                        help="also mine irrelevance tasks: correct = no tool call; a+ = abstain text "
                             "(the v9 pool's abstain row for the task when present, else a plain refusal)")
    parser.add_argument("--abstain-pool", default="data/bfcl_sft/pool_bfcl_v9.jsonl")
    parser.add_argument("--only-ids", default=None,
                        help="JSON list file: restrict mining/measurement to these task ids")
    parser.add_argument("--rates-key", choices=("id", "prompt"), default="id",
                        help="key of the rates file: task id (NOT unique across generated tasks) or sha1(prompt)[:16]")
    parser.add_argument("--rates-out", default=None,
                        help="write {task_id: passes/k} for EVERY task seen (validity-gate measurement)")
    parser.add_argument("--anchors-out", default=None,
                        help="also write retention anchor pairs built from the student's OWN correct replies "
                             "(chosen = student's passing reply; rejected = its failing reply when it has one, "
                             "else a minimal flip: a fabricated call on abstain tasks / an abstention on call tasks)")
    parser.add_argument("--anchor-cap", type=int, default=40, help="max anchors per seed category")
    parser.add_argument("--out", default="data/bfcl_sft/events_single_v1.jsonl")
    args = parser.parse_args()

    from bfas.adapters.bfcl import BFCLAdapter
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
    from bfcl_eval.utils import populate_test_cases_with_predefined_functions
    from bfcl_generate import BFCL, CHECKER_MODEL

    adapter = BFCLAdapter()
    entries, _ = adapter._load_entries()
    # official answers live next to the harness, not in the task entries
    answers: dict[str, list] = {}
    for path in (BFCL / "bfcl_eval/data/possible_answer").glob("BFCL_v4_*.json"):
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    row = json.loads(line)
                    answers[row["id"]] = row["ground_truth"]
                except (json.JSONDecodeError, KeyError):
                    continue
    tasks: list[dict] = []
    if not args.no_support:
        split = json.loads((ROOT / "configs/bfcl_support_split.json").read_text())
        for task_id in list(split["demand"]) + list(split["calibration"]):
            entry = entries.get(task_id)
            if not entry:
                continue
            category = task_id.rsplit("_", 1)[0]
            if "multi_turn" in category or "memory" in category or "web_search" in category:
                continue
            if category in IRRELEVANCE and not args.irrelevance:
                continue
            populated = populate_test_cases_with_predefined_functions([json.loads(json.dumps(entry))])[0]
            tasks.append({
                "id": task_id, "category": category, "source": "support",
                "question": populated["question"], "function": populated.get("function") or [],
                "truth": truth_from_entry({"ground_truth": answers.get(task_id)}),
            })
    for path in args.gen:
        file = ROOT / path
        if not file.is_file():
            continue
        for line in file.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("reproduced") and not row.get("verified"):
                continue
            category = row.get("seed_category") or row.get("category") or ""
            if category in IRRELEVANCE and not args.irrelevance:
                continue
            oos = str(row.get("out_of_scope", "")).lower() == "true"
            tasks.append({
                "id": row["id"], "category": category, "source": file.name, "oos": oos,
                "question": row["question"], "function": row["function"],
                "truth": [] if oos else truth_from_entry(row),
            })
    if args.only_ids:
        keep = set(json.loads((ROOT / args.only_ids).read_text()))
        tasks = [t for t in tasks if t["id"] in keep]
    if args.limit:
        tasks = tasks[: args.limit]
    rates: dict[str, float] = {}
    abstain: dict[str, str] = {}
    if args.irrelevance and (ROOT / args.abstain_pool).is_file():
        for line in (ROOT / args.abstain_pool).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "<tool_call>" not in str(row.get("response", "")) and row.get("task_id"):
                abstain.setdefault(str(row["task_id"]), str(row["response"]).strip())
    print(f"[events1] {len(tasks)} single-turn tasks ({len(abstain)} abstain rows available)", flush=True)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    handle = out_path.open("a")
    stats = {"tasks": 0, "events": 0, "skipped_no_truth": 0, "all_pass": 0, "serve_err": 0}
    anchors = open(args.anchors_out, "w", encoding="utf-8") if args.anchors_out else None
    anchor_counts: dict[str, int] = {}
    t0 = time.time()
    for task in tasks:
        stats["tasks"] += 1
        irrelevant = task["category"] in IRRELEVANCE or task.get("oos", False)
        if not task["truth"] and not irrelevant:
            stats["skipped_no_truth"] += 1
            continue
        question = task["question"]
        turn = question[0] if question and isinstance(question[0], list) else question
        messages = [dict(m) for m in turn if isinstance(m, dict)]
        prompt = adapter._render(messages, task["function"])
        language = language_for(task["category"])
        passes = 0
        wrong_replies: list[str] = []
        right_replies: list[str] = []
        for _ in range(args.k):
            try:
                reply = student_reply(prompt, args.port, args.temperature)
            except Exception:
                stats["serve_err"] += 1
                continue
            calls = parsed_calls(reply)
            ok = False
            if irrelevant:
                ok = not calls  # abstaining is the correct behaviour
            elif calls:
                try:
                    verdict = ast_checker(task["function"], calls, task["truth"], language, task["category"], CHECKER_MODEL)
                    ok = bool(verdict.get("valid"))
                except Exception:
                    ok = False
            if ok:
                passes += 1
                right_replies.append(reply.strip())
            else:
                wrong_replies.append(reply.strip())
        u_minus = (passes + 1) / (args.k + 2)
        rate_key = task["id"] if args.rates_key == "id" else __import__("hashlib").sha1(prompt.encode("utf-8")).hexdigest()[:16]
        rates[rate_key] = passes / max(1, args.k)
        if anchors is not None and right_replies:
            cat = ("oos_" + task["category"]) if task.get("oos") else task["category"]
            if anchor_counts.get(cat, 0) < args.anchor_cap:
                if wrong_replies:
                    rejected = wrong_replies[0]
                elif irrelevant:
                    fn = (task["function"][0].get("name") if task.get("function") else None) or "unknown_function"
                    rejected = "<tool_call>\n" + json.dumps({"name": fn, "arguments": {}}) + "\n</tool_call>"
                else:
                    rejected = "I can't complete this request with the available functions."
                anchor_counts[cat] = anchor_counts.get(cat, 0) + 1
                stats["anchors"] = stats.get("anchors", 0) + 1
                anchors.write(json.dumps({
                    "task_id": task["id"], "teacher": "self_anchor", "turn_index": 0,
                    "prompt": prompt, "response": right_replies[0], "_rejected": rejected,
                    "token_hint": max(len(right_replies[0]) // 4, 1), "_task_phat": u_minus,
                    "_traj": f"{task['id']}#anc", "_seed_category": cat, "_source": task["source"],
                    "_anchor": True, "_event_turn": 0, "_event_u_plus": u_minus, "_event_u_minus": u_minus,
                    "_event_dU": 0.0, "_event_k": args.k, "_event_pass": passes,
                }, ensure_ascii=False) + "\n")
                anchors.flush()
        if not wrong_replies:
            stats["all_pass"] += 1
            continue
        if irrelevant:
            target = abstain.get(task["id"]) or (
                "I can't complete this request with the available functions."
            )
        else:
            target = target_block(gt_call_strings(task["truth"]))
        if not target:
            stats["skipped_no_truth"] += 1
            continue
        stats["events"] += 1
        handle.write(json.dumps({
            "task_id": task["id"],
            "teacher": "oracle_gt",
            "turn_index": 0,
            "prompt": prompt,
            "response": target,
            "_rejected": wrong_replies[0],
            "_rejected_all": wrong_replies,
            "token_hint": max(len(target) // 4, 1),
            "_task_phat": u_minus,
            "_traj": f"{task['id']}#ev1",
            "_seed_category": ("oos_" + task["category"]) if task.get("oos") else task["category"],
            "_source": task["source"],
            "_event_turn": 0,
            "_event_u_plus": 1.0,
            "_event_u_minus": u_minus,
            "_event_dU": 1.0 - u_minus,
            "_event_k": args.k,
            "_event_pass": passes,
        }, ensure_ascii=False) + "\n")
        handle.flush()
        if stats["tasks"] % 25 == 0:
            print(f"[events1] {stats} elapsed={time.time() - t0:.0f}s", flush=True)
    handle.close()
    if anchors is not None:
        anchors.close()
        print(f"[events1] anchors -> {args.anchors_out} ({stats.get('anchors', 0)} pairs)", flush=True)
    if args.rates_out:
        rp = ROOT / args.rates_out
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(rates, indent=0))
        print(f"[events1] rates for {len(rates)} tasks -> {rp}", flush=True)
    print(f"[events1] FINAL {stats} -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
