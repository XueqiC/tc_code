#!/usr/bin/env python3
"""Make an existing BFCL training pool teacher-consistent (unified-CRCD brief §7, 2026-09-04).

The r1–r4 pools were mined with the benchmark ground truth as y^T for official
tasks (teacher == "oracle_gt") and with 3–4 calibration ids leaked in. This tool
rewrites a pool without re-mining (student continuations y^S and their outcomes
are unchanged):
  * rows on calibration ids -> dropped (held out for certification);
  * rows on official demand ids -> y^T replaced by the verified deepseek demo
    (teacher = deepseek-v4-pro-FC); demand ids without a verified demo -> dropped;
  * rows on generated ids -> kept, teacher relabelled teacher_authored_gt/_abstain;
  * anchors / self rows untouched.
Usage: bfcl_pool_teacherize.py --in POOL --out POOL_T [--report JSON]
"""
import argparse, json, collections, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from bfcl_event_mine_single import load_teacher_demos, demo_call_strings, target_block, IRRELEVANCE  # noqa: E402

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--demos", default="envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC")
    ap.add_argument("--teacher-name", default="deepseek-v4-pro-FC")
    a = ap.parse_args()
    split = json.load(open(ROOT / "configs/bfcl_support_split.json"))
    demand, calib = set(split["demand"]), set(split["calibration"])
    demos = load_teacher_demos(ROOT / a.demos)
    gen_ids = set()
    for f in (ROOT / "data/bfcl_sft").glob("gen*.jsonl"):
        for l in open(f):
            if l.strip():
                gen_ids.add(json.loads(l).get("id"))
    stats = collections.Counter(); out = open(ROOT / a.out, "w"); changed = []
    for l in open(ROOT / a.inp):
        if not l.strip():
            continue
        r = json.loads(l); tid = r.get("task_id", ""); teacher = r.get("teacher")
        base_id = tid.split("#")[0]
        if base_id in calib:
            stats["dropped_calibration"] += 1; continue
        if teacher == "oracle_gt" and int(r.get("turn_index", 0)) == 0:
            if base_id in gen_ids or base_id.startswith(("gen", "oos_")):
                cat = r.get("_seed_category", "")
                r["teacher"] = "teacher_authored_abstain" if (cat in IRRELEVANCE or cat.startswith("oos_")) else "teacher_authored_gt"
                stats["relabelled_generated"] += 1
            elif base_id in demand:
                if base_id not in demos:
                    stats["dropped_demand_no_demo"] += 1; continue
                res = demos[base_id]; cat = base_id.rsplit("_", 1)[0]
                new = res.strip() if isinstance(res, str) else target_block(demo_call_strings(res))
                if not new:
                    stats["dropped_demand_empty_demo"] += 1; continue
                changed.append({"task_id": tid, "old": r["response"][:160], "new": new[:160]})
                r["response"] = new; r["teacher"] = a.teacher_name; r["token_hint"] = max(len(new) // 4, 1)
                stats["demand_teacherized"] += 1
            else:
                stats["dropped_official_unknown"] += 1; continue
        elif teacher == "oracle_gt":
            # stateful rows: all on generated (teacher-authored) tasks
            r["teacher"] = "teacher_authored_gt"; stats["relabelled_stateful"] += 1
        else:
            stats[f"kept_{teacher}"] += 1
        out.write(json.dumps(r, ensure_ascii=False) + "\n"); stats["written"] += 1
    out.close()
    rep = {"in": a.inp, "out": a.out, "stats": dict(stats), "demand_ids_with_demo": len(demos), "examples": changed[:6]}
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    if a.report:
        (ROOT / a.report).write_text(json.dumps(rep, indent=1, ensure_ascii=False))

if __name__ == "__main__":
    main()
