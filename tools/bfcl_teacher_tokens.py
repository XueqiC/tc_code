#!/usr/bin/env python3
"""Reconstruct the BFCL teacher-output-token account (unified-CRCD brief §7/§10).

Two teacher expenditures exist for BFCL, both paid to deepseek-v4-pro(-FC):
  (1) demonstrations on the official demand split — exact counts survive in the
      official BFCL result files (output_token_count per attempt, verified or not);
  (2) synthesised task pools (gen_pool*, gen_oos, gen_stateful; the source of the
      r1–r4 training events) — the generator discarded the API usage field, so the
      cost is ESTIMATED by re-tokenising the teacher-authored fields of every stored
      row (question + ground_truth [+ scenario/initial_config for stateful]) with the
      Qwen3.5 tokenizer, then inflated by PROSE_FACTOR for fences/prose that were
      parsed away. Rejected drafts and repair calls were not stored: lower bound.
Every training-event file is mapped back to its generated rows so each round has a
teacher-token figure. Output: results/analysis/bfcl_teacher_tokens.json
"""
import glob, json, collections, statistics, sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
BFCL = PROJ / "envs/bfcl/gorilla/berkeley-function-call-leaderboard"
PROSE_FACTOR = 1.3
GEN_FIELDS = ("question", "ground_truth", "scenario", "initial_config", "why")

def flat(x):
    return sum(flat(i) for i in x) if isinstance(x, list) else (x or 0)

def demos():
    out = {"attempts": {}, "per_task": collections.defaultdict(int)}
    for d in sorted(BFCL.glob("result_demos_deepseek_v4_pro_FC_a*")):
        rows = tok = 0
        for f in d.rglob("*_result.json"):
            for l in open(f):
                if l.strip():
                    e = json.loads(l); t = flat(e.get("output_token_count", 0))
                    rows += 1; tok += t; out["per_task"][e["id"]] += t
        out["attempts"][d.name[-2:]] = {"rows": rows, "output_tokens": tok}
    verified = json.load(open(PROJ / "data/bfcl_demos_ds_verified.json"))
    out["verified_tasks"] = len(verified)
    out["demand_tasks"] = len(json.load(open(PROJ / "configs/bfcl_support_split.json"))["demand"])
    out["total_output_tokens"] = sum(a["output_tokens"] for a in out["attempts"].values())
    out["per_task"] = dict(out["per_task"])
    return out

def generation(tokenizer):
    per_id = collections.defaultdict(list); files = {}
    for f in sorted((PROJ / "data/bfcl_sft").glob("gen*.jsonl")):
        n = tok = 0
        for l in open(f):
            if not l.strip():
                continue
            r = json.loads(l)
            text = "\n".join(json.dumps(r[k], ensure_ascii=False) for k in GEN_FIELDS if k in r)
            t = len(tokenizer(text)["input_ids"]); n += 1; tok += t
            per_id[r.get("id")].append(t)
        files[f.name] = {"rows": n, "content_tokens": tok, "estimate": int(tok * PROSE_FACTOR)}
    total = sum(v["content_tokens"] for v in files.values())
    return files, {"content_tokens": total, "estimate": int(total * PROSE_FACTOR)}, {k: statistics.mean(v) for k, v in per_id.items()}

def rounds(per_id_tok, demo_per_task):
    out = {}
    for f in sorted((PROJ / "data/bfcl_sft").glob("events*.jsonl")):
        ids = set(); missing = 0; gen_tok = 0.0; demo_tok = 0; official = 0
        for l in open(f):
            if l.strip():
                ids.add(json.loads(l).get("task_id"))
        for i in ids:
            if i in per_id_tok:
                gen_tok += per_id_tok[i]
            elif i in demo_per_task:
                demo_tok += demo_per_task[i]; official += 1
            else:
                missing += 1; official += 1
        out[f.name] = {"unique_task_ids": len(ids), "generated_ids": len(ids) - official,
                       "official_ids": official, "ids_without_token_record": missing,
                       "gen_tokens_estimate": int(gen_tok * PROSE_FACTOR), "demo_tokens_exact_for_ids": demo_tok}
    return out

def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    d = demos(); files, gtotal, per_id = generation(tokenizer)
    res = {"teacher": "deepseek-v4-pro / deepseek-v4-pro-FC (ollama)", "prose_factor": PROSE_FACTOR,
           "demos_official_demand": {k: v for k, v in d.items() if k != "per_task"},
           "generation_pools": files, "generation_total": gtotal,
           "training_event_files": rounds(per_id, d["per_task"]),
           "grand_total_output_tokens": {"exact_demos": d["total_output_tokens"],
                                          "estimated_generation": gtotal["estimate"],
                                          "sum": d["total_output_tokens"] + gtotal["estimate"]},
           "caveats": ["generation usage was not logged by tools/bfcl_generate*.py before 2026-09-04; the estimate re-tokenises stored teacher-authored fields with the Qwen3.5 tokenizer (proxy for deepseek's) and cannot include rejected drafts or repair calls -> lower bound",
                        "demo counts are exact (official BFCL result files, all attempts incl. unverified); official BFCL ground truth was used only as the verifier V, never as a teacher continuation for official tasks in the reported pools except the 14 official-id single-turn events flagged below",
                        "the GT checker remains the executable verifier V; it is not a teacher"]}
    out = PROJ / "results/analysis/bfcl_teacher_tokens.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False)); print(json.dumps({k: res[k] for k in ("demos_official_demand", "generation_total", "grand_total_output_tokens")}, indent=1)); print(json.dumps(res["training_event_files"], indent=0))

if __name__ == "__main__":
    main()
