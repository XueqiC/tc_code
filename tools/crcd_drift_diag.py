#!/usr/bin/env python3
"""Corrective-overwrite diagnostics (user prompt §7): how far does a corrected model drift from
the base on states the base already handles correctly?

For a set of "mastered" prompts (the base's own correct replies = the anchor files), teacher-force
the base reply and measure, per model:
  * KL(pi_theta || pi_0) averaged over the base-reply tokens (token-level, full vocab)
  * mean next-token entropy of pi_theta on those positions
  * probability mass the model puts on starting a tool call at the first reply token
    (P(first token in {"<tool_call>", "<tool"}) vs abstaining)
  * log-prob of the base reply under pi_theta (behaviour retention)
  * expected reply length proxy: log-prob of EOS at the base reply's end position
Reported per prompt category and overall; compare CE vs pairwise arms trained on the same events.

    PYTHONPATH=src python tools/crcd_drift_diag.py --models base=Qwen/Qwen3.5-4B \
        ce=results/appworld_students/crcdr_ce_r2_s0/adapter pref=results/appworld_students/crcdr_pref_r2_s0/adapter \
        --prompts data/bfcl_sft/anchors_single_base.jsonl data/bfcl_sft/anchors_oos_base.jsonl \
        --out results/analysis/crcd_drift_v1.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def load(path: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device).eval()
    return tok, model


@torch.no_grad()
def logits_on(model, tok, prompt: str, reply: str, device: str, max_len: int):
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    r_ids = tok(reply, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    if len(p_ids) + len(r_ids) > max_len:
        p_ids = p_ids[-(max_len - len(r_ids)):]
    ids = torch.tensor([p_ids + r_ids], device=device)
    out = model(ids).logits[0]
    start = len(p_ids) - 1  # logits predicting reply tokens
    return out[start:start + len(r_ids)].float(), torch.tensor(r_ids, device=device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True, help="name=path (first must be base)")
    parser.add_argument("--prompts", nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument("--out", default="results/analysis/crcd_drift_v1.json")
    args = parser.parse_args()
    device = "cuda"
    rows = []
    for f in args.prompts:
        for l in (ROOT / f).read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                rows.append({"prompt": r["prompt"], "reply": r["response"], "cat": r.get("_seed_category", "?")})
    if args.limit:
        rows = rows[: args.limit]
    print(f"[drift] {len(rows)} mastered prompts", flush=True)
    specs = [m.split("=", 1) for m in args.models]
    base_name, base_path = specs[0]
    tok, base = load(base_path, device)
    base_logp = []
    tool_ids = {tok(t, add_special_tokens=False)["input_ids"][0] for t in ("<tool_call>", "<tool", " <tool_call>")}
    # cache base log-softmax per row
    for r in rows:
        lg, ids = logits_on(base, tok, r["prompt"], r["reply"], device, args.max_len)
        r["_base_logp"] = F.log_softmax(lg, dim=-1).cpu()
        r["_ids"] = ids.cpu()
    del base
    torch.cuda.empty_cache()
    report = {"n": len(rows), "models": {}}
    for name, path in specs:
        _, model = load(path, device)
        agg = defaultdict(lambda: defaultdict(float))
        cnt = defaultdict(int)
        for r in rows:
            lg, ids = logits_on(model, tok, r["prompt"], r["reply"], device, args.max_len)
            logp = F.log_softmax(lg, dim=-1)
            b = r["_base_logp"].to(device)
            kl = (logp.exp() * (logp - b)).sum(-1).mean().item()          # KL(theta || base) per token
            ent = -(logp.exp() * logp).sum(-1).mean().item()
            reply_lp = logp.gather(1, ids[:, None]).sum().item()
            first = logp[0].exp()
            p_tool = float(sum(first[i].item() for i in tool_ids))
            for key, val in (("kl", kl), ("entropy", ent), ("reply_logp", reply_lp), ("p_tool_first", p_tool)):
                agg["ALL"][key] += val
                agg[r["cat"]][key] += val
            cnt["ALL"] += 1
            cnt[r["cat"]] += 1
        summary = {c: {k: round(v / cnt[c], 4) for k, v in d.items()} for c, d in agg.items()}
        summary = {c: dict(s, n=cnt[c]) for c, s in summary.items()}
        report["models"][name] = summary
        print(f"[drift] {name}: {summary['ALL']}", flush=True)
        del model
        torch.cuda.empty_cache()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"[drift] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
