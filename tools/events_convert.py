#!/usr/bin/env python3
"""Convert the existing BFCL / ALFWorld miner outputs into the unified event
schema (src/bfas/events_schema.py) and write data/events_unified/<set>.jsonl.

Sets (default):
  bfcl_r2      data/bfcl_sft/pool_events_pref_v2.jsonl   student = base Qwen3.5-4B
  bfcl_r3      data/bfcl_sft/pool_events_pref_r3.jsonl   student = round-2 pref model
  alfworld_v1  data/alf_sft/events_v1.jsonl              student = base Qwen3.5-4B

Token counts use the Qwen/Qwen3.5-4B tokenizer.

    .venv/bin/python tools/events_convert.py [--sets bfcl_r2 alfworld_v1] [--out-dir data/events_unified]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bfas.events_schema import (  # noqa: E402
    BASE_STUDENT, CONVERTERS, load_raw_rows, write_events,
)

R2_STUDENT = "crcd_r2_b3pref_s0 (Qwen/Qwen3.5-4B + round-2 pref LoRA, official 47.34)"

SETS = {
    "bfcl_r2": dict(path="data/bfcl_sft/pool_events_pref_v2.jsonl", benchmark="bfcl",
                    student=BASE_STUDENT),
    "bfcl_r3": dict(path="data/bfcl_sft/pool_events_pref_r3.jsonl", benchmark="bfcl",
                    student=R2_STUDENT),
    "alfworld_v1": dict(path="data/alf_sft/events_v1.jsonl", benchmark="alfworld",
                        student=BASE_STUDENT),
    # extra sources the converters also understand
    "alfworld_v1_k8": dict(path="data/alf_sft/events_v1_k8.jsonl", benchmark="alfworld",
                           student=BASE_STUDENT),
    "bfcl_single_v1": dict(path="data/bfcl_sft/events_single_v1.jsonl", benchmark="bfcl",
                           student=BASE_STUDENT),
    "bfcl_stateful_v1": dict(path="data/bfcl_sft/events_v1.jsonl", benchmark="bfcl",
                             student=BASE_STUDENT),
}
DEFAULT_SETS = ("bfcl_r2", "bfcl_r3", "alfworld_v1")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", default=list(DEFAULT_SETS), choices=sorted(SETS))
    ap.add_argument("--out-dir", default="data/events_unified")
    ap.add_argument("--tokenizer", default=BASE_STUDENT)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    out_dir = ROOT / args.out_dir
    for name in args.sets:
        spec = SETS[name]
        rows = load_raw_rows(ROOT / spec["path"])
        conv = CONVERTERS[spec["benchmark"]]
        events = [conv(r, tok, source=spec["path"], split=args.split,
                       student_checkpoint=spec["student"]) for r in rows]
        hashes = {e.state_hash for e in events}
        out = out_dir / f"{name}.jsonl"
        n = write_events(out, events)
        s_tok = sum(e.student_output_tokens for e in events)
        t_tok = sum(e.teacher_output_tokens for e in events)
        print(f"[convert] {name}: {n} events ({len(hashes)} distinct states) "
              f"student_tokens={s_tok} teacher_tokens={t_tok} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
