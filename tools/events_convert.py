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
    # round-3 union pool made teacher-consistent (15 official rows carry a deepseek
    # demo as y^T, generated rows relabelled teacher_authored_*); rows are a union
    # of round-2 (base student) and round-3 (r2 student) events, so the student is
    # resolved per row from the state hash; self_anchor rows are skipped.
    "bfcl_r3t": dict(path="data/bfcl_sft/pool_events_pref_v3t.jsonl", benchmark="bfcl",
                     student="by_hash", skip_teachers=("self_anchor",)),
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


def student_by_hash(out_dir: Path) -> dict[tuple[str, str], str]:
    """(state_hash, sha1(y^S)) -> student checkpoint from the already-converted
    bfcl_r2 / bfcl_r3 sets.  The pair is needed because the same state can be
    mined in both rounds with different student samples."""
    from bfas.events_schema import iter_events, state_hash_of
    table: dict[tuple[str, str], str] = {}
    for name in ("bfcl_r2", "bfcl_r3"):
        f = out_dir / f"{name}.jsonl"
        if f.exists():
            for ev in iter_events(f):
                table[(ev.state_hash, state_hash_of(ev.student_continuation))] = ev.student_checkpoint
    return table


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
        skip = set(spec.get("skip_teachers", ()))
        n_skip = sum(r.get("teacher") in skip for r in rows)
        rows = [r for r in rows if r.get("teacher") not in skip]
        conv = CONVERTERS[spec["benchmark"]]
        if spec["student"] == "by_hash":
            from bfas.events_schema import state_hash_of
            table = student_by_hash(out_dir)
            students = [table.get((state_hash_of(r["prompt"]), state_hash_of(r["_rejected"])),
                                  "unknown (union pool; (state, y^S) not in bfcl_r2/bfcl_r3)")
                        for r in rows]
        else:
            students = [spec["student"]] * len(rows)
        events = [conv(r, tok, source=spec["path"], split=args.split, student_checkpoint=st)
                  for r, st in zip(rows, students)]
        if n_skip:
            print(f"[convert] {name}: skipped {n_skip} rows with teacher in {sorted(skip)}")
        if spec["student"] == "by_hash":
            from collections import Counter
            print(f"[convert] {name}: student attribution {dict(Counter(students))}")
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
