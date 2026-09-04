#!/usr/bin/env python3
"""Paired episode analysis for ALFWorld arms (user 2026-09-03): for two arms X and Y on the same
134 valid_unseen games report solved-by-X-only / Y-only / both / neither, the exact McNemar
(binomial) p-value on the discordant pairs, and a paired bootstrap CI on the success-rate difference.

    python tools/alf_paired.py results/alf_records/abl_D_weighted_s0.jsonl results/alf_records/abl_C_conseq_s0.jsonl
"""
import json, random, sys
from math import comb

def load(p):
    d = {}
    for l in open(p):
        if l.strip():
            r = json.loads(l); d[r["task_id"]] = bool(r["won"])
    return d

def main():
    x, y = load(sys.argv[1]), load(sys.argv[2])
    keys = sorted(set(x) & set(y))
    xo = sum(1 for k in keys if x[k] and not y[k]); yo = sum(1 for k in keys if y[k] and not x[k])
    both = sum(1 for k in keys if x[k] and y[k]); neither = len(keys) - xo - yo - both
    n = xo + yo
    p = sum(comb(n, i) for i in range(0, min(xo, yo) + 1)) * 2 / (2 ** n) if n else 1.0
    p = min(1.0, p)
    rng = random.Random(0); diffs = []
    for _ in range(5000):
        s = [keys[rng.randrange(len(keys))] for _ in keys]
        diffs.append((sum(x[k] for k in s) - sum(y[k] for k in s)) / len(s))
    diffs.sort()
    print(f"n={len(keys)}  X-only={xo}  Y-only={yo}  both={both}  neither={neither}  "
          f"X={sum(x[k] for k in keys)}  Y={sum(y[k] for k in keys)}  "
          f"McNemar exact p={p:.3f}  bootstrap 95% CI of (X-Y) = [{diffs[125]*100:+.1f}, {diffs[4875]*100:+.1f}] pp")

if __name__ == "__main__":
    main()
