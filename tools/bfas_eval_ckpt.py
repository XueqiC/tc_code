"""Evaluate one trained checkpoint with a benchmark's bfas adapter (official eval),
without re-running collection. Mirrors the tail of bfas.run.run_seed:

    serving_lane(adapter, checkpoint, gpu, port, log) -> adapter.evaluate(checkpoint, out_dir)

Usage:
    PYTHONPATH=src python tools/bfas_eval_ckpt.py --benchmark alfworld \
        --checkpoint results/appworld_students/alfb1_curr_s0/adapter \
        --out results/bfas/alfworld/curr_s0 --gpu <GPU-UUID or index> --port 8930 [--seed 0]

Written 2026-09-02 so mechanism arms trained on hpg (appworld_train.py on a synced
pool) can be scored with exactly the same evaluator as the rai bfas runs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bfas import run as bfas_run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=("appworld", "bfcl", "alfworld", "tau2"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("BFAS_PORT_BASE", "8930")))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter = bfas_run.make_adapter(args.benchmark, args.seed, args.port)
    checkpoint = Path(args.checkpoint)
    with bfas_run.serving_lane(adapter, checkpoint, args.gpu, args.port, out_dir / "vllm_eval.log"):
        metrics = adapter.evaluate(checkpoint, out_dir)
    (out_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"[bfas-eval] {args.benchmark} {checkpoint} headline={metrics.get('headline')} "
          f"per_category={metrics.get('per_category')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
