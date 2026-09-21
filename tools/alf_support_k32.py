#!/usr/bin/env python3
"""Freeze the K=32 ALFWorld support from the loader manifest (2026-09-21 protocol).

K counts unique runnable task instances: multiple trajectories of one game never
increase K. Selection is deterministic (sha256 over a frozen salt and the game id)
and stratified across the six official task types by largest remainder, so the
support mirrors the train distribution instead of whichever types sort first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SALT = "tc-alignment/alfworld/support/2026-09-21"


def largest_remainder(counts: dict[str, int], k: int) -> dict[str, int]:
    total = sum(counts.values())
    exact = {t: k * n / total for t, n in counts.items()}
    floors = {t: int(v) for t, v in exact.items()}
    left = k - sum(floors.values())
    order = sorted(exact, key=lambda t: (-(exact[t] - floors[t]), t))
    for t in order[:left]:
        floors[t] += 1
    return floors


def rank(game_id: str) -> str:
    return hashlib.sha256(f"{SALT}|{game_id}".encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("-k", type=int, default=32)
    args = ap.parse_args()

    report = json.loads(Path(args.manifest).read_text())
    train = report["splits"]["train"]
    data_root = Path(report["data_root"]) / "train"

    by_type: dict[str, list[str]] = {}
    for gid in train["game_ids"]:
        traj = json.loads((data_root / gid / "traj_data.json").read_text())
        by_type.setdefault(traj["task_type"], []).append(gid)

    quota = largest_remainder({t: len(v) for t, v in by_type.items()}, args.k)
    support: list[dict[str, object]] = []
    for task_type in sorted(by_type):
        chosen = sorted(by_type[task_type], key=rank)[: quota[task_type]]
        for gid in chosen:
            support.append({"game_id": gid, "task_type": task_type, "rank": rank(gid)[:16]})
    support.sort(key=lambda r: (r["task_type"], r["game_id"]))

    ids = [r["game_id"] for r in support]
    assert len(ids) == len(set(ids)) == args.k, "support must hold K unique games"

    out = {
        "protocol": "fixed-K few-shot support, ALFWorld, 2026-09-21",
        "k": args.k,
        "salt": SALT,
        "selection": "stratified by official task type (largest remainder), sha256 rank within type",
        "source_manifest": str(Path(args.manifest).resolve()),
        "source_manifest_train_sha256": train["game_ids_sha256"],
        "train_pool": train["loader_games"],
        "quota": quota,
        "support_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "support": support,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"K={args.k} from pool {train['loader_games']}")
    print("quota:", quota)
    print("support sha256:", out["support_sha256"])
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
