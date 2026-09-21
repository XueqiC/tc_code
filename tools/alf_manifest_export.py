#!/usr/bin/env python3
"""Export the ALFWorld split manifests from the official loader (K=32 protocol, 2026-09-21).

Runs inside envs/alfworld/.venv. Read-only: it never writes into the data root.
Emits, per split, the loader's own game list plus the reason every on-disk game
that the loader dropped was dropped, so the registered support size is traceable
to the harness rather than to a directory count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

OFFICIAL_TASK_TYPES = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
)
SPLIT_DIRS = {"train": "train", "valid_seen": "valid_seen", "valid_unseen": "valid_unseen"}
SPLIT_EVAL_MODE = {
    "train": "train",
    "valid_seen": "eval_in_distribution",
    "valid_unseen": "eval_out_of_distribution",
}


def disk_scan(split_dir: Path) -> tuple[list[str], dict[str, int]]:
    """Every game.tw-pddl under the split, with the reason each one is kept or dropped."""
    kept: list[str] = []
    reasons: dict[str, int] = {}

    def note(reason: str) -> None:
        reasons[reason] = reasons.get(reason, 0) + 1

    for path in sorted(split_dir.rglob("game.tw-pddl")):
        rel = str(path.parent.relative_to(split_dir))
        if "movable" in str(path) or "Sliced" in str(path):
            note("excluded_movable_or_sliced")
            continue
        try:
            game = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            note("game_file_unreadable")
            continue
        try:
            traj = json.loads((path.parent / "traj_data.json").read_text())
        except (OSError, json.JSONDecodeError):
            note("traj_data_missing_or_unreadable")
            continue
        if game.get("solvable") is not True:
            note("not_solvable")
            continue
        if traj.get("task_type") not in OFFICIAL_TASK_TYPES:
            note(f"task_type_out_of_scope:{traj.get('task_type')}")
            continue
        note("kept")
        kept.append(rel)
    return kept, reasons


def loader_games(data_root: Path, split: str) -> list[str]:
    """Ask the installed ALFWorld loader itself which games the split contains."""
    import alfworld.agents.environment as environment

    split_dir = data_root / SPLIT_DIRS[split]
    config = {
        "dataset": {
            "data_path": str(split_dir),
            "eval_id_data_path": str(split_dir),
            "eval_ood_data_path": str(split_dir),
            "num_train_games": -1,
            "num_eval_games": -1,
        },
        "env": {
            "type": "AlfredTWEnv",
            "goal_desc_human_anns_prob": 0,
            "task_types": [1, 2, 3, 4, 5, 6],
            "domain_randomization": False,
            "expert_type": "handcoded",
        },
        "general": {"training_method": "dagger"},
        "dagger": {"training": {"max_nb_steps_per_episode": 50}},
    }
    env = environment.get_environment("AlfredTWEnv")(config, train_eval=SPLIT_EVAL_MODE[split])
    files = getattr(env, "game_files", [])
    out = []
    for f in files:
        p = Path(f)
        parent = p.parent if p.name.endswith(".tw-pddl") else p
        try:
            out.append(str(parent.relative_to(split_dir)))
        except ValueError:
            out.append(str(parent))
    return sorted(out)


def task_type_of(split_dir: Path, game_id: str) -> str:
    try:
        return json.loads((split_dir / game_id / "traj_data.json").read_text()).get("task_type", "?")
    except (OSError, json.JSONDecodeError):
        return "?"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="train,valid_seen,valid_unseen")
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import alfworld

    report: dict[str, object] = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "alfworld_version": getattr(alfworld, "__version__", "unknown"),
        "alfworld_path": alfworld.__file__,
        "python": sys.version.split()[0],
        "data_root": str(data_root),
        "official_reference_sizes": {"train": 3553, "valid_seen": 140, "valid_unseen": 134},
        "splits": {},
    }

    for split in args.splits.split(","):
        split = split.strip()
        if not split:
            continue
        split_dir = data_root / SPLIT_DIRS[split]
        kept, reasons = disk_scan(split_dir)
        try:
            loader = loader_games(data_root, split)
            loader_error = None
        except Exception as exc:  # the loader is third-party; record, do not mask
            loader, loader_error = [], f"{type(exc).__name__}: {exc}"

        by_type: dict[str, int] = {}
        for gid in (loader or kept):
            t = task_type_of(split_dir, gid)
            by_type[t] = by_type.get(t, 0) + 1

        report["splits"][split] = {
            "split_dir": str(split_dir),
            "disk_game_files": sum(reasons.values()),
            "disk_kept_after_filters": len(kept),
            "disk_filter_reasons": reasons,
            "loader_games": len(loader),
            "loader_error": loader_error,
            "loader_minus_disk": sorted(set(loader) - set(kept))[:20],
            "disk_minus_loader": sorted(set(kept) - set(loader))[:20],
            "task_type_counts": by_type,
            "game_ids_sha256": hashlib.sha256(
                "\n".join(loader or kept).encode()
            ).hexdigest(),
            "game_ids": loader or kept,
        }
        print(
            f"[{split}] disk_files={sum(reasons.values())} disk_kept={len(kept)} "
            f"loader={len(loader)} err={loader_error}",
            flush=True,
        )

    out_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
