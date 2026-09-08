"""Best-effort code provenance for Git checkouts and rsynced cluster copies."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess


def repo_commit(root) -> dict:
    """Return commit, dirty-entry count (None if unavailable), and provenance.

    Prefer Git, then BEHAVIOR_ATOM_COMMIT, then data/behavior_atom_v1/COMMIT.
    Both fallbacks accept a SHA followed by an optional ``dirty=N`` token.
    Provenance is one of git/env/file/no-git. Metadata failures never abort a run.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL).strip()
        if commit:
            status = subprocess.check_output(
                ["git", "status", "--short"], cwd=root, text=True,
                stderr=subprocess.DEVNULL)
            return {"commit": commit, "dirty": status.count("\n"), "provenance": "git"}
    except Exception:
        # Provenance is best effort, including missing executables and bad output.
        pass
    for source in ("env", "file"):
        try:
            value = (os.environ.get("BEHAVIOR_ATOM_COMMIT", "") if source == "env" else
                     (Path(root) / "data/behavior_atom_v1/COMMIT").read_text(encoding="utf-8"))
            tokens = value.split()
            if not tokens:
                continue
            dirty = next((int(token[6:]) for token in tokens[1:] if token.startswith("dirty=")), None)
            if dirty is not None and dirty < 0:
                continue
            return {"commit": tokens[0], "dirty": dirty, "provenance": source}
        except Exception:
            pass
    return {"commit": "unknown", "dirty": None, "provenance": "no-git"}
