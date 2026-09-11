"""Download official HotpotQA JSON atomically; validate the frozen inventories."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "envs/hotpotqa/data"
SOURCES = {
    "train": ("hotpot_train_v1.1.json", 90447),
    "dev": ("hotpot_dev_distractor_v1.json", 7405),
}
BASE = "https://curtis.ml.cmu.edu/datasets/hotpot/"


def validate(path, count):
    with path.open(encoding="utf-8") as stream:
        rows = json.load(stream)
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError(f"{path}: expected {count} questions")
    ids = [r["_id"] for r in rows]
    if len(set(ids)) != count or any(not isinstance(r.get("question"), str)
                                       or not isinstance(r.get("answer"), str) for r in rows):
        raise ValueError(f"{path}: malformed questions or duplicate IDs")
    return ids


def download(path, count):
    if path.exists():
        return validate(path, count)
    partial = path.with_suffix(".json.part")
    for attempt in range(3):
        try:
            request = urllib.request.Request(BASE + path.name, headers={"User-Agent": "BFAS-HotpotQA/1.0"})
            print(f"Downloading {request.full_url} -> {path}", flush=True)
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
            ids = validate(partial, count)
            partial.replace(path)
            return ids
        except (OSError, ValueError) as exc:
            partial.unlink(missing_ok=True)
            if attempt == 2:
                raise RuntimeError(f"HotpotQA download failed for {path.name}: {exc}") from exc
            time.sleep(2 ** attempt)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    inventories = {}
    for split, (filename, count) in SOURCES.items():
        path = DATA / filename
        ids = download(path, count)
        manifest = json.loads((ROOT / f"configs/hotpotqa_{'support' if split == 'train' else 'eval'}_split.json").read_text())
        if hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest() != manifest["source_ids_sha256"]:
            raise ValueError(f"{filename}: question order differs from the frozen inventory")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        inventories[split] = {"file": filename, "url": BASE + filename, "count": count, "sha256": digest}
        print(f"Validated {filename}: {count} questions", flush=True)
    destination = DATA / "manifest.json"
    encoded = json.dumps(inventories, indent=2) + "\n"
    if not destination.exists() or destination.read_text() != encoded:
        destination.write_text(encoded)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        sys.exit(str(exc))
