"""Install HotpotQA from official JSON or HF parquet; validate frozen splits."""
from __future__ import annotations

import hashlib
from http.client import HTTPException
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
HF_BASE = "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/"
PARQUETS = {
    "train": ("distractor/train-00000-of-00002.parquet",
              "distractor/train-00001-of-00002.parquet"),
    "dev": ("distractor/validation-00000-of-00001.parquet",),
}


def validate_record(row, path):
    strings = ("_id", "question", "answer", "type", "level")
    if not isinstance(row, dict) or any(not isinstance(row.get(k), str) for k in strings):
        raise ValueError(f"{path}: malformed question fields")
    facts, context = row.get("supporting_facts"), row.get("context")
    if not isinstance(facts, list) or any(
        not isinstance(pair, list) or len(pair) != 2
        or not isinstance(pair[0], str) or type(pair[1]) is not int for pair in facts
    ):
        raise ValueError(f"{path}: malformed supporting_facts for {row['_id']}")
    if not isinstance(context, list) or any(
        not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[0], str)
        or not isinstance(pair[1], list) or any(not isinstance(s, str) for s in pair[1])
        for pair in context
    ):
        raise ValueError(f"{path}: malformed context for {row['_id']}")
    return row["_id"]


def validate_ids(ids, count, path, manifest=None):
    if len(ids) != count:
        raise ValueError(f"{path}: expected {count} questions, got {len(ids)}")
    available = set(ids)
    if len(available) != count:
        raise ValueError(f"{path}: duplicate IDs")
    if manifest is not None:
        if count != manifest["source_count"]:
            raise ValueError(f"{path}: count differs from the frozen inventory")
        digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        if digest != manifest["source_ids_sha256"]:
            raise ValueError(f"{path}: question order differs from the frozen inventory")
        for name in ("ids", "demand", "calibration"):
            missing = set(manifest.get(name, [])) - available
            if missing:
                raise ValueError(f"{path}: unresolved frozen {name}: {sorted(missing)[:5]}")
    return ids


def validate(path, count, manifest=None):
    with path.open(encoding="utf-8") as stream:
        rows = json.load(stream)
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError(f"{path}: expected {count} questions")
    ids = [validate_record(row, path) for row in rows]
    return validate_ids(ids, count, path, manifest)


def fetch(path, url, check):
    """Download to a sibling temporary file and validate before publishing."""
    partial = path.with_suffix(path.suffix + ".part")
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "BFAS-HotpotQA/1.0"})
            print(f"Downloading {request.full_url} -> {path}", flush=True)
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
            result = check(partial)
            partial.replace(path)
            return result
        except (OSError, ValueError, HTTPException) as exc:
            partial.unlink(missing_ok=True)
            if attempt == 2:
                raise RuntimeError(f"HotpotQA download failed for {path.name}: {exc}") from exc
            time.sleep(2 ** attempt)


def download(path, count, manifest=None):
    if path.exists():
        return validate(path, count, manifest)
    return fetch(path, BASE + path.name, lambda p: validate(p, count, manifest))


def parquet_reader():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "HuggingFace parquet conversion requires pyarrow; install into .venv only: "
            "uv pip install --python .venv/bin/python pyarrow"
        ) from exc
    return pq


def official_record(row):
    """Undo HF's struct-of-lists representation without reordering sentences."""
    try:
        result = {"_id": row["id"]}
        result.update({key: row[key] for key in ("question", "answer", "type", "level")})
        for key, values in (("supporting_facts", "sent_id"), ("context", "sentences")):
            columns = row[key]
            result[key] = [list(pair) for pair in zip(columns["title"], columns[values], strict=True)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Malformed HuggingFace parquet record {row.get('id')!r}: {exc}") from exc
    return result


def convert_parquets(paths, destination, count, manifest):
    pq = parquet_reader()
    partial = destination.with_suffix(".json.part")
    ids = []
    try:
        with partial.open("w", encoding="utf-8") as stream:
            stream.write("[")
            for path in paths:
                with pq.ParquetFile(path) as parquet:
                    for batch in parquet.iter_batches(batch_size=512):
                        for row in batch.to_pylist():
                            record = official_record(row)
                            task_id = validate_record(record, path)
                            if ids:
                                stream.write(",\n")
                            json.dump(record, stream, ensure_ascii=False)
                            ids.append(task_id)
            stream.write("]\n")
        validate_ids(ids, count, destination, manifest)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    return ids


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def from_huggingface(split, path, count, manifest):
    pq = parquet_reader()
    inputs, paths = [], []

    def check_parquet(partial):
        with pq.ParquetFile(partial):
            pass

    for remote in PARQUETS[split]:
        local = path.parent / Path(remote).name
        transport = "local" if local.exists() else "download"
        if transport == "download":
            fetch(local, HF_BASE + remote, check_parquet)
        paths.append(local)
        inputs.append({"file": local.name, "path": remote, "url": HF_BASE + remote,
                       "transport": transport, "sha256": sha256(local)})
    print(f"Converting HuggingFace distractor parquet -> {path}", flush=True)
    convert_parquets(paths, path, count, manifest)
    return {"source": "huggingface", "dataset": "hotpotqa/hotpot_qa", "config": "distractor",
            "split": "train" if split == "train" else "validation", "inputs": inputs}


def write_json(path, value):
    encoded = json.dumps(value, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == encoded:
        return
    partial = path.with_suffix(path.suffix + ".part")
    try:
        partial.write_text(encoded, encoding="utf-8")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    source_path = DATA / "SOURCE.json"
    previous = json.loads(source_path.read_text(encoding="utf-8")) if source_path.exists() else {}
    inventories = {}
    for split, (filename, count) in SOURCES.items():
        path = DATA / filename
        manifest_path = ROOT / f"configs/hotpotqa_{'support' if split == 'train' else 'eval'}_split.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if path.exists():
            validate(path, count, manifest)
            source = {"source": "existing_json"}
            if previous.get(split, {}).get("sha256") == sha256(path):
                source = previous[split]
        elif any((DATA / Path(remote).name).exists() for remote in PARQUETS[split]):
            # A user-supplied shard opts this split into the mirror immediately.
            source = from_huggingface(split, path, count, manifest)
        else:
            try:
                download(path, count, manifest)
                source = {"source": "official", "url": BASE + filename}
            except RuntimeError as exc:
                print(f"{exc}; falling back to HuggingFace", flush=True)
                source = from_huggingface(split, path, count, manifest)
        inventories[split] = {**source, "file": filename, "count": count, "sha256": sha256(path)}
        # Preserve provenance even if installing the next split fails.
        previous[split] = inventories[split]
        write_json(source_path, previous)
        print(f"Validated {filename}: {count} questions", flush=True)
    write_json(DATA / "manifest.json", inventories)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        sys.exit(str(exc))
