#!/usr/bin/env python3
"""Freeze local OOD multi-hop inventories. Never download or modify source data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bfas import hotpotqa as hp, multihop as mh


def prepare(data_dir=mh.DATA, manifest_dir=mh.MANIFEST_DIR):
    data_dir, manifest_dir = Path(data_dir), Path(manifest_dir)
    if manifest_dir.resolve().is_relative_to(data_dir.resolve()):
        raise ValueError("manifests must live outside source data")
    pending = []
    for dataset, spec in mh.DATASETS.items():
        source = data_dir / spec["source"]
        source_bytes = source.read_bytes()
        manifest = mh.make_manifest(dataset, mh.read_rows(source, source_bytes=source_bytes),
                                    hashlib.sha256(source_bytes).hexdigest())
        path = manifest_dir / mh.manifest_name(dataset)
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError(f"refusing to replace changed frozen inventory: {path}")
        pending.append((path, manifest))
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for path, manifest in pending:
        if not path.exists():
            hp.write_json(path, manifest)
    return {m["dataset"]: {k: v for k, v in m.items() if k != "ids"} for _, m in pending}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=mh.DATA)
    parser.add_argument("--manifest-dir", type=Path, default=mh.MANIFEST_DIR)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
