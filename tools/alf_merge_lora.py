#!/usr/bin/env python3
"""Merge a LoRA adapter into a full snapshot the vLLM server can load.

The served evaluation backend binds a merged snapshot, not an adapter, so each
trained arm needs one of these before it can be evaluated. Written by the
orchestrating session on the critical path; queued for Codex review.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, type=Path, help="base model snapshot")
    ap.add_argument("--adapter", required=True, type=Path, help="PEFT adapter directory")
    ap.add_argument("--out", required=True, type=Path, help="new directory for the merged snapshot")
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing existing output: {args.out}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    model = AutoModelForCausalLM.from_pretrained(
        args.base, local_files_only=True, dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True, device_map="cpu")
    model = model.merge_and_unload()
    staging = args.out.with_name(args.out.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    model.save_pretrained(staging, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base, local_files_only=True).save_pretrained(staging)
    # Gemma 4 is multimodal: the server's AutoProcessor needs the base's processor
    # metadata, which saving a causal LM plus tokenizer does not emit.
    for meta in args.base.iterdir():
        if meta.is_file() and meta.suffix != ".safetensors" and not (staging / meta.name).exists():
            shutil.copy2(meta, staging / meta.name)
    # Carry the provenance so a served campaign can be traced to its adapter.
    (staging / "merge_provenance.json").write_text(json.dumps(
        dict(base=str(args.base.resolve()), adapter=str(args.adapter.resolve())), indent=2))
    os.replace(staging, args.out)
    print(f"merged -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
