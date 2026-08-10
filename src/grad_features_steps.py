"""Per-block gradient features for fine-scale boundary estimation.

Output: results/pilot/features_gradsteps{TAG}.npz
  X_steps: [total_blocks, D]
  row_id: [total_blocks] global row indices in pilot-cache order
  block_idx: [total_blocks] response block indices within each row
  domain: [600] per-row domain labels
"""

import json
import os
from pathlib import Path

import numpy as np

from features import extract_features_stepwise


ROOT = Path(__file__).resolve().parent.parent
MODEL = os.environ.get("TC_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
TAG = os.environ.get("TC_TAG", "")
DOMAINS = ["gsm8k-code", "gsm8k-cot", "pandas", "sql", "alpaca"]
PROJ_DIM = 64
MAX_PROMPT_TOK = 640
MAX_RESP_TOK = 512


def main():
    rows = []
    domains = []
    for domain in DOMAINS:
        path = ROOT / "data" / "pilot" / f"{domain}.jsonl"
        with path.open() as handle:
            domain_rows = [json.loads(line) for line in handle]
        rows.extend(domain_rows)
        domains.extend([domain] * len(domain_rows))

    X_steps, row_id, block_idx = extract_features_stepwise(
        MODEL, rows, PROJ_DIM, MAX_PROMPT_TOK, MAX_RESP_TOK, "cuda"
    )

    out_dir = ROOT / "results" / "pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"features_gradsteps{TAG}.npz"
    np.savez_compressed(
        output_path,
        X_steps=X_steps,
        row_id=row_id,
        block_idx=block_idx,
        domain=np.asarray(domains),
    )
    print("saved", output_path)


if __name__ == "__main__":
    main()
