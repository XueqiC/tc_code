#!/usr/bin/env python3
"""C19/C19d historical BFCL condition contrasts; see cc_pairs.py --help.

The candidates command renders and checks teacher targets on CPU. Probe-parent
exclusion is opt-in via --exclude-probe-parents; no GPU is used by candidates.
Protection uses prompt/content hashes and only unambiguous source IDs, while
retaining official held-out parent exclusions. Rebuilds preserve previous probe
results as historical bytes and require a new probe before selection.

C19e select accepts --type3-cap (0.3333 = historical exact one third; 1.0
disables), --target (32; --limit remains an alias), and reporting-only
--min-per-type. Default settings preserve the legacy output bytes; overrides
record selection settings in confused_pairs.json and report.md. If --out lacks
probe_results.jsonl, all probe inputs come from --probe-from (data/cc_pairs_v1
by default), and only the selection, split and report are written to --out.
With --within-fraction 0 (C19f), only the whole seed function is held out;
every remaining selected pair trains, with train composition in report.md.

C21 evaluate compares greedy outcomes and teacher-forced kappa with the base
probe on train and held-out pairs. --model accepts a merged HF directory;
--base-url uses an existing vLLM server (optional --tag and --tokenizer).
Outputs default to results/cc_eval/<tag>/{pair_eval.jsonl,summary.json,report.md}.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from bfas.cc_pairs import main

if __name__ == "__main__":
    raise SystemExit(main())
