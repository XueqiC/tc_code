#!/usr/bin/env bash
# All downloads and temporary files stay in envs/hotpotqa. No LLM calls.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${repo_root}/.venv/bin/python" -B "${repo_root}/scripts/setup_hotpotqa.py"
