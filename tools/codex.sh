#!/bin/bash
# Project codex wrapper (user directive 2026-09-05 00:45, tc-alignment channel): ALL code work from now on is
# delegated to Codex with gpt-6-astra @ xhigh. ops/codex_task.sh still pins gpt-5.6-sol by default; this wrapper
# overrides via the env hooks it exposes. Usage: bash tools/codex.sh "<task description>"
export CODEX_MODEL=gpt-6-astra CODEX_EFFORT=xhigh
exec bash ~/hq/ops/codex_task.sh tc-alignment "$@"
