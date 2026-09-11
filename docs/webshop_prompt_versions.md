# WebShop prompt versions

`tools/webshop_eval.py` accepts `--prompt-version v1|v2`. The flag overrides
`WEBSHOP_PROMPT_VERSION`; the default is `v1`. For example:

```sh
envs/webshop/venv/bin/python tools/webshop_eval.py \
  --base-url http://localhost:8000/v1 --model student \
  --prompt-version v2 --out results/webshop/student_v2
```

`WebShopAdapter` uses the same environment variable and accepts an explicit
`prompt_version="v2"` constructor argument. It resolves the version at construction
and uses the shared evaluator prompt for teacher acquisition, student rollouts,
evaluation, and serving probes. Set `WEBSHOP_PROMPT_VERSION=v2` when running BFAS.

- **v1:** the original system prompt and bottle example, unchanged byte for byte;
  the current observation limit remains 2,500 characters.
- **v2:** the attribute/option/price verification and navigation rules, plus the
  exact deodorant trajectory from `webshop_prompt_v2_draft.md` in the main tree.
  The draft's current observation limit is 6,000 characters.

The evaluator's `--obs-chars` can override either limit. Both versions retain
15 steps, existing history limits, rendering, decoding, and action parsing.
`Thought:` lines are optional; parsing takes the last complete action line.

Evaluation records and metrics, teacher ledger rows, and demo packages save
`prompt_version`. Evaluator resumes reject a different version in the same output
directory; older unversioned records and demos count as v1. Ledger attempts and
cached demos are isolated by version. To inspect a mixed ledger, call
`load_ledger("webshop", prompt_version="v2")` (or `"v1"`).

BFAS keeps its existing v1 result and collection paths. v2 uses
`results/bfas/webshop/prompt_v2/` so collection caches cannot cross versions.
BFAS run metadata and training tags also record the version.
