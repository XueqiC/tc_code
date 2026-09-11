# Local branch integration into rtd-unified

The starting worktree was clean on `rtd-unified` at
`ff97ef214318e1f80564e429ee38447c4ced307f`.
The local branch tips used were:

| Branch | Tip | Merge base with starting HEAD |
| --- | --- | --- |
| webshop-adapter | `80035e621a81d22a5600debf72edbdff96f77537` | `a076898638b648d19a52fcacf3977327aaa066cf` |
| gemma4-bfcl | `87b4440779c976c18cdeeba157208ae49a7f62c3` | `55647f4209ada54024ed3e6fab57ef8fa632db3c` |

First attempted:

```bash
git merge --no-commit --no-ff webshop-adapter gemma4-bfcl
```

Git refused to create `ORIG_HEAD.lock` under the main repository's
`.git/worktrees/tc-alignment-uni/`: read-only filesystem. No merge commit or
merge-parent metadata was created.

The working-file fallback was:

```bash
git diff --binary a076898638b648d19a52fcacf3977327aaa066cf webshop-adapter > /tmp/rtd-webshop-adapter.patch
git apply -3 /tmp/rtd-webshop-adapter.patch
```

The three-way apply also failed: Git could not create `index.lock` in the same
read-only metadata directory. A plain `git apply --check` identified the BFCL
shell driver as the overlapping file. The remaining WebShop changes applied with:

```bash
git apply --exclude=tools/bfcl_teacher_demos.sh /tmp/rtd-webshop-adapter.patch
```

`git show` materialized the shell driver's base and incoming versions in `/tmp`.
`git merge-file tools/bfcl_teacher_demos.sh <base-file> <incoming-file>` reported
conflicts in its model default and credential handling. Its final content came
from the newer `gemma4-bfcl` driver in the next step.

For BFCL:

```bash
git diff --binary 55647f4209ada54024ed3e6fab57ef8fa632db3c gemma4-bfcl > /tmp/rtd-gemma4-bfcl.patch
git apply --check /tmp/rtd-gemma4-bfcl.patch
git apply --exclude=src/bfas/adapters/bfcl.py \
  --exclude=src/bfas/ledger.py \
  --exclude=tools/bfcl_teacher_demos.sh /tmp/rtd-gemma4-bfcl.patch
```

The check identified those three overlapping files. For the two Python files,
`git show` materialized the BFCL merge-base and incoming contents in `/tmp`, then
`git merge-file <working-file> <base-file> <incoming-file>` combined each with the
already integrated WebShop changes. Conflicts were resolved as follows:

- `src/bfas/ledger.py`: retain both WebShop `usage` and BFCL `source_id` fields,
  plus actual teacher labels, BFCL paid-failure accounting, and prerequisite
  accounting from both branches.
- `src/bfas/adapters/bfcl.py`: retain the new FC teacher gateway, provider routing,
  memory prerequisite filtering, error accounting, and checker-error exclusion.
  Keep the existing public `demo_from_result` used by offline importers. Preserve
  `gpt-5.4` as the default and use `BFAS_TEACHER` beneath the BFCL-specific override;
  bind returned episodes to the model actually requested.
- `tools/bfcl_teacher_demos.sh`: use `gemma4-bfcl`'s newer driver, including ledger
  import, resume checks, provider handling, and selective-generation restoration.
  Set its fallback to `BFAS_BFCL_TEACHER`, then `BFAS_TEACHER`, then `gpt-5.4`;
  match the BFCL inventory's fallback to the same rule.

All files under `src/bfas/rtd/unified/` retain their starting HEAD bytes. Existing
unified tests and the current ALFWorld bank config paths were preserved.

The requested pytest selection initially stopped after 470 passing tests because
`test_rtd_v11_webshop`'s injected teacher lacked the new `config` and `usage`
attributes. Its CPU stub was updated to the merged teacher interface.

Part 2 adds the ALFWorld collector, its CPU tests, and
`docs/rtd_alfworld_luna_bank.md`. The shared teacher client gained optional
per-request output limits and disabled retries, and the BFAS gateway gained an
`AcquisitionStopped` exception that stops before an unpurchased attempt is logged.

Finally, **`git add -A` succeeded** and staged the integrated files and Part 2
changes. HEAD remains unchanged: this is a staged working-tree integration,
without merge ancestry or a commit. No GPU jobs or real API calls were run.

Final validation used the tree's `.venv`, which links to
`/home/xueqi/hq/projects/tc-alignment/.venv`:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q tests/ -x \
  -k 'unified or webshop or teacher or ledger or bfcl_ollama or conformance'
```

Result: **676 passed, 2166 deselected, 2 warnings** in 172.20 seconds. The two
warnings came from the existing local tiny-model teacher-forcing check.
`git diff --cached --check` also passed. New CPU tests include capped HTTP
transport, ledger resume/crash recovery, pool integrity, and conversion through
`convert_alfworld_bank` with a stub tokenizer.
