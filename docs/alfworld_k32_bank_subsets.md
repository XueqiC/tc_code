# Frozen K=32 pi1 sub-banks

All three registrations preserve every non-bank field of
`configs/rtd/pi1_alfworld_k32_kang.yaml`, including `method: pi1_ce`.
Only bank path, manifest SHA, package count, turn count, and supervised-token
count differ. `support_size` remains 32. There were no new purchases, ledger
writes, environment replays, GPU use, or training runs.

| Bank under `artifacts/` | Packages | Turns | Authored tokens | Native boundaries | Supervised tokens |
|---|---:|---:|---:|---:|---:|
| alfworld_k32_d0_lowsweep | 15 | 136 | 2354 | 136 | 2490 |
| alfworld_k32_d0_highsweep | 15 | 288 | 6871 | 288 | 7159 |
| alfworld_k32_smartad_all87 | 87 | 1343 | 32093 | 1343 | 33436 |

The matching registrations are `configs/rtd/pi1_alfworld_k32_<name>.yaml`.

## Counting and source identity

`pi1.load_bank` performs the existing frozen-bank audit and constructs all
rows through `FrozenRenderer`; packages and turns are checked against its
returned identity. Tokens use the exact `alf_pi1_train.py --preflight` rule:
`sum(len(pi1.encode_teacher_turn(tokenizer, row, 32768)["target_ids"]) for row in rows)`.
Every complete teacher reply is encoded without automatic special tokens,
followed by native `<turn|>` ID 106. Prompts and observations are masked.
No row is truncated. D0 totals reconcile to 30 packages / 424 turns / 9,649 tokens.

Tokenizer: `google/gemma-4-12B-it`, loaded locally via `FrozenRenderer` /
`AutoTokenizer.from_pretrained(..., local_files_only=True, trust_remote_code=False)`.
Cached snapshot:
`/home/xueqi/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`.
Tokenizer identity SHA-256: `6e5d85e93f519ee8c8ce8db1432ce8fc2fdfdfcfc37c0148058127580300793d`.

Each derived `sealed/manifest.json` records the absolute source bank path,
source sealed-manifest SHA-256, selection rule, and complete included-ID list
under `derivation`. Selected sealed package files are copied byte for byte.
All derived request/integrity inventories, support, audit, and artifact hashes
are regenerated. The full frozen support is preserved, so its signed content
hash is unchanged. The entire `historical_inventory`, including original
ledger paths and SHA-256 hashes, is retained rather than prorating purchases.

## SmartAD export and repeated tasks

`tools/alfworld_teacher_pool.py:export_pool` iterates every ledger row, derives
a separate query ID, verifies each candidate, and appends every package/record.
`method="plain"` does **not** reduce this to one package per task;
`candidates_per_task` reports collection shortfall and does not filter exports.
`pi1.load_bank` likewise loops over usable query IDs without task deduplication.
The all87 bank therefore contains all 87 verified candidates: 28 tasks with
3 packages, one with 2, and one with 1; two of the 32 support tasks have none.

The subset tool copies the existing SmartAD sealed packages instead of calling
`export_pool` again, avoiding unnecessary environment replay and any overwrite
of `candidate_sets.json`. Candidate IDs were checked against the source
collection index and its bound bank-manifest hash.
Collection metadata inside the package bytes remains `smartad`.
The new config selects ordinary plain CE; no SmartAD selection or weighted
loss is invoked. No collection is fabricated or relabeled `plain`.

For read-only historical accounting, `collection_cost` requires the original
collection explicitly (derived banks have no new `.collection` directory).
Use `method="ce"` with the original D0 collection for either D0 subset, and
`method="smartad"` with `artifacts/alfworld_k32_smartad_n3.collection` for all87.
This follows collection identity; `method="ce"` with a SmartAD collection
would correctly raise `method and collection identity differ`. All original
failed and unselected purchases remain attributable to their source collection;
zero new purchases does not mean zero historical cost.

## Reproduction

Run against fresh output/config paths (existing paths are refused):

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python tools/alf_bank_subset.py \
  --source /home/xueqi/hq/projects/tc-alignment/data/rtd/v1_alfworld_k32_d0 \
  --output artifacts/alfworld_k32_d0_lowsweep --selection sweep-low \
  --config configs/rtd/pi1_alfworld_k32_d0_lowsweep.yaml

CUDA_VISIBLE_DEVICES='' .venv/bin/python tools/alf_bank_subset.py \
  --source /home/xueqi/hq/projects/tc-alignment/data/rtd/v1_alfworld_k32_d0 \
  --output artifacts/alfworld_k32_d0_highsweep --selection sweep-high \
  --config configs/rtd/pi1_alfworld_k32_d0_highsweep.yaml

CUDA_VISIBLE_DEVICES='' .venv/bin/python tools/alf_bank_subset.py \
  --source artifacts/alfworld_k32_smartad_n3 \
  --output artifacts/alfworld_k32_smartad_all87 --selection all-usable \
  --config configs/rtd/pi1_alfworld_k32_smartad_all87.yaml
```

For explicit selection, replace `--selection` with `--package-ids ids.json`
(a JSON array) or repeated `--package-id ID`. `--expected-source-sha256`
can additionally pin the source manifest. Sweep invocations print the full
sweep-share table and both ordered ID lists as JSON.

## Sweep shares and selections

Share is the number of commands matching
`^(go to|open|close) (cabinet|drawer) \d+$` divided by all commands in that
package. Sort by exact rational share, then lexicographic package ID;
take the first/last 15. A 1/4 tie crosses the cut and is resolved by ID.

| Package ID | Matching / total commands | Sweep share | Sub-bank |
|---|---:|---:|---|
| `1fb142dbf57dcf970f353f0e65384e895b8e3d6c7d3317bc19760b54be1a1bf8` | 0/6 | 0.000000 | low |
| `4826dffe590f2f0cceb9029f2e02b858e03ddf3bab6c89768e13dc093f2d7440` | 0/20 | 0.000000 | low |
| `701892425e455f66dc4f29ecd8361a7b1dc73aa345968c9b6345813ca73492b0` | 0/29 | 0.000000 | low |
| `8392f4d6500886113cf9710d357877fc78f1fc45fdb0802b3fda1f84ace10bb5` | 0/4 | 0.000000 | low |
| `8bfd1b68a6e907e212b990b305a7cd2d902318eb2f9ba1b56bfe6a70d4ee8628` | 0/4 | 0.000000 | low |
| `ace63e18226b06fae98deebe60ab4f3bdbee79a8602ed536664be56cb44954a9` | 0/7 | 0.000000 | low |
| `b5e88a12160c0fd22648096f8fb826db6900ecd2988f7942c1812a03ab0dd4c7` | 0/4 | 0.000000 | low |
| `b7be71c5f3888ea9c5d2202d4904831b7a044bab2e019b8dadb9ccb028fab633` | 0/7 | 0.000000 | low |
| `c77cbb2d9ead3c799cc2dec71e30dbb3b6fe642781bbbba6c098f2cf1f9c2e2b` | 0/6 | 0.000000 | low |
| `cae93dc82af4e919f3a480dd37b9186a2846b08b0295c7bc5317d3cd733d0efa` | 0/4 | 0.000000 | low |
| `d9a2b2475b57b06a0b9fe3a9a4bdd14826cd083735855770060c015f2c09081c` | 1/8 | 0.125000 | low |
| `93caed2c2e9b823ce605818d21b3849dac2db24b6d1f5d5e59872b28bb128b22` | 2/11 | 0.181818 | low |
| `3ac5ae4b5cd163723930fe26d1e2e3153eb3dd6a754e8440b84d488b64d13ac8` | 2/10 | 0.200000 | low |
| `2d6e9911520a1b2b1f43615865edc1cf5939b3deaf4c2a8777bb5c49a2d17b6c` | 2/8 | 0.250000 | low |
| `d3f231fc325f2eff22465cb9e8759ea1892b23403b849d996856ee8d9d16a747` | 2/8 | 0.250000 | low |
| `ef2b83f02d9b9bf2dcbddab1812a6c31dd71f3709e6688d94090c40bd12ecc1d` | 2/8 | 0.250000 | high |
| `b54a22c3905cdeba5186758668a82f264fc04fe43c76c5375a593e3279562acd` | 2/7 | 0.285714 | high |
| `4b51e2a3cb6289da371d8462bfbc92cba57cdf58afecde1d2f50ea49dd22f0b8` | 3/9 | 0.333333 | high |
| `8b0f4280b1735f06a415ebb727a630a19d5b8ff98e1d097405b589e434b32326` | 9/25 | 0.360000 | high |
| `74acf3e27de7ed3d7da13516130c157af53ddf98afe6562e257d5471396d8097` | 3/7 | 0.428571 | high |
| `830a5a26e53e166ad539dbdb01fc255dbd18cc06b464f4e75777231c450450c3` | 7/16 | 0.437500 | high |
| `1d888ffcc6b99cea8ef77144555449799fdaa197e763dda64944930afed4d281` | 4/8 | 0.500000 | high |
| `e1e9ebde478518df871cd30b549608b2d35b3b11dccc7b2ea0b0a788ab888d60` | 7/13 | 0.538462 | high |
| `94ed508d0f7fd6d62dc863fe1d9728d886fa4eb9eb1c2d6323cbd63d7bf5bf49` | 10/18 | 0.555556 | high |
| `3b4113a8a2a7800f7dc65e4f744b5d29304ea531a0e05515aedb960ef70cfeaa` | 23/38 | 0.605263 | high |
| `f9112082664f7e04526111e3548362e6416b9334a0df75e35f785d909f757ed8` | 24/37 | 0.648649 | high |
| `2cdace730840750a6b06d8254827fe85115dfbcd236fdbe2686cb71de4a0267c` | 18/27 | 0.666667 | high |
| `2abc7f69636c15ba88064f7fcd2d40bdc128abae0372d4536a338584026867ef` | 17/23 | 0.739130 | high |
| `1523a82e1439e38a4dd6ff34769654a5bbf620a1ed1636573a184e476934c12a` | 25/33 | 0.757576 | high |
| `609df9a36261d5bc31855983501c39c919e754bdbb53ffcd671110bcb76fea8c` | 15/19 | 0.789474 | high |

Low IDs in sweep rank order:

```json
[
  "1fb142dbf57dcf970f353f0e65384e895b8e3d6c7d3317bc19760b54be1a1bf8",
  "4826dffe590f2f0cceb9029f2e02b858e03ddf3bab6c89768e13dc093f2d7440",
  "701892425e455f66dc4f29ecd8361a7b1dc73aa345968c9b6345813ca73492b0",
  "8392f4d6500886113cf9710d357877fc78f1fc45fdb0802b3fda1f84ace10bb5",
  "8bfd1b68a6e907e212b990b305a7cd2d902318eb2f9ba1b56bfe6a70d4ee8628",
  "ace63e18226b06fae98deebe60ab4f3bdbee79a8602ed536664be56cb44954a9",
  "b5e88a12160c0fd22648096f8fb826db6900ecd2988f7942c1812a03ab0dd4c7",
  "b7be71c5f3888ea9c5d2202d4904831b7a044bab2e019b8dadb9ccb028fab633",
  "c77cbb2d9ead3c799cc2dec71e30dbb3b6fe642781bbbba6c098f2cf1f9c2e2b",
  "cae93dc82af4e919f3a480dd37b9186a2846b08b0295c7bc5317d3cd733d0efa",
  "d9a2b2475b57b06a0b9fe3a9a4bdd14826cd083735855770060c015f2c09081c",
  "93caed2c2e9b823ce605818d21b3849dac2db24b6d1f5d5e59872b28bb128b22",
  "3ac5ae4b5cd163723930fe26d1e2e3153eb3dd6a754e8440b84d488b64d13ac8",
  "2d6e9911520a1b2b1f43615865edc1cf5939b3deaf4c2a8777bb5c49a2d17b6c",
  "d3f231fc325f2eff22465cb9e8759ea1892b23403b849d996856ee8d9d16a747"
]
```

High IDs in sweep rank order:

```json
[
  "ef2b83f02d9b9bf2dcbddab1812a6c31dd71f3709e6688d94090c40bd12ecc1d",
  "b54a22c3905cdeba5186758668a82f264fc04fe43c76c5375a593e3279562acd",
  "4b51e2a3cb6289da371d8462bfbc92cba57cdf58afecde1d2f50ea49dd22f0b8",
  "8b0f4280b1735f06a415ebb727a630a19d5b8ff98e1d097405b589e434b32326",
  "74acf3e27de7ed3d7da13516130c157af53ddf98afe6562e257d5471396d8097",
  "830a5a26e53e166ad539dbdb01fc255dbd18cc06b464f4e75777231c450450c3",
  "1d888ffcc6b99cea8ef77144555449799fdaa197e763dda64944930afed4d281",
  "e1e9ebde478518df871cd30b549608b2d35b3b11dccc7b2ea0b0a788ab888d60",
  "94ed508d0f7fd6d62dc863fe1d9728d886fa4eb9eb1c2d6323cbd63d7bf5bf49",
  "3b4113a8a2a7800f7dc65e4f744b5d29304ea531a0e05515aedb960ef70cfeaa",
  "f9112082664f7e04526111e3548362e6416b9334a0df75e35f785d909f757ed8",
  "2cdace730840750a6b06d8254827fe85115dfbcd236fdbe2686cb71de4a0267c",
  "2abc7f69636c15ba88064f7fcd2d40bdc128abae0372d4536a338584026867ef",
  "1523a82e1439e38a4dd6ff34769654a5bbf620a1ed1636573a184e476934c12a",
  "609df9a36261d5bc31855983501c39c919e754bdbb53ffcd671110bcb76fea8c"
]
```

The same complete lists, table, tokenizer identity, and count receipts are in
`artifacts/alfworld_k32_bank_subsets.json`.

## CPU validation

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q tests/test_alf_bank_subset.py
```

Tests exercise explicit-ID CLI round trips, unchanged sealed bytes, original
ledger provenance, deterministic 15/15 partitioning (including the boundary
tie), recipe equality, all87 candidate coverage, malformed selections,
overwrite/tamper rejection, and the real `alf_pi1_train.py --preflight` for
each of the three configs. All three preflights use only CPU and cached files.

Result: `17 passed in 59.55s` (2026-09-22, `CUDA_VISIBLE_DEVICES=''`).
