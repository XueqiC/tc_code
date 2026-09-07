# RTD v1 data-access table — frozen C23 inventory

This is broker/ingestion research accounting, **not selector input**. Cost rows
that describe a subset, merged artifact or derived events are **not additive**.
Every payload stays under `data/rtd/v1_bfcl/sealed/`; raw input files remain
read-only. Source hashes are recorded in the bank's sealed audit.

| Source | Observed content / historical cost | Authorized RTD access |
|---|---|---|
| `data/bfcl_sft/demos_ds.json` | 34 merged verified demos from 40 demand parents × 3 attempts. Associated archive account: **1,233,607 exact output tokens**; no extra charge for this merged copy. | Ingestion provenance only. It cannot reconstruct failed attempts or assign their costs. |
| `result_demos_deepseek_v4_pro_FC_a1/**/*_result.json` under the BFCL checkout | 111 records; **465,825 exact output tokens**. Current demand portion: 40 attempts / 22,291 tokens. | One package per real attempt; failures retained, raw nested usage preserved. |
| `result_demos_deepseek_v4_pro_FC_a2/**/*_result.json` | 72 records; **344,800 exact output tokens**. Current demand portion: 40 attempts / 23,212 tokens. | Same; memory prefix dependencies retain their request IDs. |
| `result_demos_deepseek_v4_pro_FC_a3/**/*_result.json` | 72 records; **422,982 exact output tokens**. Current demand portion: 40 attempts / 24,815 tokens. | Same; unrelated earlier-split entries are historical-accounting-only. |
| `data/bfcl_sft/pool_events_pref_v3t.jsonl` | **302 rows**: 249 `teacher_authored_gt`, 38 `teacher_authored_abstain`, 15 DeepSeek demo rows. Teacher-authored GT is generator evidence. Broader generation cost **142,727 (~143k) tokens estimated**, not another event bill. | Sealed aliases to the originating package by full prompt/content; no row-level charge. 247 matched rows, 55 unavailable without matching selected-source records. Raw `_rejected`, success and Delta-U fields never become source samples or selector features. |
| `data/bfcl_sft/gen_pool_v3.jsonl` | **303 items**, five official seed parents; **33,452 estimated output tokens** in the historical ledger. | One estimated item package; authored question and ground truth revealed together. Reused IDs are not keys. |
| `data/bfcl_sft/gen_oos.jsonl` | **140 items**, 28 official seed parents; **10,613 estimated output tokens**. | Same. Empty authored ground truth is an empty action plus EOS; no invented refusal. |
| `results/analysis/bfcl_teacher_tokens.json` | Exact demo total + estimated generator total and per-file estimates. | Privileged cost reconstruction/audit. Never actual-cost features before purchase. |
| `data/bfcl_sft/calibration_ids.json` | **60 protected generated prompt hashes** (SHA1 prefixes), no teacher expenditure added. | Exclusion metadata only. Never training/gradients/acquisition/feature statistics/preconditioning/hyperparameter choice. |
| `data/bfcl_sft/calibration_ids_official_v2.json` | **65 official calibration tasks**, no teacher expenditure added. | Same permanent protection, including their official parent identities. |
| Any additional `data/bfcl_sft/calibration_ids*.json` | Loaded by glob, never silently ignored. | Same permanent exclusion contract. |
| `configs/support_split.json` | Shared split manifest (includes non-BFCL IDs); heldout/calibration sections are exclusions. | Exclusion metadata only; shared support entries do not authorize extra BFCL teacher evidence. |
| `configs/bfcl_support_split.json` | 40 BFCL demand IDs and 10 old calibration IDs. | Demand identifies support parents; calibration remains excluded. |
| `data/behavior_atom_v1/probes*.json`, `data/bfcl_atom_v1/probes*.json`, when present | Previously registered probe prompt/content exclusions. | Reuse `cc_pairs.Exclusions`; no probe outcome/teacher text as RTD input. |
| Official BFCL task entries loaded by `BFCLAdapter` | Question, tool schemas, initial configuration and memory dependency declarations. No teacher label price attached. | Recover support parent content and valid initial states. Official answers remain verifier-only for official tasks. |

The 120 demand attempts total **70,318 exact output tokens**. The full 1,233,607
figure additionally includes 135 prerequisite/stale records (1,163,289 tokens).
These remain in the historical account; they are not amortized across current
packages. The 84 usable single-turn demand attempts total 21,203 tokens.

Generated items were originally emitted through batched writer calls with possible
verification/repair calls. Their provider request boundaries, usage, discarded
prose and drafts are incomplete. The requested item package convention is
**exploratory**, with estimated costs; it cannot certify real online request
accounting. Per-item allocation preserves each recorded file estimate, weighting
stored teacher-authored UTF-8 bytes and assigning integer remainders in row order.
No gen-ID averaging, pair-count amplification or hidden-usage-derived cap is used.

## Broader generation account (audit only unless selected above)

These are all file entries in the existing 142,727-token generator account. They
are not silently imported as additional candidates. Their sum is the historical
total; the two selected file estimates are already included in it.

| File under `data/bfcl_sft/` | Rows | Recorded output-token estimate | C23 access |
|---|---:|---:|---|
| `gen_fb_smoke.jsonl` | 3 | 633 | Historical cost audit only |
| `gen_mem_smoke.jsonl` | 6 | 1,209 | Historical cost audit only |
| `gen_mf_smoke.jsonl` | 2 | 993 | Historical cost audit only |
| `gen_mt_smoke.jsonl` | 3 | 1,537 | Historical cost audit only |
| `gen_oos.jsonl` | 140 | 10,613 | Selected item packages |
| `gen_oos_smoke.jsonl` | 3 | 278 | Historical cost audit only |
| `gen_pool.jsonl` | 30 | 4,570 | Historical cost audit only |
| `gen_pool_v2.jsonl` | 26 | 3,667 | Historical cost audit only |
| `gen_pool_v3.jsonl` | 303 | 33,452 | Selected item packages |
| `gen_pool_v4.jsonl` | 33 | 4,329 | Historical cost audit only |
| `gen_reply_smoke.jsonl` | 2 | 986 | Historical cost audit only |
| `gen_silent_smoke.jsonl` | 2 | 1,229 | Historical cost audit only |
| `gen_smoke.jsonl` | 6 | 933 | Historical cost audit only |
| `gen_stateful.jsonl` | 72 | 25,213 | Historical cost audit only |
| `gen_stateful_v2.jsonl` | 72 | 24,122 | Historical cost audit only |
| `gen_stateful_v3.jsonl` | 64 | 28,363 | Historical cost audit only |
| `gen_unified_smoke.jsonl` | 2 | 202 | Historical cost audit only |
| `gen_ws_smoke.jsonl` | 2 | 390 | Historical cost audit only |

## Measured bank and split

The official parent union is **m=40**; all generator seed parents are already
in the demo demand set. The content-hash split is 22/18 parents, with explicit
mapping in `configs/rtd/v1_bfcl_support.json`.

The sealed inventory has 698 entries, with 420 structurally available before
budgeting (84 demos + 336 generated items), and 278 unavailable (36 stateful
requests, 135 prerequisite/stale records, 107 protected generated items).
The available subset costs **55,370 = 21,203 exact + 34,167 estimated** output
tokens. Aggregate historical cost is a separate quantity.

The frozen uniform 131,072-token replay cap makes the 10/25/50% budget checkpoints
purchase-infeasible. The hard ledger must offer continue-training only at those
limits. Caps are not reduced based on sealed actual usage. No online request is
authorized by these retrospective estimates.

## Input hashes

The table below pins all files used in this build. The per-source `Exclusions`
metadata additionally records hashes of loaded probe exclusion manifests.

| Input | SHA256 |
|---|---|
| `configs/bfcl_support_split.json` | `69fb6957da0dadc1ea773d8fbe1793aaf0fa6db71737e2e8821457b38bdefed2` |
| `configs/support_split.json` | `4e607a35a15163765f51b038863e2e7f6589bdf5df121b884f77d01be8a29f96` |
| `data/bfcl_sft/calibration_ids.json` | `f4fff0c4e9ce4f34d82d67a4508eb52ff9b96ea310b2c2dcc29e4332e8e78b82` |
| `data/bfcl_sft/calibration_ids_official_v2.json` | `462efd11a02dca7779c336d3a622e17aa4f76ba4076d0f5e2fcaef0d2df2dc48` |
| `data/bfcl_sft/demos_ds.json` | `f1a0246aa37c03ced788348de51849c5ab72e1af461ff89df068c56efb930691` |
| `data/bfcl_sft/gen_oos.jsonl` | `dd01dc8345f2ddc95e2b35f25ee02c48152b07533a83936f93c6965af6f292c0` |
| `data/bfcl_sft/gen_pool_v3.jsonl` | `c15921eb747c3956ab852c4ca9e1a50d3f92b7bdf76119fc7fa981648c8de78f` |
| `data/bfcl_sft/pool_events_pref_v3t.jsonl` | `1ccbff20d4801a87aae0701dfa4f4f1e45c3610994b4d9b15b9634da7b8c0269` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a1/deepseek-v4-pro-FC/agentic/BFCL_v4_web_search_result.json` | `e552f8fc86828088c5fb2ddafe4bb79fefc2c7eb0e02af8bf74ce98bf20c6975` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a1/deepseek-v4-pro-FC/agentic/memory/kv/BFCL_v4_memory_kv_prereq_result.json` | `151f6c7be2a96b1afdb0fcc70f8493a9d80ad14a56872c3a0a6915172cdf3974` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a1/deepseek-v4-pro-FC/agentic/memory/kv/BFCL_v4_memory_kv_result.json` | `b28d4a1ebb713ca04843215b3928185cd54f3c2fe8bc39d01f902d9a544bb3bc` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a1/deepseek-v4-pro-FC/agentic/memory/rec_sum/BFCL_v4_memory_rec_sum_prereq_result.json` | `e0c33a82016db3e52e1aacf5004037ffa727a9fdac96f69d812cd620f57327cc` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a1/deepseek-v4-pro-FC/agentic/memory/rec_sum/BFCL_v4_memory_rec_sum_result.json` | `5f46cd2b7ed03d5677523ad5f11ab3c1d302194baec0c6327e2fd515f1c4cd6e` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a1/deepseek-v4-pro-FC/agentic/memory/vector/BFCL_v4_memory_vector_prereq_result.json` | `f48cc5134068d0ad02942e47c2ce7cd726c9c1c4231f9fee01808f8d3da0f0a6` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a1/deepseek-v4-pro-FC/agentic/memory/vector/BFCL_v4_memory_vector_result.json` | `f9e107d1162baf0d92f8edf2c9edad1ba4336da25859d64ed1f205c208926861` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a1/deepseek-v4-pro-FC/live/BFCL_v4_live_irrelevance_result.json` | `62f968d4eccf770095894edf9405583f7357892c7898b7661acfbb22bae008b1` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a1/deepseek-v4-pro-FC/live/BFCL_v4_live_multiple_result.json` | `5eeaae7c652682b9e37568659a029b2e322b2dd42b6f85d38f9888187d9e3be1` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a1/deepseek-v4-pro-FC/live/BFCL_v4_live_parallel_multiple_result.json` | `3a773f36d0005a9ea1ad14b24d75a9994676ee265cf9b1625f01ef41cc1063cb` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a1/deepseek-v4-pro-FC/live/BFCL_v4_live_parallel_result.json` | `6c510ea6e27b278514157ff8aaa5970d490237268b126f7085034c181011e673` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a1/deepseek-v4-pro-FC/live/BFCL_v4_live_relevance_result.json` | `d0f60aeba56f518df109c4c8878e1bdfa0122255a550a227717fe3a4fc94e501` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a1/deepseek-v4-pro-FC/live/BFCL_v4_live_simple_result.json` | `b8dfc87d8810bea21b53b533c755005ada850c7e931579bfeac6b13f062cb206` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a1/deepseek-v4-pro-FC/multi_turn/BFCL_v4_multi_turn_base_result.json` | `d9b8d82b9c423324f504c820cd8810c9d6d16ba54fdd7bfd386c24aba64987bf` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a1/deepseek-v4-pro-FC/multi_turn/BFCL_v4_multi_turn_long_context_result.json` | `1cd1599c3aae3fd78a80e7a9a3554c1fe4053e0a3b2712d2bcc57c629d2338bb` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a1/deepseek-v4-pro-FC/multi_turn/BFCL_v4_multi_turn_miss_func_result.json` | `c66dfa7c4d911d69c379eecbc0721623fc3e112432d527cf14b9de16bb0f817f` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a1/deepseek-v4-pro-FC/multi_turn/BFCL_v4_multi_turn_miss_param_result.json` | `ad0631d3f09b85a13c77c4a64e67cb36cfd159f85ede4db3f2846eff3d84f811` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a1/deepseek-v4-pro-FC/non_live/BFCL_v4_irrelevance_result.json` | `68a907bcf870463db5f5851245bdf6a4d2c4fe9d1e674617a5aa47cd3cb70f03` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a1/deepseek-v4-pro-FC/non_live/BFCL_v4_multiple_result.json` | `c20e46aee6bb934474f8916630117041e141cf1c4ed4b3a9f43516e827cf6646` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a1/deepseek-v4-pro-FC/non_live/BFCL_v4_parallel_multiple_result.json` | `cd9f8f4e41548a78c148c9f287200c2b20600d173b69916ee884b968dfbc5b74` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a1/deepseek-v4-pro-FC/non_live/BFCL_v4_parallel_result.json` | `ea0ad9c06faab8aae2784366edef7dc8fb98e21121b4e981c8d90227a88f09d6` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a1/deepseek-v4-pro-FC/non_live/BFCL_v4_simple_java_result.json` | `ce1748962ddfea9ec267cb8eb2e83993abc84a2285cdad1e5281a981b283e774` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a1/deepseek-v4-pro-FC/non_live/BFCL_v4_simple_javascript_result.json` | `efa3fd4cf852a5ec386350a6b9dec145fbb7879faaa99329a8626e8f23a46889` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a1/deepseek-v4-pro-FC/non_live/BFCL_v4_simple_python_result.json` | `dbf9a1e4cb3ef4541737978a1cab757dba77449e19be1e84c44d8428cda5bb75` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a2/deepseek-v4-pro-FC/agentic/BFCL_v4_web_search_result.json` | `4103ede670619eec7b1789927370f354e2d5f792182a9842414ddb611c7512da` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a2/deepseek-v4-pro-FC/agentic/memory/kv/BFCL_v4_memory_kv_prereq_result.json` | `c0257a48d1efb5ad76630094d4d30d75fc34a800a9f66b5cb78ac5446f8f6f04` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a2/deepseek-v4-pro-FC/agentic/memory/kv/BFCL_v4_memory_kv_result.json` | `76bd8ff592a09b372a9bfe492b4a1c8a12c4f5e3673f55973872937a9e10c7c4` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a2/deepseek-v4-pro-FC/agentic/memory/rec_sum/BFCL_v4_memory_rec_sum_prereq_result.json` | `8d6145be7fa79fef14957925dccd9651407d2f1085e28d62e8f212df5255614d` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a2/deepseek-v4-pro-FC/agentic/memory/rec_sum/BFCL_v4_memory_rec_sum_result.json` | `9638d95a3be924d412f1099af9b6ed3d903b06044bd74e36d4b4e42e26b2598b` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a2/deepseek-v4-pro-FC/agentic/memory/vector/BFCL_v4_memory_vector_prereq_result.json` | `3c0eac68c1deee37dbb438260876eb0f2ff20c5cfefeade44fe1b4763df0498f` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a2/deepseek-v4-pro-FC/agentic/memory/vector/BFCL_v4_memory_vector_result.json` | `67cd7889d4a9843dfe60173fc260c281f3ff1749ed897436f531f6abca25f52d` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a2/deepseek-v4-pro-FC/live/BFCL_v4_live_irrelevance_result.json` | `53f4fdf045eef45c3c47f44cb5be021aab797f5749a2b3b045f2a152eaa98bc2` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a2/deepseek-v4-pro-FC/live/BFCL_v4_live_multiple_result.json` | `ff3fcf383f0403872dcd0fcb85d21beca0bdb5f2d3ac190009237914ec689595` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a2/deepseek-v4-pro-FC/live/BFCL_v4_live_parallel_multiple_result.json` | `14c752bf2055c5cc12bc7a67589a3a36bbd319b7c2af96cf57acf3ba6ec2b4a7` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a2/deepseek-v4-pro-FC/live/BFCL_v4_live_parallel_result.json` | `6099e439773062e45d53c81cd4fa10c544304db1ed603a569324449bf548a481` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a2/deepseek-v4-pro-FC/live/BFCL_v4_live_relevance_result.json` | `a7ba28402adb6071b6df0b8352732af7a9fb6995415d0d003392fe8388fb9eeb` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a2/deepseek-v4-pro-FC/live/BFCL_v4_live_simple_result.json` | `aaf41f913b11ebde490781272608e23ba4147ecb151d388d391d8629337794dd` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a2/deepseek-v4-pro-FC/multi_turn/BFCL_v4_multi_turn_base_result.json` | `7fb6fb0d15b9a99c8bda10daeb6f51117c493d352f62e36fad88b0f395b60d94` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a2/deepseek-v4-pro-FC/multi_turn/BFCL_v4_multi_turn_long_context_result.json` | `719ffbf348c0e9f5b5ac50c41605a8cbc91f2161b9fe20d5be2100417620ee12` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a2/deepseek-v4-pro-FC/multi_turn/BFCL_v4_multi_turn_miss_func_result.json` | `47a4b54b0b5260b34d1bd2b98a0ffe4488d080dde8698a72ee3c33f7e3f8f26b` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a2/deepseek-v4-pro-FC/multi_turn/BFCL_v4_multi_turn_miss_param_result.json` | `7240299af4b7a051581aa46cd59b241ebab1e7faee4515b5db352def08a2ab4e` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a2/deepseek-v4-pro-FC/non_live/BFCL_v4_irrelevance_result.json` | `d1d47aec812f41a3357e89979a795b61e9ba826d1afdff28b89849241bd9458e` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a2/deepseek-v4-pro-FC/non_live/BFCL_v4_multiple_result.json` | `47dfa9320e2e5ab3110277eda01cd3dcd5e60f26b21211bd5f84da4aad2aeb0d` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a2/deepseek-v4-pro-FC/non_live/BFCL_v4_parallel_multiple_result.json` | `579af5180fc490cbb7ddc3f8bf4cc848e08d57be5bed1f29c4dee935432df31c` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a2/deepseek-v4-pro-FC/non_live/BFCL_v4_parallel_result.json` | `858376a756fd08451c669a8c3a41280fd8ff017770d45b4599171ee78cb9b192` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a2/deepseek-v4-pro-FC/non_live/BFCL_v4_simple_java_result.json` | `fc59c231fe64777545242b875a332e23b4d17d6d40b75e81074ed1253a240d61` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a2/deepseek-v4-pro-FC/non_live/BFCL_v4_simple_javascript_result.json` | `5ac6de577e8a5055e2899cd21e7f194bf2435d26247ca2d5def652c9af616338` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a2/deepseek-v4-pro-FC/non_live/BFCL_v4_simple_python_result.json` | `ea4673d10a925754198c7064b57d5562622e5bbfb389fcfb790c2562e4ea7f81` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a3/deepseek-v4-pro-FC/agentic/BFCL_v4_web_search_result.json` | `72083e66663ee13177403b65dd3de94896d03979463eed667fa4a21f5bc0ec3f` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a3/deepseek-v4-pro-FC/agentic/memory/kv/BFCL_v4_memory_kv_prereq_result.json` | `cb10166fa6d40a58199c623db52491a30e39841176b250003caf4d368e94b85d` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a3/deepseek-v4-pro-FC/agentic/memory/kv/BFCL_v4_memory_kv_result.json` | `3f82dd3501e709979092f168c9fa684f174c778e0e515426d40c0de129060b40` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a3/deepseek-v4-pro-FC/agentic/memory/rec_sum/BFCL_v4_memory_rec_sum_prereq_result.json` | `3852cc375a4efd092b71a1cfa924b7694f332f3406cfcd9fb30488a9e0957ae9` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a3/deepseek-v4-pro-FC/agentic/memory/rec_sum/BFCL_v4_memory_rec_sum_result.json` | `41f8daf6f345551f9437131ad4392d6775928ccba6ab010841eeb05eb3eaa382` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a3/deepseek-v4-pro-FC/agentic/memory/vector/BFCL_v4_memory_vector_prereq_result.json` | `dd0675d92296e0e1cff56ac77edf4a49b9fb3ec62928b00343cdd093982eafb9` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a3/deepseek-v4-pro-FC/agentic/memory/vector/BFCL_v4_memory_vector_result.json` | `eb9144de18a4bfa4e33165841527273f5116cc0f75aaf839f252490c67d1821a` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a3/deepseek-v4-pro-FC/live/BFCL_v4_live_irrelevance_result.json` | `467393880d3ae835e647c356a90064ab505c03f452ae92277a216f07618dba37` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a3/deepseek-v4-pro-FC/live/BFCL_v4_live_multiple_result.json` | `e46666bb14ea586f6c09ba9c753ca1badd52ca7e93f0bc3ba45f23967a728cae` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a3/deepseek-v4-pro-FC/live/BFCL_v4_live_parallel_multiple_result.json` | `7dbbd70e8498f9244dc5fafbea32462caf30511e519001aee0ab2666d7fd062e` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a3/deepseek-v4-pro-FC/live/BFCL_v4_live_parallel_result.json` | `0caf3a778a13c7ef3c0101987d49647635e9faab5c370c58258059fd2509604d` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a3/deepseek-v4-pro-FC/live/BFCL_v4_live_relevance_result.json` | `f0bc2ec70d446c628cbd82ca63d0d860433537cb4cf6bff8c8378f9d77239598` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a3/deepseek-v4-pro-FC/live/BFCL_v4_live_simple_result.json` | `213ddd8891f9c9e9aa125a5d5a3ead318d346d53adf8978300c0ba25581f544c` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a3/deepseek-v4-pro-FC/multi_turn/BFCL_v4_multi_turn_base_result.json` | `bace5286eac075c981b70e0114f90c373e33109d25899f5ec884a72155f393d8` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a3/deepseek-v4-pro-FC/multi_turn/BFCL_v4_multi_turn_long_context_result.json` | `281201cadf136047a31c6e6b0dfda64800bd267b1d136e03f2ef476a740ed0ef` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a3/deepseek-v4-pro-FC/multi_turn/BFCL_v4_multi_turn_miss_func_result.json` | `12498d5172899a6d4d1227459dfe1905b5552594401530afbf5de8c9fa605de0` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a3/deepseek-v4-pro-FC/multi_turn/BFCL_v4_multi_turn_miss_param_result.json` | `addb123a3cad67fea76be9cd51dd9862d9a4ab337d4f9431df4f20a273517e41` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a3/deepseek-v4-pro-FC/non_live/BFCL_v4_irrelevance_result.json` | `77940985de76ec6dfac75db9ba10976b36d16f031980233c97cda12de743b496` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a3/deepseek-v4-pro-FC/non_live/BFCL_v4_multiple_result.json` | `9f569193b3ccc1b7ccd3fcdd547cc5dac3c82b23d46844265d423540e736c879` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a3/deepseek-v4-pro-FC/non_live/BFCL_v4_parallel_multiple_result.json` | `139062572f73049397229c5ce66af48c935f6cdf836268bbf15dbcfbca32be06` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a3/deepseek-v4-pro-FC/non_live/BFCL_v4_parallel_result.json` | `aab39c4d0c76c3f3412f93817179ab464024e3fb1640f3ff2ce99e53bc8c50cf` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a3/deepseek-v4-pro-FC/non_live/BFCL_v4_simple_java_result.json` | `9e763c4e87d2983e48f1772e574e43407e5a5e5c32b56625fa667b978b84c612` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a3/deepseek-v4-pro-FC/non_live/BFCL_v4_simple_javascript_result.json` | `d4e54e6c821626f9e90e6d71c826675a856441b26ab136c7cf45916f0292c78a` |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a3/deepseek-v4-pro-FC/non_live/BFCL_v4_simple_python_result.json` | `bc9270ff1f1cc5cc53670b93d3fabdd2ff291bf89aa37e455e9b700eca9bc276` |
| `results/analysis/bfcl_teacher_tokens.json` | `99930a60b18ff5d9072d0fdf9e87f1c439aeaf5986a2f9c938c5be9dcfd3a2f4` |
