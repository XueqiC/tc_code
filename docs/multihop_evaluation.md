# Multi-hop construction and evaluation report

The current table replaces BFCL with Bamboogle, MuSiQue, and 2WikiMultiHopQA:
ALFWorld / HotpotQA / Bamboogle / MuSiQue / 2Wiki. SmartAD, Agent Distillation
(Kang), and Structured Agent Distillation all evaluate on HotpotQA; SmartAD and
Kang also use these three OOD sets. This does not claim that Structured Agent
Distillation originally reported all three OOD sets.

This delivery covers construction and CPU validation. No training, teacher
purchases, model evaluations, external API calls, cluster sync, or commits were
performed. Inventory checks and test counts are recorded in
[multihop_construction_report.json](multihop_construction_report.json).

## One execution protocol

`tools/hotpotqa_eval.py --dataset {hotpotqa,2wiki,musique,bamboogle}` uses the
existing `bfas.hotpotqa.run_episode` / `episode_stream`, `build_messages`,
`student_generator`, and `Wikipedia`. There is no second episode loop,
dataset retriever, or prompt. All four QA datasets use the unchanged six-shot
ReAct prompt (`react6-e52ba17b32d9`), **7 steps, 100 completion tokens,
temperature 0**, and the same thought/action fallback and stop strings.
Existing endpoint-specific decoding is inherited unchanged; compare models using
the same endpoint class and preserved Wikipedia cache.

Live Wikipedia remains the default. `--offline` explicitly replays cached
Wikipedia queries; a cache miss fails the campaign without becoming a scored
failure. That flag controls retrieval, not model-endpoint connectivity. This
live-Wikipedia setup differs from the baselines' fixed Wikipedia/dense-retrieval
setting, so absolute scores are not directly comparable across papers.

The loader passes only ID, question, type, and scoring fields to the harness.
**Only `question` is rendered into prompts.** Source supporting sentences,
paragraphs, distractors, aliases, answerability flags, gold answers, `evidences`,
and decomposition answers are never appended to prompts or used to build tool
queries. Independent Wikipedia observations or model responses may naturally
contain the same answer text. Audit data is kept separate and attached only
after the episode and scoring finish; it cannot steer generation or retrieval.

## Local inventory and normalization

| Dataset / CLI name | File in `envs/multihop/data` | Rows | Normalization | Audit level |
|---|---|---:|---|---|
| 2WikiMultiHopQA / `2wiki` | `2wiki_dev.json` | 12,576 | Preserve `_id`, question, answer, type | Supporting titles and sentence indices |
| MuSiQue / `musique` | `musique_validation.json` | 2,417 | `id` → `_id`; preserve aliases and `answerable`; type `multihop` | Supporting paragraphs and original paragraph indices |
| Bamboogle / `bamboogle` | `bamboogle_test.json` | 125 | `Question`/`Answer` → question/answer; type `multihop` | None |

MuSiQue is JSON Lines despite its `.json` extension; the loader accepts both JSON
arrays and JSON Lines. Bamboogle has no native IDs: `bamboogle-000000` means source
row 0. Full source-file, ordered source-ID, and normalized scoring-row SHA-256
values bind this numbering and the frozen selected order. Duplicate IDs, source
content/order drift, manifest drift, invalid scoring fields, and unexpected row
counts fail instead of filtering or replacing rows. The preparation command
never downloads, removes, or rewrites source data. `musique_train.json` is unused.

The local MuSiQue file contains **2,417 answerable rows, 0 unanswerable rows, and
680 rows with nonempty aliases**. CPU fixtures also cover unanswerable items.

## Selection in the HotpotQA check's reporting form

The existing structural check records: “First n IDs, in stored order, from
configs/hotpotqa_support_split.json. That train inventory is
random.Random(0).sample(source_ids, 200). No filtering, replacement, or selection
by student outcomes.” The new manifests use that same two-stage rule, with the
entire evaluation inventory sampled once to allow extensions to the full split:

- **2Wiki:** First n IDs, in stored order, from configs/multihop_2wiki_eval_split.json. That dev inventory is random.Random(0).sample(source_ids, 12576). No filtering, replacement, or selection by student outcomes.
- **MuSiQue:** First n IDs, in stored order, from configs/multihop_musique_eval_split.json. That validation inventory is random.Random(0).sample(source_ids, 2417). No filtering, replacement, or selection by student outcomes.
- **Bamboogle:** First n IDs, in stored order, from configs/multihop_bamboogle_eval_split.json. That test inventory is random.Random(0).sample(source_ids, 125). No filtering, replacement, or selection by student outcomes.

A full-size sample without replacement freezes a random permutation. Its size
is independent of `n`; runtime selection is `manifest.ids[:n]`. Every larger
`n` is a strict prefix extension of a smaller one. The report/config `selection`
includes the rule, seed, source/sample counts, checksums, ordered task IDs, and
source/selected answerability counts. Excluded rows: **0**. No selection uses
student correctness, aliases, answerability, type, or annotation availability.

OOD evaluation rejects nonzero `--start`, zero `n`, and out-of-range `n`.
Omitting `--n` evaluates the full frozen inventory. A larger prefix requires a
new output directory; saved identities cannot be rebound. The existing official
HotpotQA **dev evaluation** retains its original first 500 distractor-dev rows;
it is distinct from the train structural check above. Neither existing HotpotQA
manifest is changed.

## Answers, aliases, and the judge seam

For answerable MuSiQue rows, compute the existing HotpotQA answer EM and token
F1 against the canonical answer and **every alias**, recording the maximum EM
and maximum F1 independently. The canonical answer and aliases stay in the
record for reproduction. HotpotQA, 2Wiki, and Bamboogle keep single-reference
EM/F1 behavior.

For `answerable=false`, **retain the row** and its original answer/aliases for
provenance, but score explicit abstention instead of the factual source answer.
After the same case/punctuation/article/whitespace normalization, the **entire**
`finish[...]` answer must equal one of:

`noanswer`; `unanswerable`; `cannot be answered`; `this question cannot be answered`;
`i cannot answer this question`; `i cannot determine the answer`;
`insufficient information`; `not enough information to answer`.

Accepted statements receive EM=F1=1; everything else receives EM=F1=0, without
partial F1 for shared refusal words. Empty answers, unfinished episodes, factual
guesses, and abstentions with an added answer score zero. This finite policy is
versioned; it is not a semantic refusal judge. No per-row hint changes the common
prompt. MuSiQue's flag concerns its provided evidence, not global unanswerability
from live Wikipedia. That limitation would matter for a future mixed inventory;
the frozen local split has no such rows.

**SmartAD and Kang score factual answers with an LLM judge. Our recorded EM/F1
numbers therefore will not match their judged scores.** The observed prediction
`2240 feet` against gold `2240` remains EM=0, F1=2/3, despite possible semantic
correctness; see the [earlier failure analysis](2026-09-13-route1-route2-analysis-zh.md).
Normalization is not silently loosened to hide this mismatch.

`bfas.hotpotqa.score_answer(prediction, question)` is the post-generation scoring
seam, with targets from `reference_answers`. Its version,
`hotpotqa-em-f1-alias-abstention-v1`, is bound into evaluation identity. A later
versioned judge can consume saved question/prediction/reference records and add
a separate metric while preserving EM/F1, selection, prompt, decoding, and
retrieval. No judge/client, judge dependency, or external API call is added now.

## Meaningful audit fields by dataset

The audit adapts the existing HotpotQA check's `tools/hotpotqa_step_audit.py`
from the `mech-hotpotqa` worktree. Version 2 preserves the
`unicode-word-sequence-casefold-v1` matching rule: contiguous casefolded Unicode
words, ignoring inter-word punctuation/whitespace, with half-open character
offsets into the actual observation. There is no stemming, embedding, semantic
matching, or evidence alias expansion.

Every step reports `annotation_status`, `annotation_level`, and
`annotation_complete`. The episode catalog is `supporting_facts_audit`;
`metrics.json` reports `annotation_status_counts`, and config records
`step_audit.field_applicability`.

| Field | HotpotQA / 2Wiki | MuSiQue | Bamboogle |
|---|---|---|---|
| `annotation_status` | `available` when sentence text exists | `coarser_than_sentence_level` when paragraph text exists | `unavailable` |
| `supporting_fact_matches` (sentence indices/offsets) | Meaningful | `null` | `null` |
| `all_supporting_facts_seen_before_step` | Sentence-catalog lexical exposure | `null` | `null` |
| `supporting_paragraph_matches` (paragraph indices/offsets) | `null` | Whole-paragraph lexical matches | `null` |
| `all_supporting_paragraphs_seen_before_step` | `null` | Paragraph-catalog lexical exposure | `null` |
| Query/previous-observation shared strings and count | Meaningful | Meaningful | Meaningful |
| `query_previous_observation_overlap.supporting_title_reuse` | Annotated titles | Supporting paragraph titles | `null` |
| Step/episode end reasons | Meaningful | Meaningful | Meaningful |

“All seen” measures exposure **before** that step, from earlier successful
retrieval observations. Missing/empty catalogs or missing unit text produce
unknown completeness (`null`), never zero coverage or vacuous success. Partial
catalogs mark missing units unavailable and `annotation_complete=false`.
MuSiQue paragraphs are never split into invented sentence annotations; a fragment
cannot certify whole-paragraph exposure. These are lexical diagnostics, not
semantic evidence sufficiency or proof that an annotated retrieval route was
necessary. Alternative legal paths can still receive full answer scores.

## Commands and table integration

Verify/reconstruct the same frozen manifests from existing local files; reruns
are idempotent and refuse replacement of changed inventories:

```bash
.venv/bin/python tools/multihop_prepare.py
```

Evaluation uses an **already served** checkpoint. These commands were not run:

```bash
.venv/bin/python tools/hotpotqa_eval.py \
  --dataset musique --base-url http://localhost:8900/v1 --model bfas-policy \
  --n 200 --out results/paper_baselines/table1_musique_smartad

.venv/bin/python tools/hotpotqa_eval.py \
  --dataset bamboogle --base-url http://localhost:8900/v1 --model bfas-policy \
  --out results/paper_baselines/table1_bamboogle_smartad
```

Use matching dataset/method/seed directory names for each checkpoint. No OOD
training/purchase registration is added. `tools/table1_aggregate.py` now reads
five benchmarks (45 potential method/seed evaluations), excludes archived BFCL,
and records secondary F1 fractions for all QA columns. Standalone evaluator
records with `config.dataset` use 100×EM as the primary percentage. Missing
training spend remains null and is excluded from receipt counts; original
training receipts belong to the source checkpoint's training run.

New initial-student scores default to null. The table prints pending cells and
withholds average gain until all benchmarks have both scores and references.
Supply `--initial-bamboogle`, `--initial-musique`, and `--initial-2wiki` only after
measuring those references with the same protocol and prefix. The LaTeX body
requires seven columns (method, five benchmarks, average gain). The consuming
paper itself is outside this change.

Tests use temporary synthetic inventories and stub model/Wikipedia responses;
network and real model/teacher entry points are blocked. Coverage includes
normalization, aliases, unanswerable rows, annotation granularity/status,
prefix extensions, inventory integrity, prompt/decoding identity, seven-step
limits, fallback requests, prompt leakage, resume identity, and table ingestion.
The companion JSON records the final test command and per-file counts.
