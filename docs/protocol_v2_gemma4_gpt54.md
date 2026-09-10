# Protocol v2 (2026-09-10): student gemma-4-12B-it, teacher GPT-5.4, benchmarks ALFWorld / WebShop / BFCL v4

Decided by the user on 2026-09-10 after the Qwen/DeepSeek stop and the hpg suspension.
AppWorld is dropped (simplified harness floors every base at 0/40; the official scaffold is
costly, and gemma-4-12B-it degenerates on it).

## Student and teacher
- Student: `google/gemma-4-12B-it` (dense 11.95B, Apache-2.0, native tool calling).
  Base scores: ALFWorld valid_seen 56.43, WebShop SR 19.4 / score 54.8 (obs 6000; 18.2 / 53.7 at obs 2500), BFCL v4 45.63.
- Teacher: GPT-5.4 on Azure OpenAI (credentials: ~/hq/secrets/llm_apis.env, exported as
  AZURE_LLM_ENDPOINT / AZURE_LLM_KEY). Every teacher output token is charged, failures included.

## Evaluation protocol (frozen)
| Benchmark | Evaluation set | Steps | Decoding | Metric |
|---|---|---|---|---|
| ALFWorld | valid_seen, 140 games (bfas adapter, ReAct scaffold, vLLM) | 40 | greedy | success rate |
| WebShop | test sessions 0–499, full 1.18M-product index, ReAct 1-shot prompt of tools/webshop_eval.py (obs 6000 chars for the current page, history 600 chars, prompt ≤60k chars; 2500 was tried first and showed only ~3 of 10 results per page) | 15 | greedy (T=0, max_tokens 128) | success rate (reward = 1); score (100·mean reward) as diagnostic |
| BFCL v4 | full 5,217 entries, official scripts, Gemma 4 native FC handler (gemma4_fc) | harness | official default T=0.001 (vLLM) | overall accuracy |

Web-search categories have no search backend (SERPAPI unset) for every model; treat them as
environment-consistent, tool-less items.

## Support sets
- ALFWorld: unchanged (N=178, S_d 142, S_c 36); reuse the GPT-5.4 pool (219 calls, 107 verified,
  189,541 tokens).
- WebShop: pool = train sessions 500–6909 (6,410); N = clip(0.05·6410, 50, 250) = 250,
  S_d 200, S_c 50; teacher demos = GPT-5.4 playing the same ReAct prompt, verified iff reward = 1
  within 15 steps, up to three attempts (greedy, then two at T=0.7).
- BFCL: unchanged split (N=235, S_d 188, S_c 47); teacher demos re-collected with GPT-5.4 FC
  through the Azure handler.

## Budget plan (teacher output tokens)
- BFCL re-collection: < 0.1M. WebShop: 1–2M (200 tasks × ≤3 attempts × ~2–5k). ALFWorld: 0.
- Hard stop and ask the user if the plan would exceed 2M tokens.

## Student rendering
- BFCL rows: prompt = Gemma 4 chat template with tools, add_generation_prompt (ends with the
  empty thought channel); target = native `<|tool_call>call:NAME{…}<tool_call|>` text as the
  template renders it. ALFWorld / WebShop rows: chat template, plain-text action targets.
- Deployment-exact rule: training prompts are byte-identical to what the evaluator sends.
