# Teacher provider options (2026-09-10): Lambda vs OpenRouter

## Our needs
- Teacher = strong, non-Chinese, text-only black box, per-token billed (the paper's setting).
- Native function calling for BFCL (tools / tool_calls); plain text for ALFWorld and WebShop.
- Volume: ALFWorld pool already bought with GPT-5.4 on Azure (219 calls, 190k completion tokens).
  Remaining: WebShop pool 200 tasks x 3 attempts (~450 attempts x ~10 steps x ~3k prompt tokens
  = ~13.5M input, ~0.5M output); BFCL demand 40 tasks x 3 attempts (~1M input incl. memory chains).
  Option (2) (6-8 attempts) roughly doubles WebShop.
- Consistency: ALFWorld bank is GPT-5.4; keeping GPT-5.4 for WebShop/BFCL avoids a two-teacher paper.

## Lambda (lambda.ai/inference)
- The Lambda Inference API is winding down ("As the Inference API winds down, you can continue
  deploying ... on NVIDIA GPU instances"); docs page redirects; no shutdown date found.
- Only open-weight models (Llama / DeepSeek / Qwen / Hermes class) — same class as ollama.com,
  which we already have and where gpt-oss:120b and mistral-large-3 gave 0/5 on WebShop.
- Verdict: not recommended.

## OpenRouter (openrouter.ai)
- Aggregator, OpenAI-compatible endpoint https://openrouter.ai/api/v1, model ids like
  openai/gpt-5.4; provider pricing passed through with no inference markup; 5.5% fee on credit
  purchase (Stripe); prompts/completions not logged by default; no weekly quota, no daily caps
  on paid models; provider 429s are retried across providers automatically.
- Tool calling (tools / tool_choice) is supported for GPT-5.x, Claude, Gemini, gpt-oss.
- Prices (input / output per 1M tokens):
  | model | in | out | est. WebShop pool | note |
  |---|---|---|---|---|
  | openai/gpt-5.4 | 2.50 | 15 | ~$41 (~$80 with 6-8 attempts) | same teacher as ALFWorld bank |
  | openai/gpt-5.5 | 5 | 30 | ~$80 | stronger, not needed |
  | openai/gpt-5.6-sol | 2 (promo) | 10 | ~$32 | |
  | openai/gpt-5.6-luna | 0.20 | 1.20 | ~$3 | probed 1/5 on WebShop; good for prompt iteration |
  | anthropic/claude-sonnet-5 | 2 | 10 | ~$32 | alternate frontier teacher |
  | anthropic/claude-opus-5 | 5 | 25 | ~$80 | |
  | google/gemini-3.1-pro-preview | 2 | 12 | ~$32 | alternate frontier teacher |
  | google/gemini-3.8-flash | 0.75 | 3.75 | ~$12 | cheap tier |
  | openai/gpt-oss-120b | 0.03 | 0.17 | ~$0.5 | already 0/5 via ollama |
  | meta-llama/llama-4-maverick | 0.20 | 0.70 | ~$3 | open weights |
  BFCL adds < $5 with GPT-5.4. Prompt caching (where the provider supports it) lowers input cost.

## Recommended plans
- A (recommended): OpenRouter + openai/gpt-5.4. Teacher unchanged across all three benchmarks,
  ~$50-100 total for WebShop + BFCL, no weekly quota. Prepay $100.
- B: A plus one alternate frontier teacher (claude-sonnet-5 or gemini-3.1-pro) for a
  teacher-robustness appendix, ~$35 each.
- C: cheap iteration tier (gpt-5.6-luna, gemini-3.8-flash) for prompt v2 probes before spending
  GPT-5.4 tokens; a few dollars.
- D: Lambda — no.

## What I need from the user
- An OpenRouter account with credits; put the key in ~/hq/secrets/llm_apis.env as
  OPENROUTER_API_KEY (never in Discord). Code change: register openrouter/<model> entries in the
  BFCL handler (same OpenAI-compatible path as the ollama entries) and point the WebShop/ALFWorld
  teacher client at the OpenRouter base URL (small Codex task).

Sources: https://lambda.ai/inference , https://openrouter.ai/openai , https://openrouter.ai/anthropic ,
https://openrouter.ai/google , https://openrouter.ai/meta-llama , https://openrouter.ai/docs/faq ,
https://openrouter.ai/docs/api-reference/limits , https://openrouter.ai/openai/gpt-5.4
