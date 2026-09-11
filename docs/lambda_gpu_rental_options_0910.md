# Lambda GPU rental for the remaining campaign (2026-09-10)

## Workload to cover (hpg suspended, rai rarely has a >=66 GB card free)
- Student gemma-4-12B-it, LoRA r16 trainables, memory budget 60 GB, rollouts through vLLM
  (12B bf16 ~24 GB + KV). One 80 GB card per arm is enough; multi-GPU is only for parallel arms.
- Runs left: unified P1 on ALFWorld (smoke + V0 + D0/D1/D2/D3 + 3 variants ~ 10 runs), the same
  on WebShop and BFCL (~20 runs), Table 1 baselines for the new pair (6 methods x 3 benchmarks
  = 18 runs), evals. Reference: v1.1 V2 (Qwen 4B, BFCL) reached round 2 step 10 after 22 h on a
  B200; the 12B student is slower per step but P1 arms replay a fixed schedule.
  Estimate 12-24 h per P1 arm on H100, 3-5 h per baseline run.
  => ~500-700 GPU-hours total (wide range: 350-1,000).

## Lambda on-demand prices (per GPU-hour, per-minute billing, no spot, no egress fees)
| instance | VRAM | $/GPU-h | 600 GPU-h | note |
|---|---|---|---|---|
| GH200 | 96 GB | 2.29 | ~$1,400 | cheapest H100-class; ARM CPU (Grace): torch/vLLM aarch64 wheels exist, but the py3.8 WebShop env (pyserini/Java) and BFCL harness need re-validation |
| H100 PCIe | 80 GB | 3.29 | ~$2,000 | safest x86 choice |
| H100 SXM (1x/8x) | 80 GB | 3.99 | ~$2,400 | 8x node = all P1 arms of one benchmark in parallel, ~3 days ~$2,300 |
| A100 80 GB | 80 GB | 2.79 | ~$2,500 (x1.5 slower) | no saving vs H100 |
| A100 40 GB | 40 GB | 1.99 | n/a | too small for 12B training + vLLM |
| B200 | 180 GB | 6.69 | ~$2,000 (x2 faster) | needs Blackwell stack; no real saving |
Clusters (2 weeks-1 year) are $5.54-6.16 per H100-hour: worse than on-demand for us.
Storage: persistent filesystem billed per GiB-month (rate shown at creation; typically ~$0.20),
keeps models/envs/checkpoints across instances; ~250 GB => ~$50/month. Idle instances are
billed while running: 1x H100 left on overnight = ~$80-96/day, so auto-terminate when idle.

## Recommendation
1. Cheapest sound plan: 1x H100 PCIe on demand (~$3.3/h), persistent filesystem, arms run
   back to back with an auto-shutdown watcher; budget ~$2,000 +-50% for everything left,
   ~$700 for the ALFWorld P1 block alone.
2. If speed matters more than money: an 8x H100 SXM node for ~3 days per benchmark block
   (same $/GPU-h, 8 arms in parallel, ~$2,300 per block).
3. Try GH200 first for one arm if the stack installs (saves ~30%); fall back to H100 PCIe.
4. Skip A100 (no saving), B200 (no saving, stack risk), reserved clusters (more expensive).
5. Ask Lambda for research credits (lambda.ai/research) before paying list price.

Sources: https://lambda.ai/pricing , https://lambda.ai/service/gpu-cloud ,
https://docs.lambda.ai/public-cloud/billing/ , https://docs.lambda.ai/public-cloud/filesystems/

# gpt-5.6-luna: where is it cheapest?
| channel | input / cached / output per 1M | notes |
|---|---|---|
| OpenAI official, Standard | 0.20 / 0.02 / 1.20 | prompt caching cuts repeated prefixes 10x |
| OpenAI official, Flex or Batch | 0.10 / 0.01 / 0.60 | 50% off; Flex is synchronous (slower) so it works for interactive episodes; Batch does not |
| OpenRouter | 0.20 / 0.02 / 1.20 + 5.5% on credits | pass-through, no Flex/Batch discount |
| Azure APIM (ours) | list price | weekly quota exhausted (403) |
For the WebShop pool (~13.5M input, 0.5M output) luna costs ~$3 standard, ~$1 with Flex + caching;
GPT-5.4 costs ~$41 standard, ~$15-20 with Flex + caching. Cheapest = official OpenAI API with
Flex service tier and prompt caching; OpenRouter is +5.5% and only wins if we cannot open an
OpenAI billing account. Source: https://developers.openai.com/api/docs/pricing
