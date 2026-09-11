# WebShop prompt v2 (draft, 2026-09-10) — pending user decision (option ①)

Problem: every teacher (GPT-5.4 none/low, gpt-5.6-luna, gpt-oss:120b) verifies 0–20% at
reward==1 with the current prompt, while the student base reaches 19.4% SR. The current
1-shot example is a synthetic trajectory with invented observations and no reasoning about
the goal attributes; the ReAct paper's WebShop prompt is a full real trajectory with Thought
lines that (a) restate the required attributes/options/price, (b) compare candidates against
them, (c) pick options explicitly before Buy Now. WebShop's reward is 1 only when attributes,
options and price all match, so demonstrating the check is what the prompt is missing.

Proposed changes (protocol re-freeze; student base must be re-evaluated, ~45 min):
1. Replace the synthetic example by the ReAct-style real trajectory (below), keeping the
   Observation/Action format the evaluator already parses.
2. Add explicit rules: verify every attribute and option before Buy Now; if the item page
   lacks a required option or price is over the limit, click[< Prev] / click[Back to Search]
   and try another candidate; never buy on the search page.
3. Keep the Thought: line optional (student rendering unchanged), OBS_CHARS=6000, 15 steps.

## Example (ReAct WebShop prompt, adapted to our action grammar)
Observation: WebShop [SEP] Instruction: i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars [SEP] Search
Thought: I need 3 ounce, bright citrus, for sensitive skin, and under $50.
Action: search[3 ounce bright citrus deodorant sensitive skin]
Observation: [Back to Search] [SEP] Page 1 (Total results: 50) [SEP] [Next >] [SEP] B078GWRC1J [SEP] Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce [SEP] $10.99 [SEP] B078GTKVXY [SEP] Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce [SEP] $10.99 [SEP] B08KBVJ4XN [SEP] Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) [SEP] $15.95
Thought: B078GWRC1J is bright citrus, 3 ounce, sensitive skin, $10.99 < $50. It matches everything; I will check its options.
Action: click[B078GWRC1J]
Observation: [Back to Search] [SEP] [< Prev] [SEP] scent [SEP] assorted scents [SEP] bright citrus [SEP] calming lavender [SEP] ginger fresh [SEP] simply non-scents [SEP] size [SEP] travel set (4-pack) [SEP] 3 ounce (pack of 1) [SEP] 3-ounce (2-pack) [SEP] Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce [SEP] Price: $10.99 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now
Thought: The instruction needs scent bright citrus and size 3 ounce; both options exist, so I select them before buying.
Action: click[bright citrus]
Observation: You have clicked bright citrus.
Action: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1).
Thought: Scent, size and price all match the instruction. Buy.
Action: click[Buy Now]

Cost estimate for re-freeze: student base 500 sessions ≈ 45 min on GPU2; teacher re-probe
5 tasks × 3 attempts ≈ 10 min on ollama (mistral/gpt-oss) or Azure after quota reset.
