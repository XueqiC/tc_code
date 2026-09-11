# GPU rental beyond Lambda (2026-09-11)

Need: one 80-96 GB card per arm (gemma-4-12B LoRA, 60 GB frozen budget), runs of 3-24 h,
~500-700 GPU-h left. Lambda reference: H100 PCIe $3.29/h, GH200 $2.29/h (ARM).

## Paid, per-second billing
| provider / SKU | $/h | 600 GPU-h | notes |
|---|---|---|---|
| RunPod RTX PRO 6000 96 GB, Secure | 2.09 | ~$1,250 | same Blackwell card as rai GPU1/GPU4: our vLLM/transformers stack already runs on it; 96 GB = full 60 GB budget with headroom. Best fit. |
| RunPod RTX PRO 6000 96 GB, Community | 1.69 | ~$1,000 | community hosts, slightly less reliable |
| RunPod A100 80 GB, Secure / Community | 1.59 / 1.39 | ~$1,400 / $1,250 (x1.5 slower) | fine for LoRA 12B; cheapest per run after RTX PRO 6000 |
| RunPod H100 PCIe, Secure / Community | 2.89 / 1.99 | ~$1,700 / $1,200 | |
| RunPod H100 SXM, Secure / Community | 3.49 / 2.69 | ~$2,100 / $1,600 | |
| Vast.ai marketplace H100 | ~1.5-2.5 | ~$900-1,500 | peer hosts, per-second, interruptible 50%+ cheaper; reliability varies by host, check host rating and verified status |
| Lambda H100 PCIe | 3.29 | ~$2,000 | reference |
| Nebius / Crusoe / CoreWeave | 3-5.5 | ~$2,000-3,300 | enterprise tier, no benefit for us |
| AWS / Azure / GCP on demand | 5-7 | ~$3,000-4,200 | only with research credits |
RunPod storage: network volume $0.07/GB-month (<1 TB), keeps envs/models between pods; billing per second; pods can be stopped (volume persists, GPU released).

## Free / academic (worth applying in parallel, 1-2 weeks turnaround)
- NSF ACCESS "Explore" allocation: graduate student + advisor co-PI, project description only,
  processed continuously; Delta (A100 40/80 GB, H200 141 GB) and DeltaAI (GH200 96 GB, ARM).
  allocations.access-ci.org. Enough for the whole remaining campaign at zero cost.
- NAIRR Pilot (nairrpilot.org): similar, with cloud credits from Azure/AWS/GCP partners.
- Cloud research credits: Google Cloud Research Credits, AWS Cloud Credit for Research, Azure
  for Research; typically $1k-5k, weeks of paperwork.
- FSU RCC (campus HPC) if the advisor has an allocation: check for A100/H100 partitions.

## Recommendation
1. Now: RunPod Secure Cloud, one RTX PRO 6000 96 GB pod ($2.09/h) with a 300 GB network
   volume; identical GPU family to rai, so no stack re-validation; ~$1,250 for everything left,
   ~$450 for the ALFWorld P1 block. Run arms back to back; stop the pod when idle (volume stays).
   If 96 GB pods are unavailable in the region, take A100 80 GB Secure ($1.59/h).
2. In parallel: file an ACCESS Explore request (advisor co-PI) and the NAIRR form; if it lands,
   move the WebShop/BFCL blocks and Table 1 baselines there for free.
3. Avoid: hyperscalers at list price, Vast.ai for the long P1 runs (interruptions), B200/H200
   (no gain for a 12B LoRA student).

Sources: https://www.runpod.io/pricing , https://vast.ai/pricing , https://lambda.ai/pricing ,
https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison ,
https://www.spheron.network/blog/gpu-cloud-pricing-comparison-2026/ ,
https://delta.ncsa.illinois.edu/delta-allocations/ , https://support.access-ci.org/documentation/resources/delta-gpu ,
https://grantedai.com/blog/ai-compute-grants-gpu-credits-guide
