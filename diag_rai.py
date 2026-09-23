import json, glob, sys
from math import comb
root = "/home/xueqi/hq/projects/tc-alignment-vllm"
def load(tag):
    d = sorted(glob.glob(f"{root}/runs/{tag}-*/campaigns/{tag}/artifacts/tasks"))[-1]
    out = {}
    for f in glob.glob(d + "/*.json"):
        r = json.load(open(f))["record"]; out[r["task_id"]] = r
    return out
def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); return min(1.0, sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2)
base_tag, tags = sys.argv[1], sys.argv[2:]
base = load(base_tag); nb = sum(r["success"] for r in base.values())
print(f"{base_tag}: {nb}/{len(base)} = {nb/len(base)*100:.2f}%")
types = sorted({t.split("-")[0] for t in base})
for t in tags:
    d = load(t); n = sum(r["success"] for r in d.values())
    b = sum(base[i]["success"] and not d[i]["success"] for i in base); c = sum(d[i]["success"] and not base[i]["success"] for i in base)
    caps = sum(r["horizon_reached"] for r in d.values()); fb = sum(1 for r in d.values() for x in r["turns"] if x.get("parser_fallback")); nt = sum(len(r["turns"]) for r in d.values())
    print(f"{t:14s} {n}/{len(d)}  {(n-nb)/len(d)*100:+.2f}  disc {b+c} (b{b}/c{c})  p={mcnemar(b,c):.3f} | fallback {fb/max(nt,1)*100:.1f}% caps {caps}")
    print("   per type: " + "  ".join(f"{ty.split('_')[1] if '_' in ty else ty}:{sum(d[i]['success'] for i in base if i.startswith(ty))}/{sum(base[i]['success'] for i in base if i.startswith(ty))}" for ty in types))
