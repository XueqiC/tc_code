"""Historical, replayable BFCL condition contrasts. No teacher API is used.

Candidate IDs identify content, not the reused generator IDs. The generate
phase uses a base-only vLLM completions server; score uses the native CE path
in behavior.margin after the server has been stopped to release GPU memory.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import re
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
BASE_MODEL = "Qwen/Qwen3.5-4B"
LEDGER = "results/analysis/bfcl_teacher_tokens.json"
TYPES = ("binding", "multi_turn", "call_vs_abstain")
DEFAULT_TYPE3_CAP = 0.3333  # Historical exact one third, not a stricter decimal cap.
LEAKAGE_REASONS = ("prompt_hash", "content_hash", "unique_id", "official_id", "parent")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def parent_id(task_id):
    task_id = task_id.split("#", 1)[0]
    if task_id.startswith(("gen_", "oos_", "genmt_")):
        return task_id.split("_", 1)[1].rsplit("_", 1)[0]
    return task_id


def render_prompt(question, functions):
    # This is the actual training/mining/evaluation renderer, not a template copy.
    from bfas.adapters.bfcl import BFCLAdapter
    try:
        return BFCLAdapter()._render(question[0], functions)
    except ImportError:
        # On HPG the training and BFCL dependency environments are separate,
        # just as for CheckerBridge. Render in that venv without a GPU/model.
        python = os.environ.get("BFCL_VENV_PYTHON", str(ROOT / "envs/bfcl/.venv/bin/python"))
        code = ("import json,sys; from contextlib import redirect_stdout; "
                "from bfas.adapters.bfcl import BFCLAdapter; q,f=json.load(sys.stdin); "
                "\nwith redirect_stdout(sys.stderr): p=BFCLAdapter()._render(q[0],f)"
                "\nprint(json.dumps(p))")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(ROOT), os.environ.get("PYTHONPATH", "")]))
        result = subprocess.run([python, "-c", code], input=json.dumps([question, functions]),
                                text=True, stdout=subprocess.PIPE, check=True, timeout=120, cwd=ROOT, env=env)
        return json.loads(result.stdout)


def thinking_off(prompt):
    from tools.behavior_atom.gpu_driver import _thinking_off
    return _thinking_off(prompt)


def parse_calls(text):
    """Same tool-call JSON representation as the miner, but reject broken tags.

    In particular malformed calls must never be credited as abstentions.
    """
    blobs = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S)
    if text.count("<tool_call>") != len(blobs) or text.count("</tool_call>") != len(blobs):
        raise ValueError("unbalanced tool_call tags")
    calls = []
    for blob in blobs:
        call = json.loads(blob)
        if not isinstance(call, dict) or not isinstance(call.get("name"), str) or not call["name"]:
            raise ValueError("tool call needs a function name")
        args = call.get("arguments", {})
        if not isinstance(args, dict):
            raise ValueError("tool call arguments must be an object")
        calls.append({call["name"]: args})
    return calls


def normalize_truth(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = ast.literal_eval(value)
    if isinstance(value, dict):
        value = value.get("ground_truth", value)
    if not isinstance(value, list) or any(not isinstance(c, dict) for c in value):
        raise ValueError("ground_truth must be a BFCL checker list")
    return value


def typed_value(value, schema, category="simple_python"):
    """Represent a supplied value, never infer defaults or round numbers.

    BFCL's Java/JavaScript AST boundary takes source literals as strings even
    for numeric schemas (its misleading error calls that wire type String).
    """
    declared = schema.get("type", "any")
    kind = declared.lower()
    javascript = "javascript" in category
    java = "java" in category and not javascript
    if kind in {"string", "char"}:
        value = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, allow_nan=False)
    elif kind in {"integer", "byte", "short", "long", "bigint", "float", "double", "number"}:
        if type(value) is bool:
            raise ValueError("boolean is not a numeric argument")
        try:
            number = Decimal(str(value))
        except InvalidOperation as exc:
            raise ValueError(f"cannot render {value!r} as {declared}") from exc
        if not number.is_finite():
            raise ValueError("nonfinite argument")
        if kind in {"float", "double", "number"}:
            value = float(number)
            if not math.isfinite(value) or Decimal(str(value)) != number:
                raise ValueError("float conversion would change the accepted value")
        else:
            if number != number.to_integral_value():
                raise ValueError("integer conversion would change the accepted value")
            value = int(number)
    elif kind in {"boolean", "bool"}:
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            value = value.lower() == "true"
        if type(value) is not bool:
            raise ValueError(f"cannot render {value!r} as boolean")
    elif kind in {"array", "arraylist", "tuple", "list", "dict", "object", "hashmap"}:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = ast.literal_eval(value)
        if kind in {"array", "arraylist", "tuple", "list"}:
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"cannot render {value!r} as array")
            value = [typed_value(v, schema.get("items", {})) for v in value]
        else:
            if not isinstance(value, dict):
                raise ValueError(f"cannot render {value!r} as dict")
            properties = schema.get("properties", {})
            value = {k: typed_value(v, properties.get(k, {})) for k, v in value.items()}
    elif kind != "any":
        raise ValueError(f"unsupported argument type: {declared}")
    if not (java or javascript):
        return value
    if isinstance(value, str):
        return value
    if java and isinstance(value, list):
        item = schema.get("items", {})
        literals = [json.dumps(v, ensure_ascii=False) if isinstance(v, str)
                    else typed_value(v, item, category) for v in value]
        if kind == "arraylist":
            return ("new ArrayList<>()" if not literals else
                    "new ArrayList<>(Arrays.asList(" + ", ".join(literals) + "))")
        return "new " + {"integer": "int"}.get(item.get("type"), item.get("type", "Object")) + "[]{" + ", ".join(literals) + "}"
    if java and isinstance(value, dict):
        return 'new HashMap<>() {{ ' + " ".join(
            "put(" + json.dumps(k, ensure_ascii=False) + ", " + json.dumps(v, ensure_ascii=False) + ");"
            for k, v in value.items()) + " }}"
    literal = json.dumps(value, ensure_ascii=False, allow_nan=False)
    if javascript and kind == "float":
        literal = format(Decimal(literal), "f")  # JS checker does not parse exponents
    if java and kind == "float":
        literal += "f"
    if java and kind == "long":
        literal += "L"
    if javascript and kind == "bigint":
        literal += "n"
    return literal


def nonempty_accepted(accepted):
    options = accepted if isinstance(accepted, list) else [accepted]
    if not options:
        raise ValueError("no accepted values")
    return [v for v in options if v is not None and v != ""]


def materialize_truth(value, schema):
    """BFCL dict GT stores another accepted-values list under each dict key."""
    kind = schema.get("type", "any").lower()
    if kind in {"dict", "object", "hashmap"} and isinstance(value, dict):
        properties = schema.get("properties", {})
        result = {}
        for key, accepted in value.items():
            choices = nonempty_accepted(accepted)
            if choices:
                result[key] = materialize_truth(choices[0], properties.get(key, {}))
        return result
    if kind in {"array", "arraylist", "tuple", "list"} and isinstance(value, list):
        return [materialize_truth(v, schema.get("items", {})) for v in value]
    return value


def truth_calls(truth, functions=None, category="simple_python"):
    schemas = {f["name"]: f.get("parameters", {}).get("properties", {}) for f in functions or []}
    calls = []
    for item in truth:
        for name, args in item.items():
            if functions is not None and name not in schemas:
                raise ValueError(f"ground-truth function absent from schema: {name}")
            rendered = {}
            for key, accepted in args.items():
                nonempty = nonempty_accepted(accepted)
                if nonempty:
                    if functions is not None and key not in schemas[name]:
                        raise ValueError(f"ground-truth argument absent from schema: {name}.{key}")
                    schema = schemas.get(name, {}).get(key, {})
                    rendered[key] = typed_value(materialize_truth(nonempty[0], schema), schema, category)
            calls.append({name: rendered})
    return calls


def render_calls(calls):
    # An empty truth renders zero tool calls (empty text + EOS when scored).
    # No invented refusal prose is attributed to a teacher.
    return "\n".join("<tool_call>\n" + json.dumps({"name": name, "arguments": args},
                                               ensure_ascii=False, allow_nan=False) + "\n</tool_call>"
                     for call in calls for name, args in call.items())


def render_truth(truth, functions=None, category="simple_python"):
    return render_calls(truth_calls(truth, functions, category))


def legacy_render_truth(truth):
    """The historical miner format, retained only for the before/after audit."""
    return render_calls([{name: {k: v[0] if isinstance(v, list) and v else v for k, v in args.items()}}
                         for item in truth for name, args in item.items()])


def unpack_pool_prompt(prompt, renderer):
    """Recover only a fully round-trippable single-turn historical state."""
    tools = re.search(r"<tools>\n(.*?)\n</tools>", prompt, re.S)
    messages = re.findall(r"<\|im_start\|>(system|user|assistant|tool)\n(.*?)<\|im_end\|>\n",
                          prompt, re.S)
    if (not tools or [role for role, _ in messages] != ["system", "user"]
            or prompt.count("<|im_start|>") != 3 or prompt.count("<|im_end|>") != 2):
        raise ValueError("pool-only state is not a recoverable single user turn")
    functions = [json.loads(line) for line in tools[1].splitlines()]
    question = [[dict(role="user", content=messages[1][1])]]
    if renderer(question, functions) != prompt:
        raise ValueError("pool prompt does not round-trip through BFCLAdapter")
    return question, functions


def question_content_hash(question, functions, truth):
    """Probe content_hash: SHA1 of sorted question/function/ground_truth JSON."""
    value = dict(question=question, function=functions, ground_truth=normalize_truth(truth))
    return hashlib.sha1(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                   allow_nan=False).encode()).hexdigest()


def generated_id(tid):
    return tid.startswith(("gen_", "oos_", "genmt_"))


class Exclusions:
    def __init__(self, *, exclude_probe_parents=False):
        self.ids, self.parents, self.hashes = defaultdict(set), defaultdict(set), defaultdict(set)
        self.probe_parents = defaultdict(set)
        self.content_hashes, self.official_ids = defaultdict(set), defaultdict(set)
        self.prompt_counts = {}
        self.probes = []
        self.exclude_probe_parents = exclude_probe_parents
        self.sources = []

    def load(self, path, *, heldout_only=False, probe=False):
        path = Path(path)
        obj = json.loads(path.read_text())
        reason = str(path)
        self.sources.append(dict(path=reason, sha256=digest(obj), probe=probe,
                                 ignored_unused_spares=len(obj.get("spare_unused", [])) if isinstance(obj, dict) and probe else 0))
        if probe:
            self.probes.extend(dict(source=reason, record=p) for p in
                               (obj.get("probes", []) if isinstance(obj, dict) else obj)
                               if isinstance(p, dict))

        def visit(value, key=""):
            if isinstance(value, dict):
                for k, v in value.items():
                    if k in {"meta", "note", "why", "prompt", "question", "function", "truth", "ground_truth"} or (probe and k == "spare_unused"):
                        continue
                    # Probe parent IDs are metadata, not a ban on every sibling.
                    if probe and k in {"parent_task_id", "parent_task", "group", "seed_task", "parent"}:
                        if isinstance(v, str):
                            self.probe_parents[v.split("#", 1)[0]].add(reason)
                        continue
                    visit(v, k)
            elif isinstance(value, list):
                for v in value:
                    visit(v, key)
            elif isinstance(value, str):
                if key in {"hashes", "prompt_hash", "prompt_sha1", "prompt_sha256"}:
                    self.hashes[value].add(reason)
                elif key == "content_hash":
                    self.content_hashes[value].add(reason)
                elif key in {"", "id", "ids", "task_id", "task_ids", "gen_ids", "_traj",
                             "calibration", "certification", "heldout", "held_out", "test",
                             "validation", "confirm", "P_confirm", "eval"}:
                    tid = value.split("#", 1)[0]
                    self.ids[tid].add(reason)
                    if not probe:
                        self.parents[tid].add(reason)
                        if not generated_id(tid):
                            self.official_ids[tid].add(reason)
                    else:
                        self.probe_parents[parent_id(tid)].add(reason)

        if heldout_only:
            def heldouts(value):
                if isinstance(value, dict):
                    for k, v in value.items():
                        if any(w in k.lower() for w in ("calib", "certif", "held", "test", "valid", "confirm", "eval")):
                            # Normalize list-container keys without treating support/demand as held out.
                            visit(v, "ids")
                        elif isinstance(v, dict):
                            heldouts(v)
            heldouts(obj)
        else:
            visit(obj)

    def index_prompts(self, rows):
        """Count distinct exact prompts across ALL source pools, including skipped rows."""
        prompts = defaultdict(set)
        for tid, prompt in rows:
            prompts[tid.split("#", 1)[0]].add(prompt)
        self.prompt_counts = {tid: len(values) for tid, values in sorted(prompts.items())}

    def matches(self, task_id, parent, prompt, *, content_hashes=(),
                exclude_probe_parents=None, previous_rule=False):
        """Content first; reused generator IDs alone never identify protected states."""
        if exclude_probe_parents is None:
            exclude_probe_parents = self.exclude_probe_parents
        sha1 = hashlib.sha1(prompt.encode()).hexdigest()
        sha256 = hashlib.sha256(prompt.encode()).hexdigest()
        tid = task_id.split("#", 1)[0]
        parent = parent.split("#", 1)[0]
        entries = [("prompt_hash", h, self.hashes) for h in (sha1, sha1[:16], sha256)]
        if previous_rule:
            entries += [("ids", tid, self.ids), ("parent", parent, self.parents)]
        else:
            entries += [("content_hash", h, self.content_hashes) for full in sorted(content_hashes)
                        for h in (full, full[:16])]
            if self.prompt_counts.get(tid) == 1:
                entries.append(("unique_id", tid, self.ids))
            entries.append(("official_id", tid, self.official_ids))
            # A generated parent is safe by ID only when it identifies one prompt.
            # Official held-out parents remain protected even if absent from pools.
            if not generated_id(parent) or self.prompt_counts.get(parent) == 1:
                entries.append(("parent", parent, self.parents))
        if exclude_probe_parents:
            entries.append(("parent", parent, self.probe_parents))
        return [dict(source=source, mechanism=kind, value=value,
                     probe_parent=lookup is self.probe_parents)
                for kind, value, lookup in entries for source in sorted(lookup.get(value, set()))]

    def reasons(self, task_id, parent, prompt):
        return sorted({m["source"] for m in self.matches(task_id, parent, prompt)})

    def check_probe_prompts(self, states, todo, leakage_rows):
        """Assert each declared probe hash is protected and absent from all eligible states."""
        by_source = {}
        eligible = [s["prompt"] for s in states] + [t["state"]["prompt"] for t in todo]
        eligible_hashes = {h for p in eligible for h in
                           (hashlib.sha1(p.encode()).hexdigest(), hashlib.sha1(p.encode()).hexdigest()[:16],
                            hashlib.sha256(p.encode()).hexdigest())}
        for item in self.probes:
            p, source = item["record"], item["source"]
            result = by_source.setdefault(source, dict(probe_records=0, checked_records=0,
                                                       distinct_prompt_hashes=set(), present_in_source=set(), source_rows=0,
                                                       missing_hash_records=[], failures=[]))
            result["probe_records"] += 1
            hashes = {p[k] for k in ("prompt_sha1", "prompt_hash", "prompt_sha256") if p.get(k)}
            label = p.get("probe_id", p.get("task_id"))
            if not hashes:
                result["missing_hash_records"].append(label)
                continue
            result["checked_records"] += 1
            result["distinct_prompt_hashes"].add(p.get("prompt_sha1", p.get("prompt_hash", p.get("prompt_sha256"))))
            if not all(source in self.hashes[h] for h in hashes) or hashes & eligible_hashes:
                result["failures"].append(label)
            if "prompt" in p:
                actual = {hashlib.sha1(p["prompt"].encode()).hexdigest(),
                          hashlib.sha1(p["prompt"].encode()).hexdigest()[:16],
                          hashlib.sha256(p["prompt"].encode()).hexdigest()}
                if not hashes <= actual or not self.matches("", "", p["prompt"]):
                    result["failures"].append(label)
            for r in leakage_rows:
                if hashes & {r["prompt_sha1"], r["prompt_sha1"][:16], r["prompt_sha256"]}:
                    result["present_in_source"].add(r["prompt_sha1"])
                    result["source_rows"] += 1
                    if not any(m["mechanism"] == "prompt_hash" and m["source"] == source for m in r["matches"]):
                        result["failures"].append(r["source"])
        for result in by_source.values():
            for key in ("distinct_prompt_hashes", "present_in_source"):
                result[key] = len(result[key])
            result["passed"] = not result["failures"] and not result["missing_hash_records"]
        failures = {s: r["failures"] for s, r in by_source.items() if r["failures"]}
        if failures:
            raise AssertionError(f"probe prompt exclusion failed: {failures}")
        return dict(passed=all(r["passed"] for r in by_source.values()), by_source=by_source,
                    eligible_prompt_overlap=0)


def load_exclusions(root, *, exclude_probe_parents=False):
    ex = Exclusions(exclude_probe_parents=exclude_probe_parents)
    # Both spellings are accepted; the checked-in artifacts use behavior_atom_v1.
    required = [root / "data/behavior_atom_v1/probes.json", root / "data/behavior_atom_v1/probes_disc.json",
                root / "configs/support_split.json"]
    calibration = sorted((root / "data/bfcl_sft").glob("calibration_ids*.json"))
    if not calibration:
        raise ValueError("no calibration_ids*.json exclusion files")
    for p in required[:2]:
        ex.load(p, probe=True)
    for p in sorted((root / "data/bfcl_atom_v1").glob("probes*.json")):
        ex.load(p, probe=True)
    for p in calibration:
        ex.load(p)
    ex.load(required[2], heldout_only=True)
    bfcl_split = root / "configs/bfcl_support_split.json"
    if bfcl_split.exists():
        ex.load(bfcl_split, heldout_only=True)
    return ex


def binding_distance(calls, truth):
    """Normalized argument mismatch to the closest allowed binding.

    Numeric mismatches use |x-y|/(1+|x-y|); other values use exact typed
    equality. Calls are matched without ordering by minimum assignment. This
    is a diagnostic only; all correctness comes from the BFCL checker.
    """
    if not calls or not truth:
        return 0.0 if not calls and not truth else 1.0
    actual = [(n, a) for c in calls for n, a in c.items()]
    expected = [(n, a) for c in truth for n, a in c.items()]
    if len(actual) != len(expected):
        return 1.0
    if len(actual) > 12:
        return None  # avoid an exponential diagnostic on long call sequences

    def value_distance(a, b):
        if type(a) is type(b) and a == b:
            return 0.0
        if type(a) in (int, float) and type(b) in (int, float):
            delta = abs(a - b)
            return delta / (1 + delta)
        return 1.0

    def cost(a, b):
        if a[0] != b[0]:
            return 1.0
        keys = set(a[1]) | set(b[1])
        losses = []
        for key in keys:
            options = b[1].get(key, [])
            options = options if isinstance(options, list) else [options]
            if key not in a[1]:
                losses.append(0.0 if "" in options else 1.0)
            elif not options:
                losses.append(1.0)
            else:
                losses.append(min(value_distance(a[1][key], v) for v in options))
        return sum(losses) / max(1, len(losses))

    dp = {0: 0.0}
    for a in actual:
        nxt = {}
        for mask, total in dp.items():
            for j, b in enumerate(expected):
                if not mask & (1 << j):
                    key = mask | (1 << j)
                    nxt[key] = min(nxt.get(key, math.inf), total + cost(a, b))
        dp = nxt
    return dp[(1 << len(expected)) - 1] / len(expected)


def balanced(rows, seed):
    groups = defaultdict(list)
    for p in rows:
        groups[p["seed_function"]].append(p)
    for items in groups.values():
        items.sort(key=lambda p: digest([seed, p["pair_id"]]))
    result = []
    keys = sorted(groups, key=lambda k: digest([seed, k]))
    while any(groups.values()):
        for key in keys:
            if groups[key]:
                result.append(groups[key].pop())
    return result


def cap_pairs(rows, seed, limit=None, *, type3_cap=DEFAULT_TYPE3_CAP):
    if not math.isfinite(type3_cap) or not 0 <= type3_cap <= 1:
        raise ValueError("type3_cap must be a finite fraction between zero and one")
    if limit is not None and limit < 1:
        raise ValueError("target must be positive")
    if type3_cap == 1:
        return sorted(balanced(rows, seed)[:limit], key=lambda p: p["pair_id"])
    # Integer arithmetic avoids rounding away a pair at an exact cap boundary.
    cap = Fraction(1, 3) if type3_cap == DEFAULT_TYPE3_CAP else Fraction(str(type3_cap))
    binding = balanced([p for p in rows if p["type"] in TYPES[:2]], seed)
    boundary = balanced([p for p in rows if p["type"] == "call_vs_abstain"], seed)
    n3 = min(len(boundary), len(binding) * cap.numerator // (cap.denominator - cap.numerator))
    if limit is None:
        return sorted(binding + boundary[:n3], key=lambda p: p["pair_id"])
    n3 = min(n3, limit * cap.numerator // cap.denominator)
    selected = binding[:limit - n3] + boundary[:n3]
    return sorted(selected, key=lambda p: p["pair_id"])


def counts(rows):
    return dict(total=len(rows), by_type={t: sum(p["type"] == t for p in rows) for t in TYPES},
                by_seed_function=dict(sorted(Counter(p["seed_function"] for p in rows).items())))


def leakage_summary(rows, *, exclude_probe_parents, source=None, previous_rule=False):
    """Source-row counts; primary mechanisms partition rows, matches may overlap."""
    groups = defaultdict(list)
    for row in rows:
        matches = [m for m in row["previous_matches" if previous_rule else "matches"] if (exclude_probe_parents or not m["probe_parent"])
                   and (source is None or m["source"] == source)]
        if matches:
            groups[row["parent"]].append({m["mechanism"] for m in matches})
    by_seed = {}
    order = ("ids", "prompt_hash", "parent") if previous_rule else LEAKAGE_REASONS
    for seed, mechanisms in sorted(groups.items()):
        primary = Counter(next(k for k in order if k in m) for m in mechanisms)
        by_seed[seed] = dict(total=len(mechanisms), by_reason={k: primary[k] for k in order},
                             matching_reasons={k: sum(k in m for m in mechanisms) for k in order})
    return dict(total=sum(g["total"] for g in by_seed.values()), by_seed_function=by_seed)


def build_candidates(raw_sources, pool, exclusions, *, seed=0, renderer=render_prompt):
    """raw_sources is [(path, rows)]; pool is (path, rows). CPU only."""
    audit = dict(exclusion_sources=exclusions.sources, excluded=[], skipped=[],
                 input_rows={str(p): len(rows) for p, rows in [*raw_sources, pool]})
    states, todo, pool_by_key = {}, [], defaultdict(list)
    leakage_rows = {}
    prompts, content_by_source, content_by_prompt = {}, {}, defaultdict(set)
    id_prompts = []
    for path, rows in raw_sources:
        for line, row in enumerate(rows, 1):
            source = f"{path}:{line}"
            prompt = renderer(row["question"], row["function"])
            prompts[source] = prompt
            id_prompts.append((row["id"], prompt))
            hashes = {question_content_hash(row["question"], row["function"], row["ground_truth"])}
            content_by_source[source] = hashes
            content_by_prompt[prompt].update(hashes)
    for line, row in enumerate(pool[1], 1):
        pool_by_key[row["task_id"], row["prompt"]].append((line, row))
        id_prompts.append((row["task_id"], row["prompt"]))
        hashes = set(content_by_prompt[row["prompt"]])
        # Prefer independent authored GT where available; pool-only targets can
        # supply exact bindings, but must not replace GT accepted alternatives.
        if not hashes:
            try:
                question, functions = unpack_pool_prompt(row["prompt"], renderer)
                calls = parse_calls(row["response"])
                truth = [{name: {k: [v] for k, v in args.items()}} for c in calls for name, args in c.items()]
                hashes.add(question_content_hash(question, functions, truth))
            except (ValueError, TypeError):
                pass  # prompt/ID exclusions still apply to unrecoverable states
        content_by_source[f"{pool[0]}:{line}"] = hashes
    exclusions.index_prompts(id_prompts)

    def leakage_record(tid, parent, prompt, source):
        return dict(task_id=tid, parent=parent, source=source,
                    prompt_sha1=hashlib.sha1(prompt.encode()).hexdigest(),
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    content_hashes=sorted(content_by_source[source]),
                    matches=exclusions.matches(tid, parent, prompt, content_hashes=content_by_source[source],
                                               exclude_probe_parents=True),
                    previous_matches=exclusions.matches(tid, parent, prompt, exclude_probe_parents=True,
                                                        previous_rule=True))

    # Verify protection even for rows skipped as unverified/unsupported. They
    # do not count as leakage exclusions in the before/after policy comparison.
    all_source_rows = {}
    for path, rows in raw_sources:
        for line, row in enumerate(rows, 1):
            source = f"{path}:{line}"
            all_source_rows[source] = leakage_record(row["id"], row.get("seed_task") or parent_id(row["id"]),
                                                     prompts[source], source)
    for line, row in enumerate(pool[1], 1):
        source = f"{pool[0]}:{line}"
        all_source_rows[source] = leakage_record(row["task_id"], parent_id(row["task_id"]), row["prompt"], source)

    def exclude(source):
        if source not in leakage_rows:
            record = all_source_rows[source]
            matches = record["matches"]
            leakage_rows[source] = record
            active = [m for m in matches if exclusions.exclude_probe_parents or not m["probe_parent"]]
            if active:
                audit["excluded"].append(dict(record, matches=active, reasons=sorted({m["source"] for m in active})))
        return any(exclusions.exclude_probe_parents or not m["probe_parent"] for m in leakage_rows[source]["matches"])

    def add(row, source, *, prompt=None, pool_row=None):
        tid = row["id"]
        parent = row.get("seed_task") or parent_id(tid)
        prompt = prompt if prompt is not None else prompts[source]
        if exclude(source):
            return
        truth = normalize_truth(row["ground_truth"])
        category = row.get("seed_category", parent.rsplit("_", 1)[0]).removeprefix("oos_")
        abstain = row.get("out_of_scope", False) or "irrelevance" in category
        if bool(truth) == bool(abstain):
            audit["skipped"].append(dict(source=source, reason="inconsistent call/abstain ground truth"))
            return
        evidence = pool_row
        if evidence is None:
            matches = [(line, r) for line, r in pool_by_key.get((tid, prompt), [])
                       if r.get("teacher") in {"teacher_authored_gt", "teacher_authored_abstain"}]
            if len({r["response"] for _, r in matches}) > 1:
                audit["skipped"].append(dict(source=source, reason="conflicting historical teacher responses"))
                return
            evidence = matches[0] if matches else None
        # Keep the old text for a checker-backed, reproducible before/after audit.
        # Bad pool formatting can be recovered when independent GT is present.
        teacher_text = evidence[1]["response"] if evidence else legacy_render_truth(truth)
        sid = "state_" + digest([prompt, truth])[:20]
        state = dict(state_id=sid, task_id=tid, seed_task=parent, category=category,
                     question=row["question"], function=row["function"], ground_truth=truth,
                     out_of_scope=bool(abstain), prompt=prompt, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                     teacher_text=teacher_text, y_plus=teacher_text,
                     teacher="teacher_authored_gt", rendered_from_gt=evidence is None,
                     provenance=dict(kind="teacher_authored_gt", historical_teacher_evidence=True,
                                     source=source, teacher_text_source=(f"{pool[0]}:{evidence[0]}" if evidence else source),
                                     original_teacher=evidence[1]["teacher"] if evidence else "teacher_authored_gt",
                                     truth_source=row.get("truth_source", "ground_truth"), ledger=LEDGER,
                                     new_teacher_tokens=0), sources=[source])
        if sid in states:
            states[sid]["sources"].append(source)
        else:
            states[sid] = state

    for path, rows in raw_sources:
        for line, row in enumerate(rows, 1):
            source = f"{path}:{line}"
            if "verified" in row and not row["verified"]:
                audit["skipped"].append(dict(source=source, reason="unverified generated row"))
                continue
            add(row, source)

    for line, row in enumerate(pool[1], 1):
        source, tid = f"{pool[0]}:{line}", row["task_id"]
        category = row.get("_seed_category", "")
        if "multi_turn" in category or "memory" in category:
            if not exclude(source):
                todo.append(dict(candidate_id="todo_" + digest([source, row["prompt"]])[:20], type="multi_turn",
                                 task_id=tid, seed_task=parent_id(tid), category=category, source=source,
                                 state=row, missing_side="s2", status="needs_historical_or_future_teacher_side",
                                 fabricated_side=False, ledger=LEDGER, new_teacher_tokens=0))
            continue
        if not tid.startswith(("gen_", "oos_")) or row.get("teacher") not in {
                "teacher_authored_gt", "teacher_authored_abstain"}:
            continue
        # Known raw states were already joined by (id, exact prompt). Pool-only
        # variants still carry a complete prompt and executable teacher action.
        if exclude(source):
            continue
        if any(s["task_id"] == tid and s["prompt"] == row["prompt"] for s in states.values()):
            continue
        try:
            question, functions = unpack_pool_prompt(row["prompt"], renderer)
            calls = parse_calls(row["response"])
        except (ValueError, TypeError) as exc:
            audit["skipped"].append(dict(source=source, reason=str(exc)))
            continue
        truth = [{name: {k: [v] for k, v in args.items()}} for c in calls for name, args in c.items()]
        add(dict(id=tid, seed_task=parent_id(tid), seed_category=category,
                 question=question, function=functions, ground_truth=truth, out_of_scope=not calls,
                 truth_source="pool_teacher_response_exact_binding"), source, prompt=row["prompt"], pool_row=(line, row))

    # Quarantine contradictory labels on an identical state rather than pair them.
    by_prompt = defaultdict(list)
    for s in states.values():
        by_prompt[s["prompt_sha256"]].append(s["state_id"])
    for ids in by_prompt.values():
        if len(ids) > 1:
            for sid in ids:
                audit["skipped"].append(dict(source=states[sid]["sources"], reason="conflicting truths for identical prompt"))
                del states[sid]

    # Connected seed/schema groups prevent aliases of an identical schema from
    # leaking out of the whole-function holdout.
    parent = {s["seed_task"]: s["seed_task"] for s in states.values()}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    schemas = defaultdict(list)
    for s in states.values():
        schemas[digest(s["function"])].append(s["seed_task"])
    for seeds in schemas.values():
        for p in seeds[1:]:
            a, b = find(seeds[0]), find(p)
            parent[max(a, b)] = min(a, b)
    pairs = []
    for a, b in combinations(sorted(states.values(), key=lambda s: s["state_id"]), 2):
        same_seed = a["seed_task"] == b["seed_task"]
        same_schema = a["function"] == b["function"]
        if not (same_seed or same_schema) or a["question"] == b["question"]:
            continue
        if a["out_of_scope"] != b["out_of_scope"]:
            kind = "call_vs_abstain"
        elif a["out_of_scope"]:
            continue
        else:
            ca, cb = truth_calls(a["ground_truth"]), truth_calls(b["ground_truth"])
            if (Counter(n for c in ca for n in c) != Counter(n for c in cb for n in c)
                    or binding_distance(ca, b["ground_truth"]) == 0
                    or binding_distance(cb, a["ground_truth"]) == 0):
                continue
            kind = "binding"
        pairs.append(dict(pair_id="cc_" + digest([a["state_id"], b["state_id"]])[:20], type=kind,
                          seed_function=find(a["seed_task"]), seed_tasks=sorted({a["seed_task"], b["seed_task"]}),
                          schema_match="identical_function_list" if same_schema else "same_seed_task",
                          source_priority="P1_historical_teacher_variant", s1=a, s2=b))
    selected = cap_pairs(pairs, seed)
    audit["exclude_probe_parents"] = exclusions.exclude_probe_parents
    audit["leakage_comparison"] = {
        mode: dict(leakage_summary(leakage_rows.values(), exclude_probe_parents=enabled),
                   by_source={s["path"]: leakage_summary(leakage_rows.values(), exclude_probe_parents=enabled, source=s["path"])
                              for s in exclusions.sources})
        for mode, enabled in (("probe_parents_off", False), ("probe_parents_on", True))}
    audit["additional_probe_parent_exclusions"] = [r for r in leakage_rows.values()
                                                   if r["matches"] and all(m["probe_parent"] for m in r["matches"])]
    audit["source_id_prompt_counts"] = exclusions.prompt_counts
    active = lambda matches: any(exclusions.exclude_probe_parents or not m["probe_parent"] for m in matches)
    audit["recovered_rows"] = [r for r in leakage_rows.values() if active(r["previous_matches"]) and not active(r["matches"])]
    audit["newly_excluded_rows"] = [r for r in leakage_rows.values() if active(r["matches"]) and not active(r["previous_matches"])]
    audit["previous_rule"] = leakage_summary(leakage_rows.values(), exclude_probe_parents=exclusions.exclude_probe_parents,
                                             previous_rule=True)
    current = audit["leakage_comparison"]["probe_parents_on" if exclusions.exclude_probe_parents else "probe_parents_off"]
    audit["recovery_by_seed_function"] = {
        group: dict(source_rows=sum(r["parent"] == group for r in leakage_rows.values()),
                    previous_excluded=audit["previous_rule"]["by_seed_function"].get(group, {}).get("total", 0),
                    excluded=current["by_seed_function"].get(group, {}).get("total", 0),
                    recovered=sum(r["parent"] == group for r in audit["recovered_rows"]),
                    newly_excluded=sum(r["parent"] == group for r in audit["newly_excluded_rows"]))
        for group in sorted({r["parent"] for r in leakage_rows.values()})}
    audit["probe_prompt_check"] = exclusions.check_probe_prompts(states.values(), todo, list(all_source_rows.values()))
    audit.update(unique_states=len(states), before_type3_cap=counts(pairs), candidates=counts(selected),
                 type3_removed_by_cap=len(pairs) - len(selected), type2_todo=len(todo),
                 seed=seed, new_teacher_tokens=0, ledger=LEDGER,
                 identity="state=prompt+truth; pool join=(task_id,prompt); prompt/content hashes protect states; "
                          "ID exclusion requires one distinct source prompt or an official held-out ID; protected parent lineage retained")
    return selected, todo, audit


class VLLMClient:
    def __init__(self, url, max_tokens=512, *, model=BASE_MODEL):
        self.url, self.max_tokens = url.rstrip("/"), max_tokens
        self.model = model

    def select_model(self, model=None):
        """Evaluation may use an adapted server; probe keeps validate_base()."""
        with urllib.request.urlopen(self.url + "/models", timeout=30) as response:
            models = json.load(response)["data"]
        matches = [m for m in models if model is None or m.get("id") == model]
        if len(matches) != 1:
            raise ValueError("evaluation needs one served model; select it with --served-model")
        self.model = matches[0]["id"]
        return matches[0]

    def validate_base(self):
        with urllib.request.urlopen(self.url + "/models", timeout=30) as response:
            models = json.load(response)["data"]
        matches = [m for m in models if m.get("id") == BASE_MODEL and m.get("root") == BASE_MODEL]
        if len(matches) != 1 or matches[0].get("parent"):
            raise ValueError("probe requires the unadapted Qwen/Qwen3.5-4B server (matching id/root, no parent)")
        return matches[0]

    def generate(self, prompt, *, seed, temperature):
        payload = dict(model=self.model, prompt=thinking_off(prompt), temperature=temperature,
                       seed=seed, max_tokens=self.max_tokens, n=1, top_p=1.0, top_k=-1,
                       repetition_penalty=1.0, skip_special_tokens=False)
        req = urllib.request.Request(self.url + "/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as response:
            result = json.load(response)
        choice = result["choices"][0]
        return dict(output=choice["text"], finish_reason=choice.get("finish_reason"),
                    usage=result.get("usage"), request_seed=seed, temperature=temperature)


def checker_probe(side):
    return dict(function=side["function"], truth=side["ground_truth"], category=side["category"],
                expected="abstain" if side["out_of_scope"] else "call")


def checked_output(output, side, checker):
    try:
        calls = parse_calls(output)
    except (ValueError, TypeError) as exc:
        return dict(outcome=0, calls=None, parse_error=str(exc), checker_version=checker.checker_version)
    # Checker/infrastructure exceptions propagate: they are never student errors.
    verdict = checker.check(checker_probe(side), calls)
    if type(verdict.get("valid")) is not bool:
        raise ValueError(f"non-boolean BFCL verdict: {verdict}")
    return dict(outcome=int(verdict["valid"]), calls=calls, verdict=verdict,
                checker_version=checker.checker_version)


def validate_teachers(pairs, checker):
    """CPU audit and schema rendering; preserve passing pool bytes and all GT.

    Mutate every occurrence of a side, including independently loaded JSONL
    copies. Before counts always refer to the original historical rendering.
    """
    states = {s["state_id"]: s for p in pairs for s in (p["s1"], p["s2"])}
    records = {}
    for sid, side in sorted(states.items()):
        original = side.get("y_plus_original", side["teacher_text"])
        before = checked_output(original, side, checker)
        after = checked_output(side["teacher_text"], side, checker) if "y_plus_original" in side else before
        rendering_error = None
        if after["outcome"] != 1 or (side["rendered_from_gt"] and "y_plus_original" not in side):
            try:
                text = render_truth(side["ground_truth"], side["function"], side["category"])
            except (ValueError, TypeError, SyntaxError) as exc:
                rendering_error = str(exc)
            else:
                after = checked_output(text, side, checker)
                if text != original:
                    side["y_plus_original"] = original
                side.update(teacher_text=text, rendered_from_gt=True)
                side["provenance"]["kind"] = "teacher_authored_gt/rendered_typed"
        side["y_plus"] = side["teacher_text"]
        if not side["teacher_text"].strip():
            side["y_plus_from_student"] = False  # No student probe exists yet.
        records[sid] = dict(after, before=before, retyped=side["teacher_text"] != original)
        if rendering_error:
            records[sid]["rendering_error"] = rendering_error
        if after["outcome"] != 1 and "java" in side["category"]:
            any_args = [(name, key) for f in side["function"] for name in [f["name"]]
                        for key, spec in f.get("parameters", {}).get("properties", {}).items()
                        if spec.get("type") == "any"]
            incompatible = [f"{name}.{key}" for call in side["ground_truth"] for name, key in any_args
                            if name in call and key in call[name] and nonempty_accepted(call[name][key])
                            and not isinstance(nonempty_accepted(call[name][key])[0], str)]
            if incompatible:
                records[sid]["checker_limitation"] = (
                    f"BFCL converts any arguments {incompatible} to str, but their authored accepted values "
                    "are non-strings. Rendering cannot satisfy both the wire type and unchanged GT.")
    for pair in pairs:
        for key in ("s1", "s2"):
            pair[key].update(states[pair[key]["state_id"]])
    valid = [p for p in pairs if all(records[p[s]["state_id"]]["outcome"] == 1 for s in ("s1", "s2"))]
    valid_before = [p for p in pairs if all(records[p[s]["state_id"]]["before"]["outcome"] == 1 for s in ("s1", "s2"))]
    def errors(field):
        results = [r[field] if field else r for r in records.values()]
        return dict(sorted(Counter(r.get("verdict", {}).get("error_type") or r.get("parse_error", "unknown")
                                   for r in results if r["outcome"] != 1).items()))
    return dict(checker_version=checker.checker_version, states=records, paired_states=len(states),
                valid_states_before=sum(r["before"]["outcome"] == 1 for r in records.values()),
                invalid_states_before=sum(r["before"]["outcome"] == 0 for r in records.values()),
                valid_states=sum(r["outcome"] == 1 for r in records.values()),
                invalid_states=sum(r["outcome"] == 0 for r in records.values()),
                retyped_states=sum(r["retyped"] for r in records.values()),
                errors_before=errors("before"), errors_after=errors(None),
                remaining_failures=[dict(state_id=sid, task_id=states[sid]["task_id"],
                                         error=r.get("verdict", {}).get("error", r.get("rendering_error", r.get("parse_error"))),
                                         checker_limitation=r.get("checker_limitation"))
                                    for sid, r in records.items() if r["outcome"] != 1],
                pairs_with_two_valid_teacher_texts_before=counts(valid_before),
                pairs_with_two_valid_teacher_texts=counts(valid),
                upper_bound_after_type3_cap=counts(cap_pairs(valid, 0)))


def generate_states(pairs, client, checker, *, seed=0, progress=None, samples=4):
    states = {s["state_id"]: s for p in pairs for s in (p["s1"], p["s2"])}
    records = {}
    for index, (sid, side) in enumerate(sorted(states.items())):
        teacher = checked_output(side["teacher_text"], side, checker)
        outputs = []
        for sample_index in range(1 + samples):
            request_seed = int(digest([seed, sid, sample_index])[:8], 16) % (2**31)
            raw = client.generate(side["prompt"], seed=request_seed, temperature=0.0 if sample_index == 0 else 0.7)
            outputs.append(dict(**raw, mode="greedy" if sample_index == 0 else "sample",
                                sample_index=sample_index, **checked_output(raw["output"], side, checker)))
        records[sid] = dict(state_id=sid, prompt_sha256=hashlib.sha256(side["prompt"].encode()).hexdigest(),
                            teacher_check=teacher, outputs=outputs,
                            greedy_outcome=outputs[0]["outcome"],
                            sample_pass_rate=sum(o["outcome"] for o in outputs[1:]) / samples if samples else None)
        if progress:
            progress(index + 1, len(states), records)
    return records


def teacher_score_jobs(pairs):
    jobs = {}
    for p in pairs:
        for side in (p["s1"], p["s2"]):
            for target in (p["s1"], p["s2"]):
                key = digest([side["prompt"], target["teacher_text"]])
                jobs[key] = dict(probe_id=key, prompt=side["prompt"], y_plus=target["teacher_text"], y_minus=None)
    return jobs


def score_teacher_texts(pairs, student, *, device="cuda", seed=0, context_limit=32768):
    """Reuse C16 exactly: separate prompt/response tokens, EOS, native CE, fp32 sum."""
    from bfas.behavior.margin import score_pairs
    from tools.behavior_atom import gpu_driver as driver
    jobs = teacher_score_jobs(pairs)
    for job in jobs.values():
        tok = student.tokenizer
        n = len(tok(thinking_off(job["prompt"]), add_special_tokens=False)["input_ids"])
        n += len(tok(job["y_plus"], add_special_tokens=False)["input_ids"]) + 1
        if n > context_limit:
            raise ValueError("full teacher-forced sequence exceeds context limit; refusing truncation")
    protocol = driver.frozen_settings(dict(micro_update=dict(device=device, model_id=BASE_MODEL,
                    prompt_cap=context_limit, allow_prompt_truncation=False), eval_seeds=[seed],
                    evaluation=dict(batch_size=2, token_budget=context_limit, head_chunk=128)))
    return {r["probe_id"]: dict(logp=r["logp_plus"], counts=r["counts"])
            for r in score_pairs(student, list(jobs.values()), protocol)}


def assemble_results(pairs, generations, scores=None):
    results = []
    for p in pairs:
        row = dict(pair_id=p["pair_id"], candidate_sha256=digest(p), type=p["type"],
                   seed_function=p["seed_function"], status="complete" if scores is not None else "generated")
        for label, other_label in (("s1", "s2"), ("s2", "s1")):
            side, other = p[label], p[other_label]
            record = json.loads(json.dumps(generations[side["state_id"]]))
            for output in record["outputs"]:
                calls = output["calls"]
                own = binding_distance(calls, side["ground_truth"]) if calls is not None else None
                alternate = binding_distance(calls, other["ground_truth"]) if calls is not None else None
                output.update(binding_distance_own=own, binding_distance_other=alternate,
                              condition_insensitive=bool(output["outcome"] == 0 and alternate is not None
                                                        and own is not None and alternate < own))
            record["condition_insensitive"] = any(o["condition_insensitive"] for o in record["outputs"])
            record["wrong"] = any(o["outcome"] == 0 for o in record["outputs"])
            record["kappa"] = None
            if scores is not None:
                own = scores[digest([side["prompt"], side["teacher_text"]])]
                other_score = scores[digest([side["prompt"], other["teacher_text"]])]
                record.update(logpi_own=own, logpi_other=other_score,
                              kappa=other_score["logp"] - own["logp"])
            row[label] = record
        row["kappa_definition"] = "s1=logpi(a2|s1)-logpi(a1|s1); s2=logpi(a1|s2)-logpi(a2|s2); positive means confusion"
        row["both_greedy_correct"] = all(row[s]["greedy_outcome"] == 1 for s in ("s1", "s2"))
        results.append(row)
    return results


def split_pairs(pairs, *, seed=0, heldout_seed=None, within_fraction=0.2):
    """Partition the supplied selection without resampling or applying a type cap."""
    if not 0 <= within_fraction < 1:
        raise ValueError("within_fraction must be at least zero and less than one")
    groups = defaultdict(list)
    for p in pairs:
        groups[p["seed_function"]].append(p)
    if heldout_seed is not None and heldout_seed not in groups:
        raise ValueError("requested heldout seed has no selected pairs")
    heldout_seed = heldout_seed or (min(groups, key=lambda g: (len(groups[g]), digest([seed, g]))) if groups else None)
    split = dict(seed=seed, heldout_seed_function=heldout_seed, train=[], heldout_seed=[],
                 heldout_within_seed=[], excluded_cross_partition=[], within_fraction=within_fraction,
                 isolation="whole seed/schema group; within-seed prompt-disjoint states; shared-side cross edges excluded")
    if within_fraction == 0:
        split["isolation"] = "whole seed/schema group only; all remaining selected pairs train"
    for group, items in sorted(groups.items()):
        items = sorted(items, key=lambda p: digest([seed, p["pair_id"]]))
        if group == heldout_seed:
            split["heldout_seed"].extend(p["pair_id"] for p in items)
            continue
        if within_fraction == 0:
            split["train"].extend(p["pair_id"] for p in items)
            continue
        n = min(len(items) - 1, max(1, math.ceil(len(items) * within_fraction))) if len(items) > 1 else 0
        held = items[:n]
        states = {p[s]["prompt_sha256"] for p in held for s in ("s1", "s2")}
        split["heldout_within_seed"].extend(p["pair_id"] for p in held)
        for p in items[n:]:
            key = "excluded_cross_partition" if any(p[s]["prompt_sha256"] in states for s in ("s1", "s2")) else "train"
            split[key].append(p["pair_id"])
    split["counts"] = {k: len(split[k]) for k in ("train", "heldout_seed", "heldout_within_seed", "excluded_cross_partition")}
    return split


def training_targets(pair, result):
    """Copy empty teacher targets from this state's checked base outputs only.

    Keep teacher_text and the source candidate untouched: cached kappa still
    describes the historical teacher text (including EOS-only abstentions).
    """
    pair = dict(pair)
    for label in ("s1", "s2"):
        original = pair[label]
        if original.get("teacher_text", "").strip():
            continue
        side = pair[label] = dict(original)
        record = result[label]
        if record.get("state_id") != side.get("state_id"):
            raise ValueError(f"{pair['pair_id']}: base state ID mismatch")
        # Restrict successful fallbacks to the frozen probe temperatures.
        outputs = record.get("outputs", [])
        ordered = [o for mode, temp in (("greedy", 0.0), ("sample", 0.7)) for o in outputs
                   if o.get("mode") == mode and o.get("temperature") == temp]
        verified = next((o for o in ordered if o.get("outcome") == 1
                         and o.get("verdict", {}).get("valid") is True
                         and isinstance(o.get("output"), str) and o["output"].strip()), None)
        side.update(y_plus=verified["output"] if verified else "", y_plus_from_student=verified is not None)
        if verified:
            side["y_plus_provenance"] = "student_verified_" + verified["mode"]
        wrong = next((o for o in ordered if o.get("outcome") == 0
                      and o.get("verdict", {}).get("valid") is False and o.get("calls")
                      and isinstance(o.get("output"), str) and o["output"].strip()), None)
        if wrong:
            side.update(y_minus=wrong["output"], y_minus_provenance="student_wrong")
    return pair


def select_pairs(pairs, results, *, seed=0, limit=32, heldout_seed=None, within_fraction=0.2,
                 type3_cap=DEFAULT_TYPE3_CAP, min_per_type=None):
    if min_per_type is not None and min_per_type < 0:
        raise ValueError("min_per_type must be nonnegative")
    by_id = {r["pair_id"]: r for r in results}
    if len(by_id) != len(results) or set(by_id) != {p["pair_id"] for p in pairs}:
        raise ValueError("probe results must cover each candidate exactly once")
    confused, target_exclusions = [], []
    for p in pairs:
        r = by_id[p["pair_id"]]
        if r.get("candidate_sha256") != digest(p) or r.get("status") != "complete":
            raise ValueError("stale or incomplete probe results; run both generate and score")
        if not all(r[s]["teacher_check"]["outcome"] == 1 for s in ("s1", "s2")):
            continue
        p = training_targets(p, r)
        missing = [s for s in ("s1", "s2") if not p[s].get("y_plus", "").strip()]
        if missing:
            target_exclusions.append(dict(pair_id=p["pair_id"], sides=missing,
                                          reason="empty_teacher_text_no_verified_student_output"))
            continue
        for s in ("s1", "s2"):
            kappa = r[s].get("kappa")
            if not isinstance(kappa, (int, float)) or not math.isfinite(kappa):
                raise ValueError("missing/nonfinite kappa")
        if any(r[s]["wrong"] and (r[s]["condition_insensitive"] or r[s]["kappa"] > 0) for s in ("s1", "s2")):
            confused.append(p)
    selected = cap_pairs(confused, seed, limit, type3_cap=type3_cap)
    split = split_pairs(selected, seed=seed, heldout_seed=heldout_seed, within_fraction=within_fraction)
    bundle = dict(status="complete", seed=seed, target_pairs=limit, eligible=counts(confused), selected=counts(selected),
                  shortfall=max(0, limit - len(selected)), pairs=selected, new_teacher_tokens=0, ledger=LEDGER,
                  invalid_teacher_pairs=sum(any(r[s]["teacher_check"]["outcome"] != 1 for s in ("s1", "s2")) for r in results),
                  selection_rule="wrong on a side (greedy or any of four samples) AND (condition_insensitive OR kappa>0) on that same side")
    filled = [p[s] for p in selected for s in ("s1", "s2") if p[s].get("y_plus_from_student")]
    if filled or target_exclusions:
        bundle["student_target_fallback"] = dict(
            filled_sides=len(filled), filled_states=len({s["state_id"] for s in filled}),
            by_provenance=dict(sorted(Counter(s["y_plus_provenance"] for s in filled).items())),
            student_wrong_sides=sum(s.get("y_minus_provenance") == "student_wrong" for s in filled),
            excluded_pairs=target_exclusions)
    # Retain the legacy serialized outputs for the default configuration.
    # Non-default selections record their cap alongside the existing target_pairs.
    if type3_cap != DEFAULT_TYPE3_CAP or limit != 32 or min_per_type is not None:
        bundle["type3_cap"] = type3_cap
    if min_per_type is not None:
        bundle["min_per_type"] = min_per_type
        bundle["shortfall_by_type"] = {t: max(0, min_per_type - bundle["selected"]["by_type"][t]) for t in TYPES}
    return bundle, split


def write_report(out, audit, bundle, split):
    lines = ["# BFCL condition-contrast pairs (C19)", "", f"Status: {bundle['status']}. New teacher tokens: **0**.", "",
             f"Historical teacher evidence: `{LEDGER}` (existing ledger; estimates are not new usage or teacher-token efficiency claims).",
             "", "Pool responses that pass the AST checker are preserved byte for byte. Other targets use schema-typed rendering of the first non-empty accepted GT value; arguments accepted only as empty string / None are omitted, and no schema defaults are invented. JavaScript/Java arguments use the checker's string-literal convention. Changed texts record `teacher_authored_gt/rendered_typed` provenance and `y_plus_original`. Empty ground truth renders empty text, scored with EOS.",
             "", f"Input rows: `{json.dumps(audit['input_rows'], sort_keys=True)}`.",
             f"Unique eligible states: {audit['unique_states']}; candidate pairs: {audit['candidates']['total']}; type 2 TODO states: {audit['type2_todo']} (0 complete multi_turn pairs).",
             f"Type 3 pairs removed by candidate cap: {audit['type3_removed_by_cap']}. "
             + ("The cap is reapplied after confusion selection." if "type3_cap" not in bundle else
                "The selection cap is configured separately below."), "",
             "| Stage | binding | multi_turn | call_vs_abstain | Total |", "|---|---:|---:|---:|---:|"]
    if bundle["status"] == "pending_probe":
        prior = audit.get("preserved_probe_results")
        note = (f"Previous `probe_results.jsonl` is preserved byte for byte (SHA256 `{prior['sha256']}`). "
                if prior else "")
        lines[4:4] = ["C19d CPU refresh: " + note + "**The probe must be re-run for the new candidates.** "
                      "Previous confusion and split results do not describe this candidate set.", ""]
    for label, count in [("Before cap", audit["before_type3_cap"]), ("Candidates", audit["candidates"]),
                         ("Confused eligible", bundle.get("eligible")), ("Selected", bundle.get("selected"))]:
        if count is not None:
            lines.append(f"| {label} | " + " | ".join(str(count["by_type"][t]) for t in TYPES) + f" | {count['total']} |")
    validation = audit.get("teacher_validation")
    if validation:
        lines += ["", f"CPU historical-target audit: **{validation['valid_states_before']}/{validation['paired_states']} pass before → {validation['valid_states']}/{validation['paired_states']} after**; {validation['retyped_states']} texts needed retyping; {validation['invalid_states']} still fail. Details (including original verdicts): `teacher_validation.json`.",
                  f"Failure types before: `{json.dumps(validation['errors_before'], sort_keys=True)}`; after: `{json.dumps(validation['errors_after'], sort_keys=True)}`.",
                  f"Pairs with two valid historical teacher texts: {validation['pairs_with_two_valid_teacher_texts_before']['total']} before → {validation['pairs_with_two_valid_teacher_texts']['total']} after; maximum selectable before student probing, after the type 3 cap: {validation['upper_bound_after_type3_cap']['total']}."]
        if validation.get("remaining_failures"):
            lines += ["", "Remaining failures retain the authored GT unchanged; they are rejected by the existing teacher-validity selection gate:", ""]
            for failure in validation["remaining_failures"]:
                lines.append(f"- `{failure['state_id']}` (`{failure['task_id']}`): `{json.dumps(failure['error'], ensure_ascii=False)}`. "
                             + (failure.get("checker_limitation") or ""))
    lines += ["", "| Seed function | Candidates | Confused eligible | Selected |", "|---|---:|---:|---:|"]
    for group, n in audit["candidates"]["by_seed_function"].items():
        values = [str(bundle[k]["by_seed_function"].get(group, 0)) if bundle.get(k) else "pending" for k in ("eligible", "selected")]
        lines.append(f"| {group} | {n} | {' | '.join(values)} |")
    if bundle.get("status") == "complete":
        lines += ["", f"Selected {bundle['selected']['total']} of target {bundle['target_pairs']}; shortfall {bundle['shortfall']}. Invalid-teacher pairs rejected: {bundle['invalid_teacher_pairs']}."]
        fallback = bundle.get("student_target_fallback")
        if fallback:
            lines += ["", "Empty teacher text remains the historical probe/scoring target. Training `y_plus` uses a nonempty checker-verified base greedy response, else a verified T=0.7 sample, byte for byte; `y_plus_provenance` is `student_verified_greedy` / `student_verified_sample` and `y_plus_from_student` is true. A checker-failed student call supplies `y_minus` when available (`y_minus_provenance=student_wrong`). No refusal prose is invented.",
                      f"Filled training sides: **{fallback['filled_sides']}** ({fallback['filled_states']} unique states); provenance counts: `{json.dumps(fallback['by_provenance'], sort_keys=True)}`; student-wrong negatives: {fallback['student_wrong_sides']}. Pairs excluded for missing nonempty verified targets: {len(fallback['excluded_pairs'])}."]
            for rejected in fallback["excluded_pairs"]:
                lines.append(f"- `{rejected['pair_id']}` ({', '.join(rejected['sides'])}): `{rejected['reason']}`.")
        if "type3_cap" in bundle:
            cap_note = " (disabled)" if bundle["type3_cap"] == 1 else (
                " (historical exact one third)" if bundle["type3_cap"] == DEFAULT_TYPE3_CAP else "")
            lines += ["", f"Selection settings: `--type3-cap {bundle['type3_cap']}`{cap_note}; `--target {bundle['target_pairs']}`. "
                      + ("The split holds out only the whole seed function; all remaining selected pairs train."
                         if split.get("within_fraction") == 0 else
                         "The split uses these selected pairs unchanged, with shared-state cross edges reported as excluded.")]
        if "probe_from" in bundle:
            lines += ["", f"Probe inputs (candidates, candidate audit, metadata and results): `{bundle['probe_from']}`."]
        if "min_per_type" in bundle:
            lines += ["", f"Reporting only: `--min-per-type {bundle['min_per_type']}`. Shortfalls by type: "
                      f"`{json.dumps(bundle['shortfall_by_type'], sort_keys=True)}`. This threshold does not change selection or splits."]
    lines += ["", "Confusion and split counts remain pending until a real base-student probe completes; an empty results file is not evidence of zero confused pairs.",
              "", "Generation: exact BFCLAdapter prompt plus the existing empty thinking prefix; one greedy output and four independent T=0.7 samples per unique state, deterministic per-state seeds. Correctness uses CheckerBridge/BFCL AST. Broken call tags fail parsing; checker failures stop the run.",
              "", "Binding proximity: minimum assignment across calls, normalized argument mismatch, exact typed categorical values and |x-y|/(1+|x-y|) for numbers. A wrong parsed output is condition-insensitive only when strictly closer to the opposite truth. This heuristic never supplies a correctness verdict.",
              "", "Kappa uses the native behavior.margin.score_pairs implementation: full response, EOS included, native CE followed by fp32 sum, no response or prompt truncation. κ(s1)=logπ(a2|s1)−logπ(a1|s1); κ(s2)=logπ(a1|s2)−logπ(a2|s2). Positive means confusion. Selection requires wrong AND (condition-insensitive OR κ>0) on the same side. Invalid teacher targets are excluded after checking.",
              "", f"Split: `{json.dumps(split, sort_keys=True)}`.",
              ("Whole-function holdout groups include seed aliases sharing the same schema. With `--within-fraction 0`, all remaining selected pairs train; there is no within-seed holdout or cross-partition exclusion. No type or split is padded."
               if split.get("within_fraction") == 0 else
               "Whole-function holdout groups include seed aliases sharing the same schema. Within-seed holdout has no repeated prompt states in training; crossing pairs are reported as excluded. Sparse groups can have no within-seed holdout or no training pairs. No type or split is padded."),
              "", "## Leakage audit (C19d)", "", "Protect exact rendered prompt SHA1 (full and 16-character), SHA256, and question/function/ground-truth content SHA1 (full and 16-character, matching probe content_hash). ID-only exclusions require exactly one distinct rendered prompt across all source pools; repeated rows with the same prompt still count as one. Official calibration/held-out IDs and their generated children remain protected by parent lineage. Ambiguous generated parent IDs alone do not protect unrelated variants. Unused spare probes are not protected. Both support_split.json and the existing BFCL support split calibration sets are checked. Probe siblings sharing only a parent are excluded only with `--exclude-probe-parents` (default OFF).",
              "", f"Excluded source rows: {len(audit['excluded'])}; skipped source rows: {len(audit['skipped'])}. Full source/line/reason lists and input hashes: `candidate_audit.json`."]
    comparison = audit["leakage_comparison"]
    current = comparison["probe_parents_on" if audit["exclude_probe_parents"] else "probe_parents_off"]
    lines += ["", f"Current `--exclude-probe-parents`: {'ON' if audit['exclude_probe_parents'] else 'OFF'}. Exclusions with flag OFF: **{comparison['probe_parents_off']['total']}**; ON: **{comparison['probe_parents_on']['total']}**.",
              "", f"Previous rule on the same source rows and parent flag: **{audit['previous_rule']['total']}** excluded; "
                    f"**{len(audit['recovered_rows'])} rows recovered**, {len(audit['newly_excluded_rows'])} newly excluded.",
              "", "Counts below are inspected source rows, before state deduplication or pairing. Reasons partition each row by prompt hash, content hash, unique ID, official ID, then parent. Overlapping matches and per-file OFF/ON counts are retained in candidate_audit.json. Recovered means excluded by the previous rule and allowed by the new leakage rule; other eligibility checks can still reject a row.",
              "", "| Seed function | Source rows | Prompt hash | Content hash | Unique ID | Official ID | Parent | Excluded | Previous excluded | Recovered | Newly excluded |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    empty = dict(total=0, by_reason={k: 0 for k in LEAKAGE_REASONS})
    for seed, recovery in audit["recovery_by_seed_function"].items():
        group = current["by_seed_function"].get(seed, empty)
        cols = [recovery["source_rows"], *(group["by_reason"][k] for k in LEAKAGE_REASONS),
                recovery["excluded"], recovery["previous_excluded"], recovery["recovered"], recovery["newly_excluded"]]
        lines.append(f"| {seed} | " + " | ".join(map(str, cols)) + " |")
    if audit["recovered_rows"]:
        recovered = Counter((r["source"].rsplit(":", 1)[0], r["parent"]) for r in audit["recovered_rows"])
        lines += ["", "| Source pool | Seed function | Recovered rows |", "|---|---|---:|"]
        for (source, seed), n in sorted(recovered.items()):
            lines.append(f"| {source} | {seed} | {n} |")
    check = audit["probe_prompt_check"]
    lines += ["", f"Probe prompt exclusion assertion: **{'PASS' if check['passed'] else 'INCOMPLETE'}**; "
                    f"eligible-state/TODO prompt overlap: **{check['eligible_prompt_overlap']}**. "
                    "Every declared probe prompt hash is checked for protection, every matching source row is verified to match the exclusion rule (including unverified/unsupported rows), "
                    "and all eligible states are checked before pairing (including states that form no pair).", ""]
    for source, result in check["by_source"].items():
        lines.append(f"- `{source}`: {'PASS' if result['passed'] else 'INCOMPLETE'}; "
                     f"{result['checked_records']}/{result['probe_records']} probe records, "
                     f"{result['distinct_prompt_hashes']} distinct prompt hashes protected, "
                     f"{result['present_in_source']} present in all source pools ({result['source_rows']} matching rows); "
                     f"{len(result['failures'])} failures, {len(result['missing_hash_records'])} missing hash records.")
    lines += ["", "Per exclusion file (files can overlap):", ""]
    for source in audit["exclusion_sources"]:
        n = sum(source["path"] in r["reasons"] for r in audit["excluded"])
        lines.append(f"- `{source['path']}`: {n} matching excluded rows.")
    lines += ["", "```bash", "python tools/cc_pairs.py candidates", "sbatch scripts/cc_probe_hpg.slurm",
              "# rai fallback, from this checkout (same runner; choose an available GPU):",
              "CUDA_VISIBLE_DEVICES=0 CC_PYTHON=.venv/bin/python bash scripts/cc_probe_hpg.slurm", "```", "",
              "The runner serves the base model, generates, stops its own server, computes native likelihoods, then selects. GPU work has not been launched by candidate construction."]
    if split.get("within_fraction") == 0:
        train_ids = set(split["train"])
        train = [p for p in bundle["pairs"] if p["pair_id"] in train_ids]
        train_counts = counts(train)
        composition = Counter((p["seed_function"], p["type"]) for p in train)
        lines += ["", "## Train composition (C19f)", "",
                  f"Split counts: train **{split['counts']['train']}** / heldout_seed **{split['counts']['heldout_seed']}** "
                  f"(`{split['heldout_seed_function']}`); heldout_within_seed **0**; excluded_cross_partition **0**.", "",
                  "| Train seed function | binding | multi_turn | call_vs_abstain | Total |",
                  "|---|---:|---:|---:|---:|"]
        for group, n in train_counts["by_seed_function"].items():
            lines.append(f"| {group} | " + " | ".join(str(composition[group, t]) for t in TYPES) + f" | {n} |")
        lines.append("| Total | " + " | ".join(str(train_counts["by_type"][t]) for t in TYPES)
                     + f" | {train_counts['total']} |")
    (out / "report.md").write_text("\n".join(lines) + "\n")


class LocalEvalClient:
    """Greedy native generation using the same Student wrapper as the probes."""

    def __init__(self, student, *, max_tokens=512, context_limit=32768, device="cuda"):
        self.student, self.max_tokens = student, max_tokens
        self.context_limit, self.device = context_limit, device

    def generate(self, prompt, *, seed, temperature):
        import torch
        if temperature != 0:
            raise ValueError("evaluation only supports greedy generation")
        tok = self.student.tokenizer
        tokens = tok(thinking_off(prompt), add_special_tokens=False)["input_ids"]
        if not tokens or len(tokens) + self.max_tokens > self.context_limit:
            raise ValueError("generation exceeds context limit; refusing truncation")
        ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        self.student.model.eval()
        with torch.inference_mode():
            output = self.student.generate(ids, torch.ones_like(ids), dict(max_new_tokens=self.max_tokens))
        response = output[0, len(tokens):].tolist()
        return dict(output=tok.decode(response, skip_special_tokens=False),
                    finish_reason="stop" if response and response[-1] == tok.eos_token_id else "length",
                    request_seed=seed, temperature=temperature,
                    usage=dict(prompt_tokens=len(tokens), completion_tokens=len(response)))


def load_eval_student(model, seed):
    import torch
    import appworld_train as trainer
    from tools.behavior_atom.gpu_driver import Student
    if not torch.cuda.is_available():
        raise ValueError("local model evaluation requires CUDA")
    return Student(trainer.build_base_model(str(model), seed), trainer.load_tokenizer(str(model)),
                   "unused", None, None)


def score_served_teacher_texts(pairs, client, tokenizer, *, context_limit=32768):
    """Full response + EOS likelihood, using vLLM echoed prompt logprobs.

    Tokenize prompt and response separately, exactly as score_teacher_texts.
    The server computes logprobs; unlike the native path this is not native CE.
    https://docs.vllm.ai/en/v0.15.0/api/vllm/entrypoints/openai/completion/protocol/
    """
    import numpy as np
    if tokenizer.eos_token_id is None:
        raise ValueError("EOS is required for margin likelihoods")
    scores = {}
    for key, job in teacher_score_jobs(pairs).items():
        prompt = tokenizer(thinking_off(job["prompt"]), add_special_tokens=False)["input_ids"]
        response = tokenizer(job["y_plus"], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        tokens = prompt + response
        if not prompt or len(tokens) >= context_limit:
            # vLLM's echo-only implementation internally reserves one decode token.
            raise ValueError("teacher-forced sequence exceeds server context limit; refusing truncation")
        payload = dict(model=client.model, prompt=tokens, echo=True, logprobs=1,
                       max_tokens=0, temperature=0, seed=0, add_special_tokens=False,
                       repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0)
        req = urllib.request.Request(client.url + "/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as reply:
            result = json.load(reply)
        values = (result["choices"][0].get("logprobs") or {}).get("token_logprobs")
        if not isinstance(values, list) or len(values) != len(tokens):
            raise ValueError("server must return echo logprobs for every prompt token without truncation")
        values = values[len(prompt):]
        if any(type(v) not in (float, int) or not math.isfinite(v) for v in values):
            raise ValueError("missing/nonfinite teacher-forced server logprobs")
        logp = float(np.asarray(values, dtype=np.float32).sum(dtype=np.float32))
        if not math.isfinite(logp):
            raise ValueError("nonfinite teacher-forced likelihood")
        scores[key] = dict(logp=logp, counts=dict(prompt_tokens=len(prompt), prompt_tokens_dropped=0,
                           plus=dict(effective_loss_tokens=len(response), response_tokens_dropped=0)))
    return scores


EVAL_PARTITIONS = ("train", "heldout_seed", "heldout_within_seed")


def load_eval_inputs(pairs_path, split_path, base_path):
    """Match C19 probes by pair and state identity, independent of training targets."""
    pairs_path = Path(pairs_path)
    pairs = read_jsonl(pairs_path) if pairs_path.suffix == ".jsonl" else json.loads(pairs_path.read_text())
    if isinstance(pairs, dict):
        if pairs.get("status", "complete") != "complete":
            raise ValueError("evaluation requires a completed pair selection")
        pairs = next((pairs[k] for k in ("pairs", "confused_pairs", "pair_ids", "confused_pair_ids") if k in pairs), None)
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("evaluation requires a nonempty list of C19 pairs")
    if all(isinstance(p, str) for p in pairs):
        candidates = read_jsonl(pairs_path.with_name("candidates.jsonl"))
        lookup = {p["pair_id"]: p for p in candidates}
        if len(lookup) != len(candidates) or set(pairs) - lookup.keys():
            raise ValueError("duplicate or missing candidate IDs")
        pairs = [lookup[pid] for pid in pairs]
    by_id = {p["pair_id"]: p for p in pairs}
    if len(by_id) != len(pairs):
        raise ValueError("duplicate evaluation pair IDs")
    split = json.loads(Path(split_path).read_text())
    partitions = {}
    for name in (*EVAL_PARTITIONS, "excluded_cross_partition"):
        variants = [split[k] for k in (name, name + "_pair_ids") if k in split]
        variants = [v.get("pair_ids") if isinstance(v, dict) else v for v in variants]
        if not variants and name in ("train", "heldout_seed"):
            raise ValueError(f"split requires explicit {name} pair IDs")
        ids = variants[0] if variants else []
        if (not isinstance(ids, list) or not all(isinstance(i, str) for i in ids)
                or len(ids) != len(set(ids)) or any(v != ids for v in variants)):
            raise ValueError(f"duplicate or conflicting {name} split IDs")
        if set(ids) - by_id.keys():
            raise ValueError(f"unknown {name} split IDs")
        for pid in ids:
            if pid in partitions:
                raise ValueError(f"split pair ID overlap: {pid}")
            partitions[pid] = name
    if set(by_id) != set(partitions):
        raise ValueError("split must account for every selected pair")
    pairs = [p for p in pairs if partitions[p["pair_id"]] in EVAL_PARTITIONS]
    if not pairs:
        raise ValueError("no pairs in evaluation partitions")
    base_rows = read_jsonl(base_path)
    base = {r["pair_id"]: r for r in base_rows}
    if len(base) != len(base_rows):
        raise ValueError("duplicate base probe pair IDs")
    states, prompts, versions = {}, defaultdict(set), set()
    for p in pairs:
        pid, partition = p["pair_id"], partitions[p["pair_id"]]
        if p.get("split", partition) != partition:
            raise ValueError(f"{pid}: conflicting pair split metadata")
        r = base.get(pid, {})
        if r.get("status") != "complete":
            raise ValueError(f"{pid}: missing or incomplete base probe result")
        for label in ("s1", "s2"):
            side, previous = p[label], r.get(label)
            if not isinstance(previous, dict):
                raise ValueError(f"{pid}: missing {label} base probe outcome")
            if previous.get("state_id") is not None and previous["state_id"] != side["state_id"]:
                raise ValueError(f"{pid}: base state ID mismatch")
            prompt_sha256 = hashlib.sha256(side["prompt"].encode()).hexdigest()
            if previous.get("prompt_sha256") is not None:
                matches_prompt = previous["prompt_sha256"] == prompt_sha256
            else:
                # Historical probes stored only the content-derived state ID.
                # Recompute it so a changed prompt cannot reuse a saved ID/hash.
                matches_prompt = previous.get("state_id") == "state_" + digest(
                    [side["prompt"], side["ground_truth"]])[:20]
            if not matches_prompt or side.get("prompt_sha256", prompt_sha256) != prompt_sha256:
                raise ValueError(f"{pid}: stale {label} base probe prompt identity")
            if previous["teacher_check"]["outcome"] != 1:
                raise ValueError(f"{pid}: invalid base teacher target")
            kappa = previous.get("kappa")
            if type(kappa) not in (int, float) or not math.isfinite(kappa):
                raise ValueError(f"{pid}: missing/nonfinite base kappa")
            greedy = base_greedy(previous)
            versions.add(greedy["checker_version"])
            identity = dict(side={k: side[k] for k in ("prompt", "teacher_text", "function", "ground_truth",
                                                     "category", "out_of_scope")},
                            greedy={k: greedy.get(k) for k in ("output", "outcome", "calls", "checker_version",
                                                             "temperature", "request_seed", "finish_reason")})
            sid = side["state_id"]
            if sid in states and states[sid] != identity:
                raise ValueError(f"{sid}: inconsistent repeated state or base greedy result")
            states[sid] = identity
            prompts[digest(side["prompt"])].add(partition)
    if any(len(v) > 1 for v in prompts.values()):
        raise ValueError("state prompt overlap across evaluation partitions")
    held_seeds = {p["seed_function"] for p in pairs if partitions[p["pair_id"]] == "heldout_seed"}
    if any(p["seed_function"] in held_seeds and partitions[p["pair_id"]] != "heldout_seed" for p in pairs):
        raise ValueError("heldout_seed function overlaps another partition")
    if len(versions) != 1:
        raise ValueError("base probe uses inconsistent checker versions")
    return pairs, partitions, base


def base_greedy(side):
    outputs = [o for o in side.get("outputs", []) if o.get("mode") == "greedy"]
    if len(outputs) != 1 or outputs[0].get("outcome") not in (0, 1):
        raise ValueError("base probe requires exactly one checked greedy output per state")
    if outputs[0]["outcome"] != side.get("greedy_outcome") or outputs[0].get("temperature") != 0:
        raise ValueError("inconsistent base greedy outcome or temperature")
    return outputs[0]


def emitted_call(output):
    # Count attempted malformed tool calls too; parser failures are not abstentions.
    return bool(output.get("calls")) or "<tool_call>" in output["output"] or "</tool_call>" in output["output"]


def assemble_eval_results(pairs, partitions, base, generations, scores):
    rows = assemble_results(pairs, generations, scores)
    for p, row in zip(pairs, rows):
        previous = base[p["pair_id"]]
        row.update(partition=partitions[p["pair_id"]], both_sides_correct=row["both_greedy_correct"],
                   base_both_sides_correct=all(previous[s]["greedy_outcome"] == 1 for s in ("s1", "s2")))
        for label in ("s1", "s2"):
            side = row[label]
            now, before = side["outputs"][0], base_greedy(previous[label])
            if side["teacher_check"]["outcome"] != 1:
                raise ValueError("teacher target failed current checker")
            if now["checker_version"] != before["checker_version"]:
                raise ValueError("evaluation checker version differs from base probe")
            if type(side["kappa"]) not in (int, float) or not math.isfinite(side["kappa"]):
                raise ValueError("missing/nonfinite evaluation kappa")
            side.update(task_id=p[label]["task_id"], kind="should_abstain" if p[label]["out_of_scope"] else "should_call",
                        outcome=now["outcome"], base_outcome=before["outcome"],
                        repaired=before["outcome"] == 0 and now["outcome"] == 1,
                        damaged=before["outcome"] == 1 and now["outcome"] == 0,
                        emitted_call=emitted_call(now), base_emitted_call=emitted_call(before),
                        base_kappa=previous[label]["kappa"],
                        kappa_flipped=previous[label]["kappa"] > 0 and side["kappa"] <= 0)
        row["base_kappa_positive"] = any(row[s]["base_kappa"] > 0 for s in ("s1", "s2"))
        # A pair is resolved only when neither side prefers the opposite target.
        row["kappa_flipped"] = row["base_kappa_positive"] and all(row[s]["kappa"] <= 0 for s in ("s1", "s2"))
        row["any_side_kappa_flipped"] = any(row[s]["kappa_flipped"] for s in ("s1", "s2"))
    return rows


def fraction(n, d):
    return n / d if d else None


def eval_metrics(rows):
    sides = [r[s] for r in rows for s in ("s1", "s2")]
    unique = {s["state_id"]: s for s in sides}

    def side_metrics(items):
        positive = sum(s["base_kappa"] > 0 for s in items)
        flipped = sum(s["kappa_flipped"] for s in items)
        return dict(sides=len(items), correct=sum(s["outcome"] for s in items),
                    base_correct=sum(s["base_outcome"] for s in items),
                    repaired=sum(s["repaired"] for s in items), damaged=sum(s["damaged"] for s in items),
                    base_kappa_positive=positive, kappa_flipped=flipped, kappa_flip_rate=fraction(flipped, positive))

    call_rates = {}
    for kind in ("overall", "should_call", "should_abstain"):
        items = [s for s in unique.values() if kind == "overall" or s["kind"] == kind]
        calls, base_calls = sum(s["emitted_call"] for s in items), sum(s["base_emitted_call"] for s in items)
        call_rates[kind] = dict(states=len(items), calls=calls, base_calls=base_calls,
                                call_rate=fraction(calls, len(items)), base_call_rate=fraction(base_calls, len(items)),
                                delta=fraction(calls - base_calls, len(items)))
    n = len(rows)
    correct, base_correct = sum(r["both_sides_correct"] for r in rows), sum(r["base_both_sides_correct"] for r in rows)
    positive, flipped = sum(r["base_kappa_positive"] for r in rows), sum(r["kappa_flipped"] for r in rows)
    return dict(pairs=n, both_sides_correct=correct, both_sides_correct_rate=fraction(correct, n),
                base_both_sides_correct=base_correct, base_both_sides_correct_rate=fraction(base_correct, n),
                base_kappa_positive_pairs=positive, kappa_flipped_pairs=flipped,
                kappa_flip_rate=fraction(flipped, positive), kappa_flip_fraction_all_pairs=fraction(flipped, n),
                any_side_kappa_flipped_pairs=sum(r["any_side_kappa_flipped"] for r in rows),
                sides=side_metrics(sides), by_side={s: side_metrics([r[s] for r in rows]) for s in ("s1", "s2")},
                unique_states=dict(states=len(unique), correct=sum(s["outcome"] for s in unique.values()),
                                   base_correct=sum(s["base_outcome"] for s in unique.values()),
                                   repaired=sum(s["repaired"] for s in unique.values()),
                                   damaged=sum(s["damaged"] for s in unique.values())), call_rates=call_rates)


def summarize_eval(rows):
    types = sorted({r["type"] for r in rows})

    def group(items):
        return dict(**eval_metrics(items), by_type={t: eval_metrics([r for r in items if r["type"] == t]) for t in types})

    return dict(overall=group(rows), partitions={p: group([r for r in rows if r["partition"] == p])
                                               for p in EVAL_PARTITIONS})


def write_eval_report(out, summary):
    def rate(value):
        return "N/A" if value is None else f"{value:.1%}"

    lines = [f"# Condition-contrast evaluation: {summary['tag']}", "",
             f"Model: `{summary['model']}`. Greedy, thinking off, max output {summary['max_tokens']} tokens.",
             f"Likelihood backend: `{summary['likelihood_backend']}`; full response + EOS, separate prompt/response tokenization, no truncation.",
             "", "κ(s1)=logπ(a2|s1)−logπ(a1|s1); κ(s2)=logπ(a1|s2)−logπ(a2|s2). Positive means confusion.",
             "Pair κ flip: max(base κ1, base κ2)>0 → max(current κ1, current κ2)≤0. "
             "The conditional flip rate divides by base-positive pairs; the all-pair fraction divides by every pair. "
             "Per-side flips and any-side pair flips are also in JSON. Empty denominators are N/A (JSON null).",
             "Repairs/damage compare greedy AST outcomes, never the base's four samples. "
             "Side counts include pair occurrences; unique-state counts deduplicate state_id. "
             "Call rates also deduplicate states within each table group and count malformed tool-call attempts as calls.",
             "Heldout_seed isolates an entire seed function; heldout_within_seed, when present, isolates states within other seeds.",
             "These are condition-pair metrics; official BFCL v4 task repair/damage and axis scores require the separate official campaign."]
    for partition, metrics in [("overall", summary["overall"]), *summary["partitions"].items()]:
        groups = [("all", metrics), *metrics["by_type"].items()]
        lines += ["", f"## {partition}", "", "| Type | Pairs | Both correct (base → model) | Repaired / damaged sides | Repaired / damaged states | κ flips / base-positive | κ flip rate | κ flips / all pairs |",
                  "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
        for kind, m in groups:
            lines.append(f"| {kind} | {m['pairs']} | {rate(m['base_both_sides_correct_rate'])} → {rate(m['both_sides_correct_rate'])} | "
                         f"{m['sides']['repaired']} / {m['sides']['damaged']} | {m['unique_states']['repaired']} / {m['unique_states']['damaged']} | "
                         f"{m['kappa_flipped_pairs']} / {m['base_kappa_positive_pairs']} | {rate(m['kappa_flip_rate'])} | {rate(m['kappa_flip_fraction_all_pairs'])} |")
        lines += ["", "| Type | State kind | States | Base calls | Model calls | Base call rate | Model call rate | Δ call rate |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for kind, m in groups:
            for state_kind, c in m["call_rates"].items():
                lines.append(f"| {kind} | {state_kind} | {c['states']} | {c['base_calls']} | {c['calls']} | "
                             f"{rate(c['base_call_rate'])} | {rate(c['call_rate'])} | {rate(c['delta'])} |")
    (Path(out) / "report.md").write_text("\n".join(lines) + "\n")


def run_evaluate(args):
    from tools.behavior_atom.checker_bridge import CheckerBridge
    pairs, partitions, base = load_eval_inputs(args.pairs, args.split, args.base_results)
    tag = args.tag
    if args.model:
        model_path = args.model.expanduser().resolve()
        if not (model_path / "config.json").is_file():
            raise ValueError(
                "--model must be a merged HF model directory containing config.json. "
                f"Tried directories: {model_path}. "
                "Merge adapter exports with tools/bfcl_hub_merge_export.py before evaluation.")
        tag = tag or (model_path.parent.name if model_path.name in ("hub_merged", "adapter") else model_path.name)
        model = str(model_path)
    else:
        client = VLLMClient(args.base_url, args.max_tokens)
        card = client.select_model(args.served_model)
        model = client.model
        hint = Path(card.get("root") or model)
        tag = tag or (hint.parent.name if hint.name in ("hub_merged", "adapter") else Path(model).name)
    if tag is None or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", tag):
        raise ValueError("cannot infer a safe output tag; specify --tag using letters, digits, dots, dashes or underscores")
    out = args.out or ROOT / "results/cc_eval" / tag
    # Fail before model loading or generation if the baseline checker changed.
    with CheckerBridge() as checker:
        if any(base_greedy(base[p["pair_id"]][s])["checker_version"] != checker.checker_version
               for p in pairs for s in ("s1", "s2")):
            raise ValueError("evaluation checker version differs from base probe")
        metadata = dict(tag=tag, model=model, pairs_path=str(args.pairs), split_path=str(args.split),
                        base_results_path=str(args.base_results), pairs_sha256=digest(pairs),
                        split_sha256=digest(json.loads(args.split.read_text())),
                        base_results_sha256=hashlib.sha256(args.base_results.read_bytes()).hexdigest(),
                        base_probe_state_identity_matches=[
                            dict(pair_id=p["pair_id"], base_candidate_sha256=base[p["pair_id"]].get("candidate_sha256"),
                                 candidate_sha256=digest(p))
                            for p in pairs if base[p["pair_id"]].get("candidate_sha256") != digest(p)],
                        checker_version=checker.checker_version, seed=args.seed, temperature=0, thinking=False,
                        max_tokens=args.max_tokens, context_limit=args.context_limit, new_teacher_tokens=0,
                        excluded_cross_partition=sum(p == "excluded_cross_partition" for p in partitions.values()))
        # Mark an interrupted rerun explicitly; never leave old successful reports.
        write_json(out / "summary.json", dict(**metadata, status="running"))
        write_jsonl(out / "pair_eval.jsonl", [])
        (out / "report.md").write_text(f"# {tag}\n\nEvaluation running; results are incomplete.\n")
        if args.model:
            student = load_eval_student(model_path, args.seed)
            client = LocalEvalClient(student, max_tokens=args.max_tokens, context_limit=args.context_limit)
            metadata["likelihood_backend"] = "native CE; fp32 sum (probe implementation)"
        else:
            from transformers import AutoTokenizer
            tokenizer_path = args.tokenizer or card.get("root") or card["id"]
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)
            metadata.update(model=client.model, model_card=card, base_url=args.base_url, tokenizer=str(tokenizer_path),
                            likelihood_backend="vLLM echo prompt logprobs; fp32 sum (server numerics differ from native CE)")

        def progress(n, total, records):
            print(f"[cc evaluate {tag}] generated and checked {n}/{total} states", flush=True)

        generations = generate_states(pairs, client, checker, seed=args.seed, samples=0, progress=progress)
        print(f"[cc evaluate {tag}] scoring full teacher responses and EOS", flush=True)
        if args.model:
            scores = score_teacher_texts(pairs, student, seed=args.seed, context_limit=args.context_limit)
        else:
            scores = score_served_teacher_texts(pairs, client, tokenizer, context_limit=args.context_limit)
    rows = assemble_eval_results(pairs, partitions, base, generations, scores)
    summary = dict(**metadata, status="complete", **summarize_eval(rows))
    write_jsonl(out / "pair_eval.jsonl", rows)
    write_eval_report(out, summary)
    write_json(out / "summary.json", summary)
    print(f"[cc evaluate {tag}] complete: {len(rows)} pairs → {out}", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("candidates", "probe", "select"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, default=ROOT / "data/cc_pairs_v1")
        p.add_argument("--seed", type=int, default=0)
        if name == "candidates":
            p.add_argument("--root", type=Path, default=ROOT)
            p.add_argument("--exclude-probe-parents", action="store_true",
                           help="also exclude siblings of active probes by parent task (default: OFF)")
        elif name == "probe":
            p.add_argument("--phase", choices=("generate", "score", "all"), default="all",
                           help="all requires enough GPU memory beside the external vLLM server; runner uses separate phases")
            p.add_argument("--base-url", default="http://127.0.0.1:8976/v1")
            p.add_argument("--context-limit", type=int, default=32768)
            p.add_argument("--max-tokens", type=int, default=512)
        else:
            p.add_argument("--target", "--limit", dest="limit", type=int, default=32, metavar="TARGET_PAIRS",
                           help="maximum selected pairs (default: 32; --limit is a compatibility alias)")
            p.add_argument("--type3-cap", type=float, default=DEFAULT_TYPE3_CAP, metavar="FRACTION",
                           help="maximum call_vs_abstain fraction (default: 0.3333, historical exact one third; 1.0 disables)")
            p.add_argument("--min-per-type", type=int, metavar="COUNT",
                           help="report selected-pair shortfalls against this count per type; does not affect selection")
            p.add_argument("--probe-from", type=Path, default=ROOT / "data/cc_pairs_v1", metavar="DIR",
                           help="read all probe inputs here if --out lacks probe_results.jsonl (default: data/cc_pairs_v1)")
            p.add_argument("--heldout-seed")
            p.add_argument("--within-fraction", type=float, default=0.2,
                           help="within-seed holdout fraction in [0, 1); 0 holds out only the whole seed function")
    p = sub.add_parser("evaluate", help="greedy paired-state evaluation against the base probe")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=Path, help="merged HF directory containing config.json (merge adapter exports first)")
    source.add_argument("--base-url", help="served vLLM /v1 URL, with echo prompt logprobs support")
    p.add_argument("--served-model", help="model ID when the server lists more than one model")
    p.add_argument("--tokenizer", help="HF tokenizer path/ID for remote scoring; defaults to served model root")
    p.add_argument("--tag", help="output tag; defaults to the local checkpoint tag or served model name")
    p.add_argument("--out", type=Path, help="output directory (default: results/cc_eval/<tag>)")
    p.add_argument("--pairs", type=Path, default=ROOT / "data/cc_pairs_v1_stage1/confused_pairs.json")
    p.add_argument("--split", type=Path, default=ROOT / "data/cc_pairs_v1_stage1/split.json")
    p.add_argument("--base-results", type=Path, default=ROOT / "data/cc_pairs_v1/probe_results.jsonl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--context-limit", type=int, default=32768)
    args = ap.parse_args(argv)
    if args.command == "evaluate":
        if args.max_tokens < 1 or args.context_limit < 1:
            ap.error("token limits must be positive")
        if args.model and (args.served_model or args.tokenizer):
            ap.error("--served-model and --tokenizer apply only to --base-url")
        return run_evaluate(args)
    if args.command == "select":
        if args.limit < 1:
            ap.error("--target must be positive")
        if not math.isfinite(args.type3_cap) or not 0 <= args.type3_cap <= 1:
            ap.error("--type3-cap must be a finite fraction between zero and one")
        if args.min_per_type is not None and args.min_per_type < 0:
            ap.error("--min-per-type must be nonnegative")
        if not 0 <= args.within_fraction < 1:
            ap.error("--within-fraction must be at least zero and less than one")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.command == "candidates":
        root = args.root
        raw = [(str(p.relative_to(root)), read_jsonl(p)) for p in
               (root / "data/bfcl_sft/gen_pool_v3.jsonl", root / "data/bfcl_sft/gen_oos.jsonl")]
        pool_path = root / "data/bfcl_sft/pool_events_pref_v3t.jsonl"
        pairs, todo, audit = build_candidates(raw, (str(pool_path.relative_to(root)), read_jsonl(pool_path)),
                                             load_exclusions(root, exclude_probe_parents=args.exclude_probe_parents), seed=args.seed)
        ledger = root / LEDGER
        audit["ledger_sha256"] = hashlib.sha256(ledger.read_bytes()).hexdigest()
        audit["input_sha256"] = {path: digest(rows) for path, rows in [*raw, (str(pool_path.relative_to(root)), read_jsonl(pool_path))]}
        from tools.behavior_atom.checker_bridge import CheckerBridge
        with CheckerBridge() as checker:
            validation = validate_teachers(pairs, checker)
        audit["teacher_validation"] = {k: v for k, v in validation.items() if k != "states"}
        results_path = out / "probe_results.jsonl"
        if results_path.exists():
            audit["preserved_probe_results"] = dict(sha256=hashlib.sha256(results_path.read_bytes()).hexdigest(),
                                                     status="stale_requires_probe_rerun")
        else:
            write_jsonl(results_path, [])
        write_json(out / "teacher_validation.json", validation)
        write_jsonl(out / "candidates.jsonl", pairs)
        write_jsonl(out / "candidates_type2_todo.jsonl", todo)
        write_json(out / "candidate_audit.json", audit)
        # Preserve historical result bytes, but invalidate their selection metadata.
        bundle, split = dict(status="pending_probe", pairs=[], eligible=None, selected=None, new_teacher_tokens=0), dict(status="pending_probe")
        for name in ("probe_generations.json", "probe_metadata.json", "probe_server.json"):
            (out / name).unlink(missing_ok=True)
        write_json(out / "probe_metadata.json", dict(status="pending_probe", candidates_sha256=digest(pairs)))
    else:
        source = (args.probe_from if args.command == "select" and not (out / "probe_results.jsonl").exists()
                  else out)
        pairs = read_jsonl(source / "candidates.jsonl")
        audit = json.loads((source / "candidate_audit.json").read_text())
        if args.seed != audit["seed"]:
            raise ValueError("seed must match candidate construction")
        if args.command == "probe":
            from tools.behavior_atom.checker_bridge import CheckerBridge
            from bfas.behavior.margin import LIKELIHOOD
            stamp = dict(candidates_sha256=digest(pairs), model=BASE_MODEL, seed=args.seed,
                         greedy=1, samples=4, temperature=0.7, thinking=False, max_tokens=args.max_tokens,
                         context_limit=args.context_limit, likelihood_definition=LIKELIHOOD, new_teacher_tokens=0)
            if args.max_tokens < 1 or args.context_limit < 1:
                raise ValueError("token limits must be positive")
            for side in {s["state_id"]: s for p in pairs for s in (p["s1"], p["s2"])}.values():
                if render_prompt(side["question"], side["function"]) != side["prompt"]:
                    raise ValueError("candidate prompt differs from the current BFCLAdapter rendering")
            if args.phase in {"generate", "all"}:
                # An interrupted re-probe must not leave a previous completed
                # selection or metadata looking current.
                write_json(out / "probe_metadata.json", dict(**stamp, status="running"))
                write_jsonl(out / "probe_results.jsonl", [])
                pending, pending_split = dict(status="pending_probe", pairs=[], eligible=None, selected=None), dict(status="pending_probe")
                write_json(out / "confused_pairs.json", pending)
                write_json(out / "split.json", pending_split)
                write_report(out, audit, pending, pending_split)
                def progress(n, total, records):
                    print(f"[cc probe] generated and checked {n}/{total} states", flush=True)
                    write_json(out / "probe_generations.json", dict(metadata=stamp, states=records))
                client = VLLMClient(args.base_url, args.max_tokens)
                server_model = client.validate_base() if pairs else None
                write_json(out / "probe_server.json", dict(base_url=args.base_url, model_card=server_model))
                with CheckerBridge() as checker:
                    generations = generate_states(pairs, client, checker,
                                                  seed=args.seed, progress=progress)
                write_json(out / "probe_generations.json", dict(metadata=stamp, states=generations))
                write_jsonl(out / "probe_results.jsonl", assemble_results(pairs, generations))
            else:
                cached = json.loads((out / "probe_generations.json").read_text())
                if cached["metadata"] != stamp:
                    raise ValueError("generation metadata mismatch")
                generations = cached["states"]
            if args.phase in {"score", "all"}:
                scores = {}
                if pairs:
                    import torch
                    import appworld_train as trainer
                    from tools.behavior_atom.gpu_driver import Student
                    if not torch.cuda.is_available():
                        raise ValueError("native base likelihood scoring requires a GPU")
                    student = Student(trainer.build_base_model(BASE_MODEL, args.seed), trainer.load_tokenizer(BASE_MODEL),
                                      "unused", None, None)
                    scores = score_teacher_texts(pairs, student, seed=args.seed, context_limit=args.context_limit)
                write_jsonl(out / "probe_results.jsonl", assemble_results(pairs, generations, scores))
            write_json(out / "probe_metadata.json", dict(**stamp, status="complete" if args.phase != "generate" else "generated"))
            print(f"[cc probe] {args.phase} finished; run select after scoring", flush=True)
            return 0
        metadata = json.loads((source / "probe_metadata.json").read_text())
        if metadata.get("status") != "complete" or metadata.get("candidates_sha256") != digest(pairs):
            raise ValueError("selection requires a complete probe of these candidates")
        bundle, split = select_pairs(pairs, read_jsonl(source / "probe_results.jsonl"), seed=args.seed,
                                     limit=args.limit, heldout_seed=args.heldout_seed, within_fraction=args.within_fraction,
                                     type3_cap=args.type3_cap, min_per_type=args.min_per_type)
        if source.resolve() != out.resolve():
            bundle["probe_from"] = str(source.resolve())
    write_json(out / "confused_pairs.json", bundle)
    write_json(out / "split.json", split)
    write_report(out, audit, bundle, split)
    print(json.dumps(dict(status=bundle["status"], candidates=audit["candidates"],
                          type2_todo=audit["type2_todo"], selected=bundle.get("selected")), indent=2))
    return 0
