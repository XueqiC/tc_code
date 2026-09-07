"""C16: archived checker-verified text margins, with exactly three diagnostics.

JSON null is explicit NA throughout; an empty decoded string is a real text.
CLI paths in a config are relative to that config. No ground-truth text is ever
used as a likelihood target. Sampling is opt-in on the command line, base only.
The score stage needs one GPU; pairs and analyze are CPU-only by default.

  python tools/behavior_atom_experiment.py margin --step pairs
  python tools/behavior_atom_experiment.py margin --step score
  python tools/behavior_atom_experiment.py margin --step analyze

The default config is configs/behavior_atom/margin_v1.json. --step all (the
default) runs these in order. --run-dirs overrides the scored runs only, never
the registered decode archive. --dry-run does not load torch or write files.
Optional --step pairs --sample-base 8 writes base_samples_v1.json alongside
the pairs, filling archive-missing sides only. Then run score and analyze to
retain those supplemental pairs. Running pairs/all without the flag rebuilds
the archive-only selection. No sampling is enabled by config or by Slurm.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EVALUATION_GLOBS = (
    "collect_r2/shard-*/sources/*/**/evaluations.json",
    "collect_r2/shard-00000-of-00003/base/evaluations.json",
    "joint_call_r1/sources/*/**/evaluations.json",
    "joint_abstain_r1/sources/*/**/evaluations.json",
    "halfdose_r1/sources/*/**/evaluations.json",
)
SINGLES = ("src_17", "src_14", "src_04", "src_09")
Q2_PAIRS = (("joint_src_17_abstain_anchor", "half_src_17"),
            ("joint_src_04_anchor_match", "half_src_04"))
VERDICT_RULES = (
    "仍不超过公共模式 × 幅度 → 结束编码路线",
    "只改善 likelihood 预测 → 局部几何结果,不重启原子主方法",
    "连续响应解释了特定组合并给出可预先指定的新组合预测 → 再考虑一个小型前瞻验证,不恢复 P3–P5",
)
LIKELIHOOD = "full_sequence; native CE; fp32 reduction; EOS included"


def _read(path):
    return json.loads(Path(path).read_text())


def _hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def _clean(value):
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _relative(path, root):
    return os.path.relpath(Path(path).resolve(), Path(root).resolve())


def _source_id(directory, fallback):
    summary = Path(directory) / "summary.json"
    return str(_read(summary).get("source_id", fallback)) if summary.is_file() else fallback


def _checked(record):
    # Driver only assigns 0/1 after parsing/checking. A failed check is NA,
    # never a negative verdict. Unrelated policy/likelihood errors are harmless.
    fatal = {"generate", "parse", "checker"}
    kinds = [e.get("kind") if isinstance(e, dict) else str(e).split(":", 1)[0]
             for e in record.get("errors", [])]
    return (record.get("outcome") in (0, 1) and isinstance(record.get("output"), str)
            and not fatal.intersection(kinds))


def build_pairs(probes_path, results_root, *, samples=None):
    """Select base text if eligible, else the mode (lexical tie break).

    All occurrences of the chosen text retain provenance. Samples only fill
    archive-missing sides, so adding them never changes an archived target.
    """
    raw = _read(probes_path)
    probes = raw["probes"] if isinstance(raw, dict) else raw
    by_id = {p["probe_id"]: p for p in probes}
    if len(by_id) != len(probes):
        raise ValueError("duplicate probe_id")
    paths = sorted({p for pattern in EVALUATION_GLOBS for p in Path(results_root).glob(pattern)})
    if not paths:
        raise ValueError("no evaluations.json in the registered archive paths")
    candidates, base_outcomes, archived_outcomes = defaultdict(list), {}, defaultdict(dict)
    rejected, inputs = 0, []
    for path in paths:
        is_base = path.parent.name == "base"
        sid = "base" if is_base else _source_id(path.parent, path.parent.name)
        origin = dict(state_id=sid, origin="archive", evaluations=_relative(path, results_root))
        inputs.append(dict(path=origin["evaluations"], sha256=_hash(path)))
        for index, record in enumerate(_read(path)):
            pid = record["probe_id"]
            if pid not in by_id:
                raise ValueError(f"unknown archived probe_id: {pid}")
            if not _checked(record):
                rejected += 1
                continue
            outcome = int(record["outcome"])
            if pid in archived_outcomes[sid] and archived_outcomes[sid][pid] != outcome:
                raise ValueError(f"conflicting archived outcomes: {sid}/{pid}")
            archived_outcomes[sid][pid] = outcome
            if is_base:
                base_outcomes[pid] = outcome
            candidates[pid, outcome].append((record["output"], {**origin, "row": index, "outcome": outcome}))
    sample_candidates = defaultdict(list)
    sample_records = [] if samples is None else samples["records"]
    for index, record in enumerate(sample_records):
        if record.get("state_id") != "base_sample" or record["probe_id"] not in by_id:
            raise ValueError("supplemental records must be registered probes from base_sample")
        if _checked(record):
            if not record.get("checker_version"):
                raise ValueError("base_sample requires checker provenance")
            sample_candidates[record["probe_id"], record["outcome"]].append((record["output"], dict(
                state_id="base_sample", origin="base_sample", row=index, sample_index=record["sample_index"],
                outcome=record["outcome"], checker_version=record["checker_version"])))

    def choose(items):
        if not items:
            return None, None
        counts = Counter(text for text, _ in items)
        base = [text for text, prov in items if prov["state_id"] == "base"]
        text = min(base) if base else min(counts, key=lambda t: (-counts[t], t))
        return text, [prov for candidate, prov in items if candidate == text]

    pairs, archive_both, supplemented = [], 0, []
    for pid, probe in by_id.items():
        kind = probe["kind"]
        if kind not in {"should_call", "should_abstain"}:
            raise ValueError(f"unknown probe kind: {kind}")
        pair = dict(probe_id=pid, prompt=probe["prompt"], kind=kind,
                    prompt_sha256=_digest(probe["prompt"]), base_outcome=base_outcomes.get(pid))
        archive_complete = all(candidates[pid, outcome] for outcome in (1, 0))
        archive_both += int(archive_complete)
        for side, outcome in (("plus", 1), ("minus", 0)):
            items = candidates[pid, outcome]
            if not items and sample_candidates[pid, outcome]:
                items = sample_candidates[pid, outcome]
                supplemented.append(dict(probe_id=pid, side=side))
            pair[f"y_{side}"], pair[f"y_{side}_provenance"] = choose(items)
        pair["missing_sides"] = [s for s in ("plus", "minus") if pair[f"y_{s}"] is None]
        pair["archive_both_sides"] = archive_complete
        pairs.append(pair)
    return dict(version="behavior-probe-pairs-v1", na_representation="JSON null (NA)",
                selection="base when eligible; else highest frequency; lexical tie break; samples fill missing only",
                probes_sha256=_hash(probes_path), archive_inputs=inputs, probes=pairs,
                archived_outcomes=dict(archived_outcomes),
                coverage=dict(probes=len(pairs), archive_both_sides=archive_both,
                              both_sides=sum(not p["missing_sides"] for p in pairs),
                              missing_plus=sum(p["y_plus"] is None for p in pairs),
                              missing_minus=sum(p["y_minus"] is None for p in pairs),
                              rejected_archive_records=rejected),
                base_sampling=dict(enabled=samples is not None, attempted=len(sample_records),
                                   checked=sum(_checked(r) for r in sample_records), supplemented_sides=supplemented,
                                   metadata=None if samples is None else {k: v for k, v in samples.items() if k != "records"}))


def discover_states(run_dirs, results_root=ROOT):
    """Include every delta, including saved zero controls, exactly once."""
    paths = set()
    for directory in run_dirs:
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"missing run directory: {directory}")
        found = list(directory.rglob("delta_*.npz"))
        if not found:
            raise ValueError(f"no saved deltas in {directory}")
        paths.update(p.resolve() for p in found)
    states = [dict(state_id="base", delta_path=None)]
    for path in sorted(paths):
        sid = _source_id(path.parent, path.stem.removeprefix("delta_"))
        states.append(dict(state_id=sid, delta_path=_relative(path, results_root)))
    if len({s["state_id"] for s in states}) != len(states):
        raise ValueError("duplicate state IDs across run directories; use one saved delta per state")
    return states


def _prompt_ids(tokenizer, prompt, protocol):
    from tools.behavior_atom import gpu_driver as driver
    micro = protocol["micro_update"]
    ids = tokenizer(driver._thinking_off(prompt), add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError("empty encoded prompt")
    dropped = max(0, len(ids) - micro["prompt_cap"])
    if dropped and not micro["allow_prompt_truncation"]:
        raise ValueError("probe prompt exceeds cap without truncation authorization")
    return ids[-micro["prompt_cap"]:], dropped


def score_pairs(student, pairs, protocol):
    """Likelihood-only path, reusing the driver's batched native LM head.

    Responses are never truncated: all decoded text plus EOS is scored. This
    equals driver likelihood for responses within its 512-token training cap.
    No generate/checker/ground-truth field is consulted by this function.
    """
    import torch
    import torch.nn.functional as F
    from tools.behavior_atom import gpu_driver as driver
    tok = student.tokenizer
    if tok.eos_token_id is None:
        raise ValueError("EOS is required for margin likelihoods")
    ev, device = protocol["evaluation"], protocol["micro_update"]["device"]
    if protocol["micro_update"]["ell"] != "full_sequence":
        raise ValueError("margin requires full_sequence likelihoods")
    encoded, records = [], []
    for i, p in enumerate(pairs):
        prompt, dropped = _prompt_ids(tok, p["prompt"], protocol)
        record = dict(probe_id=p["probe_id"], logp_plus=None, logp_minus=None, m=None,
                      counts=dict(prompt_tokens=len(prompt), prompt_tokens_dropped=dropped))
        records.append(record)
        for side in ("plus", "minus"):
            text = p[f"y_{side}"]
            if text is None:
                continue
            response = tok(text, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            encoded.append((prompt + response, len(prompt), response, i, side))
            record["counts"][side] = dict(effective_loss_tokens=len(response), response_tokens_dropped=0)
    student.model.eval()
    with driver._context(protocol), torch.no_grad():
        for batch in driver._batches([len(e[0]) for e in encoded], ev):
            ids, mask = driver._padded([encoded[i][0] for i in batch], tok.pad_token_id, device)
            positions, targets, counts = [], [], []
            for b, index in enumerate(batch):
                _, start, response, _, _ = encoded[index]
                positions.extend([b, t] for t in range(start - 1, start + len(response) - 1))
                targets.extend(response)
                counts.append(len(response))
            pos = torch.tensor(positions, dtype=torch.long, device=device)
            target = torch.tensor(targets, dtype=torch.long, device=device)
            values, offset = [], 0
            for logits in driver._position_logits(student, ids, mask, pos, ev["head_chunk"]):
                n = len(logits)
                values.append(-F.cross_entropy(logits, target[offset:offset+n], reduction="none").float())
                offset += n
            values = torch.cat(values)
            offset = 0
            for index, count in zip(batch, counts):
                score = values[offset:offset+count].sum(dtype=torch.float32)
                offset += count
                if not torch.isfinite(score):
                    raise ValueError("nonfinite likelihood; refusing to label a scoring error as a missing text")
                _, _, _, owner, side = encoded[index]
                records[owner][f"logp_{side}"] = float(score)
    for record in records:
        if record["logp_plus"] is not None and record["logp_minus"] is not None:
            record["m"] = record["logp_plus"] - record["logp_minus"]
    return records


def margin_matrix(states, pair_bundle, records):
    ids = [s["state_id"] for s in states]
    probes = [p["probe_id"] for p in pair_bundle["probes"]]
    lookup = {(r["state_id"], r["probe_id"]): r for r in records}
    if len(lookup) != len(records) or set(lookup) != {(s, p) for s in ids for p in probes}:
        raise ValueError("duplicate/missing/unknown state × probe records")
    return dict(version="behavior-margin-matrix-v1", orientation="states x probes", states=states,
                state_ids=ids, probe_ids=probes, pairs_sha256=_digest(pair_bundle),
                likelihood_definition=LIKELIHOOD, na_representation="JSON null (NA)",
                **{k: [[lookup[s, p][k] for p in probes] for s in ids] for k in ("m", "logp_plus", "logp_minus")})


@contextmanager
def _base_student(protocol, *, checker=False, student_factory=None):
    import torch
    from src.bfas.behavior import microupdate as micro
    from src.bfas.behavior.deltas import tensor_state_hash
    from tools.behavior_atom import gpu_driver as driver
    path = Path(protocol["micro_update"]["init_state_path"])
    if not path.is_file():
        raise ValueError(f"saved initial state is required: {path}")
    micro.cap_cpu_threads(min(8, protocol["micro_update"]["cpu_threads"]))
    with driver._context(protocol):
        micro._seed(protocol["micro_update"]["init_seed"], protocol["micro_update"]["device"].startswith("cuda"))
        if student_factory:
            student = student_factory(protocol)
        elif checker:
            student = driver.load_student(protocol)  # CheckerBridge, loaded only for opt-in samples.
        else:
            import appworld_train as trainer
            student = driver.Student(driver.load_model(protocol),
                                     trainer.load_tokenizer(protocol["micro_update"]["model_id"]), "unused", None, None)
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
            differences = driver._validate_initial_state(student.model, state)
            # The driver verifies tensor identity even without a stored digest.
            trainables_hash = tensor_state_hash(state["trainables"])
            # Old checkpoints may omit subsequently recorded config fields,
            # exactly as in collect. Tensor/layout/frozen identity checks remain mandatory.
            micro._check_settings_diff(differences, False)
            micro._restore(student.model, state)
            student.model.eval()
            micro._seed(protocol["eval_seeds"][0], protocol["micro_update"]["device"].startswith("cuda"))
            yield student, state, dict(initial_state_sha256=_hash(path), init_state_settings_diff=differences,
                                       initial_trainables_hash=trainables_hash)
        finally:
            student.close()


def score_states(pair_bundle, states, protocol, results_root, output_dir, *, student_factory=None):
    from src.bfas.behavior import microupdate as micro
    from src.bfas.behavior.deltas import apply_delta, load_delta
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records, scored_states = [], []
    with _base_student(protocol, student_factory=student_factory) as (student, initial, runtime):
        for number, state in enumerate(states, 1):
            print(f"margin: scoring {number}/{len(states)} {state['state_id']}", file=sys.stderr, flush=True)
            micro._restore(student.model, initial, verify_hash=False)
            metadata = dict(state)
            if state["delta_path"] is not None:
                path = Path(results_root) / state["delta_path"]
                apply_delta(student.model, load_delta(path))
                metadata["delta_sha256"] = _hash(path)
            scored_states.append(metadata)
            records.extend(dict(state_id=state["state_id"], **r)
                           for r in score_pairs(student, pair_bundle["probes"], protocol))
    matrix = margin_matrix(scored_states, pair_bundle, records)
    matrix.update(runtime=runtime, protocol=protocol)
    temporary = output_dir / "margins.jsonl.tmp"
    temporary.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in records))
    temporary.replace(output_dir / "margins.jsonl")
    _write(output_dir / "margin_matrix.json", matrix)
    _write(output_dir / "summary.json", dict(status="scored", states=len(states), probes=len(pair_bundle["probes"]),
                                            coverage=pair_bundle["coverage"], base_sampling=pair_bundle["base_sampling"],
                                            likelihood_definition=LIKELIHOOD, runtime=runtime))
    return matrix


def sample_base(probes_path, pair_bundle, protocol, k, *, temperature=0.8):
    """Temperature-sample incomplete probes from the restored BASE only."""
    import torch
    from tools.behavior_atom import gpu_driver as driver
    if k < 1 or temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("sampling requires positive k and temperature")
    raw = _read(probes_path)
    probes = raw["probes"] if isinstance(raw, dict) else raw
    incomplete = {p["probe_id"] for p in pair_bundle["probes"] if p["missing_sides"]}
    records = []
    with _base_student(protocol, checker=True) as (student, _, runtime):
        tok = student.tokenizer
        with torch.no_grad():
            for probe in probes:
                if probe["probe_id"] not in incomplete:
                    continue
                prompt, _ = _prompt_ids(tok, probe["prompt"], protocol)
                ids, mask = driver._padded([prompt], tok.pad_token_id, protocol["micro_update"]["device"])
                for i in range(k):
                    # Do not change Student.generate: its existing contract is greedy.
                    output = student.model.generate(input_ids=ids, attention_mask=mask, do_sample=True,
                        temperature=temperature, top_p=1.0, top_k=0, num_beams=1, use_cache=True,
                        max_new_tokens=protocol["evaluation"]["max_new_tokens"],
                        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
                    text = tok.batch_decode(output[:, ids.shape[1]:], skip_special_tokens=True)[0]
                    record = dict(state_id="base_sample", probe_id=probe["probe_id"], sample_index=i,
                                  output=text, outcome=None, errors=[], checker_version=student.checker_version)
                    try:
                        calls = student.parse(text)
                        # Includes abstention via CheckerBridge; a missing required
                        # call is a deterministic negative, as in the GPU driver.
                        verdict = (dict(valid=False, error_type="missing_call")
                                   if probe["expected"] == "call" and not calls else student.check(probe, calls))
                        if type(verdict.get("valid")) is not bool:
                            raise ValueError("checker must return boolean valid")
                        record.update(outcome=int(verdict["valid"]), called=bool(calls), verdict=verdict)
                    except Exception as exc:
                        record["errors"].append(dict(kind="checker", error=f"{type(exc).__name__}: {exc}"))
                    records.append(record)
    return dict(version="behavior-base-samples-v1", k=k, temperature=temperature,
                seed=protocol["eval_seeds"][0], probes_sha256=_hash(probes_path),
                runtime=runtime, records=records)


def _axes(matrix, pair_bundle):
    ids, pids = matrix["state_ids"], matrix["probe_ids"]
    pairs = {p["probe_id"]: p for p in pair_bundle["probes"]}
    M = np.asarray(matrix["m"], dtype=float)
    if (len(set(ids)) != len(ids) or len(set(pids)) != len(pids) or "base" not in ids
            or M.shape != (len(ids), len(pids)) or set(pids) != set(pairs) or np.isinf(M).any()):
        raise ValueError("invalid margin matrix axes/values")
    return {sid: i for i, sid in enumerate(ids)}, pids, M, pairs


def analyze_q1(matrix, pair_bundle):
    """Joint-minus-single margin on base-correct should-abstain probes only."""
    index, pids, M, pairs = _axes(matrix, pair_bundle)
    retained = [j for j, pid in enumerate(pids)
                if pairs[pid]["kind"] == "should_abstain" and pairs[pid]["base_outcome"] == 1]
    rows = []
    outcomes = pair_bundle.get("archived_outcomes", {})
    for single in SINGLES:
        for anchor in ("anchor_match", "anchor_random", "abstain_anchor"):
            pack = f"joint_{single}_{anchor}"
            per_probe, invisible = {}, []
            for j in retained:
                pid = pids[j]
                value = (M[index[pack], j] - M[index[single], j]
                         if pack in index and single in index else np.nan)
                if pairs[pid]["y_plus"] is None or pairs[pid]["y_minus"] is None:
                    value = np.nan
                per_probe[pid] = value
                a, b = outcomes.get(pack, {}).get(pid), outcomes.get(single, {}).get(pid)
                if np.isfinite(value) and a is not None and a == b:
                    invisible.append(value)
            values = np.asarray(list(per_probe.values()), dtype=float)
            valid = values[np.isfinite(values)]
            rows.append(dict(pack=pack, single=single, retained=len(retained), both_sides=len(valid),
                             na_count=len(retained)-len(valid), mean_delta_m=float(valid.mean()) if len(valid) else None,
                             raised=int((valid > 0).sum()), lowered=int((valid < 0).sum()), per_probe_delta_m=per_probe,
                             unchanged_binary_count=len(invisible),
                             unchanged_binary_mean_delta_m=float(np.mean(invisible)) if invisible else None,
                             status="ok" if pack in index and single in index else "NA: missing state"))
    return _clean(dict(question="Q1", retained_probe_ids=[pids[j] for j in retained], packs=rows))


def _profile_metrics(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return dict(cosine=float(np.clip(a @ b / denominator, -1, 1)) if denominator > 0 else None,
                l1=float(np.abs(a-b).sum()), mean_l1=float(np.abs(a-b).mean()))


def analyze_q2(matrix, pair_bundle, *, bootstrap_samples=2000, seed=0):
    """Response = m(state)-m(base); paired probe resampling within each source.

    Bootstrap keeps joint/half/base on the same probe indices and never pools
    src_17 with src_04. Percentile intervals are descriptive, not a null test.
    """
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    index, pids, M, _ = _axes(matrix, pair_bundle)
    rows = []
    for pair_number, (joint, half) in enumerate(Q2_PAIRS):
        row = dict(joint=joint, half=half, bootstrap_samples=bootstrap_samples, seed=seed+pair_number)
        if joint not in index or half not in index:
            rows.append({**row, "status": "NA: missing state", "both_sides": 0, "na_count": len(pids),
                         "cosine": None, "l1": None, "mean_l1": None, "bootstrap_ci95": None})
            continue
        a, b = M[index[joint]] - M[index["base"]], M[index[half]] - M[index["base"]]
        valid = np.isfinite(a) & np.isfinite(b)
        a, b = a[valid], b[valid]
        row.update(both_sides=len(a), na_count=len(pids)-len(a),
                   probe_ids=[p for p, v in zip(pids, valid) if v], joint_response=a, half_response=b,
                   status="ok" if len(a) else "NA: no complete probes")
        if not len(a):
            rows.append({**row, "cosine": None, "l1": None, "mean_l1": None, "bootstrap_ci95": None})
            continue
        row.update(_profile_metrics(a, b))
        rng = np.random.default_rng(seed+pair_number)
        boot = defaultdict(list)
        for _ in range(bootstrap_samples):
            selected = rng.integers(len(a), size=len(a))
            for metric, value in _profile_metrics(a[selected], b[selected]).items():
                if value is not None:
                    boot[metric].append(value)
        row["bootstrap_ci95"] = {metric: list(np.quantile(boot[metric], [0.025, 0.975])) if boot[metric] else None
                                 for metric in ("cosine", "l1", "mean_l1")}
        row["bootstrap_valid"] = {metric: len(boot[metric]) for metric in ("cosine", "l1", "mean_l1")}
        rows.append(row)
    return _clean(dict(question="Q2", response_definition="m(state) - m(base)",
                       bootstrap="same-source paired probe percentile bootstrap; no cross-source pooling; not a null test",
                       comparisons=rows))


def load_fold_codes(codes_dir, folds_path):
    """Use only build_codes' training-fold outputs, never codes_all.npz."""
    registered = _read(folds_path)
    if registered["n_folds"] != 4:
        raise ValueError("Q3 requires the registered grouped 4-fold split")
    bundles = []
    for fold in range(4):
        path = Path(codes_dir) / f"fold_{fold}" / "codes.npz"
        with np.load(path, allow_pickle=False) as archive:
            data = {k: archive[k] for k in archive.files}
        if (data["basis_scope"].item() != "training_fold" or bool(data["transductive"].item())
                or int(data["heldout_fold"]) != fold):
            raise ValueError("Q3 refuses a transductive/wrong-fold code basis")
        ids = list(map(str, data["source_ids"]))
        labels = np.asarray([registered["assignment"][sid] for sid in ids], dtype=int)
        groups = [registered["groups"][sid] for sid in ids]
        if not np.array_equal(labels, data["folds"]) or groups != list(data["source_groups"]):
            raise ValueError("saved code folds/groups differ from folds.json")
        train, test = np.flatnonzero(labels != fold), np.flatnonzero(labels == fold)
        if (not np.array_equal(train, data["train_indices"]) or not np.array_equal(test, data["test_indices"])
                or list(data["decoder_source_ids"]) != [ids[i] for i in train]):
            raise ValueError("code decoder includes held-out sources or has the wrong training indices")
        bundles.append(dict(fold=fold, source_ids=ids, source_groups=groups, train_indices=train,
                            test_indices=test, X=data["X"], delta_norm_h=data["delta_norm_h"],
                            codes_sha256=_hash(path)))
    return bundles


def _predict_conditional(B, X, y, train, test, ridge):
    """Unpenalized nuisance OLS plus ridge on nuisance-residualized H codes.

    A single training RMS scales the residual code block, retaining rotation
    invariance of the H coordinates. No held-out target enters any transform.
    """
    Btrain, Btest = B[train], B[test]
    beta = np.linalg.lstsq(Btrain, y[train], rcond=None)[0]
    baseline = Btest @ beta
    nuisance = np.linalg.lstsq(Btrain, X[train], rcond=None)[0]
    Z, Ztest = X[train] - Btrain @ nuisance, X[test] - Btest @ nuisance
    scale = np.sqrt(np.square(Z).sum() / len(train))
    if scale <= np.finfo(float).eps * max(1., np.linalg.norm(X[train])):
        return baseline, baseline.copy()
    Z, Ztest = Z / scale, Ztest / scale
    # Augmented least squares is stable for rank-deficient fold codes too.
    design = np.vstack((Z, np.sqrt(ridge) * np.eye(Z.shape[1])))
    residual = np.concatenate((y[train] - Btrain @ beta, np.zeros(Z.shape[1])))
    weights = np.linalg.lstsq(design, residual, rcond=None)[0]
    return baseline, baseline + Ztest @ weights


def analyze_q3(matrix, pair_bundle, fold_codes, *, ridge=1.0, permutation_seeds=(11, 29, 47, 71, 101)):
    """Held-out single-source prediction, common coordinate + full H norm.

    u = mean(training deltas) / ||mean(training deltas)||_H, represented in
    that fold's H-orthonormal basis. Common coordinate = <u, delta_i>_H.
    Baseline is per-probe OLS on [intercept, common coordinate, full H norm].
    The five permutations shuffle code rows within train and test separately;
    true common/norm features remain fixed and every model uses identical cells.
    """
    if ridge <= 0 or not np.isfinite(ridge) or len(permutation_seeds) != 5 or len(set(permutation_seeds)) != 5:
        raise ValueError("Q3 requires positive fixed ridge and exactly five distinct permutation seeds")
    if len(fold_codes) != 4 or {b["fold"] for b in fold_codes} != set(range(4)):
        raise ValueError("Q3 requires four distinct held-out folds")
    index, pids, M, _ = _axes(matrix, pair_bundle)
    source_ids = list(fold_codes[0]["source_ids"])
    if len(set(source_ids)) != len(source_ids) or any(s not in index for s in source_ids):
        raise ValueError("Q3 code source IDs are duplicate or missing from the matrix")
    Y = M[[index[s] for s in source_ids]] - M[index["base"]]
    shape = Y.shape
    names = ["common+norm", "common+norm+codes ridge"] + [f"permuted_codes_{s}" for s in permutation_seeds]
    predictions = {name: np.full(shape, np.nan) for name in names}
    fold_rows, seen, group_fold = [], set(), {}
    canonical_norms = np.asarray(fold_codes[0]["delta_norm_h"], dtype=float)
    canonical_groups = list(fold_codes[0]["source_groups"])
    for bundle in sorted(fold_codes, key=lambda b: b["fold"]):
        if list(bundle["source_ids"]) != source_ids or list(bundle["source_groups"]) != canonical_groups:
            raise ValueError("fold code source axes/groups differ")
        X, norms = np.asarray(bundle["X"], dtype=float), np.asarray(bundle["delta_norm_h"], dtype=float)
        train, test = np.asarray(bundle["train_indices"], dtype=int), np.asarray(bundle["test_indices"], dtype=int)
        if (not len(train) or not len(test) or len(set(train)) != len(train) or len(set(test)) != len(test)
                or set(train) & set(test) or set(train) | set(test) != set(range(len(source_ids))) or seen & set(test)):
            raise ValueError("invalid/overlapping fold partitions")
        seen.update(test)
        groups = np.asarray(canonical_groups)
        if set(groups[train]) & set(groups[test]):
            raise ValueError("parent group leaks across training and held-out sources")
        for g in groups[test]:
            if g in group_fold and group_fold[g] != bundle["fold"]:
                raise ValueError("parent group split across folds")
            group_fold[g] = bundle["fold"]
        if (X.ndim != 2 or len(X) != len(source_ids) or norms.shape != (len(source_ids),)
                or not np.isfinite(X).all() or not np.isfinite(norms).all() or (norms < 0).any()
                or not np.array_equal(norms, canonical_norms)):
            raise ValueError("invalid fold H codes/norms")
        direction = X[train].mean(axis=0)
        size = np.linalg.norm(direction)
        direction = direction / size if size else np.zeros_like(direction)
        nuisance = np.column_stack((X @ direction, norms))
        center, scale = nuisance[train].mean(axis=0), nuisance[train].std(axis=0)
        scale = np.where(scale > np.finfo(float).eps, scale, 1.)
        B = np.column_stack((np.ones(len(X)), (nuisance-center)/scale))
        permuted = []
        for seed in permutation_seeds:
            rng = np.random.default_rng(np.random.SeedSequence([seed, bundle["fold"]]))
            perm = X.copy()
            perm[train] = X[rng.permutation(train)]
            perm[test] = X[rng.permutation(test)]
            permuted.append(perm)
        for j in range(len(pids)):
            tr = train[np.isfinite(Y[train, j])]
            te = test[np.isfinite(Y[test, j])]
            if len(tr) < B.shape[1] + 1 or not len(te):
                continue
            base, fitted = _predict_conditional(B, X, Y[:, j], tr, te, ridge)
            predictions[names[0]][te, j] = base
            predictions[names[1]][te, j] = fitted
            for name, perm in zip(names[2:], permuted):
                predictions[name][te, j] = _predict_conditional(B, perm, Y[:, j], tr, te, ridge)[1]
        valid = np.isfinite(Y[test])
        for pred in predictions.values():
            valid &= np.isfinite(pred[test])
        fold_rows.append(dict(fold=bundle["fold"], train_source_ids=[source_ids[i] for i in train],
                              heldout_source_ids=[source_ids[i] for i in test], cells=int(valid.sum()),
                              mse={name: float(np.square(pred[test]-Y[test])[valid].mean()) if valid.any() else None
                                   for name, pred in predictions.items()}, codes_sha256=bundle.get("codes_sha256")))
    if seen != set(range(len(source_ids))):
        raise ValueError("not every source has exactly one held-out prediction")
    valid = np.isfinite(Y)
    for pred in predictions.values():
        valid &= np.isfinite(pred)
    mse = {name: float(np.square(pred-Y)[valid].mean()) if valid.any() else None for name, pred in predictions.items()}
    complete = bool(valid.any()) and all(f["cells"] for f in fold_rows)
    increment = mse[names[0]] - mse[names[1]] if valid.any() else None
    # This is a descriptive comparison, not a significance threshold. Five
    # permutations cannot provide a conventional p<.05 randomization result.
    numerical_tolerance = 64 * np.finfo(float).eps * max(1., float(np.square(Y[valid]).mean())) if valid.any() else None
    exceeds = bool(complete and increment > numerical_tolerance
                   and all(mse[n] - mse[names[1]] > numerical_tolerance for n in names[2:]))
    return _clean(dict(question="Q3", status="ok" if complete else "INCONCLUSIVE: missing held-out cells/folds",
        source_ids=source_ids, probe_ids=pids, cells=int(valid.sum()), na_count=int(Y.size-valid.sum()),
        excluded_states=[s for s in matrix["state_ids"] if s not in source_ids and s != "base"],
        common_definition="H projection on normalized mean training delta; u = C mean(X_train)/||mean(X_train)||",
        norm_definition="saved delta_norm_h (full H norm, including outside-basis residual)",
        estimator="per-probe nuisance OLS (intercept/common/norm) + residualized codes ridge; train-only scaling",
        ridge=ridge, ridge_selection="fixed before outcomes; no held-out tuning", permutation_seeds=list(permutation_seeds),
        permutation_scheme="code rows shuffled within outer training and held-out sets separately; nuisance features fixed",
        folds=fold_rows, mse=mse, mse_improvement=increment, exceeds_common_norm_and_all_permutations=exceeds,
        numerical_mse_tolerance=numerical_tolerance,
        permutation_rank_p=(1+sum(mse[n] <= mse[names[1]] for n in names[2:]))/6 if valid.any() else None,
        predictions=predictions))


def analyze(matrix, pair_bundle, fold_codes, *, ridge=1., bootstrap_samples=2000, seed=0,
            permutation_seeds=(11, 29, 47, 71, 101)):
    if matrix.get("pairs_sha256") != _digest(pair_bundle):
        raise ValueError("margin matrix was scored on different probe pairs")
    q1 = analyze_q1(matrix, pair_bundle)
    q2 = analyze_q2(matrix, pair_bundle, bootstrap_samples=bootstrap_samples, seed=seed)
    q3 = analyze_q3(matrix, pair_bundle, fold_codes, ridge=ridge, permutation_seeds=permutation_seeds)
    if q3["status"] != "ok":
        verdict = "INCONCLUSIVE: Q3 has insufficient held-out measurements."
    elif q3["exceeds_common_norm_and_all_permutations"]:
        verdict = VERDICT_RULES[1]
    else:
        verdict = VERDICT_RULES[0]
    return dict(status="analyzed", states=len(matrix["state_ids"]), probes=len(matrix["probe_ids"]),
                coverage=pair_bundle["coverage"], base_sampling=pair_bundle["base_sampling"],
                pairs_sha256=_digest(pair_bundle), likelihood_definition=LIKELIHOOD,
                Q1=q1, Q2=q2, Q3=q3, verdict=verdict, verdict_rules=list(VERDICT_RULES),
                verdict_scope="Descriptive, single-seed diagnostic; five permutations are not confirmatory. "
                    "The prospective branch additionally requires a pre-specified NEW-combination prediction; "
                    "these retrospective measurements alone do not satisfy it. P3–P5 remain closed.")


def report_markdown(summary):
    def fmt(x):
        return "NA" if x is None else f"{x:.6g}"
    coverage, samples = summary["coverage"], summary["base_sampling"]
    lines = ["# Behaviour-atom continuous-margin diagnostic (C16)", "",
             f"{summary['states']} states × {summary['probes']} probes. {LIKELIHOOD}. JSON null = NA.",
             "The prompt uses the driver's empty thinking prefix and configured tail cap. "
             "Every selected response token plus EOS is scored; no response truncation or length normalization.",
             f"Archived pairs with both sides: {coverage['archive_both_sides']}/{coverage['probes']}; "
             f"final coverage: {coverage['both_sides']}/{coverage['probes']}; "
             f"missing plus/minus: {coverage['missing_plus']}/{coverage['missing_minus']}.",
             f"Base samples (separate supplemental provenance): enabled={samples['enabled']}, "
             f"attempted={samples['attempted']}, checked={samples['checked']}, "
             f"supplemented sides={len(samples['supplemented_sides'])}.", "",
             "## Q1 — Do anchors raise retained margins relative to the same single source?", "",
             "Retained = should_abstain and archived base outcome 1. Δm = m(pack) − m(single). "
             "Means exclude NA; per-probe Δm is preserved in summary.json.", "",
             "| Pack | Single | Both sides | NA | Mean Δm | Raised / lowered | Same 0/1: n / mean Δm |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in summary["Q1"]["packs"]:
        lines.append(f"| {r['pack']} | {r['single']} | {r['both_sides']} | {r['na_count']} | "
                     f"{fmt(r['mean_delta_m'])} | {r['raised']} / {r['lowered']} | "
                     f"{r['unchanged_binary_count']} / {fmt(r['unchanged_binary_mean_delta_m'])} |")
    lines += ["", "## Q2 — Are the two joint profiles different from the same-source half dose?", "",
              "Profiles are m(state) − m(base). Bootstrap resamples the same probe indices for each "
              "joint/half pair, separately for src_17 and src_04. L1 is a sum; mean L1 is also reported. "
              "Intervals describe probe variability and are not a null-hypothesis test. Zero-vector cosine is NA.", "",
              "| Joint / half | Probes / NA | Cosine [95% CI] | L1 [95% CI] | Mean L1 |",
              "|---|---:|---:|---:|---:|"]
    for r in summary["Q2"]["comparisons"]:
        def interval(metric):
            ci = (r.get("bootstrap_ci95") or {}).get(metric)
            return "NA" if ci is None else f"[{fmt(ci[0])}, {fmt(ci[1])}]"
        lines.append(f"| {r['joint']} / {r['half']} | {r['both_sides']} / {r['na_count']} | "
                     f"{fmt(r['cosine'])} {interval('cosine')} | {fmt(r['l1'])} {interval('l1')} | {fmt(r['mean_l1'])} |")
    q3 = summary["Q3"]
    lines += ["", "## Q3 — Do H-metric codes add held-out value after common direction and H norm?", "",
              f"{q3['status']}. Grouped four-fold prediction on {len(q3['source_ids'])} saved single sources, "
              f"{q3['cells']} common scored cells, {q3['na_count']} NA/excluded cells. "
              "Joint, half-dose and zero controls are excluded from this source-fold prediction test.",
              q3["common_definition"] + ". " + q3["norm_definition"] + ".",
              f"{q3['estimator']}. Ridge λ={q3['ridge']} fixed before outcomes. "
              "Basis, common direction, nuisance regression and scaling use training sources only.",
              q3["permutation_scheme"] + ".", "", "| Model | Held-out MSE |", "|---|---:|"]
    for name, mse in q3["mse"].items():
        lines.append(f"| {name} | {fmt(mse)} |")
    lines += ["", "| Fold | Common cells | Common + norm MSE | With codes MSE |", "|---:|---:|---:|---:|"]
    for r in q3["folds"]:
        lines.append(f"| {r['fold']} | {r['cells']} | {fmt(r['mse']['common+norm'])} | "
                     f"{fmt(r['mse']['common+norm+codes ridge'])} |")
    lines += ["", f"MSE improvement: {fmt(q3['mse_improvement'])}. "
              f"Permutation rank diagnostic: {fmt(q3['permutation_rank_p'])} (minimum 1/6 with five permutations).",
              "", "## Pre-fixed verdict rules", "",
              *[f"- {rule}" for rule in summary["verdict_rules"]], "",
              f"Verdict: **{summary['verdict']}**", "", summary["verdict_scope"],
              "The operational descriptive increment requires lower pooled held-out MSE than common+norm "
              "and all five code permutations, with measurements in all four folds. "
              "Differences within 64 × float64 epsilon × max(1, mean squared target) are numerical ties. "
              "The plan supplies no numerical significance/effect-size threshold; this is not evidence of a new selection rule.", ""]
    return "\n".join(lines)


def _config(path):
    cfg = _read(path)
    base = Path(path).resolve().parent
    for key in ("probes", "results_root", "pairs", "output_dir", "codes_dir", "folds"):
        cfg[key] = (base / cfg[key]).resolve()
    cfg["run_dirs"] = [(base / p).resolve() for p in cfg["run_dirs"]]
    cfg["protocol"]["micro_update"]["init_state_path"] = str(
        (base / cfg["protocol"]["micro_update"]["init_state_path"]).resolve())
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/behavior_atom/margin_v1.json")
    parser.add_argument("--step", choices=("pairs", "score", "analyze", "all"), default="all")
    parser.add_argument("--run-dirs", type=Path, nargs="+", help="override saved-state run dirs (relative to cwd)")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sample-base", type=int, default=0, metavar="K",
                        help="opt-in: K temperature samples per incomplete probe, BASE only; requires GPU/checker")
    parser.add_argument("--dry-run", action="store_true", help="validate inputs and show coverage/states, no model or writes")
    args = parser.parse_args(argv)
    try:
        if args.sample_base < 0 or (args.sample_base and args.step not in {"pairs", "all"}):
            raise ValueError("--sample-base requires nonnegative K and --step pairs/all")
        cfg = _config(args.config)
        if args.run_dirs:
            cfg["run_dirs"] = [p.resolve() for p in args.run_dirs]
        if args.output_dir:
            cfg["output_dir"] = args.output_dir.resolve()
        # Set caps before importing torch/model code. Limit already-loaded BLAS
        # pools too for the CPU analysis path (threadpoolctl is optional).
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = str(min(8, max(1, int(os.environ.get(name, "8")))))
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        from tools.behavior_atom import gpu_driver as driver
        protocol = driver.frozen_settings(cfg["protocol"])
        protocol["micro_update"]["cpu_threads"] = min(8, protocol["micro_update"]["cpu_threads"])
        if protocol["micro_update"]["ell"] != "full_sequence":
            raise ValueError("margin requires full_sequence")
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:
            from contextlib import nullcontext
            limits = nullcontext()
        else:
            limits = threadpool_limits(limits=8)
        with limits:
            if args.dry_run:
                pairs = build_pairs(cfg["probes"], cfg["results_root"])
                states = discover_states(cfg["run_dirs"], cfg["results_root"])
                print(json.dumps(dict(step=args.step, coverage=pairs["coverage"], states=states,
                                      sample_base=args.sample_base), indent=2))
                return 0
            if args.step in {"pairs", "all"}:
                pairs = build_pairs(cfg["probes"], cfg["results_root"])
                if args.sample_base:
                    samples = sample_base(cfg["probes"], pairs, protocol, args.sample_base,
                                          temperature=cfg.get("sample_temperature", 0.8))
                    _write(cfg["pairs"].with_name("base_samples_v1.json"), samples)
                    pairs = build_pairs(cfg["probes"], cfg["results_root"], samples=samples)
                _write(cfg["pairs"], pairs)
                print(json.dumps(dict(pairs=str(cfg["pairs"]), coverage=pairs["coverage"],
                                      base_sampling=pairs["base_sampling"]), ensure_ascii=False))
            else:
                pairs = _read(cfg["pairs"])
            if pairs["probes_sha256"] != _hash(cfg["probes"]):
                raise ValueError("probe manifest changed since pair building")
            if args.step in {"score", "all"}:
                states = discover_states(cfg["run_dirs"], cfg["results_root"])
                score_states(pairs, states, protocol, cfg["results_root"], cfg["output_dir"])
            if args.step in {"analyze", "all"}:
                matrix = _read(cfg["output_dir"] / "margin_matrix.json")
                codes = load_fold_codes(cfg["codes_dir"], cfg["folds"])
                # Bind saved code inputs to the actual scored delta files, without
                # reopening multi-GB payloads during this CPU-only analysis step.
                code_summary = _read(cfg["codes_dir"] / "summary.json")
                hashes = {s["state_id"]: s.get("delta_sha256") for s in matrix["states"]}
                for ref in code_summary["deltas"]:
                    if hashes.get(ref["source_id"]) != ref["content_hash"]:
                        raise ValueError("code inputs differ from scored deltas")
                if matrix["runtime"]["initial_trainables_hash"] != code_summary["base_hash"]:
                    raise ValueError("codes and likelihoods use different initial states")
                summary = analyze(matrix, pairs, codes, **cfg.get("analysis", {}))
                summary["folds_sha256"] = _hash(cfg["folds"])
                summary["matrix_sha256"] = _hash(cfg["output_dir"] / "margin_matrix.json")
                markdown = report_markdown(summary)
                _write(cfg["output_dir"] / "summary.json", summary)
                (cfg["output_dir"] / "report.md").write_text(markdown)
                print(markdown)
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"margin: {exc}", file=sys.stderr)
        return 2
