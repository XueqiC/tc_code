"""Build amplitude-preserving P2.3 codes from saved collect/pilot deltas (CPU).

Default damping is ABSOLUTE: H = F + 1e-3, without Fisher normalization.
--eps-mode relative instead uses eps * median(F), including zero entries;
a zero median is rejected rather than silently choosing a different metric.
The Fisher archive's saved eps is provenance only, not added a second time.

Outputs: fold_<label>/codes.npz, codes_all.npz (TRANSDUCTIVE), summary.json,
and an unchanged responses.jsonl + its hashed manifest. Each archive contains
X, source_ids, groups/source_groups, folds, source_factor, factor and family.
train_indices/test_indices specify the ONLY outer split for a fold archive.
Unassigned controls have fold -1: encoded and diagnosed, excluded from CV.
Do not concatenate rows from different bases or pass one fold's X through
fit's entire cross-validation loop. The existing single-X fit interface does
not dispatch these outer-fold archives; codes_all is a transductive control.

The decoder is C = D_train @ decoder_coefficients (Q = sqrt(H) C). Ordered
decoder paths and content hashes refer to the original immutable deltas.
Reload them with deltas.load_delta(...).flat() to decode real parameters.
Saved tensors are fp32, flattened in manifest layout order. Flattening and
geometry promote to fp64, preserving the optional fp32 roundoff corrections
needed for exact replay. Temporary memmaps hold the flat vectors; only small
source Gram matrices and parameter blocks enter RAM, never a d x d matrix.

gpu_driver.merge saves shard completion hashes, not shard paths or deltas.
Sibling shards are located by those hashes; use --shard-dirs for moved shards.
All selected deltas must match the merged update summaries and Fisher layout.
--dry-run checks metadata/Fisher/response integrity without reading delta
payloads, fitting bases, or writing output. --pilot-smoke additionally requires
six pilot deltas, exactly two zeros, and exact zero codes/residuals in every
basis. It performs no training, model loading, rollout, or teacher call.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bfas.behavior import deltas, whiten  # noqa: E402
from src.bfas.behavior.response import read_responses  # noqa: E402

DATA = ROOT / "data/behavior_atom_v1"
PILOT = ROOT / "results/behavior_atom_v1/pilot_r1"


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    deltas.write_json_exclusive(path, value)


def _complete(directory):
    record = _read(directory / "complete.json")
    if record.get("status") != "complete" or record.get("exit_code") != 0:
        raise ValueError(f"incomplete collect artifact: {directory}")
    return record["artifact_hashes"]


def _check_saved(path, directory, hashes):
    key = path.relative_to(directory).as_posix()
    if key not in hashes or deltas.file_hash(path) != hashes[key]:
        raise ValueError(f"collect artifact hash mismatch: {path}")


def _delta_paths(run, result, shard_dirs):
    """Resolve source IDs by summaries, never by unverified directory order."""
    updates = result["updates"]
    ids = [u["source_id"] for u in updates]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("collect source IDs must be nonempty and unique")
    expected_ids = result.get("shared", {}).get("identity", {}).get("source_ids")
    if expected_ids is not None and set(expected_ids) != set(ids):
        raise ValueError("collect source coverage differs from shared identity")
    roots = [run]
    # A copied merged run can already contain the source artifacts.
    if result["stage"] == "merge" and not (run / "sources").is_dir():
        required = result.get("shard_hashes", [])
        if not required or len(set(required)) != len(required):
            raise ValueError("merged run needs unique shard completion hashes")
        candidates = ([Path(p).resolve() for p in shard_dirs] if shard_dirs else
                      [p.parent for p in run.parent.glob("*/complete.json")])
        found = {}
        for directory in sorted(set(candidates)):
            completion = directory / "complete.json"
            if completion.is_file():
                digest = deltas.file_hash(completion)
                if digest in required:
                    found.setdefault(digest, directory)
        if set(found) != set(required):
            raise ValueError("missing merged delta shards; supply --shard-dirs matching shard_hashes")
        roots = [found[digest] for digest in required]
        for directory in roots:
            hashes = _complete(directory)
            _check_saved(directory / "result.json", directory, hashes)
            shard = _read(directory / "result.json")
            if shard.get("stage") != "collect" or shard.get("shared") != result.get("shared"):
                raise ValueError("merged shard shared identity mismatch")
    found = {}
    expected = {u["source_id"]: u for u in updates}
    for directory in roots:
        for source_dir in sorted((directory / "sources").glob("*")):
            if not source_dir.is_dir() or source_dir.name.startswith("."):
                continue
            hashes = _complete(source_dir)
            summary_path = source_dir / "summary.json"
            _check_saved(summary_path, source_dir, hashes)
            summary = _read(summary_path)
            sid = summary["source_id"]
            if sid not in expected or sid in found or summary != expected[sid]:
                raise ValueError(f"duplicate/unexpected/mismatched source summary: {sid}")
            paths = list(source_dir.glob("delta_*.npz"))
            if len(paths) != 1:
                raise ValueError(f"expected exactly one saved source delta: {sid}")
            path = paths[0]
            manifest_path = Path(str(path) + ".manifest.json")
            _check_saved(manifest_path, source_dir, hashes)
            manifest = _read(manifest_path)
            if (manifest["payload_hash"] != summary["delta_payload_hash"]
                    or hashes.get(path.name) != manifest["payload_hash"]):
                raise ValueError(f"source delta payload identity mismatch: {sid}")
            found[sid] = (path, manifest)
    if set(found) != set(ids):
        raise ValueError("missing saved source deltas")
    return ids, [found[sid] for sid in ids], roots


def _layout(manifest):
    layout = manifest["layout"]
    if not layout or deltas.canonical_hash(layout) != manifest["layout_hash"]:
        raise ValueError("delta layout hash mismatch")
    names, offset = set(), 0
    for entry in layout:
        if (entry["name"] in names or entry["offset"] != offset
                or entry["numel"] <= 0 or math.prod(entry["shape"]) != entry["numel"]):
            raise ValueError("invalid delta layout order/offset/shape")
        names.add(entry["name"])
        offset += entry["numel"]
    return layout


def _metadata(ids, sources_path, folds_path):
    manifest, split = _read(sources_path), _read(folds_path)
    units = manifest.get("units", manifest.get("sources", []))
    if not isinstance(units, list) or len({s["source_id"] for s in units}) != len(units):
        raise ValueError("sources manifest requires unique units/sources")
    by_id = {s["source_id"]: s for s in units}
    if any(sid not in by_id for sid in ids):
        raise ValueError("collected source missing from sources manifest")
    assignment = split["assignment"]
    labels, groups, factors, families, raw_factors, roles = [], [], [], [], [], []
    parent_folds = {}
    for sid in ids:
        source = by_id[sid]
        group = source.get("group") or source.get("parent_task_id")
        factor = source.get("source_factor") or source.get("factor") or source.get("family")
        if not isinstance(group, str) or not group or not isinstance(factor, str) or not factor:
            raise ValueError(f"source needs group and source_factor/factor/family: {sid}")
        if sid in split.get("groups", {}) and split["groups"][sid] != group:
            raise ValueError(f"fold/source group mismatch: {sid}")
        label = assignment.get(sid)
        if label is None:
            if sid not in split.get("controls", []) and source.get("role") not in {"noop", "no_op"}:
                raise ValueError(f"missing fixed fold assignment: {sid}")
            label = -1
        elif isinstance(label, bool) or not isinstance(label, int) or label < 0:
            raise ValueError("fold assignments must be nonnegative integers")
        if "fold" in source and source["fold"] != label:
            raise ValueError(f"source/folds assignment mismatch: {sid}")
        for parent in {group, source.get("parent_task_id") or group}:
            if parent in parent_folds and parent_folds[parent] != label:
                raise ValueError("parent-task group leakage between folds")
            parent_folds[parent] = label
        labels.append(label)
        groups.append(group)
        factors.append(factor)
        families.append(source.get("family") or "")
        raw_factors.append(source.get("factor") or "")
        roles.append(source.get("role") or "")
    observed = sorted(set(labels) - {-1})
    if len(observed) < 2:
        raise ValueError("at least two nonempty fixed source folds are required")
    count = split.get("n_folds", max(observed) + 1)
    if isinstance(count, bool) or not isinstance(count, int) or count <= max(observed):
        raise ValueError("n_folds does not cover source fold assignments")
    metadata = dict(source_ids=np.asarray(ids), groups=np.asarray(groups), source_groups=np.asarray(groups),
                    folds=np.asarray(labels), source_factor=np.asarray(factors), family=np.asarray(families),
                    factor=np.asarray(raw_factors), roles=np.asarray(roles))
    return metadata, list(range(count))


def _fisher(path, layout, eps, eps_mode, block_size):
    if not np.isfinite(eps) or eps <= 0 or eps_mode not in {"absolute", "relative"}:
        raise ValueError("eps must be positive and finite; eps-mode is absolute or relative")
    with np.load(path, allow_pickle=False) as archive:
        fisher = archive["fisher"]
        saved_hash = str(archive["layout_hash"].item())
        version = str(archive["version"].item()) if "version" in archive else "fisher-npz-v1"
        saved_eps = float(archive["eps"].item()) if "eps" in archive else None
    if saved_hash != deltas.canonical_hash(layout):
        raise ValueError("Fisher layout hash mismatch")
    if fisher.shape != (sum(e["numel"] for e in layout),):
        raise ValueError("Fisher vector length differs from delta layout")
    if not np.isfinite(fisher).all() or np.any(fisher < 0):
        raise ValueError("Fisher must be finite and nonnegative")
    median = float(np.median(fisher))
    effective_eps = eps if eps_mode == "absolute" else eps * median
    if not np.isfinite(effective_eps) or effective_eps <= 0:
        raise ValueError("relative damping requires a positive Fisher median; use --eps-mode absolute")
    H = whiten.from_fisher(fisher, layout, eps=effective_eps, layout_hash=saved_hash,
                           version=version, block_size=block_size)
    stats = dict(path=str(path), content_hash=deltas.file_hash(path), version=version,
                 count=fisher.size, min=float(fisher.min()), max=float(fisher.max()),
                 mean=float(np.mean(fisher, dtype=np.float64)), median=median,
                 zero_count=int(np.count_nonzero(fisher == 0)), normalization="none",
                 eps_requested=eps, eps_mode=eps_mode, eps_effective=effective_eps,
                 archive_eps=saved_eps, archive_eps_applied=False, fisher_hash=H.fisher_hash,
                 H_min=float(H.diagonal.min()), H_max=float(H.diagonal.max()))
    return H, stats


def encode_fold(vectors, H, train_indices, *, rtol=1e-10):
    """Fit only training vectors, encode every source, and return H residuals."""
    basis = whiten.build_basis([vectors[i] for i in train_indices], H, rtol=rtol)
    X = np.empty((len(vectors), basis.rank), dtype=np.float64)
    residual = np.empty(len(vectors), dtype=np.float64)
    for i, vector in enumerate(vectors):
        X[i], residual[i] = basis.encode(vector)
    return basis, X, residual


def build_codes(run_dir, fisher_path, sources_path, folds_path, output_dir=None, *,
                eps=1e-3, eps_mode="absolute", rtol=1e-10, block_size=deltas.BLOCK_SIZE,
                shard_dirs=(), dry_run=False, pilot_smoke=False):
    """Validate inputs and atomically publish a complete, immutable codes directory."""
    if block_size <= 0 or not np.isfinite(rtol) or rtol < 0:
        raise ValueError("block_size must be positive and rtol finite/nonnegative")
    run, fisher_path, sources_path, folds_path = map(Path, (run_dir, fisher_path, sources_path, folds_path))
    run, fisher_path, sources_path, folds_path = [p.resolve() for p in (run, fisher_path, sources_path, folds_path)]
    out = Path(output_dir).resolve() if output_dir else run / "codes"
    hashes = _complete(run)
    for name in ("result.json", "responses.jsonl", "responses.jsonl.manifest.json"):
        _check_saved(run / name, run, hashes)
    result = _read(run / "result.json")
    if result.get("status") != "complete" or result.get("stage") not in {"merge", "collect", "pilot"}:
        raise ValueError("codes requires a completed merged collect run (or collect/pilot run)")
    ids, saved, roots = _delta_paths(run, result, shard_dirs)
    metadata, folds = _metadata(ids, sources_path, folds_path)
    layout, base_hash = _layout(saved[0][1]), saved[0][1]["base_hash"]
    layout_hash = deltas.canonical_hash(layout)
    for _, manifest in saved:
        _layout(manifest)
        if manifest["layout_hash"] != layout_hash:
            raise ValueError("source delta layout hash mismatch")
        if manifest["base_hash"] != base_hash:
            raise ValueError("source deltas do not share the same initial parameters")
    H, fisher_stats = _fisher(fisher_path, layout, eps, eps_mode, block_size)
    probes, records, response_manifest = read_responses(run / "responses.jsonl")
    if set(r.source_id for r in records) != set(ids):
        raise ValueError("responses and saved deltas have different source coverage")
    factors = dict(zip(ids, metadata["source_factor"]))
    if any(r.source_factor is not None and r.source_factor != factors[r.source_id] for r in records):
        raise ValueError("source_factor differs between source manifest and responses")
    if pilot_smoke and (result["stage"] != "pilot" or len(ids) != 6):
        raise ValueError("pilot-smoke requires a completed pilot with six source deltas")
    summary = dict(schema_version="behavior-codes-v1", stage="codes", status="dry-run" if dry_run else "complete",
                   run_dir=str(run), output_dir=str(out), source_ids=ids, sources=len(ids), probes=len(probes),
                   layout_hash=layout_hash, parameter_dimension=H.size, base_hash=base_hash,
                   fisher=fisher_stats, basis_rtol=rtol, basis_atol=0.0,
                   residual_fraction_definition="||delta - C x||_H / ||delta||_H; zero delta -> 0",
                   control_policy="fold -1: encoded/diagnosed, excluded from training and held-out fold sets",
                   fit_contract="Each fold X is valid only for its train_indices/test_indices; do not reuse for all CV folds. codes_all is transductive.",
                   input_hashes={str(p): deltas.file_hash(p) for p in (sources_path, folds_path, run / "result.json")},
                   shard_dirs=[str(p) for p in roots], eval_seeds=response_manifest["provenance"].get("eval_seeds"),
                   number_of_models=0, max_rollouts=0, new_teacher_tokens=0, gpu_hours=0,
                   environment_seconds=0, cpu_only=True,
                   temporary_flat_bytes=8 * H.size * len(ids), metric_bytes=8 * H.size,
                   block_workspace_order="O(block_size * sources + sources^2)", folds=[])
    labels = metadata["folds"]
    for fold in folds:
        train = np.flatnonzero((labels >= 0) & (labels != fold))
        test = np.flatnonzero(labels == fold)
        summary["folds"].append(dict(fold=fold, train_indices=train.tolist(), test_indices=test.tolist(),
                                     train_source_ids=[ids[i] for i in train], heldout_source_ids=[ids[i] for i in test],
                                     codes=f"fold_{fold}/codes.npz", basis_dimension_upper_bound=len(train)))
    if dry_run:
        summary.update(delta_payloads_verified=False, pilot_smoke="planned" if pilot_smoke else None)
        return summary
    if out.exists():
        raise FileExistsError(f"codes output already exists: {out}; choose a new --output-dir")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".codes-", dir=out.parent) as temporary:
        work = Path(temporary)
        artifacts = work / "artifacts"
        artifacts.mkdir()
        vectors, references = [], []
        norms_h, norms_euclidean = [], []
        for i, (path, manifest) in enumerate(saved):
            delta = deltas.load_delta(path)
            vector = np.lib.format.open_memmap(work / f"delta_{i}.npy", mode="w+", dtype=np.float64, shape=(H.size,))
            delta.flat(out=vector)
            vector.flush()
            vector.flags.writeable = False
            vectors.append(vector)
            norms_h.append(H.norm(vector))
            norm2 = 0.0
            for sl in H.blocks():
                norm2 += float(vector[sl] @ vector[sl])
            norms_euclidean.append(float(np.sqrt(norm2)))
            references.append(dict(source_id=ids[i], path=str(path), content_hash=manifest["payload_hash"],
                                   manifest_hash=deltas.file_hash(Path(str(path) + ".manifest.json"))))
            del delta
        norms_h = np.asarray(norms_h)
        zero = norms_h == 0
        if pilot_smoke and np.count_nonzero(zero) != 2:
            raise ValueError("pilot-smoke requires exactly two zero deltas")
        summary["deltas"] = [dict(**ref, norm_h=float(norms_h[i]), norm_euclidean=norms_euclidean[i],
                                  zero=bool(zero[i]), flat_dtype="float64", saved_tensor_dtype="float32")
                             for i, ref in enumerate(references)]
        variants = summary["folds"] + [dict(fold=None, train_indices=list(range(len(ids))), test_indices=[],
                                           codes="codes_all.npz", transductive=True)]
        for variant in variants:
            train = np.asarray(variant["train_indices"], dtype=int)
            test = np.asarray(variant["test_indices"], dtype=int)
            basis, X, residual = encode_fold(vectors, H, train, rtol=rtol)
            fractions = np.divide(residual, norms_h, out=np.zeros_like(residual), where=~zero)
            if np.any(X[zero] != 0) or np.any(residual[zero] != 0):
                raise ValueError("zero delta produced nonzero codes/residual")
            transductive = variant.get("transductive", False)
            path = artifacts / variant["codes"]
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **metadata, X=X, layout_hash=layout_hash, eps=H.eps,
                                fisher_hash=H.fisher_hash, basis_scope="transductive" if transductive else "training_fold",
                                transductive=transductive, heldout_fold=-1 if transductive else variant["fold"],
                                train_indices=train, test_indices=test,
                                delta_norm_h=norms_h, delta_norm_euclidean=norms_euclidean,
                                residual_norm_h=residual, residual_fraction=fractions,
                                decoder_coefficients=basis.coefficients, eigenvalues=basis.eigenvalues,
                                decoder_source_ids=metadata["source_ids"][train],
                                decoder_paths=np.asarray([references[i]["path"] for i in train], dtype=str),
                                decoder_content_hashes=np.asarray([references[i]["content_hash"] for i in train], dtype=str))
            basis.save(path.with_suffix(".decoder.json"), [references[i] for i in train])
            variant.update(basis_dimension=basis.rank, transductive=transductive,
                           residual_fractions={sid: float(fractions[i]) for i, sid in enumerate(ids)},
                           heldout_residual_fractions={ids[i]: float(fractions[i]) for i in test},
                           heldout_residual_norm_h={ids[i]: float(residual[i]) for i in test})
        summary["transductive"] = variants[-1]
        summary.update(delta_payloads_verified=True,
                       pilot_smoke=dict(passed=True, zero_source_ids=[ids[i] for i in np.flatnonzero(zero)]) if pilot_smoke else None)
        for name in ("responses.jsonl", "responses.jsonl.manifest.json"):
            (artifacts / name).write_bytes((run / name).read_bytes())
        _write(artifacts / "summary.json", summary)
        artifacts.rename(out)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", "--collect-dir", type=Path, help="merged collect run; required unless --pilot-smoke")
    parser.add_argument("--fisher", type=Path, default=DATA / "fisher_v3t_student_full.npz")
    parser.add_argument("--sources", "--manifest", type=Path, default=DATA / "sources.json")
    parser.add_argument("--folds", type=Path, default=DATA / "folds.json")
    parser.add_argument("--eps", type=float, default=1e-3, help="positive damping (default 1e-3, absolute)")
    parser.add_argument("--eps-mode", choices=("absolute", "relative"), default="absolute")
    parser.add_argument("--rtol", type=float, default=1e-10, help="training Gram eigenvalue cutoff relative to largest")
    parser.add_argument("--block-size", type=int, default=deltas.BLOCK_SIZE)
    parser.add_argument("--shard-dirs", nargs="+", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, help="new immutable directory (default <run>/codes)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pilot-smoke", action="store_true", help="default to pilot_r1; assert six deltas and two exact zeros")
    args = parser.parse_args(argv)
    try:
        if args.run_dir is None and not args.pilot_smoke:
            raise ValueError("--run-dir is required unless --pilot-smoke")
        summary = build_codes(args.run_dir or PILOT, args.fisher, args.sources, args.folds, args.output_dir,
                              eps=args.eps, eps_mode=args.eps_mode, rtol=args.rtol, block_size=args.block_size,
                              shard_dirs=args.shard_dirs, dry_run=args.dry_run, pilot_smoke=args.pilot_smoke)
        print(json.dumps(summary, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print(f"codes: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
