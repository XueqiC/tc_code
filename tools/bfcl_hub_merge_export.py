#!/usr/bin/env python3
"""Overlay a flattened training checkpoint onto its faithful Hub checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


class ExportError(RuntimeError):
    """Raised when a safe, faithful export cannot be produced."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay a trained causal-LM checkpoint onto the original local Hub "
            "snapshot while retaining the Hub architecture and shard layout."
        )
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        required=True,
        help="Directory containing the trained model.safetensors",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Empty output directory for the merged Hub-faithful checkpoint",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-4B",
        help="Hub model ID whose refs/main snapshot supplies the base checkpoint",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Reload the original and merged shards and verify every tensor",
    )
    return parser.parse_args()


def _snapshot_for_model(model: str) -> Path:
    local = Path(model)
    if local.is_dir() and (local / "config.json").is_file() and (
        (local / "model.safetensors.index.json").is_file()
        or (local / "model.safetensors").is_file()
    ):
        # a previously exported hub_merged directory (used to stack a round-N adapter on a round-(N-1) model)
        return local
    parts = model.strip().split("/")
    if len(parts) < 2 or any(not part or part in {".", ".."} for part in parts):
        raise ExportError(f"invalid Hub model ID: {model!r}")

    # honor HF_HOME / HF_HUB_CACHE: on hpg the populated cache lives on /blue,
    # and the stale copy under ~/.cache is an old revision without the
    # preprocessor configs vllm 0.27 requires -- resolving the home cache there
    # produced merges that could never serve
    import os as _os

    if _os.environ.get("HF_HUB_CACHE"):
        hub_root = Path(_os.environ["HF_HUB_CACHE"])
    elif _os.environ.get("HF_HOME"):
        hub_root = Path(_os.environ["HF_HOME"]) / "hub"
    else:
        hub_root = Path.home() / ".cache" / "huggingface" / "hub"
    cache_dir = hub_root / ("models--" + "--".join(parts))
    main_ref = cache_dir / "refs" / "main"
    if not main_ref.is_file():
        raise ExportError(f"missing Hub ref for {model!r}: {main_ref}")

    try:
        revision = main_ref.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ExportError(f"could not read Hub ref {main_ref}: {exc}") from exc
    if not revision or Path(revision).name != revision:
        raise ExportError(f"invalid revision in {main_ref}: {revision!r}")

    snapshot = cache_dir / "snapshots" / revision
    if not snapshot.is_dir():
        raise ExportError(
            f"refs/main for {model!r} points to missing snapshot: {snapshot}"
        )
    if not (snapshot / "config.json").is_file():
        raise ExportError(f"snapshot is missing config.json: {snapshot}")
    return snapshot


def _read_index(snapshot: Path) -> tuple[Path | None, dict[str, str], list[str]]:
    """Read the shard map, or derive it from a single-file checkpoint header."""

    index_path = snapshot / "model.safetensors.index.json"
    if not index_path.is_file():
        checkpoint = snapshot / "model.safetensors"
        if not checkpoint.is_file():
            raise ExportError(
                f"snapshot is missing checkpoint: expected {index_path} or {checkpoint}"
            )
        try:
            # Inspect only the header; the shared shard loader below loads and
            # validates the weights, including for large single-file snapshots.
            with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
                weight_map = {key: checkpoint.name for key in handle.keys()}
        except Exception as exc:
            raise ExportError(f"failed to read checkpoint header {checkpoint}: {exc}") from exc
        if any(not key for key in weight_map):
            raise ExportError(f"checkpoint contains an invalid tensor key: {checkpoint}")
        return None, weight_map, [checkpoint.name]
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportError(f"could not read shard index {index_path}: {exc}") from exc

    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ExportError(f"shard index has no non-empty weight_map: {index_path}")
    if any(not isinstance(key, str) or not key for key in weight_map):
        raise ExportError(f"shard index contains an invalid tensor key: {index_path}")
    if any(not isinstance(value, str) or not value for value in weight_map.values()):
        raise ExportError(f"shard index contains an invalid shard name: {index_path}")

    shard_names = sorted(set(weight_map.values()))
    for shard_name in shard_names:
        relative = Path(shard_name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.suffix != ".safetensors"
        ):
            raise ExportError(f"unsafe or invalid shard name in index: {shard_name!r}")
        shard_path = snapshot / relative
        if not shard_path.is_file():
            raise ExportError(f"missing shard named by index: {shard_path}")
    return index_path, weight_map, shard_names


def _load_safetensors(path: Path, description: str) -> dict[str, torch.Tensor]:
    try:
        state = load_file(str(path), device="cpu")
    except Exception as exc:
        raise ExportError(f"failed to load {description} {path}: {exc}") from exc
    if not state:
        raise ExportError(f"{description} contains no tensors: {path}")
    return state


def _load_hub_state(
    snapshot: Path, weight_map: dict[str, str], shard_names: list[str]
) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for shard_name in shard_names:
        shard_path = snapshot / shard_name
        shard_state = _load_safetensors(shard_path, "Hub shard")
        for key, tensor in shard_state.items():
            if key in state:
                raise ExportError(
                    f"duplicate Hub tensor {key!r} while loading {shard_path}"
                )
            indexed_shard = weight_map.get(key)
            if indexed_shard is None:
                raise ExportError(
                    f"Hub tensor {key!r} in {shard_path} is absent from the index"
                )
            if indexed_shard != shard_name:
                raise ExportError(
                    f"Hub tensor {key!r} is in {shard_name!r}, but the index names "
                    f"{indexed_shard!r}"
                )
            state[key] = tensor

    missing = set(weight_map).difference(state)
    if missing:
        sample = ", ".join(repr(key) for key in sorted(missing)[:5])
        raise ExportError(
            f"{len(missing)} indexed Hub tensors were not found in their shards; "
            f"first missing: {sample}"
        )
    return state


def _common_prefix_length(left: list[str], right: list[str]) -> int:
    count = 0
    for left_part, right_part in zip(left, right):
        if left_part != right_part:
            break
        count += 1
    return count


def _common_suffix_length(left: list[str], right: list[str]) -> int:
    count = 0
    for left_part, right_part in zip(reversed(left), reversed(right)):
        if left_part != right_part:
            break
        count += 1
    return count


def _shape(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(tensor.shape)


def _bit_identical(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.dtype != right.dtype or _shape(left) != _shape(right):
        return False
    return torch.equal(
        left.contiguous().reshape(-1).view(torch.uint8),
        right.contiguous().reshape(-1).view(torch.uint8),
    )


def _format_candidates(
    candidates: list[str], hub_state: dict[str, torch.Tensor]
) -> str:
    rendered = [f"{key} shape={_shape(hub_state[key])}" for key in candidates[:6]]
    if len(candidates) > 6:
        rendered.append(f"... and {len(candidates) - 6} more")
    return "; ".join(rendered)


def _find_hub_target(
    trained_key: str,
    trained_tensor: torch.Tensor,
    hub_state: dict[str, torch.Tensor],
) -> str:
    """Find the unique best Hub key using component suffixes, then shape."""

    trained_parts = trained_key.split(".")
    scored: list[tuple[int, int, str]] = []
    for hub_key in hub_state:
        hub_parts = hub_key.split(".")
        suffix_length = _common_suffix_length(trained_parts, hub_parts)
        if suffix_length:
            scored.append(
                (
                    suffix_length,
                    _common_prefix_length(trained_parts, hub_parts),
                    hub_key,
                )
            )
    if not scored:
        raise ExportError(
            f"unmapped trained tensor {trained_key!r}: no Hub key has a matching suffix"
        )

    best_suffix = max(item[0] for item in scored)
    suffix_matches = [item for item in scored if item[0] == best_suffix]
    best_prefix = max(item[1] for item in suffix_matches)
    name_matches = [item[2] for item in suffix_matches if item[1] == best_prefix]
    shape_matches = [
        key for key in name_matches if _shape(hub_state[key]) == _shape(trained_tensor)
    ]

    if not shape_matches:
        raise ExportError(
            f"shape mismatch for trained tensor {trained_key!r} "
            f"shape={_shape(trained_tensor)}; best Hub suffix match(es): "
            f"{_format_candidates(name_matches, hub_state)}"
        )
    if len(shape_matches) != 1:
        raise ExportError(
            f"ambiguous mapping for trained tensor {trained_key!r} "
            f"shape={_shape(trained_tensor)}; matching Hub targets: "
            f"{_format_candidates(shape_matches, hub_state)}"
        )
    return shape_matches[0]


def _overlay_adapter(
    hub_state: dict[str, torch.Tensor], adapter_state: dict[str, torch.Tensor]
) -> dict[str, str]:
    """Overlay adapter tensors and return Hub-target -> trained-key mappings."""

    target_to_trained: dict[str, str] = {}
    for trained_key, trained_tensor in adapter_state.items():
        target = _find_hub_target(trained_key, trained_tensor, hub_state)
        previous_key = target_to_trained.get(target)
        if previous_key is not None:
            if not _bit_identical(adapter_state[previous_key], trained_tensor):
                raise ExportError(
                    f"trained tensors {previous_key!r} and {trained_key!r} both map "
                    f"to tied Hub target {target!r}, but their values differ"
                )
        else:
            target_to_trained[target] = trained_key
        hub_state[target] = trained_tensor
    return target_to_trained


def _prepare_output(out: Path, snapshot: Path) -> None:
    out_resolved = out.resolve()
    snapshot_resolved = snapshot.resolve()
    if out_resolved == snapshot_resolved or snapshot_resolved in out_resolved.parents:
        raise ExportError(f"output directory must not be inside the Hub snapshot: {out}")
    if out.exists() and not out.is_dir():
        raise ExportError(f"output path exists and is not a directory: {out}")
    if out.is_dir() and next(out.iterdir(), None) is not None:
        raise ExportError(f"output directory must be empty: {out}")
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ExportError(f"could not create output directory {out}: {exc}") from exc


def _copy_snapshot_files(snapshot: Path, out: Path, shard_names: list[str]) -> None:
    """Copy every snapshot file except weight shards, dereferencing Hub symlinks."""

    excluded = set(shard_names)
    for source in snapshot.rglob("*"):
        relative = source.relative_to(snapshot)
        destination = out / relative
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        if relative.as_posix() in excluded:
            continue
        if not source.is_file():
            raise ExportError(f"unsupported non-file in Hub snapshot: {source}")
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
        except OSError as exc:
            raise ExportError(f"failed to copy {source} to {destination}: {exc}") from exc


def _write_shards(
    out: Path,
    hub_state: dict[str, torch.Tensor],
    weight_map: dict[str, str],
    shard_names: list[str],
) -> None:
    for shard_name in shard_names:
        shard_state = {
            key: hub_state[key]
            for key, indexed_shard in weight_map.items()
            if indexed_shard == shard_name
        }
        if not shard_state:
            raise ExportError(f"index names an empty shard: {shard_name}")
        destination = out / shard_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            save_file(shard_state, str(destination))
        except Exception as exc:
            raise ExportError(f"failed to write merged shard {destination}: {exc}") from exc


def _verify_shards(
    snapshot: Path,
    out: Path,
    weight_map: dict[str, str],
    shard_names: list[str],
    target_to_trained: dict[str, str],
    adapter_state: dict[str, torch.Tensor],
) -> None:
    """Reload original and merged state dictionaries one shard at a time."""

    for shard_name in shard_names:
        original = _load_safetensors(snapshot / shard_name, "original Hub shard")
        merged = _load_safetensors(out / shard_name, "merged shard")
        expected_keys = {
            key for key, indexed_shard in weight_map.items() if indexed_shard == shard_name
        }
        if set(original) != expected_keys:
            raise ExportError(
                f"verification found original shard/index key mismatch: {shard_name}"
            )
        if set(merged) != expected_keys:
            raise ExportError(
                f"verification found merged shard/index key mismatch: {shard_name}"
            )

        for key in expected_keys:
            trained_key = target_to_trained.get(key)
            if trained_key is None:
                if not _bit_identical(original[key], merged[key]):
                    raise ExportError(
                        f"verification failed: untouched tensor {key!r} changed"
                    )
            elif not _bit_identical(merged[key], adapter_state[trained_key]):
                raise ExportError(
                    f"verification failed: overlaid tensor {key!r} does not equal "
                    f"trained tensor {trained_key!r}"
                )


def main() -> None:
    args = _parse_args()
    adapter_dir = args.adapter.expanduser()
    out = args.out.expanduser()

    try:
        snapshot = _snapshot_for_model(args.model)
        _, weight_map, shard_names = _read_index(snapshot)

        adapter_path = adapter_dir / "model.safetensors"
        if not adapter_dir.is_dir():
            raise ExportError(f"adapter directory does not exist: {adapter_dir}")
        if not adapter_path.is_file():
            raise ExportError(f"adapter checkpoint is missing: {adapter_path}")

        hub_state = _load_hub_state(snapshot, weight_map, shard_names)
        adapter_state = _load_safetensors(adapter_path, "trained adapter checkpoint")

        target_to_trained = _overlay_adapter(hub_state, adapter_state)
        overlaid = len(target_to_trained)
        untouched = len(hub_state) - overlaid

        _prepare_output(out, snapshot)
        _copy_snapshot_files(snapshot, out, shard_names)
        # vllm >=0.27 refuses this architecture without its preprocessor
        # configs; an older snapshot revision may predate them, so pull the
        # missing ones from any sibling revision rather than failing at serve
        for name in ("preprocessor_config.json", "video_preprocessor_config.json"):
            if not (out / name).exists():
                for sibling in snapshot.parent.glob(f"*/{name}"):
                    (out / name).write_bytes(sibling.read_bytes())
                    break
        _write_shards(out, hub_state, weight_map, shard_names)

        if args.verify:
            del hub_state
            _verify_shards(
                snapshot,
                out,
                weight_map,
                shard_names,
                target_to_trained,
                adapter_state,
            )

    except ExportError as exc:
        raise SystemExit(f"error: {exc}") from exc

    verification = "; verification passed" if args.verify else ""
    print(
        f"Merged {args.model}: overlaid {overlaid} tensors, left {untouched} "
        f"untouched, wrote {len(shard_names)} shards to {out}{verification}."
    )


if __name__ == "__main__":
    main()
