"""Training-only seed streams and cross-seed data identity checks."""
import json
from pathlib import Path
import random

from ..persistence import file_hash


def training_seed(config):
    seed = config.get("training_seed", 0)
    if type(seed) is not int or seed < 0:
        raise ValueError("training seed must be a non-negative integer")
    return seed


def seed_training(seed):
    """Seed global initialization/dropout/scoring RNGs only in the train worker.

    Python accepts arbitrary integers; NumPy and Torch use 32/64-bit seeds.
    The original CLI integer remains the run identity in every receipt.
    """
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed % (2**64))  # Includes CUDA generators.


def seeded_directory(directory, seed):
    directory = Path(directory)
    return directory.with_name(f"{directory.name}_s{seed}") if seed else directory


# Include the accountant's unavailable attempts and order, not just usable rows.
PURCHASE_FIELDS = (
    "method", "benchmark", "student", "teacher", "smoke", "B", "bank",
    "certificate_sha256", "certificate_core", "purchase_seed", "purchase_order",
    "purchase_order_hash", "purchase_rule", "usable_cost_basis",
    "purchased_package_ids", "charges", "teacher_tokens_charged", "remaining_tokens",
    "blocked_next", "cost_scope", "positive_rows", "purchased_usable_packages",
    "rendering", "rows_hash", "evaluation_protocol",
)


def verify_seed_zero(directory, manifest, *, selection=False):
    """Compare only siblings in this results root; never write to a reference.

    Preparation checks purchases. Training/evaluation additionally require the
    method's selection artifacts. Exposure schedules deliberately vary by seed.
    A renamed seed-zero run in the same root is also a valid reference.
    """
    seed = manifest.get("seed", 0)
    if not seed:
        return
    directory = Path(directory)
    suffix = f"_s{seed}"
    if not directory.name.endswith(suffix):
        raise ValueError(f"seed {seed} run directory must end with {suffix}")
    zero = directory.with_name(directory.name[:-len(suffix)])
    references = []
    for candidate in sorted(directory.parent.iterdir()):
        if candidate.is_symlink() or not candidate.is_dir():
            if candidate == zero:
                raise ValueError(f"seed-zero guard: reference must be a local directory: {candidate}")
            continue
        if candidate == directory:
            continue
        path = candidate/"manifest.json"
        if not path.is_file():
            if candidate == zero:
                raise ValueError(f"seed-zero guard: missing {path}")
            continue
        try:
            reference = json.loads(path.read_text())
            if not isinstance(reference, dict):
                raise ValueError("manifest must be an object")
        except (ValueError, OSError) as error:
            if candidate == zero:
                raise ValueError(f"seed-zero guard: cannot read {path}") from error
            continue
        same_cell = all(reference.get(k) == manifest.get(k)
                        for k in ("method", "benchmark", "B", "bank", "smoke"))
        if candidate == zero or (reference.get("seed") == 0 and same_cell):
            if reference.get("seed") != 0:
                raise ValueError(f"seed-zero guard: reference is not seed zero: {candidate}")
            references.append((candidate, reference))
    artifacts = ["purchased_rows.json"]
    if selection:
        artifacts.append("training_rows.json")
        if manifest["method"] == "smartad":
            artifacts.append("smartad_selection.json")
    receipts = []
    for candidate, reference in references:
        for key in PURCHASE_FIELDS:
            if key not in manifest or key not in reference or manifest[key] != reference[key]:
                raise ValueError(f"seed-zero guard: {key} differs from {candidate}")
        hashes = {}
        for name in artifacts:
            current, expected = directory/name, candidate/name
            if not current.is_file() or not expected.is_file():
                raise ValueError(f"seed-zero guard: missing {name} in {directory} or {candidate}")
            hashes[name] = file_hash(current)
            if hashes[name] != file_hash(expected):
                raise ValueError(f"seed-zero guard: {name} differs from {candidate}")
        receipts.append(dict(run_dir=str(candidate), artifact_sha256=hashes))
    manifest["seed_zero_verification"] = dict(
        status=("verified" if selection else "purchases_verified_selection_pending")
               if receipts else "no_seed_zero_run",
        references=receipts)
