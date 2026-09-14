"""Canonical identities for new baseline artifacts; historical files stay sealed."""
from pathlib import Path

SAD = "sad"
KANG_PREFIX = "kang_first_thought_prefix"
KANG_SUMMARY = "kang_action_list_summary"
KANG_METHODS = {KANG_PREFIX, KANG_SUMMARY}


def method_name(method, config=None):
    if method == "kang":
        method = (config or {}).get("kang_mode", KANG_PREFIX)
    if method not in {SAD, "smartad", "gad", *KANG_METHODS}:
        raise ValueError("unknown baseline method: " + method)
    return method


def fidelity_metadata(method):
    method = method_name(method)
    if method in {SAD, "smartad"}:
        return dict(implementation_basis="paper_reimplementation")
    if method == KANG_SUMMARY:
        return dict(implementation_basis="local_adaptation", fidelity="deviating",
            limitation="NOT the published mechanism: retrospective action-list summary rewrites the first training target.")
    if method == KANG_PREFIX:
        return dict(implementation_basis="official_repository", fidelity="first-thought acquisition mechanism",
            source_repository="https://github.com/Nardien/agent-distillation",
            limitation="Benchmark, teacher/student, training schedule and existing SAG differences still apply.")
    return dict(implementation_basis="paper_text_adaptation")


def require_unsealed_output(directory):
    directory = Path(directory).resolve()
    if "archive" in directory.parts:
        raise ValueError("archive is sealed; choose a fresh run outside archive")
    sealed = Path(__file__).resolve().parents[4]/"archive/table1"
    if (sealed/directory.name).is_dir():
        raise ValueError("archived cell name is sealed; choose a new fidelity run name")


def record_method(manifest):
    # A historical bare Kang label describes the retrospective implementation.
    # Reading it does not upgrade old trajectories into prefixed acquisitions.
    config = manifest.setdefault("config", {})
    method = manifest["method"]
    if method == "kang" and "kang_mode" not in config:
        method = KANG_SUMMARY
    method = method_name(method, config)
    manifest.update(method=method, fidelity=fidelity_metadata(method))
    config["method"] = method
    hp = manifest.get("hyperparameters", {})
    if hp:
        hp["method"] = method
    return manifest
