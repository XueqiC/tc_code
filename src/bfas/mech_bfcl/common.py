"""Small, CPU-only persistence and provenance helpers."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HARNESS = ROOT / "envs/bfcl/gorilla/berkeley-function-call-leaderboard"
STUDENT = "google/gemma-4-12B-it"
REGISTRY = STUDENT + "-FC"
TEACHER = "openai/gpt-5.6-luna"
BUDGETS = {"shared": 8000, "C": 24000, "D": 24000}


def resolved_train_seed(args, split_seed):
    seed = getattr(args, "train_seed", None)
    return split_seed if seed is None else seed


def variant_name(arm, train_seed, split_seed):
    return arm if arm == "base" or train_seed == split_seed else f"{arm}-s{train_seed}"


def training_directory(directory, arm, train_seed, split_seed):
    name = "training" if train_seed == split_seed else f"training_s{train_seed}"
    return Path(directory) / arm / name


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    path = Path(path)
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def append_row(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as out:
        out.write(json.dumps(value, ensure_ascii=False) + "\n")
        out.flush()
        os.fsync(out.fileno())


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def setup_harness(runtime=None):
    import sys
    os.environ.setdefault("BFCL_PROJECT_ROOT", str(runtime or ROOT / "results/mech_bfcl/runtime"))
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))


def bind_run(directory, splits):
    path = Path(directory) / "protocol.json"
    generation_path = Path(directory) / "generation_protocol.json"
    budgets = dict(BUDGETS)
    if generation_path.exists():
        cap = read_json(generation_path)["max_output_tokens"]
        budgets.update(C=cap, D=cap)
    value = dict(version=1, student=STUDENT, teacher=TEACHER, service_tier="flex",
                 seed=splits["seed"], split_hash=digest(splits), budgets=budgets,
                 temperature=0.001, top_k=1, thinking=False)
    if path.exists() and read_json(path) != value:
        raise ValueError("Run directory belongs to a different protocol/split")
    write_json(path, value)
    return value
